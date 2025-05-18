import cupy as cp
from src import Particles
from src import Consts



compute_speed_energy_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_speeds_and_energies(
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    double* __restrict__ speeds,
    double* __restrict__ energies,
    const int* __restrict__ type,
    const double* me,
    const int N
) {                                     
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    double vxi = vx[i];
    double vyi = vy[i];
    double vzi = vz[i]; 
    double speed = sqrtf(vxi * vxi + vyi * vyi + vzi * vzi);
    speeds[i] = speed;
    energies[i] = 0.5 * me[type[i]] * speed * speed;

}
''', 'compute_speeds_and_energies')

interp_kernel = cp.RawKernel(r'''
extern "C" __global__
void interpolate_cross_sections(
    const double* __restrict__ energies,
    const double* __restrict__ xs_energy,
    const double* __restrict__ xs_values,
    const bool* __restrict__ mcc_mask,
    double* __restrict__ output,
    const int xs_len,
    const int N
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
                             
    if (!mcc_mask[i]) {
        output[i] = 0.0;
        return;
    }

    double E = energies[i];

    // Handle outside bounds (clip to first or last)
    if (E <= xs_energy[0]) {
        output[i] = xs_values[0];
        return;
    } else if (E >= xs_energy[xs_len - 1]) {
        output[i] = xs_values[xs_len - 1];
        return;
    }

    // Binary search for correct interval
    int low = 0;
    int high = xs_len - 2;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (E < xs_energy[mid]) {
            high = mid - 1;
        } else if (E >= xs_energy[mid + 1]) {
            low = mid + 1;
        } else {
            // interpolate
            double x0 = xs_energy[mid];
            double x1 = xs_energy[mid + 1];
            double y0 = xs_values[mid];
            double y1 = xs_values[mid + 1];
            double t = (E - x0) / (x1 - x0);
            output[i] = y0 + t * (y1 - y0);
            return;
        }
    }
}
''', 'interpolate_cross_sections')

det_collision_kernel = cp.RawKernel(r'''
extern "C" __global__
void collision_detector(const float P_T, 
                        const int* idxs, 
                        bool* mask,
                        int* output_idxs,
                        int* output_count,
                        const int n) {
                                    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return; 
                                    
    float rand = curand_uniform(&state);
    bool collided = (rand < P_T);
    mask[i] = collided;
    
    if (collided) {
        int pos = atomicAdd(output_count, 1);
        output_idxs[pos] = idxs[i];
    }
    
}
''', 'collision_detector')


def compute_speeds_and_energies(particles: Particles.Particles2D):
    

    N = particles.last_alive
    block_size = 256
    num_blocks = (N + block_size - 1) // block_size


    compute_speed_energy_kernel(
        (num_blocks,), (block_size,),
        (particles.V[0], particles.V[1], particles.V[2], particles.V_abs, particles.K, particles.part_type, particles.m_type, cp.int32(N))
    )

    #ind = cp.where(particles.mcc_mask[:N] == 1)[0]
    #return particles.V_abs[ind], particles.energy[ind]

def read_cross_section(filename):
    """
    Reads a cross-section data file and returns a CuPy array.

    Returns:
        energies: 1D CuPy array of energy values
        sigma_T: 1D CuPy array of corresponding cross-section values
    """
    try:
        with open(filename, 'r') as f:
            lines = [line.strip().split() for line in f if line.strip()]

        # Ensure correct parsing
        cross = cp.array([[float(line[0]), float(line[1])] for line in lines if len(line) == 2])

        return cross[:, 0], cross[:, 1]  # Return two 1D arrays (energies, cross-sections)

    except Exception as e:
        print(f"Error reading cross-section file {filename}: {e}")
        return None, None
    



def interp_table(x:cp.ndarray, x_tab:cp.ndarray, y_tab:cp.ndarray, mcc_mask:cp.ndarray, N:cp.int32):
    """Fast linear interpolation on GPU"""
    '''
    i = cp.searchsorted(x_tab, x) - 1
    i = cp.clip(i, 0, x_tab.size - 2)
    x0 = x_tab[i]
    x1 = x_tab[i + 1]
    y0 = y_tab[i]
    y1 = y_tab[i + 1]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    '''

    block_size = 256
    num_blocks = (N + block_size - 1) // block_size
    output = cp.zeros(N, dtype=x.dtype)
    interp_kernel(
        (num_blocks,), (block_size,),
        (x, x_tab, y_tab, mcc_mask, output, x_tab.size, cp.int32(N))
    )

    return output

'''
def det_collided_idxs(P_T, idxs):
    n = idxs.shape[0]
    mask = cp.zeros(n, dtype=bool)
    output_idxs = cp.zeros(n, dtype=cp.int32)  # Worst case all collide
    output_count = cp.zeros(1, dtype=cp.int32)
    
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    det_collision_kernel((blocks_per_grid,), (threads_per_block,), 
                     (P_T, idxs, mask, output_idxs, output_count, n))
    
    count = output_count.get()[0]
    return output_idxs[:count], mask
'''


def apply_elastic(particles, collided_idxs):
    """
    Apply elastic collisions to selected particles.
    
    In elastic collisions, only the direction of velocity changes while the speed remains constant.
    This is modeled by randomly selecting angles in 3D space.
    
    Parameters:
    -----------
    particles : Particles.Particles2D
        The particle container object
    collided_idxs : array
        Indices of particles that undergo elastic collisions
    """
    if collided_idxs.size == 0:
        return
        
    # Get current velocities
    V = particles.V[:, collided_idxs]
    Vx, Vy = V[0], V[1]
    speeds = cp.sqrt(Vx**2 + Vy**2)
    
    # Generate random scattering angles
    # For 2D: Just need one angle (phi) in [0, 2π]
    phi = 2 * cp.pi * cp.random.uniform(0, 1, size=collided_idxs.size)
    
    # Calculate new velocity components while preserving speed
    new_Vx = speeds * cp.cos(phi)
    new_Vy = speeds * cp.sin(phi)
    
    # Update particle velocities
    particles.V[0, collided_idxs] = new_Vx
    particles.V[1, collided_idxs] = new_Vy


def apply_ionization(particles, collided_idxs, max_particles):
    """
    Apply ionization events to selected particles.
    
    In ionization collisions:
    1. The incident electron loses energy
    2. A new electron is created
    3. The neutral atom becomes ionized (not explicitly tracked in our model)
    
    Parameters:
    -----------
    particles : Particles.Particles2D
        The particle container object
    collided_idxs : array
        Indices of particles that undergo ionization collisions
    max_particles : int
        Maximum number of particles allowed in the simulation
    """
    if collided_idxs.size == 0:
        return
        
    # Get current positions and velocities
    X = particles.X[:, collided_idxs]
    V = particles.V[:, collided_idxs]
    Vx, Vy = V[0], V[1]
    speeds = cp.sqrt(Vx**2 + Vy**2)
    
    # Calculate energies of incident electrons (eV)
    energies = 0.5 * Consts.me * speeds**2 * Consts.qe_1
    
    # Mean ionization energy for the target gas (typically ~15-20 eV for common gases)
    # This should ideally be a parameter passed to the function
    ionization_energy = 15.6  # Example: ionization energy for Argon in eV
    
    # Ensure all particles have enough energy for ionization
    valid_mask = energies > ionization_energy
    valid_idxs = collided_idxs[valid_mask]
    
    if valid_idxs.size == 0:
        return
    
    # Calculate remaining energy after ionization
    remaining_energy = energies[valid_mask] - ionization_energy
    
    # Calculate new speed for primary electrons
    # Energy is split between the primary and secondary electrons
    # Using a simple model where energy is randomly distributed
    energy_fraction = cp.random.uniform(0.2, 0.8, size=valid_idxs.size)
    primary_energy = remaining_energy * energy_fraction
    primary_speed = cp.sqrt(2 * primary_energy * Consts.qe / Consts.me)
    
    # Generate random angles for primary electrons
    phi_primary = 2 * cp.pi * cp.random.uniform(0, 1, size=valid_idxs.size)
    
    # Update velocities for primary electrons
    particles.V[0, valid_idxs] = primary_speed * cp.cos(phi_primary)
    particles.V[1, valid_idxs] = primary_speed * cp.sin(phi_primary)
    
    # Create secondary electrons if we have space
    if particles.last_alive + valid_idxs.size <= max_particles:
        # Calculate energy and speed for secondary electrons
        secondary_energy = remaining_energy * (1 - energy_fraction)
        secondary_speed = cp.sqrt(2 * secondary_energy * Consts.qe / Consts.me)
        
        # Generate random angles for secondary electrons
        phi_secondary = 2 * cp.pi * cp.random.uniform(0, 1, size=valid_idxs.size)
        
        # Prepare positions and velocities for new electrons
        new_X = X.copy()
        new_Vx = secondary_speed * cp.cos(phi_secondary)
        new_Vy = secondary_speed * cp.sin(phi_secondary)
        
        # Add secondary electrons to the particle array
        new_idxs = cp.arange(particles.last_alive, particles.last_alive + valid_idxs.size)
        particles.X[:, new_idxs] = new_X
        particles.V[0, new_idxs] = new_Vx
        particles.V[1, new_idxs] = new_Vy
        particles.active[new_idxs] = 1
        
        # Update the count of active particles
        particles.last_alive += valid_idxs.size


def null_collision_method(
    particles: Particles.Particles2D,
    density: float,
    cross_sections: tuple,  # ((E_el, sigma_el), (E_ion, sigma_ion))
    dt: float,
    max_particles: int,
    IONIZATION=True,
):
    """
    Optimized null collision method with pre-allocated arrays and minimal memory allocation.
    
    Updates particles.V_abs and particles.K arrays in-place, then uses them
    for cross-section calculations. Uses mcc_mask as a lookup array to determine eligible particles.
    """
    # Early exit if no particles eligible for collision
    #if particles.mcc_count == 0:
    #    return
    
    # --- UPDATE SPEEDS AND ENERGIES ---
    # Compute speeds and energies for all particles (updates particles.V_abs, particles.K in-place)
    compute_speeds_and_energies(particles)
    
   
    # --- COMPUTE CROSS SECTIONS for all particles ---
    # Interpolate elastic cross-sections for all particles at once
    elastic_sigma_T = interp_table(particles.K, *cross_sections[0], particles.mcc_mask, particles.last_alive)
    
    # Initialize total cross-section array (reuse elastic values)
    sigma_T = elastic_sigma_T
    

    # Add ionization cross-sections if enabled
    
    if IONIZATION:
        ion_thresh = cross_sections[1][0][0]
        # Create mask for high-energy particles above ionization threshold
        high_energy_mask = particles.K[:particles.last_alive] > ion_thresh
        
        # Compute ionization cross-sections for all particles (zeros for low energy)
        sigma_ion = cp.zeros(particles.last_alive, dtype=cp.float32)
        if cp.any(high_energy_mask):
            sigma_ion[high_energy_mask] = interp_table(
                particles.K[:particles.last_alive][high_energy_mask], 
                *cross_sections[1]
            )
        
        # Add to total cross-section
        sigma_T = elastic_sigma_T + sigma_ion
    
    # --- COMPUTE COLLISION FREQUENCIES ---
    # Collision frequency for all particles

    def compute_collision_frequencies(density, sigma_T, particles):
        nu = density * sigma_T * particles.V_abs[:particles.last_alive]
        # Apply mcc_mask: set frequency to 0 for ineligible particles
        nu = nu * particles.mcc_mask[:particles.last_alive]
        return nu
    
    nu = compute_collision_frequencies(density, sigma_T, particles)
    
    # Find maximum collision frequency
    nu_max = cp.max(nu)
    #if nu_max <= 0:
    #    return
   
    # --- DETERMINE COLLIDED PARTICLES ---
    
    def compute_collision_probability(nu, nu_max):
        # Generate random numbers for all particles
        rand = cp.random.random(particles.last_alive)
        
        # Collision probability for each particle
        P_coll = nu / nu_max
        
        # Determine which particles collide
        collision_mask = (rand < P_coll) & particles.mcc_mask[:particles.last_alive]
        return collision_mask
    
    collision_mask = compute_collision_probability(nu, nu_max)
    
   
    # Get indices of collided particles
    
    #collided_idxs = cp.where(collision_mask)[0]
    #collided_idxs = cp.arange(particles.last_alive)[collision_mask]
    
    return
    # --- DETERMINE COLLISION TYPE ---
    if IONIZATION:
        # Generate random numbers for collision type
        R = cp.random.random(size=collided_idxs.size)
       
        # Calculate elastic fraction for collided particles
        elastic_fraction = elastic_sigma_T[collided_idxs] / sigma_T[collided_idxs]
        is_elastic = R < elastic_fraction
       
        # Split indices by collision type
        elastic_idxs = collided_idxs[is_elastic]
        ionization_idxs = collided_idxs[~is_elastic]
       
        # Apply collisions
        if elastic_idxs.size > 0:
            apply_elastic(particles, elastic_idxs)
        if ionization_idxs.size > 0:
            apply_ionization(particles, ionization_idxs, max_particles)
    else:
        # All collisions are elastic
        apply_elastic(particles, collided_idxs)

if __name__ == "__main__":
    from cupyx.profiler import benchmark

    part = Particles.Particles2D(1000)

    print(benchmark(compute_speeds_and_energies, (part,), n_repeat=1000))




