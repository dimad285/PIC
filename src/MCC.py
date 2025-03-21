import cupy as cp
import Particles
import Consts
import Grid

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
    


def null_collision_method(particles:Particles.Particles2D, grid:Grid.Grid2D, densities, cross_sections, dt:float, max_particles:int):
    """
    Implements the null collision method with ionization and elastic collisions.

    Args:
        particles (Particles2D): Particle data.
        densities (cp.ndarray): Neutral gas densities at grid points.
        cross_sections (tuple): ((energies, elastic_cross_sections), (energies, ionization_cross_sections)).
        ionization_threshold (float): Minimum energy in eV required for ionization.
        dt (float): Time step.
        max_particles (int): Maximum number of allowed particles.
    """

    # Select only electrons (type 2)
    electron_mask = particles.part_type[:particles.last_alive] == 2
    electron_indices = cp.where(electron_mask)[0]

    if electron_indices.size == 0:
        return  # No electrons, nothing to do

    ionization_threshold = cross_sections[1][0][0]
    # Compute electron energies (in eV)
    electron_velocities = particles.V[:, electron_indices]
    speeds = cp.linalg.norm(electron_velocities, axis=0)
    energies = 0.5 * Consts.me * speeds**2 / Consts.qe  # Convert to eV

    # Interpolate cross-sections based on electron energy
    elastic_sigma_T = cp.interp(energies, cross_sections[0][0], cross_sections[0][1])
    ionization_sigma_T = cp.interp(energies, cross_sections[1][0], cross_sections[1][1])

    # Apply ionization threshold
    ionization_sigma_T *= energies > ionization_threshold

    # Compute total collision cross-section
    sigma_T = elastic_sigma_T + ionization_sigma_T

    # Compute collision frequency
    collision_frequencies = densities * sigma_T * speeds

    # Compute total probability of collision
    nu_max = cp.max(collision_frequencies) if collision_frequencies.size > 0 else 0
    P_T = 1 - cp.exp(-nu_max * dt)

    # Select colliding electrons
    colliding_mask = cp.random.uniform(0, 1, electron_indices.shape[0]) < P_T
    colliding_electron_indices = electron_indices[cp.where(colliding_mask)[0]]

    # Random numbers to decide collision type
    R_values = cp.random.uniform(0, 1, colliding_electron_indices.shape)
    ionization_prob = ionization_sigma_T[colliding_mask] / sigma_T[colliding_mask]

    # --- Ionization ---
    ionization_mask = (R_values < ionization_prob) & (energies[colliding_mask] > ionization_threshold)

    if cp.any(ionization_mask):
        ionizing_indices = colliding_electron_indices[ionization_mask]
        num_new = ionizing_indices.shape[0] * 2
        num_available = max_particles - particles.last_alive
        if num_new > num_available:
            num_to_create = 0
        else: 
            num_to_create = num_new

        if num_to_create > 0:
            new_indices = cp.arange(particles.last_alive, particles.last_alive + num_to_create)
            particles.last_alive += num_to_create


            particles.part_type[new_indices[::2]] = 2  # Electrons
            particles.part_type[new_indices[1::2]] = 1  # Ions  

            # Assign new positions
            #print(cp.repeat(particles.R[:, ionizing_indices[:num_to_create // 2]], 2, axis=1))
            source_positions = particles.R[:, ionizing_indices[:num_to_create // 2]]

            # Check for NaNs before assignment
            if cp.any(cp.isnan(source_positions)):
                print("Warning: NaN detected in source positions, replacing with random values.")
                source_positions = cp.nan_to_num(source_positions, nan=cp.random.uniform(0, grid.x_max, size=source_positions.shape))

            particles.R[:, new_indices] = cp.repeat(source_positions, 2, axis=1)


            # Assign new velocities (random low-energy)
            #theta = cp.random.uniform(0, 2 * cp.pi, num_to_create)
            #secondary_energy = cp.random.uniform(0, ionization_threshold, num_to_create)  # Secondary electrons get random low energy
            #speed = cp.sqrt(2 * secondary_energy * Consts.qe / Consts.me)
            #particles.V[0, new_indices] = speed * cp.cos(theta)
            #particles.V[1, new_indices] = speed * cp.sin(theta)

            # Reduce original electron energy
            denominator = cp.maximum(energies[ionizing_indices][:num_to_create // 2], 1e-30)
            scale_factor = cp.sqrt(cp.maximum((energies[ionizing_indices][:num_to_create // 2] - ionization_threshold) / denominator, 0))
            particles.V[:, ionizing_indices[:num_to_create // 2]] *= scale_factor

    # --- Elastic Scattering ---
    elastic_mask = ~ionization_mask
    elastic_indices = colliding_electron_indices[elastic_mask]

    if cp.any(elastic_indices):
        theta = cp.random.uniform(0, 2 * cp.pi, elastic_indices.shape[0])
        particles.V[:, elastic_indices] = cp.sqrt(2 * energies[colliding_mask][elastic_mask] * Consts.qe / Consts.me) * cp.array(
            [cp.cos(theta), cp.sin(theta)]
        )