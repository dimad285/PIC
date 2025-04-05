import cupy as cp
import Consts


cylindrical_bilinear_kernel = cp.RawKernel(r'''
extern "C" __global__
void cylindrical_bilinear_weights(
    const float *Rx, const float *Ry,  // R components are separate arrays
    float dr, float dz, 
    int m, int n, 
    int last_alive, 
    float *weights, 
    int *indices)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pid < last_alive) {
        float r = Rx[pid] / dr;      // Radial position in grid units
        float z = Ry[pid] / dz;      // Axial position in grid units
        
        // Ensure r and z are finite
        if (!isfinite(r)) r = 0.0f;
        if (!isfinite(z)) z = 0.0f;
        
        // Clamp r and z to valid ranges
        r = max(0.0f, min(r, float(m) - 1.0f - 1e-6f));
        z = max(0.0f, min(z, float(n) - 1.0f - 1e-6f));
        
        // Compute integer indices of the lower-left grid point
        int r0 = floorf(r);
        int z0 = floorf(z);
        
        // Ensure valid indexing
        r0 = max(0, min(r0, m - 2));  // Ensure r0+1 is valid
        z0 = max(0, min(z0, n - 2));  // Ensure z0+1 is valid
        
        // Compute upper indices (clamped at max index)
        int r1 = min(r0 + 1, m - 1);
        int z1 = min(z0 + 1, n - 1);
        
        // Compute fractional distances for weighting
        float wr = r - r0;
        float wz = z - z0;
        float wr_1 = 1.0f - wr;
        float wz_1 = 1.0f - wz;
        
        // Safety checks
        wr = max(0.0f, min(1.0f, wr));
        wz = max(0.0f, min(1.0f, wz));
        wr_1 = max(0.0f, min(1.0f, wr_1));
        wz_1 = max(0.0f, min(1.0f, wz_1));
        
        // Adjust weights for cylindrical volume elements (∝ r)
        float r_factor_0 = (r0 + 0.5f) * dr;  // Approximate radius at r0
        float r_factor_1 = (r1 + 0.5f) * dr;  // Approximate radius at r1
        
        // Avoid division by zero
        float norm_factor = 1.0f;
        float sum = r_factor_0 + r_factor_1;
        if (sum > 1e-6f) {
            norm_factor = 1.0f / sum;
        }
        
        // Compute interpolation weights with radial scaling
        weights[pid]                  = wr_1 * wz_1 * r_factor_0 * norm_factor;  // bottom-left
        weights[pid + last_alive]     = wr   * wz_1 * r_factor_1 * norm_factor;  // bottom-right
        weights[pid + last_alive * 2] = wr_1 * wz   * r_factor_0 * norm_factor;  // top-left
        weights[pid + last_alive * 3] = wr   * wz   * r_factor_1 * norm_factor;  // top-right
        
        // Ensure weights are finite and in [0,1]
        weights[pid]                  = max(0.0f, min(1.0f, weights[pid]));
        weights[pid + last_alive]     = max(0.0f, min(1.0f, weights[pid + last_alive]));
        weights[pid + last_alive * 2] = max(0.0f, min(1.0f, weights[pid + last_alive * 2]));
        weights[pid + last_alive * 3] = max(0.0f, min(1.0f, weights[pid + last_alive * 3]));
        
        // Compute 1D indices for array access (checking for integer overflow)
        long idx_bl = (long)z0 * (long)m + (long)r0;
        long idx_br = (long)z0 * (long)m + (long)r1;
        long idx_tl = (long)z1 * (long)m + (long)r0;
        long idx_tr = (long)z1 * (long)m + (long)r1;
        
        // Ensure indices are within valid range
        int max_idx = m * n - 1;
        indices[pid]                  = max(0, min((int)idx_bl, max_idx));  // bottom-left
        indices[pid + last_alive]     = max(0, min((int)idx_br, max_idx));  // bottom-right
        indices[pid + last_alive * 2] = max(0, min((int)idx_tl, max_idx));  // top-left
        indices[pid + last_alive * 3] = max(0, min((int)idx_tr, max_idx));  // top-right
    }
}
''', 'cylindrical_bilinear_weights')

# Kernel code for Cartesian coordinates with correct memory layout
cartesian_bilinear_kernel = cp.RawKernel(r'''
extern "C" __global__
void cartesian_bilinear_weights(
    const float *Rx, const float *Ry,
    float dx, float dy,
    int m, int n,
    int last_alive,
    int max_particles,
    float *weights,
    int *indices)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (pid < last_alive) {
        // Convert to grid coordinates
        float x = Rx[pid] / dx;
        float y = Ry[pid] / dy;
       
        // Handle NaN or inf values
        if (!isfinite(x)) x = 0.0f;
        if (!isfinite(y)) y = 0.0f;
       
        // Clamp positions to valid grid range
        x = max(0.0f, min(x, float(m - 1)));
        y = max(0.0f, min(y, float(n - 1)));
       
        // Get integer cell coordinates
        int x0 = int(floorf(x));
        int y0 = int(floorf(y));
        
        // Calculate neighbor coordinates
        int x1 = min(x0 + 1, m - 1);
        int y1 = min(y0 + 1, n - 1);
        
        // Calculate interpolation weights
        float wx = x - float(x0);
        float wy = y - float(y0);
        float wx_1 = 1.0f - wx;
        float wy_1 = 1.0f - wy;
        
        weights[0 * max_particles + pid] = wx_1 * wy_1;
        weights[1 * max_particles + pid] = wx * wy_1;
        weights[2 * max_particles + pid] = wx_1 * wy;
        weights[3 * max_particles + pid] = wx * wy;

        indices[0 * max_particles + pid] = y0 * n + x0;
        indices[1 * max_particles + pid] = y0 * n + x1;
        indices[2 * max_particles + pid] = y1 * n + x0;
        indices[3 * max_particles + pid] = y1 * n + x1;
    }
}
''', 'cartesian_bilinear_weights')

update_v_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_particle_velocity(
    float *Vx, float *Vy,              // [max_particles]
    const float *weights,             // [4, max_particles]
    const int *indices,               // [4, max_particles]
    const float *Ex, const float *Ey, // [num_grid_points]
    const int *part_type,             // [max_particles]
    const float *q_type,              // [num_types]
    const float *m_type_inv,          // [num_types]
    float dt,
    int last_alive,
    int max_particles)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= last_alive) return;

    float wx0 = weights[0 * max_particles + pid];
    float wx1 = weights[1 * max_particles + pid];
    float wx2 = weights[2 * max_particles + pid];
    float wx3 = weights[3 * max_particles + pid];

    int ix0 = indices[0 * max_particles + pid];
    int ix1 = indices[1 * max_particles + pid];
    int ix2 = indices[2 * max_particles + pid];
    int ix3 = indices[3 * max_particles + pid];

    float Epx = wx0 * Ex[ix0] + wx1 * Ex[ix1] + wx2 * Ex[ix2] + wx3 * Ex[ix3];
    float Epy = wx0 * Ey[ix0] + wx1 * Ey[ix1] + wx2 * Ey[ix2] + wx3 * Ey[ix3];

    int type = part_type[pid];
    float k = dt * q_type[type] * m_type_inv[type];
                               
    Vx[pid] += k * Epx;
    Vy[pid] += k * Epy;
}
''', 'update_particle_velocity')


update_r_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_particle_position(
    float *R_old, float *R_new, float *V, float dt, int last_alive, int max_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= last_alive) return;

    R_old[idx] = R_new[idx];
    R_old[idx + max_particles] = R_new[idx + max_particles];
    R_new[idx] += V[idx] * dt;
    R_new[idx + max_particles] += V[idx + max_particles] * dt;
}
''', 'update_particle_position')


class Particles2D():
    def __init__(self, N, cylindrical=False):

        self.cylindrical = cylindrical # 'cartesian' or 'cylindrical'
        self.N = N

        self.R = cp.zeros((2, N), dtype=cp.float32)
        self.R_old = cp.zeros((2, N), dtype=cp.float32)
        self.V = cp.zeros((3, N), dtype=cp.float32)   
        self.part_type = cp.zeros(N, dtype=cp.int32) # particle type 

        self.weights = cp.zeros((4, N), dtype=cp.float32)
        self.indices = cp.zeros((4, N), dtype=cp.int32)

        #self.R_grid = cp.zeros(N, dtype=cp.int32) # particle positions in grid space (in which cell they are)
        #self.R_grid_new = cp.zeros(N, dtype=cp.int32) # particle positions in grid space (in which cell they are)
        self.active_cells = None
        self.cell_starts = None
        self.sorted_indices = None

        self.last_alive = 0 # index of last alive particle
        
        # Species
        self.part_name = ['reserved']
        self.m_type = cp.array([0], dtype=cp.float32)
        self.m_type_inv = cp.array([0], dtype=cp.float32)
        self.q_type = cp.array([0], dtype=cp.float32)
        self.collision_model = cp.array([128], dtype=cp.int32)

        self.cross_sections = [None]
        self.species_count = 0

        self.min_vx = 0
        self.max_vx = 0
        self.min_vy = 0
        self.max_vy = 0

        self.np2c = 1
    
    @property
    def get_R(self):
        return self.R[:, :self.last_alive]

    def add_species(self, species_name:str, species_mass:float, species_charge:float, collision_model:int=0):
        self.part_name.append(species_name)
        self.m_type = cp.append(self.m_type, species_mass)
        self.m_type_inv = cp.append(self.m_type_inv, 1/species_mass)
        self.q_type = cp.append(self.q_type, species_charge)
        self.collision_model = cp.append(self.collision_model, collision_model)

        self.m_type = cp.asarray(self.m_type, dtype=cp.float32)
        self.m_type_inv = cp.asarray(self.m_type_inv, dtype=cp.float32)
        self.q_type = cp.asarray(self.q_type, dtype=cp.float32)
        self.collision_model = cp.asarray(self.collision_model, dtype=cp.int32)

        self.species_count += 1

    def update_R(self, dt):
        
        block_size = 256
        grid_size = (self.last_alive + block_size - 1) // block_size
        update_r_kernel(
            (grid_size,), (block_size,),
            (self.R_old, self.R, self.V,
            cp.float32(dt), cp.int32(self.last_alive), cp.int32(self.N))
        )
    
    '''
    def update_V(self, grid, dt):
        """
        Updates particle velocities using precomputed interpolation weights.
        """
        la = self.last_alive
        # Get interpolation weights and indices
        #weights, indices = compute_bilinear_weights(R, dx, dy, gridsize, last_alive)
        # Compute interpolated electric field components
        Ex = cp.sum(grid.E[0, self.indices[:, :la]] * self.weights[:, :la], axis=0)
        Ey = cp.sum(grid.E[1, self.indices[:, :la]] * self.weights[:, :la], axis=0)
        # Update velocities
        k = dt * self.q_type[self.part_type[:la]] * self.m_type_inv[self.part_type[:la]]
        
        self.V[0, :la] += Ex * k
        self.V[1, :la] += Ey * k
    
    '''

    def update_V(self, grid, dt):

        block_size = 256
        grid_size = (self.last_alive + block_size - 1) // block_size

        update_v_kernel(
            (grid_size,), (block_size,),
            (
                self.V[0], self.V[1],
                self.weights, self.indices,
                grid.E[0], grid.E[1],
                self.part_type,
                self.q_type, self.m_type_inv,
                cp.float32(dt), self.last_alive, self.N
            )
        )
    
    

    def boris_pusher_cartesian(self, grid, dt):
        """
        Updates particle velocities using the Boris method in Cartesian coordinates.
        """
        la = self.last_alive

        # Interpolate electric field components
        Ex = cp.sum(grid.E[0, self.indices[:, :la]] * self.weights[:, :la], axis=0)
        Ey = cp.sum(grid.E[1, self.indices[:, :la]] * self.weights[:, :la], axis=0)
        Ez = cp.sum(grid.E[2, self.indices[:, :la]] * self.weights[:, :la], axis=0)

        # Interpolate magnetic field components
        Bx = cp.sum(grid.B[0, self.indices[:, :la]] * self.weights[:, :la], axis=0)
        By = cp.sum(grid.B[1, self.indices[:, :la]] * self.weights[:, :la], axis=0)
        Bz = cp.sum(grid.B[2, self.indices[:, :la]] * self.weights[:, :la], axis=0)

        # Charge-to-mass ratio
        q_m = self.q_type[self.part_type[:la]] * self.m_type_inv[self.part_type[:la]]

        # Half-step electric field acceleration
        v_minus_x = self.V[0, :la] + 0.5 * dt * q_m * Ex
        v_minus_y = self.V[1, :la] + 0.5 * dt * q_m * Ey
        v_minus_z = self.V[2, :la] + 0.5 * dt * q_m * Ez

        # Magnetic rotation
        t_x = 0.5 * dt * q_m * Bx
        t_y = 0.5 * dt * q_m * By
        t_z = 0.5 * dt * q_m * Bz
        t_mag2 = t_x**2 + t_y**2 + t_z**2
        s_x = 2 * t_x / (1 + t_mag2)
        s_y = 2 * t_y / (1 + t_mag2)
        s_z = 2 * t_z / (1 + t_mag2)

        # v' = v_minus + v_minus x t
        v_prime_x = v_minus_x + (v_minus_y * t_z - v_minus_z * t_y)
        v_prime_y = v_minus_y + (v_minus_z * t_x - v_minus_x * t_z)
        v_prime_z = v_minus_z + (v_minus_x * t_y - v_minus_y * t_x)

        # v_plus = v_minus + 2 * (v' x s)
        v_plus_x = v_minus_x + (v_prime_y * s_z - v_prime_z * s_y)
        v_plus_y = v_minus_y + (v_prime_z * s_x - v_prime_x * s_z)
        v_plus_z = v_minus_z + (v_prime_x * s_y - v_prime_y * s_x)

        # Final velocity update with the second half-step electric field acceleration
        self.V[0, :la] = v_plus_x + 0.5 * dt * q_m * Ex
        self.V[1, :la] = v_plus_y + 0.5 * dt * q_m * Ey
        self.V[2, :la] = v_plus_z + 0.5 * dt * q_m * Ez


    def boris_pusher_cylindrical(self, grid, dt):
        """
        Updates particle velocities using the Boris method in cylindrical coordinates (z, r).
        """
        la = self.last_alive

        # Interpolate electric field components
        Ez = cp.sum(grid.E[0, self.indices[:, :la]] * self.weights[:, :la], axis=0)
        Er = cp.sum(grid.E[1, self.indices[:, :la]] * self.weights[:, :la], axis=0)
        Eθ = cp.sum(grid.E[2, self.indices[:, :la]] * self.weights[:, :la], axis=0)  # Azimuthal E-field

        # Interpolate magnetic field components
        Bz = cp.sum(grid.B[0, self.indices[:, :la]] * self.weights[:, :la], axis=0)  # Bz
        Br = cp.sum(grid.B[1, self.indices[:, :la]] * self.weights[:, :la], axis=0)  # Br
        Bθ = cp.sum(grid.B[2, self.indices[:, :la]] * self.weights[:, :la], axis=0)  # Bθ

        # Charge-to-mass ratio
        q_m = self.q_type[self.part_type[:la]] * self.m_type_inv[self.part_type[:la]]

        # Half-step electric field acceleration
        v_minus_z = self.V[0, :la] + 0.5 * dt * q_m * Ez
        v_minus_r = self.V[1, :la] + 0.5 * dt * q_m * Er
        v_minus_θ = self.V[2, :la] + 0.5 * dt * q_m * Eθ

        # Magnetic rotation
        t_z = 0.5 * dt * q_m * Bz
        t_r = 0.5 * dt * q_m * Br
        t_θ = 0.5 * dt * q_m * Bθ
        t_mag2 = t_z**2 + t_r**2 + t_θ**2
        s_z = 2 * t_z / (1 + t_mag2)
        s_r = 2 * t_r / (1 + t_mag2)
        s_θ = 2 * t_θ / (1 + t_mag2)

        # v' = v_minus + v_minus x t
        v_prime_z = v_minus_z + (v_minus_r * t_θ - v_minus_θ * t_r)
        v_prime_r = v_minus_r + (v_minus_θ * t_z - v_minus_z * t_θ)
        v_prime_θ = v_minus_θ + (v_minus_z * t_r - v_minus_r * t_z)

        # v_plus = v_minus + 2 * (v' x s)
        v_plus_z = v_minus_z + (v_prime_r * s_θ - v_prime_θ * s_r)
        v_plus_r = v_minus_r + (v_prime_θ * s_z - v_prime_z * s_θ)
        v_plus_θ = v_minus_θ + (v_prime_z * s_r - v_prime_r * s_z)

        # Final velocity update with the second half-step electric field acceleration
        self.V[0, :la] = v_plus_z + 0.5 * dt * q_m * Ez
        self.V[1, :la] = v_plus_r + 0.5 * dt * q_m * Er
        self.V[2, :la] = v_plus_θ + 0.5 * dt * q_m * Eθ

        # Apply centrifugal force correction for v_r
        r = self.X[1, :la]  # Radial position
        mask = r > 0  # Avoid division by zero
        self.V[1, mask] -= (self.V[2, mask] ** 2 / r[mask]) * dt
    


    def update_bilinear_weights(self, grid):
        # Choose block size for optimal occupancy
        threads_per_block = 256
        blocks = (self.last_alive + threads_per_block - 1) // threads_per_block
        
        # Extract parameters
        gridshape = grid.gridshape
        cell_size = grid.cell_size
        last_alive = self.last_alive
        
        # Call the appropriate kernel based on coordinate system
        if self.cylindrical:
            m, n = gridshape  # (radial, axial)
            dr, dz = cell_size
            cylindrical_bilinear_kernel(
                (blocks,), (threads_per_block,),
                (self.R[0, :last_alive], self.R[1, :last_alive],  # Pass R components separately
                cp.float32(dr), cp.float32(dz), 
                cp.int32(m), cp.int32(n), 
                cp.int32(last_alive), 
                self.weights, 
                self.indices)
            )
        else:
            m, n = gridshape
            dx, dy = cell_size
            cartesian_bilinear_kernel(
                (blocks,), (threads_per_block,),
                (self.R[0], self.R[1],  # Pass R components separately
                cp.float32(dx), cp.float32(dy), 
                cp.int32(m), cp.int32(n), 
                cp.int32(last_alive), 
                cp.int32(self.N),
                self.weights, 
                self.indices)
            )

    def sort_particles_sparse(self, num_cells):
        """
        Sort particles using a sparse array approach on GPU.
        Only stores data for non-empty cells and computes cell shifts.
    
        Parameters:
        -----------
        cell_indices : cupy.ndarray
            Array of cell indices for each particle
        num_cells : int
            Total number of cells
        
        Returns:
        --------
        active_cells : cupy.ndarray
            Indices of cells that contain particles
        cell_starts : cupy.ndarray
            Starting position for each active cell in the sorted array
        sorted_indices : cupy.ndarray
            Sorted particle indices
        """
        # Get counts for all cells
        full_counts = cp.bincount(self.R_grid[:self.last_alive], minlength=num_cells)
        #print('full_counts', full_counts, full_counts.shape)
    
        # Find non-empty cells
        active_cells = cp.nonzero(full_counts)[0]
        cell_counts = full_counts[active_cells]
    
        # Compute starting positions for active cells
        cell_starts = cp.zeros(len(active_cells), dtype=cp.int32)
        if len(active_cells) > 1:
            cp.cumsum(cell_counts[:-1], out=cell_starts[1:])
        
        # Create a mapping from all cells to active cell positions
        cell_to_active_map = -cp.ones(num_cells, dtype=cp.int32)
        cell_to_active_map[active_cells] = cp.arange(len(active_cells))
        
        # Temporary positions for atomic updates
        temp_positions = cp.zeros_like(cell_starts)
        
        # Create array for sorted particle indices
        sorted_indices = cp.zeros((self.last_alive), dtype=cp.int32)
    
        # Create a kernel for parallel particle sorting
        sort_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void sort_particles_sparse(const int* cell_indices,
                                const int* cell_to_active_map,
                                const int* cell_positions,
                                int* temp_positions,
                                int* sorted_indices,
                                const int num_particles) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < num_particles) {
                // Map the cell index to the active cell index
                int cell = cell_indices[idx];
                int active_cell_idx = cell_to_active_map[cell];
                if (active_cell_idx >= 0) {
                    // Atomically update temp_positions and compute position
                    int pos = atomicAdd(&temp_positions[active_cell_idx], 1);
                    int global_pos = cell_positions[active_cell_idx] + pos;
                    // Assign the particle index to the sorted array
                    sorted_indices[global_pos] = idx;
                }
            }
        }
        ''', 'sort_particles_sparse')
        
        # Kernel execution with error checking
        threads_per_block = 256
        blocks_per_grid = (len(self.R_grid) + threads_per_block - 1) // threads_per_block
        
        try:
            sort_kernel((blocks_per_grid,), (threads_per_block,),
                    (self.R_grid, cell_to_active_map, cell_starts, temp_positions,
                        sorted_indices, len(self.R_grid)))
        except Exception as e:
            raise RuntimeError(f"Kernel execution failed: {str(e)}")
        
        self.active_cells = active_cells
        self.cell_starts = cell_starts
        self.sorted_indices = sorted_indices


    def remove(self, indices):
        """
        Remove particles by swapping with the last active particle and decreasing the active particle counter.
        """
        num_remove = indices.size
        if num_remove == 0:
            return 0
        
        
        # Get indices of last alive particles to swap with
        swap_idx = cp.arange(self.last_alive - num_remove, self.last_alive)

        # Perform swaps
        self.R[:, indices], self.R[:, swap_idx] = self.R[:, swap_idx], self.R[:, indices]
        self.V[:, indices], self.V[:, swap_idx] = self.V[:, swap_idx], self.V[:, indices]
        self.part_type[indices], self.part_type[swap_idx] = self.part_type[swap_idx], self.part_type[indices]
        self.part_type[self.last_alive-num_remove:self.last_alive] = 0

        # Update alive count
        self.last_alive -= num_remove
        
        #self.part_type[indices] = 0


    def uniform_particle_load(self, x, y, dx, dy, n):
        la = self.last_alive
        self.R[0, la:la + n] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5, n)
        self.R[1, la:la + n] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5, n)
        self.V[0, la] = cp.random.uniform(-0.1, 0.1)
        self.V[1, la] = cp.random.uniform(-0.1, 0.1)
        self.part_type[la:la + n] = cp.random.randint(1, 3, n)
        self.last_alive += n

    def uniform_species_load(self, x1, y1, x2, y2, n, species):
        la = self.last_alive
        self.R[0, la:la + n] = cp.random.uniform(x1, x2, n)
        self.R[1, la:la + n] = cp.random.uniform(y1, y2, n)
        self.V[0, la] = cp.random.uniform(-0.1, 0.1)
        self.V[1, la] = cp.random.uniform(-0.1, 0.1)
        self.part_type[la:la + n] = self.part_name.index(species)
        self.last_alive += n

    update_r_kernel = cp.ElementwiseKernel(
        'raw T R_old, raw T R_new, raw T V, T dt, int32 last_alive',
        'T R_out',
        '''
        if (i < last_alive * 2) {  // 2 components (x,y)
            R_out = R_new[i] + V[i] * dt;
            R_old[i] = R_new[i];
        }
        ''',
        'update_r_kernel'
    )
