import cupy as cp
import Consts


cylindrical_bilinear_kernel = cp.RawKernel(r'''
extern "C" __global__
void cylindrical_bilinear_weights(
    const float *R,  // [2, max_particles]
    float dr, float dz,
    int m, int n,
    int last_alive,
    float *weights,
    int *indices)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (pid < last_alive) {
        float r = R[2*pid] / dr;      // Radial position in grid units
        float z = R[2*pid+1] / dz;      // Axial position in grid units
       
        // Compute integer indices of the lower-left grid point
        int r0 = floorf(r);
        int z0 = floorf(z);
       
        // Compute upper indices
        int r1 = r0 + 1;
        int z1 = z0 + 1;
       
        // Compute fractional distances for weighting
        float wr = r - r0;
        float wz = z - z0;
        float wr_1 = 1.0f - wr;
        float wz_1 = 1.0f - wz;
       
        // Adjust weights for cylindrical volume elements
        float r_factor_0 = (r0 + 0.5f) * dr;  // Approximate radius at r0
        float r_factor_1 = (r1 + 0.5f) * dr;  // Approximate radius at r1
       
        // Avoid division by zero
        float norm_factor = 1.0f;
        float sum = r_factor_0 + r_factor_1;
        if (sum > 1e-6f) {
            norm_factor = 1.0f / sum;
        }
       
        // Compute interpolation weights with radial scaling
        weights[4*pid]                     = wr_1 * wz_1 * r_factor_0 * norm_factor;  // bottom-left
        weights[4*pid+1]     = wr   * wz_1 * r_factor_1 * norm_factor;  // bottom-right
        weights[4*pid+2] = wr_1 * wz   * r_factor_0 * norm_factor;  // top-left
        weights[4*pid+3] = wr   * wz   * r_factor_1 * norm_factor;  // top-right
       
        // Compute 1D indices for array access (z * m + r)
        indices[4*pid]                     = z0 * m + r0;;  // bottom-left
        indices[4*pid+1]     = z0 * m + r1;  // bottom-right
        indices[4*pid+2] = z1 * m + r0;  // top-left
        indices[4*pid+3] = z1 * m + r1;  // top-right
    }
}
''', 'cylindrical_bilinear_weights')

# Kernel code for Cartesian coordinates with correct memory layout
cartesian_bilinear_kernel = cp.RawKernel(r'''
extern "C" __global__
void cartesian_bilinear_weights(
    const float *R,
    float dx, float dy,
    int m, int n,
    int last_alive,
    float *weights,
    int *indices)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (pid < last_alive) {
        // Convert to grid coordinates
        float x = R[2*pid] / dx;
        float y = R[2*pid+1] / dy;
       
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
        
        weights[4*pid] = wx_1 * wy_1;
        weights[4*pid+1] = wx * wy_1;
        weights[4*pid+2] = wx_1 * wy;
        weights[4*pid+3] = wx * wy;

        indices[4*pid] = y0 * m + x0;
        indices[4*pid+1] = y0 * m + x1;
        indices[4*pid+2] = y1 * m + x0;
        indices[4*pid+3] = y1 * m + x1;
    }
}
''', 'cartesian_bilinear_weights')

update_v_direct_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_particle_velocity_direct(
    float *V,              // [3, max_particles]
    const float *R,        // [2, max_particles] 
    const float *Ex, const float *Ey, // [num_grid_points]
    const int *part_type,             // [max_particles]
    const float *q_type,              // [num_types]
    const float *m_type_inv,          // [num_types]
    float dt, float dx, float dy, int m, int n,
    int last_alive)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= last_alive) return;

    // Calculate grid position directly
    float x = R[2*pid] / dx;
    float y = R[2*pid+1] / dy;
    
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
    
    // Calculate bilinear weights
    float w0 = wx_1 * wy_1;
    float w1 = wx * wy_1;
    float w2 = wx_1 * wy;
    float w3 = wx * wy;

    // Calculate grid indices
    int i0 = y0 * m + x0;
    int i1 = y0 * m + x1;
    int i2 = y1 * m + x0;
    int i3 = y1 * m + x1;

    // Interpolate electric field
    float Epx = w0 * Ex[i0] + w1 * Ex[i1] + w2 * Ex[i2] + w3 * Ex[i3];
    float Epy = w0 * Ey[i0] + w1 * Ey[i1] + w2 * Ey[i2] + w3 * Ey[i3];

    int type = part_type[pid];
    float k = dt * q_type[type] * m_type_inv[type];
                               
    V[3*pid] += k * Epx;
    V[3*pid+1] += k * Epy;
}
''', 'update_particle_velocity_direct')


update_v_cylindrical_direct_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_particle_velocity_cylindrical_direct(
    float *V,              // [3, max_particles]
    const float *R,        // [2, max_particles] 
    const float *Er, const float *Ez, // [num_grid_points]
    const int *part_type,             // [max_particles]
    const float *q_type,              // [num_types]
    const float *m_type_inv,          // [num_types]
    float dt, float dr, float dz, int nr, int nz,
    int last_alive)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= last_alive) return;

    // Calculate grid position directly
    float r = R[2*pid] / dr;      // Normalized r coordinate
    float z = R[2*pid+1] / dz;    // Normalized z coordinate
    
    // Handle NaN or inf values
    if (!isfinite(r)) r = 0.0f;
    if (!isfinite(z)) z = 0.0f;
    
    // Clamp positions to valid grid range
    r = max(0.0f, min(r, float(nr - 1)));
    z = max(0.0f, min(z, float(nz - 1)));
    
    // Get integer cell coordinates
    int r0 = int(floorf(r));
    int z0 = int(floorf(z));
    
    // Calculate neighbor coordinates
    int r1 = min(r0 + 1, nr - 1);
    int z1 = min(z0 + 1, nz - 1);
    
    // Calculate interpolation weights
    float wr = r - float(r0);
    float wz = z - float(z0);
    float wr_1 = 1.0f - wr;
    float wz_1 = 1.0f - wz;
    
    // Calculate bilinear weights
    float w0 = wr_1 * wz_1;
    float w1 = wr * wz_1;
    float w2 = wr_1 * wz;
    float w3 = wr * wz;

    // Calculate grid indices
    int i0 = z0 * nr + r0;
    int i1 = z0 * nr + r1;
    int i2 = z1 * nr + r0;
    int i3 = z1 * nr + r1;

    // Interpolate electric field
    float Epr = w0 * Er[i0] + w1 * Er[i1] + w2 * Er[i2] + w3 * Er[i3];
    float Epz = w0 * Ez[i0] + w1 * Ez[i1] + w2 * Ez[i2] + w3 * Ez[i3];

    int type = part_type[pid];
    float k = dt * q_type[type] * m_type_inv[type];
                                 
    V[3*pid] += k * Epr;     // Radial velocity component
    V[3*pid+1] += k * Epz;   // Axial velocity component
    
    // For completeness: handle angular velocity component if needed
    // Note: In cylindrical coordinates, careful handling of angular E-field 
    // components would be needed here if they exist
}
''', 'update_particle_velocity_cylindrical_direct')


update_r_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_particle_position(
    float *R_old, float *R_new, float *V, float dt, int last_alive)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= last_alive) return;

    R_old[idx*2] = R_new[idx*2];
    R_old[idx*2+1] = R_new[idx*2+1];
    R_new[idx*2] += V[idx*3] * dt;
    R_new[idx*2+1] += V[idx*3+1] * dt;
}
''', 'update_particle_position')


axis_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_particle_axis(
    float *R, float *V, int last_alive)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= last_alive) return;

    if (R[idx*2+1] < 0.0f) {
        V[idx*3+1] = -V[idx*3+1];
    }
}
''', 'update_particle_axis')

class Particles2D():
    def __init__(self, N, cylindrical=False):

        self.cylindrical = cylindrical # 'cartesian' or 'cylindrical'
        self.N = N

        self.R = cp.zeros((N, 2), dtype=cp.float32)
        self.R_old = cp.zeros((N, 2), dtype=cp.float32)
        self.V = cp.zeros((N, 3), dtype=cp.float32)   
        self.part_type = cp.zeros(N, dtype=cp.int32) # particle type 

        #self.weights = cp.zeros((N, 4), dtype=cp.float32)
        #self.indices = cp.zeros((N, 4), dtype=cp.int32)

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

        self.mcc_mask = cp.zeros(N, dtype=cp.bool_)

        self.cross_sections = [None]
        self.species_count = 0

        self.collision_counter = cp.zeros(1, dtype=cp.int32)

        self.min_vx = 0
        self.max_vx = 0
        self.min_vy = 0
        self.max_vy = 0

        self.np2c = 2e4
    
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

    def set_mcc_mask(self):
        model = cp.where(self.collision_model == 0)[0][0]
        self.mcc_mask = self.part_type == model

    def update_R(self, dt):
        
        block_size = 256
        grid_size = (self.last_alive + block_size - 1) // block_size
        update_r_kernel(
            (grid_size,), (block_size,),
            (self.R_old, self.R, self.V,
            cp.float32(dt), cp.int32(self.last_alive))
        )

    def update_axis(self, dt):
        block_size = 256
        grid_size = (self.last_alive + block_size - 1) // block_size
        axis_kernel(
            (grid_size,), (block_size,),
            (self.R, self.V,
            cp.int32(self.last_alive))
        )
    
    def update_V(self, grid, dt):
        block_size = 256
        grid_size = (self.last_alive + block_size - 1) // block_size

        if self.cylindrical:
            # Call cylindrical direct version
            update_v_cylindrical_direct_kernel(
                (grid_size,), (block_size,),
                (
                    self.V, self.R,
                    grid.E[0], grid.E[1],
                    self.part_type,
                    self.q_type, self.m_type_inv,
                    cp.float32(dt), cp.float32(grid.cell_size[0]), cp.float32(grid.cell_size[1]), 
                    cp.int32(grid.gridshape[0]), cp.int32(grid.gridshape[1]),
                    self.last_alive
                )
            )
            
        else:
            # Call cartesian direct version
            update_v_direct_kernel(
                (grid_size,), (block_size,),
                (
                    self.V, self.R,
                    grid.E[0], grid.E[1],
                    self.part_type,
                    self.q_type, self.m_type_inv,
                    cp.float32(dt), cp.float32(grid.cell_size[0]), cp.float32(grid.cell_size[1]), 
                    cp.int32(grid.gridshape[0]), cp.int32(grid.gridshape[1]),
                    self.last_alive
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
                (self.R,  # Pass R components separately
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
                (self.R,  # Pass R components separately
                cp.float32(dx), cp.float32(dy), 
                cp.int32(m), cp.int32(n), 
                cp.int32(last_alive), 
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
        #self.part_type[indices] = 0
        swap_idx = cp.arange(self.last_alive - num_remove, self.last_alive)
        
        self.R[indices], self.R[swap_idx] = self.R[swap_idx].copy(), self.R[indices].copy()
        self.V[indices], self.V[swap_idx] = self.V[swap_idx].copy(), self.V[indices].copy()
        
        self.part_type[indices], self.part_type[swap_idx] = self.part_type[swap_idx].copy(), self.part_type[indices].copy()
        self.part_type[self.last_alive-num_remove:self.last_alive] = 0
        
        self.last_alive -= num_remove
        
        return num_remove
            
        #self.part_type[indices] = 0

    def ionize(self, indices):
        """
        Ionize particles by creating new particles and updating the particle type.
        """
        pass

    def emit(self, indices):
        """
        Emit particles from the wall.
        """
        self.part_type[indices] = self.part_name.index('electrons')

    def sputter(self, indices):
        """
        Sputter particles from the wall.
        """
        pass

    def uniform_particle_load(self, x, y, dx, dy, n):
        la = self.last_alive
        self.R[la:la + n, 0] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5, n)
        self.R[la:la + n, 1] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5, n)
        self.V[la:la + n, 0] = cp.random.uniform(-0.1, 0.1)
        self.V[la:la + n, 0] = cp.random.uniform(-0.1, 0.1)
        self.part_type[la:la + n] = cp.random.randint(1, 3, n)
        self.last_alive += n

    def uniform_species_load(self, x1, y1, x2, y2, n, species):
        la = self.last_alive
        self.R[la:la + n, 0] = cp.random.uniform(x1, x2, n)
        self.R[la:la + n, 1] = cp.random.uniform(y1, y2, n)
        self.V[la:la + n, 0] = cp.random.uniform(-0.1, 0.1)
        self.V[la:la + n, 0] = cp.random.uniform(-0.1, 0.1)
        self.part_type[la:la + n] = self.part_name.index(species)
        self.last_alive += n
