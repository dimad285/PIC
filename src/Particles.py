import cupy as cp
import Consts

class Particles2D():
    def __init__(self, N, cylindrical=False):

        self.cylindrical = cylindrical # 'cartesian' or 'cylindrical'

        self.R = cp.zeros((2, N))
        self.V = cp.zeros((3, N))   
        self.part_type = cp.zeros(N, dtype=cp.int32) # particle type (0 - empty, 1 - proton, 2 - electron)

        self.R_grid = cp.zeros(N, dtype=cp.int32) # particle positions in grid space (in which cell they are)
        #R_grid_new = cp.zeros(max_particles, dtype=cp.int32)
        # When paarticle 'dies' it's type is set to 0 and it is swapped with the last alive particle
        self.last_alive = 0 # index of last alive particle
        # Problems can be with particle sorting
        self.part_name = ['', 'proton', 'electron']
        self.m_type = cp.array([0, Consts.mp, Consts.me], dtype=cp.float32)
        self.m_type_inv = cp.array([0, 1/Consts.mp, 1/Consts.me], dtype=cp.float32)
        self.q_type = cp.array([0, 1.0 * Consts.qe, -1.0 * Consts.qe], dtype=cp.float32)
        self.weights = cp.zeros((4, N), dtype=cp.float32)
        self.indices = cp.zeros((4, N), dtype=cp.int32)
        self.active_cells = None
        self.cell_starts = None
        self.sorted_indices = None

        self.cross_sectiond = [None]

        self.species_count = []
        self.total_count = 0

        self.min_vx = 0
        self.max_vx = 0
        self.min_vy = 0
        self.max_vy = 0

        self.np2c = 1
    
    @property
    def get_R(self):
        return self.R[:, :self.last_alive]

    def add_species(self, species_name:str, species_mass:float, species_charge:float):
        self.part_name.append(species_name)
        self.m_type = cp.append(self.m_type, species_mass)
        self.m_type_inv = cp.append(self.m_type_inv, 1/species_mass)
        self.q_type = cp.append(self.q_type, species_charge)

    def update_R(self, dt):
        self.R[:, :self.last_alive] += self.V[:2, :self.last_alive] * dt

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
        """
        Computes bilinear interpolation weights and cell indices for particles in place.
        
        Args:
            R: Particle positions array (2 x N)
            dx, dy: Grid spacing
            gridsize: Grid dimensions (m, n)
            last_alive: Number of active particles
            weights: Array to store interpolation weights for each particle (4 x last_alive)
            indices: Array to store cell indices for each particle (4 x last_alive)
        """

        

        if self.cylindrical:

            m, n = grid.gridshape  # (radial, axial)
            dr, dz = grid.cell_size
            la = self.last_alive  # Number of active particles
            # Normalize positions
            r = self.R[0, :la] / dr  # Radial position in grid units
            z = self.R[1, :la] / dz  # Axial position in grid units

            # Compute integer indices of the lower-left grid point
            r0 = cp.floor(r).astype(cp.int32)
            z0 = cp.floor(z).astype(cp.int32)
            
            # Ensure valid indexing (r0 must be at least 0, z0 at least 0)
            r0 = cp.maximum(r0, 0)
            z0 = cp.maximum(z0, 0)
            
            # Compute upper indices (clamped at max index)
            r1 = cp.minimum(r0 + 1, m - 1)
            z1 = cp.minimum(z0 + 1, n - 1)

            # Compute fractional distances for weighting
            wr = r - r0
            wz = z - z0
            wr_1 = 1.0 - wr
            wz_1 = 1.0 - wz
            
            # Adjust weights for cylindrical volume elements (∝ r)
            r_factor_0 = (r0 + 0.5) * dr  # Approximate radius at r0
            r_factor_1 = (r1 + 0.5) * dr  # Approximate radius at r1
            norm_factor = 1.0 / (r_factor_0 + r_factor_1)  # Normalize weights
            
            # Compute interpolation weights with radial scaling
            self.weights[0, :la] = wr_1 * wz_1 * r_factor_0 * norm_factor  # bottom-left
            self.weights[1, :la] = wr * wz_1 * r_factor_1 * norm_factor    # bottom-right
            self.weights[2, :la] = wr_1 * wz * r_factor_0 * norm_factor    # top-left
            self.weights[3, :la] = wr * wz * r_factor_1 * norm_factor      # top-right

            # Compute 1D indices for array access
            self.indices[0, :la] = z0 * m + r0  # bottom-left
            self.indices[1, :la] = z0 * m + r1  # bottom-right
            self.indices[2, :la] = z1 * m + r0  # top-left
            self.indices[3, :la] = z1 * m + r1  # top-right

        else:

            m, n = grid.gridshape
            dx, dy = grid.cell_size
            la = self.last_alive
            # Normalize positions
            x = self.R[0, :la] / dx
            y = self.R[1, :la] / dy
            
            # Calculate indices
            x0 = cp.floor(x).astype(cp.int32)
            y0 = cp.floor(y).astype(cp.int32)
            x1 = cp.minimum(x0 + 1, m - 1)
            y1 = cp.minimum(y0 + 1, n - 1)

            # Calculate weights
            wx = x - x0
            wy = y - y0
            wx_1 = 1.0 - wx
            wy_1 = 1.0 - wy
            
            # Update weights in place
            self.weights[0, :la] = wx_1 * wy_1  # bottom-left
            self.weights[1, :la] = wx * wy_1    # bottom-right
            self.weights[2, :la] = wx_1 * wy    # top-left
            self.weights[3, :la] = wx * wy      # top-right
            
            # Update indices in place
            self.indices[0, :la] = y0 * n + x0  # bottom-left
            self.indices[1, :la] = y0 * n + x1  # bottom-right
            self.indices[2, :la] = y1 * n + x0  # top-left
            self.indices[3, :la] = y1 * n + x1  # top-right

            self.R_grid[:la] = self.indices[0, :la] - self.indices[0, :la]//m

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


    def uniform_particle_generator_2d(self, x, y, dx, dy):
        la = self.last_alive
        self.R[0, la] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5)
        self.R[1, la] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5)
        self.V[0, la] = cp.random.uniform(-0.1, 0.1)
        self.V[1, la] = cp.random.uniform(-0.1, 0.1)
        self.part_type[la] = cp.random.randint(1, 3)
        self.last_alive += 1

    def uniform_particle_load(self, x, y, dx, dy, n):
        la = self.last_alive
        self.R[0, la:la + n] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5, n)
        self.R[1, la:la + n] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5, n)
        self.V[0, la] = cp.random.uniform(-0.1, 0.1)
        self.V[1, la] = cp.random.uniform(-0.1, 0.1)
        self.part_type[la:la + n] = cp.random.randint(1, 3, n)
        self.last_alive += n

    def uniform_species_load(self, x, y, dx, dy, n, species):
        la = self.last_alive
        self.R[0, la:la + n] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5, n)
        self.R[1, la:la + n] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5, n)
        self.V[0, la] = cp.random.uniform(-0.1, 0.1)
        self.V[1, la] = cp.random.uniform(-0.1, 0.1)
        self.part_type[la:la + n] = self.part_name.index(species)
        self.last_alive += n