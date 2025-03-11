import cupy as cp
import Consts

class Particles2D():
    def __init__(self, N):
        self.R = cp.zeros((2, N))
        self.V = cp.zeros((2, N))   
        self.part_type = cp.zeros(N, dtype=cp.int32) # particle type (0 - empty, 1 - proton, 2 - electron)

        self.R_grid = cp.zeros(N, dtype=cp.int32) # particle positions in grid space (in which cell they are)
        #R_grid_new = cp.zeros(max_particles, dtype=cp.int32)
        # When paarticle 'dies' it's type is set to 0 and it is swapped with the last alive particle
        self.last_alive = 0 # index of last alive particle
        # Problems can be with particle sorting
        self.part_name = ['', 'proton', 'electron']
        self.m_type = cp.array([0, Consts.mp, Consts.me], dtype=cp.float64)
        self.m_type_inv = cp.array([0, 1/Consts.mp, 1/Consts.me], dtype=cp.float64)
        self.q_type = cp.array([0, 1.0 * Consts.qe, -1.0 * Consts.qe], dtype=cp.float64)
        self.weights = cp.zeros((4, N), dtype=cp.float32)
        self.indices = cp.zeros((4, N), dtype=cp.int32)
        self.active_cells = None
        self.cell_starts = None
        self.sorted_indices = None

    
    @property
    def get_R(self):
        return self.R[:, :self.last_alive]

    def add_species(self, species_name:str, species_mass:float, species_charge:float):
        self.part_name.append(species_name)
        self.m_type = cp.append(self.m_type, species_mass)
        self.m_type_inv = cp.append(self.m_type_inv, 1/species_mass)
        self.q_type = cp.append(self.q_type, species_charge)

    def update_R(self, dt):
        self.R[:, :self.last_alive] += self.V[:, :self.last_alive] * dt

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
        #print('cell coords', self.R_grid[:la])

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