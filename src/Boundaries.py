import cupy as cp
import Grid

class Boundaries():
    def __init__(self, bounds, grid:Grid.Grid2D):
        
        print('creating boundary array...')
        self.conditions = self.boundary_array(bounds, grid.gridshape)
        print('creating wall map...')
        self.walls = self.mark_cell_walls_sparse(self.conditions[0], grid.gridshape)
        self.wall_lookup = cp.full(grid.m*grid.n, False, dtype=cp.bool_)
        self.wall_lookup[self.walls[0]] = True
        self.bound_tuple = []
        for i in bounds:
            x0, y0, x1, y1 = i[0]
            self.bound_tuple.append((int(x0), int(y0), int(x1), int(y1)))

        #move collision mask from collisons.py


    def boundary_array(self, input: tuple, gridsize: tuple) -> tuple[cp.ndarray, cp.ndarray]:
        # Initialize Python-native lists for indices and values
        boundary_indices = []
        boundary_values = []
        
        for segment in input:
            (m1, n1, m2, n2), V = segment  # Unpack segment

            m1 = int(m1)
            n1 = int(n1)
            m2 = int(m2)
            n2 = int(n2)

            dm = cp.sign(m2 - m1).item()  # Convert to Python scalar
            dn = cp.sign(n2 - n1).item()  # Convert to Python scalar
            
            x, y = m1, n1
            
            # Iterate over the boundary points
            while True:
                boundary_indices.append(y * gridsize[0] + x)  # Flattened index
                boundary_values.append(V)
                
                # Break loop if endpoint reached
                if (x == m2 and y == n2):
                    break
                
                # Update x and y
                x += dm
                y += dn
        
        # Convert Python-native lists to CuPy arrays
        indices = cp.array(boundary_indices, dtype=cp.int32)
        values = cp.array(boundary_values, dtype=cp.float32)
        
        return (indices, values)
    

    def mark_cell_walls_sparse(self, wall_nodes, grid_shape):
        Nx, Ny = grid_shape
        
        # Create a device array for wall nodes
        wall_nodes_gpu = cp.asarray(wall_nodes, dtype=cp.int32)
        
        # Convert node indices to 2D coordinates
        i, j = cp.divmod(wall_nodes_gpu, Ny)
        
        # Create masks for valid left and right wall positions
        left_wall_mask = j > 0
        right_wall_mask = j < Ny - 1
        
        # Compute cell indices for left walls
        left_cell_indices = i[left_wall_mask] * (Ny - 1) + (j[left_wall_mask] - 1)
        
        # Compute cell indices for right walls
        right_cell_indices = i[right_wall_mask] * (Ny - 1) + j[right_wall_mask]
        
        # Combine and unique cell indices
        unique_cell_indices, inverse_indices, counts = cp.unique(
            cp.concatenate([left_cell_indices, right_cell_indices]), 
            return_inverse=True, 
            return_counts=True
        )
        
        # Initialize wall directions array
        wall_directions = cp.zeros_like(unique_cell_indices, dtype=cp.uint8)
        
        # Mark left walls
        left_unique_indices, left_counts = cp.unique(left_cell_indices, return_counts=True)
        left_mask = cp.isin(unique_cell_indices, left_unique_indices)
        wall_directions[left_mask] |= 0b0010  # Right wall
        
        # Mark right walls
        right_unique_indices, right_counts = cp.unique(right_cell_indices, return_counts=True)
        right_mask = cp.isin(unique_cell_indices, right_unique_indices)
        wall_directions[right_mask] |= 0b0001  # Left wall
        
        return (unique_cell_indices, wall_directions)