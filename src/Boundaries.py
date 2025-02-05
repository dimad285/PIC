import numpy as np
import cupy as cp
import Grid

class Boundaries():
    def __init__(self, bounds, grid:Grid.Grid2D):
        
        print('creating boundary array...')
        self.conditions = self.boundary_array(bounds, grid.gridshape)
        print('creating wall map...')
        self.walls = self.mark_cell_walls_sparse(self.conditions[0], grid.gridshape)
        self.bound_tuple = []
        for i in bounds:
            x0, y0, x1, y1 = i[0]
            self.bound_tuple.append((int(x0), int(y0), int(x1), int(y1)))


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
        Nx, Ny = grid_shape  # Number of nodes in x and y directions
        cell_indices = []
        wall_directions = []

        for node_idx in wall_nodes:
            # Convert node index to 2D coordinates
            i, j = divmod(node_idx, Ny)

            # Vertical wall logic
            if j > 0:  # Left wall (affects the right wall of the left cell)
                cell_idx = i * (Ny - 1) + (j - 1)
                if cell_idx not in cell_indices:
                    cell_indices.append(cell_idx)
                    wall_directions.append(0)
                wall_directions[cell_indices.index(cell_idx)] |= 0b0010  # Right wall
            if j < Ny - 1:  # Right wall (affects the left wall of the right cell)
                cell_idx = i * (Ny - 1) + j
                if cell_idx not in cell_indices:
                    cell_indices.append(cell_idx)
                    wall_directions.append(0)
                wall_directions[cell_indices.index(cell_idx)] |= 0b0001  # Left wall

        # Convert lists to CuPy arrays
        cell_indices = cp.array(cell_indices, dtype=cp.int32)
        wall_directions = cp.array(wall_directions, dtype=cp.uint8)
        return (cell_indices, wall_directions)