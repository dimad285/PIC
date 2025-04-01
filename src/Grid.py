import cupy as cp
import Particles



class Grid2D():
    def __init__(self, m, n, X, Y, cylindrical=False):

        self.cylindrical = cylindrical # 'cartesian' or 'cylindrical'

        self.m = m
        self.n = n
        self.X = X
        self.Y = Y
        self.dx = X/(m-1)
        self.dy = Y/(n-1)
        self.cylindrical = cylindrical

        self.gridshape = (m, n)
        self.node_count = m*n
        self.cell_size = (self.dx, self.dy)
        self.cell_count = (m-1) * (n-1)
        self.domain = (X, Y)

        self.rho = cp.zeros(m*n)
        self.wall_rho = cp.zeros(m*n)
        self.b = cp.zeros(m*n) # righr-hand side of the Poisson equation
        self.phi = cp.zeros(m*n)
        self.E = cp.zeros((2, m*n))
        self.J = cp.zeros((2, m*n))
        self.A = cp.zeros((2, m*n))
        self.B = cp.zeros((2, m*n))
        self.NGD = cp.ones(m*n)

    def update_E(self):
        # Reshape phi into a 2D grid using Fortran-order for better y-direction performance
        phi_grid = cp.reshape(self.phi, self.gridshape, order='C')  # Use Fortran-order ('F') for column-major memory layout
        
        # Compute gradients in the x and y directions
        E_x = -cp.gradient(phi_grid, self.dx, axis=1)  # Gradient in the x-direction (axis 1)
        E_y = -cp.gradient(phi_grid, self.dy, axis=0)  # Gradient in the y-direction (axis 0)
        
        if self.cylindrical:
            E_y[0, :] = 0  # Enforce symmetry at r = 0

        # Flatten the results and assign to E
        self.E[0, :] = E_x.flatten()  # Flatten using the same Fortran-order to match the reshaping
        self.E[1, :] = E_y.flatten()

    def update_density(self, particles:Particles.Particles2D):
        """
        Updates charge density field using precomputed bilinear interpolation.
        """
        self.rho.fill(0.0)
        l_a = particles.last_alive

        if self.cylindrical:
            charge_values = particles.q_type[particles.part_type[:l_a]] * particles.np2c

            # Get radial indices of the interpolation points
            r_idx_0 = particles.indices[0, :l_a] % self.gridshape[0]  # bottom-left
            r_idx_1 = particles.indices[1, :l_a] % self.gridshape[0]  # bottom-right
            r_idx_2 = particles.indices[2, :l_a] % self.gridshape[0]  # top-left
            r_idx_3 = particles.indices[3, :l_a] % self.gridshape[0]  # top-right

            # Compute radial positions (cell-centered approximation)
            r0 = r_idx_0 * self.cell_size[0]
            r1 = r_idx_1 * self.cell_size[0]
            r2 = r_idx_2 * self.cell_size[0]
            r3 = r_idx_3 * self.cell_size[0]

            # Avoid division by zero at r = 0
            r0 = cp.maximum(r0, self.dy)
            r1 = cp.maximum(r1, self.dy)
            r2 = cp.maximum(r2, self.dy)
            r3 = cp.maximum(r3, self.dy)

            # Compute cell volume elements V = (r_{i+1} - r_i) * dz * 2π
            dr = self.cell_size[0]
            dz = self.cell_size[1]
            volume_0 = (r0 + dr / 2) * dr * dz * (2 * cp.pi)
            volume_1 = (r1 + dr / 2) * dr * dz * (2 * cp.pi)
            volume_2 = (r2 + dr / 2) * dr * dz * (2 * cp.pi)
            volume_3 = (r3 + dr / 2) * dr * dz * (2 * cp.pi)

            # Scale weights by radial volume elements
            weight_0 = particles.weights[0, :l_a] / volume_0
            weight_1 = particles.weights[1, :l_a] / volume_1
            weight_2 = particles.weights[2, :l_a] / volume_2
            weight_3 = particles.weights[3, :l_a] / volume_3

            # Deposit charge density with volume correction
            cp.add.at(self.rho, particles.indices[0, :l_a], charge_values * weight_0)
            cp.add.at(self.rho, particles.indices[1, :l_a], charge_values * weight_1)
            cp.add.at(self.rho, particles.indices[2, :l_a], charge_values * weight_2)
            cp.add.at(self.rho, particles.indices[3, :l_a], charge_values * weight_3)

        else:
            charge_values = particles.q_type[particles.part_type[:l_a]] * particles.np2c
            dV = self.cell_size[0] * self.cell_size[1]
            cp.add.at(self.rho, particles.indices[0, :l_a], charge_values * particles.weights[0, :l_a] / dV)
            cp.add.at(self.rho, particles.indices[1, :l_a], charge_values * particles.weights[1, :l_a] / dV)
            cp.add.at(self.rho, particles.indices[2, :l_a], charge_values * particles.weights[2, :l_a] / dV)
            cp.add.at(self.rho, particles.indices[3, :l_a], charge_values * particles.weights[3, :l_a] / dV)


    def update_J(self, particles:Particles.Particles2D):
        pass


    def minmax_phi(self):
        return cp.min(self.phi), cp.max(self.phi)
    
    def minmax_rho(self):
        return cp.min(self.rho), cp.max(self.rho)
    

    def save_to_txt(self, filename, fields=None, header=None):
        """
        Save grid parameters to a text file with columns where the first two columns
        are x and y coordinates, and the remaining columns are field values at those points.
        
        Parameters:
        ----------
        filename : str
            Name of the output text file.
        fields : dict, optional
            Dictionary mapping field names to field arrays. Default includes all available fields.
            Example: {'rho': self.rho, 'phi': self.phi}
        header : str, optional
            Header string for the text file. If None, a default header is created.
        
        Notes:
        ------
        This method saves the grid data in a text file with coordinates and field values
        in columns, making it easy to import into other plotting applications.
        """
        import numpy as np
        
        # If no fields are specified, use all available fields
        if fields is None:
            fields = {
                'rho': self.rho,
                'phi': self.phi,
                'E_x': self.E[0],
                'E_y': self.E[1],
                'J_x': self.J[0],
                'J_y': self.J[1],
                'A_x': self.A[0],
                'A_y': self.A[1],
                'B_x': self.B[0],
                'B_y': self.B[1],
                'NGD': self.NGD
            }
        
        # Generate x and y coordinate arrays
        x = np.zeros(self.node_count)
        y = np.zeros(self.node_count)
        
        for j in range(self.n):
            for i in range(self.m):
                idx = j * self.m + i
                x[idx] = i * self.dx
                y[idx] = j * self.dy
        
        # Convert field arrays from cupy to numpy if needed
        field_data = []
        field_names = []
        
        for name, field in fields.items():
            field_names.append(name)
            if hasattr(field, 'get'):  # Check if it's a cupy array
                field_data.append(field.get())  # Convert to numpy
            else:
                field_data.append(field)  # Already numpy or convertible
        
        # Create header if not provided
        if header is None:
            header = "x y " + " ".join(field_names)
        
        # Combine all data into a single array
        combined_data = np.column_stack([x, y] + field_data)
        
        # Save to file
        np.savetxt(filename, combined_data, header=header, comments='# ')
        
        print(f"Grid data successfully saved to {filename}")