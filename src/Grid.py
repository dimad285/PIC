import cupy as cp
import Particles



class Grid2D():
    def __init__(self, m, n, X, Y, cylindrical=False):
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
        self.b = cp.zeros(m*n) # righr-hand side of the Poisson equation
        self.phi = cp.zeros(m*n)
        self.E = cp.zeros((2, m*n))
        self.J = cp.zeros((2, m*n))
        self.A = cp.zeros((2, m*n))
        self.B = cp.zeros((2, m*n))

    def update_E(self):
        # Reshape phi into a 2D grid using Fortran-order for better y-direction performance
        phi_grid = cp.reshape(self.phi, self.gridshape, order='C')  # Use Fortran-order ('F') for column-major memory layout
        
        # Compute gradients in the x and y directions
        E_x = -cp.gradient(phi_grid, self.dx, axis=1)  # Gradient in the x-direction (axis 1)
        E_y = -cp.gradient(phi_grid, self.dy, axis=0)  # Gradient in the y-direction (axis 0)
        
        # Flatten the results and assign to E
        self.E[0, :] = E_x.flatten()  # Flatten using the same Fortran-order to match the reshaping
        self.E[1, :] = E_y.flatten()

    def update_density(self, particles:Particles.Particles2D):
        """
        Updates charge density field using precomputed bilinear interpolation.
        """
        self.rho.fill(0.0)
        l_a = particles.last_alive
        
        cp.add.at(self.rho, particles.indices[0, :l_a], particles.q_type[particles.part_type[:l_a]] * particles.weights[0, :l_a])
        cp.add.at(self.rho, particles.indices[1, :l_a], particles.q_type[particles.part_type[:l_a]] * particles.weights[1, :l_a])
        cp.add.at(self.rho, particles.indices[2, :l_a], particles.q_type[particles.part_type[:l_a]] * particles.weights[2, :l_a])
        cp.add.at(self.rho, particles.indices[3, :l_a], particles.q_type[particles.part_type[:l_a]] * particles.weights[3, :l_a])


    def update_J(self, particles:Particles.Particles2D):
        pass