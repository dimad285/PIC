import cupy as cp
import Particles


update_density_cylindrical = cp.RawKernel(r'''
extern "C" __global__
void update_density_cylindrical(
    float *rho,                         // [num_grid_points]
    const float *weights,              // [4, max_particles]
    const int *indices,                // [4, max_particles]
    const int *part_type,              // [max_particles]
    const float *q_type,               // [num_types]
    const float np2c,             // [max_particles]
    const float np2c,             // [max_particles]
    int last_alive,
    int max_particles,
    int grid_n_r,                      // radial dimension of grid (used for modulo)
    float dr, float dz, float two_pi)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= last_alive) return;

    int type = part_type[pid];
    float charge = q_type[type] * np2c;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = indices[i * max_particles + pid];     // flattened grid index
        int r_idx = idx % grid_n_r;                  // radial index

        float r = dr * r_idx;
        if (r < 1e-6f) r = dr;  // avoid div by zero at r=0

        float volume = (r + dr * 0.5f) * dr * dz * two_pi;
        float w = weights[i * max_particles + pid] / volume;

        atomicAdd(&rho[idx], charge * w);
    }
}
''', 'update_density_cylindrical')

update_density_cartesian = cp.RawKernel(r'''
extern "C" __global__
void update_density_cartesian(
    float *rho,                         // [num_grid_points]
    const float *weights,              // [4, max_particles]
    const int *indices,                // [4, max_particles]
    const int *part_type,              // [max_particles]
    const float *q_type,               // [num_types]
    const float np2c,             // [max_particles]
    int last_alive,
    int max_particles)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= last_alive) return;

    int type = part_type[pid];
    float charge = q_type[type] * np2c;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = indices[i * max_particles + pid];
        float w = weights[i * max_particles + pid];

        atomicAdd(&rho[idx], charge * w);
    }
}
''', 'update_density_cartesian')



update_E_kenrel = cp.RawKernel(r'''
extern "C" __global__
void update_E(
    const float *phi,   // [nx * ny]
    float *E,           // [2, nx * ny] - output: E[0, :] = Ex, E[1, :] = Ey
    int nx,
    int ny,
    float dx,
    float dy)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny;
    if (idx >= size) return;

    int ix = idx % nx;
    int iy = idx / nx;

    float Ex = 0.0f;
    float Ey = 0.0f;

    // Central differences, with forward/backward near edges
    if (ix > 0 && ix < nx - 1)
        Ex = -(phi[iy * nx + (ix + 1)] - phi[iy * nx + (ix - 1)]) / (2.0f * dx);
    else if (ix == 0)
        Ex = -(phi[iy * nx + (ix + 1)] - phi[iy * nx + ix]) / dx;
    else if (ix == nx - 1)
        Ex = -(phi[iy * nx + ix] - phi[iy * nx + (ix - 1)]) / dx;

    if (iy > 0 && iy < ny - 1)
        Ey = -(phi[(iy + 1) * nx + ix] - phi[(iy - 1) * nx + ix]) / (2.0f * dy);
    else if (iy == 0)
        Ey = -(phi[(iy + 1) * nx + ix] - phi[iy * nx + ix]) / dy;
    else if (iy == ny - 1)
        Ey = -(phi[iy * nx + ix] - phi[(iy - 1) * nx + ix]) / dy;
                               

    E[0 * size + idx] = Ex;
    E[1 * size + idx] = Ey;
}
''', 'update_E')

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

        self.rho = cp.zeros(m*n, dtype=cp.float32)
        self.wall_rho = cp.zeros(m*n, dtype=cp.float32)
        self.b = cp.zeros(m*n, dtype=cp.float32) # righr-hand side of the Poisson equation
        self.phi = cp.zeros(m*n, dtype=cp.float32)
        self.E = cp.zeros((2, m*n), dtype=cp.float32)
        self.J = cp.zeros((2, m*n), dtype=cp.float32)
        self.A = cp.zeros((2, m*n), dtype=cp.float32)
        self.B = cp.zeros((2, m*n), dtype=cp.float32)
        self.NGD = cp.ones(m*n, dtype=cp.float32)


    '''
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
    '''

    def update_E(self):

        nx, ny = self.gridshape
        dx, dy = self.cell_size

        threads_per_block = 256
        blocks_per_grid = (nx * ny + threads_per_block - 1) // threads_per_block

        update_E_kenrel(
            (blocks_per_grid,), (threads_per_block,),
            (self.phi, self.E, nx, ny, cp.float32(dx), cp.float32(dy))
        )
    

    def update_density(self, particles:Particles.Particles2D):
        last_alive = particles.last_alive
        block_size = 256
        grid_size = (last_alive + block_size - 1) // block_size
        self.rho.fill(0.0)

        if self.cylindrical:
            update_density_cylindrical(
                (grid_size,), (block_size,),
                (
                    self.rho,
                    particles.weights,
                    particles.indices,
                    particles.part_type,
                    particles.q_type,
                    cp.float32(particles.np2c),
                    particles.last_alive,
                    particles.N,
                    self.gridshape[0],
                    self.cell_size[0],
                    self.cell_size[1],
                    2 * cp.pi
                )
            )
        else:

            update_density_cartesian(
                (grid_size,), (block_size,),
                (
                    self.rho,
                    particles.weights,
                    particles.indices,
                    particles.part_type,
                    particles.q_type,
                    cp.float32(particles.np2c),
                    particles.last_alive,
                    particles.N
                )
            )

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

