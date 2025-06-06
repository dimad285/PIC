import cupy as cp
from src import Particles


update_density_direct_cylindrical = cp.RawKernel(r'''
extern "C" __global__
void update_density_direct_cylindrical(
    double *rho,                       // [num_grid_points]
    const double *R,                   // [2, max_particles]
    const int *part_type,             // [max_particles]
    const double *q_type,              // [num_types]
    const double np2c,                 // particles-to-cell ratio
    double dr, double dz,               // cell sizes
    int nr, int nz,                   // grid dimensions
    double two_pi,                     
    int last_alive)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= last_alive) return;

    // R[0] is r (radial), R[1] is z (axial)
    double r = R[2*pid] / dr;      // Normalized radial position
    double z = R[2*pid+1] / dz;    // Normalized axial position
    
    // Clamp positions to valid grid range
    r = max(0.0f, min(r, double(nr - 1)));
    z = max(0.0f, min(z, double(nz - 1)));
    
    // Get integer cell coordinates
    int r0 = int(floorf(r));
    int z0 = int(floorf(z));
    
    // Calculate neighbor coordinates with boundary conditions
    int r1 = min(r0 + 1, nr - 1);
    int z1 = min(z0 + 1, nz - 1);
    
    // Calculate interpolation weights
    double wr = r - double(r0);
    double wz = z - double(z0);
    double wr_1 = 1.0f - wr;
    double wz_1 = 1.0f - wz;
    
    // Calculate bilinear weights
    double w0 = wr_1 * wz_1;
    double w1 = wr * wz_1;
    double w2 = wr_1 * wz;
    double w3 = wr * wz;

    // Calculate grid indices
    int i0 = z0 * nr + r0;
    int i1 = z0 * nr + r1;
    int i2 = z1 * nr + r0;
    int i3 = z1 * nr + r1;

    // Apply charge to grid with volume weighting for cylindrical coordinates
    int type = part_type[pid];
    double charge = q_type[type] * np2c;
    
    // Add charge to grid nodes
    atomicAdd(&rho[i0], charge * w0);
    atomicAdd(&rho[i1], charge * w1);
    atomicAdd(&rho[i2], charge * w2);
    atomicAdd(&rho[i3], charge * w3);
}
''', 'update_density_direct_cylindrical')


normalize_density_by_volume = cp.RawKernel(r'''
extern "C" __global__
void normalize_density_by_volume(
    double *rho,
    int grid_n_r,
    int grid_n_z,
    double dr, double dz, double two_pi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_n_r * grid_n_z) return;

    int r_idx = idx / grid_n_z;  // fixed: r is the slow axis
    double volume;

    if (r_idx == 0) {
        // Axis cells: cylindrical volume with radius dr/2
        volume = (two_pi / 4) * dr * dr * dz;
    } else {
        // Regular cells: cylindrical shell
        double r = dr * r_idx;
        volume = (r + 0.5f * dr) * dr * dz * two_pi;
    }

    rho[idx] /= volume;
}
''', 'normalize_density_by_volume')


update_density_direct_cartesian = cp.RawKernel(r'''
extern "C" __global__
void update_density_direct_cartesian(
    double *rho,                       // [num_grid_points]
    const double *R,                   // [2, max_particles]
    const int *part_type,             // [max_particles]
    const double *q_type,              // [num_types]
    const double np2c,                 // particles-to-cell ratio
    double dx, double dy,               // cell sizes
    int m, int n,                     // grid dimensions
    int last_alive)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= last_alive) return;

    // Calculate grid position directly
    double x = R[2*pid] / dx;
    double y = R[2*pid+1] / dy;
    
    // Clamp positions to valid grid range
    x = max(0.0f, min(x, double(m - 1)));
    y = max(0.0f, min(y, double(n - 1)));
    
    // Get integer cell coordinates
    int x0 = int(floorf(x));
    int y0 = int(floorf(y));
    
    // Calculate neighbor coordinates
    int x1 = min(x0 + 1, m - 1);
    int y1 = min(y0 + 1, n - 1);
    
    // Calculate interpolation weights
    double wx = x - double(x0);
    double wy = y - double(y0);
    double wx_1 = 1.0f - wx;
    double wy_1 = 1.0f - wy;
    
    // Calculate bilinear weights
    double w0 = wx_1 * wy_1;
    double w1 = wx * wy_1;
    double w2 = wx_1 * wy;
    double w3 = wx * wy;

    // Calculate grid indices
    int i0 = y0 * m + x0;
    int i1 = y0 * m + x1;
    int i2 = y1 * m + x0;
    int i3 = y1 * m + x1;

    // Apply charge to grid
    int type = part_type[pid];
    double charge = q_type[type] * np2c;
    
    atomicAdd(&rho[i0], charge * w0);
    atomicAdd(&rho[i1], charge * w1);
    atomicAdd(&rho[i2], charge * w2);
    atomicAdd(&rho[i3], charge * w3);
}
''', 'update_density_direct_cartesian')


update_E_kenrel = cp.RawKernel(r'''
extern "C" __global__
void update_E(
    const double *phi,   // [nx * ny]
    double *E,           // [2, nx * ny] - output: E[0, :] = Ex, E[1, :] = Ey
    int nx,
    int ny,
    double dx,
    double dy)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny;
    if (idx >= size) return;

    int ix = idx % nx;
    int iy = idx / nx;

    double Ex = 0.0f;
    double Ey = 0.0f;

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

        self.rho = cp.zeros(m*n, dtype=cp.float64)
        self.rho_old = cp.zeros(m*n, dtype=cp.float64)
        self.rho_background = cp.zeros(m*n, dtype=cp.float64)
        self.b = cp.zeros(m*n, dtype=cp.float64) # right-hand side of the Poisson equation  ∆phi = b
        self.phi = cp.zeros(m*n, dtype=cp.float64)
        self.E = cp.zeros((2, m*n), dtype=cp.float64) 
        self.E_background = cp.zeros((2, m*n), dtype=cp.float64)
        self.J = cp.zeros((2, m*n), dtype=cp.float64) 
        self.A = cp.zeros((2, m*n), dtype=cp.float64)
        self.B = cp.zeros((2, m*n), dtype=cp.float64)
        self.B_background = cp.zeros((2, m*n), dtype=cp.float64)
        self.NGD = cp.ones(m*n, dtype=cp.float64)



    def update_E(self):

        nx, ny = self.gridshape
        dx, dy = self.cell_size

        threads_per_block = 256
        blocks_per_grid = (nx * ny + threads_per_block - 1) // threads_per_block

        update_E_kenrel(
            (blocks_per_grid,), (threads_per_block,),
            (self.phi, self.E, nx, ny, cp.float64(dx), cp.float64(dy))
        )
    

    def update_density(self, particles:Particles.Particles2D):
        last_alive = particles.last_alive
        block_size = 256
        grid_size = (last_alive + block_size - 1) // block_size
        self.rho.fill(0.0)

        if self.cylindrical:
            update_density_direct_cylindrical(
                (grid_size,), (block_size,),
                (
                    self.rho,
                    particles.R,
                    particles.part_type,
                    particles.q_type,
                    cp.float64(particles.np2c),
                    cp.float64(self.dy), cp.float64(self.dx),
                    cp.int32(self.n), cp.int32(self.m),
                    cp.float64(2*cp.pi),
                    cp.int32(particles.last_alive)
                )
            )
            
            # Apply normalization if needed
            normalize_density_by_volume(
                (grid_size,), (block_size,),
                (
                    self.rho,
                    cp.int32(self.n), cp.int32(self.m),
                    cp.float64(self.dy), cp.float64(self.dx),
                    cp.float64(2*cp.pi)
                )
            )

        else:
            update_density_direct_cartesian(
                (grid_size,), (block_size,),
                (
                    self.rho,
                    particles.R,
                    particles.part_type,
                    particles.q_type,
                    cp.float64(particles.np2c),
                    cp.float64(self.dx), cp.float64(self.dy),
                    cp.int32(self.m), cp.int32(self.n),
                    cp.int32(particles.last_alive)
                )
            )

            self.rho /= self.dx * self.dy

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
        np.savetxt(filename, combined_data, header=header, comments='# ', fmt='%.2e')
        
        print(f"Grid data successfully saved to {filename}")

