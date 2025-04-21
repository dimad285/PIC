import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from mpl_toolkits.mplot3d import Axes3D
from cupyx.profiler import benchmark
from cupyx.scipy.sparse import linalg
import cProfile
import pstats
import io
import Solvers



gauss_seidel_kernel = cp.RawKernel(r'''
extern "C" __global__
void gauss_seidel_red_black(
    double* phi,
    const double* rho,
    double h2,
    double omega,
    int nx,
    int ny,
    int is_red

) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny;
    if (idx >= size) return;
    int i = idx % nx;
    int j = idx / nx;

    // Only interior points (Dirichlet boundary)
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        if ((i + j) % 2 == is_red) {
            int idx_n = (j + 1) * nx + i;
            int idx_s = (j - 1) * nx + i;
            int idx_e = j * nx + (i + 1);
            int idx_w = j * nx + (i - 1);

            double update = 0.25 * (
                phi[idx_n] + phi[idx_s] +
                phi[idx_e] + phi[idx_w] -
                h2 * rho[idx]
            );
            phi[idx] = (1.0 - omega) * phi[idx] + omega * update;
        }
    }
}
''', 'gauss_seidel_red_black')





residual_kernel = cp.RawKernel(r"""
    // Residual Kernel
    extern "C" __global__ void 
    residual_kernel(
        const double* phi,
        const double* rho,
        double* res,
        double inv_h2,
        int nx,
        int ny
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int i = idx / nx;
        int j = idx % nx;
        
        if (i > 0 && i < ny-1 && j > 0 && j < nx-1) {
            int index = i * nx + j;
            double laplacian = (
                phi[index-nx] + phi[index+nx] + 
                phi[index-1] + phi[index+1] - 
                4.0 * phi[index]
            ) * inv_h2;
            res[index] = rho[index] - laplacian;
        }
        else if (i < ny && j < nx) {
            res[i * nx + j] = 0.0;
        }
    }
    """, 'residual_kernel')

restriction_kernel = cp.RawKernel(r'''
extern "C" __global__ void restriction_kernel(
    const double* fine_grid, 
    double* coarse_grid,
    int nx_fine, 
    int ny_fine, 
    int nx_coarse, 
    int ny_coarse
) {
    // Calculate coarse grid indices from thread and block IDs
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x-direction
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y-direction
    
    // Check if this thread is within coarse grid bounds
    if (i < ny_coarse && j < nx_coarse) {
        // Map to fine grid indices
        int i_fine = 2 * i;
        int j_fine = 2 * j;
        
        // Initialize sum and weight
        double weighted_sum = 0.0;
        double total_weight = 0.0;
        
        // Apply stencil weights to the 9 points
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int i_n = i_fine + di;
                int j_n = j_fine + dj;
                
                // Check if the neighbor is within bounds
                if (0 <= i_n && i_n < ny_fine && 0 <= j_n && j_n < nx_fine) {
                    // Determine weight based on position
                    double weight = 0.0;
                    if (di == 0 && dj == 0) {          // Center
                        weight = 0.25;
                    } else if (di == 0 || dj == 0) {   // Adjacent
                        weight = 0.125;
                    } else {                           // Diagonal
                        weight = 0.0625;
                    }
                    
                    weighted_sum += weight * fine_grid[i_n * nx_fine + j_n];
                    total_weight += weight;
                }
            }
        }
        
        // Normalize in case of boundary points
        if (total_weight > 0) {
            coarse_grid[i * nx_coarse + j] = weighted_sum / total_weight;
        }
    }
}
''', 'restriction_kernel')

prolongation_kernel = cp.RawKernel(r'''
    extern "C" __global__ 
    void prolong_kernel(
        const double* coarse,
        double* fine,
        int nx_coarse,
        int ny_coarse,
        int nx_fine) {
                                       
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int i_fine = idx / nx_fine;
        int j_fine = idx % nx_fine;
        
        if (i_fine < (ny_coarse-1)*2+1 && j_fine < (nx_coarse-1)*2+1) {
            int i_coarse = i_fine / 2;
            int j_coarse = j_fine / 2;
            
            bool is_i_odd = (i_fine % 2) != 0;
            bool is_j_odd = (j_fine % 2) != 0;
            
            if (!is_i_odd && !is_j_odd) {
                fine[i_fine * nx_fine + j_fine] = coarse[i_coarse * nx_coarse + j_coarse];
            }
            else if (is_i_odd && !is_j_odd) {
                if (i_coarse < ny_coarse - 1) {
                    fine[i_fine * nx_fine + j_fine] = 0.5 * (
                        coarse[i_coarse * nx_coarse + j_coarse] + 
                        coarse[(i_coarse+1) * nx_coarse + j_coarse]
                    );
                }
            }
            else if (!is_i_odd && is_j_odd) {
                if (j_coarse < nx_coarse - 1) {
                    fine[i_fine * nx_fine + j_fine] = 0.5 * (
                        coarse[i_coarse * nx_coarse + j_coarse] + 
                        coarse[i_coarse * nx_coarse + (j_coarse+1)]
                    );
                }
            }
            else {
                if (i_coarse < ny_coarse - 1 && j_coarse < nx_coarse - 1) {
                    fine[i_fine * nx_fine + j_fine] = 0.25 * (
                        coarse[i_coarse * nx_coarse + j_coarse] + 
                        coarse[(i_coarse+1) * nx_coarse + j_coarse] + 
                        coarse[i_coarse * nx_coarse + (j_coarse+1)] + 
                        coarse[(i_coarse+1) * nx_coarse + (j_coarse+1)]
                    );
                }
            }
        }
    }
    ''', 'prolong_kernel')

euclidean_norm_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void euclidean_norm(
        const double* vec,
        double* norm,
        int size
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        double sum = 0.0;
        for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
            sum += vec[i] * vec[i];
        }
        atomicAdd(norm, sum);
    }
''', 'euclidean_norm')


class MultigridSolver:
    def __init__(self, nx, ny, h, levels=4, omega=2/3):
        self.nx = nx
        self.ny = ny
        self.h = h
        self.h2 = h * h
        self.levels = levels
        self.omega = omega

        self.threads_per_block = 256
        self.blocks = (nx * ny + self.threads_per_block - 1) // self.threads_per_block


    def smooth(self, phi, rho, iterations=10):
        for _ in range(iterations):
            # Red
            gauss_seidel_kernel(
                (self.blocks,), (self.threads_per_block,),
                (phi, rho, cp.float64(self.h2), cp.float64(self.omega),
                 cp.int32(self.nx), cp.int32(self.ny), cp.int32(1))
            )
            # Black
            gauss_seidel_kernel(
                (self.blocks,), (self.threads_per_block,),
                (phi, rho, cp.float64(self.h2), cp.float64(self.omega),
                 cp.int32(self.nx), cp.int32(self.ny), cp.int32(0))
            )


    def residual(self, phi, rho):
        res = cp.zeros_like(phi)
        block_size = 256
        grid_size = (self.nx * self.ny + block_size - 1) // block_size
        inv_h2 = 1.0 / self.h2
        residual_kernel((grid_size,), (block_size,),
                        (phi, rho, res, cp.float64(inv_h2),
                         cp.int32(self.nx), cp.int32(self.ny)))
        return res

    def restrict(self, fine):
        nx_coarse = (self.nx + 1) // 2
        ny_coarse = (self.ny + 1) // 2
        coarse = cp.zeros(nx_coarse * ny_coarse, dtype=fine.dtype)

        block_size = (16, 16)
        grid_size = (
            (nx_coarse + block_size[0] - 1) // block_size[0],
            (ny_coarse + block_size[1] - 1) // block_size[1]
        )

        restriction_kernel(
            grid_size,
            block_size,
            (fine, coarse, self.nx, self.ny, nx_coarse, ny_coarse)
        )
        return coarse

    def prolong(self, coarse):
        nx_coarse = (self.nx + 1) // 2
        ny_coarse = (self.ny + 1) // 2
        nx_fine = 2 * (nx_coarse - 1) + 1
        ny_fine = 2 * (ny_coarse - 1) + 1
        fine = cp.zeros((ny_fine, nx_fine), dtype=coarse.dtype).reshape(-1)

        block_size = 256
        grid_size = (nx_fine * ny_fine + block_size - 1) // block_size

        prolongation_kernel((grid_size,), (block_size,),
                            (coarse, fine, cp.int32(nx_coarse), cp.int32(ny_coarse), cp.int32(nx_fine)))
        cp.cuda.Stream.null.synchronize()
        return fine

    def v_cycle(self, phi, rho, levels=None, max_iter=5):
        if levels is None:
            levels = self.levels
        N = self.nx

        if levels == 1:
            self.smooth(phi, rho, iterations=20)
            return phi

        self.smooth(phi, rho, iterations=max_iter//2)
        res = self.residual(phi, rho)
        res_coarse = self.restrict(res)
        phi_coarse = cp.zeros_like(res_coarse)

        coarse_solver = MultigridSolver((N+1)//2, (N+1)//2, self.h * 2, levels=levels - 1, omega=self.omega)
        phi_coarse = coarse_solver.v_cycle(phi_coarse, res_coarse, levels=levels - 1, max_iter=max_iter)

        phi += self.prolong(phi_coarse)
        self.smooth(phi, rho, iterations=max_iter)
        return phi
    
    def norm(self, vec):
        norm = cp.zeros(1, dtype=cp.float64)

        euclidean_norm_kernel(
            (self.blocks,), (self.threads_per_block,),
            (vec, norm, cp.int32(vec.size))
        )
        return cp.sqrt(norm)

    def solve(self, phi, rho, tol=1e-5, max_iter=5, max_cycles=50):
        phi.fill(0)
        rho_norm = self.norm(rho)
        for _ in range(max_cycles):
            res = self.residual(phi, rho)
            rel_res = self.norm(res) / rho_norm
            if rel_res < tol:
                break
            phi = self.v_cycle(phi, rho, levels=self.levels, max_iter=max_iter)
        return phi


# === MAIN FUNCTION ===

def plot_3d_surface(phi_flat, nx, ny, title="Potential Field", cmap="viridis"):
    """
    Plot a 3D surface of the scalar field stored in `phi_flat`.

    Parameters:
        phi_flat (np.ndarray or cp.ndarray): Flattened scalar field (shape: nx * ny)
        nx (int): Number of grid points in x-direction
        ny (int): Number of grid points in y-direction
        title (str): Plot title
        cmap (str): Matplotlib colormap
    """
    if hasattr(phi_flat, 'get'):  # Check for CuPy array
        phi_flat = phi_flat.get()

    phi = phi_flat.reshape((ny, nx))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, phi, cmap=cmap, edgecolor='k', linewidth=0.3, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('phi')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


def main(): 
    N = 129
    L = 1.0
    h = L / (N - 1)
    levels = 6

    x = cp.linspace(0, L, N, dtype=cp.float64)
    y = cp.linspace(0, L, N, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y, indexing='ij')

    # Source term
    rho = cp.sin(16*cp.pi * X) * cp.sin(16*cp.pi * Y)
    rho = rho.ravel().astype(cp.float64)
    phi = cp.zeros_like(rho)

    solver = MultigridSolver(N, N, h, levels=levels, omega=2/3)
    #phi = solver.solve(phi, -rho, tol=1e-5, max_iter=5, max_cycles=100)
    #plot_3d_surface(phi, N, N)

    # Benchmark the solver
    print(benchmark(solver.solve, (phi, -rho, 1e-5, 5, 100), n_repeat=10))
    #print(benchmark(linalg.gmres, (Lap, -rho, phi, 1e-5), n_repeat=10))


# === PROFILING ENTRY POINT ===

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    pr.dump_stats("profile.prof")
    with open("profile.txt", "w") as f:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        f.write(s.getvalue())