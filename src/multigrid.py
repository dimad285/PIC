import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark
import cProfile
import pstats
import io
from . import Boundaries
from . import Grid



gauss_seidel_kernel = cp.RawKernel(r'''
extern "C" __global__
void gauss_seidel_red_black(
    double* phi,
    const double* rho,
    const int* bc_mask,
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
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && bc_mask[idx] == 0) {
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
        const int* bc_mask,
        double inv_h2,
        int nx,
        int ny
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int i = idx / nx;
        int j = idx % nx;
        
        if (i > 0 && i < ny-1 && j > 0 && j < nx-1 && bc_mask[idx] == 0) {
            int index = i * nx + j;
            double laplacian = (
                phi[index-nx] + phi[index+nx] + 
                phi[index-1] + phi[index+1] - 
                4.0 * phi[index]
            ) * inv_h2;
            res[index] = rho[index] - laplacian;
        }
        else if (i < ny && j < nx || bc_mask[idx] == 1) {
            res[i * nx + j] = 0.0;
        }
    }
    """, 'residual_kernel')


restriction_kernel = cp.RawKernel(r'''
extern "C" __global__ void restriction_kernel(
    const double* fine_grid, 
    double* coarse_grid,
    const int* bc_mask,             // 1 = interior, 0 = boundary
    int nx_fine, 
    int ny_fine, 
    int nx_coarse, 
    int ny_coarse
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x-direction (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y-direction (row)

    if (i < ny_coarse && j < nx_coarse) {
        int i_fine = 2 * i;
        int j_fine = 2 * j;

        double weighted_sum = 0.0;
        double total_weight = 0.0;

        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int i_n = i_fine + di;
                int j_n = j_fine + dj;

                // Check fine-grid bounds
                if (i_n >= 0 && i_n < ny_fine && j_n >= 0 && j_n < nx_fine) {
                    int idx = i_n * nx_fine + j_n;

                    // 9-point stencil weights (full weighting)
                    double weight = (di == 0 && dj == 0) ? 0.25 :
                                    (di == 0 || dj == 0) ? 0.125 : 0.0625;
                    weighted_sum += weight * fine_grid[idx];
                    total_weight += weight;
                    //printf("i: %d, j: %d, i_n: %d, j_n: %d, weight: %f, fine_grid[%d]: %f\n", i, j, i_n, j_n, weight, idx, fine_grid[idx]);
                    
                }
                // else: treat out-of-bounds as zero (nothing added)
            }
        }

        // Normalize if total weight is less than 1 (e.g. near boundaries)
        if (total_weight > 0.0){
            //printf("total_weight: %f\n", total_weight);
            coarse_grid[i * nx_coarse + j] = weighted_sum / total_weight;
        }
        else
            coarse_grid[i * nx_coarse + j] = 0.0;  // or handle however you want
    }
}
''', 'restriction_kernel')



prolongation_kernel = cp.RawKernel(r'''
    extern "C" __global__ 
__global__ void prolong_kernel(
    double* fine,
    const double* coarse,
    const int* bc_mask_fine,  // 1 = valid interior, 0 = boundary
    int nx_coarse,
    int ny_coarse,
    int nx_fine
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i_fine = idx / nx_fine;
    int j_fine = idx % nx_fine;

    int fine_idx = i_fine * nx_fine + j_fine;

    // Skip if it's a boundary point on the fine grid
    if (bc_mask_fine[fine_idx] == 0) return;

    if (i_fine < (ny_coarse - 1) * 2 + 1 && j_fine < (nx_coarse - 1) * 2 + 1) {
        int i_coarse = i_fine / 2;
        int j_coarse = j_fine / 2;

        int coarse_idx = i_coarse * nx_coarse + j_coarse;

        bool is_i_odd = (i_fine % 2 != 0);
        bool is_j_odd = (j_fine % 2 != 0);

        double value = 0.0;

        if (!is_i_odd && !is_j_odd) {
            // Direct injection
            value = coarse[coarse_idx];
        } else if (is_i_odd && !is_j_odd) {
            // Vertical interpolation
            if (i_coarse + 1 < ny_coarse) {
                value = 0.5 * (coarse[coarse_idx] + coarse[(i_coarse + 1) * nx_coarse + j_coarse]);
            }
        } else if (!is_i_odd && is_j_odd) {
            // Horizontal interpolation
            if (j_coarse + 1 < nx_coarse) {
                value = 0.5 * (coarse[coarse_idx] + coarse[i_coarse * nx_coarse + (j_coarse + 1)]);
            }
        } else {
            // Diagonal interpolation
            if (i_coarse + 1 < ny_coarse && j_coarse + 1 < nx_coarse) {
                value = 0.25 * (
                    coarse[coarse_idx] +
                    coarse[(i_coarse + 1) * nx_coarse + j_coarse] +
                    coarse[i_coarse * nx_coarse + (j_coarse + 1)] +
                    coarse[(i_coarse + 1) * nx_coarse + (j_coarse + 1)]
                );
            }
        }

        fine[fine_idx] += value;
    }
}
    ''', 'prolong_kernel')

norm_kernel = cp.RawKernel(r'''
#include <cuda_runtime.h>

extern "C" __global__
void norm_kernel(
    const double* vec,
    double* norm,
    int size
) {
    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory for block-wise reduction
    __shared__ double block_sum;
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        block_sum = 0.0;
    }
    __syncthreads();
    
    // Each thread computes its element's contribution
    if (idx < size) {
        double val = vec[idx];
        atomicAdd(&block_sum, val * val);
    }
    
    // Ensure all threads in block have updated shared memory
    __syncthreads();
    
    // Only one thread per block updates the global sum
    if (threadIdx.x == 0) {
        atomicAdd(norm, block_sum);
    }
}
''', 'norm_kernel')

bc_coarsen_kernel = cp.RawKernel(r'''
extern "C" __global__
__global__ void coarsen_bc_mask(
    const int* bc_mask_fine,
    int* bc_mask_coarse,
    int nx_fine,
    int ny_fine,
    int nx_coarse,
    int ny_coarse
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x-direction
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y-direction

    if (i < ny_coarse && j < nx_coarse) {
        int i_fine = 2 * i;
        int j_fine = 2 * j;

        // Mark as interior unless a fine neighbor is boundary
        int is_interior = 1;

        // Check all 3x3 neighbors (can reduce to 2x2 if you're using a 5-point stencil)
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                int i_n = i_fine + di;
                int j_n = j_fine + dj;
                if (i_n >= 0 && i_n < ny_fine && j_n >= 0 && j_n < nx_fine) {
                    if (bc_mask_fine[i_n * nx_fine + j_n] == 0) {
                        is_interior = 0;
                    }
                } else {
                    // Out-of-bounds is treated as boundary
                    is_interior = 0;
                }
            }
        }

        bc_mask_coarse[i * nx_coarse + j] = is_interior;
    }
}
''', 'coarsen_bc_mask')


class MultigridSolver:
    def __init__(self, bc_mask, nx, ny, h, levels=4, omega=2/3):
        self.nx = nx
        self.ny = ny
        self.h = h
        self.h2 = h * h
        self.levels = levels
        self.omega = omega

        self.res0 = cp.zeros((nx, ny), dtype=cp.float64)
        self.phi_levels = []
        self.rhs_levels = []
        self.res_levels = []
        self.bc_levels = []

        self.norm_out = cp.zeros(1, dtype=cp.float64)

        for level in range(levels):
            n = self._get_n(level)
            self.phi_levels.append(cp.zeros((n * n), dtype=cp.float64))
            self.rhs_levels.append(cp.zeros((n * n), dtype=cp.float64))
            self.res_levels.append(cp.zeros((n * n), dtype=cp.float64))
            self.bc_levels.append(cp.zeros((n * n), dtype=cp.int32))

        self.bc_levels[0] = cp.asarray(bc_mask, dtype=cp.int32)

        for level in range(1, levels):
            n_fine = self._get_n(level - 1)
            n_coarse = self._get_n(level)

            # Coarsen the boundary condition mask
            self.bc_levels[level] = cp.zeros((n_coarse, n_coarse), dtype=cp.int32)
            self._coarsen_bc_mask(self.bc_levels[level - 1], self.bc_levels[level], n_fine, n_fine, n_coarse, n_coarse)

        


    def smooth(self, phi, rho, bc_mask, nx, ny, h, iterations=10):

        h2 = h * h

        block_size = 256
        grid_size = (nx * ny + block_size - 1) // block_size

        for i in range(iterations):
            # Red
            gauss_seidel_kernel(
                (grid_size,), (block_size,),
                (phi, rho, bc_mask, cp.float64(h2), cp.float64(self.omega),
                 cp.int32(nx), cp.int32(ny), cp.int32(1))
            )
            # Black
            gauss_seidel_kernel(
                (grid_size,), (block_size,),
                (phi, rho, bc_mask, cp.float64(h2), cp.float64(self.omega),
                 cp.int32(nx), cp.int32(ny), cp.int32(0))
            )



    def residual(self, phi, rho, res, bc_mask, nx, ny, h):

        block_size = 256
        grid_size = (nx * ny + block_size - 1) // block_size
        inv_h2 = 1.0 / (h * h)

        residual_kernel((grid_size,), (block_size,),
                        (phi, rho, res, bc_mask, cp.float64(inv_h2),
                         cp.int32(nx), cp.int32(ny)))


    def restrict(self, fine, coarse, bc_mask, nx, ny):
        nx_coarse = (nx + 1) // 2
        ny_coarse = (ny + 1) // 2

        block_size = (16, 16)
        grid_size = (
            (nx_coarse + block_size[0] - 1) // block_size[0],
            (ny_coarse + block_size[1] - 1) // block_size[1]
        )

        restriction_kernel(
            grid_size,
            block_size,
            (fine, coarse, bc_mask, cp.int32(nx), cp.int32(ny), cp.int32(nx_coarse), cp.int32(ny_coarse))
        )


    def prolong(self, fine, coarse, bc_mask, nx_coarse, ny_coarse):

        nx_fine = 2 * (nx_coarse - 1) + 1
        ny_fine = 2 * (ny_coarse - 1) + 1

        block_size = 256
        grid_size = (nx_fine * ny_fine + block_size - 1) // block_size

        prolongation_kernel((grid_size,), (block_size,),
                            (fine, coarse, bc_mask, cp.int32(nx_coarse), cp.int32(ny_coarse), cp.int32(nx_fine)))
        

    
    def v_cycle(self, phi, rho, levels=None, smooth_iter=5, solve_iter=20):

        self.phi_levels[0] = phi
        self.rhs_levels[0] = rho
        #self.res_levels[0].fill(0.0)

        n = self.nx
        h = self.h

        for level in range(levels - 1):
            
            self.smooth(self.phi_levels[level], self.rhs_levels[level], self.bc_levels[level], n, n, h, iterations=smooth_iter)
            self.residual(self.phi_levels[level], self.rhs_levels[level], self.res_levels[level], self.bc_levels[level], n, n, h)
            self.restrict(self.res_levels[level], self.rhs_levels[level + 1], self.bc_levels[level], n, n)

            self.phi_levels[level + 1].fill(0.0)
            #self.res_levels[level + 1].fill(0.0)

            n = (n + 1) // 2
            h = h * 2


        self.smooth(self.phi_levels[levels - 1], self.rhs_levels[levels - 1], self.bc_levels[levels-1], n, n, h, iterations=solve_iter)

        h = h / 2
        n = n * 2 - 1

        #return self.phi_levels[-1]
    
        for level in reversed(range(levels - 1)):
            
            n_coarse = (n + 1) // 2
            self.prolong(self.phi_levels[level], self.phi_levels[level+1], self.bc_levels[level], n_coarse, n_coarse)
            self.smooth(self.phi_levels[level], self.rhs_levels[level], self.bc_levels[level], n, n, h, iterations=smooth_iter)
            n = n * 2 - 1
            h = h / 2

        return self.phi_levels[0]
        
    def _get_n(self, level):

        return (self.nx + 2**level - 1) // 2 ** level
    
    def _norm(self, vec):
        block_size = 256
        self.norm_out[0] = 0.0
        grid_size = (vec.size + block_size - 1) // block_size  
        norm_kernel(
            (grid_size,), (block_size,),
            (vec, self.norm_out, cp.int32(vec.size))
        )

        return cp.sqrt(self.norm_out[0])
    
    def _coarsen_bc_mask(self, bc_mask_fine, bc_mask_coarse, nx_fine, ny_fine, nx_coarse, ny_coarse):
        block_size = (16, 16)
        grid_size = (
            (nx_coarse + block_size[0] - 1) // block_size[0],
            (ny_coarse + block_size[1] - 1) // block_size[1]
        )

        bc_coarsen_kernel(
            grid_size,
            block_size,
            (bc_mask_fine, bc_mask_coarse, cp.int32(nx_fine), cp.int32(ny_fine), cp.int32(nx_coarse), cp.int32(ny_coarse))
        )

    def solve(self, phi, rho, tol=1e-5, smooth_iter=5, solve_iter=20, max_cycles=50):
        #phi.fill(0.0)
        rho_norm = self._norm(rho)
        for _ in range(max_cycles):
            self.residual(phi, rho, self.res0, self.bc_levels[0], self.nx, self.ny, self.h)
            rel_res = self._norm(self.res0) / rho_norm
            if rel_res < tol:
                break
            phi = self.v_cycle(phi, rho, levels=self.levels, smooth_iter=smooth_iter, solve_iter=solve_iter)

        return phi, (_, rel_res.item())

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
    levels = 5

    x = cp.linspace(0, L, N, dtype=cp.float64)
    y = cp.linspace(0, L, N, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y, indexing='ij')

    # Source term
    k = 1.0
    p = 1
    rho = cp.sin(p*cp.pi * X) * cp.sin(p*cp.pi * Y)*k
    rho = rho.ravel().astype(cp.float64)
    phi = cp.zeros_like(rho)

    grid = Grid.Grid2D(N, N, h, h)
    cathode = ((N//4, N//4, N//4, N*3//4), -50)
    anode = ((N*3//4, N//4, N*3//4, N*3//4), 50)
    bound = Boundaries.Boundaries((cathode, anode), grid)
    res = cp.zeros_like(rho)

    phi[bound.conditions[0]] = bound.conditions[1]
    zeros = cp.zeros_like(phi)
    ones = cp.ones_like(phi)


    solver = MultigridSolver(bound.bc_lookup, N, N, h, levels=levels, omega=1.93)
    phi, info = solver.solve(phi, -rho, tol=1e-5, smooth_iter=10, solve_iter=30, max_cycles=100)
    #print(info)


    #phi = solver.v_cycle(phi, -rho, levels=levels, smooth_iter=3, solve_iter=30)
    phi[bound.conditions[0]] = bound.conditions[1]
    plot_3d_surface(phi, N, N)

    # Benchmark the solver
    #print(benchmark(solver.solve, (phi, -rho, 1e-5, 5, 20, 100), n_repeat=100))
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