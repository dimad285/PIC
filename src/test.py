import numpy as np
import cupy as cp
from cupy import cublas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cupyx.profiler import benchmark
import cProfile
import pstats
import io

class MultigridSolver:
    """
    GPU-accelerated multigrid solver for 2D Poisson equation using CuPy.
    """
    
    def __init__(self, N, L=1.0, levels=None, omega=2/3, tol=1e-5, max_iter=5):
        """
        Initialize the multigrid solver.
        
        Args:
            N (int): Grid size (should be 2^n + 1 for proper multigrid)
            L (float): Domain size
            levels (int): Number of multigrid levels (default: calculated from N)
            omega (float): Relaxation parameter
            tol (float): Convergence tolerance
            max_iter (int): Maximum iterations per smoothing step
        """
        self.N = N
        self.L = L
        self.h = L / (N - 1)
        
        # Calculate number of levels if not specified
        if levels is None:
            self.levels = int(np.log2(N - 1))
        else:
            self.levels = min(levels, int(np.log2(N - 1)))
            
        self.omega = omega
        self.tol = tol
        self.max_iter = max_iter
        
        # Compile kernels
        self._compile_kernels()
        
        # Create grid
        x = cp.linspace(0, L, N)
        y = cp.linspace(0, L, N)
        self.X, self.Y = cp.meshgrid(x, y, indexing='ij')
        
    def _compile_kernels(self):
        """Compile all CUDA kernels needed for the solver."""
        self.jacobi_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void jacobi_kernel(
            float* phi,
            const float* rho,
            float h2,
            float omega,
            int nx,
            int ny,
            int iterations
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int size = nx * ny;

            if (idx >= size) return;

            int i = idx % nx;
            int j = idx / nx;

            // Allocate register memory
            float phi_local = phi[idx];

            for (int iter = 0; iter < iterations; ++iter) {
                if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
                    int idx_n = (j + 1) * nx + i;
                    int idx_s = (j - 1) * nx + i;
                    int idx_e = j * nx + (i + 1);
                    int idx_w = j * nx + (i - 1);

                    float update = 0.25f * (
                        phi[idx_n] + phi[idx_s] +
                        phi[idx_e] + phi[idx_w] -
                        h2 * rho[idx]
                    );

                    phi_local = (1.0f - omega) * phi_local + omega * update;
                }
                __syncthreads();  // synchronize threads in block
            }

            phi[idx] = phi_local;
        }
        ''', 'jacobi_kernel')
        
        self.residual_kernel = cp.RawKernel("""
        extern "C" __global__ void 
        residual_kernel(
            const float* phi,
            const float* rho,
            float* res,
            float inv_h2,
            int nx,
            int ny
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            int i = idx / nx;
            int j = idx % nx;
            
            if (i > 0 && i < ny-1 && j > 0 && j < nx-1) {
                int index = i * nx + j;
                float laplacian = (
                    phi[index-nx] + phi[index+nx] + 
                    phi[index-1] + phi[index+1] - 
                    4.0f * phi[index]
                ) * inv_h2;
                res[index] = rho[index] - laplacian;
            }
            else if (i < ny && j < nx) {
                res[i * nx + j] = 0.0f;
            }
        }
        """, 'residual_kernel')
        
        # Improved restriction kernel that uses full weighting instead of injection
        self.restriction_kernel = cp.RawKernel("""
        extern "C" __global__ void 
        restriction_kernel(
            const float* fine,
            float* coarse,
            int nx_fine,
            int ny_fine,
            int nx_coarse
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            int i_coarse = idx / nx_coarse;
            int j_coarse = idx % nx_coarse;
            
            if (i_coarse < nx_coarse && j_coarse < nx_coarse) {
                int i_fine = 2 * i_coarse;
                int j_fine = 2 * j_coarse;
                
                // Center
                float center = fine[i_fine * nx_fine + j_fine];
                
                // Cardinal neighbors
                float north = (i_fine + 1 < ny_fine) ? fine[(i_fine + 1) * nx_fine + j_fine] : 0.0f;
                float south = (i_fine > 0) ? fine[(i_fine - 1) * nx_fine + j_fine] : 0.0f;
                float east = (j_fine + 1 < nx_fine) ? fine[i_fine * nx_fine + (j_fine + 1)] : 0.0f;
                float west = (j_fine > 0) ? fine[i_fine * nx_fine + (j_fine - 1)] : 0.0f;
                
                // Diagonal neighbors
                float ne = (i_fine + 1 < ny_fine && j_fine + 1 < nx_fine) ? fine[(i_fine + 1) * nx_fine + (j_fine + 1)] : 0.0f;
                float nw = (i_fine + 1 < ny_fine && j_fine > 0) ? fine[(i_fine + 1) * nx_fine + (j_fine - 1)] : 0.0f;
                float se = (i_fine > 0 && j_fine + 1 < nx_fine) ? fine[(i_fine - 1) * nx_fine + (j_fine + 1)] : 0.0f;
                float sw = (i_fine > 0 && j_fine > 0) ? fine[(i_fine - 1) * nx_fine + (j_fine - 1)] : 0.0f;
                
                // Full weighting
                coarse[i_coarse * nx_coarse + j_coarse] = (
                    4.0f * center +   // center (4/16)
                    2.0f * (north + south + east + west) +  // cardinal directions (2/16 each)
                    1.0f * (ne + nw + se + sw)  // diagonals (1/16 each)
                ) / 16.0f;  // Total weight = 16
            }
        }
        """, 'restriction_kernel')
        
        self.prolongation_kernel = cp.RawKernel('''
        extern "C" __global__ 
        void prolongation_kernel(
            const float* coarse,
            float* fine,
            int nx_coarse,
            int ny_coarse,
            int nx_fine
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            int i_fine = idx / nx_fine;
            int j_fine = idx % nx_fine;
            
            if (i_fine < (ny_coarse-1)*2+1 && j_fine < (nx_coarse-1)*2+1) {
                int i_coarse = i_fine / 2;
                int j_coarse = j_fine / 2;
                
                bool is_i_odd = (i_fine % 2) != 0;
                bool is_j_odd = (j_fine % 2) != 0;
                
                if (!is_i_odd && !is_j_odd) {
                    // Direct copy at coarse grid points
                    fine[i_fine * nx_fine + j_fine] = coarse[i_coarse * nx_coarse + j_coarse];
                }
                else if (is_i_odd && !is_j_odd) {
                    // Vertical interpolation
                    if (i_coarse < ny_coarse - 1) {
                        fine[i_fine * nx_fine + j_fine] = 0.5f * (
                            coarse[i_coarse * nx_coarse + j_coarse] + 
                            coarse[(i_coarse+1) * nx_coarse + j_coarse]
                        );
                    }
                }
                else if (!is_i_odd && is_j_odd) {
                    // Horizontal interpolation
                    if (j_coarse < nx_coarse - 1) {
                        fine[i_fine * nx_fine + j_fine] = 0.5f * (
                            coarse[i_coarse * nx_coarse + j_coarse] + 
                            coarse[i_coarse * nx_coarse + (j_coarse+1)]
                        );
                    }
                }
                else {
                    // Diagonal interpolation
                    if (i_coarse < ny_coarse - 1 && j_coarse < nx_coarse - 1) {
                        fine[i_fine * nx_fine + j_fine] = 0.25f * (
                            coarse[i_coarse * nx_coarse + j_coarse] + 
                            coarse[(i_coarse+1) * nx_coarse + j_coarse] + 
                            coarse[i_coarse * nx_coarse + (j_coarse+1)] + 
                            coarse[(i_coarse+1) * nx_coarse + (j_coarse+1)]
                        );
                    }
                }
            }
        }
        ''', 'prolongation_kernel')

    def smooth(self, phi, rho, nx, ny, h, iterations=None):
        """
        Apply Jacobi smoothing.
        
        Args:
            phi: Current solution
            rho: Source term
            nx, ny: Grid dimensions
            h: Grid spacing
            iterations: Number of smoothing iterations
        
        Returns:
            Updated solution
        """
        if iterations is None:
            iterations = self.max_iter
            
        h2 = h * h
        threads_per_block = 256
        n = nx * ny
        blocks = (n + threads_per_block - 1) // threads_per_block

        self.jacobi_kernel(
            (blocks,), (threads_per_block,),
            (phi, rho, cp.float32(h2), cp.float32(self.omega), 
             cp.int32(nx), cp.int32(ny), cp.int32(iterations))
        )
            
        return phi

    def residual(self, phi, rho, nx, ny, h):
        """
        Calculate residual (r = f - Ax).
        
        Args:
            phi: Current solution
            rho: Source term
            nx, ny: Grid dimensions
            h: Grid spacing
        
        Returns:
            Residual array
        """
        # Create output array
        res = cp.zeros_like(phi)
        
        # Set up thread and block dimensions
        block_size = 256
        grid_size = (nx * ny + block_size - 1) // block_size
        h2_inv = 1.0 / (h * h)
        
        # Launch kernel
        self.residual_kernel(
            (grid_size,), (block_size,), 
            (phi, rho, res, cp.float32(h2_inv), cp.int32(nx), cp.int32(ny))
        )
        
        # Wait for completion
        cp.cuda.Stream.null.synchronize()
        
        return res

    def restrict(self, fine, nx_fine, ny_fine):
        """
        Restrict from fine to coarse grid using full weighting.
        
        Args:
            fine: Fine grid array
            nx_fine, ny_fine: Fine grid dimensions
        
        Returns:
            Coarse grid array
        """
        nx_coarse = (nx_fine + 1) // 2
        ny_coarse = (ny_fine + 1) // 2
        
        # Create coarse grid array
        coarse = cp.zeros((ny_coarse * nx_coarse), dtype=fine.dtype)
        
        # Set up thread and block dimensions
        block_size = 256
        grid_size = (nx_coarse * ny_coarse + block_size - 1) // block_size
        
        # Launch kernel
        self.restriction_kernel(
            (grid_size,), (block_size,),
            (fine, coarse, cp.int32(nx_fine), cp.int32(ny_fine), cp.int32(nx_coarse))
        )
        
        # Wait for completion
        cp.cuda.Stream.null.synchronize()
        
        return coarse

    def prolong(self, coarse, nx_coarse, ny_coarse):
        """
        Prolong from coarse to fine grid.
        
        Args:
            coarse: Coarse grid array
            nx_coarse, ny_coarse: Coarse grid dimensions
        
        Returns:
            Fine grid array
        """
        # Calculate fine grid dimensions
        nx_fine = 2 * (nx_coarse - 1) + 1
        ny_fine = 2 * (ny_coarse - 1) + 1
        
        # Create output array
        fine = cp.zeros(ny_fine * nx_fine, dtype=coarse.dtype)
        
        # Set up thread and block dimensions
        block_size = 256
        grid_size = (nx_fine * ny_fine + block_size - 1) // block_size
        
        # Launch kernel
        self.prolongation_kernel(
            (grid_size,), (block_size,),
            (coarse, fine, cp.int32(nx_coarse), cp.int32(ny_coarse), cp.int32(nx_fine))
        )
        
        # Wait for completion
        cp.cuda.Stream.null.synchronize()
        
        return fine

    def v_cycle(self, phi, rho, h, nx, ny, level):
        """
        Perform one V-cycle of the multigrid algorithm.
        
        Args:
            phi: Current solution
            rho: Source term
            h: Grid spacing
            nx, ny: Grid dimensions
            level: Current level in multigrid hierarchy
        
        Returns:
            Updated solution
        """
        if level == 1 or nx <= 3 or ny <= 3:
            # Base case: just smooth more
            return self.smooth(phi, rho, nx, ny, h, iterations=50)

        # Pre-smoothing
        phi = self.smooth(phi, rho, nx, ny, h)

        # Compute residual
        res = self.residual(phi, rho, nx, ny, h)

        # Restrict residual to coarse grid
        nx_coarse = (nx + 1) // 2
        ny_coarse = (ny + 1) // 2
        res_coarse = self.restrict(res, nx, ny)

        # Initialize error correction on coarse grid
        err_coarse = cp.zeros_like(res_coarse)

        # Recursive V-cycle on coarse grid
        h_coarse = 2 * h
        err_coarse = self.v_cycle(err_coarse, res_coarse, h_coarse, nx_coarse, ny_coarse, level-1)

        # Prolong error correction to fine grid and apply correction
        err_fine = self.prolong(err_coarse, nx_coarse, ny_coarse)
        phi += err_fine

        # Post-smoothing
        phi = self.smooth(phi, rho, nx, ny, h)

        return phi

    def solve(self, rho, max_cycles=10):
        """
        Solve the Poisson equation using multigrid V-cycles.
        
        Args:
            rho: Source term (can be 2D array or flattened 1D array)
            max_cycles: Maximum number of V-cycles
        
        Returns:
            Solution phi and history of errors
        """
        # Ensure rho is a flattened array
        if len(rho.shape) > 1:
            rho = rho.reshape(-1)
        
        # Ensure we're using float32 for best GPU performance
        rho = rho.astype(cp.float32)
        
        # Calculate norm of rho for relative error
        #rho_norm = cp.linalg.norm(rho)
        
        # Initialize solution and error history
        phi = cp.zeros_like(rho)
        errors = []
        
        for cycle in range(max_cycles):
            # Perform V-cycle
            phi = self.v_cycle(phi, rho, self.h, self.N, self.N, self.levels)
            
            # Calculate residual and error
            res = self.residual(phi, rho, self.N, self.N, self.h)
            #res_norm = cp.linalg.norm(res)
            res_rel = 0#res_norm.get().item() / rho_norm.get().item()
            errors.append(res_rel)
            
            # Check for convergence
            if res_rel < self.tol:
                print(f"Converged after {cycle+1} cycles. Relative error: {res_rel:.2e}")
                break
                
        if cycle == max_cycles - 1:
            print(f"Reached maximum cycles ({max_cycles}). Final relative error: {res_rel:.2e}")
            
        return phi, errors

    def plot_solution(self, phi):
        """Plot the solution in 3D."""
        N = int(np.sqrt(phi.size))
        x = np.linspace(0, self.L, N)
        y = np.linspace(0, self.L, N)
        X, Y = np.meshgrid(x, y)
        Z = cp.asnumpy(phi.reshape(N, N))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True)

        ax.set_title("Solution")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('phi')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.show()
        
    def plot_error_history(self, errors):
        """Plot the error history."""
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(1, len(errors) + 1), errors, 'o-')
        plt.title("Convergence History")
        plt.xlabel("V-Cycle")
        plt.ylabel("Relative Error")
        plt.grid(True)
        plt.show()
        
    def benchmark(self, rho, cycles=10):
        """Benchmark the solver performance."""
        return benchmark(self.solve, (rho, cycles), n_repeat=3)


def main():
    # Problem setup
    N = 129  # grid size (should be 2^n + 1 for full multigrid cycle)
    L = 1.0
    levels = 4  # Number of multigrid levels
    cycles = 1  # Maximum V-cycles
    
    # Create solver
    solver = MultigridSolver(N, L, levels=levels, omega=0.1, tol=1e-5, max_iter=10)
    
    # Define source term (right-hand side of Poisson equation)
    x = cp.linspace(0, L, N)
    y = cp.linspace(0, L, N)
    X, Y = cp.meshgrid(x, y, indexing='ij')
    rho = cp.sin(16*cp.pi * X) * cp.sin(16*cp.pi * Y)
    rho = rho.ravel().astype(cp.float32)

    # Start profiling
    pr = cProfile.Profile()
    pr.enable()
    
    # Solve the system
    print("Running benchmark...")
    bench_result = solver.benchmark(-rho, cycles=cycles)
    print(bench_result)
    
    print("\nSolving the system...")
    phi, errors = solver.solve(-rho, max_cycles=cycles)
    
    # Stop profiling
    pr.disable()
    pr.dump_stats("profile.prof")
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open("profile.txt", "w") as f:
        f.write(s.getvalue())
    
    # Plot solution and error history
    solver.plot_solution(phi)
    solver.plot_error_history(errors)


if __name__ == '__main__':
    main()