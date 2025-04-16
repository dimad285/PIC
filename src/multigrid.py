import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from mpl_toolkits.mplot3d import Axes3D
from cupyx.profiler import benchmark
from cupyx.scipy.sparse.linalg import spsolve
import cProfile
import pstats
import io
import Solvers
from numba import jit



red_black_kernel_code = r'''
extern "C" __global__
void gauss_seidel_red_black(
    float* phi,
    const float* rho,
    float h2,
    float omega,
    int nx,
    int ny,
    int is_red
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny;
    if (idx >= size) return;
    int i = idx % nx;
    int j = idx / nx;

    // Only interior points
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        if ((i + j) % 2 == is_red) {
            int idx_n = (j + 1) * nx + i;
            int idx_s = (j - 1) * nx + i;
            int idx_e = j * nx + (i + 1);
            int idx_w = j * nx + (i - 1);
            
            float update = 0.25f * (
                phi[idx_n] + phi[idx_s] +
                phi[idx_e] + phi[idx_w] -
                h2 * rho[idx]
            );
            phi[idx] = (1.0f - omega) * phi[idx] + omega * update;
        }
    }
}
'''

gauss_seidel_kernel = cp.RawKernel(red_black_kernel_code, 'gauss_seidel_red_black')

jacobi_kernel = cp.RawKernel(r'''
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
    
    // Move boundary check outside the loop
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx_n = (j + 1) * nx + i;
        int idx_s = (j - 1) * nx + i;
        int idx_e = j * nx + (i + 1);
        int idx_w = j * nx + (i - 1);
        
        for (int iter = 0; iter < iterations; ++iter) {
            float update = 0.25f * (
                phi[idx_n] + phi[idx_s] +
                phi[idx_e] + phi[idx_w] -
                h2 * rho[idx]
            );
            phi_local = (1.0f - omega) * phi_local + omega * update;
            //__syncthreads();  // optional depending on memory model
        }
    }
    
    phi[idx] = phi_local;
}
''', 'jacobi_kernel')


def smooth(phi_1d, rho_1d, h, nx, ny, omega=2/3, iterations=5):
    
    h2 = h * h

    threads_per_block = 256
    n = nx * ny
    blocks = (n + threads_per_block - 1) // threads_per_block

    for _ in range(iterations):
        jacobi_kernel(
        (blocks,), (threads_per_block,),
        (phi_1d, rho_1d, cp.float32(h2), cp.float32(omega), cp.int32(nx), cp.int32(ny), cp.int32(iterations))
    )
        
    return phi_1d

def _smooth(phi_1d, rho_1d, h, nx, ny, omega=2/3, iterations=10):
    h2 = h * h
    threads_per_block = 256
    n = nx * ny
    blocks = (n + threads_per_block - 1) // threads_per_block

    for _ in range(iterations):
        # Red update
        gauss_seidel_kernel(
            (blocks,), (threads_per_block,),
            (phi_1d, rho_1d, cp.float32(h2), cp.float32(omega), cp.int32(nx), cp.int32(ny), cp.int32(1))
        )
        # Black update
        gauss_seidel_kernel(
            (blocks,), (threads_per_block,),
            (phi_1d, rho_1d, cp.float32(h2), cp.float32(omega), cp.int32(nx), cp.int32(ny), cp.int32(0))
        )


def residual(phi, rho, h, nx, ny):
    """Calculate residual using CUDA kernel"""
    residual_kernel = cp.RawKernel("""
    // Residual Kernel
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
    

    # Create output array
    res = cp.zeros_like(phi)
    
    # Set up thread and block dimensions
    block_size = 256
    grid_size = (nx * ny + block_size - 1) // block_size
    h2_inv = 1.0 / (h * h)
    
    # Launch kernel
    residual_kernel((grid_size,), (block_size,), (phi, rho, res, cp.float32(h2_inv), cp.int32(nx), cp.int32(ny)))
    
    # Wait for completion
    cp.cuda.Stream.null.synchronize()
    
    return res

'''
def residual(phi, rho, h):
    res = cp.zeros_like(phi)
    res[1:-1,1:-1] = rho[1:-1,1:-1] - (
        (phi[2:,1:-1] - 2*phi[1:-1,1:-1] + phi[:-2,1:-1]) +
        (phi[1:-1,2:] - 2*phi[1:-1,1:-1] + phi[1:-1,:-2])
    ) / h**2
    return res
'''

def restrict(fine, nx_fine, ny_fine):

    # Extract every other point in both dimensions
    # First get every other row, then every other column
    indices = cp.arange(0, nx_fine*ny_fine)[::2*nx_fine].reshape(-1, 1) + cp.arange(0, nx_fine, 2)
    indices = indices.flatten()
    
    # Create coarse grid with copy of values
    coarse = fine[indices].copy()
    
    return coarse

'''
def restrict(fine):
    # Get every other point for the coarse grid
    coarse = cp.zeros(((fine.shape[0] + 1) // 2, (fine.shape[1] + 1) // 2), dtype=fine.dtype)
    
    # Just use injection (every other point)
    coarse = fine[0::2, 0::2]
    
    return coarse
'''

def prolong(coarse, nx_coarse, ny_coarse):
    """Prolong from coarse to fine grid using CUDA kernel"""

    prolongation_kernel = cp.RawKernel('''
    extern "C" __global__ 
    void prolong_kernel(
        const float* coarse,
        float* fine,
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
                    fine[i_fine * nx_fine + j_fine] = 0.5f * (
                        coarse[i_coarse * nx_coarse + j_coarse] + 
                        coarse[(i_coarse+1) * nx_coarse + j_coarse]
                    );
                }
            }
            else if (!is_i_odd && is_j_odd) {
                if (j_coarse < nx_coarse - 1) {
                    fine[i_fine * nx_fine + j_fine] = 0.5f * (
                        coarse[i_coarse * nx_coarse + j_coarse] + 
                        coarse[i_coarse * nx_coarse + (j_coarse+1)]
                    );
                }
            }
            else {
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
    ''', 'prolong_kernel')

    # Ensure array is float32
    
    # Calculate fine grid dimensions
    nx_fine = 2 * (nx_coarse - 1) + 1
    ny_fine = 2 * (ny_coarse - 1) + 1
    
    # Create output array
    fine = cp.zeros((ny_fine, nx_fine), dtype=coarse.dtype).reshape(-1)
    
    # Set up thread and block dimensions
    block_size = 256
    grid_size = (nx_fine * ny_fine + block_size - 1) // block_size
    
    # Launch kernel
    prolongation_kernel((grid_size,), (block_size,), (coarse, fine, cp.int32(nx_coarse), cp.int32(ny_coarse), cp.int32(nx_fine)))
    
    # Wait for completion
    cp.cuda.Stream.null.synchronize()
    
    return fine


'''
def prolong(coarse):
    nx, ny = coarse.shape
    fine = cp.zeros((2 * (nx - 1) + 1, 2 * (ny - 1) + 1), dtype=coarse.dtype)

    fine[::2, ::2] = coarse  # direct copy
    fine[1::2, ::2] = 0.5 * (coarse[:-1, :] + coarse[1:, :])
    fine[::2, 1::2] = 0.5 * (coarse[:, :-1] + coarse[:, 1:])
    fine[1::2, 1::2] = 0.25 * (
        coarse[:-1, :-1] + coarse[1:, :-1] +
        coarse[:-1, 1:] + coarse[1:, 1:]
    )

    return fine
'''


def v_cycle(phi, rho, h, N, levels, omega=2/3, max_iter=5):

    if levels == 1:
        # Direct solve (just smooth more here, or use FFT)
        #A = Solvers.Laplacian_square(N, N, h, h)
        #return cp.linalg.solve(A, rho*h**2)
        return smooth(phi, rho, h, N, N, omega=omega, iterations=20)

    # Pre-smoothing
    phi = smooth(phi, rho, h, N, N, omega=omega, iterations=max_iter)
    # Residual
    res = residual(phi, rho, h, N, N)

    # Restrict
    res_coarse = restrict(res, N, N)

    # Initialize coarse grid phi
    phi_coarse = cp.zeros_like(res_coarse)

    # Recursive V-cycle on coarse grid
    h_coarse = 2 * h
    phi_coarse = v_cycle(phi_coarse, res_coarse, h_coarse, (N+1)//2, levels-1, omega=omega, max_iter=max_iter)

    # Prolong and correct
    phi += prolong(phi_coarse, (N+1)//2, (N+1)//2)
    # Post-smoothing
    phi = smooth(phi, rho, h, N, N, omega=omega, iterations=max_iter)


    return phi


def main(): 


    def plot_cycle(field):
        N = field.shape[0]
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)
        Z = field

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.3, antialiased=True)

        ax.set_title(f"Potential at V-Cycle")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('phi')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.show()

    def plot_error():
        plt.plot(errors, 'o-')
        plt.title("Error vs. V-Cycle")
        plt.xlabel("V-Cycle")
        plt.ylabel("Error")
        plt.show()




    cycles = 10
    N = 129  # grid size (should be 2^n + 1 for full multigrid cycle)
    L = 1.0
    h = L / (N - 1)
    x = cp.linspace(0, L, N)
    y = cp.linspace(0, L, N)
    X, Y = cp.meshgrid(x, y, indexing='ij')
    levels = 2

    # Define source term and exact solution
    rho = cp.sin(16*cp.pi * X) * cp.sin(16*cp.pi * Y)
    #rho += (cp.sin(1*cp.pi * X) * cp.sin(1*cp.pi * Y)) * 1 
    rho = rho.ravel()
    rho = rho.astype(cp.float32)
    rho_norm = cp.linalg.norm(rho)
    phi = cp.zeros_like(rho)
    errors = []
    potentials_np = [rho.reshape(N, N)]


    def multigrid(phi, rho, h, levels=8, omega=1.0, tol=1e-5, max_iter=50):
        #phi = cp.zeros_like(rho)
        for _ in range(cycles):
            phi = v_cycle(phi, rho, h, N, levels=levels, omega=omega, max_iter=max_iter)
            res = residual(phi, rho, h, N, N)
            res_norm = cp.linalg.norm(res)
            res_rel = res_norm/rho_norm
            #errors.append(res_rel)
            if res_rel < tol:
                #print("Converged after " + str(_) + " cycles")
                break
        #print("Final error: " + str(res_rel))
             

    #multigrid(phi, -rho, h, 2, 2/3, 1e-5, 20)
    #smooth(phi, -rho, h, N, N, omega=2/3, iterations=10)
    print(benchmark(multigrid, (phi, -rho, h, levels, 0.1, 1e-5, 5), n_repeat=100))
    #multigrid(phi, -rho, h, levels, 2/3, 1e-5, 20)
    #plot_cycle(phi.reshape(N, N).get())
    #plot_error()


# --- Test the Smoother ---
if __name__ == '__main__':

    
    # Start profiling before running simulation
    pr = cProfile.Profile()
    pr.enable()
    
    # Run simulation
    main()
    
    # Stop profiling and save results
    pr.disable()
    pr.dump_stats("profile.prof")
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open("profile.txt", "w") as f:
        f.write(s.getvalue())
