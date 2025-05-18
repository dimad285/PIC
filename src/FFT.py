import numpy as np
import cupy as cp
from scipy.fft import dstn, idstn
from matplotlib import pyplot as plt
from cupyx.profiler import benchmark




def poisson_fft_dirichlet(f, g, dx):
    """
    Solves Laplace(phi) = f with Dirichlet BCs phi|boundary = g using FFT.
    
    f  : 2D array, RHS source term
    g  : 2D array, Dirichlet BCs (same shape as f, values only on boundary)
    dx : grid spacing (assumed equal in x and y)
    """

    def laplacian(phi, dx):
        """5-point Laplacian with Dirichlet BCs (assume phi has zero on boundaries)"""
        lap = np.zeros_like(phi)
        lap[1:-1,1:-1] = (
            -4 * phi[1:-1,1:-1] +
            phi[2:,1:-1] + phi[:-2,1:-1] +
            phi[1:-1,2:] + phi[1:-1,:-2]
        ) / dx**2
        return lap

    N, M = f.shape

    # Step 1: Create phi_g
    phi_g = np.zeros_like(f)
    phi_g[0,:]  = g[0,:]
    phi_g[-1,:] = g[-1,:]
    phi_g[:,0]  = g[:,0]
    phi_g[:,-1] = g[:,-1]

    # Step 2: Compute modified source term
    lap_phi_g = laplacian(phi_g, dx)
    f_mod = f - lap_phi_g

    # Step 3: Extract interior
    f_interior = f_mod[1:-1,1:-1]

    # Step 4: DST of RHS
    f_hat = dstn(f_interior, type=1, norm='ortho')

    # Step 5: Solve in spectral space
    nx, ny = f_interior.shape
    i = np.arange(1, nx+1).reshape(-1, 1)
    j = np.arange(1, ny+1).reshape(1, -1)
    denom = (
        2 * (np.cos(np.pi * i / (nx + 1)) - 1) +
        2 * (np.cos(np.pi * j / (ny + 1)) - 1)
    ) / dx**2
    phi_hat = f_hat / denom

    # Step 6: Inverse DST to get phi_0
    phi_0 = idstn(phi_hat, type=1, norm='ortho')

    # Step 7: Combine with phi_g
    phi = phi_g.copy()
    phi[1:-1,1:-1] += phi_0
    return phi


def dst2(x):
    """DST-II using CuPy FFT"""
    N = x.shape[0]
    x_ext = cp.zeros((2*N + 2,), dtype=x.dtype)
    x_ext[1:N+1] = x
    x_ext[N+2:] = -x[::-1]
    X = cp.fft.fft(x_ext)
    return cp.imag(X[1:N+1]) * 0.5

def idst2(X):
    """Inverse DST-II using CuPy FFT"""
    N = X.shape[0]
    X_ext = cp.zeros((2*N + 2,), dtype=X.dtype)
    X_ext[1:N+1] = -1j * X
    X_ext[N+2:] = 1j * X[::-1]
    x = cp.fft.ifft(X_ext).real
    return x[1:N+1]

def dst2_2d(u):
    """2D DST-II"""
    return cp.array([dst2(row) for row in u.T]).T  # first along x, then y

def idst2_2d(U):
    """2D inverse DST-II"""
    return cp.array([idst2(row) for row in U.T]).T


def laplacian(phi, dx, dy):
    lap = cp.zeros_like(phi)
    lap[1:-1, 1:-1] = (
        (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / dx**2 +
        (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dy**2
    )
    return lap

def poisson_solver_dirichlet_2d(f, dx):
    """
    Solve Poisson: ∇²φ = f with zero Dirichlet boundary using DST (CuPy).
    f: 2D array (inner domain, no boundaries)
    dx: grid spacing
    Returns solution on interior (excluding boundary)
    """
    N, M = f.shape
    f_hat = dst2_2d(f)

    # Eigenvalues of Laplace operator
    kx = cp.arange(1, N+1)
    ky = cp.arange(1, M+1)
    lam_x = 2 * cp.cos(cp.pi * kx / (N + 1)) - 2
    lam_y = 2 * cp.cos(cp.pi * ky / (M + 1)) - 2
    denom = lam_x[:, None] + lam_y[None, :]

    u_hat = f_hat / denom / (dx**2)
    u = idst2_2d(u_hat)
    return u


def plot_3d_surface(phi, title="Potential Field", cmap="viridis"):
    """
    Plot a 3D surface of the scalar field stored in `phi_flat`.

    Parameters:
        phi_flat (np.ndarray or cp.ndarray): Flattened scalar field (shape: nx * ny)
        nx (int): Number of grid points in x-direction
        ny (int): Number of grid points in y-direction
        title (str): Plot title
        cmap (str): Matplotlib colormap
    """

    if hasattr(phi, 'get') and hasattr(phi, 'shape'):
        # If phi is a cupy array, convert to numpy
        phi_plot = np.asarray(phi.get())
    else:
        phi_plot = np.asarray(phi)


    nx, ny = phi.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, phi_plot, cmap=cmap, edgecolor='k', linewidth=0.3, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('phi')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()



nx, ny = 128, 128
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Define the source term f(x, y) on the interior
f = cp.zeros((nx, ny))
x = cp.linspace(0, Lx, nx)
y = cp.linspace(0, Ly, ny)
X, Y = cp.meshgrid(x, y, indexing='ij')

# Example source (for testing): f = 1 in center
f[nx//4:3*nx//4, ny//4:3*ny//4] = 1.0

# Boundary condition array
phi_bc = cp.zeros_like(f)

# Arbitrary Dirichlet BCs (example: sinusoidal top)
phi_bc[:, -1] = cp.sin(cp.pi * x)


#plot_3d_surface(phi_inner, title="Potential Field from FFT Poisson Solver", cmap="viridis")

# Benchmarking
#print(benchmark(poisson_fft_dirichlet, (f, g, 0.1), n_repeat=10, n_warmup=5))
