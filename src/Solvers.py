import cupy as cp
import numpy as np

def boundary_array(input: tuple, gridsize: tuple) -> cp.ndarray:
    boundary = [[], []]
    
    for i in input:
        m1, n1, m2, n2 = i[0]
        V = i[1]
        # Compute direction steps using np.sign
        dm = np.sign(m2 - m1)
        dn = np.sign(n2 - n1)
        
        x, y = m1, n1
        
        # Iterate over the boundary points
        while True:
            boundary[0].append(y * gridsize[0] + x)
            boundary[1].append(V)
            
            # Update x and y
            x += dm
            y += dn
            
            if x == m2 and y == n2:
                boundary[0].append(y * gridsize[0] + x)
                boundary[1].append(V)
                break
    
    return cp.asarray(boundary, dtype=cp.int32)


def boundary_conditions_left_gpu(A:cp.ndarray, boundary:cp.ndarray):
    for i in boundary[0]:
        A[i,:] = 0
        A[i, i] = 1



def Laplacian_square(m, n) -> cp.ndarray:

    Lap = cp.zeros((m*n, m*n), dtype=cp.float32)
    for i in range(n):
        for j in range(m):
            idx = i*m+j
            Lap[idx, idx] = -4
            if j>0:
                Lap[idx, idx-1] = 1
            if j<m-1:
                Lap[idx, idx+1] = 1
            if i>0:
                Lap[idx, idx-m] = 1
            if i<n-1:
                Lap[idx, idx+m] = 1
                
    return Lap.astype(cp.float32)


def Laplacian_cilindrical(m, n) -> cp.ndarray:

    Lap = cp.zeros((m*n, m*n), dtype=cp.float32)
    for i in range(n):
        for j in range(m):
            idx = i*m+j
            Lap[idx, idx] = -4
            if j>0:
                Lap[idx, idx-1] = 1
            if j<m-1:
                Lap[idx, idx+1] = 1
            if i>0:
                Lap[idx, idx-m] = 1
            if i<n-1:
                Lap[idx, idx+m] = 1
    return Lap.astype(cp.float32)


def setup_fft_solver(m, n):
    # Create wavenumber arrays
    kx = 2 * cp.pi * cp.fft.fftfreq(m)
    ky = 2 * cp.pi * cp.fft.fftfreq(n)
    
    # Create 2D wavenumber grid
    kx_grid, ky_grid = cp.meshgrid(kx, ky)
    
    # Compute k^2, avoiding division by zero at k=0
    k_sq = kx_grid**2 + ky_grid**2
    k_sq[0, 0] = 1.0  # Avoid division by zero
    
    return k_sq


def solve_poisson_fft_3d(rho, k_sq, epsilon0):
    # Reshape rho to 3D
    rho_3d = rho.reshape(k_sq.shape)
   
    # Compute FFT of charge density
    rho_k = cp.fft.fftn(rho_3d)
   
    # Solve Poisson equation in Fourier space
    phi_k = rho_k / (k_sq * epsilon0)
   
    # Handle k=0 mode (set to average of phi)
    phi_k[0, 0, 0] = 0
   
    # Inverse FFT to get potential
    phi = cp.fft.ifftn(phi_k).real / (k_sq.shape[0] * k_sq.shape[1] * k_sq.shape[2])
   
    return phi.ravel()  # Return as 1D array

def setup_fft_solver_3d(m, n, p):
    # Create wavenumber arrays
    kx = 2 * cp.pi * cp.fft.fftfreq(m)
    ky = 2 * cp.pi * cp.fft.fftfreq(n)
    kz = 2 * cp.pi * cp.fft.fftfreq(p)
   
    # Create 3D wavenumber grid
    kx_grid, ky_grid, kz_grid = cp.meshgrid(kx, ky, kz, indexing='ij')
   
    # Compute k^2, avoiding division by zero at k=0
    k_sq = kx_grid**2 + ky_grid**2 + kz_grid**2
    k_sq[0, 0, 0] = 1.0  # Avoid division by zero
   
    return k_sq

def solve_poisson_fft(rho, k_sq, epsilon0):

    # Reshape rho to 2D
    rho_2d = rho.reshape(k_sq.shape)
    
    # Compute FFT of charge density
    rho_k = cp.fft.fftn(rho_2d)
    
    # Solve Poisson equation in Fourier space
    phi_k = rho_k / (k_sq * epsilon0)
    
    # Handle k=0 mode (set to average of phi)
    phi_k[0, 0] = 0
    
    # Inverse FFT to get potential
    phi = cp.fft.ifftn(phi_k).real / (k_sq.shape[0] * k_sq.shape[1])
    
    return phi.ravel()  # Return as 1D array



def preconditioned_cg_solver_gpu(A, M_inv, b, x0=None, tol=1e-5, max_iter=1000):
    """
    Preconditioned Conjugate Gradient solver for Ax = b.
    
    Args:
    A: function that performs matrix-vector product
    M_inv: function that applies the inverse of the preconditioner
    b: right-hand side vector (1D CuPy array)
    x0: initial guess (if None, use zeros)
    tol: tolerance for convergence
    max_iter: maximum number of iterations
    
    Returns:
    x: solution vector (1D CuPy array)
    """
    x = x0 if x0 is not None else cp.zeros_like(b)
    r = b - A(x)
    z = M_inv(r)
    p = z.copy()
    rz = cp.dot(r, z)
    
    for i in range(max_iter):
        Ap = A(p)
        alpha = rz / cp.dot(p, Ap)
        
        x += alpha * p
        r -= alpha * Ap
        
        if cp.linalg.norm(r) < tol:
            #print(f"PCG converged in {i+1} iterations")
            return x
        
        z = M_inv(r)
        rz_new = cp.dot(r, z)
        beta = rz_new / rz
        rz = rz_new
        
        p = z + beta * p
    
    print(f"PCG did not converge within {max_iter} iterations")
    return x

def setup_poisson_operator_gpu(m, n, dx, dy):
    """
    Setup the GPU-optimized Poisson operator for the conjugate gradient solver.
    
    Args:
    m, n: grid dimensions
    dx, dy: grid spacings
    
    Returns:
    A: function that applies the Poisson operator
    """
    idx2, idy2 = 1/dx**2, 1/dy**2
    
    poisson_kernel = cp.ElementwiseKernel(
        'raw T x, int32 m, int32 n, float64 idx2, float64 idy2',
        'T y',
        '''
        int row = i % m;
        int col = i / m;
        if (row > 0 && row < m-1 && col > 0 && col < n-1) {
            y = (-2 * (idx2 + idy2) * x[i] +
                 idx2 * (x[i+1] + x[i-1]) +
                 idy2 * (x[i+m] + x[i-m]));
        } else {
            y = x[i];  // Dirichlet boundary condition
        }
        ''',
        'poisson_kernel'
    )
    
    def A(x):
        y = cp.empty_like(x)
        poisson_kernel(x, m, n, idx2, idy2, y)
        return y
    
    return A

def setup_jacobi_preconditioner_gpu(m, n, dx, dy):
    """
    Setup the Jacobi preconditioner for the Poisson equation.
    
    Args:
    m, n: grid dimensions
    dx, dy: grid spacings
    
    Returns:
    M_inv: function that applies the inverse of the preconditioner
    """
    idx2, idy2 = 1/dx**2, 1/dy**2
    diag_val = -2 * (idx2 + idy2)
    
    jacobi_kernel = cp.ElementwiseKernel(
        'T r, float64 diag_val',
        'T z',
        'z = r / diag_val',
        'jacobi_kernel'
    )
    
    def M_inv(r):
        z = cp.empty_like(r)
        jacobi_kernel(r, diag_val, z)
        return z
    
    return M_inv

def solve_poisson_pcg_gpu(rho, m, n, dx, dy, eps0, tol=1e-5, max_iter=1000):
    """
    Solve the Poisson equation using the Preconditioned Conjugate Gradient method.
    
    Args:
    rho: charge density (1D CuPy array)
    m, n: grid dimensions
    dx, dy: grid spacings
    eps0: permittivity of free space
    tol: tolerance for convergence
    max_iter: maximum number of iterations
    
    Returns:
    phi: electrostatic potential (1D CuPy array)
    """
    A = setup_poisson_operator_gpu(m, n, dx, dy)
    M_inv = setup_jacobi_preconditioner_gpu(m, n, dx, dy)
    b = -rho / eps0
    phi = preconditioned_cg_solver_gpu(A, M_inv, b, tol=tol, max_iter=max_iter)
    return phi