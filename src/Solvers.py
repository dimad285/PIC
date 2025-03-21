import cupy as cp
import numpy as np
#import cupyx as cpx
from cupyx.scipy.sparse import diags, csr_matrix, eye, kron
from cupyx.scipy.sparse.linalg import gmres
import Consts
import Grid


def boundary_conditions_left_gpu(A:cp.ndarray, boundary:cp.ndarray):
    for i in boundary:
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

def Laplacian_square_csr(m, n) -> cp.ndarray:
    size = m * n
    diagonals = []

    # Main diagonal
    main_diag = -4 * cp.ones(size)
    diagonals.append(main_diag)

    # Horizontal off-diagonals
    off_diag_h = cp.ones(size)
    off_diag_h[m - 1::m] = 0  # Remove connections across row boundaries
    diagonals.append(off_diag_h[:-1])
    diagonals.append(off_diag_h[:-1])  # Mirror for lower diagonal

    # Vertical off-diagonals
    off_diag_v = cp.ones(size - m)
    diagonals.append(off_diag_v)
    diagonals.append(off_diag_v)  # Mirror for opposite direction

    # Sparse matrix creation
    laplacian = diags(
        diagonals,
        [0, 1, -1, m, -m],  # Corresponding diagonal indices
        shape=(size, size),
        format="csr"
    )
    return laplacian

def apply_boundary_conditions(Laplacian, boundary_nodes, boundary_value=0):
    """
    Apply Dirichlet boundary conditions directly to a CuPy CSR matrix.

    Parameters:
        Laplacian (cupyx.scipy.sparse.csr_matrix): The Laplacian matrix in CSR format.
        boundary_nodes (array-like): Indices of boundary nodes in the flattened grid.
        boundary_value (float): The value to enforce at the boundary nodes.

    Returns:
        cupyx.scipy.sparse.csr_matrix: The modified Laplacian matrix.
    """
    # Convert boundary nodes to CuPy array
    boundary_nodes = cp.array(boundary_nodes, dtype=cp.int32)

    # Get matrix components
    data = Laplacian.data
    indices = Laplacian.indices
    indptr = Laplacian.indptr

    # Iterate through boundary nodes
    for node in boundary_nodes:
        # Clear the row for the boundary node
        start, end = indptr[node], indptr[node + 1]
        data[start:end] = 0
        indices[start:end] = -1  # Mark invalid indices (not strictly necessary)

        # Set the diagonal element to 1
        indptr[node + 1] = start + 1  # Only one element in the row
        data[start] = 1
        indices[start] = node

    # Return updated Laplacian
    return csr_matrix((data, indices, indptr), shape=Laplacian.shape)

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


def setup_fft_solver(m, n, dx, dy):
    # Create wavenumber arrays
    kx = 2 * cp.pi * cp.fft.fftfreq(m, dx)
    ky = 2 * cp.pi * cp.fft.fftfreq(n, dy)
    
    # Create 2D wavenumber grid
    kx_grid, ky_grid = cp.meshgrid(kx, ky)
    
    # Compute k^2, avoiding division by zero at k=0
    k_sq = kx_grid**2 + ky_grid**2
    k_sq[0, 0] = 1.0  # Avoid division by zero
    
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
    phi = cp.fft.ifftn(phi_k).real
    
    return phi.ravel()  # Return as 1D array



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
    phi = cp.fft.ifftn(phi_k).real
   
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




def cg_solver_gpu(A, b, x0=None, tol=1e-5, max_iter=1000, verbose=False):
    """
    Conjugate Gradient solver for Ax = b without preconditioning.
    
    Args:
    A: function that performs the matrix-vector product (A @ x)
    b: right-hand side vector (1D CuPy array)
    x0: initial guess (1D CuPy array, optional; if None, starts with zeros)
    tol: relative tolerance for convergence
    max_iter: maximum number of iterations
    verbose: if True, prints convergence information
    
    Returns:
    x: solution vector (1D CuPy array)
    """
    x = x0 if x0 is not None else cp.zeros_like(b)
    r = b - A(x)
    b_norm = cp.linalg.norm(b)
    if b_norm == 0:
        b_norm = 1  # Prevent division by zero if b is zero
    r_norm = cp.linalg.norm(r)
    
    if r_norm / b_norm < tol:
        if verbose:
            print("CG converged at iteration 0")
        return x
    
    p = r.copy()
    rs_old = cp.dot(r, r)

    for i in range(max_iter):
        Ap = A(p)
        alpha = rs_old / cp.dot(p, Ap)
        
        x += alpha * p
        r -= alpha * Ap
        
        r_norm = cp.linalg.norm(r)
        relative_residual = r_norm / b_norm
        if relative_residual < tol:
            if verbose:
                print(f"CG converged in {i+1} iterations, Relative Residual: {relative_residual:.2e}")
            return x
        
        rs_new = cp.dot(r, r)
        beta = rs_new / rs_old
        rs_old = rs_new
        
        p = r + beta * p
        
        if verbose and i % 10 == 0:
            print(f"Iteration {i}, Relative Residual: {relative_residual:.2e}")
    
    #raise RuntimeError("CG did not converge within the maximum number of iterations.")

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
    b_norm = cp.linalg.norm(b)
    if b_norm == 0:
        b_norm = 1  # Prevent division by zero if b is all zeros
    
    for i in range(max_iter):
        Ap = A(p)
        alpha = rz / cp.dot(p, Ap)
        
        x += alpha * p
        r -= alpha * Ap
        
        r_norm = cp.linalg.norm(r)
        relative_residual = r_norm / b_norm
        if relative_residual < tol:
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
    idx2, idy2 = 1 / dx**2, 1 / dy**2
    inv_diag = -2 * (idx2 + idy2)  # Precompute the diagonal term

    poisson_kernel = cp.ElementwiseKernel(
        'raw T x, int32 m, int32 n, float64 idx2, float64 idy2, float64 inv_diag',
        'T y',
        '''
        int row = i / n;
        int col = i % n;

        if (row > 0 && row < m-1 && col > 0 && col < n-1) {
            y = inv_diag * x[i] +
                idx2 * (x[i + 1] + x[i - 1]) +
                idy2 * (x[i + n] + x[i - n]);
        } else {
            y = x[i];  // Dirichlet boundary condition (copy boundary values)
        }
        ''',
        'poisson_kernel'
    )

    def A(x):
        """
        Applies the Poisson operator to the input vector `x`.

        Args:
        x: Input vector representing the grid values (1D CuPy array of size m*n).

        Returns:
        y: Result of applying the Poisson operator (1D CuPy array of size m*n).
        """
        y = cp.empty_like(x)
        poisson_kernel(x, m, n, idx2, idy2, inv_diag, y)
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

def solve_poisson_pcg_gpu(rho, m, n, dx, dy, eps0, phi0=None, tol=1e-5, max_iter=1000, preconditioner="jacobi"):
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
    b = -rho / eps0
    if preconditioner == "jacobi":
        M_inv = setup_jacobi_preconditioner_gpu(m, n, dx, dy)
        phi = preconditioned_cg_solver_gpu(A, M_inv, b, x0=phi0, tol=tol, max_iter=max_iter)
    elif preconditioner == "none":
        phi = cg_solver_gpu(A, b, x0=phi0, tol=tol, max_iter=max_iter)
    return phi




class Solver():
    def __init__(self, solver_type:str, grid:Grid.Grid2D, boundaries=None, tol = 1e-5):
        
        self.solver_type = solver_type

        self.m = grid.m
        self.n = grid.n
        self.dx = grid.dx
        self.dy = grid.dy
        self.cyilindrical = grid.cylindrical
        self.tol = tol

        self.boundaries = boundaries
        print(self.boundaries)

        self.init_solver()

        
    def init_solver(self):
    
        match self.solver_type:
            case 'inverse': # add matrix compression
                print('creating Laplacian...')
                self.Lap = Laplacian_square(self.m, self.n)
                if self.boundaries != None:
                    print('applying boundary conditions...')
                    boundary_conditions_left_gpu(self.Lap, self.boundaries[0])

                print('creating inverse Laplacian...')
                self.Lap = cp.linalg.inv(self.Lap)
            case 'fft':
                if not self.cyilindrical:  # Note: corrected typo from cilindrical
                    #fft_solver = Solvers.PoissonFFTSolver(m, n, dx, dy, Consts.eps0)
                    self.k_sq = setup_fft_solver(self.m, self.n, self.dx, self.dy)
                else:
                    pass
            case 'cg': # add boundary conditions
                print('Using Conjugate Gradient solver')
                self.Lap = Laplacian_square_csr(self.m, self.n)
                if self.boundaries != None:
                    print('applying boundary conditions...')
                    apply_boundary_conditions(self.Lap, *self.boundaries)
            case 'cg_fft':
                print('Using Conjugate Gradient solver with FFT preconditioner')
                self.k_sq = setup_fft_solver(self.m, self.n, self.dx, self.dy)
            case 'multigrid':
                print('Using Multigrid solver')
                # These will be initialized in the MultigridSolver class
                pass
            case 'gmres':
                print('Using GMRES solver')
                self.Lap = Laplacian_square_csr(self.m, self.n)
                if self.boundaries != None:
                    print('applying boundary conditions...')
                    apply_boundary_conditions(self.Lap, *self.boundaries)


    def solve(self, grid: Grid.Grid2D):

        if self.boundaries != None:
            grid.b[:] = -grid.rho * Consts.eps0_1
            grid.b[self.boundaries[0]] = self.boundaries[1]
        else:
            grid.b[:] = -grid.rho * Consts.eps0_1

        match self.solver_type:
            case 'inverse':
                grid.phi[:] = cp.dot(self.Lap, grid.b)
            case 'fft':
                grid.phi[:] = solve_poisson_fft(grid.rho, self.k_sq, Consts.eps0)
            case 'cg':
                grid.phi[:] = solve_poisson_pcg_gpu(
                    grid.rho, grid.m, grid.n, grid.dx, grid.dy, 
                    Consts.eps0, phi0=grid.phi, tol=self.tol, preconditioner='none'
                )
            case 'cg_precond':
                grid.phi[:] = solve_poisson_pcg_gpu(
                    grid.rho, grid.m, grid.n, grid.dx, grid.dy, 
                    Consts.eps0, phi0=grid.phi, tol=self.tol, preconditioner='jacobi'
                )
            case 'multigrid':
                # This method will be overridden in the MultigridSolver class
                # But we put a simple implementation here for completeness
                print("Warning: Using base Solver.solve for multigrid, which is not implemented!")
                pass
            case 'gmres':
                grid.phi[:], iter = gmres(self.Lap, grid.b, x0=grid.phi, tol=self.tol)
            case None:
                pass




