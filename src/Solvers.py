import cupy as cp
from typing import Tuple, Optional
from enum import Enum
import cupyx
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg

def boundary_array(input: tuple, gridsize: tuple) -> tuple[cp.ndarray, cp.ndarray]:
    # Initialize Python-native lists for indices and values
    boundary_indices = []
    boundary_values = []
    
    for segment in input:
        (m1, n1, m2, n2), V = segment  # Unpack segment
        
        dm = cp.sign(m2 - m1).item()  # Convert to Python scalar
        dn = cp.sign(n2 - n1).item()  # Convert to Python scalar
        
        x, y = m1, n1
        
        # Iterate over the boundary points
        while True:
            boundary_indices.append(y * gridsize[0] + x)  # Flattened index
            boundary_values.append(V)
            
            # Break loop if endpoint reached
            if (x == m2 and y == n2):
                break
            
            # Update x and y
            x += dm
            y += dn
    
    # Convert Python-native lists to CuPy arrays
    indices = cp.array(boundary_indices, dtype=cp.int32)
    values = cp.array(boundary_values, dtype=cp.float32)
    
    return indices, values


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

def restrict_grid(fine_grid, coarse_grid):
    """
    Restrict a 2D fine grid to a coarser grid using full weighting on GPU.
    Updates the `coarse_grid` in place.
    
    Parameters:
    - fine_grid (cupy.ndarray): 2D array of the fine grid values (on GPU).
    - coarse_grid (cupy.ndarray): 2D array of the coarser grid values (on GPU, updated in place).
    """
    fine_nx, fine_ny = fine_grid.shape
    coarse_nx, coarse_ny = coarse_grid.shape
    
    # Ensure grid sizes are compatible
    if fine_nx != 2 * coarse_nx - 1 or fine_ny != 2 * coarse_ny - 1:
        raise ValueError("Fine grid dimensions must be compatible with coarse grid dimensions.")
    
    # Restriction using full weighting
    # Fine grid center points
    coarse_grid[1:-1, 1:-1] = (
        fine_grid[2:-2:2, 2:-2:2] +  # Center points
        0.5 * (fine_grid[1:-3:2, 2:-2:2] + fine_grid[3::2, 2:-2:2] +  # Vertical neighbors
               fine_grid[2:-2:2, 1:-3:2] + fine_grid[2:-2:2, 3::2]) +  # Horizontal neighbors
        0.25 * (fine_grid[1:-3:2, 1:-3:2] + fine_grid[1:-3:2, 3::2] +  # Diagonal neighbors
                fine_grid[3::2, 1:-3:2] + fine_grid[3::2, 3::2])
    )
    
    # Boundary values (copy from fine grid)
    coarse_grid[0, :] = fine_grid[0, ::2]
    coarse_grid[-1, :] = fine_grid[-1, ::2]
    coarse_grid[:, 0] = fine_grid[::2, 0]
    coarse_grid[:, -1] = fine_grid[::2, -1]


def interpolate_grid(coarse_grid, fine_grid):
    """
    Interpolates a 2D coarse grid to a fine grid using bilinear interpolation on GPU.
    Updates the `fine_grid` in place.
    
    Parameters:
    - coarse_grid (cupy.ndarray): 2D array of the coarse grid values (on GPU).
    - fine_grid (cupy.ndarray): 2D array of the fine grid values (on GPU, updated in place).
    """
    coarse_nx, coarse_ny = coarse_grid.shape
    fine_nx, fine_ny = fine_grid.shape

    # Ensure grid sizes are compatible
    if fine_nx != 2 * coarse_nx - 1 or fine_ny != 2 * coarse_ny - 1:
        raise ValueError("Fine grid dimensions must be compatible with coarse grid dimensions.")
    
    # Copy coarse grid values to corresponding fine grid points
    fine_grid[::2, ::2] = coarse_grid
    
    # Interpolate along rows (horizontal edges)
    fine_grid[1::2, ::2] = 0.5 * (fine_grid[:-1:2, ::2] + fine_grid[2::2, ::2])
    
    # Interpolate along columns (vertical edges)
    fine_grid[::2, 1::2] = 0.5 * (fine_grid[::2, :-1:2] + fine_grid[::2, 2::2])
    
    # Interpolate interior points (center of 2x2 coarse cells)
    fine_grid[1::2, 1::2] = 0.25 * (
        fine_grid[:-1:2, :-1:2] + fine_grid[2::2, :-1:2] +  # Top-left and bottom-left
        fine_grid[:-1:2, 2::2] + fine_grid[2::2, 2::2]      # Top-right and bottom-right
    )



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



def setup_ilu_preconditioner_gpu(A_sparse):
    M = cupyx.scipy.sparse.linalg.spilu(A_sparse)  # ILU factorization
    def M_inv(r):
        return M.solve(r)
    return M_inv

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
    elif preconditioner == "ilu":
        M_inv = setup_ilu_preconditioner_gpu(A)
        phi = preconditioned_cg_solver_gpu(A, M_inv, b, x0=phi0, tol=tol, max_iter=max_iter)
    elif preconditioner == "none":
        phi = cg_solver_gpu(A, b, x0=phi0, tol=tol, max_iter=max_iter)
    return phi




    """
    A 2D Poisson equation solver using Fast Fourier Transform (FFT) method.
    Supports periodic, Dirichlet, and Neumann boundary conditions.
    Solves: ∇²φ = -ρ/ε₀
    """
    
    def __init__(self, m: int, n: int, dx: float = 1.0, dy: float = 1.0, epsilon0: float = 1.0,
                 boundary_type: BoundaryCondition = BoundaryCondition.PERIODIC):
        """
        Initialize the Poisson solver.
        
        Args:
            m: Number of grid points in x direction
            n: Number of grid points in y direction
            dx: Grid spacing in x direction
            dy: Grid spacing in y direction
            epsilon0: Permittivity constant
            boundary_type: Type of boundary condition
        """
        self.m = m
        self.n = n
        self.dx = dx
        self.dy = dy
        self.epsilon0 = epsilon0
        self.boundary_type = boundary_type
        
        # Extended grid size for non-periodic boundaries
        self.m_ext = m if boundary_type == BoundaryCondition.PERIODIC else 2 * m
        self.n_ext = n if boundary_type == BoundaryCondition.PERIODIC else 2 * n
        
        self.k_sq = self._setup_wavenumbers()
    
    def _setup_wavenumbers(self) -> cp.ndarray:
        """Set up the wavenumber arrays based on boundary conditions."""
        if self.boundary_type == BoundaryCondition.PERIODIC:
            kx = 2 * cp.pi * cp.fft.fftfreq(self.m, self.dx)
            ky = 2 * cp.pi * cp.fft.fftfreq(self.n, self.dy)
        else:
            # For Dirichlet/Neumann, use sine/cosine transforms frequencies
            kx = cp.pi * cp.arange(self.m_ext) / (self.m_ext * self.dx)
            ky = cp.pi * cp.arange(self.n_ext) / (self.n_ext * self.dy)
            
        kx_grid, ky_grid = cp.meshgrid(kx, ky, indexing='ij')
        k_sq = kx_grid**2 + ky_grid**2
        k_sq[0, 0] = 1.0  # Avoid division by zero
        return k_sq

    def _apply_dirichlet_bc(self, phi: cp.ndarray, bc_values: dict) -> cp.ndarray:
        """Apply Dirichlet boundary conditions."""
        # Extract boundary values
        top = bc_values.get('top', 0.0)
        bottom = bc_values.get('bottom', 0.0)
        left = bc_values.get('left', 0.0)
        right = bc_values.get('right', 0.0)
        
        phi[0, :] = bottom  # Bottom boundary
        phi[-1, :] = top    # Top boundary
        phi[:, 0] = left    # Left boundary
        phi[:, -1] = right  # Right boundary
        
        return phi

    def _apply_neumann_bc(self, phi: cp.ndarray, bc_values: dict) -> cp.ndarray:
        """Apply Neumann boundary conditions using finite differences."""
        dx, dy = self.dx, self.dy
        
        # Extract boundary derivatives
        dtop = bc_values.get('top', 0.0)
        dbottom = bc_values.get('bottom', 0.0)
        dleft = bc_values.get('left', 0.0)
        dright = bc_values.get('right', 0.0)
        
        # Apply Neumann conditions using one-sided differences
        phi[0, :] = phi[1, :] - dy * dbottom  # Bottom boundary
        phi[-1, :] = phi[-2, :] + dy * dtop   # Top boundary
        phi[:, 0] = phi[:, 1] - dx * dleft    # Left boundary
        phi[:, -1] = phi[:, -2] + dx * dright # Right boundary
        
        return phi

    def _extend_domain(self, rho: cp.ndarray) -> cp.ndarray:
        """Extend the domain for non-periodic boundary conditions."""
        if self.boundary_type == BoundaryCondition.PERIODIC:
            return rho
            
        rho_ext = cp.zeros((self.m_ext, self.n_ext))
        rho_ext[:self.m, :self.n] = rho
        
        if self.boundary_type == BoundaryCondition.DIRICHLET:
            # Anti-symmetric extension for Dirichlet
            rho_ext[self.m:, :self.n] = -cp.flip(rho, axis=0)
            rho_ext[:, self.n:] = -cp.flip(rho_ext[:, :self.n], axis=1)
        else:  # Neumann
            # Symmetric extension for Neumann
            rho_ext[self.m:, :self.n] = cp.flip(rho, axis=0)
            rho_ext[:, self.n:] = cp.flip(rho_ext[:, :self.n], axis=1)
            
        return rho_ext

    def solve(self, rho: cp.ndarray, bc_values: Optional[dict] = None) -> Tuple[cp.ndarray, float]:
        """
        Solve the Poisson equation with specified boundary conditions.
        
        Args:
            rho: Charge density array
            bc_values: Dictionary of boundary values/derivatives:
                      For Dirichlet: {'top': val, 'bottom': val, 'left': val, 'right': val}
                      For Neumann: {'top': dval, 'bottom': dval, 'left': dval, 'right': dval}
        
        Returns:
            Tuple of (potential array, maximum error estimate)
        """
        #if isinstance(rho, np.ndarray):
        #    rho = cp.array(rho)
        rho_2d = rho.reshape(self.m, self.n)
        
        if self.boundary_type != BoundaryCondition.PERIODIC:
            # Extend domain for non-periodic BCs
            rho_ext = self._extend_domain(rho_2d)
            
            if self.boundary_type == BoundaryCondition.DIRICHLET:
                # Use DST (Discrete Sine Transform)
                phi_k = cp.fft.dstn(rho_ext / self.epsilon0, type=1)
            else:  # Neumann
                # Use DCT (Discrete Cosine Transform)
                phi_k = cp.fft.dctn(rho_ext / self.epsilon0, type=1)
        else:
            # Standard FFT for periodic BC
            phi_k = cp.fft.fftn(rho_2d / self.epsilon0)
        
        # Solve in frequency domain
        phi_k = -phi_k / self.k_sq
        
        # Transform back to real space
        if self.boundary_type == BoundaryCondition.DIRICHLET:
            phi = cp.fft.idstn(phi_k, type=1)[:self.m, :self.n]
        elif self.boundary_type == BoundaryCondition.NEUMANN:
            phi = cp.fft.idctn(phi_k, type=1)[:self.m, :self.n]
        else:
            phi = cp.fft.ifftn(phi_k).real
        
        # Apply boundary conditions
        if bc_values is not None:
            if self.boundary_type == BoundaryCondition.DIRICHLET:
                phi = self._apply_dirichlet_bc(phi, bc_values)
            elif self.boundary_type == BoundaryCondition.NEUMANN:
                phi = self._apply_neumann_bc(phi, bc_values)
        
        # Compute error estimate
        residual = self._compute_residual(phi, rho_2d)
        max_error = cp.max(cp.abs(residual))
        
        return phi.ravel(), max_error

    def _compute_residual(self, phi: cp.ndarray, rho: cp.ndarray) -> cp.ndarray:
        """Compute the residual of the solution."""
        # Use finite differences for Laplacian to account for BCs
        dx2, dy2 = self.dx**2, self.dy**2
        
        laplacian = (
            (phi[:-2, 1:-1] - 2*phi[1:-1, 1:-1] + phi[2:, 1:-1]) / dx2 +
            (phi[1:-1, :-2] - 2*phi[1:-1, 1:-1] + phi[1:-1, 2:]) / dy2
        )
        
        return laplacian + rho[1:-1, 1:-1] / self.epsilon0