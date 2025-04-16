import cupy as cp
import numpy as np
from cupyx.scipy.sparse import diags
from cupyx.scipy.sparse.linalg import cg, lsmr, LinearOperator, spilu, spsolve, gmres
import Consts
import Grid


def boundary_conditions_left_gpu(A:cp.ndarray, boundary:cp.ndarray):
    for i in boundary:
        A[i,:] = 0
        A[i, i] = 1


def Laplacian_square(m, n, dx, dy) -> cp.ndarray:

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

    bot = cp.arange(m)
    top = cp.arange(m*(n-1), m*n)
    left = cp.arange(0, m*n, m)
    right = cp.arange(m-1, m*n, m)
    Lap[bot, :] = 0
    Lap[:, bot] = 0
    Lap[bot, bot] = 1
    Lap[top, :] = 0
    Lap[:, top] = 0
    Lap[top, top] = 1
    Lap[left, :] = 0
    Lap[:, left] = 0
    Lap[left, left] = 1
    Lap[right, :] = 0
    Lap[:, right] = 0
    Lap[right, right] = 1
                
    return Lap.astype(cp.float32)

def Laplacian_cylindrical(m, n, dr):
    
    Lap = cp.zeros((m*n, m*n), dtype=cp.float32)




def Laplacian_cartesian_csr(m, n, dx, dy):
    """
    Constructs a discrete Laplacian operator in Cartesian coordinates using a
    sparse CSR matrix representation, suitable for GPU computation with CuPy.

    Args:
        m (int): Number of grid points in the y-direction.
        n (int): Number of grid points in the x-direction.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.

    Returns:
        cupyx.scipy.sparse.csr_matrix: A sparse CSR matrix representing the Laplacian operator.
    """

    if dx == 0 or dy == 0:
        raise ValueError("Grid spacing must be non-zero.")
    elif dx != dy:
        raise ValueError("Grid spacing in x and y directions must be equal for gmres and cg solvers.")

    size = m * n

    # Main diagonal
    main_diag = cp.ones(size) * -4 #(-2.0/dx**2 - 2.0/dy**2)

    off_diag_h = cp.ones(size) #/ dx**2
    off_diag_h[m - 1::m] = 0  # Correct row-boundary removal

    off_diag_v = cp.ones(size - m) #/ dy**2  # Should be size - n


    laplacian = diags(
        [main_diag, off_diag_h, off_diag_h, off_diag_v, off_diag_v],
        [0, 1, -1, m, -m],  # Indices correspond to diagonal positions
        (size, size),
        format="csr"
    )
    return laplacian

'''
def Laplacian_cylindrical_csr(m, n, dz, dr) -> cp.ndarray:
    """
    Constructs a discrete Laplacian operator in cylindrical coordinates with axial
    symmetry (no theta dependence) using a sparse CSR matrix representation,
    suitable for GPU computation with CuPy.

    Args:
        m (int): Number of grid points in the z-direction.
        n (int): Number of grid points in the r-direction.
        dr (float): Grid spacing in the r-direction.
        dz (float): Grid spacing in the z-direction.
        r_values (cp.ndarray): 1D array of r values at each radial grid point.

    Returns:
        cp.ndarray: A sparse CSR matrix representing the Laplacian operator.
    """
    size = m * n
    diagonals = []
    r_values = cp.linspace(0, dr * (n - 1), n)
    r_values[0] = dr  # Prevent division by zero at r = 0

    # Main diagonal (Cartesian part)
    main_diag = -4 * cp.ones(size)
    diagonals.append(main_diag)

    # Horizontal off-diagonals (r-direction)
    off_diag_r = cp.ones(size)
    off_diag_r[m - 1::m] = 0  # Remove connections across row boundaries
    diagonals.append(off_diag_r[:-1])
    diagonals.append(off_diag_r[:-1])  # Mirror for lower diagonal

    # Vertical off-diagonals (z-direction) with r term added.
    off_diag_z_upper = cp.ones(size - m) 
    off_diag_z_lower = cp.ones(size - m)

    # Additional term for 1/r (du/dr)
    r_term_upper = cp.zeros(size - m)
    r_term_lower = cp.zeros(size - m)

    for i in range(n - 1):
        for j in range(m):
            index = j + i * m
            if r_values[i] != 0: #prevent division by 0
                r_term_upper[index] = 1 / (2 *  r_values[i]) * dr
                r_term_lower[index] = -1 / (2 *  r_values[i]) * dr

    off_diag_z_upper += r_term_upper
    off_diag_z_lower += r_term_lower

    diagonals.append(off_diag_z_upper)
    diagonals.append(off_diag_z_lower)

    # Diagonal indices (corrected)
    diag_indices = [0, 1, -1, m, -m]

    # Sparse matrix creation
    laplacian = diags(
        diagonals,
        diag_indices,
        shape=(size, size),
        format="csr"
    )
    return laplacian
'''

def Laplacian_cylindrical_csr(m, n, dz, dr) -> cp.ndarray:
    """
    Constructs a discrete Laplacian operator in cylindrical coordinates with axial
    symmetry (no theta dependence) using vectorized operations for performance.
    
    This implementation correctly handles the axis (r=0) to ensure both radial
    and axial components of the field are properly represented.
    
    Args:
        m (int): Number of grid points in the z-direction.
        n (int): Number of grid points in the r-direction.
        dz (float): Grid spacing in the z-direction.
        dr (float): Grid spacing in the r-direction.
   
    Returns:
        cp.ndarray: A sparse CSR matrix representing the Laplacian operator.
    """
    size = m * n
   
    # Pre-allocate arrays for diagonals
    main_diag = cp.ones(size, dtype=cp.float32) * -4  # Diagonal term
   
    # z-direction terms (constant throughout the grid)
    z_upper_diag = cp.ones(size-1, dtype=cp.float32)
    z_upper_diag[m-1::m] = 0  # Remove connections across z boundaries
    z_lower_diag = cp.copy(z_upper_diag)
   
    # r-direction terms (vary with r)
    r_upper_diag = cp.zeros(size-m, dtype=cp.float32)
    r_lower_diag = cp.zeros(size-m, dtype=cp.float32)
   
    # Create array of r indices
    r_indices = cp.arange(1, n)
   
    # Compute the coefficients for upper/lower r diagonals
    for i, r in enumerate(r_indices):
        idx_start = (i+1) * m  # Start index for r=r_indices[i]
       
        # Upper diagonal (r+1)
        if i+1 < n-1:  # If not at the outer boundary
            r_upper_diag[idx_start-m:idx_start] = 1 + 1.0/(2*r/dr)
       
        # Lower diagonal (r-1)
        r_lower_diag[idx_start-m:idx_start] = 1 - 1.0/(2*r/dr)
   
    # Special treatment for r=0 (axis)
    # At the axis, the Laplacian in cylindrical coordinates becomes:
    # ∇²φ = 4∂²φ/∂r² + ∂²φ/∂z²
    
    # For the axis points (r=0)
    main_diag[:m] = -4  # -2 for z, -2 for r
    
    # z-direction terms remain normal at axis
    # z_upper_diag and z_lower_diag already set correctly
    
    # r-direction term at axis - use the limit as r→0
    # Using L'Hôpital's rule and symmetry conditions
    r_upper_diag[:m] = 2  # Connection to r=1
    
    # Assemble diagonals into a sparse matrix
    diagonals = [main_diag, z_upper_diag, z_lower_diag, r_upper_diag, r_lower_diag]
    offsets = [0, 1, -1, m, -m]
   
    laplacian = diags(diagonals, offsets, shape=(size, size), format="csr")
    return laplacian


def apply_boundary_conditions(Laplacian, boundary_nodes):
    """
    Apply Dirichlet boundary conditions directly to a CuPy CSR matrix.
    
    Parameters:
        Laplacian (cupyx.scipy.sparse.csr_matrix): The Laplacian matrix in CSR format.
        boundary_nodes (array-like): Indices of boundary nodes in the flattened grid.
    Returns:
        cupyx.scipy.sparse.csr_matrix: The modified Laplacian matrix.
    """
    boundary_nodes = cp.array(boundary_nodes, dtype=cp.int32)
    
    # Extract CSR components
    data = Laplacian.data
    indices = Laplacian.indices
    indptr = Laplacian.indptr
    
    # Zero out all off-diagonal entries for boundary rows
    for node in boundary_nodes:
        start, end = indptr[node].item(), indptr[node + 1].item()  # Row range
        for i in range(start, end):
            if indices[i] != node:
                data[i] = 0  # Set off-diagonal entries to 0
            else:
                data[i] = 1  # Set diagonal entry to 1
    
    return Laplacian


class Solver():
    def __init__(self, solver_type:str, grid:Grid.Grid2D, cylindrical=False, boundaries=None, tol = 1e-5):
        
        self.solver_type = solver_type

        self.m = grid.m
        self.n = grid.n
        self.dx = grid.dx
        self.dy = grid.dy
        self.cylindrical = cylindrical
        self.tol = tol

        self.boundaries = boundaries
        #print(self.boundaries)

        self.init_solver()


    def jacobi_preconditioner(self, Laplacian):
        pass
        
    def init_solver(self):

        if self.cylindrical and self.solver_type != 'inverse':  # Fix the typo consistently
            print("Using cylindrical coordinates")
            self.Lap = Laplacian_cylindrical_csr(self.m, self.n, self.dx, self.dy)
        elif self.solver_type != 'inverse':
            print("Using Cartesian coordinates")
            self.Lap = Laplacian_cartesian_csr(self.m, self.n, self.dx, self.dy)



        if self.boundaries is not None and self.solver_type != 'inverse':
            print('applying boundary conditions...')
            # Assuming boundaries = [boundary_nodes, boundary_values]
            self.Lap = apply_boundary_conditions(self.Lap, self.boundaries[0])

    
        match self.solver_type:
            case 'inverse': # add matrix compression
                print('creating Laplacian...')
                self.Lap = Laplacian_square(self.m, self.n, self.dx, self.dy)
                #Chek if Lap is positive definite
                cp.linalg.cholesky(self.Lap)
                if self.boundaries != None:
                    print('applying boundary conditions...')
                    boundary_conditions_left_gpu(self.Lap, self.boundaries[0])
                print('creating inverse Laplacian...')
                self.Lap = cp.linalg.inv(self.Lap)

            case 'cg': # add boundary conditions
                print('Using Conjugate Gradient solver')
            case 'gmres':
                print('Using GMRES solver')
                #ilu = spilu(self.Lap.tocsc())
                #M_x = lambda x: ilu.solve(x)
                #self.M = LinearOperator(self.Lap.shape, matvec=M_x) 




    def solve(self, grid: Grid.Grid2D):

        grid.b[:] = -grid.rho * Consts.eps0_1 * self.dx**2
    
        # Apply boundary values to RHS vector if needed
        if self.boundaries is not None:
            boundary_nodes, boundary_values = self.boundaries
            grid.b[boundary_nodes] = boundary_values

        
        match self.solver_type:
            case 'direct':
                grid.phi[:] = spsolve(self.Lap, grid.b)
            case 'inverse':
                grid.phi[:] = cp.dot(self.Lap, grid.b)
            case 'cg':
                grid.phi[:], info = cg(self.Lap, grid.b, x0=grid.phi, tol=self.tol)
                if info > 0:
                    print(f"CG failed to converge after {info} iterations")
                elif info < 0:
                    print(f"CG failed with error code {info}")
            case 'gmres':
                # Use initial guess and capture convergence info
                grid.phi[:], info = gmres(self.Lap, grid.b, x0=grid.phi, tol=self.tol)
                if info > 0:
                    print(f"GMRES failed to converge after {info} iterations")
                elif info < 0:
                    print(f"GMRES failed with error code {info}")

            case None:
                pass




if __name__ == '__main__':
    
    lap = Laplacian_square(10, 10, 1, 1)
    cp.savetxt("lap.txt", lap, fmt='%.2f')