import numpy as np
import cupy as cp
#from numba import cuda, float32, int32

def update_R(R, V, X, Y, dt:float, last_alive, part_type):
    R[:, :last_alive] = R[:, :last_alive] + V[:, :last_alive] * dt


def compute_bilinear_weights(R: cp.ndarray, cell_size: tuple,
                             gridsize: tuple, last_alive: int, weights: cp.ndarray, indices: cp.ndarray,
                             R_grid: cp.ndarray = None):
    """
    Computes bilinear interpolation weights and cell indices for particles in place.
    
    Args:
        R: Particle positions array (2 x N)
        dx, dy: Grid spacing
        gridsize: Grid dimensions (m, n)
        last_alive: Number of active particles
        weights: Array to store interpolation weights for each particle (4 x last_alive)
        indices: Array to store cell indices for each particle (4 x last_alive)
    """
    m, n = gridsize
    dx, dy = cell_size
    # Normalize positions
    x = R[0, :last_alive] / dx
    y = R[1, :last_alive] / dy
    
    # Calculate indices
    x0 = cp.floor(x).astype(cp.int32)
    y0 = cp.floor(y).astype(cp.int32)
    x1 = cp.minimum(x0 + 1, m - 1)
    y1 = cp.minimum(y0 + 1, n - 1)

    # Calculate weights
    wx = x - x0
    wy = y - y0
    wx_1 = 1.0 - wx
    wy_1 = 1.0 - wy
    
    # Update weights in place
    weights[0, :last_alive] = wx_1 * wy_1  # bottom-left
    weights[1, :last_alive] = wx * wy_1    # bottom-right
    weights[2, :last_alive] = wx_1 * wy    # top-left
    weights[3, :last_alive] = wx * wy      # top-right
    
    # Update indices in place
    indices[0, :last_alive] = y0 * n + x0  # bottom-left
    indices[1, :last_alive] = y0 * n + x1  # bottom-right
    indices[2, :last_alive] = y1 * n + x0  # top-left
    indices[3, :last_alive] = y1 * n + x1  # top-right

    if R_grid is not None:
        R_grid[:last_alive] = indices[0, :last_alive] - indices[0, :last_alive]//m

def update_V(V: cp.ndarray, E: cp.ndarray,
             part_type: cp.ndarray, q_type: cp.ndarray, m_type_1: cp.ndarray,
             dt: float,
             last_alive: int, weights: cp.ndarray, indices: cp.ndarray):
    """
    Updates particle velocities using precomputed interpolation weights.
    """
    # Get interpolation weights and indices
    #weights, indices = compute_bilinear_weights(R, dx, dy, gridsize, last_alive)
    # Compute interpolated electric field components
    Ex = cp.sum(E[0, indices[:, :last_alive]] * weights[:, :last_alive], axis=0)
    Ey = cp.sum(E[1, indices[:, :last_alive]] * weights[:, :last_alive], axis=0)
    
    # Update velocities
    k = dt * q_type[part_type[:last_alive]] * m_type_1[part_type[:last_alive]]
    
    V[0, :last_alive] += Ex * k
    V[1, :last_alive] += Ey * k

def update_V_3d(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y, Z):
    m, n, p = gridsize
    
    # Normalize particle positions
    x = R[0] / X * (m - 1)
    y = R[1] / Y * (n - 1)
    z = R[2] / Z * (p - 1)
    
    # Calculate indices
    x0 = cp.floor(x).astype(cp.int32)
    y0 = cp.floor(y).astype(cp.int32)
    z0 = cp.floor(z).astype(cp.int32)
    
    x1 = cp.minimum(x0 + 1, m - 1)
    y1 = cp.minimum(y0 + 1, n - 1)
    z1 = cp.minimum(z0 + 1, p - 1)
    
    # Calculate weights
    wx = x - x0
    wy = y - y0
    wz = z - z0
    
    # Calculate 1D indices for the eight surrounding points
    idx000 = (z0 * n + y0) * m + x0
    idx100 = (z0 * n + y0) * m + x1
    idx010 = (z0 * n + y1) * m + x0
    idx110 = (z0 * n + y1) * m + x1
    
    idx001 = (z1 * n + y0) * m + x0
    idx101 = (z1 * n + y0) * m + x1
    idx011 = (z1 * n + y1) * m + x0
    idx111 = (z1 * n + y1) * m + x1
    
    # Perform trilinear interpolation for all components of E
    Ex = (E[0, idx000] * (1-wx) * (1-wy) * (1-wz) +
          E[0, idx100] * wx * (1-wy) * (1-wz) +
          E[0, idx010] * (1-wx) * wy * (1-wz) +
          E[0, idx110] * wx * wy * (1-wz) +
          E[0, idx001] * (1-wx) * (1-wy) * wz +
          E[0, idx101] * wx * (1-wy) * wz +
          E[0, idx011] * (1-wx) * wy * wz +
          E[0, idx111] * wx * wy * wz)
    
    Ey = (E[1, idx000] * (1-wx) * (1-wy) * (1-wz) +
          E[1, idx100] * wx * (1-wy) * (1-wz) +
          E[1, idx010] * (1-wx) * wy * (1-wz) +
          E[1, idx110] * wx * wy * (1-wz) +
          E[1, idx001] * (1-wx) * (1-wy) * wz +
          E[1, idx101] * wx * (1-wy) * wz +
          E[1, idx011] * (1-wx) * wy * wz +
          E[1, idx111] * wx * wy * wz)
    
    Ez = (E[2, idx000] * (1-wx) * (1-wy) * (1-wz) +
          E[2, idx100] * wx * (1-wy) * (1-wz) +
          E[2, idx010] * (1-wx) * wy * (1-wz) +
          E[2, idx110] * wx * wy * (1-wz) +
          E[2, idx001] * (1-wx) * (1-wy) * wz +
          E[2, idx101] * wx * (1-wy) * wz +
          E[2, idx011] * (1-wx) * wy * wz +
          E[2, idx111] * wx * wy * wz)
    
    # Update velocities (now in 3D)
    V[0] += Ex * q_type[part_type] * m_type_1[part_type] * dt 
    V[1] += Ey * q_type[part_type] * m_type_1[part_type] * dt
    V[2] += Ez * q_type[part_type] * m_type_1[part_type] * dt

def update_density_gpu(part_type: cp.ndarray, rho: cp.ndarray,
                       q: cp.ndarray,
                       last_alive: int, weights: cp.ndarray, indices: cp.ndarray, w: float = 1.0):
    """
    Updates charge density field using precomputed bilinear interpolation.
    """
    rho.fill(0.0)
    
    cp.add.at(rho, indices[0, :last_alive], w * q[part_type[:last_alive]] * weights[0, :last_alive])
    cp.add.at(rho, indices[1, :last_alive], w * q[part_type[:last_alive]] * weights[1, :last_alive])
    cp.add.at(rho, indices[2, :last_alive], w * q[part_type[:last_alive]] * weights[2, :last_alive])
    cp.add.at(rho, indices[3, :last_alive], w * q[part_type[:last_alive]] * weights[3, :last_alive])

def update_current_density_gpu(
    R: cp.ndarray, V: cp.ndarray, part_type: cp.ndarray, J: cp.ndarray,
    X: float, Y: float, gridsize: tuple, q: cp.ndarray, w=1
):
    m, n = gridsize
    dx = X / (m - 1)
    dy = Y / (n - 1)
    dV_1 = 1 / (dx * dy)
    J[0].fill(0)
    J[1].fill(0)

    # Map particle positions to grid indices
    I = R[0] / X * (m - 1)
    K = R[1] / Y * (n - 1)
    i = cp.floor(I).astype(cp.int32)
    j = cp.floor(K).astype(cp.int32)

    # Interpolation weights
    fx1 = I - i
    fy1 = K - j
    fx0 = 1 - fx1
    fy0 = 1 - fy1

    # Flattened grid indices
    k1 = j * n + i
    k2 = k1 + 1
    k3 = k1 + n
    k4 = k3 + 1

    # Compute current density contributions from each particle
    qv_x = w * q[part_type] * V[0] * dV_1
    qv_y = w * q[part_type] * V[1] * dV_1

    # Distribute current density to grid nodes with bilinear interpolation
    cp.add.at(J[0], k1, qv_x * fx0 * fy0)
    cp.add.at(J[0], k2, qv_x * fx1 * fy0)
    cp.add.at(J[0], k3, qv_x * fx0 * fy1)
    cp.add.at(J[0], k4, qv_x * fx1 * fy1)

    cp.add.at(J[1], k1, qv_y * fx0 * fy0)
    cp.add.at(J[1], k2, qv_y * fx1 * fy0)
    cp.add.at(J[1], k3, qv_y * fx0 * fy1)
    cp.add.at(J[1], k4, qv_y * fx1 * fy1)

def update_density_gpu_3d(R: cp.ndarray, part_type: cp.ndarray, rho: cp.ndarray,
                          X: float, Y: float, Z: float, gridsize: tuple, q: cp.ndarray, w = 1) -> None:
    m, n, p = gridsize
    dx = X / (m - 1)
    dy = Y / (n - 1)
    dz = Z / (p - 1)
    dV_1 = 1 / (dx * dy * dz)
    rho.fill(0)
    
    I = (R[0] / X * (m - 1))
    J = (R[1] / Y * (n - 1))
    K = (R[2] / Z * (p - 1))
    
    i = cp.floor(I).astype(cp.int32)
    j = cp.floor(J).astype(cp.int32)
    k = cp.floor(K).astype(cp.int32)
    
    fx1 = I - i
    fy1 = J - j
    fz1 = K - k
    fx0 = 1 - fx1
    fy0 = 1 - fy1
    fz0 = 1 - fz1
    
    k1 = (k * n + j) * m + i
    k2 = k1 + 1
    k3 = k1 + m
    k4 = k3 + 1
    k5 = k1 + m * n
    k6 = k5 + 1
    k7 = k5 + m
    k8 = k7 + 1
    
    charge_density = w * q[part_type] * dV_1
    
    cp.add.at(rho, k1, charge_density * fx0 * fy0 * fz0)
    cp.add.at(rho, k2, charge_density * fx1 * fy0 * fz0)
    cp.add.at(rho, k3, charge_density * fx0 * fy1 * fz0)
    cp.add.at(rho, k4, charge_density * fx1 * fy1 * fz0)
    cp.add.at(rho, k5, charge_density * fx0 * fy0 * fz1)
    cp.add.at(rho, k6, charge_density * fx1 * fy0 * fz1)
    cp.add.at(rho, k7, charge_density * fx0 * fy1 * fz1)
    cp.add.at(rho, k8, charge_density * fx1 * fy1 * fz1)

def updateE_gpu(E, phi, x, y, gridsize: tuple):
    # Reshape phi into a 2D grid using Fortran-order for better y-direction performance
    phi_grid = cp.reshape(phi, gridsize, order='C')  # Use Fortran-order ('F') for column-major memory layout
    
    # Compute grid spacing
    dx = x / (gridsize[0] - 1)
    dy = y / (gridsize[1] - 1)
    
    # Compute gradients in the x and y directions
    E_x = -cp.gradient(phi_grid, dx, axis=1)  # Gradient in the x-direction (axis 1)
    E_y = -cp.gradient(phi_grid, dy, axis=0)  # Gradient in the y-direction (axis 0)
    
    # Flatten the results and assign to E
    E[0, :] = E_x.flatten()  # Flatten using the same Fortran-order to match the reshaping
    E[1, :] = E_y.flatten()
    
def sort_particles_sparse(cell_indices, num_cells):
    """
    Sort particles using a sparse array approach on GPU.
    Only stores data for non-empty cells and computes cell shifts.
   
    Parameters:
    -----------
    cell_indices : cupy.ndarray
        Array of cell indices for each particle
    num_cells : int
        Total number of cells
       
    Returns:
    --------
    active_cells : cupy.ndarray
        Indices of cells that contain particles
    cell_starts : cupy.ndarray
        Starting position for each active cell in the sorted array
    sorted_indices : cupy.ndarray
        Sorted particle indices
    """
    # Get counts for all cells
    full_counts = cp.bincount(cell_indices, minlength=num_cells)
   
    # Find non-empty cells
    active_cells = cp.nonzero(full_counts)[0]
    cell_counts = full_counts[active_cells]
   
    # Compute starting positions for active cells
    cell_starts = cp.zeros(len(active_cells), dtype=cp.int32)
    if len(active_cells) > 1:
        cp.cumsum(cell_counts[:-1], out=cell_starts[1:])
    
    # Create a mapping from all cells to active cell positions
    cell_to_active_map = -cp.ones(num_cells, dtype=cp.int32)
    cell_to_active_map[active_cells] = cp.arange(len(active_cells))
    
    # Temporary positions for atomic updates
    temp_positions = cp.zeros_like(cell_starts)
    
    # Create array for sorted particle indices
    sorted_indices = cp.zeros(len(cell_indices), dtype=cp.int32)
   
    # Create a kernel for parallel particle sorting
    sort_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void sort_particles_sparse(const int* cell_indices,
                             const int* cell_to_active_map,
                             const int* cell_positions,
                             int* temp_positions,
                             int* sorted_indices,
                             const int num_particles) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < num_particles) {
            // Map the cell index to the active cell index
            int cell = cell_indices[idx];
            int active_cell_idx = cell_to_active_map[cell];
            if (active_cell_idx >= 0) {
                // Atomically update temp_positions and compute position
                int pos = atomicAdd(&temp_positions[active_cell_idx], 1);
                int global_pos = cell_positions[active_cell_idx] + pos;
                // Assign the particle index to the sorted array
                sorted_indices[global_pos] = idx;
            }
        }
    }
    ''', 'sort_particles_sparse')
       
    # Kernel execution with error checking
    threads_per_block = 256
    blocks_per_grid = (len(cell_indices) + threads_per_block - 1) // threads_per_block
    
    try:
        sort_kernel((blocks_per_grid,), (threads_per_block,),
                   (cell_indices, cell_to_active_map, cell_starts, temp_positions,
                    sorted_indices, len(cell_indices)))
    except Exception as e:
        raise RuntimeError(f"Kernel execution failed: {str(e)}")
        
    return active_cells, cell_starts, sorted_indices


def sort_particles_dict(cell_indices):
    """
    Sort particles using dictionary method, with GPU-friendly implementation.
    Better for sparse distributions.
    
    Parameters:
    -----------
    cell_indices : cupy.ndarray
        Array of cell indices for each particle
        
    Returns:
    --------
    unique_cells : cupy.ndarray
        Array of unique cell indices
    cell_starts : cupy.ndarray
        Starting index for each cell
    sorted_indices : cupy.ndarray
        Sorted particle indices
    """
    # Sort particles by cell index
    sorted_order = cp.argsort(cell_indices)
    sorted_cells = cell_indices[sorted_order]
    
    # Find unique cells and their starting positions
    unique_cells, cell_starts = cp.unique(sorted_cells, return_index=True)
    
    return unique_cells, cell_starts, sorted_order

def updateE_gpu_3d(E, phi, x, y, z, gridsize: tuple):
    # Reshape phi into a 3D grid (Fortran-order for better performance in the z-direction)
    phi_grid = cp.reshape(phi, gridsize, order='C')  # Column-major memory layout (C-order)
    
    # Compute grid spacing in x, y, and z directions
    dx = x / (gridsize[0] - 1)
    dy = y / (gridsize[1] - 1)
    dz = z / (gridsize[2] - 1)
    
    # Compute gradients in the x, y, and z directions
    E_x = -cp.gradient(phi_grid, dx, axis=2)  # Gradient in the x-direction (axis 2 in 3D)
    E_y = -cp.gradient(phi_grid, dy, axis=1)  # Gradient in the y-direction (axis 1 in 3D)
    E_z = -cp.gradient(phi_grid, dz, axis=0)  # Gradient in the z-direction (axis 0 in 3D)
    
    # Flatten the results and assign to E
    E[0, :] = E_x.flatten(order='C')  # Flatten along the x-direction
    E[1, :] = E_y.flatten(order='C')  # Flatten along the y-direction
    E[2, :] = E_z.flatten(order='C')  # Flatten along the z-direction

def update_B_gpu(B, A, dx, dy, gridsize: tuple):
    """
    Compute the magnetic field component Bz from the vector potential components
    Ax and Ay on a 2D grid using finite difference approximation.

    Parameters:
    - Ax, Ay: CuPy arrays representing the vector potential components (2D arrays)
    - dx, dy: Grid spacing in the x and y directions

    Returns:
    - Bz: CuPy array representing the magnetic field component Bz
    """
    # Compute partial derivatives using finite differences
    dAy_dx = (cp.roll(A[1], -1, axis=0) - cp.roll(A[1], 1, axis=0)) / (2 * dx)
    dAx_dy = (cp.roll(A[0], -1, axis=1) - cp.roll(A[0], 1, axis=1)) / (2 * dy)
    
    # Compute magnetic field component Bz
    B[:] = dAy_dx - dAx_dy

def kinetic_energy(V, M, part_type):
    return 0.5 * (V[0]**2 + V[1]**2) * M[part_type]

def kinetic_energy_ev(V, M, part_type):
    return 0.5 * (V[0]**2 + V[1]**2) * M[part_type] * 6.242e18

def total_kinetic_energy(v:cp.ndarray, M_type:cp.ndarray, part_type:cp.ndarray, last_alive:int) -> float:
    # Compute kinetic energy element-wise and sum
    return 0.5 * cp.sum((v[0, :last_alive] ** 2 + v[1, :last_alive] ** 2) * M_type[part_type[:last_alive]])

def total_potential_energy(rho: cp.ndarray, phi: cp.ndarray, dx: float, dy: float) -> float:
    dV = dx * dy
    energy_density = 0.5 * rho * phi
    total_potential_energy = cp.sum(energy_density) * dV
    return total_potential_energy

def total_momentum(v:cp.ndarray, M:cp.ndarray, part_type:cp.ndarray, last_alive:int):
    return cp.sum(cp.hypot(v[0, :last_alive], v[1, :last_alive])*M[part_type[:last_alive]])

def KE_distribution(part_type, v:cp.ndarray, M:cp.ndarray, bins:int) -> list:
    E = (v[0]**2 + v[1]**2)*M[part_type]*0.5
    x = np.arange(bins)*cp.asnumpy(cp.max(E))/bins
    return (x, cp.asnumpy(cp.histogram(E, bins, density=True)[0]))

def V_distribution(v:cp.ndarray, bins:int) -> list:
    x = np.arange(bins)
    return (x, cp.asnumpy(cp.histogram(cp.hypot(v[0], v[1]), bins, density=True)[0]))

def P_distribution(v:cp.ndarray, part_type:cp.ndarray, M:cp.ndarray, bins:int) -> list:
    x = np.arange(bins)
    return (x, cp.asnumpy(cp.histogram(cp.hypot(v[0], v[1]) * M[part_type], bins, density=True)[0]))

def Vx_distribution(v:cp.ndarray, bins:int) -> list:
    x = np.arange(bins)
    return (x, cp.asnumpy(cp.histogram(v[0], bins, density=True)[0]))

def Vy_distribution(v:cp.ndarray, bins:int) -> list:
    x = np.arange(bins)
    return (x, cp.asnumpy(cp.histogram(v[1], bins, density=True)[0]))

def update_history(history:np.ndarray, inp):
    history = np.roll(history, -1)
    history[-1] = inp
    return history

def check_gauss_law_2d(E, rho, epsilon_0, dx, dy, nx, ny):
    """
    Check Gauss's law for a 2D grid using cupy arrays.
    
    Parameters:
    E (cupy.ndarray): Electric field with shape (2, nx*ny), E[0] is Ex and E[1] is Ey.
    rho (cupy.ndarray): Charge density with shape (nx*ny).
    epsilon_0 (float): Permittivity of free space.
    dx, dy (float): Grid spacing in x and y directions.
    nx, ny (int): Number of grid points in x and y directions.
    
    Returns:
    tuple: line_integral, area_integral, relative_error
    """
    # Reshape E and rho to 2D grids
    Ex = E[0].reshape(nx, ny)
    Ey = E[1].reshape(nx, ny)
    rho_2d = rho.reshape(nx, ny)
    
    # Compute line integral of the electric field (flux through boundaries)
    line_integral = (
        cp.sum(Ex[0, :]) * dy - cp.sum(Ex[-1, :]) * dy +  # Top and bottom boundaries
        cp.sum(Ey[:, -1]) * dx - cp.sum(Ey[:, 0]) * dx    # Right and left boundaries
    )
    
    # Compute area integral of charge density (total charge divided by epsilon_0)
    total_charge = cp.sum(rho_2d) * dx * dy
    area_integral = total_charge / epsilon_0
    
    # Compute relative error, handle zero area_integral
    if abs(area_integral) > 1e-12:  # Avoid division by near-zero values
        relative_error = abs(line_integral - area_integral) / abs(area_integral)
    else:
        relative_error = float('inf')  # Undefined error if area_integral is zero
    
    return line_integral, area_integral, relative_error



def compute_divergence_error(E, rho, epsilon_0, dx, dy, nx, ny):
    """
    Compute the norm of the error between divergence of E and rho / epsilon_0,
    and the surface integral of the divergence over the domain.
    
    Parameters:
    E (cupy.ndarray): Electric field with shape (2, nx*ny), E[0] is Ex and E[1] is Ey.
    rho (cupy.ndarray): Charge density with shape (nx*ny).
    epsilon_0 (float): Permittivity of free space.
    dx, dy (float): Grid spacing in x and y directions.
    nx, ny (int): Number of grid points in x and y directions.
    
    Returns:
    tuple: (error_norm, surface_integral, boundary_flux)
        - error_norm (float): L2 norm of the error.
        - surface_integral (float): Integral of divergence of E over the domain.
        - boundary_flux (float): Line integral of E field over the domain boundary.
    """
    # Reshape E and rho to 2D grids
    Ex = E[0].reshape(nx, ny)
    Ey = E[1].reshape(nx, ny)
    rho_2d = rho.reshape(nx, ny)
    
    # Compute divergence of E
    divE = cp.zeros((nx, ny), dtype=cp.float32)
    
    # Central differences for interior points
    divE[1:-1, 1:-1] = (
        (Ex[2:, 1:-1] - Ex[:-2, 1:-1]) / (2 * dx) + 
        (Ey[1:-1, 2:] - Ey[1:-1, :-2]) / (2 * dy)
    )
    
    # One-sided differences for boundaries
    divE[0, :] = (Ex[1, :] - Ex[0, :]) / dx  # Bottom
    divE[-1, :] = (Ex[-1, :] - Ex[-2, :]) / dx  # Top
    divE[:, 0] = (Ey[:, 1] - Ey[:, 0]) / dy  # Left
    divE[:, -1] = (Ey[:, -1] - Ey[:, -2]) / dy  # Right
    
    # Compute the error
    error = divE - rho_2d / epsilon_0
    
    # Compute the L2 norm of the error
    error_norm = cp.sqrt(cp.sum(error**2))
    
    # Compute the surface integral of divergence
    surface_integral = cp.sum(divE) * dx * dy
    
    return error_norm, surface_integral