import numpy as np
import cupy as cp

def update_R(R, V, dt:tuple):
    i = cp.arange(len(R[0]), dtype=cp.int32)
    R[:, i] = R[:, i] + V[:, i] * dt

def update_V(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y):
    m, n = gridsize
    
    # Normalize particle positions
    x = R[0] / X * (m - 1)
    y = R[1] / Y * (n - 1)
    
    # Calculate indices
    x0 = cp.floor(x).astype(cp.int32)
    y0 = cp.floor(y).astype(cp.int32)
    x1 = cp.minimum(x0 + 1, m - 1)
    y1 = cp.minimum(y0 + 1, n - 1)
    
    # Calculate weights
    wx = x - x0
    wy = y - y0
    
    # Calculate 1D indices for the four surrounding points
    idx00 = y0 * m + x0
    idx10 = y0 * m + x1
    idx01 = y1 * m + x0
    idx11 = y1 * m + x1
    
    # Perform bilinear interpolation for both components of E
    Ex = (E[0, idx00] * (1-wx) * (1-wy) +
          E[0, idx10] * wx * (1-wy) +
          E[0, idx01] * (1-wx) * wy +
          E[0, idx11] * wx * wy)
    
    Ey = (E[1, idx00] * (1-wx) * (1-wy) +
          E[1, idx10] * wx * (1-wy) +
          E[1, idx01] * (1-wx) * wy +
          E[1, idx11] * wx * wy)
    
    # Update velocities
    V[0] += Ex * q_type[part_type] * m_type_1[part_type] * dt 
    V[1] += Ey * q_type[part_type] * m_type_1[part_type] * dt


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


def push_gpu_rk4(R, V, E, part_type, M, gridsize:tuple, dt):
    pass

def push_gpu_Yoshida(R, V, E, part_type, M, gridsize:tuple, dt):
    pass



def update_density_gpu(R:cp.ndarray, part_type:cp.ndarray, rho:cp.ndarray, X:float, Y:float, gridsize:tuple, q:cp.ndarray, w = 1):
    m, n = gridsize
    dx = X/(m-1)
    dy = Y/(n-1)
    dV_1 = 1/(dx*dy)
    rho.fill(0)

    I = (R[0]/X * (m-1))
    J = (R[1]/Y * (n-1))
    i = cp.floor(I).astype(cp.int32)
    j = cp.floor(J).astype(cp.int32)

    fx1 = I - i
    fy1 = J - j
    fx0 = 1 - fx1
    fy0 = 1 - fy1

    k1 = j*n + i
    k2 = k1 + 1
    k3 = k1 + n
    k4 = k3 + 1

    charge_density = w * q[part_type] * dV_1
    cp.add.at(rho, k1, charge_density * fx0 * fy0)
    cp.add.at(rho, k2, charge_density * fx1 * fy0)
    cp.add.at(rho, k3, charge_density * fx0 * fy1)
    cp.add.at(rho, k4, charge_density * fx1 * fy1)

    #print(f"Max charge density: {cp.max(rho)}, Min: {cp.min(rho)}, Mean: {cp.mean(rho)}")

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

def updateE_fft(phi_k, kx, ky):
    Ex_k = -1j * kx * phi_k
    Ey_k = -1j * ky * phi_k
    Ex = cp.fft.ifftn(Ex_k).real
    Ey = cp.fft.ifftn(Ey_k).real
    return cp.array([Ex.ravel(), Ey.ravel()])

def kinetic_energy(V, M, part_type):
    return 0.5 * (V[0]**2 + V[1]**2) * M[part_type]

def update_crossection(E):
    return 1

def MCC(R, V, part_type, M, NGD, P, dt):

    E = 0.5 * (V[0]**2 + V[1]**2) * M[part_type]
    v = cp.hypot(V[0], V[1])
    sigma = update_crossection(E)
    P[:] = 1 - cp.exp(-dt * NGD * sigma * v)



def compute_probability_distribution(probabilities, num_bins=50):
    """
    Compute the x (bin edges) and y (counts) arrays for the probability distribution.

    Parameters:
    probabilities (cupy.ndarray): Array of precomputed collision probabilities.
    num_bins (int): Number of bins for the histogram.

    Returns:
    bin_centers (numpy.ndarray): The centers of the histogram bins (x values).
    hist (numpy.ndarray): The histogram counts (y values).
    """
    # Convert CuPy array to NumPy for histogram computation
    probabilities_np = cp.asnumpy(probabilities)

    # Compute histogram
    hist, bin_edges = np.histogram(probabilities_np, bins=num_bins, density=True)

    # Compute bin centers from bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, hist


def total_kinetic_energy(v:cp.ndarray, M_type:cp.ndarray, part_type:cp.ndarray):
    return 0.5*cp.dot((v[0]**2 + v[1]**2), M_type[part_type])

def total_potential_energy(rho: cp.ndarray, phi: cp.ndarray, dx: float, dy: float) -> float:
    dV = dx * dy
    energy_density = 0.5 * rho * phi
    total_potential_energy = cp.sum(energy_density) * dV
    return total_potential_energy

def total_momentum(v:cp.ndarray, M:cp.ndarray, part_type:cp.ndarray):
    return cp.sum(cp.hypot(v[0], v[1])*M[part_type])

def KE_distribution(part_type, v:cp.ndarray, M:cp.ndarray, bins:int) -> list:
    E = (v[0]**2 + v[1]**2)*M[part_type]*0.5
    x = np.arange(bins)*cp.asnumpy(cp.max(E))/bins
    return (x, cp.asnumpy(cp.histogram(E, bins, density=True)[0]))


def V_distribution(v:cp.ndarray, bins:int) -> list:
    x = np.arange(bins)
    return (x, cp.asnumpy(cp.histogram(cp.hypot(v[0], v[1]), bins, density=True)[0]))

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


def coulumb_collision(rho:cp.ndarray, v:cp.ndarray, gridsize:tuple):
    pass
    

def boundary_field_flux(E:cp.ndarray, gridsize:tuple, X:float, Y:float):
    m, n = gridsize
    dx = X/(m - 1)
    dy = Y/(n - 1)

    F = (-cp.sum(cp.hypot(E[0, :], E[1, :]))*dx  # bottom (negative y-direction)
         + cp.sum(E[1, m*(n-1):])*dx  # top (positive y-direction)
         - cp.sum(E[0, ::m])*dy  # left (negative x-direction)
         + cp.sum(E[0, m-1::m])*dy)  # right (positive x-direction)

    return F

def check_gauss_law_2d(E, rho, epsilon_0, dx, dy, nx, ny):
    """
    Check Gauss's law for a 2D grid using cupy arrays.
    
    Parameters:
    E (cupy.ndarray): Electric field with shape (2, nx*ny)
    rho (cupy.ndarray): Charge density with shape (nx*ny)
    epsilon_0 (float): Permittivity of free space
    dx, dy (float): Grid spacing in x and y directions
    nx, ny (int): Number of grid points in x and y directions
    
    Returns:
    tuple: line_integral, area_integral, relative_error
    """
    # Reshape E and rho to 2D grid
    Ex = E[0].reshape(nx, ny)
    Ey = E[1].reshape(nx, ny)
    rho_2d = rho.reshape(nx, ny)
    
    # Compute line integral of E field
    line_integral = (
        cp.sum(Ex[0, :] - Ex[-1, :]) * dy +
        cp.sum(Ey[:, 0] - Ey[:, -1]) * dx
    )
    
    # Compute area integral of charge density
    total_charge = cp.sum(rho_2d) * dx * dy
    area_integral = total_charge / epsilon_0
    
    # Compute relative error
    relative_error = abs(line_integral - area_integral) / area_integral
    
    return line_integral, area_integral, relative_error