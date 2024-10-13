import numpy as np
import cupy as cp
from numba import jit, njit, prange
import random

class CPU:

    def __init__(self):
        pass

    @jit
    def fmaxw(self):
        return 2*(random.random() + random.random() + random.random() - 1.5)
    
    @njit(parallel=True,fastmath=True)
    def push_cpu(R, V, Ex, Ey, gridsize:tuple, dt):
        m, n = gridsize
        for i in prange(len(R[0])):
            j = int(R[0,i]*m)
            k = int(R[1,i]*n)
            idx = k*m + j
            V[0,i] += Ex[idx]*dt
            V[1,i] += Ey[idx]*dt

        R[:,:] = R + V[0:2] * dt

        for i in prange(len(R[0])):
            if R[0, i] > 1 or R[1, i] > 1 or R[0, i] < 0 or R[1, i] < 0:
                R[0, i] = 0.5
                R[1, i] = 0.5
                V[0, i] = 0
                V[1, i] = 0

    @njit(parallel=True, fastmath=True)
    def update_density(R:np.ndarray, rho:np.ndarray, gridsize:tuple, q):
        rho[:] = 0
        for i in prange(len(R[0])):
            j = int(R[0,i]*gridsize[0])
            k = int(R[1,i]*gridsize[1])
            rho[k*gridsize[0]+j] += q



    #@njit
    def updateE(Ex, Ey, phi, x, y, gridsize:tuple):
        a = np.reshape(phi, gridsize)
        dx = x/gridsize[0]
        dy = y/gridsize[1]
        Ex[:] = -np.gradient(a, dx)[1].flatten()
        Ey[:] = -np.gradient(a, dy)[0].flatten()


    @njit(parallel=True, fastmath=True)
    def update_sigma(sigma:np.ndarray, V:np.ndarray):
        V[2] = np.hypot(V[0], V[1])
        sigma[:] = (-0.8821*np.log(V[2]) + 15.1262) * 10


    @njit(parallel=True,fastmath=True)
    def MCC(self, NGD, sigma, V, R, gridsize, dt):
        for i in prange(len(R[0])):
            idx = int(R[1,i]*gridsize[0]) + int(R[0,i]*gridsize[1])
            x = -NGD[idx]*sigma[i]*V[2,i]*dt
            if np.exp(x) > np.random.random():
                V[0,i] = self.fmaxw()
                V[1,i] = self.fmaxw()


def update_R(R, V, dt:tuple):
    i = cp.arange(len(R[0]), dtype=cp.int32)
    R[:, i] = R[:, i] + V[:, i] * dt

def update_V_kernel(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y):
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
    V[0] += Ex * dt * q_type[part_type] * m_type_1[part_type]
    V[1] += Ey * dt * q_type[part_type] * m_type_1[part_type]

def update_V(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y):
    update_V_kernel(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y)


def push_gpu_rk4(R, V, E, part_type, M, gridsize:tuple, dt):
    pass

def push_gpu_Yoshida(R, V, E, part_type, M, gridsize:tuple, dt):
    pass



def update_density_gpu(R:cp.ndarray, part_type:cp.ndarray, rho:cp.ndarray, X:float, Y:float, gridsize:tuple, q:cp.ndarray, w = 1):
    m, n = gridsize
    dx = X/(m-1)
    dy = Y/(n-1)
    dV = dx*dy
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

    charge_density = w * q[part_type] / dV
    cp.add.at(rho, k1, charge_density * fx0 * fy0)
    cp.add.at(rho, k2, charge_density * fx1 * fy0)
    cp.add.at(rho, k3, charge_density * fx0 * fy1)
    cp.add.at(rho, k4, charge_density * fx1 * fy1)

    #print(f"Max charge density: {cp.max(rho)}, Min: {cp.min(rho)}, Mean: {cp.mean(rho)}")

def updateE_gpu(E, phi, x, y, gridsize:tuple):
    phi_grid = cp.reshape(phi, gridsize, order='C')
    dx = x/(gridsize[0]-1)
    dy = y/(gridsize[1]-1)
    E[0, :] = -cp.gradient(phi_grid, dx, axis=1).flatten()
    E[1, :] = -cp.gradient(phi_grid, dy, axis=0).flatten()
    #E[2, :] = cp.hypot(E[0, :], E[1, :])

    #dx = x / (gridsize[0] - 1)
    #dy = y / (gridsize[1] - 1)
    
    # Central difference for interior points
    #print(cp.shape(E[1, 1:-1]))
    #E[0, 1:-1] = -(phi[2:] - phi[:-2]) / (2 * dx)
    #E[1, 1:-1] = -(phi[2*gridsize[0]::gridsize[0]] - phi[:-2*gridsize[0]:gridsize[0]]) / (2 * dy)
    
    # Forward/backward difference for boundaries
    #E[0, 0] = -(phi[1] - phi[0]) / dx
    #E[0, -1] = -(phi[-1] - phi[-2]) / dx
    #E[1, ::gridsize[0]] = -(phi[gridsize[0]::gridsize[0]] - phi[:-gridsize[0]:gridsize[0]]) / dy
    #E[1, gridsize[0]-1::gridsize[0]] = -(phi[gridsize[0]-1::gridsize[0]] - phi[gridsize[0]-2::gridsize[0]]) / dy

def updateE_fft(phi_k, kx, ky):
    Ex_k = -1j * kx * phi_k
    Ey_k = -1j * ky * phi_k
    Ex = cp.fft.ifftn(Ex_k).real
    Ey = cp.fft.ifftn(Ey_k).real
    return cp.array([Ex.ravel(), Ey.ravel()])


def total_kinetic_energy(v:cp.ndarray, M:cp.ndarray):
    return 0.5*cp.sum((v[0]**2 + v[1]**2) * M)

def total_potential_energy(rho: cp.ndarray, phi: cp.ndarray, dx: float, dy: float) -> float:
    dV = dx * dy
    energy_density = 0.5 * rho * phi
    total_potential_energy = cp.sum(energy_density) * dV
    return total_potential_energy

def total_momentum(v:cp.ndarray, M:cp.ndarray):
    return cp.sum(cp.hypot(v[0], v[1])*M)

def KE_distribution(v:cp.ndarray, M:cp.ndarray, bins:int) -> list:
    E = (v[0]**2 + v[1]**2)*M*0.5
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
    dx = X/m
    dy = Y/n

    F = cp.sum(E[1, :m])*dx + cp.sum(E[1, m*(n-1):])*dx + cp.sum(E[0, ::m])*dy + cp.sum(E[0, m-1::m])*dy

    return F