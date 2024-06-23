import numpy as np
import cupy as cp
from numba import jit, njit, prange
import random


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
def updateE(Ex, Ey, phi, gridsize:tuple):
    a = np.reshape(phi, gridsize)
    Ex[:] = -np.gradient(a)[1].flatten()
    Ey[:] = -np.gradient(a)[0].flatten()


@njit(parallel=True, fastmath=True)
def update_sigma(sigma:np.ndarray, V:np.ndarray):
    V[2] = np.hypot(V[0], V[1])
    sigma[:] = (-0.8821*np.log(V[2]) + 15.1262) * 10


@jit
def fmaxw():
    return 2*(random.random() + random.random() + random.random() - 1.5)


@njit(parallel=True,fastmath=True)
def MCC(NGD, sigma, V, R, gridsize, dt):
    for i in prange(len(R[0])):
        idx = int(R[1,i]*gridsize[0]) + int(R[0,i]*gridsize[1])
        x = -NGD[idx]*sigma[i]*V[2,i]*dt
        if np.exp(x) > np.random.random():
            V[0,i] = fmaxw()
            V[1,i] = fmaxw()



def push_gpu(R, V, Ex, Ey, gridsize:tuple, dt):
    '''updates positions and velocities using GPU with cupy'''
    m, n = gridsize
    i = cp.arange(len(R[0]), dtype=cp.int32)
    j = (R[0, i] * m).astype(cp.int32)
    k = (R[1, i] * n).astype(cp.int32)
    idx = k * m + j
    
    V[0, i] += cp.take(Ex, idx) * dt
    V[1, i] += cp.take(Ey, idx) * dt

    R[:, i] = R[:, i] + V[:2, i] * dt

    #R[:, i] = cp.clip(R[:, i] + V[:2, i] * dt, 0, 1)

    #mask = (R[0, i] < 0) | (R[1, i] < 0) | (R[0, i] > 1) | (R[1, i] > 1)
    #R[:, i][[mask, mask]] = 0.5
    #V[:, i][[mask, mask, mask]] = 0


def update_density_gpu(R, rho, gridsize:tuple, q):
    m, n = gridsize
    rho.fill(0)
    i = (R[0] * m).astype(cp.int32)
    j = (R[1] * n).astype(cp.int32)
    k = m * j + i
    cp.add.at(rho, k, q)

def updateE_gpu(Ex, Ey, phi, gridsize:tuple):
    #print(Ex.shape, Ey.shape, phi.shape)
    ax = cp.reshape(phi, gridsize, order='C')
    ay = cp.transpose(ax)
    Ex[:] = -cp.gradient(ax, axis=1).flatten()
    Ey[:] = -cp.gradient(ax, axis=0).flatten()

def boundary_conditions_right_gpu(rho:cp.ndarray, boundary:cp.ndarray):
    rho[boundary[0]] = boundary[1]
    #rho[boundary+1] += C
    #rho[boundary-1] += C


