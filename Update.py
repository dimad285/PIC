import numpy as np
import cupy as cp
from numba import jit, njit, prange
import math
import random


@njit(parallel=True)
def push_cpu(R, V, Ex, Ey, gridsize:tuple, dt):
    m, n = gridsize
    for i in prange(len(R[0])):
        j = int(R[0,i]*m)
        k = int(R[1,i]*n)
        V[0,i] += Ex[k*m+j]*dt
        V[1,i] += Ey[k*m+j]*dt

    R[:,:] = R + V[0:2] * dt

    for i in prange(len(R[0])):
        if R[0, i] > 1 or R[1, i] > 1 or R[0, i] < 0 or R[1, i] < 0:
            R[0, i] = 0.5
            R[1, i] = 0.5
            V[0, i] = 0
            V[1, i] = 0

@njit(parallel=True, fastmath=True)
def density_update(R:np.ndarray, rho:np.ndarray, gridsize:tuple, q):
    rho[:] = 0
    for i in prange(len(R[0])):
        j = int(R[0,i]*gridsize[0])
        k = int(R[1,i]*gridsize[1])
        rho[k*gridsize[0]+j] += q

def boundary_conditions(rho:np.ndarray, gridsize:tuple, boundary:np.ndarray):
    for i in boundary:
        rho[i[0]] -= i[1]

#@njit
def updateE(Ex, Ey, phi, gridsize:tuple):
    a = np.reshape(phi, gridsize)
    Ex[:] = -np.gradient(a)[1].flatten()
    Ey[:] = -np.gradient(a)[0].flatten()


@njit(parallel=True, fastmath=True)
def update_sigma(sigma:np.ndarray, V:np.ndarray):
    V[2,:] = np.sqrt(V[0]**2 + V[1]**2)
    sigma[:] = (-0.8821*np.log(V[2])+15.1262)*0.01


@jit
def fmaxw():
    return 2*(random.random() + random.random() + random.random() - 1.5)

@jit
def MCC(NGD:np.ndarray, sigma:np.ndarray, V:np.ndarray, R:np.ndarray, gridsize:tuple, dt):
    for i in prange(len(R[0])):
        j = int(R[0,i]*gridsize[0])
        k = int(R[1,i]*gridsize[1])
        x = -NGD[k*gridsize[1]+j]*sigma[i]*V[2,i]*dt
        p = 1 - math.exp(x)
        #print(p)
        if p > random.random():
            V[0,i] = fmaxw()
            V[1,i] = fmaxw()