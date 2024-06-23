import numpy as np
import cupy as cp
from numba import njit

def boundary_array(input:tuple, gridsize:tuple) -> cp.ndarray:
    boundary = [[],[]]
    for i in input:
        m1, n1, m2, n2 = i[0]
        dm = (m2 - m1)/gridsize[0]
        dn = (n2 - n1)/gridsize[1] 
        x = m1
        y = n1
        while y != n2:
            boundary[0].append(y*gridsize[0]+x)
            boundary[1].append(i[1])
            x = int(x+dm)
            y = int(y+dn)
            #print(x,y)
    
    return cp.asarray(boundary, dtype=cp.int32)


def boundary_conditions_left_gpu(A:cp.ndarray, boundary:cp.ndarray):
    for i in boundary[0]:
        A[i,:] = 0
        A[i, i] = 1



def Laplacian(m, n) -> cp.ndarray:

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
    return Lap


@njit(parallel=False, fastmath=True)
def lu_solve(lower_triangular, upper_triangular, x, b):
    """Solve the linear system Ax = b using LU decomposition."""
    n = len(b)
    y = np.empty_like(b)
    x[:] = b
    
    # Forward substitution
    for i in range(n):
        y[i] = b[i] - np.dot(lower_triangular[i, :i], y[:i])

    # Backward substitution
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(upper_triangular[i, i + 1:], x[i + 1:])) / upper_triangular[i, i]


@njit(parallel=False, fastmath=True)
def CG_solve(A: np.ndarray, x: np.ndarray, b: np.ndarray, residual=1e-6):
    r = b - np.dot(A, x)
    p = r.copy()
    r0 = np.dot(r, r)
    while r0 > residual:
        Ap = np.dot(A, p)
        pAp = np.dot(p, Ap)
        alpha = r0 / pAp
        x += alpha * p
        r -= alpha * Ap
        r1 = np.dot(r, r)
        p = r + (r1 / pAp) * p
        r0 = r1

