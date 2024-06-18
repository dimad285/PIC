import numpy as np
from numba import njit, prange

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

