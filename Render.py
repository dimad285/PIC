from numba import njit, prange
import numpy as np


@njit(parallel=True, fastmath=True)
def draw_particles(R, pix_arr, screensize, particle_color):
    
    for i in prange(len(R[0])):
        x = int(R[0, i] * screensize[0])
        y = int(R[1, i] * screensize[1])
        pix_arr[x, y] = particle_color


@njit(parallel=True, fastmath=True)
def draw_grid(gridsize, pix_arr, screensize, colour):    
    m, n = gridsize
    dm = int(screensize[0]/m)
    dn = int(screensize[1]/n)
    for i in prange(m):
        for j in prange(screensize[1]):
            pix_arr[i*dm, j] = colour
    for j in prange(n):
        for i in prange(screensize[0]):
            pix_arr[i, j*dn] = colour
            


@njit(parallel=True, fastmath=False)
def colour_map(diag_arr:np.ndarray, pix_arr, screensize, gridsize):
    m = gridsize[0]
    n = gridsize[1]
    dm = int(screensize[0]/m)
    dn = int(screensize[1]/n)
    pix_arr[:, :] = 0
    a = diag_arr.max()
    if a < 0:
        a = diag_arr.min()
    for i in prange(n):
        for j in prange(m):
            f_color = diag_arr[i*j+j]/a
            pix_arr[j*dn:(j+1)*dn, i*dm:(i+1)*dm] = int(f_color * 256) * 0x000001