from numba import njit, prange
import numpy as np
import cupy as cp


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



def draw_particles_gpu(R, pix_arr, screensize, particle_color):
    pix_arr_gpu = cp.zeros_like(pix_arr)
    i = cp.arange(len(R[0]))
    x = (R[0, i] * screensize[0]).astype(cp.int32)
    y = (R[1, i] * screensize[1]).astype(cp.int32)
    pix_arr_gpu[x, y] = particle_color
    pix_arr[:, :] = cp.asnumpy(pix_arr_gpu)




def draw_boundary(boundary, gridsize, pix_arr, screensize, colour):
    dx = int(screensize[0]/gridsize[0])
    dy = int(screensize[1]/gridsize[1])
    for i in cp.asnumpy(boundary):
        y, x = divmod(i, gridsize[0])
        pix_arr[x*dx, y*dy] = colour


def heat_map_gpu(diag_arr:cp.ndarray, pix_arr, screensize, gridsize):
    m, n = gridsize
    dm, dn = screensize[0] // m, screensize[1] // n
    fmax = diag_arr.max()
    fmin = diag_arr.min()
    pix_arr_gpu = cp.zeros_like(pix_arr)
    diag_arr_reshaped = diag_arr.reshape(n, m)
    f_color = ((diag_arr_reshaped - fmin) / (fmax - fmin) * 255).astype(cp.uint16) << 8
    pix_arr_gpu[::dm, ::dn] = f_color.reshape(n, m).transpose()
    pix_arr[:, :] = cp.asnumpy(pix_arr_gpu)



def heat_map_2d_gpu(diag_arr:cp.ndarray, pix_arr, screensize, gridsize):
    m, n = gridsize
    dm, dn = screensize[0] // m, screensize[1] // n
    fmax = diag_arr.max()
    fmin = diag_arr.min()
    pix_arr_gpu = cp.zeros_like(pix_arr)
    f_color = ((diag_arr - fmin) / (fmax - fmin) * 255).astype(cp.uint16) << 8
    pix_arr_gpu[::dm, ::dn] = f_color.reshape(n, m).transpose()
    pix_arr[:, :] = cp.asnumpy(pix_arr_gpu)

