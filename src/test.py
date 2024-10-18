from PIC import *
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import Update
import Solvers
import scipy.sparse
import Run


m = 16  #x axis nodes
n = 16  #y axis nodes
grid = (m, n)
N = 1000 #particles
dt = 0.001
q = 10

X = 1
Y = 1

x = np.linspace(0, X, m, dtype=cp.float32)  
y = np.linspace(0, Y, n, dtype=cp.float32)

rho = cp.zeros((m*n), dtype=cp.float32)
phi = cp.zeros((m*n), dtype=cp.float32)
E = cp.zeros((3, m*n), dtype=cp.float32)

R = cp.random.rand(2, N)
R_grid = cp.zeros((2, N), dtype=cp.int16)
V = cp.zeros((3, N), dtype=cp.float32)
Q = cp.ones(N, dtype=cp.float32)
M = cp.ones(N, dtype=cp.float32)

boundary1 = ([int(m/4*3), int(n/4*3), int(m/4*3), int(n/4)], 100)
boundary2 = ([int(m/4), int(n/4*3), int(m/4), int(n/4)], -100)

boundarys = (boundary1, boundary2)
bound = Solvers.boundary_array(boundarys, grid)

print('creating Laplacian...')
#cp.savetxt('lap.txt', Lap, fmt='%i')
Lap = Solvers.Laplacian(m, n)
#Solvers.boundary_conditions_left_gpu(Lap, bound)
inv_Lap = cp.linalg.inv(Lap)
del Lap

print('creating renderer...')
plt.style.use('dark_background')
fig = plt.figure()
ax_phi = fig.add_subplot(121, projection='3d')
ax_phi.set_xlabel('X')
ax_phi.set_ylabel('Y')
ax_phi.set_zlabel('Phi')

#ax_r = fig.add_subplot(111) 

ax_E = fig.add_subplot(122)

X, Y = np.meshgrid(x, y)

plt.ion()

print('running...')
while True:

    #UPDATE
    Update.push_gpu(R, V, E,Q, M, grid, dt)
    Update.update_density_gpu(R, rho, 1, 1, grid, q)
    #Update.boundary_conditions_right_gpu(rho, bound)
    #Solvers.sor_gpu_sparse_matrix(grid, phi, rho, w=1, residual=1e-8)
    #rho[bound[0]] = bound[1]
    phi[:] = cp.dot(inv_Lap, rho)
    Update.updateE_gpu(E, phi, 1, 1, grid)

    phi_plot = phi.get()
    Ex_plot = E[0].get()
    Ey_plot = E[1].get()

    ax_phi.clear()
    #ax_r.clear()
    ax_E.clear()
    ax_phi.plot_surface(X, Y, phi_plot.reshape(grid), cmap='plasma', alpha=0.8)
    #ax_r.scatter(R[0], R[1], s=1, c='g')
    ax_E.quiver(X, Y, Ex_plot.reshape(grid), Ey_plot.reshape(grid), color='w')
    plt.draw()
    plt.pause(0.001)

plt.show()

#print(Solvers.Laplacian(m, n))
