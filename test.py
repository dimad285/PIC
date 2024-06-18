from PIC import *
import matplotlib.pyplot as plt
import numpy as np
import Update


m = 16  #x axis nodes
n = 16  #y axis nodes
grid = (m, n)
N = 1000  #particles
dt = 0.01
q = 1

x = np.linspace(0, 1, m, dtype=np.float32)  
y = np.linspace(0, 1, n, dtype=np.float32)

rho = np.zeros((m*n), dtype=np.float32)
phi = np.zeros((m*n), dtype=np.float32)
Ex = np.zeros((m*n), dtype=np.float32)
Ey = np.zeros((m*n), dtype=np.float32)

R = np.random.rand(2, N)
V = np.zeros((2, N), dtype=np.float32)

Lap = -scipy.sparse.linalg.LaplacianNd((m, n),boundary_conditions='dirichlet', dtype=np.float32).toarray()
#inv_Lap = np.linalg.inv(Lap)
#print(inv_Lap)
#L, U = scipy.linalg.lu(-Lap, permute_l=True)
#del Lap


Update.push_cpu(R, V, Ex, Ey, grid, dt)
Update.density_update(R, rho, grid, q)
#rho[0:m] -= 100000
#Solvers.lu_solve(L, U, phi, -rho)
#Solvers.cholesky_solve(L, phi, -rho)
Update.updateE(Ex, Ey, phi, grid)



def plot():
    plt.style.use('dark_background')
    fig = plt.figure()
    ax_phi = fig.add_subplot(221, projection='3d')
    ax_phi.set_xlabel('X')
    ax_phi.set_ylabel('Y')
    ax_phi.set_zlabel('Phi')

    ax_r = fig.add_subplot(223) 

    ax_E = fig.add_subplot(222)

    X, Y = np.meshgrid(x, y)

    plt.ion()

    while True:
        #Update.push_cpu(R, V, Ex, Ey, grid, dt)
        #Update.density_update(R, rho, grid, q)
        #rho[0:m] -= 100
        #Solvers.lu_solve(L, U, phi, rho)
        #Solvers.cholesky_solve(L, phi, -rho)
        #phi[:] = np.dot(inv_Lap, -rho)
        #Update.updateE(Ex, Ey, phi, grid)
        Solvers.jacobi_solver(Lap, phi, -rho)


        ax_phi.clear()
        ax_r.clear()
        ax_E.clear()
        ax_phi.plot_surface(X, Y, phi.reshape(grid), cmap='viridis', alpha=0.8)
        ax_r.scatter(R[0], R[1], s=1, c='g')
        ax_E.quiver(X, Y, Ex.reshape(grid), Ey.reshape(grid), color='w')
        plt.draw()
        plt.pause(0.001)

    plt.show()

plot()