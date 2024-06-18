import numpy as np
import cupy as cp
import time
import sys
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sdl2
import sdl2.ext
import scipy
import Solvers
import Render
import Update

# Constants
RENDER_FRAME = 1
RENDER_EVERY_PARTICLE = 1
SCREEN_SIZE = (512, 512)
DIAGNOSTICS_SIZE = (512, 512)
PARTICLE_COLOR = 0x00ff00
RENDER = True
DIAGNOSTICS = False
RUN = True

m = 64  #x axis nodes
n = 64  #y axis nodes
gridsize = (m, n)
N = 10000  #particles
dt = 0.001
q = 1

x = np.linspace(0, 1, m, dtype=np.float32)  
y = np.linspace(0, 1, n, dtype=np.float32)

rho = np.zeros((m*n), dtype=np.float32)
phi = np.zeros((m*n), dtype=np.float32)
Ex = np.zeros((m*n), dtype=np.float32)
Ey = np.zeros((m*n), dtype=np.float32)

R = np.random.rand(2, N)
V = np.zeros((3, N), dtype=np.float32)


#MCC
NGD = np.ones((m*n), dtype=np.float32)
sigma = np.zeros(N, dtype=np.float32)
P = np.zeros_like(R, dtype=np.float32)

def run():
    
    global R, V, rho, phi, Ex, Ey, RUN, RENDER, DIAGNOSTICS, dt
    framecounter = 0
    Lap = -scipy.sparse.linalg.LaplacianNd((m, n),boundary_conditions='dirichlet', dtype=np.float32).toarray()
    #L, U = scipy.linalg.lu(Lap, permute_l=True)
    inv_Lap = np.linalg.inv(Lap)
    del Lap

    if RENDER:
        sdl2.ext.init()
        window = sdl2.ext.Window("PIC", size=SCREEN_SIZE,)
        window.show()
        pixar1 = sdl2.ext.pixelaccess.pixels2d(window.get_surface())
        
        if DIAGNOSTICS:
            window_diagnostics = sdl2.ext.Window("Diagnostics", size=DIAGNOSTICS_SIZE)
            pixar2 = sdl2.ext.pixelaccess.pixels2d(window_diagnostics.get_surface())
            window_diagnostics.show()
    
    while RUN:

        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_WINDOWEVENT_CLOSE:
                RUN = False
                break

        start_time = time.time()

        framecounter += 1
        if framecounter == RENDER_FRAME and RENDER:
            pixar1[:, :] = 0
            #Render.draw_grid(gridsize, pixar1, SCREEN_SIZE, 0x808080)
            Render.draw_particles(R, pixar1, SCREEN_SIZE, PARTICLE_COLOR)
            window.refresh()
            if DIAGNOSTICS:
                Render.colour_map(phi, pixar2, DIAGNOSTICS_SIZE, gridsize)
                window_diagnostics.refresh()
            framecounter = 0


        Update.push_cpu(R, V, Ex, Ey, gridsize, dt)
        Update.density_update(R, rho, gridsize, q)
        phi[:] = np.dot(inv_Lap, -rho)
        Update.updateE(Ex, Ey, phi, gridsize)
        Update.update_sigma(sigma, V)
        Update.MCC(NGD, sigma, V, R, gridsize, dt)
        
        print(f"Frame time: {(time.time() - start_time)*1000:.2f}ms")

    return 0

if __name__ == "__main__":
    sys.exit(run())