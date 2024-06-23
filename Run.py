import numpy as np
import cupy as cp
import cupyx as cpx
import scipy
import time
import sdl2
import sdl2.ext
import Render
import Update
import Solvers
import sys


def input():
    def parse_input(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            variables = {}
            for line in lines:
                key, value = line.split('=')
                variables[key.strip()] = eval(value.strip())
        return variables

    if len(sys.argv) > 1:
        variables = parse_input(sys.argv[1])
        for key, value in variables.items():
            globals()[key] = value
    else:
        print("Please specify an input file")


def run_cpu(m, n, X, Y, N, dt, q, RENDER = True, DIAGNOSTICS = False, RENDER_FRAME = 1, SCREEN_SIZE = (512, 512), DIAGNOSTICS_SIZE = (512, 512), PARTICLE_COLOR = 0x00ff00):
    RUN = True
    #GRID
    gridsize = (m, n)
    x = np.linspace(0, X, m, dtype=np.float32)  
    y = np.linspace(0, Y, n, dtype=np.float32)
    rho = np.zeros((m*n), dtype=np.float32)
    phi = np.zeros((m*n), dtype=np.float32)
    Ex = np.zeros((m*n), dtype=np.float32)
    Ey = np.zeros((m*n), dtype=np.float32)
    #PARTICLES
    R = np.random.rand(2, N)
    V = np.zeros((3, N), dtype=np.float32)
    #MCC
    NGD = np.ones((m*n), dtype=np.float32)
    sigma = np.zeros(N, dtype=np.float32)
    P = np.zeros_like(R[0], dtype=np.float32)
    
    framecounter = 0
    Lap = cp.asarray(-scipy.sparse.linalg.LaplacianNd((m, n),boundary_conditions='dirichlet', dtype=np.float32).toarray())
    
    #L, U = scipy.linalg.lu(Lap, permute_l=True)
    inv_Lap = cp.linalg.inv(Lap)
    del Lap
    inv_Lap = cp.asnumpy(inv_Lap)

    
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

        # UPDATE

        Update.push_cpu(R, V, Ex, Ey, gridsize, dt)
        Update.update_density(R, rho, gridsize, q)
        phi[:] = np.dot(inv_Lap, -rho)
        Update.updateE(Ex, Ey, phi, gridsize)
        #Update.update_sigma(sigma, V)
        #Update.MCC(NGD, sigma, V, R, gridsize, dt)

        # RENDER
        framecounter += 1
        if framecounter == RENDER_FRAME and RENDER:
            pixar1[:, :] = 0
            #Render.draw_grid(gridsize, pixar1, SCREEN_SIZE, 0x808080)
            Render.draw_particles(R, pixar1, SCREEN_SIZE, PARTICLE_COLOR)
            window.refresh()
            if DIAGNOSTICS:
                Render.colour_map(phi, pixar2, DIAGNOSTICS_SIZE, gridsize)
                window_diagnostics.refresh()
            
            print(f"Frame time: {(time.time() - start_time)*1000/framecounter:.2f}ms")
            framecounter = 0
        elif RENDER == False:
            print(f"Frame time: {(time.time() - start_time)*1000:.2f}ms")

    return 0



def run_gpu(m, n, X, Y, N, dt, q, boundary, RENDER = True, DIAGNOSTICS = False, RENDER_FRAME = 1, SCREEN_SIZE = (512, 512), DIAGNOSTICS_SIZE = (512, 512), PARTICLE_COLOR = 0x00ff00):
    print('Starting...')
    print('creating arrays...')
    RUN = True
    #GRID
    #gridsize = cp.asarray((m, n), dtype=cp.int32)
    gridsize = (m, n)
    #x = cp.linspace(0, X, m, dtype=cp.float32)  
    #y = cp.linspace(0, Y, n, dtype=cp.float32)
    rho = cp.zeros((m*n), dtype=cp.float32)
    phi = cp.zeros((m*n), dtype=cp.float32)
    Ex = cp.zeros((m*n), dtype=cp.float32)
    Ey = cp.zeros((m*n), dtype=cp.float32)
    #PARTICLES
    R = cp.random.rand(2, N)
    V = cp.zeros((3, N), dtype=cp.float32)
    #MCC
    #NGD = cp.ones((m*n), dtype=cp.float32)
    #sigma = cp.zeros(N, dtype=cp.float32)
    #P = cp.zeros_like(R[0], dtype=cp.float32)
    
    print('creating boundary array...')
    bound = Solvers.boundary_array(boundary, gridsize)
    
    print('creating Laplacian...')
    framecounter = 0
    Lap = -Solvers.Laplacian(m, n)
    Solvers.boundary_conditions_left_gpu(Lap, bound)
    #L, U = scipy.linalg.lu(Lap, permute_l=True)
    print('creating inverse Laplacian...')
    Lap = cp.linalg.inv(Lap)
    #del Lap

    
    if RENDER:
        sdl2.ext.init()
        window = sdl2.ext.Window("PIC", size=SCREEN_SIZE,)
        window.show()
        pixar1 = sdl2.ext.pixelaccess.pixels2d(window.get_surface())
        
        if DIAGNOSTICS:
            window_diagnostics = sdl2.ext.Window("Diagnostics", size=DIAGNOSTICS_SIZE)
            pixar2 = sdl2.ext.pixelaccess.pixels2d(window_diagnostics.get_surface())
            window_diagnostics.show()
    
    print('running...')
    while RUN:

        if RENDER:
            framecounter += 1
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_WINDOWEVENT_CLOSE:
                    RUN = False
                    break

        start_time = time.time()

        # UPDATE
        Update.push_gpu(R, V, Ex, Ey, gridsize, dt)
        Update.update_density_gpu(R, rho, gridsize, q)
        rho[bound[0]] = bound[1]
        phi[:] = cp.dot(Lap, -rho)
        Update.updateE_gpu(Ex, Ey, phi, gridsize)
        #Update.update_sigma(sigma, V)
        #Update.MCC(NGD, sigma, V, R, gridsize, dt)

        # RENDER
        
        if framecounter == RENDER_FRAME and RENDER:
            pixar1[:, :] = 0
            Render.draw_particles_gpu(R, pixar1, SCREEN_SIZE, PARTICLE_COLOR)
            Render.draw_boundary(bound, gridsize, pixar1, SCREEN_SIZE, 0xff0000)
            window.refresh()
            if DIAGNOSTICS:
                pixar2[:, :] = 0
                Render.heat_map_2d_gpu(phi, pixar2, DIAGNOSTICS_SIZE, gridsize)
                window_diagnostics.refresh()
            
            print(f"Frame time: {(time.time() - start_time)*1000:.2f}ms")
            framecounter = 0
        elif RENDER == False:
            print(f"Frame time: {(time.time() - start_time)*1000:.2f}ms")
    return 0