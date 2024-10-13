import numpy as np
import cupy as cp
import scipy
import time
import Render
import Update
import Solvers
import Consts
import tkinter as tk


# MOVE RUN FUNCTION TO THE MAIN FILE

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
            diag_surface = window_diagnostics.get_surface()
            pixar2 = sdl2.ext.pixelaccess.pixels2d(diag_surface)
            
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
                Render.heat_map(phi, pixar2, DIAGNOSTICS_SIZE, gridsize)
                window_diagnostics.refresh()
            
            print(f"Frame time: {(time.time() - start_time)*1000/framecounter:.2f}ms")
            framecounter = 0
        elif RENDER == False:
            print(f"Frame time: {(time.time() - start_time)*1000:.2f}ms")

    return 0

def run_gpu(m, n, X, Y, N, dt,
            boundary = None, RENDER = True, DIAGNOSTICS = False, UI = True, RENDER_FRAME = 1, 
            SCREEN_SIZE = (512, 512), DIAGNOSTICS_SIZE = (1024, 1024), 
            solver = 'inverse', DIAG_TYPE = 'line', bins = 64):
    
    print('Starting...')
    RUN = True
    framecounter = 0
    print('creating arrays...')
    gridsize = (m, n)
    rho = cp.empty((m*n), dtype=cp.float64)
    phi = cp.empty((m*n), dtype=cp.float64)
    E = cp.empty((2, m*n), dtype=cp.float64)
    R = cp.random.uniform(0.25, 0.75, (2, N)).astype(cp.float64)
    V = cp.zeros((2, N), dtype=cp.float64)
    #M_1 = cp.ones(N, dtype=cp.float32) / Consts.me
    part_type = cp.random.randint(0, 2, N, dtype=cp.int32) + 1
    m_type = cp.array([Consts.mp, Consts.me], dtype=cp.float64)
    m_type_1 = cp.array([1/Consts.mp, 1/Consts.me], dtype=cp.float64)
    q_type = cp.array([1.0 * Consts.qe, -1.0 * Consts.qe], dtype=cp.float64)

    
    if boundary != None:
        print('creating boundary array...')
        bound = Solvers.boundary_array(boundary, gridsize)
        #print(bound)

    if solver == 'inverse':
        print('creating Laplacian...')
        Lap = Solvers.Laplacian_square(m, n)
        if boundary != None:
            print('applying boundary conditions...')
            Solvers.boundary_conditions_left_gpu(Lap, bound)
        print('creating inverse Laplacian...')
        Lap = cp.linalg.inv(Lap)
    elif solver == 'fft':
        k_sq = Solvers.setup_fft_solver(m, n)
        pass


    if UI:
        ui = tk.Tk()
        ui.mainloop()
    
    if RENDER:
        print('creating renderer...')
        renderer =Render.PICRenderer(*SCREEN_SIZE)
        
    # INIT
    
    Update.update_density_gpu(R, part_type, rho, X, Y, gridsize, q_type)
    if boundary != None:
        rho[bound[0]] = bound[1]
    if solver == 'inverse':
            phi = cp.dot(Lap, -rho*Consts.eps0_1)
    elif solver == 'fft':
        phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
    Update.updateE_gpu(E, phi, X, Y, gridsize)
    Update.update_V(R, V, E, part_type, q_type, m_type_1, gridsize, -dt*0.5, X, Y)
    
    print('running...')
    while RUN:

        start_time = time.time()

        # UPDATE

        R[:] += V[:] * dt
        Update.update_density_gpu(R, part_type, rho, X, Y, gridsize, q_type)
        if boundary != None:
            rho[bound[0]] = bound[1]
        if solver == 'inverse':
            phi = cp.dot(Lap, -rho*Consts.eps0_1)
        elif solver == 'fft':
            phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
        Update.updateE_gpu(E, phi, X, Y, gridsize)
        Update.update_V(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y)
        
        #dt = min(1/cp.max(cp.abs(V[0]))/X/(m-1), 1/cp.max(cp.abs(V[1]))/Y/(n-1)) * 0.5
        #print(dt)
        # RENDER

        if framecounter == RENDER_FRAME and RENDER:

            renderer.update_particles(R, part_type)
            renderer.render()
            renderer.change_title(f"Frame time: {(time.time() - start_time)*1000:.2f} ms     dt = {dt:.2e} s")
            if renderer.should_close():
                break

            # RENDER DIAGNOSTICS
       
            #print(f"Frame time: {(time.time() - start_time)*1000:.2f}ms")
            
            framecounter = 0
        elif RENDER == False:
            print(f"Frame time: {(time.time() - start_time)*1000:.2f}ms")

    renderer.close()

    return 0