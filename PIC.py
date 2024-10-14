import sys
from Parser import parse_config
import cupy as cp
import time
import Render
import Update
import Solvers
import Consts
import tkinter as tk

def run_gpu(m, n, X, Y, N, dt,
            boundary = None, RENDER = True, DIAGNOSTICS = False, UI = True, RENDER_FRAME = 1, 
            SCREEN_SIZE = (512, 512), DIAGNOSTICS_SIZE = (1024, 1024), 
            solver = 'inverse', DIAG_TYPE = 'line', bins = 64):
    
    print('Starting...')
    global RUN
    global FINISH
    FINISH = False
    RUN = False
    framecounter = 0
    print('creating arrays...')
    gridsize = (m, n)
    dx, dy = X / (m-1), Y / (n-1)
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
    elif solver == 'cg':
        print('Using Conjugate Gradient solver')
        


    if UI:
        def toggle_simulation():
            global RUN
            RUN = not RUN
            button.config(text="Stop" if RUN else "Start")


        def close_window():
            global RUN
            global FINISH
            RUN = False
            FINISH = True
            ui.destroy()

        ui = tk.Tk()
        ui.title("PIC Simulation Control")
        button = tk.Button(ui, text="Start", command=toggle_simulation)
        button.pack(pady=20)
        ui.protocol("WM_DELETE_WINDOW", close_window)
    
    else:
        RUN = True
    
    if RENDER:
        print('creating renderer...')
        renderer =Render.PICRenderer(*SCREEN_SIZE)
        
    # INIT
    
    Update.update_density_gpu(R, part_type, rho, X, Y, gridsize, q_type)
    if boundary != None:
        rho[bound[0]] = bound[1]
    if solver == 'inverse':
            phi = cp.dot(Lap, -rho*Consts.eps0_1 * dx * dy)
    elif solver == 'fft':
        phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
    Update.updateE_gpu(E, phi, X, Y, gridsize)
    Update.update_V(R, V, E, part_type, q_type, m_type_1, gridsize, -dt*0.5, X, Y)

    if RENDER:
        renderer.update_particles(R, part_type)
        renderer.render()

    print('running...')
    while True:
        if UI:
            ui.update()
            if FINISH:
                break
        
        if RUN:
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
            elif solver == 'cg':
                phi = Solvers.solve_poisson_pcg_gpu(rho, m, n, dx, dy, Consts.eps0)
            Update.updateE_gpu(E, phi, X, Y, gridsize)
            Update.update_V(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y)

            sim_time = time.time() - start_time

            # RENDER
            framecounter += 1
            if framecounter == RENDER_FRAME and RENDER:
                renderer.update_particles(R, part_type)
                renderer.render()
                render_time = time.time() - start_time - sim_time
                renderer.change_title(f"Frame time: {(time.time() - start_time)*1000:.2f} ms     Simulation time: {sim_time*1000:.2f} ms     Render time: {render_time*1000:.2f} ms   dt: {dt:.2e}")
                
                if renderer.should_close():
                    break

                framecounter = 0

        elif RENDER == False:
            print(RUN)

        if not UI and not RENDER:
            break

    if RENDER:
        renderer.close()
        print('Renderer closed')
    if UI and not FINISH:
        ui.destroy()

    return 0



config = parse_config('input.ini')
# Use parsed parameters
CPU = config['CPU']
GPU = config['GPU']
m = config['m']
n = config['n']
N = config['N']
dt = config['dt']
q = config['q']
X = config['X']
Y = config['Y']
boundarys = config['boundarys']

if __name__ == "__main__":
    sys.exit(run_gpu(m, n, X, Y, N, dt,
                            boundary=None, 
                            RENDER=config['RENDER'], 
                            RENDER_FRAME=config['RENDER_FRAME'], 
                            DIAGNOSTICS=config['DIAGNOSTICS'],
                            solver=config['solver'], 
                            DIAG_TYPE=config['DIAG_TYPE'], 
                            bins=config['bins'],
                            SCREEN_SIZE=config['SCREEN_SIZE'],
                            UI=config['UI'],))