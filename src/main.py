import cupy as cp
import numpy as np
import time
import src.Render as Render
import src.Update as Update
import src.Solvers as Solvers
import src.Consts as Consts
import tkinter as tk
from src.Parser import parse_config
import sys


def update_legend(renderer, label, text, y_pos, color=(0, 255, 0), x_pos=20, scale=0.5):
    renderer.update_text(label, {
        'text': text,
        'x': x_pos,
        'y': y_pos,
        'scale': scale,
        'color': color
    })

def clear_legend_entries(renderer, labels):
    for label in labels:
        renderer.update_text(label, {
            'text': '',  # Set the text to an empty string
            'x': 0,  # Optionally reset position if needed
            'y': 0,  # Optionally reset position if needed
            'scale': 0.5,
            'color': (0, 0, 0)  # Optionally set the color to black or leave as is
        })

def run_gpu(m = 16, n = 16, k = 0, X = 1, Y = 1, Z = 1, N = 1000, dt = 0.001, grid_type='2d',
            boundary = None, RENDER = True, DIAGNOSTICS = False, UI = True, RENDER_FRAME = 1, 
            SCREEN_SIZE = (512, 512), DIAGNOSTICS_SIZE = (512, 512), 
            solver = 'inverse', DIAG_TYPE = 'line', bins = 64):
    
    print('Starting...')
    # Graphical user interface
    fontfile = "C:\\Windows\\Fonts\\Arial.ttf"
    global RUN
    global FINISH
    global TRACE
    global TEXTUI
    TEXTUI = True
    TRACE = False
    FINISH = False
    RUN = False
    global selected_render_type
    selected_render_type = "particles"  # default render type
    framecounter = 0
    t = 0


    # Grid parameters
    print('creating arrays...')
    if grid_type == '2d' or grid_type == '2d_cylinder':
        gridsize = (m, n)
        dx, dy = X / (m-1), Y / (n-1)
        rho = cp.empty((m*n), dtype=cp.float64)
        phi = cp.empty((m*n), dtype=cp.float64)
        E = cp.empty((2, m*n), dtype=cp.float64)
        R = cp.random.uniform(X/4, X*3/4, (2, N)).astype(cp.float64)
        V = cp.zeros((2, N), dtype=cp.float64)
    elif grid_type == '3d':
        gridsize = (m, n, k)
        dx, dy, dz = X / (m-1), Y / (n-1), Z / (k-1)
        rho = cp.empty((m*n*k), dtype=cp.float64)
        phi = cp.empty((m*n*k), dtype=cp.float64)
        E = cp.empty((3, m*n*k), dtype=cp.float64)
        R = cp.random.uniform(X/4, X*3/4, (3, N)).astype(cp.float64)
        V = cp.zeros((3, N), dtype=cp.float64)


    # Particle parameters

    #M_1 = cp.ones(N, dtype=cp.float32) / Consts.me
    part_type = cp.random.randint(1, 3, N, dtype=cp.int32)
    m_type = cp.array([Consts.mp, Consts.me], dtype=cp.float64)
    m_type_1 = cp.array([0, 1/Consts.mp, 1/Consts.me], dtype=cp.float64)
    q_type = cp.array([0, 1.0 * Consts.qe, -1.0 * Consts.qe], dtype=cp.float64)
    
    # Diagnostics
    hystory_x = np.array([], dtype=cp.float64)
    hystory_y = np.array([], dtype=cp.float64)
    hystory_y2 = np.array([], dtype=cp.float64)

    
    if boundary != None:
        print('creating boundary array...')
        bound = Solvers.boundary_array(boundary, gridsize)
        #print(bound)

    if solver == 'inverse': # add matrix compression
        print('creating Laplacian...')
        Lap = Solvers.Laplacian_square(m, n)
        if boundary != None:
            print('applying boundary conditions...')
            Solvers.boundary_conditions_left_gpu(Lap, bound)
        print('creating inverse Laplacian...')
        Lap = cp.linalg.inv(Lap)
    elif solver == 'fft':
        if grid_type == '2d':
            k_sq = Solvers.setup_fft_solver(m, n)
        elif grid_type == '3d':
            k_sq = Solvers.setup_fft_solver_3d(m, n, k)
        elif grid_type == '2d_cylinder':
            pass
    elif solver == 'cg': # add boundary conditions
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

        def toggle_trace():
            global TRACE
            TRACE = not TRACE

        def toggle_textui():
            global TEXTUI
            TEXTUI = not TEXTUI

        def update_render_type(event):
            global selected_render_type
            selection = render_listbox.get(render_listbox.curselection())
            selected_render_type = selection
            print(f"Render type changed to: {selected_render_type}")

        ui = tk.Tk()
        ui.geometry('300x300')
        ui.title("PIC Simulation Control")

        button = tk.Button(ui, text="Start", command=toggle_simulation)
        button_trace = tk.Button(ui, text="Trace", command=toggle_trace)
        button_textui = tk.Button(ui, text="Text", command=toggle_textui)
        button.pack(pady=10)
        button_trace.pack(pady=10)
        button_textui.pack(pady=10)

        render_listbox = tk.Listbox(ui)
        render_listbox.insert(1, "particles")
        render_listbox.insert(2, "heatmap")
        render_listbox.insert(3, "line_plot")
        render_listbox.pack(pady=10)
        
        # Bind listbox selection to the update_render_type function
        render_listbox.bind('<<ListboxSelect>>', update_render_type)

        ui.protocol("WM_DELETE_WINDOW", close_window)
    
    else:
        RUN = True
    
    if RENDER:
        renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, renderer_type=selected_render_type)
        if DIAGNOSTICS:
            renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, renderer_type="heatmap")
        
    # INIT
    
    if grid_type == '2d':
        Update.update_density_gpu(R, part_type, rho, X, Y, gridsize, q_type)
        if boundary != None:
            rho[bound[0]] = bound[1]
        if solver == 'inverse':
                phi = cp.dot(Lap, -rho * Consts.eps0_1 * dx * dy)
        elif solver == 'fft':
            phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
        Update.updateE_gpu(E, phi, X, Y, gridsize)
        Update.update_V(R, V, E, part_type, q_type, m_type_1, gridsize, -dt*0.5, X, Y)

    elif grid_type == '3d':
        Update.update_density_gpu_3d(R, part_type, rho, X, Y, Z, gridsize, q_type)
        if solver == 'fft':
            phi = Solvers.solve_poisson_fft_3d(rho, k_sq, Consts.eps0)
        Update.updateE_gpu_3d(E, phi, X, Y, Z, gridsize)
        Update.update_V_3d(R, V, E, part_type, q_type, m_type_1, gridsize, -dt*0.5, X, Y, Z)

    if RENDER:
        #renderer.update_particles(R/X, part_type)
        #renderer.update_heatmap(phi, m, n)
        #renderer.update_heatmap(phi, m, n)
        #renderer.render()
        if DIAGNOSTICS:
            renderer.update_heatmap(phi, m, n)
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

            if grid_type == '2d':
                R[:] += V[:] * dt 
                Update.update_density_gpu(R, part_type, rho, X, Y, gridsize, q_type)
                if boundary != None:
                    rho[bound[0]] = bound[1]
                if solver == 'inverse':
                    phi = cp.dot(Lap, -rho * Consts.eps0_1 * dx * dy)
                elif solver == 'fft':
                    phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
                elif solver == 'cg':
                    phi = Solvers.solve_poisson_pcg_gpu(rho, m, n, dx, dy, Consts.eps0)
                Update.updateE_gpu(E, phi, X, Y, gridsize)
                Update.update_V(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y)

            elif grid_type == '3d':
                R[:] += V[:] * dt
                Update.update_density_gpu_3d(R, part_type, rho, X, Y, Z, gridsize, q_type)
                if solver == 'fft':
                    phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
                Update.updateE_gpu_3d(E, phi, X, Y, Z, gridsize)
                Update.update_V_3d(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y, Z)

            t += dt
            sim_time = time.time() - start_time

            # RENDER
            framecounter += 1
            if framecounter == RENDER_FRAME and RENDER:

                render_time = time.time() - start_time - sim_time
                KE = Update.total_kinetic_energy(V, m_type, part_type)
                PE = Update.total_potential_energy(rho, phi, dx, dy)
                TE = PE + KE
                P = Update.total_momentum(V, m_type, part_type)
                #FF = Update.boundary_field_flux(E, gridsize, X, Y)
                #Q = cp.sum(rho * dx * dy)*Consts.eps0_1
                hystory_x = cp.append(hystory_x, t)
                hystory_y = cp.append(hystory_y, TE)
                hystory_y2 = cp.append(hystory_y2, P)
                R_2d = cp.array([R[0]/X, R[1]/Y])
                
                if selected_render_type == "particles":
                    renderer.renderer_type = "particles"
                    renderer.update_particles(R_2d, part_type)
                elif selected_render_type == "heatmap" and grid_type == '2d':
                    renderer.renderer_type = "heatmap"
                    renderer.update_heatmap(phi, m, n)
                elif selected_render_type == "line_plot":
                    renderer.renderer_type = "line_plot"
                    renderer.update_line_data('Energy', hystory_x, hystory_y)
                    renderer.update_line_data('Momentum', hystory_x, hystory_y2, color=(0.0, 0.5, 0.5))

                renderer.render()
                
                if TEXTUI:

                    update_legend(renderer, 'legend_E', 'Energy', 300, color=(255, 255, 255))
                    update_legend(renderer, 'legend_P', 'Momentum', 330, color=(0, 128, 128))
                    update_legend(renderer, 'fps', f"Frame time: {(time.time() - start_time)*1000:.1f} ms", 30)
                    update_legend(renderer, 'time', f"Time: {t:.2e} s", 60)
                    update_legend(renderer, 'Energy', f"Energy: {TE:.2e} J", 90)
                    update_legend(renderer, 'Momentum', f"Momentum: {P:.2e} kg*m/s", 120)
                    update_legend(renderer, 'phi_max', f"phi_max: {cp.max(phi):.2e} V", 210)
                    update_legend(renderer, 'phi_min', f"phi_min: {cp.min(phi):.2e} V", 240)

                else:

                    clear_legend_entries(renderer, ['legend_E', 'legend_P', 'fps', 'time', 'Energy', 'Momentum', 'phi_max', 'phi_min'])

                if DIAGNOSTICS:
                    renderer.update_heatmap(phi, m, n)
                    renderer.render()


            if UI:
                
                
                #renderer.change_title(f"Frame time: {(time.time() - start_time)*1000:.2f} ms     Simulation time: {sim_time*1000:.2f} ms     Render time: {render_time*1000:.2f} ms   t: {t:.2e}")
                #renderer.change_title(f"Kinetic energy: {KE:.2e} J  Potential energy: {PE:.2e} J    Total energy: {TE:.2e} J    Total momentum: {P:.2e} kg*m/s")
               
                if renderer.should_close():
                    break

                framecounter = 0

        elif RENDER == False:
            print(RUN)

        if not UI and not RENDER:
            break

    if RENDER:
        renderer.close()
        if DIAGNOSTICS:
            renderer.close()
        print('Renderer closed')
    if UI and not FINISH:
        ui.destroy()

    return 0




if __name__ == "__main__":

    config = parse_config('input.ini')
    # Use parsed parameters
    CPU = config['CPU']
    GPU = config['GPU']
    m = config['m']
    n = config['n']
    k = config['k']
    N = config['N']
    dt = config['dt']
    q = config['q']
    X = config['X']
    Y = config['Y']
    Z = config['Z']
    boundarys = config['boundarys']

    sys.exit(run_gpu(m, n, k, X, Y, Z, N, dt, grid_type=config['grid_type'],
                            boundary=None, 
                            RENDER=config['RENDER'], 
                            RENDER_FRAME=config['RENDER_FRAME'], 
                            DIAGNOSTICS=config['DIAGNOSTICS'],
                            solver=config['solver'], 
                            DIAG_TYPE=config['DIAG_TYPE'], 
                            bins=config['bins'],
                            SCREEN_SIZE=config['SCREEN_SIZE'],
                            UI=config['UI'],))
