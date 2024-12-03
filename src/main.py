import cupy as cp
import numpy as np
import time
import src.Render as Render
import src.Update as Update
import src.Solvers as Solvers
import src.Consts as Consts
import tkinter as tk
from tkinter import ttk
import random




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
    global surafce_plot_var_name
    selected_render_type = "particles"  # default render type
    surafce_plot_var_name = 'phi'
    framecounter = 0
    t = 0
    frame_time = 0


    # INIT

    # Grid parameters
    print('creating arrays...')
    
    if grid_type == '2d' or grid_type == '2d_cylinder':

        gridsize = (m, n)
        dx, dy = X / (m-1), Y / (n-1)

        rho = cp.empty((m*n), dtype=cp.float32) # charge density
        J = cp.empty((2, m*n), dtype=cp.float32) # current density

        phi = cp.empty((m*n), dtype=cp.float32) # scalar potential
        A = cp.empty((2, m*n), dtype=cp.float32) # vector potential

        B = cp.empty((2, m*n), dtype=cp.float32) # magnetic field
        E = cp.empty((2, m*n), dtype=cp.float32) # electric field

        R = cp.empty((2, N), dtype=cp.float32) # particle positions
        V = cp.empty((2, N), dtype=cp.float32) # particle velocities

    elif grid_type == '3d':
        gridsize = (m, n, k)
        dx, dy, dz = X / (m-1), Y / (n-1), Z / (k-1)
        rho = cp.empty((m*n*k), dtype=cp.float64)
        phi = cp.empty((m*n*k), dtype=cp.float64)
        A = cp.empty((3, m*n*k), dtype=cp.float64)
        B = cp.empty((3, m*n*k), dtype=cp.float64)
        J = cp.empty((3, m*n*k), dtype=cp.float64)
        E = cp.empty((3, m*n*k), dtype=cp.float64)
        R = cp.random.uniform(X/4, X*3/4, (3, N)).astype(cp.float64)
        V = cp.zeros((3, N), dtype=cp.float64)


    # Particle parameters
    part_type = cp.empty(N, dtype=cp.int32) # 0 for dead particle. 
    # When paarticle 'dies' it's type is set to 0 and it is swapped with the last alive particle
    last_alive = 0 # index of last alive particle
    # Problems can be with particle sorting
    part_name = ['', 'proton', 'electron']
    m_type = cp.array([0, Consts.mp, Consts.me], dtype=cp.float64)
    m_type_1 = cp.array([0, 1/Consts.mp, 1/Consts.me], dtype=cp.float64)
    q_type = cp.array([0, 1.0 * Consts.qe, -1.0 * Consts.qe], dtype=cp.float64)
    
    # MCC
    cross_section = Update.read_cross_section('csf/H.txt')
    mcc_probability = cp.zeros(N, dtype=cp.float32)
    mcc_random = cp.zeros(N, dtype=cp.float32)
    part_color_types = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.5, 0.0]])
    part_colors = np.empty((N, 3), dtype=cp.float32)
    part_energy = cp.zeros(N, dtype=cp.float64) 
    part_cross_section = cp.zeros(N, dtype=cp.float32)
    
    # Diagnostics
    hystory_x = np.array([], dtype=cp.float32)
    hystory_y = np.array([], dtype=cp.float32)
    hystory_y2 = np.array([], dtype=cp.float32)
    #hystory_n = np.array([], dtype=cp.float32)
    #hystory_ne = np.array([], dtype=cp.float32)
    #hystory_ni = np.array([], dtype=cp.float32)

    
    if boundary != None:
        print('creating boundary array...')
        bound, bound_val = Solvers.boundary_array(boundary, gridsize)
        #print(bound)

    if solver == 'inverse': # add matrix compression
        print('creating Laplacian...')
        Lap = Solvers.Laplacian_square(m, n)
        if boundary != None:
            print('applying boundary conditions...')
            Solvers.boundary_conditions_left_gpu(Lap, bound)
        print('creating inverse Laplacian...')
        Lap = cp.linalg.inv(Lap)
    elif solver == 'fft' or solver == 'fft_bc':
        if grid_type == '2d':
            #fft_solver = Solvers.PoissonFFTSolver(m, n, dx, dy, Consts.eps0)
            k_sq = Solvers.setup_fft_solver(m, n, dx, dy)
        elif grid_type == '3d':
            k_sq = Solvers.setup_fft_solver_3d(m, n, k)
        elif grid_type == '2d_cylinder':
            pass
    elif solver == 'cg': # add boundary conditions
        print('Using Conjugate Gradient solver')
    elif solver == 'cg_fft':
        print('Using Conjugate Gradient solver')
        k_sq = Solvers.setup_fft_solver(m, n)
    elif solver == 'multigrid':
        number_of_levels = 2
        phi_multigrid = []
        rho_multigrid = []
        if m//((number_of_levels+1)**2) >= 2:
            pass
        else:
            number_of_levels -= 1
        for i in range(1, number_of_levels+1):
            phi_multigrid.append(cp.empty((m//(2**i) + 1, n//(2**i) + 1), dtype=cp.float32))
            rho_multigrid.append(cp.empty((m//(2**i) + 1, n//(2**i) + 1), dtype=cp.float32))
        Lap_multigrid = Solvers.Laplacian_square(*phi_multigrid[-1].shape)

    if UI:
        def toggle_simulation():
            global RUN
            RUN = not RUN
            start_stop_button.config(text="Stop" if RUN else "Start")

        def close_window():
            global RUN
            global FINISH
            RUN = False
            FINISH = True
            ui.destroy()

        def toggle_trace():
            global TRACE
            TRACE = not TRACE
            trace_button.config(text="Trace On" if TRACE else "Trace Off")

        def toggle_textui():
            global TEXTUI
            TEXTUI = not TEXTUI
            textui_button.config(text="Text On" if TEXTUI else "Text Off")

        def update_render_type(event):
            global selected_render_type
            selection = render_listbox.get(render_listbox.curselection())
            selected_render_type = selection
        
        def update_surface_type(event):
            global surafce_plot_var_name
            selection_surface = renderer_listbox_surface.get(renderer_listbox_surface.curselection()[0])
            surafce_plot_var_name = selection_surface


        ui = tk.Tk()
        ui.geometry('300x750')
        ui.title("PIC Simulation Control")

        # Use ttk widgets for buttons and scales
        style = ttk.Style()
        style.theme_use('clam')

        start_stop_button = ttk.Button(ui, text="Start", command=toggle_simulation)
        trace_button = ttk.Button(ui, text="Trace Off", command=toggle_trace)
        textui_button = ttk.Button(ui, text="Text Off", command=toggle_textui)
        start_stop_button.pack(pady=10)
        trace_button.pack(pady=10)
        textui_button.pack(pady=10)

        # Use tk.Listbox for the render type selection
        render_listbox = tk.Listbox(ui, height=8)
        render_listbox.insert(1, "particles")
        if grid_type == '2d':
            render_listbox.insert(2, "heatmap")
            render_listbox.insert(3, "line_plot_E")
            render_listbox.insert(4, "line_plot_P")
            render_listbox.insert(5, "line_plot_MCC")
            render_listbox.insert(6, "line_plot_E_distribution")
            render_listbox.insert(7, "line_plot_V_distribution")
            render_listbox.insert(8, "surface_plot")
        elif grid_type == '3d':
            render_listbox.insert(2, "line_plot")
        render_listbox.pack(pady=10)
        # Bind listbox selection to the update_render_type function
        render_listbox.bind('<<ListboxSelect>>', update_render_type)

        renderer_listbox_surface = tk.Listbox(ui, height=3)
        renderer_listbox_surface.insert(1, "phi")
        renderer_listbox_surface.insert(2, "rho")
        renderer_listbox_surface.insert(3, "J_abs")
        renderer_listbox_surface.pack(pady=10)
        renderer_listbox_surface.bind('<<ListboxSelect>>', update_surface_type)

        # Use ttk.Scale and ttk.Label for the sliders
        fov_label = ttk.Label(ui, text="FOV")
        fov_slider = ttk.Scale(ui, from_=10, to=120, orient='horizontal')
        fov_slider.set(45)  # Set default FOV to 45 degrees
        fov_label.pack(pady=5)
        fov_slider.pack(pady=10)

        camera_distance_label = ttk.Label(ui, text="Camera distance")
        camera_distance_slider = ttk.Scale(ui, from_=1, to=20, orient='horizontal')
        camera_distance_slider.set(1)  
        camera_distance_label.pack(pady=5)
        camera_distance_slider.pack(pady=10)

        camera_phi_label = ttk.Label(ui, text="Camera azimuth")
        camera_phi_slider = ttk.Scale(ui, from_=-np.pi, to=np.pi, orient='horizontal')
        camera_phi_slider.set(0)  # Set default camera position to 3 units away
        camera_phi_label.pack(pady=5)
        camera_phi_slider.pack(pady=10)

        camera_theta_label = ttk.Label(ui, text="Camera elevation")
        camera_theta_slider = ttk.Scale(ui, from_=0, to=np.pi/2, orient='horizontal')
        camera_theta_slider.set(0)  # Set default camera position to 3 units away
        camera_theta_label.pack(pady=5)
        camera_theta_slider.pack(pady=10)

        plot_scale_label = ttk.Label(ui, text="Plot scale")
        plot_scale_slider = ttk.Scale(ui, from_=0.1, to=20, orient='horizontal')
        plot_scale_slider.set(1)  # Set default camera position to 3 units away
        plot_scale_label.pack(pady=5)
        plot_scale_slider.pack(pady=10)

        ui.protocol("WM_DELETE_WINDOW", close_window)
    else:
        RUN = True
    
    if RENDER:
        if grid_type == '2d':
            renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, renderer_type='surface_plot', is_3d=False)
            #window = surf.initialize_window()
        elif grid_type == '3d':
            #renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, renderer_type=selected_render_type, is_3d=True)
            renderer = Render.Simple3DParticleRenderer(width=SCREEN_SIZE[0], height=SCREEN_SIZE[1], use_orthographic=False)
        if DIAGNOSTICS:
            renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, renderer_type="heatmap")
        

    # MAIN LOOP

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
                
                if last_alive < N-1:
                    last_alive += 1
                    R[0, last_alive] = X * 0.5 + random.uniform(-dx, dx)
                    R[1, last_alive] = X * 0.5 + random.uniform(-dx, dx)
                    V[:, last_alive] = 0
                    part_type[last_alive] = random.randint(1, 2)
                    #print(part_type)    

                last_alive = Update.update_R(R, V, X, Y, dt, last_alive, part_type)
                Update.update_density_gpu(R, part_type, rho, dx, dy, gridsize, q_type, last_alive)
                #Update.update_current_density_gpu(R, V, part_type, J, X, Y, gridsize, q_type)
                #J_abs = cp.hypot(*J)

                if boundary != None:
                    rho[bound] = bound_val
                if solver == 'inverse':
                    phi[:] = cp.dot(Lap, -rho * Consts.eps0_1 * dx * dy)
                elif solver == 'fft':
                    #phi, error = fft_solver.solve(-rho)
                    phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
                elif solver == 'cg':
                    phi = Solvers.solve_poisson_pcg_gpu(rho, m, n, dx, dy, Consts.eps0, phi0=phi, tol=1e-2, preconditioner='none')
                elif solver == 'multigrid':
                    Solvers.restrict_grid(cp.reshape(rho, (m,n)), rho_multigrid[0])
                    Solvers.restrict_grid(rho_multigrid[0], rho_multigrid[1])
                    phi_multigrid[1] = cp.dot(Lap_multigrid, -rho_multigrid[1].flatten() * Consts.eps0_1 * dx * dy).reshape(phi_multigrid[1].shape)
                    Solvers.interpolate_grid(phi_multigrid[1], phi_multigrid[0])
                    phi_tmp_1 = Solvers.solve_poisson_pcg_gpu(rho_multigrid[0].flatten(), *phi_multigrid[0].shape, X/phi_multigrid[0].shape[0], Y/phi_multigrid[0].shape[1], Consts.eps0, phi0=phi_multigrid[0].flatten(), max_iter=10)
                    phi_tmp = cp.zeros((m,n))
                    Solvers.interpolate_grid(phi_tmp_1.reshape(phi_multigrid[0].shape), phi_tmp)
                    #phi = phi_tmp.flatten()
                    phi = Solvers.solve_poisson_pcg_gpu(rho, m, n, dx, dy, Consts.eps0, phi0=phi_tmp.flatten(), max_iter=10)
                Update.updateE_gpu(E, phi, X, Y, gridsize)
                Update.update_V(R, V, E, part_type, q_type, m_type_1, gridsize, dt, dx, dy, last_alive)

                ''' 
                part_energy = Update.kinetic_energy_ev(V, m_type, part_type)
                Update.update_cross_section(part_cross_section, part_energy, cross_section)                
                Update.MCC(part_cross_section, V, 1e23, mcc_probability, dt)
                mcc_random[:] = cp.random.random(N)
                collision = mcc_random < mcc_probability
                rand_coll_num = cp.random.uniform(0, 1, collision.size)
                coll_cos = Update.collision_cos(part_energy, rand_coll_num)
                V[:, collision] = cp.hypot(V[0, collision], V[1, collision]) * coll_cos
                '''
                

            elif grid_type == '3d':
                if I < N-1:
                    R[0, I] = X * 0.5 + random.uniform(-dx, dx)
                    R[1, I] = X * 0.5 + random.uniform(-dx, dx)
                    R[2, I] = X * 0.5 + random.uniform(-dx, dx)
                    V[:, I] = 0
                    tmp = random.randint(1, 2)
                    part_type[I] = tmp
                    part_colors[I] = part_color_types[tmp]
                    I += 1
                R[:] += V[:] * dt
                Update.update_density_gpu_3d(R, part_type, rho, X, Y, Z, gridsize, q_type)
                if solver == 'fft':
                    phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
                Update.updateE_gpu_3d(E, phi, X, Y, Z, gridsize)
                Update.update_V_3d(R, V, E, part_type, q_type, m_type_1, gridsize, dt, X, Y, Z)

            t += dt
            sim_time = time.time() - start_time

            if not RENDER and framecounter == 100:
                framecounter = 0
                print(f"t = {t:.2e}, sim_time = {sim_time:.2e}")

            # RENDER
            framecounter += 1
            if framecounter == RENDER_FRAME and RENDER:
                
                if UI:
                    cam_fov = fov_slider.get()  # Convert FOV to radians
                    cam_r = camera_distance_slider.get()
                    cam_phi = camera_phi_slider.get()
                    cam_theta = camera_theta_slider.get()
                    camera_pos_value = np.array([cam_r * np.cos(cam_phi), cam_r  * np.sin(cam_phi), cam_r * np.sin(cam_theta)]) 
                else:
                    cam_fov = 45
                    cam_r = 1
                    cam_phi = 0
                    cam_theta = 0

                KE = Update.total_kinetic_energy(V, m_type, part_type)
                PE = Update.total_potential_energy(rho, phi, dx, dy)
                TE = PE + KE
                P = Update.total_momentum(V, m_type, part_type)

                dt_x = cp.reciprocal(V[0, :last_alive]) * dx
                dt_y = cp.reciprocal(V[1, :last_alive]) * dy
                dt_sq = cp.hypot(dt_x, dt_y)
                dt_max = cp.min(dt_sq)
                #dt = dt_max * 0.00001

                hystory_x = cp.append(hystory_x, t)
                hystory_y = cp.append(hystory_y, TE)
                hystory_y2 = cp.append(hystory_y2, P)

                renderer.label_list = []
                
                if selected_render_type == "particles":
                    renderer.renderer_type = "particles"
                    if grid_type == '3d':
                        renderer.update_camera(fov=cam_fov, angle_phi=cam_phi, angle_theta=cam_theta, distance=cam_r)
                        R_cpu = (R[:, :last_alive] * 2 - 1).T.get()
                        #part_color_cpu = part_color[part_type[:I]].get()
                        renderer.setup_particles(R_cpu)
                    else:
                        renderer.update_particles(R[:, :last_alive] * cam_r, part_type[:last_alive], X * cam_r, Y * cam_r, 0, 0)
                elif selected_render_type == "heatmap" and grid_type == '2d':
                    renderer.renderer_type = "heatmap"
                    renderer.update_heatmap(phi, m, n)
                elif selected_render_type == "line_plot_E":
                    renderer.renderer_type = "line_plot"
                    renderer.line_plot_type = "Energy"
                    renderer.update_line_data('Energy', hystory_x, hystory_y)
                    renderer.update_legend('line_plot_x0', '0.0', SCREEN_SIZE[1] - 20, (255, 255, 255))
                    renderer.update_legend('line_plot_x1', f'{t:.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255), SCREEN_SIZE[0] - 100, 0.5)
                    renderer.update_legend('line_plot_y0', f'{cp.min(hystory_y):.2e}', SCREEN_SIZE[1] - 60, (255, 255, 255))
                    renderer.update_legend('line_plot_y1', f'{cp.max(hystory_y):.2e}', 30, (255, 255, 255))
                    renderer.label_list.extend(['line_plot_x0', 'line_plot_x1', 'line_plot_y0', 'line_plot_y1'])
                elif selected_render_type == "line_plot_P":
                    renderer.renderer_type = "line_plot"
                    renderer.line_plot_type = "Momentum"
                    renderer.update_line_data('Momentum', hystory_x[1:], hystory_y2[1:], color=(0.0, 0.5, 0.5))
                    renderer.update_legend('line_plot_x0', '0.0', SCREEN_SIZE[1] - 20, (255, 255, 255))
                    renderer.update_legend('line_plot_x1', f'{t:.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255), SCREEN_SIZE[0] - 100, 0.5)
                    renderer.update_legend('line_plot_y0', f'{cp.min(hystory_y2):.2e}', SCREEN_SIZE[1] - 60, (255, 255, 255))
                    renderer.update_legend('line_plot_y1', f'{cp.max(hystory_y2):.2e}', 30, (255, 255, 255))
                    renderer.label_list.extend(['line_plot_x0', 'line_plot_x1', 'line_plot_y0', 'line_plot_y1'])
                elif selected_render_type == "line_plot_MCC":
                    renderer.renderer_type = "line_plot"
                    renderer.line_plot_type = "MCC"
                    dist_x, dist_y = Update.collision_probability_distribution(mcc_probability, num_bins=128)
                    renderer.update_line_data('MCC', dist_x, dist_y)
                    renderer.update_legend('line_plot_x0', f'{dist_x[0]:.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255))
                    renderer.update_legend('line_plot_x1', f'{dist_x[-1]:.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255), SCREEN_SIZE[0] - 100, 0.5)
                    renderer.update_legend('line_plot_y0', f'{cp.min(dist_y):.2e}', SCREEN_SIZE[1] - 60, (255, 255, 255))
                    renderer.update_legend('line_plot_y1', f'{cp.max(dist_y):.2e}', 30, (255, 255, 255))
                    renderer.label_list.extend(['line_plot_x0', 'line_plot_x1', 'line_plot_y0', 'line_plot_y1'])
                elif selected_render_type == "line_plot_E_distribution":
                    renderer.renderer_type = "line_plot"
                    renderer.line_plot_type = "E_distribution"
                    dist_x, dist_y = Update.KE_distribution(part_type= part_type, v=V, M=m_type, bins=128)
                    renderer.update_line_data('E_distribution', dist_x, dist_y)
                    renderer.update_legend('line_plot_x0', f'{dist_x[0]:.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255))
                    renderer.update_legend('line_plot_x1', f'{dist_x[-1]:.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255), SCREEN_SIZE[0] - 100, 0.5)
                    renderer.update_legend('line_plot_y0', f'{cp.min(dist_y):.2e}', SCREEN_SIZE[1] - 60, (255, 255, 255))
                    renderer.update_legend('line_plot_y1', f'{cp.max(dist_y):.2e}', 30, (255, 255, 255))
                    renderer.label_list.extend(['line_plot_x0', 'line_plot_x1', 'line_plot_y0', 'line_plot_y1'])
                elif selected_render_type == "line_plot_V_distribution":
                    renderer.renderer_type = "line_plot"
                    renderer.line_plot_type = "V_distribution"
                    dist_x, dist_y = Update.V_distribution(V, bins=128)
                    dist_x, dist_y = dist_x[1:], dist_y[1:]
                    renderer.update_line_data('V_distribution', dist_x, dist_y)
                    renderer.update_legend('line_plot_x0', f'{dist_x[0]:.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255))
                    renderer.update_legend('line_plot_x1', f'{dist_x[-1]:.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255), SCREEN_SIZE[0] - 100, 0.5)
                    renderer.update_legend('line_plot_y0', f'{cp.min(dist_y):.2e}', SCREEN_SIZE[1] - 60, (255, 255, 255))
                    renderer.update_legend('line_plot_y1', f'{cp.max(dist_y):.2e}', 30, (255, 255, 255)) 
                    renderer.label_list.extend(['line_plot_x0', 'line_plot_x1', 'line_plot_y0', 'line_plot_y1'])
                elif selected_render_type == "surface_plot":
                    surf_scale = plot_scale_slider.get()
                    renderer.renderer_type = "surface_plot"
                    renderer.set_fov(cam_fov)
                    renderer.set_camera(camera_pos_value)
                    
                    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, m))
                    #surafce_plot_var_name = 'phi'
                    surf_max = cp.max(locals()[surafce_plot_var_name])
                    surf_min = cp.min(locals()[surafce_plot_var_name])
                    if surf_max == 0:
                        surf_max_1 = 1
                    else:
                        surf_max_1 = 1/surf_max
                    z = cp.asnumpy(cp.reshape(locals()[surafce_plot_var_name]*surf_max_1*surf_scale, (m, n)))
                    renderer.update_surface(x, y, z)

                    '''
                    a, b = rho_multigrid[0].shape
                    x, y = np.meshgrid(np.linspace(-1, 1, b), np.linspace(-1, 1, a))
                    surf_max = cp.max(phi_multigrid[0])
                    surf_min = cp.min(phi_multigrid[0])
                    surf_max_1 = 1/surf_max
                    z = cp.asnumpy(cp.reshape(phi_multigrid[0]*surf_max_1*surf_scale, (a, b)))
                    renderer.update_surface(x, y, z)
                    '''

                    renderer.update_legend('surface_max', f"{surafce_plot_var_name}_max: {surf_max:.2e} V", 210)
                    renderer.update_legend('surface_min', f"{surafce_plot_var_name}_min: {surf_min:.2e} V", 240)
                    renderer.label_list.extend(['surface_max', 'surface_min'])
                
                
                if TEXTUI and grid_type != '3d':
                    renderer.update_legend('sim', f"Sim time: {(sim_time)*1000:.1f} ms", 80)
                    renderer.update_legend('frame', f"Frame time: {(frame_time)*1000:.1f} ms", 110)
                    renderer.update_legend('n', f"N: {last_alive+1}", 140)
                    renderer.update_legend('dt_max', f"dt_max: {dt_max:.2e}", 170)
                    renderer.update_legend('dt', f"dt: {dt:.2e}", 200)
                    renderer.label_list.extend(['sim', 'frame', 'n', 'dt_max', 'dt'])
                else:
                    print(f"Sim time: {(sim_time)*1000:.1f} ms", f'frame time: {(frame_time)*1000:.1f} ms')

                if grid_type == '3d':
                    renderer.render()
                elif grid_type == '2d':
                    renderer.render(clear = not TRACE, TEXT_RENDERING=TEXTUI)

            frame_time = time.time() - start_time

            if UI and RENDER:
                if renderer.should_close():
                    break

                framecounter = 0

    if RENDER:
        renderer.close()
        if DIAGNOSTICS:
            renderer.close()
        print('Renderer closed')
    if UI and not FINISH:
        ui.destroy()

    return 0


