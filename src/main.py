import cupy as cp
#import cupyx as cpx
from cupyx.scipy.sparse.linalg import gmres
import numpy as np
import time
import Render 
import Update 
import Solvers 
import Consts 
import simulation
from Parser import parse_config
import sys
import Interface
import tkinter as tk
import cProfile
import pstats
import io





def line_plot(renderer, data_x, data_y, SCREEN_SIZE, plot_type):
    renderer.renderer_type = "line_plot"
    renderer.line_plot_type = plot_type
    renderer.update_line_data(plot_type, data_x, data_y)
    renderer.update_legend('line_plot_x0', '0.0', SCREEN_SIZE[1] - 20, (255, 255, 255))
    renderer.update_legend('line_plot_x1', f'{cp.max(data_x):.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255), SCREEN_SIZE[0] - 100, 0.5)
    renderer.update_legend('line_plot_y0', f'{cp.min(data_y):.2e}', SCREEN_SIZE[1] - 60, (255, 255, 255))
    renderer.update_legend('line_plot_y1', f'{cp.max(data_y):.2e}', 30, (255, 255, 255))
    renderer.label_list.extend(['line_plot_x0', 'line_plot_x1', 'line_plot_y0', 'line_plot_y1'])


def run_gpu(m = 16, n = 16, k = 0, X = 1, Y = 1, Z = 1, max_particles = 1000, dt = 0.001, grid_type='2d',
            boundary = None, RENDER = True, UI = True, RENDER_FRAME = 1, 
            SCREEN_SIZE = (512, 512), solver = 'inverse'):
    
    print('Starting...')
    # Graphical user interface
    fontfile = "C:\\Windows\\Fonts\\Arial.ttf"
    selected_render_type = "particles"  # default render type
    plot_var_name = 'R'
    framecounter = 0
    t = 0
    sim_time = 0
    frame_time = 0
    MCC = False
    
    
    # INIT

    # Grid parameters
    print('creating arrays...')
    
    if grid_type == '2d' or grid_type == 'cylindrical_2':

        gridsize = (m, n)
        dx, dy = X / (m-1), Y / (n-1)

        rho = cp.zeros((m*n), dtype=cp.float32) # charge density
        b = cp.zeros((m*n), dtype=cp.float32) # right hand side of Poisson equation
        J = cp.zeros((2, m*n), dtype=cp.float32) # current density

        phi = cp.zeros((m*n), dtype=cp.float32) # scalar potential
        A = cp.zeros((2, m*n), dtype=cp.float32) # vector potential

        B = cp.zeros((2, m*n), dtype=cp.float32) # magnetic field
        E = cp.zeros((2, m*n), dtype=cp.float32) # electric field

        R = cp.zeros((2, max_particles), dtype=cp.float32) # particle positions
        V = cp.zeros((2, max_particles), dtype=cp.float32) # particle velocities

        

    elif grid_type == '3d':
        gridsize = (m, n, k)
        dx, dy, dz = X / (m-1), Y / (n-1), Z / (k-1)
        rho = cp.zeros((m*n*k), dtype=cp.float64)
        phi = cp.zeros((m*n*k), dtype=cp.float64)
        A = cp.zeros((3, m*n*k), dtype=cp.float64)
        B = cp.zeros((3, m*n*k), dtype=cp.float64)
        J = cp.zeros((3, m*n*k), dtype=cp.float64)
        E = cp.zeros((3, m*n*k), dtype=cp.float64)
        R = cp.random.uniform(X/4, X*3/4, (3, max_particles)).astype(cp.float64)
        V = cp.zeros((3, max_particles), dtype=cp.float64)
        part_color_types = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.5, 0.0]])
        #part_colors = np.zeros((max_particles, 3), dtype=cp.float32)


    # Particle parameters
    part_type = cp.zeros(max_particles, dtype=cp.int32) # 0 for dead particle. 
    R_grid = cp.zeros(max_particles, dtype=cp.int32) # particle positions in grid space (in which cell they are)
    #R_grid_new = cp.zeros(max_particles, dtype=cp.int32)
    # When paarticle 'dies' it's type is set to 0 and it is swapped with the last alive particle
    last_alive = 0 # index of last alive particle
    # Problems can be with particle sorting
    part_name = ['', 'proton', 'electron']
    m_type = cp.array([0, Consts.mp, Consts.me], dtype=cp.float64)
    m_type_1 = cp.array([0, 1/Consts.mp, 1/Consts.me], dtype=cp.float64)
    q_type = cp.array([0, 1.0 * Consts.qe, -1.0 * Consts.qe], dtype=cp.float64)
    weights = cp.zeros((4, max_particles), dtype=cp.float32)
    indices = cp.zeros((4, max_particles), dtype=cp.int32)
    
    # MCC
    if MCC:
        cross_section = Update.read_cross_section('c:/Users/Dima/Desktop/Proga/Cuda/csf/H.txt')
        mcc_probability = cp.zeros(max_particles, dtype=cp.float32)
        mcc_random = cp.zeros(max_particles, dtype=cp.float32)
        part_energy = cp.zeros(max_particles, dtype=cp.float64) 
        part_cross_section = cp.zeros(max_particles, dtype=cp.float32)
    
    # Diagnostics
    history_t = np.zeros(0, dtype=np.float32)
    history_E = np.zeros(0, dtype=np.float32)
    history_P = np.zeros(0, dtype=np.float32)
    history_sim_time = np.zeros(0, dtype=np.float32)
    #history_n = np.array([], dtype=cp.float32)
    #history_ne = np.array([], dtype=cp.float32)
    #history_ni = np.array([], dtype=cp.float32)
    
    if boundary != None:
        print('creating boundary array...')
        bound, bound_val = Solvers.boundary_array(boundary, gridsize)
        coll_map = Update.collision_map_new(bound, gridsize)
        #print(coll_map, bound)
        bound_tuple = []
        for i in boundary:
            x0, y0, x1, y1 = i[0]
            bound_tuple.append((int(x0), int(y0), int(x1), int(y1)))
        #print(bound)
        #print(bound_val)

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
        Lap = Solvers.Laplacian_square_csr(m, n)
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
            phi_multigrid.append(cp.zeros((m//(2**i) + 1, n//(2**i) + 1), dtype=cp.float32))
            rho_multigrid.append(cp.zeros((m//(2**i) + 1, n//(2**i) + 1), dtype=cp.float32))
        Lap_multigrid = Solvers.Laplacian_square(*phi_multigrid[-1].shape)
    elif solver == 'gmres':
        print('Using GMRES solver')
        Lap = Solvers.Laplacian_square_csr(m, n)
        if boundary != None:
            print('applying boundary conditions...')
            Solvers.apply_boundary_conditions(Lap, bound, bound_val)
        #Lap = Solvers.Laplacian_square(m, n)
        #cp.savetxt('laplacian.txt', Lap.todense(), fmt = '%i')
        #cp.savetxt('laplacian2.txt', Lap2, fmt = '%i')

    if UI:
        root = tk.Tk()
        gui = Interface.SimulationUI_tk(root)
    if RENDER:
        if grid_type == '2d':
            renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, max_particles, renderer_type='surface_plot', is_3d=False)
            #window = surf.initialize_window()
        elif grid_type == '3d':
            #renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, renderer_type=selected_render_type, is_3d=True)
            renderer = Render.Simple3DParticleRenderer(width=SCREEN_SIZE[0], height=SCREEN_SIZE[1], use_orthographic=False)

    # MAIN LOOP

    simulation.uniform_particle_load(X * 0.5, Y * 0.5, dx, dy, R, V, part_type, last_alive, max_particles//2)
    last_alive += max_particles//2

    print('running...')
    while True:
        if UI:
            root.update()
            state = gui.get_state()
        start_time = time.time()
        
        if state["simulation_running"]:
            # UPDATE
            if grid_type == '2d':
                
                Update.update_R(R, V, X, Y, dt, last_alive, part_type)
                #R_grid[:] = R_grid_new[:]
                Update.compute_bilinear_weights(R, dx, dy, gridsize, last_alive, weights, indices, R_grid=R_grid)
                #active_cells, counts, sorted_indices = Update.sort_particles_sparse(R_grid[:last_alive], (n-1)*(m-1))
                cell_counts, sorted_indices = Update.sort_particles_counting(R_grid[:], (m-1)*(n-1))
    
                # Verify results
                Update.verify_sorting(R_grid[:last_alive], cell_counts, sorted_indices)
                #coll_part = Update.check_collisions(R_grid_new, coll_map, last_alive)
                #print(R_grid[:last_alive]) 
                #print(active_cells)
                #print(counts)
                #print(sorted_indices)
                #last_alive = Update.remove_collided_particles(R, V, part_type, coll_part, last_alive)
                Update.update_density_gpu(part_type, rho, q_type, last_alive, weights, indices)
                #Update.update_current_density_gpu(R, V, part_type, J, X, Y, gridsize, q_type)
                #J_abs = cp.hypot(*J)

                if boundary != None:
                    b[:] = -rho * Consts.eps0_1
                    b[bound] = bound_val
                else:
                    b[:] = -rho * Consts.eps0_1

                match solver:
                    case 'inverse':
                        phi[:] = cp.dot(Lap, b)
                    case 'fft':
                        #phi, error = fft_solver.solve(-rho)
                        phi = Solvers.solve_poisson_fft(rho, k_sq, Consts.eps0)
                    case 'cg':
                        #phi, iter = cpx.scipy.sparse.linalg.cg(Lap, -rho * Consts.eps0_1 * dx * dy, x0=phi, tol=1e-2)
                        phi = Solvers.solve_poisson_pcg_gpu(rho, m, n, dx, dy, Consts.eps0, phi0=phi, tol=1e-2, preconditioner='none')
                    case 'multigrid':
                        Solvers.restrict_grid(cp.reshape(rho, (m,n)), rho_multigrid[0])
                        Solvers.restrict_grid(rho_multigrid[0], rho_multigrid[1])
                        phi_multigrid[1] = cp.dot(Lap_multigrid, -rho_multigrid[1].flatten() * Consts.eps0_1 * dx * dy).reshape(phi_multigrid[1].shape)
                        Solvers.interpolate_grid(phi_multigrid[1], phi_multigrid[0])
                        phi_tmp_1 = Solvers.solve_poisson_pcg_gpu(rho_multigrid[0].flatten(), *phi_multigrid[0].shape, X/phi_multigrid[0].shape[0], Y/phi_multigrid[0].shape[1], Consts.eps0, phi0=phi_multigrid[0].flatten(), max_iter=10)
                        phi_tmp = cp.zeros((m,n))
                        Solvers.interpolate_grid(phi_tmp_1.reshape(phi_multigrid[0].shape), phi_tmp)
                        #phi = phi_tmp.flatten()
                        phi = Solvers.solve_poisson_pcg_gpu(rho, m, n, dx, dy, Consts.eps0, phi0=phi_tmp.flatten(), max_iter=10)
                    case 'gmres':
                        phi, iter = gmres(Lap, b, x0=phi, tol=1e-2)
                Update.updateE_gpu(E, phi, X, Y, gridsize)
                Update.update_V(V, E, part_type, q_type, m_type_1, dt, last_alive, weights, indices)

                ''' 
                part_energy = Update.kinetic_energy_ev(V, m_type, part_type)
                Update.update_cross_section(part_cross_section, part_energy, cross_section)                
                Update.MCC(part_cross_section, V, 1e23, mcc_probability, dt)
                mcc_random[:] = cp.random.random(max_particles)
                collision = mcc_random < mcc_probability
                rand_coll_num = cp.random.uniform(0, 1, collision.size)
                coll_cos = Update.collision_cos(part_energy, rand_coll_num)
                V[:, collision] = cp.hypot(V[0, collision], V[1, collision]) * coll_cos
                '''
                

            KE = Update.total_kinetic_energy(V, m_type, part_type, last_alive)
            PE = Update.total_potential_energy(rho, phi, dx, dy)
            TE = PE + KE
            P = Update.total_momentum(V, m_type, part_type, last_alive)

            dt_x = cp.reciprocal(V[0, :last_alive]) * dx
            dt_y = cp.reciprocal(V[1, :last_alive]) * dy
            dt_sq = cp.hypot(dt_x, dt_y)
            if dt_sq.size == 0:
                dt_max = 0
            else:
                dt_max = cp.min(dt_sq)
            #dt = dt_max * 0.00001

            history_t = np.append(history_t, t)
            history_E = np.append(history_E, TE)
            history_P = np.append(history_P, P)
            history_sim_time = np.append(history_sim_time, sim_time)
            
            t += dt
            sim_time = time.time() - start_time

        if not RENDER and framecounter == 100:
            framecounter = 0
            print(f"t = {t:.2e}, sim_time = {sim_time:.2e}")


        # RENDER
        framecounter += 1
        if RENDER and framecounter == RENDER_FRAME:        

            cam_fov = 45  # Convert FOV to radians
            cam_r = 1
            cam_phi = 0
            cam_theta = 0
            camera_pos_value = np.array([cam_r * np.cos(cam_phi), cam_r  * np.sin(cam_phi), cam_r * np.sin(cam_theta)]) 
   
            renderer.label_list = []

            selected_render_type = state["plot_type"]
            plot_var_name = state["plot_variable"]
            
            match selected_render_type:
                case "particles":
                    renderer.renderer_type = "particles"
                    if grid_type == '3d':
                        renderer.update_camera(fov=cam_fov, angle_phi=cam_phi, angle_theta=cam_theta, distance=cam_r)
                        R_cpu = (R[:, :last_alive] * 2 - 1).T.get()
                        renderer.setup_particles(R_cpu)
                    match plot_var_name:
                        case "R":
                            renderer.update_particles(R, 0, X, 0, Y)
                            if boundary != None:
                                renderer.update_boundaries(bound_tuple, gridsize)
                        case "V":
                            #print(V[:, :last_alive])
                            renderer.update_particles(V)
                
                case "heatmap" if grid_type == '2d':
                    renderer.renderer_type = "heatmap"
                    renderer.update_heatmap(locals()[plot_var_name], m, n)\
                         
                case "line_plot":
                    match plot_var_name:
                        case "Energy":
                            line_plot(renderer, history_t, history_E, SCREEN_SIZE, "Energy")
                        case "Momentum":
                            line_plot(renderer, history_t, history_P, SCREEN_SIZE, "Momentum")
                        case "sim_time":
                            line_plot(renderer, history_t, history_sim_time, SCREEN_SIZE, "sim_time")
                        case "distribution_V":
                            dist_x, dist_y = Update.V_distribution(V, bins=128)
                            line_plot(renderer, dist_x, dist_y, SCREEN_SIZE, "distribution_V")
                        case "distribution_E":
                            dist_x, dist_y = Update.KE_distribution(part_type=part_type, v=V, M=m_type, bins=128)
                            line_plot(renderer, dist_x, dist_y, SCREEN_SIZE, "distribution_E")
                
                case "surface_plot":
                    surf_scale = 1
                    renderer.renderer_type = "surface_plot"
                    renderer.set_fov(cam_fov)
                    renderer.set_camera(camera_pos_value)
                    if state['simulation_running']:
                        x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, m))
                        #plot_var_name = 'phi'
                        surf_max = cp.max(locals()[plot_var_name])
                        surf_min = cp.min(locals()[plot_var_name])
                        if surf_max == 0:
                            surf_max_1 = 1
                        else:
                            surf_max_1 = 1/surf_max
                        z = cp.asnumpy(cp.reshape(locals()[plot_var_name]*surf_max_1*surf_scale, (m, n)))
                        renderer.update_surface(x, y, z)
                        renderer.update_legend('surface_max', f"{plot_var_name}_max: {surf_max:.2e} V", 230)
                        renderer.update_legend('surface_min', f"{plot_var_name}_min: {surf_min:.2e} V", 260)
                        renderer.label_list.extend(['surface_max', 'surface_min'])
            
            
            if state['text_enabled'] and grid_type != '3d':
                renderer.update_legend('sim', f"Sim time: {(sim_time)*1000:.1f} ms", 80)
                renderer.update_legend('frame', f"Frame time: {(frame_time)*1000:.1f} ms", 110)
                renderer.update_legend('n', f"N: {last_alive-1}", 140)
                renderer.update_legend('dt_max', f"dt_max: {dt_max:.2e}", 170)
                renderer.update_legend('dt', f"dt: {dt:.2e}", 200)
                renderer.label_list.extend(['sim', 'frame', 'n', 'dt_max', 'dt'])

            if grid_type == '3d':
                renderer.render()
            elif grid_type == '2d':
                    renderer.render(clear = not state['trace_enabled'], TEXT_RENDERING=state['text_enabled'], BOUNDARY_RENDERING = True)

            frame_time = time.time() - start_time

            if RENDER:
                if renderer.should_close():
                    break
                framecounter = 0

    if RENDER:
        renderer.close()
        print('Renderer closed')

    return 0



config = parse_config('c:/Users/Dima/Desktop/Proga/Cuda/input.ini')
# Use parsed parameters
CPU = False
GPU = True
m = config['m']
n = config['n']
k = config['k']
max_particles = config['max_particles']
dt = config['dt']
X = config['X']
Y = config['Y']
Z = config['Z']
boundarys = config['boundarys']

solver_type = 'gmres'
#print(boundarys)

if __name__ == "__main__":
    

    '''
    sys.exit(run_gpu(m, n, k, X, Y, Z, max_particles, dt, grid_type='2d',
                            boundary=boundarys, 
                            RENDER=config['RENDER'], 
                            RENDER_FRAME=config['RENDER_FRAME'], 
                            solver=solver_type.strip("'"), 
                            SCREEN_SIZE=config['SCREEN_SIZE'],
                            UI=config['UI'],))
    '''
    
    pr = cProfile.Profile()
    pr.enable()
    
    run_gpu(m, n, k, X, Y, Z, max_particles, dt, grid_type='2d',
                            boundary=boundarys, 
                            RENDER=config['RENDER'], 
                            RENDER_FRAME=config['RENDER_FRAME'], 
                            solver=solver_type.strip("'"), 
                            SCREEN_SIZE=config['SCREEN_SIZE'],
                            UI=config['UI'],)
    
    pr.disable()
    pr.dump_stats("profile.prof")
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open("profile.txt", "w") as f:
        f.write(s.getvalue())