import cupy as cp
import numpy as np
import collisions
import Solvers
import Render
import Particles
import Grid
import MCC
import time




class Diagnostics():
    
    def __init__(self):
        self.history_t = cp.array([], dtype=cp.float64)
        self.history_E = cp.array([], dtype=cp.float64)
        self.history_P = cp.array([], dtype=cp.float64)
        self.history_sim_time = cp.array([], dtype=cp.float64)

    def total_kinetic_energy(self, particles:Particles.Particles2D) -> float:
        # Compute kinetic energy element-wise and sum
        return 0.5 * cp.sum((particles.V[0, :particles.last_alive] ** 2 + particles.V[1, :particles.last_alive] ** 2) * particles.m_type[particles.part_type[:particles.last_alive]])
    
    def total_potential_energy(self, grid:Grid.Grid2D) -> float:
        dV = grid.dx * grid.dy
        energy_density = 0.5 * grid.rho * grid.phi
        total_potential_energy = cp.sum(energy_density) * dV
        return total_potential_energy
    
    def total_momentum(self, particles:Particles.Particles2D):
        Px = cp.sum(particles.V[0, :particles.last_alive] * particles.m_type[particles.part_type[:particles.last_alive]])
        Py = cp.sum(particles.V[1, :particles.last_alive] * particles.m_type[particles.part_type[:particles.last_alive]])
        return cp.hypot(Px, Py)

    def update(self, t, particles, grid, sim_time):

        KE = self.total_kinetic_energy(particles)
        PE = self.total_potential_energy(grid)
        TE = PE + KE
        P = self.total_momentum(particles)

        self.history_t = cp.append(self.history_t, t)
        self.history_E = cp.append(self.history_E, TE)
        self.history_P = cp.append(self.history_P, P)
        self.history_sim_time = cp.append(self.history_sim_time, sim_time)

    
    def KE_distribution(part_type, v:cp.ndarray, M:cp.ndarray, bins:int) -> list:
        E = (v[0]**2 + v[1]**2)*M[part_type]*0.5
        x = np.arange(bins)*cp.asnumpy(cp.max(E))/bins
        return (x, cp.asnumpy(cp.histogram(E, bins, density=True)[0]))

    def V_distribution(v:cp.ndarray, bins:int) -> list:
        x = np.arange(bins)
        return (x, cp.asnumpy(cp.histogram(cp.hypot(v[0], v[1]), bins, density=True)[0]))
    

    def check_gauss_law_2d(E, rho, epsilon_0, dx, dy, nx, ny):
        """
        Check Gauss's law for a 2D grid using cupy arrays.
        
        Parameters:
        E (cupy.ndarray): Electric field with shape (2, nx*ny), E[0] is Ex and E[1] is Ey.
        rho (cupy.ndarray): Charge density with shape (nx*ny).
        epsilon_0 (float): Permittivity of free space.
        dx, dy (float): Grid spacing in x and y directions.
        nx, ny (int): Number of grid points in x and y directions.
        
        Returns:
        tuple: line_integral, area_integral, relative_error
        """
        # Reshape E and rho to 2D grids
        Ex = E[0].reshape(nx, ny)
        Ey = E[1].reshape(nx, ny)
        rho_2d = rho.reshape(nx, ny)
        
        # Compute line integral of the electric field (flux through boundaries)
        line_integral = (
            cp.sum(Ex[0, :]) * dy - cp.sum(Ex[-1, :]) * dy +  # Top and bottom boundaries
            cp.sum(Ey[:, -1]) * dx - cp.sum(Ey[:, 0]) * dx    # Right and left boundaries
        )
        
        # Compute area integral of charge density (total charge divided by epsilon_0)
        total_charge = cp.sum(rho_2d) * dx * dy
        area_integral = total_charge / epsilon_0
        
        # Compute relative error, handle zero area_integral
        if abs(area_integral) > 1e-12:  # Avoid division by near-zero values
            relative_error = abs(line_integral - area_integral) / abs(area_integral)
        else:
            relative_error = float('inf')  # Undefined error if area_integral is zero
        
        return line_integral, area_integral, relative_error


    def compute_divergence_error(E, rho, epsilon_0, dx, dy, nx, ny):
        """
        Compute the norm of the error between divergence of E and rho / epsilon_0,
        and the surface integral of the divergence over the domain.
        
        Parameters:
        E (cupy.ndarray): Electric field with shape (2, nx*ny), E[0] is Ex and E[1] is Ey.
        rho (cupy.ndarray): Charge density with shape (nx*ny).
        epsilon_0 (float): Permittivity of free space.
        dx, dy (float): Grid spacing in x and y directions.
        nx, ny (int): Number of grid points in x and y directions.
        
        Returns:
        tuple: (error_norm, surface_integral, boundary_flux)
            - error_norm (float): L2 norm of the error.
            - surface_integral (float): Integral of divergence of E over the domain.
            - boundary_flux (float): Line integral of E field over the domain boundary.
        """
        # Reshape E and rho to 2D grids
        Ex = E[0].reshape(nx, ny)
        Ey = E[1].reshape(nx, ny)
        rho_2d = rho.reshape(nx, ny)
        
        # Compute divergence of E
        divE = cp.zeros((nx, ny), dtype=cp.float32)
        
        # Central differences for interior points
        divE[1:-1, 1:-1] = (
            (Ex[2:, 1:-1] - Ex[:-2, 1:-1]) / (2 * dx) + 
            (Ey[1:-1, 2:] - Ey[1:-1, :-2]) / (2 * dy)
        )
        
        # One-sided differences for boundaries
        divE[0, :] = (Ex[1, :] - Ex[0, :]) / dx  # Bottom
        divE[-1, :] = (Ex[-1, :] - Ex[-2, :]) / dx  # Top
        divE[:, 0] = (Ey[:, 1] - Ey[:, 0]) / dy  # Left
        divE[:, -1] = (Ey[:, -1] - Ey[:, -2]) / dy  # Right
        
        # Compute the error
        error = divE - rho_2d / epsilon_0
        
        # Compute the L2 norm of the error
        error_norm = cp.sqrt(cp.sum(error**2))
        
        # Compute the surface integral of divergence
        surface_integral = cp.sum(divE) * dx * dy
        
        return error_norm, surface_integral

    def Vx_distribution(v:cp.ndarray, bins:int) -> list:
        x = np.arange(bins)
        return (x, cp.asnumpy(cp.histogram(v[0], bins, density=True)[0]))

    def Vy_distribution(v:cp.ndarray, bins:int) -> list:
        x = np.arange(bins)
        return (x, cp.asnumpy(cp.histogram(v[1], bins, density=True)[0]))
    
    def P_distribution(v:cp.ndarray, part_type:cp.ndarray, M:cp.ndarray, bins:int) -> list:
        x = np.arange(bins)
        return (x, cp.asnumpy(cp.histogram(cp.hypot(v[0], v[1]) * M[part_type], bins, density=True)[0]))



def init_simulation(m, n, X, Y, N, dt, species):
    pass


def init_step(particles:Particles.Particles2D, grid:Grid.Grid2D, solver:Solvers.Solver, dt:float):
    particles.update_bilinear_weights(grid)
    grid.update_density(particles)
    solver.solve(grid)
    grid.update_E()
    particles.update_V(grid, -dt/2)


def step(particles:Particles.Particles2D, grid:Grid.Grid2D, dt, solver:Solvers.Solver, cross_sections, MAX_PARICLES, walls=None, IONIZATION=True):

    particles.update_R(dt)
    trace = collisions.trace_particle_paths(particles, grid, 5)
    #print(trace)
    #print(collisions.detect_collisions(trace, walls[0]))
    collided = collisions.detect_collisions(trace, walls[0], grid.m-1)
    print(collided)
    #particles.remove(cp.nonzero(collided)[0])
    #collisions.handle_wall_collisions(particles, grid, walls)
    #collisions.remove_out_of_bounds(particles, grid.X, grid.Y)
    particles.update_bilinear_weights(grid)
    grid.update_density(particles)
    solver.solve(grid)
    grid.update_E()
    particles.update_V(grid, dt)
    #MCC.null_collision_method(particles, grid, 1e18, cross_sections, dt, MAX_PARICLES, IONIZATION)
    
    #print(particles.collision_model[particles.part_type[:particles.last_alive]])



def draw(renderer:Render.PICRenderer, state, particles:Particles.Particles2D, grid:Grid.Grid2D,
        camera, frame_time, sim_time, dt, 
        diagnostics:Diagnostics,
        SCREEN_SIZE, bound_tuple=None):
    
    def line_plot(renderer:Render.PICRenderer, data_x, data_y, SCREEN_SIZE, plot_type):
        renderer.renderer_type = "line_plot"
        renderer.line_plot_type = plot_type
        renderer.update_line_data(plot_type, data_x, data_y)
        renderer.update_legend('line_plot_x0', '0.0', SCREEN_SIZE[1] - 20, (255, 255, 255))
        renderer.update_legend('line_plot_x1', f'{cp.max(data_x):.2e}', SCREEN_SIZE[1] - 20, (255, 255, 255), SCREEN_SIZE[0] - 100, 0.5)
        renderer.update_legend('line_plot_y0', f'{cp.min(data_y):.2e}', SCREEN_SIZE[1] - 60, (255, 255, 255))
        renderer.update_legend('line_plot_y1', f'{cp.max(data_y):.2e}', 30, (255, 255, 255))
        renderer.label_list.extend(['line_plot_x0', 'line_plot_x1', 'line_plot_y0', 'line_plot_y1'])
    
    renderer.label_list = []
    selected_render_type = state["plot_type"]
    plot_var_name = state["plot_variable"]
    
    m, n = grid.gridshape
    X, Y = grid.domain

    match selected_render_type:
        case "particles":
            renderer.renderer_type = "particles"
            match plot_var_name:
                case "R":
                    renderer.update_particles(particles.R, particles.part_type, 0, X, 0, Y)
                    if bound_tuple != None:
                        renderer.update_boundaries(bound_tuple, (m, n))
                case "V":
                    renderer.update_particles(particles.V[:2, :particles.last_alive], particles.part_type[:particles.last_alive], 
                                              cp.min(particles.V[0]), cp.max(particles.V[0]), cp.min(particles.V[1]), cp.max(particles.V[1]))
        
        case "heatmap":
            renderer.renderer_type = "heatmap"
            match plot_var_name:
                case "phi":
                    renderer.update_heatmap(grid.phi, m, n)
                case "rho":
                    renderer.update_heatmap(grid.rho, m, n)
                    
        case "line_plot":
            match plot_var_name:
                case "Energy":
                    line_plot(renderer, diagnostics.history_t, diagnostics.history_E, SCREEN_SIZE, "Energy")
                case "Momentum":
                    line_plot(renderer, diagnostics.history_t, diagnostics.history_P, SCREEN_SIZE, "Momentum")
                case "sim_time":
                    line_plot(renderer, diagnostics.history_t, diagnostics.history_sim_time, SCREEN_SIZE, "sim_time")
                case "distribution_V":
                    dist_x, dist_y = Diagnostics.V_distribution(particles.V, bins=128)
                    line_plot(renderer, dist_x[1:], dist_y[1:], SCREEN_SIZE, "distribution_V")
                case "distribution_E":
                    dist_x, dist_y = Diagnostics.KE_distribution(part_type=particles.part_type, v=particles.V, M=particles.m_type, bins=128)
                    line_plot(renderer, dist_x[1:], dist_y[1:], SCREEN_SIZE, "distribution_E")
        
        case "surface_plot":
            surf_scale = 1
            renderer.renderer_type = "surface_plot"
            #camera.r = gui.cam_dist.get()
            renderer.set_fov(camera.fov)
            renderer.set_camera(camera.position)
            #if state['simulation_running']:

            x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, m))
            #plot_var_name = 'phi'
            surf_max = cp.max(grid.phi)
            surf_min = cp.min(grid.phi)
            if surf_max == 0:
                surf_max_1 = 1
            else:
                surf_max_1 = 1/surf_max
            z = cp.asnumpy(cp.reshape(grid.phi*surf_max_1*surf_scale, (m, n)))
            renderer.update_surface(x, y, z)
            renderer.update_legend('surface_max', f"phi_max: {surf_max:.2e} V", 230)
            renderer.update_legend('surface_min', f"phi_min: {surf_min:.2e} V", 260)
            renderer.label_list.extend(['surface_max', 'surface_min'])

            '''
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
            '''
    
    if state['text_enabled']:
        renderer.update_legend('sim', f"Sim time: {(sim_time)*1e6:.1f} mks", 80)
        renderer.update_legend('frame', f"Frame time: {(frame_time)*1e6:.1f} mks", 110)
        renderer.update_legend('n', f"N: {particles.last_alive}", 140)
        renderer.update_legend('dt', f"dt: {dt:.2e}", 200)
        renderer.update_legend('flytime', f"min flytime {(grid.dx / cp.max(cp.linalg.norm(particles.V))):.2e}", 230)
        renderer.label_list.extend(['sim', 'frame', 'n', 'dt', 'flytime'])

    renderer.render(clear = not state['trace_enabled'], TEXT_RENDERING=state['text_enabled'], BOUNDARY_RENDERING = True)




