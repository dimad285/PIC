import cupy as cp
import numpy as np
import Update
import collisions
import Solvers
import Render
import Particles
import Grid


class Boundaries():
    pass

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
        return cp.sum(cp.hypot(particles.V[0, :particles.last_alive], particles.V[1, :particles.last_alive]) * particles.m_type[particles.part_type[:particles.last_alive]])

    def update(self, t, particles, grid, sim_time):

        KE = self.total_kinetic_energy(particles)
        PE = self.total_potential_energy(grid)
        TE = PE + KE
        P = self.total_momentum(particles)

        self.history_t = cp.append(self.history_t, t)
        self.history_E = cp.append(self.history_E, TE)
        self.history_P = cp.append(self.history_P, P)
        self.history_sim_time = cp.append(self.history_sim_time, sim_time)



def init_simulation(m, n, X, Y, N, dt, species):
    pass



def step(particles:Particles.Particles2D, grid:Grid.Grid2D, dt, solver:Solvers.Solver, walls=None):

    particles.update_R(dt)
    #print(particles.last_alive)
    #collisions.detect_collisions_simple(particles, grid, *walls)
    particles.update_bilinear_weights(grid)
    #particles.sort_particles_sparse(grid.cell_count)
    grid.update_density(particles)
    solver.solve(grid)
    grid.update_E()
    particles.update_V(grid, dt)



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
                    renderer.update_particles(particles.R, 0, X, 0, Y)
                    if bound_tuple != None:
                        renderer.update_boundaries(bound_tuple, (m, n))
                case "V":
                    renderer.update_particles(particles.V)
        
        case "heatmap":
            renderer.renderer_type = "heatmap"
            renderer.update_heatmap(grid.phi, m, n)
                    
        case "line_plot":
            match plot_var_name:
                case "Energy":
                    line_plot(renderer, diagnostics.history_t, diagnostics.history_E, SCREEN_SIZE, "Energy")
                case "Momentum":
                    line_plot(renderer, diagnostics.history_t, diagnostics.history_P, SCREEN_SIZE, "Momentum")
                case "sim_time":
                    line_plot(renderer, diagnostics.history_t, diagnostics.history_sim_time, SCREEN_SIZE, "sim_time")
                case "distribution_V":
                    dist_x, dist_y = Update.V_distribution(particles.V, bins=128)
                    line_plot(renderer, dist_x, dist_y, SCREEN_SIZE, "distribution_V")
                case "distribution_E":
                    dist_x, dist_y = Update.KE_distribution(part_type=particles.part_type, v=particles.V, M=particles.m_type, bins=128)
                    line_plot(renderer, dist_x, dist_y, SCREEN_SIZE, "distribution_E")
        
        case "surface_plot":
            surf_scale = 1
            renderer.renderer_type = "surface_plot"
            #camera.r = gui.cam_dist.get()
            renderer.set_fov(camera.fov)
            renderer.set_camera(camera.position)
            if state['simulation_running']:

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
        renderer.label_list.extend(['sim', 'frame', 'n', 'dt'])

    renderer.render(clear = not state['trace_enabled'], TEXT_RENDERING=state['text_enabled'], BOUNDARY_RENDERING = True)


def uniform_particle_generator_2d(R, V, last_alive, part_type, x, y, dx, dy):
    R[0, last_alive] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5)
    R[1, last_alive] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5)
    V[:, last_alive] = 0
    part_type[last_alive] = cp.random.randint(1, 3)
    last_alive += 1

def uniform_particle_load(R, V, last_alive, part_type, x, y, dx, dy, n):
    R[0, last_alive:last_alive + n] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5, n)
    R[1, last_alive:last_alive + n] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5, n)
    V[:, last_alive:last_alive + n] = 0
    part_type[last_alive:last_alive + n] = cp.random.randint(1, 3, n)
    last_alive += n

def uniform_species_load(R, V, last_alive, part_type, part_name, x, y, dx, dy, n, species):
    R[0, last_alive:last_alive + n] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5, n)
    R[1, last_alive:last_alive + n] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5, n)
    V[:, last_alive:last_alive + n] = 0
    part_type[last_alive:last_alive + n] = part_name.index(species)
    last_alive += n