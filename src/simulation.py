import cupy as cp
import cupyx.profiler
import numpy as np
from . import Collisions
from . import Solvers
from . import Render
from . import Particles
from . import Grid
from . import MCC
import time
from . import Boundaries
from . import Interface
import tkinter as tk
import cupyx
from . import Multigrid
from . import Consts

class SimulationState:
    def __init__(self):
        self.t = 0
        self.sim_time = 0
        self.frame_time = 0
        self.step = 0
        self.framecounter = 0
        self.running = False

        # Core components (set up later)
        self.particles = None
        self.grid = None
        self.solver = None
        self.cross_sections = []
        self.boundaries = None
        self.bound_tuple = ()
        self.walls = None

        # Config options (can be set later)
        self.dt = None
        self.max_particles = None
        self.cylindrical = False
        self.ionization = True
        self.save_fields = False
        self.solver_type = 'gmres'

    def load_config(self, config_dict):
        self.dt = float(config_dict['control']['dt'])
        self.max_particles = int(config_dict['system'].get('MAX_PARTICLES', 100000))
        self.cylindrical = bool(int(config_dict['grid']['Cylindrical']))
        self.ionization = bool(int(config_dict['system'].get('IONIZATION', 1)))
        self.save_fields = config_dict['system'].get('SAVE_FIELDS', False)
        self.solver_type = config_dict['control'].get('solver', 'gmres')

        # Setup particles, grid, solver, etc.
        self.setup_particles(self.max_particles, self.cylindrical, config_dict['species'], config_dict.get('loads', []))
        self.setup_grid(
            int(config_dict['grid']['J']),
            int(config_dict['grid']['K']),
            float(config_dict['grid'].get('x1f', 1.0)) - float(config_dict['grid'].get('x1s', 0.0)),
            float(config_dict['grid'].get('x2f', 1.0)) - float(config_dict['grid'].get('x2s', 0.0)),
            self.cylindrical
        )
        self.setup_solver(self.solver_type, config_dict.get('boundaries', None))
        self.load_cross_sections()


    def setup_particles(self, max_particles, cylindrical, species_data, load_data):
        self.particles = Particles.Particles2D(max_particles, cylindrical=cylindrical)
        for species in species_data:
            self.particles.add_species(
                species['name'], species['m'], species['q'], species['collisionModel']
            )
        for load in load_data:
            self.particles.uniform_species_load(
                x1=load['x1MinMKS'], y1=load['x2MinMKS'],
                x2=load['x1MaxMKS'], y2=load['x2MaxMKS'],
                n=int(load['n']), species=load['speciesName']
            )
        self.particles.set_mcc_mask()

    def setup_grid(self, m, n, X, Y, cylindrical):
        self.m, self.n = m, n
        self.X, self.Y = X, Y
        self.grid = Grid.Grid2D(m, n, X, Y, cylindrical=cylindrical)

    def setup_solver(self, solver_type, boundaries=None):
        if boundaries:
            print('Initializing boundaries')
            self.boundaries = Boundaries.Boundaries(boundaries, self.grid)
            self.bound_tuple = self.boundaries.bound_tuple
            self.walls = self.boundaries.walls


        self.solver = Solvers.Solver(
            solver_type,
            self.grid,
            cylindrical=self.cylindrical,
            boundaries=self.boundaries,
            tol=1e-5
        )

        if self.solver_type == 'multigrid':
            self.grid.phi[self.boundaries.conditions[0]] = self.boundaries.conditions[1]
            
    def load_cross_sections(self):
        self.cross_sections = [
            MCC.read_cross_section('csf/Ar.txt'),
            MCC.read_cross_section('csf/Ar+.txt')
        ]

    def set_particles(self, particles:Particles.Particles2D):
        self.particles = particles

    def set_grid(self, grid:Grid.Grid2D):
        self.grid = grid


class Diagnostics():
    
    def __init__(self, buffer_size=1000):
        # Initialize fixed-size buffers
        self.buffer_size = buffer_size
        self.history_t = cp.zeros(buffer_size, dtype=cp.float64)
        self.history_E = cp.zeros(buffer_size, dtype=cp.float64) 
        self.history_P = cp.zeros(buffer_size, dtype=cp.float64)
        self.history_sim_time = cp.zeros(buffer_size, dtype=cp.float64)
        
        # Track how many entries we've added
        self.entries = 0
        
        # Compile the kernel
        self.total_kinetic_energy_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void compute_energy(
            const float *Vx, const float *Vy,  // [max_particles]
            const int *part_type,              // [max_particles]
            const float *m_type,               // [num_types]
            int last_alive,
            float *energy_out)                  // Output array for results
        {  
            int pid = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Shared memory for block-level reduction
            __shared__ float shared_energy[256]; // Adjust size based on your block size
            
            // Initialize shared memory
            shared_energy[threadIdx.x] = 0.0f;
            
            // Calculate energy for this particle
            if (pid < last_alive) {
                int type = part_type[pid];
                float mass = m_type[type];
                float v_x = Vx[pid];
                float v_y = Vy[pid];
                shared_energy[threadIdx.x] = 0.5f * mass * (v_x * v_x + v_y * v_y);
            }
            
            __syncthreads();
            
            // Reduction in shared memory
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    shared_energy[threadIdx.x] += shared_energy[threadIdx.x + stride];
                }
                __syncthreads();
            }
            
            // First thread in block writes result to global memory
            if (threadIdx.x == 0) {
                atomicAdd(energy_out, shared_energy[0]);
            }
        }
        ''', 'compute_energy')

    def total_kinetic_energy(self, particles):
        # Create output array initialized to zero
        energy_out = cp.zeros(1, dtype=cp.float64)
        
        # Calculate threads and blocks
        threads_per_block = 256
        blocks_per_grid = (particles.last_alive + threads_per_block - 1) // threads_per_block
        
        # Extract velocity components for kernel
        Vx = particles.V[0, :].astype(cp.float64)
        Vy = particles.V[1, :].astype(cp.float64)
        
        # Run the kernel
        self.total_kinetic_energy_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (Vx, Vy, particles.part_type, particles.m_type, 
             particles.last_alive, energy_out)
        )
        
        return float(energy_out[0])

    
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
        # Calculate actual values
        KE = self.total_kinetic_energy(particles)
        PE = self.total_potential_energy(grid)
        TE = KE + PE
        P = self.total_momentum(particles)

        # Shift arrays if buffer is full
        if self.entries >= self.buffer_size:
            # Shift all arrays one position to the left (discard oldest value)
            self.history_t[:-1] = self.history_t[1:]
            self.history_E[:-1] = self.history_E[1:]
            self.history_P[:-1] = self.history_P[1:]
            self.history_sim_time[:-1] = self.history_sim_time[1:]
            
            # Add new value at the end
            self.history_t[-1] = t
            self.history_E[-1] = TE
            self.history_P[-1] = P
            self.history_sim_time[-1] = sim_time
        else:
            # Buffer not full yet, add value at next position
            self.history_t[self.entries] = t
            self.history_E[self.entries] = TE
            self.history_P[self.entries] = P
            self.history_sim_time[self.entries] = sim_time
            self.entries += 1
    
    def get_history_data(self):
        """
        Returns current history data (always in chronological order)
        """
        return {
            't': self.history_t[:self.entries],
            'E': self.history_E[:self.entries],
            'P': self.history_P[:self.entries],
            'sim_time': self.history_sim_time[:self.entries]
        }
    
    def clear_history(self):
        """Reset the history buffers"""
        self.entries = 0

    
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
        divE = cp.zeros((nx, ny), dtype=cp.float64)
        
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


    
def draw(renderer:Render.PICRenderer, ui_state:Interface.UIState, sim_state:SimulationState, diagnostics:Diagnostics=None):
    
    SCREEN_SIZE = (renderer.width, renderer.height)

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
    selected_render_type = ui_state.plot_type
    plot_var_name = ui_state.plot_variable
    
    n, m = sim_state.grid.gridshape
    X, Y = sim_state.grid.domain

    match selected_render_type:
        case "particles":
            renderer.renderer_type = "particles"
            match plot_var_name:
                case "R":
                    renderer.update_particles(sim_state.particles, 0, X, 0, Y)
                    if sim_state.bound_tuple != None:
                        renderer.update_boundaries(sim_state.bound_tuple, (m, n))
                case "V":
                    renderer.update_particles(sim_state.particles.V[:2, :sim_state.particles.last_alive], sim_state.particles.part_type[:sim_state.particles.last_alive], 
                                              cp.min(sim_state.particles.V[0]), cp.max(sim_state.particles.V[0]), cp.min(sim_state.particles.V[1]), cp.max(sim_state.particles.V[1]))
        
        case "heatmap":
            renderer.renderer_type = "heatmap"
            match plot_var_name:
                case "phi":
                    renderer.update_heatmap(sim_state.grid.phi, m, n)
                case "rho":
                    renderer.update_heatmap(sim_state.grid.rho, m, n)
                    
        case "line_plot":
            if diagnostics is not None:
                match plot_var_name:
                    case "Energy":
                        line_plot(renderer, diagnostics.history_t, diagnostics.history_E, SCREEN_SIZE, "Energy")
                    case "Momentum":
                        line_plot(renderer, diagnostics.history_t, diagnostics.history_P, SCREEN_SIZE, "Momentum")
                    case "sim_time":
                        line_plot(renderer, diagnostics.history_t, diagnostics.history_sim_time, SCREEN_SIZE, "sim_time")
                    case "distribution_V":
                        dist_x, dist_y = Diagnostics.V_distribution(sim_state.particles.V, bins=128)
                        line_plot(renderer, dist_x[1:], dist_y[1:], SCREEN_SIZE, "distribution_V")
                    case "distribution_E":
                        dist_x, dist_y = Diagnostics.KE_distribution(part_type=sim_state.particles.part_type, v=sim_state.particles.V, M=sim_state.particles.m_type, bins=128)
                        line_plot(renderer, dist_x[1:], dist_y[1:], SCREEN_SIZE, "distribution_E")
        
        case "surface_plot":
            surf_scale = 1
            renderer.renderer_type = "surface_plot"
            # Update camera parameters based on user input
            eye = np.array(
                [renderer.distance * np.cos(renderer.angle_phi) * np.cos(renderer.angle_theta),
                renderer.distance * np.sin(renderer.angle_phi) * np.cos(renderer.angle_theta),
                renderer.distance * np.sin(renderer.angle_theta)]
            )
            
            renderer.set_fov(45)
            renderer.set_camera(eye)
            
            x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, m))
            
            match plot_var_name:
                case "phi":
                    surf_max = cp.max(sim_state.grid.phi)
                    surf_min = cp.min(sim_state.grid.phi)
                    if surf_max == 0:
                        surf_max_1 = cp.abs(1/surf_min)
                    else:
                        surf_max_1 = cp.abs(1/surf_max)
                    z = cp.asnumpy(cp.reshape(sim_state.grid.phi*surf_scale*surf_max_1, (m, n)))
                    renderer.update_surface(x, y, z)
                    renderer.update_legend('surface_max', f"{plot_var_name}_max: {surf_max:.2e} V", 290)
                    renderer.update_legend('surface_min', f"{plot_var_name}_min: {surf_min:.2e} V", 320)
                    renderer.label_list.extend(['surface_max', 'surface_min'])

                case "rho":
                    surf_max = cp.max(sim_state.grid.rho)
                    surf_min = cp.min(sim_state.grid.rho)
                    if surf_max == 0:
                        surf_max_1 = cp.abs(1/surf_min)
                    else:
                        surf_max_1 = cp.abs(1/surf_max)
                    z = cp.asnumpy(cp.reshape(sim_state.grid.rho*surf_scale*surf_max_1, (m, n)))
                    renderer.update_surface(x, y, z)
                    renderer.update_legend('surface_max', f"{plot_var_name}_max: {surf_max:.2e} C/m^3", 290)
                    renderer.update_legend('surface_min', f"{plot_var_name}_min: {surf_min:.2e} C/m^3", 320)
                    renderer.label_list.extend(['surface_max', 'surface_min'])
            


    
    if ui_state.text_enabled:
        renderer.update_legend('sim', f"Sim time: {(sim_state.sim_time)*1e6:.1f} mks", 80)
        renderer.update_legend('frame', f"Frame time: {(sim_state.frame_time)*1e6:.1f} mks", 110)
        renderer.update_legend('n', f"N: {sim_state.particles.last_alive}", 140)
        renderer.update_legend('dt', f"dt: {sim_state.dt:.2e}", 200)
        renderer.update_legend('flytime', f"min flytime {cp.min(cp.array([cp.float64(sim_state.grid.dx) / cp.max(sim_state.particles.V[0]), cp.float64(sim_state.grid.dy) / cp.max(sim_state.particles.V[1])])):.2e}", 230)
        renderer.update_legend('t', f"t: {sim_state.t:.2e}", 260)
        renderer.label_list.extend(['sim', 'frame', 'n', 'dt', 'flytime', 't'])

    renderer.render(clear = not ui_state.trace_enabled, TEXT_RENDERING=ui_state.text_enabled, BOUNDARY_RENDERING = True)
    #renderer.adjust_particle_count(time.perf_counter() - start_draw)


class SimulationManager:
    def __init__(self, config_dict=None):

        self.config_dict = config_dict
        self.ui_enabled = config_dict['system'].get('UI', True)
        self.render_enabled = config_dict['system'].get('RENDER', True)
        self.screen_size = (
            config_dict['system'].get('SCREEN_WIDTH', 1080),
            config_dict['system'].get('SCREEN_HEIGHT', 1080)
        )
        self.render_frame_skip = config_dict['system'].get('RENDER_FRAME', 1)
        self.diagnostics_enabled = config_dict['system'].get('DIAGNOSTICS', False)

        self.state = SimulationState()
        self.gui = None
        self.renderer = None
        if self.diagnostics_enabled:
            self.diagnostics = Diagnostics()
        else:
            self.diagnostics = None

        if config_dict:
            self.state.load_config(config_dict)
            self.initialize()

    def initialize(self):

        print('Initializing')
        self._init_step(self.state.particles, self.state.grid, self.state.solver, self.state.dt, self.state.boundaries)
        
        if self.ui_enabled:
            print('Initializing UI')
            self.gui = Interface.SimulationUI_tk(tk.Tk())
        if self.render_enabled:
            print('Initializing renderer')
            self.renderer = Render.PICRenderer(
                *self.screen_size, 
                "fonts/Arial.ttf", 
                self.state.max_particles, 
                renderer_type='surface_plot', 
                is_3d=False
            )

    def restart_from_config(self, config_dict):
        del(self.state)
        self.state = SimulationState()
        self.state.load_config(config_dict)
        self.gui.sim_button.config(text="Start Simulation")
        self._init_step(self.state.particles, self.state.grid, self.state.solver, self.state.dt, self.state.boundaries)

    def simulation_step(self):
        if self.ui_enabled:
            ui_state = self.gui.get_state()
            if not (ui_state.running or ui_state.step_requested):
                return
            ui_state.step_requested = False
            if ui_state.restart_requested:
                self.restart_from_config(self.config_dict)
                ui_state.restart_requested = False
                self.state.running = False
                ui_state.running = False
                

        start_time = time.perf_counter()

        self._step(
            self.state.particles,
            self.state.grid,
            self.state.dt,
            self.state.solver,
            self.state
        )

        if self.diagnostics_enabled:

            self.diagnostics.update(
                self.state.t,
                self.state.particles,
                self.state.grid,
                self.state.sim_time
            )

        self.state.t += self.state.dt
        self.state.sim_time = time.perf_counter() - start_time
        self.state.step += 1

    def run(self):
        while not self.should_exit():
            self.update_ui()
            self.simulation_step()
            self.handle_visualization()
        self.cleanup()

    def should_exit(self) -> bool:
        if self.render_enabled and self.renderer and self.renderer.should_close():
            return True
        if self.ui_enabled and self.gui:
            try:
                # Try to access the root widget
                return not self.gui.root.winfo_exists()
            except:
                return True
        return False

    def update_ui(self):
        if self.ui_enabled:
            self.gui.update()
            self.state.running = self.gui.ui_state.running

    def handle_visualization(self):
        if self.render_enabled and self.state.framecounter == self.render_frame_skip:
            start_draw = time.perf_counter()
            draw(
                self.renderer,
                self.gui.get_state() if self.ui_enabled else {},
                self.state
            )
            self.state.frame_time = time.perf_counter() - start_draw
            self.renderer.set_title(f"OOPIC Pro - Step {self.state.step}, t = {self.state.t:.2e}")
            self.state.framecounter = 0
        else:
            self.state.framecounter += 1

    def cleanup(self):
        try:
            if self.state.save_fields:
                field_data = {
                    'rho': self.state.grid.rho,
                    'phi': self.state.grid.phi,
                }
                field_data.update({
                    'Ez': self.state.grid.E[0], 'Er': self.state.grid.E[1]
                } if self.state.cylindrical else {
                    'Ex': self.state.grid.E[0], 'Ey': self.state.grid.E[1]
                })
                filename = 'fields/fields_cylindrical.txt' if self.state.cylindrical else 'fields/fields_cartesian.txt'
                self.state.grid.save_to_txt(filename, fields=field_data)

            if self.render_enabled and self.renderer:
                self.renderer.close()
                self.renderer = None
                print('Renderer closed')
        except Exception as e:
            print(f"Error during cleanup: {e}")


    def _init_step(self, particles:Particles.Particles2D, grid:Grid.Grid2D, solver:Solvers.Solver|Multigrid.MultigridSolver, dt:float, boundaries=None):

        print('Initial Step')

        def bench_solve(solver, grid):
            grid.phi.fill(0)
            print(cupyx.profiler.benchmark(solver.solve, (grid,)))

        grid.update_density(particles)
        if boundaries is not None:
            grid.phi[boundaries.conditions[0]] = boundaries.conditions[1]
        #grid.phi[...] = solver.solve(grid.phi, -grid.rho*Consts.eps0_1, tol=1e-5, max_iter=5, max_cycles=100)
        solver.solve(grid)
        #grid.phi[:], info = solver.solve(grid.phi, -grid.rho*Consts.eps0_1, tol=1e-5, smooth_iter=3, solve_iter=50, max_cycles=100)
        grid.update_E()
        particles.update_V(grid, -dt/2)


    def _step(self, particles:Particles.Particles2D, grid:Grid.Grid2D, dt, solver:Solvers.Solver|Multigrid.MultigridSolver, sim_state:SimulationState=None):

        def coll_step(particles, grid, boundaries):
            trace = Collisions.trace_particle_paths(particles, grid, 10)
            collided = Collisions.detect_collisions(particles, trace, boundaries.wall_lookup)
            particles.R[collided] = particles.R_old[collided]
            to_emit = cp.where(particles.collision_model[particles.part_type[collided]] == 1)[0]
            to_absorb = cp.where(particles.collision_model[particles.part_type[collided]] == 0)[0]
            particles.remove(collided[to_emit])
            #particles.emit(collided[to_emit])
        
        particles.update_R(dt)
        particles.update_axis(dt)
        #coll_step(particles, grid, sim_state.boundaries)
        grid.update_density(particles)
        #grid.phi[...] = solver.solve(grid.phi, -grid.rho*Consts.eps0_1, tol=1e-5, max_iter=5, max_cycles=100)
        #grid.phi[:], info = solver.solve(grid.phi, -grid.rho*Consts.eps0_1, tol=1e-5, smooth_iter=5, solve_iter=50, max_cycles=100)
        solver.solve(grid)
        grid.update_E()
        particles.update_V(grid, dt)
        MCC.null_collision_method(particles, 6e23, sim_state.cross_sections, dt, sim_state.max_particles, sim_state.ionization)