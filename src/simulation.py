import cupy as cp
import cupyx.profiler
import numpy as np
import collisions
import Solvers
import Render
import Particles
import Grid
import MCC
import time
import Boundaries
import Interface
import tkinter as tk
import cupyx
import Multigrid
import Consts

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
        energy_out = cp.zeros(1, dtype=cp.float32)
        
        # Calculate threads and blocks
        threads_per_block = 256
        blocks_per_grid = (particles.last_alive + threads_per_block - 1) // threads_per_block
        
        # Extract velocity components for kernel
        Vx = particles.V[0, :].astype(cp.float32)
        Vy = particles.V[1, :].astype(cp.float32)
        
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

class SimulationState:
    def __init__(self):
        self.t = 0
        self.sim_time = 0
        self.frame_time = 0
        self.step = 0
        self.framecounter = 0
        self.running = False
        self.gui = None
        self.renderer = None
        self.boundaries = None
        self.walls = None
        self.bound_tuple = None
        self.diagnostics = Diagnostics()
        self.cross_sections = None
        self.particles = None
        self.grid = None
        self.solver = None


def init_simulation(m, n, X, Y, N, dt, species):
    pass


def init_step(particles:Particles.Particles2D, grid:Grid.Grid2D, solver:Solvers.Solver, dt:float, boundaries=None):

    print('Initial Step')

    def bench_solve(solver, grid):
        grid.phi.fill(0)
        print(cupyx.profiler.benchmark(solver.solve, (grid,)))

    grid.update_density(particles)
    if boundaries is not None:
        grid.phi[boundaries.conditions[0]] = boundaries.conditions[1]
    #grid.phi[...] = solver.solve(grid.phi, -grid.rho*Consts.eps0_1, tol=1e-5, max_iter=5, max_cycles=100)
    solver.solve(grid)
    grid.update_E()
    particles.update_V(grid, -dt/2)



def step(particles:Particles.Particles2D, grid:Grid.Grid2D, dt, solver:Solvers.Solver, cross_sections, MAX_PARICLES, boundaries=None, IONIZATION=True):

    def coll_step(particles, grid, boundaries):
        trace = collisions.trace_particle_paths(particles, grid, 10)
        collided = collisions.detect_collisions(particles,trace, boundaries.wall_lookup,)
        particles.R[collided] = particles.R_old[collided]
        to_emit = cp.where(particles.collision_model[particles.part_type[collided]] == 1)[0]
        to_absorb = cp.where(particles.collision_model[particles.part_type[collided]] == 0)[0]
        particles.remove(collided[to_emit])
        #particles.emit(collided[to_emit])
    
    particles.update_R(dt)
    particles.update_axis(dt)
    coll_step(particles, grid, boundaries)
    grid.update_density(particles)
    #grid.phi[...] = solver.solve(grid.phi, -grid.rho*Consts.eps0_1, tol=1e-5, max_iter=5, max_cycles=100)
    solver.solve(grid)
    grid.update_E()
    particles.update_V(grid, dt)
    #MCC.null_collision_method(particles, 1e18, cross_sections, dt, MAX_PARICLES, IONIZATION)
    



def draw(renderer:Render.PICRenderer, state, particles:Particles.Particles2D, grid:Grid.Grid2D,
        dt, sim_state:SimulationState,
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
    
    n, m = grid.gridshape
    X, Y = grid.domain

    match selected_render_type:
        case "particles":
            renderer.renderer_type = "particles"
            match plot_var_name:
                case "R":
                    renderer.update_particles(particles, 0, X, 0, Y)
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
                    surf_max = cp.max(grid.phi)
                    surf_min = cp.min(grid.phi)
                    if surf_max == 0:
                        surf_max_1 = cp.abs(1/surf_min)
                    else:
                        surf_max_1 = cp.abs(1/surf_max)
                    z = cp.asnumpy(cp.reshape(grid.phi*surf_scale*surf_max_1, (m, n)))
                    renderer.update_surface(x, y, z)
                    renderer.update_legend('surface_max', f"{plot_var_name}_max: {surf_max:.2e} V", 290)
                    renderer.update_legend('surface_min', f"{plot_var_name}_min: {surf_min:.2e} V", 320)
                    renderer.label_list.extend(['surface_max', 'surface_min'])

                case "rho":
                    surf_max = cp.max(grid.rho)
                    surf_min = cp.min(grid.rho)
                    if surf_max == 0:
                        surf_max_1 = cp.abs(1/surf_min)
                    else:
                        surf_max_1 = cp.abs(1/surf_max)
                    z = cp.asnumpy(cp.reshape(grid.rho*surf_scale*surf_max_1, (m, n)))
                    renderer.update_surface(x, y, z)
                    renderer.update_legend('surface_max', f"{plot_var_name}_max: {surf_max:.2e} C/m^3", 290)
                    renderer.update_legend('surface_min', f"{plot_var_name}_min: {surf_min:.2e} C/m^3", 320)
                    renderer.label_list.extend(['surface_max', 'surface_min'])
            


    
    if state['text_enabled']:
        renderer.update_legend('sim', f"Sim time: {(sim_state.sim_time)*1e6:.1f} mks", 80)
        renderer.update_legend('frame', f"Frame time: {(sim_state.frame_time)*1e6:.1f} mks", 110)
        renderer.update_legend('n', f"N: {particles.last_alive}", 140)
        renderer.update_legend('dt', f"dt: {dt:.2e}", 200)
        renderer.update_legend('flytime', f"min flytime {cp.min(cp.array([cp.float32(grid.dx) / cp.max(particles.V[0]), cp.float32(grid.dy) / cp.max(particles.V[1])])):.2e}", 230)
        renderer.update_legend('t', f"t: {sim_state.t:.2e}", 260)
        renderer.label_list.extend(['sim', 'frame', 'n', 'dt', 'flytime', 't'])

    renderer.render(clear = not state['trace_enabled'], TEXT_RENDERING=state['text_enabled'], BOUNDARY_RENDERING = True)
    #renderer.adjust_particle_count(time.perf_counter() - start_draw)



class SimulationConfig:
    def __init__(self, config_dict):
        # Grid parameters
        self.m = int(config_dict['grid']['J'])  # Number of cells in x/z direction
        self.n = int(config_dict['grid']['K'])  # Number of cells in y/r direction
        self.X = float(config_dict['grid'].get('x1f', 1.0)) - float(config_dict['grid'].get('x1s', 0.0))  # Domain size in x
        self.Y = float(config_dict['grid'].get('x2f', 1.0)) - float(config_dict['grid'].get('x2s', 0.0))  # Domain size in y
        self.cylindrical = bool(int(config_dict['grid']['Cylindrical']))  # 0 for Cartesian, 1 for Cylindrical
        self.ionization = bool(int(config_dict['system'].get('IONIZATION', 1)))

        # Time parameters
        self.dt = float(config_dict['control']['dt'])
        
        # Species parameters
        self.species = config_dict['species']
        
        # Control parameters
        self.control = config_dict['control']
        
        # Solver type
        self.solver_type = config_dict['control'].get('solver', 'gmres')
        
        # Load blocks
        self.loads = config_dict.get('loads', [])  # Add this line
        
        # Visualization parameters
        self.RENDER = config_dict['system'].get('RENDER', True)
        self.RENDER_FRAME = config_dict['system'].get('RENDER_FRAME', 1)
        self.UI = config_dict['system'].get('UI', True)
        self.save_fields = config_dict['system'].get('SAVE_FIELDS', False)
        self.diagnostics = config_dict['system'].get('DIAGNOSTICS', True)
        
        self.SCREEN_SIZE = (
            config_dict['system'].get('SCREEN_WIDTH', 1080),
            config_dict['system'].get('SCREEN_HEIGHT', 1080)
        )
        
        # Boundaries
        self.boundaries = config_dict.get('boundaries', [])
        
        # Maximum particles
        self.max_particles = int(config_dict['system'].get('MAX_PARTICLES', 100000))


class SimulationManager:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = SimulationState()
        self.initialize_components()
    
    def initialize_components(self):
        # Create particles instance
        self.state.particles = Particles.Particles2D(
            self.config.max_particles, 
            cylindrical=self.config.cylindrical
        )
        
        # Add species from config
        for species_data in self.config.species:
            if all(k in species_data for k in ['name', 'm', 'q', 'collisionModel']):
                self.state.particles.add_species(
                    species_data['name'],
                    species_data['m'],
                    species_data['q'],
                    species_data['collisionModel']
                )
        
        # Initialize particles from Load blocks
        if self.config.loads:  # Changed from hasattr check to direct list check
            for load_data in self.config.loads:
                if all(k in load_data for k in ['speciesName', 'x1MinMKS', 'x1MaxMKS', 'x2MinMKS', 'x2MaxMKS', 'n']):
                    x1 = load_data['x1MinMKS'] 
                    y1 = load_data['x2MinMKS']
                    x2 = load_data['x1MaxMKS']
                    y2 = load_data['x2MaxMKS']
                                       
                    self.state.particles.uniform_species_load(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        n=int(load_data['n']),
                        species=load_data['speciesName']
                    )
            self.state.particles.set_mcc_mask()

        
        # Create grid instance
        self.state.grid = Grid.Grid2D(
            self.config.m, 
            self.config.n, 
            self.config.X, 
            self.config.Y, 
            cylindrical=self.config.cylindrical
        )
        self.setup_solver()
        self.setup_visualization()
        init_step(
            self.state.particles, 
            self.state.grid, 
            self.state.solver,
            self.config.dt
        )


        cross_section_elastic = MCC.read_cross_section('csf/Ar.txt')
        cross_section_ion = MCC.read_cross_section('csf/Ar+.txt')
        self.state.cross_sections = [cross_section_elastic, cross_section_ion]
    
    def setup_solver(self):
        
        if self.config.boundaries:
            
            boundaries = Boundaries.Boundaries(self.config.boundaries, self.state.grid)
            self.state.boundaries = boundaries
            self.state.bound_tuple = boundaries.bound_tuple
            self.state.walls = boundaries.walls
            if self.config.solver_type == 'multigrid':
                self.state.solver = Multigrid.MultigridSolver(self.config.m, self.config.n, self.state.grid.dx, 
                                                              levels=6, omega=2/3)
            else:
                self.state.solver = Solvers.Solver(
                    self.config.solver_type, 
                    self.state.grid, 
                    cylindrical=self.config.cylindrical, 
                    boundaries=boundaries.conditions, 
                    tol=1e-5
                )
        else:
            self.state.solver = Solvers.Solver(
                self.config.solver_type, 
                self.state.grid, 
                cylindrical=self.config.cylindrical, 
                tol=1e-6
            )
            self.state.bound_tuple = ()
            self.state.walls = None

    def setup_visualization(self):
        if self.config.UI:
            root = tk.Tk()
            self.state.gui = Interface.SimulationUI_tk(root)
        if self.config.RENDER:
            self.state.renderer = Render.PICRenderer(
                *self.config.SCREEN_SIZE, 
                "fonts\\Arial.ttf", 
                self.config.max_particles, 
                renderer_type='surface_plot', 
                is_3d=False
            )
            #window = surf.initialize_window()
        
    
    def simulation_step(self):
        # Check state.gui instead of gui
        if not (self.state.running or self.state.gui.state["simulation_step"]):
            return
            
        start_time = time.perf_counter()
        step(
            self.state.particles,  # Use state.particles
            self.state.grid,       # Use state.grid
            self.config.dt, 
            self.state.solver,     # Use state.solver
            self.state.cross_sections,
            self.config.max_particles,
            self.state.boundaries,
            self.config.ionization
        )
        
        if self.config.diagnostics:
            self.state.diagnostics.update(
                self.state.t, 
                self.state.particles, 
                self.state.grid, 
                self.state.sim_time
            )
        
        self.state.t += self.config.dt
        self.state.sim_time = time.perf_counter() - start_time

        self.state.gui.state["simulation_step"] = False
        self.state.step += 1
    
    def should_exit(self) -> bool:
        """Check if the simulation should exit."""
        if self.config.RENDER and self.state.renderer:
            return self.state.renderer.should_close()
        
        if self.config.UI and self.state.gui:
            return self.state.gui.closed
            
        # Add any other exit conditions here
        return False
    
    def run(self):
        while True:
            if self.should_exit():
                break
                
            self.update_ui()
            self.simulation_step()
            self.handle_visualization()
            
        self.cleanup()
    
    def update_ui(self):
        if self.config.UI:
            self.state.gui.update()
            self.state.running = self.state.gui.state["simulation_running"]
            

    
    def handle_visualization(self):
        if not self.config.RENDER and self.state.framecounter == self.config.RENDER_FRAME:
            self.state.framecounter = 0
            print(f"step = {self.state.step}, t = {self.state.t:.2e}, sim_time = {self.state.sim_time:.2e}, N = {self.state.particles.last_alive}")

        self.state.framecounter += 1
        if self.config.RENDER and self.state.framecounter == self.config.RENDER_FRAME:      

            start_draw = time.perf_counter()  
            draw(
                self.state.renderer, 
                self.state.gui.get_state(), 
                self.state.particles, 
                self.state.grid, 
                self.config.dt,
                self.state,
                self.state.diagnostics, 
                self.config.SCREEN_SIZE, 
                self.state.bound_tuple
            )
            self.state.frame_time = time.perf_counter() - start_draw

            if self.state.renderer.should_close():
                self.state.running = False
            self.state.framecounter = 0
            
            self.state.renderer.set_title(f"OOPIC Pro - Step {self.state.step}, t = {self.state.t:.2e}")

    def cleanup(self):
        try:
            # First save fields if needed
            if self.config.save_fields:
                if self.config.cylindrical:
                    self.state.grid.save_to_txt(
                        'fields/fields_cylindrical.txt', 
                        fields={'rho': self.state.grid.rho, 'phi': self.state.grid.phi, 'Ez': self.state.grid.E[0], 'Er': self.state.grid.E[1]},  # Add fields here
                        header='z r rho phi Ez Er'
                    )
                else:
                    self.state.grid.save_to_txt(
                        'fields/fields_cartesian.txt', 
                        fields={'rho': self.state.grid.rho, 'phi': self.state.grid.phi, 'Ex': self.state.grid.E[0], 'Ey': self.state.grid.E[1]},  # Add fields here
                        header='x y rho phi Ex Ey'
                    )
            
            # Then cleanup renderer while OpenGL context is still valid
            if self.config.RENDER and hasattr(self.state, 'renderer') and self.state.renderer is not None:
                self.state.renderer.close()
                self.state.renderer = None
                print('Renderer closed')
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
