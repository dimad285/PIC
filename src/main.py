import time
import Render 
import Solvers 
import simulation
from Parser import parse_config
import Interface
import tkinter as tk
import cProfile
import pstats
import io
import Boundaries
import Particles
import Grid
import MCC
from ConfigParser import ConfigParser



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
        self.camera = None
        self.boundaries = None
        self.walls = None
        self.bound_tuple = None
        self.diagnostics = simulation.Diagnostics()
        self.cross_sections = None
        self.particles = None
        self.grid = None
        self.solver = None


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
        simulation.init_step(
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
            self.state.bound_tuple = boundaries.bound_tuple
            self.state.walls = boundaries.walls
            print(self.state.walls[0])
            self.state.solver = Solvers.Solver(
                self.config.solver_type, 
                self.state.grid, 
                cylindrical=self.config.cylindrical, 
                boundaries=boundaries.conditions, 
                tol=1e-6
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
            self.state.camera = Render.Camera()
            #window = surf.initialize_window()
        
    
    def simulation_step(self):
        # Check state.gui instead of gui
        if not (self.state.running or self.state.gui.state["simulation_step"]):
            return
            
        start_time = time.perf_counter()
        
        simulation.step(
            self.state.particles,  # Use state.particles
            self.state.grid,       # Use state.grid
            self.config.dt, 
            self.state.solver,     # Use state.solver
            self.state.cross_sections,
            self.config.max_particles,
            self.state.walls,
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
    
    def should_exit(self) -> bool:
        """Check if the simulation should exit."""
        if self.config.RENDER and self.state.renderer:
            return self.state.renderer.should_close()
        
        if self.config.UI and self.state.gui:
            return not self.state.gui.ui_running
            
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
        if not self.config.RENDER and self.state.framecounter == 1:
            self.state.framecounter = 0
            print(f"t = {self.state.t:.2e}, sim_time = {self.state.sim_time:.2e}")

        self.state.framecounter += 1
        if self.config.RENDER and self.state.framecounter == self.config.RENDER_FRAME:        
            simulation.draw(
                self.state.renderer, 
                self.state.gui.get_state(), 
                self.state.particles, 
                self.state.grid, 
                self.state.camera,
                self.state.frame_time, 
                self.state.sim_time, 
                self.config.dt,
                self.state.diagnostics, 
                self.config.SCREEN_SIZE, 
                self.state.bound_tuple
            )
            self.state.frame_time = time.perf_counter() - self.state.sim_time
            if self.state.renderer.should_close():
                self.state.running = False
            self.state.framecounter = 0
            self.state.step += 1

    def cleanup(self):
        try:
            # First save fields if needed
            if self.config.save_fields:
                if self.config.cylindrical:
                    self.state.grid.save_to_txt(
                        'fields/fields_cylindrical.txt', 
                        fields={'rho': self.state.grid.rho, 'phi': self.state.grid.phi}, 
                        header='z r rho phi'
                    )
                else:
                    self.state.grid.save_to_txt(
                        'fields/fields_cartesian.txt', 
                        fields={'rho': self.state.grid.rho, 'phi': self.state.grid.phi}, 
                        header='x y rho phi'
                    )
            
            # Then cleanup renderer while OpenGL context is still valid
            if self.config.RENDER and hasattr(self.state, 'renderer') and self.state.renderer is not None:
                self.state.renderer.close()
                self.state.renderer = None
                print('Renderer closed')
                
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    parser = ConfigParser()
    config_dict = parser.parse_file('test.inp')
    config = SimulationConfig(config_dict)
    manager = SimulationManager(config)
    
    # Start profiling before running simulation
    pr = cProfile.Profile()
    pr.enable()
    
    # Run simulation
    manager.run()
    
    # Stop profiling and save results
    pr.disable()
    pr.dump_stats("profile.prof")
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open("profile.txt", "w") as f:
        f.write(s.getvalue())


'''
to do:
- add multigrid solver
- check if Laplacian is positive definite and symmetric
'''
