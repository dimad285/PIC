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
import multigrid


def run_gpu(m = 16, n = 16, k = 0, X = 1, Y = 1, Z = 1, max_particles = 1000, dt = 0.001, grid_type='2d',
            boundary = None, RENDER = True, UI = True, RENDER_FRAME = 1, 
            SCREEN_SIZE = (512, 512), solver_type = 'inverse'):
    
    print('Starting...')
    fontfile = "fonts\\Arial.ttf"
    framecounter = 0
    t = 0
    sim_time = 0
    frame_time = 0
    step = 0

    # INIT
    print('creating arrays...')
    particles = Particles.Particles2D(max_particles)
    grid = Grid.Grid2D(m, n, X, Y)
    diagnostics = simulation.Diagnostics()

    if boundary != None:
        boundaries = Boundaries.Boundaries(boundary, grid)
        bound_tuple = boundaries.bound_tuple
        walls = boundaries.walls
        #solver = Solvers.Solver(solver_type, grid, boundaries.conditions)
        solver = Solvers.Solver(solver_type, grid, boundaries.conditions, 1e-5)
    else:
        solver = Solvers.Solver(solver_type, grid, tol=1e-5)
        bound_tuple = ()
        walls = None

    if UI:
        root = tk.Tk()
        gui = Interface.SimulationUI_tk(root)
    if RENDER:
        renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, max_particles, renderer_type='surface_plot', is_3d=False)
        camera = Render.Camera()
        #window = surf.initialize_window()
        
    cross_section_elastic = MCC.read_cross_section('csf/Ar.txt')
    cross_section_ion = MCC.read_cross_section('csf/Ar+.txt')
    cross_sections = [cross_section_elastic, cross_section_ion]
    
    # MAIN LOOP
    el_k = 10000
    particles.uniform_species_load(X * 0.25, Y * 0.25, X/(m-1), Y/(n-1), el_k, 'electron')
    #particles.uniform_species_load(X * 0.25, Y * 0.25, X/(m-1), Y/(n-1), el_k, 'proton')
    particles.update_bilinear_weights(grid)
    particles.sort_particles_sparse(grid.cell_count)
    #print(particles.part_type[:particles.last_alive])

    print('running...')
    while True:

        if UI:
            root.update()
            state = gui.get_state()
            
        start_time = time.perf_counter()
        
        if state["simulation_running"] or state["simulation_step"]:
            # UPDATE
            simulation.step(particles, grid, dt, solver, cross_sections, max_particles, walls)
            diagnostics.update(t, particles, grid, sim_time)
            t += dt
            sim_time = time.perf_counter() - start_time

            if state["simulation_step"]:
                state["simulation_step"] = False


        if not RENDER and framecounter == 1:
            framecounter = 0
            print(f"t = {t:.2e}, sim_time = {sim_time:.2e}")

        # RENDER
        framecounter += 1
        if RENDER and framecounter == RENDER_FRAME:        
            
            simulation.draw(renderer, state, particles, grid, camera,
                            frame_time, sim_time, dt,
                            diagnostics, SCREEN_SIZE, bound_tuple)

            frame_time = time.perf_counter() - start_time

            
            if renderer.should_close():
                break
            framecounter = 0

            step += 1

    if RENDER:
        renderer.close()
        print('Renderer closed')

    return 0



if __name__ == "__main__":
    

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
    boundaries = config['boundarys']

    solver_type = 'gmres'

    pr = cProfile.Profile()
    pr.enable()
    
    run_gpu(m, n, k, X, Y, Z, max_particles, dt, grid_type='2d',
                            boundary=boundaries, 
                            RENDER=config['RENDER'], 
                            RENDER_FRAME=config['RENDER_FRAME'], 
                            solver_type=solver_type, 
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