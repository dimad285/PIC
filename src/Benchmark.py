import cProfile
import pstats
import line_profiler
import numpy as np
import cupy as cp
from functools import wraps
import collisions

def profile_cprofile(func):
    """Decorator for cProfile profiling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            return profile.runcall(func, *args, **kwargs)
        finally:
            stats = pstats.Stats(profile)
            stats.sort_stats('cumulative')
            stats.print_stats(30)  # Print top 30 functions
    return wrapper

# Create sample data for profiling
def create_test_data(n_particles=10000, grid_shape=(16, 16)):
    Nx, Ny = grid_shape
    
    # Create random positions and velocities
    positions = np.random.rand(2, n_particles)
    velocities = np.random.randn(2, n_particles) * 0.1
    
    # Create sorted indices (simplified for testing)
    sorted_indices = np.arange(n_particles)
    
    # Create cell data
    n_cells = Nx * Ny
    active_cells = np.arange(n_cells)
    
    # Create cell shifts (simplified)
    cell_shifts = np.arange(0, n_particles + 1, n_particles // n_cells)
    if len(cell_shifts) < n_cells + 1:
        cell_shifts = np.append(cell_shifts, n_particles)
    
    # Create wall data (random walls for testing)
    cell_indices = np.arange(n_cells)
    wall_directions = np.random.randint(0, 16, size=n_cells, dtype=np.int32)
    
    # Cell size for unit domain
    cell_size = (1.0/Nx, 1.0/Ny)
    
    return (cp.array(positions), cp.array(velocities), cp.array(sorted_indices),
            cp.array(active_cells), cp.array(cell_shifts), cp.array(cell_indices),
            cp.array(wall_directions), grid_shape, cell_size)

# Decorated function for profiling
@profile_cprofile
def detect_collisions_profiled(positions, velocities, sorted_indices, active_cells, 
                             cell_shifts, cell_indices, wall_directions,
                             grid_shape, cell_size):
    """Profiled version of detect_collisions"""
    collisions.detect_collisions(positions, velocities, sorted_indices, active_cells,
                     cell_shifts, cell_indices, wall_directions,
                     grid_shape, cell_size)

# Line profiler setup
profile = line_profiler.LineProfiler()
detect_collisions_line = profile(collisions.detect_collisions)

def run_profiling(n_particles=10000, n_iterations=1):
    """Run both cProfile and line_profiler"""
    print("Preparing test data...")
    test_data = create_test_data(n_particles)
    
    print("\nRunning cProfile analysis...")
    for _ in range(n_iterations):
        detect_collisions_profiled(*test_data)
    
    print("\nRunning line-by-line analysis...")
    for _ in range(n_iterations):
        detect_collisions_line(*test_data)
    
    print("\nLine-by-line profiling results:")
    profile.print_stats()

if __name__ == "__main__":
    run_profiling()