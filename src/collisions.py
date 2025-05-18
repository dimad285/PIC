import cupy as cp
import cupyx
from src import Particles
from src import Grid
from enum import Enum, auto
import time

class WallInteractionType(Enum):
    """
    Enumeration of possible wall interaction types.
    """
    REFLECTION = auto()  # Total elastic reflection (current implementation)
    PARTICLE_DEATH = auto()  # Particle is removed from simulation
    ION_ELECTRON_EMISSION = auto()  # Generates secondary electrons
    NON_ELASTIC_REFLECTION = auto()  # Partial energy loss/angular redistribution
    ABSORPTION = auto()  # Particle is completely absorbed by the wall
    SECONDARY_ELECTRON_EMISSION = auto()  # Electron emission from wall impact
    SPUTTERING = auto()  # Material removal from wall surface
    IMPLANTATION = auto()  # Particle embeds into wall material
    STRUCTURAL_DAMAGE = auto()  # Causes damage to wall structure

collision_kernel = cp.RawKernel(r'''
extern "C" __global__
void handle_wall_collisions(
    const double* R_old, const double* R, double* V,
    const int* wall_cells, const int* wall_dirs,
    const int num_walls, const int num_particles,
    const double dx, const double dy, const int m) 
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_particles) return;

    double x_old = R_old[2 * idx];
    double y_old = R_old[2 * idx + 1];
    double x_new = R[2 * idx];
    double y_new = R[2 * idx + 1];

    // Loop through walls to check for collision
    for (int i = 0; i < num_walls; i++) {
        int cell_idx = wall_cells[i];
        int direction = wall_dirs[i];

        // Compute wall position
        double wall_x = (cell_idx % m) * dx;
        double wall_y = (cell_idx / m) * dy;

        if (direction == 0) {  // RIGHT_WALL
            wall_x += dx;
            if (x_old <= wall_x && x_new > wall_x) {
                V[2 * idx] *= -1;  // Reflect x velocity
            }
        }
        else if (direction == 1) {  // LEFT_WALL
            if (x_old >= wall_x && x_new < wall_x) {
                V[2 * idx] *= -1;  // Reflect x velocity
            }
        }
        else if (direction == 2) {  // TOP_WALL
            wall_y += dy;
            if (y_old <= wall_y && y_new > wall_y) {
                V[2 * idx + 1] *= -1;  // Reflect y velocity
            }
        }
        else if (direction == 3) {  // BOTTOM_WALL
            if (y_old >= wall_y && y_new < wall_y) {
                V[2 * idx + 1] *= -1;  // Reflect y velocity
            }
        }
    }
}
''', 'handle_wall_collisions')

trace_kernel = cp.RawKernel(
    """
    extern "C" __global__ void trace_kernel(
        const double* pos0, const double* pos1,
        int* traced_indices, int Nx, int Ny, double dx, double dy, int max_steps, int num_particles
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;
       
        // Load particle start and end positions - now accessing from transposed arrays (n, 2)
        double x_start = pos0[idx * 2];      // x-coordinate at pos0[particle_idx, 0]
        double y_start = pos0[idx * 2 + 1];  // y-coordinate at pos0[particle_idx, 1]
        double x_end = pos1[idx * 2];        // x-coordinate at pos1[particle_idx, 0]
        double y_end = pos1[idx * 2 + 1];    // y-coordinate at pos1[particle_idx, 1]
       
        // Super-sampling factor (more accurate tracing)
        const int SUPER_SAMPLE = 10;
       
        // Safety epsilon to avoid floating point rounding issues
        const double EPSILON = 1e-5f;
       
        // Convert to grid coordinates
        double gx_start = x_start / dx;
        double gy_start = y_start / dy;
        double gx_end = x_end / dx;
        double gy_end = y_end / dy;
       
        // Initial cell
        int i_start = (int)floorf(gx_start);
        int j_start = (int)floorf(gy_start);
        int i_end = (int)floorf(gx_end);
        int j_end = (int)floorf(gy_end);
       
        // Store start cell
        traced_indices[idx * max_steps] = j_start * Nx + i_start;
       
        // If start and end are in same cell, we're done
        if (i_start == i_end && j_start == j_end) {
            return;
        }
       
        // Calculate line parameters
        double delta_x = gx_end - gx_start;
        double delta_y = gy_end - gy_start;
        double line_length = sqrtf(delta_x*delta_x + delta_y*delta_y);
       
        // Calculate steps needed (with super-sampling for accuracy)
        int steps = (int)(line_length * SUPER_SAMPLE) + 1;
       
        // Initialize cell tracking
        int step_count = 1;  // We already stored initial position
        int prev_i = i_start;
        int prev_j = j_start;
        int last_added_cell = j_start * Nx + i_start;
       
        // Check if we have enough space in the buffer
        if (steps > max_steps * 10) {
            steps = max_steps * 10;  // Cap to avoid infinite loops
        }
       
        // Trace the path
        for (int s = 1; s <= steps && step_count < max_steps; s++) {
            // Calculate position at this step (with super-sampling)
            double t = (double)s / steps;
            double gx = gx_start + t * delta_x;
            double gy = gy_start + t * delta_y;
           
            // Get current cell with epsilon adjustment to handle boundary cases
            int i = (int)floorf(gx + EPSILON);
            int j = (int)floorf(gy + EPSILON);
           
            // Check boundaries
            if (i < 0 || i >= Nx || j < 0 || j >= Ny) {
                continue;  // Skip this step but continue tracing
            }
           
            // Check if we've moved to a new cell
            int curr_cell = j * Nx + i;
            if ((i != prev_i || j != prev_j) && curr_cell != last_added_cell) {
                traced_indices[idx * max_steps + step_count] = curr_cell;
                last_added_cell = curr_cell;
                step_count++;
               
                // Check if we've run out of storage space
                if (step_count >= max_steps - 1) {
                    break;
                }
            }
           
            prev_i = i;
            prev_j = j;
        }
       
        // Ensure end cell is included
        int end_cell = j_end * Nx + i_end;
        if (end_cell != last_added_cell && step_count < max_steps) {
            traced_indices[idx * max_steps + step_count] = end_cell;
            step_count++;
        }
    }
    """,
    "trace_kernel"
)

def remove_out_of_bounds(particles:Particles.Particles2D, X_max, Y_max):
    """
    Remove particles that are out of bounds by swapping with the last active particle
    and decreasing the active particle counter.

    Parameters:
    - particles: object containing particle data (positions, velocities, etc.).
      Assumes positions are stored as `particles.pos` with shape (2, n).
    - X_max, Y_max: domain size in x and y directions.

    Returns:
    - Updated particle count.
    """
    pos = particles.R  # Shape (2, n)
    count = particles.last_alive  # Last active particle index
    
    # Find out-of-bounds particles
    out_of_bounds = (pos[0, :count] < 0) | (pos[0, :count] > X_max) | \
                    (pos[1, :count] < 0) | (pos[1, :count] > Y_max)

    indices_to_remove = cp.where(out_of_bounds)[0]
    num_to_remove = indices_to_remove.size

    if num_to_remove > 0:
        # Get indices of last N active particles
        swap_indices = cp.arange(count - num_to_remove, count)

        # Ensure we don’t swap the same indices (if num_to_remove == count, it breaks)
        valid_swaps = swap_indices[swap_indices >= 0]

        # Swap positions
        pos[:, indices_to_remove[:valid_swaps.size]] = pos[:, valid_swaps]
        particles.V[:, indices_to_remove[:valid_swaps.size]] = particles.V[:, valid_swaps]
        particles.part_type[indices_to_remove[:valid_swaps.size]] = particles.part_type[valid_swaps]

        # Decrease count
        count -= num_to_remove

    particles.last_alive = count  # Update active particle count



def handle_wall_collisions(
    particles:Particles.Particles2D,  # Particles2D object
    grid:Grid.Grid2D,  # Grid object
    walls  # Wall information (wall cells and directions)
):
    """
    Handles wall collisions with simplified particle death and ion-electron emission.
   
    :param particles: Particles2D system
    :param grid: Simulation grid
    :param walls: Wall information (wall cells and directions)
    """

    epsilon = 0  # Small value to avoid numerical issues
    
    # Unpack wall information
    wall_cells, wall_dirs = walls
    dx, dy, m = grid.dx, grid.dy, grid.m-1

    # Compute wall positions
    wall_x = (wall_cells % m) * dx
    wall_y = (wall_cells // m) * dy
   
    # Direction masks
    right_wall_mask = wall_dirs == 0b0001
    left_wall_mask = wall_dirs == 0b0010
    top_wall_mask = wall_dirs == 0b0100
    bottom_wall_mask = wall_dirs == 0b1000
    
    # Adjust wall positions
    wall_x += left_wall_mask * dx
    wall_y += top_wall_mask * dy
   
    # Reshape for broadcasting
    wall_x = wall_x[:, None]
    wall_y = wall_y[:, None]
   
    # Use existing particle position tracking
    x_old, y_old = particles.R_old[:, :particles.last_alive]
    x_new, y_new = particles.R[:, :particles.last_alive]
   
    # Collision detection
    # Define wall bounds
    x_min = wall_x - dx / 2
    x_max = wall_x + dx / 2
    y_min = wall_y - dy / 2
    y_max = wall_y + dy / 2

    start_time = time.perf_counter()

    # Collision detection with range constraints
    hit_right = (
        (x_old <= wall_x - epsilon) & (x_new > wall_x + epsilon)  # X condition
        & (y_old >= y_min) & (y_old <= y_max)  # Y within range
        & right_wall_mask[:, None]
    )

    hit_left = (
        (x_old >= wall_x + epsilon) & (x_new < wall_x - epsilon)
        & (y_old >= y_min) & (y_old <= y_max)
        & left_wall_mask[:, None]
    )

    hit_top = (
        (y_old <= wall_y - epsilon) & (y_new > wall_y + epsilon)
        & (x_old >= x_min) & (x_old <= x_max)
        & top_wall_mask[:, None]
    )

    hit_bottom = (
        (y_old >= wall_y + epsilon) & (y_new < wall_y - epsilon)
        & (x_old >= x_min) & (x_old <= x_max)
        & bottom_wall_mask[:, None]
    )
    
    mask_time = time.perf_counter() - start_time
    # Combine all hit conditions
    all_hits = hit_right | hit_left | hit_top | hit_bottom
    hit_particles = cp.nonzero(cp.any(all_hits, axis=0))[0]

    detection_time = time.perf_counter() - start_time - mask_time
    print(f"Mask time: {mask_time:.2e}, Detection time: {detection_time:.2e}")
    
    if hit_particles.size == 0:
        return
    
    # Get types of hit particles
    hit_particle_types = particles.part_type[hit_particles]
    interaction_types = particles.collision_model[hit_particle_types]
   
    # Separate particles to remove and emit
    to_remove = hit_particles[interaction_types == 0]
    to_emit = hit_particles[interaction_types == 1]
    
    # IMPORTANT: Reset positions of ALL hit particles to prevent wall penetration
    particles.R[:, hit_particles] = particles.R_old[:, hit_particles]
    
    # For emitted particles, transform to electrons
    if to_emit.size > 0:
        particles.V[:, to_emit] = 0  # add electron emission energy here if needed
        particles.part_type[to_emit] = particles.part_name.index('electrons')
        
    # Remove particles with collision model 0 (only after handling all other logic)
    if to_remove.size > 0:
        particles.remove(to_remove)


def generate_secondary_electron_velocity(incident_velocity):
    """
    Generate secondary electron velocity based on incident particle.
    
    :param incident_velocity: Velocity of incident particle
    :return: Generated secondary electron velocity
    """
    # Simple velocity scaling for secondary electron generation
    scaling_factor = 0.1
    return incident_velocity * scaling_factor


def profile_wall_collisions(particles, grid, walls):
    """
    Profiling wrapper for handle_wall_collisions.
    """
    cp.cuda.Device(0).synchronize()  # Ensure synchronization before timing
    time = cupyx.time.repeat(handle_wall_collisions, (particles, grid, walls), n_repeat=10)
    print(time)


def trace_particle_paths(particles:Particles.Particles2D, grid:Grid.Grid2D, max_steps):
    Nx, Ny = grid.gridshape[0] - 1, grid.gridshape[1] - 1
    num_particles = particles.last_alive
   
    # Preallocate array for storing traced indices (max_steps per particle)
    traced_indices = cp.full((num_particles, max_steps), -1, dtype=cp.int32)
   
    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block
    trace_kernel((blocks_per_grid,), (threads_per_block,), (
        particles.R_old, 
        particles.R, 
        traced_indices, 
        cp.int32(Nx), 
        cp.int32(Ny), 
        cp.float64(grid.dx),
        cp.float64(grid.dy),
        cp.int32(max_steps), 
        cp.int32(num_particles)
    ))
    
    return traced_indices


'''
# Function to detect collisions with walls
def detect_collisions(traced_indices, wall_cells):
    # Precompute wall mask
    wall_mask = cp.isin(traced_indices, wall_cells)

    # Valid steps: both current and next index are not -1
    valid = (traced_indices[:, :-1] != -1) & (traced_indices[:, 1:] != -1)

    # Wall collisions: both current and next step are wall cells
    wall_collisions = wall_mask[:, :-1] & wall_mask[:, 1:]

    # Combined collision mask
    collision_mask = valid & wall_collisions

    # Return indices of particles that collided
    collided_indices = cp.nonzero(cp.any(collision_mask, axis=1))[0]

    return collided_indices
'''



kernel_code = r'''
extern "C" __global__
void detect_collisions_kernel(
    const int* traced_indices,
    const bool* wall_lookup,
    int* collided_indices,
    int* collision_counter,
    int num_particles,
    int max_steps)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_particles) return;

    for (int i = 0; i < max_steps - 1; ++i) {
        int a = traced_indices[idx * max_steps + i];
        int b = traced_indices[idx * max_steps + i + 1];

        if (a == -1 || b == -1) continue;

        if (wall_lookup[a] && wall_lookup[b]) {
            // Atomically write this particle index into the output buffer
            int write_idx = atomicAdd(collision_counter, 1);
            collided_indices[write_idx] = idx;
            return;  // Only record once per particle
        }
    }
}
'''

detect_collisions_kernel = cp.RawKernel(kernel_code, 'detect_collisions_kernel')


def detect_collisions(particles:Particles.Particles2D, traced_indices, wall_lookup):
    num_particles, max_steps = traced_indices.shape

    # Create a wall lookup table (bitmask)

    # Flatten traced_indices for raw access
    flat_traced = traced_indices.ravel()

    # Prepare output buffers
    max_collisions = num_particles  # worst case: all collide
    collided_indices = cp.full(max_collisions, -1, dtype=cp.int32)

    # Launch kernel
    threads_per_block = 256
    blocks = (num_particles + threads_per_block - 1) // threads_per_block
    detect_collisions_kernel(
        (blocks,), (threads_per_block,),
        (
            flat_traced,
            wall_lookup,
            collided_indices,
            particles.collision_counter,
            num_particles,
            max_steps
        )
    )

    # Slice valid results
    num_collided = particles.collision_counter.item()
    particles.collision_counter.fill(0)  # Reset counter
    return collided_indices[:num_collided]



'''
Do a check for gridless trjectorie intersection of two line segments.

https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
'''