import cupy as cp
import Particles
import Grid

def collision_map(boundaries, gridsize):
    """
    Precompute a collision map for a 2D grid with boundary conditions.

    Args:
        boundaries (list or array): Array of grid node indices representing boundaries.
        gridsize (tuple): Tuple (m, n) representing the grid dimensions.

    Returns:
        cupy.ndarray: Array of cell pairs where collisions occur.
    """
    m, n = gridsize
    total_cells = m * n
    coll_map = []

    # Iterate through each cell
    for i in range(total_cells):
        print(i)
        if i in boundaries:
            # Check right neighbor (i + 1)
            if (i + 1) % m != 0 and (i + 1) in boundaries:  # Avoid wrapping around rows
                cell_1 = i
                cell_2 = i - m
                coll_map.append([cell_1, cell_2])
            
            # Check bottom neighbor (i + m)
            if (i + m) < total_cells and (i + m) in boundaries:  # Stay within bounds
                cell_1 = i
                cell_2 = i - 1
                coll_map.append([cell_1, cell_2])

    return cp.array(coll_map)



def mark_cell_walls_sparse(wall_nodes, grid_shape):
    Nx, Ny = grid_shape  # Number of nodes in x and y directions
    cell_indices = []
    wall_directions = []

    for node_idx in wall_nodes:
        # Convert node index to 2D coordinates
        i, j = divmod(node_idx, Ny)

        # Vertical wall logic
        if j > 0:  # Left wall (affects the right wall of the left cell)
            cell_idx = i * (Ny - 1) + (j - 1)
            if cell_idx not in cell_indices:
                cell_indices.append(cell_idx)
                wall_directions.append(0)
            wall_directions[cell_indices.index(cell_idx)] |= 0b0010  # Right wall
        if j < Ny - 1:  # Right wall (affects the left wall of the right cell)
            cell_idx = i * (Ny - 1) + j
            if cell_idx not in cell_indices:
                cell_indices.append(cell_idx)
                wall_directions.append(0)
            wall_directions[cell_indices.index(cell_idx)] |= 0b0001  # Left wall

    # Convert lists to CuPy arrays
    cell_indices = cp.array(cell_indices, dtype=cp.int32)
    wall_directions = cp.array(wall_directions, dtype=cp.uint8)
    return cell_indices, wall_directions


def collision_map_new(boundaries, gridsize):
    """
    Precompute a collision map for a 2D grid with boundary conditions.
    
    Args:
        boundaries (list or array): Array of grid node indices representing boundaries.
        gridsize (tuple): Tuple (m, n) representing the grid dimensions.
    Returns:
        cupy.ndarray: Array of cell pairs where collisions occur.
    """
    m, n = gridsize
    total_cells = m * n
    
    # Convert boundaries to a set for O(1) lookup
    boundary_set = set(boundaries.tolist())
    coll_map = []
    
    # Only iterate through boundary cells
    for i in boundary_set:
        # Check right neighbor (i + 1)
        if (i + 1) % n != 0:  # Avoid wrapping around rows
            if (i + 1) in boundary_set:
                coll_map.append([i, i + m])
        
        # Check bottom neighbor (i + n)
        if (i + n) < total_cells:  # Stay within bounds
            if (i + n) in boundary_set:
                coll_map.append([i, i - 1])
    
    return cp.array(coll_map)


def check_collisions(indices: cp.ndarray, collision_pairs: cp.ndarray, last_alive: int):
    '''
    Posible solutions:

    1. In grid space, check all particles in adjacent cells (maybe with velocity direction) for a collision.

    2. Compute all possible grid cell pairs that will result in a collision. Then check them.

    '''

    """
    Check for collisions between cells using precomputed cell pairs
    
    Args:
        indices: Cell indices for each particle (from bilinear weights)
        collision_pairs: Precomputed pairs of cells where collision occurs
        last_alive: Number of active particles
    Returns:
        List of particle indices involved in collisions or empty array if none
    """
    # Handle case when there are no particles
    if last_alive == 0:
        return cp.array([], dtype=cp.int32)

    # Get cell index for each particle (bottom-left corner)
    particle_cells = indices[:last_alive]
    
    colliding_particles = []
    
    # Iterate through collision pairs
    for pair in collision_pairs:
        cell1, cell2 = pair
        
        # Find particles in cell1 and cell2
        particles_cell1 = cp.where(particle_cells == cell1)[0]
        particles_cell2 = cp.where(particle_cells == cell2)[0]
        
        # Add any particles found in these cells
        if len(particles_cell1) > 0:
            colliding_particles.extend(particles_cell1.get())
        if len(particles_cell2) > 0:
            colliding_particles.extend(particles_cell2.get())
    
    # If no collisions found, return empty array
    if not colliding_particles:
        return cp.array([], dtype=cp.int32)
        
    return cp.array(list(set(colliding_particles)))  #



def detect_collisions(positions, velocities, sorted_indices, active_cells, 
                     cell_shifts, cell_indices, wall_directions,
                     grid_shape, cell_size):
    """
    Detect and handle collisions between particles and walls in a grid-based simulation.
    
    Parameters:
    -----------
    positions : ndarray, shape=(2, n_particles)
        X,Y coordinates of particles
    velocities : ndarray, shape=(2, n_particles)
        Velocity vectors of particles
    sorted_indices : ndarray
        Indices of particles sorted by cell
    active_cells : ndarray
        Indices of cells that need collision checking
    cell_shifts : ndarray
        Starting indices in sorted_indices for each cell
    cell_indices : ndarray
        Mapping of cells to wall configurations
    wall_directions : ndarray
        Bit masks indicating wall presence (0b1000=bottom, 0b0100=top, 
        0b0010=right, 0b0001=left)
    grid_shape : tuple(int, int)
        Number of cells in X,Y directions (Nx, Ny)
    cell_size : tuple(float, float)
        Size of each cell in X,Y directions (dx, dy)
    """
    Nx, Ny = grid_shape
    Nx -= 1; Ny -= 1  # Turn node coun into cell count
    dx, dy = cell_size
    
    # Pre-compute wall direction bit masks for efficiency
    WALL_LEFT = 0b0001
    WALL_RIGHT = 0b0010
    WALL_TOP = 0b0100
    WALL_BOTTOM = 0b1000


    for active_cell in active_cells:
        if active_cell not in cell_indices:
            continue
            
        wall_idx = cp.where(cell_indices == active_cell)[0][0]
        wall_mask = wall_directions[wall_idx]
        
        start = cell_shifts[active_cell]
        end = cell_shifts[active_cell + 1] if active_cell + 1 < len(cell_shifts) else len(sorted_indices)
        particle_indices = sorted_indices[start:end]
        
        if len(particle_indices) == 0:
            continue
            
        cell_positions = positions[:, particle_indices]
        cell_velocities = velocities[:, particle_indices]
        
        # Fixed cell boundary calculation
        cy = active_cell % Ny
        cx = active_cell // Ny
        
        # Calculate exact boundaries based on cell index
        x_min = (cx + 1) * dx # Costyl' cx+1 vmesto cx, ya ne znayu pochemu
        x_max = (cx + 1) * dx
        y_min = cy * dy
        y_max = (cy + 1) * dy
        
        # Add debug prints
        # print(f"Cell {active_cell}: cx={cx}, cy={cy}, x_min={x_min:.6f}, x_max={x_max:.6f}")
        
        if wall_mask & WALL_LEFT:
            #print("Left collision   ", cell_positions[0], x_min)
            left_collision = cell_positions[0] < x_min
            cell_velocities[0, left_collision] *= -1
            cell_positions[0, left_collision] = x_min + 1e-6
            
        if wall_mask & WALL_RIGHT:
            right_collision = cell_positions[0] > x_max
            cell_velocities[0, right_collision] *= -1
            cell_positions[0, right_collision] = x_max - 1e-6
            
        if wall_mask & WALL_TOP:
            top_collision = cell_positions[1] > y_max
            cell_velocities[1, top_collision] *= -1
            cell_positions[1, top_collision] = y_max - 1e-6
            
        if wall_mask & WALL_BOTTOM:
            bottom_collision = cell_positions[1] < y_min
            cell_velocities[1, bottom_collision] *= -1
            cell_positions[1, bottom_collision] = y_min + 1e-6
        
        positions[:, particle_indices] = cell_positions
        velocities[:, particle_indices] = cell_velocities



def detect_collisions_optimized(positions, velocities, sorted_indices, active_cells, 
                              cell_shifts, cell_indices, wall_directions,
                              grid_shape, cell_size):
    """
    Optimized version of collision detection with vectorized operations and improved memory access.
    Fixes issues with wall collision detection and improves performance.

    Parameters:
    -----------
    positions : ndarray, shape=(2, n_particles)
        X,Y coordinates of particles
    velocities : ndarray, shape=(2, n_particles)
        Velocity vectors of particles
    sorted_indices : ndarray
        Indices of particles sorted by cell
    active_cells : ndarray  # cupy array
        Indices of cells that need collision checking
    cell_shifts : ndarray
        Starting indices in sorted_indices for each cell
    cell_indices : ndarray
        Mapping of cells to wall configurations
    wall_directions : ndarray
        Bit masks indicating wall presence (0b1000=bottom, 0b0100=top, 
        0b0010=right, 0b0001=left)
    grid_shape : tuple(int, int)
        Number of cells in X,Y directions (Nx, Ny)
    cell_size : tuple(float, float)
        Size of each cell in X,Y directions (dx, dy)
    """
    Nx, Ny = grid_shape
    dx, dy = cell_size
    
    # Pre-compute wall direction bit masks as uint8 for faster bit operations
    WALL_MASKS = cp.array([0b0001, 0b0010, 0b0100, 0b1000], dtype=cp.uint8)
    
    # Filter out inactive cells early
    valid_cells = cp.isin(active_cells, cell_indices)
    print("active_cells: ", active_cells, "valid_cells: ", valid_cells, end='   ')
    active_cells = active_cells[valid_cells]
    print("active_cells: ", active_cells)

    if len(active_cells) == 0:
        return
    
    # Get wall indices for all active cells at once
    wall_indices = cp.searchsorted(cell_indices, active_cells)
    wall_masks = wall_directions[wall_indices]
    
    # Prepare arrays for vectorized operations
    cell_x = (active_cells // Ny).astype(cp.float32) * dx # Move outside loop
    cell_y = (active_cells % Ny).astype(cp.float32) * dy
    
    # Pre-compute cell boundaries for all active cells
    x_mins = cell_x
    x_maxs = cell_x + dx
    y_mins = cell_y
    y_maxs = cell_y + dy
    
    # Process cells in batches for better memory efficiency
    BATCH_SIZE = 256
    n_cells = len(active_cells)
    
    # Calculate maximum particles per cell safely
    if len(cell_shifts) <= 1:
        max_particles_per_cell = len(sorted_indices)  # Use total number of particles if only one cell
    else:
        cell_sizes = cp.diff(cell_shifts)
        if len(cell_sizes) == 0:
            max_particles_per_cell = 0  # No particles case
        else:
            max_particles_per_cell = int(cp.max(cell_sizes).get())
    
    # If no particles to process, return early
    if max_particles_per_cell == 0:
        return
        
    # Pre-allocate reusable arrays for collision masks
    collision_mask = cp.zeros(max_particles_per_cell, dtype=bool)
    
    for batch_start in range(0, n_cells, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_cells)
        batch_cells = active_cells[batch_start:batch_end]
        
        # Get particle ranges for this batch
        starts = cell_shifts[batch_cells]
        ends = cell_shifts[batch_cells+1]
        
        # Process each cell in the batch
        for i, (cell_idx, start, end) in enumerate(zip(batch_cells, starts, ends)):

            #if start > end:  # Changed from start >= end to start > end
            #    continue
            print(f"Cell {cell_idx}: start={start}, end={end}")    
            # Get particles for this cell
            particle_indices = sorted_indices[start:end]
            n_particles = len(particle_indices)
            
            #if n_particles == 0:  # Skip if no particles after all
            #    continue
                
            pos = positions[:, particle_indices]
            vel = velocities[:, particle_indices]
            
            wall_mask = wall_masks[batch_start + i]
            
            # Reset collision mask for current cell
            collision_mask[:n_particles] = False
            
            # Vectorized collision detection with boundary tolerance
            EPSILON = 1e-6
            
            if wall_mask & WALL_MASKS[0]:  # Left wall
                left_mask = pos[0] <= x_mins[batch_start + i]
                collision_mask[:n_particles] |= left_mask
                print("Left collision   ", pos[0], x_mins[batch_start + i])
                print("Left mask: ", left_mask)
                vel[0, left_mask] = cp.abs(vel[0, left_mask])
                pos[0, left_mask] = x_mins[batch_start + i] + EPSILON
                
            if wall_mask & WALL_MASKS[1]:  # Right wall
                right_mask = pos[0] >= x_maxs[batch_start + i]
                collision_mask[:n_particles] |= right_mask
                print("Right collision   ", pos[0], x_maxs[batch_start + i])
                print("Right mask: ", right_mask)
                vel[0, right_mask] = -cp.abs(vel[0, right_mask])
                pos[0, right_mask] = x_maxs[batch_start + i] - EPSILON
                
            if wall_mask & WALL_MASKS[2]:  # Top wall
                top_mask = pos[1] >= y_maxs[batch_start + i]
                collision_mask[:n_particles] |= top_mask
                vel[1, top_mask] = -cp.abs(vel[1, top_mask])
                pos[1, top_mask] = y_maxs[batch_start + i] - EPSILON
                
            if wall_mask & WALL_MASKS[3]:  # Bottom wall
                bottom_mask = pos[1] <= y_mins[batch_start + i]
                collision_mask[:n_particles] |= bottom_mask
                vel[1, bottom_mask] = cp.abs(vel[1, bottom_mask])
                pos[1, bottom_mask] = y_mins[batch_start + i] + EPSILON
            
            # Update arrays only if collisions occurred
            if cp.any(collision_mask[:n_particles]):
                positions[:, particle_indices] = pos
                velocities[:, particle_indices] = vel



def detect_collisions_simple(particles:Particles.Particles2D, grid:Grid.Grid2D, wall_indices, wall_directions):
    Nx, Ny = (grid.gridshape[0] - 1, grid.gridshape[1] - 1)
    dx, dy = grid.cell_size

    valid_cells = cp.isin(particles.active_cells, wall_indices)
    valid_cells = particles.active_cells[valid_cells]

    if len(valid_cells) == 0:
        return
    
    for cell_idx in valid_cells:
        print(f"Cell {cell_idx}")
        cx = cell_idx % Ny
        cy = cell_idx // Ny
        x_min = (cx) * dx
        x_max = (cx + 1) * dx
        #print(f"Cell {cell_idx}: cx={cx}, cy={cy}, x_min={x_min:.6f}, x_max={x_max:.6f}")
        y_min = cy * dy
        y_max = (cy + 1) * dy

        for part_idx in particles.sorted_indices[particles.cell_starts[cell_idx]:particles.cell_starts[cell_idx+1]+1]:
            # Get particle position and velocity
            pos = particles.R[:, part_idx]
            vel = particles.V[:, part_idx]
            # Get wall mask for this cell
            wall_mask = wall_directions[cp.where(wall_indices == cell_idx)[0][0]]

            #print(f"Particle {part_idx}: pos={pos}, vel={vel}, wall_mask={wall_mask}")
            
            # Check for collisions with walls
            if wall_mask & 0b0001:  # Left wall
                if pos[0] < x_min:
                    print("Left collision")
                    particles.R[0, part_idx] = x_min + 1e-6
                    particles.V[0, part_idx] = cp.abs(vel[0])

            if wall_mask & 0b0010:  # Right wall
                if pos[0] > x_max:
                    print("Right collision")
                    particles.R[0, part_idx] = x_max - 1e-6
                    particles.V[0, part_idx] = -cp.abs(vel[0]) 
    
    

def remove_collided_particles(R: cp.ndarray, V: cp.ndarray, part_type: cp.ndarray, collided_indices: cp.ndarray, last_alive: int):
    """
    Remove collided particles by swapping them with last alive particles
    """
    # If no collisions, return unchanged last_alive
    if len(collided_indices) == 0:
        return last_alive

    new_last_alive = last_alive
    
    for idx in collided_indices:
        idx_scalar = int(idx.get())
        if idx_scalar >= new_last_alive:
            continue
            
        new_last_alive -= 1
        if idx_scalar == new_last_alive:
            continue
            
        R[:, idx_scalar] = R[:, new_last_alive]
        V[:, idx_scalar] = V[:, new_last_alive]
        part_type[idx_scalar] = part_type[new_last_alive]
    
    return new_last_alive



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