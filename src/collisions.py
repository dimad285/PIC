import cupy as cp


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
                coll_map.append([i, i + 1])
    
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