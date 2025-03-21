import cupy as cp
import Particles


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



def remove_collided_particles(particles:Particles.Particles2D, collided_indices: cp.ndarray):
    """
    Remove collided particles by swapping them with last alive particles
    """
    # If no collisions, return unchanged last_alive
    if len(collided_indices) == 0:
        return particles.last_alive

    new_last_alive = particles.last_alive
    
    for idx in collided_indices:
        idx_scalar = int(idx.get())
        if idx_scalar >= new_last_alive:
            continue
            
        new_last_alive -= 1
        if idx_scalar == new_last_alive:
            continue
            
        particles.R[:, idx_scalar] = particles.R[:, new_last_alive]
        particles.V[:, idx_scalar] = particles.V[:, new_last_alive]
        particles.part_type[idx_scalar] = particles.part_type[new_last_alive]
    
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