import numpy as np

class GridAlignedWalls:
    def __init__(self, nx, ny, nz, dx, dy, dz):
        """
        Initialize wall configuration for grid-aligned walls.
        
        Parameters:
        -----------
        nx, ny, nz : int
            Number of grid cells in each dimension
        dx, dy, dz : float
            Grid spacing in each dimension
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        
        # Wall markers for each cell face (True = wall present)
        # Indexed by cell number, represents right/top/front face of cell
        self.x_walls = np.zeros((nx+1, ny, nz), dtype=bool)
        self.y_walls = np.zeros((nx, ny+1, nz), dtype=bool)
        self.z_walls = np.zeros((nx, ny, nz+1), dtype=bool)
        
        # Wall potentials (for conducting walls)
        self.wall_potentials = {
            'x': np.zeros((nx+1, ny, nz)),
            'y': np.zeros((nx, ny+1, nz)),
            'z': np.zeros((nx, ny, nz+1))
        }

    def add_wall(self, direction, index, start, end, potential=0.0):
        """
        Add a wall along a grid plane.
        
        Parameters:
        -----------
        direction : str
            'x', 'y', or 'z' for wall normal direction
        index : int
            Grid index where wall is located
        start : tuple
            Starting indices in other two dimensions
        end : tuple
            Ending indices in other two dimensions
        potential : float
            Wall potential for conducting walls
        """
        if direction == 'x':
            self.x_walls[index, start[0]:end[0], start[1]:end[1]] = True
            self.wall_potentials['x'][index, start[0]:end[0], start[1]:end[1]] = potential
        elif direction == 'y':
            self.y_walls[start[0]:end[0], index, start[1]:end[1]] = True
            self.wall_potentials['y'][start[0]:end[0], index, start[1]:end[1]] = potential
        elif direction == 'z':
            self.z_walls[start[0]:end[0], start[1]:end[1], index] = True
            self.wall_potentials['z'][start[0]:end[0], start[1]:end[1], index] = potential

    def check_particle_collision(self, old_pos, new_pos):
        """
        Check if particle trajectory intersects any walls.
        
        Parameters:
        -----------
        old_pos : ndarray
            Previous particle position
        new_pos : ndarray
            New particle position
        
        Returns:
        --------
        collision : bool
            Whether collision occurred
        wall_normal : ndarray or None
            Normal vector of wall hit
        intersection : ndarray or None
            Point of intersection
        """
        # Convert positions to cell indices
        old_idx = np.array([old_pos[0]/self.dx, old_pos[1]/self.dy, old_pos[2]/self.dz])
        new_idx = np.array([new_pos[0]/self.dx, new_pos[1]/self.dy, new_pos[2]/self.dz])
        
        # Check each dimension for wall crossings
        for dim in range(3):
            if old_idx[dim] == new_idx[dim]:
                continue
                
            # Find all cell boundaries crossed
            start_idx = min(int(old_idx[dim]), int(new_idx[dim]))
            end_idx = max(int(old_idx[dim]), int(new_idx[dim])) + 1
            
            for idx in range(start_idx, end_idx):
                # Check if there's a wall at this index
                wall_present = False
                if dim == 0 and self.x_walls[idx, int(old_idx[1]), int(old_idx[2])]:
                    wall_present = True
                elif dim == 1 and self.y_walls[int(old_idx[0]), idx, int(old_idx[2])]:
                    wall_present = True
                elif dim == 2 and self.z_walls[int(old_idx[0]), int(old_idx[1]), idx]:
                    wall_present = True
                
                if wall_present:
                    # Compute intersection point
                    t = (idx * [self.dx, self.dy, self.dz][dim] - old_pos[dim]) / (new_pos[dim] - old_pos[dim])
                    intersection = old_pos + t * (new_pos - old_pos)
                    
                    # Create wall normal
                    normal = np.zeros(3)
                    normal[dim] = np.sign(new_pos[dim] - old_pos[dim])
                    
                    return True, normal, intersection
        
        return False, None, None

    def apply_field_boundary_conditions(self, phi):
        """
        Apply boundary conditions to potential field.
        
        Parameters:
        -----------
        phi : ndarray
            Electric potential field
        
        Returns:
        --------
        phi : ndarray
            Updated potential field
        """
        # Apply conducting wall boundary conditions
        for i in range(self.nx+1):
            mask = self.x_walls[i,:,:]
            if mask.any():
                phi[i,mask] = self.wall_potentials['x'][i,mask]
        
        for j in range(self.ny+1):
            mask = self.y_walls[:,j,:]
            if mask.any():
                phi[mask,j] = self.wall_potentials['y'][mask,j]
        
        for k in range(self.nz+1):
            mask = self.z_walls[:,:,k]
            if mask.any():
                phi[mask,k] = self.wall_potentials['z'][mask,k]
        
        return phi