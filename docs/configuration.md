# Configuration Guide

## Input File Structure

The simulation is configured through `input.ini`, which contains several sections:

### [Execution]
- `CPU`: Boolean, enable CPU execution
- `GPU`: Boolean, enable GPU execution

### [Grid]
- `dim`: Integer, dimensionality (2)
- `m`, `n`: Integer, grid points in x/z and y/r directions
- `X`, `Y`: Float, physical domain size

### [Particles]
- `max_particles`: Integer, maximum number of particles
- `q`: Float, particle charge

### [Time]
- `dt`: Float, time step size

### [Boundaries]
Define boundary conditions as tuples:
```ini
cathode = ([m/4, n/4, m/4, n*3/4], 0)
anode = ([m*3/4, n/4, m*3/4, n*3/4], 100)
```

### [GPU]
- `RENDER`: Boolean, enable visualization
- `RENDER_FRAME`: Integer, frame skip for rendering
- `solver`: String, solver type ('fft', 'cg', 'gmres', 'multigrid')

### [Windows]
- `SCREEN_WIDTH`, `SCREEN_HEIGHT`: Integer, window dimensions

### [UI]
- `UI`: Boolean, enable interactive interface