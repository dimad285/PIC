# API Reference

## Simulation Control

### `run_gpu()`
Main simulation loop with parameters:
- `m`, `n`: Grid dimensions
- `X`, `Y`: Domain size
- `max_particles`: Particle limit
- `dt`: Time step
- `cylindrical`: Coordinate system
- `boundary`: Boundary conditions
- `solver_type`: Field solver
- `RENDER`, `UI`: Display options

## Key Classes

### `Particles2D`
```python
class Particles2D:
    def __init__(self, N, cylindrical=False)
    def update_R(self, dt)
    def update_V(self, grid, dt)
    def update_bilinear_weights(self, grid)
```

### `Grid2D`
```python
class Grid2D:
    def __init__(self, m, n, X, Y, cylindrical=False)
    def update_density(self, particles)
    def update_E(self)
    def save_to_txt(self, filename, fields, header)
```

### `Solver`
```python
class Solver:
    def __init__(self, solver_type, grid, cylindrical=False, 
                 boundaries=None, tol=1e-5)
    def solve(self, grid)
```

### `SimulationUI`
```python
class SimulationUI_tk:
    def __init__(self, root)
    def get_state(self)
    def update(self)
```

## Utility Functions

### `parse_config(filename)`
Parses simulation configuration from INI file

### `read_cross_section(filename)`
Loads collision cross-section data