# Simulation Flow Documentation

## Entry Point

The simulation starts from `PIC_GPU.pyw`, which:
1. Loads configuration from `input.ini`
2. Calls `main.run_gpu()` with parsed parameters

## Initialization Phase (`run_gpu()`)

1. **Component Initialization**
   ```python
   particles = Particles.Particles2D(max_particles, cylindrical=cylindrical)
   grid = Grid.Grid2D(m, n, X, Y, cylindrical=cylindrical)
   diagnostics = simulation.Diagnostics()
   ```

2. **Boundary Setup**
   ```python
   if boundary != None:
       boundaries = Boundaries.Boundaries(boundary, grid)
       solver = Solvers.Solver(solver_type, grid, boundaries=boundaries.conditions)
   else:
       solver = Solvers.Solver(solver_type, grid)
   ```

3. **UI and Renderer Setup**
   ```python
   if UI:
       root = tk.Tk()
       gui = Interface.SimulationUI_tk(root)
   if RENDER:
       renderer = Render.PICRenderer(*SCREEN_SIZE, fontfile, max_particles)
   ```

## Main Simulation Loop

### 1. UI State Update
```python
if UI:
    root.update()
    state = gui.get_state()
```

### 2. Physics Step (`simulation.step()`)
Sequence of operations:
1. **Particle Position Update**
   ```python
   particles.update_R(dt)  # Move particles
   ```

2. **Boundary Processing**
   ```python
   collisions.remove_out_of_bounds(particles, grid.X, grid.Y)
   ```

3. **Grid Interpolation**
   ```python
   particles.update_bilinear_weights(grid)  # Particle-to-grid interpolation
   ```

4. **Field Solving**
   ```python
   grid.update_density(particles)  # Deposit charge density
   solver.solve(grid)              # Solve Poisson equation
   grid.update_E()                # Calculate electric field
   ```

5. **Velocity Update**
   ```python
   particles.update_V(grid, dt)  # Update particle velocities
   ```

### 3. Diagnostics Update
```python
diagnostics.update(t, particles, grid, sim_time)
```

### 4. Visualization (`simulation.draw()`)
If rendering is enabled:
1. **Plot Selection**
   - Particles view
   - Field heatmap
   - Diagnostic plots
   - Surface plots

2. **Render Update**
   ```python
   renderer.update_particles()  # or
   renderer.update_heatmap()   # or
   renderer.update_surface()
   ```

3. **Display Info**
   ```python
   renderer.update_legend('sim', f"Sim time: {sim_time}")
   renderer.update_legend('n', f"N: {particles.last_alive}")
   ```

## Termination

1. **Field Data Export**
   ```python
   if save_fields:
       grid.save_to_txt('fields/fields.txt',
                       fields={'rho': grid.rho, 'phi': grid.phi})
   ```

2. **Cleanup**
   ```python
   if RENDER:
       renderer.close()
   ```

## Time Management

- Main timestep: `dt` (configurable)
- Simulation time: `t += dt`
- Performance metrics:
  - `sim_time`: Physics computation time
  - `frame_time`: Rendering time

## Key Control Points

1. **Simulation Control**
   - Start/Stop via UI
   - Step-by-step execution
   - Real-time parameter adjustment

2. **Visualization Control**
   - Plot type selection
   - Variable selection
   - Camera control (for 3D views)

3. **Diagnostic Output**
   - Energy conservation
   - Particle count
   - Performance metrics