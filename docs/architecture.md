# Architecture Overview

## Core Components

### Particles (`src/Particles.py`)
Manages particle data and operations:
- Position and velocity arrays
- Particle weights
- Particle-grid interpolation
- Particle pushing (Boris algorithm)

### Grid (`src/Grid.py`)
Handles field quantities and grid operations:
- Charge density
- Electric potential
- Electric field
- Grid metrics and coordinates

### Solvers (`src/Solvers.py`)
Field solvers for Poisson equation:
- FFT solver
- Conjugate Gradient
- GMRES
- Multigrid

### Boundaries (`src/Boundaries.py`)
Implements boundary conditions:
- Dirichlet/Neumann conditions
- Particle-boundary interactions
- Surface charging

### MCC (`src/MCC.py`)
Monte Carlo Collision module:
- Cross-section handling
- Collision probability calculation
- Particle scattering

### Visualization (`src/Render.py`)
Real-time visualization:
- OpenGL rendering
- Particle plots
- Field visualization
- Diagnostic plots

## Data Flow

1. Particle positions → Grid (weight calculation)
2. Grid → Charge density
3. Poisson solve → Electric potential
4. Field calculation → Electric field
5. Field → Particle acceleration
6. Particle pushing → New positions
7. Boundary checking and collisions
8. Visualization update

## Performance Considerations

- GPU memory management
- Particle sorting for cache efficiency
- Efficient field interpolation
- Optimized boundary checks