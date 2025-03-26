# Particle-in-Cell (PIC) Simulation

A GPU-accelerated 2D particle-in-cell plasma simulation code supporting both Cartesian and cylindrical coordinates.

## Features

- GPU acceleration using CuPy
- 2D geometry (Cartesian and cylindrical coordinates)
- Multiple field solvers (FFT, CG, GMRES, Multigrid)
- Interactive visualization and control
- Configurable boundary conditions
- Monte Carlo Collisions (MCC) module
- Real-time diagnostics

## Requirements

- Python 3.8+
- CUDA-capable GPU
- Required packages:
  - cupy
  - numpy
  - tkinter
  - glfw
  - OpenGL

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure simulation parameters in `input.ini`

3. Run the simulation:
```bash
python PIC_GPU.pyw
```

## Configuration

The simulation is configured through `input.ini`. Key parameters include:

- Grid dimensions and size
- Number of particles
- Time step
- Boundary conditions
- Visualization settings
- Solver selection

See `docs/configuration.md` for detailed configuration options.