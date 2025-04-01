import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.collisions import profile_wall_collisions
from src.Particles import Particles2D
from src.Grid import Grid2D
from src.Boundaries import Boundaries


def main():
    particles = Particles2D(1000000)
    particles.add_species('electrons', 9.10938356e-31, -1.60217662e-19, 0)
    particles.add_species('ions', 1.67262158e-27, 1.60217662e-19, 1)
    particles.uniform_species_load(0, 0, 1, 1, 10000, 'electrons')
    grid = Grid2D(128, 128, 1, 1)
    boundaries = Boundaries([((0, 0, 0, 64), 0), ((64, 0, 64, 64), 0), ((0, 0, 0, 128), 0), ((0, 128, 128, 128), 0), ((128, 0, 128, 128), 0), ((0, 0, 128, 0), 0)], grid)
    walls = boundaries.walls

    profile_wall_collisions(particles, grid, walls)


if __name__ == "__main__":
    main()
