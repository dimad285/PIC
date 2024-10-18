from src import main
from src.Parser import parse_config
import sys

config = parse_config('input.ini')
# Use parsed parameters
CPU = config['CPU']
GPU = config['GPU']
m = config['m']
n = config['n']
k = config['k']
N = config['N']
dt = config['dt']
q = config['q']
X = config['X']
Y = config['Y']
Z = config['Z']
boundarys = config['boundarys']

if __name__ == "__main__":
    sys.exit(main.run_gpu(m, n, k, X, Y, Z, N, dt, grid_type=config['grid_type'],
                            boundary=None, 
                            RENDER=config['RENDER'], 
                            RENDER_FRAME=config['RENDER_FRAME'], 
                            DIAGNOSTICS=config['DIAGNOSTICS'],
                            solver=config['solver'], 
                            DIAG_TYPE=config['DIAG_TYPE'], 
                            bins=config['bins'],
                            SCREEN_SIZE=config['SCREEN_SIZE'],
                            UI=config['UI'],))