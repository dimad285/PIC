import sys
import Run
from Parser import parse_config

config = parse_config('input.ini')

# Use parsed parameters
CPU = config['CPU']
GPU = config['GPU']
m = config['m']
n = config['n']
N = config['N']
dt = config['dt']
q = config['q']
X = config['X']
Y = config['Y']
boundarys = config['boundarys']

if __name__ == "__main__":
    if CPU:
        sys.exit(Run.run_cpu(m, n, X, Y, N, dt, RENDER=True))
    elif GPU:
        sys.exit(Run.run_gpu(m, n, X, Y, N, dt,
                             boundary=None, 
                             RENDER=config['RENDER'], 
                             RENDER_FRAME=config['RENDER_FRAME'], 
                             DIAGNOSTICS=config['DIAGNOSTICS'],
                             solver=config['solver'], 
                             DIAG_TYPE=config['DIAG_TYPE'], 
                             bins=config['bins'],
                             SCREEN_SIZE=config['SCREEN_SIZE'],
                             UI=config['UI'],))
    else:
        print("Please select CPU or GPU in the configuration file")
