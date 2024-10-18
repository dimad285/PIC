import configparser
import ast

def parse_config(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    
    params = {}
    
    # Simulation parameters
    params['CPU'] = config.getboolean('Execution', 'CPU')
    params['GPU'] = config.getboolean('Execution', 'GPU')
    
    # Grid parameters
    params['grid_type'] = config.get('Grid', 'type')
    params['m'] = config.getint('Grid', 'm')
    params['n'] = config.getint('Grid', 'n')
    params['k'] = config.getint('Grid', 'k')
    
    # Particle parameters
    params['N'] = config.getint('Particles', 'N')
    params['dt'] = config.getfloat('Time', 'dt')
    params['q'] = config.getfloat('Particles', 'q')
    params['X'] = config.getfloat('Grid', 'X')
    params['Y'] = config.getfloat('Grid', 'Y')
    params['Z'] = config.getfloat('Grid', 'Z')
    
    # Boundary conditions
    boundary_list = ast.literal_eval(config.get('Boundaries', 'boundarys'))
    params['boundarys'] = tuple(([int(x) for x in b[0]], b[1]) for b in boundary_list)
    
    # GPU specific parameters
    if params['GPU']:
        params['RENDER'] = config.getboolean('GPU', 'RENDER')
        params['RENDER_FRAME'] = config.getint('GPU', 'RENDER_FRAME')
        params['DIAGNOSTICS'] = config.getboolean('GPU', 'DIAGNOSTICS')
        params['solver'] = config.get('GPU', 'solver')
        params['DIAG_TYPE'] = config.get('GPU', 'DIAG_TYPE')
        params['bins'] = config.getint('GPU', 'bins')
    
    # Window specific parameters
    params['SCREEN_SIZE'] = (config.getint('Windows', 'SCREEN_WIDTH'), config.getint('Windows', 'SCREEN_HEIGHT'))
    params['UI'] = config.getboolean('UI', 'UI')

    return params

# Usage
if __name__ == "__main__":
    config_params = parse_config('simulation_config.ini')
    print(config_params)