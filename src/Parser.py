import configparser
import ast

def evaluate_expression(value, variables):
    """
    Evaluate an arithmetic expression string using the given variables.

    Parameters:
        value (str): The string containing the arithmetic expression.
        variables (dict): Dictionary of variable values for evaluation.

    Returns:
        str: The evaluated result as a string.
    """
    try:
        # Safely evaluate the expression
        return str(eval(value, {"__builtins__": None}, variables))
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{value}': {e}")

def resolve_variables_and_evaluate(config, value, variables):
    """
    Resolve variables in a value string and evaluate arithmetic expressions.

    Parameters:
        config (configparser.ConfigParser): The config parser object.
        value (str): The value string potentially containing variables and expressions.
        variables (dict): Precomputed variable dictionary for evaluation.

    Returns:
        str: The evaluated result with variables resolved and expressions computed.
    """
    for section in config.sections():
        for key, val in config.items(section):
            #print(f"Processing key: {key}, value: {val}")  # Debugging statement
            try:
                variables[key] = eval(val, {"__builtins__": None}, variables)
            except Exception as e:
                raise ValueError(f"Error processing key '{key}' with value '{val}': {e}")
    return evaluate_expression(value, variables)

def parse_config(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    
    params = {}
    variables = {}  # Store resolved variables for arithmetic operations
    
    # Simulation parameters
    params['CPU'] = config.getboolean('Execution', 'CPU')
    params['GPU'] = config.getboolean('Execution', 'GPU')
    
    # Grid parameters
    for key, val in config.items('Grid'):
        try:
            # Handle boolean-like strings
            if val.lower() in {"true", "false"}:
                variables[key] = val.lower() == "true"
            # Handle numeric values explicitly
            elif val.replace('.', '', 1).isdigit():
                variables[key] = float(val) if '.' in val else int(val)
            # Directly assign non-numeric strings (e.g., "cartesian_2")
            elif val.isalpha() or "_" in val:
                variables[key] = val
            # Evaluate expressions for arithmetic and variable substitution
            else:
                variables[key] = eval(val, {"__builtins__": None}, variables)
        except Exception as e:
            raise ValueError(f"Error processing key '{key}' with value '{val}': {e}")

    params['grid_type'] = config.get('Grid', 'dim')
    params['m'] = config.getint('Grid', 'm')
    params['n'] = config.getint('Grid', 'n')
    params['k'] = config.getint('Grid', 'k')
    params['X'] = config.getfloat('Grid', 'X')
    params['Y'] = config.getfloat('Grid', 'Y')
    params['Z'] = config.getfloat('Grid', 'Z')

    # Time parameters
    params['dt'] = config.getfloat('Time', 'dt')
    
    # Boundary conditions with variable resolution and arithmetic support
    params['boundarys'] = []    
    for key, value in config.items('Boundaries'):
        resolved_value = resolve_variables_and_evaluate(config, value.strip(), variables)
        boundary = ast.literal_eval(resolved_value)
        params['boundarys'].append(boundary)
    
    # Parrticle parameters
    params['max_particles'] = config.getint('Particles', 'max_particles')

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
    config_params = parse_config('input.ini')
    print(config_params)
