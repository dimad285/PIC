import ast

class ConfigParser:
    def __init__(self):
        self.variables = {}
        self.system = {}
        self.grid = {}
        self.equipotentials = []
        self.species = []
        self.control = {}
        self.loads = []  # Change to list to store multiple Load blocks

    def parse_variables(self, content):
        """Parse Variables block"""
        for line in content:
            if '=' in line and not line.strip().startswith('//'):
                key, value = [x.strip() for x in line.split('=')]
                try:
                    self.variables[key] = self._evaluate_expression(value)
                except:
                    self.variables[key] = value

    def _evaluate_expression(self, expr):
        """Evaluate mathematical expressions with variable substitution"""
        if not isinstance(expr, str):
            return expr
            
        try:
            # First pass: substitute all variables
            modified_expr = expr
            for var, value in self.variables.items():
                if var in modified_expr:
                    # Convert value to string, handling both numbers and expressions
                    val_str = str(value) if isinstance(value, (int, float)) else f"({value})"
                    modified_expr = modified_expr.replace(var, val_str)
            
            # Second pass: evaluate the resulting expression
            safe_dict = {
                "__builtins__": None,
                "abs": abs,
                "float": float,
                "int": int,
                "pow": pow,
            }
            safe_dict.update(self.variables)
            return float(eval(modified_expr, safe_dict, {}))
        except Exception as e:
            return expr  # Return as is if not evaluatable

    def parse_system(self, content):
        """Parse System block containing backend parameters"""
        for line in content:
            if '=' in line and not line.strip().startswith('//'):
                key, value = [x.strip() for x in line.split('=')]
                try:
                    # Convert to appropriate type (int, float, or bool)
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.','',1).isdigit():
                        value = float(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    self.system[key] = value
                except ValueError:
                    self.system[key] = value

    def parse_control(self, content):
        """Parse Control block with variable substitution"""
        for line in content:
            if '=' in line and not line.strip().startswith('//'):
                key, value = [x.strip() for x in line.split('=')]
                # Handle string values (like 'gmres') differently from expressions
                if value.startswith("'") or value.startswith('"'):
                    # Strip quotes for string values
                    self.control[key] = value.strip("'\"")
                else:
                    # For non-string values, evaluate with variable substitution
                    try:
                        # First try to get the value from variables if it exists
                        if value in self.variables:
                            self.control[key] = self.variables[value]
                        else:
                            # Otherwise evaluate as expression
                            self.control[key] = self._evaluate_expression(value)
                    except:
                        # If evaluation fails, store as is
                        self.control[key] = value

    def parse_equipotential(self, content):
        """Parse Equipotential block for boundary conditions"""
        boundary = {}
        for line in content:
            if '=' in line:
                key, value = [x.strip() for x in line.split('=')]
                boundary[key] = self._evaluate_expression(value)
        
        if all(k in boundary for k in ['j1', 'k1', 'j2', 'k2', 'C']):
            # Create boundary condition tuple ((j1,k1,j2,k2), C)
            coords = (
                boundary['j1'],
                boundary['k1'],
                boundary['j2'],
                boundary['k2']
            )
            self.equipotentials.append((coords, boundary['C']))

    def parse_species(self, content):
        """Parse Species block for particle types"""
        species = {}
        for line in content:
            if '=' in line:
                key, value = [x.strip() for x in line.split('=')]
                species[key] = self._evaluate_expression(value)
        
        if all(k in species for k in ['name', 'm', 'q', 'collisionModel']):
            self.species.append(species)
        else:
            print("Warning: Incomplete species definition.")

    def parse_grid(self, content):
        """Parse Grid block containing grid parameters"""
        for line in content:
            if '=' in line:
                key, value = [x.strip() for x in line.split('=')]
                self.grid[key] = self._evaluate_expression(value)

    def parse_load(self, content):
        """Parse Load block for initial particle distribution"""
        load = {}
        for line in content:
            if '=' in line and not line.strip().startswith('//'):
                key, value = [x.strip() for x in line.split('=')]
                # First check if the value exists in variables
                if value in self.variables:
                    load[key] = self.variables[value]
                else:
                    # If not in variables, try to evaluate it
                    load[key] = self._evaluate_expression(value)
        self.loads.append(load)

    def parse_file(self, filepath):
        """Parse the entire configuration file"""
        with open(filepath, 'r') as f:
            content = f.read()

        current_block = None
        block_content = []
        previous_line = ""

        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            if line.endswith('{'):
                # If line is just '{', use previous line as block name
                if line == '{':
                    current_block = previous_line.strip()
                    print(current_block)
                else:
                    current_block = line[:-1].strip()
                block_content = []
            elif line == '}':
                if current_block == 'Variables':
                    self.parse_variables(block_content)
                elif current_block == 'System':
                    self.parse_system(block_content)
                elif current_block == 'Equipotential':
                    self.parse_equipotential(block_content)
                elif current_block == 'Species':
                    self.parse_species(block_content)
                elif current_block == 'Grid':
                    self.parse_grid(block_content)
                elif current_block == 'Control':
                    self.parse_control(block_content)
                elif current_block == 'Load':
                    self.load = self.parse_load(block_content)  # Parse Load block
                current_block = None
                block_content = []
            else:
                if current_block is None and line and not line.startswith('//'):
                    previous_line = line
                elif current_block is not None:
                    block_content.append(line)

        return self._create_config_dict()

    def _resolve_value(self, value):
        """Resolve a value by substituting variables and evaluating expressions"""
        if not isinstance(value, str):
            return value
            
        # If the value is directly in variables, return it
        if value in self.variables:
            return self.variables[value]
            
        # Otherwise try to evaluate it as an expression
        try:
            return self._evaluate_expression(value)
        except:
            return value

    def _resolve_dict_values(self, dictionary):
        """Recursively resolve all values in a dictionary"""
        resolved = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_dict_values(value)
            elif isinstance(value, list):
                resolved[key] = [self._resolve_value(v) for v in value]
            else:
                resolved[key] = self._resolve_value(value)
        return resolved

    def _create_config_dict(self):
        """Create a configuration dictionary from parsed data with resolved variables"""
        # First ensure all variables are fully resolved
        resolved_variables = self._resolve_dict_values(self.variables)
        self.variables = resolved_variables

        # Now resolve other sections using the resolved variables
        config = {
            'system': self._resolve_dict_values(self.system),
            'variables': resolved_variables,
            'boundaries': self.equipotentials,
            'species': [self._resolve_dict_values(s) for s in self.species],
            'grid': self._resolve_dict_values(self.grid),
            'control': self._resolve_dict_values(self.control),
            'loads': [self._resolve_dict_values(load) for load in self.loads]  # Resolve each load block
        }
        return config













