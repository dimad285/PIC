import cupy as cp
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
#project_root = Path(__file__).parent.parent  # This goes up one level from tests/ to the project root
#sys.path.append(str(project_root))
from src.Grid import Grid2D

print(cp.__version__)