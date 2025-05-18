import logging
from pathlib import Path

def setup_logging(log_dir: Path = Path("logs")):
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "simulation.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("PIC_Simulation")