from src.simulation import SimulationManager, SimulationConfig
import cProfile
import pstats
import io
from src.ConfigParser import ConfigParser



if __name__ == "__main__":
    parser = ConfigParser()
    config_dict = parser.parse_file('test.inp')
    config = SimulationConfig(config_dict)
    manager = SimulationManager(config)
    
    # Start profiling before running simulation
    pr = cProfile.Profile()
    pr.enable()
    
    # Run simulation
    manager.run()
    
    # Stop profiling and save results
    pr.disable()
    pr.dump_stats("profile.prof")
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open("profile.txt", "w") as f:
        f.write(s.getvalue())
