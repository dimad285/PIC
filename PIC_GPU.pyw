from src import main
import cProfile
import pstats
import io

if __name__ == "__main__":
    parser = main.ConfigParser()
    config_dict = parser.parse_file('test.inp')
    config = main.SimulationConfig(config_dict)
    manager = main.SimulationManager(config)
    
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
