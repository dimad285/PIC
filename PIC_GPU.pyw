from src import SimulationManager, ConfigParser
import cProfile
import pstats
import io

PROFILE = True

parser = ConfigParser()
config_dict = parser.parse_file('test.inp')

manager = SimulationManager(config_dict)

if PROFILE:
    pr = cProfile.Profile()
    pr.enable()

    manager.run()

    pr.disable()
    pr.dump_stats("profile.prof")
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open("profile.txt", "w") as f:
        f.write(s.getvalue())
else:
    manager.run()