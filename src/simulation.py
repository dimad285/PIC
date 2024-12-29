import cupy as cp


def uniform_particle_generator_2d(x, y, dx, dy, r, v, part_type, last_alive):
    r[0, last_alive] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5)
    r[1, last_alive] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5)
    v[:, last_alive] = 0
    part_type[last_alive] = cp.random.randint(1, 3)

def uniform_particle_load(x, y, dx, dy, r, v, part_type, last_alive, n):
    r[0, last_alive:last_alive + n] = cp.random.uniform(x - dx * 0.5, x + dx * 0.5, n)
    r[1, last_alive:last_alive + n] = cp.random.uniform(y - dy * 0.5, y + dy * 0.5, n)
    v[:, last_alive:last_alive + n] = 0
    part_type[last_alive:last_alive + n] = cp.random.randint(1, 3, n)
