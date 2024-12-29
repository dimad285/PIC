import cupy as cp
import cupyx as cpx
import Update

N = 1000000
m = 64
n = 64
X = 1
Y = 1
dx = X / (m - 1)
dy = Y / (n - 1)
dt = 0.001
gridsize = (m, n)
part_type = cp.random.randint(0, 2, N)
q_type = cp.array([1, -1], dtype=cp.float32)
m_type_1 = cp.array([1, 1], dtype=cp.float32)
last_alive = N-1

R = cp.random.rand(2, N)
V = cp.zeros((2, N), dtype=cp.float32)
E = cp.zeros((2, m*n), dtype=cp.float32) # electric field

threads_per_block = 256
blocks_per_grid = (last_alive + threads_per_block - 1) // threads_per_block
print(cpx.profiler.benchmark(Update.update_V[blocks_per_grid, threads_per_block], (R, V, E, part_type, q_type, m_type_1, gridsize, dt, dx, dy, last_alive), n_repeat = 1000))