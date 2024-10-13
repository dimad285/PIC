import cupy as cp

def boundary_array(input:tuple, gridsize:tuple) -> cp.ndarray:
    boundary = [[],[]]
    for i in input:
        m1, n1, m2, n2 = i[0]
        dm = (m2 - m1)
        if dm > 0:
            dm = 1
        elif dm < 0:
            dm = -1
        else:
            dm = 0
        dn = (n2 - n1)
        if dn > 0:
            dn = 1
        elif dn < 0:
            dn = -1
        else:
            dn = 0
        x = m1
        y = n1
        while True:
            boundary[0].append(y*gridsize[0]+x)
            boundary[1].append(i[1])
            x = int(x+dm)
            y = int(y+dn)
            #print(x, y)
            if x == m2 and y == n2:
                boundary[0].append(y*gridsize[0]+x)
                boundary[1].append(i[1])
                break
    
    return cp.asarray(boundary, dtype=cp.int32)


def boundary_conditions_left_gpu(A:cp.ndarray, boundary:cp.ndarray):
    for i in boundary[0]:
        A[i,:] = 0
        A[i, i] = 1



def Laplacian_square(m, n) -> cp.ndarray:

    Lap = cp.zeros((m*n, m*n), dtype=cp.float32)
    for i in range(n):
        for j in range(m):
            idx = i*m+j
            Lap[idx, idx] = -4
            if j>0:
                Lap[idx, idx-1] = 1
            if j<m-1:
                Lap[idx, idx+1] = 1
            if i>0:
                Lap[idx, idx-m] = 1
            if i<n-1:
                Lap[idx, idx+m] = 1
    return Lap.astype(cp.float32)


def Laplacian_cilindrical(m, n) -> cp.ndarray:

    Lap = cp.zeros((m*n, m*n), dtype=cp.float32)
    for i in range(n):
        for j in range(m):
            idx = i*m+j
            Lap[idx, idx] = -4
            if j>0:
                Lap[idx, idx-1] = 1
            if j<m-1:
                Lap[idx, idx+1] = 1
            if i>0:
                Lap[idx, idx-m] = 1
            if i<n-1:
                Lap[idx, idx+m] = 1
    return Lap.astype(cp.float32)


def setup_fft_solver(m, n):
    # Create wavenumber arrays
    kx = 2 * cp.pi * cp.fft.fftfreq(m)
    ky = 2 * cp.pi * cp.fft.fftfreq(n)
    
    # Create 2D wavenumber grid
    kx_grid, ky_grid = cp.meshgrid(kx, ky)
    
    # Compute k^2, avoiding division by zero at k=0
    k_sq = kx_grid**2 + ky_grid**2
    k_sq[0, 0] = 1.0  # Avoid division by zero
    
    return k_sq


def solve_poisson_fft(rho, k_sq, epsilon0):
    # Reshape rho to 2D
    rho_2d = rho.reshape(k_sq.shape)
    
    # Compute FFT of charge density
    rho_k = cp.fft.fftn(rho_2d)
    
    # Solve Poisson equation in Fourier space
    phi_k = rho_k / (k_sq * epsilon0)
    
    # Handle k=0 mode (set to average of phi)
    phi_k[0, 0] = 0
    
    # Inverse FFT to get potential
    phi = cp.fft.ifftn(phi_k).real / (k_sq.shape[0] * k_sq.shape[1])
    
    return phi.ravel()  # Return as 1D array
