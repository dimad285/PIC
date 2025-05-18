import cupy as cp

# Combined kernel that performs multiple smoothing iterations in a single launch
combined_smoothing_kernel = cp.RawKernel(r'''
extern "C" __global__
void combined_smoothing(
    double* phi,
    const double* rho,
    double h2,
    double omega,
    int nx,
    int ny,
    int iterations
) {
    // Calculate thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    
    const int i = idx / nx;
    const int j = idx % nx;
    const int is_interior = (i > 0 && i < ny - 1 && j > 0 && j < nx - 1) ? 1 : 0;
    const int is_red = (i + j) % 2;
    const int is_black = 1 - is_red;
   
    // Load phi value into register
    double my_phi = (is_interior) ? phi[idx] : 0.0;
   
    // Loop through multiple iterations in a single kernel launch
    for (int iter = 0; iter < iterations; iter++) {
        // Perform red-black Gauss-Seidel
        for (int color = 0; color < 2; color++) {
            // Synchronize threads before each color sweep
            __syncthreads();
           
            // Only interior points with the current color perform the update
            if (is_interior && ((color == 0 && is_red) || (color == 1 && is_black))) {
                // Load neighboring phi values
                double phi_up = phi[idx + nx];
                double phi_down = phi[idx - nx];
                double phi_left = phi[idx - 1];
                double phi_right = phi[idx + 1];
               
                // Compute update
                double update = 0.25 * (
                    phi_up + phi_down + phi_left + phi_right - h2 * rho[idx]
                );
               
                // Apply relaxation
                my_phi = (1.0 - omega) * my_phi + omega * update;
               
                // Write back to global memory
                phi[idx] = my_phi;
            }
            
            // Wait for all threads to finish this color before continuing
            __syncthreads();
        }
    }
}
''', 'combined_smoothing')