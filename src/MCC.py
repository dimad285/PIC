import cupy as cp
import numpy as np

def read_cross_section(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    cross = cp.array([[float(line[0]), float(line[1])] for line in lines])
    return cross.T

def update_cross_section(S, E, crosssection):
    i = cp.floor(E/crosssection[0, -1]).astype(cp.int32)
    j = i + 1
    k = (crosssection[1, j] - crosssection[1, i]) / (crosssection[0, j] - crosssection[0, i])
    S[:] = k * (E - crosssection[0, i]) + crosssection[1, i]

def MCC(sigma, V, NGD, P, dt):
    v = cp.hypot(V[0], V[1])
    P[:] = 1 - cp.exp(-dt * NGD * sigma * v)

def collision_cos(energy:cp.ndarray, r:cp.ndarray):
    return (2 + energy - 2 * cp.power(energy, r)) / energy

def collision_probability_distribution(probabilities, num_bins=50):
    """
    Compute the x (bin edges) and y (counts) arrays for the probability distribution.

    Parameters:
    probabilities (cupy.ndarray): Array of precomputed collision probabilities.
    num_bins (int): Number of bins for the histogram.

    Returns:
    bin_centers (numpy.ndarray): The centers of the histogram bins (x values).
    hist (numpy.ndarray): The histogram counts (y values).
    """
    # Convert CuPy array to NumPy for histogram computation
    probabilities_np = cp.asnumpy(probabilities)

    # Compute histogram
    hist, bin_edges = np.histogram(probabilities_np, bins=num_bins, density=True)

    # Compute bin centers from bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, hist