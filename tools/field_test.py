import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("fields/fields_cylindrical.txt", skiprows=1)

# Extract columns
x, y, rho, phi, Ex_loaded, Ey_loaded = data.T

# Determine grid size
x_unique = np.unique(x)
y_unique = np.unique(y)
nx, ny = len(x_unique), len(y_unique)

# Reshape to 2D grid
X = x.reshape((ny, nx))
Y = y.reshape((ny, nx))
Phi = phi.reshape((ny, nx))
Ex_loaded = Ex_loaded.reshape((ny, nx))
Ey_loaded = Ey_loaded.reshape((ny, nx))

# Compute electric field from potential
dy, dx = y_unique[1] - y_unique[0], x_unique[1] - x_unique[0]
Ey_computed, Ex_computed = np.gradient(-Phi, dy, dx)

# Plot potential
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.contourf(X, Y, Phi, cmap="viridis")
plt.colorbar(label="Potential (phi)")
plt.title("Potential Field")
plt.xlabel("x")
plt.ylabel("y")

# Quiver: loaded E field
plt.subplot(1, 3, 2)
plt.quiver(X, Y, Ex_loaded, Ey_loaded, color="red", scale=5000)
plt.title("Loaded Electric Field")
plt.xlabel("x")
plt.ylabel("y")

# Quiver: computed E field
plt.subplot(1, 3, 3)
plt.quiver(X, Y, Ex_computed, Ey_computed, color="blue", scale=5000)
plt.title("Computed Electric Field from -∇phi")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()