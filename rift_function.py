import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function
def error_function(w, theta_b):
    return -np.exp(-((w**2) / 8) - ((theta_b**2) / 1))

# Generate data
w = np.linspace(-6, 6, 100)
theta_b = np.linspace(-6, 6, 100)
w, theta_b = np.meshgrid(w, theta_b)
error = error_function(w, theta_b)

# Create a figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(w, theta_b, error, cmap='coolwarm')

# Add labels
ax.set_xlabel('w')
ax.set_ylabel('θ_b')
ax.set_zlabel('error')

# Add color bar for reference
fig.colorbar(surf)

# Enable interactive rotation
plt.ion()
plt.show(block = True)
# Instructions for the user
print("You can rotate the plot by clicking and dragging with the mouse.")