import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function
def error_function(w, theta_b):
    term1 = np.exp(-((w**2) / 8) - ((theta_b**2) / 8))
    term2 = np.exp(-(((w + 3)**2) / 2) - (((theta_b + 3)**2) / 2))
    return term1 - term2

# Generate data
w = np.linspace(-6, 6, 100)
theta_b = np.linspace(-6, 6, 100)
w, theta_b = np.meshgrid(w, theta_b)
error = error_function(w, theta_b)

# Plot the function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w, theta_b, error, cmap='viridis')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)')
plt.show()