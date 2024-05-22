import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function definitions
def f(x1, x2):
    return -np.exp(-((x1**2) / 8) - ((x2**2) / 1))

def deep_groove_function(x):
    return f(x[0], x[1])

def calc_numerical_gradient(func, x, delta_x=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_pos = np.copy(x)
        x_pos[i] += delta_x
        grad[i] = (func(x_pos) - func(x)) / delta_x
    return grad

# Nesterov's Accelerated Gradient Descent
def nesterov_descent(func, L, dimension, init_x=None, epsilon=1e-6):
    x = np.zeros(dimension) if init_x is None else init_x
    lambda_prev = 0
    lambda_curr = 1
    gamma = 1
    y_prev = x
    alpha = 0.05 / (2 * L)
    gradient = calc_numerical_gradient(func, x)
    x_path = [x.copy()]
    
    while np.linalg.norm(gradient) >= epsilon:
        y_curr = x - alpha * gradient
        x = (1 - gamma) * y_curr + gamma * y_prev
        y_prev = y_curr
        lambda_tmp = lambda_curr
        lambda_curr = (1 + np.sqrt(1 + 4 * lambda_prev**2)) / 2
        lambda_prev = lambda_tmp
        gamma = (1 - lambda_prev) / lambda_curr
        gradient = calc_numerical_gradient(func, x)
        x_path.append(x.copy())

    return x, x_path

# Gradient Descent
def gradient_descent(func, alpha, init_x, epsilon=1e-6):
    x = init_x
    gradient = calc_numerical_gradient(func, x)
    x_path = [x.copy()]
    
    while np.linalg.norm(gradient) >= epsilon:
        x -= alpha * gradient
        gradient = calc_numerical_gradient(func, x)
        x_path.append(x.copy())

    return x, x_path

# Plotting functions
def plot_function_3d():
    x1, x2 = np.linspace(-10, 10, 400), np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
    ax.set_title('3D Plot of f(x1, x2)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.show()

def plot_topology(func, path1, path2, title1, title2):
    x, y = np.linspace(-6, 6, 400), np.linspace(-6, 6, 400)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, path, title in zip(axes, [path1, path2], [title1, title2]):
        ax.contourf(X, Y, Z, levels=10, cmap='viridis')
        ax.plot(np.array(path)[:, 0], np.array(path)[:, 1], 'ro-', label='Path')
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
    plt.show()

# Parameters for optimization
L, dimension = 1, 2
init_x = np.array([3.0, 3.0])

# Perform optimizations
nesterov_x, nesterov_path = nesterov_descent(deep_groove_function, L, dimension, init_x)
gradient_x, gradient_path = gradient_descent(deep_groove_function, alpha=0.01, init_x=init_x)

# Plot the topology and paths
plot_topology(deep_groove_function, gradient_path, nesterov_path, "Gradient Descent Path", "Nesterov's Method Path")

# Plot the 3D surface
plot_function_3d()