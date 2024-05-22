import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_function_3d():
    # Define the function
    # def f(x1, x2):
    #     return 0.1 * x1**2 + x2**2
    def f(w, theta_b):
        return -np.exp(-((w**2) / 8) - ((theta_b**2) / 1))

    # Generate grid of x1 and x2 values
    x1 = np.linspace(-10, 10, 400)
    x2 = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')

    # Set labels and title
    ax.set_title('3D Plot of f(x1, x2) = 0.1x1^2 + x2^2')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')

    # Display the plot
    plt.show()

# Numerical gradient calculation
def calc_numerical_gradient(func, x, delta_x=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_pos = np.copy(x)
        x_pos[i] += delta_x
        grad[i] = (func(x_pos) - func(x)) / delta_x
    return grad

# Nesterov's Accelerated Gradient Descent
def nesterov_descent(func, L, dimension, init_x=None, numerical_gradient=True, delta_x=0.0005, gradient_func=None, epsilon=1e-6):
    if init_x is None:
        x = np.zeros(dimension)
    else:
        x = init_x

    lambda_prev = 0
    lambda_curr = 1
    gamma = 1
    y_prev = x
    alpha = 0.05 / (2 * L)

    if numerical_gradient:
        gradient = calc_numerical_gradient(func, x, delta_x)
    else:
        gradient = gradient_func(x)

    x_path = [x.copy()]
    iterations = 0
    
    while np.linalg.norm(gradient) >= epsilon:
        y_curr = x - alpha * gradient
        x = (1 - gamma) * y_curr + gamma * y_prev
        y_prev = y_curr

        lambda_tmp = lambda_curr
        lambda_curr = (1 + np.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
        lambda_prev = lambda_tmp

        gamma = (1 - lambda_prev) / lambda_curr

        if numerical_gradient:
            gradient = calc_numerical_gradient(func, x, delta_x)
        else:
            gradient = gradient_func(x)
        
        x_path.append(x.copy())
        iterations += 1

    return x, x_path, iterations

# Gradient Descent
def gradient_descent(func, alpha, init_x, epsilon=1e-6, numerical_gradient=True, delta_x=1e-5, gradient_func=None):
    x = init_x
    if numerical_gradient:
        gradient = calc_numerical_gradient(func, x, delta_x)
    else:
        gradient = gradient_func(x)
    
    x_path = [x.copy()]
    iterations = 0
    
    while np.linalg.norm(gradient) >= epsilon:
        x = x - alpha * gradient

        if numerical_gradient:
            gradient = calc_numerical_gradient(func, x, delta_x)
        else:
            gradient = gradient_func(x)
        
        x_path.append(x.copy())
        iterations += 1

    return x, x_path, iterations

# Deep Groove Function
def deep_groove_function(x):
    return -np.exp(-((x[0]**2) / 8) - ((x[1]**2) / 1))

# Plotting function
def plot_topology(func, path1, path2, title1, title2, iter1, iter2):
    x = np.linspace(-6, 6, 400)
    y = np.linspace(-6, 6, 400)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, Z, levels=10, cmap='viridis')
    plt.colorbar()
    path1 = np.array(path1)
    plt.plot(path1[:, 0], path1[:, 1], 'ro-', label='Path')
    plt.title(f"{title1}\nIterations: {iter1}")
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, Z, levels=10, cmap='viridis')
    plt.colorbar()
    path2 = np.array(path2)
    plt.plot(path2[:, 0], path2[:, 1], 'ro-', label='Path')
    plt.title(f"{title2}\nIterations: {iter2}")
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()

# Parameters for optimization
L = 1
dimension = 2
init_x = np.array([3.0, 3.0])

# Nesterov's method
nesterov_x, nesterov_path, nesterov_iters = nesterov_descent(deep_groove_function, L, dimension, init_x=init_x, epsilon=1e-6)

# Regular gradient descent
alpha = 0.01
gradient_x, gradient_path, gradient_iters = gradient_descent(deep_groove_function, alpha, init_x, epsilon=1e-6)

# Plot the topology and paths
plot_topology(deep_groove_function, gradient_path, nesterov_path, "Gradient Descent Path", "Nesterov's Method Path", gradient_iters, nesterov_iters)

# Plot the 3D surface
plot_function_3d()