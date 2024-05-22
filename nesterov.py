import numpy as np
import matplotlib.pyplot as plt

def calc_numerical_gradient(func, x, delta_x=1e-5):
    """Function for computing gradient numerically."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_pos = np.copy(x)
        x_pos[i] += delta_x
        grad[i] = (func(x_pos) - func(x)) / delta_x
    return grad

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

def deep_groove_function(x):
    # return 0.1 * x[0]**2 + x[1]**2
    return 0.1 * x[0]**2 + x[1]**2

def plot_topology(func, path1, path2, title1, title2, iter1, iter2):
    x = np.linspace(-6, 6, 400)
    y = np.linspace(-6, 6, 400)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    path1 = np.array(path1)
    plt.plot(path1[:, 0], path1[:, 1], 'ro-', label='Path')
    plt.title(f"{title1}\nIterations: {iter1}")
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    path2 = np.array(path2)
    plt.plot(path2[:, 0], path2[:, 1], 'ro-', label='Path')
    plt.title(f"{title2}\nIterations: {iter2}")
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()

# Parameters
L = 1
dimension = 2
init_x = np.array([5.0, 5.0])

# Nesterov's method
nesterov_x, nesterov_path, nesterov_iters = nesterov_descent(deep_groove_function, L, dimension, init_x=init_x, epsilon=1e-6)

# Regular gradient descent
alpha = 0.01
gradient_x, gradient_path, gradient_iters = gradient_descent(deep_groove_function, alpha, init_x, epsilon=1e-6)

# Plot the topology and paths
plot_topology(deep_groove_function, gradient_path, nesterov_path, "Gradient Descent Path", "Nesterov's Method Path", gradient_iters, nesterov_iters)