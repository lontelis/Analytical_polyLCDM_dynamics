"""
Author: Pierros Ntelis, 16 January 2025
"""

print("""
########## ########## ########## ########## ##########  
########## Solving and ploting 
########## the solutions of 
########## 6D system of simultaneous differential equations for polyLCDM
########## ########## ########## ########## ########## 
""")
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define alpha
alpha = 0.5

def friedman_system(t, y):
    """
    Defines the system of differential equations.
    """
    m, r, x, y_var, z, v, Lambda = y

    # First Friedman equation
    f = 3 + r - 2 * x - y_var + 2 * z + (alpha - 3) * v - 3 * Lambda

    # Continuity equations
    m_prime = m * (f - 3)
    r_prime = r * (f - 4)
    x_prime = x * (f - 1)
    y_prime = y_var * (f - 2)
    z_prime = z * (f - 5)
    v_prime = v * (f - alpha)
    Lambda_prime = Lambda * f

    return [m_prime, r_prime, x_prime, y_prime, z_prime, v_prime, Lambda_prime]

def plot_phase_portraits_with_vectors(variable_ranges, t_span):
    """
    Plots phase portraits with vectors for multiple variable pairs.
    """
    variable_names = ['$m$', '$r$', '$x$', '$y$', '$z$', '$v$', '$\\Lambda$']
    num_vars = len(variable_ranges)

    plt.figure(figsize=(20, 20))
    plot_idx = 1

    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            plt.subplot(num_vars - 1, num_vars - 1, plot_idx)

            # Create a grid for variables i and j
            grid_i, grid_j = np.meshgrid(variable_ranges[i], variable_ranges[j])
            
            # Initialize vectors for the derivatives
            di = np.zeros_like(grid_i)
            dj = np.zeros_like(grid_j)

            # Calculate derivatives at each grid point
            for idx in range(grid_i.shape[0]):
                for jdx in range(grid_i.shape[1]):
                    initial_conditions = [0] * num_vars
                    initial_conditions[i] = grid_i[idx, jdx]
                    initial_conditions[j] = grid_j[idx, jdx]

                    derivatives = friedman_system(0, initial_conditions)
                    di[idx, jdx] = derivatives[i]
                    dj[idx, jdx] = derivatives[j]

            # Normalize vectors for better visualization
            magnitude = np.sqrt(di**2 + dj**2)
            di /= magnitude
            dj /= magnitude

            # Plot vector field using quiver
            plt.quiver(grid_i, grid_j, di, dj, color='blue', alpha=0.6)
            plt.xlabel(variable_names[i], fontsize=12)
            plt.ylabel(variable_names[j], fontsize=12)
            plt.grid()
            plot_idx += 1

    plt.suptitle('Phase Portraits with Vector Fields', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Define ranges for each variable
m_range = np.linspace(0, 1, 10)
r_range = np.linspace(0, 1, 10)
x_range = np.linspace(0, 1, 10)
y_range = np.linspace(0, 1, 10)
z_range = np.linspace(0, 1, 10)
v_range = np.linspace(0, 1, 10)
Lambda_range = np.linspace(0, 1, 10)

variable_ranges = [m_range, r_range, x_range, y_range, z_range, v_range, Lambda_range]

# Time span (not used in this version but kept for consistency)
t_span = (-13, 1)

# Plot phase portraits with vectors
plot_phase_portraits_with_vectors(variable_ranges, t_span)