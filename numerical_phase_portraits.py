'''
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.integrate import solve_ivp

# Define alpha
alpha = 0.5

def friedman_system(t, y):
    """
    Defines the system of differential equations for the PolyΛCDM model.
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

# Initial conditions
m0 = 0#0.3
r0 = 0#1e-4
x0 = 0#0.01e-4
y0 = 0#0.02e-4
z0 = 0#0.05e-10
v0 = 0#0.005e-4
Lambda0 = 1 - m0 - r0 - x0 - y0 - z0 - v0
initial_conditions = [m0, r0, x0, y0, z0, v0, Lambda0]

# Define the variable labels for plotting
variables = ['\Omega_m', '\Omega_r', '\Omega_x', '\Omega_y', '\Omega_z', '\Omega_v', '\Omega_\Lambda']
fixed_value = 0.0  # Fixed value for the last variable

# Create grid for phase portraits
grid_size = 20
var_min, var_max = 0, 1  # Define the range of the grid
grid = np.linspace(var_min, var_max, grid_size)

# Function to compute vector field for given variable pair
def compute_vector_field_all_together(pair_indices, fixed_idx, fixed_val):
    """
    Compute the vector field for a pair of variables while fixing another variable.
    """
    var1_idx, var2_idx = pair_indices
    fixed_conditions = initial_conditions[:]
    fixed_conditions[fixed_idx] = fixed_val
    print('fixed_conditions='+str(fixed_conditions))
    # Mesh grid for the variable pair
    var1_vals, var2_vals = np.meshgrid(grid, grid)

    # Compute derivatives
    var1_dot = np.zeros_like(var1_vals)
    var2_dot = np.zeros_like(var2_vals)
    for i in range(grid_size):
        for j in range(grid_size):
            conditions = fixed_conditions[:]
            conditions[var1_idx] = var1_vals[i, j]
            conditions[var2_idx] = var2_vals[i, j]
            derivatives = friedman_system(0, conditions)
            var1_dot[i, j] = derivatives[var1_idx]
            var2_dot[i, j] = derivatives[var2_idx]

    return var1_vals, var2_vals, var1_dot, var2_dot

# Plot phase portraits
plt.ion()
plt.figure(1,figsize=(16, 12))
pairs = list(combinations(range(len(variables)), 2))  # All pairs of variables
num_plots = len(pairs)

for idx, pair in enumerate(pairs, 1):
    plt.subplot(4, 6, idx)  # Adjust subplot layout for all pairs
    var1_idx, var2_idx = pair
    var1_vals, var2_vals, var1_dot, var2_dot = compute_vector_field_all_together(pair, -1, fixed_value)

    plt.quiver(var1_vals, var2_vals, var1_dot, var2_dot, color='blue', pivot='mid')
    plt.xlabel(f'${variables[var1_idx]}$', fontsize=10)
    plt.ylabel(f'${variables[var2_idx]}$', fontsize=10)
    plt.title(f'${variables[var1_idx]}$ vs ${variables[var2_idx]}$', fontsize=12)
    plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('./savefigs/numerical_phase_portraits.pdf')
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.integrate import solve_ivp

# Define alpha
alpha = 0.5

def friedman_system(t, y):
    """
    Defines the system of differential equations for the PolyΛCDM model.
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

# Define critical points
critical_points = {
    "A_m": (1, 0, 0, 0, 0, 0, 0),
    "R_r": (0, 1, 0, 0, 0, 0, 0),
    "A_x": (0, 0, 1, 0, 0, 0, 0),
    "A_k": (0, 0, 0, 1, 0, 0, 0),
    "R_z": (0, 0, 0, 0, 1, 0, 0),
    "A_v": (0, 0, 0, 0, 0, 1, 0),
    "A_\Lambda": (0, 0, 0, 0, 0, 0, 1),
}

# Initial conditions (set all to 0 for phase portraits)
variables = ['\Omega_m', '\Omega_r', '\Omega_x', '\Omega_k', '\Omega_z', '\Omega_v', '\Omega_\Lambda']

# Grid settings
grid_size = 20
var_min, var_max = 0, 1
grid = np.linspace(var_min, var_max, grid_size)


#########################################################################################################
# Initial conditions
m0 = 0#0.3
r0 = 0#1e-4
x0 = 0#0.01e-4
y0 = 0#0.02e-4
z0 = 0#0.05e-10
v0 = 0#0.005e-4
Lambda0 = 1 - m0 - r0 - x0 - y0 - z0 - v0
initial_conditions = [m0, r0, x0, y0, z0, v0, Lambda0]
fixed_value = 0.0  # Fixed value for the last variable
# Function to compute vector field for given variable pair
def compute_vector_field_all_together(pair_indices, fixed_idx, fixed_val):
    """
    Compute the vector field for a pair of variables while fixing another variable.
    """
    var1_idx, var2_idx = pair_indices
    fixed_conditions = initial_conditions[:]
    fixed_conditions[fixed_idx] = fixed_val
    print('fixed_conditions='+str(fixed_conditions))
    # Mesh grid for the variable pair
    var1_vals, var2_vals = np.meshgrid(grid, grid)

    # Compute derivatives
    var1_dot = np.zeros_like(var1_vals)
    var2_dot = np.zeros_like(var2_vals)
    for i in range(grid_size):
        for j in range(grid_size):
            conditions = fixed_conditions[:]
            conditions[var1_idx] = var1_vals[i, j]
            conditions[var2_idx] = var2_vals[i, j]
            derivatives = friedman_system(0, conditions)
            var1_dot[i, j] = derivatives[var1_idx]
            var2_dot[i, j] = derivatives[var2_idx]

    return var1_vals, var2_vals, var1_dot, var2_dot

# Plot phase portraits
plt.ion()
plt.figure(1,figsize=(16, 12))
plt.clf()
pairs = list(combinations(range(len(variables)), 2))  # All pairs of variables
num_plots = len(pairs)

for idx, pair in enumerate(pairs, 1):
    plt.subplot(4, 6, idx)  # Adjust subplot layout for all pairs
    var1_idx, var2_idx = pair
    var1_vals, var2_vals, var1_dot, var2_dot = compute_vector_field_all_together(pair, -1, fixed_value)

    plt.quiver(var1_vals, var2_vals, var1_dot, var2_dot, color='blue', pivot='mid')
    plt.xlabel(f'${variables[var1_idx]}$', fontsize=10)
    plt.ylabel(f'${variables[var2_idx]}$', fontsize=10)
    plt.title(f'${variables[var1_idx]}$ vs ${variables[var2_idx]}$', fontsize=12)
    plt.grid(True)
    # Plot critical points
    for name, point in critical_points.items():
        if point[var1_idx] == 1 or point[var2_idx] == 1:  # Only plot relevant critical points
            plt.scatter(point[var1_idx], point[var2_idx], color='red', label='$'+name+'$')
            plt.text(point[var1_idx], point[var2_idx], f" ${name}$", color='red', fontsize=10)    

plt.tight_layout()
plt.show()
plt.savefig('./savefigs/numerical_phase_portraits_alpha'+str(alpha)+'.pdf')
stop
#########################################################################################################


# Compute vector field in pairs
def compute_vector_field(var1_idx, var2_idx):
    """
    Compute the vector field for a pair of variables.
    """
    var1_vals, var2_vals = np.meshgrid(grid, grid)
    var1_dot = np.zeros_like(var1_vals)
    var2_dot = np.zeros_like(var2_vals)

    for i in range(grid_size):
        for j in range(grid_size):
            state = np.zeros(len(variables))
            state[var1_idx] = var1_vals[i, j]
            state[var2_idx] = var2_vals[i, j]
            derivatives = friedman_system(0, state)
            var1_dot[i, j] = derivatives[var1_idx]
            var2_dot[i, j] = derivatives[var2_idx]

    return var1_vals, var2_vals, var1_dot, var2_dot

# Plot phase portraits for each pair of variables
pairs = list(combinations(range(len(variables)), 2))
output_folder = "./savefigs"


for var1_idx, var2_idx in pairs:
    var1_vals, var2_vals, var1_dot, var2_dot = compute_vector_field(var1_idx, var2_idx)
    
    plt.figure(figsize=(8, 6))
    plt.quiver(var1_vals, var2_vals, var1_dot, var2_dot, color='blue', pivot='mid')
    
    # Plot critical points
    for name, point in critical_points.items():
        if point[var1_idx] == 1 or point[var2_idx] == 1:  # Only plot relevant critical points
            plt.scatter(point[var1_idx], point[var2_idx], color='red', label='$'+name+'$')
            plt.text(point[var1_idx], point[var2_idx], f" ${name}$", color='red', fontsize=10)
    
    # Set labels and title
    plt.xlabel(f"${variables[var1_idx]}$", fontsize=14)
    plt.ylabel(f"${variables[var2_idx]}$", fontsize=14)
    plt.title(f"Phase Portrait: ${variables[var1_idx]}$ vs ${variables[var2_idx]}$", fontsize=16)
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    filename = f"{output_folder}/numerical_phase_portraits_{variables[var1_idx]}_vs_{variables[var2_idx]}.pdf"
    plt.savefig(filename)
    plt.close()
