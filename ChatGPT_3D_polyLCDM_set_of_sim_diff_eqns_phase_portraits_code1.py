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

def solve_and_plot_phase_portraits(variable_ranges, t_span, t_eval):
    """
    Solves the system for multiple values of each variable and plots phase portraits.
    """
    variable_names = ['$m$', '$r$', '$x$', '$y$', '$z$', '$v$', '$\\Lambda$']
    num_vars = len(variable_ranges)

    plt.figure(figsize=(20, 20))
    plot_idx = 1

    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            plt.subplot(num_vars - 1, num_vars - 1, plot_idx)
            for m in variable_ranges[0]:
                for r in variable_ranges[1]:
                    for x in variable_ranges[2]:
                        for y_var in variable_ranges[3]:
                            for z in variable_ranges[4]:
                                for v in variable_ranges[5]:
                                    for Lambda in variable_ranges[6]:
                                        initial_conditions = [m, r, x, y_var, z, v, Lambda]
                                        solution = solve_ivp(friedman_system, t_span, initial_conditions, t_eval=t_eval, method='RK45')
                                        variables = solution.y
                                        plt.plot(variables[i], variables[j], lw=0.5)
            plt.xlabel(variable_names[i], fontsize=12)
            plt.ylabel(variable_names[j], fontsize=12)
            plt.grid()
            plot_idx += 1

    plt.suptitle('Phase Portraits of the Variables for Multiple Values', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Define ranges for each variable
m_range = np.linspace(0, 1, 5)
r_range = np.linspace(0, 1, 5)
x_range = np.linspace(0, 1, 5)
y_range = np.linspace(0, 1, 5)
z_range = np.linspace(0, 1, 5)
v_range = np.linspace(0, 1, 5)
Lambda_range = np.linspace(0, 1, 5)

variable_ranges = [m_range, r_range, x_range, y_range, z_range, v_range, Lambda_range]

# Time span and evaluation points
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Plot phase portraits
solve_and_plot_phase_portraits(variable_ranges, t_span, t_eval)