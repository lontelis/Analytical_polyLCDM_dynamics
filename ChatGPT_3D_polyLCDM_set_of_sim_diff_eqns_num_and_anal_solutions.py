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

# Initial conditions
m0 = 0.01

x0 = 1e-11
y0 = 2e-8
z0 = 5e-9
v0 = 0.5e-13
Lambda0 =  3e-16 #1 - m0 - r0 - x0 - y0 - z0 - v0
r0 = 1- m0 - x0 - y0 - z0 - v0 - Lambda0

# Initial conditions
m0 = 0.3
r0 = 1e-4#1e-4
x0 = 0.01e-4#0.01e-4
y0 = 0.02e-4#0.02e-4
z0 = 0.05e-10#0.05e-5
v0 = 0.005e-4#0.005e-4
Lambda0 = 1 - m0 - r0 - x0 - y0 - z0 - v0

initial_conditions = [m0, r0, x0, y0, z0, v0, Lambda0]

# Time span
t_span = (0, -13)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the system
solution = solve_ivp(friedman_system, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Extract results
time = solution.t
m, r, x, y_var, z, v, Lambda = solution.y

# Plot results
plt.ion()
plt.figure(1,figsize=(12, 8))
plt.clf()
plt.plot(time, m, label='$m(t)$', lw=2)
plt.plot(time, r, label='$r(t)$', lw=2)
plt.plot(time, x, label='$x(t)$', lw=2)
plt.plot(time, y_var, label='$y(t)$', lw=2)
plt.plot(time, z, label='$z(t)$', lw=2)
plt.plot(time, v, label='$v(t)$', lw=2)
plt.plot(time, Lambda, label='$\Lambda(t)$', lw=2)

plt.title('Solutions to the Friedman Equations and Continuity Equations', fontsize=16)
plt.xlabel('Time $t$', fontsize=14)
plt.ylabel('Values of $m, r, x, y, z, v, \Lambda$', fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.show()
plt.savefig('./savefigs/ChatGPT_3D_polyLCDM_set_of_sim_diff_eqns_num_and_anal_solutions_instance.pdf')

# Phase portraits
variables = [m, r, x, y_var, z, v, Lambda]
variable_names = ['$m$', '$r$', '$x$', '$y$', '$z$', '$v$', '$\\Lambda$']
num_vars = len(variables)

plt.figure(2,figsize=(16, 12))
plt.clf()
plot_idx = 1

for i in range(num_vars):
    for j in range(i + 1, num_vars):
        plt.subplot(num_vars - 1, num_vars - 1, plot_idx)
        plt.plot(variables[i], variables[j], lw=1.5)
        plt.xlabel(variable_names[i], fontsize=12)
        plt.ylabel(variable_names[j], fontsize=12)
        plt.grid()
        plot_idx += 1

plt.suptitle('Phase Portraits of the Variables', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
plt.savefig('./savefigs/ChatGPT_3D_polyLCDM_set_of_sim_diff_eqns_num_and_anal_solutions_instance_phase_portrait.pdf')