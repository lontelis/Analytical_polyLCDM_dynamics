import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Define constants
H0 = 70  # Hubble constant in km/s/Mpc
Omega_m0 = 0.3  # Present-day matter energy density parameter
Omega_r0 = 1e-4  # Present-day radiation density parameter
Omega_x0 = 0.05  # Present-day kinetic energy density parameter
Omega_v0 = 1. - Omega_m0 - Omega_r0 - Omega_x0 # Present-day potential energy density parameter

# Define the inner integrand
def inner_integrand(a):
    numerator = (4 * Omega_r0 + 3 * Omega_x0 * a * (1 - 3 * np.log(a)) + 
                 3 * (1 - Omega_r0 - Omega_v0) * a)
    denominator = (Omega_r0 - 3 * a * np.log(a) * Omega_x0 + 
                   a**4 * Omega_v0 + 
                   (1 - Omega_r0 - Omega_v0) * a)
    return numerator / denominator

# Define the outer integrand
def outer_integrand(a, a0):
    inner_integral, _ = integrate.quad(lambda x: inner_integrand(x) / x, a0, a)
    return np.exp(0.5 * inner_integral) / a

# Function to calculate the integral for a given a
def calculate_integral(a, a0):
    result, _ = integrate.quad(lambda x: outer_integrand(x, a0), a0, a)
    return result

# Generate a range of 'a' values
a0=1e-4
a_values = np.logspace(np.log10(a0), 0, 100)  # From a=10^-4 to a=1

# Calculate the integral for each 'a' value
integral_values = [calculate_integral(a, a0) for a in a_values]

# Convert integral values to time
t_values_2nd_Friedman_eqn = np.array(integral_values) / H0  # in Hubble time units

# Plot the results
plt.ion()
plt.figure(2,figsize=(10, 6))
plt.clf()
plt.plot(a_values, t_values_2nd_Friedman_eqn)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Scale factor (a)')
plt.ylabel('Time (in Hubble time units)')
plt.title('Cosmic Time vs Scale Factor')
plt.grid(True)
plt.show()
plt.savefig('./savefigs/cosmography_2nd_friedman_equation_exact_solution_phiCDM_other_integral.pdf')

"""
# Plot the results
plt.ion()
plt.figure(3,figsize=(10, 6))
plt.clf()
plt.plot(a_values, t_values_2nd_Friedman_eqn/t_values_1st_Friedman_eqn, label='Ratio $t_{2nd F E}/t_{1st F E}$')
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Scale factor (a)')
plt.ylabel('Time Ratio (in Hubble time units)')
plt.title('Cosmic Time Ratio vs Scale Factor')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('./savefigs/cosmography_2nd_friedman_equation_approx_solution_comparison_phiCDM_other_integral.pdf')
"""