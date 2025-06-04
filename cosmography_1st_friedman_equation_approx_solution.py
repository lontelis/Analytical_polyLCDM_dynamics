import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Define constants
H0 = 70  # Hubble constant in km/s/Mpc
H0 = 70 / 3.086e19  # Hubble constant in s^-1
Omega_m0 = 0.3  # Present-day matter density parameter
Omega_r0 = 1e-4  # Present-day radiation density parameter
Omega_Lambda0 = 1 - Omega_m0 - Omega_r0  # Present-day dark energy density parameter

# Define the integrand
def integrand(a):
    return 1 / (a * np.sqrt(Omega_m0 * a**-3 + Omega_r0 * a**-4 + (1 - Omega_m0 - Omega_r0)))

# Define the range of 'a' values
a0 = 1e-4  # Initial scale factor
a_values = np.logspace(np.log10(a0), 0, 1000)  # From a0 to 1 (present day)

# Perform the numerical integration
integral_values = [integrate.quad(integrand, a0, a)[0] for a in a_values]

# Convert integral values to time
t_values_1st_Friedman_eqn = np.array(integral_values) / H0  # in Hubble time units

# Plot the results
plt.ion()
plt.figure(1,figsize=(10, 6))
plt.clf()
plt.plot(a_values, t_values_1st_Friedman_eqn)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Scale factor (a)')
#plt.ylabel('Time (in Hubble time units)')
plt.ylabel('Time (in sec)')
plt.title('Cosmic Time vs Scale Factor')
plt.grid(True)
plt.show()
plt.savefig('./savefigs/cosmography_1st_friedman_equation_approx_solution.pdf')
