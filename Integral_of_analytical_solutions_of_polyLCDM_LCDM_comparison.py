import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
H0 = 70  # Hubble constant (km/s/Mpc)
H0 = 70 / 3.086e19  # Hubble constant in s^-1
m0 = 0.3
r0 = 1e-4
x0 = 0.01e-4
y0 = 0.02e-4
z0 = 0.05e-10
v0 = 0.005e-4
Lambda0 = 1 - m0 - r0 - x0 - y0 - z0 - v0
alpha =0.5  # Chosen for example purposes
a0 = 1e-4   # Initial scale factor
a0 = 1e-6  # Initial scale factor
#a0 = np.exp(-12.)  # Initial scale factor 
#a0 = 1e-7  # Initial scale factor

# Functions to calculate the integrands
def integrand_polyLambdaCDM(a):
    denominator = (
        m0 * a**-3 +
        r0 * a**-4 +
        x0 * a**-1 +
        y0 * a**-2 +
        z0 * a**-5 +
        v0 * a**-alpha +
        Lambda0
    )**0.5
    return 1 / (a * denominator)

def integrand_LambdaCDM(a):
    denominator = (
        m0 * a**-3 +
        r0 * a**-4 +
        (1 - m0 - r0)
    )**0.5
    return 1 / (a * denominator)

# Numerical integration for PolyΛCDM and ΛCDM
a_values = np.logspace(np.log10(a0), 0, 500)  # Scale factors from a0 to 1
t_polyLambdaCDM = []
t_LambdaCDM = []

for a in a_values:
    result_poly, _ = quad(integrand_polyLambdaCDM, a0, a)
    result_Lambda, _ = quad(integrand_LambdaCDM, a0, a)
    t_polyLambdaCDM.append(result_poly / H0)
    t_LambdaCDM.append(result_Lambda / H0)

# Calculate the ratio
ratio = np.array(t_polyLambdaCDM) / np.array(t_LambdaCDM)

# Plotting the results
plt.ion()
plt.figure(1,figsize=(14, 6))

# Plot t vs a for both models
plt.subplot(1, 2, 1)
plt.plot(a_values, t_polyLambdaCDM, label=r'Poly$\Lambda$CDM', lw=2)
plt.plot(a_values, t_LambdaCDM, label=r'$\Lambda$CDM', lw=2, linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Scale Factor $a$', fontsize=14)
plt.ylabel(r'Cosmic Time $t$ ($sec$)', fontsize=14)
plt.title(r'$t$ vs $a$ for Poly$\Lambda$CDM and $\Lambda$CDM', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot the ratio
plt.subplot(1, 2, 2)
plt.plot(a_values, ratio, color='purple', lw=2)
plt.xscale('log')
plt.xlabel(r'Scale Factor $a$', fontsize=14)
plt.ylabel(r'Ratio $t_{\rm Poly\Lambda CDM} / t_{\rm \Lambda CDM}$', fontsize=14)
plt.title(r'Ratio of $t$ between Poly$\Lambda$CDM and $\Lambda$CDM', fontsize=16)
plt.axhline(1, color='gray', linestyle='--', linewidth=1)  # Reference line at ratio = 1
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
plt.savefig('./savefigs/Integral_of_analytical_solutions_of_polyLCDM_LCDM_comparison.pdf')


######################

import numpy as np

def delta_t(a, H0, Omega_m0, Omega_r0, Omega_Lambda0):
    """
    Calculate Δt = t - t₀ as a function of a and cosmological parameters.
    
    Parameters:
    a (float or array): Scale factor
    H0 (float): Hubble constant in km/s/Mpc
    Omega_m0 (float): Present-day matter density parameter
    Omega_r0 (float): Present-day radiation density parameter
    Omega_Lambda0 (float): Present-day dark energy density parameter
    
    Returns:
    float or array: Δt in seconds
    """

    numerator = abs( 2 - 2 * np.sqrt(Omega_m0 * a**(-3) + Omega_r0 * a**(-4) + Omega_Lambda0) )
    denominator = 3 * Omega_m0 * a**(-3) + 4 * Omega_r0 * a**(-4)
    
    delta_t = (1 / H0) * (numerator / denominator)
    
    return delta_t

# Example usage:
"""
a = 1.0  # Present day
H0 = 70  # km/s/Mpc
Omega_m0 = 0.3
Omega_r0 = 9.24e-5  # Typical value for radiation
Omega_Lambda0 = 0.7
result = delta_t(a_values, H0, Omega_m0, Omega_r0, Omega_Lambda0)
#print(f"Δt = {result:.2e} seconds")
"""
analytical_t_vs_a_LCDM = delta_t(a_values, H0, m0, r0, 1 - m0- r0)


# Plotting the results
plt.ion()
plt.figure(2,figsize=(14, 6))

# Plot t vs a for both models
plt.subplot(1, 2, 1)
#plt.plot(a_values, t_polyLambdaCDM, label=r'Poly$\Lambda$CDM', lw=2)
plt.plot(a_values, t_LambdaCDM, label=r'$\Lambda$CDM', lw=2, linestyle='--')
plt.plot(a_values, analytical_t_vs_a_LCDM, label=r'Analytical $\Lambda$CDM', lw=2, linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Scale Factor $a$', fontsize=14)
plt.ylabel(r'Cosmic Time $t$ ($sec$)', fontsize=14)
plt.title(r'$t$ vs $a$ for Poly$\Lambda$CDM and $\Lambda$CDM', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot the ratio
plt.subplot(1, 2, 2)
#plt.plot(a_values, ratio, color='purple', lw=2)
plt.plot(a_values, analytical_t_vs_a_LCDM/t_LambdaCDM, color='red', lw=2)
plt.xscale('log')
plt.xlabel(r'Scale Factor $a$', fontsize=14)
plt.ylabel(r'Ratio $t_{\rm Poly\Lambda CDM} / t_{\rm \Lambda CDM}$', fontsize=14)
plt.title(r'Ratio of $t$ between Poly$\Lambda$CDM and $\Lambda$CDM', fontsize=16)
plt.axhline(1, color='gray', linestyle='--', linewidth=1)  # Reference line at ratio = 1
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
plt.savefig('./savefigs/Integral_of_analytical_solutions_of_polyLCDM_LCDM_comparison_analytical.pdf')
