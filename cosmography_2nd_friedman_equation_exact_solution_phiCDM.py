import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def hubble_integrand(a, Omega_m0, Omega_x0, Omega_r0, Omega_v0):
    """Integrand of the phi-CDM model."""
    term_m = Omega_m0 * a**-3
    term_x = Omega_x0 * a**-3 * (1 - 3 * np.log(a))
    term_r = Omega_r0 * a**-4
    term_v = Omega_v0
    return 1 / (a * np.sqrt(term_m + term_x + term_r + term_v))

def phi_cdm_integral(a, a0, Omega_m0, Omega_x0, Omega_r0, Omega_v0, H0):
    """Numerically integrate the phi-CDM model."""
    result, _ = quad(hubble_integrand, a0, a, args=(Omega_m0, Omega_x0, Omega_r0, Omega_v0))
    return result / H0

def plot_phi_cdm(a_vals, a0, Omega_m0, Omega_x0, Omega_r0, Omega_v0, H0):
    """Compute and plot the evolution of time as a function of scale factor."""
    times = []
    for a in a_vals:
        times.append(phi_cdm_integral(a, a0, Omega_m0, Omega_x0, Omega_r0, Omega_v0, H0))

    times = np.array(times)

    # Plot the result
    plt.ion()
    plt.figure(2,figsize=(8, 6))
    plt.clf()
    plt.plot(a_vals, times, label="$t(a)$ from phi-CDM", color="blue", lw=2)
    plt.xlabel("Scale Factor (a)")
    plt.ylabel("$t - t_0$ (in sec)")
    plt.xscale('log')
    plt.yscale('log')    
    plt.title("Evolution of Time in the phi-CDM Model")
    plt.grid(True)
    plt.legend()
    plt.show()
    return(times)

# Parameters
a0 = 1e-4
H0 = 70 / 3.086e19  # Hubble constant in s^-1
Omega_m0 = 0.3
Omega_x0 = 0.1
Omega_r0 = 8.4e-5
Omega_v0 = 1 - Omega_m0 - Omega_x0 - Omega_r0

a_vals = np.logspace(np.log10(a0), 0, 1000)

# Plot the phi-CDM time evolution
t_values_2nd_Friedman_eqn_phiCDM = plot_phi_cdm(a_vals, a0, Omega_m0, Omega_x0, Omega_r0, Omega_v0, H0)
plt.savefig('./savefigs/cosmography_2nd_friedman_equation_exact_solution_phiCDM.pdf')

"""
# Plot the results
plt.ion()
plt.figure(3,figsize=(10, 6))
plt.clf()
plt.plot(a_values, t_values_2nd_Friedman_eqn_phiCDM/t_values_1st_Friedman_eqn_phiCDM, label='Ratio $t_{2nd F E}/t_{1st F E}$')
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Scale factor (a)')
#plt.ylabel('Time Ratio (in Hubble time units)')
plt.ylabel('Time Ratio (in sec)')
plt.title('Cosmic Time Ratio vs Scale Factor')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('./savefigs/cosmography_2nd_friedman_equation_approx_solution_comparison_phiCDM.pdf')
"""

"""
# Plot the results for comparing phiCDM and LCDM
plt.ion()
plt.figure(4,figsize=(10, 6))
plt.clf()
plt.plot(a_values, t_values_2nd_Friedman_eqn_phiCDM/t_values_1st_Friedman_eqn, label='Ratio $t_{2nd F E, \phi CDM}/t_{1st F E, \Lambda CDM}$')
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Scale factor (a)')
#plt.ylabel('Time Ratio (in Hubble time units)')
plt.ylabel('Time Ratio (in sec)')
plt.title('Cosmic Time Ratio vs Scale Factor, $\phi$CDM$/\Lambda$CDM')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('./savefigs/cosmography_2nd_friedman_equation_approx_solution_comparison_phiCDM.pdf')
"""
