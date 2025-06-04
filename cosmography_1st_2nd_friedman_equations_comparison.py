import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def hubble_parameter(a, Omega_m0, Omega_r0, H0):
    """Compute the Hubble parameter H(a) as a function of scale factor a."""
    Omega_L0 = 1 - Omega_m0 - Omega_r0
    return H0 * np.sqrt(Omega_m0 * a**-3 + Omega_r0 * a**-4 + Omega_L0)

def first_friedmann_integral(a, a0, Omega_m0, Omega_r0, H0):
    """Numerically integrate the first Friedmann equation."""
    integrand = lambda a: 1 / (a * hubble_parameter(a, Omega_m0, Omega_r0, H0))
    result, _ = quad(integrand, a0, a)
    return result

def second_friedmann_integral(a, a0, Omega_m0, Omega_r0, H0):
    """Numerically integrate the second Friedmann equation."""
    def exponential_inner_integral(a_inner):
        numerator = 3 * Omega_m0 * a_inner**-3 + 4 * Omega_r0 * a_inner**-4
        denominator = Omega_m0 * a_inner**-3 + Omega_r0 * a_inner**-4 + (1 - Omega_m0 - Omega_r0)
        return numerator / denominator

    def integrand(a):
        inner_integral, _ = quad(exponential_inner_integral, a0, a)
        return np.exp(0.5 * inner_integral) / a / H0 #*a**1.5

    result, _ = quad(integrand, a0, a) 
    return result

def analyze_differences(a_vals, a0, Omega_m0, Omega_r0, H0):
    """Analyze and plot the differences between the two integrals."""
    t_first  = []
    t_second = []

    for a in a_vals:
        t_first.append(first_friedmann_integral(a, a0, Omega_m0, Omega_r0, H0))
        t_second.append(second_friedmann_integral(a, a0, Omega_m0, Omega_r0, H0))

    t_first = np.array(t_first)
    t_second = np.array(t_second)
    
    # Calculate and plot the differences
    differences = np.abs(t_first - t_second)
    ratios      = t_first/t_second

    plt.ion()
    plt.figure(1,figsize=(12, 6))

    # Plot t(a) for both integrals
    plt.subplot(1, 2, 1)
    plt.plot(a_vals, t_first, label="First Friedmann Integral", lw=2)
    plt.plot(a_vals, t_second, label="Second Friedmann Integral", lw=2, linestyle="--")
    plt.xlabel("Scale Factor (a)")
    plt.ylabel("Time (t - t0)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(r"Time Evolution from Friedmann Equations, $T_1={:.1e}$ ".format(t_first[-1]))
   	#$T^{universe}_{2nd\ FE}=\num{t_second[-1]}$")
    
    plt.legend()
    plt.grid(True)

    # Plot differences
    plt.subplot(1, 2, 2)
    plt.plot(a_vals, ratios, label="Ratios", color="red", lw=2)
    plt.xlabel("Scale Factor (a)")
    plt.ylabel("$t_{first} / t_{second}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Ratios between Integrals")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Parameters
a0 = 1e-4
H0 = 70 / 3.086e19  # Hubble constant in s^-1
Omega_m0 = 0.3
Omega_r0 = 8.4e-5
a_vals = np.logspace(np.log10(a0), 0, 500)

# Analyze differences
analyze_differences(a_vals, a0, Omega_m0, Omega_r0, H0)
