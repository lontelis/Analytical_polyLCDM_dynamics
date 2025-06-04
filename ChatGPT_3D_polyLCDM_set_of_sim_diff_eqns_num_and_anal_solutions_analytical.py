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

# Initial conditions
m0 = 0.3
r0 = 1e-4#1e-4
x0 = 0.01e-4#0.01e-4
y0 = 0.02e-4#0.02e-4
z0 = 0.05e-10#0.05e-5
v0 = 0.005e-4#0.005e-4
Lambda0 = 1 - m0 - r0 - x0 - y0 - z0 - v0
alpha = 0.5  # Assuming alpha = 0.5 for v(t)

# Create N values
N = np.linspace(-13, 1, 1000)

# Calculate solutions
denominator = (m0 * np.exp(-3*N) + r0 * np.exp(-4*N) + x0 * np.exp(-N) + 
               y0 * np.exp(-2*N) + z0 * np.exp(-5*N) + v0 * np.exp(-alpha*N) + Lambda0)

m = m0 * np.exp(-3*N) / denominator
r = r0 * np.exp(-4*N) / denominator
x = x0 * np.exp(-N) / denominator
y = y0 * np.exp(-2*N) / denominator
z = z0 * np.exp(-5*N) / denominator
v = v0 * np.exp(-alpha*N) / denominator
Lambda = Lambda0 / denominator


w_eff = 1/3 * r  \
        - Lambda \
        -2/3 * x \
        -1/3 * y \
        + 2/3 * z \
        + ( alpha/3 - 1 ) * v

# Create the plot
plt.ion()
plt.figure(1,figsize=(11, 7))
plt.clf()
plt.plot(N, m, label='$\Omega_m(t) = \Omega_{m,0} a^{-3} \; \\left[ \\dots \\right]^{-1}$')
plt.plot(N, r, label='$\Omega_r(t) = \Omega_{r,0} a^{-4} \; \\left[ \\dots \\right]^{-1}$')
plt.plot(N, x, label='$\Omega_x(t) = \Omega_{x,0} a^{-1} \; \\left[ \\dots \\right]^{-1}$')
plt.plot(N, y, label='$\Omega_k(t) = \Omega_{k,0} a^{-2} \; \\left[ \\dots \\right]^{-1}$')
plt.plot(N, z, label='$\Omega_z(t) = \Omega_{z,0} a^{-5} \; \\left[ \\dots \\right]^{-1}$')
plt.plot(N, v, label='$\Omega_v(t) = \Omega_{v,0} a^{-\\alpha} \; \\left[ \\dots \\right]^{-1}$')
plt.plot(N, Lambda, label='$\Omega_Λ(t) = \Omega_{Λ,0} \; \\left[ \\dots \\right]^{-1}$')
plt.plot(N, abs(w_eff), 'm--', label='$|w_{\\rm eff}(t)| $')

plt.xlabel('Lapse time, $N=\ln a(t)$')
plt.ylabel('Energy density ratios, $\Omega_{s}(N)$')
plt.title('Evolution of Energy density ratios \
    \n $\\left[ \\dots \\right]  = \\left[ \Omega_{m,0}  a^{-3 }  \
    + \Omega_{r,0}  a^{-4  }  \
    + \Omega_{x,0}   a^{-1  }    \
    + \Omega_{k,0} a^{-2  }  \
    + \Omega_{z,0}  a^{-5 }   \
    + \Omega_{v,0}  a^{- \\alpha  }   \
     + \\Omega_{\Lambda,0}  \\right]$   ')
    #\n $H^{-1} \int_{a_0}^{a} \\frac{da}{a \\left[ \sum_{s \in \{ m,r,x,y,z,v,Λ\}} Ω_s(t) \\right]^{1/2} } = t - t_0$')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
plt.savefig('./savefigs/ChatGPT_3D_polyLCDM_set_of_sim_diff_eqns_num_and_anal_solutions_analytical_instance_log.pdf')

