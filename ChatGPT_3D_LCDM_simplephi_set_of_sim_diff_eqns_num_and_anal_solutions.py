"""
Author: Pierros Ntelis, 4 December 2024
"""

print("""
########## ########## ########## ########## ##########  
########## Solving and ploting 
########## the solutions of 
########## 3D system of simultaneous differential equations for LCDM
########## we add also the constratin 1 = x + y + z, on all 3 equations
########## The solution x(N), y(N), z(N), 
########## functions of lapse time $N$
########## one initial conditions is chosen
########## with intersection points, to describe each equality epoch.
########## with the effective equation of state weff = 1/3\Omega_r - \Omega_Lambda
########## adding the intersection points.
########## ########## ########## ########## ########## 
""")


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N_min,N_max=-12,1.0
N_step_size=100
var_zero_is = 0 #1e-18#3e-18 # 1e-3

# Define the system of differential equations
def system(N, Z):
    x, y, z = Z
    f = 3*x + 4*y
    dx_dN = x*( f - 3 ) 
    dy_dN = y*( f - 4 ) 
    dz_dN = (1-x-y)*(3*x + 4*y)
    #dz_dN = z* f 
    return [dx_dN, dy_dN, dz_dN]

#Analytical solutions definitions
def system_analytical_LCDM(N, Omega_m0, Omega_r0, Omega_L0): 
    # Common denominator for all equations
    denominator = (np.exp(-3*N) * Omega_m0 + 
                   np.exp(-4*N) * Omega_r0 + 
                   Omega_L0)
    
    # Calculate each Omega component
    Omega_m_N = Omega_m0 * np.exp(-3*N) / denominator
    Omega_r_N = Omega_r0 * np.exp(-4*N) / denominator
    #Omega_L_N = 1 - Omega_m_N - Omega_r_N
    Omega_L_N = Omega_L0 / denominator

    return( Omega_m_N, Omega_r_N, Omega_L_N )


# Define the initial conditions and eta span
#initial_conditions = [0.01, 1.0-0.01, 0.0]
initial_conditions = [0.009, 1-0.009-var_zero_is, var_zero_is]

N_span = (N_min,N_max)
N = np.linspace(N_min,N_max, N_step_size)

# Solve the system for each initial condition
plt.ion()
fig, axes = plt.subplots(1, 1, figsize=(14, 7))
ax1 = axes
#axes = axes.flatten()

#sol = solve_ivp(system, N_span, initial_conditions, t_eval=N, dense_output=True)
sol = solve_ivp(system, N_span, initial_conditions, t_eval=N, dense_output=True, method='RK45',rtol=1e-5, atol=1e-7)
#method='RK45',rtol=1e-5, atol=1e-7)

print('# Diagnostics')
print("N length:", len(N))           # Should be 100
print("sol.t length:", len(sol.t))   # Should be 100 with t_eval=N
print("sol.y shape:", sol.y.shape)   # Should be (3, 100)
print("sol.success:", sol.success)   # True if integration succeeded
print("sol.message:", sol.message)   # Reason if it failed

N_values     = sol.t
Omega_m      = sol.y[0]
Omega_r      = sol.y[1]
Omega_Lambda = 1-sol.y[0]-sol.y[1]
weff_LCDM = 1/3*Omega_r-Omega_Lambda

#Analytical solutions calculations of LCDM
Omega_m0 = 0.32
Omega_r0 = 2e-4
Omega_L0 = 1 - Omega_m0 - Omega_r0
Omega_m_N_LCDM_anal, Omega_r_N_LCDM_anal, Omega_L_N_LCDM_anal = system_analytical_LCDM(N, Omega_m0, Omega_r0, Omega_L0)
weff_LCDM_anal = 1/3*Omega_r_N_LCDM_anal -Omega_L_N_LCDM_anal


#Numerical solutions plotting
ax1.plot(N_values, Omega_m, 'b.', label='$x(N)=\Omega_m(N)$')
ax1.plot(N_values, Omega_r, 'y.',label='$y(N)=\Omega_r(N)$')
ax1.plot(N_values, Omega_Lambda, 'g.',label='$z(N)=\Omega_\Lambda(N)$')
ax1.plot(N_values, weff_LCDM, 'm.', label='$w_{\\rm{eff}}(N)=\\frac{1}{3}\Omega_r(N)-\Omega_\Lambda(N)$')    

#Analytical solutions plotting
ax1.plot(N_values, Omega_m_N_LCDM_anal, 'b-', label='$\Omega_m^{anal}(N), \Omega_{m,0}=$%1.2f'%Omega_m0)
ax1.plot(N_values, Omega_r_N_LCDM_anal, 'y-',label='$\Omega_r^{anal}(N), \Omega_{r,0}=$%1.1e'%Omega_r0)
ax1.plot(N_values, Omega_L_N_LCDM_anal, 'g-',label='$\Omega_\Lambda^{anal}(N), \Omega_{\Lambda,0}=$%1.2f'%Omega_L0)
ax1.plot(N_values, weff_LCDM_anal, 'm-', label='$w_{\\rm{eff}}^{anal}(N)=\\frac{1}{3}\Omega_r^{anal}(N)-\Omega_\Lambda^{anal}(N)$')    

ax1.set_title(f'3D Solutions and Initial Condition: {initial_conditions}',fontsize=(15))
ax1.set_xlabel('$N = \\ln [a(t)]$',size=20)
ax1.set_ylabel('Solution',size=20)
ax1.set_yscale('linear')
ax1.set_ylim([-2,1.1])
ax1.legend(fontsize=(15),ncol=2)
ax1.grid()



# Add a secondary x-axis for the redshift z
def N_to_redshift(N):
    return (np.exp(-N) - 1)

def redshift_to_N(z):
    return -np.log(z + 1)

# Create a secondary x-axis
secax = ax1.secondary_xaxis('top', functions=(N_to_redshift, redshift_to_N))
secax.set_xlabel('Redshift $z$',size=15)

# Set the ticks on the secondary x-axis to match the primary axis at 3 significant figures
z_ticks = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
secax.set_xticks(z_ticks)
redshift_from_N_extra_ticks = np.round(N_to_redshift(np.array([-10, -8, -6, -4, -2])),0)
extra_ticks = list(redshift_from_N_extra_ticks)
secax.set_xticks( sorted(list(secax.get_xticks()) + extra_ticks)[::-1] )
secax.set_xticks( sorted(list(secax.get_xticks()) )[::-1] )

# Create labels in scientific notation
#labels = ['$0$', '$1 \\times 10^0$', '$1 \\times 10^1$', '$1 \\times 10^2$', '$1 \\times 10^3$', '$1 \\times 10^4$', '$1 \\times 10^5$', '$1 \\times 10^6$', '$1 \\times 10^7$', '$1 \\times 10^8$']
#labels = ['$8 \\times 10^5$', '$7 \\times 10^5$', '$6 \\times 10^5$', '$5 \\times 10^5$', '$4 \\times 10^5$', '$3 \\times 10^5$', '$2 \\times 10^5$', '$1 \\times 10^5$', '$0$']
#secax.set_xticklabels(sorted(list(secax.get_xticks()) + extra_ticks)[::-1], rotation=45, ha='left', rotation_mode='default')`
secax.set_xticklabels(sorted(list(secax.get_xticks()) )[::-1], rotation=45, ha='left', rotation_mode='default')



# Function to find intersection points
def find_intersections(x, y1, y2):
    # Find where the sign of the difference changes
    diff = y1 - y2
    sign_change_indices = np.where(np.diff(np.sign(diff)))[0]

    intersections = []
    
    # Linearly interpolate to find more accurate intersection points
    for i in sign_change_indices:
        x1, x2 = x[i], x[i + 1]
        y1_diff, y2_diff = diff[i], diff[i + 1]
        
        # Linear interpolation to find intersection point
        intersection_x = x1 - y1_diff * (x2 - x1) / (y2_diff - y1_diff)
        intersections.append(intersection_x)
    
    return intersections


print('# Find intersections between m(N) and r(N)')
intersection_N_m_r = find_intersections(N_values, Omega_m, Omega_r)
intersection_m_r   = np.interp(intersection_N_m_r, N_values, Omega_r)

print('# Find intersections between m(N) and Lambda(N)')
intersection_N_m_L = find_intersections(N_values, Omega_m, Omega_Lambda)
intersection_m_L   = np.interp(intersection_N_m_L, N_values, Omega_Lambda)

print('# Find intersections between r(N) and L(N)')
intersection_N_r_L = find_intersections(N_values, Omega_r, Omega_Lambda)
intersection_r_L   = np.interp(intersection_N_r_L, N_values, Omega_Lambda)

print('# Plot intersection points')
plt.scatter(intersection_N_m_r, intersection_m_r, color='blue', \
    label='$\Omega_m(N) ∩ \Omega_r(N)$=(%1.1f,%1.1f), z=%1.1f'%(intersection_N_m_r[0], intersection_m_r[0],N_to_redshift(intersection_N_m_r[0])))
plt.scatter(intersection_N_r_L, intersection_r_L, color='green', \
    label='$\Omega_r(N) ∩ \Omega_{\Lambda}(N)$=(%1.1f,%1.1f), z=%1.1f'%(intersection_N_r_L[0], intersection_r_L[0],N_to_redshift(intersection_N_r_L[0])))
plt.scatter(intersection_N_m_L, intersection_m_L, color='red', \
    label='$\Omega_m(N) ∩ \Omega_{\Lambda}(N)$=(%1.1f,%1.1f), z=%1.1f'%(intersection_N_m_L[0], intersection_m_L[0],N_to_redshift(intersection_N_m_L[0])))

plt.legend(fontsize=(12),ncol=2)

plt.tight_layout()
plt.show()

plt.savefig('./savefigs/phase_portraits_for_cosmology_from_ChatGPT_solutions_3D_4eqns_1initcond_with_zaxis_weff_intersections_delta_eta_tmin'+str(N_min)+'_tmax'+str(N_max)+'_zero_is_'+str(var_zero_is)+'.pdf')


print("""
########## ########## ########## ########## ##########  
########## Solving and ploting 
########## the solutions of 
########## 3D system of simultaneous differential equations for phiLCDM
########## The solution m(N), x(N), y(N)
########## functions of lapse time function $N$
########## ########## ########## ########## ########## 
""")

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# example value for lambda#0 #10

# Define the system of differential equations
def system(t, variables):
    m, x, y = variables
    f_mxy = (4 - m + 2 * x - 4 * y)
    dm_dt = m * ( f_mxy - 3 ) 
    dx_dt = x * ( f_mxy - 3 )
    dy_dt = y * f_mxy
    return [dm_dt, dx_dt, dy_dt]

"""
This part of commented code, is not used in final analysis
# Analytical solutions definitions of phiLambdaCDM
def system_analytical_phiLambdaCDM(N, Omega_m0, Omega_r0, Omega_x0, Omega_v0): 
    C1 = 1. - Omega_r0 - Omega_v0
    r_denominator = Omega_r0 - 3*N*np.exp(N)*Omega_x0 + np.exp(4*N)*Omega_v0 + C1*np.exp(N)
    Omega_r_N = Omega_r0 / r_denominator
    Omega_x_N = Omega_r_N * np.exp(N) * Omega_x0 / Omega_r0 
    Omega_v_N = Omega_r_N * np.exp(4*N) * Omega_v0 / Omega_r0 
    Omega_m_N = Omega_r_N * np.exp(N) * Omega_m0 / Omega_r0 
    return( Omega_m_N, Omega_r_N, Omega_x_N, Omega_v_N )
"""

# Analytical solutions definitions of phiLambdaCDM
def system_analytical_phiLambdaCDM(N, Omega_m0, Omega_r0, Omega_x0, Omega_v0):
    # Common denominator for all equations
    denominator = (np.exp(-3*N) * Omega_m0 + 
                   np.exp(-4*N) * Omega_r0 + 
                   np.exp(-3*N) * Omega_x0 + 
                   Omega_v0)
    
    # Calculate each Omega component
    Omega_m_N = Omega_m0 * np.exp(-3*N) / denominator
    Omega_r_N = Omega_r0 * np.exp(-4*N) / denominator
    Omega_x_N = Omega_x0 * np.exp(-3*N) / denominator
    Omega_v_N = Omega_v0 / denominator
    
    return Omega_m_N, Omega_r_N, Omega_x_N, Omega_v_N

# Set the initial conditions and time span
var_zero_is = 3e-17 #2.5e-9 #2.5e-9
#initial_conditions = [0.009, 0.0, var_zero_is]  # initial values for m, x, y # 0.009
initial_conditions = [0.009, 0.000001, var_zero_is]  # initial values for m, x, y # 0.009
#initial_conditions = [0.009, var_zero_is, var_zero_is]  # initial values for m, x, y # 0.009

N_min = -12   #-13
N_max = 1     #2
N_span = (N_min, N_max)  # lapse time function range
N_eval = np.linspace(*N_span, N_step_size)  # lapse time function values to evaluate

# Solve the system of equations
solution = solve_ivp(system, N_span, initial_conditions, t_eval=N_eval, method='RK45', dense_output=True)

# Extract the results
N = solution.t
m, x, y = solution.y

r = 1-m-x-y

phi = x + y

# Plot the results with the effective equation of state

weff_phi_LCDM = 1./3.*r + x - y


# Analytical solutions calculations
Omega_m0 = 0.32
Omega_x0 = 0.000035
Omega_r0 = 2e-4
Omega_v0 = 1 - Omega_m0 - Omega_r0 - Omega_x0

Omega_m_N_anal, Omega_r_N_anal, Omega_x_N_anal, Omega_v_N_anal = system_analytical_phiLambdaCDM(N, Omega_m0, Omega_r0, Omega_x0, Omega_v0)
Omega_phi_N_anal = Omega_x_N_anal + Omega_v_N_anal

weff_phi_LCDM_anal= 1./3.*Omega_r_N_anal + Omega_x_N_anal - Omega_v_N_anal

print('# It finds the intersection points, and plots a new plot with points printed')

# Function to find intersection points
def find_intersections(x, y1, y2):
    # Find where the sign of the difference changes
    diff = y1 - y2
    sign_change_indices = np.where(np.diff(np.sign(diff)))[0]

    intersections = []
    
    # Linearly interpolate to find more accurate intersection points
    for i in sign_change_indices:
        x1, x2 = x[i], x[i + 1]
        y1_diff, y2_diff = diff[i], diff[i + 1]
        
        # Linear interpolation to find intersection point
        intersection_x = x1 - y1_diff * (x2 - x1) / (y2_diff - y1_diff)
        intersections.append(intersection_x)
    
    return intersections


print('# Find intersections between m(N) and r(N)')
intersection_N_m_r = find_intersections(N, m, r)
intersection_m_r   = np.interp(intersection_N_m_r, N, r)

print('# Find intersections between m(N) and phi(N)')
intersection_N_m_phi = find_intersections(N, m, phi)
intersection_m_phi   = np.interp(intersection_N_m_phi, N, phi)

print('# Find intersections between r(N) and phi(N)')
intersection_N_r_phi = find_intersections(N, r, phi)
intersection_r_phi   = np.interp(intersection_N_r_phi, N, phi)

print('# Find intersections between m(N) and r(N)')
intersection_N_m_r_anal = find_intersections(N, Omega_m_N_anal, Omega_r_N_anal)
intersection_m_r_anal   = np.interp(intersection_N_m_r_anal, N, Omega_r_N_anal)

print('# Find intersections between m(N) and phi(N) analytical')
intersection_N_m_phi_anal = find_intersections(N, Omega_m_N_anal, Omega_phi_N_anal)
intersection_m_phi_anal   = np.interp(intersection_N_m_phi_anal, N, Omega_phi_N_anal)

print('# Find intersections between r(N) and phi(N) analytical')
intersection_N_r_phi_anal = find_intersections(N, Omega_r_N_anal, Omega_phi_N_anal)
intersection_r_phi_anal   = np.interp(intersection_N_r_phi_anal, N, Omega_phi_N_anal)

print('# Find intersections between x(N) and phi(N) analytical')
intersection_N_x_phi_anal = find_intersections(N, Omega_x_N_anal, Omega_phi_N_anal)
intersection_x_phi_anal   = np.interp(intersection_N_x_phi_anal, N, Omega_phi_N_anal)

print('# Find intersections between v(N) and phi(N) analytical')
intersection_N_v_phi_anal = find_intersections(N, Omega_v_N_anal, Omega_phi_N_anal)
intersection_v_phi_anal   = np.interp(intersection_N_v_phi_anal, N, Omega_phi_N_anal)

print('# Find intersections between x(N) and phi(N) analytical')
intersection_N_x_v_anal = find_intersections(N, Omega_x_N_anal, Omega_v_N_anal)
intersection_x_v_anal   = np.interp(intersection_N_x_v_anal, N, Omega_v_N_anal)

# Add second upper axis the redshift for mxyr
plt.ion()
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(N, m, '.', label='$\Omega_m(N)$', color='blue')
ax1.plot(N, phi, '.', label='$\Omega_{\phi}(N)$', color='green')
ax1.plot(N, x, '.', label='$\Omega_{x}(N)$', color='red')
ax1.plot(N, y, '.', label='$\Omega_{v}(N)$', color='pink')
ax1.plot(N, r, '.', label='$\Omega_r(N)=(1-\Omega_m-x-v)(N)$', color='orange')
ax1.plot(N, weff_phi_LCDM, '.', label='$w_{\\rm eff}(N)=(\\frac{1}{3}\Omega_r+x-v)(N)$', color='purple')


# Analytical solutions plotting
ax1.plot(N_values, Omega_m_N_anal, '-', color='blue', label='$\Omega_m^{anal}(N), \Omega_{m,0}=$%1.2f'%Omega_m0)
ax1.plot(N_values, Omega_phi_N_anal, '-', color='green',label='$\Omega_\phi^{anal}(N), \Omega_{v,0}=$%1.2f'%(Omega_x0+Omega_v0))
ax1.plot(N_values, Omega_r_N_anal, '-', color='orange', label='$\Omega_r^{anal}(N), \Omega_{r,0}=$%1.1e'%Omega_r0)
ax1.plot(N_values, Omega_x_N_anal, '-', color='red',label='$\Omega_x^{anal}(N), \Omega_{x,0}=$%1.1e'%Omega_x0)
ax1.plot(N_values, Omega_v_N_anal, '-', color='pink',label='$\Omega_v^{anal}(N), \Omega_{v,0}=$%1.2f'%Omega_v0)
ax1.plot(N, weff_phi_LCDM_anal , '-', label='$w_{\\rm eff}^{anal}(N)=(\\frac{1}{3}\Omega_r+x-v)(N)$', color='purple')


ax1.set_title(f'Initial Condition: {initial_conditions}',size=15)
#ax1.set_title('Initial Condition: [$0.009, 0, 2.5 \\times 10^{-9}$], $\lambda=$%1.0f'%(_lambda_), size=15)
ax1.set_xlabel('Lapse function $N = \ln[a(t)]$',size=20)
ax1.set_ylabel('Solution, $\Omega_Z(N)$',size=20)
ax1.set_ylim([-1.1,1.1])
ax1.legend(loc='lower left',fontsize=(10),ncol=3)
ax1.grid()

# Add a secondary x-axis for the redshift z
def N_to_redshift(N):
    return (np.exp(-N) - 1)

def redshift_to_N(z):
    return -np.log(z + 1)

# Create a secondary x-axis
secax = ax1.secondary_xaxis('top', functions=(N_to_redshift, redshift_to_N))
secax.set_xlabel('Redshift $z$',size=15)

# Set the ticks on the secondary x-axis to match the primary axis at 3 significant figures
z_ticks = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
secax.set_xticks(z_ticks)
redshift_from_N_extra_ticks = np.round(N_to_redshift(np.array([-10, -8, -6, -4, -2])),0)
extra_ticks = list(redshift_from_N_extra_ticks)
#secax.set_xticks( sorted(list(secax.get_xticks()) + extra_ticks)[::-1] )
secax.set_xticks( sorted(list(secax.get_xticks()) )[::-1] )

# Create labels in scientific notation
#labels = ['$0$', '$1 \\times 10^0$', '$1 \\times 10^1$', '$1 \\times 10^2$', '$1 \\times 10^3$', '$1 \\times 10^4$', '$1 \\times 10^5$', '$1 \\times 10^6$', '$1 \\times 10^7$', '$1 \\times 10^8$']
#labels = ['$8 \\times 10^5$', '$7 \\times 10^5$', '$6 \\times 10^5$', '$5 \\times 10^5$', '$4 \\times 10^5$', '$3 \\times 10^5$', '$2 \\times 10^5$', '$1 \\times 10^5$', '$0$']
#secax.set_xticklabels(sorted(list(secax.get_xticks()) + extra_ticks)[::-1], rotation=45, ha='left', rotation_mode='default')
secax.set_xticklabels(sorted(list(secax.get_xticks()) )[::-1], rotation=45, ha='left', rotation_mode='default')



# Plotting the curves

# Plot intersection points for m(N) and r(N)
ax1.scatter(intersection_N_m_r, intersection_m_r, color='blue',  \
    label='$m(N) ∩ r(N)$=(%1.1f,%1.1f), $z_{mr}$=%1.1f'%(intersection_N_m_r[0], intersection_m_r[0],N_to_redshift(intersection_N_m_r[0])))

ax1.scatter(intersection_N_r_phi, intersection_r_phi, color='magenta', \
    label='$r(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{r\phi}$=%1.1f'%(intersection_N_r_phi[0], intersection_r_phi[0],N_to_redshift(intersection_N_r_phi[0])))

ax1.scatter(intersection_N_m_phi, intersection_m_phi, color='red', \
    label='$m(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{m\phi}$=%1.1f'%(intersection_N_m_phi[0], intersection_m_phi[0],N_to_redshift(intersection_N_m_phi[0])))

# Plot intersection points for m(N) and r(N) for analytical model
ax1.scatter(intersection_N_m_r_anal, intersection_m_r_anal, marker='x', color='blue',  \
    label='$m(N) ∩ r(N)$=(%1.1f,%1.1f), $z_{mr}$=%1.1f'%(intersection_N_m_r_anal[0], intersection_m_r_anal[0],N_to_redshift(intersection_N_m_r_anal[0])))

ax1.scatter(intersection_N_r_phi_anal, intersection_r_phi_anal, marker='x', color='magenta', \
    label='$r(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{r\phi}$=%1.1f'%(intersection_N_r_phi_anal[0], intersection_r_phi_anal[0],N_to_redshift(intersection_N_r_phi_anal[0])))

ax1.scatter(intersection_N_m_phi_anal, intersection_m_phi_anal, marker='x', color='red', \
    label='$m(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{m\phi}$=%1.1f'%(intersection_N_m_phi_anal[0], intersection_m_phi_anal[0],N_to_redshift(intersection_N_m_phi_anal[0])))


#plt.xlabel('$N = \ln[a(t)]$')
#plt.ylabel('Z(N)')
ax1.legend(fontsize=(12),ncol=3)
ax1.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('./savefigs/ChatGPT_scalarLCDM_7D_3D_set_of_sim_diff_eqns_num_solutions_mphir_weos_with_redshift_intersections_delta_N_Nmin'+str(N_min)+'_Nmax'+str(N_max)+'_zero_is_'+str(var_zero_is)+'.pdf')

print("Intersection points (N, m(N) ∩ r(N)): ", list(zip(intersection_N_m_r, intersection_m_r)))


print('# ZOOM in in the previous plot')

# Add second upper axis the redshift for mxyr
plt.ion()
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(N, m, '.', label='$\Omega_m(N)$', color='blue')
ax1.plot(N, phi, '.', label='$\Omega_{\phi}(N)$', color='green')
ax1.plot(N, x, '.', label='$\Omega_{x}(N)$', color='red')
ax1.plot(N, y, '.', label='$\Omega_{v}(N)$', color='pink')
ax1.plot(N, r, '.', label='$\Omega_r(N)=(1-\Omega_m-x-v)(N)$', color='orange')
ax1.plot(N, weff_phi_LCDM, '.', label='$w_{\\rm eff}(N)=(\\frac{1}{3}\Omega_r+x-v)(N)$', color='purple')


# Analytical solutions plotting
ax1.plot(N_values, Omega_m_N_anal, '-', color='blue', label='$\Omega_m^{anal}(N), \Omega_{m,0}=$%1.2f'%Omega_m0)
ax1.plot(N_values, Omega_phi_N_anal, '-', color='green',label='$\Omega_\phi^{anal}(N), \Omega_{v,0}=$%1.2f'%(Omega_x0+Omega_v0))
ax1.plot(N_values, Omega_r_N_anal, '-', color='orange', label='$\Omega_r^{anal}(N), \Omega_{r,0}=$%1.1e'%Omega_r0)
ax1.plot(N_values, Omega_x_N_anal, '-', color='red',label='$\Omega_x^{anal}(N), \Omega_{x,0}=$%1.1e'%Omega_x0)
ax1.plot(N_values, Omega_v_N_anal, '-', color='pink',label='$\Omega_v^{anal}(N), \Omega_{v,0}=$%1.2f'%Omega_v0)
ax1.plot(N, weff_phi_LCDM_anal , '-', label='$w_{\\rm eff}^{anal}(N)=(\\frac{1}{3}\Omega_r+x-v)(N)$', color='purple')


ax1.set_title(f'Initial Condition: {initial_conditions}',size=15)
#ax1.set_title('Initial Condition: [$0.009, 0, 2.5 \\times 10^{-9}$], $\lambda=$%1.0f'%(_lambda_), size=15)
ax1.set_xlabel('Lapse function $N = \ln[a(t)]$',size=20)
ax1.set_ylabel('Solution, $\Omega_Z(N)$',size=20)
ax1.set_ylim([-2e-4,2e-3])
ax1.legend(loc='lower left',fontsize=(10),ncol=3)
ax1.grid()

# Add a secondary x-axis for the redshift z
def N_to_redshift(N):
    return (np.exp(-N) - 1)

def redshift_to_N(z):
    return -np.log(z + 1)

# Create a secondary x-axis
secax = ax1.secondary_xaxis('top', functions=(N_to_redshift, redshift_to_N))
secax.set_xlabel('Redshift $z$',size=15)

# Set the ticks on the secondary x-axis to match the primary axis at 3 significant figures
z_ticks = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
secax.set_xticks(z_ticks)
redshift_from_N_extra_ticks = np.round(N_to_redshift(np.array([-10, -8, -6, -4, -2])),0)
extra_ticks = list(redshift_from_N_extra_ticks)
#secax.set_xticks( sorted(list(secax.get_xticks()) )[::-1] )
secax.set_xticks( sorted(list(secax.get_xticks()) + extra_ticks)[::-1] )

# Create labels in scientific notation
#labels = ['$0$', '$1 \\times 10^0$', '$1 \\times 10^1$', '$1 \\times 10^2$', '$1 \\times 10^3$', '$1 \\times 10^4$', '$1 \\times 10^5$', '$1 \\times 10^6$', '$1 \\times 10^7$', '$1 \\times 10^8$']
#labels = ['$8 \\times 10^5$', '$7 \\times 10^5$', '$6 \\times 10^5$', '$5 \\times 10^5$', '$4 \\times 10^5$', '$3 \\times 10^5$', '$2 \\times 10^5$', '$1 \\times 10^5$', '$0$']
#secax.set_xticklabels(sorted(list(secax.get_xticks()) + extra_ticks)[::-1], rotation=45, ha='left', rotation_mode='default')
secax.set_xticklabels(sorted(list(secax.get_xticks()) )[::-1], rotation=45, ha='left', rotation_mode='default')



# Plotting the curves

# Plot intersection points for m(N) and r(N)
ax1.scatter(intersection_N_m_r, intersection_m_r, color='blue',  \
    label='$m(N) ∩ r(N)$=(%1.1f,%1.1f), $z_{mr}$=%1.1f'%(intersection_N_m_r[0], intersection_m_r[0],N_to_redshift(intersection_N_m_r[0])))

ax1.scatter(intersection_N_r_phi, intersection_r_phi, color='magenta', \
    label='$r(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{r\phi}$=%1.1f'%(intersection_N_r_phi[0], intersection_r_phi[0],N_to_redshift(intersection_N_r_phi[0])))

ax1.scatter(intersection_N_m_phi, intersection_m_phi, color='red', \
    label='$m(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{m\phi}$=%1.1f'%(intersection_N_m_phi[0], intersection_m_phi[0],N_to_redshift(intersection_N_m_phi[0])))

# Plot intersection points for m(N) and r(N) for analytical model
ax1.scatter(intersection_N_m_r_anal, intersection_m_r_anal, marker='x', color='blue',  \
    label='$m(N) ∩ r(N)$=(%1.1f,%1.1f), $z_{mr}$=%1.1f'%(intersection_N_m_r_anal[0], intersection_m_r_anal[0],N_to_redshift(intersection_N_m_r_anal[0])))

ax1.scatter(intersection_N_r_phi_anal, intersection_r_phi_anal, marker='x', color='magenta', \
    label='$r(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{r\phi}$=%1.1f'%(intersection_N_r_phi_anal[0], intersection_r_phi_anal[0],N_to_redshift(intersection_N_r_phi_anal[0])))

ax1.scatter(intersection_N_m_phi_anal, intersection_m_phi_anal, marker='x', color='red', \
    label='$m(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{m\phi}$=%1.1f'%(intersection_N_m_phi_anal[0], intersection_m_phi_anal[0],N_to_redshift(intersection_N_m_phi_anal[0])))

"""
# Plot intersection points for x(N) and v(N) for analytical model
ax1.scatter(intersection_N_x_phi_anal, intersection_x_phi_anal, marker='+', color='green', \
    label='$x(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{x\phi}$=%1.1f'%(intersection_N_x_phi_anal[0], intersection_x_phi_anal[0],N_to_redshift(intersection_N_x_phi_anal[0])))

ax1.scatter(intersection_N_v_phi_anal, intersection_v_phi_anal, marker='+', color='pink', \
    label='$v(N) ∩ \\tilde{\phi}(N)$=(%1.1f,%1.1f), $z_{v\phi}$=%1.1f'%(intersection_N_v_phi_anal[0], intersection_v_phi_anal[0],N_to_redshift(intersection_N_v_phi_anal[0])))
"""
ax1.scatter(intersection_N_x_v_anal, intersection_x_v_anal, marker='o', color='red', \
    label='$x(N) ∩ v(N)$=(%1.1f,%1.1e), $z_{xv}$=%1.1f'%(intersection_N_x_v_anal[0], intersection_x_v_anal[0],N_to_redshift(intersection_N_x_v_anal[0])))




#plt.xlabel('$N = \ln[a(t)]$')
#plt.ylabel('Z(N)')
ax1.legend(fontsize=(12),ncol=2)
ax1.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('./savefigs/ChatGPT_scalarLCDM_7D_3D_set_of_sim_diff_eqns_num_solutions_mphir_weos_with_redshift_intersections_delta_N_Nmin'+str(N_min)+'_Nmax'+str(N_max)+'_zero_is_'+str(var_zero_is)+'_ZOOMIN.pdf')



print('# Compare the outputs of the two models')

plt.ion()
fig, ax3 = plt.subplots(1, 1, figsize=(14, 7))

ax3.plot(N, m-Omega_m, label='$m$', color='blue')
ax3.plot(N, phi-Omega_Lambda, label='$(\\tilde{\phi}-\Lambda)$', color='green')
#ax3.plot(N, (y**2)/Omega_Lambda, '--', label='$y^2/\Lambda$', color='red')
ax3.plot(N, r-Omega_r, label='$r$', color='orange')
ax3.plot(N, weff_phi_LCDM-weff_LCDM, '--', label='$w_{\\rm eff}$', color='purple')
ax3.set_title(f'Numerical Comparison, Initial Condition: {initial_conditions}', size=15)
#ax3.set_title('Initial Condition: [$0.009, 0, 2.5 \\times 10^{-9}$]', size=15)
ax3.set_xlabel('Lapse function $N = \ln[a(t)]$',size=20)
ax3.set_ylabel('Solution, $\Omega_s^{\phi\Lambda{\\rm CDM}}(N)-\Omega_s^{\Lambda{\\rm CDM}}(N)$',size=20)
ax3.set_ylim([-0.04,0.04])
ax3.legend(loc='lower left',fontsize=(15),ncol=2)
ax3.grid()


plt.savefig('./savefigs/ChatGPT_6D_to_3D_phiLCDM_comparison_with_3D_4eqns_LCDM_set_of_sim_diff_eqns_num_solutions.pdf')


print('# Compare the outputs of the analytical and numerical solutions LCDM')

plt.ion()
fig, ax3 = plt.subplots(1, 1, figsize=(14, 7))

ax3.plot(N, Omega_m_N_LCDM_anal-Omega_m, label='$m$', color='blue')
ax3.plot(N, Omega_L_N_LCDM_anal-Omega_Lambda, label='$\Lambda$', color='green')
ax3.plot(N, Omega_r_N_LCDM_anal-Omega_r, label='$r$', color='orange')
ax3.plot(N, weff_LCDM_anal-weff_LCDM, '--', label='$w_{\\rm eff}$', color='purple')
ax3.set_title(f'Analytical and Numerical Comparison, Initial Condition: {initial_conditions}', size=15)
#ax3.set_title('Initial Condition: [$0.009, 0, 2.5 \\times 10^{-9}$]', size=15)
ax3.set_xlabel('Lapse function $N = \ln[a(t)]$',size=20)
ax3.set_ylabel('Solution, $\Omega_{s, anal}^{\Lambda{\\rm CDM}}(N)-\Omega_s^{\Lambda{\\rm CDM}}(N)$',size=20)
ax3.set_ylim([0.0,2])
ax3.set_ylim([-0.06,0.06])
ax3.legend(loc='upper left',fontsize=(15),ncol=2)
ax3.grid()

plt.savefig('./savefigs/ChatGPT_LCDM_comparison_analytical_numerical.pdf')

print('# Compare the outputs of the analytical and numerical solutions phiLCDM')

plt.ion()
fig, ax3 = plt.subplots(1, 1, figsize=(14, 7))

ax3.plot(N, Omega_m_N_anal-m, label='$m$', color='blue')
ax3.plot(N, Omega_phi_N_anal-phi, label='$\\tilde{\phi}$', color='green')
ax3.plot(N, Omega_r_N_anal-r, label='$r$', color='orange')
ax3.plot(N, weff_phi_LCDM_anal-weff_phi_LCDM, '--', label='$w_{\\rm eff}$', color='purple')
ax3.set_title(f'Analytical and Numerical Comparison, Initial Condition: {initial_conditions}', size=15)
#ax3.set_title('Initial Condition: [$0.009, 0, 2.5 \\times 10^{-9}$]', size=15)
ax3.set_xlabel('Lapse function $N = \ln[a(t)]$',size=20)
ax3.set_ylabel('Solution, $\Omega_{s, anal}^{\phi{\\rm CDM}}(N)-\Omega_s^{\phi{\\rm CDM}}(N)$',size=20)
ax3.set_ylim([0.0,2])
ax3.set_ylim([-0.04,0.04])
ax3.legend(loc='lower left',fontsize=(15),ncol=2)
ax3.grid()

plt.savefig('./savefigs/ChatGPT_phiLCDM_comparison_analytical_numerical.pdf')


print('# Compare the analytical solutions phiCDM/LCDM')

plt.ion()
fig, ax3 = plt.subplots(1, 1, figsize=(14, 7))

ax3.plot(N, Omega_m_N_LCDM_anal-Omega_m_N_anal, '-', label='$m$', color='blue')
ax3.plot(N, Omega_L_N_LCDM_anal-Omega_phi_N_anal, '-', label='$\Lambda-\phi$', color='green')
ax3.plot(N, Omega_r_N_LCDM_anal-Omega_r_N_anal, '--', label='$r$', color='orange')
ax3.plot(N, weff_LCDM_anal-weff_phi_LCDM_anal, '--', label='$w_{\\rm eff}$', color='purple')
ax3.set_title(f'Numerical Comparison of analytical solutions', size=15)
#ax3.set_title('Initial Condition: [$0.009, 0, 2.5 \\times 10^{-9}$]', size=15)
ax3.set_xlabel('Lapse function $N = \ln[a(t)]$',size=20)
ax3.set_ylabel('Solution, $\Omega_{s, anal}^{\Lambda{\\rm CDM}}(N)-\Omega_{s, anal}^{\phi {\\rm CDM}}(N)$',size=20)
ax3.set_ylim([0.0,2])
ax3.set_ylim([-0.04,0.04])
ax3.legend(loc='lower left',fontsize=(15),ncol=2)
ax3.grid()

plt.savefig('./savefigs/ChatGPT_phiLCDM_vs_LCDM_comparison_analytical.pdf')

print('# Compare the analytical solutions phiCDM/LCDM')

plt.ion()
fig, ax3 = plt.subplots(1, 1, figsize=(14, 7))

ax3.plot(N, Omega_m_N_LCDM_anal-Omega_m_N_anal, '-', label='$m$', color='blue')
ax3.plot(N, Omega_L_N_LCDM_anal-Omega_phi_N_anal, '-', label='$\Lambda-\phi$', color='green')
ax3.plot(N, Omega_r_N_LCDM_anal-Omega_r_N_anal, '--', label='$r$', color='orange')
ax3.plot(N, weff_LCDM_anal-weff_phi_LCDM_anal, '--', label='$w_{\\rm eff}$', color='purple')
ax3.set_title(f'Numerical Comparison of analytical solutions', size=15)
#ax3.set_title('Initial Condition: [$0.009, 0, 2.5 \\times 10^{-9}$]', size=15)
ax3.set_xlabel('Lapse function $N = \ln[a(t)]$',size=20)
ax3.set_ylabel('Solution, $\Omega_{s, anal}^{\Lambda{\\rm CDM}}(N)-\Omega_{s, anal}^{\phi {\\rm CDM}}(N)$',size=20)
ax3.set_ylim([-0.0002,0.0002])
ax3.legend(loc='lower left',fontsize=(15),ncol=2)
ax3.grid()

plt.savefig('./savefigs/ChatGPT_phiLCDM_vs_LCDM_comparison_analytical_ZOOMIN.pdf')



print('# Plot the scale solutions of LCDM')

import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(-1e11,1e2,200)
time = np.linspace(0,2,200)


def a_scale_factor_LCDM(t, H_0=70, Omega_m0=0,Omega_r0=0):
    if Omega_m0==1: 
        res =  ( 3/2 * H_0 * t )  ** (2/3)
    elif Omega_r0==1: 
        res =  (  2 * H_0 *  t    ) ** (1/2)
    elif Omega_m0==0 and Omega_r0==0: 
        res = np.exp( H_0 * (t - t_0) ) 
    return(res)

def a_scale_factor_LCDM_combined(t, H_0=70, Omega_m0=0,Omega_r0=0, alpha=1):
    res =  ( 3/2 * H_0 * t )  ** (2/3) + (  2 * H_0 *  t    ) ** (1/2) + np.exp( H_0 * (t - t_0) ) 
    result = res ** alpha
    return(result)    

H_0=1
t_0=1/2

a_scale_factor_LCDM_matter    = a_scale_factor_LCDM(time, H_0=H_0, Omega_m0=1,Omega_r0=0)
a_scale_factor_LCDM_radiation = a_scale_factor_LCDM(time, H_0=H_0, Omega_m0=0,Omega_r0=1)
a_scale_factor_LCDM_Lambda    = a_scale_factor_LCDM(time, H_0=H_0, Omega_m0=0,Omega_r0=0)
a_scale_factor_LCDM_combined  = a_scale_factor_LCDM_combined(time, H_0=H_0, Omega_m0=1/3,Omega_r0=1/3, alpha=1)

plt.figure(6,)
plt.ion()
plt.clf()
plt.plot(time, a_scale_factor_LCDM_matter, 'b-', label='Matter Domination')
plt.plot(time, a_scale_factor_LCDM_radiation, 'y-', label='Radiation Domination')
plt.plot(time, a_scale_factor_LCDM_Lambda, 'g-', label='$\Lambda$ Domination')
plt.plot(time, a_scale_factor_LCDM_combined, 'r--', label='$\Lambda$CDM Combined $\Omega_{m,0}=\Omega_{r,0}=\Omega_{\Lambda,0}=1/3$')

plt.ylim(-0.1,5)
plt.legend()
plt.xlabel('Time, $t$ [units of $H_0^{-1}$]')
plt.ylabel('$a(t)$')
plt.xscale('linear')
plt.yscale('linear')
plt.grid()
plt.draw()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the parameters
H_0 = 1  # Hubble constant
Omega_m0 =  0.3  # Matter density parameter
Omega_r0 =  1e-4  # Radiation density parameter
Omega_Lambda0 = 1 - Omega_m0 - Omega_r0  # Dark energy density parameter

# Define the integrand function for the scale factor a
def integrand(a):
    return 1 / (a * np.sqrt(Omega_m0 * a**-3 + Omega_r0 * a**-4 + Omega_Lambda0))

# Define the differential equation to solve
def equations(t, y):
    a = y[0]
    dadt =  H_0 * np.sqrt(Omega_m0 * a**-1 + Omega_r0 * a**-2 + Omega_Lambda0 * a**2 ) 
    return [dadt]

# Initial conditions and time span
a0 = 1e-5  # Initial scale factor (close to zero)
t_span = (0, 2)  # Time range for integration
initial_conditions = [a0]

# Solve the differential equation using solve_ivp
solution = solve_ivp(equations, t_span, initial_conditions, t_eval=np.linspace(t_span[0], t_span[1], 100))

# Extract results
t_values = solution.t
a_values = solution.y[0]

# Plotting the scale factor a as a function of t
plt.figure(6,figsize=(10, 6))
plt.plot(t_values, a_values, 'm--',label='Numerical Scale Factor $a(t)$')
plt.title('Scale Factor as a Function of Time')
plt.xlabel('Time $t$ [in units of $H_0^{-1}$]')
plt.ylabel('Scale Factor $a$')
plt.ylim(0, np.max(a_values) * 1.1)  # Adjusting y-axis for better visibility
plt.legend()
plt.show()

'''
print('Calculate the time of radiation-matter equality, LCDM')
from sympy import symbols, sqrt, integrate

# Define symbols
a, a_RM = symbols('a a_RM', positive=True, real=True)

# Define the integrand
integrand = 1 / sqrt(1/a + a_RM/a**2)

# Perform the integral
result = integrate(integrand, (a, 0, a_RM))
result.simplify()
#2*a_RM**(3/2)*(2 - sqrt(2))/3
'''
print('Calculate the time of radiation-matter equality, LCDM, with numerics')
t_MR_LCDM = 2 * H_0 **(-1) *  Omega_m0**(-1/2) * (  Omega_r0/ Omega_m0  ) ** (3/2) * (2 - np.sqrt(2) ) / 3   
print(t_MR_LCDM) #6.198798281766191e-08
'''
print('Calculate the time of matter-cosmological-constant equality, LCDM')
# Define the integrand for the second integral
a_LM = symbols('a_LM', positive=True, real=True)
integrand_LM = 1 / (a * sqrt(a_LM**3 / a**3 + 1))

# Perform the integral from a_LM to 1
result_LM = integrate(integrand_LM, (a, 0, a_LM))
result_LM.simplify()
#2*log(1 + sqrt(2))/3
'''
print('Calculate the time of matter-cosmological-constant , LCDM, with numerics')
t_ML_LCDM=  2/3 * H_0 **(-1) *  Omega_Lambda0 **(-1/2) *  2*np.log(1 + np.sqrt(2))/3
print(t_ML_LCDM) #0.0066890043219288535

print('Calculate the time of equality, LCDM, with numerics')

import numpy as np
from scipy.integrate import quad

a_eq = 1e-4  # Scale factor at equality (example value)

# Define the integrand
def integrand(a, Omega_m0, Omega_r0, Omega_Lambda0):
    return 1 / (a * np.sqrt(Omega_m0 * a**-3 + Omega_r0 * a**-4 + Omega_Lambda0))

# Perform the integration
t_eq, error = quad(integrand, 0, a_eq, args=(Omega_m0, Omega_r0, Omega_Lambda0))

# Convert H0 to consistent time units (e.g., Gyr^-1)
H0_s = H_0 * 1e3 / (3.086e22)  # Convert H0 to s^-1
t_eq = t_eq / H0_s  # Time in seconds
t_eq_gyr = t_eq / (3.154e16)  # Convert seconds to Gyr

t_eq_gyr, error


print('Calculate the time of matter-radiation , LCDM, with whole integral numerics')

a_MR_LCDM = Omega_r0/Omega_m0

# Perform the integration
t_MR_LCDM, error = quad(integrand, 0, a_MR_LCDM, args=(Omega_m0, Omega_r0, Omega_Lambda0))


# Convert H0 to consistent time units (e.g., Gyr^-1)
H0_s = H_0 * 1e3 / (3.086e22)  # Convert H0 to s^-1
#t_MR_LCDM = t_RM_LCDM / H0_s  # Time in seconds
#t_MR_LCDM_gyr = t_RM_LCDM / (3.154e16)  # Convert seconds to Gyr
#t_MR_LCDM_gyr, error
t_MR_LCDM

print('Calculate the time of matter-cosmological-constant , LCDM, with whole integral numerics')

a_ML_LCDM = (  Omega_m0/Omega_Lambda0 )**(1/3)

# Perform the integration
t_ML_LCDM, error = quad(integrand, 0, a_ML_LCDM, args=(Omega_m0, Omega_r0, Omega_Lambda0))


# Convert H0 to consistent time units (e.g., Gyr^-1)
H0_s = H_0 * 1e3 / (3.086e22)  # Convert H0 to s^-1
#t_ML_LCDM = t_ML_LCDM / H0_s  # Time in seconds
#t_ML_LCDM_gyr = t_ML_LCDM / (3.154e16)  # Convert seconds to Gyr
#t_ML_LCDM_gyr, error
t_ML_LCDM


print('calculate piecewise function model LCDM')
import numpy as np

def R(t, t_i, t_RM):
    """Radiation dominance indicator function."""
    return 1 if t_i <= t < t_RM else 0

def M(t, t_RM, t_ML):
    """Matter dominance indicator function."""
    return 1 if t_RM <= t < t_ML else 0

def L(t, t_LM):
    """Lambda (dark energy) dominance indicator function."""
    return 1 if t >= t_LM else 0

def a(t, t_i, t_RM, t_ML, t_LM, H0, t0):
    """Combined solution for scale factor a(t)."""
    radiation_term = R(t, t_i, t_RM) * (2 * H0)**0.5 * t**0.5
    matter_term = M(t, t_RM, t_ML) * ((3 / 2) * H0)**(2 / 3) * t**(2 / 3)
    lambda_term = L(t, t_LM) * np.exp(H0 * (t - t0))
    return radiation_term + matter_term + lambda_term

# Parameters (example values)
H_0 = 1  # Hubble constant (in units consistent with t)
t0 = 0   # Initial time
t_i = 0  # Initial time for radiation
t_RM = t_MR_LCDM  # Radiation-matter equality time
t_ML = t_ML_LCDM  # Matter-Lambda equality time
t_LM = t_ML_LCDM # Lambda dominance time

time_points = np.linspace(0, 2, 1000)  # Time range (logarithmic scale)
a_values = [a(t, t_i, t_RM, t_ML, t_LM, H_0, t0) for t in time_points]

# Plotting the results
import matplotlib.pyplot as plt

plt.figure(7,figsize=(10, 6))
plt.clf()
plt.plot(time_points, a_values, '--', label='a(t)', color='blue')
plt.axvline(t_RM, color='orange', linestyle='--', label='$t_{RM}$')
plt.axvline(t_ML, color='green', linestyle='--', label='$t_{ML}$')
plt.axvline(t_LM, color='red', linestyle='--', label='$t_{LM}$')
plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('Scale Factor (a(t))', fontsize=14)
plt.title('Piecewise Scale Factor a(t)', fontsize=16)
plt.legend()
plt.ylim(-0.1,5)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

