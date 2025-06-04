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
    m, r, Lambda = y

    # First Friedman equation
    f = 3 + r - 3 * Lambda

    # Continuity equations
    m_prime = m * (f - 3)
    r_prime = r * (f - 4)
    Lambda_prime = Lambda * f

    return [m_prime, r_prime, Lambda_prime]

# Define critical points
critical_points = {
    "S_m": np.array([1, 0, 0]),
    "R_r": np.array([0, 1, 0]),
    "A_Λ": np.array([0, 0, 1]),
}

variables = ['Ω_m', 'Ω_r', 'Ω_Λ']
time_span = (-13, 1)  # Time lapse function
num_points = 1000  # Points for trajectory
t_eval = np.linspace(*time_span, num_points)

def is_near_critical(point, tol=1e-2):
    """
    Checks if a given point is close to any critical point within a tolerance.
    """
    for name, crit in critical_points.items():
        if np.linalg.norm(point - crit) < tol:
            return name
    return None  # If not near any critical point

#def plot_trajectories():
"""
Plots the phase-space trajectories and adds a histogram of endpoints.
"""
plt.ion()
fig, axes = plt.subplots(2,2, figsize=(16, 8))
axes = axes.flatten()

pairs = list(combinations(range(len(variables)), 2))

legend_labels = []  # Store legend labels
legend_handles = []  # Store plot handles

critical_endings = []  # Store names of critical end states
non_critical_count = 0  # Count of non-critical endings

for idx, (name_start, start) in enumerate(critical_points.items()):
    for name_end, end in critical_points.items():
        if name_start == name_end:
            continue  # Skip self-transitions
        
        perturbation = 0.0001 * np.random.randn(len(critical_points))
        initial_state = np.clip(start + perturbation, 0.0001, 0.9999)        
        solution = solve_ivp(
            friedman_system,
            time_span,
            initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )

        final_state = solution.y[:, -1]
        final_critical = is_near_critical(final_state)

        # Choose color for end point
        if final_critical:
            end_color = "green"
            critical_endings.append('$'+final_critical+'$')  # Track where it ends
        else:
            end_color = "black"
            non_critical_count += 1  # Count non-critical endings

        # Unicode symbols for legend
        start_symbol = "(R)"  # Red circle
        end_symbol = "(G)" if final_critical else "(B)"  # Green for critical, black for non-critical
        if final_critical:
            legend_entry = f"${name_start}$ → ${final_critical}$ {end_symbol}"
        else:
            legend_entry = f"${name_start}$ → ND {end_symbol}"

        # Plot each pair projection
        for pair_idx, (var1_idx, var2_idx) in enumerate(pairs):
            ax = axes[pair_idx]
            traj_line, = ax.plot(
                solution.y[var1_idx],
                solution.y[var2_idx],
                alpha=0.7,
            )
            
            # Store legend handle once
            if pair_idx == 0:
                legend_handles.append(traj_line)
                legend_labels.append(legend_entry)

            # Scatter start and end points
            ax.scatter(start[var1_idx], start[var2_idx], color="red", s=50, edgecolors="black")
            ax.scatter(final_state[var1_idx], final_state[var2_idx], color=end_color, s=50, edgecolors="black")

            ax.set_xlabel(f"${variables[var1_idx]}$", fontsize=10)
            ax.set_ylabel(f"${variables[var2_idx]}$", fontsize=10)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(-0.1, 1.1)
            ax.grid(True)
            ax.set_title(f"${variables[var1_idx]}$ vs ${variables[var2_idx]}$")

# Remove the last three empty subplots
for i in range(len(pairs), len(axes)):  
    fig.delaxes(axes[i])

# Add the legend back in a proper location
fig.legend(
    legend_handles,
    legend_labels,
    fontsize=8,
    bbox_to_anchor=(0.5, 0.15),  # Position to the right
    loc="center left",
    ncol=2,
)

plt.tight_layout()
plt.savefig('./savefigs/numerical_phase_portraits_with_trajectories_LCDM_all_projections_good.pdf')

# Plot histogram
plt.figure(2,figsize=(7, 5))
plt.clf()

# Sort unique critical end points
unique_ends, counts = np.unique(critical_endings, return_counts=True)

# Bar plot
plt.bar(unique_ends, counts, color="green", label="Ended at a critical point")
plt.bar("Non-Dominant", non_critical_count, color="black", label="Ended at a non-dominant point")

plt.xlabel("Final State")
plt.ylabel("Number of Trajectories")
plt.title("Histogram of Final States")
plt.xticks(rotation=5, ha="right")
plt.legend()

plt.show()
plt.savefig('./savefigs/numerical_phase_portraits_with_trajectories_LCDM_all_projections_good_histogram.pdf')
print(critical_endings)

# Call the function to plot phase-space trajectories and histogram
#plot_trajectories()
