import numpy as np
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors for interactive annotations

# Parameters
E = 210e3  # Young's modulus (MPa)
nu = 0.3   # Poisson's ratio
sigma_y = 250  # Initial yield stress (MPa)
H_iso = 1000   # Isotropic hardening modulus (MPa)
H_kin = 50000    # Kinematic hardening modulus (MPa)
mix_ratio = 1  # Mixing ratio (0 for pure isotropic, 1 for pure kinematic)
epsilon_total = np.array([0.01, 0.005, 0.003, 0, 0, 0])  # Total strain increment [exx; eyy; ezz; gxy; gyz; gxz]

# Material constants
G = E / (2 * (1 + nu))  # Shear modulus
K = E / (3 * (1 - 2 * nu))  # Bulk modulus

# Elastic stiffness tensor (4th order matrix form)
C = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
    [1 - nu, nu, nu, 0, 0, 0],
    [nu, 1 - nu, nu, 0, 0, 0],
    [nu, nu, 1 - nu, 0, 0, 0],
    [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
    [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
    [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
])

# Initialize variables
epsilon = np.zeros(6)  # Total strain [exx; eyy; ezz; gxy; gyz; gxz]
epsilon_p = np.zeros(6)  # Plastic strain
sigma = np.zeros(6)  # Stress [sxx; syy; szz; txy; tyz; txz]
alpha = np.zeros(6)  # Back stress for kinematic hardening
sigma_iso = 0  # Isotropic hardening contribution
n_steps = 100  # Number of steps
delta_epsilon = epsilon_total / n_steps  # Strain increment per step

# Storage for results
strain = np.zeros((6, n_steps))
stress = np.zeros((6, n_steps))
von_mises = np.zeros(n_steps)  # To store von Mises stress

# Incremental loop
for i in range(n_steps):
    # Apply strain increment
    epsilon += delta_epsilon

    # Elastic predictor
    sigma_trial = np.dot(C, epsilon - epsilon_p)
    dev_sigma_trial = sigma_trial - np.mean(sigma_trial[:3]) * np.array([1, 1, 1, 0, 0, 0])  # Deviatoric stress
    sigma_eff_trial = dev_sigma_trial - alpha  # Effective deviatoric stress
    J2 = 0.5 * np.dot(sigma_eff_trial, sigma_eff_trial)  # J2 invariant
    sigma_eq = np.sqrt(3 * J2)  # von Mises equivalent stress

    # Yield check
    f_yield = sigma_eq - (sigma_y + sigma_iso)

    if f_yield <= 0:
        # Elastic step
        sigma = sigma_trial
    else:
        # Plastic correction
        delta_gamma = f_yield / (3 * G + H_iso * (1 - mix_ratio) + H_kin * mix_ratio)  # Consistency parameter
        normal = sigma_eff_trial / sigma_eq  # Plastic flow direction

        # Update plastic strain
        epsilon_p += delta_gamma * normal

        # Update back stress (kinematic hardening)
        alpha += 2 * G * mix_ratio * delta_gamma * normal

        # Update isotropic hardening contribution
        sigma_iso += H_iso * (1 - mix_ratio) * delta_gamma

        # Update stress
        sigma = sigma_trial - 2 * G * delta_gamma * normal

    # Store results
    strain[:, i] = epsilon
    stress[:, i] = sigma
    von_mises[i] = sigma_eq  # Store von Mises stress

    # Print results for each step
    print(f"Step {i + 1}:")
    print(f"  Strain: {epsilon}")
    print(f"  Stress: {sigma}")
    print(f"  Von Mises Stress: {sigma_eq:.2f} MPa")
    print(f"  Plastic Strain: {epsilon_p}")
    print(f"  Back Stress (alpha): {alpha}")
    print("-" * 50)

# Plot results
plt.figure(figsize=(14, 10))

# Strain evolution
ax1 = plt.subplot(2, 2, 1)
ax1.plot(range(1, n_steps + 1), strain[0, :], 'b', label=r'$\epsilon_{xx}$')
ax1.plot(range(1, n_steps + 1), strain[1, :], 'r', label=r'$\epsilon_{yy}$')
ax1.plot(range(1, n_steps + 1), strain[2, :], 'g', label=r'$\epsilon_{zz}$')
ax1.set_xlabel('Step')
ax1.set_ylabel('Strain')
ax1.legend()
ax1.set_title('Strain Evolution')
ax1.grid(True)
mplcursors.cursor(ax1, hover=True)  # Activate interactive cursor for strain plot

# Stress evolution
ax2 = plt.subplot(2, 2, 2)
ax2.plot(range(1, n_steps + 1), stress[0, :], 'b', label=r'$\sigma_{xx}$')
ax2.plot(range(1, n_steps + 1), stress[1, :], 'r', label=r'$\sigma_{yy}$')
ax2.plot(range(1, n_steps + 1), stress[2, :], 'g', label=r'$\sigma_{zz}$')
ax2.set_xlabel('Step')
ax2.set_ylabel('Stress (MPa)')
ax2.legend()
ax2.set_title('Stress Evolution')
ax2.grid(True)
mplcursors.cursor(ax2, hover=True)  # Activate interactive cursor for stress plot

# Von Mises stress evolution
ax3 = plt.subplot(2, 2, 3)
ax3.plot(range(1, n_steps + 1), von_mises, 'm', label='Von Mises Stress')
ax3.set_xlabel('Step')
ax3.set_ylabel('Von Mises Stress (MPa)')
ax3.legend()
ax3.set_title('Von Mises Stress Evolution')
ax3.grid(True)
mplcursors.cursor(ax3, hover=True)  # Activate interactive cursor for von Mises plot

# Stress-Strain Curve
ax4 = plt.subplot(2, 2, 4)
ax4.plot(strain[0, :], stress[0, :], 'b', label=r'$\sigma_{xx}$ vs $\epsilon_{xx}$')
ax4.plot(strain[1, :], stress[1, :], 'r', label=r'$\sigma_{yy}$ vs $\epsilon_{yy}$')
ax4.plot(strain[2, :], stress[2, :], 'g', label=r'$\sigma_{zz}$ vs $\epsilon_{zz}$')
ax4.set_xlabel('Strain')
ax4.set_ylabel('Stress (MPa)')
ax4.legend()
ax4.set_title('Stress-Strain Curve')
ax4.grid(True)
mplcursors.cursor(ax4, hover=True)

plt.tight_layout()
plt.show()