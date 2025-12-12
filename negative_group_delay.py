#Negative Group Delay
import numpy as np; import matplotlib.pyplot as plt
# TIS Constants
K_TILE = 13.0
F9_FRICTION = 34.0
# TIS Geometric Friction is mapped to the Damping Constant (Gamma)
GAMMA_TIS = F9_FRICTION / K_TILE # ~2.615
# Standard (Causal) Damping for comparison (e.g., Gamma_SM = 0.1)
GAMMA_SM = 0.1
# Simulation Parameters (Normalized Units)
OMEGA_RANGE = np.linspace(0.5, 1.5, 1000) # Frequency sweep around resonance
D_OMEGA = OMEGA_RANGE[1] - OMEGA_RANGE[0] # Step size for derivative
OMEGA_0 = 1.0 # Resonance frequency (M^2 in propagator context)
OMEGA_P = 0.2 # Plasma frequency (Coupling strength)
EPSILON_INF = 1.0 # Background permittivity
def calculate_dispersion_and_group_index(omega, gamma):
    """
    Calculates complex index of refraction N and group index Ng based on a
    TIS-modified Drude-Lorentz dispersion model, where Gamma is the TIS friction.
    """
    # 1. Complex Permittivity (Epsilon)
    # The TIS propagator term (p^2 - M^2 + i*Gamma*p) is analogous to the 
    # denominator in the Lorentz oscillator model: (w_0^2 - w^2 - i*w*Gamma)
    denominator = (OMEGA_0**2 - omega**2) - 1j * omega * gamma
    
    epsilon = EPSILON_INF + (OMEGA_P**2 / denominator)    
    # 2. Complex Refractive Index (N = n + i*kappa)
    N = np.sqrt(epsilon)
    n = np.real(N)      # Refractive Index (Phase Velocity)
    kappa = np.imag(N)  # Extinction Coefficient (Absorption)    
    # 3. Group Index (Ng) - Proportional to Group Delay (tau_g)
    # n_g = n + omega * dn/d_omega
    # Calculate dn/d_omega using numerical gradient (finite difference)
    dn_domega = np.gradient(n, D_OMEGA)    
    # Group Index: if Ng < 0, the Group Delay (tau_g) is negative (anti-causal)
    n_g = n + omega * dn_domega
    return n, kappa, n_g

# Calculate TIS result
n_tis, kappa_tis, n_g_tis = calculate_dispersion_and_group_index(OMEGA_RANGE, GAMMA_TIS)
# Calculate Standard (Causal) result for comparison
n_sm, kappa_sm, n_g_sm = calculate_dispersion_and_group_index(OMEGA_RANGE, GAMMA_SM)
# Plotting
plt.figure(figsize=(12, 10))
# --- Subplot 1: Group Index (Negative Delay Check) ---
plt.subplot(2, 1, 1)
plt.plot(OMEGA_RANGE, n_g_sm, 'r--', label=f'SM Damping ($\\Gamma={GAMMA_SM}$)', alpha=0.6)
plt.plot(OMEGA_RANGE, n_g_tis, 'b-', label=f'TIS Friction ($\\Gamma_{{TIS}}={GAMMA_TIS:.3f}$)')
plt.axhline(0, color='k', linestyle=':', label='Zero Group Index ($n_g=0$)')
plt.axvline(OMEGA_0, color='gray', linestyle='--', alpha=0.5, label='Resonance ($\\omega_0$)')
plt.title('Negative Group Delay Simulation: SM vs TIS Friction (34/13)')
plt.xlabel('Normalized Frequency $\\omega$')
plt.ylabel('Group Index $n_g$ ($\\propto$ Group Delay $\\tau_g$)')
plt.ylim(-10, 10)
plt.grid(True)
plt.legend()
# --- Subplot 2: Absorption (Kappa) ---
plt.subplot(2, 1, 2)
plt.plot(OMEGA_RANGE, kappa_sm, 'r--', label=f'SM Damping ($\\Gamma={GAMMA_SM}$)', alpha=0.6)
plt.plot(OMEGA_RANGE, kappa_tis, 'b-', label=f'TIS Friction ($\\Gamma_{{TIS}}$)')
plt.axvline(OMEGA_0, color='gray', linestyle='--', alpha=0.5)
plt.title('Absorption Coefficient $\\kappa$')
plt.xlabel('Normalized Frequency $\\omega$')
plt.ylabel('Absorption $\\kappa$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('tis_negative_group_delay_simulation.png')
print("Generated plot: tis_negative_group_delay_simulation.png")