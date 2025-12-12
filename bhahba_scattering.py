# Bhahba Scattering
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import fine_structure
# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
# Standard Model Constants
ALPHA_SM = fine_structure # ~1/137.03599...
HBAR_C = 0.197326 # GeV*fm (Conversion factor)
# TIS Fundamental Constants
K_TILE = 13.0
C_PHI = 3.0
F9_FRICTION = 34.0
GEOMETRIC_FRICTION = F9_FRICTION / K_TILE  # ~2.615
TIS_VERTEX_G = 252.0 # The Geometric Coupling
# TIS Derived Alpha (from Section 14 derivation)
phi = (1 + np.sqrt(5)) / 2
alpha_inv_TIS = (13 * phi**5) - ((14/2) + (3/13))
ALPHA_TIS = 1 / alpha_inv_TIS
# ==========================================
# 2. STANDARD MODEL BHABHA SCATTERING
# ==========================================
def bhabha_sm(s, theta_rad):
    """
    Standard QED Differential Cross Section.
    Diverges as theta -> 0.
    """
    # Avoid division by zero for plot stability at exactly 0
    epsilon = 1e-9
    theta_rad = np.maximum(theta_rad, epsilon)
    
    # Pure t-channel dominance for comparison (1/t^2 scaling)
    # t = -s * sin^2(theta/2)
    # Standard Model Formula (Unpolarized)
    term1 = (1 + np.cos(theta_rad/2)**4) / (np.sin(theta_rad/2)**4)
    term2 = -2 * (np.cos(theta_rad/2)**4) / (np.sin(theta_rad/2)**2)
    term3 = (1 + np.cos(theta_rad)**2) / 2
    dsigma = (ALPHA_SM**2 / (4 * s)) * (term1 + term2 + term3)
    # Convert to nb/GeV^2 (approx) for visibility
    return dsigma * (HBAR_C**2) * 1e7 

# ==========================================
# 3. TIS MODULAR S-MATRIX
# ==========================================
def divisor_sigma5(n):
    """Computes sum of 5th powers of divisors."""
    divs = [i for i in range(1, int(n**0.5) + 1) if n % i == 0]
    res = 0
    for i in divs:
        res += i**5
        if i*i != n:
            res += (n//i)**5
    return res

def eisenstein_E6_texture(q_param, max_n=5):
    """
    Computes the E6 Modular Form texture.
    E6(q) = 1 - 504 * Sum(sigma_5(n) * q^n)
    """
    # Map scattering angle to modular parameter q
    # Heuristic mapping: Small angle = close to cusp (q -> 1)
    q = np.exp(-np.abs(q_param)) 
    series_sum = 0
    for n in range(1, max_n + 1):
        series_sum += divisor_sigma5(n) * (q**n)
    # TIS amplitude modulated by geometric coefficients (-504)
    # Scaled down for plot coherence to show the 'wobble'
    return np.abs(1 - (504/5000) * series_sum) 

def bhabha_tis(s, theta_rad):
    """
    TIS S-Matrix Trajectory.
    1. Uses ALPHA_TIS
    2. Modulates with Eisenstein Texture (E6)
    3. Applies 34/13 Geometric Friction to Propagator
    """
    # 1. Mandelstam t proxy
    t_val = -s * np.sin(theta_rad/2)**2
    # 2. TIS Propagator with Geometric Friction (34/13) # The friction acts as a regulator preventing divergence at t=0
    friction_barrier = GEOMETRIC_FRICTION * (s * 0.001) 
    # Modified propagator: 1 / sqrt(t^2 + friction^2)
    propagator_tis = 1 / np.sqrt(t_val**2 + friction_barrier**2)
    # 3. Modular Texture (E6)
    q_proxy = theta_rad * 3 # Mapping for texture frequency
    texture = eisenstein_E6_texture(q_proxy)
    # 4. Construct Cross Section # Using ALPHA_TIS instead of standard Alpha
    prefactor = (ALPHA_TIS**2 / (4 * s))
    # TIS replaces the divergent sin^4 term with the frictional propagator
    dsigma = prefactor * (propagator_tis**2) * (s**2) * texture
    return dsigma * (HBAR_C**2) * 1e7
# ==========================================
# 4. SIMULATION AND PLOTTING
# ==========================================
# Parameters
E_cm = 10.0 # GeV
s = E_cm**2
theta_deg = np.linspace(0.1, 180, 500) 
theta_rad = np.radians(theta_deg)
sigma_sm = bhabha_sm(s, theta_rad)
sigma_tis = bhabha_tis(s, theta_rad)
plt.figure(figsize=(12, 8))
# Main Plot: Cross Section
plt.subplot(2, 1, 1)
plt.plot(theta_deg, sigma_sm, 'r--', label='Standard Model (QED) - Divergent', alpha=0.6)
plt.plot(theta_deg, sigma_tis, 'b-', label='TIS Theory (Friction 34/13) - Finite', linewidth=2)
plt.yscale('log')
plt.title(f'Bhabha Scattering: SM vs TIS ($E_{{cm}}$={E_cm} GeV)')
plt.ylabel(r'$d\sigma/d\Omega$ (nb/sr)')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.subplot(2, 1, 2)
valid_idx = theta_deg > 5 # Avoid the divergence region for the ratio plot
ratio = sigma_tis[valid_idx] / sigma_sm[valid_idx]
plt.plot(theta_deg[valid_idx], ratio, 'g-', label='Ratio (TIS / SM)')
plt.xlabel('Scattering Angle (degrees)')
plt.ylabel('Ratio')
plt.title('TIS Vacuum Texture (Eisenstein Modulation)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# ==========================================
# 5. NUMERICAL CHECK
# ==========================================
print(f"--- Singularity Resolution Check (theta = 0.1 deg) ---")
print(f"Standard Model Cross Section: {sigma_sm[0]:.4e}")
print(f"TIS Cross Section:            {sigma_tis[0]:.4e}")
print(f"Ratio (TIS/SM):               {sigma_tis[0]/sigma_sm[0]:.4e}")
print(f"TIS Alpha used: 1/{1/ALPHA_TIS:.4f}")