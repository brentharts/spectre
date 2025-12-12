#Horizon Boundary Plot V1
import numpy as np; import matplotlib.pyplot as plt; from scipy.constants import G, c
# --- 1. SETUP TIS CONSTANTS --- # Derived from Section
K_TILE = 13.0      # The Vacuum Geometry
C_PHI = 3.0        # Topological Charge
C_FRICTION = 34.0  # Geometric Tension (F9)
# Ratio 34/13 derived in Section 16
FRICTION_RATIO = C_FRICTION / K_TILE 
# --- 2. SETUP BLACK HOLE PARAMETERS --- # Using the mass from our previous test
M = 6.73317655e26  # kg
Rs = (2 * G * M) / c**2  # Schwarzschild Radius
print(f"Black Hole Mass: {M:.2e} kg"); print(f"Schwarzschild Radius (Event Horizon): {Rs:.2f} meters")
# --- 3. DEFINE THE DOMAIN ---
# We plot from outside the horizon down to near-zero (The Singularity) # Using logspace to see the core clearly
r = np.logspace(np.log10(Rs * 2), np.log10(1.0), 1000) 
# --- 4. CALCULATE ACCELERATIONS --- # A. Standard Schwarzschild Gravity (Newtonian approx for trend visualization) 
# In GR, 'g' implies the spacetime curvature slope. # a_gr = -GM/r^2
a_gravity = -(G * M) / (r**2)
# B. TIS Geometric Friction # From Section 16: The Non-Linear Stress term (34/13) |psi|^2 
# This acts as a repulsive potential scaling with density (1/r^3 proxy) # Force ~ (GM/r^2) * (Friction_Ratio * Topological_Charge / r)
# We normalize the Topological Charge (3) by the Planck scale or unit length # for the effective potential. 
# For this macro-simulation, we scale it so the effect is visible relative to the Horizon (representing the 2D boundary layer).
scale_factor = 1.0 # This would theoretically be linked to Planck length L_p
a_friction = (G * M / r**2) * (FRICTION_RATIO * (C_PHI * scale_factor / r))
# C. Unified TIS Acceleration
a_tis = a_gravity + a_friction
# --- 5. FIND THE STABLE CORE --- # The core is where Acceleration = 0 (Gravity balanced by Friction) 
# Find index where acceleration flips from negative (inward) to positive (outward)
core_idx = np.where(np.diff(np.sign(a_tis)))[0]
if len(core_idx) > 0:
    r_core = r[core_idx[0]]; print(f"TIS Non-Singular Core Radius: {r_core:.4f} meters")
else:
    r_core = 0; print("Core too small to resolve in this range.")
plt.figure(figsize=(10, 6))
plt.plot(r, a_gravity, 'r--', label='Standard GR (Collapse to Singularity)', alpha=0.6); 
plt.plot(r, a_friction, 'g--', label='TIS Geometric Friction (34/13)', alpha=0.6); 
plt.plot(r, a_tis, 'b-', linewidth=2.5, label='Unified TIS Metric (Non-Singular)')
plt.axvline(x=Rs, color='k', linestyle=':', label=f'Event Horizon ({Rs:.0f} m)'); plt.axhline(y=0, color='k', linewidth=0.5)
if r_core > 0:
    plt.plot(r_core, 0, 'bo', markersize=10, label='TIS Stability Point'); plt.text(r_core, 0, f'  Core\n  {r_core:.2f} m', verticalalignment='bottom')
plt.xscale('log'); plt.xlabel('Radial Distance (meters) [Log Scale]'); plt.ylabel('Acceleration (m/s²)'); 
plt.title('TIS Resolution of the Black Hole Singularity\nGeometric Friction (34/13) vs Gravity')
plt.legend(); plt.grid(True, which="both", ls="-", alpha=0.2); plt.show()