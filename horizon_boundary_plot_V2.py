#Horizon Boundary Plot V2
import numpy as np; import matplotlib.pyplot as plt; from scipy.constants import G, c
# --- 1. SETUP TIS CONSTANTS ---
K_TILE = 13.0      # Vacuum Geometry
C_PHI = 3.0        # Topological Charge
C_FRICTION = 34.0  # Geometric Tension (F9)
FRICTION_RATIO = C_FRICTION / K_TILE 
# --- 2. SETUP BLACK HOLE ---
M = 6.73317655e26  # kg
Rs = (2 * G * M) / c**2
print(f"Black Hole Mass: {M:.2e} kg"); print(f"Schwarzschild Radius (Event Horizon): {Rs:.4f} meters")
# --- 3. SCALING CORRECTION ---
# The TIS effects operate at the fundamental tiling scale. # We scale the interaction so it becomes significant only deep inside the horizon. 
scale_factor = Rs * 1e-6   # Setting scale to 1 millionth of the horizon radius.
# --- 4. DEFINE THE SCANNING DOMAIN ---
# We scan from the Horizon (Rs) down to near-Planck scales # Logspace allows us to zoom in exponentially deep
r = np.logspace(np.log10(Rs), np.log10(scale_factor / 10), 10000) 
# --- 5. CALCULATE DYNAMICS ---
a_gravity = -(G * M) / (r**2)
# TIS Repulsive Potential # Notice we use the corrected scale_factor
a_friction = (G * M / r**2) * (FRICTION_RATIO * (C_PHI * scale_factor / r))
# Unified Acceleration
a_tis = a_gravity + a_friction
# --- 6. LOCATE THE NARIAI CORE --- # Find where the net acceleration flips from Inward (-) to Outward (+)
core_idx = np.where(np.diff(np.sign(a_tis)))[0]
if len(core_idx) > 0:
    r_core = r[core_idx[0]]
    print("-" * 50)
    print(f"TIS NON-SINGULAR CORE FOUND"); print(f"Radius: {r_core:.4e} meters"); print(f"Ratio to Horizon: {r_core/Rs:.4e}")
    print("-" * 50)
else:
    r_core = 0; print("Core not resolved. Adjust scan depth.")
# --- 7. VISUALIZATION ---
plt.figure(figsize=(12, 7))
# We plot the absolute values on a log-log scale to see the crossover clearly
plt.loglog(r, np.abs(a_gravity), 'r--', label='Standard Gravity (Collapse)', alpha=0.5)
plt.loglog(r, a_friction, 'g--', label='TIS Friction (Repulsion)', alpha=0.5)
plt.loglog(r, np.abs(a_tis), 'b-', linewidth=2, label='Unified TIS Net Acceleration')
# Mark the Core
if r_core > 0:
    plt.plot(r_core, np.abs(a_gravity[core_idx[0]]), 'bo', markersize=12, label='TIS Stability Core')
    plt.axvline(x=r_core, color='b', linestyle=':', alpha=0.6)
    plt.text(r_core, np.abs(a_gravity[core_idx[0]])*2, f'  Stable Core\n  r = {r_core:.2e} m', color='blue')
plt.axvline(x=Rs, color='k', linestyle='-', alpha=0.3, label='Event Horizon'); plt.xlabel('Radial Distance from Center (meters)')
plt.ylabel('Magnitude of Acceleration (m/s²)'); plt.title('Inside the Black Hole: Locating the TIS Nariai Core')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.gca().invert_xaxis() # We look from Horizon (left) inward to Center (right)
plt.show()