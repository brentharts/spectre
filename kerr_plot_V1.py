#Kerr Plot V1
import numpy as np; import matplotlib.pyplot as plt; from scipy.constants import G, c
# --- 1. TIS FUNDAMENTAL CONSTANTS --- # Derived in Sections 14-16
K_TILE = 13.0       # Vacuum Geometry
C_PHI = 3.0         # Topological Charge (Source of Rotation/Momentum)
C_FRICTION = 34.0   # Geometric Friction (Source of Stability)
# The Friction Ratio (34/13) prevents the singularity
FRICTION_RATIO = C_FRICTION / K_TILE 
# --- 2. KERR BLACK HOLE PARAMETERS --- # Mass of the black hole (e.g., 100 Solar Masses)
M = 6.73317655e26 # kg
# Spin parameter 'a' (0 to 1, dimensionless) # a = J / (M*c). In geometric units a_geo = a * GM/c^2
spin_param = 0.95 
# Calculate Length Scales
Rs = (2 * G * M) / c**2  # Schwarzschild Radius
a_geo = spin_param * (G * M) / c**2 # Geometric Spin Radius
# The Ring Singularity is located at r=0 in Boyer-Lindquist, # but physically it corresponds to a ring of radius 'a_geo'.
print(f"Black Hole Mass: {M:.2e} kg"); print(f"Event Horizon (Outer): ~{Rs/2 + np.sqrt((Rs/2)**2 - a_geo**2):.2f} m")
print(f"GR Ring Singularity Radius (a): {a_geo:.4f} meters")
# --- 3. TIS SCALING --- # The TIS friction operates at the microscopic tiling scale. 
# We assume the friction becomes dominant at the same relative scale as the Schwarzschild test.
# This scale factor represents the 'thickness' of the boundary layer.
scale_factor = Rs * 1e-6 
# --- 4. DYNAMICS ON THE EQUATORIAL PLANE --- # We scan radial distances 'r' approaching the ring. 
# In Kerr metric, gravity is complex, but near the ring on the equator, # we can model the effective potential.
r_scan = np.logspace(np.log10(scale_factor), np.log10(Rs), 1000)
def kerr_radial_acceleration(r, M, a):
    """Classical GR Acceleration near the Ring (Simplified model). In Boyer-Lindquist, the ring is at r=0, but represents a physical ring of radius 'a'.
    Gravity pulls INWARD toward r=0 (the ring). Centrifugal force pushes OUTWARD."""
    Sigma = r**2 # On equator, theta=pi/2, so Sigma = r^2
    # This is a simplified Newtonian-like effective potential derivative for visualization # The key is that GR gravity diverges as 1/r^n near the ring.
    accel_gr = -(G * M * r) / (r**2 + a**2)**(1.5) # Simplified effective force toward ring center
    # Actually, closer to the ring (r->0), the term dominates. # Let's use the potential V_eff ~ -GM/r roughly for the collapse component.
    accel_gr = -(G * M) / (r**2) 
    return accel_gr
# A. Standard Kerr Gravity (Collapse to Ring)
accel_gravity = kerr_radial_acceleration(r_scan, M, a_geo)
# B. TIS Geometric Friction (Repulsion) 
# The friction scales with the Topological Charge (3) and Friction Ratio (34/13). # In Kerr, the Topological Charge (3) is also the source of the spin 'a'.
# Force ~ (GM/r^2) * Friction * (Charge/r)
accel_friction = (G * M / r_scan**2) * (FRICTION_RATIO * (C_PHI * scale_factor / r_scan))
# C. Unified TIS-Kerr Dynamics
accel_net = accel_gravity + accel_friction
# --- 5. FIND THE STABLE TORUS --- # Find where acceleration flips signs
torus_idx = np.where(np.diff(np.sign(accel_net)))[0]
if len(torus_idx) > 0:
    r_torus = r_scan[torus_idx[0]]
    print("-" * 60); print(f"TIS STABLE TORUS DETECTED"); print(f"Standard GR predicts Singularity at Thickness = 0"); 
    print(f"TIS predicts Stable Torus Thickness = {r_torus:.4e} meters"); print("-" * 60)
else:
    r_torus = 0
    print("No stable torus found (adjust scale).")
# --- 6. VISUALIZATION ---
plt.figure(figsize=(12, 7))
# Plot Gravity vs Friction
plt.loglog(r_scan, np.abs(accel_gravity), 'r--', label='Standard Kerr Gravity (Collapse)', alpha=0.5); 
plt.loglog(r_scan, accel_friction, 'g--', label='TIS Friction (34/13)', alpha=0.5)
plt.loglog(r_scan, np.abs(accel_net), 'b-', linewidth=2, label='Unified Net Acceleration')
# Annotate the Torus
if r_torus > 0:
    plt.axvline(x=r_torus, color='b', linestyle=':', alpha=0.8); 
    plt.plot(r_torus, np.abs(accel_gravity[torus_idx[0]]), 'bo', markersize=10, label='Stable Torus Surface')
    plt.text(r_torus, np.abs(accel_gravity[torus_idx[0]])*3,  f'  Stable Torus\n  Thickness ~ {r_torus:.2e} m', color='blue', verticalalignment='bottom')
plt.xlabel('Distance from Ring Center (r) [meters]'); plt.ylabel('Acceleration Magnitude (m/s²)'); plt.title(f'TIS Resolution of the Kerr Ring Singularity\n(Spin a = {spin_param})')
plt.legend(); plt.grid(True, which="both", ls="-", alpha=0.2); plt.gca().invert_xaxis() # Looking inward toward the ring
plt.show()