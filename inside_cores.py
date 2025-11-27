import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c
from math import pi

# --- 1. TIS GLOBAL CONSTANTS ---
K_TILE = 13.0        # Vacuum Geometry
C_PHI = 3.0          # Topological Charge
C_FRICTION = 34.0    # Geometric Friction (F9)
FRICTION_RATIO = C_FRICTION / K_TILE 

# Physics Scale Factors (Micro-scale simulation)
# We simulate at the scale of the "Exceptional Divisor" (The Resolved Core)
SCALE_L = 1e-6       # Characteristic Length Scale
MASS_SCALE = 1e26    # Mass of the cores

# --- 2. MODULAR MATH ENGINE (Eisenstein Series) ---

def divisor_sum_sigma_k(n, k):
    """Generalized divisor function."""
    if n == 0: return 0
    s = 0
    for d in range(1, int(np.sqrt(n)) + 1):
        if n % d == 0:
            d1 = d; d2 = n // d
            s += d1**k
            if d1 != d2: s += d2**k
    return s

def eisenstein_e6_q_expansion(q, N_terms=20):
    """
    Calculates E6(q) for a field of complex q values.
    Inputs: q (numpy array of complex numbers, must be |q| < 1)
    """
    # Initialize E6 with the constant term
    E6 = np.ones_like(q, dtype=complex)
    C = -504.0 
    
    # Pre-calculate powers of q to speed up summation
    # We perform the summation iteratively to handle the array structure
    current_q_pow = q.copy()
    
    for n in range(1, N_terms + 1):
        sigma = divisor_sum_sigma_k(n, 5)
        # E6 += C * sigma * q^n
        term = C * sigma * current_q_pow
        E6 += term
        
        # Prepare next power (q^(n+1))
        current_q_pow *= q
        
    return E6

# --- 3. TIS PHYSICS ENGINE (Dual-Source Dynamics) ---

def calculate_tis_force(r_vec, m1, m2):
    """
    Calculates the force vector between two TIS cores.
    Includes:
    1. Hydrodynamic Gravity (Attractive - 1/r^2)
    2. Topological Geometric Friction (Repulsive - 34/13 * 1/r^3)
    """
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0: return np.array([0.0, 0.0])
    
    r_hat = r_vec / r_mag
    
    # A. Standard Gravity (Attractive)
    F_grav_mag = (G * m1 * m2) / (r_mag**2)
    
    # B. Geometric Friction (Repulsive)
    # The Friction Ratio (34/13) acts as the coupling constant for the repulsive term.
    # We scale it by a characteristic topological length (SCALE_L) to make it relevant at this zoom level.
    F_fric_mag = (G * m1 * m2 / r_mag**2) * (FRICTION_RATIO * (SCALE_L / r_mag))
    
    # Net Force (Gravity pulls in (-), Friction pushes out (+))
    # Since r_vec points from 1 to 2, force on 1 is towards 2 (+)
    F_net_mag = F_grav_mag - F_fric_mag
    
    return F_net_mag * r_hat

# --- 4. FIELD GENERATION (The "Texture") ---

def generate_interacting_field(grid_size, range_lim, core1_pos, core2_pos, core_radius):
    """
    Generates the superimposed Modular Potential (q) of two interacting cores.
    """
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Convert core positions to complex
    z1 = core1_pos[0] + 1j * core1_pos[1]
    z2 = core2_pos[0] + 1j * core2_pos[1]
    
    # --- Define the Modular Potential Field ---
    # We model the core as a "Source" of modular complexity.
    # The field 'q' decays from the center.
    # q = exp( -|z - center|^2 / width ) * Phase_Twist
    
    # Core 1 Field (Clockwise Twist)
    dist1 = np.abs(Z - z1)
    # Add a topological twist (vortex) to the field phase
    phase1 = np.exp(1j * np.angle(Z - z1) * 3) # Charge 3 (C_PHI) winding
    q1 = np.exp(-(dist1**2) / (2 * core_radius**2)) * 0.95 * phase1
    
    # Core 2 Field (Counter-Clockwise Twist - Antimatter/Mirror Dual?)
    # Let's give it the same chirality for now (Matter-Matter collision)
    dist2 = np.abs(Z - z2)
    phase2 = np.exp(1j * np.angle(Z - z2) * 3) 
    q2 = np.exp(-(dist2**2) / (2 * core_radius**2)) * 0.95 * phase2
    
    # --- SUPERPOSITION (Interference) ---
    # The fields interfere. We must clamp the result to |q| < 1 for the E6 series.
    q_total = q1 + q2
    
    # Soft Clamp to unit disk to prevent divergence in Eisenstein series
    # This represents the Nariai Limit (Saturation of Information)
    mag = np.abs(q_total)
    mask = mag >= 0.99
    q_total[mask] = q_total[mask] / mag[mask] * 0.99
    
    return X, Y, q_total

# --- 5. MAIN SIMULATION & VISUALIZATION ---

def simulate_and_visualize():
    # Setup
    GRID_SIZE = 600
    VIEW_RANGE = 4.0 * SCALE_L
    CORE_RADIUS = 0.6 * SCALE_L
    
    # Initial State: Two cores approaching each other
    # Slightly offset in Y to create a spiral/spin interaction
    pos1 = np.array([-1.5 * SCALE_L, -0.2 * SCALE_L])
    pos2 = np.array([ 1.5 * SCALE_L,  0.2 * SCALE_L])
    
    # Generate the Topological Field (The "q" parameter)
    X, Y, q_field = generate_interacting_field(GRID_SIZE, VIEW_RANGE, pos1, pos2, CORE_RADIUS)
    
    # Compute the Modular Texture (Eisenstein E6)
    # This represents the "Lattice Structure" of the vacuum
    print("Calculating Modular Texture (Eisenstein E6)... this may take a moment.")
    E6_field = eisenstein_e6_q_expansion(q_field, N_terms=20)
    
    # Extract Phase (The "Tiling Orientation")
    # In TIS, the Phase of E6 represents the local orientation of the 13-sided tiles.
    E6_phase = np.angle(E6_field)
    
    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111)
    
    # 1. Plot the Modular Interference Texture
    # Using 'twilight' colormap to show the cyclic nature of the phase
    im = ax.imshow(E6_phase, extent=[-VIEW_RANGE, VIEW_RANGE, -VIEW_RANGE, VIEW_RANGE], 
                   origin='lower', cmap='twilight', alpha=0.9)
    
    # 2. Overlay Field Streamlines (The "Hopfion Fluid" Flow)
    # We derive a flow field from the gradient of the potential magnitude
    # This visualizes the σ_hydrodynamic stress
    field_mag = np.abs(q_field)
    Dy, Dx = np.gradient(field_mag)
    # Rotate gradients 90 degrees to get flow lines (vortex-like)
    #ax.streamplot(X, Y, -Dy, Dx, color='white', density=1.2, linewidth=0.5, arrowsize=0.5, alpha=0.3)
    # Change color='white' to RGBA (1, 1, 1, 0.3) and remove the alpha=0.3 argument
    ax.streamplot(X, Y, -Dy, Dx, color=(1, 1, 1, 0.3), density=1.2, linewidth=0.5, arrowsize=0.5)

    # 3. Mark the Cores (The Exceptional Divisors)
    core_circle1 = plt.Circle(pos1, CORE_RADIUS/3, color='black', ec='white', lw=2, zorder=10)
    core_circle2 = plt.Circle(pos2, CORE_RADIUS/3, color='black', ec='white', lw=2, zorder=10)
    ax.add_patch(core_circle1)
    ax.add_patch(core_circle2)
    
    # 4. Annotation
    plt.title("TIS Core Interaction: Modular Interference of $E_6(q)$", color='white', fontsize=16, pad=20)
    ax.text(0, VIEW_RANGE*0.9, "Constructive Interference = Lattice Locking", color='white', ha='center', fontsize=10, alpha=0.7)
    ax.text(0, -VIEW_RANGE*0.9, "Friction Barrier (34/13) Active", color='white', ha='center', fontsize=10, alpha=0.7)
    
    # Aesthetics
    ax.set_xlabel("Spatial Dimension X (Topological Scale)", color='gray')
    ax.set_ylabel("Spatial Dimension Y", color='gray')
    ax.tick_params(axis='both', colors='gray')
    for spine in ax.spines.values(): spine.set_edgecolor('gray')
    
    # Colorbar for Phase
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Modular Phase (Tiling Orientation)', color='white', rotation=270, labelpad=15)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.show()

if __name__ == "__main__":
    simulate_and_visualize()