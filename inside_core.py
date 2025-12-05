import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c
from math import pi

# =====================================================================
# --- SECTION 1: Modular Forms / Eisenstein Series (Quantum Symmetry) ---
# =====================================================================

def divisor_sum_sigma_k(n, k):
    """Calculates the generalized divisor function sigma_k(n) = sum_{d|n} d^k."""
    if n == 0:
        return 0
    s = 0
    for d in range(1, int(np.sqrt(n)) + 1):
        if n % d == 0:
            d1 = d
            d2 = n // d
            s += d1**k
            if d1 != d2:
                s += d2**k
    return s

def eisenstein_e6_q_expansion(q, N_terms=50):
    """
    Calculates the normalized Eisenstein series E6(tau) using its q-expansion.
    
    E6(tau) = 1 - 504 * sum_{n=1}^\infty sigma_5(n) * q^n
    where q = exp(2*pi*i*tau). 
    
    Note: q is expected to be a numpy array for vectorized calculation.
    """
    E6 = 1.0 + 0j
    C = -504

    for n in range(1, N_terms + 1):
        sigma5_n = divisor_sum_sigma_k(n, 5)
        term = C * sigma5_n * (q**n)
        E6 = np.add(E6, term)
    
    return E6

def plot_eisenstein_series(title_prefix="TIS Foundation"):
    """
    Generates two plots: one for the real part and one for the imaginary part 
    of the Eisenstein series E6 on the unit disk.
    """
    N_TERMS = 50 
    GRID_POINTS = 500
    re_q = np.linspace(-1, 1, GRID_POINTS)
    im_q = np.linspace(-1, 1, GRID_POINTS)
    Re_Q, Im_Q = np.meshgrid(re_q, im_q)
    Q = Re_Q + 1j * Im_Q
    
    E6_results = np.full((GRID_POINTS, GRID_POINTS), np.nan + 1j * np.nan, dtype=complex)
    mask = np.abs(Q) < 1.0

    E6_results[mask] = eisenstein_e6_q_expansion(Q[mask], N_terms=N_TERMS)
    
    E6_real = E6_results.real
    E6_imag = E6_results.imag
    extent = [-1, 1, -1, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"{title_prefix}: Modular Symmetry of $E_6(q)$ on the Unit Disk ($q=e^{{2\\pi i \\tau}}$)", fontsize=16)

    # --- Plot 1: Real Part ---
    im1 = axes[0].imshow(E6_real, origin='lower', extent=extent, cmap='seismic')
    axes[0].set_title('Real Part of $E_6(q)$')
    axes[0].set_xlabel('Re(q)')
    axes[0].set_ylabel('Im(q)')
    plt.colorbar(im1, ax=axes[0], label='Value (Re)')

    # --- Plot 2: Imaginary Part ---
    im2 = axes[1].imshow(E6_imag, origin='lower', extent=extent, cmap='RdYlBu')
    axes[1].set_title('Imaginary Part of $E_6(q)$')
    axes[1].set_xlabel('Re(q)')
    axes[1].set_ylabel('Im(q)')
    plt.colorbar(im2, ax=axes[1], label='Value (Im)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ====================================================================
# --- SECTION 2: TIS Metric (Singularity Resolution in Geometry) ---
# ====================================================================

def calculate_and_plot_tis_resolution():
    """Calculates TIS parameters (Rs, r_core) and plots the acceleration curves."""
    
    # --- 1. SETUP TIS CONSTANTS ---
    K_TILE = 13.0
    C_PHI = 3.0       # Topological Charge
    C_FRICTION = 34.0 # Geometric Tension (F9)
    FRICTION_RATIO = C_FRICTION / K_TILE

    # --- 2. SETUP BLACK HOLE ---
    M = 6.73317655e26 # kg (Mass approximately 338 solar masses)
    Rs = (2 * G * M) / c**2 # Schwarzschild Radius
    
    # --- 3. SCALING CORRECTION ---
    scale_factor = Rs * 1e-6 # Setting scale to 1 millionth of the horizon radius.

    # --- 4. DEFINE THE SCANNING DOMAIN ---
    r = np.logspace(np.log10(Rs), np.log10(scale_factor / 10), 10000)

    # --- 5. CALCULATE DYNAMICS ---
    a_gravity = -(G * M) / (r**2)
    a_friction = (G * M / r**2) * (FRICTION_RATIO * (C_PHI * scale_factor / r))
    a_tis = a_gravity + a_friction

    # --- 6. LOCATE THE NARIAI CORE ---
    core_idx = np.where(np.diff(np.sign(a_tis)))[0]

    if len(core_idx) > 0:
        r_core = r[core_idx[0]]
        # Print results to console as before
        print(f"--- TIS Singularity Resolution Analysis ---")
        print(f"Black Hole Mass: {M:.2e} kg")
        print(f"Schwarzschild Radius (Event Horizon): {Rs:.4f} meters")
        print("-" * 50)
        print(f"TIS NON-SINGULAR CORE FOUND")
        print(f"Radius: {r_core:.4e} meters")
        print(f"Ratio to Horizon: {r_core/Rs:.4e}")
        print("-" * 50)
    else:
        r_core = 0
        print("Core not resolved. Adjust scan depth.")
    
    # --- 7. PLOTTING (Log-Log) ---
    plt.figure(figsize=(12, 7))
    plt.loglog(r, np.abs(a_gravity), 'r--', label='Standard Gravity (Collapse)', alpha=0.5)
    plt.loglog(r, a_friction, 'g--', label='TIS Friction (Repulsion)', alpha=0.5)
    plt.loglog(r, np.abs(a_tis), 'b-', linewidth=2, label='Unified TIS Net Acceleration')

    # Mark the Core
    if r_core > 0:
        accel_at_core = np.abs(a_gravity[core_idx[0]])
        plt.plot(r_core, accel_at_core, 'bo', markersize=12, markerfacecolor='w', markeredgecolor='b', zorder=5, label='TIS Stability Core')
        plt.axvline(x=r_core, color='b', linestyle=':', alpha=0.6)
        plt.text(r_core, accel_at_core * 2, f' Stable Core\n r = {r_core:.2e} m', color='blue', verticalalignment='bottom', horizontalalignment='right')
        
    plt.axvline(x=Rs, color='k', linestyle='-', alpha=0.3, label='Event Horizon')
    plt.xlabel('Radial Distance from Center (meters)')
    plt.ylabel('Magnitude of Acceleration (m/s²)')
    plt.title('2. Inside the Black Hole: Locating the TIS Nariai Core (Log-Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.gca().invert_xaxis() # We look from Horizon (left) inward to Center (right)
    plt.show()
    
    return Rs, r_core # Return the calculated parameters for Section 3 plot

# =================================================================
# --- SECTION 3: Integration: Modular Core Geometry (Contextual View) ---
# =================================================================

def plot_modular_core_integration(Rs, r_core):
    """
    Visualizes a 2D cross-section of the TIS Black Hole, showing the core 
    and event horizon at the same scale (Core will appear as a point).
    """
    GRID_POINTS = 500
    # Set the extent of the plot to slightly larger than the Event Horizon
    extent_limit = Rs * 1.0001 
    x_coords = np.linspace(-extent_limit, extent_limit, GRID_POINTS)
    y_coords = np.linspace(-extent_limit, extent_limit, GRID_POINTS)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    Z = X + 1j * Y
    R = np.abs(Z)

    # Initialize the visualization data array
    E6_magnitude = np.full(Z.shape, np.nan)
    
    # 1. Define the Core Region Mask
    mask_inside_core = R < r_core
    
    # 2. Map the Core Region to the Modular Disk (|q| < 1)
    Q_core = Z[mask_inside_core] / r_core 
    
    # 3. Calculate E6(q) for the TIS Core
    E6_values_core = eisenstein_e6_q_expansion(Q_core.flatten(), N_terms=50)
    
    # 4. Fill the Core Region with the Modular Field Magnitude
    E6_magnitude[mask_inside_core] = np.abs(E6_values_core)

    # 5. Define the Transition Layer (r_core <= R < Rs)
    mask_transition = (R >= r_core) & (R < Rs)
    E6_magnitude[mask_transition] = 0.5 # Constant background value
    
    # Ensure all NaN values (outside Rs) are masked later
    E6_magnitude[R >= Rs] = np.nan

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- Plotting the Modular Color Field ---
    vmax_val = np.nanmax(E6_magnitude[mask_inside_core]) if np.any(mask_inside_core) else 1.0
    im = ax.imshow(E6_magnitude, origin='lower', extent=[-extent_limit, extent_limit, -extent_limit, extent_limit], 
                   cmap='magma',
                   norm=plt.Normalize(vmax=vmax_val, vmin=0))
                   
    # --- Overlay Geometric Boundaries ---
    theta = np.linspace(0, 2 * pi, 500)
    
    # 1. Event Horizon (Rs) - Boundary of Spacetime Distortion
    x_rs = Rs * np.cos(theta)
    y_rs = Rs * np.sin(theta)
    ax.plot(x_rs, y_rs, color='red', linewidth=3, linestyle='-', label=r'Event Horizon $R_s$')
    
    # 2. TIS Stable Core ($r_{core}$) - Boundary of Modular Vacuum
    x_core = r_core * np.cos(theta)
    y_core = r_core * np.sin(theta)
    # The core circle is too small to see, so we mark the center
    ax.plot(0, 0, 'o', color='yellow', markersize=3, zorder=6, label=r'TIS Stable Core $r_{core}$')
    
    # --- Aesthetics ---
    ax.set_title('3. Modular Core Contextual View: Core ($r_{core}$) vs Horizon ($R_s$)', fontsize=16)
    ax.set_xlabel('Spatial Coordinate X (m)')
    ax.set_ylabel('Spatial Coordinate Y (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label=r'Modular Field Magnitude $|E_6(Z/r_{core})|$ (Tiling State Density)')
    plt.show()

# =================================================================
# --- SECTION 4: Zoomed-In Modular Core Geometry (Texture View - PHASE) ---
# =================================================================

def plot_modular_core_zoom_view(Rs, r_core):
    """
    Visualizes the 2D cross-section zoomed in strictly on the TIS Stable Core 
    (R < r_core) to clearly display the E6 modular texture using the Phase (Argument).
    """
    if r_core == 0:
        print("Cannot zoom: r_core is zero.")
        return

    GRID_POINTS = 500
    # Set the extent to be slightly larger than the core radius to see the boundary clearly
    zoom_factor = 1.05
    extent_limit = r_core * zoom_factor 
    x_coords = np.linspace(-extent_limit, extent_limit, GRID_POINTS)
    y_coords = np.linspace(-extent_limit, extent_limit, GRID_POINTS)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    Z = X + 1j * Y
    R = np.abs(Z)

    # Initialize the visualization data array for COMPLEX values
    E6_values_full = np.full(Z.shape, np.nan + 1j * np.nan, dtype=complex)
    
    # 1. Define the Core Region Mask (Everything inside r_core)
    mask_inside_core = R <= r_core
    
    # 2. Map the Core Region to the Modular Disk (|q| < 1)
    Q_core = Z[mask_inside_core] / r_core 
    
    # 3. Calculate E6(q) for the TIS Core
    E6_values_core = eisenstein_e6_q_expansion(Q_core.flatten(), N_terms=50)
    
    # 4. Fill the Complex array
    E6_values_full[mask_inside_core] = E6_values_core
    
    # --- KEY CHANGE: Plotting the Phase (Argument) ---
    E6_phase = np.angle(E6_values_full)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- Plotting the Modular Phase Field ---
    # Using hsv (cyclic colormap) for phase visualization. Phase range is [-pi, pi]
    im = ax.imshow(E6_phase, origin='lower', extent=[-extent_limit, extent_limit, -extent_limit, extent_limit], 
                   cmap='hsv',
                   norm=plt.Normalize(vmin=-pi, vmax=pi))
                   
    # --- Overlay Geometric Boundaries ---
    theta = np.linspace(0, 2 * pi, 500)
    
    # TIS Stable Core ($r_{core}$)
    x_core = r_core * np.cos(theta)
    y_core = r_core * np.sin(theta)
    ax.plot(x_core, y_core, color='black', linewidth=3, linestyle='--', label=r'TIS Stable Core Boundary $r_{core}$')

    # Center 
    ax.plot(0, 0, 'o', color='white', markersize=6, zorder=6)
    
    # --- Aesthetics ---
    ax.set_title('4. Modular Core Zoom View: Internal Phase Structure Arg($E_6(q)$)', fontsize=16)
    # Use scientific notation for the small scale
    ax.set_xlabel(f'Spatial Coordinate X (m) [Scale: $\\pm${extent_limit:.2e} m]')
    ax.set_ylabel(f'Spatial Coordinate Y (m) [Scale: $\\pm${extent_limit:.2e} m]')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')
    
    # Colorbar for phase
    cbar = plt.colorbar(im, ax=ax, ticks=[-pi, -pi/2, 0, pi/2, pi], 
                        label=r'Modular Field Phase Arg($E_6(Z/r_{core})$)')
    cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    plt.show()


# =================================================================
# --- MAIN EXECUTION ---
# =================================================================

if __name__ == "__main__":
    
    # 1. Modular Symmetry Foundation
    plot_eisenstein_series()
    
    # 2. TIS Geometric Resolution Plot (Calculates and plots the core radius)
    Rs, r_core = calculate_and_plot_tis_resolution()
    
    # 3. Merged Visualization (Contextual - Core is a point on this scale)
    plot_modular_core_integration(Rs, r_core)
    
    # 4. Corrected: Zoomed Visualization (Focuses on the internal texture using Phase)
    plot_modular_core_zoom_view(Rs, r_core)
    
    print("\nFour visualizations generated, demonstrating the TIS framework:")
    print("1. Modular Symmetry (E6) - The foundation of the vacuum structure.")
    print("2. TIS Geometric Resolution (Log-Log) - Shows the deep core stabilization (Nariai Core).")
    print("3. Modular Core Contextual View - Confirms the structured vacuum resides deep within the Event Horizon.")
    print("4. Modular Core Zoom View - **CORRECTED**: Clearly displays the Arg($E_6(q)$) phase texture inside the stable core.")