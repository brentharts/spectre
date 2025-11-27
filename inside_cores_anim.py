import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os, sys, subprocess

# --- 1. TIS GLOBAL SETTINGS ---
K_TILE = 13.0        
C_PHI = 3.0          
C_FRICTION = 34.0    
FRICTION_RATIO = C_FRICTION / K_TILE 

# --- 2. MODULAR MATH ENGINE (Texture) ---

def divisor_sum_sigma_k(n, k):
    if n == 0: return 0
    s = 0
    for d in range(1, n + 1):
        if n % d == 0:
            s += d**k
    return s

def eisenstein_e6_q_expansion(q, N_terms=12): 
    E6 = np.ones_like(q, dtype=complex)
    C = -504.0 
    current_q_pow = q.copy()
    for n in range(1, N_terms + 1):
        sigma = divisor_sum_sigma_k(n, 5)
        E6 += (C * sigma) * current_q_pow
        current_q_pow *= q
    return E6

# --- 3. TIS PHYSICS ENGINE (Collision Dynamics) ---

def calculate_tis_force(r_vec):
    """
    Calculates force. 
    In a head-on collision, the 1/r^3 term becomes the dominant 
    "hard surface" of the vacuum.
    """
    r_mag = np.linalg.norm(r_vec)
    
    # Nariai Limit Safety: Prevent division by zero
    if r_mag < 0.02: r_mag = 0.02
    
    r_hat = r_vec / r_mag
    
    # High interaction strength for visual impact
    ALPHA_G = 0.025  
    
    # A. Gravity (Attractive)
    F_grav = ALPHA_G / (r_mag**2)
    
    # B. Geometric Friction (Repulsive 34/13)
    # We tighten the range (0.15) so they get very close before bouncing
    F_fric = (ALPHA_G / r_mag**2) * (FRICTION_RATIO * (0.15 / r_mag))
    
    F_net = F_grav - F_fric
    return F_net * r_hat

# --- 4. FIELD GENERATION (The Shockwave) ---

def generate_interacting_field(grid_size, range_lim, core1_pos, core2_pos, core_radius):
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    z1 = core1_pos[0] + 1j * core1_pos[1]
    z2 = core2_pos[0] + 1j * core2_pos[1]
    
    # --- Core 1 (Twist +3) ---
    dist1 = np.abs(Z - z1)
    phase1 = np.exp(1j * np.angle(Z - z1) * 3) 
    q1 = np.exp(-(dist1**2) / (2 * core_radius**2)) * 0.98 * phase1
    
    # --- Core 2 (Twist +3) ---
    dist2 = np.abs(Z - z2)
    # Note: Even with same charge, the interference pattern in the center
    # becomes extremely complex due to the phase cancellation/addition.
    phase2 = np.exp(1j * np.angle(Z - z2) * 3) 
    q2 = np.exp(-(dist2**2) / (2 * core_radius**2)) * 0.98 * phase2
    
    # Superposition
    q_total = q1 + q2
    
    # Hard Clamp at 0.99 to represent the Nariai Information Saturation
    mag = np.abs(q_total)
    mask = mag >= 0.99
    q_total[mask] = q_total[mask] / mag[mask] * 0.99
    
    return X, Y, q_total

# --- 5. ANIMATION ENGINE ---

class TISSystem:
    def __init__(self, speed=0.15):
        # Initial Positions: HEAD ON on X-axis
        # Spaced out to allow acceleration
        self.pos1 = np.array([-2, 0.0])
        self.pos2 = np.array([ 2, 0.0])
        
        # Initial Velocities: Fast approach
        self.vel1 = np.array([ speed, 0.0]) 
        self.vel2 = np.array([-speed, 0.0])

    def step(self, dt):
        r_vec = self.pos2 - self.pos1
        
        # Force on 1 towards 2
        force = calculate_tis_force(r_vec)
        
        # DAMPING (Simulating Energy Loss/Bremsstrahlung at the shock)
        # This allows them to slow down slightly after the bounce, 
        # simulating transfer of energy to the topological field.
        self.vel1 *= 0.995
        self.vel2 *= 0.995
        
        # Update Dynamics
        self.vel1 += force * dt
        self.vel2 -= force * dt
        
        self.pos1 += self.vel1 * dt
        self.pos2 += self.vel2 * dt

def generate_collision():
    # Config
    SPEED = 0.15
    mode = 'full'
    if '--fast' in sys.argv:
        mode = 'fast'
        FRAMES = 120
        SPEED = 0.8
    else:
        FRAMES = 1500
    DT = 0.05
    GRID_SIZE = 300 
    VIEW_RANGE = 1.8
    CORE_RADIUS = 0.35
    DPI = 80
    if '--lowres' in sys.argv:
        DPI = 40
    elif '--hires' in sys.argv:
        DPI = 160
    output_dir = "tis_collision_frames_%s_%s" % (DPI, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sim = TISSystem( speed=SPEED )
    
    print(f"Simulating TIS Head-On Collision ({FRAMES} frames)...")
    
    for i in range(FRAMES):
        sim.step(DT)
        fname = f"{output_dir}/col_frame_{i:04d}.png"
        if os.path.isfile(fname): continue
        
        # Determine State for Title
        dist = np.linalg.norm(sim.pos1 - sim.pos2)
        state_text = "Approaching"
        if dist < 0.4: state_text = "** NARIAI LIMIT REACHED **"
        elif sim.vel1[0] < 0: state_text = "Repulsion (Geometric Friction)"
        
        # Generate Field
        X, Y, q_field = generate_interacting_field(
            GRID_SIZE, VIEW_RANGE, sim.pos1, sim.pos2, CORE_RADIUS
        )
        
        # Modular Texture
        E6_field = eisenstein_e6_q_expansion(q_field, N_terms=12)
        E6_phase = np.angle(E6_field)
        
        # Streamlines
        field_mag = np.abs(q_field)
        Dy, Dx = np.gradient(field_mag)
        
        # Plotting
        fig = plt.figure(figsize=(8, 8), facecolor='black')
        ax = fig.add_subplot(111)
        
        # Texture
        ax.imshow(E6_phase, extent=[-VIEW_RANGE, VIEW_RANGE, -VIEW_RANGE, VIEW_RANGE],
                  origin='lower', cmap='twilight', alpha=0.9)
        
        # Flow
        ax.streamplot(X, Y, -Dy, Dx, color=(1, 1, 1, 0.4), density=1.2, linewidth=0.6, arrowsize=0.6)
        
        # Cores
        # Color changes based on pressure (proximity)
        core_color = 'cyan'
        if dist < 0.5: core_color = 'red' # High Stress
        
        c1 = plt.Circle(sim.pos1, 0.06, color='black', ec=core_color, lw=2, zorder=10)
        c2 = plt.Circle(sim.pos2, 0.06, color='black', ec=core_color, lw=2, zorder=10)
        ax.add_patch(c1)
        ax.add_patch(c2)
        
        ax.set_xlim(-VIEW_RANGE, VIEW_RANGE)
        ax.set_ylim(-VIEW_RANGE, VIEW_RANGE)
        ax.axis('off')
        
        # Dynamic Title
        plt.title(f"Frame {i}: {state_text}\nDist: {dist:.3f} | Phase Interference", color='white', y=0.92, fontsize=10)
        
        # Save
        plt.savefig(fname, dpi=DPI, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        
        print(f"Rendered {fname} | Dist: {dist:.3f}", end='\r')

    print("\nCollision Simulation Complete.")
    print('conversion to gif')
    cmd = ['ffmpeg', '-y', '-i', 'col_frame_%04d.png', '-vf', 'palettegen', '/tmp/palette.png']
    print(cmd)
    subprocess.check_call(cmd, cwd=output_dir)
    cmd = ['ffmpeg', '-y', '-framerate', '10', '-i', 'col_frame_%04d.png', '-i', '/tmp/palette.png', '-filter_complex', '[0:v][1:v]paletteuse', '/tmp/output.gif']
    print(cmd)
    subprocess.check_call(cmd, cwd=output_dir)
    cmd = ['gifsicle', '-i', '/tmp/output.gif', '-O3', '--colors=64', '-o', '/tmp/comp.gif']
    print(cmd)
    subprocess.check_call(cmd)

if __name__ == "__main__":
    generate_collision()