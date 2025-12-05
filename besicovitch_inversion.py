import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
import io
from PIL import Image

# ==========================================
# 1. TIS GLOBAL SETTINGS & CONSTANTS
# ==========================================
K_TILE = 13.0               # Einstein Monotile Constraint
C_FRICTION = 34.0           # Geometric Friction (Fibonacci)
FRICTION_RATIO = C_FRICTION / K_TILE  # Lambda_F (34/13)

# ==========================================
# 2. MODULAR MATH ENGINE (Texture for Dynamics)
# ==========================================

def divisor_sum_sigma_k(n, k):
    if n == 0: return 0
    s = 0
    for d in range(1, n + 1):
        if n % d == 0:
            s += d**k
    return s

def eisenstein_e6_q_expansion(q, N_terms=6):
    E6 = np.ones_like(q, dtype=complex)
    C = -504.0
    current_q_pow = q.copy()
    for n in range(1, N_terms + 1):
        sigma = divisor_sum_sigma_k(n, 5)
        E6 += (C * sigma) * current_q_pow
        current_q_pow *= q
    return E6

# ==========================================
# 3. TIS PHYSICS ENGINE (Collision Dynamics)
# ==========================================

def calculate_tis_force(r_vec):
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 0.02: r_mag = 0.02 
    r_hat = r_vec / r_mag
    
    ALPHA_G = 0.025 
    
    F_grav = ALPHA_G / (r_mag**2)
    F_fric = (ALPHA_G / r_mag**2) * (FRICTION_RATIO * (0.15 / r_mag))
    
    F_net = F_grav - F_fric
    return F_net * r_hat

# --- Core Dynamics ---
class TISSystem:
    def __init__(self, speed=0.15):
        self.pos1 = np.array([-2.0, 0.0])
        self.pos2 = np.array([ 2.0, 0.0])
        self.vel1 = np.array([ speed, 0.0])
        self.vel2 = np.array([-speed, 0.0])
        self.collision_count = 0
        self.was_repelling = False # State flag to detect bounce

    def step(self, dt):
        r_vec = self.pos2 - self.pos1
        
        # Check if approaching (negative relative velocity)
        relative_vel = np.dot((self.vel2 - self.vel1), r_vec / np.linalg.norm(r_vec))
        is_approaching = relative_vel < 0
        
        force = calculate_tis_force(r_vec)
        self.vel1 *= 0.995
        self.vel2 *= 0.995
        self.vel1 += force * dt
        self.vel2 -= force * dt
        self.pos1 += self.vel1 * dt
        self.pos2 += self.vel2 * dt

        # Check for bounce (Switch from approaching to repelling)
        new_r_vec = self.pos2 - self.pos1
        new_relative_vel = np.dot((self.vel2 - self.vel1), new_r_vec / np.linalg.norm(new_r_vec))
        is_repelling = new_relative_vel > 0
        
        # If it was approaching and is now repelling, a bounce occurred.
        if is_approaching and is_repelling and np.linalg.norm(r_vec) < 0.6:
            # Check to avoid multiple counts per bounce sequence
            if not self.was_repelling:
                 self.collision_count += 1
        
        self.was_repelling = is_repelling
        
# --- Field Generation with Dynamic Besicovitch and Sign Inversion ---
def generate_interacting_field(grid_size, range_lim, core1_pos, core2_pos, core_radius, current_dist, collision_count):
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    z1 = core1_pos[0] + 1j * core1_pos[1]
    z2 = core2_pos[0] + 1j * core2_pos[1]
    
    # ------------------------------------------------------------------
    # DYNAMIC BESICOVITCH SCALING LOGIC
    # ------------------------------------------------------------------
    D_start = 3.5 
    D_end = 0.5   
    
    besicovitch_scale = 0.0
    if current_dist < D_start:
        scale_val = (D_start - current_dist) / (D_start - D_end)
        besicovitch_scale = np.clip(scale_val, 0.0, 1.0)
    
    # Sign Inversion (NEW): Flips every collision to model TIS T-reversal
    sign_inversion = (-1) ** collision_count
    
    # ------------------------------------------------------------------

    # --- Integration of Pentagonal/Besicovitch Constraints ---
    def get_tis_field_q(Z_field, z_center, radius, b_scale, sign_inv):
        """Generates a field with dynamically scaled and signed Pentagonal Distortion."""
        dist = np.abs(Z_field - z_center)
        angle = np.angle(Z_field - z_center)
        
        # Pentagonal Symmetry Modulation: Sign is now inverted based on collision count
        MOD_STRENGTH = 0.25 * b_scale * sign_inv
        pent_modulation = 1.0 + MOD_STRENGTH * np.cos(5 * angle) 
        
        # Effective Distance: Warped by the TIS Metric
        dist_eff = dist / pent_modulation
        
        phase = np.exp(1j * angle * 3)
        
        q = np.exp(-(dist_eff**2) / (2 * radius**2)) * 0.98 * phase
        return q

    # --- Core 1 Field ---
    q1 = get_tis_field_q(Z, z1, core_radius, besicovitch_scale, sign_inversion) 
    
    # --- Core 2 Field ---
    q2 = get_tis_field_q(Z, z2, core_radius, besicovitch_scale, sign_inversion) 
    
    q_total = q1 + q2
    
    mag = np.abs(q_total)
    mask = mag >= 0.99
    q_total[mask] = q_total[mask] / mag[mask] * 0.99
    
    return X, Y, q_total

# ==========================================
# 4. DYNAMIC VISUALIZATION GENERATION
# ==========================================

# --- Simulation Parameters ---
DT = 0.05
GRID_SIZE = 300 
VIEW_RANGE = 1.8
CORE_RADIUS = 0.35
FRAMES = 100 # Increased frames to show bounce and sign flip
RENDER_SKIP = 10
sim = TISSystem(speed=0.3)

# --- Plotting Setup ---
fig, ax_dyn = plt.subplots(figsize=(7, 7), facecolor='black')

# Store frames for GIF
frames = []

# --- Dynamic Collision Plot (Animated) ---
for i in range(FRAMES):
    # Physics step
    for _ in range(RENDER_SKIP):
        sim.step(DT)
    
    ax_dyn.clear()
    
    dist = np.linalg.norm(sim.pos1 - sim.pos2)
    
    # State tracking for title
    if dist < 0.4: 
        state_text = "** MAX BESICOVITCH STRESS **"
    elif sim.vel1[0] < 0: 
        state_text = "Repulsion ($\mathcal{T}_{TIS}$ Inverted)"
    else: 
        state_text = "Approaching"
    
    # Call to Field Generator with current_dist and collision_count
    X, Y, q_field = generate_interacting_field(
        GRID_SIZE, VIEW_RANGE, sim.pos1, sim.pos2, CORE_RADIUS, dist, sim.collision_count
    )
    
    E6_field = eisenstein_e6_q_expansion(q_field, N_terms=6)
    E6_phase = np.angle(E6_field)
    
    # Plot Texture
    ax_dyn.imshow(E6_phase, extent=[-VIEW_RANGE, VIEW_RANGE, -VIEW_RANGE, VIEW_RANGE],
              origin='lower', cmap='twilight', alpha=0.9)
    
    # Cores
    core_color = 'cyan' if sim.collision_count % 2 == 0 else 'red'
    
    c1 = plt.Circle(sim.pos1, 0.06, color='black', ec=core_color, lw=2, zorder=10)
    c2 = plt.Circle(sim.pos2, 0.06, color='black', ec=core_color, lw=2, zorder=10)
    ax_dyn.add_patch(c1)
    ax_dyn.add_patch(c2)
    
    # Dynamic text to show T-reversal state
    b_sign = "Positive" if sim.collision_count % 2 == 0 else "Negative (Inverted)"
    
    ax_dyn.set_xlim(-VIEW_RANGE, VIEW_RANGE)
    ax_dyn.set_ylim(-VIEW_RANGE, VIEW_RANGE)
    ax_dyn.axis('off')
    ax_dyn.set_title(f"TIS Collision: Dynamic Besicovitch Inversion\n{state_text} | $T_{{TIS}}$ Sign: {b_sign} (Count: {sim.collision_count})", color='white', fontsize=10)

    # Capture frame
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='black')
    buf.seek(0)
    frames.append(Image.open(buf))

# Save GIF
tmp = '/tmp/tis_dynamic_besicovitch_inversion.gif'
frames[0].save(tmp, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
print("Dynamic Besicovitch Inversion visualization saved: /tmp/tis_dynamic_besicovitch_inversion.gif")

import subprocess
cmd = ['gifsicle', '-i', tmp, '-O3', '--colors=64', '-o', 'tis_dynamic_besicovitch_inversion.gif']
print(cmd)
subprocess.check_call(cmd)