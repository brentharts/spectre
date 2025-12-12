#Decay of Dark Energy Plot V1
import numpy as np; import matplotlib.pyplot as plt; from scipy.constants import golden as phi
# --- 1. TIS CONSTANTS & PARAMETERS ---
K_TILE = 13.0
C_PHI = 3.0
# Decay Rate Axiom from Section 15 # C_decay = (13/3) * ln(phi)
C_DECAY = (K_TILE / C_PHI) * np.log(phi)
# Time Parameters (Gyr) derived in previous turn
T_HUBBLE = 13.8       # Visible Age
T_NOW_TIS = 259.13    # True Informational Age
T_FINAL = 1554.8      # Nariai Limit (Total Lifespan)
# Current Observed Dark Energy Density (eV^4)
RHO_OBSERVED = 3.7e-11
# --- 2. SIMULATION DOMAIN --- # We simulate from the TIS Beginning (t=0) to the Nariai Limit (t_final)
t_gyr = np.linspace(0, T_FINAL, 1000)
# Normalize time to the "Lifespan Scale" (tau) # tau goes from 0 to 1
tau = t_gyr / T_FINAL
# --- 3. CALCULATE DENSITY EVOLUTION ---
# Formula: rho(t) = rho_initial * exp( -C_decay * tau ) # First, calibrate rho_initial so that the curve hits RHO_OBSERVED at T_NOW 
# rho_observed = rho_initial * exp( -C_decay * (t_now / t_final) )
tau_now = T_NOW_TIS / T_FINAL; rho_initial = RHO_OBSERVED / np.exp(-C_DECAY * tau_now)
# Generate the full density curve
rho_lambda = rho_initial * np.exp(-C_DECAY * tau)
# --- 4. CALCULATE w PARAMETER DEVIATION ---
# TODO: The deviation |1+w| scales with the proximity to the end state? # Or simply use the linear time scaling derived previously: 
# Deviation was 0.2 at t=259. Let's assume a linear decay of tension manifests as w approaching -1?
# FOR NOW, let's just mark the current w on the plot for reference.
# --- 5. PLOT ---
fig, ax1 = plt.subplots(figsize=(12, 7))
# Plot Density
color = 'tab:blue'
ax1.set_xlabel('TIS Cosmic Age (Billions of Years)', fontsize=12); 
ax1.set_ylabel(r'Vacuum Energy Density $\rho_{\Lambda}$ ($eV^4$)', color=color, fontsize=12)
ax1.plot(t_gyr, rho_lambda, color=color, linewidth=2.5, label=r'TIS Vacuum Decay'); 
ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, alpha=0.3)
# --- 6. ANNOTATIONS ---
# A. The Beginning (Tiling Initialization)
ax1.plot(0, rho_initial, 'ko'); ax1.text(10, rho_initial, r'  $t=0$: Initial Tiling Tension', verticalalignment='center')
# B. The Present (You Are Here)
ax1.plot(T_NOW_TIS, RHO_OBSERVED, 'ro', markersize=8, label='Present Era')
# Add dotted lines to axis
ax1.axvline(x=T_NOW_TIS, color='r', linestyle='--', alpha=0.5); ax1.axhline(y=RHO_OBSERVED, color='r', linestyle='--', alpha=0.5)
label_text = (f"  PRESENT DAY\n"
              f"  t = {T_NOW_TIS:.1f} Gyr\n"
              f"  w $\\approx$ -0.8\n"
              f"  $\\rho$ = {RHO_OBSERVED:.2e}")
ax1.text(T_NOW_TIS + 30, RHO_OBSERVED*1.1, label_text, color='red', fontsize=10)
# C. The Nariai Limit (The End)
rho_final = rho_lambda[-1]; ax1.plot(T_FINAL, rho_final, 'ko'); ax1.axvline(x=T_FINAL, color='k', linestyle=':', alpha=0.8); 
ax1.text(T_FINAL - 50, rho_final*2, r'Nariai Limit (Big Freeze)', horizontalalignment='right')
# Title and Info
plt.title(f'The TIS Timeline: Geometric Decay of Dark Energy\nDecay Constant $C = 13/3 \\cdot \\ln(\\phi) \\approx {C_DECAY:.2f}$', fontsize=14)
# Show scale context # Add a secondary x-axis for "Lifespan Progress %"
ax2 = ax1.twiny(); ax2.set_xlim(ax1.get_xlim()); ax2.set_xticks([0, T_NOW_TIS, T_FINAL]); 
ax2.set_xticklabels(['0%', f'{tau_now*100:.1f}%', '100%']); ax2.set_xlabel('Percentage of Total Informational Lifespan')
plt.tight_layout(); plt.show()