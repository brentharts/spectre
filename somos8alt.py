#!/usr/bin/env python3
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# --- Configuration ---
SOMOS_PHASE2 = 779731
PHASE_TRANSITION_N = 200000 # The point of interest for histogram analysis
FORCE_INT = '--int' in sys.argv
PLOT = '--plot' in sys.argv
NE = 50
NS = 0

# --- Test Case Setup ---
if '--test1' in sys.argv:
    NS = 0
    NE = 30
    FORCE_INT = True
elif '--test2' in sys.argv:
    NS = 0
    NE = 100
    FORCE_INT = True
elif '--test3' in sys.argv:
    NS = 0
    NE = 2500
    PLOT = True
    FORCE_INT = True
elif '--test4' in sys.argv:
    NS = 0
    NE = 500000
    PLOT = True
    FORCE_INT = True
elif '--test8' in sys.argv:
    NS = 0
    NE = 2000000
    PLOT = True
    FORCE_INT = True
elif '--test9' in sys.argv:
    NS = 0
    NE = 2000000
    PLOT = True
    FORCE_INT = False
elif '--start-end' in sys.argv:
    NS = int(sys.argv[-2])
    NE = int(sys.argv[-1])

# --- Somos-8 Sequence Function ---
VALUES = []
def somos_8_sequence(init=1, num_terms=256):
    """
    Computes the Somos-8 sequence up to a specified number of terms.
    The sequence breaks when the result is not an integer.
    """
    if num_terms < 8:
        return [init] * num_terms
    s = [init] * 8
    terms = [init] * 8

    for i in range(8, num_terms):
        # s_n = (s_{n-1}s_{n-7} + s_{n-2}s_{n-6} + s_{n-3}s_{n-5} + s_{n-4}^2) / s_{n-8}
        a = (s[i-1] * s[i-7] + s[i-2] * s[i-6] + s[i-3] * s[i-5] + s[i-4]**2)
        b = s[i-8]

        next_term = a / b
        terms.append((int(a), int(b)))

        # The core check for the non-integer property
        if next_term != int(next_term):
            s.append(next_term)
            # Log magnitude in log10 scale
            if PLOT and len(s) > 8: VALUES.append(math.log10(abs(next_term) + 1))
            break
        else:
            if FORCE_INT:
                s.append(int(next_term))
            else:
                s.append(next_term)

    return s, terms

# --- Plotting Function for Large Data ---
def plot(x, y, title, xlabel, ylabel, plot_type='scatter', **kwargs):
    mode_tag = ' (mode=int)' if FORCE_INT else ' (mode=float)'
    full_title = title + mode_tag
    N = len(x)
    print(f'plotting data size: {N:,} - {full_title}')

    if N <= 1:
        print('Data size too small, skipping plot.')
        return

    plt.figure(figsize=(12, 8))

    if plot_type == 'density' and N > 5000:
        # Use 2D Histogram (Heatmap) for high-density breaking points
        print(f'Using 2D Density Plot (Heatmap) for {N:,} points.')

        # Ensure y is integer for discrete bins
        y_int = [int(v) for v in y]
        x_bins = min(200, N // 1000) if N > 100000 else min(100, N // 100)
        y_unique = sorted(list(set(y_int)))

        counts, xedges, yedges = np.histogram2d(x, y_int, bins=(x_bins, y_unique + [max(y_unique) + 1]))

        plt.imshow(counts.T, interpolation='nearest', origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect='auto', cmap='viridis')
        cbar = plt.colorbar(label='Count of Initial Values N')
        cbar.ax.tick_params(labelsize=10)
        plt.yticks([v + 0.5 for v in y_unique], [f'{v}' for v in y_unique])

    elif plot_type == 'histogram':
        # Use simple histogram for frequency analysis
        # Max bins set to 50 for clarity unless fewer unique values exist
        bins_count = min(50, len(set(x)))
        plt.hist(x, bins=bins_count, edgecolor='black', **kwargs)

    elif plot_type == 'scatter' or N <= 5000:
        # Use scatter/line for smaller data
        if kwargs.get('bars', False):
             plt.bar(x, y, width=1.0)
        else:
            # Use alpha blending for large scatter plots
            plt.scatter(x, y, s=1, alpha=0.5, marker='.')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(full_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- Main Execution Loop (Data Collection) ---
x = [] # Initializer N
y = [] # Breaking point index
strange = [] # List of N values where len(sequence) != 18
phase_trans = Counter()

print(f"Starting Somos-8 search from N={NS+1} to N={NE}...")
for i in range(NS, NE):
    initializer = i + 1
    sequence, terms = somos_8_sequence(initializer)
    lenseq = len(sequence) - 1 # Breaking point index

    x.append(initializer)
    y.append(lenseq)
    phase_trans[lenseq] += 1

    if lenseq != 17: # Standard break is at index 17 (18th term)
        strange.append((initializer, lenseq, sequence[-1], terms[-1]))

    if NE - NS <= 100:
        print(f'Initializer: {initializer}, Breaks at: {lenseq}')

print('total-strange:', len(strange))

# --- Delta Calculation for Strange Points ---
strange_N = [s[0] for s in strange]
deltas = [strange_N[i] - strange_N[i-1] for i in range(1, len(strange_N))]

if not strange_N:
    print('No strange points found in the range.')
    sys.exit()

# 1. Separate Deltas into two phases
delta_N_values = strange_N[1:] # The N value corresponding to the current delta

deltas_phase1 = [] # N <= 200k
deltas_phase2 = [] # N > 200k

for N_val, delta in zip(delta_N_values, deltas):
    if N_val <= PHASE_TRANSITION_N:
        deltas_phase1.append(delta)
    else:
        deltas_phase2.append(delta)

print(f'Phase 1 Deltas Count (N <= {PHASE_TRANSITION_N}): {len(deltas_phase1)}')
print(f'Phase 2 Deltas Count (N > {PHASE_TRANSITION_N}): {len(deltas_phase2)}')

# --- PLOT 1: Phase Transition Histograms ---
if PLOT and len(deltas_phase1) > 1 and len(deltas_phase2) > 1:
    plt.figure(figsize=(14, 6))

    # Histogram for Phase 1 (Regular)
    plt.subplot(1, 2, 1)
    # Use bins that align with common multiples (10, 20, etc.)
    bins_p1 = np.arange(min(deltas_phase1) // 10 * 10, max(deltas_phase1) + 10, 10)
    plt.hist(deltas_phase1, bins=bins_p1, color='lightblue', edgecolor='blue', align='left')
    plt.title(f'Phase 1: Delta Spacing Frequency (N <= {PHASE_TRANSITION_N})')
    plt.xlabel('Delta N (Spacing between Strange Points)')
    plt.ylabel('Frequency (Count)')
    plt.xlim(0, max(600, max(deltas_phase1)))
    plt.grid(axis='y', alpha=0.7)

    # Histogram for Phase 2 (Chaotic)
    plt.subplot(1, 2, 2)
    bins_p2 = np.arange(min(deltas_phase2) // 10 * 10, max(deltas_phase2) + 10, 10)
    plt.hist(deltas_phase2, bins=bins_p2, color='lightcoral', edgecolor='red', align='left')
    plt.title(f'Phase 2: Delta Spacing Frequency (N > {PHASE_TRANSITION_N})')
    plt.xlabel('Delta N (Spacing between Strange Points)')
    plt.ylabel('Frequency (Count)')
    plt.xlim(0, max(800, max(deltas_phase2)))
    plt.grid(axis='y', alpha=0.7)

    plt.suptitle(f'Somos8 Delta Distribution Across Phase Transition Point N={PHASE_TRANSITION_N}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- PLOT 2: Overall Breaking Points Density ---
if PLOT:
    plot(x, y,
         title=f'Somos8 Breaking Points ({NS+1}-{NE})',
         xlabel='Somos-8 Initializer (N)',
         ylabel='Breaking Point Index (k)',
         plot_type='density')

# --- PLOT 3: Delta N Time Series (Original Plot) ---
if PLOT:
    plot(delta_N_values, deltas,
         title=f'Somos8 Strange Group Deltas ({NS+1}-{NE})',
         xlabel='Somos(N)',
         ylabel='delta spacing of N',
         plot_type='scatter',
         bars=False)

# --- Final Output ---
print('\n--- Summary ---')
print(f'Total strange initial values found: {len(strange):,}')
print(f'Strange initial values ratio: {len(strange) / (NE - NS + 1) * 100:.4f}%')
print(f'Mean spacing (Delta N) in Phase 1: {np.mean(deltas_phase1):.2f}')
print(f'Mean spacing (Delta N) in Phase 2: {np.mean(deltas_phase2):.2f}')
print('-----------------')