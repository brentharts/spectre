#!/usr/bin/env python3
"""
multiplets.py -- the spectre's multiplet structure and which degeneracies survive deformation.

Established exactly (sympy, Q(sqrt15); g = 4 - sqrt15 = 1/lambda^2):

    level         species        exact value          provenance
    0.221767      {Phi}          -54+14*sqrt15 = 2g(1-g)
    0.175416      {Psi}           97-25*sqrt15
    0.127017      {Gamma,Delta,Sigma}   g          CONSERVATION LAW:
                                                    rows are all-ones --
                                                    every supertile holds
                                                    exactly one of each
    0.094750      {Pi,Xi}        -58+15*sqrt15    ACCIDENTAL: automorphism
                                                    group of M is trivial;
                                                    equality holds iff
                                                    v_Phi = 2(v_Gamma-v_Theta)
    0.016133      {Theta,Lambda}  g^2 = 1/lambda^4  COROLLARY of the triplet:
                                                    row Theta = e_Gamma,
                                                    row Lambda = e_Sigma

Three deformation programmes, in decreasing generality:

A. MATRIX PERTURBATIONS  M -> M + eps*B, ensembles of B:
     generic          : B random nonnegative
     conservation-safe: B has zero rows on Gamma/Delta/Sigma (supertile
                        content of the triplet species untouched)
     indicator-safe   : conservation-safe AND B zero on the Theta/Lambda rows
   For each of the degeneracy relations (Gamma-Delta, Delta-Sigma,
   Theta-Lambda, Pi-Xi) we report the first-order splitting susceptibility
   |d(v_i - v_j)/d eps| distribution over the ensemble.

B. GEOMETRIC (MYSTIC-AXIS) DEFORMATION  r = b/a in [1, sqrt3]
   (spectre -> hat).  All nine species share the same Tile(a,b) polygon,
   EXCEPT Gamma = Gamma1 + Gamma2 with the Mystic Gamma2 = Tile(b,a), and
   area(a,b) != area(b,a).  The physical "mass-like" observable is the
   area fraction f_i(r) = v_i * A_i(r) / sum.  We sweep r and draw the
   level diagram: which levels move together, which split.

C. EMPIRICAL PER-EDGE MIXING: per-species mean tile area and closure
   defect from the per_edge mixed tiling (mixed_tiling.py records) --
   the randomness is species-blind, so this tests that no HIDDEN species
   dependence sneaks in through the substitution geometry.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spectre as S
import importlib.util
import chirality_e8 as CE

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
RNG = np.random.default_rng(17)
NAMES = S.TILE_NAMES
I = {n: k for k, n in enumerate(NAMES)}
M0 = CE.substitution_matrix()
SQ15 = np.sqrt(15.0)
G = 4 - SQ15

PAIRS = [('Gamma', 'Delta'), ('Delta', 'Sigma'),
         ('Theta', 'Lambda'), ('Pi', 'Xi')]


def perron_vec(M):
    evals, evecs = np.linalg.eig(M)
    i = int(np.argmax(evals.real))
    v = np.abs(evecs[:, i].real)
    return v / v.sum()


# ===========================================================================
# A) perturbation susceptibilities
# ===========================================================================
def susceptibility(B, eps=1e-6):
    v1 = perron_vec(M0 + eps * B)
    return {f'{a}-{b}': (v1[I[a]] - v1[I[b]]) / eps for a, b in PAIRS}


def ensemble(kind, n=400):
    out = {f'{a}-{b}': [] for a, b in PAIRS}
    for _ in range(n):
        B = RNG.random((9, 9))
        if kind in ('conservation', 'indicator'):
            for lab in ('Gamma', 'Delta', 'Sigma'):
                B[I[lab], :] = 0.0
        if kind == 'indicator':
            for lab in ('Theta', 'Lambda'):
                B[I[lab], :] = 0.0
        s = susceptibility(B)
        for k in out:
            out[k].append(abs(s[k]))
    return {k: np.array(vs) for k, vs in out.items()}


# ===========================================================================
# B) mystic-axis (geometric) level diagram
# ===========================================================================
def tile_area(a, b):
    pts = S.get_spectre_points(a, b)
    x, y = pts[:, 0], pts[:, 1]
    return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def mystic_sweep(nr=60):
    v = perron_vec(M0)
    rs = np.linspace(1.0, np.sqrt(3.0), nr)
    fr = np.zeros((nr, 9))
    for k, r in enumerate(rs):
        A = np.full(9, tile_area(1.0, r))
        A[I['Gamma']] = tile_area(1.0, r) + tile_area(r, 1.0)   # + Mystic
        w = v * A
        fr[k] = w / w.sum()
    return rs, fr


# ===========================================================================
# C) empirical per-edge census
# ===========================================================================
def per_edge_census():
    import mixed_tiling as MT
    records = MT.build_mixed_tiling('per_edge', n_iterations=2)
    from collections import defaultdict
    areas = defaultdict(list); defects = defaultdict(list)
    for rec in records:
        lab = rec['label']
        if lab in ('Gamma1', 'Gamma2'):
            lab = 'Gamma(' + lab[-1] + ')'
        areas[lab].append(rec['area'])
        defects[lab].append(rec['closure_defect'])
    return areas, defects


# ===========================================================================
def main():
    v = perron_vec(M0)
    print('exact levels (numeric check):')
    for lev, members in [(2 * G * (1 - G), ['Phi']),
                         (97 - 25 * SQ15, ['Psi']),
                         (G, ['Gamma', 'Delta', 'Sigma']),
                         (-58 + 15 * SQ15, ['Pi', 'Xi']),
                         (G * G, ['Theta', 'Lambda'])]:
        got = [v[I[m]] for m in members]
        print(f'  {lev:.9f}  {members}  max dev {max(abs(x-lev) for x in got):.2e}')

    print('\nA) splitting susceptibilities |d(v_i - v_j)/d eps|  '
          '(median [90th pct] over 400 random B)')
    table = {}
    for kind in ('generic', 'conservation', 'indicator'):
        sus = ensemble(kind)
        table[kind] = sus
        row = '  '.join(f"{k}: {np.median(s):.2e} [{np.percentile(s,90):.2e}]"
                        for k, s in sus.items())
        print(f'  {kind:13s} {row}')

    print('\n  reading: triplet relations vanish once the conservation rows')
    print('  are protected; Theta-Lambda additionally needs its indicator')
    print('  rows intact (conservation alone is NOT enough); Pi-Xi keeps an')
    print('  O(1) susceptibility in EVERY ensemble -> the accidental doublet')
    print('  is the fragile one, first to split under any combinatorial')
    print('  deformation -- yet it survives the geometric mystic axis,')
    print('  where only Gamma moves.')

    print('\nB) mystic-axis sweep r = b/a in [1, sqrt3]')
    rs, fr = mystic_sweep()
    for r_idx, tag in [(0, 'r=1 (spectre)'), (-1, 'r=sqrt3 (hat)')]:
        print(f'  {tag}:')
        order = np.argsort(fr[r_idx])[::-1]
        for i in order:
            print(f'    {NAMES[i]:7s} {fr[r_idx, i]:.6f}')
    # which pairs split along the sweep?
    for a, b in PAIRS:
        d = np.abs(fr[:, I[a]] - fr[:, I[b]]).max()
        print(f'  max |f_{a} - f_{b}| over sweep: {d:.6f}')

    print('\nC) empirical per-edge census (iteration 2, random per-edge mix)')
    areas, defects = per_edge_census()
    for lab in sorted(areas):
        a = np.array(areas[lab]); d = np.array(defects[lab])
        print(f'  {lab:9s} n={len(a):3d}  area {a.mean():7.3f}+-{a.std():5.3f}'
              f'  defect {d.mean():.3f}+-{d.std():.3f}')

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    kinds = ('generic', 'conservation', 'indicator')
    x = np.arange(len(PAIRS))
    width = 0.25
    for kidx, kind in enumerate(kinds):
        med = [np.median(table[kind][f'{a}-{b}']) for a, b in PAIRS]
        ax.bar(x + (kidx - 1) * width, med, width, label=kind)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}\u2212{b}' for a, b in PAIRS])
    ax.set_ylabel('median |splitting| / eps')
    ax.set_title('degeneracy fragility under matrix perturbations\n'
                 '(low bar = protected)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    for i in range(9):
        ax.plot(rs, fr[:, i], color=colors[i], lw=1.6, label=NAMES[i])
    ax.set_xlabel('r = b/a   (1 = spectre, $\\sqrt{3}$ = hat)')
    ax.set_ylabel('area fraction $f_i(r)$')
    ax.set_title('mystic-axis level diagram\n'
                 '$\\Gamma$ splits off the triplet; $\\{\\Delta,\\Sigma\\}$, '
                 '$\\{\\Pi,\\Xi\\}$, $\\{\\Theta,\\Lambda\\}$ ride together')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[2]
    labs = sorted(areas)
    means = [np.mean(areas[l]) for l in labs]
    errs = [np.std(areas[l]) / np.sqrt(len(areas[l])) for l in labs]
    ax.errorbar(range(len(labs)), means, yerr=errs, fmt='o', capsize=3)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=45, ha='right')
    ax.set_ylabel('mean per-tile area (per-edge mixed)')
    ax.set_title('species-blindness of random per-edge mixing\n'
                 '(flat = no hidden species dependence)')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT, 'multiplets.png')
    fig.savefig(png, dpi=140)
    print('\nwrote', png)


if __name__ == '__main__':
    main()
