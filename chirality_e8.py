#!/usr/bin/env python3
"""
chirality_e8.py -- putting numbers on the spectre <-> E8 (Lisi) analogy.

Context
-------
Lisi's E8 proposal (arXiv:0711.0770) founders on chirality: Distler &
Garibaldi proved E8 cannot host three chiral fermion generations without
mirror partners.  The spectre is the first STRICTLY CHIRAL aperiodic
monotile -- the closed-curve version tiles the plane in one handedness only,
with no reflected copies anywhere.  So the thematic question is whether the
spectre's built-in chirality can play the role E8's structure cannot.
This script computes the places where the two systems can actually be
compared, without pretending the analogy is a mechanism:

1. CHIRALITY CENSUS of the straight-edge spectre substitution used in this
   repo: handedness (det sign) of every placed tile per iteration, the
   chirality order parameter chi = (N_R - N_L)/N, and the Mystic fraction.
   (The straight-edge Tile(1,1) admits reflections; the census quantifies
   how the substitution actually uses them, and chi -> the asymptotic
   handedness imbalance.)

2. E8 SIDE: the 240 E8 roots projected to the Coxeter plane (the classic
   30-fold picture).  The projected roots fall on 8 circles whose radii
   pair up in the GOLDEN RATIO -- this is the honest quantitative bridge
   between E8 and quasicrystal geometry (Elser-Sloane).  We compute those
   radii and their ratios numerically.

3. ALGEBRAIC COMPARISON: E8/H4 quasicrystal scaling lives in Q(sqrt5)
   (golden); the spectre inflation is lambda = sqrt(4+sqrt15), and
   sqrt15 = sqrt3*sqrt5 -- the spectre eigenvalue mixes the hexagonal
   sqrt3 with the golden sqrt5.  We verify lambda^2 = 4+sqrt15 to machine
   precision and locate lambda relative to metallic means.

4. TILE-FREQUENCY SPECTRUM: the Perron right-eigenvector of the
   substitution matrix = asymptotic frequencies of the 9 tile types; this
   is the tiling's natural "spectrum of species", the analog slot for a
   particle spectrum in Lisi/Kletetschka-style numerology.

5. MASS-RATIO TEST (falsifiable, with trials accounting): scan a small
   disciplined family of spectre-derived constants n * lambda^k against
   measured lepton/quark mass ratios; report matches AND the expected
   number of false positives at the same tolerance.  This is the standard
   the Kletetschka three-time-dimension mass claims should also be held to.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spectre as S
import tile_family as TF

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
PHI = (1 + np.sqrt(5)) / 2
LAM = np.sqrt(4 + np.sqrt(15))
TILE_NAMES = S.TILE_NAMES


# ===========================================================================
# 1) chirality census
# ===========================================================================
def chirality_census(max_iter=4):
    rows = []
    for n in range(1, max_iter + 1):
        placed = TF.placed_tiles(n)
        dets = [np.linalg.det(T[:, :2]) for T, _ in placed]
        nR = sum(1 for d in dets if d > 0)
        nL = len(dets) - nR
        mystic = sum(1 for _, lab in placed if lab == 'Gamma2')
        rows.append(dict(iter=n, n=len(dets), R=nR, L=nL,
                         chi=(nR - nL) / len(dets),
                         mystic_frac=mystic / len(dets)))
    return rows


# ===========================================================================
# 2) E8 roots and Coxeter-plane projection
# ===========================================================================
def e8_roots():
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(8); v[i] = si; v[j] = sj
                    roots.append(v)
    from itertools import product
    for signs in product((0.5, -0.5), repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            roots.append(np.array(signs))
    R = np.array(roots)
    assert len(R) == 240
    return R


def e8_simple_roots():
    """Standard E8 simple roots (Bourbaki numbering)."""
    a = np.zeros((8, 8))
    a[0] = [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]
    a[1] = [1, 1, 0, 0, 0, 0, 0, 0]
    for i in range(2, 8):
        a[i][i - 2] = -1
        a[i][i - 1] = 1
    return a


def coxeter_plane_projection(roots):
    """Project onto the plane fixed by the rotation part of a Coxeter
    element (eigenvalue e^{2 pi i / 30})."""
    simples = e8_simple_roots()
    W = np.eye(8)
    for al in simples:
        refl = np.eye(8) - 2 * np.outer(al, al) / (al @ al)
        W = refl @ W
    evals, evecs = np.linalg.eig(W)
    h = 30
    target = np.exp(2j * np.pi / h)
    idx = int(np.argmin(np.abs(evals - target)))
    v = evecs[:, idx]
    u1, u2 = np.real(v), np.imag(v)
    # orthonormalise
    u1 /= np.linalg.norm(u1)
    u2 -= (u2 @ u1) * u1
    u2 /= np.linalg.norm(u2)
    xy = np.column_stack([roots @ u1, roots @ u2])
    return xy


# ===========================================================================
# 4) tile-frequency spectrum (Perron eigenvector)
# ===========================================================================
SUBS = {
    'Gamma': ('Pi', 'Delta', None, 'Theta', 'Sigma', 'Xi', 'Phi', 'Gamma'),
    'Delta': ('Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'),
    'Theta': ('Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'),
    'Lambda': ('Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'),
    'Xi': ('Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'),
    'Pi': ('Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'),
    'Sigma': ('Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Lambda', 'Gamma'),
    'Phi': ('Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'),
    'Psi': ('Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'),
}


def substitution_matrix():
    M = np.zeros((9, 9))
    for j, lab in enumerate(TILE_NAMES):
        for s in SUBS[lab]:
            if s:
                M[TILE_NAMES.index(s), j] += 1
    return M


def tile_frequencies():
    M = substitution_matrix()
    evals, evecs = np.linalg.eig(M)
    i = int(np.argmax(evals.real))
    v = np.abs(evecs[:, i].real)
    return evals[i].real, v / v.sum()


# ===========================================================================
# 5) disciplined mass-ratio scan
# ===========================================================================
MASS_RATIOS = {
    'mu/e': 206.7682830,
    'tau/mu': 16.8170,
    'tau/e': 3477.228,
    't/b': 172.57e3 / 4.183e3,      # top/bottom (MSbar-ish, indicative)
    'b/c': 4.183 / 1.273,
    'c/s': 1273.0 / 93.5,
}


def mass_ratio_scan(tol=0.005):
    cands = {}
    for k in range(-4, 9):
        for n in range(1, 31):
            cands[f'{n}*lam^{k}'] = n * LAM ** k
            cands[f'lam^{k}/{n}'] = LAM ** k / n
    hits = []
    for rname, r in MASS_RATIOS.items():
        for cname, c in cands.items():
            if abs(c / r - 1) < tol:
                hits.append((rname, r, cname, c, abs(c / r - 1)))
    # expected false positives: candidates roughly log-uniform; for each
    # ratio, count candidates within a factor e of it and multiply by 2*tol
    n_expected = 0.0
    vals = np.array(sorted(set(cands.values())))
    for r in MASS_RATIOS.values():
        near = ((vals > r / np.e) & (vals < r * np.e)).sum()
        n_expected += near * 2 * tol / 2.0    # density * window (approx)
    return hits, len(cands), n_expected


# ===========================================================================
def main():
    print('=' * 70)
    print('1) chirality census of the spectre substitution')
    rows = chirality_census(4)
    for r in rows:
        print(f"  iter {r['iter']}: n={r['n']:5d}  R={r['R']:5d}  "
              f"L={r['L']:5d}  chi={r['chi']:+.4f}  "
              f"mystic={100*r['mystic_frac']:.2f}%")

    print('\n2) E8 Coxeter-plane projection: circle radii and ratios')
    roots = e8_roots()
    xy = coxeter_plane_projection(roots)
    radii = np.linalg.norm(xy, axis=1)
    uniq = []
    for r in sorted(radii):
        if not uniq or abs(r - uniq[-1][0]) > 1e-6:
            uniq.append([r, 1])
        else:
            uniq[-1][1] += 1
    print('  circles (radius, multiplicity):')
    for r, m in uniq:
        print(f'    r={r:.6f}  x{m}')
    rs = [u[0] for u in uniq]
    print('  adjacent / paired ratios vs phi:')
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            q = rs[j] / rs[i]
            if abs(q - PHI) < 1e-6:
                print(f'    r{j}/r{i} = {q:.9f} = phi  '
                      f'(phi = {PHI:.9f})')

    print('\n3) algebraic content')
    print(f'  spectre lambda        = {LAM:.9f}')
    print(f'  lambda^2 - 4          = {LAM**2 - 4:.9f} = sqrt(15) '
          f'({np.sqrt(15):.9f})')
    print(f'  sqrt15 = sqrt3*sqrt5  -> mixes hexagonal sqrt3 with golden '
          f'sqrt5; lambda is NOT in Q(sqrt5) (phi^2={PHI**2:.6f})')
    print(f'  nearest metallic mean: silver 1+sqrt2={1+np.sqrt(2):.4f}, '
          f'bronze (3+sqrt13)/2={(3+np.sqrt(13))/2:.4f}, lambda={LAM:.4f}')

    print('\n4) tile-frequency spectrum (Perron eigenvector)')
    ev, freq = tile_frequencies()
    print(f'  Perron eigenvalue = {ev:.9f} = 4+sqrt(15) '
          f'({4+np.sqrt(15):.9f})')
    order = np.argsort(freq)[::-1]
    for i in order:
        print(f'    {TILE_NAMES[i]:7s} {freq[i]:.6f}')

    print('\n5) mass-ratio scan (tol 0.5%)')
    hits, ncand, nexp = mass_ratio_scan()
    print(f'  candidate constants: {ncand}   '
          f'expected chance matches at this tol: ~{nexp:.1f}')
    for rname, r, cname, c, err in sorted(hits, key=lambda h: h[4]):
        print(f'    {rname:7s} {r:12.4f}  ~  {cname:12s} = {c:12.4f}  '
              f'(err {100*err:.3f}%)')
    print('  verdict: matches at or below the expected-chance count are '
          'numerology, not signal.')

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    ax.scatter(xy[:, 0], xy[:, 1], s=8, c=radii, cmap='viridis')
    for r, m in uniq:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.4,
                                color='grey', alpha=0.6))
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('240 E8 roots on the Coxeter plane\n'
                 'circle radii pair in the golden ratio $\\varphi$')

    ax = axes[1]
    its = [r['iter'] for r in rows]
    ax.plot(its, [r['chi'] for r in rows], 'o-', label='$\\chi=(R-L)/N$')
    ax.plot(its, [r['mystic_frac'] for r in rows], 's-',
            label='Mystic fraction')
    ax.set_xlabel('substitution iteration'); ax.set_xticks(its)
    ax.set_title('spectre chirality census')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar([TILE_NAMES[i] for i in order], [freq[i] for i in order],
           color='slateblue')
    ax.set_ylabel('asymptotic frequency')
    ax.set_title('tile-species spectrum (Perron eigenvector)\n'
                 f'eigenvalue $4+\\sqrt{{15}}$, inflation '
                 f'$\\lambda=\\sqrt{{4+\\sqrt{{15}}}}$')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    png = os.path.join(OUT, 'chirality_e8.png')
    fig.savefig(png, dpi=140)
    print('\nwrote', png)


if __name__ == '__main__':
    main()
