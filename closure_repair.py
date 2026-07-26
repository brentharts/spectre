#!/usr/bin/env python3
"""
closure_repair.py -- CLOSED chimeric Spectre/Hat tiles, and how much hat the
spectre lattice can absorb.

Background
----------
`tile_family.py` decomposes Tile(a,b) into 14 fixed unit edge directions
u_i plus a per-edge length vector L, and `mixed_tiling.py` showed that
per-edge mixtures of spectre (L_b=1) and hat (L_b=sqrt(3)) lengths generally
FAIL to close: the walk sum_i L_i u_i is nonzero (closure defect 0.77 mean
for random per-edge mixing).  Everything downstream -- gaps, overlaps,
foldability -- is then contaminated by tiles that are not even polygons.

This module removes that contamination in two ways.

1. REPAIR.  Closure is exactly two linear constraints,

       A L = 0,        A = U^T  (2 x 14),   U = UNIT_DIRS,

   so the nearest closed length vector to a desired L_target is an
   orthogonal projection,

       L* = L_t - W^-1 A^T (A W^-1 A^T)^-1 A L_t

   with W a diagonal weight matrix (W = I: spread the correction over all
   14 edges; W huge on a-edges: pin a=1 and pay for closure only with the
   b-edges).  Both are implemented; the a-pinned variant is the meaningful
   one, because the spectre<->hat axis only moves b-edges.

2. EXACT PARAMETERISATION.  With a-edges pinned at 1, closure reads
   A_b L_b = -A_a 1, a 2x6 system.  Its solution set is a 4-dimensional
   affine family of *exactly closed* chimeric tiles,

       L_b(c) = L_b0 + N c,   N = null(A_b) (6 x 4),   c in R^4.

   No repair needed: every tile in this family closes to machine precision.
   Annealing c against (overlap + gap) on the spectre substitution lattice
   is then a clean answer to "how much hat can the spectre lattice absorb?"

Outputs (into $EINSTEIN3D_OUT, default /tmp):
    closure_repair_tilings.png    before/after repair, per mixing mode
    closure_absorption.png        damage vs mean b-length + anneal trace
    closure_best_tile.png         the annealed chimeric tile vs spectre/hat
    closure_repair_metrics.txt    the tables printed below
"""
import os, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from shapely.ops import unary_union

import tile_family as TF

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
SQRT3 = np.sqrt(3.0)
RNG = np.random.default_rng(11)

A_MASK = (TF.EDGE_TYPES == 'a')
B_MASK = (TF.EDGE_TYPES == 'b')


def exact_unit_dirs():
    """tile_family derives UNIT_DIRS from spectre.py's float32 vertex table,
    which limits closure defects to ~1e-8.  Every edge direction of the
    14-gon is an exact multiple of 30 degrees (verified below), so snapping
    to the nearest 30 deg recovers full float64 precision -- needed if
    'exactly closed' is to mean 1e-16 rather than 1e-8.
    """
    ang = np.degrees(np.arctan2(TF.UNIT_DIRS[:, 1], TF.UNIT_DIRS[:, 0]))
    k = np.round(ang / 30.0)
    assert np.max(np.abs(ang / 30.0 - k)) < 1e-4, 'edge angles are not 30-fold'
    th = np.deg2rad(30.0 * k)
    return np.column_stack([np.cos(th), np.sin(th)])


UNIT_DIRS = exact_unit_dirs()
A_MAT = UNIT_DIRS.T                         # 2 x 14 closure operator


# ---------------------------------------------------------------------------
# 1. least-squares closure repair
# ---------------------------------------------------------------------------
def closure_defect(L):
    """The 2-vector sum_i L_i u_i.  Zero iff the 14-gon closes."""
    return A_MAT @ np.asarray(L, dtype=float)


def repair_closure(L_target, weights=None):
    """Nearest closed length vector to L_target in the W-weighted norm.

    weights: per-edge cost of being moved.  Large weight = reluctant to
    move.  None -> uniform.  Returns (L_repaired, correction).
    """
    L_t = np.asarray(L_target, dtype=float)
    w = np.ones(TF.N_EDGES) if weights is None else np.asarray(weights, float)
    Winv = np.diag(1.0 / w)
    M = A_MAT @ Winv @ A_MAT.T                       # 2 x 2, invertible
    lam = np.linalg.solve(M, A_MAT @ L_t)
    corr = -Winv @ A_MAT.T @ lam
    return L_t + corr, corr


def repair_pinned_a(L_target, a_weight=1e6):
    """Repair paying only with b-edges (a-edges held at their target)."""
    w = np.where(A_MASK, a_weight, 1.0)
    return repair_closure(L_target, weights=w)


# ---------------------------------------------------------------------------
# 2. the exactly-closed 4-parameter chimeric family
# ---------------------------------------------------------------------------
def _null_space(M, tol=1e-10):
    """Orthonormal basis of ker(M), as columns."""
    _, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > tol))
    return vt[rank:].T


A_B = A_MAT[:, B_MASK]                    # 2 x 6
A_A = A_MAT[:, A_MASK]                    # 2 x 8
NULL_B = _null_space(A_B)                 # 6 x 4
# particular solution: the spectre itself (all b = 1) already closes
L_B0 = np.ones(B_MASK.sum())


def lengths_from_c(c):
    """Exactly-closed length vector from the 4 free parameters c."""
    L = np.ones(TF.N_EDGES)
    L[B_MASK] = L_B0 + NULL_B @ np.asarray(c, dtype=float)
    return L


def c_from_lengths(L):
    """Least-squares c of a length vector (a-edges ignored)."""
    return np.linalg.lstsq(NULL_B, np.asarray(L)[B_MASK] - L_B0, rcond=None)[0]


def mean_spectreness(L):
    """Mean s in [0,1] of the b-edges: 0 = spectre, 1 = hat."""
    return float(np.mean((L[B_MASK] - 1.0) / (SQRT3 - 1.0)))


# ---------------------------------------------------------------------------
# 3. build a tiling in which every tile carries an explicit length vector
# ---------------------------------------------------------------------------
_PLACEMENT_CACHE = {}


def placements(n_iterations):
    if n_iterations not in _PLACEMENT_CACHE:
        _PLACEMENT_CACHE[n_iterations] = TF.placed_tiles(n_iterations)
    return _PLACEMENT_CACHE[n_iterations]


CANON = TF.canonical_tile_verts('any')
CANON_CENTROID = CANON[:14].mean(axis=0)


def build_tiling(length_fn, n_iterations=2):
    """length_fn(i, T, label) -> per-edge length vector for placed tile i."""
    recs = []
    for i, (T, label) in enumerate(placements(n_iterations)):
        L = np.asarray(length_fn(i, T, label), dtype=float)
        local, defect = TF.build_polygon(L, start=CANON[0],
                                         mirror_dirs=UNIT_DIRS)
        local = local - local[:14].mean(axis=0) + CANON_CENTROID
        world = TF.transform_polygon(T, local)
        poly = Polygon(world[:14])
        if not poly.is_valid:
            poly = poly.buffer(0)
        recs.append(dict(label=label, T=T, verts=world, poly=poly,
                         closure_defect=float(np.linalg.norm(defect)),
                         area=float(abs(TF.polygon_area(
                             np.vstack([world[:14], world[:1]])))),
                         edge_lengths=L.copy(),
                         s=np.full(14, mean_spectreness(L))))
    return recs


_REF_CACHE = {}


def reference_region(n_iterations=2, erode=1.2):
    """The region the lattice is supposed to cover: the union of the pure
    spectre tiling, eroded a little so that boundary effects don't dominate.

    Why this is needed: `mixed_tiling.measure` reports gaps as the interior
    HOLES of the union.  That is fine for tiles that are too big, but it
    scores a tiling of shrunken, mutually disjoint tiles as gap-free -- the
    union is then a MultiPolygon with no interior rings at all.  Annealing
    exploits exactly that loophole and drives every b-edge below 1.
    Measuring against a fixed domain closes it.
    """
    if n_iterations not in _REF_CACHE:
        recs = build_tiling(lambda i, T, lab: TF.LEN_SPECTRE, n_iterations)
        ref = unary_union([r['poly'] for r in recs]).buffer(-erode)
        if ref.is_empty:
            ref = unary_union([r['poly'] for r in recs])
        _REF_CACHE[n_iterations] = ref
    return _REF_CACHE[n_iterations]


def measure_in_domain(recs, n_iterations=2):
    """Overlap and uncovered area as fractions of the reference domain."""
    ref = reference_region(n_iterations)
    ra = ref.area
    clipped = [r['poly'].intersection(ref) for r in recs]
    clipped = [p for p in clipped if not p.is_empty]
    sum_area = float(sum(p.area for p in clipped))
    union = unary_union(clipped)
    union_area = float(union.area)
    return dict(ref_area=ra,
                overlap_frac=(sum_area - union_area) / ra,
                gap_frac=(ra - union_area) / ra,
                sum_tile_area=float(sum(r['poly'].area for r in recs)),
                mean_tile_area=float(np.mean([r['area'] for r in recs])),
                mean_edge_len=float(np.mean([r['edge_lengths'] for r in recs])),
                mean_closure_defect=float(np.mean(
                    [r['closure_defect'] for r in recs])))


def damage(L, n_iterations=2):
    """Scalar objective: (overlap + uncovered) as % of the reference domain."""
    recs = build_tiling(lambda i, T, lab: L, n_iterations)
    m = measure_in_domain(recs, n_iterations)
    return 100.0 * (m['overlap_frac'] + m['gap_frac']), m


# ---------------------------------------------------------------------------
# 4. random per-edge mixtures: before vs after repair
# ---------------------------------------------------------------------------
def mixture_lengths(mode, rng):
    if mode == 'per_edge':
        s = rng.integers(0, 2, size=14).astype(float)
        return TF.lengths_per_edge(s)
    if mode == 'per_vertex':
        sv = rng.integers(0, 2, size=14).astype(float)
        return TF.lengths_per_vertex(sv)
    if mode == 'per_edge_cont':
        return TF.lengths_per_edge(rng.random(14))
    raise ValueError(mode)


def repair_study(n_iterations=2, modes=('per_edge', 'per_vertex',
                                        'per_edge_cont')):
    rows = []
    tilings = {}
    for mode in modes:
        for repair in ('raw', 'free', 'pinned_a'):
            rng = np.random.default_rng(7)      # same mixtures every time
            store = {}

            def length_fn(i, T, label, _mode=mode, _rep=repair, _rng=rng,
                          _store=store):
                L = mixture_lengths(_mode, _rng)
                if _rep == 'free':
                    L, corr = repair_closure(L)
                elif _rep == 'pinned_a':
                    L, corr = repair_pinned_a(L)
                else:
                    corr = np.zeros(14)
                _store.setdefault('corr', []).append(corr)
                return L

            recs = build_tiling(length_fn, n_iterations)
            m = measure_in_domain(recs, n_iterations)
            corr = np.array(store['corr'])
            rows.append(dict(mode=mode, repair=repair, metrics=m,
                             mean_abs_corr=float(np.abs(corr).mean()),
                             max_abs_corr=float(np.abs(corr).max()),
                             a_corr=float(np.abs(corr[:, A_MASK]).mean()),
                             b_corr=float(np.abs(corr[:, B_MASK]).mean())))
            tilings[(mode, repair)] = recs
    return rows, tilings


# ---------------------------------------------------------------------------
# 5. annealing over the closed family
# ---------------------------------------------------------------------------
def anneal(objective, x0, bounds_fn=None, n_steps=600, T0=0.6, T1=0.005,
           step0=0.35, rng=None, verbose=True):
    """Plain simulated annealing with geometric cooling and shrinking steps."""
    rng = rng or np.random.default_rng(0)
    x = np.array(x0, dtype=float)
    f = objective(x)
    best_x, best_f = x.copy(), f
    trace = [(0, f, best_f)]
    for k in range(1, n_steps + 1):
        frac = k / n_steps
        T = T0 * (T1 / T0) ** frac
        step = step0 * (0.05 / step0) ** frac
        y = x + rng.normal(scale=step, size=x.shape)
        if bounds_fn is not None:
            y = bounds_fn(y)
        fy = objective(y)
        if fy < f or rng.random() < np.exp(-(fy - f) / max(T, 1e-9)):
            x, f = y, fy
            if f < best_f:
                best_x, best_f = x.copy(), f
        trace.append((k, f, best_f))
        if verbose and k % max(1, n_steps // 10) == 0:
            print(f'    step {k:5d}  T={T:7.4f}  f={f:8.4f}  best={best_f:8.4f}')
    return best_x, best_f, np.array(trace)


def clamp_c(c, lo=0.6, hi=SQRT3 + 0.15):
    """Keep b-edge lengths in a sane positive range by clipping in L space."""
    Lb = L_B0 + NULL_B @ c
    Lb = np.clip(Lb, lo, hi)
    return np.linalg.lstsq(NULL_B, Lb - L_B0, rcond=None)[0]


def _crossing(x, y, thr):
    """Linear interpolation of the first x where y rises through thr."""
    for i in range(1, len(y)):
        if y[i - 1] <= thr <= y[i]:
            t = (thr - y[i - 1]) / (y[i] - y[i - 1] + 1e-12)
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return float(x[-1]) if y[-1] <= thr else 0.0


def absorption_curve(n_iterations=2, n_targets=11, n_steps=250, seed=3):
    """For each target mean spectre-ness s_bar, anneal the closed family to
    minimise damage while holding <s> = s_bar.  Answers 'how much hat can
    the spectre lattice absorb?'."""
    rng = np.random.default_rng(seed)
    targets = np.linspace(0.0, 1.0, n_targets)
    out = []
    # direction in c-space that changes the mean b-length
    g = NULL_B.T @ np.ones(B_MASK.sum()) / B_MASK.sum()   # d<Lb>/dc
    for sb in targets:
        Lb_mean_target = 1.0 + (SQRT3 - 1.0) * sb

        def project(c):
            c = clamp_c(c)
            cur = np.mean(L_B0 + NULL_B @ c)
            if np.dot(g, g) > 1e-12:
                c = c + g * (Lb_mean_target - cur) / np.dot(g, g)
            return c

        def obj(c):
            return damage(lengths_from_c(project(c)), n_iterations)[0]

        c0 = project(np.zeros(NULL_B.shape[1]))
        best_c, best_f, _ = anneal(obj, c0, bounds_fn=project,
                                   n_steps=n_steps, rng=rng, verbose=False)
        best_c = project(best_c)
        L = lengths_from_c(best_c)
        d_uni, m_uni = damage(TF.lengths_per_edge(np.full(14, sb)),
                              n_iterations)
        d_best, m_best = damage(L, n_iterations)
        out.append(dict(s_bar=sb, L=L, c=best_c, damage=d_best,
                        uniform_damage=d_uni,
                        overlap=100 * m_best['overlap_frac'],
                        gap=100 * m_best['gap_frac'],
                        closure=m_best['mean_closure_defect']))
        print(f'  s_bar={sb:4.2f}  uniform Tile(1,b) damage={d_uni:8.4f}%   '
              f'best closed chimera={d_best:8.4f}%   '
              f'(overlap {100*m_best["overlap_frac"]:6.3f}, '
              f'gap {100*m_best["gap_frac"]:5.3f})')
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def _draw(ax, recs, title):
    verts = [r['verts'][:14] for r in recs]
    pc = PolyCollection(verts, facecolors='none', edgecolors='k',
                        linewidths=0.45)
    ax.add_collection(pc)
    pc2 = PolyCollection(verts, facecolors='steelblue', edgecolors='none',
                         alpha=0.30)
    ax.add_collection(pc2)
    ax.autoscale(); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=9)


def plot_repair(rows, tilings, fname):
    modes = sorted({r['mode'] for r in rows}, key=lambda m: (
        'per_edge', 'per_vertex', 'per_edge_cont').index(m))
    reps = ['raw', 'free', 'pinned_a']
    fig, axes = plt.subplots(len(modes), 3, figsize=(15, 5 * len(modes)))
    axes = np.atleast_2d(axes)
    lookup = {(r['mode'], r['repair']): r for r in rows}
    for i, mode in enumerate(modes):
        for j, rep in enumerate(reps):
            r = lookup[(mode, rep)]
            m = r['metrics']
            _draw(axes[i, j], tilings[(mode, rep)],
                  f"{mode} / {rep}\n"
                  f"closure defect {m['mean_closure_defect']:.3g}   "
                  f"overlap {100*m['overlap_frac']:.2f}%   "
                  f"gap {100*m['gap_frac']:.2f}%\n"
                  f"mean |dL| paid {r['mean_abs_corr']:.4f} "
                  f"(a {r['a_corr']:.4f} / b {r['b_corr']:.4f})")
    fig.suptitle('least-squares closure repair of chimeric Spectre/Hat tiles\n'
                 'left: unrepaired (open 14-gons)   middle: correction spread '
                 'over all edges   right: a-edges pinned, b-edges pay',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def plot_absorption(curve, trace, fname):
    sb = np.array([c['s_bar'] for c in curve])
    dmg = np.array([c['damage'] for c in curve])
    uni = np.array([c['uniform_damage'] for c in curve])
    ov = np.array([c['overlap'] for c in curve])
    gp = np.array([c['gap'] for c in curve])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.plot(sb, uni, 's--', color='0.5', label='uniform Tile(1, b)')
    ax.plot(sb, dmg, 'o-', color='crimson',
            label='best closed chimera (annealed)')
    ax.set_xlabel(r'mean spectre-ness $\bar s$  (0 = Spectre, 1 = Hat)')
    ax.set_ylabel('damage = (overlap + uncovered) % of reference domain')
    ax.set_title('how much hat can the spectre lattice absorb?')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(sb, ov, 'o-', label='overlap %')
    ax.plot(sb, gp, 's-', label='gap %')
    ax.set_xlabel(r'$\bar s$'); ax.set_ylabel('% ')
    ax.set_title('damage split: overlap vs interior gap')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    bidx = np.where(B_MASK)[0]
    Lb = np.array([c['L'][B_MASK] for c in curve])       # n_targets x 6
    for j, e in enumerate(bidx):
        ax.plot(sb, Lb[:, j], 'o-', ms=3, label=f'b-edge {e}')
    ax.plot(sb, 1.0 + (SQRT3 - 1.0) * sb, 'k--', lw=1.5,
            label='uniform Tile(1,b)')
    ax.axhline(SQRT3, color='0.6', lw=0.8)
    ax.set_xlabel(r'$\bar s$'); ax.set_ylabel('b-edge length')
    ax.set_title('which b-edges the optimum lengthens first')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def plot_best_tile(L_best, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    for L, lab, col in ((TF.LEN_SPECTRE, 'Spectre(1,1)', 'navy'),
                        (TF.LEN_HAT, r'Hat(1,$\sqrt{3}$)', 'darkorange'),
                        (L_best, 'annealed closed chimera', 'crimson')):
        v, d = TF.build_polygon(L, mirror_dirs=UNIT_DIRS)
        v = v - v[:14].mean(axis=0)
        ax.plot(v[:, 0], v[:, 1], '-', color=col, lw=2,
                label=f'{lab}  (defect {np.linalg.norm(d):.1e})')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title('closed chimeric tile found by annealing')

    ax = axes[1]
    x = np.arange(14)
    w = 0.27
    ax.bar(x - w, TF.LEN_SPECTRE, w, label='Spectre', color='navy')
    ax.bar(x, L_best, w, label='chimera', color='crimson')
    ax.bar(x + w, TF.LEN_HAT, w, label='Hat', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{t}' for i, t in enumerate(TF.EDGE_TYPES)],
                       fontsize=7)
    ax.set_xlabel('edge index / type'); ax.set_ylabel('length')
    ax.set_title('per-edge lengths')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=2)
    ap.add_argument('--steps', type=int, default=600,
                    help='anneal steps for the unconstrained search')
    ap.add_argument('--curve-steps', type=int, default=200,
                    help='anneal steps per point of the absorption curve')
    ap.add_argument('--targets', type=int, default=21)
    args = ap.parse_args()

    lines = []

    def say(s=''):
        print(s); lines.append(s)

    t0 = time.time()
    say('=== closure operator ===')
    say(f'edge types              : {"".join(TF.EDGE_TYPES)}')
    say(f'rank of A = U^T         : {np.linalg.matrix_rank(A_MAT)}  '
        f'(2 constraints on 14 lengths)')
    say(f'rank of A restricted to b-edges : {np.linalg.matrix_rank(A_B)}  '
        f'-> {NULL_B.shape[1]}-parameter family of closed tiles with a=1')
    say(f'check: spectre defect {np.linalg.norm(closure_defect(TF.LEN_SPECTRE)):.2e}'
        f'   hat defect {np.linalg.norm(closure_defect(TF.LEN_HAT)):.2e}')
    Lc = lengths_from_c(RNG.normal(scale=0.3, size=NULL_B.shape[1]))
    say(f'check: random member of the closed family, defect '
        f'{np.linalg.norm(closure_defect(Lc)):.2e}')
    say()

    say('=== 1. repair of random per-edge mixtures ===')
    rows, tilings = repair_study(args.iterations)
    hdr = (f"{'mode':15s} {'repair':9s} {'closure':>9s} {'overlap%':>9s} "
           f"{'gap%':>7s} {'mean|dL|':>9s} {'a-part':>8s} {'b-part':>8s}")
    say(hdr); say('-' * len(hdr))
    for r in rows:
        m = r['metrics']
        say(f"{r['mode']:15s} {r['repair']:9s} "
            f"{m['mean_closure_defect']:9.2e} "
            f"{100*m['overlap_frac']:9.3f} {100*m['gap_frac']:7.3f} "
            f"{r['mean_abs_corr']:9.4f} {r['a_corr']:8.4f} {r['b_corr']:8.4f}")
    png = os.path.join(OUT, 'closure_repair_tilings.png')
    plot_repair(rows, tilings, png)
    say(f'wrote {png}')
    say()

    say('=== 2. unconstrained anneal over the closed 4-parameter family ===')
    rng = np.random.default_rng(5)
    obj = lambda c: damage(lengths_from_c(clamp_c(c)), args.iterations)[0]
    best_c, best_f, trace = anneal(obj, np.zeros(NULL_B.shape[1]),
                                   bounds_fn=clamp_c, n_steps=args.steps,
                                   rng=rng)
    best_c = clamp_c(best_c)
    L_best = lengths_from_c(best_c)
    d_best, m_best = damage(L_best, args.iterations)
    say(f'best damage {d_best:.5f}%  (spectre baseline '
        f'{damage(TF.LEN_SPECTRE, args.iterations)[0]:.5f}%)')
    say(f'b-edge lengths : {np.array2string(L_best[B_MASK], precision=4)}')
    say(f'mean spectre-ness of the optimum : {mean_spectreness(L_best):.4f}')
    say(f'closure defect : {np.linalg.norm(closure_defect(L_best)):.2e}')
    png = os.path.join(OUT, 'closure_best_tile.png')
    plot_best_tile(L_best, png)
    say(f'wrote {png}')
    say()

    say('=== 3. absorption curve: damage vs enforced mean spectre-ness ===')
    curve = absorption_curve(args.iterations, n_targets=args.targets,
                             n_steps=args.curve_steps)
    hdr = (f"{'s_bar':>6s} {'uniform%':>10s} {'chimera%':>10s} "
           f"{'overlap%':>9s} {'gap%':>7s} {'gain':>8s}")
    say(hdr); say('-' * len(hdr))
    for c in curve:
        gain = c['uniform_damage'] - c['damage']
        say(f"{c['s_bar']:6.2f} {c['uniform_damage']:10.4f} "
            f"{c['damage']:10.4f} {c['overlap']:9.3f} {c['gap']:7.3f} "
            f"{gain:8.4f}")
    png = os.path.join(OUT, 'closure_absorption.png')
    plot_absorption(curve, trace, png)
    say(f'wrote {png}')

    # headline number: where each curve first crosses a damage threshold
    say()
    sb = np.array([c['s_bar'] for c in curve])
    for thr in (2.0, 5.0, 10.0):
        a = _crossing(sb, np.array([c['damage'] for c in curve]), thr)
        b = _crossing(sb, np.array([c['uniform_damage'] for c in curve]), thr)
        say(f'absorption capacity at {thr:4.1f}% damage: closed chimera '
            f's_bar = {a:.3f}   uniform Tile(1,b) s_bar = {b:.3f}   '
            f'(x{a/b:.2f})' if b > 0 else '')
    say(f'total time {time.time()-t0:.1f}s')

    with open(os.path.join(OUT, 'closure_repair_metrics.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
