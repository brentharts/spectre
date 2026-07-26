#!/usr/bin/env python3
"""
bezier_tiling.py -- hook `bezier_spectre.py` into the substitution tiler.

`bezier_spectre.py` (forked from Jan-Piotraschke/spectre-monotile-py) draws
ONE curved monotile: each straight edge is replaced by a cubic Bezier whose
two control points sit at the edge midpoint, pushed +-curve_strength along
the edge normal.  It was never wired into the substitution system, so there
was no way to ask the only interesting question about it: does the curved
tile still TILE?

It does, and the reason is worth stating precisely.

    Edge p->q, d = q-p, m = (p+q)/2, n = J d with J = rotation by -90 deg.
    Controls are m - c*n and m + c*n.  P0,P3 and P1,P2 are each swapped by
    the point reflection about m, so the whole cubic is invariant under it.

The neighbouring tile walks the same geometric edge backwards, and its own
copy of the construction lands on the same point set -- PROVIDED the two
tiles have the same handedness.  J anticommutes with reflections
(M J = -J M), so a mirrored tile would put its bulge on the wrong side and
the curved edges would cross instead of mating.  The spectre substitution
places every tile with det(T) = +1 (checked below), i.e. the tiling is
strictly chiral -- the same fact `chirality_e8.py` measures as chi = +-1 at
every iteration.  So the curved tiling inherits gap-freeness from chirality.
This script measures all of that, and the counterexample: insert one mirror
tile and the mating error jumps from 1e-16 to O(c).

Measured here
-------------
  * mating error vs curve strength (should stay at machine precision)
  * area is EXACTLY invariant under c (the two lobes of each S-curve cancel);
    perimeter grows -- so the curved family is an isoperimetric-ratio knob
  * critical curve strength c* at which the tile boundary first
    self-intersects, for Spectre, Hat, Turtle, and annealed chimeras
  * curved tilings rendered across c, and across the Tile(a,b) family

Outputs (into $EINSTEIN3D_OUT, default /tmp):
    bezier_tiling_sweep.png      tilings at several curve strengths
    bezier_metrics.png           area/perimeter/mating/self-intersection
    bezier_family.png            curved Spectre / Hat / Turtle / chimera
    bezier_tiling_metrics.txt
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

import tile_family as TF
import closure_repair as CR
from bezier_spectre import calculate_control_points

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
SQRT3 = np.sqrt(3.0)


# ---------------------------------------------------------------------------
# curved boundary construction
# ---------------------------------------------------------------------------
def edge_controls(p, q, c):
    """The two cubic control points of the bezier_spectre edge p->q.

    Identical to bezier_spectre.calculate_control_points, but expressed in
    absolute coordinates and vectorised.  Kept in sync by the unit test in
    check_against_upstream().
    """
    d = np.asarray(q, float) - np.asarray(p, float)
    n = np.array([d[1], -d[0]])              # J d
    m = np.asarray(p, float) + 0.5 * d
    return m - c * n, m + c * n


def cubic(p0, p1, p2, p3, t):
    t = np.asarray(t, float)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def curved_boundary(verts, c, samples=40):
    """Sampled curved boundary of a closed vertex loop (N,2)."""
    v = np.asarray(verts, float)
    n = len(v)
    t = np.linspace(0, 1, samples, endpoint=False)
    out = []
    for i in range(n):
        p, q = v[i], v[(i + 1) % n]
        c1, c2 = edge_controls(p, q, c)
        out.append(cubic(p, c1, c2, q, t))
    return np.vstack(out)


def curved_path(verts, c):
    """matplotlib Path with real cubic segments (no sampling)."""
    v = np.asarray(verts, float)
    pts, codes = [v[0]], [Path.MOVETO]
    for i in range(len(v)):
        p, q = v[i], v[(i + 1) % len(v)]
        c1, c2 = edge_controls(p, q, c)
        pts += [c1, c2, q]
        codes += [Path.CURVE4] * 3
    return Path(np.array(pts), codes)


def check_against_upstream(a=1.0, b=1.0, c=0.12):
    """Verify edge_controls reproduces bezier_spectre.calculate_control_points."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        p = rng.normal(size=2)
        d = rng.normal(size=2)
        q = p + d
        up1, up2 = calculate_control_points(d, c)
        up1 = np.asarray(up1) + p
        up2 = np.asarray(up2) + p
        mine1, mine2 = edge_controls(p, q, c)
        worst = max(worst, np.abs(up1 - mine1).max(), np.abs(up2 - mine2).max())
    return worst


# ---------------------------------------------------------------------------
# tiling with curved edges
# ---------------------------------------------------------------------------
def placed_curved_tiles(n_iterations=2, lengths=None, c=0.12, samples=40,
                        mirror_one=None):
    """Return per-tile dicts with straight verts, curved boundary, handedness.

    mirror_one: index of a tile to deliberately mirror, to demonstrate that
    curved mating depends on the tiling being strictly chiral.
    """
    if lengths is None:
        lengths = TF.LEN_SPECTRE
    local, defect = TF.build_polygon(lengths, mirror_dirs=CR.UNIT_DIRS)
    local = local[:14] - local[:14].mean(axis=0) + CR.CANON_CENTROID
    recs = []
    for i, (T, label) in enumerate(TF.placed_tiles(n_iterations)):
        T = T.copy()
        if mirror_one is not None and i == mirror_one:
            M = np.array([[-1.0, 0.0], [0.0, 1.0]])
            T[:, :2] = T[:, :2] @ M
        world = TF.transform_polygon(T, local)
        recs.append(dict(idx=i, label=label, verts=world,
                         det=float(np.linalg.det(T[:, :2])),
                         curve=curved_boundary(world, c, samples)))
    return recs, float(np.linalg.norm(defect))


def edge_table(recs):
    """key -> list of (tile, edge) for the straight skeleton."""
    em = {}
    for ti, r in enumerate(recs):
        v = r['verts']
        for ei in range(14):
            p, q = v[ei], v[(ei + 1) % 14]
            key = tuple(sorted([tuple(np.round(p, 4)), tuple(np.round(q, 4))]))
            em.setdefault(key, []).append((ti, ei))
    return em


def mating_error(recs, em, c, samples=64):
    """Max Hausdorff-ish distance between the two curved copies of every
    shared edge.  Zero => the curved tiling has no gaps or overlaps."""
    worst = 0.0
    n_shared = 0
    for key, users in em.items():
        if len(users) != 2:
            continue
        n_shared += 1
        curves = []
        for (ti, ei) in users:
            v = recs[ti]['verts']
            p, q = v[ei], v[(ei + 1) % 14]
            c1, c2 = edge_controls(p, q, c)
            t = np.linspace(0, 1, samples)
            curves.append(cubic(p, c1, c2, q, t))
        A, B = curves
        # compare as sets: B may be traversed in either direction
        d1 = np.abs(A - B).max()
        d2 = np.abs(A - B[::-1]).max()
        worst = max(worst, min(d1, d2))
    return worst, n_shared


# ---------------------------------------------------------------------------
# single-tile geometry vs curve strength
# ---------------------------------------------------------------------------
def shoelace(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def tile_metrics(lengths, c, samples=400):
    v, _ = TF.build_polygon(lengths, mirror_dirs=CR.UNIT_DIRS)
    v = v[:14]
    b = curved_boundary(v, c, samples)
    per = float(np.sum(np.linalg.norm(np.diff(np.vstack([b, b[:1]]), axis=0),
                                      axis=1)))
    ring = LineString(np.vstack([b, b[:1]]))
    return dict(area=abs(shoelace(b)),
                straight_area=abs(shoelace(v)),
                perimeter=per,
                straight_perimeter=float(np.sum(lengths)),
                simple=bool(ring.is_simple))


def critical_curve_strength(lengths, lo=0.0, hi=6.0, tol=1e-4, samples=800):
    """Largest c whose curved boundary is still a simple closed curve.

    Bracketed bisection: `lo` is known-simple, `hi` is grown until it is
    known-non-simple.
    """
    while tile_metrics(lengths, hi, samples)['simple']:
        hi *= 2.0
        if hi > 1e3:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if tile_metrics(lengths, mid, samples)['simple']:
            lo = mid
        else:
            hi = mid
    return lo


def tiling_overlap(c, n_iterations=1, lengths=None, samples=64):
    """Overlap fraction of the curved tiling: nonzero once bulges from tiles
    that meet only at a vertex start colliding."""
    recs, _ = placed_curved_tiles(n_iterations, lengths=lengths, c=c,
                                  samples=samples)
    polys = []
    for r in recs:
        p = Polygon(r['curve'])
        polys.append(p if p.is_valid else p.buffer(0))
    u = unary_union(polys)
    s = sum(p.area for p in polys)
    return (s - u.area) / s


def critical_tiling_strength(lengths=None, n_iterations=1, tol=1e-3,
                             thresh=1e-3, lo=0.0, hi=3.0):
    """Largest c at which the curved TILING is still (numerically) exact.

    Mating guarantees adjacent tiles agree along shared edges for every c,
    but tiles that touch only at a vertex have no such protection: past some
    c their bulges collide.  This finds that c.
    """
    if tiling_overlap(0.0, n_iterations, lengths) >= thresh:
        return np.nan          # already overlapping when straight (hat etc.)
    while tiling_overlap(hi, n_iterations, lengths) < thresh:
        hi *= 2.0
        if hi > 64:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if tiling_overlap(mid, n_iterations, lengths) < thresh:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def draw_tiling(ax, recs, c, title, cmap='twilight'):
    patches = [PathPatch(curved_path(r['verts'], c)) for r in recs]
    labels = sorted({r['label'] for r in recs})
    vals = np.array([labels.index(r['label']) for r in recs], float)
    pc = PatchCollection(patches, array=vals, cmap=cmap,
                         edgecolors='k', linewidths=0.5, alpha=0.9)
    ax.add_collection(pc)
    ax.autoscale(); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10)


def plot_sweep(n_iterations, strengths, fname):
    ncol = len(strengths)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 5))
    for ax, c in zip(np.atleast_1d(axes), strengths):
        recs, _ = placed_curved_tiles(n_iterations, c=c)
        em = edge_table(recs)
        err, nsh = mating_error(recs, em, c)
        draw_tiling(ax, recs, c,
                    f'curve strength c = {c:g}\n'
                    f'{len(recs)} tiles, {nsh} shared edges\n'
                    f'max mating error = {err:.2e}')
    fig.suptitle('Bezier-curved Spectre tiling: the curved edges mate exactly '
                 'at every curve strength\n(the S-curve is point-symmetric '
                 'about the edge midpoint, and every placement has det T = +1)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def plot_metrics(cs, fname, n_iterations=1):
    families = [('Spectre(1,1)', TF.LEN_SPECTRE, 'navy'),
                (r'Hat(1,$\sqrt{3}$)', TF.LEN_HAT, 'darkorange'),
                (r'Turtle($\sqrt{3}$,1)', TF.LEN_TURTLE, 'seagreen')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for name, L, col in families:
        m0 = tile_metrics(L, 0.0)
        a = [tile_metrics(L, c)['area'] / m0['area'] for c in cs]
        ax.plot(cs, a, '-', color=col, lw=2, label=name)
    ax.axhline(1.0, color='k', lw=0.6, ls='--')
    ax.set_xlabel('curve strength c'); ax.set_ylabel('area / straight area')
    ax.set_title('area is EXACTLY invariant under curving\n'
                 '(the two lobes of each point-symmetric S-curve cancel)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, L, col in families:
        m0 = tile_metrics(L, 0.0)
        p = [tile_metrics(L, c)['perimeter'] / m0['perimeter'] for c in cs]
        ax.plot(cs, p, '-', color=col, lw=2, label=name)
    ax.set_xlabel('curve strength c'); ax.set_ylabel('perimeter / straight')
    ax.set_title('perimeter grows: c is an isoperimetric-ratio knob\n'
                 'at fixed area and fixed tiling')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    recs_ok, _ = placed_curved_tiles(n_iterations, c=0.0)
    em = edge_table(recs_ok)
    errs, errs_mirror = [], []
    for c in cs:
        recs, _ = placed_curved_tiles(n_iterations, c=c)
        errs.append(mating_error(recs, edge_table(recs), c)[0])
        recsm, _ = placed_curved_tiles(n_iterations, c=c, mirror_one=0)
        errs_mirror.append(mating_error(recsm, em, c)[0])
    base = errs[0]
    ax.semilogy(cs, np.maximum(errs, 1e-18), 'o-', color='navy',
                label='strictly chiral tiling (det T = +1 everywhere)')
    ax.semilogy(cs, np.maximum(errs_mirror, 1e-18), 's-', color='crimson',
                label='same tiling with ONE tile mirrored')
    ax.axhline(base, color='0.5', ls='--', lw=1,
               label=f'c=0 baseline ({base:.1e}): float32 vertex table')
    ax.set_xlabel('curve strength c'); ax.set_ylabel('max mating error')
    ax.set_title('curved edges mate iff the tiling is chiral\n'
                 'J anticommutes with reflections, so a mirror tile bulges '
                 'the wrong way')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    names, c_self, c_tile = [], [], []
    fam2 = list(families) + [
        (r'chimera $\bar s$=0.25', TF.lengths_per_edge(np.full(14, 0.25)), ''),
        (r'chimera $\bar s$=0.50', TF.lengths_per_edge(np.full(14, 0.50)), '')]
    for name, L, _ in fam2:
        names.append(name)
        c_self.append(critical_curve_strength(L))
        c_tile.append(critical_tiling_strength(L, n_iterations=1))
    y = np.arange(len(names))
    ct = np.nan_to_num(np.array(c_tile), nan=0.0)
    ax.barh(y - 0.2, c_self, 0.38, color='steelblue',
            label='c* self-intersection of one tile')
    ax.barh(y + 0.2, ct, 0.38, color='indianred',
            label='c* collision in the tiling')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('critical curve strength c*')
    ax.set_title('what limits the curve strength\n'
                 'self-intersection bites first: the curved family is valid '
                 'on |c| < c*_self')
    ax.grid(alpha=0.3, axis='x'); ax.legend(fontsize=8, loc='lower right')
    for i, (a, b) in enumerate(zip(c_self, c_tile)):
        ax.text(a, i - 0.2, f' {a:.3f}', va='center', fontsize=8)
        ax.text(max(b if b == b else 0.0, 0.02), i + 0.2,
                f' {b:.3f}' if b == b else ' n/a (overlaps already at c=0)',
                va='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return {n: (a, b) for n, a, b in zip(names, c_self, c_tile)}


def plot_family(c, fname):
    fam = [('Spectre(1,1)', TF.LEN_SPECTRE),
           (r'Hat(1,$\sqrt{3}$)', TF.LEN_HAT),
           (r'Turtle($\sqrt{3}$,1)', TF.LEN_TURTLE),
           (r'chimera $\bar s$=0.5', TF.lengths_per_edge(np.full(14, 0.5)))]
    fig, axes = plt.subplots(2, len(fam), figsize=(4.2 * len(fam), 9))
    for j, (name, L) in enumerate(fam):
        v, d = TF.build_polygon(L, mirror_dirs=CR.UNIT_DIRS)
        v = v[:14] - v[:14].mean(axis=0)
        ax = axes[0, j]
        cstar = critical_curve_strength(L)
        for cc, col, a in ((0.0, 'k', 1.0), (c, 'crimson', 1.0),
                           (0.98 * cstar, 'darkorange', 0.7)):
            ax.add_patch(PathPatch(curved_path(v, cc), fill=False,
                                   edgecolor=col, lw=2, alpha=a))
        ax.autoscale(); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_title(f'{name}\nblack c=0, red c={c:g}, '
                     f'orange c=0.98 c* ({0.98*cstar:.3f})', fontsize=9)
        ax = axes[1, j]
        recs, _ = placed_curved_tiles(1, lengths=L, c=c)
        draw_tiling(ax, recs, c, f'{name} tiled at c={c:g}')
    fig.suptitle('the curved construction applied across the Tile(a,b) family',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fname, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=2)
    ap.add_argument('--c', type=float, default=0.12,
                    help='reference curve strength')
    ap.add_argument('--sweep', type=float, nargs='*',
                    default=[0.0, 0.06, 0.12, 0.20])
    args = ap.parse_args()

    lines = []

    def say(s=''):
        print(s); lines.append(s)

    say('=== agreement with bezier_spectre.calculate_control_points ===')
    say(f'max control-point discrepancy over 200 random edges: '
        f'{check_against_upstream():.2e}')
    say()

    say('=== handedness of the substitution placements ===')
    recs, defect = placed_curved_tiles(args.iterations, c=args.c)
    dets = np.array([r['det'] for r in recs])
    say(f'{len(recs)} tiles, det(T) unique values: '
        f'{np.unique(np.round(dets, 9))}  -> strictly chiral')
    em = edge_table(recs)
    shared = sum(1 for u in em.values() if len(u) == 2)
    err, _ = mating_error(recs, em, args.c)
    say(f'shared edges: {shared}   max curved mating error at c={args.c}: '
        f'{err:.2e}')
    recsm, _ = placed_curved_tiles(args.iterations, c=args.c, mirror_one=0)
    errm, _ = mating_error(recsm, em, args.c)
    say(f'with one tile mirrored: max mating error {errm:.3e}  '
        f'(ratio {errm/max(err,1e-18):.1e})')
    say()

    say('=== gap/overlap of the curved tiling (shapely, sampled boundary) ===')
    for c in args.sweep:
        rs, _ = placed_curved_tiles(args.iterations, c=c, samples=60)
        polys = []
        for r in rs:
            p = Polygon(r['curve'])
            polys.append(p if p.is_valid else p.buffer(0))
        u = unary_union(polys)
        s = sum(p.area for p in polys)
        holes = sum(Polygon(ring).area
                    for g in (u.geoms if hasattr(u, 'geoms') else [u])
                    for ring in g.interiors)
        say(f'  c={c:5.2f}  sum area {s:12.5f}  union {u.area:12.5f}  '
            f'overlap {100*(s-u.area)/s:9.2e}%  interior holes '
            f'{100*holes/u.area:9.2e}%')
    say()

    say('=== single-tile geometry vs curve strength ===')
    hdr = (f"{'c':>6s} {'area/A0':>12s} {'perim/P0':>10s} "
           f"{'isoperim Q':>11s} {'simple':>7s}")
    say(hdr); say('-' * len(hdr))
    m0 = tile_metrics(TF.LEN_SPECTRE, 0.0)
    for c in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        m = tile_metrics(TF.LEN_SPECTRE, c)
        Q = 4 * np.pi * m['area'] / m['perimeter'] ** 2
        say(f"{c:6.2f} {m['area']/m0['area']:12.10f} "
            f"{m['perimeter']/m0['perimeter']:10.5f} {Q:11.5f} "
            f"{str(m['simple']):>7s}")
    say()

    say('=== critical curve strengths c* (self-intersection onset) ===')
    png = os.path.join(OUT, 'bezier_metrics.png')
    crits = plot_metrics(np.linspace(0.0, 0.45, 19), png,
                         n_iterations=max(1, args.iterations - 1))
    for k, (a, b) in crits.items():
        say(f'  {k:28s} self-intersection c* = {a:.5f}   '
            f'tiling collision c* = {b:.5f}')
    say(f'wrote {png}')

    png = os.path.join(OUT, 'bezier_tiling_sweep.png')
    plot_sweep(args.iterations, args.sweep, png)
    say(f'wrote {png}')

    png = os.path.join(OUT, 'bezier_family.png')
    plot_family(args.c, png)
    say(f'wrote {png}')

    with open(os.path.join(OUT, 'bezier_tiling_metrics.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
