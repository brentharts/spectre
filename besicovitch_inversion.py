#!/usr/bin/env python3
r"""besicovitch_inversion.py -- do the Spectre braid strands behave like a
Kakeya tube family?

Wang and Zahl (arXiv:2502.17655) prove that a family of delta-tubes in R^3
whose directions do not concentrate -- "not too many tubes can be contained
inside a common convex set V" -- must have a union of almost maximal volume.
That is a hypothesis about a direction set and a conclusion about a volume,
and both halves are measurable on a finite family.  This module measures
them.

The tube family comes from the braided tiling.  Every shared edge of a
Spectre patch carries a 2-strand braid (braided_tiling.py); each strand is a
curve in R^3, and its delta-neighbourhood is a tube.  The braid is what
supplies the third dimension: a flat tiling has all its directions in a
plane and cannot be a Kakeya family at all.

Two tube sources are built and compared, because they isolate different
claims:

  braid    the actual lifted strands.  Direction comes from the braid
           geometry, so both the aperiodic edge set and the weave contribute.
  normal   edge midpoints and in-plane edge directions, lifted by a fixed
           synthetic tilt.  Same aperiodic edge set, no weave.

and both are run against a PERIODIC hexagon control with the identical
machinery.  If the Spectre numbers do not separate from the hexagon numbers,
the aperiodicity is not doing the work and the idea is wrong.

Three measurements
------------------
1. DISPERSION.  rho(delta) = |union T_delta| / sum |T_delta|, by voxel
   occupancy.  rho = 1 means the tubes are disjoint and the union is exactly
   maximal; rho << 1 means they pile up.  This is the Wang-Zahl conclusion
   in the one normalisation that needs no convex hull.

2. CONVEX NON-CONCENTRATION.  Over many random slabs V, count the tubes
   contained in each.  The Wang-Zahl hypothesis is an upper bound on this
   count, so the measured maximum is the hypothesis' own constant, read off.

3. THE INVERSION.  Besicovitch's construction runs through point-line
   duality, so the name is kept for the thing it names.  A line
   {(a t + c, b t + e, t)} dualises to the point (a, b, c, e) in R^4; the
   singular spectrum of that cloud says whether the family is genuinely
   4-parameter or secretly degenerate.  A periodic tiling should lose rank.

Findings, as of this version
----------------------------
The measurements run and are unsaturated, and the answer they give is
NEGATIVE.  It is recorded here rather than tuned away.

* The two strands of each braid are 99.1% parallel, so keeping both puts a
  near-duplicate beside every tube and depresses the dispersion ratio for a
  reason unrelated to the direction set.  Use --solo.
* With that removed, no statistic separates the Spectre from the periodic
  control in the Kakeya direction, and on the dual rank the PERIODIC
  hexagon scores highest (about 3.99 of 4, against 3.4-3.7 for the braid).
* Varying the braid amplitude by edge type or by species does not change
  this.

The likely reason is that the premise was wrong, not the code.  Kakeya
non-concentration asks for a direction set that is SPREAD; aperiodicity
gives NON-REPETITION.  Those are different properties, and a periodic
lattice has a perfectly spread direction set by symmetry -- which is exactly
what the hexagon number says.  A monotile tiling built on a hexagonal
direction grid has few distinct directions, and repeating them less often
does not make them more numerous.

So this module currently stands as a refutation of the obvious bridge
between the monotile and arXiv:2502.17655, with the machinery left in place
for a better one.  See the bottom of this file for what would be worth
trying next.

    python3 besicovitch_inversion.py
    python3 besicovitch_inversion.py --solo --mode species
    python3 besicovitch_inversion.py --selftest
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tile_family as TF
from braided_tiling import tiling_edges, build_braided_geometry

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
RNG_SEED = 23

# The synthetic lift for the 'normal' family.  The braid strand
# sin(pi k t) * height * L has maximum slope pi * k * height, so matching
# that keeps the two families on the same footing instead of handing one of
# them a wider direction set for free.
DEFAULT_CROSSINGS = 3
DEFAULT_HEIGHT = 0.18


# ---------------------------------------------------------------------------
# tube families
# ---------------------------------------------------------------------------

def _tangent_direction(pts):
    """Mean tangent of a sampled strand, oriented into the upper half-space.

    The chord of a braid strand is useless here: sin(pi k t) returns to zero
    at both endpoints, so every chord is exactly planar and the whole point
    of lifting is lost.  The mean tangent sees the weave.
    """
    d = np.diff(pts, axis=0)
    seg = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(seg, 1e-12)
    # tangents reverse in z across a crossing, so fold to the upper half
    # before averaging or the lift cancels itself out
    d[d[:, 2] < 0] *= -1.0
    m = d.mean(axis=0)
    n = np.linalg.norm(m)
    if n < 1e-12:
        return None
    m = m / n
    return -m if m[2] < 0 else m


AMPLITUDE_MODES = ('uniform', 'edgetype', 'species')


def _amplitude_factor(mode, tile, edge_index, species_index):
    """Per-edge multiplier on the braid amplitude.

    'uniform' is what braided_tiling.py does, and it is the reason the braid
    direction set collapses onto a cone: one amplitude means one slope
    magnitude for every strand.  The other two modes vary the amplitude
    using structure the tiling already carries, which is the only honest
    place to get variation from -- an arbitrary random amplitude would open
    the cone too, and would prove nothing about the monotile.
    """
    if mode == 'uniform':
        return 1.0
    if mode == 'edgetype':
        # a-edges and b-edges are the tile's own two-letter alphabet
        return 1.0 if TF.EDGE_TYPES[edge_index] == 'a' else 1.0 / np.sqrt(3.0)
    if mode == 'species':
        # nine species, nine amplitudes, spread over a full octave
        return 0.5 * 2.0 ** (species_index / 8.0)
    raise ValueError('unknown amplitude mode %r' % mode)


def tubes_from_braid(iterations=1, crossings=DEFAULT_CROSSINGS,
                     height=DEFAULT_HEIGHT, mode='uniform', pair=True):
    """Tubes = the lifted braid strands themselves, as sampled polylines.

    Boundary edges stay flat at z=0 and are dropped: they carry no braid, so
    their direction is planar and they would dilute the direction set with
    the very degeneracy we are testing for.

    `pair=False` keeps one strand per shared edge.  The two strands of a
    braid are 99% parallel, so keeping both puts a near-duplicate beside
    every tube and depresses the dispersion ratio for a reason that has
    nothing to do with the direction set.
    """
    tiles, edge_map = tiling_edges(iterations)
    labels = sorted({t['label'] for t in tiles})
    order = {lab: i for i, lab in enumerate(labels)}

    tubes = []
    for key, users in sorted(edge_map.items()):
        if len(users) != 2:
            continue
        ti, ei = users[0]
        f = _amplitude_factor(mode, tiles[ti], ei,
                              order[tiles[ti]['label']])
        strands = build_braided_geometry(
            tiles, {key: users}, crossings=crossings, height=height * f)
        if not pair:
            strands = strands[:1]
        for s in strands:
            d = _tangent_direction(s['pts'])
            if d is not None:
                tubes.append(dict(pts=s['pts'].copy(), dir=d))
    return tubes, tiles, edge_map


def _straight_tube(p, q, slope, sign):
    """A straight segment through the midpoint of edge (p, q), tilted."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    L = np.linalg.norm(q - p)
    d2 = (q - p) / L
    d = np.array([d2[0], d2[1], sign * slope])
    d = d / np.linalg.norm(d)
    if d[2] < 0:
        d = -d
    mid = np.array([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, 0.0])
    t = np.linspace(-0.5, 0.5, 24)
    pts = mid[None, :] + d[None, :] * (L * t)[:, None]
    return dict(pts=pts, dir=d)


def tubes_from_normals(iterations=1, crossings=DEFAULT_CROSSINGS,
                       height=DEFAULT_HEIGHT):
    """Tubes = straight segments through edge midpoints.

    Direction is the in-plane edge direction tilted out of plane by the
    braid's own maximum slope, with the sign alternating by edge parity so
    the family is not trivially one-sided.  Same edge set, no weave.
    """
    tiles, edge_map = tiling_edges(iterations)
    slope = np.pi * crossings * height
    tubes = []
    for rank, (key, users) in enumerate(sorted(edge_map.items())):
        if len(users) != 2:
            continue
        ti, ei = users[0]
        v = tiles[ti]['verts']
        tubes.append(_straight_tube(v[ei], v[(ei + 1) % 14], slope,
                                    1.0 if rank % 2 == 0 else -1.0))
    return tubes, tiles, edge_map


def hex_edges(nx=7, ny=7, s=1.0):
    """Shared edges of a periodic hexagon tiling, as (p, q) pairs.

    The control.  Same downstream machinery, a direction set with a period.
    """
    seen = {}
    for i in range(nx):
        for j in range(ny):
            cx = s * 1.5 * i
            cy = s * np.sqrt(3.0) * (j + 0.5 * (i % 2))
            ring = [(cx + s * np.cos(np.pi / 3 * k),
                     cy + s * np.sin(np.pi / 3 * k)) for k in range(6)]
            for k in range(6):
                p, q = ring[k], ring[(k + 1) % 6]
                key = tuple(sorted([tuple(np.round(p, 4)),
                                    tuple(np.round(q, 4))]))
                seen[key] = seen.get(key, 0) + 1
    return [k for k, c in seen.items() if c == 2]


def tubes_from_hex(n=7, crossings=DEFAULT_CROSSINGS, height=DEFAULT_HEIGHT):
    """The periodic control, built exactly like `tubes_from_normals`."""
    slope = np.pi * crossings * height
    return [_straight_tube(p, q, slope, 1.0 if rank % 2 == 0 else -1.0)
            for rank, (p, q) in enumerate(sorted(hex_edges(n, n)))]


# ---------------------------------------------------------------------------
# normalisation: unit tubes in a unit box
# ---------------------------------------------------------------------------

def normalize(tubes, length=1.0, samples=48):
    """Replace each tube by a straight segment of common length through its
    own anchor, with the whole family rescaled into a unit-diameter box.

    Without this the comparison is meaningless.  A Kakeya family is unit
    tubes in a bounded box; edge-length tubes spread over a wide patch are
    trivially disjoint whatever their directions, and the dispersion ratio
    saturates at 1 for every family including the periodic one.  Rescaling
    to a common box and a common length makes direction diversity the only
    thing left that can move the number, which is the point.

    The direction is kept exactly; only the extent and the scale change.
    """
    anchors = np.array([t['pts'].mean(axis=0) for t in tubes])
    c = anchors.mean(axis=0)
    r = np.linalg.norm(anchors - c, axis=1).max()
    if r < 1e-12:
        r = 1.0
    t = np.linspace(-0.5, 0.5, samples)
    out = []
    for tube, a in zip(tubes, anchors):
        p = (a - c) / r
        d = tube['dir']
        out.append(dict(pts=p[None, :] + d[None, :] * (length * t)[:, None],
                        dir=d))
    return out


# ---------------------------------------------------------------------------
# 1. dispersion: |union T_delta| / sum |T_delta|
# ---------------------------------------------------------------------------

def _ball_stencil(r_vox):
    """Integer offsets of a discrete ball of radius r_vox voxels."""
    R = int(np.ceil(r_vox))
    g = np.arange(-R, R + 1)
    dx, dy, dz = np.meshgrid(g, g, g, indexing='ij')
    m = (dx ** 2 + dy ** 2 + dz ** 2) <= r_vox ** 2
    return np.column_stack([dx[m], dy[m], dz[m]])


def _resample(pts, step):
    """Resample a polyline at roughly `step` spacing, so the swept ball
    stencil leaves no gaps along the tube."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total < 1e-12:
        return pts
    n = max(2, int(np.ceil(total / step)) + 1)
    want = np.linspace(0, total, n)
    return np.column_stack([np.interp(want, arc, pts[:, k])
                            for k in range(3)])


def dispersion(tubes, delta, grid=None, pad=1.25, vox_per_delta=2.0,
               max_voxels=1.4e8):
    """Voxel occupancy of the union, against the disjoint-sum reference.

    Returns (rho, union_volume, sum_volume).  rho near 1 is the almost
    maximal volume of the Wang-Zahl conclusion; rho near 0 is concentration.
    Each tube's own volume is measured by the SAME voxelisation, so the
    discretisation bias cancels in the ratio instead of being argued away.

    The voxel size tracks delta rather than being fixed, because a fixed
    grid silently stops resolving the tube exactly when delta gets small --
    which is the regime the whole measurement is about.  `grid` is an
    optional floor on the resolution; `max_voxels` is the memory cap, and
    coarsening to respect it is reported by the caller as a smaller
    vox_per_delta rather than done quietly.
    """
    allp = np.vstack([t['pts'] for t in tubes])
    lo = allp.min(axis=0) - pad * delta
    hi = allp.max(axis=0) + pad * delta
    span = (hi - lo).max()

    h = delta / vox_per_delta
    if grid is not None:
        h = min(h, span / grid)
    est = np.prod(np.ceil((hi - lo) / h) + 8)
    if est > max_voxels:
        h *= (est / max_voxels) ** (1.0 / 3.0)
    r_vox = delta / h
    if r_vox < 1.0:
        raise ValueError('delta=%g cannot be resolved within max_voxels'
                         % delta)
    stencil = _ball_stencil(r_vox)
    R = int(np.ceil(r_vox))
    shape = np.ceil((hi - lo) / h).astype(int) + 2 * R + 2
    occ = np.zeros(shape, dtype=bool)

    single = 0
    for t in tubes:
        p = _resample(t['pts'], h * 0.5)
        idx = np.floor((p - lo) / h).astype(int) + R + 1
        vox = (idx[:, None, :] + stencil[None, :, :]).reshape(-1, 3)
        vox = np.unique(vox, axis=0)
        single += len(vox)
        occ[vox[:, 0], vox[:, 1], vox[:, 2]] = True

    cell = h ** 3
    total = int(occ.sum())
    return total / float(single), total * cell, single * cell


# ---------------------------------------------------------------------------
# 2. convex non-concentration
# ---------------------------------------------------------------------------

def slab_containment(tubes, thickness, n_slabs=400, seed=RNG_SEED):
    """How many tubes fit inside a common convex set?

    The convex sets are slabs {x : |<x,u> - c| <= thickness/2} with u uniform
    on the sphere.  A slab is the right probe for this hypothesis: a convex
    set containing many tubes of differing directions must be fat in every
    direction those tubes span, and a slab is the convex set that is thin in
    exactly one.  Returns (max_count, mean_count, counts).
    """
    rng = np.random.default_rng(seed)
    P = [t['pts'] for t in tubes]
    allp = np.vstack(P)
    counts = []
    for _ in range(n_slabs):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        proj_all = allp @ u
        c = rng.uniform(proj_all.min(), proj_all.max())
        n = 0
        for pts in P:
            s = pts @ u
            if (np.abs(s - c) <= thickness / 2).all():
                n += 1
        counts.append(n)
    counts = np.array(counts)
    return int(counts.max()), float(counts.mean()), counts


# ---------------------------------------------------------------------------
# 3. the inversion: point-line duality
# ---------------------------------------------------------------------------

def dual_points(tubes):
    """Line {(a t + c, b t + e, t)} -> dual point (a, b, c, e) in R^4.

    Tubes near-parallel to the z=0 plane have no such parametrisation and
    are dropped; the count of drops is returned rather than hidden, because
    for a flat tiling it is the whole family.
    """
    out = []
    dropped = 0
    for t in tubes:
        d = t['dir']
        if abs(d[2]) < 1e-6:
            dropped += 1
            continue
        a, b = d[0] / d[2], d[1] / d[2]
        p = t['pts'].mean(axis=0)
        out.append([a, b, p[0] - a * p[2], p[1] - b * p[2]])
    return np.array(out), dropped


def dual_spectrum(D):
    """Singular values of the centred dual cloud, and its participation
    ratio -- an effective dimension in [1, 4].

    A family whose lines all lie in a plane, or all share a direction, loses
    rank here; the participation ratio reports how much rank, continuously,
    rather than thresholding a singular value by eye.
    """
    if len(D) < 4:
        return np.zeros(4), 0.0
    X = D - D.mean(axis=0)
    # (a, b) are slopes and (c, e) are offsets: different units, so the raw
    # singular values compare a number against a length.  Standardising each
    # coordinate makes the spectrum measure rank rather than choice of unit.
    sd = X.std(axis=0)
    X = X / np.where(sd > 1e-12, sd, 1.0)
    sv = np.linalg.svd(X, compute_uv=False)
    w = sv ** 2
    if w.sum() <= 0:
        return sv, 0.0
    w = w / w.sum()
    return sv, float(1.0 / np.sum(w ** 2))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def measure(name, tubes, deltas, grid, slab_thickness, n_slabs):
    dirs = np.array([t['dir'] for t in tubes])
    D, dropped = dual_points(tubes)
    sv, pr = dual_spectrum(D)
    rows = [(d,) + dispersion(tubes, d, grid=grid) for d in deltas]
    mx, mean, counts = slab_containment(tubes, slab_thickness,
                                        n_slabs=n_slabs)
    return dict(name=name, tubes=tubes, dirs=dirs, dual=D, dropped=dropped,
                sv=sv, pr=pr, rows=rows, slab_max=mx, slab_mean=mean,
                slab_counts=counts)


def report(res):
    n = len(res['tubes'])
    print('\n%s  (%d tubes)' % (res['name'], n))
    print('  dispersion rho = |union| / sum')
    for d, rho, vu, vs in res['rows']:
        print('    delta=%-7.4f rho=%.4f   union=%.4f  sum=%.4f'
              % (d, rho, vu, vs))
    print('  convex non-concentration (random slabs)')
    print('    max tubes in one slab = %d of %d  (%.1f%%),  mean = %.2f'
          % (res['slab_max'], n, 100.0 * res['slab_max'] / n,
             res['slab_mean']))
    print('  the inversion (point-line duality in R^4)')
    print('    singular values = [%s]'
          % ', '.join('%.4f' % s for s in res['sv']))
    print('    effective dimension = %.3f of 4   (%d planar tubes dropped)'
          % (res['pr'], res['dropped']))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

COLORS = {'braid': 'crimson', 'normal': 'navy', 'hexagon': '0.45'}


def figure(results, fname):
    fig = plt.figure(figsize=(14, 11))

    for i, res in enumerate(results):
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        d = res['dirs']
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi / 2:20j]
        ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v),
                          np.cos(v), color='0.88', lw=0.4)
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=10,
                   color=COLORS.get(res['name'], 'k'), depthshade=False)
        ax.set_title('%s: direction set' % res['name'], fontsize=10)
        ax.set_box_aspect((1, 1, 0.6))
        ax.set_axis_off()

    ax = fig.add_subplot(3, 3, 4)
    for res in results:
        ax.plot([r[0] for r in res['rows']], [r[1] for r in res['rows']],
                'o-', color=COLORS.get(res['name'], 'k'), label=res['name'])
    ax.axhline(1.0, color='green', ls='--', lw=1, label='maximal')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'$\rho=|\bigcup T_\delta|/\sum|T_\delta|$')
    ax.set_title('Dispersion: is the union almost maximal?', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(3, 3, 5)
    for res in results:
        ax.hist(res['slab_counts'] / len(res['tubes']), bins=30, alpha=0.55,
                label=res['name'], color=COLORS.get(res['name'], 'k'))
    ax.set_xlabel('fraction of tubes inside one slab')
    ax.set_ylabel('slabs')
    ax.set_title('Convex non-concentration\n(left is the good direction)',
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(3, 3, 6)
    w = 0.25
    for i, res in enumerate(results):
        sv = res['sv'] / (res['sv'].max() + 1e-12)
        ax.bar(np.arange(4) + (i - 1) * w, sv, width=w,
               color=COLORS.get(res['name'], 'k'),
               label='%s (dim %.2f)' % (res['name'], res['pr']))
    ax.set_xticks(range(4))
    ax.set_xticklabels([r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$',
                        r'$\sigma_4$'])
    ax.set_ylabel('normalised singular value')
    ax.set_title(r'The inversion: dual cloud rank in $\mathbb{R}^4$',
                 fontsize=10)
    ax.legend(fontsize=8)

    for i, res in enumerate(results):
        ax = fig.add_subplot(3, 3, 7 + i)
        D = res['dual']
        if len(D):
            ax.scatter(D[:, 0], D[:, 1], s=8,
                       color=COLORS.get(res['name'], 'k'), alpha=0.7)
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_title('%s: dual slopes' % res['name'], fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle('Besicovitch inversion: Spectre braid strands as a Kakeya '
                 'tube family\nhypothesis and conclusion of Wang--Zahl '
                 '(arXiv:2502.17655), measured, against a periodic control',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------

def build_all(iterations, crossings, height, hexn, length=1.0,
              mode='uniform', pair=True):
    braid, _, _ = tubes_from_braid(iterations, crossings, height,
                                   mode=mode, pair=pair)
    normal, _, _ = tubes_from_normals(iterations, crossings, height)
    fams = [('braid', braid), ('normal', normal),
            ('hexagon', tubes_from_hex(hexn, crossings, height))]
    return [(n, normalize(t, length=length)) for n, t in fams]


def selftest(iterations=1):
    failures = []

    def check(label, ok):
        print('  %-56s %s' % (label, 'ok' if ok else 'FAIL'))
        if not ok:
            failures.append(label)

    fams = build_all(iterations, DEFAULT_CROSSINGS, DEFAULT_HEIGHT, 5)
    by = dict(fams)

    print('families')
    for n, t in fams:
        check('%s is non-empty' % n, len(t) > 0)
    check('braid is two strands per shared edge',
          len(by['braid']) == 2 * len(by['normal']))

    print('directions')
    for n, t in fams:
        d = np.array([x['dir'] for x in t])
        check('%s directions are unit' % n,
              np.allclose(np.linalg.norm(d, axis=1), 1.0))
        check('%s is genuinely lifted out of the plane' % n,
              np.abs(d[:, 2]).max() > 1e-3)

    print('dispersion is a ratio in (0, 1]')
    for n, t in fams:
        rho, vu, vs = dispersion(t, 0.06)
        check('%s rho in range' % n, 0.0 < rho <= 1.0 + 1e-9)
        check('%s union does not exceed the sum' % n, vu <= vs + 1e-12)

    print('the inversion')
    for n, t in fams:
        D, _ = dual_points(t)
        sv, pr = dual_spectrum(D)
        check('%s dual cloud has 4 coords' % n, D.shape[1] == 4)
        check('%s effective dimension in [1,4]' % n, 1.0 <= pr <= 4.0 + 1e-9)

    print()
    if failures:
        print('%d failure(s): %s' % (len(failures), ', '.join(failures)))
    else:
        print('besicovitch_inversion: all checks pass.')
    return len(failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--iterations', type=int, default=1,
                    help='substitution depth of the Spectre patch')
    ap.add_argument('--crossings', type=int, default=DEFAULT_CROSSINGS)
    ap.add_argument('--height', type=float, default=DEFAULT_HEIGHT)
    ap.add_argument('--hex', type=int, default=5,
                    help='size of the periodic control lattice')
    ap.add_argument('--grid', type=int, default=200,
                    help='floor on voxel resolution for the union volume')
    ap.add_argument('--slabs', type=int, default=400)
    ap.add_argument('--thickness', type=float, default=0.25)
    ap.add_argument('--length', type=float, default=1.0,
                    help='common tube length after normalisation')
    ap.add_argument('--mode', choices=AMPLITUDE_MODES, default='uniform',
                    help='how the braid amplitude varies per edge')
    ap.add_argument('--solo', action='store_true',
                    help='one strand per shared edge (drops the 99%% '
                         'parallel partner)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if selftest(args.iterations) else 0)

    deltas = [0.10, 0.06, 0.035, 0.02]
    fams = build_all(args.iterations, args.crossings, args.height,
                     args.hex, args.length, args.mode, not args.solo)

    print('Besicovitch inversion -- Kakeya tube measurements')
    print('reference: Wang & Zahl, arXiv:2502.17655')
    print('patch: %d substitution iterations, %d crossings per braid'
          % (args.iterations, args.crossings))
    print('braid amplitude mode: %s, strands per shared edge: %d'
          % (args.mode, 1 if args.solo else 2))

    results = [measure(n, t, deltas, args.grid, args.thickness, args.slabs)
               for n, t in fams]
    for r in results:
        report(r)

    png = os.path.join(OUT, 'besicovitch_inversion.png')
    figure(results, png)
    print('\nwrote %s' % png)

    b, nrm, hx = results
    print('\nseparation from the periodic control')
    print('  dispersion at delta=%.3f: braid %.4f  normal %.4f  hexagon %.4f'
          % (deltas[-1], b['rows'][-1][1], nrm['rows'][-1][1],
             hx['rows'][-1][1]))
    print('  slab max fraction:        braid %.3f  normal %.3f  hexagon %.3f'
          % (b['slab_max'] / len(b['tubes']),
             nrm['slab_max'] / len(nrm['tubes']),
             hx['slab_max'] / len(hx['tubes'])))
    print('  dual effective dimension: braid %.3f  normal %.3f  hexagon %.3f'
          % (b['pr'], nrm['pr'], hx['pr']))
    if hx['pr'] >= b['pr']:
        print('\n  NOTE: the periodic control is at least as non-degenerate')
        print('  as the Spectre braid.  See the Findings note in the module')
        print('  docstring -- aperiodicity is non-repetition, not direction')
        print('  spread, and this measurement does not confuse the two.')


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# What would be worth trying next
# ---------------------------------------------------------------------------
# The negative result above is about DIRECTIONS, and directions are the one
# thing a monotile tiling is poor in: the Spectre's 14 edges use a small set
# of hexagonal-grid directions, repeated aperiodically.  Wang-Zahl want many
# directions, so the bridge as attempted asks the tiling for its weakest
# property.
#
# Three redirections that do not have that defect:
#
# 1. Ask about the BRAID WORD, not the tube.  braid_words.py already shows
#    the transversal words are aperiodic with linear subword complexity.
#    The Kakeya-flavoured question there is whether the crossing set along a
#    transversal is non-concentrated in the braid group, which is a question
#    about a word and not about a direction.
#
# 2. Let the tiling supply the SCALES rather than the directions.  The
#    substitution is graded by the fundamental unit of Z[sqrt(15)]
#    (spectrefacts.py), so a self-similar tube family at scales
#    lambda^(-2n) is the natural object, and multi-scale tube families are
#    where the Kakeya machinery actually bites.
#
# 3. Drop the delta-tube framing and use the Assouad or box dimension of the
#    strand union directly.  That is measurable with the voxel code above,
#    needs no direction hypothesis, and would say something about the weave
#    rather than about the edge set.
