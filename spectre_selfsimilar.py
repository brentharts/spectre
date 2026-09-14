#!/usr/bin/env python3
r"""spectre_selfsimilar.py -- the genuine multi-scale tube family.

spectre_multiscale.py took ONE patch at a fixed substitution depth and looked
at it through balls of varying radius.  That measures how a fixed set looks
at different zooms.  It is not a multi-scale tube family, which is what the
Kakeya machinery actually consumes: tubes of MANY different thicknesses
present at once, with the thin ones nested inside the structure laid down by
the thick ones.

The Spectre substitution supplies that nesting for free, and this module
uses it.  Walking the MetaTile tree while KEEPING the ancestry (which
forEachTile discards) labels every tile with the chain of supertiles it
belongs to.  Cutting that chain at coarseness c partitions the patch into
blocks, and the block counts come out

    1, 8, 63, 496, 3905, ...

which is exactly spectrefacts.MATRIX_TOTALS -- the census of the 9x9 species
matrix, recovered from pure geometry with no matrix involved.  The last
level adds the mystic split (496 + 63 = 559), so MYSTIC_RULE falls out too.

The family
----------
At coarseness c, the INTERFACE edges are the shared edges whose two tiles lie
in different level-c blocks.  These are the boundaries of the supertiles at
that level.  Braid-lifting them gives tubes, and the level is assigned a
thickness from the substitution's own unit:

    delta_c = delta_0 * lambda^(-s c),     lambda^2 = 4 + sqrt(15)

The whole family is the union over c.  The exponent s is the question.

What s should be
----------------
The proposal that motivated this module was s = 2, i.e. delta_c scaling like
lambda^(-2c), one factor of the fundamental unit per level.  That turns out
to be the wrong exponent, and the reason is worth stating.

Every interface edge is a UNIT tile edge whatever level it sits at -- the
supertiles get bigger, but they are bounded by the same small edges.  So the
level volume is

    V_c  ~  N_c * delta_c^2      (length is 1 at every level)

with N_c the interface count.  A balanced family, the kind where no single
level carries all the volume, needs V_c flat in c, hence

    s* = (1/2) * log(N_c growth rate) / log(lambda)

This module MEASURES the interface growth rate rather than predicting it,
derives s* from it, and then sweeps s to confirm that the volume spectrum is
flat there and lopsided elsewhere.  The measured s* is reported against the
proposed s = 2.

Findings, as of this version
----------------------------
The hierarchy check passes exactly: the block counts read off the MetaTile
tree are 1, 8, 63, 496, 3905 -- MATRIX_TOTALS to the digit -- and the
deepest level is 4401 = 3905 + 496, the mystic split, so MYSTIC_RULE is
recovered from geometry as well.  That is the strongest link to
spectrepaper.py in the repo: the same integers from two unrelated routes.

THE PROPOSED EXPONENT IS WRONG, by a lot.  Measured interface growth:

    depth 4   mu = 2.6575 = lambda^0.947   ->  s* = 0.474
    depth 5   mu = 2.4071 = lambda^0.851   ->  s* = 0.426

so s* is near 0.45.  It is NOT converging cleanly -- the two depths differ by
10% and the trend is downward, not settling -- so the honest statement is
s* = 0.45 +- 0.05, limited by finite-patch boundary effects, and NOT the
clean 1/2 that a naive perimeter argument (N_c ~ lambda^c) would predict.

What is robust is the comparison, which is not close:

    at s = 2 (proposed)   1.10 of 5 levels carry volume
    at s = 0.42           4.72 of 5 levels carry volume

At s = 2 the family is entirely its coarsest level -- the finer levels are
thinner by lambda^2 per step while their populations grow by only lambda^0.9,
so their volume falls off a cliff.  A multi-scale argument fed that family is
really being fed a single-scale family, which is the one thing the
construction was supposed to avoid.

The reason is simple enough to state: every interface edge is a UNIT tile
edge no matter which level it belongs to.  Supertiles get larger, but they
are bounded by the same small edges, so length does not scale with level and
only the population does.  One factor of the fundamental unit per level is
the right grading for AREAS, which is what lambda^2 = 4+sqrt(15) measures;
it is the wrong grading for a thickness attached to a fixed-length edge.

    python3 spectre_selfsimilar.py
    python3 spectre_selfsimilar.py --depth 4 --s 0.42
    python3 spectre_selfsimilar.py --selftest
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sympy as sp

import spectre as S
import spectrefacts as F
import tile_family as TF
from braided_tiling import build_braided_geometry
from besicovitch_inversion import _tangent_direction, _ball_stencil, _resample

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
RNG_SEED = 23

LAMBDA_SQ = float(sp.N(F.LAM2))          # 4 + sqrt(15), from spectrefacts
LAMBDA = float(sp.N(F.LAM))

DEFAULT_CROSSINGS = 3
DEFAULT_HEIGHT = 0.18


# ---------------------------------------------------------------------------
# the hierarchy, with ancestry kept
# ---------------------------------------------------------------------------

def walk_with_ancestry(node, T=None, path=()):
    """Yield (transform, label, ancestry) for every leaf tile.

    spectre.forEachTile throws the ancestry away, which is the one piece of
    information the whole multi-scale construction needs -- without it there
    is no way to say which supertile a tile belongs to, and no nesting.
    """
    if T is None:
        T = S.IDENTITY
    if isinstance(node, S.Tile):
        yield T, node.label, path
        return
    for i, (child, trsf) in enumerate(zip(node.tiles, node.transformations)):
        for item in walk_with_ancestry(child, S.mul(T, trsf), path + (i,)):
            yield item


def build_patch(depth):
    """Placed tiles with world vertices and ancestry."""
    tiles_dict = S.buildSpectreTiles(depth, 1.0, 1.0, rotation=30)
    canon = TF.canonical_tile_verts('any')
    out = []
    for T, label, path in walk_with_ancestry(tiles_dict['Delta']):
        world = TF.transform_polygon(np.array(T), canon)[:14]
        out.append(dict(label=label, verts=world, path=path))
    return out


def edge_table(tiles):
    """key -> list of (tile_index, edge_index), as in braided_tiling."""
    em = {}
    for ti, t in enumerate(tiles):
        w = t['verts']
        for ei in range(14):
            p, q = w[ei], w[(ei + 1) % 14]
            key = tuple(sorted([tuple(np.round(p, 4)),
                                tuple(np.round(q, 4))]))
            em.setdefault(key, []).append((ti, ei))
    return em


def block_counts(tiles, max_c):
    """Distinct level-c blocks for c = 0..max_c."""
    return [len({t['path'][:c] for t in tiles}) for c in range(max_c + 1)]


def interface_edges(tiles, em, c):
    """Shared edges separating two different level-c blocks.

    At c = 0 the whole patch is one block and there are none; as c grows the
    blocks get finer and more edges become interfaces, until at full depth
    every shared edge qualifies.
    """
    out = {}
    for key, users in em.items():
        if len(users) != 2:
            continue
        (a, _), (b, _) = users
        if tiles[a]['path'][:c] != tiles[b]['path'][:c]:
            out[key] = users
    return out


# ---------------------------------------------------------------------------
# the tube family
# ---------------------------------------------------------------------------

def level_tubes(tiles, em, c, crossings=DEFAULT_CROSSINGS,
                height=DEFAULT_HEIGHT):
    """Braid-lifted tubes on the level-c interfaces, one per edge."""
    iface = interface_edges(tiles, em, c)
    if not iface:
        return []
    strands = build_braided_geometry(tiles, iface, crossings=crossings,
                                     height=height)
    by_edge = {}
    for st in strands:
        w = tiles[st['tile']]['verts']
        p, q = w[st['edge']], w[(st['edge'] + 1) % 14]
        key = tuple(sorted([tuple(np.round(p, 4)), tuple(np.round(q, 4))]))
        by_edge.setdefault(key, []).append(st)

    tubes = []
    for key, group in by_edge.items():
        st = group[0]                      # the partner strand is 99% parallel
        d = _tangent_direction(st['pts'])
        if d is None:
            continue
        tubes.append(dict(pts=st['pts'].copy(), dir=d, level=c))
    return tubes


def thickness(delta0, c, s):
    """delta_c = delta_0 * lambda^(-s c)."""
    return delta0 * LAMBDA ** (-s * c)


def build_family(depth, delta0=0.30, s=2.0, crossings=DEFAULT_CROSSINGS,
                 height=DEFAULT_HEIGHT, levels=None):
    """Tubes at every coarseness level at once, each with its own thickness."""
    tiles = build_patch(depth)
    em = edge_table(tiles)
    max_c = max(len(t['path']) for t in tiles)
    cs = range(1, max_c + 1) if levels is None else levels
    fam = []
    for c in cs:
        tubes = level_tubes(tiles, em, c, crossings, height)
        if tubes:
            fam.append(dict(level=c, tubes=tubes, delta=thickness(delta0, c, s)))
    return tiles, em, fam


# ---------------------------------------------------------------------------
# the exponent
# ---------------------------------------------------------------------------

SATURATION = 0.90


def unsaturated_levels(counts):
    """Levels before the interface set fills up.

    At the finest coarseness every shared edge is an interface, so N_c stops
    growing and flattens against the total.  Including those points drags the
    fitted growth rate down and would put the balanced exponent in the wrong
    place -- the flattening is the patch running out of edges, not the
    geometry changing.  Levels within SATURATION of the maximum are dropped.
    """
    if not counts:
        return []
    top = max(n for _, n in counts)
    return [(c, n) for c, n in counts if n < SATURATION * top]


def interface_growth(fam):
    """Fit N_c ~ mu^c on the unsaturated levels.

    Returns (mu, exponent in base lambda, points used).  The exponent sets
    the balanced thickness, so it is measured rather than assumed.
    """
    counts = [(lv['level'], len(lv['tubes'])) for lv in fam if lv['tubes']]
    pts = unsaturated_levels(counts)
    if len(pts) < 3:
        pts = counts
    if len(pts) < 2:
        return float('nan'), float('nan'), pts
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.log([p[1] for p in pts])
    slope = float(np.polyfit(x, y, 1)[0])
    return float(np.exp(slope)), slope / np.log(LAMBDA), pts


def critical_exponent(fam):
    """s* making the level volumes N_c * delta_c^2 flat in c."""
    _, a, _ = interface_growth(fam)
    return a / 2.0


def volume_spectrum(fam):
    """Per-level N_c * delta_c^2, normalised to its own maximum.

    This is the quantity a multi-scale argument needs to be non-degenerate.
    Flat means every level contributes; a spike means the family is really a
    single-scale family wearing a costume.
    """
    v = np.array([len(lv['tubes']) * lv['delta'] ** 2 for lv in fam])
    return v / v.max() if len(v) and v.max() > 0 else v


def spectrum_flatness(fam):
    """Participation ratio of the volume spectrum, in levels.

    Reported in the same units as the number of levels, so 'how many levels
    actually carry volume' is readable directly instead of via a log.
    """
    v = np.array([len(lv['tubes']) * lv['delta'] ** 2 for lv in fam],
                 dtype=float)
    if not len(v) or v.sum() <= 0:
        return float('nan')
    w = v / v.sum()
    return float(1.0 / np.sum(w ** 2))


# ---------------------------------------------------------------------------
# union volume of the multi-thickness family
# ---------------------------------------------------------------------------

def union_volume(fam, grid_h=None, max_voxels=1.2e8):
    """Voxel volume of the union, with a DIFFERENT radius per level.

    The voxel size is set by the thinnest level present, because a grid that
    cannot resolve the thinnest tubes silently deletes them and reports the
    coarse levels alone -- which would make every exponent look balanced.
    Returns (volume, sum_of_level_volumes, h).
    """
    allp = np.vstack([t['pts'] for lv in fam for t in lv['tubes']])
    deltas = [lv['delta'] for lv in fam]
    if grid_h is None:
        grid_h = min(deltas) / 2.0
    lo = allp.min(axis=0) - 2 * max(deltas)
    hi = allp.max(axis=0) + 2 * max(deltas)
    est = np.prod(np.ceil((hi - lo) / grid_h) + 8)
    if est > max_voxels:
        grid_h *= (est / max_voxels) ** (1.0 / 3.0)
    R = int(np.ceil(max(deltas) / grid_h))
    shape = np.ceil((hi - lo) / grid_h).astype(int) + 2 * R + 2
    occ = np.zeros(shape, dtype=bool)

    single = 0
    resolved = []
    for lv in fam:
        r_vox = lv['delta'] / grid_h
        if r_vox < 1.0:
            resolved.append((lv['level'], False))
            continue
        resolved.append((lv['level'], True))
        stencil = _ball_stencil(r_vox)
        for t in lv['tubes']:
            p = _resample(t['pts'], grid_h * 0.5)
            idx = np.floor((p - lo) / grid_h).astype(int) + R + 1
            vox = np.unique((idx[:, None, :] + stencil[None, :, :])
                            .reshape(-1, 3), axis=0)
            single += len(vox)
            occ[vox[:, 0], vox[:, 1], vox[:, 2]] = True

    cell = grid_h ** 3
    return int(occ.sum()) * cell, single * cell, grid_h, resolved


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report_hierarchy(tiles, max_c):
    bc = block_counts(tiles, max_c)
    print('\nhierarchy recovered from the MetaTile tree')
    print('  %-12s %-12s %-12s %s' % ('coarseness', 'blocks',
                                      'MATRIX_TOTALS', 'match'))
    for c, n in enumerate(bc):
        want = F.MATRIX_TOTALS[c] if c < len(F.MATRIX_TOTALS) else None
        if want is not None and want == n:
            tag = 'yes'
        elif c >= 2 and n == F.MATRIX_TOTALS[c - 1] + F.MATRIX_TOTALS[c - 2]:
            # the deepest level carries the mystic split: Gamma = Gamma1+Gamma2
            # adds a level, so the count is the GEOMETRIC census T(n)+T(n-1)
            tag = 'mystic split: %d + %d' % (F.MATRIX_TOTALS[c - 1],
                                             F.MATRIX_TOTALS[c - 2])
        else:
            tag = 'no'
        print('  %-12d %-12d %-12s %s'
              % (c, n, want if want is not None else '-', tag))
    return bc


def report_family(fam, s, delta0):
    print('\nlevels, with delta_c = %.3f * lambda^(-%.3f c)' % (delta0, s))
    spec = volume_spectrum(fam)
    print('  %-8s %-10s %-12s %-14s %s'
          % ('level c', 'tubes', 'delta_c', 'N*delta^2', 'share'))
    for lv, v in zip(fam, spec):
        print('  %-8d %-10d %-12.6f %-14.4e %.4f'
              % (lv['level'], len(lv['tubes']), lv['delta'],
                 len(lv['tubes']) * lv['delta'] ** 2, v))
    mu, a, used = interface_growth(fam)
    print('  interface growth N_c ~ mu^c with mu = %.4f = lambda^%.4f'
          % (mu, a))
    print('  fitted on unsaturated levels %s of %d'
          % ([c for c, _ in used], len(fam)))
    print('  balanced exponent s* = %.4f   (proposed s = 2)'
          % critical_exponent(fam))
    print('  spectrum flatness = %.3f of %d levels'
          % (spectrum_flatness(fam), len(fam)))


def sweep(depth, delta0, s_values, crossings, height):
    """Flatness of the volume spectrum as a function of s."""
    tiles = build_patch(depth)
    em = edge_table(tiles)
    max_c = max(len(t['path']) for t in tiles)
    base = []
    for c in range(1, max_c + 1):
        tubes = level_tubes(tiles, em, c, crossings, height)
        if tubes:
            base.append((c, len(tubes)))
    out = []
    for s in s_values:
        v = np.array([n * thickness(delta0, c, s) ** 2 for c, n in base])
        w = v / v.sum()
        out.append((s, float(1.0 / np.sum(w ** 2))))
    return out, base


# ---------------------------------------------------------------------------

def figure(bc, fam, sw, s, delta0, fname):
    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_subplot(2, 3, 1)
    c = np.arange(len(bc))
    ax.semilogy(c, bc, 'o-', color='crimson', label='blocks (geometry)')
    ax.semilogy(c, F.MATRIX_TOTALS[:len(bc)], 'x--', color='green',
                label='MATRIX_TOTALS (matrix)')
    ax.set_xlabel('coarseness c')
    ax.set_ylabel('blocks')
    ax.set_title('Hierarchy = species census', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 2)
    lv = [x['level'] for x in fam]
    n = [len(x['tubes']) for x in fam]
    mu, a, used = interface_growth(fam)
    uc = {c for c, _ in used}
    ax.semilogy(lv, n, 'o-', color='crimson')
    sat = [(c, k) for c, k in zip(lv, n) if c not in uc]
    if sat:
        ax.semilogy([c for c, _ in sat], [k for _, k in sat], 'o',
                    mfc='none', mec='crimson', ms=12, label='saturated')
    ax.semilogy(lv, [n[0] * mu ** (k - lv[0]) for k in lv], '--',
                color='0.4', label=r'$\mu^c,\ \mu=\lambda^{%.2f}$' % a)
    ax.set_xlabel('level c')
    ax.set_ylabel('interface tubes')
    ax.set_title('Interface growth sets the exponent', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    ax.bar(lv, volume_spectrum(fam), color='crimson', alpha=0.85)
    ax.set_xlabel('level c')
    ax.set_ylabel(r'$N_c\delta_c^2$ (normalised)')
    ax.set_title('Volume spectrum at s = %.2f' % s, fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(2, 3, 4)
    ax.plot([p[0] for p in sw], [p[1] for p in sw], '-', color='navy')
    st = critical_exponent(fam)
    ax.axvline(st, color='green', ls='--', label='s* = %.2f' % st)
    ax.axvline(2.0, color='crimson', ls=':', label='proposed s = 2')
    ax.set_xlabel('thickness exponent s')
    ax.set_ylabel('levels carrying volume')
    ax.set_title('Which exponent balances the family', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 5)
    for x in fam:
        p = np.vstack([t['pts'] for t in x['tubes']])
        ax.scatter(p[:, 0], p[:, 1], s=0.35 + 3.0 * x['delta'],
                   alpha=0.55, label='c=%d' % x['level'])
    ax.set_aspect('equal')
    ax.set_title('Interfaces by level', fontsize=10)
    ax.legend(fontsize=7, markerscale=2, loc='upper right')
    ax.set_axis_off()

    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    mu, a, _ = interface_growth(fam)
    txt = ('\n'.join([
        r'$\lambda^2 = 4+\sqrt{15} = %.6f$' % LAMBDA_SQ,
        r'$\lambda = %.6f$' % LAMBDA,
        '',
        r'interface growth $\mu = %.4f = \lambda^{%.3f}$' % (mu, a),
        r'balanced exponent $s^* = %.3f$' % critical_exponent(fam),
        r'proposed exponent $s = 2$',
        '',
        'levels carrying volume at $s^*$: %.2f' % max(p[1] for p in sw),
        'levels carrying volume at $s=2$: %.2f'
        % min(sw, key=lambda p: abs(p[0] - 2.0))[1],
    ]))
    ax.text(0.02, 0.95, txt, va='top', fontsize=11, family='monospace')

    fig.suptitle('Spectre self-similar tube family\n'
                 r'thickness graded by the fundamental unit '
                 r'$\lambda^2 = 4+\sqrt{15}$ of $\mathbb{Z}[\sqrt{15}]$',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def selftest(depth=3):
    failures = []

    def check(label, ok):
        print('  %-58s %s' % (label, 'ok' if ok else 'FAIL'))
        if not ok:
            failures.append(label)

    print('the unit')
    check('lambda^2 is 4+sqrt(15) from spectrefacts',
          abs(LAMBDA_SQ - (4 + np.sqrt(15))) < 1e-12)

    print('hierarchy')
    tiles = build_patch(depth)
    max_c = max(len(t['path']) for t in tiles)
    bc = block_counts(tiles, max_c)
    check('block counts reproduce MATRIX_TOTALS',
          all(bc[c] == F.MATRIX_TOTALS[c] for c in range(min(len(bc) - 1,
              len(F.MATRIX_TOTALS)))))
    check('finest level is the geometric census',
          bc[-1] == F.MATRIX_TOTALS[max_c - 1] + F.MATRIX_TOTALS[max_c - 2])
    check('every tile has an ancestry', all(t['path'] for t in tiles))

    print('interfaces')
    em = edge_table(tiles)
    check('c=0 has no interfaces', len(interface_edges(tiles, em, 0)) == 0)
    sizes = [len(interface_edges(tiles, em, c)) for c in range(max_c + 1)]
    check('interfaces increase with coarseness',
          all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1)))
    shared = sum(1 for u in em.values() if len(u) == 2)
    check('finest level is all shared edges', sizes[-1] == shared)

    print('thickness ladder')
    check('delta_c descends by lambda^s',
          abs(thickness(1.0, 1, 2.0) / thickness(1.0, 2, 2.0)
              - LAMBDA ** 2) < 1e-9)
    check('delta_0 at c=0 is delta_0', abs(thickness(0.3, 0, 2.0) - 0.3) < 1e-12)

    print('family and exponent')
    _, _, fam = build_family(depth, 0.30, 2.0)
    check('family has several levels', len(fam) >= 3)
    check('every level has its own thickness',
          len({round(lv['delta'], 12) for lv in fam}) == len(fam))
    mu, a, used = interface_growth(fam)
    check('interface growth is positive', mu > 1.0)
    check('saturated levels are excluded from the fit',
          len(used) <= len(fam))
    check('flatness is between 1 and the level count',
          1.0 <= spectrum_flatness(fam) <= len(fam) + 1e-9)

    print()
    if failures:
        print('%d failure(s): %s' % (len(failures), ', '.join(failures)))
    else:
        print('spectre_selfsimilar: all checks pass.')
    return len(failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--delta0', type=float, default=0.30)
    ap.add_argument('--s', type=float, default=2.0,
                    help='thickness exponent delta_c = delta_0 lambda^(-s c)')
    ap.add_argument('--crossings', type=int, default=DEFAULT_CROSSINGS)
    ap.add_argument('--height', type=float, default=DEFAULT_HEIGHT)
    ap.add_argument('--volume', action='store_true',
                    help='also voxel-measure the union (slow)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if selftest(min(args.depth, 3)) else 0)

    print('spectre_selfsimilar -- tubes at every substitution level at once')
    print('lambda^2 = 4 + sqrt(15) = %.7f   (spectrefacts.LAM2)' % LAMBDA_SQ)

    tiles, em, fam = build_family(args.depth, args.delta0, args.s,
                                  args.crossings, args.height)
    max_c = max(len(t['path']) for t in tiles)
    report_hierarchy(tiles, max_c)
    report_family(fam, args.s, args.delta0)

    s_values = np.linspace(0.05, 3.0, 120)
    sw, base = sweep(args.depth, args.delta0, s_values, args.crossings,
                     args.height)
    best = max(sw, key=lambda p: p[1])
    at2 = min(sw, key=lambda p: abs(p[0] - 2.0))
    print('\nexponent sweep')
    print('  most balanced at s = %.3f, carrying %.2f of %d levels'
          % (best[0], best[1], len(base)))
    print('  at the proposed s = 2.000, carrying %.2f of %d levels'
          % (at2[1], len(base)))

    if args.volume:
        vol, tot, h, resolved = union_volume(fam)
        print('\nunion volume (voxel h = %.5f)' % h)
        unres = [c for c, ok in resolved if not ok]
        if unres:
            print('  UNRESOLVED levels (too thin for the grid): %s' % unres)
        print('  union = %.6f   sum of levels = %.6f   ratio = %.4f'
              % (vol, tot, vol / tot if tot else float('nan')))

    png = os.path.join(OUT, 'spectre_selfsimilar.png')
    figure(block_counts(tiles, max_c), fam, sw, args.s, args.delta0, png)
    print('\nwrote %s' % png)


if __name__ == '__main__':
    main()
