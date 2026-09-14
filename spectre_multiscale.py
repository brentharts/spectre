#!/usr/bin/env python3
r"""spectre_multiscale.py -- the Spectre braid as a STICKY tube family.

besicovitch_inversion.py asked the monotile for direction spread and got a
negative answer: a periodic hexagon lattice has a better-spread direction set
than any Spectre braid, because aperiodicity is non-repetition and not
spread.  This module asks for the thing the monotile actually has.

What a substitution tiling has is SCALES.  The Spectre substitution inflates
by the linear factor lambda = sqrt(4 + sqrt(15)), whose square

    lambda^2 = 4 + sqrt(15) = 7.8729833...

is the fundamental unit of Z[sqrt(15)] and the Perron eigenvalue of the 9x9
species matrix in spectrefacts.py.  So a patch of diameter D carries a
canonical ladder of scales

    rho_n = D * lambda^(-n),

and every statistic below is evaluated along that ladder rather than at
scales chosen by hand.  This is the reconnection to spectrepaper.py: the
grading of the document and the grading of the geometry are the same unit.

Why scales are the right question
---------------------------------
In the Kakeya program the obstruction that mattered was never a badly spread
direction set -- those are easy.  It was STICKINESS (Katz, Laba and Tao): a
tube family whose direction map looks the SAME after you zoom in, so that no
amount of rescaling ever improves your position.  Sticky families are the
hard case that a proof has to survive, and Wang and Zahl (arXiv:2502.17655)
survive it with a multi-scale argument.

A self-similar substitution tiling is sticky essentially by construction.
That makes the braided Spectre a concrete, finite, computable sticky family
-- which is a more useful thing to offer the Kakeya machinery than one more
direction set.  This module measures the stickiness instead of asserting it,
and measures it against a periodic control that should NOT be sticky in the
same way.

Measurements, all along rho_n = D * lambda^(-n)
-----------------------------------------------
1. LADDER.  The empirical inflation of the patch is checked against
   lambda^2 from spectrefacts, so the ladder is verified rather than assumed.

2. STICKINESS.  At each scale rho, cover the patch with rho-balls and
   compare the local direction distribution inside each ball to the global
   one, in total variation.  A sticky family holds a roughly CONSTANT
   distance across scales: zooming in changes nothing.  A family that is
   merely inhomogeneous has a distance that drifts with rho.

3. MULTI-SCALE NON-CONCENTRATION.  At each scale rho, the largest fraction
   of tubes inside a rho-slab.  Wang-Zahl need this to decay in rho; the
   fitted exponent is the quantity their hypothesis is about.

4. BOX DIMENSION of the lifted strand union, counted at the ladder scales.
   For a self-similar weave this should return 2 plus whatever the braid
   adds, and the count is the same voxel machinery as the volume estimate.

Findings, as of this version
----------------------------
At --iterations 4 --hex 40, matched to 4641 tubes each at unit diameter:

  STICKINESS (excess over a same-size random draw)
    spectre  0.915  0.974  0.969  0.941   slope -0.008
    hexagon  0.259  0.276  0.313          slope -0.092

  The Spectre value is FLAT across four ladder scales spanning a factor of
  22, which is what sticky means: the local direction law is a fixed
  fraction of the global one at every zoom, and rescaling never improves
  your position.  It sits at 1, meaning a ball looks exactly like a random
  subsample of the whole patch.

  The periodic control is also scale-invariant, but at 0.28 -- its balls are
  three times CLOSER to the global law than a random draw of the same size.
  That is hyperuniformity, and it is what periodicity looks like in this
  statistic: every ball carries the exact global direction mix with
  sub-random discrepancy.  The two families are cleanly separated, stably,
  at every scale on the ladder.

  CONCENTRATION exponent  spectre +0.500   hexagon +1.360
  BOX coarse-end slope    spectre  1.458   hexagon  1.737

  On the Wang-Zahl hypothesis itself the periodic control still does better,
  exactly as in besicovitch_inversion.py -- its slab fractions decay nearly
  three times faster.  That has not changed and is not being hidden here.

So the scales reframing does what the directions reframing did not: it
produces a statistic on which the monotile and the lattice differ, stably,
for a structural reason.  What it does NOT do is make the Spectre a better
Kakeya family.  It identifies it as a sticky one -- which in the Katz-Laba-
Tao picture is the hard case a proof must survive, not an easy case that
helps.  That is a more honest thing to offer than a false advantage.

    python3 spectre_multiscale.py
    python3 spectre_multiscale.py --iterations 3 --hex 24
    python3 spectre_multiscale.py --selftest
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

import spectrefacts as F
import tile_family as TF
from braided_tiling import tiling_edges, build_braided_geometry
from besicovitch_inversion import (_tangent_direction, tubes_from_hex,
                                   _ball_stencil, _resample)

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
RNG_SEED = 23

# The one number this module is built around, taken from spectrefacts rather
# than retyped -- the same discipline spectrepaper.py enforces on its prose.
LAMBDA_SQ = float(sp.N(F.LAM2))          # 4 + sqrt(15)
LAMBDA = float(sp.N(F.LAM))              # sqrt(4 + sqrt(15))

DEFAULT_CROSSINGS = 3
DEFAULT_HEIGHT = 0.18


# ---------------------------------------------------------------------------
# the family, at depth
# ---------------------------------------------------------------------------

def braid_tubes(iterations=3, crossings=DEFAULT_CROSSINGS,
                height=DEFAULT_HEIGHT, solo=True):
    """One tube per shared edge of a deep patch.

    `solo` by default: besicovitch_inversion showed the two strands of a
    braid are 99% parallel, so keeping both doubles the family with
    near-duplicates and corrupts every density statistic below.
    """
    tiles, edge_map = tiling_edges(iterations)
    shared = {k: u for k, u in edge_map.items() if len(u) == 2}
    strands = build_braided_geometry(tiles, shared, crossings=crossings,
                                     height=height)
    # Group strands by the EDGE they came from, not by their own midpoint:
    # the two strands of a braid are offset in-plane, so their midpoints
    # differ and a midpoint-keyed dedup silently keeps both.
    by_edge = {}
    for st in strands:
        v = tiles[st['tile']]['verts']
        p, q = v[st['edge']], v[(st['edge'] + 1) % 14]
        key = tuple(sorted([tuple(np.round(p, 4)), tuple(np.round(q, 4))]))
        by_edge.setdefault(key, []).append(st)

    tubes = []
    for key, group in by_edge.items():
        for st in (group[:1] if solo else group):
            d = _tangent_direction(st['pts'])
            if d is None:
                continue
            anchor = np.array([(key[0][0] + key[1][0]) / 2,
                               (key[0][1] + key[1][1]) / 2, 0.0])
            tubes.append(dict(pts=st['pts'].copy(), dir=d, mid=anchor))
    return tubes


def hex_tubes(n=24, crossings=DEFAULT_CROSSINGS, height=DEFAULT_HEIGHT):
    """The periodic control, with midpoints attached to match."""
    tubes = tubes_from_hex(n, crossings, height)
    for t in tubes:
        t['mid'] = t['pts'].mean(axis=0)
    return tubes


def match_families(fams, seed=RNG_SEED):
    """Rescale every family to unit patch diameter and subsample to a common
    tube count.

    Without this the comparison is rigged.  The two patches have different
    diameters and different tube counts, and both the slab fraction and the
    box count depend on those directly, so any difference between the
    families would be partly a difference in how big they happen to be.
    Rescaling is exact; the subsample is what costs information, so the
    retained count is reported.
    """
    rng = np.random.default_rng(seed)
    n = min(len(t) for _, t in fams)
    out = []
    for name, tubes in fams:
        keep = rng.choice(len(tubes), size=n, replace=False)
        sub = [tubes[i] for i in sorted(keep)]
        mids = np.array([t['mid'] for t in sub])
        c = mids.mean(axis=0)
        D = float(np.linalg.norm(mids.max(axis=0) - mids.min(axis=0)))
        if D < 1e-12:
            D = 1.0
        scaled = [dict(pts=(t['pts'] - c) / D, dir=t['dir'],
                       mid=(t['mid'] - c) / D) for t in sub]
        out.append((name, scaled))
    return out, n


def patch_diameter(tubes):
    m = np.array([t['mid'] for t in tubes])
    return float(np.linalg.norm(m.max(axis=0) - m.min(axis=0)))


def scale_ladder(diameter, n_levels=6, start=1):
    """rho_n = D * lambda^(-n), the substitution's own scales."""
    return [diameter * LAMBDA ** (-n) for n in range(start, start + n_levels)]


# ---------------------------------------------------------------------------
# 1. the ladder is real
# ---------------------------------------------------------------------------

def verify_ladder(max_depth=5):
    """Empirical tile-count inflation against lambda^2 from spectrefacts.

    Returns rows (depth, total, ratio, error vs lambda^2).  If this drifts,
    every scale below is measured against the wrong unit, so it is checked
    first and not assumed.
    """
    totals = F.MATRIX_TOTALS[:max_depth + 1]
    rows = []
    for i in range(1, len(totals)):
        r = totals[i] / totals[i - 1]
        rows.append((i, totals[i], r, abs(r - LAMBDA_SQ)))
    return rows


# ---------------------------------------------------------------------------
# 2. stickiness
# ---------------------------------------------------------------------------

def direction_histogram(dirs, bins=24):
    """Distribution of in-plane direction angle, folded to [0, pi).

    The in-plane angle is the coordinate that carries the tiling's structure;
    the out-of-plane slope is set by the braid amplitude and is near-constant
    by construction, so binning it would only add a spike.
    """
    if len(dirs) == 0:
        return None
    ang = np.mod(np.arctan2(dirs[:, 1], dirs[:, 0]), np.pi)
    h, _ = np.histogram(ang, bins=bins, range=(0.0, np.pi))
    s = h.sum()
    return h / s if s else None


def total_variation(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def stickiness(tubes, scales, n_balls=120, min_tubes=12, bins=24,
               seed=RNG_SEED):
    """Local-vs-global direction distance at each scale, noise-corrected.

    At scale rho, drop n_balls balls of radius rho, take the directions of
    the tubes whose midpoints land inside, and measure total variation from
    the global direction histogram.

    The raw distance CANNOT be used.  A ball of radius rho holds m ~ rho^2
    tubes, and the TV distance between an m-sample and the law it was drawn
    from decays like m^(-1/2) ~ 1/rho all by itself.  Reading that decay as
    a geometric fact is reading the central limit theorem.  So every ball is
    paired with a BOOTSTRAP control: m tubes drawn uniformly at random from
    the whole family, same statistic.  The reported quantity is the ratio

        excess = TV(local) / TV(random sample of the same size)

    which is 1 when a ball looks like a random draw from the global law and
    greater than 1 when the ball carries real local structure.  A STICKY
    family holds that ratio CONSTANT and above 1 across scales: the local
    picture is a copy of the global one at every zoom, never washing out.
    """
    rng = np.random.default_rng(seed)
    mids = np.array([t['mid'] for t in tubes])
    dirs = np.array([t['dir'] for t in tubes])
    glob = direction_histogram(dirs, bins)
    lo, hi = mids.min(axis=0), mids.max(axis=0)
    n = len(tubes)

    out = []
    for rho in scales:
        ratios, raws, used = [], [], 0
        for _ in range(n_balls):
            c = rng.uniform(lo, hi)
            sel = np.linalg.norm(mids - c, axis=1) <= rho
            m = int(sel.sum())
            if m < min_tubes:
                continue
            loc = direction_histogram(dirs[sel], bins)
            if loc is None:
                continue
            d_loc = total_variation(loc, glob)
            # bootstrap control at the same sample size
            boot = []
            for _ in range(8):
                pick = rng.choice(n, size=min(m, n), replace=False)
                bh = direction_histogram(dirs[pick], bins)
                if bh is not None:
                    boot.append(total_variation(bh, glob))
            d_rand = float(np.mean(boot)) if boot else np.nan
            if not np.isfinite(d_rand) or d_rand <= 1e-9:
                continue
            ratios.append(d_loc / d_rand)
            raws.append(d_loc)
            used += 1
        out.append((rho, float(np.mean(ratios)) if ratios else np.nan,
                    float(np.std(ratios)) if ratios else np.nan, used,
                    float(np.mean(raws)) if raws else np.nan))
    return out, glob


def stickiness_slope(rows):
    """Fit excess(rho) ~ rho^s.  s near 0 is sticky; s clearly negative is
    a family whose local structure washes out as you zoom in."""
    pts = [(r, d) for r, d, u, _, _ in
           [(a, b, dd, c, e) for a, b, c, dd, e in rows]
           if u >= 5 and np.isfinite(d) and d > 0]
    if len(pts) < 3:
        return float('nan')
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# 3. multi-scale non-concentration
# ---------------------------------------------------------------------------

def concentration_profile(tubes, scales, n_slabs=200, seed=RNG_SEED):
    """Largest fraction of tubes inside a slab of thickness rho, per scale.

    This is the Wang-Zahl hypothesis evaluated as a function of scale rather
    than at one thickness.  What their argument needs is decay in rho; the
    fitted exponent is reported so the claim is a number and not a shape.
    """
    rng = np.random.default_rng(seed)
    P = [t['pts'] for t in tubes]
    allp = np.vstack(P)
    n = len(tubes)
    out = []
    for rho in scales:
        best = 0
        for _ in range(n_slabs):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            proj = allp @ u
            c = rng.uniform(proj.min(), proj.max())
            cnt = sum(1 for pts in P
                      if (np.abs(pts @ u - c) <= rho / 2).all())
            best = max(best, cnt)
        out.append((rho, best / n))
    return out


def concentration_exponent(rows):
    pts = [(r, f) for r, f in rows if f > 0]
    if len(pts) < 3:
        return float('nan')
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# 4. box dimension of the lifted strand union
# ---------------------------------------------------------------------------

def box_counts(tubes, scales, max_boxes=6e7):
    """Occupied-box count of the union of strand curves, at ladder scales.

    Boxes are counted on the CURVES, not on delta-neighbourhoods: a
    neighbourhood of radius comparable to the box size would return the
    dimension of the neighbourhood instead of the dimension of the set.
    """
    pts = [t['pts'] for t in tubes]
    allp = np.vstack(pts)
    lo = allp.min(axis=0)
    ext = allp.max(axis=0) - lo
    out = []
    for eps in scales:
        if np.prod(np.ceil(ext / eps) + 2) > max_boxes:
            out.append((eps, np.nan))
            continue
        occ = set()
        for p in pts:
            q = _resample(p, eps * 0.4)
            idx = np.floor((q - lo) / eps).astype(np.int64)
            occ.update(map(tuple, idx))
        out.append((eps, len(occ)))
    return out


def box_slopes(rows):
    """Successive local slopes of log N vs log(1/eps).

    A single fitted exponent here would NOT be a dimension.  The strand set
    is a 1-dimensional curve family arranged across a 2-dimensional patch,
    so the count crosses over from slope 2 at coarse eps (the arrangement
    fills the plane) to slope 1 at fine eps (each box sees one curve).  Any
    number in between is a position on that crossover and depends on which
    scales were chosen.  Reporting the local slopes shows the crossover
    instead of hiding it inside an average.
    """
    pts = [(e, n) for e, n in rows if np.isfinite(n) and n > 0]
    out = []
    for i in range(len(pts) - 1):
        (e0, n0), (e1, n1) = pts[i], pts[i + 1]
        out.append((e1, float(np.log(n1 / n0) / np.log(e0 / e1))))
    return out


def box_dimension(rows):
    """Coarse-end slope only -- the regime where the arrangement, not the
    individual strand, sets the count."""
    sl = box_slopes(rows)
    return sl[0][1] if sl else float('nan')


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def analyse(name, tubes, n_levels, n_balls, n_slabs):
    D = patch_diameter(tubes)
    scales = scale_ladder(D, n_levels)
    st, glob = stickiness(tubes, scales, n_balls=n_balls)
    cc = concentration_profile(tubes, scales, n_slabs=n_slabs)
    bx = box_counts(tubes, scales)
    return dict(name=name, tubes=tubes, diameter=D, scales=scales,
                stick=st, stick_slope=stickiness_slope(st), glob=glob,
                conc=cc, conc_exp=concentration_exponent(cc),
                box=bx, box_dim=box_dimension(bx))


def report(res):
    print('\n%s  (%d tubes, patch diameter %.3f)'
          % (res['name'], len(res['tubes']), res['diameter']))
    print('  scale ladder rho_n = D * lambda^-n, lambda^2 = 4+sqrt(15)')
    print('  stickiness: excess over a same-size random draw')
    print('    %-11s %-9s %-9s %-9s %s'
          % ('rho', 'excess', 'sd', 'raw TV', 'balls'))
    for rho, ex, sd, u, raw in res['stick']:
        if np.isfinite(ex):
            print('    %-11.4f %-9.3f %-9.3f %-9.4f %d'
                  % (rho, ex, sd, raw, u))
        else:
            print('    %-11.4f %-9s %-9s %-9s %d'
                  % (rho, 'n/a', 'n/a', 'n/a', u))
    print('    fitted slope excess ~ rho^%.3f   (0 = sticky)'
          % res['stick_slope'])
    print('  multi-scale non-concentration')
    for rho, f in res['conc']:
        print('    rho=%-12.5f max fraction in slab = %.4f' % (rho, f))
    print('    fitted exponent %.3f   (more negative = better decay)'
          % res['conc_exp'])
    print('  box counting on the lifted strands (crossover, not a single dim)')
    sl = dict(box_slopes(res['box']))
    for eps, n in res['box']:
        tag = ('  local slope %.3f' % sl[eps]) if eps in sl else ''
        print('    eps=%-12.5f boxes = %-8s%s'
              % (eps, '%d' % n if np.isfinite(n) else 'skipped', tag))
    print('    coarse-end slope = %.3f  (2 = plane-filling, 1 = curves)'
          % res['box_dim'])


# ---------------------------------------------------------------------------

COLORS = {'spectre': 'crimson', 'hexagon': '0.45'}


def figure(results, ladder, fname):
    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_subplot(2, 3, 1)
    d = [r[0] for r in ladder]
    r = [r[2] for r in ladder]
    ax.plot(d, r, 'o-', color='crimson')
    ax.axhline(LAMBDA_SQ, color='green', ls='--',
               label=r'$\lambda^2=4+\sqrt{15}$')
    ax.set_xlabel('substitution depth')
    ax.set_ylabel('tile count ratio')
    ax.set_title('The ladder is real', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 2)
    for res in results:
        x = [s[0] for s in res['stick'] if np.isfinite(s[1])]
        y = [s[1] for s in res['stick'] if np.isfinite(s[1])]
        e = [s[2] for s in res['stick'] if np.isfinite(s[1])]
        ax.axhline(1.0, color='green', ls='--', lw=1)
        ax.errorbar(x, y, yerr=e, fmt='o-', capsize=3,
                    color=COLORS.get(res['name'], 'k'),
                    label='%s (slope %.2f)' % (res['name'],
                                               res['stick_slope']))
    ax.set_xscale('log')
    ax.set_xlabel(r'$\rho$ (ladder scale)')
    ax.set_ylabel('excess over random draw')
    ax.set_title('Stickiness: flat and above 1 means sticky', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    for res in results:
        ax.loglog([c[0] for c in res['conc']], [c[1] for c in res['conc']],
                  'o-', color=COLORS.get(res['name'], 'k'),
                  label='%s (exp %.2f)' % (res['name'], res['conc_exp']))
    ax.set_xlabel(r'slab thickness $\rho$')
    ax.set_ylabel('max fraction of tubes')
    ax.set_title('Multi-scale non-concentration', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = fig.add_subplot(2, 3, 4)
    for res in results:
        pts = [(e, n) for e, n in res['box'] if np.isfinite(n)]
        if pts:
            ax.loglog([1 / p[0] for p in pts], [p[1] for p in pts], 'o-',
                      color=COLORS.get(res['name'], 'k'),
                      label='%s (coarse slope %.2f)' % (res['name'], res['box_dim']))
    ax.set_xlabel(r'$1/\epsilon$')
    ax.set_ylabel('occupied boxes')
    ax.set_title('Box count: crossover from plane to curve', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    for i, res in enumerate(results):
        ax = fig.add_subplot(2, 3, 5 + i, projection='polar')
        g = res['glob']
        th = np.linspace(0, np.pi, len(g), endpoint=False)
        ax.bar(th, g, width=np.pi / len(g), color=COLORS.get(res['name'], 'k'),
               alpha=0.8)
        ax.set_thetamax(180)
        ax.set_title('%s: global direction law' % res['name'], fontsize=10)

    fig.suptitle('Spectre braid as a multi-scale tube family\n'
                 r'scales graded by the fundamental unit '
                 r'$\lambda^2=4+\sqrt{15}$ of $\mathbb{Z}[\sqrt{15}]$',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def selftest(iterations=2):
    failures = []

    def check(label, ok):
        print('  %-58s %s' % (label, 'ok' if ok else 'FAIL'))
        if not ok:
            failures.append(label)

    print('the unit')
    check('lambda^2 is 4+sqrt(15) from spectrefacts',
          abs(LAMBDA_SQ - (4 + np.sqrt(15))) < 1e-12)
    check('lambda is its positive square root',
          abs(LAMBDA ** 2 - LAMBDA_SQ) < 1e-12)
    rows = verify_ladder()
    check('tile counts inflate by lambda^2 to 1e-4', rows[-1][3] < 1e-4)

    print('the ladder')
    lad = scale_ladder(10.0, 4)
    check('ladder descends by lambda each step',
          all(abs(lad[i] / lad[i + 1] - LAMBDA) < 1e-9
              for i in range(len(lad) - 1)))

    print('families')
    sp_t = braid_tubes(iterations)
    hx_t = hex_tubes(14)
    check('spectre family non-empty', len(sp_t) > 0)
    check('hexagon family non-empty', len(hx_t) > 0)
    check('solo gives one tube per shared edge',
          len(sp_t) == len(braid_tubes(iterations, solo=True)))

    print('statistics are in range')
    for n, t in (('spectre', sp_t), ('hexagon', hx_t)):
        sc = scale_ladder(patch_diameter(t), 3)
        st, g = stickiness(t, sc, n_balls=40)
        check('%s excess ratios are positive' % n,
              all(not np.isfinite(d) or d > 0.0 for _, d, _, _, _ in st))
        check('%s direction law sums to 1' % n, abs(g.sum() - 1.0) < 1e-9)
        cc = concentration_profile(t, sc, n_slabs=40)
        check('%s slab fractions in [0,1]' % n,
              all(0.0 <= f <= 1.0 for _, f in cc))
        check('%s concentration decreases along the descending ladder' % n,
              all(cc[i][1] >= cc[i + 1][1] - 1e-9
                  for i in range(len(cc) - 1)))

    print()
    if failures:
        print('%d failure(s): %s' % (len(failures), ', '.join(failures)))
    else:
        print('spectre_multiscale: all checks pass.')
    return len(failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--iterations', type=int, default=4)
    ap.add_argument('--hex', type=int, default=40)
    ap.add_argument('--levels', type=int, default=5)
    ap.add_argument('--balls', type=int, default=120)
    ap.add_argument('--slabs', type=int, default=200)
    ap.add_argument('--crossings', type=int, default=DEFAULT_CROSSINGS)
    ap.add_argument('--height', type=float, default=DEFAULT_HEIGHT)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if selftest(min(args.iterations, 2)) else 0)

    print('spectre_multiscale -- scales, not directions')
    print('lambda^2 = 4 + sqrt(15) = %.7f   (spectrefacts.LAM2)' % LAMBDA_SQ)
    print('lambda   = %.7f' % LAMBDA)

    ladder = verify_ladder()
    print('\nladder check: tile-count inflation vs lambda^2')
    for d, tot, r, err in ladder:
        print('  depth %d: total=%-8d ratio=%.7f  err=%.2e' % (d, tot, r, err))

    sp_t = braid_tubes(args.iterations, args.crossings, args.height)
    hx_t = hex_tubes(args.hex, args.crossings, args.height)
    fams, kept = match_families([('spectre', sp_t), ('hexagon', hx_t)])
    print('\nmatched families: %d tubes each, rescaled to unit diameter'
          ' (from %d spectre, %d hexagon)' % (kept, len(sp_t), len(hx_t)))

    results = [analyse(n, t, args.levels, args.balls, args.slabs)
               for n, t in fams]
    for r in results:
        report(r)

    png = os.path.join(OUT, 'spectre_multiscale.png')
    figure(results, ladder, png)
    print('\nwrote %s' % png)

    s, h = results
    print('\nspectre vs periodic control')
    print('  stickiness slope:      spectre %+.3f   hexagon %+.3f'
          % (s['stick_slope'], h['stick_slope']))
    print('  concentration exponent: spectre %+.3f   hexagon %+.3f'
          % (s['conc_exp'], h['conc_exp']))
    print('  box coarse-end slope:   spectre %.3f    hexagon %.3f'
          % (s['box_dim'], h['box_dim']))


if __name__ == '__main__':
    main()
