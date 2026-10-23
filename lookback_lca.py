#!/usr/bin/python3
"""Hierarchy distance against Euclidean distance: does depth behave like look-back time?

Computation, not a Fact.  Hypothesis: if time is the level index, a source at distance d is seen at look-back
h ~ log_lambda(d / ell), where h(O, T) is the level of the lowest common ancestor of tiles O and T (1 = they share a
level-1 supertile).  What this tests is the geometry of that dictionary.  It does NOT supply a light cone: nothing here
says what propagates.

Results.  The numbers below are what `--check` prints (depth 6, 600 observers, about 500,000 tile pairs; d in tile
scales ell).  A 64 x 64 quadtree is run through the same code as a control (lambda = 2, boundary dimension 1).  Larger
runs (depth 7, more observers) were done with the prototypes and agreed to within a few percent; they are not
reproduced by --check.

 * Mean law.  <h> = 0.92 log_lambda d + 1.72 for the Spectre (d from 1 to 7.5 ell); the quadtree gives 0.90.  The slope
   is below 1 because h is capped at the top level D.  So the dictionary holds on average, and it is generic: any
   self-similar hierarchy does this.
 * Scatter.  The standard deviation of h at fixed d is about 1.1 levels (quadtree 1.0).  The distribution has a
   geometric tail: for pairs at 1 to 2 ell, P(h >= k+1) / P(h >= k) is 0.549, 0.483, 0.518 for k = 1, 2, 3, then 0.431
   and 0.383 where the cap at the top level bites.  The outline dimension 1.3993 would give nu / lambda^2 = 0.538 and
   the quadtree gives 0.5; the quadtree control reads 0.622, 0.546, 0.525, 0.504, 0.350 with the same code.  The data
   do NOT separate the Spectre from the quadtree.
 * What the tail means.  Of tiles only 1 to 2 ell apart, about 6% share no ancestor until level 5 or higher and about
   2% until level 6: neighbours in space can be several levels apart in the hierarchy.  This is not an artefact of the
   Spectre (the control has it).  If depth is look-back, the ancestor relation, not Euclidean distance, would have to be
   the causal structure.
 * Log-periodic fine structure of <h> - log_lambda d: fundamental amplitude 0.022 levels for the Spectre against 0.064
   for the quadtree.  Essentially absent.
 * In-plane isotropy was examined in the prototypes (about 10% variation of <h> between direction bins mod 30 degrees,
   different at depth 6 and 7), which is not resolved from sampling noise.  Not part of this module.

    python3 lookback_lca.py [DEPTH] [OBSERVERS]    default depth 7, 2500 observers (pair table plus the control)
    python3 lookback_lca.py --check                depth 6, 600 observers
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hierarchy_cache as HC

PRED = HC.NU / HC.LAMBDA ** 2          # 0.5381: tail decay per level if the outline dimension is 1.3993


def collect(P, nobs, dmax=17.0, margin=40.0):
    pos, _ = HC.normalised(P)
    ds, hs, ph = [], [], []
    for _, _, h, d, phi in HC.iter_pairs(P, pos, nobs, dmax, margin_ell=margin):
        ds.append(d), hs.append(h), ph.append(np.degrees(phi) % 30.0)
    return np.concatenate(ds), np.concatenate(hs), np.concatenate(ph)


def tail_ratios(d, h, D, lo, hi):
    s = (d >= lo) & (d < hi)
    surv = [(h[s] >= k).mean() for k in range(1, D + 1)]
    return surv, [surv[k] / surv[k - 1] if surv[k - 1] > 0 else float('nan') for k in range(1, D)]


def mean_law_slope(d, h, lam, edges=(1.0, 1.5, 2.2, 3.3, 5.0, 7.5)):
    x = np.log(d) / math.log(lam)
    xs, ms = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (d >= a) & (d < b)
        xs.append(x[s].mean()), ms.append(h[s].mean())
    return np.polyfit(xs, ms, 1)


def report(name, P, nobs, lam, pred, margin=40.0, dmax=17.0):
    d, h, ph = collect(P, nobs, dmax, margin)
    D = P['D']
    sl, ic = mean_law_slope(d, h, lam)
    print('\n=== %s: %d pairs; predicted tail decay per level %.4f' % (name, len(d), pred))
    print('mean law <h> = %.3f log_lam(d/ell) + %.3f' % (sl, ic))
    x = np.log(d) / math.log(lam)
    s = (d >= 1.2) & (d < 9)
    res, phase = h[s] - x[s], x[s] % 1.0
    A = np.column_stack([np.ones(s.sum()), np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)])
    c = np.linalg.lstsq(A, res, rcond=None)[0]
    print('scatter sd of h - log_lam d: %.2f levels; log-periodic fundamental amplitude %.3f levels' % (res.std(), math.hypot(c[1], c[2])))
    for lo, hi in ((1.0, 2.0), (2.0, 4.0), (4.0, 6.0)):
        surv, rat = tail_ratios(d, h, D, lo, hi)
        print('d in [%.0f,%.0f) ell  P(h>=k): %s   ratios: %s' % (lo, hi, ' '.join('%.4f' % p for p in surv), ' '.join('%.3f' % q for q in rat)))
    return d, h, sl, tail_ratios(d, h, D, 1.0, 2.0)[1]


def check():
    ok = True

    def tick(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('  %-66s %s %s' % (name, 'ok' if cond else 'FAIL', detail))

    print('lookback_lca --check')
    _, _, sl, rat = report('SPECTRE depth 6', HC.load(6), 600, HC.LAMBDA, PRED)
    tick('mean law has slope close to 1 (0.85 to 1.05)', 0.85 < sl < 1.05, '(%.3f)' % sl)
    tick('tail ratios k=2->3 and 3->4 lie in 0.45 to 0.60', all(0.45 < r < 0.60 for r in rat[1:3]), str(np.round(rat[1:3], 3)))
    _, _, slq, ratq = report('QUADTREE control', HC.quadtree(6), 600, 2.0, 0.5, margin=12.0, dmax=10.0)
    tick('quadtree control: tail ratios k=3..4 near 1/2 (0.42 to 0.58)', all(0.42 < r < 0.58 for r in ratq[2:4]), str(np.round(ratq[2:4], 3)))
    tick('quadtree control: slope below 1 for the same cap reason (0.8 to 1.05)', 0.8 < slq < 1.05, '(%.3f)' % slq)
    return ok


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(0 if check() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    depth, nobs = (int(args[0]) if args else 7), (int(args[1]) if len(args) > 1 else 2500)
    report('SPECTRE depth %d' % depth, HC.load(depth), nobs, HC.LAMBDA, PRED)
    report('QUADTREE control', HC.quadtree(6), nobs, 2.0, 0.5, margin=12.0, dmax=10.0)
