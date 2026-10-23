#!/usr/bin/python3
"""Parity clock: does the handedness flip at every level leave an alternating signature in the tile orientations?

Computation, not a Fact.  The substitution mirrors the patch at every level (hierarchy_cache.py: det = (-1)^depth), so
a level-k supertile and a level-(k+1) supertile are mirror images of each other in the construction.  If that were a
physical clock, orientation statistics that are ODD under reflection should alternate in sign with the hierarchy level.

Statistics (all reflection-odd, swap-symmetric and rotation-invariant; theta = tile orientation, phi = direction from
the observer tile o to the other tile t):
    P11 = sin(th_o + th_t - 2 phi)
    D1  = sin(th_o - phi) - sin(th_t - phi)
    S2  = sin 2(th_o - phi) + sin 2(th_t - phi)
    D3  = sin 3(th_o - phi) - sin 3(th_t - phi)
and the swirl <sin(th_i - phi_i)>, <sin 2(th_i - phi_i)> of the tiles of one level-k supertile about its centre.
Tiles are first reflected to a common handedness (hierarchy_cache.normalised); each statistic is verified to flip
sign under a mirror image of the whole configuration (the oddness check), so a nonzero mean is a genuine chirality.

Result (the --check printout, depth 5 and 6, 800 observers): the TILE statistics do not alternate.  Mean P11 for near
pairs (1 to 3 ell) is +0.25 when the common ancestor is the level-1 supertile (h = 1) and then negative, of one
sign, for h = 2, 3, 4 (depth 6: -0.12, -0.14, -0.15; depth 5: -0.10, -0.20, -0.14).  It is positive at h = 1 and
negative beyond, a single sign change, not an alternation.  For mid-range pairs (3 to 9 ell) P11 is +0.12 at h = 2 and
about -0.04 for h = 3, 4, 5.  The swirl <sin(th - phi)> about a level-k supertile centre is negative at every level:
-0.427, -0.208, -0.150, -0.055, -0.030 for k = 1..5, shrinking by a factor 1.4 to 2.7 per level and never changing
sign.  So the handedness flip of the construction is NOT visible in the orientation correlations of tiles.
(outline_chirality.py shows it IS visible in the outline shapes of supertiles.)  Depth-5 and depth-6 patches agree
on P11 for h = 1..3 to within 0.06 after normalisation, which is the check that this is not the construction's
global mirror.  swirl2 is -0.878 at k = 1 and about zero beyond.

    python3 parity_clock.py [DEPTH] [OBSERVERS]     default depth 6, 2500 observers
    python3 parity_clock.py --check                 depths 5 and 6, 800 observers
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hierarchy_cache as HC

STATS = ("P11", "D1", "S2", "D3")
BINS = {"near [1,3) ell": (1.0, 3.0), "mid [3,9) ell": (3.0, 9.0)}


def stats(th_o, th_t, phi):
    return np.array([np.sin(th_o + th_t - 2 * phi),
                     np.sin(th_o - phi) - np.sin(th_t - phi),
                     np.sin(2 * (th_o - phi)) + np.sin(2 * (th_t - phi)),
                     np.sin(3 * (th_o - phi)) - np.sin(3 * (th_t - phi))])


def pair_means(P, nobs, margin=40.0, boot=200, seed=3):
    """{bin: {h: (n_pairs, mean (4,), se (4,))}} plus the oddness-check flag."""
    pos, ang = HC.normalised(P)
    D = P['D']
    acc = {b: np.zeros((nobs, D + 1, len(STATS))) for b in BINS}
    cnt = {b: np.zeros((nobs, D + 1)) for b in BINS}
    odd = True
    n = 0
    for o, idx, h, d, phi in HC.iter_pairs(P, pos, nobs, 9.0, seed=seed, margin_ell=margin):
        st = stats(ang[o], ang[idx], phi)
        if n < 50:
            odd &= bool(np.allclose(stats(-ang[o], -ang[idx], -phi), -st, atol=1e-9))
        for b, (a0, a1) in BINS.items():
            s = (d >= a0) & (d < a1)
            for k in range(1, D + 1):
                m = s & (h == k)
                if m.any():
                    cnt[b][n, k] = m.sum()
                    acc[b][n, k] = st[:, m].sum(1)
        n += 1
    rng = np.random.default_rng(seed)
    out = {}
    for b in BINS:
        out[b] = {}
        for k in range(1, D):                       # the top level is censored by the patch edge
            tot = cnt[b][:n, k].sum()
            if tot < 1000:
                continue
            mean = acc[b][:n, k].sum(0) / tot
            reps = []
            for _ in range(boot):
                r = rng.integers(0, n, n)
                t = cnt[b][r, k].sum()
                reps.append(acc[b][r, k].sum(0) / t if t > 0 else np.full(len(STATS), np.nan))
            out[b][k] = (int(tot), mean, np.nanstd(reps, axis=0))
    return out, odd


def swirl(P):
    """{k: (n_supertiles, swirl1, se1, swirl2, se2)}: orientation about each level-k supertile's centre."""
    pos, ang = HC.normalised(P)
    D = P['D']
    out = {}
    for k in range(1, D):
        inv = HC.groups(P, k)
        G = inv.max() + 1
        c = np.bincount(inv, minlength=G).astype(float)
        cx, cy = np.bincount(inv, pos[:, 0], G) / c, np.bincount(inv, pos[:, 1], G) / c
        r = ang - np.arctan2(pos[:, 1] - cy[inv], pos[:, 0] - cx[inv])
        s1, s2 = np.bincount(inv, np.sin(r), G) / c, np.bincount(inv, np.sin(2 * r), G) / c
        out[k] = (G, s1.mean(), s1.std() / math.sqrt(G), s2.mean(), s2.std() / math.sqrt(G))
    return out


def report(depth, nobs, margin=40.0):
    P = HC.load(depth)
    res, odd = pair_means(P, nobs, margin)
    print('depth %d: %d tiles, tiles normalised to one handedness; mirror flips every statistic: %s' % (depth, len(P['pos']), odd))
    for b in BINS:
        print('\n--- pairs, %s: mean of each reflection-odd statistic by common-ancestor level h (x1000 +/- bootstrap SE) ---' % b)
        print('%3s %9s | ' % ('h', 'pairs') + ' '.join('%15s' % s for s in STATS))
        for k, (n, m, e) in sorted(res[b].items()):
            print('%3d %9d | ' % (k, n) + ' '.join('%+7.1f+/-%5.1f' % (1000 * a, 1000 * s) for a, s in zip(m, e)))
    sw = swirl(P)
    print('\n--- swirl about level-k supertile centres (x1000 +/- SE over supertiles) ---')
    print('%3s %9s | %16s %16s' % ('k', 'n_super', 'swirl1', 'swirl2'))
    for k, (G, a, ea, b, eb) in sw.items():
        print('%3d %9d | %+8.1f+/-%5.1f %+8.1f+/-%5.1f' % (k, G, 1000 * a, 1000 * ea, 1000 * b, 1000 * eb))
    return res, sw, odd


def check():
    ok = True

    def tick(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('  %-66s %s %s' % (name, 'ok' if cond else 'FAIL', detail))

    print('parity_clock --check')
    near = 'near [1,3) ell'
    R = {}
    for depth in (5, 6):
        R[depth] = report(depth, 800, margin=20.0)
        print()
    (r5, s5, o5), (r6, s6, o6) = R[5], R[6]
    tick('every statistic is odd under a mirror image of the configuration', o5 and o6)
    tick('depth 6: near-pair P11 is positive at h=1 and negative at h=2,3',
         r6[near][1][1][0] > 0 and r6[near][2][1][0] < 0 and r6[near][3][1][0] < 0,
         str([round(float(1000 * r6[near][k][1][0]), 1) for k in (1, 2, 3)]))
    tick('depth 6: no sign alternation of P11 over h = 2,3,4 (all one sign)',
         len({np.sign(r6[near][k][1][0]) for k in (2, 3, 4) if k in r6[near]}) == 1)
    d = max(abs(r5[near][k][1][0] - r6[near][k][1][0]) for k in (1, 2, 3) if k in r5[near] and k in r6[near])
    tick('depths 5 and 6 agree on P11 (h=1,2,3) to within 0.08', d < 0.08, '(max diff %.3f)' % d)
    tick('depth 6: swirl1 is negative for k = 1..4', all(s6[k][1] < 0 for k in (1, 2, 3, 4)),
         str([round(float(1000 * s6[k][1]), 1) for k in (1, 2, 3, 4)]))
    tick('depth 6: |swirl1| decreases with k for k = 1..4', all(abs(s6[k][1]) > abs(s6[k + 1][1]) for k in (1, 2, 3)))
    return ok


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(0 if check() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    report(int(args[0]) if args else 6, int(args[1]) if len(args) > 1 else 2500)
