#!/usr/bin/python3
"""Supertile outlines alternate in handedness with the level: the one place the parity flip is visible.

Computation, not a Fact.  parity_clock.py finds that orientation correlations between TILES do not alternate with the
hierarchy level.  The construction does mirror at every level (hierarchy_cache.py: det = (-1)^depth), and this module
asks whether the outline of a level-k supertile is a mirror image of the outline of the same-type level-(k+1)
supertile, as the flip suggests, or a rotated copy.

Method.  For a supertile (the set of tile centroids with a common level-k address prefix) take the signature
s_i = max normalised distance from the centroid of the points, over 36 angular bins (10 degrees; the construction only
rotates by multiples of 30 degrees, so a rotation is an exact cyclic shift).  The distance between two supertiles is
the RMS difference after the best cyclic shift ("proper", rotation only) or after also reversing the bin order
("improper", rotation + reflection).  The RAW construction coordinates are used, NOT the normalised ones: mirroring is
exactly what is being tested.  At most 6 supertiles per (level, type) are used.

Results (depth 6; the --check run prints the table):
 * Control: same level, same type: proper distance is 0 (to rounding, below 1e-13), improper distance 0.29 to 0.47.
   So the signature does see handedness.
 * Same type one level apart (k -> k+1): the mirror fits better than the rotation in every comparison (24 of 24):
   proper vs improper 0.514 vs 0.392 (2 -> 3), 0.371 vs 0.164 (3 -> 4), 0.303 vs 0.098 (4 -> 5).  The 2 -> 3 step is
   the weakest, and for Gamma it is within the 20% margin (0.550 vs 0.485).
 * Two levels apart (k -> k+2): the rotation fits better in every comparison (15 of 15), but clearly only for
   3 -> 5 (proper 0.190 vs improper 0.373); for 2 -> 4 the difference is small (0.474 vs 0.557) and the verdict
   column says "no clear preference".
 * The distances are the same for almost every supertile type at a given level: the signature is determined by the
   level, not by the type (Gamma, a two-tile metatile, is the exception, slightly different).
 * So the alternation is real in the supertile outlines, and it is the parity clock, but it lives in the outlines
   and not in the tiles (parity_clock.py).  It is a statement about the substitution as written, i.e. about its
   construction.  It is not an independent dynamical feature of the tiling: the mirror image of a Spectre tiling
   is another Spectre tiling.

    python3 outline_chirality.py [DEPTH]     default depth 6
    python3 outline_chirality.py --check     depth 6
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hierarchy_cache as HC

NB = 36


def signature(P):
    c = P.mean(0)
    v = P - c
    r = np.hypot(v[:, 0], v[:, 1])
    r = r / np.sqrt((r ** 2).mean())
    b = np.floor(((np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi)) / (2 * np.pi)) * NB).astype(int) % NB
    sig = np.zeros(NB)
    for i in range(NB):
        m = b == i
        sig[i] = r[m].max() if m.any() else 0.0
    return sig


def best(a, b, mirror):
    bb = b[::-1] if mirror else b
    return min(np.linalg.norm(a - np.roll(bb, s)) for s in range(NB)) / math.sqrt(NB)


def instances(P, per=6):
    """{(level k, supertile type): [signature, ...]} for k = 2 .. D-1, at most ``per`` each."""
    D = P['D']
    inst = {}
    for k in range(2, D):
        inv = HC.groups(P, k)
        order = np.argsort(inv, kind='stable')
        starts = np.searchsorted(inv[order], np.arange(inv.max() + 1))
        ends = np.append(starts[1:], len(inv))
        for g in range(inv.max() + 1):
            idx = order[starts[g]:ends[g]]
            L = inst.setdefault((k, P['labs'][idx[0], D - k]), [])
            if len(L) < per:
                L.append(signature(P['pos'][idx]))
    return inst


def compare(inst, dk, k, t, n=3):
    A, B = inst.get((k, t)), inst.get((k + dk, t))
    if not A or not B:
        return None
    return (np.mean([best(a, b, False) for a in A[:n] for b in B[:n]]),
            np.mean([best(a, b, True) for a in A[:n] for b in B[:n]]))


def report(depth):
    P = HC.load(depth)
    inst = instances(P)
    types = sorted({t for (_, t) in inst})
    print('outline signature = max normalised radius in %d angular bins; distance = RMS difference' % NB)
    print('same-level same-type control (proper should be ~0):')
    ctrl = []
    for (k, t), L in sorted(inst.items()):
        if len(L) >= 2:
            p = np.mean([best(L[0], x, False) for x in L[1:]])
            m = np.mean([best(L[0], x, True) for x in L[1:]])
            ctrl.append((k, t, p, m))
            print('   level %d %-7s n=%d  proper %.3f  improper %.3f' % (k, t, len(L), p, m))
    print('\nsame supertile type, k -> k+dk (RMS signature distance):')
    print('%-7s %-9s %9s %9s   verdict' % ('type', 'levels', 'proper', 'improper'))
    rows = []
    for t in types:
        for dk in (1, 2):
            for k in range(2, depth - dk):
                c = compare(inst, dk, k, t)
                if c is None:
                    continue
                p, m = c
                rows.append((t, k, dk, p, m))
                print('%-7s %d -> %d    %9.3f %9.3f   %s' % (t, k, k + dk, p, m,
                      'mirror fits better' if m < 0.8 * p else ('rotation fits better' if p < 0.8 * m else 'no clear preference')))
    return ctrl, rows


def check():
    ok = True

    def tick(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('  %-66s %s %s' % (name, 'ok' if cond else 'FAIL', detail))

    print('outline_chirality --check')
    ctrl, rows = report(6)
    print()
    tick('same-level same-type supertiles are rotations of each other (proper < 1e-9)', all(p < 1e-9 for _, _, p, _ in ctrl),
         '(max %.2g)' % max(p for _, _, p, _ in ctrl))
    tick('...and the signature sees handedness (improper > 0.2 for every type and level)', all(m > 0.2 for _, _, _, m in ctrl),
         '(min %.3f)' % min(m for _, _, _, m in ctrl))
    strong = {(t, k): (p, m) for (t, k, dk, p, m) in rows if dk == 2 and k == 3}
    tick('3 -> 5: rotation beats mirror by the 20% margin for every type', bool(strong) and all(p < 0.8 * m for p, m in strong.values()),
         '(%d types)' % len(strong))
    one = [(p, m) for (_, _, dk, p, m) in rows if dk == 1]
    two = [(p, m) for (_, _, dk, p, m) in rows if dk == 2]
    f1 = np.mean([m < p for p, m in one])
    f2 = np.mean([p < m for p, m in two])
    tick('k -> k+1: the mirror fits better than the rotation in most comparisons (> 70%)', f1 > 0.7, '(%.0f%% of %d)' % (100 * f1, len(one)))
    tick('k -> k+2: the rotation fits better than the mirror in most comparisons (> 70%)', f2 > 0.7, '(%.0f%% of %d)' % (100 * f2, len(two)))
    return ok


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(0 if check() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    report(int(args[0]) if args else 6)
