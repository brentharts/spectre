#!/usr/bin/python3
"""How large a neighbourhood must a tile see to know its ancestors?  The record capacity of a finite observer.

Computation, not a Fact (a leave-one-out estimate on a finite patch, on a coarse radius grid).

Idea (Barbour's "records"): a finite observer cannot know a level of the hierarchy unless its patch is big enough to
recognise it.  Recognizability of a primitive aperiodic substitution says an infinite patch determines every ancestor;
this measures the finite radius that suffices, level by level.

Method.  For a random interior tile, take its radius-r neighbourhood in the tile's own frame (relative positions,
labels, relative orientations, as a hashable key).  A_k is the label of the level-k supertile containing the tile;
F_k is the whole address inside it.  The leave-one-out error of predicting A_k from the OTHER tiles with the same
neighbourhood measures how well r determines it; coverage is the fraction of tiles whose neighbourhood occurs
elsewhere (the plug-in conditional entropy would drop to 0 trivially once every neighbourhood is unique).

Depth-6 patch, 60000 tiles, r in units of the tile scale ell = 2.863 edge lengths (lambda = 2.8059):

    level k   radius for 10% error   radius where the error reaches 0
       1          about 2.4 ell          3 ell
       2          about 7.7 ell         12 ell
       3          about 23 ell          32-48 ell (error 1.1% at 32; 0 at 48 with only 68% coverage)

So each further level costs a radius about 3 times larger, the same order as lambda (the grid is coarse, nothing
sharper is claimed).  An observer of radius R can name its ancestors up to k_max ~ log_lambda(R / c ell), c between
1 and 1.5.  The error for the whole address F_k equals that for the label A_k to within noise: the label is the record.
The number of distinct neighbourhoods grows like r^1.9 to r^2.0 (888, 3510, 12902 at r = 6, 12, 24 ell), i.e. about
2 log2(lambda) = 2.98 bits per level = ln(4 + sqrt15) nats, the log-period of the substrate; that equals area growth
and is not special to the Spectre beyond lambda.

The asymmetry this exposes: going to FINER scales is free (the substitution fixes the subdivision), going to COARSER
scales costs a patch about 3 times larger per level.  It is not by itself an arrow of time; it becomes one only if the
observer's patch grows in one direction, which the tiling does not supply.

    python3 records_recognizability.py [DEPTH] [CENTRES]    default depth 6, 60000 centres (about 5 min)
    python3 records_recognizability.py --check              depth 5, 8000 centres, levels 1 and 2
"""
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hierarchy_cache as HC

RADII = (1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)      # in tile scales


def centres(P, M, rmax_ell, seed=5):
    rng = np.random.default_rng(seed)
    ctr = P['pos'].mean(0)
    rad = np.linalg.norm(P['pos'] - ctr, axis=1)
    cand = np.where(rad < rad.max() - (rmax_ell + 2) * P['ell'])[0]
    return rng.choice(cand, size=min(M, len(cand)), replace=False)


def env_keys(P, pos, ang, cs, r_ell):
    """Hash of the canonical radius-r neighbourhood of each centre, in the frame of the centre tile."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    li = {l: i for i, l in enumerate(sorted(set(P['leaf'].tolist())))}
    lid = np.array([li[l] for l in P['leaf']])
    ai = np.round(ang / (math.pi / 6)).astype(int) % 12
    keys = []
    for c in cs:
        idx = np.array(tree.query_ball_point(pos[c], r_ell * P['ell']))
        ct, st = math.cos(-ang[c]), math.sin(-ang[c])
        d = pos[idx] - pos[c]
        rows = np.column_stack([np.round((ct * d[:, 0] - st * d[:, 1]) * 200).astype(np.int64),
                                np.round((st * d[:, 0] + ct * d[:, 1]) * 200).astype(np.int64),
                                lid[idx], (ai[idx] - ai[c]) % 12])
        keys.append(hash(rows[np.lexsort(rows.T[::-1])].tobytes()))
    return keys


def loo(keys, ys):
    """Leave-one-out error among covered samples, coverage, number of distinct neighbourhoods."""
    g = collections.defaultdict(collections.Counter)
    for k, y in zip(keys, ys):
        g[k][y] += 1
    wrong = seen = 0
    for c in g.values():
        if sum(c.values()) < 2:
            continue
        for y, n in c.items():
            others = {l: m - (l == y) for l, m in c.items()}
            seen += n
            wrong += n * (max(others, key=others.get) != y)
    return (wrong / seen if seen else float('nan')), seen / len(keys), len(g)


def run(depth, M, radii=RADII, levels=(1, 2, 3)):
    P = HC.load(depth)
    pos, ang = HC.normalised(P)
    cs = centres(P, M, max(radii))
    print('depth %d, %d tiles, %d centres, ell = %.3f edge lengths, lambda = %.4f' % (depth, len(P['pos']), len(cs), P['ell'], HC.LAMBDA))
    print('%6s %7s | ' % ('r/ell', 'n_env') + ' '.join('err(A%d) err(F%d) cov(A%d)' % (k, k, k) for k in levels))
    out = {}
    for r in radii:
        keys = env_keys(P, pos, ang, cs, r)
        row = []
        for k in levels:
            Ak = [P['labs'][c, depth - k] for c in cs]
            Fk = [(P['labs'][c, depth - k],) + tuple(P['slots'][c, depth - k:]) for c in cs]
            eA, cvA, ne = loo(keys, Ak)
            eF, _, _ = loo(keys, Fk)
            row.append((eA, eF, cvA))
        out[r] = (ne, row)
        print('%6.1f %7d | ' % (r, ne) + '   '.join('%6.3f %6.3f %6.2f' % x for x in row), flush=True)
    return out


def check():
    ok = True

    def tick(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('  %-66s %s %s' % (name, 'ok' if cond else 'FAIL', detail))

    print('records_recognizability --check')
    out = run(5, 8000, radii=(1.5, 2, 3, 4, 8), levels=(1, 2))
    tick('level 1: error is 0 at 3 ell', out[3][1][0][0] < 0.01, '(%.3f)' % out[3][1][0][0])
    tick('level 1: not determined at 2 ell', out[2][1][0][0] > 0.1, '(%.3f)' % out[2][1][0][0])
    tick('level 2: not determined at 4 ell', out[4][1][1][0] > 0.3, '(%.3f)' % out[4][1][1][0])
    tick('level 2: nearly determined at 8 ell', out[8][1][1][0] < 0.3, '(%.3f)' % out[8][1][1][0])
    tick('level 2 needs a larger radius than level 1 (error at 3 ell)', out[3][1][1][0] > out[3][1][0][0] + 0.3)
    return ok


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(0 if check() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    run(int(args[0]) if args else 6, int(args[1]) if len(args) > 1 else 60000)
