#!/usr/bin/python3
"""Barbour's shape complexity of the Spectre tile centroids: is there a clustering arrow along the hierarchy?

Computation, not a Fact.  Barbour, Koslowski and Mercati (arXiv:1409.0917) define, for a set of equal-mass points,
        C_S = l_rms / l_mhl,   l_rms^2 = (1/N^2) sum_{i<j} r_ij^2,   1/l_mhl = (1/N^2) sum_{i<j} 1/r_ij,
a scale-invariant measure of clustering.  In their N-body model it has a minimum at a central turning point and grows
both ways; internal observers call the turning point their past.  Here the tile centroids of the level-n patch play
the points, and the question is whether C_S grows with n.  (Their 1/r potential is a 3D choice; in 2D a logarithm
would be more natural.  A uniform patch is uniform under either, so the conclusion should not depend on it.)

What the numbers say.  C_S rises with n (9 tiles 0.360, 71 -> 0.539, 559 -> 0.604, 4401 -> 0.626, 34649 -> 0.634) but
that rise is a finite-size and outline effect that every control shares.  Against points spread uniformly over the SAME
outline (a triangular lattice cut to the patch footprint) the Spectre excess is below 0.0015 from level 2 on and
about 0.0002 at levels 4 and 5, and it sits slightly BELOW a Poisson sample of the same footprint (0.634 against
0.640 at level 5).  The approach to the limit is geometric, steps shrinking by about 1/lambda = 0.36 per level.
So positional clustering carries no arrow: the tiling is as uniform as the footprint allows.

The uniform-disc limit has a closed form, C_S -> 8 / (3 pi sqrt 2) = 0.6002 (mean inverse distance of two points in a
disc is 16 / (3 pi R), and l_rms = R / sqrt 2), which the Poisson control reproduces; --check asserts it.

    python3 barbour_complexity.py            levels 1..5 against the controls (about 1 min)
    python3 barbour_complexity.py --check    analytic disc value and the footprint control at levels 3 and 4
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DISC_LIMIT = 8 / (3 * math.pi * math.sqrt(2))


def shape_complexity(P, chunk=1500):
    """C_S = l_rms / l_mhl of an (N, 2) point set, exact O(N^2) in chunks."""
    P = np.asarray(P, float)
    N = len(P)
    P = P - P.mean(0)
    s2 = sinv = 0.0
    for a in range(0, N, chunk):
        A = P[a:a + chunk]
        for b in range(a, N, chunk):
            d2 = ((A[:, None, :] - P[b:b + chunk][None, :, :]) ** 2).sum(-1)
            if a == b:
                d2 = d2[np.triu_indices(len(A), 1)]
            d2 = d2.ravel()
            s2 += d2.sum()
            sinv += (1 / np.sqrt(d2)).sum()
    return math.sqrt(s2 / N ** 2) * (sinv / N ** 2)


def patch_polygons(depth):
    """World outlines of the tiles of the depth-n patch, from the walker in spectre_selfsimilar."""
    import spectre_selfsimilar as SS
    return [t['verts'] for t in SS.build_patch(depth)]


def centroids(polys):
    out = []
    for v in polys:
        x, y = v[:, 0], v[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        out.append([((x + x1) * cr).sum() / (3 * cr.sum()), ((y + y1) * cr).sum() / (3 * cr.sum())])
    return np.array(out)


def footprint_controls(polys, N, rng):
    """(C_S of a triangular lattice, of a Poisson sample) restricted to the patch footprint, with ~N points."""
    import shapely
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    foot = unary_union([Polygon(v).buffer(1e-9) for v in polys])
    minx, miny, maxx, maxy = foot.bounds
    a = math.sqrt(2 / (math.sqrt(3) * N / foot.area))
    k = int(max(maxx - minx, maxy - miny) / a) + 3
    g = np.array([(i * a + 0.5 * j * a + minx, j * a * math.sqrt(3) / 2 + miny)
                  for i in range(-k, 2 * k) for j in range(-k, 2 * k)])
    g = g[shapely.contains_xy(foot, g[:, 0], g[:, 1])]
    pois = []
    while len(pois) < N:
        q = np.column_stack([rng.uniform(minx, maxx, 4 * N), rng.uniform(miny, maxy, 4 * N)])
        pois.extend(q[shapely.contains_xy(foot, q[:, 0], q[:, 1])].tolist())
    return shape_complexity(g), shape_complexity(np.array(pois[:N]))


def table(levels):
    rng = np.random.default_rng(1)
    print('uniform disc, N -> infinity: %.4f' % DISC_LIMIT)
    print('%-5s %7s %9s %12s %12s %9s' % ('level', 'N', 'Spectre', 'lattice/same', 'Poisson/same', 'excess'))
    rows = []
    for L in levels:
        polys = patch_polygons(L)
        c = shape_complexity(centroids(polys))
        cl, cp = footprint_controls(polys, len(polys), rng)
        rows.append((L, len(polys), c, cl, cp))
        print('%-5d %7d %9.5f %12.5f %12.5f %+9.5f' % (L, len(polys), c, cl, cp, c - cl), flush=True)
    return rows


def check():
    ok = True

    def tick(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('  %-66s %s %s' % (name, 'ok' if cond else 'FAIL', detail))

    print('barbour_complexity --check')
    rng = np.random.default_rng(0)
    r, t = np.sqrt(rng.random(4000)), 2 * np.pi * rng.random(4000)
    c = shape_complexity(np.column_stack([r * np.cos(t), r * np.sin(t)]))
    tick('Poisson disc reproduces the analytic 8/(3 pi sqrt2) = 0.6002', abs(c - DISC_LIMIT) < 0.01, '(%.4f)' % c)
    for L in (3, 4):
        polys = patch_polygons(L)
        c = shape_complexity(centroids(polys))
        cl, cp = footprint_controls(polys, len(polys), rng)
        tick('level %d: Spectre equals a uniform lattice on the same footprint (<0.005)' % L, abs(c - cl) < 0.005,
             '(%.4f vs %.4f)' % (c, cl))
        tick('level %d: Spectre is not more clustered than Poisson on the footprint' % L, c < cp + 0.002,
             '(%.4f vs %.4f)' % (c, cp))
    return ok


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(0 if check() else 1)
    table(range(1, 6))
