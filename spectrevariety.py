#!/usr/bin/env python3
r"""spectrevariety.py -- Barbour's variety, evaluated on substitution patches.

The variety of N point masses (Barbour and Lourenco, arXiv:2608.05929, Eqs.
1-3; masses normalised to sum to one) is

    V = l_rms / l_mhl,   l_rms^2 = sum_{i<j} m_i m_j r_ij^2,
                         1/l_mhl = sum_{i<j} m_i m_j / r_ij,

a dimensionless number, unchanged by translation, rotation, reflection and
overall scale.  Here every mass is 1/N and the points are the tile centroids
of the depth-n Spectre patch.

Two controls with the same N and the same region (the union of the tiles, so
that the outer shape of the non-convex patch is not what is being measured):
a hexagonal lattice whose spacing is tuned to give N points in the region, and
N uniform random points in the region (mean and spread over several draws).

These are floating-point computations.  Unlike the Facts of spectrefacts.py
they are not exact, and the paper says so.  The analytic controls (two equal
masses, V = 1/8; an equilateral triangle, V = sqrt(3)/9) are exact.

    python3 spectrevariety.py              print the table
    python3 spectrevariety.py --selftest   analytic controls and invariances
"""

import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_family as TF
from matplotlib.path import Path
from scipy.spatial import cKDTree

MAX_DEPTH = 4
N_RANDOM = 8
SEED = 7


def variety(P, chunk=1000):
    """V = l_rms / l_mhl for equal masses at the rows of P (N x 2)."""
    P = np.asarray(P, dtype=float)
    N = len(P)
    m = 1.0 / N
    s2 = 0.0
    sinv = 0.0
    cols = np.arange(N)
    for i in range(0, N, chunk):
        A = P[i:i + chunk]
        D = np.linalg.norm(A[:, None, :] - P[None, :, :], axis=2)
        mask = cols[None, :] > np.arange(i, i + len(A))[:, None]
        Dm = D[mask]
        s2 += np.sum(Dm ** 2)
        sinv += np.sum(1.0 / Dm)
    return float(np.sqrt(m * m * s2) * (m * m * sinv))


@lru_cache(maxsize=None)
def _patch(n):
    placed = TF.placed_tiles(n)
    base = TF.canonical_tile_verts('Gamma')[:-1]
    polys = [TF.transform_polygon(T, base) for T, _ in placed]
    cents = np.array([p.mean(axis=0) for p in polys])
    return polys, cents


def _in_region(polys, cents, Q, k=8):
    tree = cKDTree(cents)
    _, nbr = tree.query(Q, k=k)
    hit = np.zeros(len(Q), dtype=bool)
    groups = {}
    for qi, row in enumerate(nbr):
        for t in row:
            groups.setdefault(int(t), []).append(qi)
    for t, qs in groups.items():
        qs = np.array(qs)
        hit[qs] |= Path(polys[t]).contains_points(Q[qs])
    return hit


def _sample_region(polys, cents, N, rng):
    lo, hi = cents.min(0) - 2.0, cents.max(0) + 2.0
    out = []
    while sum(len(o) for o in out) < N:
        Q = rng.uniform(lo, hi, size=(6 * N, 2))
        out.append(Q[_in_region(polys, cents, Q)])
    return np.vstack(out)[:N]


def _lattice(polys, cents, N):
    lo, hi = cents.min(0) - 2.0, cents.max(0) + 2.0
    x, y = polys[0][:, 0], polys[0][:, 1]
    area = N * 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    a = np.sqrt(2 * area / (np.sqrt(3) * N))
    pts = np.zeros((0, 2))
    for _ in range(60):
        rows = []
        xs = np.arange(lo[0], hi[0], a)
        for j, y in enumerate(np.arange(lo[1], hi[1], a * np.sqrt(3) / 2)):
            off = a / 2 if j % 2 else 0.0
            rows.append(np.column_stack([xs + off, np.full(len(xs), y)]))
        pts = np.vstack(rows)
        pts = pts[_in_region(polys, cents, pts)]
        if len(pts) == N:
            break
        a *= (len(pts) / N) ** 0.5
    return pts


@lru_cache(maxsize=None)
def table(max_depth=MAX_DEPTH, n_random=N_RANDOM, seed=SEED):
    """Rows: (depth, N, V_spectre, N_lattice, V_lattice, V_random_mean,
    V_random_sd).  Deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(1, max_depth + 1):
        polys, cents = _patch(n)
        N = len(cents)
        L = _lattice(polys, cents, N)
        vr = [variety(_sample_region(polys, cents, N, rng))
              for _ in range(n_random)]
        rows.append((n, N, variety(cents), len(L), variety(L),
                     float(np.mean(vr)), float(np.std(vr))))
    return rows


def selftest():
    fails = []

    def check(name, ok):
        print('  %-62s %s' % (name, 'ok' if ok else 'FAIL'))
        if not ok:
            fails.append(name)

    check('two equal masses: V = 1/8',
          abs(variety([[0, 0], [3, 1]]) - 1 / 8) < 1e-12)
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    check('equilateral triangle: V = sqrt(3)/9',
          abs(variety(tri) - np.sqrt(3) / 9) < 1e-12)
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(40, 2))
    th = 0.7
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    v0 = variety(Q)
    check('invariant under scale', abs(variety(5.5 * Q) - v0) < 1e-12)
    check('invariant under rotation and translation',
          abs(variety(Q @ Rm.T + [3, -2]) - v0) < 1e-12)
    check('invariant under reflection',
          abs(variety(Q * [1, -1]) - v0) < 1e-12)
    # control: a quantity that is NOT scale invariant must fail the same test
    def bad(P):
        P = np.asarray(P, float)
        d = np.linalg.norm(P[:, None] - P[None], axis=2)
        return d[np.triu_indices(len(P), 1)].sum()
    check('control: a non-invariant quantity changes under scale',
          abs(bad(5.5 * Q) - bad(Q)) > 1e-3)
    check('chunked sum equals unchunked sum',
          abs(variety(Q, chunk=7) - variety(Q, chunk=1000)) < 1e-12)
    rows = table()
    check('the table has one row per depth, N = 9, 71, 559, 4401',
          [r[1] for r in rows] == [9, 71, 559, 4401])
    check('the lattice control is within 1% of the patch count in N',
          all(abs(r[3] - r[1]) <= max(2, 0.01 * r[1]) for r in rows))
    print()
    print('FAILED: %s' % ', '.join(fails) if fails else 'all variety checks pass')
    return len(fails)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(1 if selftest() else 0)
    print('depth      N   V_spectre  N_lat   V_lattice   V_random(mean+-sd)')
    for n, N, vs, nl, vl, vr, sd in table():
        print('%5d %6d %11.6f %6d %11.6f %11.6f +- %.6f' % (n, N, vs, nl, vl, vr, sd))
