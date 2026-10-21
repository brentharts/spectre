#!/usr/bin/env python3
"""lift4d.py - the 4D lift of a Spectre patch (Cheritat-Ma) and its projections.

Every edge of the Spectre tiling is a unit vector at a multiple of 30 degrees.  Even multiples are hexagonal-lattice
vectors p + q w (w = e^{i pi/3}); odd multiples are z times one (z = e^{i pi/6}).  Walking along edges, a vertex gets
integer coordinates (p, q, r, s) = (u, v), u the 'red' part (even directions), v the 'green' part (odd directions):

    physical position  P_{a,b}(p,q,r,s) = a (p + q w) + b z (r + s w)        (a = b = 1: the tiling itself)

so the lift is the ring Z[zeta_12] = Z^4, and (a, b) is the Cheritat-Ma deformation.  The internal space is the contracting
plane of the inflation: the two linear functionals of (p,q,r,s) whose range stays bounded as the patch grows.
(The Galois conjugate z -> z^5 is NOT bounded: it grows by the inflation factor too.)

CLI: python3 lift4d.py [--tests-only]
"""
import argparse
import math
import sys
from collections import deque

import numpy as np

import spectre_patch as sp

Z = np.exp(1j * math.pi / 6)
W = Z * Z
HEX = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]       # e^{i pi m/3} = p + q w


def step(k):
    """Integer 4D increment of a unit edge at angle k*30 degrees."""
    k %= 12
    p, q = HEX[(k // 2) % 6]
    return (p, q, 0, 0) if k % 2 == 0 else (0, 0, p, q)


def lift_patch(levels):
    """-> (tiles, lifted): tiles = list of 14-long lists of 4D integer vertices, lifted = dict key->4-vector,
    and the number of vertex-consistency checks made (all must agree)."""
    P = sp.SpectrePatch(levels)
    polys = [P.straight_polygon(i) for i in range(P.n_tiles)]
    key = lambda q: (round(q[0], 4), round(q[1], 4))
    adj, ids = {}, {}
    for poly in polys:
        n = len(poly)
        for j in range(n):
            a, b = key(poly[j]), key(poly[(j + 1) % n])
            d = poly[(j + 1) % n] - poly[j]
            ang = math.degrees(math.atan2(d[1], d[0])) / 30.0
            k = int(round(ang))
            assert abs(ang - k) < 1e-6 and abs(np.hypot(*d) - 1) < 1e-6, "edge not a unit vector at 30 degrees"
            s = step(k)
            adj.setdefault(a, []).append((b, s))
            adj.setdefault(b, []).append((a, tuple(-c for c in s)))
    start = next(iter(adj))
    lifted = {start: (0, 0, 0, 0)}
    todo = deque([start])
    checks = 0
    while todo:
        u = todo.popleft()
        for v, s in adj[u]:
            cand = tuple(x + y for x, y in zip(lifted[u], s))
            if v in lifted:
                assert lifted[v] == cand, "inconsistent lift"
                checks += 1
            else:
                lifted[v] = cand
                todo.append(v)
    tiles = [[lifted[key(q)] for q in poly] for poly in polys]
    return tiles, lifted, checks


def project(v, a=1.0, b=1.0):
    v = np.asarray(v, dtype=float)
    return a * (v[..., 0] + v[..., 1] * W) + b * Z * (v[..., 2] + v[..., 3] * W)


def internal(v, j=5):
    """Galois conjugate  z -> z^j  of the lifted vertex."""
    v = np.asarray(v, dtype=float)
    zj = Z ** j
    return v[..., 0] + v[..., 1] * zj ** 2 + v[..., 2] * zj + v[..., 3] * zj ** 3


def tile_area(poly2d):
    x, y = poly2d.real, poly2d.imag
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def lattice_rank(vectors):
    return int(np.linalg.matrix_rank(np.asarray(vectors, dtype=float)))


def bounded_plane(levels=4):
    """The two linear functionals of the lifted coordinates with bounded range: the small singular directions of
    the lifted vertex cloud (the contracting plane of the inflation).  Returns (S (2,4), singular values/sqrt(n))."""
    _, lifted, _ = lift_patch(levels)
    V = np.array(list(lifted.values()), dtype=float)
    _, s, vt = np.linalg.svd(V - V.mean(0), full_matrices=False)
    return vt[2:], s / np.sqrt(len(V))


def window_stats(levels, S=None):
    """Extents of the physical image and of the internal image (through the bounded plane S)."""
    if S is None:
        S = bounded_plane(4)[0]
    _, lifted, _ = lift_patch(levels)
    V = np.array(list(lifted.values()), dtype=float)
    return dict(n=len(V), phys=float(np.abs(project(V)).max()), internal=float(np.abs(V @ S.T).max()))


def window_boundary_counts(levels=5, hs=(0.2, 0.1, 0.05, 0.025)):
    """Occupied and boundary cells of the internal image at several cell sizes."""
    S = bounded_plane(4)[0]
    _, lifted, _ = lift_patch(levels)
    X = np.array(list(lifted.values()), dtype=float) @ S.T
    out = []
    for h in hs:
        st = set(map(tuple, np.unique(np.floor(X / h).astype(int), axis=0)))
        bd = sum(1 for (i, j) in st if any((i + di, j + dj) not in st for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        out.append((h, len(st), bd))
    return out


def show(save=None):
    import matplotlib
    matplotlib.use("Agg" if save else matplotlib.get_backend())
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    tiles, lifted, _ = lift_patch(4)
    V = np.array(list(lifted.values()), dtype=float)
    q = V @ bounded_plane(4)[0].T
    ax[0].plot(q[:, 0], q[:, 1], ",", color="k")
    ax[0].set_title("internal-space image (window), 4 levels")
    ax[0].set_aspect("equal")
    for a, b, c in ((1, 1, "k"), (1, 0.5, "C0")):
        for t in lift_patch(2)[0][:40]:
            p = project(np.array(t), a, b)
            ax[1].plot(np.r_[p.real, p.real[0]], np.r_[p.imag, p.imag[0]], c, lw=.6)
    ax[1].set_aspect("equal")
    ax[1].set_title("projections (a,b) = (1,1) black, (1,1/2) blue")
    xs = [window_stats(L) for L in (1, 2, 3, 4)]
    ax[2].semilogy([s["phys"] for s in xs], "o-", label="physical extent")
    ax[2].semilogy([s["internal"] for s in xs], "s-", label="internal extent (bounded plane)")
    ax[2].legend()
    ax[2].set_xlabel("levels - 1")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
    else:
        plt.show()


# ------------------------------------------------------------------ tests
def test_lift_is_consistent_integer_and_reproduces_the_patch():
    tiles, lifted, checks = lift_patch(2)
    assert checks > 1000
    P = sp.SpectrePatch(2)
    for i, t in enumerate(tiles):
        assert np.allclose(project(np.array(t)) - project(np.array(t))[0],
                           (lambda q: q[:, 0] + 1j * q[:, 1])(P.straight_polygon(i)) - complex(*P.straight_polygon(i)[0]), atol=1e-9)


def test_lift_spans_four_dimensions_and_projection_is_linear():
    _, lifted, _ = lift_patch(2)
    V = np.array(list(lifted.values()), dtype=float)
    assert lattice_rank(V) == 4
    u, v = V[3], V[11]
    assert abs(project(u + v, 1.3, 0.7) - project(u, 1.3, 0.7) - project(v, 1.3, 0.7)) < 1e-12


def test_edge_deformation_keeps_every_shared_vertex_shared():
    tiles, _, _ = lift_patch(2)
    seen = {}
    for t in tiles:
        for v, p in zip(t, project(np.array(t), 1.4, 0.6)):
            assert abs(seen.setdefault(v, p) - p) < 1e-12


def test_area_scales_as_a_homogeneous_quadratic_form_in_a_b():
    t = np.array(lift_patch(1)[0][0])
    A = lambda a, b: tile_area(project(t, a, b))
    assert abs(A(2, 2) - 4 * A(1, 1)) < 1e-9 and abs(A(3, 1.5) - 2.25 * A(2, 1)) < 1e-9 and A(1, 1) > 0


def test_two_directions_grow_by_the_inflation_factor_and_two_stay_bounded():
    S, sv = bounded_plane(4)
    assert sv[1] > 20 * sv[2] and sv[3] > 0.5
    ex = [window_stats(L, S)["internal"] for L in (2, 3, 4)]
    ph = [window_stats(L, S)["phys"] for L in (2, 3, 4)]
    assert ex[2] < 1.3 * ex[1] and abs(ph[2] / ph[1] - 2.8059) < 0.1


def test_galois_conjugate_is_not_the_internal_space():
    a, b = window_stats(3, None)["phys"], None
    _, lifted, _ = lift_patch(3)
    V = np.array(list(lifted.values()), dtype=float)
    assert np.abs(internal(V, 5)).max() > 40


def test_window_boundary_grows_faster_than_a_curve_of_length_one():
    c = window_boundary_counts(5, (0.2, 0.1, 0.05))
    ratio = (c[2][2] / c[0][2]) ** 0.5            # boundary cells per halving (sparse sampling limits finer cells)
    assert ratio > 2.1 and c[2][1] / c[0][1] < 15.5   # area ~ x4 per halving, boundary clearly more than x2


def run_tests(verbose=True):
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            status = "PASS"
        except Exception as exc:  # noqa: BLE001
            failed += 1
            status = "FAIL: %s: %s" % (type(exc).__name__, exc)
        if verbose:
            print("  %-74s %s" % (name, status))
    if verbose:
        print("%d test(s), %d failed" % (len(tests), failed))
    return failed == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--png")
    a = ap.parse_args(argv)
    if a.tests_only:
        return 0 if run_tests() else 1
    S, sv = bounded_plane(4)
    print('singular values/sqrt(n):', np.round(sv, 3))
    for L in (1, 2, 3, 4):
        print(L, window_stats(L, S))
    print('window boundary counts', window_boundary_counts())
    if a.png:
        show(a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
