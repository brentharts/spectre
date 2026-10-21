"""
spectre_operator.py - the curved Spectre monotile with per-edge curve control.

spectre.py gives every edge the same curve strength.  Here each of the 14 edges
has its own curve amount ``curves[i]`` and a single ``scale`` multiplies all of
them, so the effective strength of edge i is ``scale * curves[i]``.

    from spectre_operator import curved_spectre
    x, y = curved_spectre(a=1, b=1, curves=1.0, scale=0.5)     # same as spectre.py, k=0.5
    x, y = curved_spectre(curves=[1, -1] * 7, scale=0.4)       # alternating edges
    x, y = curved_spectre(curves={3: 1.0, 7: -1.0}, scale=0.6) # only edges 3 and 7

``curves`` may be a number (all edges), a sequence of 14 numbers, or a dict
{edge_index: amount} (missing edges are straight).

Edge i runs from vertex P_i to P_{i+1}; its kind ('a' or 'b' length) is in
EDGE_KIND.  Run ``python3 spectre_operator.py`` to see all indices and a set of
variations (it first runs its own tests; --help for options).

Geometry (see symspectre.py for the exact derivation): each edge is a cubic
Bezier  B(t) = P_i + s(t) d_i + k w(t) n_i  with n_i = (d_y, -d_x), an S-curve
that is point-symmetric about the midpoint of its chord.  Because of that the
enclosed (signed) area is 2*sqrt(3)*a^2 + 3*a*b + sqrt(3)*b^2 for ANY choice of
per-edge amounts, and the vertices never move.

Corner angles are the polygon's only where the two edges meeting at a vertex
have equal amounts.  In general the turn at vertex P_i is
    (polygon turn) + arctan(2 k_i) - arctan(2 k_(i-1)),   k = scale * curves,
so neighbouring edges with very different amounts make sharp corners, and
very large amounts can make the outline cross itself; see ``is_simple``.

Only NumPy is needed (matplotlib is imported lazily for the demo).
"""

import argparse
import math
import os
import sys

import numpy as np

__all__ = [
    "N_EDGES",
    "EDGE_KIND",
    "A_EDGES",
    "B_EDGES",
    "edge_vectors",
    "vertices",
    "edge_strengths",
    "edge_beziers",
    "evaluate_edges",
    "curved_spectre_edges",
    "curved_spectre",
    "polygon_area",
    "spectre_area",
    "perimeter",
    "is_simple",
    "run_tests",
    "show_variations",
]

N_EDGES = 14
# Length type of each edge (edge i runs P_i -> P_{i+1}); same order as spectre.py.
EDGE_KIND = "bbaabbaaaabbaa"
A_EDGES = tuple(i for i, c in enumerate(EDGE_KIND) if c == "a")
B_EDGES = tuple(i for i, c in enumerate(EDGE_KIND) if c == "b")


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def edge_vectors(a=1.0, b=1.0):
    """The 14 edge vectors of the tile as an array of shape (14, 2)."""
    c, s = 0.5, math.sqrt(3.0) / 2.0  # cos / sin of 60 degrees
    return np.array(
        [
            (c * b, s * b),
            (b, 0.0),
            (0.0, a),
            (s * a, c * a),
            (c * b, -s * b),
            (-c * b, -s * b),
            (s * a, -c * a),
            (0.0, -a),
            (0.0, -a),
            (-s * a, -c * a),
            (-c * b, s * b),
            (-b, 0.0),
            (0.0, a),
            (-s * a, c * a),
        ],
        dtype=float,
    )


def vertices(a=1.0, b=1.0):
    """The 14 vertices P_0..P_13 of the straight tile, shape (14, 2); P_0 = (0, 0)."""
    d = edge_vectors(a, b)
    return np.vstack([[0.0, 0.0], np.cumsum(d, axis=0)[:-1]])


def edge_strengths(curves=1.0, scale=1.0):
    """Effective curve strength ``scale * curves[i]`` of every edge, shape (14,).

    ``curves`` is a number, a sequence of 14 numbers, or a dict {edge: amount}
    (edges not listed are straight).
    """
    if isinstance(curves, dict):
        arr = np.zeros(N_EDGES)
        for key, val in curves.items():
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
                raise ValueError("curve dict keys must be edge indices, got %r" % (key,))
            if not 0 <= key < N_EDGES:
                raise ValueError("edge index %d out of range 0..%d" % (key, N_EDGES - 1))
            arr[key] = val
    else:
        arr = np.asarray(curves, dtype=float)
        if arr.ndim == 0:
            arr = np.full(N_EDGES, float(arr))
        elif arr.shape != (N_EDGES,):
            raise ValueError(
                "curves must be a number, a dict or %d values, got shape %s"
                % (N_EDGES, arr.shape)
            )
    k = float(scale) * arr
    if not np.all(np.isfinite(k)):
        raise ValueError("curve amounts and scale must be finite")
    return k


def edge_beziers(a=1.0, b=1.0, curves=1.0, scale=1.0):
    """Bezier control points of all edges, shape (14, 4, 2): [P_i, C1, C2, P_{i+1}].

    C1 = P_i + d/2 - k n,  C2 = P_i + d/2 + k n,  n = (d_y, -d_x)  (as in spectre.py).
    """
    d = edge_vectors(a, b)
    p0 = vertices(a, b)
    p3 = np.roll(p0, -1, axis=0)  # exact closure: edge 13 ends at P_0
    k = edge_strengths(curves, scale)[:, None]
    n = np.stack([d[:, 1], -d[:, 0]], axis=1)
    mid = p0 + d / 2.0
    return np.stack([p0, mid - k * n, mid + k * n, p3], axis=1)


def evaluate_edges(ctrl, t):
    """Evaluate every edge curve at parameters ``t``: shape (14, len(t), 2)."""
    t = np.asarray(t, dtype=float)
    u = 1.0 - t
    basis = np.stack([u**3, 3 * u**2 * t, 3 * u * t**2, t**3], axis=1)  # (m, 4)
    return np.einsum("mj,ejd->emd", basis, ctrl)


def _check_samples(samples):
    if int(samples) != samples or samples < 1:
        raise ValueError("samples must be a positive integer")
    return int(samples)


def curved_spectre_edges(a=1.0, b=1.0, curves=1.0, scale=1.0, samples=50):
    """Each edge sampled from its start to its end vertex (both included):
    shape (14, samples + 1, 2).  Handy for drawing edges individually."""
    samples = _check_samples(samples)
    ctrl = edge_beziers(a, b, curves, scale)
    return evaluate_edges(ctrl, np.linspace(0.0, 1.0, samples + 1))


def curved_spectre(a=1.0, b=1.0, curves=1.0, scale=1.0, samples=50, closed=True):
    """Outline of the curved tile as two arrays (x, y).

    Parameters
    ----------
    a, b : edge lengths.
    curves : number | sequence of 14 numbers | {edge: amount}; per-edge curve amounts.
    scale : single factor multiplying every curve amount.
    samples : points per edge (the vertices are always included).
    closed : repeat the first point at the end so that the outline closes.
    """
    samples = _check_samples(samples)
    ctrl = edge_beziers(a, b, curves, scale)
    pts = evaluate_edges(ctrl, np.arange(samples) / samples).reshape(-1, 2)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    return pts[:, 0].copy(), pts[:, 1].copy()


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def polygon_area(x, y):
    """Signed shoelace area (positive for counterclockwise; the tile is clockwise)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def spectre_area(a=1.0, b=1.0):
    """Exact area of the tile, independent of all curve amounts."""
    r3 = math.sqrt(3.0)
    return 2 * r3 * a * a + 3 * a * b + r3 * b * b


def perimeter(a=1.0, b=1.0, curves=1.0, scale=1.0):
    """Exact perimeter of the curved tile.

    The speed of edge i is |B'(t)| = |d_i| * sqrt(s'(t)^2 + k_i^2 w'(t)^2) (n_i is
    perpendicular to d_i and has the same length), so
        P = sum_i |d_i| * F(k_i),   F(k) = int_0^1 sqrt(s'^2 + k^2 w'^2) dt,
    with s' = 3/2 - 3t + 3t^2 and w' = -18 t^2 + 18 t - 3.  F is evaluated by
    Gauss-Legendre quadrature (the integrand is smooth, s' >= 3/4).
    """
    nodes, weights = np.polynomial.legendre.leggauss(48)
    t, wt = 0.5 * (nodes + 1.0), 0.5 * weights
    sp_ = 1.5 - 3.0 * t + 3.0 * t**2
    wp_ = -18.0 * t**2 + 18.0 * t - 3.0
    k = edge_strengths(curves, scale)
    F = np.sqrt(sp_[None, :] ** 2 + (k[:, None] * wp_[None, :]) ** 2) @ wt
    return float((np.hypot(*edge_vectors(a, b).T) * F).sum())


def is_simple(x, y):
    """True if the polygon through the points has no self-intersections
    (proper crossings of non-adjacent segments)."""
    P = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    if len(P) > 1 and np.allclose(P[0], P[-1], atol=1e-12, rtol=0):
        P = P[:-1]
    n = len(P)
    Q = np.roll(P, -1, axis=0)
    for i in range(n):
        j = np.arange(i + 2, n if i > 0 else n - 1)
        if j.size == 0:
            continue
        p, q, r, s = P[i], Q[i], P[j], Q[j]
        e = q - p
        d1 = e[0] * (r[:, 1] - p[1]) - e[1] * (r[:, 0] - p[0])
        d2 = e[0] * (s[:, 1] - p[1]) - e[1] * (s[:, 0] - p[0])
        f = s - r
        d3 = f[:, 0] * (p[1] - r[:, 1]) - f[:, 1] * (p[0] - r[:, 0])
        d4 = f[:, 0] * (q[1] - r[:, 1]) - f[:, 1] * (q[0] - r[:, 0])
        if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
            return False
    return True


# --------------------------------------------------------------------------
# Self tests  (run with:  python3 spectre_operator.py)
# --------------------------------------------------------------------------
class _Skip(Exception):
    pass


def test_edges_close_and_kinds():
    a, b = 1.7, 0.6
    d = edge_vectors(a, b)
    assert d.shape == (N_EDGES, 2) and len(EDGE_KIND) == N_EDGES
    assert np.allclose(d.sum(axis=0), 0.0, atol=1e-12), "edges must sum to zero"
    for i, kind in enumerate(EDGE_KIND):
        assert math.isclose(np.hypot(*d[i]), a if kind == "a" else b, rel_tol=1e-12)
    assert len(A_EDGES) == 8 and len(B_EDGES) == 6 and set(A_EDGES + B_EDGES) == set(range(14))


def test_outline_passes_through_vertices():
    a, b, samples = 1.3, 0.8, 25
    x, y = curved_spectre(a, b, curves=np.linspace(-1, 1, 14), scale=0.7, samples=samples)
    P = vertices(a, b)
    assert np.allclose(np.column_stack([x, y])[:-1:samples], P, atol=1e-12)
    assert x[0] == x[-1] and y[0] == y[-1], "closed outline must repeat its first point"


def test_scale_zero_is_the_straight_polygon():
    a, b = 1.0, 1.4
    ctrl = edge_beziers(a, b, curves=np.linspace(-2, 2, 14), scale=0.0)
    pts = evaluate_edges(ctrl, np.linspace(0, 1, 11))
    d, P = edge_vectors(a, b), vertices(a, b)
    for i in range(N_EDGES):
        rel = pts[i] - P[i]
        assert np.allclose(rel[:, 0] * d[i, 1] - rel[:, 1] * d[i, 0], 0.0, atol=1e-12)
        frac = rel @ d[i] / (d[i] @ d[i])
        assert np.all(frac > -1e-12) and np.all(frac < 1 + 1e-12)


def test_matches_spectre_py_for_uniform_strength():
    try:
        import spectre
    except ImportError as exc:  # numpy missing, etc.
        raise _Skip("spectre.py not importable: %s" % exc)
    a, b, k = 1.3, 0.8, 0.45
    xs, ys = spectre.generate_monotile(a, b, k)
    pts = evaluate_edges(edge_beziers(a, b, curves=1.0, scale=k), np.linspace(0, 1, 50))
    for i in range(N_EDGES):
        lo = 1 + 50 * i
        assert np.allclose(pts[i, :, 0], xs[lo : lo + 50], atol=1e-12), "x, edge %d" % i
        assert np.allclose(pts[i, :, 1], ys[lo : lo + 50], atol=1e-12), "y, edge %d" % i


def test_single_scale_multiplies_every_curve():
    rng = np.random.default_rng(1)
    c = rng.uniform(-1, 1, N_EDGES)
    for s in (0.0, 0.3, -0.8, 2.0):
        x1, y1 = curved_spectre(curves=c, scale=s)
        x2, y2 = curved_spectre(curves=s * c, scale=1.0)
        assert np.allclose(x1, x2, atol=1e-13) and np.allclose(y1, y2, atol=1e-13)
    x0, y0 = curved_spectre(curves=c, scale=0.0)
    xz, yz = curved_spectre(curves=0.0, scale=1.0)
    assert np.allclose(x0, xz) and np.allclose(y0, yz)


def test_area_is_independent_of_every_curve_amount():
    rng = np.random.default_rng(7)
    a, b = 1.4, 0.9
    exact = spectre_area(a, b)
    for _ in range(5):
        c = rng.uniform(-1, 1, N_EDGES)
        x, y = curved_spectre(a, b, curves=c, scale=rng.uniform(0.2, 1.0), samples=2000)
        assert math.isclose(abs(polygon_area(x, y)), exact, rel_tol=0, abs_tol=1e-5)
    xs, ys = curved_spectre(a, b, curves=0.0, samples=1)
    assert math.isclose(abs(polygon_area(xs, ys)), exact, abs_tol=1e-12)


def test_every_edge_is_point_symmetric():
    rng = np.random.default_rng(3)
    ctrl = edge_beziers(1.2, 0.7, curves=rng.uniform(-1, 1, N_EDGES), scale=0.9)
    t = np.linspace(0, 1, 17)
    fwd, rev = evaluate_edges(ctrl, t), evaluate_edges(ctrl, 1 - t)
    chord_sum = ctrl[:, 0, :] + ctrl[:, 3, :]  # P_i + P_{i+1} = twice the chord midpoint
    assert np.allclose(fwd + rev, chord_sum[:, None, :], atol=1e-12)


def test_corner_turn_follows_neighbouring_amounts():
    def turns(ctrl):
        out = ctrl[:, 1] - ctrl[:, 0]  # tangent leaving P_i
        inc = np.roll(ctrl[:, 3] - ctrl[:, 2], 1, axis=0)  # tangent arriving at P_i
        cross = inc[:, 0] * out[:, 1] - inc[:, 1] * out[:, 0]
        return np.arctan2(cross, (inc * out).sum(axis=1))

    k = np.random.default_rng(11).uniform(-0.8, 0.8, N_EDGES)
    phi = np.arctan(2 * k)
    expected = turns(edge_beziers(curves=k, scale=0.0)) + phi - np.roll(phi, 1)
    actual = turns(edge_beziers(curves=k, scale=1.0))
    assert np.allclose((actual - expected + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=1e-12)
    # equal amounts on every edge: the corner angles are exactly the polygon's
    assert np.allclose(turns(edge_beziers(curves=0.0)), turns(edge_beziers(curves=1.0, scale=0.7)), atol=1e-12)


def test_changing_one_edge_changes_only_that_edge():
    base = np.linspace(-1, 1, N_EDGES)
    other = base.copy()
    other[5] += 0.5
    s = 20
    f1 = evaluate_edges(edge_beziers(curves=base, scale=0.6), np.arange(s) / s)
    f2 = evaluate_edges(edge_beziers(curves=other, scale=0.6), np.arange(s) / s)
    changed = [i for i in range(N_EDGES) if not np.allclose(f1[i], f2[i])]
    assert changed == [5], "changed edges: %s" % changed


def test_curve_input_forms_and_validation():
    c = np.zeros(N_EDGES)
    c[3], c[7] = 1.0, -0.5
    a = curved_spectre(curves={3: 1.0, 7: -0.5}, scale=0.4)
    b = curved_spectre(curves=c.tolist(), scale=0.4)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    for bad in ([1.0] * 13, [[1.0] * 14], {14: 1.0}, {-1: 1.0}, {"x": 1.0}, [float("nan")] * 14):
        try:
            curved_spectre(curves=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for curves=%r" % (bad,))
    for bad_samples in (0, -3, 2.5):
        try:
            curved_spectre(samples=bad_samples)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for samples=%r" % (bad_samples,))
    try:
        curved_spectre(scale=float("inf"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for infinite scale")


def test_perimeter_closed_form_matches_polyline_length():
    rng = np.random.default_rng(5)
    cases = [(1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.5), (1.3, 0.8, 1.0, -0.7)]
    cases.append((0.9, 1.6, rng.uniform(-1, 1, N_EDGES), 0.6))
    for a, b, c, s in cases:
        x, y = curved_spectre(a, b, c, s, samples=4000)
        poly = float(np.hypot(np.diff(x), np.diff(y)).sum())
        assert math.isclose(perimeter(a, b, c, s), poly, rel_tol=1e-6), (a, b, s)
    assert math.isclose(perimeter(1.0, 1.0, 1.0, 0.0), 14.0, rel_tol=1e-14)  # 8a + 6b
    assert perimeter(1, 1, 1.0, 0.5) > perimeter(1, 1, 1.0, 0.25) > perimeter(1, 1, 1.0, 0.0)


def test_is_simple_detects_self_intersection():
    assert is_simple([0, 1, 1, 0], [0, 0, 1, 1])  # square
    assert not is_simple([0, 1, 0, 1], [0, 1, 1, 0])  # bow-tie
    assert is_simple(*curved_spectre(curves=0.0, samples=1))  # the straight tile
    assert is_simple(*curved_spectre(curves=1.0, scale=0.5, samples=30))  # spectre.py default
    assert is_simple(*curved_spectre(curves=-1.0, scale=0.5, samples=30))  # mirrored curves
    assert not is_simple(*curved_spectre(curves=1.0, scale=3.0, samples=30))  # far too curvy


def run_tests(verbose=True):
    """Run every ``test_*`` function in this module; return True if none failed."""
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            status = "PASS"
        except _Skip as exc:
            status = "SKIP (%s)" % exc
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failed += 1
            status = "FAIL: %s: %s" % (type(exc).__name__, exc)
        if verbose:
            print("  %-58s %s" % (name, status))
    if verbose:
        print("%d test(s), %d failed" % (len(tests), failed))
    return failed == 0


# --------------------------------------------------------------------------
# Variations (matplotlib)
# --------------------------------------------------------------------------
def _variations(seed):
    rng = np.random.default_rng(seed)
    only_a = {i: 1.0 for i in A_EDGES}
    only_b = {i: 1.0 for i in B_EDGES}
    return [
        ("polygon: scale = 0", dict(curves=1.0, scale=0.0)),
        ("uniform, scale 0.5 (spectre.py)", dict(curves=1.0, scale=0.5)),
        ("uniform, scale -0.5 (mirrored)", dict(curves=1.0, scale=-0.5)),
        ("uniform, scale 1.0", dict(curves=1.0, scale=1.0)),
        ("only the 8 'a' edges, scale 0.5", dict(curves=only_a, scale=0.5)),
        ("only the 6 'b' edges, scale 0.5", dict(curves=only_b, scale=0.5)),
        ("alternating +1 / -1, scale 0.5", dict(curves=[(-1.0) ** i for i in range(N_EDGES)], scale=0.5)),
        ("random in [-1, 1], scale 0.6", dict(curves=rng.uniform(-1, 1, N_EDGES), scale=0.6)),
        ("only edge 7, scale 0.8", dict(curves={7: 1.0}, scale=0.8)),
        ("ramp -1 .. +1 over edges, scale 0.6", dict(curves=np.linspace(-1, 1, N_EDGES), scale=0.6)),
    ]


def show_variations(a=1.0, b=1.0, seed=0, save=None, show=True):
    """Draw a figure with the edge numbering and a set of per-edge variations."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("coolwarm")
    P = vertices(a, b)
    pad = 0.75
    xlim = (P[:, 0].min() - pad, P[:, 0].max() + pad)
    ylim = (P[:, 1].min() - pad, P[:, 1].max() + pad)

    def frame(ax, title, color="black"):
        ax.set_title(title, fontsize=9, color=color)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axis("off")

    def draw_tile(ax, title, curves, scale):
        k = edge_strengths(curves, scale)
        x, y = curved_spectre(a, b, curves, scale, samples=60)
        edges = curved_spectre_edges(a, b, curves, scale, samples=60)
        ax.fill(x, y, color="#dbe7ff", alpha=0.75, lw=0)
        ax.plot(*np.vstack([P, P[:1]]).T, "--", color="0.65", lw=0.8)
        for i in range(N_EDGES):  # colour = sign/size of that edge's amount
            col = "0.35" if abs(k[i]) < 1e-12 else cmap(0.5 + 0.5 * float(np.clip(k[i], -1, 1)))
            ax.plot(edges[i, :, 0], edges[i, :, 1], color=col, lw=2.2, solid_capstyle="round")
        simple = is_simple(x, y)
        frame(ax, title + ("" if simple else "\n(outline crosses itself)"), "black" if simple else "crimson")

    def draw_reference(ax):
        d = edge_vectors(a, b)
        clockwise = polygon_area(*P.T) < 0
        ax.fill(*P.T, color="#dbe7ff", alpha=0.75, lw=0)
        ax.plot(*np.vstack([P, P[:1]]).T, "-", color="0.35", lw=1.6)
        for i in range(N_EDGES):
            mid = P[i] + d[i] / 2
            nrm = np.array([-d[i, 1], d[i, 0]]) if clockwise else np.array([d[i, 1], -d[i, 0]])
            nrm = nrm / np.hypot(*nrm)
            ax.text(*(mid + 0.27 * nrm), "%d%s" % (i, EDGE_KIND[i]), ha="center", va="center",
                    fontsize=9, color="navy")
            ax.plot(*P[i], "o", ms=3, color="0.2")
        frame(ax, "edge index + length kind (a/b)")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
    axes = axes.ravel()
    draw_reference(axes[0])
    variations = _variations(seed)
    for ax, (title, kw) in zip(axes[1:11], variations):
        draw_tile(ax, title, **kw)

    ax = axes[11]  # the single scale at work: same curves, growing scale
    ref = np.linspace(-1, 1, N_EDGES)
    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    for s, col in zip(scales, plt.get_cmap("viridis")(np.linspace(0, 0.9, len(scales)))):
        x, y = curved_spectre(a, b, ref, s, samples=60)
        ax.plot(x, y, color=col, lw=1.3, label="scale %.2f" % s)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    frame(ax, "one ramp, scale 0 .. 1")

    fig.suptitle(
        "Curved Spectre monotile: per-edge curve amounts x one scale   "
        "(a=%g, b=%g; edge colour: red = positive, blue = negative, grey = straight; "
        "dashed = straight polygon)" % (a, b),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
        fig.savefig(save, dpi=130)
        print("saved", save)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run the self tests of the per-edge curved Spectre monotile and "
        "show some variations with matplotlib."
    )
    p.add_argument("--a", type=float, default=1.0, help="edge length a (default 1)")
    p.add_argument("--b", type=float, default=1.0, help="edge length b (default 1)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random variation")
    p.add_argument("--skip-tests", action="store_true", help="only show the variations")
    p.add_argument("--tests-only", action="store_true", help="run the self tests and exit (no plot)")
    p.add_argument("--no-show", action="store_true", help="do not open a window (use with --save)")
    p.add_argument("--save", metavar="PNG", help="also save the figure to this file")
    args = p.parse_args(argv)

    if not args.skip_tests:
        print("spectre_operator self tests:")
        if not run_tests():
            return 1
    if args.tests_only:
        return 0
    show_variations(args.a, args.b, args.seed, save=args.save, show=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
