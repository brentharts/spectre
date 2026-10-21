"""
spectre_patch.py - patches of Spectre tiles, edge adjacency and the tiling condition (step 5a).

The supertile substitution is a port of ``spectre.py`` in the ``brentharts/spectre`` fork
(``buildSpectreTiles``, itself from the Smith-Myers-Kaplan-Goodman-Strauss construction).  It gives
the exact position of every tile of an n-level patch as a rigid motion.  Here the same patch is
placed with the tile of THIS repository (spectre_operator): the reference tile is the mirror image
of it (with a cyclic shift of two vertices), which is handled once by ``_reference_to_tile``.

    from spectre_patch import SpectrePatch
    p = SpectrePatch(levels=2)               # 1, 9, 71, 559, ... tiles (grows by 4 + sqrt(15) per level)
    p.n_tiles, p.labels, p.placements        # (N, 3, 3) rigid motions of the tile of spectre_operator
    polys = p.polygons(curves=1.0, scale=0.5)   # the curved outlines, one (n, 2) array per tile
    p.edge_pairs()                           # which edge i of which tile meets which edge j of which tile
    p.pair_matrix()                          # 14 x 14 counts of (i, j) meetings
    p.compatible_classes()                   # groups of edge indices that must carry the SAME amount
    p.mismatch_area(curves, scale)           # exact total overlap + gap if the amounts disagree

Why the last two matter.  A curved edge is an S that is point symmetric about its midpoint, so the
tile on the other side of a shared edge sees the very same curve when both edges carry the same
amount.  If the amounts differ, the two curves cross at the midpoint and enclose two lobes of equal
area: one is an overlap of the two tiles, the other a gap.  The lobes have area
|k_i - k_j| |d|^2 I with I = int |w| s' dt, so the tiling condition is

        k_i = k_j for every pair (i, j) of edges that meet in the tiling.

The patch shows which pairs meet.  The result (see ``compatible_classes``) is that the edge indices
fall into very few classes - for the standard tiling a single one - so of the 14 per-edge amounts of
spectre_operator only the uniform curving makes a tiling; every other pattern (the optimal ones of
spectre_optimize included) is a single tile that does not tile the plane.

Run ``python3 spectre_patch.py`` to run the self tests and show the figure (--help for options).
Needs numpy and matplotlib.
"""

import argparse
import math
import os
import sys

import numpy as np

import spectre_operator as so

__all__ = [
    "TILE_NAMES",
    "SUBSTITUTIONS",
    "SpectrePatch",
    "lobe_integral",
    "pair_mismatch_area",
    "run_tests",
    "show_patch",
]

TILE_NAMES = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]

# Which tile fills each of the 8 slots of a supertile (None: slot unused).  From spectre.py.
SUBSTITUTIONS = {
    "Gamma": ("Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma"),
    "Delta": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Theta": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Lambda": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Xi": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    "Pi": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    "Sigma": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"),
    "Phi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Psi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
}


# --------------------------------------------------------------------------
# Affine maps as 3 x 3 homogeneous matrices
# --------------------------------------------------------------------------
def _rot(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _shift(v):
    m = np.eye(3)
    m[:2, 2] = v
    return m


def _apply(m, pts):
    return pts @ m[:2, :2].T + m[:2, 2]


def _reference_points(a, b):
    """The 14 vertices of the reference tile of spectre.py (counterclockwise)."""
    r3 = math.sqrt(3.0)
    a2, a3 = a * 0.5, a * r3 / 2.0
    b2, b3 = b * 0.5, b * r3 / 2.0
    return np.array([
        (0, 0), (a, 0), (a + a2, -a3), (a + a2 + b3, -a3 + b2), (a + a2 + b3, -a3 + b + b2),
        (2 * a + a2 + b3, -a3 + b + b2), (3 * a + b3, b + b2), (3 * a, 2 * b),
        (3 * a - b3, 2 * b - b2), (2 * a + a2 - b3, a3 + 2 * b - b2), (a + a2 - b3, a3 + 2 * b - b2),
        (a2 - b3, a3 + 2 * b - b2), (-b3, 2 * b - b2), (0, b),
    ], dtype=float)


def _reference_to_tile(a, b):
    """The motion taking the reference tile onto the tile of spectre_operator.  The reference tile
    is its mirror image and starts two vertices later: vertex j of ours is  M P[(j + 2) % 14] + t
    with M swapping x and y."""
    ref = _reference_points(a, b)
    ours = so.vertices(a, b)
    m = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    m[:2, 2] = ours[0] - m[:2, :2] @ ref[2]
    if not np.allclose(_apply(m, np.roll(ref, -2, axis=0)), ours, atol=1e-9):
        raise AssertionError("reference tile does not match spectre_operator (cannot happen)")
    return m


# --------------------------------------------------------------------------
# The supertile substitution (port of buildSupertiles)
# --------------------------------------------------------------------------
def _build(levels, a, b, rotation=30):
    """Tiles of every label after ``levels`` substitutions, as lists of (shape, 3x3 motion) in the
    frame of the reference tile; shape 0 is the Spectre tile (a, b), shape 1 the Mystic (b, a)."""
    pts = _reference_points(a, b)
    quad = pts[[3, 5, 7, 11]]
    tiles = {lab: [(0, np.eye(3))] for lab in TILE_NAMES if lab != "Gamma"}
    tiles["Gamma"] = [(0, np.eye(3)), (1, _shift(pts[8]) @ _rot(rotation))]
    for _ in range(levels):
        base = quad  # the quad of the current supertiles
        total, transformations = 0, [np.eye(3)]
        transformed = base
        for angle, src, dst in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            if angle != 0:
                total += angle
                rotation_m = _rot(total)
                transformed = _apply(rotation_m, base)
            else:
                rotation_m = _rot(total)
            trans = _shift(_apply(transformations[-1], base[src][None])[0] - transformed[dst])
            transformations.append(trans @ rotation_m)
        mirror = np.diag([-1.0, 1.0, 1.0])
        transformations = [mirror @ t for t in transformations]
        quad = np.array([
            _apply(transformations[6], base[2][None])[0], _apply(transformations[5], base[1][None])[0],
            _apply(transformations[3], base[2][None])[0], _apply(transformations[0], base[1][None])[0],
        ])
        new = {}
        for lab, subst in SUBSTITUTIONS.items():
            parts = []
            for name, trsf in zip(subst, transformations):
                if name:
                    parts.extend((shape, trsf @ m) for shape, m in tiles[name])
            new[lab] = parts
        tiles = new
    return tiles


def substitution_matrix():
    """Counts of each label in the supertile of each label: M[i, j] = number of tiles of label i
    in the supertile of label j (the Gamma supertile counted through its Gamma slot)."""
    idx = {lab: i for i, lab in enumerate(TILE_NAMES)}
    mat = np.zeros((9, 9), dtype=int)
    for lab, subst in SUBSTITUTIONS.items():
        for name in subst:
            if name:
                mat[idx[name], idx[lab]] += 1
    return mat


# --------------------------------------------------------------------------
# Patch
# --------------------------------------------------------------------------
class SpectrePatch:
    """An n-level patch of the Spectre tiling made of the tile of spectre_operator.

    Parameters
    ----------
    levels : number of substitutions (0: the Delta supertile alone, i.e. one tile).
    a, b : edge lengths.  Each tile is the tile (a, b); the Mystic tiles of the construction use
        (b, a), so for a != b the patch contains both.
    rigid : the construction places mirrored tiles on every other level; with ``rigid`` the whole
        patch is mirrored when needed so that every placement is a proper rotation + translation
        of the tile of spectre_operator (the right chirality).

    Attributes: ``placements`` (N, 3, 3), ``shape`` (N,) 0 = Spectre, 1 = Mystic, ``n_tiles``.
    """

    def __init__(self, levels=2, a=1.0, b=1.0, rigid=True):
        if int(levels) != levels or levels < 0 or levels > 6:
            raise ValueError("levels must be an integer between 0 and 6")
        if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
            raise ValueError("a and b must be positive and finite")
        self.levels, self.a, self.b = int(levels), float(a), float(b)
        parts = _build(self.levels, a, b)["Delta"]
        t_ab, t_ba = _reference_to_tile(a, b), _reference_to_tile(b, a)
        place, shape = [], []
        for sh, g in parts:
            place.append(g @ np.linalg.inv(t_ba if sh else t_ab))
            shape.append(sh)
        place = np.array(place)
        dets = np.linalg.det(place[:, :2, :2])
        if rigid and np.all(dets < 0):
            place = np.diag([-1.0, 1.0, 1.0]) @ place
        elif rigid and not np.all(dets > 0):
            raise AssertionError("placements of mixed chirality (cannot happen)")
        self.placements, self.shape = place, np.array(shape)
        self.n_tiles = len(place)
        self._pairs = None

    # -- geometry -------------------------------------------------------------
    def tile_params(self, i):
        """(a, b) of tile i: the Mystic tiles have the two lengths swapped."""
        return (self.b, self.a) if self.shape[i] else (self.a, self.b)

    def straight_polygon(self, i):
        v = so.vertices(*self.tile_params(i))
        return _apply(self.placements[i], v)

    def polygons(self, curves=1.0, scale=0.5, samples=50):
        """Curved outlines of all tiles: a list of (14 * samples, 2) arrays."""
        out = []
        cache = {}
        for i in range(self.n_tiles):
            key = int(self.shape[i])
            if key not in cache:
                x, y = so.curved_spectre(*self.tile_params(i), curves, scale, samples=samples, closed=False)
                cache[key] = np.column_stack([x, y])
            out.append(_apply(self.placements[i], cache[key]))
        return out

    @property
    def area(self):
        """Total area of the tiles (the Mystic tile (b, a) has a different area when a != b)."""
        return float(sum(so.spectre_area(*self.tile_params(i)) for i in range(self.n_tiles)))

    def angles(self):
        """Rotation angle of every placement in degrees, in (-180, 180]."""
        m = self.placements[:, :2, :2]
        return np.degrees(np.arctan2(m[:, 1, 0], m[:, 0, 0]))

    # -- adjacency ---------------------------------------------------------------
    def edge_pairs(self, decimals=6):
        """Pairs of meeting edges as an integer array of rows (tile A, edge i, tile B, edge j),
        each pair once (A < B).  Two edges meet when they coincide as straight segments run in
        opposite directions (the tiles are all traversed the same way round)."""
        if self._pairs is not None:
            return self._pairs
        directed = {}
        for t in range(self.n_tiles):
            poly = self.straight_polygon(t)
            nxt = np.roll(poly, -1, axis=0)
            for e in range(so.N_EDGES):
                key = (tuple(np.round(poly[e], decimals) + 0.0), tuple(np.round(nxt[e], decimals) + 0.0))
                if key in directed:
                    raise AssertionError("two tiles run the same directed edge: they overlap")
                directed[key] = (t, e)
        rows = []
        for (p, q), (t, e) in directed.items():
            other = directed.get((q, p))
            if other is not None and t < other[0]:
                rows.append((t, e, other[0], other[1]))
        self._pairs = np.array(sorted(rows), dtype=int).reshape(-1, 4)
        return self._pairs

    def boundary_edges(self):
        """(tile, edge) of every edge that has no partner: the outline of the patch."""
        pairs = self.edge_pairs()
        used = {(int(t), int(e)) for t, e, _, _ in pairs} | {(int(t), int(e)) for _, _, t, e in pairs}
        return [(t, e) for t in range(self.n_tiles) for e in range(so.N_EDGES) if (t, e) not in used]

    def pair_matrix(self):
        """Symmetric 14 x 14 matrix: how often edge i meets edge j (i != j counted both ways)."""
        mat = np.zeros((so.N_EDGES, so.N_EDGES), dtype=int)
        for _, i, _, j in self.edge_pairs():
            mat[i, j] += 1
            mat[j, i] += 1
        return mat

    def compatible_classes(self):
        """Groups (sorted lists) of edge indices forced to carry equal amounts by the tiling."""
        parent = list(range(so.N_EDGES))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _, i, _, j in self.edge_pairs():
            parent[find(int(i))] = find(int(j))
        groups = {}
        for e in range(so.N_EDGES):
            groups.setdefault(find(e), []).append(e)
        return sorted(groups.values())

    def mismatch_area(self, curves=1.0, scale=1.0):
        """Exact total area of overlaps PLUS gaps that the amounts cause along the shared edges:
        sum over meeting pairs of |k_i - k_j| |d|^2 I.  Half of it is overlap, half gap."""
        k = so.edge_strengths(curves, scale)
        total = 0.0
        for t, i, u, j in self.edge_pairs():
            length = float(np.hypot(*so.edge_vectors(*self.tile_params(int(t)))[i]))
            total += pair_mismatch_area(k[i], k[j], length)
        return total


def lobe_integral():
    """I = int_0^1 |w(t)| s'(t) dt, exactly: the area between two curves of a shared edge of chord
    length |d| whose amounts differ by dk is  |dk| |d|^2 I  (two lobes of I / 2 each).
    w <= 0 on [0, 1/2] and w(1 - t) = -w(t), s'(1 - t) = s'(t), so I = -2 int_0^(1/2) w s' dt."""
    P = np.polynomial.Polynomial
    w = P([0.0, -3.0, 9.0, -6.0])  # 3 t (1 - t) (2t - 1)
    sp = P([1.5, -3.0, 3.0])
    anti = (w * sp).integ()
    return float(-2.0 * (anti(0.5) - anti(0.0)))


def pair_mismatch_area(k1, k2, length):
    """Area between the two curves of a shared edge of the given chord length."""
    return abs(float(k1) - float(k2)) * length**2 * lobe_integral()


# --------------------------------------------------------------------------
# Self tests
# --------------------------------------------------------------------------
_PATCHES = {}


def _patch(levels, a=1.0, b=1.0):
    if (levels, a, b) not in _PATCHES:
        _PATCHES[(levels, a, b)] = SpectrePatch(levels, a, b)
    return _PATCHES[(levels, a, b)]


def _coverage_grid(polys, h=0.02, margin=0.2):
    """Number of polygons containing each cell centre of a grid: (counts (ny, nx), xs, ys)."""
    from matplotlib.path import Path

    allp = np.vstack(polys)
    lo, hi = allp.min(axis=0) - margin, allp.max(axis=0) + margin
    xs, ys = np.arange(lo[0], hi[0], h) + h / 2, np.arange(lo[1], hi[1], h) + h / 2
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    count = np.zeros(len(pts), dtype=int)
    for poly in polys:
        lo_p, hi_p = poly.min(axis=0), poly.max(axis=0)
        near = np.where((pts[:, 0] >= lo_p[0]) & (pts[:, 0] <= hi_p[0]) & (pts[:, 1] >= lo_p[1]) & (pts[:, 1] <= hi_p[1]))[0]
        if len(near):
            count[near] += Path(poly).contains_points(pts[near])
    return count.reshape(gx.shape), xs, ys


def _coverage(polys, h=0.02, margin=0.2):
    """Flat coverage counts of the grid cells and the cell area."""
    count, _, _ = _coverage_grid(polys, h, margin)
    return count.ravel(), h * h


def test_substitution_reproduces_the_counts_and_the_inflation_factor():
    # level n has 1, 9, 71, 559 tiles; independent route: the substitution matrix
    mat = substitution_matrix()
    weights = np.ones(9)
    weights[TILE_NAMES.index("Gamma")] = 2  # a bare Gamma is a Spectre plus a Mystic
    delta = np.zeros(9)
    delta[TILE_NAMES.index("Delta")] = 1
    counts = [int(weights @ np.linalg.matrix_power(mat, n) @ delta) for n in range(5)]
    assert counts[:4] == [1, 9, 71, 559], counts
    for n in range(4):
        assert SpectrePatch(n).n_tiles == counts[n]
    # the area inflation per level is the Perron eigenvalue 4 + sqrt(15) of the substitution
    perron = max(np.linalg.eigvals(mat).real)
    assert abs(perron - (4.0 + math.sqrt(15.0))) < 1e-12
    assert abs(counts[4] / counts[3] - perron) < 5e-3 and abs(counts[3] / counts[2] - perron) < 5e-2


def test_placements_are_proper_rigid_motions_in_multiples_of_thirty_degrees():
    for levels in (1, 3):
        p = _patch(levels)
        lin = p.placements[:, :2, :2]
        assert np.allclose(lin @ lin.transpose(0, 2, 1), np.eye(2), atol=1e-9)
        assert np.all(np.linalg.det(lin) > 0)  # no mirrored tiles: the tile keeps its chirality
        assert np.allclose(p.placements[:, 2], [0, 0, 1])
        ang = p.angles() / 30.0
        assert np.allclose(ang, np.round(ang), atol=1e-9)
    # the reference tile is our tile reflected: both are the same polygon up to that motion
    ref, ours = _reference_points(1.0, 2.0), so.vertices(1.0, 2.0)
    assert np.allclose(_apply(_reference_to_tile(1.0, 2.0), np.roll(ref, -2, axis=0)), ours)


def test_tiles_meet_edge_to_edge_without_overlap():
    for (levels, a, b) in ((2, 1.0, 1.0), (2, 1.0, 2.0)):
        p = _patch(levels, a, b)
        pairs = p.edge_pairs()  # raises if two tiles run the same directed edge
        # every edge either has exactly one partner or is on the outline of the patch
        assert 14 * p.n_tiles == 2 * len(pairs) + len(p.boundary_edges())
        # every tile has the area of its (a, b) or (b, a) tile; the straight polygons never overlap
        polys = [p.straight_polygon(i) for i in range(p.n_tiles)]
        for i, poly in enumerate(polys):
            assert abs(abs(so.polygon_area(*poly.T)) - so.spectre_area(*p.tile_params(i))) < 1e-9
        count, cell = _coverage(polys, h=0.05)
        assert count.max() == 1
        # the covered area of the grid is the patch area, up to the grid resolution at the outline
        assert abs(count.sum() * cell - p.area) < 0.02 * p.area


def test_meeting_edges_have_the_same_length_and_kind():
    # a meets a and b meets b; the Mystic tiles have their lengths swapped, which is how the tiling
    # forces the edge indexing (and hence the cyclic shift of the port) to be right
    p = _patch(3, 1.0, 2.0)
    kinds = np.array(list(so.EDGE_KIND))
    for t, e, u, f in p.edge_pairs():
        kt = kinds[e] if not p.shape[t] else ("a" if kinds[e] == "b" else "b")
        ku = kinds[f] if not p.shape[u] else ("a" if kinds[f] == "b" else "b")
        assert kt == ku
        lt = np.hypot(*so.edge_vectors(*p.tile_params(int(t)))[e])
        lu = np.hypot(*so.edge_vectors(*p.tile_params(int(u)))[f])
        assert abs(lt - lu) < 1e-12


def test_edge_meeting_graph_is_connected_so_only_uniform_curving_tiles():
    p = _patch(3)
    mat = p.pair_matrix()
    assert np.array_equal(mat, mat.T) and mat.sum() == 2 * len(p.edge_pairs())
    assert np.all(np.diag(mat) == 0)  # an edge never meets an edge with the same index here
    # the meeting graph on the 14 indices is connected: one class, k_i = k_j for all i, j
    assert p.compatible_classes() == [list(range(14))]
    assert _patch(2).compatible_classes() == [list(range(14))]
    # every index has partners
    assert np.all(mat.sum(axis=1) > 0)


def test_area_between_two_edge_curves_is_the_lobe_formula():
    assert abs(lobe_integral() - 3.0 / 16.0) < 1e-15  # I = 3/16 exactly
    # independent: polygons of the two curves, lobe by lobe (the signed total is zero by symmetry)
    for k1, k2, i in ((0.2, 0.5, 0), (-0.3, 0.4, 3), (0.0, 0.7, 8)):
        n = 4001
        t = np.linspace(0.0, 1.0, n)
        ctrl = so.edge_beziers(1, 1, np.array([k1] * 14), 1.0)
        c1 = so.evaluate_edges(ctrl, t)[i]
        ctrl = so.edge_beziers(1, 1, np.array([k2] * 14), 1.0)
        c2 = so.evaluate_edges(ctrl, t)[i]
        half = n // 2
        lobe1 = np.vstack([c1[: half + 1], c2[: half + 1][::-1]])
        lobe2 = np.vstack([c1[half:], c2[half:][::-1]])
        total = abs(so.polygon_area(*lobe1.T)) + abs(so.polygon_area(*lobe2.T))
        length = float(np.hypot(*so.edge_vectors(1, 1)[i]))
        assert abs(total - pair_mismatch_area(k1, k2, length)) < 1e-6, (total, pair_mismatch_area(k1, k2, length))
        assert abs(so.polygon_area(*lobe1.T) + so.polygon_area(*lobe2.T)) < 1e-9  # lobes cancel


def test_uniform_curving_tiles_other_patterns_overlap_by_the_predicted_area():
    p = _patch(1)
    # uniform curving: the curved tiles still do not overlap
    polys = p.polygons(1.0, 0.5, samples=50)
    count, cell = _coverage(polys, h=0.02)
    assert count.max() == 1
    # per-edge amounts: overlap area = half the mismatch area (the other half is a gap)
    rng = np.random.default_rng(6)
    curves = rng.uniform(-1.0, 1.0, so.N_EDGES)
    polys = p.polygons(curves, 0.5, samples=50)
    count, cell = _coverage(polys, h=0.01)
    overlap = float((count >= 2).sum() * cell)
    predicted = 0.5 * p.mismatch_area(curves, 0.5)
    assert predicted > 0.05
    assert abs(overlap - predicted) < 0.03 * predicted + 3e-3, (overlap, predicted)


def test_mismatch_area_is_zero_iff_uniform_and_scales_with_the_differences():
    p = _patch(2)
    rng = np.random.default_rng(9)
    curves = rng.uniform(-1.0, 1.0, so.N_EDGES)
    base = p.mismatch_area(curves, 0.3)
    assert base > 0
    assert abs(p.mismatch_area(curves, 0.6) - 2.0 * base) < 1e-9 * base  # linear in the amounts
    assert abs(p.mismatch_area(curves + 5.0, 0.3) - base) < 1e-9 * base  # only differences count
    assert p.mismatch_area(1.0, 0.4) == 0.0 and p.mismatch_area(-0.2, 1.0) == 0.0
    big = _patch(3)  # more tiles, proportionally more mismatch
    assert 6.0 < big.mismatch_area(curves, 0.3) / base < 9.0


def test_validation():
    for call in (
        lambda: SpectrePatch(-1),
        lambda: SpectrePatch(2.5),
        lambda: SpectrePatch(7),
        lambda: SpectrePatch(1, a=-1.0),
        lambda: SpectrePatch(1, b=float("nan")),
        lambda: _patch(1).mismatch_area([0.0] * 5, 1.0),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def run_tests(verbose=True):
    """Run every ``test_*`` function in this module; return True if none failed."""
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            status = "PASS"
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failed += 1
            status = "FAIL: %s: %s" % (type(exc).__name__, exc)
        if verbose:
            print("  %-78s %s" % (name, status))
    if verbose:
        print("%d test(s), %d failed" % (len(tests), failed))
    return failed == 0


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def show_patch(a=1.0, b=1.0, k=0.5, levels=3, seed=6, save=None, show=True):
    """Six panels: a curved patch coloured by orientation; the patch with per-edge amounts (the
    overlaps are the darker lobes); a zoom on one shared edge; which edge indices meet; the tile
    counts per level; the tiling defect against the spread of the amounts."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    p = SpectrePatch(levels, a, b)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.6))

    # 1. the curved patch
    ax = axes[0, 0]
    polys = p.polygons(1.0, k, samples=12)
    ang = np.round(p.angles() / 30.0).astype(int) % 12
    cmap = plt.get_cmap("twilight")
    ax.add_collection(PolyCollection(polys, facecolors=cmap(ang / 12.0), edgecolors="k", linewidths=0.25))
    allp = np.vstack(polys)
    ax.set_xlim(allp[:, 0].min() - 0.5, allp[:, 0].max() + 0.5)
    ax.set_ylim(allp[:, 1].min() - 0.5, allp[:, 1].max() + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("%d curved tiles (k = %g), colour = orientation in steps of 30 degrees:\n"
                 "uniform curving tiles without gaps or overlaps" % (p.n_tiles, k), fontsize=9)

    # 2. per-edge amounts: overlaps
    small = SpectrePatch(1, a, b)
    curves = np.random.default_rng(seed).uniform(-1.0, 1.0, so.N_EDGES)
    polys2 = small.polygons(curves, 0.5, samples=50)
    grid, gxs, gys = _coverage_grid(polys2, h=0.01)
    overlap = float((grid >= 2).sum() * 0.01 * 0.01)
    predicted = 0.5 * small.mismatch_area(curves, 0.5)
    ax = axes[0, 1]
    ax.add_collection(PolyCollection(polys2, facecolors=(0.8, 0.88, 1.0, 1.0), edgecolors="k", linewidths=0.5))
    ax.imshow(np.where(grid >= 2, 1.0, np.nan), extent=(gxs[0], gxs[-1], gys[0], gys[-1]), origin="lower",
              cmap="autumn", vmin=0, vmax=1.5, interpolation="nearest", zorder=3)
    lo, hi = np.vstack(polys2).min(axis=0) - 0.2, np.vstack(polys2).max(axis=0) + 0.2
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("9 tiles with random per-edge amounts: they overlap (orange) and leave gaps (white lobes);\n"
                 "overlap area %.3f, predicted from the pairs %.3f" % (overlap, predicted), fontsize=9)

    # 3. zoom on one shared edge: two curves, two lobes
    ax = axes[0, 2]
    t = np.linspace(0, 1, 200)
    for kk, col in ((0.2, "tab:blue"), (0.7, "tab:red")):
        ctrl = so.edge_beziers(1, 1, np.array([kk] * 14), 1.0)
        c = so.evaluate_edges(ctrl, t)[0]
        ax.plot(c[:, 0], c[:, 1], color=col, lw=2, label="edge amount %.1f" % kk)
    c1 = so.evaluate_edges(so.edge_beziers(1, 1, np.array([0.2] * 14), 1.0), t)[0]
    c2 = so.evaluate_edges(so.edge_beziers(1, 1, np.array([0.7] * 14), 1.0), t)[0]
    ax.fill(np.r_[c1[:101, 0], c2[:101, 0][::-1]], np.r_[c1[:101, 1], c2[:101, 1][::-1]], color="tab:orange", alpha=0.5)
    ax.fill(np.r_[c1[100:, 0], c2[100:, 0][::-1]], np.r_[c1[100:, 1], c2[100:, 1][::-1]], color="tab:green", alpha=0.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("one shared edge, amounts k_i and k_j: the curves cross at the midpoint,\n"
                 "enclosing an overlap lobe and a gap lobe of area |k_i - k_j| |d|^2 (3/32) each", fontsize=9)
    ax.axis("off")

    # 4. which edge indices meet
    ax = axes[1, 0]
    mat = p.pair_matrix()
    im = ax.imshow(np.where(mat > 0, mat, np.nan), cmap="viridis", norm=matplotlib.colors.LogNorm())
    ax.set_xticks(range(14))
    ax.set_yticks(range(14))
    ax.set_xlabel("edge j")
    ax.set_ylabel("edge i")
    fig.colorbar(im, ax=ax, shrink=0.7, label="how often edge i meets edge j")
    classes = p.compatible_classes()
    ax.set_title("meeting graph of the edges (level %d): %d class%s\n%s" % (
        levels, len(classes), "" if len(classes) == 1 else "es",
        "all 14 edges are linked: only uniform curving tiles" if len(classes) == 1 else str(classes)), fontsize=9)

    # 5. counts
    ax = axes[1, 1]
    mat9 = substitution_matrix()
    weights = np.ones(9)
    weights[TILE_NAMES.index("Gamma")] = 2
    delta = np.zeros(9)
    delta[TILE_NAMES.index("Delta")] = 1
    ns = np.arange(0, 7)
    counts = [float(weights @ np.linalg.matrix_power(mat9, n) @ delta) for n in ns]
    ax.semilogy(ns, counts, "o-", color="tab:purple")
    ax.semilogy(ns, [counts[0] * (4 + math.sqrt(15)) ** n for n in ns], "k--", lw=0.9, label="(4 + sqrt 15)^n")
    for n, c in zip(ns, counts):
        ax.annotate("%d" % c, (n, c), xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.set_xlabel("substitution level n")
    ax.set_ylabel("number of tiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("tiles per level: each level multiplies the count by 4 + sqrt(15) = 7.873", fontsize=9)

    # 6. tiling defect against the spread of the amounts
    ax = axes[1, 2]
    big = SpectrePatch(2, a, b)
    sig = np.linspace(0, 0.6, 13)
    for sd in range(5):
        base = np.random.default_rng(100 + sd).standard_normal(so.N_EDGES)
        base = (base - base.mean()) / base.std()
        ax.plot(sig, [big.mismatch_area(s * base, 1.0) / big.n_tiles for s in sig], lw=1.6, alpha=0.8)
    ax.set_xlabel("standard deviation of the 14 amounts")
    ax.set_ylabel("overlap + gap area per tile")
    ax.set_title("tiling defect per tile: linear in the spread of the amounts,\n"
                 "zero only when all agree (a common shift changes nothing)", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Patches of the curved Spectre monotile (a=%g, b=%g): supertile substitution ported from the spectre fork, "
                 "and the edge condition k_i = k_j for tiling" % (a, b), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
        fig.savefig(save, dpi=100)
        print("saved", save)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Patches of the curved Spectre monotile and the edge condition for tiling.")
    p.add_argument("--a", type=float, default=1.0, help="edge length a (default 1)")
    p.add_argument("--b", type=float, default=1.0, help="edge length b (default 1)")
    p.add_argument("--curve_strength", type=float, default=0.5, help="uniform curve strength k (default 0.5)")
    p.add_argument("--levels", type=int, default=3, help="substitution levels of the big patch (default 3)")
    p.add_argument("--seed", type=int, default=6, help="seed of the random per-edge amounts")
    p.add_argument("--skip-tests", action="store_true", help="only make the figure")
    p.add_argument("--tests-only", action="store_true", help="run the self tests and exit (no plot)")
    p.add_argument("--no-show", action="store_true", help="do not open a window (use with --save)")
    p.add_argument("--save", metavar="PNG", help="also save the figure to this file")
    args = p.parse_args(argv)

    if not args.skip_tests:
        print("spectre_patch self tests:")
        if not run_tests():
            return 1
    if args.tests_only:
        return 0
    show_patch(args.a, args.b, args.curve_strength, args.levels, args.seed, save=args.save, show=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
