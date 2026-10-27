#!/usr/bin/env python3
"""chair44_unfold.py - unfold the surface of the Chair44 tile into a flat net (a polyomino).

The seven-cube chair is a topological sphere made of 24 unit squares (48 edges, 26 vertices).
Keeping a spanning tree of the face-adjacency graph as hinges (23 kept, 25 cut) and folding every face
into the plane gives a 24-omino.  Different spanning trees give different nets; some overlap
themselves and are rejected.  Positions are small integers and half-integers in floating point, so exact.

Public API
  surface()                    faces, edges, adjacency, fold type of each edge
  unfold(tree)                 {face: 2D cell} for a spanning tree (list of face pairs)
  random_tree(rng)             spanning tree from random edge weights (Kruskal)
  count_spanning_trees()       exact, by the matrix-tree theorem
  net_stats(cells)             area, perimeter, bounding box, extra adjacencies
  canonical(cells)             canonical form of the polyomino under the 8 square symmetries
  tiles_by_translation(cells)  lattice translation tiling test (Hermite normal form search)
  survey(n, seed)              sample n trees, summarise the valid nets

CLI: python3 chair44_unfold.py [--survey N] [--png figures/chair44_unfold.png]
"""
import argparse
import itertools
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction

import chair44_core as C


# ------------------------------------------------------------------ the surface
def _corners(face):
    """4 vertices (integer 3D) of a unit face in cyclic order."""
    ax, sd, c = face
    u, v = [a for a in range(3) if a != ax]
    plane = c[ax] + (1 if sd > 0 else 0)
    out = []
    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
        p = [0, 0, 0]
        p[ax], p[u], p[v] = plane, c[u] + du, c[v] + dv
        out.append(tuple(p))
    return out


def surface(cells=C.CHAIR_CELLS):
    """faces (list), edges {frozenset(vertex pair): [face idx, face idx]},
    fold {edge: 'flat'|'convex'|'concave'}."""
    cells = set(cells)
    faces = C.exposed_faces(cells)
    edges = defaultdict(list)
    for i, f in enumerate(faces):
        v = _corners(f)
        for k in range(4):
            edges[frozenset((v[k], v[(k + 1) % 4]))].append(i)
    fold = {}
    for e, fs in edges.items():
        assert len(fs) == 2, "not a closed manifold surface"
        p, q = sorted(e)
        ax = [a for a in range(3) if p[a] != q[a]][0]
        o = [a for a in range(3) if a != ax]
        n = 0
        for d0 in (-1, 0):
            for d1 in (-1, 0):
                c = [0, 0, 0]
                c[ax] = p[ax]
                c[o[0]] = p[o[0]] + d0
                c[o[1]] = p[o[1]] + d1
                n += tuple(c) in cells
        fold[e] = {1: 'convex', 3: 'concave', 2: 'flat'}[n]
    return faces, dict(edges), fold


def adjacency(edges):
    adj = defaultdict(list)
    for e, (a, b) in edges.items():
        adj[a].append((b, e))
        adj[b].append((a, e))
    return adj


def count_spanning_trees(faces=None, edges=None):
    """Number of spanning trees of the dual graph (matrix-tree theorem, exact)."""
    import sympy as sp
    if faces is None:
        faces, edges, _ = surface()
    n = len(faces)
    L = sp.zeros(n, n)
    for a, b in edges.values():
        L[a, a] += 1
        L[b, b] += 1
        L[a, b] -= 1
        L[b, a] -= 1
    return L[1:, 1:].det()


def random_tree(rng, edges):
    """Spanning tree as a list of edge keys: Kruskal on random weights."""
    n = 1 + max(max(v) for v in edges.values())
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    order = list(edges)
    rng.shuffle(order)
    tree = []
    for e in order:
        a, b = edges[e]
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            tree.append(e)
    assert len(tree) == n - 1
    return tree


# ------------------------------------------------------------------ unfolding
def unfold(faces, edges, tree, root=0):
    """Fold the faces into the plane along the tree.  Returns {face: 2D cell (i, j)}.
    Each face keeps its four 2D corner positions; a child is attached across the shared
    edge on the side opposite its parent."""
    adj = defaultdict(list)
    for e in tree:
        a, b = edges[e]
        adj[a].append((b, e))
        adj[b].append((a, e))
    pos = {}                                  # face -> {3D vertex: 2D point}
    v0 = _corners(faces[root])
    pos[root] = dict(zip(v0, [(0, 0), (1, 0), (1, 1), (0, 1)]))
    stack = [root]
    while stack:
        a = stack.pop()
        for b, e in adj[a]:
            if b in pos:
                continue
            p3, q3 = sorted(e)
            p, q = pos[a][p3], pos[a][q3]
            ctr = (sum(x for x, _ in pos[a].values()) / 4.0, sum(y for _, y in pos[a].values()) / 4.0)
            ex, ey = q[0] - p[0], q[1] - p[1]
            s = (-ey, ex)                     # perpendicular, unit length
            mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
            if (ctr[0] - mid[0]) * s[0] + (ctr[1] - mid[1]) * s[1] > 0:
                s = (ey, -ex)                 # point away from the parent
            cv = _corners(faces[b])
            k = cv.index(p3)
            # neighbours of p3 and q3 in the cyclic order of b
            other_p = [w for w in (cv[(k - 1) % 4], cv[(k + 1) % 4]) if w != q3][0]
            other_q = [w for w in cv if w not in (p3, q3, other_p)][0]
            pos[b] = {p3: p, q3: q,
                      other_p: (p[0] + s[0], p[1] + s[1]),
                      other_q: (q[0] + s[0], q[1] + s[1])}
            stack.append(b)
    assert len(pos) == len(faces)
    cells = {}
    for f, d in pos.items():
        xs = [x for x, _ in d.values()]
        ys = [y for _, y in d.values()]
        cells[f] = (int(round(min(xs))), int(round(min(ys))))
    return cells


# ------------------------------------------------------------------ polyomino tools
def is_valid(cells):
    return len(set(cells.values())) == len(cells)


def net_stats(cells):
    S = set(cells.values()) if isinstance(cells, dict) else set(cells)
    adj = sum(((x + 1, y) in S) + ((x, y + 1) in S) for x, y in S)
    xs = [x for x, _ in S]
    ys = [y for _, y in S]
    return dict(area=len(S), perimeter=4 * len(S) - 2 * adj, adjacencies=adj,
                bbox=(max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))


def _normalise(S):
    mx = min(x for x, _ in S)
    my = min(y for _, y in S)
    return tuple(sorted((x - mx, y - my) for x, y in S))


def canonical(cells):
    S = list(cells.values()) if isinstance(cells, dict) else list(cells)
    best = None
    for sw in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                T = [((y if sw else x) * sx, (x if sw else y) * sy) for x, y in S]
                k = _normalise(T)
                if best is None or k < best:
                    best = k
    return best


def is_connected(cells):
    S = set(cells.values()) if isinstance(cells, dict) else set(cells)
    start = next(iter(S))
    seen, stack = {start}, [start]
    while stack:
        x, y = stack.pop()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + d[0], y + d[1])
            if n in S and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(S)


def tiles_by_translation(cells):
    """True iff translates of the polyomino by some lattice tile the plane exactly.
    Lattices of index N = area have Hermite basis (a, 0), (b, d), a*d = N, 0 <= b < a."""
    S = set(cells.values()) if isinstance(cells, dict) else set(cells)
    N = len(S)
    for a in range(1, N + 1):
        if N % a:
            continue
        d = N // a
        for b in range(a):
            # cell (x, y) reduces to class (x - (y // d) * b) mod a, y mod d  -> all N classes distinct
            classes = {((x - (y // d) * b) % a, y % d) for x, y in S}
            if len(classes) == N:
                return True
    return False


# ------------------------------------------------------------------ survey
def survey(n=2000, seed=1):
    faces, edges, fold = surface()
    rng = random.Random(seed)
    nets = {}
    bad = 0
    for _ in range(n):
        t = random_tree(rng, edges)
        c = unfold(faces, edges, t)
        if not is_valid(c):
            bad += 1
            continue
        key = canonical(c)
        if key not in nets:
            nets[key] = (c, t)
    st = {k: net_stats(k) for k in nets}
    return dict(samples=n, overlapping=bad, valid=n - bad, distinct=len(nets), nets=nets, stats=st,
                tilers=sum(tiles_by_translation(k) for k in nets))


# ------------------------------------------------------------------ tests
def all_trees(edges, n):
    """Every spanning tree (as edge keys) by brute force; only for tiny graphs."""
    keys = list(edges)
    for comb in itertools.combinations(keys, n - 1):
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        ok = True
        for e in comb:
            a, b = edges[e]
            ra, rb = find(a), find(b)
            if ra == rb:
                ok = False
                break
            parent[ra] = rb
        if ok:
            yield list(comb)


def test_core_geometry():
    assert C.self_check()


def test_chair_surface_is_a_sphere_with_the_expected_folds():
    faces, edges, fold = surface()
    verts = {v for f in faces for v in _corners(f)}
    assert (len(faces), len(edges), len(verts)) == (24, 48, 26)
    assert len(verts) - len(edges) + len(faces) == 2
    assert Counter(fold.values()) == {'convex': 27, 'flat': 18, 'concave': 3}


def test_spanning_tree_count_matches_the_matrix_tree_theorem_on_a_cube():
    faces, edges, _ = surface([(0, 0, 0)])
    assert len(faces) == 6 and int(count_spanning_trees(faces, edges)) == 384
    assert sum(1 for _ in all_trees(edges, 6)) == 384


def test_unfolding_the_cube_gives_exactly_the_eleven_cube_nets():
    faces, edges, _ = surface([(0, 0, 0)])
    nets = set()
    for t in all_trees(edges, 6):
        c = unfold(faces, edges, t)
        if is_valid(c):
            nets.add(canonical(c))
    assert len(nets) == 11 and all(len(n) == 6 for n in nets)


def test_every_tree_edge_stays_glued_in_the_plane():
    faces, edges, _ = surface()
    rng = random.Random(7)
    for _ in range(200):
        t = random_tree(rng, edges)
        c = unfold(faces, edges, t)
        for e in t:
            a, b = edges[e]
            assert abs(c[a][0] - c[b][0]) + abs(c[a][1] - c[b][1]) == 1
        if is_valid(c):
            st = net_stats(c)
            assert st["area"] == 24 and is_connected(c) and st["perimeter"] >= 20
            assert st["perimeter"] <= 4 * 24 - 2 * 23


def test_tilers_and_canonical_form():
    assert tiles_by_translation([(0, 0), (1, 0), (0, 1), (1, 1)])            # square
    assert tiles_by_translation([(0, 0), (1, 0), (2, 0), (3, 0)])            # bar
    plus = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]
    assert tiles_by_translation(plus)                                          # the plus tiles
    assert not tiles_by_translation([(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2), (1, 2)])  # ring
    assert canonical([(0, 0), (1, 0), (1, 1)]) == canonical([(5, 5), (5, 6), (4, 6)])


def test_survey_is_reproducible_and_finds_valid_nets():
    a, b = survey(300, 3), survey(300, 3)
    assert a["distinct"] == b["distinct"] and a["overlapping"] == b["overlapping"]
    assert 0 < a["valid"] < 300 and a["overlapping"] > 0


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


# ------------------------------------------------------------------ figure
def draw(ax, cells, faces, edges, tree, fold, title=""):
    import numpy as np
    from matplotlib.patches import Rectangle
    pal = {0: '#4c78a8', 1: '#f58518', 2: '#54a24b'}
    for f, (x, y) in cells.items():
        a, sd, _ = faces[f]
        ax.add_patch(Rectangle((x, y), 1, 1, fc=pal[a], alpha=0.35 + 0.35 * (sd > 0), ec='k', lw=0.6))
        ax.text(x + .5, y + .5, "xyz"[a] + ("+" if sd > 0 else "-"), ha='center', va='center', fontsize=6)
    ax.set_aspect('equal')
    ax.autoscale_view()
    xs = [x for x, _ in cells.values()]
    ys = [y for _, y in cells.values()]
    ax.set_xlim(min(xs) - .3, max(xs) + 1.3)
    ax.set_ylim(min(ys) - .3, max(ys) + 1.3)
    ax.axis('off')
    ax.set_title(title, fontsize=8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--survey", type=int, default=0)
    ap.add_argument("--png")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tests-only", action="store_true")
    a = ap.parse_args(argv)
    if a.tests_only:
        return 0 if run_tests() else 1
    faces, edges, fold = surface()
    verts = {v for f in faces for v in _corners(f)}
    print("faces %d  edges %d  vertices %d  Euler %d" % (len(faces), len(edges), len(verts),
                                                         len(verts) - len(edges) + len(faces)))
    print("fold types:", dict(Counter(fold.values())))
    print("spanning trees of the dual graph:", count_spanning_trees(faces, edges))
    if a.survey:
        s = survey(a.survey, a.seed)
        print("samples %d: overlapping %d, valid %d, distinct valid nets %d, translation tilers %d" % (
            s["samples"], s["overlapping"], s["valid"], s["distinct"], s["tilers"]))
        per = Counter(v["perimeter"] for v in s["stats"].values())
        print("perimeter histogram:", dict(sorted(per.items())))
        if a.png:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            best = sorted(s["nets"], key=lambda k: (s["stats"][k]["perimeter"], k))[:8]
            fig, axs = plt.subplots(2, 4, figsize=(12, 6))
            for axx, k in zip(axs.ravel(), best):
                c, t = s["nets"][k]
                draw(axx, c, faces, edges, t, fold, "perimeter %d, %dx%d%s" % (
                    s["stats"][k]["perimeter"], *s["stats"][k]["bbox"],
                    ", tiles" if tiles_by_translation(k) else ""))
            fig.suptitle("Chair44: most compact valid nets (colour = face normal axis, x/y/z)")
            fig.tight_layout()
            fig.savefig(a.png, dpi=130)
            print("wrote", a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
