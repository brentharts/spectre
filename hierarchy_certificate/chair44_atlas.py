"""Referee's independent Chair44 atlas, cell substitution and legal-block language.

Input: Tsiokos's published solid (six-birds-tiles @ d53d224, solid/r44_solid.json), read as data only:
base cubes, the 192 pyramid patches (foot point, outward normal, signed coefficient) and the eight
substitution children.  Everything else is derived here.

Conventions: a pose is (g, t), x -> g x + t, g a signed permutation with g(x)_i = s_i x_{p_i}.
Cells are unit cubes; we store doubled centres (2*centre, integer, odd coordinates).
"""
import json
import sys
import itertools
from fractions import Fraction as Fr
from collections import defaultdict, Counter

SOLID = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.json') else \
    __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),
                               '..', 'chair44_data', 'r44_solid.json')
D = json.load(open(SOLID))

# ---------------- signed permutations ----------------
def gmat(p, s):
    return tuple(tuple(s[i] if j == p[i] else 0 for j in range(3)) for i in range(3))


ALL48 = [gmat(p, s) for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)]


def det(g):
    a = g
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


O24 = [g for g in ALL48 if det(g) == 1]
assert len(O24) == 24


def gm(g, h):
    return tuple(tuple(sum(g[i][k] * h[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def ginv(g):
    return tuple(tuple(g[j][i] for j in range(3)) for i in range(3))


def gv(g, v):
    return tuple(sum(g[i][k] * v[k] for k in range(3)) for i in range(3))


def vadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def vsub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def vsc(c, a):
    return tuple(c * x for x in a)


I3 = gmat((0, 1, 2), (1, 1, 1))

# ---------------- the chair ----------------
CELLS = [tuple(c) for c in D['base_unit_cubes']]            # corners
assert len(CELLS) == 7
CC2 = [tuple(2 * x + 1 for x in c) for c in CELLS]          # doubled centres
CSET = set(CC2)
DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
FACES = []   # (doubled centre of cell, outward dir)
for c in CC2:
    for d in DIRS:
        if vadd(c, vsc(2, d)) not in CSET:
            FACES.append((c, d))
assert len(FACES) == 24

# pyramids: foot point (in units of 1, exact) -> (outward normal, coefficient); 4x scaled ints for hashing
PYR = []
for p in D['patches']:
    base = [tuple(Fr(x) for x in b) for b in p['base']]
    foot = tuple(sum(b[i] for b in base) / 4 for i in range(3))
    apex = tuple(Fr(x) for x in p['apex'])
    disp = vsub(apex, foot)
    n = tuple(0 if x == 0 else (1 if x > 0 else -1) for x in disp)
    coef = p['coefficient']
    # outward displacement = coef * height_unit along outward normal; so the outward normal is sign(coef)*n
    out = vsc(1 if coef > 0 else -1, n)
    assert abs(sum(disp)) == abs(coef) * Fr(D['height_unit'])
    PYR.append((foot, out, coef))
assert len(PYR) == 192
assert sum(c for _, _, c in PYR) == 0
# every pyramid lies on an exterior unit face with its outward normal
for foot, out, coef in PYR:
    cell2 = tuple(int(2 * (foot[i] - Fr(out[i], 2))) if out[i] else int(2 * (foot[i] // 1) + 1) for i in range(3))
    assert (cell2, out) in FACES, (foot, out)


def pose_cells(g, t):
    """doubled centres of the cells of the chair in pose (g,t)"""
    t2 = vsc(2, t)
    return frozenset(vadd(gv(g, c), t2) for c in CC2)


def pose_pyramids(g, t):
    res = {}
    for foot, out, coef in PYR:
        f = tuple(Fr(x) for x in vadd(gv(g, foot), t))
        res[f] = (gv(g, out), coef)
    return res


def contact_type(g, t):
    """relative pose of chair B=(g,t) with respect to the root chair A=(I,0)."""
    A = pose_cells(I3, (0, 0, 0))
    B = pose_cells(g, t)
    if A & B:
        return 'same' if (g == I3 and t == (0, 0, 0)) else 'overlap'
    shared = []
    for c in A:
        for d in DIRS:
            if vadd(c, vsc(2, d)) in B:
                shared.append((c, d))
    if not shared:
        return 'apart'
    PA = pose_pyramids(I3, (0, 0, 0))
    PB = pose_pyramids(g, t)
    ok = True
    for c, d in shared:
        # pyramids on this unit face of A: feet at plane c/2 + d/2
        for f, (out, coef) in PA.items():
            # foot on that face?
            centre = tuple(Fr(x, 2) for x in c)
            rel = vsub(f, centre)
            if out == d and all(rel[i] == Fr(d[i], 2) for i in range(3) if d[i]) and \
                    all(abs(rel[i]) < Fr(1, 2) for i in range(3) if not d[i]):
                if f not in PB:
                    ok = False
                    break
                outB, coefB = PB[f]
                assert outB == vsc(-1, d)
                if coef + coefB != 0:
                    ok = False
                    break
        if not ok:
            break
    return ('legal' if ok else 'illegal') + ('' if det(g) == 1 else '-mirror')


def all_contacts():
    res = {}
    for g in ALL48:
        for t in itertools.product(range(-4, 5), repeat=3):
            ct = contact_type(g, t)
            if ct not in ('apart', 'overlap', 'same'):
                res[(g, t)] = ct
    return res


# ---------------- substitution ----------------
CHILDREN = []
for (p, s), u in D['substitution_children']:
    CHILDREN.append((gmat(p, s), tuple(u)))
# Check: the eight children tile the doubled chair
dbl = set()
for c in CC2:
    for b in itertools.product((0, 1), repeat=3):
        dbl.add(tuple(2 * (2 * ((c[i] - 1) // 2) + b[i]) + 1 for i in range(3)))
cover = Counter()
for g, u in CHILDREN:
    for c in pose_cells(g, u):
        cover[c] += 1
assert set(cover) == dbl and max(cover.values()) == 1, "children do not tile the doubled chair"

LABELS = [(g, ci) for g in O24 for ci in range(7)]
LIDX = {l: i for i, l in enumerate(LABELS)}


def subst_table():
    """SUB[j][b] = label of child cell b in {0,1}^3 of a unit cell of label j (cell-local coordinates aligned
    with the ambient axes: the unit cell at doubled centre X becomes the 8 cells at 2X + (2b-1))."""
    SUB = []
    for (g, ci) in LABELS:
        # chair in pose (g, t) with cell ci at doubled centre X; supertile is pose (g, 2t) of the doubled chair
        # children: (g G_i, 2t + g u_i).  Use t = 0: cell ci at doubled centre g CC2[ci].
        X = gv(g, CC2[ci])
        row = {}
        kids = [(gm(g, G), gv(g, u)) for G, u in CHILDREN]
        for b in itertools.product((0, 1), repeat=3):
            Y = tuple(2 * X[i] + (2 * b[i] - 1) for i in range(3))     # doubled centre of child cell
            hit = None
            for kg, kt in kids:
                # which local cell of child (kg,kt) has doubled centre Y?
                loc = gv(ginv(kg), vsub(Y, vsc(2, kt)))
                if loc in CSET:
                    hit = (kg, CC2.index(loc))
            assert hit is not None
            row[b] = LIDX[hit]
        SUB.append(row)
    return SUB


SUB = subst_table()
BITS = list(itertools.product((0, 1), repeat=3))


def expand(block, k):
    """block: dict (x,y,z) in [0,k)^3 -> label; returns dict on [0,2k)^3"""
    out = {}
    for (x, y, z), j in block.items():
        for b in BITS:
            out[(2 * x + b[0], 2 * y + b[1], 2 * z + b[2])] = SUB[j][b]
    return out


def windows(block, k):
    res = []
    for o in itertools.product(range(k - 1), repeat=3):
        res.append(tuple(block[(o[0] + b[0], o[1] + b[1], o[2] + b[2])] for b in BITS))
    return res


def legal_blocks(maxn=4, verbose=True):
    """All 2x2x2 windows of sigma^n(j), any j, any n >= 1.

    Every window of sigma^(n+1)(j) lies in sigma(w) for a window w of sigma^n(j) (its parent cells span
    at most two cells per axis), so the language is the closure of the windows of sigma(j) under
    'children windows'.  We also print direct counts for n <= maxn as a cross-check."""
    seen = set()
    for n in range(1, maxn + 1):
        for j in range(168):
            blk = {(0, 0, 0): j}
            k = 1
            for _ in range(n):
                blk = expand(blk, k)
                k *= 2
            for w in windows(blk, k):
                seen.add(w)
        if verbose:
            print("  direct: windows of sigma^n(j), n<=%d:" % n, len(seen))
    frontier = list(seen)
    while frontier:
        new = []
        for w in frontier:
            for c in children_windows(w):
                if c not in seen:
                    seen.add(c)
                    new.append(c)
        frontier = new
    if verbose:
        print("  closure:", len(seen))
    return seen


def children_windows(w):
    blk = {b: w[i] for i, b in enumerate(BITS)}
    return windows(expand(blk, 2), 4)


def central(w):
    blk = {b: w[i] for i, b in enumerate(BITS)}
    e = expand(blk, 2)
    return tuple(e[(1 + b[0], 1 + b[1], 1 + b[2])] for b in BITS)


# action of O on blocks about the vertex: rotate positions (cells at corners b-1/2 ... ) and labels
def act_label(R, j):
    g, ci = LABELS[j]
    return LIDX[(gm(R, g), ci)]


def act_block(R, w):
    # cell b in {0,1}^3 has doubled centre (2b-1) relative to the vertex
    out = [None] * 8
    for i, b in enumerate(BITS):
        c = tuple(2 * x - 1 for x in b)
        c2 = gv(R, c)
        b2 = tuple((x + 1) // 2 for x in c2)
        out[BITS.index(b2)] = act_label(R, w[i])
    return tuple(out)
