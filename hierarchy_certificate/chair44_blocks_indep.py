"""Independent recomputation of the Chair44 finite inputs (prepublication review).

Reads only Tsiokos's solid (six-birds-tiles @ d53d224, solid/r44_solid.json; sha256 f320d7a0...) as data.
No code from the repository under review is imported.

Computes:
  1. raw face contacts (touching, non-overlapping lattice poses, proper and improper), mirror ones, and the legal
     ones, where 'legal' means: on every shared unit face the two decorated surfaces coincide, i.e. the sets of
     pyramids (base square, apex point) of the two tiles on that face are equal.  Paper: 2388 / 1194 / 44.
  2. the contacts occurring between touching tiles inside supertiles of every level (paper: 30).
  3. the 168-label cell substitution; primitivity exponent; row/column sums; modular coincidence depth/addresses
     (paper: depth 3, 336 of 512 addresses, among them (0,0,2)).
  4. the legal 2x2x2 blocks: windows of sigma^n(j) for n = 1..6 and the closure (paper: 168, 600, 1278, 1398, 1410,
     1410); the fixed points of the central-window map and their stabilisers (paper: 27; 24 with order 1, 3 with 8).
  5. Lemma lem:blocks: all admissible blocks (8 labels around a vertex naming pairwise disjoint chairs whose
     face-adjacent pairs are legal), their number and O-orbits (paper: 360,189 and 15,214), and whether every one of
     the 27 windows of the substitution of every admissible block is legal (paper: yes).

Usage:  python3 -I chair44_blocks_indep.py ../chair44_data/r44_solid.json      (about 15 minutes)
"""
import sys
import json
import itertools
from fractions import Fraction as Fr
from collections import Counter

D = json.load(open(sys.argv[1]))

# ---------------------------------------------------------------- linear algebra on Z^3
def mat_from(p, s):
    """signed permutation with (g x)_i = s_i x_{p_i}"""
    return tuple(tuple(s[i] if j == p[i] else 0 for j in range(3)) for i in range(3))


def mm(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def mv(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def tr(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def det(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def add(u, v):
    return tuple(x + y for x, y in zip(u, v))


def sub(u, v):
    return tuple(x - y for x, y in zip(u, v))


def sc(c, v):
    return tuple(c * x for x in v)


G48 = [mat_from(p, s) for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)]
ROT = [g for g in G48 if det(g) == 1]
I = mat_from((0, 1, 2), (1, 1, 1))

# ---------------------------------------------------------------- the tile
CUBES = [tuple(c) for c in D['base_unit_cubes']]
C2 = [tuple(2 * x + 1 for x in c) for c in CUBES]        # doubled cube centres (odd)
H = Fr(D['height_unit'])


def cells(g, t):
    """doubled centres of the 7 cubes of the chair in pose x -> g x + t"""
    return frozenset(add(mv(g, c), sc(2, t)) for c in C2)


# pyramids: (frozenset of base corners, apex), coordinates exact, in the tile frame
PYR = []
for p in D['patches']:
    base = frozenset(tuple(Fr(x) for x in b) for b in p['base'])
    apex = tuple(Fr(x) for x in p['apex'])
    PYR.append((base, apex))
assert len(PYR) == 192
assert sum(p['coefficient'] for p in D['patches']) == 0


def face_of(base):
    """(doubled centre of the face, axis) of the unit face containing a pyramid base"""
    pts = list(base)
    axis = [i for i in range(3) if len(set(q[i] for q in pts)) == 1]
    assert len(axis) == 1
    a = axis[0]
    centre = [None] * 3
    centre[a] = 2 * pts[0][a]
    for i in range(3):
        if i != a:
            centre[i] = 2 * (min(q[i] for q in pts) // 1) + 1       # unit face containing the small square
    return tuple(int(x) for x in centre), a


PYR_BY_FACE = {}
for base, apex in PYR:
    PYR_BY_FACE.setdefault(face_of(base), []).append((base, apex))
assert len(PYR_BY_FACE) == 24 and all(len(v) == 8 for v in PYR_BY_FACE.values())


def place(g, t, pt):
    return add(mv(g, pt), t)


def pyramids_on(g, t, face):
    """pyramids of the chair in pose (g,t) lying on the (absolute) face (doubled centre, axis)"""
    # map the face back into the tile frame
    gi = tr(g)
    fc, a = face
    local_c = mv(gi, sub(fc, sc(2, t)))
    la = [i for i in range(3) if gi[i][a] != 0][0]
    out = set()
    for base, apex in PYR_BY_FACE.get((local_c, la), []):
        out.add((frozenset(place(g, t, b) for b in base), place(g, t, apex)))
    return frozenset(out)


DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
REF_CELLS = cells(I, (0, 0, 0))


def shared_faces(A_cells, B_cells):
    res = []
    for c in A_cells:
        for d in DIRS:
            n = add(c, sc(2, d))
            if n in B_cells:
                res.append((add(c, d), [i for i in range(3) if d[i]][0]))
    return res


def classify(g, t):
    B = cells(g, t)
    if B & REF_CELLS:
        return None
    sh = shared_faces(REF_CELLS, B)
    if not sh:
        return None
    legal = all(pyramids_on(I, (0, 0, 0), f) == pyramids_on(g, t, f) for f in sh)
    return legal


RAW = {}
for g in G48:
    for t in itertools.product(range(-5, 6), repeat=3):
        c = classify(g, t)
        if c is not None:
            RAW[(g, t)] = c
mirror = [k for k in RAW if det(k[0]) == -1]
LEGAL = {k for k, v in RAW.items() if v}
print("raw face contacts:", len(RAW), " mirror:", len(mirror), " legal:", len(LEGAL),
      " legal mirror:", sum(1 for k in LEGAL if det(k[0]) == -1))
# range check: no contact at the border of the search box
assert all(max(abs(x) for x in t) <= 4 for g, t in RAW)


def rel(A, B):
    gi = tr(A[0])
    return (mm(gi, B[0]), mv(gi, sub(B[1], A[1])))


def touching_rel(r):
    return r in RAW


# ---------------------------------------------------------------- substitution
KIDS = [(mat_from(*pg), tuple(u)) for pg, u in D['substitution_children']]
# children of the chair in pose (g,t): x -> g (G x + u) + 2t ; check they tile the doubled chair
dbl = Counter()
for G, u in KIDS:
    for c in cells(G, tuple(u)):
        dbl[c] += 1
target = set()
for c in CUBES:
    for b in itertools.product((0, 1), repeat=3):
        target.add(tuple(2 * (2 * c[i] + b[i]) + 1 for i in range(3)))
assert set(dbl) == target and max(dbl.values()) == 1, "children do not tile the doubled chair"
# u is in unit coordinates: child i is the pose (G_i, u_i) inside the doubled chair [0,4)^3.


def children(P):
    g, t = P
    out = []
    for G, u in KIDS:
        # child cells (unit coords doubled): g (G c2 + u) + 2*(2t)
        out.append((mm(g, G), add(mv(g, u), sc(2, t))))     # x -> g (G x + u) + 2t
    return out


def norm_pose(P):
    g, t = P
    assert all(Fr(x).denominator == 1 for x in t), P
    return (g, tuple(int(x) for x in t))


def children_int(P):
    return [norm_pose(c) for c in children(P)]


# contacts inside supertiles of every level: least fixed point
REFK = children_int((I, (0, 0, 0)))
P1 = {rel(a, b) for a in REFK for b in REFK if a != b and touching_rel(rel(a, b))}
P0 = set(P1)
while True:
    new = set(P0)
    for (g, t) in P0:
        other = children_int((g, t))       # supertile in pose (g, 2t) relative: children of chair (g,t), doubled
        for a in REFK:
            for b in other:
                r = rel(a, b)
                if touching_rel(r):
                    new.add(r)
    if new == P0:
        break
    P0 = new
print("hierarchical contacts: |P1| =", len(P1), " |P0| =", len(P0), " P0 subset of legal:", P0 <= LEGAL)

# labels and the cell substitution
LAB = [(g, ci) for g in ROT for ci in range(7)]
LID = {l: i for i, l in enumerate(LAB)}
BITS = list(itertools.product((0, 1), repeat=3))


def pose_of_label(j, cellc2):
    """pose of the chair whose cube ci (label j) has doubled centre cellc2"""
    g, ci = LAB[j]
    t2 = sub(cellc2, mv(g, C2[ci]))
    assert all(x % 2 == 0 for x in t2)
    return (g, tuple(x // 2 for x in t2))


def label_at(P, cellc2):
    g, t = P
    loc = mv(tr(g), sub(cellc2, sc(2, t)))
    if loc in C2:
        return LID[(g, C2.index(loc))]
    return None


SUB = []
for j in range(168):
    X = (1, 1, 1)                                   # the cell [0,1]^3, doubled centre (1,1,1)
    P = pose_of_label(j, X)
    kids = children_int(P)
    row = []
    for b in BITS:
        # after inflation by 2 the cell [0,1]^3 becomes [0,2]^3; its sub-cell b has doubled centre 2b+1
        Y = tuple(2 * b[i] + 1 for i in range(3))
        hit = [label_at(k, Y) for k in kids]
        hit = [h for h in hit if h is not None]
        assert len(hit) == 1
        row.append(hit[0])
    SUB.append(tuple(row))
# The cell [0,1]^3 of chair P inflates to the cube [0,2]^3 of the supertile 2P; its 8 sub-cells are labelled above.
colsum = Counter(x for row in SUB for x in row)
print("cell substitution: every label appears %s times as a child (row sums of the 168x168 matrix); each label has 8 children"
      % sorted(set(colsum.values())))
import numpy as np
Mx = np.zeros((168, 168), dtype=np.int64)
for j, row in enumerate(SUB):
    for x in row:
        Mx[x, j] += 1
Pn = np.eye(168, dtype=np.int64)
for n in range(1, 6):
    Pn = (Pn @ Mx > 0).astype(np.int64)
    if Pn.min() > 0:
        print("primitive: all entries of M^%d positive" % n)
        break


# modular coincidence: addresses (digit words) of length m such that all 168 labels map to one label
def descend(j, word):
    for b in word:
        j = SUB[j][BITS.index(b)]
    return j


for m in range(1, 4):
    good = []
    for word in itertools.product(BITS, repeat=m):
        if len({descend(j, word) for j in range(168)}) == 1:
            good.append(word)
    print("depth %d: %d of %d addresses coincide" % (m, len(good), 8 ** m))
    if good:
        # address of the word as a point of Z^3 mod 2^m: most significant digit first (top level first)
        addrs = set()
        for w in good:
            a = [0, 0, 0]
            for b in w:
                a = [2 * a[i] + b[i] for i in range(3)]
            addrs.add(tuple(a))
        print("   (0,0,2) among them:", (0, 0, 2) in addrs, "; also with reversed digit order:",
              any(tuple(sum(w[k][i] << k for k in range(m)) for i in range(3)) == (0, 0, 2) for w in good))
        break


# ---------------------------------------------------------------- legal blocks
def sigma_block(blk, k):
    """blk: dict (x,y,z) in [0,k)^3 -> label; returns dict on [0,2k)^3"""
    out = {}
    for (x, y, z), j in blk.items():
        for bi, b in enumerate(BITS):
            out[(2 * x + b[0], 2 * y + b[1], 2 * z + b[2])] = SUB[j][bi]
    return out


def windows(blk, k):
    return [tuple(blk[(o[0] + b[0], o[1] + b[1], o[2] + b[2])] for b in BITS) for o in itertools.product(range(k - 1), repeat=3)]


seen = set()
counts = []
for n in range(1, 7):
    for j in range(168):
        blk, k = {(0, 0, 0): j}, 1
        for _ in range(n):
            blk, k = sigma_block(blk, k), 2 * k
        seen.update(windows(blk, k))
    counts.append(len(seen))
print("distinct windows of sigma^n(j), cumulative n=1..6:", counts)


def child_windows(w):
    return windows(sigma_block({b: w[i] for i, b in enumerate(BITS)}, 2), 4)


closed = all(set(child_windows(w)) <= seen for w in seen)
print("closed under child windows:", closed)
LEGALB = seen


def central(w):
    e = sigma_block({b: w[i] for i, b in enumerate(BITS)}, 2)
    return tuple(e[(1 + b[0], 1 + b[1], 1 + b[2])] for b in BITS)


assert all(central(w) in LEGALB for w in LEGALB)
per = [w for w in LEGALB if central(w) == w]
# periodic points of the central map
cyc = set()
for w in LEGALB:
    x = w
    for _ in range(len(LEGALB) + 1):
        x = central(x)
    y, orb = x, [x]
    y = central(y)
    while y != x:
        orb.append(y)
        y = central(y)
    cyc.update(orb)
print("central map: %d periodic blocks, %d fixed" % (len(cyc), len(per)))


def act(R, w):
    """rotate a block about its vertex: cell b at doubled offset 2b-1; labels get frame R g"""
    out = [None] * 8
    for i, b in enumerate(BITS):
        c = mv(R, tuple(2 * x - 1 for x in b))
        b2 = tuple((x + 1) // 2 for x in c)
        g, ci = LAB[w[i]]
        out[BITS.index(b2)] = LID[(mm(R, g), ci)]
    return tuple(out)


stabs = Counter(sum(1 for R in ROT if act(R, w) == w) for w in per)
print("stabiliser orders of the fixed blocks:", dict(stabs))

# ---------------------------------------------------------------- admissible blocks (Lemma lem:blocks)
POS = [tuple(2 * x - 1 for x in b) for b in BITS]          # doubled centres of the 8 cells around the origin
CAND = [[pose_of_label(j, X) for j in range(168)] for X in POS]
CELLC = {}


def pcells(P):
    c = CELLC.get(P)
    if c is None:
        c = CELLC[P] = cells(*P)
    return c


def ok_pair(A, B):
    if A == B:
        return True
    if pcells(A) & pcells(B):
        return False
    r = rel(A, B)
    if r in RAW:
        return r in LEGAL
    return True


# pairwise compatibility table between (position, label) pairs, then a bitset search
COMPAT = [[0] * 168 for _ in range(8)]      # COMPAT[i][a] = bitmask over (j, b) with j > i compatible with (i, a)
for i in range(8):
    for a in range(168):
        P = CAND[i][a]
        m = 0
        for j in range(i + 1, 8):
            for b in range(168):
                if ok_pair(P, CAND[j][b]):
                    m |= 1 << (j * 168 + b)
        COMPAT[i][a] = m
FULL = (1 << (8 * 168)) - 1
admissible = []


def rec(i, mask, labels):
    if i == 8:
        admissible.append(tuple(labels))
        return
    seg = (mask >> (i * 168)) & ((1 << 168) - 1)
    while seg:
        low = seg & -seg
        a = low.bit_length() - 1
        seg ^= low
        rec(i + 1, mask & COMPAT[i][a], labels + [a])


rec(0, FULL, [])
print("admissible blocks:", len(admissible))
orbits = set()
for w in admissible:
    orbits.add(min(act(R, w) for R in ROT))
print("O-orbits:", len(orbits))
bad = 0
illegal_adm = 0
aset = set(admissible)
for w in admissible:
    if any(c not in LEGALB for c in child_windows(w)):
        bad += 1
print("admissible blocks with an illegal window in their substitution:", bad)
print("legal blocks that are admissible:", sum(1 for w in LEGALB if w in aset), "of", len(LEGALB))
