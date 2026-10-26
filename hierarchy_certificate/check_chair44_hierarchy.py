"""Referee check of input (H): Kamenetsky's finite certificate (Prop. 7.1, conditions C1-C3) for the Chair44.

Independent of Kamenetsky's cert6.py and of the authors' reproduction.  Built on chair44_atlas.py, which reads
only Tsiokos's published solid as data.

  P1      relative poses of touching children inside one supertile
  P0      least fixed point P0 = P1 u X(P0), X(p) = child contacts of two supertiles in relative pose (g, 2t)
  Sigma(P) supertile poses (G, T) touching the reference supertile with all 64 child pairs interior-disjoint and
          every touching child pair in P
  C1      P* = closure of P0 under halving aligned poses of Sigma, no offset pose ever admissible, Sigma(P*) = 2P*
  C2      facet rules F* = union of F(p), p in P*; the touching poses whose facet triples all lie in F* are exactly P*
  C3      enclosures of a tile (pairwise compatible covers of its 24 panels by P*-neighbours); dead/live by an
          exhaustive second-shell search; (U) exactly one compatible level-1 supertile contains the tile;
          (Pr) every compatible enclosure of that supertile's central child completes the supertile.
Also: P* equals the 44 legal (pyramid-matching) contacts of the paper's atlas.

Usage: python3 check_chair44_hierarchy.py [path/to/r44_solid.json]
"""
import itertools
import sys
from collections import Counter
from chair44_atlas import (ALL48, O24, I3, CHILDREN, CC2, FACES, DIRS, gm, ginv, gv, vadd, vsub, vsc, det,
                           pose_cells, all_contacts)

# ---------- poses ----------
def compose(A, B):          # pose of B (given relative to A) in absolute terms; A=(g,t)
    g, t = A
    h, s = B
    return (gm(g, h), vadd(gv(g, s), t))


def rel(A, B):
    gi = ginv(A[0])
    return (gm(gi, B[0]), gv(gi, vsub(B[1], A[1])))


CELLS_OF = {}


def cells(P):
    c = CELLS_OF.get(P)
    if c is None:
        c = CELLS_OF[P] = pose_cells(*P)
    return c


CT = all_contacts()                                   # (g, t) -> 'legal' / 'illegal' / ...-mirror, touching only
TOUCH = set(CT)
LEGAL = {p for p, c in CT.items() if c == 'legal'}
print("touching lattice-registered poses:", len(TOUCH), " legal:", len(LEGAL))
ID = (I3, (0, 0, 0))


def overlap(A, B):
    return bool(cells(A) & cells(B))


def touching(A, B):
    return rel(A, B) in TOUCH


def compatible(A, B, P):
    """interior-disjoint, and if touching then the relative pose lies in P"""
    if A == B:
        return True
    if overlap(A, B):
        return False
    r = rel(A, B)
    return r not in TOUCH or r in P


def children(S):
    return [compose(S, c) for c in CHILDREN]


CENTRAL = [i for i, (g, u) in enumerate(CHILDREN) if g == I3 and u == (1, 1, 1)]   # Kamenetsky's T0
assert len(CENTRAL) == 1
CENTRAL = CENTRAL[0]
print("central child (frame I):", CHILDREN[CENTRAL])

# ---------- C1 ----------
REF = children(ID)
P1 = {rel(a, b) for a in REF for b in REF if a != b and touching(a, b)}


def X(p):
    g, t = p
    S2 = (g, vsc(2, t))
    return {rel(a, b) for a in REF for b in children(S2) if touching(a, b)}


P0 = set(P1)
while True:
    new = set(P0)
    for p in P0:
        new |= X(p)
    if new == P0:
        break
    P0 = new
print("|P1| =", len(P1), " |P0| =", len(P0))


def Sigma(P):
    out = set()
    for a in REF:
        for p in P:
            b = compose(a, p)
            for (gi, ui) in CHILDREN:
                G = gm(b[0], ginv(gi))
                T = vsub(b[1], gv(G, ui))
                S = (G, T)
                if S in out or S == ID:
                    continue
                ch = children(S)
                if all(compatible(x, y, P) for x in REF for y in ch):
                    out.add(S)
    return out


P = set(P0)
while True:
    Sg = Sigma(P)
    offset = [S for S in Sg if any(x % 2 for x in S[1])]
    assert not offset, "an offset supertile pose is admissible"
    newP = P | {(G, tuple(x // 2 for x in T)) for G, T in Sg}
    if newP == P:
        break
    P = newP
PSTAR = P
assert Sigma(PSTAR) == {(g, vsc(2, t)) for g, t in PSTAR}
print("C1: |P*| =", len(PSTAR), ", no offset pose admissible, Sigma(P*) = 2P*  -- holds")
assert PSTAR == LEGAL
print("    P* is exactly the set of 44 legal (pyramid-matching) contacts")

# ---------- C2 ----------
FIDX = {f: i for i, f in enumerate(FACES)}


def facets(p):
    g, t = p
    tr = set()
    for (c, d) in FACES:
        nb = vadd(c, vsc(2, d))
        for b, (c2, d2) in enumerate(FACES):
            if vadd(gv(g, c2), vsc(2, t)) == nb and gv(g, d2) == vsc(-1, d):
                tr.add((FIDX[(c, d)], b, g))
    return tr


FSTAR = set()
for p in PSTAR:
    FSTAR |= facets(p)
PF = {p for p in TOUCH if facets(p) <= FSTAR}
assert PF == PSTAR
print("C2: |F*| =", len(FSTAR), "facet triples; the touching poses obeying F* are exactly P*  -- holds")

# ---------- C3 ----------
def covering_tiles(tile, face):
    """P*-neighbours of `tile` that cover its panel `face` (contain the cell across it)"""
    c, d = face
    g, t = tile
    nb = vadd(vadd(gv(g, c), vsc(2, t)), vsc(2, gv(g, d)))
    out = []
    for p in PSTAR:
        q = compose(tile, p)
        if nb in cells(q):
            out.append(q)
    return out


def panel_covered(tile, face, tiles):
    c, d = face
    g, t = tile
    nb = vadd(vadd(gv(g, c), vsc(2, t)), vsc(2, gv(g, d)))
    return any(nb in cells(q) for q in tiles)


def enclosures(tile, patch=()):
    """all pairwise P*-compatible covers of the panels of `tile`, compatible with the tiles of `patch`"""
    res = []
    base = [q for q in patch]

    def rec(chosen):
        for face in FACES:
            if not panel_covered(tile, face, chosen + base):
                for q in covering_tiles(tile, face):
                    if all(compatible(q, x, PSTAR) for x in chosen + base + [tile]):
                        rec(chosen + [q])
                return
        res.append(frozenset(q for q in chosen + base if q != tile and touching(tile, q)))
    rec([])
    return res


NODES = [0]


def second_shell(patch, limit=10 ** 6):
    """True iff the panels of every tile of `patch` can be covered by pairwise P*-compatible tiles"""
    patch = list(patch)

    def rec(tiles, inner):
        NODES[0] += 1
        assert NODES[0] < limit
        for T in inner:
            for face in FACES:
                if not panel_covered(T, face, tiles):
                    for q in covering_tiles(T, face):
                        if all(compatible(q, x, PSTAR) for x in tiles):
                            if rec(tiles + [q], inner):
                                return True
                    return False
        return True
    return rec(patch, patch)


ENC = enclosures(ID)
print("C3: enclosures of a tile:", len(ENC))
live = []
for E in ENC:
    NODES[0] = 0
    ok = second_shell([ID] + sorted(E))
    if ok:
        live.append(E)
print("    dead / live:", len(ENC) - len(live), "/", len(live))


def compatible_supertile(S, patch):
    pset = set(patch)
    for c in children(S):
        if c in pset:
            continue
        if not all(compatible(c, x, PSTAR) for x in patch):
            return False
    for a, b in itertools.combinations(children(S), 2):
        assert compatible(a, b, PSTAR)
    return True


ucount, central_cases, pr_ok = 0, 0, 0
for E in live:
    patch = [ID] + sorted(E)
    cands = []
    for (gi, ui) in CHILDREN:
        G = ginv(gi)
        S = (G, vsub((0, 0, 0), gv(G, ui)))
        if compatible_supertile(S, patch):
            cands.append(S)
    assert len(cands) == 1, ("(U) fails", len(cands))
    ucount += 1
    S = cands[0]
    ch = children(S)
    T0 = ch[CENTRAL]
    pset = set(patch)
    if T0 == ID:
        assert all(c in pset for c in ch)
        central_cases += 1
        continue
    assert T0 in pset            # the central child touches every other child, so it lies in the enclosure
    for E2 in enclosures(T0, patch):
        assert all(c in pset or c in E2 for c in ch), "(Pr) fails"
    pr_ok += 1
print("    (U): exactly one compatible supertile for %d of %d live enclosures" % (ucount, len(live)))
print("    (Pr): %d central-child cases complete already; %d other cases complete for every compatible "
      "enclosure of the central child" % (central_cases, pr_ok))
print("C1-C3 hold: every lattice-registered P*-tiling (atlas tiling) is uniquely hierarchical (Prop. 7.1)")
