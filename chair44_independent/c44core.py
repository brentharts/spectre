"""Independent Chair44 core (written from the paper's definitions + solid/r44_solid.json geometry only).

Conventions (Tsiokos, PROOFS_registered §1): frame (p,s) means G(x)_i = s_i x_{p_i};
pose (G,t) maps x -> Gx + t; refinement (G,t) -> {(GH, 2t+Gu) : (H,u) in C}.
Everything is integer; geometry is scaled by S=20000.
"""
import itertools, json
from fractions import Fraction as F

S = 20000
CELLS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if (a, b, c) != (1, 1, 1)]
CHILDREN = [((0,1,2),(1,1,1),(0,0,0)), ((1,0,2),(1,1,-1),(0,0,4)), ((0,2,1),(1,-1,1),(0,4,0)),
            ((2,0,1),(1,-1,-1),(0,4,4)), ((2,1,0),(-1,1,1),(4,0,0)), ((1,2,0),(-1,1,-1),(4,0,4)),
            ((0,1,2),(-1,-1,1),(4,4,0)), ((0,1,2),(1,1,1),(1,1,1))]


def mat(p, s):
    return tuple(tuple(s[i] if j == p[i] else 0 for j in range(3)) for i in range(3))

def mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))

def app(A, x):
    return tuple(sum(A[i][k] * x[k] for k in range(3)) for i in range(3))

def add(x, y): return tuple(a + b for a, b in zip(x, y))
def sub(x, y): return tuple(a - b for a, b in zip(x, y))
def neg(x): return tuple(-a for a in x)
def scal(k, x): return tuple(k * a for a in x)

def det(A):
    return (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
            + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))

def transpose(A): return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))

I3 = mat((0, 1, 2), (1, 1, 1))
ALL48 = [mat(p, s) for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)]
ROT24 = [A for A in ALL48 if det(A) == 1]
RIDX = {A: i for i, A in enumerate(ROT24)}

def cell_image(A, t, c):
    """min corner of image of unit cube with min corner c under x->Ax+t"""
    lo = tuple(sum(min(0, A[i][k]) for k in range(3)) for i in range(3))
    return add(add(app(A, c), t), lo)

def pose_cells(A, t):
    return [cell_image(A, t, c) for c in CELLS]

CHILD_POSES = [(mat(p, s), u) for p, s, u in CHILDREN]

def children(A, t):
    return [(mul(A, H), add(scal(2, t), app(A, u))) for H, u in CHILD_POSES]

def compose(P1, P2):
    """(A1,t1)∘(A2,t2)"""
    (A1, t1), (A2, t2) = P1, P2
    return (mul(A1, A2), add(app(A1, t2), t1))

def inverse(P):
    A, t = P
    Ai = transpose(A)
    return (Ai, neg(app(Ai, t)))


def load_features(path):
    """Return dict: (panel key) -> list of (point_S, apex_S), from the solid's 'patches' (apex + base square).
    Panel key = (axis, doubled centre) in body coords. Outward normal recomputed from P."""
    d = json.load(open(path))
    assert [list(c) for c in CELLS] == d['base_unit_cubes'] or sorted(map(tuple, d['base_unit_cubes'])) == sorted(CELLS)
    cellset = set(CELLS)
    feats = {}
    for ptc in d['patches']:
        base = [tuple(F(v) for v in q) for q in ptc['base']]
        apex = tuple(F(v) for v in ptc['apex'])
        cen = tuple(sum(q[i] for q in base) / 4 for i in range(3))
        ax = [i for i in range(3) if len({q[i] for q in base}) == 1]
        assert len(ax) == 1
        ax = ax[0]
        # panel: unit square containing cen in plane x_ax = cen[ax]
        pc = list(cen); key = tuple(int(2 * (int(cen[i]) + F(1, 2))) if i != ax else int(2 * cen[i]) for i in range(3))
        # outward normal: which side is inside P
        lo = tuple(int(cen[i]) if i != ax else int(cen[ax]) - 1 for i in range(3))
        hi = tuple(int(cen[i]) if i != ax else int(cen[ax]) for i in range(3))
        inlo, inhi = lo in cellset, hi in cellset
        assert inlo != inhi, ptc
        nsign = 1 if inlo else -1
        h = (apex[ax] - cen[ax]) * nsign  # signed height along outward normal
        assert all(apex[i] == cen[i] for i in range(3) if i != ax)
        assert h * 10000 == nsign * 0 + h * 10000 and abs(h * 10000) == abs(ptc['coefficient']) and (h > 0) == (ptc['coefficient'] > 0), (h, ptc)
        feats.setdefault((ax, key), []).append((tuple(int(c * S) for c in cen), tuple(int(c * S) for c in apex)))
    return feats


def exposed_panels():
    cs = set(CELLS); out = []
    for c in CELLS:
        for ax in range(3):
            for d in (-1, 1):
                nb = list(c); nb[ax] += d; nb = tuple(nb)
                if nb not in cs:
                    out.append((c, ax, d, nb))
    return out


class Geometry:
    def __init__(self, solid_path):
        self.feats = load_features(solid_path)
        self.panels = exposed_panels()
        assert len(self.panels) == 24 and len(self.feats) == 24 and all(len(v) == 8 for v in self.feats.values())
        # map world unit face -> apex set, for a pose
    def face_key(self, cell, ax, d):
        c = list(cell); 
        if d == 1: c[ax] += 1
        return (ax, tuple(c))  # face = plane x_ax=c[ax], unit square with min corner c
    def apexes(self, A, t):
        """dict world face key -> frozenset of world apex points (scaled)"""
        out = {}
        tS = scal(S, t)
        for (ax, key), lst in self.feats.items():
            pts = frozenset(add(app(A, ap), tS) for _, ap in lst)
            cen = add(app(A, lst[0][0]), tS)
            # world face key from centre
            wax = [i for i in range(3) if A[i][ax] != 0][0]
            mc = tuple((cen[i] // S) if i != wax else cen[i] // S for i in range(3))
            assert cen[wax] % S == 0
            out[(wax, mc)] = pts
        return out
    def fits(self, P1, P2):
        """True iff poses P1,P2 have disjoint carriers and features agree on every shared unit face.
        Returns (overlap?, nshared, ok)"""
        c1, c2 = set(pose_cells(*P1)), set(pose_cells(*P2))
        if c1 & c2: return False, 0, False
        a1, a2 = self.apexes(*P1), self.apexes(*P2)
        shared = set(a1) & set(a2)
        ok = all(a1[f] == a2[f] for f in shared)
        return True, len(shared), ok


def compute_atlas(geo, rots=ALL48):
    cs = set(CELLS)
    shell = sorted({add(c, d) for c in CELLS for d in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]} - cs)
    cand = set()
    for A in rots:
        img = pose_cells(A, (0, 0, 0))
        for s in shell:
            for ic in img:
                t = sub(s, ic)
                if not (set(pose_cells(A, t)) & cs):
                    cand.add((A, t))
    legal = []
    root = (I3, (0, 0, 0))
    for P in sorted(cand):
        nonov, nsh, ok = geo.fits(root, P)
        assert nonov and nsh > 0
        if ok: legal.append(P)
    return shell, cand, legal
