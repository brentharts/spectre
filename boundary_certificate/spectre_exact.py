"""Independent exact construction of the Spectre substitution patches.

Written by the referee from the published construction of Smith, Myers, Kaplan,
Goodman-Strauss (arXiv:2305.17743; Kaplan's reference implementation), without
reading the authors' code.  Every point is an element of Z[zeta_12], stored as an
integer vector (p, q, r, s) meaning p + q*w + z*(r + s*w) with w = e^{i pi/3},
z = e^{i pi/6}.  All transforms are integer affine maps on Z^4, so all geometry is
exact.
"""
import cmath
import numpy as np

W = cmath.exp(1j * cmath.pi / 3)
Z = cmath.exp(1j * cmath.pi / 6)

# multiplication by z on the basis (1, w, z, zw):  1->z, w->zw, z->w, zw->w^2 = w-1
MZ = np.array([[0, 0, 0, -1],
               [0, 0, 1, 1],
               [1, 0, 0, 0],
               [0, 1, 0, 0]], dtype=object)
# complex conjugation: 1->1, w->1-w, z->z-zw, zw->-zw
CONJ = np.array([[1, 1, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, -1, -1]], dtype=object)
I4 = np.eye(4, dtype=object)
for i in range(4):
    for j in range(4):
        I4[i, j] = int(I4[i, j])


def zpow(k):
    m = I4.copy()
    for _ in range(k % 12):
        m = MZ.dot(m)
    return m


def to_c(v):
    p, q, r, s = (int(x) for x in v)
    return p + q * W + Z * (r + s * W)


def vec(*c):
    return np.array([int(x) for x in c], dtype=object)


# a transform is (A, t): x -> A x + t
def mul(T1, T2):
    A1, t1 = T1
    A2, t2 = T2
    return (A1.dot(A2), A1.dot(t2) + t1)


def apply(T, v):
    return T[0].dot(v) + T[1]


IDENT = (I4.copy(), vec(0, 0, 0, 0))


def trot(k):          # rotation by k*30 degrees
    return (zpow(k), vec(0, 0, 0, 0))


def ttrans(v):
    return (I4.copy(), v)


def trans_to(p, q):
    return ttrans(q - p)


# Kaplan's R = [-1,0,0, 0,1,0]: (x,y) -> (-x,y) = -conj
R = (-CONJ, vec(0, 0, 0, 0))

# Spectre outline Tile(1,1): edge directions in units of 30 degrees
EDGE_DIRS = [0, 10, 1, 3, 0, 2, 5, 7, 4, 6, 6, 8, 11, 9]
SPECTRE = [vec(0, 0, 0, 0)]
for k in EDGE_DIRS[:-1]:
    SPECTRE.append(SPECTRE[-1] + zpow(k).dot(vec(1, 0, 0, 0)))
assert all(x == 0 for x in SPECTRE[-1] + zpow(EDGE_DIRS[-1]).dot(vec(1, 0, 0, 0)))

# Kaplan's floating point outline, for a check of the transcription
KAPLAN = [(0, 0), (1.0, 0.0), (1.5, -0.8660254037844386), (2.366025403784439, -0.36602540378443865),
          (2.366025403784439, 0.6339745962155614), (3.366025403784439, 0.6339745962155614),
          (3.866025403784439, 1.5), (3.0, 2.0), (2.133974596215561, 1.5),
          (1.6339745962155614, 2.3660254037844393), (0.6339745962155614, 2.3660254037844393),
          (-0.3660254037844386, 2.3660254037844393), (-0.866025403784439, 1.5), (0.0, 1.0)]
for v, (x, y) in zip(SPECTRE, KAPLAN):
    assert abs(to_c(v) - complex(x, y)) < 1e-12

SPECIES = ['Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi']
SUPER_RULES = {
    'Gamma':  ['Pi', 'Delta', None, 'Theta', 'Sigma', 'Xi', 'Phi', 'Gamma'],
    'Delta':  ['Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
    'Theta':  ['Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
    'Lambda': ['Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
    'Xi':     ['Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'],
    'Pi':     ['Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'],
    'Sigma':  ['Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Lambda', 'Gamma'],
    'Phi':    ['Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
    'Psi':    ['Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'],
}
T_RULES = [(60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)]


def substitution_matrix():
    """M[i][j] = number of species-i children of a species-j supertile."""
    M = [[0] * 9 for _ in range(9)]
    for j, s in enumerate(SPECIES):
        for c in SUPER_RULES[s]:
            if c is not None:
                M[SPECIES.index(c)][j] += 1
    return M


def slot_transforms(quad):
    """Kaplan's buildSupertiles: the eight slot transforms and the new quad."""
    Ts = [IDENT]
    total = 0
    rot = IDENT
    tquad = list(quad)
    for ang, frm, to in T_RULES:
        total += ang
        if ang != 0:
            rot = trot(total // 30)
            tquad = [apply(rot, q) for q in quad]
        ttt = trans_to(tquad[to], apply(Ts[-1], quad[frm]))
        Ts.append(mul(ttt, rot))
    Ts = [mul(R, T) for T in Ts]
    new_quad = [apply(Ts[6], quad[2]), apply(Ts[5], quad[1]),
                apply(Ts[3], quad[2]), apply(Ts[0], quad[1])]
    return Ts, new_quad


def build(levels):
    """Return per level: dict species -> list of (transform, kind) leaf tiles, the quad, the slot transforms.

    kind is 'g1', 'g2' (Mystic halves) or 'p' (plain tile).  Level 0 is the single tile."""
    quad = [SPECTRE[3], SPECTRE[5], SPECTRE[7], SPECTRE[11]]
    g2 = mul(ttrans(SPECTRE[8]), trot(1))
    sys = {}
    for s in SPECIES:
        sys[s] = [(IDENT, 'p')] if s != 'Gamma' else [(IDENT, 'g1'), (g2, 'g2')]
    out = [(sys, quad, None)]
    for _ in range(levels):
        Ts, nq = slot_transforms(quad)
        new = {}
        for s in SPECIES:
            tiles = []
            for idx, c in enumerate(SUPER_RULES[s]):
                if c is None:
                    continue
                for (T, kind) in sys[c]:
                    tiles.append((mul(Ts[idx], T), kind))
            new[s] = tiles
        sys, quad = new, nq
        out.append((sys, quad, Ts))
    return out


def outline(T):
    return [apply(T, v) for v in SPECTRE]


def det_sign(T):
    """orientation of the linear part (+1 proper, -1 reflected), from the complex action."""
    a = to_c(T[0].dot(vec(1, 0, 0, 0)))
    b = to_c(T[0].dot(vec(0, 0, 1, 0)))   # image of z
    # proper iff image(z) = image(1) * z
    if abs(b - a * Z) < 1e-9:
        return 1
    assert abs(b - a * Z.conjugate()) < 1e-9
    return -1
