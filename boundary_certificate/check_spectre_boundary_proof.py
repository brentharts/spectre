"""Referee certificate for G1: the Spectre perimeter recurrence for every k.

Claim.  For every k >= 0 the perimeter of a non-Mystic level-k supertile is
    P(k) = 2 F(3k+5) + 4,   so   P(k+3) = 5 P(k+2) - 3 P(k+1) - P(k).

Method.  Only two outlines occur at each level: O_k (all eight non-Mystic species) and G_k (the Mystic).
O_{k+1} is O_k in slots 0..6 plus G_k in slot 7; G_{k+1} is the same without slot 2.  The slot transforms
of Kaplan's construction are x -> A_i x + M_i q_k, with A_i fixed and M_i a fixed integer 4x16 matrix.  The
quad vector obeys q_{k+1} = Phi q_k.  So two placed outlines have a relative placement (X, Y, A, L): X at the
identity, Y placed by x -> A x + L q_k.  Write S_tau(k) for the length of the common boundary at level k.

  (a) If X_{k+1} and tau.Y_{k+1} have disjoint interiors, the common boundary is the union of the common
      boundaries of their child pairs.  So S_tau(k+1) = sum over child pairs of S_child(k), and the
      interior overlap obeys the same sum.
  (b) Starting from the sibling pairs, the child pairs close up on a finite set T of 'live' types.  Every
      other child type is 'pruned'.  It is proved separated at every level, from radius bounds for k <= K
      and from the exact eigen-decomposition of Phi on q_0 for k > K.  So it contributes 0 to both sums.
  (c) Base, level 0: every live type has zero interior overlap.  Its common boundary length S_tau(0) is
      measured from the polygons.
  (d) By induction on k, every child pair at every level has disjoint interiors.  Then
          P_O(k+1) = 7 P_O(k) + P_G(k) - 2 sum_{sibling pairs of O} S(k),
          P_G(k+1) = 6 P_O(k) + P_G(k) - 2 sum_{sibling pairs of G} S(k),
      and the state V(k) = (P_O, P_G, S_tau) obeys V(k+1) = B V(k) with a fixed integer matrix B.
  (e) The sequence w_k = e B^k p(B) V(0), with p = x^3 - 5x^2 + 3x + 1, satisfies the recurrence given by
      the characteristic polynomial of B (Cayley-Hamilton).  So it vanishes for all k once it vanishes for
      k < dim B.

Input: Kaplan's construction as transcribed in spectre_exact.py.  No tiling theorem is assumed;
non-overlap is proved in (d).  The float steps are the radius bounds, the separation margins and the
level-0 polygon areas and lengths.  Each has a margin far above double-precision error; the margins are
printed.

Usage: python3 check_spectre_boundary_proof.py     (about 5 minutes)
"""
import sys
import numpy as np
import sympy as sp
import mpmath as mp
from collections import Counter
from shapely.geometry import Polygon
from shapely.ops import unary_union
from spectre_exact import (slot_transforms, SPECTRE, vec, apply, mul, IDENT, ttrans, trot, to_c, build)

K = 60                      # levels checked directly; the tail k > K is bounded analytically
mp.mp.dps = 60

# ---------- linear structure ----------
def basis_quad(m):
    q = [vec(0, 0, 0, 0) for _ in range(4)]
    q[m // 4][m % 4] = 1
    return q


colsT = [[None] * 16 for _ in range(8)]
colsQ = [None] * 16
A = None
for m in range(16):
    Ts, nq = slot_transforms(basis_quad(m))
    if A is None:
        A = [T[0] for T in Ts]
    for i in range(8):
        assert (Ts[i][0] == A[i]).all()
        colsT[i][m] = Ts[i][1]
    colsQ[m] = np.concatenate(nq)
M = [np.array(colsT[i], dtype=object).T for i in range(8)]
Phi = np.array(colsQ, dtype=object).T
quad0 = [SPECTRE[3], SPECTRE[5], SPECTRE[7], SPECTRE[11]]
q0 = np.concatenate(quad0)
q = quad0
for k in range(5):        # slot_transforms is linear in the quad; confirm on the actual quads
    Ts, nq = slot_transforms(q)
    for i in range(8):
        assert (Ts[i][1] == M[i].dot(np.concatenate(q))).all()
    assert (np.concatenate(nq) == Phi.dot(np.concatenate(q))).all()
    q = nq
P_ = sp.Matrix(Phi.tolist())
print("charpoly(Phi) =", sp.factor(P_.charpoly().as_expr()))
KRY = [q0]
for m in range(6):
    KRY.append(Phi.dot(KRY[-1]))
assert all(x == 0 for x in KRY[6] - 9 * KRY[4] + 9 * KRY[2] - KRY[0])
assert sp.Matrix.hstack(*[sp.Matrix(v.tolist()) for v in KRY[:6]]).rank() == 6
print("minimal polynomial of q0 under Phi: x^6 - 9x^4 + 9x^2 - 1 = (x^2-1)(x^4-8x^2+1)")


def Ainv(a):
    mm = sp.Matrix(a.tolist()).inv()
    return np.array([[int(x) for x in row] for row in mm.tolist()], dtype=object)


AINV = [Ainv(a) for a in A]
CH = {'O': [(i, 'O') for i in range(7)] + [(7, 'G')],
      'G': [(i, 'O') for i in (0, 1, 3, 4, 5, 6)] + [(7, 'G')]}


def tkey(X, Y, Am, L):
    """two types with the same key have the same placement A x + L q_k at every level k"""
    return (X, Y, tuple(int(x) for x in Am.flatten()), tuple(tuple(int(x) for x in L.dot(v)) for v in KRY[:6]))


def children(X, Y, Am, L):
    out = []
    for i, cx in CH[X]:
        for j, cy in CH[Y]:
            out.append((cx, cy, AINV[i].dot(Am).dot(A[j]), AINV[i].dot(Am.dot(M[j]) + L.dot(Phi) - M[i])))
    return out


# ---------- centroids, scales and radius bounds ----------
QK = [q0]
for k in range(K + 2):
    QK.append(Phi.dot(QK[-1]))


def cen_fun(Am, L):
    """linear map q -> centroid of the quad of a shape placed by (Am, L q), as a 4x16 integer matrix times 1/4"""
    Cm = 4 * L
    for m in range(4):
        sel = np.zeros((4, 16), dtype=object)
        for t in range(4):
            sel[t, 4 * m + t] = 1
        Cm = Cm + Am.dot(sel)
    return Cm             # centroid = (Cm q) / 4


I4 = np.eye(4, dtype=object)
C_ID = cen_fun(I4, np.zeros((4, 16), dtype=object))
C_NEXT = C_ID.dot(Phi)                                  # centroid of q_{k+1}, as a function of q_k
OFF = [cen_fun(A[i], M[i]) - C_NEXT for i in range(8)]  # child centroid minus parent centroid
SCALE = np.zeros((4, 16), dtype=object)
for t in range(4):
    SCALE[t, 8 + t] = 4
    SCALE[t, t] = -4                                    # 4 (q[2] - q[0]), the same normalisation as C


def val(Cm, k):
    return to_c(Cm.dot(QK[k])) / 4


levels = build(6)
r = {'O': [], 'G': []}
for k in range(7):
    sysn, quad, _ = levels[k]
    c = val(C_ID, k)
    for X, s in (('O', 'Delta'), ('G', 'Gamma')):
        r[X].append(max(abs(to_c(apply(T, v)) - c) for T, kd in sysn[s] for v in SPECTRE))
for k in range(6, K):
    o = [abs(val(OFF[i], k)) for i in range(8)]
    r['O'].append(max(max(o[i] for i in range(7)) + r['O'][k], o[7] + r['G'][k]))
    r['G'].append(max(max(o[i] for i in (0, 1, 3, 4, 5, 6)) + r['O'][k], o[7] + r['G'][k]))
sc = [abs(val(SCALE, k)) for k in range(K + 1)]

# ---------- the tail k >= K, from q_k = sum_j mu_j^k v_j ----------
lam = mp.sqrt(4 + mp.sqrt(15))
MW = mp.mpc(mp.mpf(1) / 2, mp.sqrt(3) / 2)
MZ_ = mp.mpc(mp.sqrt(3) / 2, mp.mpf(1) / 2)


def mp_c(v):
    a, b, c, d = (int(x) for x in v)
    return a + b * MW + MZ_ * (c + d * MW)

MU = [lam, -lam, 1 / lam, -1 / lam, mp.mpf(1), mp.mpf(-1)]
V = mp.matrix(6, 6)
for row in range(6):
    for j in range(6):
        V[row, j] = MU[j] ** row           # coefficient of KRY[row] = Phi^row q0 = sum_j mu_j^row v_j
# q0 = sum_j v_j and Phi^m q0 = sum_j mu_j^m v_j: solve for the v_j inside the Krylov span
Kc = [[mp.mpf(int(x)) for x in v] for v in KRY[:6]]


def comps(Cm):
    """components of k -> to_c(Cm q_k): returns c_j with to_c(Cm q_k) = sum_j mu_j^k c_j"""
    y = mp.matrix([mp_c(Cm.dot(KRY[m])) for m in range(6)])
    return mp.lu_solve(V, y)


def tail(Cm):
    """for k >= K: |to_c(Cm q_k)/lam^k - limit_p| <= eps, with p = k mod 2.  Returns (limit_0, limit_1, eps)."""
    c = comps(Cm)
    lim = [c[0] + c[1], c[0] - c[1]]
    eps = (abs(c[4]) + abs(c[5])) * lam ** (-K) + (abs(c[2]) + abs(c[3])) * lam ** (-2 * K)
    for k in (K, K + 1):                 # consistency with the direct values
        assert abs(mp_c(Cm.dot(QK[k])) / lam ** k - lim[k % 2]) <= eps + mp.mpf(10) ** -40 * abs(lim[k % 2])
    return lim, eps


off_tail = [tail(OFF[i]) for i in range(8)]
H = [[(abs(off_tail[i][0][p]) + off_tail[i][1]) / 4 for p in (0, 1)] for i in range(8)]
# normalised radius bound R_k = r_k / lam^k for k >= K: R_{k+1} <= max_i (H_i(p) + R_k^{X_i}) / lam
U = {'O': mp.mpf(r['O'][K]) / lam ** K, 'G': mp.mpf(r['G'][K]) / lam ** K}
RB = {('O', K % 2): U['O'], ('G', K % 2): U['G']}
p = K % 2
for step in range(400):
    nO = max(max(H[i][p] + U['O'] for i in range(7)), H[7][p] + U['G']) / lam
    nG = max(max(H[i][p] + U['O'] for i in (0, 1, 3, 4, 5, 6)), H[7][p] + U['G']) / lam
    U = {'O': nO, 'G': nG}
    p ^= 1
    for X in 'OG':
        RB[(X, p)] = max(RB.get((X, p), mp.mpf(0)), U[X])
RB = {kk: v + mp.mpf(10) ** -30 for kk, v in RB.items()}   # the iteration contracts by 1/lam; slack


def separated(X, Y, Am, L):
    """margin (in units of sc_k) of a proof that X_k and (Am, L q_k) Y_k are disjoint for all k; > 0 means proved"""
    Cy = cen_fun(Am, L) - C_ID
    m = min((abs(val(Cy, k)) - r[X][k] - r[Y][k]) / sc[k] for k in range(K))
    lim, eps = tail(Cy)
    for p in (0, 1):
        m = min(m, float((abs(lim[p]) - eps) / 4 - RB[(X, p)] - RB[(Y, p)]))
    return m


# ---------- closure ----------
start = {'O': [], 'G': []}
for X in 'OG':
    ch = CH[X]
    for a in range(len(ch)):
        for b in range(a + 1, len(ch)):
            i, cx = ch[a]
            j, cy = ch[b]
            start[X].append((cx, cy, AINV[i].dot(A[j]), AINV[i].dot(M[j] - M[i])))
live, pruned, stack = {}, {}, start['O'] + start['G']
while stack:
    t = stack.pop()
    kk = tkey(*t)
    if kk in live or kk in pruned:
        continue
    mg = separated(*t)
    if mg > 0:
        pruned[kk] = mg
        continue
    live[kk] = t
    stack.extend(children(*t))
    assert len(live) < 5000
print("live types:", len(live), " pruned types:", len(pruned), " smallest separation margin: %.4f" % min(pruned.values()))

# ---------- base: level 0 ----------
G2 = mul(ttrans(SPECTRE[8]), trot(1))
LEAVES = {'O': [IDENT], 'G': [IDENT, G2]}


def poly(shape, T):
    ps = [Polygon([complex_xy(apply(mul(T, leaf), v)) for v in SPECTRE]) for leaf in LEAVES[shape]]
    return ps


def complex_xy(v):
    z = to_c(v)
    return (z.real, z.imag)


pg = poly('G', IDENT)
assert pg[0].intersection(pg[1]).area < 1e-9      # the two halves of the Mystic do not overlap
keys = list(live)
idx = {kk: n for n, kk in enumerate(keys)}
base = []
worst_area, worst_len = 0.0, 0.0
for kk in keys:
    X, Y, Am, L = live[kk]
    px = unary_union(poly(X, IDENT))
    py = unary_union(poly(Y, (Am, L.dot(q0))))
    a = px.intersection(py).area
    ln = px.boundary.intersection(py.boundary).length
    worst_area = max(worst_area, a)
    s = round(ln)
    worst_len = max(worst_len, abs(ln - s))
    base.append(s)
assert worst_area < 1e-9 and worst_len < 1e-9
print("level 0: largest interior overlap of a live type %.1e (zero), largest non-integrality of a common boundary %.1e"
      % (worst_area, worst_len))

# ---------- the linear system ----------
n = len(keys)
B = [[0] * (n + 2) for _ in range(n + 2)]       # state (P_O, P_G, S_1..S_n)
B[0][0], B[0][1] = 7, 1
B[1][0], B[1][1] = 6, 1
for X, row in (('O', 0), ('G', 1)):
    for t in start[X]:
        kk = tkey(*t)
        if kk in idx:
            B[row][2 + idx[kk]] -= 2
for kk in keys:
    for c in children(*live[kk]):
        ck = tkey(*c)
        if ck in idx:
            B[2 + idx[kk]][2 + idx[ck]] += 1
V0 = [14, 20] + base
Bm = sp.Matrix(B)
seq, v = [], sp.Matrix(V0)
for k in range(n + 6):
    seq.append((int(v[0]), int(v[1])))
    v = Bm * v
PO = [s[0] for s in seq]
PG = [s[1] for s in seq]
print("P_O(k), k=0..9:", PO[:10])
print("P_G(k), k=0..9:", PG[:10])

# compare with the exact patches (unit-edge count on the leaves), k <= 6 if available
def exact_perimeter(tiles):
    cnt = Counter()
    for T, kd in tiles:
        pts = [tuple(int(x) for x in apply(T, w)) for w in SPECTRE]
        for i in range(14):
            cnt[frozenset((pts[i], pts[(i + 1) % 14]))] += 1
    return sum(1 for c in cnt.values() if c == 1)


for k in range(7):
    assert exact_perimeter(levels[k][0]['Delta']) == PO[k]
    assert exact_perimeter(levels[k][0]['Gamma']) == PG[k]
print("matches the exact patches for k = 0..6")

# Cayley-Hamilton: w_k = e B^k p(B) V0 obeys the recurrence of charpoly(B), of order dim B, so it vanishes
# for all k once its first dim B terms vanish
dim = n + 2
assert len(PO) >= dim + 3
assert all(PO[k + 3] - 5 * PO[k + 2] + 3 * PO[k + 1] + PO[k] == 0 for k in range(dim))
F = [0, 1]
while len(F) < 3 * len(PO) + 6:
    F.append(F[-1] + F[-2])
# 2F(3k+5)+4 obeys the same recurrence: F(n+6) = 4F(n+3) + F(n) and (x-1)(x^2-4x-1) = x^3-5x^2+3x+1
assert all(PO[k] == 2 * F[3 * k + 5] + 4 for k in range(3))
assert all(PO[k] == 2 * F[3 * k + 5] + 4 for k in range(len(PO)))
print("PROVED for all k: P(k+3) = 5P(k+2) - 3P(k+1) - P(k), hence P(k) = 2F(3k+5) + 4 "
      "(the first %d terms of e B^k p(B) V0 vanish; dim B = %d)" % (dim, dim))

# the Mystic: find and prove its minimal recurrence the same way
for order in range(1, 8):
    Mx = sp.Matrix([[PG[k + j] for j in range(order)] for k in range(order)])
    if Mx.det() != 0:
        cvec = Mx.solve(sp.Matrix([PG[k + order] for k in range(order)]))
        if all(sum(cvec[j] * PG[k + j] for j in range(order)) == PG[k + order] for k in range(dim)):
            x = sp.symbols('x')
            print("P_G obeys, for all k, the recurrence with characteristic polynomial",
                  sp.factor(x ** order - sum(cvec[j] * x ** j for j in range(order))))
            break

# ---------- the slot-pair contact sequences (the paper's thirteen) ----------
print("sibling contact lengths G_ij(k) for the parent level k = 1..10, and whether G(k)=5G(k-1)-3G(k-2)-G(k-3):")
Sv = sp.Matrix(V0)
hist = []
for k in range(12):
    hist.append([int(x) for x in Sv])
    Sv = Bm * Sv
for X in 'OG':
    ch = CH[X]
    pairs = [(ch[a][0], ch[b][0]) for a in range(len(ch)) for b in range(a + 1, len(ch))]
    for (i, j), t in zip(pairs, start[X]):
        kk = tkey(*t)
        if kk not in idx:
            continue
        g = [hist[k][2 + idx[kk]] for k in range(11)]       # g[k] = contact inside a level-(k+1) parent
        ok = [k + 1 for k in range(3, 11) if g[k] != 5 * g[k - 1] - 3 * g[k - 2] - g[k - 3]]
        if X == 'O' or ok:
            print("  %s slots %d-%d:" % (X, i, j), g[:8], "fails at parent levels" if ok else "obeys it", ok if ok else "")
