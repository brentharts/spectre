#!/usr/bin/env python3
"""spectre_e8.py - the Spectre's 4D lift (lift4d.py) placed inside E8 through the Hopf fibres, and what that does and does not say.

E8 is built here as Hurwitz pairs (a, b), a - b in (1+i)H: 240 roots of norm^2 2.  A *fibre* is a right orbit of the binary
tetrahedral group 2T, (a, b) -> (a t, b t); there are 10 fibres of 24 roots, each the vertex set of a 24-cell (HOPF.md, sections 3-5).
The fibre rotation R_w for w = (-1+i+j+k)/2 has order 3 and no fixed vector, so it makes E8 a rank-4 module over Z[omega].

The lift lattice of lift4d.py is Z[zeta_12] = Z[w6] + z Z[w6] (w6 = e^{i pi/3}, z = e^{i pi/6}): the red A2 (even edge directions) plus the
green A2 (odd ones).  Take the roots r1 = (1+i, 0) and r2 = (0, 1+i); they are Hermitian-orthogonal over Z[omega].  Then

    embed(p, q, r, s) = (p + q (1 + R_w)) r1 + (r + s (1 + R_w)) r2           (1 + omega = w6)

is Z-linear and <embed x, embed y> = 2 Re(u_x conj u_y + v_x conj v_y), where u = p + q w6, v = r + s w6.

Status of each statement (labels as in CONNECTIONS.md):
  Exact     E8 has 240 roots in 10 fibres of 24, each a 24-cell; R_w has order 3 with no fixed vector; the 240 split as 6 + 72 + 162
            (A2, its orthogonal complement E6, and the rest).
  Exact     On the 71 tiles / 518 vertices of the level-2 patch: the embedding is an isometry for the red-orthogonal-green form and is
            injective on the vertices; every Spectre edge is a root of E8 (red edges in the pole D4a, green in the pole D4b); every outline is
            a closed walk of 14 roots; consecutive same-colour edges have inner product 2 cos(turn), red-green pairs 0.  The colour split of an
            outline is 8+6 for 63 tiles and 6+8 for the other 8.  A negative control (a root orthogonal to r1 but not Hermitian-orthogonal)
            breaks the isometry.  A finite check on one patch, not a proof for all patches (but the claim is about edges, which are the same
            14 steps up to rotation, so the patch exercises every case).
  Exact     The inflation block x^4 - 8x^2 + 1 is irreducible with roots +-lambda, +-1/lambda, lambda^2 = 4 + sqrt15.  Every automorphism of E8
            has finite order, so the inflation does NOT extend to an automorphism of E8.  Q(lambda) = Q(sqrt6, sqrt10) contains sqrt15 but
            not sqrt3, sqrt5 or the golden ratio.
  Hypothesis  That sqrt15 = sqrt3 x sqrt5 (hexagonal field times icosahedral field) points to a construction combining the Eisenstein structure of E8
            with its icosian structure.  Nothing here shows it.
  Not given   A cut-and-project scheme from E8 (the Spectre window is a fractal in the contracting plane, lift4d.py, arXiv:2411.15503), or any
            choice of sites by fibres, or any physical content.

    python3 spectre_e8.py            selftest (also: --check)
"""
import itertools
import math
import sys

import numpy as np

import lift4d as L

W6 = np.exp(1j * math.pi / 3)
OMEGA = np.array([-0.5, 0.5, 0.5, 0.5])          # w = (-1 + i + j + k)/2, order 3 in 2T
_C = {}


# ---------------------------------------------------------------- quaternions and E8
def qmul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])


def _k(v):
    return tuple(np.round(v, 6) + 0.0)


def in_hurwitz(y):
    y = np.asarray(y, float)
    return bool(np.allclose(y, np.round(y), atol=1e-9) or np.allclose(y - 0.5, np.round(y - 0.5), atol=1e-9))


def units():
    """The 24 Hurwitz units (2T = vertices of a 24-cell)."""
    pts = []
    for i in range(4):
        for s in (1.0, -1.0):
            v = np.zeros(4)
            v[i] = s
            pts.append(v)
    pts += [np.array(t) for t in itertools.product((0.5, -0.5), repeat=4)]
    return np.array(pts)


def e8():
    """-> (roots (240, 8), fibre label (240,), index dict)."""
    if "e8" not in _C:
        U = units()
        omi = np.array([1.0, -1.0, 0, 0])
        R = [np.r_[a, b] for a in U for b in U if in_hurwitz(qmul(omi, a - b) / 2.0)]
        n2 = [qmul(np.array([1.0, 1, 0, 0]), u) for u in U]
        R += [np.r_[np.zeros(4), b] for b in n2] + [np.r_[a, np.zeros(4)] for a in n2]
        R = np.array(R)
        idx = {_k(r): i for i, r in enumerate(R)}
        lab = -np.ones(len(R), int)
        n = 0
        for i, r in enumerate(R):
            if lab[i] < 0:
                for t in U:
                    lab[idx[_k(np.r_[qmul(r[:4], t), qmul(r[4:], t)])]] = n
                n += 1
        _C["e8"] = (R, lab, idx)
    return _C["e8"]


def right_matrix(l):
    """8x8 matrix of (a, b) -> (a l, b l) on row vectors."""
    M = np.zeros((8, 8))
    for k in range(4):
        e = np.zeros(4)
        e[k] = 1
        c = qmul(e, l)
        M[k, :4] = c
        M[4 + k, 4:] = c
    return M


def is_24cell(X):
    X = np.asarray(X, float)
    if len(X) != 24 or np.ptp((X ** 2).sum(1)) > 1e-6:
        return False
    D = ((X[:, None] - X[None]) ** 2).sum(2)
    e2 = D[D > 1e-6].min()
    return all([int((np.round(r / e2, 5) == v).sum()) for v in (0, 1, 2, 3, 4)] == [1, 8, 6, 8, 1] for r in D)


def rw():
    return _C.setdefault("rw", right_matrix(OMEGA))


def mul_omega(x, k=1):
    for _ in range(k):
        x = x @ rw()
    return x


R1 = np.r_[1.0, 1, 0, 0, np.zeros(4)]
R2 = np.r_[np.zeros(4), 1.0, 1, 0, 0]


def embed(v, r1=R1, r2=R2):
    p, q, r, s = (float(t) for t in v)
    return p * r1 + q * (r1 + mul_omega(r1)) + r * r2 + s * (r2 + mul_omega(r2))


def form(a, b):
    """2 Re(u_a conj u_b + v_a conj v_b) for lift coordinates."""
    ua, va = a[0] + a[1] * W6, a[2] + a[3] * W6
    ub, vb = b[0] + b[1] * W6, b[2] + b[3] * W6
    return 2 * ((ua * np.conj(ub)).real + (va * np.conj(vb)).real)


def is_red(step):
    return step[2] == 0 and step[3] == 0


# ---------------------------------------------------------------- checks
def e8_checks():
    R, lab, idx = e8()
    assert len(R) == 240 and np.allclose((R ** 2).sum(1), 2)
    assert len(set(lab)) == 10 and all((lab == f).sum() == 24 for f in set(lab))
    # a fibre is a 24-cell in R^8: its 24 points have the 24-cell distance structure
    assert all(is_24cell(R[lab == f]) for f in set(lab))
    M = rw()
    assert np.allclose(M @ M @ M, np.eye(8)) and not np.allclose(M, np.eye(8))
    assert np.linalg.matrix_rank(M - np.eye(8)) == 8                 # no fixed vector
    # A2 = R_w orbit of r1 and negatives; E6 = roots orthogonal to it
    A2 = [R1, mul_omega(R1), mul_omega(R1, 2)]
    A2 += [-a for a in A2]
    assert all(_k(a) in idx for a in A2)
    comp = [i for i, r in enumerate(R) if all(abs(r @ a) < 1e-9 for a in A2)]
    assert len(comp) == 72, len(comp)
    assert 240 - 6 - 72 == 162
    return True


def spectre_checks():
    tiles, lifted, _ = L.lift_patch(2)
    assert len(tiles) == 71 and len(lifted) == 518, (len(tiles), len(lifted))
    R, lab, idx = e8()
    lab_of = lambda v: lab[idx[_k(v)]]
    pa = lab_of(R1)
    pb = lab_of(R2)
    assert pa != pb
    assert all(abs(R1 @ mul_omega(R2, k)) < 1e-9 for k in range(3))
    for k in range(12):                                               # all 12 edge directions, so the claim holds for every patch
        assert _k(embed(L.step(k))) in idx
    V = [np.array(v) for v in lifted.values()]
    E = np.array([embed(v) for v in V])
    rng = np.random.default_rng(0)
    for _ in range(4000):
        i, j = rng.integers(len(V), size=2)
        assert abs(E[i] @ E[j] - form(V[i], V[j])) < 1e-9
    assert len(np.unique(np.round(E, 6), axis=0)) == len(V)
    split = {}
    for t in tiles:
        n = len(t)
        steps = [np.array(t[(i + 1) % n]) - np.array(t[i]) for i in range(n)]
        d = [embed(s) for s in steps]
        assert np.allclose(sum(d), 0)
        for s, e in zip(steps, d):
            assert _k(e) in idx                                       # every edge is a root of E8
            assert lab_of(e) == (pa if is_red(s) else pb)
        for i in range(n):
            j = (i + 1) % n
            ip = d[i] @ d[j]
            if is_red(steps[i]) == is_red(steps[j]):
                c = 0 if is_red(steps[i]) else 2
                a = steps[i][c] + steps[i][c + 1] * W6
                b = steps[j][c] + steps[j][c + 1] * W6
                assert abs(ip - 2 * (a * np.conj(b)).real) < 1e-9
            else:
                assert abs(ip) < 1e-9
        nr = sum(is_red(s) for s in steps)
        split[nr] = split.get(nr, 0) + 1
    assert split == {8: 63, 6: 8}, split
    # negative control
    wrong = next(r for r in R if abs(r @ R1) < 1e-9 and any(abs(r @ mul_omega(R1, k)) > 1e-9 for k in range(3)))
    bad = sum(abs(embed(a, R1, wrong) @ embed(b, R1, wrong) - form(a, b)) > 1e-9 for a, b in zip(V[:200], V[1:201]))
    assert bad > 0
    return True


def inflation_check():
    import sympy as sp
    x = sp.symbols("x")
    P = x ** 4 - 8 * x ** 2 + 1
    lam = (sp.sqrt(3) + sp.sqrt(5)) / sp.sqrt(2)
    assert sp.Poly(P, x).is_irreducible
    assert sp.simplify(P.subs(x, lam)) == 0 and sp.simplify(lam ** 2 - (4 + sp.sqrt(15))) == 0
    # Q(lambda, c) has degree 4 over Q iff c lies in Q(lambda)
    deg = lambda c: sp.degree(sp.minimal_polynomial(lam + c, x), x)
    assert deg(sp.sqrt(15)) == 4
    assert deg(sp.sqrt(3)) == 8 and deg(sp.sqrt(5)) == 8 and deg((1 + sp.sqrt(5)) / 2) == 8
    roots = np.roots([1, 0, -8, 0, 1])
    assert any(abs(abs(r) - 1) > 0.1 for r in roots)                  # not an isometry, so no finite order
    return True


def selftest():
    e8_checks()
    spectre_checks()
    inflation_check()
    return True


if __name__ == "__main__":
    print("selftest:", selftest())
