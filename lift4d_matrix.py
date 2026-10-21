#!/usr/bin/env python3
"""lift4d_matrix.py - the exact inflation matrix of the Spectre in the 4D lift.

The supertile recursion of spectre_patch._build moves four key points ('quad') of the supertile.  Every map in it is a
rotation by a multiple of 60 degrees, a mirror, or a translation built from the key points themselves, so on the
lift Z[zeta_12] = Z^4 (zeta = e^{i pi/6}) it is an INTEGER matrix: rotation = multiplication by zeta^k, mirror
(x, y) -> (-x, y) is z -> -conj(z).  Acting on the four key points (Z^16) the one-level map is an integer matrix PHI.

Results (all exact, verified in the tests):
  charpoly(PHI) = (x-1)^4 (x+1)^4 (x^4 - 8 x^2 + 1)^2
  roots of x^4-8x^2+1: +-lambda, +-1/lambda with lambda^2 = 4 + sqrt(15)   (lambda = 2.8059)
  the quad vector has minimal polynomial (x^2-1)(x^4-8x^2+1)
The roots -lambda and -1/lambda are the orientation reversal (one level is antilinear).  4 + sqrt(15) is the fundamental
unit of Z[sqrt 15], norm 1, so the restriction of PHI to the x^4-8x^2+1 block has determinant 1 and trace 0: two
expanding and two contracting directions (the 4:2 cut-and-project structure), and the contraction factor on the
internal plane is 1/lambda.  The Chair44 inflation is 2 I_3 on Z^3: determinant 8, no contracting direction.

CLI: python3 lift4d_matrix.py [--tests-only]
"""
import argparse
import math
import sys

import numpy as np
import sympy as sp

import lift4d as L
import spectre_patch as sx

# multiplication by zeta on coordinates (1, zeta, zeta^2, zeta^3), using zeta^4 = zeta^2 - 1
ZM = np.array([[0, 0, 0, -1],
               [1, 0, 0, 0],
               [0, 1, 0, 1],
               [0, 0, 1, 0]], dtype=object)


def mpow(k):
    k %= 12
    m = np.eye(4, dtype=object)
    for _ in range(k):
        m = ZM.dot(m)
    return m


CONJ = np.column_stack([mpow(-j % 12).dot(np.array([1, 0, 0, 0], dtype=object)) if False else
                        mpow((-j) % 12)[:, 0] for j in range(4)])   # zeta^j -> zeta^-j  (column j = image of zeta^j)


class Aff:
    """w -> zeta^k conj^c (w) + t   on Z^4 coordinates (t a 4-vector or a symbolic callable result)."""

    def __init__(self, c, k, t):
        self.c, self.k, self.t = c, k % 12, t

    def lin(self):
        m = mpow(self.k)
        return m.dot(CONJ) if self.c else m

    def __call__(self, w):
        return self.lin().dot(w) + self.t

    def then(self, o):
        """self o o  (apply o first)."""
        return Aff((self.c + o.c) % 2, self.k + (-o.k if self.c else o.k), self.lin().dot(o.t) + self.t)


def _rot(deg):
    return Aff(0, deg // 30, np.zeros(4, dtype=object))


MIRROR = Aff(1, 6, np.zeros(4, dtype=object))


def step(base):
    """One supertile level on the four key points (list of 4 coordinate vectors)."""
    total, transforms, transformed = 0, [Aff(0, 0, np.zeros(4, dtype=object))], base
    for angle, src, dst in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
        if angle != 0:
            total += angle
            transformed = [_rot(total)(w) for w in base]
        r = _rot(total)
        t = Aff(0, 0, transforms[-1](base[src]) - transformed[dst])
        transforms.append(t.then(r))
    T = [MIRROR.then(t) for t in transforms]
    return [T[6](base[2]), T[5](base[1]), T[3](base[2]), T[0](base[1])]


def phi_matrix():
    """The 16 x 16 integer matrix of one level on the four key points (column = image of a basis vector)."""
    cols = []
    for j in range(4):
        for kk in range(4):
            base = [np.zeros(4, dtype=object) for _ in range(4)]
            base[j][kk] = 1
            cols.append(np.concatenate(step(base)))
    return sp.Matrix(np.array(cols, dtype=object).T.tolist())


def reference_quad():
    """Lifted key points pts[[3,5,7,11]] of the reference tile, as integer 4-vectors."""
    ref = sx._reference_points(1.0, 1.0)
    acc = [np.zeros(4, dtype=object)]
    for j in range(11):
        d = ref[j + 1] - ref[j]
        k = int(round(math.degrees(math.atan2(d[1], d[0])) / 30.0))
        acc.append(acc[-1] + np.array(L.step(k), dtype=object))
    return [np.array([acc[i][0], acc[i][2], acc[i][1], acc[i][3]], dtype=object) for i in (3, 5, 7, 11)]   # (p,q,r,s) -> (1,z,z^2,z^3) order


def to_pqrs(c):
    """Ring coordinates (1, z, z^2, z^3) -> the (p, q, r, s) of lift4d (p + q z^2 + r z + s z^3)."""
    return np.array([c[0], c[2], c[1], c[3]], dtype=float)


def numeric_quad(levels):
    """The key points after ``levels`` levels from the original floating-point recursion of spectre_patch."""
    quad = sx._reference_points(1.0, 1.0)[[3, 5, 7, 11]]
    for _ in range(levels):
        base, total, tr, transformed = quad, 0, [np.eye(3)], quad
        for angle, src, dst in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            if angle != 0:
                total += angle
                transformed = sx._apply(sx._rot(total), base)
            r = sx._rot(total)
            tr.append(sx._shift(sx._apply(tr[-1], base[src][None])[0] - transformed[dst]) @ r)
        tr = [np.diag([-1.0, 1.0, 1.0]) @ t for t in tr]
        quad = np.array([sx._apply(tr[6], base[2][None])[0], sx._apply(tr[5], base[1][None])[0],
                         sx._apply(tr[3], base[2][None])[0], sx._apply(tr[0], base[1][None])[0]])
    return quad


def block_matrix():
    """The 4 x 4 rational matrix of PHI on the invariant block with characteristic polynomial x^4 - 8 x^2 + 1:
    the companion matrix of that polynomial in the Krylov basis of (PHI^2 - I) applied to the quad vector."""
    P = phi_matrix()
    Q0 = sp.Matrix(np.concatenate(reference_quad()).tolist())
    Y = (P ** 2 - sp.eye(16)) * Q0
    K = sp.Matrix.hstack(Y, P * Y, P ** 2 * Y, P ** 3 * Y)
    return (K.T * K).inv() * K.T * P * K, K


def exact_bounded_plane():
    """The internal plane, exactly: a 2 x 4 matrix over Q(sqrt 15) acting on lift4d's (p, q, r, s).

    A functional f has bounded range on the supertile corners iff it kills the expanding part (Phi^2 = 4 + sqrt 15) of
    every iterate of the key-point vector.  Because each level mirrors the frame, the iterates alternate between the
    patch frame and its mirror image; so f must kill the expanding parts of the even iterates and of the mirrored odd
    ones.  Those parts span only a 2-dimensional space, whose annihilator F is 2-dimensional; the patch frame (which
    lift4d_patch fixes to the proper orientation) uses F composed with the mirror."""
    P = phi_matrix()
    Q0 = sp.Matrix(np.concatenate(reference_quad()).tolist())
    r15, I = sp.sqrt(15), sp.eye(16)
    A = P * P
    Mi = sp.Matrix(MIRROR.lin().tolist())
    Y = (A - I) * Q0

    def parts(Z):
        Ze = ((A - (4 - r15) * I) * Z) / (2 * r15)
        return [sp.Matrix(Ze[4 * i:4 * i + 4]) for i in range(4)]

    rows = [list(v) for v in parts(Y)] + [list(Mi * v) for v in parts(P * Y)]
    ns = sp.Matrix(rows).nullspace()
    F = sp.Matrix([[sp.nsimplify(sp.simplify(c)) for c in f] for f in ns])        # ring order (1, z, z^2, z^3)
    G = (F * Mi).applyfunc(sp.simplify)                                            # patch frame
    return G[:, [0, 2, 1, 3]]                                                       # to (p, q, r, s)


def window_boundary_exponents(levels=5, hs=(0.2, 0.1, 0.05, 0.025)):
    """Boundary-cell growth of the internal image (through the exact plane, orthonormalised) per halving of the cell
    size: log2(boundary cells(h/2) / boundary cells(h)).  Holes left by finite sampling are filled before counting.
    A smooth curve gives 1; a fractal boundary more.  Returns (list of (h, cells, holes, boundary), exponents)."""
    from scipy import ndimage as ndi
    G = np.array([[float(sp.N(c)) for c in row] for row in exact_bounded_plane().tolist()])
    V = np.array(list(L.lift_patch(levels)[1].values()), float)
    X = V @ np.linalg.qr(G.T)[0]
    rows = []
    for h in hs:
        ij = np.floor((X - X.min(0)) / h).astype(int)
        g = np.zeros(ij.max(0) + 3, bool)
        g[ij[:, 0] + 1, ij[:, 1] + 1] = True
        filled = ndi.binary_fill_holes(g)
        er = ndi.binary_erosion(filled, structure=ndi.generate_binary_structure(2, 1))
        rows.append((h, int(g.sum()), int(filled.sum() - g.sum()), int((filled & ~er).sum())))
    return rows, [math.log2(b[3] / a[3]) for a, b in zip(rows, rows[1:])]


def subspace_cosines(F1, F2):
    """Principal cosines between the row spaces of two 2 x 4 real matrices."""
    q1 = np.linalg.qr(np.asarray(F1, float).T)[0]
    q2 = np.linalg.qr(np.asarray(F2, float).T)[0]
    return np.linalg.svd(q1.T @ q2, compute_uv=False)


# ------------------------------------------------------------------ tests
X = sp.symbols("x")


def test_char_poly_is_the_expected_product_with_the_lambda_quartic():
    cp = sp.factor(phi_matrix().charpoly(X).as_expr())
    assert sp.expand(cp - (X - 1) ** 4 * (X + 1) ** 4 * (X ** 4 - 8 * X ** 2 + 1) ** 2) == 0


def test_lambda_squared_is_4_plus_sqrt15_and_is_the_inflation_factor():
    roots = [r for r in sp.Poly(X ** 4 - 8 * X ** 2 + 1, X).nroots()]
    top = max(abs(complex(r)) for r in roots)
    assert abs(top - math.sqrt(4 + math.sqrt(15))) < 1e-12 and abs(top - 2.8059) < 1e-4
    assert sp.simplify((4 + sp.sqrt(15)) * (4 - sp.sqrt(15))) == 1                      # a unit of Z[sqrt 15]


def test_exact_map_matches_the_floating_point_recursion():
    P = phi_matrix()
    Q = sp.Matrix(np.concatenate(reference_quad()).tolist())
    for n in (1, 2, 3):
        Q = P * Q
        phys = np.array([L.project(to_pqrs([float(c) for c in Q[4 * i:4 * i + 4]])) for i in range(4)])
        num = numeric_quad(n)
        assert np.abs(phys - (num[:, 0] + 1j * num[:, 1])).max() < 1e-9


def test_quad_vector_has_minimal_polynomial_x2_minus_1_times_the_quartic():
    P = phi_matrix()
    Q = sp.Matrix(np.concatenate(reference_quad()).tolist())
    mp = (X ** 2 - 1) * (X ** 4 - 8 * X ** 2 + 1)
    v = sum((sp.Poly(mp, X).coeff_monomial(X ** i) * P ** i * Q for i in range(7)), sp.zeros(16, 1))
    assert v == sp.zeros(16, 1)
    low = sp.Matrix.hstack(*[P ** i * Q for i in range(6)])
    assert low.rank() == 6


def test_block_is_unimodular_with_two_expanding_and_two_contracting_eigenvalues():
    M, _ = block_matrix()
    assert sp.expand(M.charpoly(X).as_expr() - (X ** 4 - 8 * X ** 2 + 1)) == 0
    assert M.det() == 1 and M.trace() == 0
    mod = sorted(abs(complex(r)) for r in sp.Poly(X ** 4 - 8 * X ** 2 + 1, X).nroots())
    assert mod[1] < 1 < mod[2] and abs(mod[0] * mod[3] - 1) < 1e-12


def test_chair44_inflation_has_no_contracting_direction():
    M = 2 * sp.eye(3)
    assert M.det() == 8 and all(abs(complex(e)) > 1 for e in M.eigenvals())


def test_exact_bounded_plane_is_the_fitted_plane_and_lies_over_q_sqrt15():
    G = exact_bounded_plane()
    assert G.shape == (2, 4)
    r15 = sp.sqrt(15)
    for c in G:
        c = sp.expand(sp.nsimplify(c))
        y = sp.Rational(sp.simplify(c.coeff(r15, 1)))
        x = sp.Rational(sp.simplify(c - y * r15))                    # raises if a non-rational part is left
        assert sp.simplify(x + y * r15 - c) == 0
    Gn = np.array([[float(sp.N(c)) for c in row] for row in G.tolist()])
    assert subspace_cosines(Gn, L.bounded_plane(4)[0]).min() > 1 - 1e-6


def test_exact_plane_is_bounded_on_the_patch_and_grows_for_the_physical_plane():
    Gn = np.array([[float(sp.N(c)) for c in row] for row in exact_bounded_plane().tolist()])
    ext = []
    for lev in (3, 4, 5):
        V = np.array(list(L.lift_patch(lev)[1].values()), float)
        ext.append(np.abs(V @ Gn.T).max())
    assert ext[2] < 1.2 * ext[1] and ext[1] < 1.5 * ext[0]
    assert abs(L.window_stats(5)["phys"] / L.window_stats(4)["phys"] - 2.8059) < 0.1


def test_window_boundary_is_fractal_not_a_smooth_curve():
    rows, ex = window_boundary_exponents(5, (0.2, 0.1, 0.05))
    assert all(r[2] <= 10 for r in rows) and all(1.1 < e < 1.4 for e in ex)
    area_ex = math.log2(rows[2][1] / rows[1][1])
    assert 1.8 < area_ex < 2.05


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-only", action="store_true")
    a = ap.parse_args(argv)
    if a.tests_only:
        return 0 if run_tests() else 1
    print("charpoly of PHI:", sp.factor(phi_matrix().charpoly(X).as_expr()))
    M, _ = block_matrix()
    print("block matrix:", M.tolist())
    print("roots:", [complex(r) for r in sp.Poly(X ** 4 - 8 * X ** 2 + 1, X).nroots()])
    return 0


if __name__ == "__main__":
    sys.exit(main())
