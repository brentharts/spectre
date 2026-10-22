#!/usr/bin/python3
"""The Spectre's growth rates as closed orbits of the BKL (Mixmaster) map: an exact fact about continued fractions.

The BKL map sends the Kasner parameter u > 1 to  u - 1  if u >= 2  and to  1/(u - 1)  if 1 < u < 2  (pyCICY
bkl.bkl_u_map).  It is the Gauss map in disguise: the number of subtractions before an inversion is the next partial
quotient, so a closed orbit is a quadratic irrational with a periodic continued fraction, and the multiplier of the
orbit is the square of the fundamental unit of the hyperbolic matrix that the partial quotients multiply to.

What this file proves, exactly (sympy, no floats):

1. The Spectre inflation.  lambda^2 = 4 + sqrt15 is an eigenvalue of [[7,6],[1,1]] = [6,1] (partial quotients 6 and 1;
   trace 8, det 1; char poly x^2 - 8x + 1, the quadratic factor of the Spectre's nine-species matrix).  The BKL orbit
   of u* = 3 + sqrt15 = [6;1,6,1,...] has period SEVEN epochs, in eras of 6 and 1 epochs, and its multiplier is
   (4 + sqrt15)^2 = lambda^4 exactly.
2. The Spectre perimeter growth.  nu = phi^3 = 2 + sqrt5 = [4;4,4,...]: a one-era orbit of 4 epochs, multiplier
   -nu^2 (the sign is det [[4,1],[1,0]] = -1: the orbit reverses orientation).  The "44" of Chair44 is that era length.
3. The Kasner denominator D(u) = 1 + u + u^2 = |u - zeta_3|^2, an Eisenstein norm, so on the hexagonal lattice Z[zeta_3]
   of the cusp torus.  At u* it is 7 lambda^2 = 28 + 7 sqrt15 exactly.

What it does not prove.  The theorem behind (1) and (2) holds for EVERY quadratic unit: every periodic continued
fraction gives a closed BKL orbit with multiplier the square of its unit (checked below over a grid of periods).  So
"lambda^2 is a BKL orbit" is not a property of the Spectre; the Spectre only selects the era pattern (6,1) and the
number 7.  There is no dynamics on the tiling for this to be the dynamics of.  The coincidence D(u*) = 7 lambda^2 with
7 = the number of epochs in the period is checked and left unexplained; it is not a Fact.  It is a one-off:
D/epsilon is rational only at the (6,1) point of this orbit (it is irrational at the (1,6) point and for (2,3), (3,2),
(2,5), ...), and (2,5) also has 7 epochs without D = 7 epsilon, so 7 = epochs is not the reason.

    python3 spectre_bkl.py            # checks
    python3 spectre_bkl.py --check
"""
import os
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

S15, S5 = sp.sqrt(15), sp.sqrt(5)
LAMBDA2 = 4 + S15
NU = 2 + S5


def bkl_step(u):
    """One epoch of the BKL map, exact.  Returns (u', derivative)."""
    if sp.N(u, 60) >= 2:
        return sp.radsimp(sp.simplify(u - 1)), sp.Integer(1)
    return sp.radsimp(sp.simplify(1 / (u - 1))), sp.simplify(-1 / (u - 1) ** 2)


def closed_orbit(u0, max_epochs=40):
    """Epochs of u0 until it returns: (list of u, era lengths, multiplier).  Raises if no return."""
    us, ders, u = [u0], [], u0
    for _ in range(max_epochs):
        u, d = bkl_step(u)
        ders.append(d)
        if sp.simplify(u - u0) == 0:
            mult = sp.radsimp(sp.simplify(sp.prod(ders)))
            eras, run = [], 0
            for x, d_ in zip(us, ders):
                run += 1
                if d_ != 1:                       # an inversion ends an era
                    eras.append(run)
                    run = 0
            assert run == 0
            return us, eras, mult
        us.append(u)
    raise ValueError("no closed orbit within %d epochs" % max_epochs)


def periodic_u(a, b):
    """The pure-periodic continued fraction [a; b, a, b, ...] as an exact surd."""
    x = sp.Symbol("x", positive=True)
    sol = sp.solve(sp.Eq(x, a + 1 / (b + 1 / x)), x)
    return [s for s in sol if sp.N(s) > 0][0]


def unit_of(a, b):
    """Larger eigenvalue of [[a,1],[1,0]] [[b,1],[1,0]]."""
    M = sp.Matrix([[a, 1], [1, 0]]) * sp.Matrix([[b, 1], [1, 0]])
    return max(M.eigenvals(), key=lambda e: sp.N(e))


def spectre_checks():
    u, eras, mult = closed_orbit(3 + S15)
    assert len(u) == 7 and eras == [6, 1]                              # 7 epochs, eras (6, 1)
    assert sp.simplify(mult - LAMBDA2 ** 2) == 0                       # multiplier lambda^4
    M = sp.Matrix([[6, 1], [1, 0]]) * sp.Matrix([[1, 1], [1, 0]])
    assert M == sp.Matrix([[7, 6], [1, 1]]) and M.trace() == 8 and M.det() == 1
    x = sp.Symbol("x")
    assert sp.expand(M.charpoly(x).as_expr() - (x ** 2 - 8 * x + 1)) == 0
    assert set(M.eigenvals()) == {4 + S15, 4 - S15}
    D = [sp.expand(1 + x + x ** 2) for x in u]
    assert sp.simplify(D[0] - 7 * LAMBDA2) == 0                        # D(u*) = 7 lambda^2
    # nu = phi^3: one era of 4 epochs, multiplier -nu^2
    u2, eras2, mult2 = closed_orbit(NU)
    assert len(u2) == 4 and eras2 == [4]
    assert sp.simplify(mult2 + NU ** 2) == 0
    assert sp.simplify(sp.GoldenRatio ** 3 - NU) == 0
    return True


def eisenstein_check():
    uu = sp.Symbol("u", real=True)
    z3 = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2
    assert sp.simplify(sp.expand(sp.Abs(uu - z3) ** 2, complex=True) - (1 + uu + uu ** 2)) == 0
    return True


def genericity_check(grid=((1, 6), (2, 3), (3, 1), (1, 4), (2, 5), (4, 3))):
    """(a != b; a = b is the one-era case, checked for nu above.)  Every periodic continued fraction gives a closed BKL orbit with multiplier (unit)^2 (up to sign)."""
    for a, b in grid:
        u0 = periodic_u(a, b)
        _, eras, mult = closed_orbit(u0)
        assert sorted(eras) == sorted([a, b]), (a, b, eras)
        assert sp.simplify(abs(mult) - unit_of(a, b) ** 2) == 0, (a, b, mult)
    # the D = 7 epsilon coincidence is a one-off: not rational for the other starting points
    for a, b in ((1, 6), (2, 3), (3, 2), (2, 5)):
        q = sp.radsimp(sp.expand(1 + periodic_u(a, b) + periodic_u(a, b) ** 2) / unit_of(a, b))
        assert not q.is_rational, (a, b, q)
    return True


def selftest():
    spectre_checks()
    eisenstein_check()
    genericity_check()
    return True


if __name__ == "__main__":
    print("selftest:", selftest())
    if "--check" in sys.argv:
        sys.exit(0)
    u, eras, mult = closed_orbit(3 + S15)
    print("BKL orbit of 3 + sqrt15: %d epochs, eras %s, multiplier %s = (4+sqrt15)^2" % (len(u), eras, mult))
    print("D(u*) = %s = 7 (4 + sqrt15)" % sp.expand(1 + u[0] + u[0] ** 2))
