#!/usr/bin/python3
"""Fibre winding of the pancake against the torus of the Mystic classes: what is exact, what is not.

Two objects each carry a Z/6 and a circle fibre, and it is tempting to identify them.  This file states what can be
proved about the pairing and labels the rest.  The pancake is the non-singular big-bang solution of Dechant, Lasenby
and Hobson as ported in pyCICY (pancake.py, horizons.py): a collapsing circle fibre over a round base S^2.

1. THE FIBRE CLOCK (exact, given the horizon law).  A null ray along the collapsing fibre accumulates the comoving
   angle  h(t) = (2/a0) ln(t/t_in)  (the logarithmic horizon coefficient 2/a0 of horizons.fibre_log_coefficient), and
   the Euler angle psi has period 4 pi on the standard S^3 (pyCICY marks that identification as not checked).  So
        turns(t) = h / 4 pi = ln(t/t_in) / (2 pi a0).
   One level of the Spectre substitution rescales lengths by lambda = sqrt(4 + sqrt15), so n levels are the clock
        turns(n) = n ln(lambda) / (2 pi a0),   levels per turn  N(a0) = 2 pi a0 / ln(lambda).
   Both are closed forms.  What they do NOT say is that the tiling hierarchy is the pancake's time: that is a choice
   of dictionary, made here only to ask whether anything lands on an integer.

2. NUMEROLOGY, LABELLED.  At a0 = 1 (the pyCICY default) N = 6.0903..., a tempting 6 = |Z/6|.  It is 1.5% off, a0
   is initial data and not fixed by anything in the tiling, and N = 6 exactly would need a0 = 3 ln(lambda)/pi, a
   number chosen to give the answer.  The selftest asserts the miss.  This is not a Fact.

3. THE ORBIFOLD (exact, and unrelated to the pancake's dynamics).  The rotation R of spectre_torsion has order 6 on
   the cusp torus T = C/Z[zeta_3].  Its stabilisers on the six site classes (orbit-stabiliser, computed from R) are
   orders 6 (centre), 2 (the three edge midpoints, one orbit) and 3 (the two hexagon vertices, one orbit), so
   T/(Z/6) is the sphere S^2(2,3,6) and the orbifold Euler characteristic is
        2 - (1 - 1/2) - (1 - 1/3) - (1 - 1/6) = 0 = chi(T)/6,
   Riemann-Hurwitz for the quotient.  Z/6 in U(1)_fibre acting on S^3 gives the lens space L(6,1) with the same
   base S^2.  That the group is the same is the whole of the overlap: no map from torus classes to pancake fibre
   is constructed here.

    python3 spectre_pancake.py            # checks
    python3 spectre_pancake.py --check
"""
import os
import sys
from fractions import Fraction as Fr

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectre_torsion as TO              # noqa: E402

LAMBDA2 = 4 + sp.sqrt(15)                 # the Spectre inflation squared, the unit of Z[sqrt15]
LAMBDA = sp.sqrt(LAMBDA2)
FIBRE_PERIOD = 4 * sp.pi                  # period of the Euler angle on the standard S^3 (identification not checked)


# --------------------------------------------------------------------------
# 1. the fibre clock
# --------------------------------------------------------------------------
def fibre_angle(a0, t_over_tin):
    """Comoving null angle along the collapsing fibre: (2/a0) ln(t/t_in)."""
    return 2 / sp.sympify(a0) * sp.log(t_over_tin)


def turns_per_level(a0):
    """Fibre turns per Spectre inflation level: ln(lambda) / (2 pi a0)."""
    return sp.log(LAMBDA) / (2 * sp.pi * sp.sympify(a0))


def levels_per_turn(a0):
    """2 pi a0 / ln(lambda)."""
    return 1 / turns_per_level(a0)


def a0_for_levels_per_turn(N):
    """The a0 that would make N levels one turn: a0 = N ln(lambda) / (2 pi).  Tuning, not a derivation."""
    return sp.sympify(N) * sp.log(LAMBDA) / (2 * sp.pi)


def clock_checks():
    a0 = sp.Symbol("a0", positive=True)
    n = sp.Symbol("n", positive=True)
    # n levels: t/t_in = lambda^n, so turns = fibre_angle / (4 pi) = n ln(lambda) / (2 pi a0)
    turns = sp.simplify(fibre_angle(a0, LAMBDA ** n) / FIBRE_PERIOD)
    assert sp.simplify(sp.expand_log(turns, force=True) - n * turns_per_level(a0)) == 0
    # lambda is a unit: lambda^2 = 4 + sqrt15 has norm 16 - 15 = 1
    assert sp.expand(LAMBDA2 * (4 - sp.sqrt(15))) == 1
    assert sp.simplify(LAMBDA ** 2 - LAMBDA2) == 0
    # numerology, labelled: N(1) is near 6 and is not 6
    N1 = sp.N(levels_per_turn(1), 30)
    assert 6.08 < N1 < 6.10 and abs(N1 - 6) > 0.08
    # the a0 that would give exactly 6 is a tuned value, not 1
    assert sp.simplify(levels_per_turn(a0_for_levels_per_turn(6)) - 6) == 0
    assert abs(sp.N(a0_for_levels_per_turn(6)) - 1) > 0.01
    return True


# --------------------------------------------------------------------------
# 2. the orbifold
# --------------------------------------------------------------------------
def stabilisers():
    """Stabiliser order in <R> of each of the six site classes, from the action of R in spectre_torsion."""
    pts = [TO.ZERO] + list(TO.E2_PTS) + list(TO.VERT_PTS)
    out = {}
    for p in pts:
        s = sum(1 for k in range(6) if TO.act(TO.mat_pow(TO.R, k), p) == p)
        out[p] = s
    return out


def orbifold_checks():
    st = stabilisers()
    assert sorted(st.values()) == [2, 2, 2, 3, 3, 6]
    orbits = {}
    for p in st:
        orb = frozenset(TO.act(TO.mat_pow(TO.R, k), p) for k in range(6))
        orbits[orb] = st[p]
    assert sorted(orbits.values()) == [2, 3, 6] and len(orbits) == 3
    for orb, m in orbits.items():
        assert len(orb) * m == 6                      # orbit-stabiliser
    chi = 2 - sum(1 - Fr(1, m) for m in orbits.values())
    assert chi == 0                                   # = chi(T) / 6, Riemann-Hurwitz
    return sorted(orbits.values())


def selftest():
    TO.exact_checks()
    clock_checks()
    assert orbifold_checks() == [2, 3, 6]
    return True


def cross_check_horizon(a0=1.0):
    """Optional float check of the 2/a0 coefficient against pyCICY, if it is importable."""
    for cand in ("/home/claude/CICY", os.path.join(HERE, "..", "CICY")):
        if os.path.isdir(os.path.join(cand, "pyCICY")):
            sys.path.insert(0, cand)
            break
    try:
        import numpy as np
        from pyCICY import pancake as pk, horizons as hz
    except Exception as e:                            # noqa: BLE001
        return "skipped (%s)" % e
    sol = pk.pancake_solution(a0, 3.0, 2.0, 0.5, t_max=6.0)
    h = hz.pancake_horizons(sol)
    i, j = int(len(h["t"]) * 0.1), int(len(h["t"]) * 0.4)
    slope = (h["fibre"][j] - h["fibre"][i]) / np.log(h["t"][j] / h["t"][i])
    return "d(fibre)/d ln t = %.4f against 2/a0 = %.4f" % (slope, 2 / a0)


if __name__ == "__main__":
    print("selftest:", selftest())
    if "--check" in sys.argv:
        sys.exit(0)
    print("levels per fibre turn at a0 = 1: %s" % sp.N(levels_per_turn(1), 8))
    print("a0 giving exactly 6:             %s" % sp.N(a0_for_levels_per_turn(6), 8))
    print("orbifold cone orders:            %s" % orbifold_checks())
    print("pyCICY cross-check:              %s" % cross_check_horizon())
