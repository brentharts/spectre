#!/usr/bin/env python3
"""chair44_diffraction.py - the exact diffraction of the Chair44 centre set.

The centres of the bounding cubes of the 8**n chairs of the depth-n patch (chair44_slice.centres)
obey an exact nested recursion, verified here for n <= 5:

    X_0 = {(1,1,1)},      X_n = 2 X_{n-1}  U  ( O_n \\ Q_n ),

where O_n is the set of odd points of [1, 2**(n+1)-1]^3 and Q_n the odd points of the corner octant
[2**n+1, 2**(n+1)-1]^3 (the notch octant).  The structure factor F_n(k) = sum_{x in X_n} exp(i k.x)
therefore satisfies  F_n(k) = A_n(k) - B_n(k) + F_{n-1}(2k)  with A_n and B_n products of three
geometric sums, and unrolls to a finite sum of closed forms.

Bragg amplitudes.  For k = 2 pi m / 2**J (m not all even, J >= 1) the limit a(k) = lim F_n(k)/8**n is

    a(k) = (7/8) 8^-(J-1) ( (-1)^(m1+m2+m3) + 1/7 ),       a(0) = 1,

so |a|^2 = 8^-2(J-1) for m1+m2+m3 even and (9/16) 8^-2(J-1) for odd, and every other k has a = 0.
There are 3*8**(J-1) even and 4*8**(J-1) odd choices of m at level J, so the intensity at level J sums to
(21/4) 8^-(J-1) and the total is 1 + (21/4) (8/7) = 7.  The chairs fill a region of volume 7*8**n, so
the density is 1/7 and the diffraction intensity is I(k) = |a|^2 / 49: the Bragg peaks carry mass 7/49 = 1/7,
all of the density.  That is the Parseval criterion: the diffraction of the centre set is pure point.

CLI: python3 chair44_diffraction.py [--tests-only] [--png figures/chair44_bragg.png]
"""
import argparse
import itertools
import sys
from fractions import Fraction

import numpy as np

import chair44_slice as CS


# ------------------------------------------------------------------ the recursion
def odd_box(n, lo=1):
    r = range(lo, 2 ** (n + 1), 2)
    return {(a, b, c) for a in r for b in r for c in r}


def centres_recursive(n):
    """X_n from the nested recursion (a set of integer triples)."""
    if n == 0:
        return {(1, 1, 1)}
    prev = centres_recursive(n - 1)
    even = {(2 * a, 2 * b, 2 * c) for a, b, c in prev}
    return even | (odd_box(n) - odd_box(n, 2 ** n + 1))


def _g(kappa, a, b):
    """sum of exp(i kappa x) over x = a, a+2, ..., b (a <= b, same parity), 0 if a > b."""
    if a > b:
        return 0.0 + 0.0j
    m = (b - a) // 2 + 1
    z = np.exp(2j * kappa)
    if abs(z - 1.0) < 1e-12:
        return m * np.exp(1j * kappa * a)
    return np.exp(1j * kappa * a) * (1 - z ** m) / (1 - z)


def _AB(n, k):
    A = np.prod([_g(c, 1, 2 ** (n + 1) - 1) for c in k])
    B = np.prod([_g(c, 2 ** n + 1, 2 ** (n + 1) - 1) for c in k])
    return A - B


def structure_factor(n, k):
    """F_n(k) from the closed forms: sum over j of (A - B) at level n-j and wave vector 2**j k."""
    k = np.asarray(k, float)
    return sum(_AB(n - j, (2 ** j) * k) for j in range(n + 1))


def direct_structure_factor(n, k):
    P = CS.centres(n).astype(float)
    return complex(np.exp(1j * (P @ np.asarray(k, float))).sum())


# ------------------------------------------------------------------ Bragg amplitudes
def level_of(m):
    """(J, m reduced mod 2**J): smallest J with 2**J k in 2 pi Z^3 for k = 2 pi m / 2**J given as
    an integer triple m over 2**Jmax; here m is a triple of Fractions in [0, 1)."""
    J = 0
    while any((Fraction(x) * 2 ** J).denominator != 1 for x in m):
        J += 1
    return J


def bragg_amplitude(m, J):
    """Exact limit a(k) (Fraction) for k = 2 pi m / 2**J, m an integer triple, J >= 0, m not all even
    when J >= 1 (m = (0,0,0) with J = 0 is the origin)."""
    if J == 0:
        return Fraction(1)
    sign = -1 if sum(m) % 2 else 1
    return Fraction(7, 8) * Fraction(1, 8 ** (J - 1)) * (sign + Fraction(1, 7))


def bragg_intensity(m, J):
    """I(k) = |a|^2 / 49 (the density of the centre set is 1/7)."""
    return bragg_amplitude(m, J) ** 2 / 49


def level_mass(J):
    """Sum of |a|^2 over the points k = 2 pi m / 2**J with m not all even (exact)."""
    if J == 0:
        return Fraction(1)
    return Fraction(21, 4) * Fraction(1, 8 ** (J - 1))


def total_mass(Jmax=None):
    """Sum of |a|^2 over all levels; exactly 7 for Jmax = None (geometric series), else partial."""
    if Jmax is None:
        return Fraction(1) + Fraction(21, 4) * Fraction(8, 7)
    return sum(level_mass(J) for J in range(Jmax + 1))


def enumerate_level(J):
    """All m in {0..2**J-1}^3 not all even (for J >= 1)."""
    r = range(2 ** J)
    return [m for m in itertools.product(r, repeat=3) if any(x % 2 for x in m)]


def finite_amplitude(n, m, J):
    """F_n(k) / 8**n at k = 2 pi m / 2**J from the closed forms (converges to bragg_amplitude)."""
    k = 2 * np.pi * np.array(m) / 2 ** J
    return structure_factor(n, k) / 8 ** n


# ------------------------------------------------------------------ weighted version
def level_mass_numeric(n, J, weights=None, chunk=512):
    """sum of |sum_j w_j exp(i k.x_j)|^2 / Vol^2 over the points k = 2 pi m / 2**J with m not all even
    (all m for J = 0), Vol = 7 * 8**n, for the depth-n patch.  weights None = unit weights."""
    P = CS.centres(n).astype(float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    ms = np.array([m for m in itertools.product(range(2 ** J), repeat=3) if J == 0 or any(x % 2 for x in m)])
    tot = 0.0
    for i in range(0, len(ms), chunk):
        K = 2 * np.pi * ms[i:i + chunk] / 2 ** J
        tot += float((np.abs(np.exp(1j * (K @ P.T)) @ w) ** 2).sum())
    return tot / (7.0 * 8 ** n) ** 2


def level_masses(n, Jmax, comp=None):
    """Level-by-level Bragg mass of the depth-n patch: unit weights (comp None) or the notch component comp."""
    w = None if comp is None else CS.notches(n)[:, comp].astype(float)
    return [level_mass_numeric(n, J, w) for J in range(Jmax + 1)]


# ------------------------------------------------------------------ tests
def test_nested_recursion_reproduces_the_centres_exactly():
    for n in range(0, 6):
        assert centres_recursive(n) == {tuple(int(v) for v in p) for p in CS.centres(n)}, n


def test_closed_form_structure_factor_matches_the_direct_sum():
    rng = np.random.default_rng(3)
    for n in (1, 2, 3, 4):
        for k in rng.uniform(0, 2 * np.pi, (6, 3)):
            assert abs(structure_factor(n, k) - direct_structure_factor(n, k)) < 1e-8 * 8 ** n
    for k in ((0, 0, 0), (np.pi, 0, 0), (np.pi, np.pi, np.pi), (np.pi / 2, 0, np.pi)):
        assert abs(structure_factor(3, k) - direct_structure_factor(3, k)) < 1e-8 * 8 ** 3


def test_bragg_amplitude_formula_is_the_limit_of_the_closed_form():
    for J, m in ((1, (1, 0, 0)), (1, (1, 1, 0)), (1, (1, 1, 1)), (2, (1, 0, 0)), (2, (1, 2, 3)), (3, (3, 5, 2))):
        exact = float(bragg_amplitude(m, J))
        assert abs(finite_amplitude(12, m, J) - exact) < 1e-9, (J, m)
    assert abs(finite_amplitude(12, (0, 0, 0), 0) - 1.0) < 1e-12


def test_non_dyadic_wave_vectors_carry_no_bragg_amplitude():
    for k in (np.array([1.0, 0.3, 0.7]), np.array([2 * np.pi / 3, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])):
        assert abs(structure_factor(10, k)) / 8 ** 10 < 1e-3


def test_level_masses_sum_to_seven_so_the_diffraction_is_pure_point():
    for J in (1, 2, 3):
        got = sum(bragg_amplitude(m, J) ** 2 for m in enumerate_level(J))
        assert got == level_mass(J), J
    assert total_mass() == 7
    assert total_mass(40) < 7 and 7 - total_mass(40) < Fraction(1, 10 ** 30)
    assert total_mass() / 49 == Fraction(1, 7)                    # the density of the centre set


def test_finite_patch_intensity_at_a_bragg_point_matches_n_times_the_amplitude():
    n = 4
    P = CS.centres(n)
    for m, J in (((1, 0, 0), 1), ((1, 1, 0), 1)):
        k = 2 * np.pi * np.array(m) / 2 ** J
        I = abs(direct_structure_factor(n, k)) ** 2 / 8 ** n            # |F|^2 / N
        assert abs(I - float(bragg_amplitude(m, J)) ** 2 * 8 ** n) / I < 0.2   # corrections O(8^-n)


def test_unweighted_level_masses_match_the_exact_ones():
    got = level_masses(4, 3)
    for J in range(4):
        assert abs(got[J] - float(level_mass(J)) / 49) < 0.03 * float(level_mass(J)) / 49 + 1e-9, J


def test_notch_weighted_diffraction_has_no_bragg_mass_at_levels_0_and_1_and_converges_to_one_seventh():
    m = level_masses(4, 4, comp=2)
    assert m[0] < 1e-5 and m[1] < 1e-5                           # only the 2^-n tail at the origin
    assert abs(m[2] - 0.0703) < 5e-4 and abs(m[3] - 0.0498) < 5e-4
    assert 0.130 < sum(m) < 1 / 7 + 1e-9                       # approaching the density 1/7 from below


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


# ------------------------------------------------------------------ figure
def show_bragg(save=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    cols = ["#222222", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    # (a) Bragg peaks on the plane k_z = 0, k in units of 2 pi
    for J in range(0, 5):
        pts, size = [], []
        ms = [(0, 0, 0)] if J == 0 else [m for m in enumerate_level(J) if m[2] == 0]
        for m in ms:
            pts.append((m[0] / 2 ** J, m[1] / 2 ** J))
            size.append(float(bragg_intensity(m, J)))
        pts = np.array(pts)
        ax[0].scatter(pts[:, 0], pts[:, 1], s=np.sqrt(np.array(size)) * 400 + 3, c=cols[J], label="J = %d" % J, alpha=.8, lw=0)
    ax[0].set_xlabel("$k_x / 2\\pi$")
    ax[0].set_ylabel("$k_y / 2\\pi$")
    ax[0].set_aspect("equal")
    ax[0].set_title("Bragg peaks in the plane $k_z=0$ (area $\\propto$ intensity)", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper right")
    # (b) level masses
    Js = np.arange(0, 6)
    exact = [float(level_mass(J)) / 49 for J in Js]
    w = level_masses(5, 5, comp=2)
    ax[1].semilogy(Js, exact, "o-", color=cols[1], label="centres, exact $\\frac{21}{4}8^{-(J-1)}/49$")
    ax[1].semilogy(Js[2:], w[2:], "s--", color=cols[2], label="notch$_z$ weighted, level 5 patch")
    ax[1].set_xlabel("level $J$ of the dyadic wave vectors")
    ax[1].set_ylabel("Bragg mass at level $J$")
    ax[1].set_title("cumulative: centres 1/7 exactly; weighted %.4f of %.4f" % (sum(w), 1 / 7), fontsize=10)
    ax[1].legend(fontsize=8)
    # (c) finite-n amplitude converges to the exact one
    ns = np.arange(1, 13)
    for (m, J), c in zip((((1, 0, 0), 1), ((1, 1, 0), 1), ((1, 0, 0), 2)), cols[1:]):
        ax[2].semilogy(ns, [abs(finite_amplitude(n, m, J)) ** 2 for n in ns], "o-", color=c, ms=4,
                       label="m=%s, J=%d" % (str(m), J))
        ax[2].axhline(float(bragg_amplitude(m, J)) ** 2, color=c, ls=":")
    ax[2].set_xlabel("depth $n$")
    ax[2].set_title("$|F_n/8^n|^2$ converges to the exact $|a|^2$ (dotted)", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.suptitle("Chair44 centre set: exact pure point diffraction on dyadic wave vectors", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
    if show:
        plt.show()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--png")
    a = ap.parse_args(argv)
    if a.png:
        show_bragg(a.png, show=False)
        print("wrote", a.png)
        return 0
    if a.tests_only:
        return 0 if run_tests() else 1
    print("level  points      |a|^2 even / odd             mass       cumulative")
    cum = Fraction(0)
    for J in range(0, 6):
        cum += level_mass(J)
        ev = bragg_amplitude((0, 0, 2) if J == 0 else (1, 1, 0), J) ** 2 if J else Fraction(1)
        od = bragg_amplitude((1, 0, 0), J) ** 2 if J else Fraction(1)
        print("%3d  %8d   %-12s %-12s %-10s %s" % (J, 1 if J == 0 else 7 * 8 ** (J - 1), ev, od, level_mass(J), cum))
    print("total mass", total_mass(), " density 1/7 =", total_mass() / 49)
    for n in (4,):
        u, w = level_masses(n, 4), level_masses(n, 4, comp=2)
        print("n=%d  level masses  unweighted %s" % (n, ["%.5f" % x for x in u]))
        print("      notch_z weighted %s  cumulative %.5f  (density 1/7 = %.5f)" % (
            ["%.5f" % x for x in w], sum(w), 1 / 7))
    return 0


if __name__ == "__main__":
    sys.exit(main())
