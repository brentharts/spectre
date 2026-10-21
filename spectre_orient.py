#!/usr/bin/env python3
"""spectre_orient.py - the orientation substitution of the Spectre, and its exact sum rule.

In the Spectre substitution (spectre_patch._build) every level places the 8 slots of a supertile with
the same eight linear maps  L_s = Mir . R(phi_s)  (Mir: x -> -x, R: rotation), with the cumulative
angles phi = 0, 60, 60, 120, 180, 180, 240, 120 degrees; only the translations depend on the level.
So the linear part of a placed tile is a product of these maps, and the sum of the linear parts over a
supertile obeys a linear recursion.  In the complex coordinate z = x + i y a column of a linear part
maps as  z -> -exp(-i phi) conj(z),  so one level is antilinear and TWO levels are the linear map
B = C conj(C) on the nine tile labels, with the 9 x 9 matrix

    C[Y, X] = sum over the slots s of the supertile of label Y that hold a tile of label X of  -exp(-i phi_s),

The orientation of a placed tile is e^{i theta} = (second column of the linear part of its motion, as
x + i y) up to the fixed mirror of the reference tile; S_n = sum_j e^{i theta_j} then satisfies
S_{n+1} = C conj(S_n) component-wise, which this module checks against the placed tiles of
spectre_patch exactly, and the eigenvalues of B give the growth of |S_n| in closed form.

CLI: python3 spectre_orient.py [--tests-only]
"""
import argparse
import math
import sys

import numpy as np

import spectre_patch as sp

PHI_DEG = (0, 60, 60, 120, 180, 180, 240, 120)          # cumulative angles of the 8 slots
LABELS = sp.TILE_NAMES
IDX = {lab: i for i, lab in enumerate(LABELS)}
LAMBDA2 = 4.0 + math.sqrt(15.0)


def slot_matrices():
    """The eight linear maps L_s = Mir R(phi_s) as 2x2 arrays."""
    mir = np.diag([-1.0, 1.0])
    out = []
    for p in PHI_DEG:
        c, s = math.cos(math.radians(p)), math.sin(math.radians(p))
        out.append(mir @ np.array([[c, -s], [s, c]]))
    return out


def C_matrix():
    """C[Y, X] = sum over slots s of Y's supertile holding X of -exp(-i phi_s)  (9 x 9 complex), so that
    z_{n+1}[Y] = sum_X C[Y, X] conj(z_n[X])."""
    C = np.zeros((9, 9), complex)
    for Y, slots in sp.SUBSTITUTIONS.items():
        for s, X in enumerate(slots):
            if X:
                C[IDX[Y], IDX[X]] += -np.exp(-1j * math.radians(PHI_DEG[s]))
    return C


def initial_vector():
    """z_0[Y]: second column of the linear part of the level-0 tiles of Y, summed (Gamma holds a Mystic
    rotated by 30 degrees as well), as x + i y."""
    z = np.zeros(9, complex)
    for lab, tiles in sp._build(0, 1.0, 1.0).items():
        for shape, m in tiles:
            z[IDX[lab]] += m[0, 1] + 1j * m[1, 1]
    return z


def predicted_sums(levels):
    """Delta-supertile sums z_n[Delta] for n = 0..levels from the recursion  z_{n+1} = C conj(z_n)."""
    z = initial_vector()
    out = [z[IDX["Delta"]]]
    for _ in range(levels):
        z = C_matrix() @ np.conj(z)
        out.append(z[IDX["Delta"]])
    return np.array(out)


def placed_sums(levels):
    """The same sums from the placed tiles of the Delta supertile (second column of each motion's linear part)."""
    out = []
    for n in range(levels + 1):
        parts = sp._build(n, 1.0, 1.0)["Delta"]
        out.append(sum(m[0, 1] + 1j * m[1, 1] for _, m in parts))
    return np.array(out)


def two_level_matrix():
    C = C_matrix()
    return C @ np.conj(C)


def eigenvalues_two_level():
    return np.linalg.eigvals(two_level_matrix())


def exact_constant_squared():
    """c^2 = (15 + 5 sqrt3 + 3 sqrt5 + 3 sqrt15) / 20, the exact square of lim |S_n| / lambda^n; found by
    high-precision recursion (80 digits at n = 80, differences 1e-70) and identified as the root of
    400 x^4 - 1200 x^3 + 840 x^2 - 180 x + 9 = 0.  A computed identification, not a proof."""
    return (15.0 + 5.0 * math.sqrt(3.0) + 3.0 * math.sqrt(5.0) + 3.0 * math.sqrt(15.0)) / 20.0


def asymptotic_constant(levels=40):
    """lim |S_n| / lambda^n from the recursion (floating point)."""
    s = predicted_sums(levels)
    return abs(s[-1]) / math.sqrt(LAMBDA2) ** levels


# ------------------------------------------------------------------ tests
def test_slot_maps_are_the_ones_of_the_construction():
    # level-1 supertile of Delta: its slot placements have linear parts L_s, in slot order
    parts = sp._build(1, 1.0, 1.0)["Delta"]
    names = [n for n in sp.SUBSTITUTIONS["Delta"] if n]
    lin = [m[:2, :2] for _, m in parts]
    Ls = slot_matrices()
    k = 0
    for s, name in enumerate(sp.SUBSTITUTIONS["Delta"]):
        if not name:
            continue
        n_here = len(sp._build(0, 1.0, 1.0)[name])
        for j in range(n_here):
            base = sp._build(0, 1.0, 1.0)[name][j][1][:2, :2]
            assert np.allclose(lin[k], Ls[s] @ base), (s, name)
            k += 1
    assert k == len(lin)


def test_recursion_reproduces_the_placed_sums_exactly():
    pred, got = predicted_sums(4), placed_sums(4)
    assert np.allclose(pred, got, atol=1e-9), (pred, got)


def test_patch_orientation_sum_is_the_second_column_sum_up_to_a_global_phase_and_mirror():
    p = sp.SpectrePatch(4)
    S = abs(np.exp(1j * np.radians(p.angles())).sum())
    assert abs(S - abs(placed_sums(4)[4])) < 1e-6 * S


def test_two_level_matrix_has_the_perron_value_as_leading_modulus():
    ev = eigenvalues_two_level()
    assert abs(max(abs(ev)) - LAMBDA2) < 1e-9


def test_growth_constant_agrees_with_the_measured_one():
    assert abs(asymptotic_constant(30) - 1.4489) < 2e-4


def test_exact_constant_is_the_limit_and_a_root_of_the_quartic():
    c2 = exact_constant_squared()
    assert abs(asymptotic_constant(40) ** 2 - c2) < 1e-12
    assert abs(400 * c2 ** 4 - 1200 * c2 ** 3 + 840 * c2 ** 2 - 180 * c2 + 9) < 1e-9


def test_other_eigenvalues_of_the_two_level_matrix_are_at_most_one_in_modulus():
    ev = sorted(abs(eigenvalues_two_level()), reverse=True)
    assert abs(ev[0] - LAMBDA2) < 1e-9 and ev[1] < 1.0 + 1e-6 and abs(ev[3] - 1.0 / LAMBDA2) < 1e-9


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
    ev = eigenvalues_two_level()
    print("eigenvalues of B = C conj(C):")
    for x in sorted(ev, key=lambda v: -abs(v)):
        print("  %s   |.|=%.6f" % (np.round(x, 6), abs(x)))
    print("lambda^2 =", LAMBDA2, "  limit |S_n|/lambda^n =", asymptotic_constant(40), " c^2 exact =", exact_constant_squared())
    return 0


if __name__ == "__main__":
    sys.exit(main())
