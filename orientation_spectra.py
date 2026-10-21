#!/usr/bin/env python3
"""orientation_spectra.py - the orientation channels of the Chair44 and Spectre substitutions.

An orientation observable of a tile is a function on the group of its orientations.  Splitting it into
irreducible representations of that group, the sum of the observable over a supertile obeys a linear
recursion in each channel, and the growth per level is the leading eigenvalue of that channel's operator:

  Chair44 (group: the 24 rotations of the cube, irreps A1, A2, E, T1, T2):
      the operator of the channel rho is  sum_i rho(H_i)  over the eight child frames H_i.
      A1: 8 (the tile count)   A2, E, T1: 2 times the identity   T2: eigenvalues 2, 2, 0.
      Together x^3 (x-8) (x-2)^20.  Every non-trivial channel grows by exactly the linear inflation 2.

  Spectre (orientations: O(2) elements, harmonics m mod 6 since the slot angles are multiples of 60 deg):
      the operator of the channel m is  sum_slots  E_{Y,X} (x) Mir R(m phi_s)  on 9 labels x 2 components.
      m = 0 and 6: lambda^2 = 4 + sqrt15 (the tile count)   m = +-1: lambda per level (the sum rule of
      spectre_orient.py)   m = 2, 3, 4: modulus <= 1, bounded.

The growth exponent of a channel is h = ln(mu)/ln(lambda_lin) with lambda_lin the linear inflation (2 and
2.8059); a sum over a supertile of N = lambda_lin^d tiles has |S|^2/N ~ lambda_lin^((2h-d) n).

CLI: python3 orientation_spectra.py [--tests-only]
"""
import argparse
import itertools
import math
import sys

import numpy as np

import chair44_core as C
import spectre_orient as O
import spectre_patch as sp

LAMBDA_LIN_SPECTRE = math.sqrt(4.0 + math.sqrt(15.0))
LAMBDA_LIN_CHAIR = 2.0


# ------------------------------------------------------------------ chair
def _matrix(f):
    p, s = f
    m = np.zeros((3, 3))
    for i in range(3):
        m[i, p[i]] = s[i]
    return m


def _perm_sign(f):
    p = f[0]
    return (-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))


def chair_irreps():
    """{name: function frame -> matrix} for the five irreps of the rotation group of the cube
    (S4): A1 trivial, A2 the sign of the permutation of the axes, E the permutation action on
    (x^2, y^2, z^2) restricted to the sum-zero plane, T1 the rotation itself, T2 = T1 x A2."""
    Q = np.array([[1, -1, 0], [1, 1, -2]], float)
    Q = Q / np.linalg.norm(Q, axis=1)[:, None]

    def perm(f):
        m = np.zeros((3, 3))
        for i in range(3):
            m[i, f[0][i]] = 1
        return m
    return {
        "A1": lambda f: np.array([[1.0]]),
        "A2": lambda f: np.array([[float(_perm_sign(f))]]),
        "E": lambda f: Q @ perm(f) @ Q.T,
        "T1": lambda f: _matrix(f),
        "T2": lambda f: _matrix(f) * _perm_sign(f),
    }


def chair_channel_operators():
    """sum_i rho(H_i) for each irrep."""
    return {k: sum(rho(f) for f, _ in C.CHILDREN) for k, rho in chair_irreps().items()}


def chair_channel_eigenvalues():
    return {k: np.sort(np.linalg.eigvals(S).real)[::-1] for k, S in chair_channel_operators().items()}


def chair_charpoly_multiplicities():
    dims = {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}
    from collections import Counter
    c = Counter()
    for k, ev in chair_channel_eigenvalues().items():
        for e in ev:
            c[round(float(e), 6)] += dims[k]
    return dict(c)


# ------------------------------------------------------------------ spectre
def spectre_channel_operator(m):
    mir = np.diag([-1.0, 1.0])

    def rot(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s], [s, c]])
    M = np.zeros((18, 18))
    for Y, slots in sp.SUBSTITUTIONS.items():
        for s, X in enumerate(slots):
            if X:
                phi = math.radians(O.PHI_DEG[s])
                M[2 * O.IDX[Y]:2 * O.IDX[Y] + 2, 2 * O.IDX[X]:2 * O.IDX[X] + 2] += mir @ rot(m * phi)
    return M


def spectre_channel_leading(m):
    return float(np.sort(np.abs(np.linalg.eigvals(spectre_channel_operator(m))))[::-1][0])


def exponents():
    """Growth exponent h = ln(mu)/ln(lambda_lin) of every channel: {name: h}, in units where the tile
    count has h = d."""
    out = {}
    for k, ev in chair_channel_eigenvalues().items():
        mx = float(np.max(np.abs(ev)))
        out["chair " + k] = math.log(mx) / math.log(LAMBDA_LIN_CHAIR) if mx > 1e-9 else float("-inf")
    for m in range(0, 7):
        mu = spectre_channel_leading(m)
        out["spectre m=%d" % m] = math.log(mu) / math.log(LAMBDA_LIN_SPECTRE)
    return out


# ------------------------------------------------------------------ tests
def test_chair_channel_spectra_assemble_to_the_known_characteristic_polynomial():
    assert chair_charpoly_multiplicities() == {8.0: 1, 2.0: 20, 0.0: 3}


def test_chair_non_trivial_channels_are_two_times_a_projection():
    ops = chair_channel_operators()
    for k in ("A2", "E", "T1"):
        assert np.allclose(ops[k], 2 * np.eye(ops[k].shape[0])), k
    t2 = ops["T2"]
    assert np.allclose(t2 @ t2, 2 * t2) and abs(np.trace(t2) - 4) < 1e-9       # 2 x a rank-2 projector
    assert np.allclose(ops["A1"], [[8.0]])


def test_chair_irreps_are_homomorphisms_of_the_rotation_group():
    frames = [f for f in ((p, s) for p in itertools.permutations(range(3))
                          for s in itertools.product((1, -1), repeat=3)) if C.frame_det(f) == 1]
    assert len(frames) == 24
    for name, rho in chair_irreps().items():
        for a in frames[:12]:
            for b in frames[::2]:
                assert np.allclose(rho(C.frame_mul(a, b)), rho(a) @ rho(b)), name


def test_spectre_channels_have_exponents_two_one_and_below_zero_exactly_as_computed():
    lam2 = 4 + math.sqrt(15)
    assert abs(spectre_channel_leading(0) - lam2) < 1e-9 and abs(spectre_channel_leading(6) - lam2) < 1e-9
    for m in (1, 5):
        assert abs(spectre_channel_leading(m) - math.sqrt(lam2)) < 1e-9
    for m in (2, 3, 4):
        assert spectre_channel_leading(m) < 1.0 + 1e-6


def test_spectre_harmonic_one_agrees_with_the_orientation_sum_recursion():
    ev = np.sort(np.abs(np.linalg.eigvals(spectre_channel_operator(1))))[::-1]
    B = np.sort(np.abs(O.eigenvalues_two_level()))[::-1]
    assert abs(ev[0] ** 2 - B[0]) < 1e-9


def test_exponent_table_values():
    h = exponents()
    assert abs(h["chair A1"] - 3) < 1e-9
    for k in ("chair A2", "chair E", "chair T1", "chair T2"):
        assert abs(h[k] - 1) < 1e-9, k
    assert abs(h["spectre m=0"] - 2) < 1e-9 and abs(h["spectre m=1"] - 1) < 1e-9
    assert all(abs(h["spectre m=%d" % m]) < 1e-5 for m in (2, 3, 4))


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
    print("chair channel eigenvalues:")
    for k, ev in chair_channel_eigenvalues().items():
        print("  %-3s %s" % (k, np.round(ev, 6)))
    print("multiplicities:", chair_charpoly_multiplicities())
    print("exponents h (tile count: chair 3, Spectre 2):")
    for k, v in exponents().items():
        print("  %-12s %.4f" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
