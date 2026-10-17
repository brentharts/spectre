#!/usr/bin/env python3
r"""chair44_spectre_link.py - the Spectre 1.399 boundary dimension, tied to this fork's
boundary and torsion scripts, and set beside the Chair44 surface.

spectre_boundary.py already derives P(k) = 5P(k-1) - 3P(k-2) - P(k-3), charpoly
(x-1)(x^2-4x-1), nu = 2+sqrt5 = phi^3 and dim = log nu / log lambda = 1.39925.  Nothing
here re-derives that: the sequence, recurrence, nu and the exponents are IMPORTED from
spectre_boundary.  What is added:

1. EXACT FIBONACCI FORM (new here, checked by computation, not proved).
   Counting the oriented directions of the boundary edges of an order-k supertile
   (the 14 Spectre edges point in 12 directions, 30 degrees apart; the histogram has
   period 6, i.e. depends only on direction mod 180) gives
        even directions (0,60,120 mod 180):  F(3k+4) + 1   (summed over the three)
        odd  directions (30,90,150 mod 180):  F(3k+3) + 1
        P(k) = 2 [F(3k+4)+1 + F(3k+3)+1] = 2 F(3k+5) + 4        (F = Fibonacci)
   verified k = 0..5 (6 with --deep).  This is why nu = phi^3: one Spectre level multiplies the
   Fibonacci index by 3.  The conserved eigenvalue-1 mode is the constant 4.  The third
   eigenvalue of x^2-4x-1 is 2-sqrt5 = psi^3 = -1/phi^3: NEGATIVE, so that component
   alternates in sign with the level parity - the same parity that flips the determinant
   (chirality) of the placement matrix between levels.

2. Z/6 STRUCTURE.  Directions mod 180 form Z/6 = Z/2 x Z/3 (CRT) - the very split used by
   spectre_torsion.class_group_action for the six Mystic site classes (anchor mod 3, sign
   mod 2).  The DFT of the histogram over Z/6 shows: the Z/2 character carries the
   golden growth (amplitude F(3k+2)); the Z/3 and faithful characters stay bounded in
   the data.  This identifies the SAME GROUP, not (proved) the same action.

3. CHAIR44 ANALOGUE and a unified exponent.  With d the dimension, lambda the linear scale
   and nu the boundary growth per level:
        D_b = log nu / log lambda,   s* = 1 - D_b/d,   mu = lambda^d / nu = lambda^(d-D_b)
   Spectre: d=2, lambda^2 = 4+sqrt15, nu = phi^3     -> D_b = 1.39925, s* = 0.30037, mu = 1.8586
   Chair44: d=3, lambda = 2,            nu = 4       -> D_b = 2,       s* = 1/3,     mu = 2
   The fractal excess D_b - (d-1) is 0.3993 for the Spectre and exactly 0 for the Chair44
   (its supertile surface is a genuine surface).  The chair's interface counts follow the
   same law the fork found for the Spectre, B_c = (T(c) P(N-c) - P(N))/2, here
   B_c = 12 * 4^N * (2^c - 1), and its face-direction histogram is exactly flat
   (4^(N+1) per +-x,+-y,+-z), the opposite of the Spectre's golden-split histogram.

    python3 chair44_spectre_link.py            # k <= 5
    python3 chair44_spectre_link.py --deep     # k <= 6 (272791 tiles, a few minutes)
"""
import argparse
import math
from collections import Counter, defaultdict

import numpy as np

import chair44 as C


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


## ------------------------------------------------------------ Spectre side
def boundary_histogram(k):
    """Counts of boundary edges by oriented direction (index j = angle / 30 degrees mod 12)."""
    from spectre_selfsimilar import build_patch, edge_table      # the fork's own builders
    tiles = build_patch(k)
    em = edge_table(tiles)
    h = np.zeros(12, dtype=int)
    for users in em.values():
        if len(users) != 1:
            continue
        ti, ei = users[0]
        w = tiles[ti]["verts"]
        d = w[(ei + 1) % 14] - w[ei]
        a = math.degrees(math.atan2(d[1], d[0])) % 360.0 / 30.0
        assert abs(a - round(a)) < 1e-3, "edge not in a multiple-of-30-degree direction"
        h[int(round(a)) % 12] += 1
    return h


def golden_prediction(k):
    """(even-class total, odd-class total, P) predicted by the Fibonacci form."""
    e, o = fib(3 * k + 4) + 1, fib(3 * k + 3) + 1
    return e, o, 2 * (e + o)


def z6_modes(h):
    """DFT over Z/6 of the direction histogram folded mod 180 degrees (h[j] = h[j+6])."""
    assert all(h[j] == h[j + 6] for j in range(6)), "histogram is not 6-periodic"
    h6 = np.array(h[:6], dtype=float)
    return np.fft.fft(h6)


def spectre_side(max_k=5, verbose=True):
    import sympy as sp
    import spectre_boundary as SB
    hist = [boundary_histogram(k) for k in range(max_k + 1)]
    P = [int(h.sum()) for h in hist]
    order, coeffs, poly = SB.find_recurrence(P)           # needs >= 7 terms (k = 0..6, --deep)
    if poly is None:                                      # fewer terms: verify the fork's published recurrence
        coeffs, order = [5, -3, -1], 3
        assert all(P[i] == 5 * P[i - 1] - 3 * P[i - 2] - P[i - 3] for i in range(3, len(P)))
        poly = sp.factor(sp.expand(SB.x ** 3 - 5 * SB.x ** 2 + 3 * SB.x + 1))
    nu = SB.growth_rate(poly)
    mu, a, s_star, dim = SB.exact_exponents(nu)
    out = {"hist": hist, "P": P, "order": order, "coeffs": [int(c) for c in coeffs],
           "poly": poly, "nu": float(sp.N(nu)), "dim": float(sp.N(dim)), "s_star": float(sp.N(s_star)),
           "mu": float(sp.N(mu))}
    out["golden_ok"] = all(P[k] == golden_prediction(k)[2] and
                           int(hist[k][0::2][:3].sum()) == golden_prediction(k)[0] and
                           int(hist[k][1::2][:3].sum()) == golden_prediction(k)[1] for k in range(max_k + 1))
    out["modes"] = [z6_modes(h) for h in hist]
    if verbose:
        print("SPECTRE   (imported from spectre_boundary.py)")
        print("  P(k) =", P)
        print("  recurrence order %s coeffs %s, charpoly %s" % (order, out["coeffs"], poly))
        print("  nu = %.6f  boundary dim = %.6f  s* = %.6f  mu = %.6f" % (out["nu"], out["dim"], out["s_star"], out["mu"]))
        print("  Fibonacci form P(k) = 2 F(3k+5) + 4 and the even/odd split F(3k+4)+1, F(3k+3)+1: %s" %
              ("exact for k = 0..%d" % max_k if out["golden_ok"] else "FAILS"))
        print("  direction histograms (index = angle/30 mod 12; period 6):")
        for k, h in enumerate(hist):
            print("   k=%d %s" % (k, h[:6].tolist()))
        print("  |DFT over Z/6| of the folded histogram (mode 0 = total, 3 = even-odd, 2/4 = Z/3, 1/5 = faithful):")
        for k, m in enumerate(out["modes"]):
            print("   k=%d %s   F(3k+2)=%d" % (k, " ".join("%8.2f" % abs(z) for z in m), fib(3 * k + 2)))
    return out


## ------------------------------------------------------------ torsion tie
def torsion_tie(verbose=True):
    """The CRT split Z/6 = Z/2 x Z/3 of spectre_torsion, as an explicit check."""
    import spectre_torsion as T
    lab = T.class_group_action()                  # asserts the 6-cycle and the CRT injectivity itself
    ok = len({(k % 3, k % 2) for k in range(6)}) == 6
    if verbose:
        print("TORSION   spectre_torsion.class_group_action labels (angle, anchor):", lab)
        print("  Z/6 = Z/3 (anchor) x Z/2 (sign), CRT-injective: %s ; same group as the direction classes mod 180" % ok)
    return lab, ok


## ------------------------------------------------------------ chair side
def chair_side(max_n=3, verbose=True):
    rows = []
    for N in range(max_n + 1):
        poses = C.sigma_depth(N)
        fo = defaultdict(list)
        for i, p in enumerate(poses):
            for key, s in C.exposed_faces(p).items():
                fo[key].append((i, s))
        normals = Counter((key[0], v[0][1]) for key, v in fo.items() if len(v) == 1)
        dig = C.tile_digits(N)
        Bc = [sum(1 for v in fo.values() if len(v) == 2 and dig[v[0][0]][:c] != dig[v[1][0]][:c])
              for c in range(1, N + 1)]
        pred = [12 * 4 ** N * (2 ** c - 1) for c in range(1, N + 1)]
        # the fork's own formula, with T(c) = 8^c tiles and P(k) = 24 * 4^k
        pred2 = [(8 ** c * 24 * 4 ** (N - c) - 24 * 4 ** N) // 2 for c in range(1, N + 1)]
        rows.append({"N": N, "surface": sum(normals.values()), "normals": dict(normals), "B": Bc,
                     "pred": pred, "pred_fork_formula": pred2})
    if verbose:
        print("CHAIR44")
        for r in rows:
            flat = len(set(r["normals"].values())) == 1 and len(r["normals"]) == 6
            print("  N=%d surface faces %d = 24*4^N; normals flat: %s (%s per direction); B_c %s == 12*4^N*(2^c-1): %s" % (
                r["N"], r["surface"], flat, next(iter(r["normals"].values())), r["B"], r["B"] == r["pred"] == r["pred_fork_formula"]))
    return rows


def unified(verbose=True):
    lam2 = 4 + math.sqrt(15)
    nu_s = 2 + math.sqrt(5)
    rows = {
        "Spectre": {"d": 2, "lam": math.sqrt(lam2), "nu": nu_s},
        "Chair44": {"d": 3, "lam": 2.0, "nu": 4.0},
    }
    for r in rows.values():
        r["D_b"] = math.log(r["nu"]) / math.log(r["lam"])
        r["s_star"] = 1 - r["D_b"] / r["d"]
        r["mu"] = r["lam"] ** r["d"] / r["nu"]
        r["excess"] = r["D_b"] - (r["d"] - 1)
    if verbose:
        print("UNIFIED   D_b = log nu/log lambda ; s* = 1 - D_b/d ; mu = lambda^d/nu ; excess = D_b-(d-1)")
        for n, r in rows.items():
            print("  %-8s d=%d lambda=%.4f nu=%.4f   D_b=%.6f  s*=%.6f  mu=%.6f  excess=%+.6f" % (
                n, r["d"], r["lam"], r["nu"], r["D_b"], r["s_star"], r["mu"], r["excess"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args(argv)
    spectre_side(6 if a.deep else 5)
    print()
    torsion_tie()
    print()
    chair_side()
    print()
    unified()


if __name__ == "__main__":
    main()
