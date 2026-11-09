#!/usr/bin/env python3
r"""spectre_mystic_xbond.py - every X-bond of the Spectre tiling belongs to a Mystic, and so the flavour frequency p = Pr(Phi^2) is exactly 1/2.

spectreatlas.py defines an *X-bond* as a shared edge on which the two tiles disagree about the edge type (a-edge of one against b-edge of the other).  The
Mystic is Tile(b, a), carrying the role-swapped sequence; every other tile carries the same sequence.  The Readme leaves one number open: the flavour
frequency p = Pr(Phi^2) (the fraction of Phi tiles whose X-charge is 2 rather than 0), measured 0.683, 0.579, 0.538, 0.519 at depths 3 to 6 and "still
falling"; if p = 1/2 then the bulk X-density is rho_X = 2g/(1+g) = 1 - sqrt15/5 (g = 4 - sqrt15).  CICY (pyCICY/theories/spectre.py) lists the same p as the
one unknown of its closed-form X-density.

Status of each statement (labels as in CONNECTIONS.md):
  Counted  M1  A shared edge is an X-bond iff exactly one of its two tiles is a Mystic, and no two Mystics share an edge.  So the X-charge of a tile is
               its number of Mystic edge-neighbours, and the X-charge of a Mystic is 14.  Checked on the whole patch at depths 3 and 4 and on the
               bonds of depth 5 (235776 bonds), for the Delta seed.
  Counted  M2  The X-bonds form disjoint 14-cycles, one for each Mystic, and each cycle passes through exactly two interior vertices of valence 4;
               every interior valence-4 vertex lies on exactly one such cycle.  So X-bonds = 14 M, valence-4 vertices = 2 M, X-bonds = 7 x (valence-4
               vertices), where M is the number of Mystics.  The last identity holds in the whole patch for all nine seeds at depths 2, 3, 4.
  Exact    M3  With the Perron frequencies in Z[sqrt15] (spectrefacts.FREQ; the Gamma metatile holds the Mystic and one normal tile), the Mystic is a
               fraction g/(1+g) = (5 - sqrt15)/10 of the tiles, the valence-4 density is 2g/(1+g) = (5 - sqrt15)/5 per tile (this is the exact density of
               CLOCK.md, now as 2 x the Mystic frequency), and charge conservation (every X-bond has one Mystic end and one normal end)
                    4 (t_G1 + t_D + t_S) + 2 t_L + 2 p t_Phi = 14 t_G2
               has the unique solution p = 1/2.  The charges 4, 4, 4, 2, {0, 2}, 0, 0, 0, 0 are the Counted atlas of spectreatlas.py (depths 3 to 6, 131
               contact classes); only Phi varies, so p is the only unknown and conservation fixes it.
  Counted  M4  Consistent with the measured sequence: p_k - 1/2 falls by a factor of about 0.4 to 0.5 per depth, and Aitken's extrapolation of
               depths 3, 4, 5 gives 0.511 (computed here).
What this is not: a proof of the atlas (the per-species charges are measured), nor of M1/M2 for every depth (they are finite checks, though M1 is what the
type-swap definition predicts).  It does not say what p means physically; it removes an unknown from the closed form.

    python3 spectre_mystic_xbond.py            report
    python3 spectre_mystic_xbond.py --tests    the checks above (about 20 s)
"""
import collections
import math
import sys

import numpy as np
import sympy as sp

import spectre as S
import spectreatlas as A
import spectrefacts as F

ROOT15 = sp.sqrt(15)
G = 4 - ROOT15


def patch_for(depth, seed="Delta"):
    tiles = S.buildSpectreTiles(depth, 1.0, 1.0)
    out = []

    def grab(T, label):
        pts = S.Mystic_SPECTRE_POINTS if label == "Gamma2" else S.SPECTRE_POINTS
        out.append((label, (T[:, :2] @ np.asarray(pts, dtype=np.float64).T).T + T[:, 2]))

    tiles[seed].forEachTile(grab)
    return out


def vertex_data(patch):
    """-> (valence per vertex key, boundary vertex keys, edge table)."""
    tab = A.edge_table(patch)
    bv = set()
    for key, ow in tab.items():
        if len(ow) == 1:
            bv.update(key)
    inc = collections.defaultdict(set)
    for i, (lab, w) in enumerate(patch):
        for p in w:
            inc[A._key(p)].add(i)
    return {v: len(s) for v, s in inc.items()}, bv, tab


def mystic_check(patch):
    """M1: -> (bonds, X-bonds, bonds that violate 'X iff exactly one Mystic', Mystic-Mystic bonds)."""
    shared, _, bad = A.bonds(patch)
    wrong = mm = 0
    for i, si, ti, j, sj, tj, isx in shared:
        mi, mj = patch[i][0] == "Gamma2", patch[j][0] == "Gamma2"
        mm += mi and mj
        wrong += isx != (mi != mj)
    return len(shared), sum(1 for e in shared if e[6]), wrong, mm, bad


def cycles(patch):
    """M2: -> (#X components, {(nodes, edges, interior valence-4 nodes): count}, interior valence-4 vertices, those on a component, #Mystics)."""
    val, bv, tab = vertex_data(patch)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    xedges = []
    for key, ow in tab.items():
        if len(ow) == 2 and ow[0][2] != ow[1][2]:
            xedges.append(key)
            parent[find(key[0])] = find(key[1])
    comp = collections.defaultdict(lambda: [set(), 0])
    for key in xedges:
        r = find(key[0])
        comp[r][0].update(key)
        comp[r][1] += 1
    v4 = {v for v, n in val.items() if n == 4 and v not in bv}
    shape = collections.Counter((len(c[0]), c[1], len(c[0] & v4)) for c in comp.values())
    on = sum(1 for v in v4 if any(v in c[0] for c in comp.values()))
    return len(comp), dict(shape), len(v4), on, sum(1 for lab, _ in patch if lab == "Gamma2")


def identity(patch):
    shared, _, _ = A.bonds(patch)
    val, bv, _ = vertex_data(patch)
    return sum(1 for e in shared if e[6]), sum(1 for v, n in val.items() if n == 4 and v not in bv)


def exact_p():
    """M3: the Perron frequencies at the level of tiles, then conservation of X-charge.  -> dict."""
    f = {n: sp.radsimp(sp.simplify(F.FREQ[n])) for n in F.SPECIES}
    N = sp.radsimp(sp.simplify(sum(f.values()) + f["Gamma"]))              # tiles per unit of metatile mass: the Gamma metatile has 2 tiles
    t = {n: sp.radsimp(sp.simplify(f[n] / N)) for n in F.SPECIES}           # per-tile frequency of the metatile's tile (Gamma: each of Gamma1, Gamma2)
    p = sp.symbols("p")
    charge = 4 * (t["Gamma"] + t["Delta"] + t["Sigma"]) + 2 * t["Lambda"] + 2 * p * t["Phi"]
    sol = sp.solve(sp.Eq(charge, 14 * t["Gamma"]), p)
    mystic = sp.radsimp(sp.simplify(t["Gamma"]))
    # X-bonds per tile = 14 t_G2; links per tile = 7; rho_X = X-bonds / links
    rho_x = sp.radsimp(sp.simplify(14 * t["Gamma"] / 7))
    return dict(t=t, N=N, p=sol, mystic=mystic, rho_x=rho_x, val4=sp.radsimp(sp.simplify(2 * t["Gamma"])))


def measured_p(depths=(3, 4, 5)):
    out = []
    for d in depths:
        r = A.analyse(d)
        ch = r["charges"]["Phi"]
        out.append(ch.get(2, 0) / (ch.get(0, 0) + ch.get(2, 0)))
    return out


def aitken(a, b, c):
    return c - (c - b) ** 2 / ((c - b) - (b - a))


def run_tests(verbose=True):
    ok = True

    def check(name, good, info=""):
        nonlocal ok
        ok &= bool(good)
        if verbose:
            print("  %-4s %-70s %s" % ("ok" if good else "FAIL", name, info))

    # M1
    for d in (3, 4, 5):
        n, x, wrong, mm, bad = mystic_check(patch_for(d))
        check("M1 depth %d: X-bond iff exactly one Mystic; no Mystic-Mystic bond" % d, n > 0 and wrong == 0 and mm == 0 and bad == 0,
              "%d bonds, %d X-bonds" % (n, x))
    # M2
    for d in (3, 4):
        P = patch_for(d)
        nc, shape, nv4, on, nm = cycles(P)
        check("M2 depth %d: X-bonds are disjoint 14-cycles with two valence-4 vertices each" % d, set(shape) == {(14, 14, 2)} and nc == nm == shape[(14, 14, 2)],
              "%d cycles, %d Mystics" % (nc, nm))
        check("M2 depth %d: every interior valence-4 vertex is on exactly one cycle, so v4 = 2 M" % d, on == nv4 == 2 * nm, "v4 = %d, M = %d" % (nv4, nm))
    # identity for all nine seeds
    worst = 0
    for seed in F.SPECIES:
        for d in (2, 3, 4):
            x, v4 = identity(patch_for(d, seed))
            worst = max(worst, abs(x - 7 * v4))
    check("M2: X-bonds = 7 x (interior valence-4 vertices), 9 seeds x depths 2-4", worst == 0, "max |difference| = %d" % worst)
    # M3
    E = exact_p()
    check("M3: Mystic frequency is (5 - sqrt15)/10", sp.simplify(E["mystic"] - (5 - ROOT15) / 10) == 0, str(sp.nsimplify(E["mystic"])))
    check("M3: valence-4 density is (5 - sqrt15)/5 = 2 x Mystic frequency (the exact density of CLOCK.md)", sp.simplify(E["val4"] - (5 - ROOT15) / 5) == 0)
    check("M3: charge conservation gives p = 1/2", E["p"] == [sp.Rational(1, 2)], str(E["p"]))
    check("M3: rho_X = 2g/(1+g) = 1 - sqrt15/5, the Readme's value if p = 1/2", sp.simplify(E["rho_x"] - (1 - ROOT15 / 5)) == 0 and sp.simplify(E["rho_x"] - 2 * G / (1 + G)) == 0)
    # the patch fractions
    P = patch_for(5)
    frac = sum(1 for lab, _ in P if lab == "Gamma2") / len(P)
    check("M3: Mystic fraction in the depth-5 patch matches 0.11270 to 1e-4", abs(frac - float((5 - ROOT15) / 10)) < 1e-4, "%.5f" % frac)
    # M4
    ps = measured_p()
    d1, d2 = ps[0] - 0.5, ps[1] - 0.5
    r = [(ps[i + 1] - 0.5) / (ps[i] - 0.5) for i in range(len(ps) - 1)]
    ait = aitken(*ps)
    check("M4: measured p falls toward 1/2 (depths 3, 4, 5)", all(a > b > 0.5 for a, b in zip(ps, ps[1:])), ", ".join("%.3f" % x for x in ps) + "; Aitken %.3f; ratios %s" % (ait, ", ".join("%.2f" % x for x in r)))
    check("M4: Aitken of the three depths is within 0.02 of 1/2", abs(ait - 0.5) < 0.02)
    if verbose:
        print("OK" if ok else "FAILED")
    return ok


def report():
    E = exact_p()
    print("Mystic frequency per tile :", sp.nsimplify(E["mystic"]), "=", float(E["mystic"]))
    print("valence-4 density per tile:", sp.nsimplify(E["val4"]), "=", float(E["val4"]))
    print("p = Pr(Phi^2) from conservation of X-charge:", E["p"])
    print("rho_X                     :", sp.nsimplify(E["rho_x"]), "=", float(E["rho_x"]))
    print("measured p at depths 3, 4, 5:", ["%.4f" % x for x in measured_p()])


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    report()
