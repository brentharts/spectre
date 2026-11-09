#!/usr/bin/env python3
r"""spectre_density_clock.py - what the Spectre itself provides as an "interaction density", and how uniform it is.

quantgrav.tex sets the local clock rate to 1/rho with rho = 1 + sum_j M_j f(|x - x_j|).  The "1" is an unexplained baseline.  The Spectre
gives one: the density of volume-carrying nodes (valence-4 vertices of the tile graph, research/spin_densities.py) is an exact number,

    rho_0 = 1 - sqrt15/5 = (5 - sqrt15)/5 = 0.22540...  per spectre,

an element of the gap-labelling module Z + Z (5 + sqrt15)/10 (Exact; recorded in NEXT_STEPS.md, reproduced here from the Perron vector of the
38 collared level-1 types, K = 6).  What this script adds is the other half: how far the density of a FINITE region departs from rho_0.

For every complete level-j supertile (its tiles and their vertex neighbours lie inside the K = 6 patch) take
        rho_j = (weighted number of valence-4 vertices in the supertile) / (number of spectres in it),
with the weights of research/spin_densities.py (a vertex shared by m pieces counts 1/m for each).  Then the clock 1/rho_j relative to 1/rho_0
is the clock rate that a local observer would read off the tiling alone, with no mass added.

Measured (Counted; finite patch K = 6, 242047 tiles, every complete supertile of levels 1..4):
  1. The area-weighted mean of rho_j is rho_0 to 0.2% at j = 1, 0.06% at j = 2 and better than 0.01% at j = 3, 4.
  2. Inside one level, rho_j depends only on the supertile's label (the 9 labels fall into two groups: label 0, the Delta, has a smaller area).
  3. The deviation of a supertile's weighted valence-4 count from rho_0 * (its number of spectres) is, for the Delta, +0.197, +0.025, +0.003
     vertices at j = 1, 2, 3, and -0.029, -0.004, -0.0005 for the other eight labels: it shrinks by a factor of about 8 per level, i.e. like
     (4 - sqrt15)^j = lambda^(-2j) = 0.127^j.  There is no O(1) or perimeter term.  A region of ~500 spectres (j = 3) is uniform in this density
     to better than 1 part in 10^5.
  This is the Pisot rigidity of the inflation (lambda^2 = 4 + sqrt15 is a Pisot unit; the counts differ from rho_0 * area only by the Galois-conjugate
  part).  Contrast, from chair44_lines.py (Exact): for the Chair44, whose inflation is the integer 2, the notch sum of a depth-n supertile is
  7 * 2^n * (1,1,1), so its deviation from the box mean grows with scale.
  So the baseline "1" in rho is a normalisation, but the background it normalises is an exact algebraic number with an extremely quiet, exactly
  decaying fluctuation: the tiling makes no gravitational potential of its own.  Gravity must still be put in by hand (the M_j); nothing here derives
  a mass-dependent clock.

    python3 spectre_density_clock.py            report (about 40 s)
    python3 spectre_density_clock.py --tests    the checks above
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "research")
from gap_collared import build  # noqa: E402

RHO0 = 1 - math.sqrt(15) / 5          # exact: (5 - sqrt15)/5
SPEC_PER_PIECE_L = 60
_C = {}


def data(K=6):
    if K in _C:
        return _C[K]
    names, levels, V, PT = build(K)
    n_tiles = len(levels[0]["lab"])
    npc = len(PT)
    _, vid = np.unique(V.reshape(-1, 4), axis=0, return_inverse=True)
    vid = vid.reshape(npc, 14)
    nv = int(vid.max()) + 1
    e = np.sort(np.stack([vid, np.roll(vid, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    ue, einv, ecnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bd_piece = (ecnt[einv.reshape(npc, 14)] == 1).any(1)
    bd_tile = np.zeros(n_tiles, bool)
    np.logical_or.at(bd_tile, PT, bd_piece)
    deg = np.bincount(ue.ravel(), minlength=nv)
    npv = np.bincount(vid.ravel(), minlength=nv)
    L = SPEC_PER_PIECE_L
    wn = (L // npv).astype(np.int64)
    c4 = (wn[vid] * (deg[vid] == 4)).sum(1)
    per_tile = np.zeros(n_tiles, np.int64)
    np.add.at(per_tile, PT, c4)
    spectres = np.bincount(PT, minlength=n_tiles) * 0 + 1
    spectres = spectres + (np.bincount(PT, minlength=n_tiles) - 1)   # 1 per tile, 2 for the Mystic (two pieces)
    # tiles that share a vertex with a boundary tile are unreliable too
    vt = np.unique(np.stack([vid.ravel(), np.repeat(PT, 14)], 1), axis=0)
    bounds = np.nonzero(np.diff(vt[:, 0]))[0] + 1
    near_bd = bd_tile.copy()
    for grp in np.split(vt[:, 1], bounds):
        if bd_tile[grp].any():
            near_bd[grp] = True
    anc = {0: np.arange(n_tiles)}
    for j in range(1, K + 1):
        anc[j] = levels[j - 1]["parent"][anc[j - 1]]
    _C[K] = dict(levels=levels, per_tile=per_tile, spectres=spectres, near_bd=near_bd, anc=anc, n=n_tiles)
    return _C[K]


def supertile_densities(j, K=6):
    """-> (rho_j array over complete level-j supertiles, number of spectres in each, label of each)."""
    d = data(K)
    a = d["anc"][j]
    nsup = len(d["levels"][j]["lab"]) if j < K else 1
    bad = np.zeros(nsup, bool)
    np.logical_or.at(bad, a, d["near_bd"])
    c4 = np.zeros(nsup)
    sp_ = np.zeros(nsup)
    np.add.at(c4, a, d["per_tile"] / SPEC_PER_PIECE_L)
    np.add.at(sp_, a, d["spectres"])
    ok = (~bad) & (sp_ > 0)
    return c4[ok] / sp_[ok], sp_[ok], d["levels"][j]["lab"][ok] if j < K else None


def table(K=6, js=(0, 1, 2, 3, 4)):
    out = []
    for j in js:
        rho, area, _ = supertile_densities(j, K)
        out.append(dict(j=j, n=len(rho), area=float(area.mean()), mean=float(np.average(rho, weights=area)),
                        spread=float(rho.std() / RHO0)))
    return out


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    check("rho_0 = (5 - sqrt15)/5 = 0.22540...", abs(RHO0 - 0.22540333075851662) < 1e-12)
    # module membership: rho_0 = 2 - 2*(5 + sqrt15)/10
    check("rho_0 = 2 - 2 (5 + sqrt15)/10, an element of Z + Z (5 + sqrt15)/10 (n = -2, m = 2)", abs(RHO0 - (2 - 2 * (5 + math.sqrt(15)) / 10)) < 1e-12)
    T = table()
    if verbose:
        for r in T:
            print("       j=%d  supertiles %6d  mean area %8.1f  mean rho %.5f" % (r["j"], r["n"], r["area"], r["mean"]))
    check("the area-weighted mean density of complete supertiles is within 0.5% of rho_0 at j = 1..4",
          all(abs(r["mean"] / RHO0 - 1) < 0.005 for r in T if r["j"] >= 1), "(%s)" % [round(r["mean"] / RHO0 - 1, 5) for r in T])
    dev = {}
    for j in (1, 2, 3):
        rho, area, lab = supertile_densities(j)
        d = rho * area - RHO0 * area
        dev[j] = {int(L): d[lab == L] for L in set(lab.tolist())}
    check("within a level the deviation depends only on the label (constant on each of the 9 labels)",
          all(np.ptp(v) < 1e-9 for j in dev for v in dev[j].values()))
    ratios = [dev[j + 1][L][0] / dev[j][L][0] for j in (1, 2) for L in (0, 1)]
    check("the deviation shrinks by (4 - sqrt15) = lambda^-2 = 0.127 per level, to within 20%, for the Delta (0) and label 1",
          all(abs(r / (4 - math.sqrt(15)) - 1) < 0.2 for r in ratios), "(ratios %s)" % [round(r, 3) for r in ratios])
    check("no supertile of level 3 deviates from rho_0 * area by more than 0.005 vertices",
          max(np.abs(v).max() for v in dev[3].values()) < 0.005)
    return ok


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    run_tests()
