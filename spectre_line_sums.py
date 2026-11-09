#!/usr/bin/env python3
r"""spectre_line_sums.py - the Spectre counterpart of chair44_polarisation.py: sums of the tile-orientation vector along straight lines.

Chair44 side (chair44_polarisation.py): the notch vector d, summed along an axis line of a patch, is bounded along the line and grows as (+-M, +-M)
across it on 13/24 of the lines (exact, limit-periodic substitution of integer inflation 2).  Here the same question for the Spectre, whose inflation
is irrational (lambda^2 = 4 + sqrt 15):

    cell observable    u = exp(i * rotation of the tile), the unit vector of the placement angle (a multiple of 30 degrees; 12 values)
    line               a straight line in direction theta through a level-5 patch (34649 tiles), sampled every 0.2 length units; the sequence
                       of tiles it enters, each counted once
    statistic          the longitudinal and transverse parts of the partial sums S_k = sum_{i<=k} u_i, rotated so that the line direction is real

Measured (finite, sampled, labelled Counted; this is NOT an exact computation like the Chair44 one):
  1. box mean of u over the patch: |mean| = 0.007.
  2. Mean |S_k| in each component grows like k^e with e between 0.26 and 0.62 (k = 20..140, 400-500 lines per direction; the shuffled-orientation control
     gives 0.39..0.55), so it is diffusive or slower.  At k = 140 the actual means are 7.0..10.5, against 5.8..8.3 for the control (a factor up to
     1.6 larger); a ballistic component would reach about 140.  There is no component with linear growth.
  So, for this observable, the Spectre shows NO ballistic polarisation along or across a line, in the directions 0, 30, 60, 90 degrees.  The Chair44
  effect is not a general property of aperiodic monotiles: it is tied to the integer inflation (limit-periodic order).

What this is not: a proof of anything for all lines, a test of other observables (edge directions, the Mystic classes, handedness Z_k of NEXT_STEPS A),
or a statement that the growth is exactly diffusive: the sampled lines are few and the constant is up to 1.6 times the control.

    python3 spectre_line_sums.py            report
    python3 spectre_line_sums.py --tests    the checks above (about 50 s)
"""
import sys

import numpy as np
from matplotlib.path import Path

import spectre_patch as sp

KS = (20, 40, 80, 140)
H = 3.0
_C = {}


def patch(level=5):
    if "p" not in _C:
        P = sp.SpectrePatch(level)
        polys = [P.straight_polygon(i) for i in range(P.n_tiles)]
        grid = {}
        for i, p in enumerate(polys):
            lo = np.floor(p.min(0) / H).astype(int)
            hi = np.floor(p.max(0) / H).astype(int)
            for a in range(lo[0], hi[0] + 1):
                for b in range(lo[1], hi[1] + 1):
                    grid.setdefault((a, b), []).append(i)
        cent = np.array([p.mean(0) for p in polys])
        c0 = cent.mean(0)
        _C["p"] = dict(u=np.exp(1j * np.radians(P.angles())), polys=polys, paths=[Path(p) for p in polys], grid=grid, c0=c0,
                       R=float(np.linalg.norm(cent - c0, axis=1).max()), n=P.n_tiles)
    return _C["p"]


def tile_at(pt):
    d = patch()
    for i in d["grid"].get(tuple(np.floor(pt / H).astype(int)), []):
        if d["paths"][i].contains_point(pt):
            return i
    return -1


def line_tiles(p0, direction, length, step=0.2):
    seq, last = [], -2
    for s in np.arange(0, length, step):
        i = tile_at(p0 + s * direction)
        if i < 0:
            return None
        if i != last:
            seq.append(i)
            last = i
    return seq


def sample_lines(deg, n_lines, seed):
    """Tile-index sequences of >= max(KS) tiles along random lines in direction deg."""
    d = patch()
    rng = np.random.default_rng(seed)
    t = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    nrm = np.array([-t[1], t[0]])
    out = []
    while len(out) < n_lines:
        p0 = d["c0"] + nrm * rng.uniform(-0.3, 0.3) * d["R"] - t * 0.3 * d["R"]
        seq = line_tiles(p0, t, 0.6 * d["R"])
        if seq and len(seq) >= max(KS):
            out.append(seq)
    return out


def mean_abs_partial_sums(u, seqs, deg):
    rot = np.exp(-1j * np.radians(deg))
    res = {k: [[], []] for k in KS}
    for seq in seqs:
        z = np.cumsum(u[seq]) * rot
        for k in KS:
            res[k][0].append(abs(z[k - 1].real))
            res[k][1].append(abs(z[k - 1].imag))
    return {k: (float(np.mean(v[0])), float(np.mean(v[1]))) for k, v in res.items()}


def exponent(stat, comp):
    return float(np.log(stat[KS[-1]][comp] / stat[KS[0]][comp]) / np.log(KS[-1] / KS[0]))


def run_tests(verbose=True, n_lines=400):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    d = patch()
    u = d["u"]
    check("level-5 patch has 34649 tiles", d["n"] == 34649)
    check("box mean of the orientation vector is small", abs(u.mean()) < 0.01, "(%.4f)" % abs(u.mean()))
    shuf = u.copy()
    np.random.default_rng(0).shuffle(shuf)
    worst_ratio, top, exps, ctrl_exps = 0.0, 0.0, [], []
    for k, deg in enumerate((0, 30, 60, 90)):
        seqs = sample_lines(deg, n_lines, seed=10 + k)
        a = mean_abs_partial_sums(u, seqs, deg)
        b = mean_abs_partial_sums(shuf, seqs, deg)
        for comp in (0, 1):
            worst_ratio = max(worst_ratio, a[KS[-1]][comp] / b[KS[-1]][comp])
            top = max(top, a[KS[-1]][comp])
            exps.append(exponent(a, comp))
            ctrl_exps.append(exponent(b, comp))
    check("growth exponent of |S_k| (k = 20..140) is between 0.15 and 0.7 in every direction and component: diffusive or slower, not linear",
          min(exps) > 0.15 and max(exps) < 0.7, "(%.2f..%.2f; control %.2f..%.2f)" % (min(exps), max(exps), min(ctrl_exps), max(ctrl_exps)))
    check("at k = 140 the actual mean |S_k| is within a factor 2 of the shuffled-orientation control", worst_ratio < 2.0, "(worst ratio %.2f)" % worst_ratio)
    check("no component is ballistic: the largest mean |S_140| is below 0.1 * 140 (a Chair44-type line would give ~140)", top < 14, "(%.1f)" % top)
    return ok


def report():
    d = patch()
    print("level-5 patch, %d tiles, box mean |u| = %.4f" % (d["n"], abs(d["u"].mean())))
    for k, deg in enumerate((0, 30)):
        a = mean_abs_partial_sums(d["u"], sample_lines(deg, 150, 10 + k), deg)
        print(deg, {kk: tuple(round(x, 2) for x in v) for kk, v in a.items()})


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    report()
