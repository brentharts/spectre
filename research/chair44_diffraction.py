"""Γ8, Chair44 part: diffraction of a planar cut of the Chair44 patch, by the same Bragg test as
research/diffraction.py (h = I/N^2 of the top peaks, stable across depth for Bragg, ~1/N for diffuse).

The cut is the middle layer z = 2**depth of the depth-n patch (cells of edge 1): a polyomino tiling by
2x2 squares and L-trominoes.  Point sets:
  centroids   centroid of every tile cross-section
  vertices    lattice corners whose four cells are all occupied and belong to >= 3 different tiles
  trominoes   centroids of the L-tromino cross-sections only (the 'defect' subset, analogue of vortices)
The integer cell lattice gives trivial peaks at |k| = 2*pi; they are reported but flagged.

Usage: python research/chair44_diffraction.py OUT.json DEPTH [DEPTH ...]
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
import diffraction as D
from chair44_slice import slice_layer


def point_sets(depth):
    lay = slice_layer(depth, 2, 2 ** depth)
    occ = lay >= 0
    ids, inv, cnt = np.unique(lay[occ], return_inverse=True, return_counts=True)
    xs, ys = np.nonzero(occ)
    cx = np.bincount(inv, xs + 0.5) / cnt; cy = np.bincount(inv, ys + 0.5) / cnt
    cen = cx + 1j * cy
    a, b, c, d = lay[:-1, :-1], lay[1:, :-1], lay[:-1, 1:], lay[1:, 1:]
    full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    nd = 1 + (b != a).astype(int) + ((c != a) & (c != b)) + ((d != a) & (d != b) & (d != c))
    vx, vy = np.nonzero(full & (nd >= 3))
    return dict(centroids=cen, vertices=(vx + 1.0) + 1j * (vy + 1.0), trominoes=cen[cnt == 3]), dict(zip(*np.unique(cnt, return_counts=True)))


def main():
    out = sys.argv[1]; res = []
    for depth in [int(x) for x in sys.argv[2:]]:
        t = time.time(); sets, sizes = point_sets(depth)
        r = dict(depth=depth, piece_sizes={int(k): int(v) for k, v in sizes.items()})
        for name, pts in sets.items():
            pts = pts - pts.mean(); L = 1.05 * np.abs(pts).max()
            I, dk = D.pattern(pts, L)
            pk, frac, tot = D.peaks(I, dk, D.NPEAK, len(pts)); del I
            for p in pk: p["lattice"] = bool(abs(p["k"] - 2 * np.pi) < 3 * dk or abs(p["k"] - 2 * np.pi * np.sqrt(2)) < 3 * dk)
            r[name] = dict(N=int(len(pts)), dk=float(dk), top_peaks=pk, top_fraction=frac, total_over_N2=tot)
            print(depth, name, len(pts), [(round(p["h"], 4), round(p["k"], 2)) for p in pk[:6]], flush=True)
        r["seconds"] = round(time.time() - t, 1); res.append(r)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
