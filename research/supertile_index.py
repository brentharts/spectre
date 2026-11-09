"""Γ8: is the 7.7295 vortex family a supertile superlattice peak, and does K5 carry the mirrored peaks?

Point sets per K: tile centres (level 0) and supertile centres at levels 1, 2 (mean of descendant tile centres),
also per supertile label.  For each set: S at the refined K4 vortex peaks and their mirrors conj(k), then a
coarse scan + local refinement of its own top peaks.  Positions use the vertex scale of module_fit (median edge 1).

Usage: python research/supertile_index.py OUT.json K [K ...]
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build
from graph_dilation import ZETA
from module_fit import sf, KMAX

REF = {"7.7295": complex(7.7294 * np.cos(np.radians(-32.05)), 7.7294 * np.sin(np.radians(-32.05))),
       "6.0925": 6.0925 * np.exp(1j * np.radians(17.89)), "7.937": 7.9372 * np.exp(1j * np.radians(162.78))}


def refine_top(pts, n=6):
    g = np.linspace(-KMAX, KMAX, 241); KX, KY = np.meshgrid(g, g)
    kk = (KX + 1j * KY).ravel(); kk = kk[(np.abs(kk) < KMAX) & (np.abs(kk) > 0.5)]
    sub = pts if len(pts) <= 20000 else pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    h = sf(kk, sub); top = []
    for i in np.argsort(h)[::-1]:
        if all(abs(kk[i] - t) > 0.3 and abs(kk[i] + t) > 0.3 for t in top): top.append(kk[i])
        if len(top) >= n: break
    out = []
    for k0 in top:
        for step in (0.02, 0.002, 0.0002):
            d = np.arange(-10, 11) * step; DX, DY = np.meshgrid(d, d)
            cand = (k0 + DX + 1j * DY).ravel(); hh = sf(cand, pts); k0 = cand[np.argmax(hh)]
        out.append(dict(k=float(abs(k0)), ang=float(np.degrees(np.angle(k0))), h=float(hh.max())))
    return out


def run(K):
    names, levels, V, PT = build(K)
    flat = V.reshape(-1, 4).astype(float); z = (flat @ ZETA).reshape(V.shape[0], 14)
    L = np.abs(z - np.roll(z, -1, 1)); z = z / np.median(L)
    cen = np.zeros(len(levels[0]["lab"]), complex); cnt = np.zeros(len(cen))
    np.add.at(cen, PT, z.mean(1)); np.add.at(cnt, PT, 1); cen /= cnt
    c0 = cen.mean(); R = np.abs(cen - c0).max()
    res = dict(K=K); sets = {}
    anc = np.arange(len(cen))
    for lev in range(0, 3):
        if lev > 0: anc = levels[lev - 1]["parent"][anc]
        ids = anc
        nG = len(levels[lev]["lab"]); s = np.zeros(nG, complex); c = np.zeros(nG)
        np.add.at(s, ids, cen); np.add.at(c, ids, 1); s = s / np.maximum(c, 1)
        keep = (c > 0) & (np.abs(s - c0) < 0.75 * R)
        sets[f"L{lev}"] = (s - c0)[keep]
        labs = levels[lev]["lab"][keep]
        for li in np.unique(labs):
            if (labs == li).sum() > 200: sets[f"L{lev}_{names[li]}"] = (s - c0)[keep][labs == li]
    for name, pts in sets.items():
        at = {q: (float(sf(np.array([k]), pts)[0]), float(sf(np.array([np.conj(k)]), pts)[0])) for q, k in REF.items()}
        top = refine_top(pts)
        res[name] = dict(N=len(pts), at_ref=at, top=top)
        print(K, name, len(pts), {q: (round(a, 3), round(b, 3)) for q, (a, b) in at.items()},
              [(round(t["k"], 4), round(t["ang"], 1), round(t["h"], 3)) for t in top], flush=True)
    return res


def main():
    out = sys.argv[1]; res = []
    for K in map(int, sys.argv[2:]):
        res.append(run(K)); json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
