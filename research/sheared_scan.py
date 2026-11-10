"""Γ8: scan S(k) for the vortex set in a ±0.01 box around each strong sheared-module point (sheared_index v1).
If maxima sit within 2e-3 at heights near the refined FFT peaks (0.49, 0.70), the fitted M is just slightly off.

Usage: python research/sheared_scan.py SHEARED.json OUT.json K
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build
from ollivier_ricci import vortices
from module_fit import sf


def main():
    src, out, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = json.load(open(src))
    _, _, V, _ = build(K)
    n = np.unique(V.reshape(-1, 4), axis=0).astype(float) / 28
    x = n @ np.exp(1j * np.pi / 6 * np.arange(4))
    c = x.mean(); R = np.abs(x - c); keep = R < 0.8 * R.max(); v = vortices(K, len(x))
    sets = {"vortex": (x - c)[keep & v], "vertices": (x - c)[keep]}
    seeds = []
    for r in d["vortex"][:40] + list(d["vortex_match"].values()):
        k = complex(r["kx"], r["ky"])
        if all(abs(k - s) > 1e-3 for s in seeds): seeds.append(k)
    res = []
    for k0 in seeds:
        row = dict(k0=[k0.real, k0.imag], k0_abs=abs(k0))
        for name, pts in sets.items():
            kk = k0; 
            for step in (1e-3, 1e-4, 1e-5):
                g = np.arange(-10, 11) * step; DX, DY = np.meshgrid(g, g)
                cand = (kk + DX + 1j * DY).ravel(); h = sf(cand, pts); kk = cand[np.argmax(h)]
            row[name] = dict(k=[kk.real, kk.imag], k_abs=abs(kk), shift=abs(kk - k0), h=float(h.max()),
                             h_at_k0=float(sf(np.array([k0]), pts)[0]))
        res.append(row)
        print(round(abs(k0), 4), {s: (round(row[s]["shift"], 5), round(row[s]["h_at_k0"], 3), round(row[s]["h"], 3)) for s in sets}, flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
