"""Γ8: seeded peak refinement (fixes module_fit's coarse scan at large K).

module_fit's 361-point coarse grid has step ~0.05, wider than a Bragg peak at K5 (width ~2pi/diameter), so it
misses the true peaks.  Here the vortex set uses sheared_index centring (x = n/28, |x-c| < 0.8 Rmax), and each
K4 refined vortex peak k (and, at odd K, conj(k)) is refined by local scans (0.05 -> 0.005 -> 5e-4 -> 5e-5).

Usage: python research/refine_seeded.py OUT.json K [K ...]
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build
from ollivier_ricci import vortices
from module_fit import sf

K4_REF = [(7.9369, -137.22), (7.9372, 162.78), (6.0924, 137.89), (6.0925, 17.89), (7.7296, -152.05), (7.7294, -32.05),
          (5.7224, 104.35), (6.0202, 97.2)]


def run(K):
    _, _, V, _ = build(K)
    n = np.unique(V.reshape(-1, 4), axis=0).astype(float) / 28
    x = n @ np.exp(1j * np.pi / 6 * np.arange(4)); c = x.mean(); R = np.abs(x - c); keep = R < 0.8 * R.max()
    pts = (x - c)[keep & vortices(K, len(x))]; rows = []
    for km, ang in K4_REF:
        k = km * np.exp(1j * np.radians(ang))
        for lab, k0 in (("same", k), ("conj", np.conj(k))):
            h0 = float(sf(np.array([k0]), pts)[0])
            for step in (0.05, 0.005, 0.0005, 0.00005):
                d = np.arange(-10, 11) * step; DX, DY = np.meshgrid(d, d)
                cand = (k0 + DX + 1j * DY).ravel(); hh = sf(cand, pts); k0 = cand[np.argmax(hh)]
            rows.append(dict(seed=km, seed_ang=ang, orient=lab, h_seed=h0, k=float(abs(k0)),
                             ang=float(np.degrees(np.angle(k0))), h=float(hh.max())))
            print(K, km, ang, lab, round(h0, 3), round(abs(k0), 4), round(np.degrees(np.angle(k0)), 2), round(hh.max(), 3), flush=True)
    return dict(K=K, N=int(len(pts)), rows=rows)


def main():
    out = sys.argv[1]; res = []
    for K in map(int, sys.argv[2:]):
        res.append(run(K)); json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
