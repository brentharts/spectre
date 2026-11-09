"""Γ8: index Bragg peaks in the sheared (deformed) Z[zeta12] module.

Lattice coordinates n = V/28 in Z^4 give x = sum n_j zeta^j and x* = sum n_j zeta^(5j).  Fit x* = M x + b + r
(deform_fit v1: r bounded, radius ~3.8).  For a zeta^5 dual-module point (k5, k*), exp(-i(k5.x + k*.x*)) = 1 on
the lattice, so with k = k5 + M^T k*:  exp(-i k.x) = exp(i k*.(r + b)).  Hence S(k)/N^2 = |<exp(i k*.r)>|^2:
the predicted peak height is the window transform at k*.  Compute it (and the direct S(k)) for vertex and vortex
sets, and compare with the refined FFT peaks of module_fit v1.

Usage: python research/sheared_index.py OUT.json K
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build
from ollivier_ricci import vortices
from module_fit import sf, module_points, KMAX

REFINED = {"vortex": [7.7295, 6.0925, 7.937], "vertices": [7.7293]}


def main():
    out, K = sys.argv[1], int(sys.argv[2])
    _, _, V, _ = build(K)
    n = np.unique(V.reshape(-1, 4), axis=0).astype(float) / 28
    x = n @ np.exp(1j * np.pi / 6 * np.arange(4)); xs = n @ np.exp(5j * np.pi / 6 * np.arange(4))
    A = np.column_stack([x.real, x.imag, np.ones(len(x))])
    P = np.linalg.lstsq(A, np.column_stack([xs.real, xs.imag]), rcond=None)[0].T     # 2x3
    M = P[:, :2]
    r = xs - (A @ P.T @ np.array([1, 1j]))
    c = x.mean(); R = np.abs(x - c); keep = R < 0.8 * R.max()
    v = vortices(K, len(x))
    res = dict(K=K, M=M.tolist(), b=P[:, 2].tolist(), r_radius=float(np.abs(r).max()))
    k5, ks, m = module_points(5)
    kv = np.column_stack([ks.real, ks.imag]) @ M          # (M^T k*) as row vectors
    k = k5 + kv[:, 0] + 1j * kv[:, 1]
    ok = (np.abs(k) < KMAX) & (np.abs(k) > 0.5); k, ks, m = k[ok], ks[ok], m[ok]
    for name, mask in (("vertices", keep), ("vortex", keep & v)):
        pts = (x - c)[mask]; rr = r[mask]
        w = sf(-ks, rr)                     # |<exp(i k*.r)>|^2 predicted
        o = np.argsort(w)[::-1][:60]
        h = sf(k[o], pts)                   # direct S(k)/N^2 at those k
        rows = [dict(m=m[i].tolist(), k=float(abs(k[i])), kx=float(k[i].real), ky=float(k[i].imag),
                     kstar=float(abs(ks[i])), pred=float(w[i]), h=float(hh)) for i, hh in zip(o, h)]
        res[name] = sorted(rows, key=lambda d: -d["h"])
        res[f"{name}_match"] = {str(q): min(rows, key=lambda d: abs(d["k"] - q)) for q in REFINED[name]}
        print(name, [(round(d["k"], 4), round(d["pred"], 3), round(d["h"], 3)) for d in res[name][:10]], flush=True)
        print(name, "match", {q: (round(d["k"], 4), round(d["h"], 3)) for q, d in res[f"{name}_match"].items()}, flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
