"""Γ8: index the vortex peaks with each nontrivial internal Galois map, zeta -> zeta^p, p in (5, 7).

zeta^7 = conj(zeta^5), so p = 7 is the mirrored internal map (cross_fit v1 hypothesis: odd-K patches are mirrored
in lattice coordinates).  For each K and p: fit x*_p = M x + b + r on all sites and on vortex sites, then index
the vortex set with k = k_p + M^T k* (module_points(p)), using the vortex window.  Report the strongest direct
S(k) and the nearest point to each refined family (7.7295, 6.0925 at 17.89 deg, 7.937).

Usage: python research/galois_fit.py OUT.json K [K ...]
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build
from ollivier_ricci import vortices
from module_fit import sf, module_points, KMAX

REFINED = {"7.7295": complex(-6.82778, -3.62323), "6.0925": complex(5.79795, 1.87136), "7.937": complex(-7.58158, 2.34919)}
ROT = np.exp(1j * np.pi / 3 * np.arange(6))      # 6-fold orbit; Friedel included via +-k


def fit(x, xs):
    A = np.column_stack([x.real, x.imag, np.ones(len(x))])
    P = np.linalg.lstsq(A, np.column_stack([xs.real, xs.imag]), rcond=None)[0].T
    return P[:, :2], xs - (A @ P.T @ np.array([1, 1j]))


def run(K):
    _, _, V, _ = build(K)
    n = np.unique(V.reshape(-1, 4), axis=0).astype(float) / 28
    x = n @ np.exp(1j * np.pi / 6 * np.arange(4)); c = x.mean(); R = np.abs(x - c); keep = R < 0.8 * R.max()
    v = vortices(K, len(x)); mask = keep & v; pts = (x - c)[mask]; res = dict(K=K)
    for p in (5, 7):
        xs = n @ np.exp(1j * p * np.pi / 6 * np.arange(4))
        for win in ("all", "vortex"):
            sel = np.ones(len(x), bool) if win == "all" else v
            M, r = fit(x[sel], xs[sel])
            rr = xs - (np.column_stack([x.real, x.imag]) @ M.T @ np.array([1, 1j]))
            k5, ks, m = module_points(p)
            kv = np.column_stack([ks.real, ks.imag]) @ M; k = k5 + kv[:, 0] + 1j * kv[:, 1]
            ok = (np.abs(k) < KMAX) & (np.abs(k) > 0.5); k, ks = k[ok], ks[ok]
            w = sf(-ks, rr[mask] - rr[mask].mean()); o = np.argsort(w)[::-1][:120]
            h = sf(k[o], pts)
            top = sorted([dict(k=float(abs(k[i])), ang=float(np.degrees(np.angle(k[i]))), pred=float(w[i]), h=float(hh))
                          for i, hh in zip(o, h)], key=lambda d: -d["h"])
            kk = k[o]; match = {}
            for name, q in REFINED.items():
                orb = np.concatenate([q * ROT, -q * ROT])
                d = np.abs(kk[:, None] - orb[None]).min(1); j = int(np.argmin(d))
                match[name] = dict(dist=float(d[j]), k=float(abs(kk[j])), h=float(h[j]))
            key = f"p{p}_{win}"
            res[key] = dict(M=M.tolist(), r_radius=float(np.abs(r - r.mean()).max()), top=top[:16], match=match)
            print(K, key, "rad", round(res[key]["r_radius"], 2), [(round(t["k"], 4), round(t["ang"], 2), round(t["h"], 3)) for t in top[:4]],
                  {a: (round(b["dist"], 4), round(b["h"], 3)) for a, b in match.items()}, flush=True)
    return res


def main():
    out = sys.argv[1]; res = []
    for K in map(int, sys.argv[2:]):
        res.append(run(K)); json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
