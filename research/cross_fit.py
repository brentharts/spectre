"""Γ8: is the alternating sign of the fitted shear M a fit artifact?  Does 7.7295 index on a vortex-only fit?

(1) Fit M on all sites at K5 and K4.  For each patch, form the sheared points k = k5 + M^T k* using its own M
    and the other K's M, and report the strongest direct vortex S(k) and the matches to the refined peaks.
(2) Fit M on vortex sites only (per K), report its residual radius, and index the vortex set with it.

Usage: python research/cross_fit.py OUT.json
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build
from ollivier_ricci import vortices
from module_fit import sf, module_points, KMAX

REFINED = [7.7295, 6.0925, 7.937]


def patch(K):
    _, _, V, _ = build(K)
    n = np.unique(V.reshape(-1, 4), axis=0).astype(float) / 28
    x = n @ np.exp(1j * np.pi / 6 * np.arange(4)); xs = n @ np.exp(5j * np.pi / 6 * np.arange(4))
    c = x.mean(); R = np.abs(x - c); keep = R < 0.8 * R.max()
    return x, xs, c, keep, vortices(K, len(x))


def fit(x, xs):
    A = np.column_stack([x.real, x.imag, np.ones(len(x))])
    P = np.linalg.lstsq(A, np.column_stack([xs.real, xs.imag]), rcond=None)[0].T
    r = xs - (A @ P.T @ np.array([1, 1j]))
    return P[:, :2], r


def index(M, x, xs, c, mask):
    k5, ks, m = module_points(5)
    kv = np.column_stack([ks.real, ks.imag]) @ M; k = k5 + kv[:, 0] + 1j * kv[:, 1]
    ok = (np.abs(k) < KMAX) & (np.abs(k) > 0.5); k, ks = k[ok], ks[ok]
    r = xs - (np.column_stack([x.real, x.imag]) @ M.T @ np.array([1, 1j]))
    w = sf(-ks, r[mask] - r[mask].mean()); o = np.argsort(w)[::-1][:80]
    h = sf(k[o], (x - c)[mask])
    rows = sorted([dict(k=float(abs(k[i])), ang=float(np.degrees(np.angle(k[i]))), pred=float(w[i]), h=float(hh))
                   for i, hh in zip(o, h)], key=lambda d: -d["h"])
    match = {str(q): min(rows, key=lambda d: abs(d["k"] - q) - 0.02 * d["h"]) for q in REFINED}
    return dict(top=rows[:16], match=match)


def main():
    out = sys.argv[1]; P = {K: patch(K) for K in (5, 4)}; Ms = {}; res = {}
    for K, (x, xs, c, keep, v) in P.items():
        Ms[K], r = fit(x, xs)
        Mv, rv = fit(x[v], xs[v])
        res[f"K{K}"] = dict(M=Ms[K].tolist(), M_vortex=Mv.tolist(), r_radius=float(np.abs(r).max()),
                            r_radius_vortex=float(np.abs(rv - rv.mean()).max()))
        res[f"K{K}"]["vortex_fit"] = index(Mv, x, xs, c, keep & v)
        print(K, "M", np.round(Ms[K], 4).tolist(), "Mv", np.round(Mv, 4).tolist(), flush=True)
    for K, (x, xs, c, keep, v) in P.items():
        for L in (5, 4):
            d = index(Ms[L], x, xs, c, keep & v); res[f"K{K}"][f"M_from_K{L}"] = d
            print(K, "M from", L, [(round(t["k"], 4), round(t["ang"], 2), round(t["h"], 3)) for t in d["top"][:6]], flush=True)
            print("   match", {q: (round(t["k"], 4), round(t["ang"], 2), round(t["h"], 3)) for q, t in d["match"].items()}, flush=True)
        d = res[f"K{K}"]["vortex_fit"]
        print(K, "vortex fit", [(round(t["k"], 4), round(t["ang"], 2), round(t["h"], 3)) for t in d["top"][:6]], flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
