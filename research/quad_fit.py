"""Γ8: is there a second-order term in the deformation x* = M x + b + r?

Fit x* on [x, 1] (linear) and on [x, 1, quadratic monomials in centred x] for each K.  Report the residual
radius and the share of residual variance that the quadratic terms explain.  A real second-order term
(which could shift the 7.7295 peak family off the linear sheared module) would explain a large share and
shrink the radius.  If the share is ~0, the residual r is window noise and the linear module is the whole story.

Usage: python research/quad_fit.py OUT.json K [K ...]
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build


def run(K):
    _, _, V, _ = build(K)
    n = np.unique(V.reshape(-1, 4), axis=0).astype(float) / 28
    x = n @ np.exp(1j * np.pi / 6 * np.arange(4)); xs = n @ np.exp(5j * np.pi / 6 * np.arange(4))
    u = (x - x.mean()) / np.abs(x - x.mean()).max()
    Y = np.column_stack([xs.real, xs.imag]); out = dict(K=K, N=int(len(x)))
    for name, cols in (("linear", [u.real, u.imag]), ("quadratic", [u.real, u.imag, u.real ** 2, u.imag ** 2, u.real * u.imag])):
        A = np.column_stack(cols + [np.ones(len(u))])
        P = np.linalg.lstsq(A, Y, rcond=None)[0]; r = Y - A @ P
        out[name] = dict(radius=float(np.hypot(*r.T).max()), rms=float(np.sqrt((r ** 2).sum(1).mean())),
                         coef=P.tolist())
    out["explained_share"] = 1 - out["quadratic"]["rms"] ** 2 / out["linear"]["rms"] ** 2
    print(K, {k: (round(out[k]["radius"], 3), round(out[k]["rms"], 4)) for k in ("linear", "quadratic")},
          "share", round(out["explained_share"], 5), flush=True)
    return out


def main():
    out = sys.argv[1]; res = []
    for K in map(int, sys.argv[2:]):
        res.append(run(K)); json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
