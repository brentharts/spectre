"""Γ8: is the Spectre vertex set a deformed model set with residual window r = x* - M x?

internal_check v1: x* = sum n_j zeta^(5j)/28 grows like x (ratio ~0.52-0.55), so the plain window is unbounded.
Fit x* ~ M x + b (M real 2x2) by least squares per K, and with the K5 fit applied to K3 and K4.
Bounded-window test: radius of r = x* - M x - b stays within 10% from K4 to K5 (and also q99).
Usage: python research/deform_fit.py OUT.json K [K ...]   (largest K first is the reference fit)
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build


def coords(K):
    _, _, V, _ = build(K)
    n = np.unique(V.reshape(-1, 4), axis=0).astype(float) / 28
    x = n @ np.exp(1j * np.pi / 6 * np.arange(4)); xs = n @ np.exp(5j * np.pi / 6 * np.arange(4))
    return x - x.mean(), xs - xs.mean()


def fit(x, xs):
    A = np.column_stack([x.real, x.imag, np.ones(len(x))])
    bx, *_ = np.linalg.lstsq(A, xs.real, rcond=None); by, *_ = np.linalg.lstsq(A, xs.imag, rcond=None)
    return np.array([bx, by])


def resid(x, xs, P):
    A = np.column_stack([x.real, x.imag, np.ones(len(x))]); f = A @ P.T
    return xs - (f[:, 0] + 1j * f[:, 1])


def stats(r):
    a = np.abs(r - r.mean())
    return dict(radius=float(a.max()), q50=float(np.quantile(a, .5)), q99=float(np.quantile(a, .99)))


def main():
    out = sys.argv[1]; Ks = [int(k) for k in sys.argv[2:]]
    data = {K: coords(K) for K in Ks}; ref = fit(*data[Ks[0]]); res = dict(M_ref=ref.tolist(), rows=[])
    for K in Ks:
        x, xs = data[K]; own = fit(x, xs)
        row = dict(K=K, phys_radius=float(np.abs(x).max()), M_own=own.tolist(),
                   r_own=stats(resid(x, xs, own)), r_ref=stats(resid(x, xs, ref)),
                   svd_M=np.linalg.svd(own[:, :2], compute_uv=False).tolist())
        print(json.dumps(row), flush=True); res["rows"].append(row)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
