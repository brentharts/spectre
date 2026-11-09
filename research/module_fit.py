"""Γ8 follow-up to peak_index v1 (top peaks not in the zeta -> zeta^5 Fourier module).

(a) Check that spectre_graph positions are exact integer zeta-coordinates (raw edge lengths, integrality).
(b) Refine the top FFT peaks of the vertex and vortex sets by direct summation S(k) = sum exp(-i k.x)
    on a fine local grid, giving k to ~1e-4, and re-index them with the zeta^5 embedding.
(c) Direct structure factor at every module point (k, k*) = 2 pi E^-1 m, m in [-B, B]^4, |k| < KMAX,
    for both candidate internal maps zeta -> zeta^5 and zeta -> zeta^7.  If the embedding is right, the
    strongest of these must match the FFT top peaks (h ~ 1.3 for vortices).

Usage: python research/module_fit.py OUT.json K
"""
import sys, json, itertools
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build
from graph_dilation import ZETA
from ollivier_ricci import vortices

KMAX = 2 * np.pi * 1.5; B = 6


def sf(k, pts):
    """|sum exp(-i k.x)|^2 / N^2 for an array of k (complex) and points (complex)."""
    out = np.empty(len(k))
    for i in range(0, len(k), 256):
        kk = k[i:i + 256, None]
        ph = (kk.real * pts.real[None] + kk.imag * pts.imag[None])
        out[i:i + 256] = np.abs(np.exp(-1j * ph).sum(1)) ** 2
    return out / len(pts) ** 2


def module_points(p):
    Zj = np.exp(1j * np.pi / 6 * np.arange(4)); Zs = np.exp(1j * p * np.pi / 6 * np.arange(4))
    E = np.column_stack([Zj.real, Zj.imag, Zs.real, Zs.imag])     # E @ (kx,ky,k*x,k*y) = (k.zeta^j + k*.zeta^pj)_j
    Ei = np.linalg.inv(E)
    m = np.array(list(itertools.product(range(-B, B + 1), repeat=4)), float)
    v = 2 * np.pi * m @ Ei.T
    k = v[:, 0] + 1j * v[:, 1]; ks = v[:, 2] + 1j * v[:, 3]
    keep = (np.abs(k) < KMAX) & (np.abs(k) > 0.5)
    return k[keep], ks[keep], m[keep].astype(int)


def main():
    out, K = sys.argv[1], int(sys.argv[2])
    _, _, V, _ = build(K)
    flat = V.reshape(-1, 4); uniq, vid = np.unique(flat, axis=0, return_inverse=True)
    z0 = uniq.astype(float) @ ZETA
    vid = vid.reshape(V.shape[0], 14)
    L = np.abs(z0[vid] - z0[np.roll(vid, -1, 1)]).ravel()
    res = dict(K=K, dtype=str(uniq.dtype), coord_abs_max=int(np.abs(uniq).max()),
               raw_edge_lengths=sorted(set(np.round(L, 6).tolist()))[:10], median_edge=float(np.median(L)))
    print(res, flush=True)
    z = z0 / np.median(L); c = z.mean(); Rr = np.abs(z - c); keep = Rr < 0.8 * Rr.max()
    v = vortices(K, len(z))
    sets = {"vertices": (z - c)[keep], "vortex": (z - c)[keep & v]}
    for p in (5, 7):
        k, ks, m = module_points(p)
        for name, pts in sets.items():
            h = sf(k, pts); o = np.argsort(h)[::-1][:20]
            res[f"zeta{p}_{name}"] = [dict(m=m[i].tolist(), k=float(abs(k[i])), kx=float(k[i].real), ky=float(k[i].imag),
                                           kstar=float(abs(ks[i])), h=float(h[i])) for i in o]
            print(p, name, len(k), [(round(abs(k[i]), 3), round(abs(ks[i]), 2), round(h[i], 3)) for i in o[:8]], flush=True)
    # (b) refine FFT-style top peaks by a coarse direct scan then a fine local scan
    for name, pts in sets.items():
        g = np.linspace(-KMAX, KMAX, 361); KX, KY = np.meshgrid(g, g)
        kk = (KX + 1j * KY).ravel(); kk = kk[(np.abs(kk) < KMAX) & (np.abs(kk) > 0.5)]
        sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20000), replace=False)]
        h = sf(kk, sub); top = []
        for i in np.argsort(h)[::-1]:
            if all(abs(kk[i] - t) > 0.3 and abs(kk[i] + t) > 0.3 for t in top): top.append(kk[i])
            if len(top) >= 8: break
        ref = []
        for k0 in top:
            for step in (0.02, 0.002, 0.0002):
                d = np.arange(-10, 11) * step; DX, DY = np.meshgrid(d, d)
                cand = (k0 + DX + 1j * DY).ravel(); hh = sf(cand, pts); k0 = cand[np.argmax(hh)]
            ref.append(dict(kx=float(k0.real), ky=float(k0.imag), k=float(abs(k0)), h=float(hh.max())))
        res[f"refined_{name}"] = ref
        print("refined", name, [(round(r['kx'], 4), round(r['ky'], 4), round(r['h'], 3)) for r in ref], flush=True)
        json.dump(res, open(out, "w"), indent=1)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
