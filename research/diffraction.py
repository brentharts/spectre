"""Γ8 foothold: diffraction of the Spectre vertex set and of the holonomy vortex subset.

Each point set is restricted to a disc of radius 0.8 * R_max about the patch centre (edge length = 1),
binned onto an M x M grid, and FFT'd.  I(k) = |sum_x exp(-i k.x)|^2.  The binning multiplies I by a smooth
form factor that only matters near the grid Nyquist, so only |k| < k_max (well below Nyquist) is used.

Bragg test: a Bragg peak has I(k) proportional to N^2, diffuse scattering has I proportional to N.  So the
top peak heights h = I/N^2 should stay fixed from K=3 to K=4 for pure point components, and fall ~1/N for
diffuse ones.  We also record the fraction of total intensity (|k| < k_max, k != 0) in the top peaks.

Usage: python research/diffraction.py OUT.json K [K ...]
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from graph_dilation import spectre_graph
from ollivier_ricci import vortices

M = 4096          # FFT grid
KMAX = 2 * np.pi * 1.5   # reciprocal radius used (edge length 1, so 1.5 cycles per edge)
NPEAK = 40


def pattern(pts, L):
    """I on the FFT grid for points in [-L, L]^2; returns I and the k-grid spacing."""
    h = 2 * L / M
    ix = np.clip(((pts.real + L) / h).astype(int), 0, M - 1); iy = np.clip(((pts.imag + L) / h).astype(int), 0, M - 1)
    rho = np.zeros((M, M), np.float32); np.add.at(rho, (ix, iy), 1)
    F = np.fft.fftshift(np.fft.fft2(rho)); del rho
    return (np.abs(F) ** 2).astype(np.float64), 2 * np.pi / (2 * L)


def peaks(I, dk, n, N):
    c = M // 2; r = int(KMAX / dk) + 2
    S = I[c - r:c + r + 1, c - r:c + r + 1].copy()
    kx, ky = np.meshgrid(np.arange(-r, r + 1) * dk, np.arange(-r, r + 1) * dk, indexing="ij")
    kk = np.hypot(kx, ky); S[(kk > KMAX)] = 0; S[r, r] = 0          # drop k = 0 (forward peak, = N^2)
    # local maxima in 3x3 neighbourhood
    P = np.pad(S, 1); loc = np.ones_like(S, bool)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a or b: loc &= S >= P[1 + a:1 + a + S.shape[0], 1 + b:1 + b + S.shape[1]]
    # integrate each local maximum over its 3x3 block (bins split the delta)
    B = sum(np.roll(np.roll(S, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1))
    cand = np.nonzero(loc & (S > 0))
    order = np.argsort(B[cand])[::-1][:n]
    tot = S.sum()
    out = [dict(kx=float(kx[cand][i]), ky=float(ky[cand][i]), k=float(kk[cand][i]), h=float(B[cand][i] / N ** 2)) for i in order]
    return out, float(sum(B[cand][order]) / tot), float(tot / N ** 2)


def analyse(z, sets, K):
    c = z.mean(); R = np.abs(z - c); Rb = 0.8 * R.max(); keep = R < Rb
    L = Rb * 1.05; res = dict(K=K, radius=float(Rb))
    for name, mask in sets.items():
        pts = (z - c)[keep & mask]
        I, dk = pattern(pts, L)
        if 2 * L / M > 0.3: print('warning: grid too coarse for KMAX', 2 * L / M, flush=True)
        pk, frac, tot = peaks(I, dk, NPEAK, len(pts)); del I
        res[name] = dict(N=int(len(pts)), dk=float(dk), top_peaks=pk, top_fraction=frac, total_over_N2=tot)
        print(K, name, len(pts), "top h", [round(p["h"], 4) for p in pk[:6]], "frac", round(frac, 3), flush=True)
    return res


def main():
    out = sys.argv[1]; Ks = [int(k) for k in sys.argv[2:]] or [3]; res = []
    for K in Ks:
        t = time.time(); z, e, _ = spectre_graph(K); v = vortices(K, len(z))
        res.append(analyse(z, {"vertices": np.ones(len(z), bool), "vortex": v, "nonvortex": ~v}, K))
        res[-1]["seconds"] = round(time.time() - t, 1)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
