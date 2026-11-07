"""Item 5: an exact, linear swirl invariant.  For the level-k Delta supertile (tiles mirrored to one
handedness), Z_k = sum_i conj(p_i - c) exp(i theta_i), with p_i tile centroids, c their mean, theta_i the
placement angle.  Im Z is the 'angular momentum' of the orientation field (sign of the swirl), and Z_k is
covariant under a constant angle offset.  Z_k obeys a linear recurrence fixed by the substitution; we fit
it and compare its roots with the spectra of the substitution matrix and the lift Phi."""
import sys, json; sys.path.insert(0, ".")
import numpy as np, spectre_patch as spm
out = []
for k in range(1, int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 7):
    P = spm.SpectrePatch(k)
    pts = np.array([P.straight_polygon(i).mean(0) for i in range(P.n_tiles)])
    z = pts[:, 0] + 1j * pts[:, 1]; z -= z.mean()
    th = np.arctan2(P.placements[:, 1, 0], P.placements[:, 0, 0])
    Z = np.sum(np.conj(z) * np.exp(1j * th)); A = np.sum(np.abs(z))
    sw = np.mean(np.sin(th - np.angle(z)))
    out.append(dict(k=k, n=int(P.n_tiles), ReZ=Z.real, ImZ=Z.imag, absZ=abs(Z), sumR=A, ImZ_over_sumR=Z.imag / A, swirl=sw))
    print(json.dumps(out[-1]), flush=True)
a = np.array([o["absZ"] for o in out]); n = np.array([o["n"] for o in out])
print("ratios |Z_k+1|/|Z_k|:", (a[1:] / a[:-1]).round(5).tolist())
print("ratios n_k+1/n_k   :", (n[1:] / n[:-1]).round(5).tolist())
c = np.array([o["ReZ"] + 1j * o["ImZ"] for o in out])
for order in (1, 2, 3):
    if len(c) > 2 * order:
        M = np.array([c[i:i + order] for i in range(len(c) - order)]); y = c[order:]
        co, res, *_ = np.linalg.lstsq(M, y, rcond=None)
        roots = np.roots(np.r_[1, -co[::-1]])
        print("order", order, "residual", np.abs(M @ co - y).max() / np.abs(y).max(), "roots", np.round(roots, 5).tolist(), "|roots|", np.round(np.abs(roots), 5).tolist())
print("lambda=%.6f lambda^2=%.6f sqrt2+sqrt3=%.6f" % (np.sqrt(4 + np.sqrt(15)), 4 + np.sqrt(15), np.sqrt(2) + np.sqrt(3)))
