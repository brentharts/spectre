"""Open question of Baake-Gaehler-Mazac-Sadun: is the boundary dimension uniform along dW?
Box-count boundary cells of the internal-space image (window) separately in 12 angular sectors about the
window centroid and fit a slope per sector.  Uniform => all sectors near log(sqrt2+sqrt3)/log(lambda)=1.110977."""
import sys, json; sys.path.insert(0, ".")
import numpy as np, lift4d
L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
S = lift4d.bounded_plane(4)[0]
_, lifted, _ = lift4d.lift_patch(L)
X = np.array(list(lifted.values()), dtype=float) @ S.T
X -= X.mean(0); print("points", len(X), "extent", np.ptp(X, 0).round(3).tolist(), flush=True)
hs = np.ptp(X[:, 0]) * np.array([1/16, 1/32, 1/64, 1/128, 1/256])
NS = 12; counts = np.zeros((len(hs), NS)); tot = []
for a, h in enumerate(hs):
    st = set(map(tuple, np.unique(np.floor(X / h).astype(int), axis=0)))
    bd = [c for c in st if any((c[0]+d0, c[1]+d1) not in st for d0, d1 in ((1,0),(-1,0),(0,1),(0,-1)))]
    B = (np.array(bd) + 0.5) * h
    sec = ((np.arctan2(B[:, 1], B[:, 0]) + np.pi) / (2*np.pi) * NS).astype(int) % NS
    counts[a] = np.bincount(sec, minlength=NS); tot.append(len(bd))
    print("h=%.5g occupied=%d boundary=%d" % (h, len(st), len(bd)), flush=True)
fit = lambda y: -np.polyfit(np.log(hs[1:]), np.log(np.maximum(y[1:], 1)), 1)[0]
print("global slope (h[1:]) %.4f   target 1.110977" % fit(np.array(tot, float)))
sl = [fit(counts[:, s]) for s in range(NS)]
print("sector slopes", np.round(sl, 4).tolist()); print("spread max-min %.4f, std %.4f" % (np.ptp(sl), np.std(sl)))
print("note: a saturating finite patch biases small-h counts; compare levels", L-1, "and", L)
