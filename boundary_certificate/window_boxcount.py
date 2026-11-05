"""Heuristic: box-counting dimension of the boundary of each window Z_p(S), from a deep point cloud of the
graph-directed system in spectre_window.py.  Not a proof.  Prediction if the seam bound is sharp:
log(sqrt2+sqrt3)/log(lam) = 1.1109772641567."""
import io, contextlib, sys
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import spectre_window as sw
lam = float(sw.LAM)
O = {k: np.array(sw.O[k].tolist(), dtype=float) for k in sw.O}
GA = {k: np.array(sw.GA[k].tolist(), dtype=float).ravel() for k in sw.GA}
DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 7
P = {(S, p): np.zeros((1, 2)) for S in 'OG' for p in (0, 1)}
for m in range(DEPTH):
    Q = {}
    for (S, p), maps in sw.MAPS.items():
        parts = []
        for (i, S2) in maps:
            Oi = O[(i, p)]
            parts.append(((GA[(i, p)] + P[(S2, 1 - p)] / lam) @ Oi))   # Oi^{-1} = Oi^T, row vectors
        Q[(S, p)] = np.concatenate(parts)
    P = Q
    print('depth', m + 1, {k: len(v) for k, v in P.items()}, flush=True)
slope_pred = np.log(np.sqrt(2) + np.sqrt(3)) / np.log(lam)
for key, pts in P.items():
    lo = pts.min(0)
    span = (pts.max(0) - lo).max()
    rows = []
    for k in range(2, DEPTH - 1):     # boxes at least lam^2 times the piece size, so the cloud has no gaps
        h = span * lam ** (-k) / 2
        ij = np.floor((pts - lo) / h).astype(np.int64) + 1
        n = ij.max() + 2
        g = np.zeros((n, n), dtype=bool)
        g[ij[:, 0], ij[:, 1]] = True
        inner = g[1:-1, 1:-1] & g[:-2, 1:-1] & g[2:, 1:-1] & g[1:-1, :-2] & g[1:-1, 2:]
        nb = int(g[1:-1, 1:-1].sum() - inner.sum())
        rows.append((h, nb, g.sum() * h * h))
        print(key, 'h=%.3e boundary boxes %d area %.5f' % rows[-1], flush=True)
    hs = np.log([1 / r[0] for r in rows]); ns = np.log([r[1] for r in rows])
    loc = np.diff(ns) / np.diff(hs)
    print(key, 'local slopes', np.round(loc, 4), 'fit', round(np.polyfit(hs, ns, 1)[0], 4),
          'pred seam', round(slope_pred, 4), 'dim1', 1.0, flush=True)
