"""Check: for each tile of a level-n supertile, its window coordinate z (from its slot path, spectre_window's
graph-directed system) and the internal projection y of its physical position are related by a fixed affine map per
frame, y = R_A z + c_A, with R_A orthogonal.  Residuals printed."""
import io, contextlib, sys
import numpy as np
import mpmath as mp
with contextlib.redirect_stdout(io.StringIO()):
    import spectre_window as sw
from spectre_exact import slot_transforms, SPECTRE, mul, IDENT, SUPER_RULES, ttrans, trot
lam = float(sw.LAM)
O = {k: np.array(sw.O[k].tolist(), dtype=float) for k in sw.O}
GA = {k: np.array(sw.GA[k].tolist(), dtype=float).ravel() for k in sw.GA}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
quads = [[SPECTRE[3], SPECTRE[5], SPECTRE[7], SPECTRE[11]]]
TS = [None]
for j in range(N):
    Ts, nq = slot_transforms(quads[-1]); TS.append(Ts); quads.append(nq)
# projection of a physical position (integer 4-vector) to F_0 orthonormal coords
Bm = mp.matrix(4, 4)
for c, v in enumerate(sw.E[0] + sw.F[0]):
    for r in range(4): Bm[r, c] = v[r]
Binv = np.array(mp.inverse(Bm).tolist(), dtype=float)[2:]
T0 = np.array(sw.T[0].tolist(), dtype=float)
def star(t): return T0 @ (Binv @ np.array([float(x) for x in t]))
SP = {'Gamma': 'G'}
recs = []
def rec(s, j, T, z):
    """species s at level j placed by T, window coordinate z (in that supertile's own frame)"""
    if j == 0:
        recs.append((T, z, s)); return
    p = (j - 1) % 2
    for i, c in enumerate(SUPER_RULES[s]):
        if c is None: continue
        rec(c, j - 1, mul(T, TS[j][i]), O[(i, p)].T @ (GA[(i, p)] + z / lam))
rec('Delta', N, IDENT, np.zeros(2))
print('tiles', len(recs))
groups = {}
for T, z, s in recs:
    groups.setdefault(tuple(T[0].flatten()), []).append((star(T[1]), z))
worst = 0
for k, v in groups.items():
    Y = np.array([a for a, b in v]); Zs = np.array([b for a, b in v])
    X = np.hstack([Zs, np.ones((len(v), 1))])
    sol, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = np.abs(X @ sol - Y).max(); R = sol[:2].T
    worst = max(worst, res)
    print('frame', len(v), 'tiles: affine residual %.2e, |R^T R - I| %.2e, c=%s' % (res, np.abs(R.T @ R - np.eye(2)).max(), np.round(sol[2], 6)))
print('WORST affine residual %.2e' % worst)
