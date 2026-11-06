"""Floating-point upper bound for dim_B of the window boundary.  A point of the boundary of the window of a tile (S1, A1)
is a limit of internal positions where that tile is absent; since every tiling covers the plane, some other tile
(S2, A2) at offset v then overlaps it, so its window touches this one.  Seeds = such overlapping placements with
windows within 2 RAD; the contact closure from them, by spectre_window.children, bounds the box count of the boundary.
Frames and window offsets come from an affine fit against a level-N patch (window_vs_patch)."""
import io, contextlib, sys, itertools
import numpy as np
import mpmath as mp
with contextlib.redirect_stdout(io.StringIO()):
    import spectre_window as sw
from spectre_exact import slot_transforms, SPECTRE, mul, apply, IDENT, SUPER_RULES, ttrans, trot, to_c
from shapely.geometry import Polygon
from shapely.ops import unary_union
lam = float(sw.LAM); RAD = float(sw.RAD)
O = {k: np.array(sw.O[k].tolist(), dtype=float) for k in sw.O}
GA = {k: np.array(sw.GA[k].tolist(), dtype=float).ravel() for k in sw.GA}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
quads = [[SPECTRE[3], SPECTRE[5], SPECTRE[7], SPECTRE[11]]]; TS = [None]
for j in range(N):
    Ts, nq = slot_transforms(quads[-1]); TS.append(Ts); quads.append(nq)
Bm = mp.matrix(4, 4)
for c, v in enumerate(sw.E[0] + sw.F[0]):
    for r in range(4): Bm[r, c] = v[r]
Bim = mp.inverse(Bm)
STARm = sw.T[0] * mp.matrix([[Bim[r, c] for c in range(4)] for r in (2, 3)])   # exact to mp.dps (60 digits)
STAR = np.array(STARm.tolist(), dtype=float)       # 2x4: internal coordinates of a lattice vector
# The closure keys round to 1e-30/1e-40, and children() multiplies offset errors by lam, so seeds must carry the
# full working precision: with float seeds every rounding variant becomes a new type and the closure never closes.
groups = {}
def rec(s, j, T, z):
    if j == 0:
        g = groups.setdefault((tuple(T[0].flatten()), 'G' if s == 'Gamma' else 'O'), [T[0], [], []])
        g[1].append(STARm * mp.matrix([mp.mpf(int(x)) for x in T[1]])); g[2].append(z); return
    p = (j - 1) % 2
    for i, c in enumerate(SUPER_RULES[s]):
        if c is not None:
            rec(c, j - 1, mul(T, TS[j][i]), sw.O[(i, p)].T * (sw.GA[(i, p)] + z / sw.LAM))
rec('Delta', N, IDENT, mp.matrix(2, 1))
FR, FRm = [], []
for (fk, S), (A, Y, Z) in groups.items():
    X = mp.matrix([[z[0], z[1], 1] for z in Z]); Ym = mp.matrix([[y[0], y[1]] for y in Y])
    NX, B2 = X.T * X, X.T * Ym                            # lu_solve takes one column at a time
    cols = [mp.lu_solve(NX, B2[:, k]) for k in range(2)]
    sol = mp.matrix([[cols[k][i] for k in range(2)] for i in range(3)])  # 3x2: rows of R^T, then c
    R, c = sol[0:2, 0:2].T, sol[2, 0:2].T
    res = max(abs(x) for x in (X * sol - Ym)); G = R.T * R; s2 = G[0, 0]
    assert res < mp.mpf(10) ** -40 and max(abs(x) for x in (G - s2 * mp.eye(2))) < mp.mpf(10) ** -40, (res, G)
    sc = mp.sqrt(s2); Rm = R / sc
    FRm.append((Rm, c, sc))
    FR.append((S, A, np.array(Rm.tolist(), dtype=float), np.array(c.tolist(), dtype=float).ravel(), float(sc)))
scale = FR[0][4]; assert all(abs(f[2] - FRm[0][2]) < mp.mpf(10) ** -40 for f in FRm)
print('window classes (shape, frame):', len(FR), ' tiles', sum(len(g[1]) for g in groups.values()), ' similarity ratio', scale, flush=True)
LEAVES = {'O': [IDENT], 'G': [IDENT, mul(ttrans(SPECTRE[8]), trot(1))]}
def shape(S, A, t):
    return unary_union([Polygon([(lambda w: (w.real, w.imag))(to_c(apply(mul((A, t), L), v))) for v in SPECTRE])
                        for L in LEAVES[S]])
# lattice vectors with small physical and internal parts
PH = np.array([[to_c(np.eye(4, dtype=object)[k]).real for k in range(4)], [to_c(np.eye(4, dtype=object)[k]).imag for k in range(4)]])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 14
rng = np.arange(-K, K + 1)
V = np.array(list(itertools.product(rng, rng, rng, rng)))
ph, it = V @ PH.T, V @ STAR.T
DPH = 12.0                                         # > twice the Mystic's diameter, in edge units
cand = V[(np.linalg.norm(ph, axis=1) < DPH) & (np.linalg.norm(it, axis=1) < 2 * RAD * scale + 1e-9 + 2 * max(np.linalg.norm(f[3]) for f in FR))]
print('lattice candidates', len(cand), ' (box edge %d; boundary hits %d)' % (K, int((np.abs(cand) == K).any(1).sum())), flush=True)
z4 = np.zeros(4, dtype=object)
def pair_seeds(ab):
    (S1, A1, R1, c1, _), (S2, A2, R2, c2, _) = FR[ab[0]], FR[ab[1]]
    P1 = shape(S1, A1, z4); out = []
    for v in cand:
        d = R1.T @ ((c2 - c1 - STAR @ v) / scale)
        if np.linalg.norm(d) > 2 * RAD + 1e-9:
            continue
        vo = np.array([int(x) for x in v], dtype=object)
        if shape(S2, A2, vo).intersection(P1).area < 1e-6:
            continue
        out.append((S1, S2, ab, tuple(int(x) for x in v)))
    return out
import multiprocessing as mpr, time
t0 = time.time()
with mpr.Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 1) as pool:
    res = pool.map(pair_seeds, [(a, b) for a in range(len(FR)) for b in range(len(FR))])
seeds = {}
for S1, S2, (a, b), v in (x for r in res for x in r):
    (R1, c1, sc), (R2, c2, _) = FRm[a], FRm[b]
    t = (S1, S2, 0, R1.T * R2, R1.T * ((c2 - c1 - STARm * mp.matrix([mp.mpf(x) for x in v])) / sc))
    seeds[sw.tkey(*t)] = t
print('seed search %.0fs' % (time.time() - t0), flush=True)
print('competitor seeds (overlapping placements, windows within 2 RAD):', len(seeds), flush=True)
I2 = sw.key2(mp.eye(2))
print('identity seeds (would mean two windows coincide):', sum(1 for t in seeds.values() if t[0] == t[1] and sw.key2(t[3]) == I2 and mp.norm(t[4]) < 1e-6), flush=True)
live, stack = {}, list(seeds.values())
while stack:
    t = stack.pop()
    if mp.norm(t[4]) > 2 * sw.RAD + mp.mpf(10) ** -20: continue
    k = sw.tkey(*t)
    if k in live: continue
    live[k] = t; stack.extend(sw.children(*t)); assert len(live) < (int(sys.argv[4]) if len(sys.argv) > 4 else 2000000)
    if len(live) % 20000 == 0: print('closure', len(live), flush=True)
keys = list(live); idx = {k: n for n, k in enumerate(keys)}
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
ent = {}
for a, k in enumerate(keys):
    for ch in sw.children(*live[k]):
        kk = sw.tkey(*ch)
        if kk in idx: ent[a, idx[kk]] = ent.get((a, idx[kk]), 0) + 1
Nm = coo_matrix((list(ent.values()), ([a for a, _ in ent], [b for _, b in ent])), shape=(len(keys),) * 2).tocsr()
rho = max(abs(np.linalg.eigvals(Nm.toarray()))) if len(keys) <= 3000 else max(abs(eigs(Nm.astype(float), k=6, which='LM', return_eigenvectors=False)))
print('boundary closure: %d live types, spectral radius %.10f (sqrt2+sqrt3 = %.10f), dim bound %.6f (seam %.6f)'
      % (len(keys), rho, np.sqrt(2) + np.sqrt(3), np.log(rho) / np.log(lam), np.log(np.sqrt(2) + np.sqrt(3)) / np.log(lam)))
