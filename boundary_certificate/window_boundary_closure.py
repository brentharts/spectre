"""Heuristic upper bound for dim_B of the window boundary.  A point of the boundary of the window of a tile (S1, A1)
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
Binv = np.array(mp.inverse(Bm).tolist(), dtype=float)[2:]
T0 = np.array(sw.T[0].tolist(), dtype=float)
STAR = T0 @ Binv                                   # 2x4: internal coordinates of a lattice vector
groups = {}
def rec(s, j, T, z):
    if j == 0:
        g = groups.setdefault((tuple(T[0].flatten()), 'G' if s == 'Gamma' else 'O'), [T[0], [], []])
        g[1].append(STAR @ T[1].astype(float)); g[2].append(z); return
    p = (j - 1) % 2
    for i, c in enumerate(SUPER_RULES[s]):
        if c is not None:
            rec(c, j - 1, mul(T, TS[j][i]), O[(i, p)].T @ (GA[(i, p)] + z / lam))
rec('Delta', N, IDENT, np.zeros(2))
FR = []
for (fk, S), (A, Y, Z) in groups.items():
    Y, Z = np.array(Y), np.array(Z)
    X = np.hstack([Z, np.ones((len(Z), 1))])
    sol, *_ = np.linalg.lstsq(X, Y, rcond=None)
    R, c = sol[:2].T, sol[2]
    res = np.abs(X @ sol - Y).max(); s2 = (R.T @ R)[0, 0]
    assert res < 1e-8 and np.abs(R.T @ R - s2 * np.eye(2)).max() < 1e-8, (res, R.T @ R)
    FR.append((S, A, R / np.sqrt(s2), c, np.sqrt(s2)))
scale = FR[0][4]; assert all(abs(f[4] - scale) < 1e-9 for f in FR)
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
        out.append((S1, S2, (R1.T @ R2).tolist(), d.tolist()))
    return out
import multiprocessing as mpr, time
t0 = time.time()
with mpr.Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 1) as pool:
    res = pool.map(pair_seeds, [(a, b) for a in range(len(FR)) for b in range(len(FR))])
seeds = {}
for S1, S2, B, d in (x for r in res for x in r):
    t = (S1, S2, 0, mp.matrix(B), mp.matrix(d))
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
    live[k] = t; stack.extend(sw.children(*t)); assert len(live) < 2000000
    if len(live) % 20000 == 0: print('closure', len(live), flush=True)
keys = list(live); idx = {k: n for n, k in enumerate(keys)}
Nm = np.zeros((len(keys), len(keys)))
for a, k in enumerate(keys):
    for ch in sw.children(*live[k]):
        kk = sw.tkey(*ch)
        if kk in idx: Nm[a, idx[kk]] += 1
rho = max(abs(np.linalg.eigvals(Nm)))
print('boundary closure: %d live types, spectral radius %.10f (sqrt2+sqrt3 = %.10f), dim bound %.6f (seam %.6f)'
      % (len(keys), rho, np.sqrt(2) + np.sqrt(3), np.log(rho) / np.log(lam), np.log(np.sqrt(2) + np.sqrt(3)) / np.log(lam)))
