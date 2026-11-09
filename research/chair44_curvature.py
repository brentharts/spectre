"""Γ6, Chair44 part: Ollivier-Ricci curvature on a planar cut of the Chair44 patch.

The cut is the z-layer with the most L-trominoes (the middle layer has only one): a tiling by 2x2 squares (4 cells) and L-trominoes (3 cells).
Chair44 has no holonomy vortices, so the defect candidate is the L-tromino (a chair cut through its notch).
Graph: the tile-adjacency graph of the cut (tiles adjacent if they share a unit cell edge).
Vertex kappa = mean of edge kappa (alpha = 0) over its edges, interior tiles only.
Tests, as for the Spectre (research/regge.py): kappa by (degree, shape), and over disjoint discs the
partial correlation of sum kappa with the tromino count, controlling for each degree-class count.

Usage: python research/chair44_curvature.py OUT.json DEPTH [DEPTH ...]
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "research")
from collections import defaultdict
import numpy as np
from ollivier_ricci import adjacency, kappa
from regge import pc
from chair44_diffraction import best_layer


def graph(depth):
    lay, _ = best_layer(depth); occ = lay >= 0
    ids, inv = np.unique(lay[occ], return_inverse=True)
    lab = -np.ones(lay.shape, int); lab[occ] = inv; n = len(ids)
    cnt = np.bincount(inv); xs, ys = np.nonzero(occ)
    z = (np.bincount(inv, xs + 0.5) + 1j * np.bincount(inv, ys + 0.5)) / cnt
    e = set()
    for A, B in ((lab[:-1], lab[1:]), (lab[:, :-1], lab[:, 1:])):
        m = (A >= 0) & (B >= 0) & (A != B)
        for a, b in zip(A[m].tolist(), B[m].tolist()): e.add((min(a, b), max(a, b)))
    # boundary tiles: touching an empty cell or the array edge
    P = np.pad(lab, 1, constant_values=-1); bd = np.zeros(n, bool)
    for dx, dy in ((0, 1), (2, 1), (1, 0), (1, 2)):
        nbv = P[dx:dx + lab.shape[0], dy:dy + lab.shape[1]]
        np.logical_or.at(bd, lab[occ & (nbv < 0)], True)
    return z, sorted(e), cnt, bd


def analyse(depth):
    z, e, size, bd = graph(depth); n = len(z)
    nb = adjacency(n, e); deg = np.array([len(x) for x in nb])
    act = ~bd; for_k = [(x, y) for x, y in e if act[x] or act[y]]
    kv = np.zeros(n); c = np.zeros(n)
    for x, y in for_k:
        k = kappa(nb, x, y, 0.0); kv[x] += k; kv[y] += k; c[x] += 1; c[y] += 1
    ok = act & (c == deg); kv = kv / np.maximum(c, 1); tro = size == 3
    res = dict(depth=depth, tiles=n, interior=int(ok.sum()), trominoes=int((ok & tro).sum()))
    tab = defaultdict(list)
    for v in np.nonzero(ok)[0]: tab[(int(deg[v]), int(size[v]))].append(kv[v])
    res["table"] = [dict(deg=d, cells=s, count=len(x), kappa_mean=float(np.mean(x)), kappa_sd=float(np.std(x)))
                    for (d, s), x in sorted(tab.items())]
    degs = sorted(set(deg[ok].tolist())); rng = np.random.default_rng(0)
    for rad in (4, 8, 16):
        idx = np.nonzero(ok)[0]; rng.shuffle(idx); cent = []
        for i in idx:
            if np.min(np.abs(z[bd] - z[i])) < rad + 2: continue
            if all(abs(z[i] - z[j]) >= 2 * rad for j in cent): cent.append(i)
            if len(cent) >= 4000: break
        rows = []
        for i in cent:
            m = ok & (np.abs(z - z[i]) < rad)
            rows.append([kv[m].sum(), (m & tro).sum()] + [(m & (deg == d)).sum() for d in degs])
        A = np.array(rows, float)
        if len(A) < 10: continue
        X = np.column_stack([np.ones(len(A)), A[:, 2:]])
        r, ci = pc(A[:, 0], A[:, 1], X)
        res[f"r{rad}"] = dict(n_discs=len(A), tromino_count=dict(r=r, ci95=ci))
    return res


def main():
    out = sys.argv[1]; res = []
    for d in [int(x) for x in sys.argv[2:]]:
        t = time.time(); res.append(analyse(d)); res[-1]["seconds"] = round(time.time() - t, 1)
        print(json.dumps(res[-1])[:1500], flush=True); json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
