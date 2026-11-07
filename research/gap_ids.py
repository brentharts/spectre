"""Item C, step 3 (R9): do the spectral gaps of a tight-binding Hamiltonian on the Spectre sit at gap labels
in the frequency module (5 - sqrt15)^{-1} Z[sqrt15] = Z + Z*alpha, alpha = (5 + sqrt15)/10?

Sites are the spectre pieces of a level-K Delta supertile (every tile one piece, the Mystic two), built exactly over
Z[zeta_12] by gap_collared.build.  Hopping t = 1 between pieces that share an edge (the "tile model"); optionally
also a "vertex model" on the vertices with hopping along tile edges.  The spectrum is diagonalised densely.

Finite patches have edge states that fill gaps.  We therefore use the bulk-projected integrated density of states
    IDS_bulk(E) = sum_{E_k < E} sum_{i in bulk} |psi_k(i)|^2 / |bulk|,
where bulk = sites at graph distance >= R from any boundary site.  A gap is an interval between consecutive
eigenvalues carrying bulk weight, of width >= GMIN (relative to the bandwidth).  For each gap we report IDS_bulk and
the label m + n*alpha with |n| <= NMAX closest to it, and the residual.  Gap labelling predicts residual -> 0 as K
grows; a residual that does not shrink, or labels that change between K, is a negative result.
Normalisation: per spectre piece, matching the per-spectre frequencies u of gap_freq.py / gap_collared.py.

Usage: python research/gap_ids.py OUT.json K [K ...]
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from math import sqrt
from collections import deque

ALPHA = (5 + sqrt(15)) / 10
R_BULK = 3
GMIN = 0.004      # minimal gap width / bandwidth
WMIN = 1e-3       # an eigenvector "carries bulk weight" if its bulk mass / mean bulk mass exceeds this
NMAX = 12


def graph(K):
    from gap_collared import build
    names, levels, V, PT = build(K)
    npc = len(PT)
    _, vid = np.unique(V.reshape(-1, 4), axis=0, return_inverse=True)
    vid = vid.reshape(npc, 14)
    e = np.sort(np.stack([vid, np.roll(vid, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(npc), 14)
    _, einv, ecnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bd = np.zeros(npc, bool); bd[owner[ecnt[einv] == 1]] = True
    order = np.argsort(einv, kind="stable")
    se, so = einv[order], owner[order]
    same = se[1:] == se[:-1]
    a, b = so[:-1][same], so[1:][same]
    keep = a != b
    pairs = np.unique(np.sort(np.stack([a[keep], b[keep]], 1), axis=1), axis=0)
    # vertex model: vertices, hopping along distinct tile edges
    ue = np.unique(e, axis=0)
    vbd = np.zeros(vid.max() + 1, bool); vbd[e[ecnt[einv] == 1].ravel()] = True
    return dict(n=npc, pairs=pairs, bd=bd, nv=int(vid.max() + 1), vpairs=ue, vbd=vbd)


def distance_from(n, pairs, src):
    adj = [[] for _ in range(n)]
    for x, y in pairs:
        adj[x].append(y); adj[y].append(x)
    d = np.full(n, -1); q = deque(np.nonzero(src)[0].tolist())
    d[src] = 0
    while q:
        x = q.popleft()
        for y in adj[x]:
            if d[y] < 0:
                d[y] = d[x] + 1; q.append(y)
    return d


def best_label(x):
    best = None
    for nn in range(-NMAX, NMAX + 1):
        m = round(x - nn * ALPHA)
        r = x - (m + nn * ALPHA)
        if best is None or abs(r) < abs(best[2]) - 1e-15:
            best = (int(m), nn, r)
    return best


def analyse(n, pairs, bdmask):
    H = np.zeros((n, n))
    H[pairs[:, 0], pairs[:, 1]] = 1.0; H[pairs[:, 1], pairs[:, 0]] = 1.0
    t = time.time()
    E, U = np.linalg.eigh(H)
    del H
    dist = distance_from(n, pairs, bdmask)
    bulk = dist >= R_BULK
    nb = int(bulk.sum())
    w = (U[bulk] ** 2).sum(0)               # bulk mass of each eigenvector
    del U
    ids = np.cumsum(w) / nb                  # IDS_bulk just above E_k
    band = E[-1] - E[0]
    carry = np.nonzero(w > WMIN * nb / n)[0]
    gaps = []
    for i, j in zip(carry[:-1], carry[1:]):
        width = E[j] - E[i]
        if width >= GMIN * band:
            x = float(ids[j - 1])            # bulk weight below the gap (edge states inside count ~0)
            m, nn, r = best_label(x)
            gaps.append(dict(lo=float(E[i]), hi=float(E[j]), width=float(width), ids=x,
                             edge_weight_in_gap=float(ids[j - 1] - ids[i]),
                             label=[m, nn], residual=float(r)))
    return dict(sites=n, bulk_sites=nb, bandwidth=float(band), eig_seconds=round(time.time() - t, 1),
                n_bulk_carrying=int(len(carry)), gaps=gaps)


def main():
    out = sys.argv[1]
    Ks = [int(k) for k in sys.argv[2:]] or [3, 4]
    res = dict(alpha=ALPHA, R_bulk=R_BULK, gmin=GMIN, nmax=NMAX, levels={})
    for K in Ks:
        g = graph(K)
        print("K", K, "pieces", g["n"], "vertices", g["nv"], flush=True)
        r = dict(tile=analyse(g["n"], g["pairs"], g["bd"]))
        if g["nv"] <= 40000:
            r["vertex"] = analyse(g["nv"], g["vpairs"], g["vbd"])
        res["levels"][K] = r
        for model, d in r.items():
            print(K, model, d["sites"], "bulk", d["bulk_sites"], "gaps", len(d["gaps"]), flush=True)
            for gp in d["gaps"]:
                print("   [%.4f, %.4f] ids=%.6f label=%s res=%.2e" % (gp["lo"], gp["hi"], gp["ids"], gp["label"], gp["residual"]), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
