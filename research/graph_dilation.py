"""H1: does graph distance on the Spectre vertex graph reproduce Euclidean distance?

For random source vertices deep in the bulk, BFS gives graph distance d_G to every vertex; the ratio
d_G / |x - y| (Euclidean, edge length 1) is binned by direction and by Euclidean distance. On a periodic
lattice the ratio tends to a direction-dependent constant (square: |cos|+|sin| over unit steps); we test whether
the Spectre limit exists and is isotropic.  Controls: square lattice and honeycomb, edge length 1.
Usage: python research/graph_dilation.py OUT.json K [n_sources]
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from collections import deque
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

ZETA = np.exp(1j * np.pi / 6) ** np.arange(4)


def spectre_graph(K):
    from gap_collared import build
    _, _, V, _ = build(K)
    flat = V.reshape(-1, 4)
    uniq, vid = np.unique(flat, axis=0, return_inverse=True)
    vid = vid.reshape(V.shape[0], 14)
    z = uniq.astype(float) @ ZETA
    e = np.stack([vid, np.roll(vid, -1, 1)], -1).reshape(-1, 2)
    e = np.unique(np.sort(e, 1), axis=0)
    L = np.abs(z[e[:, 0]] - z[e[:, 1]])
    return z / np.median(L), e, L / np.median(L)


def square(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); idx = i * n + j
    e = np.concatenate([np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1), np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)])
    return (i + 1j * j).ravel().astype(complex), e


def honeycomb(n):
    a1, a2 = np.sqrt(3), np.sqrt(3) * np.exp(1j * np.pi / 3)
    pts, idx = [], {}
    for i in range(n):
        for j in range(n):
            for s, off in ((0, 0), (1, 1j)):
                idx[(i, j, s)] = len(pts); pts.append(i * a1 + j * a2 + off)
    e = []
    for (i, j, s), k in idx.items():
        if s == 0:
            for d in ((0, 0), (1, -1), (0, -1)):
                t = (i + d[0], j + d[1], 1)
                if t in idx: e.append((k, idx[t]))
    # check bonds have length 1
    P = np.array(pts); E = np.array(e)
    E = E[np.isclose(np.abs(P[E[:, 0]] - P[E[:, 1]]), 1)]
    return P, E


def analyse(z, e, name, nsrc, rng):
    n = len(z)
    A = csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    c = z.mean(); R = np.abs(z - c); Rmax = np.percentile(R, 95)
    cand = np.nonzero(R < 0.3 * Rmax)[0]
    src = rng.choice(cand, min(nsrc, len(cand)), replace=False)
    D = shortest_path(A, unweighted=True, indices=src)
    rbins = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024]); abins = np.linspace(-np.pi, np.pi, 25)
    out = dict(graph=name, vertices=n, edges=int(len(e)), sources=int(len(src)), by_r=[], by_angle_far=[])
    rows = []
    for s, d in zip(src, D):
        dz = z - z[s]; r = np.abs(dz); ok = np.isfinite(d) & (r > 0) & (np.abs(z - c) < 0.9 * Rmax)
        rows.append(np.stack([r[ok], np.angle(dz[ok]), d[ok] / r[ok]], 1))
    X = np.concatenate(rows)
    for lo, hi in zip(rbins[:-1], rbins[1:]):
        m = (X[:, 0] >= lo) & (X[:, 0] < hi)
        if m.sum() > 50:
            q = X[m, 2]
            out["by_r"].append(dict(r=[int(lo), int(hi)], n=int(m.sum()), mean=float(q.mean()), std=float(q.std()),
                                    q05=float(np.quantile(q, .05)), q95=float(np.quantile(q, .95))))
    far = X[X[:, 0] >= np.quantile(X[:, 0], 0.75)]
    means = []
    for lo, hi in zip(abins[:-1], abins[1:]):
        m = (far[:, 1] >= lo) & (far[:, 1] < hi)
        if m.sum():
            means.append(float(far[m, 2].mean()))
            out["by_angle_far"].append(dict(angle_deg=float(np.degrees((lo + hi) / 2)), n=int(m.sum()), mean=means[-1]))
    means = np.array(means)
    out["anisotropy_far"] = float((means.max() - means.min()) / means.mean())
    print(json.dumps({k: out[k] for k in ("graph", "vertices", "anisotropy_far")}), out["by_r"][-1] if out["by_r"] else None, flush=True)
    return out


def main():
    outp = sys.argv[1]; K = int(sys.argv[2]); nsrc = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    rng = np.random.default_rng(0)
    z, e, L = spectre_graph(K)
    res = dict(K=K, spectre_edge_lengths=sorted(set(np.round(L, 4).tolist()))[:10], runs=[])
    res["runs"].append(analyse(z, e, f"spectre_K{K}", nsrc, rng))
    side = int(np.sqrt(len(z)))
    res["runs"].append(analyse(*square(side), f"square_{side}", nsrc, rng))
    res["runs"].append(analyse(*honeycomb(int(side / np.sqrt(2))), "honeycomb", nsrc, rng))
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
