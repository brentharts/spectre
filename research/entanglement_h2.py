"""H2: free-fermion correlations on the Spectre vertex graph vs square/honeycomb controls.

Half-filled hopping model H = -A (A = adjacency). Correlation matrix C = projector onto the lowest N/2 modes.
(a) Entanglement entropy S(R) of Euclidean discs of radius R about bulk centres: fit S = a R + b R log R + c.
(b) Mutual information I(A:B) between single vertices (two-site regions) vs graph distance d_G and Euclidean r;
    we report the partial correlation of log I with each, to see which distance controls the decay.
Usage: python research/entanglement_h2.py OUT.json K
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from graph_dilation import spectre_graph, square, honeycomb


def S_of(Cs):
    v = np.clip(np.linalg.eigvalsh(Cs), 1e-12, 1 - 1e-12)
    return float(-(v * np.log(v) + (1 - v) * np.log(1 - v)).sum())


def analyse(z, e, name, rng):
    n = len(z)
    A = np.zeros((n, n)); A[e[:, 0], e[:, 1]] = A[e[:, 1], e[:, 0]] = 1
    w, U = np.linalg.eigh(-A); occ = U[:, : n // 2]; C = occ @ occ.T
    c = z.mean(); Rmax = np.percentile(np.abs(z - c), 95)
    centres = rng.choice(np.nonzero(np.abs(z - c) < 0.2 * Rmax)[0], 8, replace=False)
    radii = np.linspace(2, 0.6 * Rmax, 12); disc = []
    for R in radii:
        Ss = [S_of(C[np.ix_(m, m)]) for m in (np.nonzero(np.abs(z - z[k]) < R)[0] for k in centres)]
        disc.append(dict(R=float(R), S=float(np.mean(Ss)), S_std=float(np.std(Ss))))
    Rv = np.array([d["R"] for d in disc]); Sv = np.array([d["S"] for d in disc])
    X = np.stack([Rv, Rv * np.log(Rv), np.ones_like(Rv)], 1)
    coef, res, *_ = np.linalg.lstsq(X, Sv, rcond=None)
    G = shortest_path(csr_matrix(A), unweighted=True, indices=centres)
    s1 = np.array([S_of(C[[i]][:, [i]]) for i in range(n)])
    rows = []
    for ci, k in enumerate(centres):
        far = np.nonzero((np.abs(z - c) < 0.9 * Rmax) & (G[ci] > 0) & (G[ci] <= 40))[0]
        for j in far:
            I = s1[k] + s1[j] - S_of(C[np.ix_([k, j], [k, j])])
            if I > 1e-14: rows.append((G[ci, j], abs(z[j] - z[k]), np.log(I)))
    M = np.array(rows)
    def r2(cols):
        Xr = np.column_stack([M[:, cols], np.ones(len(M))]); b, *_ = np.linalg.lstsq(Xr, M[:, 2], rcond=None)
        return float(1 - ((M[:, 2] - Xr @ b) ** 2).sum() / ((M[:, 2] - M[:, 2].mean()) ** 2).sum())
    out = dict(graph=name, vertices=n, disc=disc, fit_aR_bRlogR_c=coef.tolist(),
               mi_pairs=len(M), r2_logI_graph=r2([0]), r2_logI_euclid=r2([1]), r2_logI_both=r2([0, 1]))
    print(json.dumps({k: v for k, v in out.items() if k != "disc"}), flush=True)
    return out


def main():
    outp, K = sys.argv[1], int(sys.argv[2]); rng = np.random.default_rng(0)
    z, e, _ = spectre_graph(K); side = int(np.sqrt(len(z)))
    res = dict(K=K, runs=[analyse(z, e, f"spectre_K{K}", rng), analyse(*square(side), f"square_{side}", rng),
                          analyse(*honeycomb(int(side / np.sqrt(2))), "honeycomb", rng)])
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
