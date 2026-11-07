"""R11 step 2: Kogut-Susskind SU(2) gauge theory on the Spectre vertex graph (links = tile edges, plaquettes = pieces),
replacing the theta graph of Part II.  Graph-dependent factors only; every number is also computed on the theta graph
with the same conventions (strong coupling normalised to Part II's 2/(3 g^8)).

Strong coupling (first order in 1/g^2): plaquette p excited to j=1/2 with amplitude ~ 1/L_p, so the ground-state
mean-square area of link l is  <A_l^2> ∝ S_l = sum_{p ∋ l} 1/L_p^2   (L_p = links on the boundary of p).
Weak coupling (Gaussian, 3 colours): <A_l^2> ∝ [sqrt(B^T B)]_ll, B = oriented link->plaquette incidence (curl).
Volumes: with spin-1/2 on every link, a valence-3 vertex has zero volume and a valence-4 vertex has spectrum ±sqrt3/4
(Part II), so the volume density is the density of valence-4 vertices; the full valence histogram is reported.
Controls: honeycomb and square discs of similar size, plus a matched-mean-valence periodic graph when available.
Usage: python research/su2_graph.py OUT.json K [K ...]
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from collections import Counter
from gap_ids import distance_from
R_BULK = 4


def analyse(nv, faces, name):
    """faces: list of vertex cycles (closed polygons)."""
    eidx = {}; rows = []
    for f in faces:
        row = {}
        for a, b in zip(f, f[1:] + f[:1]):
            k = (min(a, b), max(a, b))
            if k not in eidx: eidx[k] = len(eidx)
            row[eidx[k]] = row.get(eidx[k], 0) + (1 if a < b else -1)
        rows.append(row)
    nE = len(eidx); edges = np.array(list(eidx.keys()))
    B = np.zeros((len(faces), nE))
    for i, row in enumerate(rows):
        for j, s in row.items(): B[i, j] = s
    L = np.abs(B).sum(1)
    S = (np.abs(B) / L[:, None] ** 2).sum(0)
    lam, U = np.linalg.eigh(B.T @ B)
    W = (U ** 2) @ np.sqrt(np.clip(lam, 0, None))
    nfaces = np.abs(B).sum(0)
    bdv = np.zeros(nv, bool); bdv[edges[nfaces == 1].ravel()] = True
    dist = distance_from(nv, edges, bdv)
    bulk_e = (dist[edges[:, 0]] >= R_BULK) & (dist[edges[:, 1]] >= R_BULK)
    deg = np.bincount(edges.ravel(), minlength=nv)
    bulk_v = dist >= R_BULK
    def stats(x):
        x = x[bulk_e]
        return dict(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()),
                    distinct=int(len(np.unique(np.round(x, 6)))))
    vh = Counter(deg[bulk_v].tolist())
    r = dict(graph=name, vertices=nv, links=nE, plaquettes=len(faces), bulk_links=int(bulk_e.sum()),
             strong_S=stats(S), weak_sqrtK=stats(W),
             valence_hist={int(k): int(v) for k, v in sorted(vh.items())},
             mean_valence_bulk=float(deg[bulk_v].mean()),
             frac_valence4_bulk=float((deg[bulk_v] == 4).mean()),
             plaquette_size_hist={int(k): int(v) for k, v in Counter(L.astype(int).tolist()).items()})
    print(json.dumps(r), flush=True)
    return r


def spectre_faces(K):
    from gap_collared import build
    names, levels, V, PT = build(K)
    npc = len(PT)
    _, vid = np.unique(V.reshape(-1, 4), axis=0, return_inverse=True)
    vid = vid.reshape(npc, 14)
    return int(vid.max()) + 1, [list(map(int, r)) for r in vid]


def square(n):
    idx = lambda i, j: i * (n + 1) + j
    return (n + 1) ** 2, [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(n) for j in range(n)]


def honeycomb(n):
    # brick-wall realisation: hexagons in an n x n brick lattice
    w = 2 * n + 2; idx = lambda i, j: i * w + j
    faces = []
    for i in range(n):
        for j in range(n):
            c = 2 * j + (i % 2)
            faces.append([idx(i, c), idx(i, c + 1), idx(i, c + 2), idx(i + 1, c + 2), idx(i + 1, c + 1), idx(i + 1, c)])
    used = sorted({v for f in faces for v in f}); m = {v: k for k, v in enumerate(used)}
    return len(used), [[m[v] for v in f] for f in faces]


def main():
    out = sys.argv[1]; Ks = [int(k) for k in sys.argv[2:]] or [3, 4]
    res = dict(R_bulk=R_BULK, runs=[])
    # Part II theta graph, middle link: S = 2 * (1/2^2) = 1/2  <->  <A^2> = 2/(3 g^8), so <A^2>_strong = (4/(3 g^8)) * S.
    # Weak coupling: theta middle link [sqrt K] = 2/sqrt3 with this code; weak values are reported unnormalised.
    res["strong_area_sq_per_S"] = "4/(3 g^8)"; res["theta_weak_sqrtK_middle"] = 2 / np.sqrt(3)
    for K in Ks:
        nv, faces = spectre_faces(K)
        res["runs"].append(dict(K=K, **analyse(nv, faces, f"spectre_K{K}")))
        side = int(np.sqrt(len(faces)))
        res["runs"].append(analyse(*square(side), f"square_{side}"))
        res["runs"].append(analyse(*honeycomb(side), f"honeycomb_{side}"))
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
