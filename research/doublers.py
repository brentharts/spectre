"""Item 8 (NEXT_STEPS.md): fermion doubling for a naive Dirac operator on the Spectre tiling.

Sites are tile centroids; links join tiles sharing an edge.  The naive operator is
    D_ij = (1/2) sigma . e_ij   (e_ij unit vector i->j),  anti-Hermitian; H = i D.
Doublers show up as low modes whose Fourier weight sits away from k = 0.  For every
lattice we report
  * frac0: Fourier weight of the |E| < eps modes within |k| < k0 (square lattice: ~1/4,
    i.e. 4 species; a doubler-free lattice: ~1);
  * the low-mode count per unit area relative to the same lattice's Dirac-cone fit.
References: square lattice (4 species) and a Poisson-Delaunay random lattice
(Christ-Friedberg-Lee: random lattices already lose the doublers), so the Spectre result
is read against the right baseline.  Usage: python3 doublers.py LEVELS... (e.g. 3 4 5).
"""
import sys, time, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import Delaunay

sys.path.insert(0, ".")
import spectre_patch as spm

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)


def spectre_graph(levels):
    p = spm.SpectrePatch(levels)
    cen = np.array([p.straight_polygon(i).mean(axis=0) for i in range(p.n_tiles)])
    pr = p.edge_pairs()
    edges = {(min(a, b), max(a, b)) for a, _, b, _ in pr}
    return cen, np.array(sorted(edges))


def square_graph(n):
    xy = np.array([(i, j) for i in range(n) for j in range(n)], float)
    idx = lambda i, j: i * n + j
    e = [(idx(i, j), idx(i + 1, j)) for i in range(n - 1) for j in range(n)]
    e += [(idx(i, j), idx(i, j + 1)) for i in range(n) for j in range(n - 1)]
    return xy, np.array(e)


def random_graph(npts, seed=0):
    rng = np.random.default_rng(seed)
    L = np.sqrt(npts)
    xy = rng.uniform(0, L, (npts, 2))
    tri = Delaunay(xy).simplices
    e = set()
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
            e.add((min(a, b), max(a, b)))
    return xy, np.array(sorted(e))


def dirac(xy, edges):
    n = len(xy)
    rows, cols, vals = [], [], []
    for i, j in edges:
        d = xy[j] - xy[i]
        d = d / np.linalg.norm(d)
        blk = 0.5 * (d[0] * SX + d[1] * SY)          # D_ij;  D_ji = -D_ij
        for s in range(2):
            for t in range(2):
                if blk[s, t] != 0:
                    rows += [2 * i + s, 2 * j + s]; cols += [2 * j + t, 2 * i + t]
                    vals += [1j * blk[s, t], -1j * blk[s, t]]
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))   # H = i D, Hermitian


def analyse(name, xy, edges, nlow=160):
    t0 = time.time()
    xy = xy - xy.mean(axis=0)
    nn = np.median(np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1))
    xy = xy / nn                                       # unit link length
    H = dirac(xy, edges)
    assert abs(H - H.getH()).max() < 1e-12
    n2 = H.shape[0]
    if n2 <= 12000:
        E, V = np.linalg.eigh(H.toarray())
    else:
        E, V = sla.eigsh(H, k=nlow, sigma=1e-6, which="LM")
    order = np.argsort(np.abs(E)); E, V = E[order], V[:, order]
    # interior window to suppress boundary modes
    R = np.sqrt(np.max(np.sum(xy ** 2, axis=1)))
    w = np.exp(-np.sum(xy ** 2, axis=1) / (2 * (0.35 * R) ** 2))
    kmax = np.pi * 1.5
    ks = np.linspace(-kmax, kmax, 97)
    KX, KY = np.meshgrid(ks, ks)
    K = np.column_stack([KX.ravel(), KY.ravel()])
    phase = np.exp(-1j * xy @ K.T) * w[:, None]       # (n, nk)
    m = min(nlow, len(E))
    keep = [i for i in range(m) if abs(E[i]) > 1e-9][: nlow // 2]
    S = np.zeros(len(K))
    for i in keep:
        psi = V[:, i].reshape(-1, 2)
        S += np.sum(np.abs(phase.T @ psi) ** 2, axis=1)
    S /= S.sum()
    kn = np.linalg.norm(K, axis=1)
    k0 = 0.6
    frac0 = float(S[kn < k0].sum())
    # peaks of S: report the top 6 local maxima
    S2 = S.reshape(KX.shape)
    peaks = []
    for a in range(1, 96):
        for b in range(1, 96):
            if S2[a, b] == S2[a - 1:a + 2, b - 1:b + 2].max() and S2[a, b] > 0.2 * S2.max():
                peaks.append((float(S2[a, b]), round(float(KX[a, b]), 2), round(float(KY[a, b]), 2)))
    peaks.sort(reverse=True)
    res = dict(lattice=name, sites=len(xy), links=len(edges), mean_deg=2 * len(edges) / len(xy),
               zero_modes=int(np.sum(np.abs(E[:m]) < 1e-9)), lowest=[round(float(abs(e)), 5) for e in E[:12:2]],
               frac0=round(frac0, 4), species_est=round(1 / max(frac0, 1e-9), 2), peaks=peaks[:8],
               seconds=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    levels = [int(a) for a in sys.argv[1:]] or [3, 4]
    out = []
    for L in levels:
        xy, e = spectre_graph(L)
        out.append(analyse(f"spectre-L{L}", xy, e))
        n = len(xy)
        s = int(round(np.sqrt(n)))
        out.append(analyse(f"square-{s}", *square_graph(s)))
        out.append(analyse(f"random-{n}", *random_graph(n)))
    json.dump(out, open("doublers.json", "w"), indent=1)
