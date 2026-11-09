"""Γ6 foothold: Ollivier–Ricci curvature of the Spectre vertex graph, against the holonomy vortices.

For each edge (x, y), mu_x puts mass alpha on x and (1 - alpha)/deg(x) on each neighbour.  The curvature is
kappa(x, y) = 1 - W1(mu_x, mu_y), with W1 the earth mover distance under graph distance (a small LP per edge).
alpha = 0 and alpha = 1/2 are both reported.  Vertex curvature is the mean over incident edges.

Questions (NEXT_STEPS Γ6): what is the distribution; is kappa different at vortices (flux -1 vertices from
vortex_density.star_data) once vertex degree is held fixed; do disc sums of kappa track the vortex count?
Controls: square lattice and honeycomb (constant curvature, no vortices).
Usage: python research/ollivier_ricci.py OUT.json K [K ...]
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from scipy.optimize import linprog
from graph_dilation import spectre_graph, square, honeycomb

ALPHAS = (0.0, 0.5)


def adjacency(n, e):
    nb = [[] for _ in range(n)]
    for a, b in e:
        nb[a].append(b); nb[b].append(a)
    return nb


def dist_small(nb, s, targets, cap=3):
    """BFS from s up to depth cap; distances to targets."""
    seen = {s: 0}; frontier = [s]
    for d in range(1, cap + 1):
        nxt = []
        for u in frontier:
            for w in nb[u]:
                if w not in seen:
                    seen[w] = d; nxt.append(w)
        frontier = nxt
    return [seen.get(t, cap + 1) for t in targets]


def measure(nb, x, alpha):
    pts = [x] + nb[x]; m = [alpha] + [(1 - alpha) / len(nb[x])] * len(nb[x])
    return pts, np.array(m)


def kappa(nb, x, y, alpha):
    S, a = measure(nb, x, alpha); T, b = measure(nb, y, alpha)
    C = np.array([dist_small(nb, s, T) for s in S], float)
    ns, nt = len(S), len(T)
    Aeq = np.zeros((ns + nt, ns * nt))
    for i in range(ns): Aeq[i, i * nt:(i + 1) * nt] = 1
    for j in range(nt): Aeq[ns + j, j::nt] = 1
    r = linprog(C.ravel(), A_eq=Aeq, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    return 1 - r.fun


def analyse(z, e, name, vort=None, rng=None):
    n = len(z); nb = adjacency(n, e); deg = np.array([len(x) for x in nb])
    c = z.mean(); R = np.abs(z - c); Rb = 0.8 * R.max()
    keep = (R[e[:, 0]] < Rb) & (R[e[:, 1]] < Rb)           # bulk edges
    eb = e[keep]; out = dict(graph=name, vertices=n, bulk_edges=int(len(eb)))
    for al in ALPHAS:
        k = np.array([kappa(nb, int(x), int(y), al) for x, y in eb])
        kv = np.zeros(n); cnt = np.zeros(n)
        np.add.at(kv, eb[:, 0], k); np.add.at(kv, eb[:, 1], k); np.add.at(cnt, eb[:, 0], 1); np.add.at(cnt, eb[:, 1], 1)
        inner = (R < Rb - 3) & (cnt == deg) & (deg > 0); kv = kv / np.maximum(cnt, 1)
        vals, counts = np.unique(np.round(k, 6), return_counts=True)
        r = dict(edge_mean=float(k.mean()), edge_std=float(k.std()), edge_min=float(k.min()), edge_max=float(k.max()),
                 edge_values=[[float(v), int(c)] for v, c in zip(vals, counts)][:40], n_distinct=int(len(vals)))
        if vort is not None:
            bydeg = {}
            for d in np.unique(deg[inner]):
                s = inner & (deg == d); sv = s & vort; sn = s & ~vort
                bydeg[int(d)] = dict(n_vortex=int(sv.sum()), n_other=int(sn.sum()),
                                     kappa_vortex=float(kv[sv].mean()) if sv.any() else None,
                                     kappa_other=float(kv[sn].mean()) if sn.any() else None,
                                     hist_vortex=[[float(v), int(c)] for v, c in zip(*np.unique(np.round(kv[sv], 4), return_counts=True))],
                                     hist_other=[[float(v), int(c)] for v, c in zip(*np.unique(np.round(kv[sn], 4), return_counts=True))])
            r["vertex_kappa_by_degree"] = bydeg
            # disc sums: sum of vertex kappa vs vortex count.  Controls: (a) degree sum + count (weak; v1),
            # (b) separate counts of each degree class (strong).  Overlapping random discs and a
            # non-overlapping greedy packing (independent samples; Fisher-z 95% CI).
            idx = np.nonzero(inner)[0]; degs = [int(d) for d in np.unique(deg[inner])]; corr = {}
            for rad in (4, 8, 16):
                pool = idx[R[idx] < Rb - 3 - rad]
                if len(pool) == 0: continue
                packed = []
                for ci in rng.permutation(pool):
                    if all(abs(z[ci] - z[q]) >= 2 * rad for q in packed): packed.append(ci)
                for tag, cs in (("overlap", rng.choice(pool, size=min(400, len(pool)), replace=True)), ("disjoint", np.array(packed))):
                    rows = []
                    for ci in cs:
                        m = inner & (np.abs(z - z[ci]) < rad)
                        rows.append([kv[m].sum(), vort[m].sum(), deg[m].sum(), m.sum()] + [(m & (deg == d)).sum() for d in degs])
                    X = np.array(rows, float)
                    if len(X) < 10: continue
                    res = lambda y, D: y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
                    one = np.ones(len(X)); c = dict(discs=int(len(X)), raw=float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]))
                    for ctl, D, k in (("partial_degsum", np.c_[one, X[:, 2], X[:, 3]], 2), ("partial_bydegree", np.c_[one, X[:, 4:]], len(degs))):
                        a, b = res(X[:, 0], D), res(X[:, 1], D)
                        pr = float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-300)); c[ctl] = pr
                        dof = len(X) - k - 3
                        if dof > 0 and abs(pr) < 1:
                            zf = np.arctanh(pr); h = 1.96 / np.sqrt(dof); c[ctl + "_ci95"] = [float(np.tanh(zf - h)), float(np.tanh(zf + h))]
                    corr["r%d_%s" % (rad, tag)] = c
            r["disc_sum_vs_vortices"] = corr
        out["alpha_%g" % al] = r
    return out


def vortices(K, n):
    from gap_collared import build
    from vortex_density import star_data
    _, _, V, PT = build(K)
    _, interior, flux, *_ = star_data(V, PT, len(PT))
    v = np.zeros(n, bool); v[:len(interior)] = interior & (flux == -1)
    return v


def main():
    out = sys.argv[1]; Ks = [int(k) for k in sys.argv[2:]] or [3]
    rng = np.random.default_rng(0); res = []
    for K in Ks:
        t = time.time(); z, e, _ = spectre_graph(K)
        res.append(analyse(z, e, "spectre_K%d" % K, vortices(K, len(z)), rng)); res[-1]["seconds"] = round(time.time() - t, 1)
        side = int(np.sqrt(len(z)))
        for name, (zz, ee) in (("square_%d" % side, square(side)), ("honeycomb", honeycomb(int(np.sqrt(len(z) / 2))))):
            res.append(analyse(zz, ee, name))
        for r in res[-3:]:
            print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("edge_mean", "n_distinct")}) for k, v in r.items()}), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
