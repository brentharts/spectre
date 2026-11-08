"""Plan step 4: a gauge model on the Spectre graph that is not exactly solvable.

2+1D Z2 lattice gauge theory on (Spectre graph) x (time) is dual (Wegner) to the 3D Ising model on (dual graph) x
(time): one spin per plaquette (piece) per time slice, spatial coupling w_pq = number of links shared by pieces p
and q, temporal coupling 1.  The gauge theory deconfines exactly where the dual Ising model orders, so the
critical coupling beta_c of the dual model is a graph-dependent number with no closed form.

Method: Swendsen-Wang clusters (bond probability 1 - exp(-2 beta w)), periodic in time with L_t = round(N^(1/2)),
open in space.  beta_c is estimated from the crossing of the Binder cumulant U = 1 - <m^4>/(3<m^2>^2) between the
two system sizes, for the Spectre (K levels) and for honeycomb and square controls (duals: triangular and square).
The square control checks the code against the 3D Ising value beta_c = 0.22165 (open spatial boundaries shift the
crossing a little).  The crossing must lie where max(U) > 0.15, to reject noise at U ~ 0.  A crossing is reported only if U(small) - U(large) changes sign on the beta grid.

    python research/z2_deconfinement.py OUT.json
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from su2_graph import spectre_faces, square, honeycomb


def dual_edges(faces):
    owner = {}
    w = Counter()
    for p, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            e = (min(a, b), max(a, b))
            if e in owner and owner[e] != p:
                q = owner[e]; w[(min(p, q), max(p, q))] += 1
            else:
                owner[e] = p
    E = np.array(list(w.keys()), np.int64); W = np.array(list(w.values()), float)
    return len(faces), E, W


def lattice(N, E, W, Lt):
    """space-time bonds of the dual Ising model."""
    I, J, Wt = [], [], []
    for t in range(Lt):
        I.append(E[:, 0] + t * N); J.append(E[:, 1] + t * N); Wt.append(W)
        tn = (t + 1) % Lt
        I.append(np.arange(N) + t * N); J.append(np.arange(N) + tn * N); Wt.append(np.ones(N))
    return np.concatenate(I), np.concatenate(J), np.concatenate(Wt)


def simulate(N, E, W, betas, sweeps, therm, rng):
    Lt = max(4, int(round(np.sqrt(N))))
    I, J, Wt = lattice(N, E, W, Lt); n = N * Lt
    out = []
    for beta in betas:
        s = rng.choice([-1, 1], n); m2 = m4 = 0.0; cnt = 0
        p = 1 - np.exp(-2 * beta * Wt)
        for it in range(therm + sweeps):
            act = (s[I] == s[J]) & (rng.random(len(I)) < p)
            g = coo_matrix((np.ones(act.sum()), (I[act], J[act])), shape=(n, n))
            nc, lab = connected_components(g, directed=False)
            s = rng.choice([-1, 1], nc)[lab]  # new random sign per cluster
            if it >= therm:
                m = s.mean(); m2 += m * m; m4 += m ** 4; cnt += 1
        m2 /= cnt; m4 /= cnt
        out.append(dict(beta=float(beta), m2=m2, U=1 - m4 / (3 * m2 * m2)))
    return Lt, out


def crossing(a, b):
    d = np.array([x["U"] - y["U"] for x, y in zip(a, b)]); be = np.array([x["beta"] for x in a])
    # deep in the disordered phase U ~ 0 with large noise; only accept crossings where the curves have risen
    ok = np.array([max(x["U"], y["U"]) > 0.15 for x, y in zip(a, b)])
    k = np.nonzero((np.sign(d[:-1]) != np.sign(d[1:])) & ok[:-1] & ok[1:])[0]
    if len(k) == 0: return None
    k = k[0]; return float(be[k] - d[k] * (be[k + 1] - be[k]) / (d[k + 1] - d[k]))


def main():
    out = sys.argv[1]; rng = np.random.default_rng(1)
    systems = {"spectre": [lambda: spectre_faces(3), lambda: spectre_faces(4)],
               "honeycomb": [lambda: honeycomb(16), lambda: honeycomb(40)],
               "square": [lambda: square(12), lambda: square(24)]}
    # fine grids around the transitions seen in the coarse scan (job of 2026-11-08: spectre 0.08-0.09,
    # honeycomb 0.16-0.17, square 0.22-0.23)
    grids = {"spectre": np.linspace(0.080, 0.095, 16), "honeycomb": np.linspace(0.158, 0.172, 15),
             "square": np.linspace(0.214, 0.230, 17)}
    res = {}
    for name, makers in systems.items():
        runs = []
        for mk in makers:
            nv, faces = mk(); N, E, W = dual_edges(faces)
            Lt, data = simulate(N, E, W, grids[name], 2000, 300, rng)
            zc = float(2 * W.sum() / N)
            runs.append(dict(N=N, Lt=Lt, mean_spatial_coupling=zc, weights=dict(Counter(W.astype(int).tolist())), data=data))
            print(name, N, Lt, "mean weighted coordination", round(zc, 3), flush=True)
        bc = crossing(runs[0]["data"], runs[1]["data"])
        res[name] = dict(runs=runs, beta_c=bc,
                         beta_c_times_coordination_plus2=None if bc is None else bc * (runs[1]["mean_spatial_coupling"] + 2))
        print(name, "beta_c", bc, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
