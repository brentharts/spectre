"""R8: tight-binding on the Spectre tile graph with Brent's Z2 flux inserted as pi-flux at each vortex.

Sites are spectre pieces (as in gap_ids.py); hopping between pieces sharing a tile edge.  Each shared edge e gets a
Peierls sign s(e) in {+1,-1}.  Around an interior vertex v the dual loop crosses exactly the edges incident to v, so
we solve over GF(2):  sum_{e incident v, e shared} c(e) = [flux(v) = -1]  for every interior vertex (a disc, always
solvable), and set s = (-1)^c.  Then compare bulk IDS gaps with and without flux; every gap's IDS is matched to the
module Z + Z*alpha (gauge-covariant gap labelling must still hold), and for each gap the Bott index of the projector
below it is computed on bulk-centred positions.
Usage: python research/flux_hamiltonian.py OUT.json K [K ...]
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build
from vortex_density import star_data
from gap_ids import distance_from, best_label, R_BULK, GMIN, WMIN, ALPHA

Z12 = np.exp(2j * np.pi * np.arange(4) / 12)


def setup(K):
    names, levels, V, PT = build(K)
    npc = len(PT)
    vid, interior, flux, npi, val, members, _ = star_data(V, PT, npc)
    e = np.sort(np.stack([vid, np.roll(vid, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(npc), 14)
    ue, einv, ecnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bd = np.zeros(npc, bool); bd[owner[ecnt[einv] == 1]] = True
    order = np.argsort(einv, kind="stable"); se, so = einv[order], owner[order]
    same = se[1:] == se[:-1]
    a, b, eid = so[:-1][same], so[1:][same], se[1:][same]
    keep = a != b
    a, b, eid = a[keep], b[keep], eid[keep]          # one entry per shared edge (pair may repeat)
    # GF(2) system: rows = interior vertices, cols = shared edges
    nE = len(eid); nv = int(vid.max()) + 1
    rows = [0] * nv
    for j, k in enumerate(eid):
        for v in ue[k]:
            rows[v] |= 1 << j
    vort = interior & (flux == -1)
    eqs = [(rows[v], int(vort[v])) for v in np.nonzero(interior)[0]]
    piv = {}
    for r, rhs in eqs:
        while r:
            p = r.bit_length() - 1
            if p in piv:
                pr, prhs = piv[p]; r ^= pr; rhs ^= prhs
            else:
                piv[p] = (r, rhs); break
        else:
            assert rhs == 0, "flux system inconsistent"
    c = 0
    for p in sorted(piv):                       # back-substitute, free vars = 0
        r, rhs = piv[p]
        par = bin((r & ~(1 << p)) & c).count("1") & 1
        if rhs ^ par: c |= 1 << p
    s = np.array([-1.0 if (c >> j) & 1 else 1.0 for j in range(nE)])
    pos = (V.reshape(npc, 14, 4).astype(float) @ Z12).mean(1)
    return dict(n=npc, a=a, b=b, s=s, bd=bd, pos=pos, nvort=int(vort.sum()),
                pairs=np.unique(np.sort(np.stack([a, b], 1), 1), axis=0))


def bott(Q, pos, sel):
    """Q: occupied eigenvectors as rows (k x n); positions restricted to bulk sites."""
    z = pos[sel]; Q = Q[:, sel]
    x, y = z.real, z.imag
    U = np.exp(2j * np.pi * (x - x.min()) / (np.ptp(x) * 1.05))
    W = np.exp(2j * np.pi * (y - y.min()) / (np.ptp(y) * 1.05))
    A = (Q * U) @ Q.conj().T; B = (Q * W) @ Q.conj().T
    M = A @ B @ A.conj().T @ B.conj().T
    ev = np.linalg.eigvals(M)
    return float(np.angle(ev[np.abs(ev) > 1e-8]).sum() / (2 * np.pi))


def analyse(g, flux):
    n = g["n"]; H = np.zeros((n, n))
    w = g["s"] if flux else np.ones(len(g["s"]))
    np.add.at(H, (g["a"], g["b"]), w); np.add.at(H, (g["b"], g["a"]), w)
    E, U = np.linalg.eigh(H); del H
    bulk = distance_from(n, g["pairs"], g["bd"]) >= R_BULK; nb = int(bulk.sum())
    wt = (U[bulk] ** 2).sum(0); ids = np.cumsum(wt) / nb; band = E[-1] - E[0]
    carry = np.nonzero(wt > WMIN * nb / n)[0]; gaps = []
    for i, j in zip(carry[:-1], carry[1:]):
        if E[j] - E[i] >= GMIN * band:
            x = float(ids[j - 1]); m, nn, r = best_label(x)
            gp = dict(lo=float(E[i]), hi=float(E[j]), ids=x, label=[m, nn], residual=float(r))
            if n <= 6000:
                gp["bott"] = bott(U[:, :i + 1].T, g["pos"], bulk)
            gaps.append(gp)
    return dict(bandwidth=float(band), gaps=gaps)


def main():
    out = sys.argv[1]; Ks = [int(k) for k in sys.argv[2:]] or [3, 4]
    res = dict(alpha=ALPHA, levels={})
    for K in Ks:
        t = time.time(); g = setup(K)
        r = dict(sites=g["n"], vortices=g["nvort"], plain=analyse(g, False), flux=analyse(g, True))
        r["seconds"] = round(time.time() - t, 1); res["levels"][K] = r
        for m in ("plain", "flux"):
            print(K, m, "gaps", len(r[m]["gaps"]), flush=True)
            for gp in r[m]["gaps"]:
                print("  [%.4f,%.4f] ids=%.6f label=%s res=%.1e bott=%s" % (gp["lo"], gp["hi"], gp["ids"], gp["label"], gp["residual"], gp.get("bott")), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
