"""R8 follow-up: tight-binding on the Spectre piece graph with complex flux phi at each vortex.

R8 (flux_hamiltonian.py) used real pi-flux, so H is time-reversal symmetric and every Bott index is forced to 0.
Here each shared edge, hopped a -> b, gets a U(1) phase exp(i theta_ab).  The flux through the dual loop around an
interior vertex v is the oriented sum of theta over the edges incident to v, with sign +1 when the hop a -> b runs
counter-clockwise around v.  We solve  B theta = phi * [v is a vortex]  (least squares; a disc is always solvable)
and check the residual.  phi = pi/2 breaks time reversal, so a nonzero Bott index is allowed; phi = pi is run as
a control and must reproduce R8 (all Bott 0).
Usage: python research/flux_complex.py OUT.json K [K ...]
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from gap_collared import build
from vortex_density import star_data
from gap_ids import distance_from, best_label, R_BULK, GMIN, WMIN, ALPHA

Z12 = np.exp(2j * np.pi * np.arange(4) / 12)


def setup(K):
    names, levels, V, PT = build(K)
    npc = len(PT)
    vid, interior, flux, npi, val, members, _ = star_data(V, PT, npc)
    vz = np.zeros(int(vid.max()) + 1, complex)
    vz[vid.ravel()] = V.reshape(-1, 4).astype(float) @ Z12
    e = np.sort(np.stack([vid, np.roll(vid, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(npc), 14)
    ue, einv, ecnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bd = np.zeros(npc, bool); bd[owner[ecnt[einv] == 1]] = True
    order = np.argsort(einv, kind="stable"); se, so = einv[order], owner[order]
    same = se[1:] == se[:-1]
    a, b, eid = so[:-1][same], so[1:][same], se[1:][same]
    keep = a != b
    a, b, eid = a[keep], b[keep], eid[keep]
    pos = (V.reshape(npc, 14, 4).astype(float) @ Z12).mean(1)
    vort = interior & (flux == -1)
    # oriented incidence: row v, col j, sign of the a->b hop circulating ccw around v
    rows, cols, vals = [], [], []
    for j, k in enumerate(eid):
        mid = vz[ue[k]].mean()
        for v in ue[k]:
            if not interior[v]:
                continue
            t = 1j * (mid - vz[v])                       # ccw tangent at the crossing
            sgn = np.sign(np.real((pos[b[j]] - pos[a[j]]) * np.conj(t)))
            rows.append(v); cols.append(j); vals.append(sgn)
    B = coo_matrix((vals, (rows, cols)), shape=(len(vz), len(eid))).tocsr()
    return dict(n=npc, a=a, b=b, B=B, vort=vort, interior=interior, bd=bd, pos=pos,
                nvort=int(vort.sum()), pairs=np.unique(np.sort(np.stack([a, b], 1), 1), axis=0))


def phases(g, phi):
    rhs = phi * g["vort"].astype(float)
    th = lsqr(g["B"], rhs, atol=1e-14, btol=1e-14, iter_lim=200000)[0]
    res = g["B"] @ th - rhs
    d = (res[g["interior"]] + np.pi) % (2 * np.pi) - np.pi
    return th, float(np.abs(d).max())


def bott(Q, pos, sel):
    z = pos[sel]; Q = Q[:, sel]
    x, y = z.real, z.imag
    U = np.exp(2j * np.pi * (x - x.min()) / (np.ptp(x) * 1.05))
    W = np.exp(2j * np.pi * (y - y.min()) / (np.ptp(y) * 1.05))
    A = (Q * U) @ Q.conj().T; Bm = (Q * W) @ Q.conj().T
    ev = np.linalg.eigvals(A @ Bm @ A.conj().T @ Bm.conj().T)
    return float(np.angle(ev[np.abs(ev) > 1e-8]).sum() / (2 * np.pi))


def analyse(g, th):
    n = g["n"]; H = np.zeros((n, n), complex)
    w = np.exp(1j * th)
    np.add.at(H, (g["a"], g["b"]), w); np.add.at(H, (g["b"], g["a"]), w.conj())
    E, U = np.linalg.eigh(H); del H
    bulk = distance_from(n, g["pairs"], g["bd"]) >= R_BULK; nb = int(bulk.sum())
    wt = (np.abs(U[bulk]) ** 2).sum(0); ids = np.cumsum(wt) / nb; band = E[-1] - E[0]
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
        t = time.time(); g = setup(K); r = dict(sites=g["n"], vortices=g["nvort"])
        for name, phi in (("pi", np.pi), ("pi_half", np.pi / 2)):
            th, err = phases(g, phi)
            r[name] = analyse(g, th); r[name]["flux_residual"] = err
            print(K, name, "residual %.1e gaps %d" % (err, len(r[name]["gaps"])), flush=True)
            for gp in r[name]["gaps"]:
                print("  [%.4f,%.4f] ids=%.6f label=%s bott=%s" % (gp["lo"], gp["hi"], gp["ids"], gp["label"], gp.get("bott")), flush=True)
        r["seconds"] = round(time.time() - t, 1); res["levels"][K] = r
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
