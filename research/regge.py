"""Γ6 Regge test: does Ollivier vertex curvature track the holonomy winding, not just its parity?

Every interior vertex of the Spectre tiling is geometrically flat (wedge angles sum to 2π), so the only
candidate "deficit" is the holonomy: the frame rotation accumulated around the vertex, tot = 360 * w degrees.
The vortex flag used so far is flux = (-1)^w.  Here we keep the integer w and the count of π-steps n_pi,
and ask:
  1. Within each degree class, is mean vertex kappa a function of w (Regge: kappa ∝ deficit), or only of parity?
  2. Over disjoint discs, after regressing out each degree-class count, does sum kappa track sum w
     better than it tracks the vortex count?

Usage: python research/regge.py OUT.json K [K ...]
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "research")
from collections import defaultdict
import numpy as np
from graph_dilation import spectre_graph
from ollivier_ricci import adjacency, kappa
import vortex_density as vd


def winding(K, n):
    """per graph vertex: interior flag, winding w (tot/360), n_pi.  Same vertex numbering as spectre_graph."""
    from gap_collared import build
    _, _, V, PT = build(K)
    npc = len(PT)
    _, vid = np.unique(V.reshape(-1, 4), axis=0, return_inverse=True); vid = vid.reshape(npc, 14)
    Z = vd.to_c(V.reshape(-1, 4)).reshape(npc, 14)
    R = vd.to_c(np.array(vd.ref_points(), np.int64))
    area = lambda z: 0.5 * np.imag(np.sum(np.conj(z) * np.roll(z, -1, axis=-1), axis=-1))
    det = np.sign(area(Z) / area(R)).astype(int)
    theta = np.rad2deg(np.angle(Z[:, 1] - Z[:, 0])) - det * np.rad2deg(np.angle(R[1] - R[0]))
    ccw = area(Z) > 0
    wedges = defaultdict(list)
    for p in range(npc):
        z = Z[p]
        for i in range(14):
            an, ap = np.angle(z[(i + 1) % 14] - z[i]), np.angle(z[i - 1] - z[i])
            start, width = (an, (ap - an) % (2 * np.pi)) if ccw[p] else (ap, (an - ap) % (2 * np.pi))
            wedges[int(vid[p, i])].append((p, start % (2 * np.pi), width))
    inter = np.zeros(n, bool); w = np.zeros(n, int); npi = np.zeros(n, int)
    for v, ws in wedges.items():
        if abs(sum(x[2] for x in ws) - 2 * np.pi) > 1e-6 or v >= n:
            continue
        ws = sorted(ws, key=lambda x: x[1]); k = len(ws); tot = 0
        for i in range(k):
            a, b = ws[i][0], ws[(i + 1) % k][0]
            d = ((theta[b] - theta[a]) * det[a] + 180) % 360 - 180
            d = int(round(d)); d = 180 if d == -180 else d
            tot += d; npi[v] += abs(d) == 180
        inter[v] = True; w[v] = tot // 360
    return inter, w, npi


def resid(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ b


def pc(y, x, X):
    ry, rx = resid(y, X), resid(x, X)
    r = float(np.corrcoef(ry, rx)[0, 1]); dof = len(y) - X.shape[1] - 3
    if dof < 1 or not np.isfinite(r): return r, None
    zz = np.arctanh(np.clip(r, -0.999999, 0.999999)); s = 1 / np.sqrt(dof)
    return r, [float(np.tanh(zz - 1.96 * s)), float(np.tanh(zz + 1.96 * s))]


def analyse(K):
    z, e, _ = spectre_graph(K); n = len(z)
    inter, w, npi = winding(K, n)
    nb = adjacency(n, e); deg = np.array([len(x) for x in nb])
    c = z.mean(); R = np.abs(z - c); Rb = R.max()
    act = inter & (R < Rb - 3) & (deg > 0)
    need = set(np.nonzero(act)[0].tolist())
    eb = [(x, y) for x, y in e if x in need or y in need]
    kv = np.zeros(n); cnt = np.zeros(n)
    for x, y in eb:
        k = kappa(nb, int(x), int(y), 0.0)
        kv[x] += k; kv[y] += k; cnt[x] += 1; cnt[y] += 1
    ok = act & (cnt == deg); kv = kv / np.maximum(cnt, 1)
    res = dict(K=K, n_vertices=int(ok.sum()))
    # 1. table by (degree, w, n_pi)
    tab = defaultdict(list)
    for v in np.nonzero(ok)[0]:
        tab[(int(deg[v]), int(w[v]), int(npi[v]))].append(kv[v])
    res["table"] = [dict(deg=d, w=ww, npi=p, count=len(x), kappa_mean=float(np.mean(x)), kappa_sd=float(np.std(x)))
                    for (d, ww, p), x in sorted(tab.items())]
    # 2. disjoint discs
    degs = sorted(set(deg[ok].tolist())); rng = np.random.default_rng(0)
    for rad in (4, 8, 16):
        idx = np.nonzero(ok & (R < Rb - 3 - rad))[0]; rng.shuffle(idx); cent = []
        for i in idx:
            if all(abs(z[i] - z[j]) >= 2 * rad for j in cent): cent.append(i)
            if len(cent) >= 4000: break
        rows = []
        for i in cent:
            m = ok & (np.abs(z - z[i]) < rad)
            rows.append([kv[m].sum(), (w[m] % 2).sum(), w[m].sum(), np.abs(w[m]).sum(), npi[m].sum()]
                        + [(m & (deg == d)).sum() for d in degs])
        A = np.array(rows, float)
        if len(A) < 10: continue
        X = np.column_stack([np.ones(len(A)), A[:, 5:]])
        out = {}
        for j, name in ((1, "vortex_count"), (2, "sum_w"), (3, "sum_abs_w"), (4, "sum_npi")):
            r, ci = pc(A[:, 0], A[:, j], X); out[name] = dict(r=r, ci95=ci)
        # does w add anything once the vortex count is controlled?
        r, ci = pc(A[:, 0], A[:, 3], np.column_stack([X, A[:, 1]])); out["sum_abs_w_given_vortex"] = dict(r=r, ci95=ci)
        res[f"r{rad}"] = dict(n_discs=len(A), **out)
    return res


def main():
    out = sys.argv[1]; Ks = [int(k) for k in sys.argv[2:]] or [3]; res = []
    for K in Ks:
        t = time.time(); res.append(analyse(K)); res[-1]["seconds"] = round(time.time() - t, 1)
        print(json.dumps(res[-1])[:2000], flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
