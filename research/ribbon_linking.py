r"""ribbon_linking.py -- R6: twist, writhe and linking of the lifted ribbons of the Spectre tree.

For every parent -> child ribbon of spectrespace.edge_ribbon, sampled at m points, compute
  Tw  = (1/2pi) sum_i  (T_i x u_i) . (u_{i+1} - u_i)      (u = unit width direction made normal to T)
  Wr  = Gauss double sum of the centre line with itself  (open curve, segment midpoints)
  Lk  = Gauss double sum between the centre line and the edge curve P + eps u
and the residual Lk - (Tw + Wr).  The curves are open, so Calugareanu holds only up to end terms; the residual is
reported, not assumed zero.  Totals and per-ribbon densities are given per depth and per twist sign (+1, -1, alt),
and the slope of log |sum Lk| against depth is fitted (compare with 2 log lambda of the substitution).

    python research/ribbon_linking.py OUT.json [depths ...]
"""
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import spectrespace as S  # noqa: E402


def gauss(P, Q, same):
    """Discrete Gauss linking integral of two polylines (midpoint rule); `same` skips adjacent segments."""
    dP, dQ = np.diff(P, axis=0), np.diff(Q, axis=0)
    mP, mQ = 0.5 * (P[1:] + P[:-1]), 0.5 * (Q[1:] + Q[:-1])
    r = mP[:, None] - mQ[None]
    d = np.linalg.norm(r, axis=2)
    cr = np.cross(dP[:, None], dQ[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.einsum("ijk,ijk->ij", cr, r) / d ** 3
    if same:
        n = len(dP)
        idx = np.arange(n)
        mask = np.abs(idx[:, None] - idx[None]) <= 1
        g[mask] = 0.0
    g[~np.isfinite(g)] = 0.0
    return float(g.sum() / (4 * np.pi))


def ribbon_numbers(space, nd, zscale, sign_mode, m=64, eps=0.05):
    P, off, _, _ = S.edge_ribbon(space, nd, m, zscale, 20.0, 0.09, sign_mode)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    u = off - np.einsum("ij,ij->i", off, T)[:, None] * T
    u /= np.linalg.norm(u, axis=1)[:, None]
    tw = float(np.einsum("ij,ij->i", np.cross(T[:-1], u[:-1]), u[1:] - u[:-1]).sum() / (2 * np.pi))
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    wr = gauss(P, P, True)
    lk = gauss(P, P + eps * L * u, False)
    return tw, wr, lk


def run(depth, sign_mode):
    space = S.SpectreSpace(depth)
    allv = np.vstack([nd.verts for nd in space.leaves()])
    ext = float(np.ptp(allv, axis=0).max())
    zscale = 0.45 * ext / (space.n + 0.5)
    rows = [ribbon_numbers(space, nd, zscale, sign_mode) for nd in space.nodes[1:]]
    a = np.array(rows)
    tw, wr, lk = a.T
    res = lk - (tw + wr)
    return dict(depth=depth, sign=sign_mode, ribbons=len(rows),
                Tw=float(tw.sum()), Wr=float(wr.sum()), Lk=float(lk.sum()),
                abs_Tw=float(np.abs(tw).sum()), max_abs_residual=float(np.abs(res).max()),
                mean_residual=float(res.mean()), Tw_per_ribbon=float(tw.mean()), Lk_per_ribbon=float(lk.mean()))


def main():
    out = sys.argv[1]
    depths = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    res = {"rows": []}
    for sgn in ("+1", "-1", "alt"):
        for d in depths:
            r = run(d, sgn)
            print(json.dumps(r), flush=True)
            res["rows"].append(r)
    for sgn in ("+1", "-1", "alt"):
        rr = [r for r in res["rows"] if r["sign"] == sgn and abs(r["Lk"]) > 0]
        if len(rr) >= 2:
            x = np.array([r["depth"] for r in rr], float)
            y = np.log(np.abs([r["Lk"] for r in rr]))
            res["log_abs_Lk_slope_" + sgn] = float(np.polyfit(x, y, 1)[0])
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
