"""Γ8: is the Spectre vertex set a Z[zeta12] model set at all?

module_fit v1 found no strong peak at any zeta^5 / zeta^7 module point.  Before fitting other modules, check:
  1. the edge vectors in integer zeta-coordinates (is every edge 28 * a unit of Z[zeta12]?);
  2. the internal image x* = sum n_j zeta^(p j) / 28 for p = 5, 7: a model set needs a bounded window,
     so the internal radius must stay fixed from K3 to K5 while the physical radius grows ~ 4x per level;
  3. internal spread in the vortex subset as well.
Usage: python research/internal_check.py OUT.json K [K ...]
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
from collections import Counter
import numpy as np
from gap_collared import build
from ollivier_ricci import vortices


def zc(n, p):
    return n @ np.exp(1j * np.pi / 6 * p * np.arange(4))


def main():
    out = sys.argv[1]; res = []
    for K in map(int, sys.argv[2:]):
        _, _, V, _ = build(K)
        uniq, vid = np.unique(V.reshape(-1, 4), axis=0, return_inverse=True); vid = vid.reshape(V.shape[0], 14)
        dn = (uniq[np.roll(vid, -1, 1)] - uniq[vid]).reshape(-1, 4)
        dn = np.where((dn[:, :1] < 0) | ((dn[:, :1] == 0) & (dn[:, 1:2] < 0)), -dn, dn)
        ec = Counter(map(tuple, dn.tolist()))
        r = dict(K=K, n_vertices=len(uniq), edge_vectors=[dict(n=list(k), count=c) for k, c in ec.most_common(40)],
                 edge_gcd=int(np.gcd.reduce(np.abs(dn).ravel())),
                 edges_div28=bool((dn % 28 == 0).all()), n_div28=bool((uniq % 28 == 0).all()))
        x = zc(uniq.astype(float), 1) / 28; c = x.mean(); R = np.abs(x - c)
        v = vortices(K, len(uniq))
        r["phys_radius"] = float(R.max())
        for p in (5, 7):
            xs = zc(uniq.astype(float), p) / 28; cs = xs.mean(); Rs = np.abs(xs - cs)
            r[f"int{p}"] = dict(radius=float(Rs.max()), q50=float(np.quantile(Rs, .5)), q99=float(np.quantile(Rs, .99)),
                                vortex_radius=float(Rs[v].max()) if v.any() else None)
        print(json.dumps(r)[:1500], flush=True); res.append(r)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
