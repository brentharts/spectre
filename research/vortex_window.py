"""R2: are Brent's Z2 vortices a model set?  Map every interior vertex of the level-K Delta patch to internal
space with the item-B star map (F or F*Mirror, whichever gives bounded images at this K; both are reported),
and test whether vortex and non-vortex vertices occupy separate regions (a sub-window W_v of W).

Heuristic separation tests (a finite patch cannot prove a window):
  - kNN purity: fraction of each vertex's k nearest internal neighbours with the same vortex flag;
    a model set gives purity -> 1 as K grows (only points near dW_v are mixed).
  - mixed fraction: vertices whose nearest internal neighbour has the other flag; for a window with fractal
    boundary of dimension d < 2 this decays with K.
  - area ratio of the occupied cells (grid occupancy) vs the vortex density from R1.
Also: which (level-1 supertile label, slot) carries each vortex (substitution rule for vortices).
Usage: python research/vortex_window.py K OUT.json
"""
import sys, json
from collections import Counter
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from scipy.spatial import cKDTree
from gap_collared import build
from vortex_density import star_data
import lift4d_matrix as LM


def main():
    K = int(sys.argv[1]); out = sys.argv[2]
    names, levels, V, PT = build(K)
    npc = len(PT)
    vid, interior, flux, npi, val, members, _ = star_data(V, PT, npc)
    vort = interior & (flux == -1) & (npi == 0)
    Vflat = V.reshape(-1, 4); vid_flat = vid.ravel()
    pos = np.zeros((int(vid.max()) + 1, 4)); pos[vid_flat] = Vflat
    G = LM.exact_bounded_plane()
    Gr = np.array(G[:, [0, 2, 1, 3]].evalf(20).tolist(), float)
    Mi = np.array(LM.MIRROR.lin().tolist(), float)
    F = Gr @ np.linalg.inv(Mi)
    res = dict(K=K, interior=int(interior.sum()), vortices=int(vort.sum()), maps={})
    iv = np.nonzero(interior)[0]
    lab = vort[iv]
    for name, P in (("F", F), ("F_mirror", Gr)):
        X = pos[iv] @ P.T
        r = np.linalg.norm(X - X.mean(0), axis=1)
        tr = cKDTree(X)
        info = dict(rmax=float(r.max()), r_rms=float(np.sqrt((r ** 2).mean())))
        for k in (1, 6):
            _, nn = tr.query(X, k + 1)
            same = (lab[nn[:, 1:]] == lab[:, None]).mean(1)
            info[f"purity_k{k}_all"] = float(same.mean())
            info[f"purity_k{k}_vortex"] = float(same[lab].mean()) if lab.any() else None
        _, nn = tr.query(X, 2)
        info["mixed_fraction"] = float((lab[nn[:, 1]] != lab).mean())
        # occupancy area ratio on a grid
        g = 64; lo, hi = X.min(0), X.max(0)
        cell = lambda Y: set(map(tuple, np.floor((Y - lo) / (hi - lo + 1e-12) * g).astype(int).tolist()))
        info["cell_ratio_vortex_over_all"] = len(cell(X[lab])) / len(cell(X))
        res["maps"][name] = info
        print(name, json.dumps(info), flush=True)
        np.save(f"vw_{name}_K{K}.npy", np.c_[X, lab])
    # vortex location by level-1 supertile label and slot of the vertex's pieces
    slot0 = levels[0]["slot"]; par1 = levels[0]["parent"]; lab1 = levels[1]["lab"]
    loc = Counter()
    for v in np.nonzero(vort)[0]:
        for p in members[int(v)]:
            t = PT[p]
            loc[(names[int(lab1[par1[t]])], int(slot0[t]))] += 1
    res["vortex_piece_slots"] = {f"{a}:{s}": c for (a, s), c in sorted(loc.items())}
    res["empirical_vortex_fraction_of_interior"] = float(vort.sum() / interior.sum())
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
