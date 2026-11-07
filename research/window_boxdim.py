"""Item B, step 3 (heuristic): box-counting dimension of the window boundary.

Star images of Delta-supertile tile positions at level K (frame F on even K, F*Mi on odd K; window_star.py) fill
the window.  On grids of side eps we count occupied cells and boundary cells (occupied with an empty 4-neighbour),
keeping only scales where the mean point count per occupied cell is >= MINPTS so that sampling holes are rare.
The log-log slope of boundary-cell counts against 1/eps is compared with BGMS: log(5+2 sqrt6)/log(4+sqrt15).
Heuristic: finite sampling biases the slope; this is a consistency check, not a proof.
Usage: python research/window_boxdim.py K OUT.json
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from window_star import tile_maps

MINPTS = 20

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out = sys.argv[2] if len(sys.argv) > 2 else "window_boxdim.json"
    import lift4d_matrix as LM
    G = LM.exact_bounded_plane()
    Gr = np.array(G[:, [0, 2, 1, 3]].evalf(20).tolist(), float)
    F = Gr @ np.linalg.inv(np.array(LM.MIRROR.lin().tolist(), float))
    res = {"K": K, "target": float(np.log(5 + 2 * 6 ** .5) / np.log(4 + 15 ** .5)), "levels": []}
    for k, pos in tile_maps(K):
        if k < K - 1:
            continue
        P = np.array(pos, float) @ (F if k % 2 == 0 else Gr).T / 28
        lo = P.min(0); span = float((P.max(0) - lo).max())
        rows = []
        for j in range(2, 12):
            m = 2 ** j
            idx = np.minimum(((P - lo) / span * m).astype(int), m - 1)
            occ = np.zeros((m + 2, m + 2), bool); occ[idx[:, 0] + 1, idx[:, 1] + 1] = True
            n_occ = int(occ.sum())
            inner = occ[1:-1, 1:-1] & occ[:-2, 1:-1] & occ[2:, 1:-1] & occ[1:-1, :-2] & occ[1:-1, 2:]
            n_bd = n_occ - int(inner.sum())
            rows.append(dict(m=m, occ=n_occ, bd=n_bd, ppc=len(P) / n_occ))
        good = [r for r in rows if r["ppc"] >= MINPTS]
        slope = None
        if len(good) >= 3:
            slope = float(np.polyfit(np.log([r["m"] for r in good]), np.log([r["bd"] for r in good]), 1)[0])
        local = [float(np.log(b["bd"] / a["bd"]) / np.log(2)) for a, b in zip(good, good[1:])]
        res["levels"].append(dict(k=k, n=len(P), rows=rows, slope=slope, local_slopes=local))
        print(k, len(P), "slope", slope, "local", np.round(local, 3).tolist(), "target", round(res["target"], 4), flush=True)
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
