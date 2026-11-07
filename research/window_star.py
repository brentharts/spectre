"""Item B, step 1c: which star map makes the spectre tile positions bounded in internal space?

Candidates, all on Z[zeta_12]:
  sigma5: zeta -> e^{5 i pi/6}  (Galois conjugation; commutes with rotations up to zeta -> zeta^5)
  sigma7: zeta -> e^{7 i pi/6}
  F_even/F_odd: the 2D annihilator from lift4d_matrix.exact_bounded_plane (ring order), unmirrored / mirrored.
For each, we compute the star images of all tile positions (anchor points g(0)) of the Delta supertile at levels
k = 1..K and report the radius max |star| per level (bounded => a window exists in that coordinate) and the
first-difference of radii (geometric decay => internal contraction).
Usage: python research/window_star.py K OUT.json
"""
import sys, json, cmath
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from handedness import mul, conj, sub, app, comp, zeta_pow, ref_points, ZERO, ONE

def tile_maps(K):
    import spectre_patch as spm
    pts = ref_points(); quad = [pts[3], pts[5], pts[7], pts[11]]
    mir = ([-1, 0, 0, 0], ZERO, True); Id = (ONE, ZERO, False)
    lev = {lab: [Id] for lab in spm.TILE_NAMES}
    lev["Gamma"] = [Id, (zeta_pow(1), pts[8], False)]
    radii = []
    for k in range(1, K + 1):
        bq = quad; total = 0; trs = [Id]
        for angle, src, dst in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            total += angle; r = zeta_pow(total // 30)
            tq = [mul(r, z) for z in bq]
            trs.append((r, sub(app(trs[-1], bq[src]), tq[dst]), False))
        trs = [comp(mir, tr) for tr in trs]
        quad = [app(trs[6], bq[2]), app(trs[5], bq[1]), app(trs[3], bq[2]), app(trs[0], bq[1])]
        new = {}
        for lab, subs in spm.SUBSTITUTIONS.items():
            new[lab] = [comp(tr, g) for name, tr in zip(subs, trs) if name for g in lev[name]]
        lev = new
        yield k, [g[1] for g in lev["Delta"]]

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = sys.argv[2] if len(sys.argv) > 2 else "window_star.json"
    import lift4d_matrix as LM, sympy as sp
    G = LM.exact_bounded_plane()                                  # (p,q,r,s), patch frame = F * Mi
    Gr = np.array(G[:, [0, 2, 1, 3]].evalf(20).tolist(), float)   # back to ring order: F * Mi
    Mi = np.array(LM.MIRROR.lin().tolist(), float)
    F = Gr @ np.linalg.inv(Mi)
    stars = {
        "sigma5": lambda v: sum(a * cmath.exp(5j * cmath.pi / 6) ** j for j, a in enumerate(v)),
        "sigma7": lambda v: sum(a * cmath.exp(7j * cmath.pi / 6) ** j for j, a in enumerate(v)),
        "F": lambda v: complex(*(F @ np.array(v, float))),
        "F_mirror": lambda v: complex(*(Gr @ np.array(v, float))),
        "physical": lambda v: sum(a * cmath.exp(1j * cmath.pi / 6) ** j for j, a in enumerate(v)),
    }
    res = {name: [] for name in stars}
    for k, pos in tile_maps(K):
        for name, f in stars.items():
            vals = [f(p) / 28 for p in pos]
            res[name].append(dict(k=k, n=len(vals), rmax=max(abs(v) for v in vals),
                                  mean=[float(np.mean([v.real for v in vals])), float(np.mean([v.imag for v in vals]))]))
        print(k, len(pos), {n: round(res[n][-1]["rmax"], 4) for n in stars}, flush=True)
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
