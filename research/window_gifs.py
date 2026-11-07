"""Item B, step 1: the star images of the supertile placement maps, towards an explicit GIFS for the window.

Each level k places the level-(k-1) supertiles of a label by 8 affine maps g_i(z) = a_i (conj z if m_i) + b_i on
Z[zeta_12] (same maps for every label, from the quad recursion of research/handedness.py).  The window is the closure
of the star images of all tile positions, so W = { sum_k  (linear parts) b*_{k, d_k} }: a Rauzy-type digit set.
This script checks, exactly where possible:
  1. the internal plane G (lift4d_matrix.exact_bounded_plane) is invariant under multiplication by zeta and under
     conj: G . L = R(L) . G for a 2x2 real matrix R(L)   (so the linear parts act on internal space);
  2. the star images b*_{k,i} = G b_{k,i} of the translations, k = 1..K: their decay ratio between k and k+2
     (expected lambda^-2 times a fixed rotation) and the non-decaying (+-1 eigenvalue) remainder.
Output: JSON with the R matrices, b*_{k,i} as floats, and fitted ratios.
Usage: python research/window_gifs.py K OUT.json
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
import sympy as sp
from handedness import mul, conj, sub, app, comp, zeta_pow, ref_points, ZERO, ONE

def ring_to_pqrs(c): return [c[0], c[2], c[1], c[3]]

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    out = sys.argv[2] if len(sys.argv) > 2 else "window_gifs.json"
    import lift4d_matrix as LM
    G = LM.exact_bounded_plane()                      # 2 x 4 on (p,q,r,s), entries in Q(sqrt15)
    Gf = np.array(G.evalf(30).tolist(), dtype=float)
    res = {"G": [[str(v) for v in row] for row in G.tolist()]}
    # 1. equivariance: matrices of z -> zeta z and z -> conj z on (p,q,r,s)
    def lin_matrix(f):
        cols = []
        for j in range(4):
            e = [0, 0, 0, 0]; e[j] = 1                 # ring basis vector zeta^j
            cols.append(ring_to_pqrs(f(e)))
        M = sp.Matrix(cols).T                         # columns indexed by ring basis
        P = sp.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])   # ring -> pqrs permutation
        return M * P.T                                # acts on pqrs
    out_eq = {}
    for name, f in (("zeta", lambda e: mul([0, 1, 0, 0], e)), ("conj", conj)):
        L = lin_matrix(f)
        GL = (G * L).applyfunc(sp.nsimplify)
        # solve R G = G L for 2x2 R (G has rank 2): R = GL G^T (G G^T)^-1
        R = (GL * G.T * (G * G.T).inv()).applyfunc(lambda v: sp.nsimplify(sp.radsimp(sp.simplify(v))))
        ok = (R * G - GL).applyfunc(sp.simplify) == sp.zeros(2, 4)
        out_eq[name] = dict(R=[[str(v) for v in r] for r in R.tolist()], equivariant=bool(ok),
                            R_float=np.array(R.evalf(20).tolist(), dtype=float).tolist())
        print(name, "equivariant", ok, R.evalf(6).tolist(), flush=True)
    res["equivariance"] = out_eq
    # 2. placement maps per level (same recursion as handedness.run, kept here to record trs)
    pts = ref_points(); quad = [pts[3], pts[5], pts[7], pts[11]]
    mir = ([-1, 0, 0, 0], ZERO, True); Id = (ONE, ZERO, False)
    levels = []
    for k in range(1, K + 1):
        bq = quad; total = 0; trs = [Id]
        for angle, src, dst in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            total += angle; r = zeta_pow(total // 30)
            tq = [mul(r, z) for z in bq]
            trs.append((r, sub(app(trs[-1], bq[src]), tq[dst]), False))
        trs = [comp(mir, tr) for tr in trs]
        quad = [app(trs[6], bq[2]), app(trs[5], bq[1]), app(trs[3], bq[2]), app(trs[0], bq[1])]
        bs = [Gf @ np.array(ring_to_pqrs(t[1]), dtype=float) for t in trs]
        levels.append(dict(k=k, lin=[[t[0], int(t[2])] for t in trs], bstar=[b.tolist() for b in bs],
                           b_int_maxabs=max(abs(v) for t in trs for v in t[1])))
    res["levels"] = levels
    # decay: |b*_{k+2}| / |b*_k| per slot, and the linear parts' period
    ratios = []
    for k in range(K - 2):
        r = []
        for i in range(8):
            a, b = np.array(levels[k]["bstar"][i]), np.array(levels[k + 2]["bstar"][i])
            r.append(float(np.linalg.norm(b) / np.linalg.norm(a)) if np.linalg.norm(a) > 1e-12 else None)
        ratios.append(r)
    res["ratio_k_kplus2"] = ratios
    res["lin_period2"] = all(levels[k]["lin"] == levels[k + 2]["lin"] for k in range(K - 2))
    res["lambda_minus2"] = float(1 / (4 + 15 ** 0.5))
    print("linear parts period 2:", res["lin_period2"], flush=True)
    for k in (0, 1, K // 2, K - 3): print("k", k + 1, "ratios", np.round(ratios[k], 6).tolist() if None not in ratios[k] else ratios[k])
    print("lambda^-2 =", res["lambda_minus2"])
    json.dump(res, open(out, "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
