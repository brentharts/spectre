"""Item B, step 1b: the contracting (internal) plane in Z[zeta_12] ring coordinates, found from the translations.

Step 1 (window_gifs.py) projected with lift4d_matrix.exact_bounded_plane through a guessed ring->pqrs map and saw
expansion by lambda^2, so here the plane is derived directly.  The supertile placement translations b_{k,i} in
Z^4 = Z[zeta] (handedness.py recursion, linear parts of period 2) should satisfy b_{k+2,i} = T b_{k,i} for one
rational 4x4 T.  We solve for T exactly from levels k, k+2 (Fractions), check it on all levels and slots, factor
its charpoly, take the left eigenspace for the eigenvalues of modulus < 1 (expected 8 - ... = lambda^-2) as the star map
G (2x4 real), and re-run step 1's checks with it: equivariance of zeta and conj (does G L = R G?) and decay ratios.
Usage: python research/window_gifs_plane.py K OUT.json
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
import sympy as sp
from handedness import mul, conj, sub, app, comp, zeta_pow, ref_points, ZERO, ONE

def translations(K):
    pts = ref_points(); quad = [pts[3], pts[5], pts[7], pts[11]]
    mir = ([-1, 0, 0, 0], ZERO, True); Id = (ONE, ZERO, False)
    out = []
    for k in range(1, K + 1):
        bq = quad; total = 0; trs = [Id]
        for angle, src, dst in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            total += angle; r = zeta_pow(total // 30)
            tq = [mul(r, z) for z in bq]
            trs.append((r, sub(app(trs[-1], bq[src]), tq[dst]), False))
        trs = [comp(mir, tr) for tr in trs]
        quad = [app(trs[6], bq[2]), app(trs[5], bq[1]), app(trs[3], bq[2]), app(trs[0], bq[1])]
        out.append(trs)
    return out

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = sys.argv[2] if len(sys.argv) > 2 else "window_gifs_plane.json"
    lev = translations(K)
    res = {"K": K}
    # exact T from all slots of a mid level (need 4 independent b's)
    k0 = 10
    A = sp.Matrix([lev[k0][i][1] for i in range(1, 8)]).T          # 4 x 7
    B = sp.Matrix([lev[k0 + 2][i][1] for i in range(1, 8)]).T
    assert A.rank() == 4, "translations at level k0 do not span Z^4"
    T = (B * A.T * (A * A.T).inv()).applyfunc(sp.nsimplify)
    ok = all(sp.Matrix(lev[k + 2][i][1]) == T * sp.Matrix(lev[k][i][1]) for k in range(K - 2) for i in range(8))
    x = sp.symbols("x")
    cp = sp.factor(T.charpoly(x).as_expr())
    res.update(T=[[str(v) for v in r] for r in T.tolist()], T_holds_all=bool(ok), charpoly=str(cp))
    print("T", T.tolist(), "holds", ok, "charpoly", cp, flush=True)
    # contracting left eigenspace
    Tf = np.array(T.tolist(), dtype=float)
    w, V = np.linalg.eig(Tf.T)
    small = [j for j in range(4) if abs(w[j]) < 1]
    res["eigs"] = [[float(np.real(v)), float(np.imag(v))] for v in w]
    print("eigenvalues", np.round(w, 8), flush=True)
    if len(small) != 2:
        res["error"] = "contracting space not 2-dimensional"; json.dump(res, open(out, "w"), indent=1); return
    Vs = V[:, small]
    G = np.real(Vs.T) if np.allclose(np.imag(Vs), 0) else np.vstack([np.real(Vs[:, 0]), np.imag(Vs[:, 0])])
    G = np.linalg.qr(G.T)[0].T                                     # orthonormal rows, 2 x 4
    res["G"] = G.tolist()
    def lin(f): return np.array([f([int(i == j) for i in range(4)]) for j in range(4)], dtype=float).T
    for name, f in (("zeta", lambda e: mul([0, 1, 0, 0], e)), ("conj", conj)):
        L = lin(f); GL = G @ L; R = GL @ G.T
        err = float(np.abs(R @ G - GL).max())
        res[name] = dict(R=R.tolist(), err=err, det=float(np.linalg.det(R)))
        print(name, "R", np.round(R, 6).tolist(), "equivariance err", err, "det", np.linalg.det(R), flush=True)
    ratios = []
    for k in range(K - 2):
        ratios.append([float(np.linalg.norm(G @ np.array(lev[k + 2][i][1], float)) / np.linalg.norm(G @ np.array(lev[k][i][1], float)))
                       if np.linalg.norm(G @ np.array(lev[k][i][1], float)) > 1e-9 else None for i in range(8)])
    res["ratios"] = ratios; res["lambda_minus2"] = float(1 / (4 + 15 ** 0.5))
    for k in (1, K // 2, K - 3): print("k", k + 1, "ratios", ratios[k])
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
