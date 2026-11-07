"""Item B, step 2: does the window star map intertwine the placement symmetries with fixed 2x2 maps?

Star map (job: window_star.py): F on even levels, F*Mi on odd levels, F = 2x4 annihilator in ring order
(lift4d_matrix.exact_bounded_plane, un-mirrored).  For a linear map L on Z^4 = Z[zeta12] and target functional
F_t, source F_s, we ask whether F_t L = R F_s for a 2x2 real R (i.e. L maps ker F_s into ker F_t).
Tested exactly over Q(sqrt15) for L in {zeta, zeta^2, conj, Mi} and all frame pairs.
Usage: python research/window_equiv.py OUT.json
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "research")
import sympy as sp
from handedness import mul, conj

def lin(f):
    return sp.Matrix([f([int(i == j) for i in range(4)]) for j in range(4)]).T

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "window_equiv.json"
    import lift4d_matrix as LM
    G = LM.exact_bounded_plane()
    Gr = G[:, [0, 2, 1, 3]]                       # ring order, = F * Mi
    Mi = sp.Matrix(LM.MIRROR.lin().tolist())
    F = (Gr * Mi.inv()).applyfunc(sp.simplify)
    frames = {"F": F, "FMi": Gr}
    maps = {"zeta": lin(lambda e: mul([0, 1, 0, 0], e)), "zeta2": lin(lambda e: mul([0, 0, 1, 0], e)),
            "conj": lin(conj), "Mi": Mi}
    res = {}
    for ln, L in maps.items():
        for sn, Fs in frames.items():
            for tn, Ft in frames.items():
                FtL = (Ft * L).applyfunc(sp.simplify)
                R = (FtL * Fs.T * (Fs * Fs.T).inv()).applyfunc(lambda v: sp.radsimp(sp.simplify(v)))
                ok = (R * Fs - FtL).applyfunc(sp.simplify) == sp.zeros(2, 4)
                key = f"{ln}:{sn}->{tn}"
                res[key] = dict(ok=bool(ok), R=[[str(v) for v in r] for r in R.tolist()] if ok else None)
                print(key, ok, R.evalf(6).tolist() if ok else "", flush=True)
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
