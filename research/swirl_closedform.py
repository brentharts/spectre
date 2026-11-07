"""Item 5: Z_k for any level by the exact moment recursion (no tile enumeration).
Each label keeps N, S = sum p, M0 = sum u, M1 = sum conj(p) u over its tiles, where p is the tile
centroid and u = exp(i theta) the placement's linear part.  An affine map z -> a z + b (or
a conj(z) + b) updates these linearly; Z = M1 - conj(S/N) M0.  Arithmetic in mpmath at 120 digits.
Validates against research/swirl_exact.py-style enumeration for k <= 5 (uses spectre_patch)."""
import sys, json; sys.path.insert(0, ".")
import sympy as sp, numpy as np
class _M: pass
mp = _M()
import spectre_patch as spm, spectre_operator as so
I_ = sp.I
mp.sqrt = sp.sqrt; mp.mpf = lambda x: sp.Rational(x) if not isinstance(x, sp.Basic) else x
mp.mpc = lambda x=0, y=0: sp.nsimplify(x, [sp.sqrt(3)]) + I_ * sp.nsimplify(y, [sp.sqrt(3)])
mp.conj = lambda z: sp.expand(sp.conjugate(z))
mp.expjpi = lambda q: sp.expand(sp.cos(sp.pi * q) + I_ * sp.sin(sp.pi * q))
mp.arg = lambda z: sp.arg(z); mp.pi = sp.pi; mp.degrees = lambda x: x * 180 / sp.pi
mp.nstr = lambda x, n: str(sp.N(x, n))
R3 = mp.sqrt(3)
def E(deg): return mp.expjpi(mp.mpf(deg) / 180)
# map = (a, b, m): z -> a*(conj z if m else z) + b
def app(f, z): a, b, m = f; return sp.expand(a * (mp.conj(z) if m else z) + b)
def comp(f, g):  # f o g
    a, b, m = f; c, d, n = g
    return (a * (mp.conj(c) if m else c), app(f, d), m ^ n)
def ref_points():
    h = mp.mpf(1) / 2; s = R3 / 2; a = b = mp.mpf(1)
    P = [(0, 0), (a, 0), (a + h, -s), (a + h + s, -s + h), (a + h + s, -s + b + h), (2 * a + h + s, -s + b + h),
         (3 * a + s, b + h), (3 * a, 2 * b), (3 * a - s, 2 * b - h), (2 * a + h - s, s + 2 * b - h),
         (a + h - s, s + 2 * b - h), (h - s, s + 2 * b - h), (-s, 2 * b - h), (0, b)]
    return [mp.mpc(x, y) for x, y in P]
def mat_to_map(M):
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]; m = det < 0
    a = mp.mpc(M[0, 0], M[1, 0]) if not m else mp.mpc(-M[0, 0], -M[1, 0])
    # snap the linear part to the 12th roots of unity (exact angles)
    k = int(round(float(mp.arg(a)) / (mp.pi / 6))) % 12; a = E(30 * k)
    return a, m
def run(K):
    pts = ref_points(); quad = [pts[3], pts[5], pts[7], pts[11]]
    # base tile: placement of tile = level-0 motion o inv(reference_to_tile); centroid c0 in tile frame
    base = {}
    for sh in (0, 1):
        T = spm._reference_to_tile(1.0, 1.0); Ti = np.linalg.inv(T)
        a, m = mat_to_map(Ti)
        c0 = so.vertices(1.0, 1.0).mean(0)
        # exact: inverse of the swap map x<->y with offset; recompute offset in mp from vertex 0
        ours0 = mp.mpc(*so.vertices(1.0, 1.0)[0])
        # T: z -> swap(z) + t with swap(z) = i conj z; t = ours0 - swap(ref[2])
        t = ours0 - I_ * mp.conj(pts[2])
        # inv T: z -> swap(z - t) = i conj(z - t)
        Tinv = (mp.mpc(0, 1), -I_ * mp.conj(t), True)
        cen = sum(ref_points()[(j + 2) % 14] for j in range(14)) / 14  # mean of ours = T(mean ref)
        cen = app((mp.mpc(0, 1), t, True), cen)
        base[sh] = (Tinv, cen)
    def mom(tiles):  # tiles: list of (shape, map)
        N = len(tiles); S = M0 = M1 = mp.mpc(0)
        for sh, g in tiles:
            P = comp(g, base[sh][0]); p = app(P, base[sh][1]); u = P[0]
            S += p; M0 += u; M1 += mp.conj(p) * u
        return [N, S, M0, M1, tiles[0] and comp(tiles[0][1], base[tiles[0][0]][0])[2]]
    I = (mp.mpc(1), mp.mpc(0), False)
    lev = {lab: mom([(0, I)]) for lab in spm.TILE_NAMES if lab != "Gamma"}
    lev["Gamma"] = mom([(0, I), (1, (E(30), pts[8], False))])
    def xf(f, st):
        N, S, M0, M1, chir = st; a, b, m = f
        if not m: return [N, a * S + b * N, a * M0, M1 + mp.conj(b) * a * M0, chir]
        return [N, a * mp.conj(S) + b * N, a * mp.conj(M0), mp.conj(M1) + mp.conj(b) * a * mp.conj(M0), not chir]
    out = []
    for k in range(1, K + 1):
        bq = quad; total = 0; trs = [I]
        for angle, src, dst in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            total += angle; r = (E(total), mp.mpc(0), False)
            tq = [app(r, z) for z in bq]
            trs.append((E(total), app(trs[-1], bq[src]) - tq[dst], False))
        mir = (mp.mpc(-1), mp.mpc(0), True)
        trs = [comp(mir, t) for t in trs]
        quad = [app(trs[6], bq[2]), app(trs[5], bq[1]), app(trs[3], bq[2]), app(trs[0], bq[1])]
        new = {}
        for lab, sub in spm.SUBSTITUTIONS.items():
            acc = None
            for name, t in zip(sub, trs):
                if not name: continue
                s = xf(t, lev[name])
                acc = s if acc is None else [acc[0] + s[0], acc[1] + s[1], acc[2] + s[2], acc[3] + s[3], acc[4]]
            new[lab] = acc
        lev = new
        N, S, M0, M1, chir = lev["Delta"]
        if chir:  # rigid flip z -> -conj z of the whole patch
            N, S, M0, M1, _ = xf(mir, lev["Delta"])
        Z = M1 - mp.conj(S / N) * M0
        Z = sp.expand(Z); out.append((k, N, Z)); print(json.dumps(dict(k=k, n=N, Z=str(sp.N(Z, 20)))), flush=True)
    return out

if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    out = run(K); Zs = {k: sp.nsimplify(sp.expand(Z)) for k, _, Z in out}
    L = 4 + sp.sqrt(15); A, B, C, D, E_, F, G = sp.symbols('A B C D E F G'); Lc = 4 - sp.sqrt(15)
    form = lambda k: (A * k + B) * L**k + C * (-L)**k + D + E_ * (-1)**k + F * Lc**k + G * (-Lc)**k
    k0 = K - 7
    sol = sp.solve([sp.expand(form(k) - Zs[k]) for k in range(k0, k0 + 7)], [A, B, C, D, E_, F, G], dict=True)[0]
    sol = {v: sp.radsimp(sp.nsimplify(sp.expand(e))) for v, e in sol.items()}
    for v, e in sol.items(): print(v, "=", e, "  ~", sp.N(e, 25))
    for k in range(1, K + 1):
        r = sp.N(sp.expand(form(k).subs(sol) - Zs[k]), 30)
        print("k", k, "residual", sp.N(abs(r), 5))
    print("Im A =", sp.N(sp.im(sol[A]), 30))
