"""Item A (handedness theorem), phase 1: the moment recursion in exact Z[zeta_12] arithmetic.

Same recursion as research/swirl_recursion.py, but every quantity is an element of Z[zeta], zeta = e^{i pi/6},
stored as 4 integers in the basis 1, zeta, zeta^2, zeta^3 (zeta^4 = zeta^2 - 1).  Positions are scaled by 28
(1/2 from sqrt3/2, 1/14 from the tile centroid), so everything is an integer.

For the Delta supertile at levels k = 1..K this records N, S = sum p, M0 = sum u, M1 = sum conj(p) u
(before the rigid flip that swirl_recursion applies when the supertile is mirrored) and
    W_k = N M1 - conj(S) M0 = N^2 Z_k   (unflipped),
so sign Im Z_k = sign Im W_k exactly.

Certified recurrences.  N, S, M0 and M1 of a label are linear in a state vector that evolves by a fixed
integer matrix: N (9), M0 (9x4), S together with quad (x) N (36 + 16*9), and M1 together with
quad (x) M0 (36 + 16*36 = 612).  So every rational coordinate of these sequences satisfies a monic integer
recurrence of order D <= 612.  We find the minimal recurrence (order L) by Berlekamp-Massey mod a large
prime, lift it to Z, and check it exactly on all K terms.  If K >= D + L the recurrence holds for every k
(two recurrence sequences of orders D and L that agree on D + L terms agree forever).

Usage: python research/handedness.py K OUT.json
"""
import sys, json
from fractions import Fraction as Fr

D_BOUND = {"N": 9, "M0": 36, "S": 36 + 16 * 9, "M1": 36 + 16 * 36}

# ---------- Z[zeta_12] ----------
def add(x, y): return [a + b for a, b in zip(x, y)]
def sub(x, y): return [a - b for a, b in zip(x, y)]
def smul(c, x): return [c * a for a in x]
def mul(x, y):
    c = [0] * 7
    for i in range(4):
        if x[i]:
            for j in range(4):
                c[i + j] += x[i] * y[j]
    # zeta^6 = -1, zeta^5 = zeta^3 - zeta, zeta^4 = zeta^2 - 1
    r = c[:4]
    r[0] -= c[6]
    r[3] += c[5]; r[1] -= c[5]
    r[2] += c[4]; r[0] -= c[4]
    return r
def conj(x):
    a0, a1, a2, a3 = x
    return [a0 + a2, a1, -a2, -a1 - a3]
ZERO = [0, 0, 0, 0]; ONE = [1, 0, 0, 0]
def zeta_pow(k):
    r = ONE
    for _ in range(k % 12): r = mul(r, [0, 1, 0, 0])
    return r
I_ = [0, 0, 0, 1]
def to_complex(x):
    import cmath
    z = cmath.exp(1j * cmath.pi / 6)
    return sum(a * z ** j for j, a in enumerate(x))
def im_sign(x):
    """Exact sign of Im x.  Im = a1/2 + a2*sqrt3/2 + a3, so 2 Im = (a1 + 2 a3) + a2 sqrt3."""
    p, q = x[1] + 2 * x[3], x[2]
    if p >= 0 and q >= 0: return (p > 0 or q > 0) and 1 or 0
    if p <= 0 and q <= 0: return -1 if (p < 0 or q < 0) else 0
    v = p * p - 3 * q * q  # sign of p + q sqrt3 when p, q have opposite signs
    return (1 if p > 0 else -1) if v > 0 else ((1 if q > 0 else -1) if v < 0 else 0)

# ---------- geometry, scaled by 28 ----------
SC = 28
SQ3 = [0, 2, 0, -1]  # sqrt3 = zeta + conj(zeta) = 2 zeta - zeta^3
def coord(p, q):
    """28 * (p + q * sqrt3/2) with p, q rational (multiples of 1/2): an element of Z[zeta]."""
    r = add([int(Fr(SC) * Fr(p)), 0, 0, 0], smul(int(Fr(SC, 2) * Fr(q)), SQ3))
    return r
def pt(xp, xq, yp, yq):
    return add(coord(xp, xq), mul(I_, coord(yp, yq)))
def ref_points():
    h = Fr(1, 2)
    # (x_rational, x_sqrt3/2 coeff, y_rational, y_sqrt3/2 coeff); s = sqrt3/2, a = b = 1
    P = [(0, 0, 0, 0), (1, 0, 0, 0), (1 + h, 0, 0, -1), (1 + h, 1, h, -1), (1 + h, 1, 1 + h, -1), (2 + h, 1, 1 + h, -1),
         (3, 1, 1 + h, 0), (3, 0, 2, 0), (3, -1, 2 - h, 0), (2 + h, -1, 2 - h, 1),
         (1 + h, -1, 2 - h, 1), (h, -1, 2 - h, 1), (0, -1, 2 - h, 0), (0, 0, 1, 0)]
    return [pt(*c) for c in P]

# maps (a, b, m): z -> a*(conj z if m else z) + b ; a a unit (unscaled), b scaled
def app(f, z): a, b, m = f; return add(mul(a, conj(z) if m else z), b)
def comp(f, g):
    a, b, m = f; c, d, n = g
    return (mul(a, conj(c) if m else c), app(f, d), m ^ n)

def run(K, labels=("Delta",)):
    import spectre_patch as spm
    pts = ref_points(); quad = [pts[3], pts[5], pts[7], pts[11]]
    t = smul(-1, mul(I_, conj(pts[2])))           # ours0 = 0
    Tinv = (I_, smul(-1, mul(I_, conj(t))), True)
    tot = ZERO
    for j in range(14): tot = add(tot, pts[(j + 2) % 14])
    assert all(v % 14 == 0 for v in tot)
    cen = app((I_, t, True), [v // 14 for v in tot])
    base = (Tinv, cen)
    def mom(tiles):
        N = len(tiles); S = M0 = M1 = ZERO
        for g in tiles:
            P = comp(g, base[0]); p = app(P, base[1]); u = P[0]
            S = add(S, p); M0 = add(M0, u); M1 = add(M1, mul(conj(p), u))
        return [N, S, M0, M1, comp(tiles[0], base[0])[2]]
    Id = (ONE, ZERO, False)
    lev = {lab: mom([Id]) for lab in spm.TILE_NAMES if lab != "Gamma"}
    lev["Gamma"] = mom([Id, (zeta_pow(1), pts[8], False)])
    def xf(f, st):
        N, S, M0, M1, chir = st; a, b, m = f
        if not m:
            return [N, add(mul(a, S), smul(N, b)), mul(a, M0), add(M1, mul(mul(conj(b), a), M0)), chir]
        cM0 = conj(M0)
        return [N, add(mul(a, conj(S)), smul(N, b)), mul(a, cM0), add(conj(M1), mul(mul(conj(b), a), cM0)), not chir]
    mir = ([-1, 0, 0, 0], ZERO, True)
    rec = []
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
            acc = None
            for name, tr in zip(subs, trs):
                if not name: continue
                s = xf(tr, lev[name])
                acc = s if acc is None else [acc[0] + s[0], add(acc[1], s[1]), add(acc[2], s[2]), add(acc[3], s[3]), acc[4]]
            new[lab] = acc
        lev = new
        N, S, M0, M1, chir = lev["Delta"]
        W = sub(smul(N, M1), mul(conj(S), M0))
        rec.append(dict(k=k, N=N, S=S, M0=M0, M1=M1, W=W, chir=bool(chir)))
    return rec

# ---------- recurrences ----------
P61 = (1 << 61) - 1
def bm_mod(seq, p=P61):
    """Berlekamp-Massey over GF(p): returns c with s[n] = sum_{i=1..L} c[i-1] s[n-i]."""
    s = [x % p for x in seq]
    C, B = [1], [1]; L, m, b = 0, 1, 1
    for n in range(len(s)):
        d = s[n]
        for i in range(1, L + 1): d = (d + C[i] * s[n - i]) % p
        if d == 0: m += 1; continue
        coef = d * pow(b, p - 2, p) % p
        T = C[:]
        C = C + [0] * (len(B) + m - len(C))
        for i in range(len(B)): C[i + m] = (C[i + m] - coef * B[i]) % p
        if 2 * L <= n: L, B, b, m = n + 1 - L, T, d, 1
        else: m += 1
    C = C + [0] * (L + 1 - len(C))
    return [(-C[i]) % p for i in range(1, L + 1)]
def lift(c, p=P61): return [x - p if x > p // 2 else x for x in c]
def check_rec(seq, c):
    L = len(c)
    return all(seq[n] == sum(c[i] * seq[n - 1 - i] for i in range(L)) for n in range(L, len(seq)))
def charpoly(c):
    """x^L - c1 x^{L-1} - ... - cL as a coefficient list, highest degree first."""
    return [1] + [-x for x in c]

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    out = sys.argv[2] if len(sys.argv) > 2 else "handedness.json"
    sys.path.insert(0, ".")
    rec = run(K)
    res = {"K": K, "levels": []}
    # float cross-check against swirl_recursion for small k
    for r in rec[:8]:
        Z = to_complex(r["W"]) / r["N"] ** 2 / SC
        Zf = Z.conjugate() if r["chir"] else Z
        res["levels"].append(dict(k=r["k"], N=r["N"], chir=r["chir"], Z=[Zf.real, Zf.imag]))
    signs = []
    for r in rec:
        s = im_sign(r["W"]); signs.append(-s if r["chir"] else s)  # flipped Z = conj Z
    res["ImZ_sign_all_k"] = sorted(set(signs)); res["ImZ_negative_all_k_upto_K"] = all(s < 0 for s in signs)
    res["chir_pattern"] = [r["chir"] for r in rec[:12]]
    try:
        import sympy as sp; x = sp.symbols("x")
    except ImportError:
        sp = None
    recs = {}
    for name in ("N", "S", "M0", "M1"):
        for j in range(1 if name == "N" else 4):
            seq = [r[name] if name == "N" else r[name][j] for r in rec]
            c = lift(bm_mod(seq)); L = len(c)
            ok = check_rec(seq, c)
            certified = ok and K >= D_BOUND[name] + L
            f = sp.factor_list(sp.Poly(charpoly(c), x).as_expr()) if (L and sp) else None
            recs[f"{name}[{j}]"] = dict(L=L, coeffs=c, holds_on_all_K=ok, certified=certified,
                                         factors=[[str(g), e] for g, e in f[1]] if f else [])
            print(name, j, "L", L, "ok", ok, "certified", certified, recs[f"{name}[{j}]"]["factors"], flush=True)
    res["recurrences"] = recs
    # keep exact data for phase 2 (first 120 levels are plenty for closed forms once recurrences are known)
    res["exact"] = [dict(k=r["k"], N=str(r["N"]), S=[str(v) for v in r["S"]], M0=r["M0"], M1=[str(v) for v in r["M1"]],
                         chir=r["chir"]) for r in rec[:120]]
    json.dump(res, open(out, "w"), indent=1, default=str)
    print("Im Z < 0 for all k <= K:", res["ImZ_negative_all_k_upto_K"], "chir:", res["chir_pattern"])
    print(json.dumps(res["levels"]))

if __name__ == "__main__":
    main()
