"""Item A, phase 2: closed form of the swirl slope A from the exact data of research/handedness.py.

Input: the JSON written by handedness.py (exact N, S, M0, M1 of the Delta supertile for k = 1..120).
W_k = N M1 - conj(S) M0 (unflipped, scaled by 28) has 4 integer coordinates in the zeta_12 basis.
Each coordinate is a product-sum of certified recurrence sequences, so it satisfies a recurrence of order
<= 2*14 + 12*4 = 76.  We find its minimal recurrence (order L), check it exactly on all 120 terms, and
certify it when 120 >= 76 + L.

Asymptotics.  The generating function is P(x)/Q(x).  The dominant poles are at x = 1/rho with
rho = +-lambda^4, lambda^4 = 31 + 8 sqrt15, and they are double, so W_k = k(a rho^k + c (-rho)^k) + ...
With N_k = alpha lambda^{2k} + beta lambda^{-2k}, and Z_k = Y_k/(28 N_k^2) for odd k (chir False) and its
conjugate for even k:
    A_odd = (a - c)/(28 alpha^2),  A_even = conj(a + c)/(28 alpha^2),  Z_k = A_par k + B_par + o(1).
Everything is done in Q(sqrt3, sqrt15) with sympy, so the signs of Im A_par are exact.
"""
import sys, json
import sympy as sp
from fractions import Fraction as Fr

sys.path.insert(0, "research")
from handedness import mul, conj, sub, smul, bm_mod, lift, check_rec

x = sp.symbols("x")
s15, s3 = sp.sqrt(15), sp.sqrt(3)
L4 = 31 + 8 * s15

def gf(seq, c):
    """P/Q for s_1, s_2, ... with s_n = sum c_i s_{n-i}: Q = 1 - sum c_i x^i, P = (Q * sum s_n x^{n-1}) truncated."""
    L = len(c)
    Q = 1 - sum(sp.Integer(c[i]) * x ** (i + 1) for i in range(L))
    S = sum(sp.Integer(seq[n]) * x ** n for n in range(L))
    P = sp.expand(Q * S)
    P = sum(P.coeff(x, n) * x ** n for n in range(L))
    return sp.Poly(P, x), sp.Poly(Q, x)

def double_pole_coeff(P, Q, rho):
    """For Q with a double root at x0 = 1/rho: coefficient a in s_{n+1} ~ a (n+1) rho^n... returned as the
    coefficient of k rho^k in s_k (sequence indexed k = 1, 2, ...)."""
    x0 = sp.nsimplify(1 / rho)
    # Q(x) = (1 - rho x)^2 H(x)
    H = sp.Poly(sp.cancel(Q.as_expr() / (1 - rho * x) ** 2), x)
    # s_{n} (n = index from 0 in the series, s_n = term k = n+1) ~ (n+1) rho^n P(x0)/H(x0)
    # so term_k ~ k rho^{k-1} P/H = k rho^k (P/H)/rho
    v = sp.simplify(P.as_expr().subs(x, x0) / H.as_expr().subs(x, x0) / rho)
    return sp.radsimp(v)

def main():
    data = json.load(open(sys.argv[1]))["exact"]
    out = sys.argv[2]
    K = len(data)
    Ns = [int(d["N"]) for d in data]
    Ws = []
    for d in data:
        N = int(d["N"]); S = [int(v) for v in d["S"]]; M0 = [int(v) for v in d["M0"]]; M1 = [int(v) for v in d["M1"]]
        Ws.append(sub(smul(N, M1), mul(conj(S), M0)))
    res = {"K": K}
    # N closed form
    cN = lift(bm_mod(Ns)); assert check_rec(Ns, cN) and len(cN) == 2
    lam2 = 4 + s15
    al, be = sp.symbols("al be")
    sol = sp.solve([al * lam2 + be / lam2 - Ns[0], al * lam2 ** 2 + be / lam2 ** 2 - Ns[1]], [al, be])
    alpha = sp.radsimp(sol[al]); res["alpha"] = str(alpha)
    coords = []
    for j in range(4):
        seq = [w[j] for w in Ws]
        c = lift(bm_mod(seq)); L = len(c); ok = check_rec(seq, c)
        P, Q = gf(seq, c)
        fac = sp.factor_list(Q.as_expr())[1]
        a = double_pole_coeff(P, Q, L4); cc = double_pole_coeff(P, Q, -L4)
        coords.append((a, cc))
        res[f"W[{j}]"] = dict(L=L, holds=ok, certified=ok and K >= 76 + L, Q_factors=[[str(g), e] for g, e in fac],
                              a=str(a), c=str(cc))
        print("W", j, "L", L, ok, K >= 76 + L, fac, flush=True)
    zeta = sp.Rational(1, 2) * s3 + sp.I / 2
    def assemble(vals):
        return sp.expand(sum(v * zeta ** j for j, v in enumerate(vals)))
    a_c = assemble([a for a, _ in coords]); c_c = assemble([c for _, c in coords])
    den = 28 * alpha ** 2
    A_odd = sp.radsimp(sp.expand((a_c - c_c) / den))
    A_even = sp.radsimp(sp.expand(sp.conjugate((a_c + c_c) / den)))
    for name, A in (("A_odd", A_odd), ("A_even", A_even)):
        re, im = sp.radsimp(sp.re(A)), sp.radsimp(sp.im(A))
        im_neg = bool(sp.simplify(im) < 0)  # exact: sympy decides the sign of an element of Q(sqrt3, sqrt5)
        res[name] = dict(re=str(sp.nsimplify(re)), im=str(sp.nsimplify(im)), re_float=float(re), im_float=float(im),
                         im_negative_exact=im_neg)
        print(name, "Re", re, "~", float(re), "Im", im, "~", float(im), "Im<0:", im_neg, flush=True)
    # empirical check of the slope against the exact data
    import cmath
    z = cmath.exp(1j * cmath.pi / 6)
    Zs = []
    for d, w in zip(data, Ws):
        Y = sum(v * z ** j for j, v in enumerate(w)) / 28 / int(d["N"]) ** 2
        Zs.append(Y.conjugate() if d["chir"] else Y)
    res["slope_check"] = [[k, (Zs[k] - Zs[k - 2]).real / 2, (Zs[k] - Zs[k - 2]).imag / 2] for k in (60, 61, 118, 119)]
    print(res["slope_check"])
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
