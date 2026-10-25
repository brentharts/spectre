"""[3.4.6.4] kite lattice K (hex centres c0 + Lambda, Lambda = Z(3,r3)+Z(0,2r3)) and p6m membership, exact."""
from spectre_exact import *

U = (Q3(3), R3); V2 = (Q3(0), Q3(0, 2))

def lam_coords(w):
    al = w[0] / 3
    be = (w[1] - al * R3) / Q3(0, 2)
    return al, be

def in_lambda(w):
    al, be = lam_coords(w)
    return al.b == 0 and be.b == 0 and al.a.denominator == 1 and be.a.denominator == 1

def lam_key(w):
    al, be = lam_coords(w)
    return (al.a % 1, al.b, be.a % 1, be.b)

def polar(r, deg):
    c, s = cs(deg); r = q(r); return (r * c, r * s)

def padd(p, w): return (p[0] + w[0], p[1] + w[1])
def psub(p, w): return (p[0] - w[0], p[1] - w[1])

def kite_edges_near(c0, lo, hi):
    """all K-edges (frozenset of 2 points) with hex centres c0+aU+bV2 for a,b in [lo,hi]"""
    E = set()
    for a in range(lo, hi + 1):
        for b in range(lo, hi + 1):
            C = (c0[0] + U[0] * a + V2[0] * b, c0[1] + U[1] * a + V2[1] * b)
            for k in range(6):
                M1 = padd(C, polar(R3, 30 + 60 * k)); Vt = padd(C, polar(2, 60 + 60 * k)); M2 = padd(C, polar(R3, 90 + 60 * k))
                E |= {frozenset((C, M1)), frozenset((M1, Vt)), frozenset((Vt, M2)), frozenset((M2, C))}
    return E

def is_D6(M):
    m00, m01, m10, m11 = M
    d = m00 * m11 - m01 * m10
    for k in range(0, 360, 60):
        c, s = cs(k)
        if d == 1 and (m00, m01, m10, m11) == (c, -s, s, c): return True
        if d == -1 and (m00, m01, m10, m11) == (c, s, s, -c): return True
    return False

def in_p6m(T, c0):
    return is_D6(T[0]) and in_lambda(psub(apply(T, c0), c0))

def rot_angle(M):
    """angle (deg, multiple of 30) of a rotation matrix"""
    for k in range(0, 360, 30):
        c, s = cs(k)
        if tuple(M) == (c, -s, s, c): return k
    return None
