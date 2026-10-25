"""Exact (Q(sqrt3)) re-implementation of the construction in brentharts/spectre spectre.py
(buildSpectreBase/buildSupertiles/buildSpectreTiles), written from its algorithm.
Affine maps are ((m00,m01,m10,m11),(t0,t1))."""
from q3 import *

def pts(a, b):
    a, b = q(a), q(b)
    a3, ad, b3, bd = a * R3 / 2, a / 2, b * R3 / 2, b / 2
    z = Q3(0)
    return [(z, z), (a, z), (a + ad, -a3), (a + ad + b3, -a3 + bd), (a + ad + b3, -a3 + b + bd),
            (a + a + ad + b3, -a3 + b + bd), (a + a + a + b3, b + bd), (a + a + a, b + b),
            (a + a + a - b3, b + b - bd), (a + a + ad - b3, a3 + b + b - bd), (a + ad - b3, a3 + b + b - bd),
            (ad - b3, a3 + b + b - bd), (z - b3, b + b - bd), (z, b)]

COS = {0: (1, 0), 30: (0, F(1, 2)), 60: (F(1, 2), 0), 90: (0, 0), 120: (F(-1, 2), 0), 150: (0, F(-1, 2)), 180: (-1, 0),
       210: (0, F(-1, 2)), 240: (F(-1, 2), 0), 270: (0, 0), 300: (F(1, 2), 0), 330: (0, F(1, 2))}
def cs(deg):
    deg %= 360
    c = Q3(*COS[deg]); s = Q3(*COS[(deg - 90) % 360])
    return c, s

def rot(deg):
    c, s = cs(deg)
    return ((c, -s, s, c), (Q3(0), Q3(0)))

IDT = rot(0)
MIRROR = ((Q3(-1), Q3(0), Q3(0), Q3(1)), (Q3(0), Q3(0)))

def apply(T, p):
    (m00, m01, m10, m11), (t0, t1) = T
    return (m00 * p[0] + m01 * p[1] + t0, m10 * p[0] + m11 * p[1] + t1)

def mul(A, B):
    (a00, a01, a10, a11), at = A
    (b00, b01, b10, b11), bt = B
    M = (a00 * b00 + a01 * b10, a00 * b01 + a01 * b11, a10 * b00 + a11 * b10, a10 * b01 + a11 * b11)
    return (M, apply(A, bt))

def trans(v): return ((Q3(1), Q3(0), Q3(0), Q3(1)), v)

def det(T):
    m = T[0]; return m[0] * m[3] - m[1] * m[2]

def inv(T):
    (m00, m01, m10, m11), t = T
    d = m00 * m11 - m01 * m10
    Mi = (m11 / d, -m01 / d, -m10 / d, m00 / d)
    Ti = (Mi, (Q3(0), Q3(0)))
    tt = apply(Ti, t)
    return (Mi, (-tt[0], -tt[1]))

class Tile:
    def __init__(self, label, quad): self.label = label; self.quad = quad
    def each(self, f, T=IDT): f(T, self.label)

class Meta:
    def __init__(self, tiles, trs, quad): self.tiles, self.trs, self.quad = tiles, trs, quad
    def each(self, f, T=IDT):
        for t, tr in zip(self.tiles, self.trs): t.each(f, mul(T, tr))

NAMES = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]
SUBST = (("Gamma", ("Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma")),
         ("Delta", ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma")),
         ("Theta", ("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma")),
         ("Lambda", ("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma")),
         ("Xi", ("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma")),
         ("Pi", ("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma")),
         ("Sigma", ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma")),
         ("Phi", ("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma")),
         ("Psi", ("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma")))

def base(a, b, rotation=30):
    P = pts(a, b)
    quad = [P[3], P[5], P[7], P[11]]
    tiles = {n: Tile(n, quad) for n in NAMES if n != "Gamma"}
    tiles["Gamma"] = Meta([Tile("Gamma1", quad), Tile("Gamma2", quad)], [IDT, mul(trans(P[8]), rot(rotation))], quad)
    return tiles

def supertiles(inp):
    quad = inp["Delta"].quad
    total = 0
    R = rot(0)
    trs = [R]
    tq = quad
    for ang, fr, to in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
        if ang != 0:
            total += ang; R = rot(total); tq = [apply(R, p) for p in quad]
        a = apply(trs[-1], quad[fr]); bq = tq[to]
        trs.append(mul(trans((a[0] - bq[0], a[1] - bq[1])), R))
    trs = [mul(MIRROR, t) for t in trs]
    sq = [apply(trs[6], quad[2]), apply(trs[5], quad[1]), apply(trs[3], quad[2]), apply(trs[0], quad[1])]
    out = {}
    for lab, subs in SUBST:
        out[lab] = Meta([inp[s] for s in subs if s], [t for s, t in zip(subs, trs) if s], sq)
    return out

def build(n, a, b, rotation=30):
    t = base(a, b, rotation)
    for _ in range(n): t = supertiles(t)
    return t

def placements(n, a=1, b=R3, seed="Delta"):
    out = []
    build(n, a, b)[seed].each(lambda T, lab: out.append((T, lab)))
    return out
