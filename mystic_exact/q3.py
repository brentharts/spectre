"""Exact arithmetic in Q(sqrt3): x = a + b*sqrt3, a,b Fractions."""
from fractions import Fraction as F
import math

class Q3:
    __slots__ = ('a', 'b')
    def __init__(self, a=0, b=0):
        self.a = F(a); self.b = F(b)
    def __add__(s, o): o = q(o); return Q3(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __sub__(s, o): o = q(o); return Q3(s.a - o.a, s.b - o.b)
    def __rsub__(s, o): return q(o) - s
    def __neg__(s): return Q3(-s.a, -s.b)
    def __mul__(s, o): o = q(o); return Q3(s.a * o.a + 3 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__
    def conj(s): return Q3(s.a, -s.b)
    def __truediv__(s, o):
        o = q(o); n = o.a * o.a - 3 * o.b * o.b
        r = s * o.conj(); return Q3(r.a / n, r.b / n)
    def sign(s):
        a, b = s.a, s.b
        if b == 0: return (a > 0) - (a < 0)
        if a == 0: return (b > 0) - (b < 0)
        if (a > 0) == (b > 0): return 1 if a > 0 else -1
        # opposite signs: compare a^2 vs 3b^2
        d = a * a - 3 * b * b
        return (1 if a > 0 else -1) if d > 0 else (1 if b > 0 else -1)
    def __eq__(s, o): o = q(o); return s.a == o.a and s.b == o.b
    def __hash__(s): return hash((s.a, s.b))
    def __lt__(s, o): return (s - o).sign() < 0
    def __float__(s): return float(s.a) + float(s.b) * math.sqrt(3)
    def __repr__(s): return f"({s.a}+{s.b}r3)"

def q(x): return x if isinstance(x, Q3) else Q3(x, 0)
R3 = Q3(0, 1)
HALF = Q3(F(1, 2))
