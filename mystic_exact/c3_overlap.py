"""Check the level-n Hat-point patch is a packing: map the kites of each base tile by its (p6m) placement; no kite twice."""
import sys
from kites import *
c0 = (Q3(0), Q3(0, 3))
P = pts(1, R3); Pm = pts(R3, 1)
G2base = mul(trans(P[8]), rot(30))
def kites_near(lo, hi):
    out = []
    for a in range(lo, hi + 1):
        for b in range(lo, hi + 1):
            C = (c0[0] + U[0] * a + V2[0] * b, c0[1] + U[1] * a + V2[1] * b)
            for k in range(6):
                pts4 = [C, padd(C, polar(R3, 30 + 60 * k)), padd(C, polar(2, 60 + 60 * k)), padd(C, polar(R3, 90 + 60 * k))]
                out.append(((pts4[0][0] + pts4[1][0] + pts4[2][0] + pts4[3][0]) / 4, (pts4[0][1] + pts4[1][1] + pts4[2][1] + pts4[3][1]) / 4))
    return out
def inside(poly, p):
    x, y = float(p[0]), float(p[1]); c = False
    fp = [(float(a), float(b)) for a, b in poly]
    for i in range(len(fp)):
        (x1, y1), (x2, y2) = fp[i], fp[(i + 1) % len(fp)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1): c = not c
    return c
K = kites_near(-4, 4)
hat = [k for k in K if inside(P, k)]
tur = [k for k in K if inside([apply(G2base, p) for p in Pm], k)]
print('kites in base Tile(1,r3):', len(hat), ' in base Gamma2 Tile(r3,1):', len(tur))
for n in range(1, int(sys.argv[1]) + 1):
    seen = set(); dup = 0
    for T, lab in placements(n):
        if lab == 'Gamma2':
            g = mul(T, inv(G2base)); base = tur
        else:
            g = T; base = hat
        for k in base:
            p = apply(g, k)
            if p in seen: dup += 1
            seen.add(p)
    print(f'level {n}: kites covered {len(seen)}, kites covered twice {dup}')
