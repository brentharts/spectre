"""C3 reproduction (exact). python3 run_c3.py NMAX"""
import sys, time
from kites import *
c0 = (Q3(0), Q3(0, 3))
P = pts(1, R3); Pm = pts(R3, 1)
E = kite_edges_near(c0, -4, 4)
def registered(poly, T):
    Q = [apply(T, p) for p in poly]
    return all(frozenset((Q[i], Q[(i + 1) % len(Q)])) in E for i in range(len(Q)))
G2base = mul(trans(P[8]), rot(30))
print('(a) base quad points P3,P5,P7,P11 are hex centres:', [in_lambda(psub(P[i], c0)) for i in (3, 5, 7, 11)])
print('(b) MIRROR x->-x in p6m(K):', in_p6m(MIRROR, c0))
print('(c) base Tile(1,r3) registered:', registered(P, IDT), '; Gamma2 = Tile(r3,1) at trans(P8)rot30 registered:', registered(Pm, G2base))
print('    areas (in units of kite area r3): Tile(1,r3) =', end=' ')
def area(poly):
    s = Q3(0)
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
        s = s + x1 * y2 - x2 * y1
    return s / 2
print(area(P) / R3, ' Tile(r3,1) =', area(Pm) / R3)
# 30-degree rotation is not in p6m: Gamma2 relative to Gamma1 is NOT a p6m motion
print('    Gamma2 placement in p6m?', in_p6m(G2base, c0), '(its polygon is Tile(r3,1), registered via a 30deg-rotated frame)')
nmax = int(sys.argv[1])
for n in range(1, nmax + 1):
    t = time.time()
    pl = placements(n)
    dets = {det(T) for T, _ in pl}
    reg = all(in_p6m(T, c0) for T, lab in pl if lab != 'Gamma2') and all(in_p6m(mul(T, inv(G2base)), c0) for T, lab in pl if lab == 'Gamma2')
    mys = [T for T, lab in pl if lab == 'Gamma1']
    m1 = mys[0]
    classes = {}
    for T in mys:
        classes.setdefault((T[0], lam_key(T[1])), []).append(T)
    rel = {}
    for k, Ts in classes.items():
        R = mul(Ts[0], inv(m1))[0]
        rel.setdefault(rot_angle(R), []).append(k)
    bij = all(len(v) == 1 for v in rel.values())
    print(f'level {n}: tiles {len(pl)} mystics {len(mys)} dets {sorted(float(d) for d in dets)} all placements in p6m(K): {reg} '
          f'classes mod Lambda {len(classes)} relative rotations {sorted(rel)} class<->rotation bijective {bij}  ({time.time()-t:.0f}s)')
