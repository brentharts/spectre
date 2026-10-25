"""Registration of the spectre.py patches at the Hat point for ALL levels n, by a finite-state induction:
every level-m placement map trs is in p6m(K) iff a condition on the classes of the level-(m-1) quad points mod Lambda
holds, and the next classes depend only on the current ones.  We iterate until the class-state repeats."""
from kites import *
c0 = (Q3(0), Q3(0, 3))

def level_trs(quad):
    total = 0; R = rot(0); trs = [R]; tq = quad
    for ang, fr, to in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
        if ang != 0:
            total += ang; R = rot(total); tq = [apply(R, p) for p in quad]
        a = apply(trs[-1], quad[fr]); bq = tq[to]
        trs.append(mul(trans((a[0] - bq[0], a[1] - bq[1])), R))
    trs = [mul(MIRROR, t) for t in trs]
    sq = [apply(trs[6], quad[2]), apply(trs[5], quad[1]), apply(trs[3], quad[2]), apply(trs[0], quad[1])]
    return trs, sq

def reduce(p):
    """canonical representative of p mod Lambda (keeps the class, keeps numbers small)"""
    al, be = lam_coords(psub(p, c0))
    fa, fb = al.a.__floor__(), be.a.__floor__()
    return psub(p, (U[0] * fa + V2[0] * fb, U[1] * fa + V2[1] * fb))

P = pts(1, R3)
quad = [P[3], P[5], P[7], P[11]]
seen = {}
m = 0
while True:
    state = tuple(lam_key(psub(x, c0)) for x in quad)
    if state in seen:
        print(f'class-state at level {m} repeats level {seen[state]}: induction closes; all levels checked ok')
        break
    seen[state] = m
    trs, sq = level_trs(quad)
    ok = all(in_p6m(t, c0) for t in trs)
    print(f'level {m}->{m+1}: quad classes {[(str(k[0]), str(k[2])) for k in state]} all 8 placement maps in p6m: {ok}')
    assert ok
    # replace quad by class representatives: the transition depends only on the classes (see docstring)
    quad = [reduce(x) for x in sq]
    m += 1
# sanity: the reduced recursion gives the same classes as the true recursion for 8 levels
qa = [P[3], P[5], P[7], P[11]]; qb = list(qa)
for k in range(8):
    _, qa = level_trs(qa); _, qb = level_trs(qb); qb = [reduce(x) for x in qb]
    assert [lam_key(psub(x, c0)) for x in qa] == [lam_key(psub(x, c0)) for x in qb]
print('reduced vs true recursion agree on classes for 8 levels')
