"""R4: exact identification of the holonomy group of spectre_holonomy.py.

Every edge rotation angle a is a multiple of 30 degrees (checked), so a/2 is a multiple of 15 degrees.
With w = cos 15 + k sin 15 (order 48 in SU(2), w^24 = -1) and i the unit quaternion:
    proper edge:  qz(a)                      = w^(a/30)
    flip edge:    s (0, cos a/2, sin a/2, 0)  = s i w^(-a/30)      (i k = -j)
so every element is (f, m) := i^f w^m, f in {0,1}, m mod 48, with i^2 = w^24 and i w = w^-1 i.
The script computes, in this exact integer encoding, the generated group, its order, element orders,
centre, and checks the binary dihedral (dicyclic) presentation; it does this for twist modes +1, -1, alt
and checks the sign change is a central coboundary on the tree (eq' = c(parent) eq c(child)^-1).
Usage: python research/holonomy_group.py N OUT.json
"""
import sys, json
from collections import Counter
sys.path.insert(0, ".")
import numpy as np
import spectrespace as S
import spectre_holonomy as H

M = 48


def mul(a, b):
    f1, m1 = a; f2, m2 = b
    if f2 == 0:
        return (f1, (m1 + m2) % M)
    # i^f1 w^m1 i w^m2 = i^f1 i w^-m1 w^m2
    return ((f1 + 1) % 2, (-m1 + m2 + (24 if f1 else 0)) % M)


def inv(a):
    f, m = a
    return (0, -m % M) if f == 0 else (1, (m + 24) % M)  # (i w^m)^2 = -1


def order(a):
    p, k = a, 1
    while p != (0, 0):
        p = mul(p, a); k += 1
    return k


def closure(gens):
    G = {(0, 0)}; fr = [(0, 0)]
    while fr:
        new = []
        for e in fr:
            for g in gens:
                p = mul(e, g)
                if p not in G: G.add(p); new.append(p)
        fr = new
    return G


def to_q(a):
    f, m = a
    t = np.deg2rad(15 * m)
    w = np.array([np.cos(t), 0, 0, np.sin(t)])
    return H.qmul(np.array([0., 1, 0, 0]), w) if f else w


def edge(space, nd, mode):
    L = nd.local[:, :2]
    a = np.rad2deg(np.arctan2(L[1, 0], L[0, 0]))
    k = int(round(a / 30)); assert abs(a - 30 * k) < 1e-7, a
    if S.det2(L) > 0:
        e = (0, k % M)
    else:
        e = (1, -k % M)
        if H.twist_sign(space, nd, mode) < 0: e = mul((0, 24), e)
    assert np.abs(to_q(e) - H.edge_quaternion(space, nd, mode)).max() < 1e-9
    return e


def main():
    N = int(sys.argv[1]); out = sys.argv[2]
    space = S.SpectreSpace(N)
    res = dict(N=N, nodes=len(space.nodes))
    eqs = {}
    for mode in ("1", "-1", "alt"):
        eq = {nd.id: edge(space, nd, mode) for nd in space.nodes[1:]}
        eqs[mode] = eq
        hol = {0: (0, 0)}
        for nd in space.nodes[1:]: hol[nd.id] = mul(hol[nd.parent], eq[nd.id])
        G = closure(sorted(set(eq.values())))
        Hset = closure(sorted(set(hol.values())))
        ords = Counter(order(g) for g in G)
        centre = sorted(g for g in G if all(mul(g, h) == mul(h, g) for h in G))
        info = dict(order=len(G), holonomy_values_generate=len(Hset), element_orders=dict(sorted(ords.items())),
                    centre=centre, rotations=sum(1 for g in G if g[0] == 0),
                    edge_generators=sorted(set(eq.values())))
        rot = sorted({g for g in G if g[0] == 0}, key=lambda g: g[1])
        if len(G) == 2 * len(rot) and len(rot) > 0:
            n = len(rot) // 2
            a = min((g for g in rot if order(g) == 2 * n), default=None)
            b = next(g for g in G if g[0] == 1)
            if a is not None:
                ok = (closure([a]) == set(rot) and order(b) == 4 and mul(b, b) == _pow(a, n)
                      and mul(mul(inv(b), a), b) == inv(a))
                info["dicyclic"] = dict(n=n, name=f"Dic{n} (binary dihedral, order {4*n})", presentation_holds=ok)
        print(mode, json.dumps(info, default=str), flush=True)
        res[mode] = info
    # coboundary: c(child) = c(parent) * (eq'/eq), central
    for m2 in ("-1", "alt"):
        c = {0: (0, 0)}; ok = True
        for nd in space.nodes[1:]:
            ratio = mul(eqs[m2][nd.id], inv(eqs["1"][nd.id]))
            ok &= ratio in ((0, 0), (0, 24))
            c[nd.id] = mul(c[nd.parent], ratio)
            ok &= mul(mul(c[nd.parent], eqs["1"][nd.id]), inv(c[nd.id])) == eqs[m2][nd.id]
        res[f"coboundary_1_to_{m2}"] = bool(ok)
        print("sign change 1 ->", m2, "is a central (+-1) coboundary:", ok, flush=True)
    json.dump(res, open(out, "w"), indent=1, default=str)


def _pow(a, n):
    p = (0, 0)
    for _ in range(n): p = mul(p, a)
    return p


if __name__ == "__main__":
    main()
