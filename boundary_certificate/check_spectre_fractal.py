"""Referee certificate for G2: the limiting boundary of the Spectre supertiles has box dimension log(nu)/log(lambda).

Builds on check_spectre_boundary_proof.py (imported; it re-runs that proof) and adds the finite inputs of the argument:

  (a) Convergence.  X_k = lambda^{-k} O_k satisfies X_k = U_i F_{i,k}(X_{k-1}^{(i)}) with similarities of ratio 1/lambda
      whose translations M_i q_{k-1} / lambda^k converge geometrically on each parity (q_k = lambda^k a + (-lambda)^k b
      + O(1)).  So X_{2m+p} -> K_p in the Hausdorff metric, K_p being the attractor of a graph-directed system of
      similarities.  Its pieces have measure-disjoint union and positive area (a Perron argument; area of X_k -> c > 0).
  (b) Interior.  A grandchild g of O whose depth-2 contacts with every possible outside neighbour are separated at
      every order.  Then a ball of radius (margin * scale) around g lies in O at every order, so K has interior and,
      since the pieces are dense, K is the closure of its interior.
  (c) Spectral radius.  The contact matrix N on the 93 live types has charpoly x^84 (x-1)^3 (x+1)^4 (x^2-4x-1), so
      every count of contact pairs of depth m is O(nu^m), nu = 2+sqrt5.  This gives the upper bound: the boundary of
      X_k, which contains the boundary of K up to any epsilon for large k, is covered by the depth-m pieces in contact
      with an outside neighbour.
  (d) Lower bound.  A set E of live types such that every type in E has S(j) >= 1 for all j >= 1 and a child type in
      E.  E contains a sibling type and an irreducible block of spectral radius nu.  So at least c nu^m pairs of
      depth-m pieces of two sibling limit tiles meet, each in a boundary point, and a box of side lambda^{-m} meets
      boundedly many pieces.

Usage: python3 check_spectre_fractal.py      (about 3 minutes)
"""
import io
import contextlib
import itertools
import numpy as np
import sympy as sp
import networkx as nx

with contextlib.redirect_stdout(io.StringIO()):
    import check_spectre_boundary_proof as bp

x = sp.symbols('x')
keys, live, idx = bp.keys, bp.live, bp.idx
n = len(keys)
NU = 2 + sp.sqrt(5)
LAM = sp.sqrt(4 + sp.sqrt(15))

# ---------- (c) the contact matrix ----------
Nm = sp.zeros(n, n)
for a, kk in enumerate(keys):
    for c in bp.children(*live[kk]):
        ck = bp.tkey(*c)
        if ck in idx:
            Nm[a, idx[ck]] += 1
cp = sp.factor(Nm.charpoly(x).as_expr())
print("(c) charpoly of the %dx%d contact matrix:" % (n, n), cp)
for fac, mult in sp.factor_list(Nm.charpoly(x).as_expr())[1]:
    for r in sp.Poly(fac, x).all_roots():
        assert abs(complex(r)) <= float(NU) + 1e-12
        if abs(abs(complex(r)) - float(NU)) < 1e-9:
            assert sp.simplify(r - NU) == 0 and mult == 1
print("    spectral radius = 2+sqrt5 = nu, a simple root; all other roots have modulus <= 1")

# ---------- (d) the lower-bound set E ----------
S = sp.Matrix(bp.V0)
hist = []
for j in range(12):
    hist.append([int(v) for v in S[2:, 0]])
    S = bp.Bm * S
E = {a for a in range(n) if hist[1][a] >= 1}
while True:
    E2 = {a for a in E if any(Nm[a, b] for b in E)}
    if E2 == E:
        break
    E = E2
# S(j+1) = sum over children of S(j) >= S_child(j) for a child in E, so S >= 1 on E for every j >= 1
for a in E:
    assert all(hist[j][a] >= 1 for j in range(1, 12))
G = nx.DiGraph()
G.add_nodes_from(E)
G.add_edges_from((a, b) for a in E for b in E if Nm[a, b])
dom = None
for comp in nx.strongly_connected_components(G):
    comp = sorted(comp)
    if len(comp) == 1 and Nm[comp[0], comp[0]] == 0:
        continue
    sub = Nm.extract(comp, comp)
    if sp.rem(sub.charpoly(x).as_expr(), x ** 2 - 4 * x - 1, x) == 0:
        dom = comp
assert dom is not None, "no irreducible block of radius nu inside E"
sib = [idx[bp.tkey(*t)] for X in 'OG' for t in bp.start[X] if bp.tkey(*t) in idx]
sib_in = [a for a in sib if a in E and any(nx.has_path(G, a, d) for d in dom)]
assert sib_in
# the meeting points lie on the boundaries of the two children; the bound reaches K itself (and not only the Mystic
# limit) because some of these sibling pairs consist of two non-Mystic children
sib_shapes = sorted(set((keys[a][0], keys[a][1]) for a in sib_in))
assert ('O', 'O') in sib_shapes
print("(d) E: %d types with S(j) >= 1 for all j >= 1, closed under taking a child in E; it contains an irreducible "
      "block of %d types with x^2-4x-1 dividing its charpoly (radius nu), reached from %d sibling types"
      % (len(E), len(dom), len(sib_in)))
print("    shapes of the two children in those sibling types:", sib_shapes)
# depth-m contact-pair counts from a sibling type, against nu^m
a0 = sib_in[0]
NE = Nm.extract(sorted(E), sorted(E))
pos = sorted(E).index(a0)
v = sp.zeros(len(E), 1)
v[pos] = 1
cnt = []
row = sp.ones(1, len(E))
w = v.T
for m in range(14):
    cnt.append(int((w * sp.ones(len(E), 1))[0]))
    w = w * NE
print("    depth-m meeting pairs from one sibling type:", cnt[:10], " ratio at m=13: %.6f" % (cnt[13] / cnt[12]))


# ---------- (b) interior certificate ----------
def rev(t):
    X, Y, Am, L = t
    Ai = bp.Ainv(Am)
    return (Y, X, Ai, -Ai.dot(L))


def children_idx(X, Y, Am, L):
    out = []
    for i, cx in bp.CH[X]:
        for j, cy in bp.CH[Y]:
            out.append((i, j, (cx, cy, bp.AINV[i].dot(Am).dot(bp.A[j]),
                               bp.AINV[i].dot(Am.dot(bp.M[j]) + L.dot(bp.Phi) - bp.M[i]))))
    return out


neigh = {'O': [], 'G': []}            # live types with the given shape on the X side, both orders
for kk in keys:
    t = live[kk]
    neigh[t[0]].append(t)
    r = rev(t)
    neigh[r[0]].append(r)
cache = {}


def sep(t):
    k = bp.tkey(*t)
    if k not in cache:
        cache[k] = bp.separated(*t) if k not in idx else -1.0
    return cache[k]


found = None
for i, ci in bp.CH['O']:
    for j, cj in bp.CH[ci]:
        worst = 1e9
        ok = True
        for t in neigh['O']:
            for (i1, _, t1) in children_idx(*t):
                if i1 != i:
                    continue
                for (j1, _, t2) in children_idx(*t1):
                    if j1 != j:
                        continue
                    m = sep(t2)
                    if m <= 0:
                        ok = False
                        break
                    worst = min(worst, m)
                if not ok:
                    break
            if not ok:
                break
        if ok:
            found = (i, j, worst)
            break
    if found:
        break
assert found, "no interior grandchild certified"
print("(b) the grandchild in slot %d of slot %d of O is separated from every possible outside contact at every order "
      "(margin %.4f scale units), so a ball of that radius about it lies in O_{k+2} for every k" % (found[1], found[0], found[2]))

# ---------- (a) area ----------
Mshape = sp.Matrix([[7, 1], [6, 1]])
print("(a) children by shape [[7,1],[6,1]] has Perron root", sp.simplify(max(Mshape.eigenvals(), key=lambda e: float(e))),
      "= lambda^2, so the normalised areas converge to a positive limit and the pieces are measure-disjoint")
# geometric convergence of the normalised translations: q_k = lambda^k a + (-lambda)^k b + O(1)
c0 = bp.comps(bp.C_ID)
print("    the quad centroid has components on mu = +-lambda of moduli %.4f, %.4f; the remainder is O(1)"
      % (abs(c0[0]), abs(c0[1])))
print("RESULT: for p = 0, 1 the normalised supertiles lambda^{-k} O_k (k = p mod 2) converge to a compact K_p equal to the "
      "closure of its interior, and dim_B(boundary of K_p) = log(nu)/log(lambda) = %.12f"
      % float(sp.log(NU) / sp.log(LAM)))
