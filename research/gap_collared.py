"""Item C, step 2: does collaring enlarge the frequency module beyond (5 - sqrt15)^{-1} Z[sqrt15]?

We build a level-K Delta supertile exactly over Z[zeta_12] (positions scaled by 28) and track the hierarchy.
Tiles that share a vertex are neighbours.  A level-j supertile S (j >= 1) is neighbour of N when some tile of S
touches some tile of N.  The collar signature of S is
    (label S, sorted multiset over neighbours N of (label N, sorted set of touching child-slot pairs)),
a combinatorial descriptor that does not use coordinates.  The non-self-similar geometry changes from level to level,
so coordinates cannot be used; the slot structure (8 slots) is the same at every level.  S is "complete" when none of
its tiles has an unshared edge.  Its neighbours then all lie in the patch.
For complete parents T at level j+1 we record sig(T) -> multiset of sig(children).  The script checks that this
map is well defined, both within a level and across levels.  The children's collars are forced by the parent's
collar, so it should be.  It then takes the closed set of types and builds the collared substitution matrix C,
and finds the exact Perron vector over Q(sqrt15) (eigenvalue 4 + sqrt15) with Fraction elimination.  That vector
is normalised so that it projects onto the uncollared per-spectre frequencies u of gap_freq.py.  Finally it takes the
HNF of the Z[sqrt15]-span and compares it with the step-1 module.
Usage: python research/gap_collared.py K OUT.json
"""
import sys, json
from fractions import Fraction as Fr
from collections import Counter, defaultdict
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from handedness import mul, conj, sub, app, comp, zeta_pow, ref_points, ZERO, ONE

E = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

def mat(g):
    a, b, m = g
    A = np.array([mul(a, conj(e) if m else e) for e in E], dtype=np.int64).T
    return A, np.array(b, dtype=np.int64)

def level_transforms(K):
    pts = ref_points(); quad = [pts[3], pts[5], pts[7], pts[11]]
    mir = ([-1, 0, 0, 0], ZERO, True); Id = (ONE, ZERO, False)
    out = []
    for _ in range(K):
        bq = quad; total = 0; trs = [Id]
        for angle, src, dst in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            total += angle; r = zeta_pow(total // 30)
            tq = [mul(r, z) for z in bq]
            trs.append((r, sub(app(trs[-1], bq[src]), tq[dst]), False))
        trs = [comp(mir, tr) for tr in trs]
        quad = [app(trs[6], bq[2]), app(trs[5], bq[1]), app(trs[3], bq[2]), app(trs[0], bq[1])]
        out.append([mat(t) for t in trs])
    return out  # out[k-1]: slot maps used to build level k from level k-1

def build(K, top="Delta"):
    import spectre_patch as spm
    names = spm.TILE_NAMES
    T = level_transforms(K)
    lab = np.array([names.index(top)]); A = np.eye(4, dtype=np.int64)[None]; b = np.zeros((1, 4), np.int64)
    levels = {K: dict(lab=lab, parent=np.array([-1]), slot=np.array([-1]))}
    for k in range(K, 0, -1):
        L, P, S, AA, BB = [], [], [], [], []
        for i in range(len(lab)):
            for s, c in enumerate(spm.SUBSTITUTIONS[names[lab[i]]]):
                if c:
                    As, bs = T[k - 1][s]
                    L.append(names.index(c)); P.append(i); S.append(s)
                    AA.append(A[i] @ As); BB.append(A[i] @ bs + b[i])
        lab, A, b = np.array(L), np.array(AA), np.array(BB)
        levels[k - 1] = dict(lab=lab, parent=np.array(P), slot=np.array(S))
        print("level", k - 1, len(lab), flush=True)
    # spectre pieces: every tile one piece, Gamma two (the mystic)
    gi = names.index("Gamma"); pts = ref_points()
    mA, mb = mat((zeta_pow(1), pts[8], False))
    pa, pb, ptile = [A], [b], [np.arange(len(lab))]
    g = np.nonzero(lab == gi)[0]
    pa.append(A[g] @ mA); pb.append(np.einsum("nij,j->ni", A[g], mb) + b[g]); ptile.append(g)
    PA, PB, PT = np.concatenate(pa), np.concatenate(pb), np.concatenate(ptile)
    R = np.array(pts, np.int64)  # 14 x 4
    V = np.einsum("nij,vj->nvi", PA, R) + PB[:, None, :]
    return names, levels, V, PT

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = sys.argv[2] if len(sys.argv) > 2 else "gap_collared.json"
    names, levels, V, PT = build(K)
    n_tiles = len(levels[0]["lab"]); npc = len(PT)
    _, vid = np.unique(V.reshape(-1, 4), axis=0, return_inverse=True)
    vid = vid.reshape(npc, 14)
    # boundary pieces: an edge used once
    e = np.sort(np.stack([vid, np.roll(vid, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    _, einv, ecnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bd_piece = (ecnt[einv.reshape(npc, 14)] == 1).any(1)
    bd_tile = np.zeros(n_tiles, bool); np.logical_or.at(bd_tile, PT, bd_piece)
    # tile adjacency via shared vertices
    vt = np.stack([vid.ravel(), np.repeat(PT, 14)], 1)
    vt = np.unique(vt, axis=0)
    pairs = set()
    start = 0
    vs = vt[:, 0]
    bounds = np.nonzero(np.diff(vs))[0] + 1
    for grp in np.split(vt[:, 1], bounds):
        for x in grp:
            for y in grp:
                if x < y: pairs.add((int(x), int(y)))
    pairs = np.array(sorted(pairs))
    print("tiles", n_tiles, "pieces", npc, "touching pairs", len(pairs), "boundary tiles", int(bd_tile.sum()), flush=True)
    # ancestors
    anc = {0: np.arange(n_tiles)}
    for j in range(1, K + 1):
        anc[j] = levels[j - 1]["parent"][anc[j - 1]]
    sig = {}
    for j in range(1, K):
        a1, a2 = anc[j][pairs[:, 0]], anc[j][pairs[:, 1]]
        c1, c2 = levels[j - 1]["slot"][anc[j - 1][pairs[:, 0]]], levels[j - 1]["slot"][anc[j - 1][pairs[:, 1]]]
        contact = defaultdict(set)
        for x, y, s, t in zip(a1.tolist(), a2.tolist(), c1.tolist(), c2.tolist()):
            if x != y:
                contact[(x, y)].add((s, t)); contact[(y, x)].add((t, s))
        nb = defaultdict(list)
        lab = levels[j]["lab"]
        for (x, y), st in contact.items():
            nb[x].append((int(lab[y]), tuple(sorted(st))))
        bad = np.zeros(len(lab), bool); np.logical_or.at(bad, anc[j], bd_tile)
        sig[j] = {x: (int(lab[x]), tuple(sorted(nb[x]))) for x in range(len(lab)) if not bad[x]}
        print("level", j, "supertiles", len(lab), "complete", len(sig[j]), "types", len(set(sig[j].values())), flush=True)
    # transitions sig(parent at j+1) -> Counter(sig(children at j))
    trans, conflicts, per_level = {}, 0, {}
    for j in range(1, K - 1):
        kids = defaultdict(Counter)
        par = levels[j]["parent"]
        for x, s in sig[j].items():
            kids[int(par[x])][s] += 1
        seen = 0
        for p, s in sig[j + 1].items():
            c = kids[p]
            key = s
            if key in trans and trans[key] != c:
                conflicts += 1
            trans.setdefault(key, c); seen += 1
        per_level[j + 1] = seen
    types = set(trans)
    # closed set: drop types with children outside the known set, iterate
    while True:
        drop = {t for t in types if any(c not in types for c in trans[t])}
        if not drop: break
        types -= drop
    types = sorted(types, key=repr); idx = {t: i for i, t in enumerate(types)}; n = len(types)
    print("transition conflicts", conflicts, "closed types", n, flush=True)
    res = dict(K=K, conflicts=conflicts, n_closed_types=n, parents_per_level=per_level,
               types_per_level={j: len(set(sig[j].values())) for j in sig})
    if n == 0:
        json.dump(res, open(out, "w"), indent=1); return
    C = [[0] * n for _ in range(n)]
    for t in types:
        for c, m in trans[t].items():
            C[idx[t]][idx[c]] += m
    # row vector v = x + y r with v C = (4 + r) v : x C = 4x + 15y, y C = x + 4y.  Unknowns (x, y) in Q^{2n}.
    rows = []
    for col in range(n):
        r1 = [Fr(C[i][col]) for i in range(n)] + [Fr(0)] * n; r1[col] -= 4; r1[n + col] -= 15
        r2 = [Fr(0)] * n + [Fr(C[i][col]) for i in range(n)]; r2[col] -= 1; r2[n + col] -= 4
        rows += [r1, r2]
    ns = nullspace(rows, 2 * n)
    print("nullspace dim over Q", len(ns), flush=True)
    res["nullspace_dim_Q"] = len(ns)
    x, y = ns[0][:n], ns[0][n:]
    v = [(x[i], y[i]) for i in range(n)]
    # project onto labels and match the step-1 vector u (per spectre)
    import sympy as sp
    r = sp.sqrt(15)
    proj = [(Fr(0), Fr(0))] * 9
    for t in types:
        a, bb = v[idx[t]]; L = t[0]
        proj[L] = (proj[L][0] + a, proj[L][1] + bb)
    # normaliser so that sum_L proj_L * s_L = 1 (s = spectres per label at level j; we compute in units of level-j
    # supertiles, so divide by sum proj_L * S_L with S = spectres per level-j supertile; the ratio between levels is
    # a unit, so the module is unchanged; use j = 0 counts and then correct by the exact uncollared vector below)
    vals = [sp.Rational(a.numerator, a.denominator) + sp.Rational(bb.numerator, bb.denominator) * r for a, bb in v]
    pv = [sp.Rational(a.numerator, a.denominator) + sp.Rational(bb.numerator, bb.denominator) * r for a, bb in proj]
    import spectre_patch as spm
    Mu = sp.zeros(9, 9)
    for i, a in enumerate(names):
        for c in spm.SUBSTITUTIONS[a]:
            if c: Mu[i, names.index(c)] += 1
    u = (Mu.T - (4 + r) * sp.eye(9)).nullspace()[0]
    s = sp.Matrix([2 if a == "Gamma" else 1 for a in names])
    u = u / (u.T * s)[0]
    # pv must be proportional to u: find scalar c with pv = c u
    k0 = next(i for i in range(9) if pv[i] != 0)
    c = sp.radsimp(pv[k0] / u[k0])
    ok = all(sp.simplify(sp.radsimp(pv[i] - c * u[i])) == 0 for i in range(9))
    print("projection proportional to uncollared u:", ok, flush=True)
    res["projection_matches_u"] = bool(ok)
    w = [sp.expand(sp.radsimp(z / c)) for z in vals]
    coords = []
    for z in w:
        bb = z.coeff(r); a = sp.expand(z - bb * r); coords.append((sp.Rational(a), sp.Rational(bb)))
    for i in range(9):  # include uncollared generators
        z = sp.expand(sp.radsimp(u[i])); bb = z.coeff(r); coords.append((sp.Rational(sp.expand(z - bb * r)), sp.Rational(bb)))
    gens = []
    for a, bb in coords:
        gens += [(a, bb), (15 * bb, a)]
    N = sp.ilcm(*[sp.fraction(q)[1] for g in gens for q in g])
    G = sp.Matrix([[int(q * N) for q in g] for g in gens])
    from sympy.matrices.normalforms import hermite_normal_form
    H = hermite_normal_form(G.T)
    basis = [[sp.Rational(H[i, jj], N) for i in range(2)] for jj in range(H.shape[1])]
    det = abs(sp.Matrix([[q[0], q[1]] for q in basis]).det())
    step1 = sp.Rational(1, 10)  # covolume of Z + Z(5+sqrt15)/10 in (a, b) coords
    res.update(N=int(N), basis=[[str(q) for q in bb_] for bb_ in basis], covolume=str(det),
               step1_covolume=str(step1), enlarged=bool(det != step1),
               index_over_step1=str(step1 / det) if det else None,
               freqs_sample=[str(z) for z in w[:20]])
    print("module basis", res["basis"], "covolume", det, "step1", step1, "index", res["index_over_step1"], flush=True)
    json.dump(res, open(out, "w"), indent=1)

def nullspace(rows, ncol):
    rows = [r[:] for r in rows]; piv = []; rk = 0
    for col in range(ncol):
        p = next((i for i in range(rk, len(rows)) if rows[i][col] != 0), None)
        if p is None: continue
        rows[rk], rows[p] = rows[p], rows[rk]
        pr = rows[rk]; inv = 1 / pr[col]; pr = [q * inv for q in pr]; rows[rk] = pr
        for i in range(len(rows)):
            if i != rk and rows[i][col] != 0:
                f = rows[i][col]; ri = rows[i]
                rows[i] = [a - f * bq for a, bq in zip(ri, pr)]
        piv.append(col); rk += 1
    free = [c for c in range(ncol) if c not in piv]
    out = []
    for f in free:
        v = [Fr(0)] * ncol; v[f] = Fr(1)
        for i, pc in enumerate(piv): v[pc] = -rows[i][f]
        out.append(v)
    return out

if __name__ == "__main__":
    main()
