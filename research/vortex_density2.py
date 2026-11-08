"""R1 (redo): exact density of Brent's Z2 vortices per spectre, via the closing construction of spin_densities.py.

Columns: convention-independent vortices (flux -1, no 180-degree step), all flux -1 vertices, spectres.  The flux
rule is star_data of vortex_density.py.  Rest of the method as in spin_densities.py (original docstring below).

    python research/vortex_density2.py K OUT.json

R11 step 3: exact vertex/link densities of the Spectre spin-network graph, tested against the gap-labelling module.

Graph as in su2_graph.py: vertices = corners of the spectre pieces (Gamma counts as two pieces), links = tile edges,
plaquettes = pieces.  With spin-1/2 on every link the volume density is the density of valence-4 vertices (valence-3
vertices have zero volume), and the strong-coupling area density is the plaquette density, i.e. 1 per spectre.

Exact method.  Every vertex of a tile is shared only by tiles that touch that tile, so its valence is fixed by the
tile's vertex-sharing neighbourhood.  We give each vertex weight 1/(number of pieces at it) and add up, for every
complete level-1 supertile, its weighted counts c = (valence-3, valence-4, all vertices, links, spectres).  The
level-1 collared types of gap_collared.py are used.  The script checks that c is constant on each type
("inconsistent" counts the failures).  The densities are then sum_t v_t c_t / sum_t v_t spectres_t, where v is the
exact Perron vector over Q(sqrt15) of the collared substitution.  For each density a + b sqrt15 we report whether it
lies in Z + Z(5+sqrt15)/10 (n = 10b and m = a - 5b both integers) and the least q > 0 with q*density in the module.

    python research/spin_densities.py K OUT.json
"""
import sys, json
from fractions import Fraction as Fr
from collections import Counter, defaultdict
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
import sympy as sp
from gap_collared import build, nullspace
from vortex_density import star_data


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = sys.argv[2] if len(sys.argv) > 2 else "vortex_density2.json"
    names, levels, V, PT = build(K)
    n_tiles = len(levels[0]["lab"]); npc = len(PT)
    _, vid = np.unique(V.reshape(-1, 4), axis=0, return_inverse=True)
    vid = vid.reshape(npc, 14); nv = int(vid.max()) + 1
    e = np.sort(np.stack([vid, np.roll(vid, -1, axis=1)], -1).reshape(-1, 2), axis=1)
    ue, einv, ecnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bd_piece = (ecnt[einv.reshape(npc, 14)] == 1).any(1)
    bd_tile = np.zeros(n_tiles, bool); np.logical_or.at(bd_tile, PT, bd_piece)
    deg = np.bincount(ue.ravel(), minlength=nv)
    npv = np.bincount(vid.ravel(), minlength=nv)  # pieces at each vertex
    w = 1.0 / npv
    # per piece weighted counts, exact as Fractions via integer numerators over 12 (npv in 1..6 at most)
    L = 60
    wn = (L // npv).astype(np.int64)
    assert np.all(wn * npv == L)
    _vid2, interior, flux, npi, val, members, bad_det = star_data(V, PT, npc)
    assert np.array_equal(_vid2, vid)
    vort = interior & (flux == -1) & (npi == 0); vneg = interior & (flux == -1)
    print("pieces", npc, "flux -1", int(vneg.sum()), "vortices", int(vort.sum()), "det mismatch", bad_det, flush=True)
    c3 = (wn[vid] * vort[vid]).sum(1); c4 = (wn[vid] * vneg[vid]).sum(1)
    cv = wn[vid].sum(1); cl = np.zeros(npc, np.int64)
    cs = np.full(npc, L)
    per_piece = np.stack([c3, c4, cv, cl, cs], 1)
    per_tile = np.zeros((n_tiles, 5), np.int64); np.add.at(per_tile, PT, per_piece)
    # level-1 signatures (as in gap_collared)
    vt = np.unique(np.stack([vid.ravel(), np.repeat(PT, 14)], 1), axis=0)
    bounds = np.nonzero(np.diff(vt[:, 0]))[0] + 1
    pairs = set()
    for grp in np.split(vt[:, 1], bounds):
        for x in grp:
            for y in grp:
                if x < y: pairs.add((int(x), int(y)))
    pairs = np.array(sorted(pairs))
    anc = {0: np.arange(n_tiles)}
    for j in range(1, K + 1):
        anc[j] = levels[j - 1]["parent"][anc[j - 1]]
    sig = {}
    for j in (1, 2):
        a1, a2 = anc[j][pairs[:, 0]], anc[j][pairs[:, 1]]
        s1, s2 = levels[j - 1]["slot"][anc[j - 1][pairs[:, 0]]], levels[j - 1]["slot"][anc[j - 1][pairs[:, 1]]]
        contact = defaultdict(set)
        for x, y, s, t in zip(a1.tolist(), a2.tolist(), s1.tolist(), s2.tolist()):
            if x != y:
                contact[(x, y)].add((s, t)); contact[(y, x)].add((t, s))
        nb = defaultdict(list); lab = levels[j]["lab"]
        for (x, y), st in contact.items():
            nb[x].append((int(lab[y]), tuple(sorted(st))))
        # a supertile is usable only if its whole vertex neighbourhood is in the patch: require its neighbours complete
        bad = np.zeros(len(lab), bool); np.logical_or.at(bad, anc[j], bd_tile)
        badn = bad.copy()
        for (x, y) in contact:
            if bad[y]: badn[x] = True
        sig[j] = {x: (int(lab[x]), tuple(sorted(nb[x]))) for x in range(len(lab)) if not badn[x]}
    st1 = np.zeros((len(levels[1]["lab"]), 5), np.int64); np.add.at(st1, anc[1], per_tile)
    counts = defaultdict(set)
    for x, s in sig[1].items():
        counts[s].add(tuple(st1[x].tolist()))
    inconsistent = sum(len(v) > 1 for v in counts.values())
    print("level-1 types", len(counts), "inconsistent", inconsistent, flush=True)
    # transitions level 2 -> level 1
    kids = defaultdict(Counter); par = levels[1]["parent"]
    for x, s in sig[1].items():
        kids[int(par[x])][s] += 1
    trans = {}
    for p, s in sig[2].items():
        if all(c in counts for c in kids[p]) and sum(kids[p].values()) > 0:
            trans.setdefault(s, kids[p])
    types = set(trans) & set(counts)
    while True:
        drop = {t for t in types if any(c not in types for c in trans[t])}
        if not drop: break
        types -= drop
    types = sorted(types, key=repr); idx = {t: i for i, t in enumerate(types)}; n = len(types)
    print("closed types", n, flush=True)
    res = dict(K=K, level1_types=len(counts), inconsistent=inconsistent, closed_types=n)
    C = [[0] * n for _ in range(n)]
    for t in types:
        for c, m in trans[t].items():
            C[idx[t]][idx[c]] += m
    rows = []
    for col in range(n):
        r1 = [Fr(C[i][col]) for i in range(n)] + [Fr(0)] * n; r1[col] -= 4; r1[n + col] -= 15
        r2 = [Fr(0)] * n + [Fr(C[i][col]) for i in range(n)]; r2[col] -= 1; r2[n + col] -= 4
        rows += [r1, r2]
    ns = nullspace(rows, 2 * n)
    res["nullspace_dim_Q"] = len(ns)
    r = sp.sqrt(15)
    v = [sp.Rational(ns[0][i].numerator, ns[0][i].denominator)
         + sp.Rational(ns[0][n + i].numerator, ns[0][n + i].denominator) * r for i in range(n)]
    cvec = [list(counts[t])[0] for t in types]
    tot = [sp.radsimp(sum(v[i] * cvec[i][k] for i in range(n))) for k in range(5)]
    names_d = ["vortex", "flux_minus_all", "vertices", "unused", "spectres"]
    dens = {}
    for k in (0, 1, 2):
        d = sp.expand(sp.radsimp(tot[k] / tot[4]))
        b = sp.Rational(d.coeff(r)); a = sp.Rational(sp.expand(d - b * r))
        nn, mm = 10 * b, a - 5 * b
        q = sp.ilcm(sp.fraction(nn)[1], sp.fraction(mm)[1])
        dens[names_d[k]] = dict(exact=str(d), float=float(d), in_module=bool(q == 1), least_multiplier=int(q))
        print(names_d[k], d, float(d), "in module" if q == 1 else "multiplier %d" % q, flush=True)
    res["densities_per_spectre"] = dens
    res["empirical_vortex_per_spectre"] = float(vort.sum()) / float(per_tile[:, 4].sum() / L)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
