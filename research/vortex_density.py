"""R1: the exact density of Brent's Z2 vortices (spectre_holonomy.py, test d), and whether it lies in the
gap-labelling module Z + Z*alpha, alpha = (5 + sqrt15)/10 (item C).

Flux rule (as in spectre_holonomy.test_flux): at an interior vertex with pieces P_0..P_{k-1} in cyclic order,
d_i = relative rotation of P_{i+1} with respect to P_i, in degrees in (-180, 180]; w = sum d_i / 360 and the
flux is (-1)^w.  A vertex with some |d_i| = 180 is convention dependent; a flux -1 vertex with none is a
"convention independent vortex".  Pieces: every tile, and the two halves of the Mystic (Gamma), as in
gap_collared.build and in Brent's leaves.

Exact density.  A vertex star is determined by the level-1 collared type (gap_collared.py: label plus all
vertex-sharing neighbours, with slot contacts) of any supertile containing one of its pieces.  Give each vertex
1/k to each of its k pieces; f(t) = vortex mass of a level-1 supertile of type t.  The script checks that f is
well defined on complete supertiles, then
    density per spectre = sum_t v_t f(t) / sum_t v_t s(t),
with v the exact Perron vector of the collared substitution (eigenvalue 4 + sqrt15) and s(t) the number of
spectres in a level-1 supertile with the label of t.  Normalisation is per spectre, as in gap_freq/gap_collared.

Usage: python research/vortex_density.py K OUT.json
"""
import sys, json
from fractions import Fraction as Fr
from collections import Counter, defaultdict
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from gap_collared import build, nullspace
from handedness import ref_points

ZETA = np.exp(1j * np.pi / 6) ** np.arange(4)


def to_c(X):
    return X.astype(float) @ ZETA


def star_data(V, PT, npc):
    """per piece: det, angle theta (deg); per vertex: interior flag, cyclic piece order, flux, n_pi."""
    _, vid = np.unique(V.reshape(-1, 4), axis=0, return_inverse=True)
    vid = vid.reshape(npc, 14)
    Z = to_c(V.reshape(-1, 4)).reshape(npc, 14)
    R = to_c(np.array(ref_points(), np.int64))
    area = lambda z: 0.5 * np.imag(np.sum(np.conj(z) * np.roll(z, -1, axis=-1), axis=-1))
    det = np.sign(area(Z) / area(R)).astype(int)
    e0 = R[1] - R[0]
    pe = Z[:, 1] - Z[:, 0]
    theta = np.rad2deg(np.angle(pe)) - det * np.rad2deg(np.angle(e0))
    ccw = area(Z) > 0
    wedges = defaultdict(list)
    for p in range(npc):
        z = Z[p]
        for i in range(14):
            nx, pv = z[(i + 1) % 14] - z[i], z[i - 1] - z[i]
            an, ap = np.angle(nx), np.angle(pv)
            if ccw[p]:
                start, width = an, (ap - an) % (2 * np.pi)
            else:
                start, width = ap, (an - ap) % (2 * np.pi)
            wedges[int(vid[p, i])].append((p, start % (2 * np.pi), width))
    nv = int(vid.max() + 1)
    interior = np.zeros(nv, bool); flux = np.ones(nv, int); npi = np.zeros(nv, int); val = np.zeros(nv, int)
    members = {}
    bad_det = 0
    for v, ws in wedges.items():
        if abs(sum(w[2] for w in ws) - 2 * np.pi) > 1e-6:
            continue
        ws = sorted(ws, key=lambda w: w[1]); k = len(ws)
        interior[v] = True; val[v] = k; members[v] = [w[0] for w in ws]
        tot = 0
        for i in range(k):
            a, b = ws[i][0], ws[(i + 1) % k][0]
            if det[a] != det[b]:
                bad_det += 1
            d = (theta[b] - theta[a]) * det[a]
            d = ((d + 180) % 360) - 180
            d = int(round(d)); d = 180 if d == -180 else d
            tot += d; npi[v] += abs(d) == 180
        assert tot % 360 == 0, tot
        flux[v] = -1 if (tot // 360) % 2 else 1
    return vid, interior, flux, npi, val, members, bad_det


def main():
    K = int(sys.argv[1]); out = sys.argv[2]
    names, levels, V, PT = build(K)
    npc = len(PT); n_tiles = len(levels[0]["lab"])
    vid, interior, flux, npi, val, members, bad_det = star_data(V, PT, npc)
    vort = interior & (flux == -1) & (npi == 0)
    vneg = interior & (flux == -1)
    print("pieces", npc, "interior vertices", int(interior.sum()), "flux -1", int(vneg.sum()),
          "convention independent", int(vort.sum()), "det mismatches", bad_det, flush=True)
    stats = dict(valence=Counter(val[vort].tolist()), all_valence=Counter(val[interior].tolist()))
    # mass per piece
    mass_v = np.zeros(npc, object); mass_n = np.zeros(npc, object)
    mass_v[:] = Fr(0); mass_n[:] = Fr(0)
    okpiece = np.ones(npc, bool)
    for p in range(npc):
        for v in vid[p]:
            if not interior[v]:
                okpiece[p] = False; continue
            if vort[v]: mass_v[p] += Fr(1, int(val[v]))
            if vneg[v]: mass_n[p] += Fr(1, int(val[v]))
    # level-1 collared signatures (as gap_collared.main, j = 1 and up for transitions)
    vt = np.unique(np.stack([vid.ravel(), np.repeat(PT, 14)], 1), axis=0)
    pairs = set()
    for grp in np.split(vt[:, 1], np.nonzero(np.diff(vt[:, 0]))[0] + 1):
        g = sorted(set(grp.tolist()))
        for i, x in enumerate(g):
            for y in g[i + 1:]: pairs.add((x, y))
    pairs = np.array(sorted(pairs))
    oktile = np.ones(n_tiles, bool); np.logical_and.at(oktile, PT, okpiece)
    anc = {0: np.arange(n_tiles)}
    for j in range(1, K + 1):
        anc[j] = levels[j - 1]["parent"][anc[j - 1]]
    sig = {}
    for j in range(1, K):
        a1, a2 = anc[j][pairs[:, 0]], anc[j][pairs[:, 1]]
        c1 = levels[j - 1]["slot"][anc[j - 1][pairs[:, 0]]]; c2 = levels[j - 1]["slot"][anc[j - 1][pairs[:, 1]]]
        contact = defaultdict(set)
        for x, y, s, t in zip(a1.tolist(), a2.tolist(), c1.tolist(), c2.tolist()):
            if x != y:
                contact[(x, y)].add((s, t)); contact[(y, x)].add((t, s))
        nb = defaultdict(list); lab = levels[j]["lab"]
        for (x, y), st in contact.items():
            nb[x].append((int(lab[y]), tuple(sorted(st))))
        good = np.ones(len(lab), bool); np.logical_and.at(good, anc[j], oktile)
        # complete also needs every neighbour's tiles fully interior: require all tiles touching S ok
        tb = np.zeros(len(lab), bool)
        np.logical_or.at(tb, a1, ~oktile[pairs[:, 1]]); np.logical_or.at(tb, a2, ~oktile[pairs[:, 0]])
        sig[j] = {x: (int(lab[x]), tuple(sorted(nb[x]))) for x in range(len(lab)) if good[x] and not tb[x]}
    # f(t) on level 1
    piece_tile = PT
    mv1 = defaultdict(lambda: Fr(0)); mn1 = defaultdict(lambda: Fr(0))
    sup = anc[1][piece_tile]
    for p in range(npc):
        mv1[int(sup[p])] += mass_v[p]; mn1[int(sup[p])] += mass_n[p]
    fv, fn, incons = {}, {}, 0
    for x, t in sig[1].items():
        for f, m in ((fv, mv1), (fn, mn1)):
            if t in f and f[t] != m[x]: incons += 1
            f.setdefault(t, m[x])
    print("level-1 types seen", len(fv), "inconsistent", incons, flush=True)
    # collared substitution and Perron vector
    trans = {}; conflicts = 0
    for j in range(1, K - 1):
        kids = defaultdict(Counter); par = levels[j]["parent"]
        for x, s in sig[j].items(): kids[int(par[x])][s] += 1
        for p, s in sig[j + 1].items():
            if s in trans and trans[s] != kids[p]: conflicts += 1
            trans.setdefault(s, kids[p])
    types = set(trans)
    while True:
        drop = {t for t in types if any(c not in types for c in trans[t])}
        if not drop: break
        types -= drop
    types = sorted(types, key=repr); idx = {t: i for i, t in enumerate(types)}; n = len(types)
    missing = [t for t in types if t not in fv]
    print("closed types", n, "conflicts", conflicts, "types without f", len(missing), flush=True)
    res = dict(K=K, pieces=npc, interior_vertices=int(interior.sum()), flux_minus=int(vneg.sum()),
               vortices=int(vort.sum()), det_mismatch=bad_det, f_inconsistent=incons, closed_types=n,
               conflicts=conflicts, types_without_f=len(missing),
               vortex_valence=dict(stats["valence"]), vertex_valence=dict(stats["all_valence"]))
    if n == 0 or missing or incons:
        json.dump(res, open(out, "w"), indent=1, default=str); print("STOP: cannot evaluate exactly"); return
    C = [[0] * n for _ in range(n)]
    for t in types:
        for c, m in trans[t].items(): C[idx[t]][idx[c]] += m
    rows = []
    for col in range(n):
        r1 = [Fr(C[i][col]) for i in range(n)] + [Fr(0)] * n; r1[col] -= 4; r1[n + col] -= 15
        r2 = [Fr(0)] * n + [Fr(C[i][col]) for i in range(n)]; r2[col] -= 1; r2[n + col] -= 4
        rows += [r1, r2]
    ns = nullspace(rows, 2 * n)
    import sympy as sp
    import spectre_patch as spm
    r = sp.sqrt(15)
    x, y = ns[0][:n], ns[0][n:]
    v = [sp.Rational(x[i].numerator, x[i].denominator) + sp.Rational(y[i].numerator, y[i].denominator) * r for i in range(n)]
    spc = {i: sum((2 if c == "Gamma" else 1) for c in spm.SUBSTITUTIONS[a] if c) for i, a in enumerate(names)}
    norm = sum(v[idx[t]] * spc[t[0]] for t in types)
    out_d = {}
    for name, f in (("vortex", fv), ("flux_minus_all", fn)):
        num = sum(v[idx[t]] * sp.Rational(f[t].numerator, f[t].denominator) for t in types)
        d = sp.nsimplify(sp.radsimp(num / norm))
        d = sp.expand(d); b = d.coeff(r); a = sp.expand(d - b * r)
        nn = 10 * b; m = a - nn / 2
        out_d[name] = dict(density=str(d), value=float(d), a=str(a), b=str(b), m=str(m), n=str(nn),
                           in_module=bool(sp.Rational(m).q == 1 and sp.Rational(nn).q == 1))
        print(name, "density per spectre =", d, "=", float(d), " m + n*alpha with m, n =", m, nn,
              "in module:", out_d[name]["in_module"], flush=True)
    # empirical check in the patch
    res["empirical_per_piece"] = float(vort.sum() / npc)
    res["densities"] = out_d
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
