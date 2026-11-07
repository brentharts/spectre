"""R5: parity clock.  Count leaf tiles of SpectreSpace(N) with det +1 / -1 of the global placement, and
derive the counts in closed form from the 9x9 substitution matrix with signed entries
    M_s[p][c] = sum over slots of p with child label c of det(local placement)  (meta level),
plus the Gamma -> Gamma1, Gamma2 split.  #(+1) - #(-1) = e_root^T M_s^N g, #(+1) + #(-1) = e_root^T M^N g,
with g(label) = sum of dets / count of the height-0 pieces.  Exact eigen-data of M_s (sympy) is printed,
and the -1 eigen-direction is compared with the left eigenvector of M for eigenvalue -1 (Q_-).
Usage: python research/parity_clock.py NMAX OUT.json
"""
import sys, json
from collections import Counter
sys.path.insert(0, ".")
import numpy as np
import sympy as sp
import spectrespace as S

LABS = sorted(S.SUBST)


def main():
    NMAX = int(sys.argv[1]); out = sys.argv[2]
    res = dict(empirical={}, formula={})
    sp6 = S.SpectreSpace(1)
    Ms = sp.zeros(9); M = sp.zeros(9)
    for k in range(1, NMAX + 1):
        sk = S.SpectreSpace(k) if k > 1 else sp6
        sig = {}
        for nd in sk.nodes:
            if nd.kind == "meta" and nd.height == k:
                for c in nd.children:
                    ch = sk.nodes[c]; d = int(np.sign(S.det2(ch.local[:, :2])))
                    sig[(nd.label, ch.slot)] = (ch.label, d)
        res.setdefault("slot_dets_by_height", {})[k] = sorted({(h, s, d) for (h, s), (l, d) in sig.items()})
    # the meta placement dets do not depend on root label/height except via place[k]; read them per height
    pair = S.SpectreSpace(1, root="Gamma")
    pnode = next(nd for nd in pair.nodes if nd.kind == "pair")
    gdets = [int(np.sign(S.det2(pair.nodes[c].local[:, :2]))) for c in pnode.children]
    res["gamma_split_local_dets"] = gdets
    for N in range(1, NMAX + 1):
        for root in ("Delta",):
            spc = S.SpectreSpace(N, root=root)
            cnt = Counter(int(np.sign(S.det2(nd.T[:, :2]))) for nd in spc.leaves())
            byl = Counter((nd.label, int(np.sign(S.det2(nd.T[:, :2])))) for nd in spc.leaves())
            res["empirical"][N] = dict(plus=cnt[1], minus=cnt[-1], by_label=sorted(byl.items()))
            print("N", N, "det+1", cnt[1], "det-1", cnt[-1], flush=True)
    # signed matrices per height (dets of slot placements at height k)
    heights = res["slot_dets_by_height"]
    for k, rows in heights.items():
        dset = {d for (_, _, d) in rows}
        print("height", k, "slot dets", sorted(dset), flush=True)
    for i, a in enumerate(LABS):
        for c in S.SUBST[a]:
            if c: M[i, LABS.index(c)] += 1
    d1 = {(h, s): d for (h, s, d) in heights[1]}
    uniform = all(len({d for (_, _, d) in r}) == 1 for r in heights.values())
    res["all_meta_slots_same_det_per_height"] = uniform
    eps = {k: next(iter({d for (_, _, d) in r})) for k, r in heights.items()} if uniform else None
    g = sp.Matrix([sum(gdets) if l == "Gamma" else 1 for l in LABS])
    n = sp.Matrix([2 if l == "Gamma" else 1 for l in LABS])
    e = sp.zeros(1, 9); e[0, LABS.index("Delta")] = 1
    if uniform:
        for N in range(1, NMAX + 1):
            sgn = sp.prod([eps[k] for k in range(1, N + 1)])
            tot = (e * M ** N * n)[0]; diff = sgn * (e * M ** N * g)[0]
            res["formula"][N] = dict(plus=int((tot + diff) / 2), minus=int((tot - diff) / 2))
            print("formula N", N, res["formula"][N], "matches", res["formula"][N] == {k: res["empirical"][N][k] for k in ("plus", "minus")}, flush=True)
    ev = M.T.eigenvects()
    for lam, mult, vecs in ev:
        if lam == -1:
            q = vecs[0]
            res["Q_minus_left_eigvec"] = [str(x) for x in q]
            res["Q_minus_dot_g"] = str((q.T * g)[0]); res["Q_minus_dot_n"] = str((q.T * n)[0])
            print("Q- (left eigvec of M, eigenvalue -1):", list(q), " Q-.g =", (q.T * g)[0], " Q-.n =", (q.T * n)[0], flush=True)
    res["char_poly"] = str(sp.factor(M.charpoly().as_expr()))
    print("charpoly", res["char_poly"], flush=True)
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
