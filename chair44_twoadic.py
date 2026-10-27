#!/usr/bin/env python3
r"""chair44_twoadic.py - the 2-adic structure of the Chair44 substitution, computed exactly.

The Pisot route to pure-point diffraction is unavailable for Chair44 (integer inflation 2,
rational frequencies).  The replacement is not a suggestion: Tsiokos's repository already
records it (THEOREM.md, `substitution_modular_coincidence`, with the Lee-Moody / Schlottmann
import as a cited T3 step): the substitution on 24 x 7 = 168 labelled unit cells is primitive
and has a MODULAR COINCIDENCE, so the fixed point's label classes are regular model sets with a
2-adic internal space, hence pure point and limit-periodic.  This module recomputes the finite
part independently from `symchair44` and adds a RATE.

What is computed (exact integers, no floats)
  1. the cell substitution  sigma : 168 labels -> 8 labels (2x2x2 block), built from the child poses
  2. primitivity: sigma's 168x168 matrix has column sum 8 and every entry positive at power 3
  3. modular coincidence: the least M with a position a in [0,2^M)^3 where sigma^M(i)[a] is the
     same label for EVERY i is M = 3, with a = (0,0,2) among the coincident addresses (336 of 512)
  4. the RATE.  Track the set of labels still possible after reading the top digits of a position
     (a subset automaton; only 209 subsets are reachable, 41 of them non-singleton).  The number
     of digit words of length M that have NOT yet coincided is
            c(M) = 1, 8, 64, 176, 400, 848, 1744, ...      c(M+1) = 2 c(M) + 48  (M >= 1)
     i.e. c(M) = 28 * 2^M - 48 for M >= 1.  The non-coincident fraction c(M)/8^M ~ 28 * 4^(-M) -> 0, which is
     the coincidence condition quantitatively, and the transfer matrix on the 41 non-singleton
     states has spectral radius exactly 2.  In the 2-adic internal space Z_2^3 (box dimension 3)
     the set where the label is not yet determined has box dimension log2(2) = 1: the window
     boundaries are measure-zero sets of dimension 1.

What this is and is not.  It is a finite exact computation, checked against the upstream T1n/T2
statements (depth 3, address (0,0,2)).  Pure-point diffraction and limit-periodicity of the
fixed point follow from it only through the cited theorem (T3); nothing here proves that every
tiling lies in the substitution hull, which upstream left open.  Romanov proves it (the paper's
hull section); `chair44_hull.py` recomputes the part of his finite computation that depends only
on the substitution.
That the tiling has no period is the upstream theorem; clopen windows would give a period, and
the dimension-1 undetermined set computed here is consistent with windows that are not clopen.

    python3 chair44_twoadic.py
"""
import itertools
from collections import Counter

import sympy as sp

import symchair44 as S

BITS = list(itertools.product((0, 1), repeat=3))


def labels_and_sigma():
    """Labels (frame, native cell index) for the 24 rotations x 7 cells, and sigma[label][bit]."""
    frames = [f for f in S.all_frames() if S.frame_det(f) == 1]
    labels = [(f, k) for f in frames for k in range(7)]
    index = {l: i for i, l in enumerate(labels)}

    def cell_of(frame, t, c):
        x = S.frame_apply(frame, c)
        return tuple(x[i] + t[i] - (1 if frame[1][i] < 0 else 0) for i in range(3))

    sigma = [[None] * 8 for _ in labels]
    for f in frames:
        owner = {}
        for G, t in S.refine_pose((f, (0, 0, 0)), S.CHILDREN):
            for k, c in enumerate(S.CELLS):
                owner[cell_of(G, t, c)] = (G, k)
        for kk, c in enumerate(S.CELLS):
            p = cell_of(f, (0, 0, 0), c)
            for b, a in enumerate(BITS):
                G, k = owner[tuple(2 * p[i] + a[i] for i in range(3))]
                sigma[index[(f, kk)]][b] = index[(G, k)]
    return labels, sigma


def primitive_exponent(sigma, limit=8):
    n = len(sigma)
    reach = [{j} for j in range(n)]                       # labels reachable from i in one step
    step = [set(row) for row in sigma]
    cur = [set(s) for s in step]
    for power in range(1, limit + 1):
        if all(len(c) == n for c in cur):
            return power
        cur = [set().union(*(step[j] for j in c)) for c in cur]
    return None


def column_sums(sigma):
    n = len(sigma)
    cols = Counter()
    for row in sigma:
        for j in row:
            cols[j] += 1
    return sorted(set(cols[j] for j in range(n)))


def subset_automaton(sigma):
    """Reachable label sets under the 8 maps A -> {sigma[j][b] : j in A}, from the full set."""
    full = frozenset(range(len(sigma)))
    state = {full: 0}
    order = [full]
    trans = []
    i = 0
    while i < len(order):
        row = []
        for b in range(8):
            B = frozenset(sigma[j][b] for j in order[i])
            if B not in state:
                state[B] = len(order)
                order.append(B)
            row.append(state[B])
        trans.append(row)
        i += 1
    return order, trans


def coincidence(sigma):
    """(least depth M, coincident addresses at that depth, label), from the subset automaton."""
    order, trans = subset_automaton(sigma)
    frontier = {(): 0}
    for M in range(1, 8):
        nxt = {}
        for word, s in frontier.items():
            for b in range(8):
                nxt[word + (b,)] = trans[s][b]
        frontier = nxt
        hits = [w for w, s in frontier.items() if len(order[s]) == 1]
        if hits:
            addr = sorted(tuple(sum(BITS[d][ax] << (M - 1 - k) for k, d in enumerate(w)) for ax in range(3))
                          for w in hits)
            return M, addr, len(frontier)
    return None


def noncoincident_counts(sigma, depth=14):
    order, trans = subset_automaton(sigma)
    sizes = [len(A) for A in order]
    ns = [k for k in range(len(order)) if sizes[k] > 1]
    idx = {k: i for i, k in enumerate(ns)}
    T = [[0] * len(ns) for _ in ns]
    for k in ns:
        for b in range(8):
            d = trans[k][b]
            if d in idx:
                T[idx[k]][idx[d]] += 1
    v = [0] * len(ns)
    v[idx[0]] = 1
    counts = []
    for _ in range(depth + 1):
        counts.append(sum(v))
        v = [sum(v[i] * T[i][j] for i in range(len(ns))) for j in range(len(ns))]
    return counts, T, Counter(sizes)


def transfer_charpoly(T):
    x = sp.symbols('x')
    return sp.factor(sp.Matrix(T).charpoly(x).as_expr())


def report():
    labels, sigma = labels_and_sigma()
    exp = primitive_exponent(sigma)
    M, addr, words = coincidence(sigma)
    counts, T, hist = noncoincident_counts(sigma)
    cp = transfer_charpoly(T)
    return dict(labels=len(labels), column_sums=column_sums(sigma), primitive_exponent=exp,
                coincidence_depth=M, coincident_addresses=len(addr), words_at_depth=words,
                address_0_0_2=(0, 0, 2) in addr, counts=counts, subset_sizes=dict(hist),
                transfer_dim=len(T), charpoly=cp)


def main():
    r = report()
    print("cell substitution on %d labels; column sums %s; primitive at power %s"
          % (r["labels"], r["column_sums"], r["primitive_exponent"]))
    print("modular coincidence: least depth %d, %d of %d addresses coincide; (0,0,2) among them: %s"
          % (r["coincidence_depth"], r["coincident_addresses"], r["words_at_depth"], r["address_0_0_2"]))
    print("reachable label sets by size:", dict(sorted(r["subset_sizes"].items())))
    print("non-coincident words c(M), M = 0..14:", r["counts"])
    c = r["counts"]
    print("c(M+1) = 2 c(M) + 48 for M >= 1:", all(c[m + 1] == 2 * c[m] + 48 for m in range(1, len(c) - 1)),
          "; c(M) = 28*2^M - 48:", all(c[m] == 28 * 2 ** m - 48 for m in range(1, len(c))))
    print("transfer matrix on the %d non-singleton states, characteristic polynomial:" % r["transfer_dim"], r["charpoly"])
    print("spectral radius 2 -> the undetermined set in Z_2^3 has box dimension log2 2 = 1")


if __name__ == "__main__":
    main()
