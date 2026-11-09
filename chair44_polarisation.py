#!/usr/bin/env python3
r"""chair44_polarisation.py - the notch vector along every axis line of a Chair44 patch: bounded along the line, ballistic across it.

chair44_lines.py computes the line statistics of a few named lines (the chair stacks along the body diagonal, the corner lines).  This
script measures ALL axis lines of a depth-n cell patch (side 2^n, 8^n cells), built from the 168-label cell substitution sigma of
chair44_twoadic.py, and looks at the notch vector d = g (1,1,1) (a corner direction, components +-1) as a step of a walk along the line.

Measured (a finite computation, n <= 7; labelled Counted, not proved):
  1. Box mean: the mean of d over the patch is ~0 (the 168 labels sum to 0, and the patch mean is 1.7e-3 at n = 6).
  2. LONGITUDINAL: the sum of the component of d along the line direction, over a whole axis line, is bounded (|sum| <= 6 for every one of
     the 3 * 4^n axis lines, n = 3..7): the walk closes along the line.
  3. TRANSVERSE: the two components perpendicular to the line grow ballistically on most lines.  The full-line sum is exactly (+-M, +-M) (M = 2^n,
     so a drift of (+-1, +-1) per cell) on a fraction of lines that is 0.5625, 0.547, 0.543, 0.542, 0.5417 for n = 3..7.  EXACT (section "exact fraction" below):
     the fraction is 13/24 + 1/(3 * 4^(n-1)) for every n >= 3 and every root label, so it tends to 13/24; the four sign states are equally populated.
  4. Negative control: the same labels shuffled in space give essentially no ballistic line (drift ~ 1/sqrt(M)).
  5. The drift is a property of the unoriented line: the sum does not change if the line is traversed backwards, so it singles out a line, not a
     direction of travel.  It gives no arrow of time by itself.
  6. The body diagonal is different.  Of the diagonal lines of length >= M/2 (root 5), the fraction whose longitudinal drift (sum of the three
     components of d, per cell) is below 0.5 in modulus is 0.913 at n = 5 and 0.942 at n = 7, so ballistic diagonals are a minority that shrinks; the
     largest longitudinal drift is exactly 1 per cell (of a possible 3), reached by the chair-stack line (chair44_lines.py).

Exact fraction.  "Transverse sum (+-M, +-M)" means every cell of the line has the same pair (d_x, d_y) = s, i.e. the line word lies in Sigma_s, the 42 labels
with that pair.  Write the line word at address (x, y) as tau_{d_0} ... tau_{d_{n-1}}(j), tau_{(a,b)}(l) = sigma[l,(a,b,0)] sigma[l,(a,b,1)], d_i the i-th
binary digits of (x, y).  The word lies in Sigma_s iff j lies in Pre_{d_{n-1}}( ... Pre_{d_0}(Sigma_s)), Pre_d(T) = {l : tau_d(l) in T}.  Starting from the
four Sigma_s, only 18 subsets are reachable, so counting digit strings is a walk on an 18-state graph with integer matrix Q (Q[T, Pre_d T] += 1).  The
count a_n(j) of ballistic lines is u^T Q^n w, and (Q - 4)(Q - 1) Q^3 = 0 exactly, so a_{n+2} = 5 a_{n+1} - 4 a_n for n >= 3; with a_3 = 36 and a_4 = 140
for all 168 roots this gives a_n = (13/24) 4^n + 4/3, i.e. the fraction 13/24 + 1/(3 * 4^(n-1)).  This is for the lines of the depth-n patches sigma^n(j),
that is, a statement about the 2-adic address (x, y), not a density in a particular infinite tiling.

What this is not: a statement about irrational directions, about the pyramid observable, or about any physical system.

    python3 chair44_polarisation.py            report
    python3 chair44_polarisation.py --tests    the checks above (about 15 s)
"""
import itertools
import sys
from collections import Counter

import numpy as np

import chair44_twoadic as T
import symchair44 as S

BITS = list(itertools.product((0, 1), repeat=3))
_C = {}


def setup():
    if "sig" not in _C:
        labels, sigma = T.labels_and_sigma()
        _C["sig"] = np.array(sigma)
        _C["notch"] = np.array([S.frame_apply(f, (1, 1, 1)) for f, _ in labels])
    return _C["sig"], _C["notch"]


def cube(j, n):
    """Label array of side 2^n: the depth-n cell substitution of label j."""
    sig, _ = setup()
    L = np.array([[[j]]])
    for _ in range(n):
        m = L.shape[0]
        N = np.empty((2 * m,) * 3, dtype=int)
        for b, a in enumerate(BITS):
            N[a[0]::2, a[1]::2, a[2]::2] = sig[L, b]
        L = N
    return L


def notch_cube(j, n):
    return setup()[1][cube(j, n)]


def stack_labels():
    sig, _ = setup()
    return [j for j in range(len(sig)) if sig[j, 0] == j and sig[j, 7] == j]


def line_sums(V, axis):
    """Integer sums of d over every full line along `axis`: array (M, M, 3)."""
    return V.sum(axis=axis)


def classify(j, n, axis=2):
    """-> dict with max longitudinal |sum|, fraction of lines with transverse sum (+-M, +-M), counts of the four sign states, fraction of zero lines."""
    V = notch_cube(j, n)
    M = 2 ** n
    Ssum = line_sums(V, axis).reshape(-1, 3)
    tr = [k for k in range(3) if k != axis]
    long_max = int(np.abs(Ssum[:, axis]).max())
    states = Counter()
    for v in Ssum:
        if abs(v[tr[0]]) == M and abs(v[tr[1]]) == M:
            states[(int(np.sign(v[tr[0]])), int(np.sign(v[tr[1]])))] += 1
    tot = len(Ssum)
    return dict(M=M, lines=tot, long_max=long_max, ballistic=sum(states.values()) / tot, states=dict(states),
                zero=float((np.abs(Ssum).sum(1) == 0).mean()), box_mean=V.reshape(-1, 3).mean(0))


def shuffled_ballistic(j, n, seed=0):
    V = notch_cube(j, n)
    M = 2 ** n
    rng = np.random.default_rng(seed)
    W = V.reshape(-1, 3)[rng.permutation(M ** 3)].reshape(M, M, M, 3)
    Ssum = W.sum(axis=2).reshape(-1, 3)
    return float((np.abs(Ssum[:, :2]).max(1) >= 0.5 * M).mean()), float(np.abs(Ssum[:, :2]).mean() / M)


# ------------------------------------------------------------------------------ exact fraction
D4 = [(a, b) for a in (0, 1) for b in (0, 1)]


def _pre(T, d):
    sig, _ = setup()
    bi = {b: i for i, b in enumerate(BITS)}
    a, b = d
    return frozenset(l for l in range(len(sig)) if sig[l, bi[(a, b, 0)]] in T and sig[l, bi[(a, b, 1)]] in T)


def sigma_classes():
    """{(sx, sy): frozenset of labels with notch components (sx, sy)}, the 4 classes of 42."""
    _, notch = setup()
    return {(sx, sy): frozenset(l for l in range(len(notch)) if notch[l][0] == sx and notch[l][1] == sy)
            for sx in (1, -1) for sy in (1, -1)}


def automaton():
    """-> (order, index, Q): the reachable subsets, their index, and the integer matrix Q[T, Pre_d(T)] += 1 (d over the 4 digit pairs)."""
    if "auto" not in _C:
        order, index = [], {}
        for S0 in sigma_classes().values():
            if S0 not in index:
                index[S0] = len(order)
                order.append(S0)
        i = 0
        while i < len(order):
            for d in D4:
                U = _pre(order[i], d)
                if U not in index:
                    index[U] = len(order)
                    order.append(U)
            i += 1
        Q = np.zeros((len(order), len(order)), dtype=object)
        for T in order:
            for d in D4:
                Q[index[T], index[_pre(T, d)]] += 1
        _C["auto"] = (order, index, Q)
    return _C["auto"]


def ballistic_count(j, n):
    """Exact number of the 4^n z-lines of the depth-n patch of root j whose line word lies in some Sigma_s (so with sum (+-M, +-M))."""
    order, index, Q = automaton()
    tot = 0
    for S0 in sigma_classes().values():
        v = np.zeros(len(order), dtype=object)
        v[index[S0]] = 1
        for _ in range(n):
            v = v.dot(Q)
        tot += sum(v[index[T]] for T in order if j in T)
    return int(tot)


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    sig, notch = setup()
    check("168 labels; the notch vectors of the 168 labels sum to 0 (box mean of d is 0)", len(sig) == 168 and not notch.sum(0).any())
    roots = (5, 100, 49)
    res = {}
    for j in roots:
        for n in (3, 5, 7):
            if n == 7 and j != 5:
                continue
            res[(j, n)] = classify(j, n)
    r6 = classify(5, 6)
    check("patch mean of d is small (n = 6)", np.abs(r6["box_mean"]).max() < 0.005, "(%s)" % np.round(r6["box_mean"], 4))
    check("longitudinal sum bounded: |sum along the line| <= 6 on every z line, n = 3, 5, 7 and three root labels",
          all(r["long_max"] <= 6 for r in res.values()), "(max %d)" % max(r["long_max"] for r in res.values()))
    # the same for x and y lines
    check("same bound on x and y lines (n = 5)", all(classify(5, 5, ax)["long_max"] <= 6 for ax in (0, 1)))
    f7 = res[(5, 7)]["ballistic"]
    check("transverse sum is exactly (+-M, +-M) on a fraction of lines near 13/24 (n = 7)", abs(f7 - 13 / 24) < 0.002, "(%.4f)" % f7)
    check("the fraction does not depend on the root label (n = 5)", max(abs(res[(j, 5)]["ballistic"] - res[(5, 5)]["ballistic"]) for j in roots) < 1e-9)
    st = res[(5, 7)]["states"]
    check("the four sign states (+-,+-) are equally populated to within 2%", len(st) == 4 and max(st.values()) / min(st.values()) < 1.02, "(%s)" % st)
    fr, mean_abs = shuffled_ballistic(5, 6)
    check("negative control: with the labels shuffled in space no line is ballistic", fr == 0.0 and mean_abs < 0.2, "(fraction %.4f, mean |drift| %.3f)" % (fr, mean_abs))
    # stacks: ballistic along the body diagonal
    stacks = stack_labels()
    j = stacks[0]
    L = cube(j, 5)
    diag = [L[t, t, t] for t in range(32)]
    check("on a chair stack the main diagonal is one repeated label, so the diagonal sum is 32 d(j)", set(diag) == {j} and (notch[j] != 0).all())
    # ---- exact fraction (the automaton)
    order, index, Q = automaton()
    check("the four classes Sigma_s have 42 labels each, and only 18 subsets are reachable under the four Pre maps",
          sorted(len(v) for v in sigma_classes().values()) == [42] * 4 and len(order) == 18)
    brute = {(5, n): round(classify(5, n)["ballistic"] * 4 ** n) for n in (3, 4, 5, 6, 7)}
    brute.update({(j, 5): round(classify(j, 5)["ballistic"] * 4 ** 5) for j in (100, 49)})
    check("automaton counts equal the brute-force counts of the patches (root 5, n = 3..7; roots 100, 49, n = 5)",
          all(ballistic_count(j, n) == c for (j, n), c in brute.items()), "(%s)" % [brute[(5, n)] for n in (3, 4, 5, 6, 7)])
    I = np.eye(len(order), dtype=object)
    Z = (Q - 4 * I).dot(Q - I).dot(Q).dot(Q).dot(Q)
    check("(Q - 4)(Q - 1)Q^3 = 0 exactly, so every count obeys a_(n+2) = 5 a_(n+1) - 4 a_n for n >= 3", not Z.any())
    a3 = [ballistic_count(j, 3) for j in range(168)]
    a4 = [ballistic_count(j, 4) for j in range(168)]
    check("all 168 roots have a_3 = 36 and a_4 = 140, hence a_n = (13/24) 4^n + 4/3 and the fraction is 13/24 + 1/(3 * 4^(n-1)) for every n >= 3, limit 13/24",
          set(a3) == {36} and set(a4) == {140})
    # body-diagonal lines of length >= M/2: most have zero drift vector, a shrinking minority is ballistic
    def diag_stats(n):
        V = notch_cube(5, n)
        M = 2 ** n
        zero = tot = 0
        top = 0.0
        for a in range(M):
            for b in range(M):
                ln = M - max(a, b)
                if ln < M // 2:
                    continue
                k = np.arange(ln)
                s_ = V[a + k, b + k, k].sum(0)
                zero += abs(float(s_ @ np.ones(3))) / ln < 0.5
                tot += 1
                top = max(top, abs(float(s_ @ np.ones(3))) / ln)
        return zero / tot, top
    z5, t5 = diag_stats(5)
    z7, t7 = diag_stats(7)
    check("body-diagonal lines (length >= M/2, root 5): fraction with longitudinal drift < 0.5 per cell rises from n = 5 to n = 7 and exceeds 0.9; the largest longitudinal drift is 1 per cell",
          z5 < z7 and z7 > 0.9 and abs(t5 - 1.0) < 1e-9 and abs(t7 - 1.0) < 1e-9, "(%.3f -> %.3f; max %.2f)" % (z5, z7, t7))
    return ok


def report():
    print("axis lines, notch vector d, root label 5")
    print("%3s %6s %9s %10s %12s" % ("n", "M", "lines", "long_max", "ballistic"))
    for n in range(3, 8):
        r = classify(5, n)
        print("%3d %6d %9d %10d %12.4f" % (n, r["M"], r["lines"], r["long_max"], r["ballistic"]))
    print("exact: 13/24 + 1/(3*4^(n-1)); limit 13/24 = %.4f" % (13 / 24))


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    report()
