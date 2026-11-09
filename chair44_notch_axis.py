#!/usr/bin/env python3
r"""chair44_notch_axis.py - where the Chair44 label is not determined by its 2-adic address, and what is still determined there.

CICY (pyCICY/chair44_nested.py) computes, for the 168-label cell substitution, the *coincidence* of the label at an address of the 2^M cube: after reading
M binary digits the set of labels still possible is tracked by a subset automaton (209 reachable sets, 41 with more than one label), the number of
addresses with an undetermined label is c(M) = 28 * 2^M - 48 and the label excess is E(M) = 140 * 2^M + 48.  This script recomputes the automaton from the
substitution of chair44_twoadic.py (which shares no code with CICY) and asks the question of chair44_polarisation.py instead: not the label, but its
notch vector d = g (1,1,1), the direction the notch of the chair points in.

Status of each statement (labels as in CONNECTIONS.md):
  Exact    K1  The subset automaton has 209 reachable sets and 41 with more than one label; c(M) = 28 * 2^M - 48 and E(M) = 140 * 2^M + 48 for M = 1..7
               (counted here) and, by the finite certificate of CICY, for all M.  Two independent implementations agree.
  Exact    K2  The notch AXIS (the notch vector up to sign) is determined by the address everywhere except at exactly 16 addresses of the 2^M cube, for
               every M >= 2: the 8 corner cells and the 8 central cells (M = 2..6 by position).  Proof: the number a_M of such addresses is u^T T^M e_0 for
               the 209-state transfer matrix T, so a_M - 16 satisfies a linear recurrence of order at most 210; it vanishes for M = 2..260.
  Counted  K3  At every other undetermined address (6 labels) the possible notch vectors are exactly +-v: the label is undetermined but the notch axis is not
               (the six labels form a coset of the order-6 dihedral stabiliser of the notch diagonal, CICY's six_is_a_dihedral_coset).  At the 16 exceptional
               addresses all 24 labels, hence all 8 corner directions, are possible.
  Not shown    Anything about the sign: the +- ambiguity at the undetermined addresses is the point inversion of the notch vector, and chair44 PC invariance
               (CHAIR44.md) says P alone is not a symmetry; this script does not connect the two.

    python3 chair44_notch_axis.py            report
    python3 chair44_notch_axis.py --tests    the checks above (about 5 s); with CICY_DIR set, also compares with CICY's own functions
"""
import itertools
import os
import sys
from collections import Counter

import numpy as np

import chair44_polarisation as P

BITS = list(itertools.product((0, 1), repeat=3))
_C = {}


def automaton():
    if "a" not in _C:
        sig, notch = P.setup()
        full = frozenset(range(len(sig)))
        order, idx, trans = [full], {full: 0}, []
        i = 0
        while i < len(order):
            row = []
            for b in range(8):
                B = frozenset(int(sig[j, b]) for j in order[i])
                if B not in idx:
                    idx[B] = len(order)
                    order.append(B)
                row.append(idx[B])
            trans.append(row)
            i += 1
        axes = []
        for A in order:
            axes.append(len({tuple(v) if tuple(v) > (0, 0, 0) else tuple(-x for x in v) for v in (tuple(int(t) for t in notch[j]) for j in A)}))
        vecs = [len({tuple(int(t) for t in notch[j]) for j in A}) for A in order]
        _C["a"] = (order, trans, axes, vecs)
    return _C["a"]


def table(Mmax):
    """-> rows (M, c, E, addresses with more than one notch axis, addresses with more than one notch vector)."""
    order, trans, axes, vecs = automaton()
    cnt, rows = {0: 1}, []
    for M in range(Mmax + 1):
        c = E = ax = vv = 0
        for s, n in cnt.items():
            k = len(order[s])
            if k > 1:
                c += n
            E += n * (k - 1)
            ax += n * (axes[s] > 1)
            vv += n * (vecs[s] > 1)
        rows.append((M, c, E, ax, vv))
        nxt = {}
        for s, n in cnt.items():
            for b in range(8):
                nxt[trans[s][b]] = nxt.get(trans[s][b], 0) + n
        cnt = nxt
    return rows


def exceptional_positions(M):
    order, trans, axes, vecs = automaton()
    T = np.array(trans, dtype=np.int32)
    ax = np.array(axes)
    g = np.zeros((1, 1, 1), dtype=np.int32)
    for m in range(1, M + 1):
        h = 2 ** (m - 1)
        new = np.empty((2 * h,) * 3, dtype=np.int32)
        for b, a in enumerate(BITS):
            new[a[0]::2, a[1]::2, a[2]::2] = T[:, b][g]
        g = new
    return np.argwhere(ax[g] > 1), ax[g]


def run_tests(verbose=True):
    ok = True

    def check(name, good, info=""):
        nonlocal ok
        ok &= bool(good)
        if verbose:
            print("  %-4s %-72s %s" % ("ok" if good else "FAIL", name, info))

    order, trans, axes, vecs = automaton()
    check("K1: 209 reachable label sets, 41 with more than one label", len(order) == 209 and sum(len(A) > 1 for A in order) == 41)
    rows = table(7)
    check("K1: c(M) = 28 * 2^M - 48 and E(M) = 140 * 2^M + 48 for M = 1..7", all(c == 28 * 2 ** M - 48 and e == 140 * 2 ** M + 48 for M, c, e, _, _ in rows[1:]),
          ", ".join("%d" % r[1] for r in rows[1:]))
    check("K2: exactly 16 addresses with an undetermined notch axis, M = 2..7", all(r[3] == 16 for r in rows[2:]))
    rows = table(260)
    check("K2: the same for every M = 2..260 (certificate: recurrence of order <= 210)", all(r[3] == 16 for r in rows[2:]), "M up to 260")
    for M in (2, 3, 4, 5, 6):
        pos, _ = exceptional_positions(M)
        n = 2 ** M
        corners = {(a, b, c) for a in (0, n - 1) for b in (0, n - 1) for c in (0, n - 1)}
        centre = {(a, b, c) for a in (n // 2 - 1, n // 2) for b in (n // 2 - 1, n // 2) for c in (n // 2 - 1, n // 2)}
        check("K2: M = %d the 16 are the 8 corners and the 8 central cells" % M, {tuple(p) for p in pos} == corners | centre and len(pos) == 16)
    order, trans, axes, vecs = automaton()
    six = [(len(order[s]), vecs[s], axes[s]) for s in range(len(order)) if len(order[s]) == 6]
    check("K3: every 6-label set has exactly 2 notch vectors, +-v, so one axis", six and all(v == 2 and a == 1 for _, v, a in six), "%d sets" % len(six))
    big = [(len(order[s]), vecs[s], axes[s]) for s in range(len(order)) if len(order[s]) == 24]
    check("K3: the 24-label sets carry all 8 notch vectors and 4 axes", big and all(v == 8 and a == 4 for _, v, a in big), "%d sets" % len(big))
    d = os.environ.get("CICY_DIR")
    if d:
        sys.path.insert(0, d)
        from pyCICY import chair44_nested as C
        theirs = C.coincidence_table(7)
        mine = table(7)
        check("K1: agrees with CICY's coincidence_table for M = 0..7", all(t["non_coincident"] == m[1] and t["excess"] == m[2] for t, m in zip(theirs, mine)))
        tubes = C.undetermined_is_diagonal_tubes(5)
        check("K1: CICY's four diagonal tubes cover our undetermined set (their check passes)", all(r["equal"] and r["c_formula"] and r["excess_formula"] for r in tubes))
    else:
        print("  skip CICY comparison (set CICY_DIR to a clone of brentharts/CICY)")
    if verbose:
        print("OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    for M, c, E, ax, vv in table(8):
        print("M=%d  undetermined label: %d   label excess: %d   undetermined notch axis: %d   undetermined notch vector: %d" % (M, c, E, ax, vv))
