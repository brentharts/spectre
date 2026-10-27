#!/usr/bin/env python3
r"""chair44_lines.py - lines through a Chair44 tiling: box averages versus line averages.

The hull of the Chair44 substitution is uniquely ergodic for the translation action of Z^3 (Romanov; this paper,
Section "Every tiling lies in the hull"), so the frequency of every finite patch exists, uniformly, in boxes.  This
module computes, exactly, what happens along a LINE of cells, and shows that the box result does not transfer.

What is computed (exact integers / fractions, from the cell substitution of chair44_twoadic and the tile of symchair44):

  1. BOX MEANS.  The 168 x 168 matrix of the cell substitution has all row and column sums 8, so (being primitive)
     every label has frequency 1/168 in every tiling.  Hence the box mean of a cell observable is its average over the
     168 labels: 0 for the pyramid observable (the signed heights of the pyramids on the faces of the cell; the
     heights sum to zero) and 0 for each component of the notch vector d = g(1,1,1).

  2. THE CHAIR STACK.  Six labels j satisfy sigma[j,0] = sigma[j,7] = j, so the main diagonal of sigma^n(j) is the
     single label j repeated.  They are native cell 0 of the chairs whose notch points along +-(1,1,1).  The chairs
     of sigma^n(j) then form a stack, each filling the notch of the next, and EVERY one of the seven cells of the
     chair gives a diagonal line of identical labels.  Along such a line a cell observable f is constant, so its
     Birkhoff sum is L f(label): for the pyramid observable -36 L and +36 L on the two cells that carry
     height, and +-L for every notch component.  The box mean is 0.

  3. CORNER LINES.  Along the line (t b) through the corner of sigma^n(j), b in {0,1}^3, the labels are the fixed
     word of a constant-length-2 substitution tau_b(j) = sigma[j,0] sigma[j,b], on the same 168 letters.  Its closed
     strongly connected classes and their exact letter frequencies give the exact limit of the line means of every
     cell observable, for every starting label.  They are not the box means.

What this is not.  It is exact for the lines it names (the stacks, the corner lines).  The general statement, that
no lattice direction is uniquely ergodic for the Z-action, is an argument in the paper from the internal space Z_2^3
and is conditional on it.  Lines in irrational directions are not covered by anything here.

    python3 chair44_lines.py            report
    python3 chair44_lines.py --tests    check the numbers above (about 15 s)
"""
import itertools
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

import chair44_hull as HU
import symchair44 as S

DIRECTIONS = [b for b in itertools.product((0, 1), repeat=3) if any(b)]


def _st():
    return HU._setup()


def primitive(B, limit=None):
    """True if the 0/1 matrix B is primitive (some power is positive); boolean iteration, no overflow.  By Wielandt a
    primitive n x n matrix has exponent at most (n-1)^2 + 1."""
    B = (np.asarray(B) > 0).astype(np.int64)
    n = B.shape[0]
    P = np.eye(n, dtype=np.int64)
    for _ in range(limit or (n - 1) ** 2 + 1):
        P = ((P @ B) > 0).astype(np.int64)
        if (P > 0).all():
            return True
    return False


def b_index(b):
    return b[0] * 4 + b[1] * 2 + b[2]


# --------------------------------------------------------------------------- observables
def pyramid_per_native_cell():
    """Integer sum of the signed pyramid heights on the tile-surface faces of each native cell, S.CELLS order."""
    h = [0] * len(S.CELLS)
    lab = S.label_table(S.CHILDREN)
    for pi, (ax, k) in enumerate(S.PANELS):
        sd = S.PANEL_SIGN[pi]
        cell = tuple((k[i] - 1 - (sd if i == ax else 0)) // 2 for i in range(3))
        h[S.CELLS.index(cell)] += sum(int(lab[pi * 8 + j]) for j in range(8))
    return h


def observables():
    """dict name -> list of 168 integers: 'pyramid', 'notch_x', 'notch_y', 'notch_z' (value on each label)."""
    st = _st()
    h = pyramid_per_native_cell()
    pyr = [h[k] for _, k in st['labels']]
    nt = [S.frame_apply(f, (1, 1, 1)) for f, _ in st['labels']]
    return dict(pyramid=pyr, notch_x=[int(v[0]) for v in nt], notch_y=[int(v[1]) for v in nt],
                notch_z=[int(v[2]) for v in nt])


# --------------------------------------------------------------------------- 1. box means
def full_matrix():
    st = _st()
    M = np.zeros((st['n'], st['n']), dtype=np.int64)
    for j in range(st['n']):
        for b in range(8):
            M[st['sigma'][j, b], j] += 1
    return M


def box_means():
    """Row/column sums of the 168 x 168 matrix, primitivity, and the exact box means (frequency 1/168 each)."""
    M = full_matrix()
    n = M.shape[0]
    rows, cols = set(M.sum(axis=1).tolist()), set(M.sum(axis=0).tolist())
    obs = observables()
    return dict(row_sums=sorted(rows), col_sums=sorted(cols), primitive=primitive(M),
                means={k: Fraction(sum(v), n) for k, v in obs.items()})


# --------------------------------------------------------------------------- 2. the chair stack
def stack_labels():
    st = _st()
    return [j for j in range(st['n']) if st['sigma'][j, 0] == j and st['sigma'][j, 7] == j]


def stack_check(depth=6):
    """For each stack label j: the diagonal of sigma^depth(j) is constant; and for every native cell k of the chair
    there is a diagonal line of identical labels (frame of j, k), of length at least 2^depth - 2 (exactly 2^depth on the main diagonal and 2^depth - 1 on the others).  Returns a list of dicts."""
    st = _st()
    N = 2 ** depth
    obs = observables()
    out = []
    for j in stack_labels():
        A = np.array([j], dtype=np.int64).reshape(1, 1, 1, 1)
        for _ in range(depth):
            A = HU._inflate(A)
        cube = A[0]
        frame = st['labels'][j][0]
        diag = bool(all(cube[t, t, t] == j for t in range(N)))
        offsets = {}
        for k in range(7):
            lab = st['index'][(frame, k)]
            offsets[k] = [p for p in itertools.product((0, 1), repeat=3)
                          if all(cube[p[0] + m, p[1] + m, p[2] + m] == lab for m in range(N - 2))]
        used = sorted(p for ps in offsets.values() for p in ps)
        out.append(dict(label=j, notch=tuple(int(x) for x in S.frame_apply(frame, (1, 1, 1))), diagonal_constant=diag,
                        one_line_per_cell=all(len(v) >= 1 for v in offsets.values()),
                        cells_cover_cube=sorted(set(used)) == sorted(itertools.product((0, 1), repeat=3)),
                        pyramid_per_step={k: obs['pyramid'][st['index'][(frame, k)]] for k in range(7)},
                        length=N - 2))
    return out


def tube_invariance(j, depth=3):
    """The finite check that makes the chair stack hold for EVERY n, not just the depth that was inspected.

    The seven diagonal lines of the stack fill the 'tube' of cells (x,y,z) whose difference e = (x-z, y-z) lies in a
    set T of 7 values; the cell with difference e has the label Lam(e) = (frame of j, k).  Lam is read off sigma^depth(j).
    Claim: for every n, every cell of sigma^n(j) with difference in T has label Lam(difference).  Induction on n: for
    n = 0 the single cell has difference (0,0) and label j = Lam(0,0).  A cell (X,Y,Z) of sigma^(n+1)(j) is child
    (a,b,c) of the cell (x,y,z) = ((X-a)/2, ...) of sigma^n(j); its parent difference is e' = ((e0-a+c)/2, (e1-b+c)/2).
    It suffices that for every e in T and every child position (a,b,c) compatible with e (parities), e' is in T and
    sigma[Lam(e'), (a,b,c)] = Lam(e).  That is 7 x 2 = 14 cases.  Returns (holds, number of cases, T)."""
    st = _st()
    N = 2 ** depth
    A = np.array([j], dtype=np.int64).reshape(1, 1, 1, 1)
    for _ in range(depth):
        A = HU._inflate(A)
    cube = A[0]
    frame = st['labels'][j][0]
    Lam = {}
    for k in range(7):
        lab = st['index'][(frame, k)]
        for p in itertools.product((0, 1), repeat=3):
            if all(cube[p[0] + m, p[1] + m, p[2] + m] == lab for m in range(N - 2)):
                Lam.setdefault((p[0] - p[2], p[1] - p[2]), set()).add(lab)
    if any(len(v) != 1 for v in Lam.values()) or len(Lam) != 7:
        return False, 0, sorted(Lam)
    Lam = {e: next(iter(v)) for e, v in Lam.items()}
    cases, holds = 0, Lam.get((0, 0)) == j
    for e, lab in Lam.items():
        for a, b, c in itertools.product((0, 1), repeat=3):
            if (e[0] - a + c) % 2 or (e[1] - b + c) % 2:
                continue
            e2 = ((e[0] - a + c) // 2, (e[1] - b + c) // 2)
            cases += 1
            holds &= e2 in Lam and int(st['sigma'][Lam[e2], b_index((a, b, c))]) == lab
    return bool(holds), cases, sorted(Lam)


# --------------------------------------------------------------------------- 3. corner lines
def corner_classes(b):
    """Closed strongly connected classes of tau_b(j) = sigma[j,0] sigma[j,b], with exact letter frequencies and the
    exact mean of every observable over each class."""
    from scipy.sparse.csgraph import connected_components
    st = _st()
    n, sg = st['n'], st['sigma']
    bi = b_index(b)
    M = np.zeros((n, n), dtype=np.int64)
    for j in range(n):
        for c in (sg[j, 0], sg[j, bi]):
            M[c, j] += 1
    ncomp, comp = connected_components(M.T > 0, directed=True, connection='strong')
    obs = observables()
    classes = []
    for c in range(ncomp):
        members = [j for j in range(n) if comp[j] == c]
        if any(M[k, j] and comp[k] != c for j in members for k in range(n)):
            continue                                          # not closed
        sub = sp.Matrix(M[np.ix_(members, members)].tolist())
        ns = (sub - 2 * sp.eye(len(members))).nullspace()
        assert len(ns) == 1
        pi = ns[0] / sum(ns[0])
        freq = {m: pi[i] for i, m in enumerate(members)}
        means = {k: sum(freq[m] * v[m] for m in members) for k, v in obs.items()}
        # aperiodicity: the class's matrix power has no zero entries after a few steps (primitive)
        classes.append(dict(members=members, size=len(members), frequencies=freq,
                            means={k: Fraction(int(v.p), int(v.q)) for k, v in means.items()},
                            primitive=primitive(M[np.ix_(members, members)])))
    # every label flows into a closed class: the substitution graph is finite, so a closed class is reachable
    reach_ok = True
    A = M.T > 0
    closed_members = {m for c in classes for m in c['members']}
    for j in range(n):
        seen, todo = {j}, [j]
        while todo:
            x = todo.pop()
            for y in np.where(A[x])[0]:
                if y not in seen:
                    seen.add(int(y)); todo.append(int(y))
        reach_ok &= bool(seen & closed_members)
    return dict(direction=b, classes=classes, all_reach_closed=reach_ok)


def finite_line_mean(j, b, depth, name):
    """Mean of observable `name` over the 2^depth cells of the corner line of sigma^depth(j) in direction b."""
    st = _st()
    sg = st['sigma']
    f = observables()[name]
    seq = [j]
    for _ in range(depth):
        seq = [c for x in seq for c in (sg[x, 0], sg[x, b_index(b)])]
    return Fraction(sum(f[x] for x in seq), len(seq)), seq


def corner_line_matches_cube(j, b, depth=5):
    """The corner line read off tau_b equals the line of cells (t b) of the label cube sigma^depth(j)."""
    A = np.array([j], dtype=np.int64).reshape(1, 1, 1, 1)
    for _ in range(depth):
        A = HU._inflate(A)
    cube = A[0]
    _, seq = finite_line_mean(j, b, depth, 'pyramid')
    return all(int(cube[t * b[0], t * b[1], t * b[2]]) == int(seq[t]) for t in range(2 ** depth))


# --------------------------------------------------------------------------- report / tests
def report():
    bm = box_means()
    sc = stack_check()
    cc = {b: corner_classes(b) for b in DIRECTIONS}
    tube = {j: tube_invariance(j) for j in stack_labels()}
    return dict(box=bm, stack=sc, corner=cc, pyramid_native=pyramid_per_native_cell(), tube=tube)


def run_tests(verbose=True):
    r = report()
    bm, sc = r['box'], r['stack']
    ok = True

    def chk(c, msg):
        nonlocal ok
        if not c:
            ok = False
            if verbose:
                print('FAIL:', msg)

    chk(bm['row_sums'] == [8] and bm['col_sums'] == [8] and bm['primitive'], 'matrix sums / primitivity')
    chk(all(v == 0 for v in bm['means'].values()), 'box means are zero')
    chk(r['pyramid_native'] == [0, 0, 0, -36, 0, 36, 0] and sum(r['pyramid_native']) == 0, 'pyramid per native cell')
    chk(len(sc) == 6 and sorted(s['label'] for s in sc) == stack_labels(), 'six stack labels')
    chk(sorted(s['notch'] for s in sc) == sorted([(1, 1, 1)] * 3 + [(-1, -1, -1)] * 3), 'notches along the diagonal')
    chk(all(s['diagonal_constant'] and s['one_line_per_cell'] and s['cells_cover_cube'] and s['length'] == 62
            for s in sc), 'stack lines')
    chk(all(sorted(s['pyramid_per_step'].values()) == [-36, 0, 0, 0, 0, 0, 36] for s in sc), 'pyramid steps -36 and +36')
    chk(all(h and c == 14 and len(T) == 7 for h, c, T in r['tube'].values()), 'tube invariance (all n by induction)')
    # the stack lines' means are not the box means
    st = _st()
    obs = observables()
    for s in sc:
        j = s['label']
        chk(obs['notch_x'][j] == s['notch'][0] != 0, 'notch component on a stack line is +-1')
    # corner lines
    for b, c in r['corner'].items():
        chk(c['all_reach_closed'] and all(k['primitive'] for k in c['classes']), 'corner classes %s' % (b,))
    chk(len(r['corner'][(1, 1, 1)]['classes']) == 6 and all(k['size'] == 1 for k in r['corner'][(1, 1, 1)]['classes']),
        'body diagonal: six fixed letters')
    for b in DIRECTIONS[:6]:
        cl = r['corner'][b]['classes']
        chk(len(cl) == 1 and cl[0]['size'] == 4, 'axis and face diagonals: one class of 4')
    # the exact class frequencies reproduce the finite line means from the substitution and from the cube
    for b in DIRECTIONS:
        for cls in r['corner'][b]['classes']:
            for j in cls['members'][:2]:
                for name in ('pyramid', 'notch_x', 'notch_y', 'notch_z'):
                    m, _ = finite_line_mean(j, b, 10, name)
                    chk(abs(m - cls['means'][name]) <= Fraction(1, 2 ** 6) * max(1, abs(cls['means'][name]) + 1),
                        'finite line mean near the class mean %s %s' % (b, name))
        chk(corner_line_matches_cube(r['corner'][b]['classes'][0]['members'][0], b), 'corner line equals cube line %s' % (b,))
    # the notch component d_z on two coordinate-axis corner lines: bounded sums on one, sums L on the other
    for b, bounded in (((0, 0, 1), True), ((1, 0, 0), False)):
        for j in r['corner'][b]['classes'][0]['members']:
            _, seq = finite_line_mean(j, b, 12, 'notch_z')
            sums = np.cumsum([obs['notch_z'][x] for x in seq])
            if bounded:
                chk(set(sums.tolist()) <= {0, 1, 2},
                    'd_z sums bounded on the corner line (0,0,1)')
            else:
                chk(sums.tolist() == list(range(1, len(seq) + 1)), 'd_z sums equal L on the corner line (1,0,0)')
    tz = [(int(st['sigma'][j, 0]), int(st['sigma'][j, b_index((0, 0, 1))])) for j in r['corner'][(0, 0, 1)]['classes'][0]['members']]
    chk(len(set(tz)) == 2, 'tau_(0,0,1) sends its closed class onto two words')
    if verbose:
        print('chair44_lines:', 'ok' if ok else 'FAIL')
    return ok


def main():
    if '--tests' in sys.argv:
        sys.exit(0 if run_tests() else 1)
    r = report()
    bm = r['box']
    print('row sums %s col sums %s primitive %s' % (bm['row_sums'], bm['col_sums'], bm['primitive']))
    print('box means:', {k: str(v) for k, v in bm['means'].items()})
    print('pyramid sum per native cell:', r['pyramid_native'])
    print('chair-stack labels:')
    for s in r['stack']:
        print('   ', s)
    print('tube invariance (holds, cases, differences):', {j: v for j, v in r['tube'].items()})
    print('corner lines (direction: closed classes with exact means)')
    for b, c in r['corner'].items():
        for k in c['classes']:
            print('   %s  size %d  means %s' % (b, k['size'], {n: str(v) for n, v in k['means'].items()}))


if __name__ == '__main__':
    main()
