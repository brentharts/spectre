#!/usr/bin/env python3
r"""chair44_hull.py - the legal vertex blocks of the Chair44 substitution, recomputed exactly.

Romanov (this paper, Section "Every tiling lies in the hull") proves that every Chair44 tiling lies in the
substitution hull, and that the symmetry group of a tiling is one of 1, C2 (about a face-diagonal axis), C4, D4.
His proof combines one level of the hierarchy with a finite computation on the 8 unit cells around a lattice
vertex, run on Tsiokos's published atlas.  This module recomputes the part of that finite computation that
depends only on the substitution, from `symchair44` (which generates the tile from nothing), so that it
shares no data with Tsiokos's repository.

A BLOCK is the labelling of the 2x2x2 unit cells around a lattice vertex; a label is (frame, native cell), 168 of
them.  A block is LEGAL if it occurs in some sigma^n(j).  What is computed (exact integers):

  1. the legal blocks: every 2x2x2 window of sigma^n(j), n = 1..6, over all 168 labels j:
        168, 600, 1278, 1398, 1410, 1410 new-or-repeated windows, 1410 distinct in all
  2. that 1410 is the whole language and not merely where the count stopped: the set L is closed, i.e. all 27
     windows of size 2x2x2 of sigma(w) lie in L for every w in L, and L contains the windows of sigma(j); then
     every window of every sigma^n(j) is in L, because a window of sigma^{n+1}(j) lies in sigma(w) for a
     2x2x2 block w of sigma^n(j)
  3. Phi(w), the central 2x2x2 window of sigma(w): Phi maps L to L, its periodic points are 27 blocks, all
     fixed, which are Romanov's "27 fixed seeds"
  4. the rotation group O (24 proper rotations) acts on L; the stabiliser orders over L and over the fixed seeds
     (1 and 8; the order-8 stabilisers are D4, the only subgroup of order 8 of O up to conjugacy)

What this is not.  It does not count the ADMISSIBLE blocks (360,189 in Romanov's note), which need the contact
atlas and the registration lemma and are his computation; it does not check the axial-group analysis of the
symmetry theorem.  The registration (R) and parent-tiling (H) facts are Tsiokos's and Goodman-Strauss's and are
not re-derived.

    python3 chair44_hull.py            report
    python3 chair44_hull.py --tests    check the numbers above (about 5 s)
"""
import itertools
import sys
from collections import Counter

import numpy as np

import chair44_twoadic as TA
import symchair44 as S

POS = list(itertools.product((0, 1), repeat=3))        # (dx, dy, dz) order of a block
_STATE = {}


def _setup():
    if not _STATE:
        labels, sigma = TA.labels_and_sigma()
        _STATE.update(labels=labels, n=len(labels), sigma=np.array(sigma, dtype=np.int64),
                      index={l: i for i, l in enumerate(labels)},
                      rots=[f for f in S.all_frames() if S.frame_det(f) == 1])
    return _STATE


def encode(block):
    n = _setup()['n']
    c = 0
    for x in block:
        c = c * n + int(x)
    return c


def decode(code):
    n = _setup()['n']
    out = []
    for _ in range(8):
        out.append(code % n)
        code //= n
    return tuple(reversed(out))


def _inflate(A):
    """(L, N, N, N) array of labels -> (L, 2N, 2N, 2N): every cell replaced by its 2x2x2 children."""
    sg = _setup()['sigma']
    L, N = A.shape[0], A.shape[1]
    out = np.empty((L, 2 * N, 2 * N, 2 * N), dtype=np.int64)
    for b in range(8):
        out[:, (b >> 2) & 1::2, (b >> 1) & 1::2, b & 1::2] = sg[A, b]
    return out


def _window_codes(A):
    """Distinct codes of all 2x2x2 windows of every cube in A."""
    n, N = _setup()['n'], A.shape[1]
    code = np.zeros((A.shape[0], N - 1, N - 1, N - 1), dtype=np.int64)
    for dx, dy, dz in POS:
        code = code * n + A[:, dx:N - 1 + dx, dy:N - 1 + dy, dz:N - 1 + dz]
    return np.unique(code.ravel())


def legal_blocks(depth=6):
    """(sorted codes of the legal blocks, [number of distinct windows of sigma^n(j) for n = 1..depth])."""
    if 'legal' not in _STATE:
        st = _setup()
        A = np.arange(st['n'], dtype=np.int64).reshape(st['n'], 1, 1, 1)
        seen, per_depth = set(), []
        for _ in range(depth):
            A = _inflate(A)
            w = _window_codes(A)
            per_depth.append(len(w))
            seen |= set(w.tolist())
        _STATE['legal'] = (sorted(seen), per_depth)
    return _STATE['legal']


def sigma_cube(code):
    """4x4x4 array: sigma applied to the 8 cells of a block."""
    sg = _setup()['sigma']
    cells = np.array(decode(code)).reshape(2, 2, 2)
    big = np.empty((4, 4, 4), dtype=np.int64)
    for b in range(8):
        big[(b >> 2) & 1::2, (b >> 1) & 1::2, b & 1::2] = sg[cells, b]
    return big


def child_windows(code):
    """The 27 windows of size 2x2x2 of sigma(w)."""
    big = sigma_cube(code)
    return [encode(tuple(int(big[x + dx, y + dy, z + dz]) for dx, dy, dz in POS))
            for x in range(3) for y in range(3) for z in range(3)]


def phi(code):
    """The central window of sigma(w)."""
    big = sigma_cube(code)
    return encode(tuple(int(big[1 + dx, 1 + dy, 1 + dz]) for dx, dy, dz in POS))


def act(g, code):
    """Rotate a block about its central vertex by the proper rotation frame g."""
    st = _setup()
    new = [None] * 8
    for pos, lab in zip(POS, decode(code)):
        f, k = st['labels'][lab]
        x = S.frame_apply(g, tuple(p - 1 for p in pos))
        newpos = tuple(x[i] - (1 if g[1][i] < 0 else 0) + 1 for i in range(3))
        assert all(v in (0, 1) for v in newpos)
        new[POS.index(newpos)] = st['index'][(S.frame_mul(g, f), k)]
    return encode(new)


def stabiliser_order(code):
    return sum(1 for g in _setup()['rots'] if act(g, code) == code)


def report():
    blocks, per_depth = legal_blocks()
    bset = set(blocks)
    n_depth1 = set(_window_codes(_inflate(np.arange(_setup()['n'], dtype=np.int64).reshape(-1, 1, 1, 1))).tolist())
    closed = all(c in bset for b in blocks for c in child_windows(b))
    phi_img = {b: phi(b) for b in blocks}
    periodic = []
    for b in blocks:
        y = b
        for per in range(1, 64):
            y = phi_img[y]
            if y == b:
                periodic.append((b, per))
                break
    fixed = [b for b, p in periodic if p == 1]
    rot_closed = all(act(g, b) in bset for g in _setup()['rots'] for b in blocks)
    return dict(labels=_setup()['n'], legal=len(blocks), per_depth=per_depth, closed=closed,
                contains_depth_one=n_depth1 <= bset, rotation_closed=rot_closed,
                periodic=len(periodic), periods=dict(Counter(p for _, p in periodic)), fixed=len(fixed),
                stabilisers=dict(sorted(Counter(stabiliser_order(b) for b in blocks).items())),
                stabilisers_fixed=dict(sorted(Counter(stabiliser_order(b) for b in fixed).items())))


def run_tests(verbose=True):
    r = report()
    ok = (r['labels'] == 168 and r['legal'] == 1410 and r['per_depth'] == [168, 600, 1278, 1398, 1410, 1410]
          and r['closed'] and r['contains_depth_one'] and r['rotation_closed']
          and r['periodic'] == r['fixed'] == 27 and r['periods'] == {1: 27}
          and r['stabilisers_fixed'] == {1: 24, 8: 3}
          and r['stabilisers'] == {1: 1296, 2: 96, 4: 12, 8: 6})
    if verbose:
        print('chair44_hull:', 'ok' if ok else 'FAIL', r)
    return ok


def main():
    if '--tests' in sys.argv:
        sys.exit(0 if run_tests() else 1)
    r = report()
    print('labels %d; windows of sigma^n(j), n=1..6: %s' % (r['labels'], r['per_depth']))
    print('legal blocks: %d; closed under the 27 child windows: %s; contains the depth-one windows: %s'
          % (r['legal'], r['closed'], r['contains_depth_one']))
    print('closed under the 24 rotations: %s' % r['rotation_closed'])
    print('periodic points of Phi: %d, periods %s; fixed seeds: %d' % (r['periodic'], r['periods'], r['fixed']))
    print('stabiliser orders over the legal blocks: %s' % r['stabilisers'])
    print('stabiliser orders over the fixed seeds:  %s' % r['stabilisers_fixed'])


if __name__ == '__main__':
    main()
