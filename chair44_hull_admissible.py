#!/usr/bin/env python3
r"""chair44_hull_admissible.py - the finite part of the Chair44 hull theorem, from our own atlas.

chair44_hull.py recomputes the LEGAL vertex blocks from symchair44. This module completes the second route: it
counts the ADMISSIBLE blocks with the contact atlas that symchair44 derives from its own labels, and checks the
lemma of Section 7.4. Nothing is read from Tsiokos's repository.

A block is the labelling of the 2x2x2 unit cells around a lattice vertex (cells with lower corners in {-1,0}^3),
a label being (frame, native cell); 168 labels. A block is ADMISSIBLE if
  - the chairs its labels name are consistent: a chair covering a block cell is the chair named there;
  - those chairs are pairwise disjoint;
  - every face-adjacent pair has its relative pose among the 44 exact-complement contacts (symchair44.atlas).
By registration (Tsiokos, Thm 5.6 and Lemma 5.3), every block of a tiling by Q is admissible.

What is computed (exact integers):
  1. the admissible blocks: 360,189, containing all 1410 legal blocks;
  2. the lemma: every one of the 27 windows of size 2x2x2 of sigma(b), for b admissible, is legal.
With the parent tiling (Tsiokos, Thms 6.4-6.5) this gives Theorem 1: every patch of a tiling T lies in sigma^n of a
block of D^n(T), that block is a window of sigma of an admissible block of D^{n+1}(T), hence legal.

    python3 chair44_hull_admissible.py           report (about 10 min)
    python3 chair44_hull_admissible.py --tests   check the numbers above
"""
import itertools
import sys

import chair44_hull as H
import symchair44 as S

CELL_OF_POS = [tuple(p - 1 for p in pos) for pos in H.POS]   # lower corners of the block cells


def pose_of(cell, label):
    """The chair pose (frame, t) whose native cell number k is the unit cell with lower corner `cell`."""
    f, k = H._setup()['labels'][label]
    x = S.frame_apply(f, S.CELLS[k])
    t = tuple(cell[i] - x[i] + (1 if f[1][i] < 0 else 0) for i in range(3))
    assert cell in S.pose_cells((f, t))
    return (f, t)


_CACHE = {}


def _cells(pose):
    if pose not in _CACHE:
        _CACHE[pose] = (S.pose_cells(pose), S.exposed_faces(pose))
    return _CACHE[pose]


def legal_contact_set():
    if 'legal_contacts' not in H._STATE:
        A = S.atlas()
        H._STATE['legal_contacts'] = set(A['legal'])
    return H._STATE['legal_contacts']


def adjacent(p, q):
    fp, fq = _cells(p)[1], _cells(q)[1]
    return any(k in fq and fq[k] == -sd for k, sd in fp.items())


def compatible(p, q):
    if p == q:
        return True
    if _cells(p)[0] & _cells(q)[0]:
        return False
    return not adjacent(p, q) or S.relative_pose(p, q) in legal_contact_set()


def admissible_blocks():
    n = H._setup()['n']
    out = []
    poses = [None] * 8
    chosen = []

    def rec(k):
        if k == 8:
            out.append(H.encode(chosen))
            return
        cell = CELL_OF_POS[k]
        for lab in range(n):
            p = pose_of(cell, lab)
            cells = _cells(p)[0]
            ok = True
            for j in range(k):
                q = poses[j]
                if q != p and (CELL_OF_POS[j] in cells or cell in _cells(q)[0] or not compatible(p, q)):
                    ok = False
                    break
            if ok:
                poses[k] = p
                chosen.append(lab)
                rec(k + 1)
                chosen.pop()
        poses[k] = None

    rec(0)
    return out


def report():
    legal = set(H.legal_blocks()[0])
    adm = admissible_blocks()
    adm_set = set(adm)
    windows = set()
    for b in adm:
        windows.update(H.child_windows(b))
    return dict(contacts=len(legal_contact_set()), admissible=len(adm_set), legal=len(legal),
                legal_inside=legal <= adm_set, windows=len(windows), windows_legal=windows <= legal,
                fixpoint_is_legal=(adm_set & windows) == legal)


def run_tests(verbose=True):
    r = report()
    ok = (r['contacts'] == 44 and r['admissible'] == 360189 and r['legal'] == 1410 and r['legal_inside']
          and r['windows_legal'] and r['fixpoint_is_legal'])
    if verbose:
        print('chair44_hull_admissible:', 'ok' if ok else 'FAIL', r)
    return ok


def main():
    if '--tests' in sys.argv:
        sys.exit(0 if run_tests() else 1)
    r = report()
    print('legal contacts (own atlas): %d' % r['contacts'])
    print('admissible blocks: %d; all %d legal blocks among them: %s' % (r['admissible'], r['legal'], r['legal_inside']))
    print('distinct windows of sigma(admissible): %d; all legal: %s' % (r['windows'], r['windows_legal']))
    print('admissible blocks that are windows of sigma(admissible) = legal blocks: %s' % r['fixpoint_is_legal'])


if __name__ == '__main__':
    main()
