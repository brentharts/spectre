"""Admissible 2x2x2 blocks and the window lemma (independent)."""
import itertools, sys, pickle
from c44sub import *
from c44facet import touching, rel

def build_compat(atlas):
    atlas = set(atlas)
    DZ = list(itertools.product((-1, 0, 1), repeat=3))
    poses = [label_pose(l) for l in range(168)]
    cells = [set(pose_cells(*p)) for p in poses]
    comp = {}
    for dz in DZ:
        M = [[False] * 168 for _ in range(168)]
        for l1 in range(168):
            P1 = poses[l1]; c1 = cells[l1]
            for l2 in range(168):
                A2, t2 = poses[l2]; P2 = (A2, add(t2, dz))
                if P1 == P2: M[l1][l2] = True; continue
                c2 = {add(c, dz) for c in cells[l2]}
                if c1 & c2: continue
                if touching(P1, P2) and rel(P1, P2) not in atlas: continue
                M[l1][l2] = True
        comp[dz] = M
    return comp

def admissible_blocks(comp):
    out = []
    blk = [None] * 8
    def rec(i):
        if i == 8:
            out.append(tuple(blk)); return
        for l in range(168):
            ok = True
            for j in range(i):
                if not comp[sub(E8[i], E8[j])][blk[j]][l]: ok = False; break
            if ok:
                blk[i] = l; rec(i + 1)
    rec(0)
    return out
