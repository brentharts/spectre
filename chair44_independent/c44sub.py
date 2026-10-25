"""Label substitution on 168 = 24 x 7 cell labels, legal 2x2x2 blocks (independent)."""
import itertools
from c44core import *

E8 = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
LABELS = [(r, k) for r in range(24) for k in range(7)]
LIDX = {l: i for i, l in enumerate(LABELS)}

def anchor(A, k):
    """translation t with cell k of pose (A,t) at the origin cell"""
    return neg(cell_image(A, (0, 0, 0), CELLS[k]))

def label_pose(lab, z=(0, 0, 0)):
    r, k = LABELS[lab]
    A = ROT24[r]
    return (A, add(anchor(A, k), z))

def pose_labels(A, t):
    """dict cell -> label id for a proper pose"""
    r = RIDX[A]
    return {cell_image(A, t, c): LIDX[(r, k)] for k, c in enumerate(CELLS)}

def sigma_label(lab):
    A, t = label_pose(lab)
    out = {}
    for P in children(A, t):
        for cell, l in pose_labels(*P).items():
            out[cell] = l
    blk = tuple(out[e] for e in E8)
    assert set(out[e] for e in E8) and all(e in out for e in E8)
    return blk

SIG = [sigma_label(l) for l in range(168)]

def sigma_cfg(cfg):
    """cfg: dict cell->label; returns inflated dict"""
    out = {}
    for z, l in cfg.items():
        for e, m in zip(E8, SIG[l]):
            out[add(scal(2, z), e)] = m
    return out

def block_cfg(b, z=(0, 0, 0)):
    return {add(z, e): l for e, l in zip(E8, b)}

def windows(cfg, size):
    """all 2x2x2 windows of a cfg whose support is [0,size)^3"""
    out = set()
    for z in itertools.product(range(size - 1), repeat=3):
        out.add(tuple(cfg[add(z, e)] for e in E8))
    return out

def sigma_windows(b):
    return windows(sigma_cfg(block_cfg(b)), 4)

def legal_closure():
    L = set(SIG)
    sizes = [len(L)]
    frontier = set(L)
    while frontier:
        new = set()
        for b in frontier:
            new |= sigma_windows(b)
        frontier = new - L
        L |= new
        sizes.append(len(L))
    return L, sizes
