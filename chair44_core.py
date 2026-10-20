#!/usr/bin/env python3
"""chair44_core.py - minimal exact geometry of Tsiokos' Chair44 (R44) einstein tile.

Self-contained (no data files): the seven-cube chair (2x2x2 minus one corner), the 48 cube
frames, poses x -> g x + t, and the 8-child substitution (supertile = 8 rotated chairs
filling the doubled chair).  Everything is exact integer arithmetic.

Credit: the shape and the substitution children are from Ioannis Tsiokos,
https://github.com/ioannist/six-birds-tiles (CC BY 4.0), arXiv:2609.19214 and arXiv:2609.23783.
This module re-implements the geometry independently; only the eight child poses below are
taken from that data.

Conventions: frame = (perm, signs), (g x)_i = signs[i] * x[perm[i]];  pose = (frame, t);
child poses refine as (G, t) -> (G H, 2 t + G u).
"""
import itertools

CHAIR_CELLS = frozenset(c for c in itertools.product((0, 1), repeat=3) if c != (1, 1, 1))
IDENTITY_FRAME = ((0, 1, 2), (1, 1, 1))
ROOT_POSE = (IDENTITY_FRAME, (0, 0, 0))

# (frame H, offset u): the eight children of the chair (Tsiokos, r44_atlas.json "children")
CHILDREN = (
    (((0, 1, 2), (1, 1, 1)), (0, 0, 0)),
    (((1, 0, 2), (1, 1, -1)), (0, 0, 4)),
    (((0, 2, 1), (1, -1, 1)), (0, 4, 0)),
    (((2, 0, 1), (1, -1, -1)), (0, 4, 4)),
    (((2, 1, 0), (-1, 1, 1)), (4, 0, 0)),
    (((1, 2, 0), (-1, 1, -1)), (4, 0, 4)),
    (((0, 1, 2), (-1, -1, 1)), (4, 4, 0)),
    (((0, 1, 2), (1, 1, 1)), (1, 1, 1)),
)


def frame_apply(f, v):
    p, s = f
    return tuple(s[i] * v[p[i]] for i in range(3))


def frame_mul(a, b):
    """frame of a(b(x))."""
    p, s = a
    q, t = b
    return (tuple(q[p[i]] for i in range(3)), tuple(s[i] * t[p[i]] for i in range(3)))


def frame_det(f):
    p, s = f
    inv = sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
    return (-1 if inv % 2 else 1) * s[0] * s[1] * s[2]


def vadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def pose_cells(pose):
    """Unit cells (lower corners) occupied by the chair in this pose."""
    f, t = pose
    out = []
    for c in CHAIR_CELLS:
        x = frame_apply(f, c)
        out.append(tuple(x[i] + t[i] - (1 if f[1][i] < 0 else 0) for i in range(3)))
    return frozenset(out)


def refine_pose(pose):
    G, t = pose
    return tuple((frame_mul(G, H), vadd(tuple(2 * x for x in t), frame_apply(G, u)))
                 for H, u in CHILDREN)


def sigma_depth(depth):
    """Poses of the 8**depth chairs of the depth-th supertile (depth 0 = one chair)."""
    poses = (ROOT_POSE,)
    for _ in range(depth):
        poses = tuple(c for p in poses for c in refine_pose(p))
    return poses


def patch_cells(depth):
    """{cell: tile index} of the depth-th supertile on the lattice of scale 2**depth."""
    owner = {}
    for i, pose in enumerate(sigma_depth(depth)):
        for c in pose_cells(pose):
            assert c not in owner, "overlap"
            owner[c] = i
    return owner


def exposed_faces(cells):
    """Unit boundary faces of a cell set: {(axis, sign, cell)} with cell the owning cell."""
    cells = set(cells)
    out = []
    for c in sorted(cells):
        for ax in range(3):
            for sd in (-1, 1):
                n = list(c)
                n[ax] += sd
                if tuple(n) not in cells:
                    out.append((ax, sd, c))
    return out


def self_check():
    assert len(CHAIR_CELLS) == 7
    assert all(frame_det(H) == 1 for H, _ in CHILDREN)
    dbl = frozenset(tuple(2 * a + b for a, b in zip(c, o))
                    for c in CHAIR_CELLS for o in itertools.product((0, 1), repeat=3))
    assert len(patch_cells(1)) == 56 and frozenset(patch_cells(1)) == dbl
    assert len(exposed_faces(CHAIR_CELLS)) == 24
    return True


if __name__ == "__main__":
    print("chair44_core self-check:", self_check())
