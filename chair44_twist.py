#!/usr/bin/env python3
"""chair44_twist.py - the inflation geometry ("twist") of the Chair44 supertile.

Spectre (Fact 11): the supertile quadrilateral is the tile quadrilateral scaled by
sqrt(4+sqrt15) and turned by +-theta_S = 11.565 deg, with a mirror at every level.
Chair44 is the opposite case, and exactly so:

  1. the union of the 8 child chairs IS the doubled chair 2*chair, cell for cell:
     scale 2 (rational), net rotation identity, no mirror.  Twist = 0.
  2. the 8 child frames H_i are proper rotations, but they scramble orientation:
     A = (1/8) sum_i H_i = I/4 exactly (3x3), so the mean frame of the 8^n tiles
     of a level-n supertile is I/4^n  (this is the (1/4)^d rate of the 24x24 matrix).
  3. the rotation angles of the children (0, 90, 120, 180 deg) are rational multiples
     of pi: no irrational twist angle can arise from this substitution.

CLI: python3 chair44_twist.py [--verify]
"""
import itertools
import sys
from fractions import Fraction as Fr

import chair44 as C


def matrix(frame):
    p, s = frame
    m = [[0] * 3 for _ in range(3)]
    for i in range(3):
        m[i][p[i]] = s[i]
    return m


def mat_add(a, b):
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def doubled_chair_cells():
    return frozenset(tuple(2 * a + b for a, b in zip(c, o))
                     for c in C.CHAIR_CELLS for o in itertools.product((0, 1), repeat=3))


def children_union():
    out = set()
    for pose in C.CHILDREN:
        c = C.pose_cells(pose)
        assert not (out & c), "children overlap"
        out |= c
    return frozenset(out)


def supertile_is_scaled_chair():
    """True iff the 8 children tile exactly 2*chair (scale 2, rotation 1, det +1)."""
    return children_union() == doubled_chair_cells()


def rotation_angle_deg(frame):
    """Rotation angle (degrees) of a proper frame from trace = 1 + 2 cos(theta)."""
    import math
    m = matrix(frame)
    tr = sum(m[i][i] for i in range(3))
    assert C.frame_det(frame) == 1
    return round(math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1) / 2)))), 9)


def child_angles():
    return sorted(rotation_angle_deg(H) for H, _ in C.CHILDREN)


def mean_child_matrix():
    S = [[0] * 3 for _ in range(3)]
    for H, _ in C.CHILDREN:
        S = mat_add(S, matrix(H))
    return [[Fr(x, 8) for x in row] for row in S]


def mean_frame(depth):
    """(1/8^depth) sum of the 3x3 matrices of all tiles of sigma_depth, exactly."""
    S = [[0] * 3 for _ in range(3)]
    for G, _ in C.sigma_depth(depth):
        S = mat_add(S, matrix(G))
    n = 8 ** depth
    return [[Fr(x, n) for x in row] for row in S]


def holonomy_group_order():
    return len({G for G, _ in C.sigma_depth(2)})


def verify():
    ok = True

    def chk(name, c):
        nonlocal ok
        print(("ok   " if c else "FAIL ") + name)
        ok &= bool(c)
    chk("8 children tile exactly the doubled chair (scale 2, twist 0)", supertile_is_scaled_chair())
    chk("all child frames proper (det +1): no mirror at any level",
        all(C.frame_det(H) == 1 for H, _ in C.CHILDREN))
    chk("child rotation angles %s deg, all rational multiples of pi" % child_angles(),
        set(child_angles()) <= {0.0, 90.0, 120.0, 180.0})
    I4 = [[Fr(1, 4) if i == j else Fr(0) for j in range(3)] for i in range(3)]
    chk("mean child matrix = I/4", mean_child_matrix() == I4)
    for d in (1, 2, 3):
        chk("mean frame at depth %d = I/4^%d" % (d, d),
            mean_frame(d) == [[Fr(1, 4 ** d) if i == j else Fr(0) for j in range(3)] for i in range(3)])
    chk("frames at depth 2 form all 24 rotations", holonomy_group_order() == 24)
    return ok


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
