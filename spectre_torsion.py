#!/usr/bin/python3
"""The torsion dictionary of the cusp torus.

The torus of spectre_mystic_fibres.py is T = R^2 / L with
L = Z E1 + Z E2, E1 = (3, sqrt3), E2 = (0, 2 sqrt3).  The two generators have
the same length 2 sqrt3 and meet at 60 degrees, so L is a hexagonal lattice
and T is the complex torus C / Z[zeta_3], the elliptic curve with j = 0.  The
rotation by 60 degrees is multiplication by -zeta_3^2, a unit of Z[zeta_3] of
order 6, and in the basis (E1, E2) it is the INTEGER matrix

        R = [[0, -1],
             [1,  1]]            rho E1 = E2,   rho E2 = E2 - E1.

Everything in this file about R and about the six site classes is exact
(fractions and integers).  The data enter only through the Mystic census of
spectre_mystic_fibres.py, which is double precision with asserted lattice
checks.

What it shows
-------------
1. The six site classes of the lattice modulo L are torsion points of T:
       hexagon centre      = 0
       edge midpoints (D_a, D_c, D_b) = the three nonzero points of E[2]
       hexagon vertices    = the two nonzero points of E[sqrt(-3)] = E[1-zeta_3]
2. R permutes the three points of E[2]\\0 in a 3-cycle, swaps the two vertex
   classes, fixes 0, has R^6 = 1 and R^3 = -1, and -1 fixes every point of E[2].
3. Hence Z/6 = <R> = Z/3 x Z/2 (by CRT) acts on the six Mystic classes, which
   are the pairs (point of E[2]\\0, sign) = (first vertex, first-edge angle),
   simply transitively: the group of units of Z[zeta_3] is the group of the
   six Mystic classes.

    python3 spectre_torsion.py            # checks + figure
    python3 spectre_torsion.py --check    # checks only
"""
import os
import sys
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectre_mystic_fibres as MF        # noqa: E402

R = ((0, -1), (1, 1))
I2 = ((1, 0), (0, 1))
NEG = ((-1, 0), (0, -1))
FIGDIR = os.path.join(HERE, "paper", "figures")


# --------------------------------------------------------------------------
# exact part
# --------------------------------------------------------------------------
def mat_mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2))
                 for i in range(2))


def mat_pow(A, n):
    out = I2
    for _ in range(n):
        out = mat_mul(out, A)
    return out


def act(A, p):
    """A (integer matrix) on a point of R^2/Z^2 given by Fractions."""
    q = tuple(sum(A[i][k] * p[k] for k in range(2)) for i in range(2))
    return tuple(c - (c.numerator // c.denominator) for c in q)


E2_PTS = [(Fr(1, 2), Fr(0)), (Fr(0), Fr(1, 2)), (Fr(1, 2), Fr(1, 2))]
VERT_PTS = [(Fr(1, 3), Fr(1, 3)), (Fr(2, 3), Fr(2, 3))]
ZERO = (Fr(0), Fr(0))
NAMES = {E2_PTS[0]: "D_a", E2_PTS[1]: "D_c", E2_PTS[2]: "D_b"}


def torsion_orbit(p, n=6):
    out = [p]
    for _ in range(n):
        out.append(act(R, out[-1]))
    return out


def exact_checks():
    # R is the 60-degree rotation: order 6, R^3 = -1, char poly x^2 - x + 1
    assert mat_pow(R, 6) == I2 and mat_pow(R, 3) == NEG
    assert mat_pow(R, 2) != I2 and mat_pow(R, 3) != I2
    tr = R[0][0] + R[1][1]
    det = R[0][0] * R[1][1] - R[0][1] * R[1][0]
    assert (tr, det) == (1, 1)                    # Phi_6 = x^2 - x + 1
    # the lattice is hexagonal: R maps the basis to itself up to the lattice
    # (integer entries), and E1, E2 have equal length and meet at 60 degrees
    # (checked in floating point below)
    # edge midpoints = E[2] minus 0: a 3-cycle under R, fixed by -1
    assert set(E2_PTS) == {p for p in (
        (Fr(a, 2), Fr(b, 2)) for a in (0, 1) for b in (0, 1)) if p != ZERO}
    cyc = torsion_orbit(E2_PTS[0], 3)
    assert cyc[0] == E2_PTS[0] and cyc[3] == cyc[0] and len(set(cyc[:3])) == 3
    assert set(cyc[:3]) == set(E2_PTS)
    assert all(act(NEG, p) == p for p in E2_PTS)
    assert all(act(mat_pow(R, 3), p) == p for p in E2_PTS)
    # vertex classes = E[1 - zeta_3] minus 0: swapped by R, fixed by R^2
    assert act(R, VERT_PTS[0]) == VERT_PTS[1] and act(R, VERT_PTS[1]) == VERT_PTS[0]
    assert all(act(mat_pow(R, 2), p) == p for p in VERT_PTS)
    # (1 - zeta_3) kills them: the matrix 1 - R has determinant 3
    z3 = mat_pow(R, 2)                            # zeta_3 = R^2 (order 3)
    assert mat_pow(z3, 3) == I2 and z3 != I2
    one_minus = ((1 - z3[0][0], -z3[0][1]), (-z3[1][0], 1 - z3[1][1]))
    assert one_minus[0][0] * one_minus[1][1] - one_minus[0][1] * one_minus[1][0] == 3
    assert all(act(one_minus, p) == ZERO for p in VERT_PTS)
    # and the six classes are exactly the points killed by neither more nor less
    assert act(R, ZERO) == ZERO
    return True


def float_checks():
    """R is the 60-degree rotation in the basis (E1, E2), and the lattice is
    hexagonal; the six site classes of MF are the exact torsion points."""
    E1, E2 = MF.E1, MF.E2
    B = np.array([E1, E2]).T
    th = np.pi / 3
    Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Rm = np.round(np.linalg.inv(B) @ Rot @ B, 9)
    assert np.allclose(Rm, np.array(R, float)), Rm
    assert abs(np.hypot(*E1) - np.hypot(*E2)) < 1e-12
    assert abs(np.dot(E1, E2) / np.dot(E1, E1) - 0.5) < 1e-12
    sites = {k: v for k, v in MF.SITE_UV.items()}
    want = {(0.0, 0.0): "centre", (0.5, 0.0): "edge", (0.0, 0.5): "edge",
            (0.5, 0.5): "edge", (round(1 / 3, 6), round(1 / 3, 6)): "vertex",
            (round(2 / 3, 6), round(2 / 3, 6)): "vertex"}
    assert {tuple(float(x) for x in k): v for k, v in sites.items()} == want
    return True


def class_group_action():
    """The map (angle 30 + 60 k, anchor) -> (angle + 60, R anchor) is a
    6-cycle on the six labels; its CRT split is (anchor cycle) x (sign)."""
    ang = [30 + 60 * k for k in range(6)]
    anchor0 = E2_PTS[0]                           # D_a carries 30 and 210
    lab = []
    p = anchor0
    for k in range(6):
        lab.append((ang[k], NAMES[p]))
        p = act(R, p)
    assert p == anchor0                           # returns after 6 steps
    assert [x[1] for x in lab] == ["D_a", "D_c", "D_b", "D_a", "D_c", "D_b"]
    # the anchor depends on k mod 3, the sign (angle mod 180) on k mod 2
    assert all(lab[k][1] == lab[k % 3][1] for k in range(6))
    assert len({(lab[k][1], (k % 2)) for k in range(6)}) == 6   # CRT: injective
    return lab


def selftest(depths=(3, 4)):
    exact_checks()
    float_checks()
    lab = class_group_action()
    # data: every one of the six labels occurs, and it is exactly this set
    for n in depths:
        r = MF.census(n, with_env=False)
        got = {(g[0]["angle"], g[0]["anchor"]) for g in r["classes"]}
        assert got == set(lab), (n, got)
    return True


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def figure(fname="spectre_torsion"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, FancyArrowPatch

    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
    PINK, GOLD = "#d4569c", "#c9a227"
    INK, GREY = "#222222", "#bdbab2"
    PAL = {30: BLUE, 90: ORANGE, 150: AQUA, 210: VIOLET, 270: PINK, 330: GOLD}
    CUR = {"D_a": BLUE, "D_c": ORANGE, "D_b": AQUA}

    lab = class_group_action()
    r5 = MF.census(5, with_env=False)
    cnt = {(g[0]["angle"], g[0]["anchor"]): len(g) for g in r5["classes"]}

    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 0.9], wspace=0.26)

    # (a) the torus as a hexagonal cell, sites labelled by torsion
    ax = fig.add_subplot(gs[0])
    E1, E2 = MF.E1, MF.E2
    for i in range(-1, 2):
        for j in range(-1, 2):
            o = i * E1 + j * E2
            P = np.array([o, o + E1, o + E1 + E2, o + E2])
            ax.add_patch(Polygon(P, closed=True, fc="none",
                                 ec=GREY if (i, j) != (0, 0) else INK,
                                 lw=0.7 if (i, j) != (0, 0) else 1.6,
                                 ls=(0, (4, 3)) if (i, j) != (0, 0) else "-"))
    def at(p):
        return p[0] * E1 + p[1] * E2
    pts = [(ZERO, "0", INK, (-14, -4)), (E2_PTS[0], "$D_a$", CUR["D_a"], (8, -12)),
           (E2_PTS[1], "$D_c$", CUR["D_c"], (-22, 2)), (E2_PTS[2], "$D_b$", CUR["D_b"], (8, 4)),
           (VERT_PTS[0], "$V_1$", VIOLET, (6, -13)), (VERT_PTS[1], "$V_2$", VIOLET, (6, 4))]
    for p, name, col, off in pts:
        xy = at(p)
        ax.scatter([xy[0]], [xy[1]], s=70, color=col, zorder=5,
                   edgecolors="white", linewidths=1.0)
        ax.annotate(name, xy, xytext=off, textcoords="offset points",
                    fontsize=10, color=col, zorder=6)
    for p, q in ((E2_PTS[0], E2_PTS[1]), (E2_PTS[1], E2_PTS[2]), (E2_PTS[2], E2_PTS[0])):
        pa, pb = at(p), at(q)
        ax.add_patch(FancyArrowPatch(pa, pb, connectionstyle="arc3,rad=-0.25",
                                     arrowstyle="-|>", mutation_scale=11,
                                     color="#777777", lw=1.0, zorder=3))
    ax.set_xlim(-1.3, 4.6)
    ax.set_ylim(-0.9, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) $T=\\mathbb{C}/\\mathbb{Z}[\\zeta_3]$: sites are torsion points\n"
                 "$E[2]\\setminus0=\\{D_a,D_c,D_b\\}$, "
                 "$E[\\sqrt{-3}]\\setminus0=\\{V_1,V_2\\}$", fontsize=10)

    # (b) the six Mystic classes on a hexagon, R = rotation by 60 degrees
    ax = fig.add_subplot(gs[1])
    for k, (a, c) in enumerate(lab):
        t = np.radians(90 - 60 * k)
        x, y = np.cos(t), np.sin(t)
        ax.scatter([x], [y], s=900, color=PAL[a], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(x, y, "%d°\n$%s$" % (a, c), ha="center", va="center",
                color="white", fontsize=9, zorder=4)
        t2 = np.radians(90 - 60 * (k + 1))
        x2, y2 = np.cos(t2), np.sin(t2)
        v = np.array([x2 - x, y2 - y])
        u0 = v / np.hypot(*v)
        ax.add_patch(FancyArrowPatch((x + 0.30 * u0[0], y + 0.30 * u0[1]),
                                     (x2 - 0.30 * u0[0], y2 - 0.30 * u0[1]),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#555555", lw=1.2, zorder=2))
    ax.plot([0, 0], [-0.15, 0.15], color="white")
    ax.text(0, 0, "$\\rho=R$\n$\\mathbb{Z}/6$", ha="center", va="center",
            fontsize=10, color=INK)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) the six Mystic classes form one orbit of $R$", fontsize=10)

    # (c) CRT table with level-5 counts
    ax = fig.add_subplot(gs[2])
    rows = ["D_a", "D_c", "D_b"]
    cols = [("+", 0), ("−", 180)]
    grid = np.zeros((3, 2))
    for (a, c), n in cnt.items():
        i = rows.index(c)
        j = 0 if a in (30, 90, 150) else 1
        grid[i, j] = n
    mean = grid.mean()
    ax.imshow(grid / mean, cmap="Blues", vmin=0.8, vmax=1.35)
    for i in range(3):
        for j in range(2):
            ax.text(j, i, "%d" % grid[i, j], ha="center", va="center", fontsize=12,
                    color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sign $+$\n($30,90,150$)", "sign $-$\n($210,270,330$)"],
                       fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["$%s$" % r_ for r_ in rows])
    ax.set_title("(c) $\\mathbb{Z}/3\\times\\mathbb{Z}/2$: Mystics at level 5", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, fname + "." + ext), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figures/%s.{pdf,png}" % fname)


if __name__ == "__main__":
    print("selftest:", selftest())
    if "--check" in sys.argv:
        sys.exit(0)
    for k, (a, c) in enumerate(class_group_action()):
        print("  R^%d: %3d deg anchored on %s" % (k, a, c))
    figure()
