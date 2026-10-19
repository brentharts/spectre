#!/usr/bin/python3
"""Figures for spectre_substrate.tex (the paper had none).

Three figures, each wired to numbers that spectrefacts.py already proves, so a
figure cannot drift from the text:

  spectre_family    the Tile(a,b) family: Hat / Spectre / Turtle outlines, the
                    area form A(a,b), the chirality number line
  spectre_census    the 9x9 substitution matrix and its spectrum, the census
                    (matrix vs geometric), the species frequencies
  spectre_boundary  supertile outlines for k = 0..3 and the perimeter law
                    P(k) with nu = phi^3 against lambda^2

Self-contained apart from spectre.py and spectrefacts.py in this repository.
Output goes to paper/figures/ as PDF and PNG.

    python3 spectre_figures.py            # all three
    python3 spectre_figures.py --check    # assertions only, no drawing
"""
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectre as S                       # noqa: E402
import spectrefacts as F                  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, GREY = "#222222", "#bdbab2"
TINT_B, TINT_O = "#d6e4f7", "#fbdcd0"
S3 = 3 ** 0.5
FIGDIR = os.path.join(HERE, "paper", "figures")


def _save(fig, name):
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, "%s.%s" % (name, ext)), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figures/%s.{pdf,png}" % name)


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def tile_polygon(a, b):
    return S.get_spectre_points(a, b)


def area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def level_tiles(n):
    tiles = S.buildSpectreTiles(n, 1.0, 1.0, rotation=30)
    placed = []
    tiles["Delta"].forEachTile(lambda T, l: placed.append((T.copy(), l)))
    pts = S.get_spectre_points(1.0, 1.0)
    return [(pts @ T[:, :2].T + T[:, 2], l) for T, l in placed]


def boundary_edges(polys):
    c = Counter()
    for P, _ in polys:
        k = [tuple(np.round(p, 4)) for p in P]
        for i in range(len(P)):
            c[frozenset((k[i], k[(i + 1) % len(P)]))] += 1
    return [tuple(e) for e, v in c.items() if v == 1]


def area_form(a, b):
    return 2 * S3 * a * a + 3 * a * b + S3 * b * b


def selftest():
    assert abs(area(tile_polygon(1, S3)) - 8 * S3) < 1e-9
    assert abs(area(tile_polygon(S3, 1)) - 10 * S3) < 1e-9
    for a, b in ((1, 1), (1, S3), (S3, 1), (1.3, 0.7)):
        assert abs(area(tile_polygon(a, b)) - area_form(a, b)) < 1e-9
    for k, p in enumerate(F.BOUNDARY_SEQ[:5]):
        assert len(boundary_edges(level_tiles(k))) == p, (k, p)
    assert abs(float(F.COS_CHIRAL) - (float(F.PHI ** 4) - 3) / 4) < 1e-12
    assert float(F.SPECTRE_NAIVE) > 1 > float(F.COS_CHIRAL)
    return True


# --------------------------------------------------------------------------
# figure 1: the family
# --------------------------------------------------------------------------
def fig_family():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig = plt.figure(figsize=(13.0, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1.15], wspace=0.28)

    # (a) three outlines at equal scale
    ax = fig.add_subplot(gs[0])
    shapes = (("Hat", 1, S3, BLUE, TINT_B), ("Spectre", 1, 1, INK, "#eeeeee"),
              ("Turtle (Mystic)", S3, 1, ORANGE, TINT_O))
    x0 = 0.0
    for name, a, b, ec, fc in shapes:
        P = tile_polygon(a, b)
        P = P - P.min(0)
        ax.add_patch(Polygon(P + [x0, 0], closed=True, fc=fc, ec=ec, lw=1.8))
        ax.text(x0 + P[:, 0].max() / 2, -0.9, name, ha="center", fontsize=9)
        ax.text(x0 + P[:, 0].max() / 2, -1.7,
                "Tile$(%s,%s)$\n$A=%s$" % (
                    "1" if a == 1 else r"\sqrt{3}", "1" if b == 1 else r"\sqrt{3}",
                    {"Hat": r"8\sqrt{3}", "Spectre": r"3+3\sqrt{3}",
                     "Turtle (Mystic)": r"10\sqrt{3}"}[name]),
                ha="center", va="top", fontsize=8, color="#555555")
        x0 += P[:, 0].max() + 1.2
    ax.set_xlim(-0.5, x0)
    ax.set_ylim(-3.8, 7.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) one family, three tiles", fontsize=10)

    # (b) the area form
    ax = fig.add_subplot(gs[1])
    r = np.linspace(0.3, 2.0, 200)
    ax.plot(r, 2 * S3 + 3 * r + S3 * r ** 2, color=BLUE, lw=1.8,
            label="$A(1,r)$")
    ax.plot(r, 2 * S3 * r ** 2 + 3 * r + S3, color=ORANGE, lw=1.8, ls=(0, (5, 2.5)),
            label="$A(r,1)$")
    for rr, nm in ((1 / S3, "Turtle"), (1.0, "Spectre"), (S3, "Hat")):
        ax.axvline(rr, color=GREY, lw=0.8, zorder=0)
        ax.text(rr, 0.98, nm, rotation=90, va="top", ha="right", fontsize=8,
                color="#555555", transform=ax.get_xaxis_transform())
    ax.set_xlabel("$r=b/a$ (first curve) or $a/b$ (second)")
    ax.set_ylabel("area")
    ax.set_title("(b) $A(a,b)=2\\sqrt{3}a^2+3ab+\\sqrt{3}b^2$", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean(ax)

    # (c) the chirality number line
    ax = fig.add_subplot(gs[2])
    kappa = float(F.PHI ** 4)
    cos_h = float(F.COS_CHIRAL)
    cos_s = float(F.SPECTRE_NAIVE)
    th = float(F.THETA_CHIRAL)
    t = np.linspace(0, np.pi, 300)
    ax.plot(np.cos(t) * 0 + np.linspace(-1, 1.5, 300), 0 * t, color=GREY, lw=0)
    ax.axhline(0, color=INK, lw=1)
    ax.axvspan(-1, 1, ymin=0.42, ymax=0.58, color="#d3eee3")
    ax.scatter([cos_h], [0], s=70, color=BLUE, zorder=5)
    ax.scatter([cos_s], [0], s=70, color=ORANGE, zorder=5)
    cos_t = float(F.TWIST_COS)
    th_s = float(F.TWIST_FAMILY[0][1])
    ax.scatter([cos_t], [0], s=70, color=VIOLET, zorder=6)
    ax.annotate("Hat\n$\\cos\\theta=(\\kappa-3)/4$\n$=%.4f$ → %.2f°" % (cos_h, th),
                (cos_h, 0), (cos_h - 0.26, -0.8), ha="center", fontsize=8.5,
                color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate("Spectre, computed\n$\\theta_S=%.2f°$, $\\cos\\theta_S=%.4f$\n(supertile quadrilaterals)" % (th_s, cos_t),
                (cos_t, 0), (cos_t - 0.2, 0.62), ha="center", fontsize=8.5,
                color=VIOLET, arrowprops=dict(arrowstyle="-", color=VIOLET, lw=0.8))
    ax.annotate("Spectre, Hat formula\n$(\\lambda^2-3)/4=%.4f>1$\nnot a cosine" % cos_s,
                (cos_s, 0), (cos_s + 0.02, -0.75), ha="center", fontsize=8.5,
                color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    ax.axvline(1, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.text(1.0, 1.0, "$\\cos\\theta=1$", ha="right", fontsize=8, va="top",
            transform=ax.get_xaxis_transform())
    ax.set_xlim(0.4, 1.4)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.set_title("(c) the twist of the Hat and of the Spectre", fontsize=10)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    _save(fig, "spectre_family")


# --------------------------------------------------------------------------
# figure 2: the census
# --------------------------------------------------------------------------
def fig_census():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.array(F.M.tolist(), float)
    fig = plt.figure(figsize=(13.0, 7.2))
    gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.42)

    # (a) matrix
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=3.2)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] >= 2 else INK)
    names = [n[:2] for n in F.SPECIES]
    ax.set_xticks(range(9))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_yticks(range(9))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("(a) the substitution matrix $M$", fontsize=10)

    # (b) spectrum on a log axis
    ax = fig.add_subplot(gs[0, 1])
    lam2 = float(F.LAM2)
    phi4 = float(F.PHI ** 4)
    ev_s = [("$0^{\\times5}$", 0.0), ("$-1$", -1.0), ("$1$", 1.0),
            ("$\\lambda^{-2}$", 1 / lam2), ("$\\lambda^2$", lam2)]
    ev_h = [("$-1$", -1.0), ("$1$", 1.0), ("$\\varphi^{-4}$", 1 / phi4),
            ("$\\varphi^4$", phi4)]
    ax.axhline(0, color=GREY, lw=0.8)
    for y, ev, col, nm in ((1, ev_s, ORANGE, "$M$ (Spectre, 9 species)"),
                           (0, ev_h, BLUE, "$H$ (Hat, 4 species)")):
        for lab, v in ev:
            ax.scatter([v], [y], s=46, color=col, zorder=3)
            dy = -0.3 if lab.startswith("$0") else 0.16
            ax.text(v, y + dy, lab, ha="center", fontsize=8, color=col)
        ax.text(-1.4, y + 0.45, nm, fontsize=8.5, color=col)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_ylim(-0.6, 1.9)
    ax.set_xticks([-1, 0, 1, 10])
    ax.set_xticklabels(["$-1$", "0", "1", "10"])
    ax.set_yticks([])
    ax.set_xlim(-1.6, 25)
    ax.set_title("(b) spectra: $\\lambda^2=4+\\sqrt{15}$ vs $\\varphi^4$", fontsize=10)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)

    # (c) rank drop: ranks of M^n
    ax = fig.add_subplot(gs[0, 2])
    ns = sorted(F.RANKS)
    ax.bar([0] + ns, [9] + [F.RANKS[n] for n in ns], color=[GREY] + [BLUE] * len(ns),
           width=0.6)
    for x, v in zip([0] + ns, [9] + [F.RANKS[n] for n in ns]):
        ax.text(x, v + 0.15, str(v), ha="center", fontsize=9)
    ax.set_xticks([0] + ns)
    ax.set_xticklabels(["$M^0$"] + ["$M^%d$" % n for n in ns])
    ax.set_ylabel("rank")
    ax.set_ylim(0, 10.5)
    ax.set_title("(c) the Jordan pair: rank 5 → 4", fontsize=10)
    _clean(ax)

    # (d) census
    ax = fig.add_subplot(gs[1, 0])
    ds = np.arange(len(F.MATRIX_TOTALS))
    ax.semilogy(ds, F.GEOMETRIC_TOTALS, "o-", color=ORANGE, label="geometric (with Mystic)")
    ax.semilogy(ds, F.MATRIX_TOTALS, "s--", color=BLUE, label="matrix census")
    ax.semilogy(ds, [float(F.LAM2) ** d for d in ds], ":", color=INK,
                label="$\\lambda^{2n}$")
    ax.set_xlabel("depth $n$")
    ax.set_ylabel("tiles")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(d) census: Mystic = previous generation", fontsize=10)
    _clean(ax)

    # (e) frequencies
    ax = fig.add_subplot(gs[1, 1])
    f = [float(F.FREQ[n]) for n in F.SPECIES]
    cols = [BLUE if F.FREQ[n].is_rational else ORANGE for n in F.SPECIES]
    ax.bar(range(9), f, color=ORANGE)
    ax.set_xticks(range(9))
    ax.set_xticklabels([n[:2] for n in F.SPECIES], fontsize=8)
    ax.set_ylabel("frequency")
    ax.set_title("(e) Spectre frequencies: all in $\\mathbb{Q}(\\sqrt{15})$, irrational", fontsize=9.5)
    _clean(ax)
    ax = fig.add_subplot(gs[1, 2])
    hf = [float(F.HAT_FREQ[n]) for n in F.HAT_SPECIES]
    ax.bar(range(4), hf, color=[BLUE if F.HAT_FREQ[n].is_rational else VIOLET
                                for n in F.HAT_SPECIES])
    ax.set_xticks(range(4))
    ax.set_xticklabels(F.HAT_SPECIES)
    ax.text(0, hf[0] + 0.01, "$1/3$", ha="center", fontsize=9, color=BLUE)
    ax.set_title("(f) Hat: $f_H=1/3$ is rational", fontsize=10)
    ax.set_ylabel("frequency")
    _clean(ax)
    _save(fig, "spectre_census")


# --------------------------------------------------------------------------
# figure 3: boundary
# --------------------------------------------------------------------------
def fig_boundary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection

    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.6, 1, 1.3, 1.3, 1.5], wspace=0.12)
    P = F.BOUNDARY_SEQ
    for col, k in enumerate((0, 1, 2, 3)):
        ax = fig.add_subplot(gs[col + (0 if k == 0 else 0)]) if k < 4 else None
    fig.clf()
    gs = fig.add_gridspec(1, 5, width_ratios=[0.55, 1, 1.4, 1.7, 1.55], wspace=0.10)
    for col, k in enumerate((0, 1, 2, 3)):
        ax = fig.add_subplot(gs[col])
        polys = level_tiles(k)
        ax.add_collection(PolyCollection([p for p, _ in polys],
                                         facecolors="#eeeeee", edgecolors=GREY,
                                         linewidths=0.3 if k > 1 else 0.6))
        be = boundary_edges(polys)
        assert len(be) == P[k]
        ax.add_collection(LineCollection(be, colors=ORANGE, linewidths=1.3))
        allp = np.vstack([p for p, _ in polys])
        ax.set_xlim(allp[:, 0].min() - 0.5, allp[:, 0].max() + 0.5)
        ax.set_ylim(allp[:, 1].min() - 0.5, allp[:, 1].max() + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("$k=%d$: %d tile%s\n$P=%d$ boundary edges" % (k, len(polys), "" if len(polys) == 1 else "s", P[k]),
                     fontsize=9)
    ax = fig.add_subplot(gs[4])
    ks = np.arange(len(P))
    ax.semilogy(ks, P, "o-", color=ORANGE, label="$P(k)$")
    nu = float(F.NU)
    ax.semilogy(ks, [P[1] * nu ** (i - 1) for i in ks], "--", color=AQUA,
                label="$\\nu^k$, $\\nu=\\varphi^3=%.3f$" % nu)
    ax.semilogy(ks, [P[1] * float(F.LAM) ** (i - 1) for i in ks], ":", color=BLUE,
                label="$\\lambda^k$ (naive)")
    ax.set_xlabel("$k$")
    ax.set_title("perimeter: dimension $\\log\\nu/\\log\\lambda=%.4f$" %
                 float(F.BOUNDARY_DIM), fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean(ax)
    _save(fig, "spectre_boundary")


if __name__ == "__main__":
    print("selftest:", selftest())
    if "--check" in sys.argv:
        sys.exit(0)
    fig_family()
    fig_census()
    fig_boundary()
