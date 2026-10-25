#!/usr/bin/python3
"""Mystic fibres of the Spectre/Hat tiling on the full substitution levels.

Context
-------
At the Hat point Tile(1, sqrt3) every tile of the substitution system is a
union of kites of the Laves lattice [3.4.6.4]: a Hat is 8 kites, the Mystic
(Gamma2, Tile(sqrt3, 1), the Turtle) is 10.  Rolling the lattice up on the torus
T = R^2 / L, L = Z E1 + Z E2, E1 = (3, sqrt3), E2 = (0, 2 sqrt3), (|det| = 6 sqrt3,
six kites per domain) puts every Mystic on T.  Two Mystics are in the same
FIBRE CLASS when they are translates of each other by a vector of L, i.e. when
they lie over the same point of the torus with the same orientation.

This script is self-contained (spectre.py only).  It computes, for each full
level n of the Delta-seeded substitution,

    * the number of tiles and of Mystics (the Mystic count of level n equals
      the tile total of level n-1: the "Mystic rule"),
    * the fibre classes, their sizes and their anchors on the torus,
    * the six first-edge angles 30 + 60 k and the three double curves,
    * the environment of each interior Mystic (which neighbours, where), to see
      whether one class can have more than one environment,
    * the placement of each Mystic relative to the first, as an isometry g with g(M_0) = M:
      every g is a symmetry of the kite lattice, all Mystics of a level share one handedness,
      and the fibre class of a Mystic is fixed by the rotation part of g alone.

Why the classes are at most six.  Here "Mystic" is Gamma2 = Tile(sqrt3, 1) alone (in the Spectre papers the Mystic is
the pair Gamma1 + Gamma2; either gives the same classes).  Every tile of every level lies on the [3.4.6.4] kite lattice K:
the child placement maps of each level lie in p6m(K), which is a finite-state induction on the classes mod Lambda of the
supertile quad points, proved for all levels in exact arithmetic by mystic_exact/c3_alln.py.  All placements of a level have
one determinant (each level composes every placement with one of the eight reflections of the substitution), and the
Mystic has no symmetry.  So two Mystics of one level differ by a rotation in 60 degree steps composed with a translation in
Lambda, and the fibre classes are in bijection with the rotations that occur: at most six, the anchor fixed by the angle.
A level-n Delta supertile contains a level-(n-1) Delta supertile (row Delta of the substitution matrix is all ones), so
all six occur at every level n >= 3.  The selftest below checks this in floating point through level 4; mystic_exact/
checks it exactly through level 5 and proves registration for every level.

Everything is done in double precision on coordinates in Z[sqrt3]; keys are
rounded at 1e-6, and the selftest asserts that every vertex of every tile is a
lattice site, so a rounding error cannot go unnoticed.

    python3 spectre_mystic_fibres.py            # table + figures (depth <= 5)
    python3 spectre_mystic_fibres.py --check    # assertions only
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectre as S                      # noqa: E402

S3 = 3 ** 0.5
E1 = np.array([3.0, S3])
E2 = np.array([0.0, 2 * S3])
MI = np.linalg.inv(np.array([E1, E2]).T)
TOL = 1e-6
KNOWN_TILES = {1: 9, 2: 71, 3: 559, 4: 4401}      # geometric census of Delta
KNOWN_MYSTIC = {1: 1, 2: 8, 3: 63, 4: 496}        # = matrix-census total of level n-1
MATRIX_TOTAL = {0: 1, 1: 8, 2: 63, 3: 496, 4: 3905}


def u(deg):
    t = np.radians(deg)
    return np.array([np.cos(t), np.sin(t)])


# six site classes of the lattice modulo L: centre, 3 edge midpoints, 2 vertices
SITE_UV = {}
for _name, _p in ([("centre", np.zeros(2))]
                  + [("edge", S3 * u(30 + 60 * k)) for k in range(6)]
                  + [("vertex", 2 * u(60 * k)) for k in range(6)]):
    _q = MI @ _p
    SITE_UV[tuple(np.round(_q % 1.0, 6) % 1.0)] = _name
DCURVE = {(0.5, 0.0): "D_a", (0.0, 0.5): "D_c", (0.5, 0.5): "D_b"}
ANCHOR_ANGLES = {"D_a": (30, 210), "D_c": (90, 270), "D_b": (150, 330)}


def site_kind(p, c0):
    q = MI @ (p - c0)
    return SITE_UV.get(tuple(np.round(q % 1.0, 6) % 1.0))


# --------------------------------------------------------------------------
# the tiling at the Hat point, registered on the lattice
# --------------------------------------------------------------------------
def hat_level(n):
    """Full level n: list of (polygon (14,2), label), shifted so that every
    vertex is a lattice site (the shift is found, not assumed)."""
    tiles = S.buildSpectreTiles(n, 1.0, S3, rotation=30)
    placed = []
    tiles["Delta"].forEachTile(lambda T, l: placed.append((T.copy(), l)))
    pts = S.get_spectre_points(1.0, S3)
    myst = S.get_spectre_points(S3, 1.0)
    polys = [(((myst if l == "Gamma2" else pts) @ T[:, :2].T + T[:, 2]), l)
             for T, l in placed]
    # registration: tile 0, vertex v, mapped to a site class by a shift
    offs = [np.zeros(2)] + [S3 * u(30 + 60 * k) for k in range(6)] \
        + [2 * u(60 * k) for k in range(6)]
    P0 = polys[0][0]
    c0 = None
    for v in P0:
        for off in offs:
            cand = v - off
            if all(site_kind(p, cand) for p in P0):
                c0 = cand
                break
        if c0 is not None:
            break
    assert c0 is not None, "tile 0 does not sit on the lattice"
    polys = [(P - c0, l) for P, l in polys]
    return polys


def uv_of(p):
    return MI @ p


def key_pt(p):
    return tuple(np.round(p / TOL).astype(np.int64))


# --------------------------------------------------------------------------
# fibre classes
# --------------------------------------------------------------------------
def mystic_data(polys):
    """For each mystic: (index, class key, first-edge angle, anchor curve)."""
    out = []
    for i, (P, l) in enumerate(polys):
        if l != "Gamma2":
            continue
        q0 = uv_of(P[0])
        sh = np.floor(q0 + 1e-9)                  # reduce vertex 0 into [0,1)^2
        t = sh[0] * E1 + sh[1] * E2
        Pr = P - t
        key = tuple(key_pt(p) for p in Pr)
        e = P[1] - P[0]
        ang = int(round(float(np.degrees(np.arctan2(e[1], e[0]))))) % 360
        mid = tuple(np.round(q0 % 1.0, 6) % 1.0)      # anchor = vertex 0 (an edge midpoint)
        out.append(dict(idx=i, key=key, angle=ang, anchor=DCURVE.get(mid)))
    return out


def classes_of(polys):
    data = mystic_data(polys)
    groups = defaultdict(list)
    for d in data:
        groups[d["key"]].append(d)
    cls = sorted(groups.values(), key=lambda g: (len(g), g[0]["angle"]))
    return cls


def environment(polys, i, radius=5.2, cache=None):
    """Signature of the neighbourhood of tile i: the set of (label, rounded
    vertex-0 offset) of the tiles within `radius`; None if the disc of that
    radius is not fully covered (boundary of the patch)."""
    P, _ = polys[i]
    c = P.mean(0)
    cen = cache["cen"]
    near = np.where(np.hypot(*(cen - c).T) < radius + 4.0)[0]
    # covered? sample the disc, require every sample inside some near tile
    from matplotlib.path import Path
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    samp = [c + r * np.array([np.cos(t), np.sin(t)])
            for r in (radius * 0.4, radius * 0.7, radius) for t in th]
    samp = np.array(samp)
    cover = np.zeros(len(samp), bool)
    for j in near:
        cover |= Path(polys[j][0]).contains_points(samp)
    if not cover.all():
        return None
    sig = []
    for j in near:
        if np.hypot(*(cen[j] - c)) < radius:
            d = polys[j][0][0] - P[0]
            sig.append((polys[j][1], tuple(np.round(d / 1e-4).astype(np.int64))))
    return tuple(sorted(sig))


def env_counts(res, radii=(3.5, 5.2, 8.0), sample=200):
    polys = res["polys"]
    cache = dict(cen=np.array([P.mean(0) for P, _ in polys]))
    out = {}
    for R in radii:
        row = []
        for g in res["classes"]:
            e = set()
            for d in g[:sample]:
                s_ = environment(polys, d["idx"], radius=R, cache=cache)
                if s_ is not None:
                    e.add(s_)
            row.append(len(e))
        out[R] = row
    return out


def census(n, with_env=True):
    polys = hat_level(n)
    labs = Counter(l for _, l in polys)
    cls = classes_of(polys)
    res = dict(n=n, tiles=len(polys), mystics=labs["Gamma2"], classes=cls,
               sizes=[len(g) for g in cls], polys=polys)
    # check: the anchor of a class is a single (angle, curve)
    for g in cls:
        assert len({(d["angle"], d["anchor"]) for d in g}) == 1
    res["angles"] = sorted({g[0]["angle"] for g in cls})
    if with_env:
        cen = np.array([P.mean(0) for P, _ in polys])
        cache = dict(cen=cen)
        envs = []
        for g in cls:
            e = set()
            ninterior = 0
            for d in g:
                s = environment(polys, d["idx"], cache=cache)
                if s is not None:
                    e.add(s)
                    ninterior += 1
            envs.append((ninterior, len(e)))
        res["env"] = envs
    return res


def placements(polys):
    """The isometries g, g(M_0) = M, from the first Mystic M_0 to every Mystic M of a level.

    Returns (determinants, rotation angles of the proper parts in degrees, whether every g lies in
    p6m, number of fibre classes).  g lies in p6m when its linear part is a rotation by a multiple of
    60 degrees or one of the D6 reflections, and g(0) is in Lambda (the origin is a hexagon centre
    after the registration in hat_level)."""
    my = [P for P, l in polys if l == "Gamma2"]
    P0 = my[0]
    A = np.c_[P0[:3], np.ones(3)]
    dets, angs, in_p6m = Counter(), Counter(), True
    for P in my:
        sol = np.linalg.solve(A, P[:3])
        M, b = sol[:2].T, sol[2]
        assert np.allclose(P0 @ M.T + b, P, atol=1e-6) and np.allclose(M @ M.T, np.eye(2), atol=1e-9)
        det = int(round(np.linalg.det(M)))
        ang = int(round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))) % 360
        dets[det] += 1
        angs[ang] += 1
        q = MI @ b
        # proper: a rotation by a multiple of 60 degrees; improper: a reflection in a line at a multiple of 30 degrees,
        # i.e. atan2 of its first column a multiple of 60 degrees.  Either way ang % 60 == 0.
        in_p6m &= ang % 60 == 0 and np.allclose(q, np.round(q), atol=1e-6)
    return dets, angs, in_p6m, len(classes_of(polys))


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
def selftest(depths=(1, 2, 3, 4)):
    for n in depths:
        polys = hat_level(n)
        n_my = sum(1 for _, l in polys if l == "Gamma2")
        if n in KNOWN_TILES:
            assert len(polys) == KNOWN_TILES[n], (n, len(polys))
            assert n_my == KNOWN_MYSTIC[n], (n, n_my)
        # Mystic rule: the Mystics of level n are the 'previous generation'
        # (geometric total = matrix total + matrix total one level earlier)
        if n - 1 in MATRIX_TOTAL:
            assert n_my == MATRIX_TOTAL[n - 1]
            assert len(polys) == MATRIX_TOTAL[n] + MATRIX_TOTAL[n - 1] or n not in MATRIX_TOTAL
        # every vertex of every tile is a lattice site
        for P, _ in polys:
            for p in P:
                assert site_kind(p, np.zeros(2)), "vertex off the lattice"
        # area: Hat = 8 kites, Mystic = 10 kites (kite area sqrt3)
        for P, l in polys:
            x, y = P[:, 0], P[:, 1]
            A = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            assert abs(A / S3 - (10 if l == "Gamma2" else 8)) < 1e-6, (l, A / S3)
    # the six-class bound and the orbit statement on the levels computed
    for n in depths:
        r = census(n, with_env=False)
        assert len(r["classes"]) <= 72
        assert set(r["angles"]) <= {30, 90, 150, 210, 270, 330}
        # every Mystic is a lattice symmetry of the first, one handedness per level, and the
        # fibre classes are exactly the rotations that occur (at most six; six from level 3 on)
        dets, angs, in_p6m, ncls = placements(r["polys"])
        assert in_p6m, ("a Mystic placement is not a symmetry of the kite lattice", n)
        assert len(dets) == 1, ("two handednesses in one level", n, dets)
        assert ncls == len(angs) <= 6, (n, ncls, angs)
        assert n < 3 or ncls == 6, (n, ncls)
    return True


# --------------------------------------------------------------------------
# report and figures
# --------------------------------------------------------------------------
def report(depths):
    rows = []
    print("n   tiles  mystics  classes  sizes (sorted)                      env (interior, distinct)")
    for n in depths:
        r = census(n, with_env=(n <= 4))
        rows.append(r)
        sizes = sorted(r["sizes"])
        print("%d %7d %8d %8d  %-36s %s" % (
            n, r["tiles"], r["mystics"], len(r["classes"]),
            sizes if len(sizes) <= 8 else str(sizes[:8])[:-1] + ", ...]",
            r.get("env", "")))
    return rows


def figure(rows, fname="spectre_mystic_fibres"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
    PINK, GOLD = "#d4569c", "#c9a227"
    PAL = [BLUE, ORANGE, AQUA, VIOLET, PINK, GOLD]
    ANG = [30, 90, 150, 210, 270, 330]
    INK, GREY = "#222222", "#bdbab2"

    r3 = next(r for r in rows if r["n"] == 3)
    r4 = next(r for r in rows if r["n"] == 4)
    r5 = next(r for r in rows if r["n"] == 5)

    fig = plt.figure(figsize=(13.0, 7.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], hspace=0.30, wspace=0.22)

    # (a) level 3: all 559 tiles, mystics coloured by first-edge angle
    ax = fig.add_subplot(gs[:, 0])
    polys = r3["polys"]
    ang_of = {d["idx"]: d["angle"] for g in r3["classes"] for d in g}
    fc, lw, ec = [], [], []
    for i, (P, l) in enumerate(polys):
        if l == "Gamma2":
            fc.append(PAL[ANG.index(ang_of[i])])
            ec.append(INK)
        else:
            fc.append("#f1efe9")
            ec.append(GREY)
    ax.add_collection(PolyCollection([P for P, _ in polys], facecolors=fc,
                                     edgecolors=ec, linewidths=0.5))
    allp = np.vstack([P for P, _ in polys])
    ax.set_xlim(allp[:, 0].min() - 1, allp[:, 0].max() + 1)
    ax.set_ylim(allp[:, 1].min() - 1, allp[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) level 3: %d tiles, %d Mystics\n(coloured by first-edge angle)"
                 % (r3["tiles"], r3["mystics"]), fontsize=10)
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=PAL[k], ms=8,
                                  label="%d°" % ANG[k]) for k in range(6)],
              loc="lower center", ncol=6, frameon=False, fontsize=8,
              bbox_to_anchor=(0.5, -0.04), handletextpad=0.2, columnspacing=0.9)

    # (b) class sizes relative to the mean, by level
    axb = fig.add_subplot(gs[0, 1:])
    xs = [r["n"] for r in rows]
    for xi, r in zip(xs, rows):
        if len(r["sizes"]) < 6:
            continue
        m = sum(r["sizes"]) / 6.0
        for k, g in enumerate(sorted(r["classes"], key=lambda g: g[0]["angle"])):
            axb.scatter([xi + (k - 2.5) * 0.06], [len(g) / m], s=34, color=PAL[k],
                        zorder=3, edgecolors="white", linewidths=0.6)
    axb.axhline(1.0, color=INK, lw=0.8, ls=(0, (4, 3)))
    axb.set_xticks([r["n"] for r in rows if len(r["sizes"]) >= 6])
    axb.set_xlabel("substitution level $n$")
    axb.set_ylabel("class size / mean")
    axb.set_title("(b) the six fibre classes equidistribute", fontsize=10)
    for n_ in [r["n"] for r in rows if len(r["sizes"]) >= 6]:
        r = next(q for q in rows if q["n"] == n_)
        axb.text(n_, 1.0, "%d Mystics" % r["mystics"], ha="center",
                 va="bottom", fontsize=8, color=INK, transform=axb.get_xaxis_transform())
    for s_ in ("top", "right"):
        axb.spines[s_].set_visible(False)

    # (c) distinct environments of a Mystic within radius R
    axc = fig.add_subplot(gs[1, 1])
    ec = env_counts(r5)
    Rs = sorted(ec)
    for k in range(6):
        axc.plot(range(len(Rs)), [ec[R][k] for R in Rs], "o-", color=PAL[k],
                 lw=1.2, ms=4)
    axc.set_xticks(range(len(Rs)))
    axc.set_xticklabels(["%.1f" % R for R in Rs])
    axc.set_xlabel("radius $R$ (edge $=1$)")
    axc.set_ylabel("distinct environments")
    axc.set_title("(c) environments per class: 1, 4, 9", fontsize=10)
    for s_ in ("top", "right"):
        axc.spines[s_].set_visible(False)

    # (d) the fractions of each angle by level converge to 1/6?
    axd = fig.add_subplot(gs[1, 2])
    for k, a in enumerate(ANG):
        fr = []
        for r in rows:
            tot = sum(r["sizes"])
            c = sum(len(g) for g in r["classes"] if g[0]["angle"] == a)
            fr.append(c / tot)
        axd.plot(xs, fr, "o-", color=PAL[k], lw=1.4, ms=4, label="%d°" % a)
    axd.axhline(1 / 6, color=INK, lw=0.8, ls=(0, (4, 3)))
    axd.set_xticks(xs)
    axd.set_xlabel("substitution level $n$")
    axd.set_ylabel("fraction of Mystics")
    axd.set_title("(d) angle fractions by level", fontsize=10)
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)

    os.makedirs(os.path.join(HERE, "paper", "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "paper", "figures", fname + "." + ext),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figures/%s.{pdf,png}" % fname)


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("selftest:", selftest())
        sys.exit(0)
    print("selftest:", selftest())
    dmax = 5
    rows = report(range(1, dmax + 1))
    figure(rows)
