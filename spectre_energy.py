#!/usr/bin/python3
"""How far above the crystalline floor are the monotile point sets?

For a configuration of unit density the Gaussian-kernel energy per particle is

        E(alpha) = lim (1/N) sum_{i != j} exp(-alpha |x_i - x_j|^2),

the simplest member of the family of potentials that are completely monotone
functions of the squared distance.  For that whole family the triangular lattice
is the conjectured planar minimiser at density one (Cohn--Kumar); the
OpenAI "math" collection (family 090, https://github.com/openai/math) claims a
proof.  This script does not use that claim for anything it prints.  It
measures, numerically,

    gap(alpha) = E_set(alpha) / E_A2(alpha) - 1,

for the vertex set and the centroid set of the Hat, Spectre and Turtle
tilings (the three members Tile(1,sqrt3), Tile(1,1), Tile(sqrt3,1) of the
family of tile_family.py), against four controls: the triangular lattice
(gap 0 by construction), the square lattice, the honeycomb lattice and a
Poisson process.

Method.  Level-5 Delta supertile of each tiling (34 649 tiles).  The window is a
disc inside the patch, at least one cutoff length from its boundary.  Lengths
are rescaled to unit density; the density is estimated as a ratio of sums over
the SAME set of whole tiles (vertices weighted by 1/multiplicity), so the
boundary of the window cancels, and it is checked against the exact value for
the centroids, 1/(A(a,b) + sqrt3 (b^2-a^2)/(1+lambda^2)), in selftest().

    python3 spectre_energy.py             # numbers + figure
    python3 spectre_energy.py --check     # assertions only
"""
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectre as S                         # noqa: E402

S3 = 3 ** 0.5
LAM2 = 4 + 15 ** 0.5                        # the inflation, 4 + sqrt15
FAMILY = (("Hat", 1.0, S3), ("Spectre", 1.0, 1.0), ("Turtle", S3, 1.0))
FIGDIR = os.path.join(HERE, "paper", "figures")
ALPHAS = np.logspace(-1.0, 0.7, 40)
CUT2 = 17.0                                 # exp(-17) = 4e-8 truncation


def area_form(a, b):
    return 2 * S3 * a * a + 3 * a * b + S3 * b * b


def exact_tile_density(a, b):
    """Tiles per unit area: the Mystic share is 1/(1+lambda^2)."""
    mean = area_form(a, b) + S3 * (b * b - a * a) / (1.0 + LAM2)
    return 1.0 / mean


VERTICES_PER_TILE = 6          # bulk value; checked against V - P/2 in selftest()


def exact_density(a, b, which):
    r = exact_tile_density(a, b)
    return r if which == "centroids" else VERTICES_PER_TILE * r


# --------------------------------------------------------------------------
# the tilings
# --------------------------------------------------------------------------
def polygon_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def tiling(a, b, n):
    tiles = S.buildSpectreTiles(n, a, b, rotation=30)
    placed = []
    tiles["Delta"].forEachTile(lambda T, l: placed.append((T.copy(), l)))
    base = S.get_spectre_points(a, b)
    myst = S.get_spectre_points(b, a)
    return [((myst if l == "Gamma2" else base) @ T[:, :2].T + T[:, 2]) for T, l in placed]


class Patch(object):
    def __init__(self, a, b, n=5):
        self.polys = tiling(a, b, n)
        self.centroid = np.array([P.mean(0) for P in self.polys])
        self.area = np.array([polygon_area(P) for P in self.polys])
        # unique vertices with multiplicity
        key = {}
        mult = defaultdict(int)
        owner = []
        for P in self.polys:
            ids = []
            for p in P:
                k = tuple(np.round(p / 1e-6).astype(np.int64))
                if k not in key:
                    key[k] = len(key)
                ids.append(key[k])
                mult[key[k]] += 1
            owner.append(ids)
        self.owner = owner
        self.vert = np.zeros((len(key), 2))
        for k, i in key.items():
            self.vert[i] = np.array(k) * 1e-6
        self.mult = np.array([mult[i] for i in range(len(key))], float)
        # the polygon has 14 listed vertices, but a vertex of the tile can be a
        # straight point of an edge (angle 180) shared with a neighbour's corner,
        # so the multiplicity here is "number of tiles listing it", which is what
        # the weighting needs.
        # inradius around the centre of mass, from the boundary edges
        cnt = defaultdict(int)
        for ids in owner:
            for i in range(len(ids)):
                cnt[frozenset((ids[i], ids[(i + 1) % len(ids)]))] += 1
        bnd = {v for e, c in cnt.items() if c == 1 for v in e}
        self.com = (self.centroid * self.area[:, None]).sum(0) / self.area.sum()
        bv = self.vert[sorted(bnd)]
        self.inradius = float(np.hypot(*(bv - self.com).T).min())


def window_density(P, r0, which):
    """Density of `which` in the window of whole tiles with centroid within r0
    of the centre of mass: a ratio of sums over the same tiles."""
    sel = np.where(np.hypot(*(P.centroid - P.com).T) < r0)[0]
    A = P.area[sel].sum()
    if which == "centroid":
        return len(sel) / A, sel
    w = 0.0
    for t in sel:
        w += sum(1.0 / P.mult[i] for i in P.owner[t])
    return w / A, sel


def energy_curve(points, centre, r_lo, r_hi, rho, alphas=ALPHAS):
    """Mean over reference points with r_lo <= |p - centre| < r_hi of
    sum_{q != p} exp(-alpha d^2) in lengths rescaled to unit density.
    Returns (curve, number of reference points)."""
    scale = np.sqrt(rho)
    X = (points - centre) * scale
    dmax = np.sqrt(CUT2 / alphas.min())
    tree = cKDTree(X)
    rr = np.hypot(*X.T)
    ref = np.where((rr >= r_lo * scale) & (rr < r_hi * scale))[0]
    d2 = []
    for i in ref:
        nb = tree.query_ball_point(X[i], dmax)
        dd = X[nb] - X[i]
        q = (dd ** 2).sum(1)
        d2.append(q[q > 1e-12])
    d2 = np.concatenate(d2)
    return np.array([np.exp(-a * d2).sum() / len(ref) for a in alphas]), len(ref)


# --------------------------------------------------------------------------
# controls (exact lattice sums, unit density)
# --------------------------------------------------------------------------
def lattice_energy(basis, motif, alphas=ALPHAS, N=24):
    """E(alpha) for a periodic point set: lattice `basis` (2x2 rows) with a
    motif of offsets, rescaled to density one."""
    basis = np.array(basis, float)
    motif = np.array(motif, float)
    dens = len(motif) / abs(np.linalg.det(basis))
    s = np.sqrt(dens)                       # lengths * sqrt(dens) -> density 1
    basis, motif = basis * s, motif * s
    I, J = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
    cell = I.ravel()[:, None] * basis[0] + J.ravel()[:, None] * basis[1]
    out = np.zeros(len(alphas))
    for m in motif:
        for m2 in motif:
            d = cell + (m2 - m)
            q = (d ** 2).sum(1)
            q = q[q > 1e-12]
            out += np.array([np.exp(-a * q).sum() for a in alphas])
    return out / len(motif)


def controls():
    tri = lattice_energy([(1, 0), (0.5, S3 / 2)], [(0, 0)])
    sq = lattice_energy([(1, 0), (0, 1)], [(0, 0)])
    a1, a2 = np.array((1.5, S3 / 2)), np.array((1.5, -S3 / 2))
    hon = lattice_energy([a1, a2], [(0, 0), (1, 0)])
    poisson = np.pi / ALPHAS
    return {"triangular": tri, "square": sq, "honeycomb": hon, "Poisson": poisson}


# --------------------------------------------------------------------------
def measure(n=5):
    out = {}
    for name, a, b in FAMILY:
        P = Patch(a, b, n)
        cut = np.sqrt(CUT2 / ALPHAS.min())
        # window radius in the tiling's own units; cut is in unit-density
        # lengths and the densities are below one, so cut/sqrt(rho) is larger
        r0 = 0.8 * P.inradius - 2.0 * cut / np.sqrt(exact_density(a, b, "vertices"))
        for which, pts in (("vertices", P.vert), ("centroids", P.centroid)):
            rho = exact_density(a, b, which)
            # two disjoint halves of equal area: a disc and its annulus
            r_in = r0 / np.sqrt(2.0)
            E_in, n_in = energy_curve(pts, P.com, 0.0, r_in, rho)
            E_out, n_out = energy_curve(pts, P.com, r_in, r0, rho)
            E = (n_in * E_in + n_out * E_out) / (n_in + n_out)
            out[(name, which)] = dict(E=E, spread=np.abs(E_in - E_out) / 2, rho=rho,
                                      r0=r0, n_ref=n_in + n_out,
                                      tiles=len(P.polys), inradius=P.inradius)
    return out


def selftest():
    # 1. the exact densities match the patch: centroids by a window of whole
    #    tiles, vertices by V - P/2 over A (P(4) = 3198 boundary edges)
    for name, a, b in FAMILY:
        P = Patch(a, b, 4)
        r0 = 0.5 * P.inradius
        rho, _ = window_density(P, r0, "centroid")
        assert abs(rho / exact_tile_density(a, b) - 1) < 5e-3, (name, rho)
        rho_v = (len(P.vert) - 3198 / 2.0) / P.area.sum()
        assert abs(rho_v / exact_density(a, b, "vertices") - 1) < 1e-3, (name, rho_v)
    # 2. the three tilings are tilings: every sample point is in exactly one tile
    from matplotlib.path import Path
    rng = np.random.RandomState(1)
    for name, a, b in FAMILY:
        P = Patch(a, b, 3)
        pts = P.com + rng.uniform(-1, 1, (300, 2)) * 0.45 * P.inradius
        hit = np.zeros(len(pts), int)
        for poly in P.polys:
            lo, hi = poly.min(0), poly.max(0)
            m = np.where((pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) &
                         (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1]))[0]
            if len(m):
                hit[m] += Path(poly).contains_points(pts[m])
        assert (hit == 1).all(), (name, np.bincount(hit))
    # 3. the lattice sums are right: triangular beats square beats Poisson at
    #    every alpha in range, and the large-alpha limit is the kissing number
    c = controls()
    assert (c["triangular"] <= c["square"] + 1e-12).all()
    assert (c["triangular"] <= c["honeycomb"] + 1e-12).all()
    assert (c["square"] <= c["Poisson"]).all()
    a_big = 40.0
    tri = lattice_energy([(1, 0), (0.5, S3 / 2)], [(0, 0)], alphas=np.array([a_big]))[0]
    d2 = 2 / S3                                    # nearest-neighbour^2 at density 1
    assert abs(tri / (6 * np.exp(-a_big * d2)) - 1) < 1e-3
    # 4. Poisson check of the mean-field limit: E ~ pi/alpha as alpha -> 0
    t = lattice_energy([(1, 0), (0.5, S3 / 2)], [(0, 0)], alphas=np.array([0.05]))[0]
    assert abs(t * 0.05 / np.pi - 1) < 5e-2
    return True


def report(res):
    tri = controls()["triangular"]
    print("  set                    rho        gap(alpha=0.2)    gap(alpha=1)      gap(alpha=4)   (+- half the disc/annulus difference)")
    for (name, which), d in res.items():
        g = d["E"] / tri - 1
        f = lambda a: np.interp(np.log(a), np.log(ALPHAS), g)
        h = lambda a: np.interp(np.log(a), np.log(ALPHAS), d["spread"] / tri)
        print("  %-8s %-10s %.4f  %s %s %s   (n_ref=%d, r0=%.0f)"
              % (name, which, d["rho"],
                 *["%+9.4f+-%.4f" % (f(a), h(a)) for a in (0.2, 1.0, 4.0)],
                 d["n_ref"], d["r0"]))


def figure(res, fname="spectre_energy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
    INK, GREY = "#222222", "#8a877f"
    COL = {"Hat": BLUE, "Spectre": INK, "Turtle": ORANGE}
    ctr = controls()
    tri = ctr["triangular"]

    fig = plt.figure(figsize=(13.0, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.28)
    for k, which in enumerate(("vertices", "centroids")):
        ax = fig.add_subplot(gs[k])
        for name, a, b in FAMILY:
            g = res[(name, which)]["E"] / tri - 1
            ax.plot(ALPHAS, g, color=COL[name], lw=1.8, label=name)
        for nm, ls, c in (("square", (0, (5, 2.5)), GREY), ("honeycomb", (0, (1, 2)), AQUA),
                          ("Poisson", ":", VIOLET)):
            ax.plot(ALPHAS, ctr[nm] / tri - 1, color=c, lw=1.2, ls=ls, label=nm)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(-0.003, 30)
        ax.set_xlabel("kernel width $\\alpha$ (unit density)")
        ax.set_ylabel("$E/E_{A_2}-1$")
        ax.set_title("(%s) %s of the tilings" % ("ab"[k], which), fontsize=10)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    # (c) bars at alpha = 1
    ax = fig.add_subplot(gs[2])
    a0 = 1.0
    f = lambda y: float(np.interp(np.log(a0), np.log(ALPHAS), y / tri - 1))
    labels, vals, cols = [], [], []
    for name, a, b in FAMILY:
        for which, hatch in (("vertices", ""), ("centroids", "//")):
            labels.append("%s\n%s" % (name, which[:4] + "."))
            vals.append(f(res[(name, which)]["E"]))
            cols.append(COL[name])
    for nm in ("honeycomb", "square", "Poisson"):
        labels.append(nm)
        vals.append(f(ctr[nm]))
        cols.append(GREY)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.7)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(0, 1.0)
    for yy, v in zip(y, vals):
        ax.text(max(v, 0) * 1.25 + 4e-4, yy, "%.4f" % v, va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("gap at $\\alpha=1$")
    ax.set_title("(c) at one width", fontsize=10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
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
    res = measure()
    report(res)
    figure(res)
