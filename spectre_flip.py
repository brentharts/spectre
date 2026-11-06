#!/usr/bin/python3
"""
spectre_flip.py -- visualise the +1 / -1 "flip" returned by spectre.trot_inv()

Background
----------
spectre_tiles_blender.py (plotVertices, ~line 803) does

    rot, scl = spectre.trot_inv(tile_transformation)

and then treats scl (+1 / -1) as a "flip" flag, with the meaning swapped on odd
and even iteration counts (ITER % 2).  Where does the +/-1 come from?

  * buildSupertiles() left-multiplies every sub-tile placement by the mirror
    R = diag(-1, 1)  (the "@TODO: Not trot(180)" line in spectre.py).  That is
    the "twist": every supertiling level adds one reflection, so the linear
    part of a tile transform is  Rot(theta) or Rot(theta)*diag(1,-1).
  * trot_inv() tries to recover (angle, scaleY) from the 2x2 part by comparing
    two angles read off the matrix:

        angle1 = atan2( T[1,0], T[0,0])      angle of the image of the x axis
        angle2 = atan2(-T[0,1], T[1,1])      angle of the image of the y axis,
                                             measured as if there were no mirror

    rotation   Rot(a)            -> angle1 =  angle2
    reflection Rot(a)*diag(1,-1) -> angle1 =  angle2 +/- 180

    so the decomposition is   T_lin = Rot(angle1) * diag(1, scaleY).

This script draws that picture with matplotlib only (no Blender) and checks
trot_inv() against the ground truth sign(det T_lin).

Output
------
  spectre_flip.png          overview: truth vs what Blender flags, angle1/angle2
                            plane, and per-level statistics
  spectre_flip_gallery.png  step by step  base tile -> diag(1,scl) -> Rot(angle1)
                            compared with the real transform

Usage
-----
  python3 spectre_flip.py                 # 3 iterations, writes the two PNGs
  python3 spectre_flip.py -n 4 --show     # more levels, open interactive windows
"""
import argparse
import contextlib
import io
import os
import sys
from collections import Counter

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectre  # noqa: E402

# Okabe-Ito colour-blind safe palette
BLUE = "#0072B2"      # proper rotation  (det > 0)
ORANGE = "#D55E00"    # reflection       (det < 0)
MAGENTA = "#CC79A7"   # Blender "FLIP" material
GREEN = "#009E73"
GREY = "#BBBBBB"
BLACK = "#000000"


# --------------------------------------------------------------------------
# data collection
# --------------------------------------------------------------------------
def rot_lin(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def raw_angles(T):
    """The two angles trot_inv() compares (same formulas as spectre.trot_inv)."""
    a1 = int(np.round(np.rad2deg(np.arctan2(T[1, 0], T[0, 0]))))
    if a1 == -180:
        a1 = 180
    a2 = int(np.round(np.rad2deg(np.arctan2(-T[0, 1], T[1, 1]))))
    return a1, a2


def blender_is_flip(scl, iteration):
    """Exactly the rule in spectre_tiles_blender.plotVertices (line ~807)."""
    if iteration % 2:
        return scl == 1
    return scl != 1


def collect(n_iter, edge_a, edge_b, rotation):
    """Build the tiling and record, for every tile, what trot_inv says and the truth."""
    with contextlib.redirect_stdout(io.StringIO()):  # trot() prints on new angles
        tiles = spectre.buildSpectreTiles(n_iter, edge_a, edge_b, rotation=rotation)

    recs = []

    def visit(T, label):
        T = T.copy()
        lin = T[:, :2]
        det = float(lin[0, 0] * lin[1, 1] - lin[0, 1] * lin[1, 0])
        rot, scl = spectre.trot_inv(T)
        a1, a2 = raw_angles(T)
        pts = spectre.Mystic_SPECTRE_POINTS if label == "Gamma2" else spectre.SPECTRE_POINTS
        recon = rot_lin(rot) @ np.diag([1.0, float(scl)])
        true_scl = 1 if det > 0 else -1
        recs.append({
            "T": T, "label": label, "pts": pts.copy(),
            "verts": pts.dot(lin.T) + T[:, 2],
            "det": det, "a1": a1, "a2": a2, "rot": rot, "scl": scl,
            "true_scl": true_scl,
            "mismatch": scl != true_scl,
            "recon_err": float(np.abs(recon - lin).max()),
            "blender_flip": blender_is_flip(scl, n_iter),
        })

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tiles["Delta"].forEachTile(visit)
    return recs, buf.getvalue().count("ValueError")


def level_stats(recs, n):
    tot = len(recs)
    return {
        "n": n, "tiles": tot,
        "reflected": sum(r["det"] < 0 for r in recs),
        "scl_neg": sum(r["scl"] == -1 for r in recs),
        "mismatch": sum(r["mismatch"] for r in recs),
        "bad_recon": sum(r["recon_err"] > 1e-6 for r in recs),
        "blender": sum(r["blender_flip"] for r in recs),
    }


def print_table(stats):
    hdr = ("level", "tiles", "det<0", "scl=-1", "scl wrong", "recon bad", "blender flips")
    print("  ".join(f"{h:>13}" for h in hdr))
    for s in stats:
        row = (s["n"], s["tiles"], s["reflected"], s["scl_neg"], s["mismatch"],
               s["bad_recon"], s["blender"])
        print("  ".join(f"{v:>13}" for v in row))


# --------------------------------------------------------------------------
# overview figure
# --------------------------------------------------------------------------
def draw_tiling(ax, recs, face, outline_mask=None, title=""):
    pc = PolyCollection([r["verts"] for r in recs], facecolors=face,
                        edgecolors="#333333", linewidths=0.25)
    ax.add_collection(pc)
    if outline_mask is not None and any(outline_mask):
        sel = [r["verts"] for r, m in zip(recs, outline_mask) if m]
        ax.add_collection(PolyCollection(sel, facecolors="none", edgecolors=BLACK,
                                         linewidths=1.6, zorder=3))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def overview_figure(recs, stats, n_iter, n_valerr):
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), constrained_layout=True)
    axA, axB, axC, axD = axes.ravel()

    # A: ground truth
    faceA = [BLUE if r["det"] > 0 else ORANGE for r in recs]
    draw_tiling(axA, recs, faceA,
                title=f"A. true handedness, sign(det T), {len(recs)} tiles, {n_iter} iterations")
    axA.legend(handles=[Patch(color=BLUE, label="det > 0  (rotation)"),
                        Patch(color=ORANGE, label="det < 0  (reflection)")],
               loc="lower left", fontsize=8, framealpha=0.9)

    # B: what the Blender script flags
    faceB = [MAGENTA if r["blender_flip"] else GREY for r in recs]
    nb = sum(r["blender_flip"] for r in recs)
    nm = sum(r["mismatch"] for r in recs)
    draw_tiling(axB, recs, faceB, outline_mask=[r["mismatch"] for r in recs],
                title=(f"B. tiles flagged is_flip by spectre_tiles_blender.py "
                       f"(ITER={n_iter}, {'odd: scl == 1' if n_iter % 2 else 'even: scl != 1'}), "
                       f"{nb} flagged"))
    axB.legend(handles=[Patch(color=MAGENTA, label="is_flip = True"),
                        Patch(color=GREY, label="is_flip = False"),
                        Patch(facecolor="none", edgecolor=BLACK, linewidth=1.6,
                              label=f"trot_inv scl != sign(det)  ({nm})")],
               loc="lower left", fontsize=8, framealpha=0.9)

    # C: angle1 vs angle2 plane
    cnt = Counter((r["a1"], r["a2"], r["det"] > 0, r["scl"]) for r in recs)
    mx = max(cnt.values())
    xs = np.array([-180, 180])
    axC.plot(xs, xs, "--", color=BLUE, lw=1, label="angle1 = angle2   (rotation)")
    axC.plot(xs, xs - 180, "--", color=ORANGE, lw=1, label="angle1 = angle2 ± 180   (reflection)")
    axC.plot(xs, xs + 180, "--", color=ORANGE, lw=1)
    axC.plot(xs, -xs, ":", color="#555555", lw=1.2, label="angle1 = -angle2   (trot_inv branch 2 -> +1)")
    for (a1, a2, pos, scl), n in cnt.items():
        axC.scatter(a1, a2, s=50 + 450 * n / mx, marker="o" if scl == 1 else "X",
                    color=BLUE if pos else ORANGE, edgecolor=BLACK, linewidth=0.8, zorder=3)
        axC.annotate(str(n), (a1, a2), xytext=(7, 7), textcoords="offset points", fontsize=7)
    axC.set_xlim(-195, 195)
    axC.set_ylim(-195, 195)
    axC.set_xticks(range(-180, 181, 30))
    axC.set_yticks(range(-180, 181, 30))
    axC.tick_params(labelsize=7)
    axC.set_aspect("equal")
    axC.grid(alpha=0.25)
    axC.set_xlabel("angle1 = atan2(T[1,0], T[0,0])  [deg]")
    axC.set_ylabel("angle2 = atan2(-T[0,1], T[1,1])  [deg]")
    axC.set_title("C. the two angles inside trot_inv   (marker: circle = scl +1, X = scl -1;  "
                  "colour = true handedness;  label = tile count)", fontsize=9)
    leg = axC.legend(loc="upper left", fontsize=7, framealpha=0.9)
    axC.add_artist(leg)
    axC.legend(handles=[Line2D([], [], marker="o", ls="", color="w", markeredgecolor=BLACK, label="scl = +1"),
                        Line2D([], [], marker="X", ls="", color="w", markeredgecolor=BLACK, label="scl = -1"),
                        Patch(color=BLUE, label="det > 0"), Patch(color=ORANGE, label="det < 0")],
               loc="lower right", fontsize=7, framealpha=0.9)

    # D: per level percentages
    xs = np.arange(len(stats))
    w = 0.2
    series = [("det < 0 (truly reflected)", "reflected", ORANGE),
              ("trot_inv says scl = -1", "scl_neg", GREEN),
              ("trot_inv scl wrong", "mismatch", BLACK),
              ("Blender is_flip", "blender", MAGENTA)]
    for i, (name, key, col) in enumerate(series):
        vals = [100.0 * s[key] / s["tiles"] for s in stats]
        bars = axD.bar(xs + (i - 1.5) * w, vals, w, color=col, label=name)
        for b, v in zip(bars, vals):
            axD.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                         textcoords="offset points", ha="center", fontsize=7)
    axD.set_xticks(xs)
    axD.set_xticklabels([f"level {s['n']}\n({s['tiles']} tiles)" for s in stats], fontsize=8)
    axD.set_ylabel("% of tiles")
    axD.set_ylim(0, 112)
    axD.set_title("D. one mirror R per supertiling level: truth vs trot_inv vs Blender flag", fontsize=10)
    axD.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)
    axD.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "spectre.trot_inv(): where the +1 / -1 comes from"
        + (f"   ({n_valerr} 'ValueError' fall-through prints)" if n_valerr else ""),
        fontsize=13)
    return fig


# --------------------------------------------------------------------------
# gallery figure
# --------------------------------------------------------------------------
def pick_examples(recs, limit=4):
    seen = {}
    for r in recs:
        seen.setdefault((r["rot"], r["scl"], r["det"] > 0), r)
    pool = sorted(seen.values(), key=lambda r: (r["rot"] == 0, abs(r["rot"])))
    mism = [r for r in pool if r["mismatch"]]
    good_rot = [r for r in pool if not r["mismatch"] and r["det"] > 0]
    good_ref = [r for r in pool if not r["mismatch"] and r["det"] < 0]
    ex = mism[:2] + good_rot[:1] + good_ref[:1]
    chosen = {id(r) for r in ex}
    ex += [r for r in pool if id(r) not in chosen]
    return ex[:limit]


def frame(ax, M, color_x, color_y, L=9.0):
    for vec, col, name in (((1, 0), color_x, "x"), ((0, 1), color_y, "y")):
        v = M @ (L * np.array(vec, float))
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6), zorder=4)
        ax.text(v[0] * 1.12, v[1] * 1.12, name, color=col, fontsize=9,
                ha="center", va="center", zorder=4)


def stage(ax, pts, M, title, face, ref=None):
    p = pts @ M.T
    ax.fill(p[:, 0], p[:, 1], facecolor=face, edgecolor="#333333", lw=0.8, alpha=0.75)
    ax.plot(p[0, 0], p[0, 1], "o", color=BLACK, ms=5, zorder=5)       # vertex 0
    ax.text(p[1, 0], p[1, 1], " 1", fontsize=8, zorder=5)               # vertex 1
    frame(ax, M, "#555555", GREEN)
    if ref is not None:
        r = pts @ ref.T
        ax.plot(np.append(r[:, 0], r[0, 0]), np.append(r[:, 1], r[0, 1]), "--", color=BLACK, lw=1.4)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def gallery_figure(recs):
    ex = pick_examples(recs)
    fig, axes = plt.subplots(len(ex), 3, figsize=(12, 3.9 * len(ex)), constrained_layout=True,
                             squeeze=False)
    for row, r in zip(axes, ex):
        S = np.diag([1.0, float(r["scl"])])
        Rm = rot_lin(r["rot"])
        lin = r["T"][:, :2]
        face = BLUE if r["det"] > 0 else ORANGE
        pts = r["pts"]
        allp = np.vstack([pts, pts @ S.T, pts @ (Rm @ S).T, pts @ lin.T])
        lim = np.abs(allp).max() * 1.15
        stage(row[0], pts, np.eye(2), "1. local tile, local frame", GREY)
        stage(row[1], pts, S, f"2. times diag(1, scl={r['scl']})   (y flip)", GREY)
        ok = r["recon_err"] < 1e-6
        stage(row[2], pts, Rm @ S,
              f"3. then Rot({r['rot']}): "
              + ("matches real T" if ok else f"MISMATCH, err {r['recon_err']:.2f}"),
              face if ok else "#FFD0D0", ref=lin)
        for ax in row:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        row[0].set_ylabel(
            f"{r['label']}\nangle1={r['a1']}  angle2={r['a2']}\n"
            f"trot_inv scl={r['scl']}\n"
            f"det={'+' if r['det'] > 0 else '-'}1 -> true scl={r['true_scl']}"
            + ("\nWRONG" if r["mismatch"] else ""),
            fontsize=8, color=ORANGE if r["mismatch"] else BLACK)
    fig.suptitle("T_lin = Rot(angle1) * diag(1, scl): flip y first, then rotate   (dashed outline = real T)",
                 fontsize=12)
    return fig


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-n", "--iterations", type=int, default=spectre.N_ITERATIONS)
    ap.add_argument("--rotation", type=int, default=30, help="Gamma2 rotation (spectre default 30)")
    ap.add_argument("--edge-a", type=float, default=spectre.Edge_a)
    ap.add_argument("--edge-b", type=float, default=spectre.Edge_b)
    ap.add_argument("--out", default="spectre_flip", help="output file prefix")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--show", action="store_true", help="open interactive windows")
    args = ap.parse_args()

    stats = []
    recs = []
    n_valerr = 0
    for n in range(args.iterations + 1):
        recs, nv = collect(n, args.edge_a, args.edge_b, args.rotation)
        n_valerr = nv
        stats.append(level_stats(recs, n))
    print_table(stats)
    if n_valerr:
        print(f"trot_inv fell through to its error branch {n_valerr} times at level {args.iterations}")

    odd = Counter((r["rot"], r["scl"], r["det"] > 0) for r in recs if r["mismatch"])
    if odd:
        print("\nmismatches at the top level (rot, scl reported, det>0):")
        for k, v in sorted(odd.items()):
            print(f"  {k}: {v} tiles")

    fig1 = overview_figure(recs, stats, args.iterations, n_valerr)
    fig2 = gallery_figure(recs)
    if args.show:
        plt.show()
    else:
        p1 = f"{args.out}.png"
        p2 = f"{args.out}_gallery.png"
        fig1.savefig(p1, dpi=args.dpi)
        fig2.savefig(p2, dpi=args.dpi)
        print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
