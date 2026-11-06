#!/usr/bin/python3
"""
spectrespace.py -- the Spectre substitution with full provenance, lifted into 3D

spectre.py keeps only the final 2x3 matrix of every tile.  How a tile got there
(which supertile it sits in, which slot, which mirror / rotation was applied at
which level) is thrown away.  This module keeps that tree.

Hierarchy
---------
Node          one supertile ('meta'), the base Gamma pair ('pair') or a single
              Spectre tile ('tile')
node.local    placement inside the parent INCLUDING the mirror R = diag(-1, 1)
              that buildSupertiles applies to every slot
node.T        global transform = product of the locals from the root down
node.corners  the 4 corner points of the node's quad: a natural bilinear /
              NURBS patch for every node at every level
node.center   mean of the centroids of all leaf tiles below the node

Lifting flips into 3D
---------------------
In the plane a reflection cannot be reached continuously from the identity.
In 3D it can.  L -> diag(L, det L) maps O(2) into SO(3) (a homomorphism, since
det is multiplicative) and a planar reflection across the line at angle a/2
becomes a half turn about that same line.  So every 2D flip becomes a half
twist about its mirror axis, and a root -> leaf chain of nodes becomes a ribbon
whose frame twists whenever the 2D transform flips.

The sign of the half twist (+pi or -pi) is NOT fixed by the 2D data.  That is a
free Z2 choice per flip; --twist-sign picks +1, -1, or 'alt' (alternate with
level).  It is the natural place for an arrow of time to enter.

Usage
-----
  python3 spectrespace.py                    # 3 iterations, writes spectrespace.png
  python3 spectrespace.py -n 2 --twist-sign alt --export tree.json
  python3 spectrespace.py --show
"""
import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (registers 3d)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#BBBBBB"

# ---------------------------------------------------------------------------
# 2D affine helpers: a transform is a 2x3 array [L | t]
# ---------------------------------------------------------------------------
IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
MIRROR = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # the "twist" R


def rot(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0]])


def apply(T, p):
    return T[:, :2] @ p + T[:, 2]


def mul(A, B):
    AB = A.copy()
    AB[:, :2] = A[:, :2] @ B[:, :2]
    AB[:, 2] = A[:, :2] @ B[:, 2] + A[:, 2]
    return AB


def angle1(L):
    """angle of the image of the x axis, degrees"""
    return float(np.rad2deg(np.arctan2(L[1, 0], L[0, 0])))


def det2(L):
    return float(L[0, 0] * L[1, 1] - L[0, 1] * L[1, 0])


def spectre_points(a, b):
    s3 = np.sqrt(3)
    a2, a32 = a * 0.5, a * s3 / 2
    b2, b32 = b * 0.5, b * s3 / 2
    return np.array([
        (0, 0),
        (a, 0),
        (a + a2, -a32),
        (a + a2 + b32, -a32 + b2),
        (a + a2 + b32, -a32 + b + b2),
        (a + a + a2 + b32, -a32 + b + b2),
        (a + a + a + b32, b + b2),
        (a + a + a, b + b),
        (a + a + a - b32, b + b - b2),
        (a + a + a2 - b32, a32 + b + b - b2),
        (a + a2 - b32, a32 + b + b - b2),
        (a2 - b32, a32 + b + b - b2),
        (0 - b32, b + b - b2),
        (0, b),
    ], dtype="float64")


# substitution rules (same table as spectre.buildSupertiles)
SUBST = {
    "Gamma":  ("Pi",  "Delta", None,  "Theta", "Sigma", "Xi",  "Phi",    "Gamma"),
    "Delta":  ("Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"),
    "Theta":  ("Psi", "Delta", "Pi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"),
    "Lambda": ("Psi", "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"),
    "Xi":     ("Psi", "Delta", "Pi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma"),
    "Pi":     ("Psi", "Delta", "Xi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma"),
    "Sigma":  ("Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Lambda", "Gamma"),
    "Phi":    ("Psi", "Delta", "Psi", "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"),
    "Psi":    ("Psi", "Delta", "Psi", "Phi",   "Sigma", "Psi", "Phi",    "Gamma"),
}


# ---------------------------------------------------------------------------
# the tree
# ---------------------------------------------------------------------------
@dataclass
class Node:
    id: int
    label: str
    kind: str                 # 'meta' | 'pair' | 'tile'
    height: int               # supertiling level (tiles and the base pair: 0)
    depth: int                # distance from the root
    slot: int                 # index in the parent's substitution, -1 for root
    parent: int               # parent id, -1 for root
    local: np.ndarray         # 2x3 placement in the parent frame (includes R)
    T: np.ndarray             # 2x3 global transform
    children: list = field(default_factory=list)
    center: np.ndarray = None
    corners: np.ndarray = None
    verts: np.ndarray = None  # tiles only
    count: int = 0
    _sum: np.ndarray = None


class SpectreSpace:
    def __init__(self, n, root="Delta", edge_a=10.0, edge_b=10.0, gamma_rot=30):
        self.n, self.root = n, root
        self.edge_a, self.edge_b, self.gamma_rot = edge_a, edge_b, gamma_rot
        self.pts = spectre_points(edge_a, edge_b)
        self.mpts = spectre_points(edge_b, edge_a)  # Gamma2: edges swapped
        p8 = self.pts[8]
        self.gamma2_local = mul(np.array([[1, 0, p8[0]], [0, 1, p8[1]]], float), rot(gamma_rot))

        # The quad grows with every level, so the 8 slot placements differ per
        # level: place[k] are the placements used to build height-k supertiles.
        self.quads = [self.pts[[3, 5, 7, 11]]]
        self.place, self.place_before = {}, {}
        for k in range(1, n + 1):
            before, after, sq = self._level(self.quads[k - 1])
            self.place_before[k], self.place[k] = before, after
            self.quads.append(sq)
        self._build()

    @staticmethod
    def _level(quad):
        total = 0
        rotation = rot(0)
        before = [rotation.copy()]
        tq = quad
        for ang, f, t in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1),
                          (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
            if ang != 0:
                total += ang
                rotation = rot(total)
                tq = np.array([apply(rotation, q) for q in quad])
            tt = IDENTITY.copy()
            tt[:, 2] = apply(before[-1], quad[f]) - tq[t]
            before.append(mul(tt, rotation))
        after = [mul(MIRROR, t) for t in before]  # <- the twist
        sq = np.array([apply(after[6], quad[2]), apply(after[5], quad[1]),
                       apply(after[3], quad[2]), apply(after[0], quad[1])])
        return before, after, sq

    def _build(self):
        self.nodes = []

        def rec(label, height, T, parent, slot, local):
            nd = Node(id=len(self.nodes), label=label, kind="meta", height=height,
                      depth=0 if parent < 0 else self.nodes[parent].depth + 1,
                      slot=slot, parent=parent, local=local, T=T)
            self.nodes.append(nd)
            if parent >= 0:
                self.nodes[parent].children.append(nd.id)
            if height == 0:
                if label == "Gamma":
                    nd.kind = "pair"
                    rec("Gamma1", 0, T.copy(), nd.id, 0, IDENTITY.copy())
                    rec("Gamma2", 0, mul(T, self.gamma2_local), nd.id, 1, self.gamma2_local)
                else:
                    nd.kind = "tile"
            else:
                for i, (cl, tr) in enumerate(zip(SUBST[label], self.place[height])):
                    if cl is not None:
                        rec(cl, height - 1, mul(T, tr), nd.id, i, tr)

        rec(self.root, self.n, IDENTITY.copy(), -1, -1, IDENTITY.copy())

        for nd in self.nodes:
            if nd.kind == "tile":
                pts = self.mpts if nd.label == "Gamma2" else self.pts
                nd.verts = pts @ nd.T[:, :2].T + nd.T[:, 2]
                q = pts[[3, 5, 7, 11]]
            elif nd.kind == "pair":
                q = self.quads[0]
            else:
                q = self.quads[nd.height]
            nd.corners = q @ nd.T[:, :2].T + nd.T[:, 2]

        for nd in reversed(self.nodes):  # children always have larger ids
            if nd.kind == "tile":
                nd.count, nd._sum = 1, nd.verts.mean(axis=0)
            else:
                nd.count = sum(self.nodes[c].count for c in nd.children)
                nd._sum = sum(self.nodes[c]._sum for c in nd.children)
            nd.center = nd._sum / nd.count

    def leaves(self):
        return [nd for nd in self.nodes if nd.kind == "tile"]

    def chain(self, nd):
        out = [nd]
        while out[-1].parent >= 0:
            out.append(self.nodes[out[-1].parent])
        return out[::-1]

    def to_json(self):
        rows = []
        for nd in self.nodes:
            rows.append({
                "id": nd.id, "label": nd.label, "kind": nd.kind, "height": nd.height,
                "depth": nd.depth, "slot": nd.slot, "parent": nd.parent,
                "children": nd.children,
                "local": {"angle1": angle1(nd.local[:, :2]), "det": det2(nd.local[:, :2]),
                          "t": nd.local[:, 2].tolist()},
                "T": nd.T.tolist(), "det": det2(nd.T[:, :2]),
                "center": nd.center.tolist(), "corners": nd.corners.tolist(),
            })
        return {"n": self.n, "root": self.root, "nodes": rows}


def verify_against_spectre(space):
    """Compare every leaf (label and 2x3 matrix) with spectre.py's own output."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import spectre
    with contextlib.redirect_stdout(io.StringIO()):
        tiles = spectre.buildSpectreTiles(space.n, space.edge_a, space.edge_b,
                                          rotation=space.gamma_rot)
    ref = []
    tiles[space.root].forEachTile(lambda T, lab: ref.append((lab, T.copy())))
    mine = space.leaves()
    if len(ref) != len(mine):
        return False, len(ref), len(mine), np.inf
    labels_ok = all(r[0] == m.label for r, m in zip(ref, mine))
    err = max(float(np.abs(r[1] - m.T).max()) for r, m in zip(ref, mine))
    return labels_ok and err < 1e-9, len(ref), len(mine), err


# ---------------------------------------------------------------------------
# 3D lift
# ---------------------------------------------------------------------------
def lift3(L):
    """O(2) -> SO(3):  diag(L, det L).  A reflection becomes a half turn."""
    M = np.eye(3)
    M[:2, :2] = L
    M[2, 2] = np.sign(det2(L))
    return M


def rodrigues(axis, theta):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def local_motion(L, s, twist_sign=1):
    """Frame of a child relative to its parent at fraction s in [0,1].
    s=1 equals lift3(L).  Rotations turn about z; reflections half-twist about
    their mirror axis (an in-plane line at angle a/2)."""
    a = np.arctan2(L[1, 0], L[0, 0])
    if det2(L) > 0:
        return rodrigues([0, 0, 1], a * s)
    u = [np.cos(a / 2), np.sin(a / 2), 0.0]
    return rodrigues(u, twist_sign * np.pi * s)


def shoelace(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def zpos(nd, zscale):
    return zscale * (nd.height + (0.5 if nd.kind == "pair" else 0.0))


def edge_ribbon(space, nd, m, zscale, ribbon_angle, width_scale, twist_sign_mode):
    """Sample the twisted ribbon from nd's parent down to nd."""
    par = space.nodes[nd.parent]
    sign = (-1) ** par.height if twist_sign_mode == "alt" else int(twist_sign_mode)
    L = nd.local[:, :2]
    Fp = lift3(par.T[:, :2])
    s = np.linspace(0, 1, m)
    h = 3 * s ** 2 - 2 * s ** 3            # zero velocity at the nodes -> C1 along chains
    xy = par.center[None] + (nd.center - par.center)[None] * h[:, None]
    z = zpos(par, zscale) + (zpos(nd, zscale) - zpos(par, zscale)) * s
    P = np.column_stack([xy, z])
    g = np.deg2rad(ribbon_angle)
    wdir = np.array([np.cos(g), np.sin(g), 0.0])
    w0 = width_scale * np.sqrt(shoelace(par.corners))
    w1 = width_scale * np.sqrt(shoelace(nd.corners))
    off, nz, frames = [], [], []
    for hi, si in zip(h, s):
        F = Fp @ local_motion(L, hi, sign)
        frames.append(F)
        w = w0 + (w1 - w0) * si
        off.append(0.5 * w * (F @ wdir))
        nz.append(F[2, 2])
    return P, np.array(off), np.array(nz), frames


def frame_error(space, twist_sign_mode):
    """Largest deviation between the end frame of each edge and lift3(child.T)."""
    worst = 0.0
    for nd in space.nodes[1:]:
        par = space.nodes[nd.parent]
        sign = (-1) ** par.height if twist_sign_mode == "alt" else int(twist_sign_mode)
        end = lift3(par.T[:, :2]) @ local_motion(nd.local[:, :2], 1.0, sign)
        worst = max(worst, float(np.abs(end - lift3(nd.T[:, :2])).max()))
    return worst


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(space, args):
    cmap = LinearSegmentedColormap.from_list("flip", [ORANGE, "#EEEEEE", BLUE])
    fig = plt.figure(figsize=(17, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
    ax = fig.add_subplot(gs[0], projection="3d")

    # the tiling on the floor
    floor = [np.column_stack([nd.verts, np.zeros(len(nd.verts))]) for nd in space.leaves()]
    ax.add_collection3d(Poly3DCollection(floor, facecolors=(0.75, 0.75, 0.75, 0.25),
                                         edgecolors=(0.35, 0.35, 0.35, 0.5), linewidths=0.3))

    quads, cols = [], []
    for nd in space.nodes[1:]:
        P, off, nz, _ = edge_ribbon(space, nd, args.samples, args.zscale,
                                    args.ribbon_angle, args.width, args.twist_sign)
        for i in range(len(P) - 1):
            quads.append([P[i] + off[i], P[i + 1] + off[i + 1],
                          P[i + 1] - off[i + 1], P[i] - off[i]])
            cols.append(cmap(0.5 * (0.5 * (nz[i] + nz[i + 1]) + 1)))
    ax.add_collection3d(Poly3DCollection(quads, facecolors=cols, edgecolors=(0, 0, 0, 0.12),
                                         linewidths=0.2, alpha=0.9))
    allp = np.vstack([nd.verts for nd in space.leaves()])
    zmax = args.zscale * (space.n + 0.5)
    ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
    ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), zmax))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_zticks([args.zscale * k for k in range(space.n + 1)])
    ax.set_zticklabels([f"h={k}" for k in range(space.n + 1)])
    ax.set_title(f"Twisted ribbons: every parent->child edge, {len(space.leaves())} tiles on the floor.\n"
                 f"Blue = frame normal up (+z), orange = down; a 2D flip is a half twist about its mirror axis "
                 f"(twist sign: {args.twist_sign})", fontsize=10)

    # heat table: global rotation angle per level
    ax2 = fig.add_subplot(gs[1])
    cnt = Counter()
    for nd in space.nodes:
        if nd.kind == "pair":
            continue
        a = int(round(angle1(nd.T[:, :2])))
        a = 180 if a == -180 else a
        cnt[(nd.height, a, det2(nd.T[:, :2]) > 0)] += 1
    mx = max(cnt.values())
    for (h, a, pos), c in cnt.items():
        ax2.scatter(a, h, s=60 + 700 * c / mx, color=BLUE if pos else ORANGE,
                    edgecolor="black", linewidth=0.6, zorder=3)
        ax2.annotate(str(c), (a, h), xytext=(0, 0), textcoords="offset points",
                     ha="center", va="center", fontsize=6, zorder=4)
    ax2.set_xticks(range(-180, 181, 30))
    ax2.tick_params(axis="x", labelsize=7)
    ax2.set_yticks(range(space.n + 1))
    ax2.set_ylim(-0.6, space.n + 0.6)
    ax2.set_xlim(-195, 195)
    ax2.grid(alpha=0.25)
    ax2.set_xlabel("angle1 of the global transform [deg]")
    ax2.set_ylabel("height (0 = tiles)")
    ax2.set_title("Parity is one global clock per level\n(blue det>0, orange det<0); "
                  "per-tile information is the angle", fontsize=10)
    return fig


def summary(space, twist_mode):
    print(f"nodes: {len(space.nodes)}  tiles: {len(space.leaves())}  "
          f"height {space.n}  max depth {max(n.depth for n in space.nodes)}")
    print("\nlevel-1 slot placements (what buildSupertiles injects, BEFORE and AFTER the mirror):")
    if space.n >= 1:
        for i, (b, a) in enumerate(zip(space.place_before[1], space.place[1])):
            print(f"  slot {i}: before angle1={angle1(b[:, :2]):7.1f} det={det2(b[:, :2]):+.0f}"
                  f"   after angle1={angle1(a[:, :2]):7.1f} det={det2(a[:, :2]):+.0f}")
    g2 = space.gamma2_local
    print(f"  Gamma2 inside the base Gamma pair: angle1={angle1(g2[:, :2]):.1f} det={det2(g2[:, :2]):+.0f}"
          f"  (rotation only, edges a<->b swapped)")

    by_parent = Counter()
    for nd in space.nodes[1:]:
        by_parent[(space.nodes[nd.parent].kind, det2(nd.local[:, :2]) > 0)] += 1
    print("\nlocal det by parent kind (det>0?):")
    for k, v in sorted(by_parent.items()):
        print(f"  parent {k[0]:5s} det>0={k[1]!s:5s}: {v} edges")

    print("\nper height: nodes, distinct det of global T, distinct global angles")
    for h in range(space.n, -1, -1):
        sel = [nd for nd in space.nodes if nd.height == h and nd.kind != "pair"]
        dets = sorted({int(np.sign(det2(nd.T[:, :2]))) for nd in sel})
        angs = sorted({int(round(angle1(nd.T[:, :2]))) for nd in sel})
        print(f"  h={h}: {len(sel):4d} nodes  det {dets}  angles {angs}")
    print(f"\nframe continuity: max |F_parent*motion(1) - lift3(T_child)| = "
          f"{frame_error(space, twist_mode):.2e}")


def main():
    ap = argparse.ArgumentParser(description="Spectre substitution tree lifted into 3D")
    ap.add_argument("-n", "--iterations", type=int, default=3)
    ap.add_argument("--root", default="Delta", choices=sorted(SUBST))
    ap.add_argument("--rotation", type=int, default=30)
    ap.add_argument("--twist-sign", default="+1", choices=["+1", "-1", "alt"],
                    help="direction of each half twist (not fixed by the 2D data)")
    ap.add_argument("--ribbon-angle", type=float, default=20.0,
                    help="ribbon width direction in the local frame, deg (render choice)")
    ap.add_argument("--width", type=float, default=0.09, help="ribbon width / sqrt(quad area)")
    ap.add_argument("--zscale", type=float, default=None,
                    help="world units per level (default: total height = 0.45 x tiling extent)")
    ap.add_argument("--samples", type=int, default=14)
    ap.add_argument("--elev", type=float, default=24)
    ap.add_argument("--azim", type=float, default=-58)
    ap.add_argument("--out", default="spectrespace.png")
    ap.add_argument("--export", help="write the node tree as JSON")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    space = SpectreSpace(args.iterations, root=args.root, gamma_rot=args.rotation)
    if args.zscale is None:
        allv = np.vstack([nd.verts for nd in space.leaves()])
        args.zscale = 0.45 * float(np.ptp(allv, axis=0).max()) / (space.n + 0.5)
    ok, n_ref, n_mine, err = verify_against_spectre(space)
    print(f"verify vs spectre.py: {'OK' if ok else 'MISMATCH'}  "
          f"({n_mine} tiles here, {n_ref} there, max matrix error {err:.1e})\n")
    summary(space, args.twist_sign)

    if args.export:
        with open(args.export, "w") as f:
            json.dump(space.to_json(), f)
        print(f"\nwrote {args.export}")

    fig = figure(space, args)
    if args.show:
        plt.show()
    else:
        fig.savefig(args.out, dpi=args.dpi)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
