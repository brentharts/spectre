#!/usr/bin/env python3
"""
braided_tiling.py -- a first 3D form of the einstein tile: EDGE BRAIDING.

Instead of bending/folding the tile itself, take a flat
spectre tiling and lift the SHARED EDGES of adjacent tiles into z as two
interwoven strands.  Where two tiles meet, each contributes its own copy of
the common edge; we give the two copies opposite sinusoidal z-phases and a
small opposite in-plane offset, so along every shared edge the two strands
cross over/under each other k times -- a (2-strand, k-crossing) braid.
Boundary edges (no partner) stay flat at z=0.

Outputs:
  braided_tiles.png  -- matplotlib 3D render
  braided_tiles.obj  -- ribbon mesh (one ribbon per strand + flat tile faces),
                        loadable in Blender (cf. spectre_tiles_blender.py)

Choices you can tweak at the CLI:
  --crossings K   number of over/under crossings per shared edge (default 3)
  --height H      braid amplitude in z as a fraction of edge length
  --iterations N  substitution depth (default 1 -> 9 tiles)
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tile_family as TF

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')


# ---------------------------------------------------------------------------
def tiling_edges(n_iterations=1):
    """Place tiles, return per-tile world vertex loops and an edge table.

    Edge key = rounded (sorted endpoints); two tiles sharing a geometric edge
    produce the same key with opposite orientation.
    """
    placements = TF.placed_tiles(n_iterations)
    canon = TF.canonical_tile_verts('any')
    tiles = []
    edge_map = {}           # key -> list of (tile_idx, edge_idx)
    for ti, (T, label) in enumerate(placements):
        world = TF.transform_polygon(T, canon)[:14]
        tiles.append(dict(label=label, verts=world))
        for ei in range(14):
            p, q = world[ei], world[(ei + 1) % 14]
            key = tuple(sorted([tuple(np.round(p, 4)), tuple(np.round(q, 4))]))
            edge_map.setdefault(key, []).append((ti, ei))
    return tiles, edge_map


# ---------------------------------------------------------------------------
def braid_z_profile(t, k, sign):
    """z(t) for one strand: sign * sin(pi*k*t); zero at both endpoints so
    strands rejoin the flat tiling at every vertex."""
    return sign * np.sin(np.pi * k * t)


def build_braided_geometry(tiles, edge_map, crossings=3, height=0.18,
                           offset=0.05, samples=24):
    """For every tile edge produce a sampled 3D strand.

    Shared edges: the two copies get opposite z-phase and opposite in-plane
    normal offset -> a 2-strand braid with `crossings` crossings.
    Returns strands: list of dict(tile, edge, pts[ns,3], shared: bool)
    """
    strands = []
    for key, users in edge_map.items():
        shared = len(users) == 2
        for rank, (ti, ei) in enumerate(users):
            v = tiles[ti]['verts']
            p, q = v[ei], v[(ei + 1) % 14]
            L = np.linalg.norm(q - p)
            t = np.linspace(0, 1, samples)
            xy = p[None, :] + (q - p)[None, :] * t[:, None]
            if shared:
                sign = +1 if rank == 0 else -1
                # in-plane unit normal (rotate edge dir by 90 deg)
                d = (q - p) / L
                n = np.array([-d[1], d[0]])
                # offset dies out at endpoints too
                w = np.sin(np.pi * t)
                xy = xy + sign * offset * L * w[:, None] * n[None, :]
                z = braid_z_profile(t, crossings, sign) * height * L
            else:
                z = np.zeros_like(t)
            pts = np.column_stack([xy, z])
            strands.append(dict(tile=ti, edge=ei, pts=pts, shared=shared))
    return strands


# ---------------------------------------------------------------------------
def strand_ribbon(pts, width):
    """Turn a 3D polyline into a flat-ish ribbon (two vertex rows) for OBJ."""
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    up = np.array([0, 0, 1.0])
    side = np.cross(d, up)
    nrm = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(nrm > 1e-8, side / np.maximum(nrm, 1e-12),
                    np.array([1.0, 0, 0]))
    a = pts + 0.5 * width * side
    b = pts - 0.5 * width * side
    return a, b


def export_obj(tiles, strands, fname, ribbon_width=0.06):
    """OBJ: flat tile polygons (fan-triangulated) + ribbon per strand."""
    V, F = [], []

    def add_vert(p):
        V.append(p); return len(V)

    # tile faces at z=0
    for t in tiles:
        c = t['verts'].mean(axis=0)
        ci = add_vert([c[0], c[1], 0.0])
        ring = [add_vert([p[0], p[1], 0.0]) for p in t['verts']]
        for i in range(14):
            F.append((ci, ring[i], ring[(i + 1) % 14]))

    # strand ribbons
    for s in strands:
        a, b = strand_ribbon(s['pts'], ribbon_width)
        ia = [add_vert(p.tolist()) for p in a]
        ib = [add_vert(p.tolist()) for p in b]
        for i in range(len(ia) - 1):
            F.append((ia[i], ib[i], ib[i + 1]))
            F.append((ia[i], ib[i + 1], ia[i + 1]))

    with open(fname, 'w') as f:
        f.write('# braided spectre tiling\n')
        for v in V:
            f.write(f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n')
        for a, b, c in F:
            f.write(f'f {a} {b} {c}\n')
    return len(V), len(F)


# ---------------------------------------------------------------------------
def render(tiles, strands, fname, elev=55, azim=-60):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # translucent flat tiles
    polys = [np.column_stack([t['verts'], np.zeros(14)]) for t in tiles]
    pc = Poly3DCollection(polys, facecolors='lightsteelblue',
                          edgecolors='none', alpha=0.35)
    ax.add_collection3d(pc)

    # strands: colour over-strand vs under-strand; boundary grey
    for s in strands:
        p = s['pts']
        if s['shared']:
            color = 'crimson' if p[:, 2].sum() >= 0 else 'navy'
            lw = 2.2
        else:
            color, lw = '0.55', 1.0
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, lw=lw)

    allv = np.vstack([t['verts'] for t in tiles])
    cx, cy = allv.mean(axis=0)
    r = 0.55 * (allv.max(axis=0) - allv.min(axis=0)).max()
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(-r * 0.4, r * 0.4)
    ax.set_box_aspect((1, 1, 0.4))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title('Braided spectre tiling: shared edges woven over/under in z\n'
                 '(red / blue = the two strands of each 2-braid, grey = boundary)')
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crossings', type=int, default=3)
    ap.add_argument('--height', type=float, default=0.18)
    ap.add_argument('--iterations', type=int, default=1)
    args = ap.parse_args()

    tiles, edge_map = tiling_edges(args.iterations)
    shared = sum(1 for u in edge_map.values() if len(u) == 2)
    boundary = sum(1 for u in edge_map.values() if len(u) == 1)
    print(f'tiles={len(tiles)}  shared edges={shared}  boundary edges={boundary}')
    # each shared edge is a 2-strand braid with `crossings` crossings:
    print(f'total crossings in weave = {shared * args.crossings}')

    strands = build_braided_geometry(tiles, edge_map,
                                     crossings=args.crossings,
                                     height=args.height)
    png = os.path.join(OUT, 'braided_tiles.png')
    obj = os.path.join(OUT, 'braided_tiles.obj')
    render(tiles, strands, png)
    nv, nf = export_obj(tiles, strands, obj)
    print(f'wrote {png}')
    print(f'wrote {obj}  ({nv} verts, {nf} tris)')


if __name__ == '__main__':
    main()
