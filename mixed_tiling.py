#!/usr/bin/env python3
"""
mixed_tiling.py -- tilings mixing Spectre and Hat, per-edge / per-vertex.

Upstream limitation this removes: in spectre/spectre.py the (a,b) edge
parameters are global -- one value for every tile in the tiling.  Here each
placed tile carries a per-edge (or per-vertex) "spectre-ness" vector
s in [0,1]^14 (0 = spectre scaling, 1 = hat scaling), so a single tiling can
contain spectre tiles, hat tiles, and chimeric tiles that are hat on some
edges and spectre on others.

Because mixed tiles no longer fit the substitution combinatorics exactly, we
anchor every mixed polygon at the centroid of its canonical placement and
MEASURE the damage:

  * per-tile closure defect (mixed 14-gons need not close)
  * per-tile area and per-edge length statistics
  * overlap area  = sum(tile areas) - area(union)
  * gap area      = interior holes of the union
  * coverage      = union area / canonical spectre union area

Modes:
  spectre     -- all tiles s=0                       (baseline, gap-free)
  hat         -- all tiles s=1                       (baseline; hats on the
                 spectre substitution lattice -> overlaps, since hat area is
                 larger but placements are spectre-spaced)
  per_tile    -- each tile all-spectre or all-hat at random (what you could
                 almost do before, but in one tiling)
  per_edge    -- every edge of every tile independently spectre or hat
  per_vertex  -- every vertex gets s in {0,1}; edges average their endpoints
  gradient    -- per-vertex s varies smoothly across the plane (spectre on
                 the left morphing into hat on the right)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from shapely.ops import unary_union

import tile_family as TF

RNG = np.random.default_rng(7)
OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')


# ---------------------------------------------------------------------------
def s_vectors_for_mode(mode, placements):
    """Yield (s_edge[14] or None, s_vertex[14] or None) per placed tile."""
    n = len(placements)
    canon = TF.canonical_tile_verts('any')          # local canonical verts

    for i, (T, label) in enumerate(placements):
        if mode == 'spectre':
            yield np.zeros(14), None
        elif mode == 'hat':
            yield np.ones(14), None
        elif mode == 'per_tile':
            yield np.full(14, float(RNG.integers(0, 2))), None
        elif mode == 'per_edge':
            yield RNG.integers(0, 2, size=14).astype(float), None
        elif mode == 'per_vertex':
            yield None, RNG.integers(0, 2, size=14).astype(float)
        elif mode == 'gradient':
            world = TF.transform_polygon(T, canon)[:14]
            x = world[:, 0]
            yield None, None, x                     # handled by caller
        else:
            raise ValueError(mode)


def build_mixed_tiling(mode, n_iterations=2):
    """Return list of dict records, one per placed tile."""
    placements = TF.placed_tiles(n_iterations)
    canon = TF.canonical_tile_verts('any')          # 15x2 closed loop
    canon_centroid = canon[:14].mean(axis=0)

    # world-space x range for the gradient mode
    if mode == 'gradient':
        allx = np.concatenate([TF.transform_polygon(T, canon)[:, 0]
                               for T, _ in placements])
        x0, x1 = allx.min(), allx.max()

    records = []
    for T, label in placements:
        world_canon = TF.transform_polygon(T, canon)
        anchor = world_canon[:14].mean(axis=0)      # canonical centroid

        # --- choose per-slot spectre-ness ---------------------------------
        if mode == 'spectre':
            L = TF.lengths_per_edge(np.zeros(14)); s_desc = np.zeros(14)
        elif mode == 'hat':
            L = TF.lengths_per_edge(np.ones(14)); s_desc = np.ones(14)
        elif mode == 'per_tile':
            s = np.full(14, float(RNG.integers(0, 2)))
            L = TF.lengths_per_edge(s); s_desc = s
        elif mode == 'per_edge':
            s = RNG.integers(0, 2, size=14).astype(float)
            L = TF.lengths_per_edge(s); s_desc = s
        elif mode == 'per_vertex':
            sv = RNG.integers(0, 2, size=14).astype(float)
            L = TF.lengths_per_vertex(sv)
            s_desc = 0.5 * (sv + np.roll(sv, -1))
        elif mode == 'gradient':
            sv = np.clip((world_canon[:14, 0] - x0) / (x1 - x0), 0, 1)
            L = TF.lengths_per_vertex(sv)
            s_desc = 0.5 * (sv + np.roll(sv, -1))
        else:
            raise ValueError(mode)

        # --- build local mixed polygon, then place it ---------------------
        # The substitution T may include a mirror; applying T to the locally
        # built polygon handles that automatically since we build in the
        # same local frame as the canonical points.
        local, defect = TF.build_polygon(L, start=canon[0])
        # re-anchor: match centroids so mismatch is distributed evenly
        local = local - local[:14].mean(axis=0) + canon_centroid
        world = TF.transform_polygon(T, local)
        # close the loop explicitly for area/shapely purposes
        world_closed = np.vstack([world[:14], world[:1]])

        poly = Polygon(world[:14])
        if not poly.is_valid:
            poly = poly.buffer(0)                   # heal self-intersections

        records.append(dict(
            label=label, T=T, verts=world, poly=poly,
            closure_defect=float(np.linalg.norm(defect)),
            area=float(abs(TF.polygon_area(world_closed))),
            edge_lengths=L.copy(),
            s=s_desc.copy(),
        ))
    return records


# ---------------------------------------------------------------------------
def measure(records, reference_union=None):
    polys = [r['poly'] for r in records if not r['poly'].is_empty]
    union = unary_union(polys)
    sum_area = float(sum(p.area for p in polys))
    union_area = float(union.area)
    overlap = sum_area - union_area

    # interior gaps = holes of the union
    def holes_area(geom):
        tot = 0.0
        geoms = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for g in geoms:
            for ring in g.interiors:
                tot += Polygon(ring).area
        return tot
    gaps = holes_area(union)

    m = dict(
        n_tiles=len(records),
        sum_tile_area=sum_area,
        union_area=union_area,
        overlap_area=overlap,
        overlap_frac=overlap / sum_area,
        gap_area=gaps,
        gap_frac=gaps / union_area,
        mean_tile_area=float(np.mean([r['area'] for r in records])),
        std_tile_area=float(np.std([r['area'] for r in records])),
        mean_edge_len=float(np.mean([r['edge_lengths'] for r in records])),
        mean_closure_defect=float(np.mean([r['closure_defect'] for r in records])),
        max_closure_defect=float(np.max([r['closure_defect'] for r in records])),
    )
    if reference_union is not None:
        m['coverage_of_spectre_region'] = float(
            union.intersection(reference_union).area / reference_union.area)
    return m, union


# ---------------------------------------------------------------------------
def render(records, mode, metrics, fname):
    fig, ax = plt.subplots(figsize=(10, 10))
    verts = [r['verts'][:14] for r in records]
    # colour tiles by mean spectre-ness: blue=spectre, red=hat
    svals = np.array([r['s'].mean() for r in records])
    pc = PolyCollection(verts, array=svals, cmap='coolwarm',
                        edgecolors='k', linewidths=0.5, alpha=0.85)
    pc.set_clim(0, 1)
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('mean spectre-ness s  (0 = Spectre, 1 = Hat)')
    ax.autoscale(); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(
        f"mode = {mode}   tiles = {metrics['n_tiles']}\n"
        f"overlap = {100*metrics['overlap_frac']:.2f}% of tile area   "
        f"gaps = {100*metrics['gap_frac']:.2f}% of union   "
        f"mean closure defect = {metrics['mean_closure_defect']:.3f}")
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def main():
    modes = ['spectre', 'hat', 'per_tile', 'per_edge', 'per_vertex', 'gradient']
    ref_union = None
    rows = []
    for mode in modes:
        records = build_mixed_tiling(mode, n_iterations=2)
        metrics, union = measure(records, reference_union=ref_union)
        if mode == 'spectre':
            ref_union = union                      # reference region
        png = os.path.join(OUT, f'mixed_{mode}.png')
        render(records, mode, metrics, png)
        rows.append((mode, metrics))
        print(f"\n=== {mode} ===")
        for k, v in metrics.items():
            print(f"  {k:28s} {v:.6g}" if isinstance(v, float) else f"  {k:28s} {v}")

    # summary table
    print('\n' + '=' * 100)
    hdr = f"{'mode':12s} {'overlap%':>9s} {'gap%':>7s} {'mean area':>10s} " \
          f"{'sd area':>8s} {'mean edge':>10s} {'closure':>9s}"
    print(hdr); print('-' * len(hdr))
    lines = [hdr]
    for mode, m in rows:
        line = (f"{mode:12s} {100*m['overlap_frac']:9.3f} {100*m['gap_frac']:7.3f} "
                f"{m['mean_tile_area']:10.4f} {m['std_tile_area']:8.4f} "
                f"{m['mean_edge_len']:10.4f} {m['mean_closure_defect']:9.4f}")
        print(line); lines.append(line)
    with open(os.path.join(OUT, 'mixed_metrics.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
