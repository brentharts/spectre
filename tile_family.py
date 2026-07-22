#!/usr/bin/env python3
"""
tile_family.py -- the Tile(a,b) einstein family with PER-EDGE parameters.

    Spectre = Tile(1, 1)
    Hat     = Tile(1, sqrt(3))
    Turtle  = Tile(sqrt(3), 1)

Here we decompose the 14-gon into its 14 unit edge DIRECTIONS plus a per-edge
LENGTH vector, so each edge (or each vertex) can independently be
"spectre-like" or "hat-like".  This is what allows mixed Spectre/Hat tilings.

Key facts recovered numerically below and used throughout:
  * edge type sequence (a-edge vs b-edge) around the boundary
  * the a-edges alone sum to zero, and the b-edges alone sum to zero --
    so any *uniform* (a,b) closes, but per-edge mixtures generally do NOT
    close; the closure defect is one of the things we measure.
"""
import sys, os
import numpy as np
import spectre as S  # the cloned brentharts/spectre code

SQRT3 = np.sqrt(3.0)

# ---------------------------------------------------------------------------
# Canonical edge decomposition of Tile(a,b), derived from get_spectre_points
# ---------------------------------------------------------------------------

def _edge_decomposition():
    """Return (unit_dirs[14,2], edge_types[14]) with types 'a'/'b'.

    Derived numerically: perturb b and see which edges change length.
    Uses the vertex convention of spectre/spectre.py so everything stays
    compatible with its substitution transformations.
    """
    p1 = S.get_spectre_points(1.0, 1.0)
    p2 = S.get_spectre_points(1.0, 2.0)   # only b-edges grow

    def edges(pts):
        return np.roll(pts, -1, axis=0) - pts   # edge i: v[i] -> v[i+1]

    e1, e2 = edges(p1), edges(p2)
    L1 = np.linalg.norm(e1, axis=1)
    L2 = np.linalg.norm(e2, axis=1)
    types = np.where(np.abs(L2 - L1) > 1e-6, 'b', 'a')
    units = e1 / L1[:, None]
    return units.astype('float64'), types

UNIT_DIRS, EDGE_TYPES = _edge_decomposition()
N_EDGES = 14


def edge_lengths_from_params(a_len, b_len):
    """Uniform tile: length vector for Tile(a_len, b_len)."""
    return np.where(EDGE_TYPES == 'a', a_len, b_len).astype('float64')

# canonical per-edge length vectors
LEN_SPECTRE = edge_lengths_from_params(1.0, 1.0)
LEN_HAT     = edge_lengths_from_params(1.0, SQRT3)
LEN_TURTLE  = edge_lengths_from_params(SQRT3, 1.0)


def build_polygon(lengths, start=(0.0, 0.0), mirror_dirs=None):
    """Walk the 14 unit directions with per-edge `lengths`.

    Returns (verts[15,2], closure_defect_vec[2]).
    verts[14] is the endpoint of the walk; for a closed tile it equals
    verts[0].  The closure defect |verts[14]-verts[0]| measures how badly a
    per-edge mixture fails to be a closed polygon.
    """
    dirs = UNIT_DIRS if mirror_dirs is None else mirror_dirs
    verts = np.zeros((N_EDGES + 1, 2))
    verts[0] = start
    for i in range(N_EDGES):
        verts[i + 1] = verts[i] + dirs[i] * lengths[i]
    defect = verts[-1] - verts[0]
    return verts, defect


def polygon_area(verts):
    """Signed shoelace area of a (closed-by-convention) vertex loop."""
    x, y = verts[:-1, 0], verts[:-1, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ---------------------------------------------------------------------------
# Per-edge / per-vertex parameterisation ("spectre-ness" s in [0,1] per slot)
# ---------------------------------------------------------------------------
# s = 0 -> spectre edge scaling, s = 1 -> hat edge scaling.
# a-edges: both families have length 1 (we normalise a=1), so only b-edges
# actually interpolate: L_b = 1 + (sqrt(3)-1) * s.

def lengths_per_edge(s_edge):
    """s_edge: array of 14 values in [0,1] (0=spectre, 1=hat)."""
    s_edge = np.asarray(s_edge, dtype='float64')
    L = np.ones(N_EDGES)
    bmask = (EDGE_TYPES == 'b')
    L[bmask] = 1.0 + (SQRT3 - 1.0) * s_edge[bmask]
    return L


def lengths_per_vertex(s_vertex):
    """s_vertex: 14 values in [0,1], one per vertex.  Edge i runs from vertex
    i to vertex i+1; its parameter is the mean of its two endpoints."""
    s_vertex = np.asarray(s_vertex, dtype='float64')
    s_edge = 0.5 * (s_vertex + np.roll(s_vertex, -1))
    return lengths_per_edge(s_edge)


# ---------------------------------------------------------------------------
# Placement: reuse the substitution system of spectre/spectre.py
# ---------------------------------------------------------------------------

def placed_tiles(n_iterations=2, rotation=30):
    """Run the upstream substitution system and return a list of
    (T, label) affine placements.  T is the 2x3 matrix, label the tile name.
    Gamma2 is the 'Mystic' and uses Tile(b,a) i.e. swapped roles."""
    tiles = S.buildSpectreTiles(n_iterations, 1.0, 1.0, rotation=rotation)
    placed = []
    tiles['Delta'].forEachTile(lambda T, label: placed.append((T.copy(), label)))
    return placed


def transform_polygon(T, verts):
    """Apply upstream 2x3 affine T to an (N,2) vertex array."""
    return verts @ T[:, :2].T + T[:, 2]


def canonical_tile_verts(label):
    """Canonical (spectre) vertex loop for a placed tile label, in local
    coords, matching upstream conventions (Gamma2 = mystic = swapped a,b --
    identical when a==b==1, but kept for generality)."""
    pts = S.get_spectre_points(1.0, 1.0)
    return np.vstack([pts, pts[:1]])


if __name__ == '__main__':
    # sanity checks
    v, d = build_polygon(LEN_SPECTRE)
    print('spectre closure defect:', np.linalg.norm(d))
    print('spectre area:', polygon_area(v))
    v, d = build_polygon(LEN_HAT)
    print('hat     closure defect:', np.linalg.norm(d))
    print('hat     area:', polygon_area(v))
    print('edge types:', ''.join(EDGE_TYPES))
    # a-edges and b-edges each close independently:
    for t in 'ab':
        m = (EDGE_TYPES == t)
        print(f'sum of {t}-edge unit dirs:', UNIT_DIRS[m].sum(axis=0))
    n = len(placed_tiles(2))
    print('placed tiles @2 iterations:', n)
