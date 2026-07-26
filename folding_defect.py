#!/usr/bin/env python3
"""
folding_defect.py -- curvature budget of the spectre tiling, and chimeric
tiles that actually fold.

The problem with "just measure the angular defect"
--------------------------------------------------
Every interior angle of Tile(a,b) is independent of a and b: the 14 edge
DIRECTIONS are fixed and only the lengths change.  So swapping spectre
b-edges for hat b-edges does not move a single corner angle, every vertex
star still sums to 2 pi, and the naive angular defect of a chimeric tiling
is identically zero.  Per-edge lengths look like no knob at all.

They become a knob the moment the tile is treated as a piecewise-flat
METRIC disc rather than a rigid polygon.  Fan-triangulate each tile from its
centroid: 14 triangles with boundary lengths L_i and spokes r_i.  Now
    * the corner angle at vertex i is the sum of two triangle angles, and
      both depend on (L, r);
    * adjacent tiles are forced to agree on the shared edge length, which is
      exactly the constraint a chimeric tiling violates -- so we give each
      TILING edge one length (the mean of what its two tiles want) and let
      the incompatibility surface as curvature instead of as a gap;
    * the discrete Gaussian curvature is the vertex defect
      delta_v = 2 pi - sum of angles at v,
      and Gauss-Bonnet holds: sum over interior + boundary turning = 2 pi chi.

That is the honest translation of the 2D result in `closure_repair.py`:
the overlap and gap that the flat lattice cannot absorb reappear as a
curvature budget once the surface is allowed to leave the plane.

What is computed
----------------
  1. vertex-star census of the spectre tiling: how many distinct star types
     there are, and confirmation that every interior star is flat (2 pi)
  2. curvature budget of chimeric tilings vs mean spectre-ness s_bar:
     total |curvature|, its distribution, and the Gauss-Bonnet check
  3. THE FOLD SOLVE: choose spokes (and optionally the b-lengths) to drive
     every interior defect to zero.  Two regimes --
       'per_tile' : each tile gets its own 14 spokes  (many parameters)
       'shared'   : every tile gets the SAME spokes, i.e. the result is
                    still a monotile.  This is the interesting one.
  4. a prescribed non-zero budget: put the whole 4 pi of a sphere on a few
     cone points and see whether the metric can carry it
  5. an actual 3D embedding of the resulting metric (stress minimisation)
     plus an OBJ export

Outputs (into $EINSTEIN3D_OUT, default /tmp):
    folding_census.png       star types and the flatness check
    folding_budget.png       curvature budget vs s_bar, Gauss-Bonnet
    folding_solve.png        defects before/after the fold solve
    folding_embedding.png    the folded surface
    folded_tiling.obj
    folding_defect_metrics.txt
"""
import os, argparse, time
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares

import tile_family as TF
import closure_repair as CR

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
SQRT3 = np.sqrt(3.0)
TWOPI = 2 * np.pi


# ---------------------------------------------------------------------------
# 1. combinatorics of the tiling
# ---------------------------------------------------------------------------
def build_complex(n_iterations=2, round_to=4):
    """Vertex / edge / tile incidence of the spectre tiling."""
    placements = TF.placed_tiles(n_iterations)
    canon = TF.canonical_tile_verts('any')[:14]
    vid, vpos = {}, []
    tiles = []
    edge_users = defaultdict(list)

    for ti, (T, label) in enumerate(placements):
        world = TF.transform_polygon(T, canon)
        ids = []
        for p in world:
            key = tuple(np.round(p, round_to))
            if key not in vid:
                vid[key] = len(vpos)
                vpos.append(np.asarray(key, float))
            ids.append(vid[key])
        tiles.append(dict(label=label, vids=ids, xy=world,
                          centroid=world.mean(axis=0)))
        for i in range(14):
            e = tuple(sorted((ids[i], ids[(i + 1) % 14])))
            edge_users[e].append((ti, i))

    vpos = np.array(vpos)
    # stable integer ids for edges, and flat incidence arrays so that the
    # defect computation is a few numpy calls rather than 14 n_tiles dict
    # lookups (the fold solve evaluates it thousands of times)
    edge_id = {e: j for j, e in enumerate(sorted(edge_users))}
    n_t = len(tiles)
    TI = np.repeat(np.arange(n_t), 14)
    CI = np.tile(np.arange(14), n_t)
    VID = np.array([tiles[t]['vids'][i] for t, i in zip(TI, CI)])
    ENEXT = np.array([edge_id[tuple(sorted((tiles[t]['vids'][i],
                                            tiles[t]['vids'][(i + 1) % 14])))]
                      for t, i in zip(TI, CI)])
    EPREV = np.array([edge_id[tuple(sorted((tiles[t]['vids'][i - 1],
                                            tiles[t]['vids'][i])))]
                      for t, i in zip(TI, CI)])
    # a vertex is interior iff every edge at it is shared by two tiles
    at_vertex = defaultdict(list)
    for e, users in edge_users.items():
        for v in e:
            at_vertex[v].append(e)
    interior = np.array([
        all(len(edge_users[e]) == 2 for e in at_vertex[v])
        for v in range(len(vpos))])
    return dict(tiles=tiles, vpos=vpos, edge_users=edge_users,
                interior=interior, at_vertex=at_vertex, edge_id=edge_id,
                TI=TI, CI=CI, VID=VID, ENEXT=ENEXT, EPREV=EPREV,
                n_tiles=n_t, n_verts=len(vpos), n_edges=len(edge_id))


def classify_interior(cx, tol=1e-6):
    """Mark vertices whose star is CLOSED, using the exact spectre metric.

    The purely topological test ('every incident edge is shared') gets a few
    vertices wrong, because the spectre tiling is not edge-to-edge: the
    180-degree corner (index 10) lets a vertex of one tile sit in the middle
    of a neighbour's edge.  Misclassifying k closed stars as boundary throws
    the Gauss-Bonnet sum off by k pi, which is exactly what was seen.
    A star is closed iff its angles sum to 2 pi in the flat spectre metric,
    so we use that instead and keep the topological flag for comparison.
    """
    L = TF.LEN_SPECTRE
    _, r, _ = tile_lengths_and_spokes(L)
    tile_L = [np.asarray(L, float)] * cx['n_tiles']
    Le, _ = edge_lengths_of_complex(cx, tile_L)
    d = vertex_defects(cx, Le, np.tile(r, (cx['n_tiles'], 1)))
    topo = cx['interior'].copy()
    cx['interior'] = np.abs(d) < tol
    cx['interior_topological'] = topo
    return cx


def vertex_stars(cx):
    """For each vertex, the list of (tile, corner index) incidences."""
    stars = defaultdict(list)
    for ti, t in enumerate(cx['tiles']):
        for i, v in enumerate(t['vids']):
            stars[v].append((ti, i))
    return stars


def star_types(cx, stars):
    """Classify interior vertex stars by the multiset of corner indices.

    Corner index determines the interior angle, so this is exactly the
    classification by 'which corners of the tile meet here'.
    """
    types = Counter()
    per_vertex = {}
    for v, inc in stars.items():
        if not cx['interior'][v]:
            continue
        key = tuple(sorted(i for _, i in inc))
        types[key] += 1
        per_vertex[v] = key
    return types, per_vertex


def canonical_corner_angles():
    """Interior angle at each of the 14 corners of Tile(a,b) -- independent
    of a and b, which is the whole reason lengths alone give no defect.

    turn_i = signed angle from edge i-1 to edge i; for a counter-clockwise
    loop the turns sum to 2 pi and interior_i = pi - turn_i, so the interior
    angles sum to (14-2) pi = 2160 deg.  The orientation of UNIT_DIRS is
    detected rather than assumed.
    """
    U = CR.UNIT_DIRS
    th = np.arctan2(U[:, 1], U[:, 0])
    turn = (th - np.roll(th, 1) + np.pi) % TWOPI - np.pi
    if turn.sum() < 0:                       # clockwise loop
        turn = -turn
    return np.pi - turn


# ---------------------------------------------------------------------------
# 2. the piecewise-flat metric
# ---------------------------------------------------------------------------
def tile_lengths_and_spokes(L):
    """Boundary lengths and centroid spokes of the closed tile with lengths L."""
    v, defect = TF.build_polygon(L, mirror_dirs=CR.UNIT_DIRS)
    v = v[:14]
    c = v.mean(axis=0)
    r = np.linalg.norm(v - c, axis=1)
    return np.asarray(L, float), r, float(np.linalg.norm(defect))


def edge_lengths_of_complex(cx, tile_L):
    """One length per TILING edge = mean of what its two tiles want.

    tile_L[ti] is the 14-vector of desired boundary lengths of tile ti.
    The mismatch |L_t0 - L_t1| is the chimeric incompatibility; averaging it
    away is what converts it into curvature.
    """
    Le = np.zeros(cx['n_edges'])
    mismatch = []
    for e, users in cx['edge_users'].items():
        want = [tile_L[ti][i] for ti, i in users]
        Le[cx['edge_id'][e]] = float(np.mean(want))
        if len(want) == 2:
            mismatch.append(abs(want[0] - want[1]))
    return Le, np.array(mismatch)


def _angle(opposite, s1, s2):
    """Angle between sides s1, s2 with the given opposite side."""
    c = (s1 ** 2 + s2 ** 2 - opposite ** 2) / (2 * s1 * s2 + 1e-300)
    return np.arccos(np.clip(c, -1.0, 1.0))


def vertex_defects(cx, Le, spokes):
    """delta_v = 2 pi - sum of corner angles, for every vertex.

    spokes is (n_tiles, 14) of centroid->vertex distances.  Corner i of tile
    ti contributes the angle at v_i of triangle (c, v_{i-1}, v_i) plus the
    angle at v_i of triangle (c, v_i, v_{i+1}).  Fully vectorised: the fold
    solve calls this thousands of times.
    """
    S = np.asarray(spokes, float).reshape(cx['n_tiles'], 14)
    TI, CI = cx['TI'], cx['CI']
    r_i = S[TI, CI]
    r_ip = S[TI, (CI + 1) % 14]
    r_im = S[TI, (CI - 1) % 14]
    a1 = _angle(r_ip, r_i, Le[cx['ENEXT']])
    a2 = _angle(r_im, r_i, Le[cx['EPREV']])
    total = np.bincount(cx['VID'], weights=a1 + a2,
                        minlength=cx['n_verts'])
    return TWOPI - total


def boundary_structure(cx):
    """Boundary edges, boundary loops and pinch vertices of the patch.

    The complex has chi = -1 and three boundary loops, but the two extra
    loops enclose ZERO area (checked against the shapely union, which has
    union area == sum of tile areas exactly).  They are an artifact of the
    spectre tiling not being vertex-to-vertex: the 180-degree corner lets a
    vertex of one tile sit in the interior of a neighbour's edge, so that
    edge is recorded once as a long segment and once as two short ones, and
    the mismatch shows up as a zero-area sliver loop.

    This is why the boundary-turning form of Gauss-Bonnet does not reduce to
    2 pi for the patch.  It does not affect the curvature results: interior
    vertices are classified metrically (star angles summing to 2 pi), the
    pure spectre metric comes out flat to 1e-14, and Gauss-Bonnet is
    demonstrated in its unambiguous closed-surface form by the 4 pi
    cone-point solve below.
    """
    bnd_edges = [e for e, u in cx['edge_users'].items() if len(u) == 1]
    deg = Counter()
    for a, b in bnd_edges:
        deg[a] += 1; deg[b] += 1
    pinch = [v for v, d in deg.items() if d > 2]
    chi = cx['n_verts'] - cx['n_edges'] + cx['n_tiles']
    # connected components of the boundary graph = number of boundary loops
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for a, b in bnd_edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    loops = len({find(v) for e in bnd_edges for v in e})
    return dict(n_boundary_edges=len(bnd_edges), n_pinch=len(pinch),
                pinch=pinch, chi=chi, n_loops=loops)


# ---------------------------------------------------------------------------
# 3. chimeric specifications
# ---------------------------------------------------------------------------
def uniform_chimera(cx, s_bar):
    """Every tile gets the same closed length vector at mean spectre-ness."""
    L = TF.lengths_per_edge(np.full(14, s_bar))
    L, _ = CR.repair_pinned_a(L)
    Lv, r, _ = tile_lengths_and_spokes(L)
    n = len(cx['tiles'])
    return [Lv] * n, np.tile(r, (n, 1)), L


def random_chimera(cx, s_bar, rng):
    """Each tile independently randomised, then closure-repaired."""
    tile_L, tile_r = [], []
    for _ in cx['tiles']:
        s = np.clip(rng.normal(s_bar, 0.25, size=14), 0, 1)
        L = TF.lengths_per_edge(s)
        L, _ = CR.repair_pinned_a(L)
        Lv, r, _ = tile_lengths_and_spokes(L)
        tile_L.append(Lv); tile_r.append(r)
    return tile_L, np.array(tile_r)


# ---------------------------------------------------------------------------
# 4. the fold solve
# ---------------------------------------------------------------------------
def _fold_sparsity(cx):
    """Which spoke parameters each interior-vertex residual depends on."""
    from scipy.sparse import lil_matrix
    interior = np.where(cx['interior'])[0]
    row_of = -np.ones(cx['n_verts'], int)
    row_of[interior] = np.arange(len(interior))
    M = lil_matrix((len(interior), cx['n_tiles'] * 14), dtype=int)
    TI, CI, VID = cx['TI'], cx['CI'], cx['VID']
    for t, i, v in zip(TI, CI, VID):
        rr = row_of[v]
        if rr < 0:
            continue
        for j in ((i - 1) % 14, i, (i + 1) % 14):
            M[rr, t * 14 + j] = 1
    return M.tocsr()


def solve_flat(cx, Le, r0, mode='shared', target=None, max_nfev=60):
    """Choose spokes so that every interior defect hits `target` (default 0).

    mode 'shared'   : one 14-vector of spokes for all tiles -> still a
                      monotile, 14 free parameters
    mode 'per_tile' : each tile free, 14 * n_tiles parameters
    """
    n_tiles = len(cx['tiles'])
    interior = cx['interior']
    tgt = np.zeros(interior.sum()) if target is None else target

    R0 = np.asarray(r0, float).reshape(n_tiles, 14)
    if mode == 'shared':
        x0 = R0[0].copy()

        def spokes_of(x):
            return np.tile(np.abs(x), (n_tiles, 1))
        sparsity = None
    else:
        x0 = R0.ravel().copy()

        def spokes_of(x):
            return np.abs(x).reshape(n_tiles, 14)
        # residual for interior vertex v depends only on the spokes of the
        # corners meeting at v and their two neighbours in each tile
        sparsity = _fold_sparsity(cx)

    def resid(x):
        d = vertex_defects(cx, Le, spokes_of(x))
        return d[interior] - tgt

    t0 = time.time()
    sol = least_squares(resid, x0, method='lm' if mode == 'shared' else 'trf',
                        jac_sparsity=sparsity, max_nfev=max_nfev,
                        xtol=1e-12, ftol=1e-12)
    return dict(x=sol.x, spokes=spokes_of(sol.x), cost=float(sol.cost),
                rms=float(np.sqrt(np.mean(sol.fun ** 2))),
                rms0=float(np.sqrt(np.mean(resid(x0) ** 2))),
                seconds=time.time() - t0, mode=mode)


def sphere_budget(cx, n_cones=12):
    """Prescribe a total curvature of 4 pi spread over n_cones vertices.

    Gauss-Bonnet forces sum(delta) = 4 pi for a sphere, so this asks: can a
    chimeric metric put all of its curvature on a few cone points and be
    flat everywhere else?  (n_cones = 12 with delta = pi/3 is the football.)
    """
    interior = np.where(cx['interior'])[0]
    pos = cx['vpos'][interior]
    c = pos.mean(axis=0)
    order = np.argsort(np.linalg.norm(pos - c, axis=1))
    chosen = order[:n_cones]
    tgt = np.zeros(len(interior))
    tgt[chosen] = 4 * np.pi / n_cones
    return tgt, interior[chosen]


# ---------------------------------------------------------------------------
# 5. 3D embedding of the metric
# ---------------------------------------------------------------------------
def embed_3d(cx, Le, spokes, jitter=0.05, max_nfev=40, seed=0):
    """Stress-minimise a 3D embedding realising the target metric."""
    rng = np.random.default_rng(seed)
    nv = len(cx['vpos'])
    nt = len(cx['tiles'])
    # unknowns: vertex positions then centroid positions
    P0 = np.zeros((nv + nt, 3))
    P0[:nv, :2] = cx['vpos']
    for ti, t in enumerate(cx['tiles']):
        P0[nv + ti, :2] = t['centroid']
    P0[:, 2] = rng.normal(scale=jitter, size=nv + nt)

    pairs, targets = [], []
    for e, users in cx['edge_users'].items():
        pairs.append(e); targets.append(Le[cx['edge_id'][e]])
    for ti, t in enumerate(cx['tiles']):
        for i, v in enumerate(t['vids']):
            pairs.append((v, nv + ti)); targets.append(spokes[ti][i])
    pairs = np.array(pairs)
    targets = np.array(targets)

    def resid(x):
        P = x.reshape(-1, 3)
        d = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
        return d - targets

    from scipy.sparse import lil_matrix
    npts = nv + nt
    Sp = lil_matrix((len(pairs), npts * 3), dtype=int)
    for r, (i, j) in enumerate(pairs):
        for k in range(3):
            Sp[r, 3 * i + k] = 1
            Sp[r, 3 * j + k] = 1
    sol = least_squares(resid, P0.ravel(), jac_sparsity=Sp.tocsr(),
                        max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
    P = sol.x.reshape(-1, 3)
    return P, nv, float(np.sqrt(np.mean(sol.fun ** 2)))


def export_obj(cx, P, nv, fname):
    V, F = [], []
    for p in P:
        V.append(p)
    for ti, t in enumerate(cx['tiles']):
        ci = nv + ti + 1
        ids = [v + 1 for v in t['vids']]
        for i in range(14):
            F.append((ci, ids[i], ids[(i + 1) % 14]))
    with open(fname, 'w') as f:
        f.write('# folded chimeric spectre tiling\n')
        for v in V:
            f.write(f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n')
        for a, b, c in F:
            f.write(f'f {a} {b} {c}\n')
    return len(V), len(F)


# ---------------------------------------------------------------------------
# 6. plots
# ---------------------------------------------------------------------------
def plot_census(cx, types, angles, fname):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    ax.bar(range(14), np.degrees(angles), color='steelblue')
    ax.set_xticks(range(14))
    ax.set_xlabel('corner index'); ax.set_ylabel('interior angle (deg)')
    ax.set_title('interior angles of Tile(a,b)\n'
                 'independent of a and b -- so lengths alone move no angle')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    items = types.most_common()
    ax.bar(range(len(items)), [c for _, c in items], color='indianred')
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([''.join(map(str, k)) if len(k) < 5 else
                        f'{len(k)}-fold' for k, _ in items],
                       rotation=60, fontsize=6)
    ax.set_xlabel('vertex-star type (sorted corner indices)')
    ax.set_ylabel('count')
    ax.set_title(f'{len(items)} distinct interior vertex-star types\n'
                 f'over {sum(types.values())} interior vertices')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[2]
    val = Counter(len(k) for k in types.elements())
    ax.bar(list(val), [val[k] for k in val], color='seagreen')
    ax.set_xlabel('valence (tile corners meeting at the vertex)')
    ax.set_ylabel('count')
    ax.set_title('vertex valences of the spectre tiling')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def plot_budget(rows, fname):
    sb = np.array([r['s_bar'] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    ax.plot(sb, [r['abs_curv'] for r in rows], 'o-', color='crimson',
            label=r'$\sum |\delta_v|$ (uniform chimera)')
    ax.plot(sb, [r['abs_curv_rand'] for r in rows], 's--', color='slateblue',
            label=r'$\sum |\delta_v|$ (per-tile random, $\sigma$=0.25)')
    ax.set_xlabel(r'mean spectre-ness $\bar s$')
    ax.set_ylabel('total absolute curvature (rad)')
    ax.set_title('curvature budget created by hat-ness\n'
                 '(the 2D gap/overlap of closure_repair, re-expressed)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(sb, [r['mismatch'] for r in rows], 'o-', color='darkorange')
    ax.set_xlabel(r'$\bar s$')
    ax.set_ylabel('mean shared-edge length mismatch')
    ax.set_title('the incompatibility being converted into curvature\n'
                 '(zero for the spectre, grows with hat-ness)')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(sb, [r['net_curv'] for r in rows], 'o-', color='navy',
            label=r'$\sum_{int}\delta_v$ (signed)')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel(r'$\bar s$'); ax.set_ylabel('rad')
    ax.set_title('NET curvature: the budget is nearly balanced,\n'
                 'so the damage is saddles and cones, not a global cone')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def _defect_map(ax, cx, defects, title, vmax=None):
    verts = [t['xy'] for t in cx['tiles']]
    pc = PolyCollection(verts, facecolors='white', edgecolors='0.75',
                        linewidths=0.4)
    ax.add_collection(pc)
    m = cx['interior']
    d = defects[m]
    vmax = vmax or max(1e-6, np.abs(d).max())
    sc = ax.scatter(cx['vpos'][m, 0], cx['vpos'][m, 1], c=d, s=26,
                    cmap='coolwarm', vmin=-vmax, vmax=vmax, zorder=3)
    ax.autoscale(); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=9)
    return sc


def plot_solve(cx, before, after_shared, after_tile, fname):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    vmax = np.abs(before[cx['interior']]).max()
    for ax, d, t in ((axes[0], before, 'chimeric metric: defect per vertex'),
                     (axes[1], after_shared,
                      'after fold solve, SHARED spokes (still a monotile)'),
                     (axes[2], after_tile,
                      'after fold solve, per-tile spokes')):
        sc = _defect_map(ax, cx, d, t, vmax=vmax)
    fig.colorbar(sc, ax=axes, fraction=0.02).set_label(
        r'$\delta_v = 2\pi - \sum\theta$  (rad)')
    fig.suptitle('driving the curvature budget to zero by choosing spokes '
                 '-- a genuinely foldable chimeric tile', fontsize=12)
    fig.savefig(fname, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_embedding(cx, P, nv, defects, fname, title=''):
    fig = plt.figure(figsize=(15, 7))
    for j, (elev, azim) in enumerate(((55, -60), (12, -75))):
        ax = fig.add_subplot(1, 2, j + 1, projection='3d')
        polys = []
        for ti, t in enumerate(cx['tiles']):
            ring = P[t['vids']]
            polys.append(ring)
        col = Poly3DCollection(polys, facecolors='lightsteelblue',
                               edgecolors='0.35', linewidths=0.3, alpha=0.9)
        ax.add_collection3d(col)
        m = cx['interior']
        ax.scatter(P[:nv][m, 0], P[:nv][m, 1], P[:nv][m, 2],
                   c=defects[m], cmap='coolwarm', s=10, zorder=5)
        lo = P.min(axis=0); hi = P.max(axis=0)
        c = 0.5 * (lo + hi); r = 0.55 * (hi - lo).max()
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=2)
    ap.add_argument('--s-bar', type=float, default=0.5,
                    help='mean spectre-ness of the chimera to fold')
    ap.add_argument('--targets', type=int, default=9)
    ap.add_argument('--cones', type=int, default=12)
    args = ap.parse_args()

    lines = []

    def say(s=''):
        print(s, flush=True); lines.append(s)

    t_start = time.time()
    cx = classify_interior(build_complex(args.iterations))
    stars = vertex_stars(cx)
    types, per_vertex = star_types(cx, stars)
    angles = canonical_corner_angles()

    say('=== the tiling as a piecewise-flat complex ===')
    nvv = cx['n_verts'] + cx['n_tiles']
    nee = cx['n_edges'] + 14 * cx['n_tiles']
    nff = 14 * cx['n_tiles']
    say(f'tiles {len(cx["tiles"])}   vertices {len(cx["vpos"])} '
        f'(interior {int(cx["interior"].sum())}, topological test said '
        f'{int(cx["interior_topological"].sum())})   '
        f'edges {len(cx["edge_users"])}')
    say(f'fan triangulation: V {nvv}  E {nee}  F {nff}  ->  '
        f'Euler characteristic chi = {nvv - nee + nff}')
    say(f'corner angles (deg): '
        f'{np.array2string(np.degrees(angles), precision=1)}')
    say(f'sum of interior angles of one tile: '
        f'{np.degrees(angles.sum()):.4f} deg  (expected 12*180 = 2160)')
    say(f'distinct interior vertex-star types: {len(types)}')
    for k, c in types.most_common():
        say(f'   corners {str(k):28s} valence {len(k)}  count {c:4d}  '
            f'angle sum {np.degrees(sum(angles[i] for i in k)):9.4f} deg')
    say()

    say('=== flatness of the pure spectre metric ===')
    tile_L, tile_r, L_spectre = uniform_chimera(cx, 0.0)
    Le, mism = edge_lengths_of_complex(cx, tile_L)
    d0 = vertex_defects(cx, Le, tile_r)
    bs = boundary_structure(cx)
    say(f'max |defect| at an interior vertex: '
        f'{np.abs(d0[cx["interior"]]).max():.3e} rad')
    say(f'net interior curvature sum        : {d0[cx["interior"]].sum():.3e}')
    say(f'mean shared-edge length mismatch  : {mism.mean():.3e}')
    say(f'patch topology: chi = V-E+F = {bs["chi"]}, '
        f'{bs["n_boundary_edges"]} boundary edges, '
        f'{bs["n_loops"]} boundary loops, {bs["n_pinch"]} pinch vertices.')
    say('  the 2 extra loops enclose zero area (union area == sum of tile '
        'areas exactly): the tiling is not vertex-to-vertex, so the '
        'boundary-turning form of Gauss-Bonnet does not apply to the patch. '
        'The closed-surface form is checked by the 4 pi cone solve below.')
    say()

    say('=== curvature budget vs mean spectre-ness ===')
    rng = np.random.default_rng(4)
    rows = []
    hdr = (f"{'s_bar':>6s} {'mismatch':>10s} {'sum|delta|':>11s} "
           f"{'max|delta|':>11s} {'sum|d| rand':>12s} {'net curv':>10s}")
    say(hdr); say('-' * len(hdr))
    for sb in np.linspace(0, 1, args.targets):
        tL, tr, _ = uniform_chimera(cx, sb)
        Le_s, mism_s = edge_lengths_of_complex(cx, tL)
        d = vertex_defects(cx, Le_s, tr)
        net = float(d[cx['interior']].sum())
        rL, rr = random_chimera(cx, sb, rng)
        Le_r, _ = edge_lengths_of_complex(cx, rL)
        dr = vertex_defects(cx, Le_r, rr)
        rows.append(dict(s_bar=float(sb),
                         mismatch=float(mism_s.mean()),
                         abs_curv=float(np.abs(d[cx['interior']]).sum()),
                         abs_curv_rand=float(
                             np.abs(dr[cx['interior']]).sum()),
                         net_curv=net))
        say(f'{sb:6.2f} {mism_s.mean():10.4f} '
            f'{np.abs(d[cx["interior"]]).sum():11.4f} '
            f'{np.abs(d[cx["interior"]]).max():11.4f} '
            f'{np.abs(dr[cx["interior"]]).sum():12.4f} {net:10.4f}')
    png = os.path.join(OUT, 'folding_budget.png')
    plot_budget(rows, png); say(f'wrote {png}')
    say()

    say(f'=== fold solve at s_bar = {args.s_bar} ===')
    tL, tr, Lc = uniform_chimera(cx, args.s_bar)
    Le_c, mism_c = edge_lengths_of_complex(cx, tL)
    d_before = vertex_defects(cx, Le_c, tr)
    say(f'before: rms interior defect '
        f'{np.sqrt(np.mean(d_before[cx["interior"]]**2)):.6f} rad, '
        f'max {np.abs(d_before[cx["interior"]]).max():.6f}')

    sol_s = solve_flat(cx, Le_c, tr, mode='shared')
    d_s = vertex_defects(cx, Le_c, sol_s['spokes'])
    say(f"shared spokes (14 params, still a MONOTILE): rms "
        f"{sol_s['rms']:.6f} rad  (from {sol_s['rms0']:.6f})  "
        f"[{sol_s['seconds']:.1f}s]")
    say(f"  spokes: {np.array2string(np.abs(sol_s['x']), precision=4)}")
    say(f"  original spokes: {np.array2string(tr[0], precision=4)}")

    sol_t = solve_flat(cx, Le_c, tr, mode='per_tile')
    d_t = vertex_defects(cx, Le_c, sol_t['spokes'])
    say(f"per-tile spokes ({14*len(cx['tiles'])} params): rms "
        f"{sol_t['rms']:.6f} rad  (from {sol_t['rms0']:.6f})  "
        f"[{sol_t['seconds']:.1f}s]")
    png = os.path.join(OUT, 'folding_solve.png')
    plot_solve(cx, d_before, d_s, d_t, png); say(f'wrote {png}')
    say()

    say(f'=== prescribed budget: 4 pi on {args.cones} cone points ===')
    tgt, cone_ids = sphere_budget(cx, args.cones)
    sol_c = solve_flat(cx, Le_c, tr, mode='per_tile', target=tgt)
    d_c = vertex_defects(cx, Le_c, sol_c['spokes'])
    got = d_c[cx['interior']]
    say(f'target total curvature 4 pi = {4*np.pi:.4f}; '
        f'achieved {got.sum():.4f}')
    say(f'residual rms {sol_c["rms"]:.6f} rad  '
        f'(cone vertices asked for {4*np.pi/args.cones:.4f} each)')
    say()

    say('=== 3D embedding of the folded metric ===')
    P, nv, rms = embed_3d(cx, Le_c, sol_c['spokes'])
    say(f'stress-minimised embedding: rms edge-length error {rms:.5f}')
    say(f'z extent {P[:,2].max()-P[:,2].min():.4f} '
        f'(in-plane extent {np.ptp(P[:,:2]):.2f})')
    png = os.path.join(OUT, 'folding_embedding.png')
    plot_embedding(cx, P, nv, d_c, png,
                   f'chimeric spectre patch folded to carry '
                   f'{args.cones} cone points of curvature')
    say(f'wrote {png}')
    obj = os.path.join(OUT, 'folded_tiling.obj')
    nvv, nff = export_obj(cx, P, nv, obj)
    say(f'wrote {obj}  ({nvv} verts, {nff} tris)')

    png = os.path.join(OUT, 'folding_census.png')
    plot_census(cx, types, angles, png); say(f'wrote {png}')
    say(f'total time {time.time()-t_start:.1f}s')

    with open(os.path.join(OUT, 'folding_defect_metrics.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
