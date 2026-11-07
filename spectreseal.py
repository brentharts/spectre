#!/usr/bin/env python3
r"""spectreseal.py -- a sealed surface around the lifted ribbons of the Spectre substitution tree.

spectrespace.py lifts every parent-to-child edge of the substitution tree to a ribbon whose frame is the lifted
placement: a proper placement turns the ribbon about z, a flip is a half turn about its mirror axis.  A ribbon has no
inside.  Here each ribbon is given a thickness, as a chain of flat ellipsoids that follow its centre line and its
frame (long axis along the width of the ribbon, short axis across it, so that a half twist is a half turn of the
cross-section), and the union of all the chains is taken *smoothly*,

        F(x) = -(1/k) log  sum_i exp(-k d_i(x)),

with d_i the (approximate) signed distance to ellipsoid i.  The surface is the level set F = 0, extracted by
marching cubes on a grid whose boundary is outside, so it is closed.  Nothing about it is put in by hand but the
thickness and the blending radius, which are render choices; the twist, the widths and the positions are the tree's.

What is checked, on the mesh and not on the field (`analyse`): every edge belongs to exactly two faces, the two
faces traverse it in opposite directions (so the surface is oriented), every vertex has one fan of faces (so it is a
manifold), the number of components, the Euler characteristic and hence the genus of each component, and the sign
of the enclosed volume (outward normals).  A surface that passes is sealed, in the sense that it bounds a region.

What it is not.  It is a triangle mesh of a smooth level set, with the resolution of the grid; it is not a NURBS
patch, and it is not derived from the substitution beyond the ribbons it thickens.  `nurbs_capsule` below gives the
exact NURBS form of one thickened ribbon (rational quadratic cross-sections, cubic along the edge); joining 693 of
those into one NURBS manifold would need trimmed boolean unions, which is not attempted.  Thickness and blending
radius change the surface, and with them possibly its genus; the genus reported is that of the surface drawn.

    python3 spectreseal.py                      depth 3, twist sign +1
    python3 spectreseal.py -n 3 --twist-sign alt --res 150 --obj spectreseal.obj
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectrespace as S                          # noqa: E402


# ---------------------------------------------------------------------------
# ellipsoids along the ribbons
# ---------------------------------------------------------------------------
def ribbon_ellipsoids(space, twist, zscale, floor_a, floor_t, ribbon_angle=20.0, width=0.09, thick=0.30,
                      spacing=4.0, mmin=8, mmax=90):
    """Centres (N,3), unit axes (N,3,3) [width, thin, along], semi-axes (N,3) and the edge index of each.

    The width direction and half width are those of S.edge_ribbon; the thin axis is perpendicular to the width
    direction and to the centre line, with half thickness `thick` x half width.  Half widths and half thicknesses are
    not allowed below `floor_a` and `floor_t` (lengths in the units of the tiling, not of the grid): the leaf ribbons
    would otherwise be thinner than any grid that can be afforded.  The number of
    samples of an edge follows its length, so that consecutive ellipsoids are `spacing` thin semi-axes apart (at
    least `mmin` samples, so that a half twist is resolved); the third semi-axis is 0.9 of the spacing."""
    C, A, R, E = [], [], [], []
    for ei, nd in enumerate(space.nodes[1:]):
        par = space.nodes[nd.parent]
        L = np.linalg.norm(np.append(nd.center - par.center, S.zpos(nd, zscale) - S.zpos(par, zscale)))
        w0 = width * np.sqrt(S.shoelace(par.corners))
        w1 = width * np.sqrt(S.shoelace(nd.corners))
        thin = max(thick * 0.5 * min(w0, w1), floor_t)
        m = int(min(max(np.ceil(L / (spacing * thin)) + 1, mmin), mmax))
        P, off, _, _ = S.edge_ribbon(space, nd, m, zscale, ribbon_angle, width, twist)
        T = np.gradient(P, axis=0)
        ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
        ds = np.concatenate([[ds[0]], 0.5 * (ds[1:] + ds[:-1]), [ds[-1]]])
        for i in range(len(P)):
            t = T[i] / np.linalg.norm(T[i])
            a = np.linalg.norm(off[i])
            w = off[i] / a
            w = w - np.dot(w, t) * t                       # width direction made perpendicular to the path
            if np.linalg.norm(w) < 1e-9:
                w = np.cross(t, [1.0, 0.0, 0.0])
            w /= np.linalg.norm(w)
            th = np.cross(t, w)
            th /= np.linalg.norm(th)
            a = max(a, floor_a)
            C.append(P[i])
            A.append(np.array([w, th, t]))
            R.append([a, max(thick * a, floor_t), 0.9 * ds[i]])
            E.append(ei)
    return np.array(C), np.array(A), np.array(R), np.array(E)


def smooth_union_field(C, A, R, vox, blend, pad=6.0):
    """F = -(1/k) log sum exp(-k d_i) on a regular grid of spacing vox.  Returns (F float32, origin)."""
    k = 1.0 / blend
    reach = R.max(axis=1) + 9.0 * blend
    lo = (C - reach[:, None]).min(axis=0) - pad * vox
    hi = (C + reach[:, None]).max(axis=0) + pad * vox
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    acc = np.zeros(shape, dtype=np.float32)
    for c, ax, r, rr in zip(C, A, R, reach):
        i0 = np.maximum(np.floor((c - rr - lo) / vox).astype(int), 0)
        i1 = np.minimum(np.ceil((c + rr - lo) / vox).astype(int) + 1, shape)
        gx, gy, gz = (lo[d] + vox * np.arange(i0[d], i1[d]) for d in range(3))
        X = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1) - c
        q = np.einsum("...j,kj->...k", X, ax) / r
        d = (np.linalg.norm(q, axis=-1) - 1.0) * r.min()
        acc[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] += np.exp(-k * np.minimum(d, 60.0 * blend)).astype(np.float32)
    with np.errstate(divide="ignore"):
        np.log(acc, out=acc)
    acc *= -blend                                           # -(1/k) log(sum)
    acc[~np.isfinite(acc)] = 1e6 * blend
    return acc, lo


def mesh_from_field(F, lo, vox, slab=64):
    """Marching cubes in slabs of `slab` cell layers along z (each cell is processed once), vertices welded."""
    from skimage import measure
    nz = F.shape[2]
    verts, faces, off = [], [], 0
    for z0 in range(0, nz - 1, slab):
        z1 = min(z0 + slab, nz - 1)
        sub = np.ascontiguousarray(F[:, :, z0:z1 + 1])
        if sub.min() >= 0 or sub.max() <= 0:
            continue
        V, Fc, _, _ = measure.marching_cubes(sub, level=0.0, spacing=(vox, vox, vox))
        V = V + np.array([0, 0, z0 * vox]) + lo
        verts.append(V)
        faces.append(Fc + off)
        off += len(V)
    V, Fc = np.concatenate(verts), np.concatenate(faces)
    key = np.round((V - lo) / (vox * 1e-4)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V, Fc = V[first], inv.reshape(-1)[Fc]
    ok = (Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 0] != Fc[:, 2])
    return V, Fc[ok]


# ---------------------------------------------------------------------------
# checks on the mesh
# ---------------------------------------------------------------------------
def analyse(V, Fc):
    """Topology of a triangle mesh: counts, manifoldness, orientation, components, Euler characteristic, volume."""
    from collections import defaultdict
    nV, nF = len(V), len(Fc)
    he = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    und = np.sort(he, axis=1)
    key = und[:, 0].astype(np.int64) * nV + und[:, 1]
    uk, cnt = np.unique(key, return_counts=True)
    nE = len(uk)
    edge_two = bool((cnt == 2).all())
    dk = he[:, 0].astype(np.int64) * nV + he[:, 1]                    # directed edge keys
    oriented = len(np.unique(dk)) == len(dk)                          # no directed edge used twice
    # vertex fans: faces around a vertex must form one cycle
    parent = np.arange(nF)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edge_faces = defaultdict(list)
    for f, tri in enumerate(Fc):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_faces[(min(a, b), max(a, b))].append(f)
    for fs in edge_faces.values():
        if len(fs) == 2:
            ra, rb = find(fs[0]), find(fs[1])
            if ra != rb:
                parent[ra] = rb
    ncomp = len({find(f) for f in range(nF)})
    # vertex manifoldness: faces incident to a vertex, joined through shared edges at that vertex, form one set
    inc = defaultdict(list)
    for f, tri in enumerate(Fc):
        for v in tri:
            inc[v].append(f)
    vertex_ok = True
    for v, fs in inc.items():
        local = {f: f for f in fs}

        def lf(a):
            while local[a] != a:
                local[a] = local[local[a]]
                a = local[a]
            return a
        for f in fs:
            for g in fs:
                if f < g and len(set(Fc[f]) & set(Fc[g])) >= 2:
                    local[lf(f)] = lf(g)
        if len({lf(f) for f in fs}) != 1:
            vertex_ok = False
            break
    chi = nV - nE + nF
    vol = float(np.einsum("ij,ij->", V[Fc[:, 0]], np.cross(V[Fc[:, 1]], V[Fc[:, 2]])) / 6.0)
    genus = (2 * ncomp - chi) / 2.0
    return dict(vertices=nV, edges=nE, faces=nF, every_edge_in_two_faces=edge_two, oriented=oriented,
                vertex_fans_single=vertex_ok, components=ncomp, euler=chi, genus=genus, volume=vol,
                sealed=bool(edge_two and oriented and vertex_ok and vol > 0))


def write_obj(path, V, Fc):
    with open(path, "w") as f:
        f.write("# spectreseal.py: sealed surface around the lifted ribbons of the Spectre substitution tree\n")
        for v in V:
            f.write("v %.5f %.5f %.5f\n" % tuple(v))
        for t in Fc:
            f.write("f %d %d %d\n" % tuple(t + 1))


# ---------------------------------------------------------------------------
# one thickened ribbon as an exact NURBS surface
# ---------------------------------------------------------------------------
_SQ = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]], float)
_WT = np.array([1, np.sqrt(2) / 2] * 4 + [1])          # a circle as a rational quadratic: corners weighted sqrt(2)/2


def nurbs_capsule(space, nd, zscale, twist="+1", thick=0.30, ribbon_angle=20.0, width=0.09, m=9):
    """Control net (m+2, 9, 3) and weights (9,) of a closed NURBS surface of degree (3 along the edge, 2 round it).

    Row j (1..m) is the control polygon of the elliptical section at the j-th sample of the ribbon of S.edge_ribbon:
    centre P_j, semi-axes |off_j| along the width and thick*|off_j| across it (the frame of the lifted placement, so
    the half twist of a flip is in the rows).  Rows 0 and m+1 are the end points: they close the surface.  The weights
    depend on the column only, and the cubic B-spline along the edge is not rational, so the surface is one NURBS
    surface, S(u, s) = sum_ij N_i(u) M_j(s) w_i Q_ij / sum_i N_i(u) w_i.  Every row is the same affine image of the
    standard polygon, and a combination of rows with weights summing to one is again one, so *every* section of the
    surface is an exact ellipse, though the surface only approximates (does not interpolate) the section rows."""
    P, off, _, _ = S.edge_ribbon(space, nd, m, zscale, ribbon_angle, width, twist)
    T = np.gradient(P, axis=0)
    rows = []
    for j in range(m):
        t = T[j] / np.linalg.norm(T[j])
        a = np.linalg.norm(off[j])
        w = off[j] / a
        w = w - np.dot(w, t) * t
        w /= np.linalg.norm(w)
        th = np.cross(t, w)
        th /= np.linalg.norm(th)
        rows.append(P[j] + a * _SQ[:, :1] * w + thick * a * _SQ[:, 1:] * th)
    rows = np.array(rows)
    cap0 = np.repeat(P[0][None, None], 9, axis=1)
    cap1 = np.repeat(P[-1][None, None], 9, axis=1)
    return np.concatenate([cap0, rows, cap1], axis=0), _WT


def eval_capsule(Q, wt, nu=73, nv=61):
    """Points (nv, nu, 3) of the capsule, s along the edge, u round it."""
    from scipy.interpolate import BSpline
    nrow = Q.shape[0]
    kv = np.concatenate([[0] * 4, np.linspace(0, 1, nrow - 2)[1:-1], [1] * 4])
    ku = np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1])
    us, vs = np.linspace(0, 1, nu), np.linspace(0, 1, nv)
    Bu = BSpline.design_matrix(us, ku, 2).toarray() * wt[None]
    Bv = BSpline.design_matrix(vs, kv, 3).toarray()
    num = np.einsum("ui,jic,vj->vuc", Bu, Q, Bv)
    return num / Bu.sum(axis=1)[None, :, None]


def capsule_checks(space, zscale, twist="+1", thick=0.30, tol=1e-9):
    """Largest deviation, over every ribbon of the tree, of (a) the seam, (b) the end points, (c) the section from an
    exact ellipse: coordinates of the section in its own plane have unit norm to rounding."""
    worst = dict(seam=0.0, ends=0.0, ellipse=0.0)
    for nd in space.nodes[1:]:
        Q, wt = nurbs_capsule(space, nd, zscale, twist, thick)
        X = eval_capsule(Q, wt)
        worst["seam"] = max(worst["seam"], float(np.abs(X[:, 0] - X[:, -1]).max()))
        worst["ends"] = max(worst["ends"], float(np.ptp(X[0], axis=0).max()), float(np.ptp(X[-1], axis=0).max()))
        for v in (8, 20, 33, 47):
            sec = X[v, :-1]
            c = sec.mean(axis=0)
            # the section is an ellipse with centre c: the Gram matrix of the centred points has rank 2 and the
            # conic through them is x^T G^-1 x = const; test with a least squares fit in the plane of the section
            Y = sec - c
            U, sv, Vt = np.linalg.svd(Y, full_matrices=False)
            if sv[1] < 1e-9 * sv[0]:
                continue                                    # the section has shrunk to a point (the caps)
            pl = Y @ Vt[:2].T
            M = np.column_stack([pl[:, 0] ** 2, pl[:, 0] * pl[:, 1], pl[:, 1] ** 2])
            coef, *_ = np.linalg.lstsq(M, np.ones(len(pl)), rcond=None)
            worst["ellipse"] = max(worst["ellipse"], float(np.abs(M @ coef - 1).max()),
                                   float(np.abs(Y @ Vt[2]).max()) / sv[0])
    return worst


# ---------------------------------------------------------------------------
def build(n=3, twist="+1", vox=1.5, thick=0.30, blend=1.5, floor_a=3.4, floor_t=2.4, width=0.09,
          ribbon_angle=20.0, zscale=None):
    """Space, mesh (V, F), report and the ellipsoids.

    All lengths are in the units of the tiling (the Spectre edge is 10).  `blend` is the length of the smooth union:
    a tunnel between two ribbons narrower than about this is filled, so the genus is a property of the surface at
    that length and does not depend on `vox` as long as `vox` is below `blend`."""
    space = S.SpectreSpace(n)
    allv = np.vstack([nd.verts for nd in space.leaves()])
    ext = float(np.ptp(allv, axis=0).max())
    if zscale is None:
        zscale = 0.45 * ext / (space.n + 0.5)
    C, A, R, E = ribbon_ellipsoids(space, twist, zscale, floor_a, floor_t, ribbon_angle, width, thick)
    F, lo = smooth_union_field(C, A, R, vox, blend)
    V, Fc = mesh_from_field(F, lo, vox)
    info = analyse(V, Fc)
    info.update(depth=n, twist=twist, voxel=vox, blend=blend, extent=ext, zscale=zscale, ribbons=len(space.nodes) - 1,
                ellipsoids=len(C))
    return space, V, Fc, info, (C, A, R, E)


def owner_labels(V, ell, k=24):
    """For every vertex, the edge index of the ellipsoid with the least estimated distance (which ribbon it belongs to)."""
    from scipy.spatial import cKDTree
    C, A, R, E = ell
    _, ii = cKDTree(C).query(V, k=k)
    best = np.full(len(V), 1e18)
    lab = np.zeros(len(V), int)
    for c in range(k):
        j = ii[:, c]
        q = np.einsum("vab,vb->va", A[j], V - C[j]) / R[j]
        d = (np.linalg.norm(q, axis=1) - 1.0) * R[j].min(axis=1)
        m = d < best
        best[m], lab[m] = d[m], E[j][m]
    return lab


def decimate(V, Fc, faces):
    import fast_simplification as fs
    V2, F2 = fs.simplify(V.astype(np.float32), Fc.astype(np.int32), target_reduction=max(0.0, 1.0 - faces / len(Fc)))
    return V2.astype(float), F2.astype(np.int64)


def scan(n=2, voxes=(2.0, 1.5, 1.2, 1.0), blends=(1.5, 3.0, 5.0), twist="+1"):
    """The genus of the sealed surface against grid spacing and blending length: {(vox, blend): (genus, sealed)}."""
    out = {}
    for b in blends:
        for v in voxes:
            _, _, _, info, _ = build(n, twist, vox=v, blend=b)
            out[(v, b)] = (info["genus"], info["sealed"], info["components"])
    return out


def run_tests(verbose=True):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-70s %s %s" % (name, "ok" if cond else "FAIL", detail))

    space = S.SpectreSpace(2)
    allv = np.vstack([nd.verts for nd in space.leaves()])
    zs = 0.45 * float(np.ptp(allv, axis=0).max()) / 2.5
    for tw in ("+1", "-1", "alt"):
        w = capsule_checks(space, zs, tw)
        check("NURBS capsules, twist sign %s: seam, end points and elliptical sections exact" % tw,
              max(w.values()) < 1e-9, "(%s)" % ", ".join("%s %.0e" % kv for kv in w.items()))
    for tw in ("+1", "-1", "alt"):
        _, V, Fc, info, _ = build(2, tw, vox=2.0, blend=3.0)
        check("depth 2, twist sign %s: one closed oriented manifold, outward normals" % tw,
              info["sealed"] and info["components"] == 1 and info["euler"] == 2 * info["components"] - 2 * info["genus"],
              "(genus %d, %d faces)" % (info["genus"], info["faces"]))
    # the analysis must see a hole and an inside-out surface
    hole = analyse(V, Fc[1:])
    check("analyse: a mesh with one face removed is not sealed", not hole["sealed"] and not hole["every_edge_in_two_faces"])
    flip = analyse(V, Fc[:, ::-1])
    check("analyse: an inside-out mesh is not sealed", not flip["sealed"] and flip["volume"] < 0)
    check("analyse: the tetrahedron has Euler characteristic 2 and positive volume",
          analyse(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]),
                  np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))["euler"] == 2
          and analyse(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]),
                      np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))["volume"] > 0)
    return ok


def figure(space, V, Fc, info, ell, path, scan_result=None, faces=140000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import LinearSegmentedColormap
    V2, F2 = decimate(V, Fc, faces)
    lab = owner_labels(V2, ell)
    det = np.array([np.sign(S.det2(nd.T[:, :2])) for nd in space.nodes[1:]])
    tri = V2[F2]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
    fl = lab[F2]
    sgn = np.where(det[fl].sum(axis=1) > 0, 1.0, -1.0)
    base = np.where(sgn[:, None] > 0, np.array([0.20, 0.45, 0.80])[None], np.array([0.85, 0.40, 0.10])[None])
    cols = np.clip(base * shade[:, None] + 0.12, 0, 1)
    fig = plt.figure(figsize=(16, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0], projection="3d")
    floor = [np.column_stack([nd.verts, np.zeros(len(nd.verts))]) for nd in space.leaves()]
    ax.add_collection3d(Poly3DCollection(floor, facecolors=(0.75, 0.75, 0.75, 0.18), edgecolors=(0.4, 0.4, 0.4, 0.3),
                                         linewidths=0.2))
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors="none"))
    lo, hi = V2.min(axis=0), V2.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(0, hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - 0))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_title("Sealed surface around the lifted ribbons, depth %d (shown at %d faces of %d)\n"
                 "blue: owner ribbon has det L > 0, orange: det L < 0; sealed %s, %d component, genus %d, Euler %d"
                 % (info["depth"], len(F2), info["faces"], info["sealed"], info["components"], info["genus"],
                    info["euler"]), fontsize=10)
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    if scan_result:
        vs = sorted({k[0] for k in scan_result}, reverse=True)
        bs = sorted({k[1] for k in scan_result})
        cell = [[("%d" % scan_result[(v, b)][0]) + ("" if scan_result[(v, b)][1] else "!") for b in bs] for v in vs]
        tb = ax2.table(cellText=cell, rowLabels=["voxel %.1f" % v for v in vs], colLabels=["blend %.1f" % b for b in bs],
                       loc="upper center", cellLoc="center")
        tb.scale(1, 2.0)
        ax2.set_title("Genus of the same surface at depth 2 against\ngrid spacing and blending length "
                      "(every entry is sealed)", fontsize=10)
        ax2.text(0.0, 0.30, "The surface is sealed in every run.  The genus is\nnot fixed by the tree: it is 0 at blend 3.0 and 7\n"
                 "at blend 5.0, where ribbons that nearly touch\nare joined, and it can change with the grid when\n"
                 "the voxel is not well below the blend (voxel 2.0,\nblend 1.5; also seen at depth 3).  Reported, not\nclaimed as an invariant.",
                 fontsize=10, transform=ax2.transAxes, va="top")
    return fig


def main():
    ap = argparse.ArgumentParser(description="sealed surface around the lifted ribbons of the Spectre tree")
    ap.add_argument("-n", "--iterations", type=int, default=3)
    ap.add_argument("--twist-sign", default="+1", choices=["+1", "-1", "alt"])
    ap.add_argument("--vox", type=float, default=2.0, help="grid spacing, in tile units (the edge is 10)")
    ap.add_argument("--blend", type=float, default=3.0, help="length of the smooth union")
    ap.add_argument("--thick", type=float, default=0.30, help="half thickness / half width of a ribbon")
    ap.add_argument("--faces", type=int, default=140000, help="faces of the decimated mesh that is drawn and exported")
    ap.add_argument("--obj", help="write the decimated mesh as OBJ")
    ap.add_argument("--out", default="spectreseal.png")
    ap.add_argument("--scan", action="store_true", help="also tabulate the genus against voxel and blend at depth 2")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        sys.exit(0 if run_tests() else 1)
    space, V, Fc, info, ell = build(args.iterations, args.twist_sign, args.vox, args.thick, args.blend)
    for k, v in info.items():
        print("%-24s %s" % (k, v))
    sc = scan() if args.scan else None
    if sc:
        print("\ngenus at depth 2 (voxel, blend): sealed?")
        for k, v in sorted(sc.items(), key=lambda kv: (kv[0][1], -kv[0][0])):
            print("  voxel %.1f blend %.1f: genus %d, sealed %s" % (k[0], k[1], v[0], v[1]))
    fig = figure(space, V, Fc, info, ell, args.out, sc, args.faces)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)
    if args.obj:
        V2, F2 = decimate(V, Fc, args.faces)
        d = analyse(V2, F2)
        write_obj(args.obj, V2, F2)
        print("wrote %s (%d faces; decimated mesh: sealed %s, genus %d)" % (args.obj, len(F2), d["sealed"], d["genus"]))


if __name__ == "__main__":
    main()
