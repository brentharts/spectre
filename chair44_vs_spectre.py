#!/usr/bin/env python3
"""chair44_vs_spectre.py - compare the 2D Spectre hierarchy with the 3D Chair44 hierarchy.

Three tables, all computed (not quoted) from this fork's spectre.py and chair44.py:

  1. Hierarchy      substitution matrix, inflation, orientations, tile-type frequencies
  2. Contacts       raw geometric contacts -> contacts realised in the hierarchy
  3. Boundary       interface/boundary growth per level and the implied boundary dimension

Definitions (kept parallel on purpose)
  Chair44  contact = relative pose of a second chair sharing >=1 whole unit face (lattice
           contacts; rotations AND mirror images in the raw count).  legal = exact
           complement of the pyramid decorations.  realised = occurs inside sigma_d patches.
  Spectre  contact = relative pose of a second spectre sharing >=1 whole edge (the tiling is
           edge-to-edge; tile edges all 10 for a=b).  raw = every non-overlapping placement
           (rotations and mirror images).  realised = occurs inside the hierarchy patches.
           The Spectre has no decoration, so there is no separate 'legal' column.

Usage:  python3 chair44_vs_spectre.py [--quick] [--spectre-depth 5] [--json out.json]
"""
import argparse
import contextlib
import io
import json
import math
import sys
from collections import Counter

import numpy as np

import chair44 as C

with contextlib.redirect_stdout(io.StringIO()):
    import spectre as S

try:
    from shapely.geometry import Polygon
except ImportError:           # raw Spectre placements need it; everything else does not
    Polygon = None

## ------------------------------------------------------------------ 1. hierarchy
# Same substitution table as spectre.py / spectre_tile_counter.py (Smith-Myers-Kaplan-
# Goodman-Strauss).  Rows of `RULES` list the 8 slots of a supertile (None = empty slot).
LABELS = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]
RULES = {
    "Gamma": ("Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma"),
    "Delta": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Theta": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Lambda": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Xi": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    "Pi": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    "Sigma": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"),
    "Phi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    "Psi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
}
TILES_PER_META = np.array([2 if l == "Gamma" else 1 for l in LABELS])   # Gamma = Gamma1+Gamma2


def spectre_matrix():
    M = np.zeros((9, 9), int)            # M[child, parent]
    for j, parent in enumerate(LABELS):
        for child in RULES[parent]:
            if child is not None:
                M[LABELS.index(child), j] += 1
    return M


def spectre_tile_counts(n):
    M = spectre_matrix()
    v = np.zeros(9, object)
    v[LABELS.index("Delta")] = 1
    for _ in range(n):
        v = M.astype(object).dot(v)
    return int(TILES_PER_META.astype(object).dot(v))


def spectre_patch(n):
    """Tiles of the n-level Delta supertile: list of (2x3 affine, label)."""
    tiles = S.buildSpectreTiles(n, 10.0, 10.0)
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        tiles["Delta"].forEachTile(lambda T, lab: out.append((T.copy(), lab)))
    return out


def spectre_vertices(patch):
    P = S.SPECTRE_POINTS
    return np.array([P @ T[:, :2].T + T[:, 2] for T, _ in patch])      # (N,14,2)


def hierarchy_table():
    M = spectre_matrix()
    w, vec = np.linalg.eig(M.astype(float))
    k = int(np.argmax(w.real))
    lam = float(w[k].real)
    freq = np.abs(vec[:, k].real)
    freq = freq * TILES_PER_META
    freq = freq / freq.sum()
    patch = spectre_patch(4)
    angles = Counter(int(round(math.degrees(math.atan2(T[1, 0], T[0, 0])))) % 360 for T, _ in patch)
    for n in (1, 2, 3, 4):
        assert spectre_tile_counts(n) == len(spectre_patch(n)), "matrix disagrees with geometry"
    chair_frames = {p[0] for p in C.sigma_depth(3)}
    rows = {
        "chair44": {"tile_types": 1, "children": 8, "tiles_by_level": [8 ** n for n in range(5)],
                    "inflation_tiles": 8.0, "linear_scale": 2.0, "scale_is_integer": True,
                    "distinct_orientations": len(chair_frames),
                    "orientation_group": "rotations of the cube (order 24)"},
        "spectre": {"tile_types": 9, "children": "7-8 per supertile",
                    "tiles_by_level": [1] + [spectre_tile_counts(n) for n in range(1, 5)],
                    "inflation_tiles": lam, "linear_scale": math.sqrt(lam),
                    "scale_is_integer": False, "distinct_orientations": len(angles),
                    "orientation_group": "rotations by multiples of 30 deg",
                    "pf_eigenvalue_check": abs(lam - (4 + math.sqrt(15))) < 1e-9,
                    "type_frequency_of_tiles": {l: round(float(f), 5) for l, f in zip(LABELS, freq)}},
    }
    return rows


## -------------------------------------------------------------------- 2. contacts
def _canon(poly):
    """Pose key: rounded vertex set (the spectre has no symmetry, so this fixes the pose)."""
    q = np.rint(np.asarray(poly) * 100).astype(np.int64)
    return q[np.lexsort((q[:, 1], q[:, 0]))].tobytes()


def spectre_raw_contacts():
    """Every non-overlapping placement of a second spectre sharing a whole edge with the
    root (rotations and mirror images)."""
    if Polygon is None:
        raise RuntimeError("shapely is required for raw Spectre placements (pip install shapely)")
    A = np.array(S.SPECTRE_POINTS)
    pa = Polygon(A)
    found = {}
    for mirror in (False, True):
        Q = A * np.array([-1.0, 1.0]) if mirror else A
        for i in range(14):
            a0, a1 = A[i], A[(i + 1) % 14]
            for j in range(14):
                q0, q1 = Q[j], Q[(j + 1) % 14]
                for t0, t1 in ((a0, a1), (a1, a0)):
                    ang = math.atan2(*(t1 - t0)[::-1]) - math.atan2(*(q1 - q0)[::-1])
                    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
                    B = Q @ R.T + (t0 - R @ q0)
                    key = _canon(B)
                    if key in found:
                        continue
                    pb = Polygon(B)
                    if not pb.is_valid or pa.intersection(pb).area > 1e-6:
                        continue
                    found[key] = mirror
    return found


def spectre_observed_contacts(n):
    """Ordered relative poses of edge-sharing tile pairs in the n-level patch, in the native
    frame of the first tile."""
    patch = spectre_patch(n)
    V = spectre_vertices(patch)
    N = len(V)
    Ts = np.array([T for T, _ in patch])                    # (N,2,3)
    ends_a = np.rint(V * 100).astype(np.int64)
    ends_b = np.roll(ends_a, -1, axis=1)
    swap = (ends_a[..., 0] > ends_b[..., 0]) | ((ends_a[..., 0] == ends_b[..., 0]) & (ends_a[..., 1] > ends_b[..., 1]))
    lo = np.where(swap[..., None], ends_b, ends_a)
    hi = np.where(swap[..., None], ends_a, ends_b)
    E = np.concatenate([lo, hi], axis=2).reshape(-1, 4)
    tile_of = np.repeat(np.arange(N), 14)
    uniq, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    assert cnt.max() <= 2, "an edge belongs to more than two tiles"
    order = np.argsort(inv, kind="stable")
    shared_groups = np.flatnonzero(cnt == 2)
    first = np.searchsorted(inv[order], shared_groups)
    ia = tile_of[order[first]]
    ib = tile_of[order[first + 1]]
    pairs = {(min(a, b), max(a, b)) for a, b in zip(ia, ib)}
    poses = set()
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            Rx = Ts[x][:, :2]
            local = (V[y] - Ts[x][:, 2]) @ np.linalg.inv(Rx).T       # y's vertices in x's frame
            poses.add(_canon(local))
    return poses, {"tiles": N, "adjacent_pairs": len(pairs),
                   "shared_edges": int((cnt == 2).sum()), "boundary_edges": int((cnt == 1).sum())}


def contacts_table(quick=False):
    raw = C.face_contacts()
    deco = C.Decoration()
    legal = C.legal_contacts(deco, raw=raw)
    obs_depths = (2,) if quick else (2, 3)
    chair_obs = {d: len(C.observed_contacts(C.sigma_depth(d))) for d in obs_depths}
    chair = {"raw": len(raw), "raw_mirror": sum(C.frame_det(p[0]) < 0 for p in raw),
             "legal_exact": len(legal), "legal_mirror": sum(C.frame_det(p[0]) < 0 for p in legal),
             "realised_by_depth": chair_obs, "legal_never_realised": len(legal) - max(chair_obs.values())}
    spec = {}
    if Polygon is not None:
        rawS = spectre_raw_contacts()
        spec["raw"] = len(rawS)
        spec["raw_mirror"] = sum(1 for m in rawS.values() if m)
        spec["raw_rotation_only"] = spec["raw"] - spec["raw_mirror"]
        real = {}
        for n in ((3,) if quick else (3, 4, 5)):
            obs, info = spectre_observed_contacts(n)
            assert set(obs) <= {k for k, m in rawS.items() if not m}, "observed pose not in raw set"
            real[n] = len(obs)
        spec["realised_by_depth"] = real
    return {"chair44": chair, "spectre": spec}


## --------------------------------------------------------------------- 3. boundary
def boundary_table(chair_max=4, spectre_max=5):
    chair = []
    for d in range(0, chair_max + 1):
        s = C.patch_surface_stats(C.sigma_depth(d))
        s["depth"] = d
        s["surface_is_24_4^d"] = s["surface_faces"] == 24 * 4 ** d
        chair.append(s)
    spec = []
    for n in range(1, spectre_max + 1):
        _, info = spectre_observed_contacts(n) if n <= 4 else (None, _edge_stats(n))
        info = dict(info)
        info["depth"] = n
        spec.append(info)
    lam = 4 + math.sqrt(15)
    for i, r in enumerate(spec):
        if i:
            p = spec[i - 1]
            r["tile_ratio"] = r["tiles"] / p["tiles"]
            r["boundary_ratio"] = r["boundary_edges"] / p["boundary_edges"]
            r["boundary_dim_est"] = math.log(r["boundary_ratio"]) / math.log(math.sqrt(lam))
    for i, r in enumerate(chair):
        if i:
            p = chair[i - 1]
            r["boundary_ratio"] = r["surface_faces"] / p["surface_faces"]
            r["boundary_dim_est"] = math.log(r["boundary_ratio"]) / math.log(2.0)
    B = [14] + [r["boundary_edges"] for r in spec]            # B(0)=14: one tile has 14 edges
    return {"chair44": chair, "spectre": spec, "spectre_recurrence": fit_boundary_recurrence(B)}


def fit_boundary_recurrence(B):
    """Fit B(n+2) = a B(n+1) + b B(n) + c from the first three triples, verify on the rest.
    Returns None when there are too few points."""
    if len(B) < 6:
        return None
    A = np.array([[B[n + 1], B[n], 1] for n in range(3)], float)
    a, b, c = np.linalg.solve(A, np.array([B[n + 2] for n in range(3)], float))
    resid = [abs(a * B[n + 1] + b * B[n] + c - B[n + 2]) for n in range(3, len(B) - 2)]
    root = float(max(np.roots([1, -a, -b]).real))
    return {"a": round(float(a), 6), "b": round(float(b), 6), "c": round(float(c), 6),
            "extra_checks": len(resid), "max_residual": max(resid) if resid else None,
            "growth_root": root, "golden_cubed": (2 + math.sqrt(5)),
            "dim_estimate": math.log(root) / math.log(math.sqrt(4 + math.sqrt(15)))}


def _edge_stats(n):
    V = spectre_vertices(spectre_patch(n))
    a = np.rint(V * 100).astype(np.int64)
    b = np.roll(a, -1, axis=1)
    swap = (a[..., 0] > b[..., 0]) | ((a[..., 0] == b[..., 0]) & (a[..., 1] > b[..., 1]))
    E = np.concatenate([np.where(swap[..., None], b, a), np.where(swap[..., None], a, b)], axis=2).reshape(-1, 4)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    assert cnt.max() <= 2
    return {"tiles": len(V), "shared_edges": int((cnt == 2).sum()), "boundary_edges": int((cnt == 1).sum())}


## ------------------------------------------------------------------------ report
def fmt(x, d=4):
    return ("%.*f" % (d, x)) if isinstance(x, float) else str(x)


def report(h, c, b):
    L = []
    L.append("=" * 78)
    L.append("1. HIERARCHY")
    L.append("=" * 78)
    for name in ("chair44", "spectre"):
        r = h[name]
        L.append("%-8s types=%s children=%s inflation(tiles/level)=%s linear scale=%s %s" % (
            name, r["tile_types"], r["children"], fmt(r["inflation_tiles"]), fmt(r["linear_scale"]),
            "(integer)" if r["scale_is_integer"] else "(irrational)"))
        L.append("         tiles by level: %s" % r["tiles_by_level"])
        L.append("         orientations seen: %d  (%s)" % (r["distinct_orientations"], r["orientation_group"]))
    L.append("spectre PF eigenvalue == 4+sqrt(15): %s ; tile-type frequencies: %s" % (
        h["spectre"]["pf_eigenvalue_check"], h["spectre"]["type_frequency_of_tiles"]))
    L.append("")
    L.append("=" * 78)
    L.append("2. CONTACTS")
    L.append("=" * 78)
    ch, sp = c["chair44"], c["spectre"]
    L.append("chair44  raw=%d (mirror %d)  legal(exact decoration complement)=%d (mirror %d)" % (
        ch["raw"], ch["raw_mirror"], ch["legal_exact"], ch["legal_mirror"]))
    L.append("         realised inside sigma_d patches: %s  -> legal but never realised: %d" % (
        ch["realised_by_depth"], ch["legal_never_realised"]))
    if sp:
        L.append("spectre  raw=%d (mirror %d, rotation-only %d)  realised in patches: %s" % (
            sp["raw"], sp["raw_mirror"], sp["raw_rotation_only"], sp["realised_by_depth"]))
    L.append("")
    L.append("=" * 78)
    L.append("3. BOUNDARY (per refinement level)")
    L.append("=" * 78)
    L.append("chair44: depth tiles shared_faces surface_faces  ratio  dim_est   (surface == 24*4^d)")
    for r in b["chair44"]:
        L.append("   %2d %7d %12d %13d %7s %7s   %s" % (
            r["depth"], r["tiles"], r["shared_faces"], r["surface_faces"],
            fmt(r.get("boundary_ratio", ""), 3), fmt(r.get("boundary_dim_est", ""), 3), r["surface_is_24_4^d"]))
    L.append("spectre: depth tiles shared_edges boundary_edges  ratio  dim_est")
    for r in b["spectre"]:
        L.append("   %2d %7d %12d %13d %7s %7s" % (
            r["depth"], r["tiles"], r["shared_edges"], r["boundary_edges"],
            fmt(r.get("boundary_ratio", ""), 3), fmt(r.get("boundary_dim_est", ""), 3)))
    L.append("(dim_est = log(boundary ratio)/log(linear scale); 2 for the chair surface, 1 for a smooth curve)")
    rec = b.get("spectre_recurrence")
    if rec:
        L.append("spectre boundary recurrence (fitted, B(0)=14): B(n+2) = %g*B(n+1) + %g*B(n) + %g" % (
            rec["a"], rec["b"], rec["c"]))
        L.append("   verified on %d further levels (max residual %s); growth root %.6f vs 2+sqrt5 = %.6f;"
                 " dimension ~ %.5f" % (rec["extra_checks"], rec["max_residual"], rec["growth_root"],
                                        rec["golden_cubed"], rec["dim_estimate"]))
    L.append("chair44 boundary: exactly 24*4^d unit faces (a scaled chair), dimension exactly 2 = 3-1")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--spectre-depth", type=int, default=5)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    h = hierarchy_table()
    c = contacts_table(quick=a.quick)
    b = boundary_table(chair_max=3 if a.quick else 4, spectre_max=4 if a.quick else a.spectre_depth)
    print(report(h, c, b))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"hierarchy": h, "contacts": c, "boundary": b}, fh, indent=1, default=str)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
