#!/usr/bin/env python3
"""contact_operators.py - which faces (edges) of a tile meet which, for the Chair44 and the Spectre.

The contact matrix K counts, over a patch, how often native face (edge) a of one tile meets native face
(edge) b of a neighbour; it is symmetric.  Its support (which pairs ever meet), its bipartite structure and its
normalised spectrum are properties of the matching rules, and they are compared here:

  Spectre  14 edges, K from spectre_patch.SpectrePatch.pair_matrix
  Chair44  24 exposed unit faces of the chair, K from the shared unit faces of chair44_core.patch_cells

Also here: the linear recurrences of the supertile boundary of both (the 'boundary operators').

CLI: python3 contact_operators.py [--tests-only]
"""
import argparse
import itertools
import sys

import numpy as np

import chair44_core as C
import spectre_patch as sp


# ------------------------------------------------------------------ chair
def native_faces():
    """The 24 exposed unit faces (axis, sign, cell) of the root chair, indexed 0..23."""
    faces = C.exposed_faces(C.CHAIR_CELLS)
    return faces, {f: i for i, f in enumerate(faces)}


def _native_of(pose, ax, sd, cell, inv_cells):
    (p, s), _ = pose
    i = ax
    return (p[i], sd * s[i], inv_cells[cell])


def chair_contact_matrix(depth=3):
    """24 x 24 symmetric count matrix of native-face contacts over the depth-n patch, plus the number
    of contacting unit faces."""
    faces, fid = native_faces()
    poses = C.sigma_depth(depth)
    owner = {}
    inv = []
    for i, pose in enumerate(poses):
        # world cell -> native cell for this pose
        f, t = pose
        m = {}
        for c in C.CHAIR_CELLS:
            x = C.frame_apply(f, c)
            w = tuple(x[k] + t[k] - (1 if f[1][k] < 0 else 0) for k in range(3))
            m[w] = c
            owner[w] = i
        inv.append(m)
    K = np.zeros((24, 24), dtype=np.int64)
    n = 0
    for w, i in owner.items():
        for ax in range(3):
            nb = list(w)
            nb[ax] += 1
            nb = tuple(nb)
            j = owner.get(nb)
            if j is None or j == i:
                continue
            a = _native_of(poses[i], ax, +1, w, inv[i])
            b = _native_of(poses[j], ax, -1, nb, inv[j])
            ia = fid[(a[0], a[1], a[2])]
            ib = fid[(b[0], b[1], b[2])]
            K[ia, ib] += 1
            K[ib, ia] += 1
            n += 1
    return K, n


def spectre_contact_matrix(level=4):
    p = sp.SpectrePatch(level)
    return p.pair_matrix().astype(np.int64), len(p.edge_pairs())


# ------------------------------------------------------------------ analysis
def is_bipartite(K):
    n = len(K)
    colour = {}
    for s in range(n):
        if s in colour or not K[s].any():
            continue
        colour[s] = 0
        stack = [s]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(K[u]):
                if v not in colour:
                    colour[v] = 1 - colour[u]
                    stack.append(v)
                elif colour[v] == colour[u]:
                    return False
    return True


def components(K):
    n = len(K)
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, stack = {s}, [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(K[u]):
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def normalised_spectrum(K):
    """Eigenvalues of K / (sum of all entries), descending."""
    return np.sort(np.linalg.eigvalsh(K.astype(float) / K.sum()))[::-1]


def summary(K):
    return dict(support=int((K > 0).sum()), rank=int(np.linalg.matrix_rank(K)),
                bipartite=is_bipartite(K), components=len(components(K)),
                spectrum=normalised_spectrum(K))


# ------------------------------------------------------------------ boundary operators
def boundary_recurrences():
    """Characteristic polynomials of the supertile boundary recurrences, as coefficient lists (highest
    first): Spectre outline edge counts B(n+3) = 5B(n+2) - 3B(n+1) - B(n)  ->  (x-1)(x^2-4x-1);
    Chair44 supertile surface 24 * 4^N  ->  (x-4)."""
    import sympy as sp_
    x = sp_.symbols("x")
    return {"spectre": sp_.factor(x ** 3 - 5 * x ** 2 + 3 * x + 1), "chair": sp_.factor(x - 4)}


def spectre_outline_counts():
    return (14, 46, 182, 758, 3198, 13534, 57318)


# ------------------------------------------------------------------ tests
def test_chair_contacts_count_the_shared_unit_faces_exactly():
    K, n = chair_contact_matrix(2)
    assert n == 12 * (8 ** 2 - 4 ** 2) and K.sum() == 2 * n               # the shared-face law of chair44facts
    assert (K == K.T).all()


def test_chair_contact_counts_scale_with_the_patch():
    K2, n2 = chair_contact_matrix(2)
    K3, n3 = chair_contact_matrix(3)
    assert n3 == 12 * (8 ** 3 - 4 ** 3)
    assert abs(np.linalg.norm(normalised_spectrum(K3)) - np.linalg.norm(normalised_spectrum(K2))) < 0.05


def test_spectre_edge_meeting_matrix_is_bipartite_with_symmetric_spectrum():
    K, _ = spectre_contact_matrix(3)
    assert (K == K.T).all() and is_bipartite(K)
    ev = np.sort(np.linalg.eigvalsh(K.astype(float)))
    assert np.allclose(ev, -ev[::-1], atol=1e-6)


def test_boundary_recurrence_factorisations_and_the_spectre_counts():
    r = boundary_recurrences()
    import sympy as sp_
    x = sp_.symbols("x")
    assert sp_.expand(r["spectre"] - (x - 1) * (x ** 2 - 4 * x - 1)) == 0 and r["chair"] == x - 4
    b = spectre_outline_counts()
    assert all(b[i + 3] == 5 * b[i + 2] - 3 * b[i + 1] - b[i] for i in range(len(b) - 3))


def run_tests(verbose=True):
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            status = "PASS"
        except Exception as exc:  # noqa: BLE001
            failed += 1
            status = "FAIL: %s: %s" % (type(exc).__name__, exc)
        if verbose:
            print("  %-74s %s" % (name, status))
    if verbose:
        print("%d test(s), %d failed" % (len(tests), failed))
    return failed == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-only", action="store_true")
    a = ap.parse_args(argv)
    if a.tests_only:
        return 0 if run_tests() else 1
    for name, (K, n) in (("Spectre level 4", spectre_contact_matrix(4)), ("Chair44 depth 3", chair_contact_matrix(3))):
        s = summary(K)
        print("%s: %d contacts, support %d of %d, rank %d, bipartite %s, components %d" % (
            name, n, s["support"], K.size, s["rank"], s["bipartite"], s["components"]))
        print("  normalised spectrum", np.round(s["spectrum"], 4))
    print("boundary operators:", boundary_recurrences())
    return 0


if __name__ == "__main__":
    sys.exit(main())
