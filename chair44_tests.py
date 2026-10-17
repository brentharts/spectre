#!/usr/bin/env python3
"""chair44_tests.py - checks for chair44.py, chair44_vs_spectre.py and chair44_family.py.

    python3 chair44_tests.py           # everything (~1 min)
    python3 chair44_tests.py --quick   # skips the slow exact-vs-fast labelling cross-check

Every number asserted here is either published (arXiv:2609.19214 / 2609.23783: 2388 raw
contacts, 44 legal, 30 hierarchy contacts) or an exact identity derived in this fork
(boundary laws); none is merely 'whatever the code printed'.
"""
import sys
import time
from fractions import Fraction as Fr

import numpy as np

import chair44 as C
import chair44_family as F
import chair44_vs_spectre as V
import chair44_rigidity as R
import chair44_spectre_link as L

_RAW = None


def raw():
    global _RAW
    if _RAW is None:
        _RAW = C.face_contacts()
    return _RAW


def vol(Vv, T):
    return F.mesh_volume(Vv, T)


def test_contact_enumeration():
    r = raw()
    assert len(r) == 2388
    assert sum(C.frame_det(p[0]) < 0 for p in r) == 1194            # half are mirror images
    deco = C.Decoration()
    legal = C.legal_contacts(deco, raw=r)
    assert len(legal) == 44 and set(legal) == set(C.ATLAS44)
    assert all(C.frame_det(p[0]) == 1 for p in legal)               # no mirrors: chirality is forced
    assert len(C.legal_contacts(deco, exact=False, raw=r)) == 220   # voids allowed -> more contacts


def test_substitution_and_patches():
    assert len(C.CHILDREN) == 8
    assert all(C.frame_det(f) == 1 for f, _ in C.CHILDREN)
    for d in (1, 2, 3):
        s = C.check_patch(C.sigma_depth(d))
        assert s["tiles"] == 8 ** d and s["cells"] == 7 * 8 ** d
    for d in (2, 3):
        assert len(C.observed_contacts(C.sigma_depth(d))) == 30     # contacts the hierarchy realises
        assert C.observed_contacts(C.sigma_depth(d)) <= set(C.ATLAS44)
    assert len({p[0] for p in C.sigma_depth(3)}) == 24              # all cube rotations occur


def test_chair_boundary_law():
    for d in range(0, 4):
        s = C.patch_surface_stats(C.sigma_depth(d))
        assert s["surface_faces"] == 24 * 4 ** d
        assert s["shared_faces"] == 12 * (8 ** d - 4 ** d)


def test_meshes_and_mirror_winding():
    deco = C.Decoration()
    for Vv, T in (C.base_chair_mesh(), C.decorated_mesh(deco), C.decorated_mesh(deco, 500)):
        assert abs(vol(Vv, T) - 7) < 1e-9
    Vv, T = C.base_chair_mesh()
    W, TT, _ = C.assemble(C.sigma_depth(1), Vv, T)
    assert abs(vol(W, TT) - 56) < 1e-9
    mirror = (((0, 1, 2), (-1, 1, 1)), (0, 0, 0))                    # reflection in x
    assert C.frame_det(mirror[0]) == -1
    W, TT, _ = C.assemble([mirror], Vv, T)
    assert abs(vol(W, TT) - 7) < 1e-9                                # winding was flipped correctly


def test_spectre_hierarchy_and_boundary():
    for n in (1, 2, 3, 4):
        assert V.spectre_tile_counts(n) == len(V.spectre_patch(n))
    assert [V.spectre_tile_counts(n) for n in (1, 2, 3, 4)] == [9, 71, 559, 4401]
    h = V.hierarchy_table()
    assert abs(h["spectre"]["inflation_tiles"] - (4 + 15 ** 0.5)) < 1e-9
    B = [14]
    for n in (1, 2, 3, 4):
        _, info = V.spectre_observed_contacts(n)
        assert info["tiles"] * 14 == 2 * info["shared_edges"] + info["boundary_edges"]
        B.append(info["boundary_edges"])
    assert B == [14, 46, 182, 758, 3198]
    for n in range(len(B) - 2):                                      # exact recurrence found in this fork
        assert B[n + 2] == 4 * B[n + 1] + B[n] - 16


def test_spectre_raw_contacts():
    raw_s = V.spectre_raw_contacts()
    assert len(raw_s) == 158 and sum(raw_s.values()) == 79            # 79 rotation-only + 79 mirror
    obs, _ = V.spectre_observed_contacts(3)
    assert len(obs) == 25 and obs <= {k for k, m in raw_s.items() if not m}


def test_family_machinery():
    assert F.check(verbose=False)
    ident = F.Identification()
    rng = np.random.default_rng(3)
    fld = F.normalize(ident.project([F.gaussian_phi(F.random_atoms(rng, 4)) for _ in ident.classes]))
    assert abs(np.abs(F.grid_eval(fld, 64)).max() - 1) < 1e-12        # --amp is the peak displacement
    sf = F.SiteField(ident, None, 0.1, "cosine")                      # peak sits exactly on the site centre
    role = max(range(192), key=lambda r: abs(ident.deco.profile[r]))
    pi, ctr = ident.deco.sites[role]
    u, v = (float(x) for x in ident.native_to_chart(pi, ctr))
    assert abs(abs(float(sf.eval(pi, np.array([u]), np.array([v]))[0])) - 1) < 1e-12


def test_class_labelling_model(slow=True):
    r, reqs = F.class_requirements(C.Decoration(), raw())
    assert sum(1 for ok, q in reqs if ok) == 530 and sum(1 for ok, q in reqs if ok and not q) == 44
    res = F.merge_scan(verbose=False, max_k=2)
    assert res[1][1] == 0 and res[2][:2] == (2047, 1224)
    if slow:
        deco = C.Decoration()
        for L in ([0] * 6 + [1] * 6, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]):
            fast = sum(1 for ok, q in reqs if ok and all(L[a - 1] == L[b - 1] for a, b in q))
            prof = [Fr(L[abs(int(x)) - 1] + 1) * (1 if x > 0 else -1) for x in deco.profile]
            assert fast == len(C.legal_contacts(deco, profile=prof, raw=raw()))


def test_nonlattice_rigidity():
    ident = F.Identification()
    zoo = R.standard_fields(ident)
    for name in ("R44-like pyramids (rho=0.05)", "random projected Gaussians"):
        fld = zoo[name]
        r = R.scan(fld, ident, G=32, theta_step=6.0, tilt=False)
        assert r["lattice_zero"], name                                   # lattice contacts are exact zeros
        assert all(k[0] == k[1] for k in r["lattice_zero"])              # ... only within a panel class
        ms = r["min_slide"]
        rf = R.refine(fld, ident, ms[1], (ms[2], ms[3]), G=64)
        assert rf[0] > 1e-3, (name, rf)                                  # no non-lattice exact complement
    # a deliberately wrong pose must not score zero
    reps = ident.reps
    assert R.mu_pose(zoo["R44-like pyramids (rho=0.05)"], reps, 0, 0, -1, 1, 180.0, 0.5, 0.0, G=32) > 1e-3


def test_spectre_link():
    hs = [L.boundary_histogram(k) for k in range(5)]
    for k, h in enumerate(hs):
        e, o, P = L.golden_prediction(k)
        assert int(h.sum()) == P == 2 * L.fib(3 * k + 5) + 4                 # P(k) = 2 F(3k+5) + 4
        assert int(h[0:6:2].sum()) == e and int(h[1:6:2].sum()) == o         # even / odd direction classes
        assert all(h[j] == h[j + 6] for j in range(6))                       # directions only matter mod 180
        assert abs(abs(L.z6_modes(h)[3]) - L.fib(3 * k + 2)) < 1e-9          # the Z/2 character is golden
        assert max(abs(L.z6_modes(h)[m]) for m in (1, 2, 4, 5)) < 5          # Z/3 and faithful modes stay small
    assert [int(h.sum()) for h in hs] == [14, 46, 182, 758, 3198]
    for r in L.chair_side(3, verbose=False):
        assert r["B"] == r["pred"] == r["pred_fork_formula"]
        assert len(set(r["normals"].values())) == 1 and len(r["normals"]) == 6
    u = L.unified(verbose=False)
    assert abs(u["Spectre"]["D_b"] - 1.399253) < 1e-6 and abs(u["Spectre"]["s_star"] - 0.300373) < 1e-6
    assert u["Chair44"]["D_b"] == 2.0 and abs(u["Chair44"]["s_star"] - 1 / 3) < 1e-12
    assert abs(u["Spectre"]["mu"] - 1.858559) < 1e-6
    assert L.torsion_tie(verbose=False)[1]


def main():
    quick = "--quick" in sys.argv
    tests = [test_contact_enumeration, test_substitution_and_patches, test_chair_boundary_law,
             test_meshes_and_mirror_winding, test_spectre_hierarchy_and_boundary,
             test_spectre_raw_contacts, test_family_machinery, test_nonlattice_rigidity, test_spectre_link,
             lambda: test_class_labelling_model(slow=not quick)]
    names = [t.__name__ for t in tests[:-1]] + ["test_class_labelling_model"]
    failed = 0
    for name, t in zip(names, tests):
        t0 = time.time()
        try:
            t()
            print("PASS  %-36s %.1fs" % (name, time.time() - t0))
        except Exception as e:                      # noqa: BLE001 - report and continue
            failed += 1
            print("FAIL  %-36s %s: %s" % (name, type(e).__name__, e))
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
