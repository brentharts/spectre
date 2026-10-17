#!/usr/bin/env python3
"""symchair44_tests.py - symchair44.py generates everything; these tests compare it with the
published data (chair44_data/*), which only the TESTS read, and with chair44.py.

    python3 symchair44_tests.py          # about 10 s
    python3 symchair44_tests.py --full   # + the 3-minute selection screen over all 6561 tilings
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from fractions import Fraction as Fr

import chair44 as C
import symchair44 as S

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chair44_data")
_JSON = None


def solid_json():
    global _JSON
    if _JSON is None:
        with open(os.path.join(DATA, "r44_solid.json")) as fh:
            _JSON = json.load(fh)
    return _JSON


def fr(x):
    return Fr(int(x.p), int(x.q))


def test_mesh_equals_json():
    d = solid_json()
    solid = S.Solid(S.label_table(S.CHILDREN))
    Vj = [tuple(Fr(x) for x in v) for v in d["vertices"]]
    Vm = [tuple(fr(x) for x in v) for v in solid.vertices]
    assert len(Vm) == d["vertex_count"] == 2138 and set(Vm) == set(Vj)
    assert len(solid.edges) == d["edge_count"] and len(solid.triangles) == d["triangle_count"]

    def canon(t):
        m = min(range(3), key=lambda i: t[i])
        return t[m:] + t[:m]
    assert {canon(tuple(Vj[i] for i in t)) for t in d["triangles"]} == \
           {canon(tuple(Vm[i] for i in t)) for t in solid.triangles}        # same oriented triangles
    assert Fr(d["base_halfwidth"]) == fr(S.W0) and Fr(d["height_unit"]) == fr(S.H0)


def test_panels_sites_labels_equal_data():
    d = solid_json()
    with open(os.path.join(DATA, "native_panels.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24
    for i, r in enumerate(rows):
        ax, key = S.PANELS[i]
        assert (int(r["normal_axis"]), int(r["outward_sign"])) == (ax, S.PANEL_SIGN[i])
        assert tuple(Fr(r[k]) for k in ("center_x", "center_y", "center_z")) == tuple(Fr(int(x), 2) for x in key)
        assert [int(r["a_%d" % k]) for k in range(8)] == S.label_table(S.CHILDREN)[8 * i:8 * i + 8]
    assert S.label_table(S.CHILDREN) == d["profile"]
    for p in d["patches"]:                                           # every pyramid: base corners and apex
        pi, k = p["face"], p["role"] % 8
        assert p["role"] // 8 == pi
        ax, key = S.PANELS[pi]
        ctr = [sum(Fr(b[i]) for b in p["base"]) / 4 for i in range(3)]
        assert tuple(ctr[i] for i in range(3) if i != ax) == tuple(fr(c) for c in
                                                                    (S.SITES[p["role"]][1][a] for a in S.PANEL_AXES[pi]))
        lab = S.label_table(S.CHILDREN)[p["role"]]
        assert Fr(p["apex"][ax]) == Fr(key[ax], 2) + S.PANEL_SIGN[pi] * lab * fr(S.H0)


def test_substitution_equals_atlas_and_chair44():
    with open(os.path.join(DATA, "r44_atlas.json")) as fh:
        atlas = json.load(fh)
    pub = tuple(((tuple(c[0][0]), tuple(c[0][1])), tuple(c[1])) for c in atlas["children"])
    assert S.CHILDREN == pub == C.CHILDREN
    for d in (1, 2):
        assert [(f, tuple(t)) for f, t in S.sigma_poses(S.CHILDREN, d)] == list(C.sigma_depth(d))
    for f in S.all_frames()[:48:5]:                                   # cells and faces agree with chair44
        for t in ((0, 0, 0), (1, -2, 3)):
            assert S.pose_cells((f, t)) == C.pose_cells((f, t))
            assert S.exposed_faces((f, t)) == C.exposed_faces((f, t))


def test_double_chair_tilings():
    T = S.double_chair_tilings()
    assert len(T) == 6561 == 3 ** 8
    target = S.double_chair()
    assert len(target) == 56
    for t in T[::257]:
        assert len(t) == 8 and set().union(*[S.pose_cells(c) for c in t]) == target
        assert all(S.frame_det(f) == 1 for f, _ in t)
    assert S.CHILDREN in {tuple(sorted(t, key=S.child_order)) for t in T}


def test_gluing_and_classes():
    classes, dead, neq = S.gluing(S.CHILDREN, 3)
    assert len(classes) == 12 and not dead and neq == 43008
    assert S.label_table(S.CHILDREN, 2) == S.label_table(S.CHILDREN, 3)    # depth-2 patch already sees every contact
    assert S.orientations(S.CHILDREN) == 24
    # a tiling outside the winning family leaves only 4 free classes
    other = next(t for t in S.double_chair_tilings()[:40] if S.gluing(list(t), 2)[1] == set() and len(S.gluing(list(t), 2)[0]) < 12)
    assert len(S.gluing(list(other), 2)[0]) == 4


def test_exact_invariants_symbolic_and_numeric():
    ok, rep, solid = S.verify(symbolic=True, verbose=False)
    assert ok and solid.volume() == 7                                    # for ALL w, h
    ok2, _, num = S.verify(verbose=False)
    assert ok2 and len(num.vertices) == 2138
    # a wrong label table is caught: flip one sign and the volume is no longer 7
    lab = S.label_table(S.CHILDREN)
    lab[0] = -lab[0]
    bad = S.Solid(lab)
    assert bad.volume() != 7 and bad.is_closed_manifold()


def test_latex_and_obj():
    txt = S.latex_document()
    assert r"\begin{document}" in txt and "pmatrix" in txt and r"\frac{1}{100}" in txt
    assert txt.count(r"\begin{equation*} A_") == 8
    solid = S.Solid(S.label_table(S.CHILDREN), hscale=100)
    obj = solid.obj().splitlines()
    assert sum(1 for l in obj if l.startswith("v ")) == 2138 and sum(1 for l in obj if l.startswith("f ")) == 4272
    if shutil.which("pdflatex"):
        d = tempfile.mkdtemp()
        path = S.write_latex(os.path.join(d, "r44.tex"), pdf=True)
        assert os.path.getsize(path.replace(".tex", ".pdf")) > 10000


def test_contact_atlas():
    A = S.atlas()
    d = solid_json()
    assert len(A["raw"]) == C.RAW_COUNT == 2388 and A["mirror"] == 1194
    assert set(A["legal"]) == set(C.ATLAS44) and len(A["legal"]) == 44          # same 44 as the published atlas
    realized = C.observed_contacts(C.sigma_depth(3))                           # chair44's independent computation
    assert set(A["realized"]) == set(realized) and len(realized) == d["contact_state_count"] == 30
    assert len(A["equations"]) == d["contact_equations"] == 372                 # the JSON's own equation count
    assert S.equation_rank(A["equations"]) == 192 - d["kernel_dimension"] == 180
    ok, _ = S.verify_atlas(A, verbose=False)
    assert ok
    # conventions agree with chair44; a pose composed with its inverse is the identity pose
    real = set(A["realized"])
    for p in A["realized"]:
        assert S.relative_pose(S.ROOT, p) == p == C.relative_pose(C.ROOT_POSE, p)
        q = S.pose_inverse(p)
        assert q in real and S.pose_inverse(q) == p
    # legality is label-sensitive: with every height equal to +1 there is no exact complement at all
    assert len(S.legal_contacts([1] * 192)) == 0
    from collections import Counter
    angles = Counter(str(S.rotation_data(p[0])[0]) for p in A["realized"])
    assert sum(angles.values()) == 30 and set(angles) == {"0", "pi/2", "2*pi/3", "pi"}


def test_atlas_latex():
    A = S.atlas()
    txt = S.latex_document(atlas_data=A)
    assert "The contact atlas" in txt and r"\xrightarrow" in txt and "2388" in txt and "372" in txt
    assert txt.count(r"\begin{pmatrix}") > 60                                    # 30 + 14 poses and their axes
    if shutil.which("pdflatex"):
        d = tempfile.mkdtemp()
        path = S.write_latex(os.path.join(d, "r44.tex"), pdf=True, with_atlas=True)
        assert os.path.getsize(path.replace(".tex", ".pdf")) > 20000


def test_full_screen():
    chosen, summary = S.screen_tilings(verbose=False)
    assert summary["classes_histogram"] == {4: 6555, 12: 6}
    assert summary["survivors"] == 6 and summary["after_orientation_rule"] == 3
    assert chosen == S.CHILDREN


def main():
    full = "--full" in sys.argv
    tests = [test_mesh_equals_json, test_panels_sites_labels_equal_data, test_substitution_equals_atlas_and_chair44,
             test_double_chair_tilings, test_gluing_and_classes, test_exact_invariants_symbolic_and_numeric,
             test_latex_and_obj, test_contact_atlas, test_atlas_latex] + ([test_full_screen] if full else [])
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print("PASS  %-46s %.1fs" % (t.__name__, time.time() - t0))
        except Exception as e:                      # noqa: BLE001
            failed += 1
            print("FAIL  %-46s %s: %s" % (t.__name__, type(e).__name__, e))
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
