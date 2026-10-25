"""Lemma A as a published certificate, reduced by the rotation group O (24 proper rotations).

    python3 c44lemma_cert.py make   SOLID.json c1.pkl lemmaA_orbits.json   # write the certificate
    python3 c44lemma_cert.py verify SOLID.json lemmaA_orbits.json          # check it from scratch

The certificate lists one representative (the least block in its O-orbit) for every O-orbit of admissible 2x2x2
vertex blocks, with the orbit size. `verify` recomputes everything it relies on from the pyramid geometry:
  1. the 44-contact atlas, the legal blocks L (1410) and the admissible blocks Adm;
  2. Adm is exactly the union of the O-orbits of the listed representatives, and the orbit sizes are right;
  3. sigma commutes with O on the 168 labels, and L is O-invariant, so Win(sigma(R.b)) = R.Win(sigma(b)) and the
     lemma for a representative gives it for its whole orbit;
  4. the lemma on the representatives, split by window type: the 8 windows inside one parent cell (sigma of one
     label, legal by definition), the 12 face windows (two parent cells), the 6 edge windows (four parent cells)
     and the central window (eight parent cells).
"""
import json
import sys
import time

from c44adm import build_compat, admissible_blocks
from c44core import Geometry, compute_atlas
from c44sub import E8, SIG, legal_closure, sigma_cfg, block_cfg, add, ROT24
from c44sym import act_block
from c44sub import label_pose, compose
from c44core import children


def orbit(b):
    return {act_block(R, b) for R in ROT24}


def window_types(b):
    """(face, edge, centre) windows of sigma(b); offsets in {0,1,2}^3 by number of odd coordinates."""
    s = sigma_cfg(block_cfg(b))
    out = {0: [], 1: [], 2: [], 3: []}
    for o in ((x, y, z) for x in range(3) for y in range(3) for z in range(3)):
        w = tuple(s[add(o, e)] for e in E8)
        out[sum(c % 2 for c in o)].append(w)
    return out


def sigma_equivariant():
    """The children of a rotated chair are the rotated children (rotation about the origin, which sigma fixes)."""
    for R in ROT24:
        for j in range(168):
            A, t = label_pose(j)
            if sorted(children(*compose((R, (0, 0, 0)), (A, t)))) != sorted(compose((R, (0, 0, 0)), k)
                                                                         for k in children(A, t)):
                return False
    return True


def make(solid, pkl, out):
    import pickle
    d = pickle.load(open(pkl, 'rb'))
    adm = set(d['adm'])
    reps, seen = [], set()
    for b in sorted(adm):
        if b in seen:
            continue
        orb = orbit(b)
        assert orb <= adm, 'Adm is not O-invariant'
        seen |= orb
        reps.append([list(min(orb)), len(orb)])
    json.dump({'blocks': len(adm), 'orbits': len(reps), 'reps': sorted(reps)}, open(out, 'w'))
    print('admissible blocks', len(adm), 'orbits', len(reps))


def verify(solid, cert):
    t0 = time.time()
    C = json.load(open(cert))
    g = Geometry(solid)
    _, _, atlas = compute_atlas(g)
    assert len(set(atlas)) == 44, len(set(atlas))
    L, sizes = legal_closure()
    assert len(L) == 1410, len(L)
    adm = set(admissible_blocks(build_compat(set(atlas))))
    print('atlas 44, legal', len(L), 'admissible', len(adm), '(%.0f s)' % (time.time() - t0))
    assert sigma_equivariant(), 'sigma does not commute with O'
    assert all(act_block(R, b) in L for R in ROT24 for b in L), 'L is not O-invariant'
    print('sigma commutes with O and L is O-invariant')
    union, total = set(), 0
    for rep, size in C['reps']:
        orb = orbit(tuple(rep))
        assert tuple(rep) == min(orb) and len(orb) == size
        union |= orb
        total += size
    assert union == adm and total == len(adm) == C['blocks'], 'certificate does not cover Adm exactly'
    print('Adm = union of the', len(C['reps']), 'certified O-orbits')
    fails = {0: 0, 1: 0, 2: 0, 3: 0}
    for rep, _ in C['reps']:
        for k, ws in window_types(tuple(rep)).items():
            fails[k] += sum(w not in L for w in ws)
    print('illegal windows on the representatives by type (inside, face, edge, centre):',
          [fails[k] for k in range(4)])
    ok = not any(fails.values())
    print('STATUS:', 'PASS' if ok else 'FAIL', '(%.0f s)' % (time.time() - t0))
    return ok


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        make(*sys.argv[2:5])
    else:
        sys.exit(0 if verify(*sys.argv[2:4]) else 1)
