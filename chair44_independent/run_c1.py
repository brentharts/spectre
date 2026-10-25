"""C1 reproduction: python3 run_c1.py /path/to/six-birds-tiles/solid/r44_solid.json"""
import sys, time, pickle
from c44adm import *
from c44facet import occurring_contacts, coarsening_closure
solid = sys.argv[1]
g = Geometry(solid)
shell, cand, legal_geo = compute_atlas(g)
occ, _ = occurring_contacts()
Pst, _, _ = coarsening_closure(occ)
assert set(legal_geo) == Pst and len(Pst) == 44
print('atlas: 2388 ->', len(Pst), '(geometric == coarsening closure)')
L, sizes = legal_closure(); print('legal closure', sizes)
t = time.time(); comp = build_compat(Pst); print('compat built', round(time.time() - t))
adm = admissible_blocks(comp); print('admissible', len(adm))
print('legal subset of admissible:', L <= set(adm))
bad = [b for b in adm if not sigma_windows(b) <= L]
print('Lemma failures', len(bad))
pickle.dump({'L': L, 'adm': adm, 'atlas': sorted(Pst)}, open(sys.argv[2] if len(sys.argv) > 2 else 'c1.pkl', 'wb'))
