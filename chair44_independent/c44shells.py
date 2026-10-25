"""Partial independent check of (H): first-shell enclosures of a root chair under the 44-atlas, and how many of the
8 candidate parents are compatible with each.  python3 c44shells.py c1.pkl"""
import sys, pickle
from c44core import *
from c44facet import touching, rel
atlas = set(pickle.load(open(sys.argv[1], 'rb'))['atlas'])
root = (I3, (0, 0, 0))
rc = set(pose_cells(*root))
shell = sorted({add(c, d) for c in CELLS for d in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]} - rc)
def compat(P, Q):
    if P == Q: return True
    if set(pose_cells(*P)) & set(pose_cells(*Q)): return False
    return (not touching(P, Q)) or rel(P, Q) in atlas
nbrs = sorted(atlas)
encl = []
def rec(chosen, covered):
    todo = [s for s in shell if s not in covered]
    if not todo: encl.append(tuple(chosen)); return
    s = todo[0]
    for P in nbrs:
        cs = set(pose_cells(*P))
        if s in cs and all(compat(P, Q) for Q in chosen):
            rec(chosen + [P], covered | cs)
rec([], set())
print('first-shell enclosures:', len(encl))
parents = [compose(root, inverse(c)) for c in CHILD_POSES]
hist = {}
for E in encl:
    patch = [root] + list(E)
    ok = 0
    for S in parents:
        kids = [compose(S, c) for c in CHILD_POSES]
        if all(k in patch or all(compat(k, X) for X in patch) for k in kids): ok += 1
    hist[ok] = hist.get(ok, 0) + 1
print('enclosures by number of compatible candidate parents:', hist)

# central completion: for each enclosure, the unique compatible parent S and its central child Z (child index 7).
stats = {'root_is_central': 0, 'no_compatible_central_shell': 0, 'all_completions_force_S': 0, 'some_completion_misses_child': 0}
for E in encl:
    patch = [root] + list(E)
    S = next(S for S in parents if all(k in patch or all(compat(k, X) for X in patch) for k in [compose(S, c) for c in CHILD_POSES]))
    kids = [compose(S, c) for c in CHILD_POSES]
    Z = kids[7]
    if Z == root:
        stats['root_is_central'] += 1; assert all(k in patch for k in kids); continue
    assert Z in patch, 'central child not already a neighbour'
    comps = []
    for F_ in encl:
        Ez = [compose(Z, X) for X in F_]
        if all(X in patch or all(compat(X, Y) for Y in patch) for X in Ez):
            comps.append(Ez)
    if not comps: stats['no_compatible_central_shell'] += 1
    elif all(all(k in patch or k in Ez for k in kids) for Ez in comps): stats['all_completions_force_S'] += 1
    else: stats['some_completion_misses_child'] += 1
print('central completion:', stats)
