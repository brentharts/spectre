"""Negative controls for the window lemma.  python3 c44controls.py solid.json c1.pkl
(a) random label blocks; (b) legal blocks with one cell relabelled so that the block stays consistent and disjoint
(atlas ignored) but violates the atlas: does the lemma's conclusion fail for them?; (c) atlas enlarged by one illegal
face contact (and its inverse): is the lemma then false?"""
import sys, random, pickle
from c44adm import *
random.seed(1)
d = pickle.load(open(sys.argv[2], 'rb'))
L, adm, atlas = d['L'], set(d['adm']), set(d['atlas'])
g = Geometry(sys.argv[1])
shell, cand, _ = compute_atlas(g)
rnd = [tuple(random.randrange(168) for _ in range(8)) for _ in range(20000)]
print('(a) random blocks with all 27 windows of sigma legal:', sum(sigma_windows(b) <= L for b in rnd), '/ 20000', flush=True)
comp_all = build_compat(set(cand))
def ok_noatlas(b):
    return all(comp_all[sub(E8[i], E8[j])][b[j]][b[i]] for i in range(8) for j in range(i))
pert = set()
Ls = sorted(L)
while len(pert) < 3000:
    b = list(random.choice(Ls)); i = random.randrange(8); b[i] = random.randrange(168); b = tuple(b)
    if b not in adm and ok_noatlas(b): pert.add(b)
bad = sum(1 for b in pert if not sigma_windows(b) <= L)
print('(b) consistent+disjoint but atlas-violating perturbed blocks:', len(pert), ' with an illegal child window:', bad, flush=True)
extras = random.sample(sorted(set(cand) - atlas), 8)
for e in extras:
    A = atlas | {e, inverse(e)}
    ad = admissible_blocks(build_compat(A))
    bad = sum(1 for b in ad if not sigma_windows(b) <= L)
    print('(c) atlas + 1 illegal contact: admissible', len(ad), ' blocks violating the lemma', bad, flush=True)
