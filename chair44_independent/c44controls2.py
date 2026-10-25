"""Targeted control: take an atlas-violating (but consistent, disjoint) block whose child windows are illegal, add its
non-atlas face contacts (and inverses) to the atlas, and recount.  The lemma must then fail.  python3 c44controls2.py solid.json c1.pkl"""
import sys, random, pickle
from c44adm import *
random.seed(2)
d = pickle.load(open(sys.argv[2], 'rb'))
L, adm, atlas = d['L'], set(d['adm']), set(d['atlas'])
g = Geometry(sys.argv[1]); shell, cand, _ = compute_atlas(g)
comp_all = build_compat(set(cand))
Ls = sorted(L)
done = 0
while done < 1:
    b = list(random.choice(Ls)); i = random.randrange(8); b[i] = random.randrange(168); b = tuple(b)
    if b in adm or not all(comp_all[sub(E8[x], E8[y])][b[y]][b[x]] for x in range(8) for y in range(x)): continue
    if sigma_windows(b) <= L: continue
    poses = {label_pose(l, e) for e, l in zip(E8, b)}
    extra = {rel(P, Q) for P in poses for Q in poses if P != Q and touching(P, Q)} - atlas
    A = atlas | extra | {inverse(e) for e in extra}
    ad = admissible_blocks(build_compat(A))
    bad = sum(1 for x in ad if not sigma_windows(x) <= L)
    print(f'(c\') atlas + {len(extra)} contact(s) taken from one violating block: admissible {len(ad)}, lemma failures {bad}', flush=True)
    done += 1
