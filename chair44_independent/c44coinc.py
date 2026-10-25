"""Primitivity exponent and modular coincidence depth of the 168-label substitution (independent)."""
from c44sub import *
import itertools
# primitivity: smallest N with every label occurring in sigma^N(every label)
cfgs = {l: {(0, 0, 0): l} for l in range(168)}
for N in range(1, 6):
    cfgs = {l: sigma_cfg(c) for l, c in cfgs.items()}
    prim = all(len(set(c.values())) == 168 for c in cfgs.values())
    size = 2 ** N
    coinc = [a for a in itertools.product(range(size), repeat=3) if len({cfgs[l][a] for l in range(168)}) == 1]
    print(f'depth {N}: primitive(all labels occur in every sigma^N(j)) = {prim}; coincidence addresses = {len(coinc)}',
          (coinc[:3], [LABELS[cfgs[0][a]] for a in coinc[:3]]) if coinc else '')
    if prim and coinc: break
