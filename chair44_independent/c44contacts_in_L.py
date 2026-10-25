"""Contacts (relative poses of face-adjacent chairs) visible in legal blocks vs the 30 hierarchy contacts."""
import sys, pickle
from c44sub import *
from c44facet import occurring_contacts, touching, rel
d = pickle.load(open(sys.argv[1], 'rb')); L = d['L']; atlas = set(d['atlas'])
P0, _ = occurring_contacts()
seen = set()
for b in L:
    poses = {label_pose(l, e) for e, l in zip(E8, b)}
    for P in poses:
        for Q in poses:
            if P != Q and touching(P, Q): seen.add(rel(P, Q))
print('contacts visible in legal blocks:', len(seen), ' == hierarchy contacts P0 (30):', seen == P0, ' subset of atlas:', seen <= atlas)
print('atlas contacts that occur in NO tiling by Q (given C1):', len(atlas - seen))
