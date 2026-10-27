"""Simulated annealing for Kamenetsky's Problem 9.1 (a coarsening-closed homochiral marking of the 4D chair).

Objective: |Sigma(P0)|, the number of supertile poses admissible under the hierarchical contacts, which in 3D is
minimised exactly by the three closed assignments (44).  Kamenetsky's coordinate descent stalls on plateaus; here a
Metropolis walk accepts sideways and some uphill moves.  Every state whose objective is <= 2|P0| is given the full
closure test.  Usage: python3 anneal4d.py SECONDS SEED CACHE_DIR
"""
import math
import random
import sys
import time
import chair_nd as C
from chair_nd_fast import Fast

budget, seed, cache = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
F = Fast(4, cache_dir=cache)
OUT = F.env['OUTER']
CHO = {v: C.homochiral_choices(F.env, v) for v in OUT}
rng = random.Random(seed)
CAP = 2500
memo = {}


def score(a):
    key = tuple(a[v] for v in OUT)
    if key not in memo:
        r = F.closure(a, cap=CAP, want_sigma0=True)
        if r[0] == 'closed':
            print('CLOSED', key, r, flush=True)
            memo[key] = (0, r)
        elif r[0] == 'cap0':
            memo[key] = (10 ** 6 + r[1], r)
        elif r[0] == 'cap':
            s0 = r[3] if len(r) > 3 else r[1]
            memo[key] = (s0, r)
        else:                      # offset: use the size reached
            memo[key] = (r[1] * 2, r)
    return memo[key][0]


t0 = time.time()
best = (10 ** 9, None)
evals = 0
while time.time() - t0 < budget:
    if len(sys.argv) > 4 and sys.argv[4] == 'canonical' and best[1] is None:
        a = {v: ((0, 1, 2, 3) if sum(v) % 2 == 0 else (1, 0, 2, 3)) for v in OUT}   # Kamenetsky's canonical family
    else:
        a = {v: rng.choice(CHO[v]) for v in OUT}
    cur = score(a)
    T = 200.0
    steps = 0
    while T > 1 and time.time() - t0 < budget:
        v = rng.choice(OUT)
        old = a[v]
        a[v] = rng.choice([p for p in CHO[v] if p != old])
        new = score(a)
        evals += 1
        if new <= cur or rng.random() < math.exp((cur - new) / T):
            cur = new
            if cur < best[0]:
                best = (cur, tuple(a[w] for w in OUT))
                print('best %d after %d evals, %.0fs' % (cur, evals, time.time() - t0), flush=True)
        else:
            a[v] = old
        T *= 0.995
        steps += 1
print('done: %d evaluations (%d distinct) in %.0fs; best objective %d at %s'
      % (evals, len(memo), time.time() - t0, best[0], best[1]), flush=True)
