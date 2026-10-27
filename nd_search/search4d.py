"""Bounded search for Kamenetsky's Problem 9.1: a coarsening-closed homochiral frame assignment of the 4D chair.

Families: (F) the face v_3 = 0 carries the 3D solution (coordinate 3 fixed), the eight children with v_3 = 1 sampled;
(R) all fifteen sampled.  Every assignment is tested with chair_nd.closure_test (cap 400).  Prints any closure."""
import random, sys, time
import chair_nd as C
env = C.make(4)
SOL3 = {(0, 0, 0): (0, 1, 2), (0, 0, 1): (1, 0, 2), (0, 1, 0): (0, 2, 1), (0, 1, 1): (2, 0, 1),
        (1, 0, 0): (2, 1, 0), (1, 0, 1): (1, 2, 0), (1, 1, 0): (0, 1, 2)}
choices = {v: C.homochiral_choices(env, v) for v in env['OUTER']}
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 20261026)
t0 = time.time(); n = 0; stats = {}
best = None
while time.time() - t0 < budget:
    fam = 'F' if n % 2 == 0 else 'R'
    a = {}
    for v in env['OUTER']:
        if fam == "F" and v[3] == 0 and v[:3] in SOL3:
            a[v] = SOL3[v[:3]] + (3,)
        else:
            a[v] = rng.choice(choices[v])
    r = C.closure_test(env, a, cap=400)
    n += 1
    stats[(fam, r[0])] = stats.get((fam, r[0]), 0) + 1
    if r[0] == 'closed':
        print("CLOSED", fam, a, r, flush=True)
    if r[0] == 'offset' and (best is None or r[1] < best[0]):
        best = (r[1], fam, r[2])
print("tested", n, "assignments in %.0fs:" % (time.time() - t0), stats, "smallest pose set reaching an offset:", best, flush=True)
