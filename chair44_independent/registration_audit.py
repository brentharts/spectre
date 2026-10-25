"""Independent check of the finite inputs of Tsiokos's registration theorem (arXiv:2609.19214, Thm 5.6).

Usage:  python3 registration_audit.py r44_solid.json [candidate_certificate.json | r44_atlas.json]
                                       [companion_collision_certificate.json]      (about 3 min)

Standard-library Python 3, exact rational arithmetic. Only `vertices` and `triangles` of the solid file are
read as geometry; the feature list is recovered from the mesh. Written from the paper's statements, without
the authors' verifiers.

Part A, the mesh (what Lemmas 4.3, 4.5, 4.6 and 5.1 assume about the solid Q):
  A1  closed oriented 2-manifold, connected, Euler characteristic 2, volume 7;
  A2  the surface is the chair's boundary with 192 square pyramids: every non-flat vertex is the apex of
      exactly four triangles over an axis-parallel square of half-width 1/100 inside one of the 24 panels,
      at height a/10000 along the outward normal, 1 <= |a| <= 12, centred at panel centre +-(1/8,1/4) or
      +-(1/4,1/8); every other triangle lies in a panel, outward, outside the open pyramid bases, and on
      each panel these triangles have disjoint interiors and total area 1 - 8/2500 (so they tile it);
  A3  every edge is flat, orthogonal (pi/2 or 3pi/2) or a pyramid edge; pyramid edges: 1536, 32 of each
      sign for each of the 24 deviations alpha_j (base), beta_j (ridge); the 24 deviations are distinct
      and < pi/4 (compared via exact cos^2);
  A4  the arithmetic of Lemma 4.5 (8L^2 < 1 for L = 12/100) and of Lemma 5.1 / the clamping step.
Part B, the census (Prop. 5.2, Lemma 5.3):
  B1  F0 = poses (G, c_u - G c_v), G among all 48 signed permutations, a_u = -a_v, G n_v = -n_u: 6862;
      also all same-polarity matches (a_u = a_v, G n_v = n_u) collide with the root;
  B2  1545 of F0 overlap the root's chair, 5317 remain (F1), 83 of them integral;
  B3  for each V in F1, search a witness feature u of the root or of V with clauses (i)-(iii) of Prop. 5.2.
      This search rejects 5285 and leaves 32 survivors, all integral and proper: the 30 contacts that occur
      in supertiles and 2 more. (The paper's certificate rejects 5273 and keeps all 44; the 12 extra
      rejections are atlas contacts that never occur. Either way Lemma 5.3 follows.)
  B4  the 44 poses whose pyramids fit face to face on every shared panel (all 48 frames) contain the
      survivors; they equal the authors' 44 contacts (candidate_certificate.json or r44_atlas.json).
  B5  given companion_collision_certificate.json: each of its 5273 witnesses satisfies (i)-(iii) against
      the Partners lists recomputed here, and what it leaves is exactly those 44.
Ends with STATUS: PASS or FAIL.
"""
import itertools, json, sys, time
from fractions import Fraction as Fr

T0 = time.time()
FAIL = []

def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)

def sub(x, y): return tuple(a - b for a, b in zip(x, y))
def add(x, y): return tuple(a + b for a, b in zip(x, y))
def dot(x, y): return sum(a * b for a, b in zip(x, y))
def cross(x, y): return (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
def det3(a, b, c): return dot(a, cross(b, c))

CELLS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if (a, b, c) != (1, 1, 1)]
CELLSET = set(CELLS)
ETA, HU = Fr(1, 100), Fr(1, 10000)

# the 24 exposed unit squares (panels) of the chair: (axis, plane value, outward sign, min corner in-plane)
PANELS = []
for c in CELLS:
    for ax in range(3):
        for d in (-1, 1):
            nb = list(c); nb[ax] += d
            if tuple(nb) not in CELLSET:
                lo = list(c)
                if d == 1: lo[ax] += 1
                PANELS.append((ax, lo[ax], d, tuple(lo)))
assert len(PANELS) == 24

def in_panel_axes(ax): return [i for i in range(3) if i != ax]

def panel_of_point(p, closed=True):
    """panels whose closed (or open) square contains p"""
    out = []
    for k, (ax, val, d, lo) in enumerate(PANELS):
        if p[ax] != val: continue
        a, b = in_panel_axes(ax)
        if closed:
            if lo[a] <= p[a] <= lo[a] + 1 and lo[b] <= p[b] <= lo[b] + 1: out.append(k)
        else:
            if lo[a] < p[a] < lo[a] + 1 and lo[b] < p[b] < lo[b] + 1: out.append(k)
    return out

# ---------------------------------------------------------------- 2D exact helpers
def proj(poly, axis):
    vals = [dot(p, axis) for p in poly]
    return min(vals), max(vals)

def interiors_disjoint(P, Q):
    """convex polygons in 2D (lists of points): interiors disjoint iff some edge normal separates (SAT)"""
    for poly in (P, Q):
        n = len(poly)
        for i in range(n):
            e = sub(poly[(i + 1) % n], poly[i])
            axis = (-e[1], e[0])
            a0, a1 = proj(P, axis); b0, b1 = proj(Q, axis)
            if a1 <= b0 or b1 <= a0: return True
    return False

def area2(t):
    return abs((t[1][0]-t[0][0])*(t[2][1]-t[0][1]) - (t[2][0]-t[0][0])*(t[1][1]-t[0][1])) / 2

# ================================================================ Part A
print('Part A: the mesh of Q')
d = json.load(open(sys.argv[1]))
V = [tuple(Fr(x) for x in v) for v in d['vertices']]
TRI = [tuple(t) for t in d['triangles']]
check(sorted(map(tuple, d['base_unit_cubes'])) == sorted(CELLS), 'carrier cells are the chair [0,2]^3 \\ (1,2]^3')

# A1
dir_edges = {}
for ti, (a, b, c) in enumerate(TRI):
    for e in ((a, b), (b, c), (c, a)):
        dir_edges.setdefault(e, []).append(ti)
und = {}
for (a, b), ts in dir_edges.items():
    und.setdefault(frozenset((a, b)), []).append(((a, b), ts))
manifold = all(len(ts) == 1 for ts in dir_edges.values()) and all(
    len(v) == 2 and v[0][0] == v[1][0][::-1] for v in und.values())
check(manifold, f'closed oriented 2-manifold: every edge in exactly two triangles with opposite orientations ({len(und)} edges)')
adj = {}
for e in und:
    a, b = tuple(e); adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
seen, stack = {0}, [0]
while stack:
    x = stack.pop()
    for y in adj[x]:
        if y not in seen: seen.add(y); stack.append(y)
check(len(seen) == len(V), 'connected')
check(len(V) - len(und) + len(TRI) == 2, f'Euler characteristic V-E+F = {len(V)}-{len(und)}+{len(TRI)} = 2')
vol = sum(det3(V[a], V[b], V[c]) for a, b, c in TRI) / 6
check(vol == 7, f'volume = {vol} (outward orientation, = 7)')

# A2
on_panel = [bool(panel_of_point(p)) for p in V]
apexes = [i for i, f in enumerate(on_panel) if not f]
tri_of_vertex = {}
for ti, t in enumerate(TRI):
    for x in t: tri_of_vertex.setdefault(x, []).append(ti)
FEAT = []  # (panel index, base centre, outward normal vector, coefficient a, base corners, apex)
tent_tris = set()
ok_tents = True
for x in apexes:
    ts = tri_of_vertex[x]
    base = sorted({y for ti in ts for y in TRI[ti] if y != x})
    if len(ts) != 4 or len(base) != 4 or not all(on_panel[y] for y in base):
        ok_tents = False; continue
    pts = [V[y] for y in base]
    pans = set.intersection(*[set(panel_of_point(p)) for p in pts])
    if len(pans) != 1: ok_tents = False; continue
    k = pans.pop(); ax, val, dsg, lo = PANELS[k]
    a_, b_ = in_panel_axes(ax)
    cen = tuple(sum(p[i] for p in pts) / 4 for i in range(3))
    sq = {(p[a_] - cen[a_], p[b_] - cen[b_]) for p in pts}
    if sq != {(s * ETA, r * ETA) for s in (1, -1) for r in (1, -1)}: ok_tents = False; continue
    ap = V[x]
    if any(ap[i] != cen[i] for i in range(3) if i != ax): ok_tents = False; continue
    h = (ap[ax] - val) * dsg
    a = h / HU
    if a.denominator != 1 or not 1 <= abs(a) <= 12: ok_tents = False; continue
    off = (cen[a_] - lo[a_] - Fr(1, 2), cen[b_] - lo[b_] - Fr(1, 2))
    if off not in {(s * Fr(1, 8), r * Fr(1, 4)) for s in (1, -1) for r in (1, -1)} | \
                  {(s * Fr(1, 4), r * Fr(1, 8)) for s in (1, -1) for r in (1, -1)}:
        ok_tents = False; continue
    # each of the four triangles is (base edge, apex), oriented outward (normal has positive outward part)
    for ti in ts:
        t = [V[y] for y in TRI[ti]]
        nrm = cross(sub(t[1], t[0]), sub(t[2], t[0]))
        if nrm[ax] * dsg <= 0: ok_tents = False
        tent_tris.add(ti)
    nvec = tuple(dsg if i == ax else 0 for i in range(3))
    FEAT.append((k, cen, nvec, int(a), tuple(pts), ap))
check(ok_tents and len(FEAT) == 192 and len(tent_tris) == 768,
      f'{len(FEAT)} non-flat vertices, each the apex of a square pyramid of half-width 1/100, height a/10000, '
      f'1<=|a|<=12, at an allowed position ({len(tent_tris)} pyramid triangles)')
per_panel = {}
for f in FEAT: per_panel.setdefault(f[0], []).append(f)
check(sorted(len(v) for v in per_panel.values()) == [8] * 24, 'eight pyramids on each of the 24 panels, all positions distinct')
check(all(len({f[1] for f in v}) == 8 for v in per_panel.values()), 'positions on each panel distinct')
# flat triangles
flat_ok = True
flat_by_panel = {}
for ti, t in enumerate(TRI):
    if ti in tent_tris: continue
    pts = [V[y] for y in t]
    pans = set.intersection(*[set(panel_of_point(p)) for p in pts])
    if len(pans) != 1: flat_ok = False; continue
    k = pans.pop(); ax, val, dsg, lo = PANELS[k]
    nrm = cross(sub(pts[1], pts[0]), sub(pts[2], pts[0]))
    if not (nrm[ax] * dsg > 0 and all(nrm[i] == 0 for i in range(3) if i != ax)): flat_ok = False
    a_, b_ = in_panel_axes(ax)
    flat_by_panel.setdefault(k, []).append([(p[a_], p[b_]) for p in pts])
check(flat_ok, f'every other triangle ({len(TRI) - len(tent_tris)}) lies in one closed panel, with outward normal')
tile_ok = True
for k, tris in flat_by_panel.items():
    ax, val, dsg, lo = PANELS[k]; a_, b_ = in_panel_axes(ax)
    bases = [[(c[a_] + s * ETA, c[b_] + r * ETA) for s, r in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
             for (_, c, _, _, _, _) in per_panel[k]]
    if sum(area2(t) for t in tris) != 1 - 8 * (2 * ETA) ** 2: tile_ok = False
    for t in tris:
        if not all(interiors_disjoint(t, b) for b in bases): tile_ok = False
    bb = [(min(p[0] for p in t), max(p[0] for p in t), min(p[1] for p in t), max(p[1] for p in t)) for t in tris]
    for i in range(len(tris)):
        for j in range(i + 1, len(tris)):
            if bb[i][1] <= bb[j][0] or bb[j][1] <= bb[i][0] or bb[i][3] <= bb[j][2] or bb[j][3] <= bb[i][2]:
                continue
            if not interiors_disjoint(tris[i], tris[j]): tile_ok = False
check(tile_ok and len(flat_by_panel) == 24,
      'on each panel the flat triangles have disjoint interiors, avoid the open pyramid bases, total area 1-8/2500')
prof = d.get('patches')
if prof:
    mine = sorted((tuple(f[1]), f[3]) for f in FEAT)
    theirs = []
    for p in prof:
        base = [tuple(Fr(v) for v in q) for q in p['base']]
        theirs.append((tuple(sum(q[i] for q in base) / 4 for i in range(3)), p['coefficient']))
    check(mine == sorted(theirs), "pyramids recovered from the mesh = the file's 'patches' list (centres, coefficients)")

# A3 dihedral audit
t2 = {j: Fr(j, 100) ** 2 for j in range(1, 13)}
cls = {}
other_ok = True
for e, ((a, b), ts1), (_, ts2) in ((e, v[0], v[1]) for e, v in und.items()):
    T1, T2 = TRI[ts1[0]], TRI[ts2[0]]
    p = [V[y] for y in T1]; q = [V[y] for y in T2]
    n1 = cross(sub(p[1], p[0]), sub(p[2], p[0])); n2 = cross(sub(q[1], q[0]), sub(q[2], q[0]))
    w = [V[y] for y in T2 if y not in e][0]
    convex = dot(n1, sub(w, V[a])) < 0
    if cross(n1, n2) == (0, 0, 0) and dot(n1, n2) > 0:
        continue  # flat
    c2 = Fr(dot(n1, n2) ** 2, dot(n1, n1) * dot(n2, n2))
    if dot(n1, n2) == 0:
        continue  # pi/2 or 3pi/2
    if dot(n1, n2) < 0:
        other_ok = False; continue
    hit = [('base', j) for j in t2 if c2 == 1 / (1 + t2[j])] + [('ridge', j) for j in t2 if c2 == 1 / (1 + t2[j]) ** 2]
    if len(hit) != 1:
        other_ok = False; continue
    key = hit[0] + ('convex' if convex else 'reflex',)
    cls[key] = cls.get(key, 0) + 1
check(other_ok, 'every edge is flat, orthogonal, or a pyramid edge with one of the 24 deviations')
check(sum(cls.values()) == 1536 and len(cls) == 48 and set(cls.values()) == {32},
      f'pyramid edges: {sum(cls.values())}, 48 classes (type, j, sign), 32 each')
devs = [1 / (1 + t2[j]) for j in t2] + [1 / (1 + t2[j]) ** 2 for j in t2]
check(len(set(devs)) == 24 and min(devs) > Fr(1, 2), 'the 24 deviations are distinct and < pi/4 (cos^2 distinct, > 1/2)')
check(len([i for i in range(1, 13) for j in range(1, 13) if 10000 * i * i == 20000 * j * j + j ** 4]) == 0,
      'no solution of 10000 i^2 = 20000 j^2 + j^4 with 1 <= i,j <= 12 (alpha_i != beta_j)')
# A4
L = Fr(12, 100)
check(8 * L * L < 1 and 63 * L * L < 1, 'Lemma 4.5: 8L^2 < 1 and 63L^2 < 1 for L = 3/25 (solid angle > 4pi/3, > 7pi/4)')
check(12 * HU < ETA, 'Lemma 5.1: pyramid heights <= 12/10000 < rho = 1/100, so the rho-eroded cells lie in int Q')
check(3 * Fr(2, 100) ** 2 < Fr(1, 100) and Fr(2, 100) > 12 * HU,
      'clamping step: moving <= 1/50 per coordinate is < 1/10 in norm, and the 1/50-eroded cells lie in int Q')

# ================================================================ Part B
print('Part B: the census of feature companions')
SP = [(p, s) for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)]
def mat(p, s): return tuple(tuple(s[i] if j == p[i] else 0 for j in range(3)) for i in range(3))
G48 = [mat(p, s) for p, s in SP]
def app(A, x): return tuple(sum(A[i][k] * x[k] for k in range(3)) for i in range(3))
def mul(A, B): return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))
def tr(A): return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))
def detm(A): return det3(*A)
def inv(P): A, t = P; return (tr(A), tuple(-x for x in app(tr(A), t)))
def comp(P, Q): (A, s), (B, t) = P, Q; return (mul(A, B), add(app(A, t), s))

def cubes(P):
    """min corners (x8, integers) of the 7 image cells"""
    A, t = P
    lo = tuple(sum(min(0, A[i][k]) for k in range(3)) for i in range(3))
    return [tuple(int(8 * x) for x in add(add(app(A, c), t), lo)) for c in CELLS]

def overlap(P, Q):
    cp, cq = cubes(P), cubes(Q)
    return any(all(abs(x - y) < 8 for x, y in zip(a, b)) for a in cp for b in cq)

ROOT = (mat((0, 1, 2), (1, 1, 1)), (0, 0, 0))
F0, same = {}, set()
for ui, u in enumerate(FEAT):
    for v in FEAT:
        for A in G48:
            Gn = app(A, v[2])
            t = sub(u[1], app(A, v[1]))
            if v[3] == -u[3] and Gn == tuple(-x for x in u[2]):
                F0.setdefault((A, t), set()).add(ui)
            if v[3] == u[3] and Gn == u[2]:
                same.add((A, t))
check(all(all((8 * x).denominator == 1 for x in t) for _, t in F0), 'translations of F0 lie in (1/8)Z^3')
check(len(F0) == 6862, f'|F0| = {len(F0)} (paper: 6862)')
check(all(overlap(ROOT, P) for P in same), f'all {len(same)} same-polarity full matches overlap the root (excluded either way)')
F1 = {P: us for P, us in F0.items() if not overlap(ROOT, P)}
nint = sum(1 for _, t in F1 if all(x.denominator == 1 for x in t))
check(len(F0) - len(F1) == 1545 and len(F1) == 5317 and nint == 83,
      f'{len(F0) - len(F1)} collide with the root, {len(F1)} in F1, {nint} integral (paper: 1545, 5317, 83)')
check(all(inv(P) in F1 for P in F1), 'F1 is closed under inversion')
partners = {ui: [P for P, us in F1.items() if ui in us] for ui in range(len(FEAT))}
check(all(partners[ui] for ui in partners), 'Partners(u) is nonempty for every feature u')

def witness(other, ui):
    ps = partners[ui]
    return other not in ps and ps and all(overlap(W, other) for W in ps)

rejected, survivors, tests = [], [], 0
for P in sorted(F1):
    w = next((('root', ui) for ui in range(len(FEAT)) if witness(P, ui)), None)
    if w is None:
        Pi = inv(P)
        w = next((('mate', ui) for ui in range(len(FEAT)) if witness(Pi, ui)), None)
    if w is None:
        survivors.append(P)
    else:
        rejected.append(P); tests += len(partners[w[1]])
check(len(rejected) == 5317 - 32 and len(survivors) == 32,
      f'{len(rejected)} rejected by a witness of the same kind (root or mate feature), {len(survivors)} survive '
      f'({tests} option collisions for the first witness found)')
check(all(detm(A) == 1 and all(x.denominator == 1 for x in t) for A, t in survivors), 'survivors are integral and proper')

# B4: atlas by face-to-face fit (different criterion)
def apex_world(P):
    A, t = P
    out = {}
    for (k, c, n, a, _, ap) in FEAT:
        ax, val, dsg, lo = PANELS[k]
        wc = add(app(A, c), t)
        wax = [i for i in range(3) if A[i][ax] != 0][0]
        key = (wax, tuple(wc[i] // 1 if i != wax else wc[i] for i in range(3)))
        out.setdefault(key, set()).add(add(app(A, ap), t))
    return out
rootap = apex_world(ROOT)
cand = set()
for A in G48:
    for c in CELLS:
        for dd in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            s = add(c, dd)
            if s in CELLSET: continue
            for ic in CELLS:
                P = (A, (0, 0, 0))
                img = cubes(P)[CELLS.index(ic)]
                t = tuple(Fr(x) for x in sub(s, tuple(y // 8 for y in img)))
                if not overlap(ROOT, (A, t)): cand.add((A, t))
fit = []
for P in cand:
    ap = apex_world(P)
    sh = set(ap) & set(rootap)
    if sh and all(ap[f] == rootap[f] for f in sh): fit.append(P)
check(len(fit) == 44 and set(survivors) <= set(fit),
      f'the {len(fit)} poses whose pyramids fit on every shared panel (of {len(cand)} face-neighbour poses, '
      f'all 48 frames) contain all {len(survivors)} survivors')

# the contacts that actually occur in supertiles: closure of the children's contacts under sigma
CH = [(mat(tuple(p), tuple(s_)), tuple(Fr(x) for x in u)) for (p, s_), u in d['substitution_children']]
def kids(P): A, t = P; return [(mul(A, H), add(tuple(2 * x for x in t), app(A, u))) for H, u in CH]
def touching(P, Q):
    return any(sum(abs(x - y) for x, y in zip(a, b)) == 8 and all(abs(x - y) <= 8 for x, y in zip(a, b))
               for a in cubes(P) for b in cubes(Q))
def contacts_among(ps):
    return {comp(inv(P), Q) for P in ps for Q in ps if P != Q and touching(P, Q)}
occ = contacts_among(kids(ROOT))
while True:
    new = set().union(*[contacts_among(kids(ROOT) + kids(R)) for R in occ]) - occ
    if not new: break
    occ |= new
check(len(occ) == 30 and occ <= set(survivors) and occ <= set(fit),
      f'the {len(occ)} contacts occurring in supertiles all survive; the other survivors: '
      f'{len(set(survivors) - occ)}; atlas contacts rejected here: {len(set(fit) - set(survivors))}, none occurring')

for path in sys.argv[2:]:
    ref = json.load(open(path))
    if 'legal_contacts' in ref:
        theirs = sorted((mat(tuple(p), tuple(s_)), tuple(Fr(x) for x in t)) for (p, s_), t in ref['legal_contacts'])
        check(theirs == sorted(fit), f"the authors' {len(theirs)} contacts in {path} = the fitting poses above")
    if 'witnesses' in ref:
        role = {}
        for p in d['patches']:
            base = [tuple(Fr(v) for v in q) for q in p['base']]
            c = tuple(sum(q[i] for q in base) / 4 for i in range(3))
            role[p['role']] = next(ui for ui, f in enumerate(FEAT) if f[1] == c)
        okw, theirrej, ntest = True, set(), 0
        for w in ref['witnesses']:
            (p, s_), t8 = w['rejected_pose']
            P = (mat(tuple(p), tuple(s_)), tuple(Fr(x, 8) for x in t8))
            ui = role[w['native_role']]
            other = P if w['marker_owner'] == 0 else inv(P)
            if not (P in F1 and witness(other, ui) and len(partners[ui]) == w['possible_partners']):
                okw = False
            theirrej.add(P); ntest += w['possible_partners']
        check(okw and len(theirrej) == len(ref['witnesses']) == 5273 and ntest == 299975,
              f"each of the authors' {len(theirrej)} witnesses in {path} satisfies (i)-(iii) against Partners "
              f"recomputed here ({ntest} option collisions)")
        check(sorted(set(F1) - theirrej) == sorted(fit), "the authors' certificate leaves exactly the 44 fitting poses")

print(f'{time.time() - T0:.0f} s')
print('STATUS:', 'FAIL' if FAIL else 'PASS')
sys.exit(1 if FAIL else 0)
