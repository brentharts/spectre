"""Explicit tilings from mixed closed walks in G_K; measure the symmetry group of a large central region.
python3 c44patch.py c1.pkl"""
import sys, pickle
from c44sym import *

def closed_mixed_walk(nodes, W, comp):
    cs = set(comp)
    # find a closed walk using >=2 labels: BFS inside SCC
    from collections import deque
    edges = [(b, j, W[b][j]) for b in comp for j in (-1, 0, 1) if W[b][j] in cs]
    labs = {j for _, j, _ in edges}
    j1, j2 = sorted(labs)[:2]
    e1 = next(e for e in edges if e[1] == j1); e2 = next(e for e in edges if e[1] == j2)
    def path(a, b):
        prev = {a: None}; q = deque([a])
        while q:
            x = q.popleft()
            if x == b: break
            for j in (-1, 0, 1):
                y = W[x][j]
                if y in cs and y not in prev: prev[y] = (x, j); q.append(y)
        out = []; x = b
        while prev[x] is not None:
            x0, j = prev[x]; out.append((x0, j)); x = x0
        return out[::-1]
    walk = [(e1[0], e1[1])] + path(e1[2], e2[0]) + [(e2[0], e2[1])] + path(e2[2], e1[0])
    return walk  # list of (block, label): block --label--> next block

def build(top_block, labels_down, e):
    """top block at level N centred at vertex 0; labels_down[n] = j_n for n=N-1..0 (w_n = W_{j_n}(w_{n+1})).
    Returns level-0 labelled cfg and v_0."""
    N = len(labels_down)
    cfg = block_cfg(top_block, (-1, -1, -1))
    v = (0, 0, 0)
    for n in range(N - 1, -1, -1):
        cfg = sigma_cfg(cfg)
    vN = (0, 0, 0)
    # v_n = v_{n+1} + j_n 2^n e, in level-0 units
    v = vN
    for n in range(N - 1, -1, -1):
        v = add(v, scal(labels_down[N - 1 - n] * 2 ** n, e))
    return cfg, v

def chairs(cfg):
    """set of chair poses whose 7 cells are all present and consistently labelled"""
    out = set()
    for z, l in cfg.items():
        r, k = LABELS[l]
        A = ROT24[r]
        P = (A, add(anchor(A, k), z))
        if all(cfg.get(c) == m for c, m in pose_labels(*P).items()):
            out.add(P)
    return out

def patch_symmetries(cfg, v0, radius):
    C = chairs(cfg)
    def near(P, r):
        return all(max(abs(x - y - 0.5) for x, y in zip(c, v0)) <= r for c in pose_cells(*P))
    inner = [P for P in C if near(P, radius)]
    C0 = min(inner, key=lambda P: max(abs(x - y) for c in pose_cells(*P) for x, y in zip(c, v0)))
    cand = [P for P in C if near(P, 3)]
    found = []
    for P in cand:
        g = compose(P, inverse(C0))
        if all(compose(g, X) in C for X in inner):
            found.append(g)
    return found, len(inner), len(C)

if __name__ == '__main__':
    d = pickle.load(open(sys.argv[1], 'rb'))
    L = d['L']
    for name, e, Kname in [('C4', (0, 0, 1), 'C4'), ('C2diag', (1, 1, 0), 'C2d')]:
        Oe = set(about_axis(e))
        Kset = Oe
        stabs = {b: stab(b) for b in L}
        nodes = [b for b in L if (stabs[b] & Oe) == Kset]
        ns = set(nodes)
        W = {b: {j: window(b, scal(j, e)) for j in (-1, 0, 1)} for b in nodes}
        comp = max(tarjan(nodes, lambda b: [W[b][j] for j in (-1, 0, 1) if W[b][j] in ns]), key=len)
        walk = closed_mixed_walk(nodes, W, comp)
        p = len(walk)
        labs = [j for _, j in walk]
        print(name, 'closed walk length', p, 'labels', labs)
        # top block: walk[0][0] at level N; going down: w_{N-1} = W_{labs[0]}(w_N), ...
        N = 5
        seq = [labs[i % p] for i in range(N)]
        top = walk[0][0]
        x = top
        for i in range(N):
            assert W[x][seq[i]] == walk[(i + 1) % p][0]; x = W[x][seq[i]]
        cfg, v0 = build(top, seq, e)
        found, ni, nc = patch_symmetries(cfg, v0, 12)
        lin = sorted(RIDX[g[0]] for g in found)
        print('   patch tiles', nc, 'inner', ni, '|symmetries of inner region|', len(found),
              'linear parts about axis', [app(ROT24[i], (1, 2, 3)) for i in lin])
        J = sum(j * 2 ** m for m, j in enumerate(labs))
        print('   J =', J, ' 2^p-1 =', 2 ** p - 1, ' 2-adic axis position integer?', J % (2 ** p - 1) == 0)
