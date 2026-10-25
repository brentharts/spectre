"""C2: symmetry classification computations (independent).
python3 c44sym.py c1.pkl"""
import sys, pickle, itertools
from c44sub import *

V = (1, 1, 1)  # centre vertex of a block (cells with min corners {0,1}^3)

def rot_about(R, c):
    """isometry x -> R(x-c)+c with c integer"""
    return (R, sub(c, app(R, c)))

def act_block(R, b):
    g = rot_about(R, V)
    cfg = {}
    for e, l in zip(E8, b):
        P = compose(g, label_pose(l, e))
        for cell, m in pose_labels(*P).items():
            if cell in cfg: assert cfg[cell] == m
            cfg[cell] = m
    return tuple(cfg[e] for e in E8)

def stab(b):
    return frozenset(i for i, R in enumerate(ROT24) if act_block(R, b) == b)

def window(b, off):
    s = sigma_cfg(block_cfg(b))
    return tuple(s[add((1, 1, 1), add(off, e))] for e in E8)

def tarjan(nodes, succ):
    idx, low, on, st, out = {}, {}, set(), [], []
    counter = [0]
    sys.setrecursionlimit(100000)
    def sc(v):
        idx[v] = low[v] = counter[0]; counter[0] += 1; st.append(v); on.add(v)
        for w in succ(v):
            if w not in idx: sc(w); low[v] = min(low[v], low[w])
            elif w in on: low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = st.pop(); on.discard(w); comp.append(w)
                if w == v: break
            out.append(comp)
    for v in nodes:
        if v not in idx: sc(v)
    return out

def rot_index(M):
    return RIDX[M]

def about_axis(e):
    return [i for i, R in enumerate(ROT24) if app(R, e) == e]

if __name__ == '__main__':
    d = pickle.load(open(sys.argv[1], 'rb'))
    L = d['L']
    # equivariance of sigma under O
    for R in ROT24:
        for l in range(168):
            A, t = label_pose(l)
            kidsR = sorted(children(*compose((R, (0,0,0)), (A, t))))
            kids = sorted(compose((R, (0,0,0)), k) for k in children(A, t))
            assert kidsR == kids
    print('sigma commutes with all 24 rotations: yes')
    print('L invariant under O:', all(act_block(R, b) in L for R in ROT24 for b in L))
    # Phi-periodic legal blocks
    Phi = {b: window(b, (0, 0, 0)) for b in L}
    assert all(Phi[b] in L for b in L)
    per = set()
    for b in L:
        seen = []
        x = b
        for _ in range(len(L) + 1):
            x = Phi[x]
        # x is now on a cycle
        y = x
        while True:
            per.add(y); y = Phi[y]
            if y == x: break
    fixed = [b for b in per if Phi[b] == b]
    print('Phi-periodic legal blocks', len(per), 'of which fixed', len(fixed))
    from collections import Counter
    print('stabiliser orders of periodic blocks', Counter(len(stab(b)) for b in per))
    # stabiliser structure of the order-8 ones
    for b in per:
        S = stab(b)
        if len(S) > 1:
            axes = sorted(tuple(x for x in ROT24[i][0]) for i in S)
    # axial graphs
    results = {}
    for name, e, K in [('C2coord', (0, 0, 1), 'C2'), ('C4', (0, 0, 1), 'C4'), ('C2diag', (1, 1, 0), 'C2d'), ('C2diag-', (1, -1, 0), 'C2d')]:
        Oe = set(about_axis(e))
        if K == 'C2':
            Kset = {i for i in Oe if mul(ROT24[i], ROT24[i]) == I3}
        else:
            Kset = Oe
        stabs = {b: stab(b) for b in L}
        nodes = [b for b in L if (stabs[b] & Oe) == Kset]
        nodeset = set(nodes)
        W = {b: {j: window(b, scal(j, e)) for j in (-1, 0, 1)} for b in nodes}
        succ = lambda b: [W[b][j] for j in (-1, 0, 1) if W[b][j] in nodeset]
        comps = tarjan(nodes, succ)
        mixed = []
        for c in comps:
            cs = set(c)
            labs = {j for b in c for j in (-1, 0, 1) if W[b][j] in cs}
            if len(labs) >= 2: mixed.append((len(c), sorted(labs)))
            elif labs: pass
        nontriv = [(len(c), sorted({j for b in c for j in (-1,0,1) if W[b][j] in set(c)})) for c in comps if any(W[b][j] in set(c) for b in c for j in (-1,0,1))]
        print(name, 'e=', e, '|K|=', len(Kset), 'nodes', len(nodes), 'SCCs with a cycle:', nontriv, 'mixed:', mixed)
        results[name] = (nodes, W, comps, Kset, Oe)
        # full stabiliser histogram of nodes
        print('   full-stabiliser orders of G_K nodes', Counter(len(stabs[b]) for b in nodes))
    if len(sys.argv) > 2: pickle.dump({'per': per, 'fixed': fixed}, open(sys.argv[2], 'wb'))
