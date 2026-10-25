"""Like c44patch.py but uses the shortest closed walks with no 0-label run, and a larger region.
python3 c44patch2.py c1.pkl N radius"""
import sys, pickle, itertools
from c44patch import *

def short_cycles(W, comp, maxlen=4):
    cs = set(comp); out = []
    for b in comp:
        for labs in itertools.product((-1, 0, 1), repeat=1):
            pass
    for L_ in range(1, maxlen + 1):
        for b in comp:
            for labs in itertools.product((-1, 0, 1), repeat=L_):
                x = b; ok = True
                for j in labs:
                    y = W[x][j]
                    if y not in cs: ok = False; break
                    x = y
                if ok and x == b and len(set(labs)) > 1:
                    out.append((b, labs))
        if out: return out
    return out

if __name__ == '__main__':
    d = pickle.load(open(sys.argv[1], 'rb'))
    L = d['L']; N = int(sys.argv[2]); rad = int(sys.argv[3])
    stabs = {b: stab(b) for b in L}
    for name, e in [('C4', (0, 0, 1)), ('C2diag', (1, 1, 0))]:
        Oe = set(about_axis(e))
        nodes = [b for b in L if (stabs[b] & Oe) == Oe]
        ns = set(nodes)
        W = {b: {j: window(b, scal(j, e)) for j in (-1, 0, 1)} for b in nodes}
        comp = max(tarjan(nodes, lambda b: [W[b][j] for j in (-1, 0, 1) if W[b][j] in ns]), key=len)
        cyc = short_cycles(W, comp)
        print(name, 'shortest mixed closed walks:', sorted({c[1] for c in cyc}))
        b, labs = cyc[0]
        p = len(labs)
        seq = [labs[i % p] for i in range(N)]
        cfg, v0 = build(b, seq, e)
        found, ni, nc = patch_symmetries(cfg, v0, rad)
        print('   walk', labs, 'stab orders along walk', [len(stabs[x]) for x in [b]], 'N', N, 'tiles', nc, 'inner(radius %d)' % rad, ni,
              '|Sym(inner)|', len(found), 'linear parts', sorted(app(g[0], (1, 2, 3)) for g in found),
              'fixed-point translations', sorted({g[1] for g in found}))
