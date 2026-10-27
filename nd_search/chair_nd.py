"""The N-dimensional chair C_N = [0,2]^N minus (1,2]^N, its substitution by a frame assignment, and Kamenetsky's
coarsening-closure test (C1).  Written from the definitions in Kamenetsky, arXiv:2610.00916, Secs. 5-6; no code from
there is used.

Pose (g, t): x -> g x + t, g a signed permutation (tuple of (perm, signs)), t integer, in unit-cube coordinates.
Cells are stored as doubled centres 2u+1.  The level-1 supertile 2C_N has the central child (id, (1,...,1)) and, for
each v in {0,1}^N other than (1,...,1), the child (D_s(v) pi_v, c(v) + 2v), where D_s(v) flips the coordinates with
v_i = 1 and c(v)_i = 2 there.  An assignment is homochiral when every child frame is a rotation.
"""
import itertools
import sys
from functools import lru_cache


def make(N):
    perms = list(itertools.permutations(range(N)))
    signs = list(itertools.product((1, -1), repeat=N))

    def perm_parity(p):
        s, seen = 0, [False] * N
        for i in range(N):
            if not seen[i]:
                j, L = i, 0
                while not seen[j]:
                    seen[j] = True
                    j = p[j]
                    L += 1
                s += L - 1
        return s % 2

    # g as a tuple of (column index, sign) per row: (g x)_i = s_i x_{p_i}
    def gm(g, h):
        # (g h x)_i = s_i (h x)_{p_i} = s_i t_{p_i} x_{q_{p_i}}
        return tuple((h[g[i][0]][0], g[i][1] * h[g[i][0]][1]) for i in range(N))

    def ginv(g):
        out = [None] * N
        for i, (p, s) in enumerate(g):
            out[p] = (i, s)
        return tuple(out)

    def gv(g, v):
        return tuple(s * v[p] for p, s in g)

    def det(g):
        return (-1) ** perm_parity([p for p, s in g]) * eval('*'.join(str(s) for p, s in g))

    ALL = [tuple(zip(p, s)) for p in perms for s in signs]
    ROT = [g for g in ALL if det(g) == 1]
    ID = tuple((i, 1) for i in range(N))
    one = (1,) * N
    CELLS = [tuple(2 * x + 1 for x in u) for u in itertools.product((0, 1), repeat=N) if u != one]
    CSET = frozenset(CELLS)

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    @lru_cache(maxsize=None)
    def cells(P):
        g, t = P
        t2 = tuple(2 * x for x in t)
        return frozenset(add(gv(g, c), t2) for c in CELLS)

    def compose(A, B):
        return (gm(A[0], B[0]), add(gv(A[0], B[1]), A[1]))

    def rel(A, B):
        gi = ginv(A[0])
        return (gm(gi, B[0]), gv(gi, sub(B[1], A[1])))

    UNIT = [tuple(1 if j == i else 0 for j in range(N)) for i in range(N)]

    def touching_cells(a, b):
        for c in a:
            for e in UNIT:
                e2 = tuple(2 * x for x in e)
                if add(c, e2) in b or sub(c, e2) in b:
                    return True
        return False

    TOUCH = {}

    def classify(p):
        """None if apart, 'overlap', or 'touch'"""
        r = TOUCH.get(p)
        if r is None:
            a, b = cells((ID, (0,) * N)), cells(p)
            if a & b:
                r = 'overlap'
            elif touching_cells(a, b):
                r = 'touch'
            else:
                r = 'apart'
            TOUCH[p] = r
        return r

    def flips(v):
        return tuple((i, -1 if v[i] else 1) for i in range(N))

    OUTER = [v for v in itertools.product((0, 1), repeat=N) if v != one]

    def children_of(assign):
        """assign: dict v -> permutation tuple pi_v"""
        ch = [(ID, one)]
        for v in OUTER:
            pi = tuple((assign[v][i], 1) for i in range(N))
            g = gm(flips(v), pi)
            t = tuple((2 if v[i] else 0) + 2 * v[i] for i in range(N))
            ch.append((g, t))
        return ch

    return dict(N=N, ALL=ALL, ROT=ROT, ID=ID, CELLS=CELLS, cells=cells, compose=compose, rel=rel, classify=classify,
                children_of=children_of, OUTER=OUTER, gm=gm, ginv=ginv, gv=gv, det=det, perm_parity=perm_parity,
                perms=perms, flips=flips)


def closure_test(env, assign, cap=300):
    """Kamenetsky's (C1): returns ('closed', |P*|) or ('offset', i) or ('cap', |P|)."""
    N = env['N']
    ID = env['ID']
    cells, compose, rel, classify = env['cells'], env['compose'], env['rel'], env['classify']
    CH = env['children_of'](assign)
    # the children must tile 2C_N
    allc = [c for ch in CH for c in cells(ch)]
    assert len(allc) == len(set(allc)) == (2 ** N - 1) * 2 ** N
    REF = [(g, t) for g, t in CH]

    def ch_of(S):
        return [compose(S, c) for c in CH]

    def touch(a, b):
        return classify(rel(a, b)) == 'touch'

    P1 = {rel(a, b) for a in REF for b in REF if a != b and touch(a, b)}

    def X(p):
        g, t = p
        S2 = (g, tuple(2 * x for x in t))
        return {rel(a, b) for a in REF for b in ch_of(S2) if touch(a, b)}

    P0 = set(P1)
    frontier = set(P1)
    while frontier:
        new = set()
        for p in frontier:
            new |= X(p)
        frontier = new - P0
        P0 |= new
        if len(P0) > cap:
            return ('cap0', len(P0), None)

    def Sigma(P):
        out = set()
        for a in REF:
            for p in P:
                b = compose(a, p)
                for c in CH:
                    G = env['gm'](b[0], env['ginv'](c[0]))
                    T = tuple(x - y for x, y in zip(b[1], env['gv'](G, c[1])))
                    S = (G, T)
                    if S in out or S == (ID, (0,) * N):
                        continue
                    ok = True
                    for y in ch_of(S):
                        for x in REF:
                            k = classify(rel(x, y))
                            if k == 'overlap' or (k == 'touch' and rel(x, y) not in P):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        out.add(S)
                        if len(out) > cap:
                            return out
        return out

    P = set(P0)
    sizes = [len(P)]
    while True:
        Sg = Sigma(P)
        if len(Sg) > cap:
            return ('cap', len(Sg), sizes)
        if any(any(x % 2 for x in S[1]) for S in Sg):
            return ('offset', len(P), sizes)
        newP = P | {(G, tuple(x // 2 for x in T)) for G, T in Sg}
        sizes.append(len(newP))
        if newP == P:
            assert Sg == {(g, tuple(2 * x for x in t)) for g, t in P}
            return ('closed', len(P), sizes)
        if len(newP) > cap:
            return ('cap', len(newP), sizes)
        P = newP


def homochiral_choices(env, v):
    """permutations pi with D_s(v) pi a rotation: parity(pi) = |v| mod 2"""
    return [p for p in env['perms'] if env['perm_parity'](list(p)) == sum(v) % 2]


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    env = make(N)
    if N == 3:
        # Theorem B of Kamenetsky: exactly 3 of the 3^7 homochiral assignments are coarsening-closed (|P*| = 44)
        choices = [homochiral_choices(env, v) for v in env['OUTER']]
        res = {}
        closed = []
        for combo in itertools.product(*choices):
            a = dict(zip(env['OUTER'], combo))
            r = closure_test(env, a)
            res[r[0]] = res.get(r[0], 0) + 1
            if r[0] == 'closed':
                closed.append((combo, r[1], r[2]))
        print("N=3 over %d assignments:" % sum(res.values()), res)
        for c in closed:
            print("  closed:", c)
