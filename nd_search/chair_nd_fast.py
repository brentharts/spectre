"""Vectorised version of chair_nd.closure_test (Kamenetsky's coarsening closure, C1) for the N-dimensional chair.

Poses are integer ids g * TB + tindex(t), with g an index into the 2^N N! signed permutations and t in [-R, R]^N.
A table CLS[id] in {0 apart, 1 overlap, 2 touch} classifies every relative pose of two chairs.
Validated against chair_nd.closure_test (python3 chair_nd_fast.py validate).
"""
import itertools
import os
import pickle
import sys
import numpy as np
import chair_nd as C

R = 4


class Fast:
    def __init__(self, N, cache_dir=None):
        self.N = N
        env = self.env = C.make(N)
        self.G = env['ALL']
        self.gid = {g: i for i, g in enumerate(self.G)}
        ng = len(self.G)
        self.M = np.array([[[s if j == p else 0 for j in range(N)] for (p, s) in g] for g in self.G], dtype=np.int64)
        self.comp = np.array([[self.gid[env['gm'](a, b)] for b in self.G] for a in self.G], dtype=np.int64)
        self.inv = np.array([self.gid[env['ginv'](a)] for a in self.G], dtype=np.int64)
        self.W = 2 * R + 1
        self.TB = self.W ** N
        self.pw = np.array([self.W ** (N - 1 - i) for i in range(N)], dtype=np.int64)
        path = os.path.join(cache_dir or '.', 'cls_N%d_R%d.npy' % (N, R))
        if os.path.exists(path):
            self.CLS = np.load(path)
        else:
            CLS = np.zeros(ng * self.TB, dtype=np.int8)
            for gi, g in enumerate(self.G):
                for t in itertools.product(range(-R, R + 1), repeat=N):
                    k = env['classify']((g, t))
                    CLS[gi * self.TB + self.tidx(t)] = {'apart': 0, 'overlap': 1, 'touch': 2}[k]
            np.save(path, CLS)
            self.CLS = CLS

    def tidx(self, t):
        return int(sum((x + R) * self.W ** (self.N - 1 - i) for i, x in enumerate(t)))

    def rel_ids(self, ga, ta, gb, tb):
        """vectorised rel((ga,ta),(gb,tb)) -> (ids, inrange mask); ga, gb int arrays, ta, tb (n,N)"""
        gi = self.inv[ga]
        rg = self.comp[gi, gb]
        d = tb - ta
        rt = np.einsum('nij,nj->ni', self.M[gi], d)
        ok = np.all(np.abs(rt) <= R, axis=1)
        ids = rg * self.TB + np.clip(rt + R, 0, self.W - 1) @ self.pw
        return ids, ok

    def closure(self, assign, cap=400, want_sigma0=False):
        N = self.N
        CH = self.env['children_of'](assign)
        cg = np.array([self.gid[g] for g, t in CH], dtype=np.int64)
        ct = np.array([t for g, t in CH], dtype=np.int64)
        K = len(CH)
        I, J = np.meshgrid(np.arange(K), np.arange(K), indexing='ij')
        I, J = I.ravel(), J.ravel()

        def child_pairs(Sg, ST):
            """for supertiles S_n = (Sg[n], ST[n]) vs the reference, the relative pose ids of all K*K child pairs"""
            n = len(Sg)
            gb = self.comp[np.repeat(Sg, K * K), np.tile(cg[J], n)]
            tb = np.einsum('nij,nj->ni', self.M[np.repeat(Sg, K * K)], np.tile(ct[J], (n, 1))) + np.repeat(ST, K * K, axis=0)
            ga = np.tile(cg[I], n)
            ta = np.tile(ct[I], (n, 1))
            ids, ok = self.rel_ids(ga, ta, gb, tb)
            cls = np.where(ok, self.CLS[ids], 0)
            return ids.reshape(n, K * K), cls.reshape(n, K * K)

        idg = self.gid[self.env['ID']]
        # P1
        ids, cls = child_pairs(np.array([idg]), np.zeros((1, N), dtype=np.int64))
        P = set(ids[0][(cls[0] == 2)].tolist())
        P0 = set(P)
        frontier = set(P)
        while frontier:
            fr = np.array(sorted(frontier))
            g = fr // self.TB
            t = np.array([[(r // self.W ** (N - 1 - i)) % self.W - R for i in range(N)] for r in fr % self.TB], dtype=np.int64)
            ids, cls = child_pairs(g, 2 * t)
            new = set(ids[cls == 2].tolist())
            frontier = new - P0
            P0 |= new
            if len(P0) > cap:
                return ('cap0', len(P0), None)

        def Sigma(Pset):
            Parr = np.zeros(len(self.G) * self.TB, dtype=bool)
            Parr[list(Pset)] = True
            pl = np.array(sorted(Pset))
            pg = pl // self.TB
            pt = np.array([[(r // self.W ** (N - 1 - i)) % self.W - R for i in range(N)] for r in pl % self.TB], dtype=np.int64)
            # b = a o p for each child a, p in P; S = b o c^{-1} for each child c
            cand = set()
            for a in range(K):
                bg = self.comp[cg[a], pg]
                btv = np.einsum('ij,nj->ni', self.M[cg[a]], pt) + ct[a]
                for c in range(K):
                    Sg = self.comp[bg, self.inv[cg[c]]]
                    ST = btv - np.einsum('nij,j->ni', self.M[Sg], ct[c])
                    for x in zip(Sg.tolist(), map(tuple, ST.tolist())):
                        cand.add(x)
            cand.discard((idg, (0,) * N))
            cand = sorted(cand)
            Sg = np.array([c[0] for c in cand], dtype=np.int64)
            ST = np.array([c[1] for c in cand], dtype=np.int64)
            out = []
            B = 4096
            for s in range(0, len(cand), B):
                ids, cls = child_pairs(Sg[s:s + B], ST[s:s + B])
                bad = (cls == 1) | ((cls == 2) & ~Parr[ids])
                good = ~bad.any(axis=1)
                for k in np.nonzero(good)[0]:
                    out.append((int(Sg[s + k]), tuple(int(v) for v in ST[s + k])))
                if len(out) > cap:
                    break
            return out

        Pcur = set(P0)
        sizes = [len(Pcur)]
        s0 = None
        while True:
            Sg = Sigma(Pcur)
            if s0 is None:
                s0 = len(Sg)
            if len(Sg) > cap:
                return ('cap', len(Sg), sizes) if not want_sigma0 else ('cap', len(Sg), sizes, s0)
            if any(any(v % 2 for v in T) for g, T in Sg):
                return ('offset', len(Pcur), sizes)
            assert all(abs(v) <= 2 * R for g, T in Sg for v in T)
            newP = Pcur | {g * self.TB + self.tidx(tuple(v // 2 for v in T)) for g, T in Sg}
            sizes.append(len(newP))
            if newP == Pcur:
                assert {g * self.TB + self.tidx(tuple(v // 2 for v in T)) for g, T in Sg} == Pcur
                return ('closed', len(Pcur), sizes)
            if len(newP) > cap:
                return ('cap', len(newP), sizes)
            Pcur = newP


if __name__ == '__main__' and sys.argv[1:2] == ['validate']:
    import random, time
    F3 = Fast(3, cache_dir=sys.argv[2] if len(sys.argv) > 2 else '.')
    choices = [C.homochiral_choices(F3.env, v) for v in F3.env['OUTER']]
    res = {}
    t0 = time.time()
    for combo in itertools.product(*choices):
        a = dict(zip(F3.env['OUTER'], combo))
        r = F3.closure(a, cap=300)
        res[r[0]] = res.get(r[0], 0) + 1
        if r[0] == 'closed':
            print('closed', combo, r[1], r[2])
    print('N=3 fast:', res, '%.0fs' % (time.time() - t0))
    F4 = Fast(4, cache_dir=sys.argv[2] if len(sys.argv) > 2 else '.')
    rng = random.Random(5)
    for k in range(3):
        a = {v: rng.choice(C.homochiral_choices(F4.env, v)) for v in F4.env['OUTER']}
        t = time.time(); r1 = F4.closure(a, cap=400); t1 = time.time() - t
        t = time.time(); r2 = C.closure_test(F4.env, a, cap=400); t2 = time.time() - t
        print('N=4 sample', k, 'fast', r1[:2], '%.2fs' % t1, ' slow', r2[:2], '%.2fs' % t2)
