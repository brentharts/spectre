"""Item F: rational Cech cohomology of the 3D Chair44 hull (Anderson-Putnam, collared cubes).

Cells.  A collared cube type is a legal 3x3x3 window of the cell labelling (168 labels, `chair44_twoadic`); the
middle cube is the tile.  Each type has 27 local cells (eps, D): direction set D, offset eps_i in {0,1} for i not in
D, 0 for i in D.  The Anderson-Putnam complex K glues two local cells when they are the same cell of the tiling in
some legal patch; the classes come from union-find over every face, edge and vertex shared by neighbouring cubes in
large patches sigma^n(j).  Collaring makes the substitution force the border, so Cech H^k(Omega) = lim (H^k(K), f*).

Completeness.  The legal 4x4x4 windows found in the patches are checked to be closed under sigma (every 4x4x4
window of sigma(w) is again found).  Every identification is a sub-window of a 4x4x4 window, so closure means the
complex is the complete one.

Ranks.  f is a chain map, so C = C_nil + C_inv (Fitting decomposition) as complexes, and H(C_inv) is the direct
limit: dim H^k(Omega; Q) = rank f^m|C_k - rank d_k f^m - rank d_{k+1} f^m, for m with rank f^m = rank f^(m+1).
Ranks are computed in floating point (SVD) of (f/2^k)^m.  Checks: H^0 = 1, plus the Euler characteristic.  The
nonzero eigenvalues of f on each C_k are reported too, since the cohomology eigenvalues are among them.

    python research/chair44_cohomology.py DEPTH OUT.json
"""
import sys, json, itertools
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import chair44_twoadic as TA

BITS = list(itertools.product((0, 1), repeat=3))
labels, SIG = TA.labels_and_sigma()
SIG = np.array(SIG, np.int64)
NL = len(labels)


def inflate(A):
    """(..., N, N, N) label array -> (..., 2N, 2N, 2N)."""
    sh = A.shape[:-3]; N = A.shape[-3:]
    out = np.empty(sh + tuple(2 * n for n in N), np.int64)
    for b, (x, y, z) in enumerate(BITS):
        out[..., x::2, y::2, z::2] = SIG[A, b]
    return out


def windows(A, s):
    """All windows of shape s of each patch in A (L, N, N, N), as rows of uint8 (one per window)."""
    L, N = A.shape[0], A.shape[1]
    v = np.lib.stride_tricks.sliding_window_view(A, s, axis=(1, 2, 3))
    return v.reshape(-1, s[0] * s[1] * s[2]).astype(np.uint8), v.shape[1:4]


def as_keys(rows):
    rows = np.ascontiguousarray(rows)
    return rows.view(np.dtype((np.void, rows.shape[1]))).ravel()


SLOTS = [(eps, D) for D in itertools.chain.from_iterable(itertools.combinations(range(3), k) for k in range(4))
         for eps in itertools.product(*[(0,) if i in D else (0, 1) for i in range(3)])]
SLOT = {s: i for i, s in enumerate(SLOTS)}
assert len(SLOTS) == 27


def main():
    depth = int(sys.argv[1]); out = sys.argv[2]
    A = np.arange(NL, dtype=np.int64).reshape(NL, 1, 1, 1)
    for _ in range(depth):
        A = inflate(A)
    print("patches", A.shape, flush=True)
    # cube types: 3x3x3 windows
    w3, shp = windows(A, (3, 3, 3))
    k3 = as_keys(w3)
    uk, inv = np.unique(k3, return_inverse=True)
    T = len(uk); tid = inv.reshape(A.shape[0], *shp)
    types = np.frombuffer(uk.tobytes(), np.uint8).reshape(T, 27).astype(np.int64)
    print("collared cube types", T, flush=True)
    # closure of the 4x4x4 language
    w4, _ = windows(A, (4, 4, 4))
    k4 = np.unique(as_keys(w4)); V4 = np.frombuffer(k4.tobytes(), np.uint8).reshape(-1, 4, 4, 4).astype(np.int64)
    big = inflate(V4)
    w4b, _ = windows(big, (4, 4, 4))
    missing = int((~np.isin(np.unique(as_keys(w4b)), k4)).sum())
    print("4x4x4 windows", len(k4), "missing after sigma", missing, flush=True)
    # union-find over (type, slot): neighbouring cubes share cells
    I, J = [], []
    L, n1, n2, n3 = tid.shape
    for s_idx, (eps, D) in enumerate(SLOTS):
        # the same tiling cell seen from the neighbour cube c+d (d_i in {0,1} with eps_i=1 -> neighbour at +1, slot 0)
        free = [i for i in range(3) if i not in D and eps[i] == 1]
        for r in range(1, len(free) + 1):
            for sub in itertools.combinations(free, r):
                e2 = tuple(0 if i in sub else eps[i] for i in range(3))
                t_idx = SLOT[(e2, D)]
                sl_a = tuple(slice(0, [n1, n2, n3][i] - (1 if i in sub else 0)) for i in range(3))
                sl_b = tuple(slice(1 if i in sub else 0, [n1, n2, n3][i]) for i in range(3))
                a = tid[(slice(None),) + sl_a].ravel(); b = tid[(slice(None),) + sl_b].ravel()
                pr = np.unique(np.stack([a * 27 + s_idx, b * 27 + t_idx], 1), axis=0)
                I.append(pr[:, 0]); J.append(pr[:, 1])
    I = np.concatenate(I); J = np.concatenate(J)
    g = coo_matrix((np.ones(len(I)), (I, J)), shape=(27 * T, 27 * T))
    _, cls = connected_components(g, directed=False)
    cls = cls.reshape(T, 27)
    # renumber classes per dimension
    dim_of = np.array([len(D) for _, D in SLOTS])
    cid = {}; ncell = [0] * 4; cmap = np.zeros((T, 27), np.int64)
    for t in range(T):
        for s in range(27):
            key = (dim_of[s], cls[t, s])
            if key not in cid:
                cid[key] = ncell[dim_of[s]]; ncell[dim_of[s]] += 1
            cmap[t, s] = cid[key]
    print("cells per dimension", ncell, flush=True)
    # boundary matrices d_k : C_k -> C_{k-1}
    dmat = {}
    for k in (1, 2, 3):
        M = np.zeros((ncell[k - 1], ncell[k]))
        done = set()
        for t in range(T):
            for s, (eps, D) in enumerate(SLOTS):
                if len(D) != k or cmap[t, s] in done: continue
                done.add(cmap[t, s]); col = cmap[t, s]
                for r, i in enumerate(D):
                    Dm = tuple(x for x in D if x != i)
                    hi = tuple(1 if j == i else eps[j] for j in range(3))
                    M[cmap[t, SLOT[(hi, Dm)]], col] += (-1) ** r
                    M[cmap[t, SLOT[(eps, Dm)]], col] -= (-1) ** r
        dmat[k] = M
    # substitution chain map f_k
    tkey = {bytes(types[t].astype(np.uint8).tobytes()): t for t in range(T)}
    big = inflate(types.reshape(T, 3, 3, 3))  # (T, 6, 6, 6)
    child = np.zeros((T, 8), np.int64); bad = 0
    for t in range(T):
        for b, (x, y, z) in enumerate(BITS):
            key = big[t, 1 + x:4 + x, 1 + y:4 + y, 1 + z:4 + z].astype(np.uint8).tobytes()
            if key in tkey: child[t, b] = tkey[key]
            else: bad += 1
    print("children missing", bad, flush=True)
    F = {k: np.zeros((ncell[k], ncell[k])) for k in range(4)}
    done = set()
    for t in range(T):
        for s, (eps, D) in enumerate(SLOTS):
            k = len(D); col = cmap[t, s]
            if (k, col) in done: continue
            done.add((k, col))
            for delta in itertools.product(*[(0, 1) if i in D else (0,) for i in range(3)]):
                q = [2 * eps[i] + delta[i] for i in range(3)]
                b = tuple(min(q[i], 1) for i in range(3)); e2 = tuple(q[i] - b[i] for i in range(3))
                F[k][cmap[child[t, BITS.index(b)], SLOT[(e2, D)]], col] += 1
    # consistency: f is a chain map
    chain_ok = all(np.allclose(dmat[k] @ F[k], F[k - 1] @ dmat[k]) for k in (1, 2, 3))
    print("chain map", chain_ok, flush=True)
    res = dict(depth=depth, cube_types=T, windows4=len(k4), missing_after_sigma=missing, children_missing=bad,
               cells=ncell, chain_map=chain_ok)
    # Fitting-invariant ranks
    P = {k: np.linalg.matrix_power(F[k] / 2 ** k, 1) for k in range(4)}
    hist = []
    Pm = {k: np.eye(ncell[k]) for k in range(4)}
    prev = None
    for m in range(1, 13):
        Pm = {k: P[k] @ Pm[k] for k in range(4)}
        r = [int(np.linalg.matrix_rank(Pm[k])) for k in range(4)]
        rd = {k: int(np.linalg.matrix_rank(dmat[k] @ Pm[k])) for k in (1, 2, 3)}
        H = [r[k] - (rd[k] if k > 0 else 0) - (rd[k + 1] if k < 3 else 0) for k in range(4)]
        hist.append(dict(m=m, rank_f=r, rank_df=rd, H=H)); print(m, r, rd, H, flush=True)
        if prev == r and m >= 4: break
        prev = r
    res["iterations"] = hist; res["H_Q"] = hist[-1]["H"]
    res["euler"] = sum((-1) ** k * hist[-1]["rank_f"][k] for k in range(4))
    ev = {}
    for k in range(4):
        e = np.linalg.eigvals(F[k]); e = e[np.abs(e) > 1e-6]
        c = {}
        for z in e:
            key = "%.4f%+.4fi" % (z.real, z.imag); c[key] = c.get(key, 0) + 1
        ev[k] = dict(sorted(c.items(), key=lambda kv: -abs(complex(kv[0].replace("i", "j")))))
    res["nonzero_eigenvalues_f"] = ev
    print("H^k(Omega;Q) dims", res["H_Q"], "euler", res["euler"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
