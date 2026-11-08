"""Item F follow-up: test whether the level-k identity-frame Chair44 supertile is invariant under the mirror
P = (swap x and y), acting on cell positions (p -> P p + shift) and on tile frames (R -> P R P).
If it is, for every k, every pseudoscalar of the supertiles (e.g. X_k of chair44_handedness.py) vanishes.

    python research/chair44_mirror.py KMAX OUT.json
"""
import sys, json, itertools
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
import chair44_twoadic as TA
import symchair44 as S

BITS = list(itertools.product((0, 1), repeat=3))
P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], np.int64)


def main():
    kmax = int(sys.argv[1]); out = sys.argv[2]
    labels, SIG = TA.labels_and_sigma(); SIG = np.array(SIG, np.int64)
    Ms = [np.array(S.frame_matrix(f), np.int64) for f, _ in labels]
    keys = sorted({M.tobytes() for M in Ms}); fid = {b: i for i, b in enumerate(keys)}
    F = np.array([fid[M.tobytes()] for M in Ms])  # frame id of each label
    Mof = {fid[M.tobytes()]: M for M in Ms}
    mirror_f = {i: fid.get((P @ M @ P).tobytes(), -1) for i, M in Mof.items()}
    ident = [i for i, (f, k) in enumerate(labels) if f == ((0, 1, 2), (1, 1, 1))]
    res = []
    for k in range(kmax + 1):
        pts = []
        for kk, c in enumerate(S.CELLS):
            A = np.array([[[ident[kk]]]], np.int64)
            for _ in range(k):
                n = A.shape[0]; B = np.empty((2 * n,) * 3, np.int64)
                for b, (x, y, z) in enumerate(BITS):
                    B[x::2, y::2, z::2] = SIG[A, b]
                A = B
            idx = np.indices(A.shape).reshape(3, -1).T + np.array(c, np.int64) * 2 ** k
            pts.append(np.column_stack([idx, F[A.ravel()]]))
        X = np.concatenate(pts)
        Y = np.column_stack([X[:, :3] @ P.T, np.array([mirror_f[f] for f in X[:, 3]])])
        Y[:, :3] += X[:, :3].min(0) - Y[:, :3].min(0)
        ok = bool((Y[:, 3] >= 0).all()) and set(map(tuple, X.tolist())) == set(map(tuple, Y.tolist()))
        res.append(dict(k=k, N=len(X), invariant=ok)); print(k, len(X), ok, flush=True)
    json.dump(dict(levels=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
