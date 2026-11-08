"""Item F: a handedness invariant for the 3D Chair44 hull, the analogue of item A.

Item A measures how positions and orientations are correlated inside a supertile: W_k = N M1 - conj(S) M0.
The 3D analogue has to be a rotation-invariant PSEUDOSCALAR: a number that rotations leave unchanged and that a
mirror would negate.  The Chair44 hull contains only proper rotations of the tile, so a nonzero value of such a
number is a chirality of the hull, just as Im Z_k is for the Spectre.

Definition.  Take the level-k chair supertile with the identity frame.  Each unit cell c has a position p_c and
the proper rotation R_c of the tile it belongs to.  Let w(R) be the axial vector of R
(w_i = eps_ijk R_kj, i.e. 2 sin(theta) times the axis).  Then
    X_k = sum_c (N p_c - S) . w(R_c),     N = number of cells, S = sum_c p_c.
Covariance: a global rotation Q sends p -> Q p and R -> Q R Q^-1 (frames relative to the root), and
w(Q R Q^-1) = Q w(R).  So X_k is rotation invariant.  A mirror M gives w(M R M^-1) = det(M) M w(R) = -M w(R),
so X_k changes sign.  Everything is an exact integer.  The size scale is N^2 2^k, which gives
    chi_k = X_k / (N^2 2^k).
The script also reports the symmetric part sum_c (N p_c - S) (x) w(R_c): its trace is X_k, and its traceless
part is a tensor (shape) correlation.

    python research/chair44_handedness.py KMAX OUT.json
"""
import sys, json, itertools
sys.path.insert(0, "."); sys.path.insert(0, "research")
import numpy as np
import chair44_twoadic as TA
import symchair44 as S

BITS = list(itertools.product((0, 1), repeat=3))


def axial(frame):
    M = np.array(S.frame_matrix(frame), dtype=np.int64)
    return np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]], np.int64)


def main():
    kmax = int(sys.argv[1]); out = sys.argv[2]
    labels, SIG = TA.labels_and_sigma(); SIG = np.array(SIG, np.int64)
    W = np.array([axial(f) for f, _ in labels])
    ident = [i for i, (f, k) in enumerate(labels) if f == ((0, 1, 2), (1, 1, 1))]
    assert len(ident) == 7
    # the 7 cells of the identity-frame chair, at their native positions
    res = []
    for k in range(0, kmax + 1):
        tot_X = 0; T = np.zeros((3, 3), dtype=object); N = 0; Ssum = np.zeros(3, dtype=object)
        blocks = []
        for kk, c in enumerate(S.CELLS):
            A = np.array([[[ident[kk]]]], np.int64)
            for _ in range(k):
                n = A.shape[0]; B = np.empty((2 * n,) * 3, np.int64)
                for b, (x, y, z) in enumerate(BITS):
                    B[x::2, y::2, z::2] = SIG[A, b]
                A = B
            blocks.append((np.array(c, np.int64) * 2 ** k, A))
        # first pass: N and S
        for off, A in blocks:
            n = A.shape[0]; idx = np.indices(A.shape).reshape(3, -1).T + off
            N += idx.shape[0]; Ssum = Ssum + idx.sum(0).astype(object)
        for off, A in blocks:
            idx = (np.indices(A.shape).reshape(3, -1).T + off).astype(object)
            w = W[A.ravel()].astype(object)
            d = idx * N - Ssum
            T = T + d.T @ w
        X = int(sum(T[i, i] for i in range(3)))
        chi = X / (N ** 2 * 2 ** k)
        Tl = [[int(T[i, j]) for j in range(3)] for i in range(3)]
        res.append(dict(k=k, N=N, X=str(X), chi=chi, T=[[str(v) for v in r] for r in Tl]))
        print(k, N, X, chi, flush=True)
    json.dump(dict(levels=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
