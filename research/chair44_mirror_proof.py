"""Item Γ14: a finite proof that the Chair44 substitution commutes with the x <-> y mirror.

chair44_twoadic.labels_and_sigma() describes the substitution on unit cells: a cell carries a label (frame f,
native cell index k) for the 24 rotations f and the 7 cells c_k of the chair, and sigma[label][bit] is the
label of the child cell in octant `bit` of the doubled cell.  The mirror P swaps x and y.  It maps integer unit
cubes to integer unit cubes exactly, a tile with frame f to a tile with frame P f P (again a rotation), and the
cell f(c_k) of that tile to (P f P)(P c_k).  So P acts on labels by
    m(f, k) = (P f P, index of P c_k),
and on octants by bit a -> P a.  The script checks, for all 168 labels and 8 octants,
    sigma[m(l)][P b] == m(sigma[l][b]),
and that the 7 cells of the identity-frame chair (the level-0 seed) form an m-invariant set.  Together these give,
by induction on k, that every level-k identity-frame supertile equals its mirror image: the finite version of what
job 1715771 tested numerically for k <= 6.

    python research/chair44_mirror_proof.py
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "research")
import chair44_twoadic as TA
import symchair44 as S

P = (1, 0, 2)


def conj(f):
    """P f P for the frame f = (perm, signs) with (f x)_i = s_i x_{p_i}."""
    p, s = f
    # (P f P x)_i = (f P x)_{P i} = s_{P i} (P x)_{p_{P i}} = s_{P i} x_{P p_{P i}}
    return (tuple(P[p[P[i]]] for i in range(3)), tuple(s[P[i]] for i in range(3)))


def main():
    labels, sigma = TA.labels_and_sigma()
    index = {l: i for i, l in enumerate(labels)}
    cidx = {c: k for k, c in enumerate(S.CELLS)}
    m = [index[(conj(f), cidx[tuple(S.CELLS[k][P[i]] for i in range(3))])] for f, k in labels]
    pb = [TA.BITS.index(tuple(a[P[i]] for i in range(3))) for a in TA.BITS]
    bad = [(l, b) for l in range(len(labels)) for b in range(8) if sigma[m[l]][pb[b]] != m[sigma[l][b]]]
    seed = {index[(S.IDENTITY, k)] for k in range(7)}
    print("labels", len(labels), "mirror is an involution:", all(m[m[l]] == l for l in range(len(m))))
    print("commutation failures:", len(bad), "of", 8 * len(labels))
    print("seed invariant:", {m[l] for l in seed} == seed)
    print("PROVED" if not bad and {m[l] for l in seed} == seed else "NOT PROVED")


if __name__ == "__main__":
    main()
