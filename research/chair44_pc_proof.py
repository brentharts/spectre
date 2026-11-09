"""Item Γ14, decoration half: the Chair44 hull with bump/dent decoration is invariant under PC.

P is the x <-> y mirror and C flips every feature height (bump <-> dent).  A tile with frame f carries the
decoration f(D) of the reference chair.  Two finite facts close the argument:
  1. chair44_mirror_proof: P commutes with the substitution on (frame, cell) labels and fixes the seed, so P maps
     every identity-frame supertile to itself, sending the tile with frame f to the tile with frame P f P.
  2. G2 on the reference chair: P(D) = -D = C(D) exactly, feature by feature (192 of 192).
Then PC sends the decoration f(D) of a tile to P f P (P C D) = (P f P)(D), the decoration the mirrored tile carries,
so PC maps every decorated supertile to itself.

    python research/chair44_pc_proof.py r44_features.csv
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "research")
import chair44_twoadic as TA
import symchair44 as S
from chair44_mirror_proof import P, conj
from chair44_decoration_symmetry import features


def frame_part():
    labels, sigma = TA.labels_and_sigma()
    index = {l: i for i, l in enumerate(labels)}
    cidx = {c: k for k, c in enumerate(S.CELLS)}
    m = [index[(conj(f), cidx[tuple(S.CELLS[k][P[i]] for i in range(3))])] for f, k in labels]
    pb = [TA.BITS.index(tuple(a[P[i]] for i in range(3))) for a in TA.BITS]
    bad = sum(sigma[m[l]][pb[b]] != m[sigma[l][b]] for l in range(len(labels)) for b in range(8))
    seed = {index[(S.IDENTITY, k)] for k in range(7)}
    return bad == 0 and {m[l] for l in seed} == seed


def decoration_part(path):
    feat = features(path)
    img = {(tuple(p[P[i]] for i in range(3)), P[ax], s): c for (p, ax, s), c in feat.items()}
    ok = img.keys() == feat.keys() and all(img[k] == -feat[k] for k in feat)
    return ok, len(feat)


def main():
    a = frame_part(); b, n = decoration_part(sys.argv[1])
    print("frame-level mirror commutes and fixes seed:", a)
    print("P(D) = -D on all %d features:" % n, b)
    print("PROVED" if a and b else "NOT PROVED")


if __name__ == "__main__":
    main()
