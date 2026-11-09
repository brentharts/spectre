"""Item G2: how the axis permutations of the chair act on the Chair44 bump/dent decoration.

The chair [0,2]^3 minus [1,2]^3 is invariant under the 6 permutations of the axes: 3 rotations about the
diagonal (det +1) and 3 mirrors (det -1).  Each of the 192 pyramid features (Tsiokos, CC BY 4.0; table
models/data/r44_features.csv of brentharts/spectre-monotile-py) is a position on a panel plus a signed height j
(+ bump, - dent).  For each permutation we count the features whose image carries the same height and those whose
image carries the opposite height.

    python research/chair44_decoration_symmetry.py r44_features.csv
"""
import csv, sys, itertools
from fractions import Fraction as F

PC = ("panel_cx", "panel_cy", "panel_cz")


def features(path):
    feat = {}
    for r in csv.DictReader(open(path)):
        ax = int(r["normal_axis"]); p = [F(r[k]) for k in ("apex_x", "apex_y", "apex_z")]
        p[ax] = F(r[PC[ax]])          # foot of the pyramid on its panel
        feat[(tuple(p), ax, int(r["outward_sign"]))] = int(r["coefficient"])
    return feat


def main():
    feat = features(sys.argv[1])
    for perm in itertools.permutations(range(3)):
        even = sum(perm[i] > perm[j] for i in range(3) for j in range(i + 1, 3)) % 2 == 0
        img = {(tuple(p[perm[i]] for i in range(3)), perm.index(ax), s): c for (p, ax, s), c in feat.items()}
        assert img.keys() == feat.keys()
        same = sum(img[k] == feat[k] for k in feat); opp = sum(img[k] == -feat[k] for k in feat)
        print(perm, "rotation" if even else "mirror", "same height", same, "opposite height", opp, "of", len(feat))


if __name__ == "__main__":
    main()
