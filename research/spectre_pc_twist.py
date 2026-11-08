"""Item G4: is the ribbon twist sign the Spectre's analogue of the Chair44 bump/dent exchange C?

Mirror the lifted Spectre tree through the xz-plane (M = diag(1,-1,1)): centres c -> M c, transforms
T -> M T M, local placements L -> M L M (so rotation angles a -> -a and reflection axes a/2 -> -a/2).
Then compare, for every edge of the tree, the mirror image of the twist-sign-s ribbon (centre line P and
width offsets) with the ribbon that spectrespace.edge_ribbon builds on the mirrored tree with sign s' and
ribbon angle g'.  Exactly one of (s' = -s, g' = -g) should match, and that pair is the Spectre's version of PC.

    python research/spectre_pc_twist.py [depth]
"""
import sys, os, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import spectrespace as S

M2 = np.diag([1.0, -1.0]); M3 = np.diag([1.0, -1.0, 1.0])


def conj23(A):
    B = A.copy(); B[:, :2] = M2 @ A[:, :2] @ M2; B[:, 2] = M2 @ A[:, 2]; return B


def mirrored(space):
    sp = copy.deepcopy(space)
    for nd in sp.nodes:
        nd.local = conj23(nd.local); nd.T = conj23(nd.T)
        nd.center = M2 @ nd.center
        if nd.corners is not None: nd.corners = nd.corners @ M2
    return sp


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sp = S.SpectreSpace(depth); mp = mirrored(sp)
    g, z, m = 20.0, 50.0, 17
    for s in ("+1", "-1"):
        for s2 in ("+1", "-1"):
            for g2 in (g, -g):
                worst = 0.0
                for nd, nd2 in zip(sp.nodes[1:], mp.nodes[1:]):
                    P, off, _, _ = S.edge_ribbon(sp, nd, m, z, g, 0.09, s)
                    Q, off2, _, _ = S.edge_ribbon(mp, nd2, m, z, g2, 0.09, s2)
                    worst = max(worst, np.abs(P @ M3 - Q).max(), np.abs(off @ M3 - off2).max())
                print(f"mirror of (sign {s}, angle {g:+}) vs (sign {s2}, angle {g2:+}): max deviation {worst:.2e}",
                      "MATCH" if worst < 1e-9 else "")


if __name__ == "__main__":
    main()
