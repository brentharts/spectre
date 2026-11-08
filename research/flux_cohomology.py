"""R3: is Brent's Z2 vertex flux a pattern-equivariant coboundary?

flux(v) = (-1)^w(v) (spectre_holonomy.test_flux rule) is a local, hence pattern-equivariant, Z2 2-cochain on
the vertex stars.  spectre_holonomy shows flux = product of the edge signs eps relating local and tree lifts,
i.e. flux = delta(eps) -- but eps is defined through the (non-local) tree, so this does not decide the class
in the pattern-equivariant (= Cech) cohomology H^2(Omega; Z2).

Test: find a Z2 1-cochain c on tile adjacencies that is pattern-equivariant with radius R (c(A,B) depends only
on the labelled patch within distance R of the shared edge, read in A's frame, symmetrised in A<->B) with
    sum_{edges e of star v} c(e) = [flux(v) = -1]   for every interior vertex v whose R-patches are complete.
Solved over GF(2).  Solvable at some R  ==>  the flux class is zero (vortices are removable by a local
re-gauging).  Unsolvable for all R tried ==> evidence (not proof) for a nonzero class; the obstruction
vector (a GF(2) combination of star equations with zero left side and odd right side) is reported.
Usage: python research/flux_cohomology.py DEPTH R1,R2,... OUT.json
"""
import sys, json
from collections import Counter
sys.path.insert(0, ".")
import numpy as np
from scipy.spatial import cKDTree
import spectrespace as S
import spectre_holonomy as H


def stars_and_flux(space):
    leaves = space.leaves()
    out = []
    for key, ws in H.vertex_stars(space).items():
        if abs(sum(w[2] for w in ws) - 2 * np.pi) > 1e-6:
            continue
        ws = sorted(ws, key=lambda w: w[1]); k = len(ws)
        tot = 0
        for i in range(k):
            a, b = ws[i][0], ws[(i + 1) % k][0]
            d, _ = H.rel_rotation_deg(leaves[a].T[:, :2], leaves[b].T[:, :2]); tot += d
        out.append((np.array(ws[0][3]), [w[0] for w in ws], (tot // 360) % 2))
    return out


def main():
    depth = int(sys.argv[1]); radii = [float(r) for r in sys.argv[2].split(",")]; outp = sys.argv[3]
    space = S.SpectreSpace(depth)
    leaves = space.leaves()
    cen = np.array([nd.verts.mean(0) for nd in leaves])
    lab = [nd.label for nd in leaves]
    Tinv = []
    for nd in leaves:
        A = nd.T[:, :2]; Ai = np.linalg.inv(A); Tinv.append((Ai, -Ai @ nd.T[:, 2]))
    tree = cKDTree(cen)
    stars = stars_and_flux(space)
    # patch boundary: vertices of the convex-ish hull region; use distance from the patch centre of mass
    allv = np.concatenate([nd.verts for nd in leaves]); c0 = allv.mean(0)
    # a vertex is "complete for R" if the disc of radius R+diam around it is covered: approximate by
    # requiring every star vertex to be interior and the disc to lie inside the convex hull minus margin
    from scipy.spatial import ConvexHull
    hull = ConvexHull(allv)
    def inside_margin(p, m):
        return np.all(hull.equations[:, :2] @ p + hull.equations[:, 2] <= -m)
    diam = np.linalg.norm(leaves[0].verts - leaves[0].verts.mean(0), axis=1).max() * 2
    res = dict(depth=depth, pieces=len(leaves), stars=len(stars),
               flux_minus=int(sum(s[2] for s in stars)), runs={})
    print("depth", depth, "pieces", len(leaves), "interior stars", len(stars), "flux -1", res["flux_minus"], flush=True)

    def key_in(a, mid, R):
        Ai, t = Tinv[a]
        idx = tree.query_ball_point(mid, R + diam)
        items = []
        for j in idx:
            p = Ai @ cen[j] + t
            if np.linalg.norm(cen[j] - mid) <= R + 1e-6 or j == a:
                q = Ai @ leaves[j].T[:, :2]
                items.append((lab[j], tuple(np.round(p, 2) + 0.0), tuple(np.round(q.ravel(), 3) + 0.0)))
        return tuple(sorted(items))

    for R in radii:
        margin = R + 2 * diam
        good = [s for s in stars if inside_margin(s[0], margin)]
        ekey = {}; types = {}
        rows = []
        for pos, mem, f in good:
            k = len(mem); row = 0
            for i in range(k):
                a, b = mem[i], mem[(i + 1) % k]
                e = (min(a, b), max(a, b))
                if e not in ekey:
                    mid = pos  # the shared vertex lies on the shared edge; use the edge midpoint instead
                    mid = 0.5 * (cen[a] + cen[b])
                    kk = min(key_in(a, mid, R), key_in(b, mid, R))
                    ekey[e] = types.setdefault(kk, len(types))
                row ^= 1 << ekey[e]
            rows.append((row, f))
        # GF(2) elimination on augmented rows: bit n = rhs
        n = len(types); piv = {}
        incons = None
        for idx_r, (r, f) in enumerate(rows):
            v = r | (f << n); comb = 1 << idx_r
            while v & ((1 << n) - 1):
                hb = (v & ((1 << n) - 1)).bit_length() - 1
                if hb in piv:
                    pv, pc = piv[hb]; v ^= pv; comb ^= pc
                else:
                    piv[hb] = (v, comb); break
            else:
                if v >> n & 1 and incons is None:
                    incons = comb
        solvable = incons is None
        info = dict(R=R, stars_used=len(good), edge_types=n, rank=len(piv), solvable=solvable,
                    flux_minus_used=int(sum(g[2] for g in good)))
        if not solvable:
            used = [i for i in range(len(rows)) if incons >> i & 1]
            info["obstruction_size"] = len(used)
            info["obstruction_valences"] = dict(Counter(len(good[i][1]) for i in used))
        res["runs"][str(R)] = info
        print(json.dumps(info), flush=True)
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
