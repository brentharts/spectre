#!/usr/bin/env python3
r"""monotilefacts.py -- the Facts of the Spectre/Chair44 comparison, computed once and asked for by name.

These are the exact results of the refocused paper: the pure-point diffraction of the Chair44 centre set, the
orientation-sum recursion of the Spectre and the orientation channels of both tilings, the face-contact matrices,
and the four-dimensional lift of the Spectre with its integer inflation matrix and bounded internal plane.  They come
from the modules chair44_diffraction, spectre_orient, orientation_spectra, contact_operators, lift4d and lift4d_matrix
(each of which carries its own tests), and are appended to `spectrefacts.FACTS` by `register()`, which is idempotent.

What is and is not claimed.  Pure-pointness of the Spectre diffraction is a published theorem (Baake, Gahler, Mazac,
Sadun) and is not re-proved; what is new here is the exact analysis of the Chair44 centre set, the exact orientation
recursion, the integer inflation matrix in the lift, and a numerically identified (not proved) closed form for the
orientation constant.  The box count of the window image recorded here is a numerical estimate; the exact value
log(5+2 sqrt6)/log(4+sqrt15) for the seams of the window, a lower bound for its boundary, is Theorem thm:window of the
paper (boundary_certificate/spectre_window.py).

    python3 monotilefacts.py             print the Facts
    python3 monotilefacts.py --selftest  recompute and check them
"""
import math
import os
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectrefacts as F                      # noqa: E402

KIND = {
    'centre_diffraction': 'DX', 'orientation_recursion': 'DX', 'orientation_channels': 'D',
    'contact_matrices': 'C', 'lift_integer': 'DX', 'inflation_matrix': 'DX', 'internal_plane': 'DC',
}
_REGISTERED = False
X = sp.symbols('x')


def register():
    """Append these Facts to spectrefacts.FACTS (once)."""
    global _REGISTERED
    if _REGISTERED or any(f.key == 'centre_diffraction' for f in F.FACTS):
        _REGISTERED = True
        return F.FACTS
    _REGISTERED = True
    fact = F.fact

    import chair44_diffraction as D
    import chair44_slice as CS
    rng = np.random.default_rng(3)
    err = 0.0
    for n in (1, 2, 3):
        for k in rng.uniform(0, 2 * np.pi, (6, 3)):
            err = max(err, abs(D.structure_factor(n, k) - D.direct_structure_factor(n, k)) / 8 ** n)
    fact(key='centre_diffraction',
         claim=r'X_n=2X_{n-1}\cup(\text{odd box}\setminus\text{corner octant});\ \ \textstyle\sum_{k}I(k)=\tfrac17=\text{density}',
         value=dict(recursion_ok=all(D.centres_recursive(n) == {tuple(int(v) for v in p) for p in CS.centres(n)}
                                     for n in range(0, 5)),
                    closed_vs_direct=err,
                    level_masses=[D.level_mass(J) for J in range(0, 5)],
                    total=D.total_mass() / 49, density=Fraction(1, 7),
                    amp_axis=D.bragg_amplitude((1, 0, 0), 1), amp_diag=D.bragg_amplitude((1, 1, 1), 1),
                    tail=7 - D.total_mass(40)),
         method='the centre of every chair of the depth-n patch obeys the nested recursion above (checked '
                'against the placed patch, n<=4); the structure factor is a sum of products of geometric '
                'series (closed form) and equals the direct sum over the centres to rounding; its limit at the '
                'dyadic wave vectors k=2 pi m/2^J is the exact rational amplitude '
                '(7/8)(8^(1-J))(+-1+1/7), the levels J carry mass 21/4 8^(1-J), and the total mass divided by '
                '49 is 1/7, the density of the centres, so the Parseval sum is complete and the measure is '
                'pure point. The notch-weighted diffraction is only consistent with this numerically')

    import spectre_orient as O
    pred = O.predicted_sums(5)
    placed = O.placed_sums(5)
    fact(key='orientation_recursion',
         claim=r'S_{n+1}=C\,\overline{S_n},\ \ |S_n|\sim c\,\lambda^{n},\ \ c^{2}=\tfrac{15+5\sqrt3+3\sqrt5+3\sqrt{15}}{20}',
         value=dict(levels=5, max_rel=float(max(abs(a - b) / abs(b) for a, b in zip(pred, placed))),
                    c2=O.exact_constant_squared(), c2_recursion=O.asymptotic_constant(60) ** 2,
                    two_level=sorted(abs(e) for e in O.eigenvalues_two_level())[-1], lam2=O.LAMBDA2),
         method='the sum of exp(i angle) over the tiles of the depth-n supertile of each label obeys a linear '
                'recursion whose 9x9 matrix C (slot rotations of the construction) is applied to the '
                'complex conjugate, because the inflation is orientation reversing; two levels give C conj(C) '
                'with leading eigenvalue lambda^2 = 4+sqrt15. The recursion is compared with the sums over the '
                'placed tiles of spectre_patch up to level 5. The closed form of c^2 is a root of '
                '400x^4-1200x^3+840x^2-180x+9, identified from the recursion at 80 digits; it is identified, not '
                'proved')

    import orientation_spectra as OS
    h = OS.exponents()
    fact(key='orientation_channels',
         claim=r'\mathrm{charpoly}=x^{3}(x-8)(x-2)^{20}\ \ (\text{chair});\ \ h_{m}=2,1,0\ (\text{Spectre})',
         value=dict(chair_multiplicities=OS.chair_charpoly_multiplicities(),
                    chair_exponents={k: round(v, 6) for k, v in h.items() if k.startswith('chair')},
                    spectre_exponents={k: round(v, 6) for k, v in h.items() if k.startswith('spectre')},
                    spectre_leading=[round(OS.spectre_channel_leading(m), 6) for m in range(0, 7)]),
         method='the 24 orientation states of the Chair44 are split by the irreducible representations of the '
                'rotation group of the cube; each non-trivial channel of the child-frame matrix is exactly 2 '
                'times a projector, and the channels assemble to the characteristic polynomial of the 24x24 '
                'matrix. The Spectre orientation recursion is split by the harmonic m of the orientation '
                'angle: the leading eigenvalues give the exponents h=ln(mu)/ln(lambda_lin)')

    import contact_operators as CO
    Ks, ns = CO.spectre_contact_matrix(4)
    Kc, nc = CO.chair_contact_matrix(3)
    ss, sc = CO.summary(Ks), CO.summary(Kc)
    fact(key='contact_matrices',
         claim=r'\text{Spectre: }14\times14,\text{ rank }14,\text{ bipartite};\ \ \text{Chair44: }24\times24,\text{ rank }24,\text{ not bipartite}',
         value=dict(spectre=dict(contacts=ns, support=ss['support'], rank=ss['rank'], bipartite=ss['bipartite'],
                                 components=ss['components'], symmetric_spectrum=bool(np.allclose(
                                     np.sort(np.linalg.eigvalsh(Ks.astype(float))),
                                     -np.sort(np.linalg.eigvalsh(Ks.astype(float)))[::-1], atol=1e-6))),
                    chair=dict(contacts=nc, support=sc['support'], rank=sc['rank'], bipartite=sc['bipartite'],
                               components=sc['components']),
                    chair_law=nc == 12 * (8 ** 3 - 4 ** 3)),
         method='the number of times native edge a of one tile meets native edge b of a neighbour, counted '
                'over the level-4 Spectre patch (29,208 contacts) and the native unit face of one chair '
                'meeting the native face of a neighbour over the depth-3 chair patch (5,376 contacts, equal '
                'to 12(8^3-4^3)); support, rank, bipartiteness and components of the symmetrised matrix')

    import lift4d as L
    tiles, lifted, checks = L.lift_patch(3)
    V = np.array(list(lifted.values()), float)
    fact(key='lift_integer',
         claim=r'\text{every vertex has integer coordinates }(p,q,r,s)\in\mathbb{Z}^{4}=\mathbb{Z}[\zeta_{12}];\ \ \mathrm{rank}=4',
         value=dict(level=3, vertices=len(V), consistency_checks=checks, rank=L.lattice_rank(V),
                    area_homogeneous=abs(L.tile_area(L.project(np.array(tiles[0]), 2.0, 2.0))
                                         - 4 * L.tile_area(L.project(np.array(tiles[0]), 1.0, 1.0))) < 1e-9),
         method='every edge of the patch is a unit vector at a multiple of 30 degrees: even multiples are '
                'p+qw with w=exp(i pi/3) and odd ones z(r+sw) with z=exp(i pi/6); walking over the edges gives '
                'every vertex integer coordinates, and every vertex reached twice agrees (the lift is '
                'consistent); the Cheritat-Ma deformation a(p+qw)+b z(r+sw) is the linear projection; the '
                'level-3 patch has 3,734 vertices')

    import lift4d_matrix as LM
    P = LM.phi_matrix()
    cp = sp.factor(P.charpoly(X).as_expr())
    M4, _ = LM.block_matrix()
    q = LM.reference_quad()
    Q = sp.Matrix(np.concatenate(q).tolist())
    phys_err = 0.0
    Qn = Q
    for n in (1, 2, 3):
        Qn = P * Qn
        phys = np.array([L.project(LM.to_pqrs([float(c) for c in Qn[4 * i:4 * i + 4]])) for i in range(4)])
        num = LM.numeric_quad(n)
        phys_err = max(phys_err, float(np.abs(phys - (num[:, 0] + 1j * num[:, 1])).max()))
    fact(key='inflation_matrix',
         claim=r'\mathrm{charpoly}=(x-1)^{4}(x+1)^{4}(x^{4}-8x^{2}+1)^{2},\ \ \lambda^{2}=4+\sqrt{15}',
         value=dict(charpoly=cp, expected=sp.expand(cp - (X - 1) ** 4 * (X + 1) ** 4 * (X ** 4 - 8 * X ** 2 + 1) ** 2) == 0,
                    block_charpoly=M4.charpoly(X).as_expr(), det=M4.det(), trace=M4.trace(),
                    block=M4.tolist(), float_error=phys_err,
                    roots=[float(r) for r in sp.Poly(X ** 4 - 8 * X ** 2 + 1, X).nroots()],
                    unit_norm=sp.simplify((4 + sp.sqrt(15)) * (4 - sp.sqrt(15)))),
         method='the supertile recursion of spectre_patch moves four key points by rotations through multiples '
                'of 60 degrees, a mirror and translations built from the points themselves; on Z[zeta_12] these '
                'are integer maps (multiplication by zeta^k, z -> -conj z), so one level is a 16x16 integer '
                'matrix, which reproduces the floating-point recursion (three levels) and whose characteristic '
                'polynomial is above; the key-point vector has minimal polynomial (x^2-1)(x^4-8x^2+1) and the '
                'matrix on that block is the companion matrix of x^4-8x^2+1, of determinant 1 and trace 0')

    G = LM.exact_bounded_plane()
    Gn = np.array([[float(sp.N(c)) for c in row] for row in G.tolist()])
    ext = {}
    for lev in (3, 4):
        Vl = np.array(list(L.lift_patch(lev)[1].values()), float)
        ext[lev] = (float(np.abs(Vl @ Gn.T).max()), float(np.abs(L.project(Vl)).max()))
    rows, ex = LM.window_boundary_exponents(5, (0.2, 0.1, 0.05))
    zj = np.abs(L.internal(np.array(list(L.lift_patch(3)[1].values()), float), 5)).max()
    fact(key='internal_plane',
         claim=r'\text{bounded plane over }\mathbb{Q}(\sqrt{15});\ \ \text{window box count}\approx1.2\ \text{(exact seam dimension }1.111\text{, by the window theorem)}',
         value=dict(plane=[[sp.nsimplify(c) for c in row] for row in G.tolist()],
                    fitted_cosines=[float(c) for c in LM.subspace_cosines(Gn, L.bounded_plane(4)[0])],
                    internal_extent={k: v[0] for k, v in ext.items()},
                    physical_extent={k: v[1] for k, v in ext.items()},
                    growth=ext[4][1] / ext[3][1], galois_extent=float(zj),
                    boundary_cells=[(r[0], r[3]) for r in rows], boundary_exponents=[round(e, 3) for e in ex],
                    area_exponent=round(math.log2(rows[2][1] / rows[1][1]), 3)),
         method='a linear functional is bounded on the supertile corners exactly when it kills the expanding '
                'part (Phi^2=4+sqrt15) of every iterate of the key-point vector; each level mirrors the frame, '
                'so the odd iterates are mirrored first, and then the expanding parts span 2 dimensions and the '
                'annihilator is a 2x4 matrix over Q(sqrt15), which agrees with a plane fitted to the lifted '
                'vertices; its extent on the level-3 and level-4 patches is stable while the physical extent grows '
                'by the inflation factor; the Galois conjugate z -> z^5 is not bounded; the boundary-cell count of '
                'the window image (level-5 patch, 214,000 vertices, holes filled) grows by 2^1.2 per halving of '
                'the cell size, a box-counting estimate and not a dimension; the exact seam dimension '
                'log(5+2sqrt6)/log(4+sqrt15) = 1.110977, a lower bound for the boundary, is Theorem thm:window')
    return F.FACTS


def selftest():
    register()
    n = 0

    def ok(cond, label):
        nonlocal n
        if not cond:
            raise AssertionError(label)
        n += 1

    g = lambda k: F.get(k).value
    d = g('centre_diffraction')
    ok(d['recursion_ok'] and d['closed_vs_direct'] < 1e-8 and d['total'] == d['density'] == Fraction(1, 7)
       and d['level_masses'][1] == Fraction(21, 4) and d['tail'] < Fraction(1, 10 ** 30), 'centre diffraction')
    o = g('orientation_recursion')
    ok(o['max_rel'] < 1e-9 and abs(o['c2'] - o['c2_recursion']) < 1e-9 and abs(o['two_level'] - o['lam2']) < 1e-9,
       'orientation recursion')
    c = g('orientation_channels')
    ok(c['chair_multiplicities'] == {8.0: 1, 2.0: 20, 0.0: 3}
       and [round(v) for v in (c['spectre_exponents']['spectre m=0'], c['spectre_exponents']['spectre m=1'])] == [2, 1],
       'orientation channels')
    k = g('contact_matrices')
    ok(k['spectre']['rank'] == 14 and k['spectre']['bipartite'] and k['chair']['rank'] == 24
       and not k['chair']['bipartite'] and k['chair']['components'] == 2 and k['chair_law'], 'contacts')
    li = g('lift_integer')
    ok(li['rank'] == 4 and li['consistency_checks'] > 1000 and li['area_homogeneous'], 'integer lift')
    m = g('inflation_matrix')
    ok(m['expected'] and m['det'] == 1 and m['trace'] == 0 and m['float_error'] < 1e-9
       and m['block'] == [[0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 8], [0, 0, 1, 0]], 'inflation matrix')
    p = g('internal_plane')
    ok(min(p['fitted_cosines']) > 1 - 1e-6 and p['internal_extent'][4] < 1.3 * p['internal_extent'][3]
       and abs(p['growth'] - float(F.LAM)) < 0.1 and p['galois_extent'] > 40
       and all(1.1 < e < 1.4 for e in p['boundary_exponents']), 'internal plane')
    ok(all(key in KIND for key in KIND) and set(KIND) == {f.key for f in F.FACTS if f.key in KIND}, 'classified')
    return n


def main():
    if '--selftest' in sys.argv:
        print('monotilefacts: %d checks passed' % selftest())
        return
    register()
    for f in F.FACTS:
        if f.key in KIND:
            print('%-24s %s\n    %s' % (f.key, f.claim, f.value))


if __name__ == '__main__':
    main()
