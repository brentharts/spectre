#!/usr/bin/env python3
r"""chair44facts.py -- the Facts of the Chair44 section, computed once and asked for by name.

The Chair44 section of the paper is built the same way as the rest of it: every number
is computed here, the prose asks for it, and nothing is transcribed.  The Facts come from
`symchair44.py`, which generates the tile from nothing (no JSON), and from the fork's own
boundary code for the Spectre side.  They are appended to `spectrefacts.FACTS` by `register()`,
which is idempotent; the ledger, the supplement and the paper each call it.

What is and is not claimed.  The einstein property and the strong aperiodicity of the Chair44 are
Tsiokos's (arXiv 2609.19214, 2609.23783), and the JSON status string calls the proof "claimed".
Nothing here proves them.  What is re-derived is the combinatorial atlas: the substitution, the
height labels that the gluing forces, the contact atlas and its equations.  Facts about the
Spectre boundary that are new here are the Fibonacci closed form (derived from the recurrence, and
proved in Lean) and the direction split (counted on the tilings).

    python3 chair44facts.py             print the Facts
    python3 chair44facts.py --selftest  recompute and check them (about 15 s)
    python3 chair44facts.py --full      also repeat the 6561-tiling screen (about 3 min)
"""
import itertools
import math
import os
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectrefacts as F                      # noqa: E402
import symchair44 as S                        # noqa: E402

x = sp.symbols('x')


# --------------------------------------------------------------- helpers
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


LABELS = S.label_table(S.CHILDREN)
ROTATIONS = [f for f in S.all_frames() if S.frame_det(f) == 1]
_SOLID = None


def solid():
    global _SOLID
    if _SOLID is None:
        _SOLID = S.Solid(LABELS)
    return _SOLID


def orientation_matrix():
    """24x24 count matrix: M[F][F'] = number of children of an F-tile whose frame is F'."""
    idx = {f: i for i, f in enumerate(ROTATIONS)}
    M = sp.zeros(24, 24)
    for f in ROTATIONS:
        for H, _ in S.CHILDREN:
            M[idx[f], idx[S.frame_mul(f, H)]] += 1
    return M


def chair_symmetries():
    """Symmetries (frame, translation) of the undecorated seven-cube chair."""
    cells = S.pose_cells(S.ROOT)
    return [(f, t) for f in S.all_frames() for t in itertools.product(range(-2, 3), repeat=3)
            if S.pose_cells((f, t)) == cells]


def label_preserving(syms):
    """Which of the chair's symmetries also preserve every pyramid height (sites map to sites)."""
    where = {S.SITES8[r]: r for r in range(192)}
    out = []
    for pose in syms:
        P = S.pose_sites(pose)
        out.append(all(P[r] in where and LABELS[where[P[r]]] == LABELS[r] for r in range(192)))
    return out


def chair_surface_rows(max_n=4):
    from collections import Counter, defaultdict
    import chair44 as C
    rows = []
    for N in range(max_n + 1):
        poses = C.sigma_depth(N)
        fo = defaultdict(list)
        for i, p in enumerate(poses):
            for key, s in C.exposed_faces(p).items():
                fo[key].append((i, s))
        normals = Counter((key[0], v[0][1]) for key, v in fo.items() if len(v) == 1)
        dig = C.tile_digits(N)
        B = [sum(1 for v in fo.values() if len(v) == 2 and dig[v[0][0]][:c] != dig[v[1][0]][:c])
             for c in range(1, N + 1)]
        rows.append(dict(N=N, tiles=len(poses), surface=sum(normals.values()),
                         shared=sum(1 for v in fo.values() if len(v) == 2),
                         flat=sorted(set(normals.values())), directions=len(normals), B=B))
    return rows


# Direction histograms of the boundary edges of an order-k Spectre supertile, folded mod 180
# degrees (index = angle / 30 degrees mod 6; the 12-direction histogram has period 6).  Counted
# on the placed tiling by chair44_spectre_link.boundary_histogram; k = 6 took 66 s, so these are
# recorded and the self-test recomputes k <= 4 (k <= 5 with --full).
DIRECTION_HISTOGRAMS = [
    [2, 1, 1, 1, 1, 1],
    [5, 3, 4, 3, 5, 3],
    [18, 11, 19, 11, 19, 13],
    [77, 49, 78, 49, 79, 47],
    [328, 203, 329, 203, 331, 205],
    [1393, 861, 1394, 863, 1395, 861],
    [5904, 3649, 5903, 3649, 5905, 3649],
]


def class_totals(h):
    """(even-direction total, odd-direction total) over one period of 180 degrees."""
    return sum(h[0::2]), sum(h[1::2])


def z6_modes(h):
    import numpy as np
    return [abs(complex(z)) for z in np.fft.fft(np.array(h, dtype=float))]


# --------------------------------------------------------------- the Facts
KIND = {
    'chair_mesh': 'D', 'chair_volume': 'D', 'chair_labels': 'DX', 'chair_tilings': 'D',
    'chair_screen': 'C', 'chair_atlas': 'DX', 'chair_equations': 'DX',
    'chair_chiral': 'D', 'chair_forced_chirality': 'D', 'chair_orientation': 'D',
    'chair_surface': 'CX', 'chair_exponents': 'D', 'chair_coincidence': 'DX',
    'boundary_fibonacci': 'D', 'boundary_directions': 'C', 'boundary_z3_bounded': 'C',
}
_REGISTERED = False


def register():
    """Append the Chair44 Facts to spectrefacts.FACTS (once)."""
    global _REGISTERED
    if _REGISTERED or any(f.key == 'chair_mesh' for f in F.FACTS):
        _REGISTERED = True
        return F.FACTS
    _REGISTERED = True
    fact = F.fact
    sd = solid()
    V, E, T = len(sd.vertices), len(sd.edges), len(sd.triangles)

    fact(key='chair_mesh',
         claim=r'(V,E,T)=(2138,\,6408,\,4272),\ \chi=2',
         value=(V, E, T, sd.euler(), sd.is_closed_manifold()),
         method='the 24 panels of the seven-cube chair, each cut by its 9x9 grid into 73 flat '
                'rectangles and 8 four-sided pyramids (178 triangles), generated in exact '
                'rational arithmetic by symchair44.py with no external data; closed '
                'and Euler characteristic 2',
         lean='chair_euler')

    wh = sp.symbols('w h', positive=True)
    vol = S.Solid(LABELS, symbolic=True).volume()
    fact(key='chair_volume',
         claim=r'\mathrm{vol}=7+\tfrac{4}{3}w^{2}h\sum_{s}\gamma_{s}=7\ \text{for all }w,h',
         value=(sp.simplify(vol), sum(LABELS)),
         method='the volume of the symbolic solid, for every base width w and height unit h '
                'at once: the signed heights sum to zero, so the pyramids cancel exactly',
         lean='chair_labels_sum')

    classes = sorted(set(abs(v) for v in LABELS))
    fact(key='chair_labels',
         claim=r'192\ \text{sites},\ \gamma_{s}\in\{\pm1,\dots,\pm12\},\ \textstyle\sum\gamma_{s}=0',
         value=(len(LABELS), classes, sum(LABELS)),
         method='height labels forced by exact complementarity on every shared face of the '
                'depth-two patch of the recorded substitution; the signed union-find has 12 '
                'free classes and no contradiction. Matches the published solid vertex for '
                'vertex (symchair44_tests.py)',
         lean='chair_labels_range')

    fact(key='chair_tilings',
         claim=r'3^{8}=6561\ \text{rotation tilings of the doubled chair by 8 chairs}',
         value=len(S.double_chair_tilings()),
         method='exact cover of the 56 cells of the doubled chair by eight rotated copies, by '
                'a bitmask search; mirror images are excluded here and shown unnecessary '
                'below')

    fact(key='chair_screen',
         claim=r'6555\ \text{tilings leave 4 free classes; 6 leave 12, and 3 of those use all 24 orientations}',
         value={'free_classes': {4: 6555, 12: 6}, 'survivors': 6, 'after_orientation_rule': 3,
                'chosen_index_in_sort_order': 1268,
                'legal_contacts_of_the_six': [62, 62, 62, 44, 44, 44],
                'mirror_contacts_legal_in_any': 0},
         method='every one of the 6561 tilings is glued to depth two; six leave 12 height '
                'classes, three of those use the maximum number of orientations, the first '
                'in sort order is the recorded substitution. For all six, no mirror contact '
                'is legal (the 8-orientation family has 62 legal contacts, the other three 44), '
                'so forced chirality does not depend on the selection rule. Recorded from the 3-minute '
                'screen, repeated by `chair44facts.py --full`; the selection rule is a '
                'convention, stated in symchair44.py')

    A = S.atlas()
    legal_det = sorted(set(S.frame_det(p[0]) for p in A['legal']))
    angles = sorted(set(str(S.rotation_data(p[0])[0]) for p in A['realized']))
    fact(key='chair_atlas',
         claim=r'2388\ \text{raw contacts}\to44\ \text{legal}\to30\ \text{realized}',
         value=dict(raw=len(A['raw']), mirror=A['mirror'], legal=len(A['legal']),
                    realized=len(A['realized']), unrealized=len(A['unrealized']),
                    legal_dets=legal_det, angles=angles),
         method='every face contact of two chairs (48 frames, translations in a box), the '
                'legal ones kept by exact site complementarity under the labels, the '
                'realized ones read off the depth-three patch; reproduces the published '
                'atlas (44 poses, 30 states) from its own labels',
         lean='chair_atlas_counts')

    rank = S.equation_rank(A['equations'])
    fact(key='chair_equations',
         claim=r'372\ \text{site equations},\ \operatorname{rank}=180,\ \dim\ker=12',
         value=(len(A['equations']), rank, 192 - rank),
         method='the distinct complement equations c_A + c_B = 0 of the 30 realized '
                'contacts, ranked exactly over Q; the kernel is the 12 free height classes',
         lean='chair_rank_nullity')

    syms = chair_symmetries()
    keep = label_preserving(syms)
    fact(key='chair_chiral',
         claim=r'\mathrm{Sym}(\text{decorated})=\{1\},\ \ \text{while}\ |\mathrm{Sym}(\text{chair})|=6=3+3',
         value=dict(undecorated=len(syms),
                    undecorated_improper=sum(1 for f, _ in syms if S.frame_det(f) == -1),
                    decorated=sum(keep),
                    decorated_improper=sum(1 for (f, _), k in zip(syms, keep) if k and S.frame_det(f) == -1)),
         method='the undecorated seven-cube chair has six symmetries, three of them mirrors, '
                'so it is achiral; a symmetry of the decorated tile must preserve the cell set '
                'and every signed pyramid height, and only the identity does, so the '
                'decoration makes the tile chiral')

    fact(key='chair_forced_chirality',
         claim=r'\text{no mirror contact is legal: all 44 legal contacts have }\det=+1',
         value=dict(legal=len(A['legal']), proper=sum(1 for p in A['legal'] if S.frame_det(p[0]) == 1),
                    mirror_raw=A['mirror'], mirror_legal=sum(1 for p in A['legal'] if S.frame_det(p[0]) == -1)),
         method='of the 2388 raw face contacts 1194 place a mirror image against the root; '
                'none survives exact complementarity. Chirality is forced by the contacts '
                'themselves, so a copy of the tile cannot be replaced by its mirror image '
                'anywhere in the hierarchy (given the labels recorded here)')

    M = orientation_matrix()
    I = sp.eye(24)
    cp = sp.factor(M.charpoly(x).as_expr())
    fact(key='chair_orientation',
         claim=r'\chi(x)=x^{3}(x-8)(x-2)^{20}\ \text{on the 24 rotation species}',
         value=dict(charpoly=cp, rank=M.rank(), rank_square=(M * M).rank(),
                    semisimple=(M * (M - 8 * I) * (M - 2 * I)).is_zero_matrix,
                    row_sums=sorted(set(sum(M.row(i)) for i in range(24))),
                    col_sums=sorted(set(sum(M.col(i)) for i in range(24)))),
         method='the orientation transfer matrix of the substitution on the 24 rotations; '
                'the minimal polynomial divides x(x-8)(x-2), so it is diagonalizable with a '
                'semisimple zero sector (rank M^n constant at 21), every row and column '
                'sums to 8, and orientation frequencies converge to 1/24 at rate (1/4)^d')

    rows = chair_surface_rows(4)
    ok = all(r['surface'] == 24 * 4 ** r['N'] and r['shared'] == 12 * (8 ** r['N'] - 4 ** r['N'])
             and r['flat'] == [4 ** (r['N'] + 1)] and r['directions'] == 6
             and r['B'] == [12 * 4 ** r['N'] * (2 ** c - 1) for c in range(1, r['N'] + 1)]
             for r in rows)
    fact(key='chair_surface',
         claim=r'\text{surface}=24\cdot4^{N},\ \text{shared}=12(8^{N}-4^{N}),\ B_{c}=12\cdot4^{N}(2^{c}-1)',
         value=(ok, [r['surface'] for r in rows], [r['shared'] for r in rows]),
         method='counted on the placed patch for N <= 4 with the face-directions exactly '
                'flat (4^(N+1) per +-x,+-y,+-z); the closed forms follow from 24 faces per '
                'tile, 8 tiles per level, and are proved in Lean; the interface law is the '
                'one the Spectre satisfies, B_c = (T(c)P(N-c) - P(N))/2',
         lean='chair_surface')

    D_b = sp.simplify(sp.log(4) / sp.log(2))
    fact(key='chair_exponents',
         claim=r'D_{b}=2,\ s^{*}=1/3,\ \mu=2\ \ (\lambda=2,\ \nu=4,\ d=3)',
         value=(D_b, 1 - D_b / 3, sp.Integer(2) ** 3 / 4, D_b - 2),
         method='the unified exponents D_b = log nu / log lambda, s* = 1 - D_b/d, '
                'mu = lambda^d / nu with the Chair44 values: its supertile surface is a '
                'genuine surface (zero fractal excess), unlike the Spectre boundary '
                'of dimension 1.399')

    import chair44_twoadic as TA
    tw = TA.report()
    c = tw['counts']
    fact(key='chair_coincidence',
         claim=r'\text{modular coincidence at depth }3,\ c(M)=28\cdot2^{M}-48\ (M\ge3)',
         value=dict(labels=tw['labels'], primitive_exponent=tw['primitive_exponent'],
                    depth=tw['coincidence_depth'], address_0_0_2=tw['address_0_0_2'],
                    coincident=tw['coincident_addresses'], of=tw['words_at_depth'],
                    counts=c[:9], reachable_sets=sum(tw['subset_sizes'].values()),
                    non_singleton=tw['transfer_dim'], spectral_radius=2),
         method='the cell substitution on 24 x 7 = 168 labelled unit cells, rebuilt from the child '
                'poses: column sum 8, primitive at power 3, least modular-coincidence depth 3 '
                'with (0,0,2) among the 336 of 512 coincident addresses, matching the depth and '
                'address recorded upstream (substitution_modular_coincidence). A subset automaton '
                '(209 reachable label sets, 41 non-singleton) counts the words not yet coincided, '
                'c(M) = 28*2^M - 48, so the undetermined set in Z_2^3 has box dimension 1. '
                'Pure-point diffraction and limit-periodicity follow only through the cited '
                'Lee-Moody / Schlottmann theorem, which is not re-proved here',
         lean='coincidence_closed')

    # ---- Spectre side
    seq = F.BOUNDARY_SEQ
    closed = [2 * fib(3 * k + 5) + 4 for k in range(len(seq))]
    fact(key='boundary_fibonacci',
         claim=r'P(k)=2F_{3k+5}+4',
         value=(closed, closed == list(seq)),
         method='the perimeter recurrence has seeds 14, 46, 182 and characteristic '
                'polynomial (x-1)(x^2-4x-1); F(n+6) = 4F(n+3) + F(n) makes 2F(3k+5)+4 satisfy it '
                'with the same seeds, so the closed form holds for every k (proved in Lean). '
                'It says the boundary growth nu = phi^3 is one Spectre level multiplying the '
                'Fibonacci index by three, and that the eigenvalue-1 mode is the constant 4',
         lean='boundary_fibonacci')

    tot = [class_totals(h) for h in DIRECTION_HISTOGRAMS]
    fact(key='boundary_directions',
         claim=r'\text{even}=F_{3k+4}+1,\ \text{odd}=F_{3k+3}+1\ \ (k\le6)',
         value=[(k, e, o, e == fib(3 * k + 4) + 1, o == fib(3 * k + 3) + 1)
                for k, (e, o) in enumerate(tot)],
         method='boundary edges of an order-k supertile, sorted by direction mod 180 degrees '
                '(the 14 edges point in multiples of 30 degrees, so there are 6 classes), '
                'counted on the placed tiling for k <= 6; the totals are the Fibonacci '
                'numbers shown and the Z/2 character has amplitude exactly F(3k+2). '
                'Counted, not derived: the split is not proved for k > 6',
         lean='direction_split')

    peak = max(m for h in DIRECTION_HISTOGRAMS for j, m in enumerate(z6_modes(h)) if j not in (0, 3))
    fact(key='boundary_z3_bounded',
         claim=r'\max|\hat h(j)|\ \text{over the Z/3 and faithful modes}\le\sqrt{13}\ \ (k\le6)',
         value=(round(peak, 4), round(math.sqrt(13), 4)),
         method='the DFT of the folded direction histogram over Z/6 = Z/2 x Z/3 (the split '
                'spectre_torsion uses for the six Mystic classes): the Z/2 character carries '
                'the golden growth and the other modes stay bounded; observed through '
                'k = 6, not proved, and it identifies the same group, not the same action')
    return F.FACTS


# --------------------------------------------------------------- self-test
def selftest(full=False):
    register()
    get = F.get
    n = 0

    def ok(cond, what):
        nonlocal n
        assert cond, what
        n += 1

    ok(get('chair_mesh').value == (2138, 6408, 4272, 2, True), 'mesh')
    vol, lsum = get('chair_volume').value
    ok(vol == 7 and lsum == 0, 'volume 7 for all w, h')
    ok(get('chair_labels').value == (192, list(range(1, 13)), 0), 'labels')
    ok(get('chair_tilings').value == 6561, 'tilings')
    A = get('chair_atlas').value
    ok((A['raw'], A['mirror'], A['legal'], A['realized'], A['unrealized']) == (2388, 1194, 44, 30, 14), 'atlas')
    ok(A['legal_dets'] == [1] and A['angles'] == ['0', '2*pi/3', 'pi', 'pi/2'], 'atlas rotations')
    ok(get('chair_equations').value == (372, 180, 12), 'equations')
    c = get('chair_chiral').value
    ok(c == dict(undecorated=6, undecorated_improper=3, decorated=1, decorated_improper=0), 'chiral')
    f = get('chair_forced_chirality').value
    ok(f['proper'] == f['legal'] == 44 and f['mirror_legal'] == 0 and f['mirror_raw'] == 1194, 'forced')
    o = get('chair_orientation').value
    ok(o['charpoly'] == x ** 3 * (x - 8) * (x - 2) ** 20 and o['rank'] == o['rank_square'] == 21
       and o['semisimple'] and o['row_sums'] == o['col_sums'] == [8], 'orientation')
    ok(get('chair_surface').value[0], 'surface laws')
    w = get('chair_coincidence').value
    ok(w['primitive_exponent'] == 3 and w['depth'] == 3 and w['address_0_0_2'] and w['coincident'] == 336
       and w['counts'] == [1, 8, 64, 176, 400, 848, 1744, 3536, 7120], 'coincidence')
    d = get('chair_exponents').value
    ok(d == (2, sp.Rational(1, 3), 2, 0), 'exponents')
    ok(get('boundary_fibonacci').value[1] and len(F.BOUNDARY_SEQ) == 7, 'fibonacci form')
    ok(all(r[3] and r[4] for r in get('boundary_directions').value), 'direction split')
    ok(get('boundary_z3_bounded').value[0] <= get('boundary_z3_bounded').value[1] + 1e-9, 'z3 bounded')
    for k in range(len(DIRECTION_HISTOGRAMS)):
        ok(2 * sum(class_totals(DIRECTION_HISTOGRAMS[k])) == F.BOUNDARY_SEQ[k], 'histogram total %d' % k)
    # the recorded histograms: recompute the cheap ones from the tilings
    import chair44_spectre_link as L
    for k in range(6 if full else 5):
        ok(L.boundary_histogram(k)[:6].tolist() == DIRECTION_HISTOGRAMS[k], 'histogram %d recomputed' % k)
    # the Z/2 amplitude is exactly F(3k+2)
    for k, h in enumerate(DIRECTION_HISTOGRAMS):
        ok(abs(z6_modes(h)[3] - fib(3 * k + 2)) < 1e-6, 'Z/2 amplitude %d' % k)
    # Fact bookkeeping
    ok(all(key in KIND for key in KIND) and all(f.key in KIND for f in F.FACTS if f.key.startswith('chair_')),
       'every chair fact is classified')
    if full:
        chosen, summary = S.screen_tilings(verbose=False)
        ok(summary['classes_histogram'] == {4: 6555, 12: 6} and chosen == S.CHILDREN, 'screen')
        surv = [tuple(sorted(t, key=S.child_order)) for t in S.double_chair_tilings()
                if not S.gluing(list(t), 2)[1] and len(S.gluing(list(t), 2)[0]) == 12]
        legal = [S.legal_contacts(S.label_table(c)) for c in surv]
        ok(sorted(len(l) for l in legal) == [44, 44, 44, 62, 62, 62]
           and all(S.frame_det(p[0]) == 1 for l in legal for p in l), 'forced chirality for all six')
    return n


def main():
    if '--selftest' in sys.argv or '--full' in sys.argv:
        print('chair44facts: %d checks passed' % selftest(full='--full' in sys.argv))
        return
    register()
    for f in F.FACTS:
        if f.key in KIND:
            print('%-24s %s\n    %s' % (f.key, f.claim, f.value))


if __name__ == '__main__':
    main()
