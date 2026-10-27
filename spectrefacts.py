#!/usr/bin/env python3
r"""spectrefacts.py -- every Fact of the Spectre substrate paper, computed.

The paper this feeds used to carry its numbers as prose.  That is the
arrangement in which a census convention drifts between two sections and
nobody notices, which is exactly what had happened: the charge values of
Proposition 3 were quoted next to patch totals that include the Mystic, while
the charges are only those values when the Mystic is left out.  Both numbers
were right and the pair of them was not reproducible.

So every number in the paper is computed here, once, and the paper asks for it
by name.  Nothing is transcribed.  A Fact carries four things -- what it
claims, the value, how it was obtained, and, where the claim is closed enough
to be checked by a kernel rather than by this file, a Lean rendering.

    python3 spectrefacts.py              print the facts
    python3 spectrefacts.py --selftest   check every one of them
    python3 spectrefacts.py --census     the two censuses, side by side

The arithmetic is exact throughout: sympy over Q(sqrt15) for the spectral
theory, integers for the census and the charges.  No float appears in any
assertion, and the one place a float is printed it is marked as a decimal
expansion of something already known exactly.
"""

import sympy as sp

# --------------------------------------------------------------- the matrix
#
# The Spectre substitution on nine species, in the order the paper uses.  This
# is the one quoted input; everything below is derived from it.
#
# The Mystic is the thing to be careful about.  Geometrically the Gamma
# supertile carries a pair, Gamma_1 and Gamma_2 = Tile(b,a), and a placed patch
# contains both -- which is why the geometric census runs 1, 9, 71, 559, 4401.
# The matrix has one Gamma row, so iterating it counts Gamma_1 alone and runs
# 1, 8, 63, 496, 3905.  The two differ by exactly the Mystic count, and the
# left-eigenvector charges are defined against the nine-species vector, so they
# eat the second census and not the first.  CENSUS_NOTE below says so in one
# sentence; the paper prints it.

SPECIES = ('Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma',
           'Phi', 'Psi')

TEX = {'Gamma': r'\Gamma', 'Delta': r'\Delta', 'Theta': r'\Theta',
       'Lambda': r'\Lambda', 'Xi': r'\Xi', 'Pi': r'\Pi', 'Sigma': r'\Sigma',
       'Phi': r'\Phi', 'Psi': r'\Psi'}

M = sp.Matrix([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 2, 0, 1, 0, 1, 2, 0, 0],
    [1, 1, 2, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 1, 2, 2],
    [0, 0, 1, 1, 2, 2, 0, 2, 3],
])

INDEX = {name: i for i, name in enumerate(SPECIES)}

CENSUS_NOTE = (
    'The Gamma supertile carries the Mystic pair, so a placed patch holds '
    'both Gamma_1 and Gamma_2 while the substitution matrix has a single '
    'Gamma row. The geometric census therefore exceeds the matrix census by '
    'the Mystic count at every depth. The charges below are linear forms on '
    'the nine-species vector and are evaluated against the matrix census.')


# ------------------------------------------------------------------- facts

class Fact(object):
    """One claim, its value, and how it was got.

    `lean` is the name of the theorem in the generated Lean file when the
    claim is closed enough for a kernel to settle it, and None when it is not.
    Keeping that distinction on the Fact itself is what stops the paper
    claiming a machine check for something no machine checked.
    """

    def __init__(self, key, claim, value, method, lean=None, decimal=None):
        self.key = key
        self.claim = claim
        self.value = value
        self.method = method
        self.lean = lean
        self.decimal = decimal

    def __repr__(self):
        return 'Fact(%s)' % self.key


FACTS = []


def fact(**kw):
    FACTS.append(Fact(**kw))
    return FACTS[-1]


def get(key):
    for f in FACTS:
        if f.key == key:
            return f
    raise KeyError(key)


# --------------------------------------------------- the inflation factor

x = sp.Symbol('x')
SQ15 = sp.sqrt(15)
LAM2 = 4 + SQ15                       # the Perron eigenvalue
LAM = sp.sqrt(LAM2)                   # the linear inflation factor
G_SMALL = 4 - SQ15                    # the conjugate unit, = 1/lambda^2

CHARPOLY = sp.factor(M.charpoly(x).as_expr())

fact(key='charpoly',
     claim=r'\chi_M(x)=x^{5}(x-1)(x+1)(x^{2}-8x+1)',
     value=CHARPOLY,
     method='exact characteristic polynomial over Z, factored')

fact(key='perron',
     claim=r'\lambda^{2}=4+\sqrt{15}',
     value=LAM2,
     method='largest root of the quadratic factor',
     decimal=sp.N(LAM2, 12))

fact(key='pell',
     claim=r'4^{2}-15\cdot 1^{2}=1',
     value=4 ** 2 - 15 * 1 ** 2,
     method='the Pell equation, so lambda^2 is the fundamental unit of '
            'Z[sqrt15]',
     lean='pell_fundamental_unit')

fact(key='lam_closed',
     claim=r'\lambda=\tfrac12(\sqrt6+\sqrt{10})',
     value=sp.sqrt(6) / 2 + sp.sqrt(10) / 2,
     method='square both sides: (sqrt6+sqrt10)^2/4 = (16+4 sqrt15)/4',
     decimal=sp.N(LAM, 12))

fact(key='minpoly',
     claim=r'x^{4}-8x^{2}+1',
     value=sp.minimal_polynomial(LAM, x),
     method='minimal polynomial of lambda over Q')

fact(key='channel_hex',
     claim=r'(\lambda-\lambda^{-1})^{2}=6',
     value=sp.expand(sp.radsimp((LAM - 1 / LAM) ** 2)),
     method='lambda^2 + lambda^-2 - 2, in Z[sqrt15], no root evaluated',
     lean='channel_hexagonal')

fact(key='channel_pent',
     claim=r'(\lambda+\lambda^{-1})^{2}=10',
     value=sp.expand(sp.radsimp((LAM + 1 / LAM) ** 2)),
     method='lambda^2 + lambda^-2 + 2, in Z[sqrt15], no root evaluated',
     lean='channel_pentagonal')

fact(key='contraction',
     claim=r'\lambda^{-4}=31-8\sqrt{15}',
     value=sp.radsimp(LAM ** -4),
     method='the square of the conjugate unit',
     lean='contraction_rate',
     decimal=sp.N(LAM ** -4, 12))


# ------------------------------------------------------- the zero sector

RANKS = {n: (M ** n).rank() for n in (1, 2, 3, 4)}

fact(key='jordan',
     claim=r'\operatorname{rank}M=5,\ \operatorname{rank}M^{n}=4\ (n\ge2)',
     value=RANKS,
     method='exact elimination over Q; zero has algebraic multiplicity five '
            'and geometric multiplicity four, so M is not diagonalizable')


# ------------------------------------------------------------- the charges

def charge(**weights):
    return sp.Matrix([[weights.get(n, 0) for n in SPECIES]])


Q_PLUS = charge(Gamma=-3, Delta=-1, Xi=1, Pi=1, Sigma=-2, Phi=1, Psi=2)
Q_MINUS = charge(Gamma=-1, Delta=-1, Theta=2, Xi=1, Pi=-1, Phi=1)

fact(key='q_plus_eigen',
     claim=r'Q_{+}M=Q_{+}',
     value=list(Q_PLUS * M) == list(Q_PLUS),
     method='left eigenvector for the marginal eigenvalue +1',
     lean='q_plus_is_invariant')

fact(key='q_minus_eigen',
     claim=r'Q_{-}M=-Q_{-}',
     value=list(Q_MINUS * M) == list(-Q_MINUS),
     method='left eigenvector for the marginal eigenvalue -1',
     lean='q_minus_alternates')


# --------------------------------------------------------- the two censuses

def matrix_census(depth, seed='Delta'):
    """Species counts after `depth` substitutions, Gamma_1 only."""
    v = sp.Matrix([[1 if n == seed else 0] for n in SPECIES])
    for _ in range(depth):
        v = M * v
    return {n: int(v[INDEX[n]]) for n in SPECIES}


def geometric_census(depth, seed='Delta'):
    """The same patch as placed tiles, with the Mystic counted separately."""
    counts = matrix_census(depth, seed)
    mystic = counts['Gamma']
    out = dict(counts)
    out['Gamma_2'] = mystic
    return out


def census_total(counts):
    return sum(counts.values())


MATRIX_TOTALS = [census_total(matrix_census(d)) for d in range(6)]
GEOMETRIC_TOTALS = [census_total(geometric_census(d)) for d in range(6)]


def evaluate(q, counts):
    v = sp.Matrix([[counts[n]] for n in SPECIES])
    return int((q * v)[0])


Q_PLUS_BY_DEPTH = [evaluate(Q_PLUS, matrix_census(d)) for d in range(1, 5)]
Q_MINUS_BY_DEPTH = [evaluate(Q_MINUS, matrix_census(d)) for d in range(1, 5)]

fact(key='q_values',
     claim=r'Q_{+}\equiv-1,\quad Q_{-}=+1,-1,+1,-1',
     value=(Q_PLUS_BY_DEPTH, Q_MINUS_BY_DEPTH),
     method='evaluated on the matrix census at depths one to four',
     lean='charges_at_depth_1')

fact(key='censuses',
     claim=r'T=1,8,63,496;\ T_{\rm geom}=1,9,71,559',
     value=(MATRIX_TOTALS, GEOMETRIC_TOTALS),
     method='the same patches counted with and without the Mystic')

# The two censuses are not merely different, they are related, and the
# relation is a one-line consequence of the matrix: row Gamma is all ones, so
# N_Gamma(n) is the total at n-1, and the Mystic count is a copy of it.  That
# turns the ambiguity the prose used to carry into an identity, which is a
# better fix than a warning.

MYSTIC_RULE = all(GEOMETRIC_TOTALS[n] == MATRIX_TOTALS[n] + MATRIX_TOTALS[n - 1]
                  for n in range(1, len(MATRIX_TOTALS)))

fact(key='mystic_rule',
     claim=r'T_{\rm geom}(n)=T(n)+T(n-1)',
     value=MYSTIC_RULE,
     method='row Gamma of M is all ones, so N_Gamma(n) is the whole census at '
            'n-1 and the Mystic is a second copy of it; the geometric total '
            'is therefore the matrix total plus the previous one',
     lean='mystic_1')


# ------------------------------------------------------ the ledger identity

LEDGER = all(matrix_census(n)['Theta'] == matrix_census(n - 1)['Gamma']
             for n in range(1, 7))

fact(key='ledger',
     claim=r'N_\Theta(n)=N_\Gamma(n-1)',
     value=LEDGER,
     method='row Theta of M is the indicator of Gamma, so the count is the '
            'previous generation of Gamma verbatim')

fact(key='ledger_lambda',
     claim=r'N_\Lambda(n)=N_\Sigma(n-1)',
     value=all(matrix_census(n)['Lambda'] == matrix_census(n - 1)['Sigma']
               for n in range(1, 7)),
     method='row Lambda of M is the indicator of Sigma')


# ------------------------------------------- the two monotile phases

# The other phase of Conjecture 1, for contrast.  The Hat is generated by a
# substitution on four metatiles, not nine, and its arithmetic is a different
# quadratic field -- which makes the phase transition of the conjecture a
# change of field and not only a change of parity.

HAT_M = sp.Matrix([[3, 1, 2, 2],
                   [1, 0, 0, 0],
                   [3, 0, 1, 1],
                   [3, 0, 2, 3]])

HAT_CHARPOLY = sp.factor(HAT_M.charpoly(x).as_expr())
PHI = (1 + sp.sqrt(5)) / 2
HAT_LAM2 = sp.nsimplify(sp.Rational(7, 2) + sp.Rational(3, 2) * sp.sqrt(5))

fact(key='hat_charpoly',
     claim=r'\chi_H(x)=(x-1)(x+1)(x^{2}-7x+1)',
     value=HAT_CHARPOLY,
     method='characteristic polynomial of the four-metatile Hat substitution')

fact(key='hat_perron',
     claim=r'\kappa=\tfrac12(7+3\sqrt5)=\varphi^{4}',
     value=HAT_LAM2,
     method='largest root of the Hat quadratic, and the fourth power of the '
            'golden ratio',
     decimal=sp.N(HAT_LAM2, 12))

fact(key='phase_fields',
     claim=r'\mathbb{Q}(\sqrt{15})\ \text{against}\ \mathbb{Q}(\sqrt5)',
     value=(sp.sqrt(15), sp.sqrt(5)),
     method='the two characteristic polynomials differ in a single '
            'coefficient, seven against eight, and that coefficient is the '
            'discriminant: the Spectre to Hat transition changes the '
            'quadratic field the tiling is graded by')


# ------------------------------------------------------- the frequencies

def perron_vector():
    """The normalised Perron eigenvector, exactly, in Q(sqrt15)."""
    v = (M - LAM2 * sp.eye(9)).nullspace()[0]
    v = v / sum(v)
    return [sp.radsimp(sp.simplify(c)) for c in v]


FREQ = dict(zip(SPECIES, perron_vector()))

# Every Spectre frequency lies in Z[sqrt15] itself, not merely in Q(sqrt15):
# the denominators clear.  Recording them as integer pairs is what lets the
# kernel check the eigenvector equation in Section "the eigenvector as an
# eigenvector" without any rational arithmetic at all.

def as_z15(c):
    """Write a frequency as the pair (a, b) meaning a + b sqrt15."""
    p = sp.Poly(sp.radsimp(sp.expand(c)), SQ15)
    coeffs = p.all_coeffs()
    if len(coeffs) == 1:
        return (sp.Rational(coeffs[0]), sp.Integer(0))
    return (sp.Rational(coeffs[1]), sp.Rational(coeffs[0]))


FREQ_Z15 = {n: as_z15(FREQ[n]) for n in SPECIES}

fact(key='frequencies',
     claim=r'v\in\mathbb{Z}[\sqrt{15}]^{9},\ \textstyle\sum v=1',
     value=FREQ,
     method='nullspace of M - lambda^2 I, normalised; the denominators clear, '
            'so the frequencies are algebraic integers and the eigenvector '
            'equation can be checked without leaving Z[sqrt15]',
     lean='frequencies_are_an_eigenvector')

# Aperiodicity, as irrationality -- and the two phases differ in how robustly.
# A periodic tiling has a fundamental domain, in which every species occurs a
# whole number of times, so every relative frequency is rational.  One
# irrational frequency therefore refutes periodicity, and it matters which.

SPECTRE_IRRATIONAL = [n for n in SPECIES if not sp.simplify(FREQ[n]).is_rational]

HAT_SPECIES = ('H', 'T', 'P', 'F')
_hv = (HAT_M - HAT_LAM2 * sp.eye(4)).nullspace()[0]
_hv = _hv / sum(_hv)
HAT_FREQ = {n: sp.radsimp(sp.simplify(c)) for n, c in zip(HAT_SPECIES, _hv)}
HAT_RATIONAL = [n for n in HAT_SPECIES if HAT_FREQ[n].is_rational]

# Six clears every denominator, so the Hat frequencies are also integer pairs
# once scaled -- which is what the Lean rendering needs.
def _as_z5_times_six(c):
    p = sp.Poly(sp.radsimp(sp.expand(6 * c)), sp.sqrt(5))
    co = p.all_coeffs()
    if len(co) == 1:
        return (int(co[0]), 0)
    return (int(co[1]), int(co[0]))


HAT_FREQ_Z5 = [_as_z5_times_six(HAT_FREQ[n]) for n in HAT_SPECIES]

fact(key='hat_frequencies',
     claim=r'f_H=\tfrac13;\ f_T,f_P,f_F\notin\mathbb{Q}',
     value=HAT_FREQ,
     method='Perron eigenvector of the four-metatile Hat substitution, over '
            'Q(sqrt5)')

fact(key='aperiodicity_witness',
     claim=r'9/9\ \text{vs}\ 3/4\ \text{irrational}',
     value=(len(SPECTRE_IRRATIONAL), len(SPECIES),
            len(HAT_SPECIES) - len(HAT_RATIONAL), len(HAT_SPECIES)),
     method='a periodic tiling has rational frequencies, so an irrational one '
            'refutes periodicity; every Spectre species is such a witness, '
            'while the Hat commonest metatile is exactly one third and '
            'witnesses nothing',
     lean='every_spectre_frequency_is_irrational')

TRACE_OK = (M.trace() == 8 and HAT_M.trace() == 7)

fact(key='trace_accounting',
     claim=r'\operatorname{tr}=\text{unit}+\text{conjugate}:\ 8,\ 7',
     value=(M.trace(), HAT_M.trace()),
     method='the marginal pair +1 and -1 cancels and the zero modes '
            'contribute nothing, so the trace is the unit plus its conjugate '
            'in both phases -- the same accounting in two different fields',
     lean='trace_is_unit_plus_conjugate')



# ------------------------------------------------ the torus knot sandwich

def seifert_minors(k):
    """Leading principal minors of the (k-1)-dimensional (-2,1,1) form."""
    n = k - 1
    A = sp.zeros(n, n)
    for i in range(n):
        A[i, i] = -2
        if i + 1 < n:
            A[i, i + 1] = 1
            A[i + 1, i] = 1
    return [A[:m, :m].det() for m in range(1, n + 1)]


MINOR_RULE = all(seifert_minors(k)[m - 1] == (-1) ** m * (m + 1)
                 for k in (3, 5, 7, 9, 11, 13)
                 for m in range(1, k))

fact(key='minors',
     claim=r'D_m=(-1)^{m}(m+1)',
     value=MINOR_RULE,
     method='leading principal minors of the tridiagonal (-2,1,1) form, in '
            'closed form; the recurrence D_m = -2 D_{m-1} - D_{m-2} proves it '
            'for every k at once rather than for a sampled few',
     lean='seifert_minor_closed_form')

fact(key='signature',
     claim=r'\sigma(T(2,k))=-(k-1),\qquad u(T(2,k))=\tfrac{k-1}{2}',
     value={k: -(k - 1) for k in (3, 5, 7, 9, 11)},
     method='the minors alternate in sign, so the form is negative definite '
            'by Sylvester; the signature bound and one crossing change give '
            'the unknotting number from both sides')


# ---------------------------------------------- the birefringence candidate

A_SYM, B_SYM, P_SYM = sp.symbols('a b p', positive=True)


def tile_area(u, v):
    """The area of Tile(u, v), as an identity in Q(sqrt3)[a, b]."""
    return 2 * sp.sqrt(3) * u ** 2 + 3 * u * v + sp.sqrt(3) * v ** 2


fact(key='area_form',
     claim=r'A(a,b)=2\sqrt3\,a^{2}+3ab+\sqrt3\,b^{2}',
     value=tile_area(A_SYM, B_SYM),
     method='symbolic shoelace on the published vertex construction, an '
            'identity in Q(sqrt3)[a,b] rather than a fit to sample points')

fact(key='area_endpoints',
     claim=r'A(1,\sqrt3)=8\sqrt3,\ A(\sqrt3,1)=10\sqrt3',
     value=(sp.simplify(tile_area(1, sp.sqrt(3))),
            sp.simplify(tile_area(sp.sqrt(3), 1))),
     method='the Hat and Turtle endpoints of the Tile(a,b) family')

fact(key='mystic_asymmetry',
     claim=r'A(b,a)-A(a,b)=\sqrt3\,(b^{2}-a^{2})',
     value=sp.simplify(tile_area(B_SYM, A_SYM) - tile_area(A_SYM, B_SYM)),
     method='the exact content of "only Gamma moves": the area form is not '
            'symmetric, and the Mystic is the species that carries both')

ORDER_MAX = sp.radsimp(sp.simplify(10 * G_SMALL / (8 + 10 * G_SMALL)))

fact(key='order_parameter',
     claim=r'\max|f_\Gamma-f_\Delta|=\tfrac{35}{67}-\tfrac{20}{201}\sqrt{15}',
     value=ORDER_MAX,
     method='the splitting maximum at the Hat endpoint. The sqrt3 of the '
            'geometry cancels against itself and the maximum lies in '
            'Q(sqrt15) alone: the areas come from the hexagonal world, the '
            'frequencies from the Pisot one, and the order parameter keeps '
            'only the second',
     decimal=sp.N(ORDER_MAX, 8))

RHO_X = ((14 * G_SMALL + 12 * G_SMALL + 2 * G_SMALL ** 2
          + 4 * P_SYM * G_SMALL * (1 - G_SMALL)) / (14 * (1 + G_SMALL)))

fact(key='rho_x',
     claim=r'\rho_X(p)\ \text{with}\ p=\Pr(\Phi^{2})\ \text{the only unknown}',
     value=sp.simplify(RHO_X),
     method='bulk X-density from the per-species X-charges; every charge is '
            'even and all but one species is deterministic, so a single '
            'collared frequency is left open',
     decimal=(sp.N(RHO_X.subs(P_SYM, 0), 6), sp.N(RHO_X.subs(P_SYM, 1), 6)))

BETA = sp.radsimp(sp.simplify(2 * ORDER_MAX))

# ------------------------------------------------- the X-charge, measured
#
# These come from spectreatlas.py, which builds the tiling and counts bonds.
# Depths two to four are recomputed here on import; five and six are recorded
# because depth six is a quarter of a million tiles and thirty seconds, and a
# fact engine that takes half a minute to import does not get run.  Both are
# reproducible with `python3 spectreatlas.py --depth 6`.

X_CHARGE = {'Gamma2': 14, 'Gamma1': 4, 'Delta': 4, 'Sigma': 4,
            'Lambda': 2, 'Phi': (0, 2),
            'Theta': 0, 'Pi': 0, 'Xi': 0, 'Psi': 0}

ATLAS_CLASSES = 131

# p = Pr(Phi^2), the collared frequency, by depth.  It is not converged.
PHI_SPLIT = {3: (41, 60), 4: (377, 651), 5: (3175, 5904), 6: (25851, 49803)}
P_BY_DEPTH = {d: sp.Rational(n, t) for d, (n, t) in PHI_SPLIT.items()}

fact(key='x_charge',
     claim=r'X_{\Gamma_2}=14;\ X_{\Gamma_1}=X_\Delta=X_\Sigma=4;\ '
           r'X_\Lambda=2;\ X_\Phi\in\{0,2\}',
     value=X_CHARGE,
     method='built the tiling and counted, for every interior tile, the '
            'shared edges whose two canonical slot roles disagree; identical '
            'at depths two through six, and every value even',
     lean='x_charges_are_even')

fact(key='atlas',
     claim=r'131\ \text{contact classes, stable}',
     value=ATLAS_CLASSES,
     method='the unordered adjacency classes (species and slot on each side) '
            'number 131 at depths three, four, five and six with none gained '
            'and none lost, so per-slot statements are statements about the '
            'infinite tiling')

fact(key='phi_drift',
     claim=r'p=\Pr(\Phi^{2}):\ 0.683,\,0.579,\,0.538,\,0.519',
     value=P_BY_DEPTH,
     method='the flavour split by depth three to six. It is still moving: the '
            'value near 0.58 is what depth four gives and depth six gives '
            '0.519, so it is a measurement at a depth and not a limit')

RHO_AT_HALF = sp.radsimp(sp.simplify(RHO_X.subs(P_SYM, sp.Rational(1, 2))))

fact(key='rho_x_at_half',
     claim=r'p=\tfrac12\Longrightarrow\rho_X=\frac{2g}{1+g}=1-\frac{\sqrt{15}}{5}',
     value=RHO_AT_HALF,
     method='conditional, not established: repeated Aitken extrapolation of '
            'the depth three to six values gives 0.5107 then 0.5035, which is '
            'evidence for one half and not a proof of it. If it holds the '
            'bulk X-density has a closed form in Q(sqrt15)',
     decimal=sp.N(RHO_AT_HALF, 8))

# --------------------------------------- the chiral angle, measured in a lab
#
# Moritake et al. fabricated the Hat quasilattice in SiN and measured its
# diffraction.  The pattern is chiral, and the twist that makes it chiral has
# an exact closed form -- which turns out to be a function of the inflation
# factor this paper already computes, though the experiment does not write it
# that way.

PHI = (1 + sp.sqrt(5)) / 2
COS_CHIRAL = sp.radsimp(sp.simplify((3 * PHI - 1) / 4))
THETA_CHIRAL = sp.deg(sp.acos(COS_CHIRAL))

fact(key='chiral_angle',
     claim=r'\cos\theta_{\rm chiral}=\tfrac{1}{8}(1+3\sqrt5)',
     value=COS_CHIRAL,
     method='Moritake et al. 2026: the metatile twist accumulates as a ratio '
            'of Fibonacci terms F(2n+1)/F(2n-1), which tends to phi squared, '
            'and the limiting angle is this arccosine -- measured in '
            'diffraction from a fabricated SiN quasilattice',
     decimal=sp.N(THETA_CHIRAL, 6))

fact(key='angle_from_kappa',
     claim=r'\cos\theta_{\rm chiral}=\tfrac14(\kappa-3)',
     value=sp.simplify((3 * PHI - 1) - (HAT_LAM2 - 3)),
     method='3 phi - 1 and phi^4 - 3 are the same number, so the measured '
            'chiral angle is a function of the Hat inflation factor. The '
            'experiment and the substitution spectrum meet at one constant',
     lean='chiral_angle_from_inflation')

SPECTRE_NAIVE = sp.N((LAM2 - 3) / 4, 6)

fact(key='spectre_angle_differs',
     claim=r'(\lambda^{2}-3)/4>1',
     value=SPECTRE_NAIVE,
     method='substituting the Spectre inflation factor into the Hat formula '
            'gives a cosine greater than one, so the Spectre cannot share it; '
            'its own twist is computed below from the supertile geometry, and '
            'it is not the Q(sqrt15) angle this fact once anticipated')


# ------------------------------------------- the Spectre's own twist, computed
#
# The Hat twist above needed the Hat's metatile geometry, and the sentence that
# followed it (the Spectre "cannot share it") left the Spectre's own twist as
# an open prediction because that needs the Spectre's metatile geometry and not
# merely its eigenvalues.  The geometry is already in the substitution: a
# supertile carries a quadrilateral (four tile vertices), and building the next
# supertile moves those four points by an exact map.  The map is the one
# quoted input here, transcribed from spectre.py's buildSupertiles (the seven
# (turn, from-corner, to-corner) steps below and the four output corners); the
# selftest runs spectre.py itself and checks the result against it, so the
# transcription is not taken on trust.
#
# With corners written as complex numbers w_1..w_4 the rotations are sixth
# roots of unity, so the unmirrored map is complex linear, w -> C w, with C over
# Z[omega].  The construction then mirrors (x -> -x, i.e. z -> -conj z), so one
# level is the conjugate-linear map  F(w) = D conj(w)  with D = -conj(C), and two
# levels are  B = D conj(D), complex linear again.

TWIST_STEPS = ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1),
               (0, 2, 0), (60, 3, 1), (-120, 3, 3))
TWIST_OUT = ((6, 2), (5, 1), (3, 2), (0, 1))  # (transformation, base corner)


def _unit(deg):
    """exp(i deg) for a multiple of 60 degrees, as an exact element of Q(sqrt3, i)."""
    k = (deg // 60) % 6
    return [sp.Integer(1), sp.Rational(1, 2) + sp.sqrt(3) * sp.I / 2,
            -sp.Rational(1, 2) + sp.sqrt(3) * sp.I / 2, sp.Integer(-1),
            -sp.Rational(1, 2) - sp.sqrt(3) * sp.I / 2,
            sp.Rational(1, 2) - sp.sqrt(3) * sp.I / 2][k]


def twist_quad0(a=1, b=1):
    """Corners 3, 5, 7, 11 of the Spectre tile Tile(a, b): the quadrilateral of the
    level-0 supertile, as exact complex numbers."""
    r3 = sp.sqrt(3)
    a, b = sp.sympify(a), sp.sympify(b)   # exact: a python int / 2 would be a float
    return sp.Matrix([
        (a + a / 2 + b * r3 / 2) + sp.I * (-a * r3 / 2 + b / 2),
        (2 * a + a / 2 + b * r3 / 2) + sp.I * (-a * r3 / 2 + b + b / 2),
        3 * a + sp.I * 2 * b,
        (a / 2 - b * r3 / 2) + sp.I * (a * r3 / 2 + 2 * b - b / 2)])


def twist_maps():
    """(C, D, B): the unmirrored one-level map, the conjugate-linear one-level map
    F(w) = D conj(w), and the two-level map B = D conj(D); exact 4 x 4 matrices."""
    ws = sp.symbols('w1:5')
    total, trans = 0, [(sp.Integer(1), sp.Integer(0))]
    for angle, src, dst in TWIST_STEPS:
        total += angle
        rot = _unit(total)
        prev = trans[-1]
        trans.append((rot, sp.expand(prev[0] * ws[src] + prev[1] - rot * ws[dst])))
    out = [sp.expand(trans[t][0] * ws[c] + trans[t][1]) for t, c in TWIST_OUT]
    C = sp.Matrix([[sp.radsimp(sp.expand_complex(e.coeff(w))) for w in ws] for e in out])
    D = -C.applyfunc(sp.conjugate)
    B = (D * D.applyfunc(sp.conjugate)).applyfunc(
        lambda e: sp.radsimp(sp.expand_complex(e)))
    return C, D, B


_Q8_INDEX = {sp.sqrt(3) ** (i & 1) * sp.sqrt(5) ** ((i >> 1) & 1) * sp.I ** (i >> 2): i
             for i in range(8)}


class _Q8(object):
    """An element of Q(sqrt3, sqrt5, i) as eight rational coordinates on the basis
    sqrt3^p sqrt5^q i^r.  (sympy's algebraic-field conversion works too, but takes
    a third of a second per number; this is exact and instant.)"""

    __slots__ = ('c',)

    def __init__(self, coords):
        self.c = tuple(sp.Rational(v) for v in coords)

    @classmethod
    def from_sympy(cls, e):
        out = [sp.Rational(0)] * 8
        for term in sp.Add.make_args(sp.expand(sp.expand_complex(e))):
            coeff, rest = term.as_coeff_Mul()
            if rest not in _Q8_INDEX:
                raise ValueError('not in Q(sqrt3, sqrt5, i): %s' % term)
            out[_Q8_INDEX[rest]] += sp.Rational(coeff)
        return cls(out)

    def to_sympy(self):
        basis = [sp.Integer(1), sp.sqrt(3), sp.sqrt(5), sp.sqrt(15)]
        return sp.radsimp(sum(self.c[i] * basis[i & 3] * (sp.I if i & 4 else 1) for i in range(8)))

    def __mul__(self, o):
        if not isinstance(o, _Q8):
            return _Q8([v * o for v in self.c])
        out = [sp.Rational(0)] * 8
        for i, x in enumerate(self.c):
            if x == 0:
                continue
            for j, y in enumerate(o.c):
                if y == 0:
                    continue
                f = 1
                if (i & 1) and (j & 1):
                    f *= 3
                if (i & 2) and (j & 2):
                    f *= 5
                if (i & 4) and (j & 4):
                    f *= -1
                out[i ^ j] += x * y * f
        return _Q8(out)

    __rmul__ = __mul__

    def __add__(self, o):
        return _Q8([x + y for x, y in zip(self.c, o.c)])

    def __sub__(self, o):
        return _Q8([x - y for x, y in zip(self.c, o.c)])

    def __neg__(self):
        return _Q8([-x for x in self.c])

    def _flip(self, mask):
        return _Q8([-x if bin(i & mask).count('1') % 2 else x for i, x in enumerate(self.c)])

    def conj(self):
        return self._flip(4)

    def inverse(self):
        prod = _Q8([1, 0, 0, 0, 0, 0, 0, 0])
        for mask in range(1, 8):                 # the seven non-trivial automorphisms
            prod = prod * self._flip(mask)
        norm = (self * prod).c
        if any(norm[i] != 0 for i in range(1, 8)) or norm[0] == 0:
            raise ZeroDivisionError('zero or inconsistent norm')
        return prod * (1 / norm[0])

    def __truediv__(self, o):
        return self * o.inverse()

    def __eq__(self, o):
        return self.c == o.c

    def __ne__(self, o):
        return not self == o

    def is_zero(self):
        return all(v == 0 for v in self.c)


def _q8_nullspace(rows):
    """One null vector of a square matrix (list of lists of _Q8) with a one-dimensional
    kernel, by Gauss-Jordan elimination."""
    n = len(rows)
    m = [list(r) for r in rows]
    pivots, r = [], 0
    for col in range(n):
        piv = next((i for i in range(r, n) if not m[i][col].is_zero()), None)
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        inv = m[r][col].inverse()
        m[r] = [e * inv for e in m[r]]
        for i in range(n):
            if i != r and not m[i][col].is_zero():
                f = m[i][col]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        pivots.append(col)
        r += 1
    free = [c for c in range(n) if c not in pivots]
    if len(free) != 1:
        raise AssertionError('the kernel is not one-dimensional')
    f = free[0]
    vec = [_Q8([0] * 8) for _ in range(n)]
    vec[f] = _Q8([1, 0, 0, 0, 0, 0, 0, 0])
    for row, col in enumerate(pivots):
        vec[col] = -m[row][f]
    return vec


def _twist_eigen():
    """Exact dominant eigen-data of the quadrilateral map, in Q(sqrt3, sqrt5, i)
    (where sqrt15 = sqrt3 sqrt5): the right and left eigenvectors v, l of B for
    lambda^2, and mu with F v = mu v."""
    _, D, B = twist_maps()
    lam2 = _Q8.from_sympy(LAM2)
    Bq = [[_Q8.from_sympy(B[i, j]) for j in range(4)] for i in range(4)]
    one, zero = _Q8([1, 0, 0, 0, 0, 0, 0, 0]), _Q8([0] * 8)
    shifted = [[Bq[i][j] - (lam2 if i == j else zero) for j in range(4)] for i in range(4)]
    v = _q8_nullspace(shifted)
    l = _q8_nullspace([[shifted[j][i] for j in range(4)] for i in range(4)])
    Dq = [[_Q8.from_sympy(D[i, j]) for j in range(4)] for i in range(4)]
    vbar = [e.conj() for e in v]
    Dv = []
    for i in range(4):
        acc = zero
        for j in range(4):
            acc = acc + Dq[i][j] * vbar[j]
        Dv.append(acc)
    k = next(i for i in range(4) if not v[i].is_zero())
    mu = Dv[k] / v[k]
    if any(Dv[i] != mu * v[i] for i in range(4)):    # F maps the eigenspace to itself
        raise AssertionError('F does not preserve the dominant eigenspace')
    return v, l, mu


_TWIST_EIGEN = _twist_eigen()


def twist_exact(a=1, b=1):
    """The dominant behaviour of the quadrilateral under F, exactly.

    B v = lambda^2 v has a one-dimensional eigenspace, and F maps it to itself,
    F v = mu v with |mu|^2 = lambda^2.  Writing the quadrilateral of level 0 as
    s v plus a part that is smaller by the factor lambda^2 per two levels, the
    quadrilaterals of levels 2m and 2m+1 are asymptotic to lambda^{2m} s v and
    lambda^{2m} conj(s) mu v, so one level multiplies by  c = (conj(s)/s) mu
    with s = (l.Q0)/(l.v) for the left eigenvector l.  Returns (z, mu) with
    z = c^2 / lambda^2 = exp(-2 i theta), as sympy numbers."""
    v, l, mu = _TWIST_EIGEN
    q0 = [_Q8.from_sympy(e) for e in twist_quad0(a, b)]
    zero = _Q8([0] * 8)
    num, den = zero, zero
    for i in range(4):
        num = num + l[i] * q0[i]
        den = den + l[i] * v[i]
    s_ = num / den
    c = s_.conj() / s_ * mu
    z = c * c / _Q8.from_sympy(LAM2)
    return sp.radsimp(sp.expand_complex(z.to_sympy())), mu.to_sympy()


TWIST_B = twist_maps()[2]
TWIST_Z, TWIST_MU = twist_exact()
TWIST_COS2 = sp.radsimp(sp.simplify(sp.re(TWIST_Z)))      # cos 2 theta
TWIST_SIN2 = sp.radsimp(sp.simplify(-sp.im(TWIST_Z)))     # sin 2 theta
TWIST_TAN = sp.radsimp(sp.simplify(TWIST_SIN2 / (1 + TWIST_COS2)))
TWIST_THETA = sp.deg(sp.atan(TWIST_TAN))
TWIST_COS = sp.sqrt(sp.radsimp((1 + TWIST_COS2) / 2))

fact(key='twist_two_levels',
     claim=r'\chi_{B}(x)=(x-1)^{2}(x^{2}-8x+1),\ B\,v=\lambda^{2}v',
     value=sp.factor(TWIST_B.charpoly(x).as_expr()),
     method='the supertile quadrilateral moves by an exact map, transcribed from '
            'the construction; applied twice it is complex linear, and its '
            'characteristic polynomial contains the quadratic factor of the '
            'substitution matrix. The dominant eigenvalue is the real number '
            'lambda^2, so over two levels the quadrilateral is dilated and not '
            'rotated')

fact(key='spectre_twist',
     claim=r'\tan\theta_S=\tfrac{1}{13}(5\sqrt3-6),\ \cos2\theta_S=\tfrac1{10}(4+3\sqrt3)',
     value=(TWIST_TAN, TWIST_COS2),
     method='in the limit of high level, one level relates the quadrilaterals by '
            'ratio lambda and rotation minus theta_S on even levels and plus theta_S '
            'on odd ones (they are not similar at any finite level, since lambda^2 is '
            'not in Q(sqrt3)); theta_S is the phase of '
            '(conj(s)/s) mu from the dominant eigenvector, computed exactly',
     decimal=sp.N(TWIST_THETA, 8))

fact(key='twist_field',
     claim=r'\cos2\theta_S\in\mathbb{Q}(\sqrt3),\ \notin\mathbb{Q}(\sqrt{15})',
     value=(sp.minimal_polynomial(TWIST_COS2, x), sp.minimal_polynomial(TWIST_COS, x)),
     method='the angle lies in the field of the hexagonal tile coordinates; '
            'sqrt15 enters the one-level factor only through its modulus, '
            'lambda, and drops out of the phase. The Spectre twist therefore '
            'does not belong to Q(sqrt15), which corrects the earlier '
            'statement that it would')


def twist_at(r):
    """The twist theta_S(a=1, b=r) in degrees (decimal), from the same exact derivation."""
    z, _ = twist_exact(1, r)
    return float(sp.N(-sp.deg(sp.arg(sp.N(z, 30))) / 2, 12))


TWIST_FAMILY = [(sp.Integer(1), twist_at(1))]
for _r in (sp.sqrt(3), 1 / sp.sqrt(3)):
    TWIST_FAMILY.append((_r, twist_at(_r)))

fact(key='twist_family',
     claim=r'\theta_S(b/a)\ \text{varies continuously with the tile shape}',
     value=TWIST_FAMILY,
     method='the same derivation for Tile(1, r): the rotation depends on the '
            'quadrilateral and so on the shape, passing through the Spectre '
            'value at r = 1; at r = sqrt3 the same substitution rule gives a '
            'different angle, which is not a statement about the Hat, whose '
            'tiling is generated by its own four-metatile rule')


# ------------------------------------------- the mirror channel, from Brittenham-Hermiller

fact(key='bh_theorem',
     claim=r'u(7_1\#\overline{7_1})\le5<6=u(7_1)+u(\overline{7_1})',
     value=(5, 6),
     method='Brittenham and Hermiller 2025, arXiv:2506.24088, Theorem 1.2: '
            'the first failure of additivity of unknotting number under '
            'connected sum, settling Kirby 1.69(B) in the negative')

fact(key='bh_version',
     claim=r'\text{cite v2, not v1}',
     value=('2506.24088v2', '2025-09-15'),
     method='the first version routed through two diagrams of K15n81556 '
            'asserted to be the same knot. Wang and Zhang showed by the Jones '
            'polynomial that they are a chiral knot and its mirror, and gave '
            'a direct verification anyway; version two supplies an explicit '
            'isotopy and credits them. The theorem is unaffected and now has '
            'two independent routes, but the citation has to name the version')

fact(key='bh_gap_was_chirality',
     claim=r'\text{the gap was a mirror confusion}',
     value=True,
     method='the two diagrams differed by mirroring, which is exactly the '
            'distinction the binding mechanism turns on. Worth recording as a '
            'caution about the literature and nothing more: it is not '
            'evidence for the mechanism, and no proof layer in this paper '
            'would have caught it, since the question is about knot diagrams '
            'and not about arithmetic')

fact(key='bh_deficit_bound',
     claim=r'1\le\delta\le4',
     value=(1, 4),
     method='the deficit is bounded, not determined. Brittenham-Hermiller '
            'give an upper bound u <= 5 and Scharlemann gives u >= 2, so the '
            'gap 6 - u lies between one and four; its exact value is their '
            'own Question 4.4 and is open')

fact(key='bh_threshold',
     claim=r'k,\ell\ge7\ \text{odd}',
     value=7,
     method='their Corollary 1.3 covers T(2,2k+1) with k >= 3, that is index '
            'seven and above; T(2,3) and T(2,5) are explicitly outside it and '
            'whether they admit any partner is open, so seven is where the '
            'mechanism is known to switch on rather than where it was put')

BETA_MEASURED = sp.Rational(277, 1000)
BETA_SIGMA = sp.Rational(57, 1000)

fact(key='beta',
     claim=r'\beta_{\rm pred}=\frac{70}{67}-\frac{40}{201}\sqrt{15}',
     value=BETA,
     method='the order-parameter maximum read in degrees under the stated '
            'unit postulate -- a modelling step, not a Fact, and marked as '
            'one wherever it is used',
     decimal=sp.N(BETA, 8))

fact(key='beta_tension',
     claim=r'|\beta_{\rm pred}-\beta_{\rm obs}|/\sigma',
     value=sp.N(abs(BETA - BETA_MEASURED) / BETA_SIGMA, 4),
     method='against the 2026 joint ACT DR6 + Planck PR4 value '
            '0.277 +- 0.057 degrees')


# ---------------------------------------------------------------- reporting

def census_table():
    """The two censuses side by side -- the thing that was ambiguous."""
    rows = []
    for d in range(5):
        m = matrix_census(d)
        rows.append((d, census_total(m), census_total(geometric_census(d)),
                     m['Gamma']))
    return rows


def cross_check():
    """Compare the quoted matrix against the tiling code, if it is present.

    The matrix here is typed in from the paper.  The spectre repository builds
    the same object from its own substitution rules, and the two have no code
    in common.  Agreement is not proof that either transcribes the published
    rules -- both could be wrong the same way -- but disagreement would be
    decisive, and a typo in a nine by nine integer matrix is exactly the error
    that survives every downstream check because everything downstream is
    computed from it.

    Returns None when the repository is not on the path, a message otherwise.
    """
    try:
        import numpy as np
        import chirality_e8 as CE
        import spectre as S
    except ImportError:
        return None
    theirs = sp.Matrix(np.array(CE.substitution_matrix(), dtype=int).tolist())
    if list(S.TILE_NAMES) != list(SPECIES):
        return ('the species order differs: %r against %r'
                % (list(S.TILE_NAMES), list(SPECIES)))
    if theirs != M:
        return 'the matrices differ:\n%s' % (theirs - M)
    return 'matches brentharts/spectre chirality_e8.substitution_matrix()'


def cross_check_twist(max_level=6, moment_level=4):
    """Check the exact twist against spectre.py itself, by two routes.

    twist_maps() is transcribed from spectre.py's buildSupertiles, so this runs
    spectre.py and compares:

    1. the quadrilaterals it builds at levels 0..max_level: consecutive edge
       vectors must be related by the factor lambda e^{-i theta_S} on even levels
       and lambda e^{+i theta_S} on odd ones, and two levels by the real number
       lambda^2;
    2. the placed tiles of the level-moment_level supertile, which does not use the
       quadrilaterals at all: the complex central moments of order 2, 3 and 4 of
       the filled region must rotate between consecutive levels by p theta_S in
       the same way (a rotation of the region, an area-weighted statement).

    Returns None when spectre.py is not importable, otherwise a message that
    starts with 'matches' or says what disagreed."""
    try:
        import numpy as np
        import spectre as S
    except ImportError:
        return None
    theta = float(sp.N(TWIST_THETA / sp.deg(1), 15))          # radians
    lam = float(sp.N(LAM, 15))
    quads = [np.array([complex(p[0], p[1]) for p in
                       S.buildSpectreTiles(n, 1.0, 1.0)['Delta'].quad]) for n in range(max_level + 1)]
    for n in range(max_level - 2, max_level):
        e0 = np.diff(np.append(quads[n], quads[n][0]))
        e1 = np.diff(np.append(quads[n + 1], quads[n + 1][0]))
        e2 = np.diff(np.append(quads[n + 2], quads[n + 2][0])) if n + 2 <= max_level else None
        want = lam * np.exp(-1j * theta * (1 if n % 2 == 0 else -1))
        if np.max(np.abs(e1 / e0 - want)) > 1e-3 * lam:
            return 'one-level quadrilateral ratio %s differs from lambda e^{-+i theta}' % (e1 / e0)
        if e2 is not None and np.max(np.abs(e2 / e0 - lam ** 2)) > 1e-3 * lam ** 2:
            return 'two-level quadrilateral ratio %s differs from lambda^2' % (e2 / e0)
    pts = S.get_spectre_points(1.0, 1.0)
    pz = pts[:, 0] + 1j * pts[:, 1]
    gx, gw = np.polynomial.legendre.leggauss(8)
    gt, gw = (gx + 1) / 2, gw / 2

    def central_moments(level):
        placed = []
        S.buildSpectreTiles(level, 1.0, 1.0)['Delta'].forEachTile(lambda T, lab: placed.append(T.copy()))
        T = np.array(placed)
        z = (pts[:, 0] * T[:, 0, 0][:, None] + pts[:, 1] * T[:, 0, 1][:, None] + T[:, 0, 2][:, None]
             + 1j * (pts[:, 0] * T[:, 1, 0][:, None] + pts[:, 1] * T[:, 1, 1][:, None] + T[:, 1, 2][:, None]))
        sign = np.sign(T[:, 0, 0] * T[:, 1, 1] - T[:, 0, 1] * T[:, 1, 0])
        z0, d = z, np.roll(z, -1, axis=1) - z
        raw = []
        for p in range(5):
            acc = 0
            for t, w in zip(gt, gw):
                zz = z0 + t * d
                acc = acc + w * (np.conj(z0) + t * np.conj(d)) * zz ** p * d
            raw.append(((acc.sum(axis=1) / 2j) * sign).sum())
        c = raw[1] / raw[0]
        from math import comb
        return [sum(comb(p, k) * (-c) ** (p - k) * raw[k] for k in range(p + 1)) for p in range(5)]

    lo, hi = central_moments(moment_level - 1), central_moments(moment_level)
    for p in (2, 3, 4):
        rot = np.angle(hi[p] / lo[p])                        # = p * (rotation of the region)
        want = p * (theta if (moment_level - 1) % 2 else -theta)
        off = (rot - want + np.pi) % (2 * np.pi) - np.pi
        if abs(off) > np.radians(0.12) * p:
            return ('the order-%d moments rotate by %.4f degrees per level, not %.4f'
                    % (p, np.degrees(rot) / p, np.degrees(want) / p))
        if abs(abs(hi[p] / lo[p]) / lam ** (p + 2) - 1) > 0.02:
            return 'the order-%d moments do not scale like lambda^%d' % (p, p + 2)
    return ('matches spectre.py: quadrilaterals at levels 0..%d and the moments of the level-%d supertile '
            'rotate by -+%.4f degrees per level with ratio lambda'
            % (max_level, moment_level, float(sp.N(TWIST_THETA, 8))))


def cross_check_boundary():
    """Compare BOUNDARY_SEED against an exact-arithmetic second count.

    The perimeter sequence is this module's one measured input, in the same
    way M is: it is counted from placed tiles, not derived here.  The count
    in spectre_boundary.py identifies two edges when their endpoint
    coordinates agree to four decimal places, and a rounding key of that
    kind can fail in both directions -- merging edges that differ below the
    tolerance, splitting edges that agree above it.  Nothing in a float
    computation bounds that; it can only be removed.

    pyCICY.theories.spectre counts the same sequence with edges identified
    by EQUALITY of exact Q(sqrt3) endpoint pairs, from an independent
    transcription of the placement chain.  If the two agree, the rounding
    key is no longer the weakest link in the boundary results; if they
    disagree, the float count is wrong and everything downstream with it.

    Returns None when pyCICY is not on the path, a message otherwise.
    """
    try:
        from pyCICY.theories import spectre as X
    except ImportError:
        return None
    try:
        theirs = X.boundary_sequence(len(BOUNDARY_SEQ) - 1)
    except Exception as exc:                      # pragma: no cover
        return 'pyCICY boundary_sequence failed: %s' % exc
    if list(theirs) != list(BOUNDARY_SEQ):
        return ('the perimeter sequences differ: %r against %r'
                % (theirs, BOUNDARY_SEQ))
    return ('matches pyCICY.theories.spectre.boundary_sequence() in exact '
            'Q(sqrt3), so the four-decimal rounding key is not load-bearing')



def _lean_theorem_count():
    """Theorems in the generated Lean file (imported late: spectrelean imports
    this module)."""
    import spectrelean
    return len(spectrelean.theorem_names(spectrelean.document()))


def check_readme(path='Readme.md'):
    """Check the README's headline numbers against this module.

    The README is hand-written prose -- it is an argument, not a build
    artefact, so it is not generated.  But it quotes numbers, and a quoted
    number is exactly the thing that goes stale when the facts move.  This
    checks the ones worth checking and says which are missing.
    """
    import io
    try:
        with io.open(path, encoding='utf-8') as handle:
            text = handle.read()
    except IOError:
        return ['%s not found' % path]

    wanted = [
        ('fact count', '%d Facts' % len(FACTS)),
        ('Lean theorem count', '%d theorems' % _lean_theorem_count()),
        ('Perron eigenvalue', '4+\u221a15'),
        ('atlas classes', str(ATLAS_CLASSES)),
        ('Mystic X-charge', 'X_\u0393\u2082 = %d' % X_CHARGE['Gamma2']),
        ('matrix census', ', '.join(str(t) for t in MATRIX_TOTALS[:5])),
        ('geometric census', ', '.join(str(t) for t in GEOMETRIC_TOTALS[:5])),
        ('phi split by depth',
         ', '.join('%.3f' % float(P_BY_DEPTH[d]) for d in sorted(P_BY_DEPTH))),
        ('chiral angle', str(get('chiral_angle').decimal)[:5]),
    ]
    missing = ['%s (%r)' % (name, value) for name, value in wanted
               if value not in text]
    return missing


def selftest():
    """Check every Fact. Returns the number of problems."""
    problems = []

    def fail(msg):
        problems.append(msg)

    # the quoted input must be what the paper prints
    if M.shape != (9, 9):
        fail('the substitution matrix is not nine by nine')
    if any(c < 0 for c in M):
        fail('a substitution count is negative')

    # the spectral facts
    if sp.simplify(CHARPOLY - x ** 5 * (x - 1) * (x + 1) *
                   (x ** 2 - 8 * x + 1)) != 0:
        fail('the characteristic polynomial is not the claimed factorisation')
    if sp.simplify(LAM2 - sp.Rational(1, 1) * (4 + SQ15)) != 0:
        fail('the Perron eigenvalue is not 4 + sqrt15')
    if sp.simplify((M * _perron_col() - LAM2 * _perron_col())) != sp.zeros(9, 1):
        fail('the Perron vector does not satisfy M v = lambda^2 v')
    if sp.simplify(sum(FREQ.values()) - 1) != 0:
        fail('the frequencies do not sum to one')
    if get('channel_hex').value != 6:
        fail('the hexagonal channel identity is not six')
    if get('channel_pent').value != 10:
        fail('the pentagonal channel identity is not ten')
    if sp.simplify(get('contraction').value - (31 - 8 * SQ15)) != 0:
        fail('the contraction rate is not 31 - 8 sqrt15')
    if get('pell').value != 1:
        fail('4^2 - 15 is not one, so lambda^2 is not a unit')

    # the zero sector
    if RANKS[1] != 5 or any(RANKS[n] != 4 for n in (2, 3, 4)):
        fail('the rank profile is not 5 then 4: %r' % RANKS)
    if M.is_diagonalizable():
        fail('M is diagonalizable, so there is no Jordan pair to report')

    # the charges, and the census they eat
    if not get('q_plus_eigen').value:
        fail('Q+ is not a left eigenvector of M')
    if not get('q_minus_eigen').value:
        fail('Q- is not a left eigenvector of M')
    if Q_PLUS_BY_DEPTH != [-1, -1, -1, -1]:
        fail('Q+ is not identically -1: %r' % (Q_PLUS_BY_DEPTH,))
    if Q_MINUS_BY_DEPTH != [1, -1, 1, -1]:
        fail('Q- does not alternate: %r' % (Q_MINUS_BY_DEPTH,))
    # and the thing that went wrong in the prose: the OTHER census must not
    # give those values, or there would be nothing to be careful about
    other = [evaluate(Q_PLUS, _with_mystic(d)) for d in range(1, 5)]
    if other == Q_PLUS_BY_DEPTH:
        fail('the two censuses agree on Q+, so the note about which one the '
             'charges eat is unnecessary and should be removed')

    if MATRIX_TOTALS[:5] != [1, 8, 63, 496, 3905]:
        fail('the matrix census is not 1, 8, 63, 496, 3905: %r'
             % MATRIX_TOTALS[:5])
    if GEOMETRIC_TOTALS[:5] != [1, 9, 71, 559, 4401]:
        fail('the geometric census is not 1, 9, 71, 559, 4401: %r'
             % GEOMETRIC_TOTALS[:5])

    # the census satisfies the recurrence the spectrum predicts
    t = MATRIX_TOTALS
    if any(t[n + 1] != 8 * t[n] - t[n - 1] for n in range(2, len(t) - 1)):
        fail('the census does not obey x(n+1) = 8x(n) - x(n-1)')

    if not LEDGER:
        fail('the ledger identity fails')
    if not MYSTIC_RULE:
        fail('the geometric census is not the matrix census plus its '
             'predecessor')
    if list(M[INDEX['Gamma'], :]) != [1] * 9:
        fail('row Gamma is not all ones, so the Mystic rule has no reason')

    # the knot layer
    if not MINOR_RULE:
        fail('the Seifert minors are not (-1)^m (m+1)')

    # aperiodicity, as irrationality, in both phases
    if len(SPECTRE_IRRATIONAL) != len(SPECIES):
        fail('some Spectre frequency is rational, so not every species '
             'witnesses aperiodicity: %r' % (SPECTRE_IRRATIONAL,))
    if HAT_RATIONAL != ['H']:
        fail('the Hat rational frequencies are not exactly {H}: %r'
             % (HAT_RATIONAL,))
    if sp.simplify(HAT_FREQ['H'] - sp.Rational(1, 3)) != 0:
        fail('f_H is not exactly one third')
    if sp.simplify(sum(HAT_FREQ.values()) - 1) != 0:
        fail('the Hat frequencies do not sum to one')
    # the frequencies must really be algebraic integers, or the Lean rendering
    # would need rational arithmetic it does not have
    for n, (a, b) in FREQ_Z15.items():
        if a != int(a) or b != int(b):
            fail('the %s frequency is not in Z[sqrt15]: %r' % (n, (a, b)))
    if not TRACE_OK:
        fail('the traces are not 8 and 7')
    if M.trace() != sp.simplify(LAM2 + 1 / LAM2):
        fail('trace M is not lambda^2 + lambda^-2')
    if HAT_M.trace() != sp.simplify(HAT_LAM2 + 1 / HAT_LAM2):
        fail('trace H is not kappa + kappa^-1')

    # the other phase
    if sp.simplify(HAT_CHARPOLY - (x - 1) * (x + 1) * (x ** 2 - 7 * x + 1)) != 0:
        fail('the Hat characteristic polynomial is not the claimed one')
    if sp.simplify(HAT_LAM2 - PHI ** 4) != 0:
        fail('the Hat inflation factor is not phi^4')
    if sp.simplify(HAT_LAM2 - LAM2) == 0:
        fail('the two phases have the same inflation factor, which would '
             'remove the arithmetic half of the phase distinction')

    # the deformation axis
    if sp.simplify(tile_area(1, sp.sqrt(3)) - 8 * sp.sqrt(3)) != 0:
        fail('the Hat endpoint area is not 8 sqrt3')
    if sp.simplify(tile_area(sp.sqrt(3), 1) - 10 * sp.sqrt(3)) != 0:
        fail('the Turtle endpoint area is not 10 sqrt3')
    if sp.simplify(get('mystic_asymmetry').value
                   - sp.sqrt(3) * (B_SYM ** 2 - A_SYM ** 2)) != 0:
        fail('the mystic asymmetry is not sqrt3 (b^2 - a^2)')
    if sp.simplify(ORDER_MAX - (sp.Rational(35, 67)
                                - sp.Rational(20, 201) * SQ15)) != 0:
        fail('the order-parameter maximum does not match its closed form')
    if sp.sqrt(3) in sp.simplify(ORDER_MAX).atoms(sp.Pow):
        fail('the order parameter still carries a sqrt3, so the claim that '
             'the geometry cancels is wrong')
    if sp.simplify(BETA - 2 * ORDER_MAX) != 0:
        fail('beta is not twice the order-parameter maximum')

    # the modelling step, which is arithmetic even though its premise is not
    if sp.simplify(BETA - (sp.Rational(70, 67) -
                           sp.Rational(40, 201) * SQ15)) != 0:
        fail('the beta closed form does not match')

    # every Fact must say how it was got
    for f in FACTS:
        if not f.method or len(f.method) < 15:
            fail('%s does not say how it was computed' % f.key)
        if f.value is None:
            fail('%s has no value' % f.key)

    # the X-charge, against a live rebuild of the atlas at a cheap depth
    try:
        import spectreatlas as A
        live = A.analyse(3)
        for species, hist in live['charges'].items():
            seen = tuple(sorted(hist)) if len(hist) > 1 else sorted(hist)[0]
            if X_CHARGE.get(species) != seen:
                fail('the atlas gives X_%s = %r, the table says %r'
                     % (species, seen, X_CHARGE.get(species)))
        if len(live['atlas']) != ATLAS_CLASSES:
            fail('the atlas has %d classes, not %d'
                 % (len(live['atlas']), ATLAS_CLASSES))
    except ImportError:
        pass

    # p must be recorded as drifting, not as settled
    values = [float(P_BY_DEPTH[d]) for d in sorted(P_BY_DEPTH)]
    if not all(a > b for a, b in zip(values, values[1:])):
        fail('the Phi split is no longer monotone: %r' % values)
    if abs(values[-1] - 0.58) < 0.02:
        fail('the deepest Phi split is near 0.58, so the drift note should go')
    if sp.simplify(RHO_AT_HALF - (1 - SQ15 / 5)) != 0:
        fail('rho_X at p = 1/2 is not 1 - sqrt15/5')

    # the deficit bound must survive the v1 correction unchanged
    if get('bh_deficit_bound').value != (1, 4):
        fail('the deficit bound moved; Wang-Zhang did not change it')
    if 'v2' not in get('bh_version').value[0]:
        fail('the Brittenham-Hermiller citation does not name version two')

    # the measured chiral angle
    if sp.simplify(COS_CHIRAL - (1 + 3 * sp.sqrt(5)) / 8) != 0:
        fail('the chiral cosine is not (1 + 3 sqrt5)/8')
    if abs(float(sp.N(THETA_CHIRAL)) - 15.52) > 0.01:
        fail('the chiral angle is not 15.52 degrees: %s'
             % float(sp.N(THETA_CHIRAL)))
    if sp.simplify((3 * PHI - 1) - (HAT_LAM2 - 3)) != 0:
        fail('3 phi - 1 is not kappa - 3, so the angle does not follow from '
             'the inflation factor')
    if float(SPECTRE_NAIVE) <= 1:
        fail('the Spectre substitution into the Hat formula gives a valid '
             'cosine, so the claim that the phases cannot share it is wrong')

    verdict = cross_check()
    if verdict is not None and not verdict.startswith('matches'):
        fail('the quoted matrix disagrees with the tiling code: ' + verdict)

    # the Spectre twist, derived exactly
    target = ((4 + 3 * sp.sqrt(3)) + sp.I * (3 - 4 * sp.sqrt(3))) / 10
    if sp.simplify(TWIST_Z - target) != 0:
        fail('exp(-2 i theta_S) is not (4 + 3 sqrt3 + i (3 - 4 sqrt3)) / 10')
    if sp.simplify(TWIST_Z * sp.conjugate(TWIST_Z) - 1) != 0:
        fail('the twist factor does not have modulus one')
    if sp.simplify(TWIST_TAN - (5 * sp.sqrt(3) - 6) / 13) != 0:
        fail('tan theta_S is not (5 sqrt3 - 6) / 13')
    if abs(float(sp.N(TWIST_THETA)) - 11.5650512) > 1e-6:
        fail('the Spectre twist is not 11.56505 degrees')
    if sp.degree(sp.minimal_polynomial(TWIST_COS2, x), x) != 2 or TWIST_COS2.has(sp.sqrt(5)) \
            or TWIST_COS2.has(sp.sqrt(15)):
        fail('cos 2 theta_S is not a quadratic irrational of Q(sqrt3)')
    if sp.simplify(sp.minimal_polynomial(TWIST_COS, x) - (400 * x ** 4 - 560 * x ** 2 + 169)) != 0:
        fail('cos theta_S does not have minimal polynomial 400x^4 - 560x^2 + 169')
    quad_factor = [pp for pp, _ in sp.factor_list(sp.expand(CHARPOLY))[1] if sp.degree(pp, x) == 2]
    two_level = sp.factor_list(sp.expand(get('twist_two_levels').value))[1]
    if not any(sp.simplify(pp - qf) == 0 for pp, _ in two_level for qf in quad_factor):
        fail('the two-level map does not carry the quadratic factor of the substitution matrix')
    if sp.simplify(TWIST_MU * sp.conjugate(TWIST_MU) - LAM2) != 0:
        fail('|mu|^2 is not lambda^2')
    if abs(TWIST_FAMILY[1][1] - 27.2366) > 1e-3 or abs(TWIST_FAMILY[2][1] + 2.8039) > 1e-3:
        fail('the twist at b/a = sqrt3 and 1/sqrt3 moved')
    verdict = cross_check_twist()
    if verdict is not None and not verdict.startswith('matches'):
        fail('the twist disagrees with the tiling code: ' + verdict)

    for msg in problems:
        print('FAIL  ' + msg)
    if problems:
        print('\n%d problem(s).' % len(problems))
    else:
        print('spectrefacts: %d facts, %d with a Lean rendering -- all check.'
              % (len(FACTS), sum(1 for f in FACTS if f.lean)))
        verdict = cross_check()
        print('  cross-check: %s'
              % (verdict or 'brentharts/spectre not on the path, so the '
                            'matrix is unconfirmed by a second route'))
        bverdict = cross_check_boundary()
        print('  cross-check: %s'
              % (bverdict or 'pyCICY not on the path; the exact recount for k <= 6 '
                             'is boundary_certificate/spectre_exact.py'))
    return len(problems)


def _perron_col():
    return sp.Matrix([[FREQ[n]] for n in SPECIES])


# ================================================================= boundary
#
# The perimeter of a supertile, and the operator that governs it.  Everything
# in this block is downstream of one measured input, BOUNDARY_SEED, in exactly
# the way everything else is downstream of M: it is geometry transcribed, not
# derived here, and it is audited elsewhere by a route that shares no code
# (spectre_boundary.py counts it from placed tiles; spectre_boundary_operator.py
# rederives the recurrence from the substitution rule).  Section "The boundary,
# restated" in the paper puts it in the transcription category for that reason.

BOUNDARY_SEED = (14, 46, 182)
BOUNDARY_COEFFS = (5, -3, -1)          # P(k) = 5P(k-1) - 3P(k-2) - P(k-3)


def boundary_perimeter(kmax=8):
    """P(k), boundary edges of an order-k supertile, from the recurrence."""
    out = list(BOUNDARY_SEED)
    while len(out) <= kmax:
        a, b, c = BOUNDARY_COEFFS
        out.append(a * out[-1] + b * out[-2] + c * out[-3])
    return out[:kmax + 1]


BOUNDARY_SEQ = boundary_perimeter(6)

BOUNDARY_CHI = sp.factor(sp.expand(
    x ** 3 - BOUNDARY_COEFFS[0] * x ** 2 - BOUNDARY_COEFFS[1] * x
    - BOUNDARY_COEFFS[2]))

NU = 2 + sp.sqrt(5)                    # the boundary growth rate
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47]

fact(key='boundary_recurrence',
     claim=r'P(k)=5P(k-1)-3P(k-2)-P(k-3)',
     value=BOUNDARY_SEQ,
     method='proved for every k by the finite certificate boundary_certificate/'
            'check_spectre_boundary_proof.py (93 contact types closed under the substitution, '
            'all other placements separated at every order, Cayley-Hamilton on a 95x95 integer '
            'matrix); the printed sequence is generated from the seed 14, 46, 182, and the exact '
            'count agrees with it through k = 6',
     lean='boundary_values')

fact(key='boundary_chi',
     claim=r'\chi_{\partial}(x)=(x-1)(x^{2}-4x-1)',
     value=BOUNDARY_CHI,
     method='characteristic polynomial of the companion matrix of the '
            'recurrence; the marginal eigenvalue +1 is a conserved boundary '
            'quantity')

fact(key='boundary_growth',
     claim=r'\nu=2+\sqrt{5}=\varphi^{3}',
     value=sp.radsimp(NU),
     method='Perron root of the boundary operator; equal to the cube of the '
            'golden ratio',
     decimal=sp.N(NU, 12))

fact(key='boundary_field',
     claim=r'\nu\in\mathbb{Q}(\sqrt{5}),\ \lambda^{2}\in\mathbb{Q}(\sqrt{15})',
     value=(sp.minimal_polynomial(NU, x),
            sp.minimal_polynomial(LAM2, x)),
     method='area and perimeter of the same tile inflate in different '
            'quadratic fields')

BOUNDARY_DIM = sp.log(NU) / sp.log(LAM)

fact(key='boundary_dimension',
     claim=r'\dim_{B}\partial K=\log\nu/\log\lambda',
     value=BOUNDARY_DIM,
     method='box dimension of the boundary of the limiting supertile, proved by the certificate '
            'boundary_certificate/check_spectre_fractal.py (contact matrix with simple dominant root nu, '
            'an interior grandchild, a lower-bound set of 20 contact types)',
     decimal=sp.N(BOUNDARY_DIM, 12))

MU_BALANCE = sp.radsimp(sp.simplify(LAM2 / NU))
S_STAR = sp.simplify(1 - sp.log(NU) / sp.log(LAM2))
MU_MINPOLY = sp.minimal_polynomial(MU_BALANCE, x)

fact(key='balanced_exponent',
     claim=r's^{*}=1-\log\nu/\log\lambda^{2}',
     value=S_STAR,
     method='the thickness exponent making every substitution level carry '
            'comparable volume (a modelling choice); transcendental, being an '
            'irrational ratio of logarithms of algebraic numbers (Gelfond-Schneider)',
     decimal=sp.N(S_STAR, 12))

fact(key='compositum',
     claim=r'\mu=\lambda^{2}/\nu\ \text{has degree }4',
     value=MU_MINPOLY,
     method='the balanced ratio lies in Q(sqrt3, sqrt5) and in neither '
            'quadratic field alone; both lambda^2 and nu are Spectre quantities')

GOLDEN_ABSENT = not any(
    sp.simplify(pp - (x ** 2 - 4 * x - 1)) == 0
    for poly in (CHARPOLY, HAT_CHARPOLY)
    for pp, _ in sp.factor_list(sp.expand(poly))[1])

fact(key='golden_absent',
     claim=r'(x^{2}-4x-1)\nmid\chi_{M},\ \chi_{H}',
     value=GOLDEN_ABSENT,
     method='the boundary quadratic divides neither substitution matrix; the '
            'shared field is forced by Lucas, not by a common operator')

LUCAS_DEFLATION = all(
    sp.simplify(sp.minimal_polynomial(PHI ** n, x)
                - (x ** 2 - LUCAS[n] * x + (-1) ** n)) == 0
    for n in range(1, 7))

fact(key='lucas_deflation',
     claim=r'\varphi^{n}:\ x^{2}-L_{n}x+(-1)^{n}',
     value=LUCAS_DEFLATION,
     method='every power of the golden ratio has a Lucas quadratic for its '
            'minimal polynomial, so agreement of two such polynomials carries '
            'no information beyond Q(sqrt5)')

SPECTRE_NOT_GOLDEN = all(sp.simplify(LAM2 - PHI ** n) != 0
                         for n in range(1, 13))

fact(key='spectre_not_golden',
     claim=r'\lambda^{2}\neq\varphi^{n}',
     value=SPECTRE_NOT_GOLDEN,
     method='the trace of the Spectre unit is eight, which is not a Lucas '
            'number, so the area inflation is not golden even though the '
            'perimeter inflation is')



BOUNDARY_INVARIANT = set(
    BOUNDARY_SEQ[k + 2] - 4 * BOUNDARY_SEQ[k + 1] - BOUNDARY_SEQ[k]
    for k in range(len(BOUNDARY_SEQ) - 2))

fact(key='boundary_invariant',
     claim=r'P(k+2)-4P(k+1)-P(k)=-16',
     value=sorted(BOUNDARY_INVARIANT),
     method='the quadratic factor of chi applied to the sequence annihilates '
            'the growing part and leaves the constant belonging to the '
            'marginal eigenvalue +1; for the sequence defined by the recurrence and '
            'its seed it is proved for every k in Lean, and that sequence is the perimeter '
            'at every order by the boundary certificate',
     lean='boundary_invariant')

BOUNDARY_GROWTH_OK = all(
    BOUNDARY_SEQ[k + 3] >= 4 * BOUNDARY_SEQ[k + 2] + 30
    for k in range(len(BOUNDARY_SEQ) - 3))

fact(key='boundary_growth_bound',
     claim=r'P(k+3)\ge 4P(k+2)+30',
     value=BOUNDARY_GROWTH_OK,
     method='for the sequence defined by the recurrence (proved in Lean) the '
            'perimeter more than quadruples at every step past the seed, while '
            'lambda < 4; that sequence is the perimeter at every order by the boundary '
            'certificate',
     lean='boundary_growth')

fact(key='lam_below_four',
     claim=r'\lambda^{2}<16',
     value=sp.simplify(16 - LAM2) > 0,
     method='16 - lambda^2 = 12 - sqrt15, positive since 12^2 > 15; with the '
            'growth bound above this is why the boundary outruns the linear '
            'inflation, though the inference between them is elementary real '
            'arithmetic and is not formalised',
     lean='lam2_below_sixteen')


BOUNDARY_EXACT_FIELD = 'Q(sqrt3)'

fact(key='boundary_exact',
     claim=r'P(k)\ \text{counted exactly in }\mathbb{Q}(\sqrt3)',
     value=BOUNDARY_SEQ,
     method='the same sequence recounted with edges identified by equality '
            'of exact coordinates in Z[zeta_12] rather than by a '
            'four-decimal rounding key, in an independent implementation '
            '(boundary_certificate/spectre_exact.py, "matches the exact patches for k = 0..6" in '
            'out_spectre_boundary_proof.txt), for 0 <= k <= 6; agreement removes the rounding key '
            'from the boundary results rather than bounding its error.  A second exact recount '
            '(pyCICY.theories.spectre, in Q(sqrt3)) is run by --selftest only when pyCICY is on the path')


def _with_mystic(depth):
    """The census a reader gets by following the prose: Gamma = G1 + G2."""
    counts = matrix_census(depth)
    counts = dict(counts)
    counts['Gamma'] = 2 * counts['Gamma']
    return counts


def _report():
    print(__doc__.strip().split('\n')[0])
    print()
    for f in FACTS:
        mark = 'L' if f.lean else ' '
        print(' %s %-16s %s' % (mark, f.key, f.claim))
        if f.decimal is not None:
            print('      = %s' % f.decimal)
    print()
    print('  census, both ways:')
    print('    %-6s %-10s %-12s %s' % ('depth', 'matrix', 'geometric',
                                       'Mystic'))
    for d, m, g, mys in census_table():
        print('    %-6d %-10d %-12d %d' % (d, m, g, mys))
    print()
    print('  ' + CENSUS_NOTE)


if __name__ == '__main__':
    import sys
    if '--readme' in sys.argv:
        gaps = check_readme()
        for gap in gaps:
            print('STALE  ' + gap)
        print('Readme.md: %s'
              % ('%d headline number(s) do not match this module' % len(gaps)
                 if gaps else 'headline numbers agree with spectrefacts.py'))
        sys.exit(1 if gaps else 0)
    if '--selftest' in sys.argv:
        sys.exit(1 if selftest() else 0)
    if '--census' in sys.argv:
        for row in census_table():
            print(row)
    else:
        _report()
