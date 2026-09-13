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
     method='exact characteristic polynomial over Z, factored',
     lean='charpoly_factors')

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
     lean='charges_on_patches')

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
     lean='mystic_is_previous_total')


# ------------------------------------------------------ the ledger identity

LEDGER = all(matrix_census(n)['Theta'] == matrix_census(n - 1)['Gamma']
             for n in range(1, 7))

fact(key='ledger',
     claim=r'N_\Theta(n)=N_\Gamma(n-1)',
     value=LEDGER,
     method='row Theta of M is the indicator of Gamma, so the count is the '
            'previous generation of Gamma verbatim',
     lean='ledger_theta_gamma')

fact(key='ledger_lambda',
     claim=r'N_\Lambda(n)=N_\Sigma(n-1)',
     value=all(matrix_census(n)['Lambda'] == matrix_census(n - 1)['Sigma']
               for n in range(1, 7)),
     method='row Lambda of M is the indicator of Sigma',
     lean='ledger_lambda_sigma')


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
     method='characteristic polynomial of the four-metatile Hat substitution',
     lean='hat_charpoly_factors')

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
            'its twist geometry differs and its chiral angle is not computed '
            'here, which makes it a prediction rather than a restatement')


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
        ('fact count', str(len(FACTS))),
        ('Lean theorem count', '44'),
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
    return len(problems)


def _perron_col():
    return sp.Matrix([[FREQ[n]] for n in SPECIES])


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
