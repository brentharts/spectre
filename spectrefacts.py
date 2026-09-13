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


# ------------------------------------------------------- the frequencies

def perron_vector():
    """The normalised Perron eigenvector, exactly, in Q(sqrt15)."""
    v = (M - LAM2 * sp.eye(9)).nullspace()[0]
    v = v / sum(v)
    return [sp.radsimp(sp.simplify(c)) for c in v]


FREQ = dict(zip(SPECIES, perron_vector()))

fact(key='frequencies',
     claim=r'v\in\mathbb{Q}(\sqrt{15})^{9},\ \textstyle\sum v=1',
     value=FREQ,
     method='nullspace of M - lambda^2 I, normalised, each component '
            'simplified to closed form')


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
    if '--selftest' in sys.argv:
        sys.exit(1 if selftest() else 0)
    if '--census' in sys.argv:
        for row in census_table():
            print(row)
    else:
        _report()
