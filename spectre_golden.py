#!/usr/bin/env python3
r"""spectre_golden.py -- is nu = phi^3 a coincidence?

spectre_boundary.py found that the Spectre supertile perimeter grows by
nu = 2 + sqrt(5) = phi^3, with boundary operator characteristic polynomial
(x - 1)(x^2 - 4x - 1).  Meanwhile spectrefacts has HAT_LAM2 = 7/2 + 3sqrt(5)/2
= phi^4, also in Q(sqrt(5)).  Two golden quantities in a repo whose headline
result is a Q(sqrt(15)) / Q(sqrt(5)) divide invites the conjecture that the
Spectre's boundary combinatorics ARE the Hat's -- that x^2 - 4x - 1 is
literally a factor of something built from HAT_M.

It is not.  This module is mostly a negative result, and the negative is
worth more than the conjecture was.

The refutation
--------------
    M       charpoly  x^5 (x-1)(x+1)(x^2 - 8x + 1)
    HAT_M   charpoly      (x-1)(x+1)(x^2 - 7x + 1)
    boundary charpoly      (x-1)   (x^2 - 4x - 1)

x^2 - 4x - 1 appears in neither, and a search over natural integer matrices
built from M and HAT_M (sums, powers, transposes, adjugates, shifts) finds
it in none of them.  The search is run in full and reported so that "no"
means something.

Why the shared field is not evidence
------------------------------------
phi^n has minimal polynomial

    x^2 - L_n x + (-1)^n,      L_n = 2, 1, 3, 4, 7, 11, 18, ...  (Lucas)

so x^2 - 4x - 1 is nothing more than "phi^3" and x^2 - 7x + 1 is nothing
more than "phi^4".  Once two quantities are both powers of phi, their
minimal polynomials are forced to be adjacent Lucas quadratics, and the
resemblance carries no information beyond Q(sqrt(5)) itself.  The conjecture
was reading a tautology as a clue.

Note also that the Spectre's own unit is NOT golden: x^2 - 8x + 1 has trace
8, and 8 is not a Lucas number, so 4 + sqrt(15) is not a power of phi.  The
three operators sit at traces 8, 7 and 4 with norms +1, +1 and -1.

What survives, and it is better
--------------------------------
1. The balanced exponent needs BOTH fields.  mu = lambda^2 / nu has minimal
   polynomial

       x^4 + 32x^3 - 46x^2 - 32x + 1

   -- degree four, not two.  mu lives in the compositum Q(sqrt(3), sqrt(5)),
   which contains the Spectre's Q(sqrt(15)) and the Hat's Q(sqrt(5)) and is
   equal to neither.  So the Spectre and the Hat do meet in this exponent,
   but as a compositum rather than as one dividing the other.  That is a
   real relation and it was not visible before.

2. All three operators carry the SAME marginal structure.  M and HAT_M each
   have eigenvalues +1 and -1 -- the conserved and alternating charges of
   spectrepaper.py -- and the boundary operator has +1 in its characteristic
   polynomial and, component-wise, a period-2 sector that exhibits -1
   directly.  The phase grading is not specific to the area matrix.

An honest gap
-------------
The order-3 recurrence governs the TOTAL perimeter exactly.  Component-wise,
by (species, edge index), only 34 of 52 boundary edge types obey it once the
seed transient is dropped; the rest include period-2 components and at least
one sequence (1, 7, 22, 91, 378, 1599) that obeys neither it nor the obvious
order-4 extension.  So the true boundary transfer operator is LARGER than
the 3x3 companion matrix, and seven orders of data are not enough to pin its
full spectrum down.  What is established is the Perron eigenvalue and the
total; the rest of the spectrum is open.

    python3 spectre_golden.py
    python3 spectre_golden.py --selftest
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sympy as sp

import spectrefacts as F
from spectre_selfsimilar import build_patch, edge_table, LAMBDA

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')

x = sp.Symbol('x')
PHI = (1 + sp.sqrt(5)) / 2
NU = 2 + sp.sqrt(5)
TARGET = x ** 2 - 4 * x - 1
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76]


# ---------------------------------------------------------------------------
# 1. the search
# ---------------------------------------------------------------------------

def candidate_matrices():
    """Natural integer matrices built from M and HAT_M.

    The point of enumerating many is that a single negative proves nothing --
    'I looked at HAT_M and it wasn't there' invites 'try HAT_M squared'.  The
    list covers sums, differences, powers, transposes, adjugates and integer
    shifts of both matrices, so a clean sweep is meaningful.
    """
    M, H = F.M, F.HAT_M
    I9, I4 = sp.eye(9), sp.eye(4)
    c = {
        'M': M, 'M + I': M + I9, 'M - I': M - I9, 'M + 2I': M + 2 * I9,
        'M - 2I': M - 2 * I9, 'M^2': M * M, 'M^3': M ** 3,
        'M^T': M.T, 'M + M^T': M + M.T, 'M M^T': M * M.T,
        'adj(M)': M.adjugate(),
        'HAT_M': H, 'HAT + I': H + I4, 'HAT - I': H - I4,
        'HAT + 2I': H + 2 * I4, 'HAT - 2I': H - 2 * I4,
        'HAT + 3I': H + 3 * I4, 'HAT^2': H * H, 'HAT^3': H ** 3,
        'HAT^T': H.T, 'HAT + HAT^T': H + H.T, 'HAT HAT^T': H * H.T,
        'adj(HAT)': H.adjugate(), 'HAT^2 - HAT': H * H - H,
        'HAT^2 - I': H * H - I4, 'HAT^2 + HAT': H * H + H,
    }
    return c


def has_factor(A, target=TARGET):
    poly = sp.expand(A.charpoly(x).as_expr())
    for p, _ in sp.factor_list(poly)[1]:
        if sp.simplify(p - target) == 0:
            return True, sp.factor(poly)
    return False, sp.factor(poly)


def run_search():
    rows = []
    for name, A in candidate_matrices().items():
        hit, poly = has_factor(A)
        rows.append((name, poly, hit))
    return rows


# ---------------------------------------------------------------------------
# 2. the Lucas deflation
# ---------------------------------------------------------------------------

def lucas_table(nmax=6):
    """phi^n and its minimal polynomial, showing the resemblance is forced."""
    out = []
    for n in range(1, nmax + 1):
        mp = sp.minimal_polynomial(PHI ** n, x)
        out.append((n, sp.radsimp(sp.expand(PHI ** n)), mp, LUCAS[n],
                    (-1) ** n))
    return out


def is_power_of_phi(val, nmax=12):
    """Whether an algebraic number is phi^n for some small n."""
    for n in range(1, nmax + 1):
        if sp.simplify(val - PHI ** n) == 0:
            return n
    return None


# ---------------------------------------------------------------------------
# 3. the compositum
# ---------------------------------------------------------------------------

def compositum_facts():
    """mu = lambda^2 / nu and the field it actually lives in."""
    mu = sp.radsimp(sp.simplify(F.LAM2 / NU))
    mp = sp.minimal_polynomial(mu, x)
    return dict(mu=mu, minpoly=mp, degree=sp.degree(mp, x),
                lam2_minpoly=sp.minimal_polynomial(F.LAM2, x),
                nu_minpoly=sp.minimal_polynomial(NU, x))


# ---------------------------------------------------------------------------
# 4. the marginal structure
# ---------------------------------------------------------------------------

def marginal_eigenvalues():
    """+1 / -1 in each operator -- the phase grading of spectrepaper."""
    out = []
    for name, poly in (('M', F.CHARPOLY),
                       ('HAT_M', F.HAT_CHARPOLY),
                       ('boundary', sp.expand((x - 1) * TARGET))):
        p = sp.expand(poly)
        out.append((name, sp.simplify(p.subs(x, 1)) == 0,
                    sp.simplify(p.subs(x, -1)) == 0))
    return out


def boundary_components(max_order=6):
    """Boundary edge counts by (species, edge index), order by order."""
    comps = []
    for k in range(max_order + 1):
        tiles = build_patch(k)
        em = edge_table(tiles)
        c = Counter()
        for key, u in em.items():
            if len(u) == 1:
                ti, ei = u[0]
                c[(tiles[ti]['label'], ei)] += 1
        comps.append(c)
    return comps


def component_audit(comps, drop=1):
    """Which components obey P(k) = 5P(k-1) - 3P(k-2) - P(k-3).

    The seed order is dropped: at k = 0 the patch is a single tile and every
    edge is a Delta boundary edge, so those components start at 1 and go to
    zero forever, which no recurrence of this shape describes and which says
    nothing about the operator.
    """
    types = sorted(set().union(*[set(c) for c in comps]))
    obey, fail, period2 = [], [], []
    for t in types:
        s = [comps[k].get(t, 0) for k in range(drop, len(comps))]
        if len(s) < 4:
            continue
        if all(s[i] == 5 * s[i - 1] - 3 * s[i - 2] - s[i - 3]
               for i in range(3, len(s))):
            obey.append((t, s))
        else:
            fail.append((t, s))
            if len(set(s[::2])) == 1 and len(set(s[1::2])) == 1 \
                    and s[0] != s[1]:
                period2.append((t, s))
    return types, obey, fail, period2


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report_search(rows):
    print('\nsearch: is x^2 - 4x - 1 a factor of anything built from M or'
          ' HAT_M?')
    for name, poly, hit in rows:
        print('  %-14s %-52s %s' % (name, poly, 'HIT' if hit else '-'))
    hits = [n for n, _, h in rows if h]
    print('  hits: %s   (%d matrices searched)'
          % (', '.join(hits) if hits else 'NONE', len(rows)))
    return hits


def report_lucas():
    print('\nwhy Q(sqrt(5)) agreement is not evidence')
    print('  phi^n has minimal polynomial x^2 - L_n x + (-1)^n')
    print('  %-4s %-22s %-24s %s' % ('n', 'phi^n', 'minimal polynomial', 'L_n'))
    for n, val, mp, L, sgn in lucas_table():
        print('  %-4d %-22s %-24s %d' % (n, sp.nsimplify(val), mp, L))
    print('  so x^2-4x-1 IS phi^3 and x^2-7x+1 IS phi^4; adjacency is forced')
    n = is_power_of_phi(F.LAM2)
    print('  is the Spectre unit 4+sqrt(15) a power of phi? %s'
          % ('phi^%d' % n if n else 'NO (trace 8 is not a Lucas number)'))


def report_compositum(cf):
    print('\nwhat survives: the exponent needs both fields')
    print('  lambda^2 = 4+sqrt(15)   minpoly %s   in Q(sqrt(15))'
          % cf['lam2_minpoly'])
    print('  nu       = 2+sqrt(5)    minpoly %s   in Q(sqrt(5))'
          % cf['nu_minpoly'])
    print('  mu = lambda^2/nu = %s' % cf['mu'])
    print('    minpoly %s' % cf['minpoly'])
    print('    degree %d -- NOT quadratic, so mu is in neither field'
          % cf['degree'])
    print('    it lives in the compositum Q(sqrt(3), sqrt(5))')


def report_marginal(rows, period2, n_types):
    print('\nthe shared marginal structure')
    print('  %-12s %-10s %s' % ('operator', '+1 eigen', '-1 eigen'))
    for name, p1, m1 in rows:
        print('  %-12s %-10s %s' % (name, 'yes' if p1 else 'no',
                                    'yes' if m1 else 'no'))
    print('  boundary components with period 2 (direct evidence of -1): %d'
          % len(period2))
    for t, s in period2[:4]:
        print('    %-16s %s' % (str(t), s))


def report_gap(types, obey, fail):
    print('\nthe honest gap')
    print('  boundary edge types: %d' % len(types))
    print('  obeying the order-3 recurrence past the transient: %d'
          % len(obey))
    print('  not obeying it: %d' % len(fail))
    interesting = [(t, s) for t, s in fail
                   if len(set(s)) > 2 and max(s) > 100]
    for t, s in interesting[:3]:
        print('    %-16s %s' % (str(t), s))
    print('  so the true boundary operator is LARGER than the 3x3 companion.')
    print('  The total perimeter is governed exactly; the rest of the')
    print('  spectrum is not determined by seven orders of data.')


# ---------------------------------------------------------------------------

def figure(rows, cf, comps, fname):
    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_subplot(2, 3, 1)
    ax.axis('off')
    ax.text(0.0, 0.98, 'The three characteristic polynomials', fontsize=11,
            va='top', weight='bold')
    ax.text(0.0, 0.80,
            '\n'.join([
                r'$M:\ x^5(x-1)(x+1)(x^2-8x+1)$',
                r'$\mathrm{HAT}\_M:\ (x-1)(x+1)(x^2-7x+1)$',
                r'boundary$:\ (x-1)(x^2-4x-1)$',
                '',
                r'traces  $8,\ 7,\ 4$',
                r'norms  $+1,\ +1,\ -1$',
                '',
                r'$x^2-4x-1$ is in neither of the others.',
            ]), fontsize=11, va='top')

    ax = fig.add_subplot(2, 3, 2)
    ns = list(range(1, 7))
    tr = [LUCAS[n] for n in ns]
    ax.plot(ns, tr, 'o-', color='crimson')
    for n, t in zip(ns, tr):
        ax.annotate(r'$\varphi^{%d}$' % n, (n, t), fontsize=9,
                    textcoords='offset points', xytext=(4, 5))
    ax.axhline(4, color='green', ls='--', label=r'boundary $\nu=\varphi^3$')
    ax.axhline(7, color='navy', ls=':', label=r'HAT $\varphi^4$')
    ax.axhline(8, color='0.5', ls='-.', label=r'Spectre trace 8 (not Lucas)')
    ax.set_xlabel('n')
    ax.set_ylabel(r'trace of minpoly of $\varphi^n$ = $L_n$')
    ax.set_title('Lucas deflation', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    for name, poly, lab, col in (
            ('M', F.CHARPOLY, 'M', 'crimson'),
            ('HAT', F.HAT_CHARPOLY, 'HAT_M', 'navy'),
            ('bdy', sp.expand((x - 1) * TARGET), 'boundary', 'green')):
        roots = [complex(sp.N(r)) for r in sp.solve(sp.expand(poly), x)]
        ax.scatter([r.real for r in roots], [r.imag for r in roots],
                   s=55, color=col, label=lab, alpha=0.8)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='0.85', lw=0.8)
    ax.axhline(0, color='0.85', lw=0.8)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('Spectra: shared $\\pm1$, different units', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 4)
    ax.axis('off')
    ax.text(0.0, 0.98, 'The exponent needs both fields', fontsize=11,
            va='top', weight='bold')
    ax.text(0.0, 0.78, '\n'.join([
        r'$\mu = \lambda^2/\nu$',
        r'minpoly  $%s$' % sp.latex(cf['minpoly']),
        r'degree $%d$' % cf['degree'],
        '',
        r'$\mathbb{Q}(\sqrt{15})\ \subset\ \mathbb{Q}(\sqrt{3},\sqrt{5})'
        r'\ \supset\ \mathbb{Q}(\sqrt{5})$',
        '',
        'Spectre and Hat meet in the compositum,',
        'not by one containing the other.',
    ]), fontsize=10.5, va='top')

    ax = fig.add_subplot(2, 3, 5)
    types, obey, fail, p2 = component_audit(comps)
    ax.bar([0, 1, 2], [len(obey), len(fail) - len(p2), len(p2)],
           color=['green', '0.55', 'crimson'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['obeys\norder-3', 'other', 'period-2\n(eigen $-1$)'],
                       fontsize=8)
    ax.set_ylabel('boundary edge types')
    ax.set_title('Component audit: the operator is bigger', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(2, 3, 6)
    ks = range(len(comps))
    for t, s in (p2[:2] + [(t, s) for t, s in fail
                           if max(s) > 100][:2] + obey[:2]):
        ax.semilogy(range(1, 1 + len(s)), [max(v, 0.5) for v in s], 'o-',
                    ms=4, label=str(t), lw=1.2)
    ax.set_xlabel('supertile order k')
    ax.set_ylabel('count (clipped at 0.5)')
    ax.set_title('Sample boundary components', fontsize=10)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.suptitle('Is $\\nu=\\varphi^3$ the Hat, or just the golden ratio?\n'
                 'a negative result, plus the compositum that survives it',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def selftest(max_order=6):
    failures = []

    def check(label, ok):
        print('  %-58s %s' % (label, 'ok' if ok else 'FAIL'))
        if not ok:
            failures.append(label)

    print('the refutation')
    rows = run_search()
    check('at least 20 candidate matrices searched', len(rows) >= 20)
    check('x^2-4x-1 is in NONE of them',
          not any(h for _, _, h in rows))
    check('M charpoly really is x^5(x-1)(x+1)(x^2-8x+1)',
          sp.simplify(sp.expand(F.CHARPOLY)
                      - sp.expand(x ** 5 * (x - 1) * (x + 1)
                                  * (x ** 2 - 8 * x + 1))) == 0)
    check('HAT_M charpoly really is (x-1)(x+1)(x^2-7x+1)',
          sp.simplify(sp.expand(F.HAT_CHARPOLY)
                      - sp.expand((x - 1) * (x + 1)
                                  * (x ** 2 - 7 * x + 1))) == 0)

    print('the Lucas deflation')
    for n in range(1, 7):
        mp = sp.minimal_polynomial(PHI ** n, x)
        want = x ** 2 - LUCAS[n] * x + (-1) ** n
        check('phi^%d minpoly is x^2 - %d x + %d'
              % (n, LUCAS[n], (-1) ** n), sp.simplify(mp - want) == 0)
    check('nu is phi^3', is_power_of_phi(NU) == 3)
    check('HAT_LAM2 is phi^4', is_power_of_phi(F.HAT_LAM2) == 4)
    check('the Spectre unit is NOT a power of phi',
          is_power_of_phi(F.LAM2) is None)

    print('the compositum')
    cf = compositum_facts()
    check('mu has degree 4, not 2', cf['degree'] == 4)
    check('mu minpoly is x^4+32x^3-46x^2-32x+1',
          sp.simplify(cf['minpoly']
                      - (x ** 4 + 32 * x ** 3 - 46 * x ** 2
                         - 32 * x + 1)) == 0)
    check('mu is not in Q(sqrt(15))',
          sp.degree(sp.minimal_polynomial(cf['mu'], x), x) > 2)
    check('mu IS in Q(sqrt(3), sqrt(5))',
          sp.simplify(cf['mu'] - (-8 - 2 * sp.sqrt(15) + 5 * sp.sqrt(3)
                                  + 4 * sp.sqrt(5))) == 0)

    print('the marginal structure')
    mg = dict((n, (a, b)) for n, a, b in marginal_eigenvalues())
    check('M has +1 and -1', mg['M'] == (True, True))
    check('HAT_M has +1 and -1', mg['HAT_M'] == (True, True))
    check('the boundary operator has +1', mg['boundary'][0])

    print('the gap is reported, not hidden')
    comps = boundary_components(max_order)
    types, obey, fail, p2 = component_audit(comps)
    check('some components fail the order-3 recurrence', len(fail) > 0)
    check('period-2 components exist (eigenvalue -1)', len(p2) > 0)
    check('totals still obey it exactly',
          all(sum(comps[k].values())
              == 5 * sum(comps[k - 1].values())
              - 3 * sum(comps[k - 2].values())
              - sum(comps[k - 3].values())
              for k in range(3, len(comps))))

    print()
    if failures:
        print('%d failure(s): %s' % (len(failures), ', '.join(failures)))
    else:
        print('spectre_golden: all checks pass.')
    return len(failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--max-order', type=int, default=6)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if selftest(args.max_order) else 0)

    print('spectre_golden -- testing the phi^3 / phi^4 conjecture')

    rows = run_search()
    hits = report_search(rows)
    report_lucas()
    cf = compositum_facts()
    report_compositum(cf)

    comps = boundary_components(args.max_order)
    types, obey, fail, p2 = component_audit(comps)
    report_marginal(marginal_eigenvalues(), p2, len(types))
    report_gap(types, obey, fail)

    png = os.path.join(OUT, 'spectre_golden.png')
    figure(rows, cf, comps, png)
    print('\nwrote %s' % png)

    print('\nverdict')
    if hits:
        print('  the conjecture SURVIVES: %s' % ', '.join(hits))
    else:
        print('  the conjecture is REFUTED.  x^2-4x-1 is not a factor of any')
        print('  of the %d matrices built from M and HAT_M, and the shared'
              % len(rows))
        print('  field is forced by both quantities being powers of phi.')
        print('  What replaces it: mu = lambda^2/nu has degree 4 and lives in')
        print('  Q(sqrt(3),sqrt(5)), so the Spectre and the Hat meet in the')
        print('  compositum -- a weaker claim than the conjecture, and true.')


if __name__ == '__main__':
    main()
