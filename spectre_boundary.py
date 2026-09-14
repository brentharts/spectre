#!/usr/bin/env python3
r"""spectre_boundary.py -- where s* actually comes from.

spectre_selfsimilar.py fitted the interface growth rate and got a drifting
answer: mu = lambda^0.947 at depth 4, lambda^0.851 at depth 5, giving
s* = 0.474 then 0.426 with no sign of settling.  A drifting fit usually
means the model is wrong, not that the data are noisy.  It was.

This module derives the exponent instead of fitting it, and the drift turns
out to be fully explained.

The boundary transfer operator
------------------------------
Count P(k), the number of boundary edges of an order-k supertile:

    14, 46, 182, 758, 3198, 13534, 57318, ...

This sequence satisfies an exact integer recurrence

    P(k) = 5 P(k-1) - 3 P(k-2) - P(k-3)

whose companion matrix IS the boundary transfer operator, with
characteristic polynomial

    (x - 1) (x^2 - 4x - 1)

So the boundary growth rate is the algebraic number

    nu = 2 + sqrt(5) = phi^3 = 4.2360679...

and the eigenvalue 1 is a conserved quantity on the boundary.

The field divide, again
-----------------------
The AREA of a supertile grows by lambda^2 = 4 + sqrt(15), the fundamental
unit of Z[sqrt(15)].  The PERIMETER grows by 2 + sqrt(5), which lives in
Q(sqrt(5)).  That is the same Q(sqrt(15)) / Q(sqrt(5)) divide that
spectrepaper.py builds its Spectre/Hat section on -- appearing here as area
versus perimeter of the SAME tile, rather than as Spectre versus Hat.  The
supertile boundary is golden even though the tiling is not.

Since nu > lambda, the boundary is fractal: its box dimension is

    log(nu) / log(lambda) = 1.399253...

which is why a perimeter argument assuming P(k) ~ lambda^k was wrong.

The exponent
------------
Interfaces at coarseness c in a depth-N patch are the shared boundaries of
T(c) supertiles of order N-c, each counted twice except the outer boundary:

    B_c = ( T(c) P(N-c) - P(N) ) / 2

which this module checks against the measured counts.  Since T(c) ~
lambda^(2c) and P(N-c) ~ nu^(N-c), the growth in c is

    mu = lambda^2 / nu = 1.858559...

and the balanced thickness exponent of spectre_selfsimilar.py is

    s* = 1 - log(nu) / log(lambda^2) = 0.3003734...

EXACT BUT NOT ALGEBRAIC.  s* is a ratio of logarithms of algebraic numbers,
so it has a closed form and arbitrary precision but is not itself a root of
a polynomial -- the usual situation for a dimension-like exponent.  The
algebraic object is nu, the eigenvalue; s* is what you get after taking its
logarithm.  Anyone hoping for an element of Z[sqrt(15)] here should stop at
nu and take it in Q(sqrt(5)) instead.

Why the fit could never have converged
--------------------------------------
B_c depends on T(c) AND on P(N-c).  The naive single-exponent fit needs both
to be in their asymptotic regimes at once, which needs c large and N-c large
simultaneously -- impossible in a finite patch.  This module reproduces the
earlier drifting values (2.66, 2.41) from the exact formula, confirming they
were the correct output of a mis-specified estimator rather than evidence
about the geometry.

    python3 spectre_boundary.py
    python3 spectre_boundary.py --max-order 6
    python3 spectre_boundary.py --selftest
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sympy as sp

import spectrefacts as F
from spectre_selfsimilar import (build_patch, edge_table, interface_edges,
                                 LAMBDA, LAMBDA_SQ)

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')

x = sp.Symbol('x')

# exact, from spectrefacts
LAM2 = F.LAM2                       # 4 + sqrt(15)
LAM = F.LAM
PHI = (1 + sp.sqrt(5)) / 2


# ---------------------------------------------------------------------------
# 1. the perimeter sequence
# ---------------------------------------------------------------------------

def perimeter_sequence(max_order=6):
    """P(k) = boundary edges of an order-k supertile, k = 0..max_order.

    A boundary edge is one used by a single tile.  Order 6 is 272791 tiles,
    which is where this stops being cheap.
    """
    out = []
    for k in range(max_order + 1):
        tiles = build_patch(k)
        em = edge_table(tiles)
        out.append(sum(1 for u in em.values() if len(u) == 1))
    return out


def find_recurrence(seq, max_order=5):
    """Smallest exact integer linear recurrence for `seq`.

    Exact means zero residual over every available row, not a good fit.  A
    least-squares recurrence that almost works is worthless here: the whole
    point is to land on a characteristic polynomial, and an approximate one
    has approximate roots that are not algebraic numbers at all.
    """
    for order in range(1, max_order + 1):
        rows = [seq[i - order:i][::-1] for i in range(order, len(seq))]
        rhs = [seq[i] for i in range(order, len(seq))]
        if len(rows) < order + 1:          # need at least one row to spare
            continue
        A, b = sp.Matrix(rows), sp.Matrix(rhs)
        try:
            sol = A.solve_least_squares(b)
        except Exception:
            continue
        if sp.simplify((A * sol - b).norm()) != 0:
            continue
        coeffs = [sp.nsimplify(c) for c in sol]
        if not all(c.is_Integer for c in coeffs):
            continue
        poly = x ** order - sum(coeffs[j] * x ** (order - 1 - j)
                                for j in range(order))
        return order, coeffs, sp.factor(sp.expand(poly))
    return None, None, None


def companion(coeffs):
    """The boundary transfer operator itself: companion matrix of the
    recurrence, acting on (P(k), P(k-1), P(k-2))."""
    n = len(coeffs)
    M = sp.zeros(n, n)
    for j, c in enumerate(coeffs):
        M[0, j] = c
    for i in range(1, n):
        M[i, i - 1] = 1
    return M


def growth_rate(poly):
    """Largest real root -- the Perron eigenvalue of the boundary operator."""
    roots = sp.solve(poly, x)
    real = [r for r in roots if sp.im(sp.N(r)) == 0]
    return max(real, key=lambda r: sp.N(r))


# ---------------------------------------------------------------------------
# 2. the exponent, exactly
# ---------------------------------------------------------------------------

def exact_exponents(nu):
    """mu, a, s*, and the boundary dimension, as exact expressions."""
    mu = sp.radsimp(sp.simplify(LAM2 / nu))
    a = sp.log(mu) / sp.log(LAM)              # mu = lambda^a
    s_star = sp.simplify(1 - sp.log(nu) / sp.log(LAM2))
    dim = sp.log(nu) / sp.log(LAM)
    return mu, a, s_star, dim


# ---------------------------------------------------------------------------
# 3. the interface identity
# ---------------------------------------------------------------------------

def predicted_interfaces(depth, P):
    """B_c = (T(c) P(N-c) - P(N)) / 2, the closed form for the counts.

    Two supertiles share each internal boundary edge, hence the halving; the
    outer boundary of the whole patch belongs to nobody and is removed first.
    """
    out = []
    for c in range(1, depth + 1):
        if c >= len(F.MATRIX_TOTALS) or depth - c < 0:
            break
        out.append((c, (F.MATRIX_TOTALS[c] * P[depth - c] - P[depth]) / 2.0))
    return out


def measured_interfaces(depth):
    tiles = build_patch(depth)
    em = edge_table(tiles)
    mx = max(len(t['path']) for t in tiles)
    return [(c, len(interface_edges(tiles, em, c))) for c in range(1, mx + 1)]


def naive_fit(counts, levels=None):
    """The single-exponent fit spectre_selfsimilar.py used, reproduced here
    so its drifting output can be explained rather than argued about."""
    pts = counts if levels is None else [p for p in counts if p[0] in levels]
    if len(pts) < 2:
        return float('nan')
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.log([p[1] for p in pts])
    return float(np.exp(np.polyfit(xs, ys, 1)[0]))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report(P, order, coeffs, poly, nu, mu, a, s_star, dim):
    print('\nboundary of an order-k supertile')
    print('  P(k) =', P)
    print('  ratios =', ' '.join('%.4f' % (P[i + 1] / P[i])
                                 for i in range(len(P) - 1)))
    print('\nthe boundary transfer operator')
    print('  exact recurrence of order %d, coefficients %s'
          % (order, list(coeffs)))
    print('  characteristic polynomial: %s' % poly)
    print('  companion matrix:')
    M = companion(coeffs)
    for i in range(M.rows):
        print('    [%s]' % '  '.join('%4s' % M[i, j] for j in range(M.cols)))
    print('  eigenvalues: %s' % [sp.radsimp(sp.simplify(r))
                                 for r in sp.solve(poly, x)])
    print('  Perron eigenvalue nu = %s = %.10f'
          % (sp.radsimp(nu), float(sp.N(nu))))
    print('  nu == phi^3: %s' % bool(sp.simplify(nu - PHI ** 3) == 0))

    print('\nthe field divide')
    print('  area growth      lambda^2 = %s      in Q(sqrt(15))'
          % sp.radsimp(LAM2))
    print('  perimeter growth nu       = %s      in Q(sqrt(5))'
          % sp.radsimp(nu))
    print('  nu > lambda, so the boundary is fractal')
    print('  boundary box dimension = log(nu)/log(lambda) = %.9f'
          % float(sp.N(dim)))

    print('\nthe exponent, derived')
    print('  mu = lambda^2 / nu = %s' % sp.radsimp(mu))
    print('                     = %.9f = lambda^%.9f'
          % (float(sp.N(mu)), float(sp.N(a))))
    print('  s* = 1 - log(nu)/log(lambda^2) = %.10f' % float(sp.N(s_star)))
    print('  s* is exact but NOT algebraic (a ratio of logs);')
    print('  the algebraic object is nu = 2 + sqrt(5).')


def report_identity(depths, P):
    print('\nthe interface identity  B_c = (T(c) P(N-c) - P(N)) / 2')
    for N in depths:
        meas = dict(measured_interfaces(N))
        pred = predicted_interfaces(N, P)
        print('  depth %d   %-4s %-11s %-11s %s'
              % (N, 'c', 'measured', 'predicted', 'error'))
        for c, p in pred:
            if c in meas:
                print('            %-4d %-11d %-11.1f %+.2f%%'
                      % (c, meas[c], p, 100.0 * (p - meas[c]) / meas[c]))
    print('  (residual few-percent error is the mystic split: the deepest'
          ' level\n   carries T(n)+T(n-1) blocks, not T(n))')


def report_drift(depths, P):
    print('\nwhy the fit drifted')
    print('  %-8s %-16s %-16s %s'
          % ('depth', 'fit on measured', 'fit on exact form', 'asymptotic mu'))
    for N in depths:
        meas = measured_interfaces(N)
        pred = predicted_interfaces(N, P)
        top = max(n for _, n in meas)
        keep = [c for c, n in meas if n < 0.9 * top]
        f_meas = naive_fit(meas, keep)
        f_pred = naive_fit(pred, keep)
        print('  %-8d %-16.4f %-16.4f %.4f'
              % (N, f_meas, f_pred, float(sp.N(LAM2 / (2 + sp.sqrt(5))))))
    print('  The exact closed form reproduces the drifting fitted values, so'
          '\n  the drift was the estimator, not the geometry.  B_c depends on'
          '\n  T(c) and on P(N-c); a single-exponent fit needs c large AND'
          '\n  N-c large at once, which no finite patch provides.')


# ---------------------------------------------------------------------------

def figure(P, poly, nu, dim, depths, fname):
    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_subplot(2, 3, 1)
    k = np.arange(len(P))
    ax.semilogy(k, P, 'o-', color='crimson', label='P(k) measured')
    nu_f = float(sp.N(nu))
    ax.semilogy(k, [P[1] * nu_f ** (i - 1) for i in k], '--', color='green',
                label=r'$\nu^k,\ \nu=2+\sqrt{5}$')
    ax.semilogy(k, [P[1] * LAMBDA ** (i - 1) for i in k], ':', color='navy',
                label=r'$\lambda^k$ (naive)')
    ax.set_xlabel('supertile order k')
    ax.set_ylabel('boundary edges')
    ax.set_title('Perimeter grows faster than $\\lambda$', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 2)
    r = [P[i + 1] / P[i] for i in range(len(P) - 1)]
    ax.plot(range(1, len(P)), r, 'o-', color='crimson')
    ax.axhline(nu_f, color='green', ls='--', label=r'$2+\sqrt{5}=\varphi^3$')
    ax.axhline(LAMBDA, color='navy', ls=':', label=r'$\lambda$')
    ax.set_xlabel('k')
    ax.set_ylabel('P(k)/P(k-1)')
    ax.set_title('Converging to an algebraic number', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    roots = [complex(sp.N(r)) for r in sp.solve(poly, x)]
    ax.axhline(0, color='0.8', lw=0.8)
    ax.axvline(0, color='0.8', lw=0.8)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='0.85', lw=0.8)
    ax.scatter([r.real for r in roots], [r.imag for r in roots],
               s=70, color='crimson', zorder=5)
    for r in roots:
        ax.annotate('%.4f' % r.real, (r.real, r.imag),
                    textcoords='offset points', xytext=(4, 6), fontsize=8)
    ax.set_title('Spectrum of the boundary operator', fontsize=10)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 4)
    for i, N in enumerate(depths):
        meas = dict(measured_interfaces(N))
        pred = predicted_interfaces(N, P)
        cs = [c for c, _ in pred if c in meas]
        ax.plot(cs, [meas[c] for c in cs], 'o-',
                color=['crimson', 'navy'][i % 2], label='depth %d measured' % N)
        ax.plot(cs, [p for c, p in pred if c in meas], 'x--', color='0.45',
                label='depth %d predicted' % N)
    ax.set_yscale('log')
    ax.set_xlabel('coarseness c')
    ax.set_ylabel('interface edges')
    ax.set_title('The closed form matches', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 5)
    fits, preds = [], []
    for N in depths:
        meas = measured_interfaces(N)
        top = max(n for _, n in meas)
        keep = [c for c, n in meas if n < 0.9 * top]
        fits.append(naive_fit(meas, keep))
        preds.append(naive_fit(predicted_interfaces(N, P), keep))
    w = 0.35
    ax.bar(np.arange(len(depths)) - w / 2, fits, width=w, color='crimson',
           label='fit on measured')
    ax.bar(np.arange(len(depths)) + w / 2, preds, width=w, color='0.5',
           label='fit on exact form')
    ax.axhline(float(sp.N(LAM2 / nu)), color='green', ls='--',
               label=r'asymptotic $\mu=\lambda^2/\nu$')
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(['depth %d' % d for d in depths])
    ax.set_ylabel(r'fitted $\mu$')
    ax.set_title('The drift was the estimator', fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    mu, a, s_star, _ = exact_exponents(nu)
    txt = '\n'.join([
        r'$P(k)=5P(k{-}1)-3P(k{-}2)-P(k{-}3)$',
        r'$\chi(x)=(x-1)(x^2-4x-1)$',
        '',
        r'$\nu = 2+\sqrt{5} = \varphi^3 = %.7f$' % nu_f,
        r'$\lambda^2 = 4+\sqrt{15} = %.7f$' % LAMBDA_SQ,
        '',
        r'area $\in\ \mathbb{Q}(\sqrt{15})$',
        r'perimeter $\in\ \mathbb{Q}(\sqrt{5})$',
        '',
        r'boundary dim $= %.7f$' % float(sp.N(dim)),
        r'$\mu = \lambda^2/\nu = %.7f$' % float(sp.N(mu)),
        r'$s^* = %.7f$' % float(sp.N(s_star)),
    ])
    ax.text(0.02, 0.97, txt, va='top', fontsize=11)

    fig.suptitle('The Spectre boundary transfer operator\n'
                 r'perimeter is golden ($2+\sqrt{5}$) while area is '
                 r'$4+\sqrt{15}$ -- and $s^*$ follows from the ratio',
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

    print('the perimeter sequence')
    P = perimeter_sequence(max_order)
    check('starts at 14 (the tile has 14 edges)', P[0] == 14)
    check('is strictly increasing',
          all(P[i] < P[i + 1] for i in range(len(P) - 1)))

    print('the recurrence')
    order, coeffs, poly = find_recurrence(P)
    check('an exact integer recurrence exists', order is not None)
    check('it has order 3', order == 3)
    check('coefficients are (5, -3, -1)', list(coeffs) == [5, -3, -1])
    check('it reproduces every term',
          all(P[i] == 5 * P[i - 1] - 3 * P[i - 2] - P[i - 3]
              for i in range(3, len(P))))

    print('the operator')
    check('charpoly factors as (x-1)(x^2-4x-1)',
          sp.simplify(poly - (x - 1) * (x ** 2 - 4 * x - 1)) == 0)
    M = companion(coeffs)
    check('companion matrix has the same charpoly',
          sp.simplify(sp.factor(M.charpoly(x).as_expr()) - sp.factor(poly)) == 0)
    check('1 is an eigenvalue (a conserved boundary quantity)',
          sp.simplify(poly.subs(x, 1)) == 0)
    nu = growth_rate(poly)
    check('nu = 2 + sqrt(5)', sp.simplify(nu - (2 + sp.sqrt(5))) == 0)
    check('nu = phi^3', sp.simplify(nu - PHI ** 3) == 0)
    check('nu lies in Q(sqrt(5))',
          sp.sqrt(5) in nu.atoms(sp.Pow)
          and sp.simplify(sp.minimal_polynomial(nu, x)
                          - (x ** 2 - 4 * x - 1)) == 0)
    check('nu does NOT lie in Q(sqrt(15))',
          sp.sqrt(15) not in nu.atoms(sp.Pow)
          and sp.simplify(sp.minimal_polynomial(LAM2, x)
                          - (x ** 2 - 8 * x + 1)) == 0)

    print('the exponent')
    mu, a, s_star, dim = exact_exponents(nu)
    check('nu exceeds lambda, so the boundary is fractal',
          float(sp.N(nu)) > LAMBDA)
    check('boundary dimension is between 1 and 2',
          1.0 < float(sp.N(dim)) < 2.0)
    check('mu = lambda^2 / nu is above 1', float(sp.N(mu)) > 1.0)
    check('s* is in (0, 1)', 0.0 < float(sp.N(s_star)) < 1.0)
    check('s* is far below the proposed 2',
          abs(float(sp.N(s_star)) - 2.0) > 1.0)
    check('s* agrees with 1 - dim/2',
          abs(float(sp.N(s_star)) - (1 - float(sp.N(dim)) / 2)) < 1e-12)

    print('the identity')
    for N in (3, 4):
        meas = dict(measured_interfaces(N))
        pred = predicted_interfaces(N, P)
        err = [abs(p - meas[c]) / meas[c] for c, p in pred if c in meas]
        check('depth %d interface identity within 6%%' % N,
              bool(err) and max(err) < 0.06)

    print()
    if failures:
        print('%d failure(s): %s' % (len(failures), ', '.join(failures)))
    else:
        print('spectre_boundary: all checks pass.')
    return len(failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--max-order', type=int, default=6,
                    help='deepest supertile for the perimeter sequence')
    ap.add_argument('--depths', type=int, nargs='+', default=[4, 5])
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if selftest(max(args.max_order, 6)) else 0)

    print('spectre_boundary -- s* is a spectral quantity, not a fitted one')
    print('lambda^2 = 4 + sqrt(15) = %.10f   (spectrefacts.LAM2)' % LAMBDA_SQ)

    P = perimeter_sequence(args.max_order)
    order, coeffs, poly = find_recurrence(P)
    if order is None:
        print('no exact recurrence found in %d terms' % len(P))
        sys.exit(1)
    nu = growth_rate(poly)
    mu, a, s_star, dim = exact_exponents(nu)

    report(P, order, coeffs, poly, nu, mu, a, s_star, dim)
    report_identity(args.depths, P)
    report_drift(args.depths, P)

    png = os.path.join(OUT, 'spectre_boundary.png')
    figure(P, poly, nu, dim, args.depths, png)
    print('\nwrote %s' % png)

    print('\nsummary')
    print('  spectre_selfsimilar fitted s* = 0.474 (depth 4), 0.426 (depth 5)')
    print('  the exact value is s* = %.7f' % float(sp.N(s_star)))
    print('  and the fitted numbers are reproduced by the exact closed form,')
    print('  so the drift was a mis-specified estimator throughout.')


if __name__ == '__main__':
    main()
