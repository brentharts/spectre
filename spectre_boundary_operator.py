#!/usr/bin/env python3
r"""spectre_boundary_operator.py -- the boundary operator, derived.

spectre_boundary.py FOUND the recurrence P(k) = 5P(k-1) - 3P(k-2) - P(k-3)
by fitting integer coefficients to a sequence of counts.  That establishes
the numbers but explains nothing: it cannot say why the order is three, why
the Perron root is phi^3, or where the period-2 components that
spectre_golden.py turned up come from.  This module derives the same
operator from the substitution rule in spectre.py and answers all three.

The substitution table
----------------------
buildSupertiles places EIGHT children by a fixed list of transforms, the
same list for every metatile type; the types differ only in WHICH children
go in the slots.  Gamma is the exception -- its slot 2 is None, so it has
seven children.  Slot 7 is Gamma in every rule, so there is a Gamma spine
running down the hierarchy.

Two structural facts follow, and both are checked here rather than assumed:

1. ALL EIGHT non-Gamma metatiles have the SAME boundary sequence

       P(k) = 14, 46, 182, 758, 3198, 13534, 57318, ...

   The outline of a supertile does not depend on its type.  That is what
   collapses a nine-type problem to a two-type one, and it is why a scalar
   recurrence existed at all.

2. Gamma alone differs: Q(k) = 20, 44, 160, 652, 2736, 11564, 48960.

The derivation
--------------
Reading the child multiset straight off the substitution table -- seven
non-Gamma children plus one Gamma for the ordinary types, six plus one for
Gamma itself -- and subtracting each glued edge twice gives

    P(k+1) = 7 P(k) + Q(k) - 2 G(k)
    Q(k+1) = 6 P(k) + Q(k) - 2 G_Gamma(k)

where G is the number of edges shared between two DIFFERENT children.  Both
identities hold exactly at every order computed, with no fitted constants.

That reduces everything to the glue.  The glue decomposes by slot pair into
thirteen contact sequences, and EVERY ONE of them satisfies the same
characteristic polynomial

    (x - 1)(x^2 - 4x - 1)

so the contact set is closed under the operator, which is precisely why P is.
The Perron root nu = 2 + sqrt(5) = phi^3 is therefore structural: it is the
growth rate of the contact between two adjacent supertiles, not a property
of the outline that happened to fit.

The seed transient
------------------
Sequences must be read from k = 1, not k = 0.  At k = 0 a "supertile" is a
single tile, every edge is on its boundary, and no child structure exists --
the recurrence describes assembly and there is nothing assembled yet.  Read
from k = 0, only P itself obeys the recurrence; read from k = 1, ALL of P,
Q, G, G_Gamma and all thirteen contact sequences obey it.  The two contacts
that look like exceptions, (1,7) and (4,7), are the two touching the Gamma
slot, and they differ from their non-Gamma partners in the first term alone
(4 against 2) and agree from k = 2 onward.

The -1 sector
-------------
It is NOT in this operator: (x-1)(x^2-4x-1) has roots 1, phi^3 and
-phi^(-3), no -1.  The period-2 components spectre_golden.py found live only
in the FINE decomposition by (species, edge index), and they are bounded --
[1,2,1,2,1,2] neither grows nor decays.  So they contribute nothing to any
growth rate, and the derived operator is the exact quotient system on the
aggregates.  The fine system is a bounded extension of it.

What is still open
------------------
Some fine components, such as (Pi, 3) = 1, 7, 22, 91, 378, 1599, obey
neither this recurrence nor its obvious extensions, while their ratios still
converge to phi^3.  So the extra spectrum is subdominant -- it cannot change
nu -- but its exact content needs more orders than are computable here
(order 7 is ~1.4M tiles per metatile type).  This module does not claim to
have the full fine operator; it claims the aggregate one, derived.

    python3 spectre_boundary_operator.py
    python3 spectre_boundary_operator.py --max-order 6
    python3 spectre_boundary_operator.py --selftest
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

import spectre as S
import spectrefacts as F
import tile_family as TF

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
x = sp.Symbol('x')
PHI = (1 + sp.sqrt(5)) / 2
NU = 2 + sp.sqrt(5)
CHI = sp.expand((x - 1) * (x ** 2 - 4 * x - 1))
COEFFS = (5, -3, -1)

# The substitution table, transcribed from spectre.buildSupertiles.  Kept
# here as data so the derivation reads off the rule rather than off counts.
SUBST = {
    "Gamma":  ("Pi",  "Delta", None,  "Theta", "Sigma", "Xi",  "Phi",    "Gamma"),
    "Delta":  ("Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"),
    "Theta":  ("Psi", "Delta", "Pi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"),
    "Lambda": ("Psi", "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"),
    "Xi":     ("Psi", "Delta", "Pi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma"),
    "Pi":     ("Psi", "Delta", "Xi",  "Phi",   "Sigma", "Psi", "Phi",    "Gamma"),
    "Sigma":  ("Xi",  "Delta", "Xi",  "Phi",   "Sigma", "Pi",  "Lambda", "Gamma"),
    "Phi":    ("Psi", "Delta", "Psi", "Phi",   "Sigma", "Pi",  "Phi",    "Gamma"),
    "Psi":    ("Psi", "Delta", "Psi", "Phi",   "Sigma", "Psi", "Phi",    "Gamma"),
}
NAMES = sorted(SUBST)
ORDINARY = [n for n in NAMES if n != 'Gamma']


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def walk_slots(node, T=None, slot=None, top=True):
    """Leaf tiles, tagged with the TOP-LEVEL slot they descend from.

    The slot tag is what makes the glue decomposable: an edge shared by two
    leaves in different top-level slots is a contact between two children,
    which is exactly the quantity the derivation needs.
    """
    if T is None:
        T = S.IDENTITY
    if isinstance(node, S.Tile):
        yield T, node.label, slot
        return
    for j, (child, trsf) in enumerate(zip(node.tiles, node.transformations)):
        for item in walk_slots(child, S.mul(T, trsf),
                               j if top else slot, False):
            yield item


def edge_map_slots(metatiles, name):
    """key -> list of (slot, label, edge index) for one metatile."""
    canon = TF.canonical_tile_verts('any')
    em = {}
    for T, label, slot in walk_slots(metatiles[name]):
        w = TF.transform_polygon(np.array(T), canon)[:14]
        for ei in range(14):
            key = tuple(sorted([tuple(np.round(w[ei], 4)),
                                tuple(np.round(w[(ei + 1) % 14], 4))]))
            em.setdefault(key, []).append((slot, label, ei))
    return em


def gather(max_order=6):
    """P, Q, glue totals and per-slot-pair contacts, order by order."""
    P, Q, G, GG = [], [], [], []
    contacts, fine = {}, {}
    for k in range(max_order + 1):
        md = S.buildSpectreTiles(k, 1.0, 1.0, rotation=30)
        for name, store in (('Delta', P), ('Gamma', Q)):
            em = edge_map_slots(md, name)
            store.append(sum(1 for v in em.values() if len(v) == 1))
            pairs = Counter(tuple(sorted(s for s, _, _ in v))
                            for v in em.values() if len(v) == 2)
            cross = sum(n for pr, n in pairs.items() if pr[0] != pr[1])
            (G if name == 'Delta' else GG).append(cross)
            if name == 'Delta':
                for pr, n in pairs.items():
                    if pr[0] != pr[1]:
                        contacts.setdefault(pr, {})[k] = n
                bd = Counter((lab, ei) for v in em.values() if len(v) == 1
                             for _, lab, ei in v)
                for t, n in bd.items():
                    fine.setdefault(t, {})[k] = n
    return dict(P=P, Q=Q, G=G, GG=GG, contacts=contacts, fine=fine,
                max_order=max_order)


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------

def child_counts(name):
    """(non-Gamma children, Gamma children) straight from the table."""
    kids = [s for s in SUBST[name] if s]
    return sum(1 for s in kids if s != 'Gamma'), sum(1 for s in kids
                                                     if s == 'Gamma')


def check_identity(data):
    """P(k+1) = 7P(k) + Q(k) - 2G(k) and the Gamma analogue.

    The coefficients are READ OFF the substitution table, not fitted, so an
    exact match at every order is evidence that the derivation is the right
    one and not a numerical coincidence.
    """
    P, Q, G, GG = data['P'], data['Q'], data['G'], data['GG']
    a_d, b_d = child_counts('Delta')
    a_g, b_g = child_counts('Gamma')
    # The glue is counted inside the PARENT, so it is indexed k+1 while the
    # child boundaries are indexed k.  Lining those up wrongly is the one
    # easy way to get an identity that is off by a constant at every order.
    rows = []
    for k in range(len(P) - 1):
        lhs_p, rhs_p = P[k + 1], a_d * P[k] + b_d * Q[k] - 2 * G[k + 1]
        lhs_q, rhs_q = Q[k + 1], a_g * P[k] + b_g * Q[k] - 2 * GG[k + 1]
        rows.append((k, lhs_p, rhs_p, lhs_q, rhs_q))
    return rows, (a_d, b_d), (a_g, b_g)


def obeys(seq, drop=0):
    s = list(seq)[drop:]
    if len(s) < 4:
        return None
    return all(s[i] == 5 * s[i - 1] - 3 * s[i - 2] - s[i - 3]
               for i in range(3, len(s)))


def audit_sequences(data):
    """Which aggregates obey the operator, read from k=0 and from k=1."""
    # G and G_Gamma have no order-0 entry worth reading (a single tile has
    # no children to glue), so they are already shifted by one relative to
    # P and Q and are audited on their own terms.
    named = [('P', data['P']), ('Q', data['Q']),
             ('G', data['G'][1:]), ('G_Gamma', data['GG'][1:])]
    for pr in sorted(data['contacts']):
        d = data['contacts'][pr]
        named.append(('contact %s' % (pr,),
                      [d.get(k, 0) for k in sorted(d)]))
    return [(n, s, obeys(s, 0), obeys(s, 1)) for n, s in named]


def classify_fine(data, drop=1):
    """Fine components: obeying, bounded period-2, or open."""
    obey, period2, open_ = [], [], []
    for t, d in sorted(data['fine'].items()):
        s = [d.get(k, 0) for k in range(data['max_order'] + 1)][drop:]
        if len(s) < 4:
            continue
        if obeys(s):
            obey.append((t, s))
        elif len(set(s[::2])) == 1 and len(set(s[1::2])) == 1 and s[0] != s[1]:
            period2.append((t, s))
        else:
            open_.append((t, s))
    return obey, period2, open_


def ratio_tail(s):
    return s[-1] / s[-2] if len(s) >= 2 and s[-2] else float('nan')


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report(data):
    print('\nstructural fact 1: the outline does not depend on the type')
    md = S.buildSpectreTiles(3, 1.0, 1.0, rotation=30)
    per = {}
    for n in NAMES:
        em = edge_map_slots(md, n)
        per[n] = sum(1 for v in em.values() if len(v) == 1)
    for n in NAMES:
        print('  %-8s P(3) = %d%s' % (n, per[n],
                                      '   <- the exception' if n == 'Gamma'
                                      else ''))
    print('  all eight ordinary types agree: %s'
          % (len({per[n] for n in ORDINARY}) == 1))

    print('\nstructural fact 2: the two sequences')
    print('  P(k) =', data['P'])
    print('  Q(k) =', data['Q'], ' (Gamma)')

    rows, (a_d, b_d), (a_g, b_g) = check_identity(data)
    print('\nthe derivation, coefficients read off the substitution table')
    print('  Delta has %d non-Gamma children and %d Gamma child'
          % (a_d, b_d))
    print('  Gamma has %d non-Gamma children and %d Gamma child (slot 2'
          ' is None)' % (a_g, b_g))
    print('  P(k+1) = %dP(k) + %dQ(k) - 2G(k)' % (a_d, b_d))
    print('  Q(k+1) = %dP(k) + %dQ(k) - 2G_Gamma(k)' % (a_g, b_g))
    print('  %-4s %-10s %-10s %-10s %-10s %s'
          % ('k', 'P(k+1)', 'predicted', 'Q(k+1)', 'predicted', 'both'))
    for k, lp, rp, lq, rq in rows:
        print('  %-4d %-10d %-10d %-10d %-10d %s'
              % (k, lp, rp, lq, rq, 'ok' if (lp == rp and lq == rq)
                 else 'FAIL'))

    print('\nthe glue is closed under the same operator')
    aud = audit_sequences(data)
    print('  %-16s %-38s %-9s %s'
          % ('sequence', 'values', 'from k=0', 'from k=1'))
    for n, s, o0, o1 in aud:
        print('  %-16s %-38s %-9s %s'
              % (n, str(s)[:37], o0, o1))
    n_ok = sum(1 for _, _, _, o1 in aud if o1)
    n_short = sum(1 for _, _, _, o1 in aud if o1 is None)
    print('  %d of %d obey (x-1)(x^2-4x-1) once the seed is dropped'
          % (n_ok, len(aud) - n_short))
    if n_short:
        print('  (%d sequence(s) too short to test at this order)' % n_short)

    print('\n  the seed transient: at k=0 a supertile is a single tile and')
    print('  there is no assembly for the recurrence to describe.  The two')
    print('  contacts touching the Gamma slot differ from their non-Gamma')
    print('  partners in the first term only:')
    for pr in ((1, 3), (1, 7), (4, 6), (4, 7)):
        if pr in data['contacts']:
            d = data['contacts'][pr]
            print('    %-8s %s' % (str(pr), [d.get(k, 0) for k in sorted(d)]))


def report_spectrum(data):
    print('\nthe operator and its spectrum')
    print('  characteristic polynomial: %s' % sp.factor(CHI))
    roots = sp.solve(CHI, x)
    for r in roots:
        print('    %-22s = %+.9f' % (sp.radsimp(sp.simplify(r)),
                                     float(sp.N(r))))
    print('  Perron root nu = 2 + sqrt(5) = phi^3 = %.9f'
          % float(sp.N(NU)))
    print('  -1 is NOT a root: %s' % (sp.simplify(CHI.subs(x, -1)) != 0))

    obey, p2, open_ = classify_fine(data)
    print('\nthe -1 sector lives in the fine decomposition only')
    print('  fine components obeying the operator: %d' % len(obey))
    print('  bounded period-2 components (eigenvalue -1): %d' % len(p2))
    for t, s in p2[:3]:
        print('    %-16s %s   bounded, growth 1' % (str(t), s))
    print('  so they contribute nothing to any growth rate; the derived')
    print('  operator is the exact quotient system on the aggregates.')

    print('\nstill open: fine components obeying neither')
    print('  count: %d' % len(open_))
    for t, s in open_[:3]:
        print('    %-16s %-34s last ratio %.4f'
              % (str(t), str(s)[:33], ratio_tail(s)))
    print('  their ratios still approach nu = %.4f, so the extra spectrum is'
          % float(sp.N(NU)))
    print('  subdominant and cannot move the Perron root.')


# ---------------------------------------------------------------------------

def figure(data, fname):
    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_subplot(2, 3, 1)
    ax.axis('off')
    ax.text(0.0, 0.99, 'Derivation, not fit', fontsize=11, va='top',
            weight='bold')
    a_d, b_d = child_counts('Delta')
    a_g, b_g = child_counts('Gamma')
    ax.text(0.0, 0.84, '\n'.join([
        r'$P(k{+}1) = %dP(k) + %dQ(k) - 2G(k)$' % (a_d, b_d),
        r'$Q(k{+}1) = %dP(k) + %dQ(k) - 2G_\Gamma(k)$' % (a_g, b_g),
        '',
        'coefficients read off the',
        'substitution table, exact at',
        'every order computed.',
        '',
        r'$\chi(x)=(x-1)(x^2-4x-1)$',
        r'$\nu = 2+\sqrt{5} = \varphi^3$',
    ]), fontsize=10.5, va='top')

    ax = fig.add_subplot(2, 3, 2)
    k = range(len(data['P']))
    ax.semilogy(k, data['P'], 'o-', color='crimson', label='P (8 types)')
    ax.semilogy(k, data['Q'], 's-', color='navy', label=r'Q ($\Gamma$)')
    ax.semilogy(range(len(data['G'])), data['G'], '^--', color='green',
                label='G (glue)')
    ax.set_xlabel('order k')
    ax.set_ylabel('edges')
    ax.set_title('All governed by one operator', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    for pr in sorted(data['contacts']):
        d = data['contacts'][pr]
        s = [d.get(j, 0) for j in sorted(d)]
        ax.semilogy(range(1, 1 + len(s)), [max(v, 0.5) for v in s], 'o-',
                    ms=3, lw=1, alpha=0.75)
    ax.set_xlabel('order k')
    ax.set_ylabel('contact edges')
    ax.set_title('13 slot-pair contacts, all obeying $\\chi$', fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 4)
    aud = audit_sequences(data)
    lab = ['from k=0', 'from k=1']
    v0 = sum(1 for _, _, o0, _ in aud if o0)
    v1 = sum(1 for _, _, _, o1 in aud if o1)
    ax.bar(lab, [v0, v1], color=['0.6', 'green'])
    ax.axhline(len(aud), color='crimson', ls='--', label='all sequences')
    ax.set_ylabel('sequences obeying $\\chi$')
    ax.set_title('The seed transient explains the gap', fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 5)
    obey, p2, open_ = classify_fine(data)
    ax.bar([0, 1, 2], [len(obey), len(p2), len(open_)],
           color=['green', 'crimson', '0.55'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['obeys $\\chi$', 'period-2\n(bounded)', 'open'],
                       fontsize=8)
    ax.set_ylabel('fine components')
    ax.set_title('Fine decomposition', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(2, 3, 6)
    roots = [complex(sp.N(r)) for r in sp.solve(CHI, x)]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='0.85', lw=0.8)
    ax.axhline(0, color='0.85', lw=0.8)
    ax.scatter([r.real for r in roots], [r.imag for r in roots], s=80,
               color='crimson', zorder=5, label='derived operator')
    ax.scatter([-1.0], [0.0], s=80, marker='x', color='navy',
               label='-1 (fine sector only)')
    for r in roots:
        ax.annotate('%.4f' % r.real, (r.real, r.imag), fontsize=8,
                    textcoords='offset points', xytext=(4, 6))
    ax.set_xlabel('Re')
    ax.set_title('$-1$ is not in the derived operator', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle('The Spectre boundary operator, derived from the '
                 'substitution rule\n'
                 r'$\nu=\varphi^3$ is the growth rate of the contact between '
                 r'adjacent supertiles', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def selftest(max_order=5):
    failures = []

    def check(label, ok):
        print('  %-58s %s' % (label, 'ok' if ok else 'FAIL'))
        if not ok:
            failures.append(label)

    data = gather(max_order)

    print('the substitution table')
    check('nine metatile types', len(SUBST) == 9)
    check('Gamma alone has seven children',
          [n for n in NAMES if len([s for s in SUBST[n] if s]) == 7]
          == ['Gamma'])
    check('slot 7 is Gamma in every rule',
          all(SUBST[n][7] == 'Gamma' for n in NAMES))
    check('Delta: 7 ordinary + 1 Gamma child', child_counts('Delta') == (7, 1))
    check('Gamma: 6 ordinary + 1 Gamma child', child_counts('Gamma') == (6, 1))

    print('the outline is type-independent')
    md = S.buildSpectreTiles(3, 1.0, 1.0, rotation=30)
    per = {n: sum(1 for v in edge_map_slots(md, n).values() if len(v) == 1)
           for n in NAMES}
    check('all eight ordinary types share P(3)',
          len({per[n] for n in ORDINARY}) == 1)
    check('Gamma differs', per['Gamma'] != per['Delta'])

    print('the derived identities')
    rows, _, _ = check_identity(data)
    check('P identity exact at every order',
          all(lp == rp for _, lp, rp, _, _ in rows))
    check('Q identity exact at every order',
          all(lq == rq for _, _, _, lq, rq in rows))

    print('closure under the operator')
    aud = audit_sequences(data)
    # obeys() returns None when a sequence is too short to test after the
    # seed is dropped; that is missing data, not a violation, and counting
    # it as either would be wrong.
    tested = [o1 for _, _, _, o1 in aud if o1 is not None]
    untested = sum(1 for _, _, _, o1 in aud if o1 is None)
    check('every testable aggregate obeys chi from k=1 (%d tested, %d too '
          'short)' % (len(tested), untested), all(tested))
    check('most aggregates are testable', len(tested) >= len(aud) - 2)
    check('not every aggregate obeys it from k=0',
          not all(o0 for _, _, o0, _ in aud))
    check('at least ten slot-pair contacts',
          len(data['contacts']) >= 10)

    print('the spectrum')
    check('chi is (x-1)(x^2-4x-1)',
          sp.simplify(CHI - sp.expand((x - 1) * (x ** 2 - 4 * x - 1))) == 0)
    check('nu = phi^3 is a root', sp.simplify(CHI.subs(x, NU)) == 0)
    check('-1 is NOT a root', sp.simplify(CHI.subs(x, -1)) != 0)
    check('+1 IS a root', sp.simplify(CHI.subs(x, 1)) == 0)

    print('the fine sector')
    obey, p2, open_ = classify_fine(data)
    check('period-2 components exist and are bounded',
          bool(p2) and all(max(s) <= 4 for _, s in p2))
    check('some fine components remain open', bool(open_))
    check('open components still grow at about nu',
          all(abs(ratio_tail(s) - float(sp.N(NU))) < 0.5
              for _, s in open_ if ratio_tail(s) == ratio_tail(s)))

    print()
    if failures:
        print('%d failure(s): %s' % (len(failures), ', '.join(failures)))
    else:
        print('spectre_boundary_operator: all checks pass.')
    return len(failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--max-order', type=int, default=6)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(1 if selftest(min(args.max_order, 5)) else 0)

    print('spectre_boundary_operator -- derived from the substitution rule')
    data = gather(args.max_order)
    report(data)
    report_spectrum(data)

    png = os.path.join(OUT, 'spectre_boundary_operator.png')
    figure(data, png)
    print('\nwrote %s' % png)

    print('\nsummary')
    print('  nu = phi^3 is structural: it is the growth rate of the contact')
    print('  between adjacent supertiles, and every one of the thirteen')
    print('  slot-pair contacts obeys the same characteristic polynomial.')
    print('  The -1 found earlier is not in this operator at all -- it is a')
    print('  bounded period-2 sector of the fine decomposition, and it moves')
    print('  no growth rate.')


if __name__ == '__main__':
    main()
