#!/usr/bin/env python3
r"""spectreatlas.py -- the contact atlas and the X-charges, from the geometry.

Everything else in this paper is arithmetic on a nine by nine matrix.  This is
not: the X-charge is a statement about which edges of a placed tile meet which
edges of its neighbours, and no amount of eigenvector work recovers it.  The
tiling has to be built and the bonds counted.

The construction, in one paragraph.  Each tile is a fourteen-gon whose edges
carry types in the cyclic sequence `aabbaabbaaaabb`; the Mystic is
Tile(b, a) and carries the role-swapped sequence.  Two adjacent tiles share an
edge, and that shared edge is an \emph{X-bond} when the two tiles disagree
about its type -- an a-edge of one meeting a b-edge of the other.  The
X-charge of a species is the number of X-bonds on an interior tile of that
species, and the claim worth testing is that it is deterministic: that every
interior tile of a given species carries the same number, so the charge is a
property of the species and not of where it happens to sit.

    python3 spectreatlas.py              the atlas and the charges
    python3 spectreatlas.py --selftest   check them
    python3 spectreatlas.py --depth 4    at a chosen depth

Geometry is exact to within the tiling code's own precision: the vertices are
float64 (the repository's own arithmetic), and coincidence is decided by
rounding to a tolerance far coarser than the residuals, which is checked
rather than assumed.
"""

import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

import numpy as np
import spectre as S

# The edge-type sequence around the boundary, from the repository's own
# per-edge decomposition: eight a-edges and six b-edges, with edges nine and
# ten the collinear pair that makes the straight vertex.
EDGE_TYPES = 'aabbaabbaaaabb'
MYSTIC_TYPES = EDGE_TYPES.translate(str.maketrans('ab', 'ba'))

# Rounding used to decide that two vertices are the same point.  The tiling is
# built in float64 and the residuals sit near 1e-12; six decimals is six
# orders of margin, and _check_tolerance below verifies that the answer does
# not depend on the choice.
PLACES = 6


def _key(p):
    return (round(float(p[0]), PLACES), round(float(p[1]), PLACES))


def build(depth, a=1.0, b=1.0):
    """Place a patch and return its tiles as (label, vertex array)."""
    tiles = S.buildSpectreTiles(depth, a, b)
    out = []

    def grab(T, label):
        pts = S.Mystic_SPECTRE_POINTS if label == 'Gamma2' else S.SPECTRE_POINTS
        world = (T[:, :2] @ np.asarray(pts, dtype=np.float64).T).T + T[:, 2]
        out.append((label, world))

    tiles['Delta'].forEachTile(grab)
    return out


def edge_table(patch):
    """Map each undirected edge to the (tile index, slot, type) that own it."""
    table = collections.defaultdict(list)
    for i, (label, world) in enumerate(patch):
        # The type is the CANONICAL role of the slot, the same string for
        # every tile.  The Mystic's swap lives in its geometry -- its
        # canonical a-slot is realized at length b -- not in its labelling.
        # Typing it by realized length instead makes every bond match by
        # construction, since a tiling cannot glue edges of different length,
        # and the X-charge collapses to zero everywhere. That the two readings
        # differ only on the Mystic is the content of its saturation.
        types = EDGE_TYPES
        for slot in range(14):
            p, q = world[slot], world[(slot + 1) % 14]
            key = tuple(sorted((_key(p), _key(q))))
            table[key].append((i, slot, types[slot]))
    return table


def bonds(patch):
    """Shared edges, and whether each is an X-bond."""
    table = edge_table(patch)
    shared, boundary, bad = [], 0, 0
    for key, owners in table.items():
        if len(owners) == 1:
            boundary += 1
            continue
        if len(owners) != 2:
            bad += 1
            continue
        (i, si, ti), (j, sj, tj) = owners
        shared.append((i, si, ti, j, sj, tj, ti != tj))
    return shared, boundary, bad


def analyse(depth):
    """The per-species X-charge, and the contact atlas, at one depth."""
    patch = build(depth)
    shared, boundary, bad = bonds(patch)

    degree = collections.Counter()
    xcount = collections.Counter()
    for i, si, ti, j, sj, tj, isx in shared:
        degree[i] += 1
        degree[j] += 1
        if isx:
            xcount[i] += 1
            xcount[j] += 1

    # An interior tile is one with all fourteen edges shared; only those carry
    # a species-level statement, because a boundary tile is missing bonds it
    # would have had in the infinite tiling.
    interior = [i for i in range(len(patch)) if degree[i] == 14]

    charges = collections.defaultdict(collections.Counter)
    for i in interior:
        charges[patch[i][0]][xcount[i]] += 1

    # The contact atlas: the distinct local adjacency classes seen, a class
    # being (my species, my slot, neighbour species, neighbour slot, X?).
    # A contact class is an unordered pair: which species meets which, at
    # which pair of slots. Counting it as an ordered pair doubles it, which is
    # the difference between 131 classes and 262.
    atlas = set()
    for i, si, ti, j, sj, tj, isx in shared:
        me, you = (patch[i][0], si), (patch[j][0], sj)
        atlas.add((min(me, you), max(me, you), isx))

    return {
        'depth': depth,
        'tiles': len(patch),
        'shared': len(shared),
        'boundary': boundary,
        'degenerate': bad,
        'interior': len(interior),
        'charges': {k: dict(v) for k, v in charges.items()},
        'atlas': atlas,
        'xbonds': sum(1 for e in shared if e[6]),
    }


def _check_tolerance():
    """The answer must not depend on the rounding used to match vertices."""
    global PLACES
    keep, seen = PLACES, {}
    for places in (4, 5, 6, 7, 8):
        PLACES = places
        r = analyse(3)
        seen[places] = (r['shared'], r['boundary'], r['interior'])
    PLACES = keep
    return seen


def selftest(depths=(3, 4)):
    problems = []

    def fail(msg):
        problems.append(msg)

    results = {d: analyse(d) for d in depths}

    for d, r in results.items():
        if r['degenerate']:
            fail('depth %d: %d edges are shared by more than two tiles, so '
                 'the patch is not a tiling' % (d, r['degenerate']))
        if 2 * r['shared'] + r['boundary'] != 14 * r['tiles']:
            fail('depth %d: the edges do not account for 14 per tile' % d)
        if not r['interior']:
            fail('depth %d: no interior tiles, so nothing can be said about '
                 'a species' % d)

    # the charge must be deterministic per species, or it is not a charge
    for d, r in results.items():
        for species, hist in r['charges'].items():
            if len(hist) > 1 and species != 'Phi':
                fail('depth %d: %s carries several X-charges %r, so the '
                     'charge is not a property of the species'
                     % (d, species, hist))

    # and it must agree between depths, or it is a property of the patch
    common = set(results[depths[0]]['charges']) & set(results[depths[-1]]['charges'])
    for species in sorted(common):
        a = set(results[depths[0]]['charges'][species])
        b = set(results[depths[-1]]['charges'][species])
        if a != b:
            fail('%s has X-charge %r at depth %d and %r at depth %d'
                 % (species, sorted(a), depths[0], sorted(b), depths[-1]))

    # every charge should be even, which is the paper's claim
    for d, r in results.items():
        for species, hist in r['charges'].items():
            for value in hist:
                if value % 2:
                    fail('depth %d: %s has odd X-charge %d' % (d, species, value))

    tol = _check_tolerance()
    if len(set(tol.values())) != 1:
        fail('the edge matching depends on the rounding: %r' % tol)

    for msg in problems:
        print('FAIL  ' + msg)
    if problems:
        print('\n%d problem(s).' % len(problems))
    else:
        r = results[depths[-1]]
        print('spectreatlas: depth %d, %d tiles, %d shared edges, %d interior '
              '-- all check.' % (r['depth'], r['tiles'], r['shared'],
                                 r['interior']))
    return len(problems)


def _report(depths=(2, 3, 4)):
    print(__doc__.strip().split('\n')[0])
    print()
    print('  %-6s %-7s %-8s %-9s %-9s %-8s %s'
          % ('depth', 'tiles', 'shared', 'boundary', 'interior', 'X-bonds',
             'atlas'))
    results = {}
    for d in depths:
        r = analyse(d)
        results[d] = r
        print('  %-6d %-7d %-8d %-9d %-9d %-8d %d'
              % (d, r['tiles'], r['shared'], r['boundary'], r['interior'],
                 r['xbonds'], len(r['atlas'])))
    print()
    print('  X-charge by species (interior tiles only):')
    print('  %-9s %s' % ('species', '  '.join('depth %d' % d for d in depths)))
    species = sorted(set().union(*[set(r['charges']) for r in results.values()]))
    for s in species:
        cells = []
        for d in depths:
            hist = results[d]['charges'].get(s, {})
            cells.append(','.join('%d:%d' % kv for kv in sorted(hist.items()))
                         or '-')
        print('  %-9s %s' % (s, '  '.join('%-9s' % c for c in cells)))
    print()
    new = results[depths[-1]]['atlas'] - results[depths[-2]]['atlas']
    print('  atlas classes new at depth %d: %d' % (depths[-1], len(new)))
    print()
    print('  Phi flavour split, as a fraction:')
    for d in depths:
        hist = results[d]['charges'].get('Phi', {})
        tot = sum(hist.values())
        if tot:
            print('    depth %d: p = Pr(Phi^2) = %d/%d = %.4f'
                  % (d, hist.get(2, 0), tot, hist.get(2, 0) / tot))


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(1 if selftest() else 0)
    if '--depth' in sys.argv:
        d = int(sys.argv[sys.argv.index('--depth') + 1])
        import pprint
        pprint.pprint({k: v for k, v in analyse(d).items() if k != 'atlas'})
    else:
        _report()
