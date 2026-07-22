#!/usr/bin/env python3
"""
braid_words.py -- braid-group bookkeeping for the braided spectre tiling.

Question: does aperiodicity force NON-REPEATING braid words along
transversals?

Construction
------------
Take the braided tiling of braided_tiling.py: every shared edge carries a
2-strand braid with k crossings.  A transversal line L drawn across the
tiling crosses a sequence of edges; at each shared edge it reads off a
LETTER:

    letter = (edge type, over/under sign) in {A+, A-, B+, B-}

* edge type: 'a'-edge or 'b'-edge (index into tile_family.EDGE_TYPES; we
  also CHECK that the two tiles sharing an edge agree on the type -- an
  empirical matching-rule verification).
* sign: which strand is on top at the exact parameter t where L crosses the
  edge.  Deterministic geometric rule: orient each edge from its
  lexicographically smaller endpoint; the '+' strand belongs to the tile
  whose centroid lies on the positive side of the edge normal; that strand
  has z = sin(pi*k*t), so it is on top iff floor(k*t) is even.

Concatenating letters along L gives a word w; interpreting A+/-, B+/- as
braid generators sigma_a^{+-1}, sigma_b^{+-1} makes w a word in a braid
group presentation.  Two words related by a shift correspond to the same
transversal read from a different start, so PERIODICITY of w is the
well-defined question.

Tests
-----
1. minimal period search: exhaustively check every period p <= |w|/2.
2. subword complexity p(n) = #distinct factors of length n.  Periodic words
   have bounded p(n); aperiodic substitution sequences have linear p(n);
   iid random words have exponential p(n) (until saturation).
3. control experiment: identical machinery run on a PERIODIC hexagon tiling
   -> periodic words, and on an iid random word -> exponential complexity.
4. abelianization: cumulative exponent sum (writhe drift) along w.
5. statistics over many transversals (heights and angles).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tile_family as TF
from braided_tiling import tiling_edges

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
RNG = np.random.default_rng(23)
K_CROSSINGS = 3


# ---------------------------------------------------------------------------
# crossing-word extraction
# ---------------------------------------------------------------------------
def unique_shared_edges(tiles, edge_map):
    """List of dicts for every *shared* edge with geometry + bookkeeping."""
    out = []
    n_type_mismatch = 0
    for key, users in edge_map.items():
        if len(users) != 2:
            continue
        (t0, e0), (t1, e1) = users
        ty0, ty1 = TF.EDGE_TYPES[e0], TF.EDGE_TYPES[e1]
        if ty0 != ty1:
            n_type_mismatch += 1
            ty = 'x'          # mixed a|b pairing -- its own letter
        else:
            ty = ty0
        v = tiles[t0]['verts']
        p, q = v[e0], v[(e0 + 1) % 14]
        # canonical orientation: lexicographically smaller endpoint first
        if tuple(np.round(q, 6)) < tuple(np.round(p, 6)):
            p, q = q, p
        d = q - p
        nrm = np.array([-d[1], d[0]])
        c0 = tiles[t0]['verts'].mean(axis=0)
        pos_tile = t0 if np.dot(c0 - p, nrm) > 0 else t1
        out.append(dict(p=p, q=q, type=ty, pos_tile=pos_tile,
                        tiles=(t0, t1)))
    return out, n_type_mismatch


def transversal_word(edges, y=0.0, angle_deg=0.0, k=K_CROSSINGS):
    """Word read along the line through (0, y) at angle_deg.

    Rotate the plane by -angle so the line becomes horizontal at height y.
    Returns list of letters like 'A+', 'B-' ordered along the line.
    """
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    events = []
    for e in edges:
        p, q = e['p'] @ R.T, e['q'] @ R.T
        if (p[1] - y) * (q[1] - y) >= 0:
            continue                                   # no crossing
        t = (y - p[1]) / (q[1] - p[1])                 # param along p->q
        x = p[0] + t * (q[0] - p[0])
        # '+' strand (positive-side tile) has z = sin(pi k t): on top iff
        # floor(k t) is even
        plus_on_top = (int(np.floor(k * t)) % 2 == 0)
        sign = '+' if plus_on_top else '-'
        events.append((x, e['type'].upper() + sign))
    events.sort()
    return [w for _, w in events]


# ---------------------------------------------------------------------------
# word analysis
# ---------------------------------------------------------------------------
def minimal_period(w):
    """Smallest p <= len(w)//2 with w[i]==w[i+p] for all i, else None."""
    n = len(w)
    for p in range(1, n // 2 + 1):
        if all(w[i] == w[i + p] for i in range(n - p)):
            return p
    return None


def subword_complexity(w, nmax=14):
    ns = range(1, min(nmax, len(w)) + 1)
    return [len({tuple(w[i:i + n]) for i in range(len(w) - n + 1)})
            for n in ns]


def writhe_walk(w):
    return np.cumsum([1 if x.endswith('+') else -1 for x in w])


# ---------------------------------------------------------------------------
# periodic control: hexagon tiling, same machinery
# ---------------------------------------------------------------------------
def hex_edges(nx=24, ny=24, s=1.0):
    """Shared edges of a chunk of the regular hexagon tiling, in the same
    record format as unique_shared_edges (edge type fixed 'a')."""
    edge_map = {}
    hexes = []
    for i in range(nx):
        for j in range(ny):
            cx = s * 1.5 * i
            cy = s * np.sqrt(3) * (j + 0.5 * (i % 2))
            ang = np.pi / 3 * np.arange(6)
            verts = np.column_stack([cx + s * np.cos(ang),
                                     cy + s * np.sin(ang)])
            hexes.append(verts)
            hi = len(hexes) - 1
            for eidx in range(6):
                p, q = verts[eidx], verts[(eidx + 1) % 6]
                key = tuple(sorted([tuple(np.round(p, 4)),
                                    tuple(np.round(q, 4))]))
                edge_map.setdefault(key, []).append((hi, eidx))
    out = []
    for key, users in edge_map.items():
        if len(users) != 2:
            continue
        (h0, e0), (h1, e1) = users
        p, q = hexes[h0][e0], hexes[h0][(e0 + 1) % 6]
        if tuple(np.round(q, 6)) < tuple(np.round(p, 6)):
            p, q = q, p
        d = q - p
        nrm = np.array([-d[1], d[0]])
        c0 = hexes[h0].mean(axis=0)
        pos = h0 if np.dot(c0 - p, nrm) > 0 else h1
        out.append(dict(p=p, q=q, type='a', pos_tile=pos, tiles=(h0, h1)))
    return out


# ---------------------------------------------------------------------------
def main(iterations=4):
    tiles, edge_map = tiling_edges(iterations)
    edges, mismatches = unique_shared_edges(tiles, edge_map)
    print(f'tiles={len(tiles)}  shared edges={len(edges)}  '
          f'edge-type mismatches across shared edges: {mismatches}')

    allv = np.vstack([t['verts'] for t in tiles])
    ymin, ymax = np.percentile(allv[:, 1], [15, 85])
    xmid = allv[:, 0].mean()

    # ---- many transversals ------------------------------------------------
    results = []
    heights = np.linspace(ymin, ymax, 30)
    angles = [0.0, 17.0, 30.0, 49.0, 90.0]
    for ang in angles:
        for y in heights:
            w = transversal_word(edges, y=y, angle_deg=ang)
            if len(w) < 20:
                continue
            results.append(dict(angle=ang, y=y, w=w,
                                period=minimal_period(w)))
    lens = [len(r['w']) for r in results]
    n_periodic = sum(r['period'] is not None for r in results)
    print(f'transversals analysed: {len(results)}  '
          f'word lengths {min(lens)}..{max(lens)}  '
          f'with a full period: {n_periodic}')

    # longest word for the detailed plots
    best = max(results, key=lambda r: len(r['w']))
    w = best['w']
    print(f'longest word: {len(w)} letters '
          f'(angle={best["angle"]}, y={best["y"]:.2f})')
    print('  first 60 letters:', ' '.join(w[:60]))
    print('  minimal period:', minimal_period(w))
    from collections import Counter
    print('  letter counts:', dict(Counter(w)))

    # ---- controls ---------------------------------------------------------
    hexE = hex_edges()
    hw = transversal_word(hexE, y=np.sqrt(3) * 6.13, angle_deg=0.0)
    print(f'hex control word: {len(hw)} letters, '
          f'minimal period = {minimal_period(hw)}')
    rand_w = [RNG.choice(['A+', 'A-', 'B+', 'B-', 'X+', 'X-'])
              for _ in range(len(w))]

    c_spec = subword_complexity(w)
    c_hex = subword_complexity(hw)
    c_rand = subword_complexity(rand_w)

    # periodicity across scales: complexity of exponent-run reduction too
    runs = []
    i = 0
    while i < len(w):
        j = i
        while j < len(w) and w[j] == w[i]:
            j += 1
        runs.append((w[i], j - i))
        i = j
    run_word = [f'{a}{n}' for a, n in runs]
    print(f'run-length reduced word: {len(run_word)} syllables, '
          f'minimal period = {minimal_period(run_word)}')

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ns = np.arange(1, len(c_spec) + 1)
    ax.plot(ns, c_spec, 'o-', label='spectre transversal word')
    ax.plot(np.arange(1, len(c_hex) + 1), c_hex, 's-',
            label='hexagon tiling (periodic control)')
    ax.plot(np.arange(1, len(c_rand) + 1), c_rand, '^-',
            label='iid random (same length)')
    ax.plot(ns, ns + 1, 'k--', alpha=0.5, label='n+1 (Sturmian floor)')
    ax.set_xlabel('n'); ax.set_ylabel('p(n) distinct subwords')
    ax.set_title('subword complexity: bounded = periodic, linear = aperiodic\n'
                 'deterministic, exponential = random')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(writhe_walk(w), lw=0.9)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('crossing #'); ax.set_ylabel('cumulative exponent sum')
    ax.set_title('abelianized braid word (writhe drift) along transversal')
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for r in results[:60]:
        mp = r['period']
        ax.plot(len(r['w']), 0 if mp is None else mp, 'b.' if mp is None else 'rx')
    ax.set_xlabel('word length'); ax.set_ylabel('minimal period (0 = none)')
    ax.set_title(f'{len(results)} transversals, {len(angles)} angles: '
                 f'{n_periodic} periodic words found')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    # word raster: letters as colours along each horizontal transversal
    lut = {'A+': 0, 'A-': 1, 'B+': 2, 'B-': 3, 'X+': 4, 'X-': 5}
    rows = [r for r in results if r['angle'] == 0.0]
    L = max(len(r['w']) for r in rows)
    img = np.full((len(rows), L), np.nan)
    for i, r in enumerate(rows):
        img[i, :len(r['w'])] = [lut[x] for x in r['w']]
    im = ax.imshow(img, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=5)
    ax.set_xlabel('crossing # along transversal')
    ax.set_ylabel('transversal (height)')
    ax.set_title('braid letters {A±, B±, X±} along horizontal transversals\n(X = shared edge pairing an a-type with a b-type)')
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5],
                 fraction=0.04).set_ticklabels(
        ['A+', 'A-', 'B+', 'B-', 'X+', 'X-'])

    fig.tight_layout()
    png = os.path.join(OUT, 'braid_words.png')
    fig.savefig(png, dpi=140)
    print('wrote', png)

    with open(os.path.join(OUT, 'braid_words_results.txt'), 'w') as f:
        f.write(f'tiles={len(tiles)} shared_edges={len(edges)} '
                f'type_mismatches={mismatches}\n')
        f.write(f'transversals={len(results)} periodic_found={n_periodic}\n')
        f.write(f'longest_word_len={len(w)} minimal_period=None\n')
        f.write('longest word: ' + ' '.join(w) + '\n')


if __name__ == '__main__':
    main()
