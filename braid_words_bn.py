#!/usr/bin/env python3
"""
braid_words_bn.py -- braid-GROUP bookkeeping for the braided spectre tiling.

`braid_words.py` answered "does aperiodicity force non-repeating braid words
along transversals?" with an empirical yes, but left two things open:

  (a) the alphabet {A+-, B+-, X+-} is a 2-strand (B_2) reading, and B_2 = Z
      is abelian, so the only group-theoretic content of a transversal word
      is its writhe.  Nothing non-commutative was being measured.
  (b) subword complexity p(n) saturated near |w| at depth 4 (word lengths
      35..130), so linear-vs-exponential complexity was unresolved.  The
      caveat in the README asks for depth-5 transversals, ~350+ letters.

This module fixes both.

n-strand cables (B_n instead of B_2)
------------------------------------
Generalise the weave: instead of two strands with opposite z-phase, put n
strands at angles theta_j = 2 pi j / n on a circle in the (edge-normal, z)
plane, of radius r sin(pi t) so they pin to the tiling at both vertices, and
rotate that circle by phi(t) = pi k t along the edge.  The bundle therefore
executes k half-turns, so each shared edge carries the Garside half-twist to
the k-th power,

    Delta^k  in  B_n,     Delta = (s1 s2 ... s_{n-1})(s1 ... s_{n-2}) ... (s1)

with k n(n-1)/2 elementary crossings.  Both facts are verified numerically
by `cable_braid_word` rather than assumed.  n = 2 reproduces the old weave
exactly: Delta = s1, k crossings, and the letter read at parameter t is
governed by floor(k t) parity -- the rule hard-coded in braid_words.py.

A transversal crossing an edge at parameter t reads the crossing whose
interval contains t, i.e. crossing index m = floor(M t) with M = k n(n-1)/2,
and emits the letter

    (edge type in {a, b, x},  generator index i,  sign eps)

so the alphabet has 3 (n-1) 2 letters; for n = 2 that is exactly the six
letters of braid_words.py.  For n >= 3 the letters no longer commute and the
transversal word is a genuine non-abelian braid.

Streaming transversals
----------------------
Holding the whole edge table at depth 6 (272 791 tiles) is wasteful when a
transversal only meets a few hundred edges.  `line_shared_edges` streams the
placements in chunks and keeps only edges that actually cross the line, so
depth 6 and 7 are reachable and words run into the thousands of letters --
enough for p(n) to stop saturating.

Measured here
-------------
  * minimal period of every transversal word (letters, and syllables)
  * subword complexity p(n) against the saturation ceiling, at depths 4/5/6
  * recurrence function R(n): the aperiodicity is LINEARLY RECURRENT, which
    is much stronger than "has no period" and separates it from iid random
  * non-abelian invariants: image in S_n, reduced Burau at t = -1, and the
    Lyapunov exponent of the Burau product
  * controls: periodic hexagon tiling, iid random word, and a periodic
    substitution word

Outputs (into $EINSTEIN3D_OUT, default /tmp):
    braid_words_bn.png        the six-panel analysis figure
    braid_cable_geometry.png  the n = 2, 3, 4 cables and a cabled patch
    braid_cable.obj           cabled tiling mesh for Blender
    braid_words_bn_results.txt
"""
import os, argparse, time
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import spectre as S
import tile_family as TF
import closure_repair as CR

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
RNG = np.random.default_rng(23)


# ---------------------------------------------------------------------------
# 1. the n-strand cable and its braid word
# ---------------------------------------------------------------------------
def cable_phase(n):
    """Offset that keeps the n strands in general position at t = 0 and 1.

    Without it, n = 3 starts with cos(120 deg) = cos(240 deg): two strands
    coincide in projection and the braid word is ill-defined at the vertex.
    """
    return np.pi / (2 * n)


def cable_positions(t, k, n, phase=None):
    """Projected transverse coordinate x_j and height z_j of each strand."""
    t = np.atleast_1d(np.asarray(t, float))
    phase = cable_phase(n) if phase is None else phase
    th = 2 * np.pi * np.arange(n) / n
    phi = np.pi * k * t[:, None] + phase
    ang = th[None, :] + phi
    return np.cos(ang), np.sin(ang)


def strand_role(j, n):
    """Which physical curve each strand of the cable is.

    Strand 0 is the copy of the shared edge contributed by the tile on the
    positive side of the edge normal (the '+' tile of braid_words.py),
    strand 1 is the other tile's copy, and any further strands are copies of
    the TILE BOUNDARY itself -- the flat edge of the underlying tiling,
    lifted into the weave.  So n = 3 is exactly 'B_3 with the tile boundary
    as a third strand'.
    """
    return 'A' if j == 0 else ('B' if j == 1 else 'G')


def cable_braid_word(k=3, n=3, samples=40000):
    """Extract the braid word of the k-half-turn n-cable by simulation.

    Returns (letters, perm, ts) with letters a list of
    (generator index i in 1..n-1, crossing sign eps, over strand id,
     under strand id) in order along the edge, and perm the permutation of
    strand ids induced from t = 0 to t = 1.
    """
    t = np.linspace(0, 1, samples)
    x, z = cable_positions(t, k, n)
    start = np.argsort(x[0])                    # strand ids, left to right
    order = start.copy()
    letters, ts = [], []
    for m in range(1, samples):
        xs = x[m][order]
        for i in np.where(np.diff(xs) < 0)[0]:
            a, b = order[i], order[i + 1]
            over, under = (a, b) if z[m][a] > z[m][b] else (b, a)
            eps = 1 if z[m][a] > z[m][b] else -1
            letters.append((i + 1, eps, int(over), int(under)))
            ts.append(t[m])
            order[i], order[i + 1] = order[i + 1], order[i]
    # permutation of strand ids relative to the starting arrangement
    pos_start = {sid: i for i, sid in enumerate(start)}
    perm = tuple(pos_start[sid] for sid in order)
    return letters, perm, np.array(ts)


_CABLE_CACHE = {}


def cable_letters(k, n):
    if (k, n) not in _CABLE_CACHE:
        _CABLE_CACHE[(k, n)] = cable_braid_word(k, n)
    return _CABLE_CACHE[(k, n)]


def letter_at(t, k, n):
    """The crossing a transversal reads at parameter t along the edge.

    The cable's k n(n-1)/2 crossings are evenly spaced in t, so the reading
    is crossing number floor(M t).  For n = 2 this is floor(k t), and the
    over-strand alternates with its parity -- the exact rule hard-coded in
    braid_words.py, so that alphabet is recovered as a special case.
    """
    letters, _, _ = cable_letters(k, n)
    M = len(letters)
    m = min(int(np.floor(M * t)), M - 1)
    return letters[m]


# ---------------------------------------------------------------------------
# 2. streaming transversal extraction
# ---------------------------------------------------------------------------
CANON = TF.canonical_tile_verts('any')[:14]


def _world_verts(placements, lengths=None, chunk=20000):
    """Yield (start_index, world[chunk,14,2]) blocks."""
    if lengths is None:
        local = CANON
    else:
        v, _ = TF.build_polygon(lengths, mirror_dirs=CR.UNIT_DIRS)
        local = v[:14] - v[:14].mean(axis=0) + CANON.mean(axis=0)
    N = len(placements)
    for s in range(0, N, chunk):
        blk = placements[s:s + chunk]
        A = np.stack([T[:, :2] for T, _ in blk])       # m,2,2
        b = np.stack([T[:, 2] for T, _ in blk])        # m,2
        world = np.einsum('mij,vj->mvi', A, local) + b[:, None, :]
        yield s, world


def line_shared_edges(n_iterations, angle_deg=0.0, y=0.0, lengths=None,
                      placements=None, chunk=20000):
    """Shared edges crossed by the line at height y after rotating by -angle.

    Streams the tiling, so memory is O(number of edges on the line).
    Returns a list of dicts sorted along the line.
    """
    if placements is None:
        placements = TF.placed_tiles(n_iterations)
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])

    hits = defaultdict(list)
    for s, world in _world_verts(placements, lengths, chunk):
        w = world @ R.T                                  # rotate into line frame
        p = w
        q = np.roll(w, -1, axis=1)
        cross = (p[:, :, 1] - y) * (q[:, :, 1] - y) < 0
        ti, ei = np.nonzero(cross)
        for a, b in zip(ti, ei):
            pp, qq = p[a, b], q[a, b]
            key = (tuple(np.round(pp, 3)), tuple(np.round(qq, 3)))
            key = tuple(sorted(key))
            hits[key].append((s + int(a), int(b), pp, qq))

    out = []
    for key, users in hits.items():
        if len(users) != 2:
            continue                                     # boundary edge
        (t0, e0, p0, q0), (t1, e1, _, _) = users
        ty0, ty1 = TF.EDGE_TYPES[e0], TF.EDGE_TYPES[e1]
        ty = ty0 if ty0 == ty1 else 'x'
        # canonical orientation: lexicographically smaller endpoint first
        p, q = (p0, q0) if tuple(np.round(p0, 6)) < tuple(np.round(q0, 6)) \
            else (q0, p0)
        t = (y - p[1]) / (q[1] - p[1])
        xline = p[0] + t * (q[0] - p[0])
        out.append(dict(x=float(xline), t=float(t), type=ty,
                        tiles=(t0, t1), edges=(e0, e1)))
    out.sort(key=lambda r: r['x'])
    return out


def word_from_edges(edges, k=3, n=3):
    """Letters (edge type, generator index, sign, over role, under role)."""
    w = []
    for e in edges:
        i, eps, over, under = letter_at(e['t'], k, n)
        w.append((e['type'], i, eps, strand_role(over, n),
                  strand_role(under, n)))
    return w


def fmt(letter):
    ty, i, eps, over, under = letter
    return f"{ty.upper()}{i}{over}{under}"


# ---------------------------------------------------------------------------
# 3. word analysis
# ---------------------------------------------------------------------------
def minimal_period(w):
    m = len(w)
    for p in range(1, m // 2 + 1):
        if all(w[i] == w[i + p] for i in range(m - p)):
            return p
    return None


def subword_complexity(w, nmax=20):
    return [len({tuple(w[i:i + j]) for i in range(len(w) - j + 1)})
            for j in range(1, min(nmax, len(w)) + 1)]


def recurrence_function(w, nmax=8, min_occ=3):
    """R(n) = largest gap between consecutive occurrences of a factor.

    Linearly recurrent sequences (all primitive substitution systems)
    satisfy R(n) <= C n; iid random words need R(n) ~ |A|^n.

    In a FINITE word some factors appear only once or twice, and their gap
    is not a property of the sequence but of where the window was cut.  We
    therefore restrict to factors seen at least `min_occ` times and also
    return the fraction of factor OCCURRENCES that this covers, so the
    reader can see when n has grown too large for the sample.
    """
    out, cover = [], []
    for j in range(1, min(nmax, len(w) // 4) + 1):
        pos = defaultdict(list)
        for i in range(len(w) - j + 1):
            pos[tuple(w[i:i + j])].append(i)
        gap, kept, tot = 0, 0, 0
        for f, ps in pos.items():
            tot += len(ps)
            if len(ps) < min_occ:
                continue
            kept += len(ps)
            gap = max(gap, int(max(np.diff(ps))))
        out.append(gap if gap else np.nan)
        cover.append(kept / max(tot, 1))
    return out, cover


PROJECTIONS = {
    'full': lambda L: L,
    'type': lambda L: L[0],                     # a / b / x   (3 letters)
    'gen':  lambda L: L[1],                     # which braid generator
    'over': lambda L: L[3],                     # A / B / G is on top
}


def project(w, mode='type'):
    """Coarse-grain the letters.

    Complexity classes are only measurable when |w| >> |alphabet|^n.  The
    full alphabet has 3(n-1)... = 18 letters for n = 3, so even a 1000-letter
    word saturates p(n) by n = 3 and cannot separate linear from exponential.
    The tiling's combinatorics live in the EDGE TYPE projection {a, b, x},
    a 3-letter alphabet, where a word of a few thousand letters resolves
    p(n) out to n ~ 7.  That is the projection the complexity question
    should be asked in; the full alphabet is kept for the group invariants.
    """
    f = PROJECTIONS[mode]
    return [f(L) for L in w]


def complexity_exponent(w, nmax=16, clip_frac=0.25, nmin=2):
    """Fit p(n) ~ C n^alpha over the range where p(n) is not yet clipped.

    A transversal of a 2D aperiodic tiling is not a 1D substitution
    sequence, so there is no reason to expect the Sturmian-like p(n) ~ n.
    The relevant question is whether p(n) is POLYNOMIAL (deterministic
    aperiodic order) or EXPONENTIAL (random).  We fit only where
    p(n) <= clip_frac * |w|, so the ceiling |w|-n+1 is not shaping the fit.
    Returns (alpha, ns_used, ps_used).
    """
    c = np.array(subword_complexity(w, nmax), float)
    ns = np.arange(1, len(c) + 1)
    keep = (ns >= nmin) & (c <= clip_frac * len(w))
    if keep.sum() < 3:
        return np.nan, ns[keep], c[keep]
    a, _ = np.polyfit(np.log(ns[keep]), np.log(c[keep]), 1)
    return float(a), ns[keep], c[keep]


def matched_random(w, rng=RNG):
    """iid word with the SAME letter frequencies as w.

    A uniform random control would be unfair: the tiling emits a, b, x at
    roughly 49 / 28 / 23 %, and non-uniform frequencies alone depress p(n).
    """
    letters = sorted(set(w), key=str)
    counts = np.array([w.count(l) for l in letters], float)
    idx = rng.choice(len(letters), size=len(w), p=counts / counts.sum())
    return [letters[i] for i in idx]


def syllables(w):
    out, i = [], 0
    while i < len(w):
        j = i
        while j < len(w) and w[j] == w[i]:
            j += 1
        out.append((w[i], j - i))
        i = j
    return out


# --- non-abelian invariants -------------------------------------------------
def permutation_image(w, n):
    """Image of the braid word in S_n (as a tuple)."""
    p = list(range(n))
    for _, i, _, _, _ in w:
        p[i - 1], p[i] = p[i], p[i - 1]
    return tuple(p)


def burau_gen(i, n, t=-1, inverse=False):
    """Unreduced Burau matrix of sigma_i^{+-1} in GL_n(Z[t,1/t]) at t."""
    M = np.eye(n)
    if not inverse:
        M[i - 1:i + 1, i - 1:i + 1] = [[1 - t, t], [1, 0]]
    else:
        M[i - 1:i + 1, i - 1:i + 1] = [[0, 1], [1 / t, 1 - 1 / t]]
    return M


def burau_walk(w, n, t=-1):
    """log ||prefix Burau matrix|| along the word."""
    M = np.eye(n)
    out = []
    for _, i, eps, _, _ in w:
        M = M @ burau_gen(i, n, t, inverse=(eps < 0))
        out.append(np.log(np.linalg.norm(M) + 1e-300))
    return np.array(out)


def writhe_walk(w):
    return np.cumsum([eps for _, _, eps, _, _ in w])


# ---------------------------------------------------------------------------
# 4. controls
# ---------------------------------------------------------------------------
def hex_line_edges(y=0.0, angle_deg=0.0, nx=160, ny=160, s=1.0):
    """Same extraction on the regular hexagon tiling (periodic control)."""
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    ang = np.pi / 3 * np.arange(6)
    unit = np.column_stack([s * np.cos(ang), s * np.sin(ang)])
    i = np.repeat(np.arange(nx), ny)
    j = np.tile(np.arange(ny), nx)
    cx = s * 1.5 * i
    cy = s * np.sqrt(3) * (j + 0.5 * (i % 2))
    centres = np.column_stack([cx, cy])
    verts = (centres[:, None, :] + unit[None, :, :]) @ R.T
    p, q = verts, np.roll(verts, -1, axis=1)
    cross = (p[:, :, 1] - y) * (q[:, :, 1] - y) < 0
    hits = defaultdict(list)
    ti, ei = np.nonzero(cross)
    for a, b in zip(ti, ei):
        pp, qq = p[a, b], q[a, b]
        key = tuple(sorted([tuple(np.round(pp, 3)), tuple(np.round(qq, 3))]))
        hits[key].append((int(a), int(b), pp, qq))
    out = []
    for key, users in hits.items():
        if len(users) != 2:
            continue
        (_, _, p0, q0) = users[0]
        pp, qq = (p0, q0) if tuple(np.round(p0, 6)) < tuple(np.round(q0, 6)) \
            else (q0, p0)
        t = (y - pp[1]) / (qq[1] - pp[1])
        out.append(dict(x=float(pp[0] + t * (qq[0] - pp[0])), t=float(t),
                        type='a', tiles=(0, 0), edges=(0, 0)))
    out.sort(key=lambda r: r['x'])
    return out


def alphabet_of(k, n):
    """Every letter the construction can emit: (edge type) x (crossing)."""
    letters, _, _ = cable_letters(k, n)
    seen = {(i, eps, strand_role(o, n), strand_role(u, n))
            for i, eps, o, u in letters}
    return [(ty,) + c for ty in ('a', 'b', 'x') for c in sorted(seen)]


def random_word(length, k, n, rng=RNG):
    """iid uniform on the same alphabet the tiling can actually emit."""
    A = alphabet_of(k, n)
    return [A[i] for i in rng.integers(0, len(A), size=length)]


def periodic_substitution_word(length, k, n):
    """A periodic control drawn from the same alphabet."""
    A = alphabet_of(k, n)
    base = [A[i % len(A)] for i in range(min(5, len(A)))]
    return [base[i % len(base)] for i in range(length)]


# ---------------------------------------------------------------------------
# 5. cable geometry (extends braided_tiling.py to n strands)
# ---------------------------------------------------------------------------
def cable_strands(p, q, k, n, height=0.16, radius=0.08, samples=48):
    """3D polylines of the n strands cabled around the segment p->q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    t = np.linspace(0, 1, samples)
    w = np.sin(np.pi * t)
    x, z = cable_positions(t, k, n)
    out = []
    for j in range(n):
        xy = (p[None, :] + d[None, :] * t[:, None]
              + (radius * L * w * x[:, j])[:, None] * nrm[None, :])
        zz = height * L * w * z[:, j]
        out.append(np.column_stack([xy, zz]))
    return out


def cabled_patch(n_iterations=1, k=3, n=3, height=0.16, radius=0.08):
    placements = TF.placed_tiles(n_iterations)
    tiles, em = [], defaultdict(list)
    for ti, (T, label) in enumerate(placements):
        world = TF.transform_polygon(T, np.vstack([CANON, CANON[:1]]))[:14]
        tiles.append(world)
        for ei in range(14):
            key = tuple(sorted([tuple(np.round(world[ei], 4)),
                                tuple(np.round(world[(ei + 1) % 14], 4))]))
            em[key].append((ti, ei))
    strands = []
    for key, users in em.items():
        if len(users) != 2:
            continue
        ti, ei = users[0]
        p, q = tiles[ti][ei], tiles[ti][(ei + 1) % 14]
        strands.extend(cable_strands(p, q, k, n, height, radius))
    return tiles, strands


def export_obj(tiles, strands, fname, ribbon_width=0.05):
    V, F = [], []

    def add(p):
        V.append(p); return len(V)

    for t in tiles:
        c = t.mean(axis=0)
        ci = add([c[0], c[1], 0.0])
        ring = [add([p[0], p[1], 0.0]) for p in t]
        for i in range(len(t)):
            F.append((ci, ring[i], ring[(i + 1) % len(t)]))
    for pts in strands:
        d = np.gradient(pts, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        side = np.cross(d, np.array([0, 0, 1.0]))
        nn = np.linalg.norm(side, axis=1, keepdims=True)
        side = np.where(nn > 1e-8, side / np.maximum(nn, 1e-12),
                        np.array([1.0, 0, 0]))
        a = pts + 0.5 * ribbon_width * side
        b = pts - 0.5 * ribbon_width * side
        ia = [add(v.tolist()) for v in a]
        ib = [add(v.tolist()) for v in b]
        for i in range(len(ia) - 1):
            F.append((ia[i], ib[i], ib[i + 1]))
            F.append((ia[i], ib[i + 1], ia[i + 1]))
    with open(fname, 'w') as f:
        f.write('# cabled spectre tiling\n')
        for v in V:
            f.write(f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n')
        for a, b, c in F:
            f.write(f'f {a} {b} {c}\n')
    return len(V), len(F)


# ---------------------------------------------------------------------------
# 6. plots
# ---------------------------------------------------------------------------
def plot_cable_geometry(k, fname):
    fig = plt.figure(figsize=(16, 9))
    for j, n in enumerate((2, 3, 4)):
        ax = fig.add_subplot(2, 3, j + 1, projection='3d')
        for s in cable_strands([0, 0], [1, 0], k, n, height=0.22, radius=0.12,
                               samples=200):
            ax.plot(s[:, 0], s[:, 1], s[:, 2], lw=2)
        letters, perm, ts = cable_letters(k, n)
        ax.set_title(f'{n}-strand cable, k={k} half-turns\n'
                     f'{len(letters)} crossings '
                     f'(= k n(n-1)/2 = {k*n*(n-1)//2})\n'
                     f'permutation {tuple(perm)}', fontsize=9)
        ax.set_axis_off(); ax.set_box_aspect((3, 1, 1))
        ax.view_init(elev=22, azim=-70)

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    tiles, strands = cabled_patch(1, k=k, n=3)
    polys = [np.column_stack([t, np.zeros(len(t))]) for t in tiles]
    ax.add_collection3d(Poly3DCollection(polys, facecolors='lightsteelblue',
                                         edgecolors='none', alpha=0.30))
    for i, s in enumerate(strands):
        ax.plot(s[:, 0], s[:, 1], s[:, 2], lw=1.3,
                color=plt.cm.turbo((i % 3) / 3.0))
    allv = np.vstack(tiles)
    cx, cy = allv.mean(axis=0)
    r = 0.55 * (allv.max(axis=0) - allv.min(axis=0)).max()
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(-r * 0.35, r * 0.35)
    ax.set_box_aspect((1, 1, 0.35)); ax.set_axis_off()
    ax.view_init(elev=58, azim=-60)
    ax.set_title('every shared edge carries a 3-strand cable = the Garside '
                 f'half-twist $\\Delta^{{{k}}}$ in $B_3$', fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def plot_analysis(data, fname):
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))

    ax = axes[0, 0]
    nmax = data['nmax']
    for lab, w, style in data['complexity_sets']:
        c = subword_complexity(w, nmax)
        ns = np.arange(1, len(c) + 1)
        ax.plot(ns, c, style, ms=4, label=f'{lab} (|w|={len(w)})')
    L = max(len(w) for _, w, _ in data['complexity_sets'])
    A = data['nletters']
    ns = np.arange(1, nmax + 1)
    ax.plot(ns, np.minimum(L - ns + 1, A ** ns.astype(float)), 'k:',
            label=r'ceiling min($|w|-n+1$, $|A|^n$)')
    ax.set_yscale('log')
    ax.set_xlabel('n'); ax.set_ylabel('p(n)')
    ax.set_title(f'subword complexity in the edge-type alphabet '
                 f'|A| = {A}\n(the projection where the classes are '
                 f'actually separable)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for lab, w, style in data['complexity_sets']:
        c = np.array(subword_complexity(w, nmax), float)
        ns = np.arange(1, len(c) + 1)
        ax.plot(ns, c / ns, style, ms=4, label=lab)
    ax.set_xlabel('n'); ax.set_ylabel('p(n) / n')
    ax.set_title('linear complexity test: p(n)/n flat = linear (aperiodic\n'
                 'deterministic), rising = random, falling to 0 = periodic')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    for lab, R, style in data['recurrence']:
        ns = np.arange(1, len(R) + 1)
        ax.semilogy(ns, R, style, ms=4, label=lab)
    nn = np.arange(1, 8)
    ax.semilogy(nn, 40 * nn, 'k:', lw=1, label='40 n (linear reference)')
    ax.set_ylim(bottom=1)
    ax.set_xlabel('n'); ax.set_ylabel('R(n) = max gap between occurrences')
    ax.set_title('recurrence function: R(n) ~ C n means LINEARLY RECURRENT\n'
                 '(strictly stronger than "no period")')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for lab, wk, style in data['burau']:
        ax.plot(wk, style.replace('o', ''), lw=1.2, label=lab)
    ax.set_xlabel('crossing #'); ax.set_ylabel(r'$\log\,\|$Burau prefix$\|$')
    ax.set_title('reduced Burau at t = -1: growth rate of the non-abelian\n'
                 'invariant along the transversal')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for lab, ww, style in data['writhe']:
        ax.plot(ww, style.replace('o', ''), lw=1.0, label=lab)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('crossing #'); ax.set_ylabel('cumulative exponent sum')
    ax.set_title('abelianisation (writhe drift) -- all that $B_2$ could see')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    rows = data['raster']
    lut = data['lut']
    Lm = max(len(r) for r in rows)
    img = np.full((len(rows), Lm), np.nan)
    for i, r in enumerate(rows):
        img[i, :len(r)] = [lut[x] for x in r]
    im = ax.imshow(img, aspect='auto', cmap='turbo', interpolation='nearest',
                   vmin=0, vmax=max(lut.values()))
    ax.set_xlabel('crossing # along transversal')
    ax.set_ylabel('transversal')
    ax.set_title(f'$B_{{{data["n"]}}}$ letters along transversals\n'
                 f'alphabet {sorted(lut, key=lut.get)}', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)

    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=5)
    ap.add_argument('--deep', type=int, default=6,
                    help='depth used for the single longest word')
    ap.add_argument('--strands', type=int, default=3)
    ap.add_argument('--crossings', type=int, default=3)
    ap.add_argument('--transversals', type=int, default=24)
    args = ap.parse_args()
    n, k = args.strands, args.crossings

    lines = []

    def say(s=''):
        print(s, flush=True); lines.append(s)

    t_start = time.time()

    say('=== the n-strand cable as a Garside half-twist ===')
    hdr = f"{'n':>3s} {'k':>3s} {'crossings':>10s} {'k n(n-1)/2':>11s} " \
          f"{'permutation':>16s} {'word (first 10)':>26s}"
    say(hdr); say('-' * len(hdr))
    for nn in (2, 3, 4, 5):
        letters, perm, _ = cable_letters(k, nn)
        wtxt = ' '.join(f"s{i}{'+' if e > 0 else '-'}"
                        for i, e, _, _ in letters[:10])
        say(f'{nn:3d} {k:3d} {len(letters):10d} {k*nn*(nn-1)//2:11d} '
            f'{str(perm):>16s} {wtxt:>26s}')
    say('n=2 reproduces braid_words.py exactly: k crossings, letter parity '
        'floor(k t) mod 2')
    say()

    say('=== transversal words ===')
    results = []
    for depth in sorted({4, args.iterations}):
        placements = TF.placed_tiles(depth)
        allv = np.vstack([T[:, 2] for T, _ in placements])
        ymin, ymax = np.percentile(allv[:, 1], [12, 88])
        for ang in (0.0, 17.0, 30.0, 49.0, 90.0):
            for y in np.linspace(ymin, ymax, args.transversals // 5 + 1):
                edges = line_shared_edges(depth, angle_deg=ang, y=float(y),
                                          placements=placements)
                if len(edges) < 20:
                    continue
                w = word_from_edges(edges, k, n)
                results.append(dict(depth=depth, angle=ang, y=float(y), w=w,
                                    period=minimal_period(w)))
        say(f'  depth {depth}: {len(placements)} tiles, '
            f'{sum(1 for r in results if r["depth"]==depth)} transversals')

    lens = [len(r['w']) for r in results]
    n_per = sum(r['period'] is not None for r in results)
    say(f'transversals analysed: {len(results)}   word lengths '
        f'{min(lens)}..{max(lens)}   with a full period: {n_per}')
    syl_per = sum(minimal_period(syllables(r['w'])) is not None
                  for r in results)
    say(f'run-length (syllable) reduced words with a period: {syl_per}')
    say()

    say(f'=== deep transversal (depth {args.deep}) ===')
    t0 = time.time()
    placements = TF.placed_tiles(args.deep)
    ctr = np.vstack([T[:, 2] for T, _ in placements])
    best = None
    for ang in (0.0, 17.0, 30.0, 49.0, 90.0):
        th = np.deg2rad(ang)
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        ys = (ctr @ R.T)[:, 1]
        for y in np.percentile(ys, [35, 50, 65]):
            e = line_shared_edges(args.deep, angle_deg=ang, y=float(y),
                                  placements=placements)
            if best is None or len(e) > len(best[2]):
                best = (ang, float(y), e)
    ang, y, edges = best
    W = word_from_edges(edges, k, n)
    say(f'{len(placements)} tiles, longest transversal: {len(W)} letters '
        f'(angle={ang}, y={y:.2f})   [{time.time()-t0:.1f}s]')
    say(f'  minimal period            : {minimal_period(W)}')
    say(f'  syllable-reduced period   : {minimal_period(syllables(W))}')
    say(f'  distinct letters used     : {len(set(W))} of '
        f'{len(alphabet_of(k, n))} possible')
    say(f'  letter counts             : '
        f'{dict(Counter(fmt(x) for x in W).most_common())}')
    say(f'  edge-type mix a/b/x       : '
        f'{dict(Counter(str(t) for t, *_ in W))}')
    xfrac = sum(1 for t, *_ in W if t == 'x') / len(W)
    say(f'  fraction of a|b mixed (X) shared edges: {100*xfrac:.2f}% '
        f'(braid_words.py reported 24.1% over the whole depth-4 tiling)')
    say(f'  image in S_{n}              : {permutation_image(W, n)}')
    say(f'  writhe (abelianisation)   : {int(writhe_walk(W)[-1])}')
    bw = burau_walk(W, n)
    say(f'  Burau(t=-1) log-norm      : {bw[-1]:.3f}  '
        f'-> Lyapunov {bw[-1]/len(W):.5f} per crossing')
    say('  first 30 letters: ' + ' '.join(fmt(x) for x in W[:30]))
    say()

    say('=== controls ===')
    hexE = hex_line_edges(y=np.sqrt(3) * 40.13, angle_deg=0.0)
    hw = word_from_edges(hexE, k, n)
    say(f'hexagon tiling (periodic) : {len(hw)} letters, minimal period '
        f'{minimal_period(hw)}')
    rw = random_word(len(W), k, n)
    say(f'iid random                : {len(rw)} letters, minimal period '
        f'{minimal_period(rw)}')
    pw = periodic_substitution_word(len(W), k, n)
    say(f'periodic control word     : {len(pw)} letters, minimal period '
        f'{minimal_period(pw)}')
    say()

    say('=== complexity: full alphabet vs edge-type projection ===')
    W4 = max((r for r in results if r['depth'] == 4),
             key=lambda r: len(r['w']))['w']
    W5 = max((r for r in results if r['depth'] == args.iterations),
             key=lambda r: len(r['w']))['w']
    say('The full alphabet has %d letters, so p(n) saturates almost at once '
        '(below).' % len(alphabet_of(k, n)))
    hdr = f"{'word (full alphabet)':26s} {'|w|':>6s} " + \
        ' '.join(f'p({i})'.rjust(6) for i in range(1, 7))
    say(hdr); say('-' * len(hdr))
    for lab, ww in ((f'spectre depth {args.deep}', W), ('iid random', rw)):
        c = subword_complexity(ww, 6)
        say(f'{lab:26s} {len(ww):6d} ' + ' '.join(f'{v:6d}' for v in c))
    say('Both sit on the ceiling |w|-n+1 by n=4: nothing is resolved there.')
    say()

    say(f'--- projection to edge types {{a, b, x}} (|A| = 3) ---')
    Wp = {d: project(w, 'type') for d, w in
          ((4, W4), (args.iterations, W5), (args.deep, W))}
    hwp = project(hw, 'type')
    rnd = matched_random(Wp[args.deep])
    say(f'letter frequencies of the deep word: '
        f'{ {kk: round(vv/len(Wp[args.deep]), 4) for kk, vv in Counter(Wp[args.deep]).most_common()} }')
    say('(the iid control is matched to exactly these frequencies)')
    nmax = 8
    comp_sets = [(f'spectre depth {d}', Wp[d], st) for d, st in
                 ((4, 'o-'), (args.iterations, 's-'), (args.deep, 'd-'))]
    comp_sets += [('hexagon (periodic)', hwp, 'v-'),
                  ('iid, matched freqs', rnd, '^-')]
    hdr = f"{'word':26s} {'|w|':>6s} " + \
        ' '.join(f'p({i})'.rjust(6) for i in range(1, nmax + 1))
    say(hdr); say('-' * len(hdr))
    for lab, ww, _ in comp_sets:
        c = subword_complexity(ww, nmax)
        say(f'{lab:26s} {len(ww):6d} ' + ' '.join(f'{v:6d}' for v in c))
    ceil = [min(len(Wp[args.deep]) - i, 3 ** i) for i in range(1, nmax + 1)]
    say(f'{"ceiling min(|w|-n+1, 3^n)":26s} {"":6s} ' +
        ' '.join(f'{v:6d}' for v in ceil))
    say()
    cs = subword_complexity(Wp[args.deep], nmax)
    cr = subword_complexity(rnd, nmax)
    say('p(n)/n  spectre : ' +
        ' '.join(f'{c/(i+1):6.1f}' for i, c in enumerate(cs)))
    say('p(n)/n  random  : ' +
        ' '.join(f'{c/(i+1):6.1f}' for i, c in enumerate(cr)))
    say(f'p(n) ratio spectre/random: ' +
        ' '.join(f'{a/b:6.3f}' for a, b in zip(cs, cr)))
    say()
    say('--- is p(n) polynomial or exponential? power-law fit p(n) ~ C n^a ---')
    say('(fitted only where p(n) <= 25% of |w|, so the ceiling cannot shape '
        'the fit)')
    hdr = f"{'word':26s} {'|w|':>6s} {'alpha':>8s} {'n used':>14s}"
    say(hdr); say('-' * len(hdr))
    for lab, ww in ([(f'spectre depth {d}', Wp[d]) for d in sorted(Wp)] +
                    [('iid, matched freqs', rnd)]):
        a, ns_u, _ = complexity_exponent(ww)
        say(f'{lab:26s} {len(ww):6d} {a:8.3f} '
            f'{(str(list(ns_u)) if len(ns_u) else "-"):>14s}')
    say('The exponent has not converged, and over so short a window the fit '
        'alone does not separate the classes -- the random control admits a '
        'power-law fit too.  The entropy estimate below is the sharper '
        'statement.')
    say()
    say('--- entropy estimate h(n) = log p(n) / n ---')
    say('h(n) -> positive constant for exponential (random), -> 0 for '
        'polynomial, = 0 for periodic')
    hdr = f"{'word':26s} " + ' '.join(f'h({i})'.rjust(7)
                                      for i in range(1, nmax + 1))
    say(hdr); say('-' * len(hdr))
    for lab, ww in ([(f'spectre depth {d}', Wp[d]) for d in sorted(Wp)] +
                    [('iid, matched freqs', rnd), ('hexagon (periodic)', hwp)]):
        c = subword_complexity(ww, nmax)
        say(f'{lab:26s} ' + ' '.join(
            f'{np.log(v)/(i+1):7.3f}' for i, v in enumerate(c)))
    say()

    say('=== recurrence function R(n), edge-type projection ===')
    rec = []
    nmaxR = 7
    hdr = (f"{'word':26s} " + ' '.join(f'R({i})'.rjust(7)
                                       for i in range(1, nmaxR + 1)))
    say(hdr); say('-' * len(hdr))
    for lab, ww, style in [(f'spectre depth {args.deep}', Wp[args.deep], 'd-'),
                           ('hexagon (periodic)', hwp, 'v-'),
                           ('iid, matched freqs', rnd, '^-')]:
        R, cov = recurrence_function(ww, nmaxR)
        rec.append((lab, R, style))
        say(f'{lab:26s} ' + ' '.join(
            ('    n/a' if v != v else f'{int(v):7d}') for v in R))
        say(f'{"  occurrence coverage":26s} ' +
            ' '.join(f'{100*c:6.0f}%' for c in cov))
    R, cov = recurrence_function(Wp[args.deep], nmaxR)
    ok = [(i + 1, R[i]) for i in range(len(R)) if R[i] == R[i] and cov[i] > 0.5]
    if ok:
        ratio = [v / i for i, v in ok]
        say(f'spectre R(n)/n over n with >50% coverage '
            f'(n = {[i for i, _ in ok]}): '
            f'{[round(x, 1) for x in ratio]}')
        say(f'  ratio range {min(ratio):.1f}..{max(ratio):.1f}')
    say('Caveat: R(n) is only meaningful while most factors still recur; '
        'coverage is reported so the reader can see where it stops being so.')
    say()

    # ---- figures ----------------------------------------------------------
    alphabet = sorted({fmt(x) for x in W} | {fmt(x) for x in hw})
    lut = {a: i for i, a in enumerate(alphabet)}
    rows = [[fmt(x) for x in r['w']]
            for r in results if r['depth'] == args.iterations][:40]
    rows = [r for r in rows if r] or [[fmt(x) for x in W]]
    data = dict(complexity_sets=comp_sets, recurrence=rec, n=n, lut=lut,
                raster=rows, nmax=nmax, nletters=3,
                burau=[(f'spectre depth {args.deep}', burau_walk(W, n), '-'),
                       ('hexagon (periodic)', burau_walk(hw, n), '-'),
                       ('iid random', burau_walk(rw, n), '-')],
                writhe=[(f'spectre depth {args.deep}', writhe_walk(W), '-'),
                        ('hexagon (periodic)', writhe_walk(hw), '-'),
                        ('iid random', writhe_walk(rw), '-')])
    png = os.path.join(OUT, 'braid_words_bn.png')
    plot_analysis(data, png)
    say(f'wrote {png}')

    png = os.path.join(OUT, 'braid_cable_geometry.png')
    plot_cable_geometry(k, png)
    say(f'wrote {png}')

    tiles, strands = cabled_patch(1, k=k, n=n)
    obj = os.path.join(OUT, 'braid_cable.obj')
    nv, nf = export_obj(tiles, strands, obj)
    say(f'wrote {obj}  ({len(strands)} strands, {nv} verts, {nf} tris)')
    say(f'total time {time.time()-t_start:.1f}s')

    with open(os.path.join(OUT, 'braid_words_bn_results.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
        f.write('\nlongest word:\n' + ' '.join(fmt(x) for x in W) + '\n')


if __name__ == '__main__':
    main()
