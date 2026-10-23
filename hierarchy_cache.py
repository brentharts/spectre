#!/usr/bin/python3
"""The Spectre patch with its full ancestry, cached, plus the helpers the hierarchy computations share.

Everything in the five modules built on this one (barbour_complexity, records_recognizability, lookback_lca,
parity_clock, outline_chirality) is a COMPUTATION on a finite patch, not a Fact: none of it enters the ledger.

What is stored for a depth-D patch (one row per tile, N = 1, 9, 71, 559, 4401, ... for D = 0, 1, ...):

    pos    (N, 2)  tile centroid, in edge-length units
    ang    (N,)    orientation: atan2 of the first column of the placement matrix
    det    (N,)    +1 / -1: handedness of the placement
    slots  (N, D)  ROOT-FIRST child index taken at each level, i.e. the tile's address in the hierarchy
    labs   (N, D)  label of the supertile entered at each level (labs[:, D-k] is the level-k ancestor's type)
    leaf   (N,)    the tile's own label
    ell    float   tile scale, sqrt(tile area), used to express distances in tile units

The base Gamma metatile (Gamma1 + Gamma2) is treated as transparent: its two tiles share one address, so two tiles
with identical ``slots`` are siblings and have no common-ancestor level.

HANDEDNESS.  In the raw construction every tile of a depth-D patch has det = (-1)^D (the substitution mirrors at every
level), so a depth-6 patch and a depth-7 patch are mirror images tile for tile.  ``normalised`` reflects x -> -x when
needed so that patches of different depth can be compared; any statistic that is odd under reflection is only
meaningful after that.

    python3 hierarchy_cache.py --depth 6      build and cache a patch (about 7 s; depth 7 about 1 min, 2.1M tiles)
    python3 hierarchy_cache.py --check        counts, addresses, handedness alternation, normalisation
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.environ.get('SPECTRE_HIERARCHY_CACHE', os.path.join(HERE, 'hierarchy_cache'))

LAMBDA = math.sqrt(4 + math.sqrt(15))      # linear inflation factor per level
NU = 2 + math.sqrt(5)                      # outline-edge growth per level

# the substitution rule as spectre.buildSupertiles writes it (None = no child in that slot)
SUBST = {"Gamma": ("Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma"),
         "Delta": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
         "Theta": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
         "Lambda": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
         "Xi": ("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
         "Pi": ("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
         "Sigma": ("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"),
         "Phi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
         "Psi": ("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma")}


def build(depth):
    """Walk the MetaTile tree keeping the address of every tile.  Returns the dict described in the module docstring."""
    import spectre as S
    import tile_family as TF
    td = S.buildSpectreTiles(depth, 1.0, 1.0, rotation=30)
    canon = np.asarray(TF.canonical_tile_verts('any'), float)[:14]
    x, y = canon[:, 0], canon[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    area = cr.sum() / 2
    cen = np.array([((x + x1) * cr).sum() / (6 * area), ((y + y1) * cr).sum() / (6 * area)])
    pos, ang, det, leaf, chains = [], [], [], [], []

    def walk(node, label, T, chain, level):
        if isinstance(node, S.Tile):
            pos.append((T @ np.array([*cen, 1.0]))[:2])
            ang.append(math.atan2(T[1, 0], T[0, 0]))
            det.append(1.0 if T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0] > 0 else -1.0)
            leaf.append(node.label)
            chains.append(chain)
        elif level == 0:                       # the base Gamma metatile: Gamma1 and Gamma2 share an address
            for c, t in zip(node.tiles, node.transformations):
                walk(c, None, S.mul(T, t), chain, -1)
        else:
            kids = [s for s in SUBST[label] if s]
            for i, (c, t) in enumerate(zip(node.tiles, node.transformations)):
                walk(c, kids[i], S.mul(T, t), chain + ((label, i),), level - 1)

    walk(td['Delta'], 'Delta', S.IDENTITY, (), depth)
    return dict(pos=np.array(pos), ang=np.array(ang), det=np.array(det),
                slots=np.array([[s for (_, s) in c] for c in chains], dtype=np.int8).reshape(len(chains), depth),
                labs=np.array([[l for (l, _) in c] for c in chains]).reshape(len(chains), depth),
                leaf=np.array(leaf), ell=float(math.sqrt(abs(area))), D=int(depth))


def load(depth, rebuild=False):
    """The cached patch of this depth, building it first if needed (cache dir is git-ignored)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'patch_d%d.npz' % depth)
    if rebuild or not os.path.exists(path):
        P = build(depth)
        np.savez_compressed(path, **P)
    Z = np.load(path)
    P = {k: Z[k] for k in Z.files}
    P['ell'] = float(P['ell'])
    P['D'] = int(P['D'])
    return P


def normalised(P):
    """(pos, ang) reflected, if needed, so that every depth carries the same tile handedness (det = +1)."""
    pos, ang = P['pos'].copy(), P['ang'].copy()
    if P['det'][0] < 0:
        pos[:, 0] *= -1
        ang = np.pi - ang
    return pos, ang


def groups(P, k):
    """Index array: tile -> id of its level-k ancestor (identical address prefixes share an id)."""
    key = P['slots'][:, :P['D'] - k]
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.ravel()


def quadtree(depth=6):
    """Control hierarchy: a 2^D x 2^D grid with the 2x2 quadtree address (lambda = 2, boundary dimension 1)."""
    n = 2 ** depth
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    bits = lambda v: np.array([(v >> s) & 1 for s in range(depth - 1, -1, -1)]).T
    slots = (bits(X.ravel()) * 2 + bits(Y.ravel())).astype(np.int8)
    pos = np.column_stack([X.ravel(), Y.ravel()]).astype(float)
    return dict(pos=pos, ang=np.zeros(len(pos)), det=np.ones(len(pos)), slots=slots, D=depth, ell=1.0)


def iter_pairs(P, pos, nobs, dmax_ell, seed=3, margin_ell=40.0):
    """For ``nobs`` random interior observers yield (o, idx, h, d, phi): the other tiles within dmax_ell tile scales,
    their common-ancestor level h (1 = same level-1 supertile), distance d in tile units and the direction phi from
    the observer.  Sibling tiles (identical address) are skipped.  Interior = farther than margin_ell from the
    patch edge, so no neighbourhood is cut off."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    ell, D, slots = P['ell'], P['D'], P['slots']
    tree = cKDTree(pos)
    ctr = pos.mean(0)
    rad = np.linalg.norm(pos - ctr, axis=1)
    cand = np.where(rad < rad.max() - margin_ell * ell)[0]
    for o in rng.choice(cand, min(nobs, len(cand)), replace=False):
        idx = np.array(tree.query_ball_point(pos[o], dmax_ell * ell))
        idx = idx[idx != o]
        M = slots[idx] != slots[o]
        has = M.any(1)
        idx, M = idx[has], M[has]
        v = pos[idx] - pos[o]
        yield o, idx, D - M.argmax(1), np.hypot(v[:, 0], v[:, 1]) / ell, np.arctan2(v[:, 1], v[:, 0])


# ----------------------------------------------------------------------------------------------------------------
def check(depth=4):
    ok = True

    def tick(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('  %-62s %s %s' % (name, 'ok' if cond else 'FAIL', detail))

    print('hierarchy_cache --check')
    counts = {d: len(load(d)['pos']) for d in range(1, depth + 1)}
    tick('tile counts are 9, 71, 559, 4401', [counts[d] for d in range(1, 5)] == [9, 71, 559, 4401][:depth], str(counts))
    P = load(depth)
    D = P['D']
    tick('every tile has a length-D address', P['slots'].shape == (len(P['pos']), D))
    tick('distinct addresses = 3905 (siblings Gamma1/Gamma2 share one)', depth != 4 or len(np.unique(P['slots'], axis=0)) == 3905)
    blocks = [len(np.unique(groups(P, k))) for k in range(D, 0, -1)]
    tick('supertile counts per level read 1, 8, 63, 496 (the census)', blocks[:4] == [1, 8, 63, 496][:len(blocks[:4])], str(blocks))
    for d in range(2, depth + 1):
        Q = load(d)
        tick('depth %d: handedness uniform and equal to (-1)^depth' % d, len(set(Q['det'])) == 1 and Q['det'][0] == (-1) ** d)
    Q = load(depth)
    pos, ang = normalised(Q)
    tick('normalisation only reflects x (y unchanged, |x| unchanged)',
         np.allclose(pos[:, 1], Q['pos'][:, 1]) and np.allclose(np.abs(pos[:, 0]), np.abs(Q['pos'][:, 0])))
    n_pairs = sum(len(i) for _, i, _, _, _ in iter_pairs(Q, pos, 20, 6.0, margin_ell=10))
    tick('pair sampler returns pairs', n_pairs > 100, '(%d)' % n_pairs)
    return ok


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(0 if check() else 1)
    depth = int(sys.argv[sys.argv.index('--depth') + 1]) if '--depth' in sys.argv else 5
    P = load(depth, rebuild=True)
    print('depth %d: %d tiles cached in %s' % (depth, len(P['pos']), CACHE))
