#!/usr/bin/env python3
"""chair44_findings.py - the numbers and figures of the section on unfolding the Chair44.

Everything here is computed, not typed: chair44_unfold.py (nets of the surface),
chair44_slice.py (slices, projection, centres and notch vectors) and, for the Spectre side,
spectre.py through spectre_figures.level_tiles.  chair44facts.py turns the results into Facts
and spectrepaper.py quotes only those.  The Spectre numbers are floating point sums over placed
tiles (level 5 has 34649); the chair numbers are exact integers.

    python3 chair44_findings.py            # print the numbers, write the three figures
    python3 chair44_findings.py --check    # assertions only
"""
import itertools
import math
import os
import random
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chair44_unfold as U          # noqa: E402
import chair44_slice as SL          # noqa: E402
import spectre as S                 # noqa: E402

LAMBDA = math.sqrt(4.0 + math.sqrt(15.0))
ELL_CHAIR = 7.0 ** (1.0 / 3.0)
_CACHE = {}


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


# ------------------------------------------------------------------ nets
def nets_data(n=3000, seed=1):
    def run():
        import sympy as sp
        faces, edges, fold = U.surface()
        verts = {v for f in faces for v in U._corners(f)}
        trees = int(U.count_spanning_trees(faces, edges))
        sv = U.survey(n, seed)
        per = Counter(v['perimeter'] for v in sv['stats'].values())
        cf, ce, _ = U.surface([(0, 0, 0)])
        cube_trees = sum(1 for _ in U.all_trees(ce, 6))
        cube = set()
        for t in U.all_trees(ce, 6):
            c = U.unfold(cf, ce, t)
            if U.is_valid(c):
                cube.add(U.canonical(c))
        return dict(faces=len(faces), edges=len(edges), vertices=len(verts),
                    euler=len(verts) - len(edges) + len(faces), folds=dict(Counter(fold.values())),
                    trees=trees, samples=n, seed=seed, valid=sv['valid'], overlapping=sv['overlapping'],
                    distinct=sv['distinct'], tilers=sv['tilers'], min_perimeter=min(per),
                    perimeters=dict(sorted(per.items())), cube_trees=cube_trees, cube_nets=len(cube),
                    survey=sv, surface=(faces, edges, fold))
    return cached(('nets', n, seed), run)


# ------------------------------------------------------------------ slices
def slice_data(depths=(1, 2, 3, 4)):
    def run():
        out = dict(depths=depths, distinct=[], coset_even=[], coset_odd=[], notch_sum=[],
                   interface=[], interface_formula=[], cross_sections=set(), proj_ok=None)
        for d in depths:
            P = SL.centres(d)
            out['distinct'].append(len({tuple(p) for p in P}) == len(P) == 8 ** d)
            ev = int(np.sum(np.all(P % 2 == 0, axis=1)))
            out['coset_even'].append(ev)
            out['coset_odd'].append(int(np.sum(np.all(P % 2 == 1, axis=1))))
            out['notch_sum'].append(tuple(int(x) for x in SL.notches(d).sum(axis=0)))
            out['interface'].append(SL.interface_edges(SL.slice_layer(d, 2, 0)))
            out['interface_formula'].append(4 ** (d + 1) - 4 * 2 ** d)
        for c in range(8):
            for ax in range(3):
                lay = SL.slice_layer(2, ax, c)
                for i in np.unique(lay[lay >= 0]):
                    out['cross_sections'].add(int((lay == i).sum()))
        P = SL.centres(2)
        M = 16
        vol = np.zeros((M, M, M))
        np.add.at(vol, tuple(P.T), 1.0)
        F3 = np.fft.fftn(vol)
        out['proj_ok'] = bool(np.allclose(F3[:, :, 0], SL.structure_plane(P, None, M, 2, 0.0)) and
                              np.allclose(F3[:, :, 3], SL.structure_plane(P, None, M, 2, 2 * np.pi * 3 / M)))
        out['cross_sections'] = sorted(out['cross_sections'])
        return out
    return cached(('slice', depths), run)


# ------------------------------------------------------------------ the Spectre side
def tile_centroid():
    P = S.get_spectre_points(1.0, 1.0)
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2.0
    return np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)]), abs(A)


def spectre_points(level):
    """(centroids (N,2), rotation angles (N,)) of the level-n patch built by spectre.py.  The placements
    of spectre.py are all mirror-type (det -1); the angle is that of the rotation part, which is the same
    for every tile, so |sum exp(i theta)| is unaffected by the global mirror."""
    def run():
        T = S.buildSpectreTiles(level, 1.0, 1.0, rotation=30)
        ms = []
        T["Delta"].forEachTile(lambda M, l: ms.append(np.array(M, float)))
        ms = np.array(ms)
        assert len({int(np.sign(d)) for d in np.linalg.det(ms[:, :2, :2])}) == 1
        c, _ = tile_centroid()
        P = ms[:, :, :2] @ c + ms[:, :, 2]
        return P, np.arctan2(ms[:, 1, 0], ms[:, 0, 0])
    return cached(('sp', level), run)


def orientation_table(levels=(2, 3, 4, 5)):
    def run():
        rows = {'spectre': [], 'chair': []}
        for n in levels:
            _, th = spectre_points(n)
            s = abs(np.exp(1j * th).sum())
            rows['spectre'].append(dict(n=n, N=len(th), S=s, ratio=s / LAMBDA ** n, i0=s ** 2 / len(th)))
            ns = SL.notches(n)
            sc = int(ns.sum(axis=0)[2])
            rows['chair'].append(dict(n=n, N=len(ns), S=sc, ratio=sc / 2.0 ** n, i0=sc ** 2 / len(ns)))
        return rows
    return cached(('orient', levels), run)


def random_k_median(points, weights, ell, n_k=300, seed=1):
    d = points.shape[1]
    K = np.random.default_rng(seed).uniform(0, 2 * np.pi / ell, (n_k, d))
    v = np.abs(np.exp(1j * (K @ points.T)) @ weights) ** 2 / len(points)
    return float(np.median(v))


def power_table(levels=(2, 3, 4, 5), n_k=300, seed=1):
    def run():
        _, area = tile_centroid()
        ell_s = math.sqrt(area)
        out = {}
        for name in ('spectre', 'spectre_weighted', 'chair', 'chair_weighted'):
            ns, med = [], []
            for n in levels:
                if name.startswith('spectre'):
                    P, th = spectre_points(n)
                    w = np.exp(1j * th) if name.endswith('weighted') else np.ones(len(P))
                    ell = ell_s
                else:
                    P = SL.centres(n).astype(float)
                    w = SL.notches(n)[:, 2].astype(float) if name.endswith('weighted') else np.ones(len(P))
                    ell = ELL_CHAIR
                ns.append(len(P))
                med.append(random_k_median(P, w, ell, n_k, seed))
            out[name] = dict(N=ns, median=med,
                             slope=float(np.polyfit(np.log(ns), np.log(med), 1)[0]))
        return out
    return cached(('power', levels, n_k, seed), run)


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def fig_nets():
    import spectre_figures as SF
    plt = _style()
    from matplotlib.patches import Rectangle
    d = nets_data()
    sv = d['survey']
    faces, edges, fold = d['surface']
    nets = sv['nets']
    keys = sorted(nets, key=lambda k: (sv['stats'][k]['perimeter'], k))
    tiler = next(k for k in keys if U.tiles_by_translation(k))
    mid = keys[len(keys) // 2]
    picks = [(keys[0], 'smallest perimeter'), (tiler, 'a net that tiles the plane'), (mid, 'a median net')]
    fig = plt.figure(figsize=(11.5, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
    pal = {0: SF.BLUE, 1: SF.ORANGE, 2: SF.AQUA}
    for i, (k, name) in enumerate(picks):
        ax = fig.add_subplot(gs[i])
        c, _ = nets[k]
        for f, (x, y) in c.items():
            a, sd, _ = faces[f]
            ax.add_patch(Rectangle((x, y), 1, 1, fc=pal[a], alpha=0.35 + 0.35 * (sd > 0), ec=SF.INK, lw=0.5))
        xs = [x for x, _ in c.values()]
        ys = [y for _, y in c.values()]
        ax.set_xlim(min(xs) - .2, max(xs) + 1.2)
        ax.set_ylim(min(ys) - .2, max(ys) + 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        st = sv['stats'][k]
        ax.set_title('%s\nperimeter %d, %d$\\times$%d' % (name, st['perimeter'], st['bbox'][0], st['bbox'][1]),
                     fontsize=8)
    ax = fig.add_subplot(gs[3])
    per = d['perimeters']
    ax.bar(list(per), list(per.values()), width=1.6, color=SF.BLUE)
    ax.set_xlabel('perimeter of the net')
    ax.set_ylabel('valid nets (of %d sampled trees)' % d['samples'])
    SF._clean(ax)
    fig.tight_layout()
    SF._save(fig, 'chair44_nets')


def fig_slice():
    import spectre_figures as SF
    plt = _style()
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.7))
    lay = SL.slice_layer(3, 2, 5).astype(float)
    lay[lay < 0] = np.nan
    ax[0].imshow((lay % 20).T, origin='lower', cmap='tab20', interpolation='nearest')
    ax[0].set_title('(a) depth-3 slice $z=5$', fontsize=9)
    ax[0].axis('off')
    P = SL.centres(5)
    I = np.fft.fftshift(SL.intensity(P, None, 512))
    ax[1].imshow(np.maximum(I, 1e-3).T, origin='lower', cmap='magma', norm=LogNorm(1e-3, I.max()),
                 extent=(-np.pi, np.pi, -np.pi, np.pi))
    ax[1].set_title('(b) $k_z=0$ plane, level 5, $\\log I/N$', fontsize=9)
    n = np.arange(1, 6)
    ax[2].semilogy(n, 64.0 ** n, color=SF.INK, label='centres, Bragg: $64^{n}$')
    ax[2].semilogy(n, 3.0 * 8.0 ** n, ':', color=SF.GREY, label='random signs: $3\\cdot8^{n}$')
    ax[2].semilogy(n, 3.0 * 4.0 ** n, color=SF.ORANGE, label='notch vectors: $3\\cdot4^{n}$')
    ax[2].set_xticks(n)
    ax[2].set_xlabel('level $n$')
    ax[2].set_title('(c) $|F(0)|^{2}$', fontsize=9)
    ax[2].legend(fontsize=7, frameon=False)
    SF._clean(ax[2])
    ds = (2, 3, 4, 5)
    for name, wf, col in (('unweighted', None, SF.INK), ('notch $x$', lambda d: SL.notches(d)[:, 0].astype(float), SF.BLUE),
                          ('notch $z$', lambda d: SL.notches(d)[:, 2].astype(float), SF.ORANGE)):
        st = SL.random_k_stats(ds, wf, 300)
        ax[3].semilogy(ds, [st[d][1] for d in ds], 'o-', color=col, label=name)
    ax[3].set_xticks(ds)
    ax[3].set_xlabel('level $n$')
    ax[3].set_title('(d) median $I/N$, 300 random $k$', fontsize=9)
    ax[3].legend(fontsize=7, frameon=False)
    SF._clean(ax[3])
    fig.tight_layout()
    SF._save(fig, 'chair44_slice')


def fig_compare():
    import spectre_figures as SF
    plt = _style()
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))
    rows = orientation_table()
    n = [r['n'] for r in rows['spectre']]
    ax[0].plot(n, [r['ratio'] for r in rows['spectre']], 'o-', color=SF.BLUE,
               label='Spectre: $|\\sum e^{i\\theta}|/\\lambda^{n}$')
    ax[0].plot(n, [r['ratio'] for r in rows['chair']], 's-', color=SF.ORANGE,
               label='Chair44: $\\sum d_{z}/2^{n}$')
    ax[0].set_ylim(0, 2)
    ax[0].set_xticks(n)
    ax[0].set_xlabel('level $n$')
    ax[0].set_title('(a) orientation sum over the linear inflation$^{n}$', fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    SF._clean(ax[0])
    ax[1].semilogy(n, [r['i0'] for r in rows['spectre']], 'o-', color=SF.BLUE, label='Spectre ($d=2$)')
    ax[1].semilogy(n, [r['i0'] for r in rows['chair']], 's-', color=SF.ORANGE, label='Chair44 ($d=3$)')
    ax[1].set_xticks(n)
    ax[1].set_xlabel('level $n$')
    ax[1].set_title('(b) $|S|^{2}/N$', fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)
    SF._clean(ax[1])
    t = power_table()
    for name, st, col in (('spectre', '-', SF.BLUE), ('spectre_weighted', '--', SF.BLUE),
                          ('chair', '-', SF.ORANGE), ('chair_weighted', '--', SF.ORANGE)):
        ax[2].loglog(t[name]['N'], t[name]['median'], st, marker='o', color=col,
                     label='%s, slope %.2f' % (name.replace('_', ' '), t[name]['slope']))
    ax[2].set_xlabel('tiles $N$')
    ax[2].set_title('(c) median $I/N$ at 300 random $k$', fontsize=9)
    ax[2].legend(fontsize=6.5, frameon=False)
    SF._clean(ax[2])
    fig.tight_layout()
    SF._save(fig, 'chair44_compare')


def check():
    d = nets_data(600, 1)
    assert (d['faces'], d['edges'], d['vertices'], d['euler']) == (24, 48, 26, 2)
    assert d['cube_nets'] == 11 and d['cube_trees'] == 384
    s = slice_data()
    assert all(s['distinct']) and s['proj_ok'] and s['interface'] == s['interface_formula']
    assert s['notch_sum'] == [(2 ** n,) * 3 for n in s['depths']]
    o = orientation_table((2, 3))
    assert all(r['ratio'] == 1.0 for r in o['chair']) and abs(o['spectre'][1]['ratio'] - 1.448) < 0.01
    return True


def main():
    if '--check' in sys.argv:
        print('chair44_findings: ok' if check() else 'FAIL')
        return
    d = nets_data()
    print({k: v for k, v in d.items() if k not in ('survey', 'surface')})
    print(slice_data())
    print(orientation_table())
    print(power_table())
    fig_nets()
    fig_slice()
    fig_compare()


if __name__ == '__main__':
    main()
