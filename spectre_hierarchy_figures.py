#!/usr/bin/python3
"""Figures for the hierarchy computations (computation, not a Fact).

Four figures in paper/figures/ (PNG and PDF), each drawn from the same functions the --check of its module runs:

    spectre_hier_clustering   Barbour shape complexity of the tile centroids against uniform controls  (barbour_complexity)
    spectre_hier_records      error in naming the level-k ancestor against the neighbourhood radius     (records_recognizability)
    spectre_hier_lookback     common-ancestor level against distance, Spectre against a quadtree         (lookback_lca)
    spectre_hier_parity       tile statistics do not alternate with level, supertile outlines do        (parity_clock, outline_chirality)

Every panel carries its control or its uncertainty.  Colours are the Okabe-Ito colour-blind-safe set, and series are
also told apart by marker or line style.

    python3 spectre_hierarchy_figures.py            full run (a few minutes, the first run builds hierarchy_cache/)
    python3 spectre_hierarchy_figures.py --check    light settings, writes to a temporary directory, asserts the data
"""
import contextlib
import io
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import hierarchy_cache as HC
import barbour_complexity as BC
import records_recognizability as RR
import lookback_lca as LB
import parity_clock as PC
import outline_chirality as OC

BLUE, ORANGE, GREEN, VERMILION, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#666666'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 120, 'savefig.dpi': 200,
                     'legend.frameon': False})


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def save(fig, outdir, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, '%s.%s' % (name, ext)), bbox_inches='tight')
    plt.close(fig)
    return name


# ---------------------------------------------------------------------------------------------- 1. clustering
def fig_clustering(outdir, levels):
    with quiet():
        rows = BC.table(levels)
    L = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax[0].axhline(BC.DISC_LIMIT, color=GREY, ls=':', lw=1)
    ax[0].text(L[-1], BC.DISC_LIMIT - 0.012, 'uniform disc, N→∞: %.4f' % BC.DISC_LIMIT, ha='right', va='top', color=GREY, fontsize=8)
    ax[0].plot(L, [r[3] for r in rows], 's--', color=ORANGE, mfc='none', label='uniform lattice, same footprint')
    ax[0].plot(L, [r[4] for r in rows], '^:', color=GREEN, mfc='none', label='Poisson, same footprint')
    ax[0].plot(L, [r[2] for r in rows], 'o-', color=BLUE, label='Spectre tile centroids')
    ax[0].set_xlabel('hierarchy level n  (N = %s tiles)' % ', '.join(str(r[1]) for r in rows))
    ax[0].set_ylabel(r'shape complexity $C_S=\ell_{rms}/\ell_{mhl}$')
    ax[0].set_xticks(L)
    ax[0].legend(loc='lower right', fontsize=8)
    ax[0].set_title('The rise is the footprint, not the tiling', fontsize=10)
    ex = [r[2] - r[3] for r in rows]
    ax[1].axhline(0, color=GREY, lw=1)
    ax[1].bar(L, ex, color=BLUE, width=0.6)
    ax[1].set_xticks(L)
    ax[1].set_xlabel('hierarchy level n')
    ax[1].set_ylabel('Spectre − lattice on the same footprint')
    ax[1].set_title('Excess clustering over uniform', fontsize=10)
    for x, y in zip(L, ex):
        ax[1].annotate('%+.4f' % y, (x, y), ha='center', va='bottom' if y >= 0 else 'top', fontsize=8,
                       xytext=(0, 3 if y >= 0 else -3), textcoords='offset points')
    fig.tight_layout()
    save(fig, outdir, 'spectre_hier_clustering')
    return rows


# ---------------------------------------------------------------------------------------------- 2. records
def fig_records(outdir, depth, M, radii, levels):
    with quiet():
        out = RR.run(depth, M, radii=radii, levels=levels)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    cols, marks = (BLUE, VERMILION, GREEN), ('o', 's', '^')
    for j, k in enumerate(levels):
        y = np.array([out[r][1][j][0] for r in radii])
        ax.plot(radii, y, marks[j] + '-', color=cols[j], label='level k = %d ancestor' % k)
    ax.axhline(0.10, color=GREY, ls=':', lw=1)
    ax.text(radii[0], 0.115, '10% error', color=GREY, fontsize=8)
    ax.set_xscale('log')
    ax.set_xticks(radii)
    ax.set_xticklabels(['%g' % r for r in radii])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r'neighbourhood radius $r/\ell$ (tile scales)')
    ax.set_ylabel('leave-one-out error naming the ancestor')
    ax.set_title('Each coarser level needs a patch about 3× larger\n(depth %d, %d tiles; error among covered tiles)' % (depth, len(HC.load(depth)['pos'])), fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, outdir, 'spectre_hier_records')
    return out


# ---------------------------------------------------------------------------------------------- 3. look-back
def lookback_data(P, nobs, lam, margin, dmax):
    d, h, _ = LB.collect(P, nobs, dmax, margin)
    x = np.log(d) / math.log(lam)
    edges = np.array([1.0, 1.5, 2.2, 3.3, 5.0, 7.5, 11.0])
    c, m, s = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (d >= a) & (d < b)
        if k.sum() > 200:
            c.append(x[k].mean()), m.append(h[k].mean()), s.append(h[k].std())
    surv, _ = LB.tail_ratios(d, h, P['D'], 1.0, 2.0)
    return np.array(c), np.array(m), np.array(s), np.array(surv), len(d)


def fig_lookback(outdir, depth, nobs):
    S = lookback_data(HC.load(depth), nobs, HC.LAMBDA, 40.0 if depth > 5 else 20.0, 17.0)
    Q = lookback_data(HC.quadtree(6), nobs, 2.0, 12.0, 10.0)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    xs = np.linspace(0, 3.2, 10)
    ax[0].plot(xs, xs + (S[1][0] - S[0][0]), color=GREY, ls=':', lw=1, label='slope 1')
    ax[0].errorbar(Q[0], Q[1], Q[2], fmt='s--', color=ORANGE, mfc='none', capsize=2, label='quadtree control (λ = 2)')
    ax[0].errorbar(S[0], S[1], S[2], fmt='o-', color=BLUE, capsize=2, label='Spectre (λ = 2.806)')
    ax[0].set_xlabel(r'$\log_\lambda\, d/\ell$   (distance in e-folds of the hierarchy)')
    ax[0].set_ylabel('common-ancestor level h  (mean ± sd)')
    ax[0].set_title('Mean law: distance is look-back, on average', fontsize=10)
    ax[0].legend(fontsize=8, loc='upper left')
    ks = np.arange(1, len(S[3]) + 1)
    ax[1].semilogy(ks, S[3], 'o-', color=BLUE, label='Spectre')
    ax[1].semilogy(np.arange(1, len(Q[3]) + 1), Q[3], 's--', color=ORANGE, mfc='none', label='quadtree')
    ax[1].semilogy(ks, 0.538 ** (ks - 1), color=BLUE, ls=':', lw=1, label=r'$(\nu/\lambda^2)^{k-1}$ = 0.538$^{k-1}$')
    ax[1].semilogy(ks, 0.5 ** (ks - 1), color=ORANGE, ls=':', lw=1, label='0.5$^{k-1}$')
    ax[1].set_xlabel('level k')
    ax[1].set_ylabel(r'$P(h\geq k)$ for pairs 1 to 2 $\ell$ apart')
    ax[1].set_title('Tail: neighbours can be several levels apart', fontsize=10)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, outdir, 'spectre_hier_lookback')
    return S, Q


# ---------------------------------------------------------------------------------------------- 4. parity
def fig_parity(outdir, depths, nobs, outline_depth):
    near = 'near [1,3) ell'
    pair = {}
    with quiet():
        for d in depths:
            pair[d] = PC.pair_means(HC.load(d), nobs, margin=20.0)[0]
        sw = PC.swirl(HC.load(max(depths)))
        ctrl, rows = OC.report(outline_depth)
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.7))
    # (a) P11 by common-ancestor level
    w = 0.8 / len(depths)
    for i, (d, col) in enumerate(zip(depths, (BLUE, ORANGE, GREEN))):
        ks = sorted(pair[d][near])
        ax[0].bar(np.array(ks) + (i - (len(depths) - 1) / 2) * w, [1000 * pair[d][near][k][1][0] for k in ks], w,
                  yerr=[1000 * pair[d][near][k][2][0] for k in ks], color=col, capsize=2, label='depth %d' % d)
    ax[0].axhline(0, color=GREY, lw=1)
    ax[0].set_xlabel('common-ancestor level h of the pair')
    ax[0].set_ylabel(r'mean $\sin(\theta_o+\theta_t-2\phi)$ × 1000')
    ax[0].set_title('Tile pairs: one sign change, no alternation', fontsize=10)
    ax[0].legend(fontsize=8)
    # (b) swirl
    ks = sorted(sw)
    ax[1].semilogy(ks, [-1000 * sw[k][1] for k in ks], 'o-', color=BLUE)
    ax[1].set_xlabel('supertile level k')
    ax[1].set_ylabel(r'$-\langle\sin(\theta-\phi)\rangle$ × 1000   (log)')
    ax[1].set_xticks(ks)
    ax[1].set_title('Swirl about supertile centres: negative at every level', fontsize=10)
    # (c) outline distances
    for dk, ls, lab in ((1, '-', 'k → k+1'), (2, '--', 'k → k+2')):
        kk = sorted({k for (_, k, d_, _, _) in rows if d_ == dk})
        p = [np.mean([r[3] for r in rows if r[2] == dk and r[1] == k]) for k in kk]
        m = [np.mean([r[4] for r in rows if r[2] == dk and r[1] == k]) for k in kk]
        ax[2].plot(kk, p, 'o' + ls, color=BLUE, label='rotation only, ' + lab)
        ax[2].plot(kk, m, 's' + ls, color=VERMILION, mfc='none', label='rotation + mirror, ' + lab)
    ax[2].set_xlabel('starting level k')
    ax[2].set_ylabel('outline-signature distance (smaller = closer)')
    ax[2].set_title('Supertile outlines: mirror fits one level up,\nrotation two levels up (clear at k = 3, weak at k = 2, 4)', fontsize=10)
    ax[2].legend(fontsize=7)
    ax[2].set_xticks(sorted({k for (_, k, _, _, _) in rows}))
    fig.tight_layout()
    save(fig, outdir, 'spectre_hier_parity')
    return pair, sw, rows


# ---------------------------------------------------------------------------------------------- driver
def build(outdir, light):
    os.makedirs(outdir, exist_ok=True)
    if light:
        return dict(clust=fig_clustering(outdir, range(1, 4)),
                    rec=fig_records(outdir, 5, 4000, (1.5, 2, 3, 4, 8), (1, 2)),
                    lb=fig_lookback(outdir, 6, 400),
                    par=fig_parity(outdir, (5, 6), 500, 6))
    return dict(clust=fig_clustering(outdir, range(1, 6)),
                rec=fig_records(outdir, 6, 12000, (1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32), (1, 2, 3)),
                lb=fig_lookback(outdir, 7, 1500),
                par=fig_parity(outdir, (5, 6, 7), 1500, 7))


NAMES = ('spectre_hier_clustering', 'spectre_hier_records', 'spectre_hier_lookback', 'spectre_hier_parity')


def check():
    ok = True

    def tick(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print('  %-66s %s %s' % (name, 'ok' if cond else 'FAIL', detail))

    print('spectre_hierarchy_figures --check')
    with tempfile.TemporaryDirectory() as tmp:
        R = build(tmp, light=True)
        tick('four figures written as PNG and PDF, each non-trivial in size',
             all(os.path.getsize(os.path.join(tmp, n + e)) > 5000 for n in NAMES for e in ('.png', '.pdf')))
    rows = R['clust']
    tick('clustering: complexity rises with level and stays below the disc limit + 0.05',
         all(a[2] < b[2] for a, b in zip(rows, rows[1:])) and rows[-1][2] < BC.DISC_LIMIT + 0.05, str([round(r[2], 3) for r in rows]))
    tick('records: level-1 error is lower than level-2 error at 3 ell', R['rec'][3][1][0][0] < R['rec'][3][1][1][0])
    S, Q = R['lb']
    tick('look-back: mean level rises with distance for both hierarchies', np.all(np.diff(S[1]) > 0) and np.all(np.diff(Q[1]) > 0))
    pair, sw, orows = R['par']
    tick('parity: swirl negative for k = 1..4', all(sw[k][1] < 0 for k in (1, 2, 3, 4)))
    tick('parity: outline mirror wins for k -> k+1 on average, rotation for k -> k+2',
         np.mean([r[4] < r[3] for r in orows if r[2] == 1]) > 0.7 and np.mean([r[3] < r[4] for r in orows if r[2] == 2]) > 0.7)
    return ok


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(0 if check() else 1)
    out = os.path.join(HERE, 'paper', 'figures')
    build(out, light=False)
    for n in NAMES:
        print('wrote %s.png / .pdf in %s' % (n, out))
