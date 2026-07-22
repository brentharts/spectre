#!/usr/bin/env python3
"""
ligo_realdata.py -- the spectre lambda search on REAL L1 strain, merged with
the nariai/ligo4.py slice-wise methodology.

Analyses
--------
1. Full-span lambda search: Welch PSD -> whitened log-f spectrum -> ACF;
   z-score at lag ln(lambda), lambda = sqrt(4+sqrt(15)), against 200
   phase-randomised surrogates.  Also the full z(lambda) PROFILE over
   lambda in [1.8, 3.6] from the same surrogate set, with the spectre value
   and the hat value phi^2 marked.
2. ligo4-style slice scan (32 s slices): per-slice PSD entropy (ligo4's
   statistic) AND per-slice acf(ln lambda), to see whether any transient
   interval drives the full-span statistic.
3. ligo4's Nariai jitter line: f = c / 1.3e6 m = 230.6 Hz; excess-power
   z-score in a narrow band around f_target vs flanking sidebands.
4. Injection validation ON REAL NOISE: spectre comb f0*lambda^n injected
   into the real strain, pipeline must recover it.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import entropy
from scipy.ndimage import median_filter

from ligo_spectre_search import (LAMBDA_EXACT, log_periodic_scan,
                                 inject_spectre_comb)

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
RNG = np.random.default_rng(5)
C_LIGHT = 299792458.0
R_NARIAI = 1.3e6
F_TARGET = C_LIGHT / R_NARIAI          # ligo4's predicted jitter, ~230.6 Hz

CACHE ='ligo_cache'
from gwpy.timeseries import TimeSeries

def get_ligo_data(detector, start, end, rate=4096, cache_dir=CACHE):
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{detector}_{start}_{end}.h5")
    if os.path.exists(cache_file):
        print('loading from cache')
        return TimeSeries.read(cache_file)
    print('downloading...')
    data = TimeSeries.fetch_open_data(detector, start, end, sample_rate=rate)
    data.write(cache_file, overwrite=True)
    return data

def load_cached(detector='L1', gps_start=1266624018, duration=500, fs=4096):
    ts = get_ligo_data(detector, gps_start, gps_start+duration, fs)
    return ts, CACHE

# ---------------------------------------------------------------------------
def surrogate_acfs(resid, n_surrogates=200):
    """ACF curves of phase-randomised surrogates (computed once, reused for
    every lambda in the profile)."""
    n = len(resid)
    F = np.fft.rfft(resid)
    acfs = np.empty((n_surrogates, n))
    for s in range(n_surrogates):
        ph = np.exp(2j * np.pi * RNG.random(len(F)))
        ph[0] = 1.0
        r = np.fft.irfft(F * ph, n=n)
        a = np.correlate(r, r, mode='full')[n - 1:]
        acfs[s] = a / max(a[0], 1e-30)
    return acfs


def z_profile(lags, acf, sur_acfs, lam_grid):
    zs = []
    mu = sur_acfs.mean(axis=0)
    sd = sur_acfs.std(axis=0)
    for lam in lam_grid:
        i = int(np.argmin(np.abs(lags - np.log(lam))))
        zs.append((acf[i] - mu[i]) / max(sd[i], 1e-12))
    return np.array(zs)


def line_excess(f, P, f0, half_bw=0.5, side=8.0):
    """z-score of mean PSD in [f0-hbw, f0+hbw] vs flanking sidebands."""
    inb = (np.abs(f - f0) <= half_bw)
    sb = ((np.abs(f - f0) > 2 * half_bw) & (np.abs(f - f0) <= side))
    if inb.sum() < 1 or sb.sum() < 10:
        return np.nan
    return (P[inb].mean() - P[sb].mean()) / (P[sb].std() / np.sqrt(inb.sum()))


# ---------------------------------------------------------------------------
def main():
    ts, path = load_cached('L1')
    if ts is None:
        raise SystemExit('no cached L1 data found')
    fs = float(ts.sample_rate.value)
    x = np.asarray(ts.value, dtype='float64')
    dur = len(x) / fs
    print(f'loaded {path}: {dur:.0f} s at {fs:.0f} Hz')
    h1, _ = load_cached('H1')
    if h1 is None:
        raise RuntimeError('[note] H1 not in cache -> single-detector run; the '
              'coincidence criterion needs an H1 slice of the same span')

    lam = LAMBDA_EXACT
    ln_lam = np.log(lam)

    # ---- 1) full-span lambda search --------------------------------------
    f, P = welch(x, fs=fs, nperseg=int(8 * fs), noverlap=int(4 * fs))
    lags, acf, resid, g = log_periodic_scan(f, P, fmin=20, fmax=1500)
    sur = surrogate_acfs(resid)
    lam_grid = np.linspace(1.8, 3.6, 181)
    zprof = z_profile(lags, acf, sur, lam_grid)
    i_lam = int(np.argmin(np.abs(lam_grid - lam)))
    z_at = zprof[i_lam]
    phi2 = ((1 + np.sqrt(5)) / 2) ** 2
    z_phi = zprof[int(np.argmin(np.abs(lam_grid - phi2)))]
    print(f'[full span] z(lambda=sqrt(4+sqrt15)) = {z_at:+.2f}   '
          f'z(phi^2) = {z_phi:+.2f}   '
          f'max |z| on grid = {np.abs(zprof).max():.2f} at '
          f'lambda={lam_grid[np.argmax(np.abs(zprof))]:.3f}')

    # ---- 2) slice scan (ligo4 style) -------------------------------------
    slice_s = 32
    t_marks, ents, z_slices = [], [], []
    for t0 in range(0, int(dur) - slice_s + 1, slice_s):
        seg = x[int(t0 * fs):int((t0 + slice_s) * fs)]
        fседа = None
        fseg, Pseg = welch(seg, fs=fs, nperseg=int(4 * fs),
                           noverlap=int(2 * fs))
        band = (fseg >= 20) & (fseg <= 1500)
        ent = entropy(Pseg[band] / Pseg[band].sum())
        try:
            lg, ac, rs, _ = log_periodic_scan(fseg, Pseg, fmin=20, fmax=1500,
                                              nlog=2048, med_win=101)
            i = int(np.argmin(np.abs(lg - ln_lam)))
            zsl = ac[i] / (np.std(ac[len(ac) // 2:]) + 1e-12)
        except RuntimeError:
            zsl = np.nan
        t_marks.append(t0); ents.append(ent); z_slices.append(zsl)
        print(f'slice {t0:3d}s | entropy {ent:.4f} | acf(ln lam)/tail-sd '
              f'{zsl:+.2f}')

    # ---- 3) Nariai jitter line -------------------------------------------
    z_line = line_excess(f, P, F_TARGET)
    # context: same statistic at 30 random control frequencies
    ctrl = [line_excess(f, P, fc) for fc in RNG.uniform(60, 1400, 30)]
    ctrl = np.array([c for c in ctrl if np.isfinite(c)])
    print(f'[nariai line] f_target = {F_TARGET:.2f} Hz  '
          f'excess z = {z_line:+.2f}  '
          f'(controls: mean {ctrl.mean():+.2f}, sd {ctrl.std():.2f})')

    # ---- 4) injection on real noise --------------------------------------
    inj, freqs = inject_spectre_comb(x, fs, lam, f0=27.0)
    f2, P2 = welch(inj, fs=fs, nperseg=int(8 * fs), noverlap=int(4 * fs))
    lags2, acf2, resid2, _ = log_periodic_scan(f2, P2, fmin=20, fmax=1500)
    sur2 = surrogate_acfs(resid2, 100)
    z_inj = z_profile(lags2, acf2, sur2, [lam])[0]
    print(f'[injection into real L1] z(ln lambda) = {z_inj:+.2f}  '
          f'lines at {[f"{q:.1f}" for q in freqs]} Hz')

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.loglog(f[f > 10], P[f > 10], lw=0.5, color='k')
    ax.axvline(F_TARGET, color='crimson', ls='--',
               label=f'Nariai jitter c/r = {F_TARGET:.1f} Hz (z={z_line:+.1f})')
    for n in range(6):
        fn = 27.0 * lam ** n
        if fn < 1500:
            ax.axvline(fn, color='seagreen', ls=':', alpha=0.6)
    ax.set_xlabel('f (Hz)'); ax.set_ylabel('PSD')
    ax.set_title(f'L1 real strain PSD ({dur:.0f} s)\n'
                 'green dotted: where a spectre comb f0*lambda^n would sit')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(lam_grid, zprof, lw=1.2, color='navy')
    ax.axvline(lam, color='seagreen', ls='--',
               label=f'spectre $\\sqrt{{4+\\sqrt{{15}}}}$: z={z_at:+.2f}')
    ax.axvline(phi2, color='darkorange', ls='--',
               label=f'hat $\\varphi^2$: z={z_phi:+.2f}')
    ax.axhspan(-4, 4, color='grey', alpha=0.15, label='|z|<4')
    ax.set_xlabel('$\\lambda$'); ax.set_ylabel('z at lag ln $\\lambda$')
    ax.set_title('log-periodic comb profile z($\\lambda$), real L1 data')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(t_marks, ents, 'o-', color='teal', label='PSD entropy (ligo4)')
    ax2.plot(t_marks, z_slices, 's-', color='orchid',
             label='acf(ln$\\lambda$)/tail sd')
    ax.set_xlabel('slice start (s)'); ax.set_ylabel('entropy', color='teal')
    ax2.set_ylabel('slice comb statistic', color='orchid')
    ax.set_title(f'{slice_s} s slice scan: entropy vs comb statistic')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(lags, acf, lw=0.8, color='navy', label='real L1')
    ax.plot(lags2, acf2, lw=0.8, color='seagreen', alpha=0.8,
            label=f'L1 + injected comb (z={z_inj:+.1f})')
    for m in (1, 2, 3):
        ax.axvline(m * ln_lam, color='crimson', ls='--', alpha=0.5)
    ax.set_xlim(0, 3.5 * ln_lam)
    ax.set_xlabel('lag in ln f'); ax.set_ylabel('ACF of whitened log-PSD')
    ax.set_title('data vs injection, dashed = n*ln($\\lambda$)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT, 'ligo_realdata.png')
    fig.savefig(png, dpi=140)
    print('wrote', png)

    with open(os.path.join(OUT, 'ligo_realdata_results.txt'), 'w') as fh:
        fh.write(f'data: {os.path.basename(path)}  ({dur:.0f}s @ {fs:.0f}Hz)\n')
        fh.write(f'lambda = {lam:.6f}\n')
        fh.write(f'z(lambda)      = {z_at:+.3f}\n')
        fh.write(f'z(phi^2)       = {z_phi:+.3f}\n')
        fh.write(f'max |z| grid   = {np.abs(zprof).max():.3f} at '
                 f'{lam_grid[np.argmax(np.abs(zprof))]:.3f}\n')
        fh.write(f'nariai line z  = {z_line:+.3f} '
                 f'(controls mean {ctrl.mean():+.2f} sd {ctrl.std():.2f})\n')
        fh.write(f'injection z    = {z_inj:+.3f}\n')
        fh.write('single-detector (L1 only); coincidence pending H1 cache.\n')


if __name__ == '__main__':
    main()
