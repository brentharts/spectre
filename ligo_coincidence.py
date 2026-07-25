#!/usr/bin/env python3
"""
ligo_coincidence.py -- the two-detector test the criterion called for:
|z| > 4 at lag ln(lambda), with a secondary peak at 2*ln(lambda), in BOTH
H1 and L1 over the same 500 s span (GPS 1266624018-1266624518).

Adds a ligo4-style cross-spectral check: the coherence between H1 and L1
evaluated at the would-be comb frequencies f0*lambda^n versus control
frequencies -- a real astrophysical comb must be coherent between sites,
instrumental lines generally are not.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, coherence

import ligo_spectre_search as _lss
import ligo_realdata as _lrd

LAMBDA_EXACT = _lss.LAMBDA_EXACT
log_periodic_scan = _lss.log_periodic_scan
inject_spectre_comb = _lss.inject_spectre_comb
load_cached = _lrd.load_cached
surrogate_acfs = _lrd.surrogate_acfs
z_profile = _lrd.z_profile

OUT = os.environ.get('EINSTEIN3D_OUT', '/tmp')
RNG = np.random.default_rng(3)


def valid_segments(x, fs, min_len_s=32):
    """Contiguous non-NaN stretches of at least min_len_s seconds."""
    ok = ~np.isnan(x)
    r = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
    starts, ends = np.where(r == 1)[0], np.where(r == -1)[0]
    return [(s, e) for s, e in zip(starts, ends)
            if (e - s) / fs >= min_len_s]


def nan_welch(x, fs, nperseg):
    """Welch PSD averaged over valid segments, weighted by length."""
    segs = valid_segments(x, fs)
    acc, wsum, fref = None, 0.0, None
    for s, e in segs:
        f, P = welch(x[s:e], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        w = (e - s)
        acc = P * w if acc is None else acc + P * w
        wsum += w
        fref = f
    return fref, acc / wsum


def nan_coherence(xh, xl, fs, nperseg):
    """Coherence averaged over segments valid in BOTH detectors."""
    both = xh + xl                      # NaN wherever either is NaN
    segs = valid_segments(both, fs)
    acc, wsum, fref = None, 0.0, None
    for s, e in segs:
        f, C = coherence(xh[s:e], xl[s:e], fs=fs, nperseg=nperseg,
                         noverlap=nperseg // 2)
        w = (e - s)
        acc = C * w if acc is None else acc + C * w
        wsum += w
        fref = f
    return fref, acc / wsum


def detector_scan(x, fs, lam_grid, lam):
    f, P = nan_welch(x, fs, int(8 * fs))
    lags, acf, resid, _ = log_periodic_scan(f, P, fmin=20, fmax=1500)
    sur = surrogate_acfs(resid, 200)
    zprof = z_profile(lags, acf, sur, lam_grid)
    z1 = z_profile(lags, acf, sur, [lam])[0]
    z2 = z_profile(lags, acf, sur, [lam ** 2])[0]   # lag 2 ln lambda
    return dict(f=f, P=P, lags=lags, acf=acf, zprof=zprof, z1=z1, z2=z2)


def main():
    lam = LAMBDA_EXACT
    lam_grid = np.linspace(1.8, 3.6, 181)

    data = {}
    for det in ('H1', 'L1'):
        ts, path = load_cached(det)
        if ts is None:
            raise SystemExit(f'{det} not found in cache')
        fs = float(ts.sample_rate.value)
        x = np.asarray(ts.value, dtype='float64')
        nanfrac = np.isnan(x).mean()
        segs = valid_segments(x, fs)
        print(f'{det}: {os.path.basename(str(path))}  {len(x)/fs:.0f} s @ '
              f'{fs:.0f} Hz  nan={100*nanfrac:.1f}%  '
              f'valid segments: {[(round(s/fs), round(e/fs)) for s, e in segs]}')
        data[det] = (x, fs)

    res = {det: detector_scan(x, fs, lam_grid, lam)
           for det, (x, fs) in data.items()}
    for det in ('H1', 'L1'):
        r = res[det]
        imax = int(np.argmax(np.abs(r['zprof'])))
        print(f"[{det}] z(ln lam)={r['z1']:+.2f}  z(2 ln lam)={r['z2']:+.2f}  "
              f"max|z|={abs(r['zprof'][imax]):.2f} at lam={lam_grid[imax]:.3f}")

    passed = all(abs(res[d]['z1']) > 4 and abs(res[d]['z2']) > 2
                 for d in res)
    print(f'coincidence criterion (|z|>4 at ln lam AND secondary at 2 ln lam '
          f'in both): {"PASSED" if passed else "NOT MET (null)"}')

    # where do the two detectors' z profiles agree?
    joint = res['H1']['zprof'] * res['L1']['zprof']
    j = int(np.argmax(joint))
    print(f'max joint z_H1*z_L1 = {joint[j]:+.2f} at lambda = {lam_grid[j]:.3f} '
          f'(needs both large AND same sign to be interesting)')

    # ---- inter-site coherence at comb frequencies (ligo4-style CSD) ------
    (xh, fsh), (xl, fsl) = data['H1'], data['L1']
    n = min(len(xh), len(xl))
    fC, Cxy = nan_coherence(xh[:n], xl[:n], fsh, int(8 * fsh))
    f0 = 27.0
    comb_f = [f0 * lam ** k for k in range(6) if f0 * lam ** k < 1500]

    def coh_at(fq, hbw=0.5):
        m = np.abs(fC - fq) <= hbw
        return float(Cxy[m].mean()) if m.any() else np.nan

    comb_coh = [coh_at(fq) for fq in comb_f]
    ctrl_coh = [coh_at(fq) for fq in RNG.uniform(30, 1400, 40)]
    print(f'coherence at comb freqs: mean {np.nanmean(comb_coh):.4f}  '
          f'controls: mean {np.nanmean(ctrl_coh):.4f} '
          f'sd {np.nanstd(ctrl_coh):.4f}')

    # ---- coincident injection validation ---------------------------------
    zi = {}
    for det, (x, fs) in data.items():
        xi_in = np.nan_to_num(x, nan=0.0)      # zeros only used for std calc
        inj, freqs = inject_spectre_comb(xi_in, fs, lam, f0=f0)
        inj[np.isnan(x)] = np.nan              # restore gap
        f, P = nan_welch(inj, fs, int(8 * fs))
        lags, acf, resid, _ = log_periodic_scan(f, P, fmin=20, fmax=1500)
        sur = surrogate_acfs(resid, 100)
        zi[det] = z_profile(lags, acf, sur, [lam])[0]
    print(f"[injection] z_H1={zi['H1']:+.2f}  z_L1={zi['L1']:+.2f}  "
          f"-> coincidence criterion recoverable: "
          f"{all(abs(v) > 4 for v in zi.values())}")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    phi2 = ((1 + np.sqrt(5)) / 2) ** 2

    ax = axes[0, 0]
    for det, col in (('H1', 'navy'), ('L1', 'firebrick')):
        ax.plot(lam_grid, res[det]['zprof'], lw=1.1, color=col, label=det)
    ax.axvline(lam, color='seagreen', ls='--', label='spectre $\\lambda$')
    ax.axvline(phi2, color='darkorange', ls='--', label='hat $\\varphi^2$')
    ax.axhspan(-4, 4, color='grey', alpha=0.15)
    ax.set_xlabel('$\\lambda$'); ax.set_ylabel('z')
    ax.set_title('z($\\lambda$) profiles, both detectors')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(lam_grid, joint, lw=1.1, color='purple')
    ax.axvline(lam, color='seagreen', ls='--')
    ax.axhline(16, color='crimson', ls=':',
               label='z=4 in both (product = 16)')
    ax.set_xlabel('$\\lambda$'); ax.set_ylabel('$z_{H1} \\cdot z_{L1}$')
    ax.set_title('joint statistic (same-sign coincidence)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(fC[fC > 15], Cxy[fC > 15], lw=0.5, color='k')
    for fq in comb_f:
        ax.axvline(fq, color='seagreen', ls=':', alpha=0.7)
    ax.set_xlabel('f (Hz)'); ax.set_ylabel('H1-L1 coherence')
    ax.set_title(f'inter-site coherence; dotted = comb $f_0\\lambda^n$\n'
                 f'comb mean {np.nanmean(comb_coh):.4f} vs controls '
                 f'{np.nanmean(ctrl_coh):.4f}$\\pm${np.nanstd(ctrl_coh):.4f}')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for det, col in (('H1', 'navy'), ('L1', 'firebrick')):
        r = res[det]
        ax.plot(r['lags'], r['acf'], lw=0.8, color=col, label=det, alpha=0.8)
    for m in (1, 2, 3):
        ax.axvline(m * np.log(lam), color='crimson', ls='--', alpha=0.5)
    ax.set_xlim(0, 3.5 * np.log(lam))
    ax.set_xlabel('lag in ln f'); ax.set_ylabel('ACF of whitened log-PSD')
    ax.set_title('real-data ACFs, dashed = n ln($\\lambda$)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT, 'ligo_coincidence.png')
    fig.savefig(png, dpi=140)
    print('wrote', png)

    with open(os.path.join(OUT, 'ligo_coincidence_results.txt'), 'w') as fh:
        for det in ('H1', 'L1'):
            fh.write(f"{det}: z(ln lam)={res[det]['z1']:+.3f}  "
                     f"z(2 ln lam)={res[det]['z2']:+.3f}\n")
        fh.write(f'criterion: {"PASSED" if passed else "NOT MET (null)"}\n')
        fh.write(f'max joint z product = {joint[j]:+.3f} at '
                 f'lambda={lam_grid[j]:.3f}\n')
        fh.write(f'coherence comb {np.nanmean(comb_coh):.4f} vs controls '
                 f'{np.nanmean(ctrl_coh):.4f}+-{np.nanstd(ctrl_coh):.4f}\n')
        fh.write(f"injection: z_H1={zi['H1']:+.2f} z_L1={zi['L1']:+.2f}\n")


if __name__ == '__main__':
    main()
