## LIGO Search
Searching for a concrete, falsifiable signature: the spectre substitution's linear
inflation factor, measured to convergence and matched to the substitution
count matrix,

  λ = √(Perron eigenvalue) = √(4+√15) = 2.8058837…,  ln λ = 1.03172

(N.B. **not** φ² = 2.618, which is the hat H-metatile inflation — worth
keeping distinct in the nariai notes.) A hierarchy organised by λ imprints
a log-periodic comb f_n = f0·λⁿ. The statistic: whiten log-PSD by running
median on a uniform log-f grid, autocorrelate, read off the ACF at lag ln λ,
score against phase-randomised surrogates.

```
[data]      z(ln λ) = -0.22   (null, as expected)
[injection] z(ln λ) = +4.80   (pipeline recovers an injected spectre comb)
```

Detection criterion written into the results file: |z| > 4 at ln λ, with a
secondary peak at 2·ln λ, in **both** H1 and L1. On your machine (with
gwosc.org reachable, or with your existing `ligo_cache`), the same script
runs on real strain unchanged.

Implementation gotcha worth knowing: a synthetic f⁻⁸ seismic wall makes the
whole band leakage-dominated and the PSD perfectly monotone, which silently
zeroes the whitened residual. 
