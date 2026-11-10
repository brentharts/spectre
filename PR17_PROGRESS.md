# Progress in PR #17: Γ6 curvature and Γ8 sheared-module indexing

This PR adds two research threads.
- **Γ6:** Ollivier–Ricci curvature of the Spectre graph compared with the holonomy vortices.
- **Γ8:** indexing the vortex Bragg peaks in a sheared ℤ[ζ₁₂] Fourier module.

Each item cites the result JSON it comes from. Every result is recorded in `NEXT_STEPS.md`, together with the open questions.

## Γ6: curvature of the Spectre graph
- Vortices are negatively curved. The disc-level partial correlation with holonomy winding is −0.6 to −0.75.
- The effect survives control by per-degree counts on disjoint discs, with confidence intervals. The Regge comparison with holonomy winding is included.
- Chair44 comparison: Ollivier curvature on the tile graph of a planar Chair44 cut. The cut uses the layer with the most L-trominoes.

## Γ8: sheared-module indexing of the vortex diffraction
- **Sheared module.** In lattice coordinates (n = V/28 ∈ ℤ⁴), the internal image x* = Σ nⱼ ζ⁵ʲ fits x* = M x + b + r with r bounded. This is a linear deformation, with no quadratic term (≤ 0.1% of the residual variance, `quad_fit_v1`). With k = k₅ + Mᵀk*, the peak height is the window transform |⟨exp(i k*·r)⟩|².
- **What it indexes.** It indexes the 6.0925 and 7.937 vortex families: at K4 the matches are within 5e-4 at h ≈ 0.49 (`sheared_*`, `cross_fit_v1`, `galois_fit_v1`).
- **Vortex window.** The vortices form a model subset with about half the window radius: 1.64–1.75, against 3.75–3.77 for all sites. The shear M is the same.
- **ζ⁷ adds nothing.** It equals ζ⁵ up to a reflection of internal space, which M absorbs (`galois_fit_v1`).
- **Odd K is the physical mirror of even K.**
  - M flips sign with K parity. Swapping M between patches removes every peak.
  - Every K5 point set has S = 0 at the K4 peaks, and reproduces them at conj(k) (`supertile_index_v1`).
  - Seeded refinement gives K5 peaks at the exact conjugate angles, with magnitudes within 1.4e-3 (`refine_seeded_v1`).
  - So the substitution reverses handedness at each level.
- **The 6.0925 family is a level-2 supertile superlattice peak.** L2 supertile centres give h 0.655–0.67, and single tiles about 0.05 (`supertile_index_v1`).
- **Correction.** The coarse 361-point scan in `module_fit.py` misses peaks at K5 because they are narrower than its grid step. `refine_seeded_v1` supersedes `module_fit_k5_v1`.

## Open
- **7.7295**, the strongest vortex family (h 0.70 at K4, 0.73 at K5 mirrored), is not indexed by any test so far:
  - it is not a point of the linear sheared module, with or without a quadratic correction;
  - no Galois map or vortex window indexes it;
  - it is not a tile or supertile superlattice peak (h ≤ 0.17);
  - its ratio to 6.0925 (1.26869) is not a small ℚ(√3) number.
- **The ± families are weaker at odd K** (h 0.32 against 0.49), while 7.7295 is not.

All heavy runs were batch jobs. Scripts take `OUT.json K [K ...]` and run with `python research/<script>.py`.

