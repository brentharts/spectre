# Changes since 0948b96

The summary covers branch `chair44-pc-symmetry`. Full details are in `NEXT_STEPS.md` and `COMPARISON_TABLE.md`.

## Proofs (exact, finite checks)
- **Γ14, mirror half:** `research/chair44_mirror_proof.py`. The x↔y mirror commutes with the Chair44 cell substitution on all 168 (frame, cell) labels × 8 octants, with 0 failures, and fixes the seed. So every supertile equals its own mirror image, and the Chair44 hull is achiral.
- **Γ14, decoration half (PC):** `research/chair44_pc_proof.py`. The mirror sends each of the 192 bump/dent features to the opposite height, so P(D) = −D = C(D). Combined with the mirror half, PC maps every decorated supertile to itself.
  - Assumption: every tile carries one decoration, rotated by its frame.
  - The features table `r44_features.csv` comes from brentharts/spectre-monotile-py and is not vendored.
- **G1–G2:** `research/chair44_decoration_symmetry.py`. The decorated tile is chiral but PC-symmetric.
- **G2b:** C alone is not a hull symmetry, so PC is the only mirror-type symmetry.
- **G4:** `research/spectre_pc_twist.py`. The Spectre also has a PC: mirror plus ribbon twist sign, with the sign defined relative to the reflection axis. One structural difference: the Spectre mirror maps the hull to the mirror hull, whereas for Chair44 the mirror maps the hull to itself.

## Numerical results
- **H1:** `research/graph_dilation.py`. Spectre graph distance / Euclidean distance converges to ≈ 1.50 (K=6, 1.67M vertices). Its far-field anisotropy is 0.06, below both the square lattice (0.22) and the honeycomb (0.078). The behaviour is generic, not specific to the Spectre.
- **H2:** `research/entanglement_h2.py`. Disc entanglement follows an area law on the Spectre, the square lattice and the honeycomb.
  - The R·lnR correction is negative for the Spectre and shrinks with size, so it is not a stable signal.
  - Mutual information fits graph distance and Euclidean distance equally well, which H1 predicts.
- **R8 follow-up:** `research/flux_complex.py`. Tight-binding with complex π/2 vortex flux gives Bott index 0 in every gap at K=3,4, and the π control is also 0. This is a negative result, and R8 is closed.
- **Honeycomb β_c control:** 0.1614. β_c·(z̄+2) ≈ 1.3 on all three lattices, so the shift is explained by coordination.

## Docs
- `COMPARISON_TABLE.md` (item I): compares Spectre with periodic controls observable by observable.
- `NEXT_STEPS.md`: changes include
  - K: QUANTUM_GRAVITY open entries turned into tests;
  - J: Bell-constraint paragraph drafted (awaiting a choice from Brent; H1 supports its linearity claim);
  - Γ1–Γ14: long-horizon items. **Only Γ14 is done; Γ1–Γ13 are open.**
