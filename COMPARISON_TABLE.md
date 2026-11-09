# Spectre vs. periodic controls (item I)

Each row is an observable already computed in this project. "n/c" means the control has not been computed yet. Sources are the entries in `NEXT_STEPS.md`.

| Observable | Spectre | Square lattice | Honeycomb | Depends on the monotile? |
|---|---|---|---|---|
| Z₂ holonomy vortex density | 4 − √15 per spectre (≈ 0.0212 per vertex), exact | 0 (no holonomy defects: flat lift) | 0 (flat lift) | **Yes**: the vortices exist only because of the Spectre's reflected tiles |
| Valence-4 vertex share (bulk) | 2.95%; density exactly 1 − √15/5 per spectre | 100% | 0% | **Yes**: intermediate, and an irrational density in the gap-labelling module |
| Densities in gap-labelling module ℤ + ℤ(5+√15)/10 | all (vortices, valence 3/4, vertices, links) | trivially rational | trivially rational | **Yes** (by construction of the module) |
| Plaquette type | all 14-gons, uniform strong-coupling area | 4-gons | 6-gons | No (uniform in all three) |
| β_c, Z₂ deconfinement | ≈ 0.0855 ± 0.0007 (heuristic, no clean crossing) | 0.2229 (3D Ising 0.2217) | 0.1614 | No: β_c·(z̄+2) ≈ 1.3 for all three (z̄ = dual-lattice coordination), so the shift is explained by coordination |
| Bott indices, π-flux Hamiltonian | all 0 | 0 (time-reversal symmetry) | 0 (time-reversal symmetry) | **No**: forced by time-reversal symmetry |
| Main spectral gap under vortex flux | survives | n/a (no vortices) | n/a | n/a |
| Gap labels (vertex Hamiltonian) | stable edges at IDS 0.315/0.316 ([−5, 6]) and 0.684/0.685 ([6, −6]) | band edges only, IDS ∈ {0, 1} | IDS 1/2 (Dirac point, no gap) | **Yes**: labels from the Spectre's K-theory |
| Ribbon linking vs. twist sign (R6) | zero writhe; no sign change | n/a | n/a | No arrow of time either way |
| Mirror-type symmetry | PC holds with axis-relative twist; maps hull to the mirror hull (G4) | hull achiral | hull achiral | **Yes**: the Spectre hull is chiral |
| H1 graph/Euclidean dilation (r→∞; far-field anisotropy) | 1.50 (1.507 K=5, 1.499 K=6); anisotropy 0.06 | 1.273; 0.22 (lattice directions) | 1.273; 0.078 | Value yes (tile geometry); limit exists and is the most isotropic of the three |
| H2 disc entanglement S(R)=aR+bR·lnR+c (K=4 size); R² of ln I vs graph / Euclid distance | a=4.73, b=−0.74; R² 0.35 / 0.35 | a=3.62, b=+0.47; R² 0.08 / 0.11 | a=3.38, b=−0.39; R² 0.49 / 0.51 | Area law in all three; sign of the R·lnR correction differs (negative for Spectre and honeycomb). Graph vs Euclid distance indistinguishable (follows from H1) |

## Reading

- The monotile matters for **combinatorial and K-theoretic** quantities: vortex density, valence mix, gap labels, chirality of the hull.
- It does **not** matter for **symmetry-protected** quantities: Bott indices vanish for every graph with time-reversal symmetry.
- β_c follows coordination: with the honeycomb control (0.1614, same fine-scan job 1712312), β_c·(z̄+2) ≈ 1.3 on Spectre, square and honeycomb alike. The open question is only the jump-without-crossing on the Spectre (possible weak first order or boundary effect).
