# Spectre vs. periodic controls (item I)

Each row is an observable already computed in this project. "n/c" means the control has not been computed yet. Sources are the entries in `NEXT_STEPS.md`.

| Observable | Spectre | Square lattice | Honeycomb | Depends on the monotile? |
|---|---|---|---|---|
| Z₂ holonomy vortex density | 4 − √15 per spectre (≈ 0.0212 per vertex), exact | 0 (no holonomy defects: flat lift) | 0 (flat lift) | **Yes**: the vortices exist only because of the Spectre's reflected tiles |
| Valence-4 vertex share (bulk) | 2.95%; density exactly 1 − √15/5 per spectre | 100% | 0% | **Yes**: intermediate, and an irrational density in the gap-labelling module |
| Densities in gap-labelling module ℤ + ℤ(5+√15)/10 | all (vortices, valence 3/4, vertices, links) | trivially rational | trivially rational | **Yes** (by construction of the module) |
| Plaquette type | all 14-gons, uniform strong-coupling area | 4-gons | 6-gons | No (uniform in all three) |
| β_c, Z₂ deconfinement | ≈ 0.0855 ± 0.0007 (heuristic, no clean crossing) | 0.2229 (3D Ising 0.2217) | n/c | Unclear: shift is consistent with lower coordination; needs honeycomb control |
| Bott indices, π-flux Hamiltonian | all 0 | 0 (time-reversal symmetry) | 0 (time-reversal symmetry) | **No**: forced by time-reversal symmetry |
| Main spectral gap under vortex flux | survives | n/a (no vortices) | n/a | n/a |
| Gap labels (vertex Hamiltonian) | stable edges at IDS 0.315/0.316 ([−5, 6]) and 0.684/0.685 ([6, −6]) | band edges only, IDS ∈ {0, 1} | IDS 1/2 (Dirac point, no gap) | **Yes**: labels from the Spectre's K-theory |
| Ribbon linking vs. twist sign (R6) | zero writhe; no sign change | n/a | n/a | No arrow of time either way |
| Mirror-type symmetry | PC holds with axis-relative twist; maps hull to the mirror hull (G4) | hull achiral | hull achiral | **Yes**: the Spectre hull is chiral |
| H1 graph/Euclidean dilation | pending | pending | pending | pending |
| H2 mutual information / entanglement | pending | pending | pending | pending |

## Reading

- The monotile matters for **combinatorial and K-theoretic** quantities: vortex density, valence mix, gap labels, chirality of the hull.
- It does **not** matter for **symmetry-protected** quantities: Bott indices vanish for every graph with time-reversal symmetry.
- The β_c value is not yet interpretable: a honeycomb control (coordination 3) is the missing comparison, since the Spectre graph is mostly valence 3.
