# Progress since PR #14

PR #14 was merged on 2026-11-08. This file lists what has been added since then. Each result also has an entry in `NEXT_STEPS.md`, together with the job that produced it.

## New results

### Item F: Chair44 hull (3D)

- **Cohomology.** We computed the Čech cohomology of the hull: dim Ȟ^k(Ω; ℚ) = 1, 3, 17, 21 for k = 0..3, so the Euler characteristic is −6 (`research/chair44_cohomology.py`).
- **Handedness.** We looked for a 3D analogue of the Spectre's handedness (item A): a pseudoscalar X_k, i.e. a number that rotations leave unchanged and that a mirror would negate. X_k is computed in exact integer arithmetic and is exactly 0 for k = 0..7, up to 14.7M cells (`research/chair44_handedness.py`).
- **Why X_k is zero.** The recorded correlation tensor has the form T = −P T Pᵀ, where P swaps x and y. This points to a mirror symmetry of the supertiles.
- **Direct mirror check.** At every level k = 0..6 (up to 1.8M cells), the level-k supertile equals its own x ↔ y mirror image, with tile frames conjugated to match. This was tested as exact set equality (`research/chair44_mirror.py`).
- **Item F is closed.** The Chair44 hull is achiral, unlike the Spectre. To turn this into a proof, one would show that the substitution commutes with the mirror on the 3888 collared tile types.

### Z₂ deconfinement on the Spectre graph

- We added a three-size mode that runs Spectre levels 3, 4 and 5 on a narrow coupling grid (`research/z2_deconfinement.py spectre5`).
- The Binder cumulant U jumps at β ≈ 0.0862 for N = 4401 and at β ≈ 0.0855 for N = 34649.
- The curves never cleanly cross, so our estimate β_c ≈ 0.0855 ± 0.0007 is heuristic.
- As a control, the square lattice reproduces the 3D Ising value: β_c = 0.2229, against 0.2217 for 3D Ising.

### R9: gap labels

- The vertex Hamiltonian has no gaps in the tile-level spectrum.
- It has stable gap edges at IDS 0.315/0.316 (label [−5, 6]) and at 0.684/0.685 (label [6, −6]). Both are the same at levels 3 and 4.

### R8: π-flux Hamiltonian

- The main spectral gap survives the vortex flux.
- All Bott indices are zero, as time-reversal symmetry requires.

## Still open

- **R7, genus sweep of `spectreseal`:** still running. The earlier runs failed because a dependency was missing; this run installed it successfully. The result will be added to this PR.
- **R8 with complex π/2 flux:** not yet started.
- **Item B:** parked.
