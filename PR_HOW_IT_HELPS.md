# How this PR helps

This PR builds on Brent's ribbon and holonomy work (`spectre_holonomy.py`, `spectrespace.py`, `spectreseal.py`). It turns the Z₂ vertex flux and the lifted ribbons into research results that can be tested. Every result below is recorded in `NEXT_STEPS.md` under "Results 2026-11-08", together with the job that produced it.

## What it adds to the project

1. **It explains what the holonomy vortices are.**
   - The flux −1 vertices are the vortices; all of them have valence 3, and they don't depend on the convention.
   - **Density.** Their share of interior vertices is 0.009, 0.015, 0.018 and 0.019 at depths 3–6 (R1, `research/vortex_density.py`). The exact value is now known: **4 − √15 per spectre** (≈ 0.0212 per vertex), again in the gap-labelling module ℤ + ℤ(5+√15)/10 (`research/vortex_density2.py`).
   - **Window.** Under the star map they fall in a sub-region of the window: a vortex's nearest neighbour is another vortex 27% of the time, against a base rate of about 2% (R2, `research/vortex_window.py`). That is evidence that the vortex set is a cut-and-project set.
2. **It shows what the flux is not.**
   - **Not a local edge assignment.** No choice of edge signs depending on local edge type, out to radius 50, reproduces the flux. The obstruction is a single valence-2 star (R3, `research/flux_cohomology.py`).
   - **Not a source of handedness.** The lifted ribbons have zero writhe, so their linking number equals their twist. That linking number does not change sign with the twist sign, so the ribbons carry no arrow of time (R6, `research/ribbon_linking.py`).
   - Negative results like these keep the paper from overclaiming.
3. **It is a first step from Part I to Part II, the quantum-gravity side.**
   - **SU(2) gauge theory on the Spectre graph** (R11 step 2, `research/su2_graph.py`):
     - every plaquette is a 14-gon, and the strong-coupling area is uniform;
     - 2.95% of bulk vertices have valence 4, against the square-grid and honeycomb controls, which give 100% and 0%.
   - **R11 step 3, spin-network densities. This is now an exact result** (`research/spin_densities.py`).
     - The density of valence-4 vertices, the only ones that carry volume at spin 1/2, is exactly **1 − √15/5 per spectre**.
     - The other densities: valence-3 vertices 2√15/5, vertices 6, and links 7.
     - All of them lie in the gap-labelling module ℤ + ℤ(5+√15)/10. This is a concrete link between the tiling's K-theory and the volume structure of the spin network.
4. **It fixes the existing analysis tools and adds two new ones.**
   - Fixes:
     - The Bott index in `research/flux_hamiltonian.py` is now computed from eigenvalue phases; before this fix it returned NaN.
     - `research/holonomy_group.py` now records quaternion-convention mismatches instead of aborting.
   - **R4 independently confirms Brent's holonomy group:** Dic12 of order 48 with 0 mismatches for every twist sign, and the twist signs are coboundary-equivalent. An earlier report of 96 was an encoding bug in my script, now fixed.
   - New tools:
     - `research/genus_sweep.py` checks how stable the genus of `spectreseal` is.
     - `research/parity_clock.py`: its characteristic polynomial is λ⁵(λ−1)(λ+1)(λ²−8λ+1).
5. **It sharpens the number-theory question (item E).**
   - **The literal conjecture is refuted.** Twisting the integer boundary-word matrix by a Galois automorphism cannot give the seam rate √2+√3, because the matrix's spectrum, {1, 2±√5}, lies in ℚ(√5).
   - **Revised conjecture E′, now also refuted.** E′ said the seam rate comes from the internal contact matrix, whose unit 5+2√6 and area unit 4+√15 lie in subfields of ℚ(λ) = ℚ(√6, √10). But the Hat, the sister tiling, has published dimension log(2+√3)/log φ², and its seam unit 2+√3 is not in ℚ(√5). So E′ cannot hold in the form stated. A candidate replacement, E″, ties the seam unit to the √3 rotation geometry; the proposed test is the Hat-family Tile(a,b). For the paper we drop E′.
6. **Literature check for R11.** We found no earlier gauge-theory or spin-network work on the Spectre or Hat tilings. The closest work is the quasicrystal quantum-gravity papers arXiv:1903.10851 and arXiv:2306.01964, which use other quasicrystals. So the R11 result appears to be new.

## Further results

- **Chair44 cohomology (item F):** dim Ȟ^k(Ω; ℚ) = 1, 3, 17, 21 (Euler characteristic −6). This is the first computed Čech cohomology of the 3D Chair44 hull (`research/chair44_cohomology.py`).
- **Z₂ deconfinement:**
  - The square-lattice control reproduces 3D Ising (β_c = 0.2229 against 0.2217).
  - On the Spectre graph the transition is at β_c ≈ 0.0855 ± 0.0007, from the jump of the Binder cumulant across levels 3–5 (`research/z2_deconfinement.py`). There is no clean crossing, so this is a heuristic estimate.
- **π-flux Hamiltonian:**
  - The main spectral gap survives the vortex flux.
  - All Bott indices vanish, as time-reversal symmetry requires.
- **Chair44 handedness (`research/chair44_handedness.py`):** the natural rotation-invariant pseudoscalar vanishes exactly up to level 7 (14.7M cells), so it shows no chirality. A traceless tensor correlation remains.
- **Gap labels (R9):** the vertex Hamiltonian has four stable gap edges near IDS 0.315 and 0.685 at levels 3 and 4.

## Still running

R7 (genus sweep). The first run failed on a missing dependency and has been rerun. Its result will be added to this PR as a follow-up commit.

## Reproducing

Each script documents its own usage at the top of the file, for example `python research/spin_densities.py 6 out.json`.
