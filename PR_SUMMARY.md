# PR 13 — summary of work

## Paper
- Dropped the claim that the swirl is two-signed.
- Restructured the abstract.
- Fixed the document structure and the claims in the pancake sections.
- `spectrepaper.py --selftest` passes again.

## Research (details in `NEXT_STEPS.md`)
| Item | Result | Status |
|---|---|---|
| Earlier experiments | Harris–Luck: ω = 0, so the Ising universality class is unchanged. No doubler signal specific to the Spectre. The normalised swirl decays to 0. Exact moment recursion for Z_k up to k = 16. | done (see `PROGRESS.md`) |
| A. Handedness | Exact swirl slope A = (27/100)(√5+√15)(1−2i), so Im A < 0. Exact Z[ζ₁₂] moment recursion with certified recurrences. | proved |
| D. Bounded displacement | Spectre tile centres are a bounded displacement of a lattice, by Solomon's criterion (\|θ₂\| = 1 < λ). | proved |
| B. Window boundary | Star map identified (it alternates with level parity), and its rotation equivariance settled exactly. Box-counting slope ≈ 1.22, consistent with BGMS 1.111 but biased high. The exact GIFS needs control points from the self-similar version. | parked, heuristic |
| C. Gap labelling | Step 1: the uncollared frequency module is ℤ + ℤ(5+√15)/10 = (5−√15)⁻¹ℤ[√15]. Step 2: tiles labelled by their vertex neighbourhoods give 38 types and no new frequencies (index 1). | steps 1–2 done |

## Still to do
- C step 3: compare the gap positions in the IDS of the tight-binding Hamiltonian with the module above.
- E: a Galois-conjugate formula for seam dimensions.
- F: Chair44, a 3D counterpart.
