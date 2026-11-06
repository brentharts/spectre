# Progress since 2cc5491

The branch `next-steps-plan` is based on Brent's master (bddcf47), which is the same as 2cc5491 except for `boundary_certificate/window_boundary_closure.py`. That file goes in a separate PR.

## What was added

| File | What it does |
|---|---|
| `NEXT_STEPS.md` | Ten research directions, each with its literature status and the experimental results. |
| `spectrepaper.py` | Selftest fixes: the section-order and bibliography checks now pass. |
| `research/doublers.py` | Computes the Harris–Luck wandering exponent; tests for fermion doublers on Spectre, square and random lattices. |
| `research/harris_luck.py` | Computes the exact characteristic polynomial of the 9×9 substitution matrix and the exponent ω. |
| `research/local_dim.py` | Box-counting estimate of the window boundary's dimension and how uniform it is. |
| `research/swirl_exact.py` | Computes Z_k = Σ conj(p_i−c)e^{iθ_i} exactly by enumerating every tile, levels 1–6. |
| `research/swirl_recursion.py` | Exact moment recursion (N, S, M0, M1 per supertile label) that gives Z_k at any level without enumerating tiles; checked against enumeration for k ≤ 5 and run to k = 16. |
| `research/swirl_closedform.py` | Fits Z_k to its closed form exactly and checks the fit on every level. The version with the conjugate terms has not been run yet. |

## How it helps the paper

1. **Correct attribution.** Pure point spectrum, the cut-and-project window, the cohomology (Ȟ¹ = ℤ⁴, Ȟ² = ℤ¹⁰) and the boundary dimension log(5+2√6)/log(4+√15) are already proved in Baake–Gähler–Mazáč–Sadun, arXiv:2411.15503. The paper has to cite these and must not present them as new conjectures. Related work on the Hat, Ising, percolation and Lean is also listed for citation.
2. **The swirl claim is now stated precisely.**
   - The normalised swirl decays to 0, so it is not a nonzero invariant.
   - The exact quantity Z_k has Im Z_k < 0 at every level k = 1..16, where level 16 has 2.5·10¹⁴ tiles.
   - Z_k ≈ (Ak+B)(4+√15)^k + …, which comes from a 2×2 Jordan block at λ². The robust handedness invariant is therefore Im A < 0, with Im A ≈ −3.78 (approximate).
   - The paper can now replace "negative at every measured level" with this asymptotic statement. Once A is known in closed form, that statement can become a proof.
3. **Two claims can be stated as negative results.**
   - Ising: ω = 0, which is below the Harris–Luck threshold of 1/2. Spectre aperiodicity does not change the 2D Ising universality class; at most it adds logarithmic corrections.
   - Doublers: the naive Dirac operator shows no doubler signal specific to the Spectre. On the Spectre it behaves like a random lattice.
4. **Mistakes avoided.** Point-cloud box counting and the brute-force contact closure both failed to give clean answers: the closure passed 2M types without saturating. Both failures are documented, so the paper does not rely on them, and the redesign is to use the graph-directed IFS from the literature.
5. **The paper builds again.** `spectrepaper.py --selftest` passes.

## Still open

- **Closed form for A:** rerun the closed-form fit with the (4−√15)^k terms included, then derive A exactly from the 9-label system.
- **Local dimension:** compute it along ∂W from the graph-directed IFS, which answers the uniformity question.
- **Items 7 and 8:** the Galois explanation of √2+√3, and a redesigned operator for the doubler test.
