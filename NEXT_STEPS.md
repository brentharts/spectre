# Next steps: research directions

These are research bets, not edits to the paper. Each one says what would count as a breakthrough, the first computation to run, and when to stop. Each builds on results the paper already has:
- the 16×16 lift Φ, with characteristic polynomial χ = (x−1)⁴(x+1)⁴(x⁴−8x²+1)²;
- the seam growth rate √2+√3;
- the perimeter formula P(k) = 2F_{3k+5}+4;
- the charges Q_− and the twist θ_S;
- the measured swirl, negative at every level;
- the Lean and certificate pipeline.

## Status after the literature check (2026-11-06)
- **Items 1–3 are already published** in Baake–Gähler–Mazáč–Sadun, arXiv:2411.15503. They prove pure point spectrum, give a 4:2 cut-and-project scheme with Rauzy-fractal windows, compute Ȟ¹ = ℤ⁴ and Ȟ² = ℤ¹⁰, and give the subwindow boundary dimension log(5+2√6)/log(4+√15) in their Eq. (7), via arXiv:2311.03541. The paper must cite this; the boundary dimension is not a conjecture.
- **Still open:** whether the boundary dimension is uniform along ∂W (their open question), and items 5, 7 and 8.
- **Prior work to cite:**
  - item 4: Hat tight-binding, arXiv:2307.11054; polariton monotile, arXiv:2605.29023;
  - item 6: Ising on the Hat, arXiv:2402.11331; percolation, arXiv:2604.21165;
  - item 10: 3D monotile in Lean, arXiv:2609.19214.
- **Result, doubler experiment (2026-11-06):**
  - **Item 6:** the 9×9 substitution matrix has characteristic polynomial x⁵(x−1)(x+1)(x²−8x+1). The second-largest eigenvalue modulus is 1, so the wandering exponent is ω = 0, below the Harris–Luck threshold of 1/2 for the 2D Ising model. The Spectre's aperiodicity is irrelevant there: no new universality class. Φ has 2×2 Jordan blocks at ±1, which predicts at most logarithmic corrections. This direction is closed as a non-breakthrough.
  - **Item 8:** inconclusive, and the diagnostic needs redesign. On the square lattice the Fourier-peak test finds the expected doubler peaks at (0,π), (π,0) and (π,π), but it reports about 9 "species" rather than 4, because the k-grid counts aliased copies of the same peaks. On the Spectre and on the random Delaunay lattice the low modes have no plane-wave structure at all (frac0 ≈ 0.01–0.02 at levels 3–5), so the naive operator has no clean continuum limit there. The Spectre behaves like the random lattice, so there is no Spectre-specific signal. Next: a Christ–Friedberg–Lee-weighted operator (Voronoi/Delaunay dual weights), a staggered/Kähler–Dirac operator on the tiling's cell complex, and an alias-free species count from the index or spectral flow.
- **Result, swirl and local-dimension experiment (2026-11-06):**
  - **Item 5 (swirl):** the swirl tends to 0; it does not have a nonzero limit. The exact linear quantity Z_k = Σ conj(p_i−c)e^{iθ_i} stays in the lower half-plane at k = 1..6 (arg Z_k ≈ −63°), but |Z_k| grows by a factor of about 9.6 per level, while Σ|r| grows by λ³ ≈ 22. So swirl_k decays like roughly 0.44^k (−0.37, −0.46, −0.22, −0.12, −0.049, −0.022). The fitted growth ratio of |Z_k| is still drifting (19.6 → 9.62) and matches no eigenvalue of Φ yet. The open question is now: is sign(Im Z_k) a level-independent invariant? It would need exact ℚ(√15) arithmetic beyond k = 6.
  - **Item 1, uniformity of ∂W:** inconclusive. Box counting on the projected lift gives slopes of 1.83 (level 5) and 1.71 (level 6), not 1.11, and the sector spread is about 0.34 at both levels. Finite-patch saturation dominates, so this estimator cannot test uniformity. Next: compute the local dimension from the graph-directed IFS, not from point clouds.
  - **First closure run failed:** it ran out of disk after 221 minutes, at |live| ≈ 1.88M, and had to be rerun.
- **Result, closure rerun (2026-11-06):** the closure did not close. It passed the 2,000,000-type cap after 230 minutes, growing about 20k types per 1–2 minutes with no sign of saturation. Under the stop rule this is "report and redesign", not more compute. The contact closure is too fine: it tracks exact pairwise placements, and these do not collapse to finitely many classes at this resolution. Redesign: quotient types by translation into a fundamental domain of the 4D lattice, or use the Baake–Gähler–Mazáč–Sadun graph-directed IFS for ∂W directly (arXiv:2411.15503, arXiv:2311.03541). That IFS has finitely many pieces by construction, and the same IFS gives local dimensions for the uniformity question.
- **Item 5, level-7 attempt (2026-11-06):** blocked by tooling, not by compute. `spectre_patch.SpectrePatch` refuses levels above 6 (`spectre_patch.py:188`), so enumerating tiles cannot go further. Per the stop rule, the next step is the exact recursion: for each supertile type keep N (tile count), S = Σp, M0 = Σe^{iθ} and M1 = Σconj(p)e^{iθ}. Children placed by (rotation φ, mirror, translation t) update these linearly, and Z_k = M1 − conj(S/N)·M0. This gives Z_k for any k in ℚ(√3, i) arithmetic, and its recurrence is then read off exactly.

## Proposed directions with significant new mathematics (2026-11-07)
Items 1–3 are published and item 6 is closed, so this section asks what would be *new* mathematics, building on what the project already has: the exact moment recursion, the 16×16 lift Φ, the seam rate √2+√3, the Chair44 2-adic hull theorem, and the Lean/certificate pipeline. They are ordered by expected payoff per unit of effort.

### A. The handedness theorem (finishes item 5)
- **Statement to prove:** for every k ≥ 1, Im Z_k < 0. Moreover Z_k = (Ak+B)λ^{2k} + C(−λ²)^k + (Galois-conjugate terms in (4−√15)^k) + D + E(−1)^k, with A, …, E given in closed form in ℚ(√3, √15, i), and Im A < 0.
- **Why it is new:** it would be the first proved chirality invariant of a tiling space with no mirror images. The Jordan block at λ² is not visible in the 9×9 substitution matrix (which is diagonalisable at λ²); it comes from the coupling between positions and orientations. Explaining that coupling is itself a result.
- **Steps:**
  1. Write the moment recursion as one exact linear map T on (N, S, M0, M1) per label, over ℚ(√3, i)[√15]. Compute its Jordan form symbolically.
  2. Read A off as (left Jordan vector) · (initial moments). Prove Im A < 0 by exact algebra.
  3. Cover the finitely many k where lower terms could flip the sign by the exact values already computed for k ≤ 16, plus an explicit error bound.
  4. Optionally certify the Jordan form and the sign of A in Lean, reusing the certificate pipeline.
- **Stretch:** identify A with a pattern-equivariant cohomology class, that is, show that the linear functional p ↦ A is the projection of a class in Ȟ¹(Ω; ℂ) = ℂ⁴ onto the λ²-eigenspace. That would make "handedness" a topological invariant rather than a numerical one.
- **Stop if:** the conjugate-mode fit (K = 20) still leaves residuals. Then T has a further block, and the exact Jordan form of step 1 replaces fitting.

### B. Local dimension and multifractal spectrum of ∂W (answers the open question of Baake–Gähler–Mazáč–Sadun)
- **Statement:** compute the local dimension d(x) at every point of ∂W from the graph-directed IFS. Either prove it is constant (= log(5+2√6)/log(4+√15)), or compute the full multifractal spectrum f(α) of the natural measure on ∂W.
- **Method:** take the GIFS from arXiv:2311.03541 and arXiv:2411.15503. Check the open set condition by a finite neighbour-graph computation, as in Bandt–Mesing. Get d(x) from the thermodynamic formalism of the strongly connected components of the GIFS graph: if there is more than one component with a different spectral radius, d(x) is non-uniform.
- **Payoff:** a direct answer to a question the Spectre's discoverers left open. This item replaces the failed contact closure; the GIFS is finite by construction.

### C. Gap labelling for the Spectre
- **Statement:** compute the gap-labelling group τ*(K₀(C(Ω) ⋊ ℝ²)) ⊂ ℝ explicitly, as a ℤ-module generated by the patch frequencies. For a 2D tiling this is the image of Ȟ² = ℤ¹⁰ under the frequency map.
- **Prediction to test:** it is ℤ[λ²]-generated, that is, contained in (1/N)ℤ[√15]. Then compare it with the integrated density of states of the tight-binding Hamiltonian of item 4 on level-4/5 patches: every observed gap must sit at a predicted label.
- **Why it is new:** the cohomology is known, but the frequency module and a numerical confirmation of gap labels on a monotile have not been published (check before starting).

### D. Bounded displacement and discrepancy
- **Question:** are Spectre tile centres a bounded displacement of a lattice (BD-equivalent to ℤ²), or does the discrepancy grow?
- **Why the project can answer it:** the discrepancy is controlled by the non-Perron spectrum of the substitution acting on counts and on positions. χ(Φ) has eigenvalues ±1 in 2×2 Jordan blocks, which predicts discrepancy growing like log R for balls of radius R. That would make the Spectre **not** BD-equivalent to a lattice, unlike the Penrose and Ammann–Beenker tilings. The Hat is the natural comparison.
- **First run:** the moment recursion with N alone per label, together with the Jordan structure at ±1, gives exact counts in supertiles. Then prove the log R lower bound along the supertile sequence, following Solomon's criteria for substitution tilings.
- **Payoff:** a clean yes/no theorem, and a structural distinction between the Spectre and the classical cut-and-project tilings.

### E. A Galois-conjugate formula for seam dimensions (sharpens item 7)
- **Observation to explain:** (√2+√3)² = 5+2√6. The seam rate is the Perron root of the boundary-word substitution, and log(5+2√6)/log(4+√15) has a numerator from ℚ(√6) and a denominator from ℚ(√15).
- **Conjecture:** for substitution tilings with a Pisot-type lift, dim(seams) = log ρ(B*)/log λ, where B* is the boundary-word matrix twisted by the Galois automorphism that maps the physical eigenvalue to the internal one.
- **Test set:** Tile(1,1)/Spectre, the Hat, chair, Ammann–Beenker, and Chair44 (where the internal space is 2-adic and the conjecture predicts the answer in ℚ₂). A formula that also covers the 2-adic case would be a theorem of independent interest.

### F. Chair44: a full 3D counterpart
- The project's original 3D result already has a hull theorem and the symmetry groups 1, C₂, C₄, D₄. Missing: pure point spectrum (via 2-adic model set and the overlap algorithm), Ȟ*(Ω) of the 3D hull, and a handedness invariant analogous to A.
- **Payoff:** a complete parallel 2D/3D theory. That is the strongest version of the paper's own Part I comparison.

### Recommended order
1. **A** (days; the code exists) → 2. **D** (reuses A's recursion) → 3. **B** (needs the published GIFS) → 4. **C**, together with item 4 numerics → 5. **E** and **F** (longer).

Each item ends in either a theorem or a documented negative result. For any run over a few minutes, use the cluster.

## 1. The exact fractal geometry of the window boundary
- **Breakthrough:** a proof that dim_H(∂W) = dim_B(∂W) = log(√2+√3)/log λ. This would be the first exact fractal description of the window of an aperiodic monotile.
- **Approach:**
  1. Make the contact-closure graph exact, and certify that its spectral radius equals √2+√3, which gives the upper bound.
  2. Present ∂W as a graph-directed IFS and check the open set condition (Mauldin–Williams). Hausdorff dimension then equals box dimension.
- **Stop if:** the closure's spectral radius is above √2+√3. In that case the boundary is genuinely larger than the seams, and that becomes the new result.

## 2. Is the Spectre hull pure point? (a Pisot-type conjecture for a monotile)
- **Breakthrough:** a proof that the Spectre dynamical system has pure point spectrum, which would make it a model set, with a computable diffraction pattern.
- **Approach:** run the balanced-pair or overlap coincidence algorithm on the substitution. Coincidence is equivalent to pure point spectrum, and a finite overlap graph is a certificate that Lean can check.
- **Unknown:** whether the window is a regular model set (∂W of measure zero). Item 1 settles this.

## 3. Cohomology, K-theory and gap labelling
- **Breakthrough:** compute the Čech cohomology Ȟ*(Ω) of the hull (Anderson–Putnam complex built from the substitution) and the K-theory, and with them the gap-labelling group: every spectral gap of every Schrödinger operator on the Spectre tiling gets a label.
- **Why it matters:** cohomology is a topological invariant of the tiling space. It would show whether the Spectre is "new" or equivalent to a known cut-and-project tiling, and it tests whether Q_− and θ_S are cohomology classes.
- **First run:** build the collared Anderson–Putnam complex and compute ranks over ℤ. This is integer linear algebra.

## 4. Quantum mechanics on the Spectre: critical states and topology
- **Breakthrough:** for the tight-binding Hamiltonian on Spectre vertices or tiles, find:
  - multifractal (critical) eigenstates;
  - the Hofstadter butterfly in a magnetic field;
  - Chern numbers or edge modes inherited from the 4D lift, i.e. a topological phase that exists only because the tiling is chiral.
- **First run:** exact diagonalisation on supertiles of levels 3–5 with periodic approximants, then the level-spacing statistics, the fractal dimension D₂ of the states, and Bott indices.
- **Physics payoff:** a concrete prediction for chiral photonic or acoustic metamaterials.

## 5. The swirl as a topological invariant
- **Breakthrough:** a proof that the measured swirl has a nonzero limit with a fixed sign, given by an exact algebraic number from the Perron eigenvectors of the substitution, and identified with a cohomology class (item 3). This would be a rigorous "handedness charge" for a tiling with no mirror images.
- **First run:** compute the swirl exactly at levels 1–8 in ℚ(√15) arithmetic, fit a linear recurrence, and prove the recurrence from the substitution matrix.

## 6. Critical behaviour of statistical models on the Spectre
- **Breakthrough:** decide whether Spectre aperiodicity changes the universality class of the 2D Ising, Potts and dimer models.
  - The Harris–Luck criterion turns on the wandering exponent ω = log|λ₂|/log λ₁, which can be read off χ.
  - If ω is at the marginal value, the outcome is new.
- **First run:** compute ω exactly. Then run Monte Carlo and transfer-matrix studies of the Ising model on level-4/5 patches, and measure ν and the specific-heat exponent against the Onsager values.

## 7. Why √2+√3? Number theory of the lift
- **Breakthrough:** a structural explanation of why the seams grow like √2+√3, which lies in ℚ(√2,√3), while λ² = 4+√15 lies in ℚ(√15) (λ itself lies in ℚ(√6,√10)), and the factor x⁴−8x²+1 appears in χ.
- **Approach:** use the Galois action on Φ to explain this. The goal is a general theorem: seam dimension = log(largest Galois-twisted eigenvalue)/log λ, for all substitution tilings in a class.
- **Test:** check it on the Hat, the Tile(1,1) family and the chair. If the same formula predicts the other tilings' seam dimensions, it is a theorem worth a paper of its own.

## 8. Lattice gauge theory on a monotile lattice
- **Breakthrough:** discretise SU(2) or SU(3) Yang–Mills on the Spectre tiling, a lattice with no translational symmetry and no mirror images, and measure whether:
  - the cubic-lattice artefacts disappear (restoring rotation invariance faster);
  - chirality yields a lattice θ-term or affects fermion doubling.
- **Link:** this connects the Spectre work to the Yang–Mills repository.
- **First run:**
  1. Wilson-loop area law and string tension in 2D and 3D (Spectre × ℤ).
  2. Count fermion doublers for a naive Dirac operator.

  A missing doubler would be the headline result.

## 9. Gravity, made testable
- **Breakthrough:** replace the pancake narrative with a model whose predictions can be checked: Regge calculus or causal dynamical triangulations restricted to Spectre-derived 2+1 triangulations.
  - Compute the spectral dimension and the Hausdorff dimension of the emergent geometry.
  - Check whether the scale-dependent spectral dimension tends to dim ∂W or to 2.
- **Stop if:** it reproduces standard CDT numbers. That is a null result; publish it briefly and drop the direction.

## 10. Machine-checked aperiodicity
- **Breakthrough:** a full Lean proof that the Spectre tiles only aperiodically.
  - The existing pipeline (SpectreFacts.lean, the certificates) already checks the finite part.
  - What is missing is the combinatorial-hierarchy argument.
- **Payoff:** the first formally verified aperiodic monotile, which would also make items 1–3 checkable by machine.

## Order and resources
- **Start now:** item 1 (it continues the contact-closure run) and item 3. Both are cheap exact computations, and their results feed items 2, 5 and 7.
- **In parallel:** items 4 and 6, which are numerical.
- **Later:** items 8 and 9 are long-horizon, and item 10 needs a dedicated effort.
- **Result, swirl recursion (2026-11-06): item 5 resolved numerically to k = 16.**
  - The moment recursion (`research/swirl_recursion.py`, 120-digit arithmetic, 1.4 s) reproduces the enumerated Z_k exactly for k ≤ 5, then gives Z_k for k ≤ 16 (2.5·10¹⁴ tiles at k = 16).
  - Im Z_k < 0 at every k = 1..16, with arg Z_k → about −63°, oscillating with decreasing amplitude.
  - Z_k satisfies an order-6 linear recurrence, with relative residual 1e−11. Its roots are λ² = 4+√15 (double), −λ², +1, −1 and a spurious near-0 root. The drifting |Z| ratio is explained: Z_k ≈ (A k + B) λ^{2k} + C(−λ²)^k + …, so the ratio tends to λ² like (k+1)/k.
  - The double root means a 2×2 Jordan block at λ² in the moment recursion. The sign of Im Z_k is eventually that of Im A < 0.
  - Next: identify A in closed form in ℚ(√3, √15, i) from the 9-label linear system. That proves the sign for all k (the finite range is covered by this computation). Then the swirl normalised by Σ|r| ~ k λ^{2k} / λ^{3k} still → 0, so the invariant is Im A, not the swirl itself.
- **Result, swirl closed-form fit (2026-11-06):** the 5-term form (Ak+B)λ^{2k} + C(−λ²)^k + D + E(−1)^k fitted exactly on k = 11..15 does **not** reproduce the other levels: the residuals shrink by about 7 per level (0.099 at k = 1, 4e−9 at k = 10, 3e−7 at k = 16), which is the signature of missing (4−√15)^k terms, the Galois conjugate of λ². With that omission Im A ≈ −3.779 (approximate only). Rerun with ±(4−√15)^k terms added and a 7-level exact fit (K = 20).
