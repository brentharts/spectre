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
