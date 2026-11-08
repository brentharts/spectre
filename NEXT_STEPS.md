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
- **Result (2026-11-07), scripts `research/handedness.py`, `research/handedness_closed.py`, data `research/handedness_k900.json`:**
  - **Exact arithmetic.** The moment recursion runs exactly in ℤ[ζ₁₂] (positions scaled by 28), with no floating point.
  - **Sign check, 1 ≤ k ≤ 900.** Im Z_k < 0 holds exactly at every level in this range. The chirality of the Delta supertile alternates, starting unmirrored at k = 1.
  - **Certified recurrences** (Berlekamp–Massey, checked exactly on all terms, K ≥ D + L):
    - N, order 2: x²−8x+1
    - M0, order 4: x⁴−8x²+1
    - S, order ≤ 12: adds x²+8x+1 and x⁴−488x²+1
    - M1, order ≤ 14: (x∓1), (x²∓8x+1)², x⁴−8x²+1
    - W = N·M1 − conj(S)·M0, order ≤ 14, checked on 120 terms ≥ 76 + 14: dominant factor (x²−62x+1)², λ⁴ = 31+8√15, so the Jordan block is double.
  - **Theorem (exact).** Z_k = A·k + O(1), with the same A for both parities:
    **A = (27/100)(√5+√15)(1 − 2i)**, so Im A = −(27/50)(√5+√15) ≈ −3.29889 < 0 and arg A = −arctan 2.
    The measured slope (Z_k − Z_{k−2})/2 agrees with A to about 10⁻¹⁴ at k = 60 to 119.
  - **Correction.** The earlier fitted value Im A ≈ −3.78 was a fitting artifact.
  - **Still open:**
    - Step 3 as a written proof: an explicit envelope for Z_k − A·k − B (other poles are smaller by at least λ⁻², so for k > 900 the remainder is negligible next to −3.3k, but B and the envelope are not yet stated).
    - Step 4 (Lean).
    - Stretch goal: explain the factor 1−2i (arg A = −arctan 2) cohomologically.

### B. Local dimension and multifractal spectrum of ∂W (answers the open question of Baake–Gähler–Mazáč–Sadun)
- **Statement:** compute the local dimension d(x) at every point of ∂W from the graph-directed IFS. Either prove it is constant (= log(5+2√6)/log(4+√15)), or compute the full multifractal spectrum f(α) of the natural measure on ∂W.
- **Method:** take the GIFS from arXiv:2311.03541 and arXiv:2411.15503. Check the open set condition by a finite neighbour-graph computation, as in Bandt–Mesing. Get d(x) from the thermodynamic formalism of the strongly connected components of the GIFS graph: if there is more than one component with a different spectral radius, d(x) is non-uniform.
- **Payoff:** a direct answer to a question the Spectre's discoverers left open. This item replaces the failed contact closure; the GIFS is finite by construction.
- **Finding (2026-11-07): the GIFS is not published.**
  - arXiv:2411.15503 (Remark 8) only states that the control-point fixed-point equation lifts to a contractive IFS on 30 subwindows (5 types × 6 orientations), and defers the maps to a PhD thesis.
  - The paper gives d_H = log(5+2√6)/log(4+√15) ≈ 1.110977 (Eq. 7).
  - It says the dimension "seems to be the same everywhere along the boundary… this certainly deserves a closer analysis", which confirms the question is open.
- **Revised method: build the GIFS ourselves from the 4D lift.**
  1. Restrict the integer inflation `lift4d_matrix.phi_matrix()` to the exact contracting plane (`exact_bounded_plane()`). That gives the linear part Φ* (eigenvalues ±λ⁻¹).
  2. For each label/orientation node, the subwindow equation is W_i = ⋃_j Φ*W_j + t*_ij, with t*_ij the star images of the integer child offsets already produced by the moment recursion.
  3. Neighbour graph: nodes are pairs (i, j, δ) with δ ∈ ℤ⁴ and |δ*| ≤ diam W_i + diam W_j. This is finite because ℤ⁴ projects to a discrete set in the extended space. Close it under the substitution; this replaces the physical-space contact closure that exceeded 2M types.
  4. The boundary GIFS is the neighbour graph minus self-pairs. Get its strongly connected components and their spectral radii. The dimension is uniform if every essential SCC has spectral radius 5+2√6 measured on the λ² scale.

### C. Gap labelling for the Spectre
- **Statement:** compute the gap-labelling group τ*(K₀(C(Ω) ⋊ ℝ²)) ⊂ ℝ explicitly, as a ℤ-module generated by the patch frequencies. For a 2D tiling this is the image of Ȟ² = ℤ¹⁰ under the frequency map.
- **Prediction to test:** it is ℤ[λ²]-generated, that is, contained in (1/N)ℤ[√15]. Then compare it with the integrated density of states of the tight-binding Hamiltonian of item 4 on level-4/5 patches: every observed gap must sit at a predicted label.
- **Why it is new:** the cohomology is known, but the frequency module and a numerical confirmation of gap labels on a monotile have not been published (check before starting).

### D. Bounded displacement and discrepancy
- **Question:** are Spectre tile centres a bounded displacement of a lattice (BD-equivalent to ℤ²), or does the discrepancy grow?
- **Why the project can answer it:** the discrepancy is controlled by the non-Perron spectrum of the substitution acting on counts and on positions. χ(Φ) has eigenvalues ±1 in 2×2 Jordan blocks, which predicts discrepancy growing like log R for balls of radius R. That would make the Spectre **not** BD-equivalent to a lattice, unlike the Penrose and Ammann–Beenker tilings. The Hat is the natural comparison.
- **First run:** the moment recursion with N alone per label, together with the Jordan structure at ±1, gives exact counts in supertiles. Then prove the log R lower bound along the supertile sequence, following Solomon's criteria for substitution tilings.
- **Payoff:** a clean yes/no theorem, and a structural distinction between the Spectre and the classical cut-and-project tilings.
- **Resolution (2026-11-07): the prediction above is wrong. The answer is yes, BD-equivalent to a lattice (modulo two hypotheses below).**
  - The count matrix M (9×9, `spectre_figures.py`) has spectrum λ², λ⁻², 1, −1, 0⁵. So the non-Perron part has modulus at most 1.
  - Solomon (2011, *Discrete Comput. Geom.* 45, "Substitution tilings and separated nets with similarities to the integer lattice") proves: if every non-Perron eigenvalue θ of a primitive self-similar tiling in ℝᵈ satisfies |θ| < (λ²)^{(d−1)/d}, then the tile set is BD to a lattice. Here d = 2 and the threshold is λ ≈ 2.81 > 1.
  - The ±1 eigenvalues make supertile counts deviate from ρ·area by O(k) = O(log R). That is far below the perimeter scale R permitted by Laczkovich's criterion. The earlier heuristic confused "discrepancy grows" with "discrepancy exceeds the perimeter".
  - The Jordan blocks at ±1 belong to χ(Φ) on the 4D lift, not to M. They do not change the conclusion: polynomial growth in k is still o(λᵏ).
  - **Hypotheses still to check before calling this a theorem:**
    1. Solomon's theorem applies to the self-similar version (fractal-boundary tiles of Baake–Gähler–Mazáč–Sadun). His proof needs only boundaries of measure zero and a supertile boundary count of order λ^{D_b k} with D_b > 0, which holds.
    2. The Spectre itself differs from that version by a linear map plus a bounded displacement. BD-to-lattice is invariant under both.
  - **Consequence:** the Spectre does **not** differ from Penrose, Ammann–Beenker or the Hat (spectrum φ⁴, φ⁻⁴, ±1) on this point. The structural distinction must be sought elsewhere, for example in items B and E.

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

> **B, progress (2026-11-07).** Two preliminary checks failed, and they narrow the construction.
> (1) Projecting the supertile translations b_k (ring coordinates in Z[zeta12]) by `exact_bounded_plane()` gives expansion by lambda^2 every two levels and no zeta/conj equivariance.
> (2) No exact linear map T has b_{k+2} = T b_k. The best fit has eigenvalues ~1, 1, lambda^2, lambda^2, so Z^4 has no contracting direction.
> This is expected in hindsight, because lambda is not in Q(zeta12).
> The internal space exists only in the 16D lift (PHI acting on the 4 key points). The GIFS therefore has to be built there, exactly as in the revised method's step 1: restrict `phi_matrix()` to its contracting eigenspace.
> Scripts: `research/window_gifs.py`, `research/window_gifs_plane.py`.

> **B, star map identified (job 1696527).** The previous progress note was too pessimistic: a 2D window does exist. Its star map depends on the level's parity, like the frame mirroring.
> - On even levels, the annihilator F from `exact_bounded_plane()` (ring order, before the patch mirror) keeps every Delta tile position bounded: rmax 15.33, 16.10, 16.20 at k = 2, 4, 6.
> - On odd levels, F∘Mirror does the same: rmax 10.84, 15.33, 16.10 at k = 1, 3, 5.
> - Each one grows by about λ² per level in the wrong parity.
> - The Galois conjugations ζ ↦ ζ⁵, ζ⁷ grow like λ per level, so they give no window.
> - Script: `research/window_star.py`.
> **Next:** test whether F intertwines rotation by ζ and conjugation with fixed 2×2 maps.
> - If yes, the GIFS has 9 labels × 12 orientations × 2 chiralities as states, with internal maps R(a)·φ*.
> - If no, the states must carry the full 16D key-point data.

### B, equivariance settled (job 1696630)
Exact test over Q(sqrt15). Let F be the even-level frame and F·Mi the odd-level frame.
- zeta^2 (60°) intertwines each frame with itself. R = [[2.5108, -2.8514], [1.6811, -1.5108]] has trace 1 and det 1, so it is a rotation of order 6 in a skew basis.
- conj and the mirror swap the two frames, with R = -I and R = I.
- zeta (30°) fails in every pairing, which matches step 1. Placements never use it.
Consequence: the GIFS states are label × 6 orientations × 2 chiralities, and each internal map is R(rotation)·(the star image of the child offset), with the frame alternating by level.
Next: build the GIFS, close the neighbour graph, and compute the SCC spectral radii against 5+2√6.

### B, box-counting check (job 1696914, heuristic)
Boundary box-counting on the star images of the Delta tile positions:
- Fitted slope: 1.220 at level 6 (2.7e5 points) and 1.222 at level 7 (2.1e6 points).
- Local slopes at the finest well-sampled scales, level 7: 1.247, 1.196, 1.178. They are drifting down.
- BGMS predict log(5+2√6)/log(4+√15) = 1.111.
Reading: the boundary is clearly fractal and the frame is plausible, since the slope is nowhere near 1.0 or 1.5. The estimate is still 6–10% high. Sampling holes counted as boundary bias it upward, so it is consistent with 1.111 but does not confirm it.
The exact GIFS / neighbour-graph route stays open. It needs the internal contraction taken from the 16D PHI rather than from Z⁴, because no linear T on Z⁴ exists.

### B, why the exact GIFS is blocked here
Key points evolve linearly in 16D (Q_{k+1} = P Q_k). But the Spectre is not self-similar, so no Z-linear inflation acts on the Z⁴ tile positions. That explains why no T exists and why star images stay bounded instead of contracting. A contractive GIFS needs the control points of the self-similar (fractal-tile) version, as in BGMS Remark 8, with 5 types × 6 orientations. Building it means reconstructing their control-point recursion. B is parked there. Its heuristic support is the box-counting result above.

### C, step 1 started
The literature search (2026-11-07) found no Spectre-specific frequency module or gap labels. Pinwheel analogue: 1/264·Z[1/5]. Step 1, `research/gap_freq.py`: the uncollared supertile-frequency module, computed exactly in Q(√15).

### C, step 1 result (job 1697010, exact)
- charpoly(M) = x⁵(x−1)(x+1)(x²−8x+1).
- The per-spectre supertile frequencies u are all positive and lie in Q(√15), e.g. u_Γ = 1/2 − √15/10. (Γ, the mystic, counts as 2 spectres.)
- The Z-module spanned by u·λ^{−2k} is ℤ + ℤ·(5+√15)/10. Since N(5+√15) = 10, this equals **(5−√15)⁻¹·ℤ[√15]**, a principal fractional ideal of index 10 in (1/10)ℤ[√15].
- This confirms prediction C for the uncollared submodule.
Next: build the collared-tile substitution, either from spectre_patch neighbourhoods or from the 10 Čech H² generators. Check whether the collared frequencies enlarge the module. Then test gap positions in the IDS of the item-4 tight-binding Hamiltonian against this module.

### C, step 2 result (job 1697446, exact)
`research/gap_collared.py` (K=6, 242047 tiles) uses this collared descriptor: the label plus every vertex-sharing neighbour, with the touching child-slot pairs. Output: `research/gap_collared.json`.
- There are 38 collared types, closed under substitution. All transitions are consistent (0 conflicts).
- The Perron vector is exact over Q(√15), and its label projection is proportional to the uncollared u.
- The frequency module has basis {1, 1/2 + √15/10}, with covolume 1/10. Its **index over step 1 is 1**. Collaring at this radius adds nothing, so the module stays (5−√15)⁻¹ℤ[√15].
Next (step 3): compare the IDS gap positions of the item-4 tight-binding Hamiltonian with this module.

---

## Plan after Brent's ribbon and holonomy work (2026-11-07)

### What upstream added (master 63c83db)
- **`spectrespace.py`** keeps the full substitution tree: for every node, its local placement (including the mirror diag(−1,1) applied to every slot), its global transform, its 4 corners and its centre. It lifts O(2) → SO(3) via L ↦ diag(L, det L), so each 2D flip becomes a half twist about its mirror axis and every parent→child edge becomes a twisted ribbon.
- At depth 3: 567 placements have det −1 and 126 have det +1, and the +1 ones are exactly the Γ₁/Γ₂ pairs. So parity is one global clock per level, and the per-tile information is the rotation angle; the 30° angles come only from Γ₂.
- **`spectre_flip.py`** checks that `trot_inv()`'s ±1 agrees with sign(det).
- **`spectreseal.py`** thickens the ribbons into ellipsoid chains, takes a smooth union, and extracts a closed mesh by marching cubes. It reports whether the mesh is sealed, its Euler characteristic and its genus. The genus depends on thickness and blending, which are render choices.
- **Holonomy figure** (`paper/figures/spectre_holonomy.png`):
  - The normal is forced: n = det(L)·ẑ.
  - The edge quaternions generate a group of order 48 with binary-dihedral element orders, the same for every twist sign.
  - Changing the twist sign only multiplies node holonomies by ±1, so it is a gauge choice.
  - The Z₂ flux around a vertex is (−1)^w, where w is the winding number of the orientation field.
  - Convention-independent vortices (−1 flux, not from a 180° pair): 28 at depth 3 and 372 at depth 4. All are valence-3 vertices where three tiles meet at 120°, mostly Π/Ψ/Φ triples, and none touches a Mystic.

### Why this matters for our programme
The vortices are a **local pattern** (a vertex star). Their density is therefore a patch frequency, which item C predicts must lie in the module (5−√15)⁻¹ℤ[√15]. The twist-sign gauge result and the forced normal settle the "arrow of time" question for holonomies, and leave writhe/linking as the only place where a twist sign could still be physical. Together these give exact, falsifiable statements to put in place of the figures' descriptive claims.

### Big plan: research tasks, in priority order
Each task ends in a theorem, an exact number, or a documented negative result. Cluster runs only.

**R1. Exact vortex density (links Brent's holonomy to C).** *Cheap, high value.*
- Extend `research/gap_collared.py` to classify vertex stars (valence, labels, relative orientations), and mark the convention-independent −1 vortices.
- Compute the exact frequency per tile from the collared Perron vector over Q(√15).
- Prediction: it lies in ℤ + ℤ(5+√15)/10.
- Check: the ratio 372/28 ≈ 13.3, against λ² ≈ 7.87 per level. Expected agreement after boundary correction, because depth 4 has more interior.
- Deliverable: "vortex density = exact number", which places the flux in the gap-labelling group.

**R2. Vortices as a model set.**
- Map the vortex positions into the internal space using the star map from item B (F on even levels, F∘Mirror on odd levels).
- Determine whether they fill a sub-window W_v ⊂ W. If so, the vortex set is a model set and its density equals area(W_v)/area(W), which cross-checks R1.
- Find which supertile slots carry vortices; this gives the substitution rule for vortices.

**R3. The Z₂ flux as a cohomology class.**
- The vertex flux is a Z₂ 2-cochain on the tiling. Decide whether its class in Ȟ²(Ω; Z₂) is nonzero. Use BGMS: Ȟ²(Ω; ℤ) = ℤ¹⁰, which gives Ȟ²(Ω; Z₂) by the universal coefficient theorem.
- A nonzero class means the vortices cannot be removed by any local re-gauging, which turns "convention-independent" into a theorem.
- Approach: Anderson–Putnam complex on collared tiles (the item-3 first run, now with Z₂ coefficients). This is integer linear algebra.

**R4. Identify the order-48 group exactly.**
- Prove which group the edge holonomies generate. The candidates are the binary dihedral group of order 48 (Dic₁₂) and the binary group lifting D₁₂. Check the element orders against the counts 26 and 8 by exact quaternion arithmetic in Q(√2, √3).
- Prove that it is independent of the twist sign: the sign change is a coboundary. Short, and suitable for the paper's holonomy section.

**R5. Parity clock in closed form.**
- Derive the counts of det = ±1 placements per level from the 9×9 substitution matrix, for example 567/126 at depth 3: det +1 exactly for the Γ₁/Γ₂ pair, det −1 for the rest.
- Relate this to the −1 left eigenvector Q₋ already in the paper. Exact, a few lines.

**R6. Ribbon linking, twist and writhe.**
- For the twisted ribbons with each choice of twist sign (+1, −1, alt), compute twist and writhe exactly and check Călugăreanu: Lk = Tw + Wr.
- Holonomy cannot see the twist sign (gauge), but Lk can. Measure whether Lk or its density grows like λ^{2k}, whether it differs between the sign choices, and whether it relates to the exact swirl slope A = (27/100)(√5+√15)(1−2i) from item A.
- This is the only remaining candidate for a physical arrow of time in the ribbon picture: either a positive result or a clean negative.

**R7. Genus of the sealed surface: robust or artefact?**
- Sweep `spectreseal.py` over the thickness and blending radius at depths 2–4, and record the plateaus of genus.
- Fit genus(depth) against node counts (N_k ~ λ^{2k}).
- If a plateau exists, the plateau genus is a combinatorial invariant of the tree. The candidate formula is genus = (number of independent ribbon contacts) − (number of components) + 1. Otherwise, document it as a render artefact so the paper does not claim it.

**R8. Tight-binding with π-flux at vortices (joins item 4 and C step 3).**
- Put the Z₂ flux into the item-4 Hamiltonian as π-flux at each vortex, and compare the IDS and gaps with and without it.
- Gap labels must still lie in the C module (gap labelling is gauge-covariant). A violation would show a numerical or modelling error.
- Run Bott-index tests for flux-induced topological gaps.

**R9 (carry-over). C step 3:** gap positions of the plain tight-binding IDS against (5−√15)⁻¹ℤ[√15], on level-4/5 patches.

**R11. Gauge theory on the Spectre graph: the Part I to Part II link.**
- Goal: one shared object that is both monotile structure and quantum-geometry structure, plus a result that depends on the Spectre. Stated goal of the project: small steps linking the monotile with quantum gravity.
- Step 1 (Z₂): Brent's holonomy is already a Z₂ gauge field on the tiling. Run Z₂ lattice gauge theory on Spectre patches (plaquettes = tiles). Compare the Wilson loops and the vortex free energy with periodic controls (hexagonal, square).
- Step 2 (SU(2)): replace the two-plaquette model of Part II by the Spectre graph. Compute strong- and weak-coupling area spectra and intertwiner volume spectra at valence-3/4 vertices. Compare with a periodic control graph of matched mean valence.
- Step 3 (spin networks): treat the Spectre graph as a spin-network graph. Ask whether the area/volume densities fall in the gap-labelling module ℤ + ℤ(5+√15)/10 (the same test as R1, R9).
- Literature check first: quasicrystalline spin networks (Amaral, Aschheim, Irwin et al.).
- A null result is still useful. If the Spectre behaves like a generic graph, that is publishable, and it says which monotile features matter.

**R10 (carry-over). E:** a Galois-conjugate formula for seam dimensions. **F:** Chair44, a 3D counterpart. **B:** an exact GIFS from the self-similar control points.

### Recommended order (revised: everything gets done; ordered toward the QG link)
1. **Now:** selftest of the merged paper (cluster), then R9 (C step 3).
2. **Next:** R1 (vortex density in the C module), then R4 and R5 (exact Z₂ structure and parity clock).
3. **Then:** R3 and R2 (is the flux topological?), with R8 (flux Hamiltonian) moved up as the first part of R11.
4. **Then:** R11 steps 1–3, the first real link between Part I and Part II.
5. **Afterwards:** R6 (ribbon linking), R7 (genus robustness), E, F, B.

### Notes for syncing
- PR 13 (`swirl-paper-fixes`) conflicts with upstream master only in `spectrepaper.py`, in the abstract and the pancake section. Brent's phys rev IV–VI rewrote the same passages. Resolved in merge 9f63b7d (2026-11-07): our abstract and pointers kept, Brent's scope disclaimers and scope section added.
- `spectre_holonomy.py` is now upstream (test_flux: 28 vortices at depth 3, 372 at depth 4). R1 uses its flux rule exactly; see `research/vortex_density.py`.

## Progress log (2026-11-07, evening)

### Jobs in the queue
- Submitted, waiting to run: gap-ids (R9, 1698583), holonomy-baseline (1698605), R1 (1699051), R4 (1699394), R5 (1699432), R3 (1699761), R2 (1699881), R8 (1700304), R11 step 2 (1701292, `research/su2_graph.py`).
- Selftest (1698557) failed for an infrastructure reason: the sibling repo `oai_tree` was missing, so Parts II–III could not run.
- Scripts: `research/holonomy_group.py` (R4), `parity_clock.py` (R5), `flux_cohomology.py` (R3), `vortex_window.py` (R2), `flux_hamiltonian.py` (R8).

### R11 step 1 (Z2 gauge theory): settled exactly, negative as a probe
- Natural set-up: the gauge field lives on dual edges (tile adjacencies), the plaquettes are interior tiling vertices, and Brent's flux eta_v = flux(v) is the background: S = -beta * sum_v eta_v * prod_{e around v} s_e.
- On any simply-connected patch (open boundary) the plaquette variables are independent: fix the gauge on a spanning tree; each remaining edge then sets exactly one plaquette. So Z = prod_v 2cosh(beta), up to a constant gauge-volume factor.
  - The free energy does not depend on eta: the vortex free energy is exactly 0. The flux is a pure gauge on a disc, consistent with R3: it is a coboundary, just not a local one.
  - Every Wilson loop is W(C) = prod_{v inside C} eta_v * tanh(beta)^{N_C}, where N_C is the number of enclosed vertices. This is a pure area law with string tension -ln tanh(beta) per vertex.
  - The only Spectre-dependent quantities are the vertex density per unit area and the sign prod eta_v. That sign is the R1 vortex count mod 2 inside C.
- Periodic controls (hexagonal, square) give the same formula with their own vertex densities. Nothing here distinguishes the Spectre beyond R1.
- Consequence: a real Spectre signal needs a model that is not exactly solvable. Options are Z2 gauge-Higgs (matter on tiles, which R8 already is in the free limit), the tiling × time (2+1D, which has a deconfinement transition), or a periodic approximant with non-contractible loops. Step 2 (SU(2) area and volume spectra) goes ahead as planned.

- 2026-11-08: all queued jobs still PENDING. R6 script `research/ribbon_linking.py` and R7 sweep `research/genus_sweep.py` written (82f18ac); R6 submitted as job 1702945 at 2 CPUs (lane now 64/64). R7 waits for budget.

### Results 2026-11-08
- **Holonomy baseline (1698605):** 18/18 checks pass. Edge group has order 48 for every twist sign; the twist sign is a Z2 gauge. At K=4: 1608 flux −1 vertices, 372 of them convention independent, all of valence 3.
- **R1 vortex density (1699051):** convention-independent vortices number 28, 372 and 3604 at K=3, 4, 5. As a fraction of interior vertices that is 0.0094, 0.0150 and 0.0179 (R2 gives 0.0194 at K=6), still rising as the boundary shrinks. Every vortex has valence 3. The exact closed form was not reached: no closed level-1 types, so the script stopped with "cannot evaluate exactly". The limit is still open.
- **R2 vortex window (1699881):** in the star map F at K=6 the vortices fill 5.3% of the window cells and their k=1 purity is 0.27, against a base rate of 0.019. Vortices therefore cluster in a sub-window. The mirror map shows no structure (purity 0). This is consistent with flux being a window (cut-and-project) function.
- **R3 flux cohomology (1699761):** no assignment of local edge-type signs out to radius R=50 reproduces the flux at K=4 or K=5. The obstruction is a single valence-2 star in every run, so the flux is not a coboundary of local edge signs at these radii. Negative.
- **R5 parity clock (1699432):** completed. The characteristic polynomial is λ⁵(λ−1)(λ+1)(λ²−8λ+1); all meta slots have the same det at each height.
- **R6 ribbon linking (1702945):** Wr ≡ 0, because every centre line is a planar curve. So Lk ≈ Tw, with residual ≤0.26, apart from a numerical near-crossing outlier at depth 2. At depth 3 (693 ribbons) Lk is −16.8, −18.1 and −18.1 for twist signs +1, −1 and alt. Lk is nearly the same for all three choices and does not flip with the sign. |Tw| grows with the ribbon count (2.3, 13.7, 107 for 10, 87, 693 ribbons) with a near-constant per-ribbon value. **Clean negative:** no twist-sign arrow of time in Lk.
- **R11 step 2 (1701292):** Spectre K=3 graph: 3734 vertices, 4292 links, 559 plaquettes, all of them 14-gons.
  - Strong-coupling S is uniform (one distinct value).
  - Weak-coupling √K takes 752 distinct values, with mean 0.513 ± 0.014.
  - The bulk valence-4 fraction is 0.0295.
  - Controls: square 1.0, honeycomb 0. K=4 segfaulted: the dense eigh is too large, and the K=3 result stands.
- **R4 holonomy group (1699394):** the group computation succeeded (order 96, Dic24). **Correction (2026-11-08): 96 is a bug, see below.** It then failed an assertion comparing the edge quaternion with `spectre_holonomy.edge_quaternion`, which is a convention mismatch to fix.
- **R8 flux Hamiltonian (1700304):** failed inside the Bott index, where `logm` produced NaN. I changed the code to use the eigenvalue phases instead and am resubmitting.
- **gap-ids (1698583):** still pending.

- 2026-11-08: R8 resubmitted with eigenvalue-phase Bott index (job 1703968); R7 genus sweep submitted (job 1703987); R4 rerun with mismatch diagnostics submitted (job 1704004).
- 2026-11-08: R11 step 3 submitted (research/spin_densities.py, K=6, job 1704295): exact per-spectre valence-3/4 vertex and link densities via level-1 collared types, tested against Z + Z(5+sqrt15)/10.

### E, first test on the Spectre (2026-11-08, exact, no computation needed)
- The boundary-word operator of `spectre_boundary.py` is an integer matrix with charpoly (x−1)(x²−4x−1), spectrum {1, 2±√5} ⊂ ℚ(√5).
- A Galois automorphism acts on an integer matrix trivially, so "B twisted by Galois" has the same spectrum; at best the twist re-labels which root is "physical". Neither 2+√5 nor |2−√5| equals the seam rate √2+√3 (per level), and ℚ(√5) ∩ ℚ(√2,√3) = ℚ. **E in its literal form is refuted for the Spectre.**
- The seam rate comes from a different object: the internal contact matrix (1192 types, Theorem thm:window) with factor (x⁴−10x²+1)², roots ±√2±√3. It is not a twist of Φ either: Φ's factor is x⁴−8x²+1, roots ±λ^{±1}.
- Revised conjecture E′: dim(seams) = log ρ(C_int)/log λ, where C_int is the contact matrix of the internal GIFS, and ρ(C_int)² = 5+2√6 is a unit of ℚ(√6) with N(5+2√6)=1, as λ² = 4+√15 is a unit of ℚ(√15). λ = (√6+√10)/2 lies in ℚ(√6,√10), whose quadratic subfields are ℚ(√6), ℚ(√10) and ℚ(√15): the area unit λ² sits in ℚ(√15) and the seam unit in ℚ(√6), so both come from the field of λ, while the outline rate ν sits in ℚ(√5), outside it. Note that 5+2√6 = (√2+√3)² and λ = √2·(√3+√5)/2.
- Next for E′: compute ρ(C_int) for the Hat (internal space ℝ², BGMS), where the prediction is a unit of the analogous subfield; Chair44 is 2-adic and needs a separate contact-matrix construction (part of F).

## Revised plan (2026-11-08, after the first round of results)
The "Recommended order" sections above are out of date. Most of R1–R8 and R11 steps 1–2 have run. The current priorities follow.

### Open problems in the record
- **Order 48 or 96.** Resolved: 96 was an encoding bug in R4 (see the second batch below). Confirmation rerun: job 1705331.
- **The R1 density limit.** R1 stopped because level-1 collared types do not close. R11 step 3 closed with the same machinery (38 types), so R1 can reuse `research/spin_densities.py`'s construction: classify vortices by collared supertile. Fallback: collar at level 2, or take transitions from level 3 to level 2, at K=7 on the cluster.
- **R11 step 2 at K=4.** The dense `eigh` ran out of memory. Switch to sparse `eigsh`, or use the strong-coupling spectrum, which is exact, and spectral moments.
- **The paper selftest (1698557).** It failed because the sibling repo `oai_tree` was missing. It has never been rerun, and it gates any edit to the paper.

### Priority order
1. **Assess the pending jobs:** 1704295 (R11 step 3), 1703968 (R8), 1703987 (R7), 1704004 (R4) and 1698583 (R9). Record every result here, including negatives.
2. **Exact vortex density.** Finish R1 with the level-2 collaring fallback. Of all these tasks, this has the best chance of an exact theorem: the vortex density lies in the gap-labelling module.
3. **E′ on the Hat.** Compute ρ(C_int) for the Hat. This is a cheap and decisive test of the revised conjecture.
4. **A Spectre-specific gauge model.** R11 step 1 showed that the Z₂ theory on a disc is exactly solvable and generic. The next model must not be: either Z₂ gauge–Higgs, or 2+1D Z₂ on tiling × time, with deconfinement compared against the honeycomb.
5. **Literature check for R11,** on quasicrystalline spin networks (Amaral, Aschheim, Irwin et al.). This must happen before any QG claim goes into the paper.
6. **Afterwards:** F (Chair44: 2-adic model set and contact matrix), B (the exact GIFS), and a rerun of the selftest once `oai_tree` is available.

## Results 2026-11-08 (second batch)
- **R11 step 3, spin-network densities (1704295): positive and exact.** At K=6 there are 38 level-1 collared types, 0 of them inconsistent, and the transition system closes. The densities per spectre, all exact in ℚ(√15), are:
  - valence-3 vertices: 2√15/5 ≈ 1.549;
  - **valence-4 vertices: 1 − √15/5 = (5−√15)/5 ≈ 0.2254;**
  - vertices: 6;
  - links: 7.
  - **Module test.** All four lie in the gap-labelling module ℤ + ℤ(5+√15)/10. For the valence-4 density the coefficients are n = −2 and m = 2: 2 − 2·(5+√15)/10 = 1 − √15/5.
  - **Euler check.** V − E + F = 6 − 7 + 1 = 0 per spectre, as it must be for a planar tiling.
  - **Meaning.** With spin 1/2 on every link, only valence-4 vertices carry volume. The density of volume-carrying nodes per spectre is therefore an element of the tiling's K-theory gap-labelling group. This is the first exact Part I to Part II link.
  - **Consistency with step 2.** As a fraction of vertices this is (5−√15)/30 ≈ 0.0376. The K=3 bulk estimate in step 2 was 0.0295, low because of boundary effects.
  - **Caveat.** Membership is a necessary condition only. Every frequency of a local pattern lies in the module, so this result shows that the spin-network data is of the expected type. It does not yet single out the Spectre. The Spectre-specific content is the exact value, (5−√15)/5.
- **R4 holonomy group (1704004): the order 96 was a bug in `research/holonomy_group.py`.** The bug is now fixed.
  - **Cause.** The integer encoding took w = exp(15°·k) to have order 48 with −1 = w²⁴. In fact w has order 24 and w¹² = −1. As a result the closure doubled the group, and multiplying by the "central sign" (0,24) was the identity, so the twist sign was dropped.
  - **Evidence.** Every one of the 66464 mismatches at N=5 is a flip edge with mode −1, where q = −q_e.
  - **Confirmed (1705331).** At N=5 (43027 nodes), the twist signs 1, −1 and alt all give Dic12 of order 48, the dicyclic presentation holds, and there are **0** edge mismatches against upstream. All three sign choices are coboundary-equivalent. Brent's order 48 stands.

## Literature check for R11 (2026-11-08, plan step 5)
- **Prior art on quasicrystalline spin networks** comes from a single group, Quantum Gravity Research.
  - Amaral, Aschheim and Irwin, *Quantum Gravity at the Fifth Root of Unity* (arXiv:1903.10851).
  - Amaral, Fang, Hammock and Irwin, *Geometric State Sum Models from Quasicrystals* (2021).
  - Amaral, Clawson and Irwin, *Quasicrystalline Spin Foam with Matter* (arXiv:2306.01964). It defines quasicrystalline spin networks as a subspace of the LQG Hilbert space, made of EPRL coherent states, with examples from the 4D Elser–Sloane (E8) quasicrystal.
  - These are mostly preprints. None of them uses a planar monotile.
- **No prior work** was found that puts spin networks or lattice gauge theory on the Hat or the Spectre.
- **Consequence for the paper.** R11 step 3 (the valence-4 density in the gap-labelling module) is new as far as this search reaches. It must:
  - cite the works above as the quasicrystal-to-LQG precedent;
  - make no claim that the result is physical quantum gravity, since it is a kinematical, strong-coupling statement only.
- **Search depth.** This was a web search only. Before submission, do a full arXiv search on "aperiodic tiling" together with "spin network", "gauge", and "gap labelling" together with "lattice gauge".

## E′ on the Hat: blocked on construction (plan step 3)
- **Missing code.** The repo has no Hat implementation: no metatile substitution, no window, no internal GIFS. The test is therefore not cheap. It needs the Hat metatile inflation (H, T, P, F; factor φ²) and its window, which is a Rauzy fractal in ℝ² (Baake–Gähler–Sadun, arXiv:2305.05639). The contact matrix must then be built the way the Spectre's was in `research/window_gifs*.py`.
- **A shortcut to try first.** BGS / BGMM (arXiv:2502.03268) state that some boundaries of the Hat windows have non-integer Hausdorff dimension. If either paper gives that dimension d, E′ predicts that φ^{2d} is a unit of a quadratic subfield of ℚ(φ) = ℚ(√5). That field has no proper quadratic subfield, so E′ forces φ^{2d} ∈ ℚ(√5). This can be checked from the published number without new code.

## E′ on the Hat: decided from the literature, E′ refuted as stated (2026-11-08)
- **Published value.** Baake–Gähler–Sadun (arXiv:2305.05639, §3) state that the Koch-type window boundaries of the Hat have d_H = log(2+√3)/(2 log φ) ≈ 1.368, computed from the induced edge IFS. The inflation factor is φ².
- **E′ predicted** a seam unit in ℚ(φ²) = ℚ(√5).
- **Actual.** The Hat's seam unit is 2+√3, a unit of ℚ(√3) of norm 1, and **it is not in ℚ(√5)**.
  - So E′ fails for the Hat: "the seam unit lies in a subfield of the inflation field" holds for the Spectre only by coincidence, or because the Spectre's field ℚ(√6, √10) happens to contain ℚ(√6).
- **Pattern that survives (heuristic, labelled as such):**
  - Hat: seam unit 2+√3 = (1+√3)²/2, per inflation φ².
  - Spectre: seam unit 5+2√6 = (√2+√3)², per inflation λ².
  - Both seam rates involve √3 and are units of real quadratic fields. A natural source is the 6- and 12-fold rotational geometry of the two tiles, through cos 30° = √3/2.
  - **Candidate E″:** the seam unit comes from the rotation group of the tile edges, not from the inflation field.
  - **Test:** take Hat-family tiles Tile(a,b), whose inflation factors vary, and check whether the seam unit stays in ℚ(√3) as the inflation field changes. BGS treat the whole family.
- **Action for the paper.** Drop E′ as a conjecture. If anything is said, present the two seam units side by side as an observation.

### Submitted 2026-11-08 (third batch)

- Step 2, exact vortex density (`research/vortex_density2.py 6`): job 1708335.
- Step 4, Z₂ deconfinement on the Spectre graph against honeycomb and square controls (`research/z2_deconfinement.py`): job 1709080.
- Still queued: R8 (1703968), R7 (1703987), R9 (1698583).

- chair44_cohomology (item F, rational Cech H* of the Chair44 hull): job 1710452

### Results 2026-11-08 (third batch)

- **Step 2, exact vortex density: done.** The density of vortices (flux −1 vertices of valence 3) is exactly **4 − √15 ≈ 0.1270 per spectre**, i.e. (4 − √15)/6 ≈ 0.0212 per vertex, and lies in the gap-labelling module ℤ + ℤ(5+√15)/10. All flux −1 vertices together have density 2 − 2√15/5. The level-1 collared types (38) are closed with 0 inconsistent, so the level-2 fallback is not needed. The finite-patch values of R1 (0.019 per vertex at depth 6) approach the exact value from below, as boundary losses predict. Job 1708335.
- **Step 4, Z₂ deconfinement: coarse scan done, fine scan submitted.** The Binder curves show the ordering transition of the dual 3D Ising model clearly: Spectre at β ≈ 0.08–0.09, honeycomb 0.16–0.17, square 0.22–0.23 (3D Ising: 0.2217, so the control agrees). The automatic crossing detector reported spurious values (0.047, 0.098, 0.166) taken from noise where U ≈ 0. The script now accepts only crossings with max U > 0.15, and the rerun uses fine grids and 2000 sweeps. Job 1709080; fine rerun job 1712312.

### Results 2026-11-08 (fourth batch)

- **Item F, Chair44 cohomology: done.** The collared Anderson–Putnam complex has 3888 cube types and cells (1506, 6708, 9084, 3888). The 4×4×4 language is closed under σ at depth 6 (0 missing; depth 5 missed 108, so depth 6 is needed), every child is found, and f is a chain map. The ranks stabilise from m = 3, giving **dim Ȟ^k(Ω_Chair44; ℚ) = 1, 3, 17, 21 for k = 0..3**, with Euler characteristic −6. The values agree at depths 5 and 6. Top eigenvalues of f are 8 on C₃, 4 (×3) on C₂ and 2 (×3) on C₁. These match the volume, the 2-forms and the translations, so Ȟ¹ = ℚ³ is just the translation module, as it should be for a limit-periodic tiling. Job 1710452. Still open for F: pure point spectrum (cited, via `chair44_twoadic.py`) and the handedness invariant.
- **Step 4, Z₂ deconfinement fine scan: control passes; Spectre value not yet sharp.**
  - Square: β_c = 0.2229, within 0.005 of 3D Ising (0.2217), so the method is validated.
  - Honeycomb: β_c = 0.1614.
  - Spectre: the curves for N = 559 and N = 4401 do not cross inside 0.080–0.095. Both are flat at U ≈ 0 up to β = 0.085, and the larger lattice jumps more steeply (N = 4401: U goes 0.13 → 0.65 between 0.086 and 0.087; N = 559 rises over 0.086–0.091).
  - Interpretation: the transition is at **β_c ≈ 0.086 ± 0.001**, located by the jump and not by a crossing. β_c·(z̄+2) ≈ 1.3, close to the square value (1.30) and the honeycomb value (1.26), which fits the mean-field-like scaling with coordination.
  - Job 1712312. A crossing would need a third size, a level-5 patch.
- **R8, π-flux tight-binding: inconclusive.**
  - Results:
    - The main gap survives the flux. At level 4 it sits at IDS ≈ 0.117, with width 1.68 without flux and 1.29 with it.
    - The flux opens new gaps at IDS 0.124, 0.161, 0.674, 0.856, 0.884 and 0.985.
    - Every Bott index is 0 (|B| < 2·10⁻⁹).
  - The zero is forced: the hoppings ±1 are real, so the Hamiltonian is time-reversal symmetric and its Chern/Bott index vanishes. A topological test needs complex flux (for example π/2), not π.
  - The fitted module labels use large integers (such as (−8, 10)), with residuals up to 0.04 at level 3. That is not a test of the C module.
  - Job 1703968.

### Submitted 2026-11-08 (fifth batch)

- **Item F, handedness invariant: submitted (job 1713919, `research/chair44_handedness.py`).**
  - Definition: X_k = Σ_c (N p_c − S)·w(R_c) over the cells of the level-k chair supertile, where w(R) is the axial vector of the tile's rotation.
  - Properties: X_k is rotation invariant, changes sign under a mirror, and is an exact integer. Its normalised form is χ_k = X_k/(N²2^k).
  - Decision: a nonzero χ_k limit with fixed sign is a handedness of the Chair44 hull, the 3D analogue of A. If X_k ≡ 0, move to the traceless tensor part (also recorded) or to higher moments.
- Follow-ups from the fourth batch (not yet submitted):
  - R8 with complex flux (π/2), which allows a nonzero Bott index;
  - Z₂ deconfinement on a level-5 patch, to get a true Binder crossing for the Spectre.

### Results 2026-11-08 (fifth batch)

- **Item F, handedness (job 1713919): null.**
  - X_k = 0 exactly for every k = 0..7 (N up to 14,680,064 cells), so this pseudoscalar detects no chirality of the Chair44 hull.
  - The tensor T_k = Σ_c (N p_c − S) ⊗ w(R_c) is not zero. It always has the form [[−a, −b, b], [b, a, −b], [−(a+b), a+b, 0]]: traceless, with growing antisymmetric and symmetric parts. For example, k = 7 gives a = 1.257e12 and b = 1.296e14.
  - Next (per the decision rule): explain why X_k vanishes identically (look for an improper symmetry of the identity-frame supertile), then test whether the traceless part, normalised by N²2^k, has a limit.
- **R9, gap labels (job 1698583): partial.**
  - Tile Hamiltonian: no gaps at levels 3–5.
  - Vertex Hamiltonian: the same 4 gap edges at levels 3 and 4: IDS 0.315/0.316 with label [−5, 6], and 0.684/0.685 with label [6, −6].
  - The level-5 vertex run is not in the log.
- **R7, genus sweep (job 1703987): failed (infrastructure).**
  - `skimage` is missing on the cluster, so every setting errored. Depth 4 then ran out of memory.
  - Resubmitted as job 1715015: installs scikit-image, depths 2–3 only.
- **Z₂ on a level-5 patch (job 1714510): heuristic β_c.**
  - No Binder crossing between any pair of levels 3/4/5.
  - The jump in U (0 → 2/3) sharpens and moves down with size: its midpoint is ≈ 0.0862 at N = 4401 and ≈ 0.0855 at N = 34649, rising within Δβ ≈ 0.001.
  - Estimate: β_c ≈ 0.0855 ± 0.0007. A jump this abrupt without a crossing may point to a weakly first-order transition or to strong boundary effects (open spatial boundary). That is a question for later, not a blocker.
- Remaining follow-up: R8 with complex π/2 flux.

### Item F, why X_k vanishes (2026-11-08, from the recorded tensors)

- For every k ≤ 7 the recorded tensor satisfies T_k = −P T_k Pᵀ, where P is the mirror that swaps x and y. Check: P T Pᵀ = [[a, b, −b], [−b, −a, b], [a+b, −(a+b), 0]] = −T.
  - That is the transformation rule of an axial-vector correlation under a mirror symmetry of the configuration. So the level-k identity-frame chair supertile is invariant under the x ↔ y mirror (positions reflected, frames conjugated) at every level checked.
- Consequence: X_k = tr T_k = 0 is forced by symmetry and says nothing about the dynamics.
  - If the substitution commutes with this mirror, then the Chair44 hull is mirror-closed and every pseudoscalar observable of it vanishes. Unlike the Spectre, the Chair44 hull is achiral.
  - The recorded tensors confirm the mirror relation numerically at levels 0–7. A proof needs the substitution rule to be checked for equivariance in `symchair44.py`. That is a finite check and a small next step.
- Item F status: the handedness question is answered negatively (no chirality to find). The traceless tensor is a shape correlation, not a handedness, and is parked.

### Submitted 2026-11-08 (sixth batch)

- **R7 genus sweep, third attempt (job 1715742).** Installs scikit-image with `pip --target` into a side directory on PYTHONPATH. The second attempt (1715015) failed because `pip --user` is refused inside the virtualenv. Depths 2–3.
- **Item F mirror check (job 1715771, `research/chair44_mirror.py`).** Tests directly whether the level-k supertile (k ≤ 6) equals its x ↔ y mirror image with frames conjugated. All True: the Chair44 hull is recorded as achiral and F is closed.
- Still open: R8 with complex π/2 flux.

### Results 2026-11-08 (sixth batch)

- **Item F mirror check (job 1715771): passed. Item F is closed.**
  - For k = 0..6 (up to 1,835,008 cells), the level-k identity-frame Chair44 supertile equals its image under the x ↔ y mirror, with tile frames conjugated. This was tested as exact set equality.
  - This explains X_k ≡ 0: the Chair44 supertiles are achiral, so every pseudoscalar of them vanishes. The Spectre's handedness (item A) has no analogue in Chair44.
  - What would upgrade this to a proof: show that the substitution commutes with the mirror on the 3888 collared types. The check above is the computational evidence for that.
- **R7 (job 1715742): running.** The scikit-image install succeeded this time.

## Next ideas (2026-11-08, from the discussion with Brent after PR #14)

These ideas turn points from the discussion into tests that can be computed. The tests come first. The most speculative physics is kept apart.

### G. Connect the shapes, not only the expansion rates (top priority)

So far the Spectre and Chair44 are connected only through their expansion rates. Brent's `chair_fibers.py` places the Spectre's 28 Bézier handles against the tips of the chair pyramid, but this is a geometric construction and is not yet a model.

- **Constraint from item F.** The Chair44 hull is achiral, and the Spectre hull is chiral (item A). A construction that commutes with the mirror cannot produce Spectre handedness from Chair44. So any real bridge must break the x ↔ y mirror. That is a quick test to run on every candidate map.
- **G1.** Read `models/chair_fibers.py` and `GEOMETRY_BRIDGE.md`. Write the proposed correspondence down as an explicit map, from points on the chair tips to vertices or handles of the Spectre.
- **G2.** Apply the mirror test of `chair44_mirror.py` to that map. If the map commutes with the mirror, it cannot carry the Spectre's handedness, and we record that as a negative result.
- **G3.** If the map breaks the mirror, check whether it intertwines the substitutions, up to the matching of expansion rates. If it does, it is a genuine shape-level link.

### H. "Space is defined by interactions": compare graph distance with Euclidean distance

The claim is that space is defined by interactions. A version we can test is to ask whether graph distance on the Spectre graph reproduces Euclidean distance.

- **H1, dilation.** Over many vertex pairs, compute the ratio (graph distance)/(Euclidean distance). On periodic lattices this ratio tends to a direction-dependent constant. Does it on the Spectre, and is the limit isotropic? Run square and honeycomb controls.
- **H2, correlations.** For free fermions at half filling on the Spectre graph:
  - Compute the mutual information between two small regions. Does it decay with graph distance or with Euclidean distance? (H1 tells us whether these differ.)
  - Compute the entanglement entropy of discs. Does it follow the area law, or the area law plus a log term? Run the same controls.
- If the Spectre matches the controls, that is a clean null result (see I).

### I. A table comparing the Spectre with periodic controls

As noted in an earlier review, "the Spectre behaves like any other graph" would itself be a publishable answer. Collect every observable so far in one table, with the Spectre value next to the square and honeycomb values. The observables are:

- vortex density;
- valence-4 density;
- β_c for Z₂ deconfinement;
- Bott indices;
- gap labels;
- H1 and H2, once they have run.

The table shows which properties actually depend on the monotile. It would be the central figure of the paper.

### J. Any flow model must obey Bell's inequality limits (a constraint, not a test)

Quantum predictions are statistical. That is a fact about quantum theory, not a gap that a model can fill. Any deterministic, local "energy flow on the tiling" model satisfies CHSH ≤ 2, but experiments reach about 2.8.

So a flow model on the monotile background must give up at least one of the following:

- locality in the tiling metric. This links to the idea in H that there is "no space between" entangled particles;
- independence of the measurement settings;
- single outcomes.

Before Part II makes claims about entanglement, it should state which one it gives up.

### K. Turn the speculative documents into tests

Read Brent's `QUANTUM_GRAVITY.md` (in spectre-monotile-py). For each claim, write down a quantity that can be computed on the tiling, and say what result would refute it. Defer any claim with no such quantity. Keep the "12 directions" count (±X, ±Y, ±Z with twists) as a counting statement, and compare it with the Dic12 holonomy group (order 48) from R4.

### Priority

1. G1–G2. They are cheap and use the existing mirror code.
2. I. A local table of results we already have.
3. H1, then H2. These are cluster jobs of up to 64 CPUs.
4. K, then J. These are writing tasks.
