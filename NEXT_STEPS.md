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
- **Holonomy figure** (`paper/figures/spectre_holonomy.png`; the script is not on master yet):
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

**R10 (carry-over). E:** a Galois-conjugate formula for seam dimensions. **F:** Chair44, a 3D counterpart. **B:** an exact GIFS from the self-similar control points.

### Recommended order
1. **Now:** R9 (C step 3), which is already scripted infrastructure.
2. **Next:** R1, which reuses `gap_collared.py` and is the strongest link to Brent's figure. Then R5 and R4, which are short exact proofs.
3. **Then:** R3 (cohomology with Z₂ coefficients) and R2 (vortex window), which together could give the "flux is topological" theorem.
4. **Parallel numerics:** R6 (linking) and R8 (flux Hamiltonian).
5. **Last:** R7 (genus robustness, cheap but low value), then E and F.

### Notes for syncing
- PR 13 (`swirl-paper-fixes`) conflicts with upstream master only in `spectrepaper.py`, in the abstract and the pancake section. Brent's phys rev IV–VI rewrote the same passages. Resolve that by hand with Brent before merging; it was not merged here.
- `spectre_holonomy.py` is referenced in the figures README but not committed upstream. Ask for it before starting R1–R4, so that the vortex classification matches his conventions exactly.
