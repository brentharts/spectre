# Fixes from the referee review

This branch (`paper-review-fixes`) applies the fixes proposed in the referee report
`research/spectre_review/REPORT.md` of github.com/romanov360/big-bang-monotile (branch `spectre-paper-review`).
The report reviewed commit `5d077c8`. This branch starts from that commit.

**Build status (run on this branch).**
- `python3 spectrepaper.py --selftest`: all checks pass.
- `python3 spectrepaper.py`: regenerated `spectre_substrate.tex`. The regenerated file is identical to the hand-edited one except one sentence, which the generator now fills from computed values.
- `python3 spectre_supplement.py --selftest`: all checks pass.
- `python3 spectre_supplement.py`: regenerated `spectre_supplement.tex`, now 71 Facts, 24 with a Lean mark.
- `make facts`: spectrefacts (62 facts), monotilefacts (8 checks), chair44facts (43 checks), `chair44_hull --tests` and `chair44_lines --tests` all pass. pyCICY was not installed, so the optional pyCICY cross-check of the perimeter was skipped; the referee's exact recount covers it.
- `pdflatex`: `spectre_substrate.pdf` and `spectre_supplement.pdf` (4 pages), with no errors and no undefined references
  or citations. The page count of 28 recorded here was wrong: the build at `989ef7f` gives 30 pages, and two passes
  still leave "Label(s) may have changed", so three are needed. See `PREPUB_FIXES.md` for the current build.

Build dependencies: Python with sympy, numpy, scipy, matplotlib and shapely, plus TeX Live (latex-base, latex-recommended, latex-extra, fonts-recommended, pictures).

## What changed (report item → fix)

### Inputs (E1)
- The introduction and Sec. 2 no longer say the matrix is "the one quoted input".
- They list the actual inputs: the published rule with its placement transforms, Tsiokos's solid, (R) and (H).

### Proposition 1 (E3)
- Restated as a coincidence for the Δ seed.
- It now records that Q₋ vanishes on Λ, Σ and Ψ.

### Twist, Fact 6 (E4)
- The quadrilaterals are stated to be not similar at any finite level, with the one-line reason.
- The twist is stated as a limit.
- The phase sentence is corrected (W4).
- A proof that θ_S is irrational is added (G6).
- The ledger method string is corrected.

### Boundary recurrence (E2, G1, G2)
- The recurrence is stated as verified exactly for k ≤ 6 and conjectured beyond.
- The closed form, the −16 invariant and the growth bound are stated for the recurrence sequence. For the perimeters they are stated only on the orders counted.
- "The boundary is fractal" became a conditional statement that names the two missing inputs.
- The abstract, Table 3 caption, unfolding section and conclusion follow.
- The supplement marks the Lean-proved boundary Facts with * and explains that Lean takes the recurrence as a definition.

### Proposition 2 (E5)
- The interface count now carries the Mystic perimeter Q.

### s* (E6)
- Transcendence is argued correctly: irrationality plus Gelfond–Schneider (new reference).
- The modelling caveat is stated.

### Proposition 3 (E7)
- Deleted.

### Orientation channels (E8)
- m = ±5 is added, with h = 1.
- Harmonics are listed over Z/12.

### Novelty (E9)
- The undecorated substitution is identified as the classical 3D chair (Robinson; Lee–Moody).
- The 2-adic route is credited to Lee–Moody and Baake–Moody–Schlottmann.
- Fact 22 is presented as an explicit worked example.

### Supplement (E10)
- Lists only this paper's Facts. The shared-ledger Facts of other documents are excluded through `OFF_PAPER`.
- Both authors are named.
- The contradictory Lean-count history is removed.
- The classification sentence in the paper matches what the supplement contains.

### Fact 23 (G7)
- The c² closed form is moved out of the Fact, stated as identified-not-proved.

### Missing proofs (G5)
- One-line proofs are added for the census recurrence (all seeds, all n) and for c(M) (all M).

### Wording and smaller fixes
- **W2, W3:** Jordan structure and convergence rate.
- **W5:** internal-plane parity.
- **W6:** the lift Fact reduced to what it says, and Kraus et al. removed.
- **W7:** the dyadic amplitude sentence.
- **W8:** 3·2⁻ⁿ.
- **W9:** depth two.
- **W10:** Table 3 rows.
- **W11:** the Hat/Turtle control.
- **W12:** the hierarchy section's physics language.
- **W13:** speculation removed.
- **W14:** "independently audited" → "audited by the second author".
- **W15:** Kamenetsky's 15 live enclosures.
- **W16:** Prop. 5 stated for the Z³ label system.
- **W17:** the k = 1 convention contradiction removed.
- **W18:** the p sentence.
- **W19:** c < N.
- **W20:** the histogram defined and the totals corrected.

### Bibliography
- arXiv IDs added for Smith et al.
- The journal reference added for Baake–Gähler–Mazáč–Sadun.
- New references: Baake–Gähler–Sadun (Hat family), Baake–Moody–Schlottmann, Robinson, Gelfond–Schneider.

### Lemma 1
- The paper now mentions the independent reproduction (360,189 blocks, 15,214 orbits, no failures).

### Self-test
- Checks that asserted the old overclaims ("the boundary is fractal", "read from k = 1", "independently audited") are replaced by checks for the corrected statements.
- New checks cover the twist, the recurrence status, Gelfond–Schneider, the removed Prop. 3 and the Mystic term.

## All-orders proof of the perimeter recurrence (G1, second round)
- **New Theorem 1**: for every k, P(k+3) = 5P(k+2) − 3P(k+1) − P(k) and P(k) = 2F(3k+5) + 4. The Mystic
  perimeter Q obeys the same recurrence from k = 1 on.
- **The proof is a finite exact certificate**, `boundary_certificate/check_spectre_boundary_proof.py` (`make boundary-proof`, about 2 minutes):
  - Pairs of placed order-k outlines are tracked by their relative placement, which is linear in the quad vector q_k.
  - The pair types close on 93 types.
  - Every other placement is proved separated at every order: radius bounds for k ≤ 60, and the exact
    eigen-decomposition of Φ on q_0 for k > 60. The smallest margin is 0.097 diagonals.
  - Non-overlap of the supertiles is proved on the way, not assumed.
  - Cayley–Hamilton on the 95 × 95 integer state matrix finishes the argument.
  - It uses the referee's independent transcription of the construction, not the paper's build.
- **Restated as proved for every order:** the recurrence, the −16 invariant, the growth bound P(k+3) ≥ 4P(k+2) + 30,
  the closed form, ν, and s* given its model. The abstract, the table caption, the conclusion, the further-work list
  (now three items), the ledger methods and the supplement's * legend follow.
- **Still conditional:** fractality of the limiting boundary, which needs convergence of the rescaled outlines and a
  covering bound.
- **Correction:** the paper said every one of the thirteen slot-pair contact sequences obeys the recurrence on the orders
  counted. Three of them (slots 1–4, 1–7, 4–7) fail it at k = 4 and obey it from k = 5. The text now says so.

## Flicker attributions (checked)
Checked against the full text of arXiv:2609.23783v2:
- **No slide plane.** Flicker gives a short argument: Superchair rules force slides to be multiples of 2, then 4, and so
  on, so the only admissible slide is zero. "Argues informally" is a fair description.
- **30 of the 44 contacts.** This is stated in the main text and in footnote 15, with no proof shown. "States" is
  accurate.
- The citation is correct as written.

## Third round: the remaining open items
- **(H) is proved in the paper** as Lemma `lem:hier`: Kamenetsky's Prop. 7.1 argument, written out, including the step
  that the central child is already in the enclosure.
  - Its finite conditions are checked by `hierarchy_certificate/check_chair44_hierarchy.py`, written independently of
    Kamenetsky's `cert6.py` and of `chair44_independent/` (about 1 minute).
  - Results: P₀ = 30, P* = 44 (exactly the 44 legal contacts), Σ(P*) = 2P*, 135 tight facet triples, 33 enclosures
    (18 dead, 15 live), (U) 15/15, (Pr) 1 + 14.
- **G2, fractality, is proved** (Theorem `thm:frac`).
  - The rescaled supertiles converge on each parity to a compact set K equal to the closure of its interior, and
    dim_B ∂K = log ν/log λ = 1.39925.
  - Certificate: `boundary_certificate/check_spectre_fractal.py`. The contact matrix has characteristic polynomial
    x⁸⁴(x−1)³(x+1)⁴(x²−4x−1); a grandchild is interior with margin 0.267; 20 contact types give the lower bound.
  - It uses in addition the existence of Spectre tilings of the plane (Smith et al.).
- **Ancestor tail, settled.** For edge-adjacent tiles the ratio P(h≥k+1)/P(h≥k) is exactly 0.394, 0.498, 0.528, 0.536, …
  and tends to ν/λ² = 0.538, not 1/2.
- **Window, settled (Theorem `thm:window`).** The previous version of this branch claimed the Spectre has no
  contracting-map window. That was wrong. The slot maps use only 60° rotations and the mirror, and the planes invariant
  under those form a family.
  - On each parity the expanding slot translations span a plane that the slot maps carry to the next parity's plane.
    The marginal parts are a coboundary.
  - So the window, read bottom-up, is a graph-directed attractor of similarities of ratio 1/λ, with shape matrix [[7,6],[1,1]].
  - Its contact matrix (1192 types) has characteristic polynomial (x−1)³⁵(x+1)³⁵x¹¹¹⁴(x⁴−10x²+1)², with spectral radius √2+√3.
  - Hence dim_B ∂W = log(5+2√6)/log(4+√15) = 1.110977. This is BGMS's CASPr value, now derived independently for the
    Spectre's own window (`boundary_certificate/spectre_window.py`).
  - The regularity and measure-disjointness of the windows are cited from BGMS.
  - The paper's earlier box count (about 1.2) is now stated as an estimate against the exact value.

## Still open
- **(R)** remains quoted, but Tsiokos's Lean development has now been **rebuilt independently** (`lean_rebuild/`).
  - The build ran on Modal at `d53d224`: `lake build` succeeds (8638 jobs).
  - The regenerated axiom log is byte-identical to the committed one.
  - No `sorryAx`; only `propext`, `Classical.choice`, `Quot.sound` and named `native_decide` hooks.
  - All controls pass, including the discharge diff audit (67/67) once the delivery archives are restored from `838bca5`.
- **Four-dimensional chair** (Kamenetsky's Problem 9.1): still open.
  - My own closure test (`chair_nd.py` on the review branch) reproduces Kamenetsky's Theorem B: 3 of 2187 assignments close.
  - It was then run on 12,784 sampled 4D assignments: half extend the 3D marking on the facet v₃ = 0, half are random.
  - None closed. Every one exceeded 400 contact poses, with no offsets.
  - A vectorised version of the test (`chair_nd_fast.py`, also validated on Theorem B) then drove a simulated-annealing
    search on |Σ(P₀)|: about 12,100 evaluations over four two-hour workers. The best value was 273, against 44 for a
    closed 3D assignment. None closed.
  - The search space has about 1.5·10¹⁶ assignments, so this rules nothing out in general.
- **Upstream open PRs**: see the PR.
