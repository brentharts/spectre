# Fixes from the prepublication review

This branch (`paper-prepub-fixes`) starts from `989ef7f` (`paper-review-fixes`). It applies the fixes proposed in
`research/prepub_review/REPORT.md`, which is on branch `paper-prepub-review` of this repository. The `.tex` files are
regenerated from the generators and ledgers; none was edited by hand.

## Build status (run on this branch)

| step | result |
|---|---|
| `python3 spectrepaper.py --selftest` | all checks pass |
| `python3 spectre_supplement.py --selftest` | all checks pass (one new check: the count of numbered Facts) |
| `make facts` | passes: spectrefacts 62 facts, monotilefacts 8 checks, chair44facts 43 checks, chair44_hull and chair44_lines tests (pyCICY not installed, so its optional cross-check was skipped) |
| `make lean` | `SpectreFacts.lean`: 67 theorems, kernel accepted (Lean 4.33.1) |
| `boundary_certificate/check_spectre_fractal.py` | passes; output in `out_spectre_fractal.txt` |
| `boundary_certificate/spectre_window.py` | passes; output in `out_spectre_window.txt` |
| `pdflatex` ×3 | `spectre_substrate.pdf` 31 pages, `spectre_supplement.pdf` 4 pages; no undefined references, no "may have changed", no overfull boxes in either |

## Major

**M1. Theorem 3 (window).** Fixed with option (b) of the report, restating the theorem as what is proved. The proof is
now self-contained and no longer cites BGMS for regularity.
- **Positive area and measure-disjointness, now proved.**
  - The certificate now asserts that no contact type is the identity. The nearest candidate is at distance 4.11.
  - Pieces of depth m within δλ⁻ᵐ of each other have a live type. So a disc of radius δλ⁻ᵐ/2 meets at most 1193
    pieces.
  - The mass distribution principle then gives positive area.
  - Measure-disjointness follows from the Perron argument.
- **The theorem now states:**
  - the seams have box dimension exactly log(√2+√3)/log λ = 1.110977…;
  - the lower box dimension of ∂Z_p(S) is at least that value for all four window types. The certificate now checks
    that the lower bound reaches all four.
- **A new paragraph after the proof says what is missing for ∂W itself:**
  - an upper bound needs the neighbouring windows of a covering of internal space;
  - BGMS's lattice tiling for CASPr would supply this once W is identified with their window;
  - this is not done here.
- The abstract and the conclusion are reworded to match.

**M2. The supplement.**
- The "beyond k=6 the recurrence is a conjecture" paragraph is replaced. It now says that `charpoly` is the one input
  and that `boundary_recurrence` is proved by Theorem 1.
- "Every Fact of the paper — all 71" now reads "ledger entries", with the count of numbered Facts (25) given
  separately.
- `internal_plane` now says the 1.2 is a box count and points to the exact seam value. Its docstring and method say
  the same.
- `chair_hull_theorem` now reads C₂^diag.

**M3. Stale statements.**
- Introduction: "one quoted input and two finite computations".
- Conclusion: one unrefereed input, (R), plus Lemmas 1–2.
- Sec. 5: "for every order (Theorem 1)".
- The box count's patch level: "a level-5 patch; Figure 3 shows level 4".
- Scope paragraph: Lee–Moody [LeeMoody] with [LMS] for the hull.
- Look-back paragraph: the count is c_O G_O(ℓ) + c_G G_G(ℓ), with both glue formulas given, and the Mystic
  convention is stated (with the other convention, the first ratio is 0.368).
- Abstract: "with ratio tending exactly to ν/λ²".

**M4. The 4D paragraph.**
- 1566 is now called Kamenetsky's random-start plateau, and 273 is said to be above his structured-family value of 158.
- The 400-pose cap of the test is stated, and so is the ≤300 cap in Kamenetsky's Theorem B.
- The paper now says that the search records are not part of its build.
- The search code and logs are now in `nd_search/` (copied unchanged from the authors' working repository), with a
  README. The counts are checked against the logs: 4288+4212+4284 = 12,784 assignments in the search and
  2399+3208+3424+3101 = 12,132 evaluations in the annealing. The paper now says 12,132 rather than "about 12,100".
- The claim that the 3D score is minimised exactly by the closed assignments was not verifiable without running the
  external code, so it is removed.

**M5. Lean rebuild.**
- The paper now says that the 162 regenerated reports agree with the published log theorem by theorem, up to the
  namespace prefix, and that the post-build log equals the published one byte for byte.
  - I checked this by fetching `lean/R44/build_axioms.log` at `d53d224` from GitHub. It is identical to
    `committed_axioms_log` in `r44_build_result.json` (53,556 bytes).
- A sentence on statement fidelity is added, and so is one on trusting the compiler.
- `lean_rebuild/README.md`: the old diff-audit row now says that its output was not saved.
- The statement-fidelity claim is now checked by `lean_rebuild/discharge_audit_indep.py`, written for this review
  without importing the project's audit script. Output: `lean_rebuild/out_discharge_audit_indep.txt`.
  - All 67 Proved files differ from the newest delivery containing them only in marked compile-fix hunks.
  - 76 declaration texts differ. All are notation (parentheses, `|x|` to `abs`, `section` to `«section»`, spacing),
    five structure-valued `theorem`s that became `def`s, one delivered `sorry` placeholder replaced by a call to the
    proved theorem of the same statement, or added helper lemmas.
  - The paper now says this, citing the script.
- The misleading comment in `r44_build.py` is corrected.

## Minor

- **m1.** In the proof of Theorem 2:
  - It says which tiling is used: a limit of supertiles re-centred at deep tiles. Their neighbour placements are
    live types.
  - It cites the Spectre paper [SmithEtAl2, Thms 2.2, 5.1] instead of the Hat paper.
  - The lower bound names the boundaries involved.
  - `check_spectre_fractal.py` now asserts that O–O sibling pairs reach the block, so the bound applies to K and not
    only to the Mystic limit.
- **m2.** Lemma 1 uses Kamenetsky's labels (C1)–(C3), with (C2) (tightness of the face rules) stated separately.
- **m3.** `spectre_window.py` now asserts key separation (at least 1.909) in place of the dead `ds` line.
- **m4.** The BGMS paraphrase is corrected. Their value is a Hausdorff dimension, and their remark is about uniformity
  along the boundary.
- **m5.** Fact 24: the ±2 and ±3 sums are bounded, and the ±4 sums grow linearly, so h = 0 there.
- **m6.** Build fixes:
  - `make paper` and `paperfast` run pdflatex three times;
  - `make paper` also builds the supplement;
  - the supplement's 11 overfull boxes are fixed in `facts_appendix`: top-level `\text{}` is set as running text, and
    long filenames may break after underscores;
  - the page count in `REVIEW_FIXES.md` is corrected.
- **m7.** The `boundary_exact` method now cites `boundary_certificate/spectre_exact.py`, the exact recount for k ≤ 6.
  pyCICY is described as optional, and the selftest message is updated to match.
- **m8.** The `SpectreFacts.lean` comment now says that only the two inequalities are proved, not fractality.
  - A first wording of it began a comment line with "theorem", which made the theorem count read 68. The comment is
    rewrapped and the count is back to 67.
- **m9.** The conclusion now says "One further problem remains open".
- **m10.** The supplement's author name now matches the paper: "Brent Hartshorn".
- **m11.** `chair44_data/ATTRIBUTION.md` no longer lists Flicker's paper as a source of the data.
- **m12.** The third implementation of Lemma 2 is now in the repository: `hierarchy_certificate/chair44_blocks_indep.py`,
  with its output and a README section. The paper cites it there instead of the private repository.
- **m13.** The Figure 2 caption now reads "by Theorem 1 they equal the recurrence".
- **Bibliography.** BaakeHat now carries its journal reference: J. Math. Phys. 66 (2025) 092707.

## Not changed

- The upper bound for ∂W (option (a) of the report) is not proved.
  - The internal coordinates of `spectre_window.py` are a projection along the expanding planes E_p. That projection is
    not a Galois star map: no Galois embedding of Z[ζ12] (σ = 5, 7, 11) vanishes on E_0 or E_1, and neither does the
    physical one (|σ(e)| ≥ 1.48 on the basis vectors).
  - BGMS show that the CASPr tilings are mutually locally derivable from model sets; they do not use this projection.
    A covering of internal space by translates of W would follow from their work only once W is identified with their
    window, which the paper already says is missing.
  - Their window-boundary value is a Hausdorff dimension, which does not bound the upper box dimension of ∂Z_p(S), so
    even with the identification it would not give option (a) by itself.
  - Option (b), stating only what is proved, stands.

## Second and third review rounds

A fresh adversarial pass over the fixed paper found the following. All are fixed on this branch.

- **R2-1 to R2-7** (round 2):
  - the exact interface law, with its Mystic term, replaces the approximate one in `spectre_boundary.py`, and the selftest
    checks it at depths 3 and 4 (depth 5 was checked separately);
  - the paragraph after Proposition `sstar` gives the measured drift (2.66 at depth 4, 2.41 at depth 5, against μ ≈ 1.859)
    and says that the exact closed form reproduces it;
  - three places that called Tsiokos's result proved now say it rests on his claimed proof, which is quoted and not
    checked here;
  - the remaining round-2 items were wording fixes; R2-5 needed no change.
- **Corner lines.** The bounded/linear behaviour of the d_z sums on the (0,0,1) and (1,0,0) corner lines is now argued in
  the text (the d_z values on the closed class run +1,+1,−1,−1,…, so the sums stay in {0,1,2}; on (1,0,0) every letter has
  d_z = 1, so the sums equal L). `chair44_lines.py --tests` now checks both and that τ_(0,0,1) sends its closed class onto
  two words.
- **Centre diffraction (round 3).** Fact `centrediff` now defines the amplitude a(k) = lim 8⁻ⁿ Σ_{x∈X_n} e^{ik·x}, so
  a(0) = 1, and the Bragg intensity is |a(k)|² times the squared density 1/49. Before, the normalisation was left implicit
  and Σ I = 1/7 did not follow from the text.
- **References.** Every arXiv entry was checked against its arXiv record (authors, titles, journal references for
  BaakeSpectre, BGS and BaakeHat). The numbers the paper quotes from Kamenetsky (Problem 9.1, 2187, 3 closed at cap 300,
  30→44→44, plateau 1566, structured family 158) match the arXiv text. The Chéritat–Ma web page could not be fetched from
  this environment (the proxy returns 403), so that entry is unverified.
- **Supplement.** Its counts (25 numbered Facts, 71 ledger entries, 24 Lean-marked, 67 theorems) match the paper, the
  ledgers and `SpectreFacts.lean`. Its rows for `chair_lines`, `centre_diffraction` and `internal_plane` agree with the
  paper.

## Fourth review round

Every proof and computed claim in the paper was read again against the code. These were found and fixed.

- **Aperiodicity overclaim.** The paper said aperiodicity of the Chair44 is "re-proved". It is not: Section 6 shows that no
  tiling obeying the contact rules has a period, so aperiodicity of tilings by the solid follows only given Tsiokos's
  registration theorem (R), which is quoted. The abstract, chair intro, hull scope, caveats and conclusion now all say
  this, and the einstein property stays Tsiokos's claim.
- **Prop sstar.** The range is now 1 ≤ c ≤ N, the proof is given, and the window measure no longer reuses μ.
- **Symbol clashes.**
  - Thm sym proof: the cycle length p became q, and μ became η.
  - Prop lineaction: ν became ℙ, the ergodic measure became ℙ′, and the translation T^v became S^v.
  - Fact channels: the eigenvalue μ became ϱ.
  - Section 8: the level ℓ became j, and the probability P became Pr. P(k) is the perimeter.
- **Fact orientrec.** It now names the Δ entry as the orientation sum of the supertile, and says the growth holds for odd
  n as well as even n.
- **Parity paragraph.** θ_o, θ_t and φ are now defined.
- **Prop stack (chair stacks).** It said the seven lines have length 2ⁿ−2. The induction gives 2ⁿ on the main diagonal and
  2ⁿ−1 on the other six. This was checked on the depth-5 cube for all six labels. The proof now lists T and counts the
  cells.
- **Corner lines.** "each [closed class] reached from every label" is false for b = (1,1,1), whose six closed classes are
  fixed letters. The text now says that every label leads into a closed class, which is what the code checks.
- **Unfolding.** "Cutting along a spanning tree, 23 cuts" was wrong. The 23 tree edges are the hinges, and the other 25
  edges are cut. The number is computed by the template.
- **Notch sums.** "The notch sum is 2ⁿ" now reads 2ⁿ(1,1,1). |S|²/N = 2⁻ⁿ is stated per component, in the text, the
  table and the conclusion.
- **Fact orientsum.** It now points to the exact recursion of Fact orientrec, instead of saying "with no proof".
- **Slice figure, panel (c).** The unweighted |F(0)|² was drawn and captioned as 8ⁿ. It is 64ⁿ: 8ⁿ is I(0)/N. The plot is
  corrected in `chair44_findings.py` and `chair44_slice.py`, the figure is regenerated, and the caption is fixed.
- **Layer interface.** "Tends to 4 per level" now reads "approaching 4 over the levels computed", because the formula is
  checked only for n ≤ 4.
- **Outline comparison.** "By a 20% margin only for k=3" was ambiguous. It now states the criterion: the rotation distance
  is below 0.8 times the reflection distance.
- **Code docstrings.** `chair44_lines.py` (line lengths) and `chair44_unfold.py` (hinges against cuts, floating-point
  positions) are corrected.
- **Build.** The selftest passes, pdflatex ×3 gives 32 pages, and there are no overfull boxes and no undefined
  references. The `chair44_lines`, `chair44_unfold` and `chair44_slice` tests pass.

## Fifth review round

- **Abstract.** It used ν before defining it. It now says $(2+\sqrt5)/\lambda^2$.
- **Introduction.** The description of Propositions said they are "computations with a modelling step". Several are
  proved. It now says they are narrower in scope, proved or computed, with their hypotheses stated in them.
- **Boundary closed form.** "Multiplying the Fibonacci index by three" now reads "advancing". The "third root of
  x²−4x−1" (a quadratic) now reads "the remaining root of (x−1)(x²−4x−1), the conjugate of φ³".
- **Direction characters.** "The remaining two characters of Z/6" now names them: the Z/3 and faithful characters, each
  up to complex conjugation, which matches the table caption.
- **Orientation spectrum.** The rate (2/8)^d used d, which already denotes the dimension. It is now (2/8)ⁿ in the depth n.
- **Unfolding.** ℓ was used without a definition. It is now defined as the tile scale: √(tile area) for the Spectre and
  7^{1/3} for the chair, as in `chair44_findings.py`.
- **Lines section.** "Everything below is an exact finite computation" is not right for the proved propositions. It now
  reads "proved from, or is, an exact finite computation".
- **Notch mean.** "The chair's is exactly 2ⁿ" now reads "exactly 2ⁿ in each component".
- **Fine boundary operator.** The conclusion claimed something about the components by species and edge index that the
  body never stated. A paragraph now gives it, rerun with `spectre_golden.py` here:
  - there are 52 edge types, and 34 of them obey the recurrence after the first term;
  - 14 are bounded and have period 2;
  - (Π,3) and (Π,4) give 1, 7, 22, 91, 378, 1599, which obeys neither the recurrence nor its order-4 extension.
- **Build.** The selftest passes, pdflatex ×3 gives 32 pages, and there are no overfull boxes and no undefined
  references.

## Sixth review round
- **Supplement preface.** The Propositions wording now matches the paper's introduction. The source of the claims is
  now "the ledger modules" rather than `spectrefacts.py` alone, since the table also draws on `chair44facts.py` and
  `monotilefacts.py`. The description of the truncated method column now also mentions the cut at about a hundred
  characters, marked by an ellipsis.
- **Supplement ledger.**
  - The orientation_sums claim was S_n^chair. The method computes only the z component of the notches, so the claim
    is now S_{n,z}^chair.
  - "Romanov (this paper)" now reads "the second author, cited draft".
  - The chair_twist method no longer cites a hard-coded "Fact 11".
- **Bibliography.**
  - Kamenetsky now carries its year.
  - The Romanov entry said "this repository". It now names the brentharts/spectre repository.
- **χ_∂ forward reference.** The text factorised χ_∂ before Fact bdychi defined it. It now names the characteristic
  polynomial of the perimeter recurrence and refers ahead to the Fact.
- **Internal plane.** "The expanding part (Φ² = 4+√15)" equated a matrix with a number. It now says Φ² acts as 4+√15
  on that part.
- **Depth of the 24 orientations.** Fact chairtwist said all 24 rotations occur at depth two, while the symmetry
  section said depth three and the code checked depth three. Recomputing gives 7, 24 and 24 distinct frames at depths
  1, 2 and 3. The code now checks depth two, and the text says depth two in both places.
- **Checked without change.**
  - The centre-set Parseval sum: 21/4 per level, total 1/7.
  - The amplitudes at J=1.
  - The orientation constant: c² is a root of the stated quartic to 28 digits.
  - The channel exponents.
  - 5376 = 12(8³−4³).
  - The notch-sum and interface sequences.
  - The tail ratio ν/λ² = 0.538.
  - The proof of Proposition lineaction.
  - The net counts: Euler characteristic, 23 hinges and 25 cuts, 384 cube trees and 11 nets, minimum perimeter 20.
- **Build.** The selftest passes, pdflatex ×3 on both documents gives no overfull boxes and no undefined references,
  and `chair44_twist.py` passes.

## Seventh review round

A full re-read of the regenerated paper, checking every number and every proof step against the code or by hand.

- **Notation clash.** In the proof of Theorem bdy, the 95×95 transfer matrix was called $B$, the same letter as $B=D\bar D$
  in Fact twist. It is renamed $\mathcal{B}$.
- **Fine decomposition.**
  - The text now states the range counted: placed tiles, $1\le k\le6$, with the single tile at $k=0$ dropped.
  - The $\Psi$ sequence is corrected to $0,5,31,141,607,2581$, recomputed from `spectre_golden.component_audit`.
- **Proposition sstar.** $B_c$ is now defined as the number of interface edges at coarseness $c$.
- **Orientation matrix of the Chair44.**
  - "The minimal polynomial divides $x(x-8)(x-2)$" now says "is $x(x-8)(x-2)$". All three roots occur, so the two statements are equivalent, and the paper now says the sharper one.
  - "Both need an inflation that is a Pisot number" was wrong as written, because every integer above 1 is vacuously a Pisot number. It now says "an irrational Pisot number, with conjugates to supply a Euclidean internal space".
- **Proof of Theorem hull.** The window offset was written $v-v'+(1,1,1)$, which mixes the scales. It now reads
  $2^{-n}(v-v')+(1,1,1)$, in units of the level-$n$ lattice.
- **Checked by hand and found correct, with no change needed:**
  - Lemma axes, and the unrolling identity $x_n=y_0-2^ny_{n \bmod q}$, including the strict bound $|J_i|<2^q-1$ and $J_i\ne0$.
  - The 2-adic limit $-y_0$.
  - The stack line-length formula, and the 14 child cases.
  - The subset-automaton recurrence $c(M)=28\cdot2^M-48$.
  - The Parseval sum $\frac1{49}(1+\sum\frac{21}{4}8^{1-J})=\frac17$.
  - The trace check of $\sum H_i=2I$.
  - The orientation-sum values against $|S|^2/N$, and the minimal polynomial of $c^2$, by sympy.
  - The contact-matrix parity, the chair face counts and the interface lengths $4^{n+1}-4\cdot2^n$.
  - The nets, and the cube's 384 spanning trees.
  - Every bibliography entry is cited.
- **Build.** The selftest passes. pdflatex ×3 on the paper, the note and the supplement gives no overfull boxes, no undefined references and no errors.

## Eighth review round

- Companion note: the "artefact of the two examples" retirement condition now records that a planar
  counterexample of the other kind already exists (the Socolar–Taylor hexagonal monotile, inflation $2$,
  model sets with a $2$-adic internal space, Lee–Moody 2013), and states that the dichotomy can survive it
  only as a claim about connected tiles whose shape alone forces aperiodicity.
- Note bibliography: added `SocolarTaylor` (JCTA 118 (2011) 2207–2231) and `LeeMoodyTS` (Symmetry 5 (2013)
  1–46), both checked; fixed missing punctuation after the Smith et al. title and after "B.~Hartshorn".
- Main paper and supplement checked for claims conflicting with the Taylor–Socolar example: none.
- Selftest: all checks pass; three documents: 0 overfull, 0 undefined, 0 errors.

## Ninth review round

- §8.4: removed "centres lie on a lattice with defects" (wrong: they lie on two cosets of $2\mathbb{Z}^3$,
  Fact `chair_slices`); the cause of the rate difference is now stated as not analysed.
- §7.3, ledger `chair_coincidence`, `chair44_twoadic.py`: $c(M)=28\cdot2^M-48$ and $c(M+1)=2c(M)+48$ hold
  from $M=1$, not only from $M=3$ (checked: $c=8,64,176,400,\dots$); the range is now stated as $M\ge1$
  everywhere, and the script's check now starts at $M=1$.
- Supplement: `<` and `>` in the method column rendered as "¡" and "¿" (OT1 encoding); added
  `\usepackage[T1]{fontenc}` and `lmodern`.
- Bibliography (paper and note): put every entry in one style (authors, \emph{title}, journal
  \textbf{volume} (year) pages; arXiv; URL). Mixed "Author (year). Title." entries are converted; there are
  no changes to the content of any entry.
- Re-verified the main paper's numerical and algebraic claims (Q± values, twist, perimeter closed form and
  bound, dimensions, minimal polynomials, chair interface counts, automaton counts, Theorems 4–5, Facts
  20–25). No mathematical errors were found.
- Selftests (paper and supplement): all checks pass. The three documents: 0 overfull, 0 undefined, 0 errors.

## Tenth review round

- Lemma 2: the paragraph after the lemma now lists the implementations in order: the original computation,
  the certificate implementation, `chair44_hull_admissible.py`, then the referee's `chair44_blocks_indep.py`,
  followed by the non-vacuity check. Reworded to remove an overfull line.
- §11: "an independent implementation reproduces" Lemma 2 corrected to "three further implementations,
  described after the lemma, reproduce", which matches §7.4.
- Re-read the whole main paper against its ledgers (§§1–11, references). No other errors were found.
- Selftests (paper and supplement): all checks pass. The three documents: 0 overfull, 0 undefined, 0 errors.

## Eleventh review round

- Companion note, Remark 1: the elliptical "if the third spatial dimension, … and if look-back time" is now
  "if it is the third spatial dimension, … and if it is look-back time".
- Re-read the companion note and the supplement in full. Checked every closed form in the supplement table
  by hand: the Pell unit, λ, the channels, the contraction, φ⁴, φ³, the Fibonacci form and its seeds, the
  boundary invariant, the growth bound (equality at k = 0), cos 2θ_S from tan θ_S, c ≈ 1.4489, the census
  identity, the mesh and net Euler characteristics, the surface/shared identity, the chair exponents, and
  Σv = 1 for the frequency vector in Z[√15]⁹. The Lean marks count 24. No other errors were found.
- Selftests (paper and supplement): all checks pass. The three documents: 0 overfull, 0 undefined, 0 errors.

## Twelfth review round

- `hierarchy_certificate/README.md` called `chair44_blocks_indep.py` the "third implementation", but the paper
  lists four implementations of Lemma 2. It now reads "a further implementation", sharing no code with the
  others, which matches §7.4 and §11.
- Re-read the passages changed in rounds 10–11 as rendered (Lemma 2 paragraph, §11, note Remark 1). No other
  errors were found. Documents only; no generated file changed.

## Thirteenth review round (convergence)

- Checked the repository's other READMEs and the chair44_independent notes for implementation counts or wording
  that rounds 10–12 changed in the paper: none remain. Nothing was found, so the review has converged.
