# Fixes from the referee review

This branch (`paper-review-fixes`) applies the fixes proposed in the referee report
`research/spectre_review/REPORT.md` of github.com/romanov360/big-bang-monotile (branch `spectre-paper-review`).
The report reviewed commit `5d077c8`. This branch starts from that commit.

**Build status.** The generator `spectrepaper.py` was edited, and `spectre_substrate.tex` was edited by hand to
match it. The build environment used for these edits could not run the repository's code, so neither
`python3 spectrepaper.py` nor `--selftest` has been run.

Run, in order:

    python3 spectrepaper.py --selftest
    python3 spectrepaper.py
    python3 spectre_supplement.py --selftest
    python3 spectre_supplement.py
    make            # or the usual LaTeX build of both documents

What has been checked without running the code:
- All edited Python files parse.
- Braces and environments balance in the `.tex`.
- Every `\ref` has a label, and every `\cite` has a bibitem and vice versa.
- Every line of the edited `.tex` either occurs verbatim in the generator or is a line where the generator fills in a value. The count of such placeholder lines, 165, is unchanged from before the edits.

`spectre_supplement.pdf` is stale (it predates the current title) and must be rebuilt.

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

## Still open
- **G1 is not solved.** A proof of the perimeter recurrence for every k needs a boundary substitution for the Spectre supertiles. The paper now says so and lists it as further work.
- **(R) and (H)** remain unrefereed inputs. The abstract now says so.
- **The Flicker attributions** in Sec. 7.4 (no slide plane; 30 of 44 contacts) were left as written. They could not be checked against the full text of arXiv:2609.23783.
