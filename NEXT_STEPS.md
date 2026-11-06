# Next steps after "phys rev II/III"

This plan reviews upstream commits `0fe4f43` and `bddcf47`, which change only `spectrepaper.py`. Line numbers refer to `spectrepaper.py` at `bddcf47`. No build or selftests have been run yet.

## A. Claims in `phys_pancake` that need correcting

1. **The boundary dimension is stated as an equality (l.2721–2723).** The section says dim_B(∂W) = log(√2+√3)/log λ. Only a lower bound is proved, and l.2375–2385 calls the equality a conjecture. Write "≥" or mark the statement as a conjecture.
2. **dim_H(∂W) ≥ 1.110977 (l.2830) is not proved.** Hausdorff dimension can be smaller than lower box dimension, so the box-dimension bound does not give it. Remove the claim, or prove it separately (see E3).
3. **The theorem "dim_H = dim_B of the seams" appears twice.** Cite the existing theorem in the tilings part instead of restating it.
4. **The swirl sign contradicts itself (l.2781).** It reads "±S₀ ≈ ∓0.43". Pick one sign convention, then check the value 0.43 against the script that produced it.
5. **The tile area formula has no derivation (l.2800).** The formula is Area = √3/2(a²+b²+ab√3). Check it against the exact Spectre area as a function of (a, b), and add a selftest.
6. **Check the 16×16 matrix Φ and its companion matrix C_Φ.** The constant term of χ is 1, so C_Φ is in GL(16,ℤ). Still to check:
   - that χ = (x−1)⁴(x+1)⁴(x⁴−8x²+1)²;
   - that Φ is conjugate to the substitution matrix the paper uses.
7. **"Configurational entropy ~ c ln L" has no derivation.** Derive it, or label it a conjecture.
8. **The "Kip Thorne 1967 pancake" attribution (l.2846) looks wrong.** "Pancakes" are Zel'dovich (1970), and the singularity work is Belinsky–Khalatnikov–Lifshitz. Find the correct reference.
9. **Several physics statements have no support in the tiling results:**
   - the Ashtekar SU(2)± mapping;
   - the entangled Ψ_universe;
   - the "holographic horizon" wording (l.2838);
   - the 2+1 Weyl tensor vanishing. It is a standard fact, but it has no connection to the tiling.

   Move these to a clearly labelled speculative subsection, or cut them.
10. **Restore the disclaimer that "nothing is derived".** It was removed from `phys_section`.

## B. Abstract (`abstract_new`)

11. **"Dynamically verified during compilation" overstates what the selftests do.** The selftests check text and a few identities, not the physics.
12. **The Big Bang Pancake, Weyl and Immirzi material reads as results.** Remove it from the abstract, or mark it as speculative.
13. **Make the abstract match what is proved.** This covers the dimension statement (see A1) and P(k) = 2F_{3k+5}+4.

## C. Structure of `document()`

14. **Three sections sit in the wrong place.** `phys_section`, `phys_pancake` and `conclusion_section` come between `\part{Quantum geometry…}` and `\label{part:qg}`. As a result the label points to the wrong place, and the conclusion sits inside Part II.
15. **`\appendix` and `\footnotesize` come before the quantum-geometry material.** So all of Part II becomes appendix, set in footnotesize.
16. **`full=False` drops the conclusion and physics sections.** References to `sec:conclusion` (l.2190) would then dangle.
17. **The new `toc` and `overview` flags are both off by default.** The overview figure (l.433) references `part:qg`. Test that every combination of the two flags compiles.
18. **Two open items for the OpenAI-release part:**
    - decide where it goes in the paper;
    - check the new `brentharts/openai_math_yang_mills` link (is it public, and does it match what we cite?) and pin it to a commit.

## D. References and selftests

19. **The `Romanov` bibitem is gone.** It was replaced with inline "Romanov (2026)". Make the citation style consistent, and confirm that no `\cite{Romanov}` remains.
20. **Run the selftests on the new master.** Check `tan θ_S` (l.3321) and the conjecture wording. Add tests for the new claims in section A.
21. **Build and log check.** Build the PDF and run all selftests there, then read the LaTeX log for undefined references.

## E. Window boundary (our own work)

22. **The closure run is in progress.**
    - If it gives ρ = √2+√3 to 1e-8 with no seeds on the edge of the search box, add a paragraph to the paper as numerical evidence for the upper bound. Then rebase `window-boundary-closure` onto upstream master and open the upstream PR.
    - Otherwise, report the result before any rerun.
23. **Prove the upper bound.** If the closure works, turn it into an exact certificate of the contact graph. That would prove dim_B(∂W) = log(√2+√3)/log λ.
24. **Prove the Hausdorff-dimension lower bound by mass distribution.** This is only needed if A2 stays in the paper.
25. **Package the boundary-certificate scripts** into one reproducible run, with output hashes.

## F. Process

26. **Split the work into two PRs:**
    - one for the structural, reference and test fixes (C and D), which should be uncontroversial;
    - one for the claims and the abstract (A and B), so the physics discussion does not block the first.
27. **Agree on a label for every claim:**
    - theorem with certificate;
    - numerical evidence;
    - conjecture or speculation.

## Order

Do items 21 and 20 first, then the C+D PR, then the A+B PR. Item 22 depends on the contact-closure run finishing.
