# Independent referee report: Chair44 hull (C1), Chair44 symmetry groups (C2), Hat-point Mystic classes (C3)

Date 2026-10-25. Referee: Claude (AI), working adversarially. All numbers below come from my own code in
`research/independent/` (see its README). I wrote that code from the primary definitions before opening the authors'
scripts. Inputs: Tsiokos data at `ioannist/six-birds-tiles@d53d224`; the authors' code at `brentharts/spectre@82a80d5`,
`brentharts/six-birds-tiles@d972ac4`, `romanov360/spectre@4900054` (branch mystic-six-classes); this repo's draft at `research-plan@c41fa0a`.

Grades used: **proved** = there is a human-checkable argument; **computer-verified** = an exhaustive finite check, and I say
exactly what was checked; **conditional** = holds given a named input; **unverified/false**.

## 1. Verdicts

| claim | verdict | exactly under which assumptions |
|---|---|---|
| C1 Theorem (hull) | **TRUE, conditional** | on (R) = Tsiokos Thm 5.6 (registration plus face contacts in A44). (H) is no longer needed as an outside input: I re-derived its finite part (§2.3). For "registered chair tilings with all face contacts in A44" the theorem is **unconditional**, computer-verified. |
| C1 Corollary (spectrum etc.) | **TRUE after rewording, conditional as above** | "the space of Chair44 tilings is minimal" is **false as written** under the translation action (E1). The correct form is in §2.5. |
| C2 (symmetry groups 1, C2diag, C4, D4) | **TRUE, conditional** | on (R) and on "Q has no self-isometry" (Tsiokos App. E; its finite part I re-checked). The proof has four repairable slips (E2–E5). None changes the result. |
| C3 (≤6 Mystic classes, = 6 for n ≥ 3) | **TRUE, unconditional for every n ≥ 3**, computer-verified + proved | Registration for all n follows from a finite-state induction I supply (§4.2). The authors' p6m lemma, as written, is an incomplete argument (G6), but it is not needed. Mathematical significance is low (§4.4). |

## 2. C1: every Chair44 tiling lies in the hull

### 2.1 Computations reproduced independently (all agree with the authors)

| item | my result | how |
|---|---|---|
| children partition 2P | 56 cells, disjoint, = 2P | integer cells, `c44core.py` |
| face-neighbour poses / geometric atlas | 2388 → **44**, all proper; equals certificate `legal_contacts` as a set | pyramid apexes from `r44_solid.json` `patches`, outward normals recomputed from P |
| atlas **without the pyramid data** | contact closure of σ: 21→30; Kamenetsky coarsening closure 30→44, Σ(P\*) = 2P\* exactly, every Σ-translation even; facet rule 135 triples; tight (P_F\* = P\*); **P\* = geometric atlas** | `c44facet.py` (needs only the 8 children) |
| legal-block closure | 168 → 600 → 1278 → 1398 → **1410**, stable; = all 2×2×2 windows of σ⁵(j), j = 1..168; Win(σ(L)) = L | `c44sub.py` |
| admissible blocks | **360,189**; L ⊂ Adm | `c44adm.py` (pairwise test: same chair, or disjoint and not face-adjacent, or face-adjacent with relative pose in A44) |
| Lemma A | **0 failures**; moreover Win(σ(Adm)) = L exactly | `run_c1.py` |
| primitivity / coincidence | least primitive power 3; 336 coincidence addresses at depth 3, (0,0,2) among them; none at depth ≤ 2 | `c44coinc.py` |
| contacts visible in legal blocks | exactly the 30 hierarchy contacts | `c44contacts_in_L.py` |

So the 44-atlas is not tied to one data file. It is determined by the substitution alone, through two independent
routes (pyramid geometry, and frame-marking coarsening closure). The label, pose and atlas conventions are consistent.

### 2.2 The proof

(R) as Tsiokos states it (Thm 5.6: one isometry registers every tile with proper frame and integer shift, and "every pair of
tiles sharing a feature is in a relative pose of A44") does imply that every block is admissible:
- each of the 24 exposed unit panels of P carries 8 features (checked: 24 panels × 8 apexes);
- registered face-adjacent chairs share whole unit panels, so they share features.

Mirror images: the registering isometry may be improper. After it, all frames are proper (homochirality, Thm 7.4).
"Up to isometry" covers this. Neither the 168 labels nor the atlas needs improper frames: the 2388 candidates include
all 48 orientations, and none of the improper ones survive.

The logic of Theorem 1 is valid, including for tilings with fault planes or non-exhausting hierarchies:
- No exhaustion is used. Each patch of diameter < 2^(n−1) lies inside σⁿ of one 2×2×2 block β of Dⁿ(T).
- β is a window of σ(β′), with β′ a block of Dⁿ⁺¹(T).
- Every Dᵐ(T) is again a registered A44-tiling, so β′ is admissible.
- Window bookkeeping: σ(β′) is 4×4×4 and its 27 windows are its offsets {0,1,2}³. The window at a level-n vertex v is
  the one at offset v − (level-(n+1) vertex nearest v) ∈ {−1,0,1}³. Both my code and the authors' use exactly this.

### 2.3 (H) is reproducible: the dependence reduces to (R)

`c44shells.py`, under the 44-atlas:
- **33** first-shell enclosures. Each is compatible with **exactly one** of the 8 candidate parents.
- Central completion: 1 enclosure has the root central. 18 admit no compatible enclosure of the proposed central child.
  In the remaining 14, every compatible central enclosure contains all 8 children.
- With Σ(P\*) = 2P\* and all-even translations (§2.1), Kamenetsky's Prop. 7.1 argument (short and human-checkable)
  gives (H) for every registered A44-tiling, and the parent tiling D(T) is again a registered A44-tiling.

So C1 needs only the registration theorem (R), which is the geometric half of Tsiokos's paper and his own top review priority.

### 2.4 Negative controls (the lemma is not vacuous)

`c44controls.py`:
- (a) random 8-label blocks: 0 / 20,000 have all child windows legal.
- (b) 3,000 legal blocks with one cell relabelled, kept consistent and disjoint but violating the atlas: 2,536 have an
  illegal child window.
- (c) atlas enlarged by one *random* illegal face contact (2 samples before I stopped the run): Adm stays 360,189 and the
  lemma still holds. A random non-atlas contact never fits into an otherwise admissible vertex block, so this control
  is uninformative.
- (c′) atlas enlarged by the 5 non-atlas contacts of one violating block from (b): Adm = 360,213 and **24 lemma
  failures**. The lemma does depend on the exact atlas: weakening the atlas breaks it, so the check is not vacuous.

The authors' controls (69% of random blocks illegal; 1958/1983 perturbed blocks) test a different, weaker thing: illegal
*windows*, not failure of the lemma under a weakened atlas.

### 2.5 Errors and gaps in C1

- **E1 (overclaim, Corollary).** The corollary says "The space of Chair44 tilings is minimal and uniquely ergodic". This is
  false for the R³-translation action on the set X of all tilings by Q:
  - X = E(3)·Ω_σ is a continuum of disjoint closed translation-invariant pieces gΩ_σ (one per coset of the 48 signed
    permutations in O(3)), so it is not minimal.
  - Correct statement: "every tiling by Q is, up to an isometry, in Ω_σ; Ω_σ (with the R³ action) is minimal and uniquely
    ergodic with pure point dynamical spectrum; X is a single E(3)-orbit closure."
  - Ω_σ is invariant under the 24 proper frames: L is O-invariant (checked).
  - Also write "the hull Ω_σ is the hull of a regular model set". Do not write "the space is", and do not write "every
    tiling is limit-periodic": singular elements of a model-set hull need not be model sets.
  - "Every tiling is pure point diffractive" is correct by unique ergodicity, for any MLD choice of scatterers. Say so.
- **G1 (conditionality hidden).** The abstract of `chair44_hull.tex` ("We answer both") and the README of
  `brentharts/six-birds-tiles/verify/hull` state C1/C2 without the condition (R). Tsiokos marks the alignment argument
  as unreviewed. His Lean development is LLM-built and uses `native_decide` (compiler trust).
  - Fix: theorem = unconditional for registered A44-tilings (equivalently Kamenetsky's F\*-tilings / Flicker's arrow
    rules), and conditional for Q.
- **G2 (Lemma A not human-readable).** 360,189 cases.
  - Reduce by O-symmetry (L and Adm are O-invariant, so about 15k orbits) and publish the orbit list with a checker.
  - Better: an argument per parent residue class.
  - The consequence "only the 30 hierarchy contacts ever occur, 14 atlas contacts never do" is worth stating: it is
    checkable, and Flicker asserts it without proof.
- **G3 (novelty).**
  - Flicker (arXiv:2609.23783, p. 3) already argues informally that slide planes are impossible ("slides … multiples
    of 2 … 4 … hence the only admissible slide is the zero vector") and that only 30 of the 44 contacts occur.
  - Goodman-Strauss (2609.24779) gives (H) in Berger style.
  - The draft must cite Flicker's slide argument and position Lemma A as the rigorous version. Neither paper proves hull membership.
- **G4 ("independent data").** `symchair44.py` "reads nothing", but it hard-codes Tsiokos's 8 children.
  - Its selection rule ("max kernel, then most orientations, then first in sorted order") is tuned to pick them.
  - That is independent *code* on the *same substitution*, which is correct and necessary (σ defines Q), but "from
    independent data" overstates it.
  - Kamenetsky's Theorem B (the assignment is unique up to conjugation) is the right citation.

## 3. C2: symmetry groups

### 3.1 Computations reproduced (agree with the authors)

| item | my result |
|---|---|
| σ commutes with all 24 rotations; L invariant under O | yes / yes |
| Φ-periodic legal blocks | 27, all fixed; stabiliser orders {1: 24, 8: 3} |
| G_K, C2 coordinate axis | 18 nodes; cyclic SCCs are 6 self-loops with labels −1, 0, 1, −1, 0, 1 (constant) → **no mixed SCC**; full stabilisers {2:12, 4:2, 8:4} |
| G_K, C4 | 4 nodes, one SCC carrying labels {−1, 0, 1} |
| G_K, C2 face diagonal (both (1,1,0) and (1,−1,0)) | 14 nodes, one 5-node SCC carrying {−1, 0, 1} |
| no self-isometry, finite part | of the 6 carrier symmetries, only the identity preserves all 192 apexes |
| explicit tilings | from the 2-cycles (0,1) (C4) and (−1,0) (C2diag), at level 6 (299,592 tiles). Over all isometries mapping a near-axis chair to a chair and the radius-40 region (69,581 resp. 43,329 tiles) into the patch: exactly 4 resp. 2 symmetries, the predicted K |

The authors' patch check (`chair44_symmetry_census.py`) tests only rotations about v₀ with zero translation. Mine
enumerates every chair-to-chair isometry. Neither finite check can exclude a symmetry about a far point on the axis.
That exclusion comes from the 2-adic argument, not from the patches.

### 3.2 Proof review

Correct as stated:
- the embedding Sym(T) → O, via injectivity of the linear part, from Per(T) = 0;
- the 2-adic Lemma (1)–(3). I re-derived it. For the quarter-turn the condition is even stronger:
  δ ∈ {(0,0), (2ⁿ⁻¹, 2ⁿ⁻¹)} mod 2ⁿ.
- the monotonicity Stab(w_{n+1}) ∩ O_e ≤ Stab(w_n) ∩ O_e;
- the constant-tail ⇒ common-vertex step;
- the SCC criterion;
- the exclusion of both Klein groups (non-cyclic ⇒ vertex-centred ⇒ Stab of a fixed seed ∈ {1, D4});
- conjugacy classes: the 2-subgroups of O up to O-conjugacy are 1, C2ax, C2diag, C4, V4 (normal), V4 (non-normal), D4.
  All are covered.

Slips:
- **E2 (vertex-centred step).** "w_a = w_b with a < b ⇒ Sym(T) = Stab(w_b)" does not follow. (w_n) is defined by
  w_n = Φ(w_{n+1}), so a repeat does not make the tail periodic, and Stab(w_m) for m > b can be smaller.
  - Fix: let N₀ be the index where ∩ₙ Stab(wₙ) is reached. A block that recurs infinitely often gives a repeat
    w_a = w_b = β with N₀ ≤ a < b. Then β is Φ-periodic and Stab(β) = Sym(T).
- **E3 (order 3).** "Such a line meets the interior of a tile" is asserted. That is false for lines in general (lattice
  edge lines lie in tile boundaries).
  - Fix: on a (1,1,1)-line the three coordinates mod 1 move together, so for some t all lie in (ρ, 1−ρ), ρ = 1/100.
    The point then lies in a retained core, which is interior to its tile.
- **E4 (converse, covering).** "cover space because |v_n − p| < 2ⁿ|e|" is not sufficient (|e| = √2 on diagonals).
  - Correct: ‖v_n‖_∞ ≤ 2ⁿ|J|/(2^p − 1) + O(1), and |J| < 2^p − 1, so the covered cube around p grows like
    2ⁿ(1 − |J|/(2^p−1)).
  - Also state that K ⊆ Sym(T): every element of K fixes A pointwise and stabilises every w_n about v_n ∈ A.
- **E5 (improper symmetries).** Not addressed explicitly. One line suffices: homochirality makes every symmetry's linear
  part a quotient of two proper frames.
- **G5.** C2 inherits the (R) condition of C1 (it uses Theorem 1 and uniqueness of the hierarchy). It also uses Tsiokos's
  geometric no-self-isometry argument (carrier recovery from planar areas). Only the finite part of that is re-checked.

## 4. C3: Mystic classes at the Hat point

### 4.1 Computations

- **Exact generator.** Exact Q(√3) re-implementation of spectre.py (`spectre_exact.py`). It agrees with spectre.py's
  floats to ≤ 2.3·10⁻¹³ through level 5 (tiles 9, 71, 559, 4401, 34649), so floating point is not an issue.
- **Kite lattice.** Tile(1,√3) is a union of 8 lattice kites, and Tile(√3,1) (Γ2) of 10, for a kite lattice K with hex
  centres c₀ + Λ, c₀ = (0, 3√3). K is unique mod Λ (exact edge test).
- **Packings.** The patches are genuine packings of hats and turtles. Mapping kites by the placements, no kite is covered
  twice through level 4 (36,200 kites).
- **Placements.** Every placement (Γ2 relative to its own base frame) is in p6m(K) through level 5.
  - Determinant = (−1)ⁿ for every placement of level n. This holds for all n by construction: each level composes with
    `MIRROR` = diag(−1, 1), and the base maps have det +1.
- **Classes mod Λ** at levels 1–5: 1, 5, 6, 6, 6. Relative rotations: {0}; {0, 120, 180, 240, 300}; then all six.
  Classes ↔ rotations is bijective at every level.
- Using Γ1 or Γ2 as "the Mystic" gives identical classes (both are placed by m_i and m_i∘g₀).

### 4.2 Registration for all n (closes the authors' "checked through n = 5")

Every level-m placement map trsₖ is in p6m(K) iff a condition on the classes mod Λ of the four level-(m−1) quad points
holds, and the next classes depend only on the current ones. So the class state evolves by a deterministic finite map.

- Computed (`c3_alln.py`): the state (classes 0, 1/3·(1,1), 0, 2/3·(1,1) in Λ-coordinates) is a fixed point at level 0.
  All 8 maps are in p6m, and `MIRROR` ∈ p6m(K).
- Base maps: Γ1 = id, and Γ2's frame trans(P₈)∘rot30 registers Tile(√3,1) (it is not itself in p6m).
- Hence **every tile of every level is K-registered**.
- With one determinant per level, the classes are in bijection with relative rotations in 60°ℤ, so there are ≤ 6.
- Δ's child #1 is Δ, so level n+1 contains a (det −1) copy of level n. Conjugating by a reflection maps the six
  rotations to themselves, so all six persist from n = 3.

C3 is therefore proved for every n ≥ 3, given only the exact finite computation.

### 4.3 Errors and gaps in the authors' C3 argument

- **G6 (p6m lemma).** "g carries a lattice kite onto a lattice kite" does not follow from M₀ and M being unions of kites:
  g(kites of M₀) is a kite decomposition of M, not obviously K's.
  - A correct argument for these polygons: maximal boundary segments lie on short-edge lines (lengths in ℤ) or long-edge
    lines (lengths in √3ℤ), so g preserves the two direction classes, giving a D6 linear part.
  - The hat has a turning vertex where a long edge meets a short edge, which is an M-type site. The incident edge
    directions there pin g to the p6m coset.
  - Either repair it like this, or drop the lemma and use §4.2.
- **W1 (weak checks).**
  - `spectre_mystic_fibres.py` checks registration only as "every vertex is a lattice site". Area is checked on the first
    200 tiles only.
  - The registration and p6m assertions run in `selftest()`, with depths 1–4 only. Level 5 is only censused, in
    `report()`. So `SPECTRE.md` / `CONNECTIONS.md` "asserted through n = 5" is overstated.
  - Its p6m test accepts any det −1 linear part without checking the mirror angle. In practice this branch is dead,
    because relative maps within a level are proper.
  - My exact edge/p6m checks supersede all of these.
- **W2 (terminology).** In the Spectre paper "Mystic" is the two-tile metatile Γ1∪Γ2. The script calls Γ2 alone "the
  Mystic". Harmless for the count, but say it.

### 4.4 Significance

Given registration, "≤ 6 classes" is the statement |p6m/Λ restricted to one determinant| = 6. "All six from level 3" is
the statement that the turtle occurs in all six rotations. These are routine. Do not present C3 as a substantive theorem.
Also note that the Hat-point patch mixes Tile(1,√3) hats with Tile(√3,1) turtles. It is not a hat tiling.

## 5. Anticipated objections

| objection | valid? | fix |
|---|---|---|
| Depends on unrefereed Tsiokos results and an LLM-built Lean proof | yes, for (R) | Make the conditionality explicit everywhere. Prove the unconditional atlas-tiling version. (H) is now independent (§2.3). |
| One data file | no longer | The atlas is re-derived from σ alone (§2.1). Cite both routes. |
| Lemma A is a black box | yes | O-orbit reduction plus a published certificate, or a structural proof (G2). |
| Convention mismatch (pose, cell lower corner, window offsets) | checked: none | State the conventions. L ⊆ Adm and P\* = geometric atlas are the sanity checks. |
| Overclaim in the corollary | yes (E1) | reword |
| Novelty vs Flicker's slide argument | partly | cite it (G3) |
| Symmetry proof slips | yes, minor (E2–E5) | patch as above |
| "Patches have exactly that group" | overstated | Finite checks give only consistency; say so. |
| C3 float/registration | settled | Exact arithmetic plus the all-n induction (§4). |
| C3 importance | valid | present as a remark |

## 6. Prioritised fixes for the paper and docs

1. State C1/C2 as conditional on Tsiokos Thm 5.6 (R) in the abstract, theorem statements and READMEs. Add the
   unconditional theorem for registered A44-tilings (= Kamenetsky F\*-tilings).
2. Reword the corollary (E1): minimality and unique ergodicity of Ω_σ under R³, "hull of a regular model set", no
   per-tiling limit-periodicity.
3. Fix E2 (choice of repeat beyond stabilisation), E3, E4, E5 in the symmetry proof.
4. Replace "(H) from [Ts]" by the reproduced finite certificate (33 / 1 parent / 1–18–14, Σ(P\*) = 2P\*) with Kamenetsky's argument.
5. Cite Flicker's slide-plane argument and "30 of 44". Add the corollary that the 14 non-hierarchical atlas contacts occur
   in no tiling by Q.
6. Replace "from independent data" by "independent code; the substitution is Tsiokos's (unique up to conjugacy by Kamenetsky Thm B)".
7. Make Lemma A checkable: O-orbit list plus a stand-alone verifier, and report the negative controls.
8. C3: drop or repair the p6m lemma (G6). Cite the finite-state induction for all n. Downgrade C3 to a remark. Fix the
   "Mystic" terminology and the "checked through n = 5" statement (the assertions run only for n ≤ 4). Replace "a proof for all n is open"
   (SPECTRE.md §6) by the induction of §4.2.
9. README wording "cyclic groups have an axis through vertices of every level" is wrong as written: the axis meets every
   L_n, but no point of it is a vertex of every L_n. Fix it.

## 7. Discrepancies with the authors' code

None in any number. Method differences:
- I derive the atlas from the geometry and from σ, not from the certificate.
- I test p6m membership exactly.
- My patch test enumerates all isometries.
- The authors' `chair44_hull_prune.py` iterates X_{k+1} = Adm ∩ Win(σ(X_k)) and stops at round 1. That is the same as Lemma A.

## 8. Resolution (2026-10-25, applied on this branch)

| item | fix |
|---|---|
| E1 | `paper/chair44_hull.tex` Corollary 6 is stated for Ω_σ. Remark 7 says the space of all tilings is not minimal, and that per-tiling limit-periodicity is not claimed. |
| E2 | Vertex-centred step: β is a block that recurs after the stabilisers are constant (from N₀). |
| E3 | Order 3: a point of the axis with all coordinates in (1/100, 99/100) mod 1 is interior to a unit cell, hence to its chair (and, for the solid, farther from cell faces than the 12·10⁻⁴ pyramid height). |
| E4 | Converse: the exact identity x_n = y₀ − 2ⁿ y_{n mod p}, with y_i = J_i/(1−2^p) for the rotated label sums J_i, gives \|x_n\| ≤ μ2ⁿ + 1 with μ = maxᵢ\|J_i\|/(2^p−1) < 1. It is checked on 2,000 random cycles over 40 levels. K ≤ Sym(T) is stated. |
| E5 | Lemma 8: symmetries are proper (quotients of proper frames), with integer translation parts and finite. |
| G1 | The theorems are stated for atlas tilings (unconditional) and for the solid conditional on Tsiokos's registration theorem, in the abstract, the introduction ("What is assumed") and each statement. |
| G2 | `independent/c44lemma_cert.py` + `lemmaA_orbits.json`: 15,214 O-orbit representatives. The verifier (157 s) recomputes everything from the geometry, checks Adm = ∪ orbits and checks the windows by type. Face windows reduce to 612 pairs per axis and edge windows to 4,098 squares per axis. The negative controls are reported in the paper. |
| G3 | Flicker's slide argument and his "30 of 44" are cited. Corollary 5 states that the 14 non-hierarchical contacts occur in no atlas tiling. |
| G4 | "Independent code; same eight children (which define the tile)". Kamenetsky's Thm B is cited with its cap of 300. |
| G5 | (H) is taken from Kamenetsky Prop. 7.1, with its finite conditions reproduced, not from Tsiokos. Only (R) remains as an outside input for the solid. |
| G6, W1, W2 | The C3 statements now rest on the finite-state induction (§4.2). The fixes to `spectre_mystic_fibres.py` / `SPECTRE.md` go to brentharts/spectre as a PR, together with the corresponding corrections to the Spectre paper's §7.4 and to `verify/hull/README.md` of brentharts/six-birds-tiles. |
| "patches have exactly that group" | Remark 11 now says these are consistency checks only. |
| README "axis through vertices" | Lemma 9 says no single point of a cyclic axis need be a vertex of every L_n. The six-birds-tiles README is fixed in its PR. |
| (R) itself | Audited in `REGISTRATION_AUDIT.md`: every written step of Tsiokos §§4–5 checked, no gap found. `independent/registration_audit.py` recomputes the mesh facts (manifold, 192 pyramids, all 6,408 dihedrals), the census (6,862 → 1,545 → 5,317, 83 integral) and checks all 5,273 of the authors' witnesses (299,975 collisions). The statements about the solid now cite (R) as audited, not as an open condition. |
