# SPECTRE.md — the Spectre monotile and what its substitution determines

For a reader who has seen the pictures of the Hat and the Spectre and wants to know what
this repository actually computes, and how sure each statement is. Terms are explained on
first use; [`GLOSSARY.md`](GLOSSARY.md) lists them. Status labels as in
[`CONNECTIONS.md`](CONNECTIONS.md): **Exact**, **Counted**, **Quoted**, **Conditional**,
**Numerology**.

---

## 1. The tile

In 2023 David Smith, Joseph Samuel Myers, Craig S. Kaplan and Chaim Goodman-Strauss
found a single polygon — the **Hat** — that tiles the plane but only without any
translation period, and then a variant, the **Spectre**, that does so *without even
needing reflected copies* (it is **chiral**: every tile in the tiling has the same
handedness). Papers: Hat, arXiv:2303.10798; Spectre, arXiv:2305.17743.

The Hat, the Spectre and a third tile, the Turtle, are members of one family
**Tile(a,b)** obtained by changing the lengths of the two kinds of edge; its area is the
exact quadratic form A(a,b) = 2√3·a² + 3ab + √3·b² ([`tile_family.py`](tile_family.py)).

## 2. The substitution and its arithmetic (Exact)

Tiles group into larger and larger **supertiles**. The Spectre has nine tile species
(Γ, Δ, Θ, Λ, Ξ, Π, Σ, Φ, Ψ) and one substitution step is a 9×9 integer matrix M
([`spectre_operator.py`](spectre_operator.py), [`spectrefacts.py`](spectrefacts.py)).

* **Characteristic polynomial** χ(x) = x⁵ (x−1)(x+1)(x² − 8x + 1). The growth factor per
  step is the root λ² = 4 + √15 ≈ 7.873, the **fundamental unit** of ℤ[√15] (because
  4² − 15·1² = 1). It is a **Pisot–Vijayaraghavan unit**: its algebraic conjugate
  4 − √15 = 1/λ² has absolute value below 1. This is the property that makes the tiling
  *quasiperiodic with an internal space* (cut-and-project), unlike Chair44.
* The **linear** inflation factor is λ = ½(√6 + √10); (λ − λ⁻¹)² = 6 and
  (λ + λ⁻¹)² = 10, proved in ℤ[√15] without taking a square root.
* **Not diagonalisable.** rank M = 5 but rank Mⁿ = 4 for n ≥ 2: the zero eigenvalue has
  algebraic multiplicity 5 and geometric multiplicity 4 — one "transient" direction
  survives one step and dies at the second.
* **Two integer charges.** Q₊ and Q₋ are left eigenvectors for the eigenvalues
  +1 and −1: Q₊ is conserved, Q₋ alternates sign each level and tracks handedness.
* **Census.** Number of tiles at level n: 1, 9, 71, 559, 4401 …; counting the Mystic
  tile as a separate copy, 1, 8, 63, 496, 3905. These are linked by the identity
  T_geom(n) = T(n) + T(n−1) (a theorem, since row Γ of M is all ones).
* **All nine frequencies are irrational** (in ℤ[√15] with nonzero √15 part), so *any*
  one species is enough to refute periodicity. The Hat's commonest metatile has
  frequency exactly 1/3 — rational, so it cannot witness aperiodicity.

## 3. The Hat is a different phase, in a different field (Exact)

The Hat runs on four metatiles with χ_H(x) = (x−1)(x+1)(x² − 7x + 1) and inflation
φ⁴ ≈ 6.854, φ the golden ratio. The two polynomials share (x−1)(x+1) and differ in
one coefficient, 7 versus 8. The Spectre lives in ℚ(√15), the Hat in ℚ(√5).

A laboratory measurement agrees with this arithmetic: Moritake et al. (Nat. Commun.
**17**, 6085, 2026) fabricated the Hat quasilattice and measured a chiral pinwheel
diffraction pattern with twist θ = arccos((3φ−1)/4) ≈ 15.52°. Since 3φ − 1 = φ⁴ − 3,
this is cos θ = (κ − 3)/4 with κ the Hat inflation factor. The same expression for the
Spectre gives 1.2182 > 1: **the Spectre has no such angle** and needs its own.
*(Exact; the experiment is quoted.)*

## 4. The boundary (Exact / Counted)

The perimeter of a level-k supertile obeys
P(k) = 5P(k−1) − 3P(k−2) − P(k−3), P = 14, 46, 182, 758, 3198, 13534, …
so it grows like ν = φ³ = 2 + √5, **not like λ**. The boundary therefore has
box-counting dimension log ν / log λ ≈ **1.3993**. The recurrence is asserted equal to
counts made on the geometry. In Fibonacci form P(k) = 2F(3k+5) + 4
([`chair44_spectre_link.py`](chair44_spectre_link.py); checked k = 0…5, not proved):
one Spectre level multiplies the Fibonacci index by 3.

## 5. The four-dimensional lift (Exact)

Every edge of the Spectre tiling is a unit vector at a multiple of 30°. Writing
ω = e^{iπ/3} and z = e^{iπ/6} (so z² = ω), even multiples are p + qω and odd multiples
are z(r + sω). Walking along edges, every vertex gets **integer** coordinates
(p, q, r, s) ∈ ℤ⁴ and physical position P = (p + qω) + z(r + sω): the vertex set lives
in the ring ℤ[ζ₁₂] = ℤ⁴ (Chéritat–Ma; [`lift4d.py`](lift4d.py), [`lift4d_matrix.py`](lift4d_matrix.py)).

* The supertile recursion acts on the lifted key points of a supertile by an **integer
  matrix** ([`lift4d_matrix.py`](lift4d_matrix.py)), and ℤ⁴ splits into a plane that
  the inflation expands (the physical plane) and a **contracting plane** — the internal
  space of the cut-and-project description. The window is bounded and fractal-like.
* The Galois conjugate z ↦ z⁵ is *not* the internal space: it grows by the inflation
  factor too (tested).
* The 60° rotation acts diagonally on the two complex planes; see
  [`HOPF.md`](HOPF.md) §5 for what that does and does not mean.

## 6. Mystic fibres and the torsion dictionary (Exact group theory, Counted data)

At the Hat proportions every tile is a union of kites of the [3.4.6.4] Laves lattice. The
cusp fibre of Alpöge's complex six-sphere is a hexagonal torus T = ℂ/ℤ[ζ₃] (the elliptic
curve with j = 0). Rolling the lattice up on T, the Mystic tiles of the full levels fall
into **six translate classes** (levels 3, 4, 5: 63, 496 and 3905 Mystics; at level 5 the
class sizes are 629 to 674).

The 60° rotation is the integer matrix R = [[0,−1],[1,1]] (R⁶ = 1, R³ = −1). The six
site classes of the lattice modulo Λ are torsion points: the centre is 0, the three edge
midpoints are the three non-zero points of E[2], and the two hexagon vertices are the
non-zero points of E[√−3]. R cycles the 2-torsion points and swaps the vertices; the six
Mystic classes are one orbit of ℤ/6 = ℤ/3 × ℤ/2. **The group structure is exact; that all
six classes occur is data, and "six classes for all n ≥ 3" is a conjecture.**
The Spectre itself is not on this lattice (its area is 3 + √3 kites).
([`spectre_mystic_fibres.py`](spectre_mystic_fibres.py), [`spectre_torsion.py`](spectre_torsion.py).)

## 7. Mixmaster / BKL (Exact, with a stated non-claim)

[`spectre_bkl.py`](spectre_bkl.py) shows that the Spectre's growth rates are closed orbits
of the BKL (Mixmaster) map, an exact fact about continued fractions: the BKL map is the
**Gauss map** in disguise (the number of subtractions before an inversion is the next
partial quotient), so a closed orbit is a quadratic irrational with periodic continued
fraction.

* λ² = 4 + √15 is an eigenvalue of [[7,6],[1,1]] = [6,1] (partial quotients 6 and 1;
  trace 8, det 1). The BKL orbit of u* = 3 + √15 = [6; 1, 6, 1, …] has period **seven**
  epochs, in eras of 6 and 1, with multiplier (4+√15)² = λ⁴ exactly.
* The perimeter growth ν = φ³ = [4; 4, 4, …]: one era of 4 epochs, multiplier −ν²
  (sign: the orbit reverses orientation).
* The Kasner denominator 1 + u + u² = |u − ζ₃|² is an Eisenstein norm, hence lives on
  the same hexagonal lattice ℤ[ζ₃] as the cusp torus; at u* it equals 7λ² = 28 + 7√15.

**What it does not prove.** The theorem holds for *every* quadratic unit — every periodic
continued fraction is a closed BKL orbit with multiplier the square of its unit. So
"λ² is a BKL orbit" is not a property of the Spectre; the Spectre selects only the era
pattern (6, 1) and the number 7. No dynamical link between the tiling and the
cosmology is claimed. (Background on BKL: [`HOPF.md`](HOPF.md) §7.)

## 8. Energy and diffraction (Computation, not Facts)

The Gaussian-kernel energy per particle of the Spectre, Hat and Turtle vertex sets lies
0.064, 0.109 and 0.149 above the triangular lattice at unit density and width α = 1
(Poisson process: 0.467; [`spectre_energy.py`](spectre_energy.py)). The windowed structure
factor ([`spectre_diffraction.py`](spectre_diffraction.py)): the Hat and Turtle have their
strongest Bragg peaks on the dual of the torus lattice; the Spectre has strong peaks but
none on such a lattice. No exponents are claimed.

## 9. Where the evidence stands

The repository registers **93 Facts**, each classified as derived from the quoted matrix,
counted on the tiling, recomputed by a second implementation, quoted from the literature,
or resting on a premise ([`spectre_ledger.py`](spectre_ledger.py)). A generated Lean 4 file
with 67 theorems (27 linked to a Fact) checks the closed statements — arithmetic on the
counted values, not the counts themselves.

The one place the work asks to be trusted: that the 9×9 matrix correctly transcribes the
*published rules*. Every Fact is downstream of it. The geometric census audits it; an
audit is not a proof.

Unresolved, as stated in the Readme: the flavour frequency p = Pr(Φ²) measured at depths
3–6 is 0.683, 0.579, 0.538, 0.519 — still falling; if p → ½ then ρ_X = 1 − √15/5 ≈ 0.2254
(evidence, not proof).

## 10. How to run things

    python3 spectrefacts.py --selftest      # the 62 Spectre Facts, exact, in sympy
    python3 monotilefacts.py --selftest     # the 7 monotile Facts
    python3 lift4d.py --tests-only          # the integer lift
    python3 spectre_bkl.py                  # continued fractions / BKL orbits
    make check                              # everything

## 11. References

1. D. Smith, J. S. Myers, C. S. Kaplan, C. Goodman-Strauss, *An aperiodic monotile*, arXiv:2303.10798 (2023); *A chiral aperiodic monotile*, arXiv:2305.17743 (2023). *(cited from memory; verify before quoting)*
2. M. Baake, U. Grimm, *Aperiodic Order, Vol. 1*, Cambridge University Press (2013): cut-and-project, Pisot substitutions, diffraction.
3. Moritake et al., Nat. Commun. **17**, 6085 (2026), <https://doi.org/10.1038/s41467-026-75023-7>: the measured chiral angle (as quoted in the Readme).
4. Chéritat–Ma four-dimensional lift: as attributed in [`lift4d.py`](lift4d.py).
5. V. A. Belinsky, I. M. Khalatnikov, E. M. Lifshitz, Adv. Phys. **19** (1970) 525 (BKL); C. W. Misner, Phys. Rev. Lett. **22** (1969) 1071 (Mixmaster). *(cited from memory)*
6. Dechant, Lasenby, Hobson, the non-singular "pancake" cosmology; ported in `pyCICY` (see [`HOPF.md`](HOPF.md) §6).
