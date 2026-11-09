# CONNECTIONS.md — what links to what, and how strong each link is

The Spectre, Chair44, the Hopf fibration, Mixmaster cosmology, the complex six-sphere and
a quantum sensor have all been placed side by side in this project. Most of the
resulting links are weaker than they look. This page lists each one with a status, so a
reader can tell a theorem from a hunch.

| label | meaning |
|---|---|
| **Exact** | computed with integers/fractions or proved; a script with a self-test reproduces it |
| **Counted** | a finite check (a count up to some n, or a floating-point run with asserted checks); not a proof for all n |
| **Quoted** | a theorem from someone else's paper, not re-derived here |
| **Conditional** | proved, but resting on a stated hypothesis |
| **Elementary** | true, but nearly a restatement of definitions; does not support an inference |
| **Numerology** | a coincidence of numbers we noticed and do **not** build on |

Companion pages: [`SPECTRE.md`](SPECTRE.md), [`CHAIR44.md`](CHAIR44.md), [`E8.md`](E8.md), [`LINES.md`](LINES.md), [`CLOCK.md`](CLOCK.md), [`WILSON.md`](WILSON.md),
[`HOPF.md`](HOPF.md), [`GLOSSARY.md`](GLOSSARY.md).

---

## 1. Inside this repository

| # | link | status | where |
|---|---|---|---|
| 1 | Spectre inflation λ² = 4 + √15 is a closed BKL (Mixmaster) orbit, period 7 epochs in eras of 6 and 1, multiplier λ⁴ | **Exact** — but true for *every* quadratic unit, so it says nothing special about the Spectre | `spectre_bkl.py` |
| 2 | Spectre perimeter growth ν = φ³ = [4;4,4,…] is a one-era BKL orbit | **Exact**, same caveat | `spectre_bkl.py` |
| 3 | Kasner denominator 1+u+u² is an Eisenstein norm, so lives on the same ℤ[ζ₃] lattice as the cusp torus; equals 7λ² at u* | **Exact** | `spectre_bkl.py` |
| 4 | The "44" of Chair44 is the era length of the perimeter BKL orbit | **Numerology** (flagged in the module; the sentence should read as coincidence) | `spectre_bkl.py` |
| 5 | 60° rotation of the Spectre lift acts as diag(ω,ω) on two complex planes; ℤ/6 ⊂ the Hopf circle; 30° swaps the planes | **Exact**, **Elementary** | `lift4d.py`, [`HOPF.md`](HOPF.md) §5 |
| 6 | The six Mystic classes form one ℤ/6 = ℤ/3 × ℤ/2 orbit on the torsion points of the hexagonal torus | **Exact** (group; classes = rotations that occur, ≤ 6; six for every n ≥ 3, with registration at every level proved by an exact finite-state induction in `mystic_exact/`) | `spectre_torsion.py`, `spectre_mystic_fibres.py` |
| 7 | The ℤ/6 of direction mod 180° (boundary-edge histogram) is the same group as in 6 | **Exact** as a group; not shown to be the same *action* | `chair44_spectre_link.py` |
| 8 | Pancake cosmology: fibre clock gives N = 6.0900… levels per turn, "≈ 6 = |ℤ/6|" | **Numerology** (1.5 % off; a₀ is free; N is the small-t value and moves along the numerical solution, 7.308 for a₀ = 1.2, 3.0450… on SO(3)) | `spectre_pancake.py`, [`HOPF.md`](HOPF.md) §6 |
| 9 | Spectre pure-point diffraction via the internal contracting plane | **Exact** for the integer lift and bounded contracting plane; diffraction plots are computation, not Facts | `lift4d.py`, `spectre_diffraction.py` |
| 10 | Chair44 pure-point, limit-periodic, internal space ℤ₂³ | **Quoted** (Lee–Moody/Schlottmann) + **Exact** finite part (primitivity, coincidence at M = 3) | `chair44_twoadic.py` |
| 11 | Chair44 hull theorem: every atlas tiling is in the hull; Sym ∈ {1, C₂(diag), C₄, D₄}; only 30 of 44 contacts occur | **Exact** (computer-assisted, independently repeated) for atlas tilings; for the solid, via Tsiokos's registration theorem (R), **Quoted** and independently audited (`chair44_independent/`) | Romanov; `chair44_hull*.py` |
| 12 | Box averages are uniform (1/168), line averages are not: 6 chair stacks give ballistic sums along the body diagonal | **Exact** | `chair44_lines.py` |
| 13 | No lattice direction is uniquely ergodic | **Conditional** (hull picture; irrational directions not covered) | `chair44_lines.py` |
| 14 | Spectre and Chair44 as the "same kind of object" (both have an inflation) | **Elementary** — the difference is the point: Pisot unit + real internal space versus integer inflation + 2-adic internal space | `chair44_vs_spectre.py` |
| 15 | Spectre 4D lift lattice ℤ[ζ₁₂] = red A₂ ⊕ green A₂ embeds isometrically in E₈ (via the Hurwitz-pair model and its Z[ω] structure); every Spectre edge becomes a root, red in the pole D₄a, green in D₄b; 8+6 red/green edges per tile for 63 of 71 tiles, 6+8 for the other 8 | **Exact** on the level-2 patch (71 tiles, 518 vertices); the embedding itself is a lattice map, so the edge statement holds for every patch | `spectre_e8.py`, [`E8.md`](E8.md) |
| 16 | Spectre inflation (x⁴−8x²+1, λ² = 4+√15) extends to an automorphism of E₈ | **Falsified**: eigenvalues of modulus ≠ 1, E₈ automorphisms have finite order. ℚ(λ) = ℚ(√6, √10) contains √15 but not √3, √5 or φ | `spectre_e8.py` |
| 17 | √15 = √3·√5 (hexagonal × icosahedral field) points to a construction joining E₈'s Eisenstein and icosian structures | **Numerology** until a map is exhibited | [`E8.md`](E8.md) §4 |
| 18 | Notch vector along an axis line of a depth-n Chair44 patch: the sum along the line is bounded (\|sum\| ≤ 6, n = 3…7, **Counted**); the sum across it is exactly (±M, ±M) on a fraction **13/24 + 1/(3·4ⁿ⁻¹)** of the lines for every n ≥ 3 and every one of the 168 root labels, so the limit is 13/24 | **Exact** for the fraction (18-state subset automaton, (Q−4)(Q−1)Q³ = 0, equal to brute force for n ≤ 7); **Counted** for the bounded longitudinal sum and the equal sign states; shuffled labels give none | `chair44_polarisation.py`, [`LINES.md`](LINES.md) |
| 19 | That drift singles out a direction of travel (an arrow) | **Falsified for this observable**: the sum is the same traversed backwards, so it labels the line, not the orientation | `chair44_polarisation.py`, [`LINES.md`](LINES.md) |
| 20 | The same line sums for the Spectre (tile-orientation vector, level-5 patch, 4 directions): no ballistic component; mean \|S_k\| grows as k^e, e = 0.26…0.62, up to 1.6 times a shuffled-orientation control | **Counted** (sampled lines; negative result: the Chair44 effect is not a general monotile property, it goes with the integer inflation) | `spectre_line_sums.py`, [`LINES.md`](LINES.md) |
| 21 | Background "interaction density": the density of valence-4 vertices is exactly ρ₀ = (5−√15)/5 per spectre (∈ ℤ + ℤ(5+√15)/10); the weighted count in a complete level-j supertile differs from ρ₀·(its spectres) by (label constant)·(4−√15)ʲ — ratio 0.127 per level, 0.197 → 0.025 → 0.003 vertices for the Delta at j = 1, 2, 3 | **Exact** for ρ₀ (recorded; reproduced from the 38 collared types at K = 6); **Counted** for the decay (K = 6 patch, j ≤ 3, ratio equal to 4−√15 to 3 digits). Contrast: the Chair44 notch sum of a depth-n supertile is 7·2ⁿ(1,1,1) (**Exact**, `chair44_lines.py`). Says nothing about gravity: the tiling has no potential of its own | `spectre_density_clock.py`, [`CLOCK.md`](CLOCK.md) |
| 22 | The Spectre outline as a closed 14-step walk in 12 directions: same cyclic turning word for all 80 tiles of levels 1–2, steps sum to 0, no proper sub-run closes, the mirror tile has a different word (chiral polygon), all 559 placements of the level-3 patch are proper motions; red/green word achiral | **Exact** (integer lift; finite patches). That this gives an arrow of time is a **Hypothesis**: the closure is symmetric under reversal, only the order (chirality) carries a direction | `spectre_outline_arrow.py`, [`CLOCK.md`](CLOCK.md) |
| 23 | A U(1) Wilson action with the Spectre tiles as the only plaquettes tells corner types apart | **Falsified (Exact)**: the loop self-energy of the boundary of a union R of tiles is exactly the number of tiles in R, an area law with no perimeter and no corner term (tiles are independent, so the flux covariance is the identity); tested on 20 random unions of the level-2 patch | `spectre_wilson.py`, [`WILSON.md`](WILSON.md) §1 |
| 24 | A U(1) Wilson action on the lift lattice A₂ ⊕ A₂ (triangles in each plane weight 2, red-green parallelograms weight 1/3, isotropic Maxwell limit, relative error 1.3e-6) gives a perimeter + corner form with a constant per corner type | **Numeric**: α = 0.300, b(turn 60°) = −0.070, b(turn 120°) = −0.315, b(90°) = −0.139, max residual 0.068 on 12 loops of side 3 and 4; the sharp (turn 120°) corner is separated, the blunt (turn 60°) and the 90° corners are not (difference 0.069 against residual 0.068). Unconstrained self-energy only | `spectre_wilson.py`, [`WILSON.md`](WILSON.md) §2 |
| 25 | The Spectre outline lifted to A₂ ⊕ A₂ has the energy that perimeter + corner predicts | **Numeric, Negative**: E = 3.1826 (all 71 outlines, one value); predicted 2.884; the gap is 0.30, 0.18, 0.19 at scale 1, 2, 3, so it does not converge to 0. The mirror outline has exactly the same energy: the action is parity symmetric and cannot see the chirality of the tile | `spectre_wilson.py`, [`WILSON.md`](WILSON.md) §3 |

## 2. To the Quantum Gravity Lab fork

[Quantum_Gravity_Lab](https://github.com/brentharts/Quantum_Gravity_Lab) has a
Jaynes–Cummings engine (`LindbladJCEngine`). We drove it with a coupling or
frequency modulated by the Chair44 pyramid height along a line (`qgl_tiling.py`,
`qgl_tiling_sensor.py`, `tests/test_tiling.py`).

| result | status |
|---|---|
| The engine's phase equals the dispersive formula −(dt/Δ)Σ(gᵢ² − g₀²) for small modulation | **Counted** (tested to 0.01) |
| Along a coordinate axis, frequency-modulated, pyramid observable: tiling keeps coherence (R > 0.85 at length 64, variance/L < 0.05); a shuffled decoration loses it (R < 0.35, variance/L ≈ 1) | **Counted** (250 lines, depth 7). Bounded Birkhoff sums along axes |
| Along the body diagonal the tiling is **worse** than random (variance/L > 5) | **Counted**, explained by the chair stacks of row 12 |
| Original prediction "every line dephases more slowly" | **Falsified** — recorded as such |

The sensor is a numerical consequence of the box-versus-line theory, not an independent
physical claim.

## 3. To the CICY fork and the six-sphere

[CICY](https://github.com/brentharts/CICY) contains a pure-Python exact toolkit, an
Alpöge six-sphere module, and a Chair44 nested-automaton module.

| link | status |
|---|---|
| The set of addresses *undetermined* after M digits = union of four body-diagonal chair tubes (7·2ᴹ − 6 cells each, overlap 24, so c(M) = 28·2ᴹ − 48), M = 1…9 | **Exact** for M ≤ 9 (tubes ⊂ set for all M via the stacks; converse is Counted) |
| Transfer-matrix charpoly of the 41 non-singleton subsets x⁵(x−2)⁴(x−1)³², spectral radius 2 | **Exact** |
| The Spectre's Mystic classes sit on the cusp fibre of Alpöge's six-sphere (a hexagonal torus; Laves [3.4.6.4]) | **Exact** (lattice) + **Counted** (six classes at n = 3,4,5) |
| Matrices of orders 3 and 4 in the six-sphere construction vs the Spectre's ℤ/6 | **Numerology** unless a map is exhibited |
| "6 spheres, 24 = 24": 24 chair rotations vs 24-element groups in the six-sphere construction | **Numerology** |
| Hat, Spectre, Turtle as members of one family Tile(a,b) | **Exact** for the area form A(a,b); any identification with six-sphere data is not claimed |

(Hopf's *problem*, a complex structure on S⁶, is unrelated to the Hopf fibration; they
share a name. See [`HOPF.md`](HOPF.md) §8.)

### Pancake side, from the Quantum Gravity Lab and HopfFibration forks

What the pancake half of row 8 rests on. None of these rows links a tiling to the pancake.

| result | status |
|---|---|
| The fibre angle of the standard S³ has period 4π, the fibre length 2πR₁, the base S² radius R₂/2 | **Exact** (Quantum_Gravity_Lab `tests/test_s3_identifications.py`, from the embedding S³ ⊂ ℂ²). Whether the pancake paper's slices are S³ or SO(3) is **not checked** |
| The biaxial slice is a Berger sphere: Ricci (2s², 4 − 2s², 4 − 2s²)/R₂², scalar 8/b₀² at the pancake | **Exact**, rederived in two repositories that agree (HopfFibration `hopf_berger.py`, Quantum_Gravity_Lab `qgl_hopf.py`) |
| Local levels per turn 2πR₁/(t ln λ) along the numerical solution: 6.09 at small t, 8.48 near t = 3.1, 3.58 at t = 6 (a₀ = 1, b₀ = 3, f₀ = 2, m = 0.5) | **Counted** (float run of a leading-order model; `spectre_pancake.py` prints it when Quantum_Gravity_Lab is importable) |
| Laplace spectrum of the squashed S³ is a symmetric top; base-S² levels survive the collapse | **Exact** (symbolic, openai_math_yang_mills `SU2_BERGER_TOP.md`) |

## 4. To cosmology and quantum information

| link | status |
|---|---|
| Bianchi IX (Mixmaster) space is S³ = SU(2); two equal scale factors give a Berger sphere = Hopf fibre stretched | **Standard** background (no claim about the tilings) |
| Hopf base ℂP¹ is the Bloch sphere of a qubit | **Standard**; reviewed in Rúa Muñoz & Pineda Montoya, Symmetry **18**(6), 1024 (2026) |
| Spectre "is" a quantum system / Hopf fibre / cosmology | **Not claimed** |

## 5. How to add a link to this page

State: (a) the two objects; (b) the map between them, as a formula or a function name;
(c) the status from the table; (d) what test would break the link. A link with no
(b) is Numerology by definition.

## 6. Follow-ups already identified

Flicker authorship; the Romanov arXiv identifier and bibitem in the joint paper; the
AI-use statement; the "44" sentence of row 4; a stale supplement PDF and ledger figure; a
ring-identity off-by-3 check in CICY; an optional QGL results section; whether the pancake
paper's slices are S³ or SO(3) (it changes N in row 8 by a factor 2).
