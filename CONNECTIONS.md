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

Companion pages: [`SPECTRE.md`](SPECTRE.md), [`CHAIR44.md`](CHAIR44.md),
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
| 8 | Pancake cosmology: fibre clock gives N = 6.0900… levels per turn, "≈ 6 = |ℤ/6|" | **Numerology** (1.5 % off; a₀ is free) | `spectre_pancake.py` |
| 9 | Spectre pure-point diffraction via the internal contracting plane | **Exact** for the integer lift and bounded contracting plane; diffraction plots are computation, not Facts | `lift4d.py`, `spectre_diffraction.py` |
| 10 | Chair44 pure-point, limit-periodic, internal space ℤ₂³ | **Quoted** (Lee–Moody/Schlottmann) + **Exact** finite part (primitivity, coincidence at M = 3) | `chair44_twoadic.py` |
| 11 | Chair44 hull theorem: every atlas tiling is in the hull; Sym ∈ {1, C₂(diag), C₄, D₄}; only 30 of 44 contacts occur | **Exact** (computer-assisted, independently repeated) for atlas tilings; for the solid, via Tsiokos's registration theorem (R), **Quoted** and independently audited (`chair44_independent/`) | Romanov; `chair44_hull*.py` |
| 12 | Box averages are uniform (1/168), line averages are not: 6 chair stacks give ballistic sums along the body diagonal | **Exact** | `chair44_lines.py` |
| 13 | No lattice direction is uniquely ergodic | **Conditional** (hull picture; irrational directions not covered) | `chair44_lines.py` |
| 14 | Spectre and Chair44 as the "same kind of object" (both have an inflation) | **Elementary** — the difference is the point: Pisot unit + real internal space versus integer inflation + 2-adic internal space | `chair44_vs_spectre.py` |

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
ring-identity off-by-3 check in CICY; an optional QGL results section.
