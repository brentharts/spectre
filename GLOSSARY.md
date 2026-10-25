# GLOSSARY.md — terms used in this repository's documentation

Plain-language definitions, ordered by topic. See [`SPECTRE.md`](SPECTRE.md),
[`CHAIR44.md`](CHAIR44.md), [`HOPF.md`](HOPF.md) and [`CONNECTIONS.md`](CONNECTIONS.md) for use.

## Tilings

**Monotile / einstein tile.** One shape that tiles space, but only aperiodically. (Hat, Spectre in the plane; Chair44 in 3D.)

**Aperiodic.** Has no translation under which the tiling maps to itself.

**Substitution (inflation).** A rule that replaces each tile by a patch of smaller tiles, then rescales. Iterating builds ever larger **supertiles**. The **inflation factor** is the rescaling; for the Spectre λ² = 4 + √15 (in area, per step), for Chair44 it is the integer 2.

**Substitution matrix.** Entry (i,j) counts tiles of species i in the replacement of species j. Its largest eigenvalue is the growth rate; its eigenvector gives tile **frequencies**.

**Primitive.** Some power of the matrix has all entries positive: every species eventually contains every other.

**Census.** The count of tiles of each species at a given level.

**Hull.** The set of all tilings whose every finite patch appears in a given tiling. **Uniquely ergodic**: every patch has a frequency, and averages converge uniformly.

**Chiral.** Distinguishable from its mirror image. The Spectre tiling uses one handedness only.

## Quasicrystals and diffraction

**Pisot unit.** An algebraic integer > 1 that is a unit and whose conjugates all have absolute value < 1 (λ² = 4 + √15 is one: its conjugate is 4 − √15 ≈ 0.127). Pisot inflation gives tilings with pure-point diffraction (for these, a "model set").

**Cut-and-project / model set.** Take a lattice in a higher-dimensional space, keep the points whose "internal" coordinate lies in a window, and project to the physical space. The result is a quasicrystal; its structure is explained by the internal space.

**Internal space.** The extra directions of the higher-dimensional lattice. For the Spectre it is the contracting plane of the inflation; for Chair44 it is 2-adic (see below).

**Diffraction.** The Fourier transform of the (autocorrelation of the) point set, as seen in X-ray patterns. **Pure point**: it consists of sharp Bragg peaks only, with all the intensity in the peaks.

**Limit-periodic.** A limit of periodic patterns: periodic to any given accuracy on larger and larger regions, but not periodic overall. Chair44 is limit-periodic.

**2-adic numbers ℤ₂.** Infinite binary digit strings with "closeness" meaning agreeing on many low digits. The natural home for patterns built by doubling.

**Coincidence.** A position where the images of all starting labels under a power of the substitution agree. Modular coincidence ⇒ the fixed point is a regular model set.

## Algebra and geometry

**ℤ[√15], ℤ[ζ₃], ℤ[ζ₁₂].** The integer combinations of 1 and √15; of the sixth/third roots of unity (the hexagonal lattice, Eisenstein integers); of the twelfth roots of unity (ℤ⁴, the lattice in which Spectre vertices live).

**Fundamental unit.** The smallest unit > 1 of a number ring; every other unit is a power of it.

**Torsion point.** A point P of a group with nP = 0 for some n. On the hexagonal torus ℂ/ℤ[ζ₃], the 2-torsion E[2] has four points (0 and three edge midpoints).

**Laves lattice [3.4.6.4].** The dual of the rhombitrihexagonal tiling; the lattice of kites on which the Hat's tiles are built.

**Hopf fibration.** S³ filled by circles over S² (see [`HOPF.md`](HOPF.md)). **Clifford algebra / Clifford parallel / Clifford bundle:** three different things named after W. K. Clifford; see the table in [`HOPF.md`](HOPF.md) §4.

**Division algebra.** An algebra in which every non-zero element has an inverse: ℝ, ℂ, ℍ, 𝕆 are the normed ones.

## Cosmology

**Kasner solution.** An empty anisotropic universe expanding in two directions and contracting in one; parameter u > 1.

**BKL / Mixmaster.** Near a cosmological singularity the universe is believed to pass through an endless sequence of Kasner epochs; the transition rule on u is the BKL map (subtract 1, or invert), which is the Gauss continued-fraction map in disguise.

**Bianchi IX.** The homogeneous cosmologies whose space is S³ = SU(2). With two equal scale factors the space is a **Berger sphere** (a Hopf-fibred S³ with stretched fibre).

## Project vocabulary

**Fact.** A named, self-testing computation in `spectrefacts.py`, `monotilefacts.py` or `chair44facts.py`; the paper pulls each of its numbers from one.

**Evidence ledger / KIND.** Each Fact is tagged D (derived from the quoted matrix), C (counted on the tiling), X (re-derived by a second implementation), Q (quoted from the literature) or P (carries a premise).

**Box vs line.** Averages of a labelled tiling over a cube versus over a straight line; for Chair44 the first converge uniformly, the second do not in every direction (see [`CHAIR44.md`](CHAIR44.md) §4).

**Chair stack.** A diagonal line of identical labels that persists in every σⁿ(j).

**Undetermined set.** The addresses whose label is not fixed after reading the top M binary digits.
