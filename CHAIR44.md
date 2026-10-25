# CHAIR44.md — the Chair44 monotile: what it is, what has been proved, what has not

Written for a reader who knows what a tiling is but not what a *hull*, a *model set* or
a *2-adic internal space* is. Technical terms are explained where they first appear;
[`GLOSSARY.md`](GLOSSARY.md) collects them.

Status labels (same as [`CONNECTIONS.md`](CONNECTIONS.md)): **Exact** (computed with
integers/fractions by a script with a self-test), **Quoted** (a theorem from someone
else's work, not re-derived here), **Conditional** (a proof that rests on a stated
hypothesis), **Counted** (a finite check, not a proof for all n), **Numerology**.

---

## 1. What it is

A **monotile** is a single shape that tiles space but only in ways with no translation
period. In the plane the Hat and Spectre (see [`SPECTRE.md`](SPECTRE.md)) are such
shapes. **Chair44** is a three-dimensional one, due to Ioannis Tsiokos
([six-birds-tiles](https://github.com/ioannist/six-birds-tiles), CC BY 4.0;
arXiv:2609.19214 and 2609.23783 as cited in [`chair44.py`](chair44.py)).

* The shape is a **chair**: a 2×2×2 block of unit cubes with one corner cube removed —
  seven cubes. It has 24 exposed unit panels, each carrying 8 tiny square pyramids
  (bumps and dents). The pyramids are the "decoration" that forces aperiodicity: bare
  chairs can tile periodically; decorated ones cannot.
* **Eight rotated copies of a chair fit together into a doubled chair** — a chair
  scaled by exactly 2 (twist 0, no mirror; [`chair44_twist.py`](chair44_twist.py)). Repeat
  and you get supertiles of side 2ⁿ. This is a **substitution** with integer inflation
  factor 2 and 8 children per tile.
* "44" is the number of legal ways two chairs can touch (the **contact atlas**:
  2388 candidate contacts reduce to 44; `python3 chair44.py --verify`).

For the Spectre the inflation factor is irrational (λ² = 4 + √15), which is what makes
its tiling quasiperiodic. For Chair44 it is the integer 2, so the *tiling* is built
from a different kind of order: **limit-periodic**. It is almost periodic at every
scale 2ⁿ, and the "internal space" in which it lives is made of 2-adic numbers rather
than of real numbers.

## 2. The finite combinatorics (Exact)

Label every unit cell of a tiling by (orientation, which of the 7 cells of its chair):
24 rotations × 7 cells = **168 labels**. Inflation sends each label to a 2×2×2 block of
8 labels; call this map σ ([`chair44_twoadic.py`](chair44_twoadic.py), built from nothing
in [`symchair44.py`](symchair44.py)).

| fact | value | how it is known |
|---|---|---|
| labels | 168 | 24 × 7 |
| σ is **primitive** | its 168×168 matrix has column sum 8 and is entrywise positive at power 3 | exact, boolean iteration |
| **modular coincidence** | at M = 3 a position exists where σᴹ(i) is the same label for *every* i; 336 of the 512 addresses coincide | exact |
| not-yet-coincided words | c(M) = 1, 8, 64, 176, 400, 848, 1744, …; c(M+1) = 2c(M) + 48, i.e. **c(M) = 28·2ᴹ − 48** | exact (subset automaton: 209 reachable subsets, 41 non-singleton, transfer spectral radius exactly 2) |
| legal vertex blocks | **1410** distinct 2×2×2 windows occur in some σⁿ(j), and the set is closed under taking the 27 child windows, so it is the *whole* language | exact ([`chair44_hull.py`](chair44_hull.py)) |
| fixed seeds | the map Φ (centre window of σ(w)) has 27 periodic points, all fixed; stabilisers of order 1 (24 of them) and 8 (3, the group D4) | exact |
| admissible blocks | **360,189** | exact, by Tristan Romanov's [`chair44_hull_admissible.py`](chair44_hull_admissible.py) (about 10 minutes) |

**Why "modular coincidence" matters.** A primitive substitution with a coincidence has
a fixed point whose label classes are **regular model sets** — the strongest kind of
quasi-order: the pattern is obtained by cutting a higher-dimensional lattice with a
window in an internal space. With integer inflation 2 the internal space is the
2-adic group ℤ₂³ (here: three independent 2-adic digits strings, one per axis), so the
tiling is **pure-point diffracting and limit-periodic**. The import is from
Lee–Moody and Schlottmann; Tsiokos's repository records the argument
(`THEOREM.md`, `substitution_modular_coincidence`) and
[`chair44_twoadic.py`](chair44_twoadic.py) recomputes the finite part independently and adds the
rate c(M). *(Quoted for the model-set step; Exact for the finite part.)*

**The diffraction of the centre set is exactly known** ([`chair44_diffraction.py`](chair44_diffraction.py)).
The centres of the 8ⁿ chairs obey X_n = 2X_{n−1} ∪ (odd points minus the notch octant),
which gives closed-form Bragg amplitudes. The peak intensities sum to exactly 7
(= 1 + (21/4)(8/7)), the whole density 1/7 times 49: a **Parseval check** that the
diffraction is pure point. *(Exact, verified for n ≤ 5 and then closed-form.)*

## 3. The hull theorem (Romanov) — what it says

The **hull** of a tiling is the set of all tilings whose every finite patch occurs
somewhere in it. A tiling is **uniform** if every patch appears with a well-defined
frequency, with convergence uniform in position. The question: is *every* Chair44
tiling of space — however you build it — in the substitution hull?

Tristan Romanov's theorem, written up in the joint paper (§ "Every tiling lies in the
hull"): **yes, conditional on two quoted facts**, the registration lemma **(R)**
(Tsiokos) and the unique-parent-tiling property **(H)** (Goodman-Strauss). Given
those, every tiling is in the hull, and its symmetry group is one of
**1, C₂ (about a face-diagonal axis), C₄, D₄**, with at most 8 elements. The space of
tilings is the hull of a regular model set with internal space ℤ₂³. *(Quoted/Conditional;
the finite computations behind it are Exact, and the two hypotheses are not
re-derived here.)*

Consequence worth knowing: because the hull is uniquely ergodic on *boxes*, every
fixed-size cube drawn anywhere contains each of the 168 labels with the same limiting
frequency, 1/168. That is the next section's starting point.

## 4. Boxes versus lines (Exact + Conditional) — [`chair44_lines.py`](chair44_lines.py)

This is the part of the project that is ours rather than quoted. In one sentence:
**uniform statistics in cubes do not transfer to straight lines.**

Define two observables on a label: the **pyramid height** p (192 signed heights over the
whole atlas, summing to zero; only native cells 3 and 5 carry ±36) and the **notch
vector** d (g·(1,1,1) with g = ±1).

* **Boxes.** The mean of p and of d over a big box tends to exactly 0, because the
  168-label frequency vector is uniform (matrix primitive, exponent 3). The notch sum of
  a depth-n supertile (7·8ⁿ cells) is **7·2ⁿ·(1,1,1)**, against the ≈ 8^{n/2} you would
  get from independent random decoration. *(Exact.)*
* **Lines.** Take the **chair stacks**: six labels j with σ[j,0] = σ[j,7] = j, whose
  native cell 0 has notch ±(1,1,1). Each of the seven chair cells generates a
  constant-label line along the body diagonal in σⁿ(j) **for every n** (a tube
  induction on 14 finite cases). Along those lines the height is −36 on one and +36 on
  another and the notch is ±1, so partial sums grow **linearly**: the Birkhoff averages
  along the body diagonal do not converge to the box mean. *(Exact.)*
* **Other lines.** Along the seven corner lines the substitution restricts to
  constant-length-2 substitutions τ_b(j) = σ[j,0]σ[j,b]; the x-axis corner has a closed
  class of 4 labels with notch mean (0,1,1). Exact limits are computed. *(Exact.)*
* **All lattice directions.** The orbit closure of a lattice line in the internal
  space ℤ₂³ is one-dimensional, hence Haar-null, so no lattice direction can be
  uniquely ergodic. *(Conditional on the hull picture of §3.)* **Irrational
  directions are not covered.**

Why it matters, in a sentence: a sensor or detector that samples along a line can see
statistics that a sensor sampling a volume never sees. The experiment is run in
[Quantum_Gravity_Lab](https://github.com/brentharts/Quantum_Gravity_Lab) (see
[`CONNECTIONS.md`](CONNECTIONS.md) §2): along a coordinate axis the Jaynes–Cummings
sensor keeps coherence under the tiling decoration (R > 0.85 at length 64) and loses it
under a shuffled one (R < 0.35); along the body diagonal it does *worse* than random.

## 5. The undetermined set (Exact for M ≤ 9; the converse is Counted)

Reading the top M address digits of a position does not always determine the label.
The positions that are **still undetermined** after M digits form exactly **the union of
four body-diagonal tubes** of chair stacks, each 7·2ᴹ − 6 cells long, overlapping in 24,
giving c(M) = 4(7·2ᴹ − 6) − 24 = 28·2ᴹ − 48 — the same c(M) as §2. Checked for
M = 1…9 (the CICY fork, `chair44_nested.py`). The six rotations that fix a body diagonal
are the D₃ stabiliser of that diagonal. That the tubes lie inside the undetermined set for **all**
M follows from the stacks; that the undetermined set is *only* the tubes is checked, not
proved, for M ≤ 9.

This ties the 2-adic coincidence rate (28·4⁻ᴹ) to the ballistic diagonal of §4: **the
same four lines are where determination is slowest and where line averages fail.**

## 6. What is not established

* The hull theorem is conditional on (R) and (H); we recompute only their finite
  consequences.
* No proof for irrational directions; no claim for 2D slices or planes beyond the
  [`chair44_slice.py`](chair44_slice.py) computations.
* "Chair44 behaves like a quasicrystal in gravity / cosmology" is **not** claimed anywhere.
  The coincidence of some numbers with the Spectre's (the "44", 6s and 24s) is listed
  in [`CONNECTIONS.md`](CONNECTIONS.md) as numerology.

## 7. How to run things

    python3 chair44.py --verify             # 2388 candidate contacts -> 44, patch checks
    python3 chair44_twoadic.py              # primitivity, coincidence, c(M)
    python3 chair44_hull.py --tests         # 1410 legal blocks, 27 seeds (~5 s)
    python3 chair44_hull_admissible.py --tests   # 360,189 (~10 min)
    python3 chair44_lines.py --tests        # box means, stacks, corner lines
    python3 chair44facts.py --selftest      # all 24 Chair44 Facts

## 8. References

1. I. Tsiokos, *six-birds-tiles*, <https://github.com/ioannist/six-birds-tiles> (CC BY 4.0); arXiv:2609.19214, 2609.23783 (as cited in `chair44.py`). The shape, substitution and 44-contact atlas.
2. T. Romanov, hull and symmetry theorem, in the joint paper ([`spectrepaper.py`](spectrepaper.py)) and [`chair44_hull_admissible.py`](chair44_hull_admissible.py). The 360,189 count.
3. M. Baake, U. Grimm, *Aperiodic Order, Vol. 1: A Mathematical Invitation*, Cambridge University Press (2013). Standard text for hulls, model sets, diffraction, limit-periodicity.
4. J.-Y. Lee, R. V. Moody, *Lattice substitution systems and model sets*, Discrete Comput. Geom. **25** (2001) 173–201; M. Schlottmann, *Generalized model sets and dynamical systems* (2000). The import behind the model-set step of §2. *(cited from memory)*
5. Goodman-Strauss, unique parent tiling (H) — as quoted in Romanov's text; see the paper's bibliography.
