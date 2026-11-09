# CLOCK.md — an exact background density, and a chiral closed walk

Labels as in [`CONNECTIONS.md`](CONNECTIONS.md). Scripts: `spectre_density_clock.py` (about 15 s, in `make check`), `spectre_outline_arrow.py` (about 1 s, in `make facts`).
Written as foundations for the directional-timeline sections of `quantgrav.tex` (spectre-monotile-py); it makes **no** claim about gravity.

## 1. A density that the tiling supplies

`quantgrav.tex` sets the clock rate to 1/ρ with ρ = 1 + Σ M_j f(|x−x_j|). The baseline "1" is a normalisation. The Spectre supplies an exact baseline:

* **Exact (recorded, reproduced at K = 6):** the density of valence-4 vertices of the tile graph, the volume-carrying nodes of the spin-network reading (`research/spin_densities.py`),
  is **ρ₀ = (5−√15)/5 = 0.2254 per spectre**, an element of ℤ + ℤ(5+√15)/10 (coefficients n = −2, m = 2). The valence-3 density is 2√15/5; there are 6 vertices and 7 links per spectre.
* **Counted (K = 6 patch, 242047 tiles, complete supertiles of level 1…4):** inside a level the weighted valence-4 count minus ρ₀·(number of spectres) is a constant
  depending only on the supertile label, and it shrinks by exactly the factor **4−√15 = λ⁻² = 0.127** per level (Delta: +0.197, +0.025, +0.003 vertices at j = 1, 2, 3; the other eight labels −0.029, −0.004, −0.0005).
  There is no O(1) or perimeter term. This is the Pisot rigidity of the inflation.
* **Contrast (Exact, `chair44_lines.py`):** for the Chair44, whose inflation is the integer 2, the notch sum of a depth-n supertile is 7·2ⁿ(1,1,1): the deviation from the box mean grows.

So the background is an exact algebraic number with an extremely quiet, exactly decaying fluctuation, and the tiling makes **no gravitational potential of its own**: a clock 1/ρ read off the tiling alone is uniform to better than 10⁻⁵ over ~500 spectres (deviation under 0.001 vertex out of ~120).
Gravity still has to be put in as the M_j. Not shown: any relation between ρ₀ and a mass, or between this ρ and QGD's required ρ = (1−r_s/r)^(−1/2).

## 2. A closed, minimal, chiral walk

The 14 edges of a Spectre are unit steps in 12 directions 30° apart (`lift4d.py`). **Exact** on all 80 tiles of levels 1–2 (integer lift):

* every outline has the same cyclic turning word (units of 30°) (−2, 3, −2, −3, −2, 3, −2, 0, −2, −3, 2, −3, 2, −3), total turn −360°;
* the 14 steps sum to 0 and no proper contiguous run does: a minimal closed walk (the "consistency contract" of the toy models is flux closure);
* the mirror tile, traversed in the same sense, has the word reversed, which is **not** a cyclic rotation of the word: the straight 14-gon is chiral. A regular hexagon is achiral (negative control) and a lopsided hexagon is chiral (positive control);
* all 559 placements of the level-3 patch are proper motions (det +1): the mirror tile does not occur;
* the red/green colouring is achiral in itself, so the chirality is in the turning word.

**What it does not give.** Closure is symmetric under reversal (the same sum), as in [`LINES.md`](LINES.md) §3; reversing any polygon flips its signed area. The direction is carried only by the order, i.e. by chirality.
That a timeline whose steps are this outline has a built-in direction is a **Hypothesis**; no dynamics is tested. Related, already in the repository: the handedness theorem (Im Z_k < 0 for all k ≤ 900, `research/handedness_closed.py`),
the parity clock (`research/parity_clock.py`), and for the Chair44 the invariance of the hull under PC but not P alone (`chair44_pc_proof.py`). Brent's twist-sign results (`research/ribbon_linking.py`, R6) found
**no** arrow in the linking number: the twist sign is a gauge choice.

## 3. What this suggests for `quantgrav.tex` (not applied)

* The Conclusion's "none of the static structures exhibited an intrinsic arrow of time" can be made precise: closed minimal outlines and a proved handedness exist, but they are chirality, not time order.
* §10 should say that the baseline density is a normalisation and, if a tiling-derived value is wanted, cite ρ₀ above; the two ρ formulas of §9.1 and §10.3 disagree and neither reproduces QGD's clock.
* §8's map from ±X, ±Y, ±Z to E8 roots is undefined. The red A₂ gives six signed hexagonal directions with a native closure rule (triples at 120° sum to 0), which the cubic six-direction toy lacks. That variant is the next step (not done).
