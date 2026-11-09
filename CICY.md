# CICY.md — what the CICY fork and this repository say about each other

Labels as in [`CONNECTIONS.md`](CONNECTIONS.md). Scripts: `spectre_mystic_xbond.py` (about 18 s, in `make check`), `chair44_notch_axis.py` (about 1 s, in `make facts`; with `CICY_DIR=<clone of brentharts/CICY>` it also runs CICY's own functions).
CICY (`github.com/brentharts/CICY`, formerly pyCICY-X) is a package of exact methods for string-theory geometry. It contains two independent re-derivations of work in this repository
(`pyCICY/theories/spectre.py` for the Spectre, `pyCICY/chair44_nested.py` for the Chair44) and several modules on neighbouring structures (the Alpöge six-sphere cusp torus, the Nariai horizon, a
Hat/Laves electron model). Nothing below changes a claim of CICY; it records what agrees, what is new, and one open item that this repository closes.

## 1. The open item: p = Pr(Φ²) = 1/2

`Readme.md` and [`SPECTRE.md`](SPECTRE.md) §9 leave one number open: the fraction p of Φ tiles whose X-charge is 2 rather than 0, measured 0.683, 0.579, 0.538, 0.519 at depths 3 to 6 ("still falling";
Aitken extrapolation 0.5107 then 0.5035; "if p → ½ then ρ_X = 1 − √15/5"). CICY's `spectre.x_density(p)` and `SpectreSubstrate.missing_for_physical()` list the same p as the one unknown of the closed-form X-density.

**Counted (M1, `spectre_mystic_xbond.py`).** An X-bond (a shared edge on which the two tiles disagree about the edge type) is exactly a bond with a Mystic on one side. No two Mystics share an edge, and a
shared edge is an X-bond iff exactly one of its tiles is a Mystic. This holds for every bond of the Delta patch at depths 3, 4 and 5 (3534, 29208 and 235776 bonds). So the X-charge of a normal tile is its number of
Mystic edge-neighbours, and the Mystic's own charge is 14. The type-swap definition predicts it; the check confirms the geometry realises it.

**Counted (M2).** The X-bonds form disjoint 14-cycles, one per Mystic (63 and 496 cycles at depths 3 and 4), and each cycle passes through exactly two interior valence-4 vertices; every interior valence-4
vertex lies on exactly one cycle. Hence X-bonds = 14 M, valence-4 vertices = 2 M, and **X-bonds = 7 × (valence-4 vertices)**, which holds in the whole patch for all nine seeds (Γ, Δ, Θ, Λ, Ξ, Π, Σ, Φ, Ψ) at depths 2, 3, 4.
That is why the X-bond count per tile, 1.57782, equals 7ρ₀ with ρ₀ = (5−√15)/5 the valence-4 density of [`CLOCK.md`](CLOCK.md) §1: the two densities were the same quantity, 2 × the Mystic frequency.

**Exact (M3), given the Counted atlas.** Every X-bond has one Mystic end and one normal end, so the total X-charge on normal tiles equals the total on Mystics. With the exact Perron frequencies in ℤ[√15]
(Γ, Δ, Σ ∝ g; Λ ∝ g²; Φ ∝ 2g(1−g); g = 4−√15; the Γ metatile holds one Mystic and one normal tile) and the charges 4, 4, 4 (Γ₁, Δ, Σ), 2 (Λ), {0, 2} (Φ), 0 (the rest):

    4(t_Γ₁ + t_Δ + t_Σ) + 2 t_Λ + 2 p t_Φ = 14 t_Γ₂   ⟹   12g + 2g² + 4p g(1−g) = 14g   ⟹   p = 1/2.

The Mystic is a fraction g/(1+g) = (5−√15)/10 of all tiles (0.11270, matching the depth-5 patch to 1e-4), the valence-4 density is 2g/(1+g) = (5−√15)/5, and ρ_X = 2g/(1+g) = 1 − √15/5.
The measured sequence is consistent: p_k − ½ shrinks by 0.43 and then 0.48 per depth, and Aitken's extrapolation of depths 3, 4, 5 gives 0.511 (computed in the script).
What remains an assumption is the atlas (the per-species charges, measured at depths 3 to 6, 131 contact classes), not p.

Suggested wording (not applied): in `Readme.md` and `SPECTRE.md` §9, replace "unresolved" by "p = ½ exactly, given the atlas; see `CICY.md`"; in CICY, `x_density(F(1,2))` is the closed form and the first item of `missing_for_physical()` can go.

## 2. Chair44: two implementations agree, and the notch axis is almost always determined

CICY's `chair44_nested` and this repository's `chair44_twoadic` build the same 168-label cell substitution (24 rotations × 7 cells, integer inflation 2) from different code.
**Exact (K1).** Both give a subset automaton of 209 reachable label sets, 41 of them with more than one label, and the same coincidence counts: the number of addresses of the 2^M cube whose label is
not determined by the M binary digits is c(M) = 28·2^M − 48 and the label excess is E(M) = 140·2^M + 48 (checked here for M = 1..7; for all M by CICY's finite certificate). With `CICY_DIR` set,
`chair44_notch_axis.py --tests` compares our table with CICY's `coincidence_table(7)` and runs its four-diagonal-tubes check.

**Exact (K2, new).** Ask about the notch vector d = g(1,1,1) of [`LINES.md`](LINES.md) instead of the label. Its axis (d up to sign) is determined by the address everywhere except at **exactly 16 addresses
for every M ≥ 2: the 8 corner cells and the 8 central cells** of the 2^M cube. Proof by finite certificate (the count is a linear functional of Tᴹe₀ for the 209-state transfer matrix, so a_M − 16 obeys a recurrence of
order ≤ 210; it vanishes for M = 2..260), and the positions are checked for M = 2..6.
**Counted (K3).** Where the label is undetermined but the axis is not (the other 28·2^M − 64 addresses), the possible notch vectors are exactly ±v: the six remaining labels are a coset of the order-6 dihedral
group fixing the notch diagonal (CICY's `six_is_a_dihedral_coset`), which moves the notch vector by at most a sign. At the 16 exceptional addresses all 24 labels, so all 8 corner directions, are possible.
So the Chair44's notch axis field is a continuous function of the 2-adic address except at 16 points; only its sign is undetermined on the four tubes. Not shown: any link between that sign and the
P/C statement of [`CHAIR44.md`](CHAIR44.md) (PC is a symmetry, P alone is not).

## 3. Other contacts, none built on

* **Parity.** CICY's `parity.py` confronts Conjecture 1 of the Spectre-substrate paper (a parity-asymmetric crossover) with birefringence and CMB band powers. The Spectre is strictly chiral as a polygon
  (CONNECTIONS row 22), but every observable computed here is blind to it: flux closure, line sums, the density, and now the abelian Wilson energy ([`WILSON.md`](WILSON.md), mirror outline, same energy to 1e-9).
  A parity-odd observable has to be signed like the handedness theorem (Im Z_k < 0 for k ≤ 900, `research/`), not an energy. That bears on what Conjecture 1 can mean; it tests nothing in the CMB data.
* **Log-periodic search at 2π/log(4+√15).** In the exact density of [`CLOCK.md`](CLOCK.md) the decaying component has ratio 4−√15 > 0 per level: monotone, not oscillating. That is a statement about one
  position-space observable and does not address the diffraction-based search of `parity.py`.
* **The cusp torus.** CICY's six-sphere work puts the Hat as a subgraph of the Laves [3.4.6.4] substrate, an A₂ structure with twelve bond orbits, and finds the monodromy group is the kernel of a ℤ/12 character
  (the η² character of SL(2,ℤ), whose abelianisation is ℤ/12). This repository has twelve edge directions (ℤ[ζ₁₂] = A₂ ⊕ A₂) and a holonomy group Dic₁₂. Both contain a 12 for different reasons: **Numerology** until a map is exhibited.
* **Nariai / entropic gravity.** CICY re-derives δA = 4 S_rel on the Nariai horizon. [`CLOCK.md`](CLOCK.md) §3 asks for a derivation of the interaction density ρ; this is the nearest derivation target, not a connection yet.
