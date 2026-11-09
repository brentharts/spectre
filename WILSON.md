# WILSON.md — a U(1) Wilson action on the Spectre lattice, and what the loop energy can and cannot see

Labels as in [`CONNECTIONS.md`](CONNECTIONS.md). Script: `spectre_wilson.py` (about 16 s, in `make check`; `--table` prints the constants).
Written for open problem 1 of `quantgrav.tex` (spectre-monotile-py): the Wilson probe of `4D_Wilson_Loop_RG` is hypercubic, so all its corners are 90°, and it has no
action on the hexagonal lift lattice. Conventions are those of its `numerics/abelian_first_step`: action (1/2g²) Σ w_p θ_p², loop self-energy E = Jᵀ G J, ⟨W⟩ = exp(−g² E/2), g = 1.
Like everything here it is a finite check in a Gaussian analogue, not a statement about Yang–Mills.

## 1. Tiles as plaquettes: an area law and nothing else  (model T)

Links = tile edges of the level-2 patch (71 tiles, 588 edges), plaquettes = the 14-gon tiles (the graph of `research/su2_graph.py`).
**Exact (T1).** For a union R of tiles with boundary current J = Bᵀχ_R, E(J) = |R|. Reason: the tiles are independent as 2-chains on a disc, so B (tiles × edges) has full row rank and
B (BᵀB)⁺ Bᵀ = 1; the flux of every tile is an independent Gaussian. Tested on 20 random unions (max error 3e-14; the perimeter of the last one is 312, E = 48).
So the tiling alone, used as the plaquette surface, gives no perimeter term and no corner term. Corners have to come from plaquettes transverse to the tile.

## 2. The lift lattice A₂ ⊕ A₂  (model H)

Sites ℤ⁴ with u = p + q ω₆ (red), v = r + s ω₆ (green), on a torus of side N. Six link types (1, ω₆, ω₆² in each plane). Plaquettes: both triangles of each plane (weight 2) and the nine
red-green parallelograms (weight 1/3).
**Exact (H1).** The Fourier computation (6×6 Hessian per momentum, Landau-type gauge fixing) agrees with a dense real-space solve on the 4⁴ torus (5.6e-15); E does not depend on the gauge-fixing
weight; the Hessian annihilates pure gauge fields; E is invariant under a 60° rotation of the red plane. The weights 2 and 1/3 are not a free choice: they are the values for which the small-momentum
Hessian on pulled-back smooth fields equals the Maxwell one, vol (q²I − qqᵀ) with vol = 3/4 (relative error 1.3e-6; weights changed by 20 % give 0.19).

**Numeric (H2), thresholds chosen after seeing the data.** E = α P + b₆₀ c₆₀ + b₁₂₀ c₁₂₀ + b₉₀ c₉₀ (P links, c = corners by turn angle; the red-green corner is always 90°) on 12 structured loops
(triangles, rhombi, parallelograms and hexagons of the red plane, red-green rectangles; every side 3 or 4 links, N = 20):

| α | b(turn 60°, interior 120°) | b(turn 120°, interior 60°) | b(90°) | max residual |
|---|---|---|---|---|
| 0.3003 | −0.0696 | −0.3151 | −0.1390 | 0.068 (E up to 6.8) |

The sharp corner is separated from the other two by more than the residual; the blunt corner and the 90° corner are **not** (0.069 against 0.068). Eight held-out random interleavings of red and green
sides (3 links each) are predicted to 0.034. Finite size: N = 20 against 24 changes the constants by at most 0.002. These are effective constants at sizes 3 to 5 and the energy is the unconstrained self-energy
(the 'total' cross-check column of `abelian_first_step`), which contains scale-free interactions between sides at distance of the order of their length; they are not the programme's δm and log z_c, which come from
the constrained energy (Q a = 0) after block averaging. **Not done:** that constrained energy with a hexagonal averaging operator, which is what the open problem asks for.

## 3. The Spectre outline in that action  (H3)

The 14 edges of a Spectre lift to 8 red and 6 green unit steps (or 6 and 8), closed in A₂ ⊕ A₂. In the lift the 30° turns of the plane become red-green 90° corners: the outline has 7 turn-60° corners,
6 red-green corners and 1 straight pass.
* **Exact:** all 71 outlines of the level-2 patch have the same energy, E = 3.1826.
* **Exact:** the mirror outline (complex conjugation of the physical plane, a lattice map) is a different closed walk with exactly the same energy. The action is parity symmetric; it cannot see the chirality
  of the tile (compare [`CLOCK.md`](CLOCK.md) §2: the arrow-candidate is the chirality, and this probe is blind to it).
* **Numeric, Negative:** perimeter + corner predicts 2.884; the gap is 0.30, 0.18, 0.19 for the outline at scale 1, 2, 3 (every step repeated). It does not shrink with scale, so the Spectre outline is
  not described by one constant per corner type in the unconstrained energy, although random red-green interleavings are (0.034).
