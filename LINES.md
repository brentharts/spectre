# LINES.md — the notch vector along axis lines of a Chair44 patch

Labels as in [`CONNECTIONS.md`](CONNECTIONS.md). Script: `chair44_polarisation.py` (`--tests`, about 1 second; in `make facts`).
It extends [`chair44_lines.py`](chair44_lines.py), which treats named lines (the chair stacks and the corner lines), to **every** axis line of a patch.

## 1. What is measured

Build the depth-n cell patch (side M = 2ⁿ, 8ⁿ cells) from the 168-label substitution of `chair44_twoadic.py`. Each cell carries a notch vector
d = g(1,1,1), a corner direction with components ±1. Sum d along a whole axis line.

| quantity | result | label |
|---|---|---|
| box mean of d | about 0 (1.7·10⁻³ at n = 6); the 168 labels sum to 0 | **Exact** (labels), **Counted** (patch) |
| sum **along** the line | bounded: \|sum\| ≤ 6 on all 3·4ⁿ axis lines, n = 3…7, three root labels | **Counted** |
| sum **across** the line | exactly (±M, ±M), a drift of (±1, ±1) per cell, on a fraction 0.5625, 0.5469, 0.5430, 0.5420, 0.5417 of lines (n = 3…7) | **Counted** |
| the fraction, exactly | 13/24 + 1/(3·4ⁿ⁻¹) for every n ≥ 3 and every one of the 168 root labels, so the limit is **13/24** (§2) | **Exact** |
| the four sign states (±,±) | nearly equal (2220, 2216, 2220, 2220 at n = 7; not exactly equal) | **Counted** |
| negative control | with the labels shuffled in space, no line is ballistic | **Counted** |
| body-diagonal lines | longitudinal drift per cell is below 0.5 for 0.913 of lines at n = 5 and 0.942 at n = 7; the largest is exactly 1 per cell (the chair stack) | **Counted** |

It is consistent with the exact corner-line result of `chair44_lines.py` (the x-axis corner line has a closed class of four labels with notch mean (0,1,1)).
What is new is that it holds for all axis lines, with a measured fraction and four equal sign states.

## 2. The exact fraction

"Transverse sum (±M, ±M)" means every cell of the line has the same pair (d_x, d_y) = s, i.e. the line word lies in Σ_s, the 42 labels with that pair.
The line at address (x, y) of the depth-n patch of root j is the word τ_{d₀} … τ_{d_{n−1}}(j), with τ_{(a,b)}(l) = σ[l,(a,b,0)] σ[l,(a,b,1)] and d_i the
i-th binary digits of (x, y). It lies in Σ_s iff j ∈ Pre_{d_{n−1}}(… Pre_{d₀}(Σ_s)), where Pre_d(T) = {l : τ_d(l) ⊂ T}. Starting from the four Σ_s, only
**18 subsets** are reachable, so counting digit strings is a walk on 18 states with an integer matrix Q. The count is aₙ(j) = uᵀQⁿw, and
**(Q − 4)(Q − 1)Q³ = 0 exactly**, so aₙ₊₂ = 5aₙ₊₁ − 4aₙ for n ≥ 3. For all 168 roots a₃ = 36 and a₄ = 140, hence aₙ = (13/24)4ⁿ + 4/3, i.e. the fraction
13/24 + 1/(3·4ⁿ⁻¹). The automaton's counts equal the brute-force counts for n ≤ 7. This is a statement about the lines of the patches σⁿ(j), that is, about
the 2-adic address (x, y); it is not a density in one fixed infinite tiling.

## 3. What it says

On almost all axis lines the notch walk closes along the line and runs away across it, in one of four transverse directions chosen about equally often. So
every line carries a **polarisation** (a vector perpendicular to it), and the patch as a whole averages to zero, which is the box/line gap of
[`CHAIR44.md`](CHAIR44.md) §4 seen in a vector observable.

**It gives no arrow of time.** The sum is the same traversed backwards, so the polarisation labels the line, not a direction along it
(CONNECTIONS row 19 records that guess as falsified for this observable). An arrow would need an observable that differs between the two orders.

## 4. The Spectre counterpart (negative)

`spectre_line_sums.py` sums the tile-orientation vector u = e^{i·rotation} along straight lines in a level-5 patch (34649 tiles; directions 0°, 30°, 60°, 90°; 400–500 lines each).
No component is ballistic: mean |S_k| grows as k^e with e between 0.26 and 0.62 (control with shuffled orientations: 0.39–0.55) and at k = 140 it is 7–10.5, up to 1.6 times
the control (a ballistic line would give about 140). **Counted**, sampled, one observable. So the Chair44 effect is not a general property of aperiodic monotiles; it goes with
the integer inflation (limit-periodic order). That reading is a **Hypothesis**: other observables (edge directions, Mystic classes, the handedness Z_k) are not tested.

## 5. Hypothesis, not tested

The Chair44 has the six signed face directions ±x, ±y, ±z, and the notch vector is one sign choice per axis. That is the vocabulary of the directional
timeline toys in [Quantum-Gravity-Dynamics](https://github.com/brentharts/Quantum-Gravity-Dynamics) (`directional/`). A test that could break a link:
use the notch step as a timeline step there, ask whether the "closed flux" contract (net displacement zero over a window) can be met along the line
(this script says yes along it, no across it), and ask whether the four transverse states behave as a two-bit hidden state in the CHSH computation of
`directional/retro_boundary_model.py`. Nothing here shows that they do; until a map is written down it is **Numerology** by the rule of CONNECTIONS §5.

## 6. Not covered

Irrational directions; the pyramid observable on all lines (see `chair44_lines.py`); a proof that the longitudinal sum is bounded for all n (it is only counted for n ≤ 7);
other Spectre observables.
