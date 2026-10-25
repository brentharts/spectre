# Exact checks for the Mystic classes at the Hat point

Exact Q(√3) arithmetic, standard library only. These scripts were written independently of `spectre.py` and
`spectre_mystic_fibres.py`, by an adversarial referee, from the algorithm of `spectre.py`. They reproduce its floating-point
patches to 2.3·10⁻¹³ through level 5.

| script | checks |
|---|---|
| `c3_alln.py` | **Registration for every level.** Each level-m child placement map is in p6m of the kite lattice K iff a condition on the classes mod Λ of the level-(m−1) quad points holds, and the next classes depend only on the current ones. The class state returns after one level, so every placement at every level is in p6m(K). Composites of p6m maps are in p6m, so every tile of every level is K-registered. |
| `run_c3.py N` | Levels 1..N: every placement is in p6m(K), each level has one determinant ((−1)ⁿ), and the classes mod Λ are in bijection with the relative rotations (1, 5, 6, 6, 6 for N = 5). |
| `c3_overlap.py N` | The level-N patch is a packing: no kite is covered twice (36,200 kites at level 4). |

```
cd mystic_exact && python3 c3_alln.py && python3 run_c3.py 5 && python3 c3_overlap.py 4
```

With these, "at most six Mystic classes, and exactly six for every n ≥ 3" is proved for every n:
1. registration holds at every level (`c3_alln.py`);
2. there is one handedness per level (each level composes every placement with `MIRROR`);
3. hence classes correspond to rotations in 60°ℤ;
4. a Δ supertile contains a Δ supertile, so all six rotations persist from level 3.

This is a routine fact about p6m. "Mystic" here means Γ2 = Tile(√3, 1) alone, whereas in the Spectre papers the Mystic is
the pair Γ1 ∪ Γ2; using Γ1 or Γ2 gives the same classes.
