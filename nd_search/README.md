# Search records for Kamenetsky's Problem 9.1 (a four-dimensional chair)

These files are copied unchanged from `research/spectre_review/code/` on branch `spectre-paper-review` of
`romanov360/big-bang-monotile`, at commit `a58133f`. They are the records behind the last paragraph of the paper's
conclusion. They are not part of the paper's build and were not rerun.

| file | what it is |
|---|---|
| `chair_nd.py` | the N-dimensional chair and Kamenetsky's coarsening-closure test (C1). Its default cap is 300 poses. `python3 chair_nd.py 3` runs the 3D count. |
| `chair_nd_fast.py` | a vectorised version of the same test (`python3 chair_nd_fast.py validate` compares the two) |
| `search4d.py` | the bounded 4D search. Families F (the 3D solution on the facet v₃ = 0) and R (random), alternating, cap 400 |
| `anneal4d.py` | simulated annealing on \|Σ(P₀)\|, cap 2500. Offset states score 2\|P\| and closed states score 0 |
| `out_chair_nd_3d.txt` | 3D: of 2187 assignments, 2178 hit the cap, 6 have an offset and 3 are closed (30 → 44 → 44) |
| `out_search4d.txt` | three seeds: 4288 + 4212 + 4284 = 12,784 assignments, half F and half R, all at the cap |
| `out_anneal4d.txt` | four runs (seeds 11, 12 and 13 from random starts; seed 21 from Kamenetsky's canonical family): 2399 + 3208 + 3424 + 3101 = 12,132 evaluations. The best is 273 (seed 11), and 273 is a \|Σ(P₀)\| value, since an offset score is even. |

## Known limits of the code (from a code review; the files are left as they were run)

- In `chair_nd_fast.Fast.closure`, the second `cap` exit (`len(newP) > cap`) returns no Σ(P₀) size even with
  `want_sigma0=True`, so `anneal4d.score` falls back to |P| there, a different quantity. That path needs |P| > 2500,
  so it cannot produce any objective value at or below 2500, the reported 273 included.
- `Sigma()` stops after the first batch of 4096 candidates in which the count exceeds the cap, so a Σ size above the
  cap is only a lower bound. Values at or below the cap are exact.
- `anneal4d.py` updates `best` only on accepted moves, so the starting state of a restart (including the canonical
  start of seed 21) is never itself a candidate. Offset states score 2|P|, which can be below the Σ values of closable
  states, so the walk is drawn toward offsets. An offset score is even, and 273 is odd, so 273 is a |Σ(P₀)| value.
