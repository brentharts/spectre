# Independent rebuild of Tsiokos's Lean development (input (R))

Repository: `ioannist/six-birds-tiles` at `d53d2245bc91c01cf81b6490a1cd6732e7f73bd3`. Built on 2026-10-26 on Modal, in one
ephemeral container (8 cores, 16 GiB, no volumes), with `r44_build.py`. Full logs are in `r44_build_result.json`.

| step | result | time |
|---|---|---|
| toolchain | Lean 4.31.0 (commit 68218e8), Lake 5.0.0 | – |
| `lake exe cache get` | ok | 183 s |
| `lake build` | **ok**: "Build completed successfully (8638 jobs)" | 1702 s |
| `lake env lean R44/Axioms.lean` | ok: 162 `#print axioms` reports | 11 s |
| `scripts/controls.sh` | build, discharge build, sorry audit (0 executable, 0 marked), import closure, 4/4 negative controls and the scope regressions pass. The discharge diff audit fails only because the public `main` lacks the delivery archives | 83 s |
| discharge diff audit with the archives (`r44_diff_audit.py`) | **PASS (67/67 files)**. The archives are restored from the reader commit `838bca5`, whose `lean/R44/R44` sources are identical | seconds |

Axioms:
- The `build_axioms.log` that the build's metaprogram regenerates is **byte-identical** to the committed one (169 theorems).
- The 162 fresh `#print axioms` reports agree with it, up to the `R44.` prefix in names.
- No theorem depends on `sorryAx` or on any axiom other than `propext`, `Classical.choice`, `Quot.sound` and named
  `native_decide` hooks.
- `registration_of_tiling` and `tiling_homochiral`: the standard three plus 13 hooks.
- `geometric_hierarchy_unique`: 17 hooks.
- `r44_einstein`: 21 hooks.

The `native_decide` hooks mean those finite checks trust Lean's compiler as well as its kernel.

The first container was preempted at about 17:19 UTC and the run restarted automatically. The total cost was about $0.40.
