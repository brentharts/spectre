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
| discharge diff audit with the archives (`r44_diff_audit.py`) | reported as PASS (67/67 files) when it was run; its output was not saved. It is superseded by the next row | seconds |
| independent discharge audit (`discharge_audit_indep.py`, output `out_discharge_audit_indep.txt`) | **PASS (67/67 files)**. Written from the rule of `discharge_diff_audit.sh` without running it. The inputs are the seven delivery ZIPs under `proof/external_lean` at `838bca5` and the 67 files of `lean/R44/R44/Proved` at `d53d224`. Each file is compared with the newest delivery that contains it. It also compares every declaration's statement (header up to `:=`, comments and spacing removed). No delivered statement changes beyond notation: `|x|`→`abs x`, extra parentheses, `section`→`«section»`, and `theorem`→`def` for five structure-valued declarations. `native_mesh_dihedral_semantics` in `CompleteDihedralList.lean`, a `sorry` placeholder in the delivery, becomes a call to the proved theorem of the same statement in `MeshSemantics.lean`. The added declarations are helper lemmas, which the build proves. Seven changed lines, all marked, prove a finite check by `native_decide` where the delivery used `rfl` or a tactic proof. Their results are trusted in the same way as the other `native_decide` hooks | 1 s |

Axioms:
- The `build_axioms.log` in the tree after `lake build` (saved by `r44_build.py` after the build, as
  `committed_axioms_log` in the JSON) is byte-identical to the file committed at `d53d224` (53,556 bytes, 169 theorems;
  compared with the file fetched from GitHub on 2026-10-26).
- The 162 fresh `#print axioms` reports agree with it, up to the `R44.` prefix in names.
- No theorem depends on `sorryAx` or on any axiom other than `propext`, `Classical.choice`, `Quot.sound` and named
  `native_decide` hooks.
- `registration_of_tiling` and `tiling_homochiral`: the standard three plus 13 hooks.
- `geometric_hierarchy_unique`: 17 hooks.
- `r44_einstein`: 21 hooks.

The `native_decide` hooks mean those finite checks trust Lean's compiler as well as its kernel.

The first container was preempted at about 17:19 UTC and the run restarted automatically. The total cost was about $0.40.
