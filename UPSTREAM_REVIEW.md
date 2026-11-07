# Upstream review and research plan (2026-11-07)

The new research plan is in `NEXT_STEPS.md` and pushed to PR 13 (commit e25a058). The sync with the parent's master is not done, because of one merge conflict and one non-fast-forward push.

## Sync status
- **PR 13 branch:** merging upstream master conflicts in `spectrepaper.py`, in the abstract and the pancake section. Brent's "phys rev IV–VI" commits rewrote the same passages the PR edits. That needs a choice between his wording and ours, so the merge was backed out rather than picking a side.
- **The fork's `master`:** it couldn't be fast-forwarded. It has an extra commit that upstream doesn't have, `boundary_certificate/window_boundary_closure.py`, so updating it means overwriting that commit. Both branches were left alone until it is decided how to handle them.

## What Brent has been doing
- `spectrespace.py` keeps the full substitution tree and turns each 2D mirror flip into a half twist in 3D. That gives the twisted-ribbon picture. Parity turns out to be one global clock per level: only the Γ₁/Γ₂ placements keep det +1, and per-tile information is just the rotation angle.
- `spectreseal.py` thickens the ribbons into a closed surface and reports its genus. The genus depends on the thickness and blending chosen for rendering.
- The holonomy figure makes four claims:
  - The surface normal is forced: n = det(L)·ẑ.
  - The holonomies form a group of order 48.
  - The twist sign is only a gauge choice.
  - Some vertices carry a genuine Z₂ "vortex": 28 at depth 3 and 372 at depth 4. All are three tiles meeting at 120°, and none touches a Mystic.
- `spectre_holonomy.py`, the script behind that figure, isn't on master yet. The figures README mentions it.

## New plan (R1–R10), in recommended order
1. **R9 = C step 3:** compare the spectral gaps of the item-4 tight-binding Hamiltonian with the gap-labelling module. This is still next.
2. **R1:** exact vortex density. A vortex is a local pattern, so our gap-labelling result predicts its density lies in ℤ + ℤ(5+√15)/10. This is a falsifiable test that ties Brent's figure to our work.
3. **R5 and R4:** short exact proofs.
   - R5 derives the parity-clock counts in closed form.
   - R4 identifies the order-48 group exactly and proves the twist sign is gauge.
4. **R3 and R2:** together they could prove the flux is topological.
   - R3 decides whether the flux is a nonzero cohomology class.
   - R2 checks whether the vortices form a model set with their own window.
5. **R6 and R8**, numerical, run in parallel:
   - R6 measures ribbon linking, twist and writhe. Holonomy can't see the twist sign, but linking can, so this is the last real test of an arrow of time in the ribbon picture.
   - R8 adds the vortex flux to the tight-binding Hamiltonian.
6. **R7:** check whether the sealed surface's genus is a real invariant or a rendering artefact. Then the remaining E, F and B.

It would help to get `spectre_holonomy.py` from Brent before starting R1–R4, so the vortex classification uses his conventions exactly.
