# Audit of Tsiokos's registration theorem (R)

Date: 2026-10-25. Subject: arXiv:2609.19214, §§4–5. The paper's Theorem 5.6 says that every tiling by the solid Q is registered.
Source checked: `ioannist/six-birds-tiles` at `d53d2245bc91c01cf81b6490a1cd6732e7f73bd3`, `paper/tex/sec4_companions.tex`
and `sec5_registration.tex`. Code: `independent/registration_audit.py`, standard library only, exact rationals, about 3 min.

## Verdict

**I found no gap.**
- I checked every step of the written argument (§4 and §5).
- I recomputed every finite input it uses with code written from the paper's statements, without the authors'
  verifiers.
- Tsiokos's certificate of 5,273 collision witnesses checks out entry by entry against independently recomputed
  partner lists.
- Two steps can be replaced by simpler arguments; see the notes under steps 8 and 9.
- I also found a sharper version of Prop. 5.2: see "A sharper census" below.

This makes (R) independently checked. It is not yet refereed by a journal. The Lean development, with its
`native_decide`, is **not** relied on here.

## The argument, step by step

| Step | Paper | Claim | Status |
|---|---|---|---|
| 1 | L 4.1 | Packings by Q are locally finite (disjoint inscribed balls). | correct |
| 2 | L 4.2 | Tangent-cone solid angles sum to ≤ 4π, and to 4π in a tiling. At a generic edge point the planar sectors sum to 2π. | correct. Q's boundary is an embedded 2-manifold (A1, A2), so each tile contributes one sector at a generic point. |
| 3 | L 4.3 | The dihedral list: feature edges π ± α_j and π ∓ β_j, all other edges π/2, 3π/2 or π. The 24 deviations are distinct and < π/4. | **recomputed from the mesh** (A3): all 6,408 edges, 1,536 feature edges, 32 per class |
| 4 | L 4.4 | At a generic feature-edge point there is exactly one other tile, with the complementary feature edge. | correct: three feature sectors are > 9π/4; two plus an ordinary one are > 2π; one alone cannot complete 2π |
| 5 | L 4.5 | The interior solid angle is > 4π/3 at every point of a feature graph. | correct. Q is locally the hypograph of a 12/100-Lipschitz tent, since features lie ≥ 0.24 from the panel edges (A2). The arithmetic is in A4. |
| 6 | T 4.6 | One tile S accompanies the whole feature graph E(F). | correct: a finite closed cover of a connected space, with disjointness from step 5 |
| 7 | L 4.7, C 4.8 | E(F) = E(F′), with opposite polarity. The pose is (G, c_u − Gc_v), G a signed permutation, t ∈ ⅛Z³. | correct. The compact-isometry argument is the clean one. G ranges over all 48 frames, so mirror images are included. |
| 8 | L 5.1 | An overlap of eighth-grid cubes gives an overlap of interiors. | correct: the dents are ≤ 12/10000 deep, which is less than ρ = 1/100 (A4) |
| 9 | P 5.2, L 5.3 | Census: 6,862 → 1,545 collide → 5,317; 5,273 rejected by witnesses; 44 survive, equal to A44. | **recomputed** (B1–B5). The logic of L 5.3 is correct. |
| 10 | L 5.4 | The baseline cells of the companion component tile Z³. | correct: two integral unit squares sharing an interior point are equal |
| 11 | L 5.5 | The component's solids cover R³. | correct. The written tube-map proof needs App. E. The clamping proof needs only step 8 and an interior ball of radius > √3/50. |
| 12 | T 5.6 | Every tile is in the component, so the tiling is registered, and face-adjacent pairs lie in A44. | correct. With the clamping proof, every tile of the tiling meets the interior of a component tile, so it equals that tile. |

Notes:
- **Step 8.** The census uses the same overlap test for its option collisions. The code checks it on all
  299,975 of them.
- **Step 11.** The clamping proof suffices, so the tube homeomorphism of App. E is not needed for (R).
- **Atlas tilings.** Registered face-adjacent chairs share a whole panel. The companions of that panel's features
  are the owner of the cell across. So every face contact is a companion contact, which lies in A44. This is
  exactly the "atlas tiling" hypothesis of the hull paper.

## What the code checks (`registration_audit.py`)

Only the mesh (`vertices`, `triangles`) is read as geometry. The 192 pyramids are recovered from it.

**Mesh**
- **A1.** It is a closed oriented 2-manifold: every edge lies in exactly two triangles, with opposite
  orientations. It is connected, its Euler characteristic is 2, and its volume is 7.
- **A2.** It is the chair's boundary with 192 pyramids.
  - Every non-flat vertex is the apex of four triangles over an axis-parallel square of half-width 1/100. The
    pyramid has height a/10000 with 1 ≤ |a| ≤ 12, and sits at panel centre ± (1/8, 1/4) or ± (1/4, 1/8). There
    are eight pyramids per panel.
  - The other 3,504 triangles lie in the 24 panels, face outward and avoid the open bases. On each panel their
    interiors are pairwise disjoint and their areas sum to 1 − 8/2500, so they tile the panel minus the bases.
  - Hence the surface is embedded and Q is the decorated chair the paper describes.
  - The recovered pyramids equal the file's `patches`.
- **A3.** Every edge is flat, orthogonal, or a pyramid edge. Classifying by exact cos² of the normal angle gives
  1,536 pyramid edges in 48 classes (type, magnitude, convex or reflex), 32 per class. The 24 deviations are
  distinct and < π/4.
- **A4.** The arithmetic of Lemma 4.5 (8L² < 1 and 63L² < 1 for L = 3/25), Lemma 5.1, and the clamping step.

**Census**
- **B1.** |F0| = 6,862, over all 48 frames, with translations in ⅛Z³. All 4,947 same-polarity full matches
  collide with the root, so excluding them is harmless.
- **B2.** 1,545 poses collide with the root. That leaves 5,317 in F1, of which 83 are integral. F1 is closed
  under inversion, and every Partners(u) is nonempty.
- **B3.** My own witness search, using the root's or the mate's features, rejects 5,285 poses and keeps 32. All
  32 are integral and proper.
- **B4.** The 44 poses whose pyramids fit face to face on every shared panel, out of 2,388 face-neighbour poses
  over 48 frames, equal the authors' 44 contacts and contain the 32. The 30 contacts that occur in supertiles
  (σ-closure of the children's contacts) all survive.
- **B5.** Each of the authors' 5,273 witnesses satisfies clauses (i)–(iii) against the recomputed Partners lists,
  with 299,975 option collisions as stated. What their certificate leaves is exactly the 44.

```
python3 registration_audit.py solid/r44_solid.json certificates/candidate_certificate.json \
        certificates/companion_collision_certificate.json      # six-birds-tiles layout
python3 registration_audit.py chair44_data/r44_solid.json chair44_data/r44_atlas.json   # spectre layout
```

## A sharper census

The same one-step rule as Prop. 5.2, also allowing witnesses on the mate, rejects 12 more poses. So only **32**
relative poses can occur between feature-sharing tiles. They are the 30 supertile contacts plus 2. The 12 extra
rejections are all among the 14 atlas contacts that never occur, so this is consistent with the hull theorem
(only 30 contacts occur). It does not replace that theorem: 2 non-occurring contacts survive any one-step test.

## What remains

- **A journal referee.** This audit was carried out with an AI assistant (Claude) for T. Romanov. It should be
  read as a careful independent audit, not as peer review.
- **Mechanisation.** The written steps 1–7 and 10–12 are short and standard. Their only computer inputs are A1–A4
  and B1–B5, which are verified above in exact arithmetic.
