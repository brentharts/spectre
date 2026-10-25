# Independent verification (second implementation)

This is a second, separately written implementation of every computation behind the Chair44 hull and symmetry results of `spectre_substrate`. It also contains an
audit of Tsiokos's registration theorem. It was written from the papers' definitions by an adversarial referee, without
Tsiokos's verifiers and Brent Hartshorn's Chair44 scripts. It uses the standard library only and reads only
`chair44_data/r44_solid.json`, `r44_atlas.json` and `companion_collision_certificate.json`. (The same code is kept in `romanov360/big-bang-monotile`, `research/independent`;
nothing here depends on it.)

| file | what it checks |
|---|---|
| `registration_audit.py` | The finite inputs of the **registration theorem** (Thm 5.6): the mesh of Q (closed manifold, 192 pyramids recovered from the triangles, all 6,408 dihedral angles), the census 6,862 → 1,545 → 5,317, and each of the 5,273 collision witnesses in `chair44_data/companion_collision_certificate.json`. See `REGISTRATION_AUDIT.md` for the step-by-step check of the written proof. |
| `c44core.py` | poses, cells, refinement; the atlas from the pyramid geometry (2,388 face-neighbour poses → 44) |
| `c44facet.py` | the atlas without the pyramids: contact closure 21 → 30, coarsening closure 30 → 44 with Σ(P\*) = 2P\*, 135 facet triples, tight |
| `c44sub.py` | 168-label substitution, legal-block closure 168 → 600 → 1278 → 1398 → 1410 |
| `c44adm.py`, `run_c1.py` | the 360,189 admissible blocks and the window lemma |
| `c44lemma_cert.py`, `lemmaA_orbits.json` | the window lemma as a certificate: 15,214 O-orbit representatives. `verify` recomputes the atlas, L and Adm, checks that Adm is the union of the orbits, and checks every representative's windows. |
| `c44controls.py`, `c44controls2.py` | negative controls for the lemma |
| `c44shells.py` | the finite part of (H): 33 first shells, each with exactly one compatible parent; central completion 1/18/14 |
| `c44contacts_in_L.py` | the contacts in legal blocks are exactly the 30 hierarchy contacts |
| `c44coinc.py` | primitivity at power 3; modular coincidence at depth 3 |
| `c44sym.py` | O-equivariance; the 27 Φ-periodic legal blocks (stabilisers 1 × 24, D4 × 3); the axial graphs G_K |
| `c44patch.py`, `c44patch2.py` | explicit tilings from mixed walks; all isometries of a large central region |
| `INDEPENDENT_REVIEW.md` | the referee report, with its resolution table |

From this directory:
```
S=../chair44_data/r44_solid.json
python3 registration_audit.py $S ../chair44_data/r44_atlas.json ../chair44_data/companion_collision_certificate.json   # ~3 min
python3 c44lemma_cert.py verify $S lemmaA_orbits.json      # ~3 min
python3 run_c1.py $S c1.pkl                                # ~3 min: atlas, closure, 360189 blocks, lemma
python3 c44shells.py c1.pkl; python3 c44contacts_in_L.py c1.pkl; python3 c44coinc.py
python3 c44sym.py c1.pkl                                   # ~1.5 min
python3 c44patch2.py c1.pkl 6 40                           # ~5 min
python3 c44controls.py $S c1.pkl; python3 c44controls2.py $S c1.pkl   # ~45 min
```
