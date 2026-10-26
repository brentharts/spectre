# Certificate: unique hierarchy of atlas tilings (input (H))

`check_chair44_hierarchy.py` checks the finite conditions C1–C3 of Kamenetsky's Proposition 7.1 (arXiv:2610.00916)
for the Chair44 substitution:
- **C1:** the hierarchical contacts number 30, and the coarsening closure gives P* with |P*| = 44 and Σ(P*) = 2P*,
  with no offset pose admissible.
- **C2:** there are 135 facet triples, and they are tight.
- **C3:** a tile has 33 enclosures, 18 dead and 15 live. Every live one has exactly one compatible parent (U), and its
  central child completes that parent (Pr).

It also checks that P* is exactly the paper's 44 legal contacts. The paper's Lemma `lem:hier` turns these checks into
the proof of (H).

The code was written independently of Kamenetsky's `cert6.py` and of `chair44_independent/`. It reads only
Tsiokos's solid, `../chair44_data/r44_solid.json`, as data.

    cd hierarchy_certificate && python3 check_chair44_hierarchy.py      # about 1 minute
