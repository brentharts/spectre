# Certificate: the Spectre perimeter recurrence for every order

`check_spectre_boundary_proof.py` proves that the perimeter of a non-Mystic order-k Spectre supertile is
P(k) = 2F(3k+5) + 4 for every k. Equivalently, P(k+3) = 5P(k+2) − 3P(k+1) − P(k).

It also proves the recurrence of the Mystic perimeter Q(k), with characteristic polynomial x(x−1)(x²−4x−1), and the
recurrence of each slot-pair contact sequence.

The method is in the script's docstring:
- Track pairs of placed level-k outlines by their relative placement. It is linear in the quad vector q_k, and
  q_{k+1} = Φ q_k.
- Close the pair types under the substitution: 93 live types.
- Prove every other type separated at every order: by radius bounds for k ≤ 60, and by the exact
  eigen-decomposition of Φ on q_0 for k > 60.
- Check level 0 on the polygons.
- Close the argument with Cayley–Hamilton on the resulting 95 × 95 integer matrix.

Non-overlap of the supertiles is proved along the way, not assumed.

The only input is Kaplan's construction. `spectre_exact.py` is an independent exact transcription of it, written for
the referee review of this paper (see `REVIEW_FIXES.md`); it is checked against Kaplan's published float outline. None
of the code here imports the rest of this repository.

    cd boundary_certificate && python3 check_spectre_boundary_proof.py     # about 2 minutes

It needs numpy, sympy, mpmath and shapely. The output of a run is in `out_spectre_boundary_proof.txt`.

## Theorems 2 and 3: the limiting supertile and the window

`check_spectre_fractal.py` checks the finite inputs of Theorem 2 (`thm:frac`), the box dimension of the boundary of the
limiting supertile. Its output is in `out_spectre_fractal.txt`. It also records that sibling pairs of two non-Mystic
children reach the block that carries ν, so the lower bound applies to K itself and not only to the Mystic limit.

`spectre_window.py` checks Theorem 3 (`thm:window`), and its output is in `out_spectre_window.txt`. It computes:
- the internal planes and the graph-directed system;
- the 1192 contact types;
- that the rounding keys are exact: within one class, distinct translations are at least 1.909 apart;
- that no type is the identity. This gives positive area, through the mass distribution principle;
- the characteristic polynomial;
- the block that carries √2+√3, and the window types that the lower bound reaches. All four are reached.

The theorem proves the seam dimension and a lower bound for ∂W. An upper bound for ∂W would need the neighbouring
windows of a covering of internal space, which is not computed here.

    cd boundary_certificate && python3 check_spectre_fractal.py     # about 3 minutes
    cd boundary_certificate && python3 spectre_window.py            # about 3 minutes
