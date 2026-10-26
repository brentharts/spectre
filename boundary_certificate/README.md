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
