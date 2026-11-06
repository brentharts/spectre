![spectre spacce](spectrespace.png)

Where the flips come from. The mirror is applied to every slot of every supertile, not only Gamma. The run shows 567 parent-to-child placements with determinant -1 and 126 with +1; all 126 of the +1 ones are the Gamma1/Gamma2 pairs. So the flip is uniform: a tile’s handedness at level n is the same for every tile, and the parity is one global clock per level. What Gamma does carry is the only non-mirror twist, Gamma2’s 30° rotation with the a and b edges swapped. The 30° angles appear only among the leaf tiles, and the per-tile information is that rotation angle, not the flip.

Making the flip smooth. A reflection can’t be reached continuously from the identity in the plane, but it can in 3D. The map L → diag(L, det L) sends the 2D transforms into 3D rotations. Under it, a planar reflection across a line becomes a half turn about that same line. So each flip becomes a half twist about its mirror axis, and a root-to-leaf chain becomes a ribbon that twists at every level. The figure shows these ribbons for every parent-to-child edge, coloured by whether the ribbon’s normal points up or down, with the tiling on the floor.

What’s still open.

Twist direction: whether each half twist is +π or -π is not fixed by the 2D data, so it is a free choice per flip. --twist-sign sets it to +1, -1, or alternating with level; alternating is where an arrow of time could enter.
Ribbon appearance: the width direction (--ribbon-angle) and the height-per-level scale are rendering choices, not properties of the tiling.
Sealed surface: every node stores its four corner points, which could serve as a patch for the NURBS stage.
