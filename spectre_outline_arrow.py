#!/usr/bin/env python3
r"""spectre_outline_arrow.py - the Spectre outline as a closed, chiral, 14-step timeline.

quantgrav.tex (Conclusion) says no static structure has an arrow of time.  The directional toys of the paper need a closed walk (a "consistency
contract": the net flux of the steps is zero) and an order (which end is the past).  The Spectre outline is one: the 14 edges of a tile are unit steps in
12 directions 30 degrees apart (lift4d.py), they close, and the order matters because the outline is chiral.  This script tests exactly what that does
and does not give.

Exact (all 9 tiles of the level-1 patch and all 71 of level 2; the lift is integer):
  1. Every tile outline is the same cyclic word of turning angles, in units of 30 degrees:  w = (-2, 3, -2, -3, -2, 3, -2, 0, -2, -3, 2, -3, 2, -3),
     total turn -12 units = -360 degrees (the lift traverses the outline clockwise).
  2. Closure: the 14 unit steps sum to 0 exactly in Z[zeta_12], and no proper contiguous run of steps does (it is a minimal closed walk).
  3. Chirality (of the straight 14-gon): the mirror image, traversed in the same sense, has the word w reversed (reflection negates each turn, and
     restoring the sense of traversal negates them again and reverses the order).  It is NOT a cyclic rotation of w.  So reading the steps of a tile
     in the opposite order, with the turns kept, gives the mirror tile, not a tile of the tiling.
  4. All placements of the patch are proper rigid motions (det +1), so the mirror tile never occurs: the patch contains no step sequence in the reversed order.
  5. Reversal alone is not the content: reversing ANY polygon flips the sign of its signed area.  The content is item 3 (the reversed word is a
     different tile), not the sign.
  6. Colours: the red/green word of steps (even/odd directions) is  R R R R G G R R G G R R G G  for 63 tiles of the
     level-2 patch and its colour swap for the other 8 (those rotated by an odd multiple of 30 degrees); neither is chiral, so the chirality is in the
     turning word, not in the colouring.

Not shown: that this is a physical arrow of time, that the contract of the toy models is this closure, or any dynamics.  Closure of the steps is symmetric
under reversal (the same sum), as found in chair44_polarisation.py: the order information is carried by chirality, not by the flux.  Hypothesis
only: that a timeline whose steps are the outline of a chiral tile has a built-in direction.  A test that could break it: an achiral polygon
must give item 3 False, and does for the regular hexagon (tested below).  Note that the straight 14-gon is chiral here as a polygon, but the usual
statement that the Spectre needs no reflections refers to the curved-edge version; both are used in this repository, and only the polygon is tested.

    python3 spectre_outline_arrow.py            report
    python3 spectre_outline_arrow.py --tests
"""
import math
import sys

import numpy as np

import lift4d as L
import spectre_patch as sp

W6 = np.exp(1j * math.pi / 3)
Z12 = np.exp(1j * math.pi / 6)
WORD = (-2, 3, -2, -3, -2, 3, -2, 0, -2, -3, 2, -3, 2, -3)


def step_vector(d):
    return d[0] + d[1] * W6 + Z12 * (d[2] + d[3] * W6)


def outline(tile):
    """-> (steps as 4-integer vectors, direction index 0..11 of each, turning word in units of 30 degrees)."""
    n = len(tile)
    steps = [tuple(int(x) for x in np.array(tile[(i + 1) % n]) - np.array(tile[i])) for i in range(n)]
    dirs = [int(round(math.degrees(math.atan2(step_vector(s).imag, step_vector(s).real)) / 30)) % 12 for s in steps]
    turns = [((dirs[(i + 1) % n] - dirs[i] + 6) % 12) - 6 for i in range(n)]
    return steps, dirs, turns


def rotations(w):
    return [tuple(w[k:] + w[:k]) for k in range(len(w))]


def mirror_tile_word(w):
    """Turning word of the mirror image, traversed in the same sense (clockwise) as w: reflection negates every turn, restoring the sense by
    traversing backwards negates them again and reverses the order, so the word is w reversed."""
    return tuple(reversed(w))


def is_chiral(w):
    """True if the mirror tile has a different cyclic turning word, i.e. the polygon is not congruent to its reflection."""
    return mirror_tile_word(tuple(w)) not in set(rotations(tuple(w)))


def minimal_closed(steps):
    """True if no proper cyclic contiguous run of steps sums to zero."""
    n = len(steps)
    S = np.array(steps)
    for i in range(n):
        for ln in range(1, n):
            if not sum(S[(i + k) % n] for k in range(ln)).any():
                return False
    return True


def colour_word(steps):
    return tuple(0 if s[2] == 0 and s[3] == 0 else 1 for s in steps)


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    words, closed, minimal, signed_areas, colours = set(), True, True, [], set()
    ntiles = 0
    for lev in (1, 2):
        tiles, _, _ = L.lift_patch(lev)
        for t in tiles:
            ntiles += 1
            steps, dirs, turns = outline(t)
            words.add(min(rotations(tuple(turns))))
            closed &= all(np.array(steps).sum(0) == 0)
            minimal &= minimal_closed(steps)
            z = np.array([step_vector(np.array(p)) for p in t])
            signed_areas.append(0.5 * np.sum((z * np.conj(np.roll(z, -1))).imag))
            colours.add(min(rotations(colour_word(steps))))
    check("%d tile outlines (levels 1, 2): all have the same cyclic turning word" % ntiles, len(words) == 1 and min(rotations(WORD)) in words)
    check("total turn is -12 units (-360 degrees)", sum(WORD) == -12)
    check("the 14 unit steps sum to 0 in every outline", closed)
    check("no proper contiguous run of steps closes (minimal closed walk)", minimal)
    check("chirality: the mirror-reversed word is not a cyclic rotation of the word", is_chiral(WORD))
    check("negative control: the regular hexagon (-2)^6 is achiral, and the word (-2,-2,-1,-3,-2,-2) of a lopsided hexagon is chiral",
          not is_chiral((-2,) * 6) and is_chiral((-2, -2, -1, -3, -2, -2)))
    P = sp.SpectrePatch(3)
    dets = np.linalg.det(P.placements[:, :2, :2])
    check("all %d placements of the level-3 patch are proper rigid motions (det +1)" % P.n_tiles, np.allclose(dets, 1.0))
    check("reversing any polygon flips its signed area: the 80 tiles have one sign of area, and reversal gives the other (so orientation alone is not the content)",
          len(set(np.sign(np.round(signed_areas, 9)))) == 1)
    cl = sorted(colours)
    comp = lambda c: tuple(1 - x for x in c)
    check("colour word (0 = red even direction, 1 = green odd): two cyclic classes, one the colour-swap of the other (the 8+6 and 6+8 tiles of spectre_e8.py), and neither is chiral",
          len(cl) == 2 and min(rotations(comp(cl[0]))) == cl[1] and not any(is_chiral_word(c) for c in cl), "(%s)" % cl)
    return ok


def is_chiral_word(c):
    return tuple(reversed(c)) not in set(rotations(tuple(c)))


def report():
    tiles, _, _ = L.lift_patch(1)
    steps, dirs, turns = outline(tiles[0])
    print("directions (x30 deg):", dirs)
    print("turning word:", turns, "sum", sum(turns))
    print("colour word:", colour_word(steps))


if __name__ == "__main__":
    if "--tests" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    report()
    run_tests()
