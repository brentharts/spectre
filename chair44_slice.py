#!/usr/bin/env python3
"""chair44_slice.py - 2D slices and projections of the Chair44 hierarchy, and their diffraction.

The depth-n patch of Chair44 is 8**n chairs inside a chair scaled by 2**n.  To compare it with
the Spectre we need 2D objects, and there are two exact ones:

  slice       the cells of the patch with z = c: a 2D tiling by chair cross-sections
              (2x2 squares and L-trominoes), a polyomino tiling
  projection  the number of occupied cells (or of chair centres) above each (x, y):
              by the projection theorem its Fourier transform is the k_z = 0 plane of the 3D one

Each chair is represented by the centre of its 2x2x2 bounding cube, c = g(1,1,1) + t, an integer
point, and by its notch direction d = g(1,1,1) (the missing corner), a vector in {+-1}^3.
Exact facts used and tested here:
  * the 8**n centres are distinct integer points in [1, 2**(n+1)-1]^3
  * sum of the notch vectors = 2**n (1,1,1)   (mean child matrix is I/4, see chair44_twist.py)
so a notch-weighted structure factor has |F(0)|^2 = 4**n per component, far below the random-walk
value N = 8**n: I(0)/N = 2**-n, alpha(0) = -1/3 (hyperuniform orientation field).

Diffraction is I(k)/N with I = |sum_j w_j exp(i k.r_j)|^2: it grows like N for a Bragg peak and
stays O(1) for diffuse scattering and falls (alpha < 0) for a suppressed, hyperuniform signal.  alpha(k) = log(I_{n+1}/I_n)/log 8 measures which.

CLI: python3 chair44_slice.py [--tests-only] [--depth 5] [--png figures/chair44_slice.png]
"""
import argparse
import sys

import numpy as np

import chair44_core as C

U0 = (1, 1, 1)


# ------------------------------------------------------------------ point sets
def centres(depth):
    """(N, 3) integer array: centre of the bounding cube of every chair of the depth-n patch."""
    return np.array([C.vadd(C.frame_apply(f, U0), t) for f, t in C.sigma_depth(depth)], dtype=np.int64)


def notches(depth):
    """(N, 3) array: notch (missing-corner) direction g(1,1,1) of every chair."""
    return np.array([C.frame_apply(f, U0) for f, _ in C.sigma_depth(depth)], dtype=np.int64)


def occupancy(depth):
    """Boolean 3D array of the occupied unit cells, and the tile index of every cell (-1 = empty)."""
    own = C.patch_cells(depth)
    L = 2 ** (depth + 1)
    tile = -np.ones((L, L, L), dtype=np.int64)
    for c, i in own.items():
        tile[c] = i
    return tile >= 0, tile


# ------------------------------------------------------------------ slices and projections
def slice_layer(depth, axis=2, c=0):
    """2D array of tile indices of the layer {cell[axis] = c} (-1 = empty)."""
    _, tile = occupancy(depth)
    return np.take(tile, c, axis=axis)


def slice_cross_sections(depth, axis=2):
    """{layer c: sorted tuple of cross-section sizes} of the chairs cut by each layer."""
    out = {}
    for c in range(2 ** (depth + 1)):
        lay = slice_layer(depth, axis, c)
        ids = lay[lay >= 0]
        out[c] = tuple(sorted(int(x) for x in np.bincount(ids)[np.unique(ids)]))
    return out


def projection(points, axis=2, weights=None, size=None):
    """2D array: total weight of the points above each (x, y) (project along `axis`)."""
    keep = [a for a in range(3) if a != axis]
    size = size or int(points.max()) + 1
    out = np.zeros((size, size))
    np.add.at(out, (points[:, keep[0]], points[:, keep[1]]), 1.0 if weights is None else weights)
    return out


def interface_edges(layer):
    """Number of unit edges inside the layer between cells of two different tiles."""
    a, b = layer[:-1, :], layer[1:, :]
    c, d = layer[:, :-1], layer[:, 1:]
    return int(np.sum((a >= 0) & (b >= 0) & (a != b)) + np.sum((c >= 0) & (d >= 0) & (c != d)))


# ------------------------------------------------------------------ diffraction
def structure_plane(points, weights=None, M=512, axis=2, kz=0.0):
    """Structure factor F(kx, ky) = sum_j w_j exp(i k.r_j) on the M x M grid k = 2 pi m / M, on the
    plane k_axis = kz (kz = 0 is exactly the FFT of the projection)."""
    keep = [a for a in range(3) if a != axis]
    if kz == 0.0:
        img = projection(points, axis, weights, size=M)
        return np.fft.fft2(img)
    ph = np.exp(-1j * kz * points[:, axis]) * (1.0 if weights is None else weights)
    img = np.zeros((M, M), complex)
    np.add.at(img, (points[:, keep[0]], points[:, keep[1]]), ph)
    return np.fft.fft2(img)


def intensity(points, weights=None, M=512, axis=2, kz=0.0):
    return np.abs(structure_plane(points, weights, M, axis, kz)) ** 2 / len(points)


def local_max(I, centre, r):
    """Maximum of I within (2r+1)^2 pixels around `centre`, periodic grid."""
    ix = (centre[0] + np.arange(-r, r + 1)) % I.shape[0]
    iy = (centre[1] + np.arange(-r, r + 1)) % I.shape[1]
    return float(I[np.ix_(ix, iy)].max())


def top_peaks(I, count=12, exclude=3):
    """Largest local maxima of I (in the grid sense), as [(value, (i, j))], well separated."""
    order = np.argsort(I, axis=None)[::-1]
    out = []
    for flat in order:
        ij = np.unravel_index(flat, I.shape)
        if all(min(abs(ij[0] - p[1][0]), I.shape[0] - abs(ij[0] - p[1][0])) > exclude or
               min(abs(ij[1] - p[1][1]), I.shape[1] - abs(ij[1] - p[1][1])) > exclude for p in out):
            out.append((float(I[ij]), ij))
        if len(out) >= count:
            break
    return out


def exponent_track(depths, M=512, weights_fn=None, count=12, axis=2):
    """Track the strongest peaks of the top level down the levels.  Returns rows
    (k index, [I_n], alpha per level step); alpha = log(I_{n+1}/I_n)/log 8, each I the maximum
    within the peak width 2 pi / L_n of the position."""
    I = {}
    for d in depths:
        P = centres(d)
        w = None if weights_fn is None else weights_fn(d)
        I[d] = intensity(P, w, M, axis)
    top = depths[-1]
    rows = []
    for val, ij in top_peaks(I[top], count):
        vals = []
        for d in depths:
            r = max(1, int(round(M / 2 ** (d + 1))))
            vals.append(local_max(I[d], ij, r))
        al = [float(np.log(vals[i + 1] / vals[i]) / np.log(8)) if vals[i] > 1e-12 else float('nan')
              for i in range(len(vals) - 1)]
        rows.append((ij, vals, al))
    return rows


def alpha_at_origin(depths, weights_fn=None):
    """I(0)/N at each level and the alpha between levels (exact: no grid)."""
    vals = []
    for d in depths:
        P = centres(d)
        w = None if weights_fn is None else weights_fn(d)
        F = len(P) if w is None else w.sum()
        vals.append(abs(F) ** 2 / len(P))
    al = [float(np.log(vals[i + 1] / vals[i]) / np.log(8)) for i in range(len(vals) - 1)]
    return vals, al


def random_k_stats(depths, weights_fn=None, n_k=400, seed=1, kmax=np.pi):
    """Mean, median and maximum of I(k)/N over n_k fixed random 3D wave vectors in [0, kmax]^3,
    per level.  A falling median with a flat or falling mean means the power sits in sparse
    peaks (Bragg-like); a constant median means a diffuse continuum.  Returns {depth: (mean, median, max)}."""
    K = np.random.default_rng(seed).uniform(0, kmax, (n_k, 3))
    out = {}
    for d in depths:
        P = centres(d)
        w = np.ones(len(P)) if weights_fn is None else weights_fn(d)
        v = np.abs(np.exp(1j * (K @ P.T)) @ w) ** 2 / len(P)
        out[d] = (float(v.mean()), float(np.median(v)), float(v.max()))
    return out


WEIGHTS = {
    "unweighted": None,
    "notch_x": lambda d: notches(d)[:, 0].astype(float),
    "notch_z": lambda d: notches(d)[:, 2].astype(float),
    "notch_111": lambda d: notches(d).sum(axis=1) / 3.0,
}


# ------------------------------------------------------------------ tests
def test_centres_are_distinct_and_on_two_cosets():
    for d in (1, 2, 3, 4):
        P = centres(d)
        assert len(P) == 8 ** d and len({tuple(p) for p in P}) == len(P)
        assert P.min() == 1 and P.max() == 2 ** (d + 1) - 1
        par = {tuple(p % 2) for p in P}
        assert par == {(1, 1, 1), (0, 0, 0)} or (d == 0)
        assert int(np.sum(np.all(P % 2 == 0, axis=1))) == 8 ** (d - 1)


def test_notch_sum_is_two_to_the_n_times_one_one_one():
    for d in (1, 2, 3, 4, 5):
        assert tuple(notches(d).sum(axis=0)) == (2 ** d,) * 3
        assert set(map(tuple, np.unique(notches(d), axis=0))) <= {tuple(s) for s in np.ndindex(2, 2, 2)} | \
            {(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)}


def test_occupancy_fills_the_scaled_chair_exactly():
    occ, tile = occupancy(2)
    assert occ.sum() == 7 * 64 and len(np.unique(tile[occ])) == 64
    sc = np.zeros((8, 8, 8), bool)
    for c in C.CHAIR_CELLS:
        sc[4 * c[0]:4 * c[0] + 4, 4 * c[1]:4 * c[1] + 4, 4 * c[2]:4 * c[2] + 4] = True
    assert (occ == sc).all()


def test_slices_cut_chairs_into_squares_and_l_trominoes_only():
    seen = set()
    for c in range(8):
        for axis in range(3):
            lay = slice_layer(2, axis, c)
            for i in np.unique(lay[lay >= 0]):
                seen.add(int((lay == i).sum()))
    assert seen <= {3, 4}                                  # a chair layer has 4 cells (2x2) or 3 (L)
    assert seen == {3, 4}


def test_projection_theorem_3d_fft_plane_equals_2d_fft_of_projection():
    P = centres(2)
    M = 16
    vol = np.zeros((M, M, M))
    np.add.at(vol, tuple(P.T), 1.0)
    F3 = np.fft.fftn(vol)
    F2 = structure_plane(P, None, M, 2, 0.0)
    assert np.allclose(F3[:, :, 0], F2)
    kz = 2 * np.pi * 3 / M
    F2b = structure_plane(P, None, M, 2, kz)
    assert np.allclose(F3[:, :, 3], F2b)


def test_structure_factor_at_origin_counts_points_and_weights():
    P, w = centres(3), notches(3)[:, 2].astype(float)
    assert np.isclose(structure_plane(P, None, 32)[0, 0].real, 512.0)
    assert np.isclose(structure_plane(P, w, 32)[0, 0].real, 8.0)            # 2**3


def test_notch_weighted_intensity_at_origin_grows_as_alpha_two_thirds_exactly():
    vals, al = alpha_at_origin((1, 2, 3, 4, 5), lambda d: notches(d)[:, 2].astype(float))
    assert np.allclose(vals, [4 ** d / 8 ** d for d in (1, 2, 3, 4, 5)])
    assert np.allclose(al, [-1 / 3] * 4)                    # |sum w|^2 / N = 2^-n


def test_unweighted_origin_is_bragg():
    vals, al = alpha_at_origin((1, 2, 3, 4))
    assert np.allclose(vals, [8 ** d for d in (1, 2, 3, 4)]) and np.allclose(al, 1.0)


def test_interface_edges_of_layer_zero_are_four_to_the_n_plus_one_minus_four_times_two_to_the_n():
    got = [interface_edges(slice_layer(d, 2, 0)) for d in (1, 2, 3, 4)]
    assert got == [4 ** (d + 1) - 4 * 2 ** d for d in (1, 2, 3, 4)] == [8, 48, 224, 960]


def test_random_k_median_falls_every_level_so_power_sits_in_sparse_peaks():
    st = random_k_stats((2, 3, 4), None, n_k=200)
    med = [st[d][1] for d in (2, 3, 4)]
    assert med[0] > 2 * med[1] and med[1] > 2 * med[2]
    st2 = random_k_stats((2, 3, 4), WEIGHTS["notch_x"], n_k=200)
    assert st2[2][1] > st2[3][1] > st2[4][1]


def test_notch_weights_are_not_random_signs_column_sums_of_z_notch_stay_bounded():
    P, N = centres(4), notches(4)
    assert np.abs(projection(P, 2, N[:, 2].astype(float), 32)).max() <= 2
    assert np.abs(projection(P, 2, N[:, 0].astype(float), 32)).max() == 16


def run_tests(verbose=True):
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            status = "PASS"
        except Exception as exc:  # noqa: BLE001
            failed += 1
            status = "FAIL: %s: %s" % (type(exc).__name__, exc)
        if verbose:
            print("  %-74s %s" % (name, status))
    if verbose:
        print("%d test(s), %d failed" % (len(tests), failed))
    return failed == 0


# ------------------------------------------------------------------ figure
def show_slice(depth=5, save=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    for a, c, name in ((ax[0, 0], 0, "z = 0"), (ax[0, 1], 5, "z = 5")):
        lay = slice_layer(3, 2, c).astype(float)
        lay[lay < 0] = np.nan
        a.imshow((lay % 20).T, origin="lower", cmap="tab20", interpolation="nearest")
        a.set_title("depth-3 slice %s: chair cross-sections (squares / L-trominoes)" % name, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    P = centres(depth)
    M = 512
    proj = projection(P, 2, None, 2 ** (depth + 1))
    ax[0, 2].imshow(proj.T, origin="lower", cmap="viridis")
    ax[0, 2].set_title("projection of the %d chair centres along z (max column %d)" % (len(P), proj.max()), fontsize=9)
    I = np.fft.fftshift(intensity(P, None, M))
    ax[1, 0].imshow(np.maximum(I, 1e-3).T, origin="lower", norm=LogNorm(1e-3, I.max()), cmap="magma",
                    extent=(-np.pi, np.pi, -np.pi, np.pi))
    ax[1, 0].set_title("k_z = 0 plane = projection theorem, level %d (log I/N)" % depth, fontsize=9)
    ds = tuple(range(2, depth + 1))
    for name, wf in WEIGHTS.items():
        st = random_k_stats(ds, None if wf is None else wf, 300)
        ax[1, 1].semilogy(ds, [st[d][1] for d in ds], "o-", label=name + " median")
        ax[1, 1].semilogy(ds, [st[d][0] for d in ds], "x--", alpha=.5)
    ax[1, 1].set_xlabel("level n"); ax[1, 1].set_ylabel("I(k)/N at 300 random k")
    ax[1, 1].set_title("median (o) falls every level; mean (x) noisy: peaks", fontsize=9)
    ax[1, 1].legend(fontsize=7)
    n = np.arange(1, depth + 1)
    ax[1, 2].semilogy(n, 64.0 ** n, "k-", label="unweighted |F(0)|^2 = 64^n (Bragg)")
    ax[1, 2].semilogy(n, 3.0 * 4.0 ** n, "r-", label="notch-weighted |F(0)|^2 = 3*4^n")
    ax[1, 2].semilogy(n, 3.0 * 8.0 ** n, "b:", label="random signs ~ 3*8^n")
    ax[1, 2].set_xlabel("level n"); ax[1, 2].legend(fontsize=7)
    ax[1, 2].set_title("origin: notch vectors sum to 2^n (1,1,1)", fontsize=9)
    fig.suptitle("Chair44: slices, projection and diffraction of the hierarchy", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
    if show:
        plt.show()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--png")
    a = ap.parse_args(argv)
    if a.tests_only:
        return 0 if run_tests() else 1
    if a.png:
        show_slice(a.depth, a.png, show=False)
        print("wrote", a.png)
        return 0
    ds = tuple(range(2, a.depth + 1))
    print("unweighted centres, plane kz=0 (alpha: 1 Bragg, 0 diffuse)")
    for ij, vals, al in exponent_track(ds, 256):
        print("  k=%-10s I=%s alpha=%s" % (ij, ["%.3g" % v for v in vals], ["%.2f" % x for x in al]))
    print("notch-weighted (z component)")
    for ij, vals, al in exponent_track(ds, 256, lambda d: notches(d)[:, 2].astype(float)):
        print("  k=%-10s I=%s alpha=%s" % (ij, ["%.3g" % v for v in vals], ["%.2f" % x for x in al]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
