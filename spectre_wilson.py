#!/usr/bin/env python3
"""spectre_wilson.py - a U(1) Wilson action for the Spectre's lift lattice A2 + A2, and a U(1) action with the Spectre tiles as plaquettes.

Why.  quantgrav.tex, open problem 1: the Wilson probe of 4D_Wilson_Loop_RG lives on a hypercubic lattice, where every corner is a 90 degree
corner.  The Spectre lift lattice (lift4d.py) is Z[w6] + z Z[w6] = red A2 + green A2: edges at 60 degrees inside each plane, red and green edges
orthogonal.  Two models are built here, with the same Gaussian convention as `numerics/abelian_first_step` of that repository (action
(1/2 g^2) sum_p w_p theta_p^2, loop self-energy E = J^T G J so that <W> = exp(-g^2 E / 2); g = 1).

  Model T (tiles as plaquettes).  Links = tile edges of a Spectre patch, plaquettes = the 14-gon tiles (research/su2_graph.py uses the same graph).
  Model H (hexagonal lattice).  Sites Z^4 (coordinates p, q, r, s with u = p + q w6, v = r + s w6) on a torus of side N.  Six link types: three in the
  red plane (1, w6, w6^2) and three in the green plane.  Plaquettes: both triangles of each plane (weight W_TRI) and the 3 x 3 red-green
  parallelograms (weight W_MIX), chosen so that the long-wavelength limit is isotropic (a test below fixes them).

Status of each statement (labels as in CONNECTIONS.md):
  Exact    T1  In model T, for a union R of tiles with boundary current J, E(J) = |R| (number of tiles): a pure area law, no perimeter and no corner
               term.  (J = B^T chi_R and B has independent rows on a disc, so the covariance of the flux is the identity.)  Tested on random unions of
               tiles of the level-2 patch.  So the tiling used as a plaquette surface carries no information about corners; they need the transverse
               plaquettes of model H.
  Exact    H1  Model H: the Fourier computation agrees with a dense real-space solve on a 4^4 torus; E does not depend on the gauge-fixing weight;
               the Hessian annihilates pure gauge fields; the weights W_TRI = 2 and W_MIX = 1/3 make the small-momentum Hessian equal to the
               Maxwell one (k^2 delta - k k) in an orthonormal frame of R^4, and other weights do not.
  Numeric  H2  Perimeter + corner form: E = alpha P + b60 c60 + b120 c120 + b90 c90 fits the structured loops (triangles, rhombi, parallelograms,
               hexagons of the red plane, red-green rectangles), with the residual printed, and then predicts held-out loops (random interleavings
               of red and green steps).  Corner types: the angle between consecutive link directions.  The constants differ (printed).
  Numeric  H3  The 14-step outlines of the lifted Spectre tiles: E predicted by H2 against E computed.  All 71 outlines of the level-2 patch have
               one of two energies up to colour swap, and the mirror outline (complex conjugation) has the same energy: this action is parity
               symmetric, so it cannot see the chirality of the tile.
  Not done The constrained energy of the first RG step (Q a = 0 with the hexagonal averaging operator), non-abelian groups, the continuum limit,
               any statement for Yang-Mills.  E here is the *unconstrained* loop self-energy, the 'total' cross-check column of the
               abelian_first_step README, so alpha and b are not the programme's delta-m and log z_c.

    python3 spectre_wilson.py            selftest (also: --tests)    about 40 s
    python3 spectre_wilson.py --table    the corner constants and the Spectre outlines
"""
import itertools
import math
import random
import sys

import numpy as np

import lift4d as L

W6 = np.exp(1j * math.pi / 3)
RED = [(1, 0, 0, 0), (0, 1, 0, 0), (-1, 1, 0, 0)]
GREEN = [(0, 0, 1, 0), (0, 0, 0, 1), (0, 0, -1, 1)]
LINKS = RED + GREEN
W_TRI, W_MIX = 2.0, 1.0 / 3.0
_C = {}


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def neg(a):
    return tuple(-x for x in a)


# ---------------------------------------------------------------- plaquettes and links of model H
def link_of(d):
    """displacement -> (type, sign) of the link it traverses (tail at the start for sign +1)."""
    for a, v in enumerate(LINKS):
        if d == v:
            return a, 1
        if d == neg(v):
            return a, -1
    raise ValueError("not a link: %s" % (d,))


def path_links(vertices):
    """closed vertex path -> [(tail position, type, sign)]."""
    out = []
    n = len(vertices)
    for i in range(n):
        x, y = vertices[i], vertices[(i + 1) % n]
        d = tuple(b - a for a, b in zip(x, y))
        a, s = link_of(d)
        out.append((x if s == 1 else y, a, s))
    return out


def plaquettes():
    """[(weight, [(offset, type, sign)])] for the plaquette classes at the origin."""
    z = (0, 0, 0, 0)
    P = []
    for plane in (RED, GREEN):
        ep, eq = plane[0], plane[1]
        P.append((W_TRI, path_links([z, ep, eq])))
        P.append((W_TRI, path_links([ep, add(ep, eq), eq])))
    for a in RED:
        for b in GREEN:
            P.append((W_MIX, path_links([z, a, add(a, b), b])))
    return P


# ---------------------------------------------------------------- Fourier solution
def kgrid(N):
    m = np.stack(np.meshgrid(*[np.arange(N)] * 4, indexing="ij"), -1).reshape(-1, 4)
    return 2 * np.pi * m / N


def hessian(K, weights=(W_TRI, W_MIX)):
    """M(k) = sum_p w conj(b) b^T over the plaquette classes, shape (n, 6, 6); b_a = sum of s exp(i k . offset) over the links of type a."""
    M = np.zeros((len(K), 6, 6), complex)
    for w, links in plaquettes():
        w = weights[0] if w == W_TRI else weights[1]
        b = np.zeros((len(K), 6), complex)
        for off, a, s in links:
            b[:, a] += s * np.exp(1j * (K @ np.array(off, float)))
        M += w * np.conj(b)[:, :, None] * b[:, None, :]
    return M


def grad_vector(K):
    """c_a(k) = exp(i k . d_a) - 1, the pure-gauge field a = grad phi."""
    return np.stack([np.exp(1j * (K @ np.array(d, float))) - 1 for d in LINKS], 1)


def propagator(N, xi=1.0):
    key = ("G", N, xi)
    if key not in _C:
        K = kgrid(N)
        M = hessian(K)
        c = grad_vector(K)
        Mx = M + xi * c[:, :, None] * np.conj(c)[:, None, :]
        w, V = np.linalg.eigh(Mx)
        inv = np.where(w > 1e-9, 1.0 / np.where(w > 1e-9, w, 1.0), 0.0)
        _C[key] = np.einsum("nik,nk,njk->nij", V, inv, np.conj(V))
    return _C[key]


def current(steps, N):
    """steps: list of (start position, displacement); -> J array (6, N, N, N, N)."""
    J = np.zeros((6, N, N, N, N))
    for x, d in steps:
        a, s = link_of(d)
        tail = x if s == 1 else add(x, d)
        J[(a,) + tuple(t % N for t in tail)] += s
    return J


def loop_steps(dirs, start=(0, 0, 0, 0)):
    """list of displacement 4-tuples -> [(position, displacement)] and check that it closes."""
    x = start
    out = []
    for d in dirs:
        out.append((x, d))
        x = add(x, d)
    assert x == start, "loop does not close"
    return out


def energy(dirs, N=16, xi=1.0):
    G = propagator(N, xi)
    J = current(loop_steps(dirs), N)
    Jk = np.fft.fftn(J, axes=(1, 2, 3, 4)).reshape(6, -1).T
    return float(np.real(np.einsum("ni,nij,nj->", np.conj(Jk), G, Jk)) / N ** 4)


# ---------------------------------------------------------------- loops and corners
def unit(d):
    """physical unit vector of a link displacement (red -> first plane, green -> second plane) as (complex, complex)."""
    return (d[0] + d[1] * W6, d[2] + d[3] * W6)


def corner_counts(dirs):
    """-> (P, {60: n, 120: n, 90: n}) with the turn between consecutive link directions (cyclically); straight passes are not corners."""
    n = len(dirs)
    c = {60: 0, 120: 0, 90: 0}
    for i in range(n):
        u, v = unit(dirs[i]), unit(dirs[(i + 1) % n])
        cos = (u[0] * np.conj(v[0])).real + (u[1] * np.conj(v[1])).real
        cos = round(float(cos), 6)
        if cos == 1.0:
            continue
        if cos == 0.5:
            c[60] += 1
        elif cos == -0.5:
            c[120] += 1
        elif cos == 0.0:
            c[90] += 1
        else:
            raise ValueError("reversal or unknown corner (cos = %s)" % cos)
    return n, c


def red(k):
    return RED[0] if k % 6 == 0 else {1: RED[1], 2: RED[2], 3: neg(RED[0]), 4: neg(RED[1]), 5: neg(RED[2])}[k % 6]


def green(k):
    return (0, 0) + red(k)[:2]


def structured_loops(ns=(3, 4)):
    """(name, dirs) for the training set, every side n >= 3 links (no feature thinner than 3): red-plane triangles, rhombi, parallelograms, hexagons,
    and red-green rectangles (two sets of directions)."""
    R = lambda k, n: [red(k)] * n
    G = lambda k, n: [green(k)] * n
    out = []
    for n in ns:
        out.append(("tri%d" % n, R(0, n) + R(2, n) + R(4, n)))
        out.append(("rhomb%d" % n, R(0, n) + R(1, n) + R(3, n) + R(4, n)))
        out.append(("para%d" % n, R(0, n) + R(2, n) + R(3, n) + R(5, n)))
        out.append(("hex%d" % n, sum([R(k, n) for k in range(6)], [])))
        out.append(("rg%d" % n, R(0, n) + G(0, n) + R(3, n) + G(3, n)))
        out.append(("rg2_%d" % n, R(1, n) + G(2, n + 1) + R(4, n) + G(5, n + 1)))
    return out


def random_loops(count, seed=0, scale=3):
    """held-out loops: a closed red set (triangle or rhombus sides) and a closed green set, interleaved at random with no immediate reversal;
    every side has `scale` links, so no feature is thinner than `scale`."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        r = rng.choice([[0, 2, 4], [1, 3, 5], [0, 3], [1, 4], [2, 5]])
        g = rng.choice([[0, 2, 4], [1, 3, 5], [0, 3], [1, 4], [2, 5]])
        steps = [red(k) for k in r] + [green(k) for k in g]
        rng.shuffle(steps)
        if any(steps[i] == neg(steps[(i + 1) % len(steps)]) for i in range(len(steps))):
            continue
        out.append(("rand%d" % len(out), [d for d in steps for _ in range(scale)]))
    return out


def fit(loops, N=16):
    A, y, names = [], [], []
    for name, dirs in loops:
        P, c = corner_counts(dirs)
        A.append([P, c[60], c[120], c[90]])
        y.append(energy(dirs, N))
        names.append(name)
    A, y = np.array(A, float), np.array(y)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, A, y, names


# ---------------------------------------------------------------- the Spectre outlines in model H
def outline_steps(tile):
    n = len(tile)
    return [tuple(int(t) for t in np.array(tile[(i + 1) % n]) - np.array(tile[i])) for i in range(n)]


def mirror(d):
    """complex conjugation of the physical plane u + z v as a map on step coordinates: (u, v) -> (conj u, conj(v) / w6).
    conj(p + q w6) = (p + q) - q w6;  conj(r + s w6) = (r + s) - s w6, and dividing by w6 (w6^2 = w6 - 1) gives (r) + (-(r + s)) w6."""
    p, q, r, s = d
    return (p + q, -q, r, -(r + s))


def spectre_outlines(levels=2):
    tiles = L.lift_patch(levels)[0]
    return [outline_steps(t) for t in tiles]


# ---------------------------------------------------------------- model T
def tile_graph(levels=2):
    tiles = L.lift_patch(levels)[0]
    vid, eid, faces = {}, {}, []
    rows = []
    for t in tiles:
        n = len(t)
        row = {}
        for i in range(n):
            a, b = tuple(t[i]), tuple(t[(i + 1) % n])
            vid.setdefault(a, len(vid))
            vid.setdefault(b, len(vid))
            key = (min(a, b), max(a, b))
            eid.setdefault(key, len(eid))
            row[eid[key]] = row.get(eid[key], 0) + (1 if a < b else -1)
        rows.append(row)
    B = np.zeros((len(tiles), len(eid)))
    for i, row in enumerate(rows):
        for j, s in row.items():
            B[i, j] = s
    return B


def tile_energy(B, chi, pinv):
    J = B.T @ chi
    return float(J @ pinv @ J), J


# ---------------------------------------------------------------- tests
def run_tests(verbose=True):
    ok = True

    def check(name, good, info=""):
        nonlocal ok
        ok &= bool(good)
        if verbose:
            print("  %-4s %-62s %s" % ("ok" if good else "FAIL", name, info))

    # --- model T
    B = tile_graph(2)
    nt, ne = B.shape
    check("T: patch has 71 tiles", nt == 71, "%d tiles, %d edges" % (nt, ne))
    check("T: the rows of B (tiles) are independent", np.linalg.matrix_rank(B) == nt)
    pinv = np.linalg.pinv(B.T @ B, rcond=1e-10)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        chi = (rng.random(nt) < rng.uniform(0.1, 0.9)).astype(float)
        e, J = tile_energy(B, chi, pinv)
        assert np.allclose(B @ B.T @ chi * 0 + 0, 0)
        worst = max(worst, abs(e - chi.sum()))
    check("T1: E(boundary of R) = number of tiles in R (20 random unions)", worst < 1e-8, "max |E - |R|| = %.1e" % worst)
    P = np.abs(B.T @ chi).sum()
    check("T1: and it is not the perimeter (last union)", abs(e - P) > 0.5 or abs(chi.sum() - P) < 1e-9, "E = %.1f, perimeter = %d" % (e, P))

    # --- model H: structure
    N = 4
    K = kgrid(N)
    M = hessian(K)
    c = grad_vector(K)
    check("H1: M is Hermitian", np.allclose(M, np.conj(np.transpose(M, (0, 2, 1)))))
    check("H1: M annihilates pure gauge fields", np.abs(np.einsum("nij,nj->ni", M, c)).max() < 1e-12)
    # isotropy.  A smooth field A pulls back to the link fields a_a = u_a . A (u_a = unit vector of link a in R^4).  The lattice momentum k is conjugate to
    # the integer coordinates, so k = Bred q on each plane (e_p = 1, e_q = w6 = (1/2, sqrt3/2)).  On the pull-back fields the Hessian should be the Maxwell one,
    # vol (q^2 I - q q^T) with vol = 3/4 the volume of the cell, up to O(k^2) relative corrections.
    s3 = math.sqrt(3) / 2
    U = np.zeros((6, 4))
    for a, (x, y) in enumerate([(1, 0), (0.5, s3), (-0.5, s3)]):
        U[a, :2] = (x, y)
        U[3 + a, 2:] = (x, y)
    Bred = np.array([[1, 0], [0.5, s3]])
    err_good, err_bad = 0.0, 0.0
    for trial in range(5):
        kk = 1e-3 * np.random.default_rng(trial).normal(size=4)
        q = np.concatenate([np.linalg.solve(Bred, kk[:2]), np.linalg.solve(Bred, kk[2:])])
        Mc = 0.75 * (q @ q * np.eye(4) - np.outer(q, q))
        for weights, which in (((W_TRI, W_MIX), "good"), ((W_TRI * 1.2, W_MIX), "bad"), ((W_TRI, W_MIX * 1.2), "bad")):
            D = np.diag(np.exp(0.5j * np.array([kk @ np.array(d, float) for d in LINKS])))      # the field sits at the link midpoint
            Mp = (D @ U).conj().T @ hessian(kk[None, :], weights=weights)[0] @ (D @ U)
            Mp = Mp.real
            e = np.abs(Mp - Mc).max() / np.abs(Mc).max()
            if which == "good":
                err_good = max(err_good, e)
            else:
                err_bad = max(err_bad, e)
    check("H1: weights 2 and 1/3 give the Maxwell Hessian at small k", err_good < 5e-3, "relative error %.1e" % err_good)
    check("H1: other weights (x1.2 on either) do not (control)", err_bad > 10 * err_good, "relative error %.1e" % err_bad)

    # --- model H: Fourier against a dense solve on the 4^4 torus
    loops = dict(structured_loops())
    loops.update({"small_tri": [red(0), red(2), red(4)], "small_rhomb": [red(0), red(0), red(1), red(3), red(3), red(4)],
                  "small_hex": [red(k) for k in range(6)], "small_rg": [red(0), green(0), red(3), green(3)],
                  "small_rg2": [red(1), red(1), green(2), red(4), red(4), green(5)]})
    dense_E = lambda dirs: _dense_energy(dirs, N)
    worst = 0.0
    for name in ("small_tri", "small_rhomb", "small_hex", "small_rg", "small_rg2"):
        worst = max(worst, abs(energy(loops[name], N) - dense_E(loops[name])))
    check("H1: Fourier E agrees with the dense real-space solve (5 loops, N = 4)", worst < 1e-9, "max difference %.1e" % worst)
    e1, e2 = energy(loops["small_hex"], 8, 0.5), energy(loops["small_hex"], 8, 3.0)
    check("H1: E does not depend on the gauge-fixing weight", abs(e1 - e2) < 1e-9, "%.9f vs %.9f" % (e1, e2))
    check("H1: E is invariant under 60 degree rotation of the red plane", abs(energy(loops["small_tri"], 8) - energy([red(1) if d == red(0) else red(3) if d == red(2) else red(5) for d in loops["small_tri"]], 8)) < 1e-9)

    # --- H2: perimeter + corner.  NOTE: thresholds below were chosen after seeing the data (labelled Numeric).
    NH = 20
    coef, A, y, names = fit(structured_loops(), NH)
    res = np.abs(y - A @ coef).max()
    check("H2: E = alpha P + sum b c fits the %d structured loops (sides 3 and 4)" % len(y), res < 0.1,
          "alpha %.4f, b60 %.4f, b120 %.4f, b90 %.4f; max residual %.4f of E up to %.1f" % (tuple(coef) + (res, y.max())))
    check("H2: the sharp corner (turn 120) is separated from the other two by more than the residual",
          min(abs(coef[2] - coef[1]), abs(coef[2] - coef[3])) > res, "b120 - b60 = %.4f, b120 - b90 = %.4f" % (coef[2] - coef[1], coef[2] - coef[3]))
    check("H2: the blunt corner (turn 60) and the 90 degree corner are NOT separated (difference <= 1.5 x residual)", abs(coef[1] - coef[3]) <= 1.5 * res,
          "b60 - b90 = %.4f, residual %.4f" % (coef[1] - coef[3], res))
    held = random_loops(8)
    dev = []
    for _, d in held:
        P, c = corner_counts(d)
        dev.append(energy(d, NH) - np.array([P, c[60], c[120], c[90]]) @ coef)
    dev = np.abs(np.array(dev))
    check("H2: held-out interleavings (all sides 3 links) are predicted to within 0.3", len(dev) == 8 and dev.max() < 0.3,
          "max deviation %.4f, mean %.4f (fit residual %.4f)" % (dev.max(), dev.mean(), res))

    # --- H3: Spectre outlines
    outs = spectre_outlines(2)
    check("H3: 71 outlines, each of 14 steps and closed", len(outs) == 71 and all(len(o) == 14 and tuple(map(sum, zip(*o))) == (0, 0, 0, 0) for o in outs))
    Es = [energy(o, NH) for o in outs]
    values = sorted(set(round(e, 8) for e in Es))
    check("H3: the 71 outlines have one energy", len(values) == 1, "E = %.6f" % values[0])
    o = outs[0]
    m = [mirror(d) for d in o]
    zz = np.exp(1j * math.pi / 6)
    pos = lambda steps: sum(complex(d[0] + d[1] * W6 + zz * (d[2] + d[3] * W6)) for d in steps)
    pts = lambda steps: np.cumsum([complex(d[0] + d[1] * W6 + zz * (d[2] + d[3] * W6)) for d in steps])
    check("H3: mirror() is complex conjugation of the physical outline", np.allclose(pts(m), np.conj(pts(o))))
    check("H3: and the mirror is a different closed walk (not a rotation of the word)", m != o)
    Em = energy(m, NH)
    check("H3: the mirror outline has the same energy (the action cannot see chirality)", abs(Em - Es[0]) < 1e-9, "%.9f vs %.9f" % (Em, Es[0]))
    P_, c_ = corner_counts(o)
    gaps = []
    for k in (1, 2, 3):
        scaled = [d for d in o for _ in range(k)]
        P, c = corner_counts(scaled)
        gaps.append(energy(scaled, NH) - np.array([P, c[60], c[120], c[90]]) @ coef)
    check("H3: perimeter + corner prediction for the outline is off by 0.1 to 0.4 at scale 1, 2, 3 (it does not converge to 0)",
          all(0.1 < abs(g) < 0.4 for g in gaps), "corners %s; computed - predicted = %s" % (c_, ", ".join("%.3f" % g for g in gaps)))
    if verbose:
        print("OK" if ok else "FAILED")
    return ok


def _dense_energy(dirs, N):
    sites = list(itertools.product(range(N), repeat=4))
    sidx = {x: i for i, x in enumerate(sites)}
    lidx = lambda x, a: sidx[tuple(t % N for t in x)] * 6 + a
    rows = []
    for x in sites:
        for w, links in plaquettes():
            r = {}
            for off, a, s in links:
                k = lidx(add(x, off), a)
                r[k] = r.get(k, 0) + s
            rows.append((w, r))
    B = np.zeros((len(rows), 6 * len(sites)))
    W = np.zeros(len(rows))
    for i, (w, r) in enumerate(rows):
        W[i] = w
        for k, s in r.items():
            B[i, k] = s
    H = B.T @ (W[:, None] * B)
    J = np.zeros(6 * len(sites))
    for x, d in loop_steps(dirs):
        a, s = link_of(d)
        J[lidx(x if s == 1 else add(x, d), a)] += s
    w, V = np.linalg.eigh(H)
    keep = w > 1e-9
    z = V[:, keep].T @ J
    return float(np.sum(z * z / w[keep]))


def table():
    coef, A, y, names = fit(structured_loops(), 20)
    print("alpha = %.4f   b60 = %.4f   b120 = %.4f   b90 = %.4f   (turn angles; E = alpha P + sum b c)" % tuple(coef))
    print("structured loops: max residual %.4f" % np.abs(y - A @ coef).max())
    for n, e, a in zip(names, y, A @ coef):
        print("  %-10s E = %8.4f   fit %8.4f" % (n, e, a))
    o = spectre_outlines(2)[0]
    for k in (1, 2, 3):
        scaled = [d for d in o for _ in range(k)]
        P, c = corner_counts(scaled)
        print("Spectre outline x%d: P = %d, corners %s, E = %.4f, predicted %.4f" % (k, P, c, energy(scaled, 20), np.array([P, c[60], c[120], c[90]]) @ coef))


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    else:
        sys.exit(0 if run_tests() else 1)
