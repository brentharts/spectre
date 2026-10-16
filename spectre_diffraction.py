#!/usr/bin/python3
"""Diffraction of the Hat, Spectre and Turtle vertex sets.

Reciprocal-space version of the Hat-versus-Spectre distinction.  At the Hat
proportions a = sqrt3 b every vertex of the tiling is a site of the periodic
[3.4.6.4] Laves lattice, so the vertex set is a SUBSET OF A PERIODIC SET and its
diffraction must carry the reciprocal lattice of the torus lattice
L = Z(3,sqrt3) + Z(0,2 sqrt3).  The Spectre, a = b, is not on that lattice and
has no such obligation.  This script measures it.

Conventions.  Lengths are rescaled to unit density (rho = 6 vertices per tile
over the exact mean tile area, see spectre_energy.exact_density).  The
structure factor of a Gaussian-windowed patch is

        S(k) = | sum_j w_j exp(i k.x_j) |^2 / sum_j w_j^2 ,  w = exp(-r^2/2 s^2),

on the same unit-density window (s = 9) for every set, so peak heights are
comparable.  The smallest reciprocal vector of L, in unit-density units, is

        |b| = (2 pi / 3) / sqrt(rho_phys)     (|b_phys| = 4 pi / (sqrt3 |E1|)).

Three measurements:

  1. Bragg peaks: for the Hat and the Turtle the strong peaks sit at |k|/|b| in
     {1, sqrt3, 2, ...} (the shells of a hexagonal reciprocal lattice); the
     Spectre's strongest peak is far lower.
  2. Small k: all three are strongly hyperuniform, S(k) << 1 as k -> 0; a log-log
     minimum of the circle mean for 0.3 <= k <= 1 is reported (no exponent: the
     window leaks below ~0.35 and structure sets in above ~1).
  3. Energy: the Gaussian-kernel energy of spectre_energy.py is, exactly,
        E(alpha) + 1 = int d^2k/(2 pi)^2  (pi/alpha) exp(-k^2/4 alpha) S(k)
     for the same weights (Parseval); that identity is checked in selftest(),
     and used to split the gap above the triangular lattice into the part
     carried by small k and the part carried by the Bragg shells.

    python3 spectre_diffraction.py            # numbers + figure
    python3 spectre_diffraction.py --check    # assertions only
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectre_energy as EN               # noqa: E402

S3 = 3 ** 0.5
SIGMA = 9.0                               # window, unit-density lengths
KMAX = 9.0
DK = 0.05
FIGDIR = os.path.join(HERE, "paper", "figures")
SHELLS = [(1, "1"), (3 ** 0.5, "\\sqrt{3}"), (2, "2"), (7 ** 0.5, "\\sqrt{7}"), (3, "3")]


def b_length(a, b):
    """|b|, smallest reciprocal vector of L = Z(3,sqrt3)+Z(0,2sqrt3), in
    unit-density units for the vertex set of Tile(a,b) at its own density."""
    rho = EN.exact_density(a, b, "vertices")
    return (2 * np.pi / 3.0) / np.sqrt(rho)


def windowed_points(points, centre, rho, sigma=SIGMA):
    X = (points - centre) * np.sqrt(rho)
    m = np.hypot(*X.T) < 3.2 * sigma
    X = X[m]
    w = np.exp(-(X ** 2).sum(1) / (2 * sigma ** 2))
    return X, w


def structure_grid(X, w, kmax=KMAX, dk=DK, chunk=4000):
    ks = np.arange(-kmax, kmax + 1e-9, dk)
    KX, KY = np.meshgrid(ks, ks)
    K = np.stack([KX.ravel(), KY.ravel()], 1)
    A = np.zeros(len(K), complex)
    for i in range(0, len(K), chunk):
        ph = K[i:i + chunk] @ X.T
        A[i:i + chunk] = (np.exp(1j * ph) * w[None, :]).sum(1)
    S = (np.abs(A) ** 2 / (w ** 2).sum()).reshape(KX.shape)
    return ks, S


def radial(ks, S, edges):
    KX, KY = np.meshgrid(ks, ks)
    R = np.hypot(KX, KY)
    idx = np.digitize(R.ravel(), edges) - 1
    s = S.ravel()
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = idx == i
        if m.any():
            out[i] = s[m].mean()
    return 0.5 * (edges[1:] + edges[:-1]), out


def local_peaks(ks, S, kmin=2.0, thresh=2.0):
    """Local maxima of S on the grid (3x3), |k| >= kmin, S >= thresh."""
    KX, KY = np.meshgrid(ks, ks)
    R = np.hypot(KX, KY)
    P = S[1:-1, 1:-1]
    m = np.ones_like(P, bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                m &= P >= S[1 + dx:S.shape[0] - 1 + dx, 1 + dy:S.shape[1] - 1 + dy]
    m &= (P >= thresh) & (R[1:-1, 1:-1] >= kmin)
    ii, jj = np.where(m)
    return [(float(R[i + 1, j + 1]), float(P[i, j])) for i, j in zip(ii, jj)]


def lattice_patch(n=45):
    """The triangular lattice at unit density, for the reference S_A2."""
    a = np.sqrt(2 / S3)
    pts = [(a * (i + 0.5 * j), a * S3 / 2 * j) for i in range(-n, n + 1)
           for j in range(-n, n + 1)]
    return np.array(pts)


CACHE = os.path.join(HERE, ".cache", "spectre_diffraction.npz")


def measure(use_cache=True):
    if use_cache and os.path.isfile(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["res"].item()
    out = measure_fresh()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("X", "w")} for k, v in out.items()}
    np.savez(CACHE, res=np.array(slim, dtype=object))
    return slim


def measure_fresh():
    out = {}
    for name, a, b in EN.FAMILY:
        P = EN.Patch(a, b, 5)
        rho = EN.exact_density(a, b, "vertices")
        assert P.inradius * np.sqrt(rho) > 3.2 * SIGMA + 5, name
        X, w = windowed_points(P.vert, P.com, rho)
        ks, S = structure_grid(X, w)
        out[name] = dict(ks=ks, S=S, n=len(X), neff=float((w ** 2).sum()),
                         b=b_length(a, b), a=a, bb=b, X=X, w=w)
    L = lattice_patch()
    X, w = windowed_points(L, np.zeros(2), 1.0)
    ks, S = structure_grid(X, w)
    out["triangular"] = dict(ks=ks, S=S, n=len(X), neff=float((w ** 2).sum()), X=X, w=w)
    return out


def summarise(res):
    """Numbers used by the paper and the figure."""
    sm = {}
    edges = np.arange(0.05, KMAX, 0.05)
    for name, d in res.items():
        kc, rad = radial(d["ks"], d["S"], edges)
        d["kc"], d["rad"] = kc, rad
        if name == "triangular":
            continue
        pk = local_peaks(d["ks"], d["S"])
        pk.sort(key=lambda t: -t[1])
        top = pk[:6]
        d["peaks"] = top
        # the hyperuniform floor: the minimum of the circle mean for 0.3 <= k <= 1.0
        # (no exponent is quoted: below ~0.35 the window leaks, above ~1 structure sets in)
        m = (kc >= 0.3) & (kc <= 1.0)
        i0 = int(np.nanargmin(np.where(m, rad, np.nan)))
        d["floor"], d["floor_k"] = float(rad[i0]), float(kc[i0])
        frac, th, base, npk = lattice_match(res, name, d["b"])
        d["match"] = dict(frac=frac, theta=float(np.degrees(th)), base=base, npeaks=npk)
        sm[name] = dict(top=top, smax=top[0][1] if top else 0.0, floor=d["floor"],
                        floor_k=d["floor_k"], b=d["b"], match=d["match"])
    return sm


def recip_points(b_len, theta, kmax=KMAX, n=8):
    """Points of a hexagonal reciprocal lattice with smallest vector b_len,
    rotated by theta (radians).  The Hat's lattice has a vector at angle 0 or
    30 degrees mod 60, found numerically, not assumed."""
    B = b_len * np.array([[1.0, 0.0], [-0.5, S3 / 2]])
    c, s_ = np.cos(theta), np.sin(theta)
    Rm = np.array([[c, -s_], [s_, c]])
    pts = []
    for h in range(-n, n + 1):
        for j in range(-n, n + 1):
            q = Rm @ (h * B[0] + j * B[1])
            if 0 < np.hypot(*q) < kmax:
                pts.append(q)
    return np.array(pts)


def on_lattice_fraction(ks, S, b_len, theta, tol=0.12, thresh=30.0, kmin=1.5):
    """Share of the Bragg intensity (local maxima with S >= thresh, |k| >= kmin)
    that lies within `tol` of a point of the reciprocal lattice."""
    KX, KY = np.meshgrid(ks, ks)
    P = S[1:-1, 1:-1]
    m = np.ones_like(P, bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                m &= P >= S[1 + dx:S.shape[0] - 1 + dx, 1 + dy:S.shape[1] - 1 + dy]
    R = np.hypot(KX, KY)[1:-1, 1:-1]
    m &= (P >= thresh) & (R >= kmin)
    ii, jj = np.where(m)
    pk = np.array([[KX[i + 1, j + 1], KY[i + 1, j + 1]] for i, j in zip(ii, jj)])
    w = np.array([P[i, j] for i, j in zip(ii, jj)])
    if len(w) == 0:
        return 0.0, 0
    L = recip_points(b_len, theta)
    d = np.min(np.hypot(pk[:, None, 0] - L[None, :, 0], pk[:, None, 1] - L[None, :, 1]), axis=1)
    return float(w[d < tol].sum() / w.sum()), len(w)


def lattice_match(res, name, b_len):
    """Best orientation (0 or 30 degrees) and its on-lattice fraction at scale 1,
    and the control baseline: the best fraction over lattices rotated off those
    orientations (see below)."""
    d = res[name]
    best = max(((on_lattice_fraction(d["ks"], d["S"], b_len, t)[0], t)
                for t in (0.0, np.pi / 6)), key=lambda x: x[0])
    # control: the same lattice rotated off the two symmetric orientations
    # (by 3 to 27 degrees, and its mirror images mod 60), at three scales
    base = 0.0
    for sc in (0.9, 1.0, 1.1):
        for deg in list(np.arange(3, 28, 1.5)) + list(np.arange(33, 58, 1.5)):
            base = max(base, on_lattice_fraction(d["ks"], d["S"], b_len * sc,
                                                 np.radians(deg))[0])
    return best[0], best[1], base, on_lattice_fraction(d["ks"], d["S"], b_len, best[1])[1]


def shell_ratio(k, b):
    """|k|/|b| and the nearest hexagonal shell sqrt(h^2+hk+k^2)."""
    r = k / b
    shells = sorted({np.sqrt(h * h + h * j + j * j) for h in range(0, 6) for j in range(0, 6)} - {0.0})
    s = min(shells, key=lambda t: abs(t - r))
    return r, s


def energy_split(res, name, kc, alphas=EN.ALPHAS):
    """(total, low-k part, high-k part) of  int fhat (S - S_A2), for alphas <= 1,
    on the common grid; kc splits the integral in |k|."""
    d, t = res[name], res["triangular"]
    ks = d["ks"]
    KX, KY = np.meshgrid(ks, ks)
    R2 = KX ** 2 + KY ** 2
    dS = d["S"] - t["S"]
    tot, low, high = [], [], []
    for al in alphas:
        fh = (np.pi / al) * np.exp(-R2 / (4 * al)) * DK * DK / (2 * np.pi) ** 2
        tot.append((fh * dS).sum())
        low.append((fh * dS)[R2 < kc ** 2].sum())
        high.append((fh * dS)[R2 >= kc ** 2].sum())
    return np.array(tot), np.array(low), np.array(high)


def selftest():
    # Parseval: the k-integral of fhat * S_w equals the weighted real-space sum
    rng = np.random.RandomState(0)
    X = rng.uniform(-6, 6, (120, 2))
    w = np.exp(-(X ** 2).sum(1) / (2 * 3.0 ** 2))
    ks, S = structure_grid(X, w, kmax=12.0, dk=0.1)
    al = 0.5
    KX, KY = np.meshgrid(ks, ks)
    fh = (np.pi / al) * np.exp(-(KX ** 2 + KY ** 2) / (4 * al))
    lhs = (fh * S).sum() * 0.1 ** 2 / (2 * np.pi) ** 2
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(2)
    rhs = (w[:, None] * w[None, :] * np.exp(-al * d2)).sum() / (w ** 2).sum()
    assert abs(lhs / rhs - 1) < 2e-3, (lhs, rhs)
    # |b|: smallest reciprocal vector of L, from the basis directly
    E1, E2 = np.array([3.0, S3]), np.array([0.0, 2 * S3])
    B = np.linalg.inv(np.array([E1, E2])).T * 2 * np.pi
    assert abs(np.hypot(*B[0]) - 2 * np.pi / 3) < 1e-12
    assert abs(np.hypot(*B[0]) - np.hypot(*B[1])) < 1e-12
    assert abs(np.dot(B[0], B[1]) / np.dot(B[0], B[0]) + 0.5) < 1e-12   # 120 degrees
    # the triangular lattice has its first Bragg shell at 4 pi / (sqrt3 a)
    L = lattice_patch(30)
    X, w = windowed_points(L, np.zeros(2), 1.0)
    ks, S = structure_grid(X, w, kmax=8.0, dk=0.05)
    a = np.sqrt(2 / S3)
    kb = 4 * np.pi / (S3 * a)
    pk = [p for p in local_peaks(ks, S, kmin=2.0, thresh=50.0)]
    assert pk and abs(min(pk, key=lambda t: abs(t[0] - kb))[0] - kb) < 0.06
    return True


def figure(res, sm, fname="spectre_diffraction"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    BLUE, ORANGE, INK, GREY = "#2a78d6", "#eb6834", "#222222", "#8a877f"
    COL = {"Hat": BLUE, "Spectre": INK, "Turtle": ORANGE}
    fig = plt.figure(figsize=(13.0, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)
    kmax_plot = 8.0
    for i, name in enumerate(("Hat", "Spectre", "Turtle")):
        ax = fig.add_subplot(gs[0, i])
        d = res[name]
        ks, S = d["ks"], d["S"]
        m = np.abs(ks) <= kmax_plot
        ax.imshow(S[np.ix_(m, m)], origin="lower", cmap="Greys",
                  norm=LogNorm(vmin=0.3, vmax=max(float(S.max()) * 0.5, 50)),
                  extent=[-kmax_plot, kmax_plot, -kmax_plot, kmax_plot])
        if name != "Spectre":
            for q in recip_points(d["b"], np.radians(d["match"]["theta"]), kmax=kmax_plot):
                ax.plot([q[0]], [q[1]], "o", mfc="none", mec=ORANGE if name == "Hat" else BLUE,
                        ms=7, mew=0.8)
        ax.set_title("(%s) %s: $S(\\mathbf{k})$%s" % ("abc"[i], name,
                     ", circles: lattice of $\\Lambda^{*}$ (%d$^\\circ$)" % round(d["match"]["theta"]) if name != "Spectre" else ", off any such lattice"),
                     fontsize=9.5)
        ax.set_xlabel("$k_x$")
        if i == 0:
            ax.set_ylabel("$k_y$")
    # (d) radial S(k)
    ax = fig.add_subplot(gs[1, 0])
    for name in ("Hat", "Spectre", "Turtle"):
        d = res[name]
        ax.plot(d["kc"], d["rad"], color=COL[name], lw=1.3, label=name)
    ax.plot(res["triangular"]["kc"], res["triangular"]["rad"], color=GREY, lw=0.9,
            ls=(0, (4, 2)), label="triangular")
    for mult, lab in SHELLS:
        for name, c in (("Hat", BLUE),):
            x = mult * res["Hat"]["b"]
            if x < KMAX:
                ax.axvline(x, color=BLUE, lw=0.6, ls=":", zorder=0)
                ax.text(x, 0.03, "$%s|b|$" % lab, rotation=90, fontsize=7, color=BLUE,
                        va="bottom", ha="right", transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xlim(0, KMAX)
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("$|k|$ (unit density)")
    ax.set_ylabel("$\\langle S\\rangle$ over the circle")
    ax.set_title("(d) azimuthal mean; dotted: $|b|\\{1,\\sqrt{3},2,\\sqrt{7},3\\}$ of the Hat", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, bbox_to_anchor=(0.12, 1.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # (e) small k
    ax = fig.add_subplot(gs[1, 1])
    for name in ("Hat", "Spectre", "Turtle"):
        d = res[name]
        m = (d["kc"] > 0.1) & (d["kc"] < 3.0)
        ax.loglog(d["kc"][m], d["rad"][m], color=COL[name], lw=1.4,
                  label="%s: floor %.3f at $k=%.2f$" % (name, d["floor"], d["floor_k"]))
    t = res["triangular"]
    m = (t["kc"] > 0.1) & (t["kc"] < 3.0)
    ax.loglog(t["kc"][m], np.maximum(t["rad"][m], 1e-4), color=GREY, lw=0.9, ls=(0, (4, 2)),
              label="triangular (window leakage)")
    ax.axhline(1.0, color=GREY, lw=0.6)
    ax.text(0.37, 1.3, "Poisson, $S=1$", color=GREY, fontsize=8)
    ax.axvspan(0.1, 0.35, color="#eeeeee", zorder=0)
    ax.set_ylim(1e-4, 1e3)
    ax.set_xlabel("$|k|$")
    ax.set_ylabel("$\\langle S\\rangle$")
    ax.set_title("(e) small $k$: $S\\lesssim0.02$ at $k\\approx0.4$--$1$ (grey: window leakage)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.0, loc="upper right")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    # (f) the energy split
    ax = fig.add_subplot(gs[1, 2])
    al = EN.ALPHAS
    sel = al <= 1.0
    E_tri = EN.controls()["triangular"]
    for name, ls in (("Hat", "-"), ("Spectre", "-")):
        kc = 0.9 * res[name]["b"] if name == "Hat" else 0.9 * res["Hat"]["b"]
        tot, low, high = energy_split(res, name, kc)
        base = E_tri
        ax.plot(al[sel], (tot / base)[sel], color=COL[name], lw=1.8, label="%s: total" % name)
        ax.plot(al[sel], (low / base)[sel], color=COL[name], lw=1.0, ls=(0, (4, 2)),
                label="%s: $|k|<%.1f$" % (name, kc))
        ax.plot(al[sel], (high / base)[sel], color=COL[name], lw=1.0, ls=":",
                label="%s: $|k|\\geq%.1f$" % (name, kc))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("kernel width $\\alpha$")
    ax.set_ylabel("gap above $A_2$ from $S(\\mathbf{k})$")
    ax.set_title("(f) the energy gap, split in reciprocal space", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, fname + "." + ext), dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figures/%s.{pdf,png}" % fname)


def report(res, sm):
    for name, d in sm.items():
        m = d["match"]
        print("  %-8s |b|=%.3f  S floor %.4f at k=%.2f   on-lattice fraction %.2f (%.0f deg, %d peaks), "
              "wrong-scale baseline %.2f" % (name, d["b"], d["floor"], d["floor_k"], m["frac"],
                                           m["theta"], m["npeaks"], m["base"]))
        seen = []
        for k, s_ in sorted(d["top"], key=lambda t: -t[1]):
            if any(abs(k - q) < 0.1 for q in seen):
                continue
            seen.append(k)
            r, sh = shell_ratio(k, d["b"])
            print("      k=%.3f  S=%7.2f   k/|b|=%.3f" % (k, s_, r))


if __name__ == "__main__":
    print("selftest:", selftest())
    if "--check" in sys.argv:
        sys.exit(0)
    res = measure()
    sm = summarise(res)
    report(res, sm)
    figure(res, sm)
