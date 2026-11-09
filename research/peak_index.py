"""Γ8 peak indexing: are the Spectre Bragg peaks in the Fourier module of the Z[zeta12] model set?

Vertices are x = sum_j n_j zeta^j (j = 0..3, zeta = exp(i pi/6), edge length 1), with internal image
x* = sum_j n_j zeta^(5j).  A Bragg peak k of the model set lifts to (k, k*) with
    k . zeta^j + k* . zeta^(5j) = 2 pi m_j,   m in Z^4,   j = 0..3.
For each measured k we search m in a box, solve for k* by least squares, and report the residual
in units of the FFT bin dk.  A peak is indexed if the residual is < 1 bin.  Small |k*| = strong peak.

Pure Python (no numpy), so it reads the committed diffraction JSON directly.
Usage: python research/peak_index.py DIFF.json OUT.json [set ...]
"""
import sys, json, math, cmath, itertools

Z = [cmath.exp(1j * math.pi / 6 * j) for j in range(4)]
ZS = [cmath.exp(1j * 5 * math.pi / 6 * j) for j in range(4)]
import os
BOX = int(os.environ.get("PI_BOX", 8))
SCALE = float(os.environ.get("PI_SCALE", 1))   # positions = sum n_j zeta^j / SCALE


def dot(a, b):
    return a.real * b.real + a.imag * b.imag


def index(k, dk):
    c = [dot(k, z) for z in Z]
    A = [(z.real, z.imag) for z in ZS]           # rows of the 4x2 system A k* = 2 pi m - c
    a11 = sum(x * x for x, _ in A); a12 = sum(x * y for x, y in A); a22 = sum(y * y for _, y in A)
    det = a11 * a22 - a12 * a12
    # pre-round: m must be near (c + A k*)/2pi; scan m directly
    best = None
    for m in itertools.product(range(-BOX, BOX + 1), repeat=4):
        b = [2 * math.pi * SCALE * m[j] - c[j] for j in range(4)]
        r1 = sum(A[j][0] * b[j] for j in range(4)); r2 = sum(A[j][1] * b[j] for j in range(4))
        sx = (a22 * r1 - a12 * r2) / det; sy = (a11 * r2 - a12 * r1) / det
        res = math.sqrt(sum((A[j][0] * sx + A[j][1] * sy - b[j]) ** 2 for j in range(4)))
        if best is None or res < best[0]:
            best = (res, m, complex(sx, sy))
    res, m, ks = best
    return dict(m=list(m), kstar=abs(ks), residual_bins=res / dk)


def main():
    src, out = sys.argv[1], sys.argv[2]; sets = sys.argv[3:] or ["vertices", "vortex"]
    data = json.load(open(src)); res = []
    for r in data:
        row = dict(K=r["K"])
        for s in sets:
            d = r[s]; seen = []; lst = []
            for p in d["top_peaks"][:20]:
                k = complex(p["kx"], p["ky"])
                if any(abs(k + q) < 1e-9 for q in seen): continue      # skip Friedel partner
                seen.append(k)
                ix = index(k, d["dk"]); ix.update(k=p["k"], h=p["h"]); lst.append(ix)
                print(r["K"], s, round(p["k"], 3), round(p["h"], 3), ix, flush=True)
            row[s] = lst
        res.append(row)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
