#!/usr/bin/python3
"""The evidence ledger: what kind of evidence stands behind each Fact.

A Fact in spectrefacts.py is not one kind of thing.  Some are exact
consequences of the quoted substitution matrix, some are counted on the placed
tiling, some are quoted from the literature, some carry a modelling premise.
Only some are put to the Lean kernel.  The paper says which, one Fact at a
time; this script says it all at once, mechanically, so a reader (and the
maintainer) can see the whole shape and so that a NEW Fact cannot be added
without being classified.

Six kinds of evidence, one row each:

    D  derived     an exact computation from the quoted matrix or formulas
    C  counted     depends on a count on the placed tiling, not derivable from M
    X  second route  re-derived or re-counted by an independent implementation
    Q  quoted      taken from the literature or a measurement
    P  premise     carries a modelling step or an open parameter
    L  Lean        a theorem in the generated SpectreFacts.lean proves it

The classification is a hand-written table (KIND) because "what is this Fact
resting on" is a judgement; the script asserts that the table covers every
Fact, that no Lean-linked Fact is quoted (Q) or premised (P), and that the counts it
prints agree with spectrefacts and spectrelean.  Where a Lean-linked Fact is
counted (C), the kernel certifies the arithmetic on the counted values and not
the count itself; the figure shows those cells lighter.  The idea of a machine-readable
catalogue of what is formalised, and of how, is borrowed from the
formalization.yaml catalogue of the openai/math collection; the file written
here is modelled on it and has not been validated against that schema.

    python3 spectre_ledger.py            # table, figure, paper/spectre_formalization.yaml
    python3 spectre_ledger.py --check    # assertions only
"""
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spectrefacts as F                  # noqa: E402
import spectrelean as L                   # noqa: E402

FIGDIR = os.path.join(HERE, "paper", "figures")
KINDS = ("D", "C", "X", "Q", "P")         # L comes from Fact.lean

KIND = {
    # the substitution matrix and its arithmetic
    'charpoly': 'D', 'perron': 'D', 'pell': 'D', 'lam_closed': 'D', 'minpoly': 'D',
    'channel_hex': 'D', 'channel_pent': 'D', 'contraction': 'D', 'jordan': 'D',
    'q_plus_eigen': 'D', 'q_minus_eigen': 'D', 'q_values': 'D',
    # the two censuses are counted twice; the Mystic rule follows from M and
    # is checked against the geometric count
    'censuses': 'CX', 'mystic_rule': 'DX', 'ledger': 'D', 'ledger_lambda': 'D',
    'hat_charpoly': 'D', 'hat_perron': 'D', 'phase_fields': 'D', 'frequencies': 'D',
    'hat_frequencies': 'D', 'aperiodicity_witness': 'D', 'trace_accounting': 'D',
    'minors': 'D', 'signature': 'D',
    # area and the order parameter
    'area_form': 'D', 'area_endpoints': 'D', 'mystic_asymmetry': 'D',
    'order_parameter': 'D',
    # the X-charge and the atlas are counted on the tiling; rho_x has an open
    # parameter; its value at one half is an extrapolation
    'rho_x': 'DCP', 'x_charge': 'C', 'atlas': 'C', 'phi_drift': 'C',
    'rho_x_at_half': 'CP',
    # chirality: the angle is quoted, what follows from it is arithmetic
    'chiral_angle': 'Q', 'angle_from_kappa': 'D', 'spectre_angle_differs': 'D',
    # knots: all quoted
    'bh_theorem': 'Q', 'bh_version': 'Q', 'bh_gap_was_chirality': 'Q',
    'bh_deficit_bound': 'Q', 'bh_threshold': 'Q',
    # the birefringence step
    'beta': 'P', 'beta_tension': 'QP',
    # the boundary: the recurrence is derived, its seed is counted
    'boundary_recurrence': 'DC', 'boundary_chi': 'D', 'boundary_growth': 'D',
    'boundary_field': 'D', 'boundary_dimension': 'D', 'balanced_exponent': 'D',
    'compositum': 'D', 'golden_absent': 'D', 'lucas_deflation': 'D',
    'spectre_not_golden': 'D', 'boundary_invariant': 'D',
    'boundary_growth_bound': 'D', 'lam_below_four': 'D',
    'boundary_exact': 'CX',
}

# The Chair44 section's Facts live in chair44facts.py and are appended to F.FACTS by register().
import chair44facts as _C44                # noqa: E402
_C44.register()
KIND.update(_C44.KIND)


def proof_style(text):
    """decide / induction / other, per theorem, read from the generated Lean."""
    out = {}
    blocks = re.split(r'(?m)^theorem ', text)[1:]
    for b in blocks:
        name = re.match(r'([A-Za-z_0-9]+)', b).group(1)
        body = b.split(':=', 1)[1] if ':=' in b else ''
        body = body.split('\n\n')[0]
        if 'induction' in body or ' ih' in body or 'ih ' in body:
            out[name] = 'induction'
        elif 'decide' in body:
            out[name] = 'decide'
        else:
            out[name] = 'other'
    return out


def audit_from_header(path=None):
    """The measured axiom audit, if SpectreFacts.lean has been checked."""
    path = path or os.path.join(HERE, 'SpectreFacts.lean')
    if not os.path.isfile(path):
        return None
    head = open(path, encoding='utf-8').read(6000)
    m = re.search(r'(\d+) theorems depend on no axioms at all', head)
    m2 = re.search(r'(\d+) depend on ([A-Za-z\.\s,]+) --', head)
    if not m:
        return None
    return dict(none=int(m.group(1)), some=int(m2.group(1)) if m2 else 0,
                which=m2.group(2).strip() if m2 else '')


def ledger():
    rows = []
    for f in F.FACTS:
        k = KIND[f.key]
        rows.append(dict(key=f.key, kinds=set(k), lean=f.lean,
                         flags={c: (c in k) for c in KINDS}, L=bool(f.lean)))
    return rows


def selftest():
    keys = [f.key for f in F.FACTS]
    assert len(keys) == len(set(keys))
    missing = [k for k in keys if k not in KIND]
    assert not missing, "unclassified Facts: %s" % missing
    stale = [k for k in KIND if k not in set(keys)]
    assert not stale, "classified but not a Fact: %s" % stale
    assert all(set(v) <= set(KINDS) for v in KIND.values())
    text = L.document()
    names = L.theorem_names(text)
    linked = {f.lean for f in F.FACTS if f.lean}
    assert linked <= set(names), "a Fact names a theorem that does not exist"
    for f in F.FACTS:
        if f.lean:
            k = KIND[f.key]
            # a kernel proof is about the arithmetic: where the inputs were
            # counted (C) it certifies the arithmetic on the counted values,
            # not the count; it never certifies something quoted or premised
            assert 'Q' not in k and 'P' not in k, \
                "%s is Lean-linked but is quoted or premised: %s" % (f.key, k)
    st = proof_style(text)
    assert set(st) == set(names)
    # nothing is admitted
    assert 'sorry' not in re.sub(r'(?s)/-.*?-/', '', text).replace('sorryAx', '')
    return True


def counts():
    text = L.document()
    names = L.theorem_names(text)
    st = proof_style(text)
    linked = {f.lean for f in F.FACTS if f.lean}
    combos = Counter()
    for f in F.FACTS:
        k = "".join(c for c in KINDS if c in KIND[f.key]) + ("+L" if f.lean else "")
        combos[k] += 1
    return dict(facts=len(F.FACTS), theorems=len(names), linked=len(linked),
                facts_with_lean=sum(1 for f in F.FACTS if f.lean),
                aux=len(names) - len(linked), styles=Counter(st.values()),
                combos=combos, audit=audit_from_header())


def report():
    c = counts()
    print("  %d Facts, %d Lean theorems (%d linked to a Fact, %d auxiliary)"
          % (c["facts"], c["theorems"], c["linked"], c["aux"]))
    print("  proof styles:", dict(c["styles"]))
    print("  Facts by evidence:", dict(sorted(c["combos"].items(), key=lambda kv: -kv[1])))
    print("  axiom audit in the generated header:",
          c["audit"] if c["audit"] else "none (run `make lean`)")
    rd = os.path.join(HERE, "Readme.md")
    if os.path.isfile(rd):
        t = open(rd, encoding="utf-8").read()
        m = re.search(r"\|\s*`spectrefacts.py`\s*\|\s*(\d+) Facts", t)
        m2 = re.search(r"—\s*(\d+) theorems", t)
        if m and int(m.group(1)) != c["facts"]:
            print("  NOTE: the README says %s Facts; spectrefacts has %d." % (m.group(1), c["facts"]))
        if m2 and int(m2.group(1)) != c["theorems"]:
            print("  NOTE: the README says %s theorems; the generated Lean has %d."
                  % (m2.group(1), c["theorems"]))


def write_yaml(path=None):
    path = path or os.path.join(HERE, "paper", "spectre_formalization.yaml")
    c = counts()
    lines = [
        "# Catalogue of the Facts of spectre_substrate.tex and the evidence behind each.",
        "# Generated by spectre_ledger.py.  Modelled on the formalization.yaml of",
        "# github.com/openai/math; NOT validated against that schema.",
        'version: "v0"',
        "project:",
        '  name: "spectre"',
        '  description: "Exact arithmetic of the Spectre monotile, with a Mathlib-free Lean 4 fact layer."',
        "lean:",
        "  file: SpectreFacts.lean",
        "  generated_by: spectrelean.py",
        "  theorems: %d" % c["theorems"],
        "  linked_to_a_fact: %d" % c["linked"],
        "  auxiliary: %d" % c["aux"],
        "  proof_styles: {%s}" % ", ".join("%s: %d" % kv for kv in sorted(c["styles"].items())),
    ]
    if c["audit"]:
        lines.append("  axiom_audit: {no_axioms: %d, standard_foundation: %d}"
                     % (c["audit"]["none"], c["audit"]["some"]))
    else:
        lines.append("  axiom_audit: not-run-in-this-build  # `make lean` measures it")
    lines.append("evidence_kinds: {D: derived, C: counted-on-the-tiling, X: second-route, "
                 "Q: quoted, P: premise-or-open, L: lean-checked}")
    lines.append("facts:")
    for f in F.FACTS:
        k = "".join(x for x in KINDS if x in KIND[f.key])
        lines.append("  - key: %s" % f.key)
        lines.append("    evidence: [%s]" % ", ".join(list(k) + (["L"] if f.lean else [])))
        if f.lean:
            lines.append("    lean: %s" % f.lean)
            if "C" in KIND[f.key]:
                lines.append("    lean_scope: arithmetic-on-counted-inputs")
    with open(path, "w", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")
    return path


def figure(fname="spectre_ledger"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
    GOLD, INK, GREY = "#c9a227", "#222222", "#e7e5df"
    rows = ledger()
    order = ["D", "C", "X", "Q", "P", "L"]
    names = {"D": "derived from $M$", "C": "counted on the tiling", "X": "second route",
             "Q": "quoted", "P": "premise / open", "L": "Lean kernel"}
    col = {"D": BLUE, "C": ORANGE, "X": AQUA, "Q": VIOLET, "P": GOLD, "L": INK}
    c = counts()

    fig = plt.figure(figsize=(13.0, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1.6, 1],
                          hspace=0.85, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    n = len(rows)
    for i, kind in enumerate(order):
        for j, r in enumerate(rows):
            on = r["L"] if kind == "L" else r["flags"][kind]
            fc = col[kind] if on else GREY
            if kind == "L" and on and r["flags"]["C"]:
                fc = "#8a877f"                  # kernel sees the arithmetic, not the count
            ax.add_patch(plt.Rectangle((j, len(order) - 1 - i), 0.88, 0.8,
                                       fc=fc, ec="none"))
    ax.set_xlim(-0.2, n)
    ax.set_ylim(-0.1, len(order))
    ax.set_yticks([len(order) - 1 - i + 0.4 for i in range(len(order))])
    ax.set_yticklabels([names[k] for k in order], fontsize=8.5)
    ax.set_xticks([j + 0.44 for j in range(n)])
    ax.set_xticklabels([r["key"] for r in rows], rotation=90, fontsize=6.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) the %d Facts of the paper, in paper order, and what stands behind each "
                 "(light grey Lean cell: the kernel checks the arithmetic on counted inputs)" % n,
                 fontsize=9.5)

    ax = fig.add_subplot(gs[1, 0])
    combos = sorted(c["combos"].items(), key=lambda kv: -kv[1])
    ax.barh(range(len(combos))[::-1], [v for _, v in combos], color="#9dc0ea", height=0.7)
    ax.set_yticks(range(len(combos))[::-1])
    ax.set_yticklabels([k for k, _ in combos], fontsize=8)
    for y, (_, v) in zip(range(len(combos))[::-1], combos):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=8)
    ax.set_xlabel("Facts")
    ax.set_title("(b) Facts by evidence combination (+L: Lean-checked)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    st = c["styles"]
    segs = [("decide", st.get("decide", 0), BLUE), ("induction", st.get("induction", 0), ORANGE),
            ("other", st.get("other", 0), "#8a877f")]
    left = 0
    for nm, v, cc in segs:
        ax.barh([0], [v], left=left, color=cc, height=0.5)
        if v >= 8:
            ax.text(left + v / 2, 0, "%s\n%d" % (nm, v), ha="center", va="center",
                    color="white", fontsize=8.5)
        elif v:
            ax.text(left + v / 2, 0.36 + 0.28 * (nm == "induction"), "%s %d" % (nm, v),
                    ha="right" if nm == "induction" else "center", va="bottom",
                    color=cc, fontsize=8)
        left += v
    ax.barh([1], [c["linked"]], color=INK, height=0.5)
    ax.barh([1], [c["aux"]], left=c["linked"], color="#bdbab2", height=0.5)
    ax.text(c["linked"] / 2, 1, "linked to a Fact\n%d" % c["linked"], ha="center",
            va="center", color="white", fontsize=8.5)
    ax.text(c["linked"] + c["aux"] / 2, 1, "auxiliary\n%d" % c["aux"], ha="center",
            va="center", color=INK, fontsize=8.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["proof style", "role"], fontsize=9)
    ax.set_xlim(0, c["theorems"])
    ax.set_ylim(-0.4, 1.6)
    ax.set_xlabel("the %d theorems of SpectreFacts.lean" % c["theorems"])
    ax.set_title("(c) the Lean file", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, fname + "." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figures/%s.{pdf,png}" % fname)


if __name__ == "__main__":
    print("selftest:", selftest())
    if "--check" in sys.argv:
        sys.exit(0)
    report()
    print("wrote", os.path.relpath(write_yaml(), HERE))
    figure()
