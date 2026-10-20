#!/usr/bin/env python3
r"""spectrepaper.py -- the Spectre substrate paper, written out.

Every number, matrix and table below is a call into spectrefacts.py at build
time.  Nothing is transcribed, for the reason the previous draft demonstrated:
a census convention drifted between two sections, the charge values were quoted
against a total that does not produce them, and both halves were individually
correct.  Prose does not have a selftest.  A generated document cannot disagree
with its own arithmetic because there is only one copy of the arithmetic.

What is written by hand is the argument.  What is generated is every claim of
fact, and the Lean layer that certifies the closed ones.

    python3 spectrepaper.py              write spectre_substrate.tex
    python3 spectrepaper.py --stdout     print it instead
    python3 spectrepaper.py --selftest   check the generated document
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp

import spectrefacts as F
import spectrelean as L
import spectrevariety as VAR
import spectre_mystic_fibres as MF
import spectre_torsion as TOR
import spectre_energy as EN
import spectre_diffraction as DIF
import spectre_ledger as LED
import chair44facts as C44
C44.register()

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'spectre_substrate.tex')



# ---------------------------------------------------------------- figures

def figure_block(name, label, caption, width=r'\textwidth'):
    """A float for paper/figures/<name>.pdf, made by spectre_figures.py or
    spectre_mystic_fibres.py.  The caption quotes no number that is not also
    checked in selftest()."""
    return (r"""
\begin{figure}[t]
\centering
\includegraphics[width=%s]{%s}
\caption{%s}
\label{%s}
\end{figure}
""" % (width, name, caption, label))


FIG_FAMILY = '\nFigure~\\ref{fig:family} draws the family that this section compares.\n' + figure_block(
    'spectre_family', 'fig:family',
    r"""The Tile$(a,b)$ family.  (a) The Hat, the Spectre and the Turtle (the
Mystic) at one scale.  (b) The area form $A(a,b)=2\sqrt3\,a^2+3ab+\sqrt3\,b^2$
along $b/a$, and with the roles exchanged: the two curves meet only at the
Spectre, $a=b$.  (c) The chiral twist $\cos\theta=(\kappa-3)/4$ of the Hat, with
$\kappa=\varphi^4$, is a cosine; the same expression for the Spectre,
$(\lambda^2-3)/4$, exceeds $1$ and is not.  The Spectre has its own twist
$\theta_S$, computed from its supertile quadrilaterals (Fact~\ref{fact:twist}).""")

FIG_CENSUS = '\nFigure~\\ref{fig:census} summarises the substitution that this section analyses.\n' + figure_block(
    'spectre_census', 'fig:census',
    r"""The substitution.  (a) The matrix $M$.  (b) Its spectrum on a symmetric
logarithmic axis, against that of the Hat matrix: the Pisot pair $\lambda^{\pm2}$
with $\lambda^2=4+\sqrt{15}$, and $\varphi^{\pm4}$.  (c) The ranks of $M^n$ drop
from $5$ to $4$ and stay: the Jordan pair.  (d) Tiles by depth, with and without
the Mystic, against $\lambda^{2n}$.  (e,f) The species frequencies: irrational
for the Spectre, with the Hat's $f_H=1/3$ rational.""")

FIG_BOUNDARY = '\nFigure~\\ref{fig:boundary} shows the supertiles whose perimeter is counted here.\n' + figure_block(
    'spectre_boundary', 'fig:boundary',
    r"""Supertiles of order $k=0,\dots,3$ with their boundary edges
(orange), and the perimeter $P(k)$ against $\nu^k$ with $\nu=\varphi^3$ and
against the naive $\lambda^k$.  The edge counts drawn are counted on the
geometry and asserted equal to the recurrence of this section.""")

FIG_TORSION_CAPTION = r"""The torsion dictionary.  (a) The torus $\mathbb{C}/\mathbb{Z}[\zeta_3]$
with its six site classes: the centre $0$, the three nonzero $2$-torsion points
$D_a,D_c,D_b$ (edge midpoints) and the two nonzero points $V_1,V_2$ of
$E[\sqrt{-3}]$ (hexagon vertices); the arrows are the rotation $R$.  (b) The six
Mystic classes, first-edge angle and anchor, as one orbit of $R$.  (c) The same
classes as $\mathbb{Z}/3\times\mathbb{Z}/2$, with the number of Mystics in each at
level $5$."""

FIG_ENERGY = figure_block(
    'spectre_energy', 'fig:energy',
    r"""Gaussian-kernel energy per particle of the vertex and centroid sets of the Hat,
Spectre and Turtle tilings (level-$5$ patch, unit density), as the gap above the
triangular lattice, against the square and honeycomb lattices and a Poisson
process.  (c) The gaps at the single width $\alpha=1$.""")

FIG_LEDGER = figure_block(
    'spectre_ledger', 'fig:ledger',
    r"""The evidence ledger.  (a) Each Fact, in paper order, against the kind of
evidence behind it.  (b) Facts by combination of evidence.  (c) The Lean file by
proof style and by role.""")

_MF_CACHE = {}


def mystic_data():
    """The fibre-class census of spectre_mystic_fibres.py, levels 2..5, and the
    number of distinct environments per class at depth 5.  Computed once."""
    if not _MF_CACHE:
        rows = [MF.census(n, with_env=False) for n in (2, 3, 4, 5)]
        r5 = rows[-1]
        _MF_CACHE['rows'] = rows
        _MF_CACHE['env'] = MF.env_counts(r5)
    return _MF_CACHE['rows'], _MF_CACHE['env']


_EN_CACHE = {}


def energy_numbers():
    if not _EN_CACHE:
        res = EN.measure()
        tri = EN.controls()['triangular']
        ctr = EN.controls()
        import numpy as _np

        def gap(y, a):
            return float(_np.interp(_np.log(a), _np.log(EN.ALPHAS), y / tri - 1))
        g = {}
        for (name, which), d in res.items():
            g[(name, which, 1.0)] = gap(d['E'], 1.0)
            g[(name, which, 4.0)] = gap(d['E'], 4.0)
        for nm in ('square', 'honeycomb', 'Poisson'):
            g[(nm, 1.0)] = gap(ctr[nm], 1.0)
        lowest = min(float((d['E'] / tri - 1).min()) for d in res.values())
        noise = max(float((d['spread'] / tri).max()) for d in res.values())
        _EN_CACHE.update(g=g, lowest=lowest, noise=noise)
    return _EN_CACHE


def inject_before(text, marker, block):
    assert text.count(marker) == 1, marker
    return text.replace(marker, block + marker, 1)


def energy_subsection():
    e = energy_numbers()
    g = e['g']
    cen = [g[(n, 'centroids', 1.0)] for n in ('Hat', 'Spectre', 'Turtle')]
    return r"""
\subsection{A geometric check: energy above the triangular lattice}
\label{sec:energy}

The order parameter above is built from frequencies and areas.  A purely
geometric companion is available.  For a point set of unit density let
$E(\alpha)=\lim N^{-1}\sum_{i\ne j}e^{-\alpha|x_i-x_j|^{2}}$ be the Gaussian-kernel energy per
particle, the simplest completely monotone function of the squared distance.  The
triangular lattice is the conjectured planar minimiser for every such potential
(Cohn and Kumar), and the OpenAI mathematics collection~\cite{OpenAIMath}, family~090,
claims a proof.  That collection states that its results come from an internal
model, that not all are formalised and some may contain errors, and its own history
records withdrawn manuscripts; the floor is therefore used here only as a benchmark to
measure against, and nothing in this paper depends on it.

\textbf{Computation, not Fact.}  Figure~\ref{fig:energy} gives
$E(\alpha)/E_{A_2}(\alpha)-1$ for the vertices and for the tile centroids of the Hat,
Spectre and Turtle tilings, on the level-$5$ supertile of each, with exact densities
(six vertices and one centroid per tile, tiles per unit area
$1/(A(a,b)+\sqrt3(b^{2}-a^{2})/(1+\lambda^{2}))$) and a window kept a cutoff length from the
patch boundary.  No set falls below the floor beyond the estimated noise
(the lowest value is $%+.4f$, the largest half-difference between two disjoint windows
$%.4f$).  At $\alpha=1$ the vertex sets lie at $%.3f$ (Spectre), $%.3f$ (Hat) and $%.3f$
(Turtle) above the triangular lattice, the centroid sets at $%.4f$ to $%.4f$, the
honeycomb lattice at $%.4f$, the square lattice at $%.4f$ and a Poisson process at
$%.3f$.  The aperiodic sets are thus not near the crystalline floor at this width: they
sit above both lattices, and the Spectre vertex set is the closest of the three.  No
statement about phases is drawn from this: the order of the three tilings depends on the
set and on $\alpha$ (at $\alpha=4$ the centroid sets are ordered Hat, Spectre, Turtle), and it
is a measurement on one level.
""" % (e['lowest'], e['noise'], g[('Spectre', 'vertices', 1.0)], g[('Hat', 'vertices', 1.0)],
         g[('Turtle', 'vertices', 1.0)], min(cen), max(cen), g[('honeycomb', 1.0)],
         g[('square', 1.0)], g[('Poisson', 1.0)]) + FIG_ENERGY


FIG_DIFFRACTION = figure_block(
    'spectre_diffraction', 'fig:diffraction',
    r"""Windowed structure factor $S(\mathbf{k})$ of the vertex sets of the level-$5$ Hat, Spectre and
Turtle tilings (Gaussian window, unit density).  (a)--(c) $S(\mathbf{k})$ on a logarithmic grey scale;
circles mark the reciprocal lattice $\Lambda^{*}$ of the torus lattice, scaled to unit density, at the
orientation that fits best ($0^\circ$ for the Hat, $30^\circ$ for the Turtle).  (d) Azimuthal mean of $S$,
with the multiples of $|b|$ of the Hat marked.  (e) The same at small $|k|$ on logarithmic axes, with the
triangular lattice as a measure of window leakage.  (f) The energy gap of Section~\ref{sec:energy} computed
from $S$, and its part from $|k|$ below and above $0.9|b|_{\rm Hat}$.""")

_DIF_CACHE = {}


def diffraction_numbers():
    if not _DIF_CACHE:
        res = DIF.measure()
        sm = DIF.summarise(res)
        kc = 0.9 * res['Hat']['b']
        import numpy as _np
        sel = _np.argmin(abs(EN.ALPHAS - 1.0))
        low = {}
        for nm in ('Hat', 'Spectre'):
            tot, lo, hi = DIF.energy_split(res, nm, kc)
            low[nm] = float(lo[sel] / tot[sel])
        _DIF_CACHE.update(sm=sm, kc=kc, low=low)
    return _DIF_CACHE


def diffraction_subsection():
    d = diffraction_numbers()
    sm, low = d['sm'], d['low']
    H, S, T = sm['Hat'], sm['Spectre'], sm['Turtle']
    return r"""
\subsection{Reciprocal-Space Analysis: Structure Factors and Dual Lattices}
\label{sec:diffraction}

The torus lattice $\Lambda=\mathbb{Z}E_1+\mathbb{Z}E_2$ of the six-sphere cusp has a dual
lattice $\Lambda^{*}$, and the Hat-versus-Spectre distinction can be read there.  For the vertex
set $\{x_j\}$ of a level-$5$ patch, at unit density, let
$S(\mathbf{k})=\lvert\sum_j w_je^{i\mathbf{k}\cdot x_j}\rvert^{2}/\sum_j w_j^{2}$ with a Gaussian window $w$
of width $9$.  The window is what makes the energy of Section~\ref{sec:energy} a reciprocal-space
quantity: by Parseval, $\sum_{i,j}w_iw_je^{-\alpha|x_i-x_j|^{2}}/\sum w^{2}=\int\frac{d^{2}k}{(2\pi)^{2}}\,\frac{\pi}{\alpha}e^{-k^{2}/4\alpha}S(\mathbf{k})$,
which the selftest checks numerically.  The shortest reciprocal vector at unit density is
$|b|=%.3f$ (Hat), $%.3f$ (Turtle) and $%.3f$ for the Spectre vertex set, where there is no lattice and
the number is only the scale that the same formula would give.

\textbf{Computation, not Fact.}  Figure~\ref{fig:diffraction} shows three things.  First, the Hat and
Turtle tilings have their strongest Bragg peaks on $\Lambda^{*}$: a fraction $%.2f$ of the
Bragg intensity above threshold lies within $12\%%$ of a lattice point for the Hat (at $0^\circ$, from %d peaks)
and $%.2f$ for the Turtle (at $30^\circ$, from %d peaks), against $%.2f$ and $%.2f$ for the same lattice
rotated off those orientations or rescaled.  The Turtle's lattice is rotated by $30^\circ$ from the Hat's.
The Spectre has strong peaks as well, but its on-lattice fraction is $%.2f$ (control $%.2f$, %d peaks): none of
its strong peaks is on such a lattice, which is the reciprocal-space form of the statement that
$\mathrm{Tile}(1,1)$ is not on $\Lambda$.  Second, all three have a low-$S$ region at small $|k|$,
with $\langle S\rangle$ at its minimum $%.4f$ (Spectre, $k=%.2f$), $%.4f$ (Hat, $k=%.2f$) and
$%.4f$ (Turtle, $k=%.2f$) between the window-dominated region $k\lesssim0.35$ and the onset of structure
near $k\approx1$.  That is compatible with suppressed long-wavelength fluctuations, but no exponent is
claimed: at window width $9$ the range is too short to fit one, and the triangular lattice, which is
exactly hyperuniform, shows what window leakage alone does.  Third, writing the energy gap of
Section~\ref{sec:energy} as an integral of $S-S_{A_2}$, the part from $|k|<%.1f$ is $%.2f$ (Hat) and
$%.2f$ (Spectre) of the gap at $\alpha=1$ (panel~(f)), so at wide kernels the gap is carried by the diffuse
low-$k$ structure and the Bragg shells matter only at larger $\alpha$.  The kinks seen in the energy curves are
not explained by the first Bragg peaks; panel~(f) shows only that they occur where the diffuse part turns over,
and nothing stronger is claimed.  All of this is one patch, one window and $|k|\le9$.
""" % (H['b'], T['b'], S['b'],
         H['match']['frac'], H['match']['npeaks'], T['match']['frac'], T['match']['npeaks'],
         H['match']['base'], T['match']['base'],
         S['match']['frac'], S['match']['base'], S['match']['npeaks'],
         S['floor'], S['floor_k'], H['floor'], H['floor_k'], T['floor'], T['floor_k'],
         d['kc'], low['Hat'], low['Spectre']) + FIG_DIFFRACTION


def ledger_subsection():
    c = LED.counts()
    cb = c['combos']
    return r"""
\subsection*{An evidence ledger}
\begin{footnotesize}
The Facts do not all rest on the same kind of evidence, and the Lean file covers only
some.  \texttt{spectre\_ledger.py} classifies every Fact, by hand-written table that a
selftest forces to be complete, as derived from the quoted matrix, counted on the placed
tiling, re-derived by a second implementation, quoted from the literature, or carrying a
modelling premise or an open parameter (Figure~\ref{fig:ledger}).  Of the %d Facts, %d are
derived only and %d of those are also in the kernel; %d are quoted; the rest are counted,
premised or both.  The generated Lean file has %d theorems, %d of them linked to a Fact and
the others auxiliary.  Where a Lean-linked Fact depends on a count, the kernel certifies the
arithmetic on the counted values and not the count, and the figure marks those cells.
The machine-readable catalogue written beside the figure
(\texttt{paper/spectre\_formalization.yaml}) follows the idea of the formalization catalogue kept by the
mathematics collection~\cite{OpenAIMath}; it is not validated against that collection's schema.
\end{footnotesize}
""" % (c['facts'], cb.get('D', 0) + cb.get('D+L', 0), cb.get('D+L', 0), cb.get('Q', 0),
         c['theorems'], c['linked']) + FIG_LEDGER


def inject_after_label(text, label, block):
    """Put a figure float straight after \\label{label}."""
    marker = '\\label{%s}\n' % label
    assert text.count(marker) == 1, label
    return text.replace(marker, marker + block, 1)


# ---------------------------------------------------------------- helpers

def tex(expr):
    """A sympy object as LaTeX, compactly.

    sympy spells a factored polynomial with \\left( ... \\right) around every
    factor, which is correct and roughly twice as wide as it needs to be; at
    nine species the characteristic polynomial then overruns the text block.
    The delimiters are fixed-size here because every factor is a short linear
    or quadratic term.
    """
    out = sp.latex(expr)
    out = out.replace(r'\left(', '(').replace(r'\right)', ')')
    return out.replace(') (', ')(')


def esc(text):
    out = []
    for ch in str(text):
        out.append({'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                    '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}.get(ch, ch))
    return ''.join(out)


def number(n):
    names = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
             'eight', 'nine', 'ten', 'eleven', 'twelve']
    return names[n] if 0 <= n < len(names) else str(n)


def matrix_tex(M):
    rows = r' \\'.join('&'.join(str(int(M[i, j])) for j in range(M.shape[1]))
                       for i in range(M.shape[0]))
    return r'\begin{pmatrix}%s\end{pmatrix}' % rows


def species_row(vec):
    return '$(' + ',\\,'.join(F.TEX[n] for n in F.SPECIES) + ')$'


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1.1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,booktabs,longtable,xcolor,array}
\usepackage{listings}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{orcidlink}
\usepackage{graphicx}
\graphicspath{{paper/figures/}}

\newtheorem{fact}{Fact}
\newtheorem{prop}{Proposition}
\newtheorem{conj}{Conjecture}
\newcommand{\lam}{\lambda}
\newcommand{\code}[1]{\texttt{#1}}
\emergencystretch=2em
\hyphenation{re-flec-tion-ad-mit-ting quasi-crys-tal-line}

\lstdefinelanguage{lean}{
    morekeywords={theorem,def,structure,instance,namespace,end,deriving,
                  by,decide,omega,simp,induction,intro,Int,Nat,List,Prop},
    morecomment=[l]{--},
    morecomment=[n]{/-}{-/},
    sensitive=true
}
\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single,
        framerule=0.3pt,columns=fullflexible,keepspaces=true,
        showstringspaces=false,commentstyle=\color{gray}\itshape,
        keywordstyle=\color{blue!60!black}\bfseries}

\title{Area, Perimeter and Parity: Comparing the Spectre Monotile with Tsiokos's Chair44 Three-Dimensional Monotile}

\author{Brent S. Hartshorn \orcidlink{0009-0004-2853-655X}
        (\url{brenthartshorn@proton.me})}

\begin{document}
\maketitle
"""


def fill(text, **kw):
    """Replace @key@ markers; no % formatting, so LaTeX comments and percent signs stay as written."""
    for k, v in kw.items():
        marker = '@%s@' % k
        assert marker in text, marker
        text = text.replace(marker, str(v))
    assert '@' not in text.replace(r'\@', ''), 'unfilled marker'
    return text




def boundary_additions():
    """Two subsections of the boundary-operator section: the closed form and the directions of
    the boundary edges, and the balanced exponent (moved here from the discussion section)."""
    import chair44facts as C44
    rows = []
    for k, h in enumerate(C44.DIRECTION_HISTOGRAMS):
        e, o = C44.class_totals(h)
        modes = C44.z6_modes(h)
        other = max(m for j, m in enumerate(modes) if j not in (0, 3))
        rows.append(r'%d & %d & %d & %d & %d & %.2f \\' % (
            k, F.BOUNDARY_SEQ[k], e, o, e - o, other))
    return fill(r"""
\subsection{A closed form, and the directions of the boundary edges}
\label{sec:bdyclosed}

The recurrence has a closed form in Fibonacci numbers.

\begin{fact}
\label{fact:bdyfib}
$P(k)=2F_{3k+5}+4$ for every $k\ge0$, where $F_n$ is the Fibonacci number.
\end{fact}

\noindent
It follows from the recurrence and the seeds $14,46,182$, because
$F_{n+6}=4F_{n+3}+F_{n}$ makes $2F_{3k+5}+4$ satisfy the same recurrence, and it
is proved in Lean for all $k$ by induction (\code{boundary\_fibonacci}), not
sampled.  It says three things that the recurrence did not.  The growth
$\nu=\varphi^{3}$ is one Spectre level multiplying the Fibonacci index by three,
which is how the golden unit arises.  The marginal eigenvalue $+1$ is the
constant $4$, so $(P(k)-4)/2$ is every third Fibonacci number.  And the third
root of $x^{2}-4x-1$ is $2-\sqrt5=-\varphi^{-3}$, which is negative, so that
component alternates in sign with the parity of the level.  That is the same
parity that flips the determinant of the placement matrix, but any quantity
alternating with level parity does so, and this is weak evidence of a link to
the chirality channel and is not claimed as one.

The boundary edges also have directions.  The fourteen edges of the Spectre
point along multiples of $30^{\circ}$, so a boundary edge has one of twelve
oriented directions, and the histogram of an order-$k$ supertile has period six:
it depends only on the direction modulo $180^{\circ}$.  Splitting those six
classes by parity gives, counted on the placed tiling through $k=6$,
%
\begin{equation}
\label{eq:dirsplit}
\underbrace{h_{0}+h_{2}+h_{4}}_{\text{even}}=F_{3k+4}+1,
\qquad
\underbrace{h_{1}+h_{3}+h_{5}}_{\text{odd}}=F_{3k+3}+1,
\end{equation}
%
whose sum doubled is $P(k)$.  The difference of the two totals is exactly
$F_{3k+2}$, which is the amplitude of the $\mathbb{Z}/2$ character in
Table~\ref{tab:dirs}.  The remaining two characters of
$\mathbb{Z}/6=\mathbb{Z}/2\times\mathbb{Z}/3$ stay bounded:
their largest amplitude over $k\le6$ is @peak@.  This is the same group
$\mathbb{Z}/6$ that the torsion dictionary of
Conjecture~\ref{conj:six} uses for the six Mystic classes (anchor modulo three, sign
modulo two), and the data identify the group, not the action.

\begin{table}[h]
\centering
\caption{Boundary edges of an order-$k$ supertile by direction modulo
$180^{\circ}$.  Even and odd are the class totals of~\eqref{eq:dirsplit};
$|\hat h|$ is the largest amplitude of the $\mathbb{Z}/3$ and faithful
characters of the folded histogram.  Counted on the placed tiling
(\code{chair44\_spectre\_link.py}); the $k=6$ row took 66\,s.}
\label{tab:dirs}
\begin{tabular}{rrrrrr}
\toprule
$k$ & $P(k)$ & even & odd & even $-$ odd & $|\hat h|$ \\
\midrule
@rows@
\bottomrule
\end{tabular}
\end{table}

\textbf{Counted, not derived.}  The recurrence and the closed form are exact;
the split~\eqref{eq:dirsplit} is a count and is not proved beyond $k=6$.  What
the kernel certifies is the arithmetic that ties it to the perimeter
(\code{direction\_split}): the two totals add up to $P(k)$.  The boundedness of
the other characters is an observation.  The directions are counted with the
fork's own tile builder, so this is a count and not an independent second
route.

\subsection{The balanced exponent}
\label{sec:sstar}

A boundary that grows by $\nu$ per level while the area grows by $\lam^{2}$
raises the question of how to weigh the two against each other, and the answer
is a single exponent.

\begin{prop}
\label{prop:sstar}
Interfaces at coarseness $c$ in a depth-$N$ patch number
$\bigl(T(c)P(N-c)-P(N)\bigr)/2$, so they grow in $c$ by $\mu=\lam^{2}/\nu$.  If
each such unit edge is thickened to a neighbourhood of width
$\delta_{c}=\delta_{0}\lam^{-sc}$ and cross-section $\delta_{c}^{2}$, as for a
curve in $\mathbb{R}^{3}$, the exponent making every level carry comparable
volume is
%
\begin{equation}
\label{eq:sstar}
s^{*}=1-\frac{\log\nu}{\log\lam^{2}}=@sstar@ .
\end{equation}
\end{prop}

\noindent
The natural guess $s=2$, one factor of the unit per level, is wrong, and
Section~\ref{sec:bdyop} says why: every interface edge is a \emph{unit} tile edge
whatever level it sits at, so length does not scale with level and only the
population does.  Two cautions attach to~\eqref{eq:sstar}.  It is exact but
\emph{not algebraic}: a ratio of logarithms of algebraic numbers has a closed
form and arbitrary precision but is not a root of any polynomial.  The
algebraic object is $\nu$; $s^{*}$ is what remains after taking its logarithm,
and a reader wanting an element of $\mathbb{Z}[\sqrt{15}]$ should stop at $\nu$
and accept it in $\mathbb{Q}(\sqrt5)$.  And $s^{*}$ had to be derived rather than
fitted: because the interface count depends on $T(c)$ and on $P(N-c)$ together,
a single-exponent fit needs $c$ large and $N-c$ large at once, which no finite
patch provides.  Fitting one anyway returns values that drift with depth and
never settle --- and the closed form reproduces those drifting values, which is
how we know the drift was the estimator and not the geometry.
""", peak='%.2f' % F.get('boundary_z3_bounded').value[0], rows='\n'.join(rows),
        sstar=F.get('balanced_exponent').decimal)




def chair_section():
    """The three-dimensional control.  Every number is a Fact of chair44facts.py."""
    import chair44facts as C44
    g = F.get
    A = g('chair_atlas').value
    orient = g('chair_orientation').value
    c = g('chair_chiral').value
    forced = g('chair_forced_chirality').value
    return fill(r"""
\section{From Plane to Bulk: Tsiokos's Chair44}
\label{sec:chair}

Everything above is two-dimensional, and the question this section asks is which of it
depends on that.  Tsiokos's Chair44~\cite{Tsiokos1} is a candidate three-dimensional
einstein: a single tile that fills $\mathbb{R}^{3}$, claimed to do so only
aperiodically.  The tile is the seven-cube chair (a $2\times2\times2$ block less one corner) with
each of its $24$ unit faces carrying eight small pyramids whose heights are signed.
Its einstein property and its strong aperiodicity are the author's claimed proof and are
quoted here, not checked: they are \emph{quoted} in the evidence ledger, and nothing below
bears on them.  What is recomputed is the combinatorial atlas, and it is recomputed from
nothing: \code{symchair44.py} generates the tile in exact rational arithmetic from the cell
set and the pyramid grid, with no external data, and its output agrees vertex for vertex with
the published solid (\code{symchair44\_tests.py}).  Nine tests and ten more in
\code{chair44\_tests.py} carry the comparison; the Facts of this section are in
\code{chair44facts.py} and are listed with the others in the supplement.

\subsection{Combinatorial Reconstruction of the Chair44 Tile}

The mesh has $@V@$ vertices, $@E@$ edges and $@T@$ triangles, is a closed
manifold of Euler characteristic $2$, and has volume exactly $7$ for \emph{every} base
width $w$ and height unit $h$, because the $192$ signed heights sum to zero and the volume is
$7+\tfrac43 w^{2}h\sum_{s}\gamma_{s}$.  The substitution places eight copies of the chair
on the doubled chair, with the refinement $(G,t)\mapsto(GH,\,2t+Gu)$.  There are exactly
$3^{8}=@tilings@$ ways to cover the doubled chair by eight rotated chairs; gluing each to
depth two and demanding exact complementarity ($\gamma_A+\gamma_B=0$ for the pyramids on a
shared face) leaves four free height classes for $6555$ of them and twelve for six.  Of those six,
three use all $24$ orientations and the first in sort order is the published substitution; the
selection rule is a convention and is stated as one.  The twelve classes are the heights
$\pm1,\dots,\pm12$.  The contact atlas follows from the labels: of $@raw@$ raw face contacts,
$@mirror@$ are mirror images, $@legal@$ are legal, and $@realized@$ occur in the depth-three
hierarchy, giving $@neq@$ site equations of rank $@eqrank@$ and a kernel of dimension $@ker@$,
the twelve free classes.  These counts reproduce the published atlas from the labels
alone.  The kernel checks the arithmetic that ties them together
(\code{chair\_labels\_sum}, \code{chair\_euler}, \code{chair\_atlas\_counts},
\code{chair\_rank\_nullity}); it does not see the gluing, and it does so with
\code{decide} and \code{omega}, not \code{native\_decide}, which would trust the compiler
more than an inductive proof.

\begin{footnotesize}
\begin{table}[h]
\centering
\footnotesize
\caption{The Spectre and the Chair44 side by side.  $d$ is the dimension, $\lam$ the linear
inflation, $\nu$ the boundary growth per level, $D_b=\log\nu/\log\lam$,
$s^{*}=1-D_b/d$ and $\mu=\lam^{d}/\nu$.  The last two rows are about what is claimed, not
computed.}
\label{tab:chair}
\begin{tabular}{lll}
\toprule
 & Spectre & Chair44 \\
\midrule
$d$ & $2$ & $3$ \\
$\lam$ & $\tfrac12(\sqrt6+\sqrt{10})\approx2.806$ & $2$ \\
tile count per level & $\lam^{2}\approx7.87$ & $8=\lam^{3}$ \\
boundary growth $\nu$ & $\varphi^{3}\approx4.236$ & $4$ \\
$D_b$ & $@Db@$ & $2$ \\
$s^{*}$ & $@sstar@$ & $1/3$ \\
$\mu$ & $1.8586$ & $2$ \\
fractal excess $D_b-(d-1)$ & $+0.3993$ & $0$ \\
area and perimeter fields & $\mathbb{Q}(\sqrt{15})$, $\mathbb{Q}(\sqrt5)$ & both $\mathbb{Q}$ \\
orientation species & $9$ tiles & $24$ rotations \\
zero sector & not semisimple & semisimple \\
frequencies & irrational (Pisot) & rational, $1/24$ \\
strict chirality & yes, by the tile & yes, by the contacts \\
spectrum of the tiling & Pisot, pure point & 2-adic, pure point (cited theorem) \\
\bottomrule
\end{tabular}
\end{table}
\end{footnotesize}

\subsection{The boundary operator in three dimensions}

The Chair44 super-tile of depth $N$ has $24\cdot4^{N}$ boundary faces and
$12(8^{N}-4^{N})$ shared faces, since $24$ faces on each of $8^{N}$ tiles less two for every
shared face leaves $24\cdot4^{N}$; the identity is proved in Lean for all $N$
(\code{chair\_surface}).  The faces point equally in all six axis directions,
$4^{N+1}$ each, counted through $N=4$, which is the opposite of the Spectre's direction
histogram with its golden split.  The interface counts obey the law that Proposition~\ref{prop:sstar}
states for the Spectre,
\[
B_{c}=\tfrac12\bigl(T(c)P(N-c)-P(N)\bigr)=12\cdot4^{N}(2^{c}-1),
\qquad T(c)=8^{c},\ P(k)=24\cdot4^{k},
\]
so the interface law is not an accident of the plane.  What is different is the exponent.
With $\lam=2$ and $\nu=4$ the boundary grows by exactly $\lam^{2}$, so
$D_b=2=d-1$: the supertile boundary of Tsiokos's Chair44 is an ordinary surface, with no fractal
excess, where the Spectre's has $+0.3993$.  Area and perimeter inflate in the same field, so there is no divide
between them, and the balanced exponent is the rational $s^{*}=1/3$.  The one-tile-two-units
rule of Section~\ref{sec:bigpicture} therefore describes the Spectre and not tilings in
general: it is plausibly a feature of a planar polygon of unit edges grown by a
non-integer inflation, and it does not survive a passage to an integer inflation in three
dimensions.

\subsection{The orientation spectrum, and the 2-adic route}

Let the $24$ proper rotations be the species and let $M_{FF'}$ count the children of an
$F$-oriented chair that are $F'$-oriented.  The characteristic polynomial is
$@charpoly@$, the minimal polynomial divides $x(x-8)(x-2)$, every row and column sums to
$8$, and $\operatorname{rank}M^{n}=@orank@$ for all $n\ge1$.  The zero sector is
semisimple, there is no negative eigenvalue, and orientation frequencies converge to $1/24$
at the rate $(2/8)^{d}$ in the depth $d$.  The Spectre's matrix is the opposite on each
count: a Pisot unit, irrational frequencies and a zero sector with a Jordan pair.  The
consequence is a narrowing.  The Pisot route to pure-point diffraction and the
irrational-frequency witness of aperiodicity (Section~\ref{sec:phases}) both need an inflation
that is a Pisot number and frequencies that are irrational, and an integer inflation $2$ with
rational frequencies supplies neither.

The replacement is $2$-adic, and it is not new here: Tsiokos's repository records it
\cite{Tsiokos1}.  The substitution acts on $24\times7=168$ labelled unit cells, is primitive
(every entry of its $168\times168$ matrix is positive at power $3$), and has a modular
coincidence: at depth $3$ there are @coinc@ of $512$ addresses, among them $(0,0,2)$, at which
all $168$ labels expand to the same label.  By the theorem of Lee and Moody (2001) with
Schlottmann (2000) \cite{LeeMoody,Schlottmann,LMS}, a primitive lattice substitution with a modular
coincidence has fixed-point label classes that are regular model sets with a $2$-adic internal
space, so its diffraction is pure point and the fixed point is limit-periodic.  That implication
is a cited theorem and is not re-proved, and whether every tiling lies in the substitution hull
is left open upstream.  What is added here is a recomputation of the finite part from our own
generator, which agrees on the depth and the address, and a rate.  A subset automaton on the label
sets (only @reach@ are reachable, @nons@ of them with more than one label) counts the digit words
of length $M$ that have not yet fixed the label:
$c(M)=1,8,64,176,400,848,1744,\dots$, that is $c(M+1)=2c(M)+48$ and $c(M)=28\cdot2^{M}-48$
from $M=3$, proved in Lean for all $M$ (\code{coincidence\_closed}).  The undetermined fraction
is $c(M)/8^{M}\sim28\cdot4^{-M}$, and the transfer matrix on the non-singleton states has
spectral radius exactly $2$, so the set of addresses not yet determined, which is where the
window boundaries lie, has box dimension $\log_{2}2=1$ in the internal space $\mathbb{Z}_{2}^{3}$
of dimension $3$.  The two $2$'s are the same number as the linear inflation and the
subdominant orientation eigenvalue, and no link between them is claimed there; Section~\ref{sec:unfold} finds the first-moment sum rule that both obey.  The Spectre rests on
a Pisot unit and an irrational window; Tsiokos's Chair44 rests on an integer inflation and a $2$-adic one.

\subsection{Why Three Dimensions Do Not Automatically Resolve Chirality}

Conjecture~\ref{conj:chirality} rests on a sentence: a planar patch embedded in a
three-dimensional bulk has its reflection realised as an ambient rotation, so parity that is
forbidden on the screen is restored by embedding.  The Chair44 does not touch that sentence.
It touches the reading of the transition as an order parameter for \emph{the emergence of
the third dimension}, which suggests that three dimensions are what restores parity.

\begin{fact}
\label{fact:chairchiral}
The undecorated seven-cube chair has $@sym@$ symmetries, @mir@ of them mirrors, and is achiral.
The decorated Chair44 has only the identity, so it is chiral.  Of the $@raw@$ raw face
contacts, $@mirror@$ place a mirror image against the root, and none is legal: all $@legal@$
legal contacts are proper rotations, so no copy of the tile touches its own mirror image
anywhere in the hierarchy.
\end{fact}

\noindent
The chirality is forced twice over: by the labels, which break the three mirror planes of the
seed, and by the contacts, which admit no mirror pair.  In three dimensions the strict
chirality of the Spectre does not go away.  The escape that Conjecture~\ref{conj:chirality}
offers a planar leaf, turn it over in the bulk, is not available to a tile that already fills
the bulk.  The qualification is therefore this.  Parity is restored, if it is, by the
\emph{codimension} of the leaf and not by the \emph{dimension} of the space, so the
conjecture should be read as a statement about leaves embedded in a bulk and not about
three-dimensional order, and the Spectre-to-Hat transition is not, on this evidence, an order
parameter for the third dimension as such.

This is a qualification and not a refutation.  The conjecture says nothing about tiles of
the bulk, the Chair44 is not proposed as a substrate, and three caveats attach.  The einstein
property is Tsiokos's claim.  The forced chirality was tested on all six substitutions
that leave twelve height classes, not only the published one: none has a legal mirror contact,
so it does not depend on the selection rule, but a decoration outside that family is not
excluded.  And one example is one
example: a three-dimensional aperiodic tile whose legal contacts include a mirror image
would show that forced chirality is not a feature of dimension three, and
\code{symchair44.legal\_contacts} will test any labelled candidate for it.

This section does not show that the Chair44 is an einstein monotile, or aperiodic, or that the pyramid
decoration is the only one that works.  It re-derives its atlas and finds it self-consistent
and equal to the published one.  A numerical scan of sliding and tilted placements
of smoothed variants, in \code{chair44\_rigidity.py}, found no non-lattice contact, which is
evidence for rigidity and not a proof of it.
""", V=g('chair_mesh').value[0], E=g('chair_mesh').value[1], T=g('chair_mesh').value[2],
        tilings=g('chair_tilings').value, raw=A['raw'], mirror=A['mirror'], legal=A['legal'],
        realized=A['realized'], neq=g('chair_equations').value[0], orank=orient['rank'], eqrank=g('chair_equations').value[1],
        ker=g('chair_equations').value[2], charpoly=r'x^{3}(x-8)(x-2)^{20}',
        Db='%.4f' % float(g('boundary_dimension').decimal), sstar='%.4f' % float(g('balanced_exponent').decimal),
        sym=c['undecorated'], mir=c['undecorated_improper'],
        coinc=g('chair_coincidence').value['coincident'],
        reach=g('chair_coincidence').value['reachable_sets'],
        nons=g('chair_coincidence').value['non_singleton'])




def unfolding_section():
    """Unfolding the Chair44 and comparing it with the Spectre.  Every number is a Fact of
    chair44facts.py: chair_twist, chair_nets, chair_slices, orientation_sums, power_slopes."""
    g = F.get
    tw = g('chair_twist').value
    nt = g('chair_nets').value
    sl = g('chair_slices').value
    osum = g('orientation_sums').value
    pw = g('power_slopes').value
    sp_ratio = ', '.join('%.4f' % x for x in osum['spectre_ratio'])
    commas = lambda n: '{:,}'.format(n).replace(',', '{,}')
    return fill(r"""
\section{Unfolding the Chair44 and Comparing It with the Spectre}
\label{sec:unfold}

Section~\ref{sec:chair} compared the two tilings by their combinatorics.  This section brings
the three-dimensional tile down to two dimensions in the two ways that are exact, and then asks
the same questions of both tilings.  One way unfolds the surface of the chair into a flat net;
the other slices and projects the hierarchy.  To compare the two tilings each tile is reduced to
a point (the centroid of a Spectre; the centre of the bounding cube of a chair) and an
orientation vector ($e^{i\theta}$ for a Spectre, the direction $d=g(1,1,1)$ of the missing corner, the
notch, for a chair).  Every number is a Fact (\code{chair\_twist}, \code{chair\_nets},
\code{chair\_slices}, \code{orientation\_sums}, \code{power\_slopes}) computed by
\code{chair44\_unfold.py}, \code{chair44\_slice.py} and \code{chair44\_findings.py}; the Spectre
sums use the patches of \code{spectre.py} through level $5$, which has $@N5@$ tiles.  The same
programs are in the repository of curved-tile tools~\cite{HartshornTools}.

\subsection{The twist of the Chair44 supertile}

Fact~\ref{fact:twist} computed the twist of the Spectre supertile.  The same question for the
Chair44 has a simpler answer.

\begin{fact}
\label{fact:chairtwist}
The eight chairs of the supertile fill the doubled chair cell for cell, so the inflation is a
scaling by $2$ with no rotation and no mirror: its twist is $0$.  The eight child frames are
proper rotations through $@angles@$ degrees, two of each, and their mean matrix is exactly
$I/4$, so the mean frame over the $8^{n}$ chairs of a level-$n$ supertile is $I/4^{n}$.  At depth three all
$@group@$ rotations of the cube occur.
\end{fact}

\noindent
The Spectre supertile is turned by $\theta_{S}=11.565^{\circ}$ and mirrored at every level
(Fact~\ref{fact:twist}), and $\theta_{S}$ is irrational as an angle; every child rotation of the
chair is a rational multiple of $\pi$, so no irrational twist can arise from this substitution.  The mean
matrix $I/4$ is the statement $\sum_{i}H_{i}=2I$: the eight child rotations add up to the linear
inflation times the identity.  This is a first-moment sum rule, and it is the only structure found
here that links the two $2$'s noted at the end of Section~\ref{sec:chair}; whether it is related to
the repeated eigenvalue $2$ of the orientation matrix is not shown.

\subsection{Unfolding the surface into a net}

The surface of the chair is a sphere made of $@F@$ unit squares, with $@V@$ vertices and
$@E@$ edges, whose folds are $@convex@$ convex, $@flat@$ flat and $@concave@$ concave.  Cutting it along a
spanning tree of its face-adjacency graph, $23$ cuts, and folding every face into the plane gives a
$24$-omino.  There are $@trees@$ spanning trees (the matrix-tree theorem, exact).  Of $@samples@$
random trees, $@valid@$ give a valid net and $@over@$ overlap themselves; the valid nets have
perimeters from $@minper@$ to $50$, and $@tilers@$ of them tile the plane by translation.  The smallest
perimeter found is a sampled value and not a minimum (any $24$-omino has perimeter at least $20$).  The
control is the same code on a single cube, over all $@cubetrees@$ of its spanning trees: it returns exactly the
$@cubenets@$ nets of the cube.  Figure~\ref{fig:chairnets} shows three nets and the perimeters.
@FIGNETS@

The supertile surface is the chair surface scaled by $2$, so its net is the chair net with each
square replaced by a $2\times2$ block: the area grows by $4$ and the length by $2$, which is the
dimension $D_b=2$ already found for the boundary (Section~\ref{sec:chair}).  Unfolding therefore does not
produce a planar analogue of the Spectre's fractal outline, of dimension $@Db@$.  That is a
negative result, and the exact boundary count had already implied it.

\subsection{Slices, projections and the notch vector}

A layer $z=c$ of the depth-$n$ patch cuts each chair it meets into a square of $4$ cells or an
L-tromino of $3$ (the cross-sections found are @xs@ cells), so a slice is a tiling of a planar region by
polyominoes.  The projection of the chair centres along $z$ has, by the projection theorem, the
$k_{z}=0$ plane of the three-dimensional diffraction as its Fourier transform; this was checked
against a three-dimensional FFT.

\begin{fact}
\label{fact:chairslices}
The $8^{n}$ centres of the bounding cubes of the depth-$n$ patch are distinct integer points on two cosets of $2\mathbb{Z}^{3}$
($@odd@$ and $@even@$ points for $n=1,\dots,4$), and the notch vectors sum to $2^{n}(1,1,1)$.  The
interface length inside the layer $z=0$ is $4^{n+1}-4\cdot2^{n}$ for $n\le4$ ($@iface@$).
\end{fact}

\noindent
The sum of the notch vectors in Fact~\ref{fact:chairslices} is the sum rule of Fact~\ref{fact:chairtwist} applied to the vector
$(1,1,1)$, and it was also counted.  It is far smaller than the sum of $8^{n}$ random signs, which is of order
$8^{n/2}$: the intensity of the notch-weighted structure factor at the origin per tile is
$2^{-n}$ instead of order $1$.  The layer interface grows by a factor tending to $4$ per level,
again dimension $2$.  Figure~\ref{fig:chairslice} shows a slice, the $k_{z}=0$ plane at level $5$, the origin
values, and the median intensity at random wave vectors.
@FIGSLICE@

\subsection{The same questions asked of the Spectre}

The orientation sum of the Chair44 grows by exactly the linear inflation, $2$ per level.  The same sum for the Spectre
can be computed on the placed tiles.

\begin{fact}
\label{fact:orientsum}
For the Chair44 the notch sum is $2^{n}$ exactly ($@csum@$ at levels $2,\dots,5$).  For the Spectre,
$|\sum_{j}e^{i\theta_{j}}|/\lambda^{n}=@sratio@$ at levels $2,\dots,5$ (patches of $@sN@$ tiles), and
the ratio of the last two sums is $@step@$ against $\lambda=2.8059$.  The first is exact and
the second is a measurement over floating-point sums, with no proof.
\end{fact}

\noindent
In both tilings the orientation sum grows like the linear inflation.  The tile counts grow
like $\lambda^{dn}$, so $|S|^{2}/N\propto\lambda^{(2-d)n}$ is constant for the Spectre ($d=2$; the values
are $@si0@$) and $2^{-n}$ for the Chair44 ($d=3$).  On this evidence the two tilings differ in that
quantity through the dimension alone, and not through the details of the tile.  This is an observation about
one choice of orientation weight; it is not a theorem for the Spectre, whose inflation reverses orientation, so
that a sum rule of the form $\sum H_{i}=\lambda I$ is not available to it in the same way.

Where the power goes is asked with a deliberately loose statistic: the median of
$I(k)/N$ over $300$ fixed random wave vectors, with $k$ in the box $[0,2\pi/\ell]^{d}$ for the tile
size $\ell$, against the number of tiles.  It falls in both: as $N^{@s_s@}$ for the Spectre centroids and
$N^{@s_c@}$ for the Chair44 centres, and as $N^{@s_sw@}$ and $N^{@s_cw@}$ when each point carries its orientation
vector.  A falling median with a fixed number of random wave vectors says that the intensity is
concentrated on few wave vectors, as in a Bragg spectrum, and it says that the concentration is
faster for the Chair44 centres, which lie on a lattice with defects.  Figure~\ref{fig:chaircompare}
and Table~\ref{tab:unfold} gather the comparison.
@FIGCOMPARE@

\begin{table}[h]
\centering
\footnotesize
\caption{The two tilings under the same questions (Facts~\ref{fact:chairtwist} and~\ref{fact:orientsum}).  The first four rows are exact; the sum and the
slopes of the Spectre are measurements on placed patches.}
\label{tab:unfold}
\begin{tabular}{lll}
\toprule
 & Spectre & Chair44 \\
\midrule
supertile twist & $11.565^{\circ}$, mirrored each level & $0$, no mirror \\
orientations & $12$ (steps of $30^{\circ}$) & $24$ rotations \\
supertile outline, dimension & $@Db@$ & $2$ \\
tile count per level & $\lambda^{2}\approx7.87$ & $8$ \\
orientation sum & $\approx1.4489\,\lambda^{n}$ & $2^{n}$ exactly \\
$|S|^{2}/N$ & constant, $\approx@si0last@$ & $2^{-n}$ \\
median $I/N$, slope in $N$ & $@s_s@$ (weighted $@s_sw@$) & $@s_c@$ (weighted $@s_cw@$) \\
\bottomrule
\end{tabular}
\end{table}

\subsection{What this does and does not show}

The nets are a sample of a space of $@trees@$ trees, and nothing is claimed about the best
net.  The point and the orientation vector of each tile are choices: another decoration of the same
tiles would weight the sums differently, and the sum rule is a statement about the notch vector, not about
every function of the orientation.  The Spectre patches of \code{spectre.py} are mirror-type placements, and the
sum is unchanged by the global mirror, but the random wave vectors are drawn in a fixed box and so see the mirror
image of the set seen elsewhere.  The median statistic compares two different dimensions at one
scaling of $k$ and is noisy.  None of this decides whether the diffraction measure of either tiling is pure
point: for the Chair44 that rests on the cited $2$-adic argument of Section~\ref{sec:chair}, and for the Spectre it is open here.
""", N5=F.GEOMETRIC_TOTALS[5],
        angles=', '.join('%g' % a for a in sorted(set(tw['angles']))),
        group=tw['group'],
        F=nt['faces'], V=nt['vertices'], E=nt['edges'], convex=nt['folds']['convex'],
        flat=nt['folds']['flat'], concave=nt['folds']['concave'],
        trees=commas(nt['trees']), samples=nt['samples'], valid=nt['valid'], over=nt['overlapping'],
        minper=nt['min_perimeter'], tilers=nt['tilers'], cubetrees=nt['cube_trees'], cubenets=nt['cube_nets'],
        Db='%.4f' % float(g('boundary_dimension').decimal),
        xs=' and '.join('$%d$' % x for x in sl['cross_sections']),
        odd=', '.join(str(x) for x in sl['coset_odd']), even=', '.join(str(x) for x in sl['coset_even']),
        iface=', '.join(str(x) for x in sl['interface']),
        csum=', '.join(str(x) for x in osum['chair_sum']),
        sratio=sp_ratio, sN=', '.join(str(x) for x in osum['spectre_N']), step='%.4f' % osum['step'],
        si0=', '.join('%.4f' % x for x in osum['spectre_i0']), si0last='%.2f' % osum['spectre_i0'][-1],
        s_s='%.2f' % pw['spectre']['slope'], s_c='%.2f' % pw['chair']['slope'],
        s_sw='%.2f' % pw['spectre_weighted']['slope'], s_cw='%.2f' % pw['chair_weighted']['slope'],
        FIGNETS=figure_block('chair44_nets', 'fig:chairnets',
            r"""Unfolding the surface of the chair.  Three nets (colour is the axis of the face normal, strong
for the outward $+$ side): the smallest perimeter among the sampled valid nets, one of the
nets that tile the plane by translation, and a median one; and the perimeters of all the valid nets in the sample."""),
        FIGSLICE=figure_block('chair44_slice', 'fig:chairslice',
            r"""Slices and projection of the hierarchy.  (a) A depth-$3$ layer, coloured by tile.  (b) The $k_{z}=0$ plane
of the diffraction of the level-$5$ chair centres, which is the Fourier transform of their projection.  (c) The squared structure
factor at the origin: the centres (Bragg, $8^{n}$), random signs ($3\cdot8^{n}$) and the notch vectors ($3\cdot4^{n}$).  (d) The median of $I/N$ at
$300$ random wave vectors, unweighted and weighted by a notch component."""),
        FIGCOMPARE=figure_block('chair44_compare', 'fig:chaircompare',
            r"""The same questions for both tilings.  (a) The orientation sum over the linear inflation to the $n$: constant for both.
(b) $|S|^{2}/N$, the intensity at the origin per tile of the orientation weights: constant for the Spectre, $2^{-n}$ for the chair.
(c) The median of $I(k)/N$ at $300$ random wave vectors against the number of tiles, unweighted and weighted."""))


TITLE = 'Exact Substitution Arithmetic of the Spectre Monotile, with a Three-Dimensional Control'


def abstract():
    return fill(r"""
\begin{abstract}
The substitution of the Spectre monotile is studied as an exact object.  Its Perron eigenvalue is
the fundamental unit $\lam^{2}=@perron@$ of $\mathbb{Z}[\sqrt{15}]$, its zero sector is not
semisimple, and two marginal integer charges track handedness.  The supertile perimeter obeys a
separate recurrence, with the closed form $P(k)=2F_{3k+5}+4$, whose growth
$\nu=\varphi^{3}$ lies in $\mathbb{Q}(\sqrt5)$: area and perimeter of one tile inflate in
different quadratic fields, and the boundary is fractal, of dimension $@dim@$.  As a
three-dimensional control the same quantities are computed for the Tsiokos einstein tile Chair44,
rebuilt from scratch.  Its boundary is an ordinary surface, and its chirality is forced by its
contacts, which qualifies a conjecture that a third dimension restores parity.  Every number is a
Fact computed once and classified by the evidence behind it, and the closed ones are proved in
Lean~4.
\end{abstract}
""", perron=tex(F.get('perron').value), dim='%.3f' % float(F.get('boundary_dimension').decimal))


def introduction():
    return r"""
\section{Introduction}
\label{sec:intro}

The motivation is a question about the initial state of the universe.  Work on the
Nariai horizon, the de~Sitter spacetime $dS_2\times S^2$ whose bifurcation
spheres are compact, has related the Araki--Uhlmann relative entropy of a
coherent excitation to the modular-weighted boost-energy flux through the
horizon, and the $s$-wave sector of the horizon dynamics, linearized
Jackiw--Teitelboim gravity, is consistent with the area law
$S_{\rm rel}=\delta A/4G$ \cite{hartshorn2026}.  That is the converse direction of the result of
Dorau and Much~\cite{DorauMuch}: the Einstein equations imply the area law.  It
does not derive the normalization, whose coefficient is the Newton constant,
and the sign of $\delta A$ depends on the boundary conditions.  Nothing in the
present paper depends on that result; it is the setting in which the question
arises.  The thesis is that the Big-Bang or Penrose CCC crossover surface --- is a two-dimensional
conformal carrier with vanishing configurational entropy density --- is a
substitution monotile phase \cite{SmithEtAl, SmithEtAl2}.  That identification
is a conjecture and is labelled as one.  What can be done without it, and what
occupies most of what follows, is to compute the substitution's exact structure:
the identification is only worth making if the object identified has content,
and the content is arithmetic.

Statements are stratified by epistemic weight.  \emph{Facts} are computations
or theorems.  \emph{Propositions} are computations with a modelling step.
\emph{Conjectures} are sharp claims not yet proved.  A reader who rejects every
conjecture here still keeps every Fact, and the Facts are the reason to read on.
Each Fact is classified by the kind of evidence behind it --- derived from the
quoted substitution, counted on a placed tiling, recomputed by a second
route, quoted from the literature, or carrying a modelling premise --- and the
evidence ledger, which follows the Lean section, says which.  The Facts that are closed
statements are not merely computed twice.  They are proved: \code{spectrelean.py}
emits them as Lean~4 and the kernel checks them, which is a different kind of
evidence from a second computation and is reported separately in
Section~\ref{sec:lean}.

The first result is the substitution itself.  The Spectre inflates by the
fundamental unit of $\mathbb{Z}[\sqrt{15}]$, a Pisot number, so that the frequencies of its
tiles are irrational and its diffraction can be pure point
(Section~\ref{sec:spectral}).  Its transfer matrix has a zero sector that is not
semisimple, which means one transient direction survives a single substitution and
dies only after two.  Two marginal eigenvalues carry exact integer charges, one conserved
and one alternating, and the alternating one tracks global handedness from species
counts alone (Section~\ref{sec:charges}).  The census is stated twice, as a matrix count and
as a placed-tile count, and the relation between them is derived rather than stipulated
(Section~\ref{sec:census}): the row of the Mystic tile is all ones, so it contributes exactly
the previous generation.  The tile area is an exact quadratic form, the order parameter that
interpolates between the Spectre and the Hat is in closed form in $\mathbb{Q}(\sqrt{15})$
(Section~\ref{sec:deform}), and the Seifert minors of the torus knots $T(2,k)$ are in closed
form for every $k$ (Section~\ref{sec:knots}).

The second result is that area is not the only thing the substitution grades.  The perimeter of a
supertile obeys an exact order-three recurrence of its own, derived from the slot-pair contacts
of the rule and not fitted to counts, and its growth is $\nu=2+\sqrt5=\varphi^{3}$, a unit of
$\mathbb{Q}(\sqrt5)$, where the area grows by a unit of $\mathbb{Q}(\sqrt{15})$
(Section~\ref{sec:bdyop}).  One tile therefore carries two units, and since $\nu$ exceeds the
linear inflation the boundary is fractal.  The resemblance of $\varphi^{3}$ to the Hat's
$\varphi^{4}$ is shown to be a Lucas identity and not a shared operator.  The perimeter has a
closed form in Fibonacci numbers, its boundary edges split into direction classes with
Fibonacci totals, and a single balanced exponent weighs the two units against each other.
A fractal dimension of about $1.40$ is also the dimension at which the folded boundary between
cortex and white matter is measured, and the resemblance is recorded in
Section~\ref{sec:bigpicture} as an observation on which nothing here rests.

The third is a control.  Everything above is planar, and Section~\ref{sec:chair} puts the same
questions to the three-dimensional einstein tile Chair44, which is rebuilt from scratch.
The interface law carries over and the form of the exponents carries over, but the fractal
boundary does not (its surface has dimension exactly $2$), the Pisot route to a pure-point
spectrum is replaced by a $2$-adic one (its inflation is the integer $2$), and its chirality, forced by its
contacts, qualifies the conjecture that embedding in a third dimension restores parity.
Section~\ref{sec:unfold} then brings the Chair44 down to two dimensions, by unfolding its surface and by slicing and projecting its
hierarchy: its supertile has no twist, its orientation sum grows by exactly the linear inflation, as the Spectre's does to four digits, and
the two tilings differ in the intensity of that sum through the dimension alone.

The remainder places the Spectre next to the Hat.  Its one-parameter family
$\mathrm{Tile}(a,b)$ contains the Hat at $(1,\sqrt3)$ and the Turtle at $(\sqrt3,1)$, and
those two points, unlike the Spectre at $(1,1)$, tile on a hexagonal lattice with a torus
$\mathbb{C}/\mathbb{Z}[\zeta_3]$.  The cusp-fibre subsection (Conjecture~\ref{conj:six})
follows the Mystic tiles into that torus, finds six fibre classes on the levels computed and
a torsion dictionary that explains the number six.  The two checks that follow, one in real
space (Section~\ref{sec:energy}) and one in reciprocal space (Section~\ref{sec:diffraction}),
are measurements on finite patches, labelled \emph{Computation, not Fact}, and say the same
thing in two languages: the Hat and Turtle see their lattice and the Spectre does not.  The
collection of mathematics results associated with OpenAI~\cite{OpenAIMath} enters only as a
benchmark for the energy floor, with its own stated caveats, and no result here depends on it.

What the paper does not claim is as much a part of it as what it does.  The substitution
matrix is a transcription of the published rule and is the one quoted input.  The
Chair44's einstein property and aperiodicity are its author's claimed proof, quoted and not
checked.  The identification with a horizon, the unit postulate behind the birefringence
number, and the Weyl-curvature conjecture are modelling steps and are labelled as such.
"""


def bigpicture_section():
    return r"""
\section{Implications for Horizons, Spectra, Multi-Scale Geometry}
\label{sec:bigpicture}

The results below are arithmetic: two algebraic units, the fields they
generate, and the operators whose spectra they are.  A reader coming from
physics is owed an account of why arithmetic of this kind is the right thing
to compute, and where each piece of it is consumed.  This section is that
account.  It contains no new Facts; it says what the Facts are for.

\subsection{Why Substitution Rules Generate Units}

A substitution tiling is a rule for replacing every tile by a patch of tiles
similar to it, applied without end.  Counting tiles under the rule is a linear
map, its matrix has a leading eigenvalue, and by Perron--Frobenius that
eigenvalue is an algebraic integer.  For the Spectre it is
$\lam^{2}=%s$, the fundamental unit of $\mathbb{Z}[\sqrt{15}]$
(Section~\ref{sec:spectral}).  Nothing about that number was chosen: it is
forced by the rule, and the rule is forced by the tile.

The unit is Pisot --- its conjugate $%s$ lies inside the unit circle --- and
the Pisot property is what turns geometry into arithmetic.  Bombieri and
Taylor, and Solomyak after them, showed that a Pisot inflation gives a tiling
whose diffraction is pure point: it scatters like a crystal despite having no
period.  Bellissard's gap-labelling theorem then says that the integrated
density of states of any Hamiltonian on the tiling takes its values in the
frequency module, which for the Spectre is $\mathbb{Z}[1/\lam^{2}]=
\mathbb{Z}[\sqrt{15}]$.  So the field is not decoration.  It is the set of
numbers an experiment on this tiling is permitted to return, whether that
experiment is a diffraction pattern or a spectral gap.

\subsection{Area, Perimeter, and the Quadratic Field Divide}

The Hat, the other aperiodic monotile, inflates in $\mathbb{Q}(\sqrt5)$
(Section~\ref{sec:phases}).  A diffraction measurement can tell the two fields
apart, and the 2026 measurement cited there does.  That is the sense in which
the tiling is a \emph{phase}: two monotiles, two fields, and an observable
that reads which one is present.

What Section~\ref{sec:bdyop} adds is that the second field was already inside
the first tile.  The Spectre's area inflates in $\mathbb{Q}(\sqrt{15})$; its
perimeter inflates by $\nu=2+\sqrt5=\varphi^{3}$, in $\mathbb{Q}(\sqrt5)$.  One
tile, two units.  The consequence for physics is a rule of thumb that the
rest of this paper makes exact: \emph{the field an observable lands in tells
you what kind of quantity it is.}  Counts, densities, areas, integrated
densities of states --- anything that scales with the number of tiles --- are
$\mathbb{Q}(\sqrt{15})$ quantities.  Lengths, interfaces, contact lines,
anything that scales with a boundary --- are $\mathbb{Q}(\sqrt5)$
quantities.  Their ratio, which a multi-scale argument needs, lives in
neither and is a degree-four number in the compositum
$\mathbb{Q}(\sqrt3,\sqrt5)$ (Proposition~\ref{prop:compositum}).

The boundary is also fractal, of dimension $%s$, because $\nu$ exceeds the
linear inflation $\lam$.  A supertile's outline grows faster than its size.
That is the geometric content of ``two units'': a shape whose perimeter and
area do not scale together cannot be assigned a single inflation factor, and
the mismatch is measured by the ratio of the two fields' units.

A fractal with a Hausdorff dimension of $%s$ represents a geometric structure with a complexity that sits precisely between a 1D line and a 2D plane \cite{Kraus2013}.
Side-note: in nature and medicine, fractal dimensions around 1.40 are frequently observed in the highly folded structural boundaries of the human brain (such as the boundary between the cerebral cortex and white matter \cite{bullmore}).

\subsection{Target Programmes for the Substrate}

Three programmes draw on this.  The first is the horizon setting of the
introduction~\cite{hartshorn2026, DorauMuch}.  There the Nariai throat's
crossover surface is a two-dimensional carrier with zero configurational
entropy density, and the conjecture --- labelled as one in
Section~\ref{sec:data} --- is that it is a substitution monotile phase \cite{SmithEtAl, SmithEtAl2}.  
If so, every exact quantity here is an exact quantity of the horizon: its
gap labels, its charges, its two units.  The identification is a modelling
step; the arithmetic is not.

The second is spectral.  The integer charges of Section~\ref{sec:charges},
the X-charge of Section~\ref{sec:mass}, and the gap-label module are all
statements about what a Hamiltonian on the tiling can and cannot do, and they
are exact, so a numerical spectrum that disagrees with them is wrong rather
than approximate.  That is a useful kind of constraint to have.

The third is a control.  Everything above is two-dimensional, and a reader is
entitled to ask which of it is an accident of the plane.  Section~\ref{sec:chair}
puts the same questions to the three-dimensional einstein tile Chair44: the
substitution is rebuilt from scratch, its orientation spectrum is computed, and
its boundary exponents are set beside the Spectre's.  Some things carry over (the
interface law and the form of the exponents) and some do not (the Pisot mechanism
behind a pure-point spectrum, replaced by a $2$-adic one, and the fractal boundary), and the second list is as
informative as the first.

""" % (tex(F.LAM2), tex(F.G_SMALL), F.get('boundary_dimension').decimal, F.get('boundary_dimension').decimal)


def screen_section():
    return r"""
\section{The crossover screen and the two monotile phases}
\label{sec:screen}

A zero-entropy two-dimensional carrier admitting quasi-crystalline long-range
order must, if built from a single tile, be one of the known aperiodic monotile
phases, and the two available phases differ in exactly one theorem-level
respect: the Spectre phase is strictly chiral --- reflections are globally
forbidden --- while the Hat phase requires both handednesses.  An isolated
two-dimensional screen has no ambient notion of reflection: a chirality, once
fixed by the matching rules, is absolute.  A leaf embedded in a
three-dimensional bulk does, since reflection of a planar patch is an ambient
rotation.

\begin{conj}[Chirality relaxation]
\label{conj:chirality}
The admissible monotile phase on an isolated two-dimensional conformal screen
is the strictly chiral Spectre phase; the admissible phase on a leaf embedded
in an emergent three-dimensional bulk is the reflection-admitting Hat phase.
The Spectre-to-Hat transition is a geometric order parameter for the emergence
of the third spatial dimension: parity, forbidden on the screen, is restored by
embedding.
\end{conj}

This is a selection statement, not a dynamics, and the selection criterion ---
can reflections be realized ambiently? --- is exactly the criterion by which
the two phases differ, and by nothing else.  Everything that follows equips the
conjecture with exact structure rather than defending it: an integer census
charge locked to handedness (Section~\ref{sec:charges}), an order parameter
along the Tile$(a,b)$ continuum (Section~\ref{sec:deform}), 
and a second, arithmetic distinction between the phases
(Section~\ref{sec:phases}) that the parity argument does not supply.
"""


def spectral_section():
    return r"""
\section{Exact spectral theory of the substitution}
\label{sec:spectral}

The substitution acts on %s species %s, with $\Gamma$ carrying the Mystic pair
$\Gamma_1+\Gamma_2$, $\Gamma_2=\mathrm{Tile}(b,a)$.  The matrix is the one
quoted input of this paper; everything else is derived from it.
{\setlength{\arraycolsep}{2.6pt}
\begin{equation}
M=%s
\label{eq:M}
\end{equation}}

\begin{fact}[Inflation factor and its arithmetic]
\label{fact:lambda}
The Perron eigenvalue is $\lam^{2}=%s$, the fundamental unit of
$\mathbb{Z}[\sqrt{15}]$: $4^{2}-15\cdot1^{2}=%d$ is the least Pell solution.
The linear inflation factor is
\[
\lam=\sqrt{%s}=\tfrac12(\sqrt6+\sqrt{10})=%s\ldots,
\qquad
(\lam-\lam^{-1})^{2}=%d,
\qquad
(\lam+\lam^{-1})^{2}=%d,
\]
with minimal polynomial $%s$ and splitting field $\mathbb{Q}(\sqrt6,\sqrt{10})$.
The two channel identities are proved in $\mathbb{Z}[\sqrt{15}]$ without
evaluating a square root: each is $\lam^{2}+\lam^{-2}\mp2$, and the
$\sqrt{15}$ parts cancel.  Inflation therefore acts on all area-graded data by
multiplication by a fundamental unit.
\end{fact}

\begin{fact}[Transfer spectrum: a Jordan pair among the zero modes]
\label{fact:spectrum}
$\chi_M(x)=%s$, and the zero sector carries structure the spectrum alone does
not show:
\[
\operatorname{rank}M=%d,\qquad \operatorname{rank}M^{n}=%d\quad(n\ge2).
\]
The zero eigenvalue has algebraic multiplicity five and geometric multiplicity
four, so $M$ is not diagonalizable.  The transient sector is four census
directions erased in one substitution together with one Jordan pair erased in
exactly two: a one-step afterimage, the only place where the substitution
remembers longer than its eigenvalues admit.  The remaining spectrum is the
Pisot-unit pair $\{\lam^{2},\lam^{-2}\}$ and two marginal eigenvalues $\pm1$.
Deviations from equilibrium contract by $\lam^{-4}=%s\approx%s$ per step,
except along the marginal directions, which are exact integer charges, and
along the Jordan pair, which is gone after two.
\end{fact}
""" % (number(len(F.SPECIES)), species_row(F.SPECIES), matrix_tex(F.M),
       tex(F.get('perron').value), F.get('pell').value,
       tex(F.get('perron').value), sp.N(F.LAM, 8),
       F.get('channel_hex').value, F.get('channel_pent').value,
       tex(F.get('minpoly').value),
       tex(F.get('charpoly').value),
       F.RANKS[1], F.RANKS[2],
       tex(F.get('contraction').value), sp.N(F.LAM ** -4, 6))


def census_section():
    rows = '\n'.join(
        r'%d & %d & %d & %d \\' % (d, m, g, mys)
        for d, m, g, mys in F.census_table())
    return r"""
\section{Matrix Iteration Versus Placed Patch Counts}
\label{sec:census}

A patch can be counted two ways and the difference matters, so it is settled
here before anything uses it.

The \emph{matrix census} is what iterating~\eqref{eq:M} produces: nine species,
one $\Gamma$ entry.  The \emph{geometric census} is what a placed patch
contains, and a placed patch carries the Mystic as well, so it is larger.  The
previous draft quoted the second while computing with the first.

\begin{fact}[The Mystic is the previous generation]
\label{fact:mystic}
Row $\Gamma$ of $M$ is all ones, so $N_\Gamma(n)$ is the entire census at depth
$n-1$; the Mystic is a second copy of it.  Hence
\[
T_{\rm geom}(n)=T(n)+T(n-1),
\]
and both sequences obey $T(n{+}1)=8T(n)-T(n{-}1)$, the recurrence whose
characteristic polynomial is the quadratic factor of $\chi_M$.
\end{fact}

\begin{table}[h]
\centering
\begin{tabular}{rrrr}
\toprule
depth & matrix census & geometric census & Mystic \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{The two censuses. The fourth column is the third minus the second,
and is also the second column one row up --- which is Fact~\ref{fact:mystic}.
Charges are linear forms on the nine-species vector and are evaluated against
the matrix column.}
\end{table}
""" % rows


def charges_section():
    qp = ', '.join(str(int(c)) for c in F.Q_PLUS)
    qm = ', '.join(str(int(c)) for c in F.Q_MINUS)
    return r"""
\section{Conserved and Alternating Marginal Charges}
\label{sec:charges}

\begin{fact}[The marginal charges]
\label{fact:charges}
The left eigenvectors for the marginal eigenvalues $\pm1$, in the species order
of~\eqref{eq:M}, are
\[
Q_{+}=(%s),\qquad Q_{-}=(%s),
\]
satisfying $Q_{+}M=Q_{+}$ and $Q_{-}M=-Q_{-}$ exactly over $\mathbb{Z}$.
Evaluated on the matrix census of Section~\ref{sec:census} at depths one to
four,
\[
Q_{+}=%s,\qquad Q_{-}=%s.
\]
$Q_{+}$ is conserved; $Q_{-}$ alternates.
\end{fact}

\begin{prop}[$Q_{-}$ is the parity's census avatar]
\label{prop:parity}
The handedness census of the placed tiles gives a chirality order parameter
$\chi=\pm1$ at every depth --- the tiling is strictly single-handed at every
scale --- and the sign of $Q_{-}$ alternates in lockstep with it.  Global
geometric parity is computable from species counts alone.  Transcribing the
published placement chain into $\mathbb{Q}(\sqrt3)$ affine arithmetic, every
determinant is exactly $\pm1$ and the common determinant of a depth-$n$ patch
equals $-Q_{-}(n)$ at every depth computed.
\end{prop}

\begin{fact}[The generational ledger]
\label{fact:ledger}
Row $\Theta$ of $M$ is the indicator of $\Gamma$ and row $\Lambda$ the
indicator of $\Sigma$, so
\[
N_\Theta(n)=N_\Gamma(n-1),\qquad N_\Lambda(n)=N_\Sigma(n-1)
\]
hold exactly, not asymptotically.  Two species are a verbatim copy of an
earlier generation, which gives the modular arrow an integer bookkeeping.
\end{fact}
""" % (qp, qm,
       ', '.join('%+d' % v for v in F.Q_PLUS_BY_DEPTH),
       ', '.join('%+d' % v for v in F.Q_MINUS_BY_DEPTH))


def deformation_section():
    return r"""
\section{Deformations - Area Forms, Asymmetry, and Field Splitting}
\label{sec:deform}

\begin{fact}[The area form, and the order parameter in closed form]
\label{fact:area}
The area of $\mathrm{Tile}(a,b)$ is the exact quadratic form
\[
A(a,b)=%s,
\]
an identity in $\mathbb{Q}(\sqrt3)[a,b]$ obtained by symbolic shoelace on the
published vertex construction, not fitted to sample points.  It reproduces the
endpoints $A(1,\sqrt3)=%s$ and $A(\sqrt3,1)=%s$, and gives the Mystic asymmetry
exactly:
\[
A(b,a)-A(a,b)=%s,
\]
which is the precise content of ``only $\Gamma$ moves''.  With the Mystic pair
carrying both areas, the splitting is maximal at the Hat endpoint with exact
value
\[
\max_{r\in[1,\sqrt3]}\bigl|f_\Gamma-f_\Delta\bigr|
=\frac{10g}{8+10g}=%s=%s\ldots,
\qquad g=%s.
\]
The $\sqrt3$ of the geometry cancels and the maximum lies in
$\mathbb{Q}(\sqrt{15})$ alone: the hexagonal world supplies the areas, the
Pisot world the frequencies, and the order parameter forgets the former.
\end{fact}
""" % (tex(F.get('area_form').value),
       tex(F.get('area_endpoints').value[0]),
       tex(F.get('area_endpoints').value[1]),
       tex(F.get('mystic_asymmetry').value),
       tex(F.get('order_parameter').value),
       F.get('order_parameter').decimal,
       tex(F.G_SMALL))


def phases_section():
    return r"""
\section{Trace Accounting and Quadratic Field Separations}
\label{sec:phases}

Conjecture~\ref{conj:chirality} distinguishes the two monotile phases by
parity: the Spectre phase is strictly chiral, the Hat phase admits both
handednesses.  There is a second distinction, arithmetic rather than
geometric, and it has not been used before.

\begin{fact}[The phases are graded by different fields]
\label{fact:phases}
The Hat is generated by a substitution on four metatiles $H,T,P,F$, not nine
\cite{SmithEtAl}.  Its substitution matrix, in the order $H,T,P,F$ and up to
transposition (which does not change the characteristic polynomial), is
$\left(\begin{smallmatrix}3&1&2&2\\1&0&0&0\\3&0&1&1\\3&0&2&3\end{smallmatrix}\right)$,
and it gives
\[
\chi_H(x)=%s,
\qquad
\kappa=\tfrac12(7+3\sqrt5)=\varphi^{4}=%s\ldots,
\]
against the Spectre's $\chi_M(x)=%s$ and $\lam^{2}=%s=%s\ldots$.  The two
characteristic polynomials share the factor $(x-1)(x+1)$ and differ in a single
coefficient of the quadratic --- seven against eight --- and that coefficient
is the discriminant.  The Spectre is graded by $\mathbb{Q}(\sqrt{15})$ and the
Hat by $\mathbb{Q}(\sqrt5)$.
\end{fact}

\begin{fact}[Trace accounting, in two fields]
\label{fact:trace}
In both phases the marginal pair $\pm1$ cancels and the zero modes contribute
nothing, so the trace is the Perron unit plus its conjugate:
\[
\operatorname{tr}M=\lam^{2}+\lam^{-2}=%d,
\qquad
\operatorname{tr}H=\kappa+\kappa^{-1}=%d.
\]
The same accounting, run in $\mathbb{Q}(\sqrt{15})$ and in
$\mathbb{Q}(\sqrt5)$.
\end{fact}

\subsection{Aperiodicity as irrationality, and which tile witnesses it}

A tiling with a period is a periodic arrangement of a fundamental domain; in a
fundamental domain every species occurs a whole number of times; so in a
periodic tiling every relative frequency is rational.  One irrational frequency
therefore refutes periodicity.  The two phases differ in how much of the tiling
can do the refuting.

\begin{fact}[Every Spectre species is a witness; one Hat metatile is not]
\label{fact:witness}
All %d Spectre frequencies lie in $\mathbb{Z}[\sqrt{15}]$ with nonzero
$\sqrt{15}$ part, so any one of them refutes periodicity.  The Hat's four
metatile frequencies are
\[
f_H=%s,\qquad f_T=%s,\qquad f_P=%s,\qquad f_F=%s,
\]
and the first is rational.  The commonest Hat metatile witnesses nothing, and
the argument has to be run on a rarer one.
\end{fact}

\noindent
That the Spectre frequencies are algebraic \emph{integers} rather than merely
algebraic numbers is what lets the eigenvector equation be checked in
$\mathbb{Z}[\sqrt{15}]$ with no rational arithmetic at all, which is how
Section~\ref{sec:lean} proves it rather than recomputing it.

\subsection{Laboratory Observations of Quasicrystalline Chirality by Moritake}

The Hat phase is no longer just a mathematical object.  Moritake
\emph{et al.} (2026) ~\cite{Moritake2026} fabricated a monotile quasilattice --- the
centroids of the $T$ metatiles of an $H_6$ patch, %s holes etched in
silicon nitride over half a millimetre --- and measured its optical
diffraction.  The pattern shows sharp Bragg peaks whose positions do not move
with the illumination spot, which is the experimental signature of
quasicrystalline order, and it is \emph{chiral}: a pinwheel with no mirror
line, reversing handedness when the structure is mirrored, and responding
differently to left- and right-handed circular polarization.  That last
response is absent in every mirror-symmetric quasicrystal, the Penrose tiling
included.

For this paper the useful part is that they give the twist in closed
form.  The metatiles rotate away from the underlying honeycomb by an angle
whose coefficients follow the Fibonacci recurrence, so the ratio tends to
$\varphi^{2}$ and the limiting twist is
\[
\theta_{\rm chiral}=\arccos\frac{3\varphi-1}{4}
=\arccos\frac{1+3\sqrt5}{8}=%s\ldots^{\circ}.
\]

\begin{fact}[The measured angle is the inflation factor]
\label{fact:chiral}
$3\varphi-1$ and $\varphi^{4}-3$ are the same number, so
\[
\cos\theta_{\rm chiral}=\frac{\kappa-3}{4},
\qquad \kappa=\varphi^{4},
\]
with $\kappa$ the Hat inflation factor of Fact~\ref{fact:phases}.  The angle a
laboratory measured in a diffraction pattern and the Perron eigenvalue of the
substitution matrix are one constant, shifted and scaled.
\end{fact}

\noindent
This matters for Conjecture~\ref{conj:chirality} in a way the conjecture did
not anticipate.  The Hat phase's chirality is now an observed
laboratory property with an exactly predicted magnitude, and that magnitude is
fixed by the arithmetic of $\mathbb{Q}(\sqrt5)$.

It also yields a prediction rather than a restatement.  Substituting the
Spectre's inflation factor into the same formula gives
$(\lambda^{2}-3)/4=%s>1$, which is not a cosine: the Spectre cannot share the
Hat's twist relation.  Its own twist can still be computed, because it needs
only the geometry the substitution already carries: each supertile has a
quadrilateral of four tile vertices, and building the next supertile moves
those four points by an exact map.  The rotation can be read off that map.

\begin{fact}[The Spectre twist]
\label{fact:twist}
Successive supertile quadrilaterals are similar, with ratio $\lambda$ and a
rotation of $-\theta_S$ on even levels and $+\theta_S$ on odd ones, where
\[
\tan\theta_S=\frac{5\sqrt3-6}{13},\qquad
\cos2\theta_S=\frac{4+3\sqrt3}{10},\qquad
\theta_S=%s\ldots^{\circ}.
\]
Applied twice, the map has characteristic polynomial
$(x-1)^{2}(x^{2}-8x+1)$, which carries the quadratic factor of $\chi_M$, and
its dominant eigenvalue is the real number $\lambda^{2}$: over two levels the
quadrilateral is dilated and not rotated.
\end{fact}

\noindent
Three consequences.  First, the guess made in an earlier draft --- that the
Spectre's angle belongs to $\mathbb{Q}(\sqrt{15})$ --- is wrong.  The
$\sqrt{15}$ enters the one-level factor only through its modulus $\lambda$; the
phase lies in $\mathbb{Q}(\sqrt3,i)$, the field of the hexagonal tile
coordinates, and $\cos\theta_S$ has minimal polynomial
$400x^{4}-560x^{2}+169$ over $\mathbb{Q}$.  So the Spectre twist does not test
the field split of Fact~\ref{fact:phases}: that split is carried by the scale,
and the angle is not where it shows.  Second, the alternation of the sign is the
geometric form of a fact already in this paper: the construction mirrors at
every level, so the inflation reverses orientation, consistent with the
alternating charge $Q_{-}$ of Proposition~\ref{prop:parity}.  The Spectre's
twist is a reflection composed with a rotation, where the Hat's, as quoted
above, is a rotation of the metatiles relative to the honeycomb; the two are
not shown to be the same kind of object.  Third, $\theta_S$ is not a
constant of the substitution alone: it is the value at $b/a=1$ of a function of
the tile shape, which gives $%s^{\circ}$ at $b/a=\sqrt3$ and $%s^{\circ}$ at
$b/a=1/\sqrt3$ under the same rule (the Hat tiling is generated by its own
four-metatile rule, so these are not Hat values).

What remains a prediction is narrower than before: that a fabricated Spectre
quasilattice turns by $\theta_S=%s^{\circ}$ between inflation levels with
alternating sign, where the Hat's twist is $%s^{\circ}$.  The twist
computed here is that of the supertile quadrilaterals, and it is checked
against the placed tiles of a supertile, whose area moments rotate by the same
angle.  Whether the centroid set that Moritake \emph{et al.} measured for the
Hat has a Spectre analogue that follows it is not established here, and the
Hat value was not recomputed with this method.

This sharpens the conjecture at no cost.  If the Spectre-to-Hat transition is
the emergence of the third spatial dimension, then it is not only a parity
restoration but a change of the quadratic field that grades all area data ---
and the log-periodic fingerprint changes with it, from period
$\log(4+\sqrt{15})$ on the screen to $\log\varphi^{4}$ in the bulk.  Two
frequencies, named in advance, rather than one.  Whether either is present is a
question for data; that they differ is a Fact.
"""% (tex(F.get('hat_charpoly').value),
      F.get('hat_perron').decimal,
      tex(F.get('charpoly').value),
      tex(F.get('perron').value),
      F.get('perron').decimal,
      F.M.trace(), F.HAT_M.trace(),
      len(F.SPECIES),
      tex(F.HAT_FREQ['H']), tex(F.HAT_FREQ['T']),
      tex(F.HAT_FREQ['P']), tex(F.HAT_FREQ['F']),
      '372{,}100', F.get('chiral_angle').decimal,
      F.SPECTRE_NAIVE,
      F.get('spectre_twist').decimal,
      '%.2f' % F.TWIST_FAMILY[1][1], '%.2f' % F.TWIST_FAMILY[2][1],
      F.get('spectre_twist').decimal, F.get('chiral_angle').decimal)


def knot_section():
    return r"""
\section{The knot layer, for every $k$ at once}
\label{sec:knots}

\begin{fact}[Seifert minors in closed form]
\label{fact:sandwich}
The Seifert form of $T(2,k)$ is the $(k-1)$-dimensional tridiagonal
$(-2,1,1)$.  Its leading principal minors obey $D_{m}=-2D_{m-1}-D_{m-2}$ and
therefore
\[
D_m=(-1)^{m}(m+1),
\]
so they alternate in sign and never vanish.  By Sylvester's criterion the form
is negative definite at every size, giving $\sigma(T(2,k))=-(k-1)$ for every
odd $k$, hence $u\ge(k-1)/2$; one crossing change sends $\sigma^{k}$ to
$\sigma^{k-2}$, and induction closes the sandwich at
$u(T(2,k))=\tfrac{k-1}{2}$.
\end{fact}

\noindent
The closed form is proved by induction in Section~\ref{sec:lean}, which is
what upgrades this from a check at sampled $k$ to a statement about the
family.  Additivity of the signature under connected sum then makes the
selection rule exact arithmetic: $\sigma(K\#K)=-2(k-1)$ forces
$u(K\#K)=k-1$, so same-handed pairs cannot bind, while $\sigma(K\#\overline K)=0$
and the lower bound is lost.  Binding is a mirror-channel phenomenon as a
matter of signatures --- and the strictly chiral screen, having no mirror
channel, cannot bind at all.
"""


def xcharge_section():
    drift = ',\\ '.join('%.3f' % float(F.P_BY_DEPTH[d])
                        for d in sorted(F.P_BY_DEPTH))
    depths = ', '.join(str(d) for d in sorted(F.P_BY_DEPTH))
    return r"""
\section{The X-charge, and the mirror channel}
\label{sec:mass}

Everything so far has been arithmetic on a nine by nine matrix.  The X-charge
is not: it is a statement about which edge of a placed tile meets which edge of
its neighbour, and no eigenvector recovers it.  The tiling has to be built.

Each tile is a fourteen-gon whose slots carry canonical roles in the cyclic
sequence \code{aabbaabbaaaabb} \cite{SmithEtAl}.  A shared edge is an \emph{X-bond} when the two
tiles disagree about the role of the slot they are gluing \cite{LeeMoody}.  This is a statement
about roles and not about lengths: a tiling cannot glue edges of different
length, so typing the bond by realized length makes every bond match by
construction and the charge collapses to zero everywhere.  The two readings
differ on exactly one species, and that difference is the content of what
follows.

\begin{fact}[The contact atlas stabilizes]
\label{fact:atlas}
The unordered adjacency classes --- which species meets which, at which pair of
slots --- number %d at depths %s, with none gained and none lost.  Per-slot
statements are therefore statements about the infinite tiling and not about a
patch.
\end{fact}

\begin{fact}[X-charges, and their parity]
\label{fact:xcharge}
On interior tiles the X-status of every slot is deterministic for every species
but one:
\begin{align*}
&\underbrace{X_{\Gamma_2}=14}_{\text{Mystic, saturated}}
&&\underbrace{X_{\Gamma_1}=X_\Delta=X_\Sigma=4}_{\text{triplet}}
&&\underbrace{X_\Lambda=2}_{\text{splits }\{\Theta,\Lambda\}}\\[1ex]
&\underbrace{X_\Phi\in\{0,2\}}_{\text{flavour doubling}}
&&\underbrace{X_\Theta=X_\Pi=X_\Xi=X_\Psi=0}_{\text{sterile quartet}}
\end{align*}
Identical at every depth computed, and every value even, so the binding
spectrum is $2\delta\{0,1,2,7\}$.  Mystic saturation has a one-line reason:
$\mathrm{Tile}(b,a)$ realizes every canonical role at the other length, so
every bond it forms is a role mismatch.
\end{fact}

\begin{fact}[The flavour frequency is not converged]
\label{fact:pdrift}
$\Phi$'s two mixed slots are perfectly correlated --- never one --- so $\Phi$
splits into flavours $\Phi^{0},\Phi^{2}$.  The collared frequency
$p=\Pr(\Phi^{2})$ measured at depths %s is
\[
p=%s,
\]
still falling.  A value near $0.58$ is what depth four gives; it is a
measurement at a depth, not a limit, and earlier drafts quoted it as though it
were one.
\end{fact}

\begin{conj}[The bulk X-density in closed form]
\label{conj:rho}
If $p\to\tfrac12$ then
\[
\rho_X=\frac{2g}{1+g}=1-\frac{\sqrt{15}}{5}=%s\ldots,
\qquad g=%s,
\]
an element of $\mathbb{Q}(\sqrt{15})$ with no free parameter.  Repeated Aitken
extrapolation of the measured values gives $0.5107$ then $0.5035$, which is
evidence for one half and not a proof of it.  This is the open problem of the
previous draft, reduced from ``compute $p$'' to ``decide whether $p=\tfrac12$''.
\end{conj}

\subsection{The Onset of Non-Additivity: Thresholds in Unknotting Numbers}

Brittenham and Hermiller (2025) ~\cite{BH} proved that unknotting number is not
additive: for $K=7_1=T(2,7)$ with $u(K)=3$,
$u(K\#\overline K)\le5<6$.  Three things about that theorem bear on the
mechanism, and all three are easy to overstate.

\emph{The deficit is bounded, not determined.}  Theirs is an upper bound;
Scharlemann \cite{Scharlemann} gives $u\ge2$; so $\delta=6-u$ lies in $[1,4]$ and its value is
their own Question~4.4, open.  The binding spectrum is parameter-free in its
\emph{ratios} $\{0,1,2,7\}$ --- the X-charges halved --- and says nothing
about the scale.

\emph{The threshold is theirs.}  Their Corollary~1.3 covers $T(2,2k+1)$ for
$k\ge3$, index seven up; $T(2,3)$ and $T(2,5)$ are outside it and whether
either admits any partner is open.  So $k=7$ is where non-additivity is known
to begin, and it is also where the substrate's edge braids close.  That
coincidence is the whole of the mechanism's appeal, and it is a coincidence
until someone explains it.

\emph{The proof was corrected, and the correction was about mirroring.}  The
first version routed through two diagrams of $K15n81556$ asserted to be one
knot; Wang and Zhang (2025) ~\cite{WZ} showed by the Jones polynomial that they are a
chiral knot and its mirror, and verified the inequality directly anyway.  The
current version supplies the missing isotopy and credits them.  The theorem is
unaffected and now stands by two routes, and the deficit bound does not move.
The episode is a caution, not a result: it is \emph{not} evidence for a
mirror-sensitive mechanism, and no proof layer here would have caught it,
because the question is about knot diagrams and this kernel sees arithmetic.

Additivity of the signature then makes the selection rule arithmetic:
$\sigma(K\#K)=-2(k-1)$ forces $u(K\#K)=k-1$, so same-handed pairs cannot bind,
while $\sigma(K\#\overline K)=0$ and the bound is lost.  Binding is a
mirror-channel phenomenon as a matter of signatures, and the strictly chiral
screen has no mirror channel.
""" % (F.ATLAS_CLASSES, depths, depths, drift,
       F.get('rho_x_at_half').decimal, tex(F.G_SMALL))


def boundary_operator_section():
    return r"""
\section{The other unit: a boundary operator in $\mathbb{Q}(\sqrt5)$}
\label{sec:bdyop}

Everything above grades the substitution by area.  The perimeter is graded by
something else, and the two do not live in the same field.

Let $P(k)$ be the number of unit tile edges on the boundary of an order-$k$
supertile.  Reading the child multiset off the substitution table --- the eight
placement slots are the same for every species, and slot~$7$ is $\Gamma$ in
every rule, while $\Gamma$ alone has a null slot and therefore seven children
--- and subtracting each glued edge twice gives, with $Q$ the $\Gamma$
perimeter and $G$ the number of edges shared between two \emph{different}
children,
%%
\begin{equation}
\label{eq:bdyid}
P(k{+}1)=7P(k)+Q(k)-2G(k{+}1),
\qquad
Q(k{+}1)=6P(k)+Q(k)-2G_{\Gamma}(k{+}1).
\end{equation}
%%
The coefficients are read from the rule, not fitted to counts, and~\eqref{eq:bdyid}
is exact at every order computed.  Two structural facts make this small enough
to state.  First, all eight non-$\Gamma$ species have the \emph{same} boundary
sequence: the outline of a supertile does not depend on its type, which is what
collapses a nine-type problem to two.  Second, the glue decomposes by slot pair
into thirteen contact sequences, and every one of them satisfies the same
recurrence as $P$ itself.  The contact set is closed under the operator, which
is why $P$ is.

\begin{fact}
\label{fact:bdyrec}
$P(k)=5P(k-1)-3P(k-2)-P(k-3)$, giving $P=%s$.  Checked by the Lean kernel,
and the sequence itself counted twice --- see Fact~\ref{fact:bdyexact}.
\end{fact}

\noindent
The seed of that recurrence is counted from placed tiles, and the count used
a rounding key: two edges were identified when their endpoint coordinates
agreed to four decimals.  A key of that kind can fail in both directions, by
merging edges that differ below the tolerance and splitting edges that agree
above it, and nothing inside a floating-point computation bounds the failure
--- it can only be removed.

\begin{fact}
\label{fact:bdyexact}
The same sequence, recounted with edges identified by \emph{equality} of
exact $\mathbb{Q}(\sqrt3)$ endpoint coordinates and from an independent
transcription of the placement chain, agrees at every order computed.  The
rounding key is therefore not load-bearing for any of the boundary results.
\end{fact}

\noindent
The factorisation of $\chi_{\partial}$ is not left as a spectral remark.  Its
quadratic factor applied to the sequence annihilates the growing part and
leaves behind the constant belonging to the marginal eigenvalue, and that
identity is proved for every $k$:

\begin{fact}
\label{fact:bdyinv}
$P(k+2)-4P(k+1)-P(k)=-16$ for all $k$, by induction in Lean.  Together with
the seed this is equivalent to Fact~\ref{fact:bdyrec}, and it is the $+1$
eigenvalue made arithmetic --- the boundary's analogue of $Q_{+}$.
\end{fact}

\begin{fact}
\label{fact:bdychi}
The companion matrix of that recurrence --- the boundary transfer operator ---
has characteristic polynomial $\chi_{\partial}(x)=%s$.  The marginal eigenvalue
$+1$ is a conserved boundary quantity, of the same kind as the $\pm1$ of
Section~\ref{sec:charges}.
\end{fact}

\begin{fact}
\label{fact:nu}
The Perron root is $\nu=%s=\varphi^{3}=%s$.
\end{fact}

\noindent
So the area of a supertile inflates by $\lam^{2}=4+\sqrt{15}$ and its perimeter
by $2+\sqrt5$.  This is the $\mathbb{Q}(\sqrt{15})$ / $\mathbb{Q}(\sqrt5)$
divide of Section~\ref{sec:phases} again, but between area and perimeter of the
\emph{same} tile rather than between the Spectre and the Hat.  The supertile
boundary is golden although the tiling is not.

Since $\nu>\lam$ the boundary is fractal: its box dimension is
$\log\nu/\log\lam=%s$, so a perimeter argument assuming $P(k)\sim\lam^{k}$ is
wrong, and wrong by a definite amount.  Both halves of that comparison are
kernel-checked --- $P(k+3)\ge4P(k+2)+30$ for every $k$, and
$\lam^{2}<16$ hence $\lam<4$ --- but the inference between them is not.
Relating an integer growth bound to a real inflation rate needs an order on
$\mathbb{Z}[\sqrt{15}]$ and a square root, neither of which the Mathlib-free
file has, so what Lean certifies is the pair of bounds and the last step is
elementary arithmetic done by hand.

\subsection{What the golden appearance is not}

$\varphi^{3}$ next to the Hat's $\varphi^{4}$ invites the conjecture that the
Spectre's boundary combinatorics \emph{are} the Hat's.  They are not, and the
refutation is worth more than the conjecture.

\begin{fact}
\label{fact:goldenabsent}
$x^{2}-4x-1$ divides neither $\chi_{M}$ nor $\chi_{H}$, and a search over
integer matrices built from $M$ and $H$ --- sums, differences, powers,
transposes, adjugates, integer shifts --- finds it in none of them.
\end{fact}

\begin{fact}
\label{fact:lucas}
$\varphi^{n}$ has minimal polynomial $x^{2}-L_{n}x+(-1)^{n}$ with $L_{n}$
Lucas.  So $x^{2}-4x-1$ \emph{is} $\varphi^{3}$ and $x^{2}-7x+1$ \emph{is}
$\varphi^{4}$: once two quantities are both powers of $\varphi$, their minimal
polynomials are forced to be adjacent Lucas quadratics and the resemblance
carries no information beyond $\mathbb{Q}(\sqrt5)$.
\end{fact}

\noindent
The asymmetry is worth recording: the Spectre's own unit is not golden, since
$\chi_{M}$ contributes the trace $8$ and $8$ is not a Lucas number.  The three
operators sit at traces $8$, $7$ and $4$ with norms $+1$, $+1$ and $-1$.

What survives is weaker and true.  The ratio $\mu=\lam^{2}/\nu$ has minimal
polynomial $%s$ --- degree four, not two --- so it lies in the compositum
$\mathbb{Q}(\sqrt3,\sqrt5)$ and in neither quadratic field alone.

\begin{prop}
\label{prop:compositum}
The Spectre and the Hat meet in $\mu$ as a compositum, not by either field
containing the other.
\end{prop}
""" % (', '.join(str(n) for n in F.get('boundary_recurrence').value[:6]),
       tex(F.get('boundary_chi').value),
       tex(F.get('boundary_growth').value),
       F.get('boundary_growth').decimal,
       F.get('boundary_dimension').decimal,
       tex(F.get('compositum').value))


def lean_section():
    names = L.theorem_names(L.document())
    sample = r'''theorem q_minus_alternates : vecMul Qminus M = scale (-1) Qminus := by decide

theorem pell_fundamental_unit : lam2 * lam2inv = (1, 0) := by decide

theorem seifert_minor_closed_form (m : Nat) : D m = Dclosed m :=
  (D_closed_pair m).1'''
    return r"""
\section{Formal Verification: Machine-Checked Proofs in Lean 4}
\label{sec:lean}

Computing a Fact twice by code-disjoint routes is a good discipline and it is
still two computations; two programs can be wrong in the same way.  The closed
Facts are therefore also \emph{proved}.  \code{spectrelean.py} emits them as
Lean~4 and the kernel checks the file --- %d theorems, covering the charge
eigenvector identities, the census totals and their recurrence, the Mystic
identity of Fact~\ref{fact:mystic}, the arithmetic of $\mathbb{Z}[\sqrt{15}]$
including the Pell unit and both channel identities, the frequency
eigenvector, the X-charge parity, and the Seifert minors of
Fact~\ref{fact:sandwich}.

The file is Mathlib-free: everything is \code{Int}, \code{List Int} or a small
structure over them.
\begin{footnotesize}
\begin{lstlisting}[language=lean]
%s
\end{lstlisting}
\end{footnotesize}

\noindent
%s
""" % (len(names), sample, L.AUDIT_SENTENCE)


def boundary_section():
    return r"""
\section{Conclusion}
\label{sec:boundary}

The exact arithmetic of the Spectre substitution provides a rigorous foundation for aperiodic monotile phases in two dimensions, establishing that area data is graded by the fundamental unit of $\mathbb{Z}[\sqrt{15}]$. While the Hat phase is graded by $\mathbb{Q}(\sqrt{5})$, the Spectre's perimeter is governed by $\nu = \varphi^3$, demonstrating a field division between area and perimeter that forces a fractal boundary of dimension $1.3992$.

These arithmetic structures are complemented by exact results across several layers.
The Substitution Spectra and Charges: The transfer spectrum features a non-semisimple zero sector with a Jordan pair alongside marginal integer charges ($Q_+, Q_-$) that track global handedness.
Topology and Boundaries: Seifert minors of $T(2,k)$ yield closed-form unknotting numbers, and a boundary operator governs both perimeter and contact sequences.
Torsion and Embeddings: When placed on the six-sphere cusp torus, the Mystic tiles form six fibre classes organized as a single orbit under the torsion dictionary $R^6 = 1$.
The Boundary in Closed Form: the perimeter is $2F_{3k+5}+4$ for every $k$, proved in Lean, and its edges split into direction classes with Fibonacci totals that are counted through $k=6$.
The Three-Dimensional Control: the Einstein monotile Chair44 by Ioannis Tsiokos (2026) has an ordinary surface as its boundary, of dimension exactly $2$, a semi-simple orientation spectrum with rational frequencies, and a chirality forced by its contacts. The first two say that the fractal boundary and the Pisot mechanism for a pure-point spectrum belong to the planar tile, the chair's own being $2$-adic; the third qualifies Conjecture~\ref{conj:chirality}, since a tile that fills three-dimensional space does not recover parity merely by living there.

Ultimately, the Spectre substitution by David Smith (2023) bridges exact quadratic field arithmetic, boundary fractal scaling, and topological constraints, offering a fully computable substrate for physical and geometric analysis.

\appendix
\section{Methodological Scope and Limitations}
\footnotesize

Four things are outside the Facts, and they are collected here because they carry different
obligations.  First, the substitution matrix transcribes the published rule.  Every Fact is
downstream of it and none bears on it, so this is the one place the paper asks to be trusted:
a proof about $M$ is a proof about $M$.  It is audited, by a geometric census sharing no code with
the matrix and agreeing at every depth, and an audit is not a proof.  The three seed values of
the perimeter recurrence are in the same position, counted from placed tiles and not derived,
and the second count in exact $\mathbb{Q}(\sqrt3)$ arithmetic removes the rounding key from the
boundary results without removing the transcription.  Second, the Chair44's einstein property
and aperiodicity are its author's (Tsiokos2026) claimed proof \cite{Tsiokos1}, quoted here and not checked, and the forced
chirality of Section~\ref{sec:chair} is tested over the six substitutions of one construction family and not beyond it.  Third,
the modelling steps are modelling steps: Conjecture~\ref{conj:chirality} is a selection
statement and not a dynamics, and the unit postulate of Section~\ref{sec:data} is the whole
of the birefringence number.  Fourth, the boundary operator governs the perimeter, the glue and
the contact sequences in the aggregate; resolved by species and edge index a handful of components
obey neither it nor its obvious extensions, while their ratios still approach $\nu$, so the
extra spectrum is subdominant and cannot move the Perron root.  The aggregate operator is
claimed; the full fine operator is not claimed.  The boundary sequences are read from $k=1$, where
assembly begins, which is a convention made explicit rather than a gap.

"""


def variety_subsection():
    rows = VAR.table()
    body = '\n'.join(
        r'%d & %d & %.4f & %.4f & $%.4f\pm%.4f$ \\' % (n, N, vs, vl, vr, sd)
        for n, N, vs, nl, vl, vr, sd in rows)
    last = rows[-1]
    rel = 100.0 * abs(last[2] - last[4]) / last[4]
    pois = 100.0 * (last[5] - last[2]) / last[2]
    nsig = (last[5] - last[2]) / last[6]
    return r"""
\subsection{A computable precedent: Barbour's variety on substitution patches}

Barbour and Louren\c{c}o define the \emph{variety} of $N$ point masses, with
the masses normalised to sum to one, as $V=\ell_{\rm rms}/\ell_{\rm mhl}$ with
$\ell_{\rm rms}^{2}=\sum_{i<j}m_im_jr_{ij}^{2}$ and
$\ell_{\rm mhl}^{-1}=\sum_{i<j}m_im_j/r_{ij}$~\cite{BarbourLourenco}.  It is
dimensionless, unchanged by translation, rotation, reflection and rescaling,
and in the zero-energy, zero-momentum Newtonian $N$-body problem it grows away
from the Janus point in both time directions~\cite{BarbourLobo,BarbourEntropy}.
The question put to the substrate is only whether the depth-$n$ Spectre patch,
with one equal mass at each tile centroid, shows anything in this measure that
a control does not.  Table~\ref{tab:variety} gives $V$ for the Spectre patch,
for a hexagonal lattice tuned to give the same number of points in the same
region, and for uniform random points in the same region; the region is the
union of the tiles, so that the outer shape of the non-convex patch is held
fixed and is not what is compared.

\begin{table}[h]
\centering
\caption{Barbour's variety on the depth-$n$ Spectre patch and two controls with
the same region and (up to the lattice rounding) the same $N$.  Random: mean and
standard deviation over %d draws.  Floating-point computation
(\texttt{spectrevariety.py}), not an exact Fact.}
\label{tab:variety}
\begin{tabular}{rrrrr}
\toprule
depth & $N$ & Spectre & lattice & random \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}

\textbf{Computation, not Fact.}  The values are floating-point and the
analytic controls (two equal masses, $V=1/8$; an equilateral triangle,
$V=\sqrt3/9$) are the only exact checks.  Three things can be read off.
First, $V$ rises with depth for the Spectre and for both controls alike, from
%.3f at depth 1 to %.3f at depth %d; that rise is the approach of $N$ points
to a continuum value and is not a property of the substitution.  Second, at
depth %d the Spectre value and the lattice value differ by %.2f\%%, which is
the difference between two nearly equal numbers and shows no distinguishing
signal.  Third, the random control lies %.1f\%% above the Spectre, %.0f
standard deviations of its own spread: in this measure the patch sits at the
lattice-like, uniform end and not at the clustered end.

What this does not show is as important.  Depth is not time, and a sequence of
patches of increasing $N$ is not one system evolving; Barbour's growth is a
statement about dynamical solutions, so the rise of $V$ with depth is not an
arrow of time and is not evidence for reading the substitution as a Janus-point
flow.  The computation contributes a well-defined, scale-invariant measure
evaluated on the substrate, and the finding that the substrate is
indistinguishable from the lattice in it.
""" % (len(rows) and VAR.N_RANDOM, body, rows[0][2], last[2], last[0],
       last[0], rel, pois, nsig)


def sixsphere_subsection():
    rows, env = mystic_data()
    body = '\n'.join(
        r'%d & %d & %d & %d & %d--%d \\' % (
            r['n'], r['tiles'], r['mystics'], len(r['classes']),
            min(r['sizes']), max(r['sizes'])) for r in rows)
    r5 = rows[-1]
    share = [len(g) / float(r5['mystics']) for g in r5['classes']]
    lo, hi = 100 * min(share), 100 * max(share)
    e = [sorted(set(env[R])) for R in sorted(env)]
    envtxt = ', '.join(str(x[0]) if len(x) == 1 else '%d--%d' % (x[0], x[-1])
                       for x in e)
    text = r"""
\subsection{A neighbouring exact structure: the cusp fibre of the complex six-sphere}

One exact structure outside this paper touches the tiles, and it is stated here
so that its limits are on the page.  Alp\"oge's construction of a complex
structure on the six-sphere~\cite{Alpoge}, a preprint that is unrefereed,
builds a compact complex threefold fibred by tori over $\mathbb{P}^1$ whose cusp
fibre $W$ is a quotient of a toric surface associated with the $A_2$
triangulation of the plane.

\textbf{Imported fact (computed in \texttt{pyCICY}, not recomputed in this
paper, and not one of its numbered Facts).}  The $[3.4.6.4]$ Laves lattice,
which \texttt{pyCICY.monotile} builds exactly in $\mathbb{Q}(\sqrt3)$, is the
Hasse diagram of the cusp fan: at the regular proportions the $61$ sites of a
two-ring patch are all cells of the $A_2$ triangulation of the matching kind,
and its $102$ bonds are exactly the $102$ cover relations between them.  Under
this identification the lattice rolled up on a torus is $W$: a hexagon centre is
a $\mathrm{dP}_6$ component, an edge midpoint a double curve, a hexagon vertex a
triple point.  Per fundamental domain there is one component, three double
curves and two triple points, so $e(W)=2$ counts the hexagon vertices, and the
dual complex is a triangulated torus ($V-E+F=0$).

\textbf{Which tiles sit on the lattice.}  Not every Tile$(a,b)$ does.  At the
proportions $a=\sqrt3\,b$ of the Hat, the Hat $\mathrm{Tile}(1,\sqrt3)$ is a union
of $8$ kites and the Mystic $\mathrm{Tile}(\sqrt3,1)$, the Turtle, of $10$,
because $A(1,\sqrt3)=8\sqrt3$ and $A(\sqrt3,1)=10\sqrt3$ and a kite has area
$\sqrt3$ (Fact~\ref{fact:area}).  The Spectre does not: $A(1,1)=3+3\sqrt3$ is
$3+\sqrt3$ kites, not an integer.  The corner merge that puts a tile on the
lattice holds only at $a=\sqrt3\,b$.

\textbf{Computation, not Fact: the Mystic fibres.}  Roll the lattice up on
$T=\mathbb{R}^2/\Lambda$, $\Lambda=\mathbb{Z}(3,\sqrt3)+\mathbb{Z}(0,2\sqrt3)$
(six kites per domain).  Every tile of the Hat-point tiling then lies over the
torus, and two Mystics are in the same \emph{fibre class} when they are translates
of each other by a vector of $\Lambda$.  The tiling is aperiodic, so no tile
is a translate of every tile of its kind; the question is how many classes there
are.  \texttt{spectre\_mystic\_fibres.py} computes them on the full levels of the
substitution (not on discs cut from them): every tile vertex is asserted to be a
lattice site, the kite counts $8$ and $10$ are asserted, and the Mystic count of
level $n$ is asserted equal to the matrix census of level $n-1$.  It is
double-precision geometry with keys rounded at $10^{-6}$, hence a computation and
not a Fact.

\begin{table}[h]
\centering
\caption{Mystic fibre classes on the full levels of the Hat-point substitution.
The last column is the range of class sizes.}
\label{tab:fibres}
\begin{tabular}{rrrrr}
\toprule
$n$ & tiles & Mystics & classes & class sizes \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:fibres} and Figure~\ref{fig:fibres} show that from level $3$ on
there are exactly six classes.  They are the six first-edge angles
$30^\circ+60^\circ k$, and the first vertex of every Mystic lies on an edge
midpoint, that is on a double curve of $W$: the angles $30^\circ,210^\circ$ on
one, $90^\circ,270^\circ$ on the second, $150^\circ,330^\circ$ on the third.  At
level~$5$ the six classes hold between @@LO@@\% and @@HI@@\% of the Mystics each
(uniform would be $16.7\%$), and the spread falls with the level.  Within a
class the neighbourhoods are not all alike, and not all different either: the
number of distinct environments of a Mystic, that is of the tiles within
radius $R$ with their offsets, for $R=3.5,\,5.2,\,8$ (in units of the Hat's
short edge, sampling at most $200$ Mystics per class at level $5$) is @@ENV@@,
the same for all six classes.

\begin{conj}[Six classes]
\label{conj:six}
For every level $n\ge3$ the Mystics of the Hat-point substitution fall into
exactly six translate classes modulo $\Lambda$, one for each first-edge angle
$30^\circ+60^\circ k$.
\end{conj}

Conjecture~\ref{conj:six} rests on three data points, levels $3,4,5$, and the class count at
level $2$ is five.  The torus is what makes it a finite statement, since
$\Lambda$-translation is the only equivalence used.  The computation stops at
what it computes: it does not say the six classes are asymptotically equally
populated, only that the shares above are within @@DEV@@\% of uniform at level
$5$, and it says nothing about the Spectre, which is not on this lattice.

@@TORS@@
\textbf{The contrast.}  The substrate of a \emph{periodic} tiling descends to a
torus, which is what $W$ is.  No aperiodic tiling can: it is invariant under no
lattice.  Both objects inherit the lattice from $A_2$; the six-sphere knows
nothing of monotiles, and the relation is a contrast, not an identity of
tilings.  What the rolling-up shows is the middle case: the aperiodic tiling does
not descend, but its Mystic subset has finitely many fibres.  A related but distinct
claim in the mathematics collection~\cite{OpenAIMath}, family~155, is a finite
translational tile in $\mathbb{Z}^3$ that tiles space but admits no fully periodic
tiling; it concerns translational tiles and not the Hat and Spectre, whose tilings use
rotations and reflections, and nothing here depends on it.

\textbf{What does not connect.}  The period field of the construction is
$\mathbb{Q}(\zeta_{12})$, which shares $\mathbb{Q}(\sqrt3)$ with the tile
geometry and nothing with the field $\mathbb{Q}(\sqrt{15})$ of this paper's
inflation or with the Spectre's $\mathbb{Q}(\sqrt6,\sqrt{10})$.  The integer
$6$ occurs on both sides, in the invariant form of the six-sphere's monodromy
and in the Spectre's inflation $\lambda=(\sqrt6+\sqrt{10})/2$, but it arises
differently in each and nothing uses either; the six fibre classes are the unit group $\mu_6$ of
$\mathbb{Z}[\zeta_3]$ acting on the torus (Figure~\ref{fig:torsion}), a third source of the
integer that comes from the hexagonal lattice and is not a link between the two.  The Laves valences $(6,4,3)$ echo
the orbifold orders $(3,4,\infty)$, but the valence-$4$ sites are double curves
over the cusp and not over the order-$4$ point.  All of these are coincidences,
and no link between the two objects should be inferred from them.  The base
orbifold of the six-sphere must be hyperbolic (no Euclidean triangle group
contains both $3$ and $4$), while the substrate's symmetry is Euclidean; and the
threefold carries no holomorphic $3$-form, so it supports no four-dimensional
$\mathcal N=1$ compactification, and nothing here connects it to the screen or
to the Weyl-curvature conjecture above.

"""

    tors = r"""
\textbf{Computation, not Fact: the torsion dictionary.}  The lattice $\Lambda$ is
hexagonal ($|E_1|=|E_2|=2\sqrt3$, at $60^\circ$), so $T=\mathbb{C}/\mathbb{Z}[\zeta_3]$ is the
elliptic curve with $j=0$, and the rotation by $60^\circ$ is, in the basis $(E_1,E_2)$,
the integer matrix $R=\left(\begin{smallmatrix}0&-1\\1&1\end{smallmatrix}\right)$, with
$R^6=1$, $R^3=-1$ and characteristic polynomial $x^2-x+1$.  The six site classes of the
lattice modulo $\Lambda$ are torsion points: the centre is $0$; the edge midpoints are the
three nonzero points $(\tfrac12,0),(0,\tfrac12),(\tfrac12,\tfrac12)$ of $E[2]$, which $R$
permutes in a $3$-cycle and $-1$ fixes; the hexagon vertices are the two nonzero points
$(\tfrac13,\tfrac13),(\tfrac23,\tfrac23)$ killed by $1-\zeta_3$ (with $\zeta_3=R^2$ and
$\det(1-R^2)=3$), which $R$ swaps.  All of this is exact integer and rational arithmetic
(\texttt{spectre\_torsion.py}).  It follows that the six fibre classes, with first-edge
angle $30^\circ+60^\circ k$ and anchor $R^kD_a$, form one orbit of
$\langle R\rangle\cong\mathbb{Z}/6=\mathbb{Z}/3\times\mathbb{Z}/2$: the $\mathbb{Z}/3$ moves the
anchor around $E[2]\setminus0$, and $\mathbb{Z}/2$, generated by $-1$, flips the angle by
$180^\circ$ and fixes the anchor.  That all six occur at levels $3$, $4$ and $5$ is the data;
the group structure is not.
"""
    tors += figure_block('spectre_torsion', 'fig:torsion', FIG_TORSION_CAPTION)
    fig = figure_block(
        'spectre_mystic_fibres', 'fig:fibres',
        r"""Mystic fibres at the Hat point.  (a) Level $3$: $559$ tiles, $63$
Mystics, coloured by first-edge angle.  (b) Class size over its mean at levels
$3,4,5$.  (c) Distinct environments of a Mystic within radius $R$, for the six
classes (the curves coincide).  (d) The share of each angle by level.""")
    dev = 100 * max(abs(x - 1 / 6.0) for x in share) / (1 / 6.0)
    return (text.replace('@@TORS@@', tors).replace('@@ROWS@@', body).replace('@@DEV@@', '%.0f' % math.ceil(dev)).replace('@@LO@@', '%.1f' % lo)
            .replace('@@HI@@', '%.1f' % hi).replace('@@ENV@@', envtxt)
            .replace('Figure~\\ref{fig:fibres} show', 'Figure~\\ref{fig:fibres} show')
            + fig)


def weyl_section():
    return r"""
\section{The initial surface and the Weyl curvature: a conjecture}
\label{sec:weyl}

This section contains no computation.  It states one conjecture, says what
would support it, and lists what it does not claim, because the question it
touches --- what the substrate has to do with gravity --- is the one a reader
will ask next.

\subsection{External Background and Context}

Penrose's Weyl curvature hypothesis asks that the Weyl tensor vanish at the
initial boundary of the universe, so that the gravitational field carries no
free degrees of freedom at the start and the low entropy of the early universe
is the low entropy of the gravitational field~\cite{Penrose1979}.  It is a
hypothesis, not a theorem, and it concerns the Weyl tensor, not the Ricci
tensor: a Weyl-flat initial state still has gravity in the sense of the
Friedmann equations.  In this paper ``no free gravitational field'' is meant,
not ``no gravity''.  That the initial surface is a smooth two-dimensional
carrier and not a point of infinite density is the premise of
Section~\ref{sec:screen}, not something the Weyl hypothesis implies.  The
absence of a time-asymmetric singularity at the start has a precedent in the
Janus-point solutions of the Newtonian $N$-body problem, in which a
scale-invariant measure of structure grows away from a minimum in both time
directions~\cite{BarbourLourenco}; that result is Newtonian and is cited as a
precedent for the shape of the argument, not as support for it.

The sector of general relativity with one chirality of the Weyl tensor
removed is self-dual gravity~\cite{Plebanski1975,Penrose1976}.  A recent
preprint shows that single-minus tree amplitudes of self-dual gravity, which
had been taken to vanish, are nonzero on the half-collinear locus, that is, in
split signature or for complexified momenta, and derives them from a
recursion and, under analyticity assumptions, from an $Lw_{1+\infty}$ Ward
identity~\cite{Guevara2026}.  That preprint is unrefereed.  Two features of it
matter here: it concerns one Weyl chirality, and its kinematics are not
Lorentzian.

\subsection{The conjecture}

\begin{conj}[Chirality of the initial Weyl data]
\label{conj:weyl}
On the isolated conformal screen the strictly chiral Spectre phase carries one
chirality of the gravitational field, the self-dual sector, and the
Weyl-flat initial state is the state in which the other chirality has not yet
switched on.  The transition of Conjecture~\ref{conj:chirality}, from the
Spectre phase to the reflection-admitting Hat phase, is where the second
chirality appears: parity is restored by embedding in the bulk exactly as
both Weyl chiralities are present in the embedded geometry.
\end{conj}

It is a restatement of Conjecture~\ref{conj:chirality} in the language of the
Weyl spinors: parity-forbidden on the screen corresponds to
one-chirality-only in the gravitational sector.  The existing integer data
fit the shape of this statement and do not test it.  The sign of $Q_{-}$
alternates with the handedness of a patch (Proposition~\ref{prop:parity}), and
the X-charge defines a mirror channel that the strictly chiral screen does not
have (Section~\ref{sec:mass}); the conjecture would say that these are two
faces of one fact about which Weyl chirality is available.

@@VARIETY@@
@@SIXSPHERE@@
\subsection{What it does not claim, and what is missing}

\begin{itemize}
\item \emph{No map is given.}  There is no construction taking a tile, a
substitution level or a charge to a Weyl spinor or a self-dual curvature.
Without that the conjecture is a statement about analogy of structure.
\item \emph{The signature does not transfer.}  The amplitude results live in
split signature or complexified kinematics.  Nothing here connects them to a
real Lorentzian initial slice, and the self-dual sector is a complexified
subset of Einstein gravity, not a real cosmological state.
\item \emph{No Standard Model content.}  The substrate supplies integer
charges and a tile-frequency spectrum with degeneracy pattern
$1,1,3,2,2$ on nine species; none of this reproduces the field content of the
Standard Model.  The repository's own mass-ratio scan, run with its
false-positive accounting, returns three matches against about three expected
by chance, and is read as numerology.  The substrate is not claimed to
explain any particle.
\item \emph{No cosmological prediction follows.}  The Weyl hypothesis has
observational consequences only through a full cosmological model, and the
substrate does not supply one.
\end{itemize}

\subsection{What would support or retire it}

The conjecture would be supported by a construction that produces a
one-chirality curvature from the tile data, with the Hat phase supplying the
opposite chirality; and it is retired if the integer chirality data
(Proposition~\ref{prop:parity}) can be shown to carry no relation to the
chirality of any gravitational sector.  Neither has been attempted.
""".replace("@@VARIETY@@", variety_subsection()).replace("@@SIXSPHERE@@", sixsphere_subsection())


def data_section():
    return r"""
\section{Observational Constraints: Cosmic Birefringence and Parity Violations}
\label{sec:data}

Isotropic cosmic birefringence is measured nonzero: the 2026 joint
ACT~DR6~$+$~Planck~PR4 analysis \cite{Eskilt2026} gives
$\beta=0.277^{\circ}\pm0.057^{\circ}$.  Exact parity-evenness, the branch that
would disfavour Conjecture~\ref{conj:chirality}, is not what the sky shows ---
but the consistency is generic, since any parity-violating photon coupling
produces it.

The substrate has one intrinsic chiral quantity with a preferred magnitude,
the order-parameter maximum of Fact~\ref{fact:area}.  Identifying it with a
Stokes-level asymmetry and reading the result in degrees gives
\[
\beta_{\rm pred}=2\cdot\frac{10g}{8+10g}\ \mathrm{[deg]}=%s=%s\ldots^{\circ},
\]
$%s\sigma$ from the measurement.

That agreement is not evidence.  The unit postulate --- that a dimensionless
area fraction is an angle in degrees --- is the entire modelling step and it is
unjustified: degrees are a human convention and nothing in
$\mathbb{Q}(\sqrt{15})$ knows about them.  A closed form landing near a
measurement is what one should expect occasionally from a theory with several
dimensionless constants and a free choice of units.  The substrate supplies a
number, the identification is a guess, and the guess is unrefuted.
At the $0.05^{\circ}$ calibration class it separates from zero at $%.1f\sigma$, and at
$0.01^{\circ}$ from the WMAP$+$Planck value $0.342^{\circ}$ at $%.1f\sigma$. 
Section~\ref{sec:phases} now offers a test that needs no postulate at all.
""" % (tex(F.BETA), F.get('beta').decimal, F.get('beta_tension').value,
       float(sp.N(F.BETA / sp.Rational(5, 100))),
       float(sp.N(abs(sp.Rational(342, 1000) - F.BETA) / sp.Rational(1, 100))))


def facts_appendix():
    rows = []
    for f in F.FACTS:
        mark = r'\checkmark' if f.lean else ''
        # the table is a finding aid, not a transcript: the first sentence of
        # the method is what identifies the route, and the module carries the
        # rest for anyone who wants it
        how = f.method.split(';')[0].split('. ')[0]
        if len(how) > 110:
            how = how[:107].rsplit(' ', 1)[0] + '...'
        # long claims must be allowed to break, or the cell overruns: a
        # p-column will not break inside $...$, so the claim is split at its
        # top-level separators and each piece set as its own math group
        claim = f.claim
        for sep in (r'\qquad', r'\quad', r';\ ', ',\\ '):
            claim = claim.replace(sep, '$ ' + sep + ' $')
        rows.append(r'\code{%s} & $%s$ & %s & %s \\'
                    % (esc(f.key), claim, esc(how), mark))
    return r"""
\appendix
\footnotesize
\section{Every Fact, and how it was obtained}
\label{app:facts}

Generated from \code{spectrefacts.py}.  The last column marks the Facts that
are also proved in Lean, see \url{https://github.com/brentharts/spectre/wiki/Spectre-Lean}; the others are computations.

\setlength{\tabcolsep}{3pt}
\scriptsize
\sloppy
\begin{longtable}{>{\raggedright\arraybackslash}p{3.05cm}%%
>{\raggedright\arraybackslash}p{5.1cm}%%
>{\raggedright\arraybackslash}p{4.6cm}c}
\toprule
Key & Claim & How & Lean \\
\midrule
\endhead
%s
\bottomrule
\end{longtable}
""" % '\n'.join(rows)


BIBLIOGRAPHY = r"""
\begin{thebibliography}{99}

\bibitem{hartshorn2026} B.~Hartshorn
\emph{Gravity from Relative Entropy: Jackiw–Teitelboim Dynamics on the Nariai Horizon}
[preprint]
\url{https://doi.org/10.21203/rs.3.rs-10426867/v1}

\bibitem{DorauMuch} P.~Dorau and A.~Much, \emph{From Quantum Relative Entropy to the Semiclassical Einstein Equations}, Phys.\ Rev.\ Lett.\ \textbf{136}, 091602 (2026); arXiv:2510.24491.
\url{https://doi.org/10.1103/lmq8-nsty}


\bibitem{SmithEtAl} D.~Smith, J.S.~Myers, C.S.~Kaplan and C.~Goodman-Strauss. (2024). 
\emph{An aperiodic monotile}
Combinatorial Theory, 4(1).

\bibitem{SmithEtAl2} D.~Smith, J.S.~Myers, C.S.~Kaplan and C.~Goodman-Strauss. (2024).
\emph{A chiral aperiodic monotile}
Combinatorial Theory, 4(2).

\bibitem{Kraus2013} Y.E.~Kraus, Z.~Ringel, and O.~Zilberberg (2013)
\emph{Four-Dimensional Quantum Hall Effect in a Two-Dimensional Quasicrystal}
\url{https://doi.org/10.1103/PhysRevLett.111.226401}

\bibitem{bullmore}
E.~Bullmore, M.~Brammer, I.~Harvey, R.~Persaud, R.~Murray, M.~Ron (1994)
Fractal analysis of the boundary between white matter and cerebral cortex in magnetic resonance images: a controlled study of schizophrenic and manic-depressive patients
\url{https://doi.org/10.1017/s0033291700027926}

\bibitem{Moritake2026} Y.~Moritake, M.~Takiguchi, T.~Aihara and M.~Notomi (2026).
Chiral diffraction from aperiodic monotile structure.
\emph{Nature Communications} \textbf{17}, 6085.
\url{https://doi.org/10.1038/s41467-026-75023-7}

\bibitem{BH} M.~Brittenham and S.~Hermiller (2025). Unknotting number is not
additive under connected sum. arXiv:2506.24088v2 (15 September 2025).
\url{https://arxiv.org/abs/2506.24088}
Cited in its corrected version; see~\cite{WZ} and the discussion in
Section~\ref{sec:mass}.

\bibitem{WZ} C.~Wang and Y.~Zhang (2025). A remark on the counterexample to
the unknotting number conjecture. arXiv:2507.14265.
\url{https://arxiv.org/abs/2507.14265}

\bibitem{HartshornTools} B.~S. Hartshorn (2026). \emph{spectre-monotile-py}: Python tools for the curved Spectre
monotile, with the Chair44 unfolding, slice and comparison programs of Section~\ref{sec:unfold}.
\url{https://github.com/brentharts/spectre-monotile-py}.
\newline
\bibitem{Tsiokos1} I.~Tsiokos (2026). The Chair44 three-dimensional einstein tile.
\emph{A Strongly Aperiodic Monotile in Three Dimensions}
arXiv:2609.19214 and arXiv:2609.23783.  Code and data:
\url{https://github.com/ioannist/six-birds-tiles} (CC BY 4.0).  
\newline
\url{https://doi.org/10.48550/arXiv.2609.19214}

\bibitem{LeeMoody} J.-Y.~Lee and R.~V.~Moody (2001). Lattice substitution systems and model
sets. \emph{Discrete Comput. Geom.} 25, 173--201.

\bibitem{Schlottmann} M.~Schlottmann (2000). Generalized model sets and dynamical systems.
In \emph{Directions in Mathematical Quasicrystals}, CRM Monograph Series 13, AMS, 143--159.

\bibitem{LMS} J.-Y.~Lee, R.~V.~Moody and B.~Solomyak (2003). Consequences of pure point
diffraction spectra for multiset substitution systems. \emph{Discrete Comput. Geom.} 29, 525--560.

\bibitem{Scharlemann} M.~Scharlemann (1985). Unknotting number one knots are
prime. \emph{Invent. Math.} 82, 37--55.

\bibitem{Penrose1979} R.~Penrose (1979). Singularities and time-asymmetry.
In \emph{General Relativity: An Einstein Centenary Survey}, ed.\ S.~W.~Hawking
and W.~Israel, Cambridge University Press, 581--638.

\bibitem{Plebanski1975} J.~F.~Pleba\'nski (1975). Some solutions of complex
Einstein equations. \emph{J. Math. Phys.} 16, 2395--2402.

\bibitem{Penrose1976} R.~Penrose (1976). Nonlinear gravitons and curved
twistor theory. \emph{Gen. Relativ. Gravit.} 7, 31--52.

\bibitem{Guevara2026} A.~Guevara, A.~Lupsasca, D.~Skinner, A.~Strominger and
K.~Weil (2026). Single-minus graviton tree amplitudes are nonzero. OpenAI
preprint (unrefereed). \url{https://cdn.openai.com/pdf/graviton.pdf}

\bibitem{Alpoge} L.~Alp\"oge (2026). A compact complex threefold fibred by
tori over the projective line, and the six-sphere. Preprint (unrefereed).
\url{https://alpo.ge/s6.pdf}

\bibitem{BarbourLobo} J.~Barbour, F.~S.~N.~Lobo and M.~I.~R.~Louren\c{c}o
(2026). Structural morphology and the gravitational arrow of time.
arXiv:2607.27526.

\bibitem{BarbourEntropy} J.~Barbour (2021). Entropy and cosmological arrows of
time. arXiv:2108.10074.

\bibitem{BarbourLourenco} J.~Barbour and M.~I.~R.~Louren\c{c}o (2026). A
scale-invariant theory of the universe. arXiv:2608.05929.

\bibitem{Eskilt2026} Eskilt, J.R. \emph{et al.} (2026). Joint ACT DR6 and
Planck PR4 constraints on cosmic birefringence.

\bibitem{OpenAIMath} OpenAI (2026). \emph{Mathematics manuscript collection}: 719
manuscripts in 372 families, some with Lean formalizations.  Described by its authors
as produced by an internal OpenAI model and at differing stages of verification.
\url{https://github.com/openai/math} (consulted 15 October 2026).

\bibitem{Kuhne2026}
T.~D.~Kühne, V.~Efremkin, E~.Prodan (2026)
\emph{From Bellissard's gap labeling to atomistic quasiperiodic bilayers}
\url{https://doi.org/10.48550/arXiv.2609.36981}

\end{thebibliography}

\end{document}
"""


def supplement_pointer():
    return r"""
\appendix
\footnotesize

\section*{Declarations}

\subsection*{Funding and Competing Interests}
The author received no specific funding for this work, and declares no competing interests.

\subsection*{Code and Data Availability}
\label{sec:supp}

All source code and data supporting this study, including the Lean~4 sources, are
available at \url{https://github.com/brentharts/spectre/} and
\url{https://github.com/brentharts/CICY/}.
"""


def document():
    text = '\n'.join([
        PREAMBLE, abstract(), introduction(), bigpicture_section(),
        screen_section(),
        spectral_section(),
        census_section(), charges_section(), deformation_section(),
        phases_section(), knot_section(), xcharge_section(),
        boundary_operator_section(), chair_section(), unfolding_section(),
        lean_section(), weyl_section(), data_section(), boundary_section(),
        supplement_pointer(), BIBLIOGRAPHY,
    ])
    text = inject_after_label(text, 'sec:screen', FIG_FAMILY)
    text = inject_after_label(text, 'sec:spectral', FIG_CENSUS)
    text = inject_after_label(text, 'sec:bdyop', FIG_BOUNDARY)
    text = inject_before(text, '\\subsection{What the golden appearance is not}', boundary_additions())
    text = inject_before(text, '\\section{The knot layer, for every $k$ at once}', energy_subsection() + diffraction_subsection())
    return inject_before(text, '\\section{The initial surface and the Weyl curvature: a conjecture}',
                         ledger_subsection())


def write(path=OUTPUT):
    text = document()
    with open(path, 'w') as handle:
        handle.write(text)
    return path


def selftest():
    failures = []

    def check(label, ok):
        print('  %-58s %s' % (label, 'ok' if ok else 'FAIL'))
        if not ok:
            failures.append(label)

    text = document()
    print('structure')
    check('it opens a document', r'\begin{document}' in text)
    check('and closes it', r'\end{document}' in text)
    grouping = text.replace(r'\{', '').replace(r'\}', '')
    check('braces balance', grouping.count('{') == grouping.count('}'))
    for env in ('abstract', 'longtable', 'thebibliography', 'lstlisting'):
        check('%s is closed' % env,
              text.count(r'\begin{%s}' % env) == text.count(r'\end{%s}' % env))
    check('no section is empty', r'\section{}' not in text)

    print('the numbers are the live ones')
    check('the Perron eigenvalue appears',
          tex(F.get('perron').value) in text)
    check('both censuses appear',
          all(str(t) in text for t in F.MATRIX_TOTALS[:5] + F.GEOMETRIC_TOTALS[:5]))
    check('the charge values appear',
          all(('%+d' % v) in text for v in F.Q_MINUS_BY_DEPTH))
    check('the Lean theorem count matches the Lean file',
          str(len(L.theorem_names(L.document()))) in text)
    # The table moved to spectre_supplement.py.  What this document must
    # now guarantee is that it points at the supplement, quotes the same
    # Fact count the supplement carries, and that the supplement really
    # does hold every Fact -- checked here, from this side, so the split
    # cannot silently lose a row.
    import spectre_supplement as SUPP
    supp = SUPP.document()
    check('the table is out of the main paper',
          r'\begin{longtable}' not in text)
    check('the main paper does not depend on the supplement',
          'spectre_supplement' not in text and 'all %d of them' % len(F.FACTS) not in text)
    check('every fact is in the supplement',
          all(esc(f.key) in supp for f in F.FACTS))
    check('the supplement names this paper by its title',
          TITLE in supp)
    # A Fact carrying a `lean` name prints a checkmark in the appendix, so a
    # name with no theorem behind it is the paper claiming a machine check
    # nobody ran.  Six of these were shipped before this check existed.
    _lean_names = set(L.theorem_names(L.document()))
    check('every Lean attribution names a theorem that exists',
          all(f.lean in _lean_names for f in F.FACTS if f.lean))

    print('the caveats survive')
    check('the unit postulate is called unjustified', 'unjustified' in text)
    check('the modelling step is not called evidence',
          'is not evidence' in text)
    check('the conclusion says what the Facts do not reach',
          'What the Facts do not reach' in text
          and 'Four things are outside the Facts' in text
          and 'quoted and not checked' in text)
    check('the transcription is named as the thing taken on trust',
          'asks to be trusted' in text)
    check('the Lean section defers its limits to the boundary section',
          r'does \emph{not} see is the subject of' in text)
    check('the BH deficit is called bounded, not determined',
          'bounded, not determined' in text)
    check('the p drift is stated rather than a single value quoted',
          'not a limit' in text)
    check('the Brittenham-Hermiller reference is cited and listed',
          r'\cite{BH}' in text and 'arXiv:2506.24088' in text)
    check('the Brittenham-Hermiller citation names the corrected version',
          '2506.24088v2' in text)
    check('the Wang-Zhang correction is cited and described',
          r'\cite{WZ}' in text and '2507.14265' in text)
    check('the correction is called a caution, not evidence',
          'not} evidence' in text or 'is a caution' in text)
    check('the measured chiral angle and its identity appear',
          '15.5225' in text and r'\frac{\kappa-3}{4}' in text)
    check('the Moritake reference is cited and listed',
          r'\cite{Moritake2026}' in text and 's41467-026-75023-7' in text)
    check('the Spectre angle is computed, with its closed form',
          r'\tan\theta_S=\frac{5\sqrt3-6}{13}' in text
          and r'\cos2\theta_S=\frac{4+3\sqrt3}{10}' in text
          and str(F.get('spectre_twist').decimal)[:6] in text)
    check('the old guess that the angle is in Q(sqrt15) is withdrawn, not left standing',
          'is wrong' in text and r'400x^{4}-560x^{2}+169' in text
          and 'whatever it is' not in text)
    check('the twist fact states what it does not establish',
          'is not established here' in text and 'not recomputed with this method' in text)
    check('the Spectre angle is still presented as a prediction, now narrower',
          'prediction rather than a restatement' in text and 'narrower than before' in text)
    check('the atlas count and X-charges appear',
          str(F.ATLAS_CLASSES) in text and 'X_{\\Gamma_2}=14' in text)

    print('the boundary operator')
    check('the perimeter sequence is the live one',
          all(str(n) in text for n in F.BOUNDARY_SEQ[:6]))
    check('the boundary characteristic polynomial appears',
          tex(F.get('boundary_chi').value) in text)
    check('the growth rate is named as phi cubed',
          r'\varphi^{3}' in text and F.get('boundary_growth').decimal
          is not None and str(F.get('boundary_growth').decimal)[:6] in text)
    check('the boundary is called fractal',
          'the boundary is fractal' in text)
    check('the derived identity is present and called derived',
          r'\label{eq:bdyid}' in text
          and 'read from the rule, not fitted to counts' in text)
    check('the golden resemblance is refuted, not asserted',
          'divides neither' in text and 'Lucas' in text)
    check('the Spectre unit is said not to be golden',
          'not a Lucas number' in text)
    check('the compositum claim is stated as the weaker survivor',
          'compositum' in text and 'neither quadratic field alone' in text)

    print('the balanced exponent and the closed form')
    check('the Wang-Zhang reference stays distinct and the Kakeya material is gone',
          r'\cite{WZ}' in text and '2507.14265' in text
          and 'Kakeya' not in text and 'WangZahl' not in text)
    check('the balanced exponent is a proposition in the boundary-operator section',
          text.index(r'\label{prop:sstar}') > text.index(r'\label{sec:bdyop}')
          and text.index(r'\label{prop:sstar}') < text.index(r'\label{sec:chair}'))
    check('s* is called exact but not algebraic',
          'not algebraic' in text)
    check('the drifting fit is explained as the estimator',
          'the estimator and not the geometry' in text)
    check('the cortical-folding sentence is kept',
          'cerebral cortex and white matter' in text and r'\cite{bullmore}' in text)
    check('the seed convention is stated in the boundary section',
          'read from $k=1$' in text)
    check('the fine spectrum is left open explicitly',
          'the full fine operator is not claimed' in text)
    check('the boundary recurrence is claimed as kernel-checked',
          'Checked by the Lean kernel' in text)
    print('the big picture')
    check('the substrate section exists and precedes the screen',
          text.index(r'\label{sec:bigpicture}') < text.index(r'\label{sec:screen}'))
    check('it explains why a tiling carries a unit',
          'Perron' in text and 'Pisot' in text and 'Bellissard' in text)
    check('it states the one-tile-two-units rule of thumb',
          'tells' in text and 'what kind of quantity it is' in text)
    check('it names the Chair44 control as the third programme',
          'The third is a control' in text and r'\ref{sec:chair}' in text)
    check('it names the horizon identification as a conjecture',
          'labelled as one' in text)
    check('the area law is stated as a consistency check, not a theorem',
          'converse direction of the result of' in ' '.join(text.split())
          and 'Nothing in the' in ' '.join(text.split())
          and 'a theorem of linearized' not in text)
    check('no unrefereed self-reference is cited or listed',
          all(k not in text for k in ('HartshornJT', 'SuppMonotile', 'SuppNariai',
                                      'Research Square', 'zenodo')))
    check('the Hat substitution matrix is given in the paper itself',
          'smallmatrix' in text and '3&1&2&2' in text)
    check('the availability statement is short and names the repositories',
          text.count('github.com/brentharts/') <= 3 and 'Code and Data Availability' in text
          and 'Every Fact, and how it was obtained' not in text)
    check('the 2-adic route is stated as cited and not re-proved',
          'modular coincidence' in ' '.join(text.split()) and r'\cite{LeeMoody,Schlottmann,LMS}' in text
          and 'is a cited theorem and is not re-proved' in text and 'coincidence\\_closed' in text
          and 'a suggestion and is not established here' not in text)
    check('the title is the new one',
          r'\title{%s}' % TITLE in text
          and 'One Tile, Two Units' not in text
          and 'Torsion, X-Charges, and Boundary Arithmetic' not in text)
    check('the abstract says no result rests on a tolerance',
          'floating-point tolerance' in text)
    check('the perimeter is said to be counted twice',
          r'\label{fact:bdyexact}' in text and 'counted twice' in text)
    check('the rounding key is named as a risk and then as removed',
          'rounding key' in text and 'not load-bearing' in text)
    check('and the transcription is not claimed to be removed with it',
          'without removing the transcription' in text)
    check('the boundary invariant is stated and attributed to induction',
          r'\label{fact:bdyinv}' in text and 'by induction in Lean' in text)
    check('the unformalised last step is named as unformalised',
          'elementary arithmetic done by hand' in text)

    print('the Weyl-curvature section')
    check('it exists and is labelled a conjecture',
          r'\label{conj:weyl}' in text and 'a conjecture' in text)
    check('it sits before the modelling-step section',
          text.index(r'\label{sec:weyl}') < text.index(r'\label{sec:data}'))
    check('it says no map from tile to Weyl spinor is given',
          'No map is given' in text)
    check('it says the signature does not transfer',
          'The signature does not transfer' in text)
    check('it disclaims Standard Model content and reports the scan at chance',
          'No Standard Model content' in text and 'three matches against about three' in text)
    check('the graviton preprint is labelled unrefereed',
          'unrefereed' in text and r'\cite{Guevara2026}' in text)
    check('no gravity vs no free gravitational field is distinguished',
          'not ``no gravity' in text)
    check('its citations all have entries',
          all(('bibitem{%s}' % k) in text for k in
              ('Penrose1979', 'Plebanski1975', 'Penrose1976', 'Guevara2026', 'BarbourLourenco')))

    print('the variety computation')
    check('the variety subsection exists with a table',
          r'\label{tab:variety}' in text and 'Barbour\'s variety' in text)
    check('it is labelled a computation, not a Fact, and floating-point',
          'Computation, not Fact' in text and 'Floating-point computation' in text)
    check('it says depth is not time and no arrow is claimed',
          'Depth is not time' in ' '.join(text.split()) and 'is not an arrow of time' in ' '.join(text.split()))
    check('the table numbers are the live ones',
          ('%.4f' % VAR.table()[-1][2]) in text and ('%.4f' % VAR.table()[-1][4]) in text)
    check('all three Barbour works are cited and listed',
          all((r'bibitem{%s}' % k) in text for k in ('BarbourLourenco', 'BarbourLobo', 'BarbourEntropy'))
          and r'\cite{BarbourLobo,BarbourEntropy}' in text)

    print('the six-sphere remark')
    flat = ' '.join(text.split())
    check('the remark exists and cites Alpoge as unrefereed',
          r'\cite{Alpoge}' in text and 'unrefereed' in flat and r'bibitem{Alpoge}' in text)
    check('it states the 61 sites and 102 bonds',
          '$61$ sites' in flat and '$102$ bonds' in flat)
    check('it is labelled as imported, not one of the numbered Facts',
          'not recomputed in this paper' in flat and 'not one of its numbered Facts' in flat)
    check('it states the periodic-versus-aperiodic contrast',
          'No aperiodic tiling can' in flat)
    check('it says the fields and the shared 6 do not connect',
          'All of these are coincidences' in flat and 'no link between the two objects should be inferred' in flat)
    check('it says the regular proportions a = sqrt3 b are the exact case',
          'regular proportions' in flat and r'a=\sqrt3\,b' in flat)

    print('the figures and the Mystic fibres')
    for nm in ('spectre_family', 'spectre_census', 'spectre_boundary',
               'spectre_mystic_fibres'):
        check('figure %s is included and exists' % nm,
              (r'\includegraphics[width=\textwidth]{%s}' % nm) in text
              and os.path.isfile(os.path.join(HERE, 'paper', 'figures', nm + '.pdf')))
    check('graphicx is loaded and the path is set',
          r'\usepackage{graphicx}' in text and 'graphicspath' in text)
    check('every figure label is referenced',
          all(('\\ref{%s}' % k) in text for k in
              ('fig:family', 'fig:census', 'fig:boundary', 'fig:fibres', 'tab:fibres', 'conj:six')))
    check('the mystic-fibre module passes its own selftest', MF.selftest(depths=(1, 2, 3)))
    mrows, menv = mystic_data()
    by = {r['n']: r for r in mrows}
    check('six classes at levels 3, 4 and 5 and five at level 2',
          [len(by[n]['classes']) for n in (2, 3, 4, 5)] == [5, 6, 6, 6])
    check('the tile totals are the geometric census',
          [by[n]['tiles'] for n in (2, 3, 4, 5)] == F.GEOMETRIC_TOTALS[2:6])
    check('the Mystic counts are the matrix census one level earlier',
          [by[n]['mystics'] for n in (2, 3, 4, 5)] == F.MATRIX_TOTALS[1:5])
    check('the six anchors are the three double curves, two angles each',
          sorted({(g[0]['angle'], g[0]['anchor']) for g in by[5]['classes']})
          == [(30, 'D_a'), (90, 'D_c'), (150, 'D_b'), (210, 'D_a'), (270, 'D_c'), (330, 'D_b')])
    check('the Mystic table row for level 5 is in the text',
          ('5 & %d & %d & 6 & %d--%d' % (by[5]['tiles'], by[5]['mystics'],
                                         min(by[5]['sizes']), max(by[5]['sizes']))) in text)
    check('the six-class claim is a labelled conjecture, not a Fact',
          r'\label{conj:six}' in text and 'Computation, not Fact: the Mystic fibres' in ' '.join(text.split()))
    check('the stale blanket claim about every Tile(a,b) is gone',
          'Every Tile$(a,b)$, the Hat and the' not in ' '.join(text.split()))
    check('the Spectre is said not to be on the lattice',
          '$3+\\sqrt3$ kites' in text)
    check('the README quotes the live Mystic numbers',
          all(str(x) in open(os.path.join(HERE, 'Readme.md'), encoding='utf-8').read()
              for x in (by[5]['mystics'], min(by[5]['sizes']), max(by[5]['sizes']))))

    print('torsion, energy, ledger')
    for nm in ('spectre_torsion', 'spectre_energy', 'spectre_ledger'):
        check('figure %s is included and exists' % nm,
              (r'\includegraphics[width=\textwidth]{%s}' % nm) in text
              and os.path.isfile(os.path.join(HERE, 'paper', 'figures', nm + '.pdf')))
    check('the exact torsion checks pass', TOR.selftest(depths=(3,)))
    check('the energy script passes its selftest', EN.selftest())
    check('the ledger covers every Fact', LED.selftest())
    check('the collection is cited, listed, and hedged',
          r'\cite{OpenAIMath}' in text and r'bibitem{OpenAIMath}' in text
          and 'nothing in this paper depends on it' in ' '.join(text.split())
          and 'withdrawn manuscripts' in ' '.join(text.split()))
    check('the torsion and energy sections are labelled computations',
          'Computation, not Fact: the torsion dictionary' in ' '.join(text.split())
          and r'\label{sec:energy}' in text)
    check('every new label is referenced',
          all(('\\ref{%s}' % k) in text for k in ('fig:torsion', 'fig:energy', 'fig:ledger')))
    en = energy_numbers()
    check('no energy value is below the floor beyond the noise',
          en['lowest'] > -3 * max(en['noise'], 1e-3))
    dn = diffraction_numbers()['sm']
    check('the diffraction section is a labelled computation with live numbers and no exponent',
          'sec:diffraction' in text and 'no exponent is\nclaimed' in text
          and ('%.3f' % dn['Hat']['b']) in text and dn['Spectre']['match']['frac'] < 0.2
          and dn['Hat']['match']['frac'] > 0.7 and dn['Turtle']['match']['frac'] > 0.7)
    check('the README quotes the live ledger and energy numbers',
          all(x in open(os.path.join(HERE, 'Readme.md'), encoding='utf-8').read()
              for x in ('%.3f' % en['g'][('Spectre', 'vertices', 1.0)],
                        '%.3f' % en['g'][('Hat', 'vertices', 1.0)],
                        '%.3f' % en['g'][('Turtle', 'vertices', 1.0)],
                        '%d theorems' % LED.counts()['theorems'], '%d Facts' % LED.counts()['facts'])))


    print('unfolding the Chair44')
    flat = ' '.join(text.split())
    nt = F.get('chair_nets').value
    sl = F.get('chair_slices').value
    osum = F.get('orientation_sums').value
    pw = F.get('power_slopes').value
    import chair44_findings as FD
    import chair44_unfold as CU
    import chair44_slice as CSL
    check('the section is there and referenced from the introduction',
          r'\label{sec:unfold}' in text and text.count(r'\ref{sec:unfold}') >= 3)
    for nm in ('chair44_nets', 'chair44_slice', 'chair44_compare'):
        check('figure %s is included and exists' % nm,
              (r'\includegraphics[width=\textwidth]{%s}' % nm) in text
              and os.path.isfile(os.path.join(HERE, 'paper', 'figures', nm + '.pdf')))
    check('every label of the section is referenced',
          all(('\\ref{%s}' % k) in text for k in ('fig:chairnets', 'fig:chairslice', 'fig:chaircompare',
                                                   'tab:unfold', 'fact:chairtwist', 'fact:chairslices', 'fact:orientsum')))
    check('the net counts are the Facts',
          '{:,}'.format(nt['trees']).replace(',', '{,}') in text and nt['trees'] == 261434325600
          and (nt['faces'], nt['edges'], nt['vertices'], nt['euler']) == (24, 48, 26, 2)
          and nt['cube_nets'] == 11 and nt['cube_trees'] == 384
          and nt['valid'] + nt['overlapping'] == nt['samples'] and nt['min_perimeter'] >= 20
          and ('%d$ give a valid net' % nt['valid']) in text and ('$%d$ overlap' % nt['overlapping']) in text)
    check('the chair twist is zero with I/4 and rational angles',
          F.get('chair_twist').value['scaled_chair'] and F.get('chair_twist').value['mean']
          and set(F.get('chair_twist').value['angles']) == {0.0, 90.0, 120.0, 180.0})
    check('the notch sum is 2^n (1,1,1) and the interface law holds',
          sl['notch_sum'] == [(2 ** n,) * 3 for n in sl['depths']] and sl['interface'] == sl['interface_formula']
          and sl['distinct'] and sl['projection_theorem'] and sl['cross_sections'] == [3, 4])
    check('the chair orientation sum is exactly 2^n and the Spectre ratio is constant to 3 digits',
          osum['chair_sum'] == [2 ** n for n in osum['levels']]
          and max(osum['spectre_ratio'][1:]) - min(osum['spectre_ratio'][1:]) < 1e-3
          and abs(osum['step'] - float(F.LAM)) < 2e-3
          and all(('%.4f' % x) in text for x in osum['spectre_ratio']))
    check('the median slopes are negative, steeper for the chair, and quoted',
          pw['chair']['slope'] < pw['spectre']['slope'] < 0
          and all(('%.2f' % pw[k]['slope']) in text for k in pw))
    check('the quoted linear inflation is the live one',
          ('\\lambda=%.4f$' % float(F.LAM)) in text)
    check('the section says it does not decide pure-point diffraction',
          'None of this decides whether the diffraction measure' in flat)
    check('the Spectre has 12 orientations in the placed patch',
          len({round(float(a) * 180 / 3.141592653589793) % 360 for a in FD.spectre_points(4)[1]}) == 12)
    check('the unfolding programs pass their own tests',
          CU.run_tests(verbose=False) and CSL.run_tests(verbose=False) and FD.check())

    print()
    if failures:
        print('%d failure(s): %s' % (len(failures), ', '.join(failures)))
    else:
        print('spectrepaper: %d characters, %d sections, all checks pass.'
              % (len(text), text.count('\n\\section')))
    return len(failures)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(1 if selftest() else 0)
    elif '--stdout' in sys.argv:
        print(document())
    else:
        print('wrote ' + write())
