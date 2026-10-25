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
import spectre_diffraction as DIF
import chair44facts as C44
C44.register()
import monotilefacts as MFT
MFT.register()
import chair44_diffraction as D
import spectre_orient as O
import orientation_spectra as OS
import contact_operators as CO
import lift4d as L4
import lift4d_matrix as LM

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












def inject_before(text, marker, block):
    assert text.count(marker) == 1, marker
    return text.replace(marker, block + marker, 1)













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
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{orcidlink}
\usepackage{graphicx}
\graphicspath{{paper/figures/}}

\newtheorem{fact}{Fact}
\newtheorem{prop}{Proposition}
\newtheorem{conj}{Conjecture}
\newtheorem{thm}{Theorem}
\newtheorem{lem}{Lemma}
\newtheorem{cor}{Corollary}
\theoremstyle{definition}
\newtheorem{defn}{Definition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newcommand{\lam}{\lambda}
\newcommand{\code}[1]{\texttt{#1}}
\emergencystretch=2em
\hyphenation{re-flec-tion-ad-mit-ting quasi-crys-tal-line}

\title{@TITLE@}

\author{Brent Hartshorn \orcidlink{0009-0004-2853-655X}
        and Tristan Romanov}

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
is proved for all $k$ by induction, not
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
their largest amplitude over $k\le6$ is @peak@.  The group $\mathbb{Z}/6$ is the one that acts on the hexagonal directions of the tiling, and the data identify the group, not the action.

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
is proved is the arithmetic that ties it to the perimeter:
the two totals add up to $P(k)$.  The boundedness of
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

Everything above is two-dimensional; this section asks which of it depends on that.  Tsiokos's
Chair44~\cite{Tsiokos1} is a candidate three-dimensional einstein, claimed to fill $\mathbb{R}^{3}$ only aperiodically.  The
tile is the seven-cube chair (a $2\times2\times2$ block less one corner) with each of its $24$ unit faces carrying eight
small pyramids of signed heights.  Its einstein property and strong aperiodicity are Tsiokos's claimed proof, quoted and
not checked, and nothing below bears on them.  What is recomputed is the combinatorial atlas: \code{symchair44.py} generates
the tile in exact rational arithmetic with no external data and agrees vertex for vertex with the published solid
(\code{symchair44\_tests.py}, \code{chair44\_tests.py}); the Facts are in \code{chair44facts.py} and the supplement.

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
alone.

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
symmetry groups of a tiling & not computed & $1$, $C_{2}$, $C_{4}$, $D_{4}$ (Theorem~\ref{thm:sym}) \\
strict chirality & yes, by the tile & yes, by the contacts \\
spectrum of the tiling & Pisot, pure point & 2-adic, pure point, every tiling (Corollary~\ref{cor:hull}) \\
\bottomrule
\end{tabular}
\end{table}
\end{footnotesize}

\subsection{The boundary operator in three dimensions}

The Chair44 super-tile of depth $N$ has $24\cdot4^{N}$ boundary faces and
$12(8^{N}-4^{N})$ shared faces, since $24$ faces on each of $8^{N}$ tiles less two for every
shared face leaves $24\cdot4^{N}$; the identity holds for all $N$.  The faces point equally in all six axis directions,
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
property of Section~\ref{sec:bdyop} therefore describes the Spectre and not tilings in
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
irrational-frequency witness of aperiodicity (Section~\ref{sec:spectral}) both need an inflation
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
is a cited theorem and is not re-proved, and it concerns the fixed point; that every tiling lies in the
substitution hull, which was left open upstream, is Theorem~\ref{thm:hull} below.  What is added here is a recomputation of the finite part from our own
generator, which agrees on the depth and the address, and a rate.  A subset automaton on the label
sets (only @reach@ are reachable, @nons@ of them with more than one label) counts the digit words
of length $M$ that have not yet fixed the label:
$c(M)=1,8,64,176,400,848,1744,\dots$, that is $c(M+1)=2c(M)+48$ and $c(M)=28\cdot2^{M}-48$
from $M=3$, proved for all $M$ by induction.  The undetermined fraction
is $c(M)/8^{M}\sim28\cdot4^{-M}$, and the transfer matrix on the non-singleton states has
spectral radius exactly $2$, so the set of addresses not yet determined, which is where the
window boundaries lie, has box dimension $\log_{2}2=1$ in the internal space $\mathbb{Z}_{2}^{3}$
of dimension $3$.  The two $2$'s are the same number as the linear inflation and the
subdominant orientation eigenvalue, and no link between them is claimed there; Section~\ref{sec:unfold} finds the first-moment sum rule that both obey.  The Spectre rests on
a Pisot unit and an irrational window; Tsiokos's Chair44 rests on an integer inflation and a $2$-adic one.

\subsection{Why Three Dimensions Do Not Automatically Resolve Chirality}

A common heuristic about parity rests on a sentence: a planar
patch embedded in a three-dimensional bulk has its reflection realized as an ambient rotation, so parity that is
forbidden on the surface is restored by embedding.  The Chair44 does not touch that sentence.
It does touch the suggestion that three dimensions are what restores parity.

\begin{fact}
\label{fact:chairchiral}
The undecorated seven-cube chair has $@sym@$ symmetries, @mir@ of them mirrors, and is achiral.
The decorated Chair44 has only the identity, so it is chiral.  Of the $@raw@$ raw face
contacts, $@mirror@$ place a mirror image against the root, and none is legal: all $@legal@$
legal contacts are proper rotations, so no copy of the tile touches its own mirror image
anywhere in the hierarchy.
\end{fact}

\noindent
The chirality is forced twice over: by the labels, which break the three mirror planes of the seed, and by the contacts, which
admit no mirror pair.  A tile that already fills the bulk cannot be turned over in it, so parity is restored, if it is, by the
\emph{codimension} of the leaf and not by the \emph{dimension} of the space.

This is a qualification of the heuristic and not a refutation of it.  Three caveats attach.  The einstein property is Tsiokos's claim.  The forced chirality was tested on
all six substitutions that leave twelve height classes and does not depend on the selection rule, but a decoration outside that
family is not excluded.  And one example is one example: a three-dimensional aperiodic tile whose legal contacts include a mirror
image would show that forced chirality is not a feature of dimension three (\code{symchair44.legal\_contacts} tests any labelled
candidate).  This section does not show that the Chair44 is an einstein monotile, or aperiodic, or that the pyramid decoration
is the only one that works; it re-derives its atlas and finds it equal to the published one.  A numerical scan of sliding and
tilted placements of smoothed variants (\code{chair44\_rigidity.py}) found no non-lattice contact, which is evidence for rigidity
and not a proof of it.
""", V=g('chair_mesh').value[0], E=g('chair_mesh').value[1], T=g('chair_mesh').value[2],
        tilings=g('chair_tilings').value, raw=A['raw'], mirror=A['mirror'], legal=A['legal'],
        realized=A['realized'], neq=g('chair_equations').value[0], orank=orient['rank'], eqrank=g('chair_equations').value[1],
        ker=g('chair_equations').value[2], charpoly=r'x^{3}(x-8)(x-2)^{20}',
        Db='%.4f' % float(g('boundary_dimension').decimal), sstar='%.4f' % float(g('balanced_exponent').decimal),
        sym=c['undecorated'], mir=c['undecorated_improper'],
        coinc=g('chair_coincidence').value['coincident'],
        reach=g('chair_coincidence').value['reachable_sets'],
        nons=g('chair_coincidence').value['non_singleton'])




def hull_subsection():
    """Every Chair44 tiling lies in the hull, and the symmetry groups of tilings (Romanov).  The counts
    of legal blocks and fixed seeds are Facts of chair44facts.py (chair44_hull.py); the admissible-block
    count and the theorem statements are quoted Facts."""
    hu = F.get('chair_legal_blocks').value
    commas = lambda n: '{:,}'.format(n).replace(',', '{,}')
    return fill(r"""
\subsection{Every tiling lies in the hull, and the symmetry groups of tilings}
\label{sec:hull}

The $2$-adic route above is stated for the fixed point of the substitution.  Tsiokos left two questions open~\cite{Tsiokos1}:
whether \emph{every} tiling by the Chair44 lies in the hull of the substitution, and which symmetry groups its tilings
have.  The decorations force every tiling to be hierarchical: a short Berger-style proof is due to
Goodman-Strauss~\cite{GoodmanStrauss}, simpler equivalent matching rules to Flicker~\cite{Flicker}, and a finite
certificate in every dimension to Kamenetsky~\cite{Kamenetsky}.  This subsection answers the two questions.  The proofs are due to the second
author~\cite{Romanov}; they combine one level of the hierarchy with a finite computation on the eight unit cells around a
lattice vertex.

\paragraph{Set-up.}  A tiling by the tile $Q$ is \emph{registered}: up to an isometry every tile is a chair
$x\mapsto gx+t$ with $g$ one of the $24$ proper rotations $O$ and $t\in\mathbb{Z}^{3}$.  Label each unit cell by the frame of
the chair that covers it and by which of its seven cells it is; there are @labels@ labels, the $24\times7$ of
Section~\ref{sec:chair}.  Write $\sigma$ for the substitution, of inflation $2$.

\begin{defn}
\label{def:block}
A \emph{block} is the labelling of the eight unit cells around a lattice vertex.  It is \emph{legal} if it occurs in
$\sigma^{n}(j)$ for some label $j$ and some $n$, and \emph{admissible} if the chairs it names are consistent and pairwise
disjoint and every face-adjacent pair has a relative pose among the $44$ legal contacts of Section~\ref{sec:chair}.
\end{defn}

\noindent
Two inputs are taken from the literature~\cite{Tsiokos1,GoodmanStrauss} and are not re-derived here.

\begin{description}
\item[(R)] Every tiling by $Q$ is registered, and every pair of tiles sharing a feature is in a relative pose among the
legal contacts~\cite[Thm~5.6, Lemma~5.3]{Tsiokos1}.  Every shared unit face carries features, so all blocks of a tiling by
$Q$ are admissible.
\item[(H)] Every tiling $T$ by $Q$ has a unique parent tiling $D(T)$, again a tiling by $Q$, registered on a lattice of
spacing $2$, and $T=\sigma(D(T))$~\cite[Thms~6.4, 6.5]{Tsiokos1}.
\end{description}

\begin{lem}[computation]
\label{lem:blocks}
There are @adm@ admissible blocks.  For each of them, all $27$ windows of size $2\times2\times2$ in its substitution, a
$4\times4\times4$ block, are legal.
\end{lem}

\noindent
This is the second author's computation (\code{chair44\_vertex\_blocks.py} and \code{chair44\_hull\_prune.py}, on the atlas
published with~\cite{Tsiokos1}).  It has been repeated from our own data: \code{chair44\_hull\_admissible.py} takes the
$44$ legal contacts that \code{symchair44.py} derives from its own labels and finds the same 360{,}189 admissible blocks,
all of whose child windows are legal.  It runs for about ten minutes, so the build of this paper does not repeat it.  The
theorem is its consequence.

\begin{thm}
\label{thm:hull}
Every tiling by $Q$ lies, up to an isometry, in the hull $\Omega_{\sigma}$ of the substitution.
\end{thm}

\begin{proof}
Let $P$ be a finite patch of $T$.  For $n$ large, $P$ lies in $\sigma^{n}(\beta)$ for a block $\beta$ of $D^{n}(T)$.  By~(H),
$\beta$ is a window of $\sigma(\beta')$ for a block $\beta'$ of $D^{n+1}(T)$.  By~(R), $\beta'$ is admissible, so $\beta$ is
legal by Lemma~\ref{lem:blocks}.  Hence $P$ is legal.
\end{proof}

\begin{cor}
\label{cor:hull}
The space of Chair44 tilings is minimal and uniquely ergodic, with pure-point dynamical spectrum.  Every Chair44 tiling is
pure-point diffractive, and the space is the hull of a regular model set with internal space $\mathbb{Z}_{2}^{3}$, so its
structure is limit-periodic.
\end{cor}

\begin{proof}
The substitution is primitive, so its hull is minimal and uniquely ergodic.  The modular coincidence of~\cite[Thm~8.1]{Tsiokos1},
recomputed above, and the theorem of Lee and Moody~\cite{LeeMoody} give pure-point spectrum of the hull~\cite{LMS}.  Theorem~\ref{thm:hull} identifies the hull with the space of all tilings.
\end{proof}

\paragraph{A second route to the finite part.}  The legal blocks depend only on the substitution, so they can be recomputed
from the cell substitution of Section~\ref{sec:chair}, which our generator builds from nothing
(\code{chair44\_hull.py}).  The distinct $2\times2\times2$ windows of $\sigma^{n}(j)$ over all @labels@ labels number
@pd@ for $n=1,\dots,6$, so @legal@ blocks are legal.  The set is closed, in that all $27$ child windows of every one of
its members lie in it, and it contains the windows of $\sigma(j)$; so it is the whole language and not merely the place
where a count stopped.  The central window $\Phi(w)$ of $\sigma(w)$ maps the legal blocks to themselves and has exactly
@fixed@ periodic blocks, all of them fixed.  These are the @fixed@ fixed seeds of~\cite{Tsiokos1}, and their stabilisers in
$O$ have order $1$ for @s1@ of them and $8$ for @s8@.  Both counts agree with the second author's.  Registration~(R), the
parent tiling~(H) and the count of admissible blocks are not part of this recomputation.

\paragraph{Symmetry.}  Write $\mathrm{Sym}(T)$ for the group of isometries that map $T$ to itself.  Tsiokos showed that it
embeds in $O$, so $|\mathrm{Sym}(T)|\le24$, and exhibited orders $1$ and $8$~\cite{Tsiokos1}.

\begin{thm}
\label{thm:sym}
For every tiling $T$ by $Q$, $\mathrm{Sym}(T)$ is conjugate to one of $1$, $C_{2}^{\mathrm{diag}}$ (a half-turn about a
face-diagonal axis), $C_{4}$ and $D_{4}$, and each occurs.
\end{thm}

\begin{lem}[position of symmetry axes]
\label{lem:axes}
Let $T$ be a tiling by $Q$ and $L_{n}=2^{n}\mathbb{Z}^{3}+o_{n}$ the vertex lattice of its level-$n$ supertiles, so that
$o_{n+1}\equiv o_{n}\pmod{2^{n}}$.  Let $g\in\mathrm{Sym}(T)$ have linear part $R\neq I$.
(1) If $R$ is a half-turn or quarter-turn about a coordinate axis, the axis of $g$ is a lattice line of every $L_{n}$.
(2) If $R$ is a half-turn about a face diagonal, the axis of $g$ lies in a coordinate plane through vertices of every
$L_{n}$ and meets every $L_{n}$.
(3) If $\mathrm{Sym}(T)$ is not cyclic, its fixed point is a vertex of every $L_{n}$.
\end{lem}

\begin{proof}
The hierarchy is unique~\cite[Thm~6.6]{Tsiokos1}, so $g$ preserves every $L_{n}$.  Write $g(x)=R(x-c)+c$ with $c$ on the
axis; since $R$ preserves $2^{n}\mathbb{Z}^{3}$, $g$ preserves $L_{n}$ if and only if $(I-R)(c-o_{n})\in2^{n}\mathbb{Z}^{3}$.
(1) Take the axis along $e_{3}$.  For a half-turn $I-R=\mathrm{diag}(2,2,0)$; for a quarter-turn $I-R$ acts on the
$(e_{1},e_{2})$-plane with determinant $2$.  Either way each $\delta^{(i)}_{n}=(c_{i}-o_{n,i})\bmod2^{n}$, $i=1,2$, is $0$ or
$2^{n-1}$ (for $n=0$: $0$ or $\tfrac12$).  If $\delta^{(i)}_{n}=2^{n-1}$ then, since $o_{n+1}\equiv o_{n}\pmod{2^{n}}$,
$c_{i}-o_{n+1,i}\equiv2^{n-1}$ or $3\cdot2^{n-1}\pmod{2^{n+1}}$, neither $0$ nor $2^{n}$, against the condition at level
$n+1$.  So every $\delta^{(i)}_{n}=0$.
(2) For $R(x,y,z)=(y,x,-z)$ the condition reads $(c_{1}-o_{n,1})-(c_{2}-o_{n,2})\in2^{n}\mathbb{Z}$ and
$2(c_{3}-o_{n,3})\in2^{n}\mathbb{Z}$.  The second gives $c_{3}\equiv o_{n,3}$ as in~(1); by the first, $c+s(1,1,0)$ lies in
$L_{n}$ for $s\equiv o_{n,1}-c_{1}$.
(3) A non-cyclic subgroup of $D_{4}$ contains two half-turns about perpendicular coordinate axes, or one about a coordinate
axis and one about a perpendicular face diagonal; its fixed point is the intersection of the two axes, and by (1) and (2)
each of its coordinates is $\equiv o_{n}\pmod{2^{n}}$.
\end{proof}

\begin{proof}[Proof of Theorem~\ref{thm:sym}]
\emph{No element of order three.}  Such an element is a rotation about a line in direction $(1,\pm1,\pm1)$.  The line meets
the interior of a tile, which the rotation would have to fix; that is impossible because $Q$ has no self-isometry
(Fact~\ref{fact:chairchiral}).  So $\mathrm{Sym}(T)$ is a $2$-group, contained in a copy of $D_{4}$.  By
Theorem~\ref{thm:hull} every block below is legal.

\emph{Vertex-centred groups.}  Let $\mathrm{Sym}(T)$ be non-cyclic, with fixed point $x$, a vertex of every $L_{n}$ by
Lemma~\ref{lem:axes}(3), and let $w_{n}$ be the block of $D^{n}(T)$ at $x$.  Then $w_{n}=\Phi(w_{n+1})$.  The substitution
commutes with $O$, so $\mathrm{Stab}(w_{n+1})\le\mathrm{Stab}(w_{n})$, and since the $w_{n}$ cover balls of radius $2^{n}$
about $x$, $\mathrm{Sym}(T)=\bigcap_{n}\mathrm{Stab}(w_{n})$.  The sequence repeats, $w_{a}=w_{b}$ with $a<b$, so $w_{b}$ is
$\Phi$-periodic and $\mathrm{Sym}(T)=\mathrm{Stab}(w_{b})$.  The $\Phi$-periodic legal blocks are the @fixed@ fixed seeds,
with stabilisers $1$ and $D_{4}$; this is the count recomputed above.

\emph{Axial groups.}  Let $\mathrm{Sym}(T)=K$ be cyclic and non-trivial, with axis $A$ in direction $e$, and let $O_{e}$ be
the rotations about $e$.  By Lemma~\ref{lem:axes}, $A\cap L_{n}=v_{n}+2^{n}\mathbb{Z}e$; take $v_{n}$ nearest to a fixed
point of $A$.  Then $v_{n}-v_{n+1}=j_{n}2^{n}e$ with $j_{n}\in\{-1,0,1\}$, and the blocks $w_{n}$ of $D^{n}(T)$ at $v_{n}$
satisfy $w_{n}=W_{j_{n}}(w_{n+1})$, where $W_{j}$ is the window of $\sigma(\cdot)$ centred at $je$.  The elements of $O_{e}$
commute with every $W_{j}$, so $\mathrm{Stab}(w_{n})\cap O_{e}$ decreases to $K$; the full stabilisers may be larger, since
an element outside $O_{e}$ acts about the moving point $v_{n}$.  If $j_{n}=j$ for all $n\ge N$, then $y=v_{N}+j2^{N}e$
satisfies $y-v_{n}=j2^{n}e$ for $n\ge N$, so $y$ is a vertex of every $L_{n}$ and $\mathrm{Sym}(T)$ is $1$ or $D_{4}$ by the
previous case, not $K$.  So the shifts are not eventually constant, and their tail lies in a strongly connected component
carrying two distinct shifts of the graph $G_{K}$ whose vertices are the legal blocks $w$ with
$\mathrm{Stab}(w)\cap O_{e}=K$ and whose edges $w\to W_{j}(w)$ are labelled $j$.  Conversely, a cycle in $G_{K}$ whose
labels $j_{0},\dots,j_{p-1}$ are not all equal unrolls to a tiling in the hull whose axis position
$-\sum_{m}j_{m}2^{m}=J/(1-2^{p})$, $J=\sum_{m<p}j_{m}2^{m}$, is not an integer since $0<|J|<2^{p}-1$; by
Lemma~\ref{lem:axes}(3) its symmetry group is then cyclic about $A$, hence $K$.  A computation finds such components for
$C_{2}^{\mathrm{diag}}$ (14 blocks on each face-diagonal axis) and $C_{4}$ (4 blocks).  For a half-turn about a coordinate
axis the 18 blocks of $G_{K}$, among them blocks whose full stabiliser is $D_{4}$ or a Klein group, form only components of
constant shift.  For each group found, explicit patches of 3{,}800 to 20{,}000 tiles have exactly that symmetry group.
\end{proof}

\begin{remark}
\label{rem:order8}
In particular $|\mathrm{Sym}(T)|\le8$, and the bound is attained.  No Chair44 tiling has a $3$-fold axis, a Klein
four-group of symmetries, or exactly a $2$-fold coordinate axis.
\end{remark}

\noindent
The theorem is about tilings and does not conflict with Fact~\ref{fact:chairchiral}, which is about the tile: the
decorated chair has only the identity, and a tiling by it can still be invariant under a rotation of its whole pattern.
Nor does the bound of $8$ conflict with the $24$ orientations that occur in a depth-three supertile
(Fact~\ref{fact:chairtwist}): by Theorem~\ref{thm:hull} every tiling contains all of them, and it is symmetric under
at most a third of the $24$ rotations.

\textbf{Scope.}  Theorem~\ref{thm:hull}, Corollary~\ref{cor:hull} and Theorem~\ref{thm:sym} are conditional on~(R)
and~(H), and the corollary on the theorem of Lee, Moody and Solomyak; all three inputs are quoted.  The count in
Lemma~\ref{lem:blocks} was made on the upstream atlas and again on our own (\code{chair44\_hull\_admissible.py}); the
axial-group analysis and the explicit patches are the second author's computations.  These are not recomputed in the build
of this paper.  What the build recomputes is the language of legal blocks, its closure, the fixed seeds and their
stabilisers.  None of this bears on the einstein property or on strong aperiodicity of the
Chair44, which remain Tsiokos's claim.
""", labels=hu['labels'], adm=commas(hu['romanov']['admissible']), pd=', '.join(str(x) for x in hu['per_depth']),
        legal=hu['legal'], fixed=hu['fixed'], s1=hu['stabilisers_fixed'][1], s8=hu['stabilisers_fixed'][8])


def lines_subsection():
    """Uniform frequencies hold in boxes, not along lines.  Every number is a Fact of chair44facts.py
    (chair_lines, computed by chair44_lines.py)."""
    ln = F.get('chair_lines').value
    hu = F.get('chair_legal_blocks').value
    tup = lambda t: '(' + ','.join('%d' % x for x in t) + ')'
    rows = []
    for b, cls in ln['corner'].items():
        if b == (1, 1, 1):
            vals = sorted({(k['means']['notch_x'], k['means']['notch_y'], k['means']['notch_z']) for k in cls})
            rows.append(r'$%s$ & %d fixed letters & $%s$' % (tup(b), len(cls), r'$ and $'.join(tup(v) for v in vals)))
        else:
            k = cls[0]
            rows.append(r'$%s$ & one class of %d & $%s$' % (tup(b), k['size'],
                        tup((k['means']['notch_x'], k['means']['notch_y'], k['means']['notch_z']))))
    return fill(r"""
\subsection{Uniform frequencies hold in boxes, not along lines}
\label{sec:lines}

Corollary~\ref{cor:hull} says that every finite patch has a frequency, reached uniformly by boxes of side $L$ in every
tiling.  It is tempting to read this as a statement about any scan of a tiling, for instance the sum of an observable
along a line of cells.  It is not, and this subsection says what is true.  Everything below is an exact finite computation
(\code{chair44\_lines.py}) except the last proposition, which is conditional on the internal space of
Corollary~\ref{cor:hull}.

\paragraph{Box means.}  The matrix of the cell substitution on the @labels@ labels is primitive with every row and column
sum $8$, so each label has frequency $1/@labels@$ in every tiling, and the box mean of a cell observable is its mean over
the labels.  We use two observables of a cell.  The \emph{height} $p$ is the sum of the signed heights of the pyramids on the
faces of the cell; it vanishes on five of the seven cells of the chair and is $-36$ and $+36$ on the other two, since the $192$
heights sum to zero.  The \emph{notch vector} $d=g(1,1,1)\in\{\pm1\}^{3}$ belongs to the chair that covers the cell.  Both have box
mean exactly $0$, in each component.

\begin{prop}[chair stacks]
\label{prop:stack}
Six of the @labels@ labels, namely cell $0$ of the chairs whose notch points along $\pm(1,1,1)$ (three along each sign), satisfy
$\sigma(j)_{000}=\sigma(j)_{111}=j$.  For each of them and every $n$, the chairs of $\sigma^{n}(j)$ form a stack in which each
chair fills the notch of the next, and each of the seven cells of the chair gives a line of identical labels, of length
$2^{n}-2$, in direction $(1,1,1)$.  Along such a line $p$ is constant, $-36$ on one cell type and $+36$ on another, and
each component of $d$ is $\pm1$.
\end{prop}

\begin{proof}
Label the cells $(x,y,z)$ of the stack by the difference $e=(x-z,\,y-z)$.  The lines of the stack are exactly the cells whose
difference lies in a set $T$ of seven values, and a cell of difference $e$ has a label $\Lambda(e)$.  We show that every cell of
$\sigma^{n}(j)$ with difference in $T$ has the label $\Lambda(e)$, by induction on $n$.  For $n=0$ the single cell has difference
$(0,0)$ and label $j=\Lambda(0,0)$.  A cell of $\sigma^{n+1}(j)$ is the child $(a,b,c)\in\{0,1\}^{3}$ of a cell of
$\sigma^{n}(j)$ whose difference is $e'=\bigl((e_{0}-a+c)/2,(e_{1}-b+c)/2\bigr)$.  So it suffices that, for every $e\in T$ and
every child position compatible with $e$, the difference $e'$ lies in $T$ and $\sigma[\Lambda(e'),(a,b,c)]=\Lambda(e)$.  That
is $@cases@$ cases for each of the six labels, and they all hold (\code{chair44\_lines.py}).  The main diagonal is the case $e=(0,0)$.
\end{proof}

\noindent
The stack lies in every tiling.  It occurs in $\sigma^{n}(j)$ for every $n$, and by Theorem~\ref{thm:hull} and minimality every
legal patch occurs in every tiling.  So in every Chair44 tiling and for every $L$ there are $L$ consecutive cells on a line
in direction $(1,1,1)$ on which the mean of $p$ is exactly $-36$ (and others on which it is $+36$) and the mean of each
component of $d$ is $\pm1$, while the box means are $0$.  Averages along these lines do not converge to the box mean, and
no uniformity in $L$ can hold for them.

\paragraph{Corner lines.}  For $b\in\{0,1\}^{3}$ the cells $tb$ of $\sigma^{n}(j)$, $0\le t<2^{n}$, are the $n$-th iterate of the
constant-length-$2$ substitution $\tau_{b}(j)=\sigma[j,0]\,\sigma[j,b]$ on the same labels, since the child positions along
the line are always $0$ and $b$.  Its closed strongly connected classes, each primitive and each reached from every label, carry
exact letter frequencies and so exact limits of the mean of each cell observable along the line, for each starting label.  Table~\ref{tab:corner} gives the
mean of the notch vector; the mean of $p$ is $0$ on all of them.  The seven lines are the lines through the corner of the
cube in the seven directions in $\{0,1\}^{3}$; they are also read off the label cube, with agreement.

\begin{table}[ht]
\centering
\caption{Corner lines of $\sigma^{n}(j)$: closed classes of $\tau_{b}$ and the exact limit of the mean of the notch vector.  The
box mean is $(0,0,0)$.}
\label{tab:corner}
\begin{tabular}{lll}
direction $b$ & closed classes of $\tau_{b}$ & mean of $d$ \\ \hline
@rows@
\end{tabular}
\end{table}

\begin{prop}
\label{prop:lineaction}
Assume that the hull $\Omega$ has maximal equicontinuous factor $\pi:\Omega\to\mathbb{Z}_{2}^{3}$ with $\pi_{*}\nu$ the Haar measure
(Corollary~\ref{cor:hull}).  Let $v\in\mathbb{Z}^{3}\setminus0$.  Then the action of the translation $T^{v}$ on $\Omega$ is
not uniquely ergodic.  There are a function $f$ of a finite window of labels and a tiling $T$ such that the averages
$\frac1L\sum_{t<L}f(T^{tv}T)$ along the line through $0$ in direction $v$ converge to a number different from $\int f\,d\nu$.
\end{prop}

\begin{proof}
The closure of $\{tv:t\in\mathbb{Z}\}$ in $\mathbb{Z}_{2}^{3}$ is $H=\mathbb{Z}_{2}v$, a closed subgroup of infinite index, so its
Haar measure is $0$.  For $x\in\Omega$ the set $K=\pi^{-1}(\pi(x)+H)$ is compact, non-empty and $T^{v}$-invariant, so it carries a
$T^{v}$-ergodic probability measure $\mu$.  Since $\nu(K)=0$, $\mu\neq\nu$, and functions of a finite window of labels are dense
in $C(\Omega)$, so some such $f$ has $\int f\,d\mu\ne\int f\,d\nu$.  By Birkhoff's theorem the averages along the line converge to
$\int f\,d\mu$ for $\mu$-almost every tiling.
\end{proof}

\noindent
Proposition~\ref{prop:stack} and Table~\ref{tab:corner} are explicit instances for the directions $(1,1,1)$ and the others in
$\{0,1\}^{3}$.  The failure depends on the observable and on the line: the corner lines all have $p$-mean $0$, and a
coordinate-axis line can have bounded sums of $d_{z}$ while another has sums of $d_{z}$ growing like $L$.  Nothing here
concerns directions that are not lattice directions: the cells met by a line of irrational slope are not an orbit of a
subgroup of $\mathbb{Z}^{3}$, and the argument does not apply.
""", labels=hu['labels'], cases=ln['tube']['cases'][0], rows=' \\\\\n'.join(rows) + ' \\\\')


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
\section{Unfolding the Chair44 and Comparing it with the Spectre}
\label{sec:unfold}

Section~\ref{sec:chair} compared the two tilings by their combinatorics.  This section brings the three-dimensional tile down
to two dimensions in the two exact ways, unfolding its surface into a net and slicing and projecting the hierarchy, and then
asks the same questions of both tilings.  Each tile is reduced to a point (the centroid of a Spectre, the centre of the bounding
cube of a chair) and an orientation vector ($e^{i\theta}$ for a Spectre, the direction $d=g(1,1,1)$ of the missing corner, the
notch, for a chair).  Every number is a Fact computed by \code{chair44\_unfold.py}, \code{chair44\_slice.py} and
\code{chair44\_findings.py}, with the Spectre sums on the patches of \code{spectre.py} through level $5$, which has $@N5@$ tiles.

\subsection{The twist of the Tsiokos Chair44 supertile}

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

\subsection{Polyhedral Unfolding and Surface Net Construction}

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

\subsection{Chair44: Slices, projections and the notch vector}

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

Where the power goes is asked with a deliberately loose statistic: the median of $I(k)/N$ over $300$ fixed random wave
vectors, with $k$ in the box $[0,2\pi/\ell]^{d}$, against the number of tiles.  It falls in both, as $N^{@s_s@}$ for the Spectre
centroids and $N^{@s_c@}$ for the Chair44 centres, and as $N^{@s_sw@}$ and $N^{@s_cw@}$ with orientation weights: the intensity
is concentrated on few wave vectors, as in a Bragg spectrum, and faster for the chair, whose centres lie on a lattice with
defects.  Figure~\ref{fig:chaircompare} and Table~\ref{tab:unfold} gather the comparison.
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

\subsection{Scope, Limitations, and Epistemic Scope}

The nets are a sample of a space of $@trees@$ trees, and nothing is claimed about the best net.  The point and the orientation
vector of each tile are choices, and the sum rule is a statement about the notch vector, not about every function of the
orientation.  The Spectre patches of \code{spectre.py} are mirror-type placements; the sum is unchanged by the mirror, but the
fixed random wave vectors are not.  The median statistic compares two dimensions at one scaling of $k$ and is noisy.  None of
this decides whether the diffraction measure of either tiling is pure point: for the Spectre that is a published
theorem~\cite{BaakeHat,BaakeSpectre}, and the exact analysis of the Chair44 centre set is in Section~\ref{sec:exact}.
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












def spectral_section():
    return r"""
\section{Exact Spectral Theory of the Spectre Monotile Substitution}
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
one $\Gamma$ entry, seeded with a single $\Delta$ (the seed of Table~\ref{tab:census}; seeding with $\Gamma$ gives 7, 55,
433, \dots).  The \emph{geometric census} is what a placed patch
contains, and a placed patch carries the Mystic as well, so it is larger.  The
two are not interchangeable.

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
\label{tab:census}
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
$P(k)=5P(k-1)-3P(k-2)-P(k-3)$, giving $P=%s$.  The sequence itself is counted twice --- see Fact~\ref{fact:bdyexact}.
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
$P(k+2)-4P(k+1)-P(k)=-16$ for all $k$, by induction.  Together with
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
by $2+\sqrt5$.  This is a $\mathbb{Q}(\sqrt{15})$ / $\mathbb{Q}(\sqrt5)$
divide between area and perimeter of the
\emph{same} tile.  The supertile
boundary is golden although the tiling is not.

Since $\nu>\lam$ the boundary is fractal: its box dimension is
$\log\nu/\log\lam=%s$, so a perimeter argument assuming $P(k)\sim\lam^{k}$ is
wrong, and wrong by a definite amount.  Both halves of that comparison are
proved --- $P(k+3)\ge4P(k+2)+30$ for every $k$, and
$\lam^{2}<16$ hence $\lam<4$ --- and the inference between them is elementary
arithmetic.

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








NOTE_OUTPUT = os.path.join(HERE, 'spectre_bigbang_note.tex')
NOTE_TITLE = 'A Flat Initial Surface? A Speculative Reading of the Spectre and Chair44 Tilings'


def note_document():
    """The companion physics note: the conjecture, its background and the Weyl remark, split out of the paper."""
    abstract_ = r'''
\begin{abstract}
A conjecture about the initial state of the universe is stated so that it can be argued with: the Big Bang was a flat
two-dimensional singularity, connected to the planar Spectre monotile, and the emergence of three-dimensional space is
connected to Tsiokos's Chair44.  Two classical background facts, a reading through the inflation arithmetic of the two
tilings, and a proposal on the Weyl tensor are given.  Nothing is derived; what would support or retire the reading is
stated.
\end{abstract}
'''
    bib = (r'''
\begin{thebibliography}{9}

\bibitem{HartshornRomanov} B.~Hartshorn and T.~Romanov, \emph{''' + TITLE + r'''}, preprint (2026),
\url{https://github.com/brentharts/spectre}.

\bibitem{SmithEtAl2} D.~Smith, J.S.~Myers, C.S.~Kaplan and C.~Goodman-Strauss. (2024).
\emph{A chiral aperiodic monotile}
Combinatorial Theory, 4(2).

\bibitem{Tsiokos1} I.~Tsiokos (2026). \emph{A strongly aperiodic monotile in three dimensions},
arXiv:2609.19214.

''' + NOTE_MOVED_BIB + r'''
\end{thebibliography}

\end{document}
''')
    return '\n'.join([PREAMBLE.replace('@TITLE@', NOTE_TITLE), abstract_, thesis_section(), bib])


def write(path=OUTPUT, note_path=NOTE_OUTPUT):
    text = document()
    with open(path, 'w') as handle:
        handle.write(text)
    with open(note_path, 'w') as handle:
        handle.write(note_document())
    return path + ' and ' + note_path


#TITLE = 'Inflation, Internal Space and the Absent Clock: Exact Arithmetic of the Spectre and the Chair44'
TITLE = "The Internal Space and Boundary Growth of the Spectre and Tsiokos's Chair44 Tilings"


def abstract():
    return fill(r"""
\begin{abstract}
Two monotiles that tile only by inflation are compared by exact computation: the planar Spectre and Tsiokos's
three-dimensional Chair44.  The Spectre inflates by the unit $\lam^{2}=@perron@$ of $\mathbb{Z}[\sqrt{15}]$, its supertile
perimeter by a unit of $\mathbb{Q}(\sqrt5)$, so its boundary is fractal, and its lift to $\mathbb{Z}^{4}$ has two
expanding and two contracting directions: an exact internal plane whose window has a fractal boundary.  The Chair44
inflates by the integer $2$ on $\mathbb{Z}^{3}$, with no internal space and no twist, and its centre set has an exactly
computed pure-point diffraction measure.  Every tiling by it lies in the substitution hull, so this holds for all of
them, and the symmetry group of a tiling is one of $1$, $C_{2}$, $C_{4}$, $D_{4}$.  Orientation sums and contact matrices
separate the two.  Four finite-patch
computations then test whether the Spectre's substitution hierarchy can serve as a clock: positional clustering shows no
arrow, the radius needed to recognise an ancestor grows about threefold per level without making one, the look-back
law is shared with a quadtree, and the parity flip of the construction shows in the outlines of supertiles and not in the
tiles.
\end{abstract}
""", perron=tex(F.get('perron').value))


def introduction():
    return fill(r"""
\section{Introduction}
\label{sec:intro}

A tiling that exists only by inflation carries an arithmetic: the number by which it inflates, the field that number
generates, and what that field leaves behind.  The Spectre of Smith, Myers, Kaplan and Goodman-Strauss~\cite{SmithEtAl,SmithEtAl2}
inflates by a unit of a real quadratic field, and the Chair44 of Tsiokos~\cite{Tsiokos1} by the integer $2$.  A unit of
norm one has a conjugate that contracts, so the vertices lie in a module of rank greater than the dimension of space and
the extra directions form an internal space in which they fill a bounded window; an integer inflation $m>1$ has no
conjugate, nothing contracts, and the structure is $m$-adic.  This paper computes that difference exactly.

A companion note~\cite{BangNote} states the physical hypothesis that motivated this work; nothing here depends on it.
Statements are stratified: \emph{Facts} are computations or theorems, each classified by the evidence behind it (the table
is in the supplement); \emph{Propositions} are computations with a modelling step; results labelled \emph{computation, not
Fact} are numerical, on finite patches, and enter no ledger.

The Spectre is analysed in Sections~\ref{sec:spectral}--\ref{sec:charges} (substitution matrix, spectrum, marginal charges),
\ref{sec:bdyop} (fractal perimeter) and~\ref{sec:lift} (four-dimensional lift), the Chair44 in Sections~\ref{sec:chair}
and~\ref{sec:unfold}, with a proof that every Chair44 tiling lies in the substitution hull and a classification of the
symmetry groups of its tilings in Section~\ref{sec:hull}, and the two are compared in Section~\ref{sec:exact}.  Section~\ref{sec:hier} tests whether the
Spectre's substitution hierarchy can serve as a clock.

What is not claimed matters as much.  The substitution matrix is a transcription of the published rule and is the one quoted
input; the einstein property of the Chair44 is Tsiokos's (2026) claimed proof \cite{Tsiokos1}, quoted; the hull theorem of
Section~\ref{sec:hull} takes registration and the unique parent tiling from~\cite{Tsiokos1,GoodmanStrauss}, quoted; that the Spectre diffraction is pure
point is a published theorem~\cite{BaakeHat,BaakeSpectre}, not re-proved here.  Section~\ref{sec:conclusion} lists the rest.
""")


def twist_subsection():
    return r"""
\subsection{The twist of the Spectre supertile}
\label{sec:twist}

\begin{fact}[The twist of the supertile]
\label{fact:twist}
Successive supertile quadrilaterals are similar, with ratio $\lam$ and a rotation of $-\theta_S$ on even levels and
$+\theta_S$ on odd ones, where
\[
\tan\theta_S=\frac{5\sqrt3-6}{13},\qquad
\cos2\theta_S=\frac{4+3\sqrt3}{10},\qquad
\theta_S=%s\ldots^{\circ}.
\]
Applied twice, the map has characteristic polynomial $(x-1)^{2}(x^{2}-8x+1)$, which carries the quadratic factor of
$\chi_M$, and its dominant eigenvalue is the real number $\lam^{2}$: over two levels the quadrilateral is dilated and not
rotated.
\end{fact}

\noindent
The $\sqrt{15}$ enters the one-level factor only through its modulus $\lam$; the phase lies in
$\mathbb{Q}(\sqrt3,i)$, the field of the hexagonal tile coordinates, and $\cos\theta_S$ has minimal polynomial
$400x^{4}-560x^{2}+169$ over $\mathbb{Q}$.  The alternation of the sign is the mirror applied at each level.  The twist is
irrational as an angle, which is the contrast drawn with the Chair44 in Section~\ref{sec:unfold}.
""" % str(F.get('spectre_twist').decimal)


def _num(x, nd=2):
    return ('%%.%df' % nd) % x


def lift_section():
    g = F.get
    li, im, ip = g('lift_integer').value, g('inflation_matrix').value, g('internal_plane').value
    be = ip['boundary_exponents']
    return fill(r"""
\section{The Spectre monotile in four dimensions}
\label{sec:lift}

Every edge of the Spectre tiling is a unit vector at a multiple of $30^{\circ}$.  Following Ch\'eritat and
Ma~\cite{CheritatMa}, who show that the outlines of the deformable tiles $\mathrm{Tile}(a,b)$ are projections of a single
path in four dimensions, the even multiples are taken to lie in a first plane and the odd multiples in a second.  Write
$\omega=e^{i\pi/3}$ and $z=e^{i\pi/6}$.  An even edge is $p+q\omega$ and an odd edge is $z(r+s\omega)$ with integers, so walking
along edges assigns every vertex a point $(p,q,r,s)$ of $\mathbb{Z}^{4}$, and the projection $(u,v)\mapsto au+bv$ is
the deformation of the tile.  The higher-dimensional point of view is the one in which a two-dimensional quasicrystal
carries four-dimensional topology~\cite{Kraus2013}.

\begin{fact}[The integer lift]
\label{fact:lift}
On the level-$@lvl@$ patch ($@nv@$ vertices) every vertex receives integer coordinates $(p,q,r,s)$; the @chk@ vertex
identifications made on the way all agree, so the lift is consistent; the vertex differences have rank $@rank@$, and the
physical position is $a(p+q\omega)+bz(r+s\omega)$, which is the tiling itself at $a=b=1$ and gives a homogeneous quadratic
form in $(a,b)$ for the area of a tile.  The vertices lie in the rank-four module $\mathbb{Z}[\zeta_{12}]=\mathbb{Z}^{4}$.
\end{fact}

\begin{fact}[The inflation as an integer matrix]
\label{fact:inflmat}
The supertile recursion moves four key points of the supertile by rotations through multiples of $60^{\circ}$, a mirror
and translations built from the points themselves; on $\mathbb{Z}[\zeta_{12}]$ each is an integer map, so one level is a
$16\times16$ integer matrix $\Phi$ on the four key points.  It reproduces the floating-point recursion of the construction
for three levels (largest difference @err@), and
\[
\chi_{\Phi}(x)=(x-1)^{4}(x+1)^{4}\,(x^{4}-8x^{2}+1)^{2},\qquad
x^{4}-8x^{2}+1=\bigl(x^{2}-(4+\sqrt{15})\bigr)\bigl(x^{2}-(4-\sqrt{15})\bigr),
\]
with roots $\pm\lam,\ \pm\lam^{-1}$ and $\lam^{2}=@perron@$.  The key-point vector has minimal polynomial
$(x^{2}-1)(x^{4}-8x^{2}+1)$, and on that block $\Phi$ is the companion matrix
\[
@block@ ,
\]
of determinant $@det@$ and trace $@tr@$.
\end{fact}

\noindent
The unit $4+\sqrt{15}$ has norm one, which is why the block is unimodular: two eigenvalues expand ($\pm\lam$) and two
contract ($\pm\lam^{-1}$).  The negative roots come from the mirror applied at every level; over two levels $\Phi^{2}$
acts on the expanding pair as the real number $\lam^{2}$.  The Chair44, by contrast, inflates by $2\,I_{3}$ on
$\mathbb{Z}^{3}$, of determinant $8$: every eigenvalue expands and nothing contracts, so there is no internal space and no
window.

\begin{fact}[The internal plane]
\label{fact:plane}
A linear functional of $(p,q,r,s)$ has bounded range on the supertile corners exactly when it kills the expanding part
($\Phi^{2}=4+\sqrt{15}$) of every iterate of the key-point vector.  Each level mirrors the frame, so the odd iterates
are mirrored first; the expanding parts then span two dimensions and the annihilator is the plane
\[
@plane@
\]
(rows are functionals on $(p,q,r,s)$), with entries in $\mathbb{Q}(\sqrt{15})$.  It agrees with a plane fitted to the
lifted vertices to principal cosines above $1-10^{-6}$.  Its extent on the lifted patches is @e3@ at level $3$ and @e4@ at
level $4$, while the physical extent grows by a factor @gr@ per level (the inflation factor is @lam@).  The Galois
conjugate $z\mapsto z^{5}$, the obvious candidate for the internal space, is not bounded: its extent is @gal@ at level $3$.
\end{fact}

\noindent
The plane is bounded for one frame and its image under the mirror for the next, which is why a single fitted plane
cannot serve every level.  The internal image of the vertices is the window of a model set.  Its boundary cell count
(Figure~\ref{fig:lift}; level-$5$ patch, holes from finite sampling filled) grows by $2^{@e1@}$ and $2^{@e2@}$ at the
successive halvings of the cell size from $0.2$, against $2^{1}$ for a smooth curve, while the area grows by
$2^{@ea@}$.  The boundary is therefore not a smooth curve, with a box-counting dimension of about $1.2$.  That is an
estimate and not a theorem.  It is not the dimension $\log\nu/\log\lam$ of the supertile outline in the physical plane
(Section~\ref{sec:bdyop}), and it is consistent with, and does not derive, the windows of Rauzy type that Baake,
G\"ahler, Mazá\v{c} and Sadun obtain~\cite{BaakeHat,BaakeSpectre}.  Whether anything like this exists for a
three-dimensional tile is the question raised by Kamenetsky~\cite{Kamenetsky}, who finds that the obvious
four-dimensional analogues of the Chair44 fail a closure test; the integer inflation of the chair leaves nothing to project.
@FIGLIFT@
""", lvl=li['level'], nv=li['vertices'], chk=li['consistency_checks'], rank=li['rank'],
        err='$%s$' % ('%.0e' % im['float_error']).replace('e-', r'\times10^{-') + '}$' if False else '$' + ('%.0e' % im['float_error']).replace('e-', r'\times10^{-').replace('e+', r'\times10^{') + '}$', perron=tex(F.get('perron').value), det=im['det'], tr=im['trace'],
        block=sp.latex(sp.Matrix(im['block'])), plane=sp.latex(sp.Matrix(ip['plane'])),
        e3=_num(ip['internal_extent'][3]), e4=_num(ip['internal_extent'][4]), gr=_num(ip['growth']),
        lam=_num(float(F.LAM), 4), gal=_num(ip['galois_extent'], 0), e1=_num(be[0], 2), e2=_num(be[1], 2),
        ea=_num(ip['area_exponent'], 2),
        FIGLIFT=figure_block('lift4d', 'fig:lift',
            r"""The lift of the Spectre.  (a) The internal image of the vertices of a level-$4$ patch through the exact
bounded plane: the window.  (b) The projections $(a,b)=(1,1)$ (black) and $(1,\tfrac12)$ (blue) of the same lifted tiles.
(c) The physical extent of the lifted patch, which grows with the level, and the internal extent, which does not."""))


_DIF_CACHE = {}


def diffraction_numbers():
    if not _DIF_CACHE:
        res = DIF.measure()
        _DIF_CACHE.update(sm=DIF.summarise(res))
    return _DIF_CACHE


def hat_turtle_paragraph():
    d = diffraction_numbers()['sm']
    H, S, T = d['Hat'], d['Spectre'], d['Turtle']
    return fill(r"""
\subsection{The Hat and the Turtle see their lattice; the Spectre does not}

\textbf{Computation, not Fact.}  The Hat and the Turtle, unlike the Spectre at $(a,b)=(1,1)$, tile on a hexagonal lattice
$\mathbb{C}/\mathbb{Z}[\zeta_3]$, and that can be read in reciprocal space.  For the vertex set of a level-$5$ patch at unit
density, with a Gaussian window of width $9$, a fraction @hf@ of the Bragg intensity above threshold of the Hat lies within
$12\%$ of a point of the dual lattice (from @hn@ peaks) and @tf@ for the Turtle (from @tn@ peaks), against @hb@ and @tb@ for
the same lattice rotated or rescaled.  The Spectre has strong peaks as well, but only a fraction @sf@ of them is on such a
lattice (control @sb@, @sn@ peaks).  This is the reciprocal-space form of the statement that $\mathrm{Tile}(1,1)$ is not on the
lattice.  It is one patch and one window, no exponent is claimed, and it does not bear on the pure-point theorem.
""", hf='$%.2f$' % H['match']['frac'], hn=H['match']['npeaks'], tf='$%.2f$' % T['match']['frac'], tn=T['match']['npeaks'],
        hb='$%.2f$' % H['match']['base'], tb='$%.2f$' % T['match']['base'],
        sf='$%.2f$' % S['match']['frac'], sb='$%.2f$' % S['match']['base'], sn=S['match']['npeaks'])


def exact_section():
    g = F.get
    cd, orr, ch, ct = (g(k).value for k in ('centre_diffraction', 'orientation_recursion',
                                           'orientation_channels', 'contact_matrices'))
    ce = ch['chair_exponents']
    se = ch['spectre_exponents']
    sp_, cc = ct['spectre'], ct['chair']
    return fill(r"""
\section{Scattering Diffraction, Orientations, and Adjacency Relations}
\label{sec:exact}

This section asks four exact questions of both tilings.  The tools are in the repository of curved-tile
programs~\href{https://github.com/brentharts/spectre-monotile-py}{spectre-monotile-py}; each Fact is computed there and in \code{monotilefacts.py}.

\subsection{The diffraction of the Chair44 centres}

Take one point per chair, the centre of its bounding cube, and call the set at depth $n$ $X_{n}\subset\mathbb{Z}^{3}$.

\begin{fact}[Exact diffraction of the centre set]
\label{fact:centrediff}
$X_{n}=2X_{n-1}\cup(B_{n}\setminus B_{n}')$, where $B_{n}$ is the set of odd points of the cube $[1,2^{n+1}-1]^{3}$ and
$B_{n}'$ the odd points with every coordinate above $2^{n}$ (the corner octant); the recursion reproduces the placed
patch for $n\le4$.  The structure factor is a finite sum of products of geometric series and equals the direct sum over
the centres (relative difference @cd@).  At the dyadic wave vectors $k=2\pi m/2^{J}$, $m$ not all even, its limit is the
exact amplitude
\[
a(k)=\tfrac78\,8^{1-J}\bigl(\pm1+\tfrac17\bigr),
\]
with sign $-$ when $m_{1}+m_{2}+m_{3}$ is odd, so $a=@aa@$ on an axis and $a=@ad@$ on the diagonal at $J=1$.  The squared
amplitudes at level $J$ sum to $\tfrac{21}{4}8^{1-J}$, and
\[
\frac1{49}\Bigl(1+\sum_{J\ge1}\tfrac{21}{4}8^{1-J}\Bigr)=\frac17=\text{density of the centres}.
\]
Wave vectors that are not dyadic carry no amplitude.
\end{fact}

\noindent
The Parseval sum is complete: all the mass of the density is in the Bragg peaks, so the diffraction measure of the
centre set is pure point, and the $2$-adic route of Section~\ref{sec:chair} is made exact for this set.  The mass
of the notch-weighted measure is only numerically consistent with this.  That the diffraction of the Spectre is pure
point is a published theorem~\cite{BaakeHat,BaakeSpectre} and is not re-proved here; the contrast is only in the
route, a cut-and-project scheme with an internal space for the Spectre and a dyadic Parseval sum for the chair.
@FIGBRAGG@

\subsection{Orientation sums and their channels}

Section~\ref{sec:unfold} measured the sum of the orientation vectors of both tilings.  The chair's is exactly $2^{n}$.  For the
Spectre it can now be written as a recursion.

\begin{fact}[The orientation recursion]
\label{fact:orientrec}
Let $S_{n}$ be the vector over the nine tile labels of the sums of $e^{i\theta}$ over the tiles of the depth-$n$ supertile
of each label.  Then $S_{n+1}=C\,\overline{S_{n}}$ with $C$ the $9\times9$ matrix of the slot rotations of the construction;
the conjugate appears because the inflation reverses orientation.  Two levels give $C\overline{C}$ with leading eigenvalue
$\lam^{2}=@perron@$, and the sums over the placed tiles agree with the recursion (largest relative difference $@mr@$, to
level $5$).  Hence $|S_{n}|\sim c\,\lam^{n}$ with
\[
c^{2}=\frac{15+5\sqrt3+3\sqrt5+3\sqrt{15}}{20}=@c2@,
\]
a closed form that is identified, not proved: it is the root of $400x^{4}-1200x^{3}+840x^{2}-180x+9$ found from the
recursion to eighty digits.
\end{fact}

\noindent
The constant is the ratio @c@ measured in Section~\ref{sec:unfold}.  Splitting the orientation sums by the
irreducible representations of the rotation group of the cube for the chair, and by the harmonic $m$ of the orientation
angle for the Spectre, gives the growth exponent of each channel.

\begin{fact}[Orientation channels]
\label{fact:channels}
The $24$-dimensional orientation matrix of the Chair44 has characteristic polynomial $x^{3}(x-8)(x-2)^{20}$: the trivial
channel has eigenvalue $8$ and every non-trivial channel is exactly $2$ times a projector.  With the exponent
$h=\ln\mu/\ln\lam_{\rm lin}$ of the leading eigenvalue $\mu$ in the linear inflation $\lam_{\rm lin}$, the trivial channel
has $h=3$ and all others $h=@hc@$.  For the Spectre the channels are $h=@h0@$ for $m=0,\pm6$, $h=@h1@$ for $m=\pm1$, and
$h\le0$ for $m=\pm2,3,4$.
\end{fact}

\noindent
The intensity at the origin per tile of a channel with exponent $h$ in dimension $d$ scales as
$\lam_{\rm lin}^{(2h-d)n}$.  The chair's orientation weights therefore have $|S|^{2}/N\propto 2^{-n}$ in the non-trivial channels
($h=1$, $d=3$) and the Spectre's has an exponent $0$ for $m=\pm1$ ($h=1$, $d=2$), which is the difference drawn in
Figure~\ref{fig:chaircompare}.

\subsection{Which faces meet which}

\begin{fact}[Contact matrices]
\label{fact:contacts}
Count how often native edge $a$ of one tile meets native edge $b$ of a neighbour.  For the Spectre (level-$4$ patch,
$@sc@$ contacts) the $14\times14$ symmetric matrix has $@ss@$ nonzero entries, rank $@sr@$, is bipartite, connected, and has
a spectrum symmetric about zero.  For the Chair44 (depth-$3$ patch, $@cc@$ contacts, equal to $12(8^{3}-4^{3})$) the
$24\times24$ matrix of native unit faces has $@cs@$ nonzero entries, rank $@cr@$, is not bipartite, and has
@cx@ connected components.
\end{fact}

\noindent
The Spectre's edges fall into two classes that never meet within a class, and the chair's faces do not.  The boundary
operators differ in the same way as the inflations: the supertile perimeter of the Spectre obeys the recurrence with
characteristic polynomial $(x-1)(x^{2}-4x-1)$ of Section~\ref{sec:bdyop}, and the surface of the chair obeys
$S_{N}=24\cdot4^{N}$, with characteristic polynomial $x-4$.

@HATTURTLE@
""", cd='$%.1e$' % cd['closed_vs_direct'], aa=sp.latex(sp.Rational(cd['amp_axis'].numerator, cd['amp_axis'].denominator)),
        ad=sp.latex(sp.Rational(cd['amp_diag'].numerator, cd['amp_diag'].denominator)),
        perron=tex(F.get('perron').value), mr='%.0e' % orr['max_rel'], c2='%.4f' % orr['c2'],
        c='$%.4f$' % math.sqrt(orr['c2']),
        hc='%d' % round(ce['chair A2']), h0='%d' % round(se['spectre m=0']), h1='%d' % round(se['spectre m=1']),
        sc='{:,}'.format(sp_['contacts']).replace(',', '{,}'), ss=sp_['support'], sr=sp_['rank'],
        cc='{:,}'.format(cc['contacts']).replace(',', '{,}'), cs=cc['support'], cr=cc['rank'], cx=cc['components'],
        FIGBRAGG=figure_block('chair44_bragg', 'fig:bragg',
            r"""Bragg structure of the Chair44 centre set.  The exact amplitudes at the dyadic wave vectors by level, the
level masses $\tfrac{21}{4}8^{1-J}$ and their running sum approaching the density, and the finite-patch
intensity converging to it."""),
        HATTURTLE=hat_turtle_paragraph())


def hier_numbers():
    """The numbers of Section 'What the hierarchy supplies', computed once at build time (light settings)."""
    if _HIER_CACHE:
        return _HIER_CACHE
    import contextlib
    import io
    import hierarchy_cache as HC
    import barbour_complexity as BC
    import records_recognizability as RR
    import lookback_lca as LB
    import parity_clock as PC
    import outline_chirality as OC
    import numpy as np
    with contextlib.redirect_stdout(io.StringIO()):
        rows = BC.table(range(1, 5))
        rec = RR.run(5, 8000, radii=(2, 3, 4, 8), levels=(1, 2))
        _, _, sl, rat = LB.report('S', HC.load(6), 600, HC.LAMBDA, LB.PRED)
        _, _, slq, ratq = LB.report('Q', HC.quadtree(6), 600, 2.0, 0.5, margin=12.0, dmax=10.0)
        pair = {d: PC.pair_means(HC.load(d), 800, margin=20.0)[0] for d in (5, 6)}
        sw = PC.swirl(HC.load(6))
        ctrl, orows = OC.report(6)
    near = 'near [1,3) ell'
    one = [(p, m) for (_, _, dk, p, m) in orows if dk == 1]
    two = [(p, m) for (_, _, dk, p, m) in orows if dk == 2]
    k3 = [(p, m) for (_, k, dk, p, m) in orows if dk == 2 and k == 3]
    _HIER_CACHE.update(
        rows=rows, rec=rec, slope=sl, ratios=rat, slope_q=slq, ratios_q=ratq, pair=pair, near=near, swirl=sw,
        n1=len(one), f1=sum(m < p for p, m in one), n2=len(two), f2=sum(p < m for p, m in two),
        p35=float(np.mean([p for p, _ in k3])), m35=float(np.mean([m for _, m in k3])),
        dim=math.log(HC.NU) / math.log(HC.LAMBDA), pred=HC.NU / HC.LAMBDA ** 2)
    return _HIER_CACHE


_HIER_CACHE = {}


def hierarchy_section():
    H = hier_numbers()
    c = [r[2] for r in H['rows']]
    ex = [r[2] - r[3] for r in H['rows']]
    near = H['near']
    P6 = H['pair'][6][near]
    sw = H['swirl']
    rec = H['rec']
    r, q = H['ratios'], H['ratios_q']
    fmt3 = lambda x: '$%.3f$' % x
    fmt2 = lambda x: '$%.2f$' % x
    return fill(r"""
\section{What the substitution hierarchy supplies}
\label{sec:hier}

\textbf{Computation, not Fact.}  Whether the level index of a substitution can be read as an arrow, a record, a look-back
or a clock is a question that any physical reading of these tilings would need answered~\cite{BangNote}.  Four readings are tested on finite patches against controls.  The programs are in the \code{spectre}
repository, built on \code{hierarchy\_cache.py}, which walks the substitution tree and keeps each tile's address; the
numbers are those they print at build time, and distances are in tile scales $\ell$.

\paragraph{Clustering.}  Barbour, Koslowski and Mercati~\cite{BKM} use a scale-invariant clustering measure of equal-mass
points, $C_S=\ell_{\rm rms}/\ell_{\rm mhl}$, to identify an arrow of time in the gravitational $N$-body problem.  For the tile
centroids of the level-$n$ patch $C_S$ is @c1@, @c2@, @c3@, @c4@ for $n=1,\dots,4$, but the rise is the footprint: against a
triangular lattice cut to the same outline the excess is @e1@ at $n=1$ and at most @emax@ for $n=2,3,4$.  Positional clustering
carries no arrow.

\paragraph{Records.}  How large a neighbourhood must a tile see to name its level-$k$ ancestor?  By leave-one-out
prediction from the radius-$r$ neighbourhood, the level-$1$ ancestor is determined at $3\ell$ (error @a13@, against @a12@ at
$2\ell$), while the level-$2$ ancestor is still wrong in @a24@ of cases at $4\ell$ and @a28@ at $8\ell$; at full size the
radius grows by about a factor of three per level.  Finer levels are free and coarser ones cost a patch three times larger,
an asymmetry that is an arrow only if an observer's patch grows in one direction, which the tiling does not supply.

\paragraph{Look-back.}  Let $h$ be the level of the lowest common ancestor of two tiles at distance $d$
(Figure~\ref{fig:hlook}).  If depth were look-back time, $\langle h\rangle$ would grow as $\log_\lam d$; it does, with slope
@sl@ (@slq@ for a quadtree with $\lam=2$ through the same code), below $1$ because $h$ is capped.  For tiles $1$ to $2\ell$
apart, $P(h\ge k+1)/P(h\ge k)$ is @r2@, @r3@ for $k=2,3$ in the Spectre and @q3@, @q4@ for $k=3,4$ in the quadtree.  The
outline dimension $\ln\nu/\ln\lam=@dim@$ would give $\nu/\lam^{2}=@pred@$ and the quadtree $1/2$; the data do not separate them.
The dictionary holds on average and is shared by any self-similar hierarchy, and spatial neighbours can be several levels
apart in it.

\paragraph{Parity.}  The common determinant of the placements in a depth-$n$ patch alternates in sign with $n$
(Proposition~\ref{prop:parity}), so the construction mirrors at every level.  Were that a clock, reflection-odd statistics
would alternate with level (Figure~\ref{fig:hpar}).  For tile pairs they do not: the mean of
$\sin(\theta_o+\theta_t-2\phi)$, verified to change sign under a mirror image, for pairs $1$ to $3\ell$ apart is @p1@ when
the common ancestor is the level-$1$ supertile and @p2@, @p3@, @p4@ for levels $2,3,4$, and the swirl
$\langle\sin(\theta-\phi)\rangle$ about a level-$k$ supertile centre is negative at every level (@s1@, @s2@, @s3@ for
$k=1,2,3$).  In outlines the flip is visible: a level-$k$ supertile outline is nearer to the mirror image of the same-type
level-$(k+1)$ one than to a rotation in @f1@ of @n1@ comparisons, and nearer to a rotation of the level-$(k+2)$ one in
@f2@ of @n2@, clearly only for $k=3$ (@p35@ by rotation against @m35@ with reflection).  The alternation lives in the
shapes of supertiles and not in the tiles; the mirror image of a Spectre tiling is a Spectre tiling, so it belongs to the
substitution as written and is not a dynamics.

\paragraph{What follows.}  Clustering carries \emph{no arrow of time}; the record asymmetry and the look-back dictionary are real but
generic; the parity alternation comes from the rule and not from the tile data.  None supplies a rule for what propagates, so
the tiling \emph{has no light cone}, and none supplies a physical clock.  The missing step is a dynamics.  (Figures for the
clustering and record computations are in \code{paper/figures/spectre\_hier\_clustering} and \code{spectre\_hier\_records}.)
""" + HIER_FIGS, c1=fmt3(c[0]), c2=fmt3(c[1]), c3=fmt3(c[2]), c4=fmt3(c[3]), e1=fmt3(ex[0]), emax=fmt3(max(abs(x) for x in ex[1:])),
        a13=fmt3(rec[3][1][0][0]), a12=fmt3(rec[2][1][0][0]), a24=fmt3(rec[4][1][1][0]), a28=fmt3(rec[8][1][1][0]),
        sl=fmt2(H['slope']), slq=fmt2(H['slope_q']), r2=fmt3(r[1]), r3=fmt3(r[2]), q3=fmt3(q[2]), q4=fmt3(q[3]),
        dim='%.4f' % H['dim'], pred='%.3f' % H['pred'],
        p1=fmt2(P6[1][1][0]), p2=fmt2(P6[2][1][0]), p3=fmt2(P6[3][1][0]), p4=fmt2(P6[4][1][0]),
        s1=fmt2(sw[1][1]), s2=fmt2(sw[2][1]), s3=fmt2(sw[3][1]),
        f1=H['f1'], n1=H['n1'], f2=H['f2'], n2=H['n2'], p35=fmt2(H['p35']), m35=fmt2(H['m35']))


HIER_FIGS = figure_block(
    'spectre_hier_lookback', 'fig:hlook',
    r"""Left: mean common-ancestor level against $\log_\lam d/\ell$ (with standard deviation), Spectre and quadtree control.
Right: the tail $P(h\ge k)$ for pairs $1$ to $2\ell$ apart, with $0.538^{k-1}$ and $0.5^{k-1}$.""", width=r'0.9\textwidth') + figure_block(
    'spectre_hier_parity', 'fig:hpar',
    r"""(a) The reflection-odd pair statistic by common-ancestor level, depths $5$ to $7$.  (b) The swirl of tile orientations
about supertile centres.  (c) Outline-signature distance between same-type supertiles one and two levels apart, by
rotation only and with reflection.""")


def thesis_section():
    return r"""
\section{The conjecture}
\label{sec:thesis}

This note states the hypothesis that motivated the companion paper~\cite{HartshornRomanov}, which compares the Spectre
monotile~\cite{SmithEtAl2} and Tsiokos's Chair44~\cite{Tsiokos1} by exact computation.  Nothing computed there depends
on it, and nothing computed there tests it.  It is stated here so that it can be argued with.

\begin{conj}[The initial surface]
\label{conj:bang}
The Big Bang was not a point-like singularity but a flat two-dimensional one.  That flat surface is connected to the
Spectre monotile, and the emergence of three-dimensional space from it is connected to the Chair44.
\end{conj}

\noindent
Two background facts are kept apart from the conjecture.  The first is classical and exact.  An infinite uniform flat sheet
of surface density $\sigma$ produces a gravitational field $g=2\pi G\sigma$ that does not depend on the distance from the
sheet, because the mass in view grows as $r^{2}$ exactly as the force per unit mass falls as $r^{-2}$ (equivalently, by
Gauss's law on a pillbox).  Fields of parallel sheets are uniform and superpose, so sheets of equal and opposite density
cancel outside, and two equal sheets cancel between them.  The second concerns the setting.  Ghost-free bimetric gravity
of Hassan and Rosen~\cite{HassanRosen} gives dynamics to a second metric, and a pre-geometric origin for it has been
proposed~\cite{PreGeometric}; a picture in which the initial state consists of two sheets, one for each metric, and the Big
Bang is their separation is the kind of reading that such a setting invites.  No such model is built here, and no claim
is made about the dual-sheet cosmologies themselves.  The earlier observation that the area law
$S_{\rm rel}=\delta A/4G$ is consistent with the $s$-wave dynamics of the Nariai horizon~\cite{hartshorn2026}, the converse
of the result of Dorau and Much~\cite{DorauMuch}, is the setting in which the question arose.

What the computations of~\cite{HartshornRomanov} offer is a precise form of the contrast the conjecture needs.  A planar tile with a unimodular
inflation, here the Spectre, has an internal space of dimension two and a window; a tile with an integer inflation, here the
Chair44, has none and is $2$-adic.  The conjecture would then say that the surface is of the first kind and the bulk of the
second, and that the passage from one to the other is the emergence of the third dimension.  That is a reading and not a
result: no map from tile data to a physical quantity is given, the dichotomy is not derived from any dynamics, and the
passage between the two kinds of inflation is not constructed.  The hierarchy section of~\cite{HartshornRomanov} tried the substitution hierarchy as that quantity, and it does not
supply the map.  The reading would
be supported by a physical quantity computed from the tile data that distinguishes the two cases, and retired by showing
that the dichotomy is an artefact of the two examples, for instance by a planar tile with an integer inflation and a
pure-point diffraction measure, or by a three-dimensional tile with a unimodular one.

\begin{remark}[A proposal on the Weyl tensor]
\label{rem:weyl}
In three spacetime dimensions the Weyl tensor vanishes identically, the Riemann tensor is fixed by the Ricci tensor, and
vacuum general relativity has no local gravitational degrees of freedom: gravitational waves need a third spatial
dimension.  That is classical.  The \emph{proposal}, the first author's and not a derived result, is that this is the form in which
a flat two-dimensional initial surface would have vanishing Weyl curvature (compare Penrose's Weyl curvature hypothesis, that
the Weyl tensor vanishes at the initial singularity): a $2{+}1$ substrate has no Weyl tensor to set to zero, and the Weyl
degrees of freedom would first appear with the third spatial dimension, the step the conjecture assigns to the Chair44.
Nothing here derives it.  No map from tile data to a metric is given; it is not shown that a $2{+}1$ substrate can acquire
$3{+}1$ Weyl degrees of freedom as space emerges; and the hierarchy section of~\cite{HartshornRomanov} leaves open what
the hierarchy depth is: if the
third spatial dimension, a separate time is needed, and if look-back time, gravitational waves need a propagation rule that
the tiling does not supply.
\end{remark}
"""


def conclusion_section():
    return r"""
\section{Conclusion}
\label{sec:conclusion}

The two tilings separate cleanly in the quantities computed: the Spectre by a unit of $\mathbb{Z}[\sqrt{15}]$, a perimeter
unit of $\mathbb{Q}(\sqrt5)$ and an internal plane with a window of fractal boundary; the Chair44 by inflation $2$ on
$\mathbb{Z}^{3}$, no contracting direction, no twist, an exactly computable pure-point centre set, orientation sums $2^{n}$
and face contacts that are not bipartite.  The hierarchy tests of Section~\ref{sec:hier} are negative or generic.

What the Facts do not reach is as follows.  The substitution matrix is a transcription of the published rule and is taken
on trust.  The einstein property and aperiodicity of the Chair44 are Tsiokos's claimed proof, quoted and not checked \cite{Tsiokos1}.  The hull theorem and the symmetry classification of Section~\ref{sec:hull} are proved from the registration and parent-tiling facts of~\cite{Tsiokos1,GoodmanStrauss}, which are quoted, and from a count of admissible blocks that this paper does not repeat; the counts of legal blocks and fixed seeds are recomputed.
That the Spectre diffraction is pure point is a published theorem and is quoted.  The closed form of the orientation
constant is identified and not proved, the window dimension is a box-counting estimate, the weighted diffraction of the
chair is only numerically consistent with pure point, and the Hat and Turtle diffraction is one patch with one window.
The boundary operator governs the perimeter in the aggregate; resolved by species and edge index a handful of components obey
neither it nor its obvious extensions, while their ratios still approach $\nu$, so the extra spectrum is subdominant and cannot move
the Perron root.  The aggregate operator is claimed; the full fine operator is not claimed.  The boundary sequences are
read from $k=1$, where assembly begins, which is a convention made explicit rather than a gap.  The hierarchy results are
numerics on finite patches, with controls, and are not Facts.

Four computations would extend the work: the window of the Spectre as the attractor of the contracting maps, which would give
its dimension exactly; a four-dimensional analogue of the chair, the open problem of Kamenetsky~\cite{Kamenetsky}; an exact
count of the common-ancestor tail from the substitution, which would settle $\nu/\lam^{2}$ against $1/2$; and a rule for what
propagates on the tiling, without which no light cone can be formulated.
"""


def supplement_pointer():
    return r"""
\appendix
\footnotesize

\section*{Declarations}

\subsection*{Funding and Competing Interests}
The authors received no specific funding for this work, and declare no competing interests.

\subsection*{Code and Data Availability}
\label{sec:supp}

All source code and data supporting this study are available at:
\newline
\url{https://github.com/brentharts/spectre/}
\newline
\url{https://github.com/brentharts/spectre-monotile-py/}
\newline
\url{https://github.com/brentharts/six-birds-tiles/}

"""


NOTE_MOVED_BIB = r"""
\bibitem{HassanRosen} S.~F.~Hassan and R.~A.~Rosen, \emph{Bimetric gravity from ghost-free massive gravity},
J.\ High Energy Phys.\ \textbf{2012}, 126 (2012). \url{https://link.springer.com/article/10.1007/JHEP02(2012)126}

\bibitem{PreGeometric} A.~Addazi et al., \emph{Pre-geometric origin of ghost-free massive bigravity}, Phys.\ Lett.\ B
\textbf{882}, 140989 (2026). \url{https://doi.org/10.1016/j.physletb.2026.140989}

\bibitem{hartshorn2026} B.~Hartshorn
\emph{Gravity from Relative Entropy: Jackiw--Teitelboim Dynamics on the Nariai Horizon}
[preprint]
\url{https://doi.org/10.21203/rs.3.rs-10426867/v1}

\bibitem{DorauMuch} P.~Dorau and A.~Much, \emph{From Quantum Relative Entropy to the Semiclassical Einstein Equations}, Phys.\ Rev.\ Lett.\ \textbf{136}, 091602 (2026); arXiv:2510.24491.
\url{https://doi.org/10.1103/lmq8-nsty}


"""


BIBLIOGRAPHY = r"""
\begin{thebibliography}{99}

\bibitem{SmithEtAl} D.~Smith, J.S.~Myers, C.S.~Kaplan and C.~Goodman-Strauss. (2024).
\emph{An aperiodic monotile}
Combinatorial Theory, 4(1).

\bibitem{SmithEtAl2} D.~Smith, J.S.~Myers, C.S.~Kaplan and C.~Goodman-Strauss. (2024).
\emph{A chiral aperiodic monotile}
Combinatorial Theory, 4(2).


\bibitem{BangNote} B.~Hartshorn and T.~Romanov, \emph{A flat initial surface? A speculative reading of the Spectre and
Chair44 tilings}, companion note (2026), \code{spectre\_bigbang\_note.tex} in~\url{https://github.com/brentharts/spectre}.

\bibitem{Tsiokos1} I.~Tsiokos (2026). \emph{A strongly aperiodic monotile in three dimensions},
arXiv:2609.19214.  Code and data:
\url{https://github.com/ioannist/six-birds-tiles} (CC BY 4.0).

\bibitem{GoodmanStrauss} C.~Goodman-Strauss (2026). \emph{Notes on a strongly aperiodic monotile in $E^{3}$},
arXiv:2609.24779.

\bibitem{Flicker} F.~Flicker (2026). \emph{Matching rules for a three-dimensional strongly aperiodic monotile},
arXiv:2609.23783.

\bibitem{Romanov} T.~Romanov (2026). \emph{Every Chair44 tiling is a substitution tiling, and its symmetry
groups}.  Draft of 24 October 2026; the computations are \code{chair44\_vertex\_blocks.py},
\code{chair44\_hull\_prune.py} and \code{chair44\_symmetry\_census.py}, run on commit \code{d53d224} of~\cite{Tsiokos1}, and merged into \href{https://github.com/brentharts/six-birds-tiles/commit/ae6ecff1a32fc195f031f131a23c52243dfa9fb1}{commit ae6ecff}

\bibitem{BaakeHat} M.~Baake, F.~G\"ahler, J.~Mazá\v{c} and A.~Mitchell, \emph{Diffraction of the Hat and Spectre tilings
and some of their relatives}, arXiv:2502.03268.

\bibitem{BaakeSpectre} M.~Baake, F.~G\"ahler, J.~Mazá\v{c} and L.~Sadun, \emph{On the long-range order of the Spectre
tilings}, arXiv:2411.15503.

\bibitem{CheritatMa} A.~Ch\'eritat and N.~Ma (initial idea N.~Ma; contributions P.~Mostert),
\emph{4D lift of the tilings by the Smith et al.\ aperiodic monotile} (2023).
\url{https://www.math.univ-toulouse.fr/~cheritat/2023-monotile/4D-lift/}


\bibitem{Kraus2013} Y.E.~Kraus, Z.~Ringel, and O.~Zilberberg (2013)
\emph{Four-Dimensional Quantum Hall Effect in a Two-Dimensional Quasicrystal}
\url{https://doi.org/10.1103/PhysRevLett.111.226401}

\bibitem{Kamenetsky} D.~Kamenetsky, \emph{Towards Strongly Aperiodic Monotiles in Higher Dimensions}, arXiv:2610.00916.

\bibitem{LeeMoody} J.-Y.~Lee and R.~V.~Moody (2001). Lattice substitution systems and model
sets. \emph{Discrete Comput. Geom.} 25, 173--201.

\bibitem{Schlottmann} M.~Schlottmann (2000). Generalized model sets and dynamical systems.
In \emph{Directions in Mathematical Quasicrystals}, CRM Monograph Series 13, AMS, 143--159.

\bibitem{LMS} J.-Y.~Lee, R.~V.~Moody and B.~Solomyak (2003). Consequences of pure point
diffraction spectra for multiset substitution systems. \emph{Discrete Comput. Geom.} 29, 525--560.

\bibitem{BKM} J.~Barbour, T.~Koslowski and F.~Mercati, \emph{Identification of a gravitational arrow of time},
Phys.\ Rev.\ Lett.\ \textbf{113}, 181101 (2014); arXiv:1409.0917. \url{https://doi.org/10.1103/PhysRevLett.113.181101}

\end{thebibliography}

\end{document}
"""


def document():
    text = '\n'.join([
        PREAMBLE.replace('@TITLE@', TITLE), abstract(), introduction(),
        spectral_section(), census_section(), charges_section(), twist_subsection(),
        boundary_operator_section(), lift_section(),
        chair_section(), unfolding_section(), exact_section(), hierarchy_section(),
        conclusion_section(),
        supplement_pointer(), BIBLIOGRAPHY,
    ])
    text = inject_after_label(text, 'sec:spectral', FIG_CENSUS)
    text = inject_after_label(text, 'sec:bdyop', FIG_BOUNDARY)
    text = inject_before(text, '\\subsection{Why Three Dimensions Do Not Automatically Resolve Chirality}',
                         hull_subsection() + lines_subsection())
    return inject_before(text, '\\subsection{What the golden appearance is not}', boundary_additions())


def selftest():
    failures = []

    def check(label, ok):
        print('  %-58s %s' % (label, 'ok' if ok else 'FAIL'))
        if not ok:
            failures.append(label)

    import re
    text = document()
    flat = ' '.join(text.split())
    print('structure')
    check('it opens a document', r'\begin{document}' in text)
    check('and closes it', r'\end{document}' in text)
    grouping = text.replace(r'\{', '').replace(r'\}', '')
    check('braces balance', grouping.count('{') == grouping.count('}'))
    for env in ('abstract', 'longtable', 'thebibliography', 'fact', 'figure',
                'thm', 'lem', 'cor', 'defn', 'proof', 'description', 'remark'):
        check('%s is closed' % env, text.count(r'\begin{%s}' % env) == text.count(r'\end{%s}' % env))
    check('no section is empty', r'\section{}' not in text)
    labels = set(re.findall(r'\\label\{([^}]*)\}', text))
    refs = set()
    for m in re.finditer(r'\\(?:eq)?ref\{([^}]*)\}', text):
        refs.add(m.group(1))
    check('every reference has a label', refs <= labels)
    body = text.split(r'\begin{thebibliography}')[0]
    cited = set()
    for m in re.finditer(r'\\cite\{([^}]*)\}', body):
        cited |= {x.strip() for x in m.group(1).split(',')}
    listed = set(re.findall(r'\\bibitem\{([^}]*)\}', text))
    check('every citation is listed and every entry cited', cited == listed)
    check('the title is the new one', (r'\title{%s}' % TITLE) in text and 'One Tile, Two Units' not in text)
    check('the paper does not mention the machine-checked layer',
          'Lean' not in body and 'kernel-check' not in body and 'lstlisting' not in text)

    print('the numbers are the live ones')
    check('the Perron eigenvalue appears', tex(F.get('perron').value) in text)
    check('both censuses appear', all(str(t) in text for t in F.MATRIX_TOTALS[:5] + F.GEOMETRIC_TOTALS[:5]))
    check('the charge values appear', all(('%+d' % v) in text for v in F.Q_MINUS_BY_DEPTH))
    import spectre_supplement as SUPP
    supp = SUPP.document()
    check('the table is out of the main paper', r'\begin{longtable}' not in text)
    check('every fact is in the supplement', all(esc(f.key) in supp for f in F.FACTS))
    check('the supplement names this paper by its title', TITLE in supp)
    _lean_names = set(L.theorem_names(L.document()))
    check('every Lean attribution names a theorem that exists', all(f.lean in _lean_names for f in F.FACTS if f.lean))
    check('the Spectre angle is computed, with its closed form',
          r'\tan\theta_S=\frac{5\sqrt3-6}{13}' in text and r'\cos2\theta_S=\frac{4+3\sqrt3}{10}' in text
          and str(F.get('spectre_twist').decimal)[:6] in text)
    check('the availability statement names both repositories',
          'brentharts/spectre/' in text and 'brentharts/spectre-monotile-py' in text)
    check('the 2-adic route is stated as cited and not re-proved',
          'modular coincidence' in flat and r'\cite{LeeMoody,Schlottmann,LMS}' in text
          and 'is a cited theorem and is not re-proved' in text)

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
    check('the balanced exponent is a proposition in the boundary-operator section',
          text.index(r'\label{prop:sstar}') > text.index(r'\label{sec:bdyop}')
          and text.index(r'\label{prop:sstar}') < text.index(r'\label{sec:chair}'))
    check('s* is called exact but not algebraic',
          'not algebraic' in text)
    check('the drifting fit is explained as the estimator',
          'the estimator and not the geometry' in text)
    check('the seed convention is stated in the boundary section',
          'read from $k=1$' in text)
    check('the fine spectrum is left open explicitly',
          'the full fine operator is not claimed' in text)
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


    print('the lift, the exact comparisons and the thesis')
    mf = F.get
    li, im, ip = mf('lift_integer').value, mf('inflation_matrix').value, mf('internal_plane').value
    cd, orr, ch, ct = (mf(k).value for k in ('centre_diffraction', 'orientation_recursion',
                                            'orientation_channels', 'contact_matrices'))
    check('the monotile modules pass their own tests', MFT.selftest() > 0 and all(
        mod.run_tests(verbose=False) for mod in (D, O, OS, CO, L4, LM)))
    check('the lift section precedes the Chair44 and the hierarchy section comes last but one',
          text.index(r'\label{sec:lift}') < text.index(r'\label{sec:chair}') < text.index(r'\label{sec:exact}')
          < text.index(r'\label{sec:hier}') < text.index(r'\label{sec:conclusion}'))
    check('every new section is referenced from the introduction',
          all(('\\ref{%s}' % k) in introduction() for k in ('sec:lift', 'sec:exact', 'sec:chair', 'sec:unfold', 'sec:hier')))
    check('the integer inflation matrix and its polynomial appear',
          r'(x^{4}-8x^{2}+1)^{2}' in text and sp.latex(sp.Matrix(im['block'])) in text and im['expected']
          and im['det'] == 1 and im['trace'] == 0)
    check('the unit has norm one and the chair determinant is 8',
          im['unit_norm'] == 1 and r'of determinant $8$' in text)
    check('the internal plane is printed and the Galois conjugate is called unbounded',
          sp.latex(sp.Matrix(ip['plane'])) in text and 'is not bounded' in text)
    check('the window dimension is called an estimate and not the outline dimension',
          'an\nestimate and not a theorem' in text.replace('  ', ' ') or 'estimate and not a theorem' in flat)
    check('the lift quotes the literature as published, not re-proved',
          r'\cite{BaakeHat,BaakeSpectre}' in text and r'\cite{CheritatMa}' in text and r'\cite{Kamenetsky}' in text
          and 'does not derive' in flat)
    check('the centre diffraction is exact with the full Parseval sum',
          r'=\frac17=\text{density of the centres}' in text and 'pure point' in flat
          and cd['total'] == cd['density'] and 'only numerically consistent' in flat)
    check('the orientation constant is called identified, not proved',
          'identified, not proved' in text and ('%.4f' % orr['c2']) in text)
    check('the channel table is the live one',
          ch['chair_multiplicities'] == {8.0: 1, 2.0: 20, 0.0: 3} and r'x^{3}(x-8)(x-2)^{20}' in text)
    check('the contact counts are the live ones',
          ct['spectre']['rank'] == 14 and ct['chair']['rank'] == 24 and not ct['chair']['bipartite']
          and ('%d$ nonzero' % ct['chair']['support']) in text.replace('$@', '$') or str(ct['chair']['support']) in text)
    note = note_document()
    nflat = ' '.join(note.split())
    check('the paper is purely mathematical: the conjecture and the Weyl remark are in the companion note',
          not any(w in text for w in ('Big Bang', 'Weyl', r'\label{conj:bang}', r'\label{sec:thesis}', 'bimetric'))
          and r'\cite{BangNote}' in text and r'\bibitem{BangNote}' in text)
    check('the note states the thesis as a labelled conjecture that nothing computed tests',
          r'\label{conj:bang}' in note and 'Nothing computed there depends' in nflat and 'does not' in nflat
          and r'g=2\pi G\sigma' in note and r'\cite{HartshornRomanov}' in note)
    check('the bimetric references are cited and listed in the note',
          r'\cite{HassanRosen}' in note and r'\cite{PreGeometric}' in note and '10.1016/j.physletb.2026.140989' in note
          and all(('\\bibitem{%s}' % k) in note for k in ('HassanRosen', 'PreGeometric', 'hartshorn2026', 'DorauMuch')))
    check('the abstract makes no cosmological claim',
          not any(w in abstract() for w in ('Big Bang', 'universe', 'cosmolog', 'Weyl')))
    check('the figures are included and exist',
          all((r'\includegraphics[width=\textwidth]{%s}' % nm) in text
              and os.path.isfile(os.path.join(HERE, 'paper', 'figures', nm + '.pdf'))
              for nm in ('spectre_census', 'spectre_boundary', 'chair44_nets', 'chair44_slice', 'chair44_compare',
                         'lift4d', 'chair44_bragg')))
    check('the README quotes the live Fact count',
          ('%d Facts' % len(F.FACTS)) in open(os.path.join(HERE, 'Readme.md'), encoding='utf-8').read())

    print('the hierarchy section, the Weyl remark and the new framing')
    H = hier_numbers()
    hs = text.split(r'\label{sec:hier}')[1].split(r'\section{Conclusion}')[0]
    assert len(hs) < len(text.split(r'\label{sec:hier}')[1]), 'the section marker no longer splits the text'
    near, P6, sw = H['near'], H['pair'][6][H['near']], H['swirl']
    rec = H['rec']
    check('the hierarchy section is labelled computation, not Fact and says it supplies no clock',
          'Computation, not Fact' in hs[:300] and 'none supplies a physical clock' in ' '.join(hs.split()))
    check('the hierarchy section sits between the exact comparisons and the conclusion',
          text.index(r'\label{sec:exact}') < text.index(r'\label{sec:hier}') < text.index(r'\label{sec:conclusion}'))
    check('the quoted hierarchy numbers are the live ones',
          all(('%.3f' % r_[2]) in hs for r_ in H['rows']) and ('%.2f' % H['slope']) in hs and ('%.4f' % H['dim']) in hs
          and ('%.3f' % H['pred']) in hs and ('%d of %d' % (H['f1'], H['n1'])) in hs)
    check('clustering: no excess over the footprint control beyond level 1 (below 0.005)',
          max(abs(r_[2] - r_[3]) for r_ in H['rows'][1:]) < 0.005)
    check('records: level 1 determined at 3 ell, level 2 not at 4 ell',
          rec[3][1][0][0] < 0.01 and rec[4][1][1][0] > 0.3)
    check('look-back: slope near 1 and tail ratios 0.45 to 0.60, for the Spectre and the quadtree',
          0.85 < H['slope'] < 1.05 and 0.8 < H['slope_q'] < 1.05
          and all(0.45 < x < 0.60 for x in (H['ratios'][1], H['ratios'][2], H['ratios_q'][2], H['ratios_q'][3])))
    check('parity: one sign change of the pair statistic, negative swirl, no alternation',
          P6[1][1][0] > 0 and all(P6[k][1][0] < 0 for k in (2, 3, 4)) and all(sw[k][1] < 0 for k in (1, 2, 3, 4)))
    check('parity: the outline alternation holds in most comparisons and is clear at 3 -> 5',
          H['f1'] / H['n1'] > 0.7 and H['f2'] / H['n2'] > 0.7 and H['p35'] < 0.8 * H['m35'])
    check('the hierarchy figures are included and exist',
          all(any(os.path.isfile(os.path.join(HERE, 'paper', 'figures', nm + ext)) for ext in ('.pdf', '.png'))
              for nm in ('spectre_hier_clustering', 'spectre_hier_records', 'spectre_hier_lookback', 'spectre_hier_parity'))
          and r'\includegraphics[width=0.9\textwidth]{spectre_hier_lookback}' in text
          and r'\includegraphics[width=\textwidth]{spectre_hier_parity}' in text)
    check('Barbour et al. is cited and listed', r'\cite{BKM}' in text and r'\bibitem{BKM}' in text and 'arXiv:1409.0917' in text)
    wk = note.split(r'\label{rem:weyl}')[1].split(r'\end{remark}')[0]
    check('the Weyl remark is in the note and is called a proposal, not a result',
          note.index(r'\label{sec:thesis}') < note.index(r'\label{rem:weyl}')
          and 'proposal' in wk and 'not a derived result' in wk and 'Nothing here derives it' in wk)
    check('the old title is gone', 'Inflation With and Without' not in text)
    check('the abstract mentions the hierarchy test', 'hierarchy' in abstract() and 'quadtree' in abstract())

    print('the hull and the symmetry groups of Chair44 tilings')
    import chair44_hull as HU
    hu = F.get('chair_legal_blocks').value
    ht = F.get('chair_hull_theorem').value
    hull = ' '.join(text.split(r'\label{sec:hull}')[1].split(r'\subsection{Why Three Dimensions')[0].split())
    check('the hull subsection sits after the 2-adic route and before the chirality subsection',
          text.index(r'\subsection{The orientation spectrum, and the 2-adic route}') < text.index(r'\label{sec:hull}')
          < text.index(r'\subsection{Why Three Dimensions Do Not Automatically Resolve Chirality}'))
    check('it is referenced from the introduction, the conclusion and the abstract',
          r'\ref{sec:hull}' in introduction() and r'\ref{sec:hull}' in conclusion_section()
          and 'substitution hull' in abstract())
    check('its theorems, lemma, corollary, definition and remark are labelled',
          all(('\\label{%s}' % k) in text for k in ('thm:hull', 'thm:sym', 'cor:hull', 'lem:blocks', 'def:block',
                                                    'rem:order8')))
    check('the 2-adic subsection points to the theorem and no longer calls the hull open',
          'is left open upstream' not in text and r'Theorem~\ref{thm:hull} below' in text)
    check('the quoted and recomputed counts are the live ones',
          hu['legal'] == 1410 and hu['romanov']['admissible'] == 360189 and ht['max_order'] == 8
          and '360{,}189' in hull and ('so %d blocks are legal' % hu['legal']) in hull
          and ('exactly %d periodic blocks' % hu['fixed']) in hull
          and ('order $1$ for %d of them and $8$ for %d' % (hu['stabilisers_fixed'][1], hu['stabilisers_fixed'][8])) in hull
          and ', '.join(str(x) for x in hu['per_depth']) in hull)
    check('the inputs (R) and (H) are named as quoted and the admissible count as not repeated',
          '(R)' in hull and '(H)' in hull and 'are not re-derived here' in hull
          and 'does not repeat it' in hull and 'conditional on' in hull)
    check('the statements match the quoted Fact',
          ht['hull'] and set(ht['groups']) >= {'1', 'C4', 'D4'} and r'$1$, $C_{2}^{\mathrm{diag}}$' in hull
          and r'|\mathrm{Sym}(T)|\le8' in hull and 'does not conflict with Fact' in hull)
    check('the einstein property is still called Tsiokos\'s claim',
          'which remain Tsiokos\'s claim' in hull)
    tsio = text.split(r'\bibitem{Tsiokos1}')[1].split(r'\bibitem{')[0]
    check('2609.23783 is no longer credited to Tsiokos; the three new references are listed and cited',
          '2609.23783' not in tsio and 'arXiv:2609.19214' in tsio
          and all((r'\bibitem{%s}' % k) in text and (r'\cite{%s' % k in text or (',%s}' % k) in text)
                  for k in ('GoodmanStrauss', 'Flicker', 'Romanov')))
    check('the Chair44 table has the symmetry-group row', 'symmetry groups of a tiling' in text)
    check('chair44_hull recomputes the legal blocks, their closure and the fixed seeds', HU.run_tests(verbose=False))

    print('lines through a tiling: box means against line means')
    import chair44_lines as LN
    ln = F.get('chair_lines').value
    lines = ' '.join(text.split(r'\label{sec:lines}')[1].split(r'\subsection{Why Three Dimensions')[0].split())
    check('the lines subsection follows the hull subsection and precedes the chirality subsection',
          text.index(r'\label{sec:hull}') < text.index(r'\label{sec:lines}')
          < text.index(r'\subsection{Why Three Dimensions Do Not Automatically Resolve Chirality}'))
    check('its propositions and table are labelled and referenced',
          all(('\\label{%s}' % k) in text for k in ('prop:stack', 'prop:lineaction', 'tab:corner'))
          and r'Proposition~\ref{prop:stack}' in lines and r'Table~\ref{tab:corner}' in lines)
    check('the live counts are in the text: six stack labels, 168 labels, 14 cases, heights -36 and +36',
          len(ln['stack_labels']) == 6 and 'Six of the %d labels' % F.get('chair_legal_blocks').value['labels'] in lines
          and ('That is $%d$ cases' % ln['tube']['cases'][0]) in lines and ln['tube']['cases'][0] == 14
          and '$-36$ and $+36$' in lines)
    check('every row of the corner-line table is the Fact',
          all(('$%s$ & ' % ('(' + ','.join(str(x) for x in b) + ')')) in lines for b in ln['corner'])
          and '(0,1,1)' in lines and '(-1,-1,-1)' in lines)
    check('the conditional proposition names its assumption and excludes irrational directions',
          'Assume that the hull' in lines and 'not uniquely ergodic' in lines and 'irrational slope' in lines
          and 'conditional on the internal space' in lines)
    check('the paper does not claim that all lines fail or that the sums are bounded or ballistic for every line',
          'every line' not in lines.lower() and 'depends on the observable and on the line' in lines)
    check('chair44_lines recomputes the box means, the stacks and the corner classes', LN.run_tests(verbose=False))

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
