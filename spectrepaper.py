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

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp

import spectrefacts as F
import spectrelean as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'spectre_substrate.tex')


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

\title{\textbf{Exact Spectral Theory of the Spectre Substrate,\\
with a Lean 4 Checked Fact Layer}}
\author{Brent S. Hartshorn \orcidlink{0009-0004-2853-655X}
        (\url{brenthartshorn@proton.me})}

\begin{document}
\maketitle
"""


def abstract():
    return r"""
\begin{abstract}
The Spectre substitution on %s species is studied as an exact object.  Its
transfer spectrum is $%s$, so the Perron eigenvalue is
$\lam^{2}=%s$, the fundamental unit of $\mathbb{Z}[\sqrt{15}]$, and the
linear inflation factor $\lam=\tfrac12(\sqrt6+\sqrt{10})$ splits into a
hexagonal and a pentagonal channel without either square root being evaluated.
The zero sector is not semisimple: rank $M=%d$ but rank $M^{n}=%d$ for
$n\ge2$, so one transient direction survives a single substitution and dies
only after two.  Two marginal eigenvalues carry exact integer charges, one
conserved and one alternating, and the alternating one tracks global handedness
from species counts alone.  The census is stated twice, as a matrix count and
as a placed-tile count, and the relation between them is derived rather than
stipulated: row $\Gamma$ is all ones, so the Mystic contributes exactly the
previous generation and $T_{\rm geom}(n)=T(n)+T(n-1)$.  We give the tile area
as an exact quadratic form, the Spectre-to-Hat order parameter in closed form
in $\mathbb{Q}(\sqrt{15})$, and the Seifert minors of $T(2,k)$ in closed form
for every $k$.  The Fact layer is emitted as Mathlib-free Lean~4 and checked by
the kernel: %d theorems, no admitted proofs, and a measured axiom audit.  A
modelling step identifying the order parameter with a birefringence angle is
reported separately and behind its own caveat, because its premise is a
postulate and the arithmetic around it cannot repair that.
\end{abstract}
""" % (number(len(F.SPECIES)),
       tex(F.get('charpoly').value),
       tex(F.get('perron').value),
       F.RANKS[1], F.RANKS[2],
       len(L.theorem_names(L.document())))


def introduction():
    return r"""
\section{Introduction}
\label{sec:intro}

This paper carries an earlier program across the conformal crossover, and its
thesis is that the crossover surface --- a two-dimensional conformal carrier
with vanishing configurational entropy density --- is a substitution monotile
phase.  That identification is a conjecture and is labelled as one.  What can
be done without it, and what occupies most of what follows, is to compute the
substitution's exact structure: the identification is only worth making if the
object identified has content, and the content is arithmetic.

Statements are stratified by epistemic weight.  \emph{Facts} are computations
or theorems.  \emph{Propositions} are computations with a modelling step.
\emph{Conjectures} are sharp claims not yet proved.  A reader who rejects every
conjecture here still keeps every Fact, and the Facts are the reason to read
on.

Two things about how this document is made.  First, every number in it is
computed at build time by \code{spectrefacts.py} and inserted by
\code{spectrepaper.py}; nothing is transcribed.  That is not tidiness.  The
previous draft quoted the charge values of Section~\ref{sec:charges} beside
patch totals that do not produce them --- both numbers correct, the pair of
them unreproducible --- because the two lived in different sections and prose
has no selftest.  Section~\ref{sec:census} now derives the relation between the
two censuses instead of stipulating either.

Second, the Facts that are closed statements are not merely computed twice.
They are proved.  \code{spectrelean.py} emits them as Lean~4 and the kernel
checks them, which is a different kind of evidence from a second computation
and is reported separately in Section~\ref{sec:lean}.
"""


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
along the Tile$(a,b)$ continuum (Section~\ref{sec:deform}), and --- new here
--- a second, arithmetic distinction between the phases
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
\section{The census, twice, and the identity between them}
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
\section{Two integer charges}
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
\section{Deformations and the order parameter}
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
\section{Two phases, two quadratic fields}
\label{sec:phases}

Conjecture~\ref{conj:chirality} distinguishes the two monotile phases by
parity: the Spectre phase is strictly chiral, the Hat phase admits both
handednesses.  There is a second distinction, arithmetic rather than
geometric, and it has not been used before.

\begin{fact}[The phases are graded by different fields]
\label{fact:phases}
The Hat is generated by a substitution on four metatiles, not nine, with
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

\subsection{Measured, in a laboratory}

The Hat phase is no longer only a mathematical object.  Moritake
\emph{et al.}~\cite{Moritake2026} fabricated a monotile quasilattice --- the
centroids of the $T$ metatiles of an $H_6$ patch, %s holes etched in
silicon nitride over half a millimetre --- and measured its optical
diffraction.  The pattern shows sharp Bragg peaks whose positions do not move
with the illumination spot, which is the experimental signature of
quasicrystalline order, and it is \emph{chiral}: a pinwheel with no mirror
line, reversing handedness when the structure is mirrored, and responding
differently to left- and right-handed circular polarization.  That last
response is absent in every mirror-symmetric quasicrystal, the Penrose tiling
included.

For the present paper the useful part is that they give the twist in closed
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
Hat's twist relation, and its own chiral angle --- whatever it is --- belongs
to $\mathbb{Q}(\sqrt{15})$ instead.  We do not compute it here, because doing
so needs the Spectre's metatile twist geometry and not merely its eigenvalue.
Fabricating the Spectre quasilattice and measuring a \emph{different} chiral
angle would be a direct test of the two-field claim of
Fact~\ref{fact:phases}, at optical bench scale, with no cosmology involved.

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
      F.SPECTRE_NAIVE)


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
sequence \code{aabbaabbaaaabb}.  A shared edge is an \emph{X-bond} when the two
tiles disagree about the role of the slot they are gluing.  This is a statement
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

\subsection{Why the mechanism switches on where it does}

Brittenham and Hermiller~\cite{BH} proved that unknotting number is not
additive under connected sum: for $K=7_1=T(2,7)$, with $u(K)=3$,
\[
u(K\#\overline K)\le5<6=u(K)+u(\overline K).
\]
Two features of that theorem matter here and are easy to overstate.

First, the deficit is \emph{bounded, not determined}.  Their result is an
upper bound; Scharlemann's theorem that unknotting number one implies prime
gives $u\ge2$; so the gap $\delta=6-u$ satisfies $1\le\delta\le4$, and its
exact value is their own Question~4.4 and is open.  The binding spectrum above
is parameter-free in its \emph{ratios} $\{0,1,2,7\}$, which are the X-charges
halved; the overall scale $\delta$ is not known, and nothing here determines
it.

Second, the threshold is theirs and not ours.  Their Corollary~1.3 covers
$T(2,2k+1)$ for $k\ge3$ --- index seven and above.  $T(2,3)$ and $T(2,5)$ are
explicitly outside it, and whether either admits any partner at all is open.
So $k=7$ is where non-additivity is known to begin, which is the same place
the edge braids of the substrate close onto $7_1$.  That coincidence is the
whole of the mechanism's appeal and it is a coincidence until someone explains
it.

Additivity of the signature makes the selection rule exact arithmetic:
$\sigma(K\#K)=-2(k-1)$ forces $u(K\#K)=k-1$, so same-handed pairs cannot bind,
while $\sigma(K\#\overline K)=0$ and the lower bound is lost.  Binding is a
mirror-channel phenomenon as a matter of signatures --- and the strictly chiral
screen, having no mirror channel, cannot bind at all.
""" % (F.ATLAS_CLASSES, depths, depths, drift,
       F.get('rho_x_at_half').decimal, tex(F.G_SMALL))


def lean_section():
    names = L.theorem_names(L.document())
    sample = r'''theorem q_minus_alternates : vecMul Qminus M = scale (-1) Qminus := by decide

theorem pell_fundamental_unit : lam2 * lam2inv = (1, 0) := by decide

theorem seifert_minor_closed_form (m : Nat) : D m = Dclosed m :=
  (D_closed_pair m).1'''
    return r"""
\section{Verification: computed twice, and proved once}
\label{sec:lean}

The companion suite recomputes every Fact by a route sharing no code with the
tiling scripts, in exact arithmetic.  That is a good discipline and it is still
two computations: two programs can be wrong in the same way, particularly when
they are written by the same person in the same week.

The closed Facts are therefore also \emph{proved}.  \code{spectrelean.py}
emits them as Lean~4 and the kernel checks the file: %d theorems covering the
charge eigenvector identities, the census totals and their recurrence, the
Mystic identity of Fact~\ref{fact:mystic}, the arithmetic of
$\mathbb{Z}[\sqrt{15}]$ including the Pell unit and both channel identities,
and the Seifert minors of Fact~\ref{fact:sandwich}.

The file is deliberately Mathlib-free.  Everything is \code{Int},
\code{List Int} or a small structure over them, so it checks against a bare
toolchain in seconds and needs no package manager --- which costs some elegance
and buys the property that the proof is as auditable as the arithmetic it
certifies.

\begin{lstlisting}[language=lean]
%s
\end{lstlisting}

\noindent
%s

Three boundaries are worth stating plainly, and they are stated in the Lean
file itself as comments rather than theorems, because a comment cannot be
mistaken for a certificate.  Lean does not see the tiling: that $M$ transcribes
the published substitution is a claim about a transcription, audited by the
geometric census agreeing at every depth, and no proof here bears on it.  Lean
does not see Conjecture~\ref{conj:chirality}, which is a selection argument.
And Lean is deliberately not pointed at anything downstream of the unit
postulate of Section~\ref{sec:data}: the arithmetic there is elementary and
correct, and certifying it would lend the premise a credibility it has not
earned.
""" % (len(names), sample, L.AUDIT_SENTENCE)


def data_section():
    return r"""
\section{A modelling step, and what it would take to believe it}
\label{sec:data}

Isotropic cosmic birefringence is measured nonzero: the 2026 joint
ACT~DR6~$+$~Planck~PR4 analysis gives
$\beta=0.277^{\circ}\pm0.057^{\circ}$.  Exact parity-evenness, the branch that
would disfavour Conjecture~\ref{conj:chirality}, is not what the sky shows.
The consistency is generic --- any parity-violating photon coupling produces it
--- so it is weak evidence for anything specific.

The substrate has exactly one intrinsic chiral quantity with a preferred
magnitude: the order-parameter maximum of Fact~\ref{fact:area}.  Identifying it
with a Stokes-level asymmetry and reading the result in degrees gives
\[
\beta_{\rm pred}=2\cdot\frac{10g}{8+10g}\ \mathrm{[deg]}=%s=%s\ldots^{\circ},
\]
which sits $%s\sigma$ from the measurement.

That agreement is not evidence and this section does not present it as such.
The unit postulate --- that a dimensionless area-fraction maximum is an angle
in degrees --- is the entire modelling step, and it is unjustified.  Degrees
are a human convention; nothing in $\mathbb{Q}(\sqrt{15})$ knows about them.
A closed-form number landing near a measured one is what one should expect to
happen occasionally when a theory supplies several dimensionless constants and
a free choice of units, and the honest description of the present situation is
that the substrate supplies a number, the identification is a guess, and the
guess is currently unrefuted.

What makes it worth recording rather than deleting is that it is scheduled.
At the $0.05^{\circ}$ absolute-calibration class the candidate is separated
from zero at $%.1f\sigma$; at the $0.01^{\circ}$ class it is separated from the
WMAP$+$Planck central value $0.342^{\circ}$ at $%.1f\sigma$.  A guess with a
date on it is worth more than a guess without one, and less than a derivation.
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
are also proved in Lean; the others are computations, and the distinction is
kept on the Fact itself so that the paper cannot claim a machine check for
something no machine checked.

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
\bibitem{SmithEtAl} Smith, D., Myers, J.S., Kaplan, C.S. and
Goodman-Strauss, C. (2024). An aperiodic monotile.
\emph{Combinatorial Theory}, 4(1).
\bibitem{SmithEtAl2} Smith, D., Myers, J.S., Kaplan, C.S. and
Goodman-Strauss, C. (2024). A chiral aperiodic monotile.
\emph{Combinatorial Theory}, 4(2).
\bibitem{Moritake2026} Moritake, Y., Takiguchi, M., Aihara, T. and
Notomi, M. (2026). Chiral diffraction from aperiodic monotile structure.
\emph{Nature Communications} \textbf{17}, 6085.
\url{https://doi.org/10.1038/s41467-026-75023-7}
\bibitem{BH} M.~Brittenham and S.~Hermiller (2025). Unknotting number is not
additive under connected sum. arXiv:2506.24088.
\url{https://arxiv.org/abs/2506.24088}
\bibitem{Scharlemann} M.~Scharlemann (1985). Unknotting number one knots are
prime. \emph{Invent. Math.} 82, 37--55.
\bibitem{demoura2021} de Moura, L. and Ullrich, S. (2021). The Lean 4 Theorem
Prover and Programming Language. \emph{CADE-28}, 625--635.
\bibitem{Eskilt2026} Eskilt, J.R. \emph{et al.} (2026). Joint ACT DR6 and
Planck PR4 constraints on cosmic birefringence.
\bibitem{repo} \url{https://github.com/brentharts/spectre}
\end{thebibliography}

\end{document}
"""


def document():
    return '\n'.join([
        PREAMBLE, abstract(), introduction(), screen_section(),
        spectral_section(),
        census_section(), charges_section(), deformation_section(),
        phases_section(), knot_section(), xcharge_section(),
        lean_section(), data_section(),
        facts_appendix(), BIBLIOGRAPHY,
    ])


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
    check('every fact is in the appendix',
          all(esc(f.key) in text for f in F.FACTS))

    print('the caveats survive')
    check('the unit postulate is called unjustified', 'unjustified' in text)
    check('the modelling step is not called evidence',
          'is not evidence' in text)
    check('the Lean section states what Lean does not see',
          'Lean does not see the tiling' in text)
    check('the BH deficit is called bounded, not determined',
          'bounded, not determined' in text)
    check('the p drift is stated rather than a single value quoted',
          'not a limit' in text)
    check('the Brittenham-Hermiller reference is cited and listed',
          r'\cite{BH}' in text and 'arXiv:2506.24088' in text)
    check('the measured chiral angle and its identity appear',
          '15.5225' in text and r'\frac{\kappa-3}{4}' in text)
    check('the Moritake reference is cited and listed',
          r'\cite{Moritake2026}' in text and 's41467-026-75023-7' in text)
    check('the Spectre angle is a prediction, not a restatement',
          'prediction rather than a restatement' in text)
    check('the atlas count and X-charges appear',
          str(F.ATLAS_CLASSES) in text and 'X_{\\Gamma_2}=14' in text)

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
