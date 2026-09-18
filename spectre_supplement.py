#!/usr/bin/env python3
r"""spectre_supplement.py -- the Fact table, as a document of its own.

The main paper used to carry every Fact as an appendix. That was the right
instinct -- a paper whose numbers are all computed should show its work --
and the wrong place for it: a hundred-row table of keys and methods is a
finding aid, and it doubled the length of a document meant to be read.

This module emits it as separate supplementary material. It reuses
`spectrepaper.facts_appendix()` unchanged, so the table here is byte-for-byte
the table the main paper used to carry, drawn from the same `spectrefacts`
source that produces every number in the paper. The paper's build asserts
that the Fact count it quotes matches the count in this document. A Fact
present in one and absent from the other fails that build, which is the
only kind of guarantee that survives the two files being edited on
different days.

    python3 spectre_supplement.py           # write spectre_supplement.tex
    python3 spectre_supplement.py --stdout
    python3 spectre_supplement.py --selftest
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import spectrefacts as F
import spectrelean as L
import spectrepaper as P

OUTPUT = os.path.join(HERE, 'spectre_supplement.tex')

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{longtable,booktabs,array}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{orcidlink}
\usepackage{microtype}
\hypersetup{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black}
\newcommand{\lam}{\lambda}
\newcommand{\code}[1]{\texttt{#1}}

\title{\vspace{-1cm}
\textbf{Supplementary Material:} \\ Every Fact, and how it was obtained \\
{\large for \emph{One Tile, Two Units: Area Inflation in $\mathbb{Q}(\sqrt{15})$
and Boundary Inflation in $\mathbb{Q}(\sqrt{5})$ for the Spectre Monotile}}}

\author{Brent S. Hartshorn \orcidlink{0009-0004-2853-655X}
        (\url{brenthartshorn@proton.me})}

\begin{document}
\maketitle
"""


def preface():
    n = len(F.FACTS)
    nl = sum(1 for f in F.FACTS if f.lean)
    nt = len(L.theorem_names(L.document()))
    return r"""
\section*{What this is}

The main paper stratifies its statements by epistemic weight: \emph{Facts}
are computations or theorems, \emph{Propositions} carry one modelling step,
\emph{Conjectures} are sharp and unproved.  This document lists every Fact
--- all %d --- with the key under which \code{spectrefacts.py} computes it,
the claim in the form the paper states it, the first sentence of the method
by which it was obtained, and a mark on the %d Facts that are also proved by
the Lean~4 kernel.

Three things about the table.

\begin{itemize}
\item \textbf{It is generated, not written.}  Every row is emitted at build
  time from \code{spectrefacts.py}, the same module that produces every
  number in the main paper.  There is no second copy to drift.

\item \textbf{The Lean column is measured, not asserted.}  A checkmark means
  the Fact's \code{lean} field names a theorem that exists in
  \code{SpectreFacts.lean} --- %d theorems, kernel-accepted, no admitted
  proofs --- and the paper's build fails if a named theorem is missing.  Six
  such phantom attributions were found and removed when that check was
  added; the count of checked Facts fell from twenty-two to eighteen, and
  eighteen is the honest number.

\item \textbf{The method column is the first sentence.}  A method that runs
  to a paragraph is truncated at its first full stop or semicolon.  The full
  text is in the module, one lookup by key away.
\end{itemize}

Two of the Facts are inputs rather than results, and the paper says so:
\code{charpoly} rests on the transcribed substitution matrix, and
\code{boundary\_recurrence} on a perimeter seed counted from placed tiles.
Both are audited by an independent route --- the matrix against a second
implementation of the same substitution, the seed against an exact
$\mathbb{Q}(\sqrt3)$ recount --- and an audit is not a proof.  They are the
two rows a sceptical reader should look at first.
""" % (n, nl, nt)


def document():
    body = P.facts_appendix()
    # the appendix opened with \appendix and its own \section; here it is
    # the whole document, so those become the body directly
    body = body.replace(r'\appendix', '', 1)
    body = body.replace(r'\section{Every Fact, and how it was obtained}',
                        r'\section*{The table}', 1)
    return '\n'.join([PREAMBLE, preface(), body, r'\end{document}', ''])


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
    main = P.document()

    print('the table is complete')
    check('every Fact key appears in the supplement',
          all(P.esc(f.key) in text for f in F.FACTS))
    check('the quoted Fact count is the real one',
          ('all %d' % len(F.FACTS)) in text)
    check('the quoted Lean count is the real one',
          ('the %d Facts' % sum(1 for f in F.FACTS if f.lean)) in text)

    print('the two documents agree')
    check('the main paper quotes the same Fact count',
          ('all %d of them' % len(F.FACTS)) in main)
    check('the main paper no longer carries the table itself',
          r'\begin{longtable}' not in main)
    check('and points at this document',
          'spectre_supplement.pdf' in main)
    check('the main paper title is the one this document names',
          'One Tile, Two Units' in main and 'One Tile, Two Units' in text)

    print('the honesty notes survive')
    check('the phantom attributions are mentioned',
          'phantom attributions' in text)
    check('the two input Facts are named as inputs',
          r'\code{charpoly}' in text and r'boundary\_recurrence' in text)
    check('an audit is called not a proof',
          'an audit is not a proof' in text)

    print()
    if failures:
        print('%d failure(s): %s' % (len(failures), ', '.join(failures)))
    else:
        print('spectre_supplement: all checks pass.')
    return len(failures)


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(1 if selftest() else 0)
    elif '--stdout' in sys.argv:
        print(document())
    else:
        print('wrote ' + write())
