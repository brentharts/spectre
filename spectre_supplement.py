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
\usepackage[T1]{fontenc}
\usepackage{lmodern}
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
{\large for \emph{@TITLE@}}}

\author{Brent Hartshorn \orcidlink{0009-0004-2853-655X}
        (\url{brenthartshorn@proton.me}) and Tristan Romanov}

\begin{document}
\maketitle
"""


def preface():
    n = len(P.paper_facts())
    nf = P.document().count(r'\begin{fact}')
    nl = sum(1 for f in P.paper_facts() if f.lean)
    nt = len(L.theorem_names(L.document()))
    return r"""
\section*{What this is}

The main paper stratifies its statements: \emph{Facts} are exact statements,
each proved, verified by an exact finite computation over a stated range, or
quoted; \emph{Propositions} are of narrower scope, proved or computed, with their
modelling steps and hypotheses stated in them.  The numbered Facts of the
paper (%d of them) are drawn from a ledger of finer entries.  This document
lists every ledger entry the paper draws on --- all %d --- with the key under
which the ledger (\code{spectrefacts.py}, \code{chair44facts.py},
\code{monotilefacts.py}) computes it, the claim, the first sentence of the
method by which it was obtained, and a mark on the %d entries that name a
theorem of the Lean~4 file.  The ledger is shared with other documents; their
entries are not listed here.

Three things about the table.

\begin{itemize}
\item \textbf{It is generated, not written.}  Every row is emitted at build
  time from the ledger modules, the same modules that produce every
  number in the main paper.  There is no second copy to drift.

\item \textbf{The Lean column is measured, not asserted.}  A checkmark means
  the Fact's \code{lean} field names a theorem that exists in
  \code{SpectreFacts.lean} --- %d theorems, kernel-accepted, no admitted
  proofs --- and the paper's build fails if a named theorem is missing; six
  phantom attributions were found and removed when that check was added.  A
  Lean theorem proves the Lean statement, which can be narrower than the
  paper's: the entries marked $^{*}$ are proved in Lean for the sequence that the
  Lean file \emph{defines} by the perimeter recurrence.  That this sequence
  is the supertile perimeter at every order is Theorem~1 of the paper,
  proved by an exact finite certificate
  (\code{boundary\_certificate/}), not in Lean.

\item \textbf{The method column is the first sentence.}  A method that runs
  to a paragraph is truncated at its first full stop or semicolon, and
  then, marked by an ellipsis, at about a hundred characters.  The full
  text is in the module, one lookup by key away.
\end{itemize}

One entry is an input rather than a result, and the paper says so:
\code{charpoly} rests on the transcribed substitution matrix (the published
rule).  It is audited against a second implementation of the same
substitution, and an audit is not a proof.  The perimeter recurrence,
\code{boundary\_recurrence}, was once in the same position, counted from
placed tiles for $k\le6$; it is now proved for every $k$ by the finite
certificate of Theorem~1 (\code{boundary\_certificate/}), and its row
says so.
""" % (nf, n, nl, nt)


def document():
    body = P.facts_appendix()
    # the appendix opened with \appendix and its own \section; here it is
    # the whole document, so those become the body directly
    body = body.replace(r'\appendix', '', 1)
    body = body.replace(r'\section{Every Fact, and how it was obtained}',
                        r'\section*{The table}', 1)
    return '\n'.join([PREAMBLE.replace('@TITLE@', P.TITLE), preface(), body, r'\end{document}', ''])


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
          all(P.esc(f.key) in text for f in P.paper_facts()))
    check('the quoted Fact count is the real one',
          ('all %d' % len(P.paper_facts())) in text)
    check('the quoted Lean count is the real one',
          ('the %d entries' % sum(1 for f in P.paper_facts() if f.lean)) in text)
    check('the quoted count of numbered Facts is the real one',
          ('(%d of them)' % main.count(r'\begin{fact}')) in text)

    print('the two documents agree')
    check('the main paper no longer carries the table itself',
          r'\begin{longtable}' not in main)
    check('the main paper title is the one this document names',
          P.TITLE in main and P.TITLE in text)

    print('the honesty notes survive')
    check('the phantom attributions are mentioned',
          'phantom attributions' in text)
    check('the input entry is named as an input, and the recurrence as proved',
          r'\code{charpoly}' in text and r'boundary\_recurrence' in text and 'now proved for every' in text)
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
