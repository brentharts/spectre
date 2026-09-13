# The paper is generated. Edit the facts, not the .tex.
PYTHON ?= python3
export PYTHONPATH := ..

all: check paper

facts:
	$(PYTHON) spectrefacts.py --selftest

lean:
	$(PYTHON) spectrelean.py --check

paper: facts
	$(PYTHON) spectrepaper.py
	pdflatex -interaction=nonstopmode spectre_substrate.tex >/dev/null
	pdflatex -interaction=nonstopmode spectre_substrate.tex >/dev/null

check: facts lean
	$(PYTHON) spectrepaper.py --selftest

clean:
	rm -f spectre_substrate.tex spectre_substrate.pdf *.aux *.log *.out
	rm -rf __pycache__

.PHONY: all facts lean paper check clean
