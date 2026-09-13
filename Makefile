# ---------------------------------------------------------------- the paper
#
# "What a Substitution Determines, and What a Kernel Cannot Reach".
# The paper is generated: edit the facts, not the .tex.
#
#   make check   facts, atlas, Lean kernel, and the generated document
#   make paper   rebuild spectre_substrate.pdf
#   make lean    emit SpectreFacts.lean and check it (needs `lean` on PATH)

PYTHON ?= python3
export PYTHONPATH := .

facts:
	$(PYTHON) spectrefacts.py --selftest

atlas:
	$(PYTHON) spectreatlas.py --selftest

lean:
	$(PYTHON) spectrelean.py --check

paper: facts
	$(PYTHON) spectrepaper.py
	pdflatex -interaction=nonstopmode spectre_substrate.tex >/dev/null
	pdflatex -interaction=nonstopmode spectre_substrate.tex >/dev/null

readme:
	$(PYTHON) spectrefacts.py --readme

check: facts atlas lean readme
	$(PYTHON) spectrepaper.py --selftest

paper-clean:
	rm -f spectre_substrate.tex spectre_substrate.pdf *.aux *.log *.out
	rm -rf __pycache__

# ------------------------------------------------------------ everything else

default:
	./somos8n.py --test10

t2:
	./somos8n.py --test2

t3:
	./somos8n.py --test3

t4:
	./somos8n.py --test4

t5:
	./somos8n.py --test5

t6:
	./somos8n.py --test6

t7:
	./somos8n.py --test7

t8:
	./somos8n.py --test8

t9:
	./somos8n.py --test9


a:
	blender --python einstein_fibration.py -- --iter2 --b=0.5

b:
	blender --python einstein_fibration.py -- --iter2 --b=1

c:
	blender --python einstein_fibration.py -- --iter2 --b=1.5

d:
	blender --python einstein_fibration.py -- --iter2 --b=3

.PHONY: facts atlas lean readme paper check paper-clean
