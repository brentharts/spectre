# ---------------------------------------------------------------- the paper
#
# "What a Substitution Determines, and What a Kernel Cannot Reach".
# The paper is generated: edit the facts, not the .tex.
#
#   make check        facts, atlas, Lean kernel, and the generated document
#   make paper        rebuild spectre_substrate.pdf
#   make lean         emit SpectreFacts.lean and check it
#   make install_lean install a Lean 4 toolchain (Ubuntu/Debian, macOS)
#
# `make lean` needs `lean` on PATH.  If you have not got one, `make
# install_lean` will fetch one; see the toolchain section below.

PYTHON ?= python3
export PYTHONPATH := .

# ------------------------------------------------------------ lean toolchain
#
# Two routes, tried in that order.
#
#   elan     the upstream version manager.  Normal case.  It resolves
#            toolchains through release.lean-lang.org, so it needs that host
#            reachable as well as github.com.
#   tarball  the official binary release, unpacked into LEAN_PREFIX.  Used
#            when elan cannot reach its release index (locked-down networks,
#            proxies, CI images).  Only needs github.com.
#
# Neither touches the system prefix and neither needs root for Lean itself.

ELAN_HOME     ?= $(HOME)/.elan
LEAN_PREFIX   ?= $(HOME)/.local/lean
LEAN_VERSION  ?= 4.33.1
LEAN_URL_BASE ?= https://github.com/leanprover/lean4/releases/download
ELAN_INIT_URL ?= https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh

# Put both routes in front of PATH for every recipe below.  This is what lets
# `make lean` work in the same shell that ran `make install_lean`, with no
# sourcing of ~/.profile in between.  The tarball goes first: a half-installed
# elan leaves a `lean` shim that runs but has no toolchain behind it, and a
# real binary should win over that.
export PATH := $(LEAN_PREFIX)/bin:$(ELAN_HOME)/bin:$(PATH)

# --------------------------------------------------------------- the targets
install_cicy:
	cd .. && git clone https://github.com/brentharts/CICY.git
facts:
	PYTHONPATH=../CICY $(PYTHON) spectrefacts.py --selftest

atlas:
	$(PYTHON) spectreatlas.py --selftest

lean:
	@command -v lean >/dev/null 2>&1 || { \
	  echo ''; \
	  echo 'no `lean` on PATH.  Run:  make install_lean'; \
	  echo ''; \
	  exit 1; }
	$(PYTHON) spectrelean.py --check

paper: facts
	$(PYTHON) spectrepaper.py
	pdflatex -interaction=nonstopmode spectre_substrate.tex >/dev/null
	pdflatex -interaction=nonstopmode spectre_substrate.tex >/dev/null

paperfast:
	$(PYTHON) spectrepaper.py
	pdflatex -interaction=nonstopmode spectre_substrate.tex >/dev/null
	pdflatex -interaction=nonstopmode spectre_substrate.tex >/dev/null
	open spectre_substrate.pdf

readme:
	$(PYTHON) spectrefacts.py --readme

check: facts atlas lean readme
	$(PYTHON) spectrepaper.py --selftest

paper-clean:
	rm -f spectre_substrate.tex spectre_substrate.pdf *.aux *.log *.out
	rm -rf __pycache__

# ------------------------------------------------------------- installing it

install_lean: install_lean_deps
	@if command -v lean >/dev/null 2>&1 && lean --version >/dev/null 2>&1; then \
	  echo "lean already present: $$(lean --version)"; \
	  exit 0; \
	fi; \
	echo "==> installing elan into $(ELAN_HOME)"; \
	if curl -fsSL "$(ELAN_INIT_URL)" | sh -s -- -y --no-modify-path \
	     --default-toolchain leanprover/lean4:v$(LEAN_VERSION) >/dev/null 2>&1 \
	   && "$(ELAN_HOME)/bin/lean" --version >/dev/null 2>&1; then \
	  echo "==> elan installed a toolchain"; \
	else \
	  echo "==> elan could not fetch a toolchain; using the binary release"; \
	  $(MAKE) --no-print-directory install_lean_tarball; \
	fi
	@echo ''
	@echo "lean: $$(lean --version)"
	@echo ''
	@echo 'make lean / make check now work.  To get lean in your own shell:'
	@$(MAKE) --no-print-directory lean_env
	@echo ''

# The binary release, unpacked by hand.  No root, no package manager, no
# release index -- just github.com.
install_lean_tarball:
	@set -e; \
	os=$$(uname -s); arch=$$(uname -m); \
	case "$$os" in \
	  Linux) case "$$arch" in \
	           x86_64|amd64)  asset=linux ;; \
	           aarch64|arm64) asset=linux_aarch64 ;; \
	           *) echo "unsupported Linux arch: $$arch"; exit 1 ;; \
	         esac ;; \
	  Darwin) case "$$arch" in \
	           x86_64) asset=darwin ;; \
	           arm64)  asset=darwin_aarch64 ;; \
	           *) echo "unsupported macOS arch: $$arch"; exit 1 ;; \
	         esac ;; \
	  *) echo "unsupported OS: $$os (Linux and Darwin only)"; exit 1 ;; \
	esac; \
	url="$(LEAN_URL_BASE)/v$(LEAN_VERSION)/lean-$(LEAN_VERSION)-$$asset.tar.zst"; \
	command -v curl >/dev/null 2>&1 || { echo "need curl"; exit 1; }; \
	command -v unzstd >/dev/null 2>&1 || { echo "need zstd (unzstd)"; exit 1; }; \
	tmp=$$(mktemp -d); trap 'rm -rf "'"$$tmp"'"' EXIT; \
	echo "==> fetching $$url"; \
	curl -fL --retry 3 -o "$$tmp/lean.tar.zst" "$$url"; \
	echo "==> unpacking into $(LEAN_PREFIX)"; \
	rm -rf "$(LEAN_PREFIX)"; mkdir -p "$(LEAN_PREFIX)"; \
	unzstd -c "$$tmp/lean.tar.zst" \
	  | tar -x -C "$(LEAN_PREFIX)" --strip-components=1; \
	"$(LEAN_PREFIX)/bin/lean" --version >/dev/null

# curl, git and zstd on Debian/Ubuntu; Homebrew on macOS.  Everything else is
# left alone -- the message says what is missing rather than guessing a
# package manager.
install_lean_deps:
	@need=''; \
	for p in curl unzstd; do \
	  command -v $$p >/dev/null 2>&1 || need="$$need $$p"; \
	done; \
	if [ -n "$$need" ]; then \
	  if command -v apt-get >/dev/null 2>&1; then \
	    echo "==> apt-get install curl zstd"; \
	    sudo apt-get update -qq && sudo apt-get install -y curl zstd; \
	  elif command -v brew >/dev/null 2>&1; then \
	    echo "==> brew install curl zstd"; \
	    brew install curl zstd; \
	  else \
	    echo "missing:$$need -- install them and re-run"; exit 1; \
	  fi; \
	fi
	@$(PYTHON) -c 'import sympy, numpy' 2>/dev/null || { \
	  echo ''; \
	  echo 'note: the Python side wants sympy and numpy.  Either'; \
	  echo '  sudo apt-get install -y python3-sympy python3-numpy'; \
	  echo 'or'; \
	  echo '  $(PYTHON) -m pip install --user sympy numpy'; \
	  echo ''; }

# Print the line to paste into ~/.bashrc or ~/.zshrc.
lean_env:
	@if [ -x "$(LEAN_PREFIX)/bin/lean" ]; then \
	  echo '  export PATH="$(LEAN_PREFIX)/bin:$$PATH"'; \
	else \
	  echo '  export PATH="$(ELAN_HOME)/bin:$$PATH"'; \
	fi

uninstall_lean:
	rm -rf "$(LEAN_PREFIX)"
	@echo 'removed $(LEAN_PREFIX).  elan, if you installed it, lives in'
	@echo '$(ELAN_HOME) and is removed with: elan self uninstall'

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

.PHONY: facts atlas lean readme paper check paper-clean \
        install_lean install_lean_tarball install_lean_deps \
        lean_env uninstall_lean
