"""Item 6: exact wandering exponent (Harris-Luck) from the 9x9 substitution matrix, and the
Jordan structure of the 16x16 lift Phi (Jordan blocks at |mu|=1 give log^p fluctuations)."""
import sys; sys.path.insert(0, ".")
import sympy as s, numpy as np
import spectre_patch as spm, lift4d_matrix as lm
M = s.Matrix(np.array(spm.substitution_matrix(), dtype=int).tolist())
x = s.symbols("x")
print("subst charpoly:", s.factor(M.charpoly(x).as_expr()))
ev = sorted(((abs(complex(s.N(k))), k, m) for k, m in M.eigenvals().items()), key=lambda t: -t[0])
for a, k, m in ev: print(" eig", k, "mult", m, "|.|=%.6f" % a)
l1 = ev[0][0]; l2 = max(a for a, k, m in ev[1:]) if len(ev) > 1 else 0
om = (np.log(l2) / np.log(l1)) if l2 > 0 else float("-inf")
print("omega = log|l2|/log l1 = %.6f ; Harris-Luck Ising 2D (nu=1): omega_c = 1/2 ->" % om,
      "relevant" if om > 0.5 else ("marginal" if abs(om - .5) < 1e-9 else "irrelevant"))
J = M.jordan_form()[1]; print("subst Jordan blocks:", [J[i, i+1] for i in range(J.rows-1)])
P = s.Matrix(lm.phi_matrix().tolist()) if hasattr(lm.phi_matrix(), "tolist") else s.Matrix(lm.phi_matrix())
Pp, PJ = P.jordan_form()
print("Phi Jordan diag:", [PJ[i, i] for i in range(16)])
print("Phi superdiag:", [PJ[i, i+1] for i in range(15)])
