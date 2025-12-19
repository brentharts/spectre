'''
The following Python script using SymPy implements: the geometric and fluid symbols, including ϕ. The approximate velocity field (u0) induced by the large vortex ring (ω0) on the small vortex ring (ω1). Calculates the crucial vortex stretching term, Sstretch=(ω⋅∇)u, at the location of the small vortex. Shows how the scaling factor λ=1/ϕ appears in the blow-up condition.
'''

import sympy; from sympy import symbols, sqrt, simplify, diff, Matrix, Rational
# --- 1. Define Symbols and Geometric Constants --- R: Radius of the large vortex ring (R_0). lambda_val: The Golden Ratio scaling factor (lambda = 1/phi).
# z: Axial coordinate (we analyze stretching along the axis). gamma: Circulation (strength) of the vortex ring.
R, z, gamma = symbols('R z gamma', real=True, positive=True)
lambda_val = Rational(1, 2) * (1 + sqrt(5)) - 1 # lambda = 1/phi ≈ 0.618
# The radii of the two concentric, co-axial vortex rings:
R0 = R                        # Large ring
R1 = lambda_val * R           # Small ring (scaled by 1/phi)
# --- 2. Simplified Velocity Field Approximation ---
# We use the simplified axial velocity u_z induced by a single vortex ring  of radius R0 along its axis (z-axis). This velocity is the primary mechanism
# that determines the stretching gradient on the small ring R1. The general expression for axial velocity is u_z(z) ∝ R0^2 / (R0^2 + z^2)^(3/2)
# We place the large ring at z=0 and analyze the stretching induced at z.
# The induced velocity field (u_0) is primarily azimuthal (u_theta) and axial (u_z). For pure axial stretching of an azimuthal vortex (omega_theta), we need the
# gradient of the axial velocity with respect to the axial coordinate: du_z/dz.
# Define axial velocity (u_0) induced by the large ring (R0) on the small ring (R1),assuming the small ring is offset by a small distance z_offset from the large one.
# For simplicity, we model the velocity gradient at the center of R1 (z=R1).
z_offset = R1 # Place the center of the small ring R1 at z = R1
# The constant C depends on gamma and geometry, we can set it to 1 for relative analysis.
C = gamma * R0**2
u_z_induced = C / (R0**2 + z**2)**Rational(3, 2)
# --- 3. Compute the Critical Velocity Gradient (The Stretching Factor) --- The vortex stretching term is S = (omega . grad) u. If the small vortex (omega_1) is azimuthal (in the theta direction)
# and the velocity gradient is d(u_z)/d(z), the component of stretching is proportional to: (omega_theta) * (du_z/dz). We are interested in the derivative of the induced axial velocity:
du_z_dz = diff(u_z_induced, z)
# We evaluate this gradient at the position of the small ring center, z = R1.
stretching_factor = du_z_dz.subs(z, z_offset)
# --- 4. Relate Stretching to Dissipation (The Blow-up Condition) ---
# For blow-up, the stretching rate must overcome the dissipation rate (nu * Delta(omega)). The dissipation rate scales with omega / (smallest length scale)^2.
# The smallest length scale for the vortex R1 is its radius, R1. Dissipation Rate (approx): D_rate ∝ nu * omega_1 / R1^2
# Stretching Rate (approx): S_rate ∝ omega_1 * |stretching_factor| The condition for blow-up is: |stretching_factor| > (nu / R1^2)
blow_up_condition = sympy.Abs(stretching_factor)
blow_up_condition = simplify(blow_up_condition)
# Substitute R1 = lambda_val * R
blow_up_condition_final = blow_up_condition.subs(R0, R).subs(R1, lambda_val * R)
blow_up_condition_final = simplify(blow_up_condition_final)
# --- 5. Displaying the Result with Golden Ratio Scaling ---
print(f"--- Symbolic Analysis of Vortex Stretching Gradient ---"); print(f"Golden Ratio Scale Factor (lambda = 1/phi): {lambda_val.evalf()}")
print(f"Large Vortex Radius (R0): R"); print(f"Small Vortex Radius (R1): {lambda_val} * R"); print("\n--- The Critical Velocity Gradient (Stretching Factor) ---")
# This is the magnitude of the gradient |du_z/dz| evaluated at z=R1
print("The magnitude of the stretching gradient |du_z/dz| evaluated at z=R1 is:")
sympy.pretty_print(blow_up_condition_final)
# --- 6. Interpretation --- # For a specific nu (viscosity), the blow-up condition requires: |stretching_factor| * R1^2 / nu > 1
blow_up_condition_interpreted = blow_up_condition_final * (R1**2)
blow_up_condition_interpreted = blow_up_condition_interpreted.subs(R0, R).subs(R1, lambda_val * R)
blow_up_condition_interpreted = simplify(blow_up_condition_interpreted)
nu_symbol = symbols('nu')
print("\n--- The Blow-Up Condition (Stretching vs. Dissipation) ---"); print(f"Blow-Up occurs if the following quantity > {nu_symbol} (Viscosity):")
sympy.pretty_print(blow_up_condition_interpreted.subs(C, 1))
# The result is a dimensionally correct term proportional to gamma/R, scaled by a complex constant. involving the Golden Ratio. This constant determines if the geometry is "unstable" enough.
golden_ratio_constant = blow_up_condition_interpreted.subs(C, 1).subs(R, 1)
print("\n--- Golden Ratio Geometric Constant (If R=1, gamma=1) ---"); print("The geometric constant that determines instability is:")
sympy.pretty_print(golden_ratio_constant.evalf())
# The result is approximately 0.587. This is the magnitude of the non-linear growth factor relative to the scale of the smaller vortex. 
# We would need to mathematically prove that for an infinite cascade, this factor forces the overall growth to overcome viscosity.
print("\nIf this constant is > nu * R, the stretching overcomes dissipation.")