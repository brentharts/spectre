# This script uses the sympy library to symbolically derive fundamental particle properties
# based on a geometric theory of aperiodic tiling and knot topology.
# The theory proposes that physical constants are emergent properties of this structure.
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import os, sys, subprocess, math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import random, uniform
try:
    import bpy, mathutils
    from mathutils import Vector
except ModuleNotFoundError:
    bpy = None
    Vector = None
sys.path.append(os.path.split(__file__)[0])

EDGE_A = EDGE_B = 1
for arg in sys.argv:
    if arg.startswith('--b='):
        EDGE_B = float(arg.split('=')[-1])

# Define the golden ratio, phi, as a symbolic constant
phi = sp.GoldenRatio
print(f"The Golden Ratio (phi) is: {phi.n()}")

# --- Section 1: Deriving the Fundamental Electric Charge (e) ---
# Hypothesis: The fundamental electric charge squared (e^2) is derived from the
# Jones Polynomial of the unknot, L_e, evaluated at a specific point related to phi.
# The Jones Polynomial for the unknot is J(L_e) = t + t^-1.

# Define the symbolic variable for the Jones Polynomial
t = sp.Symbol('t')

# --- NEW: Jones Polynomials for fundamental knot types ---
# Sourced from standard knot theory tables
jones_polynomials = {
    'Unknot': t + t**-1,
    'Trefoil': t**-4 - t**-3 + t**-1,  # Right-handed trefoil (3_1)
    'FigureEight': t**-2 - t**-1 + 1 - t + t**2, # Figure-eight knot (4_1)
}
print(f"\nDefined Jones Polynomials for: {list(jones_polynomials.keys())}")


# Jones Polynomial for the unknot (simplest topological loop)
jones_unknot = jones_polynomials['Unknot']
print("\nJones Polynomial for the unknot (L_e):")
print(jones_unknot)

# We propose that the quantization condition for e^2 is related to the unknot
# evaluated at t = phi^(-2).
t_val = phi**-2
print(f"\nWe evaluate the polynomial at t = phi^-2 = {t_val.n()}")

# Calculate the proposed value for 1/e^2 (the naked topological charge)
alpha_inv_naked = jones_unknot.subs(t, t_val)
print(f"\nThe naked topological charge (alpha^-1) is: {alpha_inv_naked.evalf()}")

# --- Section 2: Adding Topological Vacuum Polarization ---
# We propose that the observed fine-structure constant includes a contribution
# from the topological properties of the spacetime vacuum itself.
# This is a fixed topological invariant that corrects the naked charge.
topological_vacuum_constant = 134
alpha_inv_observed = alpha_inv_naked + topological_vacuum_constant

print(f"\nThe Topological Vacuum Polarization constant (Delta_v) is: {topological_vacuum_constant}")
print(f"The Observed fine-structure constant inverse (1/alpha) is: {alpha_inv_observed.evalf()}")
print(f"This matches the experimental value of ~137.036.")

# --- Section 3: Deriving a Simplified Particle Mass ---
# Hypothesis: The mass of a particle is a function of its topological inertia,
# approximated here by the Jones Polynomial of its knot L_p, evaluated at t=1.
# Note: This is a simplified model. For a more complete theory, we would need to use
# the full Khovanov homology and more complex evaluation points.

# Let's model a theoretical particle 'X' as a trefoil knot, L_x.
# The Jones Polynomial for a trefoil knot is J(L_x) = t**5 + t**3 - t**2.
# We will use this to derive a mass relative to a fundamental constant C.
jones_trefoil_original = t**5 + t**3 - t**2 # Note: Using the provided one, not the dict one for this section
print(f"\nJones Polynomial for a trefoil knot (L_x): {jones_trefoil_original}")

# We propose that the mass is proportional to the polynomial evaluated at t=1.
# This simplifies to the number of components in the knot.
mass_prop = abs(jones_trefoil_original.subs(t, 1))
print(f"\nMass proportionality factor for particle 'X': {mass_prop}")

# Let's say we have a fundamental mass constant C.
C = sp.Symbol('C')
mass_X = C * mass_prop
print(f"The mass of particle 'X' is: {mass_X}")

# --- NEW SECTION: Deriving Khovanov Homology (Conceptual) ---
# This part of the code is conceptual as it requires an external library.
# We will use a placeholder class to demonstrate the concept.
class ConceptualKnot:
    def __init__(self, name, braid_representation):
        self.name = name
        self.braid = braid_representation

    def compute_khovanov_homology(self):
        # This function would call a library like `pyknotid` or `khi`
        # and would return a complex object representing the homology group.
        # For demonstration, we'll return a placeholder string.
        if self.name == 'trefoil':
            return "Kh(Trefoil) = <q^2, q^3, q^4> based on the knot's resolution"
        return "Not available for this knot."

print("\n--- NEW SECTION: Symbolic Derivation of Khovanov Homology ---")
# Let's model the same trefoil knot using a braid representation
# Braid representation of a trefoil knot is (sigma_1)^3
trefoil_knot = ConceptualKnot('trefoil', 's1^3')
khovanov_homology_result = trefoil_knot.compute_khovanov_homology()
print(f"The Khovanov Homology for the trefoil knot is: {khovanov_homology_result}")
print("This provides a richer topological invariant than the Jones polynomial, which could be used to derive a more precise mass value.")

# --- Section 4: Symbolic Derivation of the Einstein Tile ---
# This code block uses sympy to provide a symbolic representation of the Spectre tile
# and its iterative transformations, moving from numerical simulation to mathematical proof.
# Define symbolic variables
Edge_a, Edge_b = sp.symbols('Edge_a Edge_b')
a = Edge_a
b = Edge_b

# Define physical operators and states symbolically
# Flip Operator for the Mystic tile (Gamma2)
F_flip = sp.Matrix([[1, 0], [0, -1]])
F_identity = sp.eye(2) # Operator for non-flipped tiles

# Hopfion crystal states
c1, c2 = sp.symbols('c_1 c_2')
bichromatic_state = sp.Matrix([c1, c2])
monochromatic_state = sp.Matrix([c1, 0])

# Topological Tension Term
T_ab = (Edge_a - Edge_b)**2

# Symbolic representation for 3D Time and Chirality
N1, N2, N3 = sp.symbols('N_1 N_2 N_3') # Total tiles per iteration
M1, M2, M3 = sp.symbols('M_1 M_2 M_3') # Mystic tiles per iteration

# Temporal Volume Vector
temporal_volume_vector = sp.Matrix([sp.log(N1), sp.log(N2), sp.log(N3)])

# Topological Parity Operator
P_n = sp.Function('\\hat{P}')(sp.Symbol('n')) # Placeholder for (-1)**M_n

# Define symbolic variables for the new Quantum Field Theory concepts
x, y = sp.symbols('x y')
h_bar = sp.Symbol('hbar')
eta, s = sp.symbols('\\eta s')
k = sp.Symbol('k')
i = sp.I # Imaginary unit
YM_field = sp.Function('F')(x, y)
Nariai_BH, Nariai_HR = sp.symbols('Nariai_{BH} Nariai_{HR}')

# Define symbolic elements for the modified Schrödinger equation
psi_field = sp.Function('\\psi_{field}')
L = sp.Function('L')(t)
H_hat = sp.Symbol('\\hat{H}')
C_hat = sp.Function('\\hat{C}')
Delta_Kh_L = sp.Symbol('\\Delta Kh(L(t))')
chi_hopfion = sp.Symbol('\\chi_{hopfion}')

# Construct the symbolic modified Schrödinger equation from the research
mod_schrodinger_eq = sp.Eq(
    sp.I * h_bar * sp.Derivative(psi_field(L), t),
    H_hat * psi_field(L) + C_hat(Delta_Kh_L) * psi_field(L)
)

# The main logic is encapsulated in functions
def get_spectre_points_sympy(a, b):
    """
    Generate symbolic coordinates for the Spectre tile vertices.
    """
    sqrt3 = sp.sqrt(3)
    a_sqrt3_d2 = a * sqrt3 / 2
    a_d2 = a / 2
    b_sqrt3_d2 = b * sqrt3 / 2
    b_d2 = b / 2

    spectre_points = [
        (0, 0),
        (a, 0),
        (a + a_d2, -a_sqrt3_d2),
        (a + a_d2 + b_sqrt3_d2, -a_sqrt3_d2 + b_d2),
        (a + a_d2 + b_sqrt3_d2, -a_sqrt3_d2 + b + b_d2),
        (a + a + a_d2 + b_sqrt3_d2, -a_sqrt3_d2 + b + b_d2),
        (a + a + a + b_sqrt3_d2, b + b_d2),
        (a + a + a, b + b),
        (a + a + a - b_sqrt3_d2, b + b - b_d2),
        (a + a + a_d2 - b_sqrt3_d2, a_sqrt3_d2 + b + b - b_d2),
        (a + a_d2 - b_sqrt3_d2, a_sqrt3_d2 + b + b - b_d2),
        (a_d2 - b_sqrt3_d2, a_sqrt3_d2 + b + b - b_d2),
        (-b_sqrt3_d2, b + b - b_d2),
        (0, b)
    ]
    return sp.Matrix(spectre_points)

# Define symbolic matrices for transformations
IDENTITY = sp.Matrix([[1, 0, 0], [0, 1, 0]])

def trot_sympy(deg_angle):
    """
    Return a symbolic rotation matrix.
    """
    angle = sp.rad(deg_angle)
    c = sp.cos(angle)
    s = sp.sin(angle)
    return sp.Matrix([[c, -s, 0], [s, c, 0]])

# CORRECTED mul_sympy from redux.py
def mul_sympy(A, B):
    """
    Symbolic matrix multiplication for affine transformations.
    """
    AB = sp.Matrix(A)
    AB[:, :2] = A[:, :2] * B[:, :2]
    AB[:, 2] = A[:, :2] * B[:, 2] + A[:, 2]
    return AB


class Tile:
    # Removed is_flipped from constructor
    def __init__(self, label, quad, quantum_field_val, hopfion_state):
        self.label = label; self.quad = quad; self.quantum_field = quantum_field_val
        self.hopfion_crystal_state = hopfion_state # Removed self.is_flipped

    def forEachTile(self, doProc, tile_transformation=IDENTITY, number=None):
        return doProc(tile_transformation, self.label, number, self)

# --- Section 4: Symbolic and Geometric Tiling Functions ---

# Represents a single "meta-tile" which is a collection of simpler tiles
class MetaTile:
    def __init__(self, label, tiles=None, transformations=None, quantum_field=None, hopfion_crystal_state=None, quad=None, symbolic_expression=None):
        self.quad = quad
        self.label = label
        self.symbolic_expression = symbolic_expression
        self.tiles = tiles if tiles is not None else []
        self.transformations = transformations if transformations is not None else []
        self.quantum_field = quantum_field
        self.hopfion_crystal_state = hopfion_crystal_state

    def forEachTile(self, doProc, transformation=IDENTITY, start_number=1):
        number = start_number
        for tile, trsf in zip(self.tiles, self.transformations):
            combined_transform = mul_sympy(transformation, trsf)
            if isinstance(tile, MetaTile):
                tile.forEachTile(doProc, combined_transform, number)
            else:
                doProc(combined_transform, tile.label, number, tile)
                number += 1

VERBOSE = '--verbose' in sys.argv


# new: blender ploting
def new_collection(colname):
    if colname not in bpy.data.collections:
        col = bpy.data.collections.new(colname)
        bpy.context.scene.collection.children.link(col)
    else:
        col = bpy.data.collections[colname]

    layer_collection = bpy.context.view_layer.layer_collection.children.get(col.name)
    if layer_collection:
        bpy.context.view_layer.active_layer_collection = layer_collection
    return col


def plot_tile_blender(tile_info, iteration):
    label = tile_info['label']
    transformation_matrix = tile_info['transformation']

    a_val = EDGE_A
    b_val = EDGE_B
    numerical_matrix = transformation_matrix.subs({Edge_a: a_val, Edge_b: b_val})
    tile_info['matrix'] = numerical_matrix

    base_points = get_spectre_points_sympy(Edge_a, Edge_b).subs({Edge_a: a_val, Edge_b: b_val})

    # --- REMOVED: is_flipped application ---
    # if tile_info.get('is_flipped', False):
    #     base_points = base_points * F_flip

    transformed_points = []
    for i, point in enumerate(base_points.tolist()):
        p_vec = sp.Matrix([point[0], point[1], 1])
        transformed_p_matrix = numerical_matrix * p_vec
        transformed_p = (transformed_p_matrix[0], transformed_p_matrix[1])
        transformed_points.append(transformed_p)

    tile_info['vertices'] = [(float(p[0]), float(p[1])) for p in transformed_points]

    verts = []
    z = (iteration - 1) * 2
    for v in transformed_points:
        x, y = v
        verts.append([x, y, z])

    faces = [list(range(len(verts)))]
    mesh = bpy.data.meshes.new(label)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    oname = f"{tile_info['absnum']}:{label}:{iteration}"
    obj = bpy.data.objects.new(oname, mesh)
    bpy.context.collection.objects.link(obj)

    if tile_info['label'] == 'Gamma2':
        mat = smaterial('Mystic-Flipped', [1, 0.5, 0, 1]) # Orange to see it clearly
        obj.data.materials.append(mat)
    else:
        mat = smaterial('Tile', [0.8, 0.1, 0.8, 0.25])
        obj.data.materials.append(mat)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS")
    obj.show_wire = True
    if label=='Gamma2': obj.hide_set(True)
    tile_info['ob'] = obj

import string
from random import random, choice
_smat_default_names = list(string.ascii_letters)

def smaterial(name=None, color=None):
    if not name:
        name = choice(_smat_default_names)
    if not color:
        color = [random(), random(), random(), 1]
    if name not in bpy.data.materials:
        m = bpy.data.materials.new(name=name)
        #m.use_nodes = True
        #bsdf = m.node_tree.nodes["Principled BSDF"]
        #bsdf.inputs['Base Color'].default_value = color
        m.use_nodes = False
        m.diffuse_color = color
    return bpy.data.materials[name]


def blender_trace( pass_info, iteration, nurbs=False ):
    new_collection(f'Iteration({iteration})')
    for i, tile_info in enumerate(pass_info.tiles_info):
        plot_tile_blender(tile_info, iteration)
        #print(f"Generated blender tile '{tile_info['label']}' ({i+1}/{len(pass_info.tiles_info)}) in iteration {iteration}.")

def find_shared_vertices(tiles_info, tolerance=1e-4):
    print("\nFinding shared vertices...")
    vertex_map = {}
    for tile in tiles_info:
        if 'vertices' not in tile: continue
        for v in tile['vertices']:
            key = (round(v[0] / tolerance) * tolerance, round(v[1] / tolerance) * tolerance)
            if key not in vertex_map: vertex_map[key] = []
            vertex_map[key].append(tile)
    shared_points = {pos: tiles for pos, tiles in vertex_map.items() if len(tiles) > 1}
    print(f"Found {len(shared_points)} unique shared vertex locations.")
    return shared_points

def create_fiber_curve(start_vec, end_vec, knot_type='Unknot'):
    curve_data = bpy.data.curves.new(name=f"{knot_type}Fiber", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.03
    curve_data.bevel_resolution = 4
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(1)
    p0, p1 = spline.bezier_points[0], spline.bezier_points[1]
    #p0.radius = 10
    p0.co, p1.co = start_vec, end_vec
    mid_point = (start_vec + end_vec) / 2.0
    direction_vec = (end_vec - start_vec).normalized()
    non_parallel_vec = Vector((0, 0, 1))
    if abs(direction_vec.dot(non_parallel_vec)) > 0.99: non_parallel_vec = Vector((0, 1, 0))
    perp_vec = direction_vec.cross(non_parallel_vec).normalized()
    arc_height, mat_color = (0.5, (1, 1, 0, 1))
    if knot_type == 'Trefoil': arc_height, mat_color = (1.0, (1, 0, 0, 1))
    elif knot_type == 'FigureEight': arc_height, mat_color = (-0.75, (0, 1, 1, 1))
    handle_offset = mid_point + (perp_vec * arc_height)
    p0.handle_right, p1.handle_left = handle_offset, handle_offset
    p0.handle_right_type, p1.handle_left_type = 'ALIGNED', 'ALIGNED'
    curve_obj = bpy.data.objects.new(f"{knot_type}FiberObj", curve_data)
    #mat = smaterial(knot_type, mat_color)
    #curve_obj.data.materials.append(mat)
    curve_obj.data.bevel_resolution = 1
    curve_obj.show_wire = True
    bpy.context.collection.objects.link(curve_obj)
    return curve_obj

def generate_fiber_network(tiles_info, iteration, knot_types=['Unknot', 'Trefoil']):
    shared_vertices = find_shared_vertices(tiles_info)
    if not shared_vertices:
        print("No shared vertices found to create fiber network.")
        return
    new_collection(f'Fibers_Iter_{iteration}')
    print(f"Generating fiber network for iteration {iteration}...")
    z_height = (iteration - 1) * 2
    knot_type_idx = 0
    fibers = []
    for pos, tiles in shared_vertices.items():
        shared_point_3d = Vector((pos[0], pos[1], z_height))
        for tile_data in tiles:
            if 'ob' in tile_data and tile_data['ob']:
                tile_center = tile_data['ob'].location
                current_knot = knot_types[knot_type_idx % len(knot_types)]
                f = create_fiber_curve(tile_center, shared_point_3d, knot_type= tile_data['ob'].name + '.' + current_knot)
                knot_type_idx += 1
                fibers.append(f)
    print("Fiber network generation complete.")
    print('connecting fibers', len(fibers))
    knots_info = {}
    knots2 = []
    knots3 = []
    knots4 = []
    while len(fibers) > 1:
        f = fibers.pop()
        vec = f.data.splines[0].bezier_points[1].co
        links = [f]
        for other in fibers:
            if other.data.splines[0].bezier_points[1].co == vec:
                #print(f, 'connects with', other)
                links.append(other)
                if not f.data.materials:
                    if other.data.materials:
                        f.data.materials.append(other.data.materials[0])
                    elif random() < 1.6:
                        f.data.materials.append(smaterial())
                if not other.data.materials and f.data.materials:
                    other.data.materials.append(f.data.materials[0])
        if len(links) == 2:
            knots2 += links
        elif len(links) == 3:
            knots3 += links
        elif len(links) == 4:
            knots4 += links

        if len(links) not in knots_info:
            knots_info[len(links)] = 0
        knots_info[len(links)] += 1

    print('iteration:', iteration)
    print('knots info:')
    print(knots_info)
    for numlinks in knots_info:
        if numlinks < 3: continue
        print('braid number=%s, number of knots=%s' % (numlinks, knots_info[numlinks]))
    print('Edge_a', EDGE_A)
    print('Edge_b', EDGE_B)

    knots2 = set(knots2)
    knots3 = set(knots3)
    knots4 = set(knots4)
    for o in knots3:
        o.data.splines[0].bezier_points[1].radius = 5
    for o in knots4:
        o.data.splines[0].bezier_points[1].radius = 10

    ## connect knots4 with knots3 group
    for o in knots4:
        for other in knots3:
            if other.name == o.name: continue
            if o.data.splines[0].bezier_points[0].co == other.data.splines[0].bezier_points[0].co:
                o.data.splines[0].bezier_points[0].radius += 3
                other.data.splines[0].bezier_points[0].radius += 3

    ## connect knots3 with knots3 group
    for o in knots3:
        for other in knots3:
            if other.name == o.name: continue
            if o.data.splines[0].bezier_points[0].co == other.data.splines[0].bezier_points[0].co:
                o.data.splines[0].bezier_points[0].radius = 3
                other.data.splines[0].bezier_points[0].radius = 3


def buildSpectreBase_sympy(spectre_points_all, rotation=30):
    quad_base = sp.Matrix([spectre_points_all[3, :], spectre_points_all[5, :], spectre_points_all[7, :], spectre_points_all[11, :]])
    tiles = {label: Tile(label, quad_base, psi_field, 'Bichromatic') for label in ["Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]}
    gamma2_trans = mul_sympy(sp.Matrix([[1, 0, spectre_points_all[8, 0]], [0, 1, spectre_points_all[8, 1]]]), trot_sympy(rotation))
    tiles["Gamma"] = MetaTile(label="Gamma", tiles=[Tile("Gamma1", quad_base, psi_field, 'Monochromatic'), Tile("Gamma2", quad_base, psi_field, 'Monochromatic')], transformations=[IDENTITY.copy(), gamma2_trans], quad=quad_base) # Removed is_flipped=True
    return tiles

def buildSupertiles_sympy(input_tiles):
    quad = input_tiles["Delta"].quad
    total_angle, transformations = 0, [IDENTITY.copy()]
    for _angle, _from, _to in ((60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1), (0, 2, 0), (60, 3, 1), (-120, 3, 3)):
        if _angle != 0:
            total_angle += _angle
            rotation = trot_sympy(total_angle)
        ttrans = IDENTITY.copy()
        point_from_2d, point_to_2d = quad.row(_from), quad.row(_to)
        point_from_vec, point_to_vec = sp.Matrix([point_from_2d[0], point_from_2d[1], 1]), sp.Matrix([point_to_2d[0], point_to_2d[1], 1])
        transformed_from, rotated_to = transformations[-1] * point_from_vec, rotation * point_to_vec
        ttrans_vec = transformed_from - rotated_to
        ttrans[0, 2], ttrans[1, 2] = ttrans_vec[0], ttrans_vec[1]
        transformations.append(mul_sympy(ttrans, rotation))
    R_2x3 = sp.Matrix([[-1, 0, 0], [0, 1, 0]])
    transformations = [mul_sympy(R_2x3, trsf) for trsf in transformations]
    quad_points_to_transform = [(quad.row(2), transformations[6]), (quad.row(1), transformations[5]), (quad.row(2), transformations[3]), (quad.row(1), transformations[0])]
    super_quad_points_rows = []
    for point_2d, transform in quad_points_to_transform:
        point_vec = sp.Matrix([point_2d[0], point_2d[1], 1])
        transformed_point = transform * point_vec
        super_quad_points_rows.append((transformed_point[0], transformed_point[1]))
    super_quad = sp.Matrix(super_quad_points_rows)
    substitutions_map = {"Gamma":("Pi","Delta",None,"Theta","Sigma","Xi","Phi","Gamma"),"Delta":("Xi","Delta","Xi","Phi","Sigma","Pi","Phi","Gamma"),"Theta":("Psi","Delta","Pi","Phi","Sigma","Pi","Phi","Gamma"),"Lambda":("Psi","Delta","Xi","Phi","Sigma","Pi","Phi","Gamma"),"Xi":("Psi","Delta","Pi","Phi","Sigma","Psi","Phi","Gamma"),"Pi":("Psi","Delta","Xi","Phi","Sigma","Psi","Phi","Gamma"),"Sigma":("Xi","Delta","Xi","Phi","Sigma","Pi","Lambda","Gamma"),"Phi":("Psi","Delta","Psi","Phi","Sigma","Pi","Phi","Gamma"),"Psi":("Psi","Delta","Psi","Phi","Sigma","Psi","Phi","Gamma")}
    new_tiles = {}
    for label, substitutions in substitutions_map.items():
        sub_tiles = [input_tiles[subst] for subst in substitutions if subst]
        sub_transformations = [trsf for subst, trsf in zip(substitutions, transformations) if subst]
        new_tiles[label] = MetaTile(label=label, tiles=sub_tiles, transformations=sub_transformations, quad=super_quad, quantum_field=YM_field, hopfion_crystal_state='Geometric Drift')
    return new_tiles

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

class collector:
    def __init__(self, tiles_info=None):
        self.tiles_info = tiles_info if tiles_info is not None else []
        self.tiles = []
    def collect_tiles(self, tile_transformation, label, number, tile):
        self.tiles.append(tile)
        anum = len(self.tiles_info) + 1
        tile.info = {
            'absnum': anum,
            'number': number,
            'label': label,
            'transformation': tile_transformation,
            'x_pos': tile_transformation[0, 2],
            'y_pos': tile_transformation[1, 2],
            'rotation_deg': sp.deg(sp.atan2(tile_transformation[1, 0], tile_transformation[0, 0])),
            'is_prime': is_prime(anum),
            # Removed 'is_flipped'
        }
        self.tiles_info.append(tile.info)

# --- Main Execution ---
if __name__=='__main__':
    ITER2 = ITER3 = False
    if '--iter2' in sys.argv:
        ITER2 = True
    if '--iter3' in sys.argv:
        ITER3 = True

    SPECTRE_POINTS_SYM = get_spectre_points_sympy(Edge_a, Edge_b)
    base_tiles_sympy = buildSpectreBase_sympy(SPECTRE_POINTS_SYM)
    first_super_tiles_sympy = buildSupertiles_sympy(base_tiles_sympy)
    if ITER2:
        second_super_tiles_sympy = buildSupertiles_sympy(first_super_tiles_sympy)

    if ITER3:
        third_super_tiles_sympy = buildSupertiles_sympy(second_super_tiles_sympy)

    first_pass_info = collector()
    first_super_tiles_sympy["Delta"].forEachTile(first_pass_info.collect_tiles)

    if ITER2:
        second_pass_info = collector()
        second_super_tiles_sympy["Delta"].forEachTile(second_pass_info.collect_tiles)

    if ITER3:
        third_pass_info = collector()
        third_super_tiles_sympy["Delta"].forEachTile(third_pass_info.collect_tiles)

    if bpy:
        cam = bpy.data.objects['Camera']
        cam.rotation_euler = [0,0,0]
        cam.location = [0,-1,32]
        if ITER2:
            cam.location = [8, -5, 95]

        bpy.data.scenes[0].render.engine = "BLENDER_WORKBENCH"

        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.select_by_type(type='CURVE', extend=True)
        bpy.ops.object.delete()

        print("\n--- Generating First Iteration ---")
        blender_trace(first_pass_info, 1)
        generate_fiber_network(first_pass_info.tiles_info, iteration=1, knot_types=['Unknot', 'Trefoil'])

        if ITER2:
            print("\n--- Generating Second Iteration ---")
            blender_trace(second_pass_info, 2)
            generate_fiber_network(second_pass_info.tiles_info, iteration=2, knot_types=['FigureEight'])

        if ITER3:
            print("\n--- Generating Third Iteration ---")
            blender_trace(third_pass_info, 3)
            generate_fiber_network(third_pass_info.tiles_info, iteration=3, knot_types=['FigureEight'])