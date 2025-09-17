import collections
from collections import Counter

# =============================================================================
#
# This script calculates and validates the tile counts for each generation
# of a Spectre-tile-based aperiodic tiling. It employs two distinct methods:
#
# 1. Direct Substitution: A straightforward iterative application of tile
#    replacement rules, directly simulating the tiling's growth.
#    This method serves as a validation baseline.
#
# 2. Matrix-based Linear Recurrence Method:
# 2.1  Calculation by Second-Order Linear Recurrence Matrix;
#      A mathematically rigorous approach using a transition matrix
#      to solve a system of second-order linear recurrence relations.
#      This method leverages linear algebra to compute the sequences efficiently.
#
# 2.2  Calculation by First-Order Linear Recurrence Matrix;
#
# The script is structured to demonstrate that both methods yield identical results,
# confirming the correctness of the derived recurrence relations.
#
# =============================================================================

# --- Configuration ---
N_ITERATIONS = 14
# e.g., None to trace off; 'Psi' to trace its calculation;
DEBUG_TRACE_LABEL = 'Psi' 

# =============================================================================
# Section 1: Direct Substitution Method for Validation
# =============================================================================

print('--- Validation of Tile Counts by Direct Substitution ---')

# The set of unique tile labels used in the substitution system.
# Note: The original 'Gamma' tile is split into 'Gamma1' and 'Gamma2'
# for specific tracking purposes.
TILE_NAMES = ['Gamma1', 'Gamma2', 'Delta', 'Sigma', 'Theta', 'Lambda', 'Pi', 'Xi', 'Phi', 'Psi']

# The substitution rules are derived from the geometric construction of the
# Spectre supertile, as described by discoverers :
# David Smith, Joseph Samuel Myers, Craig S. Kaplan, and Chaim Goodman-Strauss, 2023
# (https://cs.uwaterloo.ca/~csk/spectre/
#  https://arxiv.org/abs/2306.10767
# https://cs.uwaterloo.ca/~csk/spectre/app.html).

# Each parent tile is replaced by a specific arrangement of child tiles.
# None indicates an empty position within the supertile structure.
original_substitution_rules = [
  ['Gamma',  ['Pi', 'Delta', None, 'Theta', 'Sigma', 'Xi', 'Phi', 'Gamma']],
  ['Delta',  ['Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma']],
  ['Theta',  ['Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma']],
  ['Lambda', ['Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma']],
  ['Xi',     ['Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma']],
  ['Pi',     ['Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma']],
  ['Sigma',  ['Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Lambda', 'Gamma']],
  ['Phi',    ['Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma']],
  ['Psi',    ['Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma']]
]

# The geometric rules are converted into a frequency map (a dict).
# This represents a system of first-order difference equations, where the
# count of each tile at step `n` is a linear combination of all tile counts
# at step `n-1`.
substitution_rules = {}
for parent, children in original_substitution_rules:
    # Filter out None values and count frequencies of children
    filtered_children = [child for child in children if child is not None]
    substitution_rules[parent] = Counter(filtered_children)

# print("\n# Substitution Rules (as frequency maps):",substitution_rules)

# Helper function to format dictionary output to match Ruby's hash style
def format_dict_ruby_style(d):
    items = [f'"{k}"=>{v}' for k, v in d.items()]
    return "{" + ", ".join(items) + "}"

# --- Initial Conditions ---

# Storage for the sequence of counts for each tile type.
tile_sequences = collections.defaultdict(list)

# At iteration n=0, the tiling starts from a single 'Delta' tile.
current_counts = collections.defaultdict(int)
current_counts['Delta'] = 1
for name in TILE_NAMES:
    tile_sequences[name].append(current_counts[name])

if DEBUG_TRACE_LABEL:
    print(f"# Iteration 0 = {format_dict_ruby_style(dict(current_counts))}")

# At iteration n=1, the initial 'Delta' tile is substituted.
# This step requires special handling for the bifurcation of 'Gamma' into 'Gamma1' and 'Gamma2'.
prev_counts = current_counts.copy()
current_counts = collections.defaultdict(int)
current_counts.update({
    'Gamma1': 1, 'Gamma2': 1, 'Delta': 1, 'Sigma': 1, 'Theta': 0, 
    'Lambda': 0, 'Pi': 1, 'Xi': 2, 'Phi': 2, 'Psi': 0
})
for name in TILE_NAMES:
    tile_sequences[name].append(current_counts[name])

if DEBUG_TRACE_LABEL:
    print(f"# Iteration 1 = {format_dict_ruby_style(current_counts)}")


# --- Iterative Calculation (n >= 2) ---

prev_counts = current_counts.copy()
for n in range(2, N_ITERATIONS):
    current_counts = collections.defaultdict(int)
    # The order of parent tiles must be the same as in the original Ruby script
    # to ensure identical floating-point summation order, though it's not critical here.
    for label, rules in substitution_rules.items():
        # The counts of Gamma1 and Gamma2 from the previous step are treated
        # as a single 'Gamma' pool for substitution purposes.
        count = prev_counts['Gamma1'] if label == 'Gamma' else prev_counts[label]
        
        # if count == 0:
        #     continue

        for sub_label, sub_count in rules.items():
            if sub_label == 'Gamma':
                # When a 'Gamma' tile is produced, it contributes to both Gamma1 and Gamma2 counts.
                current_counts['Gamma1'] += count * sub_count
                current_counts['Gamma2'] += count * sub_count
                if sub_label == DEBUG_TRACE_LABEL:
                    print(f" Debug: {label} -> {sub_label}, prev_counts[{label}]: {count}, sub_count: {sub_count}, adds:{count * sub_count}, current_counts[{sub_label}]: {current_counts['Gamma2']} ")
            else:
                # Increment the count for the specific child tile (e.g., 'Pi', 'Delta').
                current_counts[sub_label] += count * sub_count
                if sub_label == DEBUG_TRACE_LABEL:
                    print(f" Debug: {label} -> {sub_label}, prev_counts[{label}]: {count}, sub_count: {sub_count}, adds:{count * sub_count}, current_counts[{sub_label}]: {current_counts[sub_label]} ")

    for name in TILE_NAMES:
        tile_sequences[name].append(current_counts[name])

    if DEBUG_TRACE_LABEL:
        filtered_counts = {k: v for k, v in current_counts.items() if v > 0}
        print(f"# Iteration {n} = {format_dict_ruby_style(filtered_counts)}")
    
    prev_counts = current_counts.copy()

# --- Display Validation Results ---
print("\n--- Substitution Results by Tile ---")
for name in TILE_NAMES:
    # Convert all numbers to strings before joining
    print(f"# {name} = [{', '.join(map(str, tile_sequences[name]))}]")

# =============================================================================
# 2. 1次置換行列を生成するコード
# =============================================================================
import numpy as np

# LABELS = TILE_NAMES
# ラベル名から行列のインデックスを引くための辞書を作成
label_to_index = {label: i for i, label in enumerate(TILE_NAMES)}
num_labels = len(TILE_NAMES)

# num_labels x num_labels のゼロ行列を初期化
# この行列 M は v(n) = M * v(n-1) の関係を満たす
substitution_matrix = np.zeros((num_labels, num_labels), dtype=int)

# 行列を置換規則に基づいて埋める
# 列(j) = 親タイル, 行(i) = 子タイル
for j, parent_label in enumerate(TILE_NAMES):
    
    # 適用する置換規則を選択
    rules_to_apply = None
    if (parent_label == 'Gamma1') or (parent_label == 'Gamma2'):
        # 親が'Gamma1'のときは、'Gamma'の規則を使用する
        rules_to_apply = substitution_rules.get('Gamma')
    elif parent_label in substitution_rules:
        # 親が'Gamma'以外の基本タイルの場合
        rules_to_apply = substitution_rules.get(parent_label)
    # 注: 'Gamma2' や補助シーケンスは親として子を生成しないため、
    # rules_to_apply は None のままとなり、その列はゼロのままとなる。
    
    if not rules_to_apply:
        continue

    # 選択した規則に基づき、行列の要素を決定
    for child_label, count in rules_to_apply.items():
        if child_label == 'Gamma':
            # 子が 'Gamma' の場合、Gamma1 と Gamma2 の両方の行にカウントを配置
            g1_idx = label_to_index['Gamma1']
            g2_idx = label_to_index['Gamma2']
            if g2_idx != j:
                substitution_matrix[g1_idx, j] += count
            if g1_idx != j:
                substitution_matrix[g2_idx, j] += count
        elif (parent_label != 'Gamma2') and (child_label in label_to_index):
            # それ以外の子タイルの場合
            child_idx = label_to_index[child_label]
            substitution_matrix[child_idx, j] += count

print(f"--- Generated 1st-Order Substitution Matrix ({num_labels}x{num_labels}) ---")
print("      " + " ".join(f"{s:<2}" for s in [l[0:2] for l in TILE_NAMES]))
print("      " + "-" * (3 * num_labels))
for i, row in enumerate(substitution_matrix):
    row_str = " ".join(f"{x:2d}" for x in row)
    print(f"{TILE_NAMES[i]:<7}| {row_str}")

# =============================================================================
# Section 2: Matrix-Based Linear Recurrence Method
# =============================================================================

# This section implements a matrix-based approach to compute the growth of Spectre tiles
# through linear recurrence relations.

# The system tracks the counts of each tile type, including two auxiliary sequences
# (_Pi_Xi and _even) required to handle non-homogeneous recurrence terms.
# These counts are represented as vectors evolving over discrete time steps.

# At each step n, the state vector is defined as:
#     S(n) = [v(n−1), v(n−2)]ᵗ
# where v(n) is the vector of tile counts at step n.

# The evolution of the system is governed by a transition matrix M, such that:
#     S(n+1) = M · S(n)

# This formulation allows us to solve a coupled system of second-order linear recurrence
# relations using matrix exponentiation. The matrix M is structured as:
#     M = [ A  B ]
#         [ I  0 ]
# where A and B encode the recurrence coefficients, and I is the identity matrix.
# This structure ensures that:
#     v(n) = A · v(n−1) + B · v(n−2)

# ラベル定義
labels = TILE_NAMES + [ "_Pi_Xi", "_even"]
VEC_SIZE = len(labels)
N_ITERATIONS = 15

# =============================================================================
# Matrix Definitions
# =============================================================================
# Two recurrence matrices are defined:
# 1. Second-Order Recurrence Matrix:
#    Encodes the full recurrence relations for each tile type, including auxiliary sequences.
#    This is a 12×24 matrix used to compute v(n) from v(n−1) and v(n−2).
# 2. First-Order Substitution Matrix:
#    Represents a simplified substitution rule acting directly on v(n−1).
#    It is embedded into a 12×24 matrix by placing the 10×10 substitution block in the top-left
#    and copying the auxiliary recurrence rows from the second-order matrix.

matrices = [
    ("# 2.1 : Calculation by Second-Order Linear Recurrence Matrix", np.array([
    # Recurrence: a(n) = 8*a(n-1) - a(n-2). See OEIS A001090.
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    # Non-homogeneous recurrences are handled using auxiliary sequences.
    # Pi(n) = 8*_Pi_Xi(n-1) - _Pi_Xi(n-2) + _even(n-1)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    # Xi(n) = 8*_Pi_Xi(n-1) - _Pi_Xi(n-2) + _even(n-2)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
    # Phi(n) = 8*Phi(n-1) - Phi(n-2)
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
    # Psi(n) = (Psi(`1`) - 1)*Psi(n-1) - Psi(n-2) + `6`.  See OEIS A057080.
    # The constant term `6` is modeled as 6 * _even(n-1) + 6 * _even(n-2), which equals 6 for n >= 2.
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 6],
    # Auxiliary sequence _Pi_Xi follows the base recurrence. See OEIS A341927.
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    # Auxiliary sequence _even generates [0, 1, 0, 1, ...], satisfying a(n) = a(n-2).
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
],  dtype=object)),
    ("# 2.2 : Calculation by First-Order Linear Recurrence Matrix", np.zeros((12, 24), dtype=object))
]

matrices[1][1][10:12, :] = matrices[0][1][10:12, :]
matrices[1][1][:10, :10] = substitution_matrix

for title, recurrence_matrix in matrices :
    print(f"\n{title}")
    
    # =============================================================================
    # Recurrence Calculation Loop
    # =============================================================================
    
    # Initial state vectors v(0), v(1), and v(2) serve as base cases.
    # These are derived from the substitution process for the first few steps.
    v = [
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 2, 2, 0, 1, 1],
        [8, 8, 8, 8, 1, 1, 7, 6, 14, 10, 6, 0] # for v(0)==v(1)==0
    ]

    # At each iteration:
    # - The state vector is formed by concatenating v(n−1) and v(n−2)
    # - The next vector v(n) is computed via matrix multiplication:
    #       v(n) = M · [v(n−1), v(n−2)]ᵗ
    for i in range(3, N_ITERATIONS + 1):
        state_vector = np.array(v[-1] + v[-2],  dtype=object)
        next_vector = recurrence_matrix @ state_vector
        v.append(next_vector.tolist())

    # Results are printed both by iteration and by tile label.
    # Auxiliary sequences (_Pi_Xi and _even) are excluded from the summary totals.
    print("\n--- Recurrence Results by Iteration ---")
    for i, vec in enumerate(v):
        filtered = {label: val for label, val in zip(labels, vec) if not label.startswith('_') and val != 0}
        filtered["_total"] = sum(filtered.values())
        print(f"# Iteration {i} = {filtered}")

    # This dual-matrix approach demonstrates that the same tile growth process
    # can be modeled using either a second-order recurrence or a first-order substitution matrix
    # embedded in an extended state space.
    print("\n--- Recurrence Results by Tile ---")
    for label, values in zip(labels, zip(*v)):
        print(f"# {label} = [{', '.join(map(str, values))}]")

# excerpt of the output:
#--- Recurrence Results by Iteration ---
# Iteration 0 = {'Delta': 1, '_total': 1}
# Iteration 1 = {'Gamma1': 1, 'Gamma2': 1, 'Delta': 1, 'Sigma': 1, 'Pi': 1, 'Xi': 2, 'Phi': 2, '_total': 9}
# Iteration 2 = {'Gamma1': 8, 'Gamma2': 8, 'Delta': 8, 'Sigma': 8, 'Theta': 1, 'Lambda': 1, 'Pi': 7, 'Xi': 6, 'Phi': 14, 'Psi': 10, '_total': 71}
# Iteration 3 = {'Gamma1': 63, 'Gamma2': 63, 'Delta': 63, 'Sigma': 63, 'Theta': 8, 'Lambda': 8, 'Pi': 47, 'Xi': 48, 'Phi': 110, 'Psi': 86, '_total': 559}
# Iteration 4 = {'Gamma1': 496, 'Gamma2': 496, 'Delta': 496, 'Sigma': 496, 'Theta': 63, 'Lambda': 63, 'Pi': 371, 'Xi': 370, 'Phi': 866, 'Psi': 684, '_total': 4401}
# Iteration 5 = {'Gamma1': 3905, 'Gamma2': 3905, 'Delta': 3905, 'Sigma': 3905, 'Theta': 496, 'Lambda': 496, 'Pi': 2913, 'Xi': 2914, 'Phi': 6818, 'Psi': 5392, '_total': 34649}
# Iteration 6 = {'Gamma1': 30744, 'Gamma2': 30744, 'Delta': 30744, 'Sigma': 30744, 'Theta': 3905, 'Lambda': 3905, 'Pi': 22935, 'Xi': 22934, 'Phi': 53678, 'Psi': 42458, '_total': 272791}
# Iteration 7 = {'Gamma1': 242047, 'Gamma2': 242047, 'Delta': 242047, 'Sigma': 242047, 'Theta': 30744, 'Lambda': 30744, 'Pi': 180559, 'Xi': 180560, 'Phi': 422606, 'Psi': 334278, '_total': 2147679}
# Iteration 8 = {'Gamma1': 1905632, 'Gamma2': 1905632, 'Delta': 1905632, 'Sigma': 1905632, 'Theta': 242047, 'Lambda': 242047, 'Pi': 1421539, 'Xi': 1421538, 'Phi': 3327170, 'Psi': 2631772, '_total': 16908641}
# Iteration 9 = {'Gamma1': 15003009, 'Gamma2': 15003009, 'Delta': 15003009, 'Sigma': 15003009, 'Theta': 1905632, 'Lambda': 1905632, 'Pi': 11191745, 'Xi': 11191746, 'Phi': 26194754, 'Psi': 20719904, '_total': 133121449}
# Iteration 10 = {'Gamma1': 118118440, 'Gamma2': 118118440, 'Delta': 118118440, 'Sigma': 118118440, 'Theta': 15003009, 'Lambda': 15003009, 'Pi': 88112423, 'Xi': 88112422, 'Phi': 206230862, 'Psi': 163127466, '_total': 1048062951}
# Iteration 11 = {'Gamma1': 929944511, 'Gamma2': 929944511, 'Delta': 929944511, 'Sigma': 929944511, 'Theta': 118118440, 'Lambda': 118118440, 'Pi': 693707631, 'Xi': 693707632, 'Phi': 1623652142, 'Psi': 1284299830, '_total': 8251382159}
# Iteration 12 = {'Gamma1': 7321437648, 'Gamma2': 7321437648, 'Delta': 7321437648, 'Sigma': 7321437648, 'Theta': 929944511, 'Lambda': 929944511, 'Pi': 5461548627, 'Xi': 5461548626, 'Phi': 12782986274, 'Psi': 10111271180, '_total': 64962994321}
# Iteration 13 = {'Gamma1': 57641556673, 'Gamma2': 57641556673, 'Delta': 57641556673, 'Sigma': 57641556673, 'Theta': 7321437648, 'Lambda': 7321437648, 'Pi': 42998681377, 'Xi': 42998681378, 'Phi': 100640238050, 'Psi': 79605869616, '_total': 511452572409}
# Iteration 14 = {'Gamma1': 453811015736, 'Gamma2': 453811015736, 'Delta': 453811015736, 'Sigma': 453811015736, 'Theta': 57641556673, 'Lambda': 57641556673, 'Pi': 338527902391, 'Xi': 338527902390, 'Phi': 792338918126, 'Psi': 626735685754, '_total': 4026657584951}
# Iteration 15 = {'Gamma1': 3572846569215, 'Gamma2': 3572846569215, 'Delta': 3572846569215, 'Sigma': 3572846569215, 'Theta': 453811015736, 'Lambda': 453811015736, 'Pi': 2665224537743, 'Xi': 2665224537744, 'Phi': 6238071106958, 'Psi': 4934279616422, '_total': 31701808107199}

#--- Recurrence Results by Tile ---
# Gamma1 = [0, 1, 8, 63, 496, 3905, 30744, 242047, 1905632, 15003009, 118118440, 929944511, 7321437648, 57641556673, 453811015736, 3572846569215]
# Gamma2 = [0, 1, 8, 63, 496, 3905, 30744, 242047, 1905632, 15003009, 118118440, 929944511, 7321437648, 57641556673, 453811015736, 3572846569215]
# Delta = [1, 1, 8, 63, 496, 3905, 30744, 242047, 1905632, 15003009, 118118440, 929944511, 7321437648, 57641556673, 453811015736, 3572846569215]
# Sigma = [0, 1, 8, 63, 496, 3905, 30744, 242047, 1905632, 15003009, 118118440, 929944511, 7321437648, 57641556673, 453811015736, 3572846569215]
# Theta = [0, 0, 1, 8, 63, 496, 3905, 30744, 242047, 1905632, 15003009, 118118440, 929944511, 7321437648, 57641556673, 453811015736]
# Lambda = [0, 0, 1, 8, 63, 496, 3905, 30744, 242047, 1905632, 15003009, 118118440, 929944511, 7321437648, 57641556673, 453811015736]
# Pi = [0, 1, 7, 47, 371, 2913, 22935, 180559, 1421539, 11191745, 88112423, 693707631, 5461548627, 42998681377, 338527902391, 2665224537743]
# Xi = [0, 2, 6, 48, 370, 2914, 22934, 180560, 1421538, 11191746, 88112422, 693707632, 5461548626, 42998681378, 338527902390, 2665224537744]
# Phi = [0, 2, 14, 110, 866, 6818, 53678, 422606, 3327170, 26194754, 206230862, 1623652142, 12782986274, 100640238050, 792338918126, 6238071106958]
# Psi = [0, 0, 10, 86, 684, 5392, 42458, 334278, 2631772, 20719904, 163127466, 1284299830, 10111271180, 79605869616, 626735685754, 4934279616422]
