"""Convert MathQA annotated formulas to SELPH s-expressions.

MathQA format: multiply(divide(48, const_1000), 9)
SELPH format:  (multiply (divide 48 1000) 9)

Also resolves constants (const_100 → 100, const_pi → 3.14159265...) and
validates by evaluating the s-expression to check it produces a reasonable answer.
"""
import json
import re
import math
from pathlib import Path
from collections import Counter

DATA = Path(__file__).parent / "data"

# Constant resolution
CONSTANTS = {
    "const_pi": str(math.pi),
    "const_2": "2",
    "const_1": "1",
    "const_3": "3",
    "const_4": "4",
    "const_5": "5",
    "const_6": "6",
    "const_10": "10",
    "const_12": "12",
    "const_26": "26",
    "const_52": "52",
    "const_100": "100",
    "const_1000": "1000",
    "const_60": "60",
    "const_180": "180",
    "const_360": "360",
    "const_3600": "3600",
    "const_0": "0",
    "const_0_25": "0.25",
    "const_0_33": "0.3333333333",
    "const_0_2778": "0.2778",
    "const_3_6": "3.6",
    "const_1_6": "1.6",
    "const_0_3937": "0.3937",
    "const_deg_to_rad": str(math.pi / 180),
}

# Operators that map directly to SELPH (or need renaming)
OP_MAP = {
    # Arithmetic (direct)
    "add": "add",
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide",
    "power": "power",
    "sqrt": "sqrt",
    "log": "log",
    "negate": "negate",
    "inverse": "inverse",
    "floor": "floor",
    "factorial": "factorial",
    "max": "max",
    "min": "min",
    "reminder": "remainder",  # typo in MathQA
    "gcd": "gcd",
    "lcm": "lcm",
    # Trig
    "sine": "sin",
    "cosine": "cos",
    "tangent": "tan",
    # Combinatorics
    "choose": "choose",
    "permutation": "permutation",
    # Geometry
    "circle_area": "circle-area",
    "circumface": "circumference",
    "rectangle_area": "rectangle-area",
    "rectangle_perimeter": "rectangle-perimeter",
    "square_area": "square-area",
    "square_perimeter": "square-perimeter",
    "triangle_area": "triangle-area",
    "triangle_area_three_edges": "triangle-area-three-edges",
    "triangle_perimeter": "triangle-perimeter",
    "rhombus_area": "rhombus-area",
    "rhombus_perimeter": "rhombus-perimeter",
    "quadrilateral_area": "quadrilateral-area",
    "volume_cube": "volume-cube",
    "volume_cylinder": "volume-cylinder",
    "volume_rectangular_prism": "volume-rectangular-prism",
    "volume_sphere": "volume-sphere",
    "volume_cone": "volume-cone",
    "surface_cube": "surface-cube",
    "surface_cylinder": "surface-cylinder",
    "surface_rectangular_prism": "surface-rectangular-prism",
    "surface_sphere": "surface-sphere",
    "cube_edge_by_volume": "cube-edge-by-volume",
    "square_edge_by_perimeter": "square-edge-by-perimeter",
    "square_edge_by_area": "square-edge-by-area",
    "diagonal": "diagonal",
    # Other
    "speed": "speed",
    "stream_speed": "stream-speed",
    "speed_in_still_water": "speed-in-still-water",
    "negate_prob": "negate-prob",
    "original_price_before_loss": "original-price-before-loss",
    "original_price_before_gain": "original-price-before-gain",
    "p_after_gain": "price-after-gain",
}


def tokenize_formula(formula):
    """Tokenize a MathQA annotated formula into tokens."""
    tokens = []
    i = 0
    while i < len(formula):
        c = formula[i]
        if c in '(),':
            tokens.append(c)
            i += 1
        elif c == ' ':
            i += 1
        elif c == '-' and i + 1 < len(formula) and formula[i+1].isdigit():
            # Negative number
            j = i + 1
            while j < len(formula) and (formula[j].isdigit() or formula[j] == '.'):
                j += 1
            tokens.append(formula[i:j])
            i = j
        else:
            j = i
            while j < len(formula) and formula[j] not in '(), ':
                j += 1
            tokens.append(formula[i:j])
            i = j
    return tokens


def parse_formula(tokens):
    """Parse tokenized MathQA formula into a tree: (op, [args])."""
    pos = [0]

    def parse_expr():
        tok = tokens[pos[0]]
        if tok == '(':
            raise ValueError(f"Unexpected '(' at position {pos[0]}")

        # Check if it's a function call: name(
        if pos[0] + 1 < len(tokens) and tokens[pos[0] + 1] == '(':
            op = tok
            pos[0] += 2  # skip name and (
            args = []
            while tokens[pos[0]] != ')':
                if tokens[pos[0]] == ',':
                    pos[0] += 1
                    continue
                args.append(parse_expr())
            pos[0] += 1  # skip )
            return (op, args)
        else:
            # It's an atom (number or constant)
            val = tok
            pos[0] += 1
            return val

    result = parse_expr()
    return result


def tree_to_sexpr(tree):
    """Convert parsed tree to SELPH s-expression string."""
    if isinstance(tree, str):
        # Atom: resolve constants
        if tree in CONSTANTS:
            return CONSTANTS[tree]
        return tree

    op, args = tree
    selph_op = OP_MAP.get(op)
    if selph_op is None:
        selph_op = op.replace("_", "-")  # fallback: underscore to hyphen

    arg_strs = [tree_to_sexpr(a) for a in args]
    return f"({selph_op} {' '.join(arg_strs)})"


def eval_sexpr(tree):
    """Evaluate parsed tree to a numeric value (for validation)."""
    if isinstance(tree, str):
        if tree in CONSTANTS:
            return float(CONSTANTS[tree])
        try:
            return float(tree)
        except ValueError:
            return None

    op, args = tree
    vals = [eval_sexpr(a) for a in args]
    if any(v is None for v in vals):
        return None

    try:
        if op == "add":
            return vals[0] + vals[1]
        elif op == "subtract":
            return vals[0] - vals[1]
        elif op == "multiply":
            return vals[0] * vals[1]
        elif op == "divide":
            return vals[0] / vals[1] if vals[1] != 0 else None
        elif op == "power":
            return vals[0] ** vals[1]
        elif op == "sqrt":
            return math.sqrt(vals[0]) if vals[0] >= 0 else None
        elif op == "log":
            return math.log(vals[0]) if vals[0] > 0 else None
        elif op == "negate":
            return -vals[0]
        elif op == "inverse":
            return 1.0 / vals[0] if vals[0] != 0 else None
        elif op == "floor":
            return math.floor(vals[0])
        elif op == "factorial":
            return math.factorial(int(vals[0])) if vals[0] >= 0 and vals[0] == int(vals[0]) else None
        elif op == "max":
            return max(vals)
        elif op == "min":
            return min(vals)
        elif op in ("reminder", "remainder"):
            return vals[0] % vals[1] if vals[1] != 0 else None
        elif op == "gcd":
            return math.gcd(int(vals[0]), int(vals[1]))
        elif op == "lcm":
            return abs(vals[0] * vals[1]) / math.gcd(int(vals[0]), int(vals[1])) if vals[1] != 0 else None
        elif op == "choose":
            return math.comb(int(vals[0]), int(vals[1]))
        elif op == "permutation":
            return math.perm(int(vals[0]), int(vals[1]))
        elif op == "sine":
            return math.sin(vals[0])
        elif op == "cosine":
            return math.cos(vals[0])
        elif op == "tangent":
            return math.tan(vals[0])
        elif op == "circle_area":
            return math.pi * vals[0] ** 2
        elif op == "circumface":
            return 2 * math.pi * vals[0]
        elif op == "rectangle_area":
            return vals[0] * vals[1]
        elif op == "rectangle_perimeter":
            return 2 * (vals[0] + vals[1])
        elif op == "square_area":
            return vals[0] ** 2
        elif op == "square_perimeter":
            return 4 * vals[0]
        elif op == "triangle_area":
            return 0.5 * vals[0] * vals[1]
        elif op == "volume_cube":
            return vals[0] ** 3
        elif op == "volume_cylinder":
            return math.pi * vals[0] ** 2 * vals[1]
        elif op == "volume_rectangular_prism":
            return vals[0] * vals[1] * vals[2]
        elif op == "volume_sphere":
            return (4/3) * math.pi * vals[0] ** 3
        elif op == "volume_cone":
            return (1/3) * math.pi * vals[0] ** 2 * vals[1]
        elif op == "surface_cube":
            return 6 * vals[0] ** 2
        elif op == "surface_cylinder":
            return 2 * math.pi * vals[0] * (vals[0] + vals[1])
        elif op == "surface_rectangular_prism":
            return 2 * (vals[0]*vals[1] + vals[1]*vals[2] + vals[0]*vals[2])
        elif op == "surface_sphere":
            return 4 * math.pi * vals[0] ** 2
        elif op == "cube_edge_by_volume":
            return vals[0] ** (1/3)
        elif op == "square_edge_by_perimeter":
            return vals[0] / 4
        elif op == "square_edge_by_area":
            return math.sqrt(vals[0])
        elif op == "diagonal":
            return math.sqrt(vals[0]**2 + vals[1]**2)
        elif op == "triangle_area_three_edges":
            a, b, c = vals
            s = (a + b + c) / 2
            area_sq = s * (s-a) * (s-b) * (s-c)
            return math.sqrt(area_sq) if area_sq >= 0 else None
        elif op == "triangle_perimeter":
            return sum(vals)
        elif op == "rhombus_area":
            return 0.5 * vals[0] * vals[1]
        elif op == "quadrilateral_area":
            return 0.5 * vals[0] * (vals[1] + vals[2])
        elif op == "speed":
            return vals[0] / vals[1] if vals[1] != 0 else None
        elif op == "stream_speed":
            return (vals[0] - vals[1]) / 2
        elif op == "speed_in_still_water":
            return (vals[0] + vals[1]) / 2
        elif op == "negate_prob":
            return 1 - vals[0]
        elif op == "original_price_before_loss":
            return vals[0] * 100 / (100 - vals[1])
        elif op == "original_price_before_gain":
            return vals[0] * 100 / (100 + vals[1])
        elif op == "p_after_gain":
            return vals[0] * (100 + vals[1]) / 100
        elif op == "rhombus_perimeter":
            return 4 * vals[0]
        elif op == "square_perimeter":
            return 4 * vals[0]
        else:
            return None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def convert_dataset(split_name):
    """Convert a MathQA split to (problem, sexpr, computed_value) triples."""
    with open(DATA / f"{split_name}.json") as f:
        data = json.load(f)

    results = []
    errors = {"parse": 0, "unknown_op": 0, "eval_fail": 0}

    for ex in data:
        formula = ex["annotated_formula"]
        try:
            tokens = tokenize_formula(formula)
            tree = parse_formula(tokens)
            sexpr = tree_to_sexpr(tree)
            value = eval_sexpr(tree)

            results.append({
                "problem": ex["Problem"],
                "options": ex["options"],
                "correct": ex["correct"],
                "original_formula": formula,
                "selph_expr": sexpr,
                "computed_value": value,
                "rationale": ex.get("Rationale", ""),
            })
        except Exception as e:
            errors["parse"] += 1

    return results, errors


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        results, errors = convert_dataset(split)
        eval_ok = sum(1 for r in results if r["computed_value"] is not None)
        print(f"\n{split}: {len(results)} converted, {eval_ok} evaluable, errors: {errors}")

        # Show some examples
        if split == "train":
            for r in results[:8]:
                print(f"\n  Problem: {r['problem'][:100]}...")
                print(f"  Formula: {r['original_formula']}")
                print(f"  SELPH:   {r['selph_expr']}")
                print(f"  Value:   {r['computed_value']}")

    # Save converted train set
    results, _ = convert_dataset("train")
    out_path = DATA / "train_selph.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} examples to {out_path}")
