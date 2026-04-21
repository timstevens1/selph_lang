"""Explore MathQA data format and operator vocabulary."""
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"

# Load all splits
splits = {}
for name in ["train", "dev", "test"]:
    with open(DATA / f"{name}.json") as f:
        splits[name] = json.load(f)

print("Split sizes:", {k: len(v) for k, v in splits.items()})

# Look at a few examples
for i in range(5):
    ex = splits["train"][i]
    print(f"\n--- Example {i} ---")
    print(f"Problem: {ex['Problem'][:150]}")
    print(f"Rationale: {ex['Rationale'][:150]}")
    print(f"Options: {ex['options']}")
    print(f"Correct: {ex['correct']}")
    print(f"Annotated formula: {ex['annotated_formula']}")
    print(f"Linear formula: {ex['linear_formula']}")

# Collect operators and constants
op_counts = Counter()
const_counts = Counter()
for ex in splits["train"]:
    formula = ex["annotated_formula"]
    for m in re.finditer(r'([a-z_]+)\(', formula):
        op_counts[m.group(1)] += 1
    for m in re.finditer(r'(const_\w+)', formula):
        const_counts[m.group(1)] += 1

print(f"\n\nOperators ({len(op_counts)}):")
for op, count in op_counts.most_common():
    print(f"  {op}: {count}")

print(f"\nConstants ({len(const_counts)}):")
for c, count in const_counts.most_common(30):
    print(f"  {c}: {count}")

# Load reference files
print("\n--- operation_list.txt ---")
print((DATA / "operation_list.txt").read_text()[:2000])

print("\n--- constant_list.txt ---")
print((DATA / "constant_list.txt").read_text()[:2000])

# Nesting depth analysis
def nesting_depth(s):
    d, mx = 0, 0
    for c in s:
        if c == '(':
            d += 1
            mx = max(mx, d)
        elif c == ')':
            d -= 1
    return mx

depths = Counter()
for ex in splits["train"]:
    depths[nesting_depth(ex["annotated_formula"])] += 1

print("\nNesting depth distribution (train):")
for d in sorted(depths):
    print(f"  depth {d}: {depths[d]}")
