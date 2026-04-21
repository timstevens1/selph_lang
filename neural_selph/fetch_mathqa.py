"""Download MathQA and explore the operation program format."""
from datasets import load_dataset
import json
import re

ds = load_dataset("math_qa")

print("Splits:", {k: len(v) for k, v in ds.items()})
print("Columns:", ds["train"].column_names)
print()

# Look at a few examples to understand the format
for i in range(10):
    ex = ds["train"][i]
    print(f"--- Example {i} ---")
    print(f"Problem: {ex['Problem'][:120]}...")
    print(f"Rationale: {ex['Rationale'][:120]}...")
    print(f"Options: {ex['options']}")
    print(f"Correct: {ex['correct']}")
    print(f"Annotated formula: {ex['annotated_formula']}")
    print(f"Linear formula: {ex['linear_formula']}")
    print()

# Collect unique operators from annotated_formula
ops = set()
for split in ["train", "validation", "test"]:
    for ex in ds[split]:
        # Extract function names from annotated_formula
        for m in re.finditer(r'([a-z_]+)\(', ex["annotated_formula"]):
            ops.add(m.group(1))

print(f"\nUnique operators ({len(ops)}):")
for op in sorted(ops):
    print(f"  {op}")

# Count by operator frequency
from collections import Counter
op_counts = Counter()
for ex in ds["train"]:
    for m in re.finditer(r'([a-z_]+)\(', ex["annotated_formula"]):
        op_counts[m.group(1)] += 1

print(f"\nOperator frequency (train split):")
for op, count in op_counts.most_common():
    print(f"  {op}: {count}")

# Also look at constants used
consts = set()
for ex in ds["train"]:
    for m in re.finditer(r'const_(\w+)', ex["annotated_formula"]):
        consts.add(m.group(1))

print(f"\nConstants ({len(consts)}):")
for c in sorted(consts):
    print(f"  const_{c}")
