#!/usr/bin/env python3
"""Convert ARC JSON tasks to SELPH grow-v2 curriculum format.

Usage: python3 arc_to_curriculum.py <dir_or_file> [--max N]

Outputs bare S-expression format (no #grid) compatible with grow-v2's
node_to_value parser. Each grid is nested lists: ((1 2) (3 4)).
"""
import json, sys, os, glob

def grid_to_sexp(g):
    """Convert [[int,...], ...] to S-expression string."""
    rows = " ".join("(" + " ".join(str(c) for c in row) + ")" for row in g)
    return "(" + rows + ")"

def task_to_sexp(task_id, task):
    """Convert one ARC task to task-args format (arity 1)."""
    pairs = task["train"]
    lines = [f'(task-args "{task_id}" 1']
    for pair in pairs:
        inp = grid_to_sexp(pair["input"])
        out = grid_to_sexp(pair["output"])
        # Arity 1: input wrapped in a 1-element list
        lines.append(f"  (({inp}) {out})")
    # Add test pairs as held-out
    for pair in task.get("test", []):
        inp = grid_to_sexp(pair["input"])
        if "output" in pair:
            out = grid_to_sexp(pair["output"])
            lines.append(f"  (test ({inp}) {out})")
    lines.append(")")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 arc_to_curriculum.py <dir_or_file> [--max N]", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    max_tasks = None
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        max_tasks = int(sys.argv[idx + 1])

    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        files = [path]

    if max_tasks:
        files = files[:max_tasks]

    print("; ARC curriculum (auto-generated)")
    print(f"; Source: {path}")
    print(f"; Tasks: {len(files)}")
    print()

    for f in files:
        task_id = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            task = json.load(fh)
        print(task_to_sexp(task_id, task))
        print()

if __name__ == "__main__":
    main()
