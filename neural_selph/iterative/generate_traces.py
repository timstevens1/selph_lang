"""Generate iterative refinement traces from geo_train.json.

For each example, decompose the correct expression into a multi-step
refinement sequence. Also generate corrupted starting points that
require the model to identify and fix errors via execution feedback.

Trace format per step:
  (current_expr, feedback_type, feedback_detail, next_expr)

Where feedback_type is one of:
  INCOMPLETE - expression has holes
  WRONG - expression evaluates but gives wrong answer
  ERROR - expression fails to evaluate (bad function, type mismatch)
  CORRECT - expression gives the right answer (next_expr = NO_OP)
"""
from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"

GEO_FUNCTIONS = [
    "capital", "population", "continent", "currency", "language",
    "area", "head-of-state", "country-of", "largest-by-population", "divide",
]


def parse_sexpr(s: str) -> list:
    """Parse s-expression to nested list. Returns list tree."""
    s = s.strip()
    if not s.startswith("("):
        return s  # atom
    inner = s[1:-1].strip()
    result = []
    i = 0
    while i < len(inner):
        c = inner[i]
        if c in " \t\n":
            i += 1
        elif c == "(":
            depth = 1
            j = i + 1
            while j < len(inner) and depth > 0:
                if inner[j] == "(":
                    depth += 1
                elif inner[j] == ")":
                    depth -= 1
                j += 1
            result.append(parse_sexpr(inner[i:j]))
            i = j
        elif c == ")":
            # Stray close paren from malformed input — skip it
            i += 1
        elif c == '"':
            j = i + 1
            while j < len(inner) and inner[j] != '"':
                j += 1
            result.append(inner[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < len(inner) and inner[j] not in ' \t\n()"':
                j += 1
            result.append(inner[i:j])
            i = j
    return result


def tree_to_sexpr(tree) -> str:
    """Convert nested list back to s-expression string."""
    if isinstance(tree, str):
        return tree
    return "(" + " ".join(tree_to_sexpr(t) for t in tree) + ")"


def skeleton_of(t):
    """Return one-level skeleton: (func _HOLE_ ...) for lists, atom for strings."""
    if isinstance(t, str):
        return t
    return [t[0]] + ["_HOLE_" for _ in t[1:]]


def find_first_hole(template, original, path=None):
    """Find the first _HOLE_ in template and return (path, original_subtree)."""
    if path is None:
        path = []
    if isinstance(template, str):
        if template == "_HOLE_":
            return (path, original)
        return None
    if not isinstance(original, list):
        return None
    for i in range(len(template)):
        if i >= len(original):
            break
        result = find_first_hole(template[i], original[i], path + [i])
        if result is not None:
            return result
    return None


def set_at_path(template, path, value):
    """Set value at path in nested list. Deep copies first."""
    t = copy.deepcopy(template)
    if not path:
        return value
    node = t
    for idx in path[:-1]:
        node = node[idx]
    node[path[-1]] = value
    return t


def decompose_topdown(tree, kb: dict) -> list[tuple[str, str, str, str]]:
    """Generate a top-down refinement trace from _HOLE_ to the full expression.

    BFS expansion: start with _HOLE_, repeatedly replace the leftmost
    _HOLE_ with the corresponding subtree's skeleton.
    """
    target = tree_to_sexpr(tree)
    steps = []

    if isinstance(tree, str):
        steps.append(("_HOLE_", "INCOMPLETE", "", target))
        steps.append((target, "CORRECT", "", "NO_OP"))
        return steps

    # Start: _HOLE_ -> skeleton of top level
    first_skel = skeleton_of(tree)
    steps.append(("_HOLE_", "INCOMPLETE", "", tree_to_sexpr(first_skel)))

    current = first_skel

    for _ in range(20):  # safety limit
        hole = find_first_hole(current, tree)
        if hole is None:
            break
        path, orig = hole
        expanded = skeleton_of(orig)
        next_template = set_at_path(current, path, expanded)
        current_expr = tree_to_sexpr(current)
        next_expr = tree_to_sexpr(next_template)
        if current_expr != next_expr:
            steps.append((current_expr, "INCOMPLETE", "", next_expr))
        current = next_template

    # Final: correct
    final_expr = tree_to_sexpr(current)
    steps.append((final_expr, "CORRECT", "", "NO_OP"))
    return steps


def evaluate_in_kb(tree, kb: dict) -> Optional[str]:
    """Evaluate an s-expression tree against the geo KB."""
    if isinstance(tree, str):
        if tree.startswith('"') and tree.endswith('"'):
            return tree[1:-1]
        try:
            return str(float(tree)) if "." in tree else str(int(tree))
        except ValueError:
            return None

    if not tree or not isinstance(tree[0], str):
        return None

    func = tree[0]
    eval_args = []
    for arg in tree[1:]:
        val = evaluate_in_kb(arg, kb)
        if val is None:
            return None
        eval_args.append(val)

    if not eval_args:
        return None

    arg0 = eval_args[0]

    if func == "capital":
        return kb.get(arg0, {}).get("capital")
    elif func == "population":
        for country_data in kb.values():
            if country_data.get("capital") == arg0 and country_data.get("capital_population"):
                return str(country_data["capital_population"])
        return str(kb.get(arg0, {}).get("population", ""))
    elif func == "continent":
        return kb.get(arg0, {}).get("continent")
    elif func == "currency":
        return kb.get(arg0, {}).get("currency")
    elif func == "language":
        return kb.get(arg0, {}).get("language")
    elif func == "area":
        val = kb.get(arg0, {}).get("area_km2")
        return str(val) if val else None
    elif func == "head-of-state":
        return kb.get(arg0, {}).get("head_of_state")
    elif func == "country-of":
        for country, data in kb.items():
            if data.get("capital") == arg0:
                return country
        return None
    elif func == "largest-by-population":
        best, best_pop = None, 0
        for country, data in kb.items():
            if data.get("continent") == arg0:
                pop = data.get("population", 0)
                if pop and int(pop) > best_pop:
                    best, best_pop = country, int(pop)
        return best
    elif func == "divide":
        if len(eval_args) >= 2:
            try:
                a, b = float(eval_args[0]), float(eval_args[1])
                if b == 0:
                    return None
                return str(a / b)
            except (ValueError, ZeroDivisionError):
                return None
    return None


def generate_wrong_start_trace(tree, kb: dict, all_funcs: list[str]) -> list[tuple[str, str, str, str]]:
    """Generate a trace starting from a wrong expression."""
    target = tree_to_sexpr(tree)

    if isinstance(tree, str) or len(tree) < 2:
        return []

    func = tree[0]
    args = tree[1:]
    steps = []

    # Pattern 1: wrong function
    wrong_funcs = [f for f in all_funcs if f != func]
    if wrong_funcs:
        wrong_func = random.choice(wrong_funcs)
        wrong_tree = [wrong_func] + list(args)
        wrong_answer = evaluate_in_kb(wrong_tree, kb)
        if wrong_answer is not None:
            steps.append((tree_to_sexpr(wrong_tree), "WRONG", str(wrong_answer), target))
            steps.append((target, "CORRECT", "", "NO_OP"))
            return steps

    # Pattern 2: missing composition (depth >= 2)
    if isinstance(args[0], list) and len(args[0]) >= 2:
        inner_arg = args[0][1] if len(args[0]) > 1 else args[0][0]
        shortcut = [func, inner_arg]
        shortcut_answer = evaluate_in_kb(shortcut, kb)
        if shortcut_answer is not None:
            steps.append((tree_to_sexpr(shortcut), "WRONG", str(shortcut_answer), target))
            steps.append((target, "CORRECT", "", "NO_OP"))
            return steps

    return []


def extract_expr(completion: str) -> str:
    """Extract s-expression from completion string."""
    m = re.search(r"<tool_call>(.*?)</tool_call>", completion)
    if m:
        return m.group(1).strip()
    return completion.strip()


def extract_question_context(prompt: str, expr_str: str) -> tuple[str, list[str]]:
    """Extract question type and entity names from the prompt/expression.

    Returns (question_type_token, entity_names).
    """
    # Get the outermost function from the target expression as the question type
    tree = parse_sexpr(expr_str)
    if isinstance(tree, list) and tree:
        q_type = f"Q_{tree[0]}"
    else:
        q_type = "Q_unknown"

    # Extract entity names from the prompt's Q: line
    entities = []
    m = re.search(r'Q:\s*(.*?)(?:\n|$)', prompt)
    if m:
        question = m.group(1)
        for em in re.finditer(
            r'(?:^|(?<=\s))([A-Z][a-zA-Z]+(?:\s+(?:of|and|the|la|de|del|el|al|bin|von)\s+[A-Z]?[a-zA-Z]+)*(?:\s+[A-Z][a-zA-Z]+)*)',
            question
        ):
            candidate = em.group(1).strip()
            if candidate not in {"What", "Who", "Where", "Which", "How", "The", "Is",
                                  "Does", "Do", "Can", "Solve", "Find", "Calculate",
                                  "Functions", "Expression", "Square"}:
                entities.append(candidate)

    return q_type, entities


def synthesize_queries(current_expr: str, target_expr: str, kb: dict
                       ) -> list[tuple[str, str]]:
    """Synthesize useful queries for a refinement step.

    Returns list of (query_sexpr, result_str) pairs. These are sub-expressions
    of the target that evaluate to something informative.
    """
    if target_expr == "NO_OP":
        return []

    target_tree = parse_sexpr(target_expr)
    if isinstance(target_tree, str):
        return []

    queries = []
    _collect_evaluable_subs(target_tree, target_expr, kb, queries)
    return queries


def _collect_evaluable_subs(tree, full_expr: str, kb: dict, acc: list):
    """Collect evaluable sub-expressions (non-leaf, not the full expr)."""
    if isinstance(tree, list) and len(tree) >= 2:
        sub_str = tree_to_sexpr(tree)
        if sub_str != full_expr and "_HOLE_" not in sub_str:
            result = evaluate_in_kb(tree, kb)
            if result is not None:
                acc.append((sub_str, str(result)))
        for child in tree[1:]:
            _collect_evaluable_subs(child, full_expr, kb, acc)


def generate_all_traces(data: list[dict], kb: dict) -> list[list[dict]]:
    """Generate refinement traces for all examples.

    Each step is a dict with: current_expr, feedback_type, feedback_value,
    target_expr, q_type, entities.
    """
    traces = []
    for ex in data:
        expr_str = extract_expr(ex["completion"])
        tree = parse_sexpr(expr_str)
        depth = ex["depth"]

        # Extract question context
        q_type, entities = extract_question_context(ex["prompt"], expr_str)

        # Top-down decomposition trace
        td_trace = decompose_topdown(tree, kb)
        if td_trace:
            enriched = []
            for step in td_trace:
                queries = synthesize_queries(step[0], step[3], kb)
                enriched.append({
                    "current_expr": step[0],
                    "feedback_type": step[1],
                    "feedback_value": step[2],
                    "target_expr": step[3],
                    "queries": queries,
                    "q_type": q_type,
                    "entities": entities,
                })
            traces.append(enriched)

        # Wrong-start trace (for depth >= 2)
        if depth >= 2:
            wrong_trace = generate_wrong_start_trace(tree, kb, GEO_FUNCTIONS)
            if wrong_trace:
                enriched = []
                for step in wrong_trace:
                    queries = synthesize_queries(step[0], step[3], kb)
                    enriched.append({
                        "current_expr": step[0],
                        "feedback_type": step[1],
                        "feedback_value": step[2],
                        "target_expr": step[3],
                        "queries": queries,
                        "q_type": q_type,
                        "entities": entities,
                    })
                traces.append(enriched)

    return traces


def main():
    random.seed(42)
    data = json.loads((DATA_DIR / "geo_train.json").read_text())
    kb = json.loads((DATA_DIR / "geo_kb.json").read_text())

    traces = generate_all_traces(data, kb)

    # Flatten to individual steps
    all_steps = []
    for trace in traces:
        for step in trace:
            all_steps.append(step)

    from collections import Counter
    feedback_counts = Counter(s["feedback_type"] for s in all_steps)
    print(f"Total traces: {len(traces)}")
    print(f"Total steps: {len(all_steps)}")
    print(f"Feedback distribution: {dict(feedback_counts)}")
    print(f"Avg steps/trace: {len(all_steps) / len(traces):.1f}")

    # Show some example traces
    print("\n--- Example traces ---")
    for trace in traces[:3]:
        print()
        for step in trace:
            print(f"  {step['q_type']:20s} {step['feedback_type']:12s} "
                  f"{step['current_expr']:40s} -> {step['target_expr']}")

    out_path = DATA_DIR / "geo_traces.json"
    out_path.write_text(json.dumps(all_steps, indent=2))
    print(f"\nSaved {len(all_steps)} steps to {out_path}")

    trace_path = DATA_DIR / "geo_traces_full.json"
    trace_path.write_text(json.dumps(traces, indent=2))
    print(f"Saved {len(traces)} full traces to {trace_path}")


if __name__ == "__main__":
    main()
