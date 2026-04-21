"""Generate refinement traces with query (z) channel for RefinementLM v2.

Each step now has:
    - x: task context (q_type + entities)
    - y: current expression
    - y(x): execution feedback (CORRECT/WRONG/ERROR/INCOMPLETE)
    - z: query expression (a sub-expression the model chose to evaluate)
    - z(x,y): query result (what that sub-expression evaluates to)
    - target_y: next expression
    - target_z: next query

Query traces are synthesized from the target expression structure:
for a target like (population (capital "Spain")), useful queries include
evaluating sub-expressions: (capital "Spain") → "Madrid".
"""
from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Optional

from neural_selph.iterative.generate_traces import (
    parse_sexpr, tree_to_sexpr, skeleton_of, find_first_hole,
    set_at_path, evaluate_in_kb, extract_expr, extract_question_context,
    decompose_topdown, generate_wrong_start_trace,
    GEO_FUNCTIONS, DATA_DIR,
)


def synthesize_query(current_expr: str, target_expr: str, kb: dict) -> tuple[str, str]:
    """Synthesize a useful query for this refinement step.

    Strategy: find a sub-expression in the target that evaluates to something
    informative. Prefer sub-expressions that differ from the current expression.

    Returns (query_expr, query_result) or ("NO_OP", "") if no useful query.
    """
    if target_expr == "NO_OP":
        return "NO_OP", ""

    target_tree = parse_sexpr(target_expr)
    if isinstance(target_tree, str):
        return "NO_OP", ""

    # Collect evaluable sub-expressions from the target
    subs = []
    _collect_subexprs(target_tree, subs)

    # Try each sub-expression as a query
    for sub in subs:
        sub_str = tree_to_sexpr(sub) if isinstance(sub, list) else sub
        if sub_str == target_expr:
            continue  # skip the full expression itself
        if "_HOLE_" in sub_str:
            continue
        result = evaluate_in_kb(sub if isinstance(sub, list) else parse_sexpr(sub_str), kb)
        if result is not None:
            return sub_str, str(result)

    return "NO_OP", ""


def _collect_subexprs(tree, acc: list):
    """Collect all sub-expressions (non-leaf) from a tree."""
    if isinstance(tree, list) and len(tree) >= 2:
        acc.append(tree)
        for child in tree[1:]:
            _collect_subexprs(child, acc)


def generate_v2_traces(data: list[dict], kb: dict) -> list[list[dict]]:
    """Generate refinement traces with query channel.

    Each step dict has:
        current_expr, feedback_type, feedback_value,
        query_expr, query_result,
        target_expr, target_query,
        q_type, entities
    """
    traces = []
    for ex in data:
        expr_str = extract_expr(ex["completion"])
        tree = parse_sexpr(expr_str)
        depth = ex["depth"]
        q_type, entities = extract_question_context(ex["prompt"], expr_str)

        # Top-down decomposition trace
        td_steps = decompose_topdown(tree, kb)
        if td_steps:
            enriched = []
            for i, step in enumerate(td_steps):
                curr, fb_type, fb_val, target = step

                # Previous query result (from the last step's target_query)
                if i > 0 and enriched[-1]["target_query"] != "NO_OP":
                    q_expr = enriched[-1]["target_query"]
                    q_result = enriched[-1]["target_query_result"]
                else:
                    q_expr, q_result = "NO_OP", ""

                # Synthesize query for this step
                tgt_q, tgt_q_result = synthesize_query(curr, target, kb)

                enriched.append({
                    "current_expr": curr,
                    "feedback_type": fb_type,
                    "feedback_value": fb_val,
                    "query_expr": q_expr,
                    "query_result": q_result,
                    "target_expr": target,
                    "target_query": tgt_q,
                    "target_query_result": tgt_q_result,
                    "q_type": q_type,
                    "entities": entities,
                })
            traces.append(enriched)

        # Wrong-start trace
        if depth >= 2:
            wrong_steps = generate_wrong_start_trace(tree, kb, GEO_FUNCTIONS)
            if wrong_steps:
                enriched = []
                for i, step in enumerate(wrong_steps):
                    curr, fb_type, fb_val, target = step
                    if i > 0 and enriched[-1]["target_query"] != "NO_OP":
                        q_expr = enriched[-1]["target_query"]
                        q_result = enriched[-1]["target_query_result"]
                    else:
                        q_expr, q_result = "NO_OP", ""
                    tgt_q, tgt_q_result = synthesize_query(curr, target, kb)
                    enriched.append({
                        "current_expr": curr,
                        "feedback_type": fb_type,
                        "feedback_value": fb_val,
                        "query_expr": q_expr,
                        "query_result": q_result,
                        "target_expr": target,
                        "target_query": tgt_q,
                        "target_query_result": tgt_q_result,
                        "q_type": q_type,
                        "entities": entities,
                    })
                traces.append(enriched)

    return traces


def main():
    random.seed(42)
    data = json.loads((DATA_DIR / "geo_train.json").read_text())
    kb = json.loads((DATA_DIR / "geo_kb.json").read_text())

    traces = generate_v2_traces(data, kb)

    all_steps = [step for trace in traces for step in trace]
    n_with_query = sum(1 for s in all_steps if s["target_query"] != "NO_OP")

    print(f"Total traces: {len(traces)}")
    print(f"Total steps: {len(all_steps)}")
    print(f"Steps with query: {n_with_query} ({n_with_query/len(all_steps)*100:.0f}%)")

    # Show examples
    print("\n--- Example traces ---")
    for trace in traces[:3]:
        print()
        for step in trace:
            tq = step['target_query'][:30] if step['target_query'] != 'NO_OP' else '-'
            print(f"  {step['feedback_type']:12s} y={step['current_expr'][:35]:35s} "
                  f"→ {step['target_expr'][:35]}  z→{tq}")

    out_path = DATA_DIR / "geo_traces_v2.json"
    out_path.write_text(json.dumps(all_steps, indent=2))
    print(f"\nSaved {len(all_steps)} steps to {out_path}")

    trace_path = DATA_DIR / "geo_traces_v2_full.json"
    trace_path.write_text(json.dumps(traces, indent=2))
    print(f"Saved {len(traces)} full traces to {trace_path}")


if __name__ == "__main__":
    main()
