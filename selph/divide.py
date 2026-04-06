"""Divide-and-conquer synthesis for SELPH.

When a spec has multiple distinct output values, decomposes the problem:
  1. Group examples by output value
  2. For each pair of adjacent groups, find a separating condition
  3. For each group, synthesize the output expression
  4. Compose into nested if-expressions

This handles multi-way classification and piecewise functions that
the flat enumerative search can't reach.
"""

from __future__ import annotations
from typing import Any
from .ast import Node, Symbol, Number, String, List, Spec, GoalExamples
from .eval import eval_node, apply_fn, standard_env, Env, EvalError
from .synthesize import (
    synthesize, SynthesisResult, Component, _default_components,
    _expand_layer_iter, types_compatible, _ast_to_eval_value,
)
from .types import TNum, TStr, TBool, TFn, Type


def synthesize_divide_and_conquer(
    spec: Spec,
    max_depth: int = 2,
    max_candidates: int = 20000,
    extra_components: list[Component] | None = None,
    env: Env | None = None,
) -> SynthesisResult:
    """Attempt divide-and-conquer synthesis on a spec.

    Only applies when the spec has GoalExamples with multiple
    distinct output values (suggesting piecewise/classification tasks).
    """
    result = SynthesisResult()

    if not isinstance(spec.goal, GoalExamples) or not spec.goal.pairs:
        return result

    if env is None:
        env = standard_env()

    goal = spec.goal
    inputs = [_ast_to_eval_value(p[0]) for p in goal.pairs]
    outputs = [_ast_to_eval_value(p[1]) for p in goal.pairs]

    # Group examples by output value
    groups: dict[Any, list[int]] = {}
    for i, out in enumerate(outputs):
        key = out
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    # Only useful if there are 2+ distinct output values
    if len(groups) < 2:
        return result

    # Sort groups by the mean input value (gives a natural ordering for thresholds)
    sorted_groups = sorted(groups.items(), key=lambda kv: _mean_input(kv[1], inputs))

    # Build the nested if-expression bottom-up
    body = _build_nested_if(
        sorted_groups, inputs, outputs, env,
        max_depth=max_depth,
        max_candidates=max_candidates,
        extra_components=extra_components,
    )

    if body is None:
        return result

    # Wrap in lambda and verify
    fn_node = List((Symbol("lambda"), List((Symbol("x"),)), body))
    try:
        fn_val = eval_node(fn_node, env)
        all_correct = True
        for inp, expected in zip(inputs, outputs):
            actual = apply_fn(fn_val, [inp])
            if actual != expected:
                all_correct = False
                break

        if all_correct:
            result.program = fn_node
            result.source = repr(fn_node)
            result.found = True
    except (EvalError, Exception):
        pass

    return result


def _build_nested_if(
    sorted_groups: list[tuple[Any, list[int]]],
    inputs: list[Any],
    outputs: list[Any],
    env: Env,
    max_depth: int,
    max_candidates: int,
    extra_components: list[Component] | None,
) -> Node | None:
    """Recursively build nested if-expressions for sorted output groups.

    For 2 groups: (if condition then_expr else_expr)
    For 3+ groups: (if condition then_expr (if ...))
    """
    if len(sorted_groups) == 1:
        # Single group: synthesize an expression that produces this output
        out_val, indices = sorted_groups[0]
        return _synthesize_branch(out_val, indices, inputs, outputs, env,
                                   max_depth, max_candidates, extra_components)

    # Split into first group vs rest
    first_val, first_indices = sorted_groups[0]
    rest_groups = sorted_groups[1:]
    rest_indices = []
    for _, idx_list in rest_groups:
        rest_indices.extend(idx_list)

    # Find a condition that separates first_indices (True) from rest_indices (False)
    condition = _find_separator(first_indices, rest_indices, inputs, env,
                                max_depth, max_candidates, extra_components)
    if condition is None:
        # Try splitting the other way: rest is True, first is False
        condition = _find_separator(rest_indices, first_indices, inputs, env,
                                     max_depth, max_candidates, extra_components)
        if condition is not None:
            # Swap: condition means "rest", so first goes in else branch
            then_branch = _build_nested_if(rest_groups, inputs, outputs, env,
                                            max_depth, max_candidates, extra_components)
            else_branch = _synthesize_branch(first_val, first_indices, inputs, outputs, env,
                                              max_depth, max_candidates, extra_components)
            if then_branch is not None and else_branch is not None:
                return List((Symbol("if"), condition, then_branch, else_branch))
        return None

    # Build then-branch (for first group) and else-branch (for rest, recursively)
    then_branch = _synthesize_branch(first_val, first_indices, inputs, outputs, env,
                                      max_depth, max_candidates, extra_components)
    if then_branch is None:
        return None

    else_branch = _build_nested_if(rest_groups, inputs, outputs, env,
                                    max_depth, max_candidates, extra_components)
    if else_branch is None:
        return None

    return List((Symbol("if"), condition, then_branch, else_branch))


def _find_separator(
    true_indices: list[int],
    false_indices: list[int],
    inputs: list[Any],
    env: Env,
    max_depth: int,
    max_candidates: int,
    extra_components: list[Component] | None,
) -> Node | None:
    """Find a boolean expression that is True for true_indices and False for false_indices."""
    B = TBool()

    # Build component library
    comps = _default_components()
    if extra_components:
        comps.extend(extra_components)
    in_type = TNum() if isinstance(inputs[0], (int, float)) else TStr()
    comps.append(Component("x", Symbol("x"), in_type, 0))

    # Generate bool-typed programs and test them
    depth0 = [(c.node, c.type) for c in comps if c.arity == 0]
    all_progs = list(depth0)

    for depth in range(max_depth + 1):
        if depth > 0:
            new_layer = list(_expand_layer_iter(
                all_progs[-len(all_progs):] if depth == 1 else new_layer,
                all_progs, comps))
            all_progs.extend(new_layer)

        # Check each bool-typed program
        for node, typ in (all_progs if depth == 0 else new_layer):
            if not types_compatible(typ, B):
                continue

            try:
                fn_node = List((Symbol("lambda"), List((Symbol("x"),)), node))
                fn = eval_node(fn_node, env)

                all_match = True
                for i in true_indices:
                    if not apply_fn(fn, [inputs[i]]):
                        all_match = False
                        break
                if not all_match:
                    continue

                for i in false_indices:
                    if apply_fn(fn, [inputs[i]]):
                        all_match = False
                        break
                if all_match:
                    return node
            except Exception:
                continue

    return None


def _synthesize_branch(
    out_val: Any,
    indices: list[int],
    inputs: list[Any],
    outputs: list[Any],
    env: Env,
    max_depth: int,
    max_candidates: int,
    extra_components: list[Component] | None,
) -> Node | None:
    """Synthesize an expression that produces out_val for the given input indices.

    If all inputs in this group map to the same constant output, just return
    the constant. Otherwise, synthesize a function from the group's examples.
    """
    # Check if it's a constant function for this group
    group_outputs = [outputs[i] for i in indices]
    if len(set(group_outputs)) == 1:
        # Constant output
        val = group_outputs[0]
        if isinstance(val, (int, float)):
            return Number(float(val))
        if isinstance(val, str):
            return String(val)

    # Need a non-constant expression — synthesize from group examples
    group_pairs = []
    for i in indices:
        in_node = Number(float(inputs[i])) if isinstance(inputs[i], (int, float)) else String(inputs[i])
        out_node = Number(float(outputs[i])) if isinstance(outputs[i], (int, float)) else String(outputs[i])
        group_pairs.append((in_node, out_node))

    out_type = "number" if isinstance(outputs[indices[0]], (int, float)) else "string"
    sub_spec = Spec(
        type_expr=Symbol(out_type),
        goal=GoalExamples(pairs=tuple(group_pairs)),
    )

    sub_result = synthesize(
        sub_spec, max_depth=max_depth, max_candidates=max_candidates,
        extra_components=extra_components, env=env,
    )

    if sub_result.found:
        # Extract the body from (lambda (x) body)
        return sub_result.program.elements[2]

    return None


def _mean_input(indices: list[int], inputs: list[Any]) -> float:
    """Mean of input values at given indices (for ordering groups)."""
    vals = [inputs[i] for i in indices if isinstance(inputs[i], (int, float))]
    return sum(vals) / len(vals) if vals else 0.0
