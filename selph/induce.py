"""Failure-driven library induction for SELPH.

When synthesis fails on a task, this module analyzes the failure and
attempts to decompose the spec into sub-problems that can be solved
independently, then composes the solutions into a new primitive.

The key insight: if a task needs f(g(x)) but neither f nor g is in
the library, we can:
  1. Look at the input/output examples
  2. Try to find an intermediate value y = g(x) such that f(y) = output
  3. Synthesize g and f independently (both are simpler)
  4. Compose them into a new primitive

This is the "subagent decomposition" from §7 applied to synthesis itself.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .ast import Node, Symbol, Number, String, List, Spec, GoalExamples
from .eval import (
    eval_node, apply_fn, standard_env, Env, Closure, BuiltinFn, EvalError,
)
from .synthesize import synthesize, SynthesisResult, Component
from .library import Abstraction, promote_solved, register_abstractions
from .types import TNum, TStr, TBool, TFn, Type


# ── Decomposition result ─────────────────────────────────────────────

@dataclass
class InductionResult:
    """Result of failure-driven induction."""
    success: bool = False
    program: Node | None = None
    source: str | None = None
    new_primitives: list[Abstraction] = field(default_factory=list)
    decomposition: str | None = None  # human-readable description
    candidates_explored: int = 0


# ── Main entry point ─────────────────────────────────────────────────

def induce_from_failure(spec: Spec, max_depth: int = 2,
                        max_candidates: int = 10000,
                        extra_components: list[Component] | None = None,
                        env: Env | None = None) -> InductionResult:
    """Attempt to solve a failed spec by decomposition.

    Tries multiple decomposition strategies:
      1. Intermediate value search: find y such that input->y and y->output
         are both solvable
      2. Constant extraction: try the task with additional constants
      3. Type bridge: for cross-type tasks, find a bridging function

    Returns an InductionResult with any discovered primitives.
    """
    result = InductionResult()

    if not isinstance(spec.goal, GoalExamples):
        return result

    goal = spec.goal
    if not goal.pairs:
        return result

    if env is None:
        env = standard_env()

    # Strategy 1: intermediate value decomposition
    iv_result = _try_intermediate_values(
        spec, goal, max_depth, max_candidates, extra_components, env)
    if iv_result.success:
        return iv_result
    result.candidates_explored += iv_result.candidates_explored

    # Strategy 2: constant discovery
    const_result = _try_constant_discovery(
        spec, goal, max_depth, max_candidates, extra_components, env)
    if const_result.success:
        return const_result
    result.candidates_explored += const_result.candidates_explored

    return result


# ── Strategy 1: Intermediate value decomposition ─────────────────────

def _try_intermediate_values(spec: Spec, goal: GoalExamples,
                             max_depth: int, max_candidates: int,
                             extra_components: list[Component] | None,
                             env: Env) -> InductionResult:
    """Try to find intermediate values that split the task into two steps.

    For each candidate intermediate type, generate possible intermediate
    values by running all known unary functions on the inputs, then check
    if the output can be produced from those intermediates.
    """
    result = InductionResult()

    inputs = [_extract_value(p[0]) for p in goal.pairs]
    outputs = [_extract_value(p[1]) for p in goal.pairs]

    if any(v is None for v in inputs) or any(v is None for v in outputs):
        return result

    # Collect candidate intermediate values by running every unary builtin on inputs
    intermediates = _generate_intermediates(inputs, env)

    # For each candidate set of intermediates, try to synthesize
    # step1: input -> intermediate  and  step2: intermediate -> output
    for mid_name, mid_values in intermediates.items():
        if len(mid_values) != len(inputs):
            continue

        # Build step 2 spec: intermediate -> output
        step2_pairs = []
        valid = True
        for mid_val, out_val in zip(mid_values, outputs):
            mid_node = _value_to_node(mid_val)
            out_node = _value_to_node(out_val)
            if mid_node is None or out_node is None:
                valid = False
                break
            step2_pairs.append((mid_node, out_node))

        if not valid or not step2_pairs:
            continue

        # Check that intermediates aren't all the same as inputs or outputs
        if mid_values == inputs or mid_values == outputs:
            continue

        # Check that intermediates are not all identical (constant function)
        if len(set(repr(v) for v in mid_values)) == 1:
            continue

        out_type_name = _infer_type_name(outputs[0])
        mid_type_name = _infer_type_name(mid_values[0])

        step2_spec = Spec(
            type_expr=Symbol(out_type_name),
            goal=GoalExamples(pairs=tuple(step2_pairs)),
        )

        # Try to synthesize step 2
        step2_result = synthesize(
            step2_spec, max_depth=max_depth,
            max_candidates=max_candidates // 4,
            extra_components=extra_components,
            env=env,
        )
        result.candidates_explored += step2_result.candidates_explored

        if not step2_result.found:
            continue

        # Step 2 works! Now build step 1 spec: input -> intermediate
        step1_pairs = []
        for in_val, mid_val in zip(inputs, mid_values):
            in_node = _value_to_node(in_val)
            mid_node = _value_to_node(mid_val)
            if in_node is None or mid_node is None:
                valid = False
                break
            step1_pairs.append((in_node, mid_node))

        if not step1_pairs:
            continue

        step1_spec = Spec(
            type_expr=Symbol(mid_type_name),
            goal=GoalExamples(pairs=tuple(step1_pairs)),
        )

        step1_result = synthesize(
            step1_spec, max_depth=max_depth,
            max_candidates=max_candidates // 4,
            extra_components=extra_components,
            env=env,
        )
        result.candidates_explored += step1_result.candidates_explored

        if not step1_result.found:
            continue

        # Both steps solved! Compose them.
        step1_body = step1_result.program.elements[2]  # lambda body
        step2_body = step2_result.program.elements[2]

        # Build composed program: (lambda (x) (step2 (step1 x)))
        # Replace x in step2_body with step1_body
        composed_body = _substitute_var(step2_body, "x", step1_body)
        composed = List((Symbol("lambda"), List((Symbol("x"),)), composed_body))

        # Verify the composed program
        try:
            fn_val = eval_node(composed, env)
            all_correct = True
            for in_val, out_val in zip(inputs, outputs):
                actual = apply_fn(fn_val, [in_val])
                if actual != out_val:
                    all_correct = False
                    break

            if all_correct:
                # Create the new primitive
                in_type = TStr() if isinstance(inputs[0], str) else TNum()
                out_type = TStr() if isinstance(outputs[0], str) else TNum()

                abstraction = Abstraction(
                    name=f"induced_{mid_name}",
                    params=["x"],
                    body=composed_body,
                    param_types=[in_type],
                    return_type=out_type,
                    frequency=1,
                    compression=0.0,
                )

                result.success = True
                result.program = composed
                result.source = repr(composed)
                result.new_primitives = [abstraction]
                result.decomposition = (
                    f"Decomposed via {mid_name}: "
                    f"step1={step1_result.source}, step2={step2_result.source}"
                )
                return result

        except (EvalError, Exception):
            continue

    return result


def _generate_intermediates(inputs: list[Any], env: Env) -> dict[str, list[Any]]:
    """Run all known unary functions on inputs to generate candidate intermediates."""
    unary_fns = [
        "string-upper", "string-lower", "string-reverse", "string-trim",
        "string-length", "abs", "negate",
    ]

    results = {}
    for fn_name in unary_fns:
        try:
            fn_val = env.lookup(fn_name)
        except Exception:
            continue

        mid_values = []
        valid = True
        for inp in inputs:
            try:
                mid = apply_fn(fn_val, [inp])
                mid_values.append(mid)
            except Exception:
                valid = False
                break

        if valid and mid_values:
            results[fn_name] = mid_values

    # Also try binary functions with small constants
    binary_fns = ["add", "subtract", "multiply"]
    small_constants = [1.0, 2.0, -1.0, 0.5]

    for fn_name in binary_fns:
        try:
            fn_val = env.lookup(fn_name)
        except Exception:
            continue

        for const in small_constants:
            mid_values = []
            valid = True
            for inp in inputs:
                try:
                    mid = apply_fn(fn_val, [inp, const])
                    mid_values.append(mid)
                except Exception:
                    valid = False
                    break

            if valid and mid_values:
                results[f"{fn_name}_{const}"] = mid_values

    return results


# ── Strategy 2: Constant discovery ───────────────────────────────────

def _try_constant_discovery(spec: Spec, goal: GoalExamples,
                            max_depth: int, max_candidates: int,
                            extra_components: list[Component] | None,
                            env: Env) -> InductionResult:
    """Try synthesis with additional inferred constants.

    Analyzes the input/output examples to discover useful constants:
      - Output/input ratios (for multiplication)
      - Output - input differences (for addition)
      - Common factors
    """
    result = InductionResult()

    inputs = [_extract_value(p[0]) for p in goal.pairs]
    outputs = [_extract_value(p[1]) for p in goal.pairs]

    if any(v is None for v in inputs) or any(v is None for v in outputs):
        return result

    # Only works for numeric tasks
    if not all(isinstance(v, (int, float)) for v in inputs + outputs):
        return result

    discovered_constants = set()

    # Check for constant differences
    diffs = set()
    for i, o in zip(inputs, outputs):
        if i != 0:
            diffs.add(o - i)
    if len(diffs) == 1:
        discovered_constants.add(diffs.pop())

    # Check for constant ratios
    ratios = set()
    for i, o in zip(inputs, outputs):
        if i != 0:
            ratios.add(o / i)
    if len(ratios) == 1:
        discovered_constants.add(ratios.pop())

    # Check for constant modular relationships
    for i, o in zip(inputs, outputs):
        if o != 0 and i != 0:
            discovered_constants.add(o / i)
            discovered_constants.add(o - i)

    if not discovered_constants:
        return result

    # Try synthesis with discovered constants
    extra_consts = [(c, TNum()) for c in discovered_constants
                    if c == int(c)]  # only integer constants for now

    synth_result = synthesize(
        spec, max_depth=max_depth,
        max_candidates=max_candidates,
        extra_components=extra_components,
        constants=extra_consts,
        env=env,
    )
    result.candidates_explored += synth_result.candidates_explored

    if synth_result.found:
        result.success = True
        result.program = synth_result.program
        result.source = synth_result.source
        result.decomposition = f"Solved with discovered constants: {discovered_constants}"

    return result


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_value(node: Node) -> Any:
    """Extract a Python value from an AST literal."""
    if isinstance(node, Number):
        return node.value
    if isinstance(node, String):
        return node.value
    return None


def _value_to_node(val: Any) -> Node | None:
    """Convert a Python value to an AST node."""
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return Number(float(val))
    if isinstance(val, str):
        return String(val)
    return None


def _infer_type_name(val: Any) -> str:
    """Infer SELPH type name from a value."""
    if isinstance(val, (int, float)):
        return "number"
    if isinstance(val, str):
        return "string"
    if isinstance(val, bool):
        return "bool"
    return "unknown"


def _substitute_var(node: Node, var_name: str, replacement: Node) -> Node:
    """Replace a variable in an AST with another expression."""
    if isinstance(node, Symbol) and node.name == var_name:
        return replacement
    if isinstance(node, List):
        return List(tuple(_substitute_var(e, var_name, replacement)
                          for e in node.elements))
    return node
