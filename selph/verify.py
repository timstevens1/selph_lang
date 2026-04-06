"""Spec verification and reward computation for SELPH.

Implements the mechanical verification pipeline from §3.5 and §6.2:
  - Type match (hard gate)
  - Shape match (hard gate)
  - Constraint satisfaction (hard gate)
  - Goal satisfaction (Levels 0-2: fully mechanical)
  - Reward computation

This module is the "spec is the critic" system: no separate critic model
is needed for Stages 0-2. The spec defines what success looks like, and
verification is purely mechanical.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .ast import (
    Node, Symbol, Number, String, Bool, List,
    Spec, GoalExamples, GoalPattern, GoalSatisfy, GoalAll,
)
from .eval import eval_node, apply_fn, Env, standard_env, Closure, BuiltinFn, EvalError
from .types import (
    typecheck, InferenceState, TypeEnv, standard_type_env,
    TNum, TStr, TBool, TList, TFn, TNil, TTensor, TVar, Type,
    TypeCheckError,
)


# ── Verification result ──────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Structured result from verifying a program against a spec.

    This is the output that feeds into the reward computation and
    the critic/retry loop (§6.1 Loop 2).
    """
    # Hard gates (all must pass for any reward)
    type_match: bool = True
    shape_match: bool = True
    constraint_match: bool = True

    # Goal satisfaction (scalar 0-1)
    goal_score: float = 1.0

    # Detailed diagnostics for the critic
    type_error: str | None = None
    shape_error: str | None = None
    constraint_errors: list[str] = field(default_factory=list)
    goal_details: dict | None = None

    @property
    def hard_gate(self) -> bool:
        """All hard constraints must pass for any reward."""
        return self.type_match and self.shape_match and self.constraint_match

    @property
    def passed(self) -> bool:
        """Full pass: hard gate and goal score == 1.0."""
        return self.hard_gate and self.goal_score == 1.0

    def reward(self, w_goal: float = 0.7, w_parent: float = 0.2, w_global: float = 0.1,
               parent_success: float = 0.0, global_outcome: float = 0.0) -> float:
        """Compute the reward signal per §6.2.

        reward = hard_gate × (w1·goal_score + w2·parent_success + w3·global_outcome)
        """
        if not self.hard_gate:
            return 0.0
        return (w_goal * self.goal_score +
                w_parent * parent_success +
                w_global * global_outcome)


# ── Main verification entry point ────────────────────────────────────

def verify(spec: Spec, output: Any, env: Env | None = None,
           program_node: Node | None = None) -> VerificationResult:
    """Verify a value against a spec.

    Args:
        spec: The spec to verify against.
        output: The evaluated output value to check.
        env: Environment for evaluating constraint/goal expressions.
        program_node: Optional AST node of the program (for type checking).

    Returns:
        VerificationResult with all checks populated.
    """
    if env is None:
        env = standard_env()

    result = VerificationResult()

    # 1. Type match
    if spec.type_expr is not None:
        _check_type(spec, output, program_node, result)

    # 2. Shape match
    if spec.shape_expr is not None:
        _check_shape(spec, output, result)

    # 3. Constraint satisfaction
    if spec.constraints is not None:
        _check_constraints(spec, output, env, result)

    # 4. Goal satisfaction
    if spec.goal is not None:
        _check_goal(spec.goal, output, env, result)

    return result


def verify_fn(spec: Spec, program_fn: Closure | BuiltinFn,
              env: Env | None = None) -> VerificationResult:
    """Verify a function against a spec with examples/predicates.

    For specs with Level 0 goals (examples), this runs the function on
    each input and checks the output. For Level 2 goals (predicates),
    the function must be called by the caller and the output passed to verify().
    """
    if env is None:
        env = standard_env()

    result = VerificationResult()

    goal = spec.goal
    if goal is None:
        return result

    if isinstance(goal, GoalExamples):
        _check_goal_examples_fn(goal, program_fn, result)
    elif isinstance(goal, GoalAll):
        _check_goal_all_fn(goal, program_fn, env, result)
    else:
        # For non-example goals, delegate to the value-based verify
        # The caller needs to provide an output value
        result.goal_score = 0.0
        result.goal_details = {"error": "verify_fn requires GoalExamples; use verify() for other goal types"}

    # Also check type if present and we have example outputs
    if spec.type_expr is not None and isinstance(goal, GoalExamples) and goal.pairs:
        # Type-check using first example output
        first_input = _ast_to_value(goal.pairs[0][0])
        try:
            first_output = apply_fn(program_fn, [first_input])
            _check_type_value(spec, first_output, result)
        except Exception:
            pass  # type check is best-effort here

    return result


# ── Type checking ────────────────────────────────────────────────────

def _check_type(spec: Spec, output: Any, program_node: Node | None,
                result: VerificationResult):
    """Check that the output matches the spec's :type declaration."""
    expected_type_name = _type_name(spec.type_expr)
    if expected_type_name is None:
        return  # can't determine expected type

    actual_type_name = _value_type_name(output)

    if expected_type_name != actual_type_name:
        result.type_match = False
        result.type_error = f"expected type {expected_type_name}, got {actual_type_name}"


def _check_type_value(spec: Spec, output: Any, result: VerificationResult):
    """Type check just the output value against the spec's :type."""
    expected_type_name = _type_name(spec.type_expr)
    if expected_type_name is None:
        return

    actual_type_name = _value_type_name(output)
    if expected_type_name != actual_type_name:
        result.type_match = False
        result.type_error = f"expected type {expected_type_name}, got {actual_type_name}"


def _type_name(type_node: Node | None) -> str | None:
    """Extract a type name string from a type expression AST node."""
    if type_node is None:
        return None
    if isinstance(type_node, Symbol):
        return type_node.name
    return None


def _value_type_name(value: Any) -> str:
    """Get the SELPH type name for a Python runtime value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, (Closure, BuiltinFn)):
        return "function"
    if value is None:
        return "nil"
    return "unknown"


# ── Shape checking ───────────────────────────────────────────────────

def _check_shape(spec: Spec, output: Any, result: VerificationResult):
    """Check that the output matches the spec's :shape declaration.

    For now, shapes are checked on lists (length) and nested lists (dimensions).
    Tensor shape checking will integrate with JAX later.
    """
    from .ast import Shape
    shape_node = spec.shape_expr

    if isinstance(shape_node, Shape):
        if isinstance(output, list):
            _check_list_shape(output, shape_node.dims, result)
        else:
            result.shape_match = False
            result.shape_error = f"expected shaped value, got {type(output).__name__}"


def _check_list_shape(value: list, dims: tuple, result: VerificationResult):
    """Check that a nested list matches the given dimensions."""
    if not dims:
        return

    first_dim = dims[0]
    if isinstance(first_dim, int):
        if len(value) != first_dim:
            result.shape_match = False
            result.shape_error = f"dimension 0: expected {first_dim}, got {len(value)}"
            return
    # Symbolic dims and '?' always pass (they're constraints for the solver, not runtime checks)

    # Check inner dimensions recursively
    if len(dims) > 1 and value:
        for i, item in enumerate(value):
            if not isinstance(item, list):
                result.shape_match = False
                result.shape_error = f"expected nested list at dimension 1, got {type(item).__name__} at index {i}"
                return
            _check_list_shape(item, dims[1:], result)
            if not result.shape_match:
                return


# ── Constraint checking ──────────────────────────────────────────────

def _check_constraints(spec: Spec, output: Any, env: Env,
                       result: VerificationResult):
    """Check the spec's :constraints field against the output.

    Constraints can be:
      - A single predicate: (lambda (out) ...)
      - A list of named constraints: ((max-length 200) (non-empty) ...)
      - A function application: (constraint-fn output)
    """
    constraint_node = spec.constraints
    if constraint_node is None:
        return

    try:
        constraint_val = eval_node(constraint_node, env)
    except EvalError as e:
        # Constraint is an unevaluated expression — try known patterns
        if isinstance(constraint_node, List):
            _check_constraint_list(constraint_node, output, env, result)
            return
        result.constraint_match = False
        result.constraint_errors.append(f"failed to evaluate constraint: {e}")
        return

    # If the constraint evaluated to a callable, apply it to the output
    if isinstance(constraint_val, (Closure, BuiltinFn)):
        try:
            check = apply_fn(constraint_val, [output])
            if not check:
                result.constraint_match = False
                result.constraint_errors.append("constraint predicate returned false")
        except EvalError as e:
            result.constraint_match = False
            result.constraint_errors.append(f"constraint evaluation error: {e}")


def _check_constraint_list(node: List, output: Any, env: Env,
                           result: VerificationResult):
    """Check a list of named constraints like (max-length 200)."""
    head = node.elements[0] if node.elements else None

    if isinstance(head, Symbol):
        name = head.name
        args = node.elements[1:]

        if name == "max-length" and args:
            max_len = _ast_to_value(args[0])
            if isinstance(max_len, (int, float)):
                actual_len = len(output) if isinstance(output, (str, list)) else 0
                if actual_len > int(max_len):
                    result.constraint_match = False
                    result.constraint_errors.append(
                        f"max-length {int(max_len)} exceeded: actual length {actual_len}")
            return

        if name == "min-length" and args:
            min_len = _ast_to_value(args[0])
            if isinstance(min_len, (int, float)):
                actual_len = len(output) if isinstance(output, (str, list)) else 0
                if actual_len < int(min_len):
                    result.constraint_match = False
                    result.constraint_errors.append(
                        f"min-length {int(min_len)} not met: actual length {actual_len}")
            return

        if name == "non-empty":
            if isinstance(output, (str, list)) and len(output) == 0:
                result.constraint_match = False
                result.constraint_errors.append("non-empty constraint violated: value is empty")
            elif output is None:
                result.constraint_match = False
                result.constraint_errors.append("non-empty constraint violated: value is nil")
            return

        if name == "one-of" and args:
            allowed = [_ast_to_value(a) for a in args]
            if output not in allowed:
                result.constraint_match = False
                result.constraint_errors.append(
                    f"one-of constraint: {output!r} not in {allowed}")
            return

        if name == "matches" and args:
            import re
            pattern = _ast_to_value(args[0])
            if isinstance(pattern, str) and isinstance(output, str):
                if not re.fullmatch(pattern, output):
                    result.constraint_match = False
                    result.constraint_errors.append(
                        f"regex constraint: {output!r} does not match {pattern!r}")
            return

    # Fallback: try evaluating as a predicate applied to output
    try:
        constraint_fn = eval_node(node, env)
        if isinstance(constraint_fn, (Closure, BuiltinFn)):
            check = apply_fn(constraint_fn, [output])
            if not check:
                result.constraint_match = False
                result.constraint_errors.append("constraint predicate returned false")
    except EvalError as e:
        result.constraint_match = False
        result.constraint_errors.append(f"unknown constraint: {e}")


# ── Goal checking ────────────────────────────────────────────────────

def _check_goal(goal: Node, output: Any, env: Env,
                result: VerificationResult):
    """Check goal satisfaction. Dispatches by goal level."""

    if isinstance(goal, GoalExamples):
        _check_goal_examples_value(goal, output, result)

    elif isinstance(goal, GoalPattern):
        _check_goal_pattern(goal, output, env, result)

    elif isinstance(goal, GoalSatisfy):
        _check_goal_satisfy(goal, output, env, result)

    elif isinstance(goal, GoalAll):
        _check_goal_all(goal, output, env, result)

    else:
        # Level 3+ goals can't be verified mechanically
        result.goal_score = 0.0
        result.goal_details = {
            "level": "unsupported",
            "message": f"goal type {type(goal).__name__} requires model-as-judge"
        }


# --- Level 0: Examples ---

def _check_goal_examples_value(goal: GoalExamples, output: Any,
                               result: VerificationResult):
    """Level 0: For a value output, check if it matches any expected output.

    This is used when verifying a single output value (not a function).
    Checks if the output matches any of the expected outputs from the examples.
    """
    expected_outputs = [_ast_to_value(pair[1]) for pair in goal.pairs]
    if output in expected_outputs:
        result.goal_score = 1.0
        result.goal_details = {"level": 0, "matched": True}
    else:
        result.goal_score = 0.0
        result.goal_details = {
            "level": 0,
            "matched": False,
            "output": output,
            "expected_one_of": expected_outputs,
        }


def _check_goal_examples_fn(goal: GoalExamples, program_fn, result: VerificationResult):
    """Level 0: Run a function on each input example and check outputs."""
    total = len(goal.pairs)
    passed = 0
    failures = []

    for input_node, expected_node in goal.pairs:
        input_val = _ast_to_value(input_node)
        expected_val = _ast_to_value(expected_node)

        try:
            actual = apply_fn(program_fn, [input_val])
        except Exception as e:
            failures.append({
                "input": input_val,
                "expected": expected_val,
                "error": str(e),
            })
            continue

        if actual == expected_val:
            passed += 1
        else:
            failures.append({
                "input": input_val,
                "expected": expected_val,
                "actual": actual,
            })

    result.goal_score = passed / total if total > 0 else 1.0
    result.goal_details = {
        "level": 0,
        "total": total,
        "passed": passed,
        "failures": failures,
    }


# --- Level 1: Patterns ---

def _check_goal_pattern(goal: GoalPattern, output: Any, env: Env,
                        result: VerificationResult):
    """Level 1: Check output against a structural pattern.

    Pattern is an S-expression with named constraints, e.g.:
      (word :length 5 :starts-with "h")
    """
    pattern = goal.pattern

    if not isinstance(pattern, List) or not pattern.elements:
        result.goal_score = 0.0
        result.goal_details = {"level": 1, "error": "pattern must be a non-empty list"}
        return

    checks_passed = 0
    checks_total = 0
    errors = []

    # Parse keyword-value pairs from the pattern
    i = 1  # skip the pattern type name (e.g., "word")
    elements = pattern.elements
    while i < len(elements):
        if isinstance(elements[i], Symbol) and elements[i].name.startswith(":"):
            constraint_name = elements[i].name
            if i + 1 < len(elements):
                constraint_val = _ast_to_value(elements[i + 1])
                checks_total += 1

                if constraint_name == ":length":
                    if isinstance(output, (str, list)):
                        if len(output) == int(constraint_val):
                            checks_passed += 1
                        else:
                            errors.append(f"length: expected {int(constraint_val)}, got {len(output)}")
                    else:
                        errors.append(f"length check requires string or list, got {type(output).__name__}")

                elif constraint_name == ":starts-with":
                    if isinstance(output, str) and output.startswith(str(constraint_val)):
                        checks_passed += 1
                    else:
                        errors.append(f"starts-with: expected prefix {constraint_val!r}")

                elif constraint_name == ":ends-with":
                    if isinstance(output, str) and output.endswith(str(constraint_val)):
                        checks_passed += 1
                    else:
                        errors.append(f"ends-with: expected suffix {constraint_val!r}")

                elif constraint_name == ":contains":
                    if isinstance(output, str) and str(constraint_val) in output:
                        checks_passed += 1
                    elif isinstance(output, list) and constraint_val in output:
                        checks_passed += 1
                    else:
                        errors.append(f"contains: {constraint_val!r} not found")

                elif constraint_name == ":min":
                    if isinstance(output, (int, float)) and output >= constraint_val:
                        checks_passed += 1
                    else:
                        errors.append(f"min: expected >= {constraint_val}, got {output}")

                elif constraint_name == ":max":
                    if isinstance(output, (int, float)) and output <= constraint_val:
                        checks_passed += 1
                    else:
                        errors.append(f"max: expected <= {constraint_val}, got {output}")

                elif constraint_name == ":type":
                    expected = str(constraint_val)
                    actual = _value_type_name(output)
                    if expected == actual:
                        checks_passed += 1
                    else:
                        errors.append(f"type: expected {expected}, got {actual}")

                else:
                    # Unknown constraint — skip
                    checks_total -= 1

                i += 2
            else:
                i += 1
        else:
            i += 1

    result.goal_score = checks_passed / checks_total if checks_total > 0 else 1.0
    result.goal_details = {
        "level": 1,
        "checks_total": checks_total,
        "checks_passed": checks_passed,
        "errors": errors,
    }


# --- Level 2: Satisfy (predicates) ---

def _check_goal_satisfy(goal: GoalSatisfy, output: Any, env: Env,
                        result: VerificationResult):
    """Level 2: Evaluate the predicate lambda on the output."""
    try:
        predicate = eval_node(goal.predicate, env)
    except EvalError as e:
        result.goal_score = 0.0
        result.goal_details = {"level": 2, "error": f"failed to evaluate predicate: {e}"}
        return

    if not isinstance(predicate, (Closure, BuiltinFn)):
        result.goal_score = 0.0
        result.goal_details = {"level": 2, "error": f"predicate is not callable: {predicate!r}"}
        return

    try:
        check = apply_fn(predicate, [output])
    except EvalError as e:
        result.goal_score = 0.0
        result.goal_details = {"level": 2, "error": f"predicate raised error: {e}"}
        return

    if check:
        result.goal_score = 1.0
    else:
        result.goal_score = 0.0

    result.goal_details = {
        "level": 2,
        "predicate_result": check,
        "output": output,
    }


# --- Goal composition: (:all ...) ---

def _check_goal_all(goal: GoalAll, output: Any, env: Env,
                    result: VerificationResult):
    """Check all sub-goals, score is the average."""
    sub_results = []
    total_score = 0.0

    for sub_goal in goal.goals:
        sub_result = VerificationResult()
        _check_goal(sub_goal, output, env, sub_result)
        sub_results.append(sub_result)
        total_score += sub_result.goal_score

    n = len(goal.goals)
    result.goal_score = total_score / n if n > 0 else 1.0
    result.goal_details = {
        "level": "all",
        "sub_results": [
            {"score": sr.goal_score, "details": sr.goal_details}
            for sr in sub_results
        ],
    }


def _check_goal_all_fn(goal: GoalAll, program_fn, env: Env,
                       result: VerificationResult):
    """Check all sub-goals for a function program."""
    sub_results = []
    total_score = 0.0

    for sub_goal in goal.goals:
        sub_result = VerificationResult()
        if isinstance(sub_goal, GoalExamples):
            _check_goal_examples_fn(sub_goal, program_fn, sub_result)
        else:
            sub_result.goal_score = 0.0
            sub_result.goal_details = {"error": "non-example goals in :all require verify() with output"}
        sub_results.append(sub_result)
        total_score += sub_result.goal_score

    n = len(goal.goals)
    result.goal_score = total_score / n if n > 0 else 1.0
    result.goal_details = {
        "level": "all",
        "sub_results": [
            {"score": sr.goal_score, "details": sr.goal_details}
            for sr in sub_results
        ],
    }


# ── Utility ──────────────────────────────────────────────────────────

def _ast_to_value(node: Node) -> Any:
    """Convert a literal AST node to a Python runtime value."""
    if isinstance(node, Number):
        return node.value
    if isinstance(node, String):
        return node.value
    if isinstance(node, Bool):
        return node.value
    if isinstance(node, List):
        return [_ast_to_value(e) for e in node.elements]
    if isinstance(node, Symbol):
        return node.name
    return node


# ── Convenience: verify from source strings ──────────────────────────

def verify_from_source(spec_source: str, program_source: str,
                       input_val: Any = None) -> VerificationResult:
    """Parse a spec and program from source, run verification.

    If the program is a function and the spec has examples, runs verify_fn.
    If input_val is provided, evaluates the program with that input and
    verifies the output.
    """
    from .parser import parse
    from .eval import eval_node

    env = standard_env()
    spec_node = parse(spec_source)
    if not isinstance(spec_node, Spec):
        raise ValueError(f"expected a spec, got {type(spec_node).__name__}")

    program_node = parse(program_source)
    program_val = eval_node(program_node, env)

    if isinstance(program_val, (Closure, BuiltinFn)):
        if isinstance(spec_node.goal, GoalExamples):
            return verify_fn(spec_node, program_val, env)
        elif input_val is not None:
            output = apply_fn(program_val, [input_val])
            return verify(spec_node, output, env, program_node)
        else:
            return verify_fn(spec_node, program_val, env)
    else:
        return verify(spec_node, program_val, env, program_node)
