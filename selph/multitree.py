"""Multi-tree synthesis for SELPH.

Allows the synthesizer to search across multiple arbitrary namespace trees
simultaneously. Each tree contributes callable components to the search
space, with paths preserved for traceability.

A program synthesized from multiple trees might look like:
  (lambda (x) (ns-get math-lib "double" ((ns-get str-lib "clean") x)))

But for efficiency, we resolve namespace lookups at synthesis time
and generate direct calls:
  (lambda (x) (math.double (str.clean x)))

Usage:
  result = multi_synthesize(
      spec,
      trees=[ops_tree, data_tree, cache_tree],
      tree_names=["ops", "data", "cache"],
  )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .ast import Node, Symbol, Number, String, List, Spec, GoalExamples
from .eval import (
    Namespace, Closure, BuiltinFn, Env, standard_env,
    eval_node, apply_fn, EvalError,
)
from .synthesize import (
    synthesize, SynthesisResult, Component, _default_components,
    _infer_literal_type, make_validation_fn,
)
from .types import Type, TFn, TNum, TStr, TBool, TList, TVar


# ── Extract components from a Namespace ──────────────────────────────

@dataclass
class TreeComponent(Component):
    """A component extracted from a namespace tree.

    Carries the source tree name and path for traceability.
    """
    tree_name: str = ""
    path: list[str] = field(default_factory=list)


def extract_components(ns: Namespace, tree_name: str = "",
                       prefix: str = "", env: Env | None = None,
                       max_depth: int = 5) -> list[TreeComponent]:
    """Extract callable components from a namespace tree.

    Recursively walks the namespace. For each callable leaf (Closure,
    BuiltinFn), creates a Component with the appropriate type signature.
    For non-callable leaves (numbers, strings), creates constant components.

    The component's node is a Symbol that will be registered in the
    eval environment so it can be called directly.
    """
    if env is None:
        env = standard_env()

    components = []
    _extract_recursive(ns, tree_name, prefix, env, components, 0, max_depth)
    return components


def _extract_recursive(ns: Namespace, tree_name: str, prefix: str,
                       env: Env, components: list, depth: int, max_depth: int):
    """Recursively extract components from a namespace."""
    if depth > max_depth:
        return

    for key, val in ns.items():
        path_str = f"{prefix}.{key}" if prefix else key
        full_name = f"{tree_name}.{path_str}" if tree_name else path_str
        path = path_str.split(".")

        if isinstance(val, Namespace):
            _extract_recursive(val, tree_name, path_str, env, components, depth + 1, max_depth)

        elif isinstance(val, (Closure, BuiltinFn)):
            # Infer type from the callable
            comp_type = _infer_callable_type(val, env)
            if comp_type is not None:
                # Register in env so the synthesizer can call it by name
                env.define(full_name, val)
                components.append(TreeComponent(
                    name=full_name,
                    node=Symbol(full_name),
                    type=comp_type,
                    arity=len(comp_type.params) if isinstance(comp_type, TFn) else 0,
                    tree_name=tree_name,
                    path=path,
                ))

        elif isinstance(val, (int, float)):
            components.append(TreeComponent(
                name=full_name,
                node=Number(float(val)),
                type=TNum(),
                arity=0,
                tree_name=tree_name,
                path=path,
            ))

        elif isinstance(val, str):
            components.append(TreeComponent(
                name=full_name,
                node=String(val),
                type=TStr(),
                arity=0,
                tree_name=tree_name,
                path=path,
            ))

        elif isinstance(val, bool):
            components.append(TreeComponent(
                name=full_name,
                node=Symbol("true") if val else Symbol("false"),
                type=TBool(),
                arity=0,
                tree_name=tree_name,
                path=path,
            ))


def _infer_callable_type(fn: Closure | BuiltinFn, env: Env) -> TFn | None:
    """Infer the type signature of a callable by probing it.

    Tries calling with sample values to determine input/output types.
    Returns None if type can't be determined.
    """
    test_inputs = [
        (0.0, TNum()),
        ("", TStr()),
        (True, TBool()),
    ]

    if isinstance(fn, Closure):
        arity = len(fn.params)
    elif isinstance(fn, BuiltinFn):
        # Try to determine arity by calling with increasing args
        arity = _probe_arity(fn)
        if arity is None:
            return None
    else:
        return None

    if arity == 0:
        # Nullary function — try calling with no args
        try:
            result = apply_fn(fn, [])
            ret_type = _value_to_type(result)
            return TFn((), ret_type)
        except Exception:
            return None

    # Try each input type for the first parameter
    for test_val, test_type in test_inputs:
        try:
            args = [test_val] * arity
            result = apply_fn(fn, args)
            ret_type = _value_to_type(result)
            param_types = tuple([test_type] * arity)
            return TFn(param_types, ret_type)
        except Exception:
            continue

    return None


def _probe_arity(fn: BuiltinFn) -> int | None:
    """Determine the arity of a builtin by trying different arg counts."""
    for arity in range(4):
        try:
            args = [0.0] * arity
            apply_fn(fn, args)
            return arity
        except TypeError:
            continue
        except Exception:
            return arity  # right arity but wrong type
    return None


def _value_to_type(val: Any) -> Type:
    """Infer the SELPH type of a Python value."""
    if isinstance(val, bool):
        return TBool()
    if isinstance(val, (int, float)):
        return TNum()
    if isinstance(val, str):
        return TStr()
    if isinstance(val, list):
        if val and isinstance(val[0], str):
            return TList(TStr())
        return TList(TNum())
    if isinstance(val, Namespace):
        return TVar(0)
    return TVar(0)


# ── Multi-tree synthesis ─────────────────────────────────────────────

def multi_synthesize(
    spec: Spec,
    trees: list[Namespace],
    tree_names: list[str] | None = None,
    max_depth: int = 2,
    max_candidates: int = 50000,
    env: Env | None = None,
    enable_if: bool = False,
    include_builtins: bool = True,
) -> SynthesisResult:
    """Synthesize a program searching across multiple namespace trees.

    Each tree contributes its callable entries as synthesis components.
    Trees can be arbitrary — ops libraries, data stores, cached results,
    domain-specific knowledge, etc.

    Args:
        spec: The spec to satisfy.
        trees: List of Namespace values to search across.
        tree_names: Optional names for each tree (for traceability).
        max_depth: Maximum AST depth.
        max_candidates: Search budget.
        env: Eval environment (trees' callables will be registered here).
        enable_if: Whether to generate if-expressions.
        include_builtins: Whether to include default builtin components.
    """
    if env is None:
        env = standard_env()

    if tree_names is None:
        tree_names = [f"tree{i}" for i in range(len(trees))]

    # Extract components from all trees
    all_tree_components: list[Component] = []
    for ns, name in zip(trees, tree_names):
        comps = extract_components(ns, tree_name=name, env=env)
        all_tree_components.extend(comps)

    # Run synthesis with tree components as extra_components
    return synthesize(
        spec,
        max_depth=max_depth,
        max_candidates=max_candidates,
        extra_components=all_tree_components if all_tree_components else None,
        env=env,
        enable_if=enable_if,
    )


# ── Utilities ────────────────────────────────────────────────────────

def namespace_to_components(ns: Namespace, name: str = "",
                            env: Env | None = None) -> list[Component]:
    """Convenience: extract components from a single namespace."""
    if env is None:
        env = standard_env()
    return extract_components(ns, tree_name=name, env=env)


def trace_solution(result: SynthesisResult, trees: list[Namespace],
                   tree_names: list[str]) -> dict[str, list[str]]:
    """Trace which trees a synthesized solution draws from.

    Returns {tree_name: [paths_used]} showing which entries from
    each tree appear in the solution.
    """
    if not result.found or result.program is None:
        return {}

    usage = {name: [] for name in tree_names}

    def _scan(node: Node):
        if isinstance(node, Symbol):
            for name in tree_names:
                if node.name.startswith(name + "."):
                    path = node.name[len(name) + 1:]
                    usage[name].append(path)
        if isinstance(node, List):
            for e in node.elements:
                _scan(e)

    _scan(result.program)

    # Remove empty trees
    return {k: v for k, v in usage.items() if v}
