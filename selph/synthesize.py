"""Bottom-up enumerative program synthesizer for SELPH.

Given a spec, searches for the smallest program that satisfies it.
The type system prunes the search space: at each step, only type-compatible
compositions are considered.

This is the non-neural baseline that validates the spec system works as
a search objective (§14 step 5). If enumerative search with type pruning
can solve Stage 0-1 tasks efficiently, the constrained neural generator will too.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .ast import Node, Symbol, Number, String, Bool, List, Spec, GoalExamples
from .eval import (
    eval_node, apply_fn, standard_env, Env, Closure, BuiltinFn, EvalError,
)
from .verify import verify, verify_fn, VerificationResult
from .types import (
    TNum, TStr, TBool, TList, TFn, TVar, TNil, Type,
    InferenceState, TypeEnv, standard_type_env,
    typecheck, TypeCheckError, fresh_builtin_type, _POLYMORPHIC, _builtin_types,
)


# ── Synthesis result ─────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    """Result of a synthesis search."""
    program: Node | None = None
    source: str | None = None
    verification: VerificationResult | None = None
    candidates_explored: int = 0
    candidates_type_pruned: int = 0
    found: bool = False
    fitness_score: float | None = None  # for optimization goals


# ── Component library ────────────────────────────────────────────────

_component_counter = 0

@dataclass
class Component:
    """A named primitive or constant available for synthesis."""
    name: str
    node: Node
    type: Type  # concrete type signature (no TVars for monomorphic)
    arity: int  # number of arguments (0 for constants/values)
    is_polymorphic: bool = False
    insertion_order: int = -1  # set on creation; higher = newer

    def __post_init__(self):
        global _component_counter
        if self.insertion_order == -1:
            self.insertion_order = _component_counter
            _component_counter += 1

    def to_namespace(self):
        """Convert to a SELPH Namespace so programs can inspect this component."""
        from .eval import Namespace
        entries = {
            "name": self.name,
            "arity": float(self.arity),
            "insertion-order": float(self.insertion_order),
            "return-type": repr(self.type.ret) if isinstance(self.type, TFn) else repr(self.type),
        }
        if isinstance(self.type, TFn) and self.type.params:
            entries["input-type"] = repr(self.type.params[0])
            entries["param-count"] = float(len(self.type.params))
        return Namespace(entries, name=self.name)


def _default_components(target_type: Type | None = None) -> list[Component]:
    """Build the default component library for synthesis.

    Filters to components relevant to the target type to keep the
    search space manageable.
    """
    components = []
    N = TNum()
    S = TStr()
    B = TBool()

    # --- Constants ---
    for v in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, -1.0, -2.0]:
        components.append(Component(
            name=str(int(v)) if v == int(v) else str(v),
            node=Number(v),
            type=N,
            arity=0,
        ))

    components.append(Component(name='""', node=String(""), type=S, arity=0))
    components.append(Component(name="true", node=Bool(True), type=B, arity=0))
    components.append(Component(name="false", node=Bool(False), type=B, arity=0))

    # --- Monomorphic functions ---
    mono_sigs = _builtin_types()
    # Skip some that are aliases or rarely useful in synthesis
    # Skip operator aliases (we keep the word forms: add, subtract, etc.)
    # but include multiply/divide since they have no word-form alias
    skip = {"nil", "+", "-", "%"}

    for name, typ in mono_sigs.items():
        if name in skip:
            continue
        if isinstance(typ, TFn):
            components.append(Component(
                name=name,
                node=Symbol(name),
                type=typ,
                arity=len(typ.params),
            ))

    # --- Key polymorphic functions (instantiated for target type) ---
    # We add specific instantiations rather than full polymorphism
    # to keep the search space finite

    # String list operations (common in Stage 0-1)
    components.extend([
        Component("identity", Symbol("identity"), TFn((S,), S), 1),
        Component("identity", Symbol("identity"), TFn((N,), N), 1),
        Component("head", Symbol("head"), TFn((TList(S),), S), 1),
        Component("tail", Symbol("tail"), TFn((TList(S),), TList(S)), 1),
        Component("reverse", Symbol("reverse"), TFn((TList(S),), TList(S)), 1),
        Component("sort", Symbol("sort"), TFn((TList(S),), TList(S)), 1),
        Component("length", Symbol("length"), TFn((TList(S),), N), 1),
        Component("head", Symbol("head"), TFn((TList(N),), N), 1),
        Component("reverse", Symbol("reverse"), TFn((TList(N),), TList(N)), 1),
        Component("sort", Symbol("sort"), TFn((TList(N),), TList(N)), 1),
    ])

    return components


# ── Type compatibility ───────────────────────────────────────────────

def types_compatible(a: Type, b: Type) -> bool:
    """Check if two types are compatible (could unify) without side effects.

    This is a lightweight check for pruning — not full unification.
    """
    if isinstance(a, TVar) or isinstance(b, TVar):
        return True

    if type(a) != type(b):
        return False

    if isinstance(a, TList) and isinstance(b, TList):
        return types_compatible(a.elem, b.elem)

    if isinstance(a, TFn) and isinstance(b, TFn):
        if len(a.params) != len(b.params) and not a.variadic and not b.variadic:
            return False
        # Check return type compatibility
        return types_compatible(a.ret, b.ret)

    # Same concrete type
    return True


def return_type(t: Type) -> Type | None:
    """Get the return type of a function type, or None if not a function."""
    if isinstance(t, TFn):
        return t.ret
    return None


# ── Bottom-up enumeration ────────────────────────────────────────────

def synthesize(spec: Spec, max_depth: int = 3, max_candidates: int = 50000,
               extra_components: list[Component] | None = None,
               input_names: list[str] | None = None,
               input_types: list[Type] | None = None,
               constants: list[tuple[Any, Type]] | None = None,
               env: Env | None = None,
               enable_if: bool = False,
               validation_fn: Any = None,
               heuristic: Any = None,
               ) -> SynthesisResult:
    """Search for a program satisfying the spec.

    For specs with Level 0 goals (examples), synthesizes a function
    (lambda (input) body) that maps inputs to outputs.

    For specs with Level 1-2 goals, synthesizes a value.

    Args:
        spec: The spec to satisfy.
        max_depth: Maximum AST depth to search.
        max_candidates: Stop after this many candidates.
        extra_components: Additional components beyond defaults.
        input_names: Names for function parameters (default: ["x"]).
        input_types: Types for function parameters (inferred from examples if not given).
        constants: Extra constant values to include as (value, type) pairs.
    """
    from .ast import GoalMinimize, GoalMaximize

    result = SynthesisResult()
    goal = spec.goal

    # Optimization goals: minimize/maximize a fitness function
    if isinstance(goal, (GoalMinimize, GoalMaximize)):
        if env is None:
            env = standard_env()
        return _synthesize_optimize(spec, goal, max_depth, max_candidates,
                                     extra_components, env, heuristic, result)

    # Determine if we're synthesizing a function or a value
    if isinstance(goal, GoalExamples):
        return _synthesize_function(spec, goal, max_depth, max_candidates,
                                     extra_components, input_names, input_types,
                                     constants, env, enable_if, validation_fn,
                                     heuristic, result)
    else:
        return _synthesize_value(spec, max_depth, max_candidates,
                                  extra_components, constants, env, result)


def _synthesize_function(spec: Spec, goal: GoalExamples,
                         max_depth: int, max_candidates: int,
                         extra_components: list[Component] | None,
                         input_names: list[str] | None,
                         input_types: list[Type] | None,
                         constants: list[tuple[Any, Type]] | None,
                         env: Env | None,
                         enable_if: bool,
                         validation_fn: Any,
                         heuristic: Any,
                         result: SynthesisResult) -> SynthesisResult:
    """Synthesize a function from input/output examples."""

    # Infer input/output types from examples
    if not goal.pairs:
        return result

    first_in, first_out = goal.pairs[0]
    in_type = input_types[0] if input_types else _infer_literal_type(first_in)
    out_type = _infer_literal_type(first_out)

    if input_names is None:
        input_names = ["x"]

    # Build the component library
    components = _default_components(out_type)
    if extra_components:
        components.extend(extra_components)

    # Add input variable as a component
    for name, typ in zip(input_names, input_types or [in_type]):
        components.append(Component(
            name=name,
            node=Symbol(name),
            type=typ,
            arity=0,
        ))

    # Add constants extracted from examples
    example_constants = _extract_constants(goal)
    for val, typ in example_constants:
        node = _value_to_node(val)
        if node is not None:
            components.append(Component(
                name=repr(node),
                node=node,
                type=typ,
                arity=0,
            ))

    if constants:
        for val, typ in constants:
            node = _value_to_node(val)
            if node is not None:
                components.append(Component(
                    name=repr(node),
                    node=node,
                    type=typ,
                    arity=0,
                ))

    # Build the evaluation environment
    if env is None:
        env = standard_env()

    # Build namespace tree for scoped component access
    from .namespace import build_namespace, scope_for_context
    ns_tree = build_namespace(components)

    # Get scoped components: those relevant to the output type.
    # Include components for the original input type AND for intermediate
    # types that depth-1 compositions might produce (e.g., string->number
    # components at depth 1 mean we need number->X components at depth 2).
    scoped_components = scope_for_context(ns_tree, target_output=out_type, input_type=in_type)
    scoped_names = {c.name for c in scoped_components}

    # Add components for intermediate types (depth-2 inputs = depth-1 outputs)
    intermediate_types = set()
    for comp in scoped_components:
        if isinstance(comp.type, TFn):
            ret_key = type(comp.type.ret).__name__
            intermediate_types.add(comp.type.ret)
    # Also include out_type as an intermediate (depth-1 output feeding depth-2)
    intermediate_types.add(out_type)
    if in_type != out_type:
        intermediate_types.add(in_type)

    for itype in intermediate_types:
        for comp in scope_for_context(ns_tree, target_output=out_type, input_type=itype):
            if comp.name not in scoped_names:
                scoped_components.append(comp)
                scoped_names.add(comp.name)
        # Also include components that take the intermediate type and return out_type
        for comp in scope_for_context(ns_tree, target_output=None, input_type=itype):
            if comp.name not in scoped_names:
                scoped_components.append(comp)
                scoped_names.add(comp.name)

    # When if-expressions are enabled, also include bool-returning components
    if enable_if:
        for itype in intermediate_types:
            bool_scope = scope_for_context(ns_tree, target_output=TBool(), input_type=itype)
            for comp in bool_scope:
                if comp.name not in scoped_names:
                    scoped_components.append(comp)
                    scoped_names.add(comp.name)
    # Always include constants and the input variable
    scoped_names = {c.name for c in scoped_components}
    for comp in components:
        if comp.arity == 0 and comp.name not in scoped_names:
            scoped_components.append(comp)

    # Sort components for search ordering.
    if heuristic is not None:
        # Use the SELPH heuristic function: (heuristic component-ns) -> score
        # Higher score = tried first.
        def _selph_priority(c):
            try:
                ns = c.to_namespace()
                score = apply_fn(heuristic, [ns])
                return -float(score)  # negate for ascending sort
            except Exception:
                return 0.0
        scoped_components.sort(key=_selph_priority)
    elif extra_components:
        # Default: output type match first, then recency
        def _component_priority(c):
            ret_matches = 0
            if isinstance(c.type, TFn) and types_compatible(c.type.ret, out_type):
                ret_matches = 1
            return (-ret_matches, -c.insertion_order)
        scoped_components.sort(key=_component_priority)

    # Extract example inputs for observational equivalence pruning
    example_inputs = [_ast_to_eval_value(pair[0]) for pair in goal.pairs]

    # Layered bottom-up enumeration
    # Each depth level only generates NEW programs (not duplicating previous)
    all_programs: list[tuple[Node, Type]] = []
    prev_layer: list[tuple[Node, Type]] = []

    # Depth 0: all constants and variables
    for comp in scoped_components:
        if comp.arity == 0:
            prev_layer.append((comp.node, comp.type))
    all_programs.extend(prev_layer)

    # Track observational equivalence: programs that produce the same
    # outputs on example inputs are redundant
    seen_behaviors: set[tuple] = set()

    def _test_candidate(body_node, body_type):
        """Test a single candidate. Returns True if solution found."""
        if not types_compatible(body_type, out_type):
            result.candidates_type_pruned += 1
            return False

        result.candidates_explored += 1
        if result.candidates_explored > max_candidates:
            return None  # budget exhausted

        params_node = List(tuple(Symbol(n) for n in input_names))
        fn_node = List((Symbol("lambda"), params_node, body_node))

        try:
            fn_val = eval_node(fn_node, env)

            # Observational equivalence: check outputs on example inputs
            behavior = []
            for inp_val in example_inputs:
                try:
                    out = apply_fn(fn_val, [inp_val])
                    behavior.append(repr(out))
                except Exception:
                    behavior.append("ERR")
            behavior_key = tuple(behavior)

            if behavior_key in seen_behaviors:
                return False
            seen_behaviors.add(behavior_key)

            vr = verify_fn(spec, fn_val, env)
            if vr.goal_score == 1.0:
                # Held-out validation: check generalization if validator provided
                if validation_fn is not None:
                    if not validation_fn(fn_val):
                        return False  # passes examples but fails validation
                result.program = fn_node
                result.source = repr(fn_node)
                result.verification = vr
                result.found = True
                return True
        except (EvalError, Exception):
            pass
        return False

    # Test depth-0 candidates
    for body_node, body_type in prev_layer:
        check = _test_candidate(body_node, body_type)
        if check is True:
            return result
        if check is None:
            return result

    # Expand depth by depth
    for depth in range(1, max_depth + 1):
        new_layer = []
        # Dedup within each layer by behavior on example inputs
        # This prevents depth-2 explosion from redundant depth-1 programs
        layer_behaviors: set[tuple] = set()

        # Sort programs so those containing library symbols come first as args.
        # This prioritizes compositions involving recently-promoted primitives.
        if extra_components:
            extra_names = {c.name for c in extra_components}
            def _has_extra(node):
                if isinstance(node, Symbol) and node.name in extra_names:
                    return True
                if isinstance(node, List):
                    return any(_has_extra(e) for e in node.elements)
                return False
            prev_layer.sort(key=lambda nt: (0 if _has_extra(nt[0]) else 1))
            all_programs.sort(key=lambda nt: (0 if _has_extra(nt[0]) else 1))

        for body_node, body_type in _expand_layer_iter(prev_layer, all_programs, scoped_components):
            check = _test_candidate(body_node, body_type)
            if check is True:
                return result
            if check is None:
                return result

            # Layer-level dedup: evaluate on examples, skip if behavior seen
            params_node = List(tuple(Symbol(n) for n in input_names))
            fn_node = List((Symbol("lambda"), params_node, body_node))
            try:
                fn_val = eval_node(fn_node, env)
                beh = tuple(repr(apply_fn(fn_val, [inp])) for inp in example_inputs[:5])
                if beh in layer_behaviors:
                    continue
                layer_behaviors.add(beh)
            except Exception:
                pass

            new_layer.append((body_node, body_type))

        all_programs.extend(new_layer)

        # Try if-expressions using all programs accumulated so far
        if not enable_if:
            prev_layer = new_layer
            continue

        expected_outputs = [_ast_to_eval_value(pair[1]) for pair in goal.pairs]

        # Two-pass if generation:
        # Pass 1 (targeted): use expected outputs for fast matching
        # Pass 2 (exploratory): generate without expected outputs to build
        #   up the if-pool for nested ifs at the next depth
        if_layer = []

        # Targeted pass — find solutions
        for body_node, body_type in _generate_if_programs(
                all_programs, example_inputs, env, input_names,
                expected_outputs=expected_outputs):
            check = _test_candidate(body_node, body_type)
            if check is True:
                return result
            if check is None:
                return result
            if_layer.append((body_node, body_type))

        # Add if-expressions to the program pool for nested ifs
        all_programs.extend(if_layer)
        new_layer.extend(if_layer)
        prev_layer = new_layer

    return result


def _synthesize_value(spec: Spec, max_depth: int, max_candidates: int,
                      extra_components: list[Component] | None,
                      constants: list[tuple[Any, Type]] | None,
                      env: Env | None,
                      result: SynthesisResult) -> SynthesisResult:
    """Synthesize a value satisfying a spec (Level 1-2 goals)."""

    # Infer target type from spec
    target_type = None
    if spec.type_expr and isinstance(spec.type_expr, Symbol):
        target_type = {"string": TStr(), "number": TNum(), "bool": TBool()}.get(spec.type_expr.name)

    components = _default_components(target_type)
    if extra_components:
        components.extend(extra_components)

    if constants:
        for val, typ in constants:
            node = _value_to_node(val)
            if node is not None:
                components.append(Component(
                    name=repr(node),
                    node=node,
                    type=typ,
                    arity=0,
                ))

    if env is None:
        env = standard_env()

    all_programs: list[tuple[Node, Type]] = []
    prev_layer: list[tuple[Node, Type]] = []

    for comp in components:
        if comp.arity == 0:
            prev_layer.append((comp.node, comp.type))
    all_programs.extend(prev_layer)

    def _test_value(node, typ):
        if target_type and not types_compatible(typ, target_type):
            result.candidates_type_pruned += 1
            return False
        result.candidates_explored += 1
        if result.candidates_explored > max_candidates:
            return None
        try:
            val = eval_node(node, env)
            vr = verify(spec, val, env, node)
            if vr.passed:
                result.program = node
                result.source = repr(node)
                result.verification = vr
                result.found = True
                return True
        except (EvalError, Exception):
            pass
        return False

    for node, typ in prev_layer:
        check = _test_value(node, typ)
        if check is True:
            return result
        if check is None:
            return result

    for depth in range(1, max_depth + 1):
        new_layer = []
        for node, typ in _expand_layer_iter(prev_layer, all_programs, components):
            check = _test_value(node, typ)
            if check is True:
                return result
            if check is None:
                return result
            new_layer.append((node, typ))
        all_programs.extend(new_layer)
        prev_layer = new_layer

    return result


# ── Optimization synthesis ────────────────────────────────────────────

def _synthesize_optimize(spec: Spec, goal, max_depth: int,
                          max_candidates: int,
                          extra_components: list[Component] | None,
                          env: Env,
                          heuristic: Any,
                          result: SynthesisResult) -> SynthesisResult:
    """Synthesize a program that minimizes/maximizes a fitness function.

    Instead of stopping at the first correct program, evaluates every
    candidate with the fitness function and tracks the best score.

    The fitness function is a SELPH callable: (fitness candidate) -> number.
    For GoalMinimize, lower is better. For GoalMaximize, higher is better.
    """
    from .ast import GoalMinimize, GoalMaximize

    minimize = isinstance(goal, GoalMinimize)
    fitness_node = goal.fitness

    # Evaluate the fitness function
    fitness_fn = eval_node(fitness_node, env)

    # Build component library
    components = _default_components()
    if extra_components:
        components.extend(extra_components)

    # Add x as an input variable for function-valued optimization
    # (the fitness function may expect a callable)
    components.append(Component("x", Symbol("x"), TNum(), 0))

    if heuristic is not None:
        def _prio(c):
            try:
                return -float(apply_fn(heuristic, [c.to_namespace()]))
            except Exception:
                return 0.0
        components.sort(key=_prio)

    # Generate candidates and score each one
    best_score = float('inf') if minimize else float('-inf')
    best_program = None
    best_source = None

    all_programs: list[tuple[Node, Type]] = []
    prev_layer: list[tuple[Node, Type]] = []

    for comp in components:
        if comp.arity == 0:
            prev_layer.append((comp.node, comp.type))
    all_programs.extend(prev_layer)

    seen_behaviors: set[tuple] = set()

    def _test_opt_candidate(node, typ):
        nonlocal best_score, best_program, best_source

        result.candidates_explored += 1
        if result.candidates_explored > max_candidates:
            return None  # budget exhausted

        # Try two approaches:
        # 1. Evaluate directly (for value-level optimization)
        # 2. Wrap in lambda (for function-level optimization)
        score = None
        used_node = node

        # Approach 1: direct evaluation
        try:
            val = eval_node(node, env)
            score = float(apply_fn(fitness_fn, [val]))
        except Exception:
            pass

        # Approach 2: lambda wrapping (if direct eval failed or node has free vars)
        if score is None:
            try:
                fn_node = List((Symbol("lambda"), List((Symbol("x"),)), node))
                fn_val = eval_node(fn_node, env)
                score = float(apply_fn(fitness_fn, [fn_val]))
                used_node = fn_node
            except Exception:
                pass

        if score is not None:
            if minimize:
                if score < best_score:
                    best_score = score
                    best_program = used_node
                    best_source = repr(used_node)
            else:
                if score > best_score:
                    best_score = score
                    best_program = used_node
                    best_source = repr(used_node)
        return False  # never "found" — keep searching

    # Test depth 0
    for node, typ in prev_layer:
        check = _test_opt_candidate(node, typ)
        if check is None:
            break

    # Expand
    for depth in range(1, max_depth + 1):
        new_layer = []
        for node, typ in _expand_layer_iter(prev_layer, all_programs, components):
            check = _test_opt_candidate(node, typ)
            if check is None:
                break
            new_layer.append((node, typ))
        else:
            all_programs.extend(new_layer)
            prev_layer = new_layer
            continue
        break  # budget exhausted

    if best_program is not None:
        result.program = best_program
        result.source = best_source
        result.found = True
        result.fitness_score = best_score

    return result


# ── Program expansion ────────────────────────────────────────────────

def _expand_layer_iter(new_programs, all_programs, components):
    """Lazily yield (node, type) for the next depth layer.

    For arity-1: apply to new_programs only.
    For arity-2: at least one arg from new_programs.
    For if-expressions: combine a bool condition with two same-typed branches.
    Yields one at a time — never materializes the full list.
    """
    new_set = set(id(n) for n, _ in new_programs)

    for comp in components:
        if comp.arity == 0 or not isinstance(comp.type, TFn):
            continue

        fn_type = comp.type
        arity = len(fn_type.params)

        if arity == 1:
            for arg_node, arg_type in new_programs:
                if types_compatible(arg_type, fn_type.params[0]):
                    yield (List((comp.node, arg_node)), fn_type.ret)

        elif arity == 2:
            # Case 1: arg1 from new, arg2 from all
            for arg1_node, arg1_type in new_programs:
                if not types_compatible(arg1_type, fn_type.params[0]):
                    continue
                for arg2_node, arg2_type in all_programs:
                    if types_compatible(arg2_type, fn_type.params[1]):
                        yield (List((comp.node, arg1_node, arg2_node)), fn_type.ret)
            # Case 2: arg1 from old only, arg2 from new
            for arg1_node, arg1_type in all_programs:
                if id(arg1_node) in new_set:
                    continue
                if not types_compatible(arg1_type, fn_type.params[0]):
                    continue
                for arg2_node, arg2_type in new_programs:
                    if types_compatible(arg2_type, fn_type.params[1]):
                        yield (List((comp.node, arg1_node, arg2_node)), fn_type.ret)

    # Note: if-expressions are generated separately via _generate_if_programs()
    # to allow example-guided condition pruning.


def _generate_if_programs(all_programs: list[tuple[Node, Type]],
                          example_inputs: list[Any],
                          env: Env,
                          input_names: list[str],
                          expected_outputs: list[Any] | None = None):
    """Generate if-expressions guided by example inputs.

    Instead of blindly combining all (cond, then, else) triples, we:
    1. Evaluate each bool-typed program on the example inputs
    2. Keep only conditions that produce a non-trivial partition
       (not all-true or all-false)
    3. Group conditions by their partition pattern (dedup equivalent ones)
    4. For each unique partition, try (then, else) pairs
    """
    B = TBool()

    # Find all bool-typed programs that reference a variable
    bool_progs = [(n, t) for n, t in all_programs
                  if types_compatible(t, B) and _has_variable(n)]
    val_progs = [(n, t) for n, t in all_programs
                 if not types_compatible(t, B)]

    if not bool_progs or not val_progs or not example_inputs:
        return []

    # Evaluate each condition on example inputs to get its partition
    # Build lambda once per condition, not once per (condition, input)
    partitions: dict[tuple[bool, ...], Node] = {}
    for cond_node, _ in bool_progs:
        try:
            fn_node = List((Symbol("lambda"),
                           List(tuple(Symbol(n) for n in input_names)),
                           cond_node))
            fn = eval_node(fn_node, env)
        except Exception:
            continue

        pattern = []
        valid = True
        for inp in example_inputs:
            try:
                result = apply_fn(fn, [inp])
                pattern.append(bool(result))
            except Exception:
                valid = False
                break

        if not valid:
            continue

        key = tuple(pattern)
        if all(key) or not any(key):
            continue
        if key not in partitions:
            partitions[key] = cond_node

    if not partitions:
        return []

    # For each partition, find the branches by matching expected outputs.
    # Instead of trying all (then, else) pairs, we evaluate each branch
    # candidate on the example inputs and check if it matches the expected
    # outputs for its partition side. Only combine matching branches.
    from .library import tree_size
    small_vals = [(n, t) for n, t in val_progs if tree_size(n) <= 4]
    if not small_vals:
        small_vals = val_progs[:30]

    # We need to know the expected outputs to match branches.
    # These come from the spec, which is accessible via the caller.
    # Since we don't have the spec here, we pass expected_outputs
    # through from the caller. For now, fall back to brute-force
    # but with aggressive deduplication.

    # Pre-evaluate all branch candidates and index by partial outputs
    branch_data = []
    for val_node, val_type in small_vals:
        try:
            fn_node = List((Symbol("lambda"),
                           List(tuple(Symbol(n) for n in input_names)),
                           val_node))
            fn = eval_node(fn_node, env)
            outputs = tuple(apply_fn(fn, [inp]) for inp in example_inputs)
            branch_data.append((val_node, val_type, outputs))
        except Exception:
            continue

    if expected_outputs is not None:
        # FAST PATH: use expected outputs to directly match branches.
        # For a given partition, the then-branch must produce expected[i]
        # at true indices, and the else-branch at false indices.
        # Index branches by their outputs on each partition's indices.
        for partition, cond_node in partitions.items():
            true_indices = [i for i, v in enumerate(partition) if v]
            false_indices = [i for i, v in enumerate(partition) if not v]

            # What outputs do we need from then-branch and else-branch?
            needed_then = tuple(expected_outputs[i] for i in true_indices)
            needed_else = tuple(expected_outputs[i] for i in false_indices)

            # Index: partial_output_key -> list of (node, type)
            then_matches = []
            else_matches = []
            for val_node, val_type, outputs in branch_data:
                actual_then = tuple(outputs[i] for i in true_indices)
                actual_else = tuple(outputs[i] for i in false_indices)
                if actual_then == needed_then:
                    then_matches.append((val_node, val_type))
                if actual_else == needed_else:
                    else_matches.append((val_node, val_type))

            # Combine matches (small: typically 0-5 each)
            for then_node, then_type in then_matches:
                for else_node, else_type in else_matches:
                    if id(then_node) == id(else_node):
                        continue
                    if types_compatible(then_type, else_type):
                        yield (
                            List((Symbol("if"), cond_node, then_node, else_node)),
                            then_type,
                        )
    else:
        # SLOW PATH: no expected outputs, enumerate all pairs (deduped)
        seen_outputs = {}
        deduped = []
        for val_node, val_type, outputs in branch_data:
            if outputs not in seen_outputs:
                seen_outputs[outputs] = True
                deduped.append((val_node, val_type, outputs))

        for partition, cond_node in partitions.items():
            for then_node, then_type, then_out in deduped:
                for else_node, else_type, else_out in deduped:
                    if then_out == else_out:
                        continue
                    if types_compatible(then_type, else_type):
                        yield (
                            List((Symbol("if"), cond_node, then_node, else_node)),
                            then_type,
                        )


def _has_variable(node):
    """Check if an AST node references a non-builtin symbol (a variable)."""
    if isinstance(node, Symbol) and not _is_component_name(node.name):
        return True
    if isinstance(node, List):
        return any(_has_variable(e) for e in node.elements)
    return False


def _expand_if_exprs(new_programs, all_programs, new_set):
    """Generate (if cond then else) expressions.

    To keep search tractable:
    - Only use conditions that reference the input variable (contain a Symbol
      that isn't a builtin). Conditions like (> 1 2) are useless.
    - Then/else branches must have matching types but different AST structure.
    - Prioritize: conditions from new_programs, branches from all.
    """
    B = TBool()

    # Collect bool programs that reference a variable (useful conditions)
    def _has_variable(node):
        if isinstance(node, Symbol) and not _is_component_name(node.name):
            return True
        if isinstance(node, List):
            return any(_has_variable(e) for e in node.elements)
        return False

    bool_new = [(n, t) for n, t in new_programs
                if types_compatible(t, B) and _has_variable(n)]
    bool_old = [(n, t) for n, t in all_programs
                if types_compatible(t, B) and _has_variable(n) and id(n) not in new_set]

    val_all = [(n, t) for n, t in all_programs if not types_compatible(t, B)]
    val_new = [(n, t) for n, t in new_programs if not types_compatible(t, B)]

    if not (bool_new or bool_old) or not val_all:
        return

    # Case 1: condition from new
    for cond_node, _ in bool_new:
        for then_node, then_type in val_all:
            for else_node, else_type in val_all:
                if id(then_node) == id(else_node):
                    continue
                if types_compatible(then_type, else_type):
                    yield (List((Symbol("if"), cond_node, then_node, else_node)),
                           then_type)

    # Case 2: condition from old, at least one branch from new
    for cond_node, _ in bool_old:
        for then_node, then_type in val_new:
            for else_node, else_type in val_all:
                if id(then_node) == id(else_node):
                    continue
                if types_compatible(then_type, else_type):
                    yield (List((Symbol("if"), cond_node, then_node, else_node)),
                           then_type)
        val_old = [(n, t) for n, t in val_all if id(n) not in new_set]
        for then_node, then_type in val_old:
            for else_node, else_type in val_new:
                if id(then_node) == id(else_node):
                    continue
                if types_compatible(then_type, else_type):
                    yield (List((Symbol("if"), cond_node, then_node, else_node)),
                           then_type)


# Builtin/component names that shouldn't count as "variable references"
_COMPONENT_NAMES = {
    "add", "subtract", "multiply", "divide", "modulo", "abs", "negate",
    "min", "max", "floor", "ceil", "*", "/",
    "<", ">", "<=", ">=", "=", "!=",
    "not", "even", "odd", "and", "or",
    "string-upper", "string-lower", "string-reverse", "string-trim",
    "string-length", "string-contains", "string-split", "string-join",
    "string-starts-with", "string-ends-with", "string-replace",
    "substring", "char-at", "concat", "to-string", "to-number",
    "head", "tail", "length", "nth", "cons", "append", "reverse",
    "sort", "range", "empty?", "contains", "list",
    "map", "filter", "reduce", "compose", "pipe", "apply", "identity",
    "if", "let", "lambda", "true", "false",
}

def _is_component_name(name: str) -> bool:
    return name in _COMPONENT_NAMES


# ── Helpers ──────────────────────────────────────────────────────────

def _infer_literal_type(node: Node) -> Type:
    """Infer the type of a literal AST node."""
    if isinstance(node, Number):
        return TNum()
    if isinstance(node, String):
        return TStr()
    if isinstance(node, Bool):
        return TBool()
    if isinstance(node, List):
        if node.elements:
            elem_type = _infer_literal_type(node.elements[0])
            return TList(elem_type)
        return TList(TVar(0))
    return TVar(0)  # unknown


def _extract_constants(goal: GoalExamples) -> list[tuple[Any, Type]]:
    """Extract unique constant values from goal examples.

    These become additional components in the search space.
    """
    seen = set()
    constants = []

    for in_node, out_node in goal.pairs:
        for node in [in_node, out_node]:
            if isinstance(node, Number) and node.value not in seen:
                seen.add(node.value)
                constants.append((node.value, TNum()))
            elif isinstance(node, String) and node.value not in seen:
                seen.add(node.value)
                constants.append((node.value, TStr()))

    return constants


def _ast_to_eval_value(node: Node) -> Any:
    """Convert an AST literal to a Python value for evaluation."""
    if isinstance(node, Number):
        return node.value
    if isinstance(node, String):
        return node.value
    if isinstance(node, Bool):
        return node.value
    return None


def _value_to_node(val: Any) -> Node | None:
    """Convert a Python value to an AST node."""
    if isinstance(val, bool):
        return Bool(val)
    if isinstance(val, (int, float)):
        return Number(float(val))
    if isinstance(val, str):
        return String(val)
    return None


# ── Convenience ──────────────────────────────────────────────────────

def make_validation_fn(oracle_fn, test_inputs: list[Any]) -> callable:
    """Build a held-out validation function from a known oracle.

    The returned function takes a SELPH function value and returns True
    if it produces the same outputs as oracle_fn on all test_inputs.

    Usage:
        validator = make_validation_fn(
            lambda x: x.upper(),
            ["test1", "test2", "test3"]
        )
        result = synthesize(spec, validation_fn=validator)
    """
    def validate(fn_val):
        for inp in test_inputs:
            try:
                actual = apply_fn(fn_val, [inp])
                expected = oracle_fn(inp)
                if actual != expected:
                    return False
            except Exception:
                return False
        return True
    return validate


def synthesize_from_source(spec_source: str, **kwargs) -> SynthesisResult:
    """Parse a spec from source and synthesize a program."""
    from .parser import parse
    node = parse(spec_source)
    if not isinstance(node, Spec):
        raise ValueError(f"expected a spec, got {type(node).__name__}")
    return synthesize(node, **kwargs)
