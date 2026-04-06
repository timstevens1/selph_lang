"""Library extraction for SELPH (Loop 3).

Mines successful programs for recurring sub-expressions, extracts them
as named abstractions (macros), and produces new components for the
synthesizer. Follows the DreamCoder wake-sleep paradigm (§6.1 Loop 3).

The key operations:
  1. Sub-tree extraction: collect all sub-trees from a corpus of programs
  2. Anti-unification: find the most specific generalization of two trees
  3. Compression scoring: keep abstractions that compress the corpus
  4. Integration: produce defmacro nodes and Component entries
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Any
from .ast import Node, Symbol, Number, String, Bool, List
from .types import Type, TFn, TNum, TStr, TBool, TList, TVar, InferenceState
from .synthesize import Component
from .eval import Env, standard_env, eval_node, Closure, BuiltinFn


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class Abstraction:
    """An extracted library primitive."""
    name: str
    params: list[str]        # parameter names
    body: Node               # the macro body (with params as symbols)
    param_types: list[Type]  # inferred types for each param
    return_type: Type        # inferred return type
    frequency: int           # how many times the pattern occurred
    compression: float       # bits saved by using this abstraction

    @property
    def arity(self) -> int:
        return len(self.params)

    @property
    def type(self) -> Type:
        return TFn(tuple(self.param_types), self.return_type)

    def to_defmacro(self) -> Node:
        """Generate the (defmacro name (params...) body) AST node."""
        return List((
            Symbol("defmacro"),
            Symbol(self.name),
            List(tuple(Symbol(p) for p in self.params)),
            self.body,
        ))

    def to_component(self) -> Component:
        """Generate a Component for the synthesizer."""
        return Component(
            name=self.name,
            node=Symbol(self.name),
            type=self.type,
            arity=self.arity,
        )

    def __repr__(self):
        params_str = " ".join(self.params)
        return f"<abstraction {self.name} ({params_str}) freq={self.frequency} comp={self.compression:.1f}>"


# ── Sub-tree extraction ──────────────────────────────────────────────

def subtrees(node: Node) -> list[Node]:
    """Extract all sub-trees from an AST node, including the node itself."""
    result = [node]
    if isinstance(node, List):
        for elem in node.elements:
            result.extend(subtrees(elem))
    return result


def tree_size(node: Node) -> int:
    """Count the number of nodes in an AST."""
    if isinstance(node, List):
        return 1 + sum(tree_size(e) for e in node.elements)
    return 1


def tree_fingerprint(node: Node) -> str:
    """Structural fingerprint of a tree, ignoring variable names.

    Two trees with the same fingerprint have the same structure and
    the same primitive operations, differing only in leaf values/variables.
    """
    if isinstance(node, Number):
        return f"N({node.value})"
    if isinstance(node, String):
        return f"S({node.value})"
    if isinstance(node, Bool):
        return f"B({node.value})"
    if isinstance(node, Symbol):
        return f"V"  # all variables/symbols collapse to V
    if isinstance(node, List):
        if not node.elements:
            return "()"
        children = " ".join(tree_fingerprint(e) for e in node.elements)
        return f"({children})"
    return "?"


def tree_structural_fingerprint(node: Node) -> str:
    """Fingerprint preserving primitive names but abstracting over arguments.

    E.g., (string-upper x) and (string-upper y) have the same fingerprint,
    but (string-upper x) and (string-lower x) don't.
    """
    if isinstance(node, Number):
        return "NUM"
    if isinstance(node, String):
        return "STR"
    if isinstance(node, Bool):
        return "BOOL"
    if isinstance(node, Symbol):
        # Distinguish builtins from variables
        if _is_builtin(node.name):
            return node.name
        return "_"
    if isinstance(node, List):
        if not node.elements:
            return "()"
        children = " ".join(tree_structural_fingerprint(e) for e in node.elements)
        return f"({children})"
    return "?"


_BUILTINS = {
    "add", "+", "subtract", "-", "multiply", "*", "divide", "/", "modulo", "%",
    "abs", "negate", "floor", "ceil", "min", "max",
    "<", ">", "<=", ">=", "=", "!=",
    "not", "and", "or", "even", "odd",
    "concat", "string-length", "substring", "string-upper", "string-lower",
    "string-reverse", "string-contains", "string-split", "string-join",
    "string-starts-with", "string-ends-with", "string-replace", "string-trim",
    "char-at", "to-string", "to-number",
    "list", "cons", "head", "tail", "length", "nth", "append", "reverse",
    "range", "empty?", "contains", "sort",
    "map", "filter", "reduce", "compose", "pipe", "apply",
    "identity", "if", "let", "lambda", "defmacro", "define", "do", "quote",
}


def _is_builtin(name: str) -> bool:
    return name in _BUILTINS


# ── Anti-unification ─────────────────────────────────────────────────

def anti_unify(a: Node, b: Node) -> tuple[Node, dict[str, list[Node]]]:
    """Find the most specific generalization of two AST trees.

    Returns (pattern, substitutions) where pattern has fresh parameter
    symbols where a and b differ, and substitutions maps param names
    to the [a_value, b_value] that were replaced.

    Example:
        anti_unify((add x 1), (add y 2))
        -> ((add _p0 _p1), {"_p0": [x, y], "_p1": [1, 2]})

        anti_unify((string-upper x), (string-upper y))
        -> ((string-upper _p0), {"_p0": [x, y]})

        anti_unify((add x 1), (multiply x 1))
        -> (_p0, {"_p0": [(add x 1), (multiply x 1)]})
    """
    state = {"counter": 0}
    subs: dict[str, list[Node]] = {}

    def _au(a: Node, b: Node) -> Node:
        # Identical nodes unify to themselves
        if _nodes_equal(a, b):
            return a

        # Both are lists with same head -> recurse
        if (isinstance(a, List) and isinstance(b, List) and
                len(a.elements) == len(b.elements) and
                len(a.elements) > 0):
            # Check if heads match (both are the same builtin)
            head_a, head_b = a.elements[0], b.elements[0]
            if _nodes_equal(head_a, head_b):
                children = [head_a]
                for ca, cb in zip(a.elements[1:], b.elements[1:]):
                    children.append(_au(ca, cb))
                return List(tuple(children))

        # Different -> introduce parameter
        name = f"_p{state['counter']}"
        state["counter"] += 1
        subs[name] = [a, b]
        return Symbol(name)

    pattern = _au(a, b)
    return pattern, subs


def anti_unify_many(nodes: list[Node]) -> tuple[Node, list[str]]:
    """Anti-unify a list of AST nodes pairwise, returning the common pattern.

    Returns (pattern, param_names).
    """
    if len(nodes) == 0:
        return Symbol("_"), []
    if len(nodes) == 1:
        return nodes[0], []

    pattern = nodes[0]
    all_params = set()
    for other in nodes[1:]:
        pattern, subs = anti_unify(pattern, other)
        all_params.update(subs.keys())

    # Collect all parameter names in the final pattern
    params = sorted(_collect_params(pattern))
    return pattern, params


def _collect_params(node: Node) -> set[str]:
    """Collect all parameter symbols (_p0, _p1, ...) in a node."""
    if isinstance(node, Symbol) and node.name.startswith("_p"):
        return {node.name}
    if isinstance(node, List):
        result = set()
        for e in node.elements:
            result |= _collect_params(e)
        return result
    return set()


def register_abstractions(abstractions: list[Abstraction], env: Env | None = None) -> Env:
    """Register promoted abstractions in an eval environment.

    Each abstraction becomes a defmacro so that (abs_name arg) expands
    and evaluates correctly during synthesis.
    """
    if env is None:
        env = standard_env()
    for abs in abstractions:
        macro_node = abs.to_defmacro()
        eval_node(macro_node, env)
    return env


def _nodes_equal(a: Node, b: Node) -> bool:
    """Deep equality check for AST nodes."""
    if type(a) != type(b):
        return False
    if isinstance(a, Symbol):
        return a.name == b.name
    if isinstance(a, Number):
        return a.value == b.value
    if isinstance(a, String):
        return a.value == b.value
    if isinstance(a, Bool):
        return a.value == b.value
    if isinstance(a, List):
        if len(a.elements) != len(b.elements):
            return False
        return all(_nodes_equal(ea, eb) for ea, eb in zip(a.elements, b.elements))
    return False


# ── Frequency analysis ───────────────────────────────────────────────

def count_subtrees(corpus: list[Node], min_size: int = 2) -> Counter:
    """Count structural fingerprints of all sub-trees in a corpus.

    Only counts sub-trees of size >= min_size (atoms are too small
    to be useful abstractions).
    """
    counts = Counter()
    for program in corpus:
        seen_in_program = set()
        for st in subtrees(program):
            if tree_size(st) >= min_size:
                fp = tree_structural_fingerprint(st)
                if fp not in seen_in_program:
                    counts[fp] += 1
                    seen_in_program.add(fp)
    return counts


def find_common_patterns(corpus: list[Node], min_frequency: int = 2,
                         min_size: int = 2) -> list[tuple[str, int, list[Node]]]:
    """Find recurring sub-tree patterns in a corpus.

    Returns (fingerprint, count, example_nodes) sorted by frequency × size.
    """
    counts = count_subtrees(corpus, min_size)
    # Collect example nodes for each fingerprint
    examples: dict[str, list[Node]] = {}
    for program in corpus:
        for st in subtrees(program):
            if tree_size(st) >= min_size:
                fp = tree_structural_fingerprint(st)
                if counts[fp] >= min_frequency:
                    if fp not in examples:
                        examples[fp] = []
                    if len(examples[fp]) < 10:  # cap examples
                        examples[fp].append(st)

    results = []
    for fp, count in counts.items():
        if count >= min_frequency and fp in examples:
            results.append((fp, count, examples[fp]))

    # Sort by frequency × size (bigger, more frequent patterns are more valuable)
    results.sort(key=lambda x: x[1] * tree_size(x[2][0]) if x[2] else 0, reverse=True)
    return results


# ── Compression scoring ──────────────────────────────────────────────

def compression_score(pattern: Node, params: list[str],
                      corpus: list[Node]) -> float:
    """Score an abstraction by how much it compresses the corpus.

    Compression = (total_size_saved) - (abstraction_definition_cost)

    An abstraction saves `(pattern_size - 1 - n_params)` nodes each time
    it's used (replacing a deep tree with a single call + args).
    The cost is `pattern_size + n_params + 2` (the defmacro definition).

    Only worth keeping if compression > 0.
    """
    pattern_size = tree_size(pattern)
    n_params = len(params)

    # Nodes saved per use: we replace the subtree with (name arg1 arg2 ...)
    # which is 1 + n_params nodes, so savings = pattern_size - (1 + n_params)
    savings_per_use = pattern_size - (1 + n_params)
    if savings_per_use <= 0:
        return 0.0

    # Count how many times the pattern matches in the corpus
    fp = tree_structural_fingerprint(pattern)
    uses = 0
    for program in corpus:
        for st in subtrees(program):
            if tree_structural_fingerprint(st) == fp:
                uses += 1

    # Definition cost: (defmacro name (params) body)
    definition_cost = pattern_size + n_params + 3

    total_savings = savings_per_use * uses - definition_cost
    return max(0.0, float(total_savings))


# ── Extraction pipeline ──────────────────────────────────────────────

def extract_library(corpus: list[Node],
                    min_frequency: int = 2,
                    min_size: int = 2,
                    min_compression: float = 1.0,
                    max_abstractions: int = 20,
                    name_prefix: str = "lib") -> list[Abstraction]:
    """Extract a library of abstractions from a corpus of successful programs.

    This is the main Loop 3 entry point.

    Args:
        corpus: List of successful program ASTs.
        min_frequency: Minimum times a pattern must appear.
        min_size: Minimum sub-tree size to consider.
        min_compression: Minimum compression score to keep.
        max_abstractions: Maximum number of abstractions to extract.
        name_prefix: Prefix for generated names.

    Returns:
        List of Abstraction objects, sorted by compression score.
    """
    patterns = find_common_patterns(corpus, min_frequency, min_size)
    abstractions = []
    used_fingerprints = set()

    for fp, freq, examples in patterns:
        if len(abstractions) >= max_abstractions:
            break

        # Skip if we already have an abstraction covering this pattern
        if fp in used_fingerprints:
            continue

        # Anti-unify the examples to get a pattern with parameters
        pattern, params = anti_unify_many(examples[:5])

        # If anti-unification collapsed everything to a single param, skip
        if isinstance(pattern, Symbol) and pattern.name.startswith("_p"):
            continue

        # Score compression
        comp = compression_score(pattern, params, corpus)
        if comp < min_compression:
            continue

        # Infer types for the abstraction
        param_types, ret_type = _infer_abstraction_types(pattern, params)

        # Generate a name
        name = f"{name_prefix}_{len(abstractions)}"

        # Rename params to cleaner names
        clean_params = [f"a{i}" for i in range(len(params))]
        clean_body = _rename_params(pattern, dict(zip(params, clean_params)))

        abstraction = Abstraction(
            name=name,
            params=clean_params,
            body=clean_body,
            param_types=param_types,
            return_type=ret_type,
            frequency=freq,
            compression=comp,
        )
        abstractions.append(abstraction)
        used_fingerprints.add(fp)

    abstractions.sort(key=lambda a: a.compression, reverse=True)
    return abstractions


def _rename_params(node: Node, mapping: dict[str, str]) -> Node:
    """Rename parameter symbols in an AST."""
    if isinstance(node, Symbol) and node.name in mapping:
        return Symbol(mapping[node.name])
    if isinstance(node, List):
        return List(tuple(_rename_params(e, mapping) for e in node.elements))
    return node


def _infer_abstraction_types(pattern: Node, params: list[str]) -> tuple[list[Type], Type]:
    """Infer parameter and return types for an abstraction.

    Uses heuristics based on the pattern structure. Full type inference
    would require running the type checker on the pattern, which we'll
    do in a future version.
    """
    # Simple heuristic: look at what operations are applied to each param
    param_types = []
    for p in params:
        pt = _guess_param_type(pattern, p)
        param_types.append(pt)

    ret_type = _guess_return_type(pattern)
    return param_types, ret_type


def _guess_param_type(node: Node, param: str) -> Type:
    """Guess the type of a parameter based on how it's used in the pattern."""
    if isinstance(node, List) and len(node.elements) >= 2:
        head = node.elements[0]
        if isinstance(head, Symbol):
            fn_name = head.name
            for i, arg in enumerate(node.elements[1:]):
                if isinstance(arg, Symbol) and arg.name == param:
                    return _param_type_from_function(fn_name, i)
        # Recurse
        for elem in node.elements:
            t = _guess_param_type(elem, param)
            if not isinstance(t, TVar):
                return t
    return TVar(0)


def _param_type_from_function(fn_name: str, arg_index: int) -> Type:
    """Infer a parameter's type from the function it's passed to."""
    string_fns = {"string-upper", "string-lower", "string-reverse", "string-trim",
                  "string-length", "string-contains", "string-starts-with",
                  "string-ends-with", "string-split", "string-replace", "concat"}
    num_fns = {"add", "+", "subtract", "-", "multiply", "*", "divide", "/",
               "abs", "negate", "even", "odd", "floor", "ceil",
               "<", ">", "<=", ">=", "modulo", "%", "min", "max"}

    if fn_name in string_fns:
        return TStr()
    if fn_name in num_fns:
        return TNum()
    if fn_name in {"not", "and", "or"}:
        return TBool()
    return TVar(0)


def _guess_return_type(node: Node) -> Type:
    """Guess the return type of a pattern based on the outermost operation."""
    if isinstance(node, List) and node.elements:
        head = node.elements[0]
        if isinstance(head, Symbol):
            fn = head.name
            if fn in {"string-upper", "string-lower", "string-reverse", "string-trim",
                       "concat", "substring", "string-replace", "string-join",
                       "char-at", "to-string"}:
                return TStr()
            if fn in {"add", "+", "subtract", "-", "multiply", "*", "divide", "/",
                       "abs", "negate", "floor", "ceil", "string-length",
                       "modulo", "%", "min", "max", "length", "to-number"}:
                return TNum()
            if fn in {"<", ">", "<=", ">=", "=", "!=", "not", "and", "or",
                       "even", "odd", "string-contains", "string-starts-with",
                       "string-ends-with", "empty?", "contains"}:
                return TBool()
    if isinstance(node, Number):
        return TNum()
    if isinstance(node, String):
        return TStr()
    if isinstance(node, Bool):
        return TBool()
    return TVar(0)


# ── Promote solved programs as library primitives ─────────────────────

def promote_solved(solved: list[tuple[str, Node, Type, Type]],
                   name_prefix: str = "s0") -> list[Abstraction]:
    """Promote solved programs directly as library primitives.

    This is the simplest form of library learning: if the synthesizer
    solved (lambda (x) (string-upper x)) for the "upper" task, then
    "upper" becomes a new primitive with type (-> string string).

    This is what §11 means by "the library from stage N is frozen into
    the primitive set for stage N+1."

    Args:
        solved: List of (name, body_node, input_type, output_type).
            body_node is the lambda body (not the full lambda).

    Returns:
        List of Abstractions, one per solved program.
    """
    abstractions = []
    for name, body, in_type, out_type in solved:
        # The abstraction's body is the solved program body
        # with the input variable as the parameter
        params = _collect_free_vars(body)
        if not params:
            params = ["x"]  # constant function

        abs_name = f"{name_prefix}_{name}"
        abstractions.append(Abstraction(
            name=abs_name,
            params=params,
            body=body,
            param_types=[in_type] * len(params),
            return_type=out_type,
            frequency=1,
            compression=0.0,  # not from compression; from curriculum promotion
        ))

    return abstractions


def _collect_free_vars(node: Node) -> list[str]:
    """Collect free variable names in an AST (non-builtin symbols)."""
    if isinstance(node, Symbol):
        if not _is_builtin(node.name) and not node.name.startswith("_p"):
            return [node.name]
        return []
    if isinstance(node, List):
        seen = set()
        result = []
        for elem in node.elements:
            for v in _collect_free_vars(elem):
                if v not in seen:
                    seen.add(v)
                    result.append(v)
        return result
    return []


# ── Rewrite corpus with abstractions ─────────────────────────────────

def rewrite_with_abstraction(node: Node, abstraction: Abstraction) -> Node:
    """Rewrite an AST by replacing occurrences of the abstraction's pattern
    with calls to the abstraction.

    This is a simple structural match-and-replace. A production version
    would use e-graph matching for optimal rewriting.
    """
    target_fp = tree_structural_fingerprint(abstraction.body)

    def _rewrite(n: Node) -> Node:
        # Check if this node matches the abstraction's pattern
        if tree_structural_fingerprint(n) == target_fp:
            # Extract the arguments by matching against the pattern
            args = _extract_args(n, abstraction.body, abstraction.params)
            if args is not None:
                return List((Symbol(abstraction.name),) + tuple(args))

        # Recurse into lists
        if isinstance(n, List):
            return List(tuple(_rewrite(e) for e in n.elements))
        return n

    return _rewrite(node)


def _extract_args(node: Node, pattern: Node, params: list[str]) -> list[Node] | None:
    """Try to match a node against a pattern, extracting argument values.

    Returns a list of argument nodes (one per param) if the match succeeds,
    or None if it doesn't.
    """
    bindings: dict[str, Node] = {}

    def _match(n: Node, p: Node) -> bool:
        # Parameter slot: bind it
        if isinstance(p, Symbol) and p.name in params:
            if p.name in bindings:
                return _nodes_equal(n, bindings[p.name])
            bindings[p.name] = n
            return True

        # Must be structurally equal
        if type(n) != type(p):
            return False

        if isinstance(n, Symbol):
            return n.name == p.name
        if isinstance(n, Number):
            return n.value == p.value
        if isinstance(n, String):
            return n.value == p.value
        if isinstance(n, Bool):
            return n.value == p.value

        if isinstance(n, List):
            if len(n.elements) != len(p.elements):
                return False
            return all(_match(ne, pe) for ne, pe in zip(n.elements, p.elements))

        return False

    if _match(node, pattern):
        return [bindings.get(p, Symbol("_")) for p in params]
    return None


def rewrite_corpus(corpus: list[Node], abstractions: list[Abstraction]) -> list[Node]:
    """Rewrite an entire corpus using extracted abstractions."""
    result = corpus
    for abs in abstractions:
        result = [rewrite_with_abstraction(prog, abs) for prog in result]
    return result


# ── Library persistence ──────────────────────────────────────────────

def save_library(abstractions: list[Abstraction], path: str):
    """Save a library to a SELPH source file.

    Each abstraction is written as a defmacro, plus a comment with metadata.
    The file can be loaded back with load_library().
    """
    with open(path, "w") as f:
        f.write("; SELPH library — auto-generated\n")
        f.write(f"; {len(abstractions)} abstractions\n\n")
        for abs in abstractions:
            # Metadata comment
            param_types_str = ", ".join(repr(t) for t in abs.param_types)
            f.write(f"; {abs.name}: ({param_types_str}) -> {abs.return_type!r}"
                    f"  freq={abs.frequency} comp={abs.compression:.1f}\n")
            # The defmacro
            macro_node = abs.to_defmacro()
            f.write(f"{macro_node!r}\n\n")


def load_library(path: str, env: Env | None = None) -> list[Abstraction]:
    """Load a library from a SELPH source file.

    Parses the defmacro expressions, registers them in the eval env,
    and returns Abstraction objects.

    The file format is the output of save_library().
    """
    from .parser import parse_file
    from .types import TVar

    with open(path, "r") as f:
        source = f.read()

    if env is None:
        env = standard_env()

    nodes = parse_file(source)
    abstractions = []

    for node in nodes:
        # Each node should be a (defmacro name (params) body)
        if not isinstance(node, List) or len(node.elements) != 4:
            continue
        head = node.elements[0]
        if not isinstance(head, Symbol) or head.name != "defmacro":
            continue

        name = node.elements[1].name if isinstance(node.elements[1], Symbol) else str(node.elements[1])
        params_node = node.elements[2]
        params = []
        if isinstance(params_node, List):
            for p in params_node.elements:
                if isinstance(p, Symbol):
                    params.append(p.name)
        body = node.elements[3]

        # Register in env
        eval_node(node, env)

        # Create abstraction with unknown types (will be inferred on use)
        abs = Abstraction(
            name=name,
            params=params,
            body=body,
            param_types=[TVar(0)] * len(params),
            return_type=TVar(0),
            frequency=0,
            compression=0.0,
        )
        abstractions.append(abs)

    return abstractions


# ── Library pruning ──────────────────────────────────────────────────

def prune_library(abstractions: list[Abstraction],
                  test_inputs: list[Any] | None = None,
                  env: Env | None = None,
                  usage_counts: dict[str, int] | None = None,
                  min_uses: int = 0) -> tuple[list[Abstraction], dict]:
    """Prune a library by removing redundant and unused primitives.

    Three pruning strategies applied in order:

    1. Observational equivalence: If two primitives produce identical
       outputs on all test inputs, keep the one with smaller body.

    2. Builtin equivalence: If a primitive is observationally equivalent
       to a single builtin, remove it (the builtin is already available).

    3. Usage pruning: If usage_counts is provided, remove primitives
       that were never used in any solution (min_uses threshold).

    Returns (pruned_list, stats_dict).
    """
    if env is None:
        env = standard_env()

    stats = {
        "input_count": len(abstractions),
        "obs_equiv_removed": 0,
        "builtin_equiv_removed": 0,
        "unused_removed": 0,
        "output_count": 0,
    }

    # Generate test inputs if not provided
    if test_inputs is None:
        test_inputs = _default_test_inputs()

    # Step 1: Compute behavioral fingerprint for each abstraction
    behaviors: dict[str, tuple[tuple, Abstraction]] = {}
    remaining = []

    for abs in abstractions:
        fp = _compute_behavior(abs, test_inputs, env)
        if fp is None:
            remaining.append(abs)  # can't evaluate, keep it
            continue

        if fp in behaviors:
            existing_fp, existing_abs = behaviors[fp]
            # Keep the simpler one (smaller body)
            if tree_size(abs.body) < tree_size(existing_abs.body):
                # Replace with simpler version
                behaviors[fp] = (fp, abs)
                stats["obs_equiv_removed"] += 1
            else:
                stats["obs_equiv_removed"] += 1
        else:
            behaviors[fp] = (fp, abs)

    remaining.extend(abs for _, abs in behaviors.values())

    # Step 2: Remove primitives equivalent to builtins
    builtin_behaviors = _compute_builtin_behaviors(test_inputs, env)
    after_builtin = []
    for abs in remaining:
        fp = _compute_behavior(abs, test_inputs, env)
        if fp is not None and fp in builtin_behaviors:
            stats["builtin_equiv_removed"] += 1
        else:
            after_builtin.append(abs)

    remaining = after_builtin

    # Step 3: Usage pruning
    if usage_counts is not None:
        after_usage = []
        for abs in remaining:
            if usage_counts.get(abs.name, 0) >= min_uses:
                after_usage.append(abs)
            else:
                stats["unused_removed"] += 1
        remaining = after_usage

    stats["output_count"] = len(remaining)
    return remaining, stats


def _default_test_inputs() -> list[Any]:
    """Generate a diverse set of test inputs for behavioral fingerprinting."""
    return [
        # Numbers
        0.0, 1.0, -1.0, 2.0, -5.0, 3.14, 10.0, 0.5,
        # Strings
        "", "a", "hello", "HELLO", "Hello World", "  spaces  ", "abc",
    ]


def _compute_behavior(abs: Abstraction, test_inputs: list[Any],
                      env: Env) -> str | None:
    """Compute a behavioral fingerprint for an abstraction.

    Runs the abstraction on each test input and returns a string
    encoding all outputs. Returns None if the abstraction can't be
    evaluated (e.g., references undefined symbols).
    """
    # Register the abstraction temporarily
    try:
        eval_node(abs.to_defmacro(), env)
    except Exception:
        return None

    outputs = []
    for inp in test_inputs:
        try:
            # Build (abs_name input) and evaluate
            call = List((Symbol(abs.name), _val_to_node(inp)))
            result = eval_node(call, env)
            outputs.append(repr(result))
        except Exception:
            outputs.append("ERR")

    return "|".join(outputs)


def _compute_builtin_behaviors(test_inputs: list[Any],
                               env: Env) -> set[str]:
    """Compute behavioral fingerprints for all unary builtins."""
    unary_builtins = [
        "string-upper", "string-lower", "string-reverse", "string-trim",
        "string-length", "abs", "negate", "even", "odd", "not",
        "floor", "ceil", "to-string", "to-number",
    ]

    fps = set()
    for name in unary_builtins:
        outputs = []
        for inp in test_inputs:
            try:
                fn = env.lookup(name)
                from .eval import apply_fn
                result = apply_fn(fn, [inp])
                outputs.append(repr(result))
            except Exception:
                outputs.append("ERR")
        fps.add("|".join(outputs))

    # Also add identity
    outputs = [repr(inp) for inp in test_inputs]
    fps.add("|".join(outputs))

    return fps


def _val_to_node(val: Any) -> Node:
    """Convert a Python value to an AST node for evaluation."""
    if isinstance(val, bool):
        return Bool(val)
    if isinstance(val, (int, float)):
        return Number(float(val))
    if isinstance(val, str):
        return String(val)
    return Number(0.0)  # fallback


def track_usage(solutions: list[Node], abstractions: list[Abstraction]) -> dict[str, int]:
    """Count how many times each abstraction appears in a set of solutions."""
    counts = {abs.name: 0 for abs in abstractions}

    def _scan(node: Node):
        if isinstance(node, Symbol) and node.name in counts:
            counts[node.name] += 1
        if isinstance(node, List):
            for e in node.elements:
                _scan(e)

    for sol in solutions:
        _scan(sol)

    return counts
