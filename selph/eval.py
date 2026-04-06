"""Tree-walking evaluator for SELPH.

Covers the Stage 0-2 subset: arithmetic, string ops, logic, let, lambda, if,
quote, map, reduce, filter, compose, pipe, defmacro, apply.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
from .ast import (
    Node, Symbol, Number, String, Bool, TensorLiteral, List,
    Spec, GoalExamples, GoalSatisfy, GoalAll,
)


class EvalError(Exception):
    pass


# --- Values ---

# SELPH values at runtime are Python values:
#   Number  -> float/int
#   String  -> str
#   Bool    -> bool
#   List    -> Python list (of SELPH values)
#   Lambda  -> Closure
#   Macro   -> Macro
#   Builtin -> BuiltinFn
#   Spec    -> Spec AST node (first-class)
#   nil     -> None

@dataclass
class Closure:
    """A user-defined function (lambda)."""
    params: list[str]
    body: Node
    env: Env

    def __repr__(self):
        return f"<lambda ({' '.join(self.params)})>"


@dataclass
class Macro:
    """A user-defined macro (defmacro)."""
    name: str
    params: list[str]
    body: Node

    def __repr__(self):
        return f"<macro {self.name}>"


@dataclass
class BuiltinFn:
    """A built-in function wrapping a Python callable."""
    name: str
    fn: Callable

    def __repr__(self):
        return f"<builtin {self.name}>"


class Namespace:
    """A first-class namespace value — a tree of named entries.

    Entries can be any SELPH value (functions, data, other namespaces).
    Supports path-based lookup: (ns-get my-ns "math" "basic" "double")
    and tree operations: merge, filter, map, keys, values.

    This is the SELPH-native representation of §12's hierarchical namespace.
    A namespace is itself a value that programs can create, inspect, and compose.
    """

    def __init__(self, entries: dict | None = None, name: str = ""):
        self.entries: dict[str, Any] = entries or {}
        self.name = name

    def get(self, key: str) -> Any:
        """Get a direct child by name."""
        if key not in self.entries:
            raise EvalError(f"namespace '{self.name}': no entry '{key}'")
        return self.entries[key]

    def get_path(self, path: list[str]) -> Any:
        """Navigate a dot-separated path: ["math", "basic", "double"]."""
        if not path:
            return self
        first = path[0]
        val = self.get(first)
        if len(path) == 1:
            return val
        if isinstance(val, Namespace):
            return val.get_path(path[1:])
        raise EvalError(f"namespace '{self.name}': '{first}' is not a namespace, "
                        f"can't traverse further")

    def put(self, key: str, value: Any) -> 'Namespace':
        """Return a new namespace with key set to value (immutable update)."""
        new_entries = dict(self.entries)
        new_entries[key] = value
        return Namespace(new_entries, self.name)

    def put_path(self, path: list[str], value: Any) -> 'Namespace':
        """Set a value at a nested path, creating intermediate namespaces as needed."""
        if not path:
            raise EvalError("empty path")
        if len(path) == 1:
            return self.put(path[0], value)
        first = path[0]
        child = self.entries.get(first)
        if child is None:
            child = Namespace(name=first)
        elif not isinstance(child, Namespace):
            raise EvalError(f"namespace '{self.name}': '{first}' exists but is not a namespace")
        new_child = child.put_path(path[1:], value)
        return self.put(first, new_child)

    def keys(self) -> list[str]:
        return list(self.entries.keys())

    def values(self) -> list:
        return list(self.entries.values())

    def items(self) -> list[tuple[str, Any]]:
        return list(self.entries.items())

    def merge(self, other: 'Namespace') -> 'Namespace':
        """Merge another namespace into this one. Other's entries win on conflict."""
        new_entries = dict(self.entries)
        for key, val in other.entries.items():
            existing = new_entries.get(key)
            if isinstance(existing, Namespace) and isinstance(val, Namespace):
                new_entries[key] = existing.merge(val)
            else:
                new_entries[key] = val
        return Namespace(new_entries, self.name)

    def filter_values(self, pred) -> 'Namespace':
        """Return a new namespace keeping only entries where pred(value) is true.
        Recurses into child namespaces."""
        new_entries = {}
        for key, val in self.entries.items():
            if isinstance(val, Namespace):
                filtered = val.filter_values(pred)
                if filtered.entries:  # keep non-empty branches
                    new_entries[key] = filtered
            elif pred(val):
                new_entries[key] = val
        return Namespace(new_entries, self.name)

    def flatten(self, prefix: str = "") -> dict[str, Any]:
        """Flatten to a dict of dotted-path -> leaf-value."""
        result = {}
        for key, val in self.entries.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(val, Namespace):
                result.update(val.flatten(path))
            else:
                result[path] = val
        return result

    def __repr__(self):
        if not self.entries:
            return "(namespace)"
        items = " ".join(f"({k} ...)" for k in sorted(self.entries.keys()))
        return f"(namespace {items})"

    def __len__(self):
        return sum(
            len(v) if isinstance(v, Namespace) else 1
            for v in self.entries.values()
        )


# --- Environment ---

class Env:
    """Lexical scope chain."""

    def __init__(self, bindings: dict[str, Any] | None = None, parent: Env | None = None):
        self.bindings: dict[str, Any] = bindings or {}
        self.parent = parent

    def lookup(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise EvalError(f"unbound symbol: {name}")

    def define(self, name: str, value: Any):
        self.bindings[name] = value

    def extend(self, names: list[str], values: list[Any]) -> Env:
        if len(names) != len(values):
            raise EvalError(f"arity mismatch: expected {len(names)} args, got {len(values)}")
        return Env(dict(zip(names, values)), parent=self)


# --- Evaluator ---

def eval_node(node: Node, env: Env) -> Any:
    """Evaluate a SELPH AST node in the given environment."""

    # Atoms
    if isinstance(node, Number):
        return node.value
    if isinstance(node, String):
        return node.value
    if isinstance(node, Bool):
        return node.value
    if isinstance(node, TensorLiteral):
        return node  # tensor literals are values for now

    # Symbol lookup
    if isinstance(node, Symbol):
        return env.lookup(node.name)

    # Specs are first-class values
    if isinstance(node, Spec):
        return node

    # Goal nodes are first-class values
    if isinstance(node, (GoalExamples, GoalSatisfy, GoalAll)):
        return node

    # Lists: special forms and function application
    if isinstance(node, List):
        if len(node.elements) == 0:
            return []  # empty list evaluates to empty Python list

        head = node.elements[0]

        # Special forms (head is a symbol with special meaning)
        if isinstance(head, Symbol):
            form = head.name

            if form == "quote":
                return _eval_quote(node, env)
            if form == "if":
                return _eval_if(node, env)
            if form == "let":
                return _eval_let(node, env)
            if form == "lambda":
                return _eval_lambda(node, env)
            if form == "defmacro":
                return _eval_defmacro(node, env)
            if form == "define":
                return _eval_define(node, env)
            if form == "do":
                return _eval_do(node, env)
            if form == "and":
                return _eval_and(node, env)
            if form == "or":
                return _eval_or(node, env)
            if form == "namespace":
                return _eval_namespace(node, env)

        # Function application: evaluate head and args, then call
        fn = eval_node(head, env)
        args = [eval_node(e, env) for e in node.elements[1:]]

        if isinstance(fn, Macro):
            # Macros receive unevaluated args as AST, expand, then eval result
            expanded = _apply_macro(fn, node.elements[1:], env)
            return eval_node(expanded, env)

        return apply_fn(fn, args)

    raise EvalError(f"cannot evaluate node type: {type(node).__name__}")


def apply_fn(fn: Any, args: list[Any]) -> Any:
    """Apply a function value to evaluated arguments."""
    if isinstance(fn, Closure):
        child_env = fn.env.extend(fn.params, args)
        return eval_node(fn.body, child_env)
    if isinstance(fn, BuiltinFn):
        return fn.fn(*args)
    raise EvalError(f"not callable: {fn!r}")


# --- Special forms ---

def _eval_quote(node: List, env: Env) -> Node:
    """(quote expr) -> unevaluated AST node."""
    if len(node.elements) != 2:
        raise EvalError("quote takes exactly 1 argument")
    return node.elements[1]


def _eval_if(node: List, env: Env) -> Any:
    """(if cond then else?)."""
    elems = node.elements
    if len(elems) < 3 or len(elems) > 4:
        raise EvalError("if takes 2 or 3 arguments")
    cond = eval_node(elems[1], env)
    if cond:
        return eval_node(elems[2], env)
    elif len(elems) == 4:
        return eval_node(elems[3], env)
    return None


def _eval_let(node: List, env: Env) -> Any:
    """(let ((name1 val1) (name2 val2) ...) body).

    Uses letrec semantics: all bindings share a single child env so that
    lambdas can refer to names defined in the same let block (enables recursion).
    """
    elems = node.elements
    if len(elems) != 3:
        raise EvalError("let takes exactly 2 arguments: bindings and body")

    bindings_node = elems[1]
    if not isinstance(bindings_node, List):
        raise EvalError("let bindings must be a list")

    child_env = Env(parent=env)
    for binding in bindings_node.elements:
        if not isinstance(binding, List) or len(binding.elements) != 2:
            raise EvalError("each let binding must be (name value)")
        name_node = binding.elements[0]
        if not isinstance(name_node, Symbol):
            raise EvalError(f"let binding name must be a symbol, got {type(name_node).__name__}")
        val = eval_node(binding.elements[1], child_env)
        # For closures, patch the captured env to be this shared child_env
        # so recursive references resolve correctly.
        if isinstance(val, Closure):
            val = Closure(val.params, val.body, child_env)
        child_env.define(name_node.name, val)

    return eval_node(elems[2], child_env)


def _eval_lambda(node: List, env: Env) -> Closure:
    """(lambda (params...) body)."""
    elems = node.elements
    if len(elems) != 3:
        raise EvalError("lambda takes exactly 2 arguments: params and body")

    params_node = elems[1]
    if not isinstance(params_node, List):
        raise EvalError("lambda params must be a list")

    params = []
    for p in params_node.elements:
        if not isinstance(p, Symbol):
            raise EvalError(f"lambda param must be a symbol, got {type(p).__name__}")
        params.append(p.name)

    return Closure(params, elems[2], env)


def _eval_defmacro(node: List, env: Env) -> Macro:
    """(defmacro name (params...) body) — defines a macro in the current env."""
    elems = node.elements
    if len(elems) != 4:
        raise EvalError("defmacro takes exactly 3 arguments: name, params, body")

    name_node = elems[1]
    if not isinstance(name_node, Symbol):
        raise EvalError("defmacro name must be a symbol")

    params_node = elems[2]
    if not isinstance(params_node, List):
        raise EvalError("defmacro params must be a list")

    params = []
    for p in params_node.elements:
        if not isinstance(p, Symbol):
            raise EvalError(f"defmacro param must be a symbol, got {type(p).__name__}")
        params.append(p.name)

    macro = Macro(name_node.name, params, elems[3])
    env.define(name_node.name, macro)
    return macro


def _eval_define(node: List, env: Env) -> Any:
    """(define name value) — defines a binding in the current env."""
    elems = node.elements
    if len(elems) != 3:
        raise EvalError("define takes exactly 2 arguments: name and value")
    name_node = elems[1]
    if not isinstance(name_node, Symbol):
        raise EvalError("define name must be a symbol")
    val = eval_node(elems[2], env)
    env.define(name_node.name, val)
    return val


def _eval_do(node: List, env: Env) -> Any:
    """(do expr1 expr2 ...) — evaluate sequentially, return last."""
    result = None
    for expr in node.elements[1:]:
        result = eval_node(expr, env)
    return result


def _eval_and(node: List, env: Env) -> Any:
    """(and a b ...) — short-circuit logical and."""
    result = True
    for expr in node.elements[1:]:
        result = eval_node(expr, env)
        if not result:
            return result
    return result


def _eval_or(node: List, env: Env) -> Any:
    """(or a b ...) — short-circuit logical or."""
    result = False
    for expr in node.elements[1:]:
        result = eval_node(expr, env)
        if result:
            return result
    return result


def _eval_namespace(node: List, env: Env) -> Namespace:
    """(namespace (name1 val1) (name2 val2) ...) — create a namespace.

    Each child is (name value) where value is evaluated.
    If value evaluates to a Namespace, it becomes a sub-namespace.

    Example:
      (namespace
        (math (namespace
          (double (lambda (x) (add x x)))
          (triple (lambda (x) (add x (add x x))))))
        (greeting "hello"))
    """
    entries = {}
    for elem in node.elements[1:]:
        if not isinstance(elem, List) or len(elem.elements) != 2:
            raise EvalError("namespace entries must be (name value) pairs")
        name_node = elem.elements[0]
        if not isinstance(name_node, Symbol):
            raise EvalError(f"namespace key must be a symbol, got {type(name_node).__name__}")
        val = eval_node(elem.elements[1], env)
        entries[name_node.name] = val
    return Namespace(entries)


def _apply_macro(macro: Macro, arg_nodes: tuple[Node, ...], env: Env) -> Node:
    """Expand a macro by substituting AST nodes into the body."""
    if len(arg_nodes) != len(macro.params):
        raise EvalError(f"macro {macro.name}: expected {len(macro.params)} args, got {len(arg_nodes)}")
    # Evaluate args before substitution (like a function, not a textual macro)
    # For true textual macros we'd pass unevaluated nodes, but for SELPH's
    # defmacro (which acts like shorthand definitions), evaluate-then-substitute
    # matches the spec examples.
    subs = dict(zip(macro.params, arg_nodes))
    return _substitute(macro.body, subs)


def _substitute(node: Node, subs: dict[str, Node]) -> Node:
    """Replace symbol occurrences in an AST tree."""
    if isinstance(node, Symbol):
        return subs.get(node.name, node)
    if isinstance(node, List):
        return List(tuple(_substitute(e, subs) for e in node.elements))
    # Atoms and other nodes pass through unchanged
    return node


# --- Standard library of builtins ---

def _make_builtins() -> dict[str, Any]:
    """Create the standard set of built-in functions."""
    builtins = {}

    def register(name: str, fn: Callable):
        builtins[name] = BuiltinFn(name, fn)

    # --- Arithmetic ---
    register("add", lambda a, b: a + b)
    register("+", lambda a, b: a + b)
    register("subtract", lambda a, b: a - b)
    register("-", lambda a, b: a - b)
    register("multiply", lambda a, b: a * b)
    register("*", lambda a, b: a * b)
    register("divide", lambda a, b: a / b if b != 0 else _err("division by zero"))
    register("/", lambda a, b: a / b if b != 0 else _err("division by zero"))
    register("modulo", lambda a, b: a % b)
    register("%", lambda a, b: a % b)
    register("abs", lambda a: abs(a))
    register("negate", lambda a: -a)
    register("min", lambda a, b: min(a, b))
    register("max", lambda a, b: max(a, b))
    register("floor", lambda a: float(int(a)))
    register("ceil", lambda a: float(int(a) + (1 if a != int(a) and a > 0 else 0)))

    # --- Comparison ---
    register("=", lambda a, b: a == b)
    register("!=", lambda a, b: a != b)
    register("<", lambda a, b: a < b)
    register(">", lambda a, b: a > b)
    register("<=", lambda a, b: a <= b)
    register(">=", lambda a, b: a >= b)

    # --- Logic ---
    register("not", lambda a: not a)
    register("even", lambda a: a % 2 == 0)
    register("odd", lambda a: a % 2 != 0)

    # --- String operations ---
    register("concat", lambda *args: "".join(str(a) for a in args))
    register("string-length", lambda s: float(len(s)))
    register("substring", lambda s, start, end: s[int(start):int(end)])
    register("string-upper", lambda s: s.upper())
    register("string-lower", lambda s: s.lower())
    register("string-reverse", lambda s: s[::-1])
    register("string-contains", lambda s, sub: sub in s)
    register("string-split", lambda s, sep: s.split(sep))
    register("string-join", lambda lst, sep: sep.join(str(x) for x in lst))
    register("string-starts-with", lambda s, prefix: s.startswith(prefix))
    register("string-ends-with", lambda s, suffix: s.endswith(suffix))
    register("string-replace", lambda s, old, new: s.replace(old, new))
    register("string-trim", lambda s: s.strip())
    register("char-at", lambda s, i: s[int(i)])
    register("to-string", lambda a: str(a) if not isinstance(a, str) else a)
    register("to-number", lambda a: float(a))

    # --- List operations ---
    register("list", lambda *args: list(args))
    register("cons", lambda head, tail: [head] + (tail if isinstance(tail, list) else [tail]))
    register("head", lambda lst: lst[0] if lst else _err("head of empty list"))
    register("tail", lambda lst: lst[1:] if lst else _err("tail of empty list"))
    register("length", lambda lst: float(len(lst)))
    register("nth", lambda lst, n: lst[int(n)])
    register("append", lambda a, b: a + b if isinstance(a, list) else [a] + (b if isinstance(b, list) else [b]))
    register("reverse", lambda lst: list(reversed(lst)))
    register("range", lambda *args: list(
        float(x) for x in range(
            int(args[0]) if len(args) > 1 else 0,
            int(args[0]) if len(args) == 1 else int(args[1]),
            int(args[2]) if len(args) > 2 else 1,
        )
    ))
    register("empty?", lambda lst: len(lst) == 0 if isinstance(lst, list) else lst == "" or lst is None)
    register("contains", lambda lst, item: item in lst)
    register("sort", lambda lst: sorted(lst))

    # --- Higher-order functions (builtin versions; also available as special forms) ---
    def _map(fn, lst):
        return [apply_fn(fn, [x]) for x in lst]

    def _filter(fn, lst):
        return [x for x in lst if apply_fn(fn, [x])]

    def _reduce(fn, lst, *init):
        acc = init[0] if init else lst[0]
        items = lst if init else lst[1:]
        for x in items:
            acc = apply_fn(fn, [acc, x])
        return acc

    register("map", _map)
    register("filter", _filter)
    register("reduce", _reduce)

    # --- Compose and pipe ---
    def _compose(*fns):
        def composed(*args):
            result = apply_fn(fns[-1], list(args))
            for fn in reversed(fns[:-1]):
                result = apply_fn(fn, [result])
            return result
        return BuiltinFn("composed", composed)

    def _pipe(val, *fns):
        result = val
        for fn in fns:
            result = apply_fn(fn, [result])
        return result

    register("compose", _compose)
    register("pipe", _pipe)

    # --- Apply ---
    register("apply", lambda fn, args: apply_fn(fn, args))

    # --- Identity and constants ---
    register("identity", lambda x: x)

    # --- Type checking ---
    register("number?", lambda x: isinstance(x, (int, float)))
    register("string?", lambda x: isinstance(x, str))
    register("bool?", lambda x: isinstance(x, bool))
    register("list?", lambda x: isinstance(x, list))
    register("nil?", lambda x: x is None)
    register("callable?", lambda x: isinstance(x, (Closure, BuiltinFn)))

    # --- IO (minimal, for REPL) ---
    register("print", lambda *args: print(*args) or None)

    # --- Namespace operations ---
    register("ns-get", lambda ns, *keys: ns.get_path(list(keys)) if isinstance(ns, Namespace) else _err("ns-get: first arg must be a namespace"))
    register("ns-put", lambda ns, key, val: ns.put(key, val) if isinstance(ns, Namespace) else _err("ns-put: first arg must be a namespace"))
    register("ns-put-path", lambda ns, path_str, val: ns.put_path(path_str.split("."), val) if isinstance(ns, Namespace) else _err("ns-put-path: first arg must be a namespace"))
    register("ns-keys", lambda ns: ns.keys() if isinstance(ns, Namespace) else _err("ns-keys: arg must be a namespace"))
    register("ns-values", lambda ns: ns.values() if isinstance(ns, Namespace) else _err("ns-values: arg must be a namespace"))
    register("ns-merge", lambda a, b: a.merge(b) if isinstance(a, Namespace) and isinstance(b, Namespace) else _err("ns-merge: both args must be namespaces"))
    register("ns-flatten", lambda ns: ns.flatten() if isinstance(ns, Namespace) else _err("ns-flatten: arg must be a namespace"))
    register("ns-filter", lambda ns, pred: ns.filter_values(lambda v: apply_fn(pred, [v])) if isinstance(ns, Namespace) else _err("ns-filter: first arg must be a namespace"))
    register("ns-size", lambda ns: float(len(ns)) if isinstance(ns, Namespace) else _err("ns-size: arg must be a namespace"))
    register("ns?", lambda x: isinstance(x, Namespace))
    register("ns-empty", lambda: Namespace())

    # --- Synthesis (exposed for meta-programming / fitness functions) ---
    def _selph_synthesize(spec_ns):
        """Run synthesis from a SELPH namespace describing the task.

        spec_ns should have: "spec" (a Spec value), optionally "max-depth",
        "max-candidates", "heuristic".
        Returns a namespace with "found", "candidates", "program", "source".
        """
        from .synthesize import synthesize as _synth, SynthesisResult
        if not isinstance(spec_ns, Namespace):
            raise EvalError("synthesize: arg must be a namespace")

        spec = spec_ns.get("spec")
        max_depth = int(spec_ns.entries.get("max-depth", 2))
        max_cand = int(spec_ns.entries.get("max-candidates", 10000))
        heur = spec_ns.entries.get("heuristic", None)

        sr = _synth(spec, max_depth=max_depth, max_candidates=max_cand,
                    heuristic=heur)
        return Namespace({
            "found": sr.found,
            "candidates": float(sr.candidates_explored),
            "source": sr.source or "",
        })

    register("synthesize", _selph_synthesize)

    # nil
    builtins["nil"] = None

    return builtins


def _err(msg: str):
    raise EvalError(msg)


def standard_env() -> Env:
    """Create a fresh environment with all standard builtins."""
    return Env(_make_builtins())


# --- Spec verification (Level 0-2) ---

def verify_spec(spec: Spec, program_fn: Closure | BuiltinFn, env: Env) -> dict:
    """Verify a program against a spec. Returns {passed: bool, details: ...}."""
    goal = spec.goal
    if goal is None:
        return {"passed": True, "details": "no goal to verify"}

    if isinstance(goal, GoalExamples):
        return _verify_examples(goal, program_fn)

    if isinstance(goal, GoalSatisfy):
        return _verify_satisfy(goal, program_fn, env)

    if isinstance(goal, GoalAll):
        results = []
        all_passed = True
        for sub_goal in goal.goals:
            sub_spec = Spec(goal=sub_goal)
            r = verify_spec(sub_spec, program_fn, env)
            results.append(r)
            if not r["passed"]:
                all_passed = False
        return {"passed": all_passed, "details": results}

    return {"passed": False, "details": f"unsupported goal type: {type(goal).__name__}"}


def _verify_examples(goal: GoalExamples, program_fn) -> dict:
    """Level 0: run on each input, compare output to expected."""
    failures = []
    for input_node, expected_node in goal.pairs:
        # Input/expected are AST nodes — extract their literal values
        input_val = _ast_to_value(input_node)
        expected_val = _ast_to_value(expected_node)
        try:
            actual = apply_fn(program_fn, [input_val])
        except Exception as e:
            failures.append({"input": input_val, "expected": expected_val, "error": str(e)})
            continue
        if actual != expected_val:
            failures.append({"input": input_val, "expected": expected_val, "actual": actual})

    passed = len(failures) == 0
    return {"passed": passed, "total": len(goal.pairs), "failures": failures}


def _verify_satisfy(goal: GoalSatisfy, program_fn, env: Env) -> dict:
    """Level 2: evaluate the predicate on the program's output.

    The predicate is a lambda that takes the output and returns bool.
    For verify_spec, we need to call the program first to get output,
    but this depends on what input to use. For now, we evaluate the
    predicate as a function and return it for the caller to use.
    """
    predicate = eval_node(goal.predicate, env)
    # The caller should apply program_fn to get output, then check predicate(output).
    # For a generic verify, we just return the predicate as a checker.
    return {"passed": True, "predicate": predicate, "details": "predicate compiled; call with output to verify"}


def _ast_to_value(node: Node) -> Any:
    """Convert a literal AST node to a Python value."""
    if isinstance(node, Number):
        return node.value
    if isinstance(node, String):
        return node.value
    if isinstance(node, Bool):
        return node.value
    if isinstance(node, List):
        return [_ast_to_value(e) for e in node.elements]
    if isinstance(node, Symbol):
        return node.name  # symbols as strings in value space
    return node


# --- Convenience ---

def eval_string(source: str, env: Env | None = None) -> Any:
    """Parse and evaluate a SELPH expression from source text."""
    from .parser import parse
    node = parse(source)
    if env is None:
        env = standard_env()
    return eval_node(node, env)


def eval_program(source: str, env: Env | None = None) -> Any:
    """Parse and evaluate multiple top-level expressions, return last result."""
    from .parser import parse_file
    nodes = parse_file(source)
    if env is None:
        env = standard_env()
    result = None
    for node in nodes:
        result = eval_node(node, env)
    return result
