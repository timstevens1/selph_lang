"""Type and shape checker for SELPH.

Implements Hindley-Milner-style type inference over SELPH ASTs.
Types are inferred bottom-up through the tree; type errors produce
structured diagnostics suitable for feeding back to the critic (Loop 2).

The type system serves two roles per the spec (§3.1):
  1. At eval time: catch type/shape mismatches and produce error values.
  2. At generation time (future): constrain decoding to valid token sets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union
from .ast import (
    Node, Symbol, Number, String, Bool, TensorLiteral, List,
    Spec, GoalExamples, GoalPattern, GoalSatisfy, GoalTransform,
    GoalIntent, GoalAll, TypeArrow, TypeList, Shape,
)


# ── Type representation ─────────────────────────────────────────────

@dataclass(frozen=True)
class TNum:
    """Numeric type."""
    def __repr__(self): return "number"

@dataclass(frozen=True)
class TStr:
    """String type."""
    def __repr__(self): return "string"

@dataclass(frozen=True)
class TBool:
    """Boolean type."""
    def __repr__(self): return "bool"

@dataclass(frozen=True)
class TNil:
    """Nil / unit type."""
    def __repr__(self): return "nil"

@dataclass(frozen=True)
class TList:
    """Homogeneous list type."""
    elem: Type
    def __repr__(self): return f"(list {self.elem!r})"

@dataclass(frozen=True)
class TFn:
    """Function type: (-> T1 T2 ... Tresult)."""
    params: tuple[Type, ...]
    ret: Type
    variadic: bool = False  # if True, last param type repeats

    def __repr__(self):
        params = " ".join(repr(t) for t in self.params)
        v = "..." if self.variadic else ""
        return f"(-> {params}{v} {self.ret!r})"

@dataclass
class TVar:
    """Type variable for inference. Mutable: gets unified."""
    id: int
    bound: Type | None = None

    def __repr__(self):
        if self.bound is not None:
            return repr(self.bound)
        return f"?t{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, TVar) and self.id == other.id

@dataclass(frozen=True)
class TSpec:
    """Type of a spec value."""
    def __repr__(self): return "spec"

@dataclass(frozen=True)
class TGoal:
    """Type of a goal value."""
    def __repr__(self): return "goal"

@dataclass(frozen=True)
class TTensor:
    """Tensor type with optional shape."""
    shape: tuple[Dim, ...] | None = None
    dtype: str | None = None

    def __repr__(self):
        if self.shape is None:
            return "tensor"
        dims = ", ".join(repr(d) for d in self.shape)
        s = f"tensor[{dims}]"
        if self.dtype:
            s += f" {self.dtype}"
        return s


# Dimensions can be concrete ints, symbolic names, or variables
@dataclass(frozen=True)
class DimConst:
    value: int
    def __repr__(self): return str(self.value)

@dataclass(frozen=True)
class DimSym:
    name: str
    def __repr__(self): return self.name

@dataclass
class DimVar:
    id: int
    bound: Dim | None = None
    def __repr__(self):
        if self.bound is not None:
            return repr(self.bound)
        return f"?d{self.id}"

Dim = Union[DimConst, DimSym, DimVar]

Type = Union[TNum, TStr, TBool, TNil, TList, TFn, TVar, TSpec, TGoal, TTensor]


# ── Type errors ──────────────────────────────────────────────────────

@dataclass
class TypeError_:
    """Structured type error for the critic."""
    message: str
    node: Node | None = None
    expected: Type | None = None
    actual: Type | None = None

    def __repr__(self):
        parts = [self.message]
        if self.expected is not None:
            parts.append(f"expected {self.expected!r}")
        if self.actual is not None:
            parts.append(f"got {self.actual!r}")
        return "; ".join(parts)


class TypeCheckError(Exception):
    def __init__(self, error: TypeError_):
        self.error = error
        super().__init__(str(error))


# ── Inference state ──────────────────────────────────────────────────

class InferenceState:
    """Tracks type variables and unification during inference."""

    def __init__(self):
        self._next_tvar = 0
        self._next_dvar = 0

    def fresh_tvar(self) -> TVar:
        v = TVar(self._next_tvar)
        self._next_tvar += 1
        return v

    def fresh_dvar(self) -> DimVar:
        v = DimVar(self._next_dvar)
        self._next_dvar += 1
        return v

    def resolve(self, t: Type) -> Type:
        """Follow TVar chains to get the resolved type."""
        while isinstance(t, TVar) and t.bound is not None:
            t = t.bound
        if isinstance(t, TList):
            return TList(self.resolve(t.elem))
        if isinstance(t, TFn):
            return TFn(
                tuple(self.resolve(p) for p in t.params),
                self.resolve(t.ret),
                t.variadic,
            )
        if isinstance(t, TTensor) and t.shape is not None:
            return TTensor(
                tuple(self.resolve_dim(d) for d in t.shape),
                t.dtype,
            )
        return t

    def resolve_dim(self, d: Dim) -> Dim:
        while isinstance(d, DimVar) and d.bound is not None:
            d = d.bound
        return d

    def unify(self, a: Type, b: Type, node: Node | None = None) -> Type:
        """Unify two types, binding type variables as needed."""
        a = self.resolve(a)
        b = self.resolve(b)

        if a == b:
            return a

        # TVar binds to anything
        if isinstance(a, TVar):
            if self._occurs_in(a, b):
                raise TypeCheckError(TypeError_(
                    "infinite type", node, a, b))
            a.bound = b
            return b
        if isinstance(b, TVar):
            if self._occurs_in(b, a):
                raise TypeCheckError(TypeError_(
                    "infinite type", node, b, a))
            b.bound = a
            return a

        # TNum unifies with TNum, etc.
        if type(a) == type(b):
            if isinstance(a, TList):
                elem = self.unify(a.elem, b.elem, node)
                return TList(elem)
            if isinstance(a, TFn):
                if len(a.params) != len(b.params) and not a.variadic and not b.variadic:
                    raise TypeCheckError(TypeError_(
                        f"arity mismatch in function types", node, a, b))
                # Unify param-by-param for matching lengths
                min_len = min(len(a.params), len(b.params))
                params = []
                for i in range(min_len):
                    params.append(self.unify(a.params[i], b.params[i], node))
                ret = self.unify(a.ret, b.ret, node)
                return TFn(tuple(params), ret)
            if isinstance(a, TTensor):
                return self._unify_tensors(a, b, node)
            # Same concrete type (TNum, TStr, TBool, TNil, TSpec, TGoal)
            return a

        # Number and tensor can coerce in some contexts
        # For now, strict: different concrete types don't unify
        raise TypeCheckError(TypeError_(
            "type mismatch", node, a, b))

    def _unify_tensors(self, a: TTensor, b: TTensor, node: Node | None) -> TTensor:
        """Unify two tensor types, including shape dimensions."""
        if a.shape is None:
            return b
        if b.shape is None:
            return a
        if len(a.shape) != len(b.shape):
            raise TypeCheckError(TypeError_(
                f"shape rank mismatch: {len(a.shape)} vs {len(b.shape)}", node, a, b))
        dims = []
        for da, db in zip(a.shape, b.shape):
            dims.append(self.unify_dim(da, db, node))
        dtype = a.dtype or b.dtype
        return TTensor(tuple(dims), dtype)

    def unify_dim(self, a: Dim, b: Dim, node: Node | None = None) -> Dim:
        a = self.resolve_dim(a)
        b = self.resolve_dim(b)
        if a == b:
            return a
        if isinstance(a, DimVar):
            a.bound = b
            return b
        if isinstance(b, DimVar):
            b.bound = a
            return a
        if isinstance(a, DimConst) and isinstance(b, DimConst) and a.value != b.value:
            raise TypeCheckError(TypeError_(
                f"dimension mismatch: {a.value} vs {b.value}", node))
        # DimSym with different names don't unify
        if isinstance(a, DimSym) and isinstance(b, DimSym) and a.name != b.name:
            raise TypeCheckError(TypeError_(
                f"dimension mismatch: {a.name} vs {b.name}", node))
        return a

    def _occurs_in(self, tvar: TVar, t: Type) -> bool:
        t = self.resolve(t)
        if isinstance(t, TVar):
            return t.id == tvar.id
        if isinstance(t, TList):
            return self._occurs_in(tvar, t.elem)
        if isinstance(t, TFn):
            return any(self._occurs_in(tvar, p) for p in t.params) or self._occurs_in(tvar, t.ret)
        return False


# ── Type environment ─────────────────────────────────────────────────

class TypeEnv:
    """Maps names to their types. Lexically scoped like the eval Env."""

    def __init__(self, bindings: dict[str, Type] | None = None, parent: TypeEnv | None = None):
        self.bindings: dict[str, Type] = bindings or {}
        self.parent = parent

    def lookup(self, name: str) -> Type | None:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def define(self, name: str, t: Type):
        self.bindings[name] = t

    def extend(self, names: list[str], types: list[Type]) -> TypeEnv:
        return TypeEnv(dict(zip(names, types)), parent=self)


# ── Type inference ───────────────────────────────────────────────────

def infer(node: Node, tenv: TypeEnv, state: InferenceState) -> Type:
    """Infer the type of a SELPH AST node."""

    if isinstance(node, Number):
        return TNum()
    if isinstance(node, String):
        return TStr()
    if isinstance(node, Bool):
        return TBool()
    if isinstance(node, TensorLiteral):
        dims = []
        for d in node.shape:
            if isinstance(d, int):
                dims.append(DimConst(d))
            else:
                dims.append(DimSym(d))
        return TTensor(tuple(dims), node.dtype)

    if isinstance(node, Symbol):
        # Check polymorphic builtins first (fresh type vars per use site)
        if node.name in _POLYMORPHIC:
            fresh = fresh_builtin_type(node.name, state)
            if fresh is not None:
                return fresh
        t = tenv.lookup(node.name)
        if t is None:
            raise TypeCheckError(TypeError_(
                f"unbound symbol: {node.name}", node))
        return t

    if isinstance(node, Spec):
        return TSpec()

    if isinstance(node, (GoalExamples, GoalPattern, GoalSatisfy,
                         GoalTransform, GoalIntent, GoalAll)):
        return TGoal()

    if isinstance(node, Shape):
        return TSpec()  # shapes in type annotation context

    if isinstance(node, List):
        if len(node.elements) == 0:
            return TList(state.fresh_tvar())

        head = node.elements[0]

        if isinstance(head, Symbol):
            form = head.name

            if form == "quote":
                return state.fresh_tvar()  # quoted expressions have opaque type

            if form == "if":
                return _infer_if(node, tenv, state)

            if form == "let":
                return _infer_let(node, tenv, state)

            if form == "lambda":
                return _infer_lambda(node, tenv, state)

            if form == "define":
                return _infer_define(node, tenv, state)

            if form == "defmacro":
                return state.fresh_tvar()  # macros have opaque type for now

            if form == "do":
                return _infer_do(node, tenv, state)

            if form == "and" or form == "or":
                # Type check all sub-expressions, return type of last
                t = TBool()
                for expr in node.elements[1:]:
                    t = infer(expr, tenv, state)
                return t

        # Function application
        return _infer_application(node, tenv, state)

    raise TypeCheckError(TypeError_(
        f"cannot type-check node: {type(node).__name__}", node))


def _infer_if(node: List, tenv: TypeEnv, state: InferenceState) -> Type:
    """(if cond then else?)"""
    elems = node.elements
    cond_t = infer(elems[1], tenv, state)
    state.unify(cond_t, TBool(), elems[1])
    then_t = infer(elems[2], tenv, state)
    if len(elems) == 4:
        else_t = infer(elems[3], tenv, state)
        return state.unify(then_t, else_t, node)
    return then_t


def _infer_let(node: List, tenv: TypeEnv, state: InferenceState) -> Type:
    """(let ((name1 val1) ...) body)"""
    elems = node.elements
    bindings_node = elems[1]
    child_tenv = TypeEnv(parent=tenv)

    for binding in bindings_node.elements:
        name_node = binding.elements[0]
        # Pre-define with a fresh tvar for recursive references
        tv = state.fresh_tvar()
        child_tenv.define(name_node.name, tv)
        val_t = infer(binding.elements[1], child_tenv, state)
        state.unify(tv, val_t, binding)

    return infer(elems[2], child_tenv, state)


def _infer_lambda(node: List, tenv: TypeEnv, state: InferenceState) -> Type:
    """(lambda (params...) body)"""
    elems = node.elements
    params_node = elems[1]
    param_tvars = []
    child_tenv = TypeEnv(parent=tenv)

    for p in params_node.elements:
        tv = state.fresh_tvar()
        child_tenv.define(p.name, tv)
        param_tvars.append(tv)

    body_t = infer(elems[2], child_tenv, state)
    return TFn(tuple(param_tvars), body_t)


def _infer_define(node: List, tenv: TypeEnv, state: InferenceState) -> Type:
    """(define name value)"""
    elems = node.elements
    val_t = infer(elems[2], tenv, state)
    tenv.define(elems[1].name, val_t)
    return val_t


def _infer_do(node: List, tenv: TypeEnv, state: InferenceState) -> Type:
    """(do expr1 expr2 ...) — type is last expression's type."""
    t = TNil()
    for expr in node.elements[1:]:
        t = infer(expr, tenv, state)
    return t


def _infer_application(node: List, tenv: TypeEnv, state: InferenceState) -> Type:
    """(f arg1 arg2 ...) — infer function type and unify with args."""
    fn_t = infer(node.elements[0], tenv, state)
    fn_t = state.resolve(fn_t)
    arg_types = [infer(a, tenv, state) for a in node.elements[1:]]

    # If fn_t is a TVar, constrain it to be a function
    if isinstance(fn_t, TVar):
        ret_tv = state.fresh_tvar()
        expected_fn = TFn(tuple(arg_types), ret_tv)
        state.unify(fn_t, expected_fn, node)
        return ret_tv

    if not isinstance(fn_t, TFn):
        raise TypeCheckError(TypeError_(
            "applying a non-function", node, TFn((), state.fresh_tvar()), fn_t))

    # Handle variadic functions
    if fn_t.variadic:
        if len(arg_types) < len(fn_t.params):
            raise TypeCheckError(TypeError_(
                f"too few arguments: expected at least {len(fn_t.params)}, got {len(arg_types)}",
                node))
        for i, at in enumerate(arg_types):
            param_t = fn_t.params[min(i, len(fn_t.params) - 1)]
            state.unify(param_t, at, node.elements[i + 1])
    else:
        if len(arg_types) != len(fn_t.params):
            raise TypeCheckError(TypeError_(
                f"arity mismatch: expected {len(fn_t.params)} args, got {len(arg_types)}",
                node))
        for i, (pt, at) in enumerate(zip(fn_t.params, arg_types)):
            state.unify(pt, at, node.elements[i + 1])

    return fn_t.ret


# ── Builtin type signatures ─────────────────────────────────────────

def _builtin_types() -> dict[str, Type]:
    """Type signatures for all standard builtins."""
    t = {}
    N = TNum()
    S = TStr()
    B = TBool()

    # Arithmetic: (-> number number number)
    for name in ["add", "+", "subtract", "-", "multiply", "*", "divide", "/",
                  "modulo", "%", "min", "max"]:
        t[name] = TFn((N, N), N)

    # Unary numeric
    for name in ["abs", "negate", "floor", "ceil"]:
        t[name] = TFn((N,), N)

    # Comparison: (-> number number bool)
    for name in ["<", ">", "<=", ">="]:
        t[name] = TFn((N, N), B)

    # Polymorphic equality: uses type variables
    # We'll handle = and != specially by returning fresh signatures each time
    # For now, use num->num->bool as the common case
    t["="] = TFn((N, N), B)
    t["!="] = TFn((N, N), B)

    # Logic
    t["not"] = TFn((B,), B)
    t["even"] = TFn((N,), B)
    t["odd"] = TFn((N,), B)

    # String operations
    t["concat"] = TFn((S,), S, variadic=True)
    t["string-length"] = TFn((S,), N)
    t["substring"] = TFn((S, N, N), S)
    t["string-upper"] = TFn((S,), S)
    t["string-lower"] = TFn((S,), S)
    t["string-reverse"] = TFn((S,), S)
    t["string-contains"] = TFn((S, S), B)
    t["string-split"] = TFn((S, S), TList(S))
    t["string-join"] = TFn((TList(S), S), S)
    t["string-starts-with"] = TFn((S, S), B)
    t["string-ends-with"] = TFn((S, S), B)
    t["string-replace"] = TFn((S, S, S), S)
    t["string-trim"] = TFn((S,), S)
    t["char-at"] = TFn((S, N), S)
    t["to-string"] = TFn((N,), S)  # simplified
    t["to-number"] = TFn((S,), N)

    # List operations — these need polymorphism, use TVar placeholders
    # We handle them by generating fresh type vars per call site (see below)

    # Type predicates: any -> bool
    for name in ["number?", "string?", "bool?", "list?", "nil?", "callable?"]:
        t[name] = TFn((N,), B)  # simplified: accept any, return bool

    # nil
    t["nil"] = TNil()

    # Higher-order (map, filter, reduce) and list ops handled as polymorphic below

    return t


# Polymorphic builtins get fresh type vars per call site
_POLYMORPHIC = {
    "list", "cons", "head", "tail", "length", "nth", "append", "reverse",
    "range", "empty?", "contains", "sort",
    "map", "filter", "reduce", "compose", "pipe", "apply",
    "identity", "=", "!=", "print",
}


def fresh_builtin_type(name: str, state: InferenceState) -> Type | None:
    """Generate a fresh polymorphic type signature for a builtin.

    Returns None if the name isn't a known polymorphic builtin.
    """
    N = TNum()
    S = TStr()
    B = TBool()
    a = state.fresh_tvar()
    b = state.fresh_tvar()

    sigs = {
        # Polymorphic equality
        "=": TFn((a, a), B),
        "!=": TFn((a, a), B),

        # List operations
        "list": TFn((a,), TList(a), variadic=True),
        "cons": TFn((a, TList(a)), TList(a)),
        "head": TFn((TList(a),), a),
        "tail": TFn((TList(a),), TList(a)),
        "length": TFn((TList(a),), N),
        "nth": TFn((TList(a), N), a),
        "append": TFn((TList(a), TList(a)), TList(a)),
        "reverse": TFn((TList(a),), TList(a)),
        "sort": TFn((TList(a),), TList(a)),
        "range": TFn((N,), TList(N), variadic=True),
        "empty?": TFn((TList(a),), B),
        "contains": TFn((TList(a), a), B),

        # Higher-order
        "map": TFn((TFn((a,), b), TList(a)), TList(b)),
        "filter": TFn((TFn((a,), B), TList(a)), TList(a)),
        "reduce": TFn((TFn((b, a), b), TList(a), b), b),

        # Compose: (-> (-> b c) (-> a b) (-> a c))
        "compose": (lambda c=state.fresh_tvar(): TFn(
            (TFn((b,), c), TFn((a,), b)),
            TFn((a,), c),
        ))(),

        # Pipe: (-> a (-> a b) b)  — simplified to 2-arg
        "pipe": TFn((a, TFn((a,), b)), b, variadic=True),

        # Apply: (-> (-> a b) (list a) b)  — simplified
        "apply": TFn((TFn((a,), b), TList(a)), b),

        # Identity
        "identity": TFn((a,), a),

        # Print
        "print": TFn((a,), TNil(), variadic=True),

        # Type predicates (accept anything)
        "number?": TFn((a,), B),
        "string?": TFn((a,), B),
        "bool?": TFn((a,), B),
        "list?": TFn((a,), B),
        "nil?": TFn((a,), B),
        "callable?": TFn((a,), B),
    }

    return sigs.get(name)


# ── Public API ───────────────────────────────────────────────────────

def standard_type_env() -> TypeEnv:
    """Create a type environment with all builtin type signatures."""
    return TypeEnv(_builtin_types())


def typecheck(node: Node, tenv: TypeEnv | None = None, state: InferenceState | None = None) -> Type:
    """Type-check a SELPH AST node. Returns the inferred type.

    Raises TypeCheckError on type errors.
    """
    if tenv is None:
        tenv = standard_type_env()
    if state is None:
        state = InferenceState()

    # Wrap infer to handle polymorphic builtins: when a symbol is looked up
    # and it's a known polymorphic builtin, generate fresh type vars.
    return _infer_with_poly(node, tenv, state)


def _infer_with_poly(node: Node, tenv: TypeEnv, state: InferenceState) -> Type:
    """Infer with special handling for polymorphic builtins."""
    if isinstance(node, Symbol) and node.name in _POLYMORPHIC:
        fresh = fresh_builtin_type(node.name, state)
        if fresh is not None:
            return fresh

    if isinstance(node, List) and len(node.elements) > 0:
        head = node.elements[0]
        # For application of polymorphic builtins, get fresh sig
        if isinstance(head, Symbol) and head.name in _POLYMORPHIC:
            fn_t = fresh_builtin_type(head.name, state)
            if fn_t is not None:
                fn_t = state.resolve(fn_t)
                arg_types = [_infer_with_poly(a, tenv, state) for a in node.elements[1:]]

                if not isinstance(fn_t, TFn):
                    raise TypeCheckError(TypeError_(
                        "applying a non-function", node))

                if fn_t.variadic:
                    for i, at in enumerate(arg_types):
                        param_t = fn_t.params[min(i, len(fn_t.params) - 1)]
                        state.unify(param_t, at, node.elements[i + 1])
                else:
                    if len(arg_types) != len(fn_t.params):
                        raise TypeCheckError(TypeError_(
                            f"arity mismatch: expected {len(fn_t.params)}, got {len(arg_types)}",
                            node))
                    for i, (pt, at) in enumerate(zip(fn_t.params, arg_types)):
                        state.unify(pt, at, node.elements[i + 1])

                return state.resolve(fn_t.ret)

        # Special forms dispatch to the normal infer
        if isinstance(head, Symbol) and head.name in (
            "quote", "if", "let", "lambda", "define", "defmacro", "do", "and", "or"
        ):
            return infer(node, tenv, state)

        # Generic application
        fn_t = _infer_with_poly(node.elements[0], tenv, state)
        fn_t = state.resolve(fn_t)
        arg_types = [_infer_with_poly(a, tenv, state) for a in node.elements[1:]]

        if isinstance(fn_t, TVar):
            ret_tv = state.fresh_tvar()
            expected_fn = TFn(tuple(arg_types), ret_tv)
            state.unify(fn_t, expected_fn, node)
            return ret_tv

        if not isinstance(fn_t, TFn):
            raise TypeCheckError(TypeError_(
                "applying a non-function", node, TFn((), state.fresh_tvar()), fn_t))

        if fn_t.variadic:
            for i, at in enumerate(arg_types):
                param_t = fn_t.params[min(i, len(fn_t.params) - 1)]
                state.unify(param_t, at, node.elements[i + 1])
        else:
            if len(arg_types) != len(fn_t.params):
                raise TypeCheckError(TypeError_(
                    f"arity mismatch: expected {len(fn_t.params)}, got {len(arg_types)}",
                    node))
            for i, (pt, at) in enumerate(zip(fn_t.params, arg_types)):
                state.unify(pt, at, node.elements[i + 1])

        return state.resolve(fn_t.ret)

    return infer(node, tenv, state)


def typecheck_string(source: str) -> Type:
    """Parse and type-check a SELPH expression from source text."""
    from .parser import parse
    node = parse(source)
    return typecheck(node)
