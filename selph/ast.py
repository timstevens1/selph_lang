"""AST node types for SELPH."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union


# --- Atoms ---

@dataclass(frozen=True)
class Symbol:
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Number:
    value: float

    def __repr__(self):
        if self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)


@dataclass(frozen=True)
class String:
    value: str

    def __repr__(self):
        return f'"{self.value}"'


@dataclass(frozen=True)
class TensorLiteral:
    shape: list[Union[int, str]]  # ints or symbolic dim names
    dtype: str

    def __repr__(self):
        dims = ", ".join(str(d) for d in self.shape)
        return f"#T[{dims}] {self.dtype}"


@dataclass(frozen=True)
class Bool:
    value: bool

    def __repr__(self):
        return "true" if self.value else "false"


Atom = Union[Symbol, Number, String, TensorLiteral, Bool]


# --- Compound forms ---

@dataclass(frozen=True)
class List:
    """A parenthesized list: (f a b c)."""
    elements: tuple[Node, ...]

    def __repr__(self):
        inner = " ".join(repr(e) for e in self.elements)
        return f"({inner})"


# --- Spec-related nodes ---

@dataclass(frozen=True)
class Spec:
    """(:spec :type T :shape S :goal G :constraints C :input I)."""
    type_expr: Node | None = None
    shape_expr: Node | None = None
    goal: Node | None = None
    constraints: Node | None = None
    input_expr: Node | None = None

    def __repr__(self):
        parts = ["(:spec"]
        if self.type_expr is not None:
            parts.append(f":type {self.type_expr!r}")
        if self.shape_expr is not None:
            parts.append(f":shape {self.shape_expr!r}")
        if self.goal is not None:
            parts.append(f":goal {self.goal!r}")
        if self.constraints is not None:
            parts.append(f":constraints {self.constraints!r}")
        if self.input_expr is not None:
            parts.append(f":input {self.input_expr!r}")
        parts.append(")")
        return " ".join(parts)


@dataclass(frozen=True)
class GoalExamples:
    """(:examples ((in1 -> out1) (in2 -> out2) ...))."""
    pairs: tuple[tuple[Node, Node], ...]

    def __repr__(self):
        pairs_str = " ".join(f"({a!r} -> {b!r})" for a, b in self.pairs)
        return f"(:examples ({pairs_str}))"


@dataclass(frozen=True)
class GoalPattern:
    """(:pattern expr)."""
    pattern: Node

    def __repr__(self):
        return f"(:pattern {self.pattern!r})"


@dataclass(frozen=True)
class GoalSatisfy:
    """(:satisfy predicate)."""
    predicate: Node

    def __repr__(self):
        return f"(:satisfy {self.predicate!r})"


@dataclass(frozen=True)
class GoalTransform:
    """(:transform expr :by expr)."""
    input_expr: Node
    by_expr: Node

    def __repr__(self):
        return f"(:transform {self.input_expr!r} :by {self.by_expr!r})"


@dataclass(frozen=True)
class GoalIntent:
    """(:intent "string")."""
    intent: str

    def __repr__(self):
        return f'(:intent "{self.intent}")'


@dataclass(frozen=True)
class GoalAll:
    """(:all goal1 goal2 ...)."""
    goals: tuple[Node, ...]

    def __repr__(self):
        inner = " ".join(repr(g) for g in self.goals)
        return f"(:all {inner})"


@dataclass(frozen=True)
class GoalMinimize:
    """(:minimize fitness-fn) — find program that minimizes fitness-fn(program)."""
    fitness: Node

    def __repr__(self):
        return f"(:minimize {self.fitness!r})"


@dataclass(frozen=True)
class GoalMaximize:
    """(:maximize fitness-fn) — find program that maximizes fitness-fn(program)."""
    fitness: Node

    def __repr__(self):
        return f"(:maximize {self.fitness!r})"


# --- Type expressions ---

@dataclass(frozen=True)
class TypeArrow:
    """(-> T1 T2 ... Tresult)."""
    param_types: tuple[Node, ...]
    return_type: Node

    def __repr__(self):
        params = " ".join(repr(t) for t in self.param_types)
        return f"(-> {params} {self.return_type!r})"


@dataclass(frozen=True)
class TypeList:
    """(list T)."""
    elem_type: Node

    def __repr__(self):
        return f"(list {self.elem_type!r})"


# --- Shape expressions ---

@dataclass(frozen=True)
class Shape:
    """[dim1, dim2, ...]."""
    dims: tuple[Union[int, str], ...]  # int for concrete, str for symbolic or '?'

    def __repr__(self):
        return "[" + ", ".join(str(d) for d in self.dims) + "]"


# Union of all node types
Node = Union[
    Symbol, Number, String, TensorLiteral, Bool,
    List, Spec,
    GoalExamples, GoalPattern, GoalSatisfy, GoalTransform, GoalIntent, GoalAll,
    GoalMinimize, GoalMaximize,
    TypeArrow, TypeList, Shape,
]
