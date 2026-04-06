"""Hierarchical namespace for the SELPH library (§12).

Organizes library primitives into a tree where:
  - Top-level branches partition by domain (string, number, logic, etc.)
  - Second-level branches partition by operation kind (transform, arithmetic, branch, etc.)
  - Leaves are individual components

The synthesizer queries the tree for components relevant to a given
context (target output type, input type, depth level), receiving a
filtered scope instead of the full flat list.

This implements §12.3 (generation as namespace traversal) and §12.5
(namespace as context filter).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .synthesize import Component
from .types import Type, TFn, TNum, TStr, TBool, TList, TVar


# ── Namespace tree ───────────────────────────────────────────────────

@dataclass
class NamespaceNode:
    """A node in the library namespace tree."""
    name: str
    children: dict[str, NamespaceNode] = field(default_factory=dict)
    components: list[Component] = field(default_factory=list)

    @property
    def total_components(self) -> int:
        """Total components in this subtree."""
        count = len(self.components)
        for child in self.children.values():
            count += child.total_components
        return count

    def add_child(self, name: str) -> NamespaceNode:
        if name not in self.children:
            self.children[name] = NamespaceNode(name=name)
        return self.children[name]

    def add_component(self, comp: Component):
        self.components.append(comp)

    def get_path(self, path: list[str]) -> NamespaceNode | None:
        """Navigate to a node by path segments."""
        if not path:
            return self
        first = path[0]
        if first in self.children:
            return self.children[first].get_path(path[1:])
        return None

    def all_components(self) -> list[Component]:
        """Collect all components in this subtree (flattened)."""
        result = list(self.components)
        for child in self.children.values():
            result.extend(child.all_components())
        return result

    def components_matching_type(self, target_ret: Type | None = None,
                                  target_param: Type | None = None) -> list[Component]:
        """Get components whose type signature matches the filter."""
        result = []
        for comp in self.all_components():
            if target_ret is not None and isinstance(comp.type, TFn):
                if not _types_match(comp.type.ret, target_ret):
                    continue
            if target_param is not None and isinstance(comp.type, TFn) and comp.type.params:
                if not _types_match(comp.type.params[0], target_param):
                    continue
            result.append(comp)
        return result

    def print_tree(self, indent: int = 0):
        """Print the tree structure."""
        prefix = "  " * indent
        count = f" ({len(self.components)} components)" if self.components else ""
        print(f"{prefix}{self.name}/{count}")
        for comp in self.components:
            sig = f"  {comp.type}" if isinstance(comp.type, TFn) else ""
            print(f"{prefix}  - {comp.name}{sig}")
        for child in sorted(self.children.values(), key=lambda c: c.name):
            child.print_tree(indent + 1)


# ── Type matching helpers ────────────────────────────────────────────

def _types_match(a: Type, b: Type) -> bool:
    """Loose type matching for namespace filtering."""
    if isinstance(a, TVar) or isinstance(b, TVar):
        return True
    if type(a) == type(b):
        if isinstance(a, TList) and isinstance(b, TList):
            return _types_match(a.elem, b.elem)
        return True
    return False


def _type_domain(t: Type) -> str:
    """Classify a type into a domain name for namespace placement."""
    if isinstance(t, TNum):
        return "number"
    if isinstance(t, TStr):
        return "string"
    if isinstance(t, TBool):
        return "logic"
    if isinstance(t, TList):
        return "list"
    if isinstance(t, TVar):
        return "generic"
    return "other"


def _operation_kind(comp: Component) -> str:
    """Classify a component into an operation kind for namespace placement."""
    name = comp.name

    # Constants
    if comp.arity == 0:
        return "constants"

    # Branching operations
    if any(kw in name for kw in ["thresh", "relu", "branch", "if_even", "if_odd",
                                   "classify", "piecewise", "abs_double",
                                   "negate_or", "abs_then", "relu_then", "negate_abs"]):
        return "branch"

    # String transforms
    if any(kw in name for kw in ["upper", "lower", "reverse", "trim"]):
        return "transform"

    # String analysis
    if any(kw in name for kw in ["length", "contains", "starts", "ends", "split"]):
        return "analyze"

    # Arithmetic
    if any(kw in name for kw in ["add", "subtract", "multiply", "divide", "modulo",
                                   "negate", "abs", "double", "floor", "ceil",
                                   "min", "max"]):
        return "arithmetic"

    # Comparison
    if name in ["<", ">", "<=", ">=", "=", "!="]:
        return "compare"

    # Logic
    if name in ["not", "even", "odd", "and", "or"]:
        return "predicate"

    # Composition
    if any(kw in name for kw in ["compose", "pipe", "map", "filter", "reduce"]):
        return "compose"

    return "other"


# ── Build the namespace tree ─────────────────────────────────────────

def build_namespace(components: list[Component]) -> NamespaceNode:
    """Build a hierarchical namespace tree from a flat list of components.

    Placement rules:
      - Top level: input/output type domain (string, number, logic, etc.)
      - Second level: operation kind (transform, arithmetic, branch, etc.)
      - For cross-type operations (string -> number), placed under input domain
    """
    root = NamespaceNode(name="root")

    for comp in components:
        # Determine domain from the function's input type
        if isinstance(comp.type, TFn) and comp.type.params:
            domain = _type_domain(comp.type.params[0])
        elif comp.arity == 0:
            domain = _type_domain(comp.type)
        else:
            domain = "other"

        kind = _operation_kind(comp)

        # Place in tree
        domain_node = root.add_child(domain)
        kind_node = domain_node.add_child(kind)
        kind_node.add_component(comp)

    return root


# ── Scoped queries ───────────────────────────────────────────────────

def scope_for_context(tree: NamespaceNode,
                      target_output: Type | None = None,
                      input_type: Type | None = None,
                      include_constants: bool = True) -> list[Component]:
    """Get components relevant to a synthesis context.

    Filters the namespace tree based on:
      - target_output: only include components that return this type
      - input_type: only include components that accept this input type
      - include_constants: whether to include arity-0 components

    This implements the context filter from §12.5:
      depth_filter ∩ type_filter ∩ stage_filter
    """
    result = []

    for domain_node in tree.children.values():
        # Domain filter: if we know the input type, skip mismatched domains
        if input_type is not None:
            domain_type = _type_domain(input_type)
            if domain_node.name not in (domain_type, "generic", "other"):
                # But allow cross-type: number domain has compare ops that take numbers
                # and string domain has analyze ops that return numbers
                pass  # don't skip — cross-type ops are useful

        for kind_node in domain_node.children.values():
            for comp in kind_node.components:
                # Skip constants if not wanted
                if not include_constants and comp.arity == 0:
                    continue

                # Type filter
                if target_output is not None and isinstance(comp.type, TFn):
                    if not _types_match(comp.type.ret, target_output):
                        continue

                if input_type is not None and isinstance(comp.type, TFn) and comp.type.params:
                    if not _types_match(comp.type.params[0], input_type):
                        continue

                result.append(comp)

    return result


def scope_for_branch(tree: NamespaceNode,
                     input_type: Type | None = None) -> list[Component]:
    """Get components suitable as if-expression branches.

    Prioritizes:
      1. Branch-specific operations (relu, threshold, piecewise)
      2. Simple transforms and arithmetic
      3. Constants
    """
    result = []

    # Priority 1: branch operations
    for domain_node in tree.children.values():
        branch_node = domain_node.children.get("branch")
        if branch_node:
            for comp in branch_node.components:
                if input_type is None or not isinstance(comp.type, TFn):
                    result.append(comp)
                elif _types_match(comp.type.params[0], input_type):
                    result.append(comp)

    # Priority 2: transforms and arithmetic
    for domain_node in tree.children.values():
        for kind_name in ("transform", "arithmetic"):
            kind_node = domain_node.children.get(kind_name)
            if kind_node:
                for comp in kind_node.components:
                    if comp not in result:
                        if input_type is None or not isinstance(comp.type, TFn):
                            result.append(comp)
                        elif _types_match(comp.type.params[0], input_type):
                            result.append(comp)

    # Priority 3: constants
    for domain_node in tree.children.values():
        const_node = domain_node.children.get("constants")
        if const_node:
            for comp in const_node.components:
                if comp not in result:
                    result.append(comp)

    return result


def scope_for_condition(tree: NamespaceNode,
                        input_type: Type | None = None) -> list[Component]:
    """Get components suitable as if-expression conditions.

    Returns: comparisons, predicates, and logic operations.
    """
    result = []
    B = TBool()

    for domain_node in tree.children.values():
        for kind_name in ("compare", "predicate"):
            kind_node = domain_node.children.get(kind_name)
            if kind_node:
                for comp in kind_node.components:
                    if isinstance(comp.type, TFn) and _types_match(comp.type.ret, B):
                        result.append(comp)

    return result


# ── Namespace statistics ─────────────────────────────────────────────

def namespace_stats(tree: NamespaceNode) -> dict:
    """Compute statistics about the namespace tree."""
    stats = {
        "total_components": tree.total_components,
        "domains": {},
        "max_depth": 0,
    }

    for domain_name, domain_node in tree.children.items():
        domain_stats = {
            "total": domain_node.total_components,
            "kinds": {},
        }
        for kind_name, kind_node in domain_node.children.items():
            domain_stats["kinds"][kind_name] = len(kind_node.components)
        stats["domains"][domain_name] = domain_stats

    return stats
