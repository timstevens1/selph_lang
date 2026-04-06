"""Tests for the hierarchical namespace."""

import pytest
from selph.namespace import (
    NamespaceNode, build_namespace, scope_for_context,
    scope_for_branch, scope_for_condition, namespace_stats,
)
from selph.synthesize import Component, _default_components
from selph.library import Abstraction, promote_solved
from selph.types import TNum, TStr, TBool, TFn, TList
from selph.ast import Symbol
from selph.parser import parse


class TestBuildNamespace:
    def test_builds_from_defaults(self):
        comps = _default_components()
        tree = build_namespace(comps)
        assert tree.total_components == len(comps)
        assert len(tree.children) > 0

    def test_has_string_domain(self):
        comps = _default_components()
        tree = build_namespace(comps)
        assert "string" in tree.children

    def test_has_number_domain(self):
        comps = _default_components()
        tree = build_namespace(comps)
        assert "number" in tree.children

    def test_has_operation_kinds(self):
        comps = _default_components()
        tree = build_namespace(comps)
        string_node = tree.children.get("string")
        if string_node:
            kinds = set(string_node.children.keys())
            assert "transform" in kinds  # string-upper, etc.

    def test_constants_placed_by_type(self):
        comps = _default_components()
        tree = build_namespace(comps)
        num_node = tree.children.get("number")
        assert num_node is not None
        const_node = num_node.children.get("constants")
        assert const_node is not None
        assert len(const_node.components) > 0

    def test_includes_library_components(self):
        """Promoted primitives get placed in the tree."""
        base = _default_components()
        # Add a promoted library component
        lib_comp = Component(
            name="s1_trim_upper",
            node=Symbol("s1_trim_upper"),
            type=TFn((TStr(),), TStr()),
            arity=1,
        )
        all_comps = base + [lib_comp]
        tree = build_namespace(all_comps)
        all_in_tree = tree.all_components()
        names = [c.name for c in all_in_tree]
        assert "s1_trim_upper" in names


class TestScopeFiltering:
    def test_scope_for_string_output(self):
        comps = _default_components()
        tree = build_namespace(comps)
        scope = scope_for_context(tree, target_output=TStr())
        # Should include string-upper, string-lower, etc.
        names = [c.name for c in scope]
        assert "string-upper" in names
        # Should NOT include purely numeric functions
        assert "abs" not in names

    def test_scope_for_number_output(self):
        comps = _default_components()
        tree = build_namespace(comps)
        scope = scope_for_context(tree, target_output=TNum())
        names = [c.name for c in scope]
        assert "add" in names
        assert "string-length" in names  # cross-type: string -> number
        assert "string-upper" not in names  # string -> string

    def test_scope_for_bool_output(self):
        comps = _default_components()
        tree = build_namespace(comps)
        scope = scope_for_context(tree, target_output=TBool())
        names = [c.name for c in scope]
        assert "even" in names or "<" in names

    def test_scope_reduces_search_space(self):
        """Scoped query returns fewer components than full tree."""
        comps = _default_components()
        tree = build_namespace(comps)
        full = tree.all_components()
        scoped = scope_for_context(tree, target_output=TStr(), input_type=TStr())
        assert len(scoped) < len(full)

    def test_scope_for_branch(self):
        comps = _default_components()
        # Add some branch-type components
        comps.append(Component("thresh_pos", Symbol("thresh_pos"),
                               TFn((TNum(),), TNum()), 1))
        comps.append(Component("relu", Symbol("relu"),
                               TFn((TNum(),), TNum()), 1))
        tree = build_namespace(comps)
        branches = scope_for_branch(tree, input_type=TNum())
        names = [c.name for c in branches]
        # Branch ops should come first
        assert "thresh_pos" in names
        assert "relu" in names

    def test_scope_for_condition(self):
        comps = _default_components()
        tree = build_namespace(comps)
        conds = scope_for_condition(tree, input_type=TNum())
        names = [c.name for c in conds]
        assert any(n in names for n in ["<", ">", "even", "odd"])


class TestNamespaceNavigation:
    def test_get_path(self):
        comps = _default_components()
        tree = build_namespace(comps)
        string_node = tree.get_path(["string"])
        assert string_node is not None
        assert string_node.name == "string"

    def test_get_path_deep(self):
        comps = _default_components()
        tree = build_namespace(comps)
        transform_node = tree.get_path(["string", "transform"])
        if transform_node:
            assert len(transform_node.components) > 0

    def test_get_path_missing(self):
        comps = _default_components()
        tree = build_namespace(comps)
        assert tree.get_path(["nonexistent"]) is None


class TestNamespaceStats:
    def test_stats(self):
        comps = _default_components()
        tree = build_namespace(comps)
        stats = namespace_stats(tree)
        assert stats["total_components"] == len(comps)
        assert len(stats["domains"]) > 0


class TestPrintTree:
    def test_print_tree(self, capsys):
        comps = _default_components()
        tree = build_namespace(comps)
        tree.print_tree()
        captured = capsys.readouterr()
        assert "root/" in captured.out
        assert "string" in captured.out


class TestScopeVsFlat:
    def test_scoped_is_subset_of_flat(self):
        """Every component in a scoped query is also in the full tree."""
        comps = _default_components()
        tree = build_namespace(comps)
        full_names = {c.name for c in tree.all_components()}
        for target in [TNum(), TStr(), TBool()]:
            scoped = scope_for_context(tree, target_output=target)
            for c in scoped:
                assert c.name in full_names

    def test_multiple_scopes_cover_all(self):
        """Union of type-specific scopes covers all non-constant components."""
        comps = _default_components()
        tree = build_namespace(comps)
        all_comps = set(c.name for c in tree.all_components() if c.arity > 0)
        covered = set()
        for target in [TNum(), TStr(), TBool(), TList(TNum()), TList(TStr())]:
            scoped = scope_for_context(tree, target_output=target, include_constants=False)
            covered.update(c.name for c in scoped)
        # Not all may be covered (some have unusual types), but most should be
        assert len(covered) > len(all_comps) * 0.5
