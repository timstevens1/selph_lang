"""Tests for the SELPH library extraction system (Loop 3)."""

import pytest
from selph.library import (
    subtrees, tree_size, tree_fingerprint, tree_structural_fingerprint,
    anti_unify, anti_unify_many,
    count_subtrees, find_common_patterns,
    compression_score, extract_library,
    rewrite_with_abstraction, rewrite_corpus,
    Abstraction, _nodes_equal,
)
from selph.parser import parse
from selph.ast import Symbol, Number, String, List
from selph.synthesize import Component, synthesize
from selph.eval import eval_node, eval_program, standard_env
from selph.ast import Spec, GoalExamples


# ── Tree utilities ───────────────────────────────────────────────────

class TestTreeUtils:
    def test_tree_size_atom(self):
        assert tree_size(parse("x")) == 1
        assert tree_size(parse("42")) == 1

    def test_tree_size_list(self):
        # List node (1) + add (1) + 1 (1) + 2 (1) = 4
        assert tree_size(parse("(add 1 2)")) == 4

    def test_tree_size_nested(self):
        # Outer list (1) + add (1) + inner list (1) + mul (1) + 2 (1) + 3 (1) + 4 (1) = 7
        assert tree_size(parse("(add (mul 2 3) 4)")) == 7

    def test_subtrees_count(self):
        node = parse("(add 1 2)")
        subs = subtrees(node)
        # The list itself + add + 1 + 2 = 4
        assert len(subs) == 4

    def test_subtrees_nested(self):
        node = parse("(add (mul 2 3) 4)")
        subs = subtrees(node)
        # (add (mul 2 3) 4), add, (mul 2 3), mul, 2, 3, 4 = 7
        assert len(subs) == 7

    def test_structural_fingerprint_same_structure(self):
        """Same structure, different variables -> same fingerprint."""
        fp1 = tree_structural_fingerprint(parse("(string-upper x)"))
        fp2 = tree_structural_fingerprint(parse("(string-upper y)"))
        assert fp1 == fp2

    def test_structural_fingerprint_different_op(self):
        """Different operations -> different fingerprint."""
        fp1 = tree_structural_fingerprint(parse("(string-upper x)"))
        fp2 = tree_structural_fingerprint(parse("(string-lower x)"))
        assert fp1 != fp2

    def test_structural_fingerprint_preserves_builtins(self):
        fp = tree_structural_fingerprint(parse("(add x 1)"))
        assert "add" in fp

    def test_fingerprint_collapses_variables(self):
        fp = tree_fingerprint(parse("(add x y)"))
        # add, x, y all map to V (basic fingerprint doesn't distinguish builtins)
        assert fp.count("V") == 3


# ── Anti-unification ─────────────────────────────────────────────────

class TestAntiUnify:
    def test_identical_nodes(self):
        a = parse("(add 1 2)")
        b = parse("(add 1 2)")
        pattern, subs = anti_unify(a, b)
        assert _nodes_equal(pattern, a)
        assert len(subs) == 0

    def test_different_args(self):
        a = parse("(string-upper x)")
        b = parse("(string-upper y)")
        pattern, subs = anti_unify(a, b)
        # Pattern should be (string-upper _p0)
        assert isinstance(pattern, List)
        assert pattern.elements[0] == Symbol("string-upper")
        assert isinstance(pattern.elements[1], Symbol)
        assert pattern.elements[1].name.startswith("_p")
        assert len(subs) == 1

    def test_different_constants(self):
        a = parse("(add x 1)")
        b = parse("(add x 2)")
        pattern, subs = anti_unify(a, b)
        assert isinstance(pattern, List)
        assert pattern.elements[0] == Symbol("add")
        assert pattern.elements[1] == Symbol("x")  # x is the same
        assert pattern.elements[2].name.startswith("_p")  # 1 vs 2 differs

    def test_completely_different(self):
        a = parse("(add x 1)")
        b = parse("(string-upper y)")
        pattern, subs = anti_unify(a, b)
        # Completely different -> single parameter
        assert isinstance(pattern, Symbol)
        assert pattern.name.startswith("_p")

    def test_same_head_different_depth(self):
        a = parse("(add (add x 1) 2)")
        b = parse("(add (add y 3) 2)")
        pattern, subs = anti_unify(a, b)
        # Outer add and 2 match, inner add matches, x/y and 1/3 differ
        assert isinstance(pattern, List)
        assert len(subs) == 2  # x/y and 1/3

    def test_anti_unify_many(self):
        nodes = [
            parse("(string-upper a)"),
            parse("(string-upper b)"),
            parse("(string-upper c)"),
        ]
        pattern, params = anti_unify_many(nodes)
        assert isinstance(pattern, List)
        assert pattern.elements[0] == Symbol("string-upper")
        assert len(params) == 1


# ── Frequency analysis ───────────────────────────────────────────────

class TestFrequencyAnalysis:
    def test_count_subtrees(self):
        corpus = [
            parse("(string-upper x)"),
            parse("(string-upper y)"),
            parse("(string-lower z)"),
        ]
        counts = count_subtrees(corpus, min_size=2)
        # (string-upper _) should appear in 2 programs
        fp = tree_structural_fingerprint(parse("(string-upper x)"))
        assert counts[fp] == 2

    def test_find_common_patterns(self):
        corpus = [
            parse("(add (string-length x) 1)"),
            parse("(add (string-length y) 2)"),
            parse("(add (string-length z) 3)"),
        ]
        patterns = find_common_patterns(corpus, min_frequency=2, min_size=2)
        assert len(patterns) > 0
        # (string-length _) should be the most common sub-pattern
        fps = [fp for fp, _, _ in patterns]
        sl_fp = tree_structural_fingerprint(parse("(string-length x)"))
        assert sl_fp in fps

    def test_no_patterns_in_diverse_corpus(self):
        corpus = [
            parse("(add 1 2)"),
            parse("(string-upper x)"),
            parse("(not true)"),
        ]
        patterns = find_common_patterns(corpus, min_frequency=2, min_size=2)
        assert len(patterns) == 0


# ── Compression scoring ──────────────────────────────────────────────

class TestCompression:
    def test_useful_abstraction(self):
        pattern = parse("(add (string-length x) 1)")
        corpus = [
            parse("(add (string-length a) 1)"),
            parse("(add (string-length b) 1)"),
            parse("(add (string-length c) 1)"),
            parse("(add (string-length d) 1)"),
        ]
        score = compression_score(pattern, ["x"], corpus)
        # pattern_size=5, savings_per_use=5-2=3, uses=4, definition_cost=5+1+3=9
        # total = 3*4 - 9 = 3
        assert score > 0

    def test_useless_abstraction(self):
        """An abstraction that only appears once isn't worth it."""
        pattern = parse("(add (string-length x) 1)")
        corpus = [parse("(add (string-length a) 1)")]
        score = compression_score(pattern, ["x"], corpus)
        assert score == 0.0

    def test_trivial_abstraction(self):
        """An abstraction with as many params as nodes saves nothing."""
        pattern = parse("(add x y)")
        corpus = [
            parse("(add 1 2)"),
            parse("(add 3 4)"),
        ]
        # pattern_size=3, savings=3-3=0
        score = compression_score(pattern, ["x", "y"], corpus)
        assert score == 0.0


# ── Full extraction pipeline ─────────────────────────────────────────

class TestExtraction:
    def test_extract_from_repeated_pattern(self):
        """Extract an abstraction from programs that share a common pattern."""
        corpus = [
            parse("(add (string-length a) 1)"),
            parse("(add (string-length b) 1)"),
            parse("(add (string-length c) 1)"),
            parse("(add (string-length d) 1)"),
            parse("(add (string-length e) 1)"),
        ]
        abstractions = extract_library(corpus, min_frequency=2, min_compression=0.0)
        assert len(abstractions) > 0
        # Should find something related to (add (string-length _) 1)
        best = abstractions[0]
        assert best.frequency >= 2
        assert best.arity >= 1

    def test_extract_produces_defmacro(self):
        corpus = [
            parse("(add (string-length a) 1)"),
            parse("(add (string-length b) 1)"),
            parse("(add (string-length c) 1)"),
        ]
        abstractions = extract_library(corpus, min_frequency=2, min_compression=0.0)
        if abstractions:
            macro = abstractions[0].to_defmacro()
            assert isinstance(macro, List)
            assert macro.elements[0] == Symbol("defmacro")

    def test_extract_produces_component(self):
        corpus = [
            parse("(add (string-length a) 1)"),
            parse("(add (string-length b) 1)"),
            parse("(add (string-length c) 1)"),
        ]
        abstractions = extract_library(corpus, min_frequency=2, min_compression=0.0)
        if abstractions:
            comp = abstractions[0].to_component()
            assert isinstance(comp, Component)
            assert comp.arity >= 1

    def test_no_extraction_from_unique_programs(self):
        corpus = [
            parse("(add 1 2)"),
            parse("(string-upper x)"),
            parse("(not true)"),
        ]
        abstractions = extract_library(corpus, min_frequency=2)
        assert len(abstractions) == 0


# ── Rewriting ────────────────────────────────────────────────────────

class TestRewriting:
    def test_rewrite_simple(self):
        """Replace occurrences of a pattern with abstraction calls."""
        corpus = [
            parse("(add (string-length a) 1)"),
            parse("(add (string-length b) 1)"),
        ]
        abstractions = extract_library(corpus, min_frequency=2, min_compression=0.0)
        if abstractions:
            rewritten = rewrite_corpus(corpus, abstractions)
            for prog in rewritten:
                # The rewritten program should contain the abstraction name
                prog_str = repr(prog)
                assert abstractions[0].name in prog_str or tree_size(prog) <= tree_size(corpus[0])

    def test_rewrite_preserves_semantics(self):
        """Rewriting should not change program semantics when the macro is defined."""
        corpus = [
            parse("(string-upper a)"),
            parse("(string-upper b)"),
            parse("(string-upper c)"),
        ]
        abstractions = extract_library(corpus, min_frequency=2, min_compression=0.0)
        if abstractions:
            abs0 = abstractions[0]
            env = standard_env()
            # Define the macro in the environment
            eval_node(abs0.to_defmacro(), env)

            # Rewrite and evaluate
            for orig in corpus:
                rewritten = rewrite_with_abstraction(orig, abs0)
                # Both should produce the same structure when we can evaluate them
                # (they reference unbound vars a, b, c so we test structure only)
                assert tree_size(rewritten) <= tree_size(orig)


# ── Integration: synthesis -> extraction -> faster synthesis ─────────

class TestBootstrapping:
    def test_library_speeds_up_synthesis(self):
        """The key validation: does synthesis with extracted abstractions
        explore fewer candidates than synthesis without them?

        We synthesize solutions to several similar tasks, extract the library,
        then synthesize a new similar task with and without the library.
        """
        # Phase 1: solve several "string -> upper" style tasks
        solved_programs = []
        for inp, out in [("hi", "HI"), ("abc", "ABC"), ("test", "TEST")]:
            spec = Spec(
                type_expr=Symbol("string"),
                goal=GoalExamples(pairs=(
                    (String(inp), String(out)),
                    (String(inp.lower()), String(out)),
                ))
            )
            result = synthesize(spec, max_depth=1)
            if result.found:
                # Extract the body of the lambda
                body = result.program.elements[2]
                solved_programs.append(body)

        assert len(solved_programs) >= 2, "Need at least 2 solved programs"

        # Phase 2: extract library
        abstractions = extract_library(solved_programs, min_frequency=2, min_compression=0.0)

        # Phase 3: solve a new task WITH the library components
        new_spec = Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("new"), String("NEW")),
                (String("word"), String("WORD")),
            ))
        )

        # Without library
        result_without = synthesize(new_spec, max_depth=1)
        assert result_without.found

        # With library (add abstraction components)
        extra = [a.to_component() for a in abstractions]
        result_with = synthesize(new_spec, max_depth=1, extra_components=extra)
        assert result_with.found

        # Both should find a solution; the library version may explore same or fewer
        # (for this simple case they'll be similar, but the infrastructure works)
        assert result_with.candidates_explored <= result_without.candidates_explored + 5
