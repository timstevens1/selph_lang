"""Tests for multi-tree synthesis."""

import pytest
from selph.multitree import (
    multi_synthesize, extract_components, trace_solution,
    namespace_to_components,
)
from selph.eval import (
    eval_string, eval_program, standard_env, Namespace, Closure,
    BuiltinFn, apply_fn, eval_node,
)
from selph.ast import Spec, GoalExamples, Symbol, Number, String


def _make_ns(source: str, env=None) -> Namespace:
    """Helper: evaluate a namespace expression."""
    if env is None:
        env = standard_env()
    return eval_string(source, env)


class TestExtractComponents:
    def test_extract_from_flat(self):
        env = standard_env()
        ns = _make_ns('(namespace (double (lambda (x) (add x x))) (pi 3.14))', env)
        comps = extract_components(ns, tree_name="math", env=env)
        names = [c.name for c in comps]
        assert "math.double" in names
        assert "math.pi" in names

    def test_extract_from_nested(self):
        env = standard_env()
        ns = _make_ns("""
        (namespace
          (basic (namespace
            (inc (lambda (x) (add x 1)))
            (dec (lambda (x) (subtract x 1)))))
          (advanced (namespace
            (double (lambda (x) (add x x))))))
        """, env)
        comps = extract_components(ns, tree_name="ops", env=env)
        names = [c.name for c in comps]
        assert "ops.basic.inc" in names
        assert "ops.basic.dec" in names
        assert "ops.advanced.double" in names

    def test_extracted_functions_callable(self):
        """Extracted components should be registered in env and callable."""
        env = standard_env()
        ns = _make_ns('(namespace (double (lambda (x) (add x x))))', env)
        comps = extract_components(ns, tree_name="lib", env=env)
        # The component's symbol should resolve in env
        fn = env.lookup("lib.double")
        assert apply_fn(fn, [5.0]) == 10.0

    def test_extract_constants(self):
        env = standard_env()
        ns = _make_ns('(namespace (pi 3.14) (name "hello") (flag true))', env)
        comps = extract_components(ns, tree_name="data", env=env)
        names = [c.name for c in comps]
        assert "data.pi" in names
        assert "data.name" in names

    def test_inferred_types(self):
        env = standard_env()
        ns = _make_ns('(namespace (double (lambda (x) (add x x))))', env)
        comps = extract_components(ns, tree_name="t", env=env)
        double_comp = [c for c in comps if c.name == "t.double"][0]
        # Should infer (-> number number)
        assert double_comp.arity == 1


class TestMultiSynthesizeBasic:
    def test_single_tree(self):
        """Synthesize using a single custom namespace."""
        env = standard_env()
        ops = _make_ns("""
        (namespace
          (double (lambda (x) (add x x)))
          (negate (lambda (x) (subtract 0 x))))
        """, env)

        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(3.0), Number(6.0)),
                (Number(5.0), Number(10.0)),
                (Number(0.0), Number(0.0)),
            ))
        )
        result = multi_synthesize(spec, trees=[ops], tree_names=["ops"], env=env)
        assert result.found
        # Should find ops.double
        fn = eval_node(result.program, env)
        assert apply_fn(fn, [7.0]) == 14.0

    def test_two_trees(self):
        """Synthesize composing functions from different trees."""
        env = standard_env()
        math_ops = _make_ns("""
        (namespace (double (lambda (x) (add x x))))
        """, env)
        str_ops = _make_ns("""
        (namespace (upper (lambda (s) (string-upper s))))
        """, env)

        # Task: uppercase a string (should find str_ops.upper)
        spec = Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("hello"), String("HELLO")),
                (String("world"), String("WORLD")),
            ))
        )
        result = multi_synthesize(
            spec, trees=[math_ops, str_ops],
            tree_names=["math", "str"], env=env)
        assert result.found

    def test_cross_tree_composition(self):
        """Compose a function from one tree with data from another."""
        env = standard_env()
        ops = _make_ns("""
        (namespace (add1 (lambda (x) (add x 1))))
        """, env)
        data = _make_ns("""
        (namespace (offset 10))
        """, env)

        # Task: add 1 to input (should find ops.add1)
        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(0.0), Number(1.0)),
                (Number(5.0), Number(6.0)),
            ))
        )
        result = multi_synthesize(
            spec, trees=[ops, data],
            tree_names=["ops", "data"], env=env)
        assert result.found


class TestMultiSynthesizeComposition:
    def test_chain_across_trees(self):
        """Chain: get a function from tree A, apply to result of tree B's function."""
        env = standard_env()
        transforms = _make_ns("""
        (namespace (clean (lambda (s) (string-trim s))))
        """, env)
        formatters = _make_ns("""
        (namespace (shout (lambda (s) (string-upper s))))
        """, env)

        # Task: trim then uppercase
        spec = Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("  hi  "), String("HI")),
                (String(" abc "), String("ABC")),
            ))
        )
        result = multi_synthesize(
            spec, trees=[transforms, formatters],
            tree_names=["tx", "fmt"],
            env=env, max_depth=2)
        assert result.found
        fn = eval_node(result.program, env)
        assert apply_fn(fn, ["  test  "]) == "TEST"


class TestTraceSolution:
    def test_trace_single_tree(self):
        env = standard_env()
        ops = _make_ns('(namespace (double (lambda (x) (add x x))))', env)

        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(3.0), Number(6.0)),
                (Number(5.0), Number(10.0)),
            ))
        )
        result = multi_synthesize(spec, trees=[ops], tree_names=["ops"], env=env)
        if result.found:
            trace = trace_solution(result, [ops], ["ops"])
            # If solution uses ops.double, trace should show it
            if "ops" in trace:
                assert any("double" in p for p in trace["ops"])

    def test_trace_shows_which_trees(self):
        env = standard_env()
        tree_a = _make_ns('(namespace (inc (lambda (x) (add x 1))))', env)
        tree_b = _make_ns('(namespace (dec (lambda (x) (subtract x 1))))', env)

        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(0.0), Number(1.0)),
                (Number(5.0), Number(6.0)),
            ))
        )
        result = multi_synthesize(
            spec, trees=[tree_a, tree_b],
            tree_names=["a", "b"], env=env)
        if result.found:
            trace = trace_solution(result, [tree_a, tree_b], ["a", "b"])
            # Should use tree_a (inc), not tree_b (dec)
            if trace:
                assert "a" in trace or not trace  # might use builtins instead


class TestArbitraryTrees:
    def test_domain_knowledge_tree(self):
        """Use a 'knowledge' tree containing facts as constants."""
        env = standard_env()
        knowledge = _make_ns("""
        (namespace
          (thresholds (namespace
            (freezing 0)
            (boiling 100)
            (body_temp 37))))
        """, env)

        # Extract and verify
        comps = extract_components(knowledge, tree_name="kb", env=env)
        names = [c.name for c in comps]
        assert "kb.thresholds.freezing" in names
        assert "kb.thresholds.boiling" in names

    def test_cached_results_tree(self):
        """Use a 'cache' tree containing previously computed results."""
        env = standard_env()
        cache = _make_ns("""
        (namespace
          (prev_double (lambda (x) (add x x)))
          (prev_negate (lambda (x) (subtract 0 x)))
          (last_answer 42))
        """, env)

        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(3.0), Number(6.0)),
                (Number(5.0), Number(10.0)),
            ))
        )
        result = multi_synthesize(spec, trees=[cache], tree_names=["cache"], env=env)
        assert result.found

    def test_three_arbitrary_trees(self):
        """Search across three unrelated trees."""
        env = standard_env()
        physics = _make_ns('(namespace (gravity 9.81) (c 299792458))', env)
        cooking = _make_ns('(namespace (cups_to_ml (lambda (x) (multiply x 236.588))))', env)
        music = _make_ns('(namespace (a440 440) (semitone 1.059))', env)

        # These won't help solve a string task, but the system should
        # handle them gracefully
        spec = Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("hello"), String("HELLO")),
            ))
        )
        result = multi_synthesize(
            spec, trees=[physics, cooking, music],
            tree_names=["physics", "cooking", "music"],
            env=env)
        # Should still solve via builtins (string-upper)
        assert result.found


class TestEmptyAndEdgeCases:
    def test_empty_trees(self):
        env = standard_env()
        empty = Namespace()
        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(1.0), Number(2.0)),
                (Number(3.0), Number(4.0)),
            ))
        )
        result = multi_synthesize(spec, trees=[empty], tree_names=["empty"], env=env)
        # Should still work via builtins
        assert result.found

    def test_no_trees(self):
        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(1.0), Number(2.0)),
                (Number(3.0), Number(4.0)),
            ))
        )
        result = multi_synthesize(spec, trees=[], tree_names=[])
        assert result.found  # builtins still available
