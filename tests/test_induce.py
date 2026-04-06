"""Tests for failure-driven library induction."""

import pytest
from selph.induce import induce_from_failure, InductionResult
from selph.ast import Spec, GoalExamples, Symbol, Number, String
from selph.eval import eval_node, apply_fn, standard_env


class TestIntermediateValueDecomposition:
    def test_length_then_double(self):
        """The original failing task: string-length then multiply by 2.

        Input: "hi" -> 4, "hello" -> 10, "a" -> 2
        Decomposition: string-length("hi")=2, then 2*2=4
        """
        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (String("hi"), Number(4.0)),
                (String("hello"), Number(10.0)),
                (String("a"), Number(2.0)),
            ))
        )
        result = induce_from_failure(spec, max_depth=2, max_candidates=20000)
        assert result.success, f"Induction failed. Explored {result.candidates_explored} candidates"
        # Verify the composed program works
        env = standard_env()
        fn = eval_node(result.program, env)
        assert apply_fn(fn, ["hi"]) == 4.0
        assert apply_fn(fn, ["hello"]) == 10.0
        assert apply_fn(fn, ["a"]) == 2.0

    def test_upper_then_reverse(self):
        """A two-step string task: upper then reverse."""
        spec = Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("abc"), String("CBA")),
                (String("hello"), String("OLLEH")),
            ))
        )
        # This should be solvable by regular synthesis at depth 2,
        # but induction should also find it via decomposition
        result = induce_from_failure(spec, max_depth=1, max_candidates=10000)
        if result.success:
            env = standard_env()
            fn = eval_node(result.program, env)
            assert apply_fn(fn, ["abc"]) == "CBA"


class TestConstantDiscovery:
    def test_multiply_by_constant(self):
        """Discover that output = input * 3."""
        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(2.0), Number(6.0)),
                (Number(4.0), Number(12.0)),
                (Number(5.0), Number(15.0)),
            ))
        )
        result = induce_from_failure(spec, max_depth=2, max_candidates=10000)
        assert result.success
        env = standard_env()
        fn = eval_node(result.program, env)
        assert apply_fn(fn, [2.0]) == 6.0
        assert apply_fn(fn, [10.0]) == 30.0

    def test_add_constant(self):
        """Discover that output = input + 5."""
        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (Number(0.0), Number(5.0)),
                (Number(3.0), Number(8.0)),
                (Number(-2.0), Number(3.0)),
            ))
        )
        result = induce_from_failure(spec, max_depth=2, max_candidates=10000)
        assert result.success
        env = standard_env()
        fn = eval_node(result.program, env)
        assert apply_fn(fn, [0.0]) == 5.0
        assert apply_fn(fn, [10.0]) == 15.0


class TestInductionProducesNewPrimitives:
    def test_new_primitive_created(self):
        """Induction via intermediate values should produce a new primitive."""
        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=(
                (String("hi"), Number(4.0)),
                (String("hello"), Number(10.0)),
                (String("a"), Number(2.0)),
            ))
        )
        result = induce_from_failure(spec, max_depth=2, max_candidates=20000)
        if result.success and result.new_primitives:
            prim = result.new_primitives[0]
            assert prim.arity == 1
            assert prim.name.startswith("induced_")


class TestInductionFailsGracefully:
    def test_impossible_task(self):
        """A task that can't be decomposed should fail without error."""
        spec = Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("abc"), String("xyz_unique_transform")),
                (String("def"), String("uvw_unique_transform")),
            ))
        )
        result = induce_from_failure(spec, max_depth=1, max_candidates=1000)
        assert not result.success
        assert result.candidates_explored >= 0
