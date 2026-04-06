"""Tests for optimization-based synthesis (minimize/maximize goals)."""

import pytest
from selph.synthesize import synthesize, SynthesisResult, Component
from selph.eval import (
    eval_string, eval_program, eval_node, apply_fn,
    standard_env, Namespace, Closure,
)
from selph.ast import Spec, GoalMinimize, GoalMaximize, Symbol, Number, String
from selph.parser import parse
from selph.types import TNum, TFn


class TestGoalParsing:
    def test_parse_minimize(self):
        node = parse("(:minimize (lambda (f) (f 5)))")
        assert isinstance(node, GoalMinimize)

    def test_parse_maximize(self):
        node = parse("(:maximize (lambda (f) (f 5)))")
        assert isinstance(node, GoalMaximize)

    def test_parse_spec_with_minimize(self):
        node = parse('(:spec :type number :goal (:minimize (lambda (f) (f 5))))')
        assert isinstance(node, Spec)
        assert isinstance(node.goal, GoalMinimize)


class TestMinimize:
    def test_minimize_constant(self):
        """Find the number that minimizes (lambda (x) (abs x))."""
        env = standard_env()
        spec = Spec(
            goal=GoalMinimize(parse("(lambda (x) (abs x))"))
        )
        result = synthesize(spec, max_depth=0, max_candidates=1000, env=env)
        assert result.found
        assert result.fitness_score == 0.0  # abs(0) = 0

    def test_minimize_distance(self):
        """Find number closest to 5: minimize |x - 5|."""
        env = standard_env()
        spec = Spec(
            goal=GoalMinimize(parse("(lambda (x) (abs (subtract x 5)))"))
        )
        result = synthesize(spec, max_depth=0, max_candidates=1000, env=env)
        assert result.found
        val = eval_node(result.program, env)
        assert val == 5.0
        assert result.fitness_score == 0.0


class TestMaximize:
    def test_maximize_constant(self):
        """Find the number that maximizes identity."""
        env = standard_env()
        spec = Spec(
            goal=GoalMaximize(parse("(lambda (x) x)"))
        )
        result = synthesize(spec, max_depth=0, max_candidates=1000, env=env)
        assert result.found
        val = eval_node(result.program, env)
        # Should pick the largest constant available (10.0)
        assert val == 10.0

    def test_maximize_negative(self):
        """Find number that maximizes -|x| (i.e., closest to 0)."""
        env = standard_env()
        spec = Spec(
            goal=GoalMaximize(parse("(lambda (x) (subtract 0 (abs x)))"))
        )
        result = synthesize(spec, max_depth=0, max_candidates=1000, env=env)
        assert result.found
        val = eval_node(result.program, env)
        assert val == 0.0


class TestOptimizeFunction:
    def test_minimize_error_function(self):
        """Synthesize a function that minimizes error on a dataset.

        Fitness = sum of squared errors on test points.
        Target: f(x) = 2*x, evaluated at x=1,2,3.
        """
        env = standard_env()
        # Fitness function: score a candidate by how well it approximates 2*x
        eval_program("""
        (define fitness (lambda (f)
          (add (abs (subtract (f 1) 2))
               (add (abs (subtract (f 2) 4))
                    (abs (subtract (f 3) 6))))))
        """, env)

        spec = Spec(
            goal=GoalMinimize(Symbol("fitness"))
        )
        result = synthesize(spec, max_depth=2, max_candidates=50000, env=env)
        assert result.found
        assert result.fitness_score == 0.0  # should find exact match
        fn = eval_node(result.program, env)
        assert apply_fn(fn, [1.0]) == 2.0
        assert apply_fn(fn, [2.0]) == 4.0


class TestSynthesizeBuiltin:
    def test_synthesize_from_selph(self):
        """Call synthesize from within a SELPH program."""
        env = standard_env()
        # Create a spec as a namespace
        eval_program("""
        (define my-spec
          (:spec :type number
                 :goal (:examples ((1 -> 2) (3 -> 4) (5 -> 6)))))
        """, env)

        # Call synthesize builtin
        result_ns = eval_string("""
        (synthesize (namespace
          (spec my-spec)
          (max-depth 2)
          (max-candidates 10000)))
        """, env)

        assert isinstance(result_ns, Namespace)
        assert result_ns.get("found") is True
        assert result_ns.get("candidates") > 0


class TestMetaSynthesis:
    def test_heuristic_optimization(self):
        """Meta-test: can we score heuristics by their synthesis performance?

        This is the prerequisite for synthesizing better heuristics:
        we need a fitness function that evaluates a heuristic by running
        synthesis with it and measuring candidates explored.
        """
        env = standard_env()

        # Define a simple task
        eval_program("""
        (define test-spec
          (:spec :type number
                 :goal (:examples ((0 -> 1) (5 -> 6) (-1 -> 0)))))
        """, env)

        # A fitness function that scores a heuristic by candidates used
        eval_program("""
        (define score-heuristic (lambda (h)
          (let ((result (synthesize (namespace
                  (spec test-spec)
                  (max-depth 2)
                  (max-candidates 5000)
                  (heuristic h)))))
            (if (ns-get result "found")
              (ns-get result "candidates")
              5000))))
        """, env)

        # Score the recency heuristic
        h = eval_string("""
        (lambda (comp) (ns-get comp "insertion-order"))
        """, env)

        score = apply_fn(eval_string("score-heuristic", env), [h])
        assert isinstance(score, float)
        assert score > 0
        assert score <= 5000
        print(f"  Recency heuristic score: {score} candidates")
