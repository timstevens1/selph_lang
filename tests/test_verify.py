"""Tests for the SELPH spec verification system."""

import pytest
from selph.verify import (
    verify, verify_fn, verify_from_source, VerificationResult,
)
from selph.eval import eval_string, eval_program, standard_env, Closure, BuiltinFn
from selph.parser import parse
from selph.ast import (
    Spec, GoalExamples, GoalPattern, GoalSatisfy, GoalAll, GoalIntent,
    Symbol, Number, String, Bool, List, Shape,
)


# ── Type matching ────────────────────────────────────────────────────

class TestTypeMatch:
    def test_string_type_match(self):
        spec = Spec(type_expr=Symbol("string"))
        result = verify(spec, "hello")
        assert result.type_match is True
        assert result.passed is True

    def test_string_type_mismatch(self):
        spec = Spec(type_expr=Symbol("string"))
        result = verify(spec, 42.0)
        assert result.type_match is False
        assert result.hard_gate is False
        assert "expected type string" in result.type_error

    def test_number_type_match(self):
        spec = Spec(type_expr=Symbol("number"))
        result = verify(spec, 3.14)
        assert result.type_match is True

    def test_bool_type_match(self):
        spec = Spec(type_expr=Symbol("bool"))
        result = verify(spec, True)
        assert result.type_match is True

    def test_list_type_match(self):
        spec = Spec(type_expr=Symbol("list"))
        result = verify(spec, [1, 2, 3])
        assert result.type_match is True

    def test_nil_type_match(self):
        spec = Spec(type_expr=Symbol("nil"))
        result = verify(spec, None)
        assert result.type_match is True

    def test_no_type_spec_passes(self):
        spec = Spec()
        result = verify(spec, "anything")
        assert result.type_match is True


# ── Shape matching ───────────────────────────────────────────────────

class TestShapeMatch:
    def test_1d_shape(self):
        spec = Spec(shape_expr=Shape((3,)))
        result = verify(spec, [1, 2, 3])
        assert result.shape_match is True

    def test_1d_shape_mismatch(self):
        spec = Spec(shape_expr=Shape((3,)))
        result = verify(spec, [1, 2])
        assert result.shape_match is False

    def test_2d_shape(self):
        spec = Spec(shape_expr=Shape((2, 3)))
        result = verify(spec, [[1, 2, 3], [4, 5, 6]])
        assert result.shape_match is True

    def test_2d_shape_inner_mismatch(self):
        spec = Spec(shape_expr=Shape((2, 3)))
        result = verify(spec, [[1, 2], [3, 4]])
        assert result.shape_match is False

    def test_symbolic_dim_passes(self):
        """Symbolic dims like 'batch' are not checked at runtime."""
        spec = Spec(shape_expr=Shape(("batch", 3)))
        result = verify(spec, [[1, 2, 3], [4, 5, 6]])
        assert result.shape_match is True

    def test_non_list_shape_fails(self):
        spec = Spec(shape_expr=Shape((3,)))
        result = verify(spec, "not a list")
        assert result.shape_match is False


# ── Constraint checking ──────────────────────────────────────────────

class TestConstraints:
    def test_max_length_pass(self):
        spec = Spec(constraints=List((Symbol("max-length"), Number(10.0))))
        result = verify(spec, "hello")
        assert result.constraint_match is True

    def test_max_length_fail(self):
        spec = Spec(constraints=List((Symbol("max-length"), Number(3.0))))
        result = verify(spec, "hello")
        assert result.constraint_match is False
        assert any("max-length" in e for e in result.constraint_errors)

    def test_min_length_pass(self):
        spec = Spec(constraints=List((Symbol("min-length"), Number(3.0))))
        result = verify(spec, "hello")
        assert result.constraint_match is True

    def test_min_length_fail(self):
        spec = Spec(constraints=List((Symbol("min-length"), Number(10.0))))
        result = verify(spec, "hi")
        assert result.constraint_match is False

    def test_non_empty_pass(self):
        spec = Spec(constraints=List((Symbol("non-empty"),)))
        result = verify(spec, "hello")
        assert result.constraint_match is True

    def test_non_empty_fail(self):
        spec = Spec(constraints=List((Symbol("non-empty"),)))
        result = verify(spec, "")
        assert result.constraint_match is False

    def test_one_of_pass(self):
        spec = Spec(constraints=List((Symbol("one-of"), String("a"), String("b"), String("c"))))
        result = verify(spec, "b")
        assert result.constraint_match is True

    def test_one_of_fail(self):
        spec = Spec(constraints=List((Symbol("one-of"), String("a"), String("b"))))
        result = verify(spec, "z")
        assert result.constraint_match is False

    def test_matches_regex_pass(self):
        spec = Spec(constraints=List((Symbol("matches"), String(r"[a-z]+"))))
        result = verify(spec, "hello")
        assert result.constraint_match is True

    def test_matches_regex_fail(self):
        spec = Spec(constraints=List((Symbol("matches"), String(r"[0-9]+"))))
        result = verify(spec, "hello")
        assert result.constraint_match is False

    def test_lambda_constraint(self):
        """Constraint as a lambda predicate."""
        env = standard_env()
        spec = Spec(constraints=List((Symbol("lambda"), List((Symbol("x"),)),
                                      List((Symbol(">"), Symbol("x"), Number(0.0))))))
        result = verify(spec, 5.0, env)
        assert result.constraint_match is True

    def test_lambda_constraint_fail(self):
        env = standard_env()
        spec = Spec(constraints=List((Symbol("lambda"), List((Symbol("x"),)),
                                      List((Symbol(">"), Symbol("x"), Number(0.0))))))
        result = verify(spec, -1.0, env)
        assert result.constraint_match is False


# ── Level 0: Goal Examples ───────────────────────────────────────────

class TestGoalExamples:
    def test_all_pass(self):
        env = standard_env()
        fn = eval_string("(lambda (x) (string-upper x))", env)
        spec = Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("hello"), String("HELLO")),
                (String("world"), String("WORLD")),
                (String("foo"), String("FOO")),
            ))
        )
        result = verify_fn(spec, fn, env)
        assert result.goal_score == 1.0
        assert result.goal_details["passed"] == 3
        assert result.goal_details["total"] == 3

    def test_partial_pass(self):
        """A function that only works for some inputs."""
        env = standard_env()
        fn = eval_string('(lambda (x) (if (= x "a") "A" "?"))', env)
        spec = Spec(
            goal=GoalExamples(pairs=(
                (String("a"), String("A")),
                (String("b"), String("B")),
            ))
        )
        result = verify_fn(spec, fn, env)
        assert result.goal_score == 0.5
        assert result.goal_details["passed"] == 1
        assert len(result.goal_details["failures"]) == 1

    def test_all_fail(self):
        env = standard_env()
        fn = eval_string("(lambda (x) x)", env)  # identity, not upper
        spec = Spec(
            goal=GoalExamples(pairs=(
                (String("hello"), String("HELLO")),
            ))
        )
        result = verify_fn(spec, fn, env)
        assert result.goal_score == 0.0

    def test_error_in_function(self):
        """Function that throws on some inputs."""
        env = standard_env()
        fn = eval_string("(lambda (x) (/ 1 x))", env)
        spec = Spec(
            goal=GoalExamples(pairs=(
                (Number(2.0), Number(0.5)),
                (Number(0.0), Number(0.0)),  # will error
            ))
        )
        result = verify_fn(spec, fn, env)
        assert result.goal_score == 0.5
        assert any("error" in f for f in result.goal_details["failures"])

    def test_value_examples_match(self):
        """Verify a single value matches one of the expected outputs."""
        spec = Spec(goal=GoalExamples(pairs=(
            (String("_"), String("HELLO")),
            (String("_"), String("WORLD")),
        )))
        result = verify(spec, "HELLO")
        assert result.goal_score == 1.0

    def test_value_examples_no_match(self):
        spec = Spec(goal=GoalExamples(pairs=(
            (String("_"), String("HELLO")),
        )))
        result = verify(spec, "nope")
        assert result.goal_score == 0.0


# ── Level 1: Goal Patterns ──────────────────────────────────────────

class TestGoalPatterns:
    def test_length_constraint(self):
        goal = GoalPattern(List((Symbol("word"), Symbol(":length"), Number(5.0))))
        spec = Spec(goal=goal)
        result = verify(spec, "hello")
        assert result.goal_score == 1.0

    def test_length_constraint_fail(self):
        goal = GoalPattern(List((Symbol("word"), Symbol(":length"), Number(3.0))))
        spec = Spec(goal=goal)
        result = verify(spec, "hello")
        assert result.goal_score == 0.0

    def test_starts_with(self):
        goal = GoalPattern(List((Symbol("word"), Symbol(":starts-with"), String("he"))))
        spec = Spec(goal=goal)
        result = verify(spec, "hello")
        assert result.goal_score == 1.0

    def test_starts_with_fail(self):
        goal = GoalPattern(List((Symbol("word"), Symbol(":starts-with"), String("wo"))))
        spec = Spec(goal=goal)
        result = verify(spec, "hello")
        assert result.goal_score == 0.0

    def test_multiple_pattern_constraints(self):
        """Pattern with multiple constraints."""
        goal = GoalPattern(List((
            Symbol("word"),
            Symbol(":length"), Number(5.0),
            Symbol(":starts-with"), String("h"),
        )))
        spec = Spec(goal=goal)
        result = verify(spec, "hello")
        assert result.goal_score == 1.0
        assert result.goal_details["checks_passed"] == 2

    def test_partial_pattern_match(self):
        """One constraint passes, one fails — fractional score."""
        goal = GoalPattern(List((
            Symbol("word"),
            Symbol(":length"), Number(3.0),
            Symbol(":starts-with"), String("h"),
        )))
        spec = Spec(goal=goal)
        result = verify(spec, "hello")
        assert result.goal_score == 0.5  # starts-with passes, length fails

    def test_ends_with(self):
        goal = GoalPattern(List((Symbol("word"), Symbol(":ends-with"), String("lo"))))
        spec = Spec(goal=goal)
        result = verify(spec, "hello")
        assert result.goal_score == 1.0

    def test_contains(self):
        goal = GoalPattern(List((Symbol("word"), Symbol(":contains"), String("ell"))))
        spec = Spec(goal=goal)
        result = verify(spec, "hello")
        assert result.goal_score == 1.0

    def test_numeric_min_max(self):
        goal = GoalPattern(List((
            Symbol("number"),
            Symbol(":min"), Number(0.0),
            Symbol(":max"), Number(10.0),
        )))
        spec = Spec(goal=goal)
        assert verify(spec, 5.0).goal_score == 1.0
        assert verify(spec, 15.0).goal_score == 0.5  # min passes, max fails
        assert verify(spec, -1.0).goal_score == 0.5  # max passes, min fails


# ── Level 2: Goal Satisfy (predicates) ───────────────────────────────

class TestGoalSatisfy:
    def test_simple_predicate_pass(self):
        env = standard_env()
        goal = GoalSatisfy(parse("(lambda (out) (> out 0))"))
        spec = Spec(goal=goal)
        result = verify(spec, 5.0, env)
        assert result.goal_score == 1.0

    def test_simple_predicate_fail(self):
        env = standard_env()
        goal = GoalSatisfy(parse("(lambda (out) (> out 0))"))
        spec = Spec(goal=goal)
        result = verify(spec, -3.0, env)
        assert result.goal_score == 0.0

    def test_compound_predicate(self):
        """Predicate with multiple conditions from §3.4."""
        env = standard_env()
        goal = GoalSatisfy(parse("(lambda (out) (and (> out 0) (even out)))"))
        spec = Spec(goal=goal)
        assert verify(spec, 4.0, env).goal_score == 1.0
        assert verify(spec, 3.0, env).goal_score == 0.0   # odd
        assert verify(spec, -2.0, env).goal_score == 0.0   # negative

    def test_string_predicate(self):
        env = standard_env()
        goal = GoalSatisfy(parse('(lambda (out) (string-contains out "hello"))'))
        spec = Spec(goal=goal)
        assert verify(spec, "say hello world", env).goal_score == 1.0
        assert verify(spec, "goodbye", env).goal_score == 0.0

    def test_predicate_error(self):
        """Predicate that errors on the given input."""
        env = standard_env()
        goal = GoalSatisfy(parse("(lambda (out) (/ 1 out))"))
        spec = Spec(goal=goal)
        result = verify(spec, 0.0, env)
        assert result.goal_score == 0.0
        assert "error" in result.goal_details


# ── Goal composition (:all) ────────────────────────────���─────────────

class TestGoalAll:
    def test_all_pass(self):
        env = standard_env()
        goal = GoalAll((
            GoalSatisfy(parse("(lambda (out) (> out 0))")),
            GoalSatisfy(parse("(lambda (out) (< out 100))")),
        ))
        spec = Spec(goal=goal)
        result = verify(spec, 50.0, env)
        assert result.goal_score == 1.0

    def test_partial_pass(self):
        env = standard_env()
        goal = GoalAll((
            GoalSatisfy(parse("(lambda (out) (> out 0))")),
            GoalSatisfy(parse("(lambda (out) (< out 10))")),
        ))
        spec = Spec(goal=goal)
        result = verify(spec, 50.0, env)
        assert result.goal_score == 0.5  # first passes, second fails

    def test_all_fail(self):
        env = standard_env()
        goal = GoalAll((
            GoalSatisfy(parse("(lambda (out) (> out 100))")),
            GoalSatisfy(parse("(lambda (out) (< out 0))")),
        ))
        spec = Spec(goal=goal)
        result = verify(spec, 50.0, env)
        assert result.goal_score == 0.0

    def test_mixed_goal_types(self):
        """Compose different goal levels."""
        env = standard_env()
        goal = GoalAll((
            GoalSatisfy(parse("(lambda (out) (> (string-length out) 3))")),
            GoalPattern(List((Symbol("word"), Symbol(":starts-with"), String("h")))),
        ))
        spec = Spec(goal=goal)
        result = verify(spec, "hello", env)
        assert result.goal_score == 1.0


# ── Reward computation ───────────────────────────────────────────────

class TestReward:
    def test_full_pass_reward(self):
        result = VerificationResult(
            type_match=True, shape_match=True, constraint_match=True,
            goal_score=1.0,
        )
        assert result.reward() == pytest.approx(0.7)  # w_goal=0.7, rest 0

    def test_hard_gate_zero(self):
        """Type failure gates reward to zero."""
        result = VerificationResult(
            type_match=False, shape_match=True, constraint_match=True,
            goal_score=1.0,
        )
        assert result.reward() == 0.0

    def test_partial_goal_score(self):
        result = VerificationResult(goal_score=0.5)
        assert result.reward() == pytest.approx(0.35)

    def test_with_parent_and_global(self):
        result = VerificationResult(goal_score=1.0)
        r = result.reward(parent_success=1.0, global_outcome=1.0)
        assert r == pytest.approx(1.0)  # 0.7 + 0.2 + 0.1

    def test_custom_weights(self):
        result = VerificationResult(goal_score=1.0)
        r = result.reward(w_goal=1.0, w_parent=0.0, w_global=0.0)
        assert r == pytest.approx(1.0)


# ── Hard gate interaction ────────────────────────────────────────────

class TestHardGate:
    def test_type_failure_gates_everything(self):
        env = standard_env()
        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalSatisfy(parse("(lambda (out) true)")),
        )
        result = verify(spec, "not a number", env)
        assert result.type_match is False
        assert result.goal_score == 1.0  # goal passes but...
        assert result.reward() == 0.0   # ...hard gate kills reward

    def test_shape_failure_gates_everything(self):
        spec = Spec(
            shape_expr=Shape((3,)),
            goal=GoalSatisfy(parse("(lambda (out) true)")),
        )
        env = standard_env()
        result = verify(spec, [1, 2], env)  # wrong shape
        assert result.shape_match is False
        assert result.reward() == 0.0

    def test_constraint_failure_gates_everything(self):
        spec = Spec(
            constraints=List((Symbol("max-length"), Number(3.0))),
            goal=GoalSatisfy(parse("(lambda (out) true)")),
        )
        env = standard_env()
        result = verify(spec, "too long string", env)
        assert result.constraint_match is False
        assert result.reward() == 0.0


# ── Full pipeline: verify_from_source ────────────────────────────────

class TestVerifyFromSource:
    def test_function_with_examples(self):
        result = verify_from_source(
            '(:spec :type string :goal (:examples (("hello" -> "HELLO") ("world" -> "WORLD"))))',
            '(lambda (x) (string-upper x))',
        )
        assert result.goal_score == 1.0
        assert result.passed

    def test_function_with_failing_examples(self):
        result = verify_from_source(
            '(:spec :type string :goal (:examples (("hello" -> "HELLO"))))',
            '(lambda (x) x)',
        )
        assert result.goal_score == 0.0

    def test_value_with_predicate(self):
        result = verify_from_source(
            '(:spec :type number :goal (:satisfy (lambda (out) (> out 0))))',
            '5',
        )
        assert result.goal_score == 1.0
        assert result.type_match is True

    def test_value_type_mismatch(self):
        result = verify_from_source(
            '(:spec :type number)',
            '"hello"',
        )
        assert result.type_match is False

    def test_spec_from_architecture_doc(self):
        """From §12.10: spec as training data."""
        result = verify_from_source(
            '(:spec :type string :goal (:examples (("hello" -> "HELLO") ("world" -> "WORLD"))))',
            '(lambda (x) (string-upper x))',
        )
        assert result.passed
        assert result.goal_details["total"] == 2
        assert result.goal_details["passed"] == 2

    def test_subagent_constraint_spec(self):
        """From §7.2: subagent with constraints."""
        result = verify_from_source(
            '(:spec :type string :constraints (max-length 200))',
            '"a short summary"',
        )
        assert result.passed

    def test_subagent_constraint_violated(self):
        result = verify_from_source(
            '(:spec :type string :constraints (max-length 5))',
            '"this string is way too long"',
        )
        assert result.constraint_match is False
        assert result.reward() == 0.0


# ── Level 4+ (unsupported) ──────────────────────────────────────────

class TestUnsupportedGoals:
    def test_intent_goal_not_verified(self):
        """Level 4+ goals can't be mechanically verified."""
        spec = Spec(goal=GoalIntent("summarize the document"))
        result = verify(spec, "some summary")
        assert result.goal_score == 0.0
        assert "requires model-as-judge" in result.goal_details["message"]
