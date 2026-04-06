"""Tests for the SELPH type checker."""

import pytest
from selph.types import (
    typecheck, typecheck_string, TypeCheckError,
    TNum, TStr, TBool, TNil, TList, TFn, TVar, TSpec, TGoal, TTensor,
    DimConst, DimSym,
    InferenceState, standard_type_env, TypeEnv,
)
from selph.parser import parse


def resolve_type(source: str) -> str:
    """Typecheck and return the resolved type as a string."""
    state = InferenceState()
    tenv = standard_type_env()
    node = parse(source)
    t = typecheck(node, tenv, state)
    return repr(state.resolve(t))


# --- Literal types ---

class TestLiterals:
    def test_number(self):
        assert resolve_type("42") == "number"

    def test_float(self):
        assert resolve_type("3.14") == "number"

    def test_string(self):
        assert resolve_type('"hello"') == "string"

    def test_bool_true(self):
        assert resolve_type("true") == "bool"

    def test_bool_false(self):
        assert resolve_type("false") == "bool"

    def test_tensor_literal(self):
        t = resolve_type("#T[3, 4] float32")
        assert "tensor" in t
        assert "3" in t
        assert "4" in t


# --- Arithmetic ---

class TestArithmetic:
    def test_add(self):
        assert resolve_type("(+ 1 2)") == "number"

    def test_subtract(self):
        assert resolve_type("(- 10 3)") == "number"

    def test_nested(self):
        assert resolve_type("(+ (* 3 4) (- 10 5))") == "number"

    def test_add_string_error(self):
        """Adding a string to a number should fail."""
        with pytest.raises(TypeCheckError):
            typecheck_string('(+ 1 "hello")')

    def test_comparison_returns_bool(self):
        assert resolve_type("(< 1 2)") == "bool"

    def test_comparison_type_error(self):
        with pytest.raises(TypeCheckError):
            typecheck_string('(< 1 "hello")')


# --- Logic ---

class TestLogic:
    def test_not(self):
        assert resolve_type("(not true)") == "bool"

    def test_not_type_error(self):
        with pytest.raises(TypeCheckError):
            typecheck_string("(not 5)")

    def test_even(self):
        assert resolve_type("(even 4)") == "bool"

    def test_and(self):
        assert resolve_type("(and true false)") == "bool"


# --- Strings ---

class TestStrings:
    def test_concat(self):
        assert resolve_type('(concat "a" "b")') == "string"

    def test_string_length(self):
        assert resolve_type('(string-length "hi")') == "number"

    def test_string_upper(self):
        assert resolve_type('(string-upper "hi")') == "string"

    def test_string_type_error(self):
        with pytest.raises(TypeCheckError):
            typecheck_string("(string-upper 5)")

    def test_string_split(self):
        t = resolve_type('(string-split "a,b" ",")')
        assert t == "(list string)"


# --- Let ---

class TestLet:
    def test_simple(self):
        assert resolve_type("(let ((x 5)) x)") == "number"

    def test_with_computation(self):
        assert resolve_type("(let ((x 5) (y 3)) (+ x y))") == "number"

    def test_type_propagation(self):
        assert resolve_type('(let ((s "hi")) (string-length s))') == "number"

    def test_sequential_binding(self):
        assert resolve_type("(let ((x 5) (y (+ x 1))) y)") == "number"


# --- Lambda ---

class TestLambda:
    def test_lambda_type(self):
        t = resolve_type("(lambda (x) (+ x 1))")
        assert "(-> number number)" == t

    def test_lambda_application(self):
        assert resolve_type("((lambda (x) (+ x 1)) 5)") == "number"

    def test_lambda_arg_type_error(self):
        """Passing a string to a function expecting number."""
        with pytest.raises(TypeCheckError):
            typecheck_string('((lambda (x) (+ x 1)) "hi")')

    def test_lambda_string(self):
        t = resolve_type("(lambda (s) (string-upper s))")
        assert "(-> string string)" == t

    def test_higher_order(self):
        source = """
        (let ((apply-twice (lambda (f x) (f (f x)))))
          (apply-twice (lambda (x) (+ x 1)) 0))
        """
        assert resolve_type(source) == "number"


# --- If ---

class TestIf:
    def test_if_type(self):
        assert resolve_type("(if true 1 2)") == "number"

    def test_if_cond_must_be_bool(self):
        with pytest.raises(TypeCheckError):
            typecheck_string("(if 5 1 2)")

    def test_if_branches_must_match(self):
        with pytest.raises(TypeCheckError):
            typecheck_string('(if true 1 "hello")')


# --- List operations ---

class TestListOps:
    def test_list_creation(self):
        assert resolve_type("(list 1 2 3)") == "(list number)"

    def test_list_of_strings(self):
        assert resolve_type('(list "a" "b")') == "(list string)"

    def test_head(self):
        assert resolve_type("(head (list 1 2 3))") == "number"

    def test_tail(self):
        assert resolve_type("(tail (list 1 2 3))") == "(list number)"

    def test_length(self):
        assert resolve_type("(length (list 1 2 3))") == "number"

    def test_map(self):
        assert resolve_type("(map (lambda (x) (* x 2)) (list 1 2 3))") == "(list number)"

    def test_map_type_change(self):
        assert resolve_type('(map (lambda (x) (to-string x)) (list 1 2))') == "(list string)"

    def test_filter(self):
        assert resolve_type("(filter (lambda (x) (> x 2)) (list 1 2 3))") == "(list number)"

    def test_reduce(self):
        assert resolve_type("(reduce (lambda (acc x) (+ acc x)) (list 1 2 3) 0)") == "number"

    def test_sort(self):
        assert resolve_type("(sort (list 3 1 2))") == "(list number)"

    def test_append(self):
        assert resolve_type("(append (list 1) (list 2))") == "(list number)"

    def test_contains(self):
        assert resolve_type("(contains (list 1 2 3) 1)") == "bool"

    def test_range(self):
        assert resolve_type("(range 5)") == "(list number)"


# --- Polymorphic equality ---

class TestEquality:
    def test_num_equality(self):
        assert resolve_type("(= 1 2)") == "bool"

    def test_string_equality(self):
        assert resolve_type('(= "a" "b")') == "bool"

    def test_mixed_equality_error(self):
        with pytest.raises(TypeCheckError):
            typecheck_string('(= 1 "hello")')


# --- Define ---

class TestDefine:
    def test_define(self):
        state = InferenceState()
        tenv = standard_type_env()
        node = parse("(define x 42)")
        typecheck(node, tenv, state)
        assert repr(state.resolve(tenv.lookup("x"))) == "number"


# --- Recursive functions ---

class TestRecursion:
    def test_factorial(self):
        source = """
        (let ((fact (lambda (n)
                (if (<= n 1) 1
                    (* n (fact (- n 1)))))))
          (fact 5))
        """
        assert resolve_type(source) == "number"


# --- Spec and goal types ---

class TestSpecTypes:
    def test_spec_literal(self):
        t = resolve_type('(:spec :type string :goal (:intent "hello"))')
        assert t == "spec"

    def test_goal_examples(self):
        t = resolve_type('(:examples (("a" -> "b")))')
        assert t == "goal"


# --- Integration: complex programs ---

class TestIntegration:
    def test_map_filter_reduce(self):
        source = """
        (let ((nums (range 1 11))
              (evens (filter even nums))
              (doubled (map (lambda (x) (* x 2)) evens))
              (total (reduce + doubled 0)))
          total)
        """
        assert resolve_type(source) == "number"

    def test_string_pipeline(self):
        source = """
        (let ((words (string-split "hello world" " "))
              (uppered (map string-upper words))
              (result (string-join uppered ", ")))
          result)
        """
        assert resolve_type(source) == "string"

    def test_lambda_in_let(self):
        source = """
        (let ((double (lambda (x) (* x 2))))
          (double 7))
        """
        assert resolve_type(source) == "number"

    def test_compose(self):
        source = """
        (let ((inc (lambda (x) (+ x 1)))
              (double (lambda (x) (* x 2)))
              (f (compose double inc)))
          (f 5))
        """
        assert resolve_type(source) == "number"
