"""Tests for the SELPH evaluator."""

import pytest
from selph.eval import eval_string, eval_program, standard_env, EvalError, Closure, Macro, verify_spec
from selph.parser import parse
from selph.ast import Spec, GoalExamples, GoalSatisfy, String, Number


# --- Arithmetic ---

class TestArithmetic:
    def test_add(self):
        assert eval_string("(add 1 2)") == 3.0

    def test_add_operator(self):
        assert eval_string("(+ 1 2)") == 3.0

    def test_subtract(self):
        assert eval_string("(subtract 10 3)") == 7.0

    def test_multiply(self):
        assert eval_string("(* 4 5)") == 20.0

    def test_divide(self):
        assert eval_string("(/ 10 4)") == 2.5

    def test_nested_arithmetic(self):
        assert eval_string("(+ (* 3 4) (- 10 5))") == 17.0

    def test_modulo(self):
        assert eval_string("(% 10 3)") == 1.0

    def test_negate(self):
        assert eval_string("(negate 5)") == -5.0

    def test_abs(self):
        assert eval_string("(abs -7)") == 7.0

    def test_min_max(self):
        assert eval_string("(min 3 7)") == 3.0
        assert eval_string("(max 3 7)") == 7.0

    def test_division_by_zero(self):
        with pytest.raises(EvalError, match="division by zero"):
            eval_string("(/ 1 0)")


# --- Comparison ---

class TestComparison:
    def test_equal(self):
        assert eval_string('(= 1 1)') is True
        assert eval_string('(= 1 2)') is False

    def test_not_equal(self):
        assert eval_string('(!= 1 2)') is True

    def test_less_than(self):
        assert eval_string('(< 1 2)') is True
        assert eval_string('(< 2 1)') is False

    def test_greater_than(self):
        assert eval_string('(> 5 3)') is True

    def test_string_equality(self):
        assert eval_string('(= "hello" "hello")') is True
        assert eval_string('(= "hello" "world")') is False


# --- Logic ---

class TestLogic:
    def test_not(self):
        assert eval_string("(not true)") is False
        assert eval_string("(not false)") is True

    def test_and(self):
        assert eval_string("(and true true)") is True
        assert eval_string("(and true false)") is False

    def test_or(self):
        assert eval_string("(or false true)") is True
        assert eval_string("(or false false)") is False

    def test_and_short_circuit(self):
        # Should not error because short-circuit prevents evaluating the second arg
        assert eval_string("(and false (/ 1 0))") is False

    def test_or_short_circuit(self):
        assert eval_string("(or true (/ 1 0))") is True

    def test_even_odd(self):
        assert eval_string("(even 4)") is True
        assert eval_string("(even 3)") is False
        assert eval_string("(odd 3)") is True


# --- Strings ---

class TestStrings:
    def test_concat(self):
        assert eval_string('(concat "hello" " " "world")') == "hello world"

    def test_string_length(self):
        assert eval_string('(string-length "hello")') == 5.0

    def test_string_upper(self):
        assert eval_string('(string-upper "hello")') == "HELLO"

    def test_string_lower(self):
        assert eval_string('(string-lower "HELLO")') == "hello"

    def test_string_reverse(self):
        assert eval_string('(string-reverse "abc")') == "cba"

    def test_string_contains(self):
        assert eval_string('(string-contains "hello world" "world")') is True
        assert eval_string('(string-contains "hello" "xyz")') is False

    def test_substring(self):
        assert eval_string('(substring "hello" 1 4)') == "ell"

    def test_string_split(self):
        assert eval_string('(string-split "a,b,c" ",")') == ["a", "b", "c"]

    def test_string_join(self):
        assert eval_string('(string-join (list "a" "b" "c") ",")') == "a,b,c"

    def test_char_at(self):
        assert eval_string('(char-at "hello" 0)') == "h"

    def test_string_replace(self):
        assert eval_string('(string-replace "hello world" "world" "SELPH")') == "hello SELPH"

    def test_string_trim(self):
        assert eval_string('(string-trim "  hi  ")') == "hi"


# --- Let ---

class TestLet:
    def test_simple_let(self):
        assert eval_string("(let ((x 5)) x)") == 5.0

    def test_let_with_computation(self):
        assert eval_string("(let ((x 5) (y 3)) (+ x y))") == 8.0

    def test_let_sequential_binding(self):
        # Second binding can see the first
        assert eval_string("(let ((x 5) (y (+ x 1))) y)") == 6.0

    def test_nested_let(self):
        assert eval_string("(let ((x 1)) (let ((y 2)) (+ x y)))") == 3.0

    def test_let_shadows(self):
        assert eval_string("(let ((x 1)) (let ((x 2)) x))") == 2.0


# --- Lambda ---

class TestLambda:
    def test_simple_lambda(self):
        result = eval_string("(lambda (x) (+ x 1))")
        assert isinstance(result, Closure)

    def test_lambda_application(self):
        assert eval_string("((lambda (x) (+ x 1)) 5)") == 6.0

    def test_lambda_two_args(self):
        assert eval_string("((lambda (a b) (+ a b)) 3 4)") == 7.0

    def test_lambda_closure(self):
        assert eval_string("(let ((y 10)) ((lambda (x) (+ x y)) 5))") == 15.0

    def test_lambda_in_let(self):
        source = """
        (let ((double (lambda (x) (* x 2))))
          (double 7))
        """
        assert eval_string(source) == 14.0

    def test_higher_order_lambda(self):
        source = """
        (let ((apply-twice (lambda (f x) (f (f x)))))
          (apply-twice (lambda (x) (+ x 1)) 0))
        """
        assert eval_string(source) == 2.0

    def test_recursive_via_let(self):
        # Factorial using sequential let bindings
        source = """
        (let ((fact (lambda (n)
                (if (<= n 1) 1
                    (* n (fact (- n 1)))))))
          (fact 5))
        """
        assert eval_string(source) == 120.0


# --- If ---

class TestIf:
    def test_if_true(self):
        assert eval_string("(if true 1 2)") == 1.0

    def test_if_false(self):
        assert eval_string("(if false 1 2)") == 2.0

    def test_if_no_else(self):
        assert eval_string("(if false 1)") is None

    def test_if_with_condition(self):
        assert eval_string("(if (> 5 3) 10 20)") == 10.0


# --- Define ---

class TestDefine:
    def test_define_value(self):
        env = standard_env()
        eval_string("(define x 42)", env)
        assert eval_string("x", env) == 42.0

    def test_define_function(self):
        env = standard_env()
        eval_string("(define double (lambda (x) (* x 2)))", env)
        assert eval_string("(double 5)", env) == 10.0


# --- Do ---

class TestDo:
    def test_do_returns_last(self):
        env = standard_env()
        result = eval_string("(do (define x 1) (define y 2) (+ x y))", env)
        assert result == 3.0


# --- Defmacro ---

class TestDefmacro:
    def test_defmacro_attention(self):
        """The attention macro from §2.4 (using mock values)."""
        env = standard_env()
        eval_program("""
        (defmacro attention (Q K V)
          (matmul (softmax (matmul Q (transpose K))) V))
        """, env)
        macro = env.lookup("attention")
        assert isinstance(macro, Macro)
        assert macro.params == ["Q", "K", "V"]

    def test_defmacro_simple(self):
        """A simple macro that expands to arithmetic."""
        env = standard_env()
        eval_program("""
        (defmacro square (x) (* x x))
        """, env)
        assert eval_string("(square 5)", env) == 25.0

    def test_defmacro_with_let(self):
        env = standard_env()
        eval_program("""
        (defmacro inc (x) (+ x 1))
        """, env)
        assert eval_string("(let ((a 10)) (inc a))", env) == 11.0


# --- List operations ---

class TestLists:
    def test_list_creation(self):
        assert eval_string("(list 1 2 3)") == [1.0, 2.0, 3.0]

    def test_head(self):
        assert eval_string("(head (list 1 2 3))") == 1.0

    def test_tail(self):
        assert eval_string("(tail (list 1 2 3))") == [2.0, 3.0]

    def test_length(self):
        assert eval_string("(length (list 1 2 3))") == 3.0

    def test_nth(self):
        assert eval_string("(nth (list 10 20 30) 1)") == 20.0

    def test_cons(self):
        assert eval_string("(cons 0 (list 1 2))") == [0.0, 1.0, 2.0]

    def test_append(self):
        assert eval_string("(append (list 1 2) (list 3 4))") == [1.0, 2.0, 3.0, 4.0]

    def test_reverse(self):
        assert eval_string("(reverse (list 1 2 3))") == [3.0, 2.0, 1.0]

    def test_range(self):
        assert eval_string("(range 5)") == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_range_start_end(self):
        assert eval_string("(range 2 5)") == [2.0, 3.0, 4.0]

    def test_sort(self):
        assert eval_string("(sort (list 3 1 2))") == [1.0, 2.0, 3.0]

    def test_empty(self):
        assert eval_string("(empty? (list))") is True
        assert eval_string("(empty? (list 1))") is False

    def test_contains(self):
        assert eval_string("(contains (list 1 2 3) 2)") is True
        assert eval_string("(contains (list 1 2 3) 5)") is False

    def test_empty_list_literal(self):
        assert eval_string("()") == []


# --- Higher-order functions ---

class TestHigherOrder:
    def test_map(self):
        assert eval_string("(map (lambda (x) (* x 2)) (list 1 2 3))") == [2.0, 4.0, 6.0]

    def test_filter(self):
        assert eval_string("(filter (lambda (x) (> x 2)) (list 1 2 3 4 5))") == [3.0, 4.0, 5.0]

    def test_reduce(self):
        assert eval_string("(reduce (lambda (acc x) (+ acc x)) (list 1 2 3 4) 0)") == 10.0

    def test_reduce_no_init(self):
        assert eval_string("(reduce (lambda (acc x) (+ acc x)) (list 1 2 3 4))") == 10.0

    def test_compose(self):
        source = """
        (let ((inc (lambda (x) (+ x 1)))
              (double (lambda (x) (* x 2)))
              (inc-then-double (compose double inc)))
          (inc-then-double 5))
        """
        assert eval_string(source) == 12.0

    def test_pipe(self):
        assert eval_string("(pipe 5 (lambda (x) (+ x 1)) (lambda (x) (* x 2)))") == 12.0

    def test_apply(self):
        assert eval_string("(apply + (list 3 4))") == 7.0


# --- Quote ---

class TestQuote:
    def test_quote_symbol(self):
        from selph.ast import Symbol
        result = eval_string("(quote hello)")
        assert result == Symbol("hello")

    def test_quote_list(self):
        from selph.ast import List, Symbol, Number
        result = eval_string("(quote (add 1 2))")
        assert isinstance(result, List)
        assert result.elements[0] == Symbol("add")


# --- Type predicates ---

class TestTypePredicates:
    def test_number(self):
        assert eval_string("(number? 5)") is True
        assert eval_string('(number? "hi")') is False

    def test_string(self):
        assert eval_string('(string? "hi")') is True
        assert eval_string("(string? 5)") is False

    def test_bool(self):
        assert eval_string("(bool? true)") is True
        assert eval_string("(bool? 1)") is False

    def test_list(self):
        assert eval_string("(list? (list 1 2))") is True
        assert eval_string("(list? 5)") is False

    def test_nil(self):
        assert eval_string("(nil? nil)") is True
        assert eval_string("(nil? 0)") is False

    def test_callable(self):
        assert eval_string("(callable? (lambda (x) x))") is True
        assert eval_string("(callable? +)") is True
        assert eval_string("(callable? 5)") is False


# --- Spec verification ---

class TestSpecVerification:
    def test_examples_pass(self):
        """Level 0: verify a function against input/output examples."""
        env = standard_env()
        fn = eval_string("(lambda (x) (string-upper x))", env)
        spec = Spec(
            goal=GoalExamples(pairs=(
                (String("hello"), String("HELLO")),
                (String("world"), String("WORLD")),
            ))
        )
        result = verify_spec(spec, fn, env)
        assert result["passed"] is True
        assert result["total"] == 2

    def test_examples_fail(self):
        """Level 0: failing examples."""
        env = standard_env()
        fn = eval_string("(lambda (x) x)", env)  # identity, not upper
        spec = Spec(
            goal=GoalExamples(pairs=(
                (String("hello"), String("HELLO")),
            ))
        )
        result = verify_spec(spec, fn, env)
        assert result["passed"] is False
        assert len(result["failures"]) == 1

    def test_no_goal(self):
        """Spec with no goal passes trivially."""
        env = standard_env()
        fn = eval_string("(lambda (x) x)", env)
        spec = Spec()
        result = verify_spec(spec, fn, env)
        assert result["passed"] is True


# --- Errors ---

class TestErrors:
    def test_unbound_symbol(self):
        with pytest.raises(EvalError, match="unbound symbol"):
            eval_string("undefined_var")

    def test_not_callable(self):
        with pytest.raises(EvalError, match="not callable"):
            eval_string("(5 1 2)")

    def test_arity_mismatch(self):
        with pytest.raises(EvalError):
            eval_string("((lambda (x y) (+ x y)) 1)")


# --- Integration: spec example programs ---

class TestSpecPrograms:
    def test_neural_layer_structure(self):
        """§2.4 neural layer — just verify it parses and the structure evaluates
        when we mock the tensor ops as identity functions."""
        env = standard_env()
        # Mock tensor ops as simple arithmetic for structural testing
        eval_program("""
        (define W 2)
        (define x 3)
        (define b 1)
        (define matmul *)
        (define relu (lambda (x) (if (> x 0) x 0)))
        (define dropout (lambda (x rate) x))
        """, env)
        result = eval_string("(let ((h (relu (add (matmul W x) b)))) (dropout h 0.1))", env)
        assert result == 7.0  # relu(2*3 + 1) = 7

    def test_fibonacci(self):
        source = """
        (let ((fib (lambda (n)
                (if (<= n 1) n
                    (+ (fib (- n 1)) (fib (- n 2)))))))
          (fib 10))
        """
        assert eval_string(source) == 55.0

    def test_map_filter_reduce_pipeline(self):
        """Compose higher-order functions in a pipeline."""
        source = """
        (let ((nums (range 1 11))
              (evens (filter even nums))
              (doubled (map (lambda (x) (* x 2)) evens))
              (total (reduce + doubled 0)))
          total)
        """
        assert eval_string(source) == 60.0  # 2+4+6+8+10 = 30, doubled = 60

    def test_string_processing_pipeline(self):
        source = """
        (let ((words (string-split "hello world foo bar" " "))
              (uppered (map string-upper words))
              (result (string-join uppered ", ")))
          result)
        """
        assert eval_string(source) == "HELLO, WORLD, FOO, BAR"
