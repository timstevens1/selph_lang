"""Tests for SELPH-native namespaces."""

import pytest
from selph.eval import eval_string, eval_program, standard_env, Namespace, EvalError


class TestNamespaceCreation:
    def test_empty_namespace(self):
        result = eval_string("(ns-empty)")
        assert isinstance(result, Namespace)
        assert len(result) == 0

    def test_namespace_literal(self):
        result = eval_string('(namespace (x 1) (y 2))')
        assert isinstance(result, Namespace)
        assert result.get("x") == 1.0
        assert result.get("y") == 2.0

    def test_nested_namespace(self):
        result = eval_string("""
        (namespace
          (math (namespace
            (double (lambda (x) (add x x)))
            (pi 3.14)))
          (greeting "hello"))
        """)
        assert isinstance(result, Namespace)
        assert result.get("greeting") == "hello"
        inner = result.get("math")
        assert isinstance(inner, Namespace)
        assert inner.get("pi") == 3.14

    def test_namespace_with_computed_values(self):
        result = eval_string('(namespace (sum (add 1 2)) (prod (multiply 3 4)))')
        assert result.get("sum") == 3.0
        assert result.get("prod") == 12.0


class TestNamespaceAccess:
    def test_ns_get(self):
        env = standard_env()
        eval_program('(define ns (namespace (x 42) (y 99)))', env)
        assert eval_string('(ns-get ns "x")', env) == 42.0

    def test_ns_get_nested(self):
        env = standard_env()
        eval_program("""
        (define ns (namespace
          (math (namespace
            (basic (namespace
              (double (lambda (x) (add x x)))))))))
        """, env)
        fn = eval_string('(ns-get ns "math" "basic" "double")', env)
        from selph.eval import apply_fn
        assert apply_fn(fn, [5.0]) == 10.0

    def test_ns_get_missing_key(self):
        env = standard_env()
        eval_program('(define ns (namespace (x 1)))', env)
        with pytest.raises(EvalError, match="no entry"):
            eval_string('(ns-get ns "missing")', env)

    def test_ns_keys(self):
        env = standard_env()
        eval_program('(define ns (namespace (a 1) (b 2) (c 3)))', env)
        keys = eval_string('(ns-keys ns)', env)
        assert set(keys) == {"a", "b", "c"}

    def test_ns_values(self):
        env = standard_env()
        eval_program('(define ns (namespace (a 1) (b 2)))', env)
        vals = eval_string('(ns-values ns)', env)
        assert set(vals) == {1.0, 2.0}

    def test_ns_size(self):
        env = standard_env()
        eval_program('(define ns (namespace (a 1) (b (namespace (c 2) (d 3)))))', env)
        # size counts leaves: a=1, c=1, d=1 = 3
        assert eval_string('(ns-size ns)', env) == 3.0


class TestNamespaceMutation:
    def test_ns_put(self):
        env = standard_env()
        eval_program('(define ns (namespace (x 1)))', env)
        result = eval_string('(ns-put ns "y" 2)', env)
        assert isinstance(result, Namespace)
        assert result.get("y") == 2.0
        # Original unchanged (immutable)
        original = env.lookup("ns")
        assert "y" not in original.entries

    def test_ns_put_path(self):
        env = standard_env()
        eval_program('(define ns (namespace (math (namespace (pi 3.14)))))', env)
        result = eval_string('(ns-put-path ns "math.e" 2.72)', env)
        assert result.get_path(["math", "e"]) == 2.72
        assert result.get_path(["math", "pi"]) == 3.14  # preserved


class TestNamespaceComposition:
    def test_ns_merge(self):
        env = standard_env()
        eval_program("""
        (define ns1 (namespace (a 1) (b 2)))
        (define ns2 (namespace (b 99) (c 3)))
        """, env)
        result = eval_string('(ns-merge ns1 ns2)', env)
        assert result.get("a") == 1.0
        assert result.get("b") == 99.0  # ns2 wins
        assert result.get("c") == 3.0

    def test_ns_merge_deep(self):
        env = standard_env()
        eval_program("""
        (define ns1 (namespace (math (namespace (add (lambda (x) (add x 1)))))))
        (define ns2 (namespace (math (namespace (mul (lambda (x) (multiply x 2)))))))
        """, env)
        result = eval_string('(ns-merge ns1 ns2)', env)
        math = result.get("math")
        assert "add" in math.entries
        assert "mul" in math.entries

    def test_ns_filter(self):
        env = standard_env()
        eval_program('(define ns (namespace (a 1) (b 2) (c 3) (d 4)))', env)
        result = eval_string('(ns-filter ns (lambda (v) (> v 2)))', env)
        assert "c" in result.entries
        assert "d" in result.entries
        assert "a" not in result.entries

    def test_ns_flatten(self):
        env = standard_env()
        eval_program("""
        (define ns (namespace
          (math (namespace (double 2) (triple 3)))
          (name "hello")))
        """, env)
        flat = eval_string('(ns-flatten ns)', env)
        assert isinstance(flat, dict)
        assert flat["math.double"] == 2.0
        assert flat["name"] == "hello"


class TestNamespacePredicate:
    def test_ns_check(self):
        env = standard_env()
        eval_program('(define ns (namespace (x 1)))', env)
        assert eval_string('(ns? ns)', env) is True
        assert eval_string('(ns? 42)', env) is False
        assert eval_string('(ns? "hello")', env) is False


class TestNamespaceWithFunctions:
    def test_call_function_from_namespace(self):
        """Retrieve a function from a namespace and call it."""
        env = standard_env()
        eval_program("""
        (define lib (namespace
          (double (lambda (x) (add x x)))
          (inc (lambda (x) (add x 1)))))
        """, env)
        assert eval_string('((ns-get lib "double") 5)', env) == 10.0
        assert eval_string('((ns-get lib "inc") 5)', env) == 6.0

    def test_compose_from_multiple_namespaces(self):
        """Use functions from different namespaces together."""
        env = standard_env()
        eval_program("""
        (define math-lib (namespace
          (double (lambda (x) (add x x)))))
        (define str-lib (namespace
          (shout (lambda (s) (string-upper s)))))
        """, env)
        assert eval_string('((ns-get math-lib "double") 5)', env) == 10.0
        assert eval_string('((ns-get str-lib "shout") "hello")', env) == "HELLO"

    def test_namespace_as_function_argument(self):
        """Pass a namespace to a function that queries it."""
        env = standard_env()
        eval_program("""
        (define apply-from-ns (lambda (ns key x)
          ((ns-get ns key) x)))
        (define lib (namespace
          (double (lambda (x) (add x x)))
          (negate (lambda (x) (subtract 0 x)))))
        """, env)
        assert eval_string('(apply-from-ns lib "double" 5)', env) == 10.0
        assert eval_string('(apply-from-ns lib "negate" 3)', env) == -3.0

    def test_synthesize_against_namespace(self):
        """Build a namespace of test cases, query from it."""
        env = standard_env()
        eval_program("""
        (define test-cases (namespace
          (case1 (namespace (input 5) (expected 10)))
          (case2 (namespace (input 3) (expected 6)))))
        """, env)
        # Get a test case and verify
        inp = eval_string('(ns-get test-cases "case1" "input")', env)
        expected = eval_string('(ns-get test-cases "case1" "expected")', env)
        assert inp == 5.0
        assert expected == 10.0


class TestMultipleNamespaceMerge:
    def test_merge_three_namespaces(self):
        """Merge multiple arbitrary namespace trees."""
        env = standard_env()
        eval_program("""
        (define ops (namespace
          (math (namespace (add (lambda (a b) (add a b)))))))
        (define data (namespace
          (constants (namespace (pi 3.14) (e 2.72)))))
        (define cache (namespace
          (recent (namespace (last-result 42)))))
        (define unified (ns-merge (ns-merge ops data) cache))
        """, env)
        assert eval_string('(ns-get unified "constants" "pi")', env) == 3.14
        assert eval_string('(ns-get unified "recent" "last-result")', env) == 42.0
        fn = eval_string('(ns-get unified "math" "add")', env)
        from selph.eval import apply_fn
        assert apply_fn(fn, [1.0, 2.0]) == 3.0
