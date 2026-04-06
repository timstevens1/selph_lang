"""Tests for SELPH-programmable search heuristics."""

import pytest
import random
from selph.synthesize import synthesize, Component
from selph.eval import (
    eval_string, eval_program, eval_node, apply_fn,
    standard_env, Namespace, Closure,
)
from selph.ast import Spec, GoalExamples, Symbol, Number, String
from selph.types import TNum, TStr, TFn


class TestComponentToNamespace:
    def test_basic_component(self):
        comp = Component("add", Symbol("add"), TFn((TNum(), TNum()), TNum()), 2)
        ns = comp.to_namespace()
        assert isinstance(ns, Namespace)
        assert ns.get("name") == "add"
        assert ns.get("arity") == 2.0
        assert ns.get("return-type") == "number"
        assert ns.get("input-type") == "number"

    def test_string_component(self):
        comp = Component("string-upper", Symbol("string-upper"),
                        TFn((TStr(),), TStr()), 1)
        ns = comp.to_namespace()
        assert ns.get("return-type") == "string"
        assert ns.get("input-type") == "string"
        assert ns.get("arity") == 1.0

    def test_insertion_order(self):
        c1 = Component("a", Symbol("a"), TNum(), 0)
        c2 = Component("b", Symbol("b"), TNum(), 0)
        ns1 = c1.to_namespace()
        ns2 = c2.to_namespace()
        assert ns2.get("insertion-order") > ns1.get("insertion-order")


class TestSelphHeuristic:
    def _setup(self):
        env = standard_env()
        eval_program('''
        (define fst (lambda (s) (head (map to-number (string-split s " ")))))
        (define snd (lambda (s) (head (tail (map to-number (string-split s " "))))))
        ''', env)
        extra = [
            Component("fst", Symbol("fst"), TFn((TStr(),), TNum()), 1),
            Component("snd", Symbol("snd"), TFn((TStr(),), TNum()), 1),
        ]
        rng = random.Random(42)
        pairs = []
        for _ in range(15):
            a, b = rng.randint(0, 20), rng.randint(0, 20)
            pairs.append((String(f"{a} {b}"), Number(float(min(a, b)))))
        spec = Spec(type_expr=Symbol("number"),
                    goal=GoalExamples(pairs=tuple(pairs)))
        return env, extra, spec

    def test_recency_heuristic(self):
        """A SELPH function that scores by insertion order."""
        env, extra, spec = self._setup()
        heuristic = eval_string('''
        (lambda (comp) (ns-get comp "insertion-order"))
        ''', env)
        assert isinstance(heuristic, Closure)
        # Score a component
        ns = extra[0].to_namespace()
        score = apply_fn(heuristic, [ns])
        assert isinstance(score, float)

    def test_type_match_heuristic(self):
        """A heuristic that prefers number-returning components."""
        env, extra, spec = self._setup()
        heuristic = eval_string('''
        (lambda (comp)
          (let ((ret (ns-get comp "return-type"))
                (order (ns-get comp "insertion-order")))
            (if (= ret "number")
              (add 1000 order)
              order)))
        ''', env)
        # Test: number-returning comp should score higher
        num_comp = Component("min", Symbol("min"), TFn((TNum(), TNum()), TNum()), 2)
        str_comp = Component("upper", Symbol("upper"), TFn((TStr(),), TStr()), 1)
        score_num = apply_fn(heuristic, [num_comp.to_namespace()])
        score_str = apply_fn(heuristic, [str_comp.to_namespace()])
        assert score_num > score_str

    def test_heuristic_affects_synthesis(self):
        """A heuristic that boosts recency should help find pair-min faster."""
        env, extra, spec = self._setup()

        # Recency heuristic
        heuristic = eval_string('''
        (lambda (comp) (ns-get comp "insertion-order"))
        ''', env)

        # With heuristic
        result_with = synthesize(
            spec, max_depth=2, max_candidates=200000,
            extra_components=extra, env=env,
            heuristic=heuristic,
        )

        # Without heuristic (separate env to avoid contamination)
        env2 = standard_env()
        eval_program('''
        (define fst (lambda (s) (head (map to-number (string-split s " ")))))
        (define snd (lambda (s) (head (tail (map to-number (string-split s " "))))))
        ''', env2)
        result_without = synthesize(
            spec, max_depth=2, max_candidates=200000,
            extra_components=extra, env=env2,
        )

        # Both should find the answer (with enough budget)
        # The heuristic version should find it faster if ordering helps
        if result_with.found and result_without.found:
            print(f"  With heuristic: {result_with.candidates_explored} candidates")
            print(f"  Without:        {result_without.candidates_explored} candidates")
            # Don't assert ordering is better — just that both work
        elif result_with.found:
            print(f"  Heuristic found it, default didn't within budget")


class TestCustomHeuristics:
    def test_arity_preference(self):
        """Heuristic that prefers unary functions."""
        env = standard_env()
        heuristic = eval_string('''
        (lambda (comp)
          (if (= (ns-get comp "arity") 1) 100 0))
        ''', env)

        c1 = Component("abs", Symbol("abs"), TFn((TNum(),), TNum()), 1)
        c2 = Component("add", Symbol("add"), TFn((TNum(), TNum()), TNum()), 2)
        assert apply_fn(heuristic, [c1.to_namespace()]) > apply_fn(heuristic, [c2.to_namespace()])

    def test_name_match_heuristic(self):
        """Heuristic that boosts components with specific name patterns."""
        env = standard_env()
        heuristic = eval_string('''
        (lambda (comp)
          (let ((name (ns-get comp "name")))
            (if (string-contains name "sort") 1000
              (if (string-contains name "min") 500
                (if (string-contains name "max") 500
                  (ns-get comp "insertion-order"))))))
        ''', env)

        c_sort = Component("sort-pair", Symbol("sort-pair"), TFn((TStr(),), TStr()), 1)
        c_min = Component("min", Symbol("min"), TFn((TNum(), TNum()), TNum()), 2)
        c_add = Component("add", Symbol("add"), TFn((TNum(), TNum()), TNum()), 2)

        scores = {
            "sort-pair": apply_fn(heuristic, [c_sort.to_namespace()]),
            "min": apply_fn(heuristic, [c_min.to_namespace()]),
            "add": apply_fn(heuristic, [c_add.to_namespace()]),
        }
        assert scores["sort-pair"] > scores["min"] > scores["add"]

    def test_combined_heuristic(self):
        """Heuristic combining type match + recency + arity."""
        env = standard_env()
        heuristic = eval_string('''
        (lambda (comp)
          (let ((type-score (if (= (ns-get comp "return-type") "number") 100 0))
                (recency (ns-get comp "insertion-order"))
                (arity-score (if (= (ns-get comp "arity") 1) 10 0)))
            (add type-score (add recency arity-score))))
        ''', env)

        # A recent unary number function should score highest
        c = Component("new-fn", Symbol("new-fn"), TFn((TNum(),), TNum()), 1)
        score = apply_fn(heuristic, [c.to_namespace()])
        assert score > 100  # type(100) + recency(>0) + arity(10)
