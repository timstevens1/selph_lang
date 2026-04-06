"""Curriculum for discovering list sorting.

Key insight from first attempt: the system memorizes examples via
lookup tables unless we provide enough examples + held-out validation.

This version:
  - Uses 10+ examples per task (hard to memorize)
  - Adds held-out validation (reject solutions that don't generalize)
  - Disables divide-and-conquer (it produces lookup tables)
  - Focuses on composing fst/snd/min/max which ARE general
"""

import sys
import time
import random
sys.path.insert(0, ".")

from selph.ast import Spec, GoalExamples, Symbol, Number, String
from selph.synthesize import synthesize, SynthesisResult, Component, make_validation_fn
from selph.library import promote_solved, register_abstractions
from selph.induce import induce_from_failure
from selph.eval import (
    eval_string, eval_program, eval_node, apply_fn,
    standard_env, Env, Closure, BuiltinFn,
)
from selph.types import TNum, TStr, TFn


def make_pair_examples(fn, n=15, seed=42):
    """Generate examples for a pair function: "a b" -> result."""
    rng = random.Random(seed)
    pairs = []
    seen = set()
    while len(pairs) < n:
        a = rng.randint(0, 20)
        b = rng.randint(0, 20)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        result = fn(a, b)
        if isinstance(result, (int, float)):
            pairs.append((String(f"{a} {b}"), Number(float(result))))
        else:
            pairs.append((String(f"{a} {b}"), String(str(result))))
    return pairs


def make_triple_examples(fn, n=15, seed=42):
    """Generate examples for a triple function: "a b c" -> result."""
    rng = random.Random(seed)
    triples = []
    seen = set()
    while len(triples) < n:
        a, b, c = rng.randint(0, 15), rng.randint(0, 15), rng.randint(0, 15)
        key = (a, b, c)
        if key in seen:
            continue
        seen.add(key)
        result = fn(a, b, c)
        if isinstance(result, (int, float)):
            triples.append((String(f"{a} {b} {c}"), Number(float(result))))
        else:
            triples.append((String(f"{a} {b} {c}"), String(str(result))))
    return triples


def make_validator(fn, n=20, seed=99):
    """Create a validation function with held-out test cases."""
    rng = random.Random(seed)
    test_inputs = []
    for _ in range(n):
        a = rng.randint(0, 30)
        b = rng.randint(0, 30)
        test_inputs.append(f"{a} {b}")

    def validate(fn_val):
        for inp_str in test_inputs:
            parts = inp_str.split()
            a, b = int(parts[0]), int(parts[1])
            expected = fn(a, b)
            try:
                actual = apply_fn(fn_val, [inp_str])
                if isinstance(expected, (int, float)):
                    if actual != float(expected):
                        return False
                else:
                    if actual != str(expected):
                        return False
            except Exception:
                return False
        return True
    return validate


def make_triple_validator(fn, n=20, seed=99):
    rng = random.Random(seed)
    test_inputs = []
    for _ in range(n):
        a, b, c = rng.randint(0, 25), rng.randint(0, 25), rng.randint(0, 25)
        test_inputs.append(f"{a} {b} {c}")

    def validate(fn_val):
        for inp_str in test_inputs:
            parts = inp_str.split()
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
            expected = fn(a, b, c)
            try:
                actual = apply_fn(fn_val, [inp_str])
                if isinstance(expected, (int, float)):
                    if actual != float(expected):
                        return False
                else:
                    if actual != str(expected):
                        return False
            except Exception:
                return False
        return True
    return validate


def solve(name, spec, env, components, max_depth=2, max_candidates=50000,
          enable_if=False, validation_fn=None):
    """Solve with held-out validation."""
    t0 = time.perf_counter()
    result = synthesize(
        spec, max_depth=max_depth, max_candidates=max_candidates,
        extra_components=components if components else None,
        env=env, enable_if=enable_if, validation_fn=validation_fn,
    )

    if not result.found:
        ir = induce_from_failure(
            spec, max_depth=max_depth, max_candidates=max_candidates // 2,
            extra_components=components if components else None, env=env,
        )
        if ir.success:
            # Validate induction result too
            if validation_fn is not None:
                fn_val = eval_node(ir.program, env)
                if not validation_fn(fn_val):
                    ir.success = False
            if ir.success:
                result = SynthesisResult(
                    program=ir.program, source=ir.source,
                    found=True, candidates_explored=ir.candidates_explored,
                )

    elapsed = time.perf_counter() - t0
    status = "OK" if result.found else "--"
    print(f"  {status}  {name:35s}  cand={result.candidates_explored:6d}  "
          f"t={elapsed:.2f}s"
          + (f"  {result.source}" if result.found else ""))

    if result.found and result.program is not None:
        body = result.program.elements[2]
        first_in = spec.goal.pairs[0][0]
        first_out = spec.goal.pairs[0][1]
        in_type = TStr() if isinstance(first_in, String) else TNum()
        out_type = TStr() if isinstance(first_out, String) else TNum()
        promoted = promote_solved([(name, body, in_type, out_type)])
        register_abstractions(promoted, env)
        for p in promoted:
            components.append(p.to_component())

    return result


def run():
    print("=" * 70)
    print("SORTING CURRICULUM (with generalization)")
    print("=" * 70)

    env = standard_env()
    components: list[Component] = []

    # Register parsing helpers
    eval_program("""
    (define parse-pair (lambda (s) (map to-number (string-split s " "))))
    (define fst (lambda (s) (head (parse-pair s))))
    (define snd (lambda (s) (head (tail (parse-pair s)))))
    (define make-pair (lambda (a b)
      (string-join (list (to-string a) (to-string b)) " ")))
    (define parse-triple (lambda (s) (map to-number (string-split s " "))))
    (define t1 (lambda (s) (head (parse-triple s))))
    (define t2 (lambda (s) (head (tail (parse-triple s)))))
    (define t3 (lambda (s) (head (tail (tail (parse-triple s))))))
    (define make-triple (lambda (a b c)
      (string-join (list (to-string a) (to-string b) (to-string c)) " ")))
    """, env)

    N = TNum()
    S = TStr()
    components.extend([
        Component("fst", Symbol("fst"), TFn((S,), N), 1),
        Component("snd", Symbol("snd"), TFn((S,), N), 1),
        Component("make-pair", Symbol("make-pair"), TFn((N, N), S), 2),
        Component("t1", Symbol("t1"), TFn((S,), N), 1),
        Component("t2", Symbol("t2"), TFn((S,), N), 1),
        Component("t3", Symbol("t3"), TFn((S,), N), 1),
        Component("make-triple", Symbol("make-triple"), TFn((N, N, N), S), 3),
    ])

    # ── Stage 0: Pair operations ─────────────────────────────────────
    print("\n--- Stage 0: Pair operations (with 15 examples + held-out validation) ---\n")

    pair_tasks = [
        ("pair-min",
         lambda a, b: min(a, b),
         "number"),
        ("pair-max",
         lambda a, b: max(a, b),
         "number"),
        ("sort-pair",
         lambda a, b: f"{min(a,b)} {max(a,b)}",
         "string"),
    ]

    for name, fn, out_type in pair_tasks:
        examples = make_pair_examples(fn, n=15)
        spec = Spec(
            type_expr=Symbol(out_type),
            goal=GoalExamples(pairs=tuple(examples)),
        )
        validator = make_validator(fn)
        solve(name, spec, env, components, enable_if=True,
              validation_fn=validator)

    # ── Stage 1: Three-element operations ────────────────────────────
    print("\n--- Stage 1: Three-element operations ---\n")

    triple_tasks = [
        ("min-of-three",
         lambda a, b, c: min(a, b, c),
         "number"),
        ("max-of-three",
         lambda a, b, c: max(a, b, c),
         "number"),
        ("mid-of-three",
         lambda a, b, c: sorted([a, b, c])[1],
         "number"),
    ]

    for name, fn, out_type in triple_tasks:
        examples = make_triple_examples(fn, n=15)
        spec = Spec(
            type_expr=Symbol(out_type),
            goal=GoalExamples(pairs=tuple(examples)),
        )
        validator = make_triple_validator(fn)
        solve(name, spec, env, components, enable_if=True,
              validation_fn=validator)

    # ── Stage 2: Sort three ──────────────────────────────────────────
    print("\n--- Stage 2: Sort three elements ---\n")

    sort3_fn = lambda a, b, c: f"{sorted([a,b,c])[0]} {sorted([a,b,c])[1]} {sorted([a,b,c])[2]}"
    examples = make_triple_examples(sort3_fn, n=15)
    spec = Spec(
        type_expr=Symbol("string"),
        goal=GoalExamples(pairs=tuple(examples)),
    )
    validator = make_triple_validator(sort3_fn)
    solve("sort-three", spec, env, components, enable_if=True,
          validation_fn=validator)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Library: {len(components)} components")

    # Generalization test
    print("\n--- Generalization test ---\n")
    for fname in ["s0_pair-min", "s0_pair-max", "s0_sort-pair"]:
        try:
            fn = env.lookup(fname)
            tests = [("17 3", 3 if "min" in fname else 17 if "max" in fname else "3 17"),
                     ("0 99", 0 if "min" in fname else 99 if "max" in fname else "0 99"),
                     ("5 5", 5 if "min" in fname or "max" in fname else "5 5")]
            for inp, expected in tests:
                actual = apply_fn(fn, [inp])
                expected_val = float(expected) if isinstance(expected, int) else expected
                ok = "OK" if actual == expected_val else "FAIL"
                print(f"  {ok}  {fname}({inp!r}) = {actual!r}")
        except Exception as e:
            print(f"  {fname}: {e}")

    for fname in ["s0_min-of-three", "s0_max-of-three", "s0_sort-three"]:
        try:
            fn = env.lookup(fname)
            tests = [("15 3 9", "3 9 15" if "sort" in fname else (3 if "min" in fname else 15)),
                     ("1 1 1", "1 1 1" if "sort" in fname else 1)]
            for inp, expected in tests:
                actual = apply_fn(fn, [inp])
                expected_val = float(expected) if isinstance(expected, int) else expected
                ok = "OK" if actual == expected_val else "FAIL"
                print(f"  {ok}  {fname}({inp!r}) = {actual!r}")
        except Exception as e:
            print(f"  {fname}: {e}")


if __name__ == "__main__":
    run()
