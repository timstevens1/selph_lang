"""Meta-synthesis: learn a heuristic for sorting tasks.

Uses a proxy fitness function instead of running full synthesis:
  - Pre-compute the "good" components (fst, snd, min, max)
  - Pre-compute "bad" components (string-upper, concat, etc.)
  - Score a heuristic by whether it ranks good components higher

This avoids the O(meta_candidates × inner_budget) cost.
"""

import sys
import time
import random
sys.path.insert(0, ".")

from selph.ast import Spec, GoalExamples, GoalMinimize, GoalMaximize, Symbol, Number, String
from selph.synthesize import synthesize, SynthesisResult, Component, _default_components
from selph.eval import (
    eval_string, eval_program, eval_node, apply_fn,
    standard_env, Namespace, Closure, BuiltinFn,
)
from selph.types import TNum, TStr, TFn


def run():
    print("=" * 70)
    print("META-HEURISTIC FOR SORTING (proxy fitness)")
    print("=" * 70)

    env = standard_env()

    # Register pair helpers
    eval_program('''
    (define fst (lambda (s) (head (map to-number (string-split s " ")))))
    (define snd (lambda (s) (head (tail (map to-number (string-split s " "))))))
    ''', env)

    extra = [
        Component("fst", Symbol("fst"), TFn((TStr(),), TNum()), 1),
        Component("snd", Symbol("snd"), TFn((TStr(),), TNum()), 1),
    ]

    # ── Pre-compute good and bad components ──────────────────────────
    print("\n--- Component analysis ---\n")

    defaults = _default_components()
    all_comps = defaults + extra

    # "Good" components for pair-min: the ones that appear in the solution
    # (min (snd x) (fst x)) uses: min, fst, snd
    good_names = {"min", "max", "fst", "snd"}
    bad_names = {"string-upper", "string-lower", "string-reverse", "concat",
                 "string-trim", "string-length", "string-contains",
                 "string-split", "string-join", "to-string"}

    good_comps = [c for c in all_comps if c.name in good_names]
    bad_comps = [c for c in all_comps if c.name in bad_names]
    neutral_comps = [c for c in all_comps
                     if c.name not in good_names and c.name not in bad_names and c.arity > 0]

    print(f"  Good components ({len(good_comps)}): {[c.name for c in good_comps]}")
    print(f"  Bad components ({len(bad_comps)}): {[c.name for c in bad_comps]}")
    print(f"  Neutral components ({len(neutral_comps)}): {[c.name for c in neutral_comps[:10]]}...")

    # Convert to namespaces for SELPH access
    good_ns_list = [c.to_namespace() for c in good_comps]
    bad_ns_list = [c.to_namespace() for c in bad_comps]

    env.define("good-components", good_ns_list)
    env.define("bad-components", bad_ns_list)

    # ── Proxy fitness: does the heuristic rank good above bad? ───────
    print("\n--- Proxy fitness function ---\n")

    eval_program("""
    (define proxy-fitness (lambda (h)
      (let ((good-scores (map h good-components))
            (bad-scores (map h bad-components))
            (avg-good (/ (reduce + good-scores 0) (length good-scores)))
            (avg-bad (/ (reduce + bad-scores 0) (length bad-scores))))
        (subtract avg-bad avg-good))))
    """, env)

    print("  proxy-fitness: higher score = good components ranked higher than bad")
    print("  (measures avg-bad - avg-good; we want to MINIMIZE this)")
    print("  negative = good is ranked higher (what we want)")

    # ── Score hand-written heuristics ────────────────────────────────
    print("\n--- Hand-written heuristics (proxy scores) ---\n")

    heuristics = {
        "constant-0": '(lambda (comp) 0)',
        "recency": '(lambda (comp) (ns-get comp "insertion-order"))',
        "num-output": """(lambda (comp)
          (if (= (ns-get comp "return-type") "number")
            (add 100 (ns-get comp "insertion-order"))
            (ns-get comp "insertion-order")))""",
        "arity-1": """(lambda (comp)
          (if (= (ns-get comp "arity") 1)
            (add 100 (ns-get comp "insertion-order"))
            (ns-get comp "insertion-order")))""",
        "boost-fst-snd": """(lambda (comp)
          (let ((name (ns-get comp "name")))
            (if (string-contains name "fst") 10000
              (if (string-contains name "snd") 10000
                (if (string-contains name "min") 9000
                  (if (string-contains name "max") 9000
                    (ns-get comp "insertion-order")))))))""",
    }

    proxy_scores = {}
    for hname, hsrc in heuristics.items():
        h = eval_string(hsrc, env)
        fitness_fn = env.lookup("proxy-fitness")
        score = apply_fn(fitness_fn, [h])
        proxy_scores[hname] = score
        print(f"  {hname:20s}  proxy={score:8.1f}")

    # ── Now verify: does proxy score correlate with real synthesis cost? ──
    print("\n--- Verify proxy vs real performance ---\n")

    rng = random.Random(42)
    pairs = []
    for _ in range(10):
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        pairs.append((String(f"{a} {b}"), Number(float(min(a, b)))))
    pair_min_spec = Spec(type_expr=Symbol("number"),
                         goal=GoalExamples(pairs=tuple(pairs)))

    for hname in ["constant-0", "recency", "num-output", "boost-fst-snd"]:
        h = eval_string(heuristics[hname], env)
        e = standard_env()
        eval_program('''
        (define fst (lambda (s) (head (map to-number (string-split s " ")))))
        (define snd (lambda (s) (head (tail (map to-number (string-split s " "))))))
        ''', e)
        result = synthesize(pair_min_spec, max_depth=2, max_candidates=200000,
                           extra_components=extra, env=e, heuristic=h)
        cand = result.candidates_explored
        status = "OK" if result.found else "--"
        print(f"  {status}  {hname:20s}  proxy={proxy_scores[hname]:8.1f}  "
              f"real={cand:7d} candidates")

    # ── Meta-synthesis with proxy fitness ─────────────────────────────
    print("\n--- Meta-synthesis (proxy fitness) ---\n")

    spec = Spec(goal=GoalMinimize(Symbol("proxy-fitness")))

    t0 = time.perf_counter()
    result = synthesize(spec, max_depth=2, max_candidates=5000, env=env)
    elapsed = time.perf_counter() - t0

    print(f"  Explored {result.candidates_explored} candidates in {elapsed:.1f}s")

    if result.found:
        print(f"  Best heuristic: {result.source}")
        print(f"  Proxy score: {result.fitness_score:.1f}")

        # Verify with real synthesis
        print("\n--- Verify synthesized heuristic on real task ---\n")
        h = eval_node(result.program, env)
        e = standard_env()
        eval_program('''
        (define fst (lambda (s) (head (map to-number (string-split s " ")))))
        (define snd (lambda (s) (head (tail (map to-number (string-split s " "))))))
        ''', e)
        real_result = synthesize(pair_min_spec, max_depth=2, max_candidates=200000,
                                extra_components=extra, env=e, heuristic=h)
        print(f"  Real result: found={real_result.found}  candidates={real_result.candidates_explored}")
    else:
        print("  No heuristic found within budget")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for hname in sorted(proxy_scores, key=proxy_scores.get):
        print(f"  {hname:20s}  proxy={proxy_scores[hname]:8.1f}")
    if result.found:
        print(f"  {'SYNTHESIZED':20s}  proxy={result.fitness_score:8.1f}")
        print(f"\n  Program: {result.source}")
    print()


if __name__ == "__main__":
    run()
