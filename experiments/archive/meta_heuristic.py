"""Meta-synthesis experiment: can the system learn its own search heuristic?

The idea:
  1. Define a task suite (several simple synthesis problems)
  2. Define a fitness function that scores a heuristic by total candidates
     used to solve the task suite
  3. Synthesize a heuristic that minimizes that fitness

If this works, the system has improved its own search strategy using
its own synthesis machinery.
"""

import sys
import time
sys.path.insert(0, ".")

from selph.ast import Spec, GoalExamples, GoalMinimize, Symbol, Number, String
from selph.synthesize import synthesize, SynthesisResult, Component
from selph.eval import (
    eval_string, eval_program, eval_node, apply_fn,
    standard_env, Namespace, Closure, BuiltinFn,
)
from selph.types import TNum, TStr, TFn


def run():
    print("=" * 70)
    print("META-HEURISTIC SYNTHESIS")
    print("Can the system learn its own search heuristic?")
    print("=" * 70)

    env = standard_env()

    # ── Step 1: Define a task suite ──────────────────────────────────
    print("\n--- Step 1: Define task suite ---\n")

    # Simple tasks that the synthesizer can solve
    eval_program("""
    (define task1 (:spec :type number :goal (:examples ((0 -> 1) (5 -> 6) (-1 -> 0) (10 -> 11) (3 -> 4)))))
    (define task2 (:spec :type number :goal (:examples ((1 -> 2) (3 -> 6) (5 -> 10) (0 -> 0) (7 -> 14)))))
    (define task3 (:spec :type number :goal (:examples ((2 -> 0) (5 -> 3) (10 -> 8) (1 -> -1) (0 -> -2)))))
    (define task4 (:spec :type number :goal (:examples ((-5 -> 5) (-3 -> 3) (-1 -> 1) (0 -> 0) (2 -> 2) (7 -> 7)))))
    (define task5 (:spec :type number :goal (:examples ((1 -> 1) (4 -> 2) (9 -> 3) (0 -> 0) (2 -> 1)))))
    (define task-suite (list task1 task2 task3 task4 task5))
    """, env)

    print("  5 tasks defined:")
    print("  task1: x -> x+1")
    print("  task2: x -> 2*x")
    print("  task3: x -> x-2")
    print("  task4: x -> |x|")
    print("  task5: x -> floor(sqrt(x)) (approx)")

    # ── Step 2: Baseline — solve without heuristic ───────────────────
    print("\n--- Step 2: Baseline (no heuristic) ---\n")

    baseline_total = 0
    for name in ["task1", "task2", "task3", "task4", "task5"]:
        spec = env.lookup(name)
        result = synthesize(spec, max_depth=2, max_candidates=10000, env=env)
        status = "OK" if result.found else "--"
        print(f"  {status}  {name}  candidates={result.candidates_explored:5d}"
              + (f"  {result.source}" if result.found else ""))
        baseline_total += result.candidates_explored

    print(f"\n  Baseline total: {baseline_total} candidates")

    # ── Step 3: Define the fitness function ──────────────────────────
    print("\n--- Step 3: Define fitness function ---\n")

    eval_program("""
    (define score-heuristic (lambda (h)
      (reduce (lambda (total task)
        (add total
          (ns-get (synthesize (namespace
            (spec task)
            (max-depth 2)
            (max-candidates 5000)
            (heuristic h)))
          "candidates")))
        task-suite
        0)))
    """, env)

    print("  score-heuristic defined: runs synthesis with heuristic on all 4 tasks")

    # ── Step 4: Test some hand-written heuristics ────────────────────
    print("\n--- Step 4: Score hand-written heuristics ---\n")

    heuristics = {
        "recency": '(lambda (comp) (ns-get comp "insertion-order"))',
        "arity-1-first": '(lambda (comp) (if (= (ns-get comp "arity") 1) 100 (ns-get comp "insertion-order")))',
        "num-output-first": """(lambda (comp)
          (if (= (ns-get comp "return-type") "number")
            (add 100 (ns-get comp "insertion-order"))
            (ns-get comp "insertion-order")))""",
        "combined": """(lambda (comp)
          (let ((ret-bonus (if (= (ns-get comp "return-type") "number") 100 0))
                (arity-bonus (if (= (ns-get comp "arity") 1) 50 0))
                (recency (ns-get comp "insertion-order")))
            (add ret-bonus (add arity-bonus recency))))""",
    }

    scores = {}
    for name, src in heuristics.items():
        h = eval_string(src, env)
        t0 = time.perf_counter()
        score = apply_fn(env.lookup("score-heuristic"), [h])
        elapsed = time.perf_counter() - t0
        scores[name] = score
        print(f"  {name:20s}  score={score:8.0f} candidates  t={elapsed:.2f}s")

    best_manual = min(scores, key=scores.get)
    print(f"\n  Best hand-written: {best_manual} ({scores[best_manual]:.0f} candidates)")

    # ── Step 5: Synthesize a heuristic ───────────────────────────────
    print("\n--- Step 5: Synthesize a better heuristic ---\n")
    print("  Searching for: (lambda (comp) body) that minimizes total candidates...")
    print("  This is meta-synthesis: the synthesizer searching for its own search strategy.\n")

    # The fitness function for heuristic synthesis
    spec = Spec(goal=GoalMinimize(Symbol("score-heuristic")))

    t0 = time.perf_counter()
    result = synthesize(spec, max_depth=1, max_candidates=500, env=env)
    elapsed = time.perf_counter() - t0

    print(f"  Search completed in {elapsed:.1f}s, {result.candidates_explored} candidates explored")

    if result.found:
        print(f"  Found heuristic: {result.source}")
        print(f"  Fitness score: {result.fitness_score:.0f} candidates")

        # Compare to baseline and best manual
        improvement_vs_baseline = (1 - result.fitness_score / baseline_total) * 100
        improvement_vs_manual = (1 - result.fitness_score / scores[best_manual]) * 100

        print(f"\n  vs baseline (no heuristic):    {improvement_vs_baseline:+.0f}%")
        print(f"  vs best manual ({best_manual}): {improvement_vs_manual:+.0f}%")
    else:
        print("  No heuristic found within budget")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline (no heuristic):  {baseline_total:6.0f} total candidates")
    for name, score in sorted(scores.items(), key=lambda kv: kv[1]):
        marker = " <-- best manual" if name == best_manual else ""
        print(f"  {name:25s}  {score:6.0f} total candidates{marker}")
    if result.found:
        print(f"  SYNTHESIZED:              {result.fitness_score:6.0f} total candidates")
        print(f"  Program: {result.source}")
    print()


if __name__ == "__main__":
    run()
