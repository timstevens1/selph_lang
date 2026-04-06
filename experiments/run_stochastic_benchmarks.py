"""Run the stochastic process benchmark suite.

Tests the SELPH synthesizer against processes with known information-theoretic
limits. Measures whether the system can discover the generating rule for
deterministic processes and the optimal predictor for stochastic ones.

Runs in three modes:
  1. Flat: solve all tasks with no library
  2. Curriculum: solve stage by stage, promoting solutions
  3. Compare: report accuracy relative to Bayes-optimal
"""

import sys
import time
sys.path.insert(0, ".")

from selph.stochastic import build_full_suite, BenchmarkTask, BenchmarkSuite
from selph.synthesize import synthesize, Component, make_validation_fn
from selph.library import promote_solved, register_abstractions
from selph.induce import induce_from_failure
from selph.eval import eval_node, apply_fn, standard_env
from selph.types import TStr, TNum
from selph.ast import String, Number


def evaluate_program(task: BenchmarkTask, program, env) -> float:
    """Evaluate a synthesized program against a benchmark task.

    Returns accuracy: fraction of spec examples correctly predicted.
    """
    if program is None:
        return 0.0

    try:
        fn = eval_node(program, env)
    except Exception:
        return 0.0

    goal = task.spec.goal
    if not hasattr(goal, 'pairs'):
        return 0.0

    correct = 0
    total = len(goal.pairs)
    for in_node, out_node in goal.pairs:
        try:
            in_val = in_node.value if hasattr(in_node, 'value') else None
            expected = out_node.value if hasattr(out_node, 'value') else None
            if in_val is None:
                continue
            actual = apply_fn(fn, [in_val])
            if actual == expected:
                correct += 1
        except Exception:
            pass

    return correct / total if total > 0 else 0.0


def run_flat(suite: BenchmarkSuite, max_depth: int = 2,
             max_candidates: int = 30000):
    """Solve all tasks with no library (baseline)."""
    print("\n" + "=" * 70)
    print("FLAT MODE: No library, all tasks independent")
    print("=" * 70)

    env = standard_env()
    results = {}

    for stage in range(3):
        tasks = suite.tasks_for_stage(stage)
        if not tasks:
            continue
        # Enable if-expressions for Stage 2+ (branching tasks)
        use_if = stage >= 2
        print(f"\n--- Stage {stage} ({len(tasks)} tasks, if={use_if}) ---\n")

        for task in tasks:
            t0 = time.perf_counter()
            sr = synthesize(task.spec, max_depth=max_depth,
                           max_candidates=max_candidates, env=env,
                           enable_if=use_if)
            elapsed = time.perf_counter() - t0

            accuracy = 0.0
            if sr.found:
                accuracy = evaluate_program(task, sr.program, env)

            gap = task.optimal_accuracy - accuracy
            status = "OK" if sr.found else "--"
            opt_str = f"opt={task.optimal_accuracy:.0%}"
            ent_str = f"H={task.entropy_rate:.2f}" if task.entropy_rate > 0 else "H=0"

            print(f"  {status}  {task.name:30s}  acc={accuracy:.0%}  {opt_str}  "
                  f"gap={gap:+.0%}  {ent_str}  cand={sr.candidates_explored:6d}  "
                  f"t={elapsed:.2f}s"
                  + (f"  {sr.source}" if sr.found else ""))

            results[task.name] = {
                "found": sr.found,
                "accuracy": accuracy,
                "optimal": task.optimal_accuracy,
                "gap": gap,
                "candidates": sr.candidates_explored,
                "entropy_rate": task.entropy_rate,
                "stage": stage,
            }

    return results


def run_curriculum(suite: BenchmarkSuite, max_depth: int = 2,
                   max_candidates: int = 30000):
    """Solve stage by stage, promoting solutions as primitives."""
    print("\n" + "=" * 70)
    print("CURRICULUM MODE: Promote solutions between stages")
    print("=" * 70)

    env = standard_env()
    all_components: list[Component] = []
    results = {}

    for stage in range(3):
        tasks = suite.tasks_for_stage(stage)
        if not tasks:
            continue
        use_if = stage >= 2
        print(f"\n--- Stage {stage} ({len(tasks)} tasks, "
              f"{len(all_components)} lib prims, if={use_if}) ---\n")

        stage_solved = []
        for task in tasks:
            t0 = time.perf_counter()
            sr = synthesize(task.spec, max_depth=max_depth,
                           max_candidates=max_candidates,
                           extra_components=all_components if all_components else None,
                           env=env, enable_if=use_if)

            # Try induction on failure
            if not sr.found:
                ir = induce_from_failure(
                    task.spec, max_depth=max_depth,
                    max_candidates=max_candidates // 2,
                    extra_components=all_components if all_components else None,
                    env=env)
                if ir.success:
                    sr.found = True
                    sr.program = ir.program
                    sr.source = ir.source
                    sr.candidates_explored += ir.candidates_explored
                    if ir.new_primitives:
                        register_abstractions(ir.new_primitives, env)
                        all_components.extend([p.to_component() for p in ir.new_primitives])

            elapsed = time.perf_counter() - t0

            accuracy = 0.0
            if sr.found:
                accuracy = evaluate_program(task, sr.program, env)
                stage_solved.append((task, sr))

            gap = task.optimal_accuracy - accuracy
            status = "OK" if sr.found else "--"

            print(f"  {status}  {task.name:30s}  acc={accuracy:.0%}  "
                  f"opt={task.optimal_accuracy:.0%}  gap={gap:+.0%}  "
                  f"cand={sr.candidates_explored:6d}  t={elapsed:.2f}s"
                  + (f"  {sr.source}" if sr.found else ""))

            results[task.name] = {
                "found": sr.found,
                "accuracy": accuracy,
                "optimal": task.optimal_accuracy,
                "gap": gap,
                "candidates": sr.candidates_explored,
                "entropy_rate": task.entropy_rate,
                "stage": stage,
            }

        # Promote solved programs
        if stage_solved:
            items = []
            for task, sr in stage_solved:
                if sr.program is None:
                    continue
                body = sr.program.elements[2]
                items.append((task.name, body, TNum(), TNum()))

            promoted = promote_solved(items, name_prefix=f"bench_s{stage}")
            register_abstractions(promoted, env)
            all_components.extend([a.to_component() for a in promoted])
            print(f"\n  Promoted {len(promoted)} primitives")

    return results


def print_summary(flat_results: dict, curriculum_results: dict,
                  suite: BenchmarkSuite):
    """Print comparative summary."""
    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY")
    print("=" * 70)

    print(f"\n  {'task':30s}  {'flat':>6s}  {'curric':>6s}  {'optimal':>7s}  "
          f"{'flat_gap':>8s}  {'curr_gap':>8s}  {'stage':>5s}  {'H':>5s}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*5}")

    stage_stats = {}
    for task in suite.tasks:
        fr = flat_results.get(task.name, {})
        cr = curriculum_results.get(task.name, {})

        f_acc = fr.get("accuracy", 0.0)
        c_acc = cr.get("accuracy", 0.0)
        opt = task.optimal_accuracy
        f_gap = opt - f_acc
        c_gap = opt - c_acc

        print(f"  {task.name:30s}  {f_acc:5.0%}  {c_acc:5.0%}   {opt:5.0%}   "
              f"{f_gap:+7.0%}   {c_gap:+7.0%}   {task.stage:5d}  "
              f"{task.entropy_rate:5.2f}")

        stage = task.stage
        if stage not in stage_stats:
            stage_stats[stage] = {"flat_correct": 0, "curr_correct": 0,
                                  "flat_total": 0, "curr_total": 0, "total": 0}
        stage_stats[stage]["total"] += 1
        stage_stats[stage]["flat_total"] += 1
        stage_stats[stage]["curr_total"] += 1
        if f_acc == opt:
            stage_stats[stage]["flat_correct"] += 1
        if c_acc == opt:
            stage_stats[stage]["curr_correct"] += 1

    print(f"\n  Per-stage optimal-match rate:")
    for stage in sorted(stage_stats.keys()):
        s = stage_stats[stage]
        f_rate = s["flat_correct"] / s["total"] if s["total"] > 0 else 0
        c_rate = s["curr_correct"] / s["total"] if s["total"] > 0 else 0
        print(f"    Stage {stage}: flat={f_rate:.0%} ({s['flat_correct']}/{s['total']})  "
              f"curriculum={c_rate:.0%} ({s['curr_correct']}/{s['total']})")


def main():
    print("SELPH Stochastic Process Benchmarks")
    print("=" * 70)

    suite = build_full_suite(seed=42)
    print(f"\nSuite: {suite.name}")
    print(f"Tasks: {suite.total_tasks}")
    for stage in range(3):
        tasks = suite.tasks_for_stage(stage)
        if tasks:
            print(f"  Stage {stage}: {len(tasks)} tasks")

    flat_results = run_flat(suite)
    curriculum_results = run_curriculum(suite)
    print_summary(flat_results, curriculum_results, suite)


if __name__ == "__main__":
    main()
