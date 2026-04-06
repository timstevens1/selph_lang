"""Curriculum bootstrapping experiment.

The key validation question from §14:
  Does promoting Stage N solutions as primitives speed up Stage N+1 synthesis?

Protocol:
  1. Solve Stage 0 tasks (single-primitive ops)
  2. Solve Stage 1 tasks (two-step compositions) — without library
  3. Promote Stage 1 solutions as primitives
  4. Solve Stage 2 tasks (three-step compositions) — with and without library
  5. Compare: does the Stage 1 library reduce Stage 2 search?
"""

import time
import sys
sys.path.insert(0, ".")

from selph.ast import Spec, GoalExamples, Symbol, Number, String
from selph.synthesize import synthesize, Component
from selph.library import extract_library, promote_solved, register_abstractions
from selph.eval import standard_env
from selph.types import TStr, TNum, TBool


# ── Task definitions ─────────────────────────────────────────────────

def stage_0_tasks() -> list[tuple[str, Spec]]:
    """Stage 0: single-primitive string and character operations."""
    return [
        ("upper", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("hello"), String("HELLO")),
            (String("world"), String("WORLD")),
            (String("abc"), String("ABC")),
        )))),
        ("lower", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("HELLO"), String("hello")),
            (String("ABC"), String("abc")),
            (String("XYZ"), String("xyz")),
        )))),
        ("reverse", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("abc"), String("cba")),
            (String("hello"), String("olleh")),
            (String("xy"), String("yx")),
        )))),
        ("trim", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("  hi  "), String("hi")),
            (String(" x "), String("x")),
            (String("  ab  "), String("ab")),
        )))),
        ("length", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (String("hi"), Number(2.0)),
            (String("hello"), Number(5.0)),
            (String(""), Number(0.0)),
        )))),
        ("negate", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (Number(5.0), Number(-5.0)),
            (Number(-3.0), Number(3.0)),
            (Number(0.0), Number(0.0)),
        )))),
        ("abs", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (Number(-5.0), Number(5.0)),
            (Number(3.0), Number(3.0)),
            (Number(-1.0), Number(1.0)),
        )))),
        ("add1", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (Number(0.0), Number(1.0)),
            (Number(5.0), Number(6.0)),
            (Number(-1.0), Number(0.0)),
        )))),
        ("double", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (Number(1.0), Number(2.0)),
            (Number(3.0), Number(6.0)),
            (Number(5.0), Number(10.0)),
        )))),
    ]


def stage_1_tasks() -> list[tuple[str, Spec]]:
    """Stage 1: two-step compositions of Stage 0 ops."""
    return [
        ("upper_reverse", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("abc"), String("CBA")),
            (String("hello"), String("OLLEH")),
            (String("xy"), String("YX")),
        )))),
        ("lower_reverse", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("ABC"), String("cba")),
            (String("HELLO"), String("olleh")),
            (String("XY"), String("yx")),
        )))),
        ("trim_upper", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("  hi  "), String("HI")),
            (String(" abc "), String("ABC")),
            (String("  x  "), String("X")),
        )))),
        ("trim_lower", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("  HI  "), String("hi")),
            (String(" ABC "), String("abc")),
            (String("  X  "), String("x")),
        )))),
        ("trim_reverse", Spec(type_expr=Symbol("string"), goal=GoalExamples(pairs=(
            (String("  abc  "), String("cba")),
            (String(" hello "), String("olleh")),
            (String("  xy  "), String("yx")),
        )))),
        ("length_add1", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (String("hi"), Number(3.0)),
            (String("hello"), Number(6.0)),
            (String(""), Number(1.0)),
        )))),
        ("double_add1", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (Number(3.0), Number(7.0)),
            (Number(0.0), Number(1.0)),
            (Number(5.0), Number(11.0)),
        )))),
        ("add1_double", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (Number(3.0), Number(8.0)),
            (Number(0.0), Number(2.0)),
            (Number(5.0), Number(12.0)),
        )))),
        ("negate_add1", Spec(type_expr=Symbol("number"), goal=GoalExamples(pairs=(
            (Number(5.0), Number(-4.0)),
            (Number(0.0), Number(1.0)),
            (Number(-3.0), Number(4.0)),
        )))),
    ]


def stage_2_tasks() -> list[tuple[str, Spec]]:
    """Stage 2: three-step transforms. Need depth 3 without library,
    but depth 1-2 with Stage 1 promoted primitives."""
    return [
        ("trim_upper_reverse", Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("  abc  "), String("CBA")),
                (String(" hello "), String("OLLEH")),
                (String("  xy  "), String("YX")),
            ))
        )),
        ("trim_lower_reverse", Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("  ABC  "), String("cba")),
                (String(" HELLO "), String("olleh")),
                (String("  XY  "), String("yx")),
            ))
        )),
        ("trim_reverse_upper", Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("  abc  "), String("CBA")),
                (String(" hello "), String("OLLEH")),
                (String("  xy  "), String("YX")),
            ))
        )),
        ("trim_reverse_lower", Spec(
            type_expr=Symbol("string"),
            goal=GoalExamples(pairs=(
                (String("  ABC  "), String("cba")),
                (String(" HELLO "), String("olleh")),
                (String("  XY  "), String("yx")),
            ))
        )),
    ]


# ── Helpers ──────────────────────────────────────────────────────────

def solve_stage(name: str, tasks: list[tuple[str, Spec]], max_depth: int,
                max_candidates: int = 50000,
                extra_components: list[Component] | None = None,
                env=None):
    """Solve a set of tasks. Returns (results_dict, total_candidates, solved_count)."""
    results = {}
    total = 0
    solved = 0

    for task_name, spec in tasks:
        t0 = time.perf_counter()
        result = synthesize(spec, max_depth=max_depth,
                           max_candidates=max_candidates,
                           extra_components=extra_components,
                           env=env)
        elapsed = time.perf_counter() - t0

        status = "SOLVED" if result.found else "FAILED"
        print(f"  {task_name:25s}  {status}  candidates={result.candidates_explored:6d}  "
              f"pruned={result.candidates_type_pruned:6d}  time={elapsed:.3f}s"
              + (f"  -> {result.source}" if result.found else ""))

        results[task_name] = result
        total += result.candidates_explored
        if result.found:
            solved += 1

    print(f"\n  {name}: {solved}/{len(tasks)} solved, {total:,} total candidates\n")
    return results, total, solved


def promote_results(results: dict, specs: dict) -> list:
    """Promote solved programs as new primitives."""
    items = []
    for name, result in results.items():
        if not result.found:
            continue
        body = result.program.elements[2]
        spec = specs[name]
        first_in, first_out = spec.goal.pairs[0]
        in_type = TStr() if isinstance(first_in, String) else TNum()
        out_type = TStr() if isinstance(first_out, String) else TNum()
        items.append((name, body, in_type, out_type))
    return promote_solved(items)


# ── Main experiment ──────────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("CURRICULUM BOOTSTRAPPING EXPERIMENT")
    print("Does promoting Stage N solutions speed up Stage N+1 synthesis?")
    print("=" * 70)

    s0_list = stage_0_tasks()
    s1_list = stage_1_tasks()
    s2_list = stage_2_tasks()
    s1_specs = dict(s1_list)

    # ── Stage 0 ──────────────────────────────────────────────────────
    print("\n--- Stage 0: Primitive operations (max_depth=2) ---\n")
    s0_results, _, s0_solved = solve_stage("Stage 0", s0_list, max_depth=2)

    # ── Stage 1 ──────────────────────────────────────────────────────
    print("--- Stage 1: Two-step compositions (max_depth=3) ---\n")
    s1_results, s1_total, s1_solved = solve_stage("Stage 1", s1_list, max_depth=3)

    # ── Promote Stage 1 solutions ────────────────────────────────────
    print("--- Promoting Stage 1 solutions as primitives ---\n")
    s1_promoted = promote_results(s1_results, s1_specs)
    for a in s1_promoted:
        print(f"  {a.name:25s}  {a.param_types} -> {a.return_type!r}  body={a.body!r}")
    s1_components = [a.to_component() for a in s1_promoted]
    print(f"\n  {len(s1_promoted)} primitives promoted\n")

    # ── Stage 2 WITHOUT library (depth 2 — should mostly FAIL) ─────
    print("--- Stage 2 WITHOUT library (max_depth=2) ---\n")
    print("  (Stage 2 tasks need 3 ops; depth 2 can only compose 2)\n")
    s2_without, s2_wo_total, s2_wo_solved = solve_stage(
        "Stage 2 (no lib, d=2)", s2_list, max_depth=2)

    # ── Stage 2 WITH Stage 1 library (depth 2 — should SUCCEED) ──
    print("--- Stage 2 WITH Stage 1 library (max_depth=2) ---\n")
    print("  (Promoted primitives compress 2-op chains to 1 op,")
    print("   so depth 2 can now reach 3-op compositions)\n")

    # Register promoted primitives in an eval environment
    lib_env = register_abstractions(s1_promoted)

    s2_with, s2_w_total, s2_w_solved = solve_stage(
        "Stage 2 (with lib, d=2)", s2_list, max_depth=2,
        extra_components=s1_components, env=lib_env)

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Stage 0: {s0_solved}/{len(s0_list)} solved")
    print(f"  Stage 1: {s1_solved}/{len(s1_list)} solved")
    print(f"  Promoted: {len(s1_promoted)} Stage 1 primitives")

    print(f"\n  Stage 2 WITHOUT library: {s2_wo_solved}/{len(s2_list)} solved, "
          f"{s2_wo_total:,} candidates")
    print(f"  Stage 2 WITH    library: {s2_w_solved}/{len(s2_list)} solved, "
          f"{s2_w_total:,} candidates")

    if s2_wo_total > 0 and s2_w_total > 0:
        ratio = s2_w_total / s2_wo_total
        print(f"\n  Search ratio (with/without): {ratio:.2f}x")
        if ratio < 1.0:
            print(f"  -> Library REDUCES search by {(1-ratio)*100:.0f}%")
        elif ratio > 1.0:
            print(f"  -> Library INCREASES search by {(ratio-1)*100:.0f}%")
    elif s2_w_solved > s2_wo_solved:
        print(f"\n  -> Library ENABLES solving {s2_w_solved - s2_wo_solved} more tasks!")

    print(f"\n  Per-task comparison (Stage 2):")
    print(f"  {'task':25s}  {'without':>10s}  {'with':>10s}  {'delta':>10s}  {'status'}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for name, _ in s2_list:
        rw = s2_without.get(name)
        rl = s2_with.get(name)
        if rw and rl:
            cw = rw.candidates_explored
            cl = rl.candidates_explored
            delta = cl - cw
            sw = "ok" if rw.found else "--"
            sl = "ok" if rl.found else "--"
            sign = "+" if delta > 0 else ""
            print(f"  {name:25s}  {cw:10d}  {cl:10d}  {sign}{delta:9d}  {sw}->{sl}")

    print()


if __name__ == "__main__":
    run_experiment()
