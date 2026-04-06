"""Run the meta-curriculum: generate training data from synthesis,
then use it to teach meta-level capabilities.

Phase 1: Run standard curriculum, logging everything
Phase 2: Generate Meta-1.0 tasks (predict output type)
Phase 3: Solve Meta-1.0 tasks using the synthesizer
Phase 4: Evaluate: can the system learn about its own process?
"""

import sys
import time
sys.path.insert(0, ".")

from selph.logger import SynthesisLogger, SynthesisLog, extract_components_from_source, estimate_depth
from selph.meta_curriculum import generate_type_prediction_tasks
from selph.synthesize import synthesize, Component
from selph.eval import standard_env, eval_program, eval_node, apply_fn
from selph.ast import Spec, GoalExamples, Symbol, Number, String
from selph.types import TNum, TStr, TFn
from selph.taskgen import generate_stage0_tasks, generate_stage1_tasks
import selph_fast
from selph.parser import parse
import random


def run():
    print("=" * 60)
    print("META-CURRICULUM EXPERIMENT")
    print("Phase 1: Generate training data via synthesis")
    print("Phase 2: Learn from synthesis logs")
    print("=" * 60)

    logger = SynthesisLogger()

    # ── Phase 1: Run synthesis on tasks, log everything ──────────────
    print("\n--- Phase 1: Generating training data ---\n")

    macros = [
        {'name': 'fst', 'params': ['s'],
         'body': parse('(to-number (head (string-split s " ")))') },
        {'name': 'snd', 'params': ['s'],
         'body': parse('(to-number (head (tail (string-split s " "))))') },
    ]

    components = [
        {'name': 'x', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '0', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '1', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '2', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '-1', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': 'abs', 'builtin': 'abs', 'arity': 1, 'ret_type': 0, 'param_types': [0], 'priority': 0.0},
        {'name': 'negate', 'builtin': 'negate', 'arity': 1, 'ret_type': 0, 'param_types': [0], 'priority': 0.0},
        {'name': 'add', 'builtin': 'add', 'arity': 2, 'ret_type': 0, 'param_types': [0, 0], 'priority': 0.0},
        {'name': 'subtract', 'builtin': 'subtract', 'arity': 2, 'ret_type': 0, 'param_types': [0, 0], 'priority': 0.0},
        {'name': 'multiply', 'builtin': 'multiply', 'arity': 2, 'ret_type': 0, 'param_types': [0, 0], 'priority': 0.0},
        {'name': 'min', 'builtin': 'min', 'arity': 2, 'ret_type': 0, 'param_types': [0, 0], 'priority': 0.0},
        {'name': 'max', 'builtin': 'max', 'arity': 2, 'ret_type': 0, 'param_types': [0, 0], 'priority': 0.0},
        {'name': 'modulo', 'builtin': 'modulo', 'arity': 2, 'ret_type': 0, 'param_types': [0, 0], 'priority': 0.0},
    ]

    # Generate and solve many tasks with Rust, logging each
    rng = random.Random(42)
    num_tasks = 100

    # Numeric tasks
    num_fns = [
        ("add1", lambda x: float(x + 1)),
        ("double", lambda x: float(x * 2)),
        ("negate", lambda x: float(-x)),
        ("abs", lambda x: float(abs(x))),
        ("add2", lambda x: float(x + 2)),
        ("sub1", lambda x: float(x - 1)),
        ("square", lambda x: float(x * x)),
    ]

    task_count = 0
    for fn_name, fn in num_fns:
        for trial in range(5):
            r = random.Random(42 + hash(fn_name) + trial)
            inputs = [float(r.randint(-10, 10)) for _ in range(10)]
            expected = [fn(x) for x in inputs]

            t0 = time.perf_counter()
            found, source, explored = selph_fast.fast_synthesize(
                components, inputs, expected, 2, 50000, macros)
            elapsed = time.perf_counter() - t0

            log = SynthesisLog(
                task_name=f"{fn_name}_{trial}",
                input_type="number",
                output_type="number",
                num_examples=len(inputs),
                example_inputs=[str(x) for x in inputs[:5]],
                example_outputs=[str(x) for x in expected[:5]],
                found=found,
                source=source,
                candidates_explored=explored,
                time_seconds=elapsed,
                strategy="direct",
                stage=0,
            )
            if found:
                log.components_used = extract_components_from_source(source)
                log.solution_depth = estimate_depth(source)
            logger.log(log)
            task_count += 1

    # Pair tasks (cross-type: string input -> number output)
    pair_fns = [
        ("pair-min", lambda a, b: float(min(a, b))),
        ("pair-max", lambda a, b: float(max(a, b))),
        ("pair-sum", lambda a, b: float(a + b)),
        ("pair-diff", lambda a, b: float(a - b)),
    ]

    pair_comps = components + [
        {'name': 'fst', 'builtin': 'fst', 'arity': 1, 'ret_type': 0, 'param_types': [1], 'priority': 50.0},
        {'name': 'snd', 'builtin': 'snd', 'arity': 1, 'ret_type': 0, 'param_types': [1], 'priority': 50.0},
    ]
    # For pair tasks, x is string type
    pair_comps_typed = [c if c['name'] != 'x' else {**c, 'ret_type': 1} for c in pair_comps]

    for fn_name, fn in pair_fns:
        for trial in range(5):
            r = random.Random(42 + hash(fn_name) + trial)
            inputs = [f"{r.randint(0, 20)} {r.randint(0, 20)}" for _ in range(10)]
            expected = [fn(int(s.split()[0]), int(s.split()[1])) for s in inputs]

            t0 = time.perf_counter()
            found, source, explored = selph_fast.fast_synthesize(
                pair_comps_typed, inputs, expected, 2, 50000, macros)
            elapsed = time.perf_counter() - t0

            log = SynthesisLog(
                task_name=f"{fn_name}_{trial}",
                input_type="string",
                output_type="number",
                num_examples=len(inputs),
                example_inputs=inputs[:5],
                example_outputs=[str(x) for x in expected[:5]],
                found=found,
                source=source,
                candidates_explored=explored,
                time_seconds=elapsed,
                strategy="direct",
                stage=1,
            )
            if found:
                log.components_used = extract_components_from_source(source)
                log.solution_depth = estimate_depth(source)
            logger.log(log)
            task_count += 1

    print(f"  Generated {task_count} synthesis runs")
    print(f"  {logger.summary()}")

    # Save logs
    logger.save("/tmp/selph_training_data.json")
    print(f"  Saved to /tmp/selph_training_data.json")

    # ── Phase 2: Analyze training data ───────────────────────────────
    print("\n--- Phase 2: Training data analysis ---\n")

    meta1 = logger.meta1_data()
    meta4 = logger.meta4_data()

    print(f"  Meta-1 (heuristic) training examples: {len(meta1)}")
    print(f"  Meta-4 (generation) training examples: {len(meta4)}")
    print(f"  Component frequency:")
    for name, count in sorted(logger.component_frequency().items(),
                               key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"    {name:20s}  {count}")

    # ── Phase 3: Solve Meta-1.0 (type prediction) ───────────────────
    print("\n--- Phase 3: Meta-1.0 — Learn to predict output type ---\n")

    meta_tasks = generate_type_prediction_tasks(n=20, seed=42)
    print(f"  Generated {len(meta_tasks)} type prediction tasks")

    env = standard_env()
    solved = 0
    for name, spec in meta_tasks[:10]:
        result = synthesize(spec, max_depth=1, max_candidates=5000, env=env)
        status = "OK" if result.found else "--"
        if result.found:
            solved += 1
        print(f"  {status}  {name:30s}  cand={result.candidates_explored:5d}"
              + (f"  {result.source}" if result.found else ""))

    print(f"\n  Meta-1.0 solve rate: {solved}/10")

    # ── Phase 4: Can we use synthesis logs to improve? ───────────────
    print("\n--- Phase 4: Does training data predict useful patterns? ---\n")

    # Analyze: for tasks with string input -> number output, which components
    # appear most often in solutions?
    str_to_num = [l for l in logger.solved
                  if l.input_type == "string" and l.output_type == "number"]
    num_to_num = [l for l in logger.solved
                  if l.input_type == "number" and l.output_type == "number"]

    print(f"  String->Number tasks: {len(str_to_num)} solved")
    if str_to_num:
        freq = {}
        for l in str_to_num:
            for c in l.components_used:
                freq[c] = freq.get(c, 0) + 1
        print(f"  Most used components:")
        for name, count in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            print(f"    {name:20s}  {count}/{len(str_to_num)}")

    print(f"\n  Number->Number tasks: {len(num_to_num)} solved")
    if num_to_num:
        freq = {}
        for l in num_to_num:
            for c in l.components_used:
                freq[c] = freq.get(c, 0) + 1
        print(f"  Most used components:")
        for name, count in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            print(f"    {name:20s}  {count}/{len(num_to_num)}")

    avg_str = sum(l.candidates_explored for l in str_to_num) / len(str_to_num) if str_to_num else 0
    avg_num = sum(l.candidates_explored for l in num_to_num) / len(num_to_num) if num_to_num else 0
    print(f"\n  Avg candidates: str->num={avg_str:.0f}, num->num={avg_num:.0f}")
    print(f"  Insight: {'str->num is harder' if avg_str > avg_num else 'num->num is harder'}")

    print("\n" + "=" * 60)
    print("The training data reveals which components to prioritize")
    print("for each task type. This is what Meta-1 should learn.")
    print("=" * 60)


if __name__ == "__main__":
    run()
