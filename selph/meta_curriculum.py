"""Meta-curriculum for SELPH growing system.

Defines meta-level tasks that teach the system about its own
synthesis process. Each meta-task generates training data from
synthesis logs and frames it as a standard synthesis problem.

Meta-1.0: Predict output type from examples
Meta-1.1: Predict useful component categories
Meta-1.2: Predict per-component priorities
"""

from __future__ import annotations
import random
from typing import Any
from .ast import Spec, GoalExamples, Symbol, Number, String
from .eval import standard_env, Env, Namespace
from .logger import SynthesisLogger, SynthesisLog


# ── Meta-1.0: Predict output type ────────────────────────────────────

def generate_type_prediction_tasks(n: int = 50, seed: int = 42) -> list[tuple[str, Spec]]:
    """Generate tasks for Meta-1.0: predict output type from examples.

    Each task: given a set of input/output pairs encoded as a string,
    predict the output type ("number" or "string").

    The input is a string encoding of the examples:
      "1->2 3->6 5->10" (number examples)
      "hello->HELLO world->WORLD" (string examples)

    The output is 0.0 (number) or 1.0 (string).

    This teaches the system the most basic meta-skill: look at examples
    and determine what type of output is needed.
    """
    rng = random.Random(seed)
    tasks = []

    # Number -> Number examples
    num_fns = [
        ("add1", lambda x: x + 1),
        ("double", lambda x: x * 2),
        ("negate", lambda x: -x),
        ("abs", lambda x: abs(x)),
        ("square", lambda x: x * x),
        ("mod3", lambda x: x % 3),
    ]

    for fn_name, fn in num_fns:
        inputs_raw = [rng.randint(-10, 10) for _ in range(5)]
        pairs_str = " ".join(f"{x}->{int(fn(x))}" for x in inputs_raw)
        tasks.append((
            f"type_num_{fn_name}",
            Spec(
                type_expr=Symbol("number"),
                goal=GoalExamples(pairs=((String(pairs_str), Number(0.0)),))
            )
        ))

    # String -> String examples
    str_fns = [
        ("upper", lambda s: s.upper()),
        ("lower", lambda s: s.lower()),
        ("reverse", lambda s: s[::-1]),
    ]

    words = ["hello", "world", "foo", "bar", "test", "abc", "xyz",
             "HELLO", "WORLD", "Python", "Rust", "code"]

    for fn_name, fn in str_fns:
        chosen = rng.sample(words, 5)
        pairs_str = " ".join(f"{w}->{fn(w)}" for w in chosen)
        tasks.append((
            f"type_str_{fn_name}",
            Spec(
                type_expr=Symbol("number"),
                goal=GoalExamples(pairs=((String(pairs_str), Number(1.0)),))
            )
        ))

    # String -> Number examples (cross-type)
    for _ in range(4):
        chosen = rng.sample(words, 5)
        pairs_str = " ".join(f"{w}->{len(w)}" for w in chosen)
        tasks.append((
            f"type_num_strlen_{_}",
            Spec(
                type_expr=Symbol("number"),
                goal=GoalExamples(pairs=((String(pairs_str), Number(0.0)),))
            )
        ))

    # Generate multiple random examples for each to make it a real synthesis task
    # with enough examples to prevent overfitting
    real_tasks = []
    for name, single_spec in tasks:
        # Expand: create 10 variants of the same type
        variants = []
        for variant_seed in range(10):
            vrng = random.Random(seed + hash(name) + variant_seed)

            if "type_num" in name:
                fn_idx = vrng.randint(0, len(num_fns) - 1)
                fn_name, fn = num_fns[fn_idx]
                inputs_raw = [vrng.randint(-10, 10) for _ in range(5)]
                pairs_str = " ".join(f"{x}->{int(fn(x))}" for x in inputs_raw)
                variants.append((String(pairs_str), Number(0.0)))
            else:
                fn_idx = vrng.randint(0, len(str_fns) - 1)
                fn_name, fn = str_fns[fn_idx]
                chosen = vrng.sample(words, min(5, len(words)))
                pairs_str = " ".join(f"{w}->{fn(w)}" for w in chosen)
                variants.append((String(pairs_str), Number(1.0)))

        real_tasks.append((
            name,
            Spec(
                type_expr=Symbol("number"),
                goal=GoalExamples(pairs=tuple(variants))
            )
        ))

    return real_tasks[:n]


# ── Meta-1.1: Predict useful component categories ────────────────────

def generate_component_prediction_tasks(
    logger: SynthesisLogger, seed: int = 42
) -> list[tuple[str, Spec]]:
    """Generate tasks for Meta-1.1: predict useful component categories.

    Uses actual synthesis logs to create training data.
    Each task: given task features, predict which component category is most useful.

    Component categories:
      0 = arithmetic (add, subtract, multiply, etc.)
      1 = string-transform (upper, lower, reverse, trim)
      2 = comparison (min, max, <, >)
      3 = type-bridge (fst, snd, string-length, to-number)
    """
    tasks = []

    category_map = {
        "add": 0, "subtract": 0, "multiply": 0, "divide": 0, "modulo": 0,
        "abs": 0, "negate": 0,
        "string-upper": 1, "string-lower": 1, "string-reverse": 1, "string-trim": 1,
        "min": 2, "max": 2, "<": 2, ">": 2,
        "fst": 3, "snd": 3, "string-length": 3, "to-number": 3, "to-string": 3,
    }

    for i, log in enumerate(logger.solved):
        # Determine the primary category used in this solution
        category_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for comp in log.components_used:
            cat = category_map.get(comp, -1)
            if cat >= 0:
                category_counts[cat] += 1

        primary_category = max(category_counts, key=category_counts.get)
        if category_counts[primary_category] == 0:
            continue

        # Encode task features as a string
        features = f"{log.input_type} {log.output_type} {log.num_examples}"
        tasks.append((
            f"cat_{i}_{log.task_name}",
            Spec(
                type_expr=Symbol("number"),
                goal=GoalExamples(pairs=((String(features), Number(float(primary_category))),))
            )
        ))

    return tasks


# ── Generate training data from a curriculum run ─────────────────────

def log_curriculum_run(results: list, tasks: list, stage: int,
                       logger: SynthesisLogger):
    """Record synthesis results from a curriculum stage into the logger."""
    for (task_name, synth_result), task in zip(results, tasks):
        log = SynthesisLog(
            task_name=task_name,
            stage=stage,
            found=synth_result.found if hasattr(synth_result, 'found') else False,
            source=synth_result.source if hasattr(synth_result, 'source') else "",
            candidates_explored=synth_result.candidates_explored if hasattr(synth_result, 'candidates_explored') else 0,
        )

        # Extract features from spec
        if hasattr(task, 'spec'):
            spec = task.spec
        else:
            spec = task

        if isinstance(spec, Spec) and spec.goal and hasattr(spec.goal, 'pairs'):
            pairs = spec.goal.pairs
            if pairs:
                first_in = pairs[0][0]
                first_out = pairs[0][1]
                log.input_type = "string" if isinstance(first_in, String) else "number"
                log.output_type = "string" if isinstance(first_out, String) else "number"
                log.num_examples = len(pairs)
                log.example_inputs = [p[0].value if hasattr(p[0], 'value') else repr(p[0]) for p in pairs[:5]]
                log.example_outputs = [p[1].value if hasattr(p[1], 'value') else repr(p[1]) for p in pairs[:5]]

        # Extract components from solution
        if log.found and log.source:
            from .logger import extract_components_from_source, estimate_depth
            log.components_used = extract_components_from_source(log.source)
            log.solution_depth = estimate_depth(log.source)

        logger.log(log)
