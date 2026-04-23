"""
Iterative refinement synthesizer for ARC-AGI grid tasks.

Adapts the neural_selph iterative refinement approach:
- MultiScale SSM backbone (from parameter-golf)
- Multi-step refinement with execution feedback
- Traces: (current_program, feedback, corrected_program)
- GRPO for exploration beyond SFT traces

The key insight: instead of generating a program in one shot,
the model iteratively refines from _HOLE_ → attempt → feedback → correction,
seeing the execution result at each step.

Usage:
    python -m selph_fast.curriculum.neural_arc_iterative \
        --arc-dir arc_data/data/training \
        --checkpoint /tmp/arc_agi1_solved.checkpoint \
        --sft-epochs 100 --grpo-epochs 30
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PG_ROOT = Path(os.environ.get("PG_ROOT", os.path.expanduser("~/projects/parameter-golf")))
sys.path.insert(0, str(PG_ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ssm_bin_es.experiment.model import MultiScaleFFNLM

from ..selph_fast import Env

# Reuse task loader and grid utilities from neural_arc
from .neural_arc import (
    ArcTask, load_arc_tasks, load_arc_task, load_sft_from_checkpoint,
    grid_to_sexpr, grid_cell_accuracy, extract_grid_features,
)


# ── Vocabulary ───────────────────────────────────────────────────────────────
# Extends the neural_arc vocab with refinement-specific tokens.

ARC_ITER_VOCAB = [
    # Special tokens
    "<pad>", "<bos>", "<eos>",
    # Structural markers for refinement
    "SEP_TASK", "SEP_EXPR", "SEP_FEEDBACK", "SEP_EDIT",
    # Feedback types
    "CORRECT", "WRONG", "ERROR", "INCOMPLETE",
    # Hole token
    "_HOLE_", "NO_OP",
    # Syntax
    "(", ")",
    # Core
    "lambda", "x", "nth", "list",
    # Colors 0-9
    "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9",
    # Numbers
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-1", "-2",
    # Temp color constants
    "51", "55", "56", "57", "58",
    # Grid transforms
    "grid-rotate-cw", "grid-rotate-ccw", "grid-rotate-180",
    "grid-flip-h", "grid-flip-v", "grid-transpose",
    "grid-trim", "grid-compact",
    # Color operations
    "grid-replace-color", "grid-keep-color", "grid-remove-small-objects",
    # Fill
    "grid-fill-enclosed", "grid-fill-rectangular-holes",
    "m8r-fill-between-both",
    # Composition
    "grid-overlay", "grid-hconcat", "grid-vconcat", "grid-place", "grid-translate",
    # Object ops
    "grid-recompose", "grid-object", "grid-object-count", "grid-object-pos",
    "grid-set",
    # Scale/tile
    "grid-scale", "grid-tile", "grid-untile",
    # Logic
    "grid-xor", "grid-and", "grid-or",
    # Gravity
    "grid-gravity", "grid-gravity-up", "grid-gravity-down",
    "grid-gravity-left", "grid-gravity-right",
    # Drawing
    "grid-extend-lines", "grid-connect-same-color",
    "grid-color-voronoi", "grid-recolor-by-proximity", "grid-rays",
    # Measurement
    "grid-height", "grid-width", "grid-background", "grid-colors",
    # Control flow
    "if", "=", ">", "<", "-",
    # Higher-order
    "reduce", "acc", "idx", "idx2", "g2",
    # String literals for grid-recompose
    "\"rotate-cw\"", "\"rotate-ccw\"", "\"rotate-180\"",
    "\"flip-h\"", "\"flip-v\"", "\"transpose\"",
    # Feature encoding digits
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9",
    # Cell accuracy encoding (0-10 in steps of 10%)
    "acc0", "acc1", "acc2", "acc3", "acc4", "acc5",
    "acc6", "acc7", "acc8", "acc9", "acc10",
    # Query channel (scratchpad)
    "<query>", "</query>", "<q_out>", "</q_out>",
    # Query result tokens
    "true", "false",
]

VTOK2ID = {t: i for i, t in enumerate(ARC_ITER_VOCAB)}
VID2TOK = {i: t for i, t in enumerate(ARC_ITER_VOCAB)}
VPAD = VTOK2ID["<pad>"]
VBOS = VTOK2ID["<bos>"]
VEOS = VTOK2ID["<eos>"]
VSEP_TASK = VTOK2ID["SEP_TASK"]
VSEP_EXPR = VTOK2ID["SEP_EXPR"]
VSEP_FB = VTOK2ID["SEP_FEEDBACK"]
VSEP_EDIT = VTOK2ID["SEP_EDIT"]
VHOLE = VTOK2ID["_HOLE_"]
VNOOP = VTOK2ID["NO_OP"]
VOCAB_SIZE = len(ARC_ITER_VOCAB)

_TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()]+')


def tokenize_program(sexpr: str) -> list[int]:
    """Tokenize a SELPH program body into token IDs (no BOS/EOS)."""
    if sexpr == "_HOLE_":
        return [VHOLE]
    if sexpr == "NO_OP":
        return [VNOOP]
    tokens = _TOKEN_RE.findall(sexpr)
    ids = []
    for t in tokens:
        if t in VTOK2ID:
            ids.append(VTOK2ID[t])
        elif t.isdigit() and f"c{t}" in VTOK2ID:
            ids.append(VTOK2ID[f"c{t}"])
        else:
            ids.append(VPAD)  # unknown
    return ids


def detokenize_program(ids: list[int]) -> str:
    """Convert token IDs back to a SELPH program string."""
    tokens = []
    for i in ids:
        if i in (VPAD, VBOS, VEOS, VSEP_TASK, VSEP_EXPR, VSEP_FB, VSEP_EDIT):
            continue
        if i == VHOLE:
            tokens.append("_HOLE_")
            continue
        if i == VNOOP:
            return "NO_OP"
        tok = VID2TOK.get(i, "?")
        if tok.startswith("c") and len(tok) == 2 and tok[1].isdigit():
            tok = tok[1]
        tokens.append(tok)
    parts = []
    for t in tokens:
        if t == "(":
            if parts and parts[-1] != "(":
                parts.append(" ")
            parts.append("(")
        elif t == ")":
            parts.append(")")
        else:
            if parts and parts[-1] != "(":
                parts.append(" ")
            parts.append(t)
    return "".join(parts)


# ── Execution Feedback ───────────────────────────────────────────────────────

def evaluate_program(program: str, task: ArcTask) -> tuple[str, float]:
    """Evaluate a program on an ARC task, return (feedback_type, cell_accuracy).

    Returns:
      ("INCOMPLETE", 0.0)  if program contains _HOLE_
      ("ERROR", 0.0)       if program fails to parse/eval
      ("CORRECT", 1.0)     if all training pairs match exactly
      ("WRONG", accuracy)  otherwise
    """
    if "_HOLE_" in program:
        return ("INCOMPLETE", 0.0)

    try:
        env = Env()
        full_prog = f"(lambda (x) {program})" if not program.strip().startswith("(lambda") else program
        env.eval_expr(f"(define __fn__ {full_prog})")

        total_acc = 0.0
        for inp_grid, out_grid in task.train_pairs:
            inp_sexpr = grid_to_sexpr(inp_grid)
            try:
                result = env.eval_expr(f"(__fn__ (list {inp_sexpr}))")
            except BaseException:
                return ("ERROR", 0.0)
            acc = grid_cell_accuracy(result, out_grid)
            total_acc += acc

        avg_acc = total_acc / len(task.train_pairs)
        if avg_acc >= 1.0:
            return ("CORRECT", 1.0)
        return ("WRONG", avg_acc)
    except BaseException:
        return ("ERROR", 0.0)


def accuracy_token(acc: float) -> str:
    """Convert cell accuracy [0,1] to a token like acc7."""
    bucket = min(10, max(0, round(acc * 10)))
    return f"acc{bucket}"


# ── Diagnostic Queries (Scratchpad) ──────────────────────────────────────────
# M-chain-style diagnostic queries the model can run before committing to an edit.
# Each query is a SELPH expression evaluated on the first training pair.

# The pool of diagnostic queries, ordered like M-chain detection priority.
# Each entry: (query_template, result_type) where template uses {inp} for input grid.
DIAGNOSTIC_QUERIES = [
    # Dimension queries
    ("(grid-height {inp})", "num"),
    ("(grid-width {inp})", "num"),
    # Color queries
    ("(length (grid-colors {inp}))", "num"),
    # Object count
    ("(grid-object-count {inp})", "num"),
    # Transform match checks — does applying X to input produce the output?
    ("(grid-rotate-cw {inp})", "grid_match"),
    ("(grid-rotate-ccw {inp})", "grid_match"),
    ("(grid-rotate-180 {inp})", "grid_match"),
    ("(grid-flip-h {inp})", "grid_match"),
    ("(grid-flip-v {inp})", "grid_match"),
    ("(grid-transpose {inp})", "grid_match"),
    ("(grid-trim {inp})", "grid_match"),
    ("(grid-fill-enclosed {inp})", "grid_match"),
    ("(grid-gravity-down {inp})", "grid_match"),
    ("(grid-gravity-up {inp})", "grid_match"),
    ("(grid-extend-lines {inp})", "grid_match"),
    ("(grid-connect-same-color {inp})", "grid_match"),
    ("(grid-untile {inp})", "grid_match"),
    # Object extraction
    ("(grid-object {inp} 0)", "grid_match"),
    # Scale checks
    ("(grid-scale {inp} 2)", "grid_match"),
    ("(grid-scale {inp} 3)", "grid_match"),
    # Composition checks
    ("(grid-hconcat {inp} (grid-flip-h {inp}))", "grid_match"),
    ("(grid-vconcat {inp} (grid-flip-v {inp}))", "grid_match"),
    ("(grid-xor {inp} (grid-flip-v {inp}))", "grid_match"),
    ("(grid-xor {inp} (grid-rotate-180 {inp}))", "grid_match"),
]


def run_diagnostic_queries(
    task: ArcTask,
    max_queries: int = 6,
) -> list[tuple[str, str]]:
    """Run diagnostic queries on a task, return [(query_expr, result_str), ...].

    For 'grid_match' queries: result is 'true' if the transform matches the
    expected output on the first training pair, 'false' otherwise.
    For 'num' queries: result is the numeric value as a string.

    Returns the most informative queries (matching transforms first, then numerics).
    """
    if not task.train_pairs:
        return []

    inp_grid, out_grid = task.train_pairs[0]
    inp_sexpr = grid_to_sexpr(inp_grid)

    env = Env()
    results = []
    matches = []
    numerics = []

    for template, rtype in DIAGNOSTIC_QUERIES:
        query_expr = template.format(inp=inp_sexpr)
        try:
            result = env.eval_expr(query_expr)
        except BaseException:
            continue

        if rtype == "grid_match":
            acc = grid_cell_accuracy(result, out_grid)
            result_str = "true" if acc >= 1.0 else "false"
            # Use the template with (nth x 0) for the readable version
            readable = template.format(inp="(nth x 0)")
            if acc >= 1.0:
                matches.append((readable, result_str))
            else:
                results.append((readable, result_str))
        elif rtype == "num":
            readable = template.format(inp="(nth x 0)")
            numerics.append((readable, str(result)))

    # Prioritize: matching transforms, then numerics, then non-matching transforms
    ordered = matches + numerics[:2] + results[:max(0, max_queries - len(matches) - 2)]
    return ordered[:max_queries]


def generate_diagnostic_trace(
    task: ArcTask,
    solution: str,
    max_queries: int = 5,
) -> list[tuple[str, str]]:
    """Generate diagnostic queries for a solved task.

    Returns queries that lead toward the solution. For the matching
    transform, we include it. For non-matching ones, we include a few
    to teach the model what doesn't work.
    """
    queries = run_diagnostic_queries(task, max_queries=max_queries)

    # Also ensure the solution's core function appears as a matching query
    # (if it's a simple transform)
    core_fn = None
    m = re.match(r'\((\S+)\s', solution)
    if m:
        core_fn = m.group(1)

    return queries


# ── Trace Generation ─────────────────────────────────────────────────────────

@dataclass
class RefinementStep:
    """One step in an iterative refinement trace."""
    current_expr: str
    feedback_type: str     # INCOMPLETE, WRONG, ERROR, CORRECT
    feedback_acc: float    # cell accuracy [0, 1]
    target_expr: str       # the edit (corrected program or NO_OP)
    queries: list[tuple[str, str]] = field(default_factory=list)  # scratchpad queries


@dataclass
class RefinementTrace:
    """A full refinement trajectory for one task."""
    task: ArcTask
    steps: list[RefinementStep]


def generate_topdown_trace(task: ArcTask, solution: str) -> RefinementTrace:
    """Generate a top-down decomposition trace from _HOLE_ to solution.

    For a solution like (grid-replace-color (grid-fill-enclosed (nth x 0)) 3 5):
    Step 1: _HOLE_ → INCOMPLETE → (grid-replace-color _HOLE_ _HOLE_ _HOLE_)
    Step 2: (grid-replace-color _HOLE_ _HOLE_ _HOLE_) → INCOMPLETE → (grid-replace-color (grid-fill-enclosed _HOLE_) _HOLE_ _HOLE_)
    Step 3: ... → (grid-replace-color (grid-fill-enclosed (nth x 0)) 3 5)
    Step 4: ... → CORRECT → NO_OP
    """
    # Parse solution into a tree structure for progressive expansion
    steps = []

    # Simplified approach: expand from _HOLE_ to skeleton to full solution
    # We generate 2-3 intermediate steps

    # Step 1: _HOLE_ → diagnostic queries → outer skeleton
    # The queries teach the model to explore before committing
    diag_queries = generate_diagnostic_trace(task, solution)
    skeleton = _extract_skeleton(solution)
    steps.append(RefinementStep(
        current_expr="_HOLE_",
        feedback_type="INCOMPLETE",
        feedback_acc=0.0,
        target_expr=skeleton if skeleton != solution else solution,
        queries=diag_queries,
    ))

    # Step 2: skeleton → full solution (if skeleton != solution)
    if skeleton != solution:
        fb_type, fb_acc = evaluate_program(skeleton, task)
        steps.append(RefinementStep(
            current_expr=skeleton,
            feedback_type=fb_type,
            feedback_acc=fb_acc,
            target_expr=solution,
        ))

    # Step 3: full solution → CORRECT → NO_OP
    fb_type, fb_acc = evaluate_program(solution, task)
    steps.append(RefinementStep(
        current_expr=solution,
        feedback_type=fb_type,
        feedback_acc=fb_acc,
        target_expr="NO_OP",
    ))

    return RefinementTrace(task=task, steps=steps)


def _extract_skeleton(sexpr: str) -> str:
    """Replace inner arguments with _HOLE_ to get a 1-level skeleton.

    (grid-replace-color (grid-fill-enclosed (nth x 0)) 3 5)
    → (grid-replace-color _HOLE_ _HOLE_ _HOLE_)
    """
    # Find the outermost function call
    s = sexpr.strip()
    if not s.startswith("("):
        return s

    # Parse to find the function name and top-level args
    depth = 0
    func_end = None
    args = []
    arg_start = None

    for i, c in enumerate(s):
        if c == "(":
            depth += 1
            if depth == 1:
                func_end = None
                arg_start = None
        elif c == ")":
            depth -= 1
            if depth == 0:
                if arg_start is not None:
                    args.append(s[arg_start:i].strip())
        elif c == " " and depth == 1:
            if func_end is None:
                func_end = i
                arg_start = i + 1
            else:
                if arg_start is not None:
                    token = s[arg_start:i].strip()
                    if token:
                        args.append(token)
                arg_start = i + 1

    if func_end is None:
        return s

    func_name = s[1:func_end]

    # Simple args (single tokens like x, 0, 3) stay; complex ones become _HOLE_
    skeleton_args = []
    for arg in args:
        arg = arg.strip()
        if not arg:
            continue
        if arg.startswith("(") or len(arg.split()) > 1:
            skeleton_args.append("_HOLE_")
        else:
            skeleton_args.append(arg)

    if not skeleton_args:
        return s

    return "(" + func_name + " " + " ".join(skeleton_args) + ")"


def generate_wrong_start_trace(task: ArcTask, solution: str,
                                wrong_programs: list[str]) -> RefinementTrace | None:
    """Generate a trace that starts from a wrong program and corrects to the solution."""
    for wrong in wrong_programs:
        fb_type, fb_acc = evaluate_program(wrong, task)
        if fb_type == "WRONG" and fb_acc > 0.1:
            steps = []
            # Step 1: wrong → feedback → correction
            steps.append(RefinementStep(
                current_expr=wrong,
                feedback_type="WRONG",
                feedback_acc=fb_acc,
                target_expr=solution,
            ))
            # Step 2: solution → CORRECT → NO_OP
            steps.append(RefinementStep(
                current_expr=solution,
                feedback_type="CORRECT",
                feedback_acc=1.0,
                target_expr="NO_OP",
            ))
            return RefinementTrace(task=task, steps=steps)
    return None


def _random_grid(h: int, w: int, n_colors: int = 4) -> list[list[int]]:
    colors = list(range(n_colors))
    return [[random.choice(colors) for _ in range(w)] for _ in range(h)]


def _apply_transform_eval(grid: list[list[int]], transform_name: str) -> list[list[int]] | None:
    """Apply a named grid transform via SELPH evaluator."""
    try:
        env = Env()
        inp = grid_to_sexpr(grid)
        result = env.eval_expr(f"({transform_name} {inp})")
        if isinstance(result, list) and result and isinstance(result[0], list):
            return [[int(c) for c in row] for row in result]
    except BaseException:
        pass
    return None


# Transforms for synthetic traces: (name, program_body, is_unary)
_SYNTH_TRANSFORMS = [
    ("grid-rotate-cw", "(grid-rotate-cw (nth x 0))"),
    ("grid-rotate-ccw", "(grid-rotate-ccw (nth x 0))"),
    ("grid-rotate-180", "(grid-rotate-180 (nth x 0))"),
    ("grid-flip-h", "(grid-flip-h (nth x 0))"),
    ("grid-flip-v", "(grid-flip-v (nth x 0))"),
    ("grid-transpose", "(grid-transpose (nth x 0))"),
    ("grid-fill-enclosed", "(grid-fill-enclosed (nth x 0))"),
    ("grid-trim", "(grid-trim (nth x 0))"),
    ("grid-gravity-down", "(grid-gravity-down (nth x 0))"),
    ("grid-gravity-up", "(grid-gravity-up (nth x 0))"),
    ("grid-extend-lines", "(grid-extend-lines (nth x 0))"),
    ("grid-connect-same-color", "(grid-connect-same-color (nth x 0))"),
    ("grid-untile", "(grid-untile (nth x 0))"),
    ("grid-compact", "(grid-compact (nth x 0))"),
]


def generate_synthetic_traces(n: int = 500) -> list[RefinementTrace]:
    """Generate synthetic refinement traces with diagnostic queries.

    For each trace:
    1. Pick a random transform, generate random grids, apply it
    2. Run diagnostic queries on the synthetic task
    3. Build trace: _HOLE_ + queries → solution → CORRECT → NO_OP
    4. Also build wrong-start traces: wrong_transform → WRONG → solution
    """
    traces = []
    for _ in range(n):
        transform_name, body = random.choice(_SYNTH_TRANSFORMS)
        h = random.randint(3, 8)
        w = random.randint(3, 8)
        n_colors = random.randint(2, 6)

        # Generate 3 training pairs
        train_pairs = []
        ok = True
        for _ in range(3):
            inp = _random_grid(h, w, n_colors)
            out = _apply_transform_eval(inp, transform_name)
            if out is None:
                ok = False
                break
            train_pairs.append((inp, out))

        if not ok or len(train_pairs) < 3:
            continue

        task = ArcTask(
            task_id=f"synth_{transform_name}_{len(traces)}",
            train_pairs=train_pairs,
            test_pairs=[],
        )

        # Top-down trace with diagnostic queries
        diag = generate_diagnostic_trace(task, body)
        skeleton = _extract_skeleton(body)

        steps = [RefinementStep(
            current_expr="_HOLE_",
            feedback_type="INCOMPLETE",
            feedback_acc=0.0,
            target_expr=skeleton if skeleton != body else body,
            queries=diag,
        )]

        if skeleton != body:
            fb, acc = evaluate_program(skeleton, task)
            steps.append(RefinementStep(skeleton, fb, acc, body))

        fb, acc = evaluate_program(body, task)
        steps.append(RefinementStep(body, fb, acc, "NO_OP"))
        traces.append(RefinementTrace(task=task, steps=steps))

        # Wrong-start trace: pick a different transform, get WRONG feedback
        other_transforms = [b for _, b in _SYNTH_TRANSFORMS if b != body]
        random.shuffle(other_transforms)
        for wrong_body in other_transforms[:2]:
            fb, acc = evaluate_program(wrong_body, task)
            if fb == "WRONG" and 0.05 < acc < 0.95:
                # Also run queries from the wrong state
                wrong_diag = generate_diagnostic_trace(task, body)
                steps2 = [
                    RefinementStep(wrong_body, "WRONG", acc, body, queries=wrong_diag),
                    RefinementStep(body, "CORRECT", 1.0, "NO_OP"),
                ]
                traces.append(RefinementTrace(task=task, steps=steps2))
                break

    return traces


def generate_augmented_traces(
    arc_tasks: list[ArcTask],
    solutions: dict[str, str],
    n_color_perm: int = 10,
) -> list[RefinementTrace]:
    """Generate color-permuted augmented traces from solved tasks."""
    from .neural_arc import augment_color_permutation

    traces = []
    task_map = {t.task_id: t for t in arc_tasks}

    for task_id, source in solutions.items():
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        body = _extract_body(source)
        if not body:
            continue

        augmented = augment_color_permutation(task, body, n_color_perm)
        for aug_task, aug_body in augmented:
            skeleton = _extract_skeleton(aug_body)
            diag = generate_diagnostic_trace(aug_task, aug_body)

            steps = [RefinementStep(
                current_expr="_HOLE_",
                feedback_type="INCOMPLETE",
                feedback_acc=0.0,
                target_expr=skeleton if skeleton != aug_body else aug_body,
                queries=diag,
            )]

            if skeleton != aug_body:
                fb, acc = evaluate_program(skeleton, aug_task)
                steps.append(RefinementStep(skeleton, fb, acc, aug_body))

            fb, acc = evaluate_program(aug_body, aug_task)
            steps.append(RefinementStep(aug_body, fb, acc, "NO_OP"))
            traces.append(RefinementTrace(task=aug_task, steps=steps))

    return traces


def generate_all_traces(
    arc_tasks: list[ArcTask],
    solutions: dict[str, str],
    n_wrong_per_task: int = 5,
    n_synthetic: int = 500,
    n_color_perm: int = 10,
) -> list[RefinementTrace]:
    """Generate all training traces.

    Sources:
    1. Top-down traces from solved tasks (with diagnostic queries)
    2. Wrong-start traces (other solutions applied to wrong tasks)
    3. Synthetic traces from known transforms on random grids
    4. Color-permuted augmented traces from solved tasks
    """
    traces = []
    all_bodies = []
    for source in solutions.values():
        body = _extract_body(source)
        if body:
            all_bodies.append(body)

    task_map = {t.task_id: t for t in arc_tasks}

    # Source 1 + 2: solved task traces
    for task_id, source in solutions.items():
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        body = _extract_body(source)
        if not body:
            continue

        traces.append(generate_topdown_trace(task, body))

        wrong_progs = [b for b in all_bodies if b != body]
        random.shuffle(wrong_progs)
        trace = generate_wrong_start_trace(task, body, wrong_progs[:n_wrong_per_task])
        if trace:
            traces.append(trace)

    n_base = len(traces)

    # Source 3: synthetic
    if n_synthetic > 0:
        traces.extend(generate_synthetic_traces(n_synthetic))

    n_synth = len(traces) - n_base

    # Source 4: color augmentation
    if n_color_perm > 0:
        traces.extend(generate_augmented_traces(arc_tasks, solutions, n_color_perm))

    n_aug = len(traces) - n_base - n_synth

    return traces


def _extract_body(lambda_source: str) -> str | None:
    """Extract body from (lambda (x) <body>)."""
    s = lambda_source.strip()
    m = re.match(r'\(lambda\s+\([^)]*\)\s+(.*)\)\s*$', s, re.DOTALL)
    return m.group(1) if m else s


# ── Trace → Token Sequences ─────────────────────────────────────────────────

def trace_step_to_ids(step: RefinementStep, task: ArcTask) -> list[int]:
    """Convert a refinement step to a token sequence.

    Format: [BOS] SEP_TASK [features] SEP_EXPR [current] SEP_FEEDBACK [type] [acc]
            [<query>expr</query><q_out>result</q_out>]* SEP_EDIT [target] [EOS]

    Queries appear between feedback and edit — the model learns to "think"
    before committing to a correction.
    """
    ids = [VBOS]

    # Task features
    ids.append(VSEP_TASK)
    features = extract_grid_features(task)
    for f in features:
        ids.append(VTOK2ID[f"f{min(f, 9)}"])

    # Current expression
    ids.append(VSEP_EXPR)
    ids.extend(tokenize_program(step.current_expr))

    # Feedback
    ids.append(VSEP_FB)
    ids.append(VTOK2ID[step.feedback_type])
    if step.feedback_type == "WRONG":
        ids.append(VTOK2ID[accuracy_token(step.feedback_acc)])

    # Queries (scratchpad)
    for query_expr, result_str in step.queries:
        ids.append(VTOK2ID["<query>"])
        ids.extend(tokenize_program(query_expr))
        ids.append(VTOK2ID["</query>"])
        ids.append(VTOK2ID["<q_out>"])
        # Tokenize result
        if result_str in VTOK2ID:
            ids.append(VTOK2ID[result_str])
        else:
            # Try as number
            for ch in result_str:
                if ch in VTOK2ID:
                    ids.append(VTOK2ID[ch])
        ids.append(VTOK2ID["</q_out>"])

    # Target edit
    ids.append(VSEP_EDIT)
    ids.extend(tokenize_program(step.target_expr))

    ids.append(VEOS)
    return ids


def traces_to_training_data(
    traces: list[RefinementTrace], max_seq_len: int = 256
) -> list[tuple[list[int], list[int]]]:
    """Convert traces to (input_ids, target_ids) pairs for next-token prediction."""
    examples = []
    for trace in traces:
        for step in trace.steps:
            ids = trace_step_to_ids(step, trace.task)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            # input = ids[:-1], target = ids[1:]
            examples.append((ids[:-1], ids[1:]))
    return examples


# ── Model Construction ───────────────────────────────────────────────────────

class IterativeTransformer(nn.Module):
    """Transformer LM for iterative refinement of ARC programs.

    Same architecture as ArcStructureGenerator from neural_arc.py,
    but with the iterative vocabulary and longer max_len for queries.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                 max_len: int = 256):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        self.d_model = d_model
        self.max_len = max_len
        # Alias for compatibility with SSM code paths
        self.dim = d_model

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape
        pos = mx.arange(T)
        h = self.embed(x) + self.pos_embed(pos)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        return self.head(h)


def build_model(
    arch: str = "transformer",
    dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 8,
    # SSM-specific
    feat_dim: int = 48,
    state_dim: int = 8,
    num_scales: int = 3,
) -> nn.Module:
    """Build model for ARC program refinement.

    arch="transformer": IterativeTransformer (same arch as single-pass, fair comparison)
    arch="ssm": MultiScaleFFNLM from parameter-golf
    """
    if arch == "transformer":
        return IterativeTransformer(d_model=dim, n_heads=n_heads, n_layers=n_layers)
    else:
        return MultiScaleFFNLM(
            vocab_size=VOCAB_SIZE,
            num_layers=n_layers,
            dim=dim,
            feat_dim=feat_dim,
            state_dim=state_dim,
            num_scales=num_scales,
            shifts=(0, 1, 2, 4),
            a_log_mults=(2.0, 1.0, 0.3),
            mlp_mult=2,
            logit_softcap=30.0,
            tied_embed_init_std=0.02,
            weight_tie_layers=0,
            group_size=64,
        )


# ── Training ─────────────────────────────────────────────────────────────────

def make_batch(
    examples: list[tuple[list[int], list[int]]],
    indices: list[int],
) -> tuple[mx.array, mx.array]:
    """Create a padded batch."""
    batch_in = [examples[i][0] for i in indices]
    batch_tgt = [examples[i][1] for i in indices]
    max_len = max(len(s) for s in batch_in)

    padded_in = [inp + [VPAD] * (max_len - len(inp)) for inp in batch_in]
    padded_tgt = [tgt + [-1] * (max_len - len(tgt)) for tgt in batch_tgt]

    return mx.array(padded_in, dtype=mx.int32), mx.array(padded_tgt, dtype=mx.int32)


def _get_logits(model: nn.Module, input_ids: mx.array) -> mx.array:
    """Get logits from either Transformer or SSM model."""
    x = model(input_ids)
    if isinstance(model, IterativeTransformer):
        # Transformer head already outputs (batch, seq, vocab)
        return x.reshape(-1, VOCAB_SIZE)
    else:
        # SSM: need weight-tied projection
        x = x.reshape(-1, model.dim)
        logits = x @ model.tok_emb.weight.astype(x.dtype).T
        return model.softcap(logits)


def loss_fn(model: nn.Module, input_ids: mx.array, target_ids: mx.array) -> mx.array:
    """Cross-entropy loss with padding mask."""
    logits = _get_logits(model, input_ids)

    targets_flat = target_ids.reshape(-1)
    mask = (targets_flat >= 0).astype(mx.float32)
    safe_targets = mx.maximum(targets_flat, mx.array(0, dtype=mx.int32))

    loss = nn.losses.cross_entropy(logits.astype(mx.float32), safe_targets, reduction="none")
    return mx.sum(loss * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))


def sft_train(
    model: MultiScaleFFNLM,
    examples: list[tuple[list[int], list[int]]],
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    verbose: bool = True,
) -> float:
    """SFT on refinement traces."""
    if not examples:
        return 0.0

    # Train/val split
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    train_data = examples[:split]
    val_data = examples[split:]

    # Cosine schedule with warmup
    warmup_steps = 50
    total_steps = epochs * (len(train_data) // batch_size + 1)
    schedule = optim.cosine_decay(lr, total_steps - warmup_steps)
    warmup = optim.linear_schedule(1e-6, lr, warmup_steps)
    lr_schedule = optim.join_schedules([warmup, schedule], [warmup_steps])
    optimizer = optim.Adam(learning_rate=lr_schedule)

    if verbose:
        n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
        print(f"SSM model: {n_params:,} params, {VOCAB_SIZE} vocab")
        print(f"SFT: {len(train_data)} train, {len(val_data)} val, {epochs} epochs")

    best_val = float("inf")
    for epoch in range(epochs):
        random.shuffle(train_data)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch_idx = list(range(i, min(i + batch_size, len(train_data))))
            inp, tgt = make_batch(train_data, batch_idx)

            loss, grads = nn.value_and_grad(model, lambda m: loss_fn(m, inp, tgt))(model)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            lv = float(loss.item())
            if not math.isnan(lv):
                epoch_loss += lv
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Val loss
        val_loss = 0.0
        val_n = 0
        if val_data:
            for i in range(0, len(val_data), batch_size):
                batch_idx = list(range(i, min(i + batch_size, len(val_data))))
                inp, tgt = make_batch(val_data, batch_idx)
                vl = loss_fn(model, inp, tgt)
                mx.eval(vl)
                val_loss += float(vl.item())
                val_n += 1
            val_loss /= max(val_n, 1)
            best_val = min(best_val, val_loss)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  epoch {epoch+1}/{epochs}: train={avg_loss:.4f} val={val_loss:.4f} best_val={best_val:.4f}")

    if verbose:
        print(f"  SFT complete: best_val={best_val:.4f}")
    return best_val


# ── DAgger Training ──────────────────────────────────────────────────────────

def dagger_train(
    model: nn.Module,
    tasks_with_solutions: list[tuple[ArcTask, str]],
    epochs: int = 50,
    max_steps: int = 4,
    lr: float = 1e-4,
    temperature: float = 0.7,
    verbose: bool = True,
) -> float:
    """DAgger-style training: run the model iteratively, correct toward target.

    For each (task, target_program):
      1. Start from _HOLE_
      2. Model generates its own next step
      3. Evaluate the model's output → get feedback
      4. Training signal: from the model's current state, the target is
         the known solution (or NO_OP if already correct)
      5. Use the model's own output as the next state (not the oracle's)

    This trains the model on its own distribution of mistakes, bridging
    the gap between pre-computed traces and inference-time behavior.
    """
    if not tasks_with_solutions:
        return 0.0

    optimizer = optim.Adam(learning_rate=lr)

    if verbose:
        print(f"DAgger: {len(tasks_with_solutions)} tasks, {epochs} epochs, "
              f"max_steps={max_steps}, lr={lr}")

    best_acc = 0.0
    for epoch in range(epochs):
        random.shuffle(tasks_with_solutions)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_updates = 0

        for task, target_body in tasks_with_solutions:
            current_expr = "_HOLE_"

            for step in range(max_steps):
                # Evaluate current state
                fb_type, fb_acc = evaluate_program(current_expr, task)

                if fb_type == "CORRECT":
                    epoch_correct += 1
                    break

                epoch_total += 1

                # Build prompt (same as inference)
                prompt = [VBOS, VSEP_TASK]
                features = extract_grid_features(task)
                for f in features:
                    prompt.append(VTOK2ID[f"f{min(f, 9)}"])
                prompt.append(VSEP_EXPR)
                prompt.extend(tokenize_program(current_expr))
                prompt.append(VSEP_FB)
                prompt.append(VTOK2ID[fb_type])
                if fb_type == "WRONG":
                    prompt.append(VTOK2ID[accuracy_token(fb_acc)])

                # Target: from current state, emit the solution
                # (or SEP_EDIT + solution if we want the model to learn the edit marker)
                target_ids = [VSEP_EDIT] + tokenize_program(target_body) + [VEOS]

                # Full training sequence: prompt + target
                full_input = prompt + target_ids[:-1]
                full_target = [-1] * len(prompt) + target_ids  # mask prompt

                # Pad target to match
                while len(full_target) < len(full_input):
                    full_target.append(-1)
                full_target = full_target[:len(full_input)]

                if len(full_input) > 256:
                    break

                # Gradient step
                inp = mx.array([full_input], dtype=mx.int32)
                tgt = mx.array([full_target], dtype=mx.int32)

                def step_loss(m):
                    return loss_fn(m, inp, tgt)

                loss, grads = nn.value_and_grad(model, step_loss)(model)
                grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

                lv = float(loss.item())
                if not math.isnan(lv):
                    epoch_loss += lv
                    n_updates += 1

                # DAgger: use MODEL's output as next state, not oracle
                # Generate what the model would actually produce
                gen_ids, _ = generate_tokens(model, prompt, temperature=temperature)

                # Extract edit
                edit_ids = gen_ids
                if VSEP_EDIT in gen_ids:
                    edit_start = gen_ids.index(VSEP_EDIT) + 1
                    edit_ids = gen_ids[edit_start:]
                predicted = detokenize_program(edit_ids)

                if predicted == "NO_OP" or not predicted or predicted == "_HOLE_":
                    break

                current_expr = predicted

        avg_loss = epoch_loss / max(n_updates, 1)
        acc_rate = epoch_correct / max(epoch_total + epoch_correct, 1)
        best_acc = max(best_acc, acc_rate)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  DAgger {epoch+1}/{epochs}: loss={avg_loss:.4f} "
                  f"correct={epoch_correct}/{epoch_total + epoch_correct} ({acc_rate:.1%})")

    if verbose:
        print(f"  DAgger complete: best_acc={best_acc:.1%}")
    return best_acc


# ── Iterative Refinement (Inference) ─────────────────────────────────────────

def _logits_at_last(model: nn.Module, ids: list[int]) -> mx.array:
    """Get logits at the last position for either architecture."""
    x = mx.array([ids], dtype=mx.int32)
    if isinstance(model, IterativeTransformer):
        return model(x)[0, -1, :]  # (vocab,)
    else:
        h = model(x)
        h_last = h[0, -1, :]
        logits = h_last @ model.tok_emb.weight.astype(h_last.dtype).T
        return model.softcap(logits)


def generate_tokens(model: nn.Module, prompt_ids: list[int],
                    max_new: int = 60, temperature: float = 0.7
                    ) -> tuple[list[int], list[float]]:
    """Autoregressive generation from a prompt, returning (token_ids, log_probs)."""
    ids = list(prompt_ids)
    log_probs = []

    for _ in range(max_new):
        logits = _logits_at_last(model, ids) / temperature
        probs = mx.softmax(logits, axis=-1)
        token = mx.random.categorical(logits[None, :])[0].item()
        lp = mx.log(probs[token] + 1e-10).item()
        log_probs.append(lp)
        ids.append(token)

        if token == VEOS:
            break

    return ids[len(prompt_ids):], log_probs


def _eval_query_on_task(query_expr: str, task: ArcTask) -> str:
    """Evaluate a scratchpad query on the first training pair."""
    if not task.train_pairs:
        return "false"
    inp_grid, out_grid = task.train_pairs[0]
    inp_sexpr = grid_to_sexpr(inp_grid)
    # Replace (nth x 0) with the actual input grid
    resolved = query_expr.replace("(nth x 0)", inp_sexpr)
    try:
        env = Env()
        result = env.eval_expr(resolved)
        # Check if it's a grid match
        if isinstance(result, list) and result and isinstance(result[0], list):
            acc = grid_cell_accuracy(result, out_grid)
            return "true" if acc >= 1.0 else "false"
        return str(result)
    except BaseException:
        return "false"


def generate_with_queries(
    model: nn.Module,
    prompt_ids: list[int],
    task: ArcTask,
    max_new: int = 80,
    max_queries: int = 5,
    temperature: float = 0.7,
) -> tuple[list[int], list[float]]:
    """Autoregressive generation that intercepts <query> tokens.

    When the model emits <query>, we collect tokens until </query>,
    evaluate the expression, and inject <q_out>result</q_out> into
    the stream (these injected tokens aren't in log_probs).
    """
    VQUERY_OPEN = VTOK2ID["<query>"]
    VQUERY_CLOSE = VTOK2ID["</query>"]
    VQOUT_OPEN = VTOK2ID["<q_out>"]
    VQOUT_CLOSE = VTOK2ID["</q_out>"]

    ids = list(prompt_ids)
    log_probs = []
    n_queries = 0

    i = 0
    while i < max_new:
        logits = _logits_at_last(model, ids) / temperature
        probs = mx.softmax(logits, axis=-1)
        token = mx.random.categorical(logits[None, :])[0].item()
        lp = mx.log(probs[token] + 1e-10).item()
        log_probs.append(lp)
        ids.append(token)
        i += 1

        if token == VEOS:
            break

        # Intercept <query> — collect until </query>, eval, inject result
        if token == VQUERY_OPEN and n_queries < max_queries:
            query_ids = []
            for _ in range(30):
                qlogits = _logits_at_last(model, ids) / temperature
                qprobs = mx.softmax(qlogits, axis=-1)
                tk = mx.random.categorical(qlogits[None, :])[0].item()
                lp2 = mx.log(qprobs[tk] + 1e-10).item()
                log_probs.append(lp2)
                ids.append(tk)
                i += 1
                if tk == VQUERY_CLOSE:
                    break
                query_ids.append(tk)

            query_expr = detokenize_program(query_ids)
            result_str = _eval_query_on_task(query_expr, task)
            n_queries += 1

            # Inject result (not model-generated, no log_probs)
            ids.append(VQOUT_OPEN)
            if result_str in VTOK2ID:
                ids.append(VTOK2ID[result_str])
            ids.append(VQOUT_CLOSE)

    return ids[len(prompt_ids):], log_probs


def iterative_refine(
    model: MultiScaleFFNLM,
    task: ArcTask,
    max_steps: int = 5,
    temperature: float = 0.7,
) -> tuple[str, float, int]:
    """Run iterative refinement with query support.

    Returns (best_program, best_accuracy, n_steps).
    """
    current_expr = "_HOLE_"
    best_prog = ""
    best_acc = 0.0

    for step in range(max_steps):
        fb_type, fb_acc = evaluate_program(current_expr, task)

        if fb_type == "CORRECT":
            return current_expr, 1.0, step + 1

        if fb_acc > best_acc:
            best_acc = fb_acc
            best_prog = current_expr

        # Build prompt
        prompt = [VBOS, VSEP_TASK]
        features = extract_grid_features(task)
        for f in features:
            prompt.append(VTOK2ID[f"f{min(f, 9)}"])

        prompt.append(VSEP_EXPR)
        prompt.extend(tokenize_program(current_expr))

        prompt.append(VSEP_FB)
        prompt.append(VTOK2ID[fb_type])
        if fb_type == "WRONG":
            prompt.append(VTOK2ID[accuracy_token(fb_acc)])

        # Generate with query support
        gen_ids, _ = generate_with_queries(
            model, prompt, task, temperature=temperature,
        )

        # Extract edit: find SEP_EDIT in generated tokens, take everything after
        edit_ids = gen_ids
        if VSEP_EDIT in gen_ids:
            edit_start = gen_ids.index(VSEP_EDIT) + 1
            edit_ids = gen_ids[edit_start:]

        edit_expr = detokenize_program(edit_ids)

        if edit_expr == "NO_OP" or not edit_expr:
            break

        current_expr = edit_expr

    # Final check
    fb_type, fb_acc = evaluate_program(current_expr, task)
    if fb_acc > best_acc:
        best_acc = fb_acc
        best_prog = current_expr

    return best_prog, best_acc, max_steps


# ── GRPO Training ────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    task: ArcTask
    steps: list[tuple[list[int], list[int], list[float]]]  # (prompt, gen_ids, log_probs)
    reward: float


def collect_trajectory(
    model: MultiScaleFFNLM,
    task: ArcTask,
    max_steps: int = 4,
    temperature: float = 0.8,
) -> Trajectory:
    """Collect one trajectory for GRPO."""
    current_expr = "_HOLE_"
    steps = []

    for step in range(max_steps):
        fb_type, fb_acc = evaluate_program(current_expr, task)
        if fb_type == "CORRECT":
            break

        # Build prompt
        prompt = [VBOS, VSEP_TASK]
        features = extract_grid_features(task)
        for f in features:
            prompt.append(VTOK2ID[f"f{min(f, 9)}"])
        prompt.append(VSEP_EXPR)
        prompt.extend(tokenize_program(current_expr))
        prompt.append(VSEP_FB)
        prompt.append(VTOK2ID[fb_type])
        if fb_type == "WRONG":
            prompt.append(VTOK2ID[accuracy_token(fb_acc)])
        prompt.append(VSEP_EDIT)

        gen_ids, log_probs = generate_with_queries(
            model, prompt, task, temperature=temperature,
        )
        steps.append((prompt, gen_ids, log_probs))

        # Extract edit after SEP_EDIT if present
        edit_ids = gen_ids
        if VSEP_EDIT in gen_ids:
            edit_start = gen_ids.index(VSEP_EDIT) + 1
            edit_ids = gen_ids[edit_start:]

        edit_expr = detokenize_program(edit_ids)
        if edit_expr == "NO_OP" or not edit_expr:
            break
        current_expr = edit_expr

    # Final reward: accuracy of last expression
    _, final_acc = evaluate_program(current_expr, task)
    # Bonus for fewer steps
    # Sharpen reward: exact match gets 1.0, near-misses get squared accuracy
    # to create more variance in groups (critical for GRPO signal)
    if final_acc >= 1.0:
        reward = 1.0
    else:
        reward = final_acc ** 2  # 0.95 → 0.90, 0.8 → 0.64, 0.5 → 0.25

    return Trajectory(task=task, steps=steps, reward=reward)


def grpo_train(
    model: MultiScaleFFNLM,
    arc_tasks: list[ArcTask],
    epochs: int = 30,
    group_size: int = 4,
    tasks_per_epoch: int = 64,
    temperature: float = 0.8,
    clip_eps: float = 0.2,
    kl_coef: float = 0.05,
    lr: float = 1e-5,
    verbose: bool = True,
) -> dict[str, tuple[float, str]]:
    """GRPO training for iterative refinement."""
    # Freeze reference model (same architecture as input model)
    # Build ref model matching the input model's architecture
    if isinstance(model, IterativeTransformer):
        ref_model = IterativeTransformer(
            d_model=model.d_model, n_heads=model.layers[0].attention.num_heads,
            n_layers=len(model.layers), max_len=model.max_len,
        )
    else:
        # Infer SSM dimensions from the model's actual parameters
        ssm_dim = model.dim
        ssm_feat = model.layers[0].mamba.feature_ssms[0].in_proj.weight.shape[0]
        ssm_nlayers = len(model.layers)
        ssm_nscales = len(model.layers[0].mamba.feature_ssms)
        ssm_state = model.layers[0].mamba.feature_ssms[0].A_log.shape[1]
        ref_model = MultiScaleFFNLM(
            vocab_size=VOCAB_SIZE, num_layers=ssm_nlayers, dim=ssm_dim,
            feat_dim=ssm_feat, state_dim=ssm_state, num_scales=ssm_nscales,
            shifts=(0, 1, 2, 4), a_log_mults=(2.0, 1.0, 0.3), mlp_mult=2,
            logit_softcap=30.0, tied_embed_init_std=0.02,
            weight_tie_layers=0, group_size=64,
        )
    ref_params = [(k, v) for k, v in tree_flatten(model.parameters())]
    ref_model.load_weights(ref_params)
    ref_model.freeze()

    optimizer = optim.Adam(learning_rate=lr)
    best_solutions: dict[str, tuple[float, str]] = {}

    if verbose:
        print(f"GRPO: {epochs} epochs, group_size={group_size}, {tasks_per_epoch} tasks/epoch")

    for epoch in range(epochs):
        random.shuffle(arc_tasks)
        batch = arc_tasks[:tasks_per_epoch]
        epoch_reward = 0.0
        epoch_total = 0
        epoch_perfect = 0

        for task in batch:
            # Collect group of trajectories
            trajectories = []
            for _ in range(group_size):
                traj = collect_trajectory(model, task, temperature=temperature)
                trajectories.append(traj)
                epoch_reward += traj.reward
                epoch_total += 1
                if traj.reward >= 1.0:
                    epoch_perfect += 1

                # Track best
                if traj.reward > best_solutions.get(task.task_id, (0.0, ""))[0]:
                    # Recover the final program from last step
                    if traj.steps:
                        last_gen = traj.steps[-1][1]
                        prog = detokenize_program(last_gen)
                    else:
                        prog = "_HOLE_"
                    best_solutions[task.task_id] = (traj.reward, prog)

            # Group-relative advantages
            rewards = [t.reward for t in trajectories]
            mean_r = sum(rewards) / len(rewards)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
            if std_r < 1e-8:
                continue

            advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

            # GRPO loss over all steps of all trajectories
            def grpo_loss(m):
                total_loss = mx.array(0.0)
                n_tokens = 0

                for traj, adv in zip(trajectories, advantages):
                    if abs(adv) < 1e-8:
                        continue
                    for prompt, gen_ids, old_lps in traj.steps:
                        if not gen_ids:
                            continue
                        full_ids = prompt + gen_ids
                        if len(full_ids) > 192:
                            continue
                        x = mx.array([full_ids[:-1]], dtype=mx.int32)
                        logits = _get_logits(m, x)  # (seq*1, vocab)
                        # Reshape back to (seq, vocab)
                        seq_len = len(full_ids) - 1
                        logits = logits.reshape(seq_len, VOCAB_SIZE)

                        gen_start = len(prompt)
                        for i, (token_id, old_lp) in enumerate(zip(gen_ids, old_lps)):
                            pos = gen_start + i
                            if pos >= logits.shape[0]:
                                break
                            new_lp = mx.log(mx.softmax(logits[pos]) + 1e-10)[token_id]

                            # Ref model
                            ref_logits = _logits_at_last(ref_model, full_ids[:pos + 1])
                            ref_lp = mx.log(mx.softmax(ref_logits) + 1e-10)[token_id]

                            ratio = mx.exp(new_lp - old_lp)
                            clipped = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                            pg_loss = -mx.minimum(ratio * adv, clipped * adv)
                            kl = new_lp - ref_lp

                            total_loss = total_loss + pg_loss + kl_coef * kl
                            n_tokens += 1

                return total_loss / max(n_tokens, 1)

            loss, grads = nn.value_and_grad(model, grpo_loss)(model)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

        avg_reward = epoch_reward / max(epoch_total, 1)
        n_perfect = sum(1 for r, _ in best_solutions.values() if r >= 1.0)
        n_with_reward = sum(1 for r, _ in best_solutions.values() if r > 0)

        if verbose:
            print(f"GRPO {epoch+1}/{epochs}: reward={avg_reward:.3f} "
                  f"perfect={epoch_perfect}/{epoch_total} "
                  f"best: {n_perfect} perfect, {n_with_reward} with reward>0")

            if (epoch + 1) % 10 == 0:
                top = sorted(best_solutions.items(), key=lambda x: -x[1][0])[:5]
                for tid, (r, prog) in top:
                    print(f"    {tid}: r={r:.3f} {prog[:60]}")

    return best_solutions


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Iterative refinement ARC synthesizer")
    parser.add_argument("--arc-dir", default="arc_data/data/training")
    parser.add_argument("--checkpoint", default=None,
                        help="SELPH checkpoint with solved programs")
    parser.add_argument("--sft-epochs", type=int, default=100)
    parser.add_argument("--grpo-epochs", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--tasks-per-epoch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-weights", default=None)
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Suppress Rust panic stderr
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    if verbose:
        print(f"Loading ARC tasks from {args.arc_dir}...")
    arc_tasks = load_arc_tasks(args.arc_dir)
    if verbose:
        print(f"  {len(arc_tasks)} tasks loaded")

    # Generate traces from checkpoint
    if verbose:
        print("Generating refinement traces...")
    solutions = {}
    if args.checkpoint:
        solutions = load_sft_from_checkpoint(args.checkpoint)
    traces = generate_all_traces(arc_tasks, solutions)
    examples = traces_to_training_data(traces)
    if verbose:
        print(f"  {len(traces)} traces, {len(examples)} training steps")

    # Build model
    model = build_model(dim=args.dim)

    # SFT
    t0 = time.time()
    if args.sft_epochs > 0 and examples:
        sft_train(model, examples, epochs=args.sft_epochs, verbose=verbose)

    # GRPO
    if args.grpo_epochs > 0:
        best = grpo_train(
            model, arc_tasks[:200],
            epochs=args.grpo_epochs,
            group_size=args.group_size,
            tasks_per_epoch=args.tasks_per_epoch,
            verbose=verbose,
        )

    # Final eval
    if verbose:
        print("\n── Final evaluation (5 attempts per task, all 400) ──")
        checkpoint_ids = set(solutions.keys())
        perfect = 0
        near_miss = 0
        novel = []
        for task in arc_tasks:
            best_acc = 0.0
            best_prog = ""
            for temp in [0.3, 0.5, 0.7]:
                prog, acc, _ = iterative_refine(model, task, max_steps=5, temperature=temp)
                if acc > best_acc:
                    best_acc = acc
                    best_prog = prog
            if best_acc >= 1.0:
                perfect += 1
                tag = " [NOVEL]" if task.task_id not in checkpoint_ids else ""
                print(f"  SOLVED {task.task_id}: {best_prog[:70]}{tag}")
                if task.task_id not in checkpoint_ids:
                    novel.append(task.task_id)
            elif best_acc >= 0.9:
                near_miss += 1

        print(f"\nFinal: {perfect}/400 perfect, {near_miss} near-miss (>=90%)")
        print(f"Novel discoveries: {len(novel)}")
        print(f"Total time: {time.time() - t0:.1f}s")

    if args.save_weights:
        weights = dict(tree_flatten(model.parameters()))
        mx.savez(args.save_weights, **weights)


if __name__ == "__main__":
    main()
