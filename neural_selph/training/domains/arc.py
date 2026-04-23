"""ARC grid domain adapter.

Wraps the ARC-specific vocabulary, PyO3 evaluation, diagnostic queries,
trace generation, and color augmentation behind the Domain protocol.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Import ARC-specific utilities from the existing curriculum module
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "selph_fast"))

from selph_fast.selph_fast import Env

from ..base import Domain, RefinementStep, RefinementTrace

# ── Vocabulary (from neural_arc_iterative.py) ────────────────────────────────

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

_TOK2ID = {t: i for i, t in enumerate(ARC_ITER_VOCAB)}
_ID2TOK = {i: t for i, t in enumerate(ARC_ITER_VOCAB)}
_TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()]+')


# ── ARC Task ─────────────────────────────────────────────────────────────────

@dataclass
class ArcTask:
    task_id: str
    train_pairs: list[tuple[list[list[int]], list[list[int]]]]
    test_pairs: list[tuple[list[list[int]], list[list[int]] | None]]


def load_arc_task(json_path: str) -> ArcTask:
    with open(json_path) as f:
        data = json.load(f)
    task_id = Path(json_path).stem
    train = [(p["input"], p["output"]) for p in data["train"]]
    test = [(p["input"], p.get("output")) for p in data.get("test", [])]
    return ArcTask(task_id=task_id, train_pairs=train, test_pairs=test)


def load_arc_tasks(dir_path: str) -> list[ArcTask]:
    tasks = []
    for p in sorted(Path(dir_path).glob("*.json")):
        tasks.append(load_arc_task(str(p)))
    return tasks


# ── Grid Helpers ─────────────────────────────────────────────────────────────

def grid_to_sexpr(grid: list[list[int]]) -> str:
    rows = " ".join(
        "(list " + " ".join(str(c) for c in row) + ")"
        for row in grid
    )
    return f"(list {rows})"


def grid_cell_accuracy(pred: object, expected: list[list[int]]) -> float:
    if not isinstance(pred, list) or not pred:
        return 0.0
    try:
        pred_flat = [int(c) for row in pred for c in row]
        exp_flat = [c for row in expected for c in row]
    except (TypeError, ValueError):
        return 0.0
    if len(pred_flat) == 0 and len(exp_flat) == 0:
        return 1.0
    if len(pred_flat) != len(exp_flat):
        min_len = min(len(pred_flat), len(exp_flat))
        if min_len == 0:
            return 0.0
        matches = sum(1 for a, b in zip(pred_flat[:min_len], exp_flat[:min_len]) if a == b)
        return matches / max(len(pred_flat), len(exp_flat)) * 0.5
    matches = sum(1 for a, b in zip(pred_flat, exp_flat) if a == b)
    return matches / len(exp_flat)


# ── Feature Extraction ───────────────────────────────────────────────────────

def _grid_colors(grid):
    return {c for row in grid for c in row}

def _object_count_estimate(grid, bg=0):
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    count = 0
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                count += 1
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < h and 0 <= cc < w and not visited[cr][cc] and grid[cr][cc] != bg:
                        visited[cr][cc] = True
                        stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
    return count

def _has_h_symmetry(grid):
    h = len(grid)
    return all(grid[r] == grid[h - 1 - r] for r in range(h // 2))

def _has_v_symmetry(grid):
    return all(row[c] == row[len(row) - 1 - c] for row in grid for c in range(len(row) // 2))


def extract_grid_features(task: ArcTask) -> list[int]:
    """Compact 36-int feature vector for an ARC task. Values in [0, 9]."""
    features = []
    dims_preserved = True
    colors_preserved = True

    for inp, out in task.train_pairs[:4]:
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        in_colors = _grid_colors(inp)
        out_colors = _grid_colors(out)
        obj_count = _object_count_estimate(inp)
        if ih != oh or iw != ow:
            dims_preserved = False
        if in_colors != out_colors:
            colors_preserved = False
        size_change = 0 if ih * iw == oh * ow else (1 if oh * ow > ih * iw else 2)
        features.extend([
            min(ih, 9), min(iw, 9), min(oh, 9), min(ow, 9),
            min(len(in_colors), 9), min(len(out_colors), 9),
            size_change, min(obj_count, 9),
        ])

    while len(features) < 32:
        features.append(0)

    has_sym = any(_has_h_symmetry(inp) or _has_v_symmetry(inp) for inp, _ in task.train_pairs[:4])
    max_area = max((len(inp) * (len(inp[0]) if inp else 0) for inp, _ in task.train_pairs[:4]), default=0)
    features.extend([
        1 if dims_preserved else 0,
        1 if colors_preserved else 0,
        1 if has_sym else 0,
        min(max_area // 100, 9),
    ])
    return features


# ── Diagnostic Queries ───────────────────────────────────────────────────────

DIAGNOSTIC_QUERIES = [
    ("(grid-height {inp})", "num"),
    ("(grid-width {inp})", "num"),
    ("(length (grid-colors {inp}))", "num"),
    ("(grid-object-count {inp})", "num"),
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
    ("(grid-object {inp} 0)", "grid_match"),
    ("(grid-scale {inp} 2)", "grid_match"),
    ("(grid-scale {inp} 3)", "grid_match"),
    ("(grid-hconcat {inp} (grid-flip-h {inp}))", "grid_match"),
    ("(grid-vconcat {inp} (grid-flip-v {inp}))", "grid_match"),
    ("(grid-xor {inp} (grid-flip-v {inp}))", "grid_match"),
    ("(grid-xor {inp} (grid-rotate-180 {inp}))", "grid_match"),
]


def run_diagnostic_queries(task: ArcTask, max_queries: int = 6) -> list[tuple[str, str]]:
    if not task.train_pairs:
        return []
    inp_grid, out_grid = task.train_pairs[0]
    inp_sexpr = grid_to_sexpr(inp_grid)
    env = Env()

    matches, numerics, others = [], [], []
    for template, rtype in DIAGNOSTIC_QUERIES:
        query_expr = template.format(inp=inp_sexpr)
        try:
            result = env.eval_expr(query_expr)
        except BaseException:
            continue
        readable = template.format(inp="(nth x 0)")
        if rtype == "grid_match":
            acc = grid_cell_accuracy(result, out_grid)
            result_str = "true" if acc >= 1.0 else "false"
            (matches if acc >= 1.0 else others).append((readable, result_str))
        elif rtype == "num":
            numerics.append((readable, str(result)))

    ordered = matches + numerics[:2] + others[:max(0, max_queries - len(matches) - 2)]
    return ordered[:max_queries]


# ── Skeleton Extraction ──────────────────────────────────────────────────────

def _extract_skeleton(sexpr: str) -> str:
    """Replace inner arguments with _HOLE_ to get a 1-level skeleton."""
    s = sexpr.strip()
    if not s.startswith("("):
        return s
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
            if depth == 0 and arg_start is not None:
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


def _extract_body(lambda_source: str) -> str | None:
    s = lambda_source.strip()
    m = re.match(r'\(lambda\s+\([^)]*\)\s+(.*)\)\s*$', s, re.DOTALL)
    return m.group(1) if m else s


# ── Synthetic Traces ─────────────────────────────────────────────────────────

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


# ── ARC Domain ───────────────────────────────────────────────────────────────

class ArcDomain:
    """Domain adapter for ARC-AGI grid tasks.

    Implements the Domain protocol with PyO3-based evaluation, diagnostic
    queries as scratchpad, and iterative refinement traces.
    """

    def __init__(self, arc_dir: str = "arc_data/data/training",
                 checkpoint: str | None = None,
                 n_synthetic: int = 2000,
                 n_color_perm: int = 25):
        self._tok2id = _TOK2ID
        self._id2tok = _ID2TOK

        # Token IDs
        self.vocab_size = len(ARC_ITER_VOCAB)
        self.pad_id = _TOK2ID["<pad>"]
        self.bos_id = _TOK2ID["<bos>"]
        self.eos_id = _TOK2ID["<eos>"]
        self.sep_task_id = _TOK2ID["SEP_TASK"]
        self.sep_expr_id = _TOK2ID["SEP_EXPR"]
        self.sep_feedback_id = _TOK2ID["SEP_FEEDBACK"]
        self.sep_edit_id = _TOK2ID["SEP_EDIT"]
        self.query_open_id = _TOK2ID["<query>"]
        self.query_close_id = _TOK2ID["</query>"]
        self.qout_open_id = _TOK2ID["<q_out>"]
        self.qout_close_id = _TOK2ID["</q_out>"]
        self.hole_id = _TOK2ID["_HOLE_"]
        self.noop_id = _TOK2ID["NO_OP"]
        self.feedback_token_ids = {
            "CORRECT": _TOK2ID["CORRECT"],
            "WRONG": _TOK2ID["WRONG"],
            "ERROR": _TOK2ID["ERROR"],
            "INCOMPLETE": _TOK2ID["INCOMPLETE"],
        }

        self._arc_dir = arc_dir
        self._checkpoint = checkpoint
        self._n_synthetic = n_synthetic
        self._n_color_perm = n_color_perm
        self._tasks: list[ArcTask] | None = None
        self._solutions: dict[str, str] | None = None

    @property
    def tasks(self) -> list[ArcTask]:
        if self._tasks is None:
            self._tasks = load_arc_tasks(self._arc_dir)
        return self._tasks

    @property
    def solutions(self) -> dict[str, str]:
        """Load solutions from SELPH checkpoint file."""
        if self._solutions is None:
            self._solutions = {}
            if self._checkpoint and Path(self._checkpoint).exists():
                self._solutions = _load_checkpoint_solutions(self._checkpoint)
        return self._solutions

    # ── Tokenization ─────────────────────────────────────────────────

    def tokenize_program(self, sexpr: str) -> list[int]:
        if sexpr == "_HOLE_":
            return [self.hole_id]
        if sexpr == "NO_OP":
            return [self.noop_id]
        tokens = _TOKEN_RE.findall(sexpr)
        ids = []
        for t in tokens:
            if t in self._tok2id:
                ids.append(self._tok2id[t])
            elif t.isdigit() and f"c{t}" in self._tok2id:
                ids.append(self._tok2id[f"c{t}"])
            else:
                ids.append(self.pad_id)
        return ids

    def detokenize_program(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id,
                self.sep_task_id, self.sep_expr_id,
                self.sep_feedback_id, self.sep_edit_id}
        tokens = []
        for i in ids:
            if i in skip:
                continue
            if i == self.hole_id:
                tokens.append("_HOLE_")
                continue
            if i == self.noop_id:
                return "NO_OP"
            tok = self._id2tok.get(i, "?")
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

    # ── Task Encoding ────────────────────────────────────────────────

    def encode_task_prefix(self, task: ArcTask) -> list[int]:
        features = extract_grid_features(task)
        return [self._tok2id[f"f{min(f, 9)}"] for f in features]

    # ── Evaluation ───────────────────────────────────────────────────

    def evaluate(self, program: str, task: ArcTask,
                 include_test: bool = True) -> tuple[str, float]:
        if "_HOLE_" in program:
            return ("INCOMPLETE", 0.0)
        try:
            env = Env()
            full_prog = (f"(lambda (x) {program})"
                         if not program.strip().startswith("(lambda") else program)
            env.eval_expr(f"(define __fn__ {full_prog})")

            total_acc = 0.0
            for inp_grid, out_grid in task.train_pairs:
                inp_sexpr = grid_to_sexpr(inp_grid)
                try:
                    result = env.eval_expr(f"(__fn__ (list {inp_sexpr}))")
                except BaseException:
                    return ("ERROR", 0.0)
                total_acc += grid_cell_accuracy(result, out_grid)
            avg_acc = total_acc / len(task.train_pairs)

            if include_test and task.test_pairs:
                test_total, test_count = 0.0, 0
                for test_inp, test_out in task.test_pairs:
                    if test_out is None:
                        continue
                    test_count += 1
                    try:
                        result = env.eval_expr(f"(__fn__ (list {grid_to_sexpr(test_inp)}))")
                    except BaseException:
                        continue
                    test_total += grid_cell_accuracy(result, test_out)
                if test_count > 0:
                    avg_acc = min(avg_acc, test_total / test_count)

            return ("CORRECT", 1.0) if avg_acc >= 1.0 else ("WRONG", avg_acc)
        except BaseException:
            return ("ERROR", 0.0)

    def shaped_reward(self, feedback_type: str, accuracy: float,
                      program: str = "", task: Any = None) -> float:
        if feedback_type == "CORRECT" or accuracy >= 1.0:
            return 5.0
        return accuracy ** 2

    # ── Query Handling ───────────────────────────────────────────────

    def handle_query(self, query_ids: list[int], task: ArcTask) -> list[int]:
        query_expr = self.detokenize_program(query_ids)
        result_str = self._eval_query_on_task(query_expr, task)
        if result_str in self._tok2id:
            return [self._tok2id[result_str]]
        # Tokenize as program
        return self.tokenize_program(result_str)

    def _eval_query_on_task(self, query_expr: str, task: ArcTask) -> str:
        if not task.train_pairs:
            return "false"
        inp_grid, out_grid = task.train_pairs[0]
        resolved = query_expr.replace("(nth x 0)", grid_to_sexpr(inp_grid))
        try:
            env = Env()
            result = env.eval_expr(resolved)
            if isinstance(result, list) and result and isinstance(result[0], list):
                acc = grid_cell_accuracy(result, out_grid)
                return "true" if acc >= 1.0 else "false"
            return str(result)
        except BaseException:
            return "false"

    # ── Accuracy Tokens ──────────────────────────────────────────────

    def accuracy_tokens(self, accuracy: float) -> list[int]:
        bucket = min(10, max(0, round(accuracy * 10)))
        tok = f"acc{bucket}"
        return [self._tok2id[tok]] if tok in self._tok2id else []

    # ── Trace Generation ────────────────────────────���────────────────

    def generate_traces(self) -> list[RefinementTrace]:
        """Generate all training traces: solved, wrong-start, synthetic, augmented."""
        traces = []
        solutions = self.solutions
        all_bodies = [_extract_body(s) for s in solutions.values() if _extract_body(s)]
        task_map = {t.task_id: t for t in self.tasks}

        # Solved task traces + wrong-start
        for task_id, source in solutions.items():
            if task_id not in task_map:
                continue
            task = task_map[task_id]
            body = _extract_body(source)
            if not body:
                continue
            traces.append(self._topdown_trace(task, body))
            wrong_progs = [b for b in all_bodies if b != body]
            random.shuffle(wrong_progs)
            wt = self._wrong_start_trace(task, body, wrong_progs[:5])
            if wt:
                traces.append(wt)

        # Synthetic traces
        if self._n_synthetic > 0:
            traces.extend(self._synthetic_traces(self._n_synthetic))

        # Color augmentation
        if self._n_color_perm > 0:
            traces.extend(self._augmented_traces(solutions, self._n_color_perm))

        return traces

    def _topdown_trace(self, task: ArcTask, solution: str) -> RefinementTrace:
        diag = run_diagnostic_queries(task)
        skeleton = _extract_skeleton(solution)
        steps = [RefinementStep(
            current_expr="_HOLE_", feedback_type="INCOMPLETE", feedback_acc=0.0,
            target_expr=skeleton if skeleton != solution else solution,
            queries=diag,
        )]
        if skeleton != solution:
            fb_type, fb_acc = self.evaluate(skeleton, task, include_test=False)
            steps.append(RefinementStep(skeleton, fb_type, fb_acc, solution))
        fb_type, fb_acc = self.evaluate(solution, task, include_test=False)
        steps.append(RefinementStep(solution, fb_type, fb_acc, "NO_OP"))
        return RefinementTrace(task=task, steps=steps)

    def _wrong_start_trace(self, task, solution, wrong_programs):
        for wrong in wrong_programs:
            fb_type, fb_acc = self.evaluate(wrong, task, include_test=False)
            if fb_type == "WRONG" and fb_acc > 0.1:
                steps = [
                    RefinementStep(wrong, "WRONG", fb_acc, solution),
                    RefinementStep(solution, "CORRECT", 1.0, "NO_OP"),
                ]
                return RefinementTrace(task=task, steps=steps)
        return None

    def _synthetic_traces(self, n: int) -> list[RefinementTrace]:
        traces = []
        for _ in range(n):
            transform_name, body = random.choice(_SYNTH_TRANSFORMS)
            h, w = random.randint(3, 8), random.randint(3, 8)
            n_colors = random.randint(2, 6)
            pairs = []
            ok = True
            for _ in range(3):
                inp = [[random.choice(range(n_colors)) for _ in range(w)] for _ in range(h)]
                try:
                    env = Env()
                    out_result = env.eval_expr(f"({transform_name} {grid_to_sexpr(inp)})")
                    if isinstance(out_result, list) and out_result and isinstance(out_result[0], list):
                        out = [[int(c) for c in row] for row in out_result]
                        pairs.append((inp, out))
                    else:
                        ok = False; break
                except BaseException:
                    ok = False; break
            if not ok or len(pairs) < 3:
                continue
            task = ArcTask(f"synth_{transform_name}_{len(traces)}", pairs, [])
            diag = run_diagnostic_queries(task)
            skeleton = _extract_skeleton(body)
            steps = [RefinementStep("_HOLE_", "INCOMPLETE", 0.0,
                                    skeleton if skeleton != body else body, queries=diag)]
            if skeleton != body:
                fb, acc = self.evaluate(skeleton, task, include_test=False)
                steps.append(RefinementStep(skeleton, fb, acc, body))
            fb, acc = self.evaluate(body, task, include_test=False)
            steps.append(RefinementStep(body, fb, acc, "NO_OP"))
            traces.append(RefinementTrace(task=task, steps=steps))

            # Wrong-start variant
            others = [b for _, b in _SYNTH_TRANSFORMS if b != body]
            random.shuffle(others)
            for wrong_body in others[:2]:
                fb, acc = self.evaluate(wrong_body, task, include_test=False)
                if fb == "WRONG" and 0.05 < acc < 0.95:
                    wrong_diag = run_diagnostic_queries(task)
                    steps2 = [
                        RefinementStep(wrong_body, "WRONG", acc, body, queries=wrong_diag),
                        RefinementStep(body, "CORRECT", 1.0, "NO_OP"),
                    ]
                    traces.append(RefinementTrace(task=task, steps=steps2))
                    break
        return traces

    def _augmented_traces(self, solutions, n_color_perm):
        """Color-permuted augmentation. Simplified version."""
        traces = []
        task_map = {t.task_id: t for t in self.tasks}
        for task_id, source in solutions.items():
            if task_id not in task_map:
                continue
            task = task_map[task_id]
            body = _extract_body(source)
            if not body:
                continue
            for _ in range(n_color_perm):
                perm = list(range(10))
                random.shuffle(perm)
                aug_pairs = []
                for inp, out in task.train_pairs:
                    aug_inp = [[perm[c] for c in row] for row in inp]
                    aug_out = [[perm[c] for c in row] for row in out]
                    aug_pairs.append((aug_inp, aug_out))
                aug_task = ArcTask(f"{task_id}_cp{_}", aug_pairs, [])
                # Remap colors in body
                aug_body = body
                for old_c in range(10):
                    if str(old_c) in aug_body and perm[old_c] != old_c:
                        aug_body = aug_body.replace(f" {old_c})", f" __c{perm[old_c]}__)")
                for c in range(10):
                    aug_body = aug_body.replace(f"__c{c}__", str(c))

                skeleton = _extract_skeleton(aug_body)
                diag = run_diagnostic_queries(aug_task)
                steps = [RefinementStep("_HOLE_", "INCOMPLETE", 0.0,
                                        skeleton if skeleton != aug_body else aug_body,
                                        queries=diag)]
                if skeleton != aug_body:
                    fb, acc = self.evaluate(skeleton, aug_task, include_test=False)
                    steps.append(RefinementStep(skeleton, fb, acc, aug_body))
                fb, acc = self.evaluate(aug_body, aug_task, include_test=False)
                steps.append(RefinementStep(aug_body, fb, acc, "NO_OP"))
                traces.append(RefinementTrace(task=aug_task, steps=steps))
        return traces


def _load_checkpoint_solutions(checkpoint_path: str) -> dict[str, str]:
    """Load solved programs from a SELPH checkpoint file."""
    solutions = {}
    path = Path(checkpoint_path)
    if not path.exists():
        return solutions
    text = path.read_text()
    # Checkpoint format: lines like "task_id SOLVED (lambda (x) ...)"
    for line in text.strip().split("\n"):
        if "SOLVED" in line:
            parts = line.split(" ", 2)
            if len(parts) >= 3:
                task_id = parts[0]
                program = parts[2]
                solutions[task_id] = program
    return solutions
