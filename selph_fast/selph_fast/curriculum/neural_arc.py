"""
Neural structure synthesizer for ARC-AGI grid tasks.

Adapts the fitting-task neural_fit.py approach to discrete grid programs:
- No continuous parameters — programs are pure structure
- Reward = cell-level grid accuracy (not MSE via minimize)
- Vocabulary covers grid builtins (~100 tokens vs ~40 for fitting)
- Task encoding via compact grid features (not raw cells)

Usage:
    python -m selph_fast.curriculum.neural_arc \
        --arc-dir arc_data/data/training \
        --sft-epochs 200 --grpo-epochs 50 --verbose
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

from ..selph_fast import Env


# ── ARC Task Data ────────────────────────────────────────────────────────────

@dataclass
class ArcTask:
    task_id: str
    train_pairs: list[tuple[list[list[int]], list[list[int]]]]
    test_pairs: list[tuple[list[list[int]], list[list[int]] | None]]


def load_arc_task(json_path: str) -> ArcTask:
    """Load a single ARC task from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    task_id = Path(json_path).stem
    train = [(p["input"], p["output"]) for p in data["train"]]
    test = [(p["input"], p.get("output")) for p in data.get("test", [])]
    return ArcTask(task_id=task_id, train_pairs=train, test_pairs=test)


def load_arc_tasks(dir_path: str) -> list[ArcTask]:
    """Load all ARC tasks from a directory of JSON files."""
    tasks = []
    for p in sorted(Path(dir_path).glob("*.json")):
        tasks.append(load_arc_task(str(p)))
    return tasks


# ── Vocabulary ───────────────────────────────────────────────────────────────

ARC_VOCAB = [
    # Special tokens
    "<pad>", "<bos>", "<eos>", "<sep>",
    # Syntax
    "(", ")",
    # Core
    "lambda", "x", "nth", "list",
    # Colors 0-9 (prefixed to avoid ambiguity with feature/number tokens)
    "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9",
    # Numbers (for object indices, thresholds, translation offsets)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-1", "-2",
    # Temp color constants used in multi-swap patterns (50+color)
    "51", "55", "56", "57", "58",
    # Grid transforms
    "grid-rotate-cw", "grid-rotate-ccw", "grid-rotate-180",
    "grid-flip-h", "grid-flip-v", "grid-transpose",
    "grid-trim", "grid-compact",
    # Color operations
    "grid-replace-color", "grid-keep-color", "grid-remove-small-objects",
    # Fill operations
    "grid-fill-enclosed", "grid-fill-rectangular-holes",
    "m8r-fill-between-both",
    # Composition operations
    "grid-overlay", "grid-hconcat", "grid-vconcat", "grid-place", "grid-translate",
    # Object operations
    "grid-recompose", "grid-object", "grid-object-count", "grid-object-pos",
    "grid-set",
    # Scale/tile
    "grid-scale", "grid-tile", "grid-untile",
    # Logic operations
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
    # Higher-order / iteration
    "reduce", "acc", "idx", "idx2", "g2",
    # String literals for grid-recompose
    "\"rotate-cw\"", "\"rotate-ccw\"", "\"rotate-180\"",
    "\"flip-h\"", "\"flip-v\"", "\"transpose\"",
    # Feature encoding digits (for task prompt)
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9",
]

ARC_TOK2ID = {t: i for i, t in enumerate(ARC_VOCAB)}
ARC_ID2TOK = {i: t for i, t in enumerate(ARC_VOCAB)}
ARC_PAD_ID = ARC_TOK2ID["<pad>"]
ARC_BOS_ID = ARC_TOK2ID["<bos>"]
ARC_EOS_ID = ARC_TOK2ID["<eos>"]
ARC_SEP_ID = ARC_TOK2ID["<sep>"]
ARC_VOCAB_SIZE = len(ARC_VOCAB)


# ── Tokenizer ────────────────────────────────────────────────────────────────

# Regex to split s-expressions while keeping quoted strings intact
_TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()]+')


def tokenize_arc(sexpr: str) -> list[int]:
    """Tokenize an ARC program s-expression into token IDs."""
    tokens = _TOKEN_RE.findall(sexpr)
    ids = [ARC_BOS_ID]
    for t in tokens:
        if t in ARC_TOK2ID:
            ids.append(ARC_TOK2ID[t])
        elif t.startswith('"') and t.endswith('"'):
            # Unknown string literal — map to closest known
            ids.append(ARC_TOK2ID.get(t, ARC_PAD_ID))
        else:
            # Unknown token — try as color literal cN
            color_key = f"c{t}" if t.isdigit() else None
            if color_key and color_key in ARC_TOK2ID:
                ids.append(ARC_TOK2ID[color_key])
            else:
                ids.append(ARC_PAD_ID)
    ids.append(ARC_EOS_ID)
    return ids


def detokenize_arc(ids: list[int]) -> str:
    """Convert token IDs back to an s-expression string."""
    tokens = []
    for i in ids:
        if i in (ARC_PAD_ID, ARC_BOS_ID, ARC_EOS_ID, ARC_SEP_ID):
            continue
        tok = ARC_ID2TOK.get(i, "?")
        # Convert color tokens back to bare numbers
        if tok.startswith("c") and len(tok) == 2 and tok[1].isdigit():
            tok = tok[1]
        tokens.append(tok)
    # Reconstruct with proper spacing
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


# ── Grid Feature Encoder ────────────────────────────────────────────────────

def _grid_colors(grid: list[list[int]]) -> set[int]:
    return {c for row in grid for c in row}


def _grid_object_count_estimate(grid: list[list[int]], bg: int = 0) -> int:
    """Quick estimate of connected non-background regions via flood fill."""
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    count = 0
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                # BFS flood fill
                count += 1
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if visited[cr][cc] or grid[cr][cc] == bg:
                        continue
                    visited[cr][cc] = True
                    stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])
    return count


def _has_h_symmetry(grid: list[list[int]]) -> bool:
    h = len(grid)
    for r in range(h // 2):
        if grid[r] != grid[h - 1 - r]:
            return False
    return True


def _has_v_symmetry(grid: list[list[int]]) -> bool:
    for row in grid:
        w = len(row)
        for c in range(w // 2):
            if row[c] != row[w - 1 - c]:
                return False
    return True


def extract_grid_features(task: ArcTask) -> list[int]:
    """Extract compact feature vector from an ARC task.

    Per training pair (up to 4 pairs, 8 features each):
      input_h, input_w, output_h, output_w,
      n_colors_in, n_colors_out,
      size_change (0=same, 1=grow, 2=shrink),
      object_count_bucket (clamped 0-9)

    Global features (4):
      dims_preserved, colors_preserved, has_symmetry, area_bucket

    All values clamped to 0-9 for fN token encoding.
    Returns list of ints in [0, 9].
    """
    features = []
    dims_preserved = True
    colors_preserved = True

    for inp, out in task.train_pairs[:4]:
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        in_colors = _grid_colors(inp)
        out_colors = _grid_colors(out)
        obj_count = _grid_object_count_estimate(inp)

        if ih != oh or iw != ow:
            dims_preserved = False
        if in_colors != out_colors:
            colors_preserved = False

        size_change = 0 if ih * iw == oh * ow else (1 if oh * ow > ih * iw else 2)

        features.extend([
            min(ih, 9), min(iw, 9), min(oh, 9), min(ow, 9),
            min(len(in_colors), 9), min(len(out_colors), 9),
            size_change,
            min(obj_count, 9),
        ])

    # Pad if fewer than 4 pairs
    while len(features) < 32:
        features.append(0)

    # Global features
    has_sym = any(
        _has_h_symmetry(inp) or _has_v_symmetry(inp)
        for inp, _ in task.train_pairs[:4]
    )
    max_area = max(
        (len(inp) * (len(inp[0]) if inp else 0)
         for inp, _ in task.train_pairs[:4]),
        default=0,
    )
    area_bucket = min(max_area // 100, 9)

    features.extend([
        1 if dims_preserved else 0,
        1 if colors_preserved else 0,
        1 if has_sym else 0,
        area_bucket,
    ])

    return features  # length 36


def encode_arc_task(task: ArcTask) -> list[int]:
    """Encode an ARC task as a token sequence for the model prompt."""
    features = extract_grid_features(task)
    ids = [ARC_BOS_ID]
    for f in features:
        ids.append(ARC_TOK2ID[f"f{min(f, 9)}"])
    ids.append(ARC_SEP_ID)
    return ids  # length ~38


# ── Model ────────────────────────────────────────────────────────────────────

class ArcStructureGenerator(nn.Module):
    """Transformer LM generating ARC program s-expressions."""

    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                 max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(ARC_VOCAB_SIZE, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, ARC_VOCAB_SIZE)
        self.d_model = d_model
        self.max_len = max_len

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape
        pos = mx.arange(T)
        h = self.embed(x) + self.pos_embed(pos)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        return self.head(h)


# ── Generation ───────────────────────────────────────────────────────────────

def generate(model: ArcStructureGenerator, prompt_ids: list[int],
             max_new_tokens: int = 60, temperature: float = 0.8
             ) -> tuple[list[int], list[float]]:
    """Autoregressive generation with log-prob tracking for GRPO."""
    model.eval()
    ids = list(prompt_ids)
    log_probs = []

    for _ in range(max_new_tokens):
        if len(ids) >= model.max_len:
            break
        x = mx.array([ids])
        logits = model(x)[0, -1, :] / temperature
        probs = mx.softmax(logits, axis=-1)
        token = mx.random.categorical(logits[None, :])[0].item()
        lp = mx.log(probs[token] + 1e-10).item()
        log_probs.append(lp)
        ids.append(token)
        if token == ARC_EOS_ID:
            break

    return ids[len(prompt_ids):], log_probs


# ── Grid Evaluation Helpers ──────────────────────────────────────────────────

def grid_to_sexpr(grid: list[list[int]]) -> str:
    """Convert a Python grid to a SELPH nested-list expression."""
    rows = " ".join(
        "(list " + " ".join(str(c) for c in row) + ")"
        for row in grid
    )
    return f"(list {rows})"


def grid_cell_accuracy(pred: object, expected: list[list[int]]) -> float:
    """Compare predicted grid (from PyO3) to expected grid. Returns [0, 1]."""
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
        # Dimension mismatch — partial credit by min overlap
        min_len = min(len(pred_flat), len(exp_flat))
        if min_len == 0:
            return 0.0
        matches = sum(1 for a, b in zip(pred_flat[:min_len], exp_flat[:min_len]) if a == b)
        # Penalize dimension mismatch
        return matches / max(len(pred_flat), len(exp_flat)) * 0.5
    matches = sum(1 for a, b in zip(pred_flat, exp_flat) if a == b)
    return matches / len(exp_flat)


# ── Reward Function ──────────────────────────────────────────────────────────

def compute_arc_reward(program: str, task: ArcTask) -> float:
    """Evaluate a generated program on ARC training pairs.

    Returns reward in [0, 1]:
      1.0 = all training pairs produce exact grid match
      0.0 = parse error or total failure
      Intermediate = average cell-level accuracy across pairs
    """
    try:
        env = Env()
        # Define the function — wrap bare body in lambda if needed
        if not program.strip().startswith("(lambda"):
            program = f"(lambda (x) {program})"
        env.eval_expr(f"(define __fn__ {program})")

        total_acc = 0.0
        for inp_grid, out_grid in task.train_pairs:
            inp_sexpr = grid_to_sexpr(inp_grid)
            # ARC tasks pass input as (list grid) — single-arg via nth
            try:
                result = env.eval_expr(f"(__fn__ (list {inp_sexpr}))")
            except BaseException:
                # Catch Rust panics (pyo3_runtime.PanicException) + all errors
                continue
            acc = grid_cell_accuracy(result, out_grid)
            total_acc += acc

        avg_acc = total_acc / len(task.train_pairs)

        # Bonus for test pair correctness (if output is available)
        if task.test_pairs and task.test_pairs[0][1] is not None:
            test_inp, test_out = task.test_pairs[0]
            test_sexpr = grid_to_sexpr(test_inp)
            try:
                test_result = env.eval_expr(f"(__fn__ (list {test_sexpr}))")
            except BaseException:
                return avg_acc
            test_acc = grid_cell_accuracy(test_result, test_out)
            return 0.7 * avg_acc + 0.3 * test_acc

        return avg_acc
    except BaseException:
        # Catch everything including Rust panics
        return 0.0


# ── SFT Data ─────────────────────────────────────────────────────────────────

def load_sft_from_checkpoint(checkpoint_path: str) -> dict[str, str]:
    """Parse a SELPH checkpoint file into {task_id: lambda_source}."""
    solutions = {}
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                task_id, _strategy, _candidates, source = parts[0], parts[1], parts[2], parts[3]
                solutions[task_id] = source
    return solutions


def _extract_body(lambda_source: str) -> str | None:
    """Extract the body from (lambda (x) <body>)."""
    s = lambda_source.strip()
    if not s.startswith("(lambda"):
        return s
    # Find the body after (lambda (x) ...)
    # Skip "(lambda" then whitespace, then "(x)" or "(x y ...)", then whitespace
    m = re.match(r'\(lambda\s+\([^)]*\)\s+(.*)\)\s*$', s, re.DOTALL)
    if m:
        return m.group(1)
    return None


def augment_color_permutation(
    task: ArcTask, program: str, n: int = 10
) -> list[tuple[ArcTask, str]]:
    """Generate color-permuted variants of a (task, program) pair.

    Randomly permutes colors 1-9 (keeping 0 as background) in both
    the grids and the program's color constants.
    """
    results = []
    for _ in range(n):
        perm = list(range(10))
        non_bg = list(range(1, 10))
        random.shuffle(non_bg)
        for i, v in enumerate(non_bg):
            perm[i + 1] = v

        def remap_grid(grid: list[list[int]]) -> list[list[int]]:
            return [[perm[c] if c < 10 else c for c in row] for row in grid]

        new_train = [(remap_grid(inp), remap_grid(out)) for inp, out in task.train_pairs]
        new_test = [
            (remap_grid(inp), remap_grid(out) if out else None)
            for inp, out in task.test_pairs
        ]
        new_task = ArcTask(task_id=task.task_id, train_pairs=new_train, test_pairs=new_test)

        # Remap color constants in the program: replace bare digits in
        # color-argument positions. This is approximate but catches the
        # common patterns like (grid-replace-color ... 3 5)
        new_prog = program
        for old_c in range(9, 0, -1):  # reverse to avoid double-replace
            new_prog = re.sub(
                rf'\b{old_c}\b',
                f'__C{perm[old_c]}__',
                new_prog,
            )
        for c in range(1, 10):
            new_prog = new_prog.replace(f'__C{c}__', str(c))

        results.append((new_task, new_prog))
    return results


# Synthetic grid transforms — pair (transform_name, program_body, generator)
_SYNTHETIC_TRANSFORMS: list[tuple[str, str, callable]] = []


def _register_synthetic():
    """Register synthetic task generators for data augmentation."""
    transforms = [
        ("rotate-cw", "(grid-rotate-cw (nth x 0))"),
        ("rotate-ccw", "(grid-rotate-ccw (nth x 0))"),
        ("rotate-180", "(grid-rotate-180 (nth x 0))"),
        ("flip-h", "(grid-flip-h (nth x 0))"),
        ("flip-v", "(grid-flip-v (nth x 0))"),
        ("transpose", "(grid-transpose (nth x 0))"),
        ("fill-enclosed", "(grid-fill-enclosed (nth x 0))"),
        ("trim", "(grid-trim (nth x 0))"),
    ]
    for name, body in transforms:
        _SYNTHETIC_TRANSFORMS.append((name, body))


_register_synthetic()


def _random_grid(h: int, w: int, n_colors: int = 4) -> list[list[int]]:
    """Generate a random grid with given dimensions and color count."""
    colors = list(range(n_colors))
    return [[random.choice(colors) for _ in range(w)] for _ in range(h)]


def _apply_transform(grid: list[list[int]], transform: str) -> list[list[int]] | None:
    """Apply a named transform to a grid using the SELPH evaluator."""
    try:
        env = Env()
        inp_sexpr = grid_to_sexpr(grid)
        prog = f"(lambda (x) ({transform} (nth x 0)))"
        env.eval_expr(f"(define __fn__ {prog})")
        result = env.eval_expr(f"(__fn__ (list {inp_sexpr}))")
        if isinstance(result, list) and result and isinstance(result[0], list):
            return [[int(c) for c in row] for row in result]
    except BaseException:
        pass
    return None


def generate_synthetic_pairs(n: int = 200) -> list[tuple[ArcTask, str]]:
    """Generate synthetic (task, program) pairs from known transforms."""
    pairs = []
    for _ in range(n):
        name, body = random.choice(_SYNTHETIC_TRANSFORMS)
        h = random.randint(3, 10)
        w = random.randint(3, 10)
        n_colors = random.randint(2, 6)

        # Generate 3 training pairs + 1 test pair
        train_pairs = []
        test_pairs = []
        ok = True
        for i in range(4):
            inp = _random_grid(h, w, n_colors)
            transform_name = body.split("(")[1].split()[0] if "(" in body else body
            out = _apply_transform(inp, transform_name)
            if out is None:
                ok = False
                break
            if i < 3:
                train_pairs.append((inp, out))
            else:
                test_pairs.append((inp, out))

        if ok and len(train_pairs) == 3:
            task = ArcTask(
                task_id=f"synthetic_{name}_{len(pairs)}",
                train_pairs=train_pairs,
                test_pairs=test_pairs,
            )
            pairs.append((task, body))

    return pairs


def build_arc_sft_data(
    arc_tasks: list[ArcTask],
    checkpoint_path: str | None = None,
    n_augment: int = 10,
    n_synthetic: int = 200,
) -> list[tuple[ArcTask, str]]:
    """Build the full SFT training dataset.

    Sources:
    1. Solved ARC tasks from checkpoint (real solutions)
    2. Color-permuted augmentations of solved tasks
    3. Synthetic tasks from known transforms
    """
    pairs = []
    task_map = {t.task_id: t for t in arc_tasks}

    # Source 1 + 2: checkpoint solutions + augmentations
    if checkpoint_path and os.path.exists(checkpoint_path):
        solutions = load_sft_from_checkpoint(checkpoint_path)
        for task_id, source in solutions.items():
            if task_id in task_map:
                body = _extract_body(source)
                if body:
                    task = task_map[task_id]
                    pairs.append((task, body))
                    pairs.extend(augment_color_permutation(task, body, n_augment))

    # Source 3: synthetic
    pairs.extend(generate_synthetic_pairs(n_synthetic))

    return pairs


# ── SFT Warmstart ────────────────────────────────────────────────────────────

def sft_warmstart(
    model: ArcStructureGenerator,
    pairs: list[tuple[ArcTask, str]],
    epochs: int = 200,
    lr: float = 3e-3,
    verbose: bool = True,
) -> float:
    """Supervised fine-tuning on (task, program) pairs."""
    if not pairs:
        if verbose:
            print("SFT: no training pairs, skipping")
        return 0.0

    optimizer = optim.Adam(learning_rate=lr)

    if verbose:
        print(f"SFT warmstart: {len(pairs)} pairs, {epochs} epochs, lr={lr}")

    # Pre-tokenize
    tokenized = []
    for task, body in pairs:
        prompt_ids = encode_arc_task(task)
        target_ids = tokenize_arc(body)
        full_ids = prompt_ids + target_ids[1:]  # skip target BOS
        if len(full_ids) < model.max_len:
            tokenized.append((full_ids, len(prompt_ids)))

    if not tokenized:
        if verbose:
            print("SFT: all pairs exceed max_len, skipping")
        return 0.0

    if verbose:
        print(f"SFT: {len(tokenized)} pairs after length filter")

    best_loss = float("inf")
    for epoch in range(epochs):
        random.shuffle(tokenized)
        epoch_loss = 0.0
        n_batches = 0

        for full_ids, prompt_len in tokenized:
            def loss_fn(m):
                x = mx.array([full_ids[:-1]])
                logits = m(x)[0]
                target_logits = logits[prompt_len - 1:]
                target_ids_arr = mx.array(full_ids[prompt_len:])
                L = min(target_logits.shape[0], target_ids_arr.shape[0])
                if L == 0:
                    return mx.array(0.0)
                return nn.losses.cross_entropy(
                    target_logits[:L], target_ids_arr[:L], reduction="mean"
                )

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            loss_val = float(loss.item())
            if not math.isnan(loss_val):
                epoch_loss += loss_val
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        best_loss = min(best_loss, avg_loss)

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            # Sample generation
            sample_task, sample_expected = pairs[0]
            prompt = encode_arc_task(sample_task)
            gen_ids, _ = generate(model, prompt, temperature=0.3)
            generated = detokenize_arc(gen_ids)
            reward = compute_arc_reward(generated, sample_task)
            print(f"  SFT epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")
            print(f"    {sample_task.task_id}: {generated[:80]}")
            print(f"    expected:   {sample_expected[:80]}")
            print(f"    reward:     {reward:.3f}")

    if verbose:
        print(f"  SFT complete: best_loss={best_loss:.4f}")
        # Test accuracy on a sample of pairs
        n_ok = 0
        sample = pairs[:min(20, len(pairs))]
        for task, expected in sample:
            prompt = encode_arc_task(task)
            gen_ids, _ = generate(model, prompt, temperature=0.3)
            generated = detokenize_arc(gen_ids)
            r = compute_arc_reward(generated, task)
            if r >= 1.0:
                n_ok += 1
        print(f"  SFT accuracy (sample): {n_ok}/{len(sample)}")
        print()

    return best_loss


# ── GRPO Training ────────────────────────────────────────────────────────────

@dataclass
class TrajectoryData:
    gen_ids: list[int]
    log_probs: list[float]
    reward: float
    sexpr: str


def train_arc_generator(
    arc_tasks: list[ArcTask],
    sft_pairs: list[tuple[ArcTask, str]],
    sft_epochs: int = 200,
    grpo_epochs: int = 50,
    group_size: int = 8,
    temperature: float = 1.0,
    temp_decay: float = 0.98,
    clip_eps: float = 0.2,
    kl_coef: float = 0.05,
    sft_lr: float = 3e-3,
    grpo_lr: float = 1e-4,
    verbose: bool = True,
) -> ArcStructureGenerator:
    """Train a neural model to generate ARC program structures.

    Phase 1: SFT warmstart on known solutions + synthetic data.
    Phase 2: GRPO refinement using cell-level accuracy as reward.
    """
    model = ArcStructureGenerator()

    if verbose:
        n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
        print(f"ARC structure generator: {n_params:,} params")
        print(f"  vocab_size={ARC_VOCAB_SIZE}, sft_pairs={len(sft_pairs)}, arc_tasks={len(arc_tasks)}")
        print()

    # Phase 1: SFT
    if sft_epochs > 0 and sft_pairs:
        sft_warmstart(model, sft_pairs, epochs=sft_epochs, lr=sft_lr, verbose=verbose)

    if grpo_epochs == 0:
        return model

    # Snapshot for KL reference
    ref_model = ArcStructureGenerator()
    ref_params = [(k, v) for k, v in tree_flatten(model.parameters())]
    ref_model.load_weights(ref_params)
    ref_model.freeze()

    optimizer = optim.Adam(learning_rate=grpo_lr)

    if verbose:
        print(f"GRPO phase: {grpo_epochs} epochs, group_size={group_size}")
        print()

    # Track best solutions found
    best_solutions: dict[str, tuple[float, str]] = {}

    for epoch in range(grpo_epochs):
        random.shuffle(arc_tasks)
        epoch_reward = 0.0
        epoch_success = 0
        epoch_total = 0
        temp = max(0.6, temperature * (temp_decay ** epoch))

        # Sample a batch of tasks per epoch
        batch_tasks = arc_tasks[:min(64, len(arc_tasks))]

        for task in batch_tasks:
            prompt_ids = encode_arc_task(task)

            trajectories: list[TrajectoryData] = []
            for _ in range(group_size):
                gen_ids, log_probs = generate(model, prompt_ids, temperature=temp)
                sexpr = detokenize_arc(gen_ids)
                reward = compute_arc_reward(sexpr, task)

                trajectories.append(TrajectoryData(
                    gen_ids=gen_ids, log_probs=log_probs,
                    reward=reward, sexpr=sexpr,
                ))
                epoch_total += 1
                if reward >= 1.0:
                    epoch_success += 1
                epoch_reward += reward

                # Track best solution per task
                if reward > best_solutions.get(task.task_id, (0.0, ""))[0]:
                    best_solutions[task.task_id] = (reward, sexpr)

            # Group-relative advantages
            rewards = [t.reward for t in trajectories]
            mean_r = sum(rewards) / len(rewards)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
            if std_r < 1e-8:
                continue

            advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

            def loss_fn(m):
                total_loss = mx.array(0.0)
                n_tokens = 0

                for traj, adv in zip(trajectories, advantages):
                    if abs(adv) < 1e-8 or len(traj.gen_ids) == 0:
                        continue

                    full_ids = prompt_ids + traj.gen_ids
                    if len(full_ids) > model.max_len:
                        continue
                    x = mx.array([full_ids[:-1]])
                    logits = m(x)[0]

                    gen_start = len(prompt_ids)
                    for i, (token_id, old_lp) in enumerate(zip(traj.gen_ids, traj.log_probs)):
                        pos = gen_start + i
                        if pos >= logits.shape[0]:
                            break
                        new_lp = mx.log(mx.softmax(logits[pos]) + 1e-10)[token_id]

                        ref_x = mx.array([full_ids[:pos + 1]])
                        ref_logits = ref_model(ref_x)[0]
                        ref_lp = mx.log(mx.softmax(ref_logits[-1]) + 1e-10)[token_id]

                        ratio = mx.exp(new_lp - old_lp)
                        clipped = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                        pg_loss = -mx.minimum(ratio * adv, clipped * adv)
                        kl = new_lp - ref_lp

                        total_loss = total_loss + pg_loss + kl_coef * kl
                        n_tokens += 1

                return total_loss / max(n_tokens, 1)

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

        avg_reward = epoch_reward / max(epoch_total, 1)
        success_rate = epoch_success / max(epoch_total, 1)
        n_with_reward = sum(1 for r, _ in best_solutions.values() if r > 0)
        n_perfect = sum(1 for r, _ in best_solutions.values() if r >= 1.0)

        if verbose:
            print(f"GRPO {epoch + 1}/{grpo_epochs}: reward={avg_reward:.3f} "
                  f"success={success_rate:.1%} temp={temp:.2f}")
            print(f"  best_solutions: {n_perfect} perfect, {n_with_reward} with reward>0")

            if (epoch + 1) % 10 == 0:
                # Show top solutions
                top = sorted(best_solutions.items(), key=lambda x: -x[1][0])[:5]
                for tid, (r, prog) in top:
                    print(f"    {tid}: r={r:.3f} {prog[:60]}")
                print()

    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train neural ARC structure generator")
    parser.add_argument("--arc-dir", default="arc_data/data/training",
                        help="Path to ARC task JSON directory")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SELPH checkpoint with solved ARC programs")
    parser.add_argument("--sft-epochs", type=int, default=200)
    parser.add_argument("--grpo-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--n-synthetic", type=int, default=200,
                        help="Number of synthetic training pairs")
    parser.add_argument("--n-augment", type=int, default=10,
                        help="Color permutation augmentations per solved task")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-weights", default=None,
                        help="Path to save model weights after training")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Load ARC tasks
    if verbose:
        print(f"Loading ARC tasks from {args.arc_dir}...")
    arc_tasks = load_arc_tasks(args.arc_dir)
    if verbose:
        print(f"  Loaded {len(arc_tasks)} tasks")

    # Build SFT data
    if verbose:
        print("Building SFT training data...")
    sft_pairs = build_arc_sft_data(
        arc_tasks,
        checkpoint_path=args.checkpoint,
        n_augment=args.n_augment,
        n_synthetic=args.n_synthetic,
    )
    if verbose:
        n_real = len([p for p in sft_pairs if not p[0].task_id.startswith("synthetic")])
        n_synth = len(sft_pairs) - n_real
        print(f"  {len(sft_pairs)} total pairs ({n_real} real + augmented, {n_synth} synthetic)")
        print()

    # Train
    t0 = time.time()
    model = train_arc_generator(
        arc_tasks=arc_tasks,
        sft_pairs=sft_pairs,
        sft_epochs=args.sft_epochs,
        grpo_epochs=args.grpo_epochs,
        group_size=args.group_size,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")

    # Final evaluation: sweep all 400 tasks
    if verbose:
        print("\n── Final evaluation on all tasks ──")
        perfect = 0
        near_miss = 0
        with_reward = 0
        for task in tasks:
            prompt = encode_arc_task(task)
            # Try multiple temperatures
            best_r = 0.0
            best_prog = ""
            for temp in [0.3, 0.6, 0.9]:
                for _ in range(3):
                    gen_ids, _ = generate(model, prompt, temperature=temp)
                    prog = detokenize_arc(gen_ids)
                    r = compute_arc_reward(prog, task)
                    if r > best_r:
                        best_r = r
                        best_prog = prog
            if best_r >= 1.0:
                perfect += 1
                print(f"  SOLVED {task.task_id}: {best_prog[:70]}")
            elif best_r >= 0.9:
                near_miss += 1
            if best_r > 0:
                with_reward += 1
        print(f"\nFinal: {perfect}/400 perfect, {near_miss} near-miss (>=90%), "
              f"{with_reward}/400 with reward>0")

    # Save weights if requested
    if args.save_weights:
        weights = dict(tree_flatten(model.parameters()))
        mx.savez(args.save_weights, **weights)
        if verbose:
            print(f"Saved weights to {args.save_weights}")


if __name__ == "__main__":
    main()
