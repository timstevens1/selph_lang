"""
Neural structure synthesizer for fitting tasks.

A small LM generates candidate model structures as s-expressions with
(param 0) holes. minimize() fits the params. MSE on held-out data is
the reward. GRPO trains the generator.

The key simplification vs. full neural_selph: the model only generates
structure — constants are handled by the optimizer. The vocabulary is
~40 tokens instead of ~300.

Usage:
    from selph_fast.curriculum.neural_fit import train_structure_generator
    model = train_structure_generator(epochs=20, verbose=True)
"""

from __future__ import annotations

import math
import os
import random
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

from ..selph_fast import Env, Param
from ..optimize import minimize
from .fit import Task, CURRICULUM


# ── Tokenizer ────────────────────────────────────────────────────────────────

# Minimal vocabulary for math s-expressions
VOCAB = [
    "<pad>", "<bos>", "<eos>",
    # Syntax
    "(", ")",
    # Core functions
    "lambda", "x",
    "add", "multiply", "subtract", "divide",
    "if", ">", "<", ">=", "<=",
    # Param hole — the optimizer fills these
    "param", "0",
    # Numeric building blocks for thresholds
    "1", "2", "3", "4", "5", "-1", "-2", "0.5", "10",
    # Common patterns
    "abs", "min", "max",
]

TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for i, t in enumerate(VOCAB)}
PAD_ID = TOK2ID["<pad>"]
BOS_ID = TOK2ID["<bos>"]
EOS_ID = TOK2ID["<eos>"]
VOCAB_SIZE = len(VOCAB)


def tokenize(sexpr: str) -> list[int]:
    """Tokenize an s-expression into token IDs."""
    # Simple whitespace + paren tokenizer
    s = sexpr.replace("(", " ( ").replace(")", " ) ")
    tokens = s.split()
    ids = [BOS_ID]
    for t in tokens:
        if t in TOK2ID:
            ids.append(TOK2ID[t])
        else:
            # Unknown token — try as number
            ids.append(TOK2ID.get("0", PAD_ID))
    ids.append(EOS_ID)
    return ids


def detokenize(ids: list[int]) -> str:
    """Convert token IDs back to an s-expression string."""
    tokens = []
    for i in ids:
        if i in (PAD_ID, BOS_ID, EOS_ID):
            continue
        tokens.append(ID2TOK.get(i, "?"))
    # Reconstruct with proper spacing
    result = []
    for t in tokens:
        if t == "(":
            result.append("(")
        elif t == ")" and result and not result[-1].endswith("("):
            result.append(")")
        else:
            if result and not result[-1].endswith("("):
                result.append(" ")
            result.append(t)
    return "".join(result)


# ── Prompt encoding ──────────────────────────────────────────────────────────

def encode_task(task: Task, max_points: int = 8) -> list[int]:
    """Encode a task's data points as a token sequence.

    Format: <bos> x1 y1 x2 y2 ... <sep>
    Where <sep> is just EOS (the model generates after this).
    """
    ids = [BOS_ID]
    for x, y in task.data[:max_points]:
        # Encode numbers as digit sequences
        ids.extend(_encode_number(x))
        ids.extend(_encode_number(y))
    return ids


def _encode_number(n: float) -> list[int]:
    """Encode a number using available vocab tokens."""
    # Map to nearest vocab number token
    num_tokens = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
                  10: "10", -1: "-1", -2: "-2", 0.5: "0.5"}
    if n in num_tokens:
        return [TOK2ID[num_tokens[n]]]
    # For other numbers, use closest integer
    nearest = min(num_tokens.keys(), key=lambda k: abs(k - n))
    return [TOK2ID[num_tokens[nearest]]]


# ── Model ────────────────────────────────────────────────────────────────────

class StructureGenerator(nn.Module):
    """Small transformer LM that generates s-expression model structures."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 max_len: int = 64):
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

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. x: (batch, seq_len) -> logits: (batch, seq_len, vocab)"""
        B, T = x.shape
        pos = mx.arange(T)
        h = self.embed(x) + self.pos_embed(pos)

        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        return self.head(h)


# ── Generation ───────────────────────────────────────────────────────────────

def generate(model: StructureGenerator, prompt_ids: list[int],
             max_new_tokens: int = 40, temperature: float = 0.8) -> tuple[list[int], list[float]]:
    """Autoregressive generation with log-prob tracking for GRPO."""
    model.eval()
    ids = list(prompt_ids)
    log_probs = []

    for _ in range(max_new_tokens):
        x = mx.array([ids])
        logits = model(x)[0, -1, :]  # last position
        logits = logits / temperature

        # Sample
        probs = mx.softmax(logits, axis=-1)
        token = mx.random.categorical(logits[None, :])[0].item()

        # Track log-prob
        lp = mx.log(probs[token] + 1e-10).item()
        log_probs.append(lp)
        ids.append(token)

        if token == EOS_ID:
            break

    return ids[len(prompt_ids):], log_probs


# ── Reward ───────────────────────────────────────────────────────────────────

def compute_reward(sexpr: str, task: Task, optimize_steps: int = 300) -> float:
    """Evaluate a generated s-expression on a fitting task.

    Returns reward in [0, 1]:
      1.0 = perfect fit (MSE < task.threshold)
      0.0 = parse error or terrible fit
      Intermediate = proportional to log-MSE reduction
    """
    try:
        env = Env()

        # Count params in the expression
        param_count = sexpr.count("(param")
        if param_count == 0:
            # No params — can't optimize, treat as constant
            return 0.0
        if param_count > 10:
            # Too many params — penalize complexity
            return 0.0

        # Define params with unique names
        param_names = []
        modified = sexpr
        for i in range(param_count):
            name = f"_p{i}"
            param_names.append(name)
            # Replace first occurrence of (param 0) with the named param
            modified = modified.replace("(param 0)", name, 1)
            env.define(name, Param(0.0))

        # Define data
        data_str = " ".join(f"(list {x} {y})" for x, y in task.data)
        env.eval_expr(f"(define __data__ (dataset (list {data_str})))")

        # Define model
        env.eval_expr(f"(define __model__ (lambda (x) {modified}))")

        # Optimize
        result = minimize(
            env, param_names, "(model-loss __model__ __data__ mse)",
            method="nelder-mead", steps=optimize_steps,
        )

        # Compute test loss if available
        if task.test:
            test_str = " ".join(f"(list {x} {y})" for x, y in task.test)
            env.eval_expr(f"(define __test__ (dataset (list {test_str})))")
            test_mse = env.eval_expr("(model-loss __model__ __test__ mse)")
            mse = test_mse
        else:
            mse = result.loss

        # Reward: logarithmic scale, capped at [0, 1]
        if mse <= task.threshold:
            return 1.0
        elif mse > 1e6:
            return 0.0
        else:
            # Smooth reward: higher for lower MSE
            # log10(threshold) is the target, log10(mse) is current
            log_ratio = math.log10(mse + 1e-10) / math.log10(task.threshold + 1e-10)
            return max(0.0, min(1.0, 1.0 / log_ratio))

    except Exception:
        return 0.0


# ── SFT data from curriculum results ─────────────────────────────────────────

# Known good structures from the hand-coded curriculum templates
SFT_EXAMPLES = [
    # (task_pattern, structure)
    ("const",    "(param 0)"),
    ("linear",   "(add (multiply (param 0) x) (param 0))"),
    ("quad",     "(add (multiply (param 0) (multiply x x)) (add (multiply (param 0) x) (param 0)))"),
    ("piecewise", "(if (> x (param 0)) (add (multiply (param 0) x) (param 0)) (add (multiply (param 0) x) (param 0)))"),
    ("identity", "(multiply (param 0) x)"),
    ("double",   "(multiply (param 0) x)"),
    ("square",   "(multiply (param 0) (multiply x x))"),
    ("abs",      "(if (> x (param 0)) (multiply (param 0) x) (multiply (param 0) x))"),
]


# ── GRPO Training Loop ──────────────────────────────────────────────────────

@dataclass
class TrajectoryData:
    """One generation trajectory for GRPO."""
    gen_ids: list[int]
    log_probs: list[float]
    reward: float
    sexpr: str


def train_structure_generator(
    epochs: int = 20,
    group_size: int = 4,
    temperature: float = 0.8,
    clip_eps: float = 0.2,
    kl_coef: float = 0.05,
    lr: float = 1e-3,
    tasks: list[Task] | None = None,
    verbose: bool = True,
) -> StructureGenerator:
    """Train a neural model to generate model structures via GRPO."""

    if tasks is None:
        # Use a subset of curriculum tasks for training
        tasks = [t for t in CURRICULUM if t.name in [
            "const_5", "identity", "double", "linear_2x+1", "linear_neg",
            "square", "quadratic_x2-2x+1", "abs_value",
        ]]

    model = StructureGenerator()
    ref_model = StructureGenerator()
    # Copy initial weights to ref
    ref_params = [(k, v) for k, v in tree_flatten(model.parameters())]
    ref_model.load_weights(ref_params)
    ref_model.freeze()

    optimizer = optim.Adam(learning_rate=lr)

    if verbose:
        print(f"Training structure generator: {sum(p.size for _, p in tree_flatten(model.parameters())):,} params")
        print(f"  vocab_size={VOCAB_SIZE}, tasks={len(tasks)}, group_size={group_size}")
        print(f"  epochs={epochs}, lr={lr}, temperature={temperature}")
        print()

    for epoch in range(epochs):
        random.shuffle(tasks)
        epoch_reward = 0.0
        epoch_success = 0
        epoch_total = 0

        for task in tasks:
            prompt_ids = encode_task(task)

            # Collect group of trajectories
            trajectories: list[TrajectoryData] = []
            for _ in range(group_size):
                gen_ids, log_probs = generate(model, prompt_ids, temperature=temperature)
                sexpr = detokenize(gen_ids)
                reward = compute_reward(sexpr, task)

                trajectories.append(TrajectoryData(
                    gen_ids=gen_ids,
                    log_probs=log_probs,
                    reward=reward,
                    sexpr=sexpr,
                ))
                epoch_total += 1
                if reward >= 1.0:
                    epoch_success += 1
                epoch_reward += reward

            # Group-relative advantages
            rewards = [t.reward for t in trajectories]
            mean_r = sum(rewards) / len(rewards)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
            if std_r < 1e-8:
                continue  # No signal

            advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

            # GRPO loss and update
            def loss_fn(m):
                total_loss = mx.array(0.0)
                n_tokens = 0

                for traj, adv in zip(trajectories, advantages):
                    if abs(adv) < 1e-8 or len(traj.gen_ids) == 0:
                        continue

                    # Reconstruct full sequence
                    full_ids = prompt_ids + traj.gen_ids
                    x = mx.array([full_ids[:-1]])
                    logits = m(x)[0]  # (seq_len, vocab)

                    # Get log-probs for generated tokens only
                    gen_start = len(prompt_ids)
                    for i, (token_id, old_lp) in enumerate(zip(traj.gen_ids, traj.log_probs)):
                        pos = gen_start + i
                        if pos >= logits.shape[0]:
                            break
                        new_lp = mx.log_softmax(logits[pos])[token_id]

                        # Ref model log-prob
                        ref_x = mx.array([full_ids[:pos + 1]])
                        ref_logits = ref_model(ref_x)[0]
                        ref_lp = mx.log_softmax(ref_logits[-1])[token_id]

                        # PPO-clip objective
                        ratio = mx.exp(new_lp - old_lp)
                        clipped = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                        pg_loss = -mx.minimum(ratio * adv, clipped * adv)

                        # KL penalty
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

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: reward={avg_reward:.3f} success={success_rate:.1%} ({epoch_success}/{epoch_total})")

            # Show a few generated structures
            if (epoch + 1) % 5 == 0 or epoch == 0:
                sample_task = tasks[0]
                prompt = encode_task(sample_task)
                gen_ids, _ = generate(model, prompt, temperature=0.5)
                sexpr = detokenize(gen_ids)
                r = compute_reward(sexpr, sample_task)
                print(f"  Sample ({sample_task.name}): {sexpr}")
                print(f"  Reward: {r:.3f}")
                print()

    return model
