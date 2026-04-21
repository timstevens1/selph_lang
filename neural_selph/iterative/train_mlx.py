#!/usr/bin/env python3
"""Train a MultiScale SSM on s-expression refinement traces using MLX.

Uses the MLX MultiScaleFFNLM from parameter-golf directly.

Usage:
    python3 -m neural_selph.iterative.train_mlx
    python3 -m neural_selph.iterative.train_mlx --epochs 200 --dim 128
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Add parameter-golf to path so we can import the MLX model
PG_ROOT = Path(os.environ.get("PG_ROOT", os.path.expanduser("~/projects/parameter-golf")))
sys.path.insert(0, str(PG_ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ssm_bin_es.experiment.model import (
    MultiScaleFFNLM, MultiScaleLayer, S4ProjectionSSM,
    MultiScaleSSMBlock, BinaryGatedFFN, rms_norm, COMPUTE_DTYPE,
)

from neural_selph.iterative.sexpr_tokenizer import SexprTokenizer, build_default_tokenizer

DATA_DIR = Path(__file__).parent.parent / "data"


# ============================================================================
# Data Loading
# ============================================================================

def load_traces(tokenizer: SexprTokenizer, max_seq_len: int = 128
                ) -> list[tuple[list[int], list[int]]]:
    """Load traces and convert to (input_ids, target_ids) pairs."""
    steps = json.loads((DATA_DIR / "geo_traces.json").read_text())

    examples = []
    for step in steps:
        tokens = []

        # Task context: question type + entity names
        tokens.append("SEP_TASK")
        if step.get("q_type"):
            q_type = step["q_type"]
            if q_type in tokenizer.token_to_id:
                tokens.append(q_type)
        for entity in step.get("entities", []):
            if entity in tokenizer.token_to_id:
                tokens.append(entity)

        # Current expression
        tokens.append("SEP_EXPR")
        if step["current_expr"] and step["current_expr"] != "_HOLE_":
            tokens.extend(tokenizer.tokenize_sexpr(step["current_expr"]))
        else:
            tokens.append("_HOLE_")

        tokens.append("SEP_FEEDBACK")
        tokens.append(step["feedback_type"])
        if step["feedback_value"]:
            tokens.extend(tokenizer.tokenize_number(step["feedback_value"]))

        # Queries (thinking in SELPH)
        for q_expr, q_result in step.get("queries", []):
            tokens.append("<query>")
            tokens.extend(tokenizer.tokenize_sexpr(q_expr))
            tokens.append("</query>")
            tokens.append("<q_out>")
            # Tokenize result
            try:
                float(q_result)
                tokens.extend(tokenizer.tokenize_number(q_result))
            except ValueError:
                if q_result in tokenizer.token_to_id:
                    tokens.append(q_result)
            tokens.append("</q_out>")

        tokens.append("SEP_EDIT")
        if step["target_expr"] == "NO_OP":
            tokens.append("NO_OP")
        else:
            tokens.extend(tokenizer.tokenize_sexpr(step["target_expr"]))

        ids = tokenizer.encode(tokens, add_bos=True, add_eos=True)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        # input = ids[:-1], target = ids[1:]
        examples.append((ids[:-1], ids[1:]))

    return examples


def make_batch(examples: list[tuple[list[int], list[int]]],
               indices: list[int], pad_id: int
               ) -> tuple[mx.array, mx.array]:
    """Create a padded batch from selected indices."""
    batch_in = [examples[i][0] for i in indices]
    batch_tgt = [examples[i][1] for i in indices]
    max_len = max(len(s) for s in batch_in)

    # Pad inputs with pad_id, targets with -1 (will be masked)
    padded_in = []
    padded_tgt = []
    for inp, tgt in zip(batch_in, batch_tgt):
        pad_len = max_len - len(inp)
        padded_in.append(inp + [pad_id] * pad_len)
        padded_tgt.append(tgt + [-1] * pad_len)

    return mx.array(padded_in, dtype=mx.int32), mx.array(padded_tgt, dtype=mx.int32)


# ============================================================================
# Model Construction
# ============================================================================

def build_model(vocab_size: int, args) -> MultiScaleFFNLM:
    """Build a MultiScaleFFNLM for s-expression refinement."""
    a_log_mults = tuple(args.a_log_mults)
    if len(a_log_mults) != args.num_scales:
        # Auto-generate
        a_log_mults = tuple(
            2.0 * (0.1 ** (i / max(1, args.num_scales - 1)))
            for i in range(args.num_scales)
        )

    model = MultiScaleFFNLM(
        vocab_size=vocab_size,
        num_layers=args.num_layers,
        dim=args.dim,
        feat_dim=args.feat_dim,
        state_dim=args.state_dim,
        num_scales=args.num_scales,
        shifts=tuple(args.shifts),
        a_log_mults=a_log_mults,
        mlp_mult=args.mlp_mult,
        logit_softcap=30.0,
        tied_embed_init_std=0.02,
        weight_tie_layers=args.weight_tie_layers,
        group_size=64,
    )
    return model


# ============================================================================
# Loss Function
# ============================================================================

def loss_fn(model: MultiScaleFFNLM, input_ids: mx.array, target_ids: mx.array) -> mx.array:
    """Cross-entropy loss with padding mask."""
    x = model(input_ids)  # (batch, seq_len, dim)
    x = x.reshape(-1, model.dim)

    logits = x @ model.tok_emb.weight.astype(x.dtype).T
    logits = model.softcap(logits)

    targets_flat = target_ids.reshape(-1)

    # Mask padding (target == -1) — clamp to 0 for CE, then mask the loss
    mask = (targets_flat >= 0).astype(mx.float32)
    safe_targets = mx.maximum(targets_flat, mx.array(0, dtype=mx.int32))

    loss = nn.losses.cross_entropy(logits.astype(mx.float32), safe_targets, reduction="none")
    return mx.sum(loss * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))


# ============================================================================
# Training
# ============================================================================

def train(args):
    print(f"MLX backend on {mx.default_device()}")

    tokenizer = build_default_tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    examples = load_traces(tokenizer, max_seq_len=args.max_seq_len)
    print(f"Training examples: {len(examples)}")

    # Split
    random.seed(42)
    indices = list(range(len(examples)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    # Model
    model = build_model(tokenizer.vocab_size, args)
    mx.eval(model.parameters())

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Optimizer: Adam with cosine schedule
    total_steps = args.epochs * (len(train_idx) // args.batch_size + 1)

    schedule = optim.cosine_decay(args.lr, total_steps, 1e-6)
    if args.warmup_steps > 0:
        warmup = optim.linear_schedule(1e-7, args.lr, args.warmup_steps)
        schedule = optim.join_schedules([warmup, schedule], [args.warmup_steps])

    optimizer = optim.Adam(learning_rate=schedule)

    # Compile loss + grad
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop
    best_val_loss = float("inf")
    t0 = time.time()
    step = 0

    for epoch in range(args.epochs):
        random.shuffle(train_idx)
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_start in range(0, len(train_idx), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(train_idx))
            batch_idx = train_idx[batch_start:batch_end]

            inputs, targets = make_batch(examples, batch_idx, tokenizer.pad_id)

            loss, grads = loss_and_grad_fn(model, inputs, targets)

            # Gradient clipping — critical for BinaryLinear stability
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            epoch_batches += 1
            step += 1

        train_loss = epoch_loss / max(epoch_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        for batch_start in range(0, len(val_idx), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(val_idx))
            batch_idx = val_idx[batch_start:batch_end]
            inputs, targets = make_batch(examples, batch_idx, tokenizer.pad_id)
            loss = loss_fn(model, inputs, targets)
            mx.eval(loss)
            val_loss_sum += loss.item()
            val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model weights
            save_path = DATA_DIR / "iterative_model_mlx.npz"
            flat = dict(tree_flatten(model.parameters()))
            mx.savez(str(save_path), **flat)
            marker = " *"

        if epoch % max(1, args.epochs // 30) == 0 or epoch == args.epochs - 1:
            lr_val = schedule(step) if callable(schedule) else args.lr
            print(f"epoch {epoch:4d}/{args.epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  lr={lr_val:.6f}  "
                  f"elapsed={elapsed:.0f}s{marker}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Total time: {time.time() - t0:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--feat-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--shifts", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--a-log-mults", type=float, nargs="+", default=[2.0, 1.0, 0.3])
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--weight-tie-layers", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
