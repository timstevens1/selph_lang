"""Supervised fine-tuning (SFT) on refinement traces.

Domain-agnostic: operates on (input_ids, target_ids) pairs produced
by traces_to_training_data().
"""
from __future__ import annotations

import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .model import get_logits, save_checkpoint, param_count


def _make_batch(
    examples: list[tuple[list[int], list[int]]],
    indices: list[int],
    pad_id: int,
) -> tuple[mx.array, mx.array]:
    """Create a padded batch from examples at the given indices."""
    batch_in = [examples[i][0] for i in indices]
    batch_tgt = [examples[i][1] for i in indices]
    max_len = max(len(s) for s in batch_in)

    padded_in = [inp + [pad_id] * (max_len - len(inp)) for inp in batch_in]
    padded_tgt = [tgt + [-1] * (max_len - len(tgt)) for tgt in batch_tgt]

    return mx.array(padded_in, dtype=mx.int32), mx.array(padded_tgt, dtype=mx.int32)


def _loss_fn(model: nn.Module, input_ids: mx.array, target_ids: mx.array,
             vocab_size: int) -> mx.array:
    """Cross-entropy loss with padding mask (-1 targets are masked)."""
    logits = get_logits(model, input_ids, vocab_size)
    targets_flat = target_ids.reshape(-1)
    mask = (targets_flat >= 0).astype(mx.float32)
    safe_targets = mx.maximum(targets_flat, mx.array(0, dtype=mx.int32))
    loss = nn.losses.cross_entropy(logits.astype(mx.float32), safe_targets, reduction="none")
    return mx.sum(loss * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))


def sft_train(
    model: nn.Module,
    examples: list[tuple[list[int], list[int]]],
    vocab_size: int,
    pad_id: int = 0,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    warmup_steps: int = 50,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 20,
    verbose: bool = True,
) -> float:
    """SFT on (input_ids, target_ids) pairs.

    Returns best validation loss.
    """
    if not examples:
        return 0.0

    # Train/val split
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    train_data = examples[:split]
    val_data = examples[split:]

    # Cosine schedule with warmup
    total_steps = epochs * (len(train_data) // batch_size + 1)
    decay_steps = max(1, total_steps - warmup_steps)
    schedule = optim.cosine_decay(lr, decay_steps)
    warmup = optim.linear_schedule(1e-6, lr, warmup_steps)
    lr_schedule = optim.join_schedules([warmup, schedule], [warmup_steps])
    optimizer = optim.Adam(learning_rate=lr_schedule)

    if verbose:
        n_params = param_count(model)
        print(f"Model: {n_params:,} params, {vocab_size} vocab", flush=True)
        print(f"SFT: {len(train_data)} train, {len(val_data)} val, {epochs} epochs", flush=True)

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        random.shuffle(train_data)
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch_idx = list(range(i, min(i + batch_size, len(train_data))))
            inp, tgt = _make_batch(train_data, batch_idx, pad_id)

            loss, grads = nn.value_and_grad(
                model, lambda m: _loss_fn(m, inp, tgt, vocab_size)
            )(model)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            lv = float(loss.item())
            if not math.isnan(lv):
                epoch_loss += lv
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        if val_data:
            for i in range(0, len(val_data), batch_size):
                batch_idx = list(range(i, min(i + batch_size, len(val_data))))
                inp, tgt = _make_batch(val_data, batch_idx, pad_id)
                vl = _loss_fn(model, inp, tgt, vocab_size)
                mx.eval(vl)
                val_loss += float(vl.item())
                val_n += 1
            val_loss /= max(val_n, 1)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            if checkpoint_dir:
                save_checkpoint(model, os.path.join(checkpoint_dir, "sft_best.npz"))
                marker = " *"

        # Periodic checkpoint
        if checkpoint_dir and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"sft_epoch{epoch+1}.npz")
            save_checkpoint(model, path)
            marker += f" [ckpt]"

        if verbose and ((epoch + 1) % max(1, epochs // 30) == 0 or epoch == 0 or epoch == epochs - 1):
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1}/{epochs}: train={avg_loss:.4f} val={val_loss:.4f} "
                  f"best={best_val:.4f} elapsed={elapsed:.0f}s{marker}", flush=True)

    # Save final
    if checkpoint_dir:
        save_checkpoint(model, os.path.join(checkpoint_dir, "sft_final.npz"))

    if verbose:
        print(f"  SFT complete: best_val={best_val:.4f}", flush=True)
    return best_val
