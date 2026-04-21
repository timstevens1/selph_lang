"""Train RefinementLM v2 on traces with query channel.

Phase 1: Independent steps (no latent passing). Teacher-forced on full
    sequences: [x, y, y(x), z, z(x,y)] → [z', y'].

Phase 2: Unrolled traces with latent passing across steps. The latent
    model learns to carry information across refinement steps via
    gradient flow through the full trace.

Usage:
    python3 -m neural_selph.iterative.generate_traces_v2   # generate traces first
    python3 -m neural_selph.iterative.train_v2              # phase 1 from scratch
    python3 -m neural_selph.iterative.train_v2 --from-baseline  # warm-start from baseline
    python3 -m neural_selph.iterative.train_v2 --phase2 --phase2-epochs 100  # phase 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import os, sys

PG_ROOT = Path(os.environ.get("PG_ROOT", os.path.expanduser("~/projects/parameter-golf")))
sys.path.insert(0, str(PG_ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ssm_bin_es.experiment.model import MultiScaleFFNLM

from neural_selph.iterative.model_v2 import RefinementLM
from neural_selph.iterative.sexpr_tokenizer import SexprTokenizer, build_default_tokenizer

DATA_DIR = Path(__file__).parent.parent / "data"


def load_steps(tokenizer: SexprTokenizer, max_seq_len: int = 128
               ) -> list[tuple[list[int], list[int]]]:
    """Load v2 trace steps (flattened) for Phase 1 training."""
    trace_path = DATA_DIR / "geo_traces_v2.json"
    if not trace_path.exists():
        print("No v2 traces found. Run: python3 -m neural_selph.iterative.generate_traces_v2")
        sys.exit(1)

    steps = json.loads(trace_path.read_text())
    return [encode_step(step, tokenizer, max_seq_len) for step in steps]


def encode_step(step: dict, tokenizer: SexprTokenizer, max_seq_len: int = 128
                ) -> tuple[list[int], list[int]]:
    """Encode a single trace step as (input_ids, target_ids)."""
    tokens = []
    tokens.append("SEP_TASK")
    q_type = step.get("q_type", "")
    if q_type and q_type in tokenizer.token_to_id:
        tokens.append(q_type)
    for entity in step.get("entities", []):
        if entity in tokenizer.token_to_id:
            tokens.append(entity)

    tokens.append("SEP_EXPR")
    curr = step["current_expr"]
    if curr and curr != "_HOLE_":
        tokens.extend(tokenizer.tokenize_sexpr(curr))
    else:
        tokens.append("_HOLE_")

    tokens.append("SEP_FEEDBACK")
    tokens.append(step["feedback_type"])
    if step["feedback_value"]:
        tokens.extend(tokenizer.tokenize_number(step["feedback_value"]))

    tokens.append("SEP_QUERY")
    q_expr = step.get("query_expr", "NO_OP")
    if q_expr and q_expr != "NO_OP":
        tokens.extend(tokenizer.tokenize_sexpr(q_expr))
    else:
        tokens.append("NO_OP")

    tokens.append("SEP_QUERY_RESULT")
    q_result = step.get("query_result", "")
    if q_result:
        try:
            float(q_result)
            tokens.extend(tokenizer.tokenize_number(q_result))
        except ValueError:
            if q_result in tokenizer.token_to_id:
                tokens.append(q_result)

    tokens.append("SEP_QUERY_OUT")
    tgt_q = step.get("target_query", "NO_OP")
    if tgt_q and tgt_q != "NO_OP":
        tokens.extend(tokenizer.tokenize_sexpr(tgt_q))
    else:
        tokens.append("NO_OP")

    tokens.append("SEP_EDIT")
    target = step["target_expr"]
    if target == "NO_OP":
        tokens.append("NO_OP")
    else:
        tokens.extend(tokenizer.tokenize_sexpr(target))

    ids = tokenizer.encode(tokens, add_bos=True, add_eos=True)
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]

    return ids[:-1], ids[1:]


def load_traces(tokenizer: SexprTokenizer, max_seq_len: int = 128
                ) -> list[list[tuple[list[int], list[int]]]]:
    """Load full traces as lists of encoded steps for Phase 2."""
    trace_path = DATA_DIR / "geo_traces_v2_full.json"
    if not trace_path.exists():
        print("No v2 full traces found. Run: python3 -m neural_selph.iterative.generate_traces_v2")
        sys.exit(1)

    traces = json.loads(trace_path.read_text())
    encoded = []
    for trace in traces:
        steps = [encode_step(step, tokenizer, max_seq_len) for step in trace]
        if len(steps) >= 2:  # need at least 2 steps for latent to matter
            encoded.append(steps)
    return encoded


def make_batch(examples, indices, pad_id):
    batch_in = [examples[i][0] for i in indices]
    batch_tgt = [examples[i][1] for i in indices]
    max_len = max(len(s) for s in batch_in)
    padded_in = [inp + [pad_id] * (max_len - len(inp)) for inp in batch_in]
    padded_tgt = [tgt + [-1] * (max_len - len(tgt)) for tgt in batch_tgt]
    return mx.array(padded_in, dtype=mx.int32), mx.array(padded_tgt, dtype=mx.int32)


def loss_fn(model, input_ids, target_ids):
    return model.loss(input_ids, target_ids)


def train(args):
    print(f"MLX backend on {mx.default_device()}")

    tokenizer = build_default_tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    examples = load_steps(tokenizer, max_seq_len=args.max_seq_len)
    print(f"Training examples: {len(examples)}")

    # Split
    random.seed(42)
    indices = list(range(len(examples)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    # Build model
    if args.from_baseline:
        baseline_path = DATA_DIR / "iterative_model_mlx.npz"
        if not baseline_path.exists():
            print(f"Baseline model not found at {baseline_path}")
            sys.exit(1)
        model = RefinementLM.from_baseline(
            str(baseline_path), vocab_size=tokenizer.vocab_size,
            dim=args.dim, feat_dim=args.feat_dim, state_dim=args.state_dim,
            num_layers=args.num_layers, num_scales=args.num_scales,
            shifts=tuple(args.shifts), a_log_mults=tuple(args.a_log_mults),
            mlp_mult=args.mlp_mult, latent_dim=args.latent_dim,
        )
        print("Warm-started from baseline model")
    else:
        model = RefinementLM(
            vocab_size=tokenizer.vocab_size,
            dim=args.dim, feat_dim=args.feat_dim, state_dim=args.state_dim,
            num_layers=args.num_layers, num_scales=args.num_scales,
            shifts=tuple(args.shifts), a_log_mults=tuple(args.a_log_mults),
            mlp_mult=args.mlp_mult, latent_dim=args.latent_dim, group_size=64,
        )

    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Optimizer
    total_steps = args.epochs * (len(train_idx) // args.batch_size + 1)
    schedule = optim.cosine_decay(args.lr, total_steps, 1e-6)
    if args.warmup_steps > 0:
        warmup = optim.linear_schedule(1e-7, args.lr, args.warmup_steps)
        schedule = optim.join_schedules([warmup, schedule], [args.warmup_steps])
    optimizer = optim.Adam(learning_rate=schedule)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

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

            loss, grads = loss_and_grad(model, inputs, targets)
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
            save_path = DATA_DIR / "refinement_v2_mlx.npz"
            flat = dict(tree_flatten(model.parameters()))
            mx.savez(str(save_path), **flat)
            marker = " *"

        if epoch % max(1, args.epochs // 30) == 0 or epoch == args.epochs - 1:
            lr_val = schedule(step) if callable(schedule) else args.lr
            print(f"epoch {epoch:4d}/{args.epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  lr={lr_val:.6f}  "
                  f"elapsed={elapsed:.0f}s{marker}")

    print(f"\nPhase 1 best val loss: {best_val_loss:.4f}")
    print(f"Phase 1 time: {time.time() - t0:.0f}s")


def unrolled_trace_loss(model, trace_inputs, trace_targets, pad_id):
    """Loss over a full trace with latent passing across steps.

    Gradient flows through latent_model across all steps.
    """
    latent = mx.zeros((1, model.latent_dim), dtype=mx.float32)
    total_loss = mx.array(0.0)

    for inp, tgt in zip(trace_inputs, trace_targets):
        inp_mx = mx.array([inp], dtype=mx.int32)
        tgt_mx = mx.array([tgt], dtype=mx.int32)

        # Latent model: gradient flows through
        latent = model.latent_model(inp_mx, latent)

        # Token model: loss with this latent
        logits = model.token_model(inp_mx, latent)
        log = logits.reshape(-1, model.vocab_size)
        targets_flat = tgt_mx.reshape(-1)
        mask = (targets_flat >= 0).astype(mx.float32)
        safe_targets = mx.maximum(targets_flat, mx.array(0, dtype=mx.int32))
        ce = nn.losses.cross_entropy(log.astype(mx.float32), safe_targets, reduction="none")
        step_loss = mx.sum(ce * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))
        total_loss = total_loss + step_loss

    return total_loss / len(trace_inputs)


def train_phase2(args):
    """Phase 2: unrolled trace training with latent passing."""
    print(f"MLX backend on {mx.default_device()}")
    print("=== Phase 2: Unrolled Trace Training ===")

    tokenizer = build_default_tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Load full traces
    traces = load_traces(tokenizer, max_seq_len=args.max_seq_len)
    print(f"Traces with ≥2 steps: {len(traces)}")
    print(f"Avg steps/trace: {sum(len(t) for t in traces) / len(traces):.1f}")

    # Split
    random.seed(42)
    indices = list(range(len(traces)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    # Load Phase 1 model
    model = RefinementLM(vocab_size=tokenizer.vocab_size,
                         dim=args.dim, feat_dim=args.feat_dim,
                         state_dim=args.state_dim, num_layers=args.num_layers,
                         num_scales=args.num_scales, shifts=tuple(args.shifts),
                         a_log_mults=tuple(args.a_log_mults),
                         mlp_mult=args.mlp_mult, latent_dim=args.latent_dim,
                         group_size=64)
    weights_path = DATA_DIR / "refinement_v2_mlx.npz"
    if weights_path.exists():
        weights = dict(mx.load(str(weights_path)))
        model.load_weights(list(weights.items()))
        print("Loaded Phase 1 weights")
    else:
        print("WARNING: No Phase 1 weights found, training from scratch")

    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Optimizer: lower LR for fine-tuning
    p2_lr = args.lr * 0.3
    total_steps = args.phase2_epochs * len(train_idx)
    schedule = optim.cosine_decay(p2_lr, total_steps, 1e-7)
    if args.warmup_steps > 0:
        warmup = optim.linear_schedule(1e-7, p2_lr, min(args.warmup_steps, total_steps // 5))
        schedule = optim.join_schedules([warmup, schedule],
                                       [min(args.warmup_steps, total_steps // 5)])
    optimizer = optim.Adam(learning_rate=schedule)

    def trace_loss_fn(model, trace_inputs, trace_targets):
        return unrolled_trace_loss(model, trace_inputs, trace_targets, tokenizer.pad_id)

    loss_and_grad = nn.value_and_grad(model, trace_loss_fn)

    best_val_loss = float("inf")
    t0 = time.time()
    step = 0

    for epoch in range(args.phase2_epochs):
        random.shuffle(train_idx)
        model.train()
        epoch_loss = 0.0
        n_traces = 0

        for ti in train_idx:
            trace = traces[ti]
            trace_inputs = [s[0] for s in trace]
            trace_targets = [s[1] for s in trace]

            loss, grads = loss_and_grad(model, trace_inputs, trace_targets)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            n_traces += 1
            step += 1

        train_loss = epoch_loss / max(n_traces, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        for vi in val_idx:
            trace = traces[vi]
            trace_inputs = [s[0] for s in trace]
            trace_targets = [s[1] for s in trace]
            loss = trace_loss_fn(model, trace_inputs, trace_targets)
            mx.eval(loss)
            val_loss_sum += loss.item()
            val_n += 1

        val_loss = val_loss_sum / max(val_n, 1)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = DATA_DIR / "refinement_v2_mlx.npz"
            flat = dict(tree_flatten(model.parameters()))
            mx.savez(str(save_path), **flat)
            marker = " *"

        if epoch % max(1, args.phase2_epochs // 20) == 0 or epoch == args.phase2_epochs - 1:
            lr_val = schedule(step) if callable(schedule) else p2_lr
            print(f"p2 epoch {epoch:4d}/{args.phase2_epochs}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  lr={lr_val:.6f}  "
                  f"elapsed={elapsed:.0f}s{marker}")

    print(f"\nPhase 2 best val loss: {best_val_loss:.4f}")
    print(f"Phase 2 time: {time.time() - t0:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--feat-dim", type=int, default=48)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--shifts", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--a-log-mults", type=float, nargs="+", default=[2.0, 1.0, 0.3])
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--from-baseline", action="store_true",
                        help="Warm-start from trained baseline model")
    parser.add_argument("--phase2", action="store_true",
                        help="Run Phase 2 (unrolled trace training)")
    parser.add_argument("--phase2-epochs", type=int, default=100)
    args = parser.parse_args()
    if args.phase2:
        train_phase2(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
