#!/usr/bin/env python3
"""Train neural KB query synthesizer via shared training infrastructure.

Replaces the monolithic grpo_broad.py + train_tool_sft.py with:
  KB domain adapter + shared SFT/GRPO training core.

Key upgrades over the old pipeline:
- Proper GRPO with reference model, clipped ratio, KL penalty
- Tournament multi-round collection with ratcheting
- Iterative refinement (was single-pass)
- Test-validated reward shaping

Usage:
    python -m neural_selph.training.train_kb \
        --data-dir /path/to/neural_selph/data \
        --sft-epochs 100 --grpo-epochs 30
"""
from __future__ import annotations

import argparse
import os
import random

import mlx.core as mx

from .base import traces_to_training_data
from .model import build_model, load_checkpoint, param_count, save_checkpoint
from .sft import sft_train
from .grpo import grpo_train
from .domains.kb import KBDomain


def main():
    parser = argparse.ArgumentParser(description="Neural KB query synthesizer (unified training)")
    # Data
    parser.add_argument("--data-dir", required=True,
                        help="Path to neural_selph/data/ with broad_train.json and KB")
    parser.add_argument("--kb-path", default=None,
                        help="Override KB path (default: data_dir/wikidata_resolved_with_functions.jsonl)")
    # Training phases
    parser.add_argument("--sft-epochs", type=int, default=100)
    parser.add_argument("--grpo-epochs", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--tasks-per-epoch", type=int, default=16)
    parser.add_argument("--grpo-lr", type=float, default=1e-5)
    parser.add_argument("--sft-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=256)
    # Model
    parser.add_argument("--arch", default="ssm", choices=["ssm", "transformer"])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--num-scales", type=int, default=3)
    # Checkpointing
    parser.add_argument("--checkpoint-dir", default="/tmp/neural_kb_unified")
    parser.add_argument("--load-weights", default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Domain
    domain = KBDomain(
        data_dir=args.data_dir,
        kb_path=args.kb_path,
    )

    if verbose:
        print(f"KB domain: {len(domain.tasks)} tasks, vocab={domain.vocab_size}")

    # Model — defaults match the 12.2M param config from the old pipeline
    model = build_model(
        vocab_size=domain.vocab_size,
        arch=args.arch,
        dim=args.dim,
        n_layers=args.n_layers,
        feat_dim=args.feat_dim,
        state_dim=args.state_dim,
        num_scales=args.num_scales,
    )
    mx.eval(model.parameters())

    if args.load_weights:
        load_checkpoint(model, args.load_weights)
        if verbose:
            print(f"Loaded weights from {args.load_weights}")

    if verbose:
        print(f"Model: {param_count(model):,} params, arch={args.arch}")

    # SFT Phase
    if args.sft_epochs > 0 and not args.load_weights:
        if verbose:
            print("\n── SFT Phase ──")
        traces = domain.generate_traces()
        if verbose:
            print(f"Generated {len(traces)} traces "
                  f"({sum(len(t.steps) for t in traces)} steps)")
        examples = traces_to_training_data(domain, traces, max_seq_len=args.max_seq_len)
        if verbose:
            print(f"Training examples: {len(examples)}")

        sft_train(
            model, examples,
            vocab_size=domain.vocab_size,
            pad_id=domain.pad_id,
            epochs=args.sft_epochs,
            batch_size=args.batch_size,
            lr=args.sft_lr,
            checkpoint_dir=args.checkpoint_dir,
            verbose=verbose,
        )

    # GRPO Phase
    if args.grpo_epochs > 0:
        if verbose:
            print("\n── GRPO Phase ──")

        # Use a subset of tasks for GRPO (too slow on 666K)
        random.seed(42)
        random.shuffle(domain.tasks)
        grpo_tasks = domain.tasks[:5000]  # subsample for GRPO

        best = grpo_train(
            model, domain, grpo_tasks,
            epochs=args.grpo_epochs,
            group_size=args.group_size,
            n_rounds=args.n_rounds,
            tasks_per_epoch=args.tasks_per_epoch,
            lr=args.grpo_lr,
            verbose=verbose,
            checkpoint_dir=args.checkpoint_dir,
        )
        n_correct = sum(1 for r, _ in best.values() if r >= 1.0)
        if verbose:
            print(f"\nGRPO complete: {n_correct} correct solutions")

    # Cleanup
    domain.stop()


if __name__ == "__main__":
    main()
