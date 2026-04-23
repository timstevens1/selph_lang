#!/usr/bin/env python3
"""Train neural ARC synthesizer via shared training infrastructure.

Replaces the monolithic neural_arc_iterative.py with:
  ARC domain adapter + shared SFT/GRPO training core.

Usage:
    python -m neural_selph.training.train_arc \
        --arc-dir arc_data/data/training \
        --checkpoint /tmp/arc_agi1_solved.checkpoint \
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
from .domains.arc import ArcDomain


def main():
    parser = argparse.ArgumentParser(description="Neural ARC synthesizer (unified training)")
    parser.add_argument("--arc-dir", default="arc_data/data/training")
    parser.add_argument("--checkpoint", default=None,
                        help="SELPH checkpoint with solved programs")
    parser.add_argument("--sft-epochs", type=int, default=100)
    parser.add_argument("--grpo-epochs", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--tasks-per-epoch", type=int, default=64)
    parser.add_argument("--grpo-lr", type=float, default=3e-6)
    # Model
    parser.add_argument("--arch", default="ssm", choices=["ssm", "transformer"])
    parser.add_argument("--dim", type=int, default=224)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--feat-dim", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--num-scales", type=int, default=2)
    # Data
    parser.add_argument("--n-synthetic", type=int, default=2000)
    parser.add_argument("--n-color-perm", type=int, default=25)
    parser.add_argument("--checkpoint-dir", default="/tmp/neural_arc_unified")
    parser.add_argument("--load-weights", default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Suppress Rust panic stderr
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    # Domain
    domain = ArcDomain(
        arc_dir=args.arc_dir,
        checkpoint=args.checkpoint,
        n_synthetic=args.n_synthetic,
        n_color_perm=args.n_color_perm,
    )

    if verbose:
        print(f"ARC domain: {len(domain.tasks)} tasks, "
              f"{len(domain.solutions)} solved, vocab={domain.vocab_size}")

    # Model
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
        examples = traces_to_training_data(domain, traces)
        if verbose:
            print(f"Training examples: {len(examples)}")

        sft_train(
            model, examples,
            vocab_size=domain.vocab_size,
            pad_id=domain.pad_id,
            epochs=args.sft_epochs,
            checkpoint_dir=args.checkpoint_dir,
            verbose=verbose,
        )

    # GRPO Phase
    if args.grpo_epochs > 0:
        if verbose:
            print("\n── GRPO Phase ──")
        best = grpo_train(
            model, domain, domain.tasks,
            epochs=args.grpo_epochs,
            group_size=args.group_size,
            n_rounds=args.n_rounds,
            tasks_per_epoch=args.tasks_per_epoch,
            lr=args.grpo_lr,
            verbose=verbose,
            checkpoint_dir=args.checkpoint_dir,
        )
        n_perfect = sum(1 for r, _ in best.values() if r >= 5.0)
        if verbose:
            print(f"\nGRPO complete: {n_perfect} perfect solutions")

    # Restore stderr
    os.dup2(old_stderr_fd, 2)
    os.close(old_stderr_fd)


if __name__ == "__main__":
    main()
