"""Shared training infrastructure for neural SELPH synthesis.

Provides domain-agnostic SFT and GRPO training loops with a pluggable
Domain protocol. Domains (ARC grids, knowledge base queries, fitting tasks)
implement the protocol; the training core handles model building, generation,
optimization, and checkpointing.

Usage:
    from neural_selph.training import sft_train, grpo_train, build_model
    from neural_selph.training.domains.arc import ArcDomain

    domain = ArcDomain(arc_dir="arc_data/data/training", checkpoint="solved.checkpoint")
    model = build_model(domain.vocab_size, arch="ssm", dim=224, n_layers=8)
    sft_train(model, domain.generate_traces(), ...)
    grpo_train(model, domain, domain.tasks, ...)
"""

from .base import Domain, RefinementStep, RefinementTrace, Trajectory
from .model import build_model
from .sft import sft_train
from .grpo import grpo_train, tournament_collect_trajectories
from .generation import generate_tokens, generate_with_queries

__all__ = [
    "Domain", "RefinementStep", "RefinementTrace", "Trajectory",
    "build_model", "sft_train", "grpo_train",
    "tournament_collect_trajectories",
    "generate_tokens", "generate_with_queries",
]
