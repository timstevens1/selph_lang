"""Model construction for neural SELPH training.

Supports both SSM (MultiScaleFFNLM from parameter-golf) and Transformer
architectures, unified behind a common interface.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PG_ROOT = Path(os.environ.get("PG_ROOT", os.path.expanduser("~/projects/parameter-golf")))
sys.path.insert(0, str(PG_ROOT))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ssm_bin_es.experiment.model import MultiScaleFFNLM


class TransformerLM(nn.Module):
    """Transformer LM with causal masking. Fair comparison baseline for SSM."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, max_len: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = [
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dim = d_model
        self.max_len = max_len
        self._vocab_size = vocab_size

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
    vocab_size: int,
    arch: str = "ssm",
    dim: int = 224,
    n_layers: int = 8,
    n_heads: int = 8,
    feat_dim: int = 96,
    state_dim: int = 8,
    num_scales: int = 2,
    mlp_mult: int = 2,
    max_len: int = 256,
) -> nn.Module:
    """Build a language model for SELPH program generation.

    arch="ssm": MultiScaleFFNLM from parameter-golf (default)
    arch="transformer": TransformerLM (fair comparison baseline)
    """
    if arch == "transformer":
        return TransformerLM(
            vocab_size=vocab_size, d_model=dim, n_heads=n_heads,
            n_layers=n_layers, max_len=max_len,
        )
    else:
        a_mults = (2.0, 1.0, 0.3)[:num_scales]
        return MultiScaleFFNLM(
            vocab_size=vocab_size,
            num_layers=n_layers,
            dim=dim,
            feat_dim=feat_dim,
            state_dim=state_dim,
            num_scales=num_scales,
            shifts=(0, 1, 2, 4),
            a_log_mults=a_mults,
            mlp_mult=mlp_mult,
            logit_softcap=30.0,
            tied_embed_init_std=0.02,
            weight_tie_layers=0,
            group_size=64,
        )


def get_logits(model: nn.Module, input_ids: mx.array, vocab_size: int) -> mx.array:
    """Get logits from either Transformer or SSM model.

    Returns shape (batch * seq, vocab_size).
    """
    x = model(input_ids)
    if isinstance(model, TransformerLM):
        return x.reshape(-1, vocab_size)
    else:
        # SSM: weight-tied projection through embedding
        x = x.reshape(-1, model.dim)
        logits = x @ model.tok_emb.weight.astype(x.dtype).T
        return model.softcap(logits)


def logits_at_last(model: nn.Module, ids: list[int]) -> mx.array:
    """Get logits at the last position for autoregressive generation.

    Returns shape (vocab_size,).
    """
    x = mx.array([ids], dtype=mx.int32)
    if isinstance(model, TransformerLM):
        return model(x)[0, -1, :]
    else:
        h = model(x)
        h_last = h[0, -1, :]
        logits = h_last @ model.tok_emb.weight.astype(h_last.dtype).T
        return model.softcap(logits)


def save_checkpoint(model: nn.Module, path: str):
    """Save model weights to .npz checkpoint."""
    weights = dict(tree_flatten(model.parameters()))
    mx.savez(path, **weights)


def load_checkpoint(model: nn.Module, path: str):
    """Load model weights from .npz checkpoint."""
    weights = dict(mx.load(path))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())


def param_count(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def clone_model(model: nn.Module, vocab_size: int) -> nn.Module:
    """Create a frozen copy of the model for use as GRPO reference."""
    if isinstance(model, TransformerLM):
        ref = TransformerLM(
            vocab_size=vocab_size,
            d_model=model.dim,
            n_heads=model.layers[0].attention.num_heads,
            n_layers=len(model.layers),
            max_len=model.max_len,
        )
    else:
        ssm_nscales = len(model.layers[0].mamba.feature_ssms)
        ssm_feat = model.layers[0].mamba.feature_ssms[0].in_proj.weight.shape[0]
        ssm_state = model.layers[0].mamba.feature_ssms[0].A_log.shape[1]
        a_mults = (2.0, 1.0, 0.3)[:ssm_nscales]
        ref = MultiScaleFFNLM(
            vocab_size=vocab_size,
            num_layers=len(model.layers),
            dim=model.dim,
            feat_dim=ssm_feat,
            state_dim=ssm_state,
            num_scales=ssm_nscales,
            shifts=(0, 1, 2, 4),
            a_log_mults=a_mults,
            mlp_mult=2,
            logit_softcap=30.0,
            tied_embed_init_std=0.02,
            weight_tie_layers=0,
            group_size=64,
        )
    ref_params = [(k, v) for k, v in tree_flatten(model.parameters())]
    ref.load_weights(ref_params)
    ref.freeze()
    return ref
