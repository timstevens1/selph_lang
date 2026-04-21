"""RefinementLM v2: token model + separate latent model.

Two networks:
    1. TokenModel — MultiScaleFFNLM backbone, runs per-token for autoregressive
       generation. Takes latent as fixed conditioning (additive bias).
       Returns logits only.

    2. LatentModel — small MultiScale SSM, runs once per refinement step.
       Reads the input context (x, y, feedback, z, query_result) and
       produces a latent vector via gated residual update.

Refinement loop:
    for step in range(max_steps):
        latent = latent_model(input_tokens, prev_latent)   # once per step
        latent = numpy_roundtrip(latent)                    # sever graph
        tokens = generate(token_model, input_tokens, latent) # per-token AR
        y, z = decode(tokens)
        feedback = eval(y)
        query_result = eval(z)

Output sequence order: z (scratchpad) first, then y (answer):
    [... SEP_QUERY_OUT z' SEP_EDIT y' EOS]
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

from ssm_bin_es.experiment.model import (
    MultiScaleFFNLM, MultiScaleLayer,
    rms_norm, COMPUTE_DTYPE,
)


class LatentModel(nn.Module):
    """Small MultiScale SSM that produces a latent vector from input context.

    Runs once per refinement step. Reads the full input sequence, pools,
    and produces a latent via gated residual update with the previous latent.
    """

    def __init__(self, vocab_size: int, dim: int = 48, feat_dim: int = 24,
                 state_dim: int = 8, num_layers: int = 1, num_scales: int = 2,
                 shifts: tuple[int, ...] = (0, 1, 2),
                 a_log_mults: tuple[float, ...] = (2.0, 0.5),
                 mlp_mult: int = 2, latent_dim: int = 48,
                 group_size: int = 64):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim

        # Separate embedding (smaller dim than token model)
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * 0.02
        ).astype(COMPUTE_DTYPE)

        # Previous latent injection
        self.latent_proj = nn.Linear(latent_dim, dim, bias=False)

        # Small SSM backbone
        layer_kwargs = dict(
            feat_dim=feat_dim, state_dim=state_dim,
            num_scales=num_scales, shifts=shifts,
            a_log_mults=a_log_mults, mlp_mult=mlp_mult,
            group_size=group_size,
        )
        self.layers = [MultiScaleLayer(dim, **layer_kwargs) for _ in range(num_layers)]

        # Output projection: pooled hidden → latent
        self.out_proj = nn.Linear(dim, latent_dim, bias=False)
        self.gate_proj = nn.Linear(dim, latent_dim, bias=False)

    def __call__(self, input_ids: mx.array, prev_latent: mx.array = None) -> mx.array:
        """Produce latent from input context.

        Args:
            input_ids: (batch, seq_len) — the full input prompt tokens
            prev_latent: (batch, latent_dim) or None
        Returns:
            new_latent: (batch, latent_dim)
        """
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))

        if prev_latent is not None:
            bias = self.latent_proj(prev_latent.astype(COMPUTE_DTYPE))
            x = x + bias[:, None, :]

        for layer in self.layers:
            x = layer(x)

        pooled = mx.mean(rms_norm(x), axis=1)  # (batch, dim)
        new_val = mx.tanh(self.out_proj(pooled))

        if prev_latent is None:
            return new_val

        gate = mx.sigmoid(self.gate_proj(pooled))
        return gate * new_val + (1.0 - gate) * prev_latent


class TokenModel(nn.Module):
    """MultiScaleFFNLM backbone for autoregressive token generation.

    Takes a fixed latent as conditioning. Returns logits only.
    """

    def __init__(self, vocab_size: int, dim: int = 96, feat_dim: int = 48,
                 state_dim: int = 8, num_layers: int = 3, num_scales: int = 3,
                 shifts: tuple[int, ...] = (0, 1, 2, 4),
                 a_log_mults: tuple[float, ...] = (2.0, 1.0, 0.3),
                 mlp_mult: int = 2, latent_dim: int = 48,
                 weight_tie_layers: int = 0, group_size: int = 64,
                 logit_softcap: float = 30.0):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * 0.02
        ).astype(COMPUTE_DTYPE)

        # Latent injection
        self.latent_proj = nn.Linear(latent_dim, dim, bias=False)

        # Backbone
        layer_kwargs = dict(
            feat_dim=feat_dim, state_dim=state_dim,
            num_scales=num_scales, shifts=shifts,
            a_log_mults=a_log_mults, mlp_mult=mlp_mult,
            group_size=group_size,
        )
        if weight_tie_layers > 0:
            num_unique = max(num_layers // weight_tie_layers, 1)
            self.layers = [MultiScaleLayer(dim, **layer_kwargs) for _ in range(num_unique)]
            self._layer_indices = [i % num_unique for i in range(num_layers)]
        else:
            self.layers = [MultiScaleLayer(dim, **layer_kwargs) for _ in range(num_layers)]
            self._layer_indices = list(range(num_layers))

        # Zero-init for stable start
        for layer in self.layers:
            layer.mamba.out_proj.weight = mx.zeros_like(layer.mamba.out_proj.weight)
            layer.ffn.down_proj.weight = mx.zeros_like(layer.ffn.down_proj.weight)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array, latent: mx.array = None) -> mx.array:
        """Forward pass. Returns logits (batch, seq_len, vocab_size).

        Args:
            input_ids: (batch, seq_len)
            latent: (batch, latent_dim) — fixed for the entire generation
        """
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))

        if latent is not None:
            bias = self.latent_proj(latent.astype(COMPUTE_DTYPE))
            x = x + bias[:, None, :]

        for layer_idx in self._layer_indices:
            x = self.layers[layer_idx](x)

        hidden = rms_norm(x)
        return self.softcap(hidden @ self.tok_emb.weight.astype(hidden.dtype).T)


    def generate(self, input_ids: mx.array, latent: mx.array = None,
                 max_new: int = 30, eos_id: int = 2) -> list[int]:
        """Autoregressive generation. Returns list of generated token IDs.

        Uses FFT backbone (no prefill/step). The latent is held fixed
        for the entire generation.
        """
        ids = input_ids
        prompt_len = ids.shape[1]

        for _ in range(max_new):
            logits = self(ids, latent)
            last_logits = logits[:, -1, :].astype(mx.float32)
            next_id = int(mx.argmax(last_logits, axis=-1).item())
            mx.eval(last_logits)
            del logits, last_logits
            if next_id == eos_id:
                break
            ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)

        gen_ids = ids[0].tolist()[prompt_len:]
        del ids
        return gen_ids


class RefinementLM(nn.Module):
    """Combined model: TokenModel + LatentModel.

    Wraps both for joint training and parameter saving.
    """

    def __init__(self, vocab_size: int,
                 # TokenModel params
                 dim: int = 96, feat_dim: int = 48,
                 state_dim: int = 8, num_layers: int = 3, num_scales: int = 3,
                 shifts: tuple[int, ...] = (0, 1, 2, 4),
                 a_log_mults: tuple[float, ...] = (2.0, 1.0, 0.3),
                 mlp_mult: int = 2, latent_dim: int = 48,
                 weight_tie_layers: int = 0, group_size: int = 64,
                 logit_softcap: float = 30.0,
                 # LatentModel params
                 lat_dim: int = 48, lat_feat_dim: int = 24,
                 lat_num_layers: int = 1, lat_num_scales: int = 2,
                 lat_shifts: tuple[int, ...] = (0, 1, 2),
                 lat_a_log_mults: tuple[float, ...] = (2.0, 0.5)):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        self.token_model = TokenModel(
            vocab_size=vocab_size, dim=dim, feat_dim=feat_dim,
            state_dim=state_dim, num_layers=num_layers, num_scales=num_scales,
            shifts=shifts, a_log_mults=a_log_mults, mlp_mult=mlp_mult,
            latent_dim=latent_dim, weight_tie_layers=weight_tie_layers,
            group_size=group_size, logit_softcap=logit_softcap,
        )

        self.latent_model = LatentModel(
            vocab_size=vocab_size, dim=lat_dim, feat_dim=lat_feat_dim,
            state_dim=state_dim, num_layers=lat_num_layers,
            num_scales=lat_num_scales, shifts=lat_shifts,
            a_log_mults=lat_a_log_mults, mlp_mult=mlp_mult,
            latent_dim=latent_dim, group_size=group_size,
        )

    def __call__(self, input_ids: mx.array, latent: mx.array = None) -> mx.array:
        """Forward pass for training. Returns logits.

        Runs latent_model on input to get latent, then token_model with that latent.
        During training, this is a single pass on the full sequence.
        """
        new_latent = self.latent_model(input_ids, latent)
        logits = self.token_model(input_ids, new_latent)
        return logits

    def loss(self, input_ids: mx.array, target_ids: mx.array,
             latent: mx.array = None) -> mx.array:
        """Cross-entropy loss with padding mask."""
        logits = self(input_ids, latent)
        log = logits.reshape(-1, self.vocab_size)
        targets_flat = target_ids.reshape(-1)
        mask = (targets_flat >= 0).astype(mx.float32)
        safe_targets = mx.maximum(targets_flat, mx.array(0, dtype=mx.int32))
        ce = nn.losses.cross_entropy(log.astype(mx.float32), safe_targets, reduction="none")
        return mx.sum(ce * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))

    @staticmethod
    def from_baseline(baseline_path: str, vocab_size: int, **kwargs) -> "RefinementLM":
        """Initialize TokenModel from trained baseline weights."""
        model = RefinementLM(vocab_size=vocab_size, **kwargs)

        weights = dict(mx.load(baseline_path))

        # Map baseline params to token_model params
        embed_weight = None
        loadable = []
        model_params = dict(tree_flatten(model.token_model.parameters()))
        for name, val in weights.items():
            if name == "tok_emb.weight":
                embed_weight = val
                continue
            prefixed = name  # baseline params map directly to token_model
            if prefixed in model_params and model_params[prefixed].shape == val.shape:
                loadable.append((prefixed, val))

        if loadable:
            model.token_model.load_weights(loadable, strict=False)

        # Copy embedding with vocab expansion
        if embed_weight is not None:
            old_vocab = embed_weight.shape[0]
            new_emb = model.token_model.tok_emb.weight
            copy_n = min(old_vocab, vocab_size)
            new_emb[:copy_n] = embed_weight[:copy_n]
            model.token_model.tok_emb.weight = new_emb

            # Also init latent_model embedding from baseline
            lat_emb = model.latent_model.tok_emb.weight
            lat_dim = lat_emb.shape[1]
            # Project baseline embedding to latent model dim via truncation
            copy_n_lat = min(old_vocab, vocab_size)
            # Use first lat_dim dimensions of baseline embedding as init
            if embed_weight.shape[1] >= lat_dim:
                lat_emb[:copy_n_lat] = embed_weight[:copy_n_lat, :lat_dim]
            model.latent_model.tok_emb.weight = lat_emb

        return model
