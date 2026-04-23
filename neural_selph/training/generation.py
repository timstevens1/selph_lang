"""Autoregressive generation with query interception.

Shared generation logic: the model generates tokens autoregressively;
when it emits <query>, we collect tokens until </query>, hand them to
the domain's query handler, and inject the result as <q_out>...</q_out>.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from .model import logits_at_last

if TYPE_CHECKING:
    import mlx.nn as nn
    from .base import Domain


def generate_tokens(
    model: nn.Module,
    prompt_ids: list[int],
    eos_id: int,
    max_new: int = 60,
    temperature: float = 0.7,
) -> tuple[list[int], list[float]]:
    """Autoregressive generation from a prompt.

    Returns (generated_token_ids, log_probs).
    No query interception — use generate_with_queries for that.
    """
    ids = list(prompt_ids)
    log_probs = []

    for _ in range(max_new):
        logits = logits_at_last(model, ids)
        if temperature > 0:
            logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        token = mx.random.categorical(logits[None, :])[0].item()
        lp = mx.log(probs[token] + 1e-10).item()
        log_probs.append(lp)
        ids.append(token)

        if token == eos_id:
            break

    return ids[len(prompt_ids):], log_probs


def generate_with_queries(
    model: nn.Module,
    prompt_ids: list[int],
    domain: Domain,
    task: object,
    max_new: int = 80,
    max_queries: int = 5,
    temperature: float = 0.7,
) -> tuple[list[int], list[float]]:
    """Autoregressive generation that intercepts <query> tokens.

    When the model emits <query>, we collect tokens until </query>,
    call domain.handle_query(), and inject <q_out>result</q_out>.
    Injected tokens are appended to the context but NOT to log_probs
    (they aren't model-generated).

    Returns (all_generated_ids, model_log_probs).
    """
    ids = list(prompt_ids)
    log_probs = []
    n_queries = 0
    i = 0

    while i < max_new:
        logits = logits_at_last(model, ids)
        if temperature > 0:
            logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        token = mx.random.categorical(logits[None, :])[0].item()
        lp = mx.log(probs[token] + 1e-10).item()
        log_probs.append(lp)
        ids.append(token)
        i += 1

        if token == domain.eos_id:
            break

        # Intercept <query>: collect until </query>, eval, inject result
        if token == domain.query_open_id and n_queries < max_queries:
            query_ids = []
            for _ in range(40):
                qlogits = logits_at_last(model, ids)
                if temperature > 0:
                    qlogits = qlogits / temperature
                qprobs = mx.softmax(qlogits, axis=-1)
                tk = mx.random.categorical(qlogits[None, :])[0].item()
                lp2 = mx.log(qprobs[tk] + 1e-10).item()
                log_probs.append(lp2)
                ids.append(tk)
                i += 1
                if tk == domain.query_close_id:
                    break
                query_ids.append(tk)

            # Domain evaluates the query and returns result token IDs
            result_ids = domain.handle_query(query_ids, task)
            n_queries += 1

            # Inject result (not model-generated, no log_probs)
            ids.append(domain.qout_open_id)
            ids.extend(result_ids)
            ids.append(domain.qout_close_id)

    return ids[len(prompt_ids):], log_probs
