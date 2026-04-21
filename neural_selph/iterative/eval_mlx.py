#!/usr/bin/env python3
"""Evaluate iterative refinement on geo KB tasks using the MLX model.

Compares:
1. Iterative: start from _HOLE_, refine with execution feedback, up to N steps
2. One-shot: predict the full expression in a single generation pass

Usage:
    python3 -m neural_selph.iterative.eval_mlx
    python3 -m neural_selph.iterative.eval_mlx --max-steps 5 --max-examples 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PG_ROOT = Path(os.environ.get("PG_ROOT", os.path.expanduser("~/projects/parameter-golf")))
sys.path.insert(0, str(PG_ROOT))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ssm_bin_es.experiment.model import MultiScaleFFNLM

from neural_selph.iterative.sexpr_tokenizer import SexprTokenizer, build_default_tokenizer
from neural_selph.iterative.generate_traces import (
    evaluate_in_kb, parse_sexpr, extract_expr, extract_question_context,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def generate_tokens(model: MultiScaleFFNLM, input_ids: mx.array,
                    tokenizer: SexprTokenizer, max_new_tokens: int = 40,
                    temperature: float = 0.0) -> list[int]:
    """Autoregressive generation (no query support)."""
    ids = input_ids[None, :]  # (1, seq_len)
    generated = []

    for _ in range(max_new_tokens):
        x = model(ids)  # (1, seq_len, dim)
        logits = x[:, -1, :] @ model.tok_emb.weight.astype(x.dtype).T
        logits = model.softcap(logits)
        logits = logits.astype(mx.float32)

        if temperature == 0:
            next_id = int(mx.argmax(logits, axis=-1).item())
        else:
            probs = mx.softmax(logits / temperature, axis=-1)
            next_id = int(mx.random.categorical(mx.log(probs)).item())

        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break

        ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)

    return generated


def generate_with_queries(model: MultiScaleFFNLM, input_ids: mx.array,
                          tokenizer: SexprTokenizer, kb: dict,
                          max_new_tokens: int = 60, max_queries: int = 5
                          ) -> list[int]:
    """Autoregressive generation with interactive query evaluation.

    When the model emits <query>, we let it generate until </query>,
    evaluate the s-expression against the KB, inject <q_out>result</q_out>,
    then let it continue generating. Generation ends at EOS or SEP_EDIT
    followed by the edit expression.
    """
    ids = input_ids[None, :]
    generated = []

    query_start_id = tokenizer.token_to_id.get("<query>")
    query_end_id = tokenizer.token_to_id.get("</query>")
    qout_start_id = tokenizer.token_to_id.get("<q_out>")
    qout_end_id = tokenizer.token_to_id.get("</q_out>")

    n_queries = 0

    for _ in range(max_new_tokens):
        x = model(ids)
        logits = x[:, -1, :] @ model.tok_emb.weight.astype(x.dtype).T
        logits = model.softcap(logits).astype(mx.float32)
        next_id = int(mx.argmax(logits, axis=-1).item())
        mx.eval(logits)
        del x, logits

        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break

        ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)

        # Model emitted <query> — generate until </query>, then evaluate
        if next_id == query_start_id and n_queries < max_queries:
            query_ids = []
            for _ in range(20):  # max query length
                x = model(ids)
                logits = x[:, -1, :] @ model.tok_emb.weight.astype(x.dtype).T
                logits = model.softcap(logits).astype(mx.float32)
                qid = int(mx.argmax(logits, axis=-1).item())
                mx.eval(logits)
                del x, logits

                generated.append(qid)
                ids = mx.concatenate([ids, mx.array([[qid]], dtype=mx.int32)], axis=1)

                if qid == query_end_id:
                    break
                query_ids.append(qid)

            # Evaluate the query
            query_sexpr = tokenizer.decode_to_sexpr(query_ids)
            q_result = ""
            if query_sexpr and "_HOLE_" not in query_sexpr:
                tree = parse_sexpr(query_sexpr)
                result = evaluate_in_kb(tree, kb)
                if result is not None:
                    q_result = str(result)

            # Inject <q_out> result </q_out>
            result_tokens = ["<q_out>"]
            if q_result:
                try:
                    float(q_result)
                    result_tokens.extend(tokenizer.tokenize_number(q_result))
                except ValueError:
                    if q_result in tokenizer.token_to_id:
                        result_tokens.append(q_result)
            result_tokens.append("</q_out>")

            result_ids = [tokenizer.token_to_id[t] for t in result_tokens
                          if t in tokenizer.token_to_id]
            for rid in result_ids:
                generated.append(rid)
            ids = mx.concatenate([ids, mx.array([result_ids], dtype=mx.int32)], axis=1)
            n_queries += 1

    return generated


def build_input(tokenizer: SexprTokenizer, current_expr: str,
                feedback_type: str, feedback_value: str = "",
                q_type: str = "", entities: list[str] = None) -> mx.array:
    """Build input token sequence for one refinement step."""
    tokens = []

    # Task context
    tokens.append("SEP_TASK")
    if q_type and q_type in tokenizer.token_to_id:
        tokens.append(q_type)
    for entity in (entities or []):
        if entity in tokenizer.token_to_id:
            tokens.append(entity)

    tokens.append("SEP_EXPR")
    if current_expr and current_expr != "_HOLE_":
        tokens.extend(tokenizer.tokenize_sexpr(current_expr))
    else:
        tokens.append("_HOLE_")

    tokens.append("SEP_FEEDBACK")
    tokens.append(feedback_type)
    if feedback_value:
        tokens.extend(tokenizer.tokenize_number(feedback_value))

    # Don't append SEP_EDIT — let the model generate <query>...</query> or SEP_EDIT
    ids = tokenizer.encode(tokens, add_bos=True, add_eos=False)
    return mx.array(ids, dtype=mx.int32)


def extract_edit_from_gen(gen_ids: list[int], tokenizer: SexprTokenizer) -> str:
    """Extract the edit expression from generated tokens.

    If SEP_EDIT is present, take tokens after it (the actual edit).
    Otherwise decode all tokens as the edit.
    """
    sep_edit_id = tokenizer.token_to_id.get("SEP_EDIT")
    if sep_edit_id is not None and sep_edit_id in gen_ids:
        edit_ids = gen_ids[gen_ids.index(sep_edit_id) + 1:]
    else:
        edit_ids = gen_ids
    return tokenizer.decode_to_sexpr(edit_ids)


def iterative_refine(model: MultiScaleFFNLM, tokenizer: SexprTokenizer,
                     expected_answer: str, kb: dict,
                     q_type: str = "", entities: list[str] = None,
                     max_steps: int = 5, use_queries: bool = False
                     ) -> tuple[str, int, bool]:
    """Run iterative refinement loop."""
    current_expr = "_HOLE_"

    for step in range(max_steps):
        if current_expr == "_HOLE_" or "_HOLE_" in current_expr:
            feedback_type = "INCOMPLETE"
            feedback_value = ""
        else:
            tree = parse_sexpr(current_expr)
            result = evaluate_in_kb(tree, kb)
            if result is None:
                feedback_type = "ERROR"
                feedback_value = ""
            elif str(result) == str(expected_answer):
                return current_expr, step, True
            else:
                feedback_type = "WRONG"
                feedback_value = str(result)

        input_ids = build_input(tokenizer, current_expr, feedback_type,
                                feedback_value, q_type, entities)

        if use_queries:
            gen_ids = generate_with_queries(model, input_ids, tokenizer, kb,
                                            max_new_tokens=60)
        else:
            gen_ids = generate_tokens(model, input_ids, tokenizer, max_new_tokens=40)

        gen_tokens = tokenizer.decode(gen_ids)

        if "NO_OP" in gen_tokens:
            if current_expr != "_HOLE_" and "_HOLE_" not in current_expr:
                tree = parse_sexpr(current_expr)
                result = evaluate_in_kb(tree, kb)
                correct = result is not None and str(result) == str(expected_answer)
                return current_expr, step + 1, correct
            return current_expr, step + 1, False

        new_expr = extract_edit_from_gen(gen_ids, tokenizer)
        if new_expr and new_expr != current_expr:
            current_expr = new_expr

    if current_expr != "_HOLE_" and "_HOLE_" not in current_expr:
        tree = parse_sexpr(current_expr)
        result = evaluate_in_kb(tree, kb)
        correct = result is not None and str(result) == str(expected_answer)
        return current_expr, max_steps, correct

    return current_expr, max_steps, False


def one_shot_predict(model: MultiScaleFFNLM, tokenizer: SexprTokenizer,
                     expected_answer: str, kb: dict,
                     q_type: str = "", entities: list[str] = None
                     ) -> tuple[str, bool]:
    """One-shot: predict from _HOLE_ with INCOMPLETE feedback."""
    input_ids = build_input(tokenizer, "_HOLE_", "INCOMPLETE", "",
                            q_type, entities)
    gen_ids = generate_tokens(model, input_ids, tokenizer, max_new_tokens=40)
    expr = tokenizer.decode_to_sexpr(gen_ids)

    if expr and "_HOLE_" not in expr:
        tree = parse_sexpr(expr)
        result = evaluate_in_kb(tree, kb)
        correct = result is not None and str(result) == str(expected_answer)
        return expr, correct
    return expr or "", False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DATA_DIR / "iterative_model_mlx.npz"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--feat-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--shifts", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--a-log-mults", type=float, nargs="+", default=[2.0, 1.0, 0.3])
    parser.add_argument("--weight-tie-layers", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=200)
    args = parser.parse_args()

    tokenizer = build_default_tokenizer()
    kb = json.loads((DATA_DIR / "geo_kb.json").read_text())
    data = json.loads((DATA_DIR / "geo_train.json").read_text())

    # Build and load model
    model = MultiScaleFFNLM(
        vocab_size=tokenizer.vocab_size,
        num_layers=args.num_layers,
        dim=args.dim,
        feat_dim=args.feat_dim,
        state_dim=args.state_dim,
        num_scales=args.num_scales,
        shifts=tuple(args.shifts),
        a_log_mults=tuple(args.a_log_mults),
        mlp_mult=args.mlp_mult,
        logit_softcap=30.0,
        tied_embed_init_std=0.02,
        weight_tie_layers=args.weight_tie_layers,
        group_size=64,
    )

    weights = dict(mx.load(args.model_path))
    # tree_unflatten expects list of (name, array) pairs
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    print(f"Loaded model from {args.model_path}")

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")

    # Evaluate
    results_by_depth = defaultdict(lambda: {"iter_ok": 0, "one_ok": 0, "total": 0})

    examples = data[:args.max_examples] if args.max_examples > 0 else data

    for i, ex in enumerate(examples):
        depth = ex["depth"]
        answer = ex["answer"]
        expr_str = extract_expr(ex["completion"])
        q_type, entities = extract_question_context(ex["prompt"], expr_str)

        iter_expr, iter_steps, iter_ok = iterative_refine(
            model, tokenizer, answer, kb, q_type, entities,
            max_steps=args.max_steps)
        one_expr, one_ok = one_shot_predict(model, tokenizer, answer, kb,
                                             q_type, entities)

        results_by_depth[depth]["total"] += 1
        if iter_ok:
            results_by_depth[depth]["iter_ok"] += 1
        if one_ok:
            results_by_depth[depth]["one_ok"] += 1

        if i < 10 or (depth >= 2 and i < 50):
            target = extract_expr(ex["completion"])
            print(f"  d={depth} ans={answer}")
            print(f"    target:  {target}")
            print(f"    iter:    {iter_expr} ({'OK' if iter_ok else 'WRONG'}, {iter_steps}s)")
            print(f"    oneshot: {one_expr} ({'OK' if one_ok else 'WRONG'})")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(examples)}]")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Depth':>6}  {'Total':>6}  {'Iterative':>12}  {'One-shot':>12}  {'Delta':>8}")
    print("-" * 70)
    total_iter, total_one, total_n = 0, 0, 0
    for depth in sorted(results_by_depth):
        r = results_by_depth[depth]
        n = r["total"]
        ic = r["iter_ok"]
        oc = r["one_ok"]
        total_iter += ic
        total_one += oc
        total_n += n
        print(f"{depth:>6}  {n:>6}  {ic:>5} ({ic/n*100:5.1f}%)  {oc:>5} ({oc/n*100:5.1f}%)  {ic-oc:>+5}")

    print("-" * 70)
    print(f"{'ALL':>6}  {total_n:>6}  {total_iter:>5} ({total_iter/total_n*100:5.1f}%)  "
          f"{total_one:>5} ({total_one/total_n*100:5.1f}%)  {total_iter-total_one:>+5}")


if __name__ == "__main__":
    main()
