"""Eval RefinementLM v2: separate latent model + token model.

Latent model runs once per refinement step (numpy round-tripped).
Token model runs per-token, all generation inlined.
Output order: z (scratchpad) first, then y (answer).

Usage:
    python3 -m neural_selph.iterative.eval_v2
    python3 -m neural_selph.iterative.eval_v2 --max-examples 50
"""
import sys, os, json, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/projects/parameter-golf"))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from neural_selph.iterative.model_v2 import RefinementLM
from neural_selph.iterative.sexpr_tokenizer import build_default_tokenizer
from neural_selph.iterative.generate_traces import (
    evaluate_in_kb, parse_sexpr, extract_expr, extract_question_context,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def build_input(tok, expr, fb_type, fb_val="",
                q_type="", entities=None,
                query_expr="NO_OP", query_result=""):
    tokens = []
    tokens.append("SEP_TASK")
    if q_type and q_type in tok.token_to_id:
        tokens.append(q_type)
    for e in (entities or []):
        if e in tok.token_to_id:
            tokens.append(e)
    tokens.append("SEP_EXPR")
    if expr and expr != "_HOLE_":
        tokens.extend(tok.tokenize_sexpr(expr))
    else:
        tokens.append("_HOLE_")
    tokens.append("SEP_FEEDBACK")
    tokens.append(fb_type)
    if fb_val:
        tokens.extend(tok.tokenize_number(fb_val))
    tokens.append("SEP_QUERY")
    if query_expr and query_expr != "NO_OP":
        tokens.extend(tok.tokenize_sexpr(query_expr))
    else:
        tokens.append("NO_OP")
    tokens.append("SEP_QUERY_RESULT")
    if query_result:
        try:
            float(query_result)
            tokens.extend(tok.tokenize_number(query_result))
        except ValueError:
            if query_result in tok.token_to_id:
                tokens.append(query_result)
    # Generation starts after SEP_QUERY_OUT (z first, then SEP_EDIT y)
    tokens.append("SEP_QUERY_OUT")
    return tok.encode(tokens, add_bos=True, add_eos=False)


def decode_z_then_y(tok, gen_ids):
    """Split generated IDs: z tokens (before SEP_EDIT), y tokens (after)."""
    sep_edit_id = tok.token_to_id.get("SEP_EDIT")
    if sep_edit_id is not None and sep_edit_id in gen_ids:
        split_pos = gen_ids.index(sep_edit_id)
        z_ids = gen_ids[:split_pos]
        y_ids = gen_ids[split_pos + 1:]
    else:
        # No SEP_EDIT found — treat all as y
        z_ids = []
        y_ids = gen_ids

    y_expr = tok.decode_to_sexpr(y_ids)

    z_tokens = tok.decode(z_ids)
    if not z_tokens or z_tokens == ["NO_OP"]:
        z_expr = "NO_OP"
    else:
        z_expr = tok.decode_to_sexpr(z_ids)

    return y_expr, z_expr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--no-query", action="store_true",
                        help="Disable query channel — no z in input/output, isolates latent benefit")
    parser.add_argument("--data-path", default="/tmp/geo_eval_sample.json")
    args = parser.parse_args()

    tokenizer = build_default_tokenizer()
    kb = json.loads((DATA_DIR / "geo_kb.json").read_text())
    data = json.load(open(args.data_path))

    model = RefinementLM(vocab_size=tokenizer.vocab_size)
    weights = dict(mx.load(str(DATA_DIR / "refinement_v2_mlx.npz")))
    model.load_weights(list(weights.items()))
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"RefinementLM v2: {n_params:,} params", flush=True)

    if args.max_examples > 0:
        data = data[:args.max_examples]

    results = defaultdict(lambda: {"iter_ok": 0, "total": 0, "steps": []})

    t_start = time.time()
    for i, ex in enumerate(data):

        mx.eval(model.parameters())
        d = ex["depth"]
        ans = ex["answer"]
        expr_str = extract_expr(ex["completion"])
        qt, ents = extract_question_context(ex["prompt"], expr_str)

        expr = "_HOLE_"
        query_expr = "NO_OP"
        query_result = ""
        latent = mx.zeros((1, model.latent_dim), dtype=mx.float32)
        iter_ok = False
        iter_steps = args.max_steps

        for step in range(args.max_steps):
            # Feedback
            if expr == "_HOLE_" or "_HOLE_" in expr:
                fb, fv = "INCOMPLETE", ""
            else:
                tree = parse_sexpr(expr)
                result = evaluate_in_kb(tree, kb)
                if result is None:
                    fb, fv = "ERROR", ""
                elif str(result) == str(ans):
                    iter_ok = True
                    iter_steps = step
                    break
                else:
                    fb, fv = "WRONG", str(result)

            if args.no_query:
                inp_list = build_input(tokenizer, expr, fb, fv, qt, ents,
                                       "NO_OP", "")
            else:
                inp_list = build_input(tokenizer, expr, fb, fv, qt, ents,
                                       query_expr, query_result)

            # --- Latent model: __call__ + numpy round-trip ---
            ids = mx.array([inp_list], dtype=mx.int32)
            latent = model.latent_model(ids, latent)
            latent = mx.array(np.array(latent))
            # --- Token model: generate ---
            gen_ids = model.token_model.generate(
                ids, latent, max_new=args.max_new_tokens,
                eos_id=tokenizer.eos_id)
            del ids

            # Decode: z first, then y
            new_expr, new_query = decode_z_then_y(tokenizer, gen_ids)

            if new_expr and new_expr != expr:
                expr = new_expr

            # Evaluate query (skip if --no-query)
            if not args.no_query and new_query and new_query != "NO_OP" and "_HOLE_" not in new_query:
                q_tree = parse_sexpr(new_query)
                q_result = evaluate_in_kb(q_tree, kb)
                query_expr = new_query
                query_result = str(q_result) if q_result is not None else ""
            else:
                query_expr = "NO_OP"
                query_result = ""
            mx.eval()

        # Final check
        if not iter_ok and expr != "_HOLE_" and "_HOLE_" not in expr:
            tree = parse_sexpr(expr)
            result = evaluate_in_kb(tree, kb)
            iter_ok = result is not None and str(result) == str(ans)

        results[d]["total"] += 1
        if iter_ok:
            results[d]["iter_ok"] += 1
            results[d]["steps"].append(iter_steps)


        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"[{i + 1}/{len(data)}] {elapsed:.1f}s")
        mx.eval()
        mx.clear_cache()

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")

    hdr = f"{'Depth':>6}  {'Total':>6}  {'Iterative':>12}  {'AvgSteps':>8}"
    print(hdr)
    print("-" * len(hdr))
    ti = tn = 0
    for d in sorted(results):
        r = results[d]
        n_d, ic = r["total"], r["iter_ok"]
        ti += ic
        tn += n_d
        avg = sum(r["steps"]) / len(r["steps"]) if r["steps"] else 0
        print(f"{d:>6}  {n_d:>6}  {ic:>5} ({ic / n_d * 100:5.1f}%)  {avg:>7.1f}")
    print("-" * len(hdr))
    all_s = [s for r in results.values() for s in r["steps"]]
    avg_a = sum(all_s) / len(all_s) if all_s else 0
    print(f"{'ALL':>6}  {tn:>6}  {ti:>5} ({ti / tn * 100:5.1f}%)  {avg_a:>7.1f}")


if __name__ == "__main__":
    main()
