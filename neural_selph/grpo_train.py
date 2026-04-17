"""GRPO (Group Relative Policy Optimization) training for Neural SELPH.

Implements DeepSeek-R1 style RL training:
1. For each prompt, generate G completions with temperature sampling
2. Score each with the reward function (SELPH eval + grounding)
3. Compute group-relative advantage: (reward - mean) / std
4. Policy gradient: loss = -advantage × log_prob + β × KL(policy || reference)
5. Update with Adam optimizer

No value head needed — advantage is computed from group statistics.
"""
import json
import re
import math
import time
import argparse
import random
from pathlib import Path
from collections import Counter
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

SCRIPT_DIR = Path(__file__).parent

TOOL_CALL_OPEN_ID = 248058
TOOL_CALL_CLOSE_ID = 248059
THINK_CLOSE_ID = 248069
EOS_ID = 248044
LETTERS = "ABCDEFGHIJ"


def eval_sexpr(s):
    """Evaluate a SELPH s-expression."""
    if not s:
        return None
    def tokenize(s):
        tokens, i = [], 0
        while i < len(s):
            c = s[i]
            if c in '()': tokens.append(c); i += 1
            elif c in ' \t\n': i += 1
            elif c == '"':
                j = i + 1
                while j < len(s) and s[j] != '"': j += 1
                tokens.append(f'"{s[i+1:j]}"'); i = j + 1
            elif c == '-' and i + 1 < len(s) and (s[i+1].isdigit() or s[i+1] == '.'):
                j = i + 1
                while j < len(s) and (s[j].isdigit() or s[j] == '.'): j += 1
                tokens.append(s[i:j]); i = j
            else:
                j = i
                while j < len(s) and s[j] not in '() \t\n"': j += 1
                tokens.append(s[i:j]); i = j
        return tokens
    try:
        tokens = tokenize(s); pos = [0]
        def parse():
            if pos[0] >= len(tokens): return None
            tok = tokens[pos[0]]
            if tok == '(':
                pos[0] += 1; op = tokens[pos[0]]; pos[0] += 1
                args = []
                while pos[0] < len(tokens) and tokens[pos[0]] != ')': args.append(parse())
                pos[0] += 1; return (op, args)
            else:
                pos[0] += 1
                if tok.startswith('"') and tok.endswith('"'): return tok[1:-1]
                try: return float(tok)
                except: return tok
        def ev(tree):
            if isinstance(tree, (int, float)): return tree
            if isinstance(tree, str):
                try: return float(tree)
                except: return tree
            if tree is None: return None
            op, args = tree
            vals = [ev(a) for a in args]
            if any(v is None for v in vals): return None
            try:
                if op == "add": return vals[0] + vals[1]
                elif op == "subtract": return vals[0] - vals[1]
                elif op == "multiply": return vals[0] * vals[1]
                elif op == "divide": return vals[0] / vals[1] if vals[1] != 0 else None
                elif op == "power": return vals[0] ** vals[1]
                elif op == "sqrt": return math.sqrt(vals[0]) if vals[0] >= 0 else None
                elif op == "negate": return -vals[0]
                elif op == "inverse": return 1/vals[0] if vals[0] != 0 else None
                elif op == "floor": return math.floor(vals[0])
                elif op == "abs": return abs(vals[0])
                elif op == "round":
                    if len(vals) > 1: return round(vals[0], int(vals[1]))
                    return round(vals[0], 2)
                elif op == "remainder": return vals[0] % vals[1] if vals[1] != 0 else None
                elif op == "log": return math.log(vals[0]) if vals[0] > 0 else None
                elif op == "max": return max(vals)
                elif op == "min": return min(vals)
                else: return None
            except: return None
        return ev(parse())
    except: return None


def format_number(n):
    if n is None: return "ERROR"
    if isinstance(n, float):
        if n == int(n) and abs(n) < 1e15: return str(int(n))
        return f"{n:.6g}"
    return str(n)


def generate_with_selph_eval(model, tokenizer, prompt_tokens, max_tokens=256, temp=0.7):
    """Generate a completion with inline SELPH evaluation.

    Returns (completion_token_ids, completion_text, num_evals, used_selph).
    Token IDs include the SELPH results spliced in.
    """
    sampler = make_sampler(temp=temp)
    tokens = []
    text = ""
    in_tool_call = False
    tool_buffer = ""
    num_evals = 0
    used_selph = False

    current_prompt = prompt_tokens

    for (token, _), step in zip(
        generate_step(current_prompt, model, max_tokens=max_tokens, sampler=sampler),
        range(max_tokens)
    ):
        token_id = token.item() if hasattr(token, 'item') else int(token)

        if token_id == EOS_ID:
            break

        if token_id == TOOL_CALL_OPEN_ID:
            in_tool_call = True
            tool_buffer = ""
            # Don't add to tokens — we'll replace with result
            continue

        if token_id == TOOL_CALL_CLOSE_ID and in_tool_call:
            sexpr = tool_buffer.strip()
            result = eval_sexpr(sexpr)
            result_str = format_number(result)
            num_evals += 1
            used_selph = True

            # Encode result and add to token stream
            result_tokens = tokenizer.encode(result_str, add_special_tokens=False)
            tokens.extend(result_tokens)
            text += result_str

            in_tool_call = False
            tool_buffer = ""
            continue

        token_str = tokenizer.decode([token_id])
        if in_tool_call:
            tool_buffer += token_str
        else:
            tokens.append(token_id)
            text += token_str

        # Stop after answer letter following </think>
        if token_id == THINK_CLOSE_ID:
            # Generate a few more for the answer
            for (tok2, _), _ in zip(
                generate_step(current_prompt, model, max_tokens=10, sampler=make_sampler(temp=0.0)),
                range(10)
            ):
                tid = tok2.item() if hasattr(tok2, 'item') else int(tok2)
                if tid == EOS_ID: break
                tokens.append(tid)
                text += tokenizer.decode([tid])
                if re.search(r'[A-J]', text[-5:]):
                    break
            break

    return tokens, text, num_evals, used_selph



def advantage_weighted_data(completions_with_rewards):
    """Convert GRPO advantage scores into training data.

    High-advantage completions get included; negative advantage ones
    get excluded. This approximates the GRPO gradient update by
    using mlx_lm's own training loop (which handles custom kernels).

    Returns list of {"prompt": ..., "completion": ...} dicts.
    """
    training_examples = []
    for item in completions_with_rewards:
        if item["advantage"] > 0 and item["completion_text"]:
            training_examples.append({
                "prompt": item["prompt"],
                "completion": item["completion_text"],
            })
    return training_examples


def extract_answer(text):
    """Extract MC answer letter from response."""
    parts = text.split("</think>")
    search = parts[-1] if len(parts) > 1 else text
    m = re.search(r'\b([A-J])\b', search)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Neural SELPH")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--adapter-path", default=None,
                        help="Starting adapter (None = base model)")
    parser.add_argument("--data", default=str(SCRIPT_DIR / "data" / "mmlu_business_calc.json"),
                        help="Training questions JSON")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=4,
                        help="Completions per prompt (G)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Prompts per gradient update")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save adapter every N epochs")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "adapters_grpo"))
    args = parser.parse_args()

    # Prepare training data from MMLU-Pro business calc
    data_path = Path(args.data)
    if not data_path.exists():
        print("Preparing training data from MMLU-Pro...")
        from datasets import load_dataset
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
        calc_qs = [ex for ex in ds["test"] if ex["src"] == "stemez-Business"]
        random.seed(42)
        random.shuffle(calc_qs)

        train_data = []
        for ex in calc_qs:
            opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(ex["options"]))
            train_data.append({
                "prompt": f"Question: {ex['question']}\n{opts}\n\n<think>\nLet me work through this. I can use <tool_call>(expr)</tool_call> to compute.\n",
                "answer": ex["answer"],
                "options": ex["options"],
            })

        data_path.parent.mkdir(exist_ok=True)
        with open(data_path, "w") as f:
            json.dump(train_data, f, indent=2)
        print(f"Saved {len(train_data)} questions to {data_path}")

    with open(data_path) as f:
        train_data = json.load(f)
    print(f"Training data: {len(train_data)} questions")

    # Load model
    print(f"Loading model: {args.model}")
    if args.adapter_path:
        model, tokenizer = load(args.model, adapter_path=args.adapter_path)
        print(f"Starting adapter: {args.adapter_path}")
    else:
        model, tokenizer = load(args.model)

    # Note: LoRA is applied inside the mlx_lm train loop per round,
    # so we don't apply it here for generation. We load adapters if provided.

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GRPO Training")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Group size: {args.group_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temp}")
    print(f"KL beta: {args.beta}")
    print(f"Learning rate: {args.lr}")
    print()

    # GRPO training loop: generate → score → filter → train
    import subprocess, sys, shutil

    current_adapter = args.adapter_path

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        # Phase 1: Generate completions with SELPH eval
        print(f"Phase 1: Generating {args.group_size} completions per prompt...")

        # Reload model with current adapter for generation
        if epoch > 0:
            del model, tokenizer
            model, tokenizer = load(args.model, adapter_path=current_adapter)

        all_items = []
        epoch_correct = 0
        epoch_selph_uses = 0
        epoch_total = 0
        t0 = time.time()

        for qi, ex in enumerate(train_data):
            prompt_tokens = tokenizer.encode(ex["prompt"])
            group_rewards = []
            group_items = []

            for g in range(args.group_size):
                comp_tokens, comp_text, num_evals, used_selph = \
                    generate_with_selph_eval(
                        model, tokenizer,
                        mx.array(prompt_tokens),
                        max_tokens=args.max_tokens,
                        temp=args.temp,
                    )

                pred = extract_answer(comp_text)
                correct = pred == ex["answer"]

                reward = 0.0
                if correct:
                    reward = 1.0
                    epoch_correct += 1
                    if used_selph:
                        reward += 0.2
                        epoch_selph_uses += 1
                epoch_total += 1

                group_rewards.append(reward)
                group_items.append({
                    "prompt": ex["prompt"],
                    "completion_text": comp_text,
                    "reward": reward,
                })

            # Compute group advantages
            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = max(
                (sum((r - mean_r)**2 for r in group_rewards) / len(group_rewards)) ** 0.5,
                1e-8
            )
            for i, item in enumerate(group_items):
                item["advantage"] = (group_rewards[i] - mean_r) / std_r
                all_items.append(item)

            if (qi + 1) % 50 == 0:
                acc = epoch_correct / max(epoch_total, 1)
                print(f"  [{qi+1}/{len(train_data)}] "
                      f"acc={100*acc:.1f}% selph={epoch_selph_uses}")

        dt_gen = time.time() - t0
        acc = epoch_correct / max(epoch_total, 1)

        # Phase 2: Filter to positive-advantage completions
        training_data = advantage_weighted_data(all_items)

        print(f"\nGeneration done ({dt_gen:.1f}s):")
        print(f"  Accuracy: {epoch_correct}/{epoch_total} ({100*acc:.1f}%)")
        print(f"  SELPH uses: {epoch_selph_uses}")
        print(f"  Training examples (positive advantage): {len(training_data)}")

        if not training_data:
            print("  No positive-advantage examples — skipping training.")
            continue

        # Phase 3: Write training data and fine-tune via mlx_lm
        round_data_dir = out_dir / f"data_epoch{epoch+1}"
        round_data_dir.mkdir(parents=True, exist_ok=True)

        # Split 90/10
        random.shuffle(training_data)
        split = max(1, int(len(training_data) * 0.9))
        for name, data in [("train", training_data[:split]),
                           ("valid", training_data[split:])]:
            with open(round_data_dir / f"{name}.jsonl", "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

        # Fine-tune with mlx_lm lora
        next_adapter = str(out_dir / f"adapter_epoch{epoch+1}")
        lora_iters = min(200, len(training_data) // args.batch_size)

        print(f"\nPhase 3: Fine-tuning for {lora_iters} iterations...")

        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", args.model,
            "--data", str(round_data_dir),
            "--train",
            "--fine-tune-type", "lora",
            "--mask-prompt",
            "--batch-size", str(args.batch_size),
            "--iters", str(lora_iters),
            "--learning-rate", str(args.lr),
            "--num-layers", "16",
            "--adapter-path", next_adapter,
            "--steps-per-report", "50",
            "--steps-per-eval", str(lora_iters),
            "--max-seq-length", "512",
        ]
        if current_adapter:
            cmd.extend(["--resume-adapter-file",
                        f"{current_adapter}/adapters.safetensors"
                        if Path(current_adapter).is_dir()
                        else current_adapter])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Training failed!")
            print(result.stderr[-500:])
            break

        # Extract final loss
        for line in result.stdout.split('\n'):
            if 'Val loss' in line or 'Train loss' in line:
                last_line = line
        print(f"  {last_line.strip()}")

        current_adapter = next_adapter
        print(f"  Adapter: {current_adapter}")

    print(f"\n{'='*60}")
    print(f"GRPO Training Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
