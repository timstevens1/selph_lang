"""Rejection Sampling RL training for Neural SELPH.

Each round:
1. Sample N completions per prompt (temperature > 0)
2. Score with reward function (grounding + correctness)
3. Keep positive-reward completions as training data
4. Fine-tune with LoRA on the filtered data
5. Repeat

This is expert iteration / STaR / ReST — no PPO needed.
"""
import json
import re
import argparse
import time
import subprocess
import sys
from pathlib import Path
from collections import Counter

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from reward import compute_reward, extract_prompt_strings

SCRIPT_DIR = Path(__file__).parent


def build_eval_fn(kb_path: Path):
    """Build a SELPH evaluation function from a KB."""
    kb = json.loads(kb_path.read_text())

    capital_to_country = {}
    for name, entry in kb.items():
        if "capital" in entry:
            capital_to_country[entry["capital"]] = name

    continent_countries = {}
    for name, entry in kb.items():
        if "continent" in entry and "population" in entry:
            if entry["continent"] not in continent_countries:
                continent_countries[entry["continent"]] = []
            continent_countries[entry["continent"]].append((name, entry["population"]))
    largest_by_continent = {}
    for cont, pairs in continent_countries.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        largest_by_continent[cont] = pairs[0][0]

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

    def eval_sexpr(sexpr):
        tokens = tokenize(sexpr)
        pos = [0]
        def parse():
            if pos[0] >= len(tokens): return None
            tok = tokens[pos[0]]
            if tok == '(':
                pos[0] += 1; op = tokens[pos[0]]; pos[0] += 1
                args = []
                while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                    args.append(parse())
                pos[0] += 1; return (op, args)
            else:
                pos[0] += 1
                if tok.startswith('"') and tok.endswith('"'): return tok[1:-1]
                try: return float(tok)
                except: return tok

        def ev(tree):
            if isinstance(tree, (int, float, str)): return tree
            if tree is None: return None
            op, args = tree
            vals = [ev(a) for a in args]
            if any(v is None for v in vals): return None
            if op == "capital": return kb.get(str(vals[0]), {}).get("capital")
            elif op == "population":
                e = kb.get(str(vals[0]))
                if e and "population" in e: return e["population"]
                for n, e2 in kb.items():
                    if e2.get("capital") == str(vals[0]) and "capital_population" in e2:
                        return e2["capital_population"]
                return None
            elif op == "continent": return kb.get(str(vals[0]), {}).get("continent")
            elif op == "currency": return kb.get(str(vals[0]), {}).get("currency")
            elif op == "language": return kb.get(str(vals[0]), {}).get("language")
            elif op == "area": return kb.get(str(vals[0]), {}).get("area_km2")
            elif op == "head-of-state": return kb.get(str(vals[0]), {}).get("head_of_state")
            elif op == "country-of": return capital_to_country.get(str(vals[0]))
            elif op == "largest-by-population": return largest_by_continent.get(str(vals[0]))
            elif op == "divide":
                return vals[0] / vals[1] if vals[1] and vals[1] != 0 else None
            return None
        return ev(parse())

    return eval_sexpr


def extract_tc(text):
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def sample_completions(model, tokenizer, prompt, n_samples, max_tokens, temp):
    """Generate N completions for a prompt with temperature sampling."""
    completions = []
    full_prompt = prompt + " <tool_call>"
    sampler = make_sampler(temp=temp)
    for _ in range(n_samples):
        resp = generate(
            model, tokenizer,
            prompt=full_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        full = "<tool_call>" + resp
        sexpr = extract_tc(full)
        completions.append(sexpr)
    return completions


def run_rl_round(
    model, tokenizer, prompts, expected_answers, eval_fn,
    n_samples=8, max_tokens=150, temp=0.7,
):
    """Run one round of rejection sampling.

    Returns accepted (prompt, completion) pairs for fine-tuning.
    """
    accepted = []
    stats = Counter()

    for i, (prompt, expected) in enumerate(zip(prompts, expected_answers)):
        completions = sample_completions(
            model, tokenizer, prompt, n_samples, max_tokens, temp
        )

        best_reward = 0.0
        best_sexpr = None
        best_info = None

        for sexpr in completions:
            reward, info = compute_reward(sexpr, prompt, expected, eval_fn)
            if reward > best_reward:
                best_reward = reward
                best_sexpr = sexpr
                best_info = info

        stats["total"] += 1
        if best_sexpr is not None and best_reward > 0:
            stats["accepted"] += 1
            accepted.append({
                "prompt": prompt + " <tool_call>",
                "completion": f"{best_sexpr}</tool_call>",
            })
            if best_info.get("correct"):
                stats["correct"] += 1
        else:
            stats["rejected"] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(prompts)}: "
                  f"{stats['accepted']} accepted, {stats['rejected']} rejected")

    return accepted, stats


def main():
    parser = argparse.ArgumentParser(description="RL training for Neural SELPH")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--adapter-path", default=str(SCRIPT_DIR / "adapters_geo_1k"),
                        help="Starting adapter checkpoint")
    parser.add_argument("--kb", default=str(SCRIPT_DIR / "data" / "geo_kb.json"))
    parser.add_argument("--train-data", default=str(SCRIPT_DIR / "data" / "geo_train.json"))
    parser.add_argument("--rounds", type=int, default=3, help="Number of RL rounds")
    parser.add_argument("--n-samples", type=int, default=8, help="Completions per prompt")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--lora-iters", type=int, default=200,
                        help="LoRA fine-tuning iters per round")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    # Load KB and eval function
    eval_fn = build_eval_fn(Path(args.kb))

    # Load training prompts + expected answers
    geo_data = json.loads(Path(args.train_data).read_text())
    prompts = [ex["prompt"] for ex in geo_data]
    expected_answers = []
    for ex in geo_data:
        ans = ex["answer"]
        try:
            ans = int(ans)
        except (ValueError, TypeError):
            try:
                ans = float(ans)
            except (ValueError, TypeError):
                pass
        expected_answers.append(ans)

    print(f"=== Neural SELPH RL Training ===")
    print(f"Model: {args.model}")
    print(f"Starting adapter: {args.adapter_path}")
    print(f"Prompts: {len(prompts)}")
    print(f"Rounds: {args.rounds}")
    print(f"Samples/prompt: {args.n_samples}")
    print(f"Temperature: {args.temp}")
    print()

    current_adapter = args.adapter_path

    for round_idx in range(args.rounds):
        print(f"\n{'='*60}")
        print(f"=== Round {round_idx + 1}/{args.rounds} ===")
        print(f"{'='*60}")

        # Load model with current adapter
        print(f"Loading model with adapter: {current_adapter}")
        model, tokenizer = load(args.model, adapter_path=current_adapter)

        # Sample and filter
        print(f"Sampling {args.n_samples} completions per prompt...")
        t0 = time.time()
        accepted, stats = run_rl_round(
            model, tokenizer, prompts, expected_answers, eval_fn,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            temp=args.temp,
        )
        dt = time.time() - t0

        print(f"\nRound {round_idx + 1} sampling done in {dt:.1f}s:")
        print(f"  Total prompts: {stats['total']}")
        print(f"  Accepted: {stats['accepted']} ({100*stats['accepted']/stats['total']:.1f}%)")
        print(f"  Correct: {stats['correct']} ({100*stats['correct']/stats['total']:.1f}%)")

        if not accepted:
            print("  No accepted completions — stopping.")
            break

        # Write accepted completions as training data
        rl_data_dir = SCRIPT_DIR / f"training_data_rl_round{round_idx + 1}"
        rl_data_dir.mkdir(exist_ok=True)

        # Split 90/10 train/valid
        import random
        random.seed(42 + round_idx)
        random.shuffle(accepted)
        split = max(1, int(len(accepted) * 0.9))
        train_data = accepted[:split]
        valid_data = accepted[split:]

        for name, data in [("train", train_data), ("valid", valid_data)]:
            with open(rl_data_dir / f"{name}.jsonl", "w") as f:
                for ex in data:
                    f.write(json.dumps(ex) + "\n")

        # Also copy test.jsonl from geo for eval continuity
        test_src = SCRIPT_DIR / "training_data_geo" / "test.jsonl"
        if test_src.exists():
            import shutil
            shutil.copy(test_src, rl_data_dir / "test.jsonl")

        print(f"  Training data: {len(train_data)} train, {len(valid_data)} valid")
        print(f"  Saved to: {rl_data_dir}")

        # Fine-tune with LoRA
        next_adapter = str(SCRIPT_DIR / f"adapters_rl_round{round_idx + 1}")
        print(f"\nFine-tuning for {args.lora_iters} iterations...")

        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", args.model,
            "--data", str(rl_data_dir),
            "--train",
            "--fine-tune-type", "lora",
            "--mask-prompt",
            "--batch-size", str(args.batch_size),
            "--iters", str(args.lora_iters),
            "--learning-rate", str(args.lr),
            "--num-layers", "16",
            "--adapter-path", next_adapter,
            "--resume-adapter-file", f"{current_adapter}/adapters.safetensors",
            "--steps-per-report", "50",
            "--steps-per-eval", "100",
            "--max-seq-length", "512",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Training failed!")
            print(result.stderr[-500:])
            break

        # Extract final loss from output
        for line in result.stdout.split('\n'):
            if 'Val loss' in line:
                last_val = line
        print(f"  {last_val.strip()}")

        current_adapter = next_adapter
        print(f"  New adapter: {current_adapter}")

        # Free the model for next round
        del model, tokenizer

    print(f"\n{'='*60}")
    print(f"=== RL Training Complete ===")
    print(f"Final adapter: {current_adapter}")
    print(f"Run evaluation with:")
    print(f"  python evaluate.py --adapter-path {current_adapter}")


if __name__ == "__main__":
    main()
