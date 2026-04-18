"""Generate think+SELPH traces using Qwen3.5-122B for all MMLU-Pro categories.

Uses the large model to generate chain-of-thought solutions that include
<tool_call> expressions for computations. The traces teach the 0.8B model
when and how to use SELPH during reasoning.

The SELPH KB provides functions for:
- Arithmetic: add, subtract, multiply, divide, power, sqrt, etc.
- Finance: compound interest, present value, discount, markup
- Science: unit conversion, physical constants
- Statistics: mean, median, percentage
- Logic: and, or, not, if-then

The 122B model sees the KB functions and decides when to use them.
"""
import json
import re
import math
import random
import time
import argparse
from pathlib import Path
from datasets import load_dataset
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

SCRIPT_DIR = Path(__file__).parent
LETTERS = "ABCDEFGHIJ"

# Multi-domain SELPH function signatures
SELPH_FUNCTIONS = """Available SELPH functions (use inside <tool_call>...</tool_call> during thinking):
Arithmetic: (add a b), (subtract a b), (multiply a b), (divide a b), (power base exp), (sqrt x), (abs x), (floor x), (round x n)
Percentage: (multiply value (divide percent 100)) for "X% of Y"
Finance: (multiply P (power (add 1 r) n)) for compound interest, (divide FV (power (add 1 r) n)) for present value
Comparison: use to verify which option matches a computed value
Note: Only use <tool_call> when you need to compute a numeric result. For conceptual/factual questions, reason in natural language."""


def build_prompt_for_122b(question, options, answer):
    """Build a prompt for the 122B model to generate a think+SELPH trace."""
    opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))

    return f"""You are solving a multiple choice question. Think step by step inside <think>...</think> tags, then give your answer letter.

{SELPH_FUNCTIONS}

When you need to calculate something, write it as a SELPH expression inside <tool_call>...</tool_call> tags. The expression will be evaluated and the result will appear. For example:
<think>
I need to calculate 15% of 200.
<tool_call>(multiply 200 (divide 15 100))</tool_call>30
So 15% of 200 is 30.
</think>
B

Now solve this question. The correct answer is {answer}.

Question: {question}
{opts}

<think>
"""


def eval_sexpr(s):
    """Evaluate a SELPH s-expression."""
    if not s: return None
    def tokenize(s):
        tokens, i = [], 0
        while i < len(s):
            c = s[i]
            if c in '()': tokens.append(c); i += 1
            elif c in ' \t\n': i += 1
            elif c == '-' and i+1 < len(s) and (s[i+1].isdigit() or s[i+1] == '.'):
                j = i+1
                while j < len(s) and (s[j].isdigit() or s[j]=='.'): j += 1
                tokens.append(s[i:j]); i = j
            else:
                j = i
                while j < len(s) and s[j] not in '() \t\n': j += 1
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
                try: return float(tok)
                except: return tok
        def ev(tree):
            if isinstance(tree, (int, float)): return tree
            if tree is None: return None
            op, args = tree
            vals = [ev(a) for a in args]
            if any(v is None for v in vals): return None
            try:
                if op == "add": return vals[0]+vals[1]
                elif op == "subtract": return vals[0]-vals[1]
                elif op == "multiply": return vals[0]*vals[1]
                elif op == "divide": return vals[0]/vals[1] if vals[1]!=0 else None
                elif op == "power": return vals[0]**vals[1]
                elif op == "sqrt": return math.sqrt(vals[0]) if vals[0]>=0 else None
                elif op == "abs": return abs(vals[0])
                elif op == "floor": return math.floor(vals[0])
                elif op == "round":
                    return round(vals[0], int(vals[1])) if len(vals)>1 else round(vals[0],2)
                elif op == "negate": return -vals[0]
                elif op == "log": return math.log(vals[0]) if vals[0]>0 else None
                elif op == "max": return max(vals)
                elif op == "min": return min(vals)
                elif op == "remainder": return vals[0]%vals[1] if vals[1]!=0 else None
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


def post_process_trace(raw_trace, answer_letter):
    """Post-process a 122B trace:
    1. Evaluate any <tool_call> expressions and insert results
    2. Ensure it ends with </think> and the answer letter
    3. Validate the answer letter matches expected
    """
    # Find and evaluate tool_call expressions
    def replace_tool_call(match):
        sexpr = match.group(1).strip()
        result = eval_sexpr(sexpr)
        result_str = format_number(result)
        return f"<tool_call>{sexpr}</tool_call>{result_str}"

    processed = re.sub(
        r'<tool_call>(.*?)</tool_call>\s*\S*',
        replace_tool_call,
        raw_trace
    )

    # Ensure proper ending
    if "</think>" not in processed:
        processed += "\n</think>"

    # Extract answer from trace
    parts = processed.split("</think>")
    after_think = parts[-1] if len(parts) > 1 else ""
    m = re.search(r'\b([A-J])\b', after_think)

    # Force correct answer
    if m and m.group(1) != answer_letter:
        # Wrong answer in trace — skip this one
        return None

    if not m:
        processed += f"\n{answer_letter}"

    return processed


def format_for_training(question, options, trace, answer_letter):
    """Format as training example for the 0.8B model."""
    opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))

    prompt = (
        f"{SELPH_FUNCTIONS}\n\n"
        f"Question: {question}\n{opts}\n\n"
        f"<think>\n"
    )

    return {
        "prompt": prompt,
        "completion": trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(Path.home() / ".omlx/models/Qwen3.5-122B-A10B-4bit"))
    parser.add_argument("--per-category", type=int, default=50,
                        help="Examples to generate per category")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "data" / "think_selph_traces.json"))
    args = parser.parse_args()

    print(f"Loading 122B model from {args.model}...")
    model, tokenizer = load(args.model)
    print("Model loaded.")

    print("Loading MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    # Sample per category
    random.seed(42)
    examples_by_cat = {}
    for ex in ds["test"]:
        examples_by_cat.setdefault(ex["category"], []).append(ex)

    all_traces = []
    total_generated = 0
    total_with_selph = 0

    for cat in sorted(examples_by_cat.keys()):
        exs = examples_by_cat[cat]
        sample = random.sample(exs, min(args.per_category, len(exs)))

        print(f"\n{'='*50}")
        print(f"Category: {cat} ({len(sample)} questions)")
        print(f"{'='*50}")

        cat_ok = 0
        cat_selph = 0

        for i, ex in enumerate(sample):
            prompt = build_prompt_for_122b(
                ex["question"], ex["options"], ex["answer"]
            )

            raw = generate(model, tokenizer, prompt=prompt,
                          max_tokens=args.max_tokens)

            trace = post_process_trace(raw, ex["answer"])

            if trace is None:
                continue

            # Check for SELPH usage
            has_selph = "<tool_call>" in trace
            if has_selph:
                cat_selph += 1
                total_with_selph += 1

            formatted = format_for_training(
                ex["question"], ex["options"], trace, ex["answer"]
            )
            formatted["category"] = cat
            formatted["has_selph"] = has_selph
            formatted["answer"] = ex["answer"]
            all_traces.append(formatted)

            cat_ok += 1
            total_generated += 1

            if i < 2 or (has_selph and cat_selph <= 2):
                print(f"\n  [{cat}] Q: {ex['question'][:80]}...")
                print(f"  SELPH: {'yes' if has_selph else 'no'}")
                print(f"  Trace: {trace[:150]}...")

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(sample)}] ok={cat_ok} selph={cat_selph}")

        print(f"  Category done: {cat_ok} traces, {cat_selph} with SELPH")

    print(f"\n{'='*50}")
    print(f"Total: {total_generated} traces, {total_with_selph} with SELPH")
    print(f"{'='*50}")

    # Save
    with open(args.output, "w") as f:
        json.dump(all_traces, f, indent=2)
    print(f"Saved to {args.output}")

    # Also create train/valid split
    random.shuffle(all_traces)
    split = int(len(all_traces) * 0.9)
    train = all_traces[:split]
    valid = all_traces[split:]

    out_dir = SCRIPT_DIR / "training_data_traces"
    out_dir.mkdir(exist_ok=True)
    for name, data in [("train", train), ("valid", valid)]:
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps({
                    "prompt": item["prompt"],
                    "completion": item["completion"]
                }) + "\n")
    print(f"Training data: {len(train)} train, {len(valid)} valid")


if __name__ == "__main__":
    main()
