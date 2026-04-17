"""Evaluate on MMLU-Pro benchmark.

Supports:
1. Base model (no adapter) — baseline
2. SELPH-augmented model — with adapter + inline s-expression evaluation

For the SELPH-augmented version, the model generates inline <tool_call>
expressions which are evaluated against a KB before scoring.

Standard MMLU-Pro evaluation: 10-option multiple choice, report accuracy.
"""
import json
import re
import argparse
import time
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from mlx_lm import load, generate

SCRIPT_DIR = Path(__file__).parent

ANSWER_LETTERS = "ABCDEFGHIJ"


def format_prompt(question, options):
    """Format an MMLU-Pro question as a prompt."""
    opts = "\n".join(f"{ANSWER_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return f"Question: {question}\n{opts}\nAnswer:"


def extract_answer(response):
    """Extract the answer letter from model response."""
    response = response.strip()
    # Try to find a single letter answer
    # Common patterns: "A", "The answer is A", "(A)", "A."
    m = re.search(r'\b([A-J])\b', response)
    if m:
        return m.group(1)
    return None


def evaluate_baseline(model, tokenizer, dataset, num_examples=0, categories=None):
    """Evaluate base model on MMLU-Pro (standard MC)."""
    examples = list(dataset)
    if categories:
        examples = [ex for ex in examples if ex["category"] in categories]
    if num_examples > 0:
        examples = examples[:num_examples]

    correct = 0
    total = 0
    cat_correct = Counter()
    cat_total = Counter()

    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt = format_prompt(ex["question"], ex["options"])
        response = generate(model, tokenizer, prompt=prompt, max_tokens=32)
        pred = extract_answer(response)
        expected = ex["answer"]

        is_correct = pred == expected
        if is_correct:
            correct += 1
            cat_correct[ex["category"]] += 1
        total += 1
        cat_total[ex["category"]] += 1

        if i < 5 or (i < 20 and not is_correct):
            status = "OK" if is_correct else "MISS"
            print(f"[{status}] {ex['category']}: {ex['question'][:80]}...")
            print(f"  Expected: {expected}, Got: {pred}, Raw: {response[:50]}")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(examples)}, "
                  f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    dt = time.time() - t0

    print(f"\n{'='*60}")
    print(f"MMLU-Pro Results ({total} examples, {dt:.1f}s)")
    print(f"{'='*60}")
    print(f"Overall: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"\nPer category:")
    for cat in sorted(cat_total.keys()):
        c, t = cat_correct[cat], cat_total[cat]
        print(f"  {cat:20s}: {c}/{t} ({100*c/t:.1f}%)")

    return correct, total


def main():
    parser = argparse.ArgumentParser(description="MMLU-Pro evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--adapter-path", default=None,
                        help="LoRA adapter path (omit for base model)")
    parser.add_argument("--num-examples", type=int, default=0,
                        help="Number of examples (0 = all)")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Filter to specific categories")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    if args.adapter_path:
        model, tokenizer = load(args.model, adapter_path=args.adapter_path)
        print(f"Adapter: {args.adapter_path}")
    else:
        model, tokenizer = load(args.model)
        print("No adapter (base model)")

    print("Loading MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    evaluate_baseline(
        model, tokenizer, ds["test"],
        num_examples=args.num_examples,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
