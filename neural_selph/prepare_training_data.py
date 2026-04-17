"""Prepare MathQA data in MLX LoRA completions format.

Format: {"prompt": "...", "completion": "..."}
One JSONL line per example.

The prompt is the NL math problem.
The completion is the SELPH s-expression wrapped in <tool_call>...</tool_call>
(single tokens 248058/248059 in Qwen tokenizer).
"""
import json
import random
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "training_data"
OUT.mkdir(exist_ok=True)

# Load converted data
for split_name, out_name in [("train", "train"), ("dev", "valid"), ("test", "test")]:
    selph_path = DATA / f"{split_name}_selph.json"
    if not selph_path.exists():
        from convert_mathqa import convert_dataset
        results, errors = convert_dataset(split_name)
        with open(selph_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Converted {split_name}: {len(results)} examples")

    with open(selph_path) as f:
        data = json.load(f)

    # Filter to evaluable examples only
    data = [ex for ex in data if ex["computed_value"] is not None]

    # Format for MLX LoRA
    # Using <tool_call>...</tool_call> as single-token delimiters
    formatted = []
    for ex in data:
        prompt = f"Solve: {ex['problem'].strip()}\nExpression:"
        completion = f" <tool_call>{ex['selph_expr']}</tool_call>"
        formatted.append({"prompt": prompt, "completion": completion})

    # Shuffle train
    if out_name == "train":
        random.seed(42)
        random.shuffle(formatted)

    out_path = OUT / f"{out_name}.jsonl"
    with open(out_path, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")

    print(f"{out_name}: {len(formatted)} examples -> {out_path}")

# Show a few examples and verify tokenization
print("\n--- Sample training examples ---")
with open(OUT / "train.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        ex = json.loads(line)
        print(f"\nPrompt: {ex['prompt'][:100]}")
        print(f"Completion: {ex['completion'][:100]}")

# Verify single-token encoding
print("\n--- Tokenization check ---")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    for delim in ["<tool_call>", "</tool_call>"]:
        ids = tok.encode(delim, add_special_tokens=False)
        print(f"  {delim}: {ids} ({'SINGLE TOKEN' if len(ids) == 1 else f'MULTI TOKEN ({len(ids)})'}")
except Exception as e:
    print(f"  Tokenizer check skipped: {e}")
