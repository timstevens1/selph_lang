"""Evaluate the LoRA-tuned model on MathQA test set.

Uses prefix injection (<tool_call> prepended to prompt) and stop token
(</tool_call>) so the model generates only the s-expression body.
"""
import json
import re
import math
import argparse
import time
from pathlib import Path
from mlx_lm import load, generate

SCRIPT_DIR = Path(__file__).parent

# Stop token ID for </tool_call>
TOOL_CALL_CLOSE_ID = 248059


def extract_tool_call(text):
    """Extract s-expression from <tool_call>...</tool_call> delimiters."""
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_first_sexpr(text):
    """Extract first balanced s-expression from text."""
    text = text.strip()
    idx = text.find("(")
    if idx == -1:
        return None
    text = text[idx:]
    depth = 0
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth == 0:
            return text[:i + 1]
    return None


def tokenize_sexpr(s):
    """Tokenize an s-expression into a list of tokens."""
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in '()':
            tokens.append(c)
            i += 1
        elif c in ' \t\n':
            i += 1
        elif c == '-' and i + 1 < len(s) and (s[i + 1].isdigit() or s[i + 1] == '.'):
            j = i + 1
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            j = i
            while j < len(s) and s[j] not in '() \t\n':
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def eval_sexpr_string(s):
    """Evaluate a SELPH s-expression string to a numeric value."""
    if not s:
        return None
    tokens = tokenize_sexpr(s)
    pos = [0]

    def parse():
        if pos[0] >= len(tokens):
            return None
        tok = tokens[pos[0]]
        if tok == '(':
            pos[0] += 1
            op = tokens[pos[0]]
            pos[0] += 1
            args = []
            while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                args.append(parse())
            pos[0] += 1
            return (op, args)
        else:
            pos[0] += 1
            try:
                return float(tok)
            except ValueError:
                return tok

    def evaluate(tree):
        if isinstance(tree, (int, float)):
            return tree
        if isinstance(tree, str):
            try:
                return float(tree)
            except ValueError:
                return None
        op, args = tree
        vals = [evaluate(a) for a in args]
        if any(v is None for v in vals):
            return None

        ops = {
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / b if b != 0 else None,
            "power": lambda a, b: a ** b,
            "sqrt": lambda a: math.sqrt(a) if a >= 0 else None,
            "negate": lambda a: -a,
            "inverse": lambda a: 1 / a if a != 0 else None,
            "floor": lambda a: math.floor(a),
            "log": lambda a: math.log(a) if a > 0 else None,
            "remainder": lambda a, b: a % b if b != 0 else None,
            "factorial": lambda a: math.factorial(int(a)),
            "max": lambda a, b: max(a, b),
            "min": lambda a, b: min(a, b),
            "gcd": lambda a, b: math.gcd(int(a), int(b)),
            "lcm": lambda a, b: abs(a * b) / math.gcd(int(a), int(b)),
            "choose": lambda a, b: math.comb(int(a), int(b)),
            "permutation": lambda a, b: math.perm(int(a), int(b)),
            "sin": lambda a: math.sin(a),
            "cos": lambda a: math.cos(a),
            "tan": lambda a: math.tan(a),
            "circle-area": lambda r: math.pi * r ** 2,
            "circumference": lambda r: 2 * math.pi * r,
            "rectangle-area": lambda a, b: a * b,
            "rectangle-perimeter": lambda a, b: 2 * (a + b),
            "square-area": lambda a: a ** 2,
            "square-perimeter": lambda a: 4 * a,
            "triangle-area": lambda a, b: 0.5 * a * b,
            "triangle-area-three-edges": lambda a, b, c: math.sqrt(
                max(0, (s := (a + b + c) / 2) * (s - a) * (s - b) * (s - c))
            ),
            "triangle-perimeter": lambda a, b, c: a + b + c,
            "rhombus-area": lambda a, b: 0.5 * a * b,
            "rhombus-perimeter": lambda a: 4 * a,
            "quadrilateral-area": lambda a, b, c: 0.5 * a * (b + c),
            "volume-cube": lambda a: a ** 3,
            "volume-cylinder": lambda r, h: math.pi * r ** 2 * h,
            "volume-rectangular-prism": lambda a, b, c: a * b * c,
            "volume-sphere": lambda r: (4 / 3) * math.pi * r ** 3,
            "volume-cone": lambda r, h: (1 / 3) * math.pi * r ** 2 * h,
            "surface-cube": lambda a: 6 * a ** 2,
            "surface-cylinder": lambda r, h: 2 * math.pi * r * (r + h),
            "surface-rectangular-prism": lambda a, b, c: 2 * (a * b + b * c + a * c),
            "surface-sphere": lambda r: 4 * math.pi * r ** 2,
            "cube-edge-by-volume": lambda v: v ** (1 / 3),
            "square-edge-by-perimeter": lambda p: p / 4,
            "square-edge-by-area": lambda a: math.sqrt(a),
            "diagonal": lambda a, b: math.sqrt(a ** 2 + b ** 2),
            "speed": lambda d, t: d / t if t != 0 else None,
            "stream-speed": lambda a, b: (a - b) / 2,
            "speed-in-still-water": lambda a, b: (a + b) / 2,
            "negate-prob": lambda a: 1 - a,
            "price-after-gain": lambda p, g: p * (100 + g) / 100,
            "original-price-before-loss": lambda p, l: p * 100 / (100 - l),
            "original-price-before-gain": lambda p, g: p * 100 / (100 + g),
        }

        if op in ops:
            try:
                return ops[op](*vals)
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                return None
        return None

    try:
        tree = parse()
        return evaluate(tree)
    except Exception:
        return None


def check_match(got, expected, tolerance=0.01):
    """Check if two numeric values match within relative tolerance."""
    if got is None or expected is None:
        return False
    if expected == 0:
        return abs(got) < 1e-6
    return abs(got - expected) / abs(expected) < tolerance


def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural SELPH model")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--adapter-path", default=str(SCRIPT_DIR / "adapters_tc_5k"))
    parser.add_argument("--data", default=str(SCRIPT_DIR / "training_data" / "test.jsonl"))
    parser.add_argument("--num-examples", type=int, default=100,
                        help="Number of test examples (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every example, not just first 10 + misses")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    if args.no_adapter:
        model, tokenizer = load(args.model)
        print("Running WITHOUT adapter (base model)")
    else:
        model, tokenizer = load(args.model, adapter_path=args.adapter_path)
        print(f"Loaded adapter from: {args.adapter_path}")

    # Load test data
    with open(args.data) as f:
        examples = [json.loads(line) for line in f]

    if args.num_examples > 0:
        examples = examples[:args.num_examples]

    total = len(examples)
    parseable = 0
    correct = 0
    exact = 0
    eval_errors = 0

    t0 = time.time()
    for i, ex in enumerate(examples):
        # Prefix injection: append <tool_call> to prompt
        prompt = ex["prompt"] + " <tool_call>"

        # Generate with </tool_call> as stop token
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
        )

        # Extract: try <tool_call>...</tool_call> first, fall back to balanced parens
        full_response = "<tool_call>" + response
        gen_sexpr = extract_tool_call(full_response)
        if gen_sexpr is None:
            gen_sexpr = extract_first_sexpr(response)

        exp_sexpr = extract_tool_call(ex["completion"])

        gen_val = eval_sexpr_string(gen_sexpr)
        exp_val = eval_sexpr_string(exp_sexpr)

        is_parseable = gen_sexpr is not None
        is_correct = check_match(gen_val, exp_val)
        is_exact = gen_sexpr is not None and gen_sexpr == exp_sexpr

        if is_parseable:
            parseable += 1
        if is_correct:
            correct += 1
        if is_exact:
            exact += 1
        if gen_sexpr and gen_val is None:
            eval_errors += 1

        # Print details
        show = args.verbose or i < 10 or (is_parseable and not is_correct and i < 30)
        if show:
            status = "OK" if is_correct else ("PARSE" if not is_parseable else "MISS")
            print(f"\n[{status}] Example {i}")
            print(f"  Problem:  {ex['prompt'][7:90]}...")
            print(f"  Expected: {exp_sexpr}")
            print(f"  Got:      {gen_sexpr}")
            if exp_val is not None or gen_val is not None:
                print(f"  Exp val:  {exp_val}")
                print(f"  Got val:  {gen_val}")

    dt = time.time() - t0

    print(f"\n{'=' * 50}")
    print(f"=== Results ({total} examples, {dt:.1f}s) ===")
    print(f"{'=' * 50}")
    print(f"Parseable:    {parseable}/{total} ({100 * parseable / total:.1f}%)")
    print(f"Eval errors:  {eval_errors}/{total} ({100 * eval_errors / total:.1f}%)")
    print(f"Correct val:  {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"Exact match:  {exact}/{total} ({100 * exact / total:.1f}%)")
    print(f"Speed:        {dt / total:.2f}s/example")


if __name__ == "__main__":
    main()
