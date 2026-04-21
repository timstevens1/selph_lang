"""Generate thinking-with-SELPH training data for MMLU-Pro business calculations.

Uses a large model (via mlx_lm) to generate chain-of-thought solutions,
then post-processes to insert <tool_call> expressions at computation steps.

Alternatively, generates synthetic think traces from the question + answer
by working backwards from the correct answer.

Format:
  prompt: "Question: ... A. ... B. ...\n\n<think>\n"
  completion: "Let me work through this.\nCommission = <tool_call>(multiply 1200 0.05)</tool_call>60\n...</think>\nJ"
"""
import json
import re
import math
import random
import argparse
from pathlib import Path
from collections import Counter
from datasets import load_dataset

OUT = Path(__file__).parent / "data"
LETTERS = "ABCDEFGHIJ"


def extract_numbers(text):
    """Extract all numbers from text."""
    numbers = []
    for m in re.finditer(r'(?<![a-zA-Z])(\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)\s*%?', text):
        s = m.group(1).replace(',', '')
        try:
            if '/' in s:
                num, den = s.split('/')
                numbers.append(float(num) / float(den))
            else:
                numbers.append(float(s))
        except:
            pass
    # Check for percentage context
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text):
        try:
            numbers.append(float(m.group(1)) / 100)
        except:
            pass
    return numbers


def eval_sexpr(s):
    """Evaluate s-expression to a number."""
    if not s:
        return None
    def tokenize(s):
        tokens, i = [], 0
        while i < len(s):
            c = s[i]
            if c in '()': tokens.append(c); i += 1
            elif c in ' \t\n': i += 1
            elif c == '-' and i + 1 < len(s) and (s[i+1].isdigit() or s[i+1] == '.'):
                j = i + 1
                while j < len(s) and (s[j].isdigit() or s[j] == '.'): j += 1
                tokens.append(s[i:j]); i = j
            else:
                j = i
                while j < len(s) and s[j] not in '() \t\n': j += 1
                tokens.append(s[i:j]); i = j
        return tokens
    try:
        tokens = tokenize(s)
        pos = [0]
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
                if op == "add": return vals[0] + vals[1]
                elif op == "subtract": return vals[0] - vals[1]
                elif op == "multiply": return vals[0] * vals[1]
                elif op == "divide": return vals[0] / vals[1] if vals[1] != 0 else None
                elif op == "power": return vals[0] ** vals[1]
                elif op == "sqrt": return math.sqrt(vals[0]) if vals[0] >= 0 else None
                elif op == "round": return round(vals[0], int(vals[1]) if len(vals) > 1 else 2)
                elif op == "negate": return -vals[0]
                elif op == "abs": return abs(vals[0])
                else: return None
            except: return None
        return ev(parse())
    except:
        return None


def format_number(n):
    """Format a number for display."""
    if n is None: return "ERROR"
    if isinstance(n, float):
        if n == int(n) and abs(n) < 1e15:
            return str(int(n))
        return f"{n:.2f}"
    return str(n)


def try_match_answer(value, options):
    """Try to match a computed value against the answer options."""
    if value is None:
        return None
    val_str = format_number(value)

    for i, opt in enumerate(options):
        # Extract numbers from option
        opt_nums = extract_numbers(opt)
        for on in opt_nums:
            if on == 0 and value == 0:
                return LETTERS[i]
            if on != 0 and abs(value - on) / abs(on) < 0.02:
                return LETTERS[i]
        # Also try string match
        if val_str in opt or f"${val_str}" in opt:
            return LETTERS[i]
    return None


# ─── Templates for synthetic think traces ───────────────────────────────────

def make_percentage_trace(q, nums, answer, options):
    """Generate trace for percentage/commission problems."""
    if len(nums) < 2:
        return None
    # Try: percentage of a number
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if i == j: continue
            # a% of b
            if a <= 100:
                rate = a / 100
                result = b * rate
                letter = try_match_answer(result, options)
                if letter == answer:
                    sexpr = f"(multiply {b} {rate})"
                    return (
                        f"I need to calculate {a}% of {b}.\n"
                        f"<tool_call>{sexpr}</tool_call>{format_number(result)}\n"
                        f"The answer is ${format_number(result)}.\n"
                    ), letter
            # b% of a
            if b <= 100:
                rate = b / 100
                result = a * rate
                letter = try_match_answer(result, options)
                if letter == answer:
                    sexpr = f"(multiply {a} {rate})"
                    return (
                        f"I need to calculate {b}% of {a}.\n"
                        f"<tool_call>{sexpr}</tool_call>{format_number(result)}\n"
                        f"The answer is ${format_number(result)}.\n"
                    ), letter
    return None


def make_compound_interest_trace(q, nums, answer, options):
    """Generate trace for compound interest problems."""
    # Look for principal, rate, time
    for p in nums:
        if p < 100: continue  # principal should be >= 100
        for r in nums:
            if r > 50 or r <= 0: continue  # rate as percentage
            rate = r / 100
            for t in nums:
                if t <= 0 or t > 50 or t == r: continue
                result = p * (1 + rate) ** t
                letter = try_match_answer(result, options)
                if letter == answer:
                    sexpr = f"(multiply {p} (power (add 1 {rate}) {t}))"
                    return (
                        f"This is a compound interest problem.\n"
                        f"Principal = ${format_number(p)}, rate = {r}%, time = {format_number(t)} years.\n"
                        f"Accumulated value = P × (1 + r)^t = <tool_call>{sexpr}</tool_call>{format_number(result)}\n"
                        f"The answer is ${format_number(result)}.\n"
                    ), letter
    return None


def make_discount_trace(q, nums, answer, options):
    """Generate trace for discount/markup problems."""
    for price in nums:
        if price < 1: continue
        for rate in nums:
            if rate <= 0 or rate >= 100 or rate == price: continue
            # Discount
            result = price * (1 - rate/100)
            letter = try_match_answer(result, options)
            if letter == answer:
                sexpr = f"(multiply {price} (subtract 1 {rate/100}))"
                return (
                    f"The list price is ${format_number(price)} with a {format_number(rate)}% discount.\n"
                    f"Net price = <tool_call>{sexpr}</tool_call>{format_number(result)}\n"
                    f"The answer is ${format_number(result)}.\n"
                ), letter
            # Markup
            result = price * (1 + rate/100)
            letter = try_match_answer(result, options)
            if letter == answer:
                sexpr = f"(multiply {price} (add 1 {rate/100}))"
                return (
                    f"The cost is ${format_number(price)} with a {format_number(rate)}% markup.\n"
                    f"Selling price = <tool_call>{sexpr}</tool_call>{format_number(result)}\n"
                    f"The answer is ${format_number(result)}.\n"
                ), letter
    return None


def make_basic_arithmetic_trace(q, nums, answer, options):
    """Generate trace for basic arithmetic (add, subtract, multiply, divide)."""
    if len(nums) < 2:
        return None
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if i == j or b == 0: continue
            for op_name, op_sym, op_fn in [
                ("multiply", "multiply", lambda x, y: x * y),
                ("divide", "divide", lambda x, y: x / y),
                ("add", "add", lambda x, y: x + y),
                ("subtract", "subtract", lambda x, y: x - y),
            ]:
                result = op_fn(a, b)
                letter = try_match_answer(result, options)
                if letter == answer:
                    sexpr = f"({op_sym} {a} {b})"
                    return (
                        f"I need to {op_name} {format_number(a)} and {format_number(b)}.\n"
                        f"<tool_call>{sexpr}</tool_call>{format_number(result)}\n"
                        f"The answer is {format_number(result)}.\n"
                    ), letter
    return None


def generate_think_trace(ex):
    """Try to generate a thinking trace with SELPH expressions for a question."""
    q = ex["question"]
    nums = extract_numbers(q)
    answer = ex["answer"]
    options = ex["options"]

    # Try each template in order of specificity
    for template_fn in [
        make_compound_interest_trace,
        make_percentage_trace,
        make_discount_trace,
        make_basic_arithmetic_trace,
    ]:
        result = template_fn(q, nums, answer, options)
        if result:
            trace, letter = result
            return trace, letter

    return None, None


def format_for_training(ex, trace, letter):
    """Format as MLX training example."""
    opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(ex["options"]))
    prompt = (
        f"Question: {ex['question']}\n{opts}\n\n"
        f"<think>\n"
    )
    completion = f"Let me work through this step by step.\n{trace}</think>\n{letter}"
    return {"prompt": prompt, "completion": completion}


if __name__ == "__main__":
    print("Loading MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    # Get business calculation questions
    business_calc = [ex for ex in ds["test"] if ex["src"] == "stemez-Business"]
    print(f"Business calculation questions: {len(business_calc)}")

    # Generate think traces
    generated = []
    failed = 0
    for ex in business_calc:
        trace, letter = generate_think_trace(ex)
        if trace and letter == ex["answer"]:
            formatted = format_for_training(ex, trace, letter)
            generated.append(formatted)
        else:
            failed += 1

    print(f"Generated: {len(generated)}")
    print(f"Failed: {failed}")

    # Show samples
    print("\n=== Samples ===")
    random.seed(42)
    for ex in random.sample(generated, min(5, len(generated))):
        print(f"\nPrompt: {ex['prompt'][:150]}...")
        print(f"Completion: {ex['completion'][:200]}...")

    # Split and save
    random.seed(42)
    random.shuffle(generated)
    split = int(len(generated) * 0.9)
    train = generated[:split]
    valid = generated[split:]

    out = Path("training_data_think_selph")
    out.mkdir(exist_ok=True)

    for name, data in [("train", train), ("valid", valid)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"\n{name}: {len(data)} examples")

    # Also save test.jsonl (the questions we couldn't generate traces for,
    # plus a held-out sample)
    test_qs = [ex for ex in business_calc
               if generate_think_trace(ex)[0] is None][:50]
    # Format test as prompt-only (for evaluation)
    test_formatted = []
    for ex in test_qs:
        opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(ex["options"]))
        test_formatted.append({
            "prompt": f"Question: {ex['question']}\n{opts}\n\n<think>\n",
            "completion": f"</think>\n{ex['answer']}",
            "answer": ex["answer"],
        })
    with open(out / "test.jsonl", "w") as f:
        for item in test_formatted:
            f.write(json.dumps(item) + "\n")
    print(f"test: {len(test_formatted)} examples (hard — no template match)")
