"""MMLU-Pro evaluation with thinking mode + inline SELPH evaluation.

The model thinks in natural language inside <think>...</think>.
When it emits <tool_call>...</tool_call>, the s-expression is immediately
evaluated and the entire block is replaced with the result in context.
The model sees the computed value and continues reasoning.

Supports both computation and knowledge lookup:
  <tool_call>(multiply 1000 (power (add 1 0.05) 4))</tool_call>1215.51
  <tool_call>(lookup "vitamin A")</tool_call>Vitamin A is a fat-soluble vitamin...
  <tool_call>(apropos "vitamin")</tool_call>["vitamin A", "vitamin K", ...]
"""
import json
import re
import math
import argparse
import time
from pathlib import Path
from collections import Counter
from datasets import load_dataset

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

SCRIPT_DIR = Path(__file__).parent
LETTERS = "ABCDEFGHIJ"

TOOL_CALL_OPEN_ID = 248058   # <tool_call>
TOOL_CALL_CLOSE_ID = 248059  # </tool_call>
THINK_CLOSE_ID = 248069      # </think>
EOS_ID = 248044               # <|endoftext|>

# Global knowledge KB (loaded once)
KNOWLEDGE_KB = None

def load_knowledge_kb(path=None):
    """Load the knowledge KB for lookup/apropos support."""
    global KNOWLEDGE_KB
    if path is None:
        path = SCRIPT_DIR / "data" / "knowledge_kb_clean.json"
    if path.exists():
        KNOWLEDGE_KB = json.loads(path.read_text())
        print(f"Knowledge KB loaded: {len(KNOWLEDGE_KB)} entries")
    else:
        KNOWLEDGE_KB = {}
        print(f"No knowledge KB at {path}")


def kb_lookup(concept):
    """Look up a concept in the knowledge KB."""
    if not KNOWLEDGE_KB:
        return f"Unknown concept: {concept}"
    key = concept.lower().strip()
    if key in KNOWLEDGE_KB:
        return KNOWLEDGE_KB[key]["definition"]
    # Fuzzy match: check if any key contains the query
    for k, v in KNOWLEDGE_KB.items():
        if key in k or k in key:
            return v["definition"]
    return f"Unknown concept: {concept}"


def kb_apropos(keyword):
    """Search KB for entries matching a keyword."""
    if not KNOWLEDGE_KB:
        return "No entries found"
    keyword = keyword.lower().strip()
    matches = [v["term"] for k, v in KNOWLEDGE_KB.items()
               if keyword in k or keyword in v.get("definition", "").lower()[:200]]
    if matches:
        return ", ".join(matches[:10])
    return "No entries found"


def eval_sexpr_string(s):
    """Evaluate a SELPH s-expression to a value."""
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
        tokens = tokenize(s)
        pos = [0]

        def parse():
            if pos[0] >= len(tokens): return None
            tok = tokens[pos[0]]
            if tok == '(':
                pos[0] += 1; op = tokens[pos[0]]; pos[0] += 1
                args = []
                while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                    args.append(parse())
                pos[0] += 1
                return (op, args)
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
                elif op == "log": return math.log(vals[0]) if vals[0] > 0 else None
                elif op == "remainder": return vals[0] % vals[1]
                elif op == "factorial": return math.factorial(int(vals[0]))
                elif op == "max": return max(vals)
                elif op == "min": return min(vals)
                elif op == "gcd": return math.gcd(int(vals[0]), int(vals[1]))
                elif op == "abs": return abs(vals[0])
                elif op == "round": return round(vals[0], int(vals[1]) if len(vals) > 1 else 0)
                # Knowledge operations
                elif op == "lookup": return kb_lookup(str(vals[0]))
                elif op == "related": return kb_lookup(f"{vals[0]} {vals[1]}")
                elif op == "apropos": return kb_apropos(str(vals[0]))
                else: return None
            except: return None

        result = ev(parse())
        return result
    except:
        return None


def format_result(value):
    """Format an evaluated result for injection into the text stream."""
    if value is None:
        return "ERROR"
    if isinstance(value, float):
        # Clean up floating point: show reasonable precision
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def generate_with_selph(model, tokenizer, prompt, max_tokens=512):
    """Generate text with inline SELPH evaluation.

    When <tool_call> is detected, accumulate tokens until </tool_call>,
    evaluate the s-expression, replace with the result, and continue.
    """
    tokens = mx.array(tokenizer.encode(prompt))
    sampler = make_sampler(temp=0.0)

    generated_text = ""
    in_tool_call = False
    tool_call_buffer = ""
    num_evals = 0

    for (token, logprobs), _ in zip(
        generate_step(tokens, model, max_tokens=max_tokens, sampler=sampler),
        range(max_tokens)
    ):
        token_id = token.item() if hasattr(token, 'item') else int(token)

        # Stop conditions
        if token_id == EOS_ID:
            break

        if token_id == TOOL_CALL_OPEN_ID:
            # Start accumulating s-expression
            in_tool_call = True
            tool_call_buffer = ""
            continue

        if token_id == TOOL_CALL_CLOSE_ID and in_tool_call:
            # Evaluate the s-expression
            sexpr = tool_call_buffer.strip()
            result = eval_sexpr_string(sexpr)
            result_str = format_result(result)
            num_evals += 1

            # Inject result into the generated text
            generated_text += result_str

            # Re-encode the text so far and update the prompt for continued generation
            # This is the key: the model sees the evaluated result, not the s-expression
            new_tokens = mx.array(tokenizer.encode(result_str, add_special_tokens=False))
            tokens = mx.concatenate([tokens, new_tokens])

            in_tool_call = False
            tool_call_buffer = ""
            continue

        # Decode token
        token_str = tokenizer.decode([token_id])

        if in_tool_call:
            tool_call_buffer += token_str
        else:
            generated_text += token_str

        # Stop after </think> + answer
        if token_id == THINK_CLOSE_ID:
            # Generate a few more tokens for the answer
            remaining_text = ""
            for (tok2, _), _ in zip(
                generate_step(tokens, model, max_tokens=20, sampler=sampler),
                range(20)
            ):
                tid = tok2.item() if hasattr(tok2, 'item') else int(tok2)
                if tid == EOS_ID:
                    break
                remaining_text += tokenizer.decode([tid])
                if re.search(r'[A-J]', remaining_text):
                    break
            generated_text += remaining_text
            break

    return generated_text, num_evals


def extract_answer(response):
    """Extract answer letter from response (after </think> if present)."""
    # If there's a </think>, look after it
    parts = response.split("</think>")
    search_text = parts[-1] if len(parts) > 1 else response

    # Find first A-J letter
    m = re.search(r'\b([A-J])\b', search_text)
    if m:
        return m.group(1)

    # Fallback: search entire response
    m = re.search(r'\b([A-J])\b', response)
    return m.group(1) if m else None


SELPH_SYSTEM = """You have access to SELPH, a symbolic computation and knowledge system. Use <tool_call>(expression)</tool_call> during thinking to evaluate expressions. Results replace the tool_call block.

Core functions:
  Arithmetic: (add a b), (subtract a b), (multiply a b), (divide a b), (power base exp), (sqrt x), (abs x), (floor x), (round x n)
  Percentage: (multiply value (divide percent 100)) for "X% of Y"
  Finance: (multiply P (power (add 1 r) n)) for compound interest, (divide FV (power (add 1 r) n)) for present value
  Knowledge: (lookup concept) for definitions and facts, (related concept relation) for relationships between concepts

Discovery:
  (apropos "keyword") - search for functions by name
  (apropos-by-type "input-type" "output-type") - search functions by type signature

Use <tool_call> whenever you need to compute a value or look up a fact you are unsure about."""


def format_prompt_think(question, options):
    """Format prompt with thinking mode + SELPH system prompt."""
    opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        f"{SELPH_SYSTEM}\n\n"
        f"Question: {question}\n{opts}\n\n"
        f"<think>\n"
    )


def format_prompt_baseline(question, options):
    """Format standard MC prompt (no thinking)."""
    opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return f"Question: {question}\n{opts}\nAnswer:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--categories", nargs="*", default=["business"])
    parser.add_argument("--mode", choices=["baseline", "think", "think-selph"], default="think-selph")
    parser.add_argument("--src-filter", default=None, help="Filter by source (e.g. stemez-Business)")
    args = parser.parse_args()

    # Load knowledge KB for lookup/apropos
    load_knowledge_kb()

    print(f"Loading model: {args.model}")
    if args.adapter_path:
        model, tokenizer = load(args.model, adapter_path=args.adapter_path)
        print(f"Adapter: {args.adapter_path}")
    else:
        model, tokenizer = load(args.model)

    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    examples = [ex for ex in ds["test"]
                if ex["category"] in args.categories]
    if args.src_filter:
        examples = [ex for ex in examples if ex["src"] == args.src_filter]

    import random
    random.seed(42)
    if args.num_examples > 0 and args.num_examples < len(examples):
        examples = random.sample(examples, args.num_examples)

    print(f"Mode: {args.mode}")
    print(f"Examples: {len(examples)}")
    print()

    correct = 0
    total_evals = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        if args.mode == "baseline":
            prompt = format_prompt_baseline(ex["question"], ex["options"])
            from mlx_lm import generate
            resp = generate(model, tokenizer, prompt=prompt, max_tokens=32)
            num_evals = 0
        elif args.mode == "think":
            prompt = format_prompt_think(ex["question"], ex["options"])
            from mlx_lm import generate
            resp = generate(model, tokenizer, prompt=prompt, max_tokens=256)
            num_evals = 0
        else:  # think-selph
            prompt = format_prompt_think(ex["question"], ex["options"])
            resp, num_evals = generate_with_selph(model, tokenizer, prompt, max_tokens=512)

        total_evals += num_evals
        pred = extract_answer(resp)
        hit = pred == ex["answer"]
        if hit:
            correct += 1

        if i < 5 or (i < 15 and not hit):
            status = "OK" if hit else "MISS"
            print(f"[{status}] Q: {ex['question'][:80]}...")
            print(f"  Expected: {ex['answer']}, Got: {pred}")
            if num_evals > 0:
                print(f"  SELPH evals: {num_evals}")
            # Show thinking trace (truncated)
            trace = resp[:200].replace('\n', '\\n')
            print(f"  Trace: {trace}")
            print()

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(examples)}, "
                  f"Acc: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")

    dt = time.time() - t0
    n = len(examples)
    print(f"\n{'='*60}")
    print(f"MMLU-Pro {args.mode} ({n} examples, {dt:.1f}s)")
    print(f"{'='*60}")
    print(f"Accuracy: {correct}/{n} ({100*correct/n:.1f}%)")
    print(f"SELPH evaluations: {total_evals}")


if __name__ == "__main__":
    main()
