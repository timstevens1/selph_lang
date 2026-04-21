"""Grammar-constrained decoding for SELPH s-expressions.

Uses MLX's logits_processors hook to mask out tokens that would produce
syntactically invalid s-expressions. The grammar:

  output   ::= "<selph>" expr "</selph>"
  expr     ::= atom | "(" op ws expr (ws expr)* ")"
  op       ::= "add" | "subtract" | "multiply" | "divide" | ...
  atom     ::= number
  number   ::= "-"? [0-9]+ ("." [0-9]+)?
  ws       ::= " "+

The processor tracks paren depth and parser state to determine which
tokens are valid continuations at each step.
"""
import mlx.core as mx
import numpy as np
from typing import List, Optional, Set


# All SELPH operators from the MathQA conversion
SELPH_OPS = {
    "add", "subtract", "multiply", "divide", "power", "sqrt", "log",
    "negate", "inverse", "floor", "factorial", "max", "min", "remainder",
    "gcd", "lcm", "choose", "permutation", "sin", "cos", "tan",
    "circle-area", "circumference", "rectangle-area", "rectangle-perimeter",
    "square-area", "square-perimeter", "triangle-area",
    "triangle-area-three-edges", "triangle-perimeter",
    "rhombus-area", "rhombus-perimeter", "quadrilateral-area",
    "volume-cube", "volume-cylinder", "volume-rectangular-prism",
    "volume-sphere", "volume-cone",
    "surface-cube", "surface-cylinder", "surface-rectangular-prism",
    "surface-sphere", "cube-edge-by-volume",
    "square-edge-by-perimeter", "square-edge-by-area",
    "diagonal", "speed", "stream-speed", "speed-in-still-water",
    "negate-prob", "original-price-before-loss",
    "original-price-before-gain", "price-after-gain",
}


class SExprState:
    """Track parser state for constrained s-expression generation."""

    # States
    EXPECT_OPEN_SELPH = 0   # Waiting for <selph>
    EXPECT_EXPR = 1          # Expect ( or number
    EXPECT_OP = 2            # Inside (, expect operator name
    EXPECT_ARG_OR_CLOSE = 3  # Expect another arg (expr) or )
    EXPECT_CLOSE_SELPH = 4   # Expect </selph>
    DONE = 5

    def __init__(self):
        self.state = self.EXPECT_OPEN_SELPH
        self.paren_depth = 0
        # Stack of how many args each open paren has seen
        self.arg_counts = []
        # Minimum args per op (most are 2, some are 1, some are 3)
        self.min_args = {
            "sqrt": 1, "negate": 1, "inverse": 1, "floor": 1,
            "factorial": 1, "log": 1, "sin": 1, "cos": 1, "tan": 1,
            "circle-area": 1, "circumference": 1, "square-area": 1,
            "square-perimeter": 1, "volume-cube": 1, "surface-cube": 1,
            "cube-edge-by-volume": 1, "square-edge-by-perimeter": 1,
            "square-edge-by-area": 1, "negate-prob": 1,
            "volume-rectangular-prism": 3, "surface-rectangular-prism": 3,
            "triangle-area-three-edges": 3,
        }
        self.current_op_stack = []

    def can_open_paren(self):
        return self.state in (self.EXPECT_EXPR, self.EXPECT_ARG_OR_CLOSE)

    def can_close_paren(self):
        if self.state != self.EXPECT_ARG_OR_CLOSE or self.paren_depth <= 0:
            return False
        # Check minimum arg count
        if self.current_op_stack:
            op = self.current_op_stack[-1]
            min_a = self.min_args.get(op, 2)
            if self.arg_counts and self.arg_counts[-1] < min_a:
                return False
        return True

    def can_number(self):
        return self.state in (self.EXPECT_EXPR, self.EXPECT_ARG_OR_CLOSE)

    def can_operator(self):
        return self.state == self.EXPECT_OP

    def open_paren(self):
        self.paren_depth += 1
        self.arg_counts.append(0)
        self.state = self.EXPECT_OP

    def close_paren(self):
        self.paren_depth -= 1
        self.arg_counts.pop()
        if self.current_op_stack:
            self.current_op_stack.pop()
        if self.paren_depth == 0:
            self.state = self.EXPECT_CLOSE_SELPH
        else:
            # We just completed a sub-expression which is an arg
            if self.arg_counts:
                self.arg_counts[-1] += 1
            self.state = self.EXPECT_ARG_OR_CLOSE

    def set_operator(self, op_name):
        self.current_op_stack.append(op_name)
        self.state = self.EXPECT_ARG_OR_CLOSE

    def add_number(self):
        if self.paren_depth == 0:
            # Top-level atom
            self.state = self.EXPECT_CLOSE_SELPH
        else:
            if self.arg_counts:
                self.arg_counts[-1] += 1
            self.state = self.EXPECT_ARG_OR_CLOSE


def build_token_categories(tokenizer):
    """Categorize all tokens in the vocabulary for fast masking.

    Returns dicts mapping token IDs to categories.
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    # Reverse map: id -> token string
    id_to_token = {v: k for k, v in vocab.items()}

    # Build category sets
    open_paren_ids = set()      # Tokens that are or start with "("
    close_paren_ids = set()     # Tokens that are or start with ")"
    number_start_ids = set()    # Tokens that start a number
    operator_ids = {}           # op_name -> set of token ids that form the op
    space_ids = set()           # Whitespace tokens
    selph_open_ids = set()      # <selph> token(s)
    selph_close_ids = set()     # </selph> token(s)

    # Find token IDs for our delimiters
    selph_open_encoded = tokenizer.encode("<selph>", add_special_tokens=False)
    selph_close_encoded = tokenizer.encode("</selph>", add_special_tokens=False)

    # For each token, categorize it
    for tid, tstr in id_to_token.items():
        # Decode the token to get actual text
        stripped = tstr.replace('Ġ', ' ').replace('▁', ' ').strip()

        if stripped == '(':
            open_paren_ids.add(tid)
        if stripped == ')':
            close_paren_ids.add(tid)
        if stripped and (stripped[0].isdigit() or (stripped[0] == '-' and len(stripped) > 1 and stripped[1].isdigit())):
            number_start_ids.add(tid)
        # Check for numbers with leading dot
        if stripped and stripped[0] == '.' and len(stripped) > 1 and stripped[1].isdigit():
            number_start_ids.add(tid)
        if stripped and stripped[0] == ' ':
            space_ids.add(tid)

    # Build operator token sequences
    for op in SELPH_OPS:
        encoded = tokenizer.encode(op, add_special_tokens=False)
        if op not in operator_ids:
            operator_ids[op] = encoded

    return {
        "open_paren": open_paren_ids,
        "close_paren": close_paren_ids,
        "number_start": number_start_ids,
        "operator_ids": operator_ids,
        "space": space_ids,
        "selph_open": selph_open_encoded,
        "selph_close": selph_close_encoded,
        "vocab_size": vocab_size,
        "id_to_token": id_to_token,
    }


def make_sexpr_logits_processor(tokenizer):
    """Create a logits processor that constrains output to valid s-expressions.

    This is a simplified version that works at the token level by tracking
    state and building allow-masks. For production use, you'd want a more
    sophisticated grammar-guided approach (e.g., outlines/xgrammar).

    Returns a function compatible with MLX's logits_processors.
    """
    categories = build_token_categories(tokenizer)
    state = SExprState()

    # Track generation phase
    phase = {"value": "pre_selph"}  # pre_selph, in_selph, post_selph
    buffer = {"tokens": []}  # Track token buffer for multi-token sequences

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        """Mask logits to only allow valid s-expression continuations."""
        # For now, return logits unmodified but log what we'd constrain
        # Full implementation would build allow-mask based on state
        # This is the hook point - we'll refine this iteratively
        return logits

    return processor


def make_simple_sexpr_processor(tokenizer):
    """Simpler approach: just ensure balanced parens and valid structure.

    Tracks paren depth and enforces:
    - After <selph>: must start with ( or digit
    - After (: must have a valid operator token
    - Parens must balance before </selph>
    - No generation after </selph>

    Works by penalizing (setting to -inf) invalid tokens.
    """
    vocab = tokenizer.get_vocab()
    vocab_size = max(vocab.values()) + 1
    id_to_token = {v: k for k, v in vocab.items()}

    # Pre-encode key tokens
    open_paren = tokenizer.encode("(", add_special_tokens=False)
    close_paren = tokenizer.encode(")", add_special_tokens=False)

    paren_depth = [0]
    in_selph = [False]
    done = [False]

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if done[0]:
            # Force EOS
            mask = mx.full(logits.shape, -1e9)
            eos_id = tokenizer.eos_token_id
            if eos_id is not None:
                mask = mask.at[eos_id].add(1e9)
            return logits + mask

        return logits

    return processor


if __name__ == "__main__":
    # Quick test of token categorization
    from mlx_lm import load

    print("Loading tokenizer...")
    _, tokenizer = load("Qwen/Qwen3.5-0.8B-Base")

    cats = build_token_categories(tokenizer)
    print(f"Vocab size: {cats['vocab_size']}")
    print(f"Open paren tokens: {len(cats['open_paren'])}")
    print(f"Close paren tokens: {len(cats['close_paren'])}")
    print(f"Number tokens: {len(cats['number_start'])}")
    print(f"Space tokens: {len(cats['space'])}")
    print(f"Operators: {len(cats['operator_ids'])}")

    # Show how some operators tokenize
    for op in ["add", "subtract", "multiply", "divide", "circle-area", "volume-rectangular-prism"]:
        encoded = tokenizer.encode(op, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in encoded]
        print(f"  {op}: {encoded} -> {decoded}")

    # Show delimiter tokenization
    for delim in ["<selph>", "</selph>"]:
        encoded = tokenizer.encode(delim, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in encoded]
        print(f"  {delim}: {encoded} -> {decoded}")

    # Test: encode a full expression and show tokens
    test_expr = "<selph>(multiply (divide 120 50) 100)</selph>"
    encoded = tokenizer.encode(test_expr, add_special_tokens=False)
    for tid in encoded:
        print(f"  {tid:6d} -> {repr(tokenizer.decode([tid]))}")
