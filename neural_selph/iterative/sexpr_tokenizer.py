"""Custom tokenizer for s-expression refinement traces.

Small vocabulary (~300 tokens) covering:
- S-expression syntax: ( ) _HOLE_
- Function names from geo KB: capital, population, continent, etc.
- Entity names: country and city names from the KB
- Numeric tokens: digits 0-9, decimal point
- Trace structure: SEP_TASK, SEP_EXPR, SEP_FEEDBACK, SEP_EDIT
- Control: PAD, BOS, EOS, NO_OP, CORRECT, WRONG, ERROR
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# Geo KB functions (fixed set)
GEO_FUNCTIONS = [
    "capital", "population", "continent", "currency", "language",
    "area", "head-of-state", "country-of", "largest-by-population", "divide",
]

# Special tokens
SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>",
    "(", ")",
    "_HOLE_", "NO_OP",
    "SEP_TASK", "SEP_EXPR", "SEP_FEEDBACK", "SEP_EDIT",
    "SEP_QUERY", "SEP_QUERY_RESULT", "SEP_QUERY_OUT",  # v2 (archived)
    "<query>", "</query>", "<q_out>", "</q_out>",
    "CORRECT", "WRONG", "ERROR", "INCOMPLETE",
    # Numeric building blocks
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-",
    "NUM_SEP",  # separates digit sequences from rest
    # Question type tokens (what the user is asking for)
    "Q_capital", "Q_population", "Q_continent", "Q_currency",
    "Q_language", "Q_area", "Q_head-of-state", "Q_country-of",
    "Q_largest-by-population", "Q_divide", "Q_density",
    "Q_capital-population", "Q_unknown",
]


class SexprTokenizer:
    """Tokenizer for s-expression refinement traces."""

    def __init__(self, kb_path: Optional[str] = None):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        # Build vocabulary
        vocab = list(SPECIAL_TOKENS)
        vocab.extend(GEO_FUNCTIONS)

        # Add entity names from KB
        if kb_path:
            kb = json.loads(Path(kb_path).read_text())
            entities = set()
            for country, info in kb.items():
                entities.add(country)
                if info.get("capital"):
                    entities.add(info["capital"])
                if info.get("continent"):
                    entities.add(info["continent"])
                if info.get("currency"):
                    entities.add(info["currency"])
                if info.get("language"):
                    entities.add(info["language"])
                if info.get("head_of_state"):
                    entities.add(info["head_of_state"])
            vocab.extend(sorted(entities))

        # Assign IDs
        for i, tok in enumerate(vocab):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

        self.vocab_size = len(vocab)
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]

    def tokenize_sexpr(self, expr: str) -> list[str]:
        """Tokenize an s-expression into tokens.

        Handles: (func "Entity Name") and nested expressions.
        Numbers are tokenized digit-by-digit with NUM_SEP boundaries.
        """
        tokens = []
        i = 0
        expr = expr.strip()

        while i < len(expr):
            c = expr[i]

            if c in " \t\n\r":
                i += 1
                continue

            if c == "(":
                tokens.append("(")
                i += 1
            elif c == ")":
                tokens.append(")")
                i += 1
            elif c == '"':
                # Quoted string — extract and look up as entity
                j = i + 1
                while j < len(expr) and expr[j] != '"':
                    j += 1
                entity = expr[i + 1 : j]
                if entity in self.token_to_id:
                    tokens.append(entity)
                else:
                    # Unknown entity — tokenize character by character as fallback
                    for ch in entity:
                        if ch in self.token_to_id:
                            tokens.append(ch)
                i = j + 1
            elif c == "_" and expr[i:].startswith("_HOLE_"):
                tokens.append("_HOLE_")
                i += 6
            elif c.isdigit() or (c == "-" and i + 1 < len(expr) and expr[i + 1].isdigit()):
                # Number
                tokens.append("NUM_SEP")
                if c == "-":
                    tokens.append("-")
                    i += 1
                while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                    tokens.append(expr[i])
                    i += 1
                tokens.append("NUM_SEP")
            else:
                # Function name or identifier
                j = i
                while j < len(expr) and expr[j] not in " \t\n\r()\"":
                    j += 1
                word = expr[i:j]
                if word in self.token_to_id:
                    tokens.append(word)
                else:
                    # Unknown — skip (shouldn't happen with our controlled vocab)
                    pass
                i = j

        return tokens

    def tokenize_number(self, n: str) -> list[str]:
        """Tokenize a number string digit-by-digit."""
        tokens = ["NUM_SEP"]
        for c in str(n):
            if c in self.token_to_id:
                tokens.append(c)
        tokens.append("NUM_SEP")
        return tokens

    def encode(self, tokens: list[str], add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Convert token strings to IDs."""
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for t in tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        """Convert IDs back to token strings."""
        return [self.id_to_token.get(i, "?") for i in ids
                if i not in (self.pad_id, self.bos_id, self.eos_id)]

    def decode_to_sexpr(self, ids: list[int]) -> str:
        """Reconstruct an s-expression string from token IDs."""
        tokens = self.decode(ids)
        parts = []
        in_num = False
        for t in tokens:
            if t == "NUM_SEP":
                in_num = not in_num
                continue
            if t in ("SEP_TASK", "SEP_EXPR", "SEP_FEEDBACK", "SEP_EDIT",
                     "SEP_QUERY", "SEP_QUERY_RESULT", "SEP_QUERY_OUT",
                     "<query>", "</query>", "<q_out>", "</q_out>",
                     "NO_OP", "CORRECT", "WRONG", "ERROR", "INCOMPLETE"):
                continue
            if t == "(":
                parts.append("(")
            elif t == ")":
                # Remove trailing space before close paren
                if parts and parts[-1] == " ":
                    parts.pop()
                parts.append(")")
            elif t == "_HOLE_":
                parts.append("_HOLE_")
            elif in_num:
                parts.append(t)
            elif t in GEO_FUNCTIONS:
                parts.append(t)
                parts.append(" ")
            else:
                # Entity name — quote it
                parts.append(f'"{t}"')
            parts.append(" ")

        result = "".join(parts).strip()
        # Clean up spacing
        result = re.sub(r"\(\s+", "(", result)
        result = re.sub(r"\s+\)", ")", result)
        result = re.sub(r"\s+", " ", result)
        return result

    def encode_trace_step(
        self,
        question: str,
        current_expr: str,
        feedback_type: str,
        feedback_value: str = "",
        target_expr: str = "",
    ) -> tuple[list[int], list[int]]:
        """Encode one refinement step as (input_ids, target_ids).

        Input: [BOS TASK_TOKENS SEP_EXPR EXPR_TOKENS SEP_FEEDBACK FEEDBACK]
        Target: [SEP_EDIT TARGET_TOKENS EOS]
        """
        # We don't tokenize the full NL question — just extract the entity names
        # The model sees: function signatures + entity mentions + current expr + feedback
        input_tokens = []

        # Current expression
        input_tokens.append("SEP_EXPR")
        if current_expr and current_expr != "_HOLE_":
            input_tokens.extend(self.tokenize_sexpr(current_expr))
        else:
            input_tokens.append("_HOLE_")

        # Feedback
        input_tokens.append("SEP_FEEDBACK")
        input_tokens.append(feedback_type)  # CORRECT, WRONG, ERROR, INCOMPLETE
        if feedback_value:
            input_tokens.extend(self.tokenize_number(feedback_value))

        input_ids = self.encode(input_tokens, add_bos=True, add_eos=False)

        # Target: the next expression (or NO_OP if done)
        target_tokens = ["SEP_EDIT"]
        if target_expr == "NO_OP":
            target_tokens.append("NO_OP")
        elif target_expr:
            target_tokens.extend(self.tokenize_sexpr(target_expr))

        target_ids = self.encode(target_tokens, add_bos=False, add_eos=True)

        return input_ids, target_ids


def build_default_tokenizer() -> SexprTokenizer:
    """Build tokenizer from the geo KB."""
    kb_path = Path(__file__).parent.parent / "data" / "geo_kb.json"
    return SexprTokenizer(str(kb_path))


if __name__ == "__main__":
    tok = build_default_tokenizer()
    print(f"Vocabulary size: {tok.vocab_size}")

    # Test round-trip
    expr = '(population (capital "Spain"))'
    tokens = tok.tokenize_sexpr(expr)
    print(f"Expression: {expr}")
    print(f"Tokens: {tokens}")
    ids = tok.encode(tokens)
    print(f"IDs: {ids}")
    decoded = tok.decode_to_sexpr(ids)
    print(f"Decoded: {decoded}")
