"""Knowledge Base domain adapter.

Wraps the broad Wikidata KB's character-level tokenizer, persistent SELPH
server evaluation, apropos queries, and tool-trace generation behind the
Domain protocol.

Now benefits from the shared training infrastructure:
- Proper GRPO with reference model, clipped ratio, KL penalty
- Tournament multi-round collection with ratcheting
- Iterative refinement (was single-pass, now multi-step)
- Test-validated reward
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import Domain, RefinementStep, RefinementTrace

# ── KB Task ──────────────────────────────────────────────────────────────────

@dataclass
class KBTask:
    """A knowledge base query task."""
    task_id: str
    prompt: str          # NL question
    answer: str          # expected answer string
    depth: int           # composition depth (1-4)
    completion: str = "" # reference s-expression (for SFT traces)


# ── Vocabulary ───────────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>",
    "(", ")", '"',
    "_HOLE_", "NO_OP",
    "SEP_TASK", "SEP_EXPR", "SEP_FEEDBACK", "SEP_EDIT",
    "<query>", "</query>", "<q_out>", "</q_out>",
    "CORRECT", "WRONG", "ERROR", "INCOMPLETE",
    "NUM_SEP",
    # KB builtins
    "kb-get", "kb-apply", "kb-filter", "kb-is-property",
    "kb-properties", "kb-search", "kb-count", "kb-path", "kb-path-count",
    # Arithmetic builtins
    "add", "subtract", "multiply", "divide", "power", "sqrt",
    "abs", "negate", "log", "exp", "sin", "cos", "tan",
    "floor", "round", ">", "<", "=",
    # List ops
    "reduce", "map", "filter", "range", "list", "head", "tail",
    # Control
    "if", "lambda", "let", "define",
    # Discovery
    "apropos",
    # Digits for numbers
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-",
    # Question type tokens
    "Q_depth1", "Q_depth2", "Q_depth3", "Q_depth4",
]

_TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()]+')


def _build_vocab(common_tokens_path: Path | None = None) -> tuple[dict[str, int], dict[int, str]]:
    """Build the character-level vocabulary with special tokens."""
    vocab = list(SPECIAL_TOKENS)

    # Character tokens for entity names
    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        vocab.append(f"C_{c}")
    for c in " _-',;:!?/&@#%+":
        vocab.append(f"C_{c}")
    vocab.append("C_UNK")

    # Common multi-char tokens from training data
    if common_tokens_path and common_tokens_path.exists():
        common = json.loads(common_tokens_path.read_text())
        vocab.extend(common)

    tok2id = {}
    id2tok = {}
    for tok in vocab:
        if tok not in tok2id:
            tok2id[tok] = len(tok2id)
            id2tok[len(id2tok)] = tok

    return tok2id, id2tok


# ── SELPH Server ─────────────────────────────────────────────────────────────

class _SelphServer:
    """Persistent SELPH evaluation server wrapper."""

    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.proc = None

    def start(self):
        import subprocess, sys
        selph_bin = str(Path(__file__).parent.parent.parent.parent
                        / "selph_fast" / "target" / "release" / "selph")
        cmd = [selph_bin, "serve", "--kb", self.kb_path]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        while True:
            line = self.proc.stdout.readline().strip()
            if line == "READY":
                break
            if not line and self.proc.poll() is not None:
                stderr = self.proc.stderr.read()
                raise RuntimeError(f"Server failed: {stderr}")
            if line:
                print(f"  [selph] {line}", flush=True)

    def eval(self, expr: str) -> str | None:
        if self.proc is None or self.proc.poll() is not None:
            return None
        self.proc.stdin.write(expr.strip() + "\n")
        self.proc.stdin.flush()
        result = self.proc.stdout.readline().strip()
        if result.startswith("ERROR:"):
            return None
        return result if result else None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(":quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None

    def __del__(self):
        self.stop()


# ── KB Domain ────────────────────────────────────────────────────────────────

class KBDomain:
    """Domain adapter for Wikidata knowledge base queries.

    Implements the Domain protocol with:
    - Character-level tokenizer for open vocabulary entity names
    - Persistent SELPH server for fast KB evaluation
    - Apropos queries for function discovery
    - Iterative refinement (new! — was single-pass in grpo_broad.py)

    The key upgrade over the old grpo_broad.py: this now uses tournament GRPO
    with proper reference model, clipped ratio, KL penalty, and multi-round
    ratcheting — the same infrastructure as the ARC domain.
    """

    def __init__(self, data_dir: str | Path,
                 kb_path: str | Path | None = None,
                 common_tokens_path: str | Path | None = None):
        self._data_dir = Path(data_dir)
        self._kb_path = str(kb_path or self._data_dir / "wikidata_resolved_with_functions.jsonl")

        # Build tokenizer
        ct_path = Path(common_tokens_path) if common_tokens_path else (self._data_dir / "broad_common_tokens.json")
        self._tok2id, self._id2tok = _build_vocab(ct_path if ct_path.exists() else None)

        # Token IDs
        self.vocab_size = len(self._tok2id)
        self.pad_id = self._tok2id["<pad>"]
        self.bos_id = self._tok2id["<bos>"]
        self.eos_id = self._tok2id["<eos>"]
        self.sep_task_id = self._tok2id["SEP_TASK"]
        self.sep_expr_id = self._tok2id["SEP_EXPR"]
        self.sep_feedback_id = self._tok2id["SEP_FEEDBACK"]
        self.sep_edit_id = self._tok2id["SEP_EDIT"]
        self.query_open_id = self._tok2id["<query>"]
        self.query_close_id = self._tok2id["</query>"]
        self.qout_open_id = self._tok2id["<q_out>"]
        self.qout_close_id = self._tok2id["</q_out>"]
        self.hole_id = self._tok2id["_HOLE_"]
        self.noop_id = self._tok2id["NO_OP"]
        self.feedback_token_ids = {
            "CORRECT": self._tok2id["CORRECT"],
            "WRONG": self._tok2id["WRONG"],
            "ERROR": self._tok2id["ERROR"],
            "INCOMPLETE": self._tok2id["INCOMPLETE"],
        }

        self._server: _SelphServer | None = None
        self._tasks: list[KBTask] | None = None

    def _get_server(self) -> _SelphServer:
        if self._server is None:
            self._server = _SelphServer(self._kb_path)
            print("Starting SELPH server (loading KB)...", flush=True)
            self._server.start()
            print("Server ready.", flush=True)
        return self._server

    def _selph_eval(self, expr: str) -> str | None:
        return self._get_server().eval(expr)

    # ── Tokenization ─────────────────────────────────────────────────

    def _tokenize_string(self, s: str) -> list[int]:
        """Character-level tokenization for entity names."""
        ids = []
        for c in s:
            key = f"C_{c}"
            ids.append(self._tok2id.get(key, self._tok2id["C_UNK"]))
        return ids

    def _tokenize_sexpr_tokens(self, expr: str) -> list[str]:
        """Tokenize an s-expression into string tokens."""
        tokens = []
        i = 0
        expr = expr.strip()
        while i < len(expr):
            c = expr[i]
            if c in " \t\n\r":
                i += 1; continue
            if c == "(":
                tokens.append("("); i += 1
            elif c == ")":
                tokens.append(")"); i += 1
            elif c == '"':
                j = i + 1
                while j < len(expr) and expr[j] != '"':
                    if expr[j] == '\\': j += 1
                    j += 1
                entity = expr[i+1:j]
                tokens.append('"')
                for word in entity.replace("-", " - ").replace("_", " _ ").split():
                    if word in self._tok2id:
                        tokens.append(word)
                    else:
                        for ch in word:
                            tokens.append(f"C_{ch}" if f"C_{ch}" in self._tok2id else "C_UNK")
                tokens.append('"')
                i = j + 1
            elif c == "_" and expr[i:].startswith("_HOLE_"):
                tokens.append("_HOLE_"); i += 6
            elif c.isdigit() or (c == "-" and i+1 < len(expr) and expr[i+1].isdigit()):
                tokens.append("NUM_SEP")
                while i < len(expr) and expr[i] in "0123456789.-eE+":
                    tokens.append(expr[i]); i += 1
                tokens.append("NUM_SEP")
            else:
                j = i
                while j < len(expr) and expr[j] not in ' \t\n\r()"':
                    j += 1
                word = expr[i:j]
                if word in self._tok2id:
                    tokens.append(word)
                else:
                    for ch in word:
                        tokens.append(f"C_{ch}" if f"C_{ch}" in self._tok2id else "C_UNK")
                i = j
        return tokens

    def tokenize_program(self, sexpr: str) -> list[int]:
        if sexpr == "_HOLE_":
            return [self.hole_id]
        if sexpr == "NO_OP":
            return [self.noop_id]
        # Try special tokens first
        if sexpr in self._tok2id:
            return [self._tok2id[sexpr]]
        # For short results like "true", "false", "nil", numbers
        if not any(c in sexpr for c in "()\""):
            # Simple value — tokenize as string
            return self._tokenize_string(sexpr)
        # Full s-expression
        str_tokens = self._tokenize_sexpr_tokens(sexpr)
        return [self._tok2id[t] for t in str_tokens if t in self._tok2id]

    def detokenize_program(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id,
                self.sep_task_id, self.sep_expr_id,
                self.sep_feedback_id, self.sep_edit_id,
                self.query_open_id, self.query_close_id,
                self.qout_open_id, self.qout_close_id}
        tokens = [self._id2tok.get(i, "?") for i in ids if i not in skip]
        # Reconstruct from token stream
        parts = []
        in_num = False
        in_string = False
        for t in tokens:
            if t == "NO_OP":
                return "NO_OP"
            if t == "_HOLE_":
                parts.append("_HOLE_ "); continue
            if t == "NUM_SEP":
                if in_num:
                    # Closing NUM_SEP — add space after the number
                    parts.append(" ")
                in_num = not in_num
                continue
            if t in ("CORRECT", "WRONG", "ERROR", "INCOMPLETE"):
                continue
            if t == '"':
                if not in_string:
                    # Opening quote — add space before if needed
                    if parts and parts[-1] not in ("(", " ", ""):
                        parts.append(" ")
                    parts.append('"')
                    in_string = True
                else:
                    # Closing quote — add space after
                    parts.append('"')
                    in_string = False
                    parts.append(" ")
                continue
            if t.startswith("C_"):
                c = t[2:]
                parts.append("?" if c == "UNK" else c)
                continue
            if t == "(":
                parts.append("(")
            elif t == ")":
                if parts and parts[-1] == " ": parts.pop()
                parts.append(")")
            elif in_num:
                parts.append(t)
            elif in_string:
                parts.append(t)
            else:
                parts.append(t)
            if not in_string and not in_num:
                parts.append(" ")

        result = "".join(parts).strip()
        result = re.sub(r"\(\s+", "(", result)
        result = re.sub(r"\s+\)", ")", result)
        result = re.sub(r"\s+", " ", result)
        return result

    # ── Task Encoding ────────────────────────────────────────────────

    def encode_task_prefix(self, task: KBTask) -> list[int]:
        """Encode question type + NL text as tokens."""
        ids = []
        q_type = f"Q_depth{min(task.depth, 4)}"
        if q_type in self._tok2id:
            ids.append(self._tok2id[q_type])
        # NL question text (character-level)
        nl = task.prompt.split("Q: ")[1].split("\n")[0] if "Q: " in task.prompt else task.prompt
        ids.extend(self._tokenize_string(nl))
        return ids

    # ── Evaluation ───────────────────────────────────────────────────

    def evaluate(self, program: str, task: KBTask,
                 include_test: bool = True) -> tuple[str, float]:
        if not program or "_HOLE_" in program:
            return ("INCOMPLETE", 0.0)

        # Check balanced parens
        depth = 0
        for c in program:
            if c == "(": depth += 1
            elif c == ")": depth -= 1
            if depth < 0:
                return ("ERROR", 0.0)
        if depth != 0:
            return ("ERROR", 0.0)

        # Parse OK — try evaluation
        result = self._selph_eval(program)
        if result is None:
            return ("ERROR", 0.1)  # parses but fails to eval

        # Check correctness
        result_clean = result.strip().strip('"')
        expected = str(task.answer).strip().strip('"')

        if result_clean == expected:
            return ("CORRECT", 1.0)
        try:
            if abs(float(result_clean) - float(expected)) < max(abs(float(expected)) * 0.01, 0.1):
                return ("CORRECT", 1.0)
        except (ValueError, ZeroDivisionError):
            pass
        if result_clean.lower() == expected.lower():
            return ("CORRECT", 1.0)

        # Wrong answer but ran successfully — partial credit
        return ("WRONG", 0.3)

    def shaped_reward(self, feedback_type: str, accuracy: float,
                      program: str = "", task: KBTask | None = None) -> float:
        """Shaped reward with entity-grounding partial credit.

        Tiers:
            1.0  — correct answer
            0.3  — evaluates to wrong answer
            0.1  — parses but errors on eval
            0.0  — empty/holes/unbalanced

        Entity bonus (additive, up to +0.15):
            Programs that reference entities or properties mentioned in the
            NL prompt get partial credit even if the final answer is wrong.
            This rewards the model for learning to ground prompt entities
            into kb-get calls and apropos queries.
        """
        base = 0.0
        if feedback_type == "CORRECT":
            return 1.0
        elif feedback_type == "WRONG":
            base = 0.3
        elif feedback_type == "ERROR" and accuracy > 0:
            base = accuracy  # 0.1 for parses-but-fails

        # Entity grounding bonus
        if program and task:
            base += self._entity_grounding_bonus(program, task)

        return min(base, 0.95)  # cap below perfect to preserve CORRECT signal

    def _entity_grounding_bonus(self, program: str, task: KBTask) -> float:
        """Partial credit for using prompt entities in the generated program.

        Extracts entity names and property names from the NL prompt,
        checks which appear as quoted strings in the program.

        For "What is the population of the capital of France?":
          - "France" in program → +0.05
          - "capital" in program → +0.05
          - "population" in program → +0.05
          - apropos used → +0.05

        Max bonus: 0.15 (capped to avoid overshadowing correctness).
        """
        nl = task.prompt.split("Q: ")[1].split("\n")[0] if "Q: " in task.prompt else task.prompt
        nl_lower = nl.lower()

        # Extract candidate entities/properties from the NL prompt
        # Strategy: find quoted strings in the reference completion (ground truth)
        # and check if they appear in the NL prompt
        candidates = set()

        # Extract from reference completion if available
        for m in re.finditer(r'"([^"]+)"', task.completion):
            entity = m.group(1)
            candidates.add(entity)

        # Also extract likely entities by matching capitalized phrases or
        # known property patterns from the NL question
        property_words = {
            "population", "capital", "country", "area", "continent",
            "author", "director", "composer", "birthplace", "place_of_birth",
            "place_of_death", "birth_date", "death_date", "nationality",
            "genre", "language", "currency", "elevation", "coordinates",
            "defining_formula", "discoverer", "inventor", "founder",
            "mass", "density", "symbol", "atomic_number", "boiling_point",
            "melting_point", "population_density",
        }
        for prop in property_words:
            # Check if the property concept appears in the NL question
            prop_readable = prop.replace("_", " ")
            if prop_readable in nl_lower or prop in nl_lower:
                candidates.add(prop)

        if not candidates:
            return 0.0

        # Count how many candidates appear as quoted strings in the program
        program_strings = set()
        for m in re.finditer(r'"([^"]+)"', program):
            program_strings.add(m.group(1))

        # Also count unquoted property-like references (kb-get args)
        matches = 0
        for cand in candidates:
            if cand in program_strings:
                matches += 1
            elif cand.lower() in program.lower():
                matches += 0.5  # partial for unquoted reference

        # Bonus for using apropos (function discovery)
        apropos_bonus = 0.05 if "apropos" in program else 0.0

        # Scale: each entity match worth 0.05, cap at 0.15
        entity_bonus = min(matches * 0.05, 0.10)
        return min(entity_bonus + apropos_bonus, 0.15)

    # ── Query Handling ───────────────────────────────────────────────

    # ── Accuracy Tokens ──────────────────────────────────────────────

    def accuracy_tokens(self, accuracy: float) -> list[int]:
        """KB domain doesn't use accuracy tokens in prompts."""
        return []

    # ── Query Handling ───────────────────────────────────────────────

    def handle_query(self, query_ids: list[int], task: KBTask) -> list[int]:
        """Evaluate a query via the SELPH server and return result tokens."""
        query_sexpr = self.detokenize_program(query_ids)
        if not query_sexpr or "_HOLE_" in query_sexpr:
            return self._tokenize_string("nil")

        result = self._selph_eval(query_sexpr)
        if result:
            return self._tokenize_string(result[:100])
        return self._tokenize_string("nil")

    # ── Task Access ──────────────────────────────────────────────────

    @property
    def tasks(self) -> list[KBTask]:
        if self._tasks is None:
            self._load_tasks()
        return self._tasks

    def _load_tasks(self):
        train_path = self._data_dir / "broad_train.json"
        if not train_path.exists():
            self._tasks = []
            return
        examples = json.loads(train_path.read_text())
        self._tasks = [
            KBTask(
                task_id=f"kb_{i}",
                prompt=ex.get("prompt", ""),
                answer=str(ex.get("answer", "")),
                depth=ex.get("depth", 1),
                completion=ex.get("completion", ""),
            )
            for i, ex in enumerate(examples)
        ]

    # ── Trace Generation ─────────────────────────────────────────────

    def generate_traces(self) -> list[RefinementTrace]:
        """Generate iterative refinement traces from the KB training data.

        NEW: now generates multi-step traces with apropos queries,
        matching the ARC domain's iterative refinement pattern.
        Previously these were single-pass CoT traces.
        """
        traces = []
        tasks = self.tasks
        random.shuffle(tasks)

        for task in tasks:
            completion = task.completion.strip()
            if not completion:
                continue

            # Clean completion (remove tool_call tags from old format)
            completion = completion.replace("<tool_call>", "").replace("</tool_call>", "").strip()

            # Extract the final program from completion
            parts = completion.split("SEP_EDIT")
            if len(parts) == 2:
                tool_part = parts[0].strip()
                program = parts[1].strip()
            else:
                tool_part = ""
                program = completion.replace("SEP_EDIT", "").strip()

            if not program:
                continue

            # Build multi-step trace
            steps = []

            # Step 1: _HOLE_ → queries → skeleton/program
            queries = self._extract_tool_queries(tool_part)
            skeleton = self._extract_skeleton(program)
            steps.append(RefinementStep(
                current_expr="_HOLE_",
                feedback_type="INCOMPLETE",
                feedback_acc=0.0,
                target_expr=skeleton if skeleton != program else program,
                queries=queries,
            ))

            # Step 2: skeleton → full program (if different)
            if skeleton != program:
                steps.append(RefinementStep(
                    current_expr=skeleton,
                    feedback_type="INCOMPLETE",  # skeleton has holes
                    feedback_acc=0.0,
                    target_expr=program,
                ))

            # Step 3: program → CORRECT → NO_OP
            steps.append(RefinementStep(
                current_expr=program,
                feedback_type="CORRECT",
                feedback_acc=1.0,
                target_expr="NO_OP",
            ))

            traces.append(RefinementTrace(task=task, steps=steps))

        return traces

    def _extract_tool_queries(self, tool_part: str) -> list[tuple[str, str]]:
        """Extract (query_expr, result) pairs from tool call section."""
        queries = []
        for m in re.finditer(r'<query>(.*?)</query><q_out>(.*?)</q_out>', tool_part):
            queries.append((m.group(1), m.group(2)[:100]))
        return queries

    def _extract_skeleton(self, sexpr: str) -> str:
        """Replace inner arguments with _HOLE_ for 1-level skeleton."""
        s = sexpr.strip()
        if not s.startswith("("):
            return s
        depth = 0
        func_end = None
        args = []
        arg_start = None
        for i, c in enumerate(s):
            if c == "(":
                depth += 1
                if depth == 1:
                    func_end = None; arg_start = None
            elif c == ")":
                depth -= 1
                if depth == 0 and arg_start is not None:
                    args.append(s[arg_start:i].strip())
            elif c == " " and depth == 1:
                if func_end is None:
                    func_end = i; arg_start = i + 1
                else:
                    if arg_start is not None:
                        token = s[arg_start:i].strip()
                        if token: args.append(token)
                    arg_start = i + 1
        if func_end is None:
            return s
        func_name = s[1:func_end]
        skeleton_args = []
        for arg in args:
            arg = arg.strip()
            if not arg: continue
            if arg.startswith("(") or arg.startswith('"'):
                skeleton_args.append("_HOLE_")
            else:
                skeleton_args.append(arg)
        if not skeleton_args:
            return s
        return "(" + func_name + " " + " ".join(skeleton_args) + ")"

    def stop(self):
        """Stop the SELPH evaluation server."""
        if self._server:
            self._server.stop()
            self._server = None
