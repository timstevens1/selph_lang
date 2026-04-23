"""Core abstractions for the training library.

Domain protocol: the interface that ARC, KB, and fitting domains implement.
Data classes: RefinementStep, RefinementTrace, Trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RefinementStep:
    """One step in an iterative refinement trace."""
    current_expr: str
    feedback_type: str      # INCOMPLETE, WRONG, ERROR, CORRECT
    feedback_acc: float     # accuracy [0, 1]
    target_expr: str        # the edit (corrected program or NO_OP)
    queries: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class RefinementTrace:
    """A full refinement trajectory for one task."""
    task: Any               # domain-specific task object
    steps: list[RefinementStep]


@dataclass
class Trajectory:
    """A collected trajectory for GRPO training."""
    task: Any
    steps: list[tuple[list[int], list[int], list[float]]]  # (prompt, gen_ids, log_probs)
    reward: float
    final_expr: str = "_HOLE_"


@runtime_checkable
class Domain(Protocol):
    """Interface that each domain (ARC, KB, fitting) must implement.

    The training core calls these methods; domain-specific details are
    encapsulated behind this interface.
    """

    # ── Token IDs ────────────────────────────────────────────────────
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    sep_task_id: int
    sep_expr_id: int
    sep_feedback_id: int
    sep_edit_id: int
    query_open_id: int
    query_close_id: int
    qout_open_id: int
    qout_close_id: int
    hole_id: int
    noop_id: int

    # Feedback type token IDs
    feedback_token_ids: dict[str, int]  # {"CORRECT": id, "WRONG": id, ...}

    # ── Tokenization ─────────────────────────────────────────────────

    def tokenize_program(self, sexpr: str) -> list[int]:
        """Tokenize a SELPH program body into token IDs (no BOS/EOS)."""
        ...

    def detokenize_program(self, ids: list[int]) -> str:
        """Convert token IDs back to a SELPH program string."""
        ...

    # ── Task Encoding ────────────────────────────────────────────────

    def encode_task_prefix(self, task: Any) -> list[int]:
        """Encode task-specific features as a token prefix.

        Returns tokens between SEP_TASK and SEP_EXPR — e.g. grid features
        for ARC, or question type + NL tokens for KB.
        """
        ...

    # ── Evaluation ───────────────────────────────────────────────────

    def evaluate(self, program: str, task: Any,
                 include_test: bool = True) -> tuple[str, float]:
        """Evaluate a program on a task.

        Returns (feedback_type, accuracy) where feedback_type is one of
        INCOMPLETE, ERROR, WRONG, CORRECT and accuracy is in [0, 1].
        """
        ...

    def shaped_reward(self, feedback_type: str, accuracy: float,
                      program: str = "", task: Any = None) -> float:
        """Convert (feedback_type, accuracy) to a scalar reward.

        Domains define their own reward shaping. ARC uses 5.0 for perfect,
        acc**2 otherwise. KB uses 0/0.1/0.3/1.0 tiers with entity grounding bonus.
        program and task are optional — domains can use them for context-aware shaping.
        """
        ...

    # ── Query Handling ───────────────────────────────────────────────

    def handle_query(self, query_ids: list[int], task: Any) -> list[int]:
        """Evaluate a scratchpad query and return result as token IDs.

        Called when the model emits <query>...</query>. Returns the token
        IDs to inject between <q_out> and </q_out>.
        """
        ...

    # ── Accuracy Encoding (optional) ─────────────────────────────────

    def accuracy_tokens(self, accuracy: float) -> list[int]:
        """Encode accuracy as token(s) for the feedback section.

        Default: empty (no accuracy encoding). ARC overrides with acc0-acc10.
        """
        return []

    # ── Trace Generation ─────────────────────────────────────────────

    def generate_traces(self) -> list[RefinementTrace]:
        """Generate all training traces for this domain."""
        ...

    # ── Task Access ──────────────────────────────────────────────────

    @property
    def tasks(self) -> list[Any]:
        """All tasks available for training/eval."""
        ...


def trace_step_to_ids(domain: Domain, step: RefinementStep, task: Any) -> list[int]:
    """Convert a refinement step to a token sequence using the domain's tokenizer.

    Format: [BOS] SEP_TASK [features] SEP_EXPR [current] SEP_FEEDBACK [type] [acc?]
            [<query>expr</query><q_out>result</q_out>]* SEP_EDIT [target] [EOS]
    """
    ids = [domain.bos_id]

    # Task features
    ids.append(domain.sep_task_id)
    ids.extend(domain.encode_task_prefix(task))

    # Current expression
    ids.append(domain.sep_expr_id)
    ids.extend(domain.tokenize_program(step.current_expr))

    # Feedback
    ids.append(domain.sep_feedback_id)
    fb_id = domain.feedback_token_ids.get(step.feedback_type)
    if fb_id is not None:
        ids.append(fb_id)
    if step.feedback_type == "WRONG":
        ids.extend(domain.accuracy_tokens(step.feedback_acc))

    # Queries (scratchpad)
    for query_expr, result_str in step.queries:
        ids.append(domain.query_open_id)
        ids.extend(domain.tokenize_program(query_expr))
        ids.append(domain.query_close_id)
        ids.append(domain.qout_open_id)
        # Domain handles result tokenization via tokenize_program or direct lookup
        ids.extend(domain.tokenize_program(result_str))
        ids.append(domain.qout_close_id)

    # Target edit
    ids.append(domain.sep_edit_id)
    ids.extend(domain.tokenize_program(step.target_expr))

    ids.append(domain.eos_id)
    return ids


def traces_to_training_data(
    domain: Domain,
    traces: list[RefinementTrace],
    max_seq_len: int = 256,
) -> list[tuple[list[int], list[int]]]:
    """Convert traces to (input_ids, target_ids) pairs for next-token prediction."""
    examples = []
    for trace in traces:
        for step in trace.steps:
            ids = trace_step_to_ids(domain, step, trace.task)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            examples.append((ids[:-1], ids[1:]))
    return examples
