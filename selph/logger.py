"""Synthesis logging for SELPH training data generation.

Records every synthesis run as structured data that becomes training
input for meta-curriculum stages:

  Meta-1 (heuristics):  task_features → components_used
  Meta-2 (decomposition): failed_spec → strategy → sub_solutions
  Meta-3 (extraction): corpus → abstractions
  Meta-4 (generation): spec → solution
  Meta-5 (eval strategy): candidates → correct_one

All logs are stored as SELPH-native Namespace values and can be
persisted as JSON for analysis.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any
from .ast import Node, Spec, GoalExamples, Symbol, Number, String
from .eval import Namespace


@dataclass
class SynthesisLog:
    """A single synthesis run record."""
    # Task description
    task_name: str = ""
    input_type: str = ""  # "number", "string"
    output_type: str = ""
    num_examples: int = 0
    example_inputs: list = field(default_factory=list)
    example_outputs: list = field(default_factory=list)

    # Result
    found: bool = False
    source: str = ""
    candidates_explored: int = 0
    candidates_pruned: int = 0
    time_seconds: float = 0.0

    # Components used in solution
    components_used: list[str] = field(default_factory=list)
    solution_depth: int = 0

    # Strategy
    strategy: str = "direct"  # "direct", "induction", "divide_and_conquer"
    decomposition: str = ""  # for induction/D&C: how it was decomposed

    # Meta
    stage: int = 0
    library_size: int = 0

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "num_examples": self.num_examples,
            "found": self.found,
            "source": self.source,
            "candidates_explored": self.candidates_explored,
            "time_seconds": self.time_seconds,
            "components_used": self.components_used,
            "solution_depth": self.solution_depth,
            "strategy": self.strategy,
            "decomposition": self.decomposition,
            "stage": self.stage,
            "library_size": self.library_size,
        }

    def to_namespace(self) -> Namespace:
        return Namespace(self.to_dict())


class SynthesisLogger:
    """Collects synthesis logs across a curriculum run."""

    def __init__(self):
        self.logs: list[SynthesisLog] = []

    def log(self, entry: SynthesisLog):
        self.logs.append(entry)

    @property
    def total_candidates(self) -> int:
        return sum(l.candidates_explored for l in self.logs)

    @property
    def solved(self) -> list[SynthesisLog]:
        return [l for l in self.logs if l.found]

    @property
    def failed(self) -> list[SynthesisLog]:
        return [l for l in self.logs if not l.found]

    @property
    def solve_rate(self) -> float:
        return len(self.solved) / len(self.logs) if self.logs else 0.0

    # ── Training data extraction ─────────────────────────────────────

    def meta1_data(self) -> list[dict]:
        """Training data for Meta-1 (heuristics).

        Each entry: {task_features, components_used, candidates}
        The model should learn: given task features, predict useful components.
        """
        data = []
        for log in self.solved:
            data.append({
                "input_type": log.input_type,
                "output_type": log.output_type,
                "num_examples": log.num_examples,
                "components_used": log.components_used,
                "candidates": log.candidates_explored,
                "solution_depth": log.solution_depth,
            })
        return data

    def meta2_data(self) -> list[dict]:
        """Training data for Meta-2 (decomposition).

        Each entry: {spec_features, strategy, decomposition}
        Only includes tasks solved by induction or D&C.
        """
        data = []
        for log in self.solved:
            if log.strategy != "direct":
                data.append({
                    "input_type": log.input_type,
                    "output_type": log.output_type,
                    "num_examples": log.num_examples,
                    "strategy": log.strategy,
                    "decomposition": log.decomposition,
                    "source": log.source,
                })
        return data

    def meta4_data(self) -> list[dict]:
        """Training data for Meta-4 (program generation).

        Each entry: {example_inputs, example_outputs, solution}
        The model should learn: given examples, generate a program.
        """
        data = []
        for log in self.solved:
            data.append({
                "inputs": log.example_inputs,
                "outputs": log.example_outputs,
                "solution": log.source,
                "input_type": log.input_type,
                "output_type": log.output_type,
            })
        return data

    def component_frequency(self) -> dict[str, int]:
        """How often each component appears in solutions."""
        freq: dict[str, int] = {}
        for log in self.solved:
            for comp in log.components_used:
                freq[comp] = freq.get(comp, 0) + 1
        return freq

    def difficulty_distribution(self) -> list[tuple[str, int]]:
        """Tasks sorted by candidates needed (difficulty)."""
        return [(l.task_name, l.candidates_explored)
                for l in sorted(self.solved, key=lambda l: l.candidates_explored)]

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str):
        """Save all logs as JSON."""
        data = [l.to_dict() for l in self.logs]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_selph(self, path: str):
        """Save all logs as a SELPH source file.

        Each log entry becomes a quoted namespace literal.
        The file can be loaded back with load_selph().
        """
        with open(path, "w") as f:
            f.write("; SELPH synthesis log\n")
            f.write(f"; {len(self.logs)} entries\n\n")
            for log in self.logs:
                comps = " ".join(f'"{c}"' for c in log.components_used)
                inputs = " ".join(f'"{i}"' for i in log.example_inputs[:5])
                outputs = " ".join(f'"{o}"' for o in log.example_outputs[:5])
                f.write(f'(log\n')
                f.write(f'  (task-name "{log.task_name}")\n')
                f.write(f'  (input-type "{log.input_type}")\n')
                f.write(f'  (output-type "{log.output_type}")\n')
                f.write(f'  (num-examples {log.num_examples})\n')
                f.write(f'  (found {"true" if log.found else "false"})\n')
                f.write(f'  (source "{log.source}")\n')
                f.write(f'  (candidates {log.candidates_explored})\n')
                f.write(f'  (time {log.time_seconds:.4f})\n')
                f.write(f'  (depth {log.solution_depth})\n')
                f.write(f'  (strategy "{log.strategy}")\n')
                f.write(f'  (stage {log.stage})\n')
                f.write(f'  (library-size {log.library_size})\n')
                f.write(f'  (components ({comps}))\n')
                f.write(f'  (inputs ({inputs}))\n')
                f.write(f'  (outputs ({outputs})))\n\n')

    @classmethod
    def load(cls, path: str) -> 'SynthesisLogger':
        """Load logs from JSON."""
        with open(path) as f:
            data = json.load(f)
        logger = cls()
        for d in data:
            log = SynthesisLog(**{k: v for k, v in d.items()
                                  if k in SynthesisLog.__dataclass_fields__})
            logger.logs.append(log)
        return logger

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Synthesis Log: {len(self.logs)} runs",
            f"  Solved: {len(self.solved)}/{len(self.logs)} ({self.solve_rate:.0%})",
            f"  Total candidates: {self.total_candidates:,}",
        ]
        if self.solved:
            avg = self.total_candidates / len(self.logs)
            lines.append(f"  Avg candidates/task: {avg:,.0f}")

        freq = self.component_frequency()
        if freq:
            top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append(f"  Top components: {', '.join(f'{n}({c})' for n, c in top)}")

        strategies = {}
        for log in self.solved:
            strategies[log.strategy] = strategies.get(log.strategy, 0) + 1
        if strategies:
            lines.append(f"  Strategies: {strategies}")

        return "\n".join(lines)


# ── Helper: extract components from a solution AST ───────────────────

def extract_components_from_source(source: str) -> list[str]:
    """Extract component names that appear in a solution source string."""
    from .parser import parse
    from .library import _is_builtin

    try:
        node = parse(source)
    except Exception:
        return []

    components = []
    _walk_for_components(node, components)
    return components


def _walk_for_components(node, components: list[str]):
    """Walk an AST and collect symbol names."""
    if isinstance(node, Symbol):
        if node.name not in ("lambda", "let", "if", "define", "defmacro",
                              "do", "quote", "and", "or", "x"):
            if node.name not in components:
                components.append(node.name)
    elif hasattr(node, 'elements'):
        for elem in node.elements:
            _walk_for_components(elem, components)


def estimate_depth(source: str) -> int:
    """Estimate AST depth from source string."""
    max_depth = 0
    current = 0
    for c in source:
        if c == '(':
            current += 1
            max_depth = max(max_depth, current)
        elif c == ')':
            current -= 1
    # Subtract 1 for the lambda wrapper
    return max(0, max_depth - 1)
