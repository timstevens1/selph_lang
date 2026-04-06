"""Persistent curriculum state for SELPH.

Bundles library (macros + components), synthesis logs, and metadata
into a single directory that survives across runs.

Directory structure:
  state_dir/
    library.selph      — promoted macros (loadable SELPH source)
    components.json     — component metadata (types, priorities)
    logs.selph          — synthesis logs in SELPH format
    logs.json           — synthesis logs in JSON (for analysis)
    meta.json           — run metadata (stages completed, etc.)
"""

from __future__ import annotations
import json
import os
from typing import Any
from .library import save_library, load_library, Abstraction
from .logger import SynthesisLogger
from .eval import Env, standard_env
from .parser import parse
from .types import TVar


class CurriculumState:
    """Persistent state for a growing curriculum."""

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        self.logger = SynthesisLogger()
        self.macros: list[dict] = []          # for Rust fast_synthesize
        self.components: list[dict] = []      # for Rust fast_synthesize
        self.abstractions: list[Abstraction] = []  # for Python synthesize
        self.metadata: dict = {
            "runs_completed": 0,
            "total_tasks": 0,
            "total_solved": 0,
            "stages_completed": [],
        }

    def ensure_dir(self):
        os.makedirs(self.state_dir, exist_ok=True)

    # ── Save ─────────────────────────────────────────────────────────

    def save(self):
        """Persist everything to disk."""
        self.ensure_dir()

        # Library (macros as SELPH source)
        if self.abstractions:
            save_library(self.abstractions,
                        os.path.join(self.state_dir, "library.selph"))

        # Components (JSON for Rust)
        with open(os.path.join(self.state_dir, "components.json"), "w") as f:
            json.dump(self.components, f, indent=2)

        # Macros (JSON with body as source string for Rust)
        macro_data = []
        for m in self.macros:
            macro_data.append({
                "name": m["name"],
                "params": m["params"],
                "body_source": repr(m["body"]) if hasattr(m["body"], 'elements') else str(m["body"]),
            })
        with open(os.path.join(self.state_dir, "macros.json"), "w") as f:
            json.dump(macro_data, f, indent=2)

        # Logs
        self.logger.save(os.path.join(self.state_dir, "logs.json"))
        self.logger.save_selph(os.path.join(self.state_dir, "logs.selph"))

        # Metadata
        with open(os.path.join(self.state_dir, "meta.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

    # ── Load ─────────────────────────────────────────────────────────

    @classmethod
    def load(cls, state_dir: str, env: Env | None = None) -> 'CurriculumState':
        """Load state from disk. Returns a new state if directory doesn't exist."""
        state = cls(state_dir)

        if not os.path.exists(state_dir):
            return state

        # Library
        lib_path = os.path.join(state_dir, "library.selph")
        if os.path.exists(lib_path):
            if env is None:
                env = standard_env()
            state.abstractions = load_library(lib_path, env)

        # Components
        comp_path = os.path.join(state_dir, "components.json")
        if os.path.exists(comp_path):
            with open(comp_path) as f:
                state.components = json.load(f)

        # Macros
        macro_path = os.path.join(state_dir, "macros.json")
        if os.path.exists(macro_path):
            with open(macro_path) as f:
                macro_data = json.load(f)
            for m in macro_data:
                try:
                    body_node = parse(m["body_source"])
                    state.macros.append({
                        "name": m["name"],
                        "params": m["params"],
                        "body": body_node,
                    })
                except Exception:
                    pass

        # Logs
        log_path = os.path.join(state_dir, "logs.json")
        if os.path.exists(log_path):
            state.logger = SynthesisLogger.load(log_path)

        # Metadata
        meta_path = os.path.join(state_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                state.metadata = json.load(f)

        return state

    # ── Library management ───────────────────────────────────────────

    def add_macro(self, name: str, params: list[str], body_source: str):
        """Add a promoted macro to the state."""
        try:
            body_node = parse(body_source)
        except Exception:
            return

        self.macros.append({
            "name": name,
            "params": params,
            "body": body_node,
        })

        self.abstractions.append(Abstraction(
            name=name,
            params=params,
            body=body_node,
            param_types=[TVar(0)] * len(params),
            return_type=TVar(0),
            frequency=1,
            compression=0.0,
        ))

    def add_component(self, comp: dict):
        """Add a component definition to the state."""
        self.components.append(comp)

    def promote_solution(self, task_name: str, source: str,
                         ret_type: int = 0, param_types: list[int] | None = None):
        """Promote a solved program as a library macro + component."""
        macro_name = f"lib_{task_name}"

        # Extract body from (lambda (x) body)
        body_source = source
        if body_source.startswith("(lambda ("):
            # Remove lambda wrapper
            inner = body_source[len("(lambda ("):]
            # Find end of params
            paren_end = inner.index(")")
            params_str = inner[:paren_end]
            body_source = inner[paren_end + 2:-1]  # skip ") " and trailing ")"
            params = params_str.split()
        else:
            params = ["x"]

        self.add_macro(macro_name, params, body_source)
        self.add_component({
            "name": macro_name,
            "builtin": macro_name,
            "arity": len(params),
            "ret_type": ret_type,
            "param_types": param_types or [1] * len(params),
            "priority": 30.0,
        })

    # ── Stats ────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Curriculum State: {self.state_dir}",
            f"  Runs completed: {self.metadata.get('runs_completed', 0)}",
            f"  Library: {len(self.components)} components, {len(self.macros)} macros",
            f"  Logs: {len(self.logger.logs)} entries",
            f"  Solved: {len(self.logger.solved)}/{len(self.logger.logs)}"
            + (f" ({self.logger.solve_rate:.0%})" if self.logger.logs else ""),
        ]
        if self.metadata.get("stages_completed"):
            lines.append(f"  Stages: {self.metadata['stages_completed']}")
        return "\n".join(lines)
