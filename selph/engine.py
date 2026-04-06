"""Unified synthesis engine for SELPH.

Provides a single entry point that dispatches to the best available
backend (Rust or Python) based on task requirements.

Usage:
    from selph.engine import SynthesisEngine

    engine = SynthesisEngine()
    engine.add_macro("double", ["x"], "(add x x)")
    engine.add_component("double", arity=1, ret_type=0, param_types=[0])

    result = engine.solve(inputs, expected, max_depth=2)
    if result.found:
        engine.promote(result, "my_double")
        engine.save("my_library/")

    # Next run:
    engine = SynthesisEngine.load("my_library/")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .parser import parse
from .state import CurriculumState
from .logger import SynthesisLogger, SynthesisLog, extract_components_from_source, estimate_depth


@dataclass
class SolveResult:
    """Unified result from any synthesis backend."""
    found: bool = False
    source: str = ""
    candidates: int = 0
    time_seconds: float = 0.0
    components_used: list[str] = field(default_factory=list)
    depth: int = 0


class SynthesisEngine:
    """Unified interface to SELPH synthesis.

    Manages components, macros, and state. Dispatches to the Rust
    backend when available, falls back to Python.
    """

    def __init__(self, state_dir: str | None = None):
        self.components: list[dict] = []
        self.macros: list[dict] = []
        self.logger = SynthesisLogger()
        self.state_dir = state_dir
        self._rust_available = _check_rust()

    @classmethod
    def load(cls, state_dir: str) -> 'SynthesisEngine':
        """Load engine from a persisted state directory."""
        state = CurriculumState.load(state_dir)
        engine = cls(state_dir)
        engine.components = list(state.components)
        engine.macros = list(state.macros)
        engine.logger = state.logger
        return engine

    def save(self, state_dir: str | None = None):
        """Persist current state."""
        path = state_dir or self.state_dir
        if path is None:
            raise ValueError("No state_dir specified")
        state = CurriculumState(path)
        state.components = self.components
        state.macros = self.macros
        state.logger = self.logger
        # Convert macros to abstractions for library.selph
        for m in self.macros:
            try:
                body_source = repr(m['body']) if hasattr(m['body'], 'elements') else str(m['body'])
                state.add_macro(m['name'], m['params'], body_source)
            except Exception:
                pass
        state.save()

    # ── Component management ─────────────────────────────────────────

    def add_component(self, name: str, arity: int = 1,
                      ret_type: int = 0, param_types: list[int] | None = None,
                      priority: float = 0.0, builtin: str | None = None):
        """Add a component to the search space."""
        self.components.append({
            'name': name,
            'builtin': builtin or name,
            'arity': arity,
            'ret_type': ret_type,
            'param_types': param_types or ([0] * arity if arity > 0 else []),
            'priority': priority,
        })

    def add_constant(self, name: str, ret_type: int = 0, priority: float = 0.0):
        """Add a constant (arity-0 component)."""
        self.components.append({
            'name': name,
            'builtin': None,
            'arity': 0,
            'ret_type': ret_type,
            'param_types': [],
            'priority': priority,
        })

    def add_macro(self, name: str, params: list[str], body_source: str):
        """Add a user-defined macro."""
        try:
            body = parse(body_source)
            self.macros.append({
                'name': name,
                'params': params,
                'body': body,
            })
        except Exception as e:
            raise ValueError(f"Failed to parse macro body: {e}")

    # ── Synthesis ────────────────────────────────────────────────────

    def solve(self, inputs: list, expected: list,
              max_depth: int = 2, max_candidates: int = 100000,
              task_name: str = "") -> SolveResult:
        """Solve a synthesis task."""
        import time
        t0 = time.perf_counter()

        if self._rust_available:
            result = self._solve_rust(inputs, expected, max_depth, max_candidates)
        else:
            result = self._solve_python(inputs, expected, max_depth, max_candidates)

        result.time_seconds = time.perf_counter() - t0

        # Log
        log = SynthesisLog(
            task_name=task_name,
            num_examples=len(inputs),
            found=result.found,
            source=result.source,
            candidates_explored=result.candidates,
            time_seconds=result.time_seconds,
            components_used=result.components_used,
            solution_depth=result.depth,
        )
        if inputs:
            log.example_inputs = [str(i) for i in inputs[:5]]
            log.example_outputs = [str(e) for e in expected[:5]]
        self.logger.log(log)

        return result

    def _solve_rust(self, inputs, expected, max_depth, max_candidates) -> SolveResult:
        import selph_fast
        found, source, explored = selph_fast.fast_synthesize(
            self.components, inputs, expected,
            max_depth, max_candidates,
            self.macros if self.macros else None,
        )
        result = SolveResult(found=found, source=source, candidates=explored)
        if found:
            result.components_used = extract_components_from_source(source)
            result.depth = estimate_depth(source)
        return result

    def _solve_python(self, inputs, expected, max_depth, max_candidates) -> SolveResult:
        from .synthesize import synthesize
        from .ast import Spec, GoalExamples, Symbol, Number, String

        pairs = []
        for i, e in zip(inputs, expected):
            in_node = String(str(i)) if isinstance(i, str) else Number(float(i))
            out_node = String(str(e)) if isinstance(e, str) else Number(float(e))
            pairs.append((in_node, out_node))

        out_type = "string" if isinstance(expected[0], str) else "number"
        spec = Spec(
            type_expr=Symbol(out_type),
            goal=GoalExamples(pairs=tuple(pairs)),
        )
        sr = synthesize(spec, max_depth=max_depth, max_candidates=max_candidates)
        result = SolveResult(
            found=sr.found, source=sr.source or "", candidates=sr.candidates_explored,
        )
        if sr.found and sr.source:
            result.components_used = extract_components_from_source(sr.source)
            result.depth = estimate_depth(sr.source)
        return result

    # ── Promotion ────────────────────────────────────────────────────

    def promote(self, result: SolveResult, name: str,
                ret_type: int = 0, param_types: list[int] | None = None,
                priority: float = 30.0):
        """Promote a solution as a new library macro + component."""
        if not result.found:
            return

        # Extract body from lambda
        source = result.source
        body_source = source
        params = ['x']
        if source.startswith("(lambda ("):
            inner = source[len("(lambda ("):]
            paren_end = inner.index(")")
            params = inner[:paren_end].split()
            body_source = inner[paren_end + 2:-1]

        self.add_macro(name, params, body_source)
        self.add_component(
            name, arity=len(params), ret_type=ret_type,
            param_types=param_types or [1] * len(params),
            priority=priority, builtin=name,
        )

    # ── Interleaved solve ────────────────────────────────────────────

    def solve_suite(self, tasks: list[tuple[str, list, list]],
                    max_depth: int = 2, max_candidates: int = 100000,
                    promote_solutions: bool = True,
                    learn_rate: float = 50.0) -> list[SolveResult]:
        """Solve a suite of tasks with interleaved learning and promotion."""
        results = []
        for name, inputs, expected in tasks:
            result = self.solve(inputs, expected, max_depth, max_candidates,
                               task_name=name)
            if result.found and promote_solutions:
                self.promote(result, f"lib_{name}")
            results.append(result)
        return results

    # ── Info ─────────────────────────────────────────────────────────

    @property
    def library_size(self) -> int:
        return len(self.components)

    def summary(self) -> str:
        lines = [
            f"SynthesisEngine:",
            f"  Components: {len(self.components)}",
            f"  Macros: {len(self.macros)}",
            f"  Backend: {'Rust' if self._rust_available else 'Python'}",
            f"  Log entries: {len(self.logger.logs)}",
        ]
        if self.logger.logs:
            lines.append(f"  Solve rate: {self.logger.solve_rate:.0%}")
        return "\n".join(lines)


def _check_rust() -> bool:
    """Check if the Rust backend is available."""
    try:
        import selph_fast
        return True
    except ImportError:
        return False
