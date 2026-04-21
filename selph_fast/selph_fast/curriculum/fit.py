"""
Fitting curriculum — teaches SELPH to discover model structure and optimize parameters.

The curriculum works like the M-chain: each stage adds a new structural pattern
to the library. Solved models become composable building blocks for harder problems.

Stages:
  1. Constant      — y = c
  2. Linear        — y = wx + b
  3. Polynomial    — y = ax² + bx + c (composes with Stage 2)
  4. Piecewise     — if x > t then model_a else model_b (composes with Stages 1-3)
  5. Model select  — given data, pick the best model family automatically

Each stage:
  1. Defines candidate model templates with (param ?) holes
  2. For each task, tries each template + all library models
  3. Optimizes params via minimize()
  4. Promotes the best-fitting structure to the library
"""

from __future__ import annotations
from dataclasses import dataclass, field
from ..optimize import minimize, Solution
from ..selph_fast import Env, Param
import math


@dataclass
class Task:
    """A fitting task: find f(x) ≈ y for (x, y) pairs."""
    name: str
    data: list[tuple[float, float]]
    test: list[tuple[float, float]] = field(default_factory=list)
    threshold: float = 0.01  # MSE below this = solved

    def data_sexpr(self) -> str:
        pairs = " ".join(f"(list {x} {y})" for x, y in self.data)
        return f"(list {pairs})"

    def test_sexpr(self) -> str:
        pairs = " ".join(f"(list {x} {y})" for x, y in self.test)
        return f"(list {pairs})"


@dataclass
class ModelTemplate:
    """A candidate model structure with optimizable parameters."""
    name: str
    params: list[str]           # param names
    param_inits: dict[str, float]  # initial values
    param_bounds: dict[str, tuple[float | None, float | None]]  # bounds
    body: str                   # SELPH expression for (lambda (x) ...)
    source_stage: int           # which curriculum stage introduced this

    def setup(self, env: Env):
        """Define params and model function in the SELPH env."""
        for p in self.params:
            init = self.param_inits.get(p, 0.0)
            lo, hi = self.param_bounds.get(p, (None, None))
            # Use Python→SELPH bridge to avoid scientific notation parse issues
            if lo is not None and hi is not None:
                env.define(p, Param(init, lo, hi))
            else:
                env.define(p, Param(init))
        env.eval_expr(f"(define __model__ (lambda (x) {self.body}))")


@dataclass
class FitResult:
    """Result of fitting a template to a task."""
    template: ModelTemplate
    solution: Solution
    train_mse: float
    test_mse: float | None  # None if no test data
    complexity: int          # number of params


def fit_template(template: ModelTemplate, task: Task,
                 steps: int = 500, method: str = "nelder-mead") -> FitResult:
    """Fit a single template to a task, return the result."""
    env = Env()

    # Define the data
    env.eval_expr(f"(define __data__ (dataset {task.data_sexpr()}))")

    # Define the model template and its params
    template.setup(env)

    # Optimize
    sol = minimize(
        env,
        params=template.params,
        loss="(model-loss __model__ __data__ mse)",
        method=method,
        steps=steps,
    )

    # Evaluate on test data if available
    test_mse = None
    if task.test:
        env.eval_expr(f"(define __test__ (dataset {task.test_sexpr()}))")
        test_mse = env.eval_expr("(model-loss __model__ __test__ mse)")

    return FitResult(
        template=template,
        solution=sol,
        train_mse=sol.loss,
        test_mse=test_mse,
        complexity=len(template.params),
    )


def select_best(results: list[FitResult], threshold: float) -> FitResult | None:
    """Pick the best model: lowest test_mse (or train_mse), with complexity tiebreak."""
    passing = [r for r in results if r.train_mse <= threshold]
    if not passing:
        # None pass threshold — return the best anyway
        passing = results
    # Sort by (test_mse or train_mse, complexity)
    passing.sort(key=lambda r: (r.test_mse if r.test_mse is not None else r.train_mse, r.complexity))
    return passing[0] if passing else None


# ── Built-in model templates (Stages 1-4) ────────────────────────────────────

STAGE_1_TEMPLATES = [
    ModelTemplate(
        name="constant",
        params=["_c"],
        param_inits={"_c": 0.0},
        param_bounds={},
        body="_c",
        source_stage=1,
    ),
]

STAGE_2_TEMPLATES = [
    ModelTemplate(
        name="linear",
        params=["_w", "_b"],
        param_inits={"_w": 1.0, "_b": 0.0},
        param_bounds={},
        body="(add (multiply _w x) _b)",
        source_stage=2,
    ),
]

STAGE_3_TEMPLATES = [
    ModelTemplate(
        name="quadratic",
        params=["_a", "_bq", "_cq"],
        param_inits={"_a": 0.0, "_bq": 1.0, "_cq": 0.0},
        param_bounds={},
        body="(add (multiply _a (multiply x x)) (add (multiply _bq x) _cq))",
        source_stage=3,
    ),
]

STAGE_4_TEMPLATES = [
    ModelTemplate(
        name="piecewise_linear",
        params=["_t", "_w1", "_b1", "_w2", "_b2"],
        param_inits={"_t": 0.0, "_w1": 1.0, "_b1": 0.0, "_w2": 1.0, "_b2": 0.0},
        param_bounds={},
        body="(if (> x _t) (add (multiply _w1 x) _b1) (add (multiply _w2 x) _b2))",
        source_stage=4,
    ),
]

ALL_TEMPLATES = STAGE_1_TEMPLATES + STAGE_2_TEMPLATES + STAGE_3_TEMPLATES + STAGE_4_TEMPLATES


# ── Curriculum tasks ─────────────────────────────────────────────────────────

CURRICULUM = [
    # Stage 1: Constants
    Task("const_5", [(1, 5), (2, 5), (3, 5), (4, 5)],
         test=[(10, 5), (-3, 5)]),
    Task("const_neg", [(0, -2), (1, -2), (5, -2)],
         test=[(100, -2)]),

    # Stage 2: Linear
    Task("identity", [(1, 1), (2, 2), (3, 3), (4, 4)],
         test=[(5, 5), (10, 10)]),
    Task("double", [(1, 2), (2, 4), (3, 6), (4, 8)],
         test=[(5, 10), (0, 0)]),
    Task("linear_2x+1", [(1, 3), (2, 5), (3, 7), (4, 9)],
         test=[(0, 1), (5, 11)]),
    Task("linear_neg", [(0, 3), (1, 1), (2, -1), (3, -3)],
         test=[(4, -5), (-1, 5)]),

    # Stage 3: Polynomial
    Task("square", [(1, 1), (2, 4), (3, 9), (4, 16)],
         test=[(5, 25), (0, 0)]),
    Task("quadratic_x2-2x+1", [(0, 1), (1, 0), (2, 1), (3, 4), (4, 9)],
         test=[(5, 16), (-1, 4)]),
    Task("cubic_approx", [(0, 0), (1, 1), (2, 8), (3, 27)],
         test=[(4, 64)],
         threshold=0.1),  # quadratic won't be exact — that's the point

    # Stage 4: Piecewise
    Task("abs_value", [(-3, 3), (-2, 2), (-1, 1), (0, 0), (1, 1), (2, 2), (3, 3)],
         test=[(-5, 5), (5, 5)]),
    Task("relu", [(-3, 0), (-1, 0), (0, 0), (1, 1), (2, 2), (3, 3)],
         test=[(-5, 0), (5, 5)]),
    Task("step_at_2", [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)],
         test=[(-1, 0), (10, 1)],
         threshold=0.05),

    # Stage 5: Model selection — which structure fits?
    Task("mystery_linear", [(1, 4.1), (2, 6.9), (3, 10.2), (4, 13.0)],
         test=[(5, 15.9), (0, 1.1)],
         threshold=0.1),
    Task("mystery_quad", [(0, 2), (1, 1), (2, 2), (3, 5), (4, 10)],
         test=[(5, 17)],
         threshold=0.1),
    Task("mystery_piecewise", [(-2, -4), (-1, -2), (0, 0), (1, 1), (2, 1), (3, 1)],
         test=[(-3, -6), (5, 1)],
         threshold=0.1),
]


# ── Curriculum runner ────────────────────────────────────────────────────────

@dataclass
class CurriculumResult:
    """Result of running the full curriculum."""
    solved: list[tuple[Task, FitResult]]
    failed: list[Task]
    library: list[ModelTemplate]

    def summary(self) -> str:
        lines = [f"Solved: {len(self.solved)}/{len(self.solved) + len(self.failed)}"]
        lines.append(f"Library: {len(self.library)} models")
        lines.append("")
        for task, result in self.solved:
            marker = "✓" if (result.test_mse is None or result.test_mse <= task.threshold) else "~"
            lines.append(
                f"  {marker} {task.name:25s} → {result.template.name:20s} "
                f"train={result.train_mse:.2e} "
                f"{'test=' + f'{result.test_mse:.2e}' if result.test_mse is not None else ''} "
                f"({result.solution.steps} steps)"
            )
        for task in self.failed:
            lines.append(f"  ✗ {task.name:25s} — no model below threshold {task.threshold}")
        return "\n".join(lines)


def run_curriculum(
    tasks: list[Task] | None = None,
    templates: list[ModelTemplate] | None = None,
    verbose: bool = True,
) -> CurriculumResult:
    """Run the fitting curriculum.

    For each task, tries all templates (built-in + library), optimizes params,
    picks the simplest model that fits. Successful models are promoted to the
    library for future tasks.
    """
    if tasks is None:
        tasks = CURRICULUM
    if templates is None:
        templates = list(ALL_TEMPLATES)

    library: list[ModelTemplate] = []
    solved: list[tuple[Task, FitResult]] = []
    failed: list[Task] = []

    for task in tasks:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task.name} ({len(task.data)} points, threshold={task.threshold})")

        # Try all templates + library models
        candidates = templates + library
        results: list[FitResult] = []

        for tmpl in candidates:
            try:
                result = fit_template(tmpl, task)
                results.append(result)
                if verbose:
                    test_str = f"test={result.test_mse:.2e}" if result.test_mse is not None else ""
                    print(f"  {tmpl.name:20s}: train={result.train_mse:.2e} {test_str} ({result.solution.steps} steps)")
            except Exception as e:
                if verbose:
                    print(f"  {tmpl.name:20s}: ERROR — {e}")

        # Select best
        best = select_best(results, task.threshold)
        if best and best.train_mse <= task.threshold:
            solved.append((task, best))
            if verbose:
                print(f"  → SOLVED with {best.template.name} (MSE={best.train_mse:.2e})")

            # Promote to library: create a specialized template with the fitted params
            # as initial values (warm start for similar future problems)
            lib_model = ModelTemplate(
                name=f"{task.name}_{best.template.name}",
                params=best.template.params,
                param_inits=dict(zip(best.template.params, best.solution.params.values())),
                param_bounds=best.template.param_bounds,
                body=best.template.body,
                source_stage=best.template.source_stage,
            )
            library.append(lib_model)
        else:
            failed.append(task)
            if verbose:
                best_loss = min(r.train_mse for r in results) if results else float("inf")
                print(f"  → FAILED (best MSE={best_loss:.2e}, threshold={task.threshold})")

    result = CurriculumResult(solved=solved, failed=failed, library=library)
    if verbose:
        print(f"\n{'='*60}")
        print(result.summary())
    return result
