"""Automatic curriculum runner for SELPH.

Runs a multi-stage curriculum with automatic stage transitions:
  1. Generate tasks for current stage
  2. Solve them with the synthesizer
  3. Check transition criteria (solve rate, library convergence)
  4. Promote solved programs as primitives
  5. Advance to next stage

Transition criteria (§11.4):
  - Solve rate > threshold (default 90%)
  - Library stable (no new abstractions extracted in last round)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable
from .ast import Spec, Symbol, String, Number
from .eval import standard_env, Env
from .synthesize import synthesize, SynthesisResult, Component
from .library import promote_solved, register_abstractions, extract_library, Abstraction, prune_library, track_usage
from .induce import induce_from_failure
from .divide import synthesize_divide_and_conquer
from .taskgen import (GeneratedTask, generate_stage0_tasks, generate_stage1_tasks,
                       generate_stage2_tasks, generate_stage3_tasks,
                       generate_stage4_tasks, generate_stage5_tasks)
from .types import TStr, TNum


# ── Stage result ─────────────────────────────────────────────────────

@dataclass
class StageResult:
    """Results from running one stage of the curriculum."""
    stage: int
    tasks_total: int = 0
    tasks_solved: int = 0
    total_candidates: int = 0
    total_pruned: int = 0
    solve_time: float = 0.0
    solved_programs: list[tuple[str, SynthesisResult]] = field(default_factory=list)
    promoted: list[Abstraction] = field(default_factory=list)
    extracted: list[Abstraction] = field(default_factory=list)

    @property
    def solve_rate(self) -> float:
        return self.tasks_solved / self.tasks_total if self.tasks_total > 0 else 0.0

    @property
    def avg_candidates(self) -> float:
        return self.total_candidates / self.tasks_total if self.tasks_total > 0 else 0.0


# ── Curriculum runner ────────────────────────────────────────────────

@dataclass
class CurriculumConfig:
    """Configuration for a curriculum run."""
    max_stages: int = 6
    max_depth_per_stage: list[int] = field(default_factory=lambda: [2, 2, 2, 2, 2, 2])
    max_candidates: int = 50000
    tasks_per_stage: list[int] = field(default_factory=lambda: [15, 20, 15, 10, 12, 10])
    enable_if_from_stage: int = 4  # enable if-expressions from this stage onward
    transition_solve_rate: float = 0.7  # advance when solve rate > this
    seed: int = 42
    verbose: bool = True


def run_curriculum(config: CurriculumConfig | None = None) -> list[StageResult]:
    """Run a full multi-stage curriculum.

    Returns a list of StageResult, one per stage attempted.
    """
    if config is None:
        config = CurriculumConfig()

    task_generators = [
        generate_stage0_tasks,
        generate_stage1_tasks,
        generate_stage2_tasks,
        generate_stage3_tasks,
        generate_stage4_tasks,
        generate_stage5_tasks,
    ]

    results: list[StageResult] = []
    all_components: list[Component] = []
    env = standard_env()

    for stage in range(config.max_stages):
        if stage >= len(task_generators):
            break

        if config.verbose:
            print(f"\n{'='*60}")
            print(f"STAGE {stage}")
            print(f"{'='*60}")
            print(f"  Library: {len(all_components)} promoted primitives")
            depth = config.max_depth_per_stage[min(stage, len(config.max_depth_per_stage)-1)]
            print(f"  Max depth: {depth}")

        # Generate tasks
        n_tasks = config.tasks_per_stage[min(stage, len(config.tasks_per_stage)-1)]
        tasks = task_generators[stage](n=n_tasks, seed=config.seed + stage)

        if config.verbose:
            print(f"  Tasks generated: {len(tasks)}\n")

        # Solve tasks
        stage_result = _solve_stage(
            stage, tasks, config, all_components, env)
        results.append(stage_result)

        if config.verbose:
            print(f"\n  Solve rate: {stage_result.solve_rate:.0%} "
                  f"({stage_result.tasks_solved}/{stage_result.tasks_total})")
            print(f"  Total candidates: {stage_result.total_candidates:,}")
            print(f"  Avg candidates/task: {stage_result.avg_candidates:,.0f}")
            print(f"  Time: {stage_result.solve_time:.1f}s")

        # Check transition criteria
        if stage_result.solve_rate < config.transition_solve_rate:
            if config.verbose:
                print(f"\n  STOPPING: solve rate {stage_result.solve_rate:.0%} "
                      f"< threshold {config.transition_solve_rate:.0%}")
            break

        # Promote solved programs
        promoted = _promote_stage(stage_result, tasks)
        stage_result.promoted = promoted

        if config.verbose:
            print(f"\n  Promoted {len(promoted)} primitives:")
            for a in promoted[:10]:
                print(f"    {a.name:30s}  {a.param_types} -> {a.return_type!r}")
            if len(promoted) > 10:
                print(f"    ... and {len(promoted) - 10} more")

        # Register in eval environment and add to component library
        register_abstractions(promoted, env)
        new_components = [a.to_component() for a in promoted]
        all_components.extend(new_components)

        # Also try sub-expression extraction from solved programs
        bodies = [r.program.elements[2] for _, r in stage_result.solved_programs
                  if r.program is not None]
        extracted = extract_library(bodies, min_frequency=2, min_compression=0.0)
        stage_result.extracted = extracted
        if extracted:
            register_abstractions(extracted, env)
            all_components.extend([a.to_component() for a in extracted])
            if config.verbose:
                print(f"  Extracted {len(extracted)} sub-expression abstractions")

        # Prune the full library: remove builtin-equivalent and
        # observationally redundant primitives
        all_abstractions = promoted + extracted
        solutions = [r.program for _, r in stage_result.solved_programs
                     if r.program is not None]
        usage = track_usage(solutions, all_abstractions)

        pruned, prune_stats = prune_library(
            all_abstractions, env=env, usage_counts=usage, min_uses=0)

        if config.verbose:
            removed = prune_stats['input_count'] - prune_stats['output_count']
            if removed > 0:
                print(f"  Pruned {removed} redundant primitives "
                      f"(builtin_equiv={prune_stats['builtin_equiv_removed']}, "
                      f"obs_equiv={prune_stats['obs_equiv_removed']})")
                print(f"  Library: {prune_stats['input_count']} -> {prune_stats['output_count']}")

        if config.verbose:
            print(f"\n  Stage {stage} complete. Advancing to Stage {stage + 1}...")

    return results


def _solve_stage(stage: int, tasks: list[GeneratedTask],
                 config: CurriculumConfig,
                 components: list[Component],
                 env: Env) -> StageResult:
    """Solve all tasks in a stage."""
    depth = config.max_depth_per_stage[min(stage, len(config.max_depth_per_stage)-1)]
    use_if = stage >= config.enable_if_from_stage
    result = StageResult(stage=stage, tasks_total=len(tasks))
    t0 = time.perf_counter()

    for task in tasks:
        tr = synthesize(
            task.spec,
            max_depth=depth,
            max_candidates=config.max_candidates,
            extra_components=components if components else None,
            env=env,
            enable_if=use_if,
        )

        result.total_candidates += tr.candidates_explored
        result.total_pruned += tr.candidates_type_pruned

        if tr.found:
            result.tasks_solved += 1
            result.solved_programs.append((task.name, tr))

            # Immediately promote as a library primitive
            _promote_immediately(task, tr, stage, components, env)

            if config.verbose:
                print(f"  OK  {task.name:35s}  cand={tr.candidates_explored:6d}  "
                      f"prune={tr.candidates_type_pruned:6d}  {tr.source}")
        else:
            # Try failure-driven induction
            ir = induce_from_failure(
                task.spec,
                max_depth=depth,
                max_candidates=config.max_candidates // 2,
                extra_components=components if components else None,
                env=env,
            )
            result.total_candidates += ir.candidates_explored

            if ir.success:
                # Package as a SynthesisResult so promotion works
                fake_sr = SynthesisResult(
                    program=ir.program,
                    source=ir.source,
                    found=True,
                    candidates_explored=ir.candidates_explored,
                )
                result.tasks_solved += 1
                result.solved_programs.append((task.name, fake_sr))

                # Register any new primitives immediately
                if ir.new_primitives:
                    register_abstractions(ir.new_primitives, env)
                    for np in ir.new_primitives:
                        components.append(np.to_component())

                if config.verbose:
                    print(f"  IN  {task.name:35s}  cand={ir.candidates_explored:6d}  "
                          f"(induced)  {ir.source}")
                    if ir.decomposition:
                        print(f"      {ir.decomposition}")
            else:
                # Try divide-and-conquer
                dc = synthesize_divide_and_conquer(
                    task.spec,
                    max_depth=depth,
                    max_candidates=config.max_candidates // 2,
                    extra_components=components if components else None,
                    env=env,
                )
                result.total_candidates += dc.candidates_explored

                if dc.found:
                    result.tasks_solved += 1
                    result.solved_programs.append((task.name, dc))
                    _promote_immediately(task, dc, stage, components, env)

                    if config.verbose:
                        print(f"  DC  {task.name:35s}  cand={dc.candidates_explored:6d}  "
                              f"(divide&conquer)  {dc.source}")
                else:
                    if config.verbose:
                        print(f"  --  {task.name:35s}  cand={tr.candidates_explored:6d}  "
                              f"prune={tr.candidates_type_pruned:6d}")

    result.solve_time = time.perf_counter() - t0
    return result


def _promote_immediately(task: GeneratedTask, sr: SynthesisResult,
                         stage: int, components: list[Component],
                         env: Env):
    """Promote a single solved program immediately into the library.

    This makes it available for the very next task in the same stage.
    """
    if sr.program is None:
        return

    body = sr.program.elements[2]
    spec = task.spec
    if not (spec.goal and hasattr(spec.goal, 'pairs') and spec.goal.pairs):
        return

    first_in = spec.goal.pairs[0][0]
    first_out = spec.goal.pairs[0][1]
    in_type = TStr() if isinstance(first_in, String) else TNum()
    out_type = TStr() if isinstance(first_out, String) else TNum()

    promoted = promote_solved(
        [(task.name, body, in_type, out_type)],
        name_prefix=f"s{stage}",
    )

    register_abstractions(promoted, env)
    for p in promoted:
        components.append(p.to_component())


def _promote_stage(stage_result: StageResult,
                   tasks: list[GeneratedTask]) -> list[Abstraction]:
    """Promote solved programs from a stage as library primitives."""
    task_map = {t.name: t for t in tasks}
    items = []

    for name, result in stage_result.solved_programs:
        if result.program is None:
            continue
        body = result.program.elements[2]
        task = task_map.get(name)
        if task is None:
            continue

        # Infer types from the spec
        spec = task.spec
        if spec.goal and hasattr(spec.goal, 'pairs') and spec.goal.pairs:
            first_in = spec.goal.pairs[0][0]
            first_out = spec.goal.pairs[0][1]
            in_type = TStr() if isinstance(first_in, String) else TNum()
            out_type = TStr() if isinstance(first_out, String) else TNum()
        else:
            continue

        items.append((f"s{stage_result.stage}_{name}", body, in_type, out_type))

    return promote_solved(items, name_prefix=f"s{stage_result.stage}")


# ── Summary reporting ────────────────────────────────────────────────

def print_curriculum_summary(results: list[StageResult]):
    """Print a summary of the full curriculum run."""
    print(f"\n{'='*60}")
    print("CURRICULUM SUMMARY")
    print(f"{'='*60}\n")

    total_solved = 0
    total_tasks = 0
    total_candidates = 0
    total_promoted = 0

    for sr in results:
        total_solved += sr.tasks_solved
        total_tasks += sr.tasks_total
        total_candidates += sr.total_candidates
        total_promoted += len(sr.promoted)

        print(f"  Stage {sr.stage}: {sr.tasks_solved:3d}/{sr.tasks_total:3d} solved "
              f"({sr.solve_rate:5.0%})  "
              f"candidates={sr.total_candidates:8,}  "
              f"promoted={len(sr.promoted):3d}  "
              f"extracted={len(sr.extracted):3d}  "
              f"time={sr.solve_time:5.1f}s")

    print(f"\n  Total: {total_solved}/{total_tasks} solved "
          f"({total_solved/total_tasks:.0%})  "
          f"candidates={total_candidates:,}  "
          f"promoted={total_promoted}")

    # Show if later stages benefit from earlier ones
    if len(results) >= 2:
        print(f"\n  Stage progression:")
        for i, sr in enumerate(results):
            bar = '#' * int(sr.solve_rate * 40)
            print(f"    Stage {sr.stage}: [{bar:40s}] {sr.solve_rate:.0%}  "
                  f"avg {sr.avg_candidates:,.0f} cand/task")
