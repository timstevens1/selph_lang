"""Run the full SELPH curriculum experiment."""

import sys
sys.path.insert(0, ".")

from selph.curriculum import run_curriculum, print_curriculum_summary, CurriculumConfig


def main():
    config = CurriculumConfig(
        max_stages=6,
        max_depth_per_stage=[2, 2, 2, 2, 2, 2],
        max_candidates=50000,
        tasks_per_stage=[12, 16, 12, 8, 9, 5],
        enable_if_from_stage=4,
        transition_solve_rate=0.4,
        seed=42,
        verbose=True,
    )

    results = run_curriculum(config)
    print_curriculum_summary(results)


if __name__ == "__main__":
    main()
