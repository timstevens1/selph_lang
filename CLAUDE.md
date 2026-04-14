# SELPH Lang

A symbolic program synthesis system targeting ARC-AGI grid puzzles. Pure DSL search — no neural components.

## Running ARC Benchmarks

### 1. Build the binary

```bash
cd selph_fast && cargo build --release && cd ..
```

### 2. Generate curriculum from ARC JSON

```bash
# ARC-AGI-1
./selph_fast/target/release/selph arc arc_data/data/training/ -o /tmp/arc_agi1_train.selph
./selph_fast/target/release/selph arc arc_data/data/evaluation/ -o /tmp/arc_agi1_eval.selph

# ARC-AGI-2
./selph_fast/target/release/selph arc arc_data_2/data/training/ -o /tmp/arc_agi2_train.selph
./selph_fast/target/release/selph arc arc_data_2/data/evaluation/ -o /tmp/arc_agi2_eval.selph
```

### 3. Run synthesis

**Basic probe (M-chain only):**
```bash
./run_probe.sh /tmp/arc_agi1_train.selph --budget 200000 --depth 2
```

**Fitness probe (M-chain + near-miss analysis + boost retry):**
```bash
./run_fitness_probe.sh /tmp/arc_agi1_train.selph --budget 200000 --depth 2
```

Both scripts automatically prepend the M-chain modules (`examples/meta_curriculum/m_pool.selph` through `m_chain.selph`) in dependency order.

### 4. With scaffolding library

The scaffolding curricula (`examples/arc_grid_curriculum.selph`, `arc_object_curriculum.selph`, `arc_scaffolding_curriculum.selph`) teach object-level primitives. To include them, manually build a combined file:

```bash
META=examples/meta_curriculum
cat $META/m_pool.selph $META/m13_data_atoms.selph $META/m7_library_detection.selph \
    $META/m8_constant_fit.selph $META/m9_unary_wrap.selph $META/m10_affine_combination.selph \
    $META/m11_product_fit.selph $META/m12_structural_pair_fit.selph \
    $META/m_pool_string.selph $META/m8s_constant_string.selph \
    $META/m10s_concat_pair.selph $META/m11s_string_repeat.selph \
    $META/m_pool_grid.selph $META/m8g_constant_grid.selph $META/m_chain.selph \
    examples/arc_grid_curriculum.selph examples/arc_object_curriculum.selph \
    examples/arc_scaffolding_curriculum.selph \
    /tmp/arc_agi1_train.selph > /tmp/combined.selph
./selph_fast/target/release/selph grow-v2 /tmp/combined.selph --budget 200000 --depth 2
```

Note: as of 2026-04-13, scaffolds solve 32/32 but provide no lift on ARC tasks (28/400 with or without). The M-chain already covers the same compositions directly.

## Benchmark Results (2026-04-13)

| Set | Solved | Near-miss (>=90%) | Near-miss (>=95%) | Time |
|---|---|---|---|---|
| ARC-AGI-1 train | 29/400 (7.25%) | 83 | 24 | ~31s |
| ARC-AGI-1 eval | 3/400 (0.75%) | 80 | 27 | ~40s |

- Depth-2 flat enumeration exhausts at ~1,174 candidates per task (1,286 with all forms)
- 27/29 solved tasks use the M-chain; only 2 use Flat strategy
- Phase 2 fitness retry recovers 0 additional tasks
- Bottleneck is form coverage, not search budget

### Near-miss analysis (85 tasks, fitness >= 0.90)

| Pattern needed | Count | % |
|---|---|---|
| Spatial reasoning / line-drawing | 32 | 38% |
| Per-object conditional transform | 14 | 16% |
| Pattern completion / symmetry | 11 | 13% |
| Conditional recoloring | 9 | 11% |
| Template stamping at markers | 7 | 8% |
| Flood fill / region ops | 4 | 5% |
| Border/frame operations | 4 | 5% |
| Tiling/repeating | 4 | 5% |

### §9.58 finding: residual synthesis

The m_refine decomposer tries `solution(x) = correction(form(x))` — apply a base transform, then synthesize a correction on the residual. Result: 14 tasks triggered residual synthesis (form scored >= 0.90), but 0 corrections were found. The residual transformations require the same spatial reasoning the system lacks. The decomposition only helps when corrections are shallow (color remap, trim), which existing M-chain forms already catch.

## Key Architecture

- `selph_fast/src/main.rs` — CLI dispatcher (`arc`, `grow-v2`, etc.)
- `selph_fast/src/synth_v2.rs` — synthesis engine (Flat, Memo, RD, BD, HO, D&C strategies)
- `selph_fast/src/eval_v2.rs` — interpreter (types_v2 values)
- `selph_fast/src/arc.rs` — ARC JSON parser and curriculum generator
- `examples/meta_curriculum/` — M-chain modules (pure SELPH decomposers)
- `examples/meta_curriculum/m_chain.selph` — registers decomposers into `__decomposers__`
- `examples/meta_curriculum/m8g_*.selph` — grid form detectors (constant, symmetry, recolor, line-draw, per-object, template-stamp)
- `examples/meta_curriculum/m_refine.selph` — self-refining residual synthesis decomposer

## Data

- `arc_data/data/training/` — 400 ARC-AGI-1 training tasks (JSON)
- `arc_data/data/evaluation/` — 400 ARC-AGI-1 evaluation tasks (JSON)
- `arc_data_2/data/training/` — 1000 ARC-AGI-2 training tasks (JSON)
- `arc_data_2/data/evaluation/` — 120 ARC-AGI-2 evaluation tasks (JSON)
