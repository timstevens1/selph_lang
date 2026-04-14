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

**Parallel mode (fork-based, for depth 3+):**
```bash
./run_probe.sh /tmp/arc_agi1_train.selph --budget 200000 --depth 3 --parallel 8
```

**Checkpoint re-run (skip previously solved tasks):**
```bash
# First run creates <file>.checkpoint automatically
./run_probe.sh /tmp/arc_agi1_train.selph --budget 200000 --depth 2
# Second run skips solved tasks, only probes unsolved
./run_probe.sh /tmp/arc_agi1_train.selph --budget 200000 --depth 2
# Disable checkpoint: --no-checkpoint
# Custom checkpoint path: --checkpoint /path/to/file
```

Parallel mode uses wavefront rounds: run all unsolved tasks in parallel, merge new solutions into the library, repeat until no new solutions appear. Process-level parallelism (fork) — no Rc→Arc migration needed.

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

## Benchmark Results (2026-04-14)

| Set | Solved | Near-miss (>=90%) | Near-miss (>=95%) | Time |
|---|---|---|---|---|
| ARC-AGI-1 train | 36/400 (9.0%) | 80 | 20 | ~960s |
| ARC-AGI-1 eval | 3/400 (0.75%) | 80 | 27 | ~40s |

- Flat-depth-1 enumeration exhausts at ~1,312 candidates per task
- 34/36 Phase 1 tasks use the M-chain; 2 use Flat strategy
- `--depth` now controls strategy/decomposer chaining depth (default 2), `--flat-depth` controls flat enumeration (default 1)
- Phase 4 template transfer recovers 0 additional (2 now solved directly in Phase 1)
- Bottleneck is form coverage, not search budget or composition depth

### §9.60 spatial/object expansion (+6 tasks)

8 new Rust builtins and 6 new form detectors across three categories:

| Category | Builtins | Forms | Tasks solved |
|---|---|---|---|
| Diagonal lines | `grid-draw-line`, `grid-extend-lines`, `grid-connect-same-color`, `grid-rays` | Forms 12-14 in m8g_line_draw | +3 (1f876c06, 22168020, 22eb0ac0) |
| Proximity recolor | `grid-color-voronoi`, `grid-recolor-by-proximity` | Subform 5 in m8g_recolor | +1 (2204b7a8) |
| Noise removal | `grid-remove-small-objects`, `grid-keep-color` | Forms 10-11 in m8g_constant_grid | +1 (5582e5ca) |
| Held-out validation | — | m_chain test-pair validation | +1 (7ddcd7ec) |

Also added held-out validation in m_chain: detected forms are now checked against test pairs before acceptance, preventing overfitting from constant-grid detectors that memorize training outputs.

## AST Tools (structural editing for LLMs)

CLI tools for manipulating SELPH code by node index instead of editing raw s-expressions. Avoids paren-matching issues when an LLM generates or modifies SELPH code.

### Workflow

1. **Query** the file to see definitions and their structure:
```bash
selph ast-query file.selph --list-defs
# 0  define  square  [7]
# 1  define  double  [15]

selph ast-query file.selph --tree square
# [7] App
#   [0] define
#   [1] square
#   [6] Lambda (x)
#     [5] App
#       [2] multiply
#       [3] x
#       [4] x
```

2. **Edit** by node index (output to stdout; add `--in-place` to write back):
```bash
# Replace a node with a new expression
selph ast-edit file.selph --replace 5 '(power x 2)'

# Wrap a node in a let binding
selph ast-edit file.selph --wrap-let 5 result

# Add a new definition
selph ast-edit file.selph --insert-def triple --params 'x' --body '(multiply x 3)'

# Delete a definition
selph ast-edit file.selph --delete-def double
```

3. **Format** with consistent indentation:
```bash
selph fmt file.selph [--in-place]
```

### Key properties
- The LLM only writes small, shallow replacement expressions — never full deeply-nested files
- Node indices come from `--tree` output and map directly to the parser's arena
- Every edit round-trips through the parser, so output is always syntactically valid
- Comments are not preserved (parser discards them)

## Key Architecture

- `selph_fast/src/main.rs` — CLI dispatcher (`arc`, `grow-v2`, `ast-query`, `ast-edit`, `fmt`, etc.)
- `selph_fast/src/ast_tools.rs` — AST query/edit/format tools (structural editing by node index)
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
