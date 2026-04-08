# SELPH → ARC-AGI: Competition Plan

## Pure DSL Search for Grid Program Synthesis

**Version 0.2 — April 8, 2026**

Target: ARC Prize 2026 — ARC-AGI-2 track (static grid puzzles)
Milestone 1 deadline: June 30, 2026 (~12 weeks)
Milestone 2 deadline: September 30, 2026

---

## 1. Strategic Thesis

ARC-AGI tasks are program synthesis problems: given 2-3 input→output grid pairs, find the transformation program. This is exactly what Selph does — but in a different domain.

The hypothesis: **Selph's library accumulation is the key differentiator.** If solving 50 ARC tasks produces macros that make 50 more solvable at depth 1, we get compounding returns that brute-force approaches don't. The curriculum-driven growth that works on strings/lists/numbers should transfer to grids.

We stay pure DSL search. No neural components. The limit of enumerative search + library learning + meta-heuristics has not been found yet. If we hit a wall, we'll know exactly where it is and can add neural guidance surgically.

---

## 2. What Exists (April 8, 2026)

### 2.1 Grid Type System
- `Value::Grid(Vec<Vec<i8>>)` — row-major, values 0-9
- `TYPE_GRID = 4` in synth.rs, `Type::TGrid` in HM type system
- Full integration: `val_hash`, `vals_equal`, `value_type_tag`, unification
- `#grid` literal syntax in curriculum files: `(#grid ((0 1) (1 0)))`

### 2.2 Grid Builtins (62 total)

**Low-level (Phase 1 — 43 ops):**
- Construction: `grid-make`, `grid-from-list`, `grid-to-list`
- Access: `grid-get`, `grid-set`, `grid-row`, `grid-col`, `grid-width`, `grid-height`, `grid-size`
- Transforms: `grid-rotate-cw/ccw/180`, `grid-flip-h/v`, `grid-transpose`, `grid-crop`, `grid-overlay`, `grid-tile`, `grid-scale`, `grid-replace-color`, `grid-mask`, `grid-pad`
- Analysis: `grid-colors`, `grid-count-color`, `grid-most-common`, `grid-background`, `grid-equal`, `grid-find-color`, `grid-symmetric-h/v`, `grid-dimensions-equal`, `grid-bounding-box`, `grid-trim`
- Objects: `grid-objects` (4-conn), `grid-objects-8` (8-conn), `grid-object-count`, `grid-object-colors`
- Composition: `grid-hconcat/vconcat`, `grid-hsplit/vsplit`, `grid-quarter`

**Perceptual / mid-level (Phase 1b — 19 ops):**
- *Tier 1 — Spatial:* `grid-flood-fill`, `grid-fill-enclosed`, `grid-draw-line-h`, `grid-draw-line-v`, `grid-ray`, `grid-gravity`
- *Tier 1 — Logical:* `grid-xor`, `grid-and`, `grid-or`
- *Tier 2 — Shape analysis:* `grid-object-area`, `grid-object-center`, `grid-is-rectangle`, `grid-detect-rectangles`, `grid-objects-touching`, `grid-overlay-center`
- *Tier 3 — Advanced:* `grid-find-subgrid`, `grid-neighbor-count`, `grid-border`, `grid-fill-rect`

### 2.3 ARC Infrastructure
- `arc.rs` module: minimal hand-rolled JSON parser (zero dependencies), ARC task loader
- `selph arc` CLI command: load tasks, generate `.selph` curriculum, run synthesis
- `#grid` reader macro for curriculum spec files
- ARC-AGI-1 (400 training + 400 eval) and ARC-AGI-2 (1000 training + 120 eval) data downloaded

### 2.4 Synthesis Integration
- Grid components registered in `synth.rs` `default_synth_components()`
- Input/output type detection for grids in `cmd_synth` and `cmd_curriculum`
- Grid ops classified in `component_kind()` as Transform or Analyze
- High-arity ops (≥5 args: draw-line-h/v, ray, fill-rect) NOT registered as synth components — unreachable at typical search depths, will be composed into macros by curriculum

---

## 3. Key Learnings (April 8 session)

### 3.1 Baseline Results

**Phase 1 primitives only (43 low-level ops), depth 2, budget 5K:**
- 2/63 tasks solved (~3%): `1cf80156` (extract object), `2013d3e2` (extract quadrant)
- Both solutions used structural operations (object extraction, quarter + trim)
- Zero tasks solved via color/transform operations

**After adding 19 perceptual primitives (62 total ops), depth 2, budget 50K:**
- 0/36 tasks solved — **regression**
- Previously-solvable `1cf80156` (81 candidates before) now fails at 50K
- Root cause: combinatorial explosion from ~60+ grid components overwhelms the search budget

### 3.2 The Search Space Problem

This is the same pattern as the NL curriculum domain isolation lesson: **loading all components at once causes cross-pollution.** With 60+ grid components at arity 1-4, the depth-2 candidate space is enormous. Most candidates are nonsensical (e.g., `(grid-gravity (grid-xor x x) 0)`).

The type system helps (TYPE_GRID filtering excludes string/num ops) but is insufficient — all grid ops have compatible types with each other.

### 3.3 Analysis of ARC-AGI-1 Transformation Families (37-task sample)

| Family | % of tasks | Key primitives needed |
|---|---|---|
| Flood fill / region ops | 22% | `grid-flood-fill`, `grid-fill-enclosed` |
| Object manipulation | 25% | `grid-objects`, `grid-trim`, `grid-overlay` |
| Rectangle/shape detection | 16% | `grid-is-rectangle`, `grid-detect-rectangles`, `grid-border` |
| Per-object conditional | 16% | Requires if-expressions + object properties |
| Line drawing | 13% | `grid-draw-line-h/v`, `grid-ray` |
| Gravity / falling | 11% | `grid-gravity` |
| Symmetry | 15% | `grid-symmetric-h/v`, `grid-flip-h/v` |
| Spatial relations | 8% | `grid-objects-touching`, `grid-object-center` |
| Logical composition | 5% | `grid-xor`, `grid-and`, `grid-or` |

Most tasks require **multi-step reasoning with object detection**, not single-primitive transforms.

---

## 4. Revised Strategy: Curriculum-Driven Component Selection

### 4.1 Core Insight

The search space problem requires the same solution that worked for strings/NL: **don't load all primitives at once.** Instead:

1. **Diff-based component pruning** — Before synthesis, analyze input↔output structural differences to exclude irrelevant components:
   - Same dimensions → exclude `grid-crop`, `grid-tile`, `grid-scale`, `grid-pad`, `grid-hconcat/vconcat`
   - Dimensions differ → exclude in-place transforms, prioritize composition ops
   - Color histogram unchanged → exclude `grid-replace-color`, `grid-fill-enclosed`
   - Single object in/out → exclude multi-object ops

2. **Staged curriculum by transformation family** — Solve easy tasks first (simple transforms), promote solutions to library, use library for harder tasks. Domain isolation per stage.

3. **Priority tuning** — High-level ops (`grid-gravity`, `grid-fill-enclosed`, `grid-border`) get higher priority than raw accessors (`grid-get`, `grid-row`). Perceptual primitives should be tried before low-level composition.

### 4.2 Proposed Curriculum Stages

| Stage | Focus | Primitives loaded | Expected solve |
|---|---|---|---|
| 0 | Identity/simple transforms | rotate, flip, transpose, trim | 5-10% |
| 1 | Color operations | replace-color, fill-enclosed | +5-10% |
| 2 | Object extraction | objects, trim, nth, bounding-box | +5-10% |
| 3 | Gravity & spatial | gravity, draw-line, ray | +3-5% |
| 4 | Logical composition | xor, and, or, mask, overlay | +3-5% |
| 5 | Shape analysis + conditionals | is-rectangle, border, if-expressions | +3-5% |

Each stage loads only its target primitives + library from previous stages. Solutions get promoted as macros.

### 4.3 Diff-Based Pruning (Implementation Priority)

Before running synthesis on an ARC task, compute:
```
dims_preserved = (input.h == output.h && input.w == output.w)
dims_ratio = (output.h * output.w) / (input.h * input.w)
color_hist_input = histogram of colors in input
color_hist_output = histogram of colors in output
colors_preserved = (color_hist_input.keys == color_hist_output.keys)
object_count_input = number of connected components
object_count_output = number of connected components
```

Use these features to select a reduced component set per task.

---

## 5. Competition Constraints

| Constraint | Value |
|---|---|
| Hardware | 4× NVIDIA L4 GPUs |
| Wall-clock | 12 hours max |
| Tasks | 240 unseen (120 semi-private + 120 private) |
| Internet | None |
| Closed models | Prohibited (GPT-4o, Claude, Gemini) |
| Open-source | Required (MIT or CC0) |
| Submission | Kaggle notebook |

**Implication:** Selph's zero-dependency Rust binary is ideal. No model weights, no GPU needed for inference. The full 12 hours can go to search.

---

## 6. Next Steps (Priority Order)

1. **Implement diff-based component pruning** in `cmd_arc` — analyze input/output before synthesis, select minimal component set per task
2. **Re-run baseline** with pruning to verify no regression on previously-solvable tasks
3. **Build ARC curriculum** — order tasks by difficulty, group by transformation family, chain with library promotion
4. **Object-centric decomposition** — extend D&C strategy to extract/match/transform individual objects
5. **Meta-optimization** — synthesize task feature → strategy heuristic across ARC training set

---

## 7. Success Metrics (Revised)

| Milestone | Date | Target |
|---|---|---|
| ~~Grid primitives working~~ | ~~April 15~~ | **DONE** — 62 builtins |
| ~~ARC loader working~~ | ~~April 18~~ | **DONE** — JSON parser, CLI |
| ~~Baseline measurement~~ | ~~April 22~~ | **DONE** — 3% with low-level, 0% with all (regression) |
| Diff-based pruning | April 15, 2026 | No regression, faster per-task search |
| Staged curriculum | April 22, 2026 | 10%+ on ARC-AGI-1 training |
| Library growth pipeline | May 6, 2026 | 20%+ on training set with grown library |
| Grid-specific strategies | May 27, 2026 | 30%+ on training set |
| Competition submission | June 25, 2026 | Kaggle notebook passing |
| **Milestone 1** | **June 30, 2026** | **Competitive score on ARC-AGI-2** |

---

## 8. Risk Assessment (Updated)

| Risk | Status | Mitigation |
|---|---|---|
| Grid primitive set insufficient | **Addressed** — 62 ops covering all major families | Continue adding targeted primitives as needed |
| Search space explosion | **Active problem** | Diff-based pruning + staged curriculum + priority tuning |
| Library doesn't compound for grids | Unknown | Study which ARC tasks share sub-transformations |
| 3 min/task too slow for deep search | High likelihood | Aggressive pruning, fast path first |
| ARC-AGI-2 much harder than ARC-AGI-1 | Known | Start with ARC-AGI-1, benchmark on both |

---

## 9. What We're NOT Doing (Yet)

- **Neural guidance** — No LLMs, no learned proposers. Pure search + library + heuristics.
- **ARC-AGI-3** — Interactive environments are a fundamentally different challenge. Skip.
- **GPU compute** — Selph is CPU-only. GPUs sit idle (or run parallel search workers).
- **External data** — Only ARC training tasks. No synthetic data generation (yet).

If pure DSL search hits a clear ceiling, we'll know exactly where and why, and can add neural components surgically at the bottleneck.
