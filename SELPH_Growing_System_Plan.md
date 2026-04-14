# SELPH: Growing System Plan

## From Enumerative Solver to Self-Building Architecture

**Version 0.15 — April 13, 2026**

Cleaned up from v0.12 to reflect the project's current state. Detailed implementation logs from §9.1–§9.43 have been condensed into summaries; the full history is preserved in git. Sections 3–5 and 7–8 from the original plan have been collapsed — the project vision shifted significantly per §9.38 (symbolic-first, neural-contingent).

---

## 1. Core Thesis

The model architecture is not specified — it is *grown* by the curriculum. Each curriculum stage teaches the system a capability, and the learned capability becomes infrastructure for the next stage. The system bootstraps from a simple enumerative solver into an increasingly autonomous program synthesis engine, where every component — search, evaluation, heuristics, library management, task decomposition — is itself a SELPH program learned through earlier curriculum stages.

The architecture IS the curriculum. Change the curriculum, change the architecture.

---

## 2. Current State (April 12, 2026)

### 2.1 Three-Layer Architecture (per §9.38)

- **Layer 1: Symbolic kernel** (Rust, frozen). `synth_v2.rs`, `eval_v2.rs`, `types_v2.rs`, `meta_v2.rs`. Hot loops, type universe primitives, dispatcher, AST. Feature-complete for curriculum-only work after §9.37.
- **Layer 2: Curriculum substrate** (SELPH, growing). `__types__`, `__decomposers__`, library functions, task definitions, heuristics, meta-curriculum (M0–M13+). All domain extensions live here.
- **Layer 3: External intent oracle** (LLM API call, contingent). Not yet built. Added when a task demands it.

### 2.2 Language and Runtime

- S-expression parser, tree-walking evaluator (eval_v2), ~75 builtins
- Persistent env via `Rc<EnvNode>` — O(1) closure capture, no env rebuilding
- `Value` enum: Int(i64), Num(f64), Str(Rc<str>), Bool, List(Rc<[Value]>), Ns(Rc<NsMap>), Function(Rc<FunctionData>), Builtin(Sym), Nil
- Value::Node for AST homoiconicity (§9.36) — SELPH programs construct, inspect, and evaluate AST trees as first-class data
- Macro system via `defmacro` (desugared to `define` + `lambda`)
- Threading macros `->` (thread-first) and `->>` (thread-last) — desugared at parse time to nested applications, eliminating deep nesting in pipeline-style code
- Quasiquote (`` ` ``), unquote (`,`), and unquote-splicing (`,@`) — desugared to `make-*` and `quote` calls, enabling concise AST template construction for M-chain forms and decomposers
- `let` bindings with letrec semantics (mutual recursion via literal lambdas)
- Tail-call optimization in `apply()` — If/Let/App chains in tail position loop instead of recursing. Enables unbounded SELPH recursion (e.g., list traversal) without stack overflow.
- `member?` builtin — O(1)-per-element iterative membership test, replacing O(n)-recursive SELPH `pm-member?`
- First-class namespaces with `__types__` and `__decomposers__` registries

### 2.3 Synthesis (synth_v2.rs)

- Priority-weighted interleaved bottom-up search with type-directed pruning
- Deferred materialization (lightweight PendingDesc, on-demand node trees)
- Arity 1–3 support with adaptive type-count caps
- Type reachability filtering (40–65% candidate reduction)
- §9.55 fitness tracking: cell-level similarity scores per candidate, best-so-far tracking, fitness in SynthResult
- Early depth extension (targeted composition probes)
- Domain-based component filtering (grid ops excluded for string tasks, etc.)
- Auto-constant extraction, probe-and-filter, observational equivalence dedup
- Epsilon-equivalence for float dedup

**Strategy pipeline:** SELPH decomposers (M-chain + rx-guided forms) → RD → Flat → BD → HO → D&C → Memo

- **Recursive Decomposition (RD):** Top-down prediction — classify outermost function family, invert to derive subspecs, recursively solve. 6 family inversions (arithmetic, string-op, count, compare, HO, if-expr). Macro-aware bridge decomposition.
- **Boolean Decomposition (BD):** Tries `(and P Q)`, `(or P Q)`, `(not P)` compositions of bool-returning macros.
- **Higher-Order (HO):** Template decomposition — list-map, split-map-join, char-map-join, list-filter. Fills function holes via recursive sub-synthesis.
- **Divide & Conquer (D&C):** Partitions examples by output value, synthesizes separator conditions, composes if-expression trees.
- **Memorization (Memo):** Lookup table fallback for arbitrary mappings.

### 2.4 Meta-Curriculum (pure SELPH, `examples/meta_curriculum/`)

The M-chain is a set of recognition stages that run before enumeration. Each stage probes the spec against a shared pool and emits solutions for shapes it recognizes.

- **m_pool.selph:** Shared pool builder with flags (products, unary-l1, unary-l2, libs, constants)
- **M7:** n-ary library detection — probe library functions against task data
- **M8:** Constant/scaled-atom fit (Forms 1, 2)
- **M9:** Unary-wrapped scaling (Form 2b)
- **M10:** Affine combination (Form 3)
- **M11:** Product-shape fitting (Forms 4, 5) — gated on M6 separability
- **M12:** Structural pair-fit (Form S)
- **M13:** Data-atom extraction (deferred)
- **M0–M6:** Separability decomposers for multi-arg functions (§9.40–§9.41)

~752 LOC removed from synth_v2.rs, replaced by ~570 LOC of pure-SELPH curriculum.

### 2.5 Kernel Capabilities for Curriculum Work

- `synthesize` builtin with `:library`, `:priorities`, `:heuristic` fields
- `test-spec` builtin for candidate evaluation
- `memorize` builtin for data-as-namespace
- `eval-source`, `eval-node`, `quote` for AST manipulation
- `env-functions`, `function-arity`, `function-param-types` for library introspection
- `__decomposers__` namespace for env-driven decomposer registry
- `__types__` namespace for type definitions
- Heuristic-in-spec (§9.34) for SELPH-side meta-learning

### 2.6 Validated Curricula

| Domain | Tasks | Solved | Key mechanism |
|--------|-------|--------|---------------|
| Physics (Stages 1–7) | 37 | **37/37** | M-chain + affine fit + separability decomposers |
| Strings / POS | 28 | **28/28** | Type-dependent pools + Forms 1–6 recognizers |
| Grids (ARC scaffolding) | 13 | **13/13** | Forms 1–8, cross-type bridges, object indexing |
| Original 3-domain chain | 55 | **55/55** | Sequence → CF → NL, library cascade |
| ARC-AGI-1 (training) | 400 | **28/400** | Grid Forms 1-9 + grid-untile + object-level primitives + color-map |
| ARC-AGI-1 (evaluation) | 400 | **3/400** | M-chain forms tuned to training distribution; eval set is harder |
| ARC-AGI-1 (eval + post-mortem) | 400 | **~4/400** | Post-mortem scaffold pipeline recovers ~1 additional task |

### 2.7 CLI Commands

```
selph eval       Evaluate SELPH files or expressions
selph parse      Parse and print AST
selph grow       Run curriculum (synth_v2 core)
selph arc        Convert ARC JSON to curriculum format
selph repl       Interactive REPL
selph help       Show usage
```

---

## 3. The Curriculum IS the Architecture

### 3.1 No Fixed Architecture

Traditional ML: choose an architecture (transformer, SSM, etc.), then train it on data.
SELPH: choose a curriculum, then grow the architecture from it.

The "architecture" at any point is the set of learned SELPH programs that constitute the solver. Each curriculum stage adds a new learned component.

### 3.2 Tuning via Curriculum

To make the system better at a specific domain:
- Add domain-specific tasks to the task-level curriculum
- Add domain-specific types in `__types__`
- Add domain-specific decomposers in `__decomposers__`
- (Optionally) add heuristics in spec namespaces

To make the system more general:
- Add diverse tasks spanning many domains
- Let library extraction discover cross-domain abstractions

### 3.3 Efficiency via Curriculum

The curriculum design problem is: **what sequence of tasks, at what difficulty gradient, with what meta-level training, produces a system that reaches a given capability in the fewest total synthesis steps?**

---

## 4. Completed Work Summary

This section condenses the detailed implementation logs from §8 and §9.1–§9.43 of the original plan. The full history is in git.

### 4.1 Foundation Phase (April 6–7, 2026)

- Built the enumerative solver with 55+ builtins, type inference, macro system
- Validated 3-domain chained curriculum: sequence (13) → CF (20) → NL (22) = **55/55**
- Key enablers: arity-3 synthesis, boolean decomposition, type reachability filtering, RL reward propagation, early depth extension, bytecode VM (22.9x speedup)
- Domain isolation via separate `grow` runs prevents cross-domain pollution
- Meta-optimization Stages 1–4: `priority-plus-type-match` heuristic (+5 tasks at budget 5000)

### 4.2 Search Function Curriculum (April 8, 2026)

Five stages implemented as pure SELPH scripts (`examples/stage0_traversal.selph` through `examples/stage4_5_unified_search.selph`):
- Stage 0: Learned subtree selector (8.3x word-domain speedup)
- Stage 1: Smart search (flat → memo fallback, 12/12 vs 5/12 flat-only)
- Stage 2: Boolean decomposition in SELPH (7/7 vs 6/7 flat)
- Stage 3: Divide & conquer in SELPH (4/4 vs 0/4 flat)
- Stage 4–5: Unified search function (16/16, memo-first optimal)

### 4.3 Recursive Decomposition (April 9, 2026)

Synthesis reframed as top-down prediction: all strategies are instances of SELECT outermost function → DERIVE subspecs → RECURSIVELY synthesize. 64x candidate reduction on test tasks. Macro bridge decomposition unlocked `first_half = (string-take x (half_len x))` at 11 candidates.

### 4.4 Interpreter Rebuild (April 10, 2026)

Decision: rebuild `types.rs` + `eval.rs` core rather than patch. Key changes:
- `Rc<str>` / `Rc<[Value]>` — cloning is refcount bump
- Persistent env via `Rc<EnvNode>` — O(1) closure capture
- Special forms as Rust enum (no string dispatch in hot path)
- `defmacro` desugared to `define` + `lambda`
- VM deleted (its speedup was masking env-rebuild waste)

### 4.5 synth_v2 and Strategy Ports (April 10, 2026)

`synth_v2.rs` written from scratch against v2 types. All strategies ported (Flat, RD, BD, HO, D&C, Memo). Full chained curriculum validates 55/55 on new core. ~4,883 lines of legacy strategy modules deleted.

### 4.6 Minimum Viable Kernel (§9.33, April 10, 2026)

Four kernel items to reach curriculum-only work:
1. **Heuristic-in-spec** (§9.34) — `synthesize` reads `:heuristic` from spec namespace. ✓
2. **AST homoiconicity** (§9.36) — `Value::Node` + 25 builtins (make-*, node-*, eval-node, quote). ✓
3. **Env-driven decomposers** (§9.37) — `__decomposers__` namespace registry. ✓
4. **Types from namespace** (§9.37) — `__types__` namespace. ✓

After §9.37 the kernel is feature-complete. All future domain work happens in `examples/*.selph`.

### 4.7 Physics Curriculum (April 10–11, 2026)

7 stages, 37 tasks, 100% solve rate:
- Stages 1–4: Linear, power laws, multiplicative composition, additive+multiplicative mixing (24/24)
- Stage 5/6: Pendulum, SHM, decay, Lorentz family (8/8)
- Stage 7: Oscillator family — library reuse via M7 (5/5, 47x cheaper than re-derivation)

Key innovations:
- Multi-arg synthesis via indexed atoms
- Constant-hole synthesis (focused affine-fit pass)
- Pure-SELPH separability decomposers (M0–M6)
- Meta-curriculum M7–M12 replaced ~752 LOC of kernel recognition code

### 4.8 Key Design Decisions (preserved from work logs)

- **Domain isolation is essential:** Loading helpers from all domains in a single run causes catastrophic search pollution. Chain separate runs per domain.
- **Decomposition is just another candidate:** The distinction between "strategy selection" and "search" is artificial. Template decomposition fills function-typed holes via recursive sub-synthesis.
- **The library IS the dispatch table:** Promoted macros live in the environment as named functions. `dispatch("reverse", arg)` is just env lookup + apply.
- **Use native types, not string encoding:** Sequence inputs should be lists, not strings. The type system then excludes irrelevant operations automatically.
- **Audit before optimizing:** The bytecode VM was a workaround for env-rebuild waste. After fixing the actual hot-path allocations, the tree walker matched VM performance.
- **Recognition logic is easy; the shared substrate it reads from is the hard part:** When migrating kernel passes to M-stages, the pool-isolation gap caused regressions. Future M-stage curricula should plan for shared data structures at design time.
- **Priority scoring must normalize by arity:** Sum-based scoring creates systematic bias toward higher-arity compositions. Average-based scoring normalizes this.
- **Scaffolding must match the composition chain:** Each step in a multi-step composition needs a separate curriculum stage so promoted macros have correct types and high priority.
- **Depth 2 exhausts — search ordering only matters when budget constrains:** At depth 2, flat enumeration produces ~1,174 candidates and explores them all. Priority reordering, fitness-guided boosting, and online learning are no-ops on an exhausted search. These techniques become valuable at depth 3+ where combinatorial explosion exceeds budget.
- **Fitness is diagnostic gold, not a search signal (at depth 2):** Cell-level fitness on failed tasks reveals 80 near-misses (>=90% accuracy) on ARC-AGI-1 evaluation. This directly identifies which M-chain forms to build next. The 28/400 was on training; evaluation is 3/400 — the M-chain forms are distribution-specific.

### 4.9 Fitness-Guided Synthesis (§9.55, April 13, 2026)

Inspired by evolutionary ARC-AGI-2 approaches (Imbue Darwinian Evolver, SOAR). Added:

**Rust kernel:**
- `value_similarity()` for cell-level grid comparison
- `test_candidate` returns fitness (0.0–1.0) alongside outcome
- `synthesize_inner` tracks best-so-far candidate during enumeration
- `SynthResult`/`StrategyResult` carry fitness + best_source for post-mortem
- `bi_synthesize_args` accepts `("heuristic" fn)` for priority boosting
- Progress counter `[N/M]` and fitness on FAIL lines

**SELPH curriculum (3 new M-chain modules):**
- `m_fitness_grid.selph`: grid-cell-accuracy, grid-fitness, fitness-category
- `m_near_miss.selph`: per-example failure analysis, unary correction retry
- `m_boost_heuristic.selph`: extract ops from near-miss source, build priority map, closed retry loop with `run-fitness-post-mortem`

**Key finding:** Boost-based retry recovered 0/136 evaluation tasks because depth 2 exhausts. The bottleneck is M-chain form coverage, not search efficiency. The 80 evaluation near-misses are the actionable target.

---

## 5. Strategic Reframe: Symbolic-First, Neural-Contingent (§9.38)

### 5.1 The Shift

The original Architecture Spec assumed two halves: a symbolic substrate and a neural synthesizer that would eventually subsume it. The work through §9.37 built the symbolic half to a level the spec didn't anticipate. Two findings prompted the reframe:

1. Legacy heuristics no longer beat synth_v2's default order — type-aware enumeration does what the neural model was supposed to do.
2. Per-task heuristics CAN beat default order, but a global "learned policy" doesn't generalize — per-task scaffolding does.

**The SELPH symbolic substrate is the system, not a stepping stone to a neural system.**

### 5.2 What the Symbolic Substrate Genuinely Can't Do

1. **Fuzzy specs without examples.** Layer 3 (LLM oracle) handles this.
2. **Cross-domain transfer not explicitly designed in.** At small scale, explicit is cleaner.
3. **Deep search at scale.** Theoretical wall, not observed. Mitigations: more decomposers, per-task heuristics.
4. **Perceptual / Gestalt reasoning.** The long tail of ARC-AGI's hard puzzles. Concrete tests needed.

Each has a fallback that doesn't require training a custom model — use an external LLM as a specialized oracle for the specific subtask.

### 5.3 What Would Force Re-evaluation

1. A target domain the substrate provably can't reach
2. LLM oracle pattern proves unreliable
3. Per-call cost dominance at production scale
4. The self-hosting endgame becomes the actual goal

### 5.4 Dropped Work

- No "Stage 0 model" (§11.5 / §14 step 5)
- No GRPO / RL pipeline (§14 step 7)
- No constrained decoder for neural generation
- No three-loop training (§6 Loops 1+2)
- §13 self-hosting removed from critical path
- Differentiable evaluator deferred indefinitely
- Shape system deferred indefinitely

---

## 6. Success Criteria (revised April 10, 2026 per §9.38)

The growing system plan succeeds if **the symbolic substrate handles the domains we care about without requiring a custom-trained neural component.**

### Phase 1 — Substrate completeness ✓

The §9.33 minimum viable kernel for curriculum-only work is built:

- **Item 1** — Heuristic-in-spec for `synthesize`. **Done §9.34.**
- **Item 2** — AST homoiconicity (`Value::Node` + 25 builtins). **Done §9.36.**
- **Item 3** — Env-driven decomposer registry via `__decomposers__` and type-keyed via `__types__`. **Done §9.37.**
- **Item 4** — Custom types via `__types__` namespace. **Done §9.37.**

### Phase 2 — Curriculum-driven heuristic improvement ✓ (partial)

A SELPH-defined heuristic outperforms the default ordering on at least one task without requiring a custom-trained component.

**Met (partial):** §9.34.3 found that the `pro-add` heuristic gives a 2.3× speedup on the increment benchmark. The §9.32.4 finding qualifies this: the *generality* of the speedup is uncertain — synth_v2's type pruning already does most of the work.

**Remaining:** Demonstrate at least one heuristic that outperforms default order *across multiple tasks in the same domain*.

### Phase 3 — Curriculum-driven decomposer extension ✓ (substantially)

A SELPH-defined decomposer solves tasks that the hardcoded chain can't.

**Met (substantially):** §9.37 wired up env-driven decomposer dispatch. The M0–M6 separability decomposers and M7–M12 recognition chain are the proof — 37/37 physics curriculum with ~752 LOC of kernel code replaced by pure SELPH.

### Phase 6 — New-domain curriculum extension ✓ (partially)

The system, given a new domain, can be extended via curriculum work alone — no Rust changes, no neural training — to reach competence.

**Operational definition:** ≥80% of held-out tasks in a new domain solved without modifying Rust code.

**Status:**
- Physics: **37/37** (100%) — fully validated
- Strings/POS: **28/28** (100%) — validated via type-dependent pools + M-chain
- Grids: **9/9** (100%) on custom curriculum — ARC-AGI-1 eval at **14/400** (3.5%)

### Phase 7 — External LLM oracle integration (FUTURE)

An external LLM is invoked from SELPH curriculum code as an intent-translation oracle. The LLM **never generates SELPH programs directly.**

**Status: not yet built.** Added when the first task that needs it appears.

### Measurability

Each phase is measurable:
- Phase 1 (substrate complete): test suite + curriculum solve rate
- Phase 2 (heuristic): A/B comparison via `examples/heuristic_meta_loop.selph` pattern
- Phase 3 (decomposer): A/B comparison via M-chain on/off
- Phase 6 (new domain): solve rate on held-out tasks in the new domain
- Phase 7 (LLM oracle): solve rate on tasks with natural-language intent

---

## 7. Active Work

### 9.44 Stage 7 lands — library reuse buried by score ordering (April 11, 2026)

§9.43.8 listed Stage 7 — the "multi-equation curriculum" — as the
next physics direction. The premise from §9.31.5 was that *once a
base equation is in the library, derived equations from the same
physical context should solve in dramatically fewer candidates than
the base*. This is the §9.31.5 / §9.38 thesis in its most testable
form: same physical scaffold → many equations → cheap reuse.

Stage 7 lands with a clean two-part finding: every task solves, but
**library reuse never fires**. The substrate is more capable than
the §9.31.5 plan assumed (the affine-fit pass alone covers the
oscillator family) but the *mechanism* the plan called for —
derived equations cheaply reusing earlier ones — does not surface
under the current synth ordering.

#### 9.44.1 The curriculum

Five tasks, all 2-arg `(m, k)`, all in the undamped mass-spring
oscillator family. The hypothesis: `osc_T` is the seed, the rest
should solve via short library-reuse compositions.

```
osc_T(m,k)        = 2π√(m/k)              base
osc_freq(m,k)     = 1/T                    same-arity reuse
osc_T_sq(m,k)     = T² = 4π²·m/k           reuse via square
osc_omega(m,k)    = √(k/m) = 2π/T          reuse OR direct
osc_omega_sq(m,k) = k/m                    trivial direct
```

#### 9.44.2 Results

All 5 tasks solve at the default budget (200k):

```
Flat  osc_T          201265 cand   0.984s   2π·√(m/k) via affine fit
Flat  osc_freq       201405 cand   1.046s   0.159·√(k/m) via affine fit
Flat  osc_T_sq       200057 cand   1.103s   4π²·m/k via affine fit
Flat  osc_omega       14798 cand   0.286s   √k/√m direct
Flat  osc_omega_sq       89 cand   0.000s   k/m direct
```

**The §9.31.5 thesis is not validated.** Every task lands, but via
re-derivation, not via the library reuse pathway.

#### 9.44.3 Diagnosis

The score formula `comp.priority + avg(arg_priorities)` puts
`(divide 1 (osc_T x))` at score ~65, buried under millions of
higher-scoring candidates. The affine-fit pass then runs after
enumeration exhausts and finds the closed form.

#### 9.44.5 Why this becomes meta-curriculum work

Library reuse is a *recognition* problem. The meta-curriculum
already has the machinery (M5/M6 pattern). An M7 stage teaches
"library detection": probe each library function L on task inputs,
compute residuals, emit candidates directly.

This matches the §9.38 stance: when the substrate doesn't surface
a useful candidate shape, the answer is a curriculum step that
teaches the system *to look for that shape directly*.

---

### 9.45 Migrate kernel fallbacks into the meta-curriculum (April 11, 2026)

Auditing the kernel showed 12 hand-rolled recognition passes accumulated since §9.38. §9.45 migrated 7 of them (Forms 1, 2, 2b, 3, 4, 5, S, L) into pure-SELPH M-stages, validated at 37/37 physics, and deleted ~752 LOC from synth_v2.rs.

#### Key results

- **§9.44 cliffhanger answered:** After M-chain integration, oscillator family solves in 12,928 total candidates (was ~617k). `osc_T`, `osc_freq`, `osc_T_sq` all at 1 candidate via M7 library detection. **47× cheaper.**
- **Shared pool architecture:** `m_pool.selph` with flag-driven extensions (products, unary-l1, unary-l2, libs, constants). All M-stages consume it.
- **Per-stage pool isolation gap:** Initial migration lost 3/8 Stage 6 tasks because M-stages built narrow private pools. Fixed via shared pool refactor — `decay_N` and `lorentz_gamma` recovered.

#### Deferred follow-ups

- RDB deletion (~140 LOC) — needs M7 to fire on single-arg specs
- DLS/RDC/LCI deletion (~220 LOC) — M13 output not yet wired into M-chain
- `physics_lib.selph` extraction — move bundled helpers out of M7 smoke tests

#### Architectural lesson

**Recognition logic is easy; the shared substrate the recognition reads from is the hard part.** Future M-stage curricula should plan for shared data structures at design time, not bolt them on after a regression.

---

### 9.46 Strings probe — the M-chain is numeric-specialist, not domain-general (April 11, 2026)

Tested `make-pool` against string tasks. Finding: the M-chain is numeric-specialist — string curriculum results are byte-identical with and without the chain. The chain's recognition forms (affine fit, product fit, unary wraps) are numeric patterns that don't apply to string operations.

**ARC decision:** This confirmed that extending to grids requires a domain-specific approach (Path E: type-dependent pools), not just turning on the existing M-chain.

---

### 9.47 P1: Type-dependent pools — strings 9/9 via the M-chain (April 11, 2026)

Path E first instance: string family lands. String-specific recognizer forms added to the M-chain, dispatched by output type. Strings 9/9 (was 6/9 without chain), physics 37/37 unchanged, no Rust changes.

#### 9.47.2 String POS curriculum — context-sensitive disambiguation (April 11, 2026)

Form 3 (multi-classify) + Form 4 (pair-equality) added to m8s. 17/17 POS curriculum including "run" ambiguity, three-class POS, subject/object. Chain demonstrates partition-by-feature shapes.

#### 9.47.3 String chain expands (April 11, 2026)

Word-internal features + Form 5 (ordering) + Form 6 (conjunction) + simplest-discriminator preference. 28/28 POS curriculum. Chain has 6 recognizer forms with primitive Occam's-razor scoring.

#### 9.47.4 Held-out validation (April 11, 2026)

Reject memorization, require generalization. Validation ensures synthesized programs work on unseen examples, not just training data.

#### 9.47.5 Library-driven feature accumulation (April 12, 2026)

The meta-learning thesis on NL: solved word-level tasks become features for sentence-level tasks. Library accumulation means the pool grows richer with each curriculum stage, enabling increasingly complex classifications.

#### 9.47.6 Pool pruning for scaling (April 12, 2026)

Short-circuit F6, priority ordering, degenerate filter, filter Bool fix. 28/28 string POS (gerund_context solved), 2.9× faster.

---

### 9.48 P2: Grid domain — ARC Prize target (April 12, 2026)

Grid domain added as the third domain after physics and strings:
- 11 grid builtins (grid-rows, grid-cols, grid-get, grid-set, grid-map, grid-filter-rows, grid-transpose, grid-rotate, grid-flip-h, grid-flip-v, grid-subgrid)
- Pool builder for grid-specific atoms
- Detector Forms 1 (color-count patterns) + 2 (spatial transforms)
- Chain dispatch for grid output type

**9/9 grid curriculum.** ARC-AGI-1 eval: **14/400.**

Next steps:
1. **Form 3 — spatial decomposition.** Detect rows/columns with uniform properties and emit grid-filter/grid-subgrid compositions. Common ARC pattern (~15% of tasks).
2. **Form 4 — overlay/compose.** Detect multi-layer structure and emit grid overlays. Common ARC pattern.
3. **Learned pool pruning** — replace hand-coded priority order with meta-learned pool selector.

---

### 9.50 Post-mortem meta-learning + object-level grid primitives (April 12, 2026)

Two threads: (1) infrastructure for the system to analyze its own failures and guide what to build next, and (2) expanding the grid primitive vocabulary to cover ARC task patterns beyond simple spatial transforms.

#### Kernel changes

- **`StrategyResult` diagnostics:** `output_type` and `m_chain_ran` fields on every synthesis result.
- **`__curriculum_results__` binding:** grow-v2 accumulates per-task result namespaces and binds them into the env after the task loop.
- **`run-post-mortem` hook:** If defined in env, grow-v2 calls it with the results list and prints diagnostics.
- **`synthesize-args` test forwarding fix:** The builtin was not forwarding held-out test pairs, allowing memorization to pass. Fixed.
- **`extra-seeds` field:** Both `synthesize` and `synthesize-args` builtins accept an `"extra-seeds"` list of literal values injected as depth-0 constants. Widens the reachable type set for cross-type compositions.
- **`as_grid` type loosening:** Grid builtins now accept both `Value::Int` and `Value::Num` cells.

#### New grid builtins (11 total)

| Builtin | Signature | Purpose |
|---------|-----------|---------|
| `grid-scale` | Grid × Int → Grid | Scale each cell to N×N block |
| `grid-tile` | Grid × Int [× Int] → Grid | Tile grid N×M times |
| `grid-fill-enclosed` | Grid → Grid | Fill interior holes |
| `grid-compact` | Grid → Grid | Remove all-zero rows/cols |
| `grid-object` | Grid × Int → Grid | Nth object (sorted by size, 0=largest) |
| `grid-object-pos` | Grid × Int → (row, col) | Position of nth object |
| `grid-place` | Grid × Grid × Int × Int → Grid | Overlay object at position |
| `grid-translate` | Grid × Int × Int → Grid | Shift non-bg cells by (dr, dc) |
| `grid-find-color` | Grid × Int → List | Positions of a color |
| `grid-mask` | Grid × Grid × Int → Grid | Apply mask with fill color |
| `grid-blank` | Int × Int [× Int] → Grid | Create uniform grid |

Size guards on `grid-scale` (max 10×) and `grid-tile` (max 100×100) prevent memory explosion during enumeration.

#### Primitive catalog: cross-type bridges

Added to `primitive_components()` for Flat enumeration:
- **Grid → Int:** `grid-height`, `grid-width`, `grid-object-count`
- **(Grid, Int) → Grid:** `grid-scale`, `grid-tile`, `grid-object`
- **(Grid, Int, Int) → Grid:** `grid-translate`

Binary Grid × Grid → Grid ops (`grid-hconcat`, `grid-vconcat`, `grid-xor`) stay out of the primitive catalog to avoid combinatorial explosion. The M-chain handles them via targeted probing.

#### M-chain Forms 6–8 (pure SELPH, `m8g_constant_grid.selph`)

- **Form 6 — Translation.** Probes `grid-translate(x, dr, dc)` for offsets −2..+2. O(25) probes. Catches shift-right, shift-down, diagonal translations.
- **Form 7 — Mirror.** Probes `grid-hconcat(x, grid-flip-h(x))` and `grid-vconcat(x, grid-flip-v(x))` in both orders. 4 probes.
- **Form 8 — Fill enclosed.** Probes `grid-fill-enclosed(x)`.

Refactored m8g dispatcher to use `m8g-try-chain` helper (list of thunks, first match wins). Eliminates deeply nested let/if chains.

#### `post_mortem.selph` — diagnostic spec analysis

Classifies each failed task across 6 dimensions (size, colors, constant-out, dims-consistent, objects, scale). Probes unary grid builtins at depth 1+2. Discovers implicit constants (dimension ratios) and retries with `extra-seeds`.

#### Scaffolding curriculum (`arc_scaffolding_curriculum.selph`)

13 tasks across 5 stages, all with held-out test data:

| Stage | Tasks | Solved | Solution pattern |
|-------|-------|--------|------------------|
| A: Extractors | grid_height, grid_width | 2/2 | `grid-height(x)`, `grid-width(x)` via Flat |
| B: Single-op | extract_object, compact, scale_2x, scale_3x | 4/4 | `grid-trim`, `grid-compact`, `grid-scale(x,2/3)` |
| C: Object access | largest_object, smallest_object | 2/2 | `grid-object(x,0)`, `grid-object(x,1)` via Flat |
| D: Translation | shift_right, shift_down | 2/2 | `grid-translate(x,0,1)`, `grid-translate(x,1,0)` via Form 6 |
| E: Composition | fill_enclosed, mirror_h, mirror_v | 3/3 | Form 8 + Form 7 (hconcat/vconcat + flip) |

**13/13 solved.** All become library functions for M7 to compose against ARC tasks.

#### ARC-AGI-1 results

**23/400** (was 14/400). 9 new tasks solved:

| Pattern | Tasks | Solution |
|---------|-------|----------|
| Object extraction | 2 | `grid-object(x, 0)` |
| Translation | 1 | `grid-translate(x, 1, 0)` |
| Mirror (h+v) | 4 | `grid-hconcat(x, grid-flip-h(x))` etc. |
| Scaling | 2 | `grid-scale(x, 2)`, `grid-scale(x, 3)` |

All via M-chain at 1 candidate each.

#### Bug fixes

- `synthesize-args`: held-out test forwarding (exposed 266 false memorization recoveries)
- `grid-fill-enclosed`: separate per-value flood fills + fill with nearest differing neighbor (handles bg≠0 cases)
- `as_grid`: accepts Num cells alongside Int

#### Key architectural insights

1. **The type system was the bottleneck, not search depth.** Grid-scale needed Int constants alongside Grid values. Adding cross-type bridges to the primitive catalog unlocked scale/tile tasks immediately.
2. **Binary ops in primitives cause combinatorial explosion.** Grid × Grid compositions must stay in the M-chain as targeted probes, not in the enumerator's catalog.
3. **Object indexing works.** `grid-object(x, 0)` for "extract largest" is the right abstraction — sorted access by size makes objects first-class without complex selection logic.
4. **M-chain forms are cheap and high-leverage.** Forms 6–8 added 5 ARC solves at 0 enumeration cost (1 candidate each). Each form is ~30 lines of pure SELPH.

#### Next steps

1. ~~Form 9 — object recomposition.~~ **Done (§9.53).** `grid-recompose` Rust builtin + M-chain Form 9 with two sub-forms: per-object rigid transform (via `grid-probe-recomp`) and object selection (`grid-object`/`grid-trim`). Per-obj-transform found 0 ARC-AGI-1 tasks (rigid per-object transforms are rare); obj-select added 2 tasks. Also fixed Form 4 arity-1 unwrap bug and cycle handling (+2 tasks), and ARC curriculum test-data format (test rows were leaking into training). **27/400 ARC-AGI-1.**
2. **Pattern repetition detection.** Inverse of `grid-tile`: find the minimal repeating unit in a grid. Catches self-tiling tasks like 007bbfb7. **Addressed as §9.52 next step 3** (inverse scaffolds).
3. **Object sort options.** `grid-objects` currently returns objects in scan order, `grid-object` sorts by size. Adding sort-by-position (topmost, leftmost) would handle tasks that reference objects spatially.
4. **Soft reachability.** When the search space exhausts at ~1200 candidates (hard type wall), the post-mortem could retry with a wider component set. Currently all retries find 0 additional tasks — the gap is in operation vocabulary, not type filtering.
5. ~~Close the meta-learning loop.~~ **Done (§9.52).** Post-mortem now generates scaffolding tasks, solves them, binds solutions as library functions, and re-attempts failures. First auto-recovery: `ccd554ac`.

---

### 9.51 Threading macros and let bindings for LLM-friendly code (April 13, 2026)

Deeply nested S-expressions are hard for LLMs to read and generate correctly. Two syntax features address this without changing the evaluator or AST node types.

#### Threading macros (`->`, `->>`)

Desugared in `convert_tree` (same pass as `defmacro`). No new `SpecialForm` or `Node` variants.

```
;; thread-first: value threaded as first arg of each step
(-> (grid-new 3 3 0)
    (grid-fill 1 1 5)
    (grid-set 0 0 2))
;; desugars to: (grid-set (grid-fill (grid-new 3 3 0) 1 1 5) 0 0 2)

;; thread-last: value threaded as last arg of each step
(->> x (f a) (g b c))
;; desugars to: (g b c (f a x))
```

Bare symbols work: `(-> x f g)` → `(g (f x))`. Mixed bare and applied forms work freely.

#### `let` bindings

Already a first-class `Node::Let` with letrec semantics — mutually recursive lambdas in the same binding block see each other. Combined with threading:

```
(let ((base (grid-new 3 3 0)))
  (-> base
      (grid-fill 1 1 5)
      (grid-set 0 0 2)))
```

**Impact:** Pipeline-style SELPH code (M-chain dispatchers, pool builders, post-mortem analysis) can now be written as flat threading pipelines instead of deeply nested calls. This directly improves LLM code generation accuracy for SELPH.

---

### 9.52 Closed meta-learning loop: post-mortem to scaffolding to recovery (April 13, 2026)

The meta-learning loop that was outlined in §9.50 step 5 ("close the meta-learning loop") is now implemented end-to-end. The post-mortem no longer just classifies failures — it acts on them.

#### Architecture

```
curriculum -> synthesis (3/400)
  -> diagnose (Rust: grid-diagnose-spec)
  -> prescribe (SELPH: 11 ranked buckets)
  -> scaffold (SELPH: generate synthetic tasks from prescriptions)
  -> solve scaffolds -> bind as library functions
  -> targeted re-attempt (only buckets matching scaffolded families)
  = 4/400 total
```

#### Rust builtins for fast diagnosis (eval_v2.rs, +729 lines)

| Builtin | Purpose |
|---------|---------|
| `grid-diagnose-spec` | Full spec classification: size, colors, scale, dims, subtype |
| `grid-probe-recomp` | Object count conservation, per-object transforms, single-edit |
| `grid-probe-extract` | Compact, object-trim, color-mask extraction probes |
| `grid-probe-scale` | Scale factor detection against constant list |

Plus internal helpers: `grid_cc_with_pos` (CC with position data), `grids_equal`, `parse_spec_grid_pairs`, raw grid transform functions. These replaced interpreted SELPH probes that took 10+ minutes on 400 tasks — now <3 seconds.

#### Prescription system (post_mortem.selph)

`pm-prescribe` maps diagnosis signatures to 11 fine-grained action buckets:

| Priority | Prescription | Tasks (of 397) |
|----------|-------------|----------------|
| P2 | recomp/single-obj-edit | 14 |
| P3 | recomp/obj-count-change | 97 |
| P3 | extract/complex | 90 |
| P3 | color-filtering | 38 |
| P3 | scaling-or-tiling | 22 |
| P3 | recomp/few-objects | 12 |
| P4 | color-introduction | 87 |
| P4 | recomp/many-objects | 14 |
| P4 | grid-composition | 12 |
| P5 | structural-rearrangement | 2 |
| P6 | unclassified | 9 |

Sub-classification uses `grid-probe-recomp` to distinguish object-count-change, single-object-edit, few-objects, many-objects, and per-object-transform patterns within the recomposition bucket.

#### Scaffolding generator

`pm-generate-scaffolding` creates synthetic training tasks from prescriptions. 7 scaffold families:

1. **Scaling:** grid-tile (2x2, 3x1, 1x3), grid-scale (2x, 3x)
2. **Tile-mirror:** hconcat+flip-h, vconcat+flip-v, checkerboard, scale+flip
3. **Extraction:** grid-trim, grid-compact, grid-object (largest, 2nd)
4. **Object compositions:** object+rotate, object+flip, object+scale
5. **Extract by position:** 2nd-largest object, trim+compact
6. **Color swap:** replace-color, keep-bg-only
7. **Grid composition:** hconcat-self, vconcat-self, hconcat+rotate

Each scaffold uses small synthetic grids (2x2, 3x3) and is designed to be trivially solvable. Solutions become library functions discoverable by M7.

#### First auto-recovery

Task `ccd554ac` solved with `(lambda (x) (grid-tile (nth x 0) (grid-height (nth x 0))))` — tile a grid N times where N is its height. The scaffolding taught `grid-tile` as a library function; synthesis composed it with the built-in `grid-height` to solve a task that the initial synthesis couldn't reach.

#### Performance

- Post-mortem diagnosis: <3s on 400 tasks (was 10+ min before Rust builtins)
- 35 scaffolds generated and solved
- Targeted re-attempt: ~5 min (filters to ~280 tasks matching scaffolded buckets)
- Total wall time: 5:40 for full 400-task eval with closed loop

#### Flags

- `__pm_skip_recovery__` — skip probe-based auto-recovery phase
- `__pm_skip_scaffolding__` — skip scaffolding generation + re-attempt

#### Next steps

1. **Task-data scaffolds.** Use failing tasks' own training inputs as scaffold inputs instead of synthetic grids. The task knows what grid shapes it needs — a scaffold built from the actual data is more likely to produce a library function that M7 can compose for the original task.

2. **Depth-3 composition scaffolds.** Current scaffolds are depth 1-2 (single ops or 2-op combos). Many ARC tasks need 3-step chains (e.g., extract object, transform it, place it back). Generate scaffolds with 3-operation compositions.

3. **Inverse scaffolds.** Many extraction tasks are inversions of grow operations. "Given a tiled grid, extract the tile" is the inverse of `grid-tile`. Generating both forward and inverse scaffolds for each operation doubles coverage.

4. **Iterative scaffold refinement.** After the first scaffolding pass recovers some tasks, the recovered solutions become library functions for a second pass. Multiple scaffold-solve-retry iterations could compound recoveries.

5. **Prescription-guided M-chain form generation.** The prescription system knows "97 tasks need object-count-change handling." Instead of scaffolding, generate a new M-chain form (as a SELPH function) that directly probes for the pattern and registers it via `__decomposers__`. This is deeper than scaffolding — it extends the recognition chain itself.

6. **Unwrap arity-1 inputs once in `detect-constant-grid`.** Currently each form must remember to `(head inp)` to unwrap the arity-1 wrapper `(list grid)` → `grid`. Pool-based forms get this for free from `make-pool-grid`; direct-probe forms do it ad-hoc (Form 4 had a bug here). Unwrap once at the top of `detect-constant-grid` — `(let ((grids (map head inputs)) ...)` — and pass raw grids to all forms. Eliminates a class of silent bugs.

---

### 9.53 Form 9, Form 4 fixes, ARC curriculum format (April 13, 2026)

Three changes, 23→27/400 ARC-AGI-1 with proper held-out test validation.

#### Form 9 — object recomposition (m8g_constant_grid.selph)

New `grid-recompose` Rust builtin: `(grid-recompose grid "rotate-cw")` applies a named rigid transform to each connected-component object in-place. Extracts existing CC detection + transform + place-back logic from `grid-probe-recomp`.

M-chain Form 9 has two sub-forms:
- **Per-object transform:** calls `grid-probe-recomp` on the spec, emits `grid-recompose` if a rigid transform matches all pairs. Found 0 ARC-AGI-1 tasks (rigid per-object transforms are rare in the dataset).
- **Object selection:** if `grid-probe-recomp` detects obj-select (output = nth-largest object or trim), emits `grid-object(x, N)` or `grid-trim(x)`. +2 tasks (`1f85a75f`, `be94b721`).

#### Form 4 — color-map bug fixes

1. **Arity-1 unwrap bug.** `m8g-detect-color-map` used `(head inputs)` which returns the arity-1 wrapper `(list grid)`, not the grid. `(length first-in)` returned 1 instead of the grid height, breaking dimension checks and cell iteration. Fixed to `(head (head inputs))`. Same fix in `m8g-verify-color-map`.

2. **Cycle handling.** Chained `grid-replace-color` calls fail on bidirectional swaps (e.g., 5↔8): the second replacement undoes the first. Fixed with two-pass temp-color approach: pass 1 moves each from-color to temp (from+50), pass 2 moves each temp to final to-color. ARC colors 0-9, temps 50-59.

+2 tasks (`b1948b0a`: 6→2, `c8f0f002`: 7→5). +1 more with cycles (`d511f180`: 5↔8). Task `0d3d703e` (8 cyclic remaps) solves standalone but fails in full 400-task run — likely silent error from an earlier task corrupting M-chain state.

#### ARC curriculum format fix (arc.rs)

`arc_dir_to_curriculum` previously emitted `:test` as a bare separator between training and test rows. The parser silently ignored `:test` (not a recognized symbol in App context) and parsed test rows as additional training pairs. Result: Form 5 (classify-by-scalar) memorized all examples including test data, and `validate_held_out` got empty test lists (returning true unconditionally). This inflated scores to 209/400.

Fixed to emit `(test input output)` — 3-element tuples matching `parse_curriculum_tasks` expectations. Also switched from `#grid` reader macro to plain nested lists for grow-v2 compatibility.

#### Form 8 — fill-enclosed (no bug)

Form 8 correctly unwraps inputs and probes `grid-fill-enclosed`. Confirmed 0 ARC-AGI-1 training tasks where `grid-fill-enclosed(input) == output`. ARC "fill holes" tasks (e.g., `00d62c1b`) introduce new colors (0→4), not adjacent-neighbor fill. Form 8 is correct but targets a pattern absent from this dataset.

---

### 9.54 Scaffold pipeline + prescription-guided form generation (April 13, 2026)

Six improvements to the post-mortem meta-learning loop, targeting higher ARC-AGI-1 coverage through richer scaffolding and dynamic M-chain extension.

#### Items 1-4: Scaffold pipeline improvements (post_mortem.selph)

**Item 1 — Task-data scaffolds.** New `pm-task-data-scaffolds-for-rx`: instead of only using synthetic 3x3/4x4 grids (`pm-sample-grids`), extracts training inputs from failing tasks themselves and applies the same operation families. Scaffolds built from actual task data are more likely to produce library functions that M7 can compose. Size guard filters grids >10x10. Names use `td_<op>_<task>` to avoid collisions. 5 op-family lists cover scaling, extraction, recomp, color-filter, and composition. `pm-generate-scaffolding` now takes `(prescriptions fails)` and appends task-data scaffolds after synthetic + inverse scaffolds.

**Item 2 — Depth-3 composition scaffolds.** Scaffold tuple format changed from `(name spec)` to `(name spec depth)` with backward-compatible default depth 2. New `pm-scaffold-depth3-compositions` generates 7 three-step chain scaffolds (e.g., `grid-trim(grid-rotate-cw(grid-object(x, 0)))`). `pm-solve-scaffolding` reads depth from each tuple, uses budget 100000 for depth>=3 (vs 50000 for depth-2). Wired into `extract/complex` and `recomp/` prescription buckets.

**Item 3 — Inverse scaffolds.** New `pm-scaffold-invertible?` whitelist (tiling, scaling, concat, mirror — NOT extraction or color-swap). `pm-scaffold-inverse-pair` swaps inputs/outputs, bumps depth for compositions. `pm-generate-inverse-scaffolds` filters and maps forward scaffolds to inverses. Appended after forward scaffolds in `pm-generate-scaffolding`.

**Item 4 — Iterative scaffold refinement.** `pm-run-scaffolding-loop` now iterates up to `__pm_max_iterations__` times (default 3). Each iteration: generate scaffolds -> solve -> bind solutions as `pm-lib-<name>` via `eval-source` -> retry failures -> re-diagnose remaining -> re-prescribe -> recurse. New helper `pm-bind-recovered` binds solutions as library functions. `pm-prepare-next-iteration` uses `->` threading macro for clean re-diagnosis pipeline. Early termination on: no scaffolds generated, none solved, none recovered, or max iterations.

#### Item 5: Prescription-guided M-chain form generation (post_mortem.selph)

New capability: the post-mortem generates decomposer functions from failure analysis and registers them dynamically in `__decomposers__`. Runs as "Phase 1.5" between prescription aggregation and scaffolding.

**Color-probe decomposer.** `pm-rx-color-decomposer` probes each task spec for single-color removal patterns: for each non-background color c in training inputs, checks if `grid-replace-color(input, c, background)` matches the output across all pairs (including test validation). Targets `recomp/obj-count-change` (97 tasks), `color-filtering`, and `recomp/single-obj-edit` buckets. Registered via `eval-source` into `__decomposers__` namespace.

Supporting functions: `pm-grid-colors` (extract non-background colors), `pm-try-color-keep` (probe keep/remove per color), `pm-probe-color-form` (full spec probe with test validation), `pm-check-color-all-pairs` (verify across all training pairs), `pm-generate-rx-forms` (orchestrator that inspects prescription buckets and registers decomposers for high-count gaps).

#### Item 6: Arity-1 unwrap cleanup (m8g_constant_grid.selph) — REVERTED

Attempted to unwrap `(list grid) -> grid` once at the top of `detect-constant-grid` and remove ad-hoc `(head inp)` unwraps from Forms 4, 6, 7, 8. Caused 27->24 regression. Investigation needed: likely interaction with `parse_spec_grid_pairs` which has its own unwrap logic that mishandles 1-row grids when receiving already-unwrapped input. Reverted; Forms still use per-form `(head inp)` unwrapping.

#### Infrastructure: grid-untile builtin + 64MB stack (eval_v2.rs, main.rs)

**`grid-untile` Rust builtin.** Given a grid, tries all factor pairs (nr, nc) dividing (height, width). If all sub-grids are identical, returns the smallest tile; otherwise nil. Added to `m_pool_grid.selph` unary ops. Supports inverse scaffolds for tiling tasks.

**64MB default stack.** `main()` now spawns `real_main()` on a thread with 64MB stack (was system default 8MB). The tree-walking evaluator's recursion depth exceeds 8MB when the M-chain + post-mortem functions are loaded together (~2000 define statements in env). Configurable via `RUST_MIN_STACK` env var.

**Deferred post-mortem loading.** New `--post-mortem <file>` flag for `grow-v2`. Loads the post-mortem SELPH file AFTER the curriculum completes, just before calling `run-post-mortem`. Loading ~190 post-mortem defines into the preamble caused OOM: SELPH closures capture the full env chain, so every M-chain call during synthesis bloated by the post-mortem functions. Deferred loading keeps the synthesis env lean (~30s for 400 tasks, same as baseline).

#### Results

ARC-AGI-1: **27->28/400**. The +1 (`7b7f7511`) is from `grid-untile` in the M-chain pool -- directly solves "extract repeating tile" via `(lambda (x) (grid-untile (nth x 0)))`. Post-mortem scaffolding generated 176 scaffold solutions in iteration 1, but 0 ARC tasks recovered (M7 library detection couldn't compose scaffolded functions to match failing tasks). Iterative loop correctly stopped after 0 recoveries.

#### Next steps

1. ~~**Investigate Item 6 regression.**~~ **Done (§9.56).** Root cause: `parse_spec_grid_pairs` heuristic confuses 1-row grids with arity-1 wrappers. Fixed by checking whether `l[0][0]` is a list (grid row) or scalar (1-row grid cell).

2. **Expand rx-guided forms.** Current color-probe covers single-color removal. Add: multi-color removal, color-keep (retain only one color), background swap, object-count-based branching.

3. ~~**Reduce stack usage.**~~ **Done (§9.56).** TCO in `apply()` handles If/Let/App tail positions iteratively. `member?` builtin replaces O(n)-recursive `pm-member?`. Tested at depth 10,000 on 8MB stack.

4. **Profile scaffold iteration.** Measure per-iteration yield to determine if 3 iterations is optimal or if 2 suffices.

---

### 9.55 Quasiquote for AST template construction (April 13, 2026)

Every M-chain form and post-mortem decomposer spent ~30-40% of its code manually constructing AST nodes via `make-app`, `make-symbol`, `make-int`, etc. Added quasiquote (`` ` ``), unquote (`,`), and unquote-splicing (`,@`) as syntactic sugar that desugars entirely into existing constructs.

#### Parser changes (parser.rs)

Three new token kinds (`Backtick`, `Comma`, `CommaAt`) recognized in the tokenizer. The parser wraps them as `App([quasiquote, child])`, `App([unquote, child])`, `App([unquote-splicing, child])` — standard Lisp reader-macro approach. `` ` `` added to `is_delimiter`.

#### Desugaring (eval_v2.rs, fourth pass in `convert_tree`)

Recursive `desugar_qq` function with depth tracking for nested quasiquotes:

| Pattern | Desugars to |
|---|---|
| `` `atom `` | `(quote atom)` |
| `` `,expr `` at depth 1 | `expr` (evaluated at runtime) |
| `` `(f ,x y) `` | `(make-app (quote f) x (quote y))` |
| `` `(f ,@xs y) `` | `(apply make-app (append (append (list (quote f)) xs) (list (quote y))))` |
| `` `(if ,c t e) `` | `(make-if c (quote t) (quote e))` |
| `` `(lambda (x) ,body) `` | `(make-lambda (list "x") body)` |
| `` `(let ((a ,v)) ,b) `` | `(make-let (list (list "a" v)) b)` |

**No new builtins, no new Node variants, no new SpecialForms.** Uses existing `quote`, `make-app`, `make-if`, `make-lambda`, `make-let`, `list`, `append`, `apply`.

#### Practical impact

Before:
```
(make-lambda (list "x")
  (make-app "grid-replace-color"
    (make-app "nth" (make-symbol "x") (make-int 0))
    (make-int c) (make-int bg)))
```

After:
```
`(lambda (x) (grid-replace-color (nth x 0) ,(make-int c) ,(make-int bg)))
```

Directly reduces the cost of writing new M-chain forms and rx-guided decomposers. The analysis logic (the hard, unique part of each form) stays unchanged; the AST emission that follows it becomes near-free.

#### Tests

18 new tests (all passing): atoms, unquote, splicing, mixed, nested QQ, if/lambda/let in QQ, practical grid patterns. 449 existing tests unaffected.

#### V1 engine removal + Python layer deletion

Removed the entire v1 engine and Python layer in the same session:

**Rust v1 modules deleted (13 files, ~16,000 lines):** `eval.rs`, `synth.rs`, `vm.rs`, `hm.rs`, `meta.rs`, `verify.rs`, `stochastic.rs`, `multitree.rs`, `abstraction.rs`, `library.rs`, `namespace.rs`, `taskgen.rs`, `trace.rs`.

**V1 CLI commands removed (8):** `synth`, `grow`/`curriculum` (legacy), `bench`, `generate`, `verify`, `multi-synth`, `meta-opt`, `arc --synth`. The `eval` and `repl` commands were ported to eval_v2; `grow` now aliases `grow-v2`; `arc` retained as a curriculum converter only.

**Python layer deleted:** `lib.rs` (1,872 lines PyO3 bindings), `selph/` package (19 modules), `tests/` (15 test files), `experiments/` (10 scripts), `pyproject.toml`, pyo3 dependency from Cargo.toml.

**Result:** 38,774 → 18,896 lines of Rust (51% reduction). 9 pre-existing multitree test failures eliminated. 251 tests pass, 0 fail. The codebase is now purely: Rust kernel (9 files) + SELPH curriculum scripts.

---

### 9.56 TCO, member? builtin, and arity-1 unwrap fix (April 13, 2026)

Three changes addressing the §9.54 next steps: stack usage reduction, a builtin for the most common recursive pattern, and a correctness fix for the grid unwrap heuristic.

#### Tail-call optimization in apply()

`apply_tco` replaces the single `eval(body)` call with an iterative loop that walks tail positions:

- **If branches:** Evaluates condition, follows the taken branch without recursing.
- **Let bodies:** Evaluates bindings, continues with body in tail position.
- **App in tail position:** Evaluates function + args. If target is a `Function`, rebinds parameters and loops (zero stack growth). If target is a `Builtin`, calls directly.
- **All other nodes:** Falls through to normal `eval()`.

This covers the standard recursive patterns in SELPH (`pm-member?`, `pm-all?`, `pm-any?`, accumulator loops, mutual recursion via let). Tested at depth 10,000 with 8MB stack — previously hit `MAX_EVAL_DEPTH=256` at depth 257.

Non-tail recursion (e.g., fibonacci) is unaffected — it still uses the normal eval path with depth tracking.

#### member? builtin

`(member? needle list)` — iterative Rust membership test with `(needle, list)` arg order matching `pm-member?`. Drop-in replacement: O(n) scan in Rust vs O(n) recursive SELPH calls, each adding a stack frame. Uses existing `values_equal` for comparison.

#### parse_spec_grid_pairs unwrap fix

The arity-1 unwrap heuristic in `parse_spec_grid_pairs` confused 1-row grids with arity-1 wrappers. Both match `l.len() == 1 && l[0] is List`. The fix adds a deeper check: the inner list's first element must itself be a list (a grid row), not a scalar. A 1-row grid like `((1 2 3))` has `l[0][0] = Int`, so it's correctly identified as a grid, not a wrapper.

This was the root cause of the §9.54 Item 6 regression (27→24 when unwrapping was consolidated). The per-form `(head inp)` unwrapping in Forms 4/6/7/8 remains unchanged for now — the fix enables future consolidation but doesn't force it.

#### Results

- 260 tests pass (251 existing + 4 member? + 5 TCO)
- ARC-AGI-1 training: **28/400** in **27.67s** (unchanged)
- TCO performance benefit is in post-mortem / scaffolding paths where `pm-member?` and recursive analysis run over hundreds of failure results

#### Next steps

1. **Replace `pm-member?` with `member?` in post_mortem.selph.** The builtin is registered; the SELPH code still uses the recursive version. Swapping it eliminates the deepest recursion path in post-mortem analysis.
2. **Reduce default stack to 8MB.** With TCO handling tail recursion, the 64MB default may no longer be necessary. Validate on the full scaffolding pipeline before changing.
3. **Consolidate Form 4/6/7/8 unwrapping.** The Rust heuristic fix unblocks the §9.54 Item 6 refactor — unwrap once at the top of `detect-constant-grid` instead of per-form.

---

### 9.57 Composable recursive dispatch (April 13, 2026)

Sub-synthesis in HO/D&C/Induction/RD previously used flat enumeration only. This meant strategies couldn't compose: HO could detect "map F over objects" but couldn't call the M-chain to discover F = `grid-rotate-cw(grid-flip-h(x))`. Each strategy was an island.

#### The change

Added `strategy_depth` parameter to the dispatcher. When `strategy_depth > 0`, sub-problems route through the full strategy chain (M-chain → Flat → RD → BD → HO → D&C → Induction → Memo) with decremented depth. At depth 0, flat only (previous behavior). Default is 1.

**New function:** `sub_synthesize(components, inputs, expected, env, universe, max_depth, max_candidates, strategy_depth)` — replaces direct `synthesize()` calls in all decomposition strategies.

**Strategies wired through recursive dispatch:**
- **HO** (4 templates): per-element sub-synthesis → full chain
- **D&C** (separator + branch): both sub-problems → full chain
- **Induction** (step1 + step2): both halves → full chain
- **RD**: flat fallback → full chain (self-recursion via `rd_depth` unchanged)
- **BD**: no change (probes predicates directly, no sub-synthesis)

**No infinite recursion risk:** `strategy_depth` strictly decrements. RD has its own `rd_depth` cap. D&C recurses structurally (fewer groups each level). All terminate.

#### Compositions now possible

```
HO("map F over objects") → sub_synthesize(F) → M-chain: F = rotate(flip(x))
D&C("classify by cond") → sub_synthesize(branch) → HO: map transform
Induction("x → mid → y") → sub_synthesize(mid→y) → M-chain: affine fit
RD("f(g(x))") → sub_synthesize(g) → D&C: conditional inner function
```

#### Results

- 260/260 tests pass
- 5 tests updated: tasks previously solvable only by Memo or D&C now solved earlier by RD or D&C with composed sub-strategies (expected — the whole point)
- ARC-AGI results pending

#### Next steps

1. **Benchmark ARC-AGI-1 with recursive dispatch.** Compare 28/400 baseline (flat sub-synthesis) vs new score. If tasks move, identify which compositions fired.
2. **Expose `strategy_depth` as CLI flag.** Trivial change — pass through `cmd_grow_v2` instead of hardcoding 1. Enables `--strategy-depth 2` for deeper composition at the cost of search budget.
3. **Write new grid decomposition forms in SELPH.** The recursive dispatch unblocks this: a SELPH form that calls `synthesize` (the builtin) now gets the full chain for free. Priority forms:
   - **Object-map**: extract objects → map transform per object → recompose
   - **Conditional-per-cell**: classify cells by neighborhood → apply per-class rule
   - **Pattern-repeat**: detect repeating unit → synthesize unit transform → tile
4. **Budget management.** Monitor whether recursive dispatch causes meaningful slowdown. If so, reduce per-strategy sub-budgets or add early-exit heuristics.

---

## 8. Future Directions

### 9.49 Towards semantic understanding: unifying NL and physics via graded equivalence (April 12, 2026)

A design discussion prompted by the convergence of three threads:
the Stage 7 library-reuse wall (§9.44), the NL string curriculum
(§9.47), and the physics meta-curriculum (§9.40–§9.43). All three
hit the same shape of problem: **given context, select the right
search prior**.

#### 9.49.1 The unifying observation

Three capabilities that appear distinct are structurally identical:

| Capability | Surface form | Underlying shape |
|---|---|---|
| Library detection (§9.44) | "this task looks like an oscillator" | context → namespace → search prior |
| Domain detection for NL | "this sentence is about kinematics" | context → namespace → search prior |
| Synonym discovery | "`mv` ≡ `momentum`" | context ��� namespace → search prior |

All three are **graded similarity judgments over witness sets,
anchored by predictively-grounded cores**. Building the mechanism
once buys all three.

The connection to §9.38's three-layer architecture: this mechanism
lives in Layer 2 (curriculum substrate). It extends the
`__decomposers__` / `__types__` namespace infrastructure with a new
namespace concept — `__ontology__` — that bundles primitive
constants, typical operators, and search priors per domain. Domain
detection becomes namespace selection.

#### 9.49.2 Graded equivalence as the core primitive

Replace boolean equality with a graded equivalence predicate:

```
equiv?(a, b) → score ∈ [0, 1]
```

Two terms are synonymous to the degree that **swapping them
preserves correctness across a witness set of contexts**. This is
the symbolic analog of cosine similarity in embedding space, but
the witness set is interpretable and verifiable.

**Witness sets as substrate.** For each term, maintain the contexts
it has been used in and the outputs that resulted. SELPH already
produces this data during synthesis — solve traces are the witness sets.

**Predictive coherence as the loss function.** Wrong synonyms get
pruned because substitutions break predictions.

**Spread by composition and isomorphism.** SELPH's homoiconicity
(§9.36) makes detecting structural isomorphism cheap: two Node
trees that differ only in leaf names but share shape are candidates.

This approach is **coherentist** rather than foundationalist:
symbols don't need external grounding if they have enough internal
cross-validation against predictively-grounded anchors.

#### 9.49.3 Shortcutting: hand-written corpora as symbolic RAG

The architecture supports a clean division of labor:

- **Hand-write seed anchors.** Dimensional types, primitive constants, the first 5–10 grounded terms per domain.
- **Hand-write terminal facts.** Specific equivalences, named theorems, lookup data.
- **Learn the structure.** Which decomposers fire when, which equivalences hold across contexts, which libraries reuse.

**Where SELPH exceeds LLM-RAG:** SELPH can **execute** retrieved facts, **validate** them against witness sets, and **reject** them when they fail. The retriever doesn't have to be perfect.

#### 9.49.4 Prose-to-structure as a meta-learning task

Converting natural-language prose into SELPH nodes is itself a
decomposer — a string-to-Node transformation. Progression:

1. **Rigid templates.** Hand-write 10–20 sentence patterns.
2. **Learn the grammar.** Meta-curriculum generalizes from templates.
3. **Discover synonyms via substitution.** Graded equivalence earns its keep.
4. **Fail gracefully to the LLM oracle.** Per §9.38, Layer 3.

Physics prose is the right starting domain — technical writing uses
a small, regular grammar that 20–30 rigid templates cover.

#### 9.49.5 Constraints and honest limits

1. **Equivalence detection is undecidable in general.** Must be namespace-local.
2. **Sense disambiguation requires term-in-context.** SELPH's namespace machinery handles this.
3. **Witness set explosion.** Pruning heuristics needed.
4. **Non-compositional language is out of scope.** The LLM oracle is the escape valve.
5. **Cold start needs seed anchors.** Bounded work per domain.

#### 9.49.6 Experiments and next steps

**Experiment 1: Graded `equiv?` within physics**
Implement over the 32+ solved physics tasks. Success: correct
identities score ≥0.9; spurious pairings ≤0.1. Pure SELPH.

**Experiment 2: Prose-to-Node for physics**
15 rigid templates. Success: ≥80% parse accuracy on 10 held-out
sentences.

**Experiment 3: Library detection via `equiv?`**
Use graded equivalence as retrieval ranker for M7 library detection.
Success: library reuse fires on ≥3 of 5 oscillator tasks.

#### 9.49.7 Relationship to existing architecture

This extends Layer 2 with three new capabilities:
- **Graded equivalence** as a namespace-local predicate
- **External corpus support** as a retrieval target
- **Prose-to-structure** as a meta-curriculum decomposer

All pure SELPH curriculum work. No Rust kernel changes. No LLM oracle changes.

The per-domain pool pattern (§9.47) provides the dispatch
infrastructure: each domain's pool can include corpus-retrieved
entries ranked by equivalence score rather than fixed priority order.
