# SELPH: Growing System Plan

## From Enumerative Solver to Self-Building Architecture

**Version 0.13 — April 12, 2026**

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
- First-class namespaces with `__types__` and `__decomposers__` registries

### 2.3 Synthesis (synth_v2.rs)

- Priority-weighted interleaved bottom-up search with type-directed pruning
- Deferred materialization (lightweight PendingDesc, on-demand node trees)
- Arity 1–3 support with adaptive type-count caps
- Type reachability filtering (40–65% candidate reduction)
- RL reward propagation (partial match scores adjust pool priorities)
- Early depth extension (targeted composition probes)
- Domain-based component filtering (grid ops excluded for string tasks, etc.)
- Auto-constant extraction, probe-and-filter, observational equivalence dedup
- Epsilon-equivalence for float dedup

**Strategy pipeline:** SELPH decomposers (M-chain) → RD → Flat → BD → HO → D&C → Memo

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
| Grids (ARC) | 9 | **9/9** | Grid builtins + detector Forms 1+2 + chain dispatch |
| Original 3-domain chain | 55 | **55/55** | Sequence → CF → NL, library cascade |
| ARC-AGI-1 (eval) | 400 | **14/400** | Grid synthesis + pool pruning |

### 2.7 CLI Commands

```
selph eval       Evaluate SELPH files or expressions
selph parse      Parse and print AST
selph synth      Synthesize from examples
selph grow       Run curriculum (legacy core)
selph grow-v2    Run curriculum (v2 core, primary)
selph bench      Run stochastic benchmark suite
selph generate   Generate a curriculum in .selph format
selph verify     Verify a program against a spec
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

### 9.50 Post-mortem meta-learning — diagnostic-driven curriculum (April 12, 2026)

Infrastructure for the system to analyze its own failures after a curriculum run and guide what to build next.

#### Kernel changes

- **`StrategyResult` diagnostics:** `output_type` and `m_chain_ran` fields on every synthesis result. Surfaces what type the task wanted and whether the M-chain was consulted.
- **`__curriculum_results__` binding:** `cmd_grow_v2` accumulates per-task result namespaces (name, found, candidates, output-type, m-chain-ran, strategy, spec, test, arity, depth) and binds them into the env after the task loop.
- **`run-post-mortem` hook:** If defined in env, grow-v2 calls it with the results list and prints the returned diagnostics.
- **`synthesize-args` test forwarding fix:** The builtin was not forwarding held-out test pairs to `synthesize_args_with_test`, allowing memorization solutions to pass. Fixed to read the `"test"` field from the spec namespace.

#### `post_mortem.selph` — pure SELPH spec analysis

Classifies each failed task across 6 dimensions:
- **size:** same-size / shrink / grow / reshape / mixed
- **colors:** same-colors / color-subset / color-superset / new-colors
- **constant-out:** whether all training outputs are identical
- **dims-consistent:** whether all outputs share the same dimensions
- **objects:** foreground color count (proxy for object complexity)
- **scale:** integer height ratio between input/output

Also probes every unary grid builtin (depth 1) and depth-2 compositions against each failure's spec data.

#### Key findings

- **ARC-AGI-1 (400 training):** 14/400 solved (M-chain spatial transforms). Post-mortem: same-size=251, shrink=98, grow=36. 20 tasks have integer scaling (2x/3x). Zero tasks solvable by depth-1 or depth-2 builtin probing beyond what M-chain already catches.
- **ARC-AGI-2 (120 eval):** 0/120 solved. Post-mortem: same-size=81, shrink=27, same-colors=60, color-subset=38.
- **Bug found:** `synthesize-args` builtin was not forwarding held-out test data, allowing M-chain Form 5 memorization to pass. The post-mortem's retry loop exposed this — 266 "recovered" tasks were all training-data memorization that failed held-out validation.

#### New grid builtins

- `grid-scale` — scale grid by integer factor
- `grid-tile` — tile grid NxM times
- `grid-fill-enclosed` — fill interior background cells
- `grid-compact` — remove all-zero rows and columns

Added to `m_pool_grid` unary catalog: `grid-fill-enclosed`, `grid-compact`.

#### Scaffolding curriculum (`arc_scaffolding_curriculum.selph`)

8 tasks for grid operations the M-chain doesn't cover: extract_object, scale_2x, scale_3x, tile_2x2, mirror_h, mirror_v, fill_enclosed, compact. With held-out validation to reject memorization.

**Results:** 2/8 solve (extract_object via `grid-trim`, compact via `grid-compact`). Remaining 6 need multi-arg builtins (scale/tile need a factor arg not reachable from arity-1 spec) or depth-2 compositions (mirror needs `grid-hconcat(x, grid-flip-h(x))`).

#### Status

The post-mortem pipeline works end-to-end: run → diagnose → probe → report. The diagnostic breakdown is actionable — it identifies exactly which operation families to prioritize. The scaffolding curriculum demonstrates the pattern: hand-write tasks for needed primitives, solve them to build library, compose for harder tasks.

**Next:** Depth-2 composition probing in the M-chain (binary ops like `grid-hconcat` applied to two unary results). Multi-arg scaffolding for scale/tile. Fill-enclosed bug fix.

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
