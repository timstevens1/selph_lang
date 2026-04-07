# SELPH: Growing System Plan

## From Enumerative Solver to Self-Building Architecture

**Version 0.8 — April 7, 2026**

Based on implementation experience with the v0.1 Architecture Spec, the Rust-native migration, the April 6-7 session (decomposition via synthesis, tracing, 7.2x search optimization, namespace literals, memorization), the April 7 session that added: bytecode VM (22.9x eval speedup), `synthesize` builtin `:library`/`:priorities` support, early depth extension for compositional search, optimization curriculum with meta-optimization, NL curriculum, and chained curriculum execution, and the April 7 evening session that added: unified library/tree/namespace loading, TYPE_LIST in the type system, higher-order synthesis via fused map components, 3-word sentence structures, and variable-length sentence tagging via `(string-join (map pos_tag (string-split x " ")) " ")`.

---

## 1. Core Thesis

The model architecture is not specified — it is *grown* by the curriculum. Each curriculum stage teaches the system a capability, and the learned capability becomes infrastructure for the next stage. The system bootstraps from a simple enumerative solver into an increasingly autonomous program synthesis engine, where every component — search, evaluation, heuristics, library management, task decomposition — is itself a SELPH program learned through earlier curriculum stages.

The architecture IS the curriculum. Change the curriculum, change the architecture.

---

## 2. What Exists (April 6, 2026)

### 2.1 Standalone Rust Binary
The system now runs as a single Rust binary (`selph`) with zero runtime dependencies. The Python/PyO3 implementation remains for reference but is no longer the primary execution path.

- S-expression parser with full EBNF from v0.1 spec
- Tree-walking evaluator with 55+ builtins
- First-class namespaces: functions, data, and cache in the same tree
- Hindley-Milner type inference (as second-pass pruning over u8 type tags: NUM=0, STR=1, BOOL=2, LIST=3, ANY=255)
- Macro system (defmacro) with letrec semantics (Rc<RefCell> shared scope)
- Eval depth limit (256) prevents stack overflow from deep macro chains

### 2.2 Builtins (55 total)
- **Arithmetic:** add, subtract, multiply, divide, modulo, abs, negate, min, max, floor, ceil, round, pow, sqrt, log
- **Comparison:** <, >, <=, >=, =, not, even, odd
- **String:** upper, lower, reverse, trim, length, contains, split, join, concat, nth, slice, starts-with, ends-with, replace, chars
- **Character:** char-code, code-char (char/number conversion)
- **String analysis:** count-char (occurrence counting)
- **List:** list, head, tail, length, cons, nth, slice, sort, reverse, append, range, contains, zip, enumerate
- **Higher-order:** map, reduce, filter, apply, identity
- **Type:** type-of, number?, string?, bool?, list?, nil?, function?
- **Namespace:** get, put, keys, values, merge, size, flatten, has, get-or, empty, ns?
- **Meta/self-hosting:** synthesize, synthesize-optimize, eval-source, eval-in, try, define, error
- **Introspection:** `__builtins__` namespace with type metadata for all builtins

### 2.3 Spec and Verification (Rust-native)
- Specs as first-class values with typed goals (Levels 0-3)
- verify.rs: VerificationResult with type/shape/constraint gates + goal scoring
- Reward computation: `hard_gate * (w_goal + w_parent + w_global)`
- Held-out validation for generalization (`--validate` flag)
- Optimization synthesis: `--minimize`/`--maximize` for objective-driven search

### 2.4 Synthesis
- **Priority-weighted interleaved search:** candidates sorted by combined priority of component + arguments. High-value compositions tried first regardless of which component they use.
- **Deferred materialization:** pending candidates stored as lightweight descriptors (component index + pool indices), node trees built on-demand during testing. Eliminates gigabyte allocations from cloned node vectors.
- **Arity 1-3 support:** unary, binary, and ternary compositions (e.g., `string-replace` takes 3 args). Arity-3 has adaptive type-count cap to prevent cubic blowup at deeper depths.
- **Type reachability filtering:** computes which types can eventually produce the target output type through available component chains. Pool entries with unreachable types are excluded, reducing search space 40-65%.
- **RL reward propagation:** partial match scores (fraction of examples correct) adjust pool entry priorities during synthesis. Entries with partial matches get boosted; zero-match same-type entries get penalized. Coefficients (cold_penalty, warm_bonus) learned online and persisted across runs.
- **Boolean decomposition fallback:** when flat synthesis fails on bool-target tasks, tries all (and P Q), (or P Q), (not P) combinations of bool-returning macros. O(macros²), essentially instant.
- Bottom-up enumerative search with HM type-directed pruning
- If-expression synthesis with expected-output matching
- Divide-and-conquer for multi-way classification
- Failure-driven induction (intermediate value decomposition)
- Observational equivalence deduplication
- **Bytecode VM fast path:** candidates compiled to bytecodes for evaluation (22.9x speedup). Macros with unsupported constructs (e.g., `ns` special form) automatically fall back to tree-walker.
- **Early depth extension:** after each depth-1 candidate, immediately try composing it with arity-1 components whose return type matches the target. Finds `(is_noun (last_word x))` in 8 candidates instead of exhausting 200K. Key enabler for compositional NL tasks.
- **Auto-constant extraction:** unique characters and numbers from examples added to pool, scored by frequency
- **Probe-and-filter:** macros that error on actual inputs automatically excluded
- **Trivial promotion skip:** prevents self-referential macros (e.g., `(defmacro f (x) (f x))`)

### 2.5 Library System
- Promotion: solved programs become single-step primitives (immediate, within the curriculum loop)
- Extraction: anti-unification + compression scoring (abstraction.rs)
- Pruning: observational + builtin equivalence + usage tracking
- **Trained model = library file:** the .selph output IS the model. Contains solutions, extracted abstractions, and learned heuristic. Loading it restores the full learned state.
- Trivial wrapper detection prevents re-promoting existing macros on reload

### 2.6 Search Strategy
- **Interleaved priority learning:** component priorities update after every solve (learn_rate boost). No explicit meta-synthesis needed — the priority weights capture the signal.
- Programmable heuristics via `make_selph_scorer` / `make_selph_depth_filter`
- First-class namespace trees: `--tree` flag on synth and grow commands
- `synthesize` exposed as a SELPH builtin for self-improvement loops; accepts `:library` (namespace of macros) and `:priorities` (namespace of component name → priority boost)
- Meta-heuristic synthesis available via `--meta` flag (failure-triggered, not periodic)

### 2.7 CLI Commands (10 total)
```
selph eval       Evaluate SELPH files or expressions
selph parse      Parse and print AST
selph synth      Synthesize from examples (--tree, --minimize, --maximize, --validate)
selph grow       Run curriculum (--meta, --extract, --validate, --filter, --tree)
selph bench      Run stochastic benchmark suite
selph generate   Generate a curriculum in .selph format
selph verify     Verify a program against a spec
selph multi-synth Synthesize across namespace trees
selph repl       Interactive REPL
selph help       Show usage
```

### 2.8 Rust Modules (17 source files)
types.rs, parser.rs, eval.rs, synth.rs, hm.rs, library.rs, namespace.rs, induce.rs, divide.rs, verify.rs, abstraction.rs, multitree.rs, stochastic.rs, meta.rs, taskgen.rs, intern.rs, vm.rs

222 tests, all passing.

### 2.9 Validated Results

**Sequence curriculum (13 tasks): 13/13 (100%)**

| Task | Candidates | Solution |
|------|-----------|----------|
| const_1 | 2 | `1` |
| const_5 | 6 | `5` |
| identity | 13 | `(idx x)` |
| double_idx | 1,987 | `(add (idx x) (idx x))` |
| triple_idx | 2,395 | `(add (idx x) (double_idx x))` |
| last_plus_1 | 1,510 | `(add (count-char x x) (last x))` |
| last_minus_1 | 1,721 | `(subtract (v3 x) (count-char x x))` |
| **squares** | **5,930** | **`(multiply (idx x) (idx x))`** |
| triangular | 2,099 | `(add (idx x) (last x))` |
| fibonacci | 9,574 | `(add (v3 x) (v2 x))` |
| double_prev | 2,509 | `(add (last x) (last x))` |
| **cubes** | **9,794** | **`(multiply (idx x) (squares x))`** |
| idx_plus_last | 59 | `(triangular x)` |

**Context-free language curriculum (20 tasks): 20/20 (100%) in 15.7 seconds** (was 293s before optimizations; 96s after search opts; 15.7s after bytecode VM)

| Task | Candidates | Solution |
|------|-----------|----------|
| count_a | 14 | `(count-char x "a")` |
| count_b | 13 | `(count-char x "b")` |
| str_len | 40 | `(string-length x)` |
| equal_ab | 9,373 | `(= (count-char x "a") (count-char x "b"))` |
| more_a | 8,895 | `(< (count-char x "b") (count-char x "a"))` |
| remove_a | 48 | `(string-replace x "a" "")` |
| remove_b | 32 | `(string-replace x "b" "")` |
| a_to_b | 24 | `(string-replace x "a" "b")` |
| starts_a | 9 | `(string-starts-with x "a")` |
| ends_b | 21 | `(string-ends-with x "b")` |
| all_a | 790 | `(string-starts-with (string-replace x x x) (string-replace x "b" "a"))` |
| not_starts_a | 118,423 | compositional via string-replace |
| starts_a_and_ends_b | 1,402 | compositional via string-replace |
| starts_a_or_ends_b | 124,237 | `(or (string-starts-with x "a") (string-ends-with x "b"))` |
| no_b_before_a | 812 | `(string-starts-with (string-replace x x x) (string-replace x "b" ""))` |
| **anbn** | **61,386** | **`(and (equal_ab x) (no_b_before_a x))`** |
| palindrome | 1,502 | compositional via string-replace |
| count_open | 14 | `(count-char x "(")` |
| count_close | 15 | `(count-char x ")")` |
| matched_parens | 35,933 | `(= (count-char x "(") (count-char x ")"))` |

Key results:
- **a^n b^n recognized:** `(and (equal_ab x) (no_b_before_a x))` — composes two promoted macros with boolean logic
- **Arity-3 synthesis:** `string-replace` tasks solved compositionally (e.g., `(string-replace x "a" "")`)
- **Library cascade:** 20 macros promoted, each building on prior solutions
- **Boolean logic:** `and`, `or`, `not` enable predicate composition
- **Type filtering:** useful-type reachability analysis reduces candidates 40-65%
- **RL reward propagation:** partial match scores adjust pool priorities within each synthesis run

**Character-native formal language (12 tasks): 10/12 (83%)**
- ascending: `(next-char (c3 x))` — character arithmetic
- step2: `(next-char (next-char (c3 x)))` — double composition

### 2.10 Self-Hosting Infrastructure
SELPH programs can now express their own infrastructure:

```lisp
; The self-improvement loop works end-to-end:
(do (define result (synthesize spec))
    (define learned (eval-source (ns-get result "source")))
    (learned 7))  ; => 14
```

Example SELPH programs in `examples/`: curriculum.selph, scoping.selph, taskgen.selph, verify.selph, heuristics.selph, filter.selph

### 2.11 Bytecode VM (22.9x eval speedup)
Candidates compiled to bytecodes via `vm.rs`. String interning (`intern.rs`) replaces string comparisons with integer lookups. Rc-wrapped node pools eliminate cloning. Combined effect: CF curriculum 96s → 15.7s (6.1x on top of the 7.2x search optimization = **18.6x total** from baseline 293s).

VM supports: builtins, pre-compiled macro calls, if-expressions, constants. Unsupported constructs (e.g., `ns` special form in memorized macros) fall back to tree-walker automatically.

### 2.12 Chained Curriculum
Validated: sequence (13 tasks) → CF (20 tasks) = 33/33 (100%). Library grows from 6 helpers → 17 after sequence → 37 after CF. Cross-domain macro filtering via `scope_library_for_task` ensures sequence macros don't pollute CF search space.

### 2.13 Optimization Curriculum & Meta-Optimization
`(opt-task ...)` form in curriculum runner dispatches to `synthesize_optimize`. Validated on 5 tasks including constant optimization, constrained optimization, and **meta-optimization** — a fitness function that calls `synthesize` internally to measure candidate count, enabling the system to optimize its own search strategy.

First self-improvement loop: `meta_count_char_boost` synthesized a priority boost by running synthesis inside the fitness function (0.7s, 1754 candidates).

### 2.14 Natural Language Curriculum
**22/22 solved (100%).** `examples/nl_tasks.selph` with `nl_helpers.selph` (loaded as `--library`). 6 stages:

- **Stage 0 (word ops):** 5/5 — `first_char`, `last_char` synthesized; word extraction composed from library helpers (`first_word`, `last_word`)
- **Stage 1 (vocabulary):** `is_noun`, `is_verb`, `is_adj` memorized as namespace lookups (11-12 entries each). `is_article` found computable pattern `(or (starts-with "a") (ends-with "e"))`
- **Stage 1b (morphology):** `is_plural` → `(string-ends-with x "s")`, `is_gerund` → `(string-ends-with x "g")`
- **Stage 2 (sentence patterns):** compositions found via flat synthesis + BD:
  - `starts_with_article` → `(is_article (first_word x))` (12.7K cand)
  - `ends_with_noun` → `(is_noun (last_word x))` (10.9K cand)
  - `article_noun` → `(and (starts_with_article x) (ends_with_noun x))` (BD, instant)
- **Stage 3 (POS tagging):** D&C found a **decision tree** for POS tagging:
  `(if (is_noun x) "noun" (if (is_verb x) "verb" (if (is_adj x) "adj" (if (is_article x) "art" "pron"))))`
- **Stage 4 (per-position tagging):** `tag_first`, `tag_second`, `tag_last` — D&C if-expression cascades composing vocabulary predicates with word extractors
- **Stage 5 (structure assembly):** `structure_2w` — full grammatical structure classifier via D&C:
  `(if (is_article (first_word x)) "art noun" (if (is_noun (last_word x)) "adj noun" ...))`
- **Stage 6 (3-word structures):** `structure_3w` — D&C decision tree for 3-word sentences:
  `(if (is_adj (second_word x)) "art adj noun" (if (is_article (first_word x)) "art noun verb" ...))`

**Key insight:** D&C discovers decision trees for fixed-length structure classification. For variable-length sentences, the system uses a different strategy: `(string-join (map pos_tag (string-split x " ")) " ")` — higher-order synthesis via fused map components. Both approaches compose library primitives, but one enumerates output patterns while the other generalizes.

### 2.15 Unified Library Loading
Libraries, trees, and namespaces are unified into a single concept. A library file is evaluated; if it produces a `Value::Namespace`, its `RustMacro` entries are extracted as synthesis components. Falls back to structural `defmacro` parsing for legacy files. `--library` replaces `--tree` (kept as alias for backward compatibility).

### 2.16 TYPE_LIST and Higher-Order Synthesis
`TYPE_LIST = 3` added to the type system alongside NUM, STR, BOOL. List operations (`string-split`, `string-join`, `head`, `tail`) registered as synthesis components with proper type annotations. Higher-order synthesis via **fused map components**: for each unary macro `m`, the synthesizer auto-generates `map_m(list) → list` that materializes as `(map m list)` during candidate construction. **Chained early depth extension** follows intermediate types — when a depth-1 composition returns LIST (not the target STR), it immediately probes arity-2 compositions like `string-join` to reach the target in one step.

Validated: `(string-join (map pos_tag (string-split x " ")) " ")` found in 230 candidates (0.054s). Works for 2-word, 3-word, and any-length sentences.

The `synthesize` builtin accepts `:library` (namespace of macros) for SELPH-level library control, and `:priorities` for search tuning.

---

## 3. The Growing System Plan

### 3.1 Principle

Every component of the solver is a potential curriculum target. Instead of hardcoding infrastructure in Python/Rust and training a model to use it, we:

1. Run the Python/Rust infrastructure to generate examples of what each component does
2. Define a curriculum stage that teaches a SELPH program to do the same thing
3. Replace the infrastructure component with the learned SELPH program
4. The learned program becomes a primitive for the next stage

Each replacement is validated: does the SELPH version match the performance of the Python/Rust version it replaces? If yes, promote. If no, keep training.

### 3.2 Meta-Curriculum Stages

The curriculum has two axes: **task complexity** (what problems to solve) and **meta-level** (what part of the solver to learn). The meta-levels are:

```
Meta-0: Solve tasks           (current: Stages 0-5)
Meta-1: Learn search strategy (started: heuristics as SELPH programs)
Meta-2: Learn decomposition   (started: induction/D&C as Python)
Meta-3: Learn pattern extraction (library extraction as Python)
Meta-4: Learn candidate generation (expansion as Rust)
Meta-5: Learn evaluation strategy (which programs to eval first)
```

Each meta-level uses the infrastructure from the level below it, and the learned programs become infrastructure for the level above.

### 3.3 Meta-0: Solve Tasks (DONE)

**What:** Given a spec, find a program that satisfies it.
**Learned capability:** Composition of primitives (arithmetic, strings, branching).
**Infrastructure used:** Evaluator, type checker, enumerative search.
**Validated:** 6-stage curriculum, 98% solve rate.

### 3.4 Meta-1: Learn Search Strategy

**What:** Given a task's features, predict which components to try first.
**Training data:** For each solved task, record (task_features → components_used).
**Spec:**
```lisp
(:spec :type (-> namespace number)
       :goal (:minimize
         (lambda (heuristic)
           (total-candidates-using heuristic task-suite))))
```
**Current status:** Heuristics as SELPH programs work. Interleaved online learning updates priorities. Domain-specific heuristic gives 23.5x speedup.
**What's missing:** The heuristic needs to read task features (not just component metadata) to generalize across task types.

**Curriculum design:**
- Stage M1.0: Learn to predict output type from examples
  - Input: a set of (input, output) pairs
  - Output: "number" or "string"
  - This is itself a synthesis task
- Stage M1.1: Learn to predict useful component categories
  - Input: task features (output type, input type, example patterns)
  - Output: priority boost for component categories (arithmetic, string-ops, comparisons)
- Stage M1.2: Learn task-specific heuristics
  - Input: full task description (spec as namespace)
  - Output: per-component priority assignment
  - Trained by running synthesis with candidate heuristics and measuring candidates

### 3.5 Meta-2: Learn Decomposition

**What:** Given a failed spec, decompose it into solvable sub-problems.
**Training data:** For each task solved by induction or D&C, record (failed_spec → decomposition_strategy → sub-specs).
**Current status:** Induction (intermediate value search) and D&C (partition by output value) are Python functions.

**Curriculum design:**
- Stage M2.0: Learn to identify decomposable specs
  - Input: a spec that the flat solver failed on
  - Output: "decomposable" or "not decomposable"
  - Training data: specs that induction/D&C solved vs. those that failed
- Stage M2.1: Learn to choose intermediate values
  - Input: a failed spec's examples
  - Output: which unary function to apply to inputs as a bridge
  - Training data: successful inductions (e.g., "string-length was the bridge")
- Stage M2.2: Learn to partition examples for D&C
  - Input: examples with multiple output values
  - Output: which condition separates the groups
  - Training data: successful D&C decompositions

### 3.6 Meta-3: Learn Pattern Extraction

**What:** Given a corpus of solved programs, identify reusable abstractions.
**Training data:** For each library extraction run, record (corpus → extracted_abstractions).
**Current status:** Anti-unification + compression scoring in Python.

**Curriculum design:**
- Stage M3.0: Learn to identify repeated sub-trees
  - Input: a list of programs
  - Output: the most common sub-expression
- Stage M3.1: Learn to generalize (anti-unify)
  - Input: two similar programs
  - Output: the common pattern with parameter holes
- Stage M3.2: Learn to score abstractions
  - Input: a candidate abstraction + corpus
  - Output: compression score (is it worth adding?)

### 3.7 Meta-4: Learn Candidate Generation

**What:** Given a set of components and a target type, generate the most promising candidate programs.
**Training data:** For each synthesis run, record (components + target_type → winning_program + its depth in the search order).
**Current status:** Bottom-up enumeration in Rust, ordered by priority.

This is the most critical meta-level — it's where the enumerative solver becomes a neural generator. Instead of enumerating all type-valid programs and testing each, the system learns to propose programs directly.

**Curriculum design:**
- Stage M4.0: Learn to complete partial programs
  - Input: a program with a hole `(add ? 1)`
  - Output: the expression that fills the hole
  - Training data: solved programs with sub-expressions masked
- Stage M4.1: Learn to generate programs from specs
  - Input: a spec (examples)
  - Output: a complete program
  - Training data: (spec, solution) pairs from the synthesis history
- Stage M4.2: Learn to generate programs conditioned on library
  - Input: a spec + available library (as namespace)
  - Output: a program using library primitives
  - This is where the system learns to use its own abstractions

### 3.8 Meta-5: Learn Evaluation Strategy

**What:** Given a set of candidate programs, decide which to evaluate first.
**Training data:** For each synthesis run, record which candidates were eventually correct vs. which were dead ends.
**Current status:** Observational equivalence deduplication.

**Curriculum design:**
- Stage M5.0: Learn to predict program correctness without running it
  - Input: a program + spec
  - Output: estimated probability of correctness
  - Training data: programs labeled as correct/incorrect from synthesis runs
- Stage M5.1: Learn to rank candidates
  - Input: a list of candidate programs + spec
  - Output: ranking by estimated quality
  - This replaces the enumerative "test everything" with "test the best first"

---

## 4. The Bootstrapping Sequence

The meta-levels don't all need to be learned before the system is useful. The bootstrapping sequence is:

```
Phase 1: Enumerative solver (DONE)
  - Validates the architecture
  - Generates training data for all meta-levels
  - Establishes the curriculum framework

Phase 2: Learned heuristics (VALIDATED)
  - Meta-1 replaces hardcoded search ordering
  - First loop validated: meta_count_char_boost synthesized a priority
    boost by running synthesis inside the fitness function (0.7s)
  - The synthesize builtin accepts :priorities and :library for
    SELPH-level control of search strategy
  - Training data: synthesis logs from Phase 1 + chained curriculum (33 tasks)
  - Validation: does learned heuristic match domain-specific hand-written ones?

Phase 3: Learned decomposition
  - Meta-2 replaces hardcoded induction/D&C
  - Training data: decomposition logs from Phase 1
  - Validation: can the system decompose novel tasks it hasn't seen?

Phase 4: Neural candidate generation
  - Meta-4 replaces enumerative search with direct program proposal
  - Training data: (spec → solution) pairs from Phases 1-3
  - This is where the system stops being enumerative and starts being generative
  - Requires constrained decoding (§3.1 of v0.1 spec)

Phase 5: Self-improving loop
  - Meta-3 (pattern extraction) and Meta-5 (eval strategy) learned
  - The system generates its own training data by solving tasks
  - Library extraction is itself a learned program
  - The curriculum generates new stages autonomously
```

### 4.1 Phase Transitions

Each phase transition is validated by a performance test:

| Transition | Validation |
|-----------|------------|
| Phase 1→2 | Learned heuristic matches hand-written heuristic on held-out tasks |
| Phase 2→3 | Learned decomposition solves tasks that flat synthesis can't |
| Phase 3→4 | Neural generator solves tasks faster than enumerative + heuristic |
| Phase 4→5 | Self-generated curriculum stages produce genuine new capabilities |

---

## 5. Training Data Generation

The key insight: **the enumerative solver is a training data factory.** Every synthesis run produces:

1. **(spec, solution)** pairs — training data for Meta-4 (program generation)
2. **(task_features, components_used)** pairs — training data for Meta-1 (heuristics)
3. **(failed_spec, decomposition, sub_solutions)** triples — training data for Meta-2
4. **(corpus, abstractions)** pairs — training data for Meta-3
5. **(candidate_set, correct_candidate)** pairs — training data for Meta-5

The Rust synthesizer can generate this data fast. A single curriculum run over 62 tasks produces ~60 (spec, solution) pairs. Running 1000 variant curricula (different random seeds, task orderings, component sets) produces 60K training examples.

### 5.1 Data Quality

Not all training data is equal:
- Solutions found in few candidates are "easy" examples — good for early curriculum stages
- Solutions found in many candidates are "hard" examples — good for later stages
- Failed tasks produce negative examples — "the system couldn't solve this"
- Induced solutions (via decomposition) produce meta-examples — "how to decompose"

The curriculum for each meta-level should start with easy examples and progress to hard ones, mirroring the task-level curriculum.

### 5.2 Data Freshness

As the system improves (better heuristics, better decomposition), the training data should be regenerated. Old data reflects the old system's capabilities. New data reflects the improved system's capabilities. This is the wake-sleep cycle from §6.1 Loop 3:
- **Wake:** Solve tasks, collect solutions
- **Sleep:** Train on collected solutions, extract patterns
- **Wake again:** Solve with improved system, collect better solutions

---

## 6. The Curriculum IS the Architecture

### 6.1 No Fixed Architecture

Traditional ML: choose an architecture (transformer, SSM, etc.), then train it on data.
SELPH: choose a curriculum, then grow the architecture from it.

The "architecture" at any point is the set of learned SELPH programs that constitute the solver. At Phase 1, it's just builtins. At Phase 2, it's builtins + learned heuristic. At Phase 4, it's builtins + heuristic + decomposer + generator. Each curriculum stage adds a new learned component.

### 6.2 Tuning via Curriculum

To make the system better at a specific domain:
- Add domain-specific tasks to the task-level curriculum
- Add domain-specific primitives to the component library
- Run the meta-curriculum to learn domain-specific heuristics and decomposition strategies

To make the system more general:
- Add diverse tasks spanning many domains
- Run longer meta-curriculum stages to learn cross-domain patterns
- Let library extraction discover domain-agnostic abstractions

### 6.3 Efficiency via Curriculum

The order and composition of curriculum stages determines training efficiency:
- Start with the easiest tasks → fewer candidates needed → more training data per second
- Promote solutions immediately → later tasks benefit within the same stage
- Prune redundant library entries → smaller search space
- Learn heuristics online → each task informs the next

The curriculum design problem is: **what sequence of tasks, at what difficulty gradient, with what meta-level training, produces a system that reaches a given capability in the fewest total synthesis steps?**

This is itself an optimization problem — and it could eventually be solved by the system itself (meta-meta-level: learn to design curricula).

---

## 7. Relationship to v0.1 Spec

### 7.1 What Carries Forward
- The language (§2): unchanged
- Type/shape system (§3): unchanged
- Spec system (§3.3-3.6): unchanged, extended with `:minimize`/`:maximize`
- Reward structure (§6.2): unchanged
- Namespace system (§12): built and validated
- Curriculum bootstrapping (§11): validated, central to everything

### 7.2 What Changes
- **Self-hosting (§13):** Not a one-time "flip" from Python to SELPH. Instead, a gradual curriculum-driven growth where each component is learned incrementally.
- **MVP (§14):** Steps 4-7 (neural training) are reframed as Phase 4 of the growing system, reached after Phases 2-3 build up the meta-capabilities.
- **Differentiable evaluator (§4.3):** May not be needed if the neural generator learns to propose good programs directly (bypassing gradient-based parameter optimization).
- **Tree-structured context (§5):** Still needed for neural generation, but the context is now the full namespace tree, not just a program tree path.

### 7.3 What's New
- **Meta-curriculum:** The curriculum has two axes (task complexity × meta-level), not just one.
- **Training data as a byproduct:** Every synthesis run produces training data for higher meta-levels.
- **Interleaved learning:** Priorities update online during the curriculum, not in a separate training phase.
- **Rust acceleration:** The enumerative solver runs fast enough to be a training data factory.

---

## 8. Completed Steps

### 8.1 ~~Performance: Interleaved search memory optimization~~ ✓
Replaced cloned node vectors with lightweight `PendingDesc` descriptors (component index + pool indices + score). Node trees materialized on-demand during testing. Eliminates gigabyte allocations.

### 8.2 ~~Heuristic optimization via solution ranking~~ ✓
Pool snapshots captured during synthesis enable O(snapshot_size) rank-based evaluation of candidate heuristics. Infrastructure built but superseded by online RL approach (see 8.6).

### 8.3 ~~Context-free language curriculum~~ ✓
**20/20 tasks solved (100%).** Full curriculum: counting → equality → string replacement → structure checks → boolean logic → a^n b^n recognition → palindrome → bracket matching. Key enablers: arity-3 synthesis for `string-replace`, boolean logic components (`and`/`or`/`not`), type reachability filtering.

### 8.6 ~~Online RL reward propagation~~ ✓
Partial match scores during synthesis adjust pool entry priorities. Cold penalty (zero-match entries) and warm bonus (partial matches) coefficients learned online after each solve and persisted to library. Replaces static heuristic templates.

### 8.7 ~~Type reachability filtering~~ ✓
Fixed-point computation of "useful types" — types that can eventually produce the target output through some component chain. Pool entries with unreachable types excluded. 40-65% candidate reduction on sequence tasks, critical for enabling arity-3 without blowup.

### 8.8 ~~Arity-3 synthesis~~ ✓
Candidate generation and materialization extended to arity-3 with adaptive type-count cap (max 15 matching entries per argument position) to prevent cubic blowup at deeper depths.

### 8.9 ~~Boolean decomposition fallback~~ ✓
When flat synthesis fails on bool-target tasks, tries all `(and P Q)`, `(or P Q)`, `(not P)` combinations of bool-returning macros. O(macros²).

---

## 9. Immediate Next Steps

### ~~9.1 Chained curriculum execution~~ ✓
Validated: sequence (13 tasks) → CF (20 tasks) = 33/33 (100%). Library grows from 6 helpers → 17 after sequence → 37 after CF. Cross-domain macro filtering ensures sequence macros don't pollute CF search space. RL coefficients transfer across domains.

### ~~9.2 Early depth extension~~ ✓
Implemented as targeted composition probe: after each depth-1 candidate, immediately try composing it with arity-1 components whose return type matches the target. Finds `(is_noun (last_word x))` in 8 candidates instead of exhausting 200K. Combined with bytecode VM, the CF curriculum runs in 15.7s (was 293s baseline).

### 9.4 Curriculum-driven RL coefficient specialization
Currently one (cold, warm) pair for all tasks. Different task types may benefit from different coefficients. Per-domain coefficients could be stored as namespace metadata and selected based on inferred task type. The `:priorities` field in the `synthesize` builtin partially addresses this.

### ~~9.5 Meta-2: Learn decomposition as SELPH programs~~ ✓
The `synthesize` builtin accepts `:library` — a SELPH program can restrict the component set and call synthesis on sub-problems. Decomposition IS parameterized synthesis, expressible end-to-end from SELPH.

### 9.6 Deeper curriculum: context-sensitive languages
With a^n b^n solved, the next frontier is context-sensitive patterns:
- a^n b^n c^n (equal counts of three symbols)
- Copy language: ww (string repeated twice)
- Reversal: w^R (string followed by its reverse)
These likely need `reduce`/`fold` over characters or recursive decomposition.

### ~~9.7 Optimization curriculum~~ ✓ (Stage 0-2)
`(opt-task ...)` form in curriculum runner. Stage 0-1 validated (constant + constrained optimization). Stage 2 (meta-optimization) validated — fitness function calls `synthesize` internally to optimize search priorities. First self-improvement loop: `meta_count_char_boost` found in 0.7s.

Remaining: Stage 3 (multi-task meta-optimization across full task suite), Stage 4 (learned component filtering).

### 9.8 Vector/tensor builtins
Prerequisite for optimization curriculum Stage 4+ (weight vector learning, linear models). Not yet started.

### ~~9.10 Natural language curriculum~~ ✓ (Stage 0-6)
22/22 tasks solved. Full pipeline: word ops → vocabulary memorization → morphological rules → sentence pattern recognition → POS tagging → per-position tagging → structure classification (2-word and 3-word via D&C).

### ~~9.12 3-word sentence structures~~ ✓
`structure_3w` solved via D&C decision tree composing `is_adj`, `is_article`, `is_noun` with `second_word`, `first_word`. Required unified library loading so `nl_helpers.selph` word extractors were available as synthesis components.

### ~~9.14 `map` + `string-join` as synthesis components~~ ✓
TYPE_LIST (u8=3) added to the type system. `string-split`, `string-join`, `head`, `tail` registered as components. Fused map components (`map_<macro>`) auto-generated for each unary macro. Chained early depth extension probes intermediate types. Variable-length sentence tagging solved: `(string-join (map pos_tag (string-split x " ")) " ")` in 230 candidates.

**Remaining NL directions:**
- **Richer grammar:** Subject-verb agreement, determiner-noun agreement. Combines morphological rules with exception tables: `(or (is_regular_plural x) (ns-get-or irregular_plurals x false))`.
- **Multi-word expressions:** Recognize phrases like "ice cream", "New York" — requires lookahead beyond single-word POS tagging.
- **Parse trees:** Beyond flat structure strings, produce hierarchical representations: `(np (art "the") (noun "cat"))`. Requires namespace-valued outputs.
- **`reduce`/`fold` as synthesis components:** For context-sensitive patterns (a^n b^n c^n) and accumulator-based processing.

### 9.11 Curriculum design principles (updated)
- **Every builtin needs a teaching task:** `and`/`or`/`not` needed scaffolding tasks before `anbn` could compose them
- **Arity-3 requires depth-aware caps:** cubic enumeration at deeper depths is infeasible; cap by type-matching pool size
- **Macro promotion must preserve semantics:** the `x→s` variable substitution must be word-boundary-aware; broken macros silently poison the library
- **Type filtering is multiplicative:** combining useful-type reachability with return-type gating compounds the savings
- **Early depth extension finds compositions that exhaustive search misses:** For NL tasks, `(is_noun (last_word x))` is a depth-2 composition found in 8 candidates via targeted probe, vs 200K+ via exhaustive enumeration
- **VM fallback is essential for correctness:** Macros with `ns` (namespace) constructs can't be VM-compiled. Candidates calling these macros must fall through to the tree-walker, not fail silently
- **Two kinds of knowledge:** Computable (rules) and associative (facts). Both become macros. The memorization fallback handles facts; synthesis handles rules. Compositions bridge them: `(or (rule x) (has exceptions x))`
- **Spurious correlations in examples:** The synthesizer finds any pattern that fits training data. NL examples must include adversarial cases that break surface-level string patterns (e.g., "apple" is not an article even though it starts with "a")
- **D&C is the key to multi-output classification:** When the output has multiple distinct values (e.g., "noun", "verb", "adj", "art", "pron"), divide-and-conquer finds if-expression decision trees that compose library predicates. This is more powerful than flat synthesis — it composes across output categories, not just within a single expression.
- **The system discovers classifiers, not string manipulators:** For `structure_2w`, instead of `concat(tag_first(x), " ", tag_last(x))`, D&C found `(if (is_article (first_word x)) "art noun" ...)` — a decision tree. The curriculum doesn't need `concat` in the component set; D&C with if-expressions handles classification naturally.
- **Libraries, trees, and namespaces are the same thing:** A library IS a namespace. Loading a file evaluates it and extracts `RustMacro` entries from the resulting namespace. No separate `--tree` concept — everything goes through `--library`. Homoiconicity demands it: data must be expressible as code, and code must be loadable as data.
- **Fused components bridge higher-order and first-order synthesis:** Instead of teaching the synthesizer about function-typed pool entries, generate `map_<macro>` components that materialize as `(map macro list)`. The synthesizer treats them as regular arity-1 list→list operations. This is decomposition: higher-order synthesis becomes first-order + naming.
- **Chained early depth extension catches multi-step type conversions:** The original probe only tried compositions whose return type matched the target. With TYPE_LIST, the answer requires LIST as an intermediate: `string-split → map → string-join`. The chained probe follows intermediate useful types, testing arity-2 compositions on the result.

### 9.13 Next priorities (updated April 7, 2026)

With 3-word structures, unified loading, TYPE_LIST, and `map` synthesis all complete, the priority shifts toward scaling and meta-learning:

**Highest-value next steps:**

1. **Full chained curriculum: sequence → CF → NL.** Chain all three domains into one pipeline: 13 + 20 + 22 = 55 tasks, ~40+ macros. This is the training set for real meta-optimization. Tests cross-domain library scaling.

2. **Meta-optimization Stage 3: multi-task heuristic learning.** With 55 tasks and traces, synthesize a priority function that minimizes total candidates across the full suite. Uses the `:priorities` field in the `synthesize` builtin.

3. **Types as a SELPH program.** The type vocabulary (NUM, STR, BOOL, LIST) is currently hardcoded. Making type inference a curriculum target lets the system learn to expand its own type vocabulary. A type inference stage teaches "if `string-split` produces it, and `head` consumes it, they share a type."

4. **`reduce`/`fold` as synthesis components.** Extends higher-order synthesis beyond `map`. Needed for context-sensitive languages (a^n b^n c^n) and accumulator-based processing. Same fused-component approach: `reduce_<macro>` auto-generated for each binary macro.

**Lower priority (infrastructure):**
- Fix the `synthesize` builtin's `:library` extraction for defmacro-captured macros
- 9.6 context-sensitive languages (a^n b^n c^n) — needs `reduce`/`fold`
- 9.8 vector/tensor builtins — prerequisite for linear models
- VM compilation of `ns` in the candidate path

---

## 10. Success Criteria

The growing system plan succeeds if:

1. **Phase 2 validation:** A learned SELPH heuristic outperforms the default ordering on held-out tasks without domain-specific engineering.

2. **Phase 3 validation:** A learned SELPH decomposer solves tasks that the flat solver + hand-written induction can't.

3. **Phase 4 validation:** A neural SELPH generator (trained on synthesis logs) proposes correct programs in fewer attempts than the enumerative solver.

4. **Phase 5 validation:** The system, given a new domain with new primitives, autonomously designs a curriculum that reaches competence without human-specified task ordering.

Each criterion is measurable with the existing benchmarking infrastructure.
