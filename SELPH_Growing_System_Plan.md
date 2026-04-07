# SELPH: Growing System Plan

## From Enumerative Solver to Self-Building Architecture

**Version 0.4 — April 6, 2026**

Based on implementation experience with the v0.1 Architecture Spec and the Rust-native migration. Supersedes v0.2 with validated results from the standalone Rust binary.

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
- Hindley-Milner type inference (as second-pass pruning over u8 type tags)
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
- `synthesize` exposed as a SELPH builtin for self-improvement loops
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

### 2.8 Rust Modules (15 source files)
types.rs, parser.rs, eval.rs, synth.rs, hm.rs, library.rs, namespace.rs, induce.rs, divide.rs, verify.rs, abstraction.rs, multitree.rs, stochastic.rs, meta.rs, taskgen.rs

215 tests, all passing.

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

**Context-free language curriculum (20 tasks): 20/20 (100%) in 293 seconds**

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

Phase 2: Learned heuristics (STARTED)
  - Meta-1 replaces hardcoded search ordering
  - Training data: synthesis logs from Phase 1
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

### 9.1 Chained curriculum execution
The train→save→reload loop works. Next: build a standard multi-stage pipeline:
```
selph grow sequence_tasks.selph --library seq_helpers.selph -o model.selph
selph grow formal_lang_tasks.selph --library model.selph -o model.selph
selph grow context_free_tasks.selph --library model.selph -o model.selph
```
Each stage grows the shared library. The probe-and-filter ensures macros from incompatible input formats are automatically excluded. Validate that RL coefficients transfer across domains.

### 9.2 Reduce search space for hard tasks
`equal_ab` (9,373 candidates) and `matched_parens` (35,933) are still expensive. Opportunities:
- **Partial match pruning between depths:** If a depth-1 candidate matches 0 examples AND returns the target type, skip it as an argument for depth-2 compositions entirely (not just deprioritize).
- **Component-level type scoping per depth:** At the final depth, only generate candidates whose return type matches the target. At intermediate depths, allow all useful types.
- **Early termination on partial match:** If a depth-1 candidate matches >50% of examples, immediately try composing it with comparisons/logic before exhausting the full depth-1 pool.

### 9.3 Generalize boolean decomposition to N-ary
Current decomposition tries pairs. Extend to:
- Chains: `(and P (and Q R))` for 3+ predicates
- Mixed: `(and P (or Q R))` for disjunctive sub-conditions
- Numeric: `(and (= (f x) (g x)) (> (h x) 0))` for non-macro compositions
This would handle more complex classification tasks without exhaustive depth-3 search.

### 9.4 Curriculum-driven RL coefficient specialization
Currently one (cold, warm) pair for all tasks. Different task types may benefit from different coefficients:
- Numeric tasks: strong cold penalty (many string candidates to prune)
- Boolean tasks: weaker cold penalty (intermediates cross types frequently)
- String tasks: strong warm bonus (partial string matches are informative)
Store per-domain coefficients in the library, select based on inferred task type.

### 9.5 Meta-2: Learn decomposition as SELPH programs
The boolean decomposition fallback is hardcoded Rust. Reframe it as a synthesis target: given a failed spec + a library of bool macros, find a SELPH program that combines them. Training data: the `(and (equal_ab x) (no_b_before_a x))` solution for `anbn`, etc. This moves decomposition from infrastructure to learned capability.

### 9.6 Deeper curriculum: context-sensitive languages
With a^n b^n solved, the next frontier is context-sensitive patterns:
- a^n b^n c^n (equal counts of three symbols)
- Copy language: ww (string repeated twice)
- Reversal: w^R (string followed by its reverse)
These likely need `reduce`/`fold` over characters or recursive decomposition, pushing the system toward Meta-2 capabilities.

### 9.7 Curriculum design principles (updated)
- **Every builtin needs a teaching task:** `and`/`or`/`not` needed scaffolding tasks before `anbn` could compose them
- **Arity-3 requires depth-aware caps:** cubic enumeration at deeper depths is infeasible; cap by type-matching pool size
- **Macro promotion must preserve semantics:** the `x→s` variable substitution must be word-boundary-aware; broken macros silently poison the library
- **Type filtering is multiplicative:** combining useful-type reachability with return-type gating compounds the savings
- **RL coefficients converge slowly:** most tasks are easy (low difficulty), so coefficient updates are small; learning requires enough hard tasks in the curriculum
- **Static heuristic templates are harmful:** hand-coded "match-output-type" heuristics crush intermediate-type components; learned RL rewards are safer because they adjust pool entries, not component ordering

---

## 10. Success Criteria

The growing system plan succeeds if:

1. **Phase 2 validation:** A learned SELPH heuristic outperforms the default ordering on held-out tasks without domain-specific engineering.

2. **Phase 3 validation:** A learned SELPH decomposer solves tasks that the flat solver + hand-written induction can't.

3. **Phase 4 validation:** A neural SELPH generator (trained on synthesis logs) proposes correct programs in fewer attempts than the enumerative solver.

4. **Phase 5 validation:** The system, given a new domain with new primitives, autonomously designs a curriculum that reaches competence without human-specified task ordering.

Each criterion is measurable with the existing benchmarking infrastructure.
