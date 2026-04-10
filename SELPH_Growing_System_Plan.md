# SELPH: Growing System Plan

## From Enumerative Solver to Self-Building Architecture

**Version 0.12 — April 10, 2026**

Based on implementation experience with the v0.1 Architecture Spec, the Rust-native migration, the April 6-7 session (decomposition via synthesis, tracing, 7.2x search optimization, namespace literals, memorization), the April 7 session that added: bytecode VM (22.9x eval speedup), `synthesize` builtin `:library`/`:priorities` support, early depth extension for compositional search, optimization curriculum with meta-optimization, NL curriculum, and chained curriculum execution, the April 7 evening session that added: unified library/tree/namespace loading, TYPE_LIST in the type system, higher-order synthesis via fused map components, 3-word sentence structures, and variable-length sentence tagging via `(string-join (map pos_tag (string-split x " ")) " ")`, and the April 7 late session that added: full 3-domain chained curriculum (55/55 tasks), trace instrumentation infrastructure (`--trace` JSON output), and meta-optimization Stage 3 — multi-task heuristic learning across the full task suite, producing the `priority-plus-type-match` heuristic (26/55 vs 21/55 baseline at budget 5000).

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
- **String indexing:** string-take, string-drop (prefix/suffix by count)
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
- **Alt specs:** Expected values can be SELPH expressions. `(or "quick" "speedy")` in the expected position means either output is acceptable. Parsed as `Value::Alt(Vec<Value>)`, matched via `vals_equal`. Enables multi-answer tasks (e.g., synonyms, non-deterministic outputs) in curricula:
  ```lisp
  (task "synonyms" 3
    ("fast" (or "quick" "speedy"))
    ("big" (or "large" "huge")))
  ```

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
- **Early depth extension:** after each candidate, immediately try composing it with arity-1 and arity-2 components whose return type matches the target. Finds `(is_noun (last_word x))` in 8 candidates instead of exhausting 200K. Expensive chained probes (arity-2 intermediate → arity-2 final) gated to depth 2+ to avoid blowing up the search at depth 1. Key enabler for compositional NL tasks.
- **Domain-based component filtering:** at synthesis start, computes "reachable types" from input/output types and excludes all components whose param/return types are outside that set. Grid components excluded for string tasks, string ops excluded for pure numeric tasks, etc. Combined with lazy grid registration (`default_synth_components_opts`), reduces component count from 120 to 67 for non-grid tasks. **300x speedup** on arithmetic curriculum, **155x candidate reduction** on trivial string tasks.
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

### 2.7 CLI Commands (11 total)
```
selph eval       Evaluate SELPH files or expressions
selph parse      Parse and print AST
selph synth      Synthesize from examples (--tree, --minimize, --maximize, --validate)
selph grow       Run curriculum (--meta, --extract, --validate, --filter, --tree, --trace, --heuristic)
selph bench      Run stochastic benchmark suite
selph generate   Generate a curriculum in .selph format
selph verify     Verify a program against a spec
selph multi-synth Synthesize across namespace trees
selph meta-opt   Optimize heuristics from trace data (--trace, --library, --budget)
selph repl       Interactive REPL
selph help       Show usage
```

### 2.8 Rust Modules (18 source files)
types.rs, parser.rs, eval.rs, synth.rs, hm.rs, library.rs, namespace.rs, induce.rs, divide.rs, verify.rs, abstraction.rs, multitree.rs, stochastic.rs, meta.rs, taskgen.rs, intern.rs, vm.rs, trace.rs

230 tests, all passing.

### 2.9 Validated Results

**Sequence curriculum (13 tasks): 13/13 (100%) — list inputs, 0.7s**

Inputs are native lists `(idx v0 v1 v2 v3)`, not encoded strings. No `seq_helpers.selph` needed — `head`, `nth`, `tail` are builtin list ops. Type filtering excludes all string operations, yielding 28x fewer candidates vs string encoding.

| Task | Candidates | Solution |
|------|-----------|----------|
| const_1 | 2 | `1` |
| const_5 | 6 | `5` |
| identity | 502 | `(head x)` |
| double_idx | 81 | `(multiply (head x) 2)` |
| triple_idx | 33 | `(multiply (head x) 3)` |
| last_plus_1 | 3,425 | `(add (nth x 1) 4)` |
| last_minus_1 | 558 | `(subtract (nth x 1) 4)` |
| **squares** | **249** | **`(multiply (nth x 0) (head x))`** |
| triangular | 1,048 | `(add (nth x 4) (head x))` |
| fibonacci | 1,056 | `(add (nth x 4) (nth x 3))` |
| double_prev | 453 | `(multiply (nth x 1) 16)` |
| **cubes** | **6,116** | **`(multiply (squares x) (nth x 0))`** |
| idx_plus_last | 1,307 | `(add (nth x 4) (nth x 0))` |

Key improvement: `squares` found in 249 candidates (was 5,930 with string encoding). `cubes` composes `squares` — library cascade works with list types.

Previous string-encoded results (for reference): 415K candidates, 27s. List inputs: 14.8K candidates, 0.7s.

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
Validated: sequence (13) → CF (20) → NL (22) = **55/55 (100%)** in ~31 minutes (release build). Library grows from 6 helpers → 20 after sequence → 47 after CF → 69 after NL. Domain isolation via separate `grow` runs (not a single run) is essential: loading all helpers simultaneously causes cross-domain pollution (sequence tasks find wrong D&C solutions using NL macros like `first_word`). Orchestrated via `run_full_chain.sh`.

| Stage | Tasks | Solved | Candidates | Time | Library |
|-------|-------|--------|-----------|------|---------|
| Sequence (list inputs) | 13 | 13/13 | **14.8K** | **0.7s** | 0 → 15 |
| CF | 20 | 20/20 | 802K | 111s | 22 → 42 |
| NL | 22 | 22/22 | 2.8M | 1650s | 47 → 69 |
| **Total** | **55** | **55/55** | **3.7M** | **~29m** | **0 → 69** |

Previous (string-encoded sequence): 4.8M candidates, ~31m. List inputs yield 28x sequence speedup and correct compositional solutions (no D&C memorization for squares/cubes).

RL coefficients transfer across stages: cold=-50.8, warm=30.4 after CF, carried into NL.

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

### 2.17 List-Typed Inputs and Element Type Inference
Curriculum tasks can now use native list inputs: `((4 0 1 4 9) 16)` instead of `("4 0 1 4 9" 16)`. The task parser (`node_to_value`) recursively converts `Node::App` to `Value::List`. The grow command infers element types from list contents: if all elements are numbers, `head`/`nth` return `TYPE_NUM` instead of `TYPE_ANY`.

This eliminates the need for `seq_helpers.selph` — `idx` becomes `(head x)`, `v3` becomes `(nth x 4)`, `last` becomes `(head (list-reverse x))`. Type filtering correctly excludes string operations since inputs are `TYPE_LIST`, not `TYPE_STR`.

**28x candidate reduction:** sequence curriculum drops from 415K candidates (string encoding) to 14.8K (list inputs). `squares` goes from 5,930 to 249 candidates. `cubes` still composes `squares` — library cascade works with list types.

Additional list synthesis components: `nth : list × num → any`, `list-length : list → num`, `list-reverse : list → list`.

**TYPE_ANY fix:** Pool entries with `ret_type = TYPE_ANY` (e.g., from polymorphic builtins like `head : list(a) → a`) now match any parameter type during candidate generation. Previously, `TYPE_ANY` was only treated as a wildcard on the *parameter* side, not the *return* side, causing `head(x)` to be invisible to `multiply`. When HM inference returns `TYPE_ANY` for an unresolved type variable but the component has been patched with a concrete return type, the concrete type is preferred.

### 2.18 Merged Library Output
The `grow` command now serializes all macros from `all_macros` (the in-memory deduplicated set) instead of embedding raw library source strings. This ensures chained curriculum outputs contain all macros from all loaded libraries, not just the first one. D&C solutions are now also registered into `all_macros` so they can be used by later tasks in the same run and persisted correctly.

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
**Current status:** Heuristics as SELPH programs work. Interleaved online learning updates priorities. Domain-specific heuristic gives 23.5x speedup. **Multi-task meta-optimization validated (Stage 3):** `selph meta-opt` evaluates 8 candidate heuristic programs across the full 55-task suite using trace data from chained curriculum runs. Winner: `priority-plus-type-match` — blends learned priority with +50 return-type match bonus. Solves 26/55 vs 21/55 baseline at budget 5000. Notable per-task improvements: `first_of_sentence` 50x faster (1152→23 candidates), `last_of_sentence` 46x faster. **Stage 4 (synthesized heuristic search) implemented:** `selph meta-opt --synthesize` enumerates heuristic programs bottom-up over ctx fields, scoring via behavior-vector rank (no SELPH eval). Two-phase: rank-based enumeration (~2s) + synthesis validation on hard subset (~3s). Awaiting validation on full 55-task trace data.
**What's missing:** The heuristic doesn't yet read task-specific features (example patterns, example count) beyond type tags. Stage 4 needs validation on the full chained curriculum traces to measure actual improvement over Stage 3.

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

Phase 2: Learned heuristics (INTEGRATED — heuristic wired into grow)
  - Meta-1 replaces hardcoded search ordering
  - First loop validated: meta_count_char_boost synthesized a priority
    boost by running synthesis inside the fitness function (0.7s)
  - The synthesize builtin accepts :priorities and :library for
    SELPH-level control of search strategy
  - Multi-task meta-optimization (Stage 3) validated on 55-task suite:
    priority-plus-type-match heuristic solves 26/55 vs 21/55 baseline
    (1.1x speedup, 50x on first_of_sentence)
  - `--heuristic` flag integrated: loads a SELPH lambda, applies per-task
    via meta::apply_heuristic() with full task context. Sequence curriculum:
    13,721→11,263 cand (~2x wall-clock speedup), identity 42x faster.
  - Training data: trace JSON from 3-domain chained curriculum (55 tasks)
  - Validation: learned heuristic solves 5 more tasks than baseline
  - Stage 4 implemented: `meta-opt --synthesize` enumerates heuristic
    programs bottom-up. Awaiting validation on full trace data.
  - Remaining: validate Stage 4 on 55-task traces, synthesize deeper heuristics

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

### 8.10 ~~Full 3-domain chained curriculum~~ ✓
Sequence (13) → CF (20) → NL (22) = 55/55 (100%) via `run_full_chain.sh`. Three separate `grow` runs with domain-scoped helpers, each feeding its output library to the next. Total: 4.8M candidates, ~31 minutes (release build). Library grows from 6 → 69 macros. Key insight: domain isolation via separate runs is essential — single-run with all helpers causes cross-domain search pollution.

### 8.11 ~~Trace instrumentation~~ ✓
`--trace trace.json` flag on `grow` command. `trace.rs` captures per-task: candidates explored, wall time, solving strategy (Flat/BD/Induction/D&C/Memo), task features (input type, output type, num examples, library size), components used, components available. Manual JSON serialization (no serde dependency). Three trace files generated per chained run.

### 8.12 ~~Meta-optimization Stage 3: multi-task heuristic learning~~ ✓
`selph meta-opt` command reads trace JSON + curriculum files, rebuilds training tasks, evaluates 8 hand-crafted heuristic templates by re-running synthesis across the full 55-task suite. Winner: `priority-plus-type-match` — `(lambda (ctx) (add (ns-get ctx "priority") (if (= (ns-get ctx "ret-type") (ns-get ctx "output-type")) 50.0 0.0)))`. Solves 26/55 vs 21/55 baseline at budget 5000 (1.1x candidate reduction, up to 50x on individual tasks). Saved to `chain_output/heuristic_priority_plus_type_match.selph`.

### 8.13 ~~Fused reduce components~~ ✓
Fused `reduce_<fn>` components for binary builtins (`add`, `subtract`, `multiply`, `min`, `max`, `concat`) and binary macros. Same pattern as fused map: registered as arity-1 `list → ret_type`, materialized as `(reduce fn arg)`, handled in early depth extension. Validated: `remove_c` independently discovered `(reduce concat (string-split x "c"))` — the synthesizer used reduce without being taught explicitly.

### 8.14 ~~Context-sensitive language curriculum~~ ✓ (a^n b^n c^n)
**a^n b^n c^n solved** via boolean decomposition of promoted CF macros. 14-task curriculum: count_c → equal_bc → remove_c → no_c_before_b → equal_abc → abc_order → anbncn → string halving → copy/reversal → reduce tests. 14/14 solved (100%) standalone at budget 50K. Key: the CS language didn't need `reduce` — string-replace + ordering compositions sufficed.

### 8.15 ~~New builtins and synth components~~ ✓
- **`string-take`** (str, num) → str: first N characters. Arity-2 alternative to arity-3 `string-slice`.
- **`string-drop`** (str, num) → str: everything after first N characters.
- **`string-chars`**, **`divide`**, **`floor`**, **`string-slice`** registered as synth components (previously builtins only).
- **`reduce` empty-list fix**: returns error instead of panicking on `l[0]` when list is empty.

### 8.16 ~~Type-override bug fix~~ ✓
Removed blanket `param_types` override in `cmd_synth`/`cmd_curriculum` that rewrote all macro types to match input type. This destroyed inferred types for cross-type macros (e.g., `halve: num→num` became `str→str` on string-input tasks), causing probe-and-filter to incorrectly exclude them. Also added type-aware probe-and-filter: macros whose param type doesn't match the input type are skipped (not probed with wrong input). Together, these fixes enable cross-type compositions like `(halve (string-length x))`.

### 8.17 ~~Arithmetic and string slicing curricula~~ ✓
- **`arithmetic_tasks.selph`** (2 tasks): `halve` → `(floor (divide x 2))` in 169K candidates (0.19s), `third` → `(floor (divide x 3))` in 3.9K (0.68s, warmed by halve). Total: 0.9s (was 269.6s before domain filtering fix, see §8.19).
- **`string_slice_tasks.selph`** (7 tasks): prefix/suffix extraction via `string-take`/`string-drop`, composes with promoted `halve` for `half_len` → `(halve (string-length x))`.
- **`run_full_chain.sh`** extended to 6 stages: seq → CF → NL → arithmetic → string slicing → CS.

### 8.18 ~~Meta-optimization Stage 4: synthesized heuristic search~~ ✓
`selph meta-opt --synthesize` enumerates heuristic program bodies bottom-up instead of choosing from 8 hand-crafted templates. Two-phase approach:

- **Phase A (rank-based, ~2s):** Bottom-up enumeration over 7 ctx fields (`priority`, `arity`, `ret-type`, `first-param-type`, `input-type`, `output-type`, `num-examples`) + 8 numeric constants, composed with arithmetic (`add`, `subtract`, `multiply`, `min`, `max`), comparison (`=`, `<`, `>`), unary (`abs`, `negate`), and if-expressions. Behavior-hash dedup eliminates observationally equivalent candidates. Rank scoring computed directly from behavior vectors (no SELPH eval needed — microsecond cost per candidate).
- **Phase B (synthesis validation, ~3s):** Top-K candidates from Phase A validated by running actual synthesis on the 15 hardest tasks.

Key design: the "flattening trick" — ctx namespace fields become depth-0 atoms during enumeration, then get wrapped back into `(ns-get ctx "field")` via `wrap_as_lambda()` to produce valid SELPH. This transforms heuristic synthesis from "programs over namespaces" into "arithmetic expressions over 7 variables" — exactly what the enumerator does well.

Infrastructure: `enumerate_heuristic_candidates()` and `validate_heuristic_candidates()` in `meta.rs`, `run_meta_eval_with_snapshots()` in `main.rs` for PoolSnapshot capture. 230 tests (8 new), all passing.

### 8.19 ~~Domain-based component filtering and search regression fix~~ ✓ (April 8, 2026)

The ARC-AGI grid infrastructure (commit 91b1655) added ~50 grid components to `default_synth_components`, inflating the component count from 46 to 120 for ALL synthesis runs. Combined with the early depth extension's speculative probing (which multiplies combinatorially with component count), this caused severe regressions on non-grid tasks:

| Benchmark | Before (46 comp) | Broken (120 comp) | Fixed (67 comp) |
|-----------|-------------------|--------------------|------------------|
| string-upper | 22 cand, 0.004s | 3,412 cand, 0.473s | 159 cand, 0.012s |
| arithmetic halve | — | 275K cand, 2.5s (overfitted) | 169K cand, 0.19s (correct) |
| arithmetic third | — | 149s (overfitted w/ if-exprs) | 0.68s (correct `floor(divide x 3)`) |

**Three fixes:**

1. **Domain-based component filtering** (`synthesize_full`): Computes "reachable types" from input/output types. A component is kept only if all its param types and return type are in the reachable set. For string→string tasks, grid components are unreachable. Also always includes BOOL (for if-conditions) and NUM (common intermediate). STR and LIST co-include each other (string-split/join bridge them).

2. **Chained probe depth gating**: The early depth extension's expensive chained probes (arity-1 → intermediate type → arity-2 final) now only run at depth 2+. Cheap arity-1 and arity-2 probes still run at all depths (needed for finding `(add (add x x) 1)` at depth 1).

3. **Lazy grid registration**: `default_synth_components_opts(macros, include_grid)` only registers grid components when the `include_grid` flag is true. Call sites in `cmd_synth`, `cmd_curriculum`, and `cmd_arc` pass `input_is_grid`. The no-arg `default_synth_components()` wrapper defaults to `false`.

**Net effect:** Arithmetic curriculum 269.6s → 0.9s (**300x**). CF curriculum 20/20 in 31.7s. "third" now finds the correct generalization instead of memorizing with if-expressions.

---

## 9. Immediate Next Steps

### ~~9.1 Chained curriculum execution~~ ✓ (extended to 3-domain)
Validated: sequence (13) → CF (20) → NL (22) = 55/55 (100%). Library grows 6 → 69 macros. Domain isolation via separate `grow` runs prevents cross-domain pollution. RL coefficients transfer across stages.

### ~~9.2 Early depth extension~~ ✓
Implemented as targeted composition probe: after each depth-1 candidate, immediately try composing it with arity-1 components whose return type matches the target. Finds `(is_noun (last_word x))` in 8 candidates instead of exhausting 200K. Combined with bytecode VM, the CF curriculum runs in 15.7s (was 293s baseline).

### 9.4 Curriculum-driven RL coefficient specialization
Currently one (cold, warm) pair for all tasks. Different task types may benefit from different coefficients. Per-domain coefficients could be stored as namespace metadata and selected based on inferred task type. The `:priorities` field in the `synthesize` builtin partially addresses this.

### ~~9.5 Meta-2: Learn decomposition as SELPH programs~~ ✓
The `synthesize` builtin accepts `:library` — a SELPH program can restrict the component set and call synthesis on sub-problems. Decomposition IS parameterized synthesis, expressible end-to-end from SELPH.

### 9.16 Higher-order template decomposition ✓ (April 8, 2026)

`decompose.rs` adds 4 templates that derive sub-specs from outer specs and fill function-typed holes via recursive `synthesize()` calls. Pipeline: Flat → BD → IN → **HO** → DC → Memo.

**Templates:**
1. **list-map:** `(lambda (x) (map HOLE x))` — element-wise list transformation
2. **split-map-join:** `(lambda (x) (string-join (map HOLE (string-split x SEP)) SEP))` — delimiter discovery + per-part transformation
3. **char-map-join:** `(lambda (x) (string-join (map HOLE (string-chars x)) ""))` — per-character transformation
4. **list-filter:** `(lambda (x) (filter HOLE x))` — element predicate synthesis with boolean labeling

**How it works:** Each template (1) checks applicability by type/structure, (2) derives a sub-spec by pairing decomposed input/output elements, (3) deduplicates sub-spec pairs (aborting on conflicts), (4) calls `synth::synthesize()` recursively with budget/4, (5) composes the sub-solution into the template AST, (6) verifies against all original examples.

**Results on test curriculum (6 tasks, budget 200K):**

| Task | Strategy | Candidates | Solution |
|------|----------|-----------|----------|
| list_double | HO (list-map) | 16 | `(map (lambda (x) (add x x)) x)` |
| list_negate | HO (list-map) | 83 | `(map (lambda (x) (subtract x (add x x))) x)` |
| upper_words | Flat | 117K | `(string-upper (string-drop x 0))` |
| reverse_words | **HO (split-map-join)** | **818** | `(string-join (map (lambda (x) (string-reverse (string-drop x 0))) (string-split x " ")) " ")` |
| upper_chars | Flat | 1,346 | `(string-upper (string-drop x 0))` |
| keep_positive | HO (list-filter) | 229 | `(filter (lambda (x) (< (negate (add x x)) x)) x)` |

**Key result:** `reverse_words` is the proof point — `(string-join (map string-reverse (string-split x " ")) " ")` is a genuinely compositional solution unreachable by flat enumeration. The system split by space, synthesized `string-reverse` for each word, and rejoined.

**Difference from fused components (9.14):** Fused `map_<macro>` components only work with pre-existing macros. Template decomposition discovers *new* element-level functions via sub-synthesis. Fused is O(1) (no search), templates are O(sub-budget) but strictly more powerful.

**Design insight:** Decomposition should ultimately be just another candidate in the search space, not a fallback. The function-typed hole in `(map [HOLE] list)` is a sub-synthesis call embedded in candidate generation. The strategy selector that decides *when* to decompose should itself be a learnable Selph program — the meta-language endgame where `synthesize` is a component the solver can call recursively on derived sub-specs.

### 9.17 Priority scoring fix: average instead of sum ✓ (April 8, 2026)

PendingDesc scores were computed as `comp.priority + sum(arg_priorities)`. Since `x` has priority 100:
- Arity-1 `string-upper(x)`: score = 0 + 100 = **100**
- Arity-2 `concat(x, x)`: score = 0 + 100 + 100 = **200**
- Arity-3 `string-replace(x, x, x)`: score = 0 + 100 + 100 + 100 = **300**

Higher arities always scored higher, so the simplest arity-1 solutions were tested **last**. The budget was consumed by arity-2/3 candidates before `string-upper(x)` was ever reached. This caused `string-upper(x)`, `string-lower(x)`, and other trivial arity-1 compositions to go unfound, falling to memorization even with 200K budget.

**Fix:** Changed to average of arg priorities: `comp.priority + avg(arg_priorities)`. Now all compositions involving `x` score 100 regardless of arity. Ties broken by generation order (arity-1 first, then arity-2, then arity-3).

**Impact on Stage 0 instruction ops (4 tasks, no helpers):**
- Before: 144s, 593K candidates. `upper` and `lower` memorized. `reverse` found at 92K via complex roundabout.
- After: **1.5s, 8K candidates**. All 4 found cleanly: `(string-upper x)`, `(string-lower x)`, `(string-join (reverse (string-chars x)) "")`, `(length (string-chars x))`.

**Requires validation:** The scoring change affects ALL synthesis. The full 55-task chained curriculum needs re-running to verify no regressions. CF tasks like `equal_ab` relied on arity-2 compositions being prioritized — the average scoring might slow them down.

### 9.18 Instruction-following curriculum and `dispatch` builtin ✓ (April 8, 2026)

**Goal:** Bridge from example-based synthesis to instruction-conditioned execution. In the current system, specs are opaque — `("aabbcc", true)` gives the synthesizer no signal about HOW to solve the problem. In NL instructions, the intent is in the input: `"reverse hello"` names the operation.

**`dispatch` builtin:** Special form in `eval_inner` (not a regular builtin) that looks up a macro by name (a string) and applies it. Implemented as a special form because it needs the dynamic environment with promoted macros — regular builtins only get `fn(&[Value])` without env access. Intentionally NOT in `BUILTIN_NAMES` so the VM compiler fails on dispatch candidates, triggering the tree-walker fallback where the special form has env access.

```lisp
(dispatch "reverse" "hello")  ; → looks up "reverse" macro, applies to "hello" → "olleh"
```

Registered as a synthesis component: arity 2, `(STR, ANY) → ANY`, priority 15.

**Curriculum design (3 files):**

1. `instruction_ops.selph` — Pure operations WITHOUT NL helpers loaded. Promotes clean wrappers: `reverse`, `upper`, `lower`, `len`. Run as separate grow pass to prevent word extractors from contaminating the operations.

2. `instruction_dispatch.selph` — Dispatch tasks WITH NL helpers + ops library. The synthesizer discovers `(dispatch (first_word x) (last_word x))` for 2-word instructions and `(dispatch (second_word x) (last_word x))` for 3-word NL instructions.

3. `instruction_tasks.selph` — Combined single-file version (for reference).

**Results:**
- `dispatch` found `(dispatch (first_word x) (second_word x))` in **1574 candidates** with clean macros.
- HO decomposition found `(string-join (map reverse (string-split x " ")) " ")` for `reverse_words`.
- `upper_words` found `(upper x)` — trivial application of the promoted macro.

**Remaining issue:** Complex promoted macros (e.g., `reverse` promoted as `(string-join (reverse (string-chars s)) "")`) fail when called via `dispatch` because the dispatch env doesn't propagate all necessary builtins for nested calls. The builtin `reverse` (list reverse) is needed inside the macro body but shadows/conflicts in the dispatch context. Clean macros like `(string-reverse s)` work perfectly. Fix: either ensure simple promotions (may require `string-reverse(x)` to be found first — see §9.17) or propagate the full macro env in dispatch.

**Design insight: the library IS the dispatch table.** Each promoted macro is an entry. `dispatch` is the meta-primitive that connects string-valued operation names to the learned function library. The system doesn't learn HOW dispatch works — it learns WHAT to dispatch on (argument extraction from instructions) and WHICH operations to learn (the Stage 0 curriculum). This parallels `map`: the system doesn't learn how `map` works, it learns what to map.

**Future directions:**
- **Tokenization curriculum:** Teach "command" as a part of speech. The NL curriculum has noun/verb/adj/art but not command/imperative. A command POS type would let the system classify instruction keywords.
- **String equality in synthesis:** Currently `=` is num-only. Adding `(str, str) → bool` equality would enable `(= (first_word x) "reverse")` for dispatch conditions, as an alternative to `dispatch`.
- **Env propagation fix:** Make dispatch's env include all promoted macros so complex compositions work.
- **Word-level auto-constants:** Extract words (not just chars) from examples. Would enable `(= (first_word x) "reverse")` without needing helper-provided constants.

### 9.6 Deeper curriculum: context-sensitive languages (PARTIALLY DONE)
a^n b^n c^n **solved** without needing reduce — the synthesizer found clever string-replace + ordering compositions, composing promoted CF macros via boolean decomposition. Copy language (ww) and reversal (ww^R) require string halving, which depends on the scaffolding chain: arithmetic (halve) → string slicing (string-take/drop) → dynamic halving (first_half/second_half). `first_half` remains unsolved compositionally — the arity-2 early depth extension probe doesn't reach `(string-take x (half_len x))` within budget. See 9.15 for details.

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
- **Use native types, not string encoding:** Sequence inputs should be lists `(4 0 1 4 9)`, not strings `"4 0 1 4 9"`. With list inputs, the type system sees `list → number` and excludes all string operations automatically. This yields 28x candidate reduction for the sequence curriculum. String encoding forces parsing helpers (`idx`, `v0`) and allows string ops to pollute the search space. Lisp got this right: data should carry its type.
- **Domain isolation is essential for chained curricula:** Loading helpers from all domains in a single `grow` run causes catastrophic search pollution. Sequence tasks find wrong D&C solutions using NL helpers (`first_word`, `string-starts-with`). The fix is chaining separate `grow` runs, where each run only loads domain-relevant helpers plus the accumulated library from prior stages. This is orchestrated via `run_full_chain.sh`.
- **Meta-optimization works at modest scale:** 8 hand-crafted heuristic templates, evaluated on 55 tasks at budget 5000 each, found a heuristic that solves 5 additional tasks. The key signal is return-type matching: boosting components whose output type matches the target. This is a learned version of what type reachability filtering does statically — but applied as a soft priority rather than a hard filter.
- **Trace data enables offline heuristic search:** By recording per-task features, candidate counts, and component usage during curriculum runs, heuristic optimization can be done offline — re-running synthesis with different priority orderings without regenerating traces. The `meta-opt` command demonstrates this: it reads trace JSON, rebuilds training tasks from curriculum files, and evaluates heuristics by actual synthesis, not proxy metrics.
- **Cross-type composition requires correct type inference:** Macros like `halve` (num→num) are used as intermediate compositions on string-input tasks via `(halve (string-length x))`. The blanket type override (forcing all macros to match input type) destroys this — `infer_macro_types` must be trusted. The probe-and-filter must also respect types: don't test `halve("ab")` when halve is `num→num`.
- **Arity-3 is effectively unreachable without helpers:** `string-slice` (str, num, num) → str was never found by the synthesizer despite being registered. The combinatorial space of three-argument compositions is too large. The fix is decomposing arity-3 into arity-2 wrappers: `string-take` (str, num) → str and `string-drop` (str, num) → str. This is the same decomposition principle as fused map/reduce — reduce the arity the synthesizer must handle.
- **The synthesizer finds creative workarounds:** When `string-take` wasn't available, the system discovered `(string-join (tail (string-chars x)) "")` for `drop_first` and `(reduce concat (string-split x "c"))` for `remove_c`. These are correct but non-obvious compositions — evidence that the search explores a richer space than hand-designed solutions would suggest.
- **Scaffolding must match the composition chain:** `first_half` requires `(string-take x (half_len x))` which chains: arithmetic (floor, divide) → halve macro → half_len macro → string-take composition. Each step must be a separate curriculum stage so the promoted macro has correct types and high priority. Missing any link in the chain causes memorization fallback.
- **Decomposition is just another candidate:** The distinction between "strategy selection" and "search" is artificial. `(string-join (map f (string-split x " ")) " ")` is a program in the search space — the only reason the flat enumerator can't find it is that `f` is a function-typed argument requiring sub-synthesis. Template decomposition (§9.16) fills this gap: when the enumerator encounters a higher-order component like `map`, it derives a sub-spec and calls `synthesize()` recursively. The long-term design: the strategy selector is itself a Selph program, and `synthesize` is a component the solver can invoke on derived sub-specs.
- **Integer decomposition as a future meta-learning target:** Analogous to boolean decomposition (BD) for bool targets, an integer decomposition (ID) would try arithmetic compositions of numeric pool entries when flat synthesis fails on numeric targets. O(entries²) cost. Would catch `(floor (divide x 2))` directly instead of requiring scaffolding. Candidate for Meta-2 curriculum.
- **Priority scoring must normalize by arity:** Sum-based scoring (`comp.priority + sum(arg_priorities)`) creates a systematic bias toward higher-arity compositions. Arity-2 with (x, x) scores 200 vs arity-1 with x at 100. The budget is consumed by the ~900 arity-2 candidates before ANY arity-1 candidate is tested. Average-based scoring (`comp.priority + avg(arg_priorities)`) normalizes this. All compositions involving `x` score 100 regardless of arity. Arity-1 is tested first due to generation order.
- **Promoted macros must be pure operations for dispatch:** If NL helpers (first_word, last_word) are loaded during operation learning, the promoted macros bake in word extraction: `(string-reverse (last_word s))` instead of `(string-reverse s)`. These composite macros fail when `dispatch` calls them with single-word arguments because the extraction is redundant but harmless — however, they fail when the inner macro body references other macros not available in the dispatch env. **Domain isolation extends to the operation level:** learn pure operations in one run (no helpers), dispatch in another (with helpers + operations).
- **dispatch as a special form, not a builtin:** The `dispatch` primitive needs the dynamic environment to find promoted macros. Regular builtins (`fn(&[Value])`) don't have env access. Implementing dispatch as a special form in `eval_inner` gives it full env access. It must NOT be in `BUILTIN_NAMES` so the VM compiler fails and triggers the tree-walker fallback. This is the same pattern as `ns` — constructs that need the environment can't be VM-compiled.
- **The library IS the dispatch table:** In SELPH, promoted macros live in the environment as named functions. `dispatch("reverse", arg)` is just `env_lookup("reverse")` + `apply`. No separate dispatch table needed — the library file IS the model, and the model IS the dispatch table. This is homoiconicity at the system level: the trained model (library) serves as both the learned knowledge base and the runtime dispatch mechanism.

### 9.15 Context-sensitive language status (April 7, 2026)

**Solved:**
- a^n b^n c^n via boolean decomposition composing CF-promoted macros
- `reduce` as a synthesis component — `remove_c` found `(reduce concat (string-split x "c"))`
- `half_len` → `(halve (string-length x))` via cross-type composition after type-override fix

**Remaining:**
- `first_half` — `(string-take x (half_len x))` not found within budget. The arity-2 early depth extension probe generates the candidate but budget exhausts before reaching it. Needs either higher budget, priority boosting for `string-take`, or integer decomposition.
- `second_half` — found a roundabout `string-drop` composition but not the clean `(string-drop x (half_len x))`
- Copy language (ww) and reversal (ww^R) — blocked on first_half/second_half
- `rejoin` trivially solves as `x` (identity) since reducing concat over chars of a string gives the string back

### 9.13 Next priorities (updated April 7, 2026)

With 55-task chained curriculum validated and meta-optimization Stage 3 complete, the priority shifts toward integrating the learned heuristic and extending the system's capabilities:

**Completed (April 7 evening):**

1. ~~**Full chained curriculum: sequence → CF → NL.**~~ ✓ Chain all three domains via `run_full_chain.sh`: 13 + 20 + 22 = 55/55 solved (100%). Library grows from 6 → 20 → 47 → 69 macros across stages. Domain isolation via separate grow runs prevents cross-domain pollution. Trace output via `--trace` flag captures per-task candidates, strategy, timing, features, and components used.

   Results: Seq 27s (415K cand), CF 135s (907K cand), NL 1680s (3.5M cand). Total: ~31 minutes, 4.8M candidates.

   Key insight: Loading all three helper libraries in a single run causes massive search space pollution — sequence tasks find wrong D&C solutions using NL helpers like `first_word`. Chaining separate runs with domain-specific helpers is essential.

2. ~~**Meta-optimization Stage 3: multi-task heuristic learning.**~~ ✓ `selph meta-opt` command reads trace JSON from chained runs, rebuilds training tasks, and evaluates 8 candidate heuristics by re-running synthesis on the full 55-task suite. Winner: `priority-plus-type-match` — blends learned priority with +50 return-type match bonus.

   Results at budget 5000: baseline 21/55 solved → **26/55 solved** (+5 tasks, 1.1x speedup). Notable: `first_of_sentence` 50x faster (1152→23 cand), `last_of_sentence` 46x faster. Heuristic saved to `chain_output/heuristic_priority_plus_type_match.selph`.

   Infrastructure added: `trace.rs` (JSON serialization), `--trace` flag on `grow`, `meta-opt` CLI command with trace JSON parser. All without external dependencies (no serde).

**Completed (April 7 night):**

1. ~~**Integrate winning heuristic into grow command.**~~ ✓ `--heuristic <file>` flag loads a SELPH lambda and applies it per task via `meta::apply_heuristic()`. Each component is scored by the heuristic with full task context (input/output types, example count) and component metadata (name, arity, return type, priority). Replaces static priority ordering with learned, task-dependent ordering.

   Results on sequence curriculum (budget 5000): baseline 13,721 cand / 6.1s → **11,263 cand / 3.1s** (~2x speedup). Notable: `identity` 42x faster (502→12 cand), `last_plus_1` ~10x faster (3425→350 cand). The heuristic composes with existing learned-priority and RL-coefficient systems — heuristic scoring incorporates the accumulated priority from prior solves.

**Highest-value next steps:**

1. **Unblock `first_half` / copy language.** `(string-take x (half_len x))` is the simplest correct solution but the arity-2 early depth extension doesn't reach it within budget. Options: (a) integer decomposition fallback (try arithmetic compositions of numeric pool entries when flat synthesis fails — analogous to BD), (b) priority boosting for `string-take`/`string-drop` when target is string and numeric intermediates exist, (c) deeper scaffolding tasks.

2. **Types as a SELPH program.** The type vocabulary (NUM, STR, BOOL, LIST) is currently hardcoded. Making type inference a curriculum target lets the system learn to expand its own type vocabulary. A type inference stage teaches "if `string-split` produces it, and `head` consumes it, they share a type." See §9.24.3 for the concrete plan: types as namespaces with predicates, registered in a `__types__` namespace, with the synthesizer caching Sym handles for hot-path checks.

3. ~~**Deeper meta-optimization.**~~ ✓ Stage 4 implemented: `meta-opt --synthesize` enumerates heuristic programs bottom-up over ctx fields, scored via behavior-vector rank. Two-phase: Phase A (cheap rank enumeration, ~2s) + Phase B (synthesis validation on hard subset, ~3s). Needs validation on the full 55-task traces.

**Completed this session:**
- ~~`reduce`/`fold` as synthesis components~~ ✓ — fused reduce for builtins and macros
- ~~Context-sensitive languages~~ ✓ — a^n b^n c^n solved, 6-stage chain pipeline
- ~~`string-take`/`string-drop` builtins~~ ✓ — arity-2 alternatives to arity-3 `string-slice`
- ~~Type-override bug fix~~ ✓ — enables cross-type macro compositions
- ~~Arithmetic and string slicing curricula~~ ✓

**Completed (April 8, 2026):**
- ~~Higher-order template decomposition~~ ✓ — `decompose.rs` with list-map, split-map-join, char-map-join, list-filter. Fills function holes via recursive sub-synthesis. Key result: `reverse_words` solved compositionally (818 cand) where flat synthesis can't reach it. See §9.16.
- ~~Priority scoring fix~~ ✓ — PendingDesc scores used sum of arg priorities, biasing toward higher arities (arity-2 scored 200 vs arity-1 at 100 when both use `x`). Changed to average: `comp.priority + avg(arg_priorities)`. Stage 0 ops: 144s/92K → 1.5s/8K cand. `string-upper(x)` now found at 4K cand instead of memorizing. See §9.17.
- ~~`dispatch` builtin~~ ✓ — Special form in eval that looks up a macro by name (string) and applies it. Registered as arity-2 synth component (STR, ANY) → ANY. Enables instruction-following: `(dispatch (first_word x) (last_word x))` found in 1574 cand with clean macros. See §9.18.
- ~~Instruction-following curriculum~~ ✓ (partial) — Three files: `instruction_ops.selph` (pure operations), `instruction_dispatch.selph` (dispatch tasks), `instruction_tasks.selph` (combined). Two-stage pipeline validated. Dispatch works with simple macros but complex promoted macros (e.g., `string-join(reverse(string-chars x), "")`) fail in dispatch env due to incomplete macro propagation. See §9.18.

**Completed (April 8, 2026 — meta-learning session):**

- ~~Meta-learning unification~~ ✓ — Search builtins for SELPH-programmable search functions:
  - `test-spec` builtin: `(test-spec candidate spec) → match-fraction` — test any candidate function against a spec
  - `memorize` builtin: `(memorize spec) → namespace` — returns data as a namespace (not a lambda), composable in the library tree
  - `synthesize` extended: accepts library as namespace tree, source string, or list. Data namespaces automatically become depth-0 synthesis atoms.
  - `ns-get` and `ns-get-or` registered as synthesis components — enables `(ns-get vocab x)` compositions
  - Namespace values emitted as Symbol nodes for env resolution (was String literal — the root cause of data-driven synthesis failure)
  - Libraries as first-class namespace trees with subtree selection: `(ns-get library "math")` selects a sub-library

- ~~Domain isolation in meta-opt~~ ✓ — `all_components_available` recorded in traces, per-task component filtering during meta-opt evaluation. Seq tasks see 67 components instead of 171.

- ~~Input awareness~~ ✓ — 5 new TaskContext fields: `avg-input-len`, `has-spaces`, `max-num-value`, `output-is-bool`, `num-distinct-outputs`. CTX_FIELDS expanded from 7 to 13 (including `usage-count`).

- ~~Online learning~~ ✓ — `comp_priority_boost` vector updated between depths from `comp_best_match`. New `comp_warm_bonus` RL coefficient (default 15.0).

- ~~`usage-count` in heuristic context~~ ✓ — SynthComponent tracks how many prior tasks used it. Heuristic can learn `(add priority (multiply usage-count K))` instead of hard-coded +50. Priority accumulation skipped when heuristic is loaded.

- ~~`--skip-stage3` flag~~ ✓ — Skip hand-crafted heuristic evaluation, go straight to Stage 4. Baseline stats from traces (no synthesis needed).

**Key result: search function as a SELPH program.** A working search function in SELPH that orchestrates synthesis + memorization:
```lisp
(lambda (spec library)
  (let ((r (synthesize (ns ("spec" spec) ("library" library) ("max-candidates" 5000)))))
    (if (ns-get r "found") r
      (let ((data (memorize spec))
            (enriched (ns-put library "data" data))
            (r2 (synthesize (ns ("spec" spec) ("library" enriched) ("max-candidates" 10000)))))
        r2))))
```
This composes flat synthesis with data-driven memorization — `memorize` creates a data namespace, adds it to the library, and synthesis finds `(ns-get data x)` in 4331 candidates.

### 9.19 Search function curriculum (next steps)

The search function should be LEARNED, not hand-written. The curriculum teaches it progressively:

**Stage 0: Library traversal.** ✓ Given a library tree with multiple subtrees and a task, learn which subtree to pass to `synthesize`. Implemented as pure SELPH script (`examples/stage0_traversal.selph`): 3 subtrees (math/string/word), 15 tasks, 60 profiling runs. Feature encoding: `outtype-startchar-hasspace` (e.g., `number-digit-space` → math). Selector synthesized via `memorize` + `ns-get lookup` composition (8615 candidates). Results: 15/15 solved, overall 1.5x reduction, **word domain 8.3x** (up to 37x per task). Key insight: domain type filtering already handles math/string separation — the learned selector adds value for word-level helpers where type filtering can't distinguish. `defmacro` requires single-body (`do` for multi-expression).

**Stage 1: Data-driven search.** ✓ Tasks where flat synthesis fails but memorize + re-synthesize works. Implemented as `examples/stage1_data_driven.selph`: 12 tasks (5 lookup, 5 computable, 2 hybrid). Smart search = try flat → if fails → memorize → re-synthesize with data as tree → `(ns-get data x)`. Results: flat 5/12 solved, smart 12/12 solved (3.5x fewer total candidates). Category C (lookup+composition) solved via direct memorization of final outputs — `(ns-get data x)` suffices without needing `(f (ns-get data x))` compositions.

**Stage 2: Boolean decomposition as SELPH.** ✓ Implemented as `examples/stage2_bool_decompose.selph`. Enumerates library predicates via `dispatch`, computes truth vectors, tries all pairwise compositions (and/or/not). Uses `test-spec` for verification. Results: flat 6/7, BD 7/7 — solved `(and (is_long x) (not (starts_a x)))` which flat couldn't. Key bugs found: `=` doesn't handle booleans (workaround: `bool-eq`), `eval-source` uses fresh env (predicates must be in-scope or string-prepended).

**Stage 3: Divide and conquer as SELPH.** ✓ Implemented as `examples/stage3_divide_conquer.selph`. Groups examples by output value (manual reduce, no `group-by` builtin needed), synthesizes separator conditions, composes nested if-expressions via string concatenation + `eval-source`. Results: flat 0/4, D&C 4/4 — solved 2-way and 3-way classification tasks including `(if (is_short x) "short" (if (not (is_long x)) "med" "long"))`.

**Stage 4: Budget/strategy allocation.** ✓ Implemented as `examples/stage4_5_unified_search.selph`. Unified search function composes flat, BD, D&C, and memo strategies in a cascade. 16/16 tasks solved across all domains: flat handles 7 (computable patterns), D&C handles 6 (multi-output classification), memo handles 3 (arbitrary lookups). Total: 73K candidates.

**Stage 5: Self-optimization.** ✓ Compared 3 strategy orderings (flat-first, BD-first, memo-first) on the 16-task suite. All solve 16/16; memo-first wins with 71,446 candidates (vs 73,458 flat-first). The optimal search function is a SELPH program that routes by task features: bool output → BD, multi-valued output → D&C, else → memo→flat cascade.

**Prerequisites for the curriculum:**
- ~~`bool-decompose` expressible in SELPH (needs library introspection)~~ ✓ done via `dispatch` + truth vector enumeration
- ~~`group-by` or `partition` builtin for D&C~~ ✓ not needed — manual `reduce` + `filter` suffices
- Traces as SELPH-readable data (for self-optimization)

**Lower priority (infrastructure):**
- 9.8 vector/tensor builtins — prerequisite for linear models
- VM compilation of `ns` in the candidate path

### 9.20 Synthesis probe pipeline fixes (April 9, 2026) ✓

Four bugs in synth.rs prevented cross-type macro compositions like `(string-take x (half_len x))`:

1. **Probe dedup contamination:** arity-2 probe results inserted behavior hashes into `seen`, deduping later main candidates and skipping their probes. Fix: deduped main candidates now fall through to probes instead of `continue`.

2. **Cross-macro type inference:** `infer_macro_types` used a fresh env without other macros, so `half_len → halve(string-length(s))` couldn't resolve `halve` and got typed `NUM→NUM` instead of `STR→NUM`. Fix: pass full macro list to type inference env.

3. **Type-skipped candidates bypass probes:** `EvalResult::Skipped → continue` prevented probes on non-target-type candidates. Fix: fall through to probes when ret_type is in useful_types.

4. **VM miscompiles macro chains:** VM compiled macro-calling-macro compositions but executed wrong results without error. Fix: force tree-walker fallback for probe compositions from macro-based type-skipped candidates.

**Results:** halve=`(floor (divide x 2))`, half_len=`(halve (string-length x))`, first_half=`(string-take x (half_len x))`, second_half=`(string-drop x (half_len x))`. Full 6-stage chain: 74/76 solved (cubes regression from dedup timing change). Copy language (ww) and reversal (ww^R) now unblocked — `ww` found as `(string-ends-with s (first_half s))`.

### 9.21 Recursive decomposition: synthesis as top-down prediction (April 9, 2026)

**Key insight:** All search strategies (flat, BD, HO, D&C, induction, memo) are instances of a single recursive step:

```
synthesize(spec) →
  1. SELECT f (the outermost function)
  2. DERIVE subspecs by "inverting" f on the examples
  3. RECURSIVELY synthesize each subspec
```

The strategies differ only in which `f` they select:

| Strategy | f selected | Subspec derivation |
|----------|-----------|-------------------|
| Flat | single composition | None (leaf) |
| BD | `and`/`or`/`not` | Find component predicates |
| HO | `map`/`reduce`/`filter` | Element function: input_i → output_i |
| D&C | `if(cond, then, else)` | Partition examples, find separator |
| Induction | any bridge `f` | Compute f(input), derive input → f(input) |
| Memo | lookup table | None (base case) |

Step 2 (subspec derivation) is **deterministic** once you know f. The hard problem is step 1: predicting f. And step 1 is itself a synthesis problem — given (spec_features → f_choice) training pairs from the library, learn the mapping.

**Training data from the current library (74 solved programs):**

Analysis of outermost functions across the 6-stage chain:

- **Arithmetic composition** (21%): `add`, `subtract`, `multiply`, `floor` — outermost is a numeric op. Spec features: num input or num output.
- **String predicate** (18%): `string-ends-with`, `string-starts-with`, `count-char` — outermost checks a string property. Spec features: bool output, string input.
- **Boolean composition** (10%): `and`, `or`, `not` — BD pattern. Spec features: bool output, 2 distinct outputs, library has bool-returning macros.
- **Higher-order** (7%): `reduce`, `string-join(map(...))` — element-wise transformation. Spec features: output structure mirrors input structure.
- **If-expression** (1%): `if(cond, ...)` — D&C. Spec features: 3+ distinct string outputs.
- **Memo/lookup** (4%): `ns-get-or` — no computable pattern. Spec features: arbitrary input→output mapping.
- **Delegation** (15%): outermost is a promoted macro. Spec features: composition of previously learned capabilities.
- **Constants/identity** (24%): literal values. Spec features: constant or identity output.

**The decomposition prediction curriculum:**

Stage D0: **Classify outermost function family.** Given spec features (input type, output type, output cardinality, input structure), predict which family: constant, comparison, boolean-comp, arithmetic, string-op, higher-order, if-expression, memo. Training data: 74 (features, family) pairs from library.

Stage D1: **Predict specific outermost function.** Within the predicted family, predict the exact function. For boolean-comp: is it `and`, `or`, or `not`? For string-op: is it `string-take`, `string-ends-with`, `count-char`? Training data: same 74 pairs, finer labels.

Stage D2: **Derive subspecs.** Given f and the examples, compute the subspec(s). This is the "inversion" step. For `f = string-take`: subspec is (input → string-length(output)). For `f = and`: subspecs are the two predicate specs. This can be implemented as template-specific inverters.

Stage D3: **Recursive composition.** Chain stages D0-D2 into a recursive synthesizer that calls itself on subspecs. Fall back to flat search for leaf-level subproblems (depth 1 compositions).

**Relationship to current meta-optimization:** The decomposition predictor **replaces** priority tuning. Instead of ordering candidates within brute-force search, it predicts the answer structure top-down and only uses brute-force for leaf problems. Meta-optimization (Stage 3-4) becomes a special case: the priority heuristic is equivalent to a decomposition predictor that always selects "flat search" but reorders the candidates.

**Implementation approach:** The predictor should be a SELPH program learned via the existing synthesis infrastructure. Given that the training set is small (74 examples), the predictor can be a decision tree (D&C synthesis) or a memorized lookup table, and will grow as the library grows.

**Prototype validated (April 9, 2026):** Hand-written decision tree predictor in `examples/decomposition_predictor.selph` scores 7/7 on test cases. Predicts family from (input_type, output_type, has_bool_macros, num_distinct_outputs). Key decision boundaries: bool output + bool macros → bool-comp; bool output without → compare; num output + list input → arith; num output + str input → count; str output + high cardinality → if-expr.

**Concrete recursive synthesizer design:**

```lisp
(define recursive-synthesize
  (lambda (spec library depth-limit)
    ; Base cases
    (if (= depth-limit 0) (memorize spec)
      (let ((features (extract-features spec))
            (family (predict-family features)))
        ; Select candidate outermost functions for this family
        (let ((candidates (family->functions family library)))
          ; Try each candidate f
          (reduce (lambda (best f)
            (if (ns-get best "found") best
              ; Derive subspecs by inverting f on examples
              (let ((subspecs (invert f spec)))
                (if (nil? subspecs) best
                  ; Recursively solve each subspec
                  (let ((sub-results (map (lambda (ss)
                    (recursive-synthesize ss library (- depth-limit 1)))
                    subspecs)))
                    ; If all subspecs solved, compose and verify
                    (if (all (lambda (r) (ns-get r "found")) sub-results)
                      (let ((composed (compose f sub-results)))
                        (if (test-spec composed spec)
                          (ns ("found" true) ("source" composed))
                          best))
                      best))))))
            (ns ("found" false))
            candidates))))))
```

The key functions that need to be learned/implemented:
1. `predict-family` — learned from library (Stage D0, prototype done)
2. `family->functions` — maps family to candidate functions (from library metadata)
3. `invert` — given f and input→output pairs, derive the subspec. Per-family templates:
   - arith: if f is binary like `add`, try splitting output = f(a, b) for each pair
   - bool-comp: if f is `and`, derive two bool subspecs
   - string-op: if f is `string-take`, derive the numeric subspec
   - if-expr: partition examples by output value, derive condition + branch specs
   - compare: if f is `string-ends-with`, derive what string to check
4. `compose` — build the AST from f and sub-solutions (mechanical)
5. `test-spec` — already a builtin

~~**Next implementation steps:**~~
1. ~~Implement `invert` for the arith and string-op families~~ ✓
2. ~~Wire into the grow command as Strategy 0~~ ✓
3. ~~Measure: how many tasks does recursive decomposition solve at depth 1 that flat search needs depth 2+ for?~~ ✓
4. ~~Learn `predict-family` via synthesis~~ ✓

### 9.22 Recursive decomposition implementation (April 9, 2026)

New module `recursive_decompose.rs` (~900 lines) implementing synthesis as top-down prediction. Wired into the grow command as **Strategy 0** (before flat synthesis), using 1/4 of the total budget.

**Core pipeline:** predict_family → family_candidates → try_single_function (invert → sub-synthesize → compose → verify) → fallback to flat synthesis.

**Family prediction:** Hand-coded decision tree classifies specs by (input_type, output_type, num_distinct_outputs, has_bool_macros, output_is_substring). Learned predictor support via `--learn-rd` (trains after grow) and `--rd-predictor <file>` (loads for use). Memo-based predictor as fallback when D&C synthesis fails on training data.

**Inversion implemented for 6 families:**

| Family | Inversions | Key patterns |
|--------|-----------|--------------|
| Arithmetic (unary) | negate | `output = -g(x)` → `g(x) = -output` |
| Arithmetic (binary) | add, subtract, multiply, divide, pow | Both `f(x,k)` and `f(k,x)` orderings |
| String-op | string-take, string-drop, concat, string-upper/lower/reverse | Prefix/suffix detection, character-level inversion |
| Count | string-length, count-char | Direct application, character search |
| Compare | string-starts-with, string-ends-with, contains, even, odd | Constant extraction from true/false examples |
| HO | map (list + split-map-join), filter | Per-element subspec derivation |
| IfExpr | if (D&C delegation) | Delegates to `divide::divide_and_conquer` |

**Macro-aware decomposition:** `try_macro_decomposition` evaluates each unary macro on inputs, checks for direct match or useful bridge intermediate, then sub-synthesizes `f` such that `f(macro(input)) = expected`. Composes as `(lambda (x) (f (m x)))`.

**Composition fix:** `extract_body` unwraps the Lambda returned by sub-synthesis so composed programs are `(string-take x (half_len x))` not `(string-take x (lambda (x) (half_len x)))`.

**Guards:** Arithmetic inversion skipped for list inputs (prevents regressions on sequence tasks). `output_is_substring` overrides IfExpr classification for substring extraction tasks.

**CLI flags:**
- `--no-rd` — disable recursive decomposition
- `--learn-rd` — after grow, collect (features, family) training pairs and synthesize a predictor
- `--rd-predictor <file>` — load a learned predictor for family prediction

**Results:**

| Curriculum | Solve rate | RD contributions |
|------------|-----------|------------------|
| rd_test (12 tasks) | 12/12 | 12 by RD, **74 total candidates** (was 4686 flat-only) |
| NL (22 tasks) | 22/22 | is_plural/is_gerund: 0 cand; pos_tag: 0.05s via RD(if/D&C) (was 13.8s) |
| sequence (13 tasks) | 12/13 | No regression (cubes still unsolved) |
| first_half/second_half | 2/2 | **11 cand each** via RD(string-take/drop) with promoted half_len |

Key wins:
- **Zero-cost solutions** for direct applications and comparison-with-constant patterns
- **260x speedup** on pos_tag (D&C runs as Strategy 0 instead of waiting for flat to fail)
- **64x candidate reduction** on rd_test (74 vs 4686)
- **Semantically correct solutions** — avoids coincidental flat-search matches
- **Macro bridge** unlocks `first_half = (string-take x (half_len x))` at 11 candidates

**Completed (April 9, 2026 — recursive composition):**

- ~~Recursive multi-level decomposition~~ ✓ — `recursive_sub_synthesize` tries RD recursively (depth-limited, default 2 levels) before falling back to flat synthesis. Enables multi-level top-down prediction: e.g., predict `string-upper` as outermost → invert → recursive RD predicts `string-take` for subspec → compose `(string-upper (string-take x 3))` in 4 candidates (vs 452 flat). Overall 5.8x candidate reduction on multi-level tasks (283 vs 1637). No regressions on any curriculum.

**Completed (April 9, 2026 — generic inversion & macro evaluation fix):**

- ~~Learn `invert` per-family from solved programs (Stage D2)~~ ✓ — Generic evaluation-based inversion replaces per-function hand-coding for new functions. Three new capabilities:
  1. **Fixed macro evaluation bug:** `try_macro_decomposition` was silently failing because it called `eval` on macro body nodes (which have unbound parameters). Fix: construct `Value::RustMacro` directly and use `eval::apply`. This unlocks all `f(m(x))` bridge patterns.
  2. **Generic binary inversion** (catch-all in `try_single_function`): For any binary function not in the hand-coded match, probes `f(input, k)` and `f(k, input)` with candidate constants, then sub-synthesizes variable-k patterns. Covers `min`, `max`, `slice`, `nth`, and any future builtins.
  3. **Macro-as-outermost** (Phase 3 in `try_rd_recursive`): Precomputes intermediate table of all unary functions applied to inputs, then checks `m(g(input)) = expected` for each (macro, function) pair. Enables `count_a(first_two(x))` and similar patterns where a promoted macro is the outermost function.

  Results:
  - `half_len` found as `RD(halve∘string-length generic)` — 0 candidates (promoted macro as outermost)
  - `count_a_prefix` found as `RD(count_a∘first_two generic)` — macro-as-outermost with macro-as-inner
  - `shout_first` found as `RD(f∘first_word)` — newly working bridge decomposition
  - `last_word_len` found as `RD(f∘last_word)` — newly working bridge decomposition
  - No regressions on rd_test (12/12), sequence (12/13), rd_bridge_half (2/2), rd_bridge_test (8/8)

**Completed (April 10, 2026 — eval::apply env hoisting fix):**

- ~~Performance regression in macro evaluation~~ ✓ — `eval::apply` for `Value::RustMacro` was rebuilding the entire default env from scratch on every macro call (line 263), then copying the caller's env into it. Each rebuild allocated ~150 builtin entries plus a `__builtins__` introspection namespace with ~120 metadata entries (each its own HashMap with name/arity/params/returns + Vec<String>). For nested macro calls, this compounded recursively — a single curriculum task could trigger thousands of make_default_env() calls.

  **Fix:** push parameter scope onto the existing env, eval, then pop. The caller's env already contains the default builtins and macros, so no rebuild is needed. The lexical scope chain is preserved more naturally too.

  **Impact on full curriculum (55 tasks):** 23+ minutes → 18 minutes (54/55 solved, was incomplete in previous runs). The slowdown was entirely from environment construction, not evaluation. **No VM/bytecode tricks needed** — the interpreter just had to stop rebuilding its symbol table on every function call.

  **Lesson:** Before reaching for compiler tricks, check whether the interpreter is doing avoidable work in its hot path.

  Also fixed in this session: `examples/full_curriculum.selph` now uses native list inputs `(4 0 1 2 3)` instead of string-encoded `"4 0 1 2 3"`, matching `examples/sequence_tasks_list.selph` so the chained curriculum works correctly.

**Next implementation steps:**
1. Improve learned predictor: train across multiple curricula, better feature engineering
2. Deeper recursion for 3+ level compositions (currently depth 2)
3. ~~Investigate other `make_default_env()` callers in `map`/`reduce`/`filter`/`test-spec` for similar wins~~ — see §9.23

### 9.23 Hot-path env hoisting and caller-env threading (April 10, 2026)

Continuing from §9.22's `eval::apply` env-rebuild fix, audited every `make_default_env()` caller in `synth.rs` and `eval.rs` for the same pattern: rebuilding the default environment inside a loop or per-call when a single hoisted env would do.

**Hoisted out of inner loops in `synth.rs` (six sites):**
- The tree-walker fallback in the main `test` closure (per-input env build → per-candidate env build).
- `validate_candidate` — env rebuilt per validation example.
- `generate_if_programs` — D&C had two `for inp in inputs` loops (bool conditions + value branches), each rebuilding env per (pool entry, input). Hoisted to function scope so a single env is reused across both passes.
- `eval_fitness` and `test_and_score` in `synthesize_optimize` — env rebuilt per base example.
- The probe-and-filter `.filter()` closure — env rebuilt per probed component. Hoisted out of the closure with `let mut probe_env`.
- `infer_macro_types` `try_call` closure — env rebuilt on each probe call. Wrapped in a `RefCell` so the closure can reuse it.
- `make_selph_filter` and `make_selph_scorer` — these closures are called once per `(component, depth)` pair across an entire synthesis run, and were rebuilding the default env on every call. Now capture `RefCell<Env>` at construction time.

**Hoisted in `eval.rs`:**
- `test-spec` — env was rebuilt for each spec pair. Now built once before the loop.

**Caller-env threading for higher-order builtins (`eval.rs`):**
Added `apply_builtin_in_env(name, args, Some(env))` alongside the existing `apply_builtin(name, args)`. The `Value::Builtin(name)` arm in `eval::apply` now threads the caller's env into the slow path. `map`, `reduce`, `filter`, `apply`, and `test-spec` reuse the caller's env when invoked from a tree-walker context. This:
1. Avoids rebuilding the default env per inner call (the same fix as §9.22 but for the higher-order family).
2. Fixes a latent bug: when `map`/`filter`/`reduce` is called with a `RustMacro` whose body references *other* macros, the freshly-built default env didn't contain those other macros, so the call would fail. The hoisted-env path threads the surrounding macro environment through.

**Validation:** Sequence stage standalone (13 tasks, budget 200K): baseline 72.2s → optimized 47.6s (**1.52x speedup**). 12/13 solved either way. 230 tests passing (same 9 pre-existing `multitree::tests::test_extract_*` failures, unrelated to this change).

**Lesson reinforced:** the April 10 macro-apply fix, today's hoistings, and the upcoming Tier-1 changes (§9.24) are all the same pattern — a small allocation that happens in a hot loop dwarfs hundreds of lines of compiler infrastructure built to "speed things up." Audit before optimizing.

### 9.24 Interpreter rebuild and types-as-SELPH (April 10, 2026)

After §9.22 and §9.23, the picture is clear: the tree-walking evaluator isn't slow because tree walking is slow. It's slow because its data structures and hot paths do pathologically wasteful things that are orthogonal to the tree-vs-bytecode distinction. Each fix we land is a 1.3–1.5x speedup from removing a single allocation. The bytecode VM (22.9x speedup, ~600 lines) was a workaround for problems that lived elsewhere; after the env-rebuild fix the gap to the tree walker shrank dramatically and the VM is increasingly net-negative complexity.

This subsection sketches the proposed direction: **rebuild the core interpreter (`types.rs` + `eval.rs` + small support, ~2500 lines) against new data structures, migrate consumers module-by-module, delete the VM, and bake in the substrate for the "types as SELPH program" curriculum target referenced in §9.13.**

#### 9.24.1 Tier-1 hot-path fixes (the rebuild's data-structure motivation)

**Value representation.** Currently `Value::Str(String)` and `Value::List(Vec<Value>)` deep-clone on every clone. Every `apply_builtin` slow-path call does `let l = list(&args[1])?` which clones the entire vector AND every nested value. For string-heavy curricula this is enormous.
- Switch to `Value::Str(Rc<str>)` and `Value::List(Rc<[Value]>)`. Cloning becomes a refcount bump.
- Affects nearly every consumer because almost everything threads `Value` by value, but the migration is mechanical.

**Closure / Env representation.** `Lambda → Closure(..., env.clone(), ...)` deep-clones the env (a `Vec<HashMap<Sym, Value>>`) on every lambda creation; `apply` does it again on every call. The default env's ~270 entries (including the `__builtins__` introspection namespace) get copied each time. This is the same root cause as the §9.22 fix, but for closures rather than rust-macros.
- Switch to a persistent env: `Env = Rc<Scope>` with parent pointers. Closure capture is O(1); closure call pushes one new scope via Rc. Single change probably exceeds the entire bytecode VM's contribution to runtime.

**`env_lookup` linear walk.** Every `Node::Symbol(name)` walks the scope stack from top to bottom on every reference. The default env (with all builtins + macros) sits at the bottom, so common names like `add`, `string-upper`, `nth` pay full-stack-walk cost on every lookup inside any function body. HashMap default `SipHash` on `u32` Sym keys is overkill.
- `FxHashMap<Sym, Value>` for scopes (drops SipHash).
- Optionally: resolve symbols at parse time into `(scope_depth, slot)` indices so eval is O(1). The VM already does this for builtins via `CallBuiltin(sym, arity)`; pre-resolution would generalize it without keeping the bytecode infrastructure around.

**Special-form dispatch.** `eval_inner` does `let name_str = resolve(*name); match name_str.as_str() { "define" => ..., "do" => ..., ... }` on EVERY function application. Every `(add x 1)` pays for: resolve sym → string → ~13 string-equality compares → fall through.
- Pre-intern special-form Syms once. Compare `Sym` u32s, or use a small dense match. Eliminates a string allocation + a chain of string compares per function call — and this is per-App-node in the absolute hottest loop.

**`apply_builtin_slow` does the same `resolve → string → match` pattern** — same fix.

**Tier-2 wins (lower priority):**
- `recursive_decompose.rs` has 7 more `make_default_env()` callers; same hoisting opportunities.
- `value_to_string` in `string-join` allocates a `String` per element then joins them; should append into a single buffer.
- `BUILTIN_DISPATCH.with(|d| d.get(&name).map(|f| f(args)))` per-call thread-local + HashMap lookup. A static perfect-hash table or dense Sym-indexed Vec lookup avoids the closure overhead.

**Tier-3 (architectural):**
- The VM is probably net-negative once Tier-1 lands. It only handles a subset of nodes (no `ns`, no closures, no `dispatch`, no `let`). Its claimed 22.9x speedup was measured against an env-rebuilding tree walker. After the §9.22 fix that gap collapsed; after Tier-1 it should collapse entirely. Bug surface from "VM compiles but executes wrong" is real (we hit it as bug #4 in §9.20). Right experiment: apply Tier-1, force `force_tree_walker` everywhere, measure. If the curriculum runs in roughly current time, delete `vm.rs`.
- AST resolution at parse time: rewrite `Node::Symbol(Sym)` into `LocalVar(slot)`, `BuiltinRef(fn)`, `MacroRef(rc)` based on lexical scope. This is what most fast Lisps do. Combined with persistent env it gets to roughly Stalin-Scheme territory without leaving the interpreter model.

#### 9.24.2 Why this is a rebuild, not a patch

The reasons it makes sense to think of this as rebuilding the core rather than patching in place:

1. **The data-structure changes ripple anyway.** Every `make_default_env`-rebuild site we're hoisting is also a site that would change under the new Env representation. If we patch in place, we touch each consumer twice. If we rebuild against a stable new API, we touch them once.
2. **The semantics are well-understood.** It's a Lisp with lexical scope, defmacro, first-class namespaces, dispatch. We're transcribing, not designing. A rewrite of `types.rs` + `eval.rs` is 1–2 days of focused work.
3. **The current code has accumulated cruft.** `dispatch` as a special form (because builtins can't see env), `ns` as a special form, two parallel evaluators (tree walker + VM), multiple env-building patterns scattered across modules. A clean version is much smaller.
4. **It unlocks the type-system rewrite (§9.24.3) at the same moment.** The two changes share data-structure dependencies — doing them sequentially means migrating consumers twice.

What stays untouched: `synth.rs`, `recursive_decompose.rs`, `induce.rs`, `divide.rs`, `verify.rs`, `abstraction.rs`, `multitree.rs`, `stochastic.rs`, `meta.rs`, `taskgen.rs`, `trace.rs`, `decompose.rs`, `library.rs` — collectively ~10K+ lines, validated by curricula. They consume `Value`/`Env`/`Node` and only need touching where the API surface changes (which is exactly the surface we're auditing for hoisting opportunities anyway).

#### 9.24.3 Types as a SELPH program

The current type system has a structural smell: types live in two places that mirror each other.
- **At the value level**: closed Rust enum variants — `Num`, `Str`, `Bool`, `List`, `Grid`, `Namespace`, `Alt`.
- **At the synthesis level**: `u8` tags `TYPE_NUM=0, TYPE_STR=1, TYPE_BOOL=2, TYPE_LIST=3, TYPE_ANY=255` plus grid tags.

These are doubly-encoded — every new category requires a Rust enum variant AND a `u8` tag AND HM updates AND reachable-types updates AND probe-filter updates. That's why ARC-AGI required `Value::Grid` AND `TYPE_GRID` AND ~50 grid components AND a lazy-registration flag. The type vocabulary is calcified into Rust at two levels at once. You can't learn it; you can't extend it without recompiling; you can't even hide grid components without a Rust-level boolean.

The "types as SELPH" idea is the dual of "code as SELPH data" — both are about pulling something out of Rust into the homoiconic data layer where it can be learned, transformed, and composed. **A type becomes a namespace:**

```lisp
(deftype Number
  (predicate (lambda (v) (number? v)))
  (parents ())
  (priority 0))

(deftype Grid
  (predicate (lambda (v)
    (and (list? v) (all list? v) (rectangular? v)
         (all (lambda (row) (all int? row)) v))))
  (parents (List))
  (priority 50))
```

The type *namespace* (`__types__`) becomes the source of truth. The synthesizer caches Sym handles into that namespace for hot-path checks, but it's no longer a closed enum. Same shadow-cache pattern as intern (Sym shadows String) and the VM (bytecode shadows the tree).

**What changes in the interpreter rebuild to support this:**

1. **`Value` collapses.** `Grid`, `Alt`, possibly `Namespace` are no longer Rust variants. A grid is just `List` of `List` of `Num`; "grid-ness" is a predicate in the `Grid` type namespace. Faster cloning, smaller enum, fewer match arms, and crucially: **the Rust universe is no longer the type universe.**

2. **`SynthComponent` references types by Sym, not u8.** Type-reachability becomes a graph search over Sym-keyed nodes instead of a u8 bitmap. Slightly slower per-check, much more flexible.

3. **`infer_macro_types` becomes principled.** Right now it probes a macro with `Value::Str("5 1 2 3 4")` and `Value::Num(3.0)` and pattern-matches the result against the u8 enum. In the new world, it asks: "for each registered type T, does `(T.predicate (m sample))` return true?" Generalizes for free to user-defined types; the same predicate that synthesis uses is the one that learning uses.

4. **`hm.rs` either dies or is rewritten.** HM over u8 tags doesn't directly apply when types are predicates. This is a real loss in one direction (no parametric polymorphism for free) and a gain in another (refinement types, runtime-checked, learnable). Most synthesis use of HM is filtering, which still works with predicates: "for the example values, does this candidate's output satisfy the target predicate?"

5. **Grid stuff becomes a SELPH library.** `examples/grid.selph` defines the Grid type, the grid operations, the perceptual primitives. ARC-AGI ports from a Rust patch to a curriculum stage. The `--include-grid` flag and lazy registration disappear.

#### 9.24.4 Architectural shape after the rebuild

```
Layer 0 (Rust):       Value, Env, eval, parser, intern.
                      ~1500 lines. Stable. No types here.
Layer 1 (SELPH):      __types__, __builtins__, primitive type defs.
                      Loaded at startup. Replaceable.
Layer 2 (SELPH):      Synthesis support, decomposition, heuristics,
                      type learning. Curriculum target.
Layer 3 (Rust):       synth.rs, recursive_decompose.rs, etc.
                      Cached fast paths for what SELPH expresses.
                      Optional — the system would still work without
                      them, just slower.
```

The Rust ↔ SELPH boundary moves UP. More of the system lives as SELPH data. The Rust layer becomes a runtime + accelerator, not the source of semantic truth. This is the trajectory the plan keeps gesturing at ("the trained model = library file", "the library IS the dispatch table", "self-hosting infrastructure"). Right now those phrases are aspirational because the type system is still in Rust. After the rebuild they become literal.

#### 9.24.5 Concrete next steps

1. ~~**Land §9.23 benchmark.**~~ Cancelled — strategic shift to the rebuild path made the §9.23 baseline obsolete. Those env-hoisting changes are still committed (they're correct improvements to the old core, and they validated the rebuild thesis: 1.52x sequence-stage speedup from a 5-line fix), but no full-chain benchmark was needed.
2. ~~**Sketch the new `Value` enum.**~~ ✓ See §9.25.1. `Rc<str>` / `Rc<[Value]>`, no `Grid`, no `Alt`, `Namespace` kept as a fast Map variant keyed by `Sym`. Closures and macros collapsed into a single `Function` variant per the decomposition philosophy. `Int(i64)` added as a distinct variant from `Num(f64)`.
3. ~~**Sketch the type representation.**~~ ✓ Deferred per user direction. Types will live in a SELPH namespace tree where namespace nesting encodes the subtype hierarchy; the eventual `__types__` namespace is the source of truth and the synthesizer caches Sym handles into it. The interim shim is a `Value::type_sym()` method returning a Sym for primitive variants — minimal surface for synth migration.
4. ~~**Rewrite `eval.rs` against the new types.**~~ ✓ See §9.25.2. `eval_v2.rs` is in place: persistent Env, special forms as a Rust enum matched in `eval_inner`, no `Grid` handling, ~75 builtins ported (arithmetic, comparison, string, list, namespace, type predicates, higher-order). Bucket-6 meta builtins (`synthesize`, `test-spec`, `memorize`, `eval-source`, `synthesize-optimize`) are stubbed pending the synth.rs migration. Wired into `selph eval-v2 <file>` and validated end-to-end against `examples/hello.selph`, `examples/heuristics.selph`, `examples/decomposition_predictor.selph` (7/7 correct).
5. **NEXT: Migrate `synth.rs`** to Sym-typed components, calling type predicates for the probe step. Rewrite or stub `hm.rs`. Decide between (a) writing `synth_v2.rs` from scratch against the new types or (b) compatibility shim with a Value boundary conversion. Lean toward (a) for the same reason we did the rebuild — incremental approach means doing the work twice. See §9.25.3.
6. **Migrate `recursive_decompose.rs`, `divide.rs`, `induce.rs`, `verify.rs`, etc.**, one module at a time, behind a feature flag if needed.
7. **Delete `vm.rs`** once the new tree walker hits comparable numbers.
8. **Reload the curricula.** They should still pass — the new type model is more flexible, not more restrictive. Grid stuff becomes a SELPH library.
9. **Then** start the curriculum target: teach SELPH programs to predict types from spec features, learn type predicates from examples, infer types from usage. This is the §9.13 "Types as a SELPH program" item, finally on a foundation that allows it.

**Decision point:** step 2 was the "is this even worth it" gate. Resolved: the new types feel obviously better. We're proceeding.

**Estimated payoff:** the working hypothesis is 3–5x curriculum runtime improvement and ~30% codebase reduction (deleting VM, simplifying special forms, removing make_default_env-rebuild scaffolding, removing the Grid variant). More importantly, it sets up Phase 4 (neural generation), ARC-AGI proper, and Phase 5 (self-curriculum) on a foundation that doesn't fight the rest of the plan.

### 9.25 Rebuild milestone: types_v2 and eval_v2 (April 10, 2026)

Steps 2 and 4 of §9.24.5 landed in a single session. This subsection captures what was actually built so future work has a concrete reference.

#### 9.25.1 `selph_fast/src/types_v2.rs` (~680 lines)

The new core types live alongside `types.rs` without disturbing it. Highlights:

- **`Value` enum** — 9 variants:
  - `Int(i64)` and `Num(f64)` are distinct. Coercion rule: `Int op Int = Int` (with overflow → Num); `Int op Num = Num`. Many builtins naturally produce Int (`string-length`, `nth`, `length`, `count-char`, list lengths, etc.) and synthesis can now reason about integer-vs-float without sniffing `n == (n as i64) as f64`.
  - `Str(Rc<str>)` and `List(Rc<[Value]>)` — cloning is a refcount bump, not a deep copy.
  - `Ns(Rc<NsMap>)` where `NsMap = HashMap<Sym, Value>`. Sym keys drop a lot of allocation in `ns-get` paths; lookup is hashed-u32 instead of hashed-String.
  - `Function(Rc<FunctionData>)` — collapses today's `Closure` and `RustMacro`. The data carries `params`, `body: NodeRef`, `captured_env: Env`, and an optional `letrec_scope`. The decomposition rationale: when a sub-synthesizer wants to use a function as a primitive, the captured env IS the relevant search context — no inheritance of the parent's full search environment. Capture is O(1) (Rc bump of the env node).
  - `Builtin(Sym)`, `Bool`, `Nil`.
  - **Gone**: `Grid`, `Alt`, distinct `Closure`/`RustMacro`. Grid becomes a SELPH library (eventually); Alt was synth-internal and lifts out of Value entirely.

- **`Env`** — persistent linked list of scopes via `Rc<EnvNode>`. Cloning is one Rc bump; `push_scope` is O(1); `lookup` walks parents top-first; `define` mutates the top scope in place via `RefCell`. The bottom default scope (~75 builtin entries) is built once at startup and shared via Rc across every env — no more 270-entry HashMap copies on every lambda.

- **`Node` and `SpecialForm`** — special forms become a Rust enum matched directly in `eval_inner`. No string compares, no `match name_str.as_str() { "define" => ... }` chain in the hot path. `Node` adds `Int(i64)` and `SpecialApp(SpecialForm, Vec<usize>)` variants.

- **`SpecialForm` enum** — `Define`, `Do`, `Quote`, `And`, `Or`, `Try`, `EvalIn`, `Dispatch`, `Ns`. **No `Defmacro`** — the parser desugars `(defmacro name (params) body...)` to `(define name (lambda (params) (do body...)))`. This is sound because the new `Function` captures its definition-site env, which at top level contains the rest of the library — the same set of bindings the old defmacro could see at call time.

- **Type substrate (placeholder)** — `Value::type_sym() -> Option<Sym>` returns the canonical primitive type Sym (`Int`, `Num`, `String`, `Bool`, `List`, `Function`, `Namespace`). Pre-interned in a thread-local for fast access. This is the minimal surface synth.rs needs to switch from `u8` tags to `Sym`-keyed types. The full type system (predicates, namespace tree, subtype hierarchy via namespace nesting) is deferred to a later milestone.

- **`BuiltinTable`** — Sym-indexed `Vec<Option<BuiltinFn>>`, no thread-local HashMap, no SipHash. Lookup is `table.get(sym.0 as usize).copied().flatten()`. `BuiltinFn` signature is `fn(args: &[Value], env: &Env) -> Result<Value, String>` — env is always passed (cheap because Env is Rc-internal), eliminating the §9.23 `Option<&mut Env>` threading complexity.

9 unit tests in `types_v2::tests` cover Value cloning, persistent env chains, special-form Sym dispatch, `Function`-with-captured-env, and primitive type Sym distinction.

#### 9.25.2 `selph_fast/src/eval_v2.rs` (~2000 lines)

The new evaluator uses `types_v2` and lives alongside `eval.rs`. Highlights:

- **`eval(nodes, idx, env: &Env)`** — depth-tracked tree walker. `&Env` (not `&mut`) because Env is Rc-internal; mutation flows through `env.define(...)` via `RefCell`. Handles all Node variants including the new `SpecialApp`.

- **`apply(f, args, env)`** — for `Function`, uses `fd.captured_env.push_scope(params)` (no clone, just Rc bump). For `Builtin`, dispatches via `BUILTIN_TABLE.lookup(*sym)`. The caller's env is used only for Builtin dispatch — Functions use their own captured env, which is the whole point of unifying closures and macros.

- **Special-form handlers** for `Define`, `Do`, `And`, `Or`, `Try`, `Dispatch`, `Ns`. `Quote` and `EvalIn` are stubbed with TODOs (need new-Node `node_to_source` and parser-v2 respectively).

- **Truthiness rule**: `is_truthy(v)` — falsy values are `Bool(false)`, `Nil`, `Int(0)`, `Num(0.0)`, empty `Str`, empty `List`. Everything else (including empty `Ns`, functions, builtins) is truthy. This matches the user's April 10 call: more aggressive than today's eval, which only treats `Bool(false)` and `Nil` as falsy.

- **~75 builtins ported** across buckets 1-5 (arithmetic, comparison, string, list, namespace, type predicates). Bucket 6 (synthesize, synthesize-optimize, test-spec, memorize, eval-source) stubbed with clear errors. Bucket 7 (grid) deliberately not ported — becomes a SELPH library.

- **Old-Node → new-Node converter** (`convert_tree`) — transitional shim that lets the existing parser feed eval_v2 without parser changes. Three jobs: classify integer-valued `Num` literals as `Int`, resolve App-with-special-form-symbol heads into `SpecialApp`, desugar `(defmacro name (params) body...)` into `(define name (lambda (params) (do body...)))`. Two-pass walk; appends new nodes for the desugared lambda. Throwaway code; deletes when parser.rs is updated.

- **CLI integration**: `selph eval-v2 <file.selph>` runs the existing parser, applies `convert_tree`, and evaluates against eval_v2.

30 tests in `eval_v2::tests` covering literals, lambda, define+call, letrec factorial, map/reduce/filter, all the new builtins, the converter, and end-to-end runs of `examples/heuristics.selph` and `examples/decomposition_predictor.selph`. All passing.

**Real-file validation:**
- `examples/hello.selph` — runs end-to-end through `selph eval-v2`, output identical to old `selph eval` except the desugared defmacro displays as `<lambda (x)>` instead of `<macro (x)>`.
- `examples/heuristics.selph` — runs end-to-end with zero errors.
- `examples/decomposition_predictor.selph` — returns "7 / 7 correct" (the §9.21 prototype works on the new core).
- `examples/scoping.selph` — runs **further** than old eval. The old eval errors on `unbound: defmacro` partway through (it can't handle multi-body defmacros). eval_v2 desugars them correctly and exposes a latent bug in the file at the very end where `(reduce f init list)` is called with the wrong argument order. **This is the first concrete case where eval_v2 is strictly more capable than eval.rs on a real curriculum file.**

#### 9.25.3 Bucket (b): synth.rs migration — the next step

This is the big remaining piece. `synth.rs` is ~3000 lines, references the OLD Value/Env types pervasively, and is the gateway to making the bucket-6 stubs (`synthesize`, `test-spec`, `memorize`, etc.) actually work in eval_v2.

**The core type-system change:** `SynthComponent { param_types: Vec<u8>, ret_type: u8, ... }` → `Vec<Sym>` and `Sym`. Type-reachability becomes a graph search over Sym-keyed nodes instead of a u8 bitmap. `infer_macro_types`'s "probe with str/num and pattern-match the result against the enum" pattern becomes "probe with sample, then for each registered primitive type Sym call `Value::type_sym()`." Generalizes for free when the full type system lands.

**Two implementation options:**

**Option A — Write `synth_v2.rs` from scratch** against the new types. Slimmer, more focused. Loses some battle-tested code but probably gains clarity by removing accreted scaffolding (the §9.23 hoisting, the per-strategy macro definitions, the multiple env-build patterns). Bucket-6 stubs in eval_v2 then delegate to synth_v2 functions. Lean toward this for the same reason we did the rebuild in the first place: incremental approach means doing the work twice.

**Option B — Compatibility shim** at the Value boundary. eval_v2 calls into synth.rs by converting new Value ↔ old Value at the call site. Faster to land but the Sym-keyed type-system migration doesn't get the benefit, and the conversion cost is real. Probably wrong long-term.

**Suggested ordering for Option A:**
1. Define `synth_v2::SynthComponent` with `Vec<Sym>` types.
2. Port `default_synth_components` builder, generating components from the new builtin table and registered macros.
3. Port `synthesize_full` core loop. The §9.23 env-hoisting work informs the new design — env construction is explicit and minimal, not rebuilt per candidate.
4. Port the probe-and-filter pipeline using `Value::type_sym()` for type identity.
5. Port the strategy pipeline (Flat, BD, HO, D&C, Memo) one strategy at a time, validating each against representative tasks.
6. Wire bucket-6 stubs in eval_v2 to delegate to `synth_v2`.
7. Validate the chained curriculum end-to-end on the new core.

**Decision points along the way:**
- Does `infer_macro_types` survive, or does it get replaced by something cleaner now that types are first-class Syms?
- Does `hm.rs` survive at all, or does the predicate-based approach replace it entirely? (Current bet: replace.)
- Can the bucket-6 SELPH-vs-Rust question get answered as part of this work? (User note: "I have a feeling that we will want those to be defined in selph eventually, but I'm not exactly sure right now.")
- Is `recursive_decompose.rs` migrated as part of bucket (b) or as a separate step? It depends on synth's API surface, so probably as a separate step (§9.24.5 step 6).

This is a multi-day project. The pacing should be: get the core SynthComponent + synthesize_full path working against the new Value, then port one strategy at a time, validating against the existing curricula at each step.

---

## 10. Success Criteria

The growing system plan succeeds if:

1. **Phase 2 validation:** A learned SELPH heuristic outperforms the default ordering on held-out tasks without domain-specific engineering. **Partially met:** `priority-plus-type-match` solves 5 more tasks than baseline (26 vs 21 at budget 5000) with up to 50x speedup on individual tasks. The heuristic is domain-general (return-type matching), not domain-specific. Stage 4 (synthesized heuristic search) implemented — can now search the space of heuristic programs instead of choosing from hand-crafted templates. Remaining: validate Stage 4 on full traces and on held-out tasks from unseen domains.

2. **Phase 3 validation:** A learned SELPH decomposer solves tasks that the flat solver + hand-written induction can't. **Substantially met:** Recursive decomposition (§9.22) is now Strategy 0 — it runs *before* flat synthesis, predicts the outermost function family, inverts to derive subspecs, and recursively sub-synthesizes. All strategies (flat, BD, HO, D&C, memo) are unified under one recursive step. The family predictor can be learned via `--learn-rd`. Remaining: recursive multi-level decomposition (D3), learned inversion (D2).

3. **Phase 4 validation:** A neural SELPH generator (trained on synthesis logs) proposes correct programs in fewer attempts than the enumerative solver.

4. **Phase 5 validation:** The system, given a new domain with new primitives, autonomously designs a curriculum that reaches competence without human-specified task ordering.

Each criterion is measurable with the existing benchmarking infrastructure.
