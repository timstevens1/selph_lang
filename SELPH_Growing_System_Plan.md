# SELPH: Growing System Plan

## From Enumerative Solver to Self-Building Architecture

**Version 0.2 — April 2026**

Based on implementation experience with the v0.1 Architecture Spec. Supersedes §13 (Self-Hosting) and §14 (MVP) with a concrete, incremental path.

---

## 1. Core Thesis

The model architecture is not specified — it is *grown* by the curriculum. Each curriculum stage teaches the system a capability, and the learned capability becomes infrastructure for the next stage. The system bootstraps from a simple enumerative solver into an increasingly autonomous program synthesis engine, where every component — search, evaluation, heuristics, library management, task decomposition — is itself a SELPH program learned through earlier curriculum stages.

The architecture IS the curriculum. Change the curriculum, change the architecture.

---

## 2. What Exists (April 2026)

### 2.1 Language and Evaluator
- S-expression parser with full EBNF from v0.1 spec
- Tree-walking evaluator (Python + Rust) with ~60 builtins
- First-class namespaces: functions, data, and cache in the same tree
- Hindley-Milner type inference with shape tracking
- Macro system (defmacro) in both Python and Rust evaluators

### 2.2 Spec and Verification
- Specs as first-class values with typed goals (Levels 0-2 mechanical)
- `(:examples ...)` — input/output pairs (Level 0)
- `(:pattern ...)` — structural constraints (Level 1)
- `(:satisfy ...)` — predicate functions (Level 2)
- `(:minimize ...)` / `(:maximize ...)` — optimization objectives
- Reward computation: `hard_gate × (w_goal + w_parent + w_global)`
- Held-out validation for generalization

### 2.3 Synthesis
- Bottom-up enumerative search with type-directed pruning
- If-expression synthesis with expected-output matching
- Divide-and-conquer for multi-way classification
- Failure-driven induction (intermediate value decomposition)
- Observational equivalence deduplication
- Rust accelerated inner loop (18,700x fewer candidates on pair-min)

### 2.4 Library System
- Promotion: solved programs become single-step primitives
- Extraction: anti-unification + compression scoring
- Pruning: observational + builtin equivalence + usage tracking
- Persistence: save/load as SELPH source files
- Immediate promotion within stages (not just at transitions)

### 2.5 Search Strategy
- Programmable heuristics as SELPH functions
- Component priorities with type-aware + recency ordering
- Interleaved online learning: priorities update after each solve
- Multi-tree synthesis across arbitrary namespaces
- `synthesize` exposed as a SELPH builtin

### 2.6 Validated Results
- 6-stage curriculum: 61/62 tasks (98%), Stages 0-4 at 100%
- Library bootstrapping: 0/4 → 4/4 at Stage 2, 98% search reduction
- Sorting: pair-min in 5 candidates (Rust) vs 93,429 (Python)
- Stochastic process benchmarks with Bayes-optimal baselines

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

## 8. Immediate Next Steps

1. **Logging infrastructure:** Record all synthesis runs as (spec, solution, candidates, components_used, decomposition_strategy) tuples. This is the training data factory.

2. **Meta-1 curriculum:** Design tasks for learning search heuristics. Start with "predict output type from examples" (the simplest meta-task).

3. **Constrained decoding prototype:** Hook the type system into a small model's token mask. This is needed before Phase 4 but can be developed in parallel.

4. **Curriculum optimizer:** Given a task suite and a compute budget, find the ordering and difficulty gradient that minimizes total candidates. This is a meta-meta-task that the system could eventually learn.

---

## 9. Success Criteria

The growing system plan succeeds if:

1. **Phase 2 validation:** A learned SELPH heuristic outperforms the default ordering on held-out tasks without domain-specific engineering.

2. **Phase 3 validation:** A learned SELPH decomposer solves tasks that the flat solver + hand-written induction can't.

3. **Phase 4 validation:** A neural SELPH generator (trained on synthesis logs) proposes correct programs in fewer attempts than the enumerative solver.

4. **Phase 5 validation:** The system, given a new domain with new primitives, autonomously designs a curriculum that reaches competence without human-specified task ordering.

Each criterion is measurable with the existing benchmarking infrastructure.
