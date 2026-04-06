# SELPH

**Symbolic Evaluation Language for Programmable Hierarchies**

A homoiconic language for program synthesis where the system grows its own capabilities through curriculum-driven library learning.

## Quick Start

```bash
# Build the standalone binary (requires Rust)
cd selph_fast
rustc --edition 2024 -O -o selph src/main.rs

# Evaluate expressions
./selph eval -e "(add 1 2)"                          # => 3
./selph eval -e "(map (lambda (x) (* x x)) (list 1 2 3 4 5))"  # => (1 4 9 16 25)

# Synthesize programs from examples
./selph synth -e "0->1 5->6 -1->0"                   # => (lambda (x) (add x 1))
./selph synth -e "hello->HELLO world->WORLD"          # => (lambda (x) (string-upper x))

# Synthesize with a library
./selph synth spec.selph --library my_lib.selph

# Interactive REPL
./selph repl
```

## What It Does

SELPH synthesizes programs from input/output examples. Give it examples of what a function should do, and it finds the simplest program that matches.

```
$ selph synth -e "1->2 3->6 5->10 0->0"
(lambda (x) (add x x))
```

The key feature: **solutions become building blocks for harder tasks.** A solved program gets saved to a library file as a named macro, and future synthesis can compose it with other primitives at no extra depth cost.

## The Growing Library

SELPH libraries are `.selph` files containing `defmacro` definitions:

```lisp
; seq_helpers.selph — sequence prediction accessors
(defmacro idx (s) (to-number (head (string-split s " "))))
(defmacro v0 (s) (to-number (head (tail (string-split s " ")))))
(defmacro last (s) (to-number (head (tail (tail (tail (tail (string-split s " "))))))))
```

When you solve a task, promote the solution to the library:

```lisp
; Add to library after solving "squares"
(defmacro seq_squares (s) (multiply (idx s) (idx s)))
```

Now future tasks can use `seq_squares` as a single-step primitive. For example, **cubes** becomes solvable as `(multiply (idx x) (seq_squares x))` — composing the position with the already-learned squares function.

### Demonstrated Library Cascade

Starting from 6 base accessor macros, the system discovers:

| Task | Solution | Builds on |
|------|----------|-----------|
| f(i) = i | `(idx x)` | base accessor |
| f(i) = i² | `(multiply (idx x) (idx x))` | idx |
| **f(i) = i³** | **`(multiply (idx x) (seq_squares x))`** | **promoted squares** |
| Fibonacci | `(add (v2 x) (v3 x))` | base accessors |
| sum_prev | `(seq_fib x)` | **promoted Fibonacci, 330 candidates vs 10K** |

Each promoted solution compresses the search space for the next task. Without the library, cubes is unreachable. With it, found in 50K candidates.

## Architecture

### Standalone Binary (665KB, no dependencies)

The `selph` binary includes:
- **Parser** — S-expression tokenizer + recursive descent
- **Evaluator** — 40+ builtins (arithmetic, strings, lists, higher-order)
- **Synthesizer** — bottom-up enumeration with type pruning and observational equivalence dedup
- **Macro system** — `defmacro` for library abstractions
- **Library loader** — reads `.selph` files, registers macros as synthesis components

### Python Package (for experimentation)

The `selph/` Python package adds:
- Hindley-Milner type inference
- Spec verification (Level 0-2 goals + reward computation)
- If-expression synthesis with expected-output matching
- Divide-and-conquer synthesis for nested branching
- Failure-driven induction (intermediate value decomposition)
- Optimization goals (`:minimize` / `:maximize`)
- Programmable SELPH heuristics for search ordering
- Hierarchical namespaces with scoped component access
- Multi-tree synthesis across arbitrary namespace trees
- Curriculum runner with automatic stage transitions
- Library pruning (observational + builtin equivalence)
- Persistent state (library + logs + metadata)

### Rust Accelerator (PyO3)

The `selph_fast` crate provides Python bindings for:
- `fast_synthesize` — synthesis with macros, 18,700x fewer candidates than Python on pair-min
- `fast_interleaved` — online priority learning during synthesis
- `batch_eval` / `batch_verify` — vectorized evaluation
- `synthesize_heuristic` — meta-synthesis of search heuristics

## Curriculum Design

The system's capabilities are determined by its curriculum — the sequence of tasks it's asked to solve. Each task's solution becomes a library primitive for the next.

### Sequence Prediction Curriculum

```
Level 0: Constants         — f(i) = c           → teaches constant functions
Level 1: Arithmetic        — f(i) = i, 2i, i+k  → teaches position and step
Level 2: Quadratic         — f(i) = i², i(i+1)/2 → composes Level 1 primitives
Level 3: Modular           — f(i) = i mod k      → teaches cyclic patterns
Level 4: Second-order      — Fibonacci, 2^i      → teaches recurrence from context
Level 5: Compositions      — i³, i² mod 3        → composes Level 2 + Level 3
```

### Formal Language Curriculum

```
Stage 0: Constant/repeat   — "aaa..." → 'a'      → teaches character constants
Stage 1: Alternation       — "abab" → 'a'         → teaches period-2 patterns
Stage 2: Arithmetic on chars — "abcde" → 'f'      → teaches idx + base_char
Stage 3: Repeat-each       — "aabb" → 'a'         → teaches longer periods
Stage 4: Mirror            — "abcba" → 'a'        → teaches palindrome structure
Stage 5: Proto context-free — depth patterns       → teaches nesting
```

### Validated Results

- **6-stage task curriculum:** 61/62 tasks solved (98%), Stages 0-4 at 100%
- **Library bootstrapping:** 0/4 → 4/4 at Stage 2 with library (98% search reduction)
- **Sorting pair-min:** 5 candidates with Rust (vs 93,429 with Python)
- **Cubes discovered:** `i × i²` found by composing promoted `squares` primitive
- **Fibonacci reuse:** 330 candidates with library vs 10,372 without

## The Growing System Thesis

> The model architecture is not specified — it is grown by the curriculum.

Instead of designing a fixed architecture and training it, SELPH grows its capabilities incrementally. Each curriculum stage teaches a capability that becomes infrastructure for the next. The "architecture" at any point is the set of learned SELPH programs that constitute the solver.

See [SELPH_Growing_System_Plan.md](SELPH_Growing_System_Plan.md) for the full roadmap.

### Meta-Curriculum Levels

```
Meta-0: Solve tasks               (done — 6 stages, 98% solve rate)
Meta-1: Learn search strategy     (started — heuristics as SELPH programs)
Meta-2: Learn decomposition       (started — induction/D&C in Python)
Meta-3: Learn pattern extraction   (library extraction in Python)
Meta-4: Neural candidate generation (next — constrained decoding)
Meta-5: Learn evaluation strategy  (future)
```

## File Structure

```
selph/              — Python package
  parser.py         — S-expression parser
  eval.py           — Tree-walking evaluator + Namespace values
  types.py          — Hindley-Milner type inference
  verify.py         — Spec verification + reward computation
  synthesize.py     — Bottom-up synthesis (Python)
  library.py        — Library extraction, promotion, pruning, persistence
  engine.py         — Unified synthesis interface
  state.py          — Persistent curriculum state
  ...

selph_fast/         — Rust crate
  src/main.rs       — Standalone binary (eval + synth + REPL)
  src/parser.rs     — Rust parser
  src/types.rs      — Shared types (Node, Value, Env)
  src/lib.rs        — PyO3 bindings

examples/           — Example SELPH programs and libraries
experiments/        — Curriculum experiments
tests/              — 404 tests
```

## Running Tests

```bash
# Python tests
python -m pytest tests/ -q

# Rust parser tests
cd selph_fast && cargo test
```

## Design Documents

- [SELPH_Architecture_Spec.md](SELPH_Architecture_Spec.md) — Original v0.1 spec
- [SELPH_Growing_System_Plan.md](SELPH_Growing_System_Plan.md) — Updated v0.2 plan
