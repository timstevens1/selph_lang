# SELPH Optimization Extension — Progress

## Summary

Built a working pipeline where a neural model generates symbolic program structures as SELPH s-expressions, and a separate optimizer fits continuous parameters. Demonstrated on fitting tasks (linear regression through piecewise functions). SFT+GRPO training achieves 93-100% success rate on a curriculum of 15 tasks.

This work lives on the `rust_kb` branch and spans both the Rust kernel (new types, builtins, PyO3 bridge) and Python (optimizer, curriculum, neural synthesizer).

## What was built

### Phase 0: Param type (Rust)
- `Value::Param(f64, Option<f64>, Option<f64>)` — optimizable continuous parameter with optional bounds
- 6 builtins: `param`, `param?`, `param-value`, `param-bounds`, `freeze`, `thaw`
- Params participate in arithmetic (extract current value) and comparisons
- 17 tests
- **File**: `selph_fast/src/types_v2.rs`, `selph_fast/src/eval_v2.rs`

### Phase 1: Loss functions and datasets (Rust)
- `dataset` — validates list-of-pairs structure
- `dataset?` — predicate
- `mse`, `mae` — loss functions over prediction/target lists
- `model-loss` — applies a SELPH function to dataset inputs, computes loss against outputs
- 14 tests
- **File**: `selph_fast/src/eval_v2.rs`

### Phase 2: PyO3 bridge
- Restructured crate: `lib.rs` as shared module root, `main.rs` uses `selph_core::*`
- `pybridge.rs` with PyO3 bindings behind `python` feature flag
- `Env` class: `eval_expr()`, `load_file()`, `define()`, `lookup()`, `eval_to_string()`
- `Param` class with `value`, `min`, `max` attributes
- Full Value conversion: Int↔int, Num↔float, Str↔str, Bool↔bool, List↔list, Ns↔dict, Param↔Param, Nil↔None
- `pyproject.toml` for maturin builds
- Build: `maturin develop --features python` → `import selph_fast`
- Default build (no python feature) unchanged, all 305 tests pass
- **Files**: `selph_fast/src/lib.rs`, `selph_fast/src/pybridge.rs`, `selph_fast/Cargo.toml`, `selph_fast/pyproject.toml`

### Phase 2b: minimize() (Python)
- `selph_fast.minimize(env, params, loss_expr)` — optimizes SELPH Param values
- Two methods: Nelder-Mead (gradient-free, good for 2-20 params) and finite-difference gradient descent
- Returns `Solution` with final params, loss, step count, convergence flag, loss history
- Python drives the loop; SELPH evaluator computes the loss at each iteration
- Tested: linear regression (2 params, 2ms), multivariate (3 params, 8ms), polynomial (3 params), bounded optimization
- **File**: `selph_fast/selph_fast/optimize.py`

### Phase 3: Fitting curriculum (Python)
- 15 curriculum tasks: constants → linear → quadratic → piecewise → model selection
- 4 built-in model template families with optimizable `(param ?)` holes
- Library promotion: solved models become warm-started candidates for future tasks
- Model selection: tries all templates + library, picks simplest that fits
- Results: **12/15 solved**. Failures (cubic, relu, step) reveal where new templates needed
- **File**: `selph_fast/selph_fast/curriculum/fit.py`

### Phase 4: Neural structure synthesizer (Python + MLX)
- 107K-parameter transformer generates s-expression model structures
- ~30 token vocabulary covering SELPH math primitives
- `compute_reward()`: generate structure → `minimize()` fits params → MSE as reward
- **SFT warmstart**: 15 (task, structure) training pairs, 200 epochs, loss 1.67 → 0.05, **15/15 accuracy**
- **GRPO refinement**: group-relative advantages on rollout rewards, peaks at **100% success rate**
- **File**: `selph_fast/selph_fast/curriculum/neural_fit.py`

## Architecture

```
Neural model (107K params, MLX)
  │ generates s-expression structure
  ▼
"(add (multiply (param 0) x) (param 0))"
  │
  ▼
selph_fast.minimize()  ← Python optimizer (Nelder-Mead)
  │ evaluates loss via SELPH
  ▼
Env.eval_expr("(model-loss model data mse)")  ← PyO3 → Rust SELPH evaluator
  │
  ▼
Solution(w=2.0, b=1.0, loss≈0)
  │
  ▼
GRPO reward signal → update neural model
```

## Key design decisions

1. **Separate structure from parameters.** The neural model only learns ~30 tokens of structural vocabulary. Constants are handled by the optimizer. This is dramatically simpler than having a model learn both.

2. **SELPH describes, Python executes.** The SELPH evaluator computes loss symbolically. Python drives the optimization loop and GRPO training. No tensors in SELPH.

3. **Plain lists for datasets, scalar Param values.** No new Value variants for datasets. Param stores f64 directly (not a handle). This is correct for 2-20 parameter problems. Tensor-shaped parameters would need handles to external storage.

4. **PyO3 over subprocess.** Direct Python↔Rust FFI replaces the old subprocess-based bridge. ~1000x faster for hot-loop evaluation.

5. **SFT before GRPO.** Cold-start GRPO fails (0% success) because random token sequences are almost never valid s-expressions. SFT on 15 known-good pairs provides the syntactic foundation; GRPO refines toward better structures.

## Branch state

- **`main`**: includes all hole-fc decomposer work (merged from hole-fc)
- **`rust_kb`**: all of the above + merged neural_selph experiments
- **`neural_selph`**: original neural SELPH work (KB, GRPO, traces) — now merged into rust_kb

## What's next

Two workstreams, not yet connected:

### Learning controller (parameter-golf)
Roadmap at `parameter-golf/LEARNING_CONTROLLER_ROADMAP.md`. Not started. Phases:
- Phase 0: Telemetry instrumentation in training loop
- Phase 1: Rule-based controller with hand-written strategies
- Phase 2: GRPO-trained controller on rollout comparisons
- Phase 3: Cross-run transfer and strategy accumulation

### SELPH optimization extension (this repo)
Roadmap at `selph_lang_neural_selph/OPTIMIZATION_EXTENSION_ROADMAP.md`. Through Phase 3. Next:
- **Generalization testing**: can the neural synthesizer generate correct structures for unseen tasks?
- **Phase 3b**: Decision trees as program synthesis (approximate spec satisfaction in the synthesizer)
- **Phase 4**: Hybrid models — synthesizer emits templates with `(param ?)` holes, optimizer fills them
- **Phase 5**: Model architecture descriptions as optimizable SELPH programs

### Convergence
When both workstreams mature, the learning controller's decision logic migrates to SELPH: telemetry via `__training__` builtins, HP adjustments as `__control__` actions, strategy library via SELPH cascade. The Python controller becomes a SELPH-based one with neural structure generation and symbolic validation.
