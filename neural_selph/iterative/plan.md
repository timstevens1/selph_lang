# Iterative S-Expression Refinement with MultiScale SSM

## What This Is

A tiny neural policy that iteratively refines SELPH s-expressions using execution feedback. Connects three projects:

1. **parameter-golf** — MultiScale SSM backbone (S4 banks + temporal routing + SwiGLU FFN)
2. **recursive_dsl_synthesis_design.md** — architecture specification (iterative refinement, query channel, latent state, halt signal)
3. **neural_selph** — SELPH evaluator as the execution oracle

Instead of one-shot generation (where a 0.8B model struggles with composition), a 378K-param model builds expressions incrementally: `_HOLE_` → `(capital _HOLE_)` → `(capital "Spain")` → CORRECT.

## Results (2026-04-20)

### Current: GRPO-trained (outcome-based RL)

378K params, 15 min GRPO training (20 epochs) after SFT, MLX backend. Group size 4, temperature 0.7.

| Depth | SFT Baseline | After GRPO | Eval subset |
|-------|-------------|------------|-------------|
| 1 | 90.2% | **96.1%** | 51 |
| 2 | 53.8% | **59.0%** | 39 |
| 3 | 80.0% | 70.0% | 10 |
| **ALL** | **75.0%** | **79.0%** | **100** |

+4pp overall via GRPO. Query usage decreased from ~52% → ~35% during training (model learned selectivity). With-queries and without-queries eval are still identical — the query channel improves overall policy but hasn't yet shown differential value at greedy eval. Depth 3 regression is likely noise (n=10).

### SFT Baseline + Thinking-in-SELPH

378K params, 6 min training on M4 Max, MLX backend. 892-token vocabulary with query tokens.

| Depth | No Query | With Query | Examples |
|-------|----------|------------|----------|
| 1 | 85% | 85% | 100 |
| 2 | 56% | 56% | 100 |
| 3 | 90% | 91% | 100 |
| 4 | 71% | 71% | 7 |
| **ALL** | **76.9%** | **77.2%** | **307** |

Val loss: 0.58. The query channel adds +1 task — mechanically working but not yet trained to be useful. GRPO teaches the model *when* and *what* to query (+4pp, see above).

### Original Baseline (pre-query tokens)

378K params, 885-token vocab, no query channel.

| Depth | Accuracy | Examples |
|-------|----------|----------|
| 1 | 82% | 100 |
| 2 | 73% | 100 |
| 3 | 86% | 100 |
| 4 | 14% | 7 |
| **ALL** | **78.8%** | **307** |

### v2 Architecture (archived)

Two-model design (TokenModel + LatentModel), 458K params. Explored and abandoned.

| Config | Accuracy | Notes |
|--------|----------|-------|
| Phase 1 (independent steps) | 42.0% | Latent not passed across steps |
| Phase 2 (unrolled, 5 epochs) | 56.7% | Latent passing helps +14.7pp |
| Phase 2, no query | 56.7% | Query channel adds nothing |
| Token model only (no latent) | 0.3% | Collapses without conditioning |

**Conclusion:** Separate latent model adds complexity without beating the simple baseline. The longer v2 sequence format dilutes the training signal. Archived in `archive_v2/`.

### One-shot baseline

0% — the model was trained on traces (multi-step), not one-shot completion. Validates that the iterative mechanism is load-bearing.

## Architecture

### Current Design: Baseline + Query Channel

Single MultiScaleFFNLM model, autoregressive next-token prediction. The model can optionally "think in SELPH" by generating query expressions that get evaluated before producing the edit.

**Input format:**
```
[BOS SEP_TASK Q_capital Canada SEP_EXPR (capital _HOLE_) SEP_FEEDBACK INCOMPLETE]
```

**Output (without thinking):**
```
[SEP_EDIT (capital "Canada") EOS]
```

**Output (with thinking):**
```
[<query>(capital "Canada")</query><q_out>Ottawa</q_out> SEP_EDIT (capital "Canada") EOS]
```

The model generates a single autoregressive stream. When it emits `<query>`, generation continues until `</query>`, the s-expression is evaluated against the KB, the result is injected as `<q_out>...</q_out>`, and generation resumes. The model can issue multiple queries before committing with `SEP_EDIT`.

### Backbone

Imported directly from parameter-golf's MLX implementation:
- `MultiScaleSSMBlock`: K fixed-param S4 SSMs at different timescales + shift buffers + router SSM + softmax gating
- `BinaryGatedFFN`: SwiGLU with BinaryLinear projections
- `MultiScaleLayer`: SSM block + FFN with residual connections and learned scales

Config: 3 layers, dim=96, feat_dim=48, state_dim=8, num_scales=3, shifts=(0,1,2,4).

### Inference

Uses FFT-based `backbone()` for autoregressive generation (O(L²) total). The `prefill()` + `step()` O(1) path exists in parameter-golf but is not used — at sequence lengths of 15-30 tokens, the O(L²) cost is negligible and avoids complexity.

## Training Data

### Trace Generation (`generate_traces.py`)

From 2,946 geo KB examples (222 countries, depths 1-4), generates 12,608 refinement steps across 3,879 traces.

Two trace types:
1. **Top-down decomposition**: `_HOLE_` → `(func _HOLE_)` → `(func arg)` → CORRECT. BFS expansion, filling the leftmost hole at each step.
2. **Wrong-start correction** (depth ≥ 2): Start from a wrong expression (wrong function or missing composition), get WRONG feedback, produce the correct expression.

20% of steps include synthesized queries — evaluable sub-expressions of the target expression. 3,086 total query pairs.

### Tokenizer (`sexpr_tokenizer.py`)

892-token vocabulary:
- S-expression syntax: `(`, `)`, `_HOLE_`, `NO_OP`
- 10 geo functions: `capital`, `population`, `continent`, etc.
- ~600 entity names from the KB (countries, cities, currencies, etc.)
- 13 question type tokens: `Q_capital`, `Q_population`, etc.
- Trace structure: `SEP_TASK`, `SEP_EXPR`, `SEP_FEEDBACK`, `SEP_EDIT`
- Query tokens: `<query>`, `</query>`, `<q_out>`, `</q_out>`
- Digits for numeric feedback values

## Known Issues (Resolved)

### parse_sexpr Infinite Loop (FIXED)

**Root cause of all previous eval hangs.** When the model generates malformed s-expressions with stray `)` characters, `parse_sexpr` would enter an infinite loop. The `else` (atom) branch matched `)` but the inner while loop's stop-chars included `)`, so `j` never advanced past `i`.

**Fix:** Added `elif c == ")": i += 1` to skip stray close parens.

**Lesson:** This was misdiagnosed for hours as MLX graph accumulation, subprocess batching issues, or model architecture problems. Always check if generated output can cause downstream parser hangs.

### MLX Lazy Evaluation

MLX's lazy evaluation model means computation graphs can accumulate across calls. This is manageable with `mx.eval()` at generation boundaries. The O(L²) FFT generation path works cleanly for 300+ examples without any special cleanup.

## File Structure

```
neural_selph/iterative/
├── plan.md                 # This file
├── sexpr_tokenizer.py      # 892-token vocab with query tokens
├── generate_traces.py      # Traces with synthesized query annotations
├── train_mlx.py            # Training: MultiScaleFFNLM on query-augmented traces
├── eval_mlx.py             # Eval with optional interactive query evaluation
├── grpo_mlx.py             # GRPO training: outcome-based RL for query channel
└── archive_v2/             # Archived two-model design (TokenModel + LatentModel)
    ├── model_v2.py
    ├── train_v2.py
    ├── eval_v2.py
    └── generate_traces_v2.py
```

Data files (in `neural_selph/data/`):
- `geo_kb.json` — 222 countries with capital, population, continent, etc.
- `geo_train.json` — 2,946 compositional queries at depths 1-4
- `geo_traces.json` — 12,608 individual refinement steps (with query annotations)
- `geo_traces_full.json` — 3,879 full multi-step traces
- `iterative_model_mlx.npz` — SFT model weights (378K params)
- `iterative_model_grpo.npz` — GRPO-trained model weights (378K params)

## Training Commands

```bash
# Generate traces (includes query annotations)
python3 -m neural_selph.iterative.generate_traces

# Train (edit + query traces, ~6 min)
python3 -m neural_selph.iterative.train_mlx --epochs 200 --dim 96 --feat-dim 48 --num-layers 3

# Eval without queries
python3 -u -m neural_selph.iterative.eval_mlx --dim 96 --feat-dim 48 --num-layers 3

# Eval with interactive query evaluation
# (pass --use-queries flag once eval_mlx.py main() is updated)
```

## What We Learned

1. **Iterative refinement works at tiny scale.** 378K params achieves 77-79% on compositional queries that a 0.8B one-shot model struggled with. The execution feedback loop is the key mechanism.

2. **Question context is essential.** Without `Q_type` + entity tokens, the model mode-collapses to a single expression regardless of the question.

3. **Simplicity wins.** The baseline (single autoregressive model, no latent, no separate heads) outperforms the multi-model v2 architecture (78.8% vs 56.7%). The extra complexity dilutes the training signal on small data.

4. **Latent passing helps within its architecture.** Phase 2 unrolled training improved v2 from 42% to 56.7% — but the architecture itself is worse than the baseline, so the latent's benefit is moot.

5. **Query channel needs RL, not supervision.** Teacher-forced queries from synthesized traces add nothing (+0.3pp). GRPO adds +4pp overall but the query channel hasn't yet shown differential value at greedy eval — the improvement is in the edit policy itself.

6. **Always check parsers on generated output.** The parse_sexpr infinite loop on stray `)` was misdiagnosed as MLX graph accumulation for hours. Generated output is adversarial input to downstream parsers.

7. **Teacher forcing ≠ autoregressive.** Single-pass greedy decode (argmax at each position) produces garbage. Autoregressive generation is required at eval time.

8. **GRPO improves overall policy, not just queries.** 20 epochs of GRPO on 128 tasks/epoch with G=4 improved eval from 75% → 79%. The model became more selective about queries (52% → 35% usage) but with-queries ≈ without-queries at eval. The RL signal improves the edit decisions more than the query decisions on this small KB. Hypothesis: queries become differentially useful when the function space is large enough that the model can't memorize all mappings.

## Next Steps

### Immediate

1. ~~**GRPO for query channel**~~ **Done.** +4pp (75% → 79%). Infrastructure in `grpo_mlx.py`. Query channel improves overall policy but hasn't shown differential value on the small geo KB.

2. **Broad Wikidata KB + SELPH Rust backend**: Move from 222-country geo KB to full Wikidata dump. Open-ended function set. Use the Rust SELPH evaluator (with new `apropos` builtin) as the execution oracle. The model must use `apropos` to discover relevant functions before composing them.

### Near-term

3. **`apropos`-driven function discovery**: `(apropos "boil")` → `(boiling-point)`. Already implemented in Rust SELPH (substring + type filter + namespace search). Key mechanism for open-world KB tasks where the model can't memorize all functions.

4. **Scale model to 1-5M params**: Larger KB needs more capacity. MultiScale SSM backbone scales well. BPE tokenizer for open vocabulary.

5. **Depth-2 gap analysis**: Depth 2 is the weakest (59% after GRPO) despite being simpler than depth 3. Investigate data distribution vs model limitation.

### Longer-term

6. **ARC-AGI integration**: Use the iterative refinement policy as a decomposer proposer for SELPH's M-chain. The model proposes grid transformation templates with holes; SELPH fills them via sub-synthesis.

7. **Domain-tagged apropos**: Beyond substring/type filtering, tag functions by domain (geography, science, math) so `apropos` can narrow by domain. The model learns to identify the domain first, then search within it.
