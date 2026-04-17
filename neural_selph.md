# Neural SELPH

Training a neural model to generate SELPH s-expressions as a method of externalizing knowledge from model weights into a symbolic, evaluable, updatable knowledge base.

## Core Idea

Standard autoregressive LM: `P(token | context)` — all knowledge baked into weights.

Neural SELPH: `P(expr | context)` where `expr` is either a literal token OR a SELPH s-expression that evaluates to a token. The model learns to delegate to the symbolic substrate whenever knowledge is better represented as a lookup/computation than as a memorized association.

```
Standard:       "the capital of spain is" → "madrid"
Neural SELPH:   "the capital of spain is" → (capital "spain") → evaluates to "madrid"
```

## Why

- **Updatable knowledge.** Facts live in a SELPH KB, not weights. Correct a fact, add new ones, swap domains — no retraining.
- **Compositional generalization.** If the model learns `(capital X)` and `(population X)` separately, it can compose `(population (capital "spain"))` without ever seeing that specific composition.
- **Verifiability.** Every factual claim has an inspectable symbolic derivation.
- **Smaller models.** The model only needs to learn the *schema* of knowledge (countries have capitals), not the facts themselves (which capital goes where).

## Architecture (Inference)

Straightforward. The model autoregressively emits tokens. Sometimes those tokens form s-expressions instead of literal text. When an s-expression closes (parens balance), evaluate it against the KB, splice the result back into the context, continue generating.

This is Toolformer-style inline evaluation with SELPH as the executor.

### Delimiter choice: `<tool_call>` / `</tool_call>`

We use Qwen's existing `<tool_call>` (token 248058) and `</tool_call>` (token 248059) as single-token delimiters. These are already in the vocabulary with dedicated IDs — no need for multi-token sequences.

**Why not `<selph>...</selph>`?** That tokenizes to 4 tokens each (`<`, `sel`, `ph`, `>`). At 200 training iterations, the model learned s-expression structure (81% parseable) but failed to reliably emit the multi-token opening delimiter (0% extracted). Single-token delimiters solve this.

**Inference strategy: prefix injection.** We append `<tool_call>` to the prompt so the model starts generating inside the expression. The model emits `</tool_call>` as a stop token when done. This doubled accuracy (0% → 6%) even at 200 iters.

### What lives where

| Component | Responsibility |
|---|---|
| Neural model (weights) | Syntax, grammar, style, reasoning patterns, NL→SELPH schema mapping |
| SELPH KB | Factual knowledge, domain-specific computations, anything with a ground truth source |
| SELPH evaluator | Deterministic execution of s-expressions against the KB |

## Architecture (Training) — The Hard Problem

No corpus of (natural language context, s-expression) pairs exists. Three approaches, in order of practicality:

### Approach 1: Synthetic from structured data

Import structured knowledge (e.g. Wikidata triples) into SELPH as definitions. Generate NL contexts mechanically (templates or existing LM). Pair them:

```
Wikidata triple: (capital, spain, madrid)
NL context:      "the capital of spain is"
Target:          (capital "spain")
```

Least interesting but actually produces training data. Start here.

### Approach 2: Distillation from a pretrained LM

Take a model that already predicts "madrid". Identify factual predictions (where the model draws on memorized knowledge). Replace those predictions with s-expressions that produce the same answer. Fine-tune the model to prefer the s-expression route.

Training signal: does the s-expression evaluate to the same token the original model would have predicted?

### Approach 3: RL with evaluation reward

The model generates freely — sometimes literal tokens, sometimes s-expressions. Reward = the evaluated output matches ground truth text. The model discovers on its own when s-expressions are useful.

No paired corpus needed, but high variance and slow convergence. Likely needs Approach 1 as a warm start.

## The Hole Question

We considered having the model output templates with holes and specs (mirroring SELPH's hole-fc architecture). The problem: for factual knowledge, I/O specs are circular — if the model knows enough to write the spec, it already knows the answer.

Holes work when the model contributes *structure* and the hole captures a *fact*:

```
"water boils at ___ degrees fahrenheit"
Model emits: (to-fahrenheit ?h1)
?h1 is a typed KB query: {type: temperature, substance: water, property: boiling-point}
```

The model knows the computation (fahrenheit conversion), the KB resolves the fact (100C). But the spec language for this is type/relation constraints, not I/O pairs — closer to a structured query than a synthesis spec.

For training, holes don't help with the core data problem. At inference, they could be useful for compositional queries where the model knows the shape but not every leaf value.

## Bootstrap Path

1. Build SELPH KB from structured data (Wikidata import)
2. Generate paired (NL context, s-expression) corpus mechanically
3. Fine-tune a small model on the paired corpus
4. Model proposes new s-expressions for facts not in the original KB
5. Validate by checking evaluated answers against ground truth
6. Expand KB with validated facts, retrain
7. KB and model grow together

## Dataset Plan

Starting with math/physics because SELPH already has a working curriculum for compositional arithmetic and physics formulas.

### Phase 1: MathQA (37K problems)

MathQA annotates word problems with nested operation programs using 58 operators:
```
multiply(divide(multiply(48, const_1000), const_3600), 9)
```

This converts mechanically to s-expressions:
```
(multiply (divide (multiply 48 1000) 3600) 9)
```

**Steps:**
1. Download MathQA from HuggingFace (`allenai/math_qa`)
2. Write converter: MathQA operation programs → SELPH s-expressions
3. Build (NL problem text, s-expression, numeric answer) triples
4. Validate: evaluate each s-expression in SELPH, confirm it produces the expected answer
5. Analyze: what SELPH primitives are needed? How many MathQA ops map to existing builtins vs need new ones?

### Phase 2: AI Feynman (120 equations)

The existing SELPH physics curriculum already solves many of these formulas. AI Feynman provides variable names, units, and formula descriptions — enough to synthesize NL prompts.

**Steps:**
1. Map AI Feynman equations to existing SELPH solutions where they overlap
2. Generate NL prompts from Feynman metadata (variable names, physical descriptions)
3. For equations not in the current curriculum, use SELPH synthesis to find solutions
4. Build (NL description, SELPH expression, numeric evaluation) triples

### Phase 3: PHYBench + SymPyBench (evaluation)

Hold out for evaluation. 500 NL physics problems (PHYBench) + 15K parameterized problems (SymPyBench) with SymPy expression tree answers. Test whether the model generalizes beyond training equations.

### Phase 4: LILA (134K problems, stretch)

Python program solutions across 23 math task types. Parse Python AST → s-expressions. Larger and more diverse than MathQA but needs a Python-to-SELPH transpiler.

### Dataset summary

| Dataset | Size | Input | Output | Conversion effort |
|---|---|---|---|---|
| **MathQA** | 37K | NL word problems | Nested op programs | Mechanical (rename ops) |
| **AI Feynman** | 120 | Tabular data + metadata | Symbolic formulas | Partial overlap with existing curriculum |
| **PHYBench** | 500 | NL physics problems | SymPy expressions | Parse SymPy → s-expr |
| **SymPyBench** | 15K | NL physics problems | SymPy code | Parse SymPy → s-expr |
| **LILA** | 134K | NL math problems | Python programs | AST transpiler needed |

## Key Design Decisions (Open)

- **Escape mechanism.** RESOLVED — using `<tool_call>` / `</tool_call>` (single tokens in Qwen vocab). With prefix injection at inference.
- **Tokenization.** S-expressions are trivially tokenizable (parens, atoms, strings). Could use character-level, BPE over s-expr tokens, or treat each SELPH primitive as a single token (like ToolkenGPT).
- **Domain isolation.** Probably want domain-specific KB pools (geography, physics, chemistry) rather than one monolithic KB. Matches the M-chain lesson from ARC work.
- **When NOT to delegate.** "The cat sat on the ___" is distributional prediction, not factual lookup. The model needs to learn the boundary.
- **KB scale.** Wikidata has ~100M triples. How much can SELPH's evaluator handle efficiently?

## Prior Work

Closest existing systems:

| System | Relationship |
|---|---|
| **ToolkenGPT** (Hao et al., NeurIPS 2024) | Token-level tool invocation via learned embeddings. Closest mechanism. |
| **Toolformer** (Schick et al., NeurIPS 2023) | Inline API calls during generation, self-supervised via perplexity reduction. Closest training signal. |
| **PAL / Program of Thoughts** (ICML/TMLR 2023) | LM generates programs instead of answers. Same delegation principle, but at problem level not token level. |
| **DreamCoder** (Ellis et al., PLDI 2021) | Wake-sleep loop: synthesize programs, compress into library, retrain neural guide. Closest to the bootstrap path. |
| **LILO** (Grand et al., ICLR 2024) | LLM synthesis + Stitch compression + AutoDoc. Closes the neural/symbolic library loop. |
| **"From Tool Calling to Symbolic Thinking"** (de la Torre, 2025) | Proposes LLM + persistent Lisp REPL with s-expressions. Same architectural vision, no implementation. |
| **LLM-SR** (Shojaee et al., ICLR 2025 Oral) | LM proposes symbolic equation templates with constant holes. Parallel to SELPH's constant-hole synthesis. |
| **CodeIt** (Butt et al., ICML 2024) | Self-play program synthesis for ARC. Hindsight relabeling as training signal. |
| **REALM / RAG** (ICML/NeurIPS 2020) | Knowledge externalization via retrieval. Same motivation, but text retrieval not symbolic computation. |

The specific gap: no existing system combines (1) token-level delegation, (2) unified DSL, (3) KB evaluation as training signal, and (4) growing library that the model learns to compose.

## Relationship to ARC Work

Separate direction. The ARC work's bottleneck is form coverage (computational structure), not factual knowledge. Neural SELPH targets factual knowledge externalization. However, the infrastructure overlaps:

- The SELPH evaluator is shared
- The hole-fc architecture could serve both
- A neural model trained on SELPH could eventually propose decomposers for ARC too (bridging back)

## Experiment Log

### Exp 1: MathQA → SELPH, Qwen3.5-0.8B-Base, LoRA (2026-04-17)

**Setup:** 29,445 train / 4,425 valid / 2,948 test examples. MathQA operation programs converted mechanically to SELPH s-expressions. LoRA fine-tuning via mlx-lm, mask-prompt, lr=1e-5, batch=4, 16 layers.

**Delimiter experiment (200 iters):**

| Method | Parseable | Correct Value | Notes |
|---|---|---|---|
| `<selph>` (4-token), regex extract | 0% | 0% | Model skips multi-token delimiter |
| `<selph>`, balanced-paren extract | 81% | 3% | Structure learned, semantics weak |
| `<tool_call>` (1-token), regex extract | 2% | 0% | Opening token not emitted |
| `<tool_call>`, prefix injection | 86% | 6% | 2x accuracy, 2x faster |

**Key findings:**
- Model learns s-expression syntax fast (81% parseable at 200 iters)
- Multi-token delimiters fail; single-token + prefix injection works
- Bottleneck at 200 iters is semantic (wrong ops/numbers), not syntactic
- Constrained decoding (outlines + grammar) gives 100% parseable but doesn't help semantics
- `</tool_call>` closing token IS learned; opening token needs prefix injection

**5000-iter run results:**

| Metric | 200 iters | 5000 iters |
|---|---|---|
| Val loss | 0.500 | **0.246** |
| Test loss | — | **0.277** |
| Parseable | 86% | **98%** |
| Correct value | 6% | **10%** |
| Exact match | 2% | **5%** |
| Speed | 0.47s/ex | **0.37s/ex** |

Key: 98% parseable = the model has nearly mastered s-expression syntax. 0 eval errors on parsed expressions. The 10% correct value bottleneck is semantic (word problem understanding + operator selection), not syntactic. This is a 0.8B base model with LoRA — the SELPH generation mechanism works.

### Exp 2: Geography KB — Function Signatures in Prompt (2026-04-17)

**Architecture decision:** The model reads available functions from the prompt, not from memorized training. This means when SELPH's library grows (new decomposers, new domains), the model learns to use new functions without retraining — they just appear in the signature block.

**Prompt format:**
```
Functions: capital(country) → city, population(entity) → number, continent(country) → continent, ...

Q: What is the population of the capital of Spain?
A: <tool_call>(population (capital "Spain"))</tool_call>
```

**Dataset:** 2,946 examples from Wikidata (222 countries), 4 composition depths:
- Depth 1 (1,466): Direct lookups — `(capital "Spain")`
- Depth 2 (1,242): One composition — `(population (capital "Brazil"))`, `(divide (population X) (area X))`
- Depth 3 (231): Two compositions — `(continent (country-of (capital "Malaysia")))`
- Depth 4 (7): Three compositions — `(population (capital (largest-by-population "South America")))`

**Generalization test:** Train on depth 1+2, evaluate on depth 3+4. Also: add unseen function to signature block at inference, test if model composes with it.

**Connection to SELPH:** `env-function-names` and `env-lookup` builtins already support runtime library introspection. The model's signature block is the NL equivalent — a serialized view of what SELPH currently knows. As the M-chain grows, the signature block grows, and the model can compose over new capabilities without retraining. New `apropos` and `apropos-by-type` builtins added to support dynamic library search.

**Geo KB results (1k iters from MathQA 5k checkpoint):**

| Set | Parseable | Exact match | Notes |
|---|---|---|---|
| Valid (depth 1+2) | 100% | **100%** | Perfect on train distribution |
| Test (depth 3+4) | 100% | **0%** | Compositional generalization fails |

**Critical finding: the model takes shortcuts.** Instead of `(continent (country-of (capital "Germany")))` it produces `(continent "Germany")` — which gives the *same correct answer* via a simpler path. The model optimizes for output correctness, not compositional structure.

This reveals a training data design principle: **composition tests must use examples where the shortcut gives the wrong answer.** `(population (capital "Spain"))` is a good test because `population("Spain") ≠ population("Madrid")` — the intermediate step changes the result. `(continent (country-of (capital "Germany")))` is a bad test because `continent("Germany") = continent("Berlin")` — the indirection is redundant.

Next: regenerate test set with "necessary composition" examples only.

### Exp 3: RL via Rejection Sampling with Grounding Reward (2026-04-17)

**Reward function:** Expression must (1) parse, (2) contain only string literals found in the prompt (no smuggled knowledge), (3) evaluate to the correct answer in SELPH. Simplicity bonus for fewer tokens.

**Key principle:** Model contributes STRUCTURE, SELPH contributes KNOWLEDGE. `"Europe"` as a literal → rejected (not in prompt). `(continent "Germany")` → accepted ("Germany" is in prompt, SELPH resolves to "Europe").

**Round 1 results (2946 prompts × 4 samples, temp 0.7):**
- Accepted: 2,723/2,946 (92.4%)
- Rejected: 223 (7.6%) — failed grounding or wrong answer

**Compositional test set (depth 3+4):**

| Metric | Pre-RL | Post-RL Round 1 |
|---|---|---|
| Parseable | 100% | 100% |
| Correct answer | 87.0% | **88.2%** |

Small lift. The grounding constraint works (prevents literal answer smuggling) but most shortcuts are grounded — `(continent "Germany")` is valid because "Germany" appears in the prompt. The remaining 12% failures are `largest-by-population` queries where the model hasn't learned the aggregation function.

**Insight:** The RL reward correctly shapes behavior but the training distribution needs more examples where composition is *required by the grounding constraint* — queries where no shortcut exists that uses only prompt constants.

## Long-Term Future Directions

### Holes as Subagents

SELPH holes become subagent invocations rather than pure synthesis targets. The surrounding s-expression provides structural context; the hole spec can range from precise I/O pairs to vague natural language:

```
;; Precise — normal SELPH synthesis
(? :examples ((3 → 9) (4 → 16)))

;; Semi-formal — constrained NL
(? :type Country :hint "largest by area in South America")

;; Vague — pure NL, needs an LLM
(? "which country is known for carnival and the Amazon")

;; Meta — the hole is about problem structure itself
(? "what operation relates population to GDP here?")
```

Dispatch routes precise specs to SELPH synthesis, vague specs to an LLM agent. The outer SELPH expression acts as a **contract** — whatever the subagent returns must type-check and satisfy the enclosing computation. The LLM never touches the arithmetic or lookup; it only resolves the parts that require world knowledge or analogy.

### Partial Evaluation Feedback Loop

Instead of the LLM generating a complete expression in one shot:

1. LLM generates a partial expression with holes: `(/ (population "spain") ?)`
2. SELPH evaluates what it can, reports: `population = 47,420,000, hole expects a number`
3. LLM fills the hole with grounded context: `(area "spain")`

This is hole-fc applied to LLM-guided synthesis. The LLM plays the role of the decomposer, SELPH plays the role of the sub-synthesizer and verifier. Each round of partial evaluation gives the LLM strictly more information than it had before, scoped to exactly the part it needs to resolve.

### Reasoning Enhancement via Offloading LLM Weaknesses

The KB primitives worth exposing aren't about knowledge — they're about **operations LLMs fail at**:

| LLM weakness | SELPH primitive | Example |
|---|---|---|
| Arithmetic | Numeric ops | `(/ (population "spain") (area "spain"))` |
| Constraint tracking | `filter`, `and`/`or` | `(filter countries (lambda c (and (> (pop c) 10M) (< (area c) 500K))))` |
| Ordering/ranking | `sort-by`, `min-by` | `(sort-by (lambda c (/ (pop c) (area c))) (countries-in "EU"))` |
| Set operations | `intersection`, `union` | `(intersection (members "NATO") (members "EU"))` |
| Counting/aggregation | `count`, `group-by`, `mean` | `(mean (map gdp-per-capita (countries-in "EU")))` |
| Faithful hypotheticals | `let`-binding overrides | `(let ((area (* (area "spain") 0.5))) (/ (pop "spain") area))` |
| Multi-hop lookup | Composition | `(population (capital (largest-by-area "South America")))` |

The LLM's job reduces to *translating intent into structure*. SELPH handles the mechanical computation that LLMs hallucinate on.

### Compiled Reasoning: LLM Discoveries → SELPH Decomposers

If the LLM discovers that "density questions" always have the shape `(/ (quantity X) (area X))`, it registers that as a SELPH decomposer. Future density questions are handled by SELPH directly — the LLM's reasoning gets *compiled into* the knowledge base. The system gets smarter without retraining the LLM.

This is the M-chain idea applied to natural language reasoning: learned problem structure, expressed formally, verified empirically. The bootstrap path (§Bootstrap Path above) already describes KB growth; this extends it to reasoning patterns, not just facts.

### Bidirectional Flow for Debugging

For domains like debugging, the subagent and SELPH exchange structured context:

```
(diagnose
  (? "describe the symptom")
  (diff
    (expected-output (? "what should this code do?"))
    (actual-output (run-fn problematic-function test-input))))
```

SELPH computes the actual output and the diff. The subagent only needs to describe intent — SELPH handles the mechanical comparison. Diagnosis is grounded in real execution, not the LLM's guess about what the code does. The same pattern applies to riddles (LLM proposes structure, SELPH checks constraints) and any domain where the reasoning shape is formalizable but the content requires world knowledge.

### Epistemic Tagging (Structured Natural Language)

A lightweight intermediate between pure NL reasoning and full formalization: tag each reasoning step with its epistemic status:

- **observation** — grounded in SELPH evaluation
- **hypothesis** — proposed by the LLM, untested
- **deduction** — follows from previous steps
- **assumption** — taken as given, may be wrong

A structural checker validates coherence ("you claimed to deduce X but your premises don't support it") without requiring full formalization of the content. This gives verifiability where it's cheap while keeping the expressive power of natural language for the hard parts.
