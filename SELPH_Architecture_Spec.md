# SELPH: Symbolic Evaluation Language for Programmable Hierarchies

## A Formal Architecture for Homoiconic Neural Program Synthesis

**Version 0.1 — Working Draft**

---

## 1. Motivation

Current LLM-based systems maintain a hard separation between the model (weights in embedding space), the output (tokens in discrete space), the execution environment (external tools/APIs), and the training regime (loss functions and optimizers defined in Python). This separation prevents the system from reasoning about, modifying, or composing any of these components using the same mechanisms it uses for everything else.

We propose a unified architecture in which programs, data, model architectures, execution traces, specifications, critiques, and training signals are all expressed in a single homoiconic language. The system generates, evaluates, criticizes, and learns from S-expressions end-to-end, collapsing the boundaries between "the thing being built," "the thing doing the building," and "the thing evaluating the result."

---

## 2. Core Language: SELPH

### 2.1 Design Principles

1. **Homoiconicity.** Code and data share the same representation (S-expressions). A program is a list. A list is data. Transformation of programs is list manipulation.

2. **Tensor-native primitives.** The base vocabulary includes differentiable tensor operations as first-class atoms, not library calls.

3. **Minimal grammar.** The CFG must be simple enough for efficient constrained decoding with near-zero overhead at generation time.

4. **Typed.** A shape/type system constrains generation to reject dimensionally invalid programs at decode time.

5. **Abstractable.** First-class macro/abstraction support allows the system to define new primitives from discovered patterns.

### 2.2 Grammar (EBNF)

```
program     ::= atom | list
list        ::= '(' element* ')'
element     ::= program | spec
atom        ::= symbol | number | string | tensor-literal
symbol      ::= [a-zA-Z_][a-zA-Z0-9_-]*
number      ::= [0-9]+ ('.' [0-9]+)?
string      ::= '"' [^"]* '"'
tensor-lit  ::= '#T' shape dtype

spec        ::= '(:spec' spec-body ')'
spec-body   ::= ':type' type-expr
              | ':shape' shape-expr
              | ':goal' goal-expr
              | ':constraints' constraint-list
              | ':input' program
              | spec-body spec-body

goal-expr   ::= '(:examples' example-list ')'        ; Level 0
              | '(:pattern' program ')'                ; Level 1
              | '(:satisfy' program ')'                ; Level 2 (lambda predicate)
              | '(:transform' program ':by' program ')'; Level 3
              | '(:intent' string ')'                  ; Level 4+
              | '(:all' goal-expr+ ')'                 ; Composition

type-expr   ::= 'scalar' | 'vector' | 'matrix' | 'tensor'
              | 'symbol' | 'string' | 'bool'
              | '(-> ' type-expr+ type-expr ')'
              | '(list ' type-expr ')'

shape-expr  ::= '[' dim-expr (',' dim-expr)* ']'
dim-expr    ::= number | symbol | '?'
```

### 2.3 Primitive Vocabulary

The language partitions its atoms into four categories:

**Tensor operations (differentiable):**
`matmul`, `add`, `subtract`, `hadamard`, `outer-product`, `transpose`, `reshape`, `slice`, `concat`, `conv`, `softmax`, `relu`, `gelu`, `sigmoid`, `tanh`, `layer-norm`, `embed`, `linear`, `dropout`

**Structural operations (non-differentiable, discrete):**
`if`, `let`, `lambda`, `defmacro`, `quote`, `eval`, `apply`, `map`, `reduce`, `filter`, `compose`, `pipe`

**Delegation operations:**
`submodel`, `subagent`, `delegate`

**Meta operations:**
`spec`, `substitute`, `rewrite`, `trace`

### 2.4 Example Programs

A simple neural layer:
```lisp
(let ((h (relu (add (matmul W x) b))))
  (dropout h 0.1))
```

An attention head:
```lisp
(defmacro attention (Q K V)
  (matmul (softmax (matmul Q (transpose K))) V))
```

An agentic decomposition (Stage 4+, natural language goals):
```lisp
(let ((country (subagent
        (:spec :type string
               :goal (:intent "identify the largest country by area"))))
      (capital (subagent
        (:spec :type string
               :goal (:intent "find the capital of")
               :input country))))
  capital)
```

The same task at Stage 2 (no natural language needed):
```lisp
(let ((country (subagent
        (:spec :type string
               :goal (:satisfy (lambda (out)
                 (= out (max-by area countries)))))))
      (capital (subagent
        (:spec :type string
               :goal (:satisfy (lambda (out)
                 (= out (lookup capital-of country))))))))
  capital)
```

---

## 3. Type and Shape System

### 3.1 Purpose

The type system serves two roles simultaneously:

1. **At generation time:** Integrated with constrained decoding to mask tokens that would produce ill-typed programs. At any point during left-to-right generation, the valid token set is restricted to symbols whose types are compatible with the current position in the AST.

2. **At evaluation time:** Shape mismatches, type errors, and spec violations produce structured error values that feed the critic.

### 3.2 Typing Rules

Every node in the program tree carries a type signature:

```
matmul  : (-> matrix matrix matrix)
relu    : (-> tensor tensor)          ; shape-preserving
softmax : (-> tensor tensor)          ; shape-preserving
embed   : (-> int [vocab, dim] vector)
linear  : (-> vector [in, out] vector)
concat  : (-> tensor tensor int tensor) ; along specified axis
```

Shape inference propagates constraints through the tree. For `(matmul A B)`, if `A : [m, k]` and `B : [k, n]`, the result is typed `[m, n]`. If the inner dimensions don't match, generation is constrained to reject `B` or the enclosing expression is flagged.

### 3.3 Spec Types

Specs are first-class typed values. A spec acts as a **contract** between a parent node and the child that will fill its slot.

```
(:spec :type T :shape S :goal G :constraints C)
```

The child's evaluated output must satisfy the spec's type, shape, and constraint fields. The `:goal` field defines what the node should accomplish — and critically, **the goal representation is staged with the curriculum** so that specs never exceed the model's current comprehension level.

### 3.4 Staged Goal Representation

The `:goal` field is not a fixed-format string. It is a SELPH expression whose complexity matches the current training stage. The principle: **a spec must be expressible in the language the model has already mastered.**

**Level 0 — Goals as input/output examples.**
No natural language. The goal is a set of concrete examples the evaluator can check mechanically.

```lisp
(:spec :type string
       :goal (:examples (("a" -> "b") ("b" -> "c") ("y" -> "z"))))
```

Verification: execute the program on each input, compare output to expected. Fully mechanical. No judge needed.

**Level 1 — Goals as pattern constraints.**
Once the model has learned basic string/word operations, goals can use structural patterns expressed in primitives the model already knows.

```lisp
(:spec :type string
       :goal (:pattern (word :length 5 :starts-with "h")))
```

Verification: check the output against the pattern. Still fully mechanical.

**Level 2 — Goals as executable predicates.**
Once the model has logic and arithmetic, goals become lambda expressions — programs that test the output.

```lisp
(:spec :type number
       :goal (:satisfy (lambda (out in) (and (> out in) (even out)))))
```

Verification: evaluate the predicate on the output. Mechanical. The goal is itself a program in the same language.

**Level 3 — Goals as symbolic relations.**
Once the model can compose multi-step reasoning, goals can describe transformations and relationships between structured values.

```lisp
(:spec :type sentence
       :goal (:transform input :by reverse-word-order :preserving meaning))
```

Verification: partially mechanical (structure checks), partially model-judged (meaning preservation). This is where the model begins judging its own outputs, but only for properties it has already learned to evaluate.

**Level 4+ — Goals as natural language.**
Only after the model can produce and comprehend sentences does the goal become a prose description.

```lisp
(:spec :type string
       :goal (:intent "summarize the document focusing on financial risks"))
```

Verification: LLM-as-judge. The model evaluates its own output against the natural language intent. This is only possible because the model is now capable enough to be a judge — a capability it developed at earlier stages.

### 3.5 Goal Verification Rules

The verification method follows from the goal's level:

| Goal level | Form | Verification | Judge required? |
|-----------|------|--------------|-----------------|
| 0 | Input/output examples | Execute and compare | No |
| 1 | Pattern constraints | Match against pattern | No |
| 2 | Executable predicates | Evaluate the lambda | No |
| 3 | Symbolic relations | Partial mechanical + model | Partial |
| 4+ | Natural language intent | Model self-evaluation | Yes |

The system bootstraps its own evaluation capability alongside its generation capability. At no stage does verification require abilities the model hasn't already demonstrated. By the time the model needs to judge natural language intent, it has already learned language through Stages 0–3.

### 3.6 Goals Are Programs

Because goals at every level are SELPH expressions (not a separate metalanguage), they participate in the same homoiconic properties as everything else:

- Goals can be composed: `(:goal (:all pred1 pred2 pred3))`
- Goals can be transformed: a parent can weaken or strengthen a child's goal
- Goals can be learned: Loop 3 can extract common goal patterns into reusable goal templates
- Goals can be generated: a conductor node generates its children's goals in the same language it uses for everything else

The `:intent` string form used at Level 4+ is syntactic sugar — the model could equivalently express the same goal as a predicate, if the predicate were complex enough. Natural language goals are just the most compressed representation for complex specifications that would be unwieldy as formal predicates.

---

## 4. Evaluation Model

### 4.1 Evaluation Order

Programs are evaluated **left-to-right, depth-first**, matching standard Lisp evaluation semantics. This evaluation order is also the **generation order**: the model generates the tree in the same sequence it would be executed.

For a node `(f a b c)`:

1. Evaluate `a` (recursively, depth-first)
2. Evaluate `b` (with `a`'s result now available as context)
3. Evaluate `c` (with `a` and `b`'s results available)
4. Apply `f` to the evaluated results

### 4.2 The Two Evaluators

The system maintains two parallel evaluation pathways:

**Differentiable evaluator.** Operates on tensor operations in embedding space. Executes the forward pass of any sub-program composed of differentiable primitives. Supports reverse-mode automatic differentiation over the program's AST. Gradients flow through the tree structure.

**Symbolic evaluator.** Operates on structural and meta operations in token space. Handles `if`, `let`, `defmacro`, `subagent`, `critique`, `rewrite`. Produces discrete results. Not differentiable.

The boundary between the two evaluators is the **token boundary**: the moment a continuous distribution over vocabulary items collapses to a discrete symbol selection.

### 4.3 Continuous Relaxation (Training Mode)

During training, discrete operation choices can optionally be relaxed via Gumbel-softmax:

```
result = Σ_i  softmax(logit_i / τ) · eval(op_i, args)
```

where `op_i` ranges over type-compatible operations at a given node. This allows gradient signal to flow through operation *selection*, not just operation *parameters*. The temperature τ anneals toward 0 over training, hardening soft mixtures into discrete choices.

This relaxation is only tractable because the typed vocabulary at each node is small (typically 5–20 compatible operations, not the full vocabulary).

---

## 5. Tree-Structured Context and Generation

### 5.1 Context Window Construction

When generating any node in the program tree, the model receives a context window constructed from:

1. **Root-to-current path.** The chain of ancestor S-expressions from the root down to the current node's parent. Each ancestor is represented as its operator and spec, not its full subtree.

2. **Sibling specs and results.** For each sibling to the left of the current node: its original spec (intent annotation left by the parent) and its evaluated output. For siblings to the right: their specs only (not yet evaluated).

3. **Current spec.** The spec/intent annotation left by the parent for the slot this node will fill.

This gives the model O(depth + branching_factor × depth) tokens of context rather than O(total_nodes), enabling deep trees without context overflow.

### 5.2 Example Context

For a tree:
```lisp
(answer-question
  (find-entity (:spec :type country :goal (:intent "largest by area")))   ; → "Russia"
  (lookup-attr (:spec :type city :goal (:intent "capital of") :input left)))  ; generating this
```

The model generating `lookup-attr`'s implementation sees:

```
[PATH]    answer-question → _
[LEFT]    (:spec :type country :goal (:intent "largest by area")) → "Russia"
[SELF]    (:spec :type city :goal (:intent "capital of") :input "Russia")
```

It does **not** see the internal subtree of `find-entity` — only its spec and result.

### 5.3 Generation as Tree-MDP

Each node generation is a decision point in a tree-structured Markov Decision Process:

- **State:** The context window (path + sibling specs/results + current spec)
- **Action:** The generated S-expression for this node
- **Transition:** Evaluation of the generated expression, producing a result
- **Reward:** Spec satisfaction (see §6)

Because each node has a well-defined state, action, and reward, per-node credit assignment is direct — no need to decompose a global reward across an opaque sequence.

---

## 6. Training Regime

### 6.1 Three Nested Loops

**Loop 1 — Parameter optimization (innermost, milliseconds).**
Standard gradient descent through the differentiable evaluator. For a fixed program structure, adjust continuous parameters (weight matrices, bias vectors, embedding tables) to minimize task loss. This is conventional backpropagation operating on the subset of the program tree that consists of differentiable operations.

**Loop 2 — Structural evaluation and retry (middle, seconds to minutes).**
There is no separate critic model. The evaluation loop is simple:

1. The node generates an S-expression.
2. The evaluator runs it and produces an output.
3. The output is checked against the node's spec:
   - **Type/shape match:** Mechanically verified. Binary pass/fail.
   - **Constraint satisfaction:** Mechanically verified where constraints are formal. Binary pass/fail.
   - **Intent satisfaction:** The same base model, given the spec's `:intent`, the evaluated output, and the parent context, judges whether the output fulfills the intent (LLM-as-judge). Produces a scalar score.
4. If the node fails (or scores below a threshold), the model may **retry**: it receives its own output as additional context and generates a replacement S-expression for the same slot. The retry sees:
   - The original spec
   - The previous attempt's S-expression
   - The previous attempt's evaluated output
   - The judge's score/explanation

This is just the model critiquing itself — no separate critic architecture, no separate training loop for the critic. The judge call is a forward pass of the same model with a different prompt framing.

The RL signal is straightforward: each attempt at a node is an action, the spec-match score is the reward, and the policy is updated to favor generations that satisfy specs on the first try. Over time, the model learns to generate correct S-expressions without needing retries, and the retry budget can be reduced.

```
reward(node) = type_match × shape_match × constraint_score × intent_score
```

No rewrite proposals, no separate critic training, no bootstrapping problem. The spec *is* the critic.

**Loop 3 — Library extraction and abstraction (outermost, hours to days).**
Successful program patterns are mined for recurring sub-expressions. Common sub-trees are refactored into new named abstractions (macros) and added to the primitive vocabulary.

Following the DreamCoder paradigm, this operates as a wake-sleep cycle:
- **Wake:** Generate programs, evaluate them, collect successful ones.
- **Sleep (abstraction):** Refactor successful programs to expose shared structure. Extract new library primitives via anti-unification or e-graph matching.
- **Sleep (dreaming):** Retrain the neural search policy on fantasized programs sampled from the updated library.

New primitives are themselves S-expressions:
```lisp
(defmacro self-attention (x Wq Wk Wv)
  (attention (matmul x Wq) (matmul x Wk) (matmul x Wv)))
```

Once extracted, `self-attention` becomes a single token in the vocabulary, compressing what was previously a deep sub-tree into an atom. The type system is extended with the new primitive's signature. Constrained decoding tables are regenerated.

### 6.2 Reward Structure

Per-node rewards are computed from spec matching. The verification method depends on the goal level (see §3.5):

| Signal | Verification method | Type |
|--------|---------------------|------|
| Type match | Mechanical | Hard pass/fail |
| Shape match | Mechanical | Hard pass/fail |
| Constraint satisfaction | Mechanical | Hard pass/fail |
| Goal satisfaction | **Depends on goal level** (see below) | Scalar 0–1 |
| Parent success | Propagated after parent evaluates | Discounted scalar |
| Global outcome | Propagated from root | Heavily discounted scalar |

Goal satisfaction verification is staged:
- **Level 0 (examples):** Execute on inputs, compare outputs. Fully mechanical.
- **Level 1 (patterns):** Match output against pattern. Fully mechanical.
- **Level 2 (predicates):** Evaluate the lambda on the output. Fully mechanical.
- **Level 3 (relations):** Partially mechanical, partially model-judged.
- **Level 4+ (natural language):** Model self-evaluation (LLM-as-judge).

At early curriculum stages, **all rewards are mechanical.** No judge is needed until the model is capable enough to be one. This eliminates the cold-start evaluation problem.

Hard constraints gate the reward — if type or shape fails, the reward is zero regardless of other signals:

```
reward(node) = hard_gate × (w1 · goal_score + w2 · parent_success + w3 · global_outcome)
```

On retry, only the best attempt's reward is used for the policy gradient update, but all attempts contribute training data (the failed attempts are negative examples for the same state).

---

## 7. Delegation and Hierarchical Composition

### 7.1 Submodel Invocation

The `submodel` primitive dispatches evaluation of a subtree to a specialist model:

```lisp
(submodel :id "vision-encoder"
  (:spec :type tensor :shape [batch, 512] :goal (:satisfy (lambda (out) (unit-norm out))))
  input-image)
```

The conductor model generating the outer tree does not need to know the internal architecture of the vision encoder. It sees only the spec (what it expects) and the result (what it gets). Different levels of the tree naturally correspond to different levels of abstraction and can be served by different models.

### 7.2 Subagent Invocation

The `subagent` primitive is similar but permits the child to generate arbitrary sub-trees:

```lisp
(subagent
  (:spec :type string
         :goal (:intent "summarize the document")
         :constraints (max-length 200)))
```

A subagent receives the spec as its task description and generates whatever program tree it needs to satisfy it. Its internal tree is opaque to the parent — only the evaluated result crosses the boundary.

### 7.3 Fast/Slow Weight Interpretation

This hierarchy maps naturally onto fast weight architectures:

- **Outermost nodes (conductor):** Slow weights. Change rarely. Represent high-level strategy and decomposition. Trained primarily by Loop 3 (library extraction) and Loop 2 (structural RL).
- **Middle nodes (composition):** Medium-speed adaptation. Route data between specialists, manage control flow. Trained by Loops 2 and 1.
- **Leaf nodes (tensor ops):** Fast weights. Change per-input via the differentiable evaluator. Trained by Loop 1 (gradient descent). These are the input-dependent, context-varying parameters analogous to fast weight programmers.

The boundary between "structure" and "parameter" is not hardcoded — it emerges from which training loop dominates at each depth of the tree.

---

## 8. Self-Referential Properties

Because the language is homoiconic, several closure properties hold:

1. **Programs can describe architectures.** A neural network architecture is an S-expression. The model can generate, inspect, and rewrite architectures as data.

2. **Specs are programs.** A specification is an S-expression that can be evaluated (to check satisfaction), transformed (to relax or tighten constraints), and composed (to build complex contracts from simple ones).

3. **Training signals are programs.** A reward function can be expressed as an S-expression, evaluated by the same evaluator, and itself subjected to rewriting.

4. **The library is a program.** The set of learned abstractions is a list of `defmacro` expressions. Library extraction is a program transformation. The vocabulary is data.

5. **Self-critique is just re-evaluation.** The model judges its own output by receiving the spec and the result in a judge prompt — no separate critic architecture. The judge's output feeds directly into the reward. The model improves by learning to satisfy its own specs on the first attempt.

This closure means the system can, in principle, reason about and modify any component of itself using the same mechanisms it uses for any other task. The generator can propose new training objectives. The critic can critique its own critique strategy. The library extractor can abstract its own extraction heuristics.

---

## 9. Relationship to Prior Work

| Component | Prior Work | What SELPH adds |
|-----------|-----------|-----------------|
| Tree credit assignment | Tree-GRPO, TEMPO, TreePO | Per-node specs as contracts; typed reward decomposition |
| Hierarchical agents | ReAcTree, AOP | Formal homoiconic language; specs as first-class values |
| Library learning | DreamCoder, Stitch | Integration with neural constrained decoding; tensor-native DSL |
| Differentiable programs | NEAR, NeuralTerpret | Homoiconic representation; unified with agentic delegation |
| Constrained decoding | XGrammar, Outlines, Guidance | Type-aware masking over a tensor-native S-expression grammar |
| Fast weight programming | FWP, DeltaNet, GLA, Mamba2 | Explicit symbolic representation of the slow/fast boundary |
| Homoiconic LLM output | Pel, Lisp-REPL paper | Full training regime; tree-structured context; differentiable eval |
| Self-specifications | Self-Spec | Specs as typed, evaluatable, composable S-expressions |

---

## 10. Open Questions and Risks

1. **Tokenizer alignment.** BPE vocabularies trained on natural language are misaligned with S-expression syntax. Options: custom tokenizer; character-level model; byte-level encoding with learned grouping.

2. **Continuous relaxation tractability.** Gumbel-softmax over operation choices requires executing all type-compatible operations in parallel during training. For nodes with >20 compatible ops, this may be prohibitive. Possible mitigation: top-k relaxation with learned pruning.

3. **Retry budget vs. training efficiency.** Allowing retries at each node provides learning signal but multiplies inference cost. The system must learn to reduce retries over time. Possible schedule: start with generous retry budget (3–5 attempts per node), anneal toward single-shot generation as training progresses.

4. **Library bloat.** Unconstrained abstraction extraction could produce an ever-growing vocabulary that dilutes the constrained decoding advantage. Mitigation: compression-based pruning (keep an abstraction only if it compresses the corpus more than its description cost).

5. **Evaluation depth.** Very deep trees require many sequential generation steps. Each node's generation depends on its left siblings' results, limiting parallelism. Mitigation: speculative execution with spec-based predictions; breadth-first generation of specs followed by parallel depth-first execution of leaves.

6. **Goal level transitions.** When should a curriculum stage graduate from mechanical goals (examples/predicates) to model-judged goals (natural language)? Too early and the judge is unreliable. Too late and the model doesn't learn to work with ambiguous specifications. Mitigation: overlap stages — introduce Level 3 goals alongside Level 2 goals, using the mechanical verification as ground truth to calibrate the model's self-judgment.

---

## 11. Curriculum Bootstrapping

The architecture naturally supports training increasingly capable models where each stage's learned abstractions become the next stage's primitives. This is not a separate training trick — it falls directly out of the library learning mechanism (Loop 3) applied across a staged curriculum.

### 11.1 Core Principle

At each stage:

1. The model trains on tasks at the current complexity level
2. Loop 3 extracts successful patterns into new library primitives
3. The new primitives are added to the vocabulary and type system
4. The constrained decoding tables are regenerated
5. The next stage begins with a richer language and a smaller effective search space

The model from stage N becomes the initialization for stage N+1. The library from stage N is frozen into the primitive set for stage N+1. Each stage is strictly easier than it would be without the prior stages' abstractions, because the building blocks are higher-level.

### 11.2 Proposed Curriculum

**Stage 0 — Tokens and embeddings.**
Primitives: character-level operations, embed, lookup, concat-strings.
Tasks: Predict next character. Spell words. Tokenize input.
Learned abstractions: `word`, `token`, `encode`.
Model: Tiny (10–50M params). Trains from scratch on the SELPH grammar itself.

This stage teaches the model the language. It learns to produce valid S-expressions, satisfy trivial type specs, and manipulate the most basic units. The model at this stage is barely useful — but it can reliably output well-formed SELPH.

**Stage 1 — Word and phrase composition.**
Primitives: Stage 0 library + basic string operations.
Tasks: Generate words matching a pattern. Compose phrases. Simple templates.
Learned abstractions: `phrase`, `noun-phrase`, `verb-phrase`, `template-fill`.
Model: Small (50–200M params), initialized from Stage 0.

**Stage 2 — Sentences and simple logic.**
Primitives: Stage 1 library + logical operators (and, or, not, implies), comparison, arithmetic.
Tasks: Generate grammatical sentences. Solve arithmetic. Evaluate boolean expressions. Simple if/then reasoning.
Learned abstractions: `sentence`, `arithmetic-expr`, `logical-chain`, `conditional`.
Model: 200M–1B params, initialized from Stage 1.

The key transition here: the model starts producing programs that have *two levels* — an outer structure (sentence template) composed of inner primitives (phrases). Tree-structured context begins to matter.

**Stage 3 — Paragraphs and multi-step reasoning.**
Primitives: Stage 2 library + sequence operations (map, reduce, filter), state tracking.
Tasks: Generate coherent paragraphs. Solve multi-step word problems. Chain logical deductions. Follow instructions with 3–5 steps.
Learned abstractions: `paragraph`, `reasoning-chain`, `step-sequence`, `state-update`.
Model: 1–3B params, initialized from Stage 2.

At this stage the model is generating trees 3–4 levels deep. The delegation mechanism (`subagent`) can start being used — a paragraph-level node delegates to sentence-level nodes using specs.

**Stage 4 — Research and synthesis.**
Primitives: Stage 3 library + retrieval operations, comparison, summarization patterns.
Tasks: Answer questions requiring multiple sources. Compare and contrast. Produce structured analyses. Simple literature review.
Learned abstractions: `search-and-synthesize`, `compare`, `evidence-chain`, `structured-report`.
Model: 3–7B params, initialized from Stage 3.

**Stage 5 — Complex reasoning and planning.**
Primitives: Stage 4 library + planning operators, constraint satisfaction, backtracking.
Tasks: Logic puzzles. Multi-constraint planning. Code generation. Mathematical proofs. Tasks requiring self-correction via retry.
Learned abstractions: `plan`, `constrained-search`, `prove`, `solve-with-backtrack`.
Model: 7B+ params, initialized from Stage 4.

**Stage 6 — Architecture search and meta-learning.**
Primitives: Stage 5 library + tensor operations (matmul, attention, etc.), architecture description primitives.
Tasks: Design small neural networks for given tasks. Optimize hyperparameters. Compose specialist models. Propose training curricula.
Learned abstractions: `attention-block`, `residual-layer`, `training-loop`, `architecture`.
Model: 7B+ params, initialized from Stage 5.

At this stage, the system becomes self-referential: it can generate programs that describe neural architectures, evaluate them via the differentiable evaluator, and learn which architectural patterns work. The library it builds at this stage includes the building blocks of the models it's running on.

### 11.3 Why This Works

**Search space compression.** At each stage, the effective vocabulary for any given node is small — the model chooses among the current stage's primitives, not among raw tensor operations. A Stage 4 model generating a research synthesis selects from abstractions like `search-and-synthesize` and `compare`, not from `matmul` and `relu`. The combinatorial explosion is managed by raising the abstraction level.

**Curriculum as library.** Traditional curriculum learning adjusts data difficulty but keeps the model's action space fixed. Here, the action space itself grows with the curriculum. Each stage doesn't just train on harder data — it trains with a more powerful language. The model at Stage 5 literally has words (primitives) that the Stage 2 model didn't have.

**No catastrophic forgetting of primitives.** Earlier abstractions are frozen into the vocabulary. The Stage 4 model can still generate sentences (it has `sentence` as a primitive) but doesn't need to think about character-level operations to do so — that's compiled away. This is analogous to how humans don't think about letter formation when writing essays.

**Model size scales with abstraction depth.** Smaller models handle lower stages because the programs are shallow and the vocabulary is small. Larger models are needed only when the tree depth and vocabulary size demand it. You don't waste a 7B model on tokenization.

**Transfer is structural, not just parametric.** When a Stage N model initializes Stage N+1, it transfers both weights *and* the library. The library transfer is arguably more important — it defines what concepts the model can express. This is transferring the language of thought, not just the statistical associations.

### 11.4 Stage Transitions

A stage transitions to the next when:

1. **Saturation:** The model's performance on current-stage tasks stops improving.
2. **Library stability:** Loop 3 stops discovering new useful abstractions (the library has converged).
3. **Spec pass rate:** The model satisfies >95% of current-stage specs on the first attempt (retry rate drops below 5%).

At transition:
- The current library is frozen and added to the primitive set
- The type system is extended with new primitive signatures
- Constrained decoding tables are regenerated
- The model is optionally expanded (additional layers/width) before initializing the next stage
- New training data at the higher complexity level is introduced

### 11.5 Bootstrapping the First Model

Stage 0 is the cold-start problem. The initial model must learn to produce valid SELPH from scratch. Approaches:

1. **Grammar pre-training.** Generate a large synthetic corpus of valid SELPH programs (randomly sampled from the CFG). Pre-train the model on next-token prediction over this corpus. This teaches syntax without semantics.

2. **Trivial-task fine-tuning.** Fine-tune on tasks with known solutions expressed in SELPH: arithmetic, string manipulation, simple lookups. Use the constrained decoder to guarantee syntactic validity while the model learns semantic correctness.

3. **Distillation from an existing LLM.** Use a large existing model (Claude, GPT) to generate SELPH programs for simple tasks. Fine-tune the small Stage 0 model on these. The large model doesn't need to be perfect — it just needs to produce syntactically valid SELPH with roughly correct semantics. The RL loop will refine.

Option 3 is probably the fastest path: use an existing LLM as a teacher to bootstrap the first SELPH model, then the curriculum takes over.

---

## 12. Library Namespace Design

### 12.1 Problem

As the library grows across curriculum stages, the flat vocabulary becomes unnavigable. A Stage 5 model might have 500+ primitives. Choosing from a flat list at each node wastes both context and decoding bandwidth. The model needs structure to find the right primitive efficiently.

### 12.2 Hierarchical Namespace

Library primitives are organized into a tree-structured namespace using dot-separated paths:

```
ops.tensor.matmul
ops.tensor.add
ops.tensor.softmax
ops.string.concat
ops.string.split
ops.logic.and
ops.logic.implies
text.char.uppercase
text.word.stem
text.phrase.noun-phrase
text.sentence.compose
models.language.translate.spanish
models.language.summarize
models.vision.encode
models.vision.classify
search.web.google
search.files.path
search.memory.recall
tasks.reasoning.chain
tasks.reasoning.prove
tasks.research.synthesize
flow.if
flow.let
flow.map
flow.pipe
```

### 12.3 Generation as Namespace Traversal

When the model generates a primitive, it doesn't emit the full path as a monolithic token. It traverses the namespace tree level by level:

```
Step 1: Choose category    → ops | text | models | search | tasks | flow
Step 2: Choose subcategory → ops.tensor | ops.string | ops.logic
Step 3: Choose primitive   → ops.tensor.matmul | ops.tensor.add | ...
```

At each step, constrained decoding masks everything outside the valid children of the current namespace node. The type system further narrows: if the current position requires a `(-> matrix matrix matrix)`, only namespace branches containing compatible types are offered.

This turns a flat 500-way choice into a sequence of 5–15 way choices across 2–4 levels. The branching factor at each level stays small and semantically coherent.

### 12.4 Namespace Structure Mirrors Program Structure

There is a natural correspondence between depth in the program tree and depth in the library namespace:

| Program tree level | Typical namespace depth | Example |
|-------------------|------------------------|---------|
| Root (conductor) | Shallow — `tasks.*` | `tasks.research.synthesize` |
| Mid-level (composition) | Medium — `flow.*`, `models.*` | `flow.pipe`, `models.language.translate` |
| Leaf (computation) | Deep — `ops.*`, `text.*` | `ops.tensor.matmul`, `text.char.uppercase` |

A conductor node generating a high-level plan should never need to browse `ops.tensor.*`. A leaf node doing matrix arithmetic should never need to browse `tasks.research.*`. The tree-structured context (§5) already tells the model what level of abstraction it's operating at — the namespace structure reinforces that by grouping primitives at matching abstraction levels.

### 12.5 Namespace as Context Filter

The namespace can be used to pre-filter the vocabulary before constrained decoding even runs. Given the current node's position in the program tree:

1. **Depth heuristic:** Shallow nodes see top-level namespaces (`tasks`, `models`, `flow`). Deep nodes see operational namespaces (`ops`, `text`). Both can access any namespace, but the default presentation is filtered.

2. **Type filter:** The type constraint from the current position eliminates incompatible branches. If the slot requires `(-> string string)`, the `ops.tensor.*` branch is entirely masked.

3. **Stage filter:** At curriculum Stage 2, the model hasn't learned Stage 5 primitives. The namespace only includes branches populated by the current and prior stages.

These filters compose: `depth_filter ∩ type_filter ∩ stage_filter` produces a small, relevant set of namespace branches at each generation step.

### 12.6 Library Learning Populates the Namespace

When Loop 3 extracts a new abstraction, it must be placed in the namespace. The placement follows from the abstraction's properties:

- **What it operates on** determines the top-level category (`ops` for tensor/string operations, `text` for language operations, `models` for delegating to sub-models)
- **What it composes** determines the subcategory (an abstraction built from `text.word.*` primitives goes into `text.phrase.*` or `text.sentence.*`)
- **Its type signature** must be consistent with its namespace siblings

Placement can be automated: examine the abstraction's definition, trace which namespace branches its constituent primitives come from, and insert it one level above the deepest common ancestor. An abstraction that composes `text.word.stem` and `text.word.lemmatize` becomes `text.word.normalize`. An abstraction that composes `text.sentence.compose` and `ops.logic.implies` might become `tasks.reasoning.step`.

### 12.7 Namespace as Compression Signal

The namespace structure itself provides a signal for library extraction. If the model frequently traverses the same namespace path to reach a primitive — e.g., always using `ops.tensor.matmul` followed by `ops.tensor.softmax` followed by `ops.tensor.matmul` — that traversal pattern is a candidate for abstraction. The new primitive (`models.attention.single-head`) compresses three namespace traversals into one.

This means the namespace gets *shallower* in frequently-used areas over time. Early in training, computing attention requires navigating deep into `ops.tensor.*` three times. After library extraction, it's a single step into `models.attention.*`. The namespace tree self-prunes toward efficiency.

### 12.8 Interaction with Tokenizer

Each namespace level can be a single token or a sequence of tokens in the model's vocabulary. Two options:

**Option A — Path segments as tokens.** Each namespace level is one token: `ops`, `.tensor`, `.matmul`. The model generates three tokens to specify a primitive. The constrained decoder masks at each level. Simple, works with any tokenizer.

**Option B — Full paths as tokens.** Common full paths (e.g., `ops.tensor.matmul`) are single tokens in the vocabulary. Less common paths decompose into segments. This is essentially BPE applied to the namespace — frequent paths get compressed. This mirrors how programming language IDEs do auto-complete: common operations are quick, obscure ones require more keystrokes.

Option A is simpler to implement and more flexible. Option B is more efficient at inference. A practical path: start with Option A, then apply frequency-based path merging (analogous to BPE) as the library stabilizes.

### 12.9 Unified Namespace: Library, Cache, and Data

The namespace is not just for function definitions. It is a unified store for three kinds of values that are traditionally separate systems:

**Functions (library).** Reusable abstractions extracted by Loop 3.
```
ops.tensor.matmul        → (lambda (A B) ...)
models.attention.head     → (defmacro attention (Q K V) ...)
tasks.reasoning.chain     → (defmacro chain (steps) ...)
```

**Data (reference knowledge).** Facts, datasets, and reference values. Populated at system initialization or by data-loading tasks.
```
data.countries.russia.capital   → "Moscow"
data.countries.russia.area      → 17098242
data.models.bert.vocab-size     → 30522
data.config.max-retries         → 3
```

**Results (cache).** Memoized outputs from evaluated nodes. Populated at runtime during program execution.
```
cache.tree-42.node-3.result     → "Russia"
cache.tree-42.node-5.result     → "Moscow"
cache.search.web.{query-hash}   → [result-1, result-2, ...]
```

All three use the same namespace traversal, the same lookup mechanism, and the same constrained decoding path. The model doesn't distinguish between "call a function," "look up a fact," and "retrieve a cached result" — they're all namespace resolutions. The difference is only in what the path resolves to: a callable, a value, or a memoized value.

### 12.10 Specs as Training Data

A key consequence of this unification: **specs are training data for the nodes they annotate.**

For a differentiable node with a Level 0 goal:
```lisp
(:spec :type string
       :goal (:examples (("hello" -> "HELLO") ("world" -> "WORLD"))))
```

The examples are literally the supervised training pairs for the parameters inside that node. Loop 1 (gradient descent) uses the spec's examples as the dataset and the spec's goal predicate as the loss function. The spec *is* the training configuration.

This means the parent that wrote the spec is also defining the child's training signal. A conductor node that decomposes a task into subtasks is simultaneously:
1. Defining what each subtask should do (the spec)
2. Providing training data for the subtask's implementation (the goal examples)
3. Defining the acceptance criterion (the goal predicate)

The spec is the contract, the dataset, and the test suite — one object.

### 12.11 Namespace as Working Memory

The cache namespace solves the state management problem. Instead of threading state through the program tree as explicit arguments, nodes read and write to the namespace:

```lisp
; A search node writes its results to the namespace
(let ((results (search.web.google query)))
  (store cache.current.search-results results)
  results)

; A later node reads from the namespace
(let ((results (lookup cache.current.search-results)))
  (text.sentence.compose (summarize results)))
```

The `store` and `lookup` operations are namespace writes and reads. They're visible in the program tree (not hidden side effects), and they're accessible to sibling/parent nodes through the standard context mechanism.

For cross-tree persistence (results that should survive beyond the current program), nodes write to `data.*` rather than `cache.*`. Cache is ephemeral (scoped to one program execution). Data is persistent (survives across executions and is available to future programs).

### 12.12 Global vs. Local Namespace Scoping

The namespace has three scoping levels:

| Scope | Prefix | Lifetime | Visibility |
|-------|--------|----------|------------|
| Global | `ops.*`, `data.*`, `models.*` | Permanent | All programs, all nodes |
| Program | `cache.*` | One program execution | All nodes in current tree |
| Node | `local.*` | One node evaluation | Current node only |

Global scope holds the library and reference data. Program scope holds memoized results and working memory for the current task. Node scope holds temporaries that shouldn't leak.

When the model generates a namespace path, the scope is determined by the first segment. Constrained decoding can restrict which scopes are available based on context — a pure computation node might only see `ops.*` and `local.*`, while a conductor node might see all scopes.

---

## 13. Self-Hosting: The Model as a SELPH Program

*This section describes a long-term direction, not an MVP concern. It is speculative and included to establish the logical endpoint of the architecture.*

### 13.1 The Closure Argument

Every component of the system is already a SELPH expression: programs, specs, data, library entries, training signals, cached results. The one remaining exception is the model itself — a conventional transformer implemented in Python/PyTorch, trained to *output* SELPH but not *written in* SELPH.

If the model were also a SELPH program, the system would be fully closed. Every component could be inspected, modified, and composed using the same language. The model could reason about its own architecture, propose modifications to itself, and evaluate those modifications using the same mechanisms it uses for any other task.

### 13.2 Type Signature

The model is a function from spec to program:

```lisp
model : (-> spec namespace program)
```

It takes a spec (what to do) and a namespace (what's available — library, cached results, context), and produces a program (how to do it). This is the same type signature the model already implements implicitly. Self-hosting just makes it explicit.

When the model delegates via `subagent`, it is invoking itself recursively with a narrower spec:

```lisp
(defmacro model (spec namespace)
  (let ((plan (decompose spec namespace)))
    (if (leaf? plan)
        (generate-leaf plan namespace)
        (let ((sub-specs (children plan)))
          (map (lambda (s) (model s namespace)) sub-specs)))))
```

The tree of S-expressions we've been describing throughout this spec *is* the call tree of this recursive program. The conductor isn't a separate orchestration layer — it's the top-level call. The leaf generators aren't separate worker models — they're the base case.

### 13.3 Bootstrap Path

The model cannot be written in SELPH before SELPH models exist. The curriculum provides a staged transition from external scaffolding to self-hosting:

**Stages 0–3: Scaffolded.**
The model is a conventional transformer (Python/PyTorch). It is trained to output SELPH programs but is not itself a SELPH program. The evaluator, the constrained decoder, and the training loops are all external infrastructure.

```
[Python/PyTorch model] → generates → [SELPH programs]
```

**Stages 4–5: Partial self-description.**
The model has learned to write SELPH programs that describe neural architectures and reasoning strategies. Some of these programs are functionally equivalent to components of the model itself. At this stage, begin replacing individual components of the conventional model with SELPH-described equivalents:

- Replace the attention routing logic with a SELPH program that selects which library entries to attend to
- Replace the output head with a SELPH program that traverses the namespace and selects primitives
- Replace the prompt construction logic with a SELPH program that builds tree-structured context

Each replacement is validated: does the SELPH-described component match the performance of the Python component it replaces? If yes, the SELPH version is promoted. If no, it stays as a candidate for further training.

```
[Hybrid model: Python shell + SELPH components] → generates → [SELPH programs]
```

**Stage 6: The flip.**
Once enough components have been replaced, the model is predominantly a SELPH program executed by the SELPH evaluator. The Python/PyTorch layer becomes pure infrastructure — it runs the evaluator, manages GPU memory, and handles I/O — but the model's logic, architecture, and routing are all SELPH expressions.

```
[SELPH evaluator] → runs → [SELPH model] → generates → [SELPH programs]
```

The flip doesn't need to happen all at once. It's a gradual process of replacing opaque components with transparent ones, validated at each step.

### 13.4 Self-Modification

Once the model is a SELPH program, Loop 2 (structural evaluation and retry) applies to the model itself, not just to its outputs. The model can:

1. **Inspect its own architecture.** The program that defines the model is stored in the namespace at `models.self.current`. It's an S-expression. The model can read it.

2. **Propose modifications.** Given a task it failed, the model can examine which component of its own program was responsible (using the same tree-structured credit assignment from §5.3) and propose a rewrite.

3. **Evaluate modifications.** The proposed rewrite is a new SELPH program. The evaluator runs it on validation tasks. If it performs better, the rewrite is accepted and `models.self.current` is updated.

4. **Learn from modifications.** Successful self-modifications become training data for Loop 2. The model learns which kinds of architectural changes tend to improve performance, building a meta-library of architectural patterns.

This is not unconstrained self-modification. Every proposed change must pass through the same spec-satisfaction pipeline as any other generated program. The spec for the model itself is its performance on a held-out validation set. The type system constrains modifications to be architecturally valid. The namespace scoping prevents the model from modifying its own evaluation infrastructure.

### 13.5 RL Formulation

The self-hosting model has a clean RL interpretation:

| Component | RL concept |
|-----------|-----------|
| Current spec + namespace context | State |
| Generated SELPH program | Action |
| Evaluate program against spec | Environment step |
| Spec satisfaction score | Reward |
| The model's own SELPH program | Policy (parameterized by its weights) |

The policy *is* a SELPH program with differentiable parameters. Loop 1 optimizes the parameters (policy gradient on the continuous weights within the program). Loop 2 optimizes the structure (RL on the discrete program architecture). Loop 3 optimizes the language (library extraction that changes the action space itself).

Standard RL has a fixed action space and optimizes the policy. SELPH RL has a policy that can modify its own action space (by extracting new library primitives) and eventually modify its own structure (self-modification). The curriculum constrains this to happen gradually and safely — the model earns the ability to modify deeper components of itself by first proving competence at modifying external programs.

### 13.6 What Self-Hosting Buys You

**Interpretability.** The model is a program you can read. Not "interpret the attention patterns" — literally read the code that defines what the model does. Every routing decision, every composition choice, every delegation strategy is an explicit S-expression.

**Targeted modification.** Instead of fine-tuning billions of parameters to change one behavior, you rewrite the specific node in the model's program tree responsible for that behavior. The rest of the model is untouched.

**Architecture search as program synthesis.** Finding better architectures is the same problem as finding better programs. The same Loop 2 + Loop 3 machinery that improves task programs improves the model's own architecture. No separate NAS pipeline.

**Transferable architecture.** The model's architecture is a SELPH program stored in the namespace. It can be shared, versioned, diffed, and composed with other architectures. A good attention variant discovered by one model instance can be imported into another via library sharing.

**Auditability.** Every decision the model makes is traceable to a specific node in a specific program. Self-modification is logged as a sequence of namespace updates. The full history of how the model arrived at its current architecture is a list of SELPH rewrites.

### 13.7 What Self-Hosting Costs You

**Performance.** A SELPH program interpreted by a Python evaluator is orders of magnitude slower than a compiled PyTorch model. This is mitigated by the compilation path (§10 discussion) — subtrees of pure tensor ops can be lowered to XLA/Triton — but the dynamic, self-modifying parts will always be slower than static compiled code. The practical tradeoff: use SELPH for the high-level routing and architectural decisions (which are called infrequently) and compile the inner loops to native GPU kernels.

**Stability.** A model that can modify itself can potentially destabilize itself. Mitigations: version control on `models.self.current` with rollback; validation gates on self-modification (changes must improve held-out performance); rate limiting on self-modification frequency; certain components marked as immutable (the evaluator itself, the spec-checking logic, the reward computation).

**Verification.** How do you verify that a self-modified model is still safe/aligned? The spec system helps — the model's own spec defines its acceptable behavior — but quis custodiet ipsos custodes? The evaluation infrastructure (the SELPH evaluator, the namespace, the type system) must remain outside the model's self-modification scope. These are the invariants that the model cannot change about itself.

---

## 14. Minimum Viable Implementation

A practical first implementation targets Stages 0–2 of the curriculum:

1. Define a SELPH subset with ~30 primitives covering string ops, arithmetic, and basic logic
2. Implement the grammar as a CFG for constrained decoding (e.g., via XGrammar or llama.cpp grammar mode)
3. Build a simple evaluator (not yet differentiable) as a Python/JAX interpreter over SELPH ASTs
4. Generate synthetic training corpus of valid SELPH programs for Stage 0
5. Train a small model (50–200M) on next-token prediction over the synthetic corpus with constrained decoding
6. Implement the spec-match reward and tree-structured context for Stage 1 tasks
7. Implement Loop 2 (per-node RL with spec-based rewards and self-retry) using GRPO
8. Run through Stages 0–2 and validate: does the library grow? Do later stages benefit from earlier abstractions? Does the retry rate decrease over training?

Defer to a later phase: the differentiable evaluator (Loop 1), library extraction via e-graphs (Loop 3), delegation/subagent, Stages 3+, and self-hosting (§13).

The key validation question at the MVP stage is whether the curriculum bootstrapping works: does a model trained with Stage 1 abstractions learn Stage 2 tasks faster than a model trained from scratch on Stage 2?
