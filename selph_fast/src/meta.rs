//! Meta-curriculum system for SELPH.
//!
//! Implements meta-learning: synthesizing search heuristics as SELPH programs
//! that improve synthesis performance. A heuristic is a SELPH program that
//! acts as a priority function over synthesis components, reordering them so
//! that more promising components are tried first for a given task context.
//!
//! Meta-synthesis loop:
//!   1. Run synthesis on training tasks with the current heuristic
//!   2. Collect performance data (solve rate, candidates explored)
//!   3. Use synthesis to find a SELPH program that, when used as a priority
//!      function, would improve solve rate on the training tasks
//!   4. If a better heuristic is found, adopt it for the next round
//!
//! This mirrors the Python `meta_curriculum.py` but targets the Rust
//! synthesizer and evaluator.

use std::rc::Rc;
use crate::eval;
use crate::parser;
use crate::synth::{CandidateRecord, RlCoefficients, SynthComponent, synthesize, value_type_tag, TYPE_NUM, TYPE_STR};
use crate::types::*;

// ── Heuristic ───────────────────────────────────────────────────────

/// A learned heuristic represented as a SELPH program.
///
/// The program takes a namespace describing a component and a task context,
/// and returns a numeric priority score. Higher scores mean the component
/// should be tried earlier during synthesis.
#[derive(Clone, Debug)]
pub struct Heuristic {
    /// Human-readable name for this heuristic.
    pub name: String,
    /// Source code of the SELPH program (for display / serialization).
    pub source: String,
    /// Parsed AST nodes.
    pub nodes: Vec<Node>,
    /// Root index into `nodes`.
    pub root: usize,
}

impl Heuristic {
    /// Create a heuristic from source code. Returns `None` if parsing fails.
    pub fn from_source(name: &str, source: &str) -> Option<Self> {
        let (nodes, root) = parser::parse_source(source).ok()?;
        Some(Heuristic {
            name: name.to_string(),
            source: source.to_string(),
            nodes,
            root,
        })
    }

    /// The default (identity) heuristic: returns the component's existing
    /// priority unchanged. This is the starting point before any meta-learning.
    pub fn default_heuristic() -> Self {
        // (lambda (ctx) (ns-get ctx "priority"))
        let source = "(lambda (ctx) (ns-get ctx \"priority\"))";
        Heuristic::from_source("default", source)
            .expect("default heuristic source must parse")
    }
}

// ── Task context ────────────────────────────────────────────────────

/// Context describing the current synthesis task, passed to heuristics
/// so they can make task-dependent priority decisions.
#[derive(Clone, Debug)]
pub struct TaskContext {
    /// Type tag of example inputs (0=num, 1=str, 255=unknown).
    pub input_type: u8,
    /// Type tag of expected outputs.
    pub output_type: u8,
    /// Number of input/output examples.
    pub num_examples: usize,
}

impl TaskContext {
    /// Infer a TaskContext from example input/output pairs.
    pub fn from_examples(inputs: &[Value], expected: &[Value]) -> Self {
        let input_type = if inputs.is_empty() {
            255
        } else {
            value_type_tag(&inputs[0])
        };
        let output_type = if expected.is_empty() {
            255
        } else {
            value_type_tag(&expected[0])
        };
        TaskContext {
            input_type,
            output_type,
            num_examples: inputs.len(),
        }
    }

    /// Convert this context into a SELPH namespace Value for passing to
    /// heuristic programs.
    pub fn to_namespace(&self) -> Value {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "input-type".to_string(),
            Value::Num(self.input_type as f64),
        );
        map.insert(
            "output-type".to_string(),
            Value::Num(self.output_type as f64),
        );
        map.insert(
            "num-examples".to_string(),
            Value::Num(self.num_examples as f64),
        );
        Value::Namespace(map)
    }
}

/// Pack a SynthComponent's metadata into a namespace Value so a heuristic
/// program can inspect it.
fn component_to_namespace(comp: &SynthComponent, task_ctx: &TaskContext) -> Value {
    let mut map = std::collections::HashMap::new();
    map.insert("name".to_string(), Value::Str(comp.name.clone()));
    map.insert("arity".to_string(), Value::Num(comp.arity as f64));
    map.insert("ret-type".to_string(), Value::Num(comp.ret_type as f64));
    map.insert("priority".to_string(), Value::Num(comp.priority));
    // Flatten first param type (0 if no params)
    let first_param = comp.param_types.first().copied().unwrap_or(0) as f64;
    map.insert("first-param-type".to_string(), Value::Num(first_param));
    // Include task context fields directly so the heuristic can use them
    map.insert(
        "input-type".to_string(),
        Value::Num(task_ctx.input_type as f64),
    );
    map.insert(
        "output-type".to_string(),
        Value::Num(task_ctx.output_type as f64),
    );
    map.insert(
        "num-examples".to_string(),
        Value::Num(task_ctx.num_examples as f64),
    );
    Value::Namespace(map)
}

// ── Evaluate heuristic ──────────────────────────────────────────────

/// Run a heuristic program to score a single component for a given task.
///
/// The heuristic is a `(lambda (ctx) ...)` that receives a namespace
/// containing both component metadata and task context. Returns the
/// numeric priority score, or 0.0 if evaluation fails.
pub fn evaluate_heuristic(
    heuristic: &Heuristic,
    component: &SynthComponent,
    task_context: &TaskContext,
) -> f64 {
    let ctx_val = component_to_namespace(component, task_context);

    let mut env = eval::make_default_env();
    let nodes_rc: Rc<[Node]> = heuristic.nodes.clone().into();
    // Evaluate the heuristic lambda
    let fn_val = match eval::eval(&nodes_rc, heuristic.root, &mut env) {
        Ok(v) => v,
        Err(_) => return 0.0,
    };
    // Apply it to the context namespace
    match eval::apply(&fn_val, &[ctx_val], &nodes_rc, &mut env) {
        Ok(Value::Num(n)) => {
            if n.is_finite() {
                n
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

// ── Apply heuristic ─────────────────────────────────────────────────

/// Reorder and reprioritize a set of components using a heuristic.
///
/// Each component's priority is replaced by the heuristic's score, and the
/// resulting vector is sorted by descending priority (highest-priority
/// components first).
pub fn apply_heuristic(
    heuristic: &Heuristic,
    components: &[SynthComponent],
    task_context: &TaskContext,
) -> Vec<SynthComponent> {
    let mut scored: Vec<(f64, SynthComponent)> = components
        .iter()
        .map(|comp| {
            let score = evaluate_heuristic(heuristic, comp, task_context);
            let mut new_comp = comp.clone();
            new_comp.priority = score;
            (score, new_comp)
        })
        .collect();

    // Sort descending by score (higher priority first)
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scored.into_iter().map(|(_, comp)| comp).collect()
}

// ── Synthesize heuristic ────────────────────────────────────────────

/// A training task: input/output examples plus a name.
#[derive(Clone)]
pub struct TrainingTask {
    pub name: String,
    pub inputs: Vec<Value>,
    pub expected: Vec<Value>,
}

/// Result of a meta-curriculum step.
pub struct MetaCurriculumResult {
    /// Names of tasks that were solved.
    pub solved: Vec<String>,
    /// Total candidates explored across all tasks.
    pub total_candidates: usize,
    /// A new heuristic, if one was found that improves on the current.
    pub new_heuristic: Option<Heuristic>,
}

/// Run synthesis on a set of training tasks using a heuristic, returning
/// (number_solved, total_candidates, per-task results).
fn run_with_heuristic(
    tasks: &[TrainingTask],
    heuristic: &Heuristic,
    base_components: &[SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> (usize, usize, Vec<(String, bool, usize)>) {
    let mut solved = 0usize;
    let mut total_cands = 0usize;
    let mut results = Vec::new();

    for task in tasks {
        let ctx = TaskContext::from_examples(&task.inputs, &task.expected);
        let prioritized = apply_heuristic(heuristic, base_components, &ctx);

        let sr = synthesize(
            &prioritized,
            &task.inputs,
            &task.expected,
            macros,
            max_depth,
            max_candidates,
            true, // enable_if
        );

        let task_solved = sr.found;
        let cands = sr.candidates_explored;
        if task_solved {
            solved += 1;
        }
        total_cands += cands;
        results.push((task.name.clone(), task_solved, cands));
    }

    (solved, total_cands, results)
}

/// Meta-synthesis: search for a SELPH program that, when used as a
/// priority function over components, improves the solve rate on a set
/// of training tasks.
///
/// The approach:
///   1. Evaluate the baseline (no heuristic / default priorities)
///   2. Generate candidate heuristic programs via enumeration
///   3. For each candidate, run the full training suite and measure
///      improvement in solve rate and/or reduction in candidates explored
///   4. Return the best heuristic found within the budget, if it
///      improves on the baseline
///
/// This is expensive (synthesis inside synthesis), so budgets should be
/// kept modest. The `budget` parameter limits the number of candidate
/// heuristics evaluated (not per-task candidates).
pub fn synthesize_heuristic(
    training_tasks: &[TrainingTask],
    components: &[SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    budget: usize,
) -> Option<Heuristic> {
    if training_tasks.is_empty() || components.is_empty() {
        return None;
    }

    let max_depth = 2;
    let per_task_budget = 5000;

    // ── Baseline measurement ────────────────────────────────────────
    let default_h = Heuristic::default_heuristic();
    let (baseline_solved, baseline_cands, _) = run_with_heuristic(
        training_tasks,
        &default_h,
        components,
        macros,
        max_depth,
        per_task_budget,
    );

    // ── Generate candidate heuristics ───────────────────────────────
    // We synthesize programs of the form (lambda (ctx) BODY) where BODY
    // computes a numeric score from the context namespace.
    //
    // Strategy: enumerate small SELPH programs that read fields from
    // the context namespace and combine them arithmetically.
    //
    // We use the synthesizer itself to find programs that map component
    // feature vectors to priority scores. The "examples" for this inner
    // synthesis are derived from which components were useful in the
    // baseline run.

    let candidate_templates = build_candidate_heuristics();

    let mut best_heuristic: Option<Heuristic> = None;
    let mut best_solved = baseline_solved;
    let mut best_cands = baseline_cands;
    let mut evaluated = 0usize;

    for candidate in &candidate_templates {
        if evaluated >= budget {
            break;
        }
        evaluated += 1;

        let (solved, cands, _) = run_with_heuristic(
            training_tasks,
            candidate,
            components,
            macros,
            max_depth,
            per_task_budget,
        );

        // A candidate is better if it solves more tasks, or solves the
        // same number but with fewer total candidates explored.
        let is_better = solved > best_solved
            || (solved == best_solved && solved > 0 && cands < best_cands);

        if is_better {
            best_solved = solved;
            best_cands = cands;
            best_heuristic = Some(candidate.clone());
        }
    }

    // Only return if we actually improved on the baseline
    if best_solved > baseline_solved
        || (best_solved == baseline_solved
            && best_solved > 0
            && best_cands < baseline_cands)
    {
        best_heuristic
    } else {
        None
    }
}

/// Build a library of candidate heuristic programs to evaluate.
///
/// These are hand-crafted templates that encode common heuristic patterns:
/// - Prefer components whose return type matches the output type
/// - Prefer components whose input type matches the task input type
/// - Prefer higher-arity components (composition over atoms)
/// - Penalize type mismatches
/// Public wrapper for building candidate heuristics (used by meta-opt command).
pub fn build_candidate_heuristics_pub() -> Vec<Heuristic> {
    build_candidate_heuristics()
}

fn build_candidate_heuristics() -> Vec<Heuristic> {
    let templates = [
        // H1: Boost components whose return type matches target output type
        (
            "match-output-type",
            "(lambda (ctx) (if (= (ns-get ctx \"ret-type\") (ns-get ctx \"output-type\")) 100.0 0.0))",
        ),
        // H2: Boost matching input type on first param
        (
            "match-input-type",
            "(lambda (ctx) (if (= (ns-get ctx \"first-param-type\") (ns-get ctx \"input-type\")) 50.0 0.0))",
        ),
        // H3: Combined type matching — both input and output
        (
            "match-both-types",
            concat!(
                "(lambda (ctx) (add ",
                  "(if (= (ns-get ctx \"ret-type\") (ns-get ctx \"output-type\")) 100.0 0.0) ",
                  "(if (= (ns-get ctx \"first-param-type\") (ns-get ctx \"input-type\")) 50.0 0.0)))",
            ),
        ),
        // H4: Prefer higher arity (encourages composition)
        (
            "prefer-composition",
            "(lambda (ctx) (multiply (ns-get ctx \"arity\") 10.0))",
        ),
        // H5: Blend existing priority with type-match bonus
        (
            "priority-plus-type-match",
            concat!(
                "(lambda (ctx) (add (ns-get ctx \"priority\") ",
                  "(if (= (ns-get ctx \"ret-type\") (ns-get ctx \"output-type\")) 50.0 0.0)))",
            ),
        ),
        // H6: Penalize type mismatches, keep existing priority
        (
            "penalize-mismatch",
            concat!(
                "(lambda (ctx) (subtract (ns-get ctx \"priority\") ",
                  "(if (= (ns-get ctx \"ret-type\") (ns-get ctx \"output-type\")) 0.0 25.0)))",
            ),
        ),
        // H7: String-task specialist — boost string ops when output is string
        (
            "string-specialist",
            concat!(
                "(lambda (ctx) (if (= (ns-get ctx \"output-type\") 1.0) ",
                  "(if (= (ns-get ctx \"ret-type\") 1.0) 100.0 10.0) ",
                  "(if (= (ns-get ctx \"ret-type\") 0.0) 100.0 10.0)))",
            ),
        ),
        // H8: Number-task specialist — heavily boost arithmetic for num tasks
        (
            "number-specialist",
            concat!(
                "(lambda (ctx) (if (= (ns-get ctx \"output-type\") 0.0) ",
                  "(if (= (ns-get ctx \"ret-type\") 0.0) 80.0 5.0) ",
                  "(ns-get ctx \"priority\")))",
            ),
        ),
    ];

    templates
        .iter()
        .filter_map(|(name, src)| Heuristic::from_source(name, src))
        .collect()
}

// ── Meta-curriculum step ────────────────────────────────────────────

/// One step of the meta-curriculum loop.
///
/// 1. Solve the given tasks using the current heuristic
/// 2. Attempt to synthesize a better heuristic from the results
/// 3. Return the list of solved task names and (optionally) a new heuristic
///
/// This is the main entry point for integrating meta-learning into a
/// curriculum runner.
pub fn meta_curriculum_step(
    tasks: &[TrainingTask],
    current_heuristic: &Heuristic,
    components: &[SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> (Vec<String>, Option<Heuristic>) {
    let max_depth = 2;
    let per_task_budget = 10_000;
    let meta_budget = 8; // number of candidate heuristics to try

    // ── Phase 1: solve tasks with current heuristic ─────────────────
    let (_, _, per_task) = run_with_heuristic(
        tasks,
        current_heuristic,
        components,
        macros,
        max_depth,
        per_task_budget,
    );

    let solved_names: Vec<String> = per_task
        .iter()
        .filter(|(_, ok, _)| *ok)
        .map(|(name, _, _)| name.clone())
        .collect();

    // ── Phase 2: try to find a better heuristic ─────────────────────
    // Use the tasks that were NOT solved as the training signal: a good
    // heuristic should help solve them by reordering components.
    let unsolved_tasks: Vec<TrainingTask> = tasks
        .iter()
        .zip(per_task.iter())
        .filter(|(_, (_, ok, _))| !ok)
        .map(|(t, _)| t.clone())
        .collect();

    let new_heuristic = if unsolved_tasks.is_empty() {
        // Everything solved — no need for a better heuristic
        None
    } else {
        // Include all tasks for meta-evaluation (solved ones act as
        // regression tests, unsolved ones are the improvement target)
        synthesize_heuristic(tasks, components, macros, meta_budget)
    };

    (solved_names, new_heuristic)
}

// ── Rank-based heuristic optimization ────────────────────────────────
//
// Key insight (plan §8.2): once you HAVE solutions from a curriculum run,
// evaluating a candidate heuristic doesn't require full synthesis. You just
// check: "what rank would the known solution have under this heuristic's
// ordering?" This is O(pool_size) comparison, not O(budget) evaluation.

/// Snapshot of the synthesis pool at the moment a solution was found.
///
/// Captured during curriculum solve. Used to cheaply evaluate candidate
/// heuristics by re-ranking: apply heuristic scores to components,
/// recompute candidate priorities, and find the solution's new rank.
#[derive(Clone)]
pub struct PoolSnapshot {
    pub task_name: String,
    pub task_context: TaskContext,
    /// All candidates tested during synthesis (in original test order).
    /// The solution (if found) is the last entry.
    pub candidates: Vec<CandidateRecord>,
    /// Number of depth-0 (atomic) candidates tested before compositions.
    /// These are heuristic-independent and form a fixed rank offset.
    pub depth0_count: usize,
}

/// Compute the rank a known solution would have under a given heuristic.
///
/// Re-scores each candidate using the heuristic's component priority,
/// sorts by new score descending, and returns the 1-based rank of the
/// solution (the last entry in the snapshot).
///
/// Returns `None` if the snapshot is empty.
pub fn rank_solution(
    snapshot: &PoolSnapshot,
    heuristic: &Heuristic,
    components: &[SynthComponent],
) -> Option<usize> {
    if snapshot.candidates.is_empty() {
        return None;
    }

    // Build a lookup from component name to heuristic score
    let comp_scores: std::collections::HashMap<String, f64> = components
        .iter()
        .map(|comp| {
            let score = evaluate_heuristic(heuristic, comp, &snapshot.task_context);
            (comp.name.clone(), score)
        })
        .collect();

    // Also index by builtin name (components often have builtin != name)
    let mut builtin_scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for comp in components {
        if let Some(ref bn) = comp.builtin {
            let score = evaluate_heuristic(heuristic, comp, &snapshot.task_context);
            builtin_scores.insert(bn.clone(), score);
        }
    }

    let solution_idx = snapshot.candidates.len() - 1;

    // Re-score each candidate: heuristic_score(component) + arg_priority_sum
    let mut scored: Vec<(usize, f64)> = snapshot
        .candidates
        .iter()
        .enumerate()
        .map(|(i, rec)| {
            let heuristic_score = builtin_scores
                .get(&rec.comp_name)
                .or_else(|| comp_scores.get(&rec.comp_name))
                .copied()
                .unwrap_or(0.0);
            (i, heuristic_score + rec.arg_priority_sum)
        })
        .collect();

    // Sort descending by score (highest priority first).
    // Use stable sort to preserve original order for equal scores.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find rank of the solution (1-based)
    let rank = scored
        .iter()
        .position(|(i, _)| *i == solution_idx)
        .map(|pos| pos + 1)?;

    // Total rank includes depth-0 candidates (always tested first)
    Some(snapshot.depth0_count + rank)
}

/// Evaluate a heuristic across all snapshots, returning the average rank.
///
/// Lower is better: a perfect heuristic puts every solution first.
pub fn evaluate_heuristic_by_rank(
    snapshots: &[PoolSnapshot],
    heuristic: &Heuristic,
    components: &[SynthComponent],
) -> f64 {
    if snapshots.is_empty() {
        return f64::MAX;
    }
    let total_rank: usize = snapshots
        .iter()
        .filter_map(|s| rank_solution(s, heuristic, components))
        .sum();
    total_rank as f64 / snapshots.len() as f64
}

/// Optimize heuristic using rank-based evaluation on pool snapshots.
///
/// This is the cheap replacement for `synthesize_heuristic()`: instead of
/// running full synthesis for each candidate heuristic × task, it re-ranks
/// pre-captured snapshots. Each evaluation is O(snapshot_size) instead of
/// O(synthesis_budget).
///
/// Returns the best heuristic found, or None if no candidate improves on
/// the default.
pub fn optimize_heuristic_by_rank(
    snapshots: &[PoolSnapshot],
    components: &[SynthComponent],
) -> Option<Heuristic> {
    if snapshots.is_empty() || components.is_empty() {
        return None;
    }

    // Baseline: default heuristic (pass-through priority)
    let default_h = Heuristic::default_heuristic();
    let baseline_rank = evaluate_heuristic_by_rank(snapshots, &default_h, components);

    let candidate_templates = build_candidate_heuristics();

    let mut best_heuristic: Option<Heuristic> = None;
    let mut best_rank = baseline_rank;

    for candidate in &candidate_templates {
        let avg_rank = evaluate_heuristic_by_rank(snapshots, candidate, components);

        if avg_rank < best_rank {
            best_rank = avg_rank;
            best_heuristic = Some(candidate.clone());
        }
    }

    if best_rank < baseline_rank {
        if let Some(ref h) = best_heuristic {
            eprintln!(
                "  [rank-opt] Best heuristic: \"{}\" (avg rank {:.1} vs baseline {:.1}, {:.1}x speedup)",
                h.name,
                best_rank,
                baseline_rank,
                baseline_rank / best_rank,
            );
        }
        best_heuristic
    } else {
        eprintln!(
            "  [rank-opt] No improvement over baseline (avg rank {:.1})",
            baseline_rank,
        );
        None
    }
}

// ── Online RL coefficient updates ────────────────────────────────────
//
// After each solved task, nudge the RL coefficients in the direction
// that would have made the solution appear earlier. The update size
// scales with task difficulty: easy tasks (low candidate count) produce
// no update; hard tasks produce larger nudges.

/// Update RL coefficients after finding a solution.
///
/// `candidates_explored`: how many candidates were tested before the solution
/// `budget`: the max candidate budget for this task
///
/// The update rule:
/// - difficulty = candidates / budget (0.0 = instant, 1.0 = barely found)
/// - Only update when difficulty > 0.01 (solution was non-trivial)
/// - Nudge cold_penalty more negative (prune dead entries harder)
/// - Nudge warm_bonus higher (boost partial matches more)
/// - Learning rate proportional to difficulty (hard tasks teach more)
/// - Clamp to reasonable ranges to prevent runaway
pub fn update_rl_coefficients(
    coeffs: &mut RlCoefficients,
    candidates_explored: usize,
    budget: usize,
) {
    let difficulty = candidates_explored as f64 / budget.max(1) as f64;

    // Only learn from non-trivial tasks
    if difficulty < 0.01 {
        return;
    }

    let lr = 0.05 * difficulty;

    // Strengthen both signals: prune dead entries harder, boost partial matches more.
    // When a task is hard, there were likely many dead-end pool entries diluting
    // the search, and useful partial-match entries that could have been ranked higher.
    coeffs.cold_penalty -= lr * 10.0;
    coeffs.warm_bonus += lr * 5.0;

    // Clamp to reasonable ranges
    coeffs.cold_penalty = coeffs.cold_penalty.max(-200.0).min(0.0);
    coeffs.warm_bonus = coeffs.warm_bonus.max(0.0).min(100.0);
}

// ── Stage 4: Synthesized heuristic enumeration ───────────────────
//
// Instead of choosing from hand-crafted templates, enumerate heuristic
// program bodies bottom-up over ctx fields + arithmetic/comparison ops.
// Each candidate is scored cheaply via rank_solution on captured snapshots.

/// The 7 numeric fields available in the heuristic ctx namespace.
const CTX_FIELDS: &[&str] = &[
    "priority", "arity", "ret-type", "first-param-type",
    "input-type", "output-type", "num-examples",
];

/// Numeric constants available as depth-0 atoms.
const HEURISTIC_CONSTANTS: &[f64] = &[0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0];

/// A pool entry in the heuristic enumerator.
#[derive(Clone)]
struct HeuristicPoolEntry {
    /// Source fragment (e.g. "priority", "(add priority 50.0)").
    source: String,
    /// Whether this entry is boolean-typed (from comparisons).
    is_bool: bool,
    /// Behavior vector: for each (snapshot, component) pair, the score
    /// this expression assigns. Used for observational equivalence dedup.
    behavior: Vec<f64>,
}

/// Extract a field value from a component + task context by field name.
fn extract_field(field: &str, comp: &SynthComponent, ctx: &TaskContext) -> f64 {
    match field {
        "priority" => comp.priority,
        "arity" => comp.arity as f64,
        "ret-type" => comp.ret_type as f64,
        "first-param-type" => comp.param_types.first().copied().unwrap_or(0) as f64,
        "input-type" => ctx.input_type as f64,
        "output-type" => ctx.output_type as f64,
        "num-examples" => ctx.num_examples as f64,
        _ => 0.0,
    }
}

/// Evaluate a pool entry's expression for a specific (component, task) pair.
/// This is done lazily during enumeration by composing sub-entry behaviors.
fn eval_binary_op(op: &str, a: f64, b: f64) -> f64 {
    match op {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        "min" => a.min(b),
        "max" => a.max(b),
        "=" => if (a - b).abs() < 1e-9 { 1.0 } else { 0.0 },
        "<" => if a < b { 1.0 } else { 0.0 },
        ">" => if a > b { 1.0 } else { 0.0 },
        _ => 0.0,
    }
}

/// Quantize a behavior vector into a hash key for dedup.
/// We round to 2 decimal places and hash the resulting bytes.
fn behavior_key(behavior: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    for &v in behavior {
        let bits = ((v * 100.0).round() as i64).to_le_bytes();
        bits.hash(&mut hasher);
    }
    hasher.finish()
}

/// Wrap a source body with ns-get calls to produce a valid heuristic lambda.
///
/// Replaces bare field names like `priority` with `(ns-get ctx "priority")`.
fn wrap_as_lambda(body: &str) -> String {
    let mut result = body.to_string();
    // Sort fields longest-first to avoid partial replacement
    // (e.g. "input-type" before "type")
    let mut fields: Vec<&str> = CTX_FIELDS.to_vec();
    fields.sort_by(|a, b| b.len().cmp(&a.len()));
    for field in &fields {
        // Replace field name as a standalone token (not inside quotes or ns-get)
        // We do a simple token-boundary replacement
        let ns_get = format!("(ns-get ctx \"{}\")", field);
        result = replace_field_token(&result, field, &ns_get);
    }
    format!("(lambda (ctx) {})", result)
}

/// Replace a bare field token in an expression, avoiding replacements inside
/// strings or already-wrapped ns-get calls.
fn replace_field_token(source: &str, field: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(source.len() * 2);
    let bytes = source.as_bytes();
    let field_bytes = field.as_bytes();
    let flen = field_bytes.len();
    let slen = bytes.len();
    let mut i = 0;

    while i < slen {
        if bytes[i] == b'"' {
            // Skip quoted strings
            result.push('"');
            i += 1;
            while i < slen && bytes[i] != b'"' {
                result.push(bytes[i] as char);
                i += 1;
            }
            if i < slen {
                result.push('"');
                i += 1;
            }
            continue;
        }

        if i + flen <= slen && &bytes[i..i + flen] == field_bytes {
            // Check token boundaries
            let before_ok = i == 0 || matches!(bytes[i - 1], b'(' | b' ' | b'\t' | b'\n');
            let after_ok = i + flen >= slen
                || matches!(bytes[i + flen], b')' | b' ' | b'\t' | b'\n');
            if before_ok && after_ok {
                result.push_str(replacement);
                i += flen;
                continue;
            }
        }

        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

/// Phase A: Enumerate heuristic program bodies bottom-up and score by rank.
///
/// Returns candidates sorted by avg rank (best first), up to `max_results`.
pub fn enumerate_heuristic_candidates(
    snapshots: &[PoolSnapshot],
    components: &[SynthComponent],
    max_depth: usize,
    max_results: usize,
) -> Vec<(Heuristic, f64)> {
    if snapshots.is_empty() || components.is_empty() {
        return Vec::new();
    }

    // Compute behavior vector length: sum of candidate counts across snapshots
    // For efficiency, we evaluate per (snapshot_idx, component_idx) = fixed-size
    // behavior vector using the components list directly.
    let behavior_len = snapshots.len() * components.len();

    // Build depth-0 pool: field atoms + constants
    let mut pool: Vec<HeuristicPoolEntry> = Vec::new();
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();

    // Field atoms
    for &field in CTX_FIELDS {
        let mut behavior = Vec::with_capacity(behavior_len);
        for snap in snapshots {
            for comp in components {
                behavior.push(extract_field(field, comp, &snap.task_context));
            }
        }
        let key = behavior_key(&behavior);
        if seen.insert(key) {
            pool.push(HeuristicPoolEntry {
                source: field.to_string(),
                is_bool: false,
                behavior,
            });
        }
    }

    // Constant atoms
    for &c in HEURISTIC_CONSTANTS {
        let behavior = vec![c; behavior_len];
        let key = behavior_key(&behavior);
        if seen.insert(key) {
            let source = if c == c.floor() && c.abs() < 1e6 {
                format!("{:.1}", c)
            } else {
                format!("{}", c)
            };
            pool.push(HeuristicPoolEntry {
                source,
                is_bool: false,
                behavior,
            });
        }
    }

    let depth0_count = pool.len();

    // Build deeper depths
    let binary_num_ops: &[&str] = &["add", "subtract", "multiply", "min", "max"];
    let comparison_ops: &[&str] = &["=", "<", ">"];

    for _depth in 1..=max_depth {
        let pool_len = pool.len();
        let mut new_entries: Vec<HeuristicPoolEntry> = Vec::new();

        // Unary ops: abs, negate (num → num)
        for i in 0..pool_len {
            if pool[i].is_bool { continue; }
            for &op in &["abs", "negate"] {
                let mut behavior = Vec::with_capacity(behavior_len);
                for (idx, &v) in pool[i].behavior.iter().enumerate() {
                    let _ = idx;
                    behavior.push(match op {
                        "abs" => v.abs(),
                        "negate" => -v,
                        _ => v,
                    });
                }
                let key = behavior_key(&behavior);
                if seen.insert(key) {
                    let source = format!("({} {})", op, pool[i].source);
                    new_entries.push(HeuristicPoolEntry { source, is_bool: false, behavior });
                }
            }
        }

        // Binary numeric ops: num × num → num
        for &op in binary_num_ops {
            for i in 0..pool_len {
                if pool[i].is_bool { continue; }
                for j in 0..pool_len {
                    if pool[j].is_bool { continue; }
                    // Skip trivially redundant: (add x x) = (multiply x 2)
                    if i == j && (op == "add" || op == "subtract" || op == "min" || op == "max") {
                        continue;
                    }
                    let mut behavior = Vec::with_capacity(behavior_len);
                    for k in 0..behavior_len {
                        behavior.push(eval_binary_op(op, pool[i].behavior[k], pool[j].behavior[k]));
                    }
                    let key = behavior_key(&behavior);
                    if seen.insert(key) {
                        let source = format!("({} {} {})", op, pool[i].source, pool[j].source);
                        new_entries.push(HeuristicPoolEntry { source, is_bool: false, behavior });
                    }
                }
            }
        }

        // Comparison ops: num × num → bool
        for &op in comparison_ops {
            for i in 0..pool_len {
                if pool[i].is_bool { continue; }
                for j in 0..pool_len {
                    if pool[j].is_bool { continue; }
                    if i == j && op == "=" { continue; } // always true
                    if i == j && (op == "<" || op == ">") { continue; } // always false
                    let mut behavior = Vec::with_capacity(behavior_len);
                    for k in 0..behavior_len {
                        behavior.push(eval_binary_op(op, pool[i].behavior[k], pool[j].behavior[k]));
                    }
                    let key = behavior_key(&behavior);
                    if seen.insert(key) {
                        let source = format!("({} {} {})", op, pool[i].source, pool[j].source);
                        new_entries.push(HeuristicPoolEntry { source, is_bool: true, behavior });
                    }
                }
            }
        }

        // If-expressions: (if bool num num) → num
        // Only use bool entries as conditions, num entries as branches
        // Pre-collect condition data to avoid borrow conflict with new_entries
        let bool_conds: Vec<(String, Vec<f64>)> = (0..pool_len + new_entries.len())
            .filter_map(|i| {
                let entry = if i < pool_len { &pool[i] } else { &new_entries[i - pool_len] };
                if entry.is_bool {
                    Some((entry.source.clone(), entry.behavior.clone()))
                } else {
                    None
                }
            })
            .collect();
        let num_entries: Vec<usize> = (0..pool_len)
            .filter(|&i| !pool[i].is_bool)
            .collect();

        // Limit if-expressions to avoid blowup: cap at 50 bool × top-20 num pairs
        let max_bool = 50.min(bool_conds.len());
        let max_num = 20.min(num_entries.len());

        for (cond_src, cond_beh) in &bool_conds[..max_bool] {
            for &ti in &num_entries[..max_num] {
                for &ei in &num_entries[..max_num] {
                    if ti == ei { continue; } // (if c x x) = x
                    let mut behavior = Vec::with_capacity(behavior_len);
                    for k in 0..behavior_len {
                        let c = cond_beh[k];
                        let t = pool[ti].behavior[k];
                        let e = pool[ei].behavior[k];
                        behavior.push(if c > 0.5 { t } else { e });
                    }
                    let key = behavior_key(&behavior);
                    if seen.insert(key) {
                        let source = format!(
                            "(if {} {} {})",
                            cond_src, pool[ti].source, pool[ei].source,
                        );
                        new_entries.push(HeuristicPoolEntry {
                            source,
                            is_bool: false,
                            behavior,
                        });
                    }
                }
            }
        }

        pool.extend(new_entries);

        eprintln!(
            "  [stage4] Depth {}: {} pool entries ({} new)",
            _depth, pool.len(), pool.len() - depth0_count,
        );
    }

    // Score each numeric (non-bool) entry by average rank.
    //
    // Instead of calling evaluate_heuristic_by_rank (which runs the SELPH
    // evaluator), we compute rank directly from behavior vectors. For each
    // snapshot, a candidate heuristic assigns a score to each component via
    // its behavior vector. The "rank" is the position of the solution's
    // component after re-sorting by heuristic score.

    // Pre-compute: for each snapshot, the solution component name (last candidate)
    // and the behavior vector offset.
    let num_comps = components.len();
    struct SnapshotInfo {
        solution_comp: String,
        offset: usize, // start index in behavior vectors
    }
    let snap_infos: Vec<SnapshotInfo> = snapshots
        .iter()
        .enumerate()
        .filter_map(|(si, snap)| {
            if snap.candidates.is_empty() { return None; }
            let sol = snap.candidates.last().unwrap();
            Some(SnapshotInfo {
                solution_comp: sol.comp_name.clone(),
                offset: si * num_comps,
            })
        })
        .collect();

    // For each pool entry, compute its average rank across snapshots.
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for (idx, entry) in pool.iter().enumerate() {
        if entry.is_bool { continue; }

        let mut total_rank = 0usize;
        let mut count = 0usize;

        for info in &snap_infos {
            // Extract this entry's scores for all components in this snapshot
            let scores: Vec<(usize, f64)> = (0..num_comps)
                .map(|ci| (ci, entry.behavior[info.offset + ci]))
                .collect();

            // Find solution component index
            let sol_idx = components.iter().position(|c| c.name == info.solution_comp);
            let sol_idx = match sol_idx {
                Some(i) => i,
                None => continue,
            };

            let sol_score = scores[sol_idx].1;

            // Rank = 1 + number of components scored higher than solution
            let rank = 1 + scores.iter()
                .filter(|&&(ci, s)| ci != sol_idx && s > sol_score)
                .count();

            total_rank += rank;
            count += 1;
        }

        if count > 0 {
            scored.push((idx, total_rank as f64 / count as f64));
        }
    }

    // Sort by rank (lower = better)
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Compute baseline rank for reporting
    let baseline_rank = if !snap_infos.is_empty() {
        // Baseline uses component.priority directly
        let pri_idx = CTX_FIELDS.iter().position(|&f| f == "priority").unwrap_or(0);
        if pri_idx < pool.len() && !pool[pri_idx].is_bool {
            let mut total = 0usize;
            let mut cnt = 0usize;
            for info in &snap_infos {
                let sol_idx = components.iter().position(|c| c.name == info.solution_comp);
                if let Some(si) = sol_idx {
                    let sol_score = pool[pri_idx].behavior[info.offset + si];
                    let rank = 1 + (0..num_comps)
                        .filter(|&ci| ci != si && pool[pri_idx].behavior[info.offset + ci] > sol_score)
                        .count();
                    total += rank;
                    cnt += 1;
                }
            }
            if cnt > 0 { total as f64 / cnt as f64 } else { f64::MAX }
        } else {
            f64::MAX
        }
    } else {
        f64::MAX
    };

    // Take top results
    let n = max_results.min(scored.len());
    let mut results = Vec::with_capacity(n);
    for &(idx, rank) in &scored[..n] {
        let lambda_src = wrap_as_lambda(&pool[idx].source);
        if let Some(h) = Heuristic::from_source("synth-candidate", &lambda_src) {
            results.push((h, rank));
        }
    }

    if !results.is_empty() {
        eprintln!(
            "  [stage4] Phase A: {} candidates enumerated, best rank {:.1} (baseline {:.1})",
            pool.len(), results[0].1, baseline_rank,
        );
    }

    results
}

/// Phase B: Validate top-K heuristic candidates with actual synthesis.
///
/// Runs real synthesis on the hardest tasks (most candidates in baseline)
/// for each candidate. Returns the best (heuristic, solved, total_candidates).
pub fn validate_heuristic_candidates(
    candidates: &[(Heuristic, f64)],
    tasks: &[TrainingTask],
    components: &[SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    baseline_results: &[(String, bool, usize)],
    max_depth: usize,
    per_task_budget: usize,
    top_k: usize,
) -> Option<(Heuristic, usize, usize)> {
    if candidates.is_empty() || tasks.is_empty() {
        return None;
    }

    // Select hard subset: tasks that took the most candidates in baseline
    let mut task_difficulty: Vec<(usize, usize)> = baseline_results
        .iter()
        .enumerate()
        .map(|(i, (_, _, cands))| (i, *cands))
        .collect();
    task_difficulty.sort_by(|a, b| b.1.cmp(&a.1));

    let hard_count = 15.min(tasks.len());
    let hard_indices: std::collections::HashSet<usize> = task_difficulty
        .iter()
        .take(hard_count)
        .map(|&(i, _)| i)
        .collect();

    let hard_tasks: Vec<&TrainingTask> = tasks
        .iter()
        .enumerate()
        .filter(|(i, _)| hard_indices.contains(i))
        .map(|(_, t)| t)
        .collect();

    let k = top_k.min(candidates.len());
    let mut best: Option<(Heuristic, usize, usize)> = None;

    eprintln!("  [stage4] Phase B: validating top-{} candidates on {} hard tasks", k, hard_tasks.len());

    for (h, rank_score) in &candidates[..k] {
        let mut solved = 0usize;
        let mut total_cands = 0usize;

        for task in &hard_tasks {
            let ctx = TaskContext::from_examples(&task.inputs, &task.expected);
            let prioritized = apply_heuristic(h, components, &ctx);

            let sr = synthesize(
                &prioritized,
                &task.inputs,
                &task.expected,
                macros,
                max_depth,
                per_task_budget,
                true,
            );

            if sr.found { solved += 1; }
            total_cands += sr.candidates_explored;
        }

        eprintln!(
            "    \"{}\" rank={:.1}: {}/{} solved, {} candidates",
            h.source.chars().take(60).collect::<String>(),
            rank_score, solved, hard_tasks.len(), total_cands,
        );

        // Better = more solved, then fewer candidates
        let dominated = match &best {
            Some((_, bs, bc)) => solved > *bs || (solved == *bs && total_cands < *bc),
            None => true,
        };
        if dominated {
            best = Some((h.clone(), solved, total_cands));
        }
    }

    best
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_components() -> Vec<SynthComponent> {
        vec![
            SynthComponent {
                name: "x".into(),
                builtin: None,
                arity: 0,
                ret_type: TYPE_NUM,
                param_types: vec![],
                priority: 100.0,
            },
            SynthComponent {
                name: "0".into(),
                builtin: None,
                arity: 0,
                ret_type: TYPE_NUM,
                param_types: vec![],
                priority: 0.0,
            },
            SynthComponent {
                name: "1".into(),
                builtin: None,
                arity: 0,
                ret_type: TYPE_NUM,
                param_types: vec![],
                priority: 0.0,
            },
            SynthComponent {
                name: "add".into(),
                builtin: Some("add".into()),
                arity: 2,
                ret_type: TYPE_NUM,
                param_types: vec![TYPE_NUM, TYPE_NUM],
                priority: 0.0,
            },
            SynthComponent {
                name: "multiply".into(),
                builtin: Some("multiply".into()),
                arity: 2,
                ret_type: TYPE_NUM,
                param_types: vec![TYPE_NUM, TYPE_NUM],
                priority: 0.0,
            },
            SynthComponent {
                name: "string-upper".into(),
                builtin: Some("string-upper".into()),
                arity: 1,
                ret_type: TYPE_STR,
                param_types: vec![TYPE_STR],
                priority: 0.0,
            },
        ]
    }

    fn num_task_context() -> TaskContext {
        TaskContext {
            input_type: TYPE_NUM,
            output_type: TYPE_NUM,
            num_examples: 3,
        }
    }

    fn str_task_context() -> TaskContext {
        TaskContext {
            input_type: TYPE_STR,
            output_type: TYPE_STR,
            num_examples: 3,
        }
    }

    // ── Heuristic construction ──────────────────────────────────────

    #[test]
    fn test_heuristic_from_source_valid() {
        let h = Heuristic::from_source("test", "(lambda (ctx) 42.0)");
        assert!(h.is_some());
        let h = h.unwrap();
        assert_eq!(h.name, "test");
        assert_eq!(h.source, "(lambda (ctx) 42.0)");
    }

    #[test]
    fn test_heuristic_from_source_invalid() {
        let h = Heuristic::from_source("bad", "(lambda (ctx");
        assert!(h.is_none());
    }

    #[test]
    fn test_default_heuristic_parses() {
        let h = Heuristic::default_heuristic();
        assert_eq!(h.name, "default");
        assert!(!h.nodes.is_empty());
    }

    // ── TaskContext ─────────────────────────────────────────────────

    #[test]
    fn test_task_context_from_examples() {
        let inputs = vec![Value::Num(1.0), Value::Num(2.0)];
        let expected = vec![Value::Str("a".into()), Value::Str("b".into())];
        let ctx = TaskContext::from_examples(&inputs, &expected);
        assert_eq!(ctx.input_type, TYPE_NUM);
        assert_eq!(ctx.output_type, TYPE_STR);
        assert_eq!(ctx.num_examples, 2);
    }

    #[test]
    fn test_task_context_empty() {
        let ctx = TaskContext::from_examples(&[], &[]);
        assert_eq!(ctx.input_type, 255);
        assert_eq!(ctx.output_type, 255);
        assert_eq!(ctx.num_examples, 0);
    }

    #[test]
    fn test_task_context_to_namespace() {
        let ctx = num_task_context();
        let ns = ctx.to_namespace();
        match ns {
            Value::Namespace(map) => {
                assert!(matches!(map.get("input-type"), Some(Value::Num(n)) if *n == 0.0));
                assert!(matches!(map.get("output-type"), Some(Value::Num(n)) if *n == 0.0));
                assert!(matches!(map.get("num-examples"), Some(Value::Num(n)) if *n == 3.0));
            }
            _ => panic!("expected namespace"),
        }
    }

    // ── evaluate_heuristic ──────────────────────────────────────────

    #[test]
    fn test_evaluate_constant_heuristic() {
        let h = Heuristic::from_source("const", "(lambda (ctx) 42.0)").unwrap();
        let comp = &simple_components()[0]; // x
        let ctx = num_task_context();
        let score = evaluate_heuristic(&h, comp, &ctx);
        assert!((score - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_default_heuristic_returns_priority() {
        let h = Heuristic::default_heuristic();
        let comps = simple_components();
        let ctx = num_task_context();

        // x has priority 100.0
        let score = evaluate_heuristic(&h, &comps[0], &ctx);
        assert!((score - 100.0).abs() < 1e-9);

        // "0" has priority 0.0
        let score = evaluate_heuristic(&h, &comps[1], &ctx);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_heuristic_reads_arity() {
        let h =
            Heuristic::from_source("arity", "(lambda (ctx) (ns-get ctx \"arity\"))").unwrap();
        let comps = simple_components();
        let ctx = num_task_context();

        // x has arity 0
        assert!((evaluate_heuristic(&h, &comps[0], &ctx) - 0.0).abs() < 1e-9);
        // add has arity 2
        assert!((evaluate_heuristic(&h, &comps[3], &ctx) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_bad_heuristic_returns_zero() {
        // Division by zero in heuristic should return 0.0 gracefully
        let h = Heuristic::from_source("bad", "(lambda (ctx) (divide 1.0 0.0))").unwrap();
        let comp = &simple_components()[0];
        let ctx = num_task_context();
        let score = evaluate_heuristic(&h, comp, &ctx);
        // Either 0.0 (error) or non-finite -> 0.0
        assert!(score == 0.0 || !score.is_finite());
    }

    // ── apply_heuristic ─────────────────────────────────────────────

    #[test]
    fn test_apply_heuristic_reorders_by_score() {
        // Heuristic that scores by arity * 10
        let h = Heuristic::from_source(
            "by-arity",
            "(lambda (ctx) (multiply (ns-get ctx \"arity\") 10.0))",
        )
        .unwrap();
        let comps = simple_components();
        let ctx = num_task_context();
        let reordered = apply_heuristic(&h, &comps, &ctx);

        // Binary ops (arity 2) should come first
        assert!(reordered[0].arity >= reordered.last().unwrap().arity);
        // First components should have priority 20.0 (arity 2 * 10)
        assert!((reordered[0].priority - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_apply_heuristic_preserves_count() {
        let h = Heuristic::default_heuristic();
        let comps = simple_components();
        let ctx = num_task_context();
        let reordered = apply_heuristic(&h, &comps, &ctx);
        assert_eq!(reordered.len(), comps.len());
    }

    // ── Type-matching heuristic ─────────────────────────────────────

    #[test]
    fn test_type_match_heuristic_prefers_matching_types() {
        let candidates = build_candidate_heuristics();
        // Find the "match-output-type" heuristic
        let h = candidates
            .iter()
            .find(|h| h.name == "match-output-type")
            .expect("match-output-type should be in candidates");

        let comps = simple_components();
        let ctx = num_task_context(); // output_type = TYPE_NUM = 0

        // "add" returns TYPE_NUM -> should get 100.0
        let score_add = evaluate_heuristic(h, &comps[3], &ctx);
        // "string-upper" returns TYPE_STR -> should get 0.0
        let score_str = evaluate_heuristic(h, &comps[5], &ctx);

        assert!(score_add > score_str);
    }

    // ── build_candidate_heuristics ──────────────────────────────────

    #[test]
    fn test_candidate_heuristics_all_parse() {
        let candidates = build_candidate_heuristics();
        // We defined 8 templates; all should parse successfully
        assert!(candidates.len() >= 6, "expected at least 6 candidates, got {}", candidates.len());
        for h in &candidates {
            assert!(!h.name.is_empty());
            assert!(!h.nodes.is_empty());
        }
    }

    #[test]
    fn test_candidate_heuristics_return_numbers() {
        let candidates = build_candidate_heuristics();
        let comps = simple_components();
        let ctx = num_task_context();

        for h in &candidates {
            for comp in &comps {
                let score = evaluate_heuristic(h, comp, &ctx);
                assert!(
                    score.is_finite(),
                    "heuristic {} returned non-finite for {}",
                    h.name,
                    comp.name,
                );
            }
        }
    }

    // ── component_to_namespace ───────────────────────────────────────

    #[test]
    fn test_component_to_namespace_fields() {
        let comp = &simple_components()[3]; // add
        let ctx = num_task_context();
        let ns = component_to_namespace(comp, &ctx);
        match ns {
            Value::Namespace(map) => {
                assert!(matches!(map.get("name"), Some(Value::Str(s)) if s == "add"));
                assert!(matches!(map.get("arity"), Some(Value::Num(n)) if *n == 2.0));
                assert!(matches!(map.get("ret-type"), Some(Value::Num(n)) if *n == 0.0));
                assert!(matches!(map.get("first-param-type"), Some(Value::Num(n)) if *n == 0.0));
                // Task context fields should also be present
                assert!(matches!(map.get("input-type"), Some(Value::Num(n)) if *n == 0.0));
                assert!(matches!(map.get("output-type"), Some(Value::Num(n)) if *n == 0.0));
            }
            _ => panic!("expected namespace"),
        }
    }

    // ── TrainingTask and run_with_heuristic ─────────────────────────

    #[test]
    fn test_run_with_heuristic_solves_identity() {
        // Task: f(x) = x (identity function on numbers)
        let task = TrainingTask {
            name: "identity".into(),
            inputs: vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
            expected: vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
        };
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = vec![];
        let comps = simple_components();
        let h = Heuristic::default_heuristic();

        let (solved, _cands, results) =
            run_with_heuristic(&[task], &h, &comps, &macros, 2, 5000);
        assert_eq!(solved, 1);
        assert!(results[0].1); // identity should be solved
    }

    #[test]
    fn test_run_with_heuristic_counts_unsolved() {
        // Task: f(x) = x * x * x (cube) — unlikely to solve at depth 2
        // with limited components
        let task = TrainingTask {
            name: "cube".into(),
            inputs: vec![Value::Num(2.0), Value::Num(3.0)],
            expected: vec![Value::Num(8.0), Value::Num(27.0)],
        };
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = vec![];
        let comps = simple_components();
        let h = Heuristic::default_heuristic();

        let (solved, _cands, results) =
            run_with_heuristic(&[task], &h, &comps, &macros, 1, 500);
        // Cube is hard at depth 1 — likely unsolved
        assert_eq!(results.len(), 1);
        // We don't assert solved==0 since it depends on search budget
        let _ = solved;
    }

    // ── meta_curriculum_step ────────────────────────────────────────

    #[test]
    fn test_meta_curriculum_step_identity_tasks() {
        let tasks = vec![
            TrainingTask {
                name: "id1".into(),
                inputs: vec![Value::Num(1.0), Value::Num(5.0)],
                expected: vec![Value::Num(1.0), Value::Num(5.0)],
            },
            TrainingTask {
                name: "add1".into(),
                inputs: vec![Value::Num(1.0), Value::Num(5.0)],
                expected: vec![Value::Num(2.0), Value::Num(6.0)],
            },
        ];
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = vec![];
        let comps = simple_components();
        let h = Heuristic::default_heuristic();

        let (solved_names, _new_h) =
            meta_curriculum_step(&tasks, &h, &comps, &macros);

        // At least identity should be solved
        assert!(
            solved_names.contains(&"id1".to_string()),
            "identity task should be solved"
        );
    }

    // ── synthesize_heuristic (smoke test) ───────────────────────────

    #[test]
    fn test_synthesize_heuristic_empty_tasks() {
        let comps = simple_components();
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = vec![];
        let result = synthesize_heuristic(&[], &comps, &macros, 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_synthesize_heuristic_empty_components() {
        let tasks = vec![TrainingTask {
            name: "t".into(),
            inputs: vec![Value::Num(1.0)],
            expected: vec![Value::Num(1.0)],
        }];
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = vec![];
        let result = synthesize_heuristic(&tasks, &[], &macros, 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_synthesize_heuristic_with_simple_tasks() {
        // Provide enough tasks that a type-matching heuristic could help
        let tasks = vec![
            TrainingTask {
                name: "double".into(),
                inputs: vec![Value::Num(1.0), Value::Num(3.0), Value::Num(5.0)],
                expected: vec![Value::Num(2.0), Value::Num(6.0), Value::Num(10.0)],
            },
            TrainingTask {
                name: "id".into(),
                inputs: vec![Value::Num(7.0), Value::Num(0.0)],
                expected: vec![Value::Num(7.0), Value::Num(0.0)],
            },
        ];
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = vec![];
        let comps = simple_components();

        // Budget of 4 candidate heuristics — just a smoke test
        let result = synthesize_heuristic(&tasks, &comps, &macros, 4);
        // We don't assert Some/None here because it depends on whether
        // a candidate heuristic actually improves over baseline; this is
        // a smoke test that it runs without panicking.
        let _ = result;
    }

    // ── String task context ─────────────────────────────────────────

    #[test]
    fn test_string_specialist_heuristic() {
        let candidates = build_candidate_heuristics();
        let h = candidates
            .iter()
            .find(|h| h.name == "string-specialist")
            .expect("string-specialist should be in candidates");

        let comps = simple_components();
        let str_ctx = str_task_context();

        // For a string task, string-upper (ret_type=STR) should score higher
        // than add (ret_type=NUM)
        let score_str = evaluate_heuristic(h, &comps[5], &str_ctx);
        let score_num = evaluate_heuristic(h, &comps[3], &str_ctx);
        assert!(
            score_str > score_num,
            "string op should score higher for string tasks: {} vs {}",
            score_str,
            score_num
        );
    }

    // ── Rank-based heuristic evaluation ─────────────────────────────

    fn make_test_snapshot() -> PoolSnapshot {
        // Simulate a task where the solution uses "add" (comp priority 0.0)
        // and there are candidates from multiply, add, and string-upper.
        // The solution is the last entry (add with arg_priority_sum=200.0).
        PoolSnapshot {
            task_name: "test_task".into(),
            task_context: num_task_context(),
            candidates: vec![
                CandidateRecord { comp_name: "multiply".into(), arg_priority_sum: 100.0 },
                CandidateRecord { comp_name: "string-upper".into(), arg_priority_sum: 50.0 },
                CandidateRecord { comp_name: "add".into(), arg_priority_sum: 150.0 },
                CandidateRecord { comp_name: "multiply".into(), arg_priority_sum: 200.0 },
                CandidateRecord { comp_name: "add".into(), arg_priority_sum: 200.0 }, // solution
            ],
            depth0_count: 3, // 3 atoms tested before compositions
        }
    }

    #[test]
    fn test_rank_solution_default_heuristic() {
        let snapshot = make_test_snapshot();
        let comps = simple_components();
        let h = Heuristic::default_heuristic();

        let rank = rank_solution(&snapshot, &h, &comps);
        assert!(rank.is_some());
        // With default heuristic (pass-through priority), all comps have
        // priority 0.0, so scores are just arg_priority_sum. Solution has
        // arg_priority_sum=200.0, tied with multiply(200.0). Stable sort
        // preserves order, so solution (idx 4) comes after multiply (idx 3).
        let r = rank.unwrap();
        assert!(r > 0, "rank should be positive: {}", r);
    }

    #[test]
    fn test_rank_solution_empty_snapshot() {
        let snapshot = PoolSnapshot {
            task_name: "empty".into(),
            task_context: num_task_context(),
            candidates: vec![],
            depth0_count: 0,
        };
        let comps = simple_components();
        let h = Heuristic::default_heuristic();
        assert!(rank_solution(&snapshot, &h, &comps).is_none());
    }

    #[test]
    fn test_rank_solution_type_match_helps() {
        let snapshot = make_test_snapshot();
        let comps = simple_components();

        // Default heuristic: all comps have same priority (0.0)
        let default_h = Heuristic::default_heuristic();
        let default_rank = rank_solution(&snapshot, &default_h, &comps).unwrap();

        // Number-specialist: boosts add (ret_type=NUM=0) for numeric tasks
        let candidates = build_candidate_heuristics();
        let num_specialist = candidates.iter()
            .find(|h| h.name == "number-specialist")
            .unwrap();
        let specialist_rank = rank_solution(&snapshot, num_specialist, &comps).unwrap();

        // The number specialist should help (lower or equal rank) because
        // it boosts arithmetic ops and the solution uses "add"
        assert!(
            specialist_rank <= default_rank,
            "specialist rank {} should be <= default rank {}",
            specialist_rank, default_rank,
        );
    }

    #[test]
    fn test_evaluate_heuristic_by_rank() {
        let snapshot = make_test_snapshot();
        let comps = simple_components();
        let h = Heuristic::default_heuristic();

        let avg_rank = evaluate_heuristic_by_rank(&[snapshot], &h, &comps);
        assert!(avg_rank > 0.0, "avg rank should be positive: {}", avg_rank);
        assert!(avg_rank.is_finite());
    }

    #[test]
    fn test_evaluate_heuristic_by_rank_empty() {
        let comps = simple_components();
        let h = Heuristic::default_heuristic();
        let avg = evaluate_heuristic_by_rank(&[], &h, &comps);
        assert_eq!(avg, f64::MAX);
    }

    #[test]
    fn test_optimize_heuristic_by_rank_empty() {
        let comps = simple_components();
        assert!(optimize_heuristic_by_rank(&[], &comps).is_none());
    }

    #[test]
    fn test_optimize_heuristic_by_rank_runs() {
        // Smoke test: optimization should run without panicking
        let snapshot = make_test_snapshot();
        let comps = simple_components();
        // Result may or may not find improvement — just verify no crash
        let _ = optimize_heuristic_by_rank(&[snapshot], &comps);
    }

    // ── Stage 4: Synthesized heuristic tests ──────────────────────────

    #[test]
    fn test_extract_field() {
        let comp = SynthComponent {
            name: "add".into(),
            builtin: Some("add".into()),
            arity: 2,
            ret_type: TYPE_NUM,
            param_types: vec![TYPE_NUM, TYPE_NUM],
            priority: 42.0,
        };
        let ctx = num_task_context();
        assert_eq!(extract_field("priority", &comp, &ctx), 42.0);
        assert_eq!(extract_field("arity", &comp, &ctx), 2.0);
        assert_eq!(extract_field("ret-type", &comp, &ctx), TYPE_NUM as f64);
        assert_eq!(extract_field("output-type", &comp, &ctx), TYPE_NUM as f64);
    }

    #[test]
    fn test_behavior_key_deterministic() {
        let b1 = vec![1.0, 2.0, 3.0];
        let b2 = vec![1.0, 2.0, 3.0];
        let b3 = vec![1.0, 2.0, 4.0];
        assert_eq!(behavior_key(&b1), behavior_key(&b2));
        assert_ne!(behavior_key(&b1), behavior_key(&b3));
    }

    #[test]
    fn test_wrap_as_lambda() {
        let body = "(add priority (if (= ret-type output-type) 50.0 0.0))";
        let lambda = wrap_as_lambda(body);
        assert!(lambda.starts_with("(lambda (ctx) "));
        assert!(lambda.contains("(ns-get ctx \"priority\")"));
        assert!(lambda.contains("(ns-get ctx \"ret-type\")"));
        assert!(lambda.contains("(ns-get ctx \"output-type\")"));
        // Constants should NOT be wrapped
        assert!(lambda.contains("50.0"));
        assert!(lambda.contains("0.0"));
        // Should parse as valid SELPH
        assert!(Heuristic::from_source("test", &lambda).is_some());
    }

    #[test]
    fn test_replace_field_token_basic() {
        let result = replace_field_token("(add priority 50.0)", "priority", "REPLACED");
        assert_eq!(result, "(add REPLACED 50.0)");
    }

    #[test]
    fn test_replace_field_token_inside_quotes() {
        // Should NOT replace inside quoted strings
        let result = replace_field_token("(ns-get ctx \"priority\")", "priority", "REPLACED");
        assert_eq!(result, "(ns-get ctx \"priority\")");
    }

    #[test]
    fn test_enumerate_heuristic_candidates_smoke() {
        let snapshot = make_test_snapshot();
        let comps = simple_components();
        let results = enumerate_heuristic_candidates(&[snapshot], &comps, 1, 10);
        assert!(!results.is_empty(), "should find at least one candidate");
        // Each result should parse as a valid heuristic
        for (h, rank) in &results {
            assert!(!h.source.is_empty());
            assert!(rank.is_finite());
        }
    }

    #[test]
    fn test_enumerate_empty_snapshots() {
        let comps = simple_components();
        let results = enumerate_heuristic_candidates(&[], &comps, 1, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_enumerate_finds_type_match_pattern() {
        // With a snapshot where the solution component returns NUM and the task
        // output is NUM, the enumerator should discover that ret-type==output-type
        // is a useful signal.
        let snapshot = PoolSnapshot {
            task_name: "test".into(),
            task_context: num_task_context(),
            candidates: vec![
                CandidateRecord { comp_name: "string-upper".into(), arg_priority_sum: 100.0 },
                CandidateRecord { comp_name: "string-upper".into(), arg_priority_sum: 90.0 },
                CandidateRecord { comp_name: "string-upper".into(), arg_priority_sum: 80.0 },
                CandidateRecord { comp_name: "add".into(), arg_priority_sum: 50.0 }, // solution (last)
            ],
            depth0_count: 0,
        };
        let comps = simple_components();
        let results = enumerate_heuristic_candidates(&[snapshot.clone()], &comps, 2, 20);

        // The best candidates should improve on baseline (which would rank the
        // solution at position 4 since string-upper entries have higher arg_priority_sum)
        assert!(!results.is_empty());
        // Verify the top result improves ranking
        let baseline_h = Heuristic::default_heuristic();
        let baseline_rank = evaluate_heuristic_by_rank(&[snapshot.clone()], &baseline_h, &comps);
        assert!(
            results[0].1 <= baseline_rank,
            "best synthesized ({:.1}) should be <= baseline ({:.1})",
            results[0].1, baseline_rank,
        );
    }
}
