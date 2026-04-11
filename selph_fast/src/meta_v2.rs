//! meta_v2 — heuristic API for the v2 synthesis path.
//!
//! Step-6 follow-on (§9.31). Ports the four pieces of `meta.rs` that
//! `cmd_curriculum` needs to re-enable `--heuristic` after the §9.30
//! migration: `Heuristic`, `TaskContext`, `evaluate_heuristic`, and
//! `apply_heuristic`. Operates on `synth_v2::SynthComponent` and
//! `types_v2::Value`.
//!
//! What's intentionally NOT ported here:
//!   - `synthesize_heuristic`, `meta_curriculum_step`,
//!     `enumerate_heuristic_candidates`, `evaluate_heuristic_configs`,
//!     `validate_heuristic_candidates` — all used only by `cmd_meta_optimize`,
//!     which still runs through legacy `synth.rs::synthesize` and is its own
//!     future sub-iteration of step 6.
//!   - `update_rl_coefficients` — already in `meta.rs`, type-clean (operates
//!     only on `synth::RlCoefficients` which is a struct of f64s). Kept there
//!     until `cmd_meta_optimize` is ported, then both move together.
//!
//! Heuristic context namespace fields exposed to programs:
//!   - `name`              — component name (string)
//!   - `arity`             — number of args (number)
//!   - `ret-type`          — Sym→u32→f64 of the return type
//!   - `first-param-type`  — Sym→u32→f64 of the first parameter type
//!   - `priority`          — base priority (number)
//!   - `usage-count`       — how many prior tasks used this component (number)
//!   - `target-type`       — Sym→u32→f64 of the inferred task output type
//!   - `input-type`        — Sym→u32→f64 of the inferred task input type
//!   - `output-type`       — alias for target-type (kept for legacy heuristic compatibility)
//!   - `num-examples`      — example count (number)
//!   - `avg-input-len`     — mean string-input length (number; 0 for non-strings)
//!   - `has-spaces`        — fraction of string inputs with spaces (number)
//!   - `max-num-value`     — max numeric value seen in inputs (number)
//!   - `output-is-bool`    — 1.0 if all outputs are bool, else 0.0
//!   - `num-distinct-outputs` — distinct output value count (number)
//!
//! The legacy v1 namespace exposed type fields as small u8 codes; the v2
//! version exposes them as Sym→u32 cast to f64. The semantics for the
//! priority-plus-type-match heuristic are preserved: types compare equal iff
//! they reference the same Sym, which means iff they were interned from the
//! same string. Heuristics that did `(= ret-type target-type)` work without
//! changes — only the absolute numeric values differ from v1.

use std::collections::HashSet;
use std::rc::Rc;

use crate::eval_v2;
use crate::intern::{intern, Sym};
use crate::parser;
use crate::synth_v2::{infer_uniform_type_sym, type_sym_or_any, val_hash, SynthComponent};
use crate::types_v2::{self, type_any, Env, NsMap, Value};

// ── Heuristic ───────────────────────────────────────────────────────────────

/// A learned (or hand-written) priority heuristic, expressed as a SELPH
/// `(lambda (ctx) ...)` program. Mirrors `meta::Heuristic` but holds the
/// converted v2 node tree directly so each `evaluate_heuristic` call avoids
/// re-parsing.
#[derive(Clone, Debug)]
pub struct Heuristic {
    pub name: String,
    pub source: String,
    /// Parsed and v2-converted nodes. The lambda root is at `root`.
    pub nodes: Rc<[types_v2::Node]>,
    pub root: usize,
}

impl Heuristic {
    /// Parse a heuristic from source. The source must be a single
    /// expression (typically `(lambda (ctx) body)` or
    /// `(define name (lambda (ctx) body))`). Returns `None` on parse error.
    pub fn from_source(name: &str, source: &str) -> Option<Self> {
        let (old_nodes, root) = parser::parse_source(source).ok()?;
        let new_nodes = eval_v2::convert_tree(&old_nodes);
        let nodes_rc: Rc<[types_v2::Node]> = new_nodes.into();
        Some(Heuristic {
            name: name.to_string(),
            source: source.to_string(),
            nodes: nodes_rc,
            root,
        })
    }

    /// The default identity heuristic: returns the component's existing
    /// `priority` field unchanged. Used as the starting point before any
    /// meta-learning.
    pub fn default_heuristic() -> Self {
        Heuristic::from_source("default", "(lambda (ctx) (ns-get ctx \"priority\"))")
            .expect("default heuristic source must parse")
    }
}

// ── Task context ────────────────────────────────────────────────────────────

/// Context features describing the current synthesis task. Computed once per
/// task and passed (via `component_to_namespace`) to every heuristic
/// evaluation, so the heuristic can make task-dependent priority decisions.
#[derive(Clone, Debug)]
pub struct TaskContext {
    pub input_type_sym: Sym,
    pub output_type_sym: Sym,
    pub num_examples: usize,
    pub avg_input_len: f64,
    pub has_spaces: f64,
    pub max_num_value: f64,
    pub output_is_bool: f64,
    pub num_distinct_outputs: f64,
}

impl TaskContext {
    /// Build a TaskContext from a task's example pairs. Mirrors
    /// `meta::TaskContext::from_examples` but operates on `types_v2::Value`
    /// and uses `Sym`-based type tags.
    pub fn from_examples(inputs: &[Value], expected: &[Value]) -> Self {
        let input_type_sym = inputs
            .first()
            .map(type_sym_or_any)
            .unwrap_or_else(type_any);
        let output_type_sym = expected
            .first()
            .map(type_sym_or_any)
            .unwrap_or_else(type_any);

        // Rich features computed from actual example content. The legacy
        // version (meta.rs::TaskContext::from_examples) only inspected the
        // first input for some of these; this version walks all inputs to
        // get a more reliable signal — same shape, same fields.
        let mut total_str_len: f64 = 0.0;
        let mut str_count: usize = 0;
        let mut space_count: usize = 0;
        let mut max_num: f64 = 0.0;
        let mut has_num = false;

        for inp in inputs {
            match inp {
                Value::Str(s) => {
                    total_str_len += s.len() as f64;
                    str_count += 1;
                    if s.contains(' ') {
                        space_count += 1;
                    }
                }
                Value::Int(n) => {
                    let f = *n as f64;
                    if !has_num || f > max_num {
                        max_num = f;
                    }
                    has_num = true;
                }
                Value::Num(n) => {
                    if !has_num || *n > max_num {
                        max_num = *n;
                    }
                    has_num = true;
                }
                Value::List(items) => {
                    for item in items.iter() {
                        match item {
                            Value::Int(n) => {
                                let f = *n as f64;
                                if !has_num || f > max_num {
                                    max_num = f;
                                }
                                has_num = true;
                            }
                            Value::Num(n) => {
                                if !has_num || *n > max_num {
                                    max_num = *n;
                                }
                                has_num = true;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        let avg_input_len = if str_count > 0 {
            total_str_len / str_count as f64
        } else {
            0.0
        };
        let has_spaces = if str_count > 0 {
            space_count as f64 / str_count as f64
        } else {
            0.0
        };
        let max_num_value = if has_num { max_num } else { 0.0 };

        let output_is_bool = if !expected.is_empty()
            && expected.iter().all(|v| matches!(v, Value::Bool(_)))
        {
            1.0
        } else {
            0.0
        };

        let mut distinct: HashSet<u64> = HashSet::new();
        for v in expected {
            distinct.insert(val_hash(v));
        }
        let num_distinct_outputs = distinct.len() as f64;

        TaskContext {
            input_type_sym,
            output_type_sym,
            num_examples: inputs.len(),
            avg_input_len,
            has_spaces,
            max_num_value,
            output_is_bool,
            num_distinct_outputs,
        }
    }
}

/// Pack a SynthComponent + TaskContext into a namespace Value the heuristic
/// program can inspect via `(ns-get ctx "<key>")`. The target type is the
/// caller-supplied desired output type — typically the inferred output type
/// of the task spec, the same as `task_context.output_type_sym`. Kept as a
/// separate parameter so future synth_v2 sub-tasks (HO inner-lambda
/// specs, RD intermediate goals) can pass a different target.
pub(crate) fn component_to_namespace(
    comp: &SynthComponent,
    task_ctx: &TaskContext,
    target_type: Sym,
) -> Value {
    let mut map = NsMap::with_capacity(16);
    let f = |s: Sym| Value::Num(s.0 as f64);

    map.insert(intern("name"), Value::str(&comp.name));
    map.insert(intern("arity"), Value::Num(comp.arity as f64));
    map.insert(intern("ret-type"), f(comp.ret_type));
    map.insert(intern("priority"), Value::Num(comp.priority));
    map.insert(intern("usage-count"), Value::Num(comp.usage_count));
    map.insert(
        intern("first-param-type"),
        f(comp.param_types.first().copied().unwrap_or_else(type_any)),
    );

    map.insert(intern("input-type"), f(task_ctx.input_type_sym));
    map.insert(intern("output-type"), f(task_ctx.output_type_sym));
    map.insert(intern("target-type"), f(target_type));
    map.insert(
        intern("num-examples"),
        Value::Num(task_ctx.num_examples as f64),
    );
    map.insert(intern("avg-input-len"), Value::Num(task_ctx.avg_input_len));
    map.insert(intern("has-spaces"), Value::Num(task_ctx.has_spaces));
    map.insert(intern("max-num-value"), Value::Num(task_ctx.max_num_value));
    map.insert(intern("output-is-bool"), Value::Num(task_ctx.output_is_bool));
    map.insert(
        intern("num-distinct-outputs"),
        Value::Num(task_ctx.num_distinct_outputs),
    );

    Value::ns(map)
}

// ── Evaluate heuristic ──────────────────────────────────────────────────────

/// Run the heuristic program against one component, returning the numeric
/// score (or 0.0 on any failure: parse error in the heuristic body, eval
/// error during apply, non-numeric return). Keeping the failure mode silent
/// matches `meta::evaluate_heuristic` — heuristics are best-effort scoring
/// hints, not part of the correctness path.
pub fn evaluate_heuristic(
    heuristic: &Heuristic,
    component: &SynthComponent,
    task_context: &TaskContext,
    target_type: Sym,
    env: &Env,
) -> f64 {
    let ctx_val = component_to_namespace(component, task_context, target_type);

    let fn_val = match eval_v2::eval(&heuristic.nodes, heuristic.root, env) {
        Ok(v) => v,
        Err(_) => return 0.0,
    };

    match eval_v2::apply(&fn_val, &[ctx_val], env) {
        Ok(Value::Num(n)) if n.is_finite() => n,
        Ok(Value::Int(n)) => n as f64,
        _ => 0.0,
    }
}

// ── Apply heuristic ─────────────────────────────────────────────────────────

/// Score every component in `components` with the heuristic, replace each
/// component's `priority` field with the score, and re-sort the result by
/// descending priority. The synth_v2 enumerator already factors `comp.priority`
/// into its candidate scoring (see `synth_v2::synthesize` around the
/// `pending.sort_by(|a, b| b.score...)` step), so the only required hook is
/// to mutate the catalog before calling `synthesize_with_strategies`.
///
/// `target_type` should be the task's inferred output type (same Sym that
/// `synth_v2::infer_uniform_type_sym(expected)` returns); the caller computes
/// it once per task.
pub fn apply_heuristic(
    heuristic: &Heuristic,
    components: &[SynthComponent],
    task_context: &TaskContext,
    target_type: Sym,
    env: &Env,
) -> Vec<SynthComponent> {
    let mut scored: Vec<(f64, SynthComponent)> = components
        .iter()
        .map(|comp| {
            let score = evaluate_heuristic(heuristic, comp, task_context, target_type, env);
            let mut new_comp = comp.clone();
            new_comp.priority = score;
            (score, new_comp)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scored.into_iter().map(|(_, comp)| comp).collect()
}

/// Convenience wrapper that infers the target type from `expected` instead
/// of taking it as an explicit parameter. Mirrors how `meta::apply_heuristic`
/// is called from `cmd_curriculum`.
pub fn apply_heuristic_for_task(
    heuristic: &Heuristic,
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
) -> Vec<SynthComponent> {
    let task_ctx = TaskContext::from_examples(inputs, expected);
    let target = infer_uniform_type_sym(expected).unwrap_or_else(type_any);
    apply_heuristic(heuristic, components, &task_ctx, target, env)
}

// ── Value-form application path (§9.34) ─────────────────────────────────────

/// Score and re-sort `components` using a heuristic supplied as a
/// pre-evaluated `Value::Function` (or `Value::Builtin`). This is the path
/// `bi_synthesize` takes when a heuristic is passed in via the spec
/// namespace — `(synthesize (ns ("spec" ...) ("heuristic" my-lambda)))`.
///
/// The difference from `apply_heuristic_for_task`: that helper takes a
/// `Heuristic` struct, which holds the *parsed lambda Node* and re-evaluates
/// it on every component (one `eval_v2::eval` call per component, which
/// constructs a fresh `Value::Function` from the Lambda Node every time).
/// This helper takes the resolved Function directly and skips the eval
/// step — call `eval_v2::apply` once per component, no re-construction.
///
/// Returns `Err` if `func` isn't a callable Value. The caller is expected
/// to validate this at the spec-namespace boundary so the error message
/// can name the field.
pub fn apply_heuristic_value_for_task(
    func: &Value,
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
) -> Result<Vec<SynthComponent>, String> {
    if !matches!(func, Value::Function(_) | Value::Builtin(_)) {
        return Err(format!(
            "heuristic must be a function value, got {:?}",
            func
        ));
    }

    let task_ctx = TaskContext::from_examples(inputs, expected);
    let target = infer_uniform_type_sym(expected).unwrap_or_else(type_any);

    let mut scored: Vec<(f64, SynthComponent)> = components
        .iter()
        .map(|comp| {
            let ctx_val = component_to_namespace(comp, &task_ctx, target);
            let score = match eval_v2::apply(func, &[ctx_val], env) {
                Ok(Value::Num(n)) if n.is_finite() => n,
                Ok(Value::Int(n)) => n as f64,
                _ => 0.0,
            };
            let mut new_comp = comp.clone();
            new_comp.priority = score;
            (score, new_comp)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(scored.into_iter().map(|(_, comp)| comp).collect())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_v2::{Dispatch, LiteralKind};

    fn make_test_env() -> Env {
        eval_v2::make_default_env()
    }

    fn dummy_component(name: &str, ret: Sym, priority: f64) -> SynthComponent {
        SynthComponent {
            name: name.to_string(),
            dispatch: Dispatch::Literal(LiteralKind::Int(0)),
            arity: 0,
            param_types: vec![],
            ret_type: ret,
            priority,
            usage_count: 0.0,
        }
    }

    #[test]
    fn default_heuristic_returns_priority() {
        let h = Heuristic::default_heuristic();
        let int_sym = intern("Int");
        let comp = dummy_component("c", int_sym, 42.0);
        let inputs = vec![Value::Int(1), Value::Int(2)];
        let expected = vec![Value::Int(2), Value::Int(4)];
        let task = TaskContext::from_examples(&inputs, &expected);
        let env = make_test_env();
        let score = evaluate_heuristic(&h, &comp, &task, int_sym, &env);
        assert!((score - 42.0).abs() < 1e-9, "got {}", score);
    }

    #[test]
    fn type_match_heuristic_boosts_matching() {
        let h = Heuristic::from_source(
            "type-match",
            "(lambda (ctx) (let ((p (ns-get ctx \"priority\")) \
                                 (r (ns-get ctx \"ret-type\")) \
                                 (t (ns-get ctx \"target-type\"))) \
                              (if (= r t) (add p 100) p)))",
        )
        .expect("parse");
        let int_sym = intern("Int");
        let str_sym = intern("String");
        let inputs = vec![Value::Int(1)];
        let expected = vec![Value::Int(2)];
        let task = TaskContext::from_examples(&inputs, &expected);
        let env = make_test_env();
        let int_comp = dummy_component("int_thing", int_sym, 10.0);
        let str_comp = dummy_component("str_thing", str_sym, 10.0);
        let int_score = evaluate_heuristic(&h, &int_comp, &task, int_sym, &env);
        let str_score = evaluate_heuristic(&h, &str_comp, &task, int_sym, &env);
        assert!((int_score - 110.0).abs() < 1e-9, "int got {}", int_score);
        assert!((str_score - 10.0).abs() < 1e-9, "str got {}", str_score);
    }

    #[test]
    fn apply_heuristic_sorts_descending() {
        let h = Heuristic::from_source(
            "type-match",
            "(lambda (ctx) (if (= (ns-get ctx \"ret-type\") (ns-get ctx \"target-type\")) \
                              100 0))",
        )
        .expect("parse");
        let int_sym = intern("Int");
        let str_sym = intern("String");
        let comps = vec![
            dummy_component("a", str_sym, 5.0),
            dummy_component("b", int_sym, 3.0),
            dummy_component("c", str_sym, 7.0),
            dummy_component("d", int_sym, 1.0),
        ];
        let inputs = vec![Value::Int(1)];
        let expected = vec![Value::Int(2)];
        let task = TaskContext::from_examples(&inputs, &expected);
        let env = make_test_env();
        let out = apply_heuristic(&h, &comps, &task, int_sym, &env);
        // The two int-typed components should sort to the top with priority 100;
        // the two string-typed below them with priority 0. Within each tier, the
        // order isn't guaranteed (sort is unstable).
        assert_eq!(out.len(), 4);
        assert!((out[0].priority - 100.0).abs() < 1e-9);
        assert!((out[1].priority - 100.0).abs() < 1e-9);
        assert!((out[2].priority - 0.0).abs() < 1e-9);
        assert!((out[3].priority - 0.0).abs() < 1e-9);
        let top_names: HashSet<&str> = out[..2].iter().map(|c| c.name.as_str()).collect();
        assert!(top_names.contains("b") && top_names.contains("d"));
    }
}
