//! Template-based higher-order decomposition for SELPH synthesis.
//!
//! When flat synthesis fails, this module tries decomposition templates
//! that break a complex task into simpler sub-tasks solved by recursive
//! synthesis calls.
//!
//! Templates:
//!   1. list-map:       (lambda (x) (map HOLE x))
//!   2. split-map-join: (lambda (x) (string-join (map HOLE (string-split x SEP)) SEP))
//!   3. char-map-join:  (lambda (x) (string-join (map HOLE (string-chars x)) ""))
//!   4. list-filter:    (lambda (x) (filter HOLE x))

use std::collections::HashMap;
use std::rc::Rc;
use crate::intern::{intern, resolve};
use crate::eval;
use crate::synth::{self, SynthComponent, vals_equal, val_hash};
use crate::types::*;

// ── Result type ────────────────���────────────────────────────────────

pub struct DecompResult {
    pub found: bool,
    pub nodes: Vec<Node>,
    pub root: usize,
    pub candidates_explored: usize,
    pub template_used: String,
}

impl DecompResult {
    fn empty() -> Self {
        DecompResult {
            found: false,
            nodes: Vec::new(),
            root: 0,
            candidates_explored: 0,
            template_used: String::new(),
        }
    }
}

// ── Main entry point ────────────────────────────────────────────────

/// Attempt to solve a failed synthesis task via template decomposition.
///
/// Tries each template in order, returning the first successful result.
/// Budget is split across templates; short-circuits on first success.
pub fn try_decomposition(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> DecompResult {
    if inputs.is_empty() || expected.is_empty() || inputs.len() != expected.len() {
        return DecompResult::empty();
    }

    let budget_per_template = max_candidates / 4;
    let mut total_explored: usize = 0;

    // Template 1: list-map
    let r = try_list_map(components, inputs, expected, macros, max_depth, budget_per_template);
    total_explored += r.candidates_explored;
    if r.found { return DecompResult { candidates_explored: total_explored, ..r }; }

    // Template 2: split-map-join
    let r = try_split_map_join(components, inputs, expected, macros, max_depth, budget_per_template);
    total_explored += r.candidates_explored;
    if r.found { return DecompResult { candidates_explored: total_explored, ..r }; }

    // Template 3: char-map-join
    let r = try_char_map_join(components, inputs, expected, macros, max_depth, budget_per_template);
    total_explored += r.candidates_explored;
    if r.found { return DecompResult { candidates_explored: total_explored, ..r }; }

    // Template 4: list-filter
    let r = try_list_filter(components, inputs, expected, macros, max_depth, budget_per_template);
    total_explored += r.candidates_explored;
    if r.found { return DecompResult { candidates_explored: total_explored, ..r }; }

    DecompResult { candidates_explored: total_explored, ..DecompResult::empty() }
}

// ── Shared helpers ────────────��─────────────────────────────────────

/// Verify a composed lambda program against all input/output examples.
fn verify_composed(
    nodes: &[Node],
    lambda_idx: usize,
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> bool {
    let nodes_rc: Rc<[Node]> = nodes.to_vec().into();
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        let mut env = eval::make_default_env();
        for (nm, ps, mn, mr) in macros {
            env_define(
                &mut env,
                intern(nm),
                Value::RustMacro(
                    ps.iter().map(|s| intern(s)).collect(),
                    mn.clone().into(),
                    *mr,
                ),
            );
        }
        let fv = match eval::eval(&nodes_rc, lambda_idx, &mut env) {
            Ok(v) => v,
            Err(_) => return false,
        };
        match eval::apply(&fv, &[inp.clone()], &nodes_rc, &mut env) {
            Ok(ref v) if vals_equal(v, exp) => {}
            _ => return false,
        }
    }
    true
}

/// Deduplicate a sub-spec, returning None if there are conflicting pairs
/// (same input, different output).
fn dedup_spec(pairs: Vec<(Value, Value)>) -> Option<Vec<(Value, Value)>> {
    let mut seen: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut unique: Vec<(Value, Value)> = Vec::new();

    for (inp, out) in &pairs {
        let h = val_hash(inp);
        let mut found = false;
        if let Some(indices) = seen.get(&h) {
            for &idx in indices {
                if vals_equal(&unique[idx].0, inp) {
                    // Same input — check output matches
                    if !vals_equal(&unique[idx].1, out) {
                        return None; // Conflict
                    }
                    found = true;
                    break;
                }
            }
        }
        if !found {
            let idx = unique.len();
            seen.entry(h).or_default().push(idx);
            unique.push((inp.clone(), out.clone()));
        }
    }

    if unique.len() < 3 {
        return None; // Too few unique pairs to synthesize reliably
    }

    Some(unique)
}

// ── Template 1: list-map ───────────��────────────────────────────────

/// Try (lambda (x) (map HOLE x)) decomposition.
///
/// Applicable when all inputs and outputs are lists of the same length.
fn try_list_map(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> DecompResult {
    let mut result = DecompResult::empty();

    // Check applicability: all inputs and outputs must be lists of same length
    let mut sub_pairs: Vec<(Value, Value)> = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (in_list, out_list) = match (inp, out) {
            (Value::List(il), Value::List(ol)) if il.len() == ol.len() && !il.is_empty() => (il, ol),
            _ => return result,
        };
        for (ie, oe) in in_list.iter().zip(out_list.iter()) {
            sub_pairs.push((ie.clone(), oe.clone()));
        }
    }

    // Deduplicate sub-spec
    let sub_spec = match dedup_spec(sub_pairs) {
        Some(s) => s,
        None => return result,
    };

    let sub_inputs: Vec<Value> = sub_spec.iter().map(|(i, _)| i.clone()).collect();
    let sub_expected: Vec<Value> = sub_spec.iter().map(|(_, o)| o.clone()).collect();

    // Sub-synthesis
    let sr = synth::synthesize(
        components, &sub_inputs, &sub_expected, macros,
        max_depth, max_candidates, false,
    );
    result.candidates_explored += sr.candidates_explored;

    if !sr.found { return result; }

    let sub_nodes = sr.nodes.unwrap();
    let sub_root = sr.root.unwrap();

    // Compose: (lambda (x) (map sub-fn x))
    // nodes[0] = Symbol("x")
    // nodes[1..1+N] = sub-solution nodes (remapped by offset 1)
    // nodes[1+N] = Symbol("map")
    // nodes[2+N] = App([1+N, sub_root+1, 0])   -- (map sub-fn x)
    // nodes[3+N] = Lambda(["x"], 2+N)
    let mut nodes = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sub_offset = nodes.len();
    for nd in &sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_root_remapped = sub_root + sub_offset;

    let map_sym = nodes.len();
    nodes.push(Node::Symbol(intern("map")));

    let map_app = nodes.len();
    nodes.push(Node::App(vec![map_sym, sub_root_remapped, x_idx]));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], map_app));

    // Verify
    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
        result.found = true;
        result.nodes = nodes;
        result.root = lambda_idx;
        result.template_used = "list-map".to_string();
    }

    result
}

// ── Template 2: split-map-join ─────────��────────────────────────────

/// Common delimiters to try, ordered by frequency.
const DELIMITERS: &[&str] = &[" ", ",", "-", ".", "/", ":", ";", "_", "|"];

/// Try (lambda (x) (string-join (map HOLE (string-split x SEP)) SEP)).
///
/// Applicable when all inputs and outputs are strings that split into
/// the same number of parts by a common delimiter.
fn try_split_map_join(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> DecompResult {
    let mut result = DecompResult::empty();

    // Check all inputs and outputs are strings
    let strs_in: Vec<&str> = match inputs.iter().map(|v| match v {
        Value::Str(s) => Some(s.as_str()),
        _ => None,
    }).collect::<Option<Vec<_>>>() {
        Some(v) => v,
        None => return result,
    };
    let strs_out: Vec<&str> = match expected.iter().map(|v| match v {
        Value::Str(s) => Some(s.as_str()),
        _ => None,
    }).collect::<Option<Vec<_>>>() {
        Some(v) => v,
        None => return result,
    };

    // Try each delimiter
    for &sep in DELIMITERS {
        let sub_spec = match derive_split_spec(&strs_in, &strs_out, sep) {
            Some(s) => s,
            None => continue,
        };

        let sub_inputs: Vec<Value> = sub_spec.iter().map(|(i, _)| i.clone()).collect();
        let sub_expected: Vec<Value> = sub_spec.iter().map(|(_, o)| o.clone()).collect();

        // Sub-synthesis
        let sr = synth::synthesize(
            components, &sub_inputs, &sub_expected, macros,
            max_depth, max_candidates, false,
        );
        result.candidates_explored += sr.candidates_explored;

        if !sr.found { continue; }

        let sub_nodes = sr.nodes.unwrap();
        let sub_root = sr.root.unwrap();

        // Compose: (lambda (x) (string-join (map sub-fn (string-split x SEP)) SEP))
        let mut nodes = Vec::new();

        let x_idx = nodes.len();
        nodes.push(Node::Symbol(intern("x")));            // 0: x

        let sep_idx = nodes.len();
        nodes.push(Node::Str(sep.to_string()));            // 1: SEP

        let split_sym = nodes.len();
        nodes.push(Node::Symbol(intern("string-split")));  // 2: string-split

        let split_app = nodes.len();
        nodes.push(Node::App(vec![split_sym, x_idx, sep_idx])); // 3: (string-split x SEP)

        // Import sub-solution
        let sub_offset = nodes.len();
        for nd in &sub_nodes {
            nodes.push(synth::remap_node(nd, sub_offset));
        }
        let sub_root_remapped = sub_root + sub_offset;

        let map_sym = nodes.len();
        nodes.push(Node::Symbol(intern("map")));

        let map_app = nodes.len();
        nodes.push(Node::App(vec![map_sym, sub_root_remapped, split_app])); // (map sub-fn split-result)

        let sep_idx2 = nodes.len();
        nodes.push(Node::Str(sep.to_string()));             // SEP for join

        let join_sym = nodes.len();
        nodes.push(Node::Symbol(intern("string-join")));

        let join_app = nodes.len();
        nodes.push(Node::App(vec![join_sym, map_app, sep_idx2])); // (string-join map-result SEP)

        let lambda_idx = nodes.len();
        nodes.push(Node::Lambda(vec![intern("x")], join_app));

        // Verify
        if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
            result.found = true;
            result.nodes = nodes;
            result.root = lambda_idx;
            result.template_used = format!("split-map-join(\"{}\")", sep);
            return result;
        }
    }

    result
}

/// Derive a sub-spec from splitting inputs and outputs by a delimiter.
/// Returns None if the delimiter doesn't produce consistent splits.
fn derive_split_spec(
    inputs: &[&str],
    outputs: &[&str],
    sep: &str,
) -> Option<Vec<(Value, Value)>> {
    let mut pairs: Vec<(Value, Value)> = Vec::new();
    let mut any_multi = false;

    for (&inp, &out) in inputs.iter().zip(outputs.iter()) {
        let in_parts: Vec<&str> = inp.split(sep).collect();
        let out_parts: Vec<&str> = out.split(sep).collect();

        if in_parts.len() != out_parts.len() {
            return None;
        }
        if in_parts.len() >= 2 {
            any_multi = true;
        }

        for (ip, op) in in_parts.iter().zip(out_parts.iter()) {
            pairs.push((Value::Str(ip.to_string()), Value::Str(op.to_string())));
        }
    }

    if !any_multi {
        return None; // Delimiter didn't actually split anything
    }

    dedup_spec(pairs)
}

// ��─ Template 3: char-map-join ───────────────────────────────────────

/// Try (lambda (x) (string-join (map HOLE (string-chars x)) "")).
///
/// Applicable when all inputs and outputs are strings of the same length.
fn try_char_map_join(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> DecompResult {
    let mut result = DecompResult::empty();

    // Check all are strings of same length per example
    let mut sub_pairs: Vec<(Value, Value)> = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (s_in, s_out) = match (inp, out) {
            (Value::Str(a), Value::Str(b)) if a.chars().count() == b.chars().count()
                && !a.is_empty() => (a, b),
            _ => return result,
        };
        for (ci, co) in s_in.chars().zip(s_out.chars()) {
            sub_pairs.push((
                Value::Str(ci.to_string()),
                Value::Str(co.to_string()),
            ));
        }
    }

    let sub_spec = match dedup_spec(sub_pairs) {
        Some(s) => s,
        None => return result,
    };

    let sub_inputs: Vec<Value> = sub_spec.iter().map(|(i, _)| i.clone()).collect();
    let sub_expected: Vec<Value> = sub_spec.iter().map(|(_, o)| o.clone()).collect();

    // Sub-synthesis
    let sr = synth::synthesize(
        components, &sub_inputs, &sub_expected, macros,
        max_depth, max_candidates, false,
    );
    result.candidates_explored += sr.candidates_explored;

    if !sr.found { return result; }

    let sub_nodes = sr.nodes.unwrap();
    let sub_root = sr.root.unwrap();

    // Compose: (lambda (x) (string-join (map sub-fn (string-chars x)) ""))
    let mut nodes = Vec::new();

    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let chars_sym = nodes.len();
    nodes.push(Node::Symbol(intern("string-chars")));

    let chars_app = nodes.len();
    nodes.push(Node::App(vec![chars_sym, x_idx]));     // (string-chars x)

    let sub_offset = nodes.len();
    for nd in &sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_root_remapped = sub_root + sub_offset;

    let map_sym = nodes.len();
    nodes.push(Node::Symbol(intern("map")));

    let map_app = nodes.len();
    nodes.push(Node::App(vec![map_sym, sub_root_remapped, chars_app]));

    let empty_sep = nodes.len();
    nodes.push(Node::Str(String::new()));

    let join_sym = nodes.len();
    nodes.push(Node::Symbol(intern("string-join")));

    let join_app = nodes.len();
    nodes.push(Node::App(vec![join_sym, map_app, empty_sep]));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], join_app));

    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
        result.found = true;
        result.nodes = nodes;
        result.root = lambda_idx;
        result.template_used = "char-map-join".to_string();
    }

    result
}

// ���─ Template 4: list-filter ─────────────────────────────────────────

/// Try (lambda (x) (filter HOLE x)) decomposition.
///
/// Applicable when all inputs are lists and all outputs are ordered
/// subsets of the corresponding inputs.
fn try_list_filter(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> DecompResult {
    let mut result = DecompResult::empty();

    // Check applicability and derive boolean labels
    let mut sub_pairs: Vec<(Value, Value)> = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (in_list, out_list) = match (inp, out) {
            (Value::List(il), Value::List(ol)) if !il.is_empty() => (il, ol),
            _ => return result,
        };

        // Check that out_list is an ordered subset of in_list
        let mut out_idx = 0;
        for elem in in_list {
            let keep = if out_idx < out_list.len() && vals_equal(elem, &out_list[out_idx]) {
                out_idx += 1;
                true
            } else {
                false
            };
            sub_pairs.push((elem.clone(), Value::Bool(keep)));
        }

        // All output elements must have been matched
        if out_idx != out_list.len() {
            return result;
        }
    }

    // Deduplicate, checking for conflicts
    let sub_spec = match dedup_spec(sub_pairs) {
        Some(s) => s,
        None => return result,
    };

    let sub_inputs: Vec<Value> = sub_spec.iter().map(|(i, _)| i.clone()).collect();
    let sub_expected: Vec<Value> = sub_spec.iter().map(|(_, o)| o.clone()).collect();

    // Sub-synthesis
    let sr = synth::synthesize(
        components, &sub_inputs, &sub_expected, macros,
        max_depth, max_candidates, false,
    );
    result.candidates_explored += sr.candidates_explored;

    if !sr.found { return result; }

    let sub_nodes = sr.nodes.unwrap();
    let sub_root = sr.root.unwrap();

    // Compose: (lambda (x) (filter sub-fn x))
    let mut nodes = Vec::new();

    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sub_offset = nodes.len();
    for nd in &sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_root_remapped = sub_root + sub_offset;

    let filter_sym = nodes.len();
    nodes.push(Node::Symbol(intern("filter")));

    let filter_app = nodes.len();
    nodes.push(Node::App(vec![filter_sym, sub_root_remapped, x_idx]));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], filter_app));

    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
        result.found = true;
        result.nodes = nodes;
        result.root = lambda_idx;
        result.template_used = "list-filter".to_string();
    }

    result
}
