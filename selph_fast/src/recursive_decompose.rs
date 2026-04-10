//! Recursive decomposition: synthesis as top-down prediction.
//!
//! All search strategies (flat, BD, HO, D&C, induction, memo) are instances
//! of a single recursive step:
//!
//!   synthesize(spec) →
//!     1. SELECT f (the outermost function)
//!     2. DERIVE subspecs by "inverting" f on the examples
//!     3. RECURSIVELY synthesize each subspec
//!
//! This module implements:
//!   - `predict_family`: classify spec → outermost function family
//!   - `invert`: given f and examples, derive subspecs
//!   - `try_recursive_decomposition`: the top-level entry point

use std::rc::Rc;
use crate::intern::{intern, resolve};
use crate::eval;
use crate::synth::{self, SynthComponent, vals_equal, val_hash};
use crate::types::*;

// ── Result type ────────────────────────────────────────────────────

pub struct RecursiveDecompResult {
    pub found: bool,
    pub nodes: Vec<Node>,
    pub root: usize,
    pub candidates_explored: usize,
    pub strategy_used: String,
}

impl RecursiveDecompResult {
    pub fn empty() -> Self {
        RecursiveDecompResult {
            found: false,
            nodes: Vec::new(),
            root: 0,
            candidates_explored: 0,
            strategy_used: String::new(),
        }
    }
}

// ── Family classification ──────────────────────────────────────────

/// Function families for the outermost function prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Family {
    Constant,
    Arithmetic,
    Compare,
    BoolComp,
    StringOp,
    Count,
    HigherOrder,
    IfExpr,
}

/// Features extracted from a spec for family prediction.
struct SpecFeatures {
    input_type: &'static str,   // "num", "str", "list"
    output_type: &'static str,  // "num", "str", "bool"
    num_distinct_outputs: usize,
    has_bool_macros: bool,
    output_is_substring: bool,  // all outputs are substrings of inputs
    _output_shorter: bool,      // all string outputs shorter than inputs
}

fn extract_spec_features(
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> SpecFeatures {
    let input_type = match &inputs[0] {
        Value::Num(_) => "num",
        Value::Str(_) => "str",
        Value::List(_) => "list",
        _ => "unknown",
    };
    let output_type = if expected.iter().all(|v| matches!(v, Value::Bool(_))) {
        "bool"
    } else if expected.iter().all(|v| matches!(v, Value::Num(_))) {
        "num"
    } else if expected.iter().all(|v| matches!(v, Value::Str(_))) {
        "str"
    } else {
        "unknown"
    };

    let mut distinct: Vec<String> = expected.iter().map(|v| format!("{:?}", v)).collect();
    distinct.sort();
    distinct.dedup();
    let num_distinct_outputs = distinct.len();

    let has_bool_macros = macros.iter().any(|(_, params, nodes, root)| {
        if params.len() != 1 { return false; }
        // Heuristic: check if macro name suggests boolean return
        // More robust: try evaluating on a sample input
        let nodes_rc: Rc<[Node]> = nodes.clone().into();
        let mut env = eval::make_default_env();
        for (nm, ps, mn, mr) in macros {
            env_define(&mut env, intern(nm),
                Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), mn.clone().into(), *mr));
        }
        if let Ok(fv) = eval::eval(&nodes_rc, *root, &mut env) {
            if let Ok(result) = eval::apply(&fv, &[inputs[0].clone()], &nodes_rc, &mut env) {
                return matches!(result, Value::Bool(_));
            }
        }
        false
    });

    // Check if outputs are substrings of inputs (string-take/string-drop pattern)
    let output_is_substring = input_type == "str" && output_type == "str" && {
        inputs.iter().zip(expected.iter()).all(|(i, e)| {
            if let (Value::Str(si), Value::Str(se)) = (i, e) {
                si.contains(se.as_str())
            } else { false }
        })
    };

    let output_shorter = input_type == "str" && output_type == "str" && {
        inputs.iter().zip(expected.iter()).all(|(i, e)| {
            if let (Value::Str(si), Value::Str(se)) = (i, e) {
                se.len() < si.len()
            } else { false }
        })
    };

    SpecFeatures {
        input_type,
        output_type,
        num_distinct_outputs,
        has_bool_macros,
        output_is_substring,
        _output_shorter: output_shorter,
    }
}

/// Predict which function family the outermost function belongs to.
/// Decision tree from the decomposition_predictor.selph prototype.
fn predict_family(features: &SpecFeatures) -> Family {
    if features.output_type == "bool" {
        if features.has_bool_macros {
            Family::BoolComp
        } else {
            Family::Compare
        }
    } else if features.output_type == "num" {
        if features.input_type == "list" {
            Family::Arithmetic
        } else if features.input_type == "str" {
            Family::Count
        } else {
            Family::Arithmetic
        }
    } else if features.output_type == "str" {
        if features.output_is_substring {
            // Outputs are substrings of inputs — this is a string-take/drop pattern,
            // not a classification task, even if there are many distinct outputs.
            Family::StringOp
        } else if features.num_distinct_outputs > 4 {
            Family::IfExpr
        } else if features.num_distinct_outputs == 1 {
            Family::Constant
        } else {
            Family::StringOp
        }
    } else {
        Family::Arithmetic // fallback
    }
}

// ── Candidate outermost functions per family ───────────────────────

/// For a given family, return candidate outermost function names to try.
fn family_candidates(family: Family, features: &SpecFeatures) -> Vec<&'static str> {
    match family {
        Family::Arithmetic => vec![
            "add", "subtract", "multiply", "floor", "negate", "abs",
            "divide", "modulo", "pow",
        ],
        Family::Count => vec![
            "string-length", "count-char",
        ],
        Family::Compare => vec![
            "string-ends-with", "string-starts-with", "contains",
            "even", "odd",
        ],
        Family::BoolComp => vec!["and", "or", "not"],
        Family::StringOp => {
            if features.output_is_substring {
                // Prefer substring operations when output is substring of input
                vec![
                    "string-take", "string-drop",
                    "string-replace", "string-upper", "string-lower",
                    "string-reverse", "string-trim", "concat",
                ]
            } else {
                vec![
                    "concat", "string-replace",
                    "string-upper", "string-lower", "string-reverse",
                    "string-trim", "string-take", "string-drop",
                ]
            }
        }
        Family::Constant => vec![],
        Family::HigherOrder => vec!["map", "filter"],
        Family::IfExpr => vec!["if"],
    }
}

// ── Inversion: derive subspecs from outermost f ────────────────────

/// Subspec: a derived synthesis problem from inversion.
struct SubSpec {
    inputs: Vec<Value>,
    expected: Vec<Value>,
}

/// Try to invert a binary arithmetic function on the examples.
///
/// Given f(x, k) = output for each example, where x is the input:
///   - If f is `add`: k = output - input for each pair. If consistent → subspec for k.
///   - If f is `subtract`: k = input - output. Or output = input - k → k = input - output.
///   - If f is `multiply`: k = output / input.
///
/// Also tries f(k, x) = output (reversed argument order).
///
/// Returns the subspec for the second argument (the one to synthesize).
fn invert_binary_arith(
    fn_name: &str,
    inputs: &[Value],
    expected: &[Value],
) -> Option<SubSpec> {
    // Extract numeric values
    let nums_in: Vec<f64> = inputs.iter().filter_map(|v| {
        match v { Value::Num(n) => Some(*n), _ => None }
    }).collect();
    let nums_out: Vec<f64> = expected.iter().filter_map(|v| {
        match v { Value::Num(n) => Some(*n), _ => None }
    }).collect();

    if nums_in.len() != inputs.len() || nums_out.len() != expected.len() {
        return None;
    }

    // For binary ops, try both argument orderings:
    //   f(input, k) = output  →  k = f_inv(output, input)
    //   f(k, input) = output  →  k = f_inv2(output, input)
    let derived: Vec<Option<f64>> = match fn_name {
        "add" => {
            // output = input + k  →  k = output - input
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| Some(o - i))
                .collect()
        }
        "subtract" => {
            // output = input - k  →  k = input - output
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| Some(i - o))
                .collect()
        }
        "multiply" => {
            // output = input * k  →  k = output / input
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| if *i != 0.0 { Some(o / i) } else { None })
                .collect()
        }
        "divide" => {
            // output = input / k  →  k = input / output
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| if *o != 0.0 { Some(i / o) } else { None })
                .collect()
        }
        "modulo" => {
            // Hard to invert modulo — skip
            return None;
        }
        "pow" => {
            // output = input^k  →  k = log(output)/log(input)
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| {
                    if *i > 0.0 && *i != 1.0 && *o > 0.0 {
                        let k = o.ln() / i.ln();
                        if (k - k.round()).abs() < 1e-9 { Some(k.round()) } else { None }
                    } else { None }
                }).collect()
        }
        _ => return None,
    };

    let k_values: Vec<f64> = match derived.into_iter().collect::<Option<Vec<f64>>>() {
        Some(v) => v,
        None => return None,
    };

    // Check if k is constant — if so, it's a trivial constant, not worth a subspec
    if !k_values.is_empty() && k_values.windows(2).all(|w| w[0] == w[1]) {
        // Constant k — still useful, the subspec is just a constant synthesis
        // But only if the constant isn't already in the typical range
    }

    Some(SubSpec {
        inputs: inputs.to_vec(),
        expected: k_values.into_iter().map(Value::Num).collect(),
    })
}

/// Try to invert a unary arithmetic function.
///
/// Given f(g(x)) = output, we need g(x) = f_inv(output) for each example.
fn invert_unary_arith(
    fn_name: &str,
    inputs: &[Value],
    expected: &[Value],
) -> Option<SubSpec> {
    let nums_out: Vec<f64> = expected.iter().filter_map(|v| {
        match v { Value::Num(n) => Some(*n), _ => None }
    }).collect();
    if nums_out.len() != expected.len() {
        return None;
    }

    let derived: Vec<Option<f64>> = match fn_name {
        "negate" => {
            // output = -g(x)  →  g(x) = -output
            nums_out.iter().map(|o| Some(-o)).collect()
        }
        "abs" => {
            // output = |g(x)|  →  g(x) = ±output (ambiguous, skip)
            return None;
        }
        "floor" | "ceil" => {
            // output = floor(g(x))  →  g(x) ∈ [output, output+1) (ambiguous)
            return None;
        }
        _ => return None,
    };

    let sub_expected: Vec<f64> = match derived.into_iter().collect::<Option<Vec<f64>>>() {
        Some(v) => v,
        None => return None,
    };

    Some(SubSpec {
        inputs: inputs.to_vec(),
        expected: sub_expected.into_iter().map(Value::Num).collect(),
    })
}

/// Try to invert a binary arithmetic function with reversed args: f(k, input) = output.
fn invert_binary_arith_reversed(
    fn_name: &str,
    inputs: &[Value],
    expected: &[Value],
) -> Option<SubSpec> {
    let nums_in: Vec<f64> = inputs.iter().filter_map(|v| {
        match v { Value::Num(n) => Some(*n), _ => None }
    }).collect();
    let nums_out: Vec<f64> = expected.iter().filter_map(|v| {
        match v { Value::Num(n) => Some(*n), _ => None }
    }).collect();

    if nums_in.len() != inputs.len() || nums_out.len() != expected.len() {
        return None;
    }

    let derived: Vec<Option<f64>> = match fn_name {
        "add" => {
            // Same as normal: output = k + input → k = output - input
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| Some(o - i))
                .collect()
        }
        "subtract" => {
            // output = k - input  →  k = output + input
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| Some(o + i))
                .collect()
        }
        "multiply" => {
            // Same as normal
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| if *i != 0.0 { Some(o / i) } else { None })
                .collect()
        }
        "divide" => {
            // output = k / input  →  k = output * input
            nums_in.iter().zip(nums_out.iter())
                .map(|(i, o)| Some(o * i))
                .collect()
        }
        _ => return None,
    };

    let k_values: Vec<f64> = match derived.into_iter().collect::<Option<Vec<f64>>>() {
        Some(v) => v,
        None => return None,
    };

    Some(SubSpec {
        inputs: inputs.to_vec(),
        expected: k_values.into_iter().map(Value::Num).collect(),
    })
}

/// Invert string-take: output = (string-take input k)
/// → k = len(output) for each example (if output is a prefix of input).
fn invert_string_take(
    inputs: &[Value],
    expected: &[Value],
) -> Option<SubSpec> {
    let mut k_values = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        if let (Value::Str(si), Value::Str(so)) = (inp, out) {
            if si.starts_with(so.as_str()) {
                k_values.push(Value::Num(so.chars().count() as f64));
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(SubSpec {
        inputs: inputs.to_vec(),
        expected: k_values,
    })
}

/// Invert string-drop: output = (string-drop input k)
/// → k = len(input) - len(output) (if output is a suffix of input).
fn invert_string_drop(
    inputs: &[Value],
    expected: &[Value],
) -> Option<SubSpec> {
    let mut k_values = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        if let (Value::Str(si), Value::Str(so)) = (inp, out) {
            if si.ends_with(so.as_str()) {
                k_values.push(Value::Num((si.chars().count() - so.chars().count()) as f64));
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(SubSpec {
        inputs: inputs.to_vec(),
        expected: k_values,
    })
}

/// Invert concat: output = (concat input suffix) or (concat prefix input).
/// Try both orderings.
fn invert_concat(
    inputs: &[Value],
    expected: &[Value],
) -> Vec<(SubSpec, bool)> {
    let mut results = Vec::new();

    // Try output = concat(input, suffix) → suffix = output[len(input)..]
    let mut suffixes = Vec::new();
    let mut valid = true;
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        if let (Value::Str(si), Value::Str(so)) = (inp, out) {
            if so.starts_with(si.as_str()) {
                suffixes.push(Value::Str(so[si.len()..].to_string()));
            } else {
                valid = false;
                break;
            }
        } else {
            valid = false;
            break;
        }
    }
    if valid && !suffixes.is_empty() {
        results.push((SubSpec {
            inputs: inputs.to_vec(),
            expected: suffixes,
        }, false)); // false = input is arg1
    }

    // Try output = concat(prefix, input) → prefix = output[..len(output)-len(input)]
    let mut prefixes = Vec::new();
    valid = true;
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        if let (Value::Str(si), Value::Str(so)) = (inp, out) {
            if so.ends_with(si.as_str()) {
                let prefix_len = so.len() - si.len();
                prefixes.push(Value::Str(so[..prefix_len].to_string()));
            } else {
                valid = false;
                break;
            }
        } else {
            valid = false;
            break;
        }
    }
    if valid && !prefixes.is_empty() {
        results.push((SubSpec {
            inputs: inputs.to_vec(),
            expected: prefixes,
        }, true)); // true = input is arg2 (reversed)
    }

    results
}

/// Invert count-char: output = (count-char input ch).
/// We need to find ch such that counting it in input gives output.
fn invert_count_char(
    inputs: &[Value],
    expected: &[Value],
) -> Option<SubSpec> {
    // For count-char, the second arg is a character to count.
    // Try common characters and see if counting them gives the right output.
    let chars_to_try: Vec<char> = {
        let mut chars = Vec::new();
        for inp in inputs {
            if let Value::Str(s) = inp {
                for c in s.chars() {
                    if !chars.contains(&c) {
                        chars.push(c);
                    }
                }
            }
        }
        chars
    };

    for ch in &chars_to_try {
        let ch_str = ch.to_string();
        let mut matches = true;
        for (inp, out) in inputs.iter().zip(expected.iter()) {
            if let (Value::Str(s), Value::Num(n)) = (inp, out) {
                let count = s.matches(&ch_str).count() as f64;
                if count != *n {
                    matches = false;
                    break;
                }
            } else {
                matches = false;
                break;
            }
        }
        if matches {
            // The second argument is a constant character — build a trivial subspec
            return Some(SubSpec {
                inputs: inputs.to_vec(),
                expected: vec![Value::Str(ch_str); inputs.len()],
            });
        }
    }
    None
}

/// Invert unary string ops (string-upper, string-lower, string-reverse, string-trim).
fn invert_unary_string(
    fn_name: &str,
    inputs: &[Value],
    expected: &[Value],
) -> Option<SubSpec> {
    // For unary string ops, output = f(g(x)). We need g(x) = f_inv(output).
    let sub_expected: Vec<Value> = expected.iter().filter_map(|v| {
        if let Value::Str(s) = v {
            let inv = match fn_name {
                "string-upper" => s.to_lowercase(),
                "string-lower" => s.to_uppercase(),
                "string-reverse" => s.chars().rev().collect(),
                "string-trim" => return None, // can't reliably invert trim
                _ => return None,
            };
            Some(Value::Str(inv))
        } else {
            None
        }
    }).collect();

    if sub_expected.len() != expected.len() {
        return None;
    }

    // Verify the inversion: f(sub_expected[i]) should equal expected[i]
    for (sub, exp) in sub_expected.iter().zip(expected.iter()) {
        let result = eval::apply_builtin(intern(fn_name), &[sub.clone()]);
        match result {
            Ok(ref v) if vals_equal(v, exp) => {}
            _ => return None,
        }
    }

    Some(SubSpec {
        inputs: inputs.to_vec(),
        expected: sub_expected,
    })
}

// ── Composition: build AST from f + sub-solutions ──────────────────

/// Extract the body index from a sub-solution.
/// If the sub_root is a Lambda, return its body index. Otherwise return sub_root itself.
/// This is needed because synthesis returns (lambda (x) body) but we want just `body`
/// since we're building our own outer lambda.
fn extract_body(nodes: &[Node], sub_root: usize) -> usize {
    match &nodes[sub_root] {
        Node::Lambda(_, body) => *body,
        _ => sub_root,
    }
}

/// Build (lambda (x) (f sub_body)) for unary f.
fn compose_unary(
    fn_name: &str,
    sub_nodes: &[Node],
    sub_root: usize,
) -> (Vec<Node>, usize) {
    let mut nodes = Vec::new();

    // x variable (needed in node list for lambda binding, referenced by sub-solution)
    let _x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    // Import sub-solution
    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_body = extract_body(&nodes, sub_root + sub_offset);

    // f symbol
    let f_idx = nodes.len();
    nodes.push(Node::Symbol(intern(fn_name)));

    // (f sub_body)
    let app_idx = nodes.len();
    nodes.push(Node::App(vec![f_idx, sub_body]));

    // (lambda (x) (f sub_body))
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], app_idx));

    (nodes, lambda_idx)
}

/// Build (lambda (x) (f x sub_body)) for binary f where input is arg1.
fn compose_binary_input_first(
    fn_name: &str,
    sub_nodes: &[Node],
    sub_root: usize,
) -> (Vec<Node>, usize) {
    let mut nodes = Vec::new();

    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_body = extract_body(&nodes, sub_root + sub_offset);

    let f_idx = nodes.len();
    nodes.push(Node::Symbol(intern(fn_name)));

    // (f x sub_body)
    let app_idx = nodes.len();
    nodes.push(Node::App(vec![f_idx, x_idx, sub_body]));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], app_idx));

    (nodes, lambda_idx)
}

/// Build (lambda (x) (f sub_body x)) for binary f where input is arg2.
fn compose_binary_input_second(
    fn_name: &str,
    sub_nodes: &[Node],
    sub_root: usize,
) -> (Vec<Node>, usize) {
    let mut nodes = Vec::new();

    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_body = extract_body(&nodes, sub_root + sub_offset);

    let f_idx = nodes.len();
    nodes.push(Node::Symbol(intern(fn_name)));

    // (f sub_body x)
    let app_idx = nodes.len();
    nodes.push(Node::App(vec![f_idx, sub_body, x_idx]));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], app_idx));

    (nodes, lambda_idx)
}

// ── Verification ───────────────────────────────────────────────────

/// Verify a composed lambda against all examples.
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

// ── Main entry point ───────────────────────────────────────────────

/// Attempt recursive decomposition on a synthesis task.
///
/// 1. Extract spec features and predict outermost function family
/// 2. For each candidate function in the family, try to invert it
/// 3. Recursively synthesize the derived subspec
/// 4. Compose and verify
/// Default RD recursion depth (how many levels of decomposition to try).
const DEFAULT_RD_DEPTH: usize = 2;

pub fn try_recursive_decomposition(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> RecursiveDecompResult {
    try_recursive_decomposition_with_predictor(
        components, inputs, expected, macros, max_depth, max_candidates, None,
    )
}

/// Attempt recursive decomposition with an optional learned predictor.
pub fn try_recursive_decomposition_with_predictor(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    predictor: Option<&LearnedPredictor>,
) -> RecursiveDecompResult {
    try_rd_recursive(
        components, inputs, expected, macros, max_depth, max_candidates,
        predictor, DEFAULT_RD_DEPTH,
    )
}

/// Internal recursive entry point with depth tracking.
fn try_rd_recursive(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    predictor: Option<&LearnedPredictor>,
    rd_depth: usize,
) -> RecursiveDecompResult {
    if inputs.is_empty() || expected.is_empty() || inputs.len() != expected.len() {
        return RecursiveDecompResult::empty();
    }

    let features = extract_spec_features(inputs, expected, macros);
    let family = predict_family_with_learned(&features, predictor, inputs, expected, macros);

    // Budget per candidate outermost function — generous since we predict
    // the family and only try a few candidates within it.
    let sub_budget = max_candidates / 2;
    let mut total_explored: usize = 0;

    let candidates = family_candidates(family, &features);

    // Also try secondary family candidates for cross-family solutions
    let secondary_families = secondary_family_candidates(family, &features);

    let all_candidates: Vec<&str> = candidates.iter().copied()
        .chain(secondary_families.iter().copied())
        .collect();

    for fn_name in &all_candidates {
        if total_explored >= max_candidates {
            break;
        }

        let remaining = max_candidates.saturating_sub(total_explored);
        let budget = remaining.min(sub_budget);

        let result = try_single_function(
            fn_name, components, inputs, expected, macros,
            max_depth, budget, &features, rd_depth, predictor,
        );
        total_explored += result.candidates_explored;

        if result.found {
            return RecursiveDecompResult {
                candidates_explored: total_explored,
                ..result
            };
        }
    }

    // Phase 2: Try promoted macros as outermost or intermediate functions.
    // For each unary macro m, evaluate m(input) on all inputs. If:
    //   - m(input) == expected: direct match → (lambda (x) (m x))
    //   - m(input) ≠ input and ≠ expected: try synthesizing expected from m(input)
    //     i.e., output = f(m(input)) → subspec is (m(input) → output)
    //   - Also try m as inner: output = m(g(x)) → need to invert m.
    let remaining_budget = max_candidates.saturating_sub(total_explored);
    if remaining_budget > 0 {
        let macro_result = try_macro_decomposition(
            components, inputs, expected, macros, max_depth, remaining_budget,
            rd_depth, predictor,
        );
        total_explored += macro_result.candidates_explored;
        if macro_result.found {
            return RecursiveDecompResult {
                candidates_explored: total_explored,
                ..macro_result
            };
        }
    }

    // Phase 3: Try promoted macros as outermost via inversion probing.
    // m(g(x)) = expected — find v_i such that m(v_i) = expected_i, then synthesize g.
    let remaining_budget2 = max_candidates.saturating_sub(total_explored);
    if remaining_budget2 > 0 {
        let outer_result = try_macro_as_outermost(
            components, inputs, expected, macros, max_depth, remaining_budget2,
            rd_depth, predictor,
        );
        total_explored += outer_result.candidates_explored;
        if outer_result.found {
            return RecursiveDecompResult {
                candidates_explored: total_explored,
                ..outer_result
            };
        }
    }

    RecursiveDecompResult {
        candidates_explored: total_explored,
        ..RecursiveDecompResult::empty()
    }
}

/// Try promoted macros as outermost or bridge functions.
///
/// For each unary macro m in the library:
///   1. Evaluate m(input) on all examples
///   2. If m(input) == expected → direct solution (m x)
///   3. If m(input) is a useful intermediate → synthesize f such that f(m(input)) = expected
///      Then compose as (lambda (x) (f (m x)))
fn try_macro_decomposition(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    rd_depth: usize,
    predictor: Option<&LearnedPredictor>,
) -> RecursiveDecompResult {
    let mut result = RecursiveDecompResult::empty();
    // Cap per-macro sub-synthesis budget. With many macros in the library,
    // each one gets a small budget — we're looking for easy compositions.
    // Total Phase 2 budget is 1/4 of max_candidates.
    let phase2_budget = max_candidates / 4;
    let sub_budget = 500usize.min(phase2_budget / 4);

    for (mname, mparams, mnodes, mroot) in macros {
        if mparams.len() != 1 {
            continue; // only unary macros
        }
        if mname.starts_with("__selph_") {
            continue; // skip internal macros
        }
        if result.candidates_explored >= phase2_budget {
            break;
        }

        // Evaluate macro on all inputs
        let nodes_rc: Rc<[Node]> = mnodes.clone().into();
        let fv = Value::RustMacro(
            mparams.iter().map(|s| intern(s)).collect(),
            nodes_rc.clone(),
            *mroot,
        );
        let mut intermediates = Vec::with_capacity(inputs.len());
        let mut valid = true;
        for inp in inputs {
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in macros {
                env_define(&mut env, intern(nm),
                    Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), mn.clone().into(), *mr));
            }
            match eval::apply(&fv, &[inp.clone()], &nodes_rc, &mut env) {
                Ok(v) => intermediates.push(v),
                Err(_) => { valid = false; break; }
            }
        }
        if !valid || intermediates.len() != inputs.len() {
            continue;
        }
        // Check 1: direct match — m(input) == expected
        if intermediates.iter().zip(expected.iter()).all(|(a, b)| vals_equal(a, b)) {
            // Build (lambda (x) (m x))
            let mut nodes = Vec::new();
            let x_idx = nodes.len();
            nodes.push(Node::Symbol(intern("x")));
            let m_idx = nodes.len();
            nodes.push(Node::Symbol(intern(mname)));
            let app_idx = nodes.len();
            nodes.push(Node::App(vec![m_idx, x_idx]));
            let lambda_idx = nodes.len();
            nodes.push(Node::Lambda(vec![intern("x")], app_idx));

            if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                result.found = true;
                result.nodes = nodes;
                result.root = lambda_idx;
                result.strategy_used = format!("RD({} direct)", mname);
                return result;
            }
        }

        // Check 2: skip if intermediates are identical to inputs (identity macro)
        if intermediates.iter().zip(inputs.iter()).all(|(a, b)| vals_equal(a, b)) {
            continue;
        }

        // Check 3: skip if all intermediates are the same (constant macro)
        if intermediates.len() > 1 && intermediates.windows(2).all(|w| vals_equal(&w[0], &w[1])) {
            continue;
        }

        // Check 3b: skip if intermediate type doesn't plausibly lead to output type.
        // E.g., if intermediates are numbers and expected is strings, there's no path.
        let inter_is_num = intermediates.iter().all(|v| matches!(v, Value::Num(_)));
        let inter_is_str = intermediates.iter().all(|v| matches!(v, Value::Str(_)));
        let inter_is_bool = intermediates.iter().all(|v| matches!(v, Value::Bool(_)));
        let exp_is_num = expected.iter().all(|v| matches!(v, Value::Num(_)));
        let exp_is_str = expected.iter().all(|v| matches!(v, Value::Str(_)));
        let exp_is_bool = expected.iter().all(|v| matches!(v, Value::Bool(_)));
        // Skip if types are incompatible (no function maps num→str or str→num without builtins)
        if (inter_is_num && exp_is_str) || (inter_is_str && exp_is_num && !exp_is_bool) {
            continue;
        }
        // Skip if intermediates are bool — limited composability
        if inter_is_bool && !exp_is_bool {
            continue;
        }

        // Check 4: use intermediates as bridge — synthesize f such that f(m(input)) = expected
        // First, try direct RD on the subspec (cheap — just inversion probing)
        // Only fall back to flat sub-synthesis with a tight budget.
        let remaining = max_candidates.saturating_sub(result.candidates_explored);
        let budget = remaining.min(sub_budget);
        if budget == 0 { break; }

        let sr = recursive_sub_synthesize(
            components, &intermediates, expected, macros,
            max_depth, budget, false, rd_depth, predictor,
        );
        result.candidates_explored += sr.candidates_explored;

        if sr.found {
            let f_nodes = sr.nodes.unwrap();
            let f_root = sr.root.unwrap();

            // Compose: (lambda (x) (f (m x)))
            // The f lambda takes an input; we need to substitute (m x) for f's input.
            // Approach: build (lambda (x) (f_body[x := (m x)]))
            // Simpler: build nodes for (m x), then compose f over it.
            let mut nodes = Vec::new();

            // x symbol
            let x_idx = nodes.len();
            nodes.push(Node::Symbol(intern("x")));

            // m symbol
            let m_sym_idx = nodes.len();
            nodes.push(Node::Symbol(intern(mname)));

            // (m x)
            let m_app_idx = nodes.len();
            nodes.push(Node::App(vec![m_sym_idx, x_idx]));

            // Import f's sub-solution
            let f_offset = nodes.len();
            for nd in &f_nodes {
                nodes.push(synth::remap_node(nd, f_offset));
            }
            let f_root_remapped = f_root + f_offset;

            // f is a lambda — extract its body and substitute its parameter with (m x)
            // f = (lambda (param) body) — we need body[param := (m x)]
            if let Node::Lambda(_, f_body_idx) = &nodes[f_root_remapped] {
                let f_body_idx = *f_body_idx;

                // Substitute: in the body, replace Symbol("x") references
                // with m_app_idx (since f's param is named "x")
                let composed_body = substitute_symbol_idx(&nodes, f_body_idx, intern("x"), m_app_idx);
                let composed_body_idx = nodes.len();
                nodes.push(composed_body);

                let lambda_idx = nodes.len();
                nodes.push(Node::Lambda(vec![intern("x")], composed_body_idx));

                if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                    result.found = true;
                    result.nodes = nodes;
                    result.root = lambda_idx;
                    result.strategy_used = format!("RD(f∘{})", mname);
                    return result;
                }
            }
        }
    }

    result
}

/// Substitute all references to a symbol `target` in a node subtree with `replacement_idx`.
/// Returns a new node (which should be appended to the node list).
fn substitute_symbol_idx(nodes: &[Node], idx: usize, target: crate::intern::Sym, replacement_idx: usize) -> Node {
    match &nodes[idx] {
        Node::Symbol(name) if *name == target => {
            // This case is handled by parents checking children
            Node::Symbol(*name)
        }
        Node::App(children) => {
            let new_children: Vec<usize> = children.iter().map(|&c| {
                if let Node::Symbol(name) = &nodes[c] {
                    if *name == target { return replacement_idx; }
                }
                c
            }).collect();
            Node::App(new_children)
        }
        Node::If(c, t, e) => {
            let nc = if matches!(&nodes[*c], Node::Symbol(n) if *n == target) { replacement_idx } else { *c };
            let nt = if matches!(&nodes[*t], Node::Symbol(n) if *n == target) { replacement_idx } else { *t };
            let ne = if matches!(&nodes[*e], Node::Symbol(n) if *n == target) { replacement_idx } else { *e };
            Node::If(nc, nt, ne)
        }
        Node::Let(bindings, body) => {
            let new_bindings: Vec<(crate::intern::Sym, usize)> = bindings.iter().map(|(name, v)| {
                let ni = if matches!(&nodes[*v], Node::Symbol(n) if *n == target) { replacement_idx } else { *v };
                (*name, ni)
            }).collect();
            let nb = if matches!(&nodes[*body], Node::Symbol(n) if *n == target) { replacement_idx } else { *body };
            Node::Let(new_bindings, nb)
        }
        other => other.clone(),
    }
}

/// Secondary family candidates to try after the primary family.
fn secondary_family_candidates(primary: Family, features: &SpecFeatures) -> Vec<&'static str> {
    match primary {
        // If we predicted arithmetic but it's str→num, also try string ops
        Family::Arithmetic if features.input_type == "str" => {
            vec!["string-length", "count-char"]
        }
        // If we predicted string-op, also try arithmetic on string lengths
        Family::StringOp => {
            vec!["string-length"]
        }
        _ => vec![],
    }
}

/// Recursive sub-synthesis: try RD first (if depth > 0), then flat synthesis.
/// Returns a SynthResult-compatible tuple (found, nodes, root, candidates).
fn recursive_sub_synthesize(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    enable_if: bool,
    rd_depth: usize,
    predictor: Option<&LearnedPredictor>,
) -> synth::SynthResult {
    // At rd_depth > 0, try recursive decomposition first
    if rd_depth > 0 && inputs.len() >= 2 {
        let rd_budget = max_candidates / 3; // reserve 2/3 for flat fallback
        let rd = try_rd_recursive(
            components, inputs, expected, macros,
            max_depth, rd_budget, predictor, rd_depth - 1,
        );
        if rd.found {
            return synth::SynthResult {
                found: true,
                nodes: Some(rd.nodes),
                root: Some(rd.root),
                candidates_explored: rd.candidates_explored,
            };
        }
        // RD failed — fall through to flat with remaining budget
        let remaining = max_candidates.saturating_sub(rd.candidates_explored);
        let sr = synth::synthesize(
            components, inputs, expected, macros,
            max_depth, remaining, enable_if,
        );
        return synth::SynthResult {
            candidates_explored: sr.candidates_explored + rd.candidates_explored,
            ..sr
        };
    }

    // rd_depth == 0: flat synthesis only
    synth::synthesize(components, inputs, expected, macros, max_depth, max_candidates, enable_if)
}

/// Try a single outermost function: invert → sub-synthesize → compose → verify.
fn try_single_function(
    fn_name: &str,
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    features: &SpecFeatures,
    rd_depth: usize,
    predictor: Option<&LearnedPredictor>,
) -> RecursiveDecompResult {
    let mut result = RecursiveDecompResult::empty();

    // Determine function arity and try inversion
    match fn_name {
        // ── Unary arithmetic ──
        // Only attempt on numeric/string inputs — list inputs need different strategies
        "negate" | "abs" | "floor" | "ceil" if features.input_type != "list" => {
            if let Some(subspec) = invert_unary_arith(fn_name, inputs, expected) {
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, max_candidates, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_unary(fn_name, &sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = format!("RD({})", fn_name);
                        return result;
                    }
                }
            }
        }

        // ── Binary arithmetic ──
        "add" | "subtract" | "multiply" | "divide" | "modulo" | "pow"
            if features.input_type != "list" => {
            // Try f(input, k) = output
            if let Some(subspec) = invert_binary_arith(fn_name, inputs, expected) {
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, max_candidates / 2, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_binary_input_first(
                        fn_name, &sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = format!("RD({} x k)", fn_name);
                        return result;
                    }
                }
            }
            // Try f(k, input) = output
            if let Some(subspec) = invert_binary_arith_reversed(fn_name, inputs, expected) {
                let remaining = max_candidates.saturating_sub(result.candidates_explored);
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, remaining.min(max_candidates / 2), false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_binary_input_second(
                        fn_name, &sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = format!("RD({} k x)", fn_name);
                        return result;
                    }
                }
            }
        }

        // ── String-take / string-drop ──
        "string-take" => {
            if let Some(subspec) = invert_string_take(inputs, expected) {
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, max_candidates, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_binary_input_first(
                        "string-take", &sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = "RD(string-take)".to_string();
                        return result;
                    }
                }
            }
        }
        "string-drop" => {
            if let Some(subspec) = invert_string_drop(inputs, expected) {
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, max_candidates, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_binary_input_first(
                        "string-drop", &sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = "RD(string-drop)".to_string();
                        return result;
                    }
                }
            }
        }

        // ── Concat ──
        "concat" => {
            let concat_specs = invert_concat(inputs, expected);
            for (subspec, reversed) in concat_specs {
                let remaining = max_candidates.saturating_sub(result.candidates_explored);
                if remaining == 0 { break; }
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, remaining / 2, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = if reversed {
                        compose_binary_input_second("concat", &sub_nodes, sub_root)
                    } else {
                        compose_binary_input_first("concat", &sub_nodes, sub_root)
                    };
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = format!("RD(concat{})",
                            if reversed { " k x" } else { " x k" });
                        return result;
                    }
                }
            }
        }

        // ── Unary string ops ──
        "string-upper" | "string-lower" | "string-reverse" | "string-trim" => {
            // First check if direct application works (no sub-synthesis needed)
            let direct_match = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
                match eval::apply_builtin(intern(fn_name), &[inp.clone()]) {
                    Ok(ref v) => vals_equal(v, exp),
                    Err(_) => false,
                }
            });
            if direct_match {
                // Build (lambda (x) (fn_name x))
                let mut nodes = Vec::new();
                let x_idx = nodes.len();
                nodes.push(Node::Symbol(intern("x")));
                let f_idx = nodes.len();
                nodes.push(Node::Symbol(intern(fn_name)));
                let app_idx = nodes.len();
                nodes.push(Node::App(vec![f_idx, x_idx]));
                let lambda_idx = nodes.len();
                nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                result.found = true;
                result.nodes = nodes;
                result.root = lambda_idx;
                result.strategy_used = format!("RD({} direct)", fn_name);
                return result;
            }

            // Try inversion for composition: output = f(g(x))
            if let Some(subspec) = invert_unary_string(fn_name, inputs, expected) {
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, max_candidates, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_unary(fn_name, &sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = format!("RD({})", fn_name);
                        return result;
                    }
                }
            }
        }

        // ── String-replace ──
        "string-replace" => {
            // string-replace is arity-3, complex inversion — skip for now
        }

        // ── Count functions ──
        "string-length" => {
            // output = string-length(g(x)) — only useful if output is numeric
            // and input is string. g(x) is a string→string function.
            // Can't invert without knowing what string has that length.
            // But: if input is string and output = string-length(input), check directly.
            if features.input_type == "str" && features.output_type == "num" {
                let direct_match = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
                    match eval::apply_builtin(intern("string-length"), &[inp.clone()]) {
                        Ok(ref v) => vals_equal(v, exp),
                        Err(_) => false,
                    }
                });
                if direct_match {
                    let mut nodes = Vec::new();
                    let x_idx = nodes.len();
                    nodes.push(Node::Symbol(intern("x")));
                    let f_idx = nodes.len();
                    nodes.push(Node::Symbol(intern("string-length")));
                    let app_idx = nodes.len();
                    nodes.push(Node::App(vec![f_idx, x_idx]));
                    let lambda_idx = nodes.len();
                    nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                    result.found = true;
                    result.nodes = nodes;
                    result.root = lambda_idx;
                    result.strategy_used = "RD(string-length direct)".to_string();
                    return result;
                }
            }
        }

        "count-char" => {
            if let Some(subspec) = invert_count_char(inputs, expected) {
                // subspec.expected is the constant character for each example.
                // If they're all the same, we can build (count-char x "ch") directly.
                let all_same = subspec.expected.windows(2).all(|w| vals_equal(&w[0], &w[1]));
                if all_same {
                    if let Value::Str(ch) = &subspec.expected[0] {
                        let mut nodes = Vec::new();
                        let x_idx = nodes.len();
                        nodes.push(Node::Symbol(intern("x")));
                        let ch_idx = nodes.len();
                        nodes.push(Node::Str(ch.clone()));
                        let f_idx = nodes.len();
                        nodes.push(Node::Symbol(intern("count-char")));
                        let app_idx = nodes.len();
                        nodes.push(Node::App(vec![f_idx, x_idx, ch_idx]));
                        let lambda_idx = nodes.len();
                        nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                        if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                            result.found = true;
                            result.nodes = nodes;
                            result.root = lambda_idx;
                            result.strategy_used = format!("RD(count-char \"{}\")", ch);
                            return result;
                        }
                    }
                }
            }
        }

        // ── Comparison/predicate functions ──
        "string-ends-with" | "string-starts-with" | "contains" => {
            // These are binary functions returning bool.
            // Try: output = f(input, k) where k is a constant string.
            if features.output_type == "bool" && features.input_type == "str" {
                // Collect unique substrings from inputs that could be the constant
                let candidate_strings = extract_candidate_constants(inputs, expected, fn_name);
                for cs in &candidate_strings {
                    let matches = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
                        match eval::apply_builtin(
                            intern(fn_name),
                            &[inp.clone(), Value::Str(cs.clone())]
                        ) {
                            Ok(ref v) => vals_equal(v, exp),
                            Err(_) => false,
                        }
                    });
                    if matches {
                        let mut nodes = Vec::new();
                        let x_idx = nodes.len();
                        nodes.push(Node::Symbol(intern("x")));
                        let c_idx = nodes.len();
                        nodes.push(Node::Str(cs.clone()));
                        let f_idx = nodes.len();
                        nodes.push(Node::Symbol(intern(fn_name)));
                        let app_idx = nodes.len();
                        nodes.push(Node::App(vec![f_idx, x_idx, c_idx]));
                        let lambda_idx = nodes.len();
                        nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = format!("RD({} \"{}\")", fn_name, cs);
                        return result;
                    }
                }
            }
        }

        "even" | "odd" => {
            // Direct application check
            if features.output_type == "bool" && features.input_type == "num" {
                let direct_match = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
                    match eval::apply_builtin(intern(fn_name), &[inp.clone()]) {
                        Ok(ref v) => vals_equal(v, exp),
                        Err(_) => false,
                    }
                });
                if direct_match {
                    let mut nodes = Vec::new();
                    let x_idx = nodes.len();
                    nodes.push(Node::Symbol(intern("x")));
                    let f_idx = nodes.len();
                    nodes.push(Node::Symbol(intern(fn_name)));
                    let app_idx = nodes.len();
                    nodes.push(Node::App(vec![f_idx, x_idx]));
                    let lambda_idx = nodes.len();
                    nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                    result.found = true;
                    result.nodes = nodes;
                    result.root = lambda_idx;
                    result.strategy_used = format!("RD({} direct)", fn_name);
                    return result;
                }
            }
        }

        // ── Boolean composition ──
        "and" | "or" | "not" => {
            // Handled by the existing BD strategy — skip to avoid duplication
        }

        // ── Higher-order: map ──
        "map" => {
            // Try list-map: (lambda (x) (map HOLE x))
            // Applicable when all inputs/outputs are lists of same length
            if let Some(subspec) = invert_map_list(inputs, expected) {
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, max_candidates, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_map(&sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = "RD(map)".to_string();
                        return result;
                    }
                }
            }
            // Try split-map-join for string inputs
            if features.input_type == "str" && features.output_type == "str" {
                for &sep in &[" ", ",", "-", ".", "/", ":", ";", "_", "|"] {
                    if let Some(subspec) = invert_split_map_join(inputs, expected, sep) {
                        let remaining = max_candidates.saturating_sub(result.candidates_explored);
                        if remaining == 0 { break; }
                        let sr = recursive_sub_synthesize(
                            components, &subspec.inputs, &subspec.expected, macros,
                            max_depth, remaining / 4, false, rd_depth, predictor,
                        );
                        result.candidates_explored += sr.candidates_explored;
                        if sr.found {
                            let sub_nodes = sr.nodes.unwrap();
                            let sub_root = sr.root.unwrap();
                            let (nodes, lambda_idx) = compose_split_map_join(
                                &sub_nodes, sub_root, sep);
                            if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                                result.found = true;
                                result.nodes = nodes;
                                result.root = lambda_idx;
                                result.strategy_used = format!("RD(split-map-join \"{}\")", sep);
                                return result;
                            }
                        }
                    }
                }
            }
        }

        // ── Higher-order: filter ──
        "filter" => {
            if let Some(subspec) = invert_filter(inputs, expected) {
                let sr = recursive_sub_synthesize(
                    components, &subspec.inputs, &subspec.expected, macros,
                    max_depth, max_candidates, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_filter(&sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = "RD(filter)".to_string();
                        return result;
                    }
                }
            }
        }

        // ── If-expression (D&C) ──
        "if" => {
            // Delegate to divide-and-conquer
            let dr = crate::divide::divide_and_conquer(
                components, inputs, expected, macros, max_depth, max_candidates,
            );
            result.candidates_explored += dr.candidates_explored;
            if dr.found {
                result.found = true;
                result.nodes = dr.nodes;
                result.root = dr.root;
                result.strategy_used = "RD(if/D&C)".to_string();
                return result;
            }
        }

        // ── Generic fallback: try evaluation-based inversion for any binary function ──
        _ => {
            let gen_result = try_generic_binary_inversion(
                fn_name, components, inputs, expected, macros,
                max_depth, max_candidates.saturating_sub(result.candidates_explored),
                rd_depth, predictor,
            );
            result.candidates_explored += gen_result.candidates_explored;
            if gen_result.found {
                return RecursiveDecompResult {
                    candidates_explored: result.candidates_explored,
                    ..gen_result
                };
            }
        }
    }

    result
}

/// Extract candidate constant strings for comparison functions.
/// For starts-with: try common prefixes. For ends-with: common suffixes.
fn extract_candidate_constants(
    inputs: &[Value],
    expected: &[Value],
    fn_name: &str,
) -> Vec<String> {
    let mut candidates = Vec::new();

    // Separate true and false examples
    let true_inputs: Vec<&str> = inputs.iter().zip(expected.iter())
        .filter_map(|(i, e)| {
            if let (Value::Str(s), Value::Bool(true)) = (i, e) {
                Some(s.as_str())
            } else { None }
        }).collect();
    let false_inputs: Vec<&str> = inputs.iter().zip(expected.iter())
        .filter_map(|(i, e)| {
            if let (Value::Str(s), Value::Bool(false)) = (i, e) {
                Some(s.as_str())
            } else { None }
        }).collect();

    if true_inputs.is_empty() {
        return candidates;
    }

    match fn_name {
        "string-starts-with" => {
            // Find common prefixes of true inputs
            for len in 1..=true_inputs[0].len() {
                let prefix = &true_inputs[0][..len];
                if true_inputs.iter().all(|s| s.starts_with(prefix))
                    && false_inputs.iter().all(|s| !s.starts_with(prefix))
                {
                    candidates.push(prefix.to_string());
                }
            }
        }
        "string-ends-with" => {
            // Find common suffixes of true inputs
            for len in 1..=true_inputs[0].len() {
                let suffix = &true_inputs[0][true_inputs[0].len() - len..];
                if true_inputs.iter().all(|s| s.ends_with(suffix))
                    && false_inputs.iter().all(|s| !s.ends_with(suffix))
                {
                    candidates.push(suffix.to_string());
                }
            }
        }
        "contains" => {
            // Try substrings from true inputs
            for s in &true_inputs {
                for len in 1..=s.len().min(10) {
                    for start in 0..=s.len() - len {
                        let sub = &s[start..start + len];
                        if true_inputs.iter().all(|t| t.contains(sub))
                            && false_inputs.iter().all(|t| !t.contains(sub))
                        {
                            if !candidates.contains(&sub.to_string()) {
                                candidates.push(sub.to_string());
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    // Prefer shorter candidates
    candidates.sort_by_key(|s| s.len());
    candidates.truncate(5);
    candidates
}

// ── Generic function evaluation ────────────────────────────────────

/// Evaluate any function (builtin or promoted macro) on given arguments.
/// Returns None on error or if the function is not found.
fn eval_function(
    fn_name: &str,
    args: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> Option<Value> {
    // Try builtin first
    if let Ok(v) = eval::apply_builtin(intern(fn_name), args) {
        return Some(v);
    }
    // Try as promoted macro
    for (mname, mparams, mnodes, mroot) in macros {
        if mname == fn_name && mparams.len() == args.len() {
            let nodes_rc: Rc<[Node]> = mnodes.clone().into();
            let fv = Value::RustMacro(
                mparams.iter().map(|s| intern(s)).collect(),
                nodes_rc.clone(),
                *mroot,
            );
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in macros {
                env_define(&mut env, intern(nm),
                    Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), mn.clone().into(), *mr));
            }
            return eval::apply(&fv, args, &nodes_rc, &mut env).ok();
        }
    }
    None
}

/// Generate candidate constant values for inversion probing.
/// These are values that might appear as the second argument to a binary function.
fn generate_inversion_candidates(
    inputs: &[Value],
    expected: &[Value],
) -> Vec<Value> {
    let mut candidates: Vec<Value> = Vec::new();
    let mut seen_nums: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut seen_strs: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut add_num = |n: f64, seen: &mut std::collections::HashSet<i64>| -> Option<Value> {
        let ni = n as i64;
        if (ni as f64 - n).abs() < 1e-9 && seen.insert(ni) {
            Some(Value::Num(n))
        } else {
            None
        }
    };

    // Small integers
    for i in 0..=10 {
        if seen_nums.insert(i) {
            candidates.push(Value::Num(i as f64));
        }
    }

    // Numbers derived from inputs and outputs
    for vals in [inputs, expected] {
        for v in vals {
            match v {
                Value::Num(n) => {
                    if let Some(v) = add_num(*n, &mut seen_nums) { candidates.push(v); }
                    if let Some(v) = add_num(n.abs(), &mut seen_nums) { candidates.push(v); }
                    if *n != 0.0 {
                        if let Some(v) = add_num(n / 2.0, &mut seen_nums) { candidates.push(v); }
                    }
                }
                Value::Str(s) => {
                    let len = s.len() as f64;
                    if let Some(v) = add_num(len, &mut seen_nums) { candidates.push(v); }
                    if let Some(v) = add_num((len / 2.0).floor(), &mut seen_nums) { candidates.push(v); }
                    // Individual characters as strings
                    for ch in s.chars() {
                        let cs = ch.to_string();
                        if seen_strs.insert(cs.clone()) {
                            candidates.push(Value::Str(cs));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Numeric differences and ratios between paired inputs/outputs
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        if let (Value::Num(a), Value::Num(b)) = (inp, exp) {
            if let Some(v) = add_num(b - a, &mut seen_nums) { candidates.push(v); }
            if let Some(v) = add_num(a - b, &mut seen_nums) { candidates.push(v); }
            if *a != 0.0 {
                if let Some(v) = add_num(b / a, &mut seen_nums) { candidates.push(v); }
            }
        }
    }

    // Substrings of string inputs (prefixes and suffixes up to length 10)
    for v in inputs {
        if let Value::Str(s) = v {
            for len in 1..=s.len().min(10) {
                // Prefix
                let prefix = &s[..len.min(s.len())];
                if seen_strs.insert(prefix.to_string()) {
                    candidates.push(Value::Str(prefix.to_string()));
                }
                // Suffix
                if s.len() >= len {
                    let suffix = &s[s.len() - len..];
                    if seen_strs.insert(suffix.to_string()) {
                        candidates.push(Value::Str(suffix.to_string()));
                    }
                }
            }
        }
    }

    // Booleans
    candidates.push(Value::Bool(true));
    candidates.push(Value::Bool(false));

    // Common string constants
    for s in &[" ", ",", "-", ".", "/", ":", ";", "_", "|", ""] {
        if seen_strs.insert(s.to_string()) {
            candidates.push(Value::Str(s.to_string()));
        }
    }

    candidates
}

// ── Generic inversion: any binary function ─────────────────────────

/// Try to invert any binary function via evaluation probing.
///
/// For each candidate constant k, evaluates f(input, k) or f(k, input) and
/// checks against expected output. If a constant works for all examples,
/// builds the solution directly. If per-example k values are found, creates
/// a subspec for sub-synthesis.
fn try_generic_binary_inversion(
    fn_name: &str,
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    rd_depth: usize,
    predictor: Option<&LearnedPredictor>,
) -> RecursiveDecompResult {
    let mut result = RecursiveDecompResult::empty();
    let candidates = generate_inversion_candidates(inputs, expected);

    // ── Constant-k: f(input, k) ──
    for k in &candidates {
        let all_match = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
            eval_function(fn_name, &[inp.clone(), k.clone()], macros)
                .map(|v| vals_equal(&v, exp))
                .unwrap_or(false)
        });
        if all_match {
            // Build (lambda (x) (f x k))
            let mut nodes = Vec::new();
            let x_idx = nodes.len();
            nodes.push(Node::Symbol(intern("x")));
            let k_idx = nodes.len();
            nodes.push(match k {
                Value::Num(n) => Node::Num(*n),
                Value::Str(s) => Node::Str(s.clone()),
                Value::Bool(b) => Node::Bool(*b),
                _ => continue,
            });
            let f_idx = nodes.len();
            nodes.push(Node::Symbol(intern(fn_name)));
            let app_idx = nodes.len();
            nodes.push(Node::App(vec![f_idx, x_idx, k_idx]));
            let lambda_idx = nodes.len();
            nodes.push(Node::Lambda(vec![intern("x")], app_idx));
            if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                result.found = true;
                result.nodes = nodes;
                result.root = lambda_idx;
                result.strategy_used = format!("RD({} x k=const generic)", fn_name);
                return result;
            }
        }
    }

    // ── Constant-k: f(k, input) ──
    for k in &candidates {
        let all_match = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
            eval_function(fn_name, &[k.clone(), inp.clone()], macros)
                .map(|v| vals_equal(&v, exp))
                .unwrap_or(false)
        });
        if all_match {
            // Build (lambda (x) (f k x))
            let mut nodes = Vec::new();
            let x_idx = nodes.len();
            nodes.push(Node::Symbol(intern("x")));
            let k_idx = nodes.len();
            nodes.push(match k {
                Value::Num(n) => Node::Num(*n),
                Value::Str(s) => Node::Str(s.clone()),
                Value::Bool(b) => Node::Bool(*b),
                _ => continue,
            });
            let f_idx = nodes.len();
            nodes.push(Node::Symbol(intern(fn_name)));
            let app_idx = nodes.len();
            nodes.push(Node::App(vec![f_idx, k_idx, x_idx]));
            let lambda_idx = nodes.len();
            nodes.push(Node::Lambda(vec![intern("x")], app_idx));
            if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                result.found = true;
                result.nodes = nodes;
                result.root = lambda_idx;
                result.strategy_used = format!("RD({} k=const x generic)", fn_name);
                return result;
            }
        }
    }

    // ── Variable-k: f(input, g(input)) — derive per-example k values ──
    // For each example, find the first candidate k that satisfies f(input, k) = output
    let mut per_example_k: Vec<Value> = Vec::with_capacity(inputs.len());
    let mut all_found = true;
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        let mut found_k = None;
        for k in &candidates {
            if let Some(v) = eval_function(fn_name, &[inp.clone(), k.clone()], macros) {
                if vals_equal(&v, exp) {
                    found_k = Some(k.clone());
                    break;
                }
            }
        }
        if let Some(k) = found_k {
            per_example_k.push(k);
        } else {
            all_found = false;
            break;
        }
    }

    if all_found && !per_example_k.is_empty() {
        // Check if all k values are the same type (synthesizable)
        // and not all the same (already handled by constant-k path above)
        let all_same = per_example_k.windows(2).all(|w| vals_equal(&w[0], &w[1]));
        if !all_same {
            let sr = recursive_sub_synthesize(
                components, inputs, &per_example_k, macros,
                max_depth, max_candidates / 2, false, rd_depth, predictor,
            );
            result.candidates_explored += sr.candidates_explored;
            if sr.found {
                let sub_nodes = sr.nodes.unwrap();
                let sub_root = sr.root.unwrap();
                let (nodes, lambda_idx) = compose_binary_input_first(fn_name, &sub_nodes, sub_root);
                if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                    result.found = true;
                    result.nodes = nodes;
                    result.root = lambda_idx;
                    result.strategy_used = format!("RD({} x g(x) generic)", fn_name);
                    return result;
                }
            }
        }
    }

    // ── Variable-k reversed: f(g(input), input) ──
    let mut per_example_k_rev: Vec<Value> = Vec::with_capacity(inputs.len());
    let mut all_found_rev = true;
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        let mut found_k = None;
        for k in &candidates {
            if let Some(v) = eval_function(fn_name, &[k.clone(), inp.clone()], macros) {
                if vals_equal(&v, exp) {
                    found_k = Some(k.clone());
                    break;
                }
            }
        }
        if let Some(k) = found_k {
            per_example_k_rev.push(k);
        } else {
            all_found_rev = false;
            break;
        }
    }

    if all_found_rev && !per_example_k_rev.is_empty() {
        let all_same = per_example_k_rev.windows(2).all(|w| vals_equal(&w[0], &w[1]));
        if !all_same {
            let remaining = max_candidates.saturating_sub(result.candidates_explored);
            if remaining > 0 {
                let sr = recursive_sub_synthesize(
                    components, inputs, &per_example_k_rev, macros,
                    max_depth, remaining / 2, false, rd_depth, predictor,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = compose_binary_input_second(fn_name, &sub_nodes, sub_root);
                    if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        result.strategy_used = format!("RD({} g(x) x generic)", fn_name);
                        return result;
                    }
                }
            }
        }
    }

    result
}

// ── Generic inversion: macros as outermost ─────────────────────────

/// Try promoted macros as outermost functions via inversion probing.
///
/// For each unary macro m, tries to find per-example values v_i such that
/// m(v_i) = expected_i, then sub-synthesizes g such that g(input_i) = v_i.
/// Composes as (lambda (x) (m (g x))).
///
/// This is the inverse direction of try_macro_decomposition, which does f(m(x)).
fn try_macro_as_outermost(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    rd_depth: usize,
    predictor: Option<&LearnedPredictor>,
) -> RecursiveDecompResult {
    let mut result = RecursiveDecompResult::empty();
    let sub_budget = max_candidates / 8;

    // Precompute per-example intermediate values for all unary functions
    // (builtins + macros). These serve as candidate pre-images for inversion.
    // Each entry: (fn_name, [intermediate_for_example_0, ...])
    let mut intermediate_table: Vec<(String, Vec<Value>)> = Vec::new();

    // Collect unary builtins from components
    for comp in components {
        if comp.arity != 1 { continue; }
        let fname = comp.name.clone();
        let comp_sym = intern(&comp.name);
        let mut intermediates = Vec::with_capacity(inputs.len());
        let mut valid = true;
        for inp in inputs {
            match eval::apply_builtin(comp_sym, &[inp.clone()]) {
                Ok(v) => intermediates.push(v),
                Err(_) => { valid = false; break; }
            }
        }
        if valid && intermediates.len() == inputs.len() {
            intermediate_table.push((fname, intermediates));
        }
    }

    // Collect unary macros
    for (mname, mparams, mnodes, mroot) in macros {
        if mparams.len() != 1 { continue; }
        if mname.starts_with("__selph_") { continue; }
        let nodes_rc: Rc<[Node]> = mnodes.clone().into();
        let fv = Value::RustMacro(
            mparams.iter().map(|s| intern(s)).collect(),
            nodes_rc.clone(),
            *mroot,
        );
        let mut intermediates = Vec::with_capacity(inputs.len());
        let mut valid = true;
        for inp in inputs {
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in macros {
                env_define(&mut env, intern(nm),
                    Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), mn.clone().into(), *mr));
            }
            match eval::apply(&fv, &[inp.clone()], &nodes_rc, &mut env) {
                Ok(v) => intermediates.push(v),
                Err(_) => { valid = false; break; }
            }
        }
        if valid && intermediates.len() == inputs.len() {
            intermediate_table.push((mname.clone(), intermediates));
        }
    }

    // For each unary macro m (candidate outermost), check if m(g(x)) = expected
    // by probing m on precomputed intermediates
    for (mname, mparams, _mnodes, _mroot) in macros {
        if mparams.len() != 1 { continue; }
        if mname.starts_with("__selph_") { continue; }
        if result.candidates_explored >= max_candidates { break; }

        // For each inner function g, check if m(g(input)) = expected for all examples
        for (gname, g_intermediates) in &intermediate_table {
            // Skip self-composition (m(m(x))) — usually not useful
            if gname == mname { continue; }

            let all_match = g_intermediates.iter().zip(expected.iter()).all(|(inter, exp)| {
                let r = eval_function(mname, &[inter.clone()], macros);
                let ok = r.as_ref().map(|v| vals_equal(v, exp)).unwrap_or(false);
                ok
            });

            if all_match {
                // Found: m(g(input)) = expected where g = gname
                // Build (lambda (x) (m (g x)))
                let mut nodes = Vec::new();
                let x_idx = nodes.len();
                nodes.push(Node::Symbol(intern("x")));
                let g_idx = nodes.len();
                nodes.push(Node::Symbol(intern(gname)));
                let g_app_idx = nodes.len();
                nodes.push(Node::App(vec![g_idx, x_idx]));
                let m_idx = nodes.len();
                nodes.push(Node::Symbol(intern(mname)));
                let m_app_idx = nodes.len();
                nodes.push(Node::App(vec![m_idx, g_app_idx]));
                let lambda_idx = nodes.len();
                nodes.push(Node::Lambda(vec![intern("x")], m_app_idx));

                if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                    result.found = true;
                    result.nodes = nodes;
                    result.root = lambda_idx;
                    result.strategy_used = format!("RD({}∘{} generic)", mname, gname);
                    return result;
                }
            }
        }

        // Fallback: sub-synthesize g(x) such that m(g(x)) = expected
        // First, compute what g(x) needs to produce by finding pre-images of m
        // Try a small sub-synthesis where intermediates → expected via m
        // We need to find values v_i such that m(v_i) = expected_i
        // Use the intermediate_table values as a candidate pool
        let mut per_example_v: Vec<Value> = Vec::with_capacity(inputs.len());
        let mut all_found = true;

        for exp in expected.iter() {
            let mut found_v = None;
            // Search through all computed intermediates for any value that m maps to exp
            for (_gname, g_intermediates) in &intermediate_table {
                for inter in g_intermediates {
                    if let Some(v) = eval_function(mname, &[inter.clone()], macros) {
                        if vals_equal(&v, exp) {
                            found_v = Some(inter.clone());
                            break;
                        }
                    }
                }
                if found_v.is_some() { break; }
            }
            // Also try simple constant candidates
            if found_v.is_none() {
                let simple_candidates = generate_inversion_candidates(inputs, expected);
                for k in &simple_candidates {
                    if let Some(v) = eval_function(mname, &[k.clone()], macros) {
                        if vals_equal(&v, exp) {
                            found_v = Some(k.clone());
                            break;
                        }
                    }
                }
            }
            if let Some(v) = found_v {
                per_example_v.push(v);
            } else {
                all_found = false;
                break;
            }
        }

        if !all_found || per_example_v.is_empty() { continue; }

        // Skip if v values equal the inputs (already handled by try_macro_decomposition)
        if per_example_v.iter().zip(inputs.iter()).all(|(v, i)| vals_equal(v, i)) {
            continue;
        }

        // Sub-synthesize g such that g(input_i) = v_i
        let remaining = max_candidates.saturating_sub(result.candidates_explored);
        let budget = remaining.min(sub_budget);
        if budget == 0 { break; }

        let sr = recursive_sub_synthesize(
            components, inputs, &per_example_v, macros,
            max_depth, budget, false, rd_depth, predictor,
        );
        result.candidates_explored += sr.candidates_explored;

        if sr.found {
            let sub_nodes = sr.nodes.unwrap();
            let sub_root = sr.root.unwrap();
            let (nodes, lambda_idx) = compose_unary(mname, &sub_nodes, sub_root);
            if verify_composed(&nodes, lambda_idx, inputs, expected, macros) {
                result.found = true;
                result.nodes = nodes;
                result.root = lambda_idx;
                result.strategy_used = format!("RD({}(g(x)) generic)", mname);
                return result;
            }
        }
    }

    result
}

// ── HO inversion helpers ───────────────────────────────────────────

/// Invert map on lists: derive per-element subspec.
/// Applicable when all inputs/outputs are lists of same length.
fn invert_map_list(inputs: &[Value], expected: &[Value]) -> Option<SubSpec> {
    let mut sub_inputs = Vec::new();
    let mut sub_expected = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (il, ol) = match (inp, out) {
            (Value::List(il), Value::List(ol)) if il.len() == ol.len() && !il.is_empty() => (il, ol),
            _ => return None,
        };
        for (ie, oe) in il.iter().zip(ol.iter()) {
            sub_inputs.push(ie.clone());
            sub_expected.push(oe.clone());
        }
    }
    // Deduplicate — check for conflicts
    let mut seen: std::collections::HashMap<u64, (Value, Value)> = std::collections::HashMap::new();
    let mut unique_inputs = Vec::new();
    let mut unique_expected = Vec::new();
    for (inp, exp) in sub_inputs.iter().zip(sub_expected.iter()) {
        let h = synth::val_hash(inp);
        if let Some((prev_i, prev_e)) = seen.get(&h) {
            if vals_equal(prev_i, inp) && !vals_equal(prev_e, exp) {
                return None; // conflict
            }
            if vals_equal(prev_i, inp) { continue; } // duplicate
        }
        seen.insert(h, (inp.clone(), exp.clone()));
        unique_inputs.push(inp.clone());
        unique_expected.push(exp.clone());
    }
    if unique_inputs.len() < 2 { return None; }
    Some(SubSpec { inputs: unique_inputs, expected: unique_expected })
}

/// Invert split-map-join: derive per-part subspec from string splitting.
fn invert_split_map_join(inputs: &[Value], expected: &[Value], sep: &str) -> Option<SubSpec> {
    let mut sub_inputs = Vec::new();
    let mut sub_expected = Vec::new();
    let mut any_multi = false;
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (si, so) = match (inp, out) {
            (Value::Str(si), Value::Str(so)) => (si, so),
            _ => return None,
        };
        let in_parts: Vec<&str> = si.split(sep).collect();
        let out_parts: Vec<&str> = so.split(sep).collect();
        if in_parts.len() != out_parts.len() { return None; }
        if in_parts.len() >= 2 { any_multi = true; }
        for (ip, op) in in_parts.iter().zip(out_parts.iter()) {
            sub_inputs.push(Value::Str(ip.to_string()));
            sub_expected.push(Value::Str(op.to_string()));
        }
    }
    if !any_multi { return None; }
    // Deduplicate
    let mut seen: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut unique_inputs = Vec::new();
    let mut unique_expected = Vec::new();
    for (inp, exp) in sub_inputs.iter().zip(sub_expected.iter()) {
        if let (Value::Str(si), Value::Str(se)) = (inp, exp) {
            if let Some(prev) = seen.get(si) {
                if prev != se { return None; } // conflict
                continue;
            }
            seen.insert(si.clone(), se.clone());
            unique_inputs.push(inp.clone());
            unique_expected.push(exp.clone());
        }
    }
    if unique_inputs.len() < 2 { return None; }
    Some(SubSpec { inputs: unique_inputs, expected: unique_expected })
}

/// Invert filter: derive boolean labels for each element.
fn invert_filter(inputs: &[Value], expected: &[Value]) -> Option<SubSpec> {
    let mut sub_inputs = Vec::new();
    let mut sub_expected = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (il, ol) = match (inp, out) {
            (Value::List(il), Value::List(ol)) if !il.is_empty() => (il, ol),
            _ => return None,
        };
        let mut out_idx = 0;
        for elem in il {
            let keep = if out_idx < ol.len() && vals_equal(elem, &ol[out_idx]) {
                out_idx += 1;
                true
            } else {
                false
            };
            sub_inputs.push(elem.clone());
            sub_expected.push(Value::Bool(keep));
        }
        if out_idx != ol.len() { return None; }
    }
    // Deduplicate
    let mut seen: std::collections::HashMap<u64, (Value, Value)> = std::collections::HashMap::new();
    let mut unique_inputs = Vec::new();
    let mut unique_expected = Vec::new();
    for (inp, exp) in sub_inputs.iter().zip(sub_expected.iter()) {
        let h = synth::val_hash(inp);
        if let Some((prev_i, prev_e)) = seen.get(&h) {
            if vals_equal(prev_i, inp) && !vals_equal(prev_e, exp) {
                return None; // conflict
            }
            if vals_equal(prev_i, inp) { continue; }
        }
        seen.insert(h, (inp.clone(), exp.clone()));
        unique_inputs.push(inp.clone());
        unique_expected.push(exp.clone());
    }
    if unique_inputs.len() < 3 { return None; }
    Some(SubSpec { inputs: unique_inputs, expected: unique_expected })
}

/// Compose (lambda (x) (map sub-fn x))
fn compose_map(sub_nodes: &[Node], sub_root: usize) -> (Vec<Node>, usize) {
    let mut nodes = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_body = extract_body(&nodes, sub_root + sub_offset);

    // Wrap sub_body back in a lambda for map's function argument
    let inner_lambda = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], sub_body));

    let map_sym = nodes.len();
    nodes.push(Node::Symbol(intern("map")));
    let map_app = nodes.len();
    nodes.push(Node::App(vec![map_sym, inner_lambda, x_idx]));
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], map_app));

    (nodes, lambda_idx)
}

/// Compose (lambda (x) (string-join (map sub-fn (string-split x SEP)) SEP))
fn compose_split_map_join(sub_nodes: &[Node], sub_root: usize, sep: &str) -> (Vec<Node>, usize) {
    let mut nodes = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sep_idx = nodes.len();
    nodes.push(Node::Str(sep.to_string()));

    let split_sym = nodes.len();
    nodes.push(Node::Symbol(intern("string-split")));
    let split_app = nodes.len();
    nodes.push(Node::App(vec![split_sym, x_idx, sep_idx]));

    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_body = extract_body(&nodes, sub_root + sub_offset);

    let inner_lambda = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], sub_body));

    let map_sym = nodes.len();
    nodes.push(Node::Symbol(intern("map")));
    let map_app = nodes.len();
    nodes.push(Node::App(vec![map_sym, inner_lambda, split_app]));

    let sep2_idx = nodes.len();
    nodes.push(Node::Str(sep.to_string()));
    let join_sym = nodes.len();
    nodes.push(Node::Symbol(intern("string-join")));
    let join_app = nodes.len();
    nodes.push(Node::App(vec![join_sym, map_app, sep2_idx]));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], join_app));

    (nodes, lambda_idx)
}

/// Compose (lambda (x) (filter sub-fn x))
fn compose_filter(sub_nodes: &[Node], sub_root: usize) -> (Vec<Node>, usize) {
    let mut nodes = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(synth::remap_node(nd, sub_offset));
    }
    let sub_body = extract_body(&nodes, sub_root + sub_offset);

    let inner_lambda = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], sub_body));

    let filter_sym = nodes.len();
    nodes.push(Node::Symbol(intern("filter")));
    let filter_app = nodes.len();
    nodes.push(Node::App(vec![filter_sym, inner_lambda, x_idx]));
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], filter_app));

    (nodes, lambda_idx)
}

// ── Learned predictor support ──────────────────────────────────────

/// Extract the outermost function name from a solved program's AST.
/// `root` can be a Lambda (from synthesis results) or a direct body node
/// (from defmacro storage in all_macros).
/// Returns None for constants, identity, or unparseable structures.
pub fn extract_outermost_function(nodes: &[Node], root: usize) -> Option<String> {
    // If root is a Lambda, unwrap to body
    let body_idx = match &nodes[root] {
        Node::Lambda(_, body) => *body,
        _ => root, // direct body (from defmacro storage)
    };
    match &nodes[body_idx] {
        Node::App(children) if !children.is_empty() => {
            match &nodes[children[0]] {
                Node::Symbol(sym) => Some(resolve(*sym)),
                _ => None,
            }
        }
        Node::If(..) => Some("if".to_string()),
        _ => None,
    }
}

/// Classify an outermost function name into a family label string.
pub fn classify_outermost(fn_name: &str) -> &'static str {
    match fn_name {
        "add" | "subtract" | "multiply" | "divide" | "modulo" | "pow"
        | "abs" | "negate" | "floor" | "ceil" | "round" | "sqrt" | "log"
        | "min" | "max" => "arith",

        "string-length" | "count-char" => "count",

        "string-upper" | "string-lower" | "string-reverse" | "string-trim"
        | "string-take" | "string-drop" | "concat" | "string-replace"
        | "string-join" | "chars" | "string-chars" | "string-split" => "string-op",

        "string-ends-with" | "string-starts-with" | "contains"
        | "<" | ">" | "<=" | ">=" | "=" | "even" | "odd" => "compare",

        "and" | "or" | "not" => "bool-comp",

        "map" | "reduce" | "filter" => "ho",

        "if" => "if-expr",

        "ns-get" | "ns-get-or" => "memo",

        _ => "delegate", // promoted macro as outermost
    }
}

/// Encode spec features as a compact string for use as synthesis input.
/// Format: "input_type:output_type:n_distinct:has_bool:is_substr"
pub fn encode_features(
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> String {
    let features = extract_spec_features(inputs, expected, macros);
    format!("{}:{}:{}:{}:{}",
        features.input_type,
        features.output_type,
        features.num_distinct_outputs.min(9), // cap for simpler patterns
        if features.has_bool_macros { 1 } else { 0 },
        if features.output_is_substring { 1 } else { 0 },
    )
}

/// A training pair for the family predictor: (encoded_features, family_label).
#[derive(Debug, Clone)]
pub struct PredictorTrainingPair {
    pub features: String,
    pub family: String,
}

/// Learned predictor: a SELPH program that maps feature string → family string.
pub struct LearnedPredictor {
    pub nodes: Vec<Node>,
    pub root: usize,
}

impl LearnedPredictor {
    /// Parse a predictor from SELPH source code.
    pub fn from_source(source: &str) -> Option<Self> {
        use crate::parser::parse_file;
        let (nodes, roots) = parse_file(source).ok()?;
        if roots.is_empty() { return None; }
        Some(LearnedPredictor {
            root: roots[roots.len() - 1],
            nodes,
        })
    }

    /// Evaluate the predictor on encoded features. Returns a family string.
    pub fn predict(&self, encoded_features: &str) -> Option<String> {
        let nodes_rc: Rc<[Node]> = self.nodes.clone().into();
        let mut env = eval::make_default_env();
        let fv = eval::eval(&nodes_rc, self.root, &mut env).ok()?;
        let result = eval::apply(
            &fv,
            &[Value::Str(encoded_features.to_string())],
            &nodes_rc,
            &mut env,
        ).ok()?;
        match result {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Convert a family string back to Family enum.
fn family_from_str(s: &str) -> Family {
    match s {
        "arith" => Family::Arithmetic,
        "count" => Family::Count,
        "string-op" => Family::StringOp,
        "compare" => Family::Compare,
        "bool-comp" => Family::BoolComp,
        "ho" => Family::HigherOrder,
        "if-expr" => Family::IfExpr,
        "memo" => Family::Arithmetic, // no dedicated family — fallback
        "delegate" => Family::Arithmetic, // promoted macro — try arith
        "const" => Family::Constant,
        _ => Family::Arithmetic, // fallback
    }
}

/// Synthesize a family predictor from training pairs.
/// Deduplicates by majority vote, tries D&C first, then memo fallback.
pub fn learn_predictor(
    training: &[PredictorTrainingPair],
    budget: usize,
) -> Option<(Vec<Node>, usize, usize)> {
    if training.is_empty() { return None; }

    // Deduplicate: for each unique feature string, pick the most common family
    let mut votes: std::collections::HashMap<String, std::collections::HashMap<String, usize>>
        = std::collections::HashMap::new();
    for pair in training {
        *votes.entry(pair.features.clone())
            .or_default()
            .entry(pair.family.clone())
            .or_insert(0) += 1;
    }
    let deduped: Vec<PredictorTrainingPair> = votes.into_iter().map(|(features, family_votes)| {
        let family = family_votes.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(f, _)| f)
            .unwrap();
        PredictorTrainingPair { features, family }
    }).collect();

    if deduped.is_empty() { return None; }

    let inputs: Vec<Value> = deduped.iter()
        .map(|p| Value::Str(p.features.clone()))
        .collect();
    let expected: Vec<Value> = deduped.iter()
        .map(|p| Value::Str(p.family.clone()))
        .collect();

    // Strategy 1: D&C (decision tree) — generalizes if patterns exist
    let components = synth::default_synth_components_opts(&[], false);
    let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    let dr = crate::divide::divide_and_conquer(
        &components, &inputs, &expected, &macros, 2, budget / 2,
    );
    if dr.found {
        return Some((dr.nodes, dr.root, dr.candidates_explored));
    }

    // Strategy 2: Memorization — always works, grows with training data.
    // Build: (lambda (x) (ns-get-or (ns ("k1" "v1") ("k2" "v2") ...) x "arith"))
    let mut nodes = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let mut ns_children = vec![];
    let ns_sym = nodes.len();
    nodes.push(Node::Symbol(intern("ns")));
    ns_children.push(ns_sym);

    for pair in &deduped {
        let key_idx = nodes.len();
        nodes.push(Node::Str(pair.features.clone()));
        let val_idx = nodes.len();
        nodes.push(Node::Str(pair.family.clone()));
        // ns expects pairs: (ns ("key" "val") ...)
        let pair_idx = nodes.len();
        nodes.push(Node::App(vec![key_idx, val_idx]));
        ns_children.push(pair_idx);
    }

    let ns_app = nodes.len();
    nodes.push(Node::App(ns_children));

    let default_idx = nodes.len();
    nodes.push(Node::Str("arith".to_string()));

    let ngo_sym = nodes.len();
    nodes.push(Node::Symbol(intern("ns-get-or")));

    let lookup_app = nodes.len();
    nodes.push(Node::App(vec![ngo_sym, ns_app, x_idx, default_idx]));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], lookup_app));

    Some((nodes, lambda_idx, 0))
}

/// Predict family using a learned predictor, falling back to the hand-coded tree.
fn predict_family_with_learned(
    features: &SpecFeatures,
    predictor: Option<&LearnedPredictor>,
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> Family {
    if let Some(pred) = predictor {
        let encoded = encode_features(inputs, expected, macros);
        if let Some(family_str) = pred.predict(&encoded) {
            return family_from_str(&family_str);
        }
    }
    // Fallback to hand-coded decision tree
    predict_family(features)
}
