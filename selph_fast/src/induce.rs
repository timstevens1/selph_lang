//! Failure-driven induction for SELPH.
//!
//! When synthesis fails on a task, this module analyzes the failure and
//! attempts to decompose the spec into sub-problems that can be solved
//! independently, then composes the solutions into a new primitive.
//!
//! Two strategies:
//!   1. Intermediate value decomposition: find y = g(x) such that
//!      f(y) = output, then synthesize g and f independently.
//!   2. Constant discovery: analyze input/output pairs to discover
//!      useful constants (differences, ratios), then retry synthesis.

use crate::eval;
use crate::synth::{self, SynthComponent};
use crate::types::*;

// ── Result type ─────────────────────────────────────────────────────

/// Result of failure-driven induction.
pub struct InductionResult {
    pub found: bool,
    pub nodes: Vec<Node>,
    pub root: usize,
    pub candidates_explored: usize,
    /// Human-readable description of how the program was found.
    pub decomposition: String,
}

impl InductionResult {
    fn empty() -> Self {
        InductionResult {
            found: false,
            nodes: Vec::new(),
            root: 0,
            candidates_explored: 0,
            decomposition: String::new(),
        }
    }
}

// ── Main entry point ────────────────────────────────────────────────

/// Attempt to solve a failed synthesis task by decomposition.
///
/// Tries two strategies in order:
///   1. Intermediate value decomposition
///   2. Constant discovery
///
/// Returns an `InductionResult` with the composed program if successful.
pub fn induce_from_failure(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> InductionResult {
    if inputs.is_empty() || expected.is_empty() || inputs.len() != expected.len() {
        return InductionResult::empty();
    }

    // Strategy 1: intermediate value decomposition
    let iv = try_intermediate_values(
        components, inputs, expected, macros, max_depth, max_candidates,
    );
    if iv.found {
        return iv;
    }
    let mut total_explored = iv.candidates_explored;

    // Strategy 2: constant discovery
    let remaining = max_candidates.saturating_sub(total_explored);
    let cd = try_constant_discovery(
        components, inputs, expected, macros, max_depth, remaining,
    );
    total_explored += cd.candidates_explored;
    if cd.found {
        return InductionResult {
            candidates_explored: total_explored,
            ..cd
        };
    }

    InductionResult {
        candidates_explored: total_explored,
        ..InductionResult::empty()
    }
}

// ── Strategy 1: Intermediate value decomposition ────────────────────

/// Unary builtin functions to try as intermediate transformations.
const UNARY_FNS: &[&str] = &[
    "abs", "negate", "floor", "ceil",
    "string-upper", "string-lower", "string-reverse", "string-trim",
    "string-length",
];

/// Binary builtin functions paired with small constants.
const BINARY_FNS: &[&str] = &["add", "subtract", "multiply"];
const SMALL_CONSTANTS: &[f64] = &[1.0, 2.0, -1.0, 0.5];

/// Named set of intermediate values produced by running a function on inputs.
struct IntermediateSet {
    name: String,
    values: Vec<Value>,
}

/// Generate candidate intermediate values by running known functions on inputs.
fn generate_intermediates(inputs: &[Value]) -> Vec<IntermediateSet> {
    let mut results = Vec::new();

    // Unary functions
    for &fn_name in UNARY_FNS {
        let mut values = Vec::with_capacity(inputs.len());
        let mut valid = true;
        for inp in inputs {
            match eval::apply_builtin(fn_name, &[inp.clone()]) {
                Ok(v) => values.push(v),
                Err(_) => { valid = false; break; }
            }
        }
        if valid && values.len() == inputs.len() {
            results.push(IntermediateSet {
                name: fn_name.to_string(),
                values,
            });
        }
    }

    // Binary functions with small constants
    for &fn_name in BINARY_FNS {
        for &c in SMALL_CONSTANTS {
            let const_val = Value::Num(c);
            let mut values = Vec::with_capacity(inputs.len());
            let mut valid = true;
            for inp in inputs {
                match eval::apply_builtin(fn_name, &[inp.clone(), const_val.clone()]) {
                    Ok(v) => values.push(v),
                    Err(_) => { valid = false; break; }
                }
            }
            if valid && values.len() == inputs.len() {
                results.push(IntermediateSet {
                    name: format!("{}_{}", fn_name, c),
                    values,
                });
            }
        }
    }

    results
}

/// Check if all values in a slice are the same (constant function -- not useful).
fn all_identical(vals: &[Value]) -> bool {
    if vals.len() <= 1 {
        return true;
    }
    vals.windows(2).all(|w| synth::vals_equal(&w[0], &w[1]))
}

/// Check if two value slices are element-wise equal.
fn slices_equal(a: &[Value], b: &[Value]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| synth::vals_equal(x, y))
}

fn try_intermediate_values(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> InductionResult {
    let mut result = InductionResult::empty();
    let intermediates = generate_intermediates(inputs);
    let budget_per_step = max_candidates / 4;

    for iset in &intermediates {
        // Skip if intermediates are identical to inputs or outputs
        if slices_equal(&iset.values, inputs) || slices_equal(&iset.values, expected) {
            continue;
        }
        // Skip constant functions
        if all_identical(&iset.values) {
            continue;
        }

        // Step 2: try to synthesize intermediate -> expected output
        let step2 = synth::synthesize(
            components, &iset.values, expected, macros,
            max_depth, budget_per_step, false,
        );
        result.candidates_explored += step2.candidates_explored;

        if !step2.found {
            continue;
        }

        // Step 1: try to synthesize input -> intermediate
        let step1 = synth::synthesize(
            components, inputs, &iset.values, macros,
            max_depth, budget_per_step, false,
        );
        result.candidates_explored += step1.candidates_explored;

        if !step1.found {
            continue;
        }

        // Both steps solved -- compose them.
        let step1_nodes = step1.nodes.unwrap();
        let step1_root = step1.root.unwrap();
        let step2_nodes = step2.nodes.unwrap();
        let step2_root = step2.root.unwrap();

        match compose_steps(
            &step1_nodes, step1_root,
            &step2_nodes, step2_root,
            inputs, expected, macros,
        ) {
            Some((nodes, root)) => {
                result.found = true;
                result.nodes = nodes;
                result.root = root;
                result.decomposition = format!(
                    "Decomposed via {}: step1 -> intermediate, step2 -> output",
                    iset.name,
                );
                return result;
            }
            None => continue,
        }
    }

    result
}

/// Compose two synthesized lambda programs into (lambda (x) (step2 (step1 x))).
///
/// Both step1 and step2 are stored as node trees ending with
/// Lambda(["x"], body_root). We substitute step1's body into step2's
/// parameter reference.
fn compose_steps(
    step1_nodes: &[Node], step1_lambda: usize,
    step2_nodes: &[Node], step2_lambda: usize,
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> Option<(Vec<Node>, usize)> {
    // step1_lambda and step2_lambda point to Lambda nodes.
    // Extract their body roots.
    let step1_body = match &step1_nodes[step1_lambda] {
        Node::Lambda(_, body) => *body,
        _ => return None,
    };
    let step2_body = match &step2_nodes[step2_lambda] {
        Node::Lambda(_, body) => *body,
        _ => return None,
    };

    // Build a combined node tree:
    //   [step1_nodes...] [remapped step2_nodes...] [substituted_body] [Lambda]
    //
    // In step2, every reference to Symbol("x") should be replaced with
    // the body expression from step1 (which itself references x).

    // Start with step1's nodes (indices 0..step1_nodes.len())
    let mut nodes = step1_nodes.to_vec();

    // Append step2's nodes, remapped by offset
    let step2_offset = nodes.len();
    for nd in step2_nodes {
        nodes.push(synth::remap_node(nd, step2_offset));
    }
    let step2_body_remapped = step2_body + step2_offset;

    // Now substitute: in step2's body subtree, replace Symbol("x") with
    // a reference to step1's body.
    let composed_body = substitute_x(&nodes, step2_body_remapped, step1_body);
    let composed_root_idx = nodes.len();
    nodes.push(composed_body);

    // Wrap in lambda
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec!["x".into()], composed_root_idx));

    // Verify the composed program produces correct outputs
    let macro_env: Vec<(String, Vec<String>, Vec<Node>, usize)> = macros.to_vec();
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        let mut env = eval::make_default_env();
        for (nm, ps, mn, mr) in &macro_env {
            env_define(
                &mut env,
                nm.clone(),
                Value::RustMacro(ps.clone(), mn.clone(), *mr),
            );
        }
        let fv = match eval::eval(&nodes, lambda_idx, &mut env) {
            Ok(v) => v,
            Err(_) => return None,
        };
        match eval::apply(&fv, &[inp.clone()], &nodes, &mut env) {
            Ok(ref v) if synth::vals_equal(v, exp) => {}
            _ => return None,
        }
    }

    Some((nodes, lambda_idx))
}

/// Recursively substitute Symbol("x") in a node subtree with a reference
/// to `replacement_idx`. Returns a new node (not added to the vec).
///
/// Since nodes reference children by index, and we want to replace the
/// leaf Symbol("x") references in the step2 subtree with step1's body,
/// we rebuild the relevant node pointing to `replacement_idx` where
/// it previously pointed to a Symbol("x") node.
fn substitute_x(nodes: &[Node], idx: usize, replacement_idx: usize) -> Node {
    match &nodes[idx] {
        Node::Symbol(name) if name == "x" => {
            // Instead of returning the symbol, return a reference that
            // the parent will use. Since we can't return "an index",
            // we handle this at the App/If/Let level.
            // This case is handled by the callers checking children.
            Node::Symbol("x".into()) // sentinel -- handled by parent
        }
        Node::App(children) => {
            let new_children: Vec<usize> = children.iter().map(|&c| {
                if is_x_symbol(nodes, c) { replacement_idx } else { c }
            }).collect();
            Node::App(new_children)
        }
        Node::If(c, t, e) => {
            let nc = if is_x_symbol(nodes, *c) { replacement_idx } else { *c };
            let nt = if is_x_symbol(nodes, *t) { replacement_idx } else { *t };
            let ne = if is_x_symbol(nodes, *e) { replacement_idx } else { *e };
            Node::If(nc, nt, ne)
        }
        Node::Let(bindings, body) => {
            let new_bindings: Vec<(String, usize)> = bindings.iter().map(|(name, idx_val)| {
                let ni = if is_x_symbol(nodes, *idx_val) { replacement_idx } else { *idx_val };
                (name.clone(), ni)
            }).collect();
            let nb = if is_x_symbol(nodes, *body) { replacement_idx } else { *body };
            Node::Let(new_bindings, nb)
        }
        other => other.clone(),
    }
}

/// Check if nodes[idx] is Symbol("x").
fn is_x_symbol(nodes: &[Node], idx: usize) -> bool {
    matches!(&nodes[idx], Node::Symbol(name) if name == "x")
}

// ── Strategy 2: Constant discovery ──────────────────────────────────

fn try_constant_discovery(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> InductionResult {
    let mut result = InductionResult::empty();

    // Only works for all-numeric tasks
    let nums_in: Vec<f64> = match inputs.iter().map(|v| match v {
        Value::Num(n) => Some(*n),
        _ => None,
    }).collect::<Option<Vec<f64>>>() {
        Some(v) => v,
        None => return result,
    };

    let nums_out: Vec<f64> = match expected.iter().map(|v| match v {
        Value::Num(n) => Some(*n),
        _ => None,
    }).collect::<Option<Vec<f64>>>() {
        Some(v) => v,
        None => return result,
    };

    let mut discovered: Vec<f64> = Vec::new();

    // Check for constant difference: output - input
    {
        let diffs: Vec<f64> = nums_in.iter().zip(nums_out.iter())
            .map(|(i, o)| o - i)
            .collect();
        if !diffs.is_empty() && diffs.windows(2).all(|w| w[0] == w[1]) {
            discovered.push(diffs[0]);
        }
    }

    // Check for constant ratio: output / input
    {
        let ratios: Vec<Option<f64>> = nums_in.iter().zip(nums_out.iter())
            .map(|(i, o)| if *i != 0.0 { Some(o / i) } else { None })
            .collect();
        if let Some(ratios) = ratios.into_iter().collect::<Option<Vec<f64>>>() {
            if !ratios.is_empty() && ratios.windows(2).all(|w| w[0] == w[1]) {
                discovered.push(ratios[0]);
            }
        }
    }

    // Also add per-pair differences and ratios as candidates
    for (i, o) in nums_in.iter().zip(nums_out.iter()) {
        if *i != 0.0 {
            let ratio = o / i;
            if ratio == ratio.floor() {
                discovered.push(ratio);
            }
        }
        discovered.push(o - i);
    }

    // Deduplicate and keep only integer constants
    discovered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    discovered.dedup();
    discovered.retain(|c| *c == c.floor() && *c != 0.0);

    if discovered.is_empty() {
        return result;
    }

    // Build augmented component set with discovered constants
    let mut augmented = components.to_vec();
    for c in &discovered {
        // Skip if this constant already exists in components
        let name = if *c == (*c as i64) as f64 {
            format!("{}", *c as i64)
        } else {
            format!("{}", c)
        };
        let already_exists = components.iter().any(|comp| comp.name == name);
        if already_exists {
            continue;
        }
        augmented.push(SynthComponent {
            name,
            builtin: None,
            arity: 0,
            ret_type: 0,
            param_types: vec![],
            priority: 0.0,
        });
    }

    // Retry synthesis with augmented components
    let synth_result = synth::synthesize(
        &augmented, inputs, expected, macros,
        max_depth, max_candidates, false,
    );
    result.candidates_explored += synth_result.candidates_explored;

    if synth_result.found {
        let const_strs: Vec<String> = discovered.iter().map(|c| {
            if *c == (*c as i64) as f64 { format!("{}", *c as i64) }
            else { format!("{}", c) }
        }).collect();

        result.found = true;
        result.nodes = synth_result.nodes.unwrap();
        result.root = synth_result.root.unwrap();
        result.decomposition = format!(
            "Solved with discovered constants: [{}]",
            const_strs.join(", "),
        );
    }

    result
}
