//! Bottom-up enumerative program synthesizer for SELPH.
//!
//! Given input/output examples, searches for the smallest program that
//! satisfies them. Uses type pruning, observational equivalence dedup,
//! and (optionally) hash-based if-expression generation.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use crate::types::*;
use crate::eval;
use crate::hm;

// ── Type tag constants ──────────────────────────────────────────────

pub const TYPE_NUM: u8 = 0;
pub const TYPE_STR: u8 = 1;
pub const TYPE_BOOL: u8 = 2;
pub const TYPE_ANY: u8 = 255;

// ── Domain / Kind classification (§12) ──────────────────────────────

/// Top-level domain: what "world" a component operates in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    Number,
    String,
    Logic,
    List,
    Generic,
}

/// Operation kind: what role a component plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Constants,
    Transform,
    Arithmetic,
    Compare,
    Predicate,
    Compose,
    Analyze,
    Branch,
    Other,
}

/// Classify a type tag into a Domain.
pub fn type_tag_domain(tag: u8) -> Domain {
    match tag {
        TYPE_NUM  => Domain::Number,
        TYPE_STR  => Domain::String,
        TYPE_BOOL => Domain::Logic,
        TYPE_ANY  => Domain::Generic,
        _         => Domain::Generic,
    }
}

/// Classify a component into a Domain based on its primary input type
/// (or return type for constants).
pub fn component_domain(comp: &SynthComponent) -> Domain {
    if comp.arity == 0 {
        return type_tag_domain(comp.ret_type);
    }
    if !comp.param_types.is_empty() {
        type_tag_domain(comp.param_types[0])
    } else {
        Domain::Generic
    }
}

/// Classify a component into a Kind based on its name and arity.
pub fn component_kind(comp: &SynthComponent) -> Kind {
    let name = comp.name.as_str();

    // Constants (arity 0)
    if comp.arity == 0 {
        return Kind::Constants;
    }

    // Branching operations
    if ["thresh", "relu", "branch", "if_even", "if_odd",
        "classify", "piecewise", "abs_double",
        "negate_or", "abs_then", "relu_then", "negate_abs"]
        .iter().any(|kw| name.contains(kw))
    {
        return Kind::Branch;
    }

    // String transforms
    if ["string-upper", "string-lower", "string-reverse", "string-trim"]
        .contains(&name)
    {
        return Kind::Transform;
    }

    // String/list analysis (returns a different type than input)
    if ["string-length", "string-contains", "string-starts-with",
        "string-ends-with", "string-split", "length", "empty?", "contains"]
        .contains(&name)
    {
        return Kind::Analyze;
    }

    // Arithmetic
    if ["add", "subtract", "multiply", "divide", "modulo",
        "negate", "abs", "double", "floor", "ceil",
        "min", "max", "to-number"]
        .contains(&name)
    {
        return Kind::Arithmetic;
    }

    // Comparison
    if ["<", ">", "<=", ">=", "=", "!="].contains(&name) {
        return Kind::Compare;
    }

    // Logic / predicates
    if ["not", "even", "odd", "and", "or"].contains(&name) {
        return Kind::Predicate;
    }

    // Composition / higher-order
    if ["compose", "pipe", "map", "filter", "reduce", "apply"]
        .contains(&name)
    {
        return Kind::Compose;
    }

    Kind::Other
}

// ── Namespace-based scoping (§12.5) ─────────────────────────────────

/// Filter components based on the synthesis context.
///
/// Mirrors the Python `scope_for_context()`: given a target output type
/// and input type (as type tags), returns only the components whose
/// signatures are compatible.  This dramatically reduces the search
/// space by not trying string ops when building numeric expressions, etc.
///
/// Rules:
///   - A component is included when its return type matches `target_output`
///     (or either side is `TYPE_ANY`).
///   - When `input_type` is given, the component's first param must match
///     (or either side is `TYPE_ANY`).
///   - Constants (arity 0) are always included unless their return type
///     conflicts with `target_output`.
///   - Generic/cross-type components (e.g. `to-string`, `string-length`)
///     are kept when they bridge between the requested input and output
///     domains.
pub fn scope_for_context<'a>(
    components: &'a [SynthComponent],
    target_output: Option<u8>,
    input_type: Option<u8>,
) -> Vec<&'a SynthComponent> {
    components.iter().filter(|comp| {
        // Return-type filter
        if let Some(target) = target_output {
            if target != TYPE_ANY && comp.ret_type != TYPE_ANY && comp.ret_type != target {
                return false;
            }
        }

        // Input-type filter (first param)
        if let Some(inp) = input_type {
            if comp.arity > 0 && !comp.param_types.is_empty() {
                let first_param = comp.param_types[0];
                if inp != TYPE_ANY && first_param != TYPE_ANY && first_param != inp {
                    return false;
                }
            }
        }

        true
    }).collect()
}

/// Return components suitable as if-expression conditions.
/// Only keeps components that return bool.
pub fn scope_for_condition(components: &[SynthComponent]) -> Vec<&SynthComponent> {
    components.iter().filter(|comp| {
        comp.ret_type == TYPE_BOOL
    }).collect()
}

/// Return components suitable as if-expression branches.
/// Prioritises branch-specific ops, transforms, arithmetic, then constants.
pub fn scope_for_branch<'a>(
    components: &'a [SynthComponent],
    input_type: Option<u8>,
) -> Vec<&'a SynthComponent> {
    let mut result: Vec<&SynthComponent> = Vec::new();
    let mut seen: HashSet<*const SynthComponent> = HashSet::new();

    let accept = |comp: &SynthComponent| -> bool {
        if let Some(inp) = input_type {
            if comp.arity > 0 && !comp.param_types.is_empty() {
                let first_param = comp.param_types[0];
                if inp != TYPE_ANY && first_param != TYPE_ANY && first_param != inp {
                    return false;
                }
            }
        }
        true
    };

    // Priority 1: branch operations
    for comp in components {
        if component_kind(comp) == Kind::Branch && accept(comp) {
            if seen.insert(comp as *const _) {
                result.push(comp);
            }
        }
    }
    // Priority 2: transforms and arithmetic
    for comp in components {
        let k = component_kind(comp);
        if (k == Kind::Transform || k == Kind::Arithmetic) && accept(comp) {
            if seen.insert(comp as *const _) {
                result.push(comp);
            }
        }
    }
    // Priority 3: constants
    for comp in components {
        if component_kind(comp) == Kind::Constants {
            if seen.insert(comp as *const _) {
                result.push(comp);
            }
        }
    }

    result
}

// ── Core structs ────────────────────────────────────────────────────

/// A named primitive or function available for synthesis.
#[derive(Clone)]
pub struct SynthComponent {
    pub name: String,
    pub builtin: Option<String>,
    pub arity: usize,
    /// 0=num, 1=str, 2=bool, 255=any
    pub ret_type: u8,
    pub param_types: Vec<u8>,
    pub priority: f64,
}

/// A candidate program stored as a flattened node tree.
#[derive(Clone)]
pub struct SynthPool {
    pub nodes: Vec<Node>,
    pub root: usize,
    pub ret_type: u8,
    pub priority: f64,
}

/// Result of a synthesis search.
pub struct SynthResult {
    pub found: bool,
    pub nodes: Option<Vec<Node>>,
    pub root: Option<usize>,
    pub candidates_explored: usize,
}

impl SynthResult {
    fn empty() -> Self {
        SynthResult {
            found: false,
            nodes: None,
            root: None,
            candidates_explored: 0,
        }
    }

    fn success(nodes: Vec<Node>, root: usize, explored: usize) -> Self {
        SynthResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: explored,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Infer the type tag for a runtime Value.
pub fn value_type_tag(v: &Value) -> u8 {
    match v {
        Value::Num(_) => TYPE_NUM,
        Value::Str(_) => TYPE_STR,
        Value::Bool(_) => TYPE_BOOL,
        _ => TYPE_ANY,
    }
}

/// Infer a uniform type tag from a slice of Values.
/// Returns `Some(tag)` if all values share the same type, `None` otherwise.
pub fn infer_uniform_type(values: &[Value]) -> Option<u8> {
    if values.is_empty() {
        return None;
    }
    let first = value_type_tag(&values[0]);
    if values.iter().all(|v| value_type_tag(v) == first) {
        Some(first)
    } else {
        None
    }
}

/// Remap all index references in a node by adding `offset`.
pub fn remap_node(node: &Node, offset: usize) -> Node {
    match node {
        Node::App(c) => Node::App(c.iter().map(|i| i + offset).collect()),
        Node::If(c, t, e) => Node::If(c + offset, t + offset, e + offset),
        Node::Lambda(p, b) => Node::Lambda(p.clone(), b + offset),
        Node::Let(bs, b) => Node::Let(
            bs.iter().map(|(n, i)| (n.clone(), i + offset)).collect(),
            b + offset,
        ),
        other => other.clone(),
    }
}

/// Hash a Value for observational equivalence dedup.
pub fn val_hash(v: &Value) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match v {
        Value::Num(n) => { 0u8.hash(&mut h); n.to_bits().hash(&mut h); }
        Value::Str(s) => { 1u8.hash(&mut h); s.hash(&mut h); }
        Value::Bool(b) => { 2u8.hash(&mut h); b.hash(&mut h); }
        Value::List(items) => {
            4u8.hash(&mut h);
            for item in items {
                val_hash(item).hash(&mut h);
            }
        }
        _ => { 3u8.hash(&mut h); }
    }
    h.finish()
}

/// Check if two Values are equal (structural equality for synthesis).
pub fn vals_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        (Value::List(xs), Value::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(a, b)| vals_equal(a, b))
        }
        _ => false,
    }
}

// ── HM type-aware pruning ────────────────────────────────────────────

/// Convert a SynthPool's u8 ret_type into an HM Type.
/// For TYPE_ANY, returns a fresh type variable.
pub fn pool_hm_type(pool: &SynthPool, counter: &mut u32) -> hm::Type {
    hm::type_from_tag(pool.ret_type, counter)
}

/// HM-based compatibility check: verifies that a component can accept
/// the given argument pool types using Hindley-Milner unification.
///
/// This is used as a second-pass filter after the fast u8 tag check.
/// It catches cases where the u8 system is too coarse (e.g., TYPE_ANY
/// matching everything).
///
/// Returns `true` if the types are compatible.
pub fn hm_check_application(
    comp: &SynthComponent,
    arg_pools: &[&SynthPool],
    counter: &mut u32,
) -> bool {
    // Get the HM type signature for this component.
    let (param_types, ret_type) = if let Some(builtin) = &comp.builtin {
        // Try to get a precise polymorphic signature.
        hm::builtin_type(builtin, counter)
            .unwrap_or_else(|| hm::component_fn_type(comp, counter))
    } else {
        hm::component_fn_type(comp, counter)
    };

    if param_types.len() != arg_pools.len() {
        return false;
    }

    // Convert argument pool types to HM types.
    let arg_types: Vec<hm::Type> = arg_pools.iter()
        .map(|p| hm::type_from_tag(p.ret_type, counter))
        .collect();

    let mut subst = hm::Subst::new();
    hm::type_compatible(&param_types, &ret_type, &arg_types, &mut subst)
}

/// Compute the HM return type of applying a component to the given
/// argument pools. Returns the resolved return type tag (u8) for
/// storing in the resulting SynthPool.
///
/// Falls back to the component's declared ret_type if HM inference
/// doesn't narrow it further.
pub fn hm_infer_ret_type(
    comp: &SynthComponent,
    arg_pools: &[&SynthPool],
    counter: &mut u32,
) -> u8 {
    let (param_types, ret_type) = if let Some(builtin) = &comp.builtin {
        hm::builtin_type(builtin, counter)
            .unwrap_or_else(|| hm::component_fn_type(comp, counter))
    } else {
        hm::component_fn_type(comp, counter)
    };

    if param_types.len() != arg_pools.len() {
        return comp.ret_type;
    }

    let arg_types: Vec<hm::Type> = arg_pools.iter()
        .map(|p| hm::type_from_tag(p.ret_type, counter))
        .collect();

    let mut subst = hm::Subst::new();
    if hm::type_compatible(&param_types, &ret_type, &arg_types, &mut subst) {
        hm::type_to_tag(&ret_type, &subst)
    } else {
        comp.ret_type
    }
}

/// Build macro env from macro tuples for eval.
fn build_macro_env(
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> Vec<(String, Vec<String>, Vec<Node>, usize)> {
    macros.to_vec()
}

// ── SELPH-programmable depth filter ─────────────────────────────────

/// Build a Rust closure from a SELPH scoring/filter function.
///
/// The SELPH function receives a namespace:
///   { "name": str, "priority": num, "arity": num, "ret-type": num,
///     "depth": num }
/// and should return:
///   - A number: used as priority score override (negative = exclude)
///   - A boolean: true = include (keep original priority), false = exclude
///
/// As a filter (backward compat): returns true/false.
/// As a scorer (new): returns a number that replaces the priority.
pub fn make_selph_depth_filter(
    filter_val: Value,
) -> Box<dyn Fn(&SynthComponent, usize) -> bool> {
    Box::new(move |comp: &SynthComponent, depth: usize| -> bool {
        let mut ctx = std::collections::HashMap::new();
        ctx.insert("name".to_string(), Value::Str(comp.name.clone()));
        ctx.insert("priority".to_string(), Value::Num(comp.priority));
        ctx.insert("arity".to_string(), Value::Num(comp.arity as f64));
        ctx.insert("ret-type".to_string(), Value::Num(comp.ret_type as f64));
        ctx.insert("depth".to_string(), Value::Num(depth as f64));
        let ns_arg = Value::Namespace(ctx);

        let empty_nodes: Vec<Node> = Vec::new();
        let mut env = eval::make_default_env();
        match eval::apply(&filter_val, &[ns_arg], &empty_nodes, &mut env) {
            Ok(Value::Bool(b)) => b,
            Ok(Value::Num(n)) => n >= 0.0,  // negative = exclude
            _ => true,
        }
    })
}

/// Build a SELPH scoring function that returns a priority adjustment.
/// Used with interleaved search: the score is added to the base priority.
pub fn make_selph_scorer(
    scorer_val: Value,
) -> Box<dyn Fn(&SynthComponent, usize) -> f64> {
    Box::new(move |comp: &SynthComponent, depth: usize| -> f64 {
        let mut ctx = std::collections::HashMap::new();
        ctx.insert("name".to_string(), Value::Str(comp.name.clone()));
        ctx.insert("priority".to_string(), Value::Num(comp.priority));
        ctx.insert("arity".to_string(), Value::Num(comp.arity as f64));
        ctx.insert("ret-type".to_string(), Value::Num(comp.ret_type as f64));
        ctx.insert("depth".to_string(), Value::Num(depth as f64));
        let ns_arg = Value::Namespace(ctx);

        let empty_nodes: Vec<Node> = Vec::new();
        let mut env = eval::make_default_env();
        match eval::apply(&scorer_val, &[ns_arg], &empty_nodes, &mut env) {
            Ok(Value::Num(n)) => n,
            Ok(Value::Bool(true)) => comp.priority,
            Ok(Value::Bool(false)) => -1.0, // excluded
            _ => comp.priority,
        }
    })
}

// ── Main synthesis entry point ──────────────────────────────────────

/// Bottom-up enumerative synthesis with type pruning, observational
/// equivalence dedup, and optional if-expression generation.
///
/// - `components`: the primitive/function library
/// - `inputs`: example input values
/// - `expected`: expected output values (one per input)
/// - `macros`: library macros (name, params, nodes, root)
/// - `max_depth`: max AST depth to explore
/// - `max_candidates`: budget cap
/// - `enable_if`: when true, generate if-expressions after each depth
///
/// Returns a `SynthResult` with the found program (if any).
pub fn synthesize(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    enable_if: bool,
) -> SynthResult {
    synthesize_with_validation(
        components, inputs, expected, macros,
        max_depth, max_candidates, enable_if, None, &[],
    )
}

/// Like `synthesize`, but with optional held-out validation examples
/// and extra bindings (e.g., from namespace trees).
///
/// When `validation_examples` is `Some`, a candidate that passes all
/// training examples is additionally checked against the validation
/// pairs.  Only candidates that pass both sets are returned as solutions.
///
/// `extra_bindings` are injected into every eval environment so that
/// tree-extracted closures/builtins can be called by their qualified
/// names during candidate evaluation.
pub fn synthesize_with_validation(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    enable_if: bool,
    validation_examples: Option<&[(Value, Value)]>,
    extra_bindings: &[(String, Value)],
) -> SynthResult {
    synthesize_full(components, inputs, expected, macros, max_depth,
        max_candidates, enable_if, validation_examples, extra_bindings, None)
}

/// Full synthesizer with optional SELPH-programmable depth filter.
///
/// `depth_filter` is called once per (component, depth) before the inner loop.
/// It receives `(&SynthComponent, depth)` and returns `true` to include the
/// component at that depth.  This lets SELPH programs control pruning.
pub fn synthesize_full(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    enable_if: bool,
    validation_examples: Option<&[(Value, Value)]>,
    extra_bindings: &[(String, Value)],
    depth_filter: Option<&dyn Fn(&SynthComponent, usize) -> bool>,
) -> SynthResult {
    const MAX_POOL: usize = 100_000;
    let mut explored: usize = 0;
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let macro_env = build_macro_env(macros);

    // ── Namespace-based scoping ─────────────────────────────────────
    // Infer the input/output types from examples, then filter out
    // components that can never participate in a correct program.
    //
    // Key insight: we only filter by *return type* at the top level
    // (the final expression must produce `target_output_type`).  We do
    // NOT filter by input_type globally, because inner sub-expressions
    // can have any type — e.g. (add (idx x) (v0 x)) needs `add` which
    // takes nums even though `x` is a string.
    //
    // We also fix up the `x` variable's type tag to match the actual
    // input type so that type-based composition checks work correctly.
    let target_output_type = infer_uniform_type(expected);
    let input_type = infer_uniform_type(inputs);

    // Fix up `x` variable type: its ret_type should match the actual
    // input type, not always be TYPE_NUM.
    let scoped_components: Vec<&SynthComponent> = {
        // Keep all components — only exclude those that are provably
        // useless:  pure string→string ops when output is numeric, and
        // pure num→num ops when output is string (at depth >= 2 these
        // may still be useful as intermediates, but str→str when we
        // need num can never help).
        let mut scoped: Vec<&SynthComponent> = Vec::new();
        for comp in components.iter() {
            // Fix x's type tag to match actual input type
            // (We can't mutate, so we handle x specially in the pool below)

            // Exclude pure string→string ops when target is numeric
            if let Some(target) = target_output_type {
                if target == TYPE_NUM
                    && comp.ret_type == TYPE_STR
                    && comp.arity > 0
                    && !comp.param_types.is_empty()
                    && comp.param_types[0] == TYPE_STR
                {
                    continue; // e.g. string-upper, string-lower when we need numbers
                }
                // Exclude pure num→num ops when target is string (and no macros bridge)
                if target == TYPE_STR
                    && comp.ret_type == TYPE_NUM
                    && comp.arity > 0
                    && !comp.param_types.is_empty()
                    && comp.param_types[0] == TYPE_NUM
                {
                    continue;
                }
            }
            scoped.push(comp);
        }
        // Always include bool-producing components for if-conditions
        if enable_if {
            let conditions = scope_for_condition(components);
            for c in conditions {
                if !scoped.iter().any(|s| std::ptr::eq(*s, c)) {
                    scoped.push(c);
                }
            }
        }
        scoped
    };

    // Use scoped components for the rest of synthesis.
    let components = &scoped_components;

    // Closure: evaluate a candidate, check against expected outputs,
    // and dedup via behavior hash.
    let test = |entry: &SynthPool,
                seen: &mut HashSet<Vec<u64>>,
                explored: &mut usize|
        -> Option<(Vec<Node>, usize)>
    {
        *explored += 1;
        if *explored > max_candidates {
            return None;
        }

        // Wrap the body in (lambda (x) body)
        let mut ln = entry.nodes.clone();
        let lr = ln.len();
        ln.push(Node::Lambda(vec!["x".into()], entry.root));

        let mut beh = Vec::new();
        let mut ok = true;
        for (inp, exp) in inputs.iter().zip(expected.iter()) {
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in &macro_env {
                env_define(
                    &mut env,
                    nm.clone(),
                    Value::RustMacro(ps.clone(), mn.clone(), *mr),
                );
            }
            for (name, val) in extra_bindings {
                env_define(&mut env, name.clone(), val.clone());
            }
            let fv = match eval::eval(&ln, lr, &mut env) {
                Ok(v) => v,
                Err(_) => { ok = false; break; }
            };
            match eval::apply(&fv, &[inp.clone()], &ln, &mut env) {
                Ok(a) => {
                    beh.push(val_hash(&a));
                    if !vals_equal(&a, exp) { ok = false; }
                }
                Err(_) => { ok = false; break; }
            }
        }

        // Observational equivalence dedup
        if !beh.is_empty() {
            if seen.contains(&beh) { return None; }
            seen.insert(beh);
        }

        if ok && !inputs.is_empty() {
            // Held-out validation: if validation examples provided,
            // check the candidate generalises beyond training data.
            if let Some(val_exs) = validation_examples {
                if !validate_candidate(&ln, lr, val_exs, &macro_env, extra_bindings) {
                    return None; // passes training but fails validation
                }
            }
            Some((ln, lr))
        } else {
            None
        }
    };

    // ── Depth 0: atoms ──────────────────────────────────────────────

    let mut pool: Vec<SynthPool> = Vec::new();
    for comp in components {
        if comp.arity != 0 { continue; }
        let mut nodes = Vec::new();
        if comp.name == "x" {
            nodes.push(Node::Symbol("x".into()));
        } else if let Ok(n) = comp.name.parse::<f64>() {
            nodes.push(Node::Num(n));
        } else if comp.name == "true" {
            nodes.push(Node::Bool(true));
        } else if comp.name == "false" {
            nodes.push(Node::Bool(false));
        } else {
            nodes.push(Node::Str(comp.name.clone()));
        }
        // Fix x's type to match actual input type
        let actual_ret_type = if comp.name == "x" {
            input_type.unwrap_or(comp.ret_type)
        } else {
            comp.ret_type
        };
        pool.push(SynthPool {
            nodes,
            root: 0,
            ret_type: actual_ret_type,
            priority: comp.priority,
        });
    }

    // Test atoms
    for e in &pool {
        if explored >= max_candidates {
            return SynthResult { found: false, nodes: None, root: None, candidates_explored: explored };
        }
        if let Some((n, r)) = test(e, &mut seen, &mut explored) {
            return SynthResult::success(n, r, explored);
        }
    }

    // ── Depth 1..max_depth: compose ─────────────────────────────────

    // HM type variable counter for fresh variables during synthesis.
    let mut hm_counter: u32 = 0;

    let mut prev_start: usize = 0;
    let mut prev_end: usize = pool.len();

    for _depth in 1..=max_depth {
        let mut new_entries: Vec<SynthPool> = Vec::new();
        let prev = prev_start..prev_end;
        let all_end = prev_end;

        // ── Priority-weighted interleaving ────────────────────────────
        // Instead of iterating all combos for component A, then all for B,
        // we generate ALL type-valid candidates first with their priority
        // scores, sort by priority descending, then test in that order.
        // This ensures high-priority combos (e.g. multiply(idx, squares))
        // are tried before low-priority ones (e.g. min(3, 4)), regardless
        // of which component they use.
        //
        // The priority of a candidate = component.priority + sum(arg.priority)
        // This naturally prioritizes compositions of useful components.

        struct PendingCandidate {
            entry: SynthPool,
            score: f64,
        }

        let mut pending: Vec<PendingCandidate> = Vec::new();

        for comp in components {
            if comp.arity == 0 || comp.builtin.is_none() { continue; }
            // SELPH-programmable depth filter (optional hard cutoff)
            if let Some(ref filter) = depth_filter {
                if !filter(comp, _depth) { continue; }
            }
            let bn = comp.builtin.as_ref().unwrap();

            if comp.arity == 1 {
                for pi in prev.clone() {
                    let p = &pool[pi];
                    if p.ret_type != comp.param_types[0] && comp.param_types[0] != 255 {
                        continue;
                    }
                    if !hm_check_application(comp, &[p], &mut hm_counter) {
                        continue;
                    }
                    let inferred_ret = hm_infer_ret_type(comp, &[p], &mut hm_counter);
                    let mut n = p.nodes.clone();
                    let fi = n.len();
                    n.push(Node::Symbol(bn.clone()));
                    let ai = n.len();
                    n.push(Node::App(vec![fi, p.root]));
                    let score = comp.priority + p.priority;
                    pending.push(PendingCandidate {
                        entry: SynthPool {
                            nodes: n, root: ai,
                            ret_type: inferred_ret, priority: score,
                        },
                        score,
                    });
                }
            } else if comp.arity == 2 {
                // Case 1: arg1 from new (prev), arg2 from all
                for pi in prev.clone() {
                    let p1 = &pool[pi];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 {
                        continue;
                    }
                    for ai in 0..all_end {
                        let p2 = &pool[ai];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 {
                            continue;
                        }
                        if !hm_check_application(comp, &[p1, p2], &mut hm_counter) {
                            continue;
                        }
                        let inferred_ret = hm_infer_ret_type(comp, &[p1, p2], &mut hm_counter);
                        let mut n = p1.nodes.clone();
                        let off = n.len();
                        for nd in &p2.nodes {
                            n.push(remap_node(nd, off));
                        }
                        let fi = n.len();
                        n.push(Node::Symbol(bn.clone()));
                        let api = n.len();
                        n.push(Node::App(vec![fi, p1.root, p2.root + off]));
                        let score = comp.priority + p1.priority + p2.priority;
                        pending.push(PendingCandidate {
                            entry: SynthPool {
                                nodes: n, root: api,
                                ret_type: inferred_ret, priority: score,
                            },
                            score,
                        });
                    }
                }
                // Case 2: arg1 from old, arg2 from new (prev)
                for ai in 0..prev_start {
                    let p1 = &pool[ai];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 {
                        continue;
                    }
                    for pi in prev.clone() {
                        let p2 = &pool[pi];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 {
                            continue;
                        }
                        if !hm_check_application(comp, &[p1, p2], &mut hm_counter) {
                            continue;
                        }
                        let inferred_ret = hm_infer_ret_type(comp, &[p1, p2], &mut hm_counter);
                        let mut n = p1.nodes.clone();
                        let off = n.len();
                        for nd in &p2.nodes {
                            n.push(remap_node(nd, off));
                        }
                        let fi = n.len();
                        n.push(Node::Symbol(bn.clone()));
                        let api = n.len();
                        n.push(Node::App(vec![fi, p1.root, p2.root + off]));
                        let score = comp.priority + p1.priority + p2.priority;
                        pending.push(PendingCandidate {
                            entry: SynthPool {
                                nodes: n, root: api,
                                ret_type: inferred_ret, priority: score,
                            },
                            score,
                        });
                    }
                }
            }
        }

        // Sort by priority descending — high-value compositions tested first
        pending.sort_by(|a, b| b.score.partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal));

        // Test in priority order
        for cand in pending {
            if let Some((sn, sr)) = test(&cand.entry, &mut seen, &mut explored) {
                return SynthResult::success(sn, sr, explored);
            }
            if explored > max_candidates {
                return SynthResult { found: false, nodes: None, root: None, candidates_explored: explored };
            }
            if pool.len() + new_entries.len() < MAX_POOL {
                new_entries.push(cand.entry);
            }
        }

        // ── If-expression generation (when enabled) ─────────────────

        if enable_if && !inputs.is_empty() {
            let if_entries = generate_if_programs(
                &pool, 0..pool.len() + new_entries.len(),
                inputs, expected, &macro_env, extra_bindings,
            );
            for e in if_entries {
                if let Some((sn, sr)) = test(&e, &mut seen, &mut explored) {
                    return SynthResult::success(sn, sr, explored);
                }
                if explored > max_candidates {
                    return SynthResult { found: false, nodes: None, root: None, candidates_explored: explored };
                }
                if pool.len() + new_entries.len() < MAX_POOL {
                    new_entries.push(e);
                }
            }
        }

        prev_start = pool.len();
        pool.extend(new_entries);
        prev_end = pool.len();
    }

    SynthResult {
        found: false,
        nodes: None,
        root: None,
        candidates_explored: explored,
    }
}

// ── Validation helper ───────────────────────────────────────────────

/// Check that a candidate (already wrapped as a lambda) produces the
/// expected output for every held-out validation example.
fn validate_candidate(
    nodes: &[Node],
    lambda_root: usize,
    validation_examples: &[(Value, Value)],
    macro_env: &[(String, Vec<String>, Vec<Node>, usize)],
    extra_bindings: &[(String, Value)],
) -> bool {
    for (inp, exp) in validation_examples {
        let mut env = eval::make_default_env();
        for (nm, ps, mn, mr) in macro_env {
            env_define(
                &mut env,
                nm.clone(),
                Value::RustMacro(ps.clone(), mn.clone(), *mr),
            );
        }
        for (name, val) in extra_bindings {
            env_define(&mut env, name.clone(), val.clone());
        }
        let fv = match eval::eval(nodes, lambda_root, &mut env) {
            Ok(v) => v,
            Err(_) => return false,
        };
        match eval::apply(&fv, &[inp.clone()], nodes, &mut env) {
            Ok(a) => {
                if !vals_equal(&a, exp) {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }
    true
}

// ── If-expression generation ────────────────────────────────────────

/// Generate if-expressions using hash-based branch matching.
///
/// Key insight from the Python implementation: instead of brute-force
/// trying all (cond, then, else) triples, we:
///
/// 1. Find all bool-typed programs and evaluate them on inputs to get
///    partition patterns (which inputs are true vs false).
/// 2. Skip trivial partitions (all true or all false).
/// 3. For each partition, use expected outputs to find matching branches:
///    - true_branch must produce expected[i] for all i where partition[i]=true
///    - false_branch must produce expected[i] for all i where partition[i]=false
/// 4. Only combine branches that both match their respective expected outputs.
///
/// This turns O(n^3) brute force into O(n) evaluation + O(small) combination.
fn generate_if_programs(
    pool: &[SynthPool],
    range: std::ops::Range<usize>,
    inputs: &[Value],
    expected: &[Value],
    macro_env: &[(String, Vec<String>, Vec<Node>, usize)],
    extra_bindings: &[(String, Value)],
) -> Vec<SynthPool> {
    let mut results = Vec::new();

    // Separate bool-typed pools (conditions) from value-typed pools (branches)
    let mut bool_pools: Vec<usize> = Vec::new();
    let mut val_pools: Vec<usize> = Vec::new();

    for i in range.clone() {
        if i >= pool.len() { break; }
        let p = &pool[i];
        if p.ret_type == 2 {
            // Bool-typed: potential condition
            // Only useful if it references a variable (contains Symbol "x")
            if pool_has_variable(&p.nodes) {
                bool_pools.push(i);
            }
        } else {
            val_pools.push(i);
        }
    }

    if bool_pools.is_empty() || val_pools.is_empty() || inputs.is_empty() {
        return results;
    }

    // Step 1: Evaluate each bool condition on all inputs to get its partition
    // partition: Vec<bool> for each input
    // Dedup by partition pattern: only keep first condition per pattern.
    let mut partitions: HashMap<Vec<bool>, usize> = HashMap::new(); // pattern -> pool index

    for &bi in &bool_pools {
        let p = &pool[bi];
        let mut ln = p.nodes.clone();
        let lr = ln.len();
        ln.push(Node::Lambda(vec!["x".into()], p.root));

        let mut pattern = Vec::with_capacity(inputs.len());
        let mut valid = true;

        for inp in inputs {
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in macro_env {
                env_define(
                    &mut env,
                    nm.clone(),
                    Value::RustMacro(ps.clone(), mn.clone(), *mr),
                );
            }
            for (name, val) in extra_bindings {
                env_define(&mut env, name.clone(), val.clone());
            }
            let fv = match eval::eval(&ln, lr, &mut env) {
                Ok(v) => v,
                Err(_) => { valid = false; break; }
            };
            match eval::apply(&fv, &[inp.clone()], &ln, &mut env) {
                Ok(Value::Bool(b)) => pattern.push(b),
                Ok(_) => { valid = false; break; }
                Err(_) => { valid = false; break; }
            }
        }

        if !valid { continue; }

        // Skip trivial partitions (all true or all false)
        let any_true = pattern.iter().any(|&b| b);
        let any_false = pattern.iter().any(|&b| !b);
        if !any_true || !any_false { continue; }

        // Keep only first condition per partition pattern
        partitions.entry(pattern).or_insert(bi);
    }

    if partitions.is_empty() {
        return results;
    }

    // Step 2: Evaluate each value-pool candidate on all inputs
    // Store (pool_index, outputs) for branch matching.
    struct BranchData {
        pool_idx: usize,
        outputs: Vec<Value>,
    }
    let mut branches: Vec<BranchData> = Vec::new();

    for &vi in &val_pools {
        let p = &pool[vi];
        let mut ln = p.nodes.clone();
        let lr = ln.len();
        ln.push(Node::Lambda(vec!["x".into()], p.root));

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut valid = true;

        for inp in inputs {
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in macro_env {
                env_define(
                    &mut env,
                    nm.clone(),
                    Value::RustMacro(ps.clone(), mn.clone(), *mr),
                );
            }
            for (name, val) in extra_bindings {
                env_define(&mut env, name.clone(), val.clone());
            }
            let fv = match eval::eval(&ln, lr, &mut env) {
                Ok(v) => v,
                Err(_) => { valid = false; break; }
            };
            match eval::apply(&fv, &[inp.clone()], &ln, &mut env) {
                Ok(v) => outputs.push(v),
                Err(_) => { valid = false; break; }
            }
        }

        if valid && outputs.len() == inputs.len() {
            branches.push(BranchData { pool_idx: vi, outputs });
        }
    }

    if branches.is_empty() {
        return results;
    }

    // Step 3: For each partition, use expected outputs to find matching branches.
    // Hash expected outputs at true/false indices to quickly match.
    for (pattern, &cond_idx) in &partitions {
        let true_indices: Vec<usize> = pattern.iter().enumerate()
            .filter(|(_, b)| **b).map(|(i, _)| i).collect();
        let false_indices: Vec<usize> = pattern.iter().enumerate()
            .filter(|(_, b)| !**b).map(|(i, _)| i).collect();

        // Find branches that match expected on the true side
        let mut then_matches: Vec<usize> = Vec::new();
        // Find branches that match expected on the false side
        let mut else_matches: Vec<usize> = Vec::new();

        for (bi, bd) in branches.iter().enumerate() {
            // Check true-side match
            let then_ok = true_indices.iter().all(|&i| vals_equal(&bd.outputs[i], &expected[i]));
            if then_ok {
                then_matches.push(bi);
            }
            // Check false-side match
            let else_ok = false_indices.iter().all(|&i| vals_equal(&bd.outputs[i], &expected[i]));
            if else_ok {
                else_matches.push(bi);
            }
        }

        // Step 4: Combine matching branches (typically very few)
        for &ti in &then_matches {
            for &ei in &else_matches {
                let then_pool_idx = branches[ti].pool_idx;
                let else_pool_idx = branches[ei].pool_idx;
                // Skip if then and else are the same program
                if then_pool_idx == else_pool_idx { continue; }

                let cond_p = &pool[cond_idx];
                let then_p = &pool[then_pool_idx];
                let else_p = &pool[else_pool_idx];

                // Build merged node tree: cond_nodes + then_nodes + else_nodes + If node
                let mut nodes = cond_p.nodes.clone();
                let cond_root = cond_p.root;

                let then_off = nodes.len();
                for nd in &then_p.nodes {
                    nodes.push(remap_node(nd, then_off));
                }
                let then_root = then_p.root + then_off;

                let else_off = nodes.len();
                for nd in &else_p.nodes {
                    nodes.push(remap_node(nd, else_off));
                }
                let else_root = else_p.root + else_off;

                let if_idx = nodes.len();
                nodes.push(Node::If(cond_root, then_root, else_root));

                // Determine result type: use then-branch type
                let ret_type = then_p.ret_type;

                results.push(SynthPool {
                    nodes,
                    root: if_idx,
                    ret_type,
                    priority: 0.0,
                });
            }
        }
    }

    results
}

/// Check if a node tree references a variable (Symbol "x").
/// Conditions that don't reference the input are useless (e.g., (> 1 2)).
fn pool_has_variable(nodes: &[Node]) -> bool {
    for node in nodes {
        if let Node::Symbol(name) = node {
            if !is_builtin_name(name) {
                return true;
            }
        }
    }
    false
}

/// Builtin/component names that don't count as "variable references".
fn is_builtin_name(name: &str) -> bool {
    matches!(name,
        "add" | "subtract" | "multiply" | "divide" | "modulo" | "abs" | "negate"
        | "min" | "max" | "floor" | "ceil" | "*" | "/"
        | "<" | ">" | "<=" | ">=" | "=" | "!="
        | "not" | "even" | "odd" | "and" | "or"
        | "string-upper" | "string-lower" | "string-reverse" | "string-trim"
        | "string-length" | "string-contains" | "string-split" | "string-join"
        | "string-starts-with" | "string-ends-with" | "string-replace"
        | "substring" | "char-at" | "concat" | "to-string" | "to-number"
        | "head" | "tail" | "length" | "nth" | "cons" | "append" | "reverse"
        | "sort" | "range" | "empty?" | "contains" | "list"
        | "map" | "filter" | "reduce" | "compose" | "pipe" | "apply" | "identity"
        | "if" | "let" | "lambda" | "true" | "false"
    )
}

// ── Macro type inference ────────────────────────────────────────────

/// Probe a macro with sample inputs to infer its parameter and return types.
/// Returns (param_type_tags, return_type_tag).
fn infer_macro_types(
    name: &str,
    params: &[String],
    mnodes: &[Node],
    mroot: usize,
) -> (Vec<u8>, u8) {
    // Default: all-numeric
    let default_params = vec![TYPE_NUM; params.len()];
    let default_ret = TYPE_NUM;

    if params.is_empty() {
        // Zero-arity macro: evaluate it directly to find return type
        let mut env = eval::make_default_env();
        if let Ok(val) = eval::eval(mnodes, mroot, &mut env) {
            return (vec![], value_type_tag(&val));
        }
        return (vec![], default_ret);
    }

    // Probe with different input types to figure out what the macro accepts
    let test_str = Value::Str("5 1 2 3 4".into());
    let test_num = Value::Num(3.0);

    // For each param slot, try string and numeric inputs
    let mut param_types = Vec::with_capacity(params.len());
    let mut ret_type = default_ret;

    // Try all-string args first (common for seq macros like idx, v0, etc.)
    let str_args: Vec<Value> = vec![test_str.clone(); params.len()];
    let num_args: Vec<Value> = vec![test_num.clone(); params.len()];

    let try_call = |args: &[Value]| -> Option<Value> {
        let mut env = eval::make_default_env();
        let val = Value::RustMacro(
            params.iter().cloned().collect(),
            mnodes.to_vec(),
            mroot,
        );
        eval::apply(&val, args, mnodes, &mut env).ok()
    };

    // Try string args
    if let Some(result) = try_call(&str_args) {
        param_types = vec![TYPE_STR; params.len()];
        ret_type = value_type_tag(&result);
    }
    // Try numeric args
    else if let Some(result) = try_call(&num_args) {
        param_types = vec![TYPE_NUM; params.len()];
        ret_type = value_type_tag(&result);
    }
    // Fallback
    else {
        param_types = default_params;
        ret_type = default_ret;
    }

    // For single-param macros, also try the other type to see if it's generic
    if params.len() == 1 {
        let alt_args = if param_types[0] == TYPE_STR {
            vec![test_num.clone()]
        } else {
            vec![test_str.clone()]
        };
        if try_call(&alt_args).is_some() {
            // Accepts both types — mark as TYPE_ANY
            param_types[0] = TYPE_ANY;
        }
    }

    let _ = name; // suppress unused warning
    (param_types, ret_type)
}

// ── Default component library ───────────────────────────────────────

/// Build the default synthesis component library, including any macros.
pub fn default_synth_components(
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> Vec<SynthComponent> {
    let mut comps = vec![
        SynthComponent { name: "x".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 100.0 },
        SynthComponent { name: "0".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "1".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "2".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "3".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "4".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "5".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "6".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "7".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "10".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "-1".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
    ];

    // Unary num->num
    for name in &["abs", "negate"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: 0, param_types: vec![0], priority: 0.0,
        });
    }

    // Binary num->num->num
    for name in &["add", "subtract", "multiply", "min", "max", "modulo"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: 0, param_types: vec![0, 0], priority: 0.0,
        });
    }

    // String ops: unary str->str
    for name in &["string-upper", "string-lower", "string-reverse", "string-trim"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: 1, param_types: vec![1], priority: 0.0,
        });
    }

    // string-length: str->num
    comps.push(SynthComponent {
        name: "string-length".into(), builtin: Some("string-length".into()),
        arity: 1, ret_type: 0, param_types: vec![1], priority: 0.0,
    });

    // Comparison operators: num->num->bool (for if-expression conditions)
    for name in &["<", ">", "<=", ">=", "=", "!="] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: 2, param_types: vec![0, 0], priority: 0.0,
        });
    }

    // Unary num->bool predicates
    for name in &["even", "odd"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: 2, param_types: vec![0], priority: 0.0,
        });
    }

    // Add macro components — infer types by probing with sample inputs
    for (mname, params, mnodes, mroot) in macros {
        let (inferred_param, inferred_ret) = infer_macro_types(mname, params, mnodes, *mroot);
        comps.push(SynthComponent {
            name: mname.clone(),
            builtin: Some(mname.clone()),
            arity: params.len(),
            ret_type: inferred_ret,
            param_types: inferred_param,
            priority: 30.0,
        });
    }

    comps
}

/// Build synthesis components including tree-extracted components.
///
/// Returns `(components, extra_bindings)` where:
/// - `components` includes both the default library and tree-extracted entries
/// - `extra_bindings` are the name-value pairs that must be injected into
///   the eval environment so tree-extracted closures can be called
pub fn default_synth_components_with_trees(
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    trees: &[(String, Value)],
) -> (Vec<SynthComponent>, Vec<(String, Value)>) {
    let mut comps = default_synth_components(macros);
    let mut extra_bindings: Vec<(String, Value)> = Vec::new();
    let mut env = eval::make_default_env();

    for (name, ns_val) in trees {
        let tree_comps = crate::multitree::extract_components_from_namespace(
            ns_val, name, "", &mut env, 5,
        );
        comps.extend(tree_comps);
    }

    // Collect all bindings that were added to env beyond the default scope.
    // The default env has one scope; extract_components_from_namespace adds
    // entries to it via env_define.
    if let Some(scope) = env.last() {
        let default_env = eval::make_default_env();
        let default_scope = default_env.last().unwrap();
        for (k, v) in scope {
            if !default_scope.contains_key(k) {
                extra_bindings.push((k.clone(), v.clone()));
            }
        }
    }

    (comps, extra_bindings)
}

// ── Optimization direction ───────────────────────────────────────────

/// Whether to minimize or maximize the fitness score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptDirection {
    Minimize,
    Maximize,
}

/// Result of an optimization synthesis search.
pub struct OptSynthResult {
    pub found: bool,
    pub nodes: Option<Vec<Node>>,
    pub root: Option<usize>,
    pub candidates_explored: usize,
    pub fitness_score: Option<f64>,
}

impl OptSynthResult {
    fn empty() -> Self {
        OptSynthResult {
            found: false,
            nodes: None,
            root: None,
            candidates_explored: 0,
            fitness_score: None,
        }
    }
}

// ── Optimization synthesis ──────────────────────────────────────────

/// Synthesize a program that minimizes or maximizes a fitness function.
///
/// Instead of stopping at the first correct program, enumerates ALL
/// candidates up to the budget, evaluates each with the fitness function,
/// and returns the one with the best (lowest for minimize, highest for
/// maximize) score.
///
/// The fitness function is represented as a node tree (fitness_nodes,
/// fitness_root) that, when evaluated, should be a callable taking one
/// argument (the candidate program or its output) and returning a number.
///
/// For each candidate:
///   1. Wrap candidate as `(lambda (x) body)`.
///   2. Call `(fitness_fn candidate_lambda)` to get a numeric score.
///   3. Track the best score seen.
///
/// Optionally, `base_inputs` / `base_expected` can provide example
/// constraints that candidates must satisfy before being scored.
pub fn synthesize_optimize(
    components: &[SynthComponent],
    direction: OptDirection,
    fitness_nodes: &[Node],
    fitness_root: usize,
    base_inputs: &[Value],
    base_expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    extra_bindings: &[(String, Value)],
) -> OptSynthResult {
    const MAX_POOL: usize = 100_000;
    let mut explored: usize = 0;
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let macro_env = build_macro_env(macros);

    let mut best_score: Option<f64> = None;
    let mut best_nodes: Option<Vec<Node>> = None;
    let mut best_root: Option<usize> = None;

    let is_better = |new: f64, old: f64| -> bool {
        match direction {
            OptDirection::Minimize => new < old,
            OptDirection::Maximize => new > old,
        }
    };

    // Use namespace-based scoping like the normal synthesizer
    let target_output_type = if base_expected.is_empty() {
        None
    } else {
        infer_uniform_type(base_expected)
    };
    let input_type = if base_inputs.is_empty() {
        None
    } else {
        infer_uniform_type(base_inputs)
    };
    let scoped_components: Vec<&SynthComponent> = {
        let mut scoped: Vec<&SynthComponent> = Vec::new();
        for comp in components.iter() {
            if let Some(target) = target_output_type {
                if target == TYPE_NUM
                    && comp.ret_type == TYPE_STR
                    && comp.arity > 0
                    && !comp.param_types.is_empty()
                    && comp.param_types[0] == TYPE_STR
                {
                    continue;
                }
                if target == TYPE_STR
                    && comp.ret_type == TYPE_NUM
                    && comp.arity > 0
                    && !comp.param_types.is_empty()
                    && comp.param_types[0] == TYPE_NUM
                {
                    continue;
                }
            }
            scoped.push(comp);
        }
        let conditions = scope_for_condition(components);
        for c in conditions {
            if !scoped.iter().any(|s| std::ptr::eq(*s, c)) {
                scoped.push(c);
            }
        }
        scoped
    };
    let components = &scoped_components;

    // Evaluate fitness for a candidate. Returns Some(score) or None on error.
    let eval_fitness = |entry: &SynthPool, macro_env: &[(String, Vec<String>, Vec<Node>, usize)]| -> Option<f64> {
        // Wrap body in (lambda (x) body)
        let mut ln = entry.nodes.clone();
        let lr = ln.len();
        ln.push(Node::Lambda(vec!["x".into()], entry.root));

        // If base examples are provided, check them first
        if !base_inputs.is_empty() {
            for (inp, exp) in base_inputs.iter().zip(base_expected.iter()) {
                let mut env = eval::make_default_env();
                for (nm, ps, mn, mr) in macro_env {
                    env_define(&mut env, nm.clone(), Value::RustMacro(ps.clone(), mn.clone(), *mr));
                }
                for (name, val) in extra_bindings {
                    env_define(&mut env, name.clone(), val.clone());
                }
                let fv = match eval::eval(&ln, lr, &mut env) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                match eval::apply(&fv, &[inp.clone()], &ln, &mut env) {
                    Ok(a) => {
                        if !vals_equal(&a, exp) { return None; }
                    }
                    Err(_) => return None,
                }
            }
        }

        // Evaluate the fitness function on the candidate lambda
        let mut env = eval::make_default_env();
        for (nm, ps, mn, mr) in macro_env {
            env_define(&mut env, nm.clone(), Value::RustMacro(ps.clone(), mn.clone(), *mr));
        }
        for (name, val) in extra_bindings {
            env_define(&mut env, name.clone(), val.clone());
        }

        // Evaluate the fitness function node
        let ffit = match eval::eval(fitness_nodes, fitness_root, &mut env) {
            Ok(v) => v,
            Err(_) => return None,
        };

        // Evaluate the candidate lambda
        let candidate = match eval::eval(&ln, lr, &mut env) {
            Ok(v) => v,
            Err(_) => return None,
        };

        // Apply fitness_fn(candidate)
        match eval::apply(&ffit, &[candidate], &ln, &mut env) {
            Ok(Value::Num(score)) => Some(score),
            _ => None,
        }
    };

    // Test and score a candidate, updating best if improved.
    let mut test_and_score = |entry: &SynthPool,
                               seen: &mut HashSet<Vec<u64>>,
                               explored: &mut usize|
    {
        *explored += 1;
        if *explored > max_candidates {
            return;
        }

        // Behavior hash for dedup
        let mut ln = entry.nodes.clone();
        let lr = ln.len();
        ln.push(Node::Lambda(vec!["x".into()], entry.root));

        let mut beh = Vec::new();
        if !base_inputs.is_empty() {
            for inp in base_inputs {
                let mut env = eval::make_default_env();
                for (nm, ps, mn, mr) in &macro_env {
                    env_define(&mut env, nm.clone(), Value::RustMacro(ps.clone(), mn.clone(), *mr));
                }
                for (name, val) in extra_bindings {
                    env_define(&mut env, name.clone(), val.clone());
                }
                let fv = match eval::eval(&ln, lr, &mut env) {
                    Ok(v) => v,
                    Err(_) => return,
                };
                match eval::apply(&fv, &[inp.clone()], &ln, &mut env) {
                    Ok(a) => beh.push(val_hash(&a)),
                    Err(_) => return,
                }
            }
        } else {
            // No base inputs: use direct evaluation for dedup
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in &macro_env {
                env_define(&mut env, nm.clone(), Value::RustMacro(ps.clone(), mn.clone(), *mr));
            }
            for (name, val) in extra_bindings {
                env_define(&mut env, name.clone(), val.clone());
            }
            match eval::eval(&entry.nodes, entry.root, &mut env) {
                Ok(v) => beh.push(val_hash(&v)),
                Err(_) => return,
            }
        }

        if !beh.is_empty() {
            if seen.contains(&beh) { return; }
            seen.insert(beh);
        }

        // Score with fitness function
        if let Some(score) = eval_fitness(entry, &macro_env) {
            let dominated = match best_score {
                Some(bs) => !is_better(score, bs),
                None => false,
            };
            if !dominated {
                best_score = Some(score);
                // Store the lambda-wrapped version
                let mut ln = entry.nodes.clone();
                let lr = ln.len();
                ln.push(Node::Lambda(vec!["x".into()], entry.root));
                best_nodes = Some(ln);
                best_root = Some(lr);
            }
        }
    };

    // Depth 0: atoms
    let mut pool: Vec<SynthPool> = Vec::new();
    for comp in components {
        if comp.arity != 0 { continue; }
        let mut nodes = Vec::new();
        if comp.name == "x" {
            nodes.push(Node::Symbol("x".into()));
        } else if let Ok(n) = comp.name.parse::<f64>() {
            nodes.push(Node::Num(n));
        } else if comp.name == "true" {
            nodes.push(Node::Bool(true));
        } else if comp.name == "false" {
            nodes.push(Node::Bool(false));
        } else {
            nodes.push(Node::Str(comp.name.clone()));
        }
        pool.push(SynthPool {
            nodes,
            root: 0,
            ret_type: comp.ret_type,
            priority: comp.priority,
        });
    }

    for e in &pool {
        if explored >= max_candidates { break; }
        test_and_score(e, &mut seen, &mut explored);
    }

    // Depth 1..max_depth: compose
    let mut prev_start: usize = 0;
    let mut prev_end: usize = pool.len();

    for _depth in 1..=max_depth {
        if explored >= max_candidates { break; }
        let mut new_entries: Vec<SynthPool> = Vec::new();
        let prev = prev_start..prev_end;
        let all_end = prev_end;

        for comp in components {
            if comp.arity == 0 || comp.builtin.is_none() { continue; }
            let bn = comp.builtin.as_ref().unwrap();

            if comp.arity == 1 {
                for pi in prev.clone() {
                    let p = &pool[pi];
                    if p.ret_type != comp.param_types[0] && comp.param_types[0] != 255 {
                        continue;
                    }
                    let mut n = p.nodes.clone();
                    let fi = n.len();
                    n.push(Node::Symbol(bn.clone()));
                    let ai = n.len();
                    n.push(Node::App(vec![fi, p.root]));
                    let e = SynthPool {
                        nodes: n, root: ai,
                        ret_type: comp.ret_type, priority: comp.priority,
                    };
                    test_and_score(&e, &mut seen, &mut explored);
                    if explored > max_candidates { break; }
                    if pool.len() + new_entries.len() < MAX_POOL {
                        new_entries.push(e);
                    }
                }
            } else if comp.arity == 2 {
                // Case 1: arg1 from new (prev), arg2 from all
                for pi in prev.clone() {
                    let p1 = &pool[pi];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 {
                        continue;
                    }
                    for ai in 0..all_end {
                        let p2 = &pool[ai];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 {
                            continue;
                        }
                        let mut n = p1.nodes.clone();
                        let off = n.len();
                        for nd in &p2.nodes {
                            n.push(remap_node(nd, off));
                        }
                        let fi = n.len();
                        n.push(Node::Symbol(bn.clone()));
                        let api = n.len();
                        n.push(Node::App(vec![fi, p1.root, p2.root + off]));
                        let e = SynthPool {
                            nodes: n, root: api,
                            ret_type: comp.ret_type, priority: comp.priority,
                        };
                        test_and_score(&e, &mut seen, &mut explored);
                        if explored > max_candidates { break; }
                        if pool.len() + new_entries.len() < MAX_POOL {
                            new_entries.push(e);
                        }
                    }
                    if explored > max_candidates { break; }
                }
                // Case 2: arg1 from old, arg2 from new (prev)
                if explored <= max_candidates {
                    for ai in 0..prev_start {
                        let p1 = &pool[ai];
                        if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 {
                            continue;
                        }
                        for pi in prev.clone() {
                            let p2 = &pool[pi];
                            if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 {
                                continue;
                            }
                            let mut n = p1.nodes.clone();
                            let off = n.len();
                            for nd in &p2.nodes {
                                n.push(remap_node(nd, off));
                            }
                            let fi = n.len();
                            n.push(Node::Symbol(bn.clone()));
                            let api = n.len();
                            n.push(Node::App(vec![fi, p1.root, p2.root + off]));
                            let e = SynthPool {
                                nodes: n, root: api,
                                ret_type: comp.ret_type, priority: comp.priority,
                            };
                            test_and_score(&e, &mut seen, &mut explored);
                            if explored > max_candidates { break; }
                            if pool.len() + new_entries.len() < MAX_POOL {
                                new_entries.push(e);
                            }
                        }
                        if explored > max_candidates { break; }
                    }
                }
            }
            if explored > max_candidates { break; }
        }

        prev_start = pool.len();
        pool.extend(new_entries);
        prev_end = pool.len();
    }

    if best_nodes.is_some() {
        OptSynthResult {
            found: true,
            nodes: best_nodes,
            root: best_root,
            candidates_explored: explored,
            fitness_score: best_score,
        }
    } else {
        OptSynthResult {
            found: false,
            nodes: None,
            root: None,
            candidates_explored: explored,
            fitness_score: None,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_components() -> Vec<SynthComponent> {
        default_synth_components(&[])
    }

    #[test]
    fn test_component_domain_classification() {
        let comps = make_test_components();
        let abs_comp = comps.iter().find(|c| c.name == "abs").unwrap();
        assert_eq!(component_domain(abs_comp), Domain::Number);

        let upper_comp = comps.iter().find(|c| c.name == "string-upper").unwrap();
        assert_eq!(component_domain(upper_comp), Domain::String);

        let even_comp = comps.iter().find(|c| c.name == "even").unwrap();
        assert_eq!(component_domain(even_comp), Domain::Number);
    }

    #[test]
    fn test_component_kind_classification() {
        let comps = make_test_components();
        let abs_comp = comps.iter().find(|c| c.name == "abs").unwrap();
        assert_eq!(component_kind(abs_comp), Kind::Arithmetic);

        let upper = comps.iter().find(|c| c.name == "string-upper").unwrap();
        assert_eq!(component_kind(upper), Kind::Transform);

        let lt = comps.iter().find(|c| c.name == "<").unwrap();
        assert_eq!(component_kind(lt), Kind::Compare);

        let even = comps.iter().find(|c| c.name == "even").unwrap();
        assert_eq!(component_kind(even), Kind::Predicate);

        let zero = comps.iter().find(|c| c.name == "0").unwrap();
        assert_eq!(component_kind(zero), Kind::Constants);

        let strlen = comps.iter().find(|c| c.name == "string-length").unwrap();
        assert_eq!(component_kind(strlen), Kind::Analyze);
    }

    #[test]
    fn test_scope_for_context_num_output() {
        let comps = make_test_components();
        let scoped = scope_for_context(&comps, Some(TYPE_NUM), None);
        // Should include numeric ops (add, abs, etc.)
        assert!(scoped.iter().any(|c| c.name == "add"));
        assert!(scoped.iter().any(|c| c.name == "abs"));
        // Should NOT include string->string ops
        assert!(!scoped.iter().any(|c| c.name == "string-upper"));
        // Should include string-length (returns num)
        assert!(scoped.iter().any(|c| c.name == "string-length"));
    }

    #[test]
    fn test_scope_for_context_str_output() {
        let comps = make_test_components();
        let scoped = scope_for_context(&comps, Some(TYPE_STR), None);
        // Should include string transforms
        assert!(scoped.iter().any(|c| c.name == "string-upper"));
        // Should NOT include numeric arithmetic
        assert!(!scoped.iter().any(|c| c.name == "add"));
        assert!(!scoped.iter().any(|c| c.name == "abs"));
    }

    #[test]
    fn test_scope_for_context_with_input_type() {
        let comps = make_test_components();
        // Numeric input, numeric output
        let scoped = scope_for_context(&comps, Some(TYPE_NUM), Some(TYPE_NUM));
        assert!(scoped.iter().any(|c| c.name == "add"));
        assert!(scoped.iter().any(|c| c.name == "abs"));
        // string-upper takes str input, should be excluded
        assert!(!scoped.iter().any(|c| c.name == "string-upper"));
    }

    #[test]
    fn test_scope_for_context_filters_reduce_count() {
        let comps = make_test_components();
        let all_count = comps.len();
        let num_scoped = scope_for_context(&comps, Some(TYPE_NUM), Some(TYPE_NUM));
        // Scoped list should be smaller than full list
        assert!(num_scoped.len() < all_count,
            "scoped ({}) should be less than all ({})", num_scoped.len(), all_count);
    }

    #[test]
    fn test_scope_for_condition() {
        let comps = make_test_components();
        let conds = scope_for_condition(&comps);
        // All returned components should have bool return type
        for c in &conds {
            assert_eq!(c.ret_type, TYPE_BOOL, "condition component '{}' should return bool", c.name);
        }
        // Should include comparisons and predicates
        assert!(conds.iter().any(|c| c.name == "<"));
        assert!(conds.iter().any(|c| c.name == "even"));
    }

    #[test]
    fn test_scope_for_branch() {
        let comps = make_test_components();
        let branches = scope_for_branch(&comps, Some(TYPE_NUM));
        // Should include arithmetic
        assert!(branches.iter().any(|c| c.name == "add") || branches.iter().any(|c| c.name == "abs"));
        // Should include constants
        assert!(branches.iter().any(|c| c.name == "0"));
    }

    #[test]
    fn test_infer_uniform_type() {
        let nums = vec![Value::Num(1.0), Value::Num(2.0)];
        assert_eq!(infer_uniform_type(&nums), Some(TYPE_NUM));

        let strs = vec![Value::Str("a".into()), Value::Str("b".into())];
        assert_eq!(infer_uniform_type(&strs), Some(TYPE_STR));

        let mixed = vec![Value::Num(1.0), Value::Str("a".into())];
        assert_eq!(infer_uniform_type(&mixed), None);

        let empty: Vec<Value> = vec![];
        assert_eq!(infer_uniform_type(&empty), None);
    }

    #[test]
    fn test_value_type_tag() {
        assert_eq!(value_type_tag(&Value::Num(1.0)), TYPE_NUM);
        assert_eq!(value_type_tag(&Value::Str("a".into())), TYPE_STR);
        assert_eq!(value_type_tag(&Value::Bool(true)), TYPE_BOOL);
        assert_eq!(value_type_tag(&Value::Nil), TYPE_ANY);
    }
}
