//! Bottom-up enumerative program synthesizer for SELPH.
//!
//! Given input/output examples, searches for the smallest program that
//! satisfies them. Uses type pruning, observational equivalence dedup,
//! and (optionally) hash-based if-expression generation.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use crate::types::*;
use crate::eval;
use crate::hm;
use crate::intern::{intern, resolve};

// ── Type tag constants ──────────────────────────────────────────────

pub const TYPE_NUM: u8 = 0;
pub const TYPE_STR: u8 = 1;
pub const TYPE_BOOL: u8 = 2;
pub const TYPE_LIST: u8 = 3;
pub const TYPE_GRID: u8 = 4;
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
        TYPE_LIST => Domain::List,
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
    /// How many prior tasks in this curriculum run used this component.
    pub usage_count: f64,
}

/// A candidate program stored as a flattened node tree.
#[derive(Clone)]
pub struct SynthPool {
    pub nodes: Vec<Node>,
    pub root: usize,
    pub ret_type: u8,
    pub priority: f64,
}

/// RL reward coefficients for partial-match-driven priority adjustment.
/// These control how strongly partial match signals affect pool entry
/// priorities during synthesis. Optimized by the meta loop.
#[derive(Clone, Copy, Debug)]
pub struct RlCoefficients {
    /// Penalty applied to pool entries that match 0 examples when their
    /// return type matches the target. Negative value (default: -50.0).
    pub cold_penalty: f64,
    /// Scale factor for partial match bonus. A candidate matching fraction
    /// f of examples gets priority += f * warm_bonus. Default: 30.0.
    pub warm_bonus: f64,
    /// Inter-depth component priority boost. Components whose candidates
    /// achieved partial matches get boosted for the next depth. Default: 15.0.
    pub comp_warm_bonus: f64,
}

impl Default for RlCoefficients {
    fn default() -> Self {
        RlCoefficients { cold_penalty: -50.0, warm_bonus: 30.0, comp_warm_bonus: 15.0 }
    }
}

/// Result of a synthesis search.
pub struct SynthResult {
    pub found: bool,
    pub nodes: Option<Vec<Node>>,
    pub root: Option<usize>,
    pub candidates_explored: usize,
}

/// Lightweight record of a candidate for heuristic re-ranking.
///
/// Captures only the component name and argument priority sum — enough
/// to re-score under a different heuristic without re-enumerating.
/// The heuristic controls component priority; arg_priority_sum is the
/// fixed contribution from arguments (heuristic-independent).
#[derive(Clone, Debug)]
pub struct CandidateRecord {
    pub comp_name: String,
    pub arg_priority_sum: f64,
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
        Value::List(_) => TYPE_LIST,
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
/// When `b` is `Value::Alt`, returns true if `a` matches any alternative.
pub fn vals_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        (Value::List(xs), Value::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(a, b)| vals_equal(a, b))
        }
        // Alt on the expected side: actual must match any alternative
        (_, Value::Alt(alts)) => alts.iter().any(|alt| vals_equal(a, alt)),
        _ => false,
    }
}

// ── Parallel synthesis types and helpers ─────────────────────────────

/// Lightweight descriptor for a candidate to be tested.
/// Stores indices rather than cloned node trees for memory efficiency.
pub struct PendingDesc {
    pub comp_idx: usize,    // index into components slice
    pub arg1: usize,        // pool index of first argument
    pub arg2: usize,        // pool index of second argument (unused for arity 1)
    pub arg3: usize,        // pool index of third argument (unused for arity 1-2)
    pub ret_type: u8,       // inferred return type
    pub score: f64,
}

/// Result of evaluating a single candidate in the parallel phase.
pub enum EvalResult {
    /// Skipped due to type gate — no evaluation performed.
    Skipped,
    /// VM compilation failed — needs sequential tree-walker fallback.
    NeedsTreeWalker,
    /// Successfully evaluated via VM fast path.
    Evaluated {
        beh: Vec<u64>,
        matches: usize,
        evaluated: usize,
    },
}

/// Materialize a PendingDesc into a full SynthPool entry by building
/// the node tree from pool entries and the component.
pub fn materialize(
    desc: &PendingDesc,
    pool: &[SynthPool],
    components: &[&SynthComponent],
) -> SynthPool {
    let comp = &components[desc.comp_idx];
    let bn = comp.builtin.as_ref().unwrap();

    if comp.arity == 1 && bn.starts_with("map_") {
        let macro_name = &bn[4..];
        let p = &pool[desc.arg1];
        let mut n = p.nodes.clone();
        let map_sym = n.len();
        n.push(Node::Symbol(intern("map")));
        let fn_sym = n.len();
        n.push(Node::Symbol(intern(macro_name)));
        let api = n.len();
        n.push(Node::App(vec![map_sym, fn_sym, p.root]));
        SynthPool { nodes: n, root: api, ret_type: desc.ret_type, priority: desc.score }
    } else if comp.arity == 1 && bn.starts_with("reduce_") {
        let fn_name = &bn[7..];
        let p = &pool[desc.arg1];
        let mut n = p.nodes.clone();
        let reduce_sym = n.len();
        n.push(Node::Symbol(intern("reduce")));
        let fn_sym = n.len();
        n.push(Node::Symbol(intern(fn_name)));
        let api = n.len();
        n.push(Node::App(vec![reduce_sym, fn_sym, p.root]));
        SynthPool { nodes: n, root: api, ret_type: desc.ret_type, priority: desc.score }
    } else if comp.arity == 1 {
        let p = &pool[desc.arg1];
        let mut n = p.nodes.clone();
        let fi = n.len();
        n.push(Node::Symbol(intern(bn)));
        let ai = n.len();
        n.push(Node::App(vec![fi, p.root]));
        SynthPool { nodes: n, root: ai, ret_type: desc.ret_type, priority: desc.score }
    } else if comp.arity == 2 {
        let p1 = &pool[desc.arg1];
        let p2 = &pool[desc.arg2];
        let mut n = p1.nodes.clone();
        let off = n.len();
        for nd in &p2.nodes {
            n.push(remap_node(nd, off));
        }
        let fi = n.len();
        n.push(Node::Symbol(intern(bn)));
        let api = n.len();
        n.push(Node::App(vec![fi, p1.root, p2.root + off]));
        SynthPool { nodes: n, root: api, ret_type: desc.ret_type, priority: desc.score }
    } else {
        // Arity 3
        let p1 = &pool[desc.arg1];
        let p2 = &pool[desc.arg2];
        let p3 = &pool[desc.arg3];
        let mut n = p1.nodes.clone();
        let off2 = n.len();
        for nd in &p2.nodes {
            n.push(remap_node(nd, off2));
        }
        let off3 = n.len();
        for nd in &p3.nodes {
            n.push(remap_node(nd, off3));
        }
        let fi = n.len();
        n.push(Node::Symbol(intern(bn)));
        let api = n.len();
        n.push(Node::App(vec![fi, p1.root, p2.root + off2, p3.root + off3]));
        SynthPool { nodes: n, root: api, ret_type: desc.ret_type, priority: desc.score }
    }
}

/// Pure candidate evaluation via VM fast path.
/// No side effects — does not touch `seen`, `explored`, or any shared state.
pub fn eval_candidate(
    entry: &SynthPool,
    inputs: &[Value],
    expected: &[Value],
    vm_ctx: &crate::vm::CompileCtx,
    vm_macro_chunks: &[Option<crate::vm::MacroChunk>],
    target_output_type: Option<u8>,
) -> EvalResult {
    // Type gate
    if let Some(target) = target_output_type {
        if entry.ret_type != target && entry.ret_type != TYPE_ANY && target != TYPE_ANY {
            return EvalResult::Skipped;
        }
    }

    // Try VM compilation
    let chunk = match crate::vm::compile(&entry.nodes, entry.root, vm_ctx) {
        Ok(c) => c,
        Err(_) => return EvalResult::NeedsTreeWalker,
    };

    let mut vm_stack = Vec::with_capacity(32);
    let mut beh = Vec::with_capacity(inputs.len());
    let mut matches = 0usize;
    let mut evaluated = 0usize;

    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        match crate::vm::execute(&chunk, inp, vm_macro_chunks, &mut vm_stack) {
            Ok(a) => {
                beh.push(val_hash(&a));
                evaluated += 1;
                if vals_equal(&a, exp) { matches += 1; }
            }
            Err(_) => break,
        }
    }

    EvalResult::Evaluated { beh, matches, evaluated }
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
        let inferred = hm::type_to_tag(&ret_type, &subst);
        // If HM gives TYPE_ANY (unresolved var) but the component has a
        // concrete ret_type (e.g., head patched to NUM for list-of-numbers),
        // prefer the concrete type.
        if inferred == TYPE_ANY && comp.ret_type != TYPE_ANY {
            comp.ret_type
        } else {
            inferred
        }
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

        let empty_nodes: Rc<[Node]> = Vec::<Node>::new().into();
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

        let empty_nodes: Rc<[Node]> = Vec::<Node>::new().into();
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
        max_candidates, enable_if, validation_examples, extra_bindings, None, None,
        RlCoefficients::default())
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
    snapshot: Option<&mut Vec<CandidateRecord>>,
    rl_coeffs: RlCoefficients,
) -> SynthResult {
    const MAX_POOL: usize = 100_000;
    let mut explored: usize = 0;
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let macro_env = build_macro_env(macros);
    let mut snapshot = snapshot;

    // ── VM setup for fast candidate evaluation ──────────────────────
    let mut vm_ctx = crate::vm::CompileCtx::new(crate::eval::BUILTIN_NAMES, macros);
    let vm_macro_chunks = crate::vm::compile_macros(macros, &vm_ctx);
    let mut vm_stack: Vec<Value> = Vec::with_capacity(32);

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

    // Compute which types are "useful" — can eventually produce the target
    // output type through some chain of available components. A type t is
    // useful if: (a) t == target, or (b) some component takes t as input
    // and returns a useful type. This is a fixed-point computation.
    // Pool entries whose type is not useful are dead weight.
    let useful_types: HashSet<u8> = if let Some(target) = target_output_type {
        let mut useful = HashSet::new();
        useful.insert(target);
        useful.insert(TYPE_ANY);
        // Fixed-point: keep adding types that feed into useful types
        loop {
            let mut changed = false;
            for comp in components.iter() {
                if comp.arity == 0 { continue; }
                // If this component returns a useful type, all its param types are useful
                if useful.contains(&comp.ret_type) || comp.ret_type == TYPE_ANY {
                    for &pt in &comp.param_types {
                        if useful.insert(pt) { changed = true; }
                    }
                }
            }
            if !changed { break; }
        }
        useful
    } else {
        // No target type — all types are useful
        let mut all = HashSet::new();
        all.insert(TYPE_NUM); all.insert(TYPE_STR); all.insert(TYPE_BOOL); all.insert(TYPE_LIST); all.insert(TYPE_ANY);
        all
    };

    // Fix up `x` variable type: its ret_type should match the actual
    // input type, not always be TYPE_NUM.
    //
    // ── Domain-based component filtering ──────────────────────────────
    // Determine which type domains are "reachable" from the input/output
    // types.  A component is kept only if ALL its param types and its
    // return type belong to the reachable set.  This eliminates entire
    // irrelevant domains (e.g. all grid ops for string tasks) early,
    // massively reducing the combinatorial search space.
    let reachable_types: HashSet<u8> = {
        let mut reach = HashSet::new();
        reach.insert(TYPE_ANY); // always reachable
        // Seed with input and output types
        if let Some(it) = input_type { reach.insert(it); }
        if let Some(ot) = target_output_type { reach.insert(ot); }
        // Always include BOOL (needed for if-conditions) and NUM (common intermediate)
        reach.insert(TYPE_BOOL);
        reach.insert(TYPE_NUM);
        // If either input or output is LIST or STR, include both
        // (string-split produces LIST from STR, string-join produces STR from LIST)
        if reach.contains(&TYPE_STR) || reach.contains(&TYPE_LIST) {
            reach.insert(TYPE_STR);
            reach.insert(TYPE_LIST);
        }
        reach
    };

    let scoped_components: Vec<&SynthComponent> = {
        let mut scoped: Vec<&SynthComponent> = Vec::new();
        for comp in components.iter() {
            if comp.arity == 0 {
                // Atoms: keep if their type is reachable
                if reachable_types.contains(&comp.ret_type) {
                    scoped.push(comp);
                }
                continue;
            }
            // Skip components that operate in unreachable domains
            let all_types_reachable = comp.param_types.iter().all(|&pt| reachable_types.contains(&pt))
                && reachable_types.contains(&comp.ret_type);
            if !all_types_reachable { continue; }

            // Exclude pure string→string ops when target is numeric
            if let Some(target) = target_output_type {
                if target == TYPE_NUM
                    && comp.ret_type == TYPE_STR
                    && !comp.param_types.is_empty()
                    && comp.param_types[0] == TYPE_STR
                {
                    continue;
                }
                // Exclude pure num→num ops when target is string
                if target == TYPE_STR
                    && comp.ret_type == TYPE_NUM
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
                    // Only add if all types are reachable
                    let ok = c.param_types.iter().all(|&pt| reachable_types.contains(&pt))
                        && reachable_types.contains(&c.ret_type);
                    if ok { scoped.push(c); }
                }
            }
        }
        scoped
    };

    // ── Probe-and-filter: remove macros that error on actual inputs ──
    // Library macros from previous curricula may expect different input
    // formats (e.g. space-separated vs raw strings). Test each unary
    // macro against the first input — if it errors, exclude it.
    let scoped_components: Vec<&SynthComponent> = if !inputs.is_empty() {
        let test_input = &inputs[0];
        scoped_components.into_iter().filter(|comp| {
            // Only probe macros (arity 1, have a builtin name that matches a macro)
            if comp.arity != 1 { return true; }
            let bn = match &comp.builtin {
                Some(n) => n,
                None => return true,
            };
            // Check if this is a library macro (not a builtin)
            let is_macro = macros.iter().any(|(mn, _, _, _)| mn == bn);
            if !is_macro { return true; }
            // Only probe macros whose param type matches the input type.
            // Macros with different param types (e.g. num→num macro when input
            // is str) are meant for intermediate compositions, not raw input.
            if let Some(inp) = input_type {
                if !comp.param_types.is_empty()
                    && comp.param_types[0] != inp
                    && comp.param_types[0] != TYPE_ANY
                    && inp != TYPE_ANY
                {
                    return true; // keep — don't probe with wrong input type
                }
            }
            // Probe: try calling the macro with the first input
            let macro_data = macros.iter().find(|(mn, _, _, _)| mn == bn);
            if let Some((_, params, mnodes, mroot)) = macro_data {
                let val = Value::RustMacro(params.iter().map(|s| intern(s)).collect(), Rc::from(mnodes.clone()), *mroot);
                let empty: Rc<[Node]> = Vec::<Node>::new().into();
                let mut env = eval::make_default_env();
                for (nm, ps, mn, mr) in &macro_env {
                    env_define(&mut env, intern(nm),
                        Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr));
                }
                eval::apply(&val, &[test_input.clone()], &empty, &mut env).is_ok()
            } else {
                true
            }
        }).collect()
    } else {
        scoped_components
    };

    // Use scoped components for the rest of synthesis.
    let components = &scoped_components;

    // Closure: evaluate a candidate, check against expected outputs,
    // and dedup via behavior hash.
    //
    // Returns (solution_or_none, match_fraction):
    //   - match_fraction in [0.0, 1.0]: fraction of examples matched
    //   - -1.0 if skipped (type gate, budget, dedup) — no eval happened
    //
    // The match_fraction enables RL-style reward propagation: partial
    // matches signal that a component is "on the right track," and
    // zero matches signal dead ends for Bayesian pruning.
    let test = |entry: &SynthPool,
                seen: &mut HashSet<Vec<u64>>,
                explored: &mut usize,
                vm_stack: &mut Vec<Value>|
        -> (Option<(Vec<Node>, usize)>, f64)
    {
        // Type gate: skip evaluation if the candidate's return type
        // cannot match the expected output type. This avoids expensive
        // eval calls for clearly wrong-typed candidates.
        if let Some(target) = target_output_type {
            if entry.ret_type != target && entry.ret_type != TYPE_ANY && target != TYPE_ANY {
                return (None, -1.0);
            }
        }

        *explored += 1;
        if *explored > max_candidates {
            return (None, -1.0);
        }

        let mut beh = Vec::new();
        let mut matches = 0usize;
        let mut evaluated = 0usize;

        // Try VM compilation — if it succeeds, use fast path for all examples
        if let Ok(chunk) = crate::vm::compile(&entry.nodes, entry.root, &vm_ctx) {
            // ── VM fast path: no env creation, no tree walking ──
            for (inp, exp) in inputs.iter().zip(expected.iter()) {
                match crate::vm::execute(&chunk, inp, &vm_macro_chunks, vm_stack) {
                    Ok(a) => {
                        beh.push(val_hash(&a));
                        evaluated += 1;
                        if vals_equal(&a, exp) { matches += 1; }
                    }
                    Err(_) => break,
                }
            }
        } else {
            // ── Tree-walker fallback for unsupported constructs ──
            let mut ln = entry.nodes.clone();
            let lr = ln.len();
            ln.push(Node::Lambda(vec![intern("x")], entry.root));
            let ln_rc: Rc<[Node]> = ln.into();

            for (inp, exp) in inputs.iter().zip(expected.iter()) {
                let mut env = eval::make_default_env();
                for (nm, ps, mn, mr) in &macro_env {
                    env_define(
                        &mut env,
                        intern(nm),
                        Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr),
                    );
                }
                for (name, val) in extra_bindings {
                    env_define(&mut env, intern(name), val.clone());
                }
                let fv = match eval::eval(&ln_rc, lr, &mut env) {
                    Ok(v) => v,
                    Err(_) => break,
                };
                match eval::apply(&fv, &[inp.clone()], &ln_rc, &mut env) {
                    Ok(a) => {
                        beh.push(val_hash(&a));
                        evaluated += 1;
                        if vals_equal(&a, exp) { matches += 1; }
                    }
                    Err(_) => break,
                }
            }
        }

        let match_frac = if evaluated > 0 {
            matches as f64 / inputs.len() as f64
        } else {
            0.0
        };

        // Observational equivalence dedup
        if !beh.is_empty() {
            if seen.contains(&beh) {
                return (None, -1.0);
            }
            seen.insert(beh);
        }

        if matches == inputs.len() && !inputs.is_empty() {
            // Held-out validation: if validation examples provided,
            // check the candidate generalises beyond training data.
            if let Some(val_exs) = validation_examples {
                // Validation always uses tree-walker (correctness over speed)
                let mut ln = entry.nodes.clone();
                let lr = ln.len();
                ln.push(Node::Lambda(vec![intern("x")], entry.root));
                let ln_rc: Rc<[Node]> = ln.into();
                if !validate_candidate(&ln_rc, lr, val_exs, &macro_env, extra_bindings) {
                    return (None, match_frac);
                }
                (Some((ln_rc.to_vec(), lr)), match_frac)
            } else {
                // Build the lambda-wrapped nodes for the result
                let mut ln = entry.nodes.clone();
                let lr = ln.len();
                ln.push(Node::Lambda(vec![intern("x")], entry.root));
                (Some((ln, lr)), match_frac)
            }
        } else {
            (None, match_frac)
        }
    };

    // ── Depth 0: atoms ──────────────────────────────────────────────

    let mut pool: Vec<SynthPool> = Vec::new();
    for comp in components {
        if comp.arity != 0 { continue; }
        let mut nodes = Vec::new();
        if comp.name == "x" {
            nodes.push(Node::Symbol(intern("x")));
        } else if let Ok(n) = comp.name.parse::<f64>() {
            nodes.push(Node::Num(n));
        } else if comp.name == "true" {
            nodes.push(Node::Bool(true));
        } else if comp.name == "false" {
            nodes.push(Node::Bool(false));
        } else if comp.builtin.is_some() && comp.ret_type == TYPE_ANY {
            // Named binding (e.g. data namespace from library tree) — emit as
            // symbol so it resolves from the eval env via extra_bindings.
            nodes.push(Node::Symbol(intern(&comp.name)));
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

    // ── Auto-extract constants from examples ──────────────────────
    // Extract unique characters and numbers from examples, scored by
    // frequency. Constants that appear more often across examples get
    // higher priority — the interleaved search will try them first.
    //
    // This replaces hard-coded string constants like "a", "b", "(", ")"
    // and makes the synthesizer adapt to any domain automatically.
    {
        let mut seen_str_constants: HashSet<String> = HashSet::new();
        let mut seen_num_constants: HashSet<i64> = HashSet::new();
        let mut str_freq: HashMap<String, usize> = HashMap::new();

        // Collect existing constants already in the pool
        for p in &pool {
            if let Node::Str(s) = &p.nodes[p.root] {
                seen_str_constants.insert(s.clone());
            }
            if let Node::Num(n) = &p.nodes[p.root] {
                seen_num_constants.insert(*n as i64);
            }
        }

        // Count character frequency across ALL examples
        let n_examples = inputs.len().max(1);
        for val in inputs.iter().chain(expected.iter()) {
            if let Value::Str(s) = val {
                // Count unique chars per example (not total occurrences)
                let mut seen_in_example: HashSet<char> = HashSet::new();
                for ch in s.chars() {
                    if seen_in_example.insert(ch) {
                        *str_freq.entry(ch.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Add string constants with frequency-based priority
        // Priority = (frequency / n_examples) * 20
        // A char in every example gets priority 20+, rare chars get ~1
        for (cs, freq) in &str_freq {
            if !seen_str_constants.contains(cs) {
                seen_str_constants.insert(cs.clone());
                let priority = (*freq as f64 / n_examples as f64) * 20.0;
                pool.push(SynthPool {
                    nodes: vec![Node::Str(cs.clone())],
                    root: 0,
                    ret_type: TYPE_STR,
                    priority,
                });
            }
        }

        // Extract small numeric constants from outputs and list elements
        for val in inputs.iter().chain(expected.iter()) {
            let nums: Vec<f64> = match val {
                Value::Num(n) => vec![*n],
                Value::List(elems) => elems.iter().filter_map(|v| {
                    if let Value::Num(n) = v { Some(*n) } else { None }
                }).collect(),
                _ => vec![],
            };
            for n in nums {
                let ni = n as i64;
                if ni.abs() <= 100 && !seen_num_constants.contains(&ni) {
                    seen_num_constants.insert(ni);
                    pool.push(SynthPool {
                        nodes: vec![Node::Num(n)],
                        root: 0,
                        ret_type: TYPE_NUM,
                        priority: 0.0,
                    });
                }
            }
        }

        // Add empty string if any string values are present
        let has_strings = inputs.iter().chain(expected.iter())
            .any(|v| matches!(v, Value::Str(_)));
        if has_strings && !seen_str_constants.contains("") {
            pool.push(SynthPool {
                nodes: vec![Node::Str(String::new())],
                root: 0,
                ret_type: TYPE_STR,
                priority: 0.0,
            });
        }
    }

    // Track partial match scores per pool entry for RL reward propagation.
    // Indexed parallel to pool: match_fraction in [0.0, 1.0], or -1.0 if
    // not evaluated (type-gated or deduped).
    let mut pool_scores: Vec<f64> = Vec::new();

    // Test atoms
    for e in &pool {
        if explored >= max_candidates {
            return SynthResult { found: false, nodes: None, root: None, candidates_explored: explored };
        }
        let (result, score) = test(e, &mut seen, &mut explored, &mut vm_stack);
        pool_scores.push(score);
        if let Some((n, r)) = result {
            return SynthResult::success(n, r, explored);
        }
    }

    // ── Depth 1..max_depth: compose ─────────────────────────────────

    // Per-component reward accumulator for RL-style priority updates.
    // Between depths, components whose candidates got partial matches
    // get priority boosts, while components with only zero-match
    // candidates get deprioritized.
    let mut comp_best_match: HashMap<usize, f64> = HashMap::new();
    let mut comp_priority_boost: Vec<f64> = vec![0.0; components.len()];

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
        //
        // Memory optimization: we store lightweight descriptors (component
        // index + pool indices + inferred type) instead of cloned node trees.
        // Node trees are built on-demand during testing, so only one
        // materialized candidate exists at a time.

        let mut pending: Vec<PendingDesc> = Vec::new();

        for (ci, comp) in components.iter().enumerate() {
            if comp.arity == 0 || comp.builtin.is_none() { continue; }
            // SELPH-programmable depth filter (optional hard cutoff)
            if let Some(ref filter) = depth_filter {
                if !filter(comp, _depth) { continue; }
            }

            if comp.arity == 1 {
                for pi in prev.clone() {
                    let p = &pool[pi];
                    if p.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p.ret_type != 255 {
                        continue;
                    }
                    if !hm_check_application(comp, &[p], &mut hm_counter) {
                        continue;
                    }
                    let inferred_ret = hm_infer_ret_type(comp, &[p], &mut hm_counter);
                    let score = (comp.priority + comp_priority_boost[ci]) + p.priority;
                    pending.push(PendingDesc {
                        comp_idx: ci, arg1: pi, arg2: 0, arg3: 0,
                        ret_type: inferred_ret, score,
                    });
                }
            } else if comp.arity == 2 {
                // Case 1: arg1 from new (prev), arg2 from all
                for pi in prev.clone() {
                    let p1 = &pool[pi];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 {
                        continue;
                    }
                    for ai in 0..all_end {
                        let p2 = &pool[ai];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 {
                            continue;
                        }
                        if !hm_check_application(comp, &[p1, p2], &mut hm_counter) {
                            continue;
                        }
                        let inferred_ret = hm_infer_ret_type(comp, &[p1, p2], &mut hm_counter);
                        // Use average of arg priorities instead of sum so arity-2
                        // doesn't automatically outrank arity-1 compositions.
                        let score = (comp.priority + comp_priority_boost[ci]) + (p1.priority + p2.priority) / 2.0;
                        pending.push(PendingDesc {
                            comp_idx: ci, arg1: pi, arg2: ai, arg3: 0,
                            ret_type: inferred_ret, score,
                        });
                    }
                }
                // Case 2: arg1 from old, arg2 from new (prev)
                for ai in 0..prev_start {
                    let p1 = &pool[ai];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 {
                        continue;
                    }
                    for pi in prev.clone() {
                        let p2 = &pool[pi];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 {
                            continue;
                        }
                        if !hm_check_application(comp, &[p1, p2], &mut hm_counter) {
                            continue;
                        }
                        let inferred_ret = hm_infer_ret_type(comp, &[p1, p2], &mut hm_counter);
                        let score = (comp.priority + comp_priority_boost[ci]) + (p1.priority + p2.priority) / 2.0;
                        pending.push(PendingDesc {
                            comp_idx: ci, arg1: ai, arg2: pi, arg3: 0,
                            ret_type: inferred_ret, score,
                        });
                    }
                }
            } else if comp.arity == 3 {
                // Arity-3: at least one arg must be from prev (current depth).
                // Guard: skip arity-3 when the pool is large enough that
                // cubic enumeration would dominate the search budget.
                // At depth 1 (small pool of atoms), arity-3 is cheap.
                // At depth 2+ with many pool entries, skip it.
                let max_per_arg: usize = 15;
                let type_counts: Vec<usize> = comp.param_types.iter().map(|&pt| {
                    (0..all_end).filter(|&i| pool[i].ret_type == pt || pt == 255).count()
                }).collect();
                if type_counts.iter().any(|&c| c > max_per_arg) {
                    continue;
                }
                // Case 1: arg1 from prev, arg2+arg3 from all
                for pi in prev.clone() {
                    let p1 = &pool[pi];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 { continue; }
                    for a2 in 0..all_end {
                        let p2 = &pool[a2];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 { continue; }
                        for a3 in 0..all_end {
                            let p3 = &pool[a3];
                            if p3.ret_type != comp.param_types[2] && comp.param_types[2] != 255 && p3.ret_type != 255 { continue; }
                            if !hm_check_application(comp, &[p1, p2, p3], &mut hm_counter) { continue; }
                            let inferred_ret = hm_infer_ret_type(comp, &[p1, p2, p3], &mut hm_counter);
                            let score = (comp.priority + comp_priority_boost[ci]) + (p1.priority + p2.priority + p3.priority) / 3.0;
                            pending.push(PendingDesc {
                                comp_idx: ci, arg1: pi, arg2: a2, arg3: a3,
                                ret_type: inferred_ret, score,
                            });
                        }
                    }
                }
                // Case 2: arg1 from old, arg2 from prev, arg3 from all
                for a1 in 0..prev_start {
                    let p1 = &pool[a1];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 { continue; }
                    for pi in prev.clone() {
                        let p2 = &pool[pi];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 { continue; }
                        for a3 in 0..all_end {
                            let p3 = &pool[a3];
                            if p3.ret_type != comp.param_types[2] && comp.param_types[2] != 255 && p3.ret_type != 255 { continue; }
                            if !hm_check_application(comp, &[p1, p2, p3], &mut hm_counter) { continue; }
                            let inferred_ret = hm_infer_ret_type(comp, &[p1, p2, p3], &mut hm_counter);
                            let score = (comp.priority + comp_priority_boost[ci]) + (p1.priority + p2.priority + p3.priority) / 3.0;
                            pending.push(PendingDesc {
                                comp_idx: ci, arg1: a1, arg2: pi, arg3: a3,
                                ret_type: inferred_ret, score,
                            });
                        }
                    }
                }
                // Case 3: arg1+arg2 from old, arg3 from prev
                for a1 in 0..prev_start {
                    let p1 = &pool[a1];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 { continue; }
                    for a2 in 0..prev_start {
                        let p2 = &pool[a2];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 { continue; }
                        for pi in prev.clone() {
                            let p3 = &pool[pi];
                            if p3.ret_type != comp.param_types[2] && comp.param_types[2] != 255 && p3.ret_type != 255 { continue; }
                            if !hm_check_application(comp, &[p1, p2, p3], &mut hm_counter) { continue; }
                            let inferred_ret = hm_infer_ret_type(comp, &[p1, p2, p3], &mut hm_counter);
                            let score = (comp.priority + comp_priority_boost[ci]) + (p1.priority + p2.priority + p3.priority) / 3.0;
                            pending.push(PendingDesc {
                                comp_idx: ci, arg1: a1, arg2: a2, arg3: pi,
                                ret_type: inferred_ret, score,
                            });
                        }
                    }
                }
            }
        }

        // Sort descriptors by priority descending — high-value compositions tested first
        pending.sort_by(|a, b| b.score.partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal));


        // ── Parallel candidate evaluation ──────────────────────────────
        // Check if inputs are safe to share across threads (no Rc-bearing variants).
        // If safe and batch is large enough, use rayon for parallel evaluation.
        // Otherwise fall back to sequential.
        let min_parallel_threshold: usize = std::env::var("SELPH_SEQ")
            .map(|_| usize::MAX)  // SELPH_SEQ=1 forces sequential
            .unwrap_or(128);
        let use_parallel = pending.len() >= min_parallel_threshold
            && crate::types::values_are_sync_safe(inputs)
            && crate::types::values_are_sync_safe(expected);

        if use_parallel {
            // ── Batched parallel path ────────────────────────────────────
            use rayon::prelude::*;

            let batch_size = {
                let threads = rayon::current_num_threads();
                (threads * 4).clamp(256, 4096).min(pending.len())
            };

            let mut pending_offset = 0;
            while pending_offset < pending.len() {
                let batch_end = (pending_offset + batch_size).min(pending.len());
                let batch = &pending[pending_offset..batch_end];
                pending_offset = batch_end;

                // PARALLEL PHASE: materialize + evaluate via VM
                let results: Vec<(SynthPool, EvalResult, usize)> = batch
                    .par_iter()
                    .map(|desc| {
                        let entry = materialize(desc, &pool, components);
                        let eval = eval_candidate(
                            &entry, inputs, expected,
                            &vm_ctx, &vm_macro_chunks, target_output_type,
                        );
                        (entry, eval, desc.comp_idx)
                    })
                    .collect();

                // SEQUENTIAL PHASE: dedup, budget, solution check, probes
                for (ri, (entry, eval, comp_idx)) in results.into_iter().enumerate() {
                    let desc = &batch[ri];

                    // Snapshot recording
                    if let Some(ref mut snap) = snapshot {
                        let comp = &components[desc.comp_idx];
                        let bn = comp.builtin.as_ref().unwrap();
                        let arg_psum = desc.score - comp.priority;
                        snap.push(CandidateRecord {
                            comp_name: bn.clone(),
                            arg_priority_sum: arg_psum,
                        });
                    }

                    let match_frac = match eval {
                        EvalResult::Skipped => { continue; }
                        EvalResult::NeedsTreeWalker => {
                            // Fall back to sequential test() for this candidate
                            let (result, mf) = test(&entry, &mut seen, &mut explored, &mut vm_stack);
                            if mf >= 0.0 {
                                let best = comp_best_match.entry(comp_idx).or_insert(0.0);
                                if mf > *best { *best = mf; }
                            }
                            if let Some((sn, sr)) = result {
                                return SynthResult::success(sn, sr, explored);
                            }
                            mf
                        }
                        EvalResult::Evaluated { beh, matches, evaluated } => {
                            explored += 1;
                            if explored > max_candidates {
                                return SynthResult { found: false, nodes: None, root: None, candidates_explored: explored };
                            }

                            let mf = if evaluated > 0 {
                                matches as f64 / inputs.len() as f64
                            } else { 0.0 };

                            // Observational equivalence dedup
                            if !beh.is_empty() {
                                if seen.contains(&beh) { continue; }
                                seen.insert(beh);
                            }

                            // RL reward
                            if mf >= 0.0 {
                                let best = comp_best_match.entry(comp_idx).or_insert(0.0);
                                if mf > *best { *best = mf; }
                            }

                            // Solution check
                            if matches == inputs.len() && !inputs.is_empty() {
                                if let Some(val_exs) = validation_examples {
                                    let mut ln = entry.nodes.clone();
                                    let lr = ln.len();
                                    ln.push(Node::Lambda(vec![intern("x")], entry.root));
                                    let ln_rc: Rc<[Node]> = ln.into();
                                    if !validate_candidate(&ln_rc, lr, val_exs, &macro_env, extra_bindings) {
                                        mf // continue — failed validation
                                    } else {
                                        return SynthResult::success(ln_rc.to_vec(), lr, explored);
                                    }
                                } else {
                                    let mut ln = entry.nodes.clone();
                                    let lr = ln.len();
                                    ln.push(Node::Lambda(vec![intern("x")], entry.root));
                                    return SynthResult::success(ln, lr, explored);
                                }
                            } else {
                                mf
                            }
                        }
                    };

                    // ── Early depth extension probes (sequential) ────────────
                    {
                        if let Some(target) = target_output_type {
                            for comp2 in components.iter() {
                                if comp2.arity != 1 || comp2.builtin.is_none() { continue; }
                                let returns_target = comp2.ret_type == target || comp2.ret_type == TYPE_ANY;
                                let returns_useful = useful_types.contains(&comp2.ret_type);
                                if !returns_target && !returns_useful { continue; }
                                if entry.ret_type != comp2.param_types[0] && comp2.param_types[0] != TYPE_ANY && entry.ret_type != TYPE_ANY {
                                    continue;
                                }
                                let bn2 = comp2.builtin.as_ref().unwrap();
                                let composed = if bn2.starts_with("map_") {
                                    let macro_name = &bn2[4..];
                                    let mut cn = entry.nodes.clone();
                                    let map_sym = cn.len();
                                    cn.push(Node::Symbol(crate::intern::intern("map")));
                                    let fn_sym = cn.len();
                                    cn.push(Node::Symbol(crate::intern::intern(macro_name)));
                                    let api = cn.len();
                                    cn.push(Node::App(vec![map_sym, fn_sym, entry.root]));
                                    SynthPool {
                                        nodes: cn, root: api,
                                        ret_type: comp2.ret_type,
                                        priority: comp2.priority + entry.priority,
                                    }
                                } else if bn2.starts_with("reduce_") {
                                    let fn_name = &bn2[7..];
                                    let mut cn = entry.nodes.clone();
                                    let reduce_sym = cn.len();
                                    cn.push(Node::Symbol(crate::intern::intern("reduce")));
                                    let fn_sym = cn.len();
                                    cn.push(Node::Symbol(crate::intern::intern(fn_name)));
                                    let api = cn.len();
                                    cn.push(Node::App(vec![reduce_sym, fn_sym, entry.root]));
                                    SynthPool {
                                        nodes: cn, root: api,
                                        ret_type: comp2.ret_type,
                                        priority: comp2.priority + entry.priority,
                                    }
                                } else {
                                    let mut cn = entry.nodes.clone();
                                    let fi = cn.len();
                                    cn.push(Node::Symbol(crate::intern::intern(bn2)));
                                    let api = cn.len();
                                    cn.push(Node::App(vec![fi, entry.root]));
                                    SynthPool {
                                        nodes: cn, root: api,
                                        ret_type: comp2.ret_type,
                                        priority: comp2.priority + entry.priority,
                                    }
                                };
                                let (cresult, _cfrac) = test(&composed, &mut seen, &mut explored, &mut vm_stack);
                                if let Some((sn, sr)) = cresult {
                                    return SynthResult::success(sn, sr, explored);
                                }
                                if _depth >= 2 && composed.ret_type != target && useful_types.contains(&composed.ret_type) {
                                    for comp3 in components.iter() {
                                        if comp3.arity != 2 || comp3.builtin.is_none() { continue; }
                                        if comp3.ret_type != target && comp3.ret_type != TYPE_ANY { continue; }
                                        let bn3 = comp3.builtin.as_ref().unwrap();
                                        let pool_len = all_end;
                                        let new_len = new_entries.len();
                                        for ji in 0..(pool_len + new_len) {
                                            let p = if ji < pool_len { &pool[ji] } else { &new_entries[ji - pool_len] };
                                            if (composed.ret_type == comp3.param_types[0] || comp3.param_types[0] == TYPE_ANY)
                                                && (p.ret_type == comp3.param_types[1] || comp3.param_types[1] == TYPE_ANY) {
                                                let mut cn = composed.nodes.clone();
                                                let off = cn.len();
                                                for nd in &p.nodes { cn.push(remap_node(nd, off)); }
                                                let fi = cn.len();
                                                cn.push(Node::Symbol(crate::intern::intern(bn3)));
                                                let api = cn.len();
                                                cn.push(Node::App(vec![fi, composed.root, p.root + off]));
                                                let chained = SynthPool {
                                                    nodes: cn, root: api, ret_type: comp3.ret_type,
                                                    priority: comp3.priority + composed.priority + p.priority,
                                                };
                                                let (cr, _) = test(&chained, &mut seen, &mut explored, &mut vm_stack);
                                                if let Some((sn, sr)) = cr { return SynthResult::success(sn, sr, explored); }
                                                if explored > max_candidates { break; }
                                            }
                                            if (p.ret_type == comp3.param_types[0] || comp3.param_types[0] == TYPE_ANY)
                                                && (composed.ret_type == comp3.param_types[1] || comp3.param_types[1] == TYPE_ANY) {
                                                let mut cn = p.nodes.clone();
                                                let off = cn.len();
                                                for nd in &composed.nodes { cn.push(remap_node(nd, off)); }
                                                let fi = cn.len();
                                                cn.push(Node::Symbol(crate::intern::intern(bn3)));
                                                let api = cn.len();
                                                cn.push(Node::App(vec![fi, p.root, composed.root + off]));
                                                let chained = SynthPool {
                                                    nodes: cn, root: api, ret_type: comp3.ret_type,
                                                    priority: comp3.priority + p.priority + composed.priority,
                                                };
                                                let (cr, _) = test(&chained, &mut seen, &mut explored, &mut vm_stack);
                                                if let Some((sn, sr)) = cr { return SynthResult::success(sn, sr, explored); }
                                                if explored > max_candidates { break; }
                                            }
                                        }
                                    }
                                }
                                if explored > max_candidates { break; }
                            }
                            'arity2_par: for comp2 in components.iter() {
                                if comp2.arity != 2 || comp2.builtin.is_none() { continue; }
                                if comp2.ret_type != target && comp2.ret_type != TYPE_ANY { continue; }
                                let bn2 = comp2.builtin.as_ref().unwrap();
                                let pool_len = all_end;
                                let new_len = new_entries.len();
                                for ji in 0..(pool_len + new_len) {
                                    let p = if ji < pool_len { &pool[ji] } else { &new_entries[ji - pool_len] };
                                    if (entry.ret_type == comp2.param_types[0] || comp2.param_types[0] == TYPE_ANY)
                                        && (p.ret_type == comp2.param_types[1] || comp2.param_types[1] == TYPE_ANY) {
                                        let mut cn = entry.nodes.clone();
                                        let off = cn.len();
                                        for nd in &p.nodes { cn.push(remap_node(nd, off)); }
                                        let fi = cn.len();
                                        cn.push(Node::Symbol(crate::intern::intern(bn2)));
                                        let api = cn.len();
                                        cn.push(Node::App(vec![fi, entry.root, p.root + off]));
                                        let composed = SynthPool {
                                            nodes: cn, root: api, ret_type: comp2.ret_type,
                                            priority: comp2.priority + entry.priority + p.priority,
                                        };
                                        let (cr, _) = test(&composed, &mut seen, &mut explored, &mut vm_stack);
                                        if let Some((sn, sr)) = cr { return SynthResult::success(sn, sr, explored); }
                                        if explored > max_candidates { break 'arity2_par; }
                                    }
                                    if (p.ret_type == comp2.param_types[0] || comp2.param_types[0] == TYPE_ANY)
                                        && (entry.ret_type == comp2.param_types[1] || comp2.param_types[1] == TYPE_ANY) {
                                        let mut cn = p.nodes.clone();
                                        let off = cn.len();
                                        for nd in &entry.nodes { cn.push(remap_node(nd, off)); }
                                        let fi = cn.len();
                                        cn.push(Node::Symbol(crate::intern::intern(bn2)));
                                        let api = cn.len();
                                        cn.push(Node::App(vec![fi, p.root, entry.root + off]));
                                        let composed = SynthPool {
                                            nodes: cn, root: api, ret_type: comp2.ret_type,
                                            priority: comp2.priority + p.priority + entry.priority,
                                        };
                                        let (cr, _) = test(&composed, &mut seen, &mut explored, &mut vm_stack);
                                        if let Some((sn, sr)) = cr { return SynthResult::success(sn, sr, explored); }
                                        if explored > max_candidates { break 'arity2_par; }
                                    }
                                }
                            }
                        }
                    }

                    if explored > max_candidates {
                        return SynthResult { found: false, nodes: None, root: None, candidates_explored: explored };
                    }
                    if pool.len() + new_entries.len() < MAX_POOL
                        && useful_types.contains(&entry.ret_type)
                    {
                        let mut adjusted_entry = entry;
                        if match_frac == 0.0 {
                            if let Some(target) = target_output_type {
                                if adjusted_entry.ret_type == target {
                                    adjusted_entry.priority += rl_coeffs.cold_penalty;
                                }
                            }
                        } else if match_frac > 0.0 {
                            adjusted_entry.priority += match_frac * rl_coeffs.warm_bonus;
                        }
                        new_entries.push(adjusted_entry);
                    }
                }
            }
        } else {
            // ── Sequential fallback ──────────────────────────────────────
            for desc in &pending {
                let entry = materialize(desc, &pool, components);

                // Record candidate in snapshot before testing
                if let Some(ref mut snap) = snapshot {
                    let comp = &components[desc.comp_idx];
                    let bn = comp.builtin.as_ref().unwrap();
                    let arg_psum = desc.score - comp.priority;
                    snap.push(CandidateRecord {
                        comp_name: bn.clone(),
                        arg_priority_sum: arg_psum,
                    });
                }

                let (result, match_frac) = test(&entry, &mut seen, &mut explored, &mut vm_stack);

                // RL reward: track best partial match per component
                if match_frac >= 0.0 {
                    let best = comp_best_match.entry(desc.comp_idx).or_insert(0.0);
                    if match_frac > *best { *best = match_frac; }
                }

                if let Some((sn, sr)) = result {
                    return SynthResult::success(sn, sr, explored);
                }

                // ── Early depth extension probes ────────────────────────
                {
                    if let Some(target) = target_output_type {
                        for comp2 in components.iter() {
                            if comp2.arity != 1 || comp2.builtin.is_none() { continue; }
                            let returns_target = comp2.ret_type == target || comp2.ret_type == TYPE_ANY;
                            let returns_useful = useful_types.contains(&comp2.ret_type);
                            if !returns_target && !returns_useful { continue; }
                            if entry.ret_type != comp2.param_types[0] && comp2.param_types[0] != TYPE_ANY && entry.ret_type != TYPE_ANY {
                                continue;
                            }
                            let bn2 = comp2.builtin.as_ref().unwrap();
                            let composed = if bn2.starts_with("map_") {
                                let macro_name = &bn2[4..];
                                let mut cn = entry.nodes.clone();
                                let map_sym = cn.len();
                                cn.push(Node::Symbol(crate::intern::intern("map")));
                                let fn_sym = cn.len();
                                cn.push(Node::Symbol(crate::intern::intern(macro_name)));
                                let api = cn.len();
                                cn.push(Node::App(vec![map_sym, fn_sym, entry.root]));
                                SynthPool {
                                    nodes: cn, root: api,
                                    ret_type: comp2.ret_type,
                                    priority: comp2.priority + entry.priority,
                                }
                            } else if bn2.starts_with("reduce_") {
                                let fn_name = &bn2[7..];
                                let mut cn = entry.nodes.clone();
                                let reduce_sym = cn.len();
                                cn.push(Node::Symbol(crate::intern::intern("reduce")));
                                let fn_sym = cn.len();
                                cn.push(Node::Symbol(crate::intern::intern(fn_name)));
                                let api = cn.len();
                                cn.push(Node::App(vec![reduce_sym, fn_sym, entry.root]));
                                SynthPool {
                                    nodes: cn, root: api,
                                    ret_type: comp2.ret_type,
                                    priority: comp2.priority + entry.priority,
                                }
                            } else {
                                let mut cn = entry.nodes.clone();
                                let fi = cn.len();
                                cn.push(Node::Symbol(crate::intern::intern(bn2)));
                                let api = cn.len();
                                cn.push(Node::App(vec![fi, entry.root]));
                                SynthPool {
                                    nodes: cn, root: api,
                                    ret_type: comp2.ret_type,
                                    priority: comp2.priority + entry.priority,
                                }
                            };
                            let (cresult, _cfrac) = test(&composed, &mut seen, &mut explored, &mut vm_stack);
                            if let Some((sn, sr)) = cresult {
                                return SynthResult::success(sn, sr, explored);
                            }
                            if _depth >= 2 && composed.ret_type != target && useful_types.contains(&composed.ret_type) {
                                for comp3 in components.iter() {
                                    if comp3.arity != 2 || comp3.builtin.is_none() { continue; }
                                    if comp3.ret_type != target && comp3.ret_type != TYPE_ANY { continue; }
                                    let bn3 = comp3.builtin.as_ref().unwrap();
                                    let pool_len = all_end;
                                    let new_len = new_entries.len();
                                    for ji in 0..(pool_len + new_len) {
                                        let p = if ji < pool_len { &pool[ji] } else { &new_entries[ji - pool_len] };
                                        if (composed.ret_type == comp3.param_types[0] || comp3.param_types[0] == TYPE_ANY)
                                            && (p.ret_type == comp3.param_types[1] || comp3.param_types[1] == TYPE_ANY) {
                                            let mut cn = composed.nodes.clone();
                                            let off = cn.len();
                                            for nd in &p.nodes { cn.push(remap_node(nd, off)); }
                                            let fi = cn.len();
                                            cn.push(Node::Symbol(crate::intern::intern(bn3)));
                                            let api = cn.len();
                                            cn.push(Node::App(vec![fi, composed.root, p.root + off]));
                                            let chained = SynthPool {
                                                nodes: cn, root: api, ret_type: comp3.ret_type,
                                                priority: comp3.priority + composed.priority + p.priority,
                                            };
                                            let (cr, _) = test(&chained, &mut seen, &mut explored, &mut vm_stack);
                                            if let Some((sn, sr)) = cr { return SynthResult::success(sn, sr, explored); }
                                            if explored > max_candidates { break; }
                                        }
                                        if (p.ret_type == comp3.param_types[0] || comp3.param_types[0] == TYPE_ANY)
                                            && (composed.ret_type == comp3.param_types[1] || comp3.param_types[1] == TYPE_ANY) {
                                            let mut cn = p.nodes.clone();
                                            let off = cn.len();
                                            for nd in &composed.nodes { cn.push(remap_node(nd, off)); }
                                            let fi = cn.len();
                                            cn.push(Node::Symbol(crate::intern::intern(bn3)));
                                            let api = cn.len();
                                            cn.push(Node::App(vec![fi, p.root, composed.root + off]));
                                            let chained = SynthPool {
                                                nodes: cn, root: api, ret_type: comp3.ret_type,
                                                priority: comp3.priority + p.priority + composed.priority,
                                            };
                                            let (cr, _) = test(&chained, &mut seen, &mut explored, &mut vm_stack);
                                            if let Some((sn, sr)) = cr { return SynthResult::success(sn, sr, explored); }
                                            if explored > max_candidates { break; }
                                        }
                                    }
                                }
                            }
                            if explored > max_candidates { break; }
                        }
                        'arity2: for comp2 in components.iter() {
                            if comp2.arity != 2 || comp2.builtin.is_none() { continue; }
                            if comp2.ret_type != target && comp2.ret_type != TYPE_ANY { continue; }
                            let bn2 = comp2.builtin.as_ref().unwrap();
                            let pool_len = all_end;
                            let new_len = new_entries.len();
                            for ji in 0..(pool_len + new_len) {
                                let p = if ji < pool_len { &pool[ji] } else { &new_entries[ji - pool_len] };
                                if (entry.ret_type == comp2.param_types[0] || comp2.param_types[0] == TYPE_ANY)
                                    && (p.ret_type == comp2.param_types[1] || comp2.param_types[1] == TYPE_ANY) {
                                    let mut cn = entry.nodes.clone();
                                    let off = cn.len();
                                    for nd in &p.nodes { cn.push(remap_node(nd, off)); }
                                    let fi = cn.len();
                                    cn.push(Node::Symbol(crate::intern::intern(bn2)));
                                    let api = cn.len();
                                    cn.push(Node::App(vec![fi, entry.root, p.root + off]));
                                    let composed = SynthPool {
                                        nodes: cn, root: api, ret_type: comp2.ret_type,
                                        priority: comp2.priority + entry.priority + p.priority,
                                    };
                                    let (cr, _) = test(&composed, &mut seen, &mut explored, &mut vm_stack);
                                    if let Some((sn, sr)) = cr { return SynthResult::success(sn, sr, explored); }
                                    if explored > max_candidates { break 'arity2; }
                                }
                                if (p.ret_type == comp2.param_types[0] || comp2.param_types[0] == TYPE_ANY)
                                    && (entry.ret_type == comp2.param_types[1] || comp2.param_types[1] == TYPE_ANY) {
                                    let mut cn = p.nodes.clone();
                                    let off = cn.len();
                                    for nd in &entry.nodes { cn.push(remap_node(nd, off)); }
                                    let fi = cn.len();
                                    cn.push(Node::Symbol(crate::intern::intern(bn2)));
                                    let api = cn.len();
                                    cn.push(Node::App(vec![fi, p.root, entry.root + off]));
                                    let composed = SynthPool {
                                        nodes: cn, root: api, ret_type: comp2.ret_type,
                                        priority: comp2.priority + p.priority + entry.priority,
                                    };
                                    let (cr, _) = test(&composed, &mut seen, &mut explored, &mut vm_stack);
                                    if let Some((sn, sr)) = cr { return SynthResult::success(sn, sr, explored); }
                                    if explored > max_candidates { break 'arity2; }
                                }
                            }
                        }
                    }
                }

                if explored > max_candidates {
                    return SynthResult { found: false, nodes: None, root: None, candidates_explored: explored };
                }
                if pool.len() + new_entries.len() < MAX_POOL
                    && useful_types.contains(&entry.ret_type)
                {
                    let mut adjusted_entry = entry;
                    if match_frac == 0.0 {
                        if let Some(target) = target_output_type {
                            if adjusted_entry.ret_type == target {
                                adjusted_entry.priority += rl_coeffs.cold_penalty;
                            }
                        }
                    } else if match_frac > 0.0 {
                        adjusted_entry.priority += match_frac * rl_coeffs.warm_bonus;
                    }
                    new_entries.push(adjusted_entry);
                }
            }
        }

        // ── RL priority update: boost components with partial matches ──
        // Components whose candidates got partial matches are "warm" —
        // their depth-(N+1) compositions should be tried earlier.
        if rl_coeffs.comp_warm_bonus != 0.0 {
            for (&ci, &best_match) in &comp_best_match {
                if best_match > 0.0 {
                    comp_priority_boost[ci] += best_match * rl_coeffs.comp_warm_bonus;
                }
            }
            comp_best_match.clear();
        }

        // ── If-expression generation (when enabled) ─────────────────

        if enable_if && !inputs.is_empty() {
            let if_entries = generate_if_programs(
                &pool, 0..pool.len() + new_entries.len(),
                inputs, expected, &macro_env, extra_bindings,
            );
            for e in if_entries {
                let (result, _) = test(&e, &mut seen, &mut explored, &mut vm_stack);
                if let Some((sn, sr)) = result {
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
    let nodes_rc: Rc<[Node]> = nodes.to_vec().into();
    for (inp, exp) in validation_examples {
        let mut env = eval::make_default_env();
        for (nm, ps, mn, mr) in macro_env {
            env_define(
                &mut env,
                intern(nm),
                Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr),
            );
        }
        for (name, val) in extra_bindings {
            env_define(&mut env, intern(name), val.clone());
        }
        let fv = match eval::eval(&nodes_rc, lambda_root, &mut env) {
            Ok(v) => v,
            Err(_) => return false,
        };
        match eval::apply(&fv, &[inp.clone()], &nodes_rc, &mut env) {
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
        ln.push(Node::Lambda(vec![intern("x")], p.root));
        let ln_rc: Rc<[Node]> = ln.into();

        let mut pattern = Vec::with_capacity(inputs.len());
        let mut valid = true;

        for inp in inputs {
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in macro_env {
                env_define(
                    &mut env,
                    intern(nm),
                    Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr),
                );
            }
            for (name, val) in extra_bindings {
                env_define(&mut env, intern(name), val.clone());
            }
            let fv = match eval::eval(&ln_rc, lr, &mut env) {
                Ok(v) => v,
                Err(_) => { valid = false; break; }
            };
            match eval::apply(&fv, &[inp.clone()], &ln_rc, &mut env) {
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
        ln.push(Node::Lambda(vec![intern("x")], p.root));
        let ln_rc: Rc<[Node]> = ln.into();

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut valid = true;

        for inp in inputs {
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in macro_env {
                env_define(
                    &mut env,
                    intern(nm),
                    Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr),
                );
            }
            for (name, val) in extra_bindings {
                env_define(&mut env, intern(name), val.clone());
            }
            let fv = match eval::eval(&ln_rc, lr, &mut env) {
                Ok(v) => v,
                Err(_) => { valid = false; break; }
            };
            match eval::apply(&fv, &[inp.clone()], &ln_rc, &mut env) {
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
            if !is_builtin_name(&resolve(*name)) {
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
        let mnodes_rc: Rc<[Node]> = mnodes.to_vec().into();
        if let Ok(val) = eval::eval(&mnodes_rc, mroot, &mut env) {
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

    let mnodes_rc: Rc<[Node]> = mnodes.to_vec().into();
    let try_call = |args: &[Value]| -> Option<Value> {
        let mut env = eval::make_default_env();
        let val = Value::RustMacro(
            params.iter().map(|s| intern(s)).collect(),
            mnodes_rc.clone(),
            mroot,
        );
        eval::apply(&val, args, &mnodes_rc, &mut env).ok()
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
    default_synth_components_opts(macros, false)
}

/// Build synthesis components with optional grid domain.
/// When `include_grid` is false, all grid-typed components are omitted,
/// reducing the search space for non-grid tasks.
pub fn default_synth_components_opts(
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    include_grid: bool,
) -> Vec<SynthComponent> {
    let mut comps = vec![
        SynthComponent { name: "x".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 100.0, usage_count: 0.0 },
        SynthComponent { name: "0".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "1".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "2".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "3".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "4".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "5".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "6".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "7".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "10".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "-1".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        // String constants — common characters for formal language tasks
        SynthComponent { name: "a".into(), builtin: None, arity: 0, ret_type: 1, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "b".into(), builtin: None, arity: 0, ret_type: 1, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: "(".into(), builtin: None, arity: 0, ret_type: 1, param_types: vec![], priority: 0.0, usage_count: 0.0 },
        SynthComponent { name: ")".into(), builtin: None, arity: 0, ret_type: 1, param_types: vec![], priority: 0.0, usage_count: 0.0 },
    ];

    // Unary num->num
    for name in &["abs", "negate"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: 0, param_types: vec![0], priority: 0.0, usage_count: 0.0 });
    }

    // Binary num->num->num
    for name in &["add", "subtract", "multiply", "min", "max", "modulo"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: 0, param_types: vec![0, 0], priority: 0.0, usage_count: 0.0 });
    }

    // String ops: unary str->str
    for name in &["string-upper", "string-lower", "string-reverse", "string-trim"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: 1, param_types: vec![1], priority: 0.0, usage_count: 0.0 });
    }

    // string-length: str->num
    comps.push(SynthComponent {
        name: "string-length".into(), builtin: Some("string-length".into()),
        arity: 1, ret_type: 0, param_types: vec![1], priority: 0.0, usage_count: 0.0 });

    // string-nth: (str, num) -> str (get character at index)
    comps.push(SynthComponent {
        name: "string-nth".into(), builtin: Some("string-nth".into()),
        arity: 2, ret_type: 1, param_types: vec![1, 0], priority: 0.0, usage_count: 0.0 });

    // char-code: str -> num (character to ASCII code)
    comps.push(SynthComponent {
        name: "char-code".into(), builtin: Some("char-code".into()),
        arity: 1, ret_type: 0, param_types: vec![1], priority: 0.0, usage_count: 0.0 });

    // code-char: num -> str (ASCII code to character)
    comps.push(SynthComponent {
        name: "code-char".into(), builtin: Some("code-char".into()),
        arity: 1, ret_type: 1, param_types: vec![0], priority: 0.0, usage_count: 0.0 });

    // count-char: (str, str) -> num (count occurrences)
    comps.push(SynthComponent {
        name: "count-char".into(), builtin: Some("count-char".into()),
        arity: 2, ret_type: 0, param_types: vec![1, 1], priority: 0.0, usage_count: 0.0 });

    // string-replace: (str, str, str) -> str
    comps.push(SynthComponent {
        name: "string-replace".into(), builtin: Some("string-replace".into()),
        arity: 3, ret_type: 1, param_types: vec![1, 1, 1], priority: 0.0, usage_count: 0.0 });

    // concat: (str, str) -> str
    comps.push(SynthComponent {
        name: "concat".into(), builtin: Some("concat".into()),
        arity: 2, ret_type: 1, param_types: vec![1, 1], priority: 0.0, usage_count: 0.0 });

    // string-starts-with: (str, str) -> bool
    comps.push(SynthComponent {
        name: "string-starts-with".into(), builtin: Some("string-starts-with".into()),
        arity: 2, ret_type: 2, param_types: vec![1, 1], priority: 0.0, usage_count: 0.0 });

    // string-ends-with: (str, str) -> bool
    comps.push(SynthComponent {
        name: "string-ends-with".into(), builtin: Some("string-ends-with".into()),
        arity: 2, ret_type: 2, param_types: vec![1, 1], priority: 0.0, usage_count: 0.0 });

    // dispatch: (str, any) -> any — look up macro by name, apply to arg
    // Enables instruction-following: (dispatch (first_word x) (last_word x))
    comps.push(SynthComponent {
        name: "dispatch".into(), builtin: Some("dispatch".into()),
        arity: 2, ret_type: TYPE_ANY, param_types: vec![TYPE_STR, TYPE_ANY], priority: 15.0, usage_count: 0.0 });

    // Comparison operators: num->num->bool (for if-expression conditions)
    for name in &["<", ">", "<=", ">=", "=", "!="] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: 2, param_types: vec![0, 0], priority: 0.0, usage_count: 0.0 });
    }

    // Unary num->bool predicates
    for name in &["even", "odd"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: 2, param_types: vec![0], priority: 0.0, usage_count: 0.0 });
    }

    // Boolean logic: bool->bool, (bool,bool)->bool
    comps.push(SynthComponent {
        name: "not".into(), builtin: Some("not".into()),
        arity: 1, ret_type: 2, param_types: vec![2], priority: 0.0, usage_count: 0.0 });
    comps.push(SynthComponent {
        name: "and".into(), builtin: Some("and".into()),
        arity: 2, ret_type: 2, param_types: vec![2, 2], priority: 0.0, usage_count: 0.0 });
    comps.push(SynthComponent {
        name: "or".into(), builtin: Some("or".into()),
        arity: 2, ret_type: 2, param_types: vec![2, 2], priority: 0.0, usage_count: 0.0 });

    // List operations
    comps.push(SynthComponent {
        name: "string-split".into(), builtin: Some("string-split".into()),
        arity: 2, ret_type: TYPE_LIST, param_types: vec![TYPE_STR, TYPE_STR], priority: 10.0, usage_count: 0.0 });
    comps.push(SynthComponent {
        name: "string-join".into(), builtin: Some("string-join".into()),
        arity: 2, ret_type: TYPE_STR, param_types: vec![TYPE_LIST, TYPE_STR], priority: 10.0, usage_count: 0.0 });
    comps.push(SynthComponent {
        name: "head".into(), builtin: Some("head".into()),
        arity: 1, ret_type: TYPE_ANY, param_types: vec![TYPE_LIST], priority: 0.0, usage_count: 0.0 });
    comps.push(SynthComponent {
        name: "tail".into(), builtin: Some("tail".into()),
        arity: 1, ret_type: TYPE_LIST, param_types: vec![TYPE_LIST], priority: 0.0, usage_count: 0.0 });
    comps.push(SynthComponent {
        name: "nth".into(), builtin: Some("nth".into()),
        arity: 2, ret_type: TYPE_ANY, param_types: vec![TYPE_LIST, TYPE_NUM], priority: 0.0, usage_count: 0.0 });
    comps.push(SynthComponent {
        name: "list-length".into(), builtin: Some("length".into()),
        arity: 1, ret_type: TYPE_NUM, param_types: vec![TYPE_LIST], priority: 0.0, usage_count: 0.0 });
    comps.push(SynthComponent {
        name: "list-reverse".into(), builtin: Some("reverse".into()),
        arity: 1, ret_type: TYPE_LIST, param_types: vec![TYPE_LIST], priority: 0.0, usage_count: 0.0 });

    // string-chars: str -> list (split string into character list)
    comps.push(SynthComponent {
        name: "string-chars".into(), builtin: Some("string-chars".into()),
        arity: 1, ret_type: TYPE_LIST, param_types: vec![TYPE_STR], priority: 5.0, usage_count: 0.0 });

    // string-take: (str, num) -> str (first N characters)
    comps.push(SynthComponent {
        name: "string-take".into(), builtin: Some("string-take".into()),
        arity: 2, ret_type: TYPE_STR, param_types: vec![TYPE_STR, TYPE_NUM], priority: 5.0, usage_count: 0.0 });

    // string-drop: (str, num) -> str (everything after first N characters)
    comps.push(SynthComponent {
        name: "string-drop".into(), builtin: Some("string-drop".into()),
        arity: 2, ret_type: TYPE_STR, param_types: vec![TYPE_STR, TYPE_NUM], priority: 5.0, usage_count: 0.0 });

    // divide: (num, num) -> num
    comps.push(SynthComponent {
        name: "divide".into(), builtin: Some("divide".into()),
        arity: 2, ret_type: TYPE_NUM, param_types: vec![TYPE_NUM, TYPE_NUM], priority: 0.0, usage_count: 0.0 });

    // floor: num -> num
    comps.push(SynthComponent {
        name: "floor".into(), builtin: Some("floor".into()),
        arity: 1, ret_type: TYPE_NUM, param_types: vec![TYPE_NUM], priority: 0.0, usage_count: 0.0 });

    // string-slice: (str, num, num) -> str
    comps.push(SynthComponent {
        name: "string-slice".into(), builtin: Some("string-slice".into()),
        arity: 3, ret_type: TYPE_STR, param_types: vec![TYPE_STR, TYPE_NUM, TYPE_NUM], priority: 5.0, usage_count: 0.0 });

    // ── Grid components (ARC-AGI) — only when grid domain is active ──
    if include_grid {

    // Grid unary transforms: Grid → Grid
    for name in &[
        "grid-rotate-cw", "grid-rotate-ccw", "grid-rotate-180",
        "grid-flip-h", "grid-flip-v", "grid-transpose", "grid-trim",
        "grid-border",
    ] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: TYPE_GRID, param_types: vec![TYPE_GRID], priority: 0.0, usage_count: 0.0 });
    }
    // Grid → Num analysis
    for name in &[
        "grid-width", "grid-height", "grid-most-common", "grid-background",
        "grid-object-count", "grid-object-area",
    ] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: TYPE_NUM, param_types: vec![TYPE_GRID], priority: 0.0, usage_count: 0.0 });
    }
    // Grid → Bool predicates
    for name in &["grid-symmetric-h", "grid-symmetric-v", "grid-is-rectangle"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: TYPE_BOOL, param_types: vec![TYPE_GRID], priority: 0.0, usage_count: 0.0 });
    }
    // Grid → List analysis
    for name in &[
        "grid-colors", "grid-bounding-box", "grid-size", "grid-object-center",
        "grid-objects", "grid-objects-8", "grid-object-colors",
        "grid-detect-rectangles", "grid-quarter",
    ] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: TYPE_LIST, param_types: vec![TYPE_GRID], priority: 0.0, usage_count: 0.0 });
    }
    // Grid × Num → Grid
    for name in &["grid-scale", "grid-gravity", "grid-fill-enclosed"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: TYPE_GRID, param_types: vec![TYPE_GRID, TYPE_NUM], priority: 0.0, usage_count: 0.0 });
    }
    // Grid × Num → Num
    comps.push(SynthComponent {
        name: "grid-count-color".into(), builtin: Some("grid-count-color".into()),
        arity: 2, ret_type: TYPE_NUM, param_types: vec![TYPE_GRID, TYPE_NUM], priority: 0.0, usage_count: 0.0 });
    // Grid × Num → List
    for name in &["grid-row", "grid-col", "grid-find-color", "grid-hsplit", "grid-vsplit"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: TYPE_LIST, param_types: vec![TYPE_GRID, TYPE_NUM], priority: 0.0, usage_count: 0.0 });
    }
    // Grid × Num × Num → Grid
    for name in &["grid-replace-color", "grid-tile", "grid-pad"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 3, ret_type: TYPE_GRID, param_types: vec![TYPE_GRID, TYPE_NUM, TYPE_NUM], priority: 0.0, usage_count: 0.0 });
    }
    // Grid × Num × Num → Num
    for name in &["grid-get", "grid-neighbor-count"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 3, ret_type: TYPE_NUM, param_types: vec![TYPE_GRID, TYPE_NUM, TYPE_NUM], priority: 0.0, usage_count: 0.0 });
    }
    // Grid × Grid → Grid
    for name in &["grid-hconcat", "grid-vconcat", "grid-mask", "grid-xor", "grid-and", "grid-or", "grid-overlay-center"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: TYPE_GRID, param_types: vec![TYPE_GRID, TYPE_GRID], priority: 0.0, usage_count: 0.0 });
    }
    // Grid × Grid → Bool
    for name in &["grid-equal", "grid-dimensions-equal", "grid-objects-touching"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: TYPE_BOOL, param_types: vec![TYPE_GRID, TYPE_GRID], priority: 0.0, usage_count: 0.0 });
    }
    // Grid × Grid → List
    comps.push(SynthComponent {
        name: "grid-find-subgrid".into(), builtin: Some("grid-find-subgrid".into()),
        arity: 2, ret_type: TYPE_LIST, param_types: vec![TYPE_GRID, TYPE_GRID], priority: 0.0, usage_count: 0.0 });
    // Grid × Num × Num × Num → Grid (flood-fill)
    comps.push(SynthComponent {
        name: "grid-flood-fill".into(), builtin: Some("grid-flood-fill".into()),
        arity: 4, ret_type: TYPE_GRID, param_types: vec![TYPE_GRID, TYPE_NUM, TYPE_NUM, TYPE_NUM], priority: 0.0, usage_count: 0.0 });
    // High-arity: draw-line-h (5), draw-line-v (5), ray (6), fill-rect (6)
    // These are available as builtins but NOT registered as synth components
    // because arity >= 5 makes them unreachable at typical search depths.
    // They'll be composed into macros by the curriculum instead.

    // Grid-make: Num × Num × Num → Grid
    comps.push(SynthComponent {
        name: "grid-make".into(), builtin: Some("grid-make".into()),
        arity: 3, ret_type: TYPE_GRID, param_types: vec![TYPE_NUM, TYPE_NUM, TYPE_NUM], priority: 0.0, usage_count: 0.0 });
    } // end if include_grid

    // Add macro components — infer types by probing with sample inputs
    for (mname, params, mnodes, mroot) in macros {
        let (inferred_param, inferred_ret) = infer_macro_types(mname, params, mnodes, *mroot);
        comps.push(SynthComponent {
            name: mname.clone(),
            builtin: Some(mname.clone()),
            arity: params.len(),
            ret_type: inferred_ret,
            param_types: inferred_param,
            priority: 30.0, usage_count: 0.0 });
    }

    // ns-get: (any, str) -> any — namespace lookup
    comps.push(SynthComponent {
        name: "ns-get".into(), builtin: Some("ns-get".into()),
        arity: 2, ret_type: TYPE_ANY, param_types: vec![TYPE_ANY, TYPE_STR],
        priority: 10.0, usage_count: 0.0 });

    // ns-get-or: (any, str, any) -> any — namespace lookup with default
    // Enables compositions like (ns-get-or vocab x "") for data-driven synthesis
    comps.push(SynthComponent {
        name: "ns-get-or".into(), builtin: Some("ns-get-or".into()),
        arity: 3, ret_type: TYPE_ANY, param_types: vec![TYPE_ANY, TYPE_STR, TYPE_ANY],
        priority: 10.0, usage_count: 0.0 });

    // Fused map components: for each unary macro m, register map_m(list) -> list
    // These emit (map m list) during materialization, enabling higher-order synthesis
    // without teaching the synthesizer about function-typed pool entries.
    for (mname, params, _mnodes, _mroot) in macros {
        if params.len() == 1 {
            comps.push(SynthComponent {
                name: format!("map_{}", mname),
                builtin: Some(format!("map_{}", mname)),
                arity: 1,
                ret_type: TYPE_LIST,
                param_types: vec![TYPE_LIST],
                priority: 25.0,
                usage_count: 0.0,
            });
        }
    }

    // Fused reduce components for binary builtins with known return types.
    // These emit (reduce <fn_name> <arg>) during materialization.
    // NOT and/or — they are special forms, not callable values.
    for &(name, ret) in &[
        ("add", TYPE_NUM), ("subtract", TYPE_NUM), ("multiply", TYPE_NUM),
        ("min", TYPE_NUM), ("max", TYPE_NUM), ("concat", TYPE_STR),
    ] {
        comps.push(SynthComponent {
            name: format!("reduce_{}", name),
            builtin: Some(format!("reduce_{}", name)),
            arity: 1,
            ret_type: ret,
            param_types: vec![TYPE_LIST],
            priority: 25.0,
            usage_count: 0.0,
        });
    }

    // Fused reduce components for binary macros: reduce_m(list) -> inferred_ret
    for (mname, params, mnodes, mroot) in macros {
        if params.len() == 2 {
            let (_, inferred_ret) = infer_macro_types(mname, params, mnodes, *mroot);
            comps.push(SynthComponent {
                name: format!("reduce_{}", mname),
                builtin: Some(format!("reduce_{}", mname)),
                arity: 1,
                ret_type: inferred_ret,
                param_types: vec![TYPE_LIST],
                priority: 25.0,
                usage_count: 0.0,
            });
        }
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
                extra_bindings.push((resolve(*k).to_string(), v.clone()));
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
    let fitness_nodes_rc: Rc<[Node]> = fitness_nodes.to_vec().into();
    let eval_fitness = |entry: &SynthPool, macro_env: &[(String, Vec<String>, Vec<Node>, usize)]| -> Option<f64> {
        // Wrap body in (lambda (x) body)
        let mut ln = entry.nodes.clone();
        let lr = ln.len();
        ln.push(Node::Lambda(vec![intern("x")], entry.root));
        let ln_rc: Rc<[Node]> = ln.into();

        // If base examples are provided, check them first
        if !base_inputs.is_empty() {
            for (inp, exp) in base_inputs.iter().zip(base_expected.iter()) {
                let mut env = eval::make_default_env();
                for (nm, ps, mn, mr) in macro_env {
                    env_define(&mut env, intern(nm), Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr));
                }
                for (name, val) in extra_bindings {
                    env_define(&mut env, intern(name), val.clone());
                }
                let fv = match eval::eval(&ln_rc, lr, &mut env) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                match eval::apply(&fv, &[inp.clone()], &ln_rc, &mut env) {
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
            env_define(&mut env, intern(nm), Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr));
        }
        for (name, val) in extra_bindings {
            env_define(&mut env, intern(name), val.clone());
        }

        // Evaluate the fitness function node
        let ffit = match eval::eval(&fitness_nodes_rc, fitness_root, &mut env) {
            Ok(v) => v,
            Err(_) => return None,
        };

        // Evaluate the candidate lambda
        let candidate = match eval::eval(&ln_rc, lr, &mut env) {
            Ok(v) => v,
            Err(_) => return None,
        };

        // Apply fitness_fn(candidate)
        match eval::apply(&ffit, &[candidate], &ln_rc, &mut env) {
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
        ln.push(Node::Lambda(vec![intern("x")], entry.root));
        let ln_rc: Rc<[Node]> = ln.into();

        let mut beh = Vec::new();
        if !base_inputs.is_empty() {
            for inp in base_inputs {
                let mut env = eval::make_default_env();
                for (nm, ps, mn, mr) in &macro_env {
                    env_define(&mut env, intern(nm), Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr));
                }
                for (name, val) in extra_bindings {
                    env_define(&mut env, intern(name), val.clone());
                }
                let fv = match eval::eval(&ln_rc, lr, &mut env) {
                    Ok(v) => v,
                    Err(_) => return,
                };
                match eval::apply(&fv, &[inp.clone()], &ln_rc, &mut env) {
                    Ok(a) => beh.push(val_hash(&a)),
                    Err(_) => return,
                }
            }
        } else {
            // No base inputs: use direct evaluation for dedup
            let mut env = eval::make_default_env();
            for (nm, ps, mn, mr) in &macro_env {
                env_define(&mut env, intern(nm), Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), Rc::from(mn.clone()), *mr));
            }
            for (name, val) in extra_bindings {
                env_define(&mut env, intern(name), val.clone());
            }
            let entry_nodes_rc: Rc<[Node]> = entry.nodes.clone().into();
            match eval::eval(&entry_nodes_rc, entry.root, &mut env) {
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
                ln.push(Node::Lambda(vec![intern("x")], entry.root));
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
            nodes.push(Node::Symbol(intern("x")));
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
                    if p.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p.ret_type != 255 {
                        continue;
                    }
                    let mut n = p.nodes.clone();
                    let fi = n.len();
                    n.push(Node::Symbol(intern(bn)));
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
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 {
                        continue;
                    }
                    for ai in 0..all_end {
                        let p2 = &pool[ai];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 {
                            continue;
                        }
                        let mut n = p1.nodes.clone();
                        let off = n.len();
                        for nd in &p2.nodes {
                            n.push(remap_node(nd, off));
                        }
                        let fi = n.len();
                        n.push(Node::Symbol(intern(bn)));
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
                        if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 {
                            continue;
                        }
                        for pi in prev.clone() {
                            let p2 = &pool[pi];
                            if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 {
                                continue;
                            }
                            let mut n = p1.nodes.clone();
                            let off = n.len();
                            for nd in &p2.nodes {
                                n.push(remap_node(nd, off));
                            }
                            let fi = n.len();
                            n.push(Node::Symbol(intern(bn)));
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
            } else if comp.arity == 3 {
                // Case 1: arg1 from prev, arg2+arg3 from all
                'a3c1: for pi in prev.clone() {
                    let p1 = &pool[pi];
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 { continue; }
                    for a2 in 0..all_end {
                        let p2 = &pool[a2];
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 { continue; }
                        for a3 in 0..all_end {
                            let p3 = &pool[a3];
                            if p3.ret_type != comp.param_types[2] && comp.param_types[2] != 255 && p3.ret_type != 255 { continue; }
                            let mut n = p1.nodes.clone();
                            let off2 = n.len();
                            for nd in &p2.nodes { n.push(remap_node(nd, off2)); }
                            let off3 = n.len();
                            for nd in &p3.nodes { n.push(remap_node(nd, off3)); }
                            let fi = n.len();
                            n.push(Node::Symbol(intern(bn)));
                            let api = n.len();
                            n.push(Node::App(vec![fi, p1.root, p2.root + off2, p3.root + off3]));
                            let e = SynthPool { nodes: n, root: api, ret_type: comp.ret_type, priority: comp.priority };
                            test_and_score(&e, &mut seen, &mut explored);
                            if explored > max_candidates { break 'a3c1; }
                            if pool.len() + new_entries.len() < MAX_POOL { new_entries.push(e); }
                        }
                    }
                }
                // Case 2: arg1 from old, arg2 from prev, arg3 from all
                if explored <= max_candidates {
                    'a3c2: for a1 in 0..prev_start {
                        let p1 = &pool[a1];
                        if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 { continue; }
                        for pi in prev.clone() {
                            let p2 = &pool[pi];
                            if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 { continue; }
                            for a3 in 0..all_end {
                                let p3 = &pool[a3];
                                if p3.ret_type != comp.param_types[2] && comp.param_types[2] != 255 && p3.ret_type != 255 { continue; }
                                let mut n = p1.nodes.clone();
                                let off2 = n.len();
                                for nd in &p2.nodes { n.push(remap_node(nd, off2)); }
                                let off3 = n.len();
                                for nd in &p3.nodes { n.push(remap_node(nd, off3)); }
                                let fi = n.len();
                                n.push(Node::Symbol(intern(bn)));
                                let api = n.len();
                                n.push(Node::App(vec![fi, p1.root, p2.root + off2, p3.root + off3]));
                                let e = SynthPool { nodes: n, root: api, ret_type: comp.ret_type, priority: comp.priority };
                                test_and_score(&e, &mut seen, &mut explored);
                                if explored > max_candidates { break 'a3c2; }
                                if pool.len() + new_entries.len() < MAX_POOL { new_entries.push(e); }
                            }
                        }
                    }
                }
                // Case 3: arg1+arg2 from old, arg3 from prev
                if explored <= max_candidates {
                    'a3c3: for a1 in 0..prev_start {
                        let p1 = &pool[a1];
                        if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 && p1.ret_type != 255 { continue; }
                        for a2 in 0..prev_start {
                            let p2 = &pool[a2];
                            if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 && p2.ret_type != 255 { continue; }
                            for pi in prev.clone() {
                                let p3 = &pool[pi];
                                if p3.ret_type != comp.param_types[2] && comp.param_types[2] != 255 && p3.ret_type != 255 { continue; }
                                let mut n = p1.nodes.clone();
                                let off2 = n.len();
                                for nd in &p2.nodes { n.push(remap_node(nd, off2)); }
                                let off3 = n.len();
                                for nd in &p3.nodes { n.push(remap_node(nd, off3)); }
                                let fi = n.len();
                                n.push(Node::Symbol(intern(bn)));
                                let api = n.len();
                                n.push(Node::App(vec![fi, p1.root, p2.root + off2, p3.root + off3]));
                                let e = SynthPool { nodes: n, root: api, ret_type: comp.ret_type, priority: comp.priority };
                                test_and_score(&e, &mut seen, &mut explored);
                                if explored > max_candidates { break 'a3c3; }
                                if pool.len() + new_entries.len() < MAX_POOL { new_entries.push(e); }
                            }
                        }
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

    #[test]
    fn test_vm_string_reverse() {
        use crate::types::Node;
        use crate::eval::BUILTIN_NAMES;
        use crate::intern::intern;
        let nodes = vec![
            Node::Symbol(intern("x")),
            Node::Symbol(intern("string-reverse")),
            Node::App(vec![1, 0]),
        ];
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = vec![];
        let ctx = crate::vm::CompileCtx::new(BUILTIN_NAMES, &macros);
        let chunk = crate::vm::compile(&nodes, 2, &ctx).expect("should compile");
        let macro_chunks = crate::vm::compile_macros(&macros, &ctx);
        let mut stack = Vec::new();
        let result = crate::vm::execute(&chunk, &Value::Str("hello".into()), &macro_chunks, &mut stack);
        eprintln!("VM string-reverse result: {:?}", result);
        assert!(result.is_ok(), "VM should succeed");
        assert!(vals_equal(&result.unwrap(), &Value::Str("olleh".into())), "should produce olleh");
    }

    #[test]
    fn test_synth_finds_string_reverse() {
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = vec![];
        let comps = vec![
            SynthComponent { name: "x".into(), builtin: None, arity: 0, ret_type: TYPE_STR, param_types: vec![], priority: 100.0, usage_count: 0.0 },
            SynthComponent { name: "string-reverse".into(), builtin: Some("string-reverse".into()), arity: 1, ret_type: TYPE_STR, param_types: vec![TYPE_STR], priority: 0.0, usage_count: 0.0 },
        ];
        let inputs = vec![Value::Str("hello".into()), Value::Str("ab".into())];
        let expected = vec![Value::Str("olleh".into()), Value::Str("ba".into())];
        let sr = synthesize_with_validation(&comps, &inputs, &expected, &macros, 1, 100, false, None, &[]);
        eprintln!("synth found={}, cand={}", sr.found, sr.candidates_explored);
        if sr.found {
            let src = crate::node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap());
            eprintln!("synth solution: {}", src);
        }
        assert!(sr.found);
    }
}
