//! Bottom-up enumerative program synthesizer for SELPH.
//!
//! Given input/output examples, searches for the smallest program that
//! satisfies them. Uses type pruning, observational equivalence dedup,
//! and (optionally) hash-based if-expression generation.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use crate::types::*;
use crate::eval;

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

/// Build macro env from macro tuples for eval.
fn build_macro_env(
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> Vec<(String, Vec<String>, Vec<Node>, usize)> {
    macros.to_vec()
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
    const MAX_POOL: usize = 100_000;
    let mut explored: usize = 0;
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let macro_env = build_macro_env(macros);

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
        pool.push(SynthPool {
            nodes,
            root: 0,
            ret_type: comp.ret_type,
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

    let mut prev_start: usize = 0;
    let mut prev_end: usize = pool.len();

    for _depth in 1..=max_depth {
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
            }
        }

        // ── If-expression generation (when enabled) ─────────────────

        if enable_if && !inputs.is_empty() {
            let if_entries = generate_if_programs(
                &pool, 0..pool.len() + new_entries.len(),
                inputs, expected, &macro_env,
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
        SynthComponent { name: "5".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
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

    // Add macro components
    for (name, params, _, _) in macros {
        comps.push(SynthComponent {
            name: name.clone(),
            builtin: Some(name.clone()),
            arity: params.len(),
            ret_type: 0,
            param_types: vec![0; params.len()],
            priority: 30.0,
        });
    }

    comps
}
