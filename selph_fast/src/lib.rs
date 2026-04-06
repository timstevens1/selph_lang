use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString, PyTuple};
use std::collections::HashMap;

/// SELPH value representation in Rust.
#[derive(Clone, Debug)]
enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
    List(Vec<Value>),
    Nil,
    /// A closure: (param_names, body_index, captured_env)
    Closure(Vec<String>, usize, Env),
    /// A builtin function identified by name
    Builtin(String),
    /// A macro: (param_names, body_nodes, body_root)
    /// Macros carry their own node pool since they were compiled separately.
    RustMacro(Vec<String>, Vec<Node>, usize),
}

/// AST node representation — a flattened program.
#[derive(Clone, Debug)]
enum Node {
    Num(f64),
    Str(String),
    Bool(bool),
    Symbol(String),
    /// Application: (fn_node, arg_nodes...)
    App(Vec<usize>),
    /// If: (cond, then, else)
    If(usize, usize, usize),
    /// Lambda: (params, body)
    Lambda(Vec<String>, usize),
    /// Let: (bindings, body) where bindings = [(name, val_node), ...]
    Let(Vec<(String, usize)>, usize),
}

/// Environment: stack of scopes.
type Env = Vec<HashMap<String, Value>>;

fn env_lookup(env: &Env, name: &str) -> Option<Value> {
    for scope in env.iter().rev() {
        if let Some(v) = scope.get(name) {
            return Some(v.clone());
        }
    }
    None
}

fn env_define(env: &mut Env, name: String, val: Value) {
    if let Some(scope) = env.last_mut() {
        scope.insert(name, val);
    }
}

/// Evaluate a node in the program.
fn eval(nodes: &[Node], idx: usize, env: &mut Env) -> Result<Value, String> {
    match &nodes[idx] {
        Node::Num(n) => Ok(Value::Num(*n)),
        Node::Str(s) => Ok(Value::Str(s.clone())),
        Node::Bool(b) => Ok(Value::Bool(*b)),
        Node::Symbol(name) => {
            env_lookup(env, name).ok_or_else(|| format!("unbound: {}", name))
        }
        Node::Lambda(params, body) => {
            Ok(Value::Closure(params.clone(), *body, env.clone()))
        }
        Node::If(cond, then_br, else_br) => {
            let cond_val = eval(nodes, *cond, env)?;
            match cond_val {
                Value::Bool(true) => eval(nodes, *then_br, env),
                Value::Bool(false) => eval(nodes, *else_br, env),
                Value::Num(n) => {
                    if n != 0.0 {
                        eval(nodes, *then_br, env)
                    } else {
                        eval(nodes, *else_br, env)
                    }
                }
                _ => Err("if: condition must be bool or number".to_string()),
            }
        }
        Node::Let(bindings, body) => {
            env.push(HashMap::new());
            for (name, val_idx) in bindings {
                let val = eval(nodes, *val_idx, env)?;
                env_define(env, name.clone(), val);
            }
            let result = eval(nodes, *body, env);
            env.pop();
            result
        }
        Node::App(children) => {
            if children.is_empty() {
                return Ok(Value::List(vec![]));
            }
            let fn_val = eval(nodes, children[0], env)?;
            let mut args = Vec::new();
            for &arg_idx in &children[1..] {
                args.push(eval(nodes, arg_idx, env)?);
            }
            apply(&fn_val, &args, nodes, env)
        }
    }
}

/// Apply a function value to arguments.
fn apply(
    fn_val: &Value,
    args: &[Value],
    nodes: &[Node],
    env: &mut Env,
) -> Result<Value, String> {
    match fn_val {
        Value::Closure(params, body, closed_env) => {
            let mut new_env = closed_env.clone();
            let mut scope = HashMap::new();
            for (i, param) in params.iter().enumerate() {
                if i < args.len() {
                    scope.insert(param.clone(), args[i].clone());
                }
            }
            new_env.push(scope);
            eval(nodes, *body, &mut new_env)
        }
        Value::Builtin(name) => apply_builtin(name, args),
        Value::RustMacro(params, macro_nodes, body_root) => {
            // Macros carry their own node pool. Evaluate the body with
            // params bound to the provided args.
            let mut macro_env = make_default_env();
            // Also copy current env's user-defined bindings
            for scope in env.iter() {
                for (k, v) in scope {
                    env_define(&mut macro_env, k.clone(), v.clone());
                }
            }
            let mut scope = HashMap::new();
            for (i, param) in params.iter().enumerate() {
                if i < args.len() {
                    scope.insert(param.clone(), args[i].clone());
                }
            }
            macro_env.push(scope);
            eval(macro_nodes, *body_root, &mut macro_env)
        }
        _ => Err(format!("not callable: {:?}", fn_val)),
    }
}

/// Built-in function dispatch.
fn apply_builtin(name: &str, args: &[Value]) -> Result<Value, String> {
    match name {
        // Arithmetic
        "add" | "+" => num_binop(args, |a, b| a + b),
        "subtract" | "-" => num_binop(args, |a, b| a - b),
        "multiply" | "*" => num_binop(args, |a, b| a * b),
        "divide" | "/" => {
            let (a, b) = get_nums(args)?;
            if b == 0.0 {
                Err("division by zero".to_string())
            } else {
                Ok(Value::Num(a / b))
            }
        }
        "modulo" | "%" => {
            let (a, b) = get_nums(args)?;
            Ok(Value::Num(a % b))
        }
        "abs" => Ok(Value::Num(get_num(&args[0])?.abs())),
        "negate" => Ok(Value::Num(-get_num(&args[0])?)),
        "min" => num_binop(args, |a, b| a.min(b)),
        "max" => num_binop(args, |a, b| a.max(b)),
        "floor" => Ok(Value::Num(get_num(&args[0])?.floor())),
        "ceil" => Ok(Value::Num(get_num(&args[0])?.ceil())),

        // Comparison
        "<" => Ok(Value::Bool(get_nums(args)?.0 < get_nums(args)?.1)),
        ">" => Ok(Value::Bool(get_nums(args)?.0 > get_nums(args)?.1)),
        "<=" => Ok(Value::Bool(get_nums(args)?.0 <= get_nums(args)?.1)),
        ">=" => Ok(Value::Bool(get_nums(args)?.0 >= get_nums(args)?.1)),
        "=" => match (&args[0], &args[1]) {
            (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(a == b)),
            (Value::Str(a), Value::Str(b)) => Ok(Value::Bool(a == b)),
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a == b)),
            _ => Ok(Value::Bool(false)),
        },
        "!=" => match (&args[0], &args[1]) {
            (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(a != b)),
            (Value::Str(a), Value::Str(b)) => Ok(Value::Bool(a != b)),
            _ => Ok(Value::Bool(true)),
        },

        // Logic
        "not" => match &args[0] {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            _ => Err("not: expected bool".to_string()),
        },
        "even" => Ok(Value::Bool(get_num(&args[0])? % 2.0 == 0.0)),
        "odd" => Ok(Value::Bool(get_num(&args[0])? % 2.0 != 0.0)),

        // String ops
        "string-upper" => Ok(Value::Str(get_str(&args[0])?.to_uppercase())),
        "string-lower" => Ok(Value::Str(get_str(&args[0])?.to_lowercase())),
        "string-reverse" => Ok(Value::Str(get_str(&args[0])?.chars().rev().collect())),
        "string-trim" => Ok(Value::Str(get_str(&args[0])?.trim().to_string())),
        "string-length" => Ok(Value::Num(get_str(&args[0])?.len() as f64)),
        "string-contains" => {
            let s = get_str(&args[0])?;
            let sub = get_str(&args[1])?;
            Ok(Value::Bool(s.contains(&sub)))
        }
        "string-split" => {
            let s = get_str(&args[0])?;
            let sep = get_str(&args[1])?;
            let parts: Vec<Value> = s.split(&sep).map(|p| Value::Str(p.to_string())).collect();
            Ok(Value::List(parts))
        }
        "string-join" => {
            let lst = get_list(&args[0])?;
            let sep = get_str(&args[1])?;
            let strs: Vec<String> = lst
                .iter()
                .map(|v| match v {
                    Value::Str(s) => s.clone(),
                    Value::Num(n) => {
                        if *n == (*n as i64) as f64 {
                            format!("{}", *n as i64)
                        } else {
                            format!("{}", n)
                        }
                    }
                    _ => format!("{:?}", v),
                })
                .collect();
            Ok(Value::Str(strs.join(&sep)))
        }
        "concat" => {
            let mut result = String::new();
            for arg in args {
                match arg {
                    Value::Str(s) => result.push_str(s),
                    Value::Num(n) => result.push_str(&format!("{}", n)),
                    _ => result.push_str(&format!("{:?}", arg)),
                }
            }
            Ok(Value::Str(result))
        }
        "to-string" => match &args[0] {
            Value::Num(n) => {
                if *n == (*n as i64) as f64 {
                    Ok(Value::Str(format!("{}", *n as i64)))
                } else {
                    Ok(Value::Str(format!("{}", n)))
                }
            }
            Value::Str(s) => Ok(Value::Str(s.clone())),
            _ => Ok(Value::Str(format!("{:?}", args[0]))),
        },
        "to-number" => match &args[0] {
            Value::Str(s) => s
                .parse::<f64>()
                .map(Value::Num)
                .map_err(|e| format!("to-number: {}", e)),
            Value::Num(n) => Ok(Value::Num(*n)),
            _ => Err("to-number: expected string or number".to_string()),
        },

        // List ops
        "list" => Ok(Value::List(args.to_vec())),
        "head" => {
            let lst = get_list(&args[0])?;
            lst.first()
                .cloned()
                .ok_or_else(|| "head of empty list".to_string())
        }
        "tail" => {
            let lst = get_list(&args[0])?;
            if lst.is_empty() {
                Err("tail of empty list".to_string())
            } else {
                Ok(Value::List(lst[1..].to_vec()))
            }
        }
        "length" => Ok(Value::Num(get_list(&args[0])?.len() as f64)),
        "cons" => {
            let mut lst = get_list(&args[1])?;
            lst.insert(0, args[0].clone());
            Ok(Value::List(lst))
        }
        "identity" => Ok(args[0].clone()),

        _ => Err(format!("unknown builtin: {}", name)),
    }
}

fn get_num(v: &Value) -> Result<f64, String> {
    match v {
        Value::Num(n) => Ok(*n),
        _ => Err(format!("expected number, got {:?}", v)),
    }
}

fn get_nums(args: &[Value]) -> Result<(f64, f64), String> {
    Ok((get_num(&args[0])?, get_num(&args[1])?))
}

fn get_str(v: &Value) -> Result<String, String> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        _ => Err(format!("expected string, got {:?}", v)),
    }
}

fn get_list(v: &Value) -> Result<Vec<Value>, String> {
    match v {
        Value::List(lst) => Ok(lst.clone()),
        _ => Err(format!("expected list, got {:?}", v)),
    }
}

fn num_binop(args: &[Value], op: fn(f64, f64) -> f64) -> Result<Value, String> {
    let (a, b) = get_nums(args)?;
    Ok(Value::Num(op(a, b)))
}

/// Make an env with default builtins plus user-defined macros.
fn make_env_with_macros(macro_defs: &[(String, Vec<String>, Vec<Node>, usize)]) -> Env {
    let mut env = make_default_env();
    for (name, params, body_nodes, body_root) in macro_defs {
        env_define(&mut env, name.clone(),
            Value::RustMacro(params.clone(), body_nodes.clone(), *body_root));
    }
    env
}

// ── Heuristic synthesis ─────────────────────────────────────────────

/// Evaluate a SELPH heuristic program on a component.
/// The heuristic is a compiled SELPH lambda. We bind component metadata
/// as variables (comp-name, comp-arity, comp-ret-type) and evaluate.
fn eval_selph_heuristic(
    heuristic_nodes: &[Node],
    heuristic_root: usize,
    comp: &RustComponent,
    macro_defs: &[(String, Vec<String>, Vec<Node>, usize)],
) -> f64 {
    let mut env = make_env_with_macros(macro_defs);
    // Bind component metadata as variables
    env_define(&mut env, "comp-name".to_string(), Value::Str(comp.name.clone()));
    env_define(&mut env, "comp-arity".to_string(), Value::Num(comp.arity as f64));
    env_define(&mut env, "comp-ret-type".to_string(), Value::Num(comp.ret_type as f64));
    env_define(&mut env, "comp-param-count".to_string(),
        Value::Num(comp.param_types.len() as f64));

    match eval(heuristic_nodes, heuristic_root, &mut env) {
        Ok(Value::Num(n)) => n,
        _ => 0.0,
    }
}

/// Synthesize a heuristic by generating SELPH programs and scoring each
/// by running synthesis on the task suite.
///
/// The heuristic components are operations on comp-name, comp-arity,
/// comp-ret-type — accessed as SELPH variables.
fn synthesize_selph_heuristic(
    base_components: &[RustComponent],
    tasks: &[(Vec<Value>, Vec<Value>)],
    macro_defs: &[(String, Vec<String>, Vec<Node>, usize)],
    synth_budget: usize,
    synth_depth: usize,
) -> (Vec<Node>, usize, f64, usize) {
    // Build heuristic components — operations on comp metadata
    let mut h_comps: Vec<RustComponent> = vec![
        // Constants
        RustComponent { name: "0".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        RustComponent { name: "10".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        RustComponent { name: "50".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        RustComponent { name: "100".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        // Component metadata accessors
        RustComponent { name: "comp-arity".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        RustComponent { name: "comp-ret-type".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        // Arithmetic for combining scores
        RustComponent { name: "add".into(), builtin: Some("add".into()), arity: 2, ret_type: 0, param_types: vec![0, 0], priority: 0.0 },
        RustComponent { name: "multiply".into(), builtin: Some("multiply".into()), arity: 2, ret_type: 0, param_types: vec![0, 0], priority: 0.0 },
        // String ops for name matching
        RustComponent { name: "comp-name".into(), builtin: None, arity: 0, ret_type: 1, param_types: vec![], priority: 0.0 },
        RustComponent { name: "string-length".into(), builtin: Some("string-length".into()), arity: 1, ret_type: 0, param_types: vec![1], priority: 0.0 },
    ];

    // Add string-contains for matching component names against known substrings
    let mut known_substrings: Vec<String> = Vec::new();
    for comp in base_components {
        known_substrings.push(comp.name.clone());
        for part in comp.name.split('-') {
            if part.len() >= 2 && !known_substrings.contains(&part.to_string()) {
                known_substrings.push(part.to_string());
            }
        }
    }
    // Add each substring as a constant
    for s in &known_substrings {
        h_comps.push(RustComponent {
            name: format!("\"{}\"", s),
            builtin: None,
            arity: 0,
            ret_type: 1, // string
            param_types: vec![],
            priority: 0.0,
        });
    }
    // string-contains: (string, string) -> bool(as num: 0/1)
    h_comps.push(RustComponent {
        name: "string-contains".into(),
        builtin: Some("string-contains".into()),
        arity: 2,
        ret_type: 0, // returns 0.0 or 1.0 (we'll treat bool as num)
        param_types: vec![1, 1],
        priority: 0.0,
    });

    // Generate candidate heuristic programs (depth 0-2)
    // and score each by running synthesis
    let empty_inputs: Vec<Value> = Vec::new();
    let empty_expected: Vec<Value> = Vec::new();

    // Build program pool for heuristics
    let mut all_pool: Vec<PoolEntry> = Vec::new();
    let mut prev_layer: Vec<usize> = Vec::new();

    for comp in &h_comps {
        if comp.arity != 0 { continue; }
        let mut nodes = Vec::new();
        if comp.name.starts_with('"') && comp.name.ends_with('"') {
            nodes.push(Node::Str(comp.name[1..comp.name.len()-1].to_string()));
        } else if let Ok(n) = comp.name.parse::<f64>() {
            nodes.push(Node::Num(n));
        } else {
            nodes.push(Node::Symbol(comp.name.clone()));
        }
        let idx = all_pool.len();
        all_pool.push(PoolEntry { nodes, root: 0, ret_type: comp.ret_type, max_priority: comp.priority });
        prev_layer.push(idx);
    }

    // Expand to depth 1-2
    for depth in 1..=2 {
        let new_entries = expand_layer_rust(&prev_layer, &all_pool, &h_comps);
        let start = all_pool.len();
        all_pool.extend(new_entries);
        prev_layer = (start..all_pool.len()).collect();
    }

    // Score each candidate heuristic
    let mut best_score = usize::MAX;
    let mut best_nodes: Vec<Node> = vec![Node::Num(0.0)];
    let mut best_root: usize = 0;
    let mut tested = 0usize;

    let all_entries: Vec<usize> = (0..all_pool.len()).collect();

    for &idx in &all_entries {
        let entry = &all_pool[idx];
        // Only consider number-returning programs as heuristics
        if entry.ret_type != 0 { continue; }

        tested += 1;

        // Apply this heuristic to set priorities on base_components
        let mut comps = base_components.to_vec();
        for comp in &mut comps {
            comp.priority = eval_selph_heuristic(
                &entry.nodes, entry.root, comp, macro_defs);
        }
        comps.sort_by(|a, b| b.priority.partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal));

        // Score: total candidates across all tasks
        let mut total = 0usize;
        for (inputs, expected) in tasks {
            let (result, explored) = rust_synthesize(
                &comps, inputs, expected, synth_depth, synth_budget, macro_defs);
            total += if result.is_some() { explored } else { synth_budget };
            if total >= best_score { break; }
        }

        if total < best_score {
            best_score = total;
            best_nodes = entry.nodes.clone();
            best_root = entry.root;
        }
    }

    (best_nodes, best_root, best_score as f64, tested)
}

// ── Rust-native synthesizer ──────────────────────────────────────────

/// A component available for synthesis (Rust-native).
#[derive(Clone, Debug)]
struct RustComponent {
    name: String,
    /// For arity-0: a single Node index in a shared node pool.
    /// For arity-1+: a builtin name to apply.
    builtin: Option<String>,
    arity: usize,
    /// Type: 0=num, 1=str, 2=bool
    ret_type: u8,
    param_types: Vec<u8>,
    priority: f64,
}

/// A generated program in the synthesis pool.
#[derive(Clone)]
struct PoolEntry {
    /// The node tree for this program (flattened)
    nodes: Vec<Node>,
    root: usize,
    ret_type: u8,
    /// Highest priority among components used in this program.
    /// Used to sort arguments so high-priority programs are tried first.
    max_priority: f64,
}

/// Run bottom-up synthesis entirely in Rust.
/// Returns (program_source, candidates_explored) or None.
fn rust_synthesize(
    components: &[RustComponent],
    inputs: &[Value],
    expected: &[Value],
    max_depth: usize,
    max_candidates: usize,
    macro_defs: &[(String, Vec<String>, Vec<Node>, usize)],
) -> (Option<(Vec<Node>, usize)>, usize) {
    let total_examples = inputs.len();
    let mut candidates_explored: usize = 0;

    // Behavior fingerprint for dedup
    let mut seen_behaviors: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();

    // Program pool: (nodes, root, ret_type)
    let mut all_pool: Vec<PoolEntry> = Vec::new();
    let mut prev_layer: Vec<usize> = Vec::new(); // indices into all_pool

    // Depth 0: constants and atoms
    for comp in components {
        if comp.arity != 0 {
            continue;
        }
        let mut nodes = Vec::new();
        let root = match comp.name.as_str() {
            "x" => { nodes.push(Node::Symbol("x".to_string())); 0 }
            _ => {
                if let Ok(n) = comp.name.parse::<f64>() {
                    nodes.push(Node::Num(n));
                } else {
                    nodes.push(Node::Str(comp.name.clone()));
                }
                0
            }
        };
        let idx = all_pool.len();
        all_pool.push(PoolEntry { nodes, root, ret_type: comp.ret_type, max_priority: comp.priority });
        prev_layer.push(idx);
    }

    // Test depth-0 and expand
    for depth in 0..=max_depth {
        let entries_to_test = if depth == 0 {
            prev_layer.clone()
        } else {
            // Generate new layer
            let new_entries = expand_layer_rust(
                &prev_layer, &all_pool, components);
            let start = all_pool.len();
            all_pool.extend(new_entries);
            (start..all_pool.len()).collect()
        };

        // Test each entry
        let mut kept = Vec::new();
        for &idx in &entries_to_test {
            candidates_explored += 1;
            if candidates_explored > max_candidates {
                return (None, candidates_explored);
            }

            let entry = &all_pool[idx];

            // Wrap in lambda and evaluate on all inputs
            let mut lambda_nodes = entry.nodes.clone();
            let param_idx = lambda_nodes.len();
            lambda_nodes.push(Node::Symbol("x".to_string()));
            let lambda_root = lambda_nodes.len();
            lambda_nodes.push(Node::Lambda(vec!["x".to_string()], entry.root));

            // Compute behavior fingerprint for dedup
            let mut behavior: Vec<u64> = Vec::new();
            let mut all_match = true;

            for (inp, exp) in inputs.iter().zip(expected.iter()) {
                let mut env = make_env_with_macros(macro_defs);
                let fn_val = match eval(&lambda_nodes, lambda_root, &mut env) {
                    Ok(v) => v,
                    Err(_) => { all_match = false; break; }
                };
                match apply(&fn_val, &[inp.clone()], &lambda_nodes, &mut env) {
                    Ok(actual) => {
                        behavior.push(value_hash(&actual));
                        if !values_equal(&actual, exp) {
                            all_match = false;
                        }
                    }
                    Err(_) => { all_match = false; break; }
                }
            }

            // Dedup
            if !behavior.is_empty() {
                if seen_behaviors.contains(&behavior) {
                    continue;
                }
                seen_behaviors.insert(behavior);
            }

            if all_match && !inputs.is_empty() {
                // Found a solution!
                return (Some((lambda_nodes, lambda_root)), candidates_explored);
            }

            kept.push(idx);
        }

        if depth > 0 {
            prev_layer = kept;
        }
    }

    (None, candidates_explored)
}

fn expand_layer_rust(
    prev_layer: &[usize],
    all_pool: &[PoolEntry],
    components: &[RustComponent],
) -> Vec<PoolEntry> {
    let mut new_entries = Vec::new();
    // Sort argument indices by max_priority descending so high-priority
    // programs are tried first as arguments
    let mut all_indices: Vec<usize> = (0..all_pool.len()).collect();
    all_indices.sort_by(|&a, &b| all_pool[b].max_priority.partial_cmp(&all_pool[a].max_priority)
        .unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_prev: Vec<usize> = prev_layer.to_vec();
    sorted_prev.sort_by(|&a, &b| all_pool[b].max_priority.partial_cmp(&all_pool[a].max_priority)
        .unwrap_or(std::cmp::Ordering::Equal));

    for comp in components {
        if comp.arity == 0 || comp.builtin.is_none() {
            continue;
        }
        let builtin_name = comp.builtin.as_ref().unwrap();

        if comp.arity == 1 {
            for &prev_idx in &sorted_prev {
                let prev = &all_pool[prev_idx];
                if !type_compatible(prev.ret_type, comp.param_types[0]) {
                    continue;
                }
                // Build (builtin prev_body)
                let mut nodes = prev.nodes.clone();
                let fn_idx = nodes.len();
                nodes.push(Node::Symbol(builtin_name.clone()));
                let app_idx = nodes.len();
                nodes.push(Node::App(vec![fn_idx, prev.root]));
                new_entries.push(PoolEntry {
                    nodes,
                    root: app_idx,
                    ret_type: comp.ret_type,
                    max_priority: comp.priority,
                });
            }
        } else if comp.arity == 2 {
            // arg1 from prev_layer, arg2 from all
            for &p1_idx in &sorted_prev {
                let p1 = &all_pool[p1_idx];
                if !type_compatible(p1.ret_type, comp.param_types[0]) {
                    continue;
                }
                for &p2_idx in &all_indices {
                    let p2 = &all_pool[p2_idx];
                    if !type_compatible(p2.ret_type, comp.param_types[1]) {
                        continue;
                    }
                    // Merge both node trees
                    let mut nodes = p1.nodes.clone();
                    let offset = nodes.len();
                    // Remap p2's indices
                    for node in &p2.nodes {
                        nodes.push(remap_node(node, offset));
                    }
                    let fn_idx = nodes.len();
                    nodes.push(Node::Symbol(builtin_name.clone()));
                    let app_idx = nodes.len();
                    nodes.push(Node::App(vec![fn_idx, p1.root, p2.root + offset]));
                    new_entries.push(PoolEntry {
                        nodes,
                        root: app_idx,
                        ret_type: comp.ret_type,
                        max_priority: comp.priority,
                    });
                }
            }
            // arg1 from old (not in prev), arg2 from prev
            let prev_set: std::collections::HashSet<usize> = prev_layer.iter().copied().collect();
            for &p1_idx in &all_indices {
                if prev_set.contains(&p1_idx) { continue; }
                let p1 = &all_pool[p1_idx];
                if !type_compatible(p1.ret_type, comp.param_types[0]) {
                    continue;
                }
                for &p2_idx in &sorted_prev {
                    let p2 = &all_pool[p2_idx];
                    if !type_compatible(p2.ret_type, comp.param_types[1]) {
                        continue;
                    }
                    let mut nodes = p1.nodes.clone();
                    let offset = nodes.len();
                    for node in &p2.nodes {
                        nodes.push(remap_node(node, offset));
                    }
                    let fn_idx = nodes.len();
                    nodes.push(Node::Symbol(builtin_name.clone()));
                    let app_idx = nodes.len();
                    nodes.push(Node::App(vec![fn_idx, p1.root, p2.root + offset]));
                    new_entries.push(PoolEntry {
                        nodes,
                        root: app_idx,
                        ret_type: comp.ret_type,
                        max_priority: comp.priority,
                    });
                }
            }
        }
    }

    new_entries
}

fn remap_node(node: &Node, offset: usize) -> Node {
    match node {
        Node::App(children) => Node::App(children.iter().map(|c| c + offset).collect()),
        Node::If(c, t, e) => Node::If(c + offset, t + offset, e + offset),
        Node::Lambda(params, body) => Node::Lambda(params.clone(), body + offset),
        Node::Let(bindings, body) => {
            let new_bindings = bindings.iter()
                .map(|(name, idx)| (name.clone(), idx + offset))
                .collect();
            Node::Let(new_bindings, body + offset)
        }
        other => other.clone(),
    }
}

fn type_compatible(actual: u8, expected: u8) -> bool {
    actual == expected || expected == 255 // 255 = any
}

fn value_hash(v: &Value) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match v {
        Value::Num(n) => { 0u8.hash(&mut hasher); n.to_bits().hash(&mut hasher); }
        Value::Str(s) => { 1u8.hash(&mut hasher); s.hash(&mut hasher); }
        Value::Bool(b) => { 2u8.hash(&mut hasher); b.hash(&mut hasher); }
        Value::Nil => { 3u8.hash(&mut hasher); }
        _ => { 4u8.hash(&mut hasher); }
    }
    hasher.finish()
}

fn node_to_source(nodes: &[Node], idx: usize) -> String {
    match &nodes[idx] {
        Node::Num(n) => {
            if *n == (*n as i64) as f64 { format!("{}", *n as i64) }
            else { format!("{}", n) }
        }
        Node::Str(s) => format!("\"{}\"", s),
        Node::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
        Node::Symbol(name) => name.clone(),
        Node::App(children) => {
            if children.is_empty() { return "()".to_string(); }
            let parts: Vec<String> = children.iter().map(|c| node_to_source(nodes, *c)).collect();
            format!("({})", parts.join(" "))
        }
        Node::If(c, t, e) => format!("(if {} {} {})",
            node_to_source(nodes, *c),
            node_to_source(nodes, *t),
            node_to_source(nodes, *e)),
        Node::Lambda(params, body) => format!("(lambda ({}) {})",
            params.join(" "),
            node_to_source(nodes, *body)),
        Node::Let(bindings, body) => {
            let bs: Vec<String> = bindings.iter()
                .map(|(name, idx)| format!("({} {})", name, node_to_source(nodes, *idx)))
                .collect();
            format!("(let ({}) {})", bs.join(" "), node_to_source(nodes, *body))
        }
    }
}

// ── Interleaved search ──────────────────────────────────────────────

/// Extract component names that appear in a solution's AST.
fn extract_used_components(nodes: &[Node], root: usize) -> Vec<String> {
    let mut used = Vec::new();
    let mut visited = std::collections::HashSet::new();
    fn walk(nodes: &[Node], idx: usize, used: &mut Vec<String>,
            visited: &mut std::collections::HashSet<usize>) {
        if !visited.insert(idx) { return; }
        if idx >= nodes.len() { return; }
        match &nodes[idx] {
            Node::Symbol(name) => {
                if !used.contains(name) {
                    used.push(name.clone());
                }
            }
            Node::App(children) => {
                for &c in children { walk(nodes, c, used, visited); }
            }
            Node::If(c, t, e) => {
                walk(nodes, *c, used, visited);
                walk(nodes, *t, used, visited);
                walk(nodes, *e, used, visited);
            }
            Node::Lambda(_, body) => { walk(nodes, *body, used, visited); }
            Node::Let(bindings, body) => {
                for (_, v) in bindings { walk(nodes, *v, used, visited); }
                walk(nodes, *body, used, visited);
            }
            _ => {}
        }
    }
    walk(nodes, root, &mut used, &mut visited);
    used
}

/// Interleaved curriculum: solve tasks one by one, updating component
/// priorities after each success based on which components were used.
///
/// Priority update rule:
///   - Components appearing in a solution get boosted by `learn_rate`
///   - Components that were tried (in the component list) but didn't
///     appear in the solution are unchanged
///   - Priorities accumulate across tasks
///
/// This is online learning of the search strategy during the curriculum.
fn interleaved_synthesize(
    base_components: &[RustComponent],
    tasks: &[(Vec<Value>, Vec<Value>)],
    macro_defs: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    learn_rate: f64,
) -> (Vec<(bool, String, usize)>, HashMap<String, f64>) {
    let mut priorities: HashMap<String, f64> = HashMap::new();
    for comp in base_components {
        priorities.insert(comp.name.clone(), comp.priority);
    }

    // Also track: how many times each component appeared in a solution
    let mut solution_counts: HashMap<String, usize> = HashMap::new();

    let mut results = Vec::new();

    for (task_idx, (inputs, expected)) in tasks.iter().enumerate() {
        // Apply current priorities
        let mut comps = base_components.to_vec();
        for comp in &mut comps {
            if let Some(&p) = priorities.get(&comp.name) {
                comp.priority = p;
            }
        }
        comps.sort_by(|a, b| b.priority.partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal));

        // Solve
        let (result, explored) = rust_synthesize(
            &comps, inputs, expected, max_depth, max_candidates, macro_defs);

        match result {
            Some((nodes, root)) => {
                let source = node_to_source(&nodes, root);

                // Extract which components were used in the solution
                let used = extract_used_components(&nodes, root);

                // Update priorities: boost used components
                for name in &used {
                    let entry = priorities.entry(name.clone()).or_insert(0.0);
                    *entry += learn_rate;
                    let count = solution_counts.entry(name.clone()).or_insert(0);
                    *count += 1;
                }

                // Also promote the solution as a new macro for future tasks
                // Extract the lambda body (skip the lambda wrapper)
                if let Node::Lambda(params, body_idx) = &nodes[root] {
                    let macro_name = format!("_solved_{}", task_idx);
                    // We can't easily add macros mid-run without mutability issues,
                    // but we boost the components that made it work, which has the
                    // same effect on search ordering.
                }

                results.push((true, source, explored));
            }
            None => {
                results.push((false, String::new(), explored));
            }
        }
    }

    (results, priorities)
}

// ── Python interface ────────────────────────────────────────────────

/// Convert a Python AST node to our Rust Node representation.
/// Flattens the tree into a Vec<Node> with index references.
fn py_to_nodes(py: Python, obj: &Bound<PyAny>, nodes: &mut Vec<Node>) -> PyResult<usize> {
    let type_name = obj.get_type().name()?.to_string();

    match type_name.as_str() {
        "Number" => {
            let val: f64 = obj.getattr("value")?.extract()?;
            let idx = nodes.len();
            nodes.push(Node::Num(val));
            Ok(idx)
        }
        "String" => {
            let val: String = obj.getattr("value")?.extract()?;
            let idx = nodes.len();
            nodes.push(Node::Str(val));
            Ok(idx)
        }
        "Bool" => {
            let val: bool = obj.getattr("value")?.extract()?;
            let idx = nodes.len();
            nodes.push(Node::Bool(val));
            Ok(idx)
        }
        "Symbol" => {
            let name: String = obj.getattr("name")?.extract()?;
            let idx = nodes.len();
            nodes.push(Node::Symbol(name));
            Ok(idx)
        }
        "List" => {
            let elements = obj.getattr("elements")?;
            let elems: Vec<Bound<PyAny>> = elements.extract()?;

            if elems.is_empty() {
                let idx = nodes.len();
                nodes.push(Node::App(vec![]));
                return Ok(idx);
            }

            // Check for special forms
            let first_type = elems[0].get_type().name()?.to_string();
            if first_type == "Symbol" {
                let name: String = elems[0].getattr("name")?.extract()?;

                match name.as_str() {
                    "if" if elems.len() == 4 => {
                        let cond = py_to_nodes(py, &elems[1], nodes)?;
                        let then_br = py_to_nodes(py, &elems[2], nodes)?;
                        let else_br = py_to_nodes(py, &elems[3], nodes)?;
                        let idx = nodes.len();
                        nodes.push(Node::If(cond, then_br, else_br));
                        return Ok(idx);
                    }
                    "lambda" if elems.len() == 3 => {
                        let params_obj = elems[1].getattr("elements")?;
                        let param_nodes: Vec<Bound<PyAny>> = params_obj.extract()?;
                        let params: Vec<String> = param_nodes
                            .iter()
                            .map(|p| p.getattr("name").unwrap().extract().unwrap())
                            .collect();
                        let body = py_to_nodes(py, &elems[2], nodes)?;
                        let idx = nodes.len();
                        nodes.push(Node::Lambda(params, body));
                        return Ok(idx);
                    }
                    "let" if elems.len() == 3 => {
                        let bindings_obj = elems[1].getattr("elements")?;
                        let binding_nodes: Vec<Bound<PyAny>> = bindings_obj.extract()?;
                        let mut bindings = Vec::new();
                        for b in &binding_nodes {
                            let b_elems_obj = b.getattr("elements")?;
                            let b_elems: Vec<Bound<PyAny>> = b_elems_obj.extract()?;
                            let name: String = b_elems[0].getattr("name")?.extract()?;
                            let val_idx = py_to_nodes(py, &b_elems[1], nodes)?;
                            bindings.push((name, val_idx));
                        }
                        let body = py_to_nodes(py, &elems[2], nodes)?;
                        let idx = nodes.len();
                        nodes.push(Node::Let(bindings, body));
                        return Ok(idx);
                    }
                    _ => {}
                }
            }

            // Generic application
            let mut children = Vec::new();
            for elem in &elems {
                children.push(py_to_nodes(py, elem, nodes)?);
            }
            let idx = nodes.len();
            nodes.push(Node::App(children));
            Ok(idx)
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported node type: {}",
            type_name
        ))),
    }
}

fn value_to_py(py: Python, val: &Value) -> PyResult<PyObject> {
    match val {
        Value::Num(n) => Ok((*n).into_pyobject(py)?.into_any().unbind()),
        Value::Str(s) => Ok(s.as_str().into_pyobject(py)?.into_any().unbind()),
        Value::Bool(b) => {
            // PyBool::new returns a Borrowed in PyO3 0.27, use to_object instead
            let py_bool = PyBool::new(py, *b);
            Ok(py_bool.to_owned().into_any().unbind())
        }
        Value::List(lst) => {
            let py_list = PyList::empty(py);
            for v in lst {
                py_list.append(value_to_py(py, v)?)?;
            }
            Ok(py_list.into_any().unbind())
        }
        Value::Nil => Ok(py.None()),
        _ => Ok(py.None()),
    }
}

fn py_to_value(py: Python, obj: &Bound<PyAny>) -> PyResult<Value> {
    if let Ok(n) = obj.extract::<f64>() {
        return Ok(Value::Num(n));
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::Str(s));
    }
    if let Ok(lst) = obj.downcast::<PyList>() {
        let mut vals = Vec::new();
        for item in lst.iter() {
            vals.push(py_to_value(py, &item)?);
        }
        return Ok(Value::List(vals));
    }
    if obj.is_none() {
        return Ok(Value::Nil);
    }
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "cannot convert to Value: {:?}",
        obj
    )))
}

/// Make default environment with builtins.
fn make_default_env() -> Env {
    let builtins = [
        "add", "+", "subtract", "-", "multiply", "*", "divide", "/",
        "modulo", "%", "abs", "negate", "min", "max", "floor", "ceil",
        "<", ">", "<=", ">=", "=", "!=",
        "not", "even", "odd",
        "string-upper", "string-lower", "string-reverse", "string-trim",
        "string-length", "string-contains", "string-split", "string-join",
        "concat", "to-string", "to-number",
        "list", "head", "tail", "length", "cons", "identity",
    ];
    let mut scope = HashMap::new();
    for name in builtins {
        scope.insert(name.to_string(), Value::Builtin(name.to_string()));
    }
    vec![scope]
}

// ── Python module ───────────────────────────────────────────────────

/// A compiled program — AST converted to Rust once, reusable.
#[pyclass]
struct CompiledProgram {
    nodes: Vec<Node>,
    root: usize,
}

#[pymodule]
mod selph_fast {
    use super::*;

    /// Compile a SELPH AST to a reusable Rust representation.
    #[pyfunction]
    fn compile(py: Python, ast_node: &Bound<PyAny>) -> PyResult<CompiledProgram> {
        let mut nodes = Vec::new();
        let root = py_to_nodes(py, ast_node, &mut nodes)?;
        Ok(CompiledProgram { nodes, root })
    }

    /// Evaluate a compiled program on a list of inputs.
    /// Much faster than batch_eval because it avoids re-parsing the AST.
    #[pyfunction]
    fn run_compiled(
        py: Python,
        program: &CompiledProgram,
        inputs: &Bound<PyList>,
    ) -> PyResult<PyObject> {
        let results = PyList::empty(py);
        for input_obj in inputs.iter() {
            let input_val = py_to_value(py, &input_obj)?;
            let mut env = make_default_env();
            let fn_val = match eval(&program.nodes, program.root, &mut env) {
                Ok(v) => v,
                Err(_) => {
                    results.append(py.None())?;
                    continue;
                }
            };
            match apply(&fn_val, &[input_val], &program.nodes, &mut env) {
                Ok(val) => results.append(value_to_py(py, &val)?)?,
                Err(_) => results.append(py.None())?,
            }
        }
        Ok(results.into_any().unbind())
    }

    /// Verify a compiled program against expected outputs.
    #[pyfunction]
    fn verify_compiled(
        py: Python,
        program: &CompiledProgram,
        inputs: &Bound<PyList>,
        expected: &Bound<PyList>,
    ) -> PyResult<(usize, usize)> {
        let total = inputs.len();
        let mut matching = 0;
        for (input_obj, expected_obj) in inputs.iter().zip(expected.iter()) {
            let input_val = py_to_value(py, &input_obj)?;
            let expected_val = py_to_value(py, &expected_obj)?;
            let mut env = make_default_env();
            let fn_val = match eval(&program.nodes, program.root, &mut env) {
                Ok(v) => v,
                Err(_) => continue,
            };
            match apply(&fn_val, &[input_val], &program.nodes, &mut env) {
                Ok(actual) => {
                    if values_equal(&actual, &expected_val) {
                        matching += 1;
                    }
                }
                Err(_) => {}
            }
        }
        Ok((matching, total))
    }

    /// Evaluate N candidate programs on the same inputs, return scores.
    /// candidates: list of compiled programs
    /// inputs: list of input values
    /// expected: list of expected output values
    /// Returns: list of (num_matching, total) for each candidate.
    /// This keeps the hot loop entirely in Rust.
    #[pyfunction]
    fn batch_score_candidates(
        py: Python,
        candidates: Vec<Bound<PyAny>>,
        inputs: &Bound<PyList>,
        expected: &Bound<PyList>,
    ) -> PyResult<Vec<(usize, usize)>> {
        // Pre-convert inputs and expected to Rust Values
        let rust_inputs: Vec<Value> = inputs
            .iter()
            .map(|obj| py_to_value(py, &obj).unwrap_or(Value::Nil))
            .collect();
        let rust_expected: Vec<Value> = expected
            .iter()
            .map(|obj| py_to_value(py, &obj).unwrap_or(Value::Nil))
            .collect();
        let total = rust_inputs.len();

        let mut results = Vec::new();

        for cand_obj in &candidates {
            // Convert AST to nodes
            let mut nodes = Vec::new();
            let root = match py_to_nodes(py, cand_obj, &mut nodes) {
                Ok(r) => r,
                Err(_) => {
                    results.push((0, total));
                    continue;
                }
            };

            let mut matching = 0;
            for (input_val, expected_val) in rust_inputs.iter().zip(rust_expected.iter()) {
                let mut env = make_default_env();
                let fn_val = match eval(&nodes, root, &mut env) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match apply(&fn_val, &[input_val.clone()], &nodes, &mut env) {
                    Ok(actual) => {
                        if values_equal(&actual, expected_val) {
                            matching += 1;
                        }
                    }
                    Err(_) => {}
                }
            }
            results.push((matching, total));
        }

        Ok(results)
    }

    /// Score many candidates at once, pre-compiling inputs.
    /// Programs: list of AST nodes (not pre-compiled)
    /// Returns match counts for each.
    /// All conversion + evaluation stays in Rust.
    #[pyfunction]
    fn score_many(
        py: Python,
        programs: &Bound<PyList>,
        inputs: &Bound<PyList>,
        expected: &Bound<PyList>,
    ) -> PyResult<Vec<usize>> {
        let rust_inputs: Vec<Value> = inputs
            .iter()
            .map(|obj| py_to_value(py, &obj).unwrap_or(Value::Nil))
            .collect();
        let rust_expected: Vec<Value> = expected
            .iter()
            .map(|obj| py_to_value(py, &obj).unwrap_or(Value::Nil))
            .collect();

        let mut results = Vec::with_capacity(programs.len());

        for prog_obj in programs.iter() {
            let mut nodes = Vec::new();
            let root = match py_to_nodes(py, &prog_obj, &mut nodes) {
                Ok(r) => r,
                Err(_) => {
                    results.push(0);
                    continue;
                }
            };

            let mut matching = 0usize;
            for (inp, exp) in rust_inputs.iter().zip(rust_expected.iter()) {
                let mut env = make_default_env();
                let fn_val = match eval(&nodes, root, &mut env) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match apply(&fn_val, &[inp.clone()], &nodes, &mut env) {
                    Ok(actual) => {
                        if values_equal(&actual, exp) {
                            matching += 1;
                        }
                    }
                    Err(_) => {}
                }
            }
            results.push(matching);
        }

        Ok(results)
    }

    /// Evaluate a SELPH AST node and return the result.
    #[pyfunction]
    fn fast_eval(py: Python, ast_node: &Bound<PyAny>) -> PyResult<PyObject> {
        let mut nodes = Vec::new();
        let root = py_to_nodes(py, ast_node, &mut nodes)?;
        let mut env = make_default_env();
        match eval(&nodes, root, &mut env) {
            Ok(val) => value_to_py(py, &val),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Evaluate a SELPH program on multiple inputs at once.
    /// program: a lambda AST node
    /// inputs: list of Python values
    /// Returns: list of results (or None for errors)
    #[pyfunction]
    fn batch_eval(
        py: Python,
        program: &Bound<PyAny>,
        inputs: &Bound<PyList>,
    ) -> PyResult<PyObject> {
        let mut nodes = Vec::new();
        let root = py_to_nodes(py, program, &mut nodes)?;

        let results = PyList::empty(py);
        for input_obj in inputs.iter() {
            let input_val = py_to_value(py, &input_obj)?;
            let mut env = make_default_env();

            // Evaluate the lambda to get a closure
            let fn_val = match eval(&nodes, root, &mut env) {
                Ok(v) => v,
                Err(_) => {
                    results.append(py.None())?;
                    continue;
                }
            };

            // Apply the closure to the input
            match apply(&fn_val, &[input_val], &nodes, &mut env) {
                Ok(val) => results.append(value_to_py(py, &val)?)?,
                Err(_) => results.append(py.None())?,
            }
        }

        Ok(results.into_any().unbind())
    }

    /// Evaluate a lambda on multiple inputs, comparing to expected outputs.
    /// Returns (num_matching, total).
    #[pyfunction]
    fn batch_verify(
        py: Python,
        program: &Bound<PyAny>,
        inputs: &Bound<PyList>,
        expected: &Bound<PyList>,
    ) -> PyResult<(usize, usize)> {
        let mut nodes = Vec::new();
        let root = py_to_nodes(py, program, &mut nodes)?;

        let total = inputs.len();
        let mut matching = 0;

        for (input_obj, expected_obj) in inputs.iter().zip(expected.iter()) {
            let input_val = py_to_value(py, &input_obj)?;
            let expected_val = py_to_value(py, &expected_obj)?;
            let mut env = make_default_env();

            let fn_val = match eval(&nodes, root, &mut env) {
                Ok(v) => v,
                Err(_) => continue,
            };

            match apply(&fn_val, &[input_val], &nodes, &mut env) {
                Ok(actual) => {
                    if values_equal(&actual, &expected_val) {
                        matching += 1;
                    }
                }
                Err(_) => {}
            }
        }

        Ok((matching, total))
    }

    /// Full synthesis in Rust. Takes component descriptions and examples,
    /// returns (found, source, candidates_explored).
    ///
    /// components: list of dicts with keys: name, builtin, arity, ret_type, param_types, priority
    ///   - ret_type/param_types: 0=num, 1=str, 2=bool
    ///   - builtin: the function name to call (None for constants)
    /// inputs: list of input values
    /// expected: list of expected output values
    /// max_depth: max AST depth
    /// max_candidates: search budget
    #[pyfunction]
    fn fast_synthesize(
        py: Python,
        components: &Bound<PyList>,
        inputs: &Bound<PyList>,
        expected: &Bound<PyList>,
        max_depth: usize,
        max_candidates: usize,
        macros: Option<&Bound<PyList>>,
    ) -> PyResult<(bool, String, usize)> {
        // Convert components
        let mut rust_comps = Vec::new();
        for comp_obj in components.iter() {
            let dict = comp_obj.downcast::<PyDict>()?;
            let name: String = dict.get_item("name")?.unwrap().extract()?;
            let arity: usize = dict.get_item("arity")?.unwrap().extract()?;
            let ret_type: u8 = dict.get_item("ret_type")?.unwrap().extract()?;

            let builtin: Option<String> = if let Some(b) = dict.get_item("builtin")? {
                if b.is_none() { None } else { Some(b.extract()?) }
            } else { None };

            let param_types: Vec<u8> = if let Some(pt) = dict.get_item("param_types")? {
                pt.extract()?
            } else { vec![] };

            let priority: f64 = if let Some(p) = dict.get_item("priority")? {
                p.extract().unwrap_or(0.0)
            } else { 0.0 };

            rust_comps.push(RustComponent {
                name, builtin, arity, ret_type, param_types, priority,
            });
        }

        // Sort by priority descending
        rust_comps.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        // Convert macros: compile each body to Rust nodes
        let mut macro_defs: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
        if let Some(macro_list) = macros {
            for macro_obj in macro_list.iter() {
                let dict = macro_obj.downcast::<PyDict>()?;
                let name: String = dict.get_item("name")?.unwrap().extract()?;
                let params: Vec<String> = dict.get_item("params")?.unwrap().extract()?;
                let body_ast = dict.get_item("body")?.unwrap();
                let mut body_nodes = Vec::new();
                let body_root = py_to_nodes(py, &body_ast, &mut body_nodes)?;
                macro_defs.push((name, params, body_nodes, body_root));
            }
        }

        // Convert inputs/expected
        let rust_inputs: Vec<Value> = inputs.iter()
            .map(|obj| py_to_value(py, &obj).unwrap_or(Value::Nil))
            .collect();
        let rust_expected: Vec<Value> = expected.iter()
            .map(|obj| py_to_value(py, &obj).unwrap_or(Value::Nil))
            .collect();

        // Run synthesis
        let (result, explored) = rust_synthesize(
            &rust_comps, &rust_inputs, &rust_expected,
            max_depth, max_candidates, &macro_defs);

        match result {
            Some((nodes, root)) => {
                let source = node_to_source(&nodes, root);
                Ok((true, source, explored))
            }
            None => Ok((false, String::new(), explored)),
        }
    }

    /// Optimization synthesis in Rust.
    /// Generates candidates in Rust, but calls a Python fitness function
    /// to score each one. Returns the best-scoring program.
    ///
    /// fitness_fn: Python callable that takes a SELPH AST node and returns a float score.
    /// minimize: if true, lower score is better; if false, higher is better.
    ///
    /// Candidates are generated in Rust (fast), but scored via Python callback
    /// (necessary since fitness may call arbitrary SELPH code like synthesize).
    #[pyfunction]
    fn fast_optimize(
        py: Python,
        components: &Bound<PyList>,
        max_depth: usize,
        max_candidates: usize,
        fitness_fn: &Bound<PyAny>,
        minimize: bool,
        macros: Option<&Bound<PyList>>,
    ) -> PyResult<(bool, String, usize, f64)> {
        // Convert components
        let mut rust_comps = Vec::new();
        for comp_obj in components.iter() {
            let dict = comp_obj.downcast::<PyDict>()?;
            let name: String = dict.get_item("name")?.unwrap().extract()?;
            let arity: usize = dict.get_item("arity")?.unwrap().extract()?;
            let ret_type: u8 = dict.get_item("ret_type")?.unwrap().extract()?;
            let builtin: Option<String> = if let Some(b) = dict.get_item("builtin")? {
                if b.is_none() { None } else { Some(b.extract()?) }
            } else { None };
            let param_types: Vec<u8> = if let Some(pt) = dict.get_item("param_types")? {
                pt.extract()?
            } else { vec![] };
            let priority: f64 = if let Some(p) = dict.get_item("priority")? {
                p.extract().unwrap_or(0.0)
            } else { 0.0 };
            rust_comps.push(RustComponent {
                name, builtin, arity, ret_type, param_types, priority,
            });
        }
        rust_comps.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        // Convert macros
        let mut macro_defs: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
        if let Some(macro_list) = macros {
            for macro_obj in macro_list.iter() {
                let dict = macro_obj.downcast::<PyDict>()?;
                let name: String = dict.get_item("name")?.unwrap().extract()?;
                let params: Vec<String> = dict.get_item("params")?.unwrap().extract()?;
                let body_ast = dict.get_item("body")?.unwrap();
                let mut body_nodes = Vec::new();
                let body_root = py_to_nodes(py, &body_ast, &mut body_nodes)?;
                macro_defs.push((name, params, body_nodes, body_root));
            }
        }

        // Generate all candidates in Rust, score each via Python callback
        let mut best_score: f64 = if minimize { f64::INFINITY } else { f64::NEG_INFINITY };
        let mut best_source = String::new();
        let mut best_found = false;
        let mut candidates_explored: usize = 0;
        let mut seen_behaviors: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();

        // Build program pool
        let mut all_pool: Vec<PoolEntry> = Vec::new();
        let mut prev_layer: Vec<usize> = Vec::new();

        for comp in &rust_comps {
            if comp.arity != 0 { continue; }
            let mut nodes = Vec::new();
            if comp.name == "x" {
                nodes.push(Node::Symbol("x".to_string()));
            } else if let Ok(n) = comp.name.parse::<f64>() {
                nodes.push(Node::Num(n));
            } else {
                nodes.push(Node::Str(comp.name.clone()));
            }
            let idx = all_pool.len();
            all_pool.push(PoolEntry { nodes, root: 0, ret_type: comp.ret_type, max_priority: comp.priority });
            prev_layer.push(idx);
        }

        for depth in 0..=max_depth {
            let entries_to_test = if depth == 0 {
                prev_layer.clone()
            } else {
                let new_entries = expand_layer_rust(&prev_layer, &all_pool, &rust_comps);
                let start = all_pool.len();
                all_pool.extend(new_entries);
                (start..all_pool.len()).collect()
            };

            let mut kept = Vec::new();
            for &idx in &entries_to_test {
                candidates_explored += 1;
                if candidates_explored > max_candidates { break; }

                let entry = &all_pool[idx];
                let mut lambda_nodes = entry.nodes.clone();
                let _param_idx = lambda_nodes.len();
                lambda_nodes.push(Node::Symbol("x".to_string()));
                let lambda_root = lambda_nodes.len();
                lambda_nodes.push(Node::Lambda(vec!["x".to_string()], entry.root));

                let source = node_to_source(&lambda_nodes, lambda_root);

                // Convert the program to a Python AST for the fitness function
                // We pass the source string and let Python parse+eval it
                let py_source = source.as_str().into_pyobject(py)?.into_any().unbind();
                let score_result = fitness_fn.call1((py_source,));

                match score_result {
                    Ok(score_obj) => {
                        if let Ok(score) = score_obj.extract::<f64>() {
                            let is_better = if minimize { score < best_score } else { score > best_score };
                            if is_better {
                                best_score = score;
                                best_source = source;
                                best_found = true;
                            }
                        }
                    }
                    Err(_) => {} // fitness function errored, skip
                }

                kept.push(idx);
            }

            if candidates_explored > max_candidates { break; }
            if depth > 0 { prev_layer = kept; }
        }

        Ok((best_found, best_source, candidates_explored, best_score))
    }

    /// Search for the best priority assignment entirely in Rust.
    ///
    /// Takes a task suite (list of (inputs, expected) pairs), component
    /// definitions, macros, and priority search config. Runs synthesis
    /// for each priority assignment and returns the best one.
    ///
    /// This keeps the ENTIRE optimization loop in Rust — no Python callbacks.
    #[pyfunction]
    fn fast_search_priorities(
        py: Python,
        components: &Bound<PyList>,
        task_suite: &Bound<PyList>,  // list of (inputs_list, expected_list) tuples
        macros: Option<&Bound<PyList>>,
        tunable_names: Vec<String>,
        priority_values: Vec<f64>,
        max_depth: usize,
        synth_budget: usize,
    ) -> PyResult<(PyObject, usize, f64)> {
        // Parse components
        let mut base_comps: Vec<RustComponent> = Vec::new();
        for comp_obj in components.iter() {
            let dict = comp_obj.downcast::<PyDict>()?;
            let name: String = dict.get_item("name")?.unwrap().extract()?;
            let arity: usize = dict.get_item("arity")?.unwrap().extract()?;
            let ret_type: u8 = dict.get_item("ret_type")?.unwrap().extract()?;
            let builtin: Option<String> = if let Some(b) = dict.get_item("builtin")? {
                if b.is_none() { None } else { Some(b.extract()?) }
            } else { None };
            let param_types: Vec<u8> = if let Some(pt) = dict.get_item("param_types")? {
                pt.extract()?
            } else { vec![] };
            base_comps.push(RustComponent {
                name, builtin, arity, ret_type, param_types, priority: 0.0,
            });
        }

        // Parse macros
        let mut macro_defs: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
        if let Some(macro_list) = macros {
            for macro_obj in macro_list.iter() {
                let dict = macro_obj.downcast::<PyDict>()?;
                let name: String = dict.get_item("name")?.unwrap().extract()?;
                let params: Vec<String> = dict.get_item("params")?.unwrap().extract()?;
                let body_ast = dict.get_item("body")?.unwrap();
                let mut body_nodes = Vec::new();
                let body_root = py_to_nodes(py, &body_ast, &mut body_nodes)?;
                macro_defs.push((name, params, body_nodes, body_root));
            }
        }

        // Parse task suite
        let mut tasks: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();
        for task_obj in task_suite.iter() {
            let tuple = task_obj.downcast::<PyTuple>()?;
            let inputs_list = tuple.get_item(0)?.downcast::<PyList>()?.clone();
            let expected_list = tuple.get_item(1)?.downcast::<PyList>()?.clone();
            let inputs: Vec<Value> = inputs_list.iter()
                .map(|o| py_to_value(py, &o).unwrap_or(Value::Nil)).collect();
            let expected: Vec<Value> = expected_list.iter()
                .map(|o| py_to_value(py, &o).unwrap_or(Value::Nil)).collect();
            tasks.push((inputs, expected));
        }

        // Map tunable names to component indices
        let tunable_indices: Vec<usize> = tunable_names.iter()
            .filter_map(|name| base_comps.iter().position(|c| &c.name == name))
            .collect();

        // Exhaustive search over priority assignments
        let n_tunable = tunable_indices.len();
        let n_values = priority_values.len();
        let total_assignments = n_values.pow(n_tunable as u32);

        let mut best_score = usize::MAX;
        let mut best_priorities: Vec<(String, f64)> = Vec::new();
        let mut assignments_tested = 0usize;

        for assignment_idx in 0..total_assignments {
            // Decode the assignment index into priority values
            let mut idx = assignment_idx;
            let mut comps = base_comps.clone();
            let mut current_priorities: Vec<(String, f64)> = Vec::new();

            for &comp_idx in &tunable_indices {
                let val_idx = idx % n_values;
                idx /= n_values;
                let priority = priority_values[val_idx];
                comps[comp_idx].priority = priority;
                current_priorities.push((comps[comp_idx].name.clone(), priority));
            }

            // Sort components by priority
            comps.sort_by(|a, b| b.priority.partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal));

            // Score: total candidates across all tasks
            let mut total = 0usize;
            let mut all_solved = true;
            for (inputs, expected) in &tasks {
                let (result, explored) = rust_synthesize(
                    &comps, inputs, expected, max_depth, synth_budget, &macro_defs);
                if result.is_none() {
                    total += synth_budget;
                    all_solved = false;
                } else {
                    total += explored;
                }
                // Early exit if already worse than best
                if total >= best_score { break; }
            }

            assignments_tested += 1;

            if total < best_score {
                best_score = total;
                best_priorities = current_priorities;
            }
        }

        // Convert result to Python
        let result_dict = PyDict::new(py);
        for (name, priority) in &best_priorities {
            result_dict.set_item(name, *priority)?;
        }

        Ok((result_dict.into_any().unbind(), assignments_tested, best_score as f64))
    }

    /// Synthesize a SELPH heuristic program that minimizes total synthesis
    /// candidates across a task suite. The heuristic itself is a SELPH program.
    ///
    /// Returns (found, heuristic_source, score, heuristics_tested)
    #[pyfunction]
    fn synthesize_heuristic(
        py: Python,
        components: &Bound<PyList>,
        task_suite: &Bound<PyList>,
        macros: Option<&Bound<PyList>>,
        synth_budget: usize,
        synth_depth: usize,
    ) -> PyResult<(bool, String, f64, usize)> {
        // Parse components
        let mut base_comps: Vec<RustComponent> = Vec::new();
        for comp_obj in components.iter() {
            let dict = comp_obj.downcast::<PyDict>()?;
            let name: String = dict.get_item("name")?.unwrap().extract()?;
            let arity: usize = dict.get_item("arity")?.unwrap().extract()?;
            let ret_type: u8 = dict.get_item("ret_type")?.unwrap().extract()?;
            let builtin: Option<String> = if let Some(b) = dict.get_item("builtin")? {
                if b.is_none() { None } else { Some(b.extract()?) }
            } else { None };
            let param_types: Vec<u8> = if let Some(pt) = dict.get_item("param_types")? {
                pt.extract()?
            } else { vec![] };
            base_comps.push(RustComponent {
                name, builtin, arity, ret_type, param_types, priority: 0.0,
            });
        }

        // Parse macros
        let mut macro_defs: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
        if let Some(macro_list) = macros {
            for macro_obj in macro_list.iter() {
                let dict = macro_obj.downcast::<PyDict>()?;
                let name: String = dict.get_item("name")?.unwrap().extract()?;
                let params: Vec<String> = dict.get_item("params")?.unwrap().extract()?;
                let body_ast = dict.get_item("body")?.unwrap();
                let mut body_nodes = Vec::new();
                let body_root = py_to_nodes(py, &body_ast, &mut body_nodes)?;
                macro_defs.push((name, params, body_nodes, body_root));
            }
        }

        // Parse task suite
        let mut tasks: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();
        for task_obj in task_suite.iter() {
            let tuple = task_obj.downcast::<PyTuple>()?;
            let inputs_list = tuple.get_item(0)?.downcast::<PyList>()?.clone();
            let expected_list = tuple.get_item(1)?.downcast::<PyList>()?.clone();
            let inputs: Vec<Value> = inputs_list.iter()
                .map(|o| py_to_value(py, &o).unwrap_or(Value::Nil)).collect();
            let expected: Vec<Value> = expected_list.iter()
                .map(|o| py_to_value(py, &o).unwrap_or(Value::Nil)).collect();
            tasks.push((inputs, expected));
        }

        // Synthesize heuristic
        let (best_nodes, best_root, best_score, tested) = synthesize_selph_heuristic(
            &base_comps, &tasks, &macro_defs, synth_budget, synth_depth);

        let source = node_to_source(&best_nodes, best_root);
        Ok((best_score < (synth_budget * tasks.len()) as f64, source, best_score, tested))
    }

    /// Run an interleaved curriculum: solve tasks sequentially, updating
    /// component priorities after each success.
    ///
    /// Returns a list of (found, source, candidates) for each task,
    /// plus the final priority map.
    #[pyfunction]
    fn fast_interleaved(
        py: Python,
        components: &Bound<PyList>,
        task_suite: &Bound<PyList>,
        macros: Option<&Bound<PyList>>,
        max_depth: usize,
        max_candidates: usize,
        learn_rate: f64,
    ) -> PyResult<(Vec<(bool, String, usize)>, PyObject)> {
        // Parse components
        let mut base_comps: Vec<RustComponent> = Vec::new();
        for comp_obj in components.iter() {
            let dict = comp_obj.downcast::<PyDict>()?;
            let name: String = dict.get_item("name")?.unwrap().extract()?;
            let arity: usize = dict.get_item("arity")?.unwrap().extract()?;
            let ret_type: u8 = dict.get_item("ret_type")?.unwrap().extract()?;
            let builtin: Option<String> = if let Some(b) = dict.get_item("builtin")? {
                if b.is_none() { None } else { Some(b.extract()?) }
            } else { None };
            let param_types: Vec<u8> = if let Some(pt) = dict.get_item("param_types")? {
                pt.extract()?
            } else { vec![] };
            let priority: f64 = if let Some(p) = dict.get_item("priority")? {
                p.extract().unwrap_or(0.0)
            } else { 0.0 };
            base_comps.push(RustComponent {
                name, builtin, arity, ret_type, param_types, priority,
            });
        }

        // Parse macros
        let mut macro_defs: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
        if let Some(macro_list) = macros {
            for macro_obj in macro_list.iter() {
                let dict = macro_obj.downcast::<PyDict>()?;
                let name: String = dict.get_item("name")?.unwrap().extract()?;
                let params: Vec<String> = dict.get_item("params")?.unwrap().extract()?;
                let body_ast = dict.get_item("body")?.unwrap();
                let mut body_nodes = Vec::new();
                let body_root = py_to_nodes(py, &body_ast, &mut body_nodes)?;
                macro_defs.push((name, params, body_nodes, body_root));
            }
        }

        // Parse task suite
        let mut tasks: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();
        for task_obj in task_suite.iter() {
            let tuple = task_obj.downcast::<PyTuple>()?;
            let inputs_list = tuple.get_item(0)?.downcast::<PyList>()?.clone();
            let expected_list = tuple.get_item(1)?.downcast::<PyList>()?.clone();
            let inputs: Vec<Value> = inputs_list.iter()
                .map(|o| py_to_value(py, &o).unwrap_or(Value::Nil)).collect();
            let expected: Vec<Value> = expected_list.iter()
                .map(|o| py_to_value(py, &o).unwrap_or(Value::Nil)).collect();
            tasks.push((inputs, expected));
        }

        // Run interleaved synthesis
        let (results, final_priorities) = interleaved_synthesize(
            &base_comps, &tasks, &macro_defs,
            max_depth, max_candidates, learn_rate);

        // Return final priorities as a dict
        let priority_dict = PyDict::new(py);
        for (name, priority) in &final_priorities {
            priority_dict.set_item(name, *priority)?;
        }

        Ok((results, priority_dict.into_any().unbind()))
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        _ => false,
    }
}
