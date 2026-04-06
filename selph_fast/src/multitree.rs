//! Multi-tree synthesis for SELPH.
//!
//! Allows the synthesizer to search across multiple arbitrary namespace trees
//! simultaneously. Each tree contributes callable components to the search
//! space, with paths preserved for traceability.
//!
//! A program synthesized from multiple trees might use components like:
//!   `math.double`, `str.clean`, `data.lookup`
//!
//! For efficiency, namespace lookups are resolved at synthesis time and
//! components are registered directly in the eval environment.

use std::collections::HashMap;
use crate::types::*;
use crate::eval;
use crate::synth::{
    SynthComponent, SynthResult, synthesize_with_validation,
    default_synth_components_with_trees,
    value_type_tag, TYPE_NUM, TYPE_STR, TYPE_BOOL,
};

// ── Extract components from a Namespace ──────────────────────────────

/// Recursively walk a `Value::Namespace`, extracting callable entries as
/// `SynthComponent`s with qualified names.
///
/// - `ns`: a `Value::Namespace` to walk
/// - `tree_name`: the tree's identifier (e.g. "math"), used as the prefix
/// - `prefix`: dot-separated path within the namespace (used for recursion)
/// - `env`: eval environment; callables are registered here so the
///   synthesizer can call them by name
/// - `max_depth`: maximum nesting depth to recurse into
///
/// Returns a vector of `SynthComponent`s, one per callable or constant leaf.
pub fn extract_components_from_namespace(
    ns: &Value,
    tree_name: &str,
    prefix: &str,
    env: &mut Env,
    max_depth: usize,
) -> Vec<SynthComponent> {
    let mut components = Vec::new();
    extract_recursive(ns, tree_name, prefix, env, &mut components, 0, max_depth);
    components
}

fn extract_recursive(
    ns: &Value,
    tree_name: &str,
    prefix: &str,
    env: &mut Env,
    components: &mut Vec<SynthComponent>,
    depth: usize,
    max_depth: usize,
) {
    if depth > max_depth {
        return;
    }

    let map = match ns {
        Value::Namespace(m) => m,
        _ => return,
    };

    // Sort keys for deterministic ordering
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();

    for key in keys {
        let val = &map[key];
        let path_str = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{}.{}", prefix, key)
        };
        let full_name = if tree_name.is_empty() {
            path_str.clone()
        } else {
            format!("{}.{}", tree_name, path_str)
        };

        match val {
            Value::Namespace(_) => {
                extract_recursive(
                    val, tree_name, &path_str, env, components,
                    depth + 1, max_depth,
                );
            }
            Value::Closure(..) | Value::Builtin(_) => {
                if let Some((param_types, ret_type)) = infer_callable_type(val, env) {
                    // Register the callable in the env so the synthesizer can
                    // invoke it by its qualified name.
                    env_define(env, full_name.clone(), val.clone());

                    components.push(SynthComponent {
                        name: full_name.clone(),
                        builtin: Some(full_name),
                        arity: param_types.len(),
                        ret_type,
                        param_types,
                        priority: 0.0,
                    });
                }
            }
            Value::Num(n) => {
                // Constant number — generate a component whose node is a
                // numeric literal.  No builtin needed; the synthesizer
                // will emit a Num node via the name (parseable as f64).
                let name_str = if *n == (*n as i64) as f64 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                };
                components.push(SynthComponent {
                    name: name_str,
                    builtin: None,
                    arity: 0,
                    ret_type: TYPE_NUM,
                    param_types: vec![],
                    priority: 0.0,
                });
            }
            Value::Str(s) => {
                components.push(SynthComponent {
                    name: s.clone(),
                    builtin: None,
                    arity: 0,
                    ret_type: TYPE_STR,
                    param_types: vec![],
                    priority: 0.0,
                });
            }
            Value::Bool(b) => {
                components.push(SynthComponent {
                    name: if *b { "true".to_string() } else { "false".to_string() },
                    builtin: None,
                    arity: 0,
                    ret_type: TYPE_BOOL,
                    param_types: vec![],
                    priority: 0.0,
                });
            }
            _ => {
                // Skip non-extractable values (Nil, List, RustMacro, etc.)
            }
        }
    }
}

// ── Type inference for callables ─────────────────────────────────────

/// Probe a callable value with test inputs to infer its parameter types
/// and return type as u8 type tags.
///
/// Returns `Some((param_types, ret_type))` on success, `None` if the
/// signature cannot be determined.
///
/// Strategy:
/// - For closures, arity is known from the parameter list.
/// - For builtins, arity is probed by trying 0..4 arguments.
/// - Each arity is tested with sample values (0.0, "", true) to find
///   a combination that succeeds, then the return type is inferred.
pub fn infer_callable_type(
    val: &Value,
    env: &mut Env,
) -> Option<(Vec<u8>, u8)> {
    let arity = match val {
        Value::Closure(params, _, _, _, _) => params.len(),
        Value::Builtin(_) => probe_arity(val, env)?,
        _ => return None,
    };

    if arity == 0 {
        return None; // Nullary builtins are rare, skip
    }

    // Test inputs: (value, type_tag)
    let test_inputs: [(Value, u8); 3] = [
        (Value::Num(0.0), TYPE_NUM),
        (Value::Str(String::new()), TYPE_STR),
        (Value::Bool(true), TYPE_BOOL),
    ];

    let empty_nodes: Vec<Node> = Vec::new();

    for (test_val, test_tag) in &test_inputs {
        let args: Vec<Value> = vec![test_val.clone(); arity];
        let val_clone = val.clone();
        let nodes_clone = empty_nodes.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut trial_env = eval::make_default_env();
            eval::apply(&val_clone, &args, &nodes_clone, &mut trial_env)
        }));
        if let Ok(Ok(result_val)) = result {
            let ret_type = value_type_tag(&result_val);
            let param_types = vec![*test_tag; arity];
            return Some((param_types, ret_type));
        }
    }

    None
}

/// Determine the arity of a builtin by trying increasing argument counts.
/// Uses catch_unwind because some builtins index args without bounds checks.
fn probe_arity(val: &Value, _env: &mut Env) -> Option<usize> {
    let empty_nodes: Vec<Node> = Vec::new();
    for arity in 1..=4 {
        let args: Vec<Value> = vec![Value::Num(0.0); arity];
        let val_clone = val.clone();
        let nodes_clone = empty_nodes.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut trial_env = eval::make_default_env();
            eval::apply(&val_clone, &args, &nodes_clone, &mut trial_env)
        }));
        match result {
            Ok(Ok(_)) => return Some(arity),
            Ok(Err(e)) => {
                // Type mismatch means arity is correct, just wrong test type
                if e.contains("expected") && !e.contains("not callable") {
                    return Some(arity);
                }
                continue;
            }
            Err(_) => continue, // Panic (index out of bounds) — wrong arity
        }
    }
    None
}

// ── Multi-tree synthesis ─────────────────────────────────────────────

/// Synthesize a program searching across multiple namespace trees.
///
/// Each tree contributes its callable entries as synthesis components.
/// Trees can be arbitrary — ops libraries, data stores, cached results,
/// domain-specific knowledge, etc.
///
/// Tree-extracted closures and builtins are injected as `extra_bindings`
/// into the core synthesizer's eval environment, so they can be called
/// by their qualified names during candidate evaluation.
///
/// # Arguments
/// - `inputs`: example input values
/// - `expected`: expected output values (one per input)
/// - `trees`: slice of `(name, namespace_value)` pairs
/// - `macros`: library macros (name, params, nodes, root)
/// - `max_depth`: maximum AST depth to explore
/// - `max_candidates`: search budget
/// - `include_builtins`: whether to include default builtin components
///
/// Returns a `SynthResult` with the found program (if any).
pub fn multi_synthesize(
    inputs: &[Value],
    expected: &[Value],
    trees: &[(String, Value)],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    include_builtins: bool,
) -> SynthResult {
    let (all_components, extra_bindings) = if include_builtins {
        default_synth_components_with_trees(macros, trees)
    } else {
        // Minimal: at least provide the input variable `x`, plus tree components
        let mut env = eval::make_default_env();
        let mut comps = vec![SynthComponent {
            name: "x".into(),
            builtin: None,
            arity: 0,
            ret_type: TYPE_NUM,
            param_types: vec![],
            priority: 100.0,
        }];
        for (name, ns_val) in trees {
            let tree_comps = extract_components_from_namespace(
                ns_val, name, "", &mut env, 5,
            );
            comps.extend(tree_comps);
        }
        // Collect extra bindings from the env
        let default_env = eval::make_default_env();
        let default_scope = default_env.last().unwrap();
        let mut bindings = Vec::new();
        if let Some(scope) = env.last() {
            for (k, v) in scope {
                if !default_scope.contains_key(k) {
                    bindings.push((k.clone(), v.clone()));
                }
            }
        }
        (comps, bindings)
    };

    synthesize_with_validation(
        &all_components,
        inputs,
        expected,
        macros,
        max_depth,
        max_candidates,
        true, // enable_if
        None, // no validation examples
        &extra_bindings,
    )
}

// ── Solution tracing ─────────────────────────────────────────────────

/// Given a synthesized solution, trace which tree names appear in it.
///
/// Scans all nodes in the solution for `Symbol` references whose names
/// start with one of the `tree_names` followed by a dot.  Returns a
/// sorted, deduplicated list of `"tree_name.path"` strings found.
///
/// # Arguments
/// - `nodes`: the node arena from the synthesis result
/// - `root`: the root index of the solution
/// - `tree_names`: the names of the trees to check for
pub fn trace_solution(
    nodes: &[Node],
    root: usize,
    tree_names: &[String],
) -> Vec<String> {
    let mut usage: HashMap<String, Vec<String>> = HashMap::new();
    for name in tree_names {
        usage.insert(name.clone(), Vec::new());
    }

    scan_node(nodes, root, tree_names, &mut usage);

    // Flatten to a list of "tree_name: path1, path2, ..." entries,
    // but only for trees that actually contributed.
    let mut result = Vec::new();
    let mut sorted_names: Vec<&String> = usage.keys().collect();
    sorted_names.sort();

    for name in sorted_names {
        let paths = &usage[name];
        if !paths.is_empty() {
            let mut sorted_paths = paths.clone();
            sorted_paths.sort();
            sorted_paths.dedup();
            for path in sorted_paths {
                result.push(format!("{}.{}", name, path));
            }
        }
    }

    result
}

/// Recursively scan a node tree for symbol references from named trees.
fn scan_node(
    nodes: &[Node],
    idx: usize,
    tree_names: &[String],
    usage: &mut HashMap<String, Vec<String>>,
) {
    if idx >= nodes.len() {
        return;
    }
    match &nodes[idx] {
        Node::Symbol(name) => {
            for tree_name in tree_names {
                let prefix = format!("{}.", tree_name);
                if name.starts_with(&prefix) {
                    let path = name[prefix.len()..].to_string();
                    if let Some(paths) = usage.get_mut(tree_name) {
                        paths.push(path);
                    }
                }
            }
        }
        Node::App(children) => {
            for &child in children {
                scan_node(nodes, child, tree_names, usage);
            }
        }
        Node::If(c, t, e) => {
            scan_node(nodes, *c, tree_names, usage);
            scan_node(nodes, *t, tree_names, usage);
            scan_node(nodes, *e, tree_names, usage);
        }
        Node::Lambda(_, body) => {
            scan_node(nodes, *body, tree_names, usage);
        }
        Node::Let(bindings, body) => {
            for (_, val_idx) in bindings {
                scan_node(nodes, *val_idx, tree_names, usage);
            }
            scan_node(nodes, *body, tree_names, usage);
        }
        _ => {}
    }
}

// ── Convenience ──────────────────────────────────────────────────────

/// Extract components from a single namespace (convenience wrapper).
pub fn namespace_to_components(
    ns: &Value,
    name: &str,
) -> Vec<SynthComponent> {
    let mut env = eval::make_default_env();
    extract_components_from_namespace(ns, name, "", &mut env, 5)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn empty_ns() -> Value {
        Value::Namespace(HashMap::new())
    }

    fn make_ns(entries: Vec<(&str, Value)>) -> Value {
        let mut map = HashMap::new();
        for (k, v) in entries {
            map.insert(k.to_string(), v);
        }
        Value::Namespace(map)
    }

    // ── extract_components_from_namespace ─────────────────────────────

    #[test]
    fn test_extract_empty_namespace() {
        let ns = empty_ns();
        let mut env = eval::make_default_env();
        let comps = extract_components_from_namespace(&ns, "test", "", &mut env, 5);
        assert!(comps.is_empty());
    }

    #[test]
    fn test_extract_numeric_constants() {
        let ns = make_ns(vec![
            ("pi", Value::Num(3.14)),
            ("zero", Value::Num(0.0)),
        ]);
        let mut env = eval::make_default_env();
        let comps = extract_components_from_namespace(&ns, "math", "", &mut env, 5);
        assert_eq!(comps.len(), 2);
        for comp in &comps {
            assert_eq!(comp.arity, 0);
            assert_eq!(comp.ret_type, TYPE_NUM);
        }
    }

    #[test]
    fn test_extract_string_constants() {
        let ns = make_ns(vec![
            ("greeting", Value::Str("hello".to_string())),
        ]);
        let mut env = eval::make_default_env();
        let comps = extract_components_from_namespace(&ns, "data", "", &mut env, 5);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].ret_type, TYPE_STR);
        assert_eq!(comps[0].name, "hello");
    }

    #[test]
    fn test_extract_bool_constants() {
        let ns = make_ns(vec![
            ("flag", Value::Bool(true)),
        ]);
        let mut env = eval::make_default_env();
        let comps = extract_components_from_namespace(&ns, "config", "", &mut env, 5);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].ret_type, TYPE_BOOL);
        assert_eq!(comps[0].name, "true");
    }

    #[test]
    fn test_extract_builtins() {
        let ns = make_ns(vec![
            ("double", Value::Builtin("add".to_string())),
        ]);
        let mut env = eval::make_default_env();
        let comps = extract_components_from_namespace(&ns, "ops", "", &mut env, 5);
        // "add" is a known builtin taking 2 nums, so it should be extracted
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].name, "ops.double");
        assert_eq!(comps[0].arity, 2);
        assert_eq!(comps[0].ret_type, TYPE_NUM);
    }

    #[test]
    fn test_extract_nested_namespace() {
        let inner = make_ns(vec![
            ("value", Value::Num(42.0)),
        ]);
        let outer = make_ns(vec![
            ("deep", inner),
        ]);
        let mut env = eval::make_default_env();
        let comps = extract_components_from_namespace(&outer, "root", "", &mut env, 5);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].ret_type, TYPE_NUM);
    }

    #[test]
    fn test_extract_max_depth_limit() {
        let deep = make_ns(vec![("val", Value::Num(1.0))]);
        let mid = make_ns(vec![("inner", deep)]);
        let top = make_ns(vec![("mid", mid)]);
        let mut env = eval::make_default_env();
        // Depth 0 means only the top-level namespace entries (which is
        // itself a namespace, so nothing extracted)
        let comps = extract_components_from_namespace(&top, "t", "", &mut env, 0);
        // At depth 0, we enter the top namespace (depth=0) and see "mid"
        // which is a namespace. Recursing would be depth=1 > max_depth=0,
        // so it stops.
        assert!(comps.is_empty());
    }

    #[test]
    fn test_extract_non_namespace_is_noop() {
        let mut env = eval::make_default_env();
        let comps = extract_components_from_namespace(
            &Value::Num(42.0), "t", "", &mut env, 5,
        );
        assert!(comps.is_empty());
    }

    // ── infer_callable_type ──────────────────────────────────────────

    #[test]
    fn test_infer_builtin_add() {
        let val = Value::Builtin("add".to_string());
        let mut env = eval::make_default_env();
        let result = infer_callable_type(&val, &mut env);
        assert!(result.is_some());
        let (params, ret) = result.unwrap();
        assert_eq!(params, vec![TYPE_NUM, TYPE_NUM]);
        assert_eq!(ret, TYPE_NUM);
    }

    #[test]
    fn test_infer_builtin_string_upper() {
        let val = Value::Builtin("string-upper".to_string());
        let mut env = eval::make_default_env();
        let result = infer_callable_type(&val, &mut env);
        assert!(result.is_some());
        let (params, ret) = result.unwrap();
        assert_eq!(params, vec![TYPE_STR]);
        assert_eq!(ret, TYPE_STR);
    }

    #[test]
    fn test_infer_builtin_not() {
        let val = Value::Builtin("not".to_string());
        let mut env = eval::make_default_env();
        let result = infer_callable_type(&val, &mut env);
        assert!(result.is_some());
        let (params, ret) = result.unwrap();
        assert_eq!(params, vec![TYPE_BOOL]);
        assert_eq!(ret, TYPE_BOOL);
    }

    #[test]
    fn test_infer_non_callable() {
        let val = Value::Num(42.0);
        let mut env = eval::make_default_env();
        assert!(infer_callable_type(&val, &mut env).is_none());
    }

    #[test]
    fn test_infer_closure() {
        // Create a simple closure: (lambda (a b) (add a b))
        let nodes = vec![
            Node::Symbol("add".to_string()),   // 0
            Node::Symbol("a".to_string()),      // 1
            Node::Symbol("b".to_string()),      // 2
            Node::App(vec![0, 1, 2]),           // 3: (add a b)
        ];
        let env = eval::make_default_env();
        let closure = Value::Closure(
            vec!["a".to_string(), "b".to_string()],
            3,          // body index
            env.clone(),
            nodes,
            None,
        );
        let mut test_env = eval::make_default_env();
        let result = infer_callable_type(&closure, &mut test_env);
        assert!(result.is_some());
        let (params, ret) = result.unwrap();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], TYPE_NUM);
        assert_eq!(ret, TYPE_NUM);
    }

    // ── trace_solution ───────────────────────────────────────────────

    #[test]
    fn test_trace_empty_solution() {
        let nodes = vec![Node::Num(42.0)];
        let tree_names = vec!["math".to_string(), "str".to_string()];
        let result = trace_solution(&nodes, 0, &tree_names);
        assert!(result.is_empty());
    }

    #[test]
    fn test_trace_with_tree_references() {
        let nodes = vec![
            Node::Symbol("math.double".to_string()),  // 0
            Node::Symbol("x".to_string()),             // 1
            Node::App(vec![0, 1]),                     // 2: (math.double x)
        ];
        let tree_names = vec!["math".to_string(), "str".to_string()];
        let result = trace_solution(&nodes, 2, &tree_names);
        assert_eq!(result, vec!["math.double".to_string()]);
    }

    #[test]
    fn test_trace_multiple_trees() {
        let nodes = vec![
            Node::Symbol("str.clean".to_string()),     // 0
            Node::Symbol("x".to_string()),              // 1
            Node::App(vec![0, 1]),                      // 2: (str.clean x)
            Node::Symbol("math.double".to_string()),    // 3
            Node::App(vec![3, 2]),                      // 4: (math.double (str.clean x))
        ];
        let tree_names = vec!["math".to_string(), "str".to_string()];
        let result = trace_solution(&nodes, 4, &tree_names);
        assert_eq!(result, vec![
            "math.double".to_string(),
            "str.clean".to_string(),
        ]);
    }

    #[test]
    fn test_trace_nested_paths() {
        let nodes = vec![
            Node::Symbol("lib.math.trig.sin".to_string()),  // 0
            Node::Symbol("x".to_string()),                    // 1
            Node::App(vec![0, 1]),                            // 2
        ];
        let tree_names = vec!["lib".to_string()];
        let result = trace_solution(&nodes, 2, &tree_names);
        assert_eq!(result, vec!["lib.math.trig.sin".to_string()]);
    }

    #[test]
    fn test_trace_if_expression() {
        let nodes = vec![
            Node::Symbol("ops.check".to_string()),  // 0
            Node::Symbol("x".to_string()),           // 1
            Node::App(vec![0, 1]),                   // 2: (ops.check x)
            Node::Symbol("ops.then".to_string()),    // 3
            Node::App(vec![3, 1]),                   // 4: (ops.then x)
            Node::Num(0.0),                          // 5
            Node::If(2, 4, 5),                       // 6: (if (ops.check x) (ops.then x) 0)
        ];
        let tree_names = vec!["ops".to_string()];
        let result = trace_solution(&nodes, 6, &tree_names);
        assert_eq!(result, vec![
            "ops.check".to_string(),
            "ops.then".to_string(),
        ]);
    }

    #[test]
    fn test_trace_deduplicates() {
        let nodes = vec![
            Node::Symbol("math.add".to_string()),   // 0
            Node::Symbol("x".to_string()),            // 1
            Node::App(vec![0, 1, 1]),                 // 2: (math.add x x)
            // Duplicate symbol reference
            Node::Symbol("math.add".to_string()),    // 3
            Node::App(vec![3, 2, 2]),                 // 4: (math.add (math.add x x) (math.add x x))
        ];
        let tree_names = vec!["math".to_string()];
        let result = trace_solution(&nodes, 4, &tree_names);
        // "math.add" should appear only once despite multiple occurrences
        assert_eq!(result, vec!["math.add".to_string()]);
    }

    // ── multi_synthesize ─────────────────────────────────────────────

    #[test]
    fn test_multi_synthesize_with_builtin_tree() {
        // Build a tree that contains "add" as a component named "ops.plus"
        let ns = make_ns(vec![
            ("plus", Value::Builtin("add".to_string())),
        ]);
        let trees = vec![("ops".to_string(), ns)];
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();

        // Simple spec: double(x) = x + x
        // This should be findable with the default builtins alone
        let inputs = vec![Value::Num(1.0), Value::Num(3.0), Value::Num(5.0)];
        let expected = vec![Value::Num(2.0), Value::Num(6.0), Value::Num(10.0)];

        let result = multi_synthesize(
            &inputs, &expected, &trees, &macros,
            2, 50000, true,
        );

        assert!(result.found);
    }

    #[test]
    fn test_multi_synthesize_no_trees() {
        let trees: Vec<(String, Value)> = vec![];
        let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();

        // Simple identity: f(x) = x
        let inputs = vec![Value::Num(1.0), Value::Num(2.0)];
        let expected = vec![Value::Num(1.0), Value::Num(2.0)];

        let result = multi_synthesize(
            &inputs, &expected, &trees, &macros,
            1, 10000, true,
        );

        assert!(result.found);
    }

    // ── namespace_to_components ───────────────────────────────────────

    #[test]
    fn test_namespace_to_components_convenience() {
        let ns = make_ns(vec![
            ("val", Value::Num(99.0)),
            ("flag", Value::Bool(false)),
        ]);
        let comps = namespace_to_components(&ns, "cfg");
        assert_eq!(comps.len(), 2);
    }
}
