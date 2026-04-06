//! Library management for SELPH.
//!
//! Handles persistence (save/load .selph files), promotion of solved
//! programs into macros, behavioral pruning, and usage tracking.

use std::collections::HashMap;
use crate::types::*;
use crate::parser;
use crate::eval;

// ── Builtin set (for equivalence checking) ──────────────────────────

const UNARY_BUILTINS: &[&str] = &[
    "string-upper", "string-lower", "string-reverse", "string-trim",
    "string-length", "abs", "negate", "even", "odd", "not",
    "floor", "ceil", "to-string", "to-number", "identity",
];

// ── Persistence ─────────────────────────────────────────────────────

/// Save a library of macros to a `.selph` file.
///
/// Each entry is `(name, params, body_source)`.  The file is written as
/// a sequence of `(defmacro name (params...) body)` expressions that
/// `load_library` can parse back.
pub fn save_library(
    macros: &[(String, Vec<String>, String)],
    path: &str,
) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("; SELPH library -- auto-generated\n");
    out.push_str(&format!("; {} macros\n\n", macros.len()));

    for (name, params, body_source) in macros {
        out.push_str(&format!(
            "(defmacro {} ({}) {})\n\n",
            name,
            params.join(" "),
            body_source,
        ));
    }

    std::fs::write(path, out).map_err(|e| format!("save_library: {}", e))
}

/// Load a library from a `.selph` file.
///
/// Returns a vector of `(name, params, body_nodes, body_root)` tuples.
/// `body_nodes` is the node pool for the body and `body_root` is the
/// index of the root node inside that pool.
pub fn load_library(
    path: &str,
) -> Result<Vec<(String, Vec<String>, Vec<Node>, usize)>, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("load_library: {}", e))?;

    if source.trim().is_empty() {
        return Ok(Vec::new());
    }

    let (nodes, roots) = parser::parse_file(&source)
        .map_err(|e| format!("load_library parse: {}", e))?;

    let mut macros = Vec::new();

    for root in roots {
        // Each root should be (defmacro name (params...) body)
        let children = match &nodes[root] {
            Node::App(children) if children.len() == 4 => children.clone(),
            _ => continue,
        };

        // children[0] must be the symbol "defmacro"
        let head_name = match &nodes[children[0]] {
            Node::Symbol(s) => s.clone(),
            _ => continue,
        };
        if head_name != "defmacro" {
            continue;
        }

        // children[1] = macro name
        let macro_name = match &nodes[children[1]] {
            Node::Symbol(s) => s.clone(),
            _ => continue,
        };

        // children[2] = param list (App of symbols)
        let params: Vec<String> = match &nodes[children[2]] {
            Node::App(param_indices) => {
                param_indices
                    .iter()
                    .filter_map(|&i| {
                        if let Node::Symbol(s) = &nodes[i] {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            _ => continue,
        };

        // children[3] = body root index
        let body_root = children[3];

        // We store the entire node pool; the body references into it
        // via indices.  This matches how RustMacro works in eval.
        macros.push((macro_name, params, nodes.clone(), body_root));
    }

    Ok(macros)
}

// ── Promotion ───────────────────────────────────────────────────────

/// Promote a solved lambda into a defmacro definition.
///
/// `source` should look like `(lambda (x) body)`.  Returns
/// `Some((macro_name, defmacro_source))` where the lambda parameter is
/// rewritten to `s` and the expression is wrapped in `defmacro`.
///
/// Returns `None` if `source` cannot be parsed as a lambda.
pub fn promote_solution(name: &str, source: &str) -> Option<(String, String)> {
    let (nodes, root) = parser::parse_source(source).ok()?;

    match &nodes[root] {
        Node::Lambda(params, body_idx) => {
            if params.is_empty() {
                return None;
            }

            // Rebuild the body source, substituting each original param
            // with `s` (single-arg convention).  If there are multiple
            // params we keep them as-is.
            let body_source = if params.len() == 1 {
                let original = &params[0];
                let body_src = node_to_source(&nodes, *body_idx);
                // Simple textual replacement of the parameter name.
                // Safe because SELPH identifiers don't appear as
                // substrings of other identifiers unless they share a
                // prefix, and we use word-boundary-aware replacement.
                replace_identifier(&body_src, original, "s")
            } else {
                node_to_source(&nodes, *body_idx)
            };

            let macro_params = if params.len() == 1 {
                vec!["s".to_string()]
            } else {
                params.clone()
            };

            let macro_name = name.to_string();
            let defmacro_source = format!(
                "(defmacro {} ({}) {})",
                macro_name,
                macro_params.join(" "),
                body_source,
            );
            Some((macro_name, defmacro_source))
        }
        _ => None,
    }
}

/// Replace all occurrences of identifier `old` with `new` in source text.
///
/// Only replaces whole identifiers, not substrings of longer names.
fn replace_identifier(source: &str, old: &str, new: &str) -> String {
    let mut result = String::with_capacity(source.len());
    let chars: Vec<char> = source.chars().collect();
    let old_chars: Vec<char> = old.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if i + old_chars.len() <= chars.len()
            && chars[i..i + old_chars.len()] == old_chars[..]
        {
            // Check that the character before (if any) is a delimiter
            let before_ok = i == 0 || is_delimiter(chars[i - 1]);
            // Check that the character after (if any) is a delimiter
            let after_ok = i + old_chars.len() >= chars.len()
                || is_delimiter(chars[i + old_chars.len()]);

            if before_ok && after_ok {
                result.push_str(new);
                i += old_chars.len();
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

fn is_delimiter(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n' | '(' | ')' | '"' | ',')
}

// ── Pruning ─────────────────────────────────────────────────────────

/// Prune a library by removing behaviourally-redundant macros.
///
/// Each macro is evaluated on every `test_input`.  Two macros that
/// produce identical output vectors are considered equivalent; the one
/// with the smaller body (fewer nodes) is kept.  Macros whose behaviour
/// matches a unary builtin are also removed.
///
/// Input/output tuples are `(name, params, nodes, body_root)`.
pub fn prune_library(
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    test_inputs: &[Value],
) -> Vec<(String, Vec<String>, Vec<Node>, usize)> {
    // Step 1: compute behavioural fingerprint for each macro
    let mut fingerprints: Vec<(usize, String)> = Vec::new(); // (index, fingerprint)

    for (i, (name, params, nodes, body_root)) in macros.iter().enumerate() {
        if let Some(fp) = compute_macro_fingerprint(name, params, nodes, *body_root, test_inputs) {
            fingerprints.push((i, fp));
        } else {
            // Can't evaluate -- keep it by giving it a unique fingerprint
            fingerprints.push((i, format!("__unevaluable_{}", i)));
        }
    }

    // Step 2: compute builtin fingerprints for exclusion
    let builtin_fps = compute_builtin_fingerprints(test_inputs);

    // Step 3: group by fingerprint, keep simplest per group, exclude
    //         builtin-equivalent macros
    let mut best_per_fp: HashMap<String, usize> = HashMap::new();

    for &(idx, ref fp) in &fingerprints {
        // Skip macros equivalent to builtins
        if builtin_fps.contains(fp) {
            continue;
        }

        match best_per_fp.get(fp) {
            Some(&existing_idx) => {
                let existing_size = body_size(&macros[existing_idx].2, macros[existing_idx].3);
                let new_size = body_size(&macros[idx].2, macros[idx].3);
                if new_size < existing_size {
                    best_per_fp.insert(fp.clone(), idx);
                }
            }
            None => {
                best_per_fp.insert(fp.clone(), idx);
            }
        }
    }

    // Collect results, preserving original order
    let mut kept_indices: Vec<usize> = best_per_fp.values().copied().collect();
    kept_indices.sort();

    kept_indices
        .into_iter()
        .map(|i| macros[i].clone())
        .collect()
}

/// Compute a behavioural fingerprint for a single macro.
///
/// Registers the macro in a fresh env, calls it on each test input,
/// and joins the string representations of the outputs.
fn compute_macro_fingerprint(
    name: &str,
    params: &[String],
    nodes: &[Node],
    body_root: usize,
    test_inputs: &[Value],
) -> Option<String> {
    // Only fingerprint unary macros (the common case for library prims)
    if params.len() != 1 {
        return None;
    }

    let mut env = eval::make_default_env();
    env_define(
        &mut env,
        name.to_string(),
        Value::RustMacro(params.to_vec(), nodes.to_vec(), body_root),
    );

    let mut outputs = Vec::new();
    for input in test_inputs {
        match eval::apply(
            &Value::RustMacro(params.to_vec(), nodes.to_vec(), body_root),
            &[input.clone()],
            &[] as &[Node],
            &mut env,
        ) {
            Ok(val) => outputs.push(eval::value_to_string(&val)),
            Err(_) => outputs.push("ERR".to_string()),
        }
    }

    Some(outputs.join("|"))
}

/// Compute fingerprints for all unary builtins so we can detect macros
/// that duplicate builtin behaviour.
fn compute_builtin_fingerprints(test_inputs: &[Value]) -> std::collections::HashSet<String> {
    let mut fps = std::collections::HashSet::new();

    for &builtin_name in UNARY_BUILTINS {
        let mut outputs = Vec::new();
        for input in test_inputs {
            let args: Vec<Value> = vec![input.clone()];
            match eval::apply_builtin(builtin_name, &args) {
                Ok(val) => outputs.push(eval::value_to_string(&val)),
                Err(_) => outputs.push("ERR".to_string()),
            }
        }
        fps.insert(outputs.join("|"));
    }

    // Also add the identity fingerprint (input == output)
    let identity_outputs: Vec<String> = test_inputs
        .iter()
        .map(|v| eval::value_to_string(v))
        .collect();
    fps.insert(identity_outputs.join("|"));

    fps
}

/// Count the number of nodes reachable from `idx` in the pool.
fn body_size(nodes: &[Node], idx: usize) -> usize {
    match &nodes[idx] {
        Node::Num(_) | Node::Str(_) | Node::Bool(_) | Node::Symbol(_) => 1,
        Node::App(children) => {
            1 + children.iter().map(|&c| body_size(nodes, c)).sum::<usize>()
        }
        Node::If(c, t, e) => {
            1 + body_size(nodes, *c) + body_size(nodes, *t) + body_size(nodes, *e)
        }
        Node::Lambda(_, body) => 1 + body_size(nodes, *body),
        Node::Let(bindings, body) => {
            1 + bindings
                .iter()
                .map(|(_, v)| body_size(nodes, *v))
                .sum::<usize>()
                + body_size(nodes, *body)
        }
    }
}

// ── Usage tracking ──────────────────────────────────────────────────

/// Extract component (identifier) names referenced in a SELPH source string.
///
/// Returns all symbol names found in the parsed AST, deduplicated.
pub fn extract_components(source: &str) -> Vec<String> {
    let parsed = parser::parse_source(source);
    let (nodes, root) = match parsed {
        Ok(pair) => pair,
        Err(_) => return Vec::new(),
    };

    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    collect_symbols(&nodes, root, &mut seen, &mut result);
    result
}

fn collect_symbols(
    nodes: &[Node],
    idx: usize,
    seen: &mut std::collections::HashSet<String>,
    result: &mut Vec<String>,
) {
    match &nodes[idx] {
        Node::Symbol(name) => {
            if seen.insert(name.clone()) {
                result.push(name.clone());
            }
        }
        Node::App(children) => {
            for &c in children {
                collect_symbols(nodes, c, seen, result);
            }
        }
        Node::If(c, t, e) => {
            collect_symbols(nodes, *c, seen, result);
            collect_symbols(nodes, *t, seen, result);
            collect_symbols(nodes, *e, seen, result);
        }
        Node::Lambda(_, body) => {
            collect_symbols(nodes, *body, seen, result);
        }
        Node::Let(bindings, body) => {
            for (_, v) in bindings {
                collect_symbols(nodes, *v, seen, result);
            }
            collect_symbols(nodes, *body, seen, result);
        }
        _ => {}
    }
}

/// Count how many of the given `solutions` reference each macro name.
///
/// For each solution source string the identifiers are extracted; any
/// identifier that appears in `macro_names` increments that macro's
/// count.
pub fn track_usage(
    solutions: &[String],
    macro_names: &[String],
) -> HashMap<String, usize> {
    let name_set: std::collections::HashSet<&str> =
        macro_names.iter().map(|s| s.as_str()).collect();

    let mut counts: HashMap<String, usize> = macro_names
        .iter()
        .map(|n| (n.clone(), 0))
        .collect();

    for solution in solutions {
        let components = extract_components(solution);
        for comp in &components {
            if name_set.contains(comp.as_str()) {
                *counts.entry(comp.clone()).or_insert(0) += 1;
            }
        }
    }

    counts
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_roundtrip() {
        let macros = vec![
            (
                "double".to_string(),
                vec!["x".to_string()],
                "(add x x)".to_string(),
            ),
            (
                "inc".to_string(),
                vec!["n".to_string()],
                "(add n 1)".to_string(),
            ),
        ];

        let path = "/tmp/selph_test_library.selph";
        save_library(&macros, path).unwrap();
        let loaded = load_library(path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "double");
        assert_eq!(loaded[0].1, vec!["x".to_string()]);
        assert_eq!(loaded[1].0, "inc");
        assert_eq!(loaded[1].1, vec!["n".to_string()]);

        // Clean up
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_promote_solution_simple() {
        let result = promote_solution("upper", "(lambda (x) (string-upper x))");
        assert!(result.is_some());
        let (name, defmacro_src) = result.unwrap();
        assert_eq!(name, "upper");
        assert!(defmacro_src.contains("defmacro"));
        assert!(defmacro_src.contains("string-upper"));
        assert!(defmacro_src.contains("(s)"));
    }

    #[test]
    fn test_promote_solution_not_lambda() {
        let result = promote_solution("foo", "(add 1 2)");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_components() {
        let components = extract_components("(add (string-upper x) 1)");
        assert!(components.contains(&"add".to_string()));
        assert!(components.contains(&"string-upper".to_string()));
        assert!(components.contains(&"x".to_string()));
    }

    #[test]
    fn test_track_usage() {
        let solutions = vec![
            "(double x)".to_string(),
            "(add (double y) (inc y))".to_string(),
            "(inc z)".to_string(),
        ];
        let macro_names = vec!["double".to_string(), "inc".to_string(), "unused".to_string()];
        let usage = track_usage(&solutions, &macro_names);

        assert_eq!(usage["double"], 2);
        assert_eq!(usage["inc"], 2);
        assert_eq!(usage["unused"], 0);
    }

    #[test]
    fn test_prune_removes_builtin_equivalent() {
        // A macro that does exactly what string-upper does
        let source = "(string-upper s)";
        let (nodes, root) = parser::parse_source(source).unwrap();
        let macros = vec![(
            "my-upper".to_string(),
            vec!["s".to_string()],
            nodes,
            root,
        )];

        let test_inputs = vec![
            Value::Str("hello".to_string()),
            Value::Str("world".to_string()),
        ];

        let pruned = prune_library(&macros, &test_inputs);
        assert_eq!(pruned.len(), 0, "builtin-equivalent macro should be pruned");
    }

    #[test]
    fn test_prune_keeps_novel_macro() {
        // A macro that does something no builtin does
        let source = "(concat (string-upper s) (string-lower s))";
        let (nodes, root) = parser::parse_source(source).unwrap();
        let macros = vec![(
            "upper-lower".to_string(),
            vec!["s".to_string()],
            nodes,
            root,
        )];

        let test_inputs = vec![
            Value::Str("Hello".to_string()),
            Value::Str("World".to_string()),
        ];

        let pruned = prune_library(&macros, &test_inputs);
        assert_eq!(pruned.len(), 1, "novel macro should be kept");
    }

    #[test]
    fn test_prune_deduplicates() {
        // Two macros with identical behaviour -- keep the simpler one
        let src1 = "(string-upper (identity s))"; // bigger
        let (nodes1, root1) = parser::parse_source(src1).unwrap();

        // This one is equivalent to string-upper, which is a builtin,
        // so both would be pruned.  Use a non-builtin behaviour instead.
        let src_a = "(concat s s)";
        let (nodes_a, root_a) = parser::parse_source(src_a).unwrap();

        let src_b = "(concat (identity s) (identity s))"; // bigger, same behaviour
        let (nodes_b, root_b) = parser::parse_source(src_b).unwrap();

        let macros = vec![
            ("dup-big".to_string(), vec!["s".to_string()], nodes_b, root_b),
            ("dup-small".to_string(), vec!["s".to_string()], nodes_a, root_a),
        ];

        let test_inputs = vec![
            Value::Str("ab".to_string()),
            Value::Str("xy".to_string()),
        ];

        let pruned = prune_library(&macros, &test_inputs);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].0, "dup-small");
    }

    #[test]
    fn test_body_size() {
        let (nodes, root) = parser::parse_source("(add x 1)").unwrap();
        assert!(body_size(&nodes, root) >= 3); // app + symbol + num at minimum
    }

    #[test]
    fn test_load_library_empty() {
        let path = "/tmp/selph_test_empty.selph";
        std::fs::write(path, "").unwrap();
        let loaded = load_library(path).unwrap();
        assert!(loaded.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_replace_identifier() {
        assert_eq!(replace_identifier("(add x x)", "x", "s"), "(add s s)");
        // Should not replace substring matches
        assert_eq!(
            replace_identifier("(add xx x)", "x", "s"),
            "(add xx s)"
        );
    }
}
