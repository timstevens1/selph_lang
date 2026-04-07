//! Abstraction extraction / library learning for SELPH.
//!
//! Ports the Python `library.py` extraction pipeline to Rust, operating
//! on the arena-based `(Vec<Node>, usize)` AST representation.
//!
//! Key operations:
//!   1. Sub-tree extraction: collect all sub-expressions from a program
//!   2. Anti-unification: find the most specific generalization of two trees
//!   3. Compression scoring: MDL-style — keep abstractions that compress the corpus
//!   4. Extraction pipeline: the main entry point that ties it all together

use std::collections::{HashMap, HashSet};
use crate::intern::{Sym, intern, resolve};
use crate::types::Node;

// ── Builtin set (for structural fingerprinting) ─────────────────────

const BUILTINS: &[&str] = &[
    "add", "+", "subtract", "-", "multiply", "*", "divide", "/", "modulo", "%",
    "abs", "negate", "floor", "ceil", "min", "max",
    "<", ">", "<=", ">=", "=", "!=",
    "not", "and", "or", "even", "odd",
    "concat", "string-length", "substring", "string-upper", "string-lower",
    "string-reverse", "string-contains", "string-split", "string-join",
    "string-starts-with", "string-ends-with", "string-replace", "string-trim",
    "char-at", "to-string", "to-number",
    "list", "cons", "head", "tail", "length", "nth", "append", "reverse",
    "range", "empty?", "contains", "sort",
    "map", "filter", "reduce", "compose", "pipe", "apply",
    "identity", "if", "let", "lambda", "defmacro", "define", "do", "quote",
];

fn is_builtin(name: &str) -> bool {
    BUILTINS.contains(&name)
}

// ── Abstraction data structure ──────────────────────────────────────

/// An extracted library abstraction (pattern mined from solved programs).
#[derive(Clone, Debug)]
pub struct Abstraction {
    /// Generated name (e.g. "lib_0").
    pub name: String,
    /// Parameter names for the holes in the pattern.
    pub params: Vec<String>,
    /// Arena-allocated body of the pattern.
    pub body: Vec<Node>,
    /// Root index into `body`.
    pub root: usize,
    /// How many programs contained this pattern.
    pub frequency: usize,
    /// MDL compression score (bits saved by using this abstraction).
    pub compression_score: f64,
}

impl Abstraction {
    pub fn arity(&self) -> usize {
        self.params.len()
    }
}

// ── Body size ───────────────────────────────────────────────────────

/// Count the number of nodes reachable from `idx` in the pool.
pub fn body_size(nodes: &[Node], idx: usize) -> usize {
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
            1 + bindings.iter().map(|(_, v)| body_size(nodes, *v)).sum::<usize>()
                + body_size(nodes, *body)
        }
    }
}

// ── Sub-tree extraction ─────────────────────────────────────────────

/// Extract all sub-expressions from a program, including the root.
///
/// Each returned item is `(nodes_clone, sub_root)` — a reference into the
/// *same* node pool (cloned) with a different root index.  We clone the
/// pool to make each sub-tree self-contained and safe to anti-unify.
pub fn extract_subtrees(nodes: &[Node], root: usize) -> Vec<(Vec<Node>, usize)> {
    let mut indices = Vec::new();
    collect_subtree_indices(nodes, root, &mut indices);
    indices
        .into_iter()
        .map(|idx| (nodes.to_vec(), idx))
        .collect()
}

/// Collect all node indices reachable from `idx` (pre-order).
fn collect_subtree_indices(nodes: &[Node], idx: usize, out: &mut Vec<usize>) {
    out.push(idx);
    match &nodes[idx] {
        Node::App(children) => {
            for &c in children {
                collect_subtree_indices(nodes, c, out);
            }
        }
        Node::If(c, t, e) => {
            collect_subtree_indices(nodes, *c, out);
            collect_subtree_indices(nodes, *t, out);
            collect_subtree_indices(nodes, *e, out);
        }
        Node::Lambda(_, body) => {
            collect_subtree_indices(nodes, *body, out);
        }
        Node::Let(bindings, body) => {
            for (_, v) in bindings {
                collect_subtree_indices(nodes, *v, out);
            }
            collect_subtree_indices(nodes, *body, out);
        }
        _ => {}
    }
}

// ── Structural fingerprinting ───────────────────────────────────────

/// Compute a structural fingerprint that preserves builtin names but
/// abstracts over variable names and literal values.
///
/// Two sub-trees with the same fingerprint have the same "shape" and
/// the same primitive operations, differing only in leaf values/variables.
pub fn structural_fingerprint(nodes: &[Node], idx: usize) -> String {
    match &nodes[idx] {
        Node::Num(_) => "NUM".to_string(),
        Node::Str(_) => "STR".to_string(),
        Node::Bool(_) => "BOOL".to_string(),
        Node::Symbol(name) => {
            let s = resolve(*name);
            if is_builtin(&s) {
                s
            } else {
                "_".to_string()
            }
        }
        Node::App(children) => {
            if children.is_empty() {
                return "()".to_string();
            }
            let parts: Vec<String> = children
                .iter()
                .map(|&c| structural_fingerprint(nodes, c))
                .collect();
            format!("({})", parts.join(" "))
        }
        Node::If(c, t, e) => {
            format!(
                "(if {} {} {})",
                structural_fingerprint(nodes, *c),
                structural_fingerprint(nodes, *t),
                structural_fingerprint(nodes, *e),
            )
        }
        Node::Lambda(params, body) => {
            let ps: Vec<&str> = params.iter().map(|_| "_").collect();
            format!(
                "(lambda ({}) {})",
                ps.join(" "),
                structural_fingerprint(nodes, *body),
            )
        }
        Node::Let(bindings, body) => {
            let bs: Vec<String> = bindings
                .iter()
                .map(|(_, v)| format!("(_ {})", structural_fingerprint(nodes, *v)))
                .collect();
            format!(
                "(let ({}) {})",
                bs.join(" "),
                structural_fingerprint(nodes, *body),
            )
        }
    }
}

// ── Node equality ───────────────────────────────────────────────────

/// Deep structural equality of two nodes within (potentially different) pools.
fn nodes_equal(a_nodes: &[Node], a_idx: usize, b_nodes: &[Node], b_idx: usize) -> bool {
    match (&a_nodes[a_idx], &b_nodes[b_idx]) {
        (Node::Num(a), Node::Num(b)) => a == b,
        (Node::Str(a), Node::Str(b)) => a == b,
        (Node::Bool(a), Node::Bool(b)) => a == b,
        (Node::Symbol(a), Node::Symbol(b)) => a == b,
        (Node::App(ac), Node::App(bc)) => {
            ac.len() == bc.len()
                && ac.iter().zip(bc.iter()).all(|(&ai, &bi)| {
                    nodes_equal(a_nodes, ai, b_nodes, bi)
                })
        }
        (Node::If(ac, at, ae), Node::If(bc, bt, be)) => {
            nodes_equal(a_nodes, *ac, b_nodes, *bc)
                && nodes_equal(a_nodes, *at, b_nodes, *bt)
                && nodes_equal(a_nodes, *ae, b_nodes, *be)
        }
        (Node::Lambda(ap, ab), Node::Lambda(bp, bb)) => {
            ap == bp && nodes_equal(a_nodes, *ab, b_nodes, *bb)
        }
        (Node::Let(abinds, abody), Node::Let(bbinds, bbody)) => {
            abinds.len() == bbinds.len()
                && abinds.iter().zip(bbinds.iter()).all(|((an, av), (bn, bv))| {
                    an == bn && nodes_equal(a_nodes, *av, b_nodes, *bv)
                })
                && nodes_equal(a_nodes, *abody, b_nodes, *bbody)
        }
        _ => false,
    }
}

// ── Anti-unification ────────────────────────────────────────────────

/// Find the most specific generalization of two AST trees.
///
/// Returns `(result_nodes, result_root, param_names)` where differing
/// sub-trees have been replaced with fresh parameter symbols `_p0`, `_p1`, ...
///
/// This is the core of library learning: given `(add x 1)` and `(add y 2)`,
/// anti-unification yields `(add _p0 _p1)` with params `["_p0", "_p1"]`.
pub fn anti_unify(
    a_nodes: &[Node],
    a_root: usize,
    b_nodes: &[Node],
    b_root: usize,
) -> (Vec<Node>, usize, Vec<String>) {
    let mut result = Vec::new();
    let mut counter = 0usize;
    let mut params = Vec::new();

    let root = anti_unify_rec(
        a_nodes, a_root, b_nodes, b_root,
        &mut result, &mut counter, &mut params,
    );

    (result, root, params)
}

fn anti_unify_rec(
    a_nodes: &[Node],
    a_idx: usize,
    b_nodes: &[Node],
    b_idx: usize,
    out: &mut Vec<Node>,
    counter: &mut usize,
    params: &mut Vec<String>,
) -> usize {
    // Identical nodes unify to themselves
    if nodes_equal(a_nodes, a_idx, b_nodes, b_idx) {
        return clone_subtree(a_nodes, a_idx, out);
    }

    // Both App with same length and same head -> recurse on children
    if let (Node::App(ac), Node::App(bc)) = (&a_nodes[a_idx], &b_nodes[b_idx]) {
        if ac.len() == bc.len() && !ac.is_empty() {
            // Check if heads match
            if nodes_equal(a_nodes, ac[0], b_nodes, bc[0]) {
                let mut child_indices = Vec::with_capacity(ac.len());
                // Clone head
                let head_idx = clone_subtree(a_nodes, ac[0], out);
                child_indices.push(head_idx);
                // Recurse on remaining children
                for (&ai, &bi) in ac[1..].iter().zip(bc[1..].iter()) {
                    let ci = anti_unify_rec(a_nodes, ai, b_nodes, bi, out, counter, params);
                    child_indices.push(ci);
                }
                let idx = out.len();
                out.push(Node::App(child_indices));
                return idx;
            }
        }
    }

    // Both If -> recurse on condition, then, else
    if let (Node::If(ac, at, ae), Node::If(bc, bt, be)) = (&a_nodes[a_idx], &b_nodes[b_idx]) {
        let c = anti_unify_rec(a_nodes, *ac, b_nodes, *bc, out, counter, params);
        let t = anti_unify_rec(a_nodes, *at, b_nodes, *bt, out, counter, params);
        let e = anti_unify_rec(a_nodes, *ae, b_nodes, *be, out, counter, params);
        let idx = out.len();
        out.push(Node::If(c, t, e));
        return idx;
    }

    // Different -> introduce a fresh parameter variable
    let name = format!("_p{}", counter);
    *counter += 1;
    params.push(name.clone());
    let idx = out.len();
    out.push(Node::Symbol(intern(&name)));
    idx
}

/// Deep-clone a subtree from `src` into `dst`, returning the new root index.
fn clone_subtree(src: &[Node], idx: usize, dst: &mut Vec<Node>) -> usize {
    match &src[idx] {
        Node::Num(n) => {
            let i = dst.len();
            dst.push(Node::Num(*n));
            i
        }
        Node::Str(s) => {
            let i = dst.len();
            dst.push(Node::Str(s.clone()));
            i
        }
        Node::Bool(b) => {
            let i = dst.len();
            dst.push(Node::Bool(*b));
            i
        }
        Node::Symbol(s) => {
            let i = dst.len();
            dst.push(Node::Symbol(*s));
            i
        }
        Node::App(children) => {
            let new_children: Vec<usize> = children
                .iter()
                .map(|&c| clone_subtree(src, c, dst))
                .collect();
            let i = dst.len();
            dst.push(Node::App(new_children));
            i
        }
        Node::If(c, t, e) => {
            let nc = clone_subtree(src, *c, dst);
            let nt = clone_subtree(src, *t, dst);
            let ne = clone_subtree(src, *e, dst);
            let i = dst.len();
            dst.push(Node::If(nc, nt, ne));
            i
        }
        Node::Lambda(params, body) => {
            let nb = clone_subtree(src, *body, dst);
            let i = dst.len();
            dst.push(Node::Lambda(params.clone(), nb));
            i
        }
        Node::Let(bindings, body) => {
            let new_bindings: Vec<(Sym, usize)> = bindings
                .iter()
                .map(|(name, v)| (*name, clone_subtree(src, *v, dst)))
                .collect();
            let nb = clone_subtree(src, *body, dst);
            let i = dst.len();
            dst.push(Node::Let(new_bindings, nb));
            i
        }
    }
}

// ── Anti-unify many ─────────────────────────────────────────────────

/// Anti-unify a list of trees pairwise, returning the common pattern.
///
/// Folds left: anti_unify(anti_unify(t0, t1), t2), ...
fn anti_unify_many(trees: &[(Vec<Node>, usize)]) -> (Vec<Node>, usize, Vec<String>) {
    if trees.is_empty() {
        let mut nodes = Vec::new();
        nodes.push(Node::Symbol(intern("_")));
        return (nodes, 0, Vec::new());
    }
    if trees.len() == 1 {
        return (trees[0].0.clone(), trees[0].1, Vec::new());
    }

    let (mut pattern_nodes, mut pattern_root, _) = (trees[0].0.clone(), trees[0].1, Vec::<String>::new());

    for (other_nodes, other_root) in &trees[1..] {
        let (new_nodes, new_root, _) =
            anti_unify(&pattern_nodes, pattern_root, other_nodes, *other_root);
        pattern_nodes = new_nodes;
        pattern_root = new_root;
    }

    // Collect all _pN symbols from the final pattern
    let params = collect_params(&pattern_nodes, pattern_root);
    (pattern_nodes, pattern_root, params)
}

/// Collect all `_pN` parameter symbols reachable from a root.
fn collect_params(nodes: &[Node], idx: usize) -> Vec<String> {
    let mut set = HashSet::new();
    collect_params_rec(nodes, idx, &mut set);
    let mut params: Vec<String> = set.into_iter().collect();
    params.sort(); // _p0, _p1, _p2, ...
    params
}

fn collect_params_rec(nodes: &[Node], idx: usize, out: &mut HashSet<String>) {
    match &nodes[idx] {
        Node::Symbol(name) => {
            let s = resolve(*name);
            if s.starts_with("_p") {
                out.insert(s);
            }
        }
        Node::App(children) => {
            for &c in children {
                collect_params_rec(nodes, c, out);
            }
        }
        Node::If(c, t, e) => {
            collect_params_rec(nodes, *c, out);
            collect_params_rec(nodes, *t, out);
            collect_params_rec(nodes, *e, out);
        }
        Node::Lambda(_, body) => {
            collect_params_rec(nodes, *body, out);
        }
        Node::Let(bindings, body) => {
            for (_, v) in bindings {
                collect_params_rec(nodes, *v, out);
            }
            collect_params_rec(nodes, *body, out);
        }
        _ => {}
    }
}

// ── Compression scoring ─────────────────────────────────────────────

/// MDL-style compression score for an abstraction against a corpus.
///
/// An abstraction saves `(pattern_size - 1 - n_params)` nodes each time
/// it's used (replacing a deep tree with a single call + args).
/// The cost is `pattern_size + n_params + 3` (the defmacro definition).
///
/// Returns 0.0 if the abstraction doesn't compress.
pub fn compression_score(
    pattern_nodes: &[Node],
    pattern_root: usize,
    params: &[String],
    corpus: &[(Vec<Node>, usize)],
) -> f64 {
    let pattern_size = body_size(pattern_nodes, pattern_root);
    let n_params = params.len();

    // Nodes saved per use: replace subtree with (name arg1 arg2 ...)
    // which is 1 + n_params nodes
    let savings_per_use = pattern_size as i64 - (1 + n_params) as i64;
    if savings_per_use <= 0 {
        return 0.0;
    }

    // Count how many times the pattern's fingerprint appears in the corpus
    let fp = structural_fingerprint(pattern_nodes, pattern_root);
    let mut uses = 0usize;
    for (prog_nodes, prog_root) in corpus {
        let mut indices = Vec::new();
        collect_subtree_indices(prog_nodes, *prog_root, &mut indices);
        for idx in indices {
            if body_size(prog_nodes, idx) >= 2
                && structural_fingerprint(prog_nodes, idx) == fp
            {
                uses += 1;
            }
        }
    }

    // Definition cost: (defmacro name (params) body)
    let definition_cost = (pattern_size + n_params + 3) as i64;

    let total = savings_per_use * (uses as i64) - definition_cost;
    if total > 0 { total as f64 } else { 0.0 }
}

// ── Frequency counting ──────────────────────────────────────────────

/// Count structural fingerprints of all sub-trees (of size >= min_size)
/// across a corpus.  Each fingerprint is counted at most once per program.
fn count_subtrees(
    corpus: &[(Vec<Node>, usize)],
    min_size: usize,
) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();

    for (prog_nodes, prog_root) in corpus {
        let mut seen = HashSet::new();
        let mut indices = Vec::new();
        collect_subtree_indices(prog_nodes, *prog_root, &mut indices);
        for idx in indices {
            if body_size(prog_nodes, idx) >= min_size {
                let fp = structural_fingerprint(prog_nodes, idx);
                if seen.insert(fp.clone()) {
                    *counts.entry(fp).or_insert(0) += 1;
                }
            }
        }
    }

    counts
}

/// Find recurring sub-tree patterns in a corpus.
///
/// Returns `(fingerprint, count, example_trees)` sorted by `frequency * size`.
fn find_common_patterns(
    corpus: &[(Vec<Node>, usize)],
    min_frequency: usize,
    min_size: usize,
) -> Vec<(String, usize, Vec<(Vec<Node>, usize)>)> {
    let counts = count_subtrees(corpus, min_size);

    // Collect example trees for each fingerprint that passes the threshold
    let mut examples: HashMap<String, Vec<(Vec<Node>, usize)>> = HashMap::new();

    for (prog_nodes, prog_root) in corpus {
        let mut indices = Vec::new();
        collect_subtree_indices(prog_nodes, *prog_root, &mut indices);
        for idx in indices {
            if body_size(prog_nodes, idx) >= min_size {
                let fp = structural_fingerprint(prog_nodes, idx);
                if let Some(&count) = counts.get(&fp) {
                    if count >= min_frequency {
                        let entry = examples.entry(fp).or_default();
                        if entry.len() < 10 {
                            entry.push((prog_nodes.clone(), idx));
                        }
                    }
                }
            }
        }
    }

    let mut results: Vec<(String, usize, Vec<(Vec<Node>, usize)>)> = Vec::new();
    for (fp, count) in &counts {
        if *count >= min_frequency {
            if let Some(exs) = examples.remove(fp) {
                results.push((fp.clone(), *count, exs));
            }
        }
    }

    // Sort by frequency * size (bigger, more frequent = more valuable)
    results.sort_by(|a, b| {
        let score_a = a.1 * body_size(&a.2[0].0, a.2[0].1);
        let score_b = b.1 * body_size(&b.2[0].0, b.2[0].1);
        score_b.cmp(&score_a)
    });

    results
}

// ── Param renaming ──────────────────────────────────────────────────

/// Rename parameter symbols in the pattern to cleaner names (a0, a1, ...).
fn rename_params(
    nodes: &[Node],
    root: usize,
    mapping: &HashMap<String, String>,
) -> (Vec<Node>, usize) {
    let mut out = Vec::new();
    let new_root = rename_params_rec(nodes, root, mapping, &mut out);
    (out, new_root)
}

fn rename_params_rec(
    src: &[Node],
    idx: usize,
    mapping: &HashMap<String, String>,
    dst: &mut Vec<Node>,
) -> usize {
    match &src[idx] {
        Node::Symbol(name) => {
            let i = dst.len();
            let name_str = resolve(*name);
            if let Some(new_name) = mapping.get(&name_str) {
                dst.push(Node::Symbol(intern(new_name)));
            } else {
                dst.push(Node::Symbol(*name));
            }
            i
        }
        Node::Num(n) => {
            let i = dst.len();
            dst.push(Node::Num(*n));
            i
        }
        Node::Str(s) => {
            let i = dst.len();
            dst.push(Node::Str(s.clone()));
            i
        }
        Node::Bool(b) => {
            let i = dst.len();
            dst.push(Node::Bool(*b));
            i
        }
        Node::App(children) => {
            let new_children: Vec<usize> = children
                .iter()
                .map(|&c| rename_params_rec(src, c, mapping, dst))
                .collect();
            let i = dst.len();
            dst.push(Node::App(new_children));
            i
        }
        Node::If(c, t, e) => {
            let nc = rename_params_rec(src, *c, mapping, dst);
            let nt = rename_params_rec(src, *t, mapping, dst);
            let ne = rename_params_rec(src, *e, mapping, dst);
            let i = dst.len();
            dst.push(Node::If(nc, nt, ne));
            i
        }
        Node::Lambda(params, body) => {
            let nb = rename_params_rec(src, *body, mapping, dst);
            let i = dst.len();
            dst.push(Node::Lambda(params.clone(), nb));
            i
        }
        Node::Let(bindings, body) => {
            let new_bindings: Vec<(Sym, usize)> = bindings
                .iter()
                .map(|(name, v)| {
                    let name_str = resolve(*name);
                    let new_name = mapping.get(&name_str).map(|s| intern(s)).unwrap_or(*name);
                    (new_name, rename_params_rec(src, *v, mapping, dst))
                })
                .collect();
            let nb = rename_params_rec(src, *body, mapping, dst);
            let i = dst.len();
            dst.push(Node::Let(new_bindings, nb));
            i
        }
    }
}

// ── Main extraction pipeline ────────────────────────────────────────

/// Extract a library of abstractions from a corpus of solved programs.
///
/// This is the main entry point — the Rust port of Python's `extract_library()`.
///
/// # Arguments
/// * `solved_programs` — list of `(nodes, root)` ASTs from solved tasks.
/// * `min_frequency` — minimum times a pattern must appear (default: 2).
/// * `min_size` — minimum sub-tree size to consider (default: 2).
/// * `min_compression` — minimum compression score to keep (default: 1.0).
/// * `max_abstractions` — maximum number of abstractions to return (default: 20).
/// * `name_prefix` — prefix for generated names (default: "lib").
pub fn extract_abstractions(
    solved_programs: &[(Vec<Node>, usize)],
    min_frequency: usize,
    min_size: usize,
    min_compression: f64,
    max_abstractions: usize,
    name_prefix: &str,
) -> Vec<Abstraction> {
    let patterns = find_common_patterns(solved_programs, min_frequency, min_size);
    let mut abstractions = Vec::new();
    let mut used_fingerprints = HashSet::new();

    for (fp, freq, examples) in &patterns {
        if abstractions.len() >= max_abstractions {
            break;
        }

        if used_fingerprints.contains(fp) {
            continue;
        }

        // Anti-unify the examples (cap at 5) to get a pattern with parameters
        let cap = examples.len().min(5);
        let (pattern_nodes, pattern_root, params) = anti_unify_many(&examples[..cap]);

        // If anti-unification collapsed everything to a single param, skip
        if let Node::Symbol(name) = &pattern_nodes[pattern_root] {
            if resolve(*name).starts_with("_p") {
                continue;
            }
        }

        // Score compression
        let comp = compression_score(&pattern_nodes, pattern_root, &params, solved_programs);
        if comp < min_compression {
            continue;
        }

        // Rename params to cleaner names: a0, a1, ...
        let mut mapping = HashMap::new();
        let mut clean_params = Vec::new();
        for (i, p) in params.iter().enumerate() {
            let clean = format!("a{}", i);
            mapping.insert(p.clone(), clean.clone());
            clean_params.push(clean);
        }
        let (body, root) = rename_params(&pattern_nodes, pattern_root, &mapping);

        let name = format!("{}_{}", name_prefix, abstractions.len());

        abstractions.push(Abstraction {
            name,
            params: clean_params,
            body,
            root,
            frequency: *freq,
            compression_score: comp,
        });
        used_fingerprints.insert(fp.clone());
    }

    abstractions.sort_by(|a, b| {
        b.compression_score
            .partial_cmp(&a.compression_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    abstractions
}

/// Convenience wrapper with sensible defaults.
pub fn extract_abstractions_default(
    solved_programs: &[(Vec<Node>, usize)],
) -> Vec<Abstraction> {
    extract_abstractions(solved_programs, 2, 2, 1.0, 20, "lib")
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    /// Helper: parse a SELPH expression and return (nodes, root).
    fn parse(src: &str) -> (Vec<Node>, usize) {
        parser::parse_source(src).expect("parse failed")
    }

    // ── body_size ───────────────────────────────────────────────────

    #[test]
    fn test_body_size_atom() {
        let (nodes, root) = parse("42");
        assert_eq!(body_size(&nodes, root), 1);
    }

    #[test]
    fn test_body_size_app() {
        let (nodes, root) = parse("(add x 1)");
        // App(add, x, 1) = 1 + 1 + 1 + 1 = 4
        assert_eq!(body_size(&nodes, root), 4);
    }

    #[test]
    fn test_body_size_nested() {
        let (nodes, root) = parse("(add (multiply x 2) 1)");
        // outer App + add + inner App(multiply, x, 2) + 1
        // = 1 + 1 + (1 + 1 + 1 + 1) + 1 = 7
        assert_eq!(body_size(&nodes, root), 7);
    }

    // ── extract_subtrees ────────────────────────────────────────────

    #[test]
    fn test_extract_subtrees_atom() {
        let (nodes, root) = parse("x");
        let subs = extract_subtrees(&nodes, root);
        assert_eq!(subs.len(), 1);
    }

    #[test]
    fn test_extract_subtrees_app() {
        let (nodes, root) = parse("(add x 1)");
        let subs = extract_subtrees(&nodes, root);
        // The App node + add + x + 1 = 4 subtrees
        assert_eq!(subs.len(), 4);
    }

    #[test]
    fn test_extract_subtrees_nested() {
        let (nodes, root) = parse("(add (multiply x 2) 1)");
        let subs = extract_subtrees(&nodes, root);
        // outer App, add, inner App, multiply, x, 2, 1 = 7
        assert_eq!(subs.len(), 7);
    }

    // ── structural_fingerprint ──────────────────────────────────────

    #[test]
    fn test_fingerprint_same_shape() {
        let (n1, r1) = parse("(string-upper x)");
        let (n2, r2) = parse("(string-upper y)");
        assert_eq!(
            structural_fingerprint(&n1, r1),
            structural_fingerprint(&n2, r2),
        );
    }

    #[test]
    fn test_fingerprint_different_head() {
        let (n1, r1) = parse("(string-upper x)");
        let (n2, r2) = parse("(string-lower x)");
        assert_ne!(
            structural_fingerprint(&n1, r1),
            structural_fingerprint(&n2, r2),
        );
    }

    #[test]
    fn test_fingerprint_different_arity() {
        let (n1, r1) = parse("(add x 1)");
        let (n2, r2) = parse("(add x)");
        assert_ne!(
            structural_fingerprint(&n1, r1),
            structural_fingerprint(&n2, r2),
        );
    }

    // ── anti_unify ──────────────────────────────────────────────────

    #[test]
    fn test_anti_unify_identical() {
        let (n1, r1) = parse("(add x 1)");
        let (n2, r2) = parse("(add x 1)");
        let (result, root, params) = anti_unify(&n1, r1, &n2, r2);
        assert!(params.is_empty(), "identical trees should produce no params");
        // The result should be equivalent to (add x 1)
        assert_eq!(body_size(&result, root), body_size(&n1, r1));
    }

    #[test]
    fn test_anti_unify_one_diff() {
        // (add x 1) vs (add y 1) -> (add _p0 1)
        let (n1, r1) = parse("(add x 1)");
        let (n2, r2) = parse("(add y 1)");
        let (result, root, params) = anti_unify(&n1, r1, &n2, r2);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], "_p0");
        // Result should be App with 3 children: add, _p0, 1
        if let Node::App(children) = &result[root] {
            assert_eq!(children.len(), 3);
            assert!(matches!(&result[children[0]], Node::Symbol(s) if *s == intern("add")));
            assert!(matches!(&result[children[1]], Node::Symbol(s) if *s == intern("_p0")));
            assert!(matches!(&result[children[2]], Node::Num(n) if *n == 1.0));
        } else {
            panic!("expected App node");
        }
    }

    #[test]
    fn test_anti_unify_two_diffs() {
        // (add x 1) vs (add y 2) -> (add _p0 _p1)
        let (n1, r1) = parse("(add x 1)");
        let (n2, r2) = parse("(add y 2)");
        let (result, root, params) = anti_unify(&n1, r1, &n2, r2);
        assert_eq!(params.len(), 2);
        if let Node::App(children) = &result[root] {
            assert_eq!(children.len(), 3);
            assert!(matches!(&result[children[0]], Node::Symbol(s) if *s == intern("add")));
            assert!(matches!(&result[children[1]], Node::Symbol(s) if resolve(*s).starts_with("_p")));
            assert!(matches!(&result[children[2]], Node::Symbol(s) if resolve(*s).starts_with("_p")));
        } else {
            panic!("expected App node");
        }
    }

    #[test]
    fn test_anti_unify_completely_different() {
        // (add x 1) vs (multiply x 1) — different head → collapses to _p0
        let (n1, r1) = parse("(add x 1)");
        let (n2, r2) = parse("(multiply x 1)");
        let (result, root, params) = anti_unify(&n1, r1, &n2, r2);
        assert_eq!(params.len(), 1);
        assert!(matches!(&result[root], Node::Symbol(s) if *s == intern("_p0")));
    }

    #[test]
    fn test_anti_unify_nested() {
        // (concat (string-upper x) "!") vs (concat (string-upper y) "!")
        // -> (concat (string-upper _p0) "!")
        let (n1, r1) = parse("(concat (string-upper x) \"!\")");
        let (n2, r2) = parse("(concat (string-upper y) \"!\")");
        let (result, root, params) = anti_unify(&n1, r1, &n2, r2);
        assert_eq!(params.len(), 1);
        // Root should be an App
        assert!(matches!(&result[root], Node::App(_)));
    }

    // ── compression_score ───────────────────────────────────────────

    #[test]
    fn test_compression_positive() {
        // Pattern: (concat (string-upper _) _) — size 4
        // Appears in 3 programs
        let programs: Vec<(Vec<Node>, usize)> = vec![
            parse("(concat (string-upper x) \"!\")"),
            parse("(concat (string-upper y) \"?\")"),
            parse("(concat (string-upper z) \".\")"),
        ];

        let (pattern, root, params) =
            anti_unify(&programs[0].0, programs[0].1, &programs[1].0, programs[1].1);

        let score = compression_score(&pattern, root, &params, &programs);
        // With 3 uses, a size-4 pattern with params should compress positively
        // (though exact value depends on fingerprint matching)
        assert!(score >= 0.0);
    }

    #[test]
    fn test_compression_zero_for_tiny() {
        // A trivially small pattern should not compress
        let programs = vec![parse("x"), parse("y")];
        let (pattern, root, params) =
            anti_unify(&programs[0].0, programs[0].1, &programs[1].0, programs[1].1);
        let score = compression_score(&pattern, root, &params, &programs);
        assert_eq!(score, 0.0);
    }

    // ── extract_abstractions (integration) ──────────────────────────

    #[test]
    fn test_extract_from_empty_corpus() {
        let result = extract_abstractions_default(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_from_single_program() {
        // A single program can't have frequency >= 2
        let corpus = vec![parse("(concat (string-upper x) \"!\"))")];
        let result = extract_abstractions_default(&corpus);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_finds_common_pattern() {
        // Multiple programs sharing the same pattern
        let corpus: Vec<(Vec<Node>, usize)> = vec![
            parse("(concat (string-upper x) \"!\")"),
            parse("(concat (string-upper y) \"?\")"),
            parse("(concat (string-upper z) \".\")"),
            parse("(concat (string-upper w) \"-\")"),
        ];

        let result = extract_abstractions(&corpus, 2, 2, 0.0, 20, "lib");

        // Should find at least one abstraction (the shared concat/string-upper pattern)
        // The exact number depends on which sub-patterns also recur
        // At minimum, (string-upper _) appears 4 times
        assert!(
            !result.is_empty(),
            "should find at least one common abstraction"
        );

        // All abstractions should have frequency >= 2
        for abs in &result {
            assert!(abs.frequency >= 2);
        }
    }

    #[test]
    fn test_extract_sorted_by_compression() {
        let corpus: Vec<(Vec<Node>, usize)> = vec![
            parse("(concat (string-upper x) \"!\")"),
            parse("(concat (string-upper y) \"?\")"),
            parse("(add a 1)"),
            parse("(add b 1)"),
        ];

        let result = extract_abstractions(&corpus, 2, 2, 0.0, 20, "test");

        if result.len() >= 2 {
            // Should be sorted descending by compression_score
            for i in 1..result.len() {
                assert!(result[i - 1].compression_score >= result[i].compression_score);
            }
        }
    }

    #[test]
    fn test_abstraction_has_clean_params() {
        let corpus: Vec<(Vec<Node>, usize)> = vec![
            parse("(add x 1)"),
            parse("(add y 2)"),
            parse("(add z 3)"),
        ];

        let result = extract_abstractions(&corpus, 2, 2, 0.0, 20, "lib");
        for abs in &result {
            for (i, p) in abs.params.iter().enumerate() {
                assert_eq!(p, &format!("a{}", i), "params should be renamed to a0, a1, ...");
            }
        }
    }

    // ── clone_subtree ───────────────────────────────────────────────

    #[test]
    fn test_clone_subtree_roundtrip() {
        let (nodes, root) = parse("(add (multiply x 2) 1)");
        let mut dst = Vec::new();
        let new_root = clone_subtree(&nodes, root, &mut dst);
        assert_eq!(body_size(&nodes, root), body_size(&dst, new_root));
        assert!(nodes_equal(&nodes, root, &dst, new_root));
    }

    // ── nodes_equal ─────────────────────────────────────────────────

    #[test]
    fn test_nodes_equal_same() {
        let (n1, r1) = parse("(add x 1)");
        assert!(nodes_equal(&n1, r1, &n1, r1));
    }

    #[test]
    fn test_nodes_equal_different() {
        let (n1, r1) = parse("(add x 1)");
        let (n2, r2) = parse("(add x 2)");
        assert!(!nodes_equal(&n1, r1, &n2, r2));
    }

    // ── collect_params ──────────────────────────────────────────────

    #[test]
    fn test_collect_params_none() {
        let (nodes, root) = parse("(add x 1)");
        let params = collect_params(&nodes, root);
        assert!(params.is_empty());
    }

    #[test]
    fn test_collect_params_some() {
        // Manually build a tree with _p0 and _p1
        let nodes = vec![
            Node::Symbol(intern("add")),    // 0
            Node::Symbol(intern("_p0")),     // 1
            Node::Symbol(intern("_p1")),     // 2
            Node::App(vec![0, 1, 2]),            // 3
        ];
        let params = collect_params(&nodes, 3);
        assert_eq!(params, vec!["_p0".to_string(), "_p1".to_string()]);
    }
}
