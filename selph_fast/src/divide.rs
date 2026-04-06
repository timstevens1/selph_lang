//! Divide-and-conquer synthesis for SELPH.
//!
//! When a spec has multiple distinct output values, decomposes the problem:
//!   1. Group examples by output value
//!   2. Sort groups by mean input value
//!   3. For each adjacent pair of groups, find a separating condition (bool expr)
//!   4. For each group, synthesize the output expression
//!   5. Compose into nested if-expressions

use std::collections::HashMap;
use crate::types::*;
use crate::eval;
use crate::synth::{self, SynthComponent, SynthPool, remap_node, vals_equal, val_hash};

// ── Public result type ──────────────────────────────────────────────

/// Result of a divide-and-conquer synthesis attempt.
pub struct DivideResult {
    pub found: bool,
    pub nodes: Vec<Node>,
    pub root: usize,
    pub candidates_explored: usize,
}

impl DivideResult {
    fn empty() -> Self {
        DivideResult {
            found: false,
            nodes: Vec::new(),
            root: 0,
            candidates_explored: 0,
        }
    }
}

// ── Public entry point ──────────────────────────────────────────────

/// Attempt divide-and-conquer synthesis.
///
/// Only useful when `expected` contains 2+ distinct output values,
/// suggesting a piecewise/classification task.
///
/// Returns a `DivideResult` whose `nodes`/`root` encode the body
/// (not yet wrapped in a lambda). The caller wraps it in
/// `(lambda (x) body)` if needed.
pub fn divide_and_conquer(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> DivideResult {
    if inputs.is_empty() || expected.is_empty() {
        return DivideResult::empty();
    }

    // Group example indices by output value (using val_hash for keys).
    let mut groups: Vec<(Value, Vec<usize>)> = Vec::new();
    let mut key_map: HashMap<u64, usize> = HashMap::new();

    for (i, out) in expected.iter().enumerate() {
        let h = val_hash(out);
        // Need to handle hash collisions: verify actual equality.
        let mut found_group = false;
        if let Some(&gi) = key_map.get(&h) {
            if vals_equal(&groups[gi].0, out) {
                groups[gi].1.push(i);
                found_group = true;
            }
        }
        if !found_group {
            // Check all groups for true equality (collision safety).
            let mut matched = false;
            for (gi, (rep, indices)) in groups.iter_mut().enumerate() {
                if vals_equal(rep, out) {
                    indices.push(i);
                    key_map.insert(h, gi);
                    matched = true;
                    break;
                }
            }
            if !matched {
                key_map.insert(h, groups.len());
                groups.push((out.clone(), vec![i]));
            }
        }
    }

    // Only useful with 2+ distinct output values.
    if groups.len() < 2 {
        return DivideResult::empty();
    }

    // Sort groups by mean input value (gives natural threshold ordering).
    groups.sort_by(|a, b| {
        let ma = mean_input(&a.1, inputs);
        let mb = mean_input(&b.1, inputs);
        ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_groups: Vec<(Value, Vec<usize>)> = groups;

    // Build the nested if-expression.
    let mut total_explored: usize = 0;
    let body = build_nested_if(
        &sorted_groups,
        inputs,
        expected,
        components,
        macros,
        max_depth,
        max_candidates,
        &mut total_explored,
    );

    match body {
        Some((nodes, root)) => {
            // Verify the full program against all examples.
            let mut ln = nodes.clone();
            let lr = ln.len();
            ln.push(Node::Lambda(vec!["x".into()], root));

            let all_ok = verify_program(&ln, lr, inputs, expected, macros);

            if all_ok {
                DivideResult {
                    found: true,
                    nodes: ln,
                    root: lr,
                    candidates_explored: total_explored,
                }
            } else {
                DivideResult {
                    found: false,
                    nodes: Vec::new(),
                    root: 0,
                    candidates_explored: total_explored,
                }
            }
        }
        None => DivideResult {
            found: false,
            nodes: Vec::new(),
            root: 0,
            candidates_explored: total_explored,
        },
    }
}

// ── Internal helpers ────────────────────────────────────────────────

/// Mean of input values at given indices (for ordering groups).
fn mean_input(indices: &[usize], inputs: &[Value]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for &i in indices {
        if let Value::Num(n) = &inputs[i] {
            sum += n;
            count += 1;
        }
    }
    if count > 0 { sum / count as f64 } else { 0.0 }
}

/// Verify a lambda program against all input/output examples.
fn verify_program(
    nodes: &[Node],
    lambda_root: usize,
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> bool {
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        let mut env = eval::make_default_env();
        for (nm, ps, mn, mr) in macros {
            env_define(
                &mut env,
                nm.clone(),
                Value::RustMacro(ps.clone(), mn.clone(), *mr),
            );
        }
        let fv = match eval::eval(nodes, lambda_root, &mut env) {
            Ok(v) => v,
            Err(_) => return false,
        };
        match eval::apply(&fv, &[inp.clone()], nodes, &mut env) {
            Ok(ref a) if vals_equal(a, exp) => {}
            _ => return false,
        }
    }
    true
}

/// Recursively build nested if-expressions for sorted output groups.
///
/// For 1 group: synthesize a branch expression.
/// For 2+ groups: `(if condition then_expr else_expr)` where
///   else_expr recurses on the remaining groups.
fn build_nested_if(
    sorted_groups: &[(Value, Vec<usize>)],
    inputs: &[Value],
    expected: &[Value],
    components: &[SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    total_explored: &mut usize,
) -> Option<(Vec<Node>, usize)> {
    if sorted_groups.len() == 1 {
        let (out_val, indices) = &sorted_groups[0];
        return synthesize_branch(
            out_val, indices, inputs, expected,
            components, macros, max_depth, max_candidates, total_explored,
        );
    }

    // Split: first group vs rest.
    let (first_val, first_indices) = &sorted_groups[0];
    let rest_groups = &sorted_groups[1..];
    let rest_indices: Vec<usize> = rest_groups.iter()
        .flat_map(|(_, idx)| idx.iter().copied())
        .collect();

    // Try to find a separator: true for first_indices, false for rest_indices.
    let cond = find_separator(
        first_indices, &rest_indices, inputs,
        components, macros, max_depth, max_candidates, total_explored,
    );

    if let Some((cond_nodes, cond_root)) = cond {
        // Build then-branch (first group).
        let then_branch = synthesize_branch(
            first_val, first_indices, inputs, expected,
            components, macros, max_depth, max_candidates, total_explored,
        );
        if then_branch.is_none() {
            return None;
        }
        let (then_nodes, then_root) = then_branch.unwrap();

        // Build else-branch (rest groups, recursively).
        let else_branch = build_nested_if(
            rest_groups, inputs, expected,
            components, macros, max_depth, max_candidates, total_explored,
        );
        if else_branch.is_none() {
            return None;
        }
        let (else_nodes, else_root) = else_branch.unwrap();

        // Merge: cond_nodes + then_nodes + else_nodes + If node.
        return Some(merge_if(
            cond_nodes, cond_root,
            then_nodes, then_root,
            else_nodes, else_root,
        ));
    }

    // Try the other way: true for rest_indices, false for first_indices.
    let cond_rev = find_separator(
        &rest_indices, first_indices, inputs,
        components, macros, max_depth, max_candidates, total_explored,
    );

    if let Some((cond_nodes, cond_root)) = cond_rev {
        // Swapped: then-branch is the rest groups, else-branch is first group.
        let then_branch = build_nested_if(
            rest_groups, inputs, expected,
            components, macros, max_depth, max_candidates, total_explored,
        );
        if then_branch.is_none() {
            return None;
        }
        let (then_nodes, then_root) = then_branch.unwrap();

        let else_branch = synthesize_branch(
            first_val, first_indices, inputs, expected,
            components, macros, max_depth, max_candidates, total_explored,
        );
        if else_branch.is_none() {
            return None;
        }
        let (else_nodes, else_root) = else_branch.unwrap();

        return Some(merge_if(
            cond_nodes, cond_root,
            then_nodes, then_root,
            else_nodes, else_root,
        ));
    }

    None
}

/// Merge condition, then-branch, and else-branch node pools into a
/// single node pool with an If node at the root.
fn merge_if(
    cond_nodes: Vec<Node>, cond_root: usize,
    then_nodes: Vec<Node>, then_root: usize,
    else_nodes: Vec<Node>, else_root: usize,
) -> (Vec<Node>, usize) {
    let mut nodes = cond_nodes;

    let then_off = nodes.len();
    for nd in &then_nodes {
        nodes.push(remap_node(nd, then_off));
    }
    let remapped_then_root = then_root + then_off;

    let else_off = nodes.len();
    for nd in &else_nodes {
        nodes.push(remap_node(nd, else_off));
    }
    let remapped_else_root = else_root + else_off;

    let if_idx = nodes.len();
    nodes.push(Node::If(cond_root, remapped_then_root, remapped_else_root));

    (nodes, if_idx)
}

/// Find a boolean expression that is True for `true_indices` and
/// False for `false_indices`.
///
/// Generates candidate boolean programs via bottom-up enumeration
/// and tests each one.
fn find_separator(
    true_indices: &[usize],
    false_indices: &[usize],
    inputs: &[Value],
    components: &[SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    total_explored: &mut usize,
) -> Option<(Vec<Node>, usize)> {
    let macro_env: Vec<(String, Vec<String>, Vec<Node>, usize)> = macros.to_vec();

    // Build depth-0 atoms from components.
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

    // Test depth-0 bool programs.
    for entry in &pool {
        if entry.ret_type != 2 { continue; } // only bool-typed
        *total_explored += 1;
        if *total_explored > max_candidates { return None; }
        if test_separator(entry, true_indices, false_indices, inputs, &macro_env) {
            return Some((entry.nodes.clone(), entry.root));
        }
    }

    // Depth 1..max_depth: compose programs and check bool-typed ones.
    let mut prev_start: usize = 0;
    let mut prev_end: usize = pool.len();
    const MAX_POOL: usize = 50_000;

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
                    if e.ret_type == 2 {
                        *total_explored += 1;
                        if *total_explored > max_candidates { return None; }
                        if test_separator(&e, true_indices, false_indices, inputs, &macro_env) {
                            return Some((e.nodes.clone(), e.root));
                        }
                    }
                    if pool.len() + new_entries.len() < MAX_POOL {
                        new_entries.push(e);
                    }
                }
            } else if comp.arity == 2 {
                // arg1 from prev layer, arg2 from all
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
                        if e.ret_type == 2 {
                            *total_explored += 1;
                            if *total_explored > max_candidates { return None; }
                            if test_separator(&e, true_indices, false_indices, inputs, &macro_env) {
                                return Some((e.nodes.clone(), e.root));
                            }
                        }
                        if pool.len() + new_entries.len() < MAX_POOL {
                            new_entries.push(e);
                        }
                    }
                }
                // arg1 from old, arg2 from prev layer
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
                        if e.ret_type == 2 {
                            *total_explored += 1;
                            if *total_explored > max_candidates { return None; }
                            if test_separator(&e, true_indices, false_indices, inputs, &macro_env) {
                                return Some((e.nodes.clone(), e.root));
                            }
                        }
                        if pool.len() + new_entries.len() < MAX_POOL {
                            new_entries.push(e);
                        }
                    }
                }
            }
        }

        prev_start = pool.len();
        pool.extend(new_entries);
        prev_end = pool.len();
    }

    None
}

/// Test whether a candidate bool program separates true_indices from
/// false_indices: must evaluate to `true` for every true_index input
/// and `false` for every false_index input.
fn test_separator(
    entry: &SynthPool,
    true_indices: &[usize],
    false_indices: &[usize],
    inputs: &[Value],
    macro_env: &[(String, Vec<String>, Vec<Node>, usize)],
) -> bool {
    let mut ln = entry.nodes.clone();
    let lr = ln.len();
    ln.push(Node::Lambda(vec!["x".into()], entry.root));

    // Check true indices first.
    for &i in true_indices {
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
            Err(_) => return false,
        };
        match eval::apply(&fv, &[inputs[i].clone()], &ln, &mut env) {
            Ok(Value::Bool(true)) => {}
            _ => return false,
        }
    }

    // Check false indices.
    for &i in false_indices {
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
            Err(_) => return false,
        };
        match eval::apply(&fv, &[inputs[i].clone()], &ln, &mut env) {
            Ok(Value::Bool(false)) => {}
            _ => return false,
        }
    }

    true
}

/// Synthesize an expression for a single output group.
///
/// If every example in the group maps to the same constant, return
/// that constant directly. Otherwise, delegate to the enumerative
/// synthesizer on the group's subset of examples.
fn synthesize_branch(
    out_val: &Value,
    indices: &[usize],
    inputs: &[Value],
    expected: &[Value],
    components: &[SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
    total_explored: &mut usize,
) -> Option<(Vec<Node>, usize)> {
    // Check if constant output for this group.
    let group_outputs: Vec<&Value> = indices.iter().map(|&i| &expected[i]).collect();
    let all_same = group_outputs.windows(2).all(|w| vals_equal(w[0], w[1]));

    if all_same {
        // Return a constant node.
        let node = match out_val {
            Value::Num(n) => Node::Num(*n),
            Value::Str(s) => Node::Str(s.clone()),
            Value::Bool(b) => Node::Bool(*b),
            _ => return None,
        };
        return Some((vec![node], 0));
    }

    // Non-constant: synthesize from the group's examples.
    let group_inputs: Vec<Value> = indices.iter().map(|&i| inputs[i].clone()).collect();
    let group_expected: Vec<Value> = indices.iter().map(|&i| expected[i].clone()).collect();

    let result = synth::synthesize(
        components,
        &group_inputs,
        &group_expected,
        macros,
        max_depth,
        max_candidates,
        false, // don't enable if-expressions inside branches
    );

    *total_explored += result.candidates_explored;

    if result.found {
        let nodes = result.nodes.unwrap();
        let root = result.root.unwrap();
        // The synthesizer returns (lambda (x) body). We want just the body,
        // so return the full nodes but with root pointing to the lambda's body.
        // Actually, the lambda is at `root` and the body is the inner expression.
        // We need to extract the body from the lambda node.
        if let Node::Lambda(_, body_root) = &nodes[root] {
            let body_root = *body_root;
            // Return the nodes without the lambda wrapper; root = body_root.
            // The nodes still contain the lambda node but we just don't
            // reference it — the indices remain valid.
            Some((nodes, body_root))
        } else {
            // Shouldn't happen, but fall back to using root directly.
            Some((nodes, root))
        }
    } else {
        None
    }
}
