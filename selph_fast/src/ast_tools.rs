//! AST query and structural editing tools.
//!
//! These CLI tools let an LLM manipulate SELPH code by node index
//! instead of editing raw s-expression text, avoiding paren-matching issues.

use crate::intern::{intern, resolve, Sym};
use crate::parser::{parse_file, parse_source};
use crate::types::{node_to_source, Node};

// ── Data types ──────────────────────────────────────────────────────

pub struct DefInfo {
    pub order: usize,
    pub root_idx: usize,
    pub kind: String,
    pub name: String,
}

// ── Query operations ────────────────────────────────────────────────

/// List top-level definitions with their arena indices.
pub fn list_defs(nodes: &[Node], roots: &[usize]) -> Vec<DefInfo> {
    let mut defs = Vec::new();
    for (i, &root) in roots.iter().enumerate() {
        let (kind, name) = match &nodes[root] {
            Node::App(children) if children.len() >= 2 => {
                let head = match &nodes[children[0]] {
                    Node::Symbol(s) => resolve(*s),
                    _ => continue,
                };
                if head != "define" && head != "defmacro" {
                    continue;
                }
                let name = match &nodes[children[1]] {
                    Node::Symbol(s) => resolve(*s),
                    Node::Str(s) => s.clone(),
                    _ => format!("<node {}>", children[1]),
                };
                (head, name)
            }
            _ => continue,
        };
        defs.push(DefInfo {
            order: i,
            root_idx: root,
            kind,
            name,
        });
    }
    defs
}

/// Render a node as an indented tree with arena indices.
pub fn tree_display(nodes: &[Node], root: usize) -> String {
    let mut out = String::new();
    tree_rec(nodes, root, 0, &mut out);
    out
}

fn tree_rec(nodes: &[Node], idx: usize, depth: usize, out: &mut String) {
    let indent = "  ".repeat(depth);
    match &nodes[idx] {
        Node::Num(n) => {
            if *n == (*n as i64) as f64 {
                out.push_str(&format!("[{}] {}{}\n", idx, indent, *n as i64));
            } else {
                out.push_str(&format!("[{}] {}{}\n", idx, indent, n));
            }
        }
        Node::Str(s) => {
            out.push_str(&format!("[{}] {}\"{}\"", idx, indent, s));
            out.push('\n');
        }
        Node::Bool(b) => {
            out.push_str(&format!("[{}] {}{}\n", idx, indent, b));
        }
        Node::Symbol(s) => {
            out.push_str(&format!("[{}] {}{}\n", idx, indent, resolve(*s)));
        }
        Node::App(children) => {
            out.push_str(&format!("[{}] {}App\n", idx, indent));
            for &c in children {
                tree_rec(nodes, c, depth + 1, out);
            }
        }
        Node::If(c, t, e) => {
            out.push_str(&format!("[{}] {}If\n", idx, indent));
            tree_rec(nodes, *c, depth + 1, out);
            tree_rec(nodes, *t, depth + 1, out);
            tree_rec(nodes, *e, depth + 1, out);
        }
        Node::Lambda(params, body) => {
            let ps: Vec<String> = params.iter().map(|p| resolve(*p)).collect();
            out.push_str(&format!("[{}] {}Lambda ({})\n", idx, indent, ps.join(" ")));
            tree_rec(nodes, *body, depth + 1, out);
        }
        Node::Let(bindings, body) => {
            let bs: Vec<String> = bindings.iter().map(|(n, _)| resolve(*n)).collect();
            out.push_str(&format!("[{}] {}Let ({})\n", idx, indent, bs.join(" ")));
            for (_, v) in bindings {
                tree_rec(nodes, *v, depth + 1, out);
            }
            tree_rec(nodes, *body, depth + 1, out);
        }
    }
}

// ── Arena helpers ───────────────────────────────────────────────────

/// Append nodes from a separate parse into the main arena, shifting indices.
/// Returns the new root index in the main arena.
fn append_parsed_nodes(arena: &mut Vec<Node>, new_nodes: Vec<Node>, new_root: usize) -> usize {
    let offset = arena.len();
    for node in new_nodes {
        let shifted = shift_node(node, offset);
        arena.push(shifted);
    }
    new_root + offset
}

fn shift_node(node: Node, offset: usize) -> Node {
    match node {
        Node::Num(_) | Node::Str(_) | Node::Bool(_) | Node::Symbol(_) => node,
        Node::App(children) => Node::App(children.into_iter().map(|c| c + offset).collect()),
        Node::If(c, t, e) => Node::If(c + offset, t + offset, e + offset),
        Node::Lambda(params, body) => Node::Lambda(params, body + offset),
        Node::Let(bindings, body) => Node::Let(
            bindings.into_iter().map(|(n, v)| (n, v + offset)).collect(),
            body + offset,
        ),
    }
}

/// Replace all references to `old_idx` with `new_idx` in the arena and roots.
fn patch_references(nodes: &mut Vec<Node>, roots: &mut Vec<usize>, old_idx: usize, new_idx: usize) {
    for i in 0..nodes.len() {
        match &mut nodes[i] {
            Node::App(children) => {
                for c in children.iter_mut() {
                    if *c == old_idx {
                        *c = new_idx;
                    }
                }
            }
            Node::If(c, t, e) => {
                if *c == old_idx { *c = new_idx; }
                if *t == old_idx { *t = new_idx; }
                if *e == old_idx { *e = new_idx; }
            }
            Node::Lambda(_, body) => {
                if *body == old_idx { *body = new_idx; }
            }
            Node::Let(bindings, body) => {
                for (_, v) in bindings.iter_mut() {
                    if *v == old_idx { *v = new_idx; }
                }
                if *body == old_idx { *body = new_idx; }
            }
            _ => {}
        }
    }
    for r in roots.iter_mut() {
        if *r == old_idx {
            *r = new_idx;
        }
    }
}

// ── Edit operations ─────────────────────────────────────────────────

/// Replace the node at `target_idx` with a freshly parsed expression.
pub fn replace_node(
    nodes: &mut Vec<Node>,
    roots: &mut Vec<usize>,
    target_idx: usize,
    new_source: &str,
) -> Result<(), String> {
    if target_idx >= nodes.len() {
        return Err(format!("node index {} out of range (arena has {} nodes)", target_idx, nodes.len()));
    }
    let (new_nodes, new_root) = parse_source(new_source)
        .map_err(|e| format!("parse error in replacement: {}", e.message))?;
    let replacement_idx = append_parsed_nodes(nodes, new_nodes, new_root);
    patch_references(nodes, roots, target_idx, replacement_idx);
    Ok(())
}

/// Wrap the node at `target_idx` in `(let ((name <node>)) name)`.
pub fn wrap_let(
    nodes: &mut Vec<Node>,
    roots: &mut Vec<usize>,
    target_idx: usize,
    name: &str,
) -> Result<(), String> {
    if target_idx >= nodes.len() {
        return Err(format!("node index {} out of range", target_idx));
    }
    let sym = intern(name);
    // Body: a symbol reference to the let-bound name
    let body_idx = nodes.len();
    nodes.push(Node::Symbol(sym));
    // The let node: binds name=target_idx, body=body_idx
    let let_idx = nodes.len();
    nodes.push(Node::Let(vec![(sym, target_idx)], body_idx));
    patch_references(nodes, roots, target_idx, let_idx);
    // Fix: the let's own binding should still point to the original target,
    // but patch_references just changed it. Restore it.
    if let Node::Let(bindings, _) = &mut nodes[let_idx] {
        bindings[0].1 = target_idx;
    }
    Ok(())
}

/// Insert a new top-level define.
pub fn insert_def(
    nodes: &mut Vec<Node>,
    roots: &mut Vec<usize>,
    name: &str,
    params: &[&str],
    body_source: &str,
) -> Result<(), String> {
    let (body_nodes, body_root) = parse_source(body_source)
        .map_err(|e| format!("parse error in body: {}", e.message))?;
    let body_idx = append_parsed_nodes(nodes, body_nodes, body_root);

    let param_syms: Vec<Sym> = params.iter().map(|p| intern(p)).collect();

    // Build: (define name (lambda (params...) body))
    let define_sym_idx = nodes.len();
    nodes.push(Node::Symbol(intern("define")));

    let name_sym_idx = nodes.len();
    nodes.push(Node::Symbol(intern(name)));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(param_syms, body_idx));

    let app_idx = nodes.len();
    nodes.push(Node::App(vec![define_sym_idx, name_sym_idx, lambda_idx]));

    roots.push(app_idx);
    Ok(())
}

/// Delete a top-level definition by name.
pub fn delete_def(
    nodes: &[Node],
    roots: &mut Vec<usize>,
    name: &str,
) -> Result<(), String> {
    let target_sym = intern(name);
    let pos = roots.iter().position(|&root| {
        match &nodes[root] {
            Node::App(children) if children.len() >= 2 => {
                // Check head is define/defmacro
                match &nodes[children[0]] {
                    Node::Symbol(s) => {
                        let h = resolve(*s);
                        if h != "define" && h != "defmacro" {
                            return false;
                        }
                    }
                    _ => return false,
                }
                // Check name matches
                match &nodes[children[1]] {
                    Node::Symbol(s) => *s == target_sym,
                    _ => false,
                }
            }
            _ => false,
        }
    });
    match pos {
        Some(i) => { roots.remove(i); Ok(()) }
        None => Err(format!("no definition named '{}' found", name)),
    }
}

// ── Pretty-print ────────────────────────────────────────────────────

/// Re-emit source with consistent indentation.
pub fn fmt_source(nodes: &[Node], roots: &[usize]) -> String {
    let mut out = String::new();
    for (i, &root) in roots.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        fmt_node(nodes, root, 0, &mut out);
        out.push('\n');
    }
    out
}

fn fmt_node(nodes: &[Node], idx: usize, indent: usize, out: &mut String) {
    match &nodes[idx] {
        Node::Num(n) => {
            if *n == (*n as i64) as f64 {
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{}", n));
            }
        }
        Node::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '"' => out.push_str("\\\""),
                    _ => out.push(c),
                }
            }
            out.push('"');
        }
        Node::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Node::Symbol(s) => out.push_str(&resolve(*s)),
        Node::App(children) => {
            if children.is_empty() {
                out.push_str("()");
                return;
            }
            // Check if this is short enough to fit on one line
            let one_line = node_to_source(nodes, idx);
            if one_line.len() <= 60 || children.len() <= 2 {
                out.push_str(&one_line);
            } else {
                // Multi-line: head on first line, args indented
                out.push('(');
                fmt_node(nodes, children[0], indent + 2, out);
                let child_indent = indent + 2;
                for &c in &children[1..] {
                    out.push('\n');
                    out.push_str(&" ".repeat(child_indent));
                    fmt_node(nodes, c, child_indent, out);
                }
                out.push(')');
            }
        }
        Node::If(c, t, e) => {
            let one_line = node_to_source(nodes, idx);
            if one_line.len() <= 60 {
                out.push_str(&one_line);
            } else {
                out.push_str("(if ");
                fmt_node(nodes, *c, indent + 4, out);
                let body_indent = indent + 2;
                out.push('\n');
                out.push_str(&" ".repeat(body_indent));
                fmt_node(nodes, *t, body_indent, out);
                out.push('\n');
                out.push_str(&" ".repeat(body_indent));
                fmt_node(nodes, *e, body_indent, out);
                out.push(')');
            }
        }
        Node::Lambda(params, body) => {
            let ps: Vec<String> = params.iter().map(|p| resolve(*p)).collect();
            let one_line = node_to_source(nodes, idx);
            if one_line.len() <= 60 {
                out.push_str(&one_line);
            } else {
                out.push_str(&format!("(lambda ({})", ps.join(" ")));
                let body_indent = indent + 2;
                out.push('\n');
                out.push_str(&" ".repeat(body_indent));
                fmt_node(nodes, *body, body_indent, out);
                out.push(')');
            }
        }
        Node::Let(bindings, body) => {
            let one_line = node_to_source(nodes, idx);
            if one_line.len() <= 60 {
                out.push_str(&one_line);
            } else {
                out.push_str("(let (");
                let binding_indent = indent + 6;
                for (i, (n, v)) in bindings.iter().enumerate() {
                    if i > 0 {
                        out.push('\n');
                        out.push_str(&" ".repeat(binding_indent));
                    }
                    out.push('(');
                    out.push_str(&resolve(*n));
                    out.push(' ');
                    fmt_node(nodes, *v, binding_indent + 2, out);
                    out.push(')');
                }
                out.push(')');
                let body_indent = indent + 2;
                out.push('\n');
                out.push_str(&" ".repeat(body_indent));
                fmt_node(nodes, *body, body_indent, out);
                out.push(')');
            }
        }
    }
}

// ── Emit modified source ────────────────────────────────────────────

/// Re-emit all roots as source (compact, one-line-per-form).
pub fn emit_source(nodes: &[Node], roots: &[usize]) -> String {
    let mut out = String::new();
    for (i, &root) in roots.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&node_to_source(nodes, root));
    }
    out.push('\n');
    out
}
