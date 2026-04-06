//! Core types for the SELPH runtime.

use std::collections::HashMap;

/// AST node.
#[derive(Clone, Debug)]
pub enum Node {
    Num(f64),
    Str(String),
    Bool(bool),
    Symbol(String),
    App(Vec<usize>),
    If(usize, usize, usize),
    Lambda(Vec<String>, usize),
    Let(Vec<(String, usize)>, usize),
}

/// Runtime value.
#[derive(Clone, Debug)]
pub enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
    List(Vec<Value>),
    Nil,
    /// Closure: (params, body_index, captured_env, captured_nodes)
    Closure(Vec<String>, usize, Env, Vec<Node>),
    Builtin(String),
    RustMacro(Vec<String>, Vec<Node>, usize),
    Namespace(HashMap<String, Value>),
}

/// Lexical environment: stack of scopes.
pub type Env = Vec<HashMap<String, Value>>;

pub fn env_lookup(env: &Env, name: &str) -> Option<Value> {
    for scope in env.iter().rev() {
        if let Some(v) = scope.get(name) {
            return Some(v.clone());
        }
    }
    None
}

pub fn env_define(env: &mut Env, name: String, val: Value) {
    if let Some(scope) = env.last_mut() {
        scope.insert(name, val);
    }
}

/// Source-code representation of a node tree.
pub fn node_to_source(nodes: &[Node], idx: usize) -> String {
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
