//! Core types for the SELPH runtime.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::intern::{Sym, resolve};

/// AST node.
#[derive(Clone, Debug)]
pub enum Node {
    Num(f64),
    Str(String),
    Bool(bool),
    Symbol(Sym),
    App(Vec<usize>),
    If(usize, usize, usize),
    Lambda(Vec<Sym>, usize),
    Let(Vec<(Sym, usize)>, usize),
}

/// Runtime value.
#[derive(Clone, Debug)]
pub enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
    List(Vec<Value>),
    Nil,
    /// Closure: (params, body_index, captured_env, captured_nodes, letrec_scope)
    /// The optional letrec_scope is a shared mutable scope that enables
    /// self-referential and mutually-recursive let bindings.
    Closure(Vec<Sym>, usize, Env, Rc<[Node]>, Option<SharedScope>),
    Builtin(Sym),
    RustMacro(Vec<Sym>, Rc<[Node]>, usize),
    Namespace(HashMap<String, Value>),
    /// 2D grid of integers 0-9 (row-major). Used for ARC-AGI tasks.
    Grid(Vec<Vec<i8>>),
    /// Spec alternative: matches if actual equals any of the contained values.
    /// Used in synthesis specs, e.g. (or "quick" "speedy") means either is acceptable.
    Alt(Vec<Value>),
}

/// Lexical environment: stack of scopes.
pub type Env = Vec<HashMap<Sym, Value>>;

/// Shared mutable scope for letrec bindings, enabling self/mutual recursion.
pub type SharedScope = Rc<RefCell<HashMap<Sym, Value>>>;

pub fn env_lookup(env: &Env, name: Sym) -> Option<Value> {
    for scope in env.iter().rev() {
        if let Some(v) = scope.get(&name) {
            return Some(v.clone());
        }
    }
    None
}

pub fn env_define(env: &mut Env, name: Sym, val: Value) {
    if let Some(scope) = env.last_mut() {
        scope.insert(name, val);
    }
}

// SAFETY: Value contains Rc<[Node]> and Rc<RefCell<...>> in Closure/RustMacro
// variants, which are !Send and !Sync. However, during parallel synthesis
// evaluation, shared references (&[Value]) to inputs/expected only contain
// Num/Str/Bool/List/Grid variants — never Closure or RustMacro. The parallel
// code gates on values_are_sync_safe() and falls back to sequential if any
// Rc-containing variant is present.
unsafe impl Send for Value {}
unsafe impl Sync for Value {}

/// Check that a slice of Values contains no Rc-bearing variants,
/// making it safe to share across threads.
pub fn values_are_sync_safe(values: &[Value]) -> bool {
    values.iter().all(|v| match v {
        Value::Closure(..) | Value::RustMacro(..) => false,
        Value::List(items) => values_are_sync_safe(items),
        Value::Alt(items) => values_are_sync_safe(items),
        Value::Namespace(map) => values_are_sync_safe(&map.values().cloned().collect::<Vec<_>>()),
        _ => true,
    })
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
        Node::Symbol(sym) => resolve(*sym),
        Node::App(children) => {
            if children.is_empty() { return "()".to_string(); }
            let parts: Vec<String> = children.iter().map(|c| node_to_source(nodes, *c)).collect();
            format!("({})", parts.join(" "))
        }
        Node::If(c, t, e) => format!("(if {} {} {})",
            node_to_source(nodes, *c),
            node_to_source(nodes, *t),
            node_to_source(nodes, *e)),
        Node::Lambda(params, body) => {
            let param_strs: Vec<String> = params.iter().map(|p| resolve(*p)).collect();
            format!("(lambda ({}) {})", param_strs.join(" "), node_to_source(nodes, *body))
        }
        Node::Let(bindings, body) => {
            let bs: Vec<String> = bindings.iter()
                .map(|(name, idx)| format!("({} {})", resolve(*name), node_to_source(nodes, *idx)))
                .collect();
            format!("(let ({}) {})", bs.join(" "), node_to_source(nodes, *body))
        }
    }
}
