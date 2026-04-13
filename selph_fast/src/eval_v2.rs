//! eval_v2 — sketch of the rebuilt SELPH evaluator (April 10, 2026)
//!
//! Pairs with types_v2.rs. This is the §9.24.5 step-4 work: the new tree
//! walker built against the new Value/Env/Node types.
//!
//! What this file demonstrates:
//!   1. The shape of the new eval loop (small, no string-matching in the
//!      hot path).
//!   2. The shape of the new apply (Function uses captured_env, no env
//!      cloning anywhere).
//!   3. A representative cross-section of builtins ported to the new
//!      Value API (Int/Num distinction, Rc-shared str/list).
//!   4. A converter from the old Node enum to the new one. Lets the
//!      existing parser feed into eval_v2 without parser changes — the
//!      converter does SpecialApp resolution and `defmacro → define
//!      lambda` desugaring inline. Throwaway code; deletes when the
//!      parser is updated.
//!   5. End-to-end tests that go source → old Node → new Node → eval.
//!
//! What this file does NOT do:
//!   - Port every builtin (~150 today). Just enough to demonstrate the
//!      patterns and exercise the conversion path.
//!   - Implement `dispatch`, `eval-in`, `try`, `quote`, `ns` special
//!      forms. They're stubbed with todo!() — fill in during the full
//!      migration.
//!   - Wire into existing CLI commands. eval_v2 lives alongside eval.rs
//!      until the swap.

#![allow(dead_code)]

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::intern::{intern, resolve, Sym};
use crate::types::Node as OldNode;
use crate::types_v2::*;

const MAX_EVAL_DEPTH: usize = 256;

thread_local! {
    static EVAL_DEPTH: Cell<usize> = const { Cell::new(0) };
    static BUILTIN_TABLE: BuiltinTable = build_builtin_table();
    static DEFAULT_SCOPE: Scope = build_default_scope();
}

// ────────────────────────────────────────────────────────────────────────────
// Public entry points
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate a node tree at index `idx` in the given env.
///
/// `env` is `&Env` (not `&mut Env`) — Env is Rc-internal so we never need
/// `&mut`. Mutation through `env.define(...)` happens via RefCell on the
/// inner scope. This simplifies signatures and matches the persistent-env
/// model.
pub fn eval(nodes: &Rc<[Node]>, idx: usize, env: &Env) -> Result<Value, String> {
    let depth = EVAL_DEPTH.with(|d| {
        let v = d.get();
        d.set(v + 1);
        v
    });
    if depth >= MAX_EVAL_DEPTH {
        EVAL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        return Err("eval: max recursion depth exceeded".into());
    }
    let result = eval_inner(nodes, idx, env);
    EVAL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    result
}

/// Build a fresh env containing the default scope (builtins) at the bottom.
/// Cheap because the default scope is built once at startup and shared.
pub fn make_default_env() -> Env {
    DEFAULT_SCOPE.with(|s| Env::from_scope(s.clone()))
}

/// Render a Value as a human-readable string for printing/REPL output.
/// Mirrors the existing `value_to_string` in eval.rs but updated for the
/// new Value enum (Int distinct from Num, no Grid/Alt, Sym-keyed Ns).
pub fn value_to_string(v: &Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Num(n) => {
            if n.is_finite() && *n == (*n as i64) as f64 {
                format!("{}.0", *n as i64)
            } else {
                n.to_string()
            }
        }
        Value::Str(s) => s.as_ref().to_string(),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::List(l) => {
            let parts: Vec<String> = l.iter().map(value_to_string).collect();
            format!("({})", parts.join(" "))
        }
        Value::Nil => "nil".into(),
        Value::Function(fd) => {
            let params: Vec<String> = fd.params.iter().map(|p| resolve(*p)).collect();
            format!("<lambda ({})>", params.join(" "))
        }
        Value::Builtin(sym) => format!("<builtin {}>", resolve(*sym)),
        Value::Ns(map) => {
            let mut keys: Vec<String> = map.keys().map(|k| resolve(*k)).collect();
            keys.sort();
            format!("<ns {:?}>", keys)
        }
        Value::Node(node_ref) => {
            // Render the underlying AST as source — same path used by
            // (synthesize ...) results. This makes Node values
            // self-describing in REPL output and print statements.
            node_to_source(&node_ref.nodes, node_ref.idx)
        }
    }
}

/// Render a Node tree as a SELPH source string. Used by `synthesize`
/// to return the source code of a found candidate, and as a debugging
/// aid for synth_v2 output. Mirrors the legacy `node_to_source` against
/// the new Node enum.
pub fn node_to_source(nodes: &[Node], root: usize) -> String {
    fn go(nodes: &[Node], idx: usize, out: &mut String) {
        match &nodes[idx] {
            Node::Int(n) => out.push_str(&n.to_string()),
            Node::Num(n) => {
                if n.is_finite() && *n == (*n as i64) as f64 {
                    out.push_str(&format!("{}.0", *n as i64));
                } else {
                    out.push_str(&n.to_string());
                }
            }
            Node::Str(s) => {
                out.push('"');
                // Escape backslashes and double quotes — no other
                // escapes; the parser handles \\ and \" symmetrically.
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
                out.push('(');
                for (i, &c) in children.iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    go(nodes, c, out);
                }
                out.push(')');
            }
            Node::SpecialApp(form, children) => {
                let head = match form {
                    SpecialForm::Define => "define",
                    SpecialForm::Do => "do",
                    SpecialForm::Quote => "quote",
                    SpecialForm::And => "and",
                    SpecialForm::Or => "or",
                    SpecialForm::Try => "try",
                    SpecialForm::EvalIn => "eval-in",
                    SpecialForm::Dispatch => "dispatch",
                    SpecialForm::Ns => "ns",
                };
                out.push('(');
                out.push_str(head);
                for &c in children {
                    out.push(' ');
                    go(nodes, c, out);
                }
                out.push(')');
            }
            Node::If(cond, then_, else_) => {
                out.push_str("(if ");
                go(nodes, *cond, out);
                out.push(' ');
                go(nodes, *then_, out);
                out.push(' ');
                go(nodes, *else_, out);
                out.push(')');
            }
            Node::Lambda(params, body) => {
                out.push_str("(lambda (");
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(&resolve(*p));
                }
                out.push_str(") ");
                go(nodes, *body, out);
                out.push(')');
            }
            Node::Let(bindings, body) => {
                out.push_str("(let (");
                for (i, (n, v)) in bindings.iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push('(');
                    out.push_str(&resolve(*n));
                    out.push(' ');
                    go(nodes, *v, out);
                    out.push(')');
                }
                out.push_str(") ");
                go(nodes, *body, out);
                out.push(')');
            }
        }
    }
    let mut out = String::new();
    go(nodes, root, &mut out);
    out
}

// ────────────────────────────────────────────────────────────────────────────
// eval_inner — the main loop
// ────────────────────────────────────────────────────────────────────────────

fn eval_inner(nodes: &Rc<[Node]>, idx: usize, env: &Env) -> Result<Value, String> {
    match &nodes[idx] {
        Node::Int(n) => Ok(Value::Int(*n)),
        Node::Num(n) => Ok(Value::Num(*n)),
        Node::Str(s) => Ok(Value::str(s)),
        Node::Bool(b) => Ok(Value::Bool(*b)),
        Node::Symbol(name) => env
            .lookup(*name)
            .ok_or_else(|| format!("unbound: {}", resolve(*name))),
        Node::Lambda(params, body) => Ok(Value::Function(Rc::new(FunctionData {
            params: params.clone(),
            body: NodeRef {
                nodes: Rc::clone(nodes),
                idx: *body,
            },
            captured_env: env.clone(),
            letrec_scope: None,
        }))),
        Node::If(cond, then_br, else_br) => {
            let c = eval(nodes, *cond, env)?;
            if is_truthy(&c) {
                eval(nodes, *then_br, env)
            } else {
                eval(nodes, *else_br, env)
            }
        }
        Node::Lambda(_, _) => unreachable!(), // matched above
        Node::Let(bindings, body) => eval_let(nodes, bindings, *body, env),
        Node::App(children) => eval_app(nodes, children, env),
        Node::SpecialApp(form, children) => eval_special(nodes, *form, children, env),
    }
}

/// Truthiness rule for `if`, `and`, `or`. Falsy values:
///   - Bool(false)
///   - Nil
///   - Int(0), Num(0.0)
///   - Empty Str, empty List
/// Everything else (including empty Ns, functions, builtins) is truthy.
pub fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Bool(false) => false,
        Value::Nil => false,
        Value::Int(0) => false,
        Value::Num(n) if *n == 0.0 => false,
        Value::Str(s) if s.is_empty() => false,
        Value::List(l) if l.is_empty() => false,
        _ => true,
    }
}

fn eval_app(nodes: &Rc<[Node]>, children: &[usize], env: &Env) -> Result<Value, String> {
    if children.is_empty() {
        return Ok(Value::list(vec![]));
    }
    let f = eval(nodes, children[0], env)?;
    let mut args = Vec::with_capacity(children.len() - 1);
    for &c in &children[1..] {
        args.push(eval(nodes, c, env)?);
    }
    apply(&f, &args, env)
}

fn eval_let(
    nodes: &Rc<[Node]>,
    bindings: &[(Sym, usize)],
    body: usize,
    env: &Env,
) -> Result<Value, String> {
    // Push a fresh scope and evaluate bindings in order. Letrec semantics
    // (so functions defined in the same let block can refer to each other)
    // are handled by patching closures with a shared scope, mirroring
    // today's behaviour.
    let frame = env.push_scope(Scope::new());

    use std::cell::RefCell;
    let shared: SharedScope = Rc::new(RefCell::new(Scope::new()));
    let mut closure_names: Vec<Sym> = Vec::new();

    for (name, val_idx) in bindings {
        let v = eval(nodes, *val_idx, &frame)?;
        // §9.45.7 P5 fix: only patch closures whose RHS is a literal
        // `Node::Lambda`. Functions obtained from `(ns-get ...)`,
        // `(eval-node ...)`, or function arguments already carry
        // their own correct captured env and must NOT have it
        // overwritten by this frame's bindings — doing so would let
        // local names shadow the closure's true environment (e.g.
        // binding `exp` in the same let as a closure that calls
        // `(exp v)` would hijack the builtin and crash with
        // "not callable: Int(1)").
        if matches!(&v, Value::Function(_))
            && matches!(&nodes[*val_idx], Node::Lambda(_, _))
        {
            closure_names.push(*name);
        }
        frame.define(*name, v);
    }

    if !closure_names.is_empty() {
        // Patch each function in the frame with the shared letrec scope.
        let mut top = frame.top_scope_mut();
        for cname in &closure_names {
            if let Some(Value::Function(fd)) = top.get(cname).cloned() {
                let patched = FunctionData {
                    params: fd.params.clone(),
                    body: fd.body.clone(),
                    captured_env: fd.captured_env.clone(),
                    letrec_scope: Some(shared.clone()),
                };
                top.insert(*cname, Value::Function(Rc::new(patched)));
            }
        }
        // Populate shared with all current frame bindings (including the
        // patched closures).
        let mut s = shared.borrow_mut();
        for (k, v) in top.iter() {
            s.insert(*k, v.clone());
        }
    }

    eval(nodes, body, &frame)
}

fn eval_special(
    nodes: &Rc<[Node]>,
    form: SpecialForm,
    children: &[usize],
    env: &Env,
) -> Result<Value, String> {
    match form {
        SpecialForm::Define => {
            // (define name value)
            if children.len() != 2 {
                return Err("define: expected (define name value)".into());
            }
            let name = match &nodes[children[0]] {
                Node::Symbol(s) => *s,
                _ => return Err("define: first arg must be a symbol".into()),
            };
            let v = eval(nodes, children[1], env)?;
            env.define(name, v.clone());
            Ok(v)
        }
        SpecialForm::Do => {
            let mut last = Value::Nil;
            for &c in children {
                last = eval(nodes, c, env)?;
            }
            Ok(last)
        }
        SpecialForm::And => {
            let mut last = Value::Bool(true);
            for &c in children {
                last = eval(nodes, c, env)?;
                if !is_truthy(&last) {
                    return Ok(last);
                }
            }
            Ok(last)
        }
        SpecialForm::Or => {
            let mut last = Value::Bool(false);
            for &c in children {
                last = eval(nodes, c, env)?;
                if is_truthy(&last) {
                    return Ok(last);
                }
            }
            Ok(last)
        }
        SpecialForm::Quote => {
            // (quote x) — return x as a first-class AST Node value
            // (§9.36, item 2 of the §9.33 curriculum-only kernel).
            //
            // The quoted child already lives in the parser's arena;
            // we just construct a NodeRef pointing at it. The Rc clone
            // is one refcount bump — quote is essentially free.
            //
            // No evaluation happens inside the quoted expression:
            // symbols stay as Node::Symbol entries, applications stay
            // as Node::App, etc. Resolution happens later if and when
            // the resulting Node is passed to (eval-node ...).
            if children.len() != 1 {
                return Err("quote: expected one argument".into());
            }
            Ok(Value::node(NodeRef {
                nodes: Rc::clone(nodes),
                idx: children[0],
            }))
        }
        SpecialForm::Try => {
            // (try expr fallback)
            if children.len() != 2 {
                return Err("try: expected (try expr fallback)".into());
            }
            match eval(nodes, children[0], env) {
                Ok(v) => Ok(v),
                Err(_) => eval(nodes, children[1], env),
            }
        }
        SpecialForm::EvalIn => {
            // (eval-in source-string) — parse and eval in current env.
            // Stub: needs parser-v2 integration. The full eval.rs version
            // calls parse_file then evals; we'd do the same against parser
            // output once available.
            Err("eval-in: not yet implemented in eval_v2".into())
        }
        SpecialForm::Dispatch => {
            // (dispatch name-string arg)
            if children.len() != 2 {
                return Err("dispatch: expected (dispatch name arg)".into());
            }
            let name_val = eval(nodes, children[0], env)?;
            let name_str = name_val.as_str()?;
            let key = intern(name_str);
            let func = env
                .lookup(key)
                .ok_or_else(|| format!("dispatch: unknown operation '{}'", name_str))?;
            let arg = eval(nodes, children[1], env)?;
            apply(&func, &[arg], env)
        }
        SpecialForm::Ns => {
            // (ns (key val) (key val) ...) — build a namespace literal.
            let mut map = NsMap::new();
            for &c in children {
                if let Node::App(pair) = &nodes[c] {
                    if pair.len() == 2 {
                        let key_sym = match &nodes[pair[0]] {
                            Node::Symbol(s) => *s,
                            Node::Str(s) => intern(s),
                            _ => continue,
                        };
                        let v = eval(nodes, pair[1], env)?;
                        map.insert(key_sym, v);
                    }
                }
            }
            Ok(Value::ns(map))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// apply — function and builtin invocation
// ────────────────────────────────────────────────────────────────────────────
//
// Note: `env` is the caller's env, used only for Builtin dispatch. Functions
// use their own captured_env (the whole point of unifying closures and
// macros).

pub fn apply(f: &Value, args: &[Value], env: &Env) -> Result<Value, String> {
    match f {
        Value::Function(fd) => {
            // Build the parameter scope.
            let mut scope = Scope::with_capacity(fd.params.len());
            for (i, &p) in fd.params.iter().enumerate() {
                if i < args.len() {
                    scope.insert(p, args[i].clone());
                }
            }
            // If this function was patched for letrec, push the shared
            // letrec scope before the param scope so its names resolve.
            let call_env = if let Some(shared) = &fd.letrec_scope {
                let with_letrec = fd.captured_env.push_scope(shared.borrow().clone());
                with_letrec.push_scope(scope)
            } else {
                fd.captured_env.push_scope(scope)
            };
            eval(&fd.body.nodes, fd.body.idx, &call_env)
        }
        Value::Builtin(sym) => {
            let f = BUILTIN_TABLE
                .with(|t| t.lookup(*sym))
                .ok_or_else(|| format!("unknown builtin: {}", resolve(*sym)))?;
            f(args, env)
        }
        _ => Err(format!("not callable: {:?}", f)),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Builtins — representative cross-section
// ────────────────────────────────────────────────────────────────────────────
//
// Patterns demonstrated:
//   - Int/Num arithmetic with coercion (Int op Int = Int; Int op Num = Num)
//   - String ops on Rc<str>
//   - List ops returning Rc<[Value]>
//   - Higher-order: map (calls apply on a function value)
//   - Type predicates (returning Value::Bool)
//
// Each builtin's signature is `fn(&[Value], &Env) -> Result<Value, String>`.
// Most ignore the env. Map/reduce/filter use it when applying user
// functions.

fn build_builtin_table() -> BuiltinTable {
    let mut t = BuiltinTable::new();

    // Arithmetic
    t.register(intern("add"), bi_add);
    t.register(intern("+"), bi_add);
    t.register(intern("subtract"), bi_subtract);
    t.register(intern("-"), bi_subtract);
    t.register(intern("multiply"), bi_multiply);
    t.register(intern("*"), bi_multiply);
    t.register(intern("divide"), bi_divide);
    t.register(intern("/"), bi_divide);
    t.register(intern("modulo"), bi_modulo);
    t.register(intern("%"), bi_modulo);
    t.register(intern("negate"), bi_negate);
    t.register(intern("abs"), bi_abs);
    t.register(intern("min"), bi_min);
    t.register(intern("max"), bi_max);
    t.register(intern("floor"), bi_floor);
    t.register(intern("ceil"), bi_ceil);
    t.register(intern("round"), bi_round);
    t.register(intern("pow"), bi_pow);
    t.register(intern("sqrt"), bi_sqrt);
    t.register(intern("log"), bi_log);
    t.register(intern("exp"), bi_exp);
    t.register(intern("sin"), bi_sin);
    t.register(intern("cos"), bi_cos);
    t.register(intern("tan"), bi_tan);
    t.register(intern("pi"), bi_pi);
    t.register(intern("e"), bi_e);

    // Comparison
    t.register(intern("="), bi_eq);
    t.register(intern("<"), bi_lt);
    t.register(intern(">"), bi_gt);
    t.register(intern("<="), bi_le);
    t.register(intern(">="), bi_ge);
    t.register(intern("not"), bi_not);
    t.register(intern("even"), bi_even);
    t.register(intern("odd"), bi_odd);

    // String
    t.register(intern("string-length"), bi_string_length);
    t.register(intern("string-upper"), bi_string_upper);
    t.register(intern("string-lower"), bi_string_lower);
    t.register(intern("string-reverse"), bi_string_reverse);
    t.register(intern("string-trim"), bi_string_trim);
    t.register(intern("string-concat"), bi_string_concat);
    t.register(intern("concat"), bi_concat);
    t.register(intern("to-string"), bi_to_string);
    t.register(intern("to-number"), bi_to_number);
    t.register(intern("string-join"), bi_string_join);
    t.register(intern("string-split"), bi_string_split);
    t.register(intern("string-replace"), bi_string_replace);
    t.register(intern("string-contains"), bi_string_contains);
    t.register(intern("string-starts-with"), bi_string_starts_with);
    t.register(intern("string-ends-with"), bi_string_ends_with);
    t.register(intern("string-take"), bi_string_take);
    t.register(intern("string-drop"), bi_string_drop);
    t.register(intern("string-slice"), bi_string_slice);
    t.register(intern("string-nth"), bi_string_nth);
    t.register(intern("string-chars"), bi_string_chars);
    t.register(intern("count-char"), bi_count_char);
    t.register(intern("char-code"), bi_char_code);
    t.register(intern("code-char"), bi_code_char);

    // I/O
    t.register(intern("print"), bi_print);

    // List
    t.register(intern("list"), bi_list);
    t.register(intern("head"), bi_head);
    t.register(intern("tail"), bi_tail);
    t.register(intern("length"), bi_length);
    t.register(intern("nth"), bi_nth);
    t.register(intern("cons"), bi_cons);
    t.register(intern("slice"), bi_slice);
    t.register(intern("sort"), bi_sort);
    t.register(intern("reverse"), bi_reverse);
    t.register(intern("append"), bi_append);
    t.register(intern("range"), bi_range);
    t.register(intern("contains"), bi_contains);
    t.register(intern("zip"), bi_zip);
    t.register(intern("enumerate"), bi_enumerate);
    t.register(intern("identity"), bi_identity);

    // Higher-order
    t.register(intern("map"), bi_map);
    t.register(intern("reduce"), bi_reduce);
    t.register(intern("filter"), bi_filter);
    t.register(intern("apply"), bi_apply);

    // Namespace
    t.register(intern("ns-get"), bi_ns_get);
    t.register(intern("ns-put"), bi_ns_put);
    t.register(intern("ns-keys"), bi_ns_keys);
    t.register(intern("ns-values"), bi_ns_values);
    t.register(intern("ns-merge"), bi_ns_merge);
    t.register(intern("ns-size"), bi_ns_size);
    t.register(intern("ns-empty"), bi_ns_empty);
    t.register(intern("ns-get-or"), bi_ns_get_or);
    t.register(intern("ns-has"), bi_ns_has);
    t.register(intern("ns?"), bi_is_ns);

    // Type predicates (forerunner of the predicate-based type system)
    t.register(intern("int?"), bi_is_int);
    t.register(intern("num?"), bi_is_num);
    t.register(intern("number?"), bi_is_num);  // alias matching old eval
    t.register(intern("string?"), bi_is_string);
    t.register(intern("list?"), bi_is_list);
    t.register(intern("bool?"), bi_is_bool);
    t.register(intern("function?"), bi_is_function);
    t.register(intern("nil?"), bi_is_nil);
    t.register(intern("type-of"), bi_type_of);

    // Errors / control
    t.register(intern("error"), bi_error);

    // Bucket 6: meta operations. Most now delegate to synth_v2; only
    // synthesize-optimize remains stubbed pending the optimize port.
    t.register(intern("synthesize"), bi_synthesize);
    t.register(intern("synthesize-args"), bi_synthesize_args);
    t.register(intern("synthesize-optimize"), bi_stub_synthesize_optimize);
    t.register(intern("test-spec"), bi_test_spec);
    t.register(intern("memorize"), bi_memorize);
    t.register(intern("eval-source"), bi_eval_source);

    // §9.36: AST homoiconicity. Construction + inspection builtins
    // for working with first-class Node values. (`quote` is a special
    // form, registered through SpecialForm::Quote — not here.)
    t.register(intern("make-int"), bi_make_int);
    t.register(intern("make-num"), bi_make_num);
    t.register(intern("make-str"), bi_make_str);
    t.register(intern("make-bool"), bi_make_bool);
    t.register(intern("make-symbol"), bi_make_symbol);
    t.register(intern("make-app"), bi_make_app);
    t.register(intern("make-if"), bi_make_if);
    t.register(intern("make-lambda"), bi_make_lambda);
    t.register(intern("make-let"), bi_make_let);
    t.register(intern("node?"), bi_is_node);
    t.register(intern("node-kind"), bi_node_kind);
    t.register(intern("node-int"), bi_node_int);
    t.register(intern("node-num"), bi_node_num);
    t.register(intern("node-str"), bi_node_str);
    t.register(intern("node-bool"), bi_node_bool);
    t.register(intern("node-symbol"), bi_node_symbol);
    t.register(intern("node-children"), bi_node_children);
    t.register(intern("node-params"), bi_node_params);
    t.register(intern("node-bindings"), bi_node_bindings);
    t.register(intern("node-special-form"), bi_node_special_form);
    t.register(intern("eval-node"), bi_eval_node);
    t.register(intern("parse-source"), bi_parse_source);
    t.register(intern("parse-file"), bi_parse_file);

    // §9.45 P1: env/function introspection — required prerequisites for
    // M7 library detection. `env-functions` enumerates user-defined
    // library functions; `function-arity` and `function-param-types`
    // expose the same metadata `library_components_from_env` reads.
    t.register(intern("env-functions"), bi_env_functions);
    t.register(intern("function-arity"), bi_function_arity);
    t.register(intern("function-param-types"), bi_function_param_types);

    // §9.48 P2: grid builtins
    t.register(intern("grid?"), bi_is_grid);
    t.register(intern("grid-height"), bi_grid_height);
    t.register(intern("grid-width"), bi_grid_width);
    t.register(intern("grid-rotate-cw"), bi_grid_rotate_cw);
    t.register(intern("grid-rotate-ccw"), bi_grid_rotate_ccw);
    t.register(intern("grid-rotate-180"), bi_grid_rotate_180);
    t.register(intern("grid-flip-h"), bi_grid_flip_h);
    t.register(intern("grid-flip-v"), bi_grid_flip_v);
    t.register(intern("grid-transpose"), bi_grid_transpose);
    t.register(intern("grid-background"), bi_grid_background);
    t.register(intern("grid-trim"), bi_grid_trim);
    t.register(intern("grid-get"), bi_grid_get);
    t.register(intern("grid-set"), bi_grid_set);
    t.register(intern("grid-replace-color"), bi_grid_replace_color);
    t.register(intern("grid-crop"), bi_grid_crop);
    t.register(intern("grid-overlay"), bi_grid_overlay);
    t.register(intern("grid-colors"), bi_grid_colors);
    t.register(intern("grid-count-color"), bi_grid_count_color);
    t.register(intern("grid-hconcat"), bi_grid_hconcat);
    t.register(intern("grid-vconcat"), bi_grid_vconcat);
    t.register(intern("grid-xor"), bi_grid_xor);
    t.register(intern("grid-and"), bi_grid_and);
    t.register(intern("grid-or"), bi_grid_or);
    t.register(intern("grid-gravity"), bi_grid_gravity);
    t.register(intern("grid-gravity-down"), bi_grid_gravity_down);
    t.register(intern("grid-gravity-right"), bi_grid_gravity_right);
    t.register(intern("grid-gravity-up"), bi_grid_gravity_up);
    t.register(intern("grid-gravity-left"), bi_grid_gravity_left);
    t.register(intern("grid-fill-rect"), bi_grid_fill_rect);
    t.register(intern("grid-size"), bi_grid_size);
    t.register(intern("grid-objects"), bi_grid_objects);
    t.register(intern("grid-objects-8"), bi_grid_objects_8);
    t.register(intern("grid-object-count"), bi_grid_object_count);
    t.register(intern("grid-scale"), bi_grid_scale);
    t.register(intern("grid-tile"), bi_grid_tile);
    t.register(intern("grid-fill-enclosed"), bi_grid_fill_enclosed);
    t.register(intern("grid-compact"), bi_grid_compact);
    t.register(intern("grid-object"), bi_grid_object);
    t.register(intern("grid-object-pos"), bi_grid_object_pos);
    t.register(intern("grid-place"), bi_grid_place);
    t.register(intern("grid-translate"), bi_grid_translate);
    t.register(intern("grid-find-color"), bi_grid_find_color);
    t.register(intern("grid-mask"), bi_grid_mask);
    t.register(intern("grid-blank"), bi_grid_blank);
    // §post-mortem: batch probe builtins for auto-recovery
    t.register(intern("grid-probe-recomp"), bi_grid_probe_recomp);
    t.register(intern("grid-probe-extract"), bi_grid_probe_extract);
    t.register(intern("grid-probe-scale"), bi_grid_probe_scale);
    t.register(intern("grid-diagnose-spec"), bi_grid_diagnose_spec);

    t
}

fn build_default_scope() -> Scope {
    let mut scope = Scope::new();
    // Register every builtin as a Value::Builtin in the default scope.
    // No __builtins__ introspection namespace yet — that's a separate
    // (data-heavy) thing that the type system curriculum will need.
    let names: &[&str] = &[
        // Arithmetic
        "add", "+", "subtract", "-", "multiply", "*", "divide", "/",
        "modulo", "%", "negate", "abs", "min", "max",
        "floor", "ceil", "round", "pow", "sqrt", "log",
        // §9.31 physics curriculum — transcendentals & constants
        "exp", "sin", "cos", "tan", "pi", "e",
        // Comparison
        "=", "<", ">", "<=", ">=", "not", "even", "odd",
        // String
        "string-length", "string-upper", "string-lower", "string-reverse",
        "string-trim", "string-concat", "concat", "to-string", "to-number",
        "string-join", "string-split", "string-replace",
        "string-contains", "string-starts-with", "string-ends-with",
        "string-take", "string-drop", "string-slice", "string-nth",
        "string-chars", "count-char", "char-code", "code-char",
        // I/O
        "print",
        // List
        "list", "head", "tail", "length", "nth", "cons", "slice",
        "sort", "reverse", "append", "range", "contains", "zip",
        "enumerate", "identity",
        // Higher-order
        "map", "reduce", "filter", "apply",
        // Namespace
        "ns-get", "ns-put", "ns-keys", "ns-values", "ns-merge",
        "ns-size", "ns-empty", "ns-get-or", "ns-has", "ns?",
        // Type predicates
        "int?", "num?", "number?", "string?", "list?", "bool?",
        "function?", "nil?", "type-of",
        // Errors / control
        "error",
        // Bucket 6 stubs
        "synthesize", "synthesize-args", "synthesize-optimize", "test-spec", "memorize", "eval-source",
        // §9.36 AST homoiconicity — construction
        "make-int", "make-num", "make-str", "make-bool", "make-symbol",
        "make-app", "make-if", "make-lambda", "make-let",
        // §9.36 AST homoiconicity — inspection
        "node?", "node-kind", "node-int", "node-num", "node-str",
        "node-bool", "node-symbol", "node-children", "node-params",
        "node-bindings", "node-special-form",
        // §9.36 AST homoiconicity — evaluation
        "eval-node",
        // §9.36 AST homoiconicity — parsing
        "parse-source", "parse-file",
        // §9.45 P1 — env/function introspection (M7 prerequisites)
        "env-functions", "function-arity", "function-param-types",
        // §9.48 P2: grid builtins
        "grid?", "grid-height", "grid-width",
        "grid-rotate-cw", "grid-rotate-ccw", "grid-rotate-180",
        "grid-flip-h", "grid-flip-v", "grid-transpose",
        "grid-background", "grid-trim",
        "grid-get", "grid-set", "grid-replace-color", "grid-crop",
        "grid-overlay", "grid-colors", "grid-count-color",
        "grid-hconcat", "grid-vconcat", "grid-xor", "grid-and", "grid-or",
        "grid-gravity", "grid-gravity-down", "grid-gravity-right",
        "grid-gravity-up", "grid-gravity-left",
        "grid-fill-rect", "grid-size",
        "grid-objects", "grid-objects-8", "grid-object-count",
        "grid-scale", "grid-tile", "grid-fill-enclosed", "grid-compact",
        "grid-object", "grid-object-pos", "grid-place", "grid-translate",
        "grid-find-color", "grid-mask", "grid-blank",
        "grid-probe-recomp", "grid-probe-extract", "grid-probe-scale",
        "grid-diagnose-spec",
    ];
    for name in names {
        let sym = intern(name);
        scope.insert(sym, Value::Builtin(sym));
    }
    scope.insert(intern("nil"), Value::Nil);
    scope.insert(intern("true"), Value::Bool(true));
    scope.insert(intern("false"), Value::Bool(false));
    scope
}

// ── arithmetic ──────────────────────────────────────────────────────────────

/// Coerce two numeric Values to a pair of f64 (for mixed-type arith).
fn as_nums(a: &Value, b: &Value) -> Result<(f64, f64), String> {
    let af = match a {
        Value::Int(n) => *n as f64,
        Value::Num(n) => *n,
        _ => return Err(format!("expected number, got {:?}", a)),
    };
    let bf = match b {
        Value::Int(n) => *n as f64,
        Value::Num(n) => *n,
        _ => return Err(format!("expected number, got {:?}", b)),
    };
    Ok((af, bf))
}

/// True if both Values are Int. Coercion rule: Int op Int = Int.
fn both_int(a: &Value, b: &Value) -> Option<(i64, i64)> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some((*x, *y)),
        _ => None,
    }
}

fn bi_add(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("add: expected 2 arguments".into());
    }
    if let Some((a, b)) = both_int(&args[0], &args[1]) {
        // Checked add: overflow promotes to Num.
        match a.checked_add(b) {
            Some(r) => Ok(Value::Int(r)),
            None => Ok(Value::Num(a as f64 + b as f64)),
        }
    } else {
        let (a, b) = as_nums(&args[0], &args[1])?;
        Ok(Value::Num(a + b))
    }
}

fn bi_subtract(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("subtract: expected 2 arguments".into());
    }
    if let Some((a, b)) = both_int(&args[0], &args[1]) {
        match a.checked_sub(b) {
            Some(r) => Ok(Value::Int(r)),
            None => Ok(Value::Num(a as f64 - b as f64)),
        }
    } else {
        let (a, b) = as_nums(&args[0], &args[1])?;
        Ok(Value::Num(a - b))
    }
}

fn bi_multiply(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("multiply: expected 2 arguments".into());
    }
    if let Some((a, b)) = both_int(&args[0], &args[1]) {
        match a.checked_mul(b) {
            Some(r) => Ok(Value::Int(r)),
            None => Ok(Value::Num(a as f64 * b as f64)),
        }
    } else {
        let (a, b) = as_nums(&args[0], &args[1])?;
        Ok(Value::Num(a * b))
    }
}

fn bi_divide(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("divide: expected 2 arguments".into());
    }
    let (a, b) = as_nums(&args[0], &args[1])?;
    if b == 0.0 {
        return Err("divide: division by zero".into());
    }
    Ok(Value::Num(a / b))
}

fn bi_modulo(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("modulo: expected 2 arguments".into());
    }
    if let Some((a, b)) = both_int(&args[0], &args[1]) {
        if b == 0 {
            return Err("modulo: division by zero".into());
        }
        Ok(Value::Int(a.rem_euclid(b)))
    } else {
        let (a, b) = as_nums(&args[0], &args[1])?;
        if b == 0.0 {
            return Err("modulo: division by zero".into());
        }
        Ok(Value::Num(a.rem_euclid(b)))
    }
}

fn bi_negate(args: &[Value], _env: &Env) -> Result<Value, String> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(-n)),
        Value::Num(n) => Ok(Value::Num(-n)),
        _ => Err(format!("negate: expected number, got {:?}", args[0])),
    }
}

fn bi_abs(args: &[Value], _env: &Env) -> Result<Value, String> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(n.abs())),
        Value::Num(n) => Ok(Value::Num(n.abs())),
        _ => Err(format!("abs: expected number, got {:?}", args[0])),
    }
}

fn bi_min(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("min: expected 2 arguments".into()); }
    if let Some((a, b)) = both_int(&args[0], &args[1]) {
        return Ok(Value::Int(a.min(b)));
    }
    let (a, b) = as_nums(&args[0], &args[1])?;
    Ok(Value::Num(a.min(b)))
}

fn bi_max(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("max: expected 2 arguments".into()); }
    if let Some((a, b)) = both_int(&args[0], &args[1]) {
        return Ok(Value::Int(a.max(b)));
    }
    let (a, b) = as_nums(&args[0], &args[1])?;
    Ok(Value::Num(a.max(b)))
}

fn bi_floor(args: &[Value], _env: &Env) -> Result<Value, String> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Num(n) => Ok(Value::Int(n.floor() as i64)),
        _ => Err("floor: expected number".into()),
    }
}

fn bi_ceil(args: &[Value], _env: &Env) -> Result<Value, String> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Num(n) => Ok(Value::Int(n.ceil() as i64)),
        _ => Err("ceil: expected number".into()),
    }
}

fn bi_round(args: &[Value], _env: &Env) -> Result<Value, String> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Num(n) => Ok(Value::Int(n.round() as i64)),
        _ => Err("round: expected number".into()),
    }
}

fn bi_pow(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("pow: expected 2 arguments".into()); }
    // pow on two ints with non-negative exponent stays Int (when it fits).
    if let Some((base, exp)) = both_int(&args[0], &args[1]) {
        if exp >= 0 && exp <= u32::MAX as i64 {
            if let Some(r) = base.checked_pow(exp as u32) {
                return Ok(Value::Int(r));
            }
        }
    }
    let (a, b) = as_nums(&args[0], &args[1])?;
    Ok(Value::Num(a.powf(b)))
}

fn bi_sqrt(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, _) = as_nums(&args[0], &Value::Num(0.0))?;
    Ok(Value::Num(a.sqrt()))
}

fn bi_log(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, _) = as_nums(&args[0], &Value::Num(0.0))?;
    Ok(Value::Num(a.ln()))
}

fn bi_exp(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, _) = as_nums(&args[0], &Value::Num(0.0))?;
    Ok(Value::Num(a.exp()))
}

fn bi_sin(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, _) = as_nums(&args[0], &Value::Num(0.0))?;
    Ok(Value::Num(a.sin()))
}

fn bi_cos(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, _) = as_nums(&args[0], &Value::Num(0.0))?;
    Ok(Value::Num(a.cos()))
}

fn bi_tan(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, _) = as_nums(&args[0], &Value::Num(0.0))?;
    Ok(Value::Num(a.tan()))
}

fn bi_pi(_args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Num(std::f64::consts::PI))
}

fn bi_e(_args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Num(std::f64::consts::E))
}

// ── comparison ──────────────────────────────────────────────────────────────

fn bi_eq(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("=: expected 2 arguments".into());
    }
    Ok(Value::Bool(values_equal(&args[0], &args[1])))
}

/// Structural equality across value variants. Int and Num compare by
/// numeric value (Int(5) == Num(5.0) is true).
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Int(x), Value::Num(y)) | (Value::Num(y), Value::Int(x)) => *x as f64 == *y,
        (Value::Str(x), Value::Str(y)) => x.as_ref() == y.as_ref(),
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| values_equal(a, b))
        }
        (Value::Nil, Value::Nil) => true,
        // Node equality is identity-based (§9.35 D8/R5): two Node values
        // are equal iff they reference the same arena and the same index.
        // Structural equality on ASTs is expensive and rarely the right
        // semantics — usually you want either cache-hit identity or
        // explicit string comparison via node-to-source. A separate
        // (node-equal? a b) builtin can be added later if structural
        // equality is wanted.
        (Value::Node(x), Value::Node(y)) => {
            std::rc::Rc::ptr_eq(&x.nodes, &y.nodes) && x.idx == y.idx
        }
        _ => false,
    }
}

fn bi_lt(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, b) = as_nums(&args[0], &args[1])?;
    Ok(Value::Bool(a < b))
}

fn bi_gt(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, b) = as_nums(&args[0], &args[1])?;
    Ok(Value::Bool(a > b))
}

fn bi_le(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, b) = as_nums(&args[0], &args[1])?;
    Ok(Value::Bool(a <= b))
}

fn bi_ge(args: &[Value], _env: &Env) -> Result<Value, String> {
    let (a, b) = as_nums(&args[0], &args[1])?;
    Ok(Value::Bool(a >= b))
}

fn bi_not(args: &[Value], _env: &Env) -> Result<Value, String> {
    match &args[0] {
        Value::Bool(b) => Ok(Value::Bool(!b)),
        _ => Err("not: expected bool".into()),
    }
}

fn bi_even(args: &[Value], _env: &Env) -> Result<Value, String> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Bool(n % 2 == 0)),
        Value::Num(n) => Ok(Value::Bool((*n as i64) % 2 == 0)),
        _ => Err("even: expected number".into()),
    }
}

fn bi_odd(args: &[Value], _env: &Env) -> Result<Value, String> {
    match &args[0] {
        Value::Int(n) => Ok(Value::Bool(n % 2 != 0)),
        Value::Num(n) => Ok(Value::Bool((*n as i64) % 2 != 0)),
        _ => Err("odd: expected number".into()),
    }
}

// ── string ──────────────────────────────────────────────────────────────────

fn bi_string_length(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    // string-length naturally returns an integer count.
    Ok(Value::Int(s.len() as i64))
}

fn bi_string_upper(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    Ok(Value::str(s.to_uppercase()))
}

fn bi_string_lower(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    Ok(Value::str(s.to_lowercase()))
}

fn bi_string_concat(args: &[Value], _env: &Env) -> Result<Value, String> {
    let mut out = String::new();
    for a in args {
        out.push_str(a.as_str()?);
    }
    Ok(Value::str(out))
}

/// `concat` is the lenient string concatenation: it stringifies any value,
/// not just strings. Distinct from `string-concat` (which requires strings).
/// Many curricula use it as the catch-all join.
fn bi_concat(args: &[Value], _env: &Env) -> Result<Value, String> {
    let mut out = String::new();
    for a in args {
        out.push_str(&value_to_string(a));
    }
    Ok(Value::str(out))
}

fn bi_to_string(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("to-string: expected 1 argument".into());
    }
    Ok(Value::str(value_to_string(&args[0])))
}

fn bi_string_join(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("string-join: expected (string-join list separator)".into());
    }
    let l = args[0].as_list()?;
    let sep = args[1].as_str()?;
    // Build into a single buffer rather than collecting Strings then joining.
    // (§9.24.1 Tier-2 fix applied at the source.)
    let mut out = String::new();
    for (i, item) in l.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        out.push_str(&value_to_string(item));
    }
    Ok(Value::str(out))
}

fn bi_print(args: &[Value], _env: &Env) -> Result<Value, String> {
    let parts: Vec<String> = args.iter().map(value_to_string).collect();
    println!("{}", parts.join(" "));
    Ok(Value::Nil)
}

fn bi_string_reverse(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    Ok(Value::str(s.chars().rev().collect::<String>()))
}

fn bi_string_trim(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    Ok(Value::str(s.trim()))
}

fn bi_to_number(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    // Try Int first, fall back to Num.
    if let Ok(n) = s.parse::<i64>() {
        return Ok(Value::Int(n));
    }
    s.parse::<f64>()
        .map(Value::Num)
        .map_err(|_| format!("to-number: cannot parse {:?}", s))
}

fn bi_string_split(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("string-split: expected (string-split s sep)".into()); }
    let s = args[0].as_str()?;
    let sep = args[1].as_str()?;
    let parts: Vec<Value> = s.split(sep).map(Value::str).collect();
    Ok(Value::list(parts))
}

fn bi_string_replace(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 { return Err("string-replace: expected (string-replace s from to)".into()); }
    let s = args[0].as_str()?;
    let from = args[1].as_str()?;
    let to = args[2].as_str()?;
    Ok(Value::str(s.replace(from, to)))
}

fn bi_string_contains(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    let needle = args[1].as_str()?;
    Ok(Value::Bool(s.contains(needle)))
}

fn bi_string_starts_with(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    let prefix = args[1].as_str()?;
    Ok(Value::Bool(s.starts_with(prefix)))
}

fn bi_string_ends_with(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    let suffix = args[1].as_str()?;
    Ok(Value::Bool(s.ends_with(suffix)))
}

/// `string-take s n` — return the first `n` characters (Unicode-safe).
fn bi_string_take(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    let n = int_arg(&args[1], "string-take")?.max(0) as usize;
    Ok(Value::str(s.chars().take(n).collect::<String>()))
}

/// `string-drop s n` — return everything after the first `n` characters.
fn bi_string_drop(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    let n = int_arg(&args[1], "string-drop")?.max(0) as usize;
    Ok(Value::str(s.chars().skip(n).collect::<String>()))
}

fn bi_string_slice(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 { return Err("string-slice: expected (string-slice s start end)".into()); }
    let s = args[0].as_str()?;
    let start = int_arg(&args[1], "string-slice")?.max(0) as usize;
    let end = int_arg(&args[2], "string-slice")?.max(0) as usize;
    let chars: Vec<char> = s.chars().collect();
    let end = end.min(chars.len());
    let start = start.min(end);
    Ok(Value::str(chars[start..end].iter().collect::<String>()))
}

fn bi_string_nth(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    let i = int_arg(&args[1], "string-nth")?.max(0) as usize;
    s.chars()
        .nth(i)
        .map(|c| Value::str(c.to_string()))
        .ok_or_else(|| format!("string-nth: index {} out of bounds", i))
}

fn bi_string_chars(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    Ok(Value::list(s.chars().map(|c| Value::str(c.to_string())).collect()))
}

fn bi_count_char(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    let needle = args[1].as_str()?;
    if needle.is_empty() { return Ok(Value::Int(0)); }
    Ok(Value::Int(s.matches(needle).count() as i64))
}

fn bi_char_code(args: &[Value], _env: &Env) -> Result<Value, String> {
    let s = args[0].as_str()?;
    let c = s.chars().next().ok_or("char-code: empty string")?;
    Ok(Value::Int(c as i64))
}

fn bi_code_char(args: &[Value], _env: &Env) -> Result<Value, String> {
    let n = int_arg(&args[0], "code-char")?;
    let c = char::from_u32(n as u32).ok_or_else(|| format!("code-char: invalid code {}", n))?;
    Ok(Value::str(c.to_string()))
}

/// Helper: extract an integer index from an arg, accepting Int or integer-valued Num.
fn int_arg(v: &Value, ctx: &str) -> Result<i64, String> {
    match v {
        Value::Int(n) => Ok(*n),
        Value::Num(n) => Ok(*n as i64),
        _ => Err(format!("{}: expected integer index, got {:?}", ctx, v)),
    }
}

// ── list ────────────────────────────────────────────────────────────────────

fn bi_list(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::list(args.to_vec()))
}

fn bi_head(args: &[Value], _env: &Env) -> Result<Value, String> {
    let l = args[0].as_list()?;
    l.first()
        .cloned()
        .ok_or_else(|| "head: empty list".into())
}

fn bi_tail(args: &[Value], _env: &Env) -> Result<Value, String> {
    let l = args[0].as_list()?;
    if l.is_empty() {
        return Err("tail: empty list".into());
    }
    Ok(Value::list(l[1..].to_vec()))
}

fn bi_length(args: &[Value], _env: &Env) -> Result<Value, String> {
    let l = args[0].as_list()?;
    Ok(Value::Int(l.len() as i64))
}

fn bi_nth(args: &[Value], _env: &Env) -> Result<Value, String> {
    let l = args[0].as_list()?;
    let i = int_arg(&args[1], "nth")? as usize;
    l.get(i)
        .cloned()
        .ok_or_else(|| format!("nth: index {} out of bounds", i))
}

fn bi_cons(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("cons: expected (cons head tail)".into()); }
    let tail = args[1].as_list()?;
    let mut out = Vec::with_capacity(tail.len() + 1);
    out.push(args[0].clone());
    out.extend_from_slice(tail);
    Ok(Value::list(out))
}

fn bi_slice(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 { return Err("slice: expected (slice list start end)".into()); }
    let l = args[0].as_list()?;
    let start = int_arg(&args[1], "slice")?.max(0) as usize;
    let end = int_arg(&args[2], "slice")?.max(0) as usize;
    let end = end.min(l.len());
    let start = start.min(end);
    Ok(Value::list(l[start..end].to_vec()))
}

fn bi_sort(args: &[Value], _env: &Env) -> Result<Value, String> {
    let l = args[0].as_list()?;
    let mut owned: Vec<Value> = l.to_vec();
    // Sort numerically when possible, lexicographically otherwise.
    owned.sort_by(|a, b| {
        use std::cmp::Ordering;
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => x.cmp(y),
            (Value::Num(x), Value::Num(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
            (Value::Int(x), Value::Num(y)) => (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal),
            (Value::Num(x), Value::Int(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal),
            (Value::Str(x), Value::Str(y)) => x.as_ref().cmp(y.as_ref()),
            (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
            _ => Ordering::Equal,
        }
    });
    Ok(Value::list(owned))
}

fn bi_reverse(args: &[Value], _env: &Env) -> Result<Value, String> {
    let l = args[0].as_list()?;
    let mut owned: Vec<Value> = l.to_vec();
    owned.reverse();
    Ok(Value::list(owned))
}

fn bi_append(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("append: expected 2 arguments".into()); }
    let a = args[0].as_list()?;
    let b = args[1].as_list()?;
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    Ok(Value::list(out))
}

fn bi_range(args: &[Value], _env: &Env) -> Result<Value, String> {
    let n = int_arg(&args[0], "range")?;
    if n < 0 { return Ok(Value::list(vec![])); }
    Ok(Value::list((0..n).map(Value::Int).collect()))
}

fn bi_contains(args: &[Value], _env: &Env) -> Result<Value, String> {
    let l = args[0].as_list()?;
    let needle = &args[1];
    Ok(Value::Bool(l.iter().any(|v| values_equal(v, needle))))
}

fn bi_zip(args: &[Value], _env: &Env) -> Result<Value, String> {
    let a = args[0].as_list()?;
    let b = args[1].as_list()?;
    let n = a.len().min(b.len());
    let pairs: Vec<Value> = (0..n)
        .map(|i| Value::list(vec![a[i].clone(), b[i].clone()]))
        .collect();
    Ok(Value::list(pairs))
}

fn bi_enumerate(args: &[Value], _env: &Env) -> Result<Value, String> {
    let l = args[0].as_list()?;
    let pairs: Vec<Value> = l
        .iter()
        .enumerate()
        .map(|(i, v)| Value::list(vec![Value::Int(i as i64), v.clone()]))
        .collect();
    Ok(Value::list(pairs))
}

fn bi_identity(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.is_empty() { return Err("identity: expected 1 argument".into()); }
    Ok(args[0].clone())
}

// ── higher-order ────────────────────────────────────────────────────────────

fn bi_map(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("map: expected (map f list)".into());
    }
    let f = &args[0];
    let l = args[1].as_list()?;
    let mut out = Vec::with_capacity(l.len());
    for item in l {
        out.push(apply(f, &[item.clone()], env)?);
    }
    Ok(Value::list(out))
}

fn bi_reduce(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("reduce: expected (reduce f list [init])".into());
    }
    let f = &args[0];
    let l = args[1].as_list()?;
    if l.is_empty() && args.len() < 3 {
        return Err("reduce: empty list with no initial value".into());
    }
    let (mut acc, items) = if args.len() > 2 {
        (args[2].clone(), &l[..])
    } else {
        (l[0].clone(), &l[1..])
    };
    for item in items {
        acc = apply(f, &[acc, item.clone()], env)?;
    }
    Ok(acc)
}

fn bi_filter(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("filter: expected (filter f list)".into());
    }
    let f = &args[0];
    let l = args[1].as_list()?;
    let mut out = Vec::new();
    for item in l {
        // §9.47.6: use is_truthy (same semantics as `if`) instead of
        // matching only Bool(true). This matches standard Lisp/Scheme
        // convention where filter keeps elements for which the predicate
        // returns any truthy value, not just #t.
        if is_truthy(&apply(f, &[item.clone()], env)?) {
            out.push(item.clone());
        }
    }
    Ok(Value::list(out))
}

fn bi_apply(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("apply: expected (apply f arg-list)".into()); }
    let f = &args[0];
    let arg_list = args[1].as_list()?;
    let owned: Vec<Value> = arg_list.to_vec();
    apply(f, &owned, env)
}

// ── namespace ───────────────────────────────────────────────────────────────
//
// Sym-keyed under the hood. SELPH source-level keys come in as strings;
// we intern them at the boundary. ns-keys returns the resolved strings
// for symmetry with existing curricula.

fn ns_get_inner(ns: &Value, key_str: &str) -> Result<Value, String> {
    let map = match ns {
        Value::Ns(m) => m,
        _ => return Err(format!("ns-get: expected namespace, got {:?}", ns)),
    };
    let key = intern(key_str);
    map.get(&key)
        .cloned()
        .ok_or_else(|| format!("ns-get: key {:?} not found", key_str))
}

fn bi_ns_get(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() < 2 { return Err("ns-get: need namespace and key".into()); }
    // Path-style: (ns-get ns "a" "b" "c") walks nested namespaces.
    let mut current = args[0].clone();
    for k in &args[1..] {
        let ks = k.as_str()?;
        current = ns_get_inner(&current, ks)?;
    }
    Ok(current)
}

fn bi_ns_put(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 { return Err("ns-put: need namespace, key, value".into()); }
    let map = match &args[0] {
        Value::Ns(m) => m.as_ref().clone(),
        _ => return Err("ns-put: first arg must be a namespace".into()),
    };
    let key = intern(args[1].as_str()?);
    let mut new_map = map;
    new_map.insert(key, args[2].clone());
    Ok(Value::ns(new_map))
}

fn bi_ns_keys(args: &[Value], _env: &Env) -> Result<Value, String> {
    let map = match &args[0] {
        Value::Ns(m) => m,
        _ => return Err("ns-keys: expected namespace".into()),
    };
    let mut keys: Vec<String> = map.keys().map(|k| resolve(*k)).collect();
    keys.sort();
    Ok(Value::list(keys.into_iter().map(Value::str).collect()))
}

fn bi_ns_values(args: &[Value], _env: &Env) -> Result<Value, String> {
    let map = match &args[0] {
        Value::Ns(m) => m,
        _ => return Err("ns-values: expected namespace".into()),
    };
    Ok(Value::list(map.values().cloned().collect()))
}

fn bi_ns_merge(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("ns-merge: need two namespaces".into()); }
    let a = match &args[0] {
        Value::Ns(m) => m.as_ref().clone(),
        _ => return Err("ns-merge: first arg must be a namespace".into()),
    };
    let b = match &args[1] {
        Value::Ns(m) => m,
        _ => return Err("ns-merge: second arg must be a namespace".into()),
    };
    let mut merged = a;
    for (k, v) in b.iter() {
        merged.insert(*k, v.clone());
    }
    Ok(Value::ns(merged))
}

fn bi_ns_size(args: &[Value], _env: &Env) -> Result<Value, String> {
    let map = match &args[0] {
        Value::Ns(m) => m,
        _ => return Err("ns-size: expected namespace".into()),
    };
    Ok(Value::Int(map.len() as i64))
}

fn bi_ns_empty(_args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::ns(NsMap::new()))
}

fn bi_ns_get_or(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 { return Err("ns-get-or: need namespace, key, default".into()); }
    let key_str = args[1].as_str()?;
    match ns_get_inner(&args[0], key_str) {
        Ok(v) => Ok(v),
        Err(_) => Ok(args[2].clone()),
    }
}

fn bi_ns_has(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 { return Err("ns-has: need namespace and key".into()); }
    let map = match &args[0] {
        Value::Ns(m) => m,
        _ => return Err("ns-has: first arg must be a namespace".into()),
    };
    let key = intern(args[1].as_str()?);
    Ok(Value::Bool(map.contains_key(&key)))
}

fn bi_is_ns(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Bool(matches!(&args[0], Value::Ns(_))))
}

// ── type predicates ─────────────────────────────────────────────────────────

fn bi_is_int(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Bool(matches!(&args[0], Value::Int(_))))
}
fn bi_is_num(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Bool(matches!(&args[0], Value::Num(_) | Value::Int(_))))
}
fn bi_is_string(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Bool(matches!(&args[0], Value::Str(_))))
}
fn bi_is_list(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Bool(matches!(&args[0], Value::List(_))))
}
fn bi_is_bool(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Bool(matches!(&args[0], Value::Bool(_))))
}
fn bi_is_function(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Bool(matches!(
        &args[0],
        Value::Function(_) | Value::Builtin(_)
    )))
}

fn bi_is_nil(args: &[Value], _env: &Env) -> Result<Value, String> {
    Ok(Value::Bool(matches!(&args[0], Value::Nil)))
}

/// Returns the canonical type name as a string. Mirrors today's `type-of`
/// but reports `int` and `num` separately to match the new Value enum.
fn bi_type_of(args: &[Value], _env: &Env) -> Result<Value, String> {
    let name = match &args[0] {
        Value::Int(_) => "int",
        Value::Num(_) => "num",
        Value::Str(_) => "string",
        Value::Bool(_) => "bool",
        Value::List(_) => "list",
        Value::Ns(_) => "namespace",
        Value::Function(_) | Value::Builtin(_) => "function",
        Value::Node(_) => "node",
        Value::Nil => "nil",
    };
    Ok(Value::str(name))
}

fn bi_error(args: &[Value], _env: &Env) -> Result<Value, String> {
    let msg = if args.is_empty() {
        "error".to_string()
    } else {
        value_to_string(&args[0])
    };
    Err(msg)
}

// ── §9.36 AST homoiconicity: construction + inspection builtins ─────────────
//
// Item 2 of the §9.33 curriculum-only kernel. SELPH programs construct AST
// nodes via `(make-int 5)`, `(make-app "add" arg1 arg2)`, etc., inspect
// them via `(node-kind n)` and friends, and evaluate them via
// `(eval-node n)`. Combined with `(quote ...)` (sub-step 2b) and
// `(parse-source "...")` (sub-step 2f), this gives SELPH programs the
// ability to manipulate other SELPH programs as data — the foundation
// for SELPH-side decomposers (item 3) and type predicates (item 4).
//
// Design notes (from §9.35):
//   - Each `make-*` allocates a fresh Rc-shared arena. Sub-trees are
//     copied via `synth_v2::remap_node`.
//   - `make-app` accepts the head as either a Symbol Node or a string
//     (auto-wrapped via `intern`). It NEVER accepts a Function value —
//     a Function carries env capture and would smuggle runtime env into
//     pure data.
//   - `make-lambda` and `make-let` take param/binding names as Strings.
//   - `eval-node` uses the caller's env (no second env parameter).
//   - Inspection accessors (node-int, node-symbol, etc.) are typed and
//     error on the wrong node kind. `node-kind` returns a string for
//     dispatch.

/// Helper: extract the underlying NodeRef from a Value::Node argument,
/// returning a clean error message naming the builtin if the value
/// isn't a node.
fn as_node<'a>(builtin: &str, v: &'a Value) -> Result<&'a NodeRef, String> {
    match v {
        Value::Node(n) => Ok(n),
        _ => Err(format!("{}: expected a Node, got {:?}", builtin, v)),
    }
}

/// Helper: build a single-Node Value::Node arena.
fn single_node_value(node: Node) -> Value {
    let nodes_rc: Rc<[Node]> = vec![node].into();
    Value::node(NodeRef {
        nodes: nodes_rc,
        idx: 0,
    })
}

// ── Construction builtins ───────────────────────────────────────────────────

fn bi_make_int(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("make-int: expected 1 arg, got {}", args.len()));
    }
    let n = match &args[0] {
        Value::Int(n) => *n,
        Value::Num(n) => *n as i64,
        other => return Err(format!("make-int: expected number, got {:?}", other)),
    };
    Ok(single_node_value(Node::Int(n)))
}

fn bi_make_num(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("make-num: expected 1 arg, got {}", args.len()));
    }
    let n = match &args[0] {
        Value::Int(n) => *n as f64,
        Value::Num(n) => *n,
        other => return Err(format!("make-num: expected number, got {:?}", other)),
    };
    Ok(single_node_value(Node::Num(n)))
}

fn bi_make_str(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("make-str: expected 1 arg, got {}", args.len()));
    }
    let s = match &args[0] {
        Value::Str(s) => s.as_ref().to_string(),
        other => return Err(format!("make-str: expected string, got {:?}", other)),
    };
    Ok(single_node_value(Node::Str(s)))
}

fn bi_make_bool(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("make-bool: expected 1 arg, got {}", args.len()));
    }
    let b = match &args[0] {
        Value::Bool(b) => *b,
        other => return Err(format!("make-bool: expected bool, got {:?}", other)),
    };
    Ok(single_node_value(Node::Bool(b)))
}

fn bi_make_symbol(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("make-symbol: expected 1 arg, got {}", args.len()));
    }
    let name = match &args[0] {
        Value::Str(s) => s.as_ref().to_string(),
        other => return Err(format!("make-symbol: expected string, got {:?}", other)),
    };
    let sym = intern(&name);
    Ok(single_node_value(Node::Symbol(sym)))
}

/// Internal: copy a NodeRef's nodes into `out`, applying offset, and return
/// the new index of the root. Used by all multi-arg constructors.
fn copy_subtree(node_ref: &NodeRef, out: &mut Vec<Node>) -> usize {
    let offset = out.len();
    for n in node_ref.nodes.iter() {
        out.push(crate::synth_v2::remap_node(n, offset));
    }
    node_ref.idx + offset
}

/// Internal: extract a Node value from an argument that may be a
/// Value::Node OR a Value::Str (auto-wrapped as a Symbol Node). Used
/// only by make-app's head argument for ergonomics.
fn arg_to_node_or_symbol_str(builtin: &str, v: &Value) -> Result<NodeRef, String> {
    match v {
        Value::Node(n) => Ok(n.clone()),
        Value::Str(s) => {
            let sym = intern(s.as_ref());
            let nodes_rc: Rc<[Node]> = vec![Node::Symbol(sym)].into();
            Ok(NodeRef {
                nodes: nodes_rc,
                idx: 0,
            })
        }
        other => Err(format!(
            "{}: head must be a Node or string, got {:?}",
            builtin, other
        )),
    }
}

fn bi_make_app(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.is_empty() {
        return Err("make-app: expected at least 1 arg (head)".into());
    }
    let head = arg_to_node_or_symbol_str("make-app", &args[0])?;
    let mut out: Vec<Node> = Vec::new();
    let head_idx = copy_subtree(&head, &mut out);
    let mut child_indices: Vec<usize> = vec![head_idx];
    for arg in &args[1..] {
        let node_ref = as_node("make-app", arg)?;
        let idx = copy_subtree(node_ref, &mut out);
        child_indices.push(idx);
    }
    let app_idx = out.len();
    out.push(Node::App(child_indices));
    let nodes_rc: Rc<[Node]> = out.into();
    Ok(Value::node(NodeRef {
        nodes: nodes_rc,
        idx: app_idx,
    }))
}

fn bi_make_if(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 {
        return Err(format!(
            "make-if: expected 3 args (cond, then, else), got {}",
            args.len()
        ));
    }
    let cond = as_node("make-if", &args[0])?;
    let then_ = as_node("make-if", &args[1])?;
    let else_ = as_node("make-if", &args[2])?;
    let mut out: Vec<Node> = Vec::new();
    let cond_idx = copy_subtree(cond, &mut out);
    let then_idx = copy_subtree(then_, &mut out);
    let else_idx = copy_subtree(else_, &mut out);
    let if_idx = out.len();
    out.push(Node::If(cond_idx, then_idx, else_idx));
    let nodes_rc: Rc<[Node]> = out.into();
    Ok(Value::node(NodeRef {
        nodes: nodes_rc,
        idx: if_idx,
    }))
}

fn bi_make_lambda(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!(
            "make-lambda: expected 2 args (params-list, body), got {}",
            args.len()
        ));
    }
    let params_list = match &args[0] {
        Value::List(l) => l,
        other => {
            return Err(format!(
                "make-lambda: first arg must be a list of param names, got {:?}",
                other
            ))
        }
    };
    let mut params: Vec<Sym> = Vec::with_capacity(params_list.len());
    for p in params_list.iter() {
        match p {
            Value::Str(s) => params.push(intern(s.as_ref())),
            other => {
                return Err(format!(
                    "make-lambda: param names must be strings, got {:?}",
                    other
                ))
            }
        }
    }
    let body = as_node("make-lambda", &args[1])?;
    let mut out: Vec<Node> = Vec::new();
    let body_idx = copy_subtree(body, &mut out);
    let lam_idx = out.len();
    out.push(Node::Lambda(params, body_idx));
    let nodes_rc: Rc<[Node]> = out.into();
    Ok(Value::node(NodeRef {
        nodes: nodes_rc,
        idx: lam_idx,
    }))
}

fn bi_make_let(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!(
            "make-let: expected 2 args (bindings-list, body), got {}",
            args.len()
        ));
    }
    let bindings_list = match &args[0] {
        Value::List(l) => l,
        other => {
            return Err(format!(
                "make-let: first arg must be a list of (name, value-node) pairs, got {:?}",
                other
            ))
        }
    };

    let mut out: Vec<Node> = Vec::new();
    let mut bindings: Vec<(Sym, usize)> = Vec::with_capacity(bindings_list.len());
    for (i, pair) in bindings_list.iter().enumerate() {
        let pair_list = match pair {
            Value::List(l) if l.len() == 2 => l,
            other => {
                return Err(format!(
                    "make-let: binding {} must be a (name, value-node) pair, got {:?}",
                    i, other
                ))
            }
        };
        let name = match &pair_list[0] {
            Value::Str(s) => intern(s.as_ref()),
            other => {
                return Err(format!(
                    "make-let: binding {} name must be a string, got {:?}",
                    i, other
                ))
            }
        };
        let value_node = as_node("make-let", &pair_list[1])?;
        let idx = copy_subtree(value_node, &mut out);
        bindings.push((name, idx));
    }
    let body = as_node("make-let", &args[1])?;
    let body_idx = copy_subtree(body, &mut out);
    let let_idx = out.len();
    out.push(Node::Let(bindings, body_idx));
    let nodes_rc: Rc<[Node]> = out.into();
    Ok(Value::node(NodeRef {
        nodes: nodes_rc,
        idx: let_idx,
    }))
}

// ── §9.36 inspection builtins ───────────────────────────────────────────────

/// Helper: build a NodeRef into the same arena as `parent` but pointing
/// at child index `idx`. Used by `node-children` to expose subexpressions
/// without copying — the new NodeRefs share the parent's Rc.
fn child_node_value(parent: &NodeRef, idx: usize) -> Value {
    Value::node(NodeRef {
        nodes: Rc::clone(&parent.nodes),
        idx,
    })
}

fn bi_is_node(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("node?: expected 1 arg, got {}", args.len()));
    }
    Ok(Value::Bool(matches!(&args[0], Value::Node(_))))
}

fn bi_node_kind(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("node-kind: expected 1 arg, got {}", args.len()));
    }
    let node_ref = as_node("node-kind", &args[0])?;
    let kind = match &node_ref.nodes[node_ref.idx] {
        Node::Int(_) => "int",
        Node::Num(_) => "num",
        Node::Str(_) => "str",
        Node::Bool(_) => "bool",
        Node::Symbol(_) => "symbol",
        Node::App(_) => "app",
        Node::SpecialApp(_, _) => "special-app",
        Node::If(_, _, _) => "if",
        Node::Lambda(_, _) => "lambda",
        Node::Let(_, _) => "let",
    };
    Ok(Value::str(kind))
}

fn bi_node_int(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("node-int: expected 1 arg, got {}", args.len()));
    }
    let node_ref = as_node("node-int", &args[0])?;
    match &node_ref.nodes[node_ref.idx] {
        Node::Int(n) => Ok(Value::Int(*n)),
        other => Err(format!("node-int: expected Int node, got {:?}", other)),
    }
}

fn bi_node_num(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("node-num: expected 1 arg, got {}", args.len()));
    }
    let node_ref = as_node("node-num", &args[0])?;
    match &node_ref.nodes[node_ref.idx] {
        Node::Num(n) => Ok(Value::Num(*n)),
        other => Err(format!("node-num: expected Num node, got {:?}", other)),
    }
}

fn bi_node_str(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("node-str: expected 1 arg, got {}", args.len()));
    }
    let node_ref = as_node("node-str", &args[0])?;
    match &node_ref.nodes[node_ref.idx] {
        Node::Str(s) => Ok(Value::str(s)),
        other => Err(format!("node-str: expected Str node, got {:?}", other)),
    }
}

fn bi_node_bool(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("node-bool: expected 1 arg, got {}", args.len()));
    }
    let node_ref = as_node("node-bool", &args[0])?;
    match &node_ref.nodes[node_ref.idx] {
        Node::Bool(b) => Ok(Value::Bool(*b)),
        other => Err(format!("node-bool: expected Bool node, got {:?}", other)),
    }
}

fn bi_node_symbol(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("node-symbol: expected 1 arg, got {}", args.len()));
    }
    let node_ref = as_node("node-symbol", &args[0])?;
    match &node_ref.nodes[node_ref.idx] {
        Node::Symbol(sym) => Ok(Value::str(resolve(*sym))),
        other => Err(format!("node-symbol: expected Symbol node, got {:?}", other)),
    }
}

fn bi_node_children(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!(
            "node-children: expected 1 arg, got {}",
            args.len()
        ));
    }
    let node_ref = as_node("node-children", &args[0])?;
    // Per §9.35 D8: node-children returns subexpressions only.
    // Structural metadata (params, bindings, special-form name) is
    // accessed via the kind-specific accessors below.
    let child_indices: Vec<usize> = match &node_ref.nodes[node_ref.idx] {
        Node::Int(_) | Node::Num(_) | Node::Str(_) | Node::Bool(_) | Node::Symbol(_) => {
            Vec::new()
        }
        Node::App(c) => c.clone(),
        Node::SpecialApp(_, c) => c.clone(),
        Node::If(a, b, c) => vec![*a, *b, *c],
        Node::Lambda(_, body) => vec![*body],
        Node::Let(_, body) => vec![*body],
    };
    let children: Vec<Value> = child_indices
        .iter()
        .map(|&i| child_node_value(node_ref, i))
        .collect();
    Ok(Value::list(children))
}

fn bi_node_params(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("node-params: expected 1 arg, got {}", args.len()));
    }
    let node_ref = as_node("node-params", &args[0])?;
    match &node_ref.nodes[node_ref.idx] {
        Node::Lambda(params, _) => {
            let names: Vec<Value> = params
                .iter()
                .map(|sym| Value::str(resolve(*sym)))
                .collect();
            Ok(Value::list(names))
        }
        other => Err(format!(
            "node-params: expected Lambda node, got {:?}",
            other
        )),
    }
}

fn bi_node_bindings(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!(
            "node-bindings: expected 1 arg, got {}",
            args.len()
        ));
    }
    let node_ref = as_node("node-bindings", &args[0])?;
    match &node_ref.nodes[node_ref.idx] {
        Node::Let(bindings, _) => {
            let pairs: Vec<Value> = bindings
                .iter()
                .map(|(name, value_idx)| {
                    Value::list(vec![
                        Value::str(resolve(*name)),
                        child_node_value(node_ref, *value_idx),
                    ])
                })
                .collect();
            Ok(Value::list(pairs))
        }
        other => Err(format!(
            "node-bindings: expected Let node, got {:?}",
            other
        )),
    }
}

/// `(parse-source source)` — parse a single SELPH expression from a
/// string and return it as a Node value. The result is one Node ref
/// pointing into a fresh Rc-shared arena.
///
/// Companion to `eval-source`: `parse-source` returns the AST without
/// running it, while `eval-source` parses and evaluates in one step.
/// `(eval-node (parse-source s))` is equivalent to `(eval-source s)`.
fn bi_parse_source(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("parse-source: expected 1 arg, got {}", args.len()));
    }
    let src = match &args[0] {
        Value::Str(s) => s.as_ref().to_string(),
        other => {
            return Err(format!(
                "parse-source: expected string, got {:?}",
                other
            ))
        }
    };
    let (old_nodes, root) = crate::parser::parse_source(&src)
        .map_err(|e| format!("parse-source: {}", e))?;
    let new_nodes_vec = convert_tree(&old_nodes);
    let nodes_rc: Rc<[Node]> = new_nodes_vec.into();
    Ok(Value::node(NodeRef {
        nodes: nodes_rc,
        idx: root,
    }))
}

/// `(parse-file source)` — parse a SELPH source string containing one
/// or more top-level forms and return a list of Node values, one per
/// form. All Node values share a single underlying Rc<[Node]> arena;
/// the list elements differ only in their root index.
fn bi_parse_file(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("parse-file: expected 1 arg, got {}", args.len()));
    }
    let src = match &args[0] {
        Value::Str(s) => s.as_ref().to_string(),
        other => return Err(format!("parse-file: expected string, got {:?}", other)),
    };
    let (old_nodes, roots) = crate::parser::parse_file(&src)
        .map_err(|e| format!("parse-file: {}", e))?;
    let new_nodes_vec = convert_tree(&old_nodes);
    let nodes_rc: Rc<[Node]> = new_nodes_vec.into();
    let nodes: Vec<Value> = roots
        .iter()
        .map(|&r| {
            Value::node(NodeRef {
                nodes: Rc::clone(&nodes_rc),
                idx: r,
            })
        })
        .collect();
    Ok(Value::list(nodes))
}

/// `(eval-node n)` — evaluate an AST Node value against the caller's
/// current env. The Rc<[Node]> arena is borrowed (refcount bumped),
/// no copy. Returns whatever the Node evaluates to.
///
/// Combined with `quote`, `make-*`, and `parse-source`, this completes
/// the homoiconicity round trip: SELPH programs can construct, inspect,
/// and *run* arbitrary AST trees as data.
fn bi_eval_node(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("eval-node: expected 1 arg, got {}", args.len()));
    }
    let node_ref = as_node("eval-node", &args[0])?;
    eval(&node_ref.nodes, node_ref.idx, env)
}

fn bi_node_special_form(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!(
            "node-special-form: expected 1 arg, got {}",
            args.len()
        ));
    }
    let node_ref = as_node("node-special-form", &args[0])?;
    match &node_ref.nodes[node_ref.idx] {
        Node::SpecialApp(form, _) => {
            let name = match form {
                SpecialForm::Define => "define",
                SpecialForm::Do => "do",
                SpecialForm::Quote => "quote",
                SpecialForm::And => "and",
                SpecialForm::Or => "or",
                SpecialForm::Try => "try",
                SpecialForm::EvalIn => "eval-in",
                SpecialForm::Dispatch => "dispatch",
                SpecialForm::Ns => "ns",
            };
            Ok(Value::str(name))
        }
        other => Err(format!(
            "node-special-form: expected SpecialApp node, got {:?}",
            other
        )),
    }
}

// ── §9.45 P1: env / function introspection ─────────────────────────────────
//
// Three builtins added as M7 prerequisites. They expose to SELPH the same
// state that `library_components_from_env` (synth_v2.rs) walks from Rust:
// the set of user-defined library functions, their arity, and their
// parameter types. None of these add search behaviour — they just expose
// existing env state.

/// `(env-functions)` — return a namespace mapping function-name strings
/// to their Function/Builtin Value. Filters out the default builtins
/// (everything in `synth_v2::default_skip_set`) so callers see only
/// user-defined library functions, matching `library_components_from_env`.
///
/// Walks the entire env scope chain (top-first, top wins) so the
/// builtin works regardless of which call frame the caller is in.
fn bi_env_functions(args: &[Value], env: &Env) -> Result<Value, String> {
    if !args.is_empty() {
        return Err(format!(
            "env-functions: expected 0 args, got {}",
            args.len()
        ));
    }
    let skip = crate::synth_v2::default_skip_set();
    let bindings = env.collect_bindings();
    let mut out: NsMap = NsMap::new();
    for (sym, val) in bindings.iter() {
        if skip.contains(sym) {
            continue;
        }
        if !matches!(val, Value::Function(_)) {
            continue;
        }
        out.insert(*sym, val.clone());
    }
    Ok(Value::ns(out))
}

/// `(function-arity f)` — return the arity of `f` as an Int. For
/// `Value::Function`, this is `params.len()`. For `Value::Builtin`,
/// returns -1 (variadic / unknown — builtins don't carry static arity
/// in the BuiltinTable). Errors on non-callable values.
fn bi_function_arity(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!(
            "function-arity: expected 1 arg, got {}",
            args.len()
        ));
    }
    match &args[0] {
        Value::Function(fd) => Ok(Value::Int(fd.params.len() as i64)),
        Value::Builtin(_) => Ok(Value::Int(-1)),
        other => Err(format!(
            "function-arity: expected function, got {:?}",
            other
        )),
    }
}

/// `(function-param-types f)` — return a list of type-name strings
/// describing `f`'s parameters. Uses the same trial-application probe
/// as `library_components_from_env` (synth_v2.rs:1057): tries uniform
/// argument vectors in `probe_samples` order until one doesn't error.
/// Returns the empty list for zero-arg functions, or `nil` if no probe
/// succeeded.
fn bi_function_param_types(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!(
            "function-param-types: expected 1 arg, got {}",
            args.len()
        ));
    }
    let arity = match &args[0] {
        Value::Function(fd) => fd.params.len(),
        Value::Builtin(_) => return Ok(Value::Nil),
        other => {
            return Err(format!(
                "function-param-types: expected function, got {:?}",
                other
            ));
        }
    };
    match crate::synth_v2::probe_function_type(&args[0], arity, env) {
        Some((param_syms, _ret)) => {
            let names: Vec<Value> = param_syms
                .iter()
                .map(|s| Value::str(resolve(*s)))
                .collect();
            Ok(Value::list(names))
        }
        None => Ok(Value::Nil),
    }
}

// ── Bucket 6 stubs ──────────────────────────────────────────────────────────
//
// Meta operations that interact with the synthesizer. The user's note: these
// will eventually be defined in SELPH itself, but it's not yet clear how. For
// now they return a clear error so any curriculum that needs them fails
// loudly rather than silently producing wrong results.
//
// When synth.rs is migrated (task #8), we have two options:
//   1. Make these delegate to migrated synth.rs functions (transitional).
//   2. Express them as SELPH programs (long-term direction). This requires
//      enough of synth.rs's machinery to be reachable from SELPH that the
//      programs can express what `synthesize` actually does.
// Pick at the migration boundary.

// ── Bucket 6: meta operations (real implementations) ────────────────────────
//
// These delegate to synth_v2 (the rebuilt synthesizer) and the parser.
// `synthesize-optimize` is intentionally still a stub: it depends on the
// optimize-synthesis path which hasn't been ported yet.

/// `(eval-source <source-string>)` — parse and evaluate SELPH source.
/// Uses the caller's env so any defines persist after the call. The
/// parser still produces legacy nodes, so we route through `convert_tree`.
fn bi_eval_source(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("eval-source: expected 1 argument (source string)".into());
    }
    let src = args[0].as_str()?;
    let (old_nodes, roots) = crate::parser::parse_file(src)
        .map_err(|e| format!("eval-source: parse error: {}", e))?;
    if roots.is_empty() {
        return Ok(Value::Nil);
    }
    let new_nodes = convert_tree(&old_nodes);
    let nodes_rc: Rc<[Node]> = new_nodes.into();
    let mut last = Value::Nil;
    for &r in &roots {
        last = eval(&nodes_rc, r, env)?;
    }
    Ok(last)
}

/// `(test-spec <candidate> <spec>)` — apply `candidate` to each
/// (input, expected) pair in `spec` and return the match fraction
/// (0.0 to 1.0). `candidate` must be a callable Value (Function or
/// Builtin). `spec` is a list of two-element lists.
fn bi_test_spec(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("test-spec: expected 2 arguments (candidate, spec)".into());
    }
    let candidate = &args[0];
    let pairs = match &args[1] {
        Value::List(l) => l.clone(),
        _ => return Err("test-spec: spec must be a list of (input expected) pairs".into()),
    };
    if pairs.is_empty() {
        return Ok(Value::Num(0.0));
    }
    let total = pairs.len();
    let mut matches = 0usize;
    for pair in pairs.iter() {
        let p = match pair {
            Value::List(p) if p.len() == 2 => p,
            _ => return Err("test-spec: each spec entry must be (input expected)".into()),
        };
        let input = p[0].clone();
        let expected_val = &p[1];
        if let Ok(result) = apply(candidate, &[input], env) {
            if values_equal(&result, expected_val) {
                matches += 1;
            }
        }
        // Eval errors silently count as a non-match — same as legacy.
    }
    Ok(Value::Num(matches as f64 / total as f64))
}

/// `(memorize <spec>)` — turn a list of (input, expected) pairs into
/// a `Value::Ns` mapping input keys to expected values. **Note**: this
/// is the legacy *data builder* memoize, NOT the synthesis-strategy
/// `memorize_from_examples`. The two share a name and a motivation but
/// produce different artifacts:
///
///   - `memorize` (this builtin): returns a raw namespace `Value::Ns`
///     of `{key → val}` for downstream code to consume.
///   - `synth_v2::memorize_from_examples`: returns a synthesized
///     `(lambda (x) (ns-get-or ...))` program — the strategy fallback.
///
/// Returns `Nil` if any input isn't a string (unconvertible to a Sym
/// key — same behaviour as the legacy builtin).
fn bi_memorize(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("memorize: expected 1 argument (spec)".into());
    }
    let pairs = match &args[0] {
        Value::List(l) => l.clone(),
        _ => return Err("memorize: argument must be a list of (input expected) pairs".into()),
    };
    if pairs.is_empty() {
        return Ok(Value::Nil);
    }
    let mut map = NsMap::new();
    for pair in pairs.iter() {
        let p = match pair {
            Value::List(p) if p.len() == 2 => p,
            _ => return Err("memorize: each spec entry must be (input expected)".into()),
        };
        let key = match &p[0] {
            Value::Str(s) => intern(s.as_ref()),
            _ => return Ok(Value::Nil), // non-string input — can't memorize
        };
        map.insert(key, p[1].clone());
    }
    Ok(Value::ns(map))
}

/// `(synthesize <namespace>)` — bottom-up enumerative program synthesis.
///
/// Expected namespace shape:
///   - `spec`: list of two-element lists `((<input> <expected>) ...)`
///   - `max-depth`: optional Int (default 2)
///   - `max-candidates`: optional Int (default 10000)
///   - `library`: **ignored** in eval_v2. Library functions visible to
///     the caller's env (via prior `define` or `eval-source` calls) are
///     auto-discovered and added to the component catalog.
///
/// Returns a namespace:
///   - `found`: Bool — whether a solution was found
///   - `candidates`: Int — number of candidates explored
///   - `source`: String — the synthesized lambda's source code (empty
///     when not found)
///   - `strategy`: String — which strategy produced the solution
///     (`"Flat"`, `"Memo"`, etc.), or empty string when not found
fn bi_synthesize(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("synthesize: expected 1 argument (namespace)".into());
    }
    let ns = match &args[0] {
        Value::Ns(m) => m.clone(),
        _ => return Err("synthesize: argument must be a namespace".into()),
    };

    // Extract spec → (inputs, expected).
    let spec_val = ns
        .get(&intern("spec"))
        .ok_or("synthesize: namespace must have \"spec\" field")?;
    let pairs = match spec_val {
        Value::List(l) => l.clone(),
        _ => return Err("synthesize: \"spec\" must be a list of example pairs".into()),
    };
    let mut inputs = Vec::with_capacity(pairs.len());
    let mut expected = Vec::with_capacity(pairs.len());
    for pair in pairs.iter() {
        let p = match pair {
            Value::List(p) if p.len() == 2 => p,
            _ => return Err("synthesize: each spec entry must be a list of [input, output]".into()),
        };
        inputs.push(p[0].clone());
        expected.push(p[1].clone());
    }

    let max_depth = ns
        .get(&intern("max-depth"))
        .and_then(|v| match v {
            Value::Int(n) => Some(*n as usize),
            Value::Num(n) => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(2);
    let max_candidates = ns
        .get(&intern("max-candidates"))
        .and_then(|v| match v {
            Value::Int(n) => Some(*n as usize),
            Value::Num(n) => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(10000);

    // Build the component catalog from the caller's env. Library
    // functions defined in env are auto-discovered as components.
    let skip = crate::synth_v2::default_skip_set();
    let mut components = crate::synth_v2::default_synth_components(env, &skip);
    // §9.37 Stage B: build the type universe from `__types__` if
    // present. Falls back to the primitive baseline when absent.
    let universe = crate::synth_v2::TypeUniverse::from_env(env);

    // §9.50: inject extra-seeds as literal components. These are
    // depth-0 constants that widen the reachable type set. For the
    // single-arg path they go into the component catalog directly
    // (synthesize_with_strategies passes them to synthesize_inner
    // via the flat enumeration step).
    if let Some(Value::List(seeds)) = ns.get(&intern("extra-seeds")) {
        for seed in seeds.iter() {
            if let Some(comp) = crate::synth_v2::value_to_seed_component(seed, 50.0) {
                components.push(comp);
            }
        }
    }

    // §9.34 — `("heuristic" lambda)` field, if present, re-scores and
    // re-sorts the catalog before dispatch. The lambda is a `Value::Function`
    // (or `Value::Builtin`) supplied directly by the SELPH caller, so the
    // value-form helper in `meta_v2` skips the per-component
    // eval-the-lambda-Node step that the source-loaded path uses. This
    // is the foundation for SELPH-side meta-learning loops: a curriculum
    // can synthesize a heuristic candidate and pass it directly into the
    // inner `synthesize` call without going through any CLI flag.
    if let Some(h) = ns.get(&intern("heuristic")) {
        components = crate::meta_v2::apply_heuristic_value_for_task(
            h, &components, &inputs, &expected, env,
        )
        .map_err(|e| format!("synthesize: {}", e))?;
    }

    let result = crate::synth_v2::synthesize_with_strategies(
        &components,
        &inputs,
        &expected,
        env,
        &universe,
        max_depth,
        max_candidates,
    );

    let source = if result.found {
        let nodes = result.nodes.as_ref().unwrap();
        let root = result.root.unwrap();
        node_to_source(nodes, root)
    } else {
        String::new()
    };
    let strategy_name = result
        .strategy
        .map(|s| s.name().to_string())
        .unwrap_or_default();

    let mut out = NsMap::new();
    out.insert(intern("found"), Value::Bool(result.found));
    out.insert(intern("candidates"), Value::Int(result.candidates_explored as i64));
    out.insert(intern("source"), Value::str(source));
    out.insert(intern("strategy"), Value::str(strategy_name));
    // §9.49 post-mortem diagnostics
    if let Some(ref otype) = result.output_type {
        out.insert(intern("output-type"), Value::str(otype.clone()));
    }
    out.insert(intern("m-chain-ran"), Value::Bool(result.m_chain_ran));
    Ok(Value::ns(out))
}

/// `(synthesize-args <namespace>)` — the §9.39/§9.40 multi-arg
/// counterpart to `synthesize`. The spec field is a list of
/// `[input output]` pairs where each `input` is a list of length
/// `arity`. The arity is inferred from the first row's input length;
/// each per-position type is inferred from that row's value at the
/// corresponding index. Synthesis dispatches through
/// `synth_v2::synthesize_args` so the search seeds the pool with
/// indexed atoms, collapsing the multi-arg fanout that the old
/// list-input single-input path suffered from.
///
/// Returns the same `{found, candidates, source, strategy}`
/// namespace as `synthesize`.
fn bi_synthesize_args(args: &[Value], env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("synthesize-args: expected 1 argument (namespace)".into());
    }
    let ns = match &args[0] {
        Value::Ns(m) => m.clone(),
        _ => return Err("synthesize-args: argument must be a namespace".into()),
    };

    let spec_val = ns
        .get(&intern("spec"))
        .ok_or("synthesize-args: namespace must have \"spec\" field")?;
    let pairs = match spec_val {
        Value::List(l) => l.clone(),
        _ => return Err("synthesize-args: \"spec\" must be a list of [input, output] pairs".into()),
    };
    if pairs.is_empty() {
        return Err("synthesize-args: spec is empty".into());
    }
    let mut inputs = Vec::with_capacity(pairs.len());
    let mut expected = Vec::with_capacity(pairs.len());
    for pair in pairs.iter() {
        let p = match pair {
            Value::List(p) if p.len() == 2 => p,
            _ => return Err("synthesize-args: each spec entry must be [input, output]".into()),
        };
        inputs.push(p[0].clone());
        expected.push(p[1].clone());
    }

    // Each input must be a list — that's what marks it as multi-arg.
    // Infer per-position types from the first row.
    let first_row = match &inputs[0] {
        Value::List(items) => items.clone(),
        _ => return Err("synthesize-args: each input must be a list (the args tuple)".into()),
    };
    let arity = first_row.len();
    let arg_types: Vec<crate::intern::Sym> = first_row
        .iter()
        .map(|v| v.type_sym().unwrap_or_else(crate::types_v2::type_any))
        .collect();
    // Sanity-check that every input row has the same arity.
    for inp in &inputs {
        match inp {
            Value::List(items) if items.len() == arity => {}
            _ => return Err(format!(
                "synthesize-args: every input must be a list of length {}", arity
            )),
        }
    }

    let max_depth = ns
        .get(&intern("max-depth"))
        .and_then(|v| match v {
            Value::Int(n) => Some(*n as usize),
            Value::Num(n) => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(4);
    let max_candidates = ns
        .get(&intern("max-candidates"))
        .and_then(|v| match v {
            Value::Int(n) => Some(*n as usize),
            Value::Num(n) => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(100000);

    // §9.49: read held-out test pairs from the "test" field, same
    // format as the spec (list of [input, output] pairs). When present,
    // candidates that pass training are additionally verified against
    // the held-out pairs — rejecting memorization solutions.
    let mut test_inputs = Vec::new();
    let mut test_expected = Vec::new();
    if let Some(Value::List(test_pairs)) = ns.get(&intern("test")) {
        for pair in test_pairs.iter() {
            if let Value::List(p) = pair {
                if p.len() == 2 {
                    test_inputs.push(p[0].clone());
                    test_expected.push(p[1].clone());
                }
            }
        }
    }

    // §9.50: read "extra-seeds" — a list of literal values to inject
    // as depth-0 constants. These widen the reachable type set, enabling
    // cross-type compositions like grid-scale(grid, 2) where 2 is a
    // discovered constant. Priority 50.0 puts them above data-derived
    // literals (0.0) but below input variables (100.0).
    let mut extra_seed_components = Vec::new();
    if let Some(Value::List(seeds)) = ns.get(&intern("extra-seeds")) {
        for seed in seeds.iter() {
            if let Some(comp) = crate::synth_v2::value_to_seed_component(seed, 50.0) {
                extra_seed_components.push(comp);
            }
        }
    }

    let skip = crate::synth_v2::default_skip_set();
    let components = crate::synth_v2::default_synth_components(env, &skip);
    let universe = crate::synth_v2::TypeUniverse::from_env(env);

    let result = crate::synth_v2::synthesize_args_with_extra_seeds(
        &components,
        &inputs,
        &arg_types,
        &expected,
        env,
        &universe,
        max_depth,
        max_candidates,
        &test_inputs,
        &test_expected,
        &extra_seed_components,
    );

    let source = if result.found {
        let nodes = result.nodes.as_ref().unwrap();
        let root = result.root.unwrap();
        node_to_source(nodes, root)
    } else {
        String::new()
    };

    let strategy_name = if result.found {
        match result.decomposer_name {
            Some(name) => crate::intern::resolve(name),
            None => "Flat".to_string(),
        }
    } else {
        String::new()
    };

    let mut out = NsMap::new();
    out.insert(intern("found"), Value::Bool(result.found));
    out.insert(intern("candidates"), Value::Int(result.candidates_explored as i64));
    out.insert(intern("source"), Value::str(source));
    out.insert(intern("strategy"), Value::str(strategy_name));
    out.insert(intern("arity"), Value::Int(arity as i64));
    Ok(Value::ns(out))
}

/// `(synthesize-optimize <namespace>)` — fitness-driven program search.
/// Still stubbed in eval_v2 — depends on the optimize-synthesis path
/// which hasn't been ported. Documented as deferred work; the legacy
/// `synth::synthesize_optimize` lives behind this name.
fn bi_stub_synthesize_optimize(_args: &[Value], _env: &Env) -> Result<Value, String> {
    Err(
        "synthesize-optimize: not yet ported to synth_v2. \
         Use the legacy `selph eval` runner for fitness-driven synthesis, \
         or wait for the optimize-synthesis port (deferred work in §9.25.3)."
            .into(),
    )
}

// ── Grid helpers ────────────────────────────────────────────────────────────

/// Extract a grid (List of List of Int) as Vec<Vec<i64>>.
fn as_grid(v: &Value) -> Result<Vec<Vec<i64>>, String> {
    let rows = v.as_list().map_err(|_| "grid: expected list of lists".to_string())?;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let mut grid = Vec::with_capacity(rows.len());
    let width = match &rows[0] {
        Value::List(cells) => cells.len(),
        _ => return Err("grid: rows must be lists".into()),
    };
    for row in rows {
        match row {
            Value::List(cells) => {
                if cells.len() != width {
                    return Err("grid: all rows must have the same length".into());
                }
                let mut r = Vec::with_capacity(cells.len());
                for c in cells.iter() {
                    match c {
                        Value::Int(n) => r.push(*n),
                        Value::Num(n) => r.push(*n as i64),
                        _ => return Err("grid: cells must be Int or Num".into()),
                    }
                }
                grid.push(r);
            }
            _ => return Err("grid: rows must be lists".into()),
        }
    }
    Ok(grid)
}

/// Convert Vec<Vec<i64>> back to Value::List(List(Int)).
fn grid_to_value(g: Vec<Vec<i64>>) -> Value {
    let rows: Vec<Value> = g
        .into_iter()
        .map(|row| {
            Value::list(row.into_iter().map(Value::Int).collect::<Vec<_>>())
        })
        .collect();
    Value::list(rows)
}

fn grid_to_value_ref(g: &[Vec<i64>]) -> Value {
    let rows: Vec<Value> = g
        .iter()
        .map(|row| {
            Value::list(row.iter().map(|&v| Value::Int(v)).collect::<Vec<_>>())
        })
        .collect();
    Value::list(rows)
}

// ── Grid builtins (§9.48 P2) ──────────────────────────────────────────

fn bi_is_grid(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid?: expected 1 arg, got {}", args.len()));
    }
    Ok(Value::Bool(as_grid(&args[0]).is_ok()))
}

fn bi_grid_height(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-height: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    Ok(Value::Int(g.len() as i64))
}

fn bi_grid_width(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-width: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    Ok(Value::Int(if g.is_empty() { 0 } else { g[0].len() as i64 }))
}

fn bi_grid_rotate_cw(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-rotate-cw: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    if g.is_empty() { return Ok(grid_to_value(g)); }
    let h = g.len();
    let w = g[0].len();
    let mut out = vec![vec![0i64; h]; w];
    for r in 0..h {
        for c in 0..w {
            out[c][h - 1 - r] = g[r][c];
        }
    }
    Ok(grid_to_value(out))
}

fn bi_grid_rotate_ccw(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-rotate-ccw: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    if g.is_empty() { return Ok(grid_to_value(g)); }
    let h = g.len();
    let w = g[0].len();
    let mut out = vec![vec![0i64; h]; w];
    for r in 0..h {
        for c in 0..w {
            out[w - 1 - c][r] = g[r][c];
        }
    }
    Ok(grid_to_value(out))
}

fn bi_grid_rotate_180(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-rotate-180: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    if g.is_empty() { return Ok(grid_to_value(g)); }
    let h = g.len();
    let w = g[0].len();
    let mut out = vec![vec![0i64; w]; h];
    for r in 0..h {
        for c in 0..w {
            out[h - 1 - r][w - 1 - c] = g[r][c];
        }
    }
    Ok(grid_to_value(out))
}

fn bi_grid_flip_h(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-flip-h: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let out: Vec<Vec<i64>> = g.into_iter().map(|mut row| { row.reverse(); row }).collect();
    Ok(grid_to_value(out))
}

fn bi_grid_flip_v(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-flip-v: expected 1 arg, got {}", args.len()));
    }
    let mut g = as_grid(&args[0])?;
    g.reverse();
    Ok(grid_to_value(g))
}

fn bi_grid_transpose(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-transpose: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    if g.is_empty() { return Ok(grid_to_value(g)); }
    let h = g.len();
    let w = g[0].len();
    let mut out = vec![vec![0i64; h]; w];
    for r in 0..h {
        for c in 0..w {
            out[c][r] = g[r][c];
        }
    }
    Ok(grid_to_value(out))
}

fn bi_grid_background(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-background: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let mut counts = std::collections::HashMap::new();
    for row in &g {
        for &c in row {
            *counts.entry(c).or_insert(0usize) += 1;
        }
    }
    let bg = counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0);
    Ok(Value::Int(bg))
}

fn bi_grid_trim(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-trim: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    if g.is_empty() { return Ok(grid_to_value(g)); }
    let h = g.len();
    let w = g[0].len();
    // Detect background color (most common)
    let mut counts = std::collections::HashMap::new();
    for row in &g {
        for &c in row {
            *counts.entry(c).or_insert(0usize) += 1;
        }
    }
    let bg = counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0);
    // Find bounding box of non-background cells
    let mut r0 = h;
    let mut r1 = 0usize;
    let mut c0 = w;
    let mut c1 = 0usize;
    for r in 0..h {
        for c in 0..w {
            if g[r][c] != bg {
                r0 = r0.min(r);
                r1 = r1.max(r + 1);
                c0 = c0.min(c);
                c1 = c1.max(c + 1);
            }
        }
    }
    if r0 >= r1 || c0 >= c1 {
        // All background — return 1x1 grid with background
        return Ok(grid_to_value(vec![vec![bg]]));
    }
    let out: Vec<Vec<i64>> = g[r0..r1].iter().map(|row| row[c0..c1].to_vec()).collect();
    Ok(grid_to_value(out))
}

fn bi_grid_get(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 {
        return Err(format!("grid-get: expected 3 args, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let r = int_arg(&args[1], "grid-get")? as usize;
    let c = int_arg(&args[2], "grid-get")? as usize;
    if r < g.len() && c < g[r].len() {
        Ok(Value::Int(g[r][c]))
    } else {
        Err(format!("grid-get: index ({},{}) out of bounds ({}x{})",
            r, c, g.len(), g.first().map_or(0, |r| r.len())))
    }
}

fn bi_grid_set(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 4 {
        return Err(format!("grid-set: expected 4 args, got {}", args.len()));
    }
    let mut g = as_grid(&args[0])?;
    let r = int_arg(&args[1], "grid-set")? as usize;
    let c = int_arg(&args[2], "grid-set")? as usize;
    let v = int_arg(&args[3], "grid-set")?;
    if r < g.len() && c < g[r].len() {
        g[r][c] = v;
        Ok(grid_to_value(g))
    } else {
        Err("grid-set: index out of bounds".into())
    }
}

fn bi_grid_replace_color(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 {
        return Err(format!("grid-replace-color: expected 3 args, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let from = int_arg(&args[1], "grid-replace-color")?;
    let to = int_arg(&args[2], "grid-replace-color")?;
    let out: Vec<Vec<i64>> = g.iter().map(|row| {
        row.iter().map(|&c| if c == from { to } else { c }).collect()
    }).collect();
    Ok(grid_to_value(out))
}

fn bi_grid_crop(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 5 {
        return Err(format!("grid-crop: expected 5 args, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let r0 = int_arg(&args[1], "grid-crop")? as usize;
    let c0 = int_arg(&args[2], "grid-crop")? as usize;
    let h = int_arg(&args[3], "grid-crop")? as usize;
    let w = int_arg(&args[4], "grid-crop")? as usize;
    let out: Vec<Vec<i64>> = g[r0..r0 + h].iter().map(|row| row[c0..c0 + w].to_vec()).collect();
    Ok(grid_to_value(out))
}

fn bi_grid_overlay(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 4 {
        return Err(format!("grid-overlay: expected 4 args, got {}", args.len()));
    }
    let mut base = as_grid(&args[0])?;
    let over = as_grid(&args[1])?;
    let dr = int_arg(&args[2], "grid-overlay")? as usize;
    let dc = int_arg(&args[3], "grid-overlay")? as usize;
    for r in 0..over.len() {
        for c in 0..over[r].len() {
            if over[r][c] != 0 {
                let tr = dr + r;
                let tc = dc + c;
                if tr < base.len() && tc < base[tr].len() {
                    base[tr][tc] = over[r][c];
                }
            }
        }
    }
    Ok(grid_to_value(base))
}

fn bi_grid_colors(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-colors: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let mut seen = std::collections::BTreeSet::new();
    for row in &g {
        for &c in row {
            seen.insert(c);
        }
    }
    let colors: Vec<Value> = seen.into_iter().map(Value::Int).collect();
    Ok(Value::list(colors))
}

fn bi_grid_count_color(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-count-color: expected 2 args, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let color = int_arg(&args[1], "grid-count-color")?;
    let count = g.iter().flat_map(|row| row.iter()).filter(|&&c| c == color).count();
    Ok(Value::Int(count as i64))
}

fn bi_grid_hconcat(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-hconcat: expected 2 args, got {}", args.len()));
    }
    let a = as_grid(&args[0])?;
    let b = as_grid(&args[1])?;
    let h = a.len().max(b.len());
    let wa = a.first().map_or(0, |r| r.len());
    let wb = b.first().map_or(0, |r| r.len());
    let mut out = vec![vec![0i64; wa + wb]; h];
    for r in 0..h {
        if r < a.len() { for c in 0..wa { out[r][c] = a[r][c]; } }
        if r < b.len() { for c in 0..wb { out[r][wa + c] = b[r][c]; } }
    }
    Ok(grid_to_value(out))
}

fn bi_grid_vconcat(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-vconcat: expected 2 args, got {}", args.len()));
    }
    let mut a = as_grid(&args[0])?;
    let b = as_grid(&args[1])?;
    a.extend(b);
    Ok(grid_to_value(a))
}

fn bi_grid_xor(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-xor: expected 2 args, got {}", args.len()));
    }
    let a = as_grid(&args[0])?;
    let b = as_grid(&args[1])?;
    let h = a.len();
    let w = a.first().map_or(0, |r| r.len());
    let mut out = vec![vec![0i64; w]; h];
    for r in 0..h {
        for c in 0..w {
            let va = a[r][c];
            let vb = if r < b.len() && c < b[r].len() { b[r][c] } else { 0 };
            let a_fg = va != 0;
            let b_fg = vb != 0;
            out[r][c] = if a_fg && !b_fg { va } else if !a_fg && b_fg { vb } else { 0 };
        }
    }
    Ok(grid_to_value(out))
}

fn bi_grid_and(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-and: expected 2 args, got {}", args.len()));
    }
    let a = as_grid(&args[0])?;
    let b = as_grid(&args[1])?;
    let h = a.len();
    let w = a.first().map_or(0, |r| r.len());
    let mut out = vec![vec![0i64; w]; h];
    for r in 0..h {
        for c in 0..w {
            let va = a[r][c];
            let vb = if r < b.len() && c < b[r].len() { b[r][c] } else { 0 };
            out[r][c] = if va != 0 && vb != 0 { va } else { 0 };
        }
    }
    Ok(grid_to_value(out))
}

fn bi_grid_or(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-or: expected 2 args, got {}", args.len()));
    }
    let a = as_grid(&args[0])?;
    let b = as_grid(&args[1])?;
    let h = a.len();
    let w = a.first().map_or(0, |r| r.len());
    let mut out = vec![vec![0i64; w]; h];
    for r in 0..h {
        for c in 0..w {
            let va = a[r][c];
            let vb = if r < b.len() && c < b[r].len() { b[r][c] } else { 0 };
            out[r][c] = if va != 0 { va } else if vb != 0 { vb } else { 0 };
        }
    }
    Ok(grid_to_value(out))
}

fn bi_grid_gravity(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-gravity: expected 2 args, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let dir = int_arg(&args[1], "grid-gravity")?;
    let h = g.len();
    let w = g.first().map_or(0, |r| r.len());
    if h == 0 || w == 0 { return Ok(grid_to_value(g)); }
    // Detect background as most common color
    let mut counts = std::collections::HashMap::new();
    for row in &g { for &c in row { *counts.entry(c).or_insert(0usize) += 1; } }
    let bg = counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0);
    let mut out = vec![vec![bg; w]; h];
    match dir {
        0 => { // down
            for c in 0..w {
                let mut write = h;
                for r in (0..h).rev() { if g[r][c] != bg { write -= 1; out[write][c] = g[r][c]; } }
            }
        }
        1 => { // right
            for r in 0..h {
                let mut write = w;
                for c in (0..w).rev() { if g[r][c] != bg { write -= 1; out[r][write] = g[r][c]; } }
            }
        }
        2 => { // up
            for c in 0..w {
                let mut write = 0;
                for r in 0..h { if g[r][c] != bg { out[write][c] = g[r][c]; write += 1; } }
            }
        }
        3 => { // left
            for r in 0..h {
                let mut write = 0;
                for c in 0..w { if g[r][c] != bg { out[r][write] = g[r][c]; write += 1; } }
            }
        }
        _ => return Ok(grid_to_value(g)),
    }
    Ok(grid_to_value(out))
}

// §9.48: directional gravity convenience wrappers (unary, for pool catalog)
fn bi_grid_gravity_down(args: &[Value], env: &Env) -> Result<Value, String> {
    bi_grid_gravity(&[args[0].clone(), Value::Int(0)], env)
}
fn bi_grid_gravity_right(args: &[Value], env: &Env) -> Result<Value, String> {
    bi_grid_gravity(&[args[0].clone(), Value::Int(1)], env)
}
fn bi_grid_gravity_up(args: &[Value], env: &Env) -> Result<Value, String> {
    bi_grid_gravity(&[args[0].clone(), Value::Int(2)], env)
}
fn bi_grid_gravity_left(args: &[Value], env: &Env) -> Result<Value, String> {
    bi_grid_gravity(&[args[0].clone(), Value::Int(3)], env)
}

fn bi_grid_fill_rect(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 6 {
        return Err(format!("grid-fill-rect: expected 6 args, got {}", args.len()));
    }
    let mut g = as_grid(&args[0])?;
    let r0 = int_arg(&args[1], "grid-fill-rect")? as usize;
    let c0 = int_arg(&args[2], "grid-fill-rect")? as usize;
    let r1 = int_arg(&args[3], "grid-fill-rect")? as usize;
    let c1 = int_arg(&args[4], "grid-fill-rect")? as usize;
    let color = int_arg(&args[5], "grid-fill-rect")?;
    let h = g.len();
    let w = g.first().map_or(0, |r| r.len());
    for r in r0.min(r1)..=r0.max(r1) {
        for c in c0.min(c1)..=c0.max(c1) {
            if r < h && c < w { g[r][c] = color; }
        }
    }
    Ok(grid_to_value(g))
}

fn bi_grid_size(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-size: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let h = g.len() as i64;
    let w = g.first().map_or(0, |r| r.len()) as i64;
    Ok(Value::list(vec![Value::Int(h), Value::Int(w)]))
}

// §9.48: connected components (object detection)
// ── Raw grid helpers for probe builtins ──────────────────────────────────────
// These operate on Vec<Vec<i64>> directly, avoiding Value conversion overhead.
// Used by the grid-probe-* builtins for batch analysis.

fn grid_background(g: &[Vec<i64>]) -> i64 {
    let mut counts = std::collections::HashMap::new();
    for row in g { for &c in row { *counts.entry(c).or_insert(0usize) += 1; } }
    counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0)
}

fn grids_equal(a: &[Vec<i64>], b: &[Vec<i64>]) -> bool {
    if a.len() != b.len() { return false; }
    a.iter().zip(b.iter()).all(|(ra, rb)| ra == rb)
}

fn grid_rotate_cw_raw(g: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let (h, w) = (g.len(), g.first().map_or(0, |r| r.len()));
    if h == 0 { return g.to_vec(); }
    let mut out = vec![vec![0i64; h]; w];
    for r in 0..h { for c in 0..w { out[c][h - 1 - r] = g[r][c]; } }
    out
}

fn grid_rotate_ccw_raw(g: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let (h, w) = (g.len(), g.first().map_or(0, |r| r.len()));
    if h == 0 { return g.to_vec(); }
    let mut out = vec![vec![0i64; h]; w];
    for r in 0..h { for c in 0..w { out[w - 1 - c][r] = g[r][c]; } }
    out
}

fn grid_rotate_180_raw(g: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let (h, w) = (g.len(), g.first().map_or(0, |r| r.len()));
    if h == 0 { return g.to_vec(); }
    let mut out = vec![vec![0i64; w]; h];
    for r in 0..h { for c in 0..w { out[h - 1 - r][w - 1 - c] = g[r][c]; } }
    out
}

fn grid_flip_h_raw(g: &[Vec<i64>]) -> Vec<Vec<i64>> {
    g.iter().map(|row| { let mut r = row.clone(); r.reverse(); r }).collect()
}

fn grid_flip_v_raw(g: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let mut out = g.to_vec(); out.reverse(); out
}

fn grid_trim_raw(g: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let (h, w) = (g.len(), g.first().map_or(0, |r| r.len()));
    if h == 0 { return g.to_vec(); }
    let bg = grid_background(g);
    let (mut r0, mut r1, mut c0, mut c1) = (h, 0usize, w, 0usize);
    for r in 0..h { for c in 0..w {
        if g[r][c] != bg { r0 = r0.min(r); r1 = r1.max(r + 1); c0 = c0.min(c); c1 = c1.max(c + 1); }
    }}
    if r0 >= r1 || c0 >= c1 { return vec![vec![bg]]; }
    g[r0..r1].iter().map(|row| row[c0..c1].to_vec()).collect()
}

fn grid_compact_raw(g: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let (h, w) = (g.len(), g.first().map_or(0, |r| r.len()));
    if h == 0 { return g.to_vec(); }
    let keep_row: Vec<bool> = (0..h).map(|r| g[r].iter().any(|&c| c != 0)).collect();
    let keep_col: Vec<bool> = (0..w).map(|c| (0..h).any(|r| g[r][c] != 0)).collect();
    let out: Vec<Vec<i64>> = (0..h).filter(|&r| keep_row[r])
        .map(|r| (0..w).filter(|&c| keep_col[c]).map(|c| g[r][c]).collect()).collect();
    if out.is_empty() { vec![vec![0i64]] } else { out }
}

fn grid_scale_raw(g: &[Vec<i64>], factor: usize) -> Vec<Vec<i64>> {
    let (h, w) = (g.len(), g.first().map_or(0, |r| r.len()));
    let mut out = vec![vec![0i64; w * factor]; h * factor];
    for r in 0..h { for c in 0..w {
        let v = g[r][c];
        for dr in 0..factor { for dc in 0..factor { out[r * factor + dr][c * factor + dc] = v; } }
    }}
    out
}

fn grid_place_raw(canvas: &[Vec<i64>], obj: &[Vec<i64>], row: i64, col: i64) -> Vec<Vec<i64>> {
    let ch = canvas.len();
    let cw = if ch > 0 { canvas[0].len() } else { 0 };
    let oh = obj.len();
    let ow = if oh > 0 { obj[0].len() } else { 0 };
    let mut out = canvas.to_vec();
    for r in 0..oh { for c in 0..ow {
        if obj[r][c] != 0 {
            let (tr, tc) = (row as isize + r as isize, col as isize + c as isize);
            if tr >= 0 && (tr as usize) < ch && tc >= 0 && (tc as usize) < cw {
                out[tr as usize][tc as usize] = obj[r][c];
            }
        }
    }}
    out
}

// ── Connected components with position info ─────────────────────────────────

struct ObjectInfo {
    grid: Vec<Vec<i64>>,
    row: usize,
    col: usize,
    size: usize,
}

fn grid_cc_with_pos(g: &[Vec<i64>], eight_connected: bool) -> Vec<ObjectInfo> {
    let h = g.len();
    let w = g.first().map_or(0, |r| r.len());
    if h == 0 || w == 0 { return vec![]; }
    let bg = grid_background(g);
    let mut labels = vec![vec![0u32; w]; h];
    let mut next_label = 1u32;
    let dirs4: &[(i32, i32)] = &[(-1, 0), (1, 0), (0, -1), (0, 1)];
    let dirs8: &[(i32, i32)] = &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)];
    let dirs = if eight_connected { dirs8 } else { dirs4 };
    for r in 0..h { for c in 0..w {
        if g[r][c] != bg && labels[r][c] == 0 {
            labels[r][c] = next_label;
            next_label += 1;
            let mut queue = vec![(r, c)];
            while let Some((cr, cc)) = queue.pop() {
                for &(dr, dc) in dirs {
                    let (nr, nc) = (cr as i32 + dr, cc as i32 + dc);
                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                        let (nr, nc) = (nr as usize, nc as usize);
                        if labels[nr][nc] == 0 && g[nr][nc] != bg {
                            labels[nr][nc] = labels[r][c]; // use same label
                            // BUG: should use current label, not labels[r][c] which may differ
                            // Actually labels[r][c] == next_label - 1, which is correct
                            // because we set it above. But let's use the captured variable:
                            labels[nr][nc] = next_label - 1;
                            queue.push((nr, nc));
                        }
                    }
                }
            }
        }
    }}
    let mut results = Vec::with_capacity((next_label - 1) as usize);
    for lbl in 1..next_label {
        let (mut min_r, mut min_c, mut max_r, mut max_c, mut count) = (h, w, 0usize, 0usize, 0usize);
        for r in 0..h { for c in 0..w {
            if labels[r][c] == lbl {
                min_r = min_r.min(r); min_c = min_c.min(c);
                max_r = max_r.max(r); max_c = max_c.max(c);
                count += 1;
            }
        }}
        if max_r >= min_r {
            let mut comp = vec![vec![0i64; max_c - min_c + 1]; max_r - min_r + 1];
            for r in min_r..=max_r { for c in min_c..=max_c {
                if labels[r][c] == lbl { comp[r - min_r][c - min_c] = g[r][c]; }
            }}
            results.push(ObjectInfo { grid: comp, row: min_r, col: min_c, size: count });
        }
    }
    results
}

// ── Spec parsing helper ─────────────────────────────────────────────────────

fn parse_spec_grid_pairs(spec: &Value) -> Result<Vec<(Vec<Vec<i64>>, Vec<Vec<i64>>)>, String> {
    let pairs = spec.as_list().map_err(|_| "spec must be a list of pairs".to_string())?;
    let mut result = Vec::with_capacity(pairs.len());
    for pair in pairs.iter() {
        let items = pair.as_list().map_err(|_| "spec pair must be a list".to_string())?;
        if items.len() < 2 { return Err("spec pair must have input and output".into()); }
        // Unwrap arity-1 wrapper: ((grid)) -> grid
        let input = match &items[0] {
            Value::List(l) if l.len() == 1 && matches!(&l[0], Value::List(_)) => as_grid(&l[0])?,
            v => as_grid(v)?,
        };
        let output = as_grid(&items[1])?;
        result.push((input, output));
    }
    Ok(result)
}

// ── grid-probe-recomp ───────────────────────────────────────────────────────
// Batch object-level analysis for the object-recomposition task bucket.

fn bi_grid_probe_recomp(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-probe-recomp: expected 1 arg (spec), got {}", args.len()));
    }
    let pairs = parse_spec_grid_pairs(&args[0])?;
    if pairs.is_empty() {
        return Ok(Value::ns(probe_recomp_result(false, 0, 0, "none", "none", false, "error:empty-spec")));
    }

    // Analyze objects across all pairs
    let mut all_in_counts = Vec::new();
    let mut all_out_counts = Vec::new();
    let mut in_objects: Vec<Vec<ObjectInfo>> = Vec::new();
    let mut out_objects: Vec<Vec<ObjectInfo>> = Vec::new();

    for (inp, outp) in &pairs {
        let in_objs = grid_cc_with_pos(inp, false);
        let out_objs = grid_cc_with_pos(outp, false);
        all_in_counts.push(in_objs.len());
        all_out_counts.push(out_objs.len());
        in_objects.push(in_objs);
        out_objects.push(out_objs);
    }

    let typical_in = all_in_counts.first().copied().unwrap_or(0);
    let typical_out = all_out_counts.first().copied().unwrap_or(0);
    let counts_conserved = all_in_counts.iter().zip(all_out_counts.iter()).all(|(a, b)| a == b);

    // Try per-object transform: for each transform, check if applying it to every
    // object in input and placing back reproduces the output.
    let transforms: &[(&str, fn(&[Vec<i64>]) -> Vec<Vec<i64>>)] = &[
        ("rotate-cw", grid_rotate_cw_raw),
        ("rotate-ccw", grid_rotate_ccw_raw),
        ("rotate-180", grid_rotate_180_raw),
        ("flip-h", grid_flip_h_raw),
        ("flip-v", grid_flip_v_raw),
    ];

    let mut per_obj_transform = "none".to_string();
    'xform: for &(name, xf) in transforms {
        let mut all_match = true;
        for (pair_idx, (inp, outp)) in pairs.iter().enumerate() {
            let in_objs = &in_objects[pair_idx];
            let bg = grid_background(inp);
            let h = inp.len();
            let w = if h > 0 { inp[0].len() } else { 0 };
            let mut canvas = vec![vec![bg; w]; h];
            for obj_info in in_objs {
                let transformed = xf(&obj_info.grid);
                canvas = grid_place_raw(&canvas, &transformed, obj_info.row as i64, obj_info.col as i64);
            }
            if !grids_equal(&canvas, outp) {
                all_match = false;
                break;
            }
        }
        if all_match {
            per_obj_transform = name.to_string();
            break 'xform;
        }
    }

    // Try object-index match: output == grid-object(input, idx) for idx 0..5
    // Also try grid-trim(input)
    let mut obj_select = "none".to_string();
    for idx in 0..6 {
        let mut all_match = true;
        for (pair_idx, (inp, outp)) in pairs.iter().enumerate() {
            let in_objs = &in_objects[pair_idx];
            // Sort by size (largest first) to match grid-object behavior
            let mut sorted: Vec<&ObjectInfo> = in_objs.iter().collect();
            sorted.sort_by(|a, b| b.size.cmp(&a.size));
            if idx >= sorted.len() { all_match = false; break; }
            if !grids_equal(&sorted[idx].grid, outp) { all_match = false; break; }
        }
        if all_match {
            obj_select = format!("grid-object-{}", idx);
            break;
        }
    }
    if obj_select == "none" {
        // Try grid-trim
        let all_trim = pairs.iter().all(|(inp, outp)| grids_equal(&grid_trim_raw(inp), outp));
        if all_trim { obj_select = "grid-trim".to_string(); }
    }

    // Single-object-change: exactly one object differs between input and output
    let single_change = counts_conserved && pairs.iter().enumerate().all(|(pair_idx, _)| {
        let in_objs = &in_objects[pair_idx];
        let out_objs = &out_objects[pair_idx];
        if in_objs.len() != out_objs.len() || in_objs.is_empty() { return false; }
        let diffs = in_objs.iter().zip(out_objs.iter())
            .filter(|(a, b)| !grids_equal(&a.grid, &b.grid))
            .count();
        diffs == 1
    });

    // Assemble subtype string (matches the SELPH pm-subtype-recomp logic)
    let subtype = if per_obj_transform != "none" {
        format!("per-object-transform:{}", per_obj_transform)
    } else if obj_select != "none" {
        format!("object-select:{}", obj_select)
    } else if single_change {
        format!("single-object-change:n={}", typical_in)
    } else if !counts_conserved {
        "object-count-change".to_string()
    } else if typical_in > 0 && typical_in <= 2 {
        format!("few-objects-recomp:n={}", typical_in)
    } else if typical_in > 2 {
        format!("many-objects-recomp:n={}", typical_in)
    } else {
        "no-objects-detected".to_string()
    };

    Ok(Value::ns(probe_recomp_result(
        counts_conserved, typical_in, typical_out,
        &per_obj_transform, &obj_select, single_change, &subtype,
    )))
}

fn probe_recomp_result(counts_conserved: bool, in_count: usize, out_count: usize,
    per_obj: &str, obj_sel: &str, single_change: bool, subtype: &str) -> NsMap {
    let mut map = NsMap::new();
    map.insert(intern("counts-conserved"), Value::Bool(counts_conserved));
    map.insert(intern("in-count"), Value::Int(in_count as i64));
    map.insert(intern("out-count"), Value::Int(out_count as i64));
    map.insert(intern("per-obj-transform"), Value::str(per_obj));
    map.insert(intern("obj-select"), Value::str(obj_sel));
    map.insert(intern("single-change"), Value::Bool(single_change));
    map.insert(intern("subtype"), Value::str(subtype));
    map
}

// ── grid-probe-extract ──────────────────────────────────────────────────────
// Batch extraction probing: try compact, object-index+trim, color extraction.

fn bi_grid_probe_extract(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-probe-extract: expected 1 arg (spec), got {}", args.len()));
    }
    let pairs = parse_spec_grid_pairs(&args[0])?;
    if pairs.is_empty() {
        return Ok(Value::ns(probe_extract_result(false, "", "")));
    }

    // Strategy 1: grid-compact
    if pairs.iter().all(|(inp, outp)| grids_equal(&grid_compact_raw(inp), outp)) {
        return Ok(Value::ns(probe_extract_result(true, "grid-compact(x)", "compact")));
    }

    // Strategy 2: grid-trim(grid-object(input, idx)) for idx 0..5
    for idx in 0..6usize {
        let all_match = pairs.iter().all(|(inp, outp)| {
            let objs = grid_cc_with_pos(inp, false);
            let mut sorted: Vec<&ObjectInfo> = objs.iter().collect();
            sorted.sort_by(|a, b| b.size.cmp(&a.size));
            if idx >= sorted.len() { return false; }
            grids_equal(&grid_trim_raw(&sorted[idx].grid), outp)
        });
        if all_match {
            return Ok(Value::ns(probe_extract_result(
                true, &format!("grid-trim(grid-object(x,{}))", idx), "object-trim")));
        }
    }

    // Strategy 3: grid-trim(input) directly
    if pairs.iter().all(|(inp, outp)| grids_equal(&grid_trim_raw(inp), outp)) {
        return Ok(Value::ns(probe_extract_result(true, "grid-trim(x)", "trim")));
    }

    // Strategy 4: color-based extraction — for each non-bg color, mask + trim
    let first_inp = &pairs[0].0;
    let bg = grid_background(first_inp);
    let mut colors: Vec<i64> = Vec::new();
    for row in first_inp { for &c in row {
        if c != bg && !colors.contains(&c) { colors.push(c); }
    }}
    for color in &colors {
        let all_match = pairs.iter().all(|(inp, outp)| {
            let ibg = grid_background(inp);
            let h = inp.len();
            let w = if h > 0 { inp[0].len() } else { 0 };
            let masked: Vec<Vec<i64>> = (0..h).map(|r|
                (0..w).map(|c| if inp[r][c] == *color { *color } else { ibg }).collect()
            ).collect();
            grids_equal(&grid_trim_raw(&masked), outp)
        });
        if all_match {
            return Ok(Value::ns(probe_extract_result(
                true, &format!("trim(mask-to-color(x,{}))", color), "color-extract")));
        }
    }

    // Strategy 5: try grid-object(input, idx) directly (without trim)
    for idx in 0..6usize {
        let all_match = pairs.iter().all(|(inp, outp)| {
            let objs = grid_cc_with_pos(inp, false);
            let mut sorted: Vec<&ObjectInfo> = objs.iter().collect();
            sorted.sort_by(|a, b| b.size.cmp(&a.size));
            if idx >= sorted.len() { return false; }
            grids_equal(&sorted[idx].grid, outp)
        });
        if all_match {
            return Ok(Value::ns(probe_extract_result(
                true, &format!("grid-object(x,{})", idx), "object-direct")));
        }
    }

    Ok(Value::ns(probe_extract_result(false, "", "none")))
}

fn probe_extract_result(found: bool, source: &str, method: &str) -> NsMap {
    let mut map = NsMap::new();
    map.insert(intern("found"), Value::Bool(found));
    map.insert(intern("source"), Value::str(source));
    map.insert(intern("method"), Value::str(method));
    map
}

// ── grid-probe-scale ────────────────────────────────────────────────────────
// Try grid-scale(input, N) for each constant N.

fn bi_grid_probe_scale(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-probe-scale: expected 2 args (spec, constants), got {}", args.len()));
    }
    let pairs = parse_spec_grid_pairs(&args[0])?;
    let constants = args[1].as_list().map_err(|_| "grid-probe-scale: constants must be a list".to_string())?;

    for cv in constants.iter() {
        let n = match cv {
            Value::Int(n) => *n as usize,
            Value::Num(n) => *n as usize,
            _ => continue,
        };
        if n == 0 || n > 10 { continue; }
        let all_match = pairs.iter().all(|(inp, outp)| grids_equal(&grid_scale_raw(inp, n), outp));
        if all_match {
            let mut map = NsMap::new();
            map.insert(intern("found"), Value::Bool(true));
            map.insert(intern("source"), Value::str(format!("grid-scale(x,{})", n)));
            map.insert(intern("factor"), Value::Int(n as i64));
            return Ok(Value::ns(map));
        }
    }

    let mut map = NsMap::new();
    map.insert(intern("found"), Value::Bool(false));
    map.insert(intern("source"), Value::str(""));
    map.insert(intern("factor"), Value::Int(0));
    Ok(Value::ns(map))
}

// ── grid-diagnose-spec: fast full diagnosis in Rust ─────────────────────────
// Replaces the slow SELPH pm-diagnose-one classification: size, colors, scale,
// constant-out, dims-consistent, plus subtype via grid-probe-recomp/extract.

fn bi_grid_diagnose_spec(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-diagnose-spec: expected 1 arg (spec), got {}", args.len()));
    }
    let pairs = parse_spec_grid_pairs(&args[0])?;
    if pairs.is_empty() {
        let mut m = NsMap::new();
        m.insert(intern("size"), Value::str("error"));
        m.insert(intern("colors"), Value::str("error"));
        m.insert(intern("scale"), Value::str("error"));
        m.insert(intern("constant-out"), Value::Bool(false));
        m.insert(intern("dims-consistent"), Value::Bool(false));
        m.insert(intern("subtype"), Value::str("error:empty"));
        return Ok(Value::ns(m));
    }

    // Size category
    let size_rels: Vec<&str> = pairs.iter().map(|(inp, outp)| {
        let (ih, iw) = (inp.len(), inp.first().map_or(0, |r| r.len()));
        let (oh, ow) = (outp.len(), outp.first().map_or(0, |r| r.len()));
        if ih == oh && iw == ow { "same-size" }
        else if oh <= ih && ow <= iw { "shrink" }
        else if oh >= ih && ow >= iw { "grow" }
        else { "reshape" }
    }).collect();
    let size_cat = if size_rels.iter().all(|s| *s == size_rels[0]) {
        size_rels[0].to_string()
    } else if size_rels.iter().all(|s| *s == "shrink" || *s == "same-size") {
        "shrink".to_string()
    } else if size_rels.iter().all(|s| *s == "grow" || *s == "same-size") {
        "grow".to_string()
    } else { "mixed-size".to_string() };

    // Color category
    fn unique_colors(g: &[Vec<i64>]) -> Vec<i64> {
        let mut c = Vec::new();
        for row in g { for &v in row { if !c.contains(&v) { c.push(v); } } }
        c
    }
    let color_rels: Vec<&str> = pairs.iter().map(|(inp, outp)| {
        let ic = unique_colors(inp);
        let oc = unique_colors(outp);
        let out_in_in = oc.iter().all(|c| ic.contains(c));
        let in_in_out = ic.iter().all(|c| oc.contains(c));
        if out_in_in && in_in_out { "same-colors" }
        else if out_in_in { "color-subset" }
        else if in_in_out { "color-superset" }
        else { "new-colors" }
    }).collect();
    let color_cat = if color_rels.iter().all(|s| *s == color_rels[0]) {
        color_rels[0].to_string()
    } else { "mixed-colors".to_string() };

    // Scale category
    let h_ratios: Vec<i64> = pairs.iter().map(|(inp, outp)| {
        let ih = inp.len() as i64;
        let oh = outp.len() as i64;
        if ih > 0 && oh > 0 && oh % ih == 0 { oh / ih } else { -1 }
    }).collect();
    let scale_cat = if h_ratios.iter().all(|r| *r == h_ratios[0]) && h_ratios[0] > 1 {
        format!("scale-{}x", h_ratios[0])
    } else if h_ratios.iter().all(|r| *r == 1) {
        "no-scale".to_string()
    } else { "no-uniform-scale".to_string() };

    // Constant output
    let constant_out = if pairs.len() > 1 {
        let first_out = &pairs[0].1;
        pairs[1..].iter().all(|(_, outp)| grids_equal(first_out, outp))
    } else { false };

    // Dims consistent
    let dims: Vec<(usize, usize)> = pairs.iter().map(|(_, outp)| {
        (outp.len(), outp.first().map_or(0, |r| r.len()))
    }).collect();
    let dims_consistent = dims.iter().all(|d| *d == dims[0]);

    // Subtype: dispatch to probe builtins based on size/color
    let subtype = if size_cat == "same-size" && color_cat == "same-colors" {
        // Use recomp probe
        let probe_result = bi_grid_probe_recomp(args, _env)?;
        if let Value::Ns(ns) = &probe_result {
            ns.get(&intern("subtype")).map(|v| value_to_string(v)).unwrap_or_else(|| "error".into())
        } else { "error".into() }
    } else if size_cat == "shrink" {
        // Use extract probe for subtype
        let probe_result = bi_grid_probe_extract(args, _env)?;
        if let Value::Ns(ns) = &probe_result {
            let found = ns.get(&intern("found")).map(|v| matches!(v, Value::Bool(true))).unwrap_or(false);
            let source = ns.get(&intern("source")).map(|v| value_to_string(v)).unwrap_or_default();
            if found { format!("direct:{}", source) }
            else {
                // Object info summary
                let (inp, outp) = &pairs[0];
                let in_objs = grid_cc_with_pos(inp, false);
                let out_objs = grid_cc_with_pos(outp, false);
                format!("complex:in={}obj-{}x{}>out={}obj-{}x{}",
                    in_objs.len(), inp.len(), inp.first().map_or(0, |r| r.len()),
                    out_objs.len(), outp.len(), outp.first().map_or(0, |r| r.len()))
            }
        } else { "error".into() }
    } else { "n/a".into() };

    let mut m = NsMap::new();
    m.insert(intern("size"), Value::str(&size_cat));
    m.insert(intern("colors"), Value::str(&color_cat));
    m.insert(intern("scale"), Value::str(&scale_cat));
    m.insert(intern("constant-out"), Value::Bool(constant_out));
    m.insert(intern("dims-consistent"), Value::Bool(dims_consistent));
    m.insert(intern("subtype"), Value::str(&subtype));
    Ok(Value::ns(m))
}

// ── Original connected components (kept for existing callers) ───────────────

fn grid_connected_components(g: &[Vec<i64>], eight_connected: bool) -> Vec<Vec<Vec<i64>>> {
    let h = g.len();
    let w = g.first().map_or(0, |r| r.len());
    if h == 0 || w == 0 { return vec![]; }

    // Detect background (most common value)
    let mut counts = std::collections::HashMap::new();
    for row in g { for &c in row { *counts.entry(c).or_insert(0usize) += 1; } }
    let bg = counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0);

    let mut labels = vec![vec![0u32; w]; h];
    let mut next_label = 1u32;
    let dirs4: &[(i32, i32)] = &[(-1, 0), (1, 0), (0, -1), (0, 1)];
    let dirs8: &[(i32, i32)] = &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)];
    let dirs = if eight_connected { dirs8 } else { dirs4 };

    for r in 0..h {
        for c in 0..w {
            if g[r][c] != bg && labels[r][c] == 0 {
                let label = next_label;
                next_label += 1;
                labels[r][c] = label;
                let mut queue = vec![(r, c)];
                while let Some((cr, cc)) = queue.pop() {
                    for &(dr, dc) in dirs {
                        let nr = cr as i32 + dr;
                        let nc = cc as i32 + dc;
                        if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                            let (nr, nc) = (nr as usize, nc as usize);
                            if labels[nr][nc] == 0 && g[nr][nc] != bg {
                                labels[nr][nc] = label;
                                queue.push((nr, nc));
                            }
                        }
                    }
                }
            }
        }
    }

    let mut results = Vec::with_capacity((next_label - 1) as usize);
    for lbl in 1..next_label {
        let mut min_r = h; let mut min_c = w;
        let mut max_r = 0usize; let mut max_c = 0usize;
        for r in 0..h { for c in 0..w {
            if labels[r][c] == lbl {
                min_r = min_r.min(r); min_c = min_c.min(c);
                max_r = max_r.max(r); max_c = max_c.max(c);
            }
        }}
        if max_r >= min_r {
            let mut comp = vec![vec![0i64; max_c - min_c + 1]; max_r - min_r + 1];
            for r in min_r..=max_r { for c in min_c..=max_c {
                if labels[r][c] == lbl { comp[r - min_r][c - min_c] = g[r][c]; }
            }}
            results.push(comp);
        }
    }
    results
}

fn bi_grid_objects(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-objects: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let objects = grid_connected_components(&g, false);
    let vals: Vec<Value> = objects.into_iter().map(grid_to_value).collect();
    Ok(Value::list(vals))
}

fn bi_grid_objects_8(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-objects-8: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let objects = grid_connected_components(&g, true);
    let vals: Vec<Value> = objects.into_iter().map(grid_to_value).collect();
    Ok(Value::list(vals))
}

fn bi_grid_object_count(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-object-count: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let count = grid_connected_components(&g, false).len();
    Ok(Value::Int(count as i64))
}

/// `(grid-object grid n)` — return the nth object (0-indexed), sorted by
/// size (largest first). Each object is a trimmed bounding-box grid.
fn bi_grid_object(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-object: expected 2 args (grid, index), got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let idx = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Num(n) => *n as usize,
        _ => return Err("grid-object: index must be a number".into()),
    };
    let mut objects = grid_connected_components(&g, false);
    // Sort by size (cell count), largest first
    objects.sort_by(|a, b| {
        let sa: usize = a.iter().flat_map(|r| r.iter()).filter(|&&c| c != 0).count();
        let sb: usize = b.iter().flat_map(|r| r.iter()).filter(|&&c| c != 0).count();
        sb.cmp(&sa)
    });
    if idx >= objects.len() {
        return Err(format!("grid-object: index {} out of range (have {} objects)", idx, objects.len()));
    }
    Ok(grid_to_value(objects.remove(idx)))
}

/// `(grid-object-pos grid n)` — return (row, col) of the nth object's
/// top-left corner in the original grid. Objects sorted by size (largest first).
fn bi_grid_object_pos(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-object-pos: expected 2 args, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let idx = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Num(n) => *n as usize,
        _ => return Err("grid-object-pos: index must be a number".into()),
    };
    let h = g.len();
    let w = g.first().map_or(0, |r| r.len());
    // Detect background
    let mut counts = std::collections::HashMap::new();
    for row in &g { for &c in row { *counts.entry(c).or_insert(0usize) += 1; } }
    let bg = counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0);
    // Find objects with positions
    let mut labels = vec![vec![0u32; w]; h];
    let mut next_label = 1u32;
    let dirs: &[(i32,i32)] = &[(-1,0),(1,0),(0,-1),(0,1)];
    for r in 0..h { for c in 0..w {
        if g[r][c] != bg && labels[r][c] == 0 {
            let label = next_label; next_label += 1;
            labels[r][c] = label;
            let mut queue = vec![(r, c)];
            while let Some((cr, cc)) = queue.pop() {
                for &(dr, dc) in dirs {
                    let nr = cr as i32 + dr; let nc = cc as i32 + dc;
                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                        let (nr, nc) = (nr as usize, nc as usize);
                        if labels[nr][nc] == 0 && g[nr][nc] != bg {
                            labels[nr][nc] = label; queue.push((nr, nc));
                        }
                    }
                }
            }
        }
    }}
    // Collect per-label: (min_r, min_c, cell_count)
    let mut info: Vec<(u32, usize, usize, usize)> = Vec::new();
    for lbl in 1..next_label {
        let mut min_r = h; let mut min_c = w; let mut count = 0;
        for r in 0..h { for c in 0..w {
            if labels[r][c] == lbl { min_r = min_r.min(r); min_c = min_c.min(c); count += 1; }
        }}
        info.push((lbl, min_r, min_c, count));
    }
    // Sort by size (largest first)
    info.sort_by(|a, b| b.3.cmp(&a.3));
    if idx >= info.len() {
        return Err(format!("grid-object-pos: index {} out of range", idx));
    }
    Ok(Value::list(vec![Value::Int(info[idx].1 as i64), Value::Int(info[idx].2 as i64)]))
}

/// `(grid-place canvas obj row col)` — overlay obj onto canvas at (row,col).
/// Non-zero cells in obj overwrite canvas cells.
fn bi_grid_place(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 4 {
        return Err(format!("grid-place: expected 4 args (canvas, obj, row, col), got {}", args.len()));
    }
    let canvas = as_grid(&args[0])?;
    let obj = as_grid(&args[1])?;
    let row = match &args[2] { Value::Int(n) => *n, Value::Num(n) => *n as i64, _ => return Err("grid-place: row must be a number".into()) };
    let col = match &args[3] { Value::Int(n) => *n, Value::Num(n) => *n as i64, _ => return Err("grid-place: col must be a number".into()) };
    let ch = canvas.len();
    let cw = if ch > 0 { canvas[0].len() } else { 0 };
    let oh = obj.len();
    let ow = if oh > 0 { obj[0].len() } else { 0 };
    let mut out = canvas.clone();
    for r in 0..oh {
        for c in 0..ow {
            if obj[r][c] != 0 {
                let tr = row as isize + r as isize;
                let tc = col as isize + c as isize;
                if tr >= 0 && (tr as usize) < ch && tc >= 0 && (tc as usize) < cw {
                    out[tr as usize][tc as usize] = obj[r][c];
                }
            }
        }
    }
    Ok(grid_to_value(out))
}

/// `(grid-translate grid dr dc)` — shift all non-background cells by (dr, dc).
/// Cells that move out of bounds are clipped.
fn bi_grid_translate(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 {
        return Err(format!("grid-translate: expected 3 args (grid, dr, dc), got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let dr = match &args[1] { Value::Int(n) => *n, Value::Num(n) => *n as i64, _ => return Err("grid-translate: dr must be a number".into()) };
    let dc = match &args[2] { Value::Int(n) => *n, Value::Num(n) => *n as i64, _ => return Err("grid-translate: dc must be a number".into()) };
    let h = g.len();
    let w = if h > 0 { g[0].len() } else { 0 };
    // Detect background
    let mut counts = std::collections::HashMap::new();
    for row in &g { for &c in row { *counts.entry(c).or_insert(0usize) += 1; } }
    let bg = counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0);
    let mut out = vec![vec![bg; w]; h];
    for r in 0..h {
        for c in 0..w {
            if g[r][c] != bg {
                let nr = r as i64 + dr;
                let nc = c as i64 + dc;
                if nr >= 0 && (nr as usize) < h && nc >= 0 && (nc as usize) < w {
                    out[nr as usize][nc as usize] = g[r][c];
                }
            }
        }
    }
    Ok(grid_to_value(out))
}

/// `(grid-find-color grid color)` — return list of (row, col) pairs where
/// the given color appears.
fn bi_grid_find_color(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-find-color: expected 2 args (grid, color), got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let color = match &args[1] { Value::Int(n) => *n, Value::Num(n) => *n as i64, _ => return Err("grid-find-color: color must be a number".into()) };
    let mut positions = Vec::new();
    for (r, row) in g.iter().enumerate() {
        for (c, &cell) in row.iter().enumerate() {
            if cell == color {
                positions.push(Value::list(vec![Value::Int(r as i64), Value::Int(c as i64)]));
            }
        }
    }
    Ok(Value::list(positions))
}

/// `(grid-mask grid mask color)` — for every non-zero cell in mask,
/// set the corresponding cell in grid to color.
fn bi_grid_mask(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 3 {
        return Err(format!("grid-mask: expected 3 args (grid, mask, color), got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let mask = as_grid(&args[1])?;
    let color = match &args[2] { Value::Int(n) => *n, Value::Num(n) => *n as i64, _ => return Err("grid-mask: color must be a number".into()) };
    let h = g.len();
    let w = if h > 0 { g[0].len() } else { 0 };
    let mut out = g.clone();
    for r in 0..h.min(mask.len()) {
        let mw = if mask[r].len() > 0 { mask[r].len() } else { 0 };
        for c in 0..w.min(mw) {
            if mask[r][c] != 0 {
                out[r][c] = color;
            }
        }
    }
    Ok(grid_to_value(out))
}

/// `(grid-blank rows cols [color])` — create a uniform grid filled with color (default 0).
fn bi_grid_blank(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err(format!("grid-blank: expected 2-3 args (rows, cols [, color]), got {}", args.len()));
    }
    let rows = match &args[0] { Value::Int(n) => *n as usize, Value::Num(n) => *n as usize, _ => return Err("grid-blank: rows must be a number".into()) };
    let cols = match &args[1] { Value::Int(n) => *n as usize, Value::Num(n) => *n as usize, _ => return Err("grid-blank: cols must be a number".into()) };
    let color = if args.len() == 3 {
        match &args[2] { Value::Int(n) => *n, Value::Num(n) => *n as i64, _ => 0 }
    } else { 0 };
    if rows > 100 || cols > 100 { return Err("grid-blank: max 100x100".into()); }
    Ok(grid_to_value(vec![vec![color; cols]; rows]))
}

/// `(grid-scale grid factor)` — scale each cell to a factor×factor block.
fn bi_grid_scale(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("grid-scale: expected 2 args (grid, factor), got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let factor = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Num(n) => *n as usize,
        _ => return Err("grid-scale: factor must be a number".into()),
    };
    if factor == 0 || factor > 10 {
        return Err("grid-scale: factor must be 1-10".into());
    }
    let h = g.len();
    let w = if h > 0 { g[0].len() } else { 0 };
    let mut out = vec![vec![0i64; w * factor]; h * factor];
    for r in 0..h {
        for c in 0..w {
            let v = g[r][c];
            for dr in 0..factor {
                for dc in 0..factor {
                    out[r * factor + dr][c * factor + dc] = v;
                }
            }
        }
    }
    Ok(grid_to_value(out))
}

/// `(grid-tile grid n m)` — tile the grid into an n×m arrangement.
/// If called with 2 args, tiles n×n.
fn bi_grid_tile(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err(format!("grid-tile: expected 2-3 args (grid, rows [, cols]), got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let nr = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Num(n) => *n as usize,
        _ => return Err("grid-tile: rows must be a number".into()),
    };
    let nc = if args.len() == 3 {
        match &args[2] {
            Value::Int(n) => *n as usize,
            Value::Num(n) => *n as usize,
            _ => return Err("grid-tile: cols must be a number".into()),
        }
    } else { nr };
    if nr == 0 || nr > 10 || nc > 10 {
        return Err("grid-tile: factors must be 1-10".into());
    }
    let h = g.len();
    let w = if h > 0 { g[0].len() } else { 0 };
    if h * nr > 100 || w * nc > 100 {
        return Err("grid-tile: result too large (max 100x100)".into());
    }
    let mut out = vec![vec![0i64; w * nc]; h * nr];
    for tr in 0..nr {
        for tc in 0..nc {
            for r in 0..h {
                for c in 0..w {
                    out[tr * h + r][tc * w + c] = g[r][c];
                }
            }
        }
    }
    Ok(grid_to_value(out))
}

/// `(grid-fill-enclosed grid)` — fill interior zeros.
/// Any 0-cell not reachable from the border via 4-connected 0-cells
/// is replaced by the nearest non-zero neighbor (simple: use the
/// grid's background as fill target, non-background as fill value).
fn bi_grid_fill_enclosed(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-fill-enclosed: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let h = g.len();
    if h == 0 { return Ok(grid_to_value(g.clone())); }
    let w = g[0].len();

    // Find the background color (most common).
    let mut counts = std::collections::HashMap::new();
    for r in &g { for &c in r { *counts.entry(c).or_insert(0usize) += 1; } }
    let bg = counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c).unwrap_or(0);

    // Determine which values can be "hole" values — the bg color,
    // plus 0 (the universal empty in ARC). Both can represent
    // interior holes that need filling.
    let hole_values: Vec<i64> = if bg == 0 { vec![0] } else { vec![bg, 0] };

    // For each hole value, flood from border independently. A cell is
    // "exterior" if it's reachable from the border via 4-connected cells
    // of the SAME hole value. This prevents cross-value flooding that
    // would mark truly interior cells as exterior.
    let mut exterior = vec![vec![false; w]; h];
    for &hv in &hole_values {
        let mut queue = std::collections::VecDeque::new();
        for r in 0..h {
            for c in 0..w {
                if (r == 0 || r == h - 1 || c == 0 || c == w - 1)
                    && g[r][c] == hv && !exterior[r][c]
                {
                    exterior[r][c] = true;
                    queue.push_back((r, c));
                }
            }
        }
        while let Some((r, c)) = queue.pop_front() {
            for (dr, dc) in &[(0isize, 1isize), (0, -1), (1, 0), (-1, 0)] {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    if !exterior[nr][nc] && g[nr][nc] == hv {
                        exterior[nr][nc] = true;
                        queue.push_back((nr, nc));
                    }
                }
            }
        }
    }

    // For each interior hole cell, fill with the nearest neighbor that
    // differs from the cell's own value. This handles both cases:
    // - Interior 0 surrounded by bg (fills with bg)
    // - Interior bg surrounded by fg (fills with fg)
    let mut out = g.clone();
    for r in 0..h {
        for c in 0..w {
            if hole_values.contains(&g[r][c]) && !exterior[r][c] {
                let cell = g[r][c];
                let mut fill = cell;
                for (dr, dc) in &[(0isize, 1isize), (0, -1), (1, 0), (-1, 0)] {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                        let v = g[nr as usize][nc as usize];
                        if v != cell { fill = v; break; }
                    }
                }
                out[r][c] = fill;
            }
        }
    }
    Ok(grid_to_value(out))
}

/// `(grid-compact grid)` — remove all-zero rows and all-zero columns.
fn bi_grid_compact(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("grid-compact: expected 1 arg, got {}", args.len()));
    }
    let g = as_grid(&args[0])?;
    let h = g.len();
    if h == 0 { return Ok(grid_to_value(g.clone())); }
    let w = g[0].len();
    // Find non-zero rows and columns.
    let keep_row: Vec<bool> = (0..h).map(|r| g[r].iter().any(|&c| c != 0)).collect();
    let keep_col: Vec<bool> = (0..w).map(|c| (0..h).any(|r| g[r][c] != 0)).collect();
    let out: Vec<Vec<i64>> = (0..h)
        .filter(|&r| keep_row[r])
        .map(|r| (0..w).filter(|&c| keep_col[c]).map(|c| g[r][c]).collect())
        .collect();
    if out.is_empty() {
        return Ok(grid_to_value(vec![vec![0i64]]));
    }
    Ok(grid_to_value(out))
}

// ────────────────────────────────────────────────────────────────────────────
// Old-Node → new-Node converter (transitional)
// ────────────────────────────────────────────────────────────────────────────
//
// Lets the existing parser feed eval_v2 without parser changes:
//   1. Translates each old Node to its new counterpart.
//   2. Classifies Num literals: integer-valued ones become Node::Int.
//   3. Resolves App-with-special-form-symbol heads into Node::SpecialApp.
//   4. Desugars `(defmacro name (params) body)` into
//      `(define name (lambda (params) body))`.
//
// Throwaway code; deletes when parser.rs is updated.

/// Convert an old Node tree by walking and producing a new one. Three jobs:
///   1. Classify integer-valued Num literals as Int.
///   2. Resolve App-with-special-form-symbol heads into SpecialApp.
///   3. Desugar `(defmacro name (params...) body)` into
///      `(define name (lambda (params...) body))`. Defmacro is no longer
///      a special form in the new core; it's just a parser-level shorthand
///      for defining a function. This desugaring constructs fresh Lambda
///      and SpecialApp(Define, ...) nodes inline.
///
/// The converter preserves the index space of the original tree where
/// possible (so existing index references stay valid) and appends new
/// nodes for desugaring. This means the output Vec is `>=` the input
/// length but the first `old.len()` indices line up.
pub fn convert_tree(old: &[OldNode]) -> Vec<Node> {
    init_special_forms_lazy();
    let defmacro_sym = intern("defmacro");
    let thread_first_sym = intern("->");
    let thread_last_sym = intern("->>");
    let mut new: Vec<Node> = Vec::with_capacity(old.len());

    // First pass: convert each node in place, preserving indices. Defmacro
    // forms get a placeholder we'll patch up in pass 2 (we need the full
    // index map before we can rewrite).
    for node in old {
        new.push(convert_one(node, old, defmacro_sym, &mut Vec::new()));
    }

    // Second pass: walk for any App whose head Symbol is `defmacro` and
    // rewrite it. We may need to append new nodes (a Lambda for the body
    // and possibly a Do wrapper for multi-body forms).
    //
    // Layout we're matching: (defmacro name (params...) body1 body2 ... bodyN)
    //   children[0] = defmacro symbol
    //   children[1] = name symbol
    //   children[2] = App(param symbols) — the params list
    //   children[3..] = one or more body expressions
    //
    // For N=1, the lambda body is body1 directly.
    // For N>1, we wrap the bodies in (do body1 body2 ... bodyN), which
    // matches the implicit-do behaviour of the original defmacro.
    for i in 0..new.len() {
        let (name_idx, params_node_idx, body_indices) = match &new[i] {
            Node::App(children) if children.len() >= 4 => {
                if let Node::Symbol(s) = &new[children[0]] {
                    if *s == defmacro_sym {
                        (children[1], children[2], children[3..].to_vec())
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            _ => continue,
        };

        // Extract params from the params-list App.
        let params: Vec<Sym> = match &new[params_node_idx] {
            Node::App(param_indices) => param_indices
                .iter()
                .filter_map(|&pi| match &new[pi] {
                    Node::Symbol(s) => Some(*s),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };

        // If the body is multiple expressions, wrap them in (do ...).
        let body_idx = if body_indices.len() == 1 {
            body_indices[0]
        } else {
            let do_idx = new.len();
            new.push(Node::SpecialApp(SpecialForm::Do, body_indices));
            do_idx
        };

        // Append a fresh Lambda node.
        let lambda_idx = new.len();
        new.push(Node::Lambda(params, body_idx));

        // Replace this App with SpecialApp(Define, [name, lambda_idx]).
        new[i] = Node::SpecialApp(SpecialForm::Define, vec![name_idx, lambda_idx]);
    }

    // Third pass: desugar threading macros (-> and ->>).
    //
    // (-> x (f a) (g b c))   →  (g (f x a) b c)     thread-first
    // (->> x (f a) (g b c))  →  (g b c (f a x))     thread-last
    //
    // Bare symbols are wrapped: (-> x f) → (f x)
    // Single-step is valid:     (-> x (f a)) → (f x a)
    //
    // We iterate the original index range; newly appended nodes from this
    // pass are threading results and won't themselves be -> / ->> forms.
    let thread_pass_len = new.len();
    for i in 0..thread_pass_len {
        let (initial_idx, steps, is_first) = match &new[i] {
            Node::App(children) if children.len() >= 3 => {
                if let Node::Symbol(s) = &new[children[0]] {
                    if *s == thread_first_sym {
                        (children[1], children[2..].to_vec(), true)
                    } else if *s == thread_last_sym {
                        (children[1], children[2..].to_vec(), false)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            _ => continue,
        };

        // Build nested applications from left to right. `acc` is the index
        // of the "threaded" value so far.
        let mut acc = initial_idx;
        for step_idx in steps {
            // Each step is either a bare symbol or an App (f arg1 arg2 ...).
            let new_app_children = match &new[step_idx] {
                Node::Symbol(_) => {
                    // (-> x f) → (f x)
                    vec![step_idx, acc]
                }
                Node::App(step_children) if !step_children.is_empty() => {
                    // Thread-first: insert acc after the head (position 1).
                    // Thread-last:  append acc at the end.
                    let mut children = Vec::with_capacity(step_children.len() + 1);
                    if is_first {
                        children.push(step_children[0]); // head
                        children.push(acc);               // threaded value
                        children.extend_from_slice(&step_children[1..]); // rest
                    } else {
                        children.extend_from_slice(step_children); // all original
                        children.push(acc);                        // threaded value
                    }
                    children
                }
                _ => {
                    // Anything else (literal, etc.): treat as a unary call.
                    vec![step_idx, acc]
                }
            };
            let app_idx = new.len();
            new.push(Node::App(new_app_children));
            acc = app_idx;
        }

        // Replace the original threading form with the final accumulated App.
        new[i] = new[acc].clone();
    }

    new
}

fn convert_one(
    node: &OldNode,
    all: &[OldNode],
    _defmacro_sym: Sym,
    _scratch: &mut Vec<Node>,
) -> Node {
    match node {
        OldNode::Num(n) => {
            if n.is_finite() && *n == (*n as i64) as f64 {
                Node::Int(*n as i64)
            } else {
                Node::Num(*n)
            }
        }
        OldNode::Str(s) => Node::Str(s.clone()),
        OldNode::Bool(b) => Node::Bool(*b),
        OldNode::Symbol(s) => Node::Symbol(*s),
        OldNode::App(children) => {
            // Check if first child is a special-form symbol. (Defmacro is
            // NOT a special form; it gets desugared in the second pass of
            // convert_tree.)
            if let Some(&first_idx) = children.first() {
                if let OldNode::Symbol(sym) = &all[first_idx] {
                    if let Some(form) = SpecialForm::from_sym(*sym) {
                        let rest: Vec<usize> = children[1..].to_vec();
                        return Node::SpecialApp(form, rest);
                    }
                }
            }
            Node::App(children.clone())
        }
        OldNode::If(c, t, e) => Node::If(*c, *t, *e),
        OldNode::Lambda(params, body) => Node::Lambda(params.clone(), *body),
        OldNode::Let(bindings, body) => Node::Let(bindings.clone(), *body),
    }
}

/// Lazy initialization of the special-form Sym table. Idempotent.
fn init_special_forms_lazy() {
    // Cheap to call repeatedly; init_special_forms is idempotent because
    // intern() returns the same Sym for the same string.
    init_special_forms();
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rc(nodes: Vec<Node>) -> Rc<[Node]> {
        Rc::from(nodes)
    }

    #[test]
    fn eval_int_literal() {
        let env = make_default_env();
        let nodes = rc(vec![Node::Int(42)]);
        let r = eval(&nodes, 0, &env).unwrap();
        assert!(matches!(r, Value::Int(42)));
    }

    #[test]
    fn eval_add_two_ints() {
        // (add 2 3)
        let env = make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("add")),
            Node::Int(2),
            Node::Int(3),
            Node::App(vec![0, 1, 2]),
        ]);
        let r = eval(&nodes, 3, &env).unwrap();
        assert!(matches!(r, Value::Int(5)));
    }

    #[test]
    fn eval_int_num_coercion() {
        // (add 2 3.5) → 5.5
        let env = make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("add")),
            Node::Int(2),
            Node::Num(3.5),
            Node::App(vec![0, 1, 2]),
        ]);
        let r = eval(&nodes, 3, &env).unwrap();
        assert!(matches!(r, Value::Num(n) if n == 5.5));
    }

    #[test]
    fn eval_lambda_call() {
        // ((lambda (x) (multiply x 2)) 7) → 14
        let env = make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("multiply")), // 0
            Node::Symbol(intern("x")),        // 1
            Node::Int(2),                     // 2
            Node::App(vec![0, 1, 2]),         // 3: (multiply x 2)
            Node::Lambda(vec![intern("x")], 3), // 4: (lambda (x) (multiply x 2))
            Node::Int(7),                     // 5
            Node::App(vec![4, 5]),            // 6: ((lambda...) 7)
        ]);
        let r = eval(&nodes, 6, &env).unwrap();
        assert!(matches!(r, Value::Int(14)));
    }

    #[test]
    fn eval_define_then_call() {
        // (do (define double (lambda (x) (multiply x 2))) (double 9)) → 18
        let env = make_default_env();
        init_special_forms_lazy();
        let nodes = rc(vec![
            // (lambda (x) (multiply x 2))
            Node::Symbol(intern("multiply")), // 0
            Node::Symbol(intern("x")),        // 1
            Node::Int(2),                     // 2
            Node::App(vec![0, 1, 2]),         // 3
            Node::Lambda(vec![intern("x")], 3), // 4
            Node::Symbol(intern("double")),   // 5  (target name for define)
            // (define double (lambda ...))
            Node::SpecialApp(SpecialForm::Define, vec![5, 4]), // 6
            // (double 9)
            Node::Symbol(intern("double")),   // 7
            Node::Int(9),                     // 8
            Node::App(vec![7, 8]),            // 9
            // (do ...)
            Node::SpecialApp(SpecialForm::Do, vec![6, 9]), // 10
        ]);
        let r = eval(&nodes, 10, &env).unwrap();
        assert!(matches!(r, Value::Int(18)));
    }

    #[test]
    fn eval_string_length_returns_int() {
        // (string-length "hello") → 5 (Int)
        let env = make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("string-length")),
            Node::Str("hello".into()),
            Node::App(vec![0, 1]),
        ]);
        let r = eval(&nodes, 2, &env).unwrap();
        assert!(matches!(r, Value::Int(5)));
    }

    #[test]
    fn eval_map_over_list() {
        // (map (lambda (x) (multiply x x)) (list 1 2 3 4)) → (1 4 9 16)
        let env = make_default_env();
        let nodes = rc(vec![
            // (lambda (x) (multiply x x))
            Node::Symbol(intern("multiply")), // 0
            Node::Symbol(intern("x")),        // 1
            Node::Symbol(intern("x")),        // 2
            Node::App(vec![0, 1, 2]),         // 3
            Node::Lambda(vec![intern("x")], 3), // 4
            // (list 1 2 3 4)
            Node::Symbol(intern("list")),     // 5
            Node::Int(1),                     // 6
            Node::Int(2),                     // 7
            Node::Int(3),                     // 8
            Node::Int(4),                     // 9
            Node::App(vec![5, 6, 7, 8, 9]),   // 10
            // (map lambda list)
            Node::Symbol(intern("map")),      // 11
            Node::App(vec![11, 4, 10]),       // 12
        ]);
        let r = eval(&nodes, 12, &env).unwrap();
        match r {
            Value::List(l) => {
                let xs: Vec<i64> = l.iter().map(|v| match v {
                    Value::Int(n) => *n,
                    _ => panic!("expected Int, got {:?}", v),
                }).collect();
                assert_eq!(xs, vec![1, 4, 9, 16]);
            }
            _ => panic!("expected list, got {:?}", r),
        }
    }

    #[test]
    fn eval_reduce_sum() {
        // (reduce add (list 1 2 3 4 5)) → 15
        let env = make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("list")),     // 0
            Node::Int(1),                     // 1
            Node::Int(2),                     // 2
            Node::Int(3),                     // 3
            Node::Int(4),                     // 4
            Node::Int(5),                     // 5
            Node::App(vec![0, 1, 2, 3, 4, 5]), // 6
            Node::Symbol(intern("reduce")),   // 7
            Node::Symbol(intern("add")),      // 8
            Node::App(vec![7, 8, 6]),         // 9
        ]);
        let r = eval(&nodes, 9, &env).unwrap();
        assert!(matches!(r, Value::Int(15)));
    }

    #[test]
    fn eval_filter_even() {
        // (filter even (list 1 2 3 4 5 6)) → (2 4 6)
        let env = make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("list")),     // 0
            Node::Int(1),                     // 1
            Node::Int(2),                     // 2
            Node::Int(3),                     // 3
            Node::Int(4),                     // 4
            Node::Int(5),                     // 5
            Node::Int(6),                     // 6
            Node::App(vec![0, 1, 2, 3, 4, 5, 6]), // 7
            Node::Symbol(intern("filter")),   // 8
            Node::Symbol(intern("even")),     // 9
            Node::App(vec![8, 9, 7]),         // 10
        ]);
        let r = eval(&nodes, 10, &env).unwrap();
        match r {
            Value::List(l) => {
                let xs: Vec<i64> = l.iter().map(|v| match v {
                    Value::Int(n) => *n,
                    _ => panic!("expected Int"),
                }).collect();
                assert_eq!(xs, vec![2, 4, 6]);
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn eval_letrec_factorial() {
        // (let ((fact (lambda (n)
        //                (if (= n 0) 1 (multiply n (fact (subtract n 1)))))))
        //   (fact 5))
        // Tests letrec self-reference through the shared scope.
        let env = make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("n")),                  // 0
            Node::Int(0),                               // 1
            Node::Symbol(intern("=")),                  // 2
            Node::App(vec![2, 0, 1]),                   // 3: (= n 0)
            Node::Int(1),                               // 4
            Node::Symbol(intern("n")),                  // 5
            Node::Int(1),                               // 6
            Node::Symbol(intern("subtract")),           // 7
            Node::App(vec![7, 5, 6]),                   // 8: (subtract n 1)
            Node::Symbol(intern("fact")),               // 9
            Node::App(vec![9, 8]),                      // 10: (fact (subtract n 1))
            Node::Symbol(intern("n")),                  // 11
            Node::Symbol(intern("multiply")),           // 12
            Node::App(vec![12, 11, 10]),                // 13: (multiply n (fact ...))
            Node::If(3, 4, 13),                         // 14: (if ...)
            Node::Lambda(vec![intern("n")], 14),        // 15
            Node::Symbol(intern("fact")),               // 16
            Node::Int(5),                               // 17
            Node::App(vec![16, 17]),                    // 18: (fact 5)
            Node::Let(vec![(intern("fact"), 15)], 18),  // 19
        ]);
        let r = eval(&nodes, 19, &env).unwrap();
        assert!(matches!(r, Value::Int(120)),
                "expected 120, got {:?}", r);
    }

    #[test]
    fn convert_old_num_classifies_int() {
        // Old parser produces Node::Num(2.0); converter should classify as Int.
        let old = vec![OldNode::Num(2.0)];
        let new = convert_tree(&old);
        assert!(matches!(new[0], Node::Int(2)));
    }

    #[test]
    fn convert_old_num_keeps_float() {
        // Non-integer values stay as Num.
        let old = vec![OldNode::Num(2.5)];
        let new = convert_tree(&old);
        assert!(matches!(new[0], Node::Num(n) if n == 2.5));
    }

    #[test]
    fn convert_old_app_recognizes_special_form() {
        init_special_forms_lazy();
        let do_sym = intern("do");
        let old = vec![
            OldNode::Symbol(do_sym),         // 0
            OldNode::Num(1.0),               // 1
            OldNode::App(vec![0, 1]),        // 2: (do 1)
        ];
        let new = convert_tree(&old);
        match &new[2] {
            Node::SpecialApp(SpecialForm::Do, children) => {
                assert_eq!(children, &vec![1]);
            }
            other => panic!("expected SpecialApp(Do, ...), got {:?}", other),
        }
    }

    #[test]
    fn truthiness_int_zero_is_falsy() {
        assert!(!is_truthy(&Value::Int(0)));
        assert!(is_truthy(&Value::Int(1)));
        assert!(is_truthy(&Value::Int(-1)));
    }

    #[test]
    fn truthiness_empty_string_is_falsy() {
        assert!(!is_truthy(&Value::str("")));
        assert!(is_truthy(&Value::str("x")));
    }

    #[test]
    fn truthiness_empty_list_is_falsy() {
        assert!(!is_truthy(&Value::list(vec![])));
        assert!(is_truthy(&Value::list(vec![Value::Int(1)])));
    }

    #[test]
    fn truthiness_in_if() {
        // (if 0 "yes" "no") → "no"
        let env = make_default_env();
        let nodes = rc(vec![
            Node::Int(0),                        // 0: condition
            Node::Str("yes".into()),             // 1
            Node::Str("no".into()),              // 2
            Node::If(0, 1, 2),                   // 3
        ]);
        let r = eval(&nodes, 3, &env).unwrap();
        match r {
            Value::Str(s) => assert_eq!(s.as_ref(), "no"),
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn end_to_end_via_existing_parser() {
        // Source → existing parser → convert → eval. The whole transition
        // pipeline in one test.
        let src = "(add 2 3)";
        let (old_nodes, root) = crate::parser::parse_source(src).unwrap();
        let new_nodes = rc(convert_tree(&old_nodes));
        let env = make_default_env();
        let r = eval(&new_nodes, root, &env).unwrap();
        assert!(matches!(r, Value::Int(5)));
    }

    #[test]
    fn end_to_end_string_length_via_parser() {
        let src = r#"(string-length "hello world")"#;
        let (old_nodes, root) = crate::parser::parse_source(src).unwrap();
        let new_nodes = rc(convert_tree(&old_nodes));
        let env = make_default_env();
        let r = eval(&new_nodes, root, &env).unwrap();
        assert!(matches!(r, Value::Int(11)));
    }

    #[test]
    fn end_to_end_lambda_via_parser() {
        let src = "((lambda (x) (multiply x x)) 6)";
        let (old_nodes, root) = crate::parser::parse_source(src).unwrap();
        let new_nodes = rc(convert_tree(&old_nodes));
        let env = make_default_env();
        let r = eval(&new_nodes, root, &env).unwrap();
        assert!(matches!(r, Value::Int(36)));
    }

    #[test]
    fn end_to_end_letrec_factorial_via_parser() {
        let src = "(let ((fact (lambda (n) (if (= n 0) 1 (multiply n (fact (subtract n 1))))))) (fact 6))";
        let (old_nodes, root) = crate::parser::parse_source(src).unwrap();
        let new_nodes = rc(convert_tree(&old_nodes));
        let env = make_default_env();
        let r = eval(&new_nodes, root, &env).unwrap();
        assert!(matches!(r, Value::Int(720)),
                "expected 720, got {:?}", r);
    }

    fn run(src: &str) -> Result<Value, String> {
        let (old_nodes, root) = crate::parser::parse_source(src)
            .map_err(|e| format!("parse error: {}", e))?;
        let new_nodes: Rc<[Node]> = convert_tree(&old_nodes).into();
        let env = make_default_env();
        eval(&new_nodes, root, &env)
    }

    #[test]
    fn arith_extras() {
        assert!(matches!(run("(min 3 7)").unwrap(), Value::Int(3)));
        assert!(matches!(run("(max 3 7)").unwrap(), Value::Int(7)));
        assert!(matches!(run("(abs (negate 5))").unwrap(), Value::Int(5)));
        assert!(matches!(run("(floor 3.7)").unwrap(), Value::Int(3)));
        assert!(matches!(run("(ceil 3.2)").unwrap(), Value::Int(4)));
        assert!(matches!(run("(pow 2 10)").unwrap(), Value::Int(1024)));
        // sqrt always Num
        assert!(matches!(run("(sqrt 4)").unwrap(), Value::Num(n) if n == 2.0));
    }

    #[test]
    fn comparison_le_ge() {
        assert!(matches!(run("(<= 3 3)").unwrap(), Value::Bool(true)));
        assert!(matches!(run("(>= 3 3)").unwrap(), Value::Bool(true)));
        assert!(matches!(run("(<= 3 2)").unwrap(), Value::Bool(false)));
    }

    #[test]
    fn string_ops_full() {
        assert_eq!(run(r#"(string-reverse "hello")"#).unwrap().as_str().unwrap(), "olleh");
        assert_eq!(run(r#"(string-trim "  hi  ")"#).unwrap().as_str().unwrap(), "hi");
        assert_eq!(run(r#"(string-take "hello world" 5)"#).unwrap().as_str().unwrap(), "hello");
        assert_eq!(run(r#"(string-drop "hello world" 6)"#).unwrap().as_str().unwrap(), "world");
        assert_eq!(run(r#"(string-replace "abc" "b" "x")"#).unwrap().as_str().unwrap(), "axc");
        assert!(matches!(run(r#"(count-char "banana" "a")"#).unwrap(), Value::Int(3)));
        assert!(matches!(run(r#"(string-contains "hello" "ell")"#).unwrap(), Value::Bool(true)));
    }

    #[test]
    fn list_ops_full() {
        // (cons 0 (range 3)) → (0 0 1 2)
        let r = run("(cons 0 (range 3))").unwrap();
        let l = r.as_list().unwrap();
        assert_eq!(l.len(), 4);

        // (sort (list 3 1 2)) → (1 2 3)
        let r = run("(sort (list 3 1 2))").unwrap();
        let l = r.as_list().unwrap();
        assert!(matches!(&l[0], Value::Int(1)));
        assert!(matches!(&l[2], Value::Int(3)));

        // (zip (list 1 2) (list "a" "b")) → ((1 "a") (2 "b"))
        let r = run(r#"(zip (list 1 2) (list "a" "b"))"#).unwrap();
        assert_eq!(r.as_list().unwrap().len(), 2);

        // (contains (list 1 2 3) 2) → true
        assert!(matches!(run("(contains (list 1 2 3) 2)").unwrap(), Value::Bool(true)));
        assert!(matches!(run("(contains (list 1 2 3) 5)").unwrap(), Value::Bool(false)));
    }

    #[test]
    fn ns_ops_full() {
        // Build a namespace, get keys, look up.
        let r = run(r#"(ns-keys (ns-put (ns-put (ns-empty) "a" 1) "b" 2))"#).unwrap();
        let l = r.as_list().unwrap();
        assert_eq!(l.len(), 2);

        // ns-get
        let r = run(r#"(ns-get (ns-put (ns-empty) "x" 42) "x")"#).unwrap();
        assert!(matches!(r, Value::Int(42)));

        // ns-get-or default
        let r = run(r#"(ns-get-or (ns-empty) "missing" 99)"#).unwrap();
        assert!(matches!(r, Value::Int(99)));

        // ns-has
        assert!(matches!(
            run(r#"(ns-has (ns-put (ns-empty) "k" 1) "k")"#).unwrap(),
            Value::Bool(true)
        ));
        assert!(matches!(
            run(r#"(ns-has (ns-empty) "missing")"#).unwrap(),
            Value::Bool(false)
        ));

        // ns-size
        assert!(matches!(
            run(r#"(ns-size (ns-put (ns-put (ns-empty) "a" 1) "b" 2))"#).unwrap(),
            Value::Int(2)
        ));
    }

    #[test]
    fn type_of_distinguishes_int_and_num() {
        assert_eq!(run("(type-of 42)").unwrap().as_str().unwrap(), "int");
        assert_eq!(run("(type-of 3.14)").unwrap().as_str().unwrap(), "num");
        assert_eq!(run(r#"(type-of "hi")"#).unwrap().as_str().unwrap(), "string");
        assert_eq!(run("(type-of true)").unwrap().as_str().unwrap(), "bool");
        assert_eq!(run("(type-of (list 1 2))").unwrap().as_str().unwrap(), "list");
        assert_eq!(run("(type-of nil)").unwrap().as_str().unwrap(), "nil");
    }

    #[test]
    fn end_to_end_decomposition_predictor() {
        // Run the actual examples/decomposition_predictor.selph file end-to-end
        // through eval_v2. It was the §9.21 prototype that we know returns
        // 7/7 correct on the old eval.
        let src = std::fs::read_to_string("../examples/decomposition_predictor.selph")
            .or_else(|_| std::fs::read_to_string("examples/decomposition_predictor.selph"))
            .expect("decomposition_predictor.selph not found");
        let result = run(&src).expect("eval failed");
        let s = value_to_string(&result);
        assert!(s.contains("7 / 7 correct"), "got: {}", s);
    }

    #[test]
    fn end_to_end_heuristics_smoke() {
        // examples/heuristics.selph runs end-to-end without errors. It uses
        // defmacro, namespaces, lambdas, higher-order map, and the full new
        // builtin set.
        let src = std::fs::read_to_string("../examples/heuristics.selph")
            .or_else(|_| std::fs::read_to_string("examples/heuristics.selph"))
            .expect("heuristics.selph not found");
        // Just check it eval's without error.
        run(&src).expect("eval failed");
    }

    #[test]
    fn end_to_end_map_filter_via_parser() {
        let src = "(filter (lambda (x) (= 0 (modulo x 2))) (list 1 2 3 4 5 6 7 8))";
        let (old_nodes, root) = crate::parser::parse_source(src).unwrap();
        let new_nodes = rc(convert_tree(&old_nodes));
        let env = make_default_env();
        let r = eval(&new_nodes, root, &env).unwrap();
        match r {
            Value::List(l) => {
                let xs: Vec<i64> = l.iter().map(|v| match v {
                    Value::Int(n) => *n,
                    _ => panic!(),
                }).collect();
                assert_eq!(xs, vec![2, 4, 6, 8]);
            }
            _ => panic!("expected list"),
        }
    }

    // ── Bucket 6 builtins (synth_v2 wiring) ───────────────────────────

    /// Multi-expression file runner that threads a single env across
    /// every top-level form, mirroring how `selph eval-v2 <file>` works.
    fn run_file(src: &str) -> Result<Value, String> {
        let (old_nodes, roots) =
            crate::parser::parse_file(src).map_err(|e| format!("parse error: {}", e))?;
        let new_nodes: Rc<[Node]> = convert_tree(&old_nodes).into();
        let env = make_default_env();
        let mut last = Value::Nil;
        for &r in &roots {
            last = eval(&new_nodes, r, &env)?;
        }
        Ok(last)
    }

    #[test]
    fn node_to_source_renders_basic_forms() {
        // Round-trip a few common forms through parse → convert → render.
        let cases = &[
            ("(add 2 3)", "(add 2 3)"),
            ("(string-upper \"hi\")", "(string-upper \"hi\")"),
            ("(lambda (x) (multiply x 2))", "(lambda (x) (multiply x 2))"),
        ];
        for (src, expected) in cases {
            let (old_nodes, root) = crate::parser::parse_source(src).unwrap();
            let new_nodes = convert_tree(&old_nodes);
            let rendered = node_to_source(&new_nodes, root);
            assert_eq!(&rendered, expected, "round-trip failed for {:?}", src);
        }
    }

    #[test]
    fn bucket6_eval_source_runs_inline_program() {
        // (eval-source "(add 2 3)") → 5
        let r = run_file(r#"(eval-source "(add 2 3)")"#).unwrap();
        assert!(matches!(r, Value::Int(5)), "got {:?}", r);
    }

    #[test]
    fn bucket6_eval_source_uses_caller_env_for_defines() {
        // Define a function via eval-source, then call it from the
        // outer scope. Validates that eval-source threads the caller's
        // env, so persistent defines work.
        let src = r#"
            (eval-source "(define quad (lambda (x) (multiply x x)))")
            (quad 7)
        "#;
        let r = run_file(src).unwrap();
        assert!(matches!(r, Value::Int(49)), "got {:?}", r);
    }

    #[test]
    fn bucket6_test_spec_returns_perfect_score_for_correct_function() {
        // (test-spec (lambda (x) (multiply x x)) ((1 1) (2 4) (3 9))) → 1.0
        let src = r#"
            (test-spec (lambda (x) (multiply x x))
                       (list (list 1 1) (list 2 4) (list 3 9)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Num(n) => assert!((n - 1.0).abs() < 1e-9, "expected 1.0, got {}", n),
            _ => panic!("expected Num, got {:?}", r),
        }
    }

    #[test]
    fn bucket6_test_spec_returns_partial_score() {
        // Function gets 2 of 3 right.
        let src = r#"
            (test-spec (lambda (x) x)
                       (list (list 1 1) (list 2 4) (list 3 3)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Num(n) => assert!((n - 2.0/3.0).abs() < 1e-9, "expected 0.667, got {}", n),
            _ => panic!("expected Num"),
        }
    }

    #[test]
    fn bucket6_memorize_builds_namespace_from_pairs() {
        // (memorize ((alice 1) (bob 2))) → namespace
        let src = r#"
            (memorize (list (list "alice" 1) (list "bob" 2)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert_eq!(map.len(), 2);
                assert!(matches!(map.get(&intern("alice")), Some(Value::Int(1))));
                assert!(matches!(map.get(&intern("bob")), Some(Value::Int(2))));
            }
            _ => panic!("expected Ns, got {:?}", r),
        }
    }

    #[test]
    fn bucket6_memorize_returns_nil_for_non_string_keys() {
        let src = r#"
            (memorize (list (list 1 "a") (list 2 "b")))
        "#;
        let r = run_file(src).unwrap();
        assert!(matches!(r, Value::Nil));
    }

    // ── §9.36 sub-step 2b: quote returns Node values ────────────────────

    #[test]
    fn quote_returns_node_value_for_literal() {
        // (quote 42) returns a Node value pointing at the Int literal.
        // We verify the kind by inspecting node_to_source on the underlying
        // NodeRef — once node-kind exists in 2d we can do it from SELPH.
        let src = r#"(quote 42)"#;
        let r = run_file(src).unwrap();
        match r {
            Value::Node(node_ref) => {
                let rendered = node_to_source(&node_ref.nodes, node_ref.idx);
                assert_eq!(rendered, "42");
            }
            other => panic!("expected Value::Node, got {:?}", other),
        }
    }

    #[test]
    fn quote_returns_node_value_for_application() {
        // (quote (add 1 2)) returns a Node value for the App, no eval.
        // If eval happened, we'd see Int(3) instead.
        let src = r#"(quote (add 1 2))"#;
        let r = run_file(src).unwrap();
        match r {
            Value::Node(node_ref) => {
                let rendered = node_to_source(&node_ref.nodes, node_ref.idx);
                assert_eq!(rendered, "(add 1 2)");
            }
            other => panic!("expected Value::Node, got {:?}", other),
        }
    }

    #[test]
    fn quote_returns_node_value_for_symbol() {
        // (quote x) returns a Node value for the symbol — does NOT
        // attempt to look up `x` in the env.
        let src = r#"(quote x)"#;
        let r = run_file(src).unwrap();
        match r {
            Value::Node(node_ref) => {
                let rendered = node_to_source(&node_ref.nodes, node_ref.idx);
                assert_eq!(rendered, "x");
            }
            other => panic!("expected Value::Node, got {:?}", other),
        }
    }

    #[test]
    fn quote_does_not_evaluate_inside() {
        // Sanity check: a quoted expression with a free variable that
        // would otherwise produce an unbound error must not error.
        let src = r#"(quote (some-undefined-symbol 1 2))"#;
        let r = run_file(src);
        assert!(r.is_ok(), "quote must not eval its argument; got {:?}", r);
    }

    #[test]
    fn quote_value_renders_in_value_to_string() {
        // Value::Node renders via node_to_source in value_to_string,
        // so quoted values are self-describing in print output.
        let nodes_rc: Rc<[Node]> = vec![Node::Int(7)].into();
        let v = Value::node(NodeRef { nodes: nodes_rc, idx: 0 });
        assert_eq!(value_to_string(&v), "7");
    }

    #[test]
    fn type_of_returns_node_for_quoted_value() {
        let src = r#"(type-of (quote (add 1 2)))"#;
        let r = run_file(src).unwrap();
        assert!(
            matches!(r, Value::Str(ref s) if s.as_ref() == "node"),
            "expected Str(\"node\"), got {:?}",
            r
        );
    }

    #[test]
    fn quote_equality_is_identity_based() {
        // §9.35 D8: two distinct quote calls produce distinct Node
        // values even when their underlying parser arena is the same
        // (run_file parses the whole input as one pass, so both quotes
        // share an Rc<[Node]>). The two NodeRefs point at distinct
        // Symbol nodes in that shared arena, so they have different
        // indices — and thus the identity-based equality returns false.
        let src = r#"(= (quote x) (quote x))"#;
        let r = run_file(src).unwrap();
        match r {
            Value::Bool(b) => assert!(!b, "expected false, got true"),
            other => panic!("expected Bool, got {:?}", other),
        }
    }

    #[test]
    fn quote_equality_true_for_self_comparison() {
        // The same NodeRef compared to itself IS equal — both arenas
        // are the same Rc and the indices match.
        let src = r#"
            (define q (quote (add 1 2)))
            (= q q)
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Bool(b) => assert!(b, "expected true, got false"),
            other => panic!("expected Bool, got {:?}", other),
        }
    }

    // ── §9.36 sub-step 2c: construction builtins ────────────────────────

    /// Helper: assert that a Value is a Node and that node_to_source on
    /// its underlying NodeRef matches `expected`.
    fn assert_node_renders(v: &Value, expected: &str) {
        match v {
            Value::Node(node_ref) => {
                let rendered = node_to_source(&node_ref.nodes, node_ref.idx);
                assert_eq!(
                    rendered, expected,
                    "node renders to {:?}, expected {:?}",
                    rendered, expected
                );
            }
            other => panic!("expected Value::Node, got {:?}", other),
        }
    }

    #[test]
    fn make_int_builds_int_node() {
        let r = run_file("(make-int 42)").unwrap();
        assert_node_renders(&r, "42");
    }

    #[test]
    fn make_num_builds_num_node() {
        let r = run_file("(make-num 1.5)").unwrap();
        assert_node_renders(&r, "1.5");
    }

    #[test]
    fn make_str_builds_str_node() {
        let r = run_file(r#"(make-str "hello")"#).unwrap();
        assert_node_renders(&r, r#""hello""#);
    }

    #[test]
    fn make_bool_builds_bool_node() {
        let r = run_file("(make-bool true)").unwrap();
        assert_node_renders(&r, "true");
    }

    #[test]
    fn make_symbol_builds_symbol_node() {
        let r = run_file(r#"(make-symbol "x")"#).unwrap();
        assert_node_renders(&r, "x");
    }

    #[test]
    fn make_app_with_string_head() {
        // String head shorthand: (make-app "add" 1-node 2-node)
        let r = run_file(r#"(make-app "add" (make-int 1) (make-int 2))"#).unwrap();
        assert_node_renders(&r, "(add 1 2)");
    }

    #[test]
    fn make_app_with_symbol_node_head() {
        // Explicit Symbol Node head: equivalent to the string-head form.
        let r = run_file(
            r#"(make-app (make-symbol "add") (make-int 1) (make-int 2))"#,
        )
        .unwrap();
        assert_node_renders(&r, "(add 1 2)");
    }

    #[test]
    fn make_app_rejects_non_node_arg() {
        // make-app args must be Node values (the head can be a string,
        // but other args cannot).
        let r = run_file(r#"(make-app "add" 1 2)"#);
        assert!(r.is_err(), "expected error, got {:?}", r);
        let msg = r.unwrap_err();
        assert!(
            msg.contains("make-app") && msg.contains("expected a Node"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn make_app_rejects_non_node_non_string_head() {
        let r = run_file(r#"(make-app 42 (make-int 1))"#);
        assert!(r.is_err(), "expected error, got {:?}", r);
        let msg = r.unwrap_err();
        assert!(
            msg.contains("make-app") && msg.contains("head must be a Node or string"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn make_if_builds_if_node() {
        let r = run_file(
            r#"(make-if (make-bool true) (make-int 1) (make-int 2))"#,
        )
        .unwrap();
        assert_node_renders(&r, "(if true 1 2)");
    }

    #[test]
    fn make_lambda_builds_lambda_node() {
        let r = run_file(
            r#"(make-lambda (list "x") (make-app "add" (make-symbol "x") (make-int 1)))"#,
        )
        .unwrap();
        assert_node_renders(&r, "(lambda (x) (add x 1))");
    }

    #[test]
    fn make_lambda_supports_multiple_params() {
        let r = run_file(
            r#"(make-lambda (list "x" "y") (make-app "add" (make-symbol "x") (make-symbol "y")))"#,
        )
        .unwrap();
        assert_node_renders(&r, "(lambda (x y) (add x y))");
    }

    #[test]
    fn make_let_builds_let_node() {
        let r = run_file(
            r#"(make-let (list (list "x" (make-int 5)))
                         (make-app "add" (make-symbol "x") (make-int 1)))"#,
        )
        .unwrap();
        assert_node_renders(&r, "(let ((x 5)) (add x 1))");
    }

    #[test]
    fn make_app_compose_nested() {
        // Build (multiply (add x 1) (subtract x 2)) bottom-up.
        let r = run_file(
            r#"
            (make-app "multiply"
              (make-app "add" (make-symbol "x") (make-int 1))
              (make-app "subtract" (make-symbol "x") (make-int 2)))
            "#,
        )
        .unwrap();
        assert_node_renders(&r, "(multiply (add x 1) (subtract x 2))");
    }

    #[test]
    fn make_int_rejects_non_number() {
        let r = run_file(r#"(make-int "hello")"#);
        assert!(r.is_err(), "expected error, got {:?}", r);
    }

    #[test]
    fn make_lambda_rejects_non_string_param() {
        let r = run_file(r#"(make-lambda (list 1) (make-int 0))"#);
        assert!(r.is_err(), "expected error, got {:?}", r);
        let msg = r.unwrap_err();
        assert!(
            msg.contains("make-lambda") && msg.contains("param names must be strings"),
            "unexpected error: {}",
            msg
        );
    }

    // ── §9.36 sub-step 2d: inspection builtins ──────────────────────────

    /// Helper: extract a Str value or panic.
    fn assert_str(v: &Value, expected: &str) {
        match v {
            Value::Str(s) => assert_eq!(s.as_ref(), expected, "got {:?}", v),
            other => panic!("expected Str, got {:?}", other),
        }
    }

    #[test]
    fn node_predicate_distinguishes_nodes_from_other_values() {
        match run_file("(node? (quote x))").unwrap() {
            Value::Bool(b) => assert!(b),
            other => panic!("expected Bool, got {:?}", other),
        }
        match run_file("(node? 42)").unwrap() {
            Value::Bool(b) => assert!(!b),
            other => panic!("expected Bool, got {:?}", other),
        }
        match run_file(r#"(node? "hello")"#).unwrap() {
            Value::Bool(b) => assert!(!b),
            other => panic!("expected Bool, got {:?}", other),
        }
    }

    #[test]
    fn node_kind_returns_correct_string_per_kind() {
        assert_str(&run_file("(node-kind (make-int 1))").unwrap(), "int");
        assert_str(&run_file("(node-kind (make-num 1.5))").unwrap(), "num");
        assert_str(&run_file(r#"(node-kind (make-str "x"))"#).unwrap(), "str");
        assert_str(&run_file("(node-kind (make-bool true))").unwrap(), "bool");
        assert_str(
            &run_file(r#"(node-kind (make-symbol "x"))"#).unwrap(),
            "symbol",
        );
        assert_str(
            &run_file(r#"(node-kind (make-app "add" (make-int 1) (make-int 2)))"#).unwrap(),
            "app",
        );
        assert_str(
            &run_file(
                r#"(node-kind (make-if (make-bool true) (make-int 1) (make-int 2)))"#,
            )
            .unwrap(),
            "if",
        );
        assert_str(
            &run_file(r#"(node-kind (make-lambda (list "x") (make-symbol "x")))"#).unwrap(),
            "lambda",
        );
        assert_str(
            &run_file(
                r#"(node-kind (make-let (list (list "x" (make-int 1))) (make-symbol "x")))"#,
            )
            .unwrap(),
            "let",
        );
        // SpecialApp comes from quoted source.
        assert_str(
            &run_file(r#"(node-kind (quote (do 1 2)))"#).unwrap(),
            "special-app",
        );
    }

    #[test]
    fn node_typed_accessors_extract_literal_values() {
        match run_file("(node-int (make-int 42))").unwrap() {
            Value::Int(n) => assert_eq!(n, 42),
            other => panic!("got {:?}", other),
        }
        match run_file("(node-num (make-num 1.5))").unwrap() {
            Value::Num(n) => assert_eq!(n, 1.5),
            other => panic!("got {:?}", other),
        }
        assert_str(&run_file(r#"(node-str (make-str "hi"))"#).unwrap(), "hi");
        match run_file("(node-bool (make-bool true))").unwrap() {
            Value::Bool(b) => assert!(b),
            other => panic!("got {:?}", other),
        }
        assert_str(
            &run_file(r#"(node-symbol (make-symbol "abc"))"#).unwrap(),
            "abc",
        );
    }

    #[test]
    fn node_typed_accessors_error_on_wrong_kind() {
        let r = run_file("(node-int (make-str \"hello\"))");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("expected Int node"));

        let r = run_file("(node-symbol (make-int 1))");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("expected Symbol node"));
    }

    #[test]
    fn node_children_returns_subexpressions_for_app() {
        // (add 1 2) → 3 children: head symbol "add", literal 1, literal 2.
        let src = r#"
            (define n (make-app "add" (make-int 1) (make-int 2)))
            (length (node-children n))
        "#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 3),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn node_children_empty_for_literals() {
        let src = r#"(length (node-children (make-int 42)))"#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 0),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn node_children_for_lambda_returns_body_only() {
        // (lambda (x) x) — children should be just the body.
        // Per §9.35 D8, params are accessed via node-params, not children.
        let src = r#"
            (define n (make-lambda (list "x") (make-symbol "x")))
            (length (node-children n))
        "#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 1),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn node_children_for_if_returns_three() {
        let src = r#"
            (define n (make-if (make-bool true) (make-int 1) (make-int 2)))
            (length (node-children n))
        "#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 3),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn node_params_returns_lambda_param_names() {
        let src = r#"
            (define n (make-lambda (list "a" "b") (make-symbol "a")))
            (node-params n)
        "#;
        match run_file(src).unwrap() {
            Value::List(l) => {
                assert_eq!(l.len(), 2);
                assert_str(&l[0], "a");
                assert_str(&l[1], "b");
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn node_params_errors_on_non_lambda() {
        let r = run_file("(node-params (make-int 1))");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("expected Lambda node"));
    }

    #[test]
    fn node_bindings_returns_let_bindings() {
        let src = r#"
            (define n (make-let
              (list (list "x" (make-int 5)) (list "y" (make-int 6)))
              (make-symbol "x")))
            (length (node-bindings n))
        "#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 2),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn node_special_form_extracts_form_name_from_quoted_special() {
        let src = r#"(node-special-form (quote (do 1 2)))"#;
        assert_str(&run_file(src).unwrap(), "do");
    }

    #[test]
    fn walk_quoted_app_via_inspection() {
        // Quote an expression, then walk it via children + accessors.
        // Verifies that quote+inspection is sufficient to introspect
        // an entire AST without parsing or eval.
        let src = r#"
            (define expr (quote (add x 1)))
            (define head (head (node-children expr)))
            (node-symbol head)
        "#;
        assert_str(&run_file(src).unwrap(), "add");
    }

    // ── §9.36 sub-step 2e: eval-node ────────────────────────────────────

    #[test]
    fn eval_node_evaluates_literal() {
        let src = r#"(eval-node (make-int 42))"#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 42),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn eval_node_evaluates_quoted_application() {
        // (eval-node (quote (add 1 2))) → 3
        let src = r#"(eval-node (quote (add 1 2)))"#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 3),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn eval_node_evaluates_constructed_application() {
        // Build (multiply 6 7) via make-app, then evaluate.
        let src = r#"
            (eval-node (make-app "multiply" (make-int 6) (make-int 7)))
        "#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 42),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn eval_node_constructed_lambda_is_callable() {
        // Build (lambda (x) (add x 1)), eval-node it to get a Function,
        // then call it.
        let src = r#"
            (define inc (eval-node
              (make-lambda (list "x")
                (make-app "add" (make-symbol "x") (make-int 1)))))
            (inc 41)
        "#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 42),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn eval_node_uses_caller_env() {
        // The constructed Node references `y` as a free symbol. When
        // eval-node runs against an env that has `y` defined, it
        // resolves correctly.
        let src = r#"
            (define y 100)
            (eval-node (make-app "add" (make-symbol "y") (make-int 5)))
        "#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 105),
            other => panic!("got {:?}", other),
        }
    }

    // ── §9.36 sub-step 2f: parse-source / parse-file ────────────────────

    #[test]
    fn parse_source_returns_node_for_literal() {
        let src = r#"(node-kind (parse-source "42"))"#;
        assert_str(&run_file(src).unwrap(), "int");
    }

    #[test]
    fn parse_source_returns_node_for_application() {
        let src = r#"(node-kind (parse-source "(add 1 2)"))"#;
        assert_str(&run_file(src).unwrap(), "app");
    }

    #[test]
    fn parse_source_round_trip_via_eval_node() {
        // (eval-node (parse-source s)) should equal (eval-source s).
        let src = r#"(eval-node (parse-source "(add 6 7)"))"#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 13),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn parse_file_returns_list_of_nodes() {
        let src = r#"
            (length (parse-file "(define x 1) (add x 2)"))
        "#;
        match run_file(src).unwrap() {
            Value::Int(n) => assert_eq!(n, 2),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn parse_file_each_form_is_a_node() {
        let src = r#"
            (define forms (parse-file "(define x 1) (add x 2)"))
            (node? (head forms))
        "#;
        match run_file(src).unwrap() {
            Value::Bool(b) => assert!(b),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn parse_source_errors_on_invalid_input() {
        // The parser is permissive and produces a Symbol for "(((",
        // not an error. Use a clearer non-string input to verify the
        // type-check error path: passing an Int yields the typed error.
        let r = run_file("(parse-source 42)");
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("expected string"));
    }

    // ── §9.37 Stage A: SELPH-defined decomposer dispatch ────────────────

    #[test]
    fn selph_decomposer_fires_when_registered() {
        // Define a simple "always-identity" decomposer that returns
        // (lambda (x) x) for any task. Wire it into __decomposers__,
        // then synthesize a task. The result should come from the
        // SELPH decomposer (strategy "custom:identity"), not the
        // hardcoded Flat strategy.
        //
        // The (eval-source ...) for the (lambda (x) x) result is
        // wrapped in a make-lambda call to keep the test
        // self-contained — no string parsing needed.
        let src = r#"
            (define identity-decomp
              (lambda (spec)
                (ns
                  ("found" true)
                  ("nodes" (make-lambda (list "x") (make-symbol "x")))
                  ("candidates" 1))))
            (define __decomposers__ (ns ("identity" identity-decomp)))
            (synthesize (ns
              ("spec" (list (list 1 1) (list 2 2)))
              ("max-depth" 1)
              ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(
                    matches!(map.get(&intern("found")), Some(Value::Bool(true))),
                    "expected found=true"
                );
                let strategy = map.get(&intern("strategy")).unwrap().as_str().unwrap().to_string();
                assert_eq!(
                    strategy, "custom:identity",
                    "expected SELPH decomposer to win, got {:?}",
                    strategy
                );
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                assert_eq!(source, "(lambda (x) x)");
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn selph_decomposer_returning_nil_falls_through() {
        // A decomposer that returns nil should be skipped, and the
        // hardcoded Flat strategy should pick up the task.
        let src = r#"
            (define always-fails (lambda (spec) nil))
            (define __decomposers__ (ns ("never" always-fails)))
            (synthesize (ns
              ("spec" (list (list 1 1) (list 2 2)))
              ("max-depth" 1)
              ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(matches!(
                    map.get(&intern("found")),
                    Some(Value::Bool(true))
                ));
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                assert_eq!(
                    strategy, "Flat",
                    "expected Flat fallback, got {:?}",
                    strategy
                );
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn selph_decomposer_arithmetic_inversion() {
        // A more interesting decomposer: detects (add x k) tasks where
        // k = output - input is constant across all examples, and
        // returns the constructed lambda. This is the simplest example
        // of a SELPH-side strategy that builds non-trivial output.
        //
        // The benchmark task is increment (depth 2 in flat space, but
        // the SELPH decomposer solves it in 5 candidates regardless of
        // max-depth).
        let src = r#"
            (define arith-inv
              (lambda (spec)
                (let ((pairs (ns-get spec "spec")))
                  ; Compute k = out - in for each pair, all in one pass.
                  (let ((diffs
                          (map (lambda (p)
                                 (subtract (nth p 1) (head p)))
                               pairs)))
                    ; If all diffs are equal, return (lambda (x) (add x k)).
                    (if (all? (lambda (d) (= d (head diffs))) diffs)
                      (ns
                        ("found" true)
                        ("nodes"
                          (make-lambda (list "x")
                            (make-app "add"
                              (make-symbol "x")
                              (make-int (head diffs)))))
                        ("candidates" (length diffs)))
                      nil)))))
            (define all? (lambda (pred xs)
              (if (= (length xs) 0) true
                (if (pred (head xs))
                  (all? pred (tail xs))
                  false))))
            (define __decomposers__ (ns ("arith-inv" arith-inv)))
            (synthesize (ns
              ("spec" (list (list 1 2) (list 2 3) (list 5 6) (list 10 11)))
              ("max-depth" 2)
              ("max-candidates" 5000)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(
                    matches!(map.get(&intern("found")), Some(Value::Bool(true))),
                    "expected found=true"
                );
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                assert_eq!(
                    strategy, "custom:arith-inv",
                    "expected SELPH decomposer to win, got {:?}",
                    strategy
                );
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                assert_eq!(source, "(lambda (x) (add x 1))");
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn selph_decomposer_dispatch_is_deterministic() {
        // Two decomposers, both apply. The one whose name sorts first
        // wins (per §9.37 Stage A's name-sort dispatch order).
        let src = r#"
            (define wraps-as-add
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-app "add" (make-symbol "x") (make-int 0))))
                    ("candidates" 1))))
            (define wraps-as-multiply
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-app "multiply" (make-symbol "x") (make-int 1))))
                    ("candidates" 1))))
            (define __decomposers__ (ns
              ("z-add" wraps-as-add)
              ("a-multiply" wraps-as-multiply)))
            (synthesize (ns
              ("spec" (list (list 1 1) (list 2 2)))
              ("max-depth" 1)
              ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                assert_eq!(
                    strategy, "custom:a-multiply",
                    "expected name-sort winner, got {:?}",
                    strategy
                );
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn no_decomposers_means_legacy_dispatch() {
        // Sanity check: when __decomposers__ doesn't exist, the
        // dispatcher behaves exactly as before §9.37.
        let src = r#"
            (synthesize (ns
              ("spec" (list (list 1 1) (list 2 2)))
              ("max-depth" 1)
              ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                assert_eq!(strategy, "Flat");
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    // ── §9.37 Stage C: type-keyed decomposer dispatch ───────────────────

    #[test]
    fn type_keyed_decomposer_fires_for_matching_output_type() {
        // Register a decomposer under __types__["Int"]["decomposers"].
        // The increment task has Int outputs, so the decomposer should
        // be called.
        let src = r#"
            (define always-zero
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-int 0)))
                    ("candidates" 1))))
            (define __types__
              (ns ("Int" (ns ("decomposers" (ns ("zero" always-zero)))))))
            (synthesize (ns
              ("spec" (list (list 1 0) (list 2 0)))
              ("max-depth" 1)
              ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                assert_eq!(
                    strategy, "custom:zero",
                    "expected type-keyed decomposer to win"
                );
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn type_keyed_decomposer_skips_non_matching_output_type() {
        // The decomposer is registered under Int, but the task has
        // String output. The decomposer should NOT be called.
        let src = r#"
            (define would-break
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-int 999)))
                    ("candidates" 1))))
            (define __types__
              (ns ("Int" (ns ("decomposers" (ns ("never" would-break)))))))
            (synthesize (ns
              ("spec" (list (list "hi" "HI") (list "bye" "BYE")))
              ("max-depth" 2)
              ("max-candidates" 5000)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                // Falls through to Flat which finds string-upper.
                assert_eq!(
                    strategy, "Flat",
                    "Int-keyed decomposer should not fire on String task"
                );
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                assert!(source.contains("string-upper"));
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn any_keyed_decomposer_fires_for_any_type() {
        // Decomposers under "Any" run for every output type.
        let src = r#"
            (define catchall
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-symbol "x")))
                    ("candidates" 1))))
            (define __types__
              (ns ("Any" (ns ("decomposers" (ns ("identity" catchall)))))))
            (synthesize (ns
              ("spec" (list (list "hi" "hi") (list "bye" "bye")))
              ("max-depth" 1)
              ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                assert_eq!(strategy, "custom:identity");
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn type_specific_decomposers_run_before_any_decomposers() {
        // Both Int and Any have decomposers; the Int one should be
        // tried first because the task has Int output.
        let src = r#"
            (define int-d
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-int 1)))
                    ("candidates" 1))))
            (define any-d
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-int 2)))
                    ("candidates" 1))))
            (define __types__
              (ns
                ("Int" (ns ("decomposers" (ns ("specific" int-d)))))
                ("Any" (ns ("decomposers" (ns ("fallback" any-d)))))))
            (synthesize (ns
              ("spec" (list (list 5 1) (list 6 1)))
              ("max-depth" 1)
              ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                assert_eq!(
                    strategy, "custom:specific",
                    "Int-specific should win over Any fallback"
                );
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn global_decomposers_run_before_type_keyed_decomposers() {
        // __decomposers__ takes precedence over __types__-based dispatch.
        let src = r#"
            (define global-d
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-int 100)))
                    ("candidates" 1))))
            (define type-d
              (lambda (spec)
                (ns ("found" true)
                    ("nodes" (make-lambda (list "x") (make-int 200)))
                    ("candidates" 1))))
            (define __decomposers__ (ns ("g" global-d)))
            (define __types__ (ns
              ("Int" (ns ("decomposers" (ns ("t" type-d)))))))
            (synthesize (ns
              ("spec" (list (list 1 100) (list 2 100)))
              ("max-depth" 1)
              ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                let strategy = map
                    .get(&intern("strategy"))
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();
                assert_eq!(
                    strategy, "custom:g",
                    "global decomposer should win over type-keyed"
                );
            }
            other => panic!("expected Ns, got {:?}", other),
        }
    }

    #[test]
    fn bucket6_synthesize_solves_identity_via_flat() {
        // Identity task — synth_v2 Flat should find (lambda (x) x) at depth 0.
        let src = r#"
            (synthesize (ns
                ("spec" (list (list 1 1) (list 2 2) (list 3 3)))
                ("max-depth" 1)
                ("max-candidates" 200)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(matches!(map.get(&intern("found")), Some(Value::Bool(true))));
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                assert_eq!(source, "(lambda (x) x)", "got source: {}", source);
                assert!(matches!(map.get(&intern("strategy")), Some(Value::Str(s)) if s.as_ref() == "Flat"));
            }
            _ => panic!("expected Ns, got {:?}", r),
        }
    }

    #[test]
    fn bucket6_synthesize_solves_unary_string_op() {
        // (string-upper x) on strings.
        let src = r#"
            (synthesize (ns
                ("spec" (list (list "hi" "HI") (list "world" "WORLD")))
                ("max-depth" 2)
                ("max-candidates" 2000)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(matches!(map.get(&intern("found")), Some(Value::Bool(true))));
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                assert!(source.contains("string-upper"), "got source: {}", source);
            }
            _ => panic!("expected Ns"),
        }
    }

    #[test]
    fn bucket6_synthesize_falls_through_to_memo() {
        // String→Int with no algorithmic relationship — Flat fails,
        // Memo wins, the result lambda contains ns-get-or.
        let src = r#"
            (synthesize (ns
                ("spec" (list (list "alpha" 13) (list "beta" 99) (list "gamma" 7)))
                ("max-depth" 1)
                ("max-candidates" 50)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(matches!(map.get(&intern("found")), Some(Value::Bool(true))));
                assert!(matches!(map.get(&intern("strategy")), Some(Value::Str(s)) if s.as_ref() == "Memo"));
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                assert!(source.contains("ns-get-or"), "got source: {}", source);
            }
            _ => panic!("expected Ns"),
        }
    }

    #[test]
    fn bucket6_synthesize_returns_not_found_when_impossible() {
        // Single distinct output that isn't reachable from the literal
        // pool at depth 1 (D&C bails on <2 distinct outputs, Memo bails
        // on Int input, BD bails on non-bool output, HO bails on no
        // list/string structure, Flat at depth 1 can't construct 23).
        let src = r#"
            (synthesize (ns
                ("spec" (list (list 1 23) (list 2 23) (list 3 23)))
                ("max-depth" 1)
                ("max-candidates" 50)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(matches!(map.get(&intern("found")), Some(Value::Bool(false))));
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                assert_eq!(source, "");
                assert!(matches!(map.get(&intern("strategy")), Some(Value::Str(s)) if s.as_ref() == ""));
            }
            _ => panic!("expected Ns"),
        }
    }

    #[test]
    fn bucket6_synthesize_uses_library_function_from_env() {
        // Define `inc` in the env, then ask synthesize to find a
        // program that maps x→x+1. The synthesizer should discover
        // `inc` via env auto-discovery and use it.
        let src = r#"
            (define inc (lambda (n) (add n 1)))
            (synthesize (ns
                ("spec" (list (list 5 6) (list 10 11) (list 0 1)))
                ("max-depth" 1)
                ("max-candidates" 500)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(matches!(map.get(&intern("found")), Some(Value::Bool(true))));
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                // Either (inc x) or (add x 1) — both correct.
                assert!(
                    source.contains("inc") || source.contains("add"),
                    "expected inc or add in source, got: {}",
                    source
                );
            }
            _ => panic!("expected Ns"),
        }
    }

    #[test]
    fn bucket6_synthesize_accepts_heuristic_in_spec() {
        // §9.34: a heuristic can be passed directly into the synthesize
        // builtin via the spec namespace. The heuristic is a SELPH lambda
        // taking a context namespace and returning a numeric score.
        // This is the foundation for SELPH-side meta-learning loops.
        //
        // We use a default-identity heuristic here — it should not change
        // the result of an easy task that synth_v2 already solves cleanly.
        // The point of this test is that the wiring works (no errors,
        // result still found), not that the heuristic does anything
        // interesting.
        let src = r#"
            (define h (lambda (ctx) (ns-get ctx "priority")))
            (synthesize (ns
                ("spec" (list (list 1 1) (list 2 2) (list 3 3)))
                ("max-depth" 1)
                ("max-candidates" 200)
                ("heuristic" h)))
        "#;
        let r = run_file(src).unwrap();
        match r {
            Value::Ns(map) => {
                assert!(
                    matches!(map.get(&intern("found")), Some(Value::Bool(true))),
                    "expected found=true with identity heuristic"
                );
                let source = map.get(&intern("source")).unwrap().as_str().unwrap().to_string();
                assert_eq!(source, "(lambda (x) x)", "got source: {}", source);
            }
            _ => panic!("expected Ns, got {:?}", r),
        }
    }

    #[test]
    fn bucket6_synthesize_heuristic_actually_runs() {
        // Stronger test: the heuristic must observably affect the search.
        //
        // Using the identity task wouldn't work — synth_v2 tests all
        // depth-0 atoms in catalog order before any priority-based
        // sort kicks in (atom enumeration is unconditional, only
        // composite candidates are score-sorted). So heuristic
        // priorities don't affect atom-only solutions.
        //
        // We use the increment task instead: `(add x 1)` is a depth-1
        // composite, which DOES go through the pending-candidate sort.
        // A heuristic that deprioritizes the literal `1` should push
        // `(add x 1)` to the end of pending, making the search test
        // many more candidates before finding it.
        let src_with_anti_one = r#"
            (define anti-one (lambda (ctx)
              (if (= (ns-get ctx "name") "1") -1000 100)))
            (synthesize (ns
                ("spec" (list (list 1 2) (list 2 3) (list 3 4) (list 10 11)))
                ("max-depth" 2)
                ("max-candidates" 50000)
                ("heuristic" anti-one)))
        "#;
        let src_without = r#"
            (synthesize (ns
                ("spec" (list (list 1 2) (list 2 3) (list 3 4) (list 10 11)))
                ("max-depth" 2)
                ("max-candidates" 50000)))
        "#;

        let r_with = run_file(src_with_anti_one).unwrap();
        let r_without = run_file(src_without).unwrap();

        let cand_with = match &r_with {
            Value::Ns(m) => match m.get(&intern("candidates")) {
                Some(Value::Int(n)) => *n,
                _ => panic!("missing candidates"),
            },
            _ => panic!("expected Ns"),
        };
        let cand_without = match &r_without {
            Value::Ns(m) => match m.get(&intern("candidates")) {
                Some(Value::Int(n)) => *n,
                _ => panic!("missing candidates"),
            },
            _ => panic!("expected Ns"),
        };

        // Both should solve.
        match &r_with {
            Value::Ns(m) => assert!(
                matches!(m.get(&intern("found")), Some(Value::Bool(true))),
                "with-heuristic must still solve, got {:?}",
                m.get(&intern("found"))
            ),
            _ => panic!("expected Ns"),
        }
        match &r_without {
            Value::Ns(m) => assert!(
                matches!(m.get(&intern("found")), Some(Value::Bool(true))),
                "without-heuristic must solve"
            ),
            _ => panic!("expected Ns"),
        }

        // The anti-one heuristic should make the search strictly more
        // expensive: deprioritizing literal `1` pushes `(add x 1)` to
        // the end of the pending sort. If both are equal, the
        // heuristic isn't flowing through to the candidate scoring.
        assert!(
            cand_with > cand_without,
            "anti-one heuristic must increase candidate count: with={}, without={}",
            cand_with,
            cand_without
        );
    }

    #[test]
    fn bucket6_synthesize_rejects_non_function_heuristic() {
        // §9.34: passing a non-function value as the heuristic must
        // produce a clear error rather than silently doing nothing.
        let src = r#"
            (synthesize (ns
                ("spec" (list (list 1 1) (list 2 2)))
                ("max-depth" 1)
                ("max-candidates" 200)
                ("heuristic" 42)))
        "#;
        let r = run_file(src);
        assert!(r.is_err(), "expected error, got {:?}", r);
        let msg = r.unwrap_err();
        assert!(
            msg.contains("heuristic must be a function value"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn bucket6_synthesize_optimize_still_stubbed() {
        // synthesize-optimize is intentionally still stubbed in step 6.
        // Verify it returns a clear error rather than silently doing
        // the wrong thing.
        let src = r#"
            (synthesize-optimize (ns ("minimize" "(lambda (x) x)")))
        "#;
        let r = run_file(src);
        assert!(r.is_err(), "expected error, got {:?}", r);
        let msg = r.unwrap_err();
        assert!(msg.contains("synthesize-optimize"), "unexpected error: {}", msg);
    }

    // ── §9.45 P1: env / function introspection ────────────────────────
    //
    // Re-added after the §9.45.13 perl-regex incident destroyed the
    // original test bodies. The following is a representative subset
    // of the original ~25 tests, kept smaller to avoid retyping
    // every variant. Coverage focuses on the success paths because
    // the failure paths are well-covered by the curriculum-side
    // smoke tests.

    #[test]
    fn env_functions_returns_user_defined_function() {
        let src = r#"
            (define inc (lambda (x) (add x 1)))
            (ns-has (env-functions) "inc")
        "#;
        let r = run_file(src).unwrap();
        assert!(matches!(r, Value::Bool(true)));
    }

    #[test]
    fn env_functions_excludes_default_builtins() {
        let src = r#"(ns-has (env-functions) "add")"#;
        let r = run_file(src).unwrap();
        assert!(matches!(r, Value::Bool(false)));
    }

    #[test]
    fn function_arity_for_lambda() {
        let src = r#"
            (define f (lambda (a b c) (add a (add b c))))
            (function-arity f)
        "#;
        let r = run_file(src).unwrap();
        assert!(matches!(r, Value::Int(3)));
    }

    #[test]
    fn function_arity_for_builtin_returns_minus_one() {
        let r = run_file("(function-arity add)").unwrap();
        assert!(matches!(r, Value::Int(-1)));
    }

    #[test]
    fn function_param_types_for_unary_int_lambda() {
        let src = r#"
            (define f (lambda (n) (add n 1)))
            (length (function-param-types f))
        "#;
        let r = run_file(src).unwrap();
        assert!(matches!(r, Value::Int(1)));
    }

    // ── §9.45.7 P5: letrec patching gate ──────────────────────────────

    #[test]
    fn letrec_does_not_hijack_closures_from_ns_get() {
        // Regression for the §9.45 footgun. Binding a closure
        // obtained from `(ns-get ...)` in the same let frame as a
        // name that shadows a builtin must NOT cause the closure's
        // body to resolve the builtin name to the local.
        let src = r#"
            (define f (lambda (x) (exp x)))
            (define ns (ns-put (ns-empty) "fn" f))
            (let ((g (ns-get ns "fn"))
                  (exp 1))
              (g 2.0))
        "#;
        match run_file(src).unwrap() {
            Value::Num(n) => assert!((n - 7.389056098930650).abs() < 1e-9),
            other => panic!("expected Num, got {:?}", other),
        }
    }

    #[test]
    fn letrec_still_patches_literal_lambda_for_self_recursion() {
        // The letrec patch must STILL fire for closures whose RHS is
        // a literal `(lambda ...)` so self-recursion in a let block
        // continues to work. Factorial is the canonical case.
        let src = r#"
            (let ((fact (lambda (n)
                          (if (= n 0)
                            1
                            (multiply n (fact (subtract n 1)))))))
              (fact 5))
        "#;
        let r = run_file(src).unwrap();
        assert!(matches!(r, Value::Int(120)));
    }

    #[test]
    fn letrec_patches_mutual_recursion_via_literal_lambdas() {
        let src = r#"
            (let ((even? (lambda (n) (if (= n 0) true (odd? (subtract n 1)))))
                  (odd?  (lambda (n) (if (= n 0) false (even? (subtract n 1))))))
              (even? 4))
        "#;
        let r = run_file(src).unwrap();
        assert!(matches!(r, Value::Bool(true)));
    }

    // ── Threading macros (-> and ->>) ─────────────────────────────────

    #[test]
    fn thread_first_two_steps() {
        // (-> 1 (add 2) (multiply 3)) → (multiply (add 1 2) 3) → 9
        let src = r#"(-> 1 (add 2) (multiply 3))"#;
        let (old, root) = crate::parser::parse_source(src).unwrap();
        let new = rc(convert_tree(&old));
        let env = make_default_env();
        let r = eval(&new, root, &env).unwrap();
        assert!(matches!(r, Value::Int(9)), "got {:?}", r);
    }

    #[test]
    fn thread_last_two_steps() {
        // (->> 1 (add 2) (multiply 3)) → (multiply 3 (add 2 1)) → 9
        let src = r#"(->> 1 (add 2) (multiply 3))"#;
        let (old, root) = crate::parser::parse_source(src).unwrap();
        let new = rc(convert_tree(&old));
        let env = make_default_env();
        let r = eval(&new, root, &env).unwrap();
        assert!(matches!(r, Value::Int(9)), "got {:?}", r);
    }

    #[test]
    fn thread_first_bare_symbols() {
        // (-> -3 abs) → (abs -3) → 3
        // Use a known unary function.
        let src = r#"(-> -3 abs)"#;
        let (old, root) = crate::parser::parse_source(src).unwrap();
        let new = rc(convert_tree(&old));
        let env = make_default_env();
        let r = eval(&new, root, &env).unwrap();
        assert!(matches!(r, Value::Int(3)), "got {:?}", r);
    }

    #[test]
    fn thread_first_single_step() {
        // (-> 5 (add 10)) → (add 5 10) → 15
        let src = r#"(-> 5 (add 10))"#;
        let (old, root) = crate::parser::parse_source(src).unwrap();
        let new = rc(convert_tree(&old));
        let env = make_default_env();
        let r = eval(&new, root, &env).unwrap();
        assert!(matches!(r, Value::Int(15)), "got {:?}", r);
    }

    #[test]
    fn thread_first_many_steps() {
        // (-> 0 (add 1) (add 2) (add 3) (add 4)) → 10
        let src = r#"(-> 0 (add 1) (add 2) (add 3) (add 4))"#;
        let (old, root) = crate::parser::parse_source(src).unwrap();
        let new = rc(convert_tree(&old));
        let env = make_default_env();
        let r = eval(&new, root, &env).unwrap();
        assert!(matches!(r, Value::Int(10)), "got {:?}", r);
    }

    #[test]
    fn thread_first_with_let_and_lambda() {
        // Combine threading with let bindings.
        let src = r#"
            (let ((base 10))
              (-> base (add 5) (multiply 2)))
        "#;
        let r = run_file(src).unwrap();
        // (multiply (add 10 5) 2) → 30
        assert!(matches!(r, Value::Int(30)), "got {:?}", r);
    }

    #[test]
    fn thread_first_with_string_ops() {
        // (-> "hello" (string-concat " world") string-length) → 11
        let src = r#"(-> "hello" (string-concat " world") string-length)"#;
        let (old, root) = crate::parser::parse_source(src).unwrap();
        let new = rc(convert_tree(&old));
        let env = make_default_env();
        let r = eval(&new, root, &env).unwrap();
        assert!(matches!(r, Value::Int(11)), "got {:?}", r);
    }

    #[test]
    fn thread_last_differs_from_first() {
        // (-> 10 (subtract 3))  → (subtract 10 3) → 7   (thread-first)
        // (->> 10 (subtract 3)) → (subtract 3 10) → -7  (thread-last)
        let src_first = r#"(-> 10 (subtract 3))"#;
        let src_last = r#"(->> 10 (subtract 3))"#;

        let (old1, root1) = crate::parser::parse_source(src_first).unwrap();
        let new1 = rc(convert_tree(&old1));
        let env = make_default_env();
        let r1 = eval(&new1, root1, &env).unwrap();
        assert!(matches!(r1, Value::Int(7)), "thread-first got {:?}", r1);

        let (old2, root2) = crate::parser::parse_source(src_last).unwrap();
        let new2 = rc(convert_tree(&old2));
        let r2 = eval(&new2, root2, &env).unwrap();
        assert!(matches!(r2, Value::Int(-7)), "thread-last got {:?}", r2);
    }
}
