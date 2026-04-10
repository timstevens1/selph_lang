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
        if matches!(&v, Value::Function(_)) {
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
            // (quote x) — return the source representation as a string.
            // Stub for now; needs node_to_source translated to new Node.
            if children.len() != 1 {
                return Err("quote: expected one argument".into());
            }
            // TODO: implement node_to_source for new Node enum.
            Err("quote: not yet implemented in eval_v2".into())
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
    t.register(intern("synthesize-optimize"), bi_stub_synthesize_optimize);
    t.register(intern("test-spec"), bi_test_spec);
    t.register(intern("memorize"), bi_memorize);
    t.register(intern("eval-source"), bi_eval_source);

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
        "synthesize", "synthesize-optimize", "test-spec", "memorize", "eval-source",
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
        if let Value::Bool(true) = apply(f, &[item.clone()], env)? {
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
    let components = crate::synth_v2::default_synth_components(env, &skip);
    let universe = crate::synth_v2::TypeUniverse::primitives();

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
        // Int→arbitrary string with non-string inputs (Memo can't help).
        let src = r#"
            (synthesize (ns
                ("spec" (list (list 1 "foo") (list 2 "bar")))
                ("max-depth" 2)
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
}
