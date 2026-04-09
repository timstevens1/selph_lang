//! Evaluator (standalone, no PyO3 dependency)

use crate::intern::{Sym, intern, resolve};
use crate::types::*;
use std::rc::Rc;

// Maximum eval recursion depth to prevent stack overflow from
// deeply nested or self-referential macros.
const MAX_EVAL_DEPTH: usize = 256;

thread_local! {
    static EVAL_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Shared empty node slice for builtins that need a node reference (map, reduce, filter, apply).
fn empty_nodes() -> Rc<[Node]> {
    thread_local! {
        static EMPTY: Rc<[Node]> = Rc::from(Vec::<Node>::new());
    }
    EMPTY.with(|e| Rc::clone(e))
}

pub fn eval(nodes: &Rc<[Node]>, idx: usize, env: &mut Env) -> Result<Value, String> {
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

fn eval_inner(nodes: &Rc<[Node]>, idx: usize, env: &mut Env) -> Result<Value, String> {
    match &nodes[idx] {
        Node::Num(n) => Ok(Value::Num(*n)),
        Node::Str(s) => Ok(Value::Str(s.clone())),
        Node::Bool(b) => Ok(Value::Bool(*b)),
        Node::Symbol(name) => {
            env_lookup(env, *name).ok_or_else(|| format!("unbound: {}", resolve(*name)))
        }
        Node::Lambda(params, body) => {
            Ok(Value::Closure(params.clone(), *body, env.clone(), Rc::clone(nodes), None))
        }
        Node::If(cond, then_br, else_br) => {
            let cond_val = eval(nodes, *cond, env)?;
            match cond_val {
                Value::Bool(true) => eval(nodes, *then_br, env),
                Value::Num(n) if n != 0.0 => eval(nodes, *then_br, env),
                _ => eval(nodes, *else_br, env),
            }
        }
        Node::Let(bindings, body) => {
            // letrec semantics: all bindings share a single child env so that
            // closures can refer to names defined in the same let block
            // (enables self-recursion and mutual recursion).
            //
            // We use a SharedScope (Rc<RefCell<HashMap>>) so all closures
            // created in this let block share the same mutable scope.
            // After all bindings are evaluated, we populate the shared scope
            // with the final values (including patched closures), and every
            // closure that references it will see the complete set of bindings.
            env.push(std::collections::HashMap::new());

            let shared_scope: SharedScope = std::rc::Rc::new(std::cell::RefCell::new(
                std::collections::HashMap::new(),
            ));

            // Track which binding names got closure values
            let mut closure_names: Vec<Sym> = Vec::new();

            for (name, val_idx) in bindings {
                let val = eval(nodes, *val_idx, env)?;
                if matches!(&val, Value::Closure(..)) {
                    closure_names.push(*name);
                }
                env_define(env, *name, val);
            }

            // Patch closures: give them the shared scope so recursive
            // references resolve through it at call time.
            if !closure_names.is_empty() {
                if let Some(scope) = env.last_mut() {
                    for cname in &closure_names {
                        if let Some(Value::Closure(params, body_idx, captured_env, nodes_vec, _)) = scope.get(cname).cloned() {
                            scope.insert(*cname, Value::Closure(
                                params, body_idx, captured_env, nodes_vec,
                                Some(shared_scope.clone()),
                            ));
                        }
                    }
                }
                // Now populate the shared scope with all let bindings
                // (including the patched closures).
                if let Some(scope) = env.last() {
                    let mut shared = shared_scope.borrow_mut();
                    for (k, v) in scope.iter() {
                        shared.insert(*k, v.clone());
                    }
                }
            }

            let result = eval(nodes, *body, env);
            env.pop();
            result
        }
        Node::App(children) => {
            if children.is_empty() {
                return Ok(Value::List(vec![]));
            }

            // Check for special forms
            if let Node::Symbol(name) = &nodes[children[0]] {
                let name_str = resolve(*name);
                match name_str.as_str() {
                    "define" if children.len() == 3 => {
                        if let Node::Symbol(var_name) = &nodes[children[1]] {
                            let val = eval(nodes, children[2], env)?;
                            env_define(env, *var_name, val.clone());
                            return Ok(val);
                        }
                    }
                    "defmacro" if children.len() == 4 => {
                        if let Node::Symbol(macro_name) = &nodes[children[1]] {
                            if let Node::App(param_indices) = &nodes[children[2]] {
                                let params: Vec<Sym> = param_indices.iter()
                                    .filter_map(|&i| {
                                        if let Node::Symbol(s) = &nodes[i] { Some(*s) }
                                        else { None }
                                    }).collect();
                                let body_idx = children[3];
                                let macro_val = Value::RustMacro(
                                    params, Rc::clone(nodes), body_idx);
                                env_define(env, *macro_name, macro_val.clone());
                                return Ok(macro_val);
                            }
                        }
                    }
                    "do" => {
                        let mut result = Value::Nil;
                        for &child in &children[1..] {
                            result = eval(nodes, child, env)?;
                        }
                        return Ok(result);
                    }
                    "quote" if children.len() == 2 => {
                        return Ok(Value::Str(node_to_source(nodes, children[1])));
                    }
                    "and" => {
                        let mut result = Value::Bool(true);
                        for &child in &children[1..] {
                            result = eval(nodes, child, env)?;
                            if matches!(&result, Value::Bool(false)) { return Ok(result); }
                        }
                        return Ok(result);
                    }
                    "or" => {
                        let mut result = Value::Bool(false);
                        for &child in &children[1..] {
                            result = eval(nodes, child, env)?;
                            if !matches!(&result, Value::Bool(false)) { return Ok(result); }
                        }
                        return Ok(result);
                    }
                    "ns" | "namespace" => {
                        let mut entries = std::collections::HashMap::new();
                        for &child in &children[1..] {
                            if let Node::App(pair) = &nodes[child] {
                                if pair.len() == 2 {
                                    let key = match &nodes[pair[0]] {
                                        Node::Symbol(s) => resolve(*s),
                                        Node::Str(s) => s.clone(),
                                        _ => continue,
                                    };
                                    let val = eval(nodes, pair[1], env)?;
                                    entries.insert(key, val);
                                }
                            }
                        }
                        return Ok(Value::Namespace(entries));
                    }
                    // try: evaluate expr, on error return fallback
                    // (try expr fallback)
                    "try" if children.len() == 3 => {
                        match eval(nodes, children[1], env) {
                            Ok(v) => return Ok(v),
                            Err(_) => return eval(nodes, children[2], env),
                        }
                    }
                    // eval-in: evaluate a source string in the current env
                    // (eval-in "(add x 1)") — x must be bound in current scope
                    "eval-in" if children.len() == 2 => {
                        let src_val = eval(nodes, children[1], env)?;
                        let src = match &src_val {
                            Value::Str(s) => s.clone(),
                            _ => return Err("eval-in: expected string".into()),
                        };
                        let (new_nodes, roots) = crate::parser::parse_file(&src)
                            .map_err(|e| format!("eval-in: parse error: {}", e))?;
                        if roots.is_empty() {
                            return Ok(Value::Nil);
                        }
                        let new_nodes_rc: Rc<[Node]> = new_nodes.into();
                        let mut last = Value::Nil;
                        for &r in &roots {
                            last = eval(&new_nodes_rc, r, env)?;
                        }
                        return Ok(last);
                    }
                    // dispatch: look up a macro by name string and apply it
                    // (dispatch "reverse" "hello") → looks up "reverse" in env, applies to "hello"
                    "dispatch" if children.len() == 3 => {
                        let name_val = eval(nodes, children[1], env)?;
                        let name_str = match &name_val {
                            Value::Str(s) => s.clone(),
                            _ => return Err("dispatch: first arg must be string".into()),
                        };
                        let arg = eval(nodes, children[2], env)?;
                        let key = intern(&name_str);
                        let func = env_lookup(env, key)
                            .ok_or_else(|| format!("dispatch: unknown operation '{}'", name_str))?;
                        return apply(&func, &[arg], nodes, env);
                    }
                    _ => {}
                }
            }

            let fn_val = eval(nodes, children[0], env)?;
            let mut args = Vec::new();
            for &child in &children[1..] {
                args.push(eval(nodes, child, env)?);
            }
            apply(&fn_val, &args, nodes, env)
        }
    }
}

pub fn apply(fn_val: &Value, args: &[Value], _nodes: &Rc<[Node]>, env: &mut Env) -> Result<Value, String> {
    match fn_val {
        Value::Closure(params, body, closed_env, closure_nodes, letrec_scope) => {
            let mut new_env = closed_env.clone();
            // If this closure was created in a letrec, push the shared scope
            // so recursive/mutual references resolve correctly.
            if let Some(shared) = letrec_scope {
                new_env.push(shared.borrow().clone());
            }
            let mut scope = std::collections::HashMap::new();
            for (i, param) in params.iter().enumerate() {
                if i < args.len() {
                    scope.insert(*param, args[i].clone());
                }
            }
            new_env.push(scope);
            // Use the closure's captured nodes, not the caller's nodes
            eval(closure_nodes, *body, &mut new_env)
        }
        Value::Builtin(name) => apply_builtin(*name, args),
        Value::RustMacro(params, macro_nodes, body_root) => {
            let mut macro_env = make_default_env();
            for scope in env.iter() {
                for (k, v) in scope {
                    env_define(&mut macro_env, *k, v.clone());
                }
            }
            let mut scope = std::collections::HashMap::new();
            for (i, param) in params.iter().enumerate() {
                if i < args.len() {
                    scope.insert(*param, args[i].clone());
                }
            }
            macro_env.push(scope);
            eval(macro_nodes, *body_root, &mut macro_env)
        }
        _ => Err(format!("not callable: {:?}", fn_val)),
    }
}

// ── Builtin dispatch table ───────────────────────────────────────────
// Maps Sym → fn pointer for O(1) dispatch. Covers all builtins except
// those that call back into eval/apply (map, reduce, filter, apply,
// eval-source, eval-in, synthesize, synthesize-optimize).

type BuiltinFn = fn(&[Value]) -> Result<Value, String>;

fn build_dispatch_table() -> std::collections::HashMap<Sym, BuiltinFn> {
    let mut m: std::collections::HashMap<Sym, BuiltinFn> = std::collections::HashMap::new();
    // Arithmetic
    m.insert(intern("add"), |args| num2(args, |a, b| a + b));
    m.insert(intern("+"), |args| num2(args, |a, b| a + b));
    m.insert(intern("subtract"), |args| num2(args, |a, b| a - b));
    m.insert(intern("-"), |args| num2(args, |a, b| a - b));
    m.insert(intern("multiply"), |args| num2(args, |a, b| a * b));
    m.insert(intern("*"), |args| num2(args, |a, b| a * b));
    m.insert(intern("divide"), |args| { let (a, b) = nums(args)?; if b == 0.0 { Err("division by zero".into()) } else { Ok(Value::Num(a / b)) } });
    m.insert(intern("/"), |args| { let (a, b) = nums(args)?; if b == 0.0 { Err("division by zero".into()) } else { Ok(Value::Num(a / b)) } });
    m.insert(intern("modulo"), |args| num2(args, |a, b| a % b));
    m.insert(intern("%"), |args| num2(args, |a, b| a % b));
    m.insert(intern("abs"), |args| Ok(Value::Num(num(&args[0])?.abs())));
    m.insert(intern("negate"), |args| Ok(Value::Num(-num(&args[0])?)));
    m.insert(intern("min"), |args| num2(args, |a, b| a.min(b)));
    m.insert(intern("max"), |args| num2(args, |a, b| a.max(b)));
    m.insert(intern("floor"), |args| Ok(Value::Num(num(&args[0])?.floor())));
    m.insert(intern("ceil"), |args| Ok(Value::Num(num(&args[0])?.ceil())));
    m.insert(intern("round"), |args| Ok(Value::Num(num(&args[0])?.round())));
    m.insert(intern("pow"), |args| num2(args, |a, b| a.powf(b)));
    m.insert(intern("sqrt"), |args| Ok(Value::Num(num(&args[0])?.sqrt())));
    m.insert(intern("log"), |args| Ok(Value::Num(num(&args[0])?.ln())));
    // Comparison
    m.insert(intern("<"), |args| { let (a, b) = nums(args)?; Ok(Value::Bool(a < b)) });
    m.insert(intern(">"), |args| { let (a, b) = nums(args)?; Ok(Value::Bool(a > b)) });
    m.insert(intern("<="), |args| { let (a, b) = nums(args)?; Ok(Value::Bool(a <= b)) });
    m.insert(intern(">="), |args| { let (a, b) = nums(args)?; Ok(Value::Bool(a >= b)) });
    m.insert(intern("="), |args| match (&args[0], &args[1]) {
        (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(a == b)),
        (Value::Str(a), Value::Str(b)) => Ok(Value::Bool(a == b)),
        _ => Ok(Value::Bool(false)),
    });
    m.insert(intern("not"), |args| match &args[0] { Value::Bool(b) => Ok(Value::Bool(!*b)), _ => Err("not: expected bool".into()) });
    m.insert(intern("even"), |args| Ok(Value::Bool(num(&args[0])? % 2.0 == 0.0)));
    m.insert(intern("odd"), |args| Ok(Value::Bool(num(&args[0])? % 2.0 != 0.0)));
    // String operations
    m.insert(intern("string-upper"), |args| Ok(Value::Str(string(&args[0])?.to_uppercase())));
    m.insert(intern("string-lower"), |args| Ok(Value::Str(string(&args[0])?.to_lowercase())));
    m.insert(intern("string-reverse"), |args| Ok(Value::Str(string(&args[0])?.chars().rev().collect())));
    m.insert(intern("string-trim"), |args| Ok(Value::Str(string(&args[0])?.trim().to_string())));
    m.insert(intern("string-length"), |args| Ok(Value::Num(string(&args[0])?.len() as f64)));
    m.insert(intern("string-contains"), |args| Ok(Value::Bool(string(&args[0])?.contains(&string(&args[1])?))));
    m.insert(intern("string-split"), |args| {
        let s = string(&args[0])?; let sep = string(&args[1])?;
        Ok(Value::List(s.split(&sep).map(|p| Value::Str(p.to_string())).collect()))
    });
    m.insert(intern("string-join"), |args| {
        let lst = list(&args[0])?; let sep = string(&args[1])?;
        let strs: Vec<String> = lst.iter().map(|v| value_to_string(v)).collect();
        Ok(Value::Str(strs.join(&sep)))
    });
    m.insert(intern("concat"), |args| {
        let mut r = String::new();
        for a in args { r.push_str(&value_to_string(a)); }
        Ok(Value::Str(r))
    });
    m.insert(intern("to-string"), |args| Ok(Value::Str(value_to_string(&args[0]))));
    m.insert(intern("to-number"), |args| match &args[0] {
        Value::Str(s) => s.parse::<f64>().map(Value::Num).map_err(|e| format!("to-number: {}", e)),
        Value::Num(n) => Ok(Value::Num(*n)),
        _ => Err("to-number: expected string".into()),
    });
    m.insert(intern("string-nth"), |args| {
        let s = string(&args[0])?; let i = num(&args[1])? as usize;
        s.chars().nth(i).map(|c| Value::Str(c.to_string())).ok_or(format!("string-nth: index {} out of bounds", i))
    });
    m.insert(intern("string-slice"), |args| {
        let s = string(&args[0])?; let start = num(&args[1])? as usize;
        let end = if args.len() > 2 { num(&args[2])? as usize } else { s.len() };
        let chars: Vec<char> = s.chars().collect();
        let end = end.min(chars.len()); let start = start.min(end);
        Ok(Value::Str(chars[start..end].iter().collect()))
    });
    m.insert(intern("string-take"), |args| {
        let s = string(&args[0])?; let n = num(&args[1])? as usize;
        let chars: Vec<char> = s.chars().collect();
        let n = n.min(chars.len());
        Ok(Value::Str(chars[..n].iter().collect()))
    });
    m.insert(intern("string-drop"), |args| {
        let s = string(&args[0])?; let n = num(&args[1])? as usize;
        let chars: Vec<char> = s.chars().collect();
        let n = n.min(chars.len());
        Ok(Value::Str(chars[n..].iter().collect()))
    });
    m.insert(intern("count-char"), |args| {
        let s = string(&args[0])?; let c = string(&args[1])?;
        Ok(Value::Num(s.matches(&c as &str).count() as f64))
    });
    m.insert(intern("string-replace"), |args| {
        let s = string(&args[0])?; let from = string(&args[1])?; let to = string(&args[2])?;
        Ok(Value::Str(s.replace(&from as &str, &to as &str)))
    });
    m.insert(intern("string-chars"), |args| {
        let s = string(&args[0])?;
        Ok(Value::List(s.chars().map(|c| Value::Str(c.to_string())).collect()))
    });
    m.insert(intern("string-starts-with"), |args| {
        let s = string(&args[0])?; let prefix = string(&args[1])?;
        Ok(Value::Bool(s.starts_with(&prefix as &str)))
    });
    m.insert(intern("string-ends-with"), |args| {
        let s = string(&args[0])?; let suffix = string(&args[1])?;
        Ok(Value::Bool(s.ends_with(&suffix as &str)))
    });
    m.insert(intern("char-code"), |args| {
        let s = string(&args[0])?;
        s.chars().next().map(|c| Value::Num(c as u32 as f64)).ok_or("char-code: empty string".into())
    });
    m.insert(intern("code-char"), |args| {
        let n = num(&args[0])? as u32;
        char::from_u32(n).map(|c| Value::Str(c.to_string())).ok_or(format!("code-char: invalid code point {}", n))
    });
    // List operations
    m.insert(intern("list"), |args| Ok(Value::List(args.to_vec())));
    m.insert(intern("head"), |args| { let l = list(&args[0])?; l.first().cloned().ok_or("head: empty".into()) });
    m.insert(intern("tail"), |args| { let l = list(&args[0])?; if l.is_empty() { Err("tail: empty".into()) } else { Ok(Value::List(l[1..].to_vec())) } });
    m.insert(intern("length"), |args| Ok(Value::Num(list(&args[0])?.len() as f64)));
    m.insert(intern("cons"), |args| { let mut l = list(&args[1])?; l.insert(0, args[0].clone()); Ok(Value::List(l)) });
    m.insert(intern("nth"), |args| {
        let l = list(&args[0])?; let i = num(&args[1])? as usize;
        l.get(i).cloned().ok_or(format!("nth: index {} out of bounds (len {})", i, l.len()))
    });
    m.insert(intern("slice"), |args| {
        let l = list(&args[0])?; let start = num(&args[1])? as usize;
        let end = if args.len() > 2 { num(&args[2])? as usize } else { l.len() };
        let end = end.min(l.len()); let start = start.min(end);
        Ok(Value::List(l[start..end].to_vec()))
    });
    m.insert(intern("sort"), |args| {
        let mut l = list(&args[0])?;
        l.sort_by(|a, b| match (a, b) {
            (Value::Num(x), Value::Num(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
            (Value::Str(x), Value::Str(y)) => x.cmp(y),
            _ => std::cmp::Ordering::Equal,
        });
        Ok(Value::List(l))
    });
    m.insert(intern("reverse"), |args| { let mut l = list(&args[0])?; l.reverse(); Ok(Value::List(l)) });
    m.insert(intern("append"), |args| { let mut l1 = list(&args[0])?; let l2 = list(&args[1])?; l1.extend(l2); Ok(Value::List(l1)) });
    m.insert(intern("range"), |args| {
        let n = num(&args[0])? as i64;
        let start = if args.len() > 1 { num(&args[1])? as i64 } else { 0 };
        let (from, to) = if args.len() > 1 { (start, n) } else { (0, n) };
        Ok(Value::List((from..to).map(|i| Value::Num(i as f64)).collect()))
    });
    m.insert(intern("contains"), |args| {
        let l = list(&args[0])?; let target = &args[1];
        let found = l.iter().any(|v| match (v, target) {
            (Value::Num(a), Value::Num(b)) => (a - b).abs() < f64::EPSILON,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            _ => false,
        });
        Ok(Value::Bool(found))
    });
    m.insert(intern("zip"), |args| {
        let l1 = list(&args[0])?; let l2 = list(&args[1])?;
        Ok(Value::List(l1.into_iter().zip(l2).map(|(a, b)| Value::List(vec![a, b])).collect()))
    });
    m.insert(intern("enumerate"), |args| {
        let l = list(&args[0])?;
        Ok(Value::List(l.into_iter().enumerate().map(|(i, v)| Value::List(vec![Value::Num(i as f64), v])).collect()))
    });
    // Misc
    m.insert(intern("identity"), |args| Ok(args[0].clone()));
    m.insert(intern("print"), |args| {
        for a in args { print!("{}", value_to_string(a)); }
        println!();
        Ok(Value::Nil)
    });
    // Namespace operations
    m.insert(intern("ns-get"), |args| {
        if args.len() < 2 { return Err("ns-get: need namespace and key".into()); }
        let mut result = args[0].clone();
        for arg in &args[1..] {
            let key = match arg { Value::Str(s) => s.clone(), _ => return Err("ns-get: key must be string".into()) };
            result = crate::namespace::ns_get(&result, &key)?;
        }
        Ok(result)
    });
    m.insert(intern("ns-put"), |args| {
        if args.len() != 3 { return Err("ns-put: need namespace, key, value".into()); }
        let key = match &args[1] { Value::Str(s) => s.clone(), _ => return Err("ns-put: key must be string".into()) };
        crate::namespace::ns_put(&args[0], &key, args[2].clone())
    });
    m.insert(intern("ns-keys"), |args| crate::namespace::ns_keys(&args[0]).map(|ks| Value::List(ks.into_iter().map(Value::Str).collect())));
    m.insert(intern("ns-values"), |args| crate::namespace::ns_values(&args[0]).map(Value::List));
    m.insert(intern("ns-merge"), |args| {
        if args.len() != 2 { return Err("ns-merge: need two namespaces".into()); }
        crate::namespace::ns_merge(&args[0], &args[1])
    });
    m.insert(intern("ns-size"), |args| crate::namespace::ns_size(&args[0]).map(|n| Value::Num(n as f64)));
    m.insert(intern("ns-flatten"), |args| {
        crate::namespace::ns_flatten(&args[0]).map(|m| {
            let mut entries = std::collections::HashMap::new();
            for (k, v) in m { entries.insert(k, v); }
            Value::Namespace(entries)
        })
    });
    m.insert(intern("ns?"), |args| Ok(Value::Bool(matches!(&args[0], Value::Namespace(_)))));
    m.insert(intern("ns-empty"), |_args| Ok(Value::Namespace(std::collections::HashMap::new())));
    m.insert(intern("ns-get-or"), |args| {
        if args.len() != 3 { return Err("ns-get-or: need namespace, key, default".into()); }
        let key = match &args[1] { Value::Str(s) => s.clone(), _ => return Err("ns-get-or: key must be string".into()) };
        match crate::namespace::ns_get(&args[0], &key) { Ok(v) => Ok(v), Err(_) => Ok(args[2].clone()) }
    });
    m.insert(intern("ns-has"), |args| {
        if args.len() != 2 { return Err("ns-has: need namespace and key".into()); }
        let key = match &args[1] { Value::Str(s) => s.clone(), _ => return Err("ns-has: key must be string".into()) };
        Ok(Value::Bool(crate::namespace::ns_get(&args[0], &key).is_ok()))
    });
    // Type predicates
    m.insert(intern("type-of"), |args| {
        let type_name = match &args[0] {
            Value::Num(_) => "number", Value::Str(_) => "string", Value::Bool(_) => "bool",
            Value::List(_) => "list", Value::Grid(_) => "grid", Value::Namespace(_) => "namespace", Value::Nil => "nil",
            Value::Closure(..) | Value::Builtin(_) | Value::RustMacro(..) => "function",
            Value::Alt(_) => "alt",
        };
        Ok(Value::Str(type_name.to_string()))
    });
    m.insert(intern("number?"), |args| Ok(Value::Bool(matches!(&args[0], Value::Num(_)))));
    m.insert(intern("string?"), |args| Ok(Value::Bool(matches!(&args[0], Value::Str(_)))));
    m.insert(intern("bool?"), |args| Ok(Value::Bool(matches!(&args[0], Value::Bool(_)))));
    m.insert(intern("list?"), |args| Ok(Value::Bool(matches!(&args[0], Value::List(_)))));
    m.insert(intern("nil?"), |args| Ok(Value::Bool(matches!(&args[0], Value::Nil))));
    m.insert(intern("function?"), |args| Ok(Value::Bool(matches!(&args[0], Value::Closure(..) | Value::Builtin(_) | Value::RustMacro(..)))));
    m.insert(intern("try"), |args| Ok(args[0].clone()));
    m.insert(intern("define"), |_args| Err("define: must be used as a special form, not called".into()));
    m.insert(intern("error"), |args| Err(value_to_string(&args[0])));

    // ── Grid predicates ─────────────────────────────────────────────
    m.insert(intern("grid?"), |args| Ok(Value::Bool(matches!(&args[0], Value::Grid(_)))));

    // ── Grid construction & access ──────────────────────────────────
    m.insert(intern("grid-make"), |args| {
        let h = num(&args[0])? as usize;
        let w = num(&args[1])? as usize;
        let c = num(&args[2])? as i8;
        Ok(Value::Grid(vec![vec![c; w]; h]))
    });
    m.insert(intern("grid-from-list"), |args| {
        let outer = list(&args[0])?;
        let mut rows = Vec::with_capacity(outer.len());
        for row_val in &outer {
            let row_list = match row_val { Value::List(l) => l, _ => return Err("grid-from-list: expected list of lists".into()) };
            let row: Vec<i8> = row_list.iter().map(|v| match v { Value::Num(n) => Ok(*n as i8), _ => Err("grid-from-list: expected numbers".to_string()) }).collect::<Result<Vec<i8>, String>>()?;
            rows.push(row);
        }
        Ok(Value::Grid(rows))
    });
    m.insert(intern("grid-to-list"), |args| {
        let g = grid(&args[0])?;
        let rows: Vec<Value> = g.iter().map(|row| {
            Value::List(row.iter().map(|c| Value::Num(*c as f64)).collect())
        }).collect();
        Ok(Value::List(rows))
    });
    m.insert(intern("grid-width"), |args| {
        let g = grid(&args[0])?;
        Ok(Value::Num(g.first().map_or(0, |r| r.len()) as f64))
    });
    m.insert(intern("grid-height"), |args| {
        let g = grid(&args[0])?;
        Ok(Value::Num(g.len() as f64))
    });
    m.insert(intern("grid-size"), |args| {
        let g = grid(&args[0])?;
        let h = g.len() as f64;
        let w = g.first().map_or(0, |r| r.len()) as f64;
        Ok(Value::List(vec![Value::Num(h), Value::Num(w)]))
    });
    m.insert(intern("grid-get"), |args| {
        let g = grid(&args[0])?;
        let r = num(&args[1])? as usize;
        let c = num(&args[2])? as usize;
        if r < g.len() && c < g[r].len() { Ok(Value::Num(g[r][c] as f64)) }
        else { Err(format!("grid-get: index ({},{}) out of bounds ({}x{})", r, c, g.len(), g.first().map_or(0, |r| r.len()))) }
    });
    m.insert(intern("grid-set"), |args| {
        let mut g = grid(&args[0])?;
        let r = num(&args[1])? as usize;
        let c = num(&args[2])? as usize;
        let v = num(&args[3])? as i8;
        if r < g.len() && c < g[r].len() { g[r][c] = v; Ok(Value::Grid(g)) }
        else { Err("grid-set: index out of bounds".into()) }
    });
    m.insert(intern("grid-row"), |args| {
        let g = grid(&args[0])?;
        let r = num(&args[1])? as usize;
        if r < g.len() { Ok(Value::List(g[r].iter().map(|c| Value::Num(*c as f64)).collect())) }
        else { Err("grid-row: row out of bounds".into()) }
    });
    m.insert(intern("grid-col"), |args| {
        let g = grid(&args[0])?;
        let c = num(&args[1])? as usize;
        let col: Result<Vec<Value>, String> = g.iter().map(|row| {
            if c < row.len() { Ok(Value::Num(row[c] as f64)) }
            else { Err("grid-col: col out of bounds".into()) }
        }).collect();
        Ok(Value::List(col?))
    });

    // ── Grid transformations ────────────────────────────────────────
    m.insert(intern("grid-rotate-cw"), |args| {
        let g = grid(&args[0])?;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mut out = vec![vec![0i8; h]; w];
        for r in 0..h { for c in 0..w { out[c][h - 1 - r] = g[r][c]; } }
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-rotate-ccw"), |args| {
        let g = grid(&args[0])?;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mut out = vec![vec![0i8; h]; w];
        for r in 0..h { for c in 0..w { out[w - 1 - c][r] = g[r][c]; } }
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-rotate-180"), |args| {
        let g = grid(&args[0])?;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mut out = vec![vec![0i8; w]; h];
        for r in 0..h { for c in 0..w { out[h - 1 - r][w - 1 - c] = g[r][c]; } }
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-flip-h"), |args| {
        let g = grid(&args[0])?;
        let out: Vec<Vec<i8>> = g.iter().map(|row| { let mut r = row.clone(); r.reverse(); r }).collect();
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-flip-v"), |args| {
        let mut g = grid(&args[0])?;
        g.reverse();
        Ok(Value::Grid(g))
    });
    m.insert(intern("grid-transpose"), |args| {
        let g = grid(&args[0])?;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mut out = vec![vec![0i8; h]; w];
        for r in 0..h { for c in 0..w { out[c][r] = g[r][c]; } }
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-crop"), |args| {
        let g = grid(&args[0])?;
        let r1 = num(&args[1])? as usize;
        let c1 = num(&args[2])? as usize;
        let r2 = num(&args[3])? as usize;
        let c2 = num(&args[4])? as usize;
        let out: Vec<Vec<i8>> = g[r1..r2].iter().map(|row| row[c1..c2].to_vec()).collect();
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-overlay"), |args| {
        let mut base = grid(&args[0])?;
        let over = grid(&args[1])?;
        let dr = num(&args[2])? as usize;
        let dc = num(&args[3])? as usize;
        for r in 0..over.len() {
            for c in 0..over[r].len() {
                if over[r][c] != 0 {
                    let tr = dr + r; let tc = dc + c;
                    if tr < base.len() && tc < base[tr].len() { base[tr][tc] = over[r][c]; }
                }
            }
        }
        Ok(Value::Grid(base))
    });
    m.insert(intern("grid-tile"), |args| {
        let g = grid(&args[0])?;
        let nr = num(&args[1])? as usize;
        let nc = num(&args[2])? as usize;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mut out = vec![vec![0i8; w * nc]; h * nr];
        for tr in 0..nr { for tc in 0..nc { for r in 0..h { for c in 0..w { out[tr * h + r][tc * w + c] = g[r][c]; } } } }
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-scale"), |args| {
        let g = grid(&args[0])?;
        let s = num(&args[1])? as usize;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mut out = vec![vec![0i8; w * s]; h * s];
        for r in 0..h { for c in 0..w { for dr in 0..s { for dc in 0..s { out[r * s + dr][c * s + dc] = g[r][c]; } } } }
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-replace-color"), |args| {
        let g = grid(&args[0])?;
        let from = num(&args[1])? as i8;
        let to = num(&args[2])? as i8;
        let out: Vec<Vec<i8>> = g.iter().map(|row| row.iter().map(|&c| if c == from { to } else { c }).collect()).collect();
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-mask"), |args| {
        let g = grid(&args[0])?;
        let mask = grid(&args[1])?;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mut out = vec![vec![0i8; w]; h];
        for r in 0..h { for c in 0..w { out[r][c] = if r < mask.len() && c < mask[r].len() && mask[r][c] != 0 { g[r][c] } else { 0 }; } }
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-pad"), |args| {
        let g = grid(&args[0])?;
        let pad = num(&args[1])? as usize;
        let color = num(&args[2])? as i8;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let new_h = h + 2 * pad; let new_w = w + 2 * pad;
        let mut out = vec![vec![color; new_w]; new_h];
        for r in 0..h { for c in 0..w { out[r + pad][c + pad] = g[r][c]; } }
        Ok(Value::Grid(out))
    });

    // ── Grid analysis ───────────────────────────────────────────────
    m.insert(intern("grid-colors"), |args| {
        let g = grid(&args[0])?;
        let mut seen = [false; 10];
        for row in &g { for &c in row { if (c as usize) < 10 { seen[c as usize] = true; } } }
        let colors: Vec<Value> = seen.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| Value::Num(i as f64)).collect();
        Ok(Value::List(colors))
    });
    m.insert(intern("grid-count-color"), |args| {
        let g = grid(&args[0])?;
        let color = num(&args[1])? as i8;
        let count = g.iter().flat_map(|row| row.iter()).filter(|&&c| c == color).count();
        Ok(Value::Num(count as f64))
    });
    m.insert(intern("grid-most-common"), |args| {
        let g = grid(&args[0])?;
        Ok(Value::Num(detect_background(&g) as f64))
    });
    m.insert(intern("grid-background"), |args| {
        let g = grid(&args[0])?;
        Ok(Value::Num(detect_background(&g) as f64))
    });
    m.insert(intern("grid-equal"), |args| {
        let a = grid(&args[0])?;
        let b = grid(&args[1])?;
        Ok(Value::Bool(a == b))
    });
    m.insert(intern("grid-find-color"), |args| {
        let g = grid(&args[0])?;
        let color = num(&args[1])? as i8;
        let mut positions = Vec::new();
        for (r, row) in g.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v == color { positions.push(Value::List(vec![Value::Num(r as f64), Value::Num(c as f64)])); }
            }
        }
        Ok(Value::List(positions))
    });
    m.insert(intern("grid-symmetric-h"), |args| {
        let g = grid(&args[0])?;
        let sym = g.iter().all(|row| { let w = row.len(); (0..w / 2).all(|c| row[c] == row[w - 1 - c]) });
        Ok(Value::Bool(sym))
    });
    m.insert(intern("grid-symmetric-v"), |args| {
        let g = grid(&args[0])?;
        let h = g.len();
        let sym = (0..h / 2).all(|r| g[r] == g[h - 1 - r]);
        Ok(Value::Bool(sym))
    });
    m.insert(intern("grid-dimensions-equal"), |args| {
        let a = grid(&args[0])?;
        let b = grid(&args[1])?;
        let eq = a.len() == b.len() && a.first().map_or(0, |r| r.len()) == b.first().map_or(0, |r| r.len());
        Ok(Value::Bool(eq))
    });

    // ── Grid bounding box & trim ────────────────────────────────────
    m.insert(intern("grid-bounding-box"), |args| {
        let g = grid(&args[0])?;
        let bg = detect_background(&g);
        let mut min_r = g.len(); let mut min_c = g.first().map_or(0, |r| r.len());
        let mut max_r = 0usize; let mut max_c = 0usize;
        for (r, row) in g.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v != bg { min_r = min_r.min(r); min_c = min_c.min(c); max_r = max_r.max(r); max_c = max_c.max(c); }
            }
        }
        if max_r < min_r { Ok(Value::List(vec![Value::Num(0.0), Value::Num(0.0), Value::Num(0.0), Value::Num(0.0)])) }
        else { Ok(Value::List(vec![Value::Num(min_r as f64), Value::Num(min_c as f64), Value::Num((max_r + 1) as f64), Value::Num((max_c + 1) as f64)])) }
    });
    m.insert(intern("grid-trim"), |args| {
        let g = grid(&args[0])?;
        let bg = detect_background(&g);
        let mut min_r = g.len(); let mut min_c = g.first().map_or(0, |r| r.len());
        let mut max_r = 0usize; let mut max_c = 0usize;
        for (r, row) in g.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v != bg { min_r = min_r.min(r); min_c = min_c.min(c); max_r = max_r.max(r); max_c = max_c.max(c); }
            }
        }
        if max_r < min_r || min_c > max_c { Ok(Value::Grid(vec![])) }
        else if max_r >= g.len() || max_c >= g[0].len() { Ok(Value::Grid(vec![])) }
        else {
            let out: Vec<Vec<i8>> = g[min_r..=max_r].iter().map(|row| {
                if max_c < row.len() { row[min_c..=max_c].to_vec() } else { vec![] }
            }).collect();
            Ok(Value::Grid(out))
        }
    });

    // ── Grid object detection (connected components) ────────────────
    m.insert(intern("grid-objects"), |args| {
        let g = grid(&args[0])?;
        if g.is_empty() { return Ok(Value::List(vec![])); }
        let objects = grid_connected_components(&g, false);
        Ok(Value::List(objects.into_iter().map(Value::Grid).collect()))
    });
    m.insert(intern("grid-objects-8"), |args| {
        let g = grid(&args[0])?;
        if g.is_empty() { return Ok(Value::List(vec![])); }
        let objects = grid_connected_components(&g, true);
        Ok(Value::List(objects.into_iter().map(Value::Grid).collect()))
    });
    m.insert(intern("grid-object-count"), |args| {
        let g = grid(&args[0])?;
        let count = grid_connected_components(&g, false).len();
        Ok(Value::Num(count as f64))
    });
    m.insert(intern("grid-object-colors"), |args| {
        // Extract single-color masks (one grid per non-background color)
        let g = grid(&args[0])?;
        let bg = detect_background(&g);
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mut color_grids: Vec<Value> = Vec::new();
        for color in 0..10i8 {
            if color == bg { continue; }
            let has_color = g.iter().any(|row| row.contains(&color));
            if has_color {
                let mask: Vec<Vec<i8>> = g.iter().map(|row| row.iter().map(|&c| if c == color { color } else { 0 }).collect()).collect();
                color_grids.push(Value::Grid(mask));
            }
        }
        let _ = (h, w); // suppress unused
        Ok(Value::List(color_grids))
    });

    // ── Grid composition ────────────────────────────────────────────
    m.insert(intern("grid-hconcat"), |args| {
        let a = grid(&args[0])?;
        let b = grid(&args[1])?;
        let h = a.len().max(b.len());
        let wa = a.first().map_or(0, |r| r.len());
        let wb = b.first().map_or(0, |r| r.len());
        let mut out = vec![vec![0i8; wa + wb]; h];
        for r in 0..h {
            if r < a.len() { for c in 0..wa { out[r][c] = a[r][c]; } }
            if r < b.len() { for c in 0..wb { out[r][wa + c] = b[r][c]; } }
        }
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-vconcat"), |args| {
        let a = grid(&args[0])?;
        let b = grid(&args[1])?;
        let mut out = a;
        out.extend(b);
        Ok(Value::Grid(out))
    });
    m.insert(intern("grid-hsplit"), |args| {
        let g = grid(&args[0])?;
        let n = num(&args[1])? as usize;
        let w = g.first().map_or(0, |r| r.len());
        let chunk_w = w / n;
        let parts: Vec<Value> = (0..n).map(|i| {
            let start = i * chunk_w;
            let end = if i == n - 1 { w } else { start + chunk_w };
            Value::Grid(g.iter().map(|row| row[start..end].to_vec()).collect())
        }).collect();
        Ok(Value::List(parts))
    });
    m.insert(intern("grid-vsplit"), |args| {
        let g = grid(&args[0])?;
        let n = num(&args[1])? as usize;
        let h = g.len();
        let chunk_h = h / n;
        let parts: Vec<Value> = (0..n).map(|i| {
            let start = i * chunk_h;
            let end = if i == n - 1 { h } else { start + chunk_h };
            Value::Grid(g[start..end].to_vec())
        }).collect();
        Ok(Value::List(parts))
    });
    m.insert(intern("grid-quarter"), |args| {
        let g = grid(&args[0])?;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let mh = h / 2; let mw = w / 2;
        let tl: Vec<Vec<i8>> = g[..mh].iter().map(|r| r[..mw].to_vec()).collect();
        let tr: Vec<Vec<i8>> = g[..mh].iter().map(|r| r[mw..].to_vec()).collect();
        let bl: Vec<Vec<i8>> = g[mh..].iter().map(|r| r[..mw].to_vec()).collect();
        let br: Vec<Vec<i8>> = g[mh..].iter().map(|r| r[mw..].to_vec()).collect();
        Ok(Value::List(vec![Value::Grid(tl), Value::Grid(tr), Value::Grid(bl), Value::Grid(br)]))
    });

    // ── Tier 1: Perceptual primitives ───────────────────────────────

    m.insert(intern("grid-flood-fill"), |args| {
        let mut g = grid(&args[0])?;
        let r = num(&args[1])? as usize;
        let c = num(&args[2])? as usize;
        let fill = num(&args[3])? as i8;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        if r >= h || c >= w { return Ok(Value::Grid(g)); }
        let orig = g[r][c];
        if orig == fill { return Ok(Value::Grid(g)); }
        let mut queue = vec![(r, c)];
        g[r][c] = fill;
        while let Some((cr, cc)) = queue.pop() {
            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nr = cr as i32 + dr; let nc = cc as i32 + dc;
                if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                    let (nr, nc) = (nr as usize, nc as usize);
                    if g[nr][nc] == orig { g[nr][nc] = fill; queue.push((nr, nc)); }
                }
            }
        }
        Ok(Value::Grid(g))
    });

    m.insert(intern("grid-fill-enclosed"), |args| {
        let g = grid(&args[0])?;
        let fill = num(&args[1])? as i8;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        if h == 0 || w == 0 { return Ok(Value::Grid(g)); }
        // Use 0 as background (ARC convention), not detect_background
        // which can be wrong for grids where the "wall" is the majority color
        let bg = 0i8;
        // Flood-fill background from all border cells to find "exterior"
        let mut exterior = vec![vec![false; w]; h];
        let mut queue: Vec<(usize, usize)> = Vec::new();
        for r in 0..h { for c in 0..w {
            if (r == 0 || r == h - 1 || c == 0 || c == w - 1) && g[r][c] == bg {
                exterior[r][c] = true; queue.push((r, c));
            }
        }}
        while let Some((cr, cc)) = queue.pop() {
            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nr = cr as i32 + dr; let nc = cc as i32 + dc;
                if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                    let (nr, nc) = (nr as usize, nc as usize);
                    if !exterior[nr][nc] && g[nr][nc] == bg { exterior[nr][nc] = true; queue.push((nr, nc)); }
                }
            }
        }
        // Fill all non-exterior background cells
        let out: Vec<Vec<i8>> = g.iter().enumerate().map(|(r, row)| {
            row.iter().enumerate().map(|(c, &v)| {
                if v == bg && !exterior[r][c] { fill } else { v }
            }).collect()
        }).collect();
        Ok(Value::Grid(out))
    });

    m.insert(intern("grid-draw-line-h"), |args| {
        let mut g = grid(&args[0])?;
        let r = num(&args[1])? as usize;
        let c1 = num(&args[2])? as usize;
        let c2 = num(&args[3])? as usize;
        let color = num(&args[4])? as i8;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        if r < h {
            for c in c1.min(c2)..=c1.max(c2) { if c < w { g[r][c] = color; } }
        }
        Ok(Value::Grid(g))
    });

    m.insert(intern("grid-draw-line-v"), |args| {
        let mut g = grid(&args[0])?;
        let c = num(&args[1])? as usize;
        let r1 = num(&args[2])? as usize;
        let r2 = num(&args[3])? as usize;
        let color = num(&args[4])? as i8;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        if c < w {
            for r in r1.min(r2)..=r1.max(r2) { if r < h { g[r][c] = color; } }
        }
        Ok(Value::Grid(g))
    });

    m.insert(intern("grid-ray"), |args| {
        let mut g = grid(&args[0])?;
        let mut r = num(&args[1])? as i32;
        let mut c = num(&args[2])? as i32;
        let dr = num(&args[3])? as i32;
        let dc = num(&args[4])? as i32;
        let color = num(&args[5])? as i8;
        let h = g.len() as i32; let w = g.first().map_or(0, |r| r.len()) as i32;
        while r >= 0 && r < h && c >= 0 && c < w {
            g[r as usize][c as usize] = color;
            r += dr; c += dc;
        }
        Ok(Value::Grid(g))
    });

    m.insert(intern("grid-gravity"), |args| {
        let g = grid(&args[0])?;
        let dir = num(&args[1])? as i32;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        if h == 0 || w == 0 { return Ok(Value::Grid(g)); }
        let bg = detect_background(&g);
        let mut out = vec![vec![bg; w]; h];
        match dir {
            0 => { // down
                for c in 0..w {
                    let mut write = h;
                    for r in (0..h).rev() { if g[r][c] != bg { write -= 1; out[write][c] = g[r][c]; } }
                }
            }
            1 => { // up
                for c in 0..w {
                    let mut write = 0;
                    for r in 0..h { if g[r][c] != bg { out[write][c] = g[r][c]; write += 1; } }
                }
            }
            2 => { // left
                for r in 0..h {
                    let mut write = 0;
                    for c in 0..w { if g[r][c] != bg { out[r][write] = g[r][c]; write += 1; } }
                }
            }
            3 => { // right
                for r in 0..h {
                    let mut write = w;
                    for c in (0..w).rev() { if g[r][c] != bg { write -= 1; out[r][write] = g[r][c]; } }
                }
            }
            _ => return Ok(Value::Grid(g)),
        }
        Ok(Value::Grid(out))
    });

    m.insert(intern("grid-xor"), |args| {
        let a = grid(&args[0])?; let b = grid(&args[1])?;
        let bg_a = detect_background(&a); let bg_b = detect_background(&b);
        let h = a.len().max(b.len()); let w = a.first().map_or(0, |r| r.len()).max(b.first().map_or(0, |r| r.len()));
        let mut out = vec![vec![0i8; w]; h];
        for r in 0..h { for c in 0..w {
            let va = if r < a.len() && c < a[r].len() { a[r][c] } else { bg_a };
            let vb = if r < b.len() && c < b[r].len() { b[r][c] } else { bg_b };
            let a_fg = va != bg_a; let b_fg = vb != bg_b;
            out[r][c] = if a_fg && !b_fg { va } else if !a_fg && b_fg { vb } else { 0 };
        }}
        Ok(Value::Grid(out))
    });

    m.insert(intern("grid-and"), |args| {
        let a = grid(&args[0])?; let b = grid(&args[1])?;
        let bg_a = detect_background(&a); let bg_b = detect_background(&b);
        let h = a.len().max(b.len()); let w = a.first().map_or(0, |r| r.len()).max(b.first().map_or(0, |r| r.len()));
        let mut out = vec![vec![0i8; w]; h];
        for r in 0..h { for c in 0..w {
            let va = if r < a.len() && c < a[r].len() { a[r][c] } else { bg_a };
            let vb = if r < b.len() && c < b[r].len() { b[r][c] } else { bg_b };
            out[r][c] = if va != bg_a && vb != bg_b { va } else { 0 };
        }}
        Ok(Value::Grid(out))
    });

    m.insert(intern("grid-or"), |args| {
        let a = grid(&args[0])?; let b = grid(&args[1])?;
        let bg_a = detect_background(&a); let bg_b = detect_background(&b);
        let h = a.len().max(b.len()); let w = a.first().map_or(0, |r| r.len()).max(b.first().map_or(0, |r| r.len()));
        let mut out = vec![vec![0i8; w]; h];
        for r in 0..h { for c in 0..w {
            let va = if r < a.len() && c < a[r].len() { a[r][c] } else { bg_a };
            let vb = if r < b.len() && c < b[r].len() { b[r][c] } else { bg_b };
            out[r][c] = if va != bg_a { va } else if vb != bg_b { vb } else { 0 };
        }}
        Ok(Value::Grid(out))
    });

    // ── Tier 2: Shape analysis ──────────────────────────────────────

    m.insert(intern("grid-object-area"), |args| {
        let g = grid(&args[0])?;
        let bg = detect_background(&g);
        let count = g.iter().flat_map(|row| row.iter()).filter(|&&c| c != bg).count();
        Ok(Value::Num(count as f64))
    });

    m.insert(intern("grid-object-center"), |args| {
        let g = grid(&args[0])?;
        let bg = detect_background(&g);
        let mut sum_r = 0f64; let mut sum_c = 0f64; let mut count = 0f64;
        for (r, row) in g.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v != bg { sum_r += r as f64; sum_c += c as f64; count += 1.0; }
            }
        }
        if count == 0.0 { Ok(Value::List(vec![Value::Num(0.0), Value::Num(0.0)])) }
        else { Ok(Value::List(vec![Value::Num((sum_r / count).floor()), Value::Num((sum_c / count).floor())])) }
    });

    m.insert(intern("grid-is-rectangle"), |args| {
        let g = grid(&args[0])?;
        let bg = detect_background(&g);
        let mut min_r = g.len(); let mut min_c = g.first().map_or(0, |r| r.len());
        let mut max_r = 0usize; let mut max_c = 0usize; let mut fg_count = 0usize;
        for (r, row) in g.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v != bg { min_r = min_r.min(r); min_c = min_c.min(c); max_r = max_r.max(r); max_c = max_c.max(c); fg_count += 1; }
            }
        }
        if fg_count == 0 { return Ok(Value::Bool(false)); }
        let expected = (max_r - min_r + 1) * (max_c - min_c + 1);
        Ok(Value::Bool(fg_count == expected))
    });

    m.insert(intern("grid-detect-rectangles"), |args| {
        let g = grid(&args[0])?;
        if g.is_empty() { return Ok(Value::List(vec![])); }
        let objects = grid_connected_components(&g, false);
        let rects: Vec<Value> = objects.into_iter().filter(|obj| {
            let bg = 0i8; // objects are already trimmed with bg=0
            let mut fg = 0usize; let mut area = 0usize;
            let h = obj.len(); let w = obj.first().map_or(0, |r| r.len());
            for row in obj { for &c in row { if c != bg { fg += 1; } } }
            area = h * w;
            fg > 0 && fg == area
        }).map(Value::Grid).collect();
        Ok(Value::List(rects))
    });

    m.insert(intern("grid-objects-touching"), |args| {
        let a = grid(&args[0])?; let b = grid(&args[1])?;
        let bg_a = detect_background(&a); let bg_b = detect_background(&b);
        let h = a.len().max(b.len()); let w = a.first().map_or(0, |r| r.len()).max(b.first().map_or(0, |r| r.len()));
        for r in 0..h { for c in 0..w {
            let va = if r < a.len() && c < a[r].len() { a[r][c] } else { bg_a };
            if va == bg_a { continue; }
            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nr = r as i32 + dr; let nc = c as i32 + dc;
                if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                    let (nr, nc) = (nr as usize, nc as usize);
                    let vb = if nr < b.len() && nc < b[nr].len() { b[nr][nc] } else { bg_b };
                    if vb != bg_b { return Ok(Value::Bool(true)); }
                }
            }
        }}
        Ok(Value::Bool(false))
    });

    m.insert(intern("grid-overlay-center"), |args| {
        let mut base = grid(&args[0])?;
        let over = grid(&args[1])?;
        let bh = base.len(); let bw = base.first().map_or(0, |r| r.len());
        let oh = over.len(); let ow = over.first().map_or(0, |r| r.len());
        let dr = (bh as i32 - oh as i32) / 2; let dc = (bw as i32 - ow as i32) / 2;
        for r in 0..oh { for c in 0..ow {
            if over[r][c] != 0 {
                let tr = dr + r as i32; let tc = dc + c as i32;
                if tr >= 0 && tr < bh as i32 && tc >= 0 && tc < bw as i32 {
                    base[tr as usize][tc as usize] = over[r][c];
                }
            }
        }}
        Ok(Value::Grid(base))
    });

    // ── Tier 3: Advanced primitives ─────────────────────────────────

    m.insert(intern("grid-find-subgrid"), |args| {
        let g = grid(&args[0])?; let pat = grid(&args[1])?;
        let gh = g.len(); let gw = g.first().map_or(0, |r| r.len());
        let ph = pat.len(); let pw = pat.first().map_or(0, |r| r.len());
        if ph == 0 || pw == 0 || ph > gh || pw > gw { return Ok(Value::List(vec![])); }
        let pat_bg = detect_background(&pat);
        let mut positions = Vec::new();
        for r in 0..=(gh - ph) {
            for c in 0..=(gw - pw) {
                let mut matches = true;
                'check: for pr in 0..ph { for pc in 0..pw {
                    if pat[pr][pc] != pat_bg && g[r + pr][c + pc] != pat[pr][pc] { matches = false; break 'check; }
                }}
                if matches { positions.push(Value::List(vec![Value::Num(r as f64), Value::Num(c as f64)])); }
            }
        }
        Ok(Value::List(positions))
    });

    m.insert(intern("grid-neighbor-count"), |args| {
        let g = grid(&args[0])?;
        let r = num(&args[1])? as i32;
        let c = num(&args[2])? as i32;
        let h = g.len() as i32; let w = g.first().map_or(0, |r| r.len()) as i32;
        let bg = detect_background(&g);
        let mut count = 0;
        for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r + dr; let nc = c + dc;
            if nr >= 0 && nr < h && nc >= 0 && nc < w && g[nr as usize][nc as usize] != bg { count += 1; }
        }
        Ok(Value::Num(count as f64))
    });

    m.insert(intern("grid-border"), |args| {
        let g = grid(&args[0])?;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        let bg = detect_background(&g);
        let mut out = vec![vec![bg; w]; h];
        for r in 0..h { for c in 0..w {
            if g[r][c] != bg {
                let has_bg_neighbor = [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)].iter().any(|(dr, dc)| {
                    let nr = r as i32 + dr; let nc = c as i32 + dc;
                    nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 || g[nr as usize][nc as usize] == bg
                });
                if has_bg_neighbor { out[r][c] = g[r][c]; }
            }
        }}
        Ok(Value::Grid(out))
    });

    m.insert(intern("grid-fill-rect"), |args| {
        let mut g = grid(&args[0])?;
        let r1 = num(&args[1])? as usize;
        let c1 = num(&args[2])? as usize;
        let r2 = num(&args[3])? as usize;
        let c2 = num(&args[4])? as usize;
        let color = num(&args[5])? as i8;
        let h = g.len(); let w = g.first().map_or(0, |r| r.len());
        for r in r1.min(r2)..=r1.max(r2) { for c in c1.min(c2)..=c1.max(c2) {
            if r < h && c < w { g[r][c] = color; }
        }}
        Ok(Value::Grid(g))
    });

    m
}

thread_local! {
    static BUILTIN_DISPATCH: std::collections::HashMap<Sym, BuiltinFn> = build_dispatch_table();
}

/// Convert a Value to its SELPH source representation.
fn value_to_source(v: &Value) -> String {
    match v {
        Value::Num(n) => {
            if *n == n.floor() && n.abs() < 1e15 { format!("{}", *n as i64) }
            else { format!("{}", n) }
        }
        Value::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::Nil => "nil".into(),
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(|i| value_to_source(i)).collect();
            format!("(list {})", inner.join(" "))
        }
        _ => format!("{:?}", v),
    }
}

/// Extract macro tuples and data bindings from a library value.
/// Handles: namespace of functions (recursive for tree structure),
/// source strings, and lists of source strings.
/// Data namespaces (non-function values) are collected as trees for extra_bindings.
fn extract_library_macros(
    val: &Value,
    prefix: &str,
    macros_out: &mut Vec<(String, Vec<String>, Vec<Node>, usize)>,
    data_out: &mut Vec<(String, Value)>,
) {
    match val {
        Value::Namespace(map) => {
            for (key, v) in map {
                let child_name = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}/{}", prefix, key)
                };
                match v {
                    Value::RustMacro(params, nodes, root) => {
                        macros_out.push((
                            child_name,
                            params.iter().map(|s| resolve(*s)).collect(),
                            nodes.as_ref().to_vec(),
                            *root,
                        ));
                    }
                    Value::Namespace(sub) => {
                        // If sub-namespace has any functions, recurse for macros
                        let has_functions = sub.values().any(|sv|
                            matches!(sv, Value::RustMacro(..) | Value::Closure(..)));
                        if has_functions {
                            extract_library_macros(v, &child_name, macros_out, data_out);
                        } else {
                            // Pure data namespace — register as a named binding
                            data_out.push((child_name, v.clone()));
                        }
                    }
                    _ => {} // skip non-function, non-namespace values at this level
                }
            }
        }
        Value::Str(src) => {
            // Parse source string for defmacro forms
            if let Ok((nodes_vec, roots)) = crate::parser::parse_file(src) {
                for &root in &roots {
                    if let Node::App(children) = &nodes_vec[root] {
                        if children.len() == 4 {
                            if let Node::Symbol(mname) = &nodes_vec[children[1]] {
                                if let Node::App(param_indices) = &nodes_vec[children[2]] {
                                    let params: Vec<String> = param_indices.iter()
                                        .filter_map(|&i| {
                                            if let Node::Symbol(s) = &nodes_vec[i] { Some(resolve(*s)) }
                                            else { None }
                                        }).collect();
                                    macros_out.push((resolve(*mname), params, nodes_vec.clone(), children[3]));
                                }
                            }
                        }
                    }
                }
            }
        }
        Value::List(items) => {
            for item in items {
                extract_library_macros(item, prefix, macros_out, data_out);
            }
        }
        _ => {}
    }
}

pub fn apply_builtin(name: Sym, args: &[Value]) -> Result<Value, String> {
    // Fast path: direct fn pointer dispatch (O(1) hash lookup, no string allocation)
    if let Some(result) = BUILTIN_DISPATCH.with(|d| d.get(&name).map(|f| f(args))) {
        return result;
    }
    // Slow path: builtins that call back into eval/apply
    apply_builtin_slow(name, args)
}

fn apply_builtin_slow(name: Sym, args: &[Value]) -> Result<Value, String> {
    let name_str = resolve(name);
    match name_str.as_str() {
        "map" => {
            let l = list(&args[1])?;
            let empty = empty_nodes();
            let mut env = make_default_env();
            let mut results = Vec::new();
            for item in &l {
                results.push(apply(&args[0], &[item.clone()], &empty, &mut env)?);
            }
            Ok(Value::List(results))
        }
        "reduce" => {
            let l = list(&args[1])?;
            if l.is_empty() && args.len() <= 2 {
                return Err("reduce: empty list with no initial value".into());
            }
            let empty = empty_nodes();
            let mut env = make_default_env();
            let mut acc = if args.len() > 2 { args[2].clone() } else { l[0].clone() };
            let items = if args.len() > 2 { &l[..] } else { &l[1..] };
            for item in items {
                acc = apply(&args[0], &[acc, item.clone()], &empty, &mut env)?;
            }
            Ok(acc)
        }
        "filter" => {
            let l = list(&args[1])?;
            let empty = empty_nodes();
            let mut env = make_default_env();
            let mut results = Vec::new();
            for item in &l {
                if let Value::Bool(true) = apply(&args[0], &[item.clone()], &empty, &mut env)? {
                    results.push(item.clone());
                }
            }
            Ok(Value::List(results))
        }
        "apply" => {
            let func = &args[0];
            let arg_list = list(&args[1])?;
            let empty = empty_nodes();
            let mut env = make_default_env();
            apply(func, &arg_list, &empty, &mut env)
        }
        "eval-source" => {
            let src = string(&args[0])?;
            let (nodes, roots) = crate::parser::parse_file(&src)
                .map_err(|e| format!("eval-source: parse error: {}", e))?;
            if roots.is_empty() { return Ok(Value::Nil); }
            let nodes_rc: Rc<[Node]> = nodes.into();
            let mut env = make_default_env();
            let mut last = Value::Nil;
            for &r in &roots { last = eval(&nodes_rc, r, &mut env)?; }
            Ok(last)
        }
        "eval-in" => {
            let src = string(&args[0])?;
            let (nodes, roots) = crate::parser::parse_file(&src)
                .map_err(|e| format!("eval-in: parse error: {}", e))?;
            if roots.is_empty() { return Ok(Value::Nil); }
            let nodes_rc: Rc<[Node]> = nodes.into();
            let mut env = make_default_env();
            let mut last = Value::Nil;
            for &r in &roots { last = eval(&nodes_rc, r, &mut env)?; }
            Ok(last)
        }
        "test-spec" => {
            // (test-spec candidate spec)
            // candidate: a function (lambda)
            // spec: list of (input expected) pairs
            // Returns: match fraction (0.0 to 1.0)
            if args.len() != 2 {
                return Err("test-spec: expected 2 arguments (candidate, spec)".into());
            }
            let candidate = &args[0];
            let pairs = match &args[1] {
                Value::List(l) => l,
                _ => return Err("test-spec: spec must be a list of (input expected) pairs".into()),
            };
            if pairs.is_empty() {
                return Ok(Value::Num(0.0));
            }
            let empty = empty_nodes();
            let mut matches = 0usize;
            let total = pairs.len();
            for pair in pairs {
                let (input, expected_val) = match pair {
                    Value::List(p) if p.len() == 2 => (&p[0], &p[1]),
                    _ => return Err("test-spec: each spec entry must be (input expected)".into()),
                };
                let mut env = make_default_env();
                match apply(candidate, &[input.clone()], &empty, &mut env) {
                    Ok(result) => {
                        if crate::synth::vals_equal(&result, expected_val) {
                            matches += 1;
                        }
                    }
                    Err(_) => {} // eval error = no match
                }
            }
            Ok(Value::Num(matches as f64 / total as f64))
        }
        "memorize" => {
            // (memorize spec)
            // spec: list of (input expected) pairs
            // Returns: a namespace mapping inputs to outputs (data as library)
            // Returns nil if inputs are not all strings
            if args.len() != 1 {
                return Err("memorize: expected 1 argument (spec)".into());
            }
            let pairs = match &args[0] {
                Value::List(l) => l,
                _ => return Err("memorize: argument must be a list of (input expected) pairs".into()),
            };
            if pairs.is_empty() {
                return Ok(Value::Nil);
            }
            let mut map = std::collections::HashMap::new();
            for pair in pairs {
                match pair {
                    Value::List(p) if p.len() == 2 => {
                        if let Value::Str(key) = &p[0] {
                            map.insert(key.clone(), p[1].clone());
                        } else {
                            return Ok(Value::Nil); // non-string input, can't memorize
                        }
                    }
                    _ => return Err("memorize: each spec entry must be (input expected)".into()),
                }
            }
            Ok(Value::Namespace(map))
        }
        "synthesize" => {
            if args.len() != 1 {
                return Err("synthesize: expected 1 argument (namespace)".into());
            }
            let ns = match &args[0] {
                Value::Namespace(m) => m,
                _ => return Err("synthesize: argument must be a namespace".into()),
            };
            // Extract spec: list of [input, output] pairs
            let spec_val = ns.get("spec").ok_or("synthesize: namespace must have \"spec\" field")?;
            let pairs = match spec_val {
                Value::List(l) => l,
                _ => return Err("synthesize: \"spec\" must be a list of example pairs".into()),
            };
            let mut inputs = Vec::new();
            let mut expected = Vec::new();
            for pair in pairs {
                match pair {
                    Value::List(p) if p.len() == 2 => {
                        inputs.push(p[0].clone());
                        expected.push(p[1].clone());
                    }
                    _ => return Err("synthesize: each spec entry must be a list of [input, output]".into()),
                }
            }
            let max_depth = ns.get("max-depth")
                .and_then(|v| if let Value::Num(n) = v { Some(*n as usize) } else { None })
                .unwrap_or(2);
            let max_candidates = ns.get("max-candidates")
                .and_then(|v| if let Value::Num(n) = v { Some(*n as usize) } else { None })
                .unwrap_or(10000);

            // Extract library macros and data from "library" field.
            // Accepts: namespace of functions, source string, or list of source strings.
            // Data namespaces become extra_bindings (available as depth-0 atoms in synthesis).
            let mut macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
            let mut lib_data: Vec<(String, Value)> = Vec::new();
            if let Some(lib_val) = ns.get("library") {
                extract_library_macros(lib_val, "", &mut macros, &mut lib_data);
            }

            // Extract optional trees (legacy) and merge with library data
            let mut trees = extract_trees_from_ns(ns);
            trees.extend(lib_data);
            let (components, extra_bindings) = crate::synth::default_synth_components_with_trees(&macros, &trees);
            let sr = crate::synth::synthesize_with_validation(
                &components, &inputs, &expected, &macros,
                max_depth, max_candidates, true, None, &extra_bindings,
            );
            let source = if sr.found {
                node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap())
            } else {
                String::new()
            };
            let mut result = std::collections::HashMap::new();
            result.insert("found".to_string(), Value::Bool(sr.found));
            result.insert("candidates".to_string(), Value::Num(sr.candidates_explored as f64));
            result.insert("source".to_string(), Value::Str(source));
            Ok(Value::Namespace(result))
        }
        "synthesize-optimize" => {
            if args.len() != 1 {
                return Err("synthesize-optimize: expected 1 argument (namespace)".into());
            }
            let ns = match &args[0] {
                Value::Namespace(m) => m,
                _ => return Err("synthesize-optimize: argument must be a namespace".into()),
            };

            // Determine direction
            let direction = if ns.contains_key("minimize") {
                crate::synth::OptDirection::Minimize
            } else if ns.contains_key("maximize") {
                crate::synth::OptDirection::Maximize
            } else {
                return Err("synthesize-optimize: namespace must have \"minimize\" or \"maximize\" field".into());
            };

            // Get the fitness function source
            let fitness_key = if direction == crate::synth::OptDirection::Minimize { "minimize" } else { "maximize" };
            let fitness_src = match ns.get(fitness_key) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(format!("synthesize-optimize: \"{}\" field must be a string (source code)", fitness_key)),
            };

            let (fitness_nodes, fitness_root) = match crate::parser::parse_source(&fitness_src) {
                Ok(r) => r,
                Err(e) => return Err(format!("synthesize-optimize: error parsing fitness function: {}", e)),
            };

            // Optional base examples (constraints)
            let mut base_inputs = Vec::new();
            let mut base_expected = Vec::new();
            if let Some(Value::List(pairs)) = ns.get("spec") {
                for pair in pairs {
                    if let Value::List(p) = pair {
                        if p.len() == 2 {
                            base_inputs.push(p[0].clone());
                            base_expected.push(p[1].clone());
                        }
                    }
                }
            }

            let max_depth = ns.get("max-depth")
                .and_then(|v| if let Value::Num(n) = v { Some(*n as usize) } else { None })
                .unwrap_or(2);
            let max_candidates = ns.get("max-candidates")
                .and_then(|v| if let Value::Num(n) = v { Some(*n as usize) } else { None })
                .unwrap_or(10000);

            let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();

            // Extract optional trees
            let trees = extract_trees_from_ns(ns);
            let (components, extra_bindings) = crate::synth::default_synth_components_with_trees(&macros, &trees);
            let osr = crate::synth::synthesize_optimize(
                &components,
                direction,
                &fitness_nodes,
                fitness_root,
                &base_inputs,
                &base_expected,
                &macros,
                max_depth,
                max_candidates,
                &extra_bindings,
            );
            let source = if osr.found {
                node_to_source(osr.nodes.as_ref().unwrap(), osr.root.unwrap())
            } else {
                String::new()
            };
            let mut result = std::collections::HashMap::new();
            result.insert("found".to_string(), Value::Bool(osr.found));
            result.insert("candidates".to_string(), Value::Num(osr.candidates_explored as f64));
            result.insert("source".to_string(), Value::Str(source));
            if let Some(score) = osr.fitness_score {
                result.insert("fitness".to_string(), Value::Num(score));
            }
            Ok(Value::Namespace(result))
        }
        _ => Err(format!("unknown builtin: {}", name_str)),
    }
}

fn num(v: &Value) -> Result<f64, String> {
    match v { Value::Num(n) => Ok(*n), _ => Err(format!("expected number, got {:?}", v)) }
}

fn nums(args: &[Value]) -> Result<(f64, f64), String> {
    if args.len() < 2 { return Err("expected 2 arguments".into()); }
    Ok((num(&args[0])?, num(&args[1])?))
}

fn num2(args: &[Value], op: fn(f64, f64) -> f64) -> Result<Value, String> {
    let (a, b) = nums(args)?;
    Ok(Value::Num(op(a, b)))
}

fn string(v: &Value) -> Result<String, String> {
    match v { Value::Str(s) => Ok(s.clone()), _ => Err(format!("expected string, got {:?}", v)) }
}

fn list(v: &Value) -> Result<Vec<Value>, String> {
    match v { Value::List(l) => Ok(l.clone()), _ => Err(format!("expected list, got {:?}", v)) }
}

fn grid(v: &Value) -> Result<Vec<Vec<i8>>, String> {
    match v { Value::Grid(g) => Ok(g.clone()), _ => Err(format!("expected grid, got {:?}", v)) }
}

/// Detect the background color of a grid (most common color).
fn detect_background(g: &[Vec<i8>]) -> i8 {
    let mut counts = [0u32; 10];
    for row in g { for &c in row { if (c as usize) < 10 { counts[c as usize] += 1; } } }
    counts.iter().enumerate().max_by_key(|(_, n)| *n).map(|(i, _)| i as i8).unwrap_or(0)
}

/// Extract connected components from a grid, ignoring the background color.
/// Each component is returned as a bounding-box-cropped grid with background=0.
/// If `eight_connected` is true, uses 8-connectivity; otherwise 4-connectivity.
fn grid_connected_components(g: &[Vec<i8>], eight_connected: bool) -> Vec<Vec<Vec<i8>>> {
    let h = g.len();
    let w = g.first().map_or(0, |r| r.len());
    if h == 0 || w == 0 { return vec![]; }

    let bg = detect_background(g);

    let mut labels = vec![vec![0u32; w]; h];
    let mut next_label = 1u32;
    let dirs4: &[(i32, i32)] = &[(-1, 0), (1, 0), (0, -1), (0, 1)];
    let dirs8: &[(i32, i32)] = &[(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
    let dirs = if eight_connected { dirs8 } else { dirs4 };

    // BFS flood fill
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

    // Extract each component as a trimmed grid
    let num_components = (next_label - 1) as usize;
    let mut results = Vec::with_capacity(num_components);
    for lbl in 1..next_label {
        let mut min_r = h; let mut min_c = w;
        let mut max_r = 0usize; let mut max_c = 0usize;
        for r in 0..h { for c in 0..w {
            if labels[r][c] == lbl { min_r = min_r.min(r); min_c = min_c.min(c); max_r = max_r.max(r); max_c = max_c.max(c); }
        }}
        if max_r >= min_r {
            let mut comp = vec![vec![0i8; max_c - min_c + 1]; max_r - min_r + 1];
            for r in min_r..=max_r { for c in min_c..=max_c {
                if labels[r][c] == lbl { comp[r - min_r][c - min_c] = g[r][c]; }
            }}
            results.push(comp);
        }
    }
    results
}

pub fn value_to_string(v: &Value) -> String {
    match v {
        Value::Num(n) => {
            if *n == (*n as i64) as f64 { format!("{}", *n as i64) }
            else { format!("{}", n) }
        }
        Value::Str(s) => s.clone(),
        Value::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
        Value::List(l) => {
            let parts: Vec<String> = l.iter().map(|v| value_to_string(v)).collect();
            format!("({})", parts.join(" "))
        }
        Value::Nil => "nil".to_string(),
        Value::Closure(params, _, _, _, _) => format!("<lambda ({})>", params.iter().map(|p| resolve(*p)).collect::<Vec<_>>().join(" ")),
        Value::Builtin(name) => format!("<builtin {}>", resolve(*name)),
        Value::RustMacro(params, _, _) => format!("<macro ({})>", params.iter().map(|p| resolve(*p)).collect::<Vec<_>>().join(" ")),
        Value::Grid(rows) => {
            let row_strs: Vec<String> = rows.iter().map(|row| {
                let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
                format!("({})", cells.join(" "))
            }).collect();
            format!("(#grid ({}))", row_strs.join(" "))
        }
        Value::Namespace(map) => {
            let keys: Vec<&String> = map.keys().collect();
            format!("<namespace {:?}>", keys)
        }
        Value::Alt(alts) => {
            let parts: Vec<String> = alts.iter().map(|v| value_to_string(v)).collect();
            format!("(or {})", parts.join(" "))
        }
    }
}

/// Extract trees from a namespace's optional "trees" field.
///
/// The "trees" field should be a namespace where each key is a tree name
/// and each value is the tree's namespace value.
fn extract_trees_from_ns(ns: &std::collections::HashMap<String, Value>) -> Vec<(String, Value)> {
    match ns.get("trees") {
        Some(Value::Namespace(tree_map)) => {
            tree_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        }
        Some(Value::List(tree_list)) => {
            // Also support a list of (name, namespace) pairs
            tree_list.iter().filter_map(|item| {
                if let Value::List(pair) = item {
                    if pair.len() == 2 {
                        if let Value::Str(name) = &pair[0] {
                            return Some((name.clone(), pair[1].clone()));
                        }
                    }
                }
                None
            }).collect()
        }
        _ => Vec::new(),
    }
}

/// List of all builtin function names.
pub const BUILTIN_NAMES: &[&str] = &[
    "add", "+", "subtract", "-", "multiply", "*", "divide", "/",
    "modulo", "%", "abs", "negate", "min", "max", "floor", "ceil",
    "<", ">", "<=", ">=", "=", "not", "even", "odd",
    "string-upper", "string-lower", "string-reverse", "string-trim",
    "string-length", "string-contains", "string-split", "string-join",
    "concat", "to-string", "to-number",
    "list", "head", "tail", "length", "cons",
    "nth", "slice", "sort", "reverse", "append",
    "range", "contains", "zip", "enumerate",
    "round", "pow", "sqrt", "log",
    "string-nth", "string-slice", "char-code", "code-char",
    "count-char", "string-replace", "string-chars", "string-take", "string-drop",
    "string-starts-with", "string-ends-with",
    "map", "reduce", "filter", "identity", "print",
    "ns-get", "ns-put", "ns-keys", "ns-values", "ns-merge",
    "ns-size", "ns-flatten", "ns?", "ns-empty",
    "ns-get-or", "ns-has",
    "synthesize", "synthesize-optimize", "test-spec", "memorize",
    "eval-source", "eval-in", "type-of",
    "number?", "string?", "bool?", "list?", "nil?", "function?",
    "apply", "error",
    // Note: "dispatch" is intentionally NOT in BUILTIN_NAMES.
    // It's a special form handled in eval_inner, not a regular builtin.
    // Keeping it out of BUILTIN_NAMES ensures the VM compiler fails on
    // dispatch candidates, triggering the tree-walker fallback where
    // the special form has access to the full env with promoted macros.
    // Grid
    "grid?",
    "grid-make", "grid-from-list", "grid-to-list",
    "grid-width", "grid-height", "grid-size",
    "grid-get", "grid-set", "grid-row", "grid-col",
    "grid-rotate-cw", "grid-rotate-ccw", "grid-rotate-180",
    "grid-flip-h", "grid-flip-v", "grid-transpose",
    "grid-crop", "grid-overlay", "grid-tile", "grid-scale",
    "grid-replace-color", "grid-mask", "grid-pad",
    "grid-colors", "grid-count-color", "grid-most-common", "grid-background",
    "grid-equal", "grid-find-color",
    "grid-symmetric-h", "grid-symmetric-v", "grid-dimensions-equal",
    "grid-bounding-box", "grid-trim",
    "grid-objects", "grid-objects-8", "grid-object-count", "grid-object-colors",
    "grid-hconcat", "grid-vconcat", "grid-hsplit", "grid-vsplit", "grid-quarter",
    // Tier 1: Perceptual
    "grid-flood-fill", "grid-fill-enclosed",
    "grid-draw-line-h", "grid-draw-line-v", "grid-ray",
    "grid-gravity", "grid-xor", "grid-and", "grid-or",
    // Tier 2: Shape analysis
    "grid-object-area", "grid-object-center", "grid-is-rectangle",
    "grid-detect-rectangles", "grid-objects-touching", "grid-overlay-center",
    // Tier 3: Advanced
    "grid-find-subgrid", "grid-neighbor-count", "grid-border", "grid-fill-rect",
];

pub fn make_default_env() -> Env {
    let mut scope = std::collections::HashMap::new();
    for name in BUILTIN_NAMES {
        let sym = intern(name);
        scope.insert(sym, Value::Builtin(sym));
    }
    scope.insert(intern("nil"), Value::Nil);

    // __builtins__: introspectable namespace of all builtins with metadata.
    // Lets SELPH programs reason about the component library.
    let mut builtins_ns = std::collections::HashMap::new();
    let bi = |name: &str, arity: usize, params: &[&str], ret: &str| {
        let mut m = std::collections::HashMap::new();
        m.insert("name".to_string(), Value::Str(name.to_string()));
        m.insert("arity".to_string(), Value::Num(arity as f64));
        m.insert("params".to_string(), Value::List(
            params.iter().map(|p| Value::Str(p.to_string())).collect()));
        m.insert("returns".to_string(), Value::Str(ret.to_string()));
        (name.to_string(), Value::Namespace(m))
    };
    // Arithmetic
    for (n, a, p, r) in [
        ("add", 2, vec!["number", "number"], "number"),
        ("subtract", 2, vec!["number", "number"], "number"),
        ("multiply", 2, vec!["number", "number"], "number"),
        ("divide", 2, vec!["number", "number"], "number"),
        ("modulo", 2, vec!["number", "number"], "number"),
        ("abs", 1, vec!["number"], "number"),
        ("negate", 1, vec!["number"], "number"),
        ("min", 2, vec!["number", "number"], "number"),
        ("max", 2, vec!["number", "number"], "number"),
        ("floor", 1, vec!["number"], "number"),
        ("ceil", 1, vec!["number"], "number"),
        ("round", 1, vec!["number"], "number"),
        ("pow", 2, vec!["number", "number"], "number"),
        ("sqrt", 1, vec!["number"], "number"),
        ("log", 1, vec!["number"], "number"),
    ] { let (k, v) = bi(n, a, &p, r); builtins_ns.insert(k, v); }
    // Comparison
    for (n, a, p, r) in [
        ("<", 2, vec!["number", "number"], "bool"),
        (">", 2, vec!["number", "number"], "bool"),
        ("<=", 2, vec!["number", "number"], "bool"),
        (">=", 2, vec!["number", "number"], "bool"),
        ("=", 2, vec!["any", "any"], "bool"),
        ("not", 1, vec!["bool"], "bool"),
        ("even", 1, vec!["number"], "bool"),
        ("odd", 1, vec!["number"], "bool"),
    ] { let (k, v) = bi(n, a, &p, r); builtins_ns.insert(k, v); }
    // String
    for (n, a, p, r) in [
        ("string-upper", 1, vec!["string"], "string"),
        ("string-lower", 1, vec!["string"], "string"),
        ("string-reverse", 1, vec!["string"], "string"),
        ("string-trim", 1, vec!["string"], "string"),
        ("string-length", 1, vec!["string"], "number"),
        ("string-contains", 2, vec!["string", "string"], "bool"),
        ("string-split", 2, vec!["string", "string"], "list"),
        ("string-join", 2, vec!["list", "string"], "string"),
        ("concat", 2, vec!["any", "any"], "string"),
        ("to-string", 1, vec!["any"], "string"),
        ("to-number", 1, vec!["string"], "number"),
        ("string-nth", 2, vec!["string", "number"], "string"),
        ("string-slice", 3, vec!["string", "number", "number"], "string"),
        ("string-take", 2, vec!["string", "number"], "string"),
        ("string-drop", 2, vec!["string", "number"], "string"),
        ("char-code", 1, vec!["string"], "number"),
        ("code-char", 1, vec!["number"], "string"),
        ("count-char", 2, vec!["string", "string"], "number"),
        ("string-replace", 3, vec!["string", "string", "string"], "string"),
        ("string-chars", 1, vec!["string"], "list"),
        ("string-starts-with", 2, vec!["string", "string"], "bool"),
        ("string-ends-with", 2, vec!["string", "string"], "bool"),
    ] { let (k, v) = bi(n, a, &p, r); builtins_ns.insert(k, v); }
    // List
    for (n, a, p, r) in [
        ("list", 0, vec![], "list"),
        ("head", 1, vec!["list"], "any"),
        ("tail", 1, vec!["list"], "list"),
        ("length", 1, vec!["list"], "number"),
        ("cons", 2, vec!["any", "list"], "list"),
        ("nth", 2, vec!["list", "number"], "any"),
        ("slice", 3, vec!["list", "number", "number"], "list"),
        ("sort", 1, vec!["list"], "list"),
        ("reverse", 1, vec!["list"], "list"),
        ("append", 2, vec!["list", "list"], "list"),
        ("range", 1, vec!["number"], "list"),
        ("contains", 2, vec!["list", "any"], "bool"),
        ("zip", 2, vec!["list", "list"], "list"),
        ("enumerate", 1, vec!["list"], "list"),
        ("map", 2, vec!["function", "list"], "list"),
        ("reduce", 2, vec!["function", "list"], "any"),
        ("filter", 2, vec!["function", "list"], "list"),
    ] { let (k, v) = bi(n, a, &p, r); builtins_ns.insert(k, v); }
    // Grid builtins
    for (n, a, p, r) in [
        ("grid?", 1, vec!["any"], "bool"),
        ("grid-make", 3, vec!["number", "number", "number"], "grid"),
        ("grid-from-list", 1, vec!["list"], "grid"),
        ("grid-to-list", 1, vec!["grid"], "list"),
        ("grid-width", 1, vec!["grid"], "number"),
        ("grid-height", 1, vec!["grid"], "number"),
        ("grid-size", 1, vec!["grid"], "list"),
        ("grid-get", 3, vec!["grid", "number", "number"], "number"),
        ("grid-set", 4, vec!["grid", "number", "number", "number"], "grid"),
        ("grid-row", 2, vec!["grid", "number"], "list"),
        ("grid-col", 2, vec!["grid", "number"], "list"),
        ("grid-rotate-cw", 1, vec!["grid"], "grid"),
        ("grid-rotate-ccw", 1, vec!["grid"], "grid"),
        ("grid-rotate-180", 1, vec!["grid"], "grid"),
        ("grid-flip-h", 1, vec!["grid"], "grid"),
        ("grid-flip-v", 1, vec!["grid"], "grid"),
        ("grid-transpose", 1, vec!["grid"], "grid"),
        ("grid-crop", 5, vec!["grid", "number", "number", "number", "number"], "grid"),
        ("grid-overlay", 4, vec!["grid", "grid", "number", "number"], "grid"),
        ("grid-tile", 3, vec!["grid", "number", "number"], "grid"),
        ("grid-scale", 2, vec!["grid", "number"], "grid"),
        ("grid-replace-color", 3, vec!["grid", "number", "number"], "grid"),
        ("grid-mask", 2, vec!["grid", "grid"], "grid"),
        ("grid-pad", 3, vec!["grid", "number", "number"], "grid"),
        ("grid-colors", 1, vec!["grid"], "list"),
        ("grid-count-color", 2, vec!["grid", "number"], "number"),
        ("grid-most-common", 1, vec!["grid"], "number"),
        ("grid-background", 1, vec!["grid"], "number"),
        ("grid-equal", 2, vec!["grid", "grid"], "bool"),
        ("grid-find-color", 2, vec!["grid", "number"], "list"),
        ("grid-symmetric-h", 1, vec!["grid"], "bool"),
        ("grid-symmetric-v", 1, vec!["grid"], "bool"),
        ("grid-dimensions-equal", 2, vec!["grid", "grid"], "bool"),
        ("grid-bounding-box", 1, vec!["grid"], "list"),
        ("grid-trim", 1, vec!["grid"], "grid"),
        ("grid-objects", 1, vec!["grid"], "list"),
        ("grid-objects-8", 1, vec!["grid"], "list"),
        ("grid-object-count", 1, vec!["grid"], "number"),
        ("grid-object-colors", 1, vec!["grid"], "list"),
        ("grid-hconcat", 2, vec!["grid", "grid"], "grid"),
        ("grid-vconcat", 2, vec!["grid", "grid"], "grid"),
        ("grid-hsplit", 2, vec!["grid", "number"], "list"),
        ("grid-vsplit", 2, vec!["grid", "number"], "list"),
        ("grid-quarter", 1, vec!["grid"], "list"),
    ] { let (k, v) = bi(n, a, &p, r); builtins_ns.insert(k, v); }
    // Tier 1: Perceptual
    for (n, a, p, r) in [
        ("grid-flood-fill", 4, vec!["grid", "number", "number", "number"], "grid"),
        ("grid-fill-enclosed", 2, vec!["grid", "number"], "grid"),
        ("grid-draw-line-h", 5, vec!["grid", "number", "number", "number", "number"], "grid"),
        ("grid-draw-line-v", 5, vec!["grid", "number", "number", "number", "number"], "grid"),
        ("grid-ray", 6, vec!["grid", "number", "number", "number", "number", "number"], "grid"),
        ("grid-gravity", 2, vec!["grid", "number"], "grid"),
        ("grid-xor", 2, vec!["grid", "grid"], "grid"),
        ("grid-and", 2, vec!["grid", "grid"], "grid"),
        ("grid-or", 2, vec!["grid", "grid"], "grid"),
    ] { let (k, v) = bi(n, a, &p, r); builtins_ns.insert(k, v); }
    // Tier 2: Shape analysis
    for (n, a, p, r) in [
        ("grid-object-area", 1, vec!["grid"], "number"),
        ("grid-object-center", 1, vec!["grid"], "list"),
        ("grid-is-rectangle", 1, vec!["grid"], "bool"),
        ("grid-detect-rectangles", 1, vec!["grid"], "list"),
        ("grid-objects-touching", 2, vec!["grid", "grid"], "bool"),
        ("grid-overlay-center", 2, vec!["grid", "grid"], "grid"),
    ] { let (k, v) = bi(n, a, &p, r); builtins_ns.insert(k, v); }
    // Tier 3: Advanced
    for (n, a, p, r) in [
        ("grid-find-subgrid", 2, vec!["grid", "grid"], "list"),
        ("grid-neighbor-count", 3, vec!["grid", "number", "number"], "number"),
        ("grid-border", 1, vec!["grid"], "grid"),
        ("grid-fill-rect", 6, vec!["grid", "number", "number", "number", "number", "number"], "grid"),
    ] { let (k, v) = bi(n, a, &p, r); builtins_ns.insert(k, v); }

    scope.insert(intern("__builtins__"), Value::Namespace(builtins_ns));
    vec![scope]
}
