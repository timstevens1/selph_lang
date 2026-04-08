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
            Value::List(_) => "list", Value::Namespace(_) => "namespace", Value::Nil => "nil",
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
    m
}

thread_local! {
    static BUILTIN_DISPATCH: std::collections::HashMap<Sym, BuiltinFn> = build_dispatch_table();
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
            let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();

            // Extract optional trees
            let trees = extract_trees_from_ns(ns);
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
    "synthesize", "synthesize-optimize",
    "eval-source", "eval-in", "type-of",
    "number?", "string?", "bool?", "list?", "nil?", "function?",
    "apply", "error",
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

    scope.insert(intern("__builtins__"), Value::Namespace(builtins_ns));
    vec![scope]
}
