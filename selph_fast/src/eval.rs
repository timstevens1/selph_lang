//! Evaluator (standalone, no PyO3 dependency)

use crate::types::*;

pub fn eval(nodes: &[Node], idx: usize, env: &mut Env) -> Result<Value, String> {
    match &nodes[idx] {
        Node::Num(n) => Ok(Value::Num(*n)),
        Node::Str(s) => Ok(Value::Str(s.clone())),
        Node::Bool(b) => Ok(Value::Bool(*b)),
        Node::Symbol(name) => {
            env_lookup(env, name).ok_or_else(|| format!("unbound: {}", name))
        }
        Node::Lambda(params, body) => {
            Ok(Value::Closure(params.clone(), *body, env.clone(), nodes.to_vec()))
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
            env.push(std::collections::HashMap::new());
            for (name, val_idx) in bindings {
                let val = eval(nodes, *val_idx, env)?;
                env_define(env, name.clone(), val);
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
                match name.as_str() {
                    "define" if children.len() == 3 => {
                        if let Node::Symbol(var_name) = &nodes[children[1]] {
                            let val = eval(nodes, children[2], env)?;
                            env_define(env, var_name.clone(), val.clone());
                            return Ok(val);
                        }
                    }
                    "defmacro" if children.len() == 4 => {
                        if let Node::Symbol(macro_name) = &nodes[children[1]] {
                            if let Node::App(param_indices) = &nodes[children[2]] {
                                let params: Vec<String> = param_indices.iter()
                                    .filter_map(|&i| {
                                        if let Node::Symbol(s) = &nodes[i] { Some(s.clone()) }
                                        else { None }
                                    }).collect();
                                let body_idx = children[3];
                                let macro_val = Value::RustMacro(
                                    params, nodes.to_vec(), body_idx);
                                env_define(env, macro_name.clone(), macro_val.clone());
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
                    "namespace" => {
                        let mut entries = std::collections::HashMap::new();
                        for &child in &children[1..] {
                            if let Node::App(pair) = &nodes[child] {
                                if pair.len() == 2 {
                                    if let Node::Symbol(key) = &nodes[pair[0]] {
                                        let val = eval(nodes, pair[1], env)?;
                                        entries.insert(key.clone(), val);
                                    }
                                }
                            }
                        }
                        return Ok(Value::Namespace(entries));
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

pub fn apply(fn_val: &Value, args: &[Value], nodes: &[Node], env: &mut Env) -> Result<Value, String> {
    match fn_val {
        Value::Closure(params, body, closed_env, closure_nodes) => {
            let mut new_env = closed_env.clone();
            let mut scope = std::collections::HashMap::new();
            for (i, param) in params.iter().enumerate() {
                if i < args.len() {
                    scope.insert(param.clone(), args[i].clone());
                }
            }
            new_env.push(scope);
            // Use the closure's captured nodes, not the caller's nodes
            eval(closure_nodes, *body, &mut new_env)
        }
        Value::Builtin(name) => apply_builtin(name, args),
        Value::RustMacro(params, macro_nodes, body_root) => {
            let mut macro_env = make_default_env();
            for scope in env.iter() {
                for (k, v) in scope {
                    env_define(&mut macro_env, k.clone(), v.clone());
                }
            }
            let mut scope = std::collections::HashMap::new();
            for (i, param) in params.iter().enumerate() {
                if i < args.len() {
                    scope.insert(param.clone(), args[i].clone());
                }
            }
            macro_env.push(scope);
            eval(macro_nodes, *body_root, &mut macro_env)
        }
        _ => Err(format!("not callable: {:?}", fn_val)),
    }
}

pub fn apply_builtin(name: &str, args: &[Value]) -> Result<Value, String> {
    // Minimal builtins for the CLI. Add more as needed.
    match name {
        "add" | "+" => num2(args, |a, b| a + b),
        "subtract" | "-" => num2(args, |a, b| a - b),
        "multiply" | "*" => num2(args, |a, b| a * b),
        "divide" | "/" => { let (a, b) = nums(args)?; if b == 0.0 { Err("division by zero".into()) } else { Ok(Value::Num(a / b)) } },
        "modulo" | "%" => num2(args, |a, b| a % b),
        "abs" => Ok(Value::Num(num(&args[0])?.abs())),
        "negate" => Ok(Value::Num(-num(&args[0])?)),
        "min" => num2(args, |a, b| a.min(b)),
        "max" => num2(args, |a, b| a.max(b)),
        "floor" => Ok(Value::Num(num(&args[0])?.floor())),
        "ceil" => Ok(Value::Num(num(&args[0])?.ceil())),
        "<" => Ok(Value::Bool(nums(args)?.0 < nums(args)?.1)),
        ">" => Ok(Value::Bool(nums(args)?.0 > nums(args)?.1)),
        "<=" => Ok(Value::Bool(nums(args)?.0 <= nums(args)?.1)),
        ">=" => Ok(Value::Bool(nums(args)?.0 >= nums(args)?.1)),
        "=" => match (&args[0], &args[1]) {
            (Value::Num(a), Value::Num(b)) => Ok(Value::Bool(a == b)),
            (Value::Str(a), Value::Str(b)) => Ok(Value::Bool(a == b)),
            _ => Ok(Value::Bool(false)),
        },
        "not" => match &args[0] { Value::Bool(b) => Ok(Value::Bool(!*b)), _ => Err("not: expected bool".into()) },
        "even" => Ok(Value::Bool(num(&args[0])? % 2.0 == 0.0)),
        "odd" => Ok(Value::Bool(num(&args[0])? % 2.0 != 0.0)),
        "string-upper" => Ok(Value::Str(string(&args[0])?.to_uppercase())),
        "string-lower" => Ok(Value::Str(string(&args[0])?.to_lowercase())),
        "string-reverse" => Ok(Value::Str(string(&args[0])?.chars().rev().collect())),
        "string-trim" => Ok(Value::Str(string(&args[0])?.trim().to_string())),
        "string-length" => Ok(Value::Num(string(&args[0])?.len() as f64)),
        "string-contains" => Ok(Value::Bool(string(&args[0])?.contains(&string(&args[1])?))),
        "string-split" => {
            let s = string(&args[0])?;
            let sep = string(&args[1])?;
            Ok(Value::List(s.split(&sep).map(|p| Value::Str(p.to_string())).collect()))
        }
        "string-join" => {
            let lst = list(&args[0])?;
            let sep = string(&args[1])?;
            let strs: Vec<String> = lst.iter().map(|v| value_to_string(v)).collect();
            Ok(Value::Str(strs.join(&sep)))
        }
        "concat" => {
            let mut r = String::new();
            for a in args { r.push_str(&value_to_string(a)); }
            Ok(Value::Str(r))
        }
        "to-string" => Ok(Value::Str(value_to_string(&args[0]))),
        "to-number" => match &args[0] {
            Value::Str(s) => s.parse::<f64>().map(Value::Num).map_err(|e| format!("to-number: {}", e)),
            Value::Num(n) => Ok(Value::Num(*n)),
            _ => Err("to-number: expected string".into()),
        },
        "list" => Ok(Value::List(args.to_vec())),
        "head" => { let l = list(&args[0])?; l.first().cloned().ok_or("head: empty".into()) },
        "tail" => { let l = list(&args[0])?; if l.is_empty() { Err("tail: empty".into()) } else { Ok(Value::List(l[1..].to_vec())) } },
        "length" => Ok(Value::Num(list(&args[0])?.len() as f64)),
        "cons" => { let mut l = list(&args[1])?; l.insert(0, args[0].clone()); Ok(Value::List(l)) },
        "map" => {
            let l = list(&args[1])?;
            let empty: Vec<Node> = Vec::new();
            let mut env = make_default_env();
            let mut results = Vec::new();
            for item in &l {
                results.push(apply(&args[0], &[item.clone()], &empty, &mut env)?);
            }
            Ok(Value::List(results))
        }
        "reduce" => {
            let l = list(&args[1])?;
            let empty: Vec<Node> = Vec::new();
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
            let empty: Vec<Node> = Vec::new();
            let mut env = make_default_env();
            let mut results = Vec::new();
            for item in &l {
                if let Value::Bool(true) = apply(&args[0], &[item.clone()], &empty, &mut env)? {
                    results.push(item.clone());
                }
            }
            Ok(Value::List(results))
        }
        "identity" => Ok(args[0].clone()),
        "print" => {
            for a in args { print!("{}", value_to_string(a)); }
            println!();
            Ok(Value::Nil)
        }
        // Namespace operations
        "ns-get" => {
            if args.len() < 2 { return Err("ns-get: need namespace and key".into()); }
            let mut result = args[0].clone();
            for arg in &args[1..] {
                let key = match arg { Value::Str(s) => s.clone(), _ => return Err("ns-get: key must be string".into()) };
                result = crate::namespace::ns_get(&result, &key)?;
            }
            Ok(result)
        }
        "ns-put" => {
            if args.len() != 3 { return Err("ns-put: need namespace, key, value".into()); }
            let key = match &args[1] { Value::Str(s) => s.clone(), _ => return Err("ns-put: key must be string".into()) };
            crate::namespace::ns_put(&args[0], &key, args[2].clone())
        }
        "ns-keys" => crate::namespace::ns_keys(&args[0]).map(|ks| Value::List(ks.into_iter().map(Value::Str).collect())),
        "ns-values" => crate::namespace::ns_values(&args[0]).map(Value::List),
        "ns-merge" => {
            if args.len() != 2 { return Err("ns-merge: need two namespaces".into()); }
            crate::namespace::ns_merge(&args[0], &args[1])
        }
        "ns-size" => crate::namespace::ns_size(&args[0]).map(|n| Value::Num(n as f64)),
        "ns-flatten" => {
            crate::namespace::ns_flatten(&args[0]).map(|m| {
                let mut entries = std::collections::HashMap::new();
                for (k, v) in m { entries.insert(k, v); }
                Value::Namespace(entries)
            })
        }
        "ns?" => Ok(Value::Bool(matches!(&args[0], Value::Namespace(_)))),
        "ns-empty" => Ok(Value::Namespace(std::collections::HashMap::new())),
        _ => Err(format!("unknown builtin: {}", name)),
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
        Value::Closure(params, _, _, _) => format!("<lambda ({})>", params.join(" ")),
        Value::Builtin(name) => format!("<builtin {}>", name),
        Value::RustMacro(params, _, _) => format!("<macro ({})>", params.join(" ")),
        Value::Namespace(map) => {
            let keys: Vec<&String> = map.keys().collect();
            format!("<namespace {:?}>", keys)
        }
    }
}

pub fn make_default_env() -> Env {
    let builtins = [
        "add", "+", "subtract", "-", "multiply", "*", "divide", "/",
        "modulo", "%", "abs", "negate", "min", "max", "floor", "ceil",
        "<", ">", "<=", ">=", "=", "not", "even", "odd",
        "string-upper", "string-lower", "string-reverse", "string-trim",
        "string-length", "string-contains", "string-split", "string-join",
        "concat", "to-string", "to-number",
        "list", "head", "tail", "length", "cons",
        "map", "reduce", "filter", "identity", "print",
        "ns-get", "ns-put", "ns-keys", "ns-values", "ns-merge",
        "ns-size", "ns-flatten", "ns?", "ns-empty",
    ];
    let mut scope = std::collections::HashMap::new();
    for name in builtins {
        scope.insert(name.to_string(), Value::Builtin(name.to_string()));
    }
    scope.insert("nil".to_string(), Value::Nil);
    vec![scope]
}
