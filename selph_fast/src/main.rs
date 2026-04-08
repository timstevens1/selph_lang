//! SELPH CLI — standalone binary for the SELPH language.
//!
//! Usage:
//!   selph eval <file.selph>           — evaluate a SELPH file
//!   selph eval -e "(add 1 2)"         — evaluate an expression
//!   selph repl                        — interactive REPL
//!   selph synth <spec.selph>          — synthesize from a spec file
//!   selph run <curriculum.selph>      — run a curriculum

// Import core modules (shared with the PyO3 library)
mod intern;
mod types;
mod parser;
mod eval;
mod namespace;
mod synth;
mod hm;
mod induce;
mod divide;
mod decompose;
mod library;
mod verify;
mod abstraction;
mod multitree;
mod stochastic;
mod meta;
mod taskgen;
mod vm;
mod trace;

use std::env;
use std::fs;
use std::rc::Rc;
use intern::{Sym, intern, resolve};
use types::*;
use parser::*;
use eval::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "eval" => cmd_eval(&args[2..]),
        "repl" => cmd_repl(),
        "parse" => cmd_parse(&args[2..]),
        "synth" => cmd_synth(&args[2..]),
        "curriculum" | "grow" => cmd_curriculum(&args[2..]),
        "bench" => cmd_bench(&args[2..]),
        "generate" => cmd_generate(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "multi-synth" => cmd_multi_synth(&args[2..]),
        "meta-opt" => cmd_meta_optimize(&args[2..]),
        "help" | "--help" | "-h" => print_usage(),
        other => {
            // If it's a .selph file, evaluate it
            if other.ends_with(".selph") {
                cmd_eval(&args[1..]);
            } else {
                eprintln!("Unknown command: {}", other);
                print_usage();
            }
        }
    }
}

fn print_usage() {
    println!("SELPH — Symbolic Evaluation Language for Programmable Hierarchies");
    println!();
    println!("Usage:");
    println!("  selph eval <file.selph>       Evaluate a file");
    println!("  selph eval -e \"(add 1 2)\"     Evaluate an expression");
    println!("  selph parse -e \"(add 1 2)\"    Parse and print AST");
    println!("  selph synth <spec.selph>      Synthesize from a spec file");
    println!("  selph synth -e '1->2 3->6'    Synthesize from inline examples");
    println!("    --library <lib.selph>       Add library (repeatable)");
    println!("  selph grow <tasks.selph>      Run curriculum: solve, promote, save");
    println!("    --meta                      Enable meta-heuristic learning");
    println!("    --extract                   Enable abstraction extraction");
    println!("    --library <lib.selph>       Add library (repeatable)");
    println!("  selph bench                   Run stochastic benchmark suite");
    println!("  selph generate                Generate a curriculum in .selph format");
    println!("  selph verify <prog> <spec>    Verify a program against a spec");
    println!("  selph multi-synth <spec>      Synthesize with multiple libraries");
    println!("  selph repl                    Interactive REPL");
    println!("  selph help                    Show this message");
}

fn cmd_eval(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph eval <file.selph> | -e \"expr\"");
        return;
    }

    let source = if args[0] == "-e" {
        if args.len() < 2 {
            eprintln!("Missing expression after -e");
            return;
        }
        args[1].clone()
    } else {
        match fs::read_to_string(&args[0]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {}", args[0], e);
                return;
            }
        }
    };

    // Parse
    let (nodes_vec, roots) = match parse_file(&source) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };
    let nodes: Rc<[Node]> = nodes_vec.into();

    // Evaluate each top-level expression
    let mut env = make_default_env();
    for root in &roots {
        match eval(&nodes, *root, &mut env) {
            Ok(val) => {
                let s = value_to_string(&val);
                if !s.is_empty() && s != "()" {
                    println!("{}", s);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                return;
            }
        }
    }
}

fn cmd_parse(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph parse -e \"expr\"");
        return;
    }

    let source = if args[0] == "-e" {
        if args.len() < 2 { eprintln!("Missing expression"); return; }
        args[1].clone()
    } else {
        match fs::read_to_string(&args[0]) {
            Ok(s) => s,
            Err(e) => { eprintln!("Error: {}", e); return; }
        }
    };

    match parse_file(&source) {
        Ok((nodes, roots)) => {
            for root in &roots {
                println!("{}", node_to_source(&nodes, *root));
            }
        }
        Err(e) => eprintln!("Parse error: {}", e),
    }
}

fn cmd_repl() {
    use std::io::{self, Write, BufRead};

    println!("SELPH REPL (type :quit to exit)");
    let mut env = make_default_env();

    loop {
        print!("selph> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if io::stdin().lock().read_line(&mut line).is_err() || line.is_empty() {
            break;
        }

        let line = line.trim();
        if line.is_empty() { continue; }
        if line == ":quit" || line == ":q" { break; }

        // Accumulate lines until parens balance
        let mut source = line.to_string();
        while !parens_balanced(&source) {
            print!("  ... ");
            io::stdout().flush().unwrap();
            let mut more = String::new();
            if io::stdin().lock().read_line(&mut more).is_err() { break; }
            source.push(' ');
            source.push_str(more.trim());
        }

        match parse_source(&source) {
            Ok((nodes_vec, root)) => {
                let nodes: Rc<[Node]> = nodes_vec.into();
                match eval(&nodes, root, &mut env) {
                    Ok(val) => {
                        let s = value_to_string(&val);
                        if !s.is_empty() {
                            println!("{}", s);
                        }
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(e) => eprintln!("Parse error: {}", e),
        }
    }
}

fn cmd_synth(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph synth -e '1->2 3->6' [--depth N] [--budget N] [--library file.selph]");
        eprintln!("       selph synth spec.selph");
        return;
    }

    let mut max_depth: usize = 2;
    let mut max_candidates: usize = 100000;
    let mut library_paths: Vec<String> = Vec::new();
    let mut source: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-e" => { source = Some(args.get(i + 1).cloned().unwrap_or_default()); i += 2; }
            "--depth" => { max_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2); i += 2; }
            "--budget" => { max_candidates = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(100000); i += 2; }
            "--library" | "--tree" => {
                if let Some(p) = args.get(i + 1) { library_paths.push(p.clone()); }
                i += 2;
            }
            other => {
                if other.ends_with(".selph") {
                    source = Some(fs::read_to_string(other).unwrap_or_default());
                }
                i += 1;
            }
        }
    }

    let examples_str = match source {
        Some(s) => s,
        None => { eprintln!("No examples provided"); return; }
    };

    let (inputs, expected) = parse_examples(&examples_str);
    if inputs.is_empty() {
        eprintln!("No valid examples found");
        return;
    }

    // Load libraries
    let mut macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    for lib_path in &library_paths {
        match fs::read_to_string(lib_path) {
            Ok(lib_source) => {
                let lib_macros = load_library(&lib_source);
                eprintln!("Loaded {} macros from {}", lib_macros.len(), lib_path);
                macros.extend(lib_macros);
            }
            Err(e) => eprintln!("Warning: couldn't load library {}: {}", lib_path, e),
        }
    }

    // Detect input type
    let input_is_string = matches!(&inputs[0], Value::Str(_));

    // Build components
    let mut synth_comps = synth::default_synth_components(&macros);
    if input_is_string {
        for comp in &mut synth_comps {
            if comp.name == "x" { comp.ret_type = 1; }
        }
    }

    eprintln!("Synthesizing from {} examples, depth={}, budget={}, components={}",
              inputs.len(), max_depth, max_candidates, synth_comps.len());

    let start = std::time::Instant::now();
    let sr = synth::synthesize_with_validation(
        &synth_comps, &inputs, &expected, &macros,
        max_depth, max_candidates, true, None, &[]);
    let elapsed = start.elapsed();

    if sr.found {
        let source = node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap());
        println!("{}", source);
        eprintln!("Found in {} candidates ({:.3}s)", sr.candidates_explored, elapsed.as_secs_f64());
    } else {
        eprintln!("Not found ({} candidates explored, {:.3}s)", sr.candidates_explored, elapsed.as_secs_f64());
        std::process::exit(1);
    }
}

fn parse_examples(input: &str) -> (Vec<Value>, Vec<Value>) {
    let mut inputs = Vec::new();
    let mut expected = Vec::new();

    // Format 1: simple "1->2 3->6 5->10" (no spaces in values)
    if input.contains("->") && !input.contains("(\"") {
        for pair in input.split_whitespace() {
            if let Some(arrow_pos) = pair.find("->") {
                let in_str = &pair[..arrow_pos];
                let out_str = &pair[arrow_pos + 2..];

                let in_val = if let Ok(n) = in_str.parse::<f64>() {
                    Value::Num(n)
                } else {
                    Value::Str(in_str.to_string())
                };

                let out_val = if let Ok(n) = out_str.parse::<f64>() {
                    Value::Num(n)
                } else {
                    Value::Str(out_str.to_string())
                };

                inputs.push(in_val);
                expected.push(out_val);
            }
        }
        return (inputs, expected);
    }

    // Format 2: SELPH-style ("input string" expected) pairs
    // Parse as SELPH expressions
    match parse_file(input) {
        Ok((nodes, roots)) => {
            for &root in &roots {
                if let Node::App(children) = &nodes[root] {
                    if children.len() == 2 {
                        let in_val = node_to_value(&nodes, children[0]);
                        let out_val = node_to_value(&nodes, children[1]);
                        if let (Some(iv), Some(ov)) = (in_val, out_val) {
                            inputs.push(iv);
                            expected.push(ov);
                        }
                    }
                }
            }
        }
        Err(_) => {}
    }

    (inputs, expected)
}

fn node_to_value(nodes: &[Node], idx: usize) -> Option<Value> {
    match &nodes[idx] {
        Node::Num(n) => Some(Value::Num(*n)),
        Node::Str(s) => Some(Value::Str(s.clone())),
        Node::Bool(b) => Some(Value::Bool(*b)),
        Node::Symbol(s) => {
            let name = resolve(*s);
            // Try parsing as bool, then number, then fall back to string
            if name == "true" {
                Some(Value::Bool(true))
            } else if name == "false" {
                Some(Value::Bool(false))
            } else if name == "nil" {
                Some(Value::Nil)
            } else if let Ok(n) = name.parse::<f64>() {
                Some(Value::Num(n))
            } else {
                Some(Value::Str(name))
            }
        }
        // (or val1 val2 ...) → Value::Alt — any alternative is acceptable
        Node::App(children) if children.len() >= 2 => {
            if let Node::Symbol(s) = &nodes[children[0]] {
                if resolve(*s) == "or" {
                    let alts: Option<Vec<Value>> = children[1..]
                        .iter()
                        .map(|&c| node_to_value(nodes, c))
                        .collect();
                    return alts.map(Value::Alt);
                }
            }
            // List literal: (1 2 3) → Value::List([1, 2, 3])
            let elems: Vec<Value> = children.iter()
                .filter_map(|&c| node_to_value(nodes, c))
                .collect();
            if elems.len() == children.len() {
                Some(Value::List(elems))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Load a library file, returning macros as (name, params, nodes, body_root) tuples.
///
/// A library file is evaluated. If the last expression produces a Namespace,
/// its RustMacro entries are extracted. Otherwise, defmacro forms are parsed
/// structurally (legacy fallback).
///
/// This is the single entry point for loading reusable components — there is
/// no separate "tree" concept. A library IS a namespace.
fn load_library(source: &str) -> Vec<(String, Vec<String>, Vec<Node>, usize)> {
    let (nodes_vec, roots) = match parse_file(source) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    // Evaluate the file to get the last value
    let nodes_rc: Rc<[Node]> = nodes_vec.clone().into();
    let mut env = make_default_env();
    let mut last_val = Value::Nil;
    for &r in &roots {
        match eval(&nodes_rc, r, &mut env) {
            Ok(v) => last_val = v,
            Err(_) => {}
        }
    }

    // If the last value is a namespace, extract RustMacro entries from it
    if let Value::Namespace(ref map) = last_val {
        let mut macros = Vec::new();
        for (key, val) in map {
            if let Value::RustMacro(params, macro_nodes, body_root) = val {
                macros.push((
                    key.clone(),
                    params.iter().map(|s| resolve(*s)).collect(),
                    macro_nodes.as_ref().to_vec(),
                    *body_root,
                ));
            }
        }
        if !macros.is_empty() {
            return macros;
        }
    }

    // Fallback: parse defmacro forms structurally (legacy files without ns export)
    let mut macros = Vec::new();
    for &root in &roots {
        if let Node::App(children) = &nodes_vec[root] {
            if children.len() == 4 {
                if let Node::Symbol(s) = &nodes_vec[children[0]] {
                    if *s == intern("defmacro") {
                        if let Node::Symbol(name) = &nodes_vec[children[1]] {
                            if let Node::App(param_indices) = &nodes_vec[children[2]] {
                                let params: Vec<String> = param_indices.iter()
                                    .filter_map(|&i| {
                                        if let Node::Symbol(s) = &nodes_vec[i] { Some(resolve(*s)) }
                                        else { None }
                                    }).collect();
                                macros.push((
                                    resolve(*name), params,
                                    nodes_vec.clone(), children[3],
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    macros
}

// ── Synthesis Engine ────────────────────────────────────────────────

#[derive(Clone)]
struct SynthComponent {
    name: String,
    builtin: Option<String>,
    arity: usize,
    ret_type: u8,   // 0=num, 1=str, 2=bool
    param_types: Vec<u8>,
    priority: f64,
}

#[derive(Clone)]
struct SynthPool {
    nodes: Vec<Node>,
    root: usize,
    ret_type: u8,
    priority: f64,
}

fn default_synth_components(
    macros: &[(String, Vec<String>, Vec<Node>, usize)]
) -> Vec<SynthComponent> {
    let mut comps = vec![
        SynthComponent { name: "x".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 100.0 },
        SynthComponent { name: "0".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "1".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "2".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "3".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "4".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "5".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "6".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "7".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "10".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
        SynthComponent { name: "-1".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
    ];

    // Unary num->num
    for name in &["abs", "negate"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: 0, param_types: vec![0], priority: 0.0,
        });
    }

    // Binary num->num->num
    for name in &["add", "subtract", "multiply", "min", "max", "modulo"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 2, ret_type: 0, param_types: vec![0, 0], priority: 0.0,
        });
    }

    // String ops
    for name in &["string-upper", "string-lower", "string-reverse", "string-trim"] {
        comps.push(SynthComponent {
            name: name.to_string(), builtin: Some(name.to_string()),
            arity: 1, ret_type: 1, param_types: vec![1], priority: 0.0,
        });
    }
    comps.push(SynthComponent {
        name: "string-length".into(), builtin: Some("string-length".into()),
        arity: 1, ret_type: 0, param_types: vec![1], priority: 0.0,
    });

    // Add macro components
    for (name, params, _, _) in macros {
        comps.push(SynthComponent {
            name: name.clone(),
            builtin: Some(name.clone()),
            arity: params.len(),
            ret_type: 0, // assume num for now
            param_types: vec![0; params.len()],
            priority: 30.0,
        });
    }

    comps
}

fn synth_search(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    max_candidates: usize,
) -> (Option<(Vec<Node>, usize)>, usize) {
    const MAX_POOL: usize = 100000;
    let mut explored: usize = 0;
    let mut seen: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
    let macro_env = build_macro_env(macros);

    let test = |entry: &SynthPool, seen: &mut std::collections::HashSet<Vec<u64>>|
        -> Option<(Vec<Node>, usize)>
    {
        let mut ln = entry.nodes.clone();
        let lr = ln.len();
        ln.push(Node::Lambda(vec![intern("x")], entry.root));
        let ln_rc: Rc<[Node]> = ln.clone().into();
        let mut beh = Vec::new();
        let mut ok = true;
        for (inp, exp) in inputs.iter().zip(expected.iter()) {
            let mut env = make_default_env();
            for (nm, ps, mn, mr) in &macro_env {
                let mn_rc: Rc<[Node]> = mn.clone().into();
                env_define(&mut env, intern(nm), Value::RustMacro(ps.iter().map(|s| intern(s)).collect(), mn_rc, *mr));
            }
            let fv = match eval(&ln_rc, lr, &mut env) { Ok(v) => v, Err(_) => { ok = false; break; } };
            match apply(&fv, &[inp.clone()], &ln_rc, &mut env) {
                Ok(a) => { beh.push(val_hash(&a)); if !vals_equal(&a, exp) { ok = false; } }
                Err(_) => { ok = false; break; }
            }
        }
        if !beh.is_empty() { if seen.contains(&beh) { return None; } seen.insert(beh); }
        if ok && !inputs.is_empty() { Some((ln, lr)) } else { None }
    };

    // Build atom pool
    let mut pool: Vec<SynthPool> = Vec::new();
    for comp in components {
        if comp.arity != 0 { continue; }
        let mut nodes = Vec::new();
        if comp.name == "x" { nodes.push(Node::Symbol(intern("x"))); }
        else if let Ok(n) = comp.name.parse::<f64>() { nodes.push(Node::Num(n)); }
        else { nodes.push(Node::Str(comp.name.clone())); }
        pool.push(SynthPool { nodes, root: 0, ret_type: comp.ret_type, priority: comp.priority });
    }
    for e in &pool {
        explored += 1;
        if explored > max_candidates { return (None, explored); }
        if let Some(s) = test(e, &mut seen) { return (Some(s), explored); }
    }

    let mut prev_start = 0usize;
    let mut prev_end = pool.len();

    for _depth in 1..=max_depth {
        let mut new_entries: Vec<SynthPool> = Vec::new();
        let prev = prev_start..prev_end;
        let all_end = prev_end;

        for comp in components {
            if comp.arity == 0 || comp.builtin.is_none() { continue; }
            let bn = comp.builtin.as_ref().unwrap();

            let bn_sym = intern(bn);
            if comp.arity == 1 {
                for pi in prev.clone() {
                    let p = pool[pi].clone();
                    if p.ret_type != comp.param_types[0] && comp.param_types[0] != 255 { continue; }
                    let mut n = p.nodes.clone();
                    let fi = n.len(); n.push(Node::Symbol(bn_sym));
                    let ai = n.len(); n.push(Node::App(vec![fi, p.root]));
                    let e = SynthPool { nodes: n, root: ai, ret_type: comp.ret_type, priority: comp.priority };
                    explored += 1;
                    if explored > max_candidates { return (None, explored); }
                    if let Some(s) = test(&e, &mut seen) { return (Some(s), explored); }
                    if pool.len() + new_entries.len() < MAX_POOL { new_entries.push(e); }
                }
            } else if comp.arity == 2 {
                for pi in prev.clone() {
                    let p1 = pool[pi].clone();
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 { continue; }
                    for ai in 0..all_end {
                        let p2 = pool[ai].clone();
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 { continue; }
                        let mut n = p1.nodes.clone();
                        let off = n.len();
                        for nd in &p2.nodes { n.push(remap_n(nd, off)); }
                        let fi = n.len(); n.push(Node::Symbol(bn_sym));
                        let api = n.len(); n.push(Node::App(vec![fi, p1.root, p2.root + off]));
                        let e = SynthPool { nodes: n, root: api, ret_type: comp.ret_type, priority: comp.priority };
                        explored += 1;
                        if explored > max_candidates { return (None, explored); }
                        if let Some(s) = test(&e, &mut seen) { return (Some(s), explored); }
                        if pool.len() + new_entries.len() < MAX_POOL { new_entries.push(e); }
                    }
                }
                for ai in 0..prev_start {
                    let p1 = pool[ai].clone();
                    if p1.ret_type != comp.param_types[0] && comp.param_types[0] != 255 { continue; }
                    for pi in prev.clone() {
                        let p2 = pool[pi].clone();
                        if p2.ret_type != comp.param_types[1] && comp.param_types[1] != 255 { continue; }
                        let mut n = p1.nodes.clone();
                        let off = n.len();
                        for nd in &p2.nodes { n.push(remap_n(nd, off)); }
                        let fi = n.len(); n.push(Node::Symbol(bn_sym));
                        let api = n.len(); n.push(Node::App(vec![fi, p1.root, p2.root + off]));
                        let e = SynthPool { nodes: n, root: api, ret_type: comp.ret_type, priority: comp.priority };
                        explored += 1;
                        if explored > max_candidates { return (None, explored); }
                        if let Some(s) = test(&e, &mut seen) { return (Some(s), explored); }
                        if pool.len() + new_entries.len() < MAX_POOL { new_entries.push(e); }
                    }
                }
            }
        }
        prev_start = pool.len();
        pool.extend(new_entries);
        prev_end = pool.len();
    }
    (None, explored)
}

fn remap_n(node: &Node, offset: usize) -> Node {
    match node {
        Node::App(c) => Node::App(c.iter().map(|i| i + offset).collect()),
        Node::If(c, t, e) => Node::If(c + offset, t + offset, e + offset),
        Node::Lambda(p, b) => Node::Lambda(p.clone(), b + offset),
        Node::Let(bs, b) => Node::Let(
            bs.iter().map(|(n, i)| (*n, i + offset)).collect(),
            b + offset),
        other => other.clone(),
    }
}

fn build_macro_env(macros: &[(String, Vec<String>, Vec<Node>, usize)])
    -> Vec<(String, Vec<String>, Vec<Node>, usize)>
{
    macros.to_vec()
}

fn val_hash(v: &Value) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match v {
        Value::Num(n) => { 0u8.hash(&mut h); n.to_bits().hash(&mut h); }
        Value::Str(s) => { 1u8.hash(&mut h); s.hash(&mut h); }
        Value::Bool(b) => { 2u8.hash(&mut h); b.hash(&mut h); }
        _ => { 3u8.hash(&mut h); }
    }
    h.finish()
}

fn vals_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        (_, Value::Alt(alts)) => alts.iter().any(|alt| vals_equal(a, alt)),
        _ => false,
    }
}

fn cmd_curriculum(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph grow <tasks.selph> [--library base.selph] [--output grown.selph]");
        eprintln!("       selph grow <tasks.selph> --budget 200000 --depth 3");
        eprintln!("       selph grow <tasks.selph> --heuristic heuristic.selph");
        eprintln!("       selph grow <tasks.selph> --meta --extract");
        eprintln!();
        eprintln!("Task file format:");
        eprintln!("  (task \"name\" depth");
        eprintln!("    (\"input1\" expected1)");
        eprintln!("    (\"input2\" expected2))");
        return;
    }

    let mut task_file = String::new();
    let mut library_paths: Vec<String> = Vec::new();
    let mut output_path = String::from("grown_library.selph");
    let mut default_budget: usize = 200000;
    let mut default_depth: usize = 2;
    let mut enable_meta = false;
    let mut enable_extract = false;
    let mut enable_validate = false;
    let mut filter_path: Option<String> = None;
    let mut trace_path: Option<String> = None;
    let mut heuristic_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--library" | "--tree" => {
                if let Some(p) = args.get(i + 1) { library_paths.push(p.clone()); }
                i += 2;
            }
            "--output" | "-o" => { output_path = args.get(i + 1).cloned().unwrap_or(output_path); i += 2; }
            "--budget" => { default_budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_budget); i += 2; }
            "--depth" => { default_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_depth); i += 2; }
            "--meta" => { enable_meta = true; i += 1; }
            "--extract" => { enable_extract = true; i += 1; }
            "--validate" => { enable_validate = true; i += 1; }
            "--filter" => { filter_path = args.get(i + 1).cloned(); i += 2; }
            "--trace" => { trace_path = args.get(i + 1).cloned(); i += 2; }
            "--heuristic" => { heuristic_path = args.get(i + 1).cloned(); i += 2; }
            other => { task_file = other.to_string(); i += 1; }
        }
    }

    if task_file.is_empty() {
        eprintln!("No task file specified");
        return;
    }

    // Read task file
    let task_source = match fs::read_to_string(&task_file) {
        Ok(s) => s,
        Err(e) => { eprintln!("Error reading {}: {}", task_file, e); return; }
    };

    // Load libraries
    let mut all_macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    let mut base_macro_count: usize = 0; // count of macros from loaded libraries (before promotion)

    for lib_path in &library_paths {
        match fs::read_to_string(lib_path) {
            Ok(s) => {
                let lib_macros = load_library(&s);
                eprintln!("Loaded {} macros from {}", lib_macros.len(), lib_path);
                all_macros.extend(lib_macros);
            }
            Err(e) => eprintln!("Warning: couldn't load library {}: {}", lib_path, e),
        }
    }
    // Deduplicate base macros: if multiple libraries define the same name, keep the last one
    {
        let mut seen = std::collections::HashSet::new();
        let mut deduped = Vec::new();
        for m in all_macros.into_iter().rev() {
            if seen.insert(m.0.clone()) {
                deduped.push(m);
            }
        }
        deduped.reverse();
        all_macros = deduped;
    }
    base_macro_count = all_macros.len();

    // Parse tasks
    let tasks = parse_curriculum_tasks(&task_source, default_depth);
    if tasks.is_empty() {
        eprintln!("No tasks found in {}", task_file);
        return;
    }

    eprintln!();
    eprintln!("SELPH Curriculum: {} tasks", tasks.len());
    eprintln!("  Budget: {}, Default depth: {}", default_budget, default_depth);
    eprintln!("  Library: {} macros", all_macros.len());
    if enable_meta { eprintln!("  Meta-heuristic learning: enabled"); }
    if enable_extract { eprintln!("  Abstraction extraction: enabled"); }
    if enable_validate { eprintln!("  Validation: enabled (20% held out when >= 5 examples)"); }
    if let Some(ref tp) = trace_path { eprintln!("  Trace: {}", tp); }

    // Initialize curriculum trace
    let mut curriculum_trace = trace::CurriculumTrace::new(default_budget, default_depth);

    // Load SELPH depth filter if provided
    let depth_filter: Option<Box<dyn Fn(&synth::SynthComponent, usize) -> bool>> =
        if let Some(ref fp) = filter_path {
            match fs::read_to_string(fp) {
                Ok(src) => {
                    match parse_file(&src) {
                        Ok((nodes_vec, roots)) if !roots.is_empty() => {
                            let nodes: Rc<[Node]> = nodes_vec.into();
                            let mut fenv = make_default_env();
                            match eval(&nodes, roots[roots.len() - 1], &mut fenv) {
                                Ok(filter_val) => {
                                    eprintln!("  Filter: {} (SELPH program)", fp);
                                    Some(synth::make_selph_depth_filter(filter_val))
                                }
                                Err(e) => { eprintln!("  Filter error: {}", e); None }
                            }
                        }
                        _ => { eprintln!("  Filter parse error"); None }
                    }
                }
                Err(e) => { eprintln!("  Filter load error: {}", e); None }
            }
        } else {
            None
        };

    // Load SELPH heuristic if provided
    let heuristic: Option<meta::Heuristic> =
        if let Some(ref hp) = heuristic_path {
            match fs::read_to_string(hp) {
                Ok(src) => {
                    let trimmed = src.trim();
                    match meta::Heuristic::from_source("loaded", trimmed) {
                        Some(h) => {
                            eprintln!("  Heuristic: {} ({})", hp, trimmed);
                            Some(h)
                        }
                        None => { eprintln!("  Heuristic parse error: {}", hp); None }
                    }
                }
                Err(e) => { eprintln!("  Heuristic load error: {}", e); None }
            }
        } else {
            None
        };

    eprintln!("  Output: {}", output_path);
    eprintln!();

    // Solve each task with interleaved priority learning
    let mut solved = 0;
    let mut total_candidates = 0;
    let mut promoted_source = String::new();
    let total_start = std::time::Instant::now();
    let learn_rate = 50.0;

    // Priority map: learned from solutions, persists across tasks
    let mut priorities: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    // RL reward coefficients — learned across curriculum runs
    let mut current_rl_coeffs = synth::RlCoefficients::default();

    // Load learned RL coefficients from library if present
    for (mname, _params, mnodes, mroot) in &all_macros {
        let mnodes_rc: Rc<[Node]> = mnodes.clone().into();
        if mname == "__selph_rl_cold__" {
            if let Ok(val) = eval::eval(&mnodes_rc, *mroot, &mut eval::make_default_env()) {
                if let Value::Num(n) = val { current_rl_coeffs.cold_penalty = n; }
            }
        }
        if mname == "__selph_rl_warm__" {
            if let Ok(val) = eval::eval(&mnodes_rc, *mroot, &mut eval::make_default_env()) {
                if let Value::Num(n) = val { current_rl_coeffs.warm_bonus = n; }
            }
        }
    }
    if current_rl_coeffs.cold_penalty != -50.0 || current_rl_coeffs.warm_bonus != 30.0 {
        eprintln!("  Loaded RL coefficients: cold={:.1}, warm={:.1}",
            current_rl_coeffs.cold_penalty, current_rl_coeffs.warm_bonus);
    }

    // Abstraction extraction state
    let mut solved_programs: Vec<(Vec<Node>, usize)> = Vec::new();


    for (name, task_depth, inputs, expected) in &tasks {
        // Use the task-specified depth. The CLI --depth is only a fallback
        // for tasks that don't specify a depth in the curriculum file.
        let depth = *task_depth;
        let input_is_string = matches!(&inputs[0], Value::Str(_));
        let input_is_list = matches!(&inputs[0], Value::List(_));

        // Compute task features for tracing
        let output_is_boolean = expected.iter().all(|v| matches!(v, Value::Bool(_)));
        let num_distinct_outputs = {
            let mut distinct: Vec<String> = expected.iter().map(|v| format!("{:?}", v)).collect();
            distinct.sort();
            distinct.dedup();
            distinct.len()
        };
        let unary_macro_count = all_macros.iter().filter(|(_, p, _, _)| p.len() == 1).count();
        let task_features = trace::TaskFeatures {
            num_examples: inputs.len(),
            input_type: if input_is_string { "string".to_string() } else { "number".to_string() },
            output_is_boolean,
            num_distinct_outputs,
            library_size: all_macros.len(),
            bool_macros_available: unary_macro_count,
        };
        let num_synth_comps_before = 0usize; // will be set after component creation
        let mut trace_steps: Vec<trace::SolveStep> = Vec::new();
        let mut synth_comps = synth::default_synth_components(&all_macros);
        let extra_bindings: Vec<(String, Value)> = Vec::new();
        if input_is_string {
            for comp in &mut synth_comps {
                if comp.name == "x" { comp.ret_type = synth::TYPE_STR; }
            }
        } else if input_is_list {
            // Infer element type from the list contents
            let elem_type = if let Value::List(elems) = &inputs[0] {
                if elems.iter().all(|v| matches!(v, Value::Num(_))) {
                    synth::TYPE_NUM
                } else if elems.iter().all(|v| matches!(v, Value::Str(_))) {
                    synth::TYPE_STR
                } else {
                    synth::TYPE_ANY
                }
            } else { synth::TYPE_ANY };

            for comp in &mut synth_comps {
                if comp.name == "x" { comp.ret_type = synth::TYPE_LIST; }
            }
            // head/nth return the element type, not ANY
            for comp in &mut synth_comps {
                if (comp.name == "head" || comp.name == "nth") && comp.ret_type == synth::TYPE_ANY {
                    comp.ret_type = elem_type;
                }
            }
            // Note: macro param_types are already inferred by infer_macro_types.
            // Don't override them — macros may accept different types than the
            // raw input (e.g. num→num macros used as intermediate compositions).
        }

        // Apply learned priorities from previous solves
        for comp in &mut synth_comps {
            if let Some(&learned) = priorities.get(&comp.name) {
                comp.priority += learned;
            }
        }

        // Apply loaded heuristic if present — replaces static priority with
        // task-dependent scoring. Otherwise fall back to sorting by learned priority.
        if let Some(ref h) = heuristic {
            let task_ctx = meta::TaskContext::from_examples(inputs, expected);
            synth_comps = meta::apply_heuristic(h, &synth_comps, &task_ctx);
        } else {
            synth_comps.sort_by(|a, b| b.priority.partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal));
        }

        // Split examples into train/validation if --validate enabled
        let (train_inputs, train_expected, val_pairs) = if enable_validate && inputs.len() >= 5 {
            let val_count = (inputs.len() + 4) / 5; // ~20%, rounded up
            let split = inputs.len() - val_count;
            let val: Vec<(Value, Value)> = inputs[split..].iter()
                .zip(expected[split..].iter())
                .map(|(i, e)| (i.clone(), e.clone()))
                .collect();
            (&inputs[..split], &expected[..split], Some(val))
        } else {
            (inputs.as_slice(), expected.as_slice(), None)
        };

        let num_components = synth_comps.len();
        let mut task_solving_strategy: Option<String> = None;
        let mut task_total_candidates: usize = 0;
        let mut task_components_used: Vec<String> = Vec::new();
        let start = std::time::Instant::now();
        let filter_ref: Option<&dyn Fn(&synth::SynthComponent, usize) -> bool> =
            depth_filter.as_ref().map(|f| f.as_ref());
        let snap_ref: Option<&mut Vec<synth::CandidateRecord>> = None;
        let sr = synth::synthesize_full(
            &synth_comps, train_inputs, train_expected, &all_macros,
            depth, default_budget, true,
            val_pairs.as_deref(), &extra_bindings, filter_ref, snap_ref,
            current_rl_coeffs);
        let elapsed = start.elapsed();
        total_candidates += sr.candidates_explored;

        if sr.found {
                let source = node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap());
                solved += 1;
                let explored = sr.candidates_explored;
                eprintln!("  OK  {:30}  {:6} cand  {:.3}s  {}",
                         name, explored, elapsed.as_secs_f64(), source);

                task_solving_strategy = Some("Flat".to_string());
                task_total_candidates = explored;

                // Update priorities: boost components that appeared in the solution
                let used_components = library::extract_components(&source);
                task_components_used = used_components.clone();
                for comp_name in &used_components {
                    let entry = priorities.entry(comp_name.clone()).or_insert(0.0);
                    *entry += learn_rate;
                }

                // Extract body from lambda for the macro
                let body_source = extract_lambda_body(&source);

                // Skip trivial promotions: if the body is just calling an
                // existing macro with the input variable, promoting it would
                // create a self-referential or redundant macro.
                // e.g. if double_idx already exists and the solution is
                // (double_idx x), don't promote (defmacro double_idx (s) (double_idx s))
                let is_trivial = {
                    let trimmed = body_source.trim();
                    // Check if body is (existing_macro x) or (existing_macro s)
                    if trimmed.starts_with('(') && trimmed.ends_with(')') {
                        let inner = &trimmed[1..trimmed.len()-1];
                        let parts: Vec<&str> = inner.split_whitespace().collect();
                        parts.len() == 2
                            && (parts[1] == "x" || parts[1] == "s")
                            && all_macros.iter().any(|(mn, _, _, _)| mn == parts[0])
                    } else {
                        false
                    }
                };

                if is_trivial {
                    eprintln!("    (skipped promotion — trivial wrapper of existing macro)");
                } else {
                    let macro_line = format!("(defmacro {} (s) {})", name, body_source);

                    // Parse and register the new macro
                    if let Ok((mnodes, mroots)) = parse_file(&macro_line) {
                        if !mroots.is_empty() {
                            if let Node::App(children) = &mnodes[mroots[0]] {
                                if children.len() == 4 {
                                    if let Node::Symbol(mname) = &mnodes[children[1]] {
                                        if let Node::App(param_indices) = &mnodes[children[2]] {
                                            let params: Vec<String> = param_indices.iter()
                                                .filter_map(|&i| {
                                                    if let Node::Symbol(s) = &mnodes[i] { Some(resolve(*s)) }
                                                    else { None }
                                                }).collect();
                                            all_macros.push((
                                                resolve(*mname), params,
                                                mnodes.clone(), children[3],
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Accumulate for output
                    promoted_source.push_str(&format!(
                        "\n; {}: found in {} candidates\n{}\n",
                        name, explored, macro_line));
                }

                // Capture pool snapshot for rank-based heuristic optimization.
                // The snapshot records (comp_name, arg_priority_sum) for each
                // composition candidate tested. The solution is the last entry.
                // Online RL coefficient update: nudge toward finding this solution earlier
                if enable_meta {
                    meta::update_rl_coefficients(
                        &mut current_rl_coeffs, sr.candidates_explored, default_budget);
                }
                if enable_extract {
                    if let (Some(nodes), Some(root)) = (&sr.nodes, sr.root) {
                        solved_programs.push((nodes.clone(), root));
                    }
                }
        } else {
            // Fallback 0: Boolean decomposition — try (and P Q), (or P Q)
            // for all pairs of bool-returning macros. O(macros²), instant.
            let bool_result = bool_decompose(inputs, expected, &all_macros);

            if let Some((op, m1, m2, source)) = bool_result {
                solved += 1;
                eprintln!("  BD  {:30}         0.000s  {}",
                         name, source);

                task_solving_strategy = Some("BD".to_string());
                task_total_candidates = sr.candidates_explored;

                let used_components = library::extract_components(&source);
                task_components_used = used_components.clone();
                for comp_name in &used_components {
                    let entry = priorities.entry(comp_name.clone()).or_insert(0.0);
                    *entry += learn_rate;
                }

                let body_source = extract_lambda_body(&source);
                let macro_line = format!("(defmacro {} (s) {})", name, body_source);
                if let Ok((mnodes, mroots)) = parse_file(&macro_line) {
                    if !mroots.is_empty() {
                        if let Node::App(children) = &mnodes[mroots[0]] {
                            if children.len() == 4 {
                                if let Node::Symbol(mname) = &mnodes[children[1]] {
                                    if let Node::App(param_indices) = &mnodes[children[2]] {
                                        let params: Vec<String> = param_indices.iter()
                                            .filter_map(|&i| {
                                                if let Node::Symbol(s) = &mnodes[i] { Some(resolve(*s)) }
                                                else { None }
                                            }).collect();
                                        all_macros.push((
                                            resolve(*mname), params,
                                            mnodes.clone(), children[3],
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }

                promoted_source.push_str(&format!(
                    "\n; {} (bool decomp: {} {} {})\n{}\n",
                    name, op, m1, m2, macro_line));

                if enable_meta {
                    meta::update_rl_coefficients(
                        &mut current_rl_coeffs, 0, default_budget);
                }
            } else {

            // Fallback 1: Try induction
            let ir = induce::induce_from_failure(
                &synth_comps, inputs, expected, &all_macros, depth, default_budget / 2);

            if ir.found {
                let source = node_to_source(&ir.nodes, ir.root);
                solved += 1;
                total_candidates += ir.candidates_explored;
                eprintln!("  IN  {:30}  {:6} cand  {:.3}s  {}",
                         name, ir.candidates_explored, elapsed.as_secs_f64(), source);

                task_solving_strategy = Some("Induction".to_string());
                task_total_candidates = sr.candidates_explored + ir.candidates_explored;

                let used_components = library::extract_components(&source);
                task_components_used = used_components.clone();
                for comp_name in &used_components {
                    let entry = priorities.entry(comp_name.clone()).or_insert(0.0);
                    *entry += learn_rate;
                }

                let body_source = extract_lambda_body(&source);
                let macro_line = format!("(defmacro {} (s) {})", name, body_source);
                if let Ok((mnodes, mroots)) = parse_file(&macro_line) {
                    if !mroots.is_empty() {
                        if let Node::App(children) = &mnodes[mroots[0]] {
                            if children.len() == 4 {
                                if let Node::Symbol(mname) = &mnodes[children[1]] {
                                    if let Node::App(param_indices) = &mnodes[children[2]] {
                                        let params: Vec<String> = param_indices.iter()
                                            .filter_map(|&i| {
                                                if let Node::Symbol(s) = &mnodes[i] { Some(resolve(*s)) }
                                                else { None }
                                            }).collect();
                                        all_macros.push((resolve(*mname), params, mnodes.clone(), children[3]));
                                    }
                                }
                            }
                        }
                    }
                }
                promoted_source.push_str(&format!(
                    "\n; {} (induced): found in {} candidates\n{}\n",
                    name, ir.candidates_explored, macro_line));

                if enable_extract {
                    solved_programs.push((ir.nodes.clone(), ir.root));
                }
            } else {
                // Fallback 1.5: Try template decomposition (higher-order)
                let ho_r = decompose::try_decomposition(
                    &synth_comps, inputs, expected, &all_macros, depth, default_budget / 2);
                total_candidates += ho_r.candidates_explored;

                if ho_r.found {
                    let source = node_to_source(&ho_r.nodes, ho_r.root);
                    solved += 1;
                    eprintln!("  HO  {:30}  {:6} cand  {:.3}s  {}",
                             name, ho_r.candidates_explored, elapsed.as_secs_f64(), source);

                    task_solving_strategy = Some(format!("Decomp({})", ho_r.template_used));
                    task_total_candidates = sr.candidates_explored + ir.candidates_explored + ho_r.candidates_explored;

                    let used_components = library::extract_components(&source);
                    task_components_used = used_components.clone();
                    for comp_name in &used_components {
                        let entry = priorities.entry(comp_name.clone()).or_insert(0.0);
                        *entry += learn_rate;
                    }

                    let body_source = extract_lambda_body(&source);
                    let macro_line = format!("(defmacro {} (s) {})", name, body_source);
                    if let Ok((mnodes, mroots)) = parse_file(&macro_line) {
                        if !mroots.is_empty() {
                            if let Node::App(children) = &mnodes[mroots[0]] {
                                if children.len() == 4 {
                                    if let Node::Symbol(mname) = &mnodes[children[1]] {
                                        if let Node::App(param_indices) = &mnodes[children[2]] {
                                            let params: Vec<String> = param_indices.iter()
                                                .filter_map(|&i| {
                                                    if let Node::Symbol(s) = &mnodes[i] { Some(resolve(*s)) }
                                                    else { None }
                                                }).collect();
                                            all_macros.push((resolve(*mname), params, mnodes.clone(), children[3]));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    promoted_source.push_str(&format!(
                        "\n; {} (HO: {}): found in {} candidates\n{}\n",
                        name, ho_r.template_used, ho_r.candidates_explored, macro_line));

                    if enable_extract {
                        solved_programs.push((ho_r.nodes.clone(), ho_r.root));
                    }
                } else {

                // Fallback 2: Try divide-and-conquer
                let dr = divide::divide_and_conquer(
                    &synth_comps, inputs, expected, &all_macros, depth, default_budget / 2);

                if dr.found {
                    let source = node_to_source(&dr.nodes, dr.root);
                    solved += 1;
                    total_candidates += dr.candidates_explored;
                    eprintln!("  DC  {:30}  {:6} cand  {:.3}s  {}",
                             name, dr.candidates_explored, elapsed.as_secs_f64(), source);

                    task_solving_strategy = Some("D&C".to_string());
                    task_total_candidates = sr.candidates_explored + dr.candidates_explored;
                    task_components_used = library::extract_components(&source);

                    let body_source = extract_lambda_body(&source);
                    let macro_line = format!("(defmacro {} (s) {})", name, body_source);

                    // Register D&C solution as macro so later tasks can use it
                    if let Ok((mnodes, mroots)) = parse_file(&macro_line) {
                        if !mroots.is_empty() {
                            if let Node::App(children) = &mnodes[mroots[0]] {
                                if children.len() == 4 {
                                    if let Node::Symbol(mname) = &mnodes[children[1]] {
                                        if let Node::App(param_indices) = &mnodes[children[2]] {
                                            let params: Vec<String> = param_indices.iter()
                                                .filter_map(|&i| {
                                                    if let Node::Symbol(s) = &mnodes[i] { Some(resolve(*s)) }
                                                    else { None }
                                                }).collect();
                                            all_macros.push((resolve(*mname), params, mnodes.clone(), children[3]));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    promoted_source.push_str(&format!(
                        "\n; {} (D&C): found in {} candidates\n{}\n",
                        name, dr.candidates_explored, macro_line));

                    if enable_extract {
                        solved_programs.push((dr.nodes.clone(), dr.root));
                    }
                } else {
                    // Fallback 3: Memorization — namespace lookup table
                    if let Some((mem_nodes, mem_root)) = memorize_from_examples(inputs, expected) {
                        let source = node_to_source(&mem_nodes, mem_root);
                        solved += 1;
                        eprintln!("  ME  {:30}  {:6} memo  {:.3}s  {}",
                                 name, inputs.len(), elapsed.as_secs_f64(), source);

                        task_solving_strategy = Some("Memo".to_string());
                        task_total_candidates = sr.candidates_explored;

                        let body_source = extract_lambda_body(&source);
                        let macro_line = format!("(defmacro {} (s) {})", name, body_source);
                        if let Ok((mnodes, mroots)) = parse_file(&macro_line) {
                            if !mroots.is_empty() {
                                if let Node::App(children) = &mnodes[mroots[0]] {
                                    if children.len() == 4 {
                                        if let Node::Symbol(mname) = &mnodes[children[1]] {
                                            if let Node::App(param_indices) = &mnodes[children[2]] {
                                                let params: Vec<String> = param_indices.iter()
                                                    .filter_map(|&i| {
                                                        if let Node::Symbol(s) = &mnodes[i] { Some(resolve(*s)) }
                                                        else { None }
                                                    }).collect();
                                                all_macros.push((
                                                    resolve(*mname).to_string(), params,
                                                    mnodes.clone(), children[3],
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        promoted_source.push_str(&format!(
                            "\n; {} (memorized: {} entries)\n{}\n",
                            name, inputs.len(), macro_line));
                    } else {
                        let explored = sr.candidates_explored;
                        eprintln!("  --  {:30}  {:6} cand  {:.3}s",
                                 name, explored, elapsed.as_secs_f64());
                        task_total_candidates = explored;
                    }
                }
            }
            } // end HO else
        } // end bool_decompose else
        }

        // Record trace for this task
        if trace_path.is_some() {
            let task_elapsed = start.elapsed();
            let task_trace = trace::TaskTrace {
                task_name: name.clone(),
                task_depth: depth,
                features: task_features,
                steps: Vec::new(), // individual step breakdown not yet wired
                solved: task_solving_strategy.is_some(),
                solving_strategy: task_solving_strategy,
                total_candidates: task_total_candidates,
                total_wall_time_ms: task_elapsed.as_secs_f64() * 1000.0,
                components_used: task_components_used,
                components_available: num_components,
            };
            curriculum_trace.tasks.push(task_trace);
        }

        // Periodic abstraction extraction: every 10 solved tasks
        if enable_extract && solved > 0 && solved % 10 == 0 && solved_programs.len() >= 2 {
            eprintln!("  [extract] Extracting abstractions from {} solved programs...", solved_programs.len());
            let abstractions = abstraction::extract_abstractions(
                &solved_programs, 2, 2, 1.0, 10, "lib",
            );
            for abs in &abstractions {
                let macro_line = format!(
                    "(defmacro {} ({}) {})",
                    abs.name,
                    abs.params.join(" "),
                    node_to_source(&abs.body, abs.root),
                );
                eprintln!("  [extract] Extracted: {} (freq={}, compression={:.1}, params={})",
                    abs.name, abs.frequency, abs.compression_score, abs.params.join(", "));
                if let Ok((mnodes, mroots)) = parse_file(&macro_line) {
                    if !mroots.is_empty() {
                        if let Node::App(children) = &mnodes[mroots[0]] {
                            if children.len() == 4 {
                                if let Node::Symbol(mname) = &mnodes[children[1]] {
                                    if let Node::App(param_indices) = &mnodes[children[2]] {
                                        let params: Vec<String> = param_indices.iter()
                                            .filter_map(|&pi| {
                                                if let Node::Symbol(s) = &mnodes[pi] { Some(resolve(*s)) }
                                                else { None }
                                            }).collect();
                                        all_macros.push((resolve(*mname), params, mnodes.clone(), children[3]));
                                    }
                                }
                            }
                        }
                    }
                }
                promoted_source.push_str(&format!(
                    "\n; [extracted] {} (freq={}, compression={:.1})\n{}\n",
                    abs.name, abs.frequency, abs.compression_score, macro_line));
            }
            if !abstractions.is_empty() {
                eprintln!("  [extract] Added {} abstractions to library", abstractions.len());
            }
        }
    }

    // Log final RL coefficients
    if enable_meta {
        eprintln!("  [rl] Final coefficients: cold={:.1}, warm={:.1}",
            current_rl_coeffs.cold_penalty, current_rl_coeffs.warm_bonus);
    }

    // End-of-run abstraction extraction
    if enable_extract && solved_programs.len() >= 2 && (solved % 10 != 0 || solved == 0) {
        eprintln!("  [extract] Final extraction from {} solved programs...", solved_programs.len());
        let abstractions = abstraction::extract_abstractions(
            &solved_programs, 2, 2, 1.0, 10, "lib",
        );
        for abs in &abstractions {
            let macro_line = format!(
                "(defmacro {} ({}) {})",
                abs.name,
                abs.params.join(" "),
                node_to_source(&abs.body, abs.root),
            );
            eprintln!("  [extract] Extracted: {} (freq={}, compression={:.1}, params={})",
                abs.name, abs.frequency, abs.compression_score, abs.params.join(", "));
            promoted_source.push_str(&format!(
                "\n; [extracted] {} (freq={}, compression={:.1})\n{}\n",
                abs.name, abs.frequency, abs.compression_score, macro_line));
        }
        if !abstractions.is_empty() {
            eprintln!("  [extract] Added {} abstractions to library", abstractions.len());
        }
    }

    let total_elapsed = total_start.elapsed();
    eprintln!();
    eprintln!("Results: {}/{} solved ({:.0}%)",
             solved, tasks.len(),
             100.0 * solved as f64 / tasks.len() as f64);
    eprintln!("Total: {} candidates, {:.1}s",
             total_candidates, total_elapsed.as_secs_f64());
    eprintln!("Library grew by {} macros", solved);

    // Write trace JSON if requested
    if let Some(ref tp) = trace_path {
        curriculum_trace.finalize();
        let json = curriculum_trace.to_json();
        match fs::write(tp, &json) {
            Ok(_) => eprintln!("Trace saved to {} ({} tasks, {} candidates)",
                tp, curriculum_trace.total_tasks, curriculum_trace.total_candidates),
            Err(e) => eprintln!("Error writing trace to {}: {}", tp, e),
        }
    }

    // Save grown library — this IS the trained model.
    // Serialized from all_macros: deduplicated, base + promoted + extracted.
    let mut output = String::new();
    output.push_str("; SELPH library — trained model from curriculum runner\n");
    output.push_str(&format!("; {} macros ({} base, {} promoted from this run)\n",
                             all_macros.len(), base_macro_count, solved));
    if !library_paths.is_empty() {
        output.push_str(&format!("; Libraries: {}\n", library_paths.join(", ")));
    }
    if enable_meta {
        output.push_str(&format!("; Includes learned RL coefficients (cold={:.1}, warm={:.1})\n",
            current_rl_coeffs.cold_penalty, current_rl_coeffs.warm_bonus));
    }
    output.push_str("\n");

    // Serialize all macros from all_macros (base + promoted, deduplicated)
    output.push_str("; --- Library macros ---\n");
    for (mname, params, mnodes, mroot) in &all_macros {
        // Skip internal RL coefficient macros — they're written separately below
        if mname.starts_with("__selph_rl_") { continue; }
        let body = node_to_source(mnodes, *mroot);
        output.push_str(&format!("(defmacro {} ({}) {})\n",
            mname, params.join(" "), body));
    }



    // Save learned RL coefficients
    if enable_meta {
        output.push_str("\n; --- Learned RL reward coefficients ---\n");
        output.push_str(&format!("(defmacro __selph_rl_cold__ (_) {})\n", current_rl_coeffs.cold_penalty));
        output.push_str(&format!("(defmacro __selph_rl_warm__ (_) {})\n", current_rl_coeffs.warm_bonus));
    }

    match fs::write(&output_path, &output) {
        Ok(_) => eprintln!("Saved to {}", output_path),
        Err(e) => eprintln!("Error writing {}: {}", output_path, e),
    }
}

fn parse_curriculum_tasks(source: &str, default_depth: usize)
    -> Vec<(String, usize, Vec<Value>, Vec<Value>)>
{
    let mut tasks = Vec::new();

    let (nodes, roots) = match parse_file(source) {
        Ok(r) => r,
        Err(e) => { eprintln!("Parse error in task file: {}", e); return tasks; }
    };

    for &root in &roots {
        // Each task: (task "name" depth (in1 out1) (in2 out2) ...)
        if let Node::App(children) = &nodes[root] {
            if children.len() < 4 { continue; }
            if let Node::Symbol(s) = &nodes[children[0]] {
                if *s != intern("task") { continue; }
            } else { continue; }

            let name = match &nodes[children[1]] {
                Node::Str(s) => s.clone(),
                Node::Symbol(s) => resolve(*s),
                _ => continue,
            };

            let depth = match &nodes[children[2]] {
                Node::Num(n) => *n as usize,
                _ => default_depth,
            };

            let mut inputs = Vec::new();
            let mut expected = Vec::new();

            for &child_idx in &children[3..] {
                if let Node::App(pair) = &nodes[child_idx] {
                    if pair.len() == 2 {
                        if let (Some(iv), Some(ov)) = (
                            node_to_value(&nodes, pair[0]),
                            node_to_value(&nodes, pair[1]),
                        ) {
                            inputs.push(iv);
                            expected.push(ov);
                        }
                    }
                }
            }

            if !inputs.is_empty() {
                tasks.push((name, depth, inputs, expected));
            }
        }
    }

    tasks
}

/// Boolean decomposition: try (and P Q), (or P Q), (not P) for all
/// bool-returning macros in the library. Returns (op, macro1, macro2, source)
/// if a combination matches all examples. O(macros²), essentially instant.
fn bool_decompose(
    inputs: &[Value],
    expected: &[Value],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
) -> Option<(String, String, String, String)> {
    // Only applicable for boolean output tasks
    if expected.is_empty() || !matches!(&expected[0], Value::Bool(_)) {
        return None;
    }

    // Collect bool-returning unary macros and precompute their outputs
    let mut bool_macros: Vec<(String, Vec<bool>)> = Vec::new();

    for (mname, params, mnodes, mroot) in macros {
        if params.len() != 1 { continue; }
        let mnodes_rc: Rc<[Node]> = mnodes.clone().into();
        let sym_params: Vec<Sym> = params.iter().map(|s| intern(s)).collect();
        let val = Value::RustMacro(sym_params, mnodes_rc.clone(), *mroot);

        let mut outputs = Vec::new();
        let mut all_bool = true;
        for inp in inputs {
            let mut env = eval::make_default_env();
            // Also load other macros so compositions work
            for (mn2, p2, n2, r2) in macros {
                let n2_rc: Rc<[Node]> = n2.clone().into();
                env_define(&mut env, intern(mn2),
                    Value::RustMacro(p2.iter().map(|s| intern(s)).collect(), n2_rc, *r2));
            }
            match eval::apply(&val, &[inp.clone()], &mnodes_rc, &mut env) {
                Ok(Value::Bool(b)) => outputs.push(b),
                _ => { all_bool = false; break; }
            }
        }
        if all_bool && outputs.len() == inputs.len() {
            bool_macros.push((mname.clone(), outputs));
        }
    }

    let expected_bools: Vec<bool> = expected.iter().filter_map(|v| {
        if let Value::Bool(b) = v { Some(*b) } else { None }
    }).collect();
    if expected_bools.len() != expected.len() { return None; }

    // Try (not P)
    for (name, outputs) in &bool_macros {
        let negated: Vec<bool> = outputs.iter().map(|b| !b).collect();
        if negated == expected_bools {
            let source = format!("(lambda (x) (not ({} x)))", name);
            return Some(("not".into(), name.clone(), String::new(), source));
        }
    }

    // Try (and P Q) and (or P Q)
    for (i, (name1, out1)) in bool_macros.iter().enumerate() {
        for (name2, out2) in bool_macros.iter().skip(i) {
            // and
            let and_result: Vec<bool> = out1.iter().zip(out2).map(|(a, b)| *a && *b).collect();
            if and_result == expected_bools {
                let source = format!("(lambda (x) (and ({} x) ({} x)))", name1, name2);
                return Some(("and".into(), name1.clone(), name2.clone(), source));
            }

            // or
            let or_result: Vec<bool> = out1.iter().zip(out2).map(|(a, b)| *a || *b).collect();
            if or_result == expected_bools {
                let source = format!("(lambda (x) (or ({} x) ({} x)))", name1, name2);
                return Some(("or".into(), name1.clone(), name2.clone(), source));
            }

            // Also try with negations: (and P (not Q)), (and (not P) Q)
            let and_not2: Vec<bool> = out1.iter().zip(out2).map(|(a, b)| *a && !*b).collect();
            if and_not2 == expected_bools {
                let source = format!("(lambda (x) (and ({} x) (not ({} x))))", name1, name2);
                return Some(("and-not".into(), name1.clone(), name2.clone(), source));
            }
            let and_not1: Vec<bool> = out1.iter().zip(out2).map(|(a, b)| !*a && *b).collect();
            if and_not1 == expected_bools {
                let source = format!("(lambda (x) (and (not ({} x)) ({} x)))", name1, name2);
                return Some(("not-and".into(), name1.clone(), name2.clone(), source));
            }
        }
    }

    None
}

fn extract_lambda_body(source: &str) -> String {
    // "(lambda (x) body)" -> "body" with x replaced by s
    if source.starts_with("(lambda (") {
        // Find the closing paren of the params list
        let after_lparen = &source["(lambda (".len()..];
        if let Some(paren_end) = after_lparen.find(')') {
            let rest = &after_lparen[paren_end + 1..]; // " body)"
            let rest = rest.trim_start();
            if rest.len() > 1 {
                // Strip the final closing paren of the lambda
                let body = &rest[..rest.len() - 1];
                // Replace the param name x with s for macro consistency.
                // Use a proper word-boundary replacement to handle all positions.
                let mut result = String::with_capacity(body.len());
                let chars: Vec<char> = body.chars().collect();
                let mut i = 0;
                while i < chars.len() {
                    if chars[i] == 'x' {
                        let before_ok = i == 0 || !chars[i-1].is_alphanumeric() && chars[i-1] != '_' && chars[i-1] != '-';
                        let after_ok = i + 1 >= chars.len() || !chars[i+1].is_alphanumeric() && chars[i+1] != '_' && chars[i+1] != '-';
                        if before_ok && after_ok {
                            result.push('s');
                        } else {
                            result.push('x');
                        }
                    } else {
                        result.push(chars[i]);
                    }
                    i += 1;
                }
                return result;
            }
        }
    }
    source.to_string()
}

/// Build a namespace-backed lookup macro from examples.
///
/// Given string inputs and any-typed outputs, constructs:
///   (lambda (x) (ns-get-or (ns ("key1" val1) ("key2" val2) ...) x default))
///
/// Returns Some((nodes, root)) or None if inputs aren't all strings.
fn memorize_from_examples(inputs: &[Value], expected: &[Value]) -> Option<(Vec<Node>, usize)> {
    // All inputs must be strings (namespace keys)
    if inputs.is_empty() || !inputs.iter().all(|v| matches!(v, Value::Str(_))) {
        return None;
    }

    // Deduplicate: same input must map to same output
    let mut map: std::collections::HashMap<String, &Value> = std::collections::HashMap::new();
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        if let Value::Str(key) = inp {
            if let Some(existing) = map.get(key.as_str()) {
                if format!("{:?}", existing) != format!("{:?}", exp) {
                    return None; // conflicting outputs for same input
                }
            }
            map.insert(key.clone(), exp);
        }
    }

    // Infer default value from output type
    let default_val = match &expected[0] {
        Value::Bool(_) => Value::Bool(false),
        Value::Num(_) => Value::Num(0.0),
        Value::Str(_) => Value::Str(String::new()),
        _ => Value::Nil,
    };

    // Build AST: (lambda (x) (ns-get-or (ns ("k1" v1) ...) x default))
    let mut nodes: Vec<Node> = Vec::new();

    // Build namespace entries as (key value) App pairs
    let mut ns_children: Vec<usize> = Vec::new();
    // First child: the "ns" symbol
    let ns_sym_idx = nodes.len();
    nodes.push(Node::Symbol(intern("ns")));
    ns_children.push(ns_sym_idx);

    for (key, val) in &map {
        let key_idx = nodes.len();
        nodes.push(Node::Str(key.clone()));
        let val_idx = nodes.len();
        match val {
            Value::Num(n) => nodes.push(Node::Num(*n)),
            Value::Bool(b) => nodes.push(Node::Bool(*b)),
            Value::Str(s) => nodes.push(Node::Str(s.clone())),
            _ => nodes.push(Node::Bool(false)),
        }
        let pair_idx = nodes.len();
        nodes.push(Node::App(vec![key_idx, val_idx]));
        ns_children.push(pair_idx);
    }

    // (ns ...) application
    let ns_app_idx = nodes.len();
    nodes.push(Node::App(ns_children));

    // "x" symbol (parameter)
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    // default value
    let default_idx = nodes.len();
    match &default_val {
        Value::Num(n) => nodes.push(Node::Num(*n)),
        Value::Bool(b) => nodes.push(Node::Bool(*b)),
        Value::Str(s) => nodes.push(Node::Str(s.clone())),
        _ => nodes.push(Node::Bool(false)),
    }

    // "ns-get-or" symbol
    let ngo_idx = nodes.len();
    nodes.push(Node::Symbol(intern("ns-get-or")));

    // (ns-get-or (ns ...) x default)
    let body_idx = nodes.len();
    nodes.push(Node::App(vec![ngo_idx, ns_app_idx, x_idx, default_idx]));

    // (lambda (x) body)
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], body_idx));

    Some((nodes, lambda_idx))
}

fn parens_balanced(s: &str) -> bool {
    let mut depth = 0i32;
    let mut in_string = false;
    for c in s.chars() {
        if c == '"' { in_string = !in_string; }
        if in_string { continue; }
        if c == '(' { depth += 1; }
        if c == ')' { depth -= 1; }
    }
    depth <= 0
}

fn cmd_bench(args: &[String]) {
    let mut max_depth: usize = 2;
    let mut max_budget: usize = 50000;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--depth" => { max_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(max_depth); i += 2; }
            "--budget" => { max_budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(max_budget); i += 2; }
            _ => { i += 1; }
        }
    }

    let suite = stochastic::generate_benchmark_suite();
    eprintln!("SELPH Stochastic Benchmark Suite: {} tasks", suite.len());
    eprintln!("  Depth: {}, Budget: {}", max_depth, max_budget);
    eprintln!();

    let macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    let mut solved = 0usize;
    let mut total_cands = 0usize;
    let total_start = std::time::Instant::now();

    for task in &suite {
        let comps = synth::default_synth_components(&macros);
        let start = std::time::Instant::now();
        let sr = synth::synthesize(
            &comps, &task.inputs, &task.expected, &macros,
            max_depth, max_budget, true,
        );
        let elapsed = start.elapsed();
        total_cands += sr.candidates_explored;

        if sr.found {
            solved += 1;
            let source = node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap());
            let ref_str = task.reference_program.as_deref().unwrap_or("?");
            eprintln!("  OK  {:30} stage={} {:6} cand {:.3}s  found={} ref={}",
                task.name, task.stage, sr.candidates_explored,
                elapsed.as_secs_f64(), source, ref_str);
        } else {
            eprintln!("  --  {:30} stage={} {:6} cand {:.3}s  opt_acc={:.2} entropy={:.3}",
                task.name, task.stage, sr.candidates_explored,
                elapsed.as_secs_f64(), task.optimal_accuracy, task.entropy_rate);
        }
    }

    let total_elapsed = total_start.elapsed();
    eprintln!();
    eprintln!("Results: {}/{} solved ({:.0}%)",
        solved, suite.len(),
        if suite.is_empty() { 0.0 } else { 100.0 * solved as f64 / suite.len() as f64 });
    eprintln!("Total: {} candidates, {:.1}s", total_cands, total_elapsed.as_secs_f64());
}

fn cmd_generate(args: &[String]) {
    let mut stage_count: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--stages" => { stage_count = args.get(i + 1).and_then(|s| s.parse().ok()); i += 2; }
            _ => { i += 1; }
        }
    }

    let tasks = if let Some(_n) = stage_count {
        // Use the default curriculum (6 stages)
        taskgen::default_curriculum()
    } else {
        taskgen::default_curriculum()
    };

    eprintln!("Generated {} tasks", tasks.len());

    // Print each task in .selph format
    for task in &tasks {
        println!("{}", taskgen::task_to_selph(task));
        println!();
    }
}

// ── Verify command ──────────────────────────────────────────────────

fn cmd_verify(args: &[String]) {
    // Usage: selph verify -e "(lambda (x) (add x x))" --examples "1->2 3->6 5->10"
    //    or: selph verify <program.selph> --examples "1->2 3->6"
    //    or: selph verify -e "(add 1 2)" --type number
    let mut program_source: Option<String> = None;
    let mut examples_str: Option<String> = None;
    let mut expected_type: Option<String> = None;
    let mut library_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-e" => { program_source = args.get(i + 1).cloned(); i += 2; }
            "--examples" => { examples_str = args.get(i + 1).cloned(); i += 2; }
            "--type" => { expected_type = args.get(i + 1).cloned(); i += 2; }
            "--library" => { library_path = args.get(i + 1).cloned(); i += 2; }
            other if other.ends_with(".selph") => {
                if let Ok(src) = fs::read_to_string(other) {
                    program_source = Some(src);
                }
                i += 1;
            }
            _ => { i += 1; }
        }
    }

    let source = match program_source {
        Some(s) => s,
        None => { eprintln!("Usage: selph verify -e \"(lambda (x) ...)\" --examples \"1->2 3->6\""); return; }
    };

    // Parse program
    let (nodes_vec, roots) = match parse_file(&source) {
        Ok(r) => r,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };
    let nodes: Rc<[Node]> = nodes_vec.into();
    if roots.is_empty() { eprintln!("No expressions found"); return; }

    // Load library macros
    let mut env = make_default_env();
    if let Some(lib_path) = &library_path {
        if let Ok(lib_src) = fs::read_to_string(lib_path) {
            if let Ok((ln_vec, lr)) = parse_file(&lib_src) {
                let ln: Rc<[Node]> = ln_vec.into();
                for &r in &lr {
                    let _ = eval(&ln, r, &mut env);
                }
            }
        }
    }

    // Evaluate program
    let program_val = match eval(&nodes, roots[roots.len() - 1], &mut env) {
        Ok(v) => v,
        Err(e) => { eprintln!("Eval error: {}", e); return; }
    };

    // Build spec
    let mut spec = verify::Spec::new();
    spec.expected_type = expected_type;

    // Parse examples if provided
    if let Some(ex_str) = &examples_str {
        let mut example_pairs = Vec::new();
        for pair_str in ex_str.split_whitespace() {
            if let Some((inp_s, out_s)) = pair_str.split_once("->") {
                let inp: Value = if let Ok(n) = inp_s.parse::<f64>() {
                    Value::Num(n)
                } else {
                    Value::Str(inp_s.to_string())
                };
                let out: Value = if let Ok(n) = out_s.parse::<f64>() {
                    Value::Num(n)
                } else {
                    Value::Str(out_s.to_string())
                };
                example_pairs.push((inp, out));
            }
        }
        if !example_pairs.is_empty() {
            spec.goal = Some(verify::Goal::Examples(example_pairs));
        }
    }

    // Verify
    let result = verify::verify_fn(&spec, &program_val, &mut env);

    // Print result
    println!("Verification Result:");
    println!("  type_match:       {}", result.type_match);
    println!("  shape_match:      {}", result.shape_match);
    println!("  constraint_match: {}", result.constraint_match);
    println!("  goal_score:       {:.2}", result.goal_score);
    println!("  passed:           {}", result.passed());
    println!("  reward:           {:.3}", verify::reward_default(&result));

    if let Some(te) = &result.type_error {
        println!("  type_error:       {}", te);
    }
    if let Some(se) = &result.shape_error {
        println!("  shape_error:      {}", se);
    }
    for ce in &result.constraint_errors {
        println!("  constraint_error: {}", ce);
    }
    if let Some(gd) = &result.goal_details {
        println!("  goal_details:     {}", gd);
    }
}

// ── Meta-optimize command ──────────────────────────────────────────
//
// Reads trace JSON files from a chained curriculum run and synthesizes
// a priority heuristic that minimizes total candidates across all tasks.
//
// Usage:
//   selph meta-opt <tasks.selph> --library grown.selph --trace trace.json [--trace trace2.json]
//   selph meta-opt examples/full_curriculum.selph --library chain_output/stage3_nl.selph \
//     --trace chain_output/trace_seq.json --trace chain_output/trace_cf.json --trace chain_output/trace_nl.json

fn cmd_meta_optimize(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph meta-opt <tasks.selph> --library grown.selph --trace trace.json");
        return;
    }

    let mut task_files: Vec<String> = Vec::new();
    let mut library_paths: Vec<String> = Vec::new();
    let mut trace_paths: Vec<String> = Vec::new();
    let mut budget: usize = 10000;
    let mut depth: usize = 2;
    let mut synthesize_heuristic = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--library" | "--tree" => {
                if let Some(p) = args.get(i + 1) { library_paths.push(p.clone()); }
                i += 2;
            }
            "--trace" => {
                if let Some(p) = args.get(i + 1) { trace_paths.push(p.clone()); }
                i += 2;
            }
            "--budget" => {
                budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(budget);
                i += 2;
            }
            "--depth" => {
                depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(depth);
                i += 2;
            }
            "--synthesize" => { synthesize_heuristic = true; i += 1; }
            other => { task_files.push(other.to_string()); i += 1; }
        }
    }

    // Load trace files and parse them to extract task-level data
    let mut trace_tasks: Vec<TraceTask> = Vec::new();
    for tp in &trace_paths {
        match fs::read_to_string(tp) {
            Ok(json) => {
                let parsed = parse_trace_json(&json);
                eprintln!("Loaded {} task traces from {}", parsed.len(), tp);
                trace_tasks.extend(parsed);
            }
            Err(e) => eprintln!("Warning: couldn't read trace {}: {}", tp, e),
        }
    }

    if trace_tasks.is_empty() {
        eprintln!("No trace data found. Run a curriculum with --trace first.");
        return;
    }

    // Load task files to rebuild training examples
    let mut all_tasks: Vec<(String, usize, Vec<Value>, Vec<Value>)> = Vec::new();
    for tf in &task_files {
        match fs::read_to_string(tf) {
            Ok(s) => {
                let tasks = parse_curriculum_tasks(&s, depth);
                eprintln!("Loaded {} tasks from {}", tasks.len(), tf);
                all_tasks.extend(tasks);
            }
            Err(e) => eprintln!("Warning: couldn't read tasks {}: {}", tf, e),
        }
    }

    // Load libraries
    let mut all_macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    for lib_path in &library_paths {
        match fs::read_to_string(lib_path) {
            Ok(s) => {
                let lib_macros = load_library(&s);
                eprintln!("Loaded {} macros from {}", lib_macros.len(), lib_path);
                all_macros.extend(lib_macros);
            }
            Err(e) => eprintln!("Warning: couldn't load library {}: {}", lib_path, e),
        }
    }

    // Build training tasks from the curriculum tasks that appear in traces
    let trace_names: std::collections::HashSet<String> = trace_tasks.iter()
        .map(|t| t.task_name.clone())
        .collect();

    let training_tasks: Vec<meta::TrainingTask> = all_tasks.iter()
        .filter(|(name, _, _, _)| trace_names.contains(name))
        .map(|(name, _, inputs, expected)| {
            meta::TrainingTask {
                name: name.clone(),
                inputs: inputs.clone(),
                expected: expected.clone(),
            }
        })
        .collect();

    eprintln!();
    eprintln!("Meta-optimization Stage 3: Multi-task heuristic learning");
    eprintln!("  Training tasks: {} (from traces)", training_tasks.len());
    eprintln!("  Library: {} macros", all_macros.len());
    eprintln!("  Per-task budget: {}", budget);
    eprintln!();

    // Print baseline statistics from traces
    let total_cand: usize = trace_tasks.iter().map(|t| t.candidates).sum();
    let solved_count = trace_tasks.iter().filter(|t| t.solved).count();
    eprintln!("Baseline (from traces):");
    eprintln!("  Solved: {}/{}", solved_count, trace_tasks.len());
    eprintln!("  Total candidates: {}", total_cand);

    // Identify hard tasks (top 50% by candidate count)
    let mut by_difficulty: Vec<&TraceTask> = trace_tasks.iter()
        .filter(|t| t.solved && t.candidates > 100)
        .collect();
    by_difficulty.sort_by(|a, b| b.candidates.cmp(&a.candidates));

    eprintln!("  Hard tasks (>100 candidates):");
    for t in by_difficulty.iter().take(10) {
        eprintln!("    {:30} {:6} cand  strategy: {}",
            t.task_name, t.candidates,
            t.strategy.as_deref().unwrap_or("?"));
    }
    eprintln!();

    // Build synth components from the final library
    let synth_comps = synth::default_synth_components(&all_macros);

    // Run heuristic optimization
    eprintln!("Evaluating 8 candidate heuristics...");
    let default_h = meta::Heuristic::default_heuristic();

    // Evaluate baseline: run with default heuristic
    let (baseline_solved, baseline_cands, baseline_results) = run_meta_eval(
        &training_tasks, &default_h, &synth_comps, &all_macros, depth, budget);
    eprintln!("  Baseline: {}/{} solved, {} total candidates",
        baseline_solved, training_tasks.len(), baseline_cands);

    // Try each candidate heuristic
    let candidates = meta::build_candidate_heuristics_pub();
    let mut best_h: Option<meta::Heuristic> = None;
    let mut best_solved = baseline_solved;
    let mut best_cands = baseline_cands;

    for h in &candidates {
        let (h_solved, h_cands, h_results) = run_meta_eval(
            &training_tasks, h, &synth_comps, &all_macros, depth, budget);

        let is_better = h_solved > best_solved
            || (h_solved == best_solved && h_solved > 0 && h_cands < best_cands);

        let indicator = if is_better { " *** NEW BEST" } else { "" };
        eprintln!("  {}: {}/{} solved, {} candidates{}",
            h.name, h_solved, training_tasks.len(), h_cands, indicator);

        if is_better {
            best_solved = h_solved;
            best_cands = h_cands;
            best_h = Some(h.clone());

            // Print per-task improvements
            for ((name, _, _), (_, base_ok, base_c)) in h_results.iter().zip(baseline_results.iter()) {
                let (_, h_ok, h_c) = h_results.iter()
                    .find(|(n, _, _)| n == name).unwrap();
                if *h_c < *base_c && *base_c > 100 {
                    eprintln!("    {} -> {} cand ({} -> {})",
                        name, h_c, base_c, h_c);
                }
            }
        }
    }

    // ── Stage 4: Synthesize heuristics via enumeration ──────────────
    if synthesize_heuristic {
        eprintln!();
        eprintln!("Meta-optimization Stage 4: Synthesized heuristic search");
        eprintln!("  Running baseline with snapshot capture...");

        let (snap_solved, snap_cands, snap_results, snapshots) =
            run_meta_eval_with_snapshots(
                &training_tasks, &default_h, &synth_comps, &all_macros, depth, budget);
        eprintln!("  Captured {} snapshots from {}/{} solved tasks",
            snapshots.len(), snap_solved, training_tasks.len());

        if !snapshots.is_empty() {
            eprintln!();
            eprintln!("  Phase A: Enumerating heuristic programs (depth 2)...");
            let synth_candidates = meta::enumerate_heuristic_candidates(
                &snapshots, &synth_comps, 2, 20);

            if !synth_candidates.is_empty() {
                eprintln!();
                eprintln!("  Phase B: Validating top candidates with actual synthesis...");
                let phase_b_budget = budget.min(5000); // cap Phase B budget
                if let Some((synth_best, synth_solved, synth_total)) =
                    meta::validate_heuristic_candidates(
                        &synth_candidates,
                        &training_tasks,
                        &synth_comps,
                        &all_macros,
                        &snap_results,
                        depth,
                        phase_b_budget,
                        10,
                    )
                {
                    eprintln!();
                    eprintln!("  Stage 4 best: {}/{} solved (hard subset), {} candidates",
                        synth_solved, 15.min(training_tasks.len()), synth_total);
                    eprintln!("  Source: {}", synth_best.source);

                    // Compare with Stage 3 winner: run Stage 4 winner on full task set
                    let (s4_full_solved, s4_full_cands, _) = run_meta_eval(
                        &training_tasks, &synth_best, &synth_comps, &all_macros, depth, budget);

                    let s4_better = s4_full_solved > best_solved
                        || (s4_full_solved == best_solved && s4_full_cands < best_cands);

                    if s4_better {
                        eprintln!();
                        eprintln!("  *** Stage 4 heuristic beats Stage 3! ***");
                        eprintln!("  Stage 4: {}/{} solved, {} candidates",
                            s4_full_solved, training_tasks.len(), s4_full_cands);
                        if let Some(ref h3) = best_h {
                            eprintln!("  Stage 3: {}/{} solved, {} candidates (\"{}\")",
                                best_solved, training_tasks.len(), best_cands, h3.name);
                        }
                        best_h = Some(synth_best);
                        best_solved = s4_full_solved;
                        best_cands = s4_full_cands;
                    } else {
                        eprintln!();
                        eprintln!("  Stage 4 full eval: {}/{} solved, {} candidates",
                            s4_full_solved, training_tasks.len(), s4_full_cands);
                        eprintln!("  Stage 3 winner still better — keeping it.");
                    }
                } else {
                    eprintln!("  Phase B: no valid candidates passed validation.");
                }
            } else {
                eprintln!("  Phase A: no candidates enumerated (snapshots may be too small).");
            }
        } else {
            eprintln!("  No snapshots captured — cannot run Stage 4.");
        }
    }

    // ── Output final winner ──────────────────────────────────────────
    eprintln!();
    if let Some(ref h) = best_h {
        let speedup = baseline_cands as f64 / best_cands.max(1) as f64;
        let stage = if synthesize_heuristic { "Stage 3+4" } else { "Stage 3" };
        eprintln!("=== Best heuristic ({}) : \"{}\" ===", stage, h.name);
        eprintln!("  Solved: {}/{} (baseline: {})", best_solved, training_tasks.len(), baseline_solved);
        eprintln!("  Total candidates: {} (baseline: {}, {:.1}x speedup)",
            best_cands, baseline_cands, speedup);
        eprintln!("  Source: {}", h.source);
        eprintln!();
        eprintln!("To use: save as heuristic.selph and pass via --heuristic");

        // Output the heuristic as a SELPH file
        println!("; Meta-optimized heuristic: \"{}\"", h.name);
        println!("; Trained on {} tasks from chained curriculum", training_tasks.len());
        println!("; Speedup: {:.1}x ({} -> {} candidates)", speedup, baseline_cands, best_cands);
        println!("{}", h.source);
    } else {
        eprintln!("No heuristic improved on the baseline.");
        eprintln!("The default priority ordering is already near-optimal for this task suite.");
    }
}

/// Minimal parsed trace task for meta-optimization.
struct TraceTask {
    task_name: String,
    candidates: usize,
    solved: bool,
    strategy: Option<String>,
    components_used: Vec<String>,
}

/// Parse trace JSON manually (no serde dependency).
fn parse_trace_json(json: &str) -> Vec<TraceTask> {
    let mut tasks = Vec::new();

    // Split by task objects — look for "task_name" fields
    let mut pos = 0;
    while let Some(start) = json[pos..].find("\"task_name\"") {
        let abs_start = pos + start;
        // Find the task_name value
        let name = extract_json_string(json, abs_start);

        // Find total_candidates
        let candidates = if let Some(tc_start) = json[abs_start..].find("\"total_candidates\"") {
            extract_json_number(json, abs_start + tc_start) as usize
        } else { 0 };

        // Find solved
        let solved = if let Some(s_start) = json[abs_start..].find("\"solved\"") {
            let val_region = &json[abs_start + s_start..];
            val_region.contains("true") && !val_region[..20.min(val_region.len())].contains("false")
        } else { false };

        // Find solving_strategy
        let strategy = if let Some(ss_start) = json[abs_start..].find("\"solving_strategy\"") {
            let val_start = abs_start + ss_start;
            let val_region = &json[val_start..];
            if val_region.contains("null") && val_region.find("null").unwrap() < val_region.find('"').unwrap_or(999).min(30) {
                None
            } else {
                Some(extract_json_string(json, val_start))
            }
        } else { None };

        // Find components_used
        let mut components_used = Vec::new();
        if let Some(cu_start) = json[abs_start..].find("\"components_used\"") {
            let arr_start = abs_start + cu_start;
            if let Some(bracket) = json[arr_start..].find('[') {
                let arr_region_start = arr_start + bracket;
                if let Some(bracket_end) = json[arr_region_start..].find(']') {
                    let arr = &json[arr_region_start..arr_region_start + bracket_end];
                    for part in arr.split('"') {
                        let trimmed = part.trim().trim_matches(|c| c == ',' || c == '[' || c == ']' || c == ' ');
                        if !trimmed.is_empty() {
                            components_used.push(trimmed.to_string());
                        }
                    }
                }
            }
        }

        if !name.is_empty() {
            tasks.push(TraceTask {
                task_name: name,
                candidates,
                solved,
                strategy,
                components_used,
            });
        }

        pos = abs_start + 1;
    }

    tasks
}

/// Extract a JSON string value after a key at the given position.
fn extract_json_string(json: &str, key_pos: usize) -> String {
    // Find the colon after the key
    if let Some(colon) = json[key_pos..].find(':') {
        let after_colon = &json[key_pos + colon + 1..];
        // Find the opening quote
        if let Some(q1) = after_colon.find('"') {
            let after_q1 = &after_colon[q1 + 1..];
            // Find the closing quote (handle escapes simply)
            if let Some(q2) = after_q1.find('"') {
                return after_q1[..q2].to_string();
            }
        }
    }
    String::new()
}

/// Extract a JSON number value after a key at the given position.
fn extract_json_number(json: &str, key_pos: usize) -> f64 {
    if let Some(colon) = json[key_pos..].find(':') {
        let after_colon = json[key_pos + colon + 1..].trim_start();
        // Read until comma, }, or newline
        let end = after_colon.find(|c: char| c == ',' || c == '}' || c == '\n')
            .unwrap_or(after_colon.len());
        let num_str = after_colon[..end].trim();
        num_str.parse().unwrap_or(0.0)
    } else {
        0.0
    }
}

/// Run meta-evaluation: test all training tasks with a heuristic.
fn run_meta_eval(
    tasks: &[meta::TrainingTask],
    heuristic: &meta::Heuristic,
    components: &[synth::SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    per_task_budget: usize,
) -> (usize, usize, Vec<(String, bool, usize)>) {
    let mut solved = 0usize;
    let mut total_cands = 0usize;
    let mut results = Vec::new();

    for task in tasks {
        let ctx = meta::TaskContext::from_examples(&task.inputs, &task.expected);
        let prioritized = meta::apply_heuristic(heuristic, components, &ctx);

        let sr = synth::synthesize(
            &prioritized,
            &task.inputs,
            &task.expected,
            macros,
            max_depth,
            per_task_budget,
            true,
        );

        if sr.found { solved += 1; }
        total_cands += sr.candidates_explored;
        results.push((task.name.clone(), sr.found, sr.candidates_explored));
    }

    (solved, total_cands, results)
}

/// Like `run_meta_eval` but also captures PoolSnapshots for rank-based
/// heuristic synthesis (Stage 4).
fn run_meta_eval_with_snapshots(
    tasks: &[meta::TrainingTask],
    heuristic: &meta::Heuristic,
    components: &[synth::SynthComponent],
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    max_depth: usize,
    per_task_budget: usize,
) -> (usize, usize, Vec<(String, bool, usize)>, Vec<meta::PoolSnapshot>) {
    let mut solved = 0usize;
    let mut total_cands = 0usize;
    let mut results = Vec::new();
    let mut snapshots = Vec::new();

    for task in tasks {
        let ctx = meta::TaskContext::from_examples(&task.inputs, &task.expected);
        let prioritized = meta::apply_heuristic(heuristic, components, &ctx);

        let mut records: Vec<synth::CandidateRecord> = Vec::new();
        let sr = synth::synthesize_full(
            &prioritized,
            &task.inputs,
            &task.expected,
            macros,
            max_depth,
            per_task_budget,
            true,
            None,
            &[],
            None,
            Some(&mut records),
            synth::RlCoefficients::default(),
        );

        if sr.found {
            solved += 1;
            // Only capture snapshots for solved tasks (rank_solution needs a solution)
            if !records.is_empty() {
                snapshots.push(meta::PoolSnapshot {
                    task_name: task.name.clone(),
                    task_context: ctx,
                    candidates: records,
                    depth0_count: 0,
                });
            }
        }
        total_cands += sr.candidates_explored;
        results.push((task.name.clone(), sr.found, sr.candidates_explored));
    }

    (solved, total_cands, results, snapshots)
}

// ── Multi-synth command ─────────────────────────────────────────────

fn cmd_multi_synth(args: &[String]) {
    // Usage: selph multi-synth -e "1->2 3->6" --library ops.selph [--library data.selph]
    let mut examples_str: Option<String> = None;
    let mut library_paths: Vec<String> = Vec::new();
    let mut max_depth: usize = 3;
    let mut max_budget: usize = 200_000;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-e" => { examples_str = args.get(i + 1).cloned(); i += 2; }
            "--library" | "--tree" => {
                if let Some(p) = args.get(i + 1) { library_paths.push(p.clone()); }
                i += 2;
            }
            "--depth" => { max_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3); i += 2; }
            "--budget" => { max_budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(200_000); i += 2; }
            _ => { i += 1; }
        }
    }

    let ex_str = match examples_str {
        Some(s) => s,
        None => { eprintln!("Usage: selph multi-synth -e \"1->2 3->6\" --library ops.selph"); return; }
    };

    // Parse examples
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for pair_str in ex_str.split_whitespace() {
        if let Some((inp_s, out_s)) = pair_str.split_once("->") {
            let inp: Value = if let Ok(n) = inp_s.parse::<f64>() {
                Value::Num(n)
            } else {
                Value::Str(inp_s.to_string())
            };
            let out: Value = if let Ok(n) = out_s.parse::<f64>() {
                Value::Num(n)
            } else {
                Value::Str(out_s.to_string())
            };
            inputs.push(inp);
            expected.push(out);
        }
    }

    if inputs.is_empty() {
        eprintln!("No examples provided");
        return;
    }

    // Load libraries
    let mut macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    for lib_path in &library_paths {
        match fs::read_to_string(lib_path) {
            Ok(lib_source) => {
                let lib_macros = load_library(&lib_source);
                eprintln!("Loaded {} macros from {}", lib_macros.len(), lib_path);
                macros.extend(lib_macros);
            }
            Err(e) => eprintln!("Warning: couldn't load library {}: {}", lib_path, e),
        }
    }

    eprintln!("Multi-synth: {} examples, {} macros, depth {}, budget {}",
        inputs.len(), macros.len(), max_depth, max_budget);

    let synth_comps = synth::default_synth_components(&macros);
    let sr = synth::synthesize_with_validation(
        &synth_comps, &inputs, &expected, &macros,
        max_depth, max_budget, true, None, &[],
    );

    if sr.found {
        let source = node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap());
        println!("{}", source);
        eprintln!("Found in {} candidates", sr.candidates_explored);
    } else {
        eprintln!("No solution found ({} candidates explored)", sr.candidates_explored);
    }
}
