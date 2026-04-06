//! SELPH CLI — standalone binary for the SELPH language.
//!
//! Usage:
//!   selph eval <file.selph>           — evaluate a SELPH file
//!   selph eval -e "(add 1 2)"         — evaluate an expression
//!   selph repl                        — interactive REPL
//!   selph synth <spec.selph>          — synthesize from a spec file
//!   selph run <curriculum.selph>      — run a curriculum

// Import core modules (shared with the PyO3 library)
mod types;
mod parser;
mod eval;
mod namespace;
mod synth;
mod induce;
mod divide;
mod library;

use std::env;
use std::fs;
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
    println!("  selph grow <tasks.selph>      Run curriculum: solve, promote, save");
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
    let (nodes, roots) = match parse_file(&source) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };

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
            Ok((nodes, root)) => {
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
    let mut library_path: Option<String> = None;
    let mut source: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-e" => { source = Some(args.get(i + 1).cloned().unwrap_or_default()); i += 2; }
            "--depth" => { max_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2); i += 2; }
            "--budget" => { max_candidates = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(100000); i += 2; }
            "--library" => { library_path = args.get(i + 1).cloned(); i += 2; }
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

    // Parse examples: "1->2 3->6 5->10" or from a spec file
    let (inputs, expected) = parse_examples(&examples_str);
    if inputs.is_empty() {
        eprintln!("No valid examples found");
        return;
    }

    // Load library macros if specified
    let mut macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    if let Some(lib_path) = &library_path {
        match fs::read_to_string(lib_path) {
            Ok(lib_source) => {
                macros = load_library_macros(&lib_source);
                eprintln!("Loaded {} macros from {}", macros.len(), lib_path);
            }
            Err(e) => eprintln!("Warning: couldn't load library: {}", e),
        }
    }

    // Detect input type
    let input_is_string = matches!(&inputs[0], Value::Str(_));
    let output_is_string = matches!(&expected[0], Value::Str(_));

    // Build components using synth module (includes comparisons for if-expressions)
    let mut synth_comps = synth::default_synth_components(&macros);
    if input_is_string {
        for comp in &mut synth_comps {
            if comp.name == "x" { comp.ret_type = 1; }
        }
        for comp in &mut synth_comps {
            if comp.priority == 30.0 && comp.arity > 0 {
                comp.param_types = vec![1; comp.arity];
            }
        }
    }

    eprintln!("Synthesizing from {} examples, depth={}, budget={}, components={}",
              inputs.len(), max_depth, max_candidates, synth_comps.len());

    // Run synthesis with if-expression support
    let start = std::time::Instant::now();
    let sr = synth::synthesize(
        &synth_comps, &inputs, &expected, &macros,
        max_depth, max_candidates, true);
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
            // Try parsing as number
            if let Ok(n) = s.parse::<f64>() {
                Some(Value::Num(n))
            } else {
                Some(Value::Str(s.clone()))
            }
        }
        _ => None,
    }
}

fn load_library_macros(source: &str) -> Vec<(String, Vec<String>, Vec<Node>, usize)> {
    let mut macros = Vec::new();

    let (nodes, roots) = match parse_file(source) {
        Ok(r) => r,
        Err(_) => return macros,
    };

    for &root in &roots {
        if let Node::App(children) = &nodes[root] {
            if children.len() == 4 {
                if let Node::Symbol(s) = &nodes[children[0]] {
                    if s == "defmacro" {
                        if let Node::Symbol(name) = &nodes[children[1]] {
                            if let Node::App(param_indices) = &nodes[children[2]] {
                                let params: Vec<String> = param_indices.iter()
                                    .filter_map(|&i| {
                                        if let Node::Symbol(s) = &nodes[i] { Some(s.clone()) }
                                        else { None }
                                    }).collect();
                                // Clone the body subtree
                                macros.push((
                                    name.clone(), params,
                                    nodes.clone(), children[3],
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
        SynthComponent { name: "5".into(), builtin: None, arity: 0, ret_type: 0, param_types: vec![], priority: 0.0 },
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
        ln.push(Node::Lambda(vec!["x".into()], entry.root));
        let mut beh = Vec::new();
        let mut ok = true;
        for (inp, exp) in inputs.iter().zip(expected.iter()) {
            let mut env = make_default_env();
            for (nm, ps, mn, mr) in &macro_env {
                env_define(&mut env, nm.clone(), Value::RustMacro(ps.clone(), mn.clone(), *mr));
            }
            let fv = match eval(&ln, lr, &mut env) { Ok(v) => v, Err(_) => { ok = false; break; } };
            match apply(&fv, &[inp.clone()], &ln, &mut env) {
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
        if comp.name == "x" { nodes.push(Node::Symbol("x".into())); }
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

            if comp.arity == 1 {
                for pi in prev.clone() {
                    let p = pool[pi].clone();
                    if p.ret_type != comp.param_types[0] && comp.param_types[0] != 255 { continue; }
                    let mut n = p.nodes.clone();
                    let fi = n.len(); n.push(Node::Symbol(bn.clone()));
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
                        let fi = n.len(); n.push(Node::Symbol(bn.clone()));
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
                        let fi = n.len(); n.push(Node::Symbol(bn.clone()));
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
            bs.iter().map(|(n, i)| (n.clone(), i + offset)).collect(),
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
        _ => false,
    }
}

fn cmd_curriculum(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph grow <tasks.selph> [--library base.selph] [--output grown.selph]");
        eprintln!("       selph grow <tasks.selph> --budget 200000 --depth 3");
        eprintln!();
        eprintln!("Task file format:");
        eprintln!("  (task \"name\" depth");
        eprintln!("    (\"input1\" expected1)");
        eprintln!("    (\"input2\" expected2))");
        return;
    }

    let mut task_file = String::new();
    let mut library_path: Option<String> = None;
    let mut output_path = String::from("grown_library.selph");
    let mut default_budget: usize = 200000;
    let mut default_depth: usize = 3;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--library" => { library_path = args.get(i + 1).cloned(); i += 2; }
            "--output" | "-o" => { output_path = args.get(i + 1).cloned().unwrap_or(output_path); i += 2; }
            "--budget" => { default_budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_budget); i += 2; }
            "--depth" => { default_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_depth); i += 2; }
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

    // Load base library
    let mut all_macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    let mut library_source = String::new();

    if let Some(lib_path) = &library_path {
        match fs::read_to_string(lib_path) {
            Ok(s) => {
                library_source = s.clone();
                all_macros = load_library_macros(&s);
                eprintln!("Loaded {} macros from {}", all_macros.len(), lib_path);
            }
            Err(e) => eprintln!("Warning: couldn't load library: {}", e),
        }
    }

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

    for (name, depth, inputs, expected) in &tasks {
        let input_is_string = matches!(&inputs[0], Value::Str(_));
        let mut components = default_synth_components(&all_macros);
        if input_is_string {
            for comp in &mut components {
                if comp.name == "x" { comp.ret_type = 1; }
            }
            for comp in &mut components {
                if comp.priority == 30.0 && comp.arity > 0 {
                    comp.param_types = vec![1; comp.arity];
                }
            }
        }

        // Apply learned priorities from previous solves
        let mut synth_comps: Vec<synth::SynthComponent> = components.iter().map(|c| {
            let learned_priority = priorities.get(&c.name).copied().unwrap_or(0.0);
            synth::SynthComponent {
                name: c.name.clone(), builtin: c.builtin.clone(),
                arity: c.arity, ret_type: c.ret_type,
                param_types: c.param_types.clone(),
                priority: c.priority + learned_priority,
            }
        }).collect();

        // Sort by priority so high-priority components are tried first
        synth_comps.sort_by(|a, b| b.priority.partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal));

        let start = std::time::Instant::now();
        let sr = synth::synthesize(
            &synth_comps, inputs, expected, &all_macros,
            *depth, default_budget, true);
        let elapsed = start.elapsed();
        total_candidates += sr.candidates_explored;

        if sr.found {
                let source = node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap());
                solved += 1;
                let explored = sr.candidates_explored;
                eprintln!("  OK  {:30}  {:6} cand  {:.3}s  {}",
                         name, explored, elapsed.as_secs_f64(), source);

                // Update priorities: boost components that appeared in the solution
                let used_components = library::extract_components(&source);
                for comp_name in &used_components {
                    let entry = priorities.entry(comp_name.clone()).or_insert(0.0);
                    *entry += learn_rate;
                }

                // Extract body from lambda for the macro
                let body_source = extract_lambda_body(&source);
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
                                                if let Node::Symbol(s) = &mnodes[i] { Some(s.clone()) }
                                                else { None }
                                            }).collect();
                                        all_macros.push((
                                            mname.clone(), params,
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
        } else {
            // Fallback 1: Try induction
            let ir = induce::induce_from_failure(
                &synth_comps, inputs, expected, &all_macros, *depth, default_budget / 2);

            if ir.found {
                let source = node_to_source(&ir.nodes, ir.root);
                solved += 1;
                total_candidates += ir.candidates_explored;
                eprintln!("  IN  {:30}  {:6} cand  {:.3}s  {}",
                         name, ir.candidates_explored, elapsed.as_secs_f64(), source);

                let used_components = library::extract_components(&source);
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
                                                if let Node::Symbol(s) = &mnodes[i] { Some(s.clone()) }
                                                else { None }
                                            }).collect();
                                        all_macros.push((mname.clone(), params, mnodes.clone(), children[3]));
                                    }
                                }
                            }
                        }
                    }
                }
                promoted_source.push_str(&format!(
                    "\n; {} (induced): found in {} candidates\n{}\n",
                    name, ir.candidates_explored, macro_line));
            } else {
                // Fallback 2: Try divide-and-conquer
                let dr = divide::divide_and_conquer(
                    &synth_comps, inputs, expected, &all_macros, *depth, default_budget / 2);

                if dr.found {
                    let source = node_to_source(&dr.nodes, dr.root);
                    solved += 1;
                    total_candidates += dr.candidates_explored;
                    eprintln!("  DC  {:30}  {:6} cand  {:.3}s  {}",
                             name, dr.candidates_explored, elapsed.as_secs_f64(), source);

                    let body_source = extract_lambda_body(&source);
                    let macro_line = format!("(defmacro {} (s) {})", name, body_source);
                    promoted_source.push_str(&format!(
                        "\n; {} (D&C): found in {} candidates\n{}\n",
                        name, dr.candidates_explored, macro_line));
                } else {
                    let explored = sr.candidates_explored;
                    eprintln!("  --  {:30}  {:6} cand  {:.3}s",
                             name, explored, elapsed.as_secs_f64());
                }
            }
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

    // Save grown library
    let mut output = String::new();
    output.push_str("; SELPH library — auto-generated by curriculum runner\n");
    output.push_str(&format!("; {} macros ({} promoted from this run)\n\n",
                             all_macros.len(), solved));

    // Include base library
    if !library_source.is_empty() {
        output.push_str("; --- Base library ---\n");
        output.push_str(&library_source);
        output.push_str("\n");
    }

    // Add promoted solutions
    if !promoted_source.is_empty() {
        output.push_str("; --- Promoted solutions ---");
        output.push_str(&promoted_source);
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
                if s != "task" { continue; }
            } else { continue; }

            let name = match &nodes[children[1]] {
                Node::Str(s) => s.clone(),
                Node::Symbol(s) => s.clone(),
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
                // Replace the param name (x) with s for macro consistency
                return body
                    .replace("(x)", "(s)")
                    .replace("(x ", "(s ")
                    .replace(" x)", " s)")
                    .replace(" x ", " s ");
            }
        }
    }
    source.to_string()
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
