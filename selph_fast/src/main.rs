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
mod hm;
mod induce;
mod divide;
mod library;
mod verify;
mod abstraction;
mod multitree;
mod stochastic;
mod meta;
mod taskgen;

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
        "bench" => cmd_bench(&args[2..]),
        "generate" => cmd_generate(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "multi-synth" => cmd_multi_synth(&args[2..]),
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
    println!("    --tree <ns.selph>           Add namespace tree (repeatable)");
    println!("  selph grow <tasks.selph>      Run curriculum: solve, promote, save");
    println!("    --meta                      Enable meta-heuristic learning");
    println!("    --extract                   Enable abstraction extraction");
    println!("    --tree <ns.selph>           Add namespace tree (repeatable)");
    println!("  selph bench                   Run stochastic benchmark suite");
    println!("  selph generate                Generate a curriculum in .selph format");
    println!("  selph verify <prog> <spec>    Verify a program against a spec");
    println!("  selph multi-synth <spec>      Synthesize across multiple namespace trees");
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
        eprintln!("Usage: selph synth -e '1->2 3->6' [--depth N] [--budget N] [--library file.selph] [--tree ns.selph]");
        eprintln!("       selph synth spec.selph");
        return;
    }

    let mut max_depth: usize = 2;
    let mut max_candidates: usize = 100000;
    let mut library_path: Option<String> = None;
    let mut source: Option<String> = None;
    let mut tree_paths: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-e" => { source = Some(args.get(i + 1).cloned().unwrap_or_default()); i += 2; }
            "--depth" => { max_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2); i += 2; }
            "--budget" => { max_candidates = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(100000); i += 2; }
            "--library" => { library_path = args.get(i + 1).cloned(); i += 2; }
            "--tree" => {
                if let Some(p) = args.get(i + 1) { tree_paths.push(p.clone()); }
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

    // Load namespace trees
    let trees = load_namespace_trees(&tree_paths);
    if !trees.is_empty() {
        eprintln!("Loaded {} namespace trees", trees.len());
    }

    // Detect input type
    let input_is_string = matches!(&inputs[0], Value::Str(_));
    let _output_is_string = matches!(&expected[0], Value::Str(_));

    // Build components using synth module (includes comparisons for if-expressions)
    let (mut synth_comps, extra_bindings) = synth::default_synth_components_with_trees(&macros, &trees);
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

    // Run synthesis with if-expression support and tree bindings
    let start = std::time::Instant::now();
    let sr = synth::synthesize_with_validation(
        &synth_comps, &inputs, &expected, &macros,
        max_depth, max_candidates, true, None, &extra_bindings);
    let elapsed = start.elapsed();

    if sr.found {
        let source = node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap());
        println!("{}", source);
        eprintln!("Found in {} candidates ({:.3}s)", sr.candidates_explored, elapsed.as_secs_f64());

        // Trace which trees contributed
        if !trees.is_empty() {
            let tree_names: Vec<String> = trees.iter().map(|(n, _)| n.clone()).collect();
            let used = multitree::trace_solution(
                sr.nodes.as_ref().unwrap(), sr.root.unwrap(), &tree_names,
            );
            if !used.is_empty() {
                eprintln!("Trees used: {}", used.join(", "));
            }
        }
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

/// Load namespace trees from file paths.
///
/// Each file is evaluated; the last expression becomes the tree's value.
/// The tree name is derived from the file stem (e.g., "ops.selph" -> "ops").
fn load_namespace_trees(paths: &[String]) -> Vec<(String, Value)> {
    let mut trees = Vec::new();
    for (idx, path) in paths.iter().enumerate() {
        if let Ok(src) = fs::read_to_string(path) {
            if let Ok((nodes, roots)) = parse_file(&src) {
                let mut env = make_default_env();
                let mut last_val = Value::Nil;
                for &r in &roots {
                    match eval(&nodes, r, &mut env) {
                        Ok(v) => last_val = v,
                        Err(e) => { eprintln!("Error evaluating {}: {}", path, e); }
                    }
                }
                let tree_name = std::path::Path::new(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&format!("tree{}", idx))
                    .to_string();
                trees.push((tree_name, last_val));
            }
        } else {
            eprintln!("Warning: couldn't read tree file: {}", path);
        }
    }
    trees
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
        eprintln!("Usage: selph grow <tasks.selph> [--library base.selph] [--output grown.selph] [--tree ns.selph]");
        eprintln!("       selph grow <tasks.selph> --budget 200000 --depth 3");
        eprintln!("       selph grow <tasks.selph> --meta --extract");
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
    let mut enable_meta = false;
    let mut enable_extract = false;
    let mut enable_validate = false;
    let mut filter_path: Option<String> = None;
    let mut tree_paths: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--library" => { library_path = args.get(i + 1).cloned(); i += 2; }
            "--output" | "-o" => { output_path = args.get(i + 1).cloned().unwrap_or(output_path); i += 2; }
            "--budget" => { default_budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_budget); i += 2; }
            "--depth" => { default_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_depth); i += 2; }
            "--meta" => { enable_meta = true; i += 1; }
            "--extract" => { enable_extract = true; i += 1; }
            "--validate" => { enable_validate = true; i += 1; }
            "--filter" => { filter_path = args.get(i + 1).cloned(); i += 2; }
            "--tree" => {
                if let Some(p) = args.get(i + 1) { tree_paths.push(p.clone()); }
                i += 2;
            }
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

    // Load namespace trees (persist across all tasks)
    let trees = load_namespace_trees(&tree_paths);
    if !trees.is_empty() {
        eprintln!("Loaded {} namespace trees", trees.len());
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
    if !trees.is_empty() { eprintln!("  Trees: {}", trees.len()); }
    if enable_meta { eprintln!("  Meta-heuristic learning: enabled"); }
    if enable_extract { eprintln!("  Abstraction extraction: enabled"); }
    if enable_validate { eprintln!("  Validation: enabled (20% held out when >= 5 examples)"); }

    // Load SELPH depth filter if provided
    let depth_filter: Option<Box<dyn Fn(&synth::SynthComponent, usize) -> bool>> =
        if let Some(ref fp) = filter_path {
            match fs::read_to_string(fp) {
                Ok(src) => {
                    match parse_file(&src) {
                        Ok((nodes, roots)) if !roots.is_empty() => {
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

    // Meta-heuristic state — check if library contains a learned heuristic
    let mut current_heuristic: Option<meta::Heuristic> = None;
    let mut meta_training_tasks: Vec<meta::TrainingTask> = Vec::new();

    // Load learned heuristic from library if present
    for (mname, _params, mnodes, mroot) in &all_macros {
        if mname == "__selph_heuristic__" {
            let source = node_to_source(mnodes, *mroot);
            // Wrap back in lambda since defmacro strips it
            let full_source = format!("(lambda (ctx) {})", source);
            if let Some(h) = meta::Heuristic::from_source("loaded", &full_source) {
                eprintln!("  Loaded learned heuristic from library");
                current_heuristic = Some(h);
            }
            break;
        }
    }

    // Abstraction extraction state
    let mut solved_programs: Vec<(Vec<Node>, usize)> = Vec::new();

    for (name, task_depth, inputs, expected) in &tasks {
        // Use the greater of the task-specified depth and the CLI --depth,
        // so that CLI --depth 3 can unlock tasks that hardcode depth 2.
        let depth = std::cmp::max(*task_depth, default_depth);
        let input_is_string = matches!(&inputs[0], Value::Str(_));
        let (mut synth_comps, extra_bindings) = synth::default_synth_components_with_trees(&all_macros, &trees);
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

        // Apply learned priorities from previous solves
        for comp in &mut synth_comps {
            if let Some(&learned) = priorities.get(&comp.name) {
                comp.priority += learned;
            }
        }

        // Sort by learned priority — interleaved search handles the rest.
        // The priority sort ensures high-value components seed the pool first,
        // and the interleaved candidate generation (in synth.rs) sorts
        // compositions by combined priority of component + arguments.
        if enable_meta {
            if let Some(ref h) = current_heuristic {
                let task_ctx = meta::TaskContext::from_examples(inputs, expected);
                synth_comps = meta::apply_heuristic(h, &synth_comps, &task_ctx);
            } else {
                synth_comps.sort_by(|a, b| b.priority.partial_cmp(&a.priority)
                    .unwrap_or(std::cmp::Ordering::Equal));
            }
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

        let start = std::time::Instant::now();
        let filter_ref: Option<&dyn Fn(&synth::SynthComponent, usize) -> bool> =
            depth_filter.as_ref().map(|f| f.as_ref());
        let sr = synth::synthesize_full(
            &synth_comps, train_inputs, train_expected, &all_macros,
            depth, default_budget, true,
            val_pairs.as_deref(), &extra_bindings, filter_ref);
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
                }

                // Collect data for meta-heuristic and abstraction extraction
                if enable_meta {
                    meta_training_tasks.push(meta::TrainingTask {
                        name: name.clone(),
                        inputs: inputs.clone(),
                        expected: expected.clone(),
                    });
                }
                if enable_extract {
                    if let (Some(nodes), Some(root)) = (&sr.nodes, sr.root) {
                        solved_programs.push((nodes.clone(), root));
                    }
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

                // Collect data for meta-heuristic and abstraction extraction
                if enable_meta {
                    meta_training_tasks.push(meta::TrainingTask {
                        name: name.clone(),
                        inputs: inputs.clone(),
                        expected: expected.clone(),
                    });
                }
                if enable_extract {
                    solved_programs.push((ir.nodes.clone(), ir.root));
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

                    let body_source = extract_lambda_body(&source);
                    let macro_line = format!("(defmacro {} (s) {})", name, body_source);
                    promoted_source.push_str(&format!(
                        "\n; {} (D&C): found in {} candidates\n{}\n",
                        name, dr.candidates_explored, macro_line));

                    // Collect data for meta-heuristic and abstraction extraction
                    if enable_meta {
                        meta_training_tasks.push(meta::TrainingTask {
                            name: name.clone(),
                            inputs: inputs.clone(),
                            expected: expected.clone(),
                        });
                    }
                    if enable_extract {
                        solved_programs.push((dr.nodes.clone(), dr.root));
                    }
                } else {
                    let explored = sr.candidates_explored;
                    eprintln!("  --  {:30}  {:6} cand  {:.3}s",
                             name, explored, elapsed.as_secs_f64());
                }
            }
        }

        // Meta-heuristic synthesis: trigger on FAILURE, not periodically.
        // The priority learning (learn_rate boost) updates weights after every
        // solve — that's the cheap, per-task signal.  Structural meta-synthesis
        // (trying new heuristic templates) is expensive and only worth doing
        // when the current strategy fails.
        if enable_meta && !sr.found && !meta_training_tasks.is_empty() && meta_training_tasks.len() >= 3 {
            let base_comps = synth::default_synth_components(&all_macros);
            eprintln!("  [meta] Task failed — synthesizing heuristic from {} training tasks...", meta_training_tasks.len());
            if let Some(h) = meta::synthesize_heuristic(
                &meta_training_tasks, &base_comps, &all_macros, 8,
            ) {
                eprintln!("  [meta] Found heuristic: \"{}\" -- applying to subsequent tasks", h.name);
                current_heuristic = Some(h);
            } else {
                eprintln!("  [meta] No improved heuristic found");
            }
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
                                                if let Node::Symbol(s) = &mnodes[pi] { Some(s.clone()) }
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
                    "\n; [extracted] {} (freq={}, compression={:.1})\n{}\n",
                    abs.name, abs.frequency, abs.compression_score, macro_line));
            }
            if !abstractions.is_empty() {
                eprintln!("  [extract] Added {} abstractions to library", abstractions.len());
            }
        }
    }

    // End-of-run meta-heuristic synthesis — only if tasks failed AND
    // we don't already have a heuristic. Skip entirely if all tasks solved.
    let unsolved = tasks.len() - solved;
    if enable_meta && current_heuristic.is_none() && unsolved > 0 && !meta_training_tasks.is_empty() {
        let base_comps = synth::default_synth_components(&all_macros);
        eprintln!("  [meta] Final heuristic synthesis from {} training tasks ({} unsolved)...",
            meta_training_tasks.len(), unsolved);
        if let Some(h) = meta::synthesize_heuristic(
            &meta_training_tasks, &base_comps, &all_macros, 8,
        ) {
            eprintln!("  [meta] Found final heuristic: \"{}\"", h.name);
            current_heuristic = Some(h);
        } else {
            eprintln!("  [meta] No improved heuristic found at end of run");
        }
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

    // Save grown library — this IS the trained model.
    // Contains: base library + promoted solutions + extracted abstractions
    //         + learned heuristic (search strategy)
    let mut output = String::new();
    output.push_str("; SELPH library — trained model from curriculum runner\n");
    output.push_str(&format!("; {} macros ({} promoted from this run)\n",
                             all_macros.len(), solved));
    if current_heuristic.is_some() {
        output.push_str("; Includes learned search heuristic\n");
    }
    output.push_str("\n");

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

    // Save learned heuristic as a special macro.
    // Convention: __selph_heuristic__ is the learned search strategy.
    // When this library is loaded, the grow command detects it and uses it
    // as the initial heuristic for the next run.
    if let Some(ref h) = current_heuristic {
        output.push_str("\n; --- Learned search heuristic ---\n");
        output.push_str(&format!("; Heuristic: {} (learned during this curriculum run)\n", h.name));
        output.push_str(&format!("; This macro is the trained search strategy.\n"));
        output.push_str(&format!("; It scores components for priority-weighted interleaving.\n"));
        // Save heuristic as a defmacro. The heuristic source is a lambda like
        // (lambda (ctx) body). We need to extract the body for defmacro syntax.
        // Use the AST to do this correctly: evaluate the source to get the node tree,
        // then extract the lambda body.
        let heuristic_body = if h.nodes.len() > 0 {
            if let Node::Lambda(_, body_idx) = &h.nodes[h.root] {
                node_to_source(&h.nodes, *body_idx)
            } else {
                node_to_source(&h.nodes, h.root)
            }
        } else {
            h.source.clone()
        };
        output.push_str(&format!("(defmacro __selph_heuristic__ (ctx) {})\n", heuristic_body));
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
    let (nodes, roots) = match parse_file(&source) {
        Ok(r) => r,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };
    if roots.is_empty() { eprintln!("No expressions found"); return; }

    // Load library macros
    let mut env = make_default_env();
    if let Some(lib_path) = &library_path {
        if let Ok(lib_src) = fs::read_to_string(lib_path) {
            if let Ok((ln, lr)) = parse_file(&lib_src) {
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

// ── Multi-synth command ─────────────────────────────────────────────

fn cmd_multi_synth(args: &[String]) {
    // Usage: selph multi-synth -e "1->2 3->6" --tree ops.selph [--tree data.selph] [--library lib.selph]
    let mut examples_str: Option<String> = None;
    let mut tree_paths: Vec<String> = Vec::new();
    let mut library_path: Option<String> = None;
    let mut max_depth: usize = 3;
    let mut max_budget: usize = 200_000;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-e" => { examples_str = args.get(i + 1).cloned(); i += 2; }
            "--tree" => {
                if let Some(p) = args.get(i + 1) { tree_paths.push(p.clone()); }
                i += 2;
            }
            "--library" => { library_path = args.get(i + 1).cloned(); i += 2; }
            "--depth" => { max_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3); i += 2; }
            "--budget" => { max_budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(200_000); i += 2; }
            _ => { i += 1; }
        }
    }

    let ex_str = match examples_str {
        Some(s) => s,
        None => { eprintln!("Usage: selph multi-synth -e \"1->2 3->6\" --tree ops.selph"); return; }
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

    // Load library macros
    let mut macros: Vec<(String, Vec<String>, Vec<Node>, usize)> = Vec::new();
    if let Some(lib_path) = &library_path {
        if let Ok(lib_src) = fs::read_to_string(lib_path) {
            if let Ok((ln, lr)) = parse_file(&lib_src) {
                let mut lib_env = make_default_env();
                for &r in &lr {
                    let _ = eval(&ln, r, &mut lib_env);
                }
                // Extract macros via load_library
            }
            if let Ok(m) = library::load_library(lib_path) {
                macros = m;
            }
        }
    }

    // Load namespace trees using the shared helper
    let trees = load_namespace_trees(&tree_paths);

    eprintln!("Multi-tree synthesis: {} examples, {} trees, depth {}, budget {}",
        inputs.len(), trees.len(), max_depth, max_budget);

    // Use the core synthesizer with tree bindings injected
    let (synth_comps, extra_bindings) = synth::default_synth_components_with_trees(&macros, &trees);
    let sr = synth::synthesize_with_validation(
        &synth_comps, &inputs, &expected, &macros,
        max_depth, max_budget, true, None, &extra_bindings,
    );

    if sr.found {
        let source = node_to_source(sr.nodes.as_ref().unwrap(), sr.root.unwrap());
        println!("{}", source);
        eprintln!("Found in {} candidates", sr.candidates_explored);

        // Trace which trees contributed
        if !trees.is_empty() {
            let tree_names: Vec<String> = trees.iter().map(|(n, _)| n.clone()).collect();
            let used = multitree::trace_solution(
                sr.nodes.as_ref().unwrap(), sr.root.unwrap(), &tree_names,
            );
            if !used.is_empty() {
                eprintln!("Trees used: {}", used.join(", "));
            }
        }
    } else {
        eprintln!("No solution found ({} candidates explored)", sr.candidates_explored);
    }
}
