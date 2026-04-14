//! SELPH CLI — standalone binary for the SELPH language.
//!
//! Usage:
//!   selph eval <file.selph>           — evaluate a SELPH file
//!   selph eval -e "(add 1 2)"         — evaluate an expression
//!   selph repl                        — interactive REPL
//!   selph synth <spec.selph>          — synthesize from a spec file
//!   selph run <curriculum.selph>      — run a curriculum

// Core modules
mod intern;
mod types;
mod parser;
mod arc;
mod types_v2;
mod eval_v2;
mod synth_v2;
mod meta_v2;
mod ast_tools;

use std::env;
use std::fs;
use std::rc::Rc;
use std::collections::HashSet;
use intern::{intern, resolve};
use types::*;
use parser::*;

// ── Checkpoint infrastructure ─────────────────────────────────────────

/// A solved task loaded from checkpoint or reported by a worker.
struct SolvedTask {
    name: String,
    strategy: String,
    candidates: usize,
    source: String, // SELPH lambda from node_to_source
}

/// Load checkpoint file. Returns empty vec if file doesn't exist.
fn load_checkpoint(path: &str) -> Vec<SolvedTask> {
    let content = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.splitn(4, '\t').collect();
        if parts.len() < 4 { continue; }
        out.push(SolvedTask {
            name: parts[0].to_string(),
            strategy: parts[1].to_string(),
            candidates: parts[2].parse().unwrap_or(0),
            source: parts[3].to_string(),
        });
    }
    out
}

/// Save checkpoint atomically (write .tmp, rename).
fn save_checkpoint(path: &str, solved: &[SolvedTask], preamble_hash: u64) {
    let tmp = format!("{}.tmp", path);
    let mut content = format!("#preamble-hash:{:016x}\n", preamble_hash);
    for st in solved {
        content.push_str(&st.name);
        content.push('\t');
        content.push_str(&st.strategy);
        content.push('\t');
        content.push_str(&st.candidates.to_string());
        content.push('\t');
        content.push_str(&st.source);
        content.push('\n');
    }
    if fs::write(&tmp, &content).is_ok() {
        let _ = fs::rename(&tmp, path);
    }
}

/// Reconstruct a Value::Function from a SELPH source string by parsing and
/// evaluating it. Same pipeline as eval_curriculum_preamble.
fn reconstruct_solution(source: &str, env: &types_v2::Env) -> Result<types_v2::Value, String> {
    let (legacy_nodes, roots) = parse_file(source).map_err(|e| format!("{}", e))?;
    if roots.is_empty() {
        return Err("empty source".into());
    }
    let v2_nodes_vec = eval_v2::convert_tree(&legacy_nodes);
    let v2_nodes: Rc<[types_v2::Node]> = v2_nodes_vec.into();
    eval_v2::eval(&v2_nodes, roots[0], env)
}

/// Compute a hash of the non-task portion of the curriculum source.
/// Used to detect when the M-chain or other preamble definitions change.
fn compute_preamble_hash(source: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let task_sym = intern("task");
    let task_args_sym = intern("task-args");
    let (nodes, roots) = match parse_file(source) {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let mut hasher = DefaultHasher::new();
    for &root in &roots {
        let is_task = match &nodes[root] {
            Node::App(children) if !children.is_empty() => {
                if let Node::Symbol(s) = &nodes[children[0]] {
                    *s == task_sym || *s == task_args_sym
                } else { false }
            }
            _ => false,
        };
        if !is_task {
            // Hash the source representation of this preamble form.
            types::node_to_source(&nodes, root).hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn main() {
    // Tree-walking eval can recurse deeply with large SELPH environments
    // (e.g., M-chain + post-mortem). Spawn on a thread with 64MB stack.
    let stack_size = std::env::var("RUST_MIN_STACK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64 * 1024 * 1024);
    let builder = std::thread::Builder::new().stack_size(stack_size);
    let handler = builder.spawn(real_main).expect("failed to spawn main thread");
    if let Err(e) = handler.join() {
        std::panic::resume_unwind(e);
    }
}

fn real_main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "eval" | "eval-v2" => cmd_eval(&args[2..]),
        "repl" => cmd_repl(),
        "parse" => cmd_parse(&args[2..]),
        "grow" | "grow-v2" => cmd_grow_v2(&args[2..]),
        "arc" => cmd_arc(&args[2..]),
        "beam-overnight" => cmd_beam_overnight(&args[2..]),
        "ast-query" => cmd_ast_query(&args[2..]),
        "ast-edit" => cmd_ast_edit(&args[2..]),
        "fmt" => cmd_fmt(&args[2..]),
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
    println!("  selph grow <tasks.selph>      Run curriculum: solve, promote, save");
    println!("    --budget N                  Max candidates per task (default 200000)");
    println!("    --depth N                   Strategy/decomposer chaining depth (default 2)");
    println!("    --flat-depth N              Flat enumeration depth per leaf (default 1)");
    println!("    --post-mortem <pm.selph>    Post-mortem analysis script");
    println!("  selph arc <path> [--output f] Convert ARC JSON to curriculum format");
    println!("  selph repl                    Interactive REPL");
    println!("  selph ast-query <file> --list-defs         List top-level definitions");
    println!("  selph ast-query <file> --tree <idx>        Show definition as indexed tree");
    println!("  selph ast-edit <file> --replace <idx> '<expr>'   Replace node at index");
    println!("  selph ast-edit <file> --wrap-let <idx> <name>    Wrap node in let binding");
    println!("  selph ast-edit <file> --insert-def <name> --params '<p1 p2>' --body '<expr>'");
    println!("  selph ast-edit <file> --delete-def <name>        Delete a definition");
    println!("  selph fmt <file> [--in-place]              Format SELPH source");
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

    let (old_nodes, roots) = match parse_file(&source) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };
    let new_nodes_vec = eval_v2::convert_tree(&old_nodes);
    let new_nodes: Rc<[types_v2::Node]> = new_nodes_vec.into();

    let env = eval_v2::make_default_env();
    for root in &roots {
        match eval_v2::eval(&new_nodes, *root, &env) {
            Ok(val) => {
                let s = eval_v2::value_to_string(&val);
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

fn cmd_ast_query(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph ast-query <file> --list-defs | --tree <idx>");
        return;
    }

    let source = match fs::read_to_string(&args[0]) {
        Ok(s) => s,
        Err(e) => { eprintln!("Error reading {}: {}", args[0], e); return; }
    };
    let (nodes, roots) = match parse_file(&source) {
        Ok(r) => r,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };

    if args.len() < 2 {
        eprintln!("Expected --list-defs or --tree <idx>");
        return;
    }

    match args[1].as_str() {
        "--list-defs" => {
            let defs = ast_tools::list_defs(&nodes, &roots);
            for d in &defs {
                println!("{}\t{}\t{}\t[{}]", d.order, d.kind, d.name, d.root_idx);
            }
            if defs.is_empty() {
                println!("(no definitions found; {} top-level forms)", roots.len());
            }
        }
        "--tree" => {
            if args.len() < 3 {
                eprintln!("Usage: selph ast-query <file> --tree <def-index>");
                return;
            }
            let idx: usize = match args[2].parse() {
                Ok(n) => n,
                Err(_) => {
                    // Try to find by name
                    let defs = ast_tools::list_defs(&nodes, &roots);
                    match defs.iter().find(|d| d.name == args[2]) {
                        Some(d) => d.root_idx,
                        None => { eprintln!("'{}' is not a valid index or definition name", args[2]); return; }
                    }
                }
            };
            if idx >= nodes.len() {
                eprintln!("Index {} out of range (arena has {} nodes)", idx, nodes.len());
                return;
            }
            print!("{}", ast_tools::tree_display(&nodes, idx));
        }
        other => eprintln!("Unknown flag '{}'; expected --list-defs or --tree", other),
    }
}

fn cmd_ast_edit(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph ast-edit <file> --replace|--wrap-let|--insert-def|--delete-def ...");
        return;
    }

    let source = match fs::read_to_string(&args[0]) {
        Ok(s) => s,
        Err(e) => { eprintln!("Error reading {}: {}", args[0], e); return; }
    };
    let (mut nodes, mut roots) = match parse_file(&source) {
        Ok(r) => r,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };

    if args.len() < 2 {
        eprintln!("Expected an edit operation");
        return;
    }

    // Check for --in-place flag anywhere in args
    let in_place = args.iter().any(|a| a == "--in-place" || a == "-i");

    let result = match args[1].as_str() {
        "--replace" => {
            if args.len() < 4 {
                eprintln!("Usage: selph ast-edit <file> --replace <node-idx> '<expr>'");
                return;
            }
            let idx: usize = match args[2].parse() {
                Ok(n) => n,
                Err(_) => { eprintln!("Invalid index: {}", args[2]); return; }
            };
            ast_tools::replace_node(&mut nodes, &mut roots, idx, &args[3])
        }
        "--wrap-let" => {
            if args.len() < 4 {
                eprintln!("Usage: selph ast-edit <file> --wrap-let <node-idx> <name>");
                return;
            }
            let idx: usize = match args[2].parse() {
                Ok(n) => n,
                Err(_) => { eprintln!("Invalid index: {}", args[2]); return; }
            };
            ast_tools::wrap_let(&mut nodes, &mut roots, idx, &args[3])
        }
        "--insert-def" => {
            if args.len() < 3 {
                eprintln!("Usage: selph ast-edit <file> --insert-def <name> --params '<p1 p2>' --body '<expr>'");
                return;
            }
            let name = &args[2];
            let mut params_str = "";
            let mut body_str = "";
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--params" if i + 1 < args.len() => { params_str = &args[i + 1]; i += 2; }
                    "--body" if i + 1 < args.len() => { body_str = &args[i + 1]; i += 2; }
                    _ => { i += 1; }
                }
            }
            if body_str.is_empty() {
                eprintln!("Missing --body argument");
                return;
            }
            let params: Vec<&str> = if params_str.is_empty() {
                vec![]
            } else {
                params_str.split_whitespace().collect()
            };
            ast_tools::insert_def(&mut nodes, &mut roots, name, &params, body_str)
        }
        "--delete-def" => {
            if args.len() < 3 {
                eprintln!("Usage: selph ast-edit <file> --delete-def <name>");
                return;
            }
            ast_tools::delete_def(&nodes, &mut roots, &args[2])
        }
        other => { eprintln!("Unknown edit operation '{}'", other); return; }
    };

    match result {
        Ok(()) => {
            let output = ast_tools::emit_source(&nodes, &roots);
            if in_place {
                if let Err(e) = fs::write(&args[0], &output) {
                    eprintln!("Error writing {}: {}", args[0], e);
                }
            } else {
                print!("{}", output);
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn cmd_fmt(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph fmt <file> [--in-place]");
        return;
    }

    let source = match fs::read_to_string(&args[0]) {
        Ok(s) => s,
        Err(e) => { eprintln!("Error reading {}: {}", args[0], e); return; }
    };
    let (nodes, roots) = match parse_file(&source) {
        Ok(r) => r,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };

    let output = ast_tools::fmt_source(&nodes, &roots);
    let in_place = args.iter().any(|a| a == "--in-place" || a == "-i");
    if in_place {
        if let Err(e) = fs::write(&args[0], &output) {
            eprintln!("Error writing {}: {}", args[0], e);
        }
    } else {
        print!("{}", output);
    }
}

fn cmd_repl() {
    use std::io::{self, Write, BufRead};

    println!("SELPH REPL (type :quit to exit)");
    let env = eval_v2::make_default_env();

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
            Ok((old_nodes, root)) => {
                let new_nodes_vec = eval_v2::convert_tree(&old_nodes);
                let new_nodes: Rc<[types_v2::Node]> = new_nodes_vec.into();
                match eval_v2::eval(&new_nodes, root, &env) {
                    Ok(val) => {
                        let s = eval_v2::value_to_string(&val);
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

// ── Parallel wavefront infrastructure ──────────────────────────────────

/// Result reported by a worker process for a single task (via pipe).
/// Tab-delimited: name\tfound\tstrategy\tcandidates\tfitness\toutput_type\tm_chain_ran\tbest_source\tsource
struct WorkerResult {
    name: String,
    found: bool,
    strategy: String,
    candidates: usize,
    fitness: f64,
    output_type: String,
    m_chain_ran: bool,
    best_source: String, // empty if none
    source: String,      // lambda source if found, empty otherwise
}

impl WorkerResult {
    fn to_line(&self) -> String {
        format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.name,
            if self.found { "1" } else { "0" },
            self.strategy,
            self.candidates,
            self.fitness,
            self.output_type,
            if self.m_chain_ran { "1" } else { "0" },
            self.best_source,
            self.source,
        )
    }

    fn from_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.splitn(9, '\t').collect();
        if parts.len() < 9 { return None; }
        Some(WorkerResult {
            name: parts[0].to_string(),
            found: parts[1] == "1",
            strategy: parts[2].to_string(),
            candidates: parts[3].parse().unwrap_or(0),
            fitness: parts[4].parse().unwrap_or(0.0),
            output_type: parts[5].to_string(),
            m_chain_ran: parts[6] == "1",
            best_source: parts[7].to_string(),
            source: parts[8].to_string(),
        })
    }
}

/// Synthesize a single task and return a WorkerResult.
/// Used by both sequential and parallel paths.
fn synthesize_one_task(
    name: &str,
    task_depth: usize,
    inputs_legacy: &[Value],
    expected_legacy: &[Value],
    arity_hint: Option<usize>,
    test_inputs_legacy: &[Value],
    test_expected_legacy: &[Value],
    env: &types_v2::Env,
    default_budget: usize,
    strategy_depth: usize,
) -> WorkerResult {
    let force_num = inputs_legacy.iter().chain(expected_legacy.iter())
        .any(legacy_value_has_non_integral);
    let convert = |v: &Value| -> types_v2::Value {
        if force_num { legacy_value_to_v2_force_num(v) }
        else { legacy_value_to_v2(v) }
    };
    let inputs: Vec<types_v2::Value> = inputs_legacy.iter().map(&convert).collect();
    let expected: Vec<types_v2::Value> = expected_legacy.iter().map(&convert).collect();
    let test_inputs: Vec<types_v2::Value> = test_inputs_legacy.iter().map(&convert).collect();
    let test_expected: Vec<types_v2::Value> = test_expected_legacy.iter().map(&convert).collect();

    let skip = synth_v2::default_skip_set();
    let components = synth_v2::default_synth_components(env, &skip);
    let universe = synth_v2::TypeUniverse::from_env(env);

    let result = if let Some(arity) = arity_hint {
        let first = match inputs.first() {
            Some(types_v2::Value::List(items)) if items.len() == arity => items.clone(),
            _ => {
                return WorkerResult {
                    name: name.to_string(), found: false, strategy: String::new(),
                    candidates: 0, fitness: 0.0, output_type: String::new(),
                    m_chain_ran: false, best_source: String::new(), source: String::new(),
                };
            }
        };
        let arg_types: Vec<crate::intern::Sym> = first.iter()
            .map(|v| v.type_sym().unwrap_or_else(types_v2::type_any))
            .collect();
        let synth_result = synth_v2::synthesize_args_with_test(
            &components, &inputs, &arg_types, &expected,
            env, &universe, task_depth, default_budget,
            &test_inputs, &test_expected,
        );
        let strategy = if synth_result.found {
            Some(match synth_result.decomposer_name {
                Some(n) => synth_v2::Strategy::Custom(n),
                None => synth_v2::Strategy::Flat,
            })
        } else { None };
        let output_type = synth_v2::infer_uniform_type_sym(&expected)
            .map(|s| crate::intern::resolve(s))
            .unwrap_or_else(|| "Mixed".to_string());
        let has_decomposers = matches!(
            env.lookup(intern("__decomposers__")),
            Some(types_v2::Value::Ns(ref m)) if !m.is_empty()
        );
        let best_source = if !synth_result.found {
            synth_result.best_nodes.as_ref().map(|nodes| {
                let root = synth_result.best_root.unwrap_or(0);
                eval_v2::node_to_source(nodes, root)
            })
        } else { None };
        synth_v2::StrategyResult {
            found: synth_result.found,
            nodes: synth_result.nodes,
            root: synth_result.root,
            candidates_explored: synth_result.candidates_explored,
            strategy,
            output_type: Some(output_type),
            m_chain_ran: has_decomposers,
            best_fitness: synth_result.best_fitness,
            best_source,
            beam: Vec::new(),
            experience: Vec::new(),
        }
    } else {
        synth_v2::synthesize_with_strategies(
            &components, &inputs, &expected,
            env, &universe, task_depth, default_budget, strategy_depth,
        )
    };

    let strategy_name = result.strategy
        .map(|s| s.name())
        .unwrap_or_default();
    let output_type = result.output_type.clone().unwrap_or_default();
    let source = if result.found {
        let nodes_vec = result.nodes.as_ref().expect("found implies nodes");
        let root = result.root.expect("found implies root");
        eval_v2::node_to_source(nodes_vec, root)
    } else {
        String::new()
    };

    WorkerResult {
        name: name.to_string(),
        found: result.found,
        strategy: strategy_name,
        candidates: result.candidates_explored,
        fitness: result.best_fitness,
        output_type,
        m_chain_ran: result.m_chain_ran,
        best_source: result.best_source.unwrap_or_default(),
        source,
    }
}

/// Run a parallel wavefront: fork N workers, each synthesizes a subset of tasks.
/// Returns WorkerResults for all tasks in the batch.
fn run_parallel_batch(
    task_indices: &[usize],
    tasks: &[(String, usize, Vec<Value>, Vec<Value>, Option<usize>, Vec<Value>, Vec<Value>)],
    env: &types_v2::Env,
    default_budget: usize,
    n_workers: usize,
    strategy_depth: usize,
) -> Vec<WorkerResult> {
    use std::io::{BufRead, BufReader, Write as IoWrite};

    // Partition tasks round-robin across workers.
    let actual_workers = n_workers.min(task_indices.len()).max(1);
    let mut chunks: Vec<Vec<usize>> = (0..actual_workers).map(|_| Vec::new()).collect();
    for (i, &idx) in task_indices.iter().enumerate() {
        chunks[i % actual_workers].push(idx);
    }

    let mut child_pids: Vec<(libc::pid_t, std::os::unix::io::RawFd)> = Vec::new();

    for chunk in &chunks {
        if chunk.is_empty() { continue; }

        // Create pipe: [read_fd, write_fd]
        let mut fds = [0i32; 2];
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            eprintln!("  parallel: pipe() failed, falling back");
            continue;
        }
        let (read_fd, write_fd) = (fds[0], fds[1]);

        match unsafe { libc::fork() } {
            -1 => {
                eprintln!("  parallel: fork() failed");
                unsafe { libc::close(read_fd); libc::close(write_fd); }
            }
            0 => {
                // ── Child process ──
                unsafe { libc::close(read_fd); }

                // Convert raw fd to a File for buffered writing.
                let mut file: std::fs::File = unsafe {
                    std::os::unix::io::FromRawFd::from_raw_fd(write_fd)
                };

                for &task_idx in chunk {
                    let (ref name, task_depth, ref inputs, ref expected,
                         arity_hint, ref test_in, ref test_exp) = tasks[task_idx];
                    let wr = synthesize_one_task(
                        name, task_depth, inputs, expected,
                        arity_hint, test_in, test_exp,
                        env, default_budget, strategy_depth,
                    );
                    let line = wr.to_line();
                    let _ = writeln!(file, "{}", line);
                }
                drop(file); // Close write end before _exit.
                // Exit without running destructors (avoid double-free of Rc state).
                unsafe { libc::_exit(0); }
            }
            pid => {
                // ── Parent process ──
                unsafe { libc::close(write_fd); }
                child_pids.push((pid, read_fd));
            }
        }
    }

    // Read results from all child pipes concurrently (threads in parent).
    let handles: Vec<_> = child_pids.iter().map(|&(_pid, read_fd)| {
        std::thread::spawn(move || {
            let file = unsafe {
                <std::fs::File as std::os::unix::io::FromRawFd>::from_raw_fd(read_fd)
            };
            let reader = BufReader::new(file);
            let mut results = Vec::new();
            for line in reader.lines() {
                if let Ok(line) = line {
                    if let Some(wr) = WorkerResult::from_line(&line) {
                        results.push(wr);
                    }
                }
            }
            results
        })
    }).collect();

    let mut all_results = Vec::new();
    for handle in handles {
        if let Ok(results) = handle.join() {
            all_results.extend(results);
        }
    }

    // Wait for all children.
    for &(pid, _) in &child_pids {
        let mut status = 0i32;
        unsafe { libc::waitpid(pid, &mut status, 0); }
    }

    all_results
}

fn cmd_grow_v2(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph grow-v2 <tasks.selph> [--budget N] [--depth N] [--flat-depth N] [--parallel N] [--checkpoint PATH] [--no-checkpoint]");
        return;
    }

    let mut task_file = String::new();
    let mut default_budget: usize = 200000;
    let mut default_flat_depth: usize = 1;
    let mut default_strategy_depth: usize = 2;
    let mut post_mortem_file: Option<String> = None;
    let mut checkpoint_path: Option<String> = None;
    let mut no_checkpoint = false;
    let mut parallel_workers: usize = 0;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--budget" => {
                default_budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_budget);
                i += 2;
            }
            "--depth" => {
                default_strategy_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_strategy_depth);
                i += 2;
            }
            "--flat-depth" => {
                default_flat_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(default_flat_depth);
                i += 2;
            }
            "--post-mortem" => {
                post_mortem_file = args.get(i + 1).map(|s| s.to_string());
                i += 2;
            }
            "--checkpoint" => {
                checkpoint_path = args.get(i + 1).map(|s| s.to_string());
                i += 2;
            }
            "--no-checkpoint" => {
                no_checkpoint = true;
                i += 1;
            }
            "--parallel" => {
                parallel_workers = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            other => { task_file = other.to_string(); i += 1; }
        }
    }
    if task_file.is_empty() {
        eprintln!("No task file specified");
        return;
    }

    let task_source = match fs::read_to_string(&task_file) {
        Ok(s) => s,
        Err(e) => { eprintln!("Error reading {}: {}", task_file, e); return; }
    };

    let tasks = parse_curriculum_tasks(&task_source, default_flat_depth);

    // Resolve checkpoint path.
    let ckpt_path = if no_checkpoint {
        None
    } else {
        Some(checkpoint_path.unwrap_or_else(|| format!("{}.checkpoint", task_file)))
    };

    eprintln!();
    eprintln!("SELPH grow-v2: {} tasks", tasks.len());
    eprintln!("  Budget: {}, Flat depth: {}, Strategy depth: {}",
        default_budget, default_flat_depth, default_strategy_depth);
    if parallel_workers > 0 {
        eprintln!("  Parallel: {} workers", parallel_workers);
    }
    if let Some(ref p) = ckpt_path {
        eprintln!("  Checkpoint: {}", p);
    }
    eprintln!("  Core: synth_v2 + eval_v2 + types_v2 (no legacy consumer modules)");
    eprintln!();

    // Single env shared across the whole curriculum. Each solved task
    // is bound into the top scope as a Value::Function — synth_v2's
    // `library_components_from_env` will pick it up on subsequent tasks.
    let env = eval_v2::make_default_env();

    // §9.39 meta-curriculum support: evaluate every top-level form in
    // the task file that ISN'T `(task ...)` or `(task-args ...)`. This
    // lets a curriculum file mix `(define helper ...)` forms with
    // task entries — the helpers land in the same env synth uses, so
    // any task's synth can reach them as library components.
    // Runs even when there are no tasks, so a "helper-only" file
    // (M0-style) can be smoke-tested for parse/eval errors.
    if let Err(e) = eval_curriculum_preamble(&task_source, &env) {
        eprintln!("warning: preamble eval failed: {}", e);
    }

    if tasks.is_empty() {
        eprintln!("(no tasks; preamble-only run)");
        return;
    }

    // ── Checkpoint: load previously solved tasks ──────────────────
    let preamble_hash = compute_preamble_hash(&task_source);
    let mut checkpoint_solved: Vec<SolvedTask> = Vec::new();
    let mut checkpoint_names: HashSet<String> = HashSet::new();

    if let Some(ref ckpt) = ckpt_path {
        let loaded = load_checkpoint(ckpt);
        if !loaded.is_empty() {
            eprintln!("  Checkpoint: {} previously solved tasks", loaded.len());
            for st in &loaded {
                match reconstruct_solution(&st.source, &env) {
                    Ok(func @ types_v2::Value::Function(_)) => {
                        env.define(intern(&st.name), func);
                        checkpoint_names.insert(st.name.clone());
                    }
                    Ok(_) => {
                        eprintln!("    checkpoint: {} did not eval to Function, skipping", st.name);
                    }
                    Err(e) => {
                        eprintln!("    checkpoint: {} failed to reconstruct: {}", st.name, e);
                    }
                }
            }
            checkpoint_solved = loaded;
            eprintln!("    {} restored into env", checkpoint_names.len());
        }
    }

    // §9.37 Stage B: type universe is rebuilt per task from
    // `__types__` so curriculum-defined types take effect mid-run.

    let mut solved = 0usize;
    let mut total_candidates = 0usize;
    let total_start = std::time::Instant::now();

    // Per-strategy tally for the summary line.
    let mut by_strategy: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

    // §9.49 post-mortem: accumulate per-task result namespaces.
    let mut curriculum_results: Vec<types_v2::Value> = Vec::new();

    let task_count = tasks.len();

    // ── Parallel wavefront path ──────────────────────────────────────
    if parallel_workers > 0 {
        // Build initial unsolved set (excluding checkpoint-restored tasks).
        let mut unsolved: Vec<usize> = (0..tasks.len())
            .filter(|i| !checkpoint_names.contains(&tasks[*i].0))
            .collect();

        // Count checkpoint-restored tasks in solved/strategy tallies.
        for st in &checkpoint_solved {
            solved += 1;
            total_candidates += st.candidates;
            *by_strategy.entry(st.strategy.clone()).or_insert(0) += 1;
        }

        // Track the last WorkerResult per task for post-mortem.
        let mut last_result: std::collections::HashMap<String, WorkerResult> =
            std::collections::HashMap::new();

        let mut round = 0usize;
        loop {
            if unsolved.is_empty() { break; }
            round += 1;
            eprintln!("  ── Wavefront round {} ({} unsolved, {} workers) ──",
                round, unsolved.len(), parallel_workers);

            let worker_results = run_parallel_batch(
                &unsolved, &tasks, &env, default_budget, parallel_workers,
                default_strategy_depth,
            );

            // Index results by task name for fast lookup.
            let result_map: std::collections::HashMap<String, WorkerResult> =
                worker_results.into_iter().map(|wr| (wr.name.clone(), wr)).collect();

            let mut new_solved = 0usize;
            let mut still_unsolved = Vec::new();

            for &idx in &unsolved {
                let name = &tasks[idx].0;
                if let Some(wr) = result_map.get(name) {
                    total_candidates += wr.candidates;

                    if wr.found {
                        // Reconstruct and bind into env.
                        match reconstruct_solution(&wr.source, &env) {
                            Ok(func @ types_v2::Value::Function(_)) => {
                                env.define(intern(name), func);
                            }
                            Ok(_) => {
                                eprintln!("    parallel: {} did not eval to Function", name);
                            }
                            Err(e) => {
                                eprintln!("    parallel: {} reconstruct failed: {}", name, e);
                            }
                        }

                        eprintln!(
                            "  [{:>3}/{}] {:>4}  {:30}  {:>6} cand  {}",
                            idx + 1, task_count, wr.strategy, name,
                            wr.candidates, wr.source,
                        );

                        *by_strategy.entry(wr.strategy.clone()).or_insert(0) += 1;
                        solved += 1;
                        new_solved += 1;

                        checkpoint_solved.push(SolvedTask {
                            name: name.clone(),
                            strategy: wr.strategy.clone(),
                            candidates: wr.candidates,
                            source: wr.source.clone(),
                        });
                    } else {
                        still_unsolved.push(idx);
                        if wr.fitness > 0.0 {
                            eprintln!(
                                "  [{:>3}/{}] FAIL  {:30}  {:>6} cand  fitness={:.3}",
                                idx + 1, task_count, name, wr.candidates, wr.fitness,
                            );
                        }
                    }
                } else {
                    // Worker didn't report this task — shouldn't happen.
                    still_unsolved.push(idx);
                }
            }

            // Merge this round's results (overwrite earlier rounds for same task).
            for (name, wr) in result_map {
                last_result.insert(name, wr);
            }

            eprintln!("    Round {} complete: {} new solutions", round, new_solved);
            unsolved = still_unsolved;

            if new_solved == 0 { break; } // Fixed point.
        }

        // Build curriculum_results from final results (one entry per task).
        // First: checkpoint-restored tasks.
        for st in &checkpoint_solved {
            if checkpoint_names.contains(&st.name) {
                let mut rns = types_v2::NsMap::new();
                rns.insert(intern("name"), types_v2::Value::str(st.name.clone()));
                rns.insert(intern("found"), types_v2::Value::Bool(true));
                rns.insert(intern("candidates"), types_v2::Value::Int(st.candidates as i64));
                rns.insert(intern("strategy"), types_v2::Value::str(st.strategy.clone()));
                rns.insert(intern("m-chain-ran"), types_v2::Value::Bool(true));
                rns.insert(intern("fitness"), types_v2::Value::Num(1.0));
                curriculum_results.push(types_v2::Value::ns(rns));
            }
        }
        // Then: tasks that went through the wavefront.
        for idx in 0..tasks.len() {
            let (name, task_depth, inputs_legacy, expected_legacy, arity_hint,
                 test_inputs_legacy, test_expected_legacy) = &tasks[idx];
            if checkpoint_names.contains(name) { continue; }
            if let Some(wr) = last_result.get(name) {
                let mut rns = types_v2::NsMap::new();
                rns.insert(intern("name"), types_v2::Value::str(name.clone()));
                rns.insert(intern("found"), types_v2::Value::Bool(wr.found));
                rns.insert(intern("candidates"), types_v2::Value::Int(wr.candidates as i64));
                if !wr.output_type.is_empty() {
                    rns.insert(intern("output-type"), types_v2::Value::str(wr.output_type.clone()));
                }
                rns.insert(intern("m-chain-ran"), types_v2::Value::Bool(wr.m_chain_ran));
                rns.insert(intern("strategy"), types_v2::Value::str(wr.strategy.clone()));
                rns.insert(intern("fitness"), types_v2::Value::Num(wr.fitness));
                if !wr.best_source.is_empty() {
                    rns.insert(intern("best-source"), types_v2::Value::str(wr.best_source.clone()));
                }
                // For solved tasks, store solution source for Phase 4 template transfer.
                if wr.found && !wr.source.is_empty() {
                    rns.insert(intern("best-source"), types_v2::Value::str(wr.source.clone()));
                }
                rns.insert(intern("max-candidates"), types_v2::Value::Int(default_budget as i64));
                rns.insert(intern("depth"), types_v2::Value::Int(*task_depth as i64));
                if let Some(arity) = arity_hint {
                    rns.insert(intern("arity"), types_v2::Value::Int(*arity as i64));
                }
                // Convert legacy values to v2 for spec/test.
                let force_num = inputs_legacy.iter().chain(expected_legacy.iter())
                    .any(legacy_value_has_non_integral);
                let convert = |v: &Value| -> types_v2::Value {
                    if force_num { legacy_value_to_v2_force_num(v) }
                    else { legacy_value_to_v2(v) }
                };
                let inputs: Vec<types_v2::Value> = inputs_legacy.iter().map(&convert).collect();
                let expected: Vec<types_v2::Value> = expected_legacy.iter().map(&convert).collect();
                let spec_pairs: Vec<types_v2::Value> = inputs.iter()
                    .zip(expected.iter())
                    .map(|(i, e)| types_v2::Value::list(vec![i.clone(), e.clone()]))
                    .collect();
                rns.insert(intern("spec"), types_v2::Value::list(spec_pairs));
                if !test_inputs_legacy.is_empty() {
                    let test_inputs: Vec<types_v2::Value> = test_inputs_legacy.iter().map(&convert).collect();
                    let test_expected: Vec<types_v2::Value> = test_expected_legacy.iter().map(&convert).collect();
                    let test_pairs: Vec<types_v2::Value> = test_inputs.iter()
                        .zip(test_expected.iter())
                        .map(|(i, e)| types_v2::Value::list(vec![i.clone(), e.clone()]))
                        .collect();
                    rns.insert(intern("test"), types_v2::Value::list(test_pairs));
                }
                curriculum_results.push(types_v2::Value::ns(rns));
            }
        }
    } else {
    // ── Sequential path (original behavior) ──────────────────────────
    for (task_idx, (name, task_depth, inputs_legacy, expected_legacy, arity_hint,
         test_inputs_legacy, test_expected_legacy)) in tasks.iter().enumerate() {
        // Skip tasks already restored from checkpoint.
        if checkpoint_names.contains(name) {
            let st = checkpoint_solved.iter().find(|s| s.name == *name);
            let strategy_name = st.map(|s| s.strategy.as_str()).unwrap_or("Ckpt");
            let cands = st.map(|s| s.candidates).unwrap_or(0);
            eprintln!(
                "  [{:>3}/{}] CKPT  {:30}  {:>6} cand  (cached)  {}",
                task_idx + 1, task_count, name, cands, strategy_name,
            );
            *by_strategy.entry(strategy_name.to_string()).or_insert(0) += 1;
            total_candidates += cands;
            solved += 1;
            // Build a minimal post-mortem entry for checkpointed tasks.
            {
                let mut rns = types_v2::NsMap::new();
                rns.insert(intern("name"), types_v2::Value::str(name.clone()));
                rns.insert(intern("found"), types_v2::Value::Bool(true));
                rns.insert(intern("candidates"), types_v2::Value::Int(cands as i64));
                rns.insert(intern("strategy"), types_v2::Value::str(strategy_name.to_string()));
                rns.insert(intern("m-chain-ran"), types_v2::Value::Bool(true));
                rns.insert(intern("fitness"), types_v2::Value::Num(1.0));
                curriculum_results.push(types_v2::Value::ns(rns));
            }
            continue;
        }
        // Convert legacy Values → v2 Values once per task. The legacy
        // parser produces Num(f64) for everything numeric; v2 prefers
        // Int(i64) when integral so the type universe stays Int-biased
        // (matches the §9.27 design call).
        //
        // §9.31 physics carve-out: a task whose inputs/outputs contain
        // ANY non-integral float keeps every numeric value as
        // `Value::Num`. Otherwise the integral coercion kicks in for
        // float-domain tasks like `square 0.5 → 0.25`, where the
        // `0.25` would normalize fine but `0.5 * 1.0 = 0.5` could
        // produce an integer literal that the synth would then
        // misroute through the Int catalog. Per-task all-or-nothing
        // is the simplest rule that preserves curriculum behaviour
        // and lets float tasks stay floaty.
        let force_num = inputs_legacy.iter().chain(expected_legacy.iter())
            .any(legacy_value_has_non_integral);
        let convert = |v: &Value| -> types_v2::Value {
            if force_num { legacy_value_to_v2_force_num(v) }
            else { legacy_value_to_v2(v) }
        };
        let inputs: Vec<types_v2::Value> =
            inputs_legacy.iter().map(&convert).collect();
        let expected: Vec<types_v2::Value> =
            expected_legacy.iter().map(&convert).collect();
        // §9.47.4: convert held-out test rows (if any).
        let test_inputs: Vec<types_v2::Value> =
            test_inputs_legacy.iter().map(&convert).collect();
        let test_expected: Vec<types_v2::Value> =
            test_expected_legacy.iter().map(&convert).collect();

        // Rebuild the catalog per task. Cheap relative to synthesis,
        // and ensures any newly defined library function from a
        // previous task is visible.
        let skip = synth_v2::default_skip_set();
        let components = synth_v2::default_synth_components(&env, &skip);

        let start = std::time::Instant::now();
        // §9.37 Stage B: rebuild the universe per task so any
        // `__types__` mutations from prior tasks take effect.
        let universe = synth_v2::TypeUniverse::from_env(&env);

        // §9.31 multi-arg path: when the task is `task-args` with a
        // confirmed arity, infer per-position types from the first
        // example and dispatch through `synthesize_args` (Flat-only,
        // skipping the strategy chain whose decomposers all assume
        // a single-input lambda).
        let result = if let Some(arity) = arity_hint {
            // Per-position type inference from the first input list.
            let first = match inputs.first() {
                Some(types_v2::Value::List(items)) if items.len() == *arity => items.clone(),
                _ => {
                    eprintln!("  WARN  {:30}  arity hint {} but first input is not a list of that length; skipping",
                        name, arity);
                    continue;
                }
            };
            let arg_types: Vec<crate::intern::Sym> = first.iter()
                .map(|v| v.type_sym().unwrap_or_else(types_v2::type_any))
                .collect();
            let synth_result = synth_v2::synthesize_args_with_test(
                &components,
                &inputs,
                &arg_types,
                &expected,
                &env,
                &universe,
                *task_depth,
                default_budget,
                &test_inputs,
                &test_expected,
            );
            // Wrap in a StrategyResult so the success path below stays
            // shape-compatible. §9.46 fix: when synth_v2 reports a
            // curriculum-side decomposer (the M-chain) fired, label
            // the strategy with that decomposer's ns key instead of
            // the previously hardcoded `Flat`. Without this, every
            // multi-arg success was credited to Flat, including
            // chain solves — making the "By strategy" summary
            // silently misleading.
            let strategy = if synth_result.found {
                Some(match synth_result.decomposer_name {
                    Some(name) => synth_v2::Strategy::Custom(name),
                    None => synth_v2::Strategy::Flat,
                })
            } else { None };
            // §9.49: infer output type for post-mortem diagnostics.
            let output_type = synth_v2::infer_uniform_type_sym(&expected)
                .map(|s| crate::intern::resolve(s))
                .unwrap_or_else(|| "Mixed".to_string());
            let has_decomposers = matches!(
                env.lookup(intern("__decomposers__")),
                Some(types_v2::Value::Ns(ref m)) if !m.is_empty()
            );
            let best_source = if !synth_result.found {
                synth_result.best_nodes.as_ref().map(|nodes| {
                    let root = synth_result.best_root.unwrap_or(0);
                    eval_v2::node_to_source(nodes, root)
                })
            } else {
                None
            };
            synth_v2::StrategyResult {
                found: synth_result.found,
                nodes: synth_result.nodes,
                root: synth_result.root,
                candidates_explored: synth_result.candidates_explored,
                strategy,
                output_type: Some(output_type),
                m_chain_ran: has_decomposers,
                best_fitness: synth_result.best_fitness,
                best_source,
                beam: Vec::new(),
                experience: Vec::new(),
            }
        } else {
            synth_v2::synthesize_with_strategies(
                &components,
                &inputs,
                &expected,
                &env,
                &universe,
                *task_depth,
                default_budget,
                default_strategy_depth,
            )
        };
        let elapsed = start.elapsed();
        total_candidates += result.candidates_explored;

        if result.found {
            let nodes_vec = result.nodes.expect("found implies nodes");
            let root = result.root.expect("found implies root");
            let strategy_name = result
                .strategy
                .map(|s| s.name())
                .unwrap_or_else(|| "???".to_string());
            *by_strategy.entry(strategy_name.clone()).or_insert(0) += 1;

            let source = eval_v2::node_to_source(&nodes_vec, root);
            eprintln!(
                "  [{:>3}/{}] {:>4}  {:30}  {:>6} cand  {:>6.3}s  {}",
                task_idx + 1, task_count,
                strategy_name, name, result.candidates_explored,
                elapsed.as_secs_f64(), source,
            );

            // Bind the synthesized lambda into the env so subsequent
            // tasks see it as a library component. The Lambda Node at
            // `root` evaluates to a Value::Function that captures the
            // current env — exactly the binding model used by
            // `(define name (lambda ...))` in source.
            let nodes_rc: std::rc::Rc<[types_v2::Node]> = nodes_vec.into();
            match eval_v2::eval(&nodes_rc, root, &env) {
                Ok(func @ types_v2::Value::Function(_)) => {
                    env.define(intern(name), func);
                }
                Ok(other) => {
                    eprintln!("    (warning: solution did not eval to Function: {:?})", other);
                }
                Err(e) => {
                    eprintln!("    (warning: failed to bind solution as library: {})", e);
                }
            }

            // Accumulate into checkpoint.
            checkpoint_solved.push(SolvedTask {
                name: name.clone(),
                strategy: strategy_name.clone(),
                candidates: result.candidates_explored,
                source: source.clone(),
            });

            solved += 1;
        } else {
            if result.best_fitness > 0.0 {
                eprintln!(
                    "  [{:>3}/{}] FAIL  {:30}  {:>6} cand  {:>6.3}s  fitness={:.3}",
                    task_idx + 1, task_count,
                    name, result.candidates_explored, elapsed.as_secs_f64(),
                    result.best_fitness,
                );
            } else {
                eprintln!(
                    "  [{:>3}/{}] FAIL  {:30}  {:>6} cand  {:>6.3}s",
                    task_idx + 1, task_count,
                    name, result.candidates_explored, elapsed.as_secs_f64(),
                );
            }
        }

        // §9.49 post-mortem: build per-task result namespace.
        {
            let mut rns = types_v2::NsMap::new();
            rns.insert(intern("name"), types_v2::Value::str(name.clone()));
            rns.insert(intern("found"), types_v2::Value::Bool(result.found));
            rns.insert(intern("candidates"), types_v2::Value::Int(result.candidates_explored as i64));
            if let Some(ref otype) = result.output_type {
                rns.insert(intern("output-type"), types_v2::Value::str(otype.clone()));
            }
            rns.insert(intern("m-chain-ran"), types_v2::Value::Bool(result.m_chain_ran));
            let strategy_str = result.strategy
                .map(|s| s.name())
                .unwrap_or_default();
            rns.insert(intern("strategy"), types_v2::Value::str(strategy_str));
            // Include the original spec so post-mortem can retry.
            let spec_pairs: Vec<types_v2::Value> = inputs.iter()
                .zip(expected.iter())
                .map(|(i, e)| types_v2::Value::list(vec![i.clone(), e.clone()]))
                .collect();
            rns.insert(intern("spec"), types_v2::Value::list(spec_pairs));
            rns.insert(intern("max-candidates"), types_v2::Value::Int(default_budget as i64));
            rns.insert(intern("depth"), types_v2::Value::Int(*task_depth as i64));
            if let Some(arity) = arity_hint {
                rns.insert(intern("arity"), types_v2::Value::Int(*arity as i64));
            }
            // Include held-out test pairs so post-mortem can validate retries.
            if !test_inputs.is_empty() {
                let test_pairs: Vec<types_v2::Value> = test_inputs.iter()
                    .zip(test_expected.iter())
                    .map(|(i, e)| types_v2::Value::list(vec![i.clone(), e.clone()]))
                    .collect();
                rns.insert(intern("test"), types_v2::Value::list(test_pairs));
            }
            // §9.55: fitness data for near-miss analysis.
            rns.insert(intern("fitness"), types_v2::Value::Num(result.best_fitness));
            if let Some(ref best_src) = result.best_source {
                rns.insert(intern("best-source"), types_v2::Value::str(best_src.clone()));
            }
            // For solved tasks, store the solution source so Phase 4
            // template transfer can use it.
            if result.found {
                if let Some(ref st) = checkpoint_solved.last() {
                    if st.name == *name {
                        rns.insert(intern("best-source"), types_v2::Value::str(st.source.clone()));
                    }
                }
            }
            curriculum_results.push(types_v2::Value::ns(rns));
        }
    }
    } // end else (sequential path)

    let total_elapsed = total_start.elapsed();
    eprintln!();
    eprintln!("═════════════════════════════════════════════════════");
    eprintln!("  Phase 1 complete: {}/{} solved in {:.2}s ({} candidates)",
        solved, tasks.len(), total_elapsed.as_secs_f64(), total_candidates);
    if !by_strategy.is_empty() {
        let parts: Vec<String> = by_strategy.iter()
            .map(|(s, n)| format!("{}={}", s, n))
            .collect();
        eprintln!("  Strategy: {}", parts.join(", "));
    }
    // §9.55: fitness distribution summary.
    {
        let mut near_miss = 0usize;
        let mut partial = 0usize;
        let mut far = 0usize;
        let mut zero = 0usize;
        for r in &curriculum_results {
            if let types_v2::Value::Ns(ns) = r {
                if let Some(types_v2::Value::Bool(true)) = ns.get(&intern("found")) {
                    continue;
                }
                let fitness = match ns.get(&intern("fitness")) {
                    Some(types_v2::Value::Num(f)) => *f,
                    _ => 0.0,
                };
                if fitness >= 0.9 { near_miss += 1; }
                else if fitness >= 0.5 { partial += 1; }
                else if fitness > 0.0 { far += 1; }
                else { zero += 1; }
            }
        }
        let failed = near_miss + partial + far + zero;
        if failed > 0 {
            eprintln!("  Failures: {} (near-miss:{} partial:{} far:{} zero:{})",
                failed, near_miss, partial, far, zero);
        }
    }
    eprintln!("═════════════════════════════════════════════════════");

    // Save checkpoint after synthesis completes.
    if let Some(ref ckpt) = ckpt_path {
        save_checkpoint(ckpt, &checkpoint_solved, preamble_hash);
        eprintln!("  Checkpoint saved: {} entries → {}", checkpoint_solved.len(), ckpt);
    }

    // §9.54: deferred post-mortem loading. If --post-mortem <file> was given,
    // load it NOW (after curriculum, before post-mortem call) so the ~190
    // post-mortem defines don't bloat the env during synthesis.
    if let Some(ref pm_path) = post_mortem_file {
        match fs::read_to_string(pm_path) {
            Ok(pm_source) => {
                if let Err(e) = eval_curriculum_preamble(&pm_source, &env) {
                    eprintln!("warning: post-mortem load failed: {}", e);
                }
            }
            Err(e) => eprintln!("warning: could not read post-mortem file {}: {}", pm_path, e),
        }
    }

    // §9.49 post-mortem: bind results and call run-post-mortem if defined.
    env.define(
        intern("__curriculum_results__"),
        types_v2::Value::list(curriculum_results),
    );
    if let Some(pm_fn) = env.lookup(intern("run-post-mortem")) {
        if matches!(pm_fn, types_v2::Value::Function(_) | types_v2::Value::Builtin(_)) {
            eprintln!();
            eprintln!("── Post-mortem ──────────────────────────────────────");
            let results_val = env.lookup(intern("__curriculum_results__")).unwrap();
            match eval_v2::apply(&pm_fn, &[results_val], &env) {
                Ok(val) => {
                    // New format: val is a ns with "diagnoses", "prescriptions",
                    // "total-failures", "total-results".
                    // Fall back to legacy list format for backwards compat.
                    let (diagnoses, prescriptions) = if let types_v2::Value::Ns(top_ns) = &val {
                        let diags = top_ns.get(&intern("diagnoses"))
                            .and_then(|v| if let types_v2::Value::List(l) = v { Some(l.as_ref()) } else { None });
                        let rxs = top_ns.get(&intern("prescriptions"))
                            .and_then(|v| if let types_v2::Value::List(l) = v { Some(l.as_ref()) } else { None });
                        let total_f = top_ns.get(&intern("total-failures"))
                            .and_then(|v| if let types_v2::Value::Int(n) = v { Some(*n) } else { None })
                            .unwrap_or(0);
                        let total_r = top_ns.get(&intern("total-results"))
                            .and_then(|v| if let types_v2::Value::Int(n) = v { Some(*n) } else { None })
                            .unwrap_or(0);
                        let recovered = top_ns.get(&intern("recovered"))
                            .and_then(|v| if let types_v2::Value::Int(n) = v { Some(*n) } else { None })
                            .unwrap_or(0);
                        let scaffolds = top_ns.get(&intern("scaffolds-solved"))
                            .and_then(|v| if let types_v2::Value::Int(n) = v { Some(*n) } else { None })
                            .unwrap_or(0);
                        let lib_recovered = top_ns.get(&intern("library-recovered"))
                            .and_then(|v| if let types_v2::Value::Int(n) = v { Some(*n) } else { None })
                            .unwrap_or(0);
                        eprintln!("  {}/{} failed", total_f, total_r);
                        if recovered > 0 {
                            eprintln!("  {} auto-recovered (probe)", recovered);
                        }
                        eprintln!("  scaffolding: {} solved, {} tasks recovered via library", scaffolds, lib_recovered);
                        (diags, rxs)
                    } else if let types_v2::Value::List(items) = &val {
                        // Legacy: flat list of diagnoses, no prescriptions
                        (Some(items.as_ref()), None)
                    } else {
                        (None, None)
                    };

                    if let Some(items) = diagnoses {
                        if items.is_empty() {
                            eprintln!("  (no failures to analyze)");
                        } else {
                            // Tally each diagnostic dimension.
                            eprintln!();
                            let dimensions = ["size", "colors", "constant-out",
                                              "dims-consistent", "objects", "scale",
                                              "probe-1", "probe-2", "retry-found",
                                              "subtype"];
                            for dim in &dimensions {
                                let mut tally: std::collections::BTreeMap<String, usize> =
                                    std::collections::BTreeMap::new();
                                for item in items.iter() {
                                    if let types_v2::Value::Ns(ns) = item {
                                        let val = ns.get(&intern(dim))
                                            .map(|v| eval_v2::value_to_string(v))
                                            .unwrap_or_else(|| "?".into());
                                        *tally.entry(val).or_insert(0) += 1;
                                    }
                                }
                                let parts: Vec<String> = tally.iter()
                                    .map(|(k, v)| format!("{}={}", k, v))
                                    .collect();
                                eprintln!("  {:20} {}", dim, parts.join("  "));
                            }
                            eprintln!();
                            // Print per-task detail (compact).
                            for item in items.iter() {
                                if let types_v2::Value::Ns(ns) = item {
                                    let name = ns.get(&intern("name"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_else(|| "?".into());
                                    let size = ns.get(&intern("size"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    let colors = ns.get(&intern("colors"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    let p1 = ns.get(&intern("probe-1"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    let p2 = ns.get(&intern("probe-2"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    let retried = ns.get(&intern("retry-found"))
                                        .map(|v| matches!(v, types_v2::Value::Bool(true)))
                                        .unwrap_or(false);
                                    let retry_src = ns.get(&intern("retry-source"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    // Show prescription if available.
                                    let rx = ns.get(&intern("prescription"))
                                        .and_then(|v| if let types_v2::Value::Ns(rxns) = v {
                                            rxns.get(&intern("rx")).map(|r| eval_v2::value_to_string(r))
                                        } else { None })
                                        .unwrap_or_default();
                                    let sub = ns.get(&intern("subtype"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    let auto_found = ns.get(&intern("auto-found"))
                                        .map(|v| matches!(v, types_v2::Value::Bool(true)))
                                        .unwrap_or(false);
                                    let auto_src = ns.get(&intern("auto-source"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    if auto_found {
                                        eprintln!("  {:12}  AUTO-RECOVERED  {}",
                                            name, auto_src);
                                    } else if retried {
                                        eprintln!("  {:12}  RECOVERED  {}",
                                            name, retry_src);
                                    } else {
                                        let sub_str = if !sub.is_empty() && sub != "n/a"
                                            { format!(" sub={}", sub) } else { String::new() };
                                        eprintln!("  {:12}  rx={:28} size={:12} colors={:14} probe={}{}{}",
                                            name, rx, size, colors, p1,
                                            if p2 != "skipped" && p2 != "none" && !p2.is_empty()
                                                { format!(" d2={}", p2) } else { String::new() },
                                            sub_str);
                                    }
                                }
                            }
                        }
                    }

                    // Print prescription summary (ranked).
                    if let Some(rxs) = prescriptions {
                        if !rxs.is_empty() {
                            eprintln!();
                            eprintln!("── Prescriptions (ranked) ───────────────────────────");
                            for rx_item in rxs.iter() {
                                if let types_v2::Value::Ns(rx_ns) = rx_item {
                                    let rx_name = rx_ns.get(&intern("rx"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    let count = rx_ns.get(&intern("count"))
                                        .and_then(|v| if let types_v2::Value::Int(n) = v { Some(*n) } else { None })
                                        .unwrap_or(0);
                                    let priority = rx_ns.get(&intern("priority"))
                                        .and_then(|v| if let types_v2::Value::Int(n) = v { Some(*n) } else { None })
                                        .unwrap_or(99);
                                    let rationale = rx_ns.get(&intern("rationale"))
                                        .map(|v| eval_v2::value_to_string(v))
                                        .unwrap_or_default();
                                    let examples = rx_ns.get(&intern("examples"))
                                        .and_then(|v| if let types_v2::Value::List(l) = v {
                                            Some(l.iter()
                                                .take(3)
                                                .map(|e| eval_v2::value_to_string(e))
                                                .collect::<Vec<_>>()
                                                .join(", "))
                                        } else { None })
                                        .unwrap_or_default();
                                    eprintln!("  P{} {:30} {:>3} tasks  e.g. {}",
                                        priority, rx_name, count, examples);
                                    eprintln!("     {}", rationale);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  post-mortem error: {}", e);
                }
            }
        }
    }
}

/// Define a legacy macro into a v2 env so synth_v2 can discover it as
/// a library function via `library_components_from_env`. The roundtrip
/// path is: legacy nodes → source string → legacy parser → v2
/// node tree (via `eval_v2::convert_tree`) → eval. Same path the parser
/// Convert a legacy `types::Value` (as produced by `parse_curriculum_tasks`)
/// into a `types_v2::Value`. Curriculum task data is restricted to the
/// primitive variants — Num, Str, Bool, List, Nil — so this never sees
/// Function/Closure/RustMacro/Namespace/Grid/Alt. Numeric values get
/// normalized: integral f64s become Int, fractional ones stay Num.
fn legacy_value_to_v2(v: &Value) -> types_v2::Value {
    match v {
        Value::Num(n) => {
            if n.is_finite() && (n - n.round()).abs() < 1e-9
                && *n >= i64::MIN as f64 && *n <= i64::MAX as f64
            {
                types_v2::Value::Int(*n as i64)
            } else {
                types_v2::Value::Num(*n)
            }
        }
        Value::Str(s) => types_v2::Value::str(s),
        Value::Bool(b) => types_v2::Value::Bool(*b),
        Value::List(items) => {
            let v2_items: Vec<types_v2::Value> = items.iter().map(legacy_value_to_v2).collect();
            types_v2::Value::list(v2_items)
        }
        Value::Nil => types_v2::Value::Nil,
        other => {
            eprintln!("warning: legacy_value_to_v2 saw unsupported variant {:?}, using nil", other);
            types_v2::Value::Nil
        }
    }
}

/// Like `legacy_value_to_v2` but never coerces integral floats to Int.
/// Used by `cmd_grow_v2` for tasks whose value table contains at least
/// one non-integral number — keeps every numeric value as `Value::Num`
/// so the synthesizer dispatches to the Num-typed component catalog
/// (the §9.31 physics curriculum path). Integers, lists, strings, etc
/// are passed through unchanged: only `Value::Num` is special-cased.
fn legacy_value_to_v2_force_num(v: &Value) -> types_v2::Value {
    match v {
        Value::Num(n) => types_v2::Value::Num(*n),
        Value::List(items) => {
            let v2_items: Vec<types_v2::Value> =
                items.iter().map(legacy_value_to_v2_force_num).collect();
            types_v2::Value::list(v2_items)
        }
        // Non-numeric variants — same as the normalizing path.
        Value::Str(s) => types_v2::Value::str(s),
        Value::Bool(b) => types_v2::Value::Bool(*b),
        Value::Nil => types_v2::Value::Nil,
        other => {
            eprintln!("warning: legacy_value_to_v2_force_num saw unsupported variant {:?}, using nil", other);
            types_v2::Value::Nil
        }
    }
}

/// Evaluate every top-level form in `source` that ISN'T a `(task ...)`
/// or `(task-args ...)` entry against `env`. Used by `cmd_grow_v2` to
/// load `(define helper ...)` forms that ride alongside curriculum
/// tasks in the same file. The §9.39 meta-curriculum needs this so
/// the spec primitives, decomposers, and `__decomposers__` namespace
/// can be defined in the same file as the tasks that exercise them.
///
/// Errors from individual forms are reported but don't stop the
/// loop — a typo in one helper shouldn't kill the whole curriculum.
fn eval_curriculum_preamble(source: &str, env: &types_v2::Env) -> Result<(), String> {
    let (legacy_nodes, roots) = parse_file(source).map_err(|e| e.to_string())?;
    let task_sym = intern("task");
    let task_args_sym = intern("task-args");

    // Convert to v2 nodes once; per-form eval just picks the right
    // root index.
    let v2_nodes_vec = eval_v2::convert_tree(&legacy_nodes);
    let v2_nodes: Rc<[types_v2::Node]> = v2_nodes_vec.into();

    for &root in &roots {
        // Skip task forms — those go through parse_curriculum_tasks.
        let is_task = match &legacy_nodes[root] {
            Node::App(children) if !children.is_empty() => {
                if let Node::Symbol(s) = &legacy_nodes[children[0]] {
                    *s == task_sym || *s == task_args_sym
                } else { false }
            }
            _ => false,
        };
        if is_task { continue; }

        // Eval the form. Top-level `(define ...)` mutates env.top_scope.
        // Other expressions are evaluated for side effects (e.g.
        // `(print ...)`); their values are discarded.
        if let Err(e) = eval_v2::eval(&v2_nodes, root, env) {
            eprintln!("  preamble: form failed: {}", e);
        }
    }

    Ok(())
}

/// True if `v` is a numeric value with a non-integral component
/// (recursively into lists). The §9.31 physics carve-out uses this
/// per-task: any one fractional float in inputs/outputs flips the
/// task to Num-typed value conversion.
fn legacy_value_has_non_integral(v: &Value) -> bool {
    match v {
        Value::Num(n) => {
            !(n.is_finite() && (n - n.round()).abs() < 1e-9
                && *n >= i64::MIN as f64 && *n <= i64::MAX as f64)
        }
        Value::List(items) => items.iter().any(legacy_value_has_non_integral),
        _ => false,
    }
}

/// Convert a parsed AST node to a runtime Value (for curriculum spec parsing).
fn node_to_value(nodes: &[Node], idx: usize) -> Option<Value> {
    match &nodes[idx] {
        Node::Num(n) => Some(Value::Num(*n)),
        Node::Str(s) => Some(Value::Str(s.clone())),
        Node::Bool(b) => Some(Value::Bool(*b)),
        Node::Symbol(s) => {
            let name = resolve(*s);
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
        Node::App(children) if children.is_empty() => {
            Some(Value::List(vec![]))
        }
        Node::App(children) if children.len() == 1 => {
            node_to_value(nodes, children[0]).map(|v| Value::List(vec![v]))
        }
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
            if let Node::Symbol(s) = &nodes[children[0]] {
                if resolve(*s) == "#grid" && children.len() == 2 {
                    if let Some(v @ Value::List(_)) = node_to_value(nodes, children[1]) {
                        return Some(v);
                    }
                }
            }
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

fn parse_curriculum_tasks(source: &str, default_depth: usize)
    -> Vec<(String, usize, Vec<Value>, Vec<Value>, Option<usize>,
            Vec<Value>, Vec<Value>)>
{
    // Each tuple is (name, depth, inputs, expected, arity_hint,
    //                test_inputs, test_expected).
    //
    // arity_hint = None     → standard `(task ...)`. Single-input synth.
    // arity_hint = Some(N)  → `(task-args ...)`. Each input value is
    //                         expected to be a List of length N; the
    //                         synth uses indexed-atom seeds (§9.31
    //                         multi-arg path) instead of a single
    //                         InputVar.
    //
    // test_inputs / test_expected are held-out validation examples
    // (§9.47.4). Curriculum rows wrapped in `(test input output)`
    // are separated from training rows and used for post-synthesis
    // verification. Tasks without `(test ...)` rows have empty vecs.
    let mut tasks = Vec::new();

    let (nodes, roots) = match parse_file(source) {
        Ok(r) => r,
        Err(e) => { eprintln!("Parse error in task file: {}", e); return tasks; }
    };

    let task_sym = intern("task");
    let task_args_sym = intern("task-args");

    for &root in &roots {
        // Each task: (task "name" depth (in1 out1) (in2 out2) ...)
        // Or:        (task-args "name" depth (in1 out1) ...)  where
        // each in is a list whose length is the multi-arg arity.
        if let Node::App(children) = &nodes[root] {
            if children.len() < 4 { continue; }
            let head = match &nodes[children[0]] {
                Node::Symbol(s) => *s,
                _ => continue,
            };
            let is_args = if head == task_args_sym {
                true
            } else if head == task_sym {
                false
            } else {
                continue;
            };

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
            let mut test_inputs = Vec::new();
            let mut test_expected = Vec::new();
            let test_sym = intern("test");

            for &child_idx in &children[3..] {
                if let Node::App(pair) = &nodes[child_idx] {
                    if pair.len() == 2 {
                        // Regular training pair: (input output)
                        if let (Some(iv), Some(ov)) = (
                            node_to_value(&nodes, pair[0]),
                            node_to_value(&nodes, pair[1]),
                        ) {
                            inputs.push(iv);
                            expected.push(ov);
                        }
                    } else if pair.len() == 3 {
                        // §9.47.4: held-out test pair: (test input output)
                        if let Node::Symbol(s) = &nodes[pair[0]] {
                            if *s == test_sym {
                                if let (Some(iv), Some(ov)) = (
                                    node_to_value(&nodes, pair[1]),
                                    node_to_value(&nodes, pair[2]),
                                ) {
                                    test_inputs.push(iv);
                                    test_expected.push(ov);
                                }
                            }
                        }
                    }
                }
            }

            if !inputs.is_empty() {
                let arity_hint = if is_args {
                    // Infer arity from the first input list's length.
                    // Tasks with mixed list lengths fall back to None
                    // (treated as ordinary single-input).
                    if let Some(Value::List(items)) = inputs.first() {
                        let n = items.len();
                        if inputs.iter().all(|v| matches!(v, Value::List(l) if l.len() == n)) {
                            Some(n)
                        } else {
                            eprintln!("warning: task-args {} has inconsistent input arities; treating as single-input", name);
                            None
                        }
                    } else {
                        eprintln!("warning: task-args {} input is not a list; treating as single-input", name);
                        None
                    }
                } else {
                    None
                };
                tasks.push((name, depth, inputs, expected, arity_hint,
                           test_inputs, test_expected));
            }
        }
    }

    tasks
}

/// Boolean decomposition: try (and P Q), (or P Q), (not P) for all
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

// ────────────────────────────────────────────────────────────────────────────
// §9.61 Beam overnight command
// ────────────────────────────────────────────────────────────────────────────

/// Long-running beam search synthesis on ARC tasks.
///
/// Phase 1: baseline synthesis (regular grow-v2 probe) on all tasks.
/// Phase 2: beam ratchet on unsolved tasks with escalating rounds.
///
/// Solutions compound into the library across tasks (wavefront).
/// Checkpoints after each newly solved task.
fn cmd_beam_overnight(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph beam-overnight <tasks.selph> [--preamble preamble.selph] \
                   [--budget N] [--beam-width N] [--max-rounds N] [--time-limit-hours N] \
                   [--checkpoint PATH] [--no-checkpoint]");
        return;
    }

    let mut task_file = String::new();
    let mut preamble_file: Option<String> = None;
    let mut budget: usize = 50000;
    let mut beam_width: usize = 200;
    let mut beam_depth: usize = 2;
    let mut max_rounds: usize = 10;
    let mut time_limit_secs: u64 = 8 * 3600; // 8 hours default
    let mut checkpoint_path: Option<String> = None;
    let mut no_checkpoint = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--preamble" => {
                preamble_file = args.get(i + 1).map(|s| s.to_string());
                i += 2;
            }
            "--budget" => {
                budget = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(budget);
                i += 2;
            }
            "--beam-width" => {
                beam_width = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(beam_width);
                i += 2;
            }
            "--beam-depth" => {
                beam_depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(beam_depth);
                i += 2;
            }
            "--max-rounds" => {
                max_rounds = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(max_rounds);
                i += 2;
            }
            "--time-limit-hours" => {
                let hours: f64 = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(8.0);
                time_limit_secs = (hours * 3600.0) as u64;
                i += 2;
            }
            "--checkpoint" => {
                checkpoint_path = args.get(i + 1).map(|s| s.to_string());
                i += 2;
            }
            "--no-checkpoint" => {
                no_checkpoint = true;
                i += 1;
            }
            other => { task_file = other.to_string(); i += 1; }
        }
    }

    if task_file.is_empty() {
        eprintln!("No task file specified");
        return;
    }

    let task_source = match fs::read_to_string(&task_file) {
        Ok(s) => s,
        Err(e) => { eprintln!("Error reading {}: {}", task_file, e); return; }
    };

    let tasks = parse_curriculum_tasks(&task_source, 2);
    let ckpt_path = if no_checkpoint {
        None
    } else {
        Some(checkpoint_path.unwrap_or_else(|| format!("{}.beam_checkpoint", task_file)))
    };

    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  SELPH beam-overnight: {} tasks", tasks.len());
    eprintln!("  Budget/round: {}, Beam width: {}, Beam depth: {}, Max rounds: {}",
        budget, beam_width, beam_depth, max_rounds);
    eprintln!("  Time limit: {:.1} hours", time_limit_secs as f64 / 3600.0);
    if let Some(ref p) = ckpt_path {
        eprintln!("  Checkpoint: {}", p);
    }
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();

    let env = eval_v2::make_default_env();

    // Load preamble (M-chain, helpers, etc.)
    if let Some(ref preamble) = preamble_file {
        match fs::read_to_string(preamble) {
            Ok(source) => {
                if let Err(e) = eval_curriculum_preamble(&source, &env) {
                    eprintln!("warning: preamble eval failed: {}", e);
                }
                eprintln!("  Preamble loaded: {}", preamble);
            }
            Err(e) => eprintln!("warning: could not read preamble {}: {}", preamble, e),
        }
    }

    // Also eval the task file preamble (non-task defines).
    if let Err(e) = eval_curriculum_preamble(&task_source, &env) {
        eprintln!("warning: task file preamble eval failed: {}", e);
    }

    // Load checkpoint.
    let mut checkpoint_solved: Vec<SolvedTask> = Vec::new();
    let mut checkpoint_names: HashSet<String> = HashSet::new();
    if let Some(ref ckpt) = ckpt_path {
        let loaded = load_checkpoint(ckpt);
        if !loaded.is_empty() {
            eprintln!("  Checkpoint: {} previously solved tasks", loaded.len());
            for st in &loaded {
                match reconstruct_solution(&st.source, &env) {
                    Ok(func @ types_v2::Value::Function(_)) => {
                        env.define(intern(&st.name), func);
                        checkpoint_names.insert(st.name.clone());
                    }
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("    checkpoint: {} failed: {}", st.name, e);
                    }
                }
            }
            checkpoint_solved = loaded;
            eprintln!("    {} restored into env", checkpoint_names.len());
        }
    }

    let total_start = std::time::Instant::now();
    let deadline = std::time::Duration::from_secs(time_limit_secs);
    let mut solved = checkpoint_names.len();
    let mut total_candidates: usize = 0;
    let preamble_hash = compute_preamble_hash(&task_source);

    // ── Phase 1: Baseline synthesis (regular grow-v2 path) ──────────
    eprintln!();
    eprintln!("── Phase 1: Baseline synthesis ──────────────────────────────");

    let baseline_budget: usize = 200000;
    let mut unsolved_indices: Vec<usize> = Vec::new();
    let mut best_fitness: Vec<(usize, f64, String)> = Vec::new(); // (idx, fitness, best_source)

    for (idx, task) in tasks.iter().enumerate() {
        if total_start.elapsed() > deadline {
            eprintln!("  ⏰ Time limit reached during Phase 1");
            break;
        }
        let (ref name, depth, ref inputs, ref expected, arity_hint,
             ref test_inputs, ref test_expected) = *task;

        if checkpoint_names.contains(name) {
            continue;
        }

        let wr = synthesize_one_task(
            name, depth, inputs, expected, arity_hint,
            test_inputs, test_expected, &env, baseline_budget, 2,
        );
        total_candidates += wr.candidates;

        if wr.found {
            match reconstruct_solution(&wr.source, &env) {
                Ok(func @ types_v2::Value::Function(_)) => {
                    env.define(intern(name), func);
                }
                _ => {}
            }
            eprintln!("  ✓ {:>3}/{} {:30} {:>6} cand  {}",
                idx + 1, tasks.len(), name, wr.candidates, wr.source);
            solved += 1;
            checkpoint_solved.push(SolvedTask {
                name: name.clone(),
                strategy: wr.strategy.clone(),
                candidates: wr.candidates,
                source: wr.source.clone(),
            });
            checkpoint_names.insert(name.clone());
        } else {
            unsolved_indices.push(idx);
            if wr.fitness > 0.0 {
                best_fitness.push((idx, wr.fitness, wr.best_source.clone()));
            }
        }
    }

    // Save checkpoint after Phase 1.
    if let Some(ref ckpt) = ckpt_path {
        save_checkpoint(ckpt, &checkpoint_solved, preamble_hash);
    }

    eprintln!();
    eprintln!("  Phase 1 complete: {}/{} solved, {} unsolved ({} with fitness > 0)",
        solved, tasks.len(), unsolved_indices.len(),
        best_fitness.len());
    eprintln!("  Elapsed: {:.1}s", total_start.elapsed().as_secs_f64());

    // ── Phase 2: Beam ratchet on unsolved tasks ─────────────────────
    eprintln!();
    eprintln!("── Phase 2: Beam ratchet ({} rounds, depth={}, beam={}, budget={}/round) ──",
        max_rounds, beam_depth, beam_width, budget);

    // Sort unsolved by fitness descending — try near-misses first.
    let mut fitness_map: std::collections::HashMap<usize, (f64, String)> =
        best_fitness.into_iter().map(|(i, f, s)| (i, (f, s))).collect();
    unsolved_indices.sort_by(|a, b| {
        let fa = fitness_map.get(a).map(|x| x.0).unwrap_or(0.0);
        let fb = fitness_map.get(b).map(|x| x.0).unwrap_or(0.0);
        fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut beam_solved = 0usize;
    let mut beam_round = 0usize;

    // Wavefront: repeat until no new solutions or time limit.
    loop {
        if total_start.elapsed() > deadline {
            eprintln!("  ⏰ Time limit reached");
            break;
        }
        if unsolved_indices.is_empty() {
            eprintln!("  All tasks solved!");
            break;
        }

        beam_round += 1;
        let round_start = std::time::Instant::now();
        let mut round_solved = 0usize;
        let mut still_unsolved = Vec::new();

        eprintln!("  ── Beam wavefront round {} ({} unsolved) ──", beam_round, unsolved_indices.len());

        for &idx in &unsolved_indices {
            if total_start.elapsed() > deadline {
                still_unsolved.push(idx);
                continue;
            }

            let (ref name, _depth, ref inputs, ref expected, _arity_hint,
                 ref test_inputs, ref test_expected) = tasks[idx];

            // Convert values.
            let v2_inputs: Vec<types_v2::Value> = inputs.iter().map(|v| legacy_value_to_v2(v)).collect();
            let v2_expected: Vec<types_v2::Value> = expected.iter().map(|v| legacy_value_to_v2(v)).collect();
            let v2_test_inputs: Vec<types_v2::Value> = test_inputs.iter().map(|v| legacy_value_to_v2(v)).collect();
            let v2_test_expected: Vec<types_v2::Value> = test_expected.iter().map(|v| legacy_value_to_v2(v)).collect();

            // Build spec as Value pairs.
            let spec_pairs: Vec<types_v2::Value> = v2_inputs.iter().zip(v2_expected.iter())
                .map(|(i, e)| types_v2::Value::list(vec![i.clone(), e.clone()]))
                .collect();

            let skip = synth_v2::default_skip_set();
            let mut components = synth_v2::default_synth_components(&env, &skip);
            let universe = synth_v2::TypeUniverse::from_env(&env);

            // Run beam ratchet: iterative rounds with full strategy chain.
            // Each round runs M-chain + Flat (with beam) + RD + BD + HO + D&C.
            // Beam entries from Flat get injected as library for the next round.
            let mut task_found = false;
            let mut task_candidates = 0usize;
            let mut task_source = String::new();
            let mut task_strategy = String::new();

            for round in 1..=max_rounds {
                if total_start.elapsed() > deadline { break; }

                // Rebuild components each round (picks up injected library).
                let skip = synth_v2::default_skip_set();
                let components = synth_v2::default_synth_components(&env, &skip);
                let universe = synth_v2::TypeUniverse::from_env(&env);

                let sr = synth_v2::synthesize_with_strategies_beam(
                    &components, &v2_inputs, &v2_expected, &env, &universe,
                    beam_depth, budget, 2, beam_width,
                );
                task_candidates += sr.candidates_explored;

                if sr.found {
                    let nodes = sr.nodes.as_ref().unwrap();
                    let root = sr.root.unwrap();
                    if synth_v2::validate_held_out(nodes, root,
                            &v2_test_inputs, &v2_test_expected, &env) {
                        task_source = eval_v2::node_to_source(nodes, root);
                        task_strategy = sr.strategy.map(|s| s.name()).unwrap_or_default();
                        task_found = true;
                        break;
                    }
                }

                // Inject beam entries as library functions for next round.
                for (bi, entry) in sr.beam.iter().enumerate() {
                    let (n, r) = synth_v2::wrap_lambda(&entry.pool_entry);
                    let src = eval_v2::node_to_source(&n, r);
                    let lib_name = format!("__beam_{}_r{}_{}", name, round, bi);
                    match reconstruct_solution(&src, &env) {
                        Ok(func @ types_v2::Value::Function(_)) => {
                            env.define(intern(&lib_name), func);
                        }
                        _ => {}
                    }
                }
            }

            total_candidates += task_candidates;

            if task_found {
                match reconstruct_solution(&task_source, &env) {
                    Ok(func @ types_v2::Value::Function(_)) => {
                        env.define(intern(name), func);
                    }
                    _ => {}
                }
                eprintln!("  ✓ {:>3}/{} {:30} {:>6} cand  {}",
                    idx + 1, tasks.len(), name, task_candidates, task_source);
                solved += 1;
                beam_solved += 1;
                round_solved += 1;
                checkpoint_solved.push(SolvedTask {
                    name: name.clone(),
                    strategy: format!("Beam+{}", task_strategy),
                    candidates: task_candidates,
                    source: task_source.clone(),
                });
                checkpoint_names.insert(name.clone());
            } else {
                still_unsolved.push(idx);
            }
        }

        // Save checkpoint after each wavefront round.
        if let Some(ref ckpt) = ckpt_path {
            save_checkpoint(ckpt, &checkpoint_solved, preamble_hash);
        }

        eprintln!("  Round {} done: +{} solved, {:.1}s",
            beam_round, round_solved, round_start.elapsed().as_secs_f64());

        unsolved_indices = still_unsolved;

        if round_solved == 0 {
            // No progress this wavefront round.
            // Escalate: try with higher depth per beam round.
            eprintln!("  No new solutions this round. Continuing with escalated budget...");
            // Double the budget for next round, cap at 500K.
            // budget = (budget * 2).min(500000); // Keep budget fixed for now.
        }
    }

    // ── Summary ─────────────────────────────────────────────────────
    let elapsed = total_start.elapsed();
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  beam-overnight complete");
    eprintln!("  Total: {}/{} solved ({} from Phase 1, {} from beam ratchet)",
        solved, tasks.len(), solved - beam_solved, beam_solved);
    eprintln!("  Candidates: {}", total_candidates);
    eprintln!("  Elapsed: {:.1}s ({:.1} hours)",
        elapsed.as_secs_f64(), elapsed.as_secs_f64() / 3600.0);
    eprintln!("  Unsolved: {}", unsolved_indices.len());
    eprintln!("═══════════════════════════════════════════════════════════════");
}

fn cmd_arc(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: selph arc <path> [--output tasks.selph] [--depth N]");
        return;
    }

    let path = &args[0];
    let mut output_file: Option<String> = None;
    let mut depth = 1usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "-o" => { i += 1; output_file = Some(args[i].clone()); }
            "--depth" => { i += 1; depth = args[i].parse().unwrap_or(1); }
            _ => { eprintln!("Unknown flag: {}", args[i]); return; }
        }
        i += 1;
    }

    let meta = std::fs::metadata(path);
    let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);

    let tasks: Vec<arc::ArcTask> = if is_dir {
        let mut tasks = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(path).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let file_path = entry.path();
            let id = file_path.file_stem().unwrap().to_string_lossy().to_string();
            let json = std::fs::read_to_string(&file_path).unwrap();
            match arc::parse_arc_task(&id, &json) {
                Ok(task) => tasks.push(task),
                Err(e) => eprintln!("Error parsing {}: {}", file_path.display(), e),
            }
        }
        eprintln!("Loaded {} ARC tasks from {}", tasks.len(), path);
        tasks
    } else {
        let id = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().to_string();
        let json = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", path, e); std::process::exit(1);
        });
        match arc::parse_arc_task(&id, &json) {
            Ok(task) => vec![task],
            Err(e) => { eprintln!("Error parsing {}: {}", path, e); return; }
        }
    };

    if let Some(ref out) = output_file {
        let curriculum = arc::arc_dir_to_curriculum(&tasks, depth);
        std::fs::write(out, &curriculum).unwrap();
        eprintln!("Wrote curriculum ({} tasks) to {}", tasks.len(), out);
    } else {
        // Default: print task info
        for task in &tasks {
            println!("{}: {} train, {} test",
                task.id, task.train.len(), task.test.len());
        }
    }
}
