//! Bytecode VM for fast evaluation of synthesis candidates.
//!
//! Compiles a candidate program (node tree) into flat bytecodes and executes
//! on a stack machine. This avoids the overhead of tree-walking, environment
//! creation, and recursive dispatch that the standard evaluator incurs.
//!
//! The VM handles the subset of SELPH used during synthesis:
//! - Num, Str, Bool literals
//! - Symbol "x" (the single lambda parameter)
//! - App (builtin or macro calls)
//! - If expressions
//!
//! For candidates using unsupported constructs, compilation returns Err
//! and the caller falls back to the tree-walking evaluator.

use crate::intern::{Sym, intern, resolve};
use crate::types::{Node, Value};
use crate::eval::apply_builtin;

/// A single bytecode instruction.
#[derive(Clone, Debug)]
pub enum Op {
    PushNum(f64),
    PushBool(bool),
    PushStr(u16),              // index into Chunk::strings
    PushNamespace(u16),        // index into Chunk::namespaces — pre-built constant
    LoadArg,                   // push the lambda argument (x)
    CallBuiltin(Sym, u8),      // call builtin with N args from stack
    CallMacro(u16, u8),        // call pre-compiled macro chunk by index, N args
    JumpIfFalse(u16),          // pop condition, jump if false/0
    Jump(u16),                 // unconditional jump
    Return,                    // end — top of stack is result
}

/// A compiled bytecode chunk.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub ops: Vec<Op>,
    pub strings: Vec<String>,
    pub namespaces: Vec<Value>,  // pre-built namespace constants
}

/// Pre-compiled macro table entry.
#[derive(Clone, Debug)]
pub struct MacroChunk {
    pub name: String,
    pub param_count: usize,
    pub chunk: Chunk,
}

/// Compilation context: knows which names are builtins vs macros.
pub struct CompileCtx {
    pub sym_x: Sym,
    pub builtin_names: std::collections::HashSet<Sym>,
    pub macro_names: Vec<(Sym, u16)>, // (name_sym, macro_chunk_index)
    pub macro_compiled: Vec<bool>,    // which macro indices compiled successfully
}

impl CompileCtx {
    pub fn new(builtins: &[&str], macros: &[(String, Vec<String>, Vec<Node>, usize)]) -> Self {
        let sym_x = intern("x");
        let mut builtin_names = std::collections::HashSet::new();
        for &name in builtins {
            builtin_names.insert(intern(name));
        }
        let mut macro_names = Vec::new();
        for (i, (name, _, _, _)) in macros.iter().enumerate() {
            macro_names.push((intern(name), i as u16));
        }
        let macro_compiled = vec![false; macros.len()];
        CompileCtx { sym_x, builtin_names, macro_names, macro_compiled }
    }

    fn lookup_macro(&self, sym: Sym) -> Option<u16> {
        self.macro_names.iter()
            .find(|(s, _)| *s == sym)
            .map(|(_, idx)| *idx)
    }

    fn macro_compiled(&self, idx: u16) -> bool {
        self.macro_compiled.get(idx as usize).copied().unwrap_or(false)
    }
}

/// Compile a node tree into bytecodes.
pub fn compile(nodes: &[Node], root: usize, ctx: &CompileCtx) -> Result<Chunk, String> {
    let mut chunk = Chunk { ops: Vec::new(), strings: Vec::new(), namespaces: Vec::new() };
    compile_node(nodes, root, ctx, &mut chunk)?;
    chunk.ops.push(Op::Return);
    Ok(chunk)
}

/// Compile macro bodies into chunks. Returns None for macros that can't be compiled.
pub fn compile_macros(
    macros: &[(String, Vec<String>, Vec<Node>, usize)],
    ctx: &CompileCtx,
) -> Vec<Option<MacroChunk>> {
    macros.iter().map(|(name, params, nodes, root)| {
        // Build a context where the macro's params are treated as args.
        // For single-param macros, the param is like "x" — loaded via LoadArg.
        // Multi-param macros are not supported in the VM.
        if params.len() != 1 {
            return None;
        }
        let macro_ctx = MacroCompileCtx {
            param_sym: intern(&params[0]),
            outer_ctx: ctx,
        };
        let mut chunk = Chunk { ops: Vec::new(), strings: Vec::new(), namespaces: Vec::new() };
        match compile_macro_node(nodes, *root, &macro_ctx, &mut chunk) {
            Ok(()) => {
                chunk.ops.push(Op::Return);
                Some(MacroChunk {
                    name: name.clone(),
                    param_count: params.len(),
                    chunk,
                })
            }
            Err(_) => None,
        }
    }).collect()
}

struct MacroCompileCtx<'a> {
    param_sym: Sym,
    outer_ctx: &'a CompileCtx,
}

fn compile_macro_node(
    nodes: &[Node], idx: usize, ctx: &MacroCompileCtx, chunk: &mut Chunk,
) -> Result<(), String> {
    match &nodes[idx] {
        Node::Num(n) => { chunk.ops.push(Op::PushNum(*n)); Ok(()) }
        Node::Bool(b) => { chunk.ops.push(Op::PushBool(*b)); Ok(()) }
        Node::Str(s) => {
            let si = chunk.strings.len() as u16;
            chunk.strings.push(s.clone());
            chunk.ops.push(Op::PushStr(si));
            Ok(())
        }
        Node::Symbol(sym) => {
            if *sym == ctx.param_sym {
                chunk.ops.push(Op::LoadArg);
                Ok(())
            } else {
                Err("vm: macro references non-param symbol".into())
            }
        }
        Node::App(children) => {
            if children.is_empty() { return Err("vm: empty app in macro".into()); }
            if let Node::Symbol(fn_sym) = &nodes[children[0]] {
                let fn_name = resolve(*fn_sym);
                // Special form: (ns ("key" value) ...) — build namespace constant
                if fn_name == "ns" || fn_name == "namespace" {
                    let mut entries = std::collections::HashMap::new();
                    for &child_idx in &children[1..] {
                        if let Node::App(pair) = &nodes[child_idx] {
                            if pair.len() == 2 {
                                let key = match &nodes[pair[0]] {
                                    Node::Str(s) => s.clone(),
                                    Node::Symbol(s) => resolve(*s),
                                    _ => return Err("vm: non-string key in ns".into()),
                                };
                                let val = match &nodes[pair[1]] {
                                    Node::Num(n) => Value::Num(*n),
                                    Node::Bool(b) => Value::Bool(*b),
                                    Node::Str(s) => Value::Str(s.clone()),
                                    _ => return Err("vm: non-constant value in ns".into()),
                                };
                                entries.insert(key, val);
                            }
                        }
                    }
                    let ns_idx = chunk.namespaces.len() as u16;
                    chunk.namespaces.push(Value::Namespace(entries));
                    chunk.ops.push(Op::PushNamespace(ns_idx));
                    return Ok(());
                }
                // Compile arguments
                for &arg_idx in &children[1..] {
                    compile_macro_node(nodes, arg_idx, ctx, chunk)?;
                }
                let arity = (children.len() - 1) as u8;
                if ctx.outer_ctx.builtin_names.contains(fn_sym) {
                    chunk.ops.push(Op::CallBuiltin(*fn_sym, arity));
                    Ok(())
                } else if let Some(macro_idx) = ctx.outer_ctx.lookup_macro(*fn_sym) {
                    if ctx.outer_ctx.macro_compiled(macro_idx) {
                        chunk.ops.push(Op::CallMacro(macro_idx, arity));
                        Ok(())
                    } else {
                        Err(format!("vm: macro {} not VM-compilable", fn_name))
                    }
                } else {
                    Err("vm: unknown function in macro".into())
                }
            } else {
                Err("vm: non-symbol in function position of macro".into())
            }
        }
        Node::If(cond, then_br, else_br) => {
            compile_macro_node(nodes, *cond, ctx, chunk)?;
            let jf_idx = chunk.ops.len();
            chunk.ops.push(Op::JumpIfFalse(0));
            compile_macro_node(nodes, *then_br, ctx, chunk)?;
            let j_idx = chunk.ops.len();
            chunk.ops.push(Op::Jump(0));
            let else_start = chunk.ops.len() as u16;
            chunk.ops[jf_idx] = Op::JumpIfFalse(else_start);
            compile_macro_node(nodes, *else_br, ctx, chunk)?;
            let end = chunk.ops.len() as u16;
            chunk.ops[j_idx] = Op::Jump(end);
            Ok(())
        }
        _ => Err("vm: unsupported node in macro".into()),
    }
}

fn compile_node(
    nodes: &[Node], idx: usize, ctx: &CompileCtx, chunk: &mut Chunk,
) -> Result<(), String> {
    match &nodes[idx] {
        Node::Num(n) => { chunk.ops.push(Op::PushNum(*n)); Ok(()) }
        Node::Bool(b) => { chunk.ops.push(Op::PushBool(*b)); Ok(()) }
        Node::Str(s) => {
            let si = chunk.strings.len() as u16;
            chunk.strings.push(s.clone());
            chunk.ops.push(Op::PushStr(si));
            Ok(())
        }
        Node::Symbol(sym) => {
            if *sym == ctx.sym_x {
                chunk.ops.push(Op::LoadArg);
                Ok(())
            } else {
                Err(format!("vm: unbound symbol {}", resolve(*sym)))
            }
        }
        Node::App(children) => {
            if children.is_empty() { return Err("vm: empty app".into()); }
            if let Node::Symbol(fn_sym) = &nodes[children[0]] {
                // Compile arguments left-to-right
                for &arg_idx in &children[1..] {
                    compile_node(nodes, arg_idx, ctx, chunk)?;
                }
                let arity = (children.len() - 1) as u8;
                if ctx.builtin_names.contains(fn_sym) {
                    chunk.ops.push(Op::CallBuiltin(*fn_sym, arity));
                    Ok(())
                } else if let Some(macro_idx) = ctx.lookup_macro(*fn_sym) {
                    // Only emit CallMacro if the macro was successfully compiled
                    if ctx.macro_compiled(macro_idx) {
                        chunk.ops.push(Op::CallMacro(macro_idx, arity));
                        Ok(())
                    } else {
                        Err(format!("vm: macro {} not VM-compilable", resolve(*fn_sym)))
                    }
                } else {
                    Err(format!("vm: unknown function {}", resolve(*fn_sym)))
                }
            } else {
                Err("vm: non-symbol in function position".into())
            }
        }
        Node::If(cond, then_br, else_br) => {
            compile_node(nodes, *cond, ctx, chunk)?;
            let jf_idx = chunk.ops.len();
            chunk.ops.push(Op::JumpIfFalse(0)); // placeholder
            compile_node(nodes, *then_br, ctx, chunk)?;
            let j_idx = chunk.ops.len();
            chunk.ops.push(Op::Jump(0)); // placeholder
            let else_start = chunk.ops.len() as u16;
            chunk.ops[jf_idx] = Op::JumpIfFalse(else_start);
            compile_node(nodes, *else_br, ctx, chunk)?;
            let end = chunk.ops.len() as u16;
            chunk.ops[j_idx] = Op::Jump(end);
            Ok(())
        }
        Node::Lambda(_, body) => {
            // In synthesis context, the lambda just wraps the body.
            // The arg binding is handled by LoadArg.
            compile_node(nodes, *body, ctx, chunk)
        }
        _ => Err("vm: unsupported node type".into()),
    }
}

/// Execute a compiled chunk with a single argument.
/// `macro_chunks` provides pre-compiled macro bodies for CallMacro.
/// `stack` is a reusable allocation to avoid per-call Vec allocation.
pub fn execute(
    chunk: &Chunk,
    arg: &Value,
    macro_chunks: &[Option<MacroChunk>],
    stack: &mut Vec<Value>,
) -> Result<Value, String> {
    stack.clear();
    execute_inner(chunk, arg, macro_chunks, stack, 0)
}

fn execute_inner(
    chunk: &Chunk,
    arg: &Value,
    macro_chunks: &[Option<MacroChunk>],
    stack: &mut Vec<Value>,
    depth: usize,
) -> Result<Value, String> {
    if depth > 64 {
        return Err("vm: max call depth exceeded".into());
    }
    let mut ip = 0usize;
    let ops = &chunk.ops;

    while ip < ops.len() {
        match &ops[ip] {
            Op::PushNum(n) => stack.push(Value::Num(*n)),
            Op::PushBool(b) => stack.push(Value::Bool(*b)),
            Op::PushStr(si) => stack.push(Value::Str(chunk.strings[*si as usize].clone())),
            Op::PushNamespace(ni) => stack.push(chunk.namespaces[*ni as usize].clone()),
            Op::LoadArg => stack.push(arg.clone()),
            Op::CallBuiltin(sym, arity) => {
                let n = *arity as usize;
                if stack.len() < n {
                    return Err("vm: stack underflow".into());
                }
                let args_start = stack.len() - n;
                let result = apply_builtin(*sym, &stack[args_start..])?;
                stack.truncate(args_start);
                stack.push(result);
            }
            Op::CallMacro(idx, arity) => {
                let n = *arity as usize;
                if stack.len() < n {
                    return Err("vm: stack underflow on macro call".into());
                }
                let mc = macro_chunks.get(*idx as usize)
                    .and_then(|m| m.as_ref())
                    .ok_or("vm: macro not compiled")?;
                // For single-param macros: the arg is the top of stack
                if n != 1 {
                    return Err("vm: multi-param macros not supported".into());
                }
                let macro_arg = stack.pop().unwrap();
                let result = {
                    let saved_len = stack.len();
                    let r = execute_inner(&mc.chunk, &macro_arg, macro_chunks, stack, depth + 1)?;
                    // Restore stack to pre-call state (in case macro left extra values)
                    stack.truncate(saved_len);
                    r
                };
                stack.push(result);
            }
            Op::JumpIfFalse(target) => {
                let cond = stack.pop().ok_or("vm: stack underflow on jump")?;
                match cond {
                    Value::Bool(false) => { ip = *target as usize; continue; }
                    Value::Num(n) if n == 0.0 => { ip = *target as usize; continue; }
                    _ => {} // truthy — fall through
                }
            }
            Op::Jump(target) => {
                ip = *target as usize;
                continue;
            }
            Op::Return => {
                return stack.pop().ok_or("vm: empty stack at return".into());
            }
        }
        ip += 1;
    }
    stack.pop().ok_or("vm: empty stack at end".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Node;

    fn make_ctx() -> CompileCtx {
        CompileCtx::new(
            &["add", "+", "subtract", "-", "multiply", "*", "abs", "negate",
              "string-upper", "string-lower", "string-length", "concat",
              "<", ">", "=", "not", "even", "odd", "if"],
            &[],
        )
    }

    #[test]
    fn test_vm_constant() {
        let ctx = make_ctx();
        let nodes = vec![Node::Num(42.0)];
        let chunk = compile(&nodes, 0, &ctx).unwrap();
        let mut stack = Vec::new();
        let result = execute(&chunk, &Value::Num(0.0), &[], &mut stack).unwrap();
        assert!(matches!(result, Value::Num(n) if n == 42.0));
    }

    #[test]
    fn test_vm_identity() {
        let ctx = make_ctx();
        // Just the symbol x
        let nodes = vec![Node::Symbol(intern("x"))];
        let chunk = compile(&nodes, 0, &ctx).unwrap();
        let mut stack = Vec::new();
        let result = execute(&chunk, &Value::Num(7.0), &[], &mut stack).unwrap();
        assert!(matches!(result, Value::Num(n) if n == 7.0));
    }

    #[test]
    fn test_vm_add() {
        let ctx = make_ctx();
        // (add x 1)
        let nodes = vec![
            Node::Symbol(intern("add")),  // 0
            Node::Symbol(intern("x")),    // 1
            Node::Num(1.0),               // 2
            Node::App(vec![0, 1, 2]),     // 3
        ];
        let chunk = compile(&nodes, 3, &ctx).unwrap();
        let mut stack = Vec::new();
        let result = execute(&chunk, &Value::Num(5.0), &[], &mut stack).unwrap();
        assert!(matches!(result, Value::Num(n) if n == 6.0));
    }

    #[test]
    fn test_vm_nested() {
        let ctx = make_ctx();
        // (add (multiply x 2) 1)
        let nodes = vec![
            Node::Symbol(intern("multiply")),  // 0
            Node::Symbol(intern("x")),         // 1
            Node::Num(2.0),                    // 2
            Node::App(vec![0, 1, 2]),          // 3: (multiply x 2)
            Node::Symbol(intern("add")),       // 4
            Node::Num(1.0),                    // 5
            Node::App(vec![4, 3, 5]),          // 6: (add (multiply x 2) 1)
        ];
        let chunk = compile(&nodes, 6, &ctx).unwrap();
        let mut stack = Vec::new();
        let result = execute(&chunk, &Value::Num(3.0), &[], &mut stack).unwrap();
        assert!(matches!(result, Value::Num(n) if n == 7.0));
    }

    #[test]
    fn test_vm_if() {
        let ctx = make_ctx();
        // (if (> x 0) x (negate x))  — abs(x)
        let nodes = vec![
            Node::Symbol(intern(">")),       // 0
            Node::Symbol(intern("x")),       // 1
            Node::Num(0.0),                  // 2
            Node::App(vec![0, 1, 2]),        // 3: (> x 0)
            Node::Symbol(intern("x")),       // 4
            Node::Symbol(intern("negate")),  // 5
            Node::Symbol(intern("x")),       // 6
            Node::App(vec![5, 6]),           // 7: (negate x)
            Node::If(3, 4, 7),              // 8
        ];
        let chunk = compile(&nodes, 8, &ctx).unwrap();
        let mut stack = Vec::new();
        let r1 = execute(&chunk, &Value::Num(5.0), &[], &mut stack).unwrap();
        assert!(matches!(r1, Value::Num(n) if n == 5.0));
        let r2 = execute(&chunk, &Value::Num(-3.0), &[], &mut stack).unwrap();
        assert!(matches!(r2, Value::Num(n) if n == 3.0));
    }

    #[test]
    fn test_vm_string() {
        let ctx = make_ctx();
        // (string-upper x)
        let nodes = vec![
            Node::Symbol(intern("string-upper")),  // 0
            Node::Symbol(intern("x")),             // 1
            Node::App(vec![0, 1]),                 // 2
        ];
        let chunk = compile(&nodes, 2, &ctx).unwrap();
        let mut stack = Vec::new();
        let result = execute(&chunk, &Value::Str("hello".into()), &[], &mut stack).unwrap();
        assert!(matches!(result, Value::Str(s) if s == "HELLO"));
    }

    #[test]
    fn test_vm_matches_tree_walker() {
        // Compare VM output with tree-walker for a representative candidate
        use std::rc::Rc;
        let ctx = make_ctx();

        // (add (multiply x x) 1)
        let mut nodes = vec![
            Node::Symbol(intern("multiply")),
            Node::Symbol(intern("x")),
            Node::Symbol(intern("x")),
            Node::App(vec![0, 1, 2]),
            Node::Symbol(intern("add")),
            Node::Num(1.0),
            Node::App(vec![4, 3, 5]),
        ];
        let body_root = 6;

        // VM path
        let chunk = compile(&nodes, body_root, &ctx).unwrap();
        let mut stack = Vec::new();

        // Tree-walker path: wrap in lambda
        let lr = nodes.len();
        nodes.push(Node::Lambda(vec![intern("x")], body_root));
        let nodes_rc: Rc<[Node]> = nodes.into();

        for x in &[0.0, 1.0, -3.0, 10.0, 0.5] {
            let input = Value::Num(*x);
            let vm_result = execute(&chunk, &input, &[], &mut stack).unwrap();

            let mut env = crate::eval::make_default_env();
            let fv = crate::eval::eval(&nodes_rc, lr, &mut env).unwrap();
            let tw_result = crate::eval::apply(&fv, &[input.clone()], &nodes_rc, &mut env).unwrap();

            match (&vm_result, &tw_result) {
                (Value::Num(a), Value::Num(b)) => {
                    assert!((a - b).abs() < f64::EPSILON,
                        "Mismatch for x={}: vm={}, tw={}", x, a, b);
                }
                _ => panic!("Type mismatch for x={}", x),
            }
        }
    }
}
