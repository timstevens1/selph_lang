//! types_v2 — sketch of the rebuilt SELPH core (April 10, 2026)
//!
//! This file is a DESIGN SKETCH, not yet wired into the build. It defines the
//! proposed Value, Env, Node, and Type representations for the post-rebuild
//! interpreter described in §9.24 of SELPH_Growing_System_Plan.md.
//!
//! Goals:
//!   1. Cheap value cloning (Rc-shared strings/lists). The hot path of every
//!      builtin currently does deep clones; that ends here.
//!   2. Cheap closure capture and call (persistent env via parent pointers).
//!      No more env.clone() of a 270-entry HashMap on every Lambda eval.
//!   3. Distinguish integers from floats at the value level. Many builtins
//!      naturally return integer counts (string-length, nth) and many synthesis
//!      decisions hinge on integer-vs-float; encoding both as f64 hides this.
//!   4. Unify functions: closures and macros become a single `Function` variant.
//!      The semantic distinction (eval-in-caller vs eval-in-captured) collapses
//!      because library "macros" are top-level definitions whose captured env
//!      already contains the rest of the library — there's no observable
//!      difference. This aligns with the decomposition philosophy: a function's
//!      captured environment IS the relevant search context for any sub-synth
//!      that uses it.
//!   5. Fast special-form dispatch via a Rust enum (no string compares).
//!   6. Open type universe — but the actual type system design is DEFERRED.
//!      The sketch leaves a placeholder hook (`Value::type_sym()`) returning a
//!      pre-interned Sym for primitive variants. The real type system will live
//!      in a SELPH namespace tree where types contain sub-types via namespace
//!      nesting. See §9.24.3 and the future curriculum stage.
//!
//! What stays the same:
//!   - Sym (u32 interned symbol) from intern.rs
//!   - Parser produces a node tree (semantics unchanged)
//!   - Lexical scope, defmacro letrec semantics, dispatch behaviour
//!
//! What's gone:
//!   - Value::Grid (use SELPH library + deftype, eventually)
//!   - Value::Alt  (synthesizer-internal concept; lift it out of Value)
//!   - Distinct Closure and RustMacro variants (collapsed into Function)
//!   - Bytecode VM (the rebuilt tree walker should match its perf)
//!   - u8 type tags (TYPE_NUM, TYPE_STR, etc. become Sym handles)
//!   - HM as currently structured (replaced by predicate-based filtering)
//!
//! INVARIANT: this file must compile in isolation. It does not yet interact
//! with eval.rs or synth.rs. When ready, the migration is module-by-module.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::intern::Sym;

// ────────────────────────────────────────────────────────────────────────────
// Value
// ────────────────────────────────────────────────────────────────────────────
//
// The Value enum is intentionally small. Almost all heap-bearing variants use
// Rc so cloning is a refcount bump. Strings and lists — the two structural
// values that get passed around the most — are Rc-shared.
//
// Note what's missing:
//   - Grid: ARC-AGI's grid is a SELPH library type, not a Rust variant. A
//     grid value at runtime is a List of List of Num; "grid-ness" is a
//     predicate registered in the Grid type namespace.
//   - Alt: the synthesizer's "any of these matches" alternative is now a
//     synth-side concept (Vec<Value> in spec processing), not a Value variant.
//   - Namespace: kept as a fast variant (Map). Hashmap lookup is hot enough
//     that paying for List-of-pairs would be a regression. Keys become Sym
//     instead of String — this drops a lot of allocations in ns-get.
//
// Cloning a Value is now O(1) for everything except Bool/Num/Nil (which are
// O(1) anyway). The previous deep-clone-of-Vec<Value> behavior is gone.

#[derive(Clone, Debug)]
pub enum Value {
    /// 64-bit signed integer. Distinct from Num so the type system and
    /// synthesizer can reason about integer-vs-float without sniffing
    /// `n == (n as i64) as f64`. Many builtins naturally produce Int:
    /// string-length, count-char, nth indices, list lengths, etc.
    Int(i64),

    /// Floating-point number — IEEE-754 double. Used for arithmetic that
    /// needs fractional precision. Coercion rules (decided during eval
    /// rewrite): Int op Int = Int (with overflow → Num); Int op Num = Num.
    Num(f64),

    /// String — Rc-shared. Cloning a string Value is a refcount bump, not
    /// a heap allocation. Construction from a literal costs one allocation.
    Str(Rc<str>),

    /// Boolean.
    Bool(bool),

    /// Heterogeneous list — Rc-shared slice. Cloning is a refcount bump.
    /// Note: this is immutable. Mutation creates a new Rc. Synthesis/eval
    /// rarely mutates lists in place; when they do, they build a Vec then
    /// freeze it via Rc::from(vec).
    List(Rc<[Value]>),

    /// Map keyed by Sym, Rc-shared. Replaces the old Namespace(HashMap<String, Value>).
    /// Sym keying drops a lot of String allocations in the ns-get path. Each
    /// insert pays one intern() but lookup is hashed-u32 instead of hashed-String.
    ///
    /// We keep this as a Rust variant (rather than a List of pairs) because
    /// namespace lookup is in the synthesis hot path. Tradeoff accepted.
    Ns(Rc<NsMap>),

    /// Function — closures, lambdas, and library "macros" all collapse here.
    /// Carries params, body, and captured env. Cheap to clone (Rc bump).
    /// The captured env IS the search context that any sub-synthesizer using
    /// this function would inherit — explicit lexical capture aligns with
    /// the decomposition philosophy.
    ///
    /// What used to be `(defmacro f (x) body)` is now equivalent to
    /// `(define f (lambda (x) body))`. The "eval in caller env" semantics
    /// of old defmacro is unnecessary because library functions are defined
    /// at top level where the captured env already includes the rest of the
    /// library.
    Function(Rc<FunctionData>),

    /// Builtin function — a Sym handle for dispatch. The Sym→fn lookup is
    /// done via a Sym-indexed Vec<BuiltinFn> for O(1) without HashMap or
    /// thread-local. See `BuiltinTable` below.
    Builtin(Sym),

    /// nil.
    Nil,
}

/// Sym-keyed map. Wrapped in Rc for cheap cloning.
pub type NsMap = HashMap<Sym, Value>;

/// Function body, parameters, and captured environment. The same struct
/// backs lambdas, closures, and library-defined functions.
///
/// Decomposition rationale: when a sub-synthesizer wants to use this
/// function as a primitive, it doesn't need to inherit the parent's full
/// search environment — it gets exactly the captured_env, which is what
/// the function actually depends on. Capture is O(1) (just an Rc bump on
/// the env node), so even huge libraries pay constant cost per definition.
#[derive(Debug)]
pub struct FunctionData {
    pub params: Vec<Sym>,
    pub body: NodeRef,
    pub captured_env: Env,
    /// Optional shared scope for letrec — the same letrec mechanism as today.
    pub letrec_scope: Option<SharedScope>,
}

/// Reference to a node by (tree, index). Carries the Rc<[Node]> for free
/// because the tree is shared with whatever created the value.
#[derive(Clone, Debug)]
pub struct NodeRef {
    pub nodes: Rc<[Node]>,
    pub idx: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Node — the AST
// ────────────────────────────────────────────────────────────────────────────
//
// Mostly the same as before, but with two key changes:
//
//   1. Special forms become a Rust enum (SpecialForm) so dispatch in
//      eval_inner doesn't go through resolve(*name).match(...).as_str().
//      This is the §9.24.1 special-form fix.
//
//   2. Node::Symbol carries a Sym (unchanged), but the parser pre-resolves
//      well-known special form symbols into Node::SpecialApp. The eval loop
//      then has zero string work in the function-application hot path.
//
// Future possibility (not in this sketch): symbol resolution at parse time
// converts every Symbol into either LocalRef(slot), BuiltinRef(sym), or
// FreeRef(sym), so eval doesn't even hit env_lookup for most variables.
// Listed in §9.24.1 Tier-3.

#[derive(Clone, Debug)]
pub enum Node {
    /// Integer literal — evaluates to Value::Int.
    Int(i64),
    /// Floating-point literal — evaluates to Value::Num. Used when a
    /// literal has a decimal point or is outside i64 range.
    Num(f64),
    Str(String),
    Bool(bool),
    Symbol(Sym),
    /// Application: (f arg1 arg2 ...). children[0] is the function.
    App(Vec<usize>),
    /// Application of a known special form. The parser pre-resolves these
    /// from the App-with-special-form-symbol pattern so eval doesn't need
    /// to string-match in the hot path.
    SpecialApp(SpecialForm, Vec<usize>),
    If(usize, usize, usize),
    Lambda(Vec<Sym>, usize),
    Let(Vec<(Sym, usize)>, usize),
}

/// All special forms recognized by the evaluator. Adding a new special form
/// means adding a variant here AND a case in eval_inner — the compiler will
/// remind you to handle it.
///
/// Note: there is no `Defmacro`. The parser desugars
/// `(defmacro name (params) body)` to `(define name (lambda (params) body))`.
/// This is sound because the post-rebuild `Function` variant captures its
/// definition-site env, which at top level contains the rest of the library —
/// the same set of bindings the old defmacro could see at call time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialForm {
    Define,
    Do,
    Quote,
    And,
    Or,
    Try,
    EvalIn,
    Dispatch,
    Ns,
    // Future: Deftype, Defstruct, ...
}

impl SpecialForm {
    /// Parser hook: try to recognize a Sym as a special form.
    /// Pre-interns the special form symbols at startup so this is just
    /// a Sym comparison (u32 ==).
    pub fn from_sym(sym: Sym) -> Option<Self> {
        SPECIAL_FORM_SYMS.with(|s| s.borrow().get(&sym).copied())
    }
}

thread_local! {
    static SPECIAL_FORM_SYMS: RefCell<HashMap<Sym, SpecialForm>> = RefCell::new(HashMap::new());
}

/// Called once at startup to populate the special-form lookup table.
/// (The real impl will live in eval.rs init or be lazy-init at first eval.)
pub fn init_special_forms() {
    use crate::intern::intern;
    SPECIAL_FORM_SYMS.with(|s| {
        let mut map = s.borrow_mut();
        map.insert(intern("define"), SpecialForm::Define);
        map.insert(intern("do"), SpecialForm::Do);
        map.insert(intern("quote"), SpecialForm::Quote);
        map.insert(intern("and"), SpecialForm::And);
        map.insert(intern("or"), SpecialForm::Or);
        map.insert(intern("try"), SpecialForm::Try);
        map.insert(intern("eval-in"), SpecialForm::EvalIn);
        map.insert(intern("dispatch"), SpecialForm::Dispatch);
        map.insert(intern("ns"), SpecialForm::Ns);
        map.insert(intern("namespace"), SpecialForm::Ns);
        // Note: `defmacro` is intentionally not here. The parser desugars it
        // to `(define name (lambda (params) body))`.
    });
}

// ────────────────────────────────────────────────────────────────────────────
// Env — persistent environment
// ────────────────────────────────────────────────────────────────────────────
//
// The big change. Today's `Env = Vec<HashMap<Sym, Value>>` is deep-cloned on
// every closure creation and call. The new Env is a chain of Rc-shared
// scopes linked by parent pointers, like a persistent linked list. Cloning
// an Env is one Rc bump.
//
// Construction:
//   - Env::new_with_default() — start a fresh env with the default scope
//     (builtins + __builtins__ namespace) at the bottom. The default scope
//     is built once at process start and shared via Rc.
//
//   - env.push_scope(scope) — return a NEW Env whose top scope is `scope`
//     and whose parent is `self`. O(1).
//
//   - env.with_binding(name, val) — return a NEW env with a single new
//     binding pushed on top. O(1) allocation of a one-entry scope.
//
// Lookup walks the linked list of scopes, top first. Each scope is a
// FxHashMap<Sym, Value> (TODO: switch to FxHashMap once we add the dep).
// The default scope sits at the bottom and is shared across all envs.
//
// IMPORTANT: closures NEVER mutate the env they captured. Mutation (define,
// let-binding) only happens in the current frame's top scope. This makes
// the persistent representation safe — you can clone an Env freely and the
// clones are independent for mutation purposes.

#[derive(Clone, Debug)]
pub struct Env {
    /// Linked list of scopes, top of stack first. Empty == nothing in scope.
    /// `top` is the scope we'd mutate on `define`. `parent` is everything
    /// inherited from the caller.
    inner: Rc<EnvNode>,
}

#[derive(Debug)]
struct EnvNode {
    /// The scope at this level — a hash map of bindings. Mutated in place
    /// only by the owner of this Rc (single-writer). Once shared, treat as
    /// immutable; new scopes go on top.
    scope: RefCell<Scope>,
    parent: Option<Rc<EnvNode>>,
}

/// A single scope. TODO: switch to FxHashMap for faster Sym hashing.
pub type Scope = HashMap<Sym, Value>;

/// Shared mutable scope for letrec — same role as today's SharedScope, but
/// it now plugs into the persistent env model cleanly.
pub type SharedScope = Rc<RefCell<Scope>>;

impl Env {
    /// Create a new env with the given bottom scope. The default env is
    /// built once at startup; subsequent envs reuse the bottom scope's Rc.
    pub fn from_scope(scope: Scope) -> Self {
        Env {
            inner: Rc::new(EnvNode {
                scope: RefCell::new(scope),
                parent: None,
            }),
        }
    }

    /// Push a fresh scope on top of this env. Used for function call frames,
    /// let bindings, and macro expansion. O(1) allocation.
    pub fn push_scope(&self, scope: Scope) -> Self {
        Env {
            inner: Rc::new(EnvNode {
                scope: RefCell::new(scope),
                parent: Some(Rc::clone(&self.inner)),
            }),
        }
    }

    /// Look up a Sym, walking the scope chain top-first. O(scope-depth)
    /// worst case but each scope is small and the bottom (default) scope
    /// has FxHashMap<Sym, Value> O(1) lookup.
    pub fn lookup(&self, sym: Sym) -> Option<Value> {
        let mut node = Some(&self.inner);
        while let Some(n) = node {
            if let Some(v) = n.scope.borrow().get(&sym).cloned() {
                return Some(v);
            }
            node = n.parent.as_ref();
        }
        None
    }

    /// Define a binding in the TOP scope. Mutates the current frame.
    /// Used by `define` special form and let bindings.
    pub fn define(&self, sym: Sym, val: Value) {
        self.inner.scope.borrow_mut().insert(sym, val);
    }

    /// Get a handle to the top scope for letrec patching.
    /// Most code shouldn't need this — only the let-binding implementation
    /// uses it for the closure-shared-scope dance.
    pub fn top_scope_mut(&self) -> std::cell::RefMut<'_, Scope> {
        self.inner.scope.borrow_mut()
    }

    /// Immutable borrow of the top scope. Used by synth_v2 to walk the
    /// library env and discover Function values for component generation.
    pub fn top_scope(&self) -> std::cell::Ref<'_, Scope> {
        self.inner.scope.borrow()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Types — placeholder hook (real design DEFERRED)
// ────────────────────────────────────────────────────────────────────────────
//
// The full type system is deferred to a later milestone. The expected shape
// (per §9.24.3 and the user's April 10 note):
//
//   - Types live in a SELPH namespace tree (__types__) where namespace
//     nesting encodes the subtype hierarchy. Number contains Int and Float
//     as sub-namespaces; subtype check is "is X reachable by descent from Y."
//   - Each type entry is itself a namespace with fields like `predicate`,
//     `priority`, etc. Inspecting and updating the type system is just
//     ordinary namespace operations on a Value::Ns tree.
//   - Curriculum stages teach type prediction, predicate learning, and
//     type-of inference as SELPH programs over this tree.
//
// What this sketch needs RIGHT NOW: a stable way for the synthesizer to ask
// "what's the canonical type Sym for this primitive Value?" so SynthComponent
// can carry Sym-typed parameters and return types instead of u8 tags. That's
// the only type-system surface that has to exist before eval.rs is rewritten.
//
// We pre-intern the primitive type Syms once and stash them in a thread-local
// for fast access. When the real type system lands, these Syms become the
// keys at the top level of the __types__ namespace.

thread_local! {
    static PRIMITIVE_TYPE_SYMS: PrimitiveTypeSyms = PrimitiveTypeSyms::new();
}

struct PrimitiveTypeSyms {
    any: Sym,
    int: Sym,
    num: Sym,
    string: Sym,
    bool_: Sym,
    list: Sym,
    function: Sym,
    namespace: Sym,
}

impl PrimitiveTypeSyms {
    fn new() -> Self {
        use crate::intern::intern;
        Self {
            any: intern("Any"),
            int: intern("Int"),
            num: intern("Num"),
            string: intern("String"),
            bool_: intern("Bool"),
            list: intern("List"),
            function: intern("Function"),
            namespace: intern("Namespace"),
        }
    }
}

/// Canonical Sym for "any type" (top of the eventual lattice).
pub fn type_any() -> Sym {
    PRIMITIVE_TYPE_SYMS.with(|s| s.any)
}

impl Value {
    /// Return the canonical primitive type Sym for this value, or None for
    /// `Nil` (which has no useful type tag).
    ///
    /// The synthesizer uses this for fast type identity on runtime values.
    /// Once the real type system lands, this will become the leaf-level
    /// fast path; non-primitive types will be discovered by walking the
    /// __types__ namespace and calling predicates.
    pub fn type_sym(&self) -> Option<Sym> {
        PRIMITIVE_TYPE_SYMS.with(|s| match self {
            Value::Int(_) => Some(s.int),
            Value::Num(_) => Some(s.num),
            Value::Str(_) => Some(s.string),
            Value::Bool(_) => Some(s.bool_),
            Value::List(_) => Some(s.list),
            Value::Function(_) | Value::Builtin(_) => Some(s.function),
            Value::Ns(_) => Some(s.namespace),
            Value::Nil => None,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Builtin dispatch — Sym-indexed table
// ────────────────────────────────────────────────────────────────────────────
//
// Today: BUILTIN_DISPATCH is a thread_local HashMap<Sym, fn(&[Value])>.
// Per-call overhead: thread_local cell deref + HashMap lookup + SipHash on a
// u32 key.
//
// New: a Vec<Option<BuiltinFn>> indexed by Sym.0 (the u32 underneath Sym).
// Lookup is `dispatch.get(sym.0 as usize).copied().flatten()`. No hash, no
// thread-local. Built once at startup and never modified.
//
// Builtins take the current env by reference. Cloning Env is one Rc bump
// so always-pass is essentially free, and it eliminates the §9.23
// Option<&mut Env> threading complexity. Higher-order builtins
// (map/filter/reduce/apply/test-spec) use the env when calling apply on
// user functions; first-order builtins ignore it.

pub type BuiltinFn = fn(args: &[Value], env: &Env) -> Result<Value, String>;

pub struct BuiltinTable {
    /// Vec indexed by Sym.0 — None for non-builtin Syms.
    /// Grows as new builtins are registered (always at startup).
    table: Vec<Option<BuiltinFn>>,
}

impl BuiltinTable {
    pub fn new() -> Self {
        Self { table: Vec::new() }
    }

    pub fn register(&mut self, sym: Sym, f: BuiltinFn) {
        let idx = sym.0 as usize;
        if idx >= self.table.len() {
            self.table.resize(idx + 1, None);
        }
        self.table[idx] = Some(f);
    }

    pub fn lookup(&self, sym: Sym) -> Option<BuiltinFn> {
        self.table.get(sym.0 as usize).copied().flatten()
    }
}

impl Default for BuiltinTable {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helper accessors — replace today's deep-clone-on-extract patterns
// ────────────────────────────────────────────────────────────────────────────
//
// Today's `fn list(v: &Value) -> Result<Vec<Value>, String>` clones the
// entire vector. The new accessors return references (or Rc clones) so the
// caller pays only for what they actually read.

impl Value {
    /// Borrow as &str without allocation. Returns Err if not a string.
    pub fn as_str(&self) -> Result<&str, String> {
        match self {
            Value::Str(s) => Ok(s.as_ref()),
            _ => Err(format!("expected string, got {:?}", self)),
        }
    }

    /// Borrow as &[Value] without cloning the slice. Returns Err if not a list.
    pub fn as_list(&self) -> Result<&[Value], String> {
        match self {
            Value::List(l) => Ok(l.as_ref()),
            _ => Err(format!("expected list, got {:?}", self)),
        }
    }

    /// Get the underlying Rc<[Value]> — refcount bump, no element clone.
    /// Use this when you need to pass ownership of a list to something that
    /// will hold onto it.
    pub fn list_rc(&self) -> Result<Rc<[Value]>, String> {
        match self {
            Value::List(l) => Ok(Rc::clone(l)),
            _ => Err(format!("expected list, got {:?}", self)),
        }
    }

    /// Construct a Value::Str from anything &str-like. Single allocation.
    pub fn str(s: impl AsRef<str>) -> Self {
        Value::Str(Rc::from(s.as_ref()))
    }

    /// Construct a Value::List from a Vec<Value>. Freezes into Rc<[Value]>.
    pub fn list(items: Vec<Value>) -> Self {
        Value::List(Rc::from(items))
    }

    /// Construct a Value::Ns from a Sym-keyed map.
    pub fn ns(map: NsMap) -> Self {
        Value::Ns(Rc::new(map))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sketch tests — does the design at least typecheck end-to-end?
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::intern;

    #[test]
    fn value_str_clone_is_cheap() {
        let s = Value::str("hello world");
        let s2 = s.clone();
        // Both should share the same underlying Rc<str>. We can't observe
        // refcount directly without unsafe, but we can verify the contents.
        assert_eq!(s.as_str().unwrap(), s2.as_str().unwrap());
    }

    #[test]
    fn value_list_clone_is_cheap() {
        let l = Value::list(vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)]);
        let l2 = l.clone();
        assert_eq!(l.as_list().unwrap().len(), 3);
        assert_eq!(l2.as_list().unwrap().len(), 3);
    }

    #[test]
    fn env_persistent_chain() {
        let mut bottom = Scope::new();
        bottom.insert(intern("x"), Value::Num(1.0));
        let env0 = Env::from_scope(bottom);

        let mut frame = Scope::new();
        frame.insert(intern("y"), Value::Num(2.0));
        let env1 = env0.push_scope(frame);

        // env1 sees both bindings
        assert!(matches!(env1.lookup(intern("x")), Some(Value::Num(n)) if n == 1.0));
        assert!(matches!(env1.lookup(intern("y")), Some(Value::Num(n)) if n == 2.0));

        // env0 only sees x
        assert!(env0.lookup(intern("y")).is_none());
        assert!(matches!(env0.lookup(intern("x")), Some(Value::Num(n)) if n == 1.0));
    }

    #[test]
    fn env_define_writes_to_top_scope() {
        let env0 = Env::from_scope(Scope::new());
        let env1 = env0.push_scope(Scope::new());
        env1.define(intern("z"), Value::Num(3.0));

        // env1 sees z in its top scope
        assert!(matches!(env1.lookup(intern("z")), Some(Value::Num(n)) if n == 3.0));
        // env0 doesn't see z (it's in env1's top, not env0's)
        assert!(env0.lookup(intern("z")).is_none());
    }

    #[test]
    fn special_form_dispatch_is_sym_lookup() {
        init_special_forms();
        let define_sym = intern("define");
        assert_eq!(SpecialForm::from_sym(define_sym), Some(SpecialForm::Define));

        let add_sym = intern("add");
        assert_eq!(SpecialForm::from_sym(add_sym), None); // not a special form
    }

    #[test]
    fn primitive_type_sym_distinguishes_int_from_num() {
        // Int and Num must report different type Syms — that's the whole
        // point of having a separate variant.
        let int_sym = Value::Int(5).type_sym().unwrap();
        let num_sym = Value::Num(5.0).type_sym().unwrap();
        assert_ne!(int_sym, num_sym);
        assert_eq!(int_sym, intern("Int"));
        assert_eq!(num_sym, intern("Num"));
    }

    #[test]
    fn primitive_type_sym_for_each_variant() {
        assert_eq!(Value::Int(0).type_sym(), Some(intern("Int")));
        assert_eq!(Value::Num(0.0).type_sym(), Some(intern("Num")));
        assert_eq!(Value::str("hi").type_sym(), Some(intern("String")));
        assert_eq!(Value::Bool(true).type_sym(), Some(intern("Bool")));
        assert_eq!(Value::list(vec![]).type_sym(), Some(intern("List")));
        assert_eq!(Value::Nil.type_sym(), None);
    }

    #[test]
    fn function_carries_captured_env() {
        // A Function value captures whatever env it was created in. Cloning
        // the function is one Rc bump and the captured env shares its inner
        // Rc<EnvNode> with the original.
        let mut bottom = Scope::new();
        bottom.insert(intern("library_helper"), Value::Int(42));
        let env = Env::from_scope(bottom);

        let f = Value::Function(Rc::new(FunctionData {
            params: vec![intern("x")],
            body: NodeRef {
                nodes: Rc::from(vec![Node::Symbol(intern("x"))]),
                idx: 0,
            },
            captured_env: env.clone(),
            letrec_scope: None,
        }));

        let f2 = f.clone();
        // Both clones see the same captured library_helper.
        if let (Value::Function(fd1), Value::Function(fd2)) = (&f, &f2) {
            assert!(matches!(fd1.captured_env.lookup(intern("library_helper")),
                             Some(Value::Int(42))));
            assert!(matches!(fd2.captured_env.lookup(intern("library_helper")),
                             Some(Value::Int(42))));
        } else {
            panic!("expected Function values");
        }
    }

    #[test]
    fn builtin_table_lookup() {
        let mut t = BuiltinTable::new();
        let add = intern("add");
        fn fake_add(args: &[Value], _env: &Env) -> Result<Value, String> {
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                _ => Err("add: expected integers".into()),
            }
        }
        t.register(add, fake_add);

        let f = t.lookup(add).expect("registered builtin not found");
        let env = Env::from_scope(Scope::new());
        let r = f(&[Value::Int(2), Value::Int(3)], &env).unwrap();
        assert!(matches!(r, Value::Int(n) if n == 5));
    }
}
