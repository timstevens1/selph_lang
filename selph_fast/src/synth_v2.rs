//! synth_v2 — bottom-up enumerative synthesis against types_v2 (April 10, 2026)
//!
//! Pairs with `types_v2.rs` and `eval_v2.rs`. This is the §9.24.5 step-5 work:
//! the new synthesizer built against the new Value/Env/Node types and the
//! Sym-keyed type system.
//!
//! Step 1 (this file, current scope): SynthComponent + the type-reachability
//! data structure.
//!
//! What's different from `synth.rs`:
//!
//!   1. **Types are first-class Syms, not u8 tags.** A component's parameter
//!      types and return type are stored as `Vec<Sym>` and `Sym`, where the
//!      Sym keys into the (eventual) `__types__` namespace. For now those
//!      Syms are the primitive type Syms pre-interned in `types_v2`
//!      (`Int`, `Num`, `String`, `Bool`, `List`, `Function`, `Namespace`,
//!      and `Any`). When the full type system lands, the same `Sym`-shaped
//!      slot will hold user-defined type names without any code change here.
//!
//!   2. **The reachability check is a graph search over Sym keys**, not a
//!      `HashSet<u8>` bitmap. Same algorithmic complexity, but the universe
//!      is open. Adding a new type means adding a key to the `TypeUniverse`
//!      table, not editing a Rust `match`.
//!
//!   3. **There is no `infer_macro_types`.** Components for user-defined
//!      functions are built by probing the function with sample inputs and
//!      reading `Value::type_sym()` off the result. The probe step doesn't
//!      need to know about every primitive type variant — it just asks the
//!      Value what it is. This generalizes for free when the type system
//!      grows: a probe returning a Grid (eventually a SELPH-defined type)
//!      will return `Some(grid_sym)` from a real predicate-based check, not
//!      `Some(some-rust-enum-variant)`.
//!
//! What stays the same:
//!   - Bottom-up enumeration with priority-ordered pools
//!   - Observational equivalence dedup
//!   - The "domain reachability" pruning idea (input/output types seed a
//!     reachable set; components whose params or return are unreachable get
//!     dropped before enumeration starts)
//!
//! INVARIANT: this file must compile in isolation against `types_v2`,
//! `eval_v2`, and `intern`. It does not yet wire into the CLI or replace
//! `synth.rs`. Step 2 will start building components from the eval_v2
//! builtin table; step 3 will port the enumeration loop.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use std::rc::Rc;

use crate::eval_v2;
use crate::intern::{intern, Sym};
use crate::types_v2::{type_any, Env, Node, SpecialForm, Value};

// ────────────────────────────────────────────────────────────────────────────
// SynthComponent — a primitive or user-defined function available for synth
// ────────────────────────────────────────────────────────────────────────────
//
// Compared to `synth::SynthComponent`:
//   - `name: String`            stays as String for now (debug/printing).
//   - `builtin: Option<String>` becomes `dispatch: Dispatch`, an enum that
//     captures whether this component is a builtin (looked up by Sym in the
//     eval_v2 BuiltinTable), a library function (looked up by Sym in the
//     synth env), a literal constant, or a synthetic fused form like
//     `map_<macro>` / `reduce_<macro>`.
//   - `arity: usize`            stays.
//   - `param_types: Vec<u8>`    becomes `Vec<Sym>` of type Syms.
//   - `ret_type: u8`            becomes `Sym`.
//   - `priority`, `usage_count` stay.
//
// The Sym slots use the primitive type Syms from `types_v2` for now
// (`Int`, `Num`, `String`, `Bool`, `List`, `Function`, `Namespace`,
// or `Any`). When the full predicate-based type system lands, the slot
// keeps the same shape — only the set of Syms widens.

#[derive(Clone, Debug)]
pub struct SynthComponent {
    pub name: String,
    pub dispatch: Dispatch,
    pub arity: usize,
    pub param_types: Vec<Sym>,
    pub ret_type: Sym,
    pub priority: f64,
    /// How many prior tasks in this curriculum run used this component.
    pub usage_count: f64,
}

/// How a component is invoked at materialization time.
///
/// `synth.rs` overloads `Option<String>` for this — None means "literal
/// constant baked in by name", Some("map_foo") means "fused higher-order
/// form for macro foo", and Some("add") means "real builtin named add".
/// We split those out so the synth_v2 materializer doesn't have to do
/// substring checks.
#[derive(Clone, Debug)]
pub enum Dispatch {
    /// A literal value emitted directly into the candidate tree (e.g. `0`,
    /// `"a"`, the input variable `x`). Carries the literal Value so the
    /// materializer doesn't have to re-parse the name.
    Literal(LiteralKind),
    /// A builtin or library function looked up by Sym in the synth env.
    /// At materialization time the candidate emits a `Node::Symbol(sym)`
    /// in function position; eval_v2 resolves it to a `Value::Builtin` or
    /// `Value::Function` from the env.
    Named(Sym),
    /// Synthetic fused higher-order form. Materializer expands this into
    /// a real `(map fn arg)` or `(reduce fn arg)` application using the
    /// inner function Sym.
    FusedMap(Sym),
    FusedReduce(Sym),
}

/// Literal constants we want available as zero-arity components.
/// Kept as a small enum so the materializer doesn't reparse strings.
#[derive(Clone, Debug)]
pub enum LiteralKind {
    /// The synthesis input variable (resolved at apply-time, not eval-time).
    InputVar,
    Int(i64),
    Str(String),
    Bool(bool),
}

impl SynthComponent {
    /// Construct a literal-constant component (arity 0).
    pub fn literal(name: impl Into<String>, kind: LiteralKind, ret_type: Sym, priority: f64) -> Self {
        Self {
            name: name.into(),
            dispatch: Dispatch::Literal(kind),
            arity: 0,
            param_types: vec![],
            ret_type,
            priority,
            usage_count: 0.0,
        }
    }

    /// Construct a builtin/named component.
    pub fn named(
        name: impl Into<String>,
        sym: Sym,
        param_types: Vec<Sym>,
        ret_type: Sym,
        priority: f64,
    ) -> Self {
        let name = name.into();
        Self {
            name,
            dispatch: Dispatch::Named(sym),
            arity: param_types.len(),
            param_types,
            ret_type,
            priority,
            usage_count: 0.0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TypeUniverse — open Sym-keyed type universe
// ────────────────────────────────────────────────────────────────────────────
//
// In `synth.rs`, the type universe was a closed `u8` tag set. Reachability
// was a `HashSet<u8>` and a few hard-coded `match` arms. That's the bit
// the rebuild is moving away from.
//
// Here, the universe is an open Sym set. The synthesizer asks the
// `TypeUniverse` two things:
//
//   1. **Is this Sym a known type?** (used when validating components.)
//   2. **Which Syms are "reachable" from this seed set, given these
//      components?** (used to prune entire domains before enumeration.)
//
// For step 1 the universe is just the primitive type Syms registered by
// `types_v2`. When the predicate-based type system lands, the universe
// becomes the set of names in `__types__`, populated by `(deftype ...)`
// SELPH calls. Reachability is unchanged; only how the universe is
// populated differs.

#[derive(Clone, Debug)]
pub struct TypeUniverse {
    /// All known type Syms. Membership is "this Sym is a valid type
    /// identifier in the current synth context."
    known: HashSet<Sym>,
    /// `Any` is special: it satisfies any reachability query and any
    /// param-type filter.
    any: Sym,
    /// Cached primitive Syms used by the (currently hard-coded) subtype
    /// rule. When the predicate-based type system lands, the subtype
    /// relation moves into the `__types__` namespace and these go away.
    int_sym: Sym,
    num_sym: Sym,
}

impl TypeUniverse {
    /// Build a universe seeded with the primitive type Syms from
    /// `types_v2`. Equivalent to today's `{NUM, STR, BOOL, LIST, ANY}`,
    /// but as Sym keys, and with `Int` distinct from `Num` (related by
    /// the `Int <: Num` subtype rule below).
    pub fn primitives() -> Self {
        let mut known = HashSet::new();
        for name in &["Int", "Num", "String", "Bool", "List", "Function", "Namespace"] {
            known.insert(intern(name));
        }
        let any = type_any();
        known.insert(any);
        let int_sym = intern("Int");
        let num_sym = intern("Num");
        Self {
            known,
            any,
            int_sym,
            num_sym,
        }
    }

    /// Register an additional type Sym (for the eventual `(deftype ...)`
    /// integration — not used in step 1).
    pub fn register(&mut self, sym: Sym) {
        self.known.insert(sym);
    }

    /// True if `sym` is a recognized type in this universe. `Any` is always
    /// recognized.
    pub fn is_known(&self, sym: Sym) -> bool {
        self.known.contains(&sym)
    }

    /// `Any` Sym (top of the lattice).
    pub fn any(&self) -> Sym {
        self.any
    }

    /// True if a parameter slot of declared type `slot` accepts a value
    /// whose actual type is `actual`. Both sides are widened by `Any`,
    /// matching the legacy `TYPE_ANY` behaviour. Subtype rules are
    /// hard-coded for now (just `Int <: Num`); the predicate-based type
    /// system will replace this with a `__types__` namespace walk.
    pub fn slot_accepts(&self, slot: Sym, actual: Sym) -> bool {
        if slot == actual || slot == self.any || actual == self.any {
            return true;
        }
        // Int <: Num — an Int value flows into a Num slot.
        if slot == self.num_sym && actual == self.int_sym {
            return true;
        }
        false
    }

    /// True if a parameter slot of type `pt` is satisfiable given the
    /// concrete types in `reach`. This is the reachability-analysis
    /// counterpart to `slot_accepts` and intentionally has different
    /// semantics:
    ///
    ///   - `slot_accepts` is about flowing one runtime value into one
    ///     slot (so `slot_accepts(Bool, Any) = true` — an `Any`-typed
    ///     value can satisfy a `Bool` slot at runtime).
    ///   - `slot_satisfiable` is about whether the synthesizer can
    ///     statically *construct* something that fits the slot from
    ///     what it has so far. Having `Any` in the reach set should NOT
    ///     make every slot satisfiable; otherwise reachability collapses
    ///     to the entire universe and the prune step does nothing.
    ///
    /// So `slot_satisfiable` treats `Any` only as a sentinel for "this
    /// slot is intentionally unconstrained" (when `pt == Any`). It does
    /// not honour `Any` on the reach side.
    pub fn slot_satisfiable(&self, pt: Sym, reach: &HashSet<Sym>) -> bool {
        if pt == self.any || reach.contains(&pt) {
            return true;
        }
        // Subtype: a `Num` slot is satisfiable when reach contains `Int`.
        if pt == self.num_sym && reach.contains(&self.int_sym) {
            return true;
        }
        false
    }

    /// True if a component returning type `ret` should be considered
    /// useful given the types currently in the `useful` set. Like
    /// `slot_satisfiable`, this is the analysis counterpart to
    /// `slot_accepts` and does NOT honour `Any` on the useful side
    /// (otherwise the entire universe would be useful and the prune
    /// step would be a no-op).
    ///
    /// A return type IS useful when:
    ///   - it equals a useful type, or
    ///   - it is `Any` (statically unknown — keep, can't rule out), or
    ///   - it is a subtype of a useful type (e.g. an `Int`-returning
    ///     component is useful when we want a `Num`).
    pub fn ret_useful(&self, ret: Sym, useful: &HashSet<Sym>) -> bool {
        if ret == self.any || useful.contains(&ret) {
            return true;
        }
        // Subtype: an `Int` return is useful when `Num` is wanted.
        if ret == self.int_sym && useful.contains(&self.num_sym) {
            return true;
        }
        false
    }

    /// Compute the set of types reachable from a seed set using a forward
    /// closure: starting from `seeds`, repeatedly add the return types of
    /// any component whose parameters are all reachable, until fixpoint.
    ///
    /// Mirrors the `synth.rs` "useful types" pass but over Sym keys
    /// instead of u8 keys, with subtype propagation via
    /// `slot_satisfiable`.
    pub fn forward_reachable(&self, seeds: &[Sym], components: &[SynthComponent]) -> HashSet<Sym> {
        let mut reach: HashSet<Sym> = seeds.iter().copied().collect();
        loop {
            let mut changed = false;
            for comp in components {
                // Constants always contribute their return type.
                if comp.arity == 0 {
                    if reach.insert(comp.ret_type) {
                        changed = true;
                    }
                    continue;
                }
                // Non-constants contribute their return type once all
                // their parameter slots are satisfied by something in
                // the reach set.
                let all_reach = comp
                    .param_types
                    .iter()
                    .all(|pt| self.slot_satisfiable(*pt, &reach));
                if all_reach && reach.insert(comp.ret_type) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        reach
    }

    /// Backward-reachable set: which types are "useful" given a target
    /// output type? Starts from `target`, then iteratively adds the
    /// parameter types of any component whose return type is already
    /// useful. Mirrors the `synth.rs` "useful set" pass with subtype
    /// propagation via `ret_useful`.
    ///
    /// Returns the universe (everything is useful) when `target` is None.
    pub fn backward_useful(
        &self,
        target: Option<Sym>,
        components: &[SynthComponent],
    ) -> HashSet<Sym> {
        let target = match target {
            Some(t) => t,
            None => return self.known.clone(),
        };
        let mut useful: HashSet<Sym> = HashSet::new();
        useful.insert(target);
        loop {
            let mut changed = false;
            for comp in components {
                if comp.arity == 0 {
                    continue;
                }
                if self.ret_useful(comp.ret_type, &useful) {
                    for &pt in &comp.param_types {
                        if useful.insert(pt) {
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        useful
    }

    /// The reachable-types pruning step: combine forward and backward
    /// closure to keep only components that are *both* buildable from the
    /// seed inputs *and* on the path to the target output. This is the
    /// per-task entry point most callers will use.
    pub fn reachable_for_task(
        &self,
        input_types: &[Sym],
        target_output: Option<Sym>,
        components: &[SynthComponent],
    ) -> HashSet<Sym> {
        let forward = self.forward_reachable(input_types, components);
        let backward = self.backward_useful(target_output, components);
        forward.intersection(&backward).copied().collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Component-level reachability filter
// ────────────────────────────────────────────────────────────────────────────

/// Keep only components whose parameter and return types are all in the
/// reachable set. This is the per-task pruning step that runs once at the
/// start of synthesis to drop entire domains (e.g. all string ops on a
/// purely numeric task) before enumeration begins.
pub fn filter_components_by_reach<'a>(
    components: &'a [SynthComponent],
    reach: &HashSet<Sym>,
    universe: &TypeUniverse,
) -> Vec<&'a SynthComponent> {
    components
        .iter()
        .filter(|comp| {
            // Constants: keep if their return type is in reach (or is
            // statically Any — we can't rule them out).
            if comp.arity == 0 {
                return reach.contains(&comp.ret_type) || comp.ret_type == universe.any();
            }
            // Non-constants: every parameter slot must be satisfiable
            // by something in reach, and the return type must be in
            // reach (or Any).
            let params_ok = comp
                .param_types
                .iter()
                .all(|pt| universe.slot_satisfiable(*pt, reach));
            let ret_ok = reach.contains(&comp.ret_type) || comp.ret_type == universe.any();
            params_ok && ret_ok
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Probe-driven type discovery (replaces infer_macro_types)
// ────────────────────────────────────────────────────────────────────────────
//
// `synth.rs::infer_macro_types` is gone. The replacement is much smaller:
// when we want to know what type a runtime Value is, we ask it directly
// via `Value::type_sym()`. The synth_v2 component builder probes a
// user-defined function with sample inputs (just like the old code did),
// but instead of pattern-matching the result against a closed Rust enum,
// it just reads the type Sym off the returned Value. New types added to
// the universe become probe-discoverable for free.
//
// The actual probing call (which needs an Env and access to eval_v2::apply)
// belongs to step 2. This function is the post-probe analysis: given an
// optional successful result Value, return its primitive type Sym.

/// Map a runtime Value to a primitive type Sym for synth purposes.
/// Returns `Any` for `Nil` (which has no useful type tag) so the
/// component slot is unconstrained rather than missing.
pub fn type_sym_or_any(v: &Value) -> Sym {
    v.type_sym().unwrap_or_else(type_any)
}

// ────────────────────────────────────────────────────────────────────────────
// Default component catalog
// ────────────────────────────────────────────────────────────────────────────
//
// `primitive_components()` returns the static catalog of builtin SELPH
// operations available to the synthesizer, with first-class Sym types.
// This mirrors `synth::default_synth_components_opts(_, false)` but:
//
//   - Types are Syms, not u8 tags.
//   - Numeric arithmetic is typed `Num × Num → Num`, relying on the
//     `Int <: Num` subtype rule to let Int values flow into Num slots.
//     Eval_v2 still preserves Int when both operands are Int (the
//     coercion rule from §9.25.1).
//   - The input variable `x` is NOT included here. The synthesis driver
//     adds it per-task with the actual input type via `input_var_component`.
//   - Grid components are NOT included. They become a SELPH library in
//     a later milestone (per §9.24.3 / §9.25.1).
//
// Naming/Sym convention: each named component's `Dispatch::Named(sym)` is
// `intern(<name>)`, where `<name>` is exactly the symbol the eval_v2 env
// resolves to a `Value::Builtin`. The materializer will emit
// `Node::Symbol(sym)` in function position; eval_v2 looks up the builtin.

/// Build the static primitive component catalog. This is the default
/// "library of operations" the synthesizer searches over before any
/// task-specific or user-defined components are added.
pub fn primitive_components() -> Vec<SynthComponent> {
    let int = intern("Int");
    let num = intern("Num");
    let str_ = intern("String");
    let bool_ = intern("Bool");
    let list_ = intern("List");
    let any = type_any();

    let mut comps: Vec<SynthComponent> = Vec::new();

    // ── Integer literal constants ──────────────────────────────────────
    // The legacy catalog seeded the search with small integer literals.
    // We keep the same set, typed as Int (not Num) — synth_v2 prefers
    // Int-typed pool entries when the target is Int, and the subtype
    // rule lets them flow into Num slots when the target is Num.
    for n in [0i64, 1, 2, 3, 4, 5, 6, 7, 10, -1] {
        comps.push(SynthComponent::literal(
            n.to_string(),
            LiteralKind::Int(n),
            int,
            0.0,
        ));
    }

    // ── String literal constants for formal-language tasks ─────────────
    for s in ["a", "b", "(", ")"] {
        comps.push(SynthComponent::literal(
            s,
            LiteralKind::Str(s.into()),
            str_,
            0.0,
        ));
    }

    // ── Arithmetic ────────────────────────────────────────────────────
    // Typed as Int × Int → Int. The eval_v2 builtins are actually
    // generic numeric (they accept both Int and Num), but the curriculum
    // is overwhelmingly integer-valued and the type system doesn't yet
    // model the polymorphic case `(α, α) → α where α ∈ {Int, Num}`.
    // Float-using tasks can opt in via `divide` / `floor` (typed Num)
    // and the polymorphism gap is on the docket for the type-system
    // milestone (post §9.24.3 deftype work).
    //
    // The legacy `TYPE_NUM` collapsed Int and Num into one tag, so no
    // existing curriculum depends on the distinction.
    for name in &["abs", "negate"] {
        comps.push(SynthComponent::named(
            *name, intern(name), vec![int], int, 0.0,
        ));
    }
    for name in &["add", "subtract", "multiply", "min", "max", "modulo"] {
        comps.push(SynthComponent::named(
            *name, intern(name), vec![int, int], int, 0.0,
        ));
    }
    // divide and floor stay Num — they're the explicit float entry points.
    comps.push(SynthComponent::named(
        "divide", intern("divide"), vec![num, num], num, 0.0,
    ));
    comps.push(SynthComponent::named(
        "floor", intern("floor"), vec![num], num, 0.0,
    ));

    // ── String unary str→str ───────────────────────────────────────────
    for name in &["string-upper", "string-lower", "string-reverse", "string-trim"] {
        comps.push(SynthComponent::named(
            *name, intern(name), vec![str_], str_, 0.0,
        ));
    }

    // ── String → numeric / character ──────────────────────────────────
    // Note: in synth_v2, string-length returns Int (eval_v2 produces an
    // Int). Legacy used TYPE_NUM for both. The subtype rule means Num
    // targets still see this component as useful.
    comps.push(SynthComponent::named(
        "string-length",
        intern("string-length"),
        vec![str_],
        int,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "string-nth",
        intern("string-nth"),
        vec![str_, int],
        str_,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "char-code",
        intern("char-code"),
        vec![str_],
        int,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "code-char",
        intern("code-char"),
        vec![int],
        str_,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "count-char",
        intern("count-char"),
        vec![str_, str_],
        int,
        0.0,
    ));

    // ── String × String → String ──────────────────────────────────────
    comps.push(SynthComponent::named(
        "string-replace",
        intern("string-replace"),
        vec![str_, str_, str_],
        str_,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "concat",
        intern("concat"),
        vec![str_, str_],
        str_,
        0.0,
    ));

    // ── String predicates ─────────────────────────────────────────────
    for name in &["string-starts-with", "string-ends-with"] {
        comps.push(SynthComponent::named(
            *name,
            intern(name),
            vec![str_, str_],
            bool_,
            0.0,
        ));
    }

    // ── String slicing ────────────────────────────────────────────────
    comps.push(SynthComponent::named(
        "string-take",
        intern("string-take"),
        vec![str_, int],
        str_,
        5.0,
    ));
    comps.push(SynthComponent::named(
        "string-drop",
        intern("string-drop"),
        vec![str_, int],
        str_,
        5.0,
    ));
    comps.push(SynthComponent::named(
        "string-slice",
        intern("string-slice"),
        vec![str_, int, int],
        str_,
        5.0,
    ));

    // ── Comparison ────────────────────────────────────────────────────
    // Typed Int × Int → Bool for the same reason as arithmetic.
    for name in &["<", ">", "<=", ">=", "=", "!="] {
        comps.push(SynthComponent::named(
            *name,
            intern(name),
            vec![int, int],
            bool_,
            0.0,
        ));
    }

    // ── Numeric predicates ────────────────────────────────────────────
    for name in &["even", "odd"] {
        comps.push(SynthComponent::named(
            *name, intern(name), vec![int], bool_, 0.0,
        ));
    }

    // ── Boolean logic ─────────────────────────────────────────────────
    comps.push(SynthComponent::named(
        "not", intern("not"), vec![bool_], bool_, 0.0,
    ));
    comps.push(SynthComponent::named(
        "and", intern("and"), vec![bool_, bool_], bool_, 0.0,
    ));
    comps.push(SynthComponent::named(
        "or", intern("or"), vec![bool_, bool_], bool_, 0.0,
    ));

    // ── List operations ───────────────────────────────────────────────
    comps.push(SynthComponent::named(
        "string-split",
        intern("string-split"),
        vec![str_, str_],
        list_,
        10.0,
    ));
    comps.push(SynthComponent::named(
        "string-join",
        intern("string-join"),
        vec![list_, str_],
        str_,
        10.0,
    ));
    comps.push(SynthComponent::named(
        "head",
        intern("head"),
        vec![list_],
        any,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "tail",
        intern("tail"),
        vec![list_],
        list_,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "nth",
        intern("nth"),
        vec![list_, int],
        any,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "list-length",
        intern("length"),
        vec![list_],
        int,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "list-reverse",
        intern("reverse"),
        vec![list_],
        list_,
        0.0,
    ));
    comps.push(SynthComponent::named(
        "string-chars",
        intern("string-chars"),
        vec![str_],
        list_,
        5.0,
    ));

    // ── Namespace ──────────────────────────────────────────────────────
    comps.push(SynthComponent::named(
        "ns-get",
        intern("ns-get"),
        vec![any, str_],
        any,
        10.0,
    ));
    comps.push(SynthComponent::named(
        "ns-get-or",
        intern("ns-get-or"),
        vec![any, str_, any],
        any,
        10.0,
    ));

    // ── dispatch ──────────────────────────────────────────────────────
    // (dispatch name-string arg) — look up function by name in env, apply.
    comps.push(SynthComponent::named(
        "dispatch",
        intern("dispatch"),
        vec![str_, any],
        any,
        15.0,
    ));

    comps
}

/// Build the input-variable component for a task. The synthesizer uses
/// this as the "x" pool seed; its return type comes from the actual
/// input value the task supplies.
pub fn input_var_component(input_type: Sym) -> SynthComponent {
    SynthComponent::literal("x", LiteralKind::InputVar, input_type, 100.0)
}

// ────────────────────────────────────────────────────────────────────────────
// Library function components — discovered by walking + probing the env
// ────────────────────────────────────────────────────────────────────────────
//
// The legacy `default_synth_components` takes a list of macro definitions
// `(name, params, nodes, root)` extracted from the parser, builds a
// throwaway env via `make_default_env`, registers them, and probes each
// one with sample inputs to infer types via the closed Rust enum.
//
// In the new world, library functions live in the eval_v2 env as
// `Value::Function` values. Loading a library file is just `eval(...)`
// against `make_default_env()`. So the synth-side workflow becomes:
//
//   1. `make_default_env()` and eval the library file into it.
//   2. `library_components_from_env(&env)` walks the top scope, finds
//      every `Value::Function` (skipping builtins, the input var, etc.),
//      and probes each one to discover its type signature.
//   3. The probe step calls `eval_v2::apply` with sample inputs and
//      reads `Value::type_sym()` off the result. No closed enum to
//      pattern-match; new types just work.

/// Sample values used for probing library functions to discover their
/// type signatures. Order matters — earlier samples are tried first,
/// so a function that succeeds on Int gets typed as Int even if it
/// would also succeed on Num. The single-param probe additionally
/// tries the alternates to detect generic functions.
fn probe_samples() -> Vec<(Sym, Value)> {
    vec![
        (intern("Int"), Value::Int(3)),
        (intern("String"), Value::str("5 1 2 3 4")),
        (intern("List"), Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)])),
        (intern("Num"), Value::Num(3.0)),
        (intern("Bool"), Value::Bool(true)),
    ]
}

/// Probe a function value to discover its parameter and return types.
/// Tries uniform-type argument vectors in `probe_samples` order; the
/// first one that doesn't error wins. For single-parameter functions,
/// also probes the alternate types to detect generic functions
/// (returns `Any` as the param type if more than one type works).
///
/// Returns `None` if no probe succeeded — caller decides what to do
/// (skip the function, default to all-Any, etc.).
fn probe_function_type(
    func: &Value,
    arity: usize,
    env: &Env,
) -> Option<(Vec<Sym>, Sym)> {
    if arity == 0 {
        // Zero-arg: just call it.
        if let Ok(result) = eval_v2::apply(func, &[], env) {
            return Some((vec![], type_sym_or_any(&result)));
        }
        return None;
    }

    let samples = probe_samples();
    let mut chosen: Option<(Sym, Value)> = None;

    for (sym, val) in &samples {
        let args: Vec<Value> = vec![val.clone(); arity];
        if let Ok(result) = eval_v2::apply(func, &args, env) {
            chosen = Some((*sym, result));
            break;
        }
    }

    let (param_sym, result) = chosen?;
    let mut param_types = vec![param_sym; arity];
    let ret = type_sym_or_any(&result);

    // For single-param functions, probe the other types to detect
    // genericity. If more than one type works, mark the parameter as
    // Any. (Multi-param functions could in principle do this per-slot
    // but we keep parity with legacy and only do the unary case.)
    if arity == 1 {
        let any = type_any();
        let mut accepted_count = 1;
        for (sym, val) in &samples {
            if *sym == param_sym {
                continue;
            }
            if eval_v2::apply(func, std::slice::from_ref(val), env).is_ok() {
                accepted_count += 1;
                if accepted_count >= 2 {
                    param_types[0] = any;
                    break;
                }
            }
        }
    }

    Some((param_types, ret))
}

/// Walk the top scope of `env` and build a SynthComponent for each
/// `Value::Function` entry. Probes each function with sample inputs to
/// discover its type signature.
///
/// Builtins are skipped — they're already in the primitive catalog.
/// Functions whose probes all fail are skipped (we have no way to know
/// what they accept). The caller can also pass a `skip` set to exclude
/// names that are already represented as primitives or input variables.
pub fn library_components_from_env(env: &Env, skip: &HashSet<Sym>) -> Vec<SynthComponent> {
    let mut out = Vec::new();
    // Snapshot the scope into a Vec to release the borrow before we
    // call eval_v2::apply (which reborrows env scopes during evaluation).
    let entries: Vec<(Sym, Value)> = {
        let scope = env.top_scope();
        scope
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    };
    for (sym, val) in entries {
        if skip.contains(&sym) {
            continue;
        }
        let func_data = match &val {
            Value::Function(fd) => fd,
            _ => continue, // builtins and other non-function bindings
        };
        let arity = func_data.params.len();
        let Some((param_types, ret_type)) = probe_function_type(&val, arity, env) else {
            continue;
        };
        let name = crate::intern::resolve(sym);
        out.push(SynthComponent {
            name,
            dispatch: Dispatch::Named(sym),
            arity,
            param_types,
            ret_type,
            priority: 30.0,
            usage_count: 0.0,
        });
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// Fused higher-order components
// ────────────────────────────────────────────────────────────────────────────
//
// Mirrors the legacy `map_<macro>` and `reduce_<macro>` fused forms:
// for each unary library function f, generate a synthetic
// `Dispatch::FusedMap(f_sym)` component with signature `List → List`;
// for each binary library function f with a known return type,
// generate a `Dispatch::FusedReduce(f_sym)` component with signature
// `List → ret`. The materializer will expand these into real
// `(map f x)` / `(reduce f x)` applications at candidate emission time.
//
// The "fused binary builtin" cases (`reduce_add`, `reduce_concat`, etc.)
// from legacy are emitted directly here against the primitive catalog
// so library-discovery doesn't have to special-case them.

/// Built-in binary functions that fuse cleanly into a `reduce_<f>`
/// component, paired with their (Sym-typed) return type.
fn reducible_binary_builtins() -> Vec<(&'static str, Sym)> {
    let num = intern("Num");
    let str_ = intern("String");
    vec![
        ("add", num),
        ("subtract", num),
        ("multiply", num),
        ("min", num),
        ("max", num),
        ("concat", str_),
    ]
}

/// Generate fused `map_<f>` and `reduce_<f>` components from the
/// library-function components in `library`, plus fused-reduce
/// components for built-in binary functions.
pub fn fused_components(library: &[SynthComponent]) -> Vec<SynthComponent> {
    let list_ = intern("List");
    let mut out = Vec::new();

    // Fused map for each unary library function.
    for comp in library {
        if comp.arity == 1 {
            if let Dispatch::Named(inner) = comp.dispatch {
                out.push(SynthComponent {
                    name: format!("map_{}", comp.name),
                    dispatch: Dispatch::FusedMap(inner),
                    arity: 1,
                    param_types: vec![list_],
                    ret_type: list_,
                    priority: 25.0,
                    usage_count: 0.0,
                });
            }
        }
    }

    // Fused reduce for each binary library function whose return type
    // we successfully probed. (Binary functions with `ret_type == Any`
    // still get a fused-reduce, but typed `List → Any`, which lets the
    // synthesizer use them as fallback.)
    for comp in library {
        if comp.arity == 2 {
            if let Dispatch::Named(inner) = comp.dispatch {
                out.push(SynthComponent {
                    name: format!("reduce_{}", comp.name),
                    dispatch: Dispatch::FusedReduce(inner),
                    arity: 1,
                    param_types: vec![list_],
                    ret_type: comp.ret_type,
                    priority: 25.0,
                    usage_count: 0.0,
                });
            }
        }
    }

    // Fused reduce for built-in binary functions with known return types.
    for (name, ret) in reducible_binary_builtins() {
        out.push(SynthComponent {
            name: format!("reduce_{}", name),
            dispatch: Dispatch::FusedReduce(intern(name)),
            arity: 1,
            param_types: vec![list_],
            ret_type: ret,
            priority: 25.0,
            usage_count: 0.0,
        });
    }

    out
}

// ────────────────────────────────────────────────────────────────────────────
// Top-level builder
// ────────────────────────────────────────────────────────────────────────────

/// Build the full synthesis component catalog from a library env.
/// Combines the static primitive catalog, library-discovered components,
/// and fused higher-order forms. Does NOT include the input variable
/// `x` — the synthesis driver adds it per-task with the actual input
/// type via `input_var_component`.
///
/// `skip` is the set of Syms in `env` that should be ignored when
/// building library components — typically the names of builtins
/// (already in the primitive catalog) and `nil`/`true`/`false` (which
/// are bound to non-Function values anyway, so they'd be filtered).
/// Pass `&default_skip_set()` if you don't have task-specific skips.
pub fn default_synth_components(env: &Env, skip: &HashSet<Sym>) -> Vec<SynthComponent> {
    let primitives = primitive_components();
    let library = library_components_from_env(env, skip);
    let fused = fused_components(&library);
    let mut all = primitives;
    all.extend(library);
    all.extend(fused);
    all
}

/// The default "skip" set: every Sym that the eval_v2 default scope
/// binds. These are either builtins (already in the primitive catalog)
/// or constants (`nil`, `true`, `false`) that don't make sense as
/// callable components. Computed lazily by snapshotting the default env.
pub fn default_skip_set() -> HashSet<Sym> {
    let env = eval_v2::make_default_env();
    env.top_scope().keys().copied().collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Strategy dispatch (§9.25.3 step 5)
// ────────────────────────────────────────────────────────────────────────────
//
// The legacy curriculum runner has six "strategies" arranged as a fallback
// chain:
//
//   0. Recursive Decomposition (RD)  — top-down family prediction
//   1. Flat                          — bottom-up enumeration
//   2. Bool Decomposition (BD)       — combine bool macros via and/or/not
//   3. Higher-Order Decomposition    — template patterns (map, split, ...)
//   4. Divide and Conquer (D&C)      — group by output, sub-solve
//   5. Memorization (Memo)           — namespace lookup table
//
// (Induction also exists as a fallback between BD and HO; the §9.25.3
// plan lists "Flat, BD, HO, D&C, Memo" — five strategies — which roughly
// matches the reorganized post-§9.22 ordering minus RD itself.)
//
// **synth_v2 strategy scope (step 5):**
//   - **Flat** — already done in step 3 as `synthesize`.
//   - **Memo** — ported here. Smallest legacy strategy (~80 lines), fits
//     a string-input lookup pattern that several curriculum tasks rely on.
//   - **BD, HO, D&C, Induction, RD** — explicitly DEFERRED. Each is a
//     substantial port (500–2500 legacy lines) and warrants its own focused
//     sub-iteration of step 5. The dispatcher below is set up to make
//     adding them purely additive — just register a new `Strategy`
//     variant and route to it in `synthesize_with_strategies`.
//
// The dispatcher pattern is the load-bearing piece for step 6: the
// eval_v2 bucket-6 `synthesize` builtin will call
// `synthesize_with_strategies`, not `synthesize` directly, so adding a
// strategy later is invisible to the curriculum.

/// Which strategy was used to solve a task. Used for tracing and to
/// signal which fallback fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Flat,
    Memo,
    // Future: BoolDecomposition, HigherOrder, DivideAndConquer,
    //         Induction, RecursiveDecomposition.
}

impl Strategy {
    pub fn name(&self) -> &'static str {
        match self {
            Strategy::Flat => "Flat",
            Strategy::Memo => "Memo",
        }
    }
}

/// Result of a strategy-dispatched synthesis call. Mirrors `SynthResult`
/// but tags which strategy produced the solution.
#[derive(Debug)]
pub struct StrategyResult {
    pub found: bool,
    pub nodes: Option<Vec<Node>>,
    pub root: Option<usize>,
    pub candidates_explored: usize,
    pub strategy: Option<Strategy>,
}

impl StrategyResult {
    fn from_synth(r: SynthResult, strategy: Strategy) -> Self {
        Self {
            found: r.found,
            nodes: r.nodes,
            root: r.root,
            candidates_explored: r.candidates_explored,
            strategy: if r.found { Some(strategy) } else { None },
        }
    }

    fn not_found(explored: usize) -> Self {
        Self {
            found: false,
            nodes: None,
            root: None,
            candidates_explored: explored,
            strategy: None,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Memorization strategy
// ────────────────────────────────────────────────────────────────────────────
//
// When inputs are all strings and the input→output mapping is consistent,
// emit a hand-written lookup-table program:
//
//   (lambda (x) (ns-get-or (ns ("k1" v1) ("k2" v2) ...) x default))
//
// This is the legacy "Memo" fallback from `main.rs::memorize_from_examples`,
// ported to types_v2 nodes. The output value type is captured in the
// default — Bool default is `false`, Num is `0`, Int is `0`, Str is `""`,
// otherwise `nil`.
//
// Memo is the simplest "give up and recall" path. It's not synthesis in
// any creative sense — it just turns the training set into a function.
// But several curriculum tasks have small fixed input domains, and Memo
// solves them in O(1) candidates instead of exhausting Flat's budget.

/// Try to solve `(inputs, expected)` by emitting a namespace lookup
/// table. Returns `None` if any input isn't a string or if any input
/// maps to multiple distinct outputs (the table would be ambiguous).
pub fn memorize_from_examples(
    inputs: &[Value],
    expected: &[Value],
) -> Option<(Vec<Node>, usize)> {
    if inputs.is_empty() || inputs.len() != expected.len() {
        return None;
    }
    // All inputs must be strings — namespace keys are Sym-interned.
    if !inputs.iter().all(|v| matches!(v, Value::Str(_))) {
        return None;
    }

    // Deduplicate: same input must map to same output (otherwise the
    // memorized lookup is ill-defined).
    let mut map: HashMap<String, &Value> = HashMap::new();
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        if let Value::Str(key) = inp {
            let k = key.as_ref().to_string();
            if let Some(existing) = map.get(&k) {
                if !eval_v2::values_equal(existing, exp) {
                    return None; // conflicting outputs for same input
                }
            }
            map.insert(k, exp);
        }
    }

    // Default value type comes from the first expected output. New in
    // synth_v2: `Int` is distinct from `Num`.
    let default_node = match &expected[0] {
        Value::Bool(_) => Node::Bool(false),
        Value::Int(_) => Node::Int(0),
        Value::Num(_) => Node::Num(0.0),
        Value::Str(_) => Node::Str(String::new()),
        _ => Node::Bool(false),
    };

    // Build the AST: (lambda (x) (ns-get-or (ns (k v) ...) x default)).
    let mut nodes: Vec<Node> = Vec::new();

    // (ns (k1 v1) (k2 v2) ...) — note: in eval_v2, `ns` is a SpecialForm,
    // so the call site is `Node::SpecialApp(SpecialForm::Ns, ...)`. Each
    // child is a `Node::App(vec![key_idx, val_idx])` pair.
    let mut pair_children: Vec<usize> = Vec::with_capacity(map.len());
    for (key, val) in &map {
        let key_idx = nodes.len();
        nodes.push(Node::Str(key.clone()));
        let val_idx = nodes.len();
        nodes.push(value_to_node(val));
        let pair_idx = nodes.len();
        nodes.push(Node::App(vec![key_idx, val_idx]));
        pair_children.push(pair_idx);
    }
    let ns_idx = nodes.len();
    nodes.push(Node::SpecialApp(SpecialForm::Ns, pair_children));

    // x parameter reference
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    // default value
    let default_idx = nodes.len();
    nodes.push(default_node);

    // ns-get-or call: (ns-get-or (ns ...) x default)
    let ngo_idx = nodes.len();
    nodes.push(Node::Symbol(intern("ns-get-or")));
    let body_idx = nodes.len();
    nodes.push(Node::App(vec![ngo_idx, ns_idx, x_idx, default_idx]));

    // (lambda (x) body)
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], body_idx));

    Some((nodes, lambda_idx))
}

/// Convert a runtime `Value` into a leaf `Node`. Used by `memorize` to
/// embed the expected outputs as literals inside the generated namespace.
/// Non-leaf values (lists, namespaces, functions) collapse to a `false`
/// placeholder — Memo only handles flat scalar outputs in practice.
fn value_to_node(v: &Value) -> Node {
    match v {
        Value::Int(n) => Node::Int(*n),
        Value::Num(n) => Node::Num(*n),
        Value::Bool(b) => Node::Bool(*b),
        Value::Str(s) => Node::Str(s.as_ref().to_string()),
        _ => Node::Bool(false),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Strategy dispatcher
// ────────────────────────────────────────────────────────────────────────────

/// Try each strategy in order until one finds a solution. Returns the
/// first successful result tagged with the strategy that produced it.
///
/// Current order:
///   1. **Flat** — `synthesize` (bottom-up enumeration with type pruning,
///      observational dedup, and the probe-and-filter pipeline).
///   2. **Memo** — `memorize_from_examples` (namespace lookup table for
///      string-input tasks).
///
/// `flat_budget` is the candidate budget for the Flat strategy. Memo
/// is O(1) and ignores it.
pub fn synthesize_with_strategies(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    flat_budget: usize,
) -> StrategyResult {
    // Strategy 1: Flat enumerative.
    let flat = synthesize(
        components, inputs, expected, env, universe, max_depth, flat_budget,
    );
    let mut total_explored = flat.candidates_explored;
    if flat.found {
        return StrategyResult::from_synth(flat, Strategy::Flat);
    }

    // Strategy 2: Memo. Always candidate-cost 0 (no enumeration).
    if let Some((nodes, root)) = memorize_from_examples(inputs, expected) {
        return StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: total_explored,
            strategy: Some(Strategy::Memo),
        };
    }

    StrategyResult::not_found(total_explored)
}

// ────────────────────────────────────────────────────────────────────────────
// Probe-and-filter: drop library functions that error on real inputs
// ────────────────────────────────────────────────────────────────────────────
//
// This is the §9.25.3 step-4 work. The legacy `synth.rs` does this pass
// inline inside `synthesize_full`; in synth_v2 it's a free function so
// callers (and tests) can apply it independently.
//
// Motivation: library functions accumulated across curricula often
// assume specific input formats (e.g. "5 1 2 3 4" rather than "hello").
// Reachability pruning keeps them in the catalog because the type
// signature looks compatible, but they raise at runtime on the actual
// task input. Probing once with the first input — and dropping anything
// that errors — saves the synthesizer from repeatedly enumerating
// dead-end candidates.
//
// What gets probed:
//   - Unary components (the legacy heuristic).
//   - With `Dispatch::Named(sym)` AND `env.lookup(sym) == Value::Function(_)`.
//     Builtins are never probed — they're the ground truth.
//   - Whose first parameter type matches the inferred task input type
//     (under `slot_accepts`, so subtypes flow). Components meant for
//     intermediate compositions (e.g. a `String → String` helper used
//     mid-program when the input is `Int`) are left alone.
//
// What does NOT get probed:
//   - Multi-arg functions (we don't know what to put in the other slots).
//   - Fused higher-order forms — their inner function gets probed via
//     the unary check.
//   - Literals.
//
// Keeping the surface tight matches legacy and avoids accidentally
// dropping useful components for edge-case reasons.

/// Filter out unary library functions that error when called with
/// `test_input`. Builtins, multi-arg functions, fused forms, and
/// functions with mismatched first-parameter type are passed through
/// unchanged.
///
/// This is a fast, conservative filter — it only drops components on
/// hard errors from the actual call. A function that returns the wrong
/// type or wrong value is left in (synthesis will discover that during
/// the test step).
pub fn probe_filter_components(
    components: &[SynthComponent],
    env: &Env,
    test_input: &Value,
    input_type: Sym,
    universe: &TypeUniverse,
) -> Vec<SynthComponent> {
    components
        .iter()
        .filter(|comp| {
            // Only probe arity-1 Named components.
            if comp.arity != 1 {
                return true;
            }
            let sym = match comp.dispatch {
                Dispatch::Named(s) => s,
                _ => return true, // literals, fused forms — pass through
            };
            // Look up the binding. If it's not a Value::Function, leave
            // alone (builtins / missing bindings are not the target of
            // this filter).
            let val = match env.lookup(sym) {
                Some(v) => v,
                None => return true,
            };
            if !matches!(val, Value::Function(_)) {
                return true;
            }
            // Only probe when the parameter slot accepts the actual
            // input type. Functions intended for intermediate composition
            // (e.g. a num→num helper when the task input is a string)
            // are left alone — they may still be useful inside larger
            // candidate expressions.
            if !universe.slot_accepts(comp.param_types[0], input_type) {
                return true;
            }
            // Probe: call the function with the first input. Drop on
            // any error.
            eval_v2::apply(&val, std::slice::from_ref(test_input), env).is_ok()
        })
        .cloned()
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Bottom-up enumerative synthesizer (minimal core loop)
// ────────────────────────────────────────────────────────────────────────────
//
// This is the §9.25.3 step-3 work: a clean port of the legacy
// `synthesize_full` core loop, stripped to the essentials. Things
// intentionally deferred:
//
//   - Parallel candidate evaluation (rayon path)
//   - VM fast path (the VM is being deleted in step 7)
//   - RL reward propagation / partial-match priority boosts
//   - Early-extension probes (heuristic; comes later)
//   - SELPH-programmable depth filter
//   - Snapshot recording for meta-opt
//   - Held-out validation examples
//   - Auto-extraction of literal constants from examples
//   - Arity-3 components (deferred to step 5 with strategies)
//   - Macro probe-and-filter pipeline (step 4)
//   - if-expression generation (separate strategy in step 5)
//
// What this gives us: bottom-up enumeration that handles arity-1 and
// arity-2 components, type-gates with the Sym universe, deduplicates
// by observational equivalence, builds the lambda + evaluates per
// (input, expected) pair via eval_v2, and returns the first matching
// candidate. Enough to actually solve toy tasks against the new core.

/// A flattened candidate expression in the synthesis pool.
/// Mirrors `synth::SynthPool` but uses `types_v2::Node` and `Sym` types.
#[derive(Clone, Debug)]
pub struct SynthPool {
    pub nodes: Vec<Node>,
    pub root: usize,
    pub ret_type: Sym,
    pub priority: f64,
}

/// Result of a synthesis search.
#[derive(Debug)]
pub struct SynthResult {
    pub found: bool,
    pub nodes: Option<Vec<Node>>,
    pub root: Option<usize>,
    pub candidates_explored: usize,
}

impl SynthResult {
    fn not_found(explored: usize) -> Self {
        Self {
            found: false,
            nodes: None,
            root: None,
            candidates_explored: explored,
        }
    }

    fn success(nodes: Vec<Node>, root: usize, explored: usize) -> Self {
        Self {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: explored,
        }
    }
}

/// Lightweight pending-candidate descriptor — stores indices into the
/// pool, the producing component, the inferred return type, and a score.
/// Materialized into a real `SynthPool` only when about to be tested.
#[derive(Debug)]
struct PendingCandidate {
    comp_idx: usize,
    args: Vec<usize>,
    ret_type: Sym,
    score: f64,
}

/// Outcome of testing a single candidate.
#[derive(Debug)]
enum TestOutcome {
    /// Matched all training examples — synthesis is done.
    Solution,
    /// Evaluated cleanly, didn't fully match. Add to pool.
    Tested,
    /// Eval errored on at least one example. Don't add to pool —
    /// any composition would error too.
    Errored,
    /// Type-gated: candidate's return type can't match the target.
    /// Add to pool — sub-expressions of the wrong type still compose.
    Skipped,
    /// Observational equivalence dedup — same outputs as a previous
    /// candidate. Don't add to pool.
    Deduped,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Stable hash of a Value for observational-equivalence dedup.
/// Function/Builtin/Ns hash to a constant (no useful structural hash).
pub fn val_hash(v: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    match v {
        Value::Int(n) => {
            0u8.hash(&mut h);
            n.hash(&mut h);
        }
        Value::Num(n) => {
            1u8.hash(&mut h);
            n.to_bits().hash(&mut h);
        }
        Value::Str(s) => {
            2u8.hash(&mut h);
            s.as_ref().hash(&mut h);
        }
        Value::Bool(b) => {
            3u8.hash(&mut h);
            b.hash(&mut h);
        }
        Value::List(l) => {
            4u8.hash(&mut h);
            for item in l.iter() {
                val_hash(item).hash(&mut h);
            }
        }
        Value::Nil => {
            5u8.hash(&mut h);
        }
        Value::Function(_) | Value::Builtin(_) | Value::Ns(_) => {
            6u8.hash(&mut h);
        }
    }
    h.finish()
}

/// Infer the canonical type Sym shared by every value in `vs`.
/// Returns `None` if the values disagree on their primitive type.
pub fn infer_uniform_type_sym(vs: &[Value]) -> Option<Sym> {
    let first = vs.first()?.type_sym()?;
    for v in vs.iter().skip(1) {
        if v.type_sym() != Some(first) {
            return None;
        }
    }
    Some(first)
}

/// Remap every node-index reference inside `node` by adding `offset`.
/// Used when concatenating sub-expression node trees during materialization.
fn remap_node(node: &Node, offset: usize) -> Node {
    match node {
        Node::App(c) => Node::App(c.iter().map(|i| i + offset).collect()),
        Node::SpecialApp(form, c) => {
            Node::SpecialApp(*form, c.iter().map(|i| i + offset).collect())
        }
        Node::If(a, b, c) => Node::If(a + offset, b + offset, c + offset),
        Node::Lambda(p, b) => Node::Lambda(p.clone(), b + offset),
        Node::Let(bs, b) => Node::Let(
            bs.iter().map(|(n, i)| (*n, i + offset)).collect(),
            b + offset,
        ),
        // Leaf nodes carry no indices.
        Node::Int(_) | Node::Num(_) | Node::Str(_) | Node::Bool(_) | Node::Symbol(_) => {
            node.clone()
        }
    }
}

/// Build a depth-0 atom pool entry for a literal/input-var component.
/// Returns `None` if the component isn't a literal (caller should not
/// call this on arity > 0 components).
fn materialize_atom(comp: &SynthComponent) -> Option<SynthPool> {
    let node = match &comp.dispatch {
        Dispatch::Literal(LiteralKind::InputVar) => Node::Symbol(intern("x")),
        Dispatch::Literal(LiteralKind::Int(n)) => Node::Int(*n),
        Dispatch::Literal(LiteralKind::Str(s)) => Node::Str(s.clone()),
        Dispatch::Literal(LiteralKind::Bool(b)) => Node::Bool(*b),
        _ => return None,
    };
    Some(SynthPool {
        nodes: vec![node],
        root: 0,
        ret_type: comp.ret_type,
        priority: comp.priority,
    })
}

/// Build a SynthPool entry that represents `(comp arg1 arg2 ...)` by
/// concatenating the argument node trees and emitting the appropriate
/// function-call node based on `comp.dispatch`.
fn materialize_app(comp: &SynthComponent, args: &[&SynthPool]) -> SynthPool {
    debug_assert!(args.len() == comp.arity);

    let mut nodes: Vec<Node> = Vec::new();
    let mut arg_roots: Vec<usize> = Vec::with_capacity(args.len());

    for arg in args {
        let off = nodes.len();
        for n in &arg.nodes {
            nodes.push(remap_node(n, off));
        }
        arg_roots.push(arg.root + off);
    }

    let app_root = match &comp.dispatch {
        Dispatch::Named(sym) => {
            let fn_idx = nodes.len();
            nodes.push(Node::Symbol(*sym));
            let app_idx = nodes.len();
            let mut children = Vec::with_capacity(arg_roots.len() + 1);
            children.push(fn_idx);
            children.extend_from_slice(&arg_roots);
            nodes.push(Node::App(children));
            app_idx
        }
        Dispatch::FusedMap(inner_sym) => {
            // (map inner_sym arg) — arity-1 expansion of the fused form.
            debug_assert_eq!(arg_roots.len(), 1);
            let map_idx = nodes.len();
            nodes.push(Node::Symbol(intern("map")));
            let inner_idx = nodes.len();
            nodes.push(Node::Symbol(*inner_sym));
            let app_idx = nodes.len();
            nodes.push(Node::App(vec![map_idx, inner_idx, arg_roots[0]]));
            app_idx
        }
        Dispatch::FusedReduce(inner_sym) => {
            // (reduce inner_sym arg) — arity-1 expansion of the fused form.
            debug_assert_eq!(arg_roots.len(), 1);
            let reduce_idx = nodes.len();
            nodes.push(Node::Symbol(intern("reduce")));
            let inner_idx = nodes.len();
            nodes.push(Node::Symbol(*inner_sym));
            let app_idx = nodes.len();
            nodes.push(Node::App(vec![reduce_idx, inner_idx, arg_roots[0]]));
            app_idx
        }
        Dispatch::Literal(_) => {
            unreachable!("materialize_app called on a literal component (arity 0)")
        }
    };

    // Priority composition: arity 1 sums; arity 2+ averages. Matches the
    // legacy heuristic from synth.rs.
    let arg_priority_term: f64 = if args.len() == 1 {
        args[0].priority
    } else {
        let sum: f64 = args.iter().map(|a| a.priority).sum();
        sum / args.len() as f64
    };

    SynthPool {
        nodes,
        root: app_root,
        ret_type: comp.ret_type,
        priority: comp.priority + arg_priority_term,
    }
}

/// Wrap a candidate body in `(lambda (x) <body>)`. The result is the
/// program the synthesizer ultimately returns when a candidate matches —
/// a unary lambda binding the synthesis input variable.
fn wrap_lambda(entry: &SynthPool) -> (Vec<Node>, usize) {
    let mut nodes = entry.nodes.clone();
    let body_idx = entry.root;
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], body_idx));
    (nodes, lambda_idx)
}

/// Test a single candidate against all (input, expected) pairs.
/// Wraps it in a unary lambda, evaluates the lambda once, then applies
/// it to each input and compares against the expected output.
///
/// Updates `seen` for observational dedup. The `target` parameter is
/// the inferred uniform output type (`None` if examples disagree); when
/// `Some`, candidates whose return type can't satisfy the target slot
/// are skipped.
fn test_candidate(
    entry: &SynthPool,
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    seen: &mut HashSet<Vec<u64>>,
    target: Option<Sym>,
    universe: &TypeUniverse,
) -> TestOutcome {
    // Type gate. The candidate's return type must be able to flow into
    // a slot of type `target`.
    if let Some(t) = target {
        if !universe.slot_accepts(t, entry.ret_type) {
            return TestOutcome::Skipped;
        }
    }

    // Wrap as lambda and evaluate once to get the function value.
    let (nodes, lambda_idx) = wrap_lambda(entry);
    let nodes_rc: Rc<[Node]> = nodes.into();
    let f = match eval_v2::eval(&nodes_rc, lambda_idx, env) {
        Ok(v) => v,
        Err(_) => return TestOutcome::Errored,
    };

    let mut beh: Vec<u64> = Vec::with_capacity(inputs.len());
    let mut matches = 0usize;
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        match eval_v2::apply(&f, &[inp.clone()], env) {
            Ok(v) => {
                beh.push(val_hash(&v));
                if eval_v2::values_equal(&v, exp) {
                    matches += 1;
                }
            }
            Err(_) => return TestOutcome::Errored,
        }
    }

    // Observational equivalence dedup.
    if !beh.is_empty() {
        if seen.contains(&beh) {
            return TestOutcome::Deduped;
        }
        seen.insert(beh);
    }

    if matches == inputs.len() && !inputs.is_empty() {
        TestOutcome::Solution
    } else {
        TestOutcome::Tested
    }
}

/// Bottom-up enumerative synthesis against types_v2.
///
/// Searches for a unary lambda `(lambda (x) <body>)` such that applying
/// it to each `inputs[i]` yields `expected[i]`. Returns the first
/// matching candidate found, in priority-weighted enumeration order.
///
/// `env` should already contain everything the candidate body might
/// need — primitive builtins (always present in `eval_v2::make_default_env`),
/// any library functions whose Syms appear in `components`, and any
/// extra task-specific bindings the caller wants visible. The
/// synthesizer never mutates the env.
///
/// `components` is the static catalog of operations to enumerate over.
/// `input_var_component(input_type)` is added automatically as the `x`
/// pool seed, where `input_type` is inferred from `inputs`.
pub fn synthesize(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> SynthResult {
    // Empty examples: nothing to fit. Return failure rather than
    // returning an arbitrary trivial program.
    if inputs.is_empty() || inputs.len() != expected.len() {
        return SynthResult::not_found(0);
    }

    // ── Task type inference ────────────────────────────────────────────
    let input_type = infer_uniform_type_sym(inputs).unwrap_or_else(type_any);
    let target = infer_uniform_type_sym(expected);

    // ── Reachability prune ─────────────────────────────────────────────
    let seeds = vec![input_type];
    let reach = universe.reachable_for_task(&seeds, target, components);
    let scoped_refs = filter_components_by_reach(components, &reach, universe);

    // Build the working component list. We clone here so the input-var
    // component can live alongside the others without lifetime gymnastics.
    let scoped: Vec<SynthComponent> =
        scoped_refs.iter().map(|c| (*c).clone()).collect();

    // ── Probe-and-filter ──────────────────────────────────────────────
    // Drop unary library functions that error on the first task input.
    // Cheap pass; saves the enumerator from repeatedly hitting the same
    // dead end mid-search.
    let probed = probe_filter_components(&scoped, env, &inputs[0], input_type, universe);

    let mut all_components = probed;
    all_components.push(input_var_component(input_type));

    // ── Depth 0: build atom pool from constants + input variable ──────
    let mut pool: Vec<SynthPool> = Vec::new();
    for comp in &all_components {
        if comp.arity != 0 {
            continue;
        }
        if let Some(entry) = materialize_atom(comp) {
            pool.push(entry);
        }
    }

    let mut explored: usize = 0;
    let mut seen: HashSet<Vec<u64>> = HashSet::new();

    // Test depth-0 atoms.
    for entry in &pool {
        if explored >= max_candidates {
            return SynthResult::not_found(explored);
        }
        explored += 1;
        match test_candidate(entry, inputs, expected, env, &mut seen, target, universe) {
            TestOutcome::Solution => {
                let (n, r) = wrap_lambda(entry);
                return SynthResult::success(n, r, explored);
            }
            _ => {}
        }
    }

    // ── Depth 1..max_depth: bottom-up composition ──────────────────────
    let mut prev_start: usize = 0;
    let mut prev_end: usize = pool.len();

    for _depth in 1..=max_depth {
        let prev_range_start = prev_start;
        let prev_range_end = prev_end;
        let all_end = prev_end;

        // Generate type-valid candidates for this depth.
        let mut pending: Vec<PendingCandidate> = Vec::new();

        for (ci, comp) in all_components.iter().enumerate() {
            if comp.arity == 0 {
                continue;
            }

            if comp.arity == 1 {
                for pi in prev_range_start..prev_range_end {
                    let p = &pool[pi];
                    if !universe.slot_accepts(comp.param_types[0], p.ret_type) {
                        continue;
                    }
                    let score = comp.priority + p.priority;
                    pending.push(PendingCandidate {
                        comp_idx: ci,
                        args: vec![pi],
                        ret_type: comp.ret_type,
                        score,
                    });
                }
            } else if comp.arity == 2 {
                // Case 1: arg1 from prev (current depth), arg2 from anywhere.
                for p1i in prev_range_start..prev_range_end {
                    let p1 = &pool[p1i];
                    if !universe.slot_accepts(comp.param_types[0], p1.ret_type) {
                        continue;
                    }
                    for p2i in 0..all_end {
                        let p2 = &pool[p2i];
                        if !universe.slot_accepts(comp.param_types[1], p2.ret_type) {
                            continue;
                        }
                        let score = comp.priority + (p1.priority + p2.priority) / 2.0;
                        pending.push(PendingCandidate {
                            comp_idx: ci,
                            args: vec![p1i, p2i],
                            ret_type: comp.ret_type,
                            score,
                        });
                    }
                }
                // Case 2: arg1 from older depths, arg2 from prev.
                // Avoids duplicating Case 1 when both args are at prev.
                for p1i in 0..prev_range_start {
                    let p1 = &pool[p1i];
                    if !universe.slot_accepts(comp.param_types[0], p1.ret_type) {
                        continue;
                    }
                    for p2i in prev_range_start..prev_range_end {
                        let p2 = &pool[p2i];
                        if !universe.slot_accepts(comp.param_types[1], p2.ret_type) {
                            continue;
                        }
                        let score = comp.priority + (p1.priority + p2.priority) / 2.0;
                        pending.push(PendingCandidate {
                            comp_idx: ci,
                            args: vec![p1i, p2i],
                            ret_type: comp.ret_type,
                            score,
                        });
                    }
                }
            }
            // Arity-3 deferred to step 5.
        }

        // Priority-weighted enumeration: high-score candidates first.
        pending.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut new_entries: Vec<SynthPool> = Vec::new();

        for desc in pending {
            if explored >= max_candidates {
                return SynthResult::not_found(explored);
            }
            explored += 1;

            let comp = &all_components[desc.comp_idx];
            let arg_refs: Vec<&SynthPool> =
                desc.args.iter().map(|&i| &pool[i]).collect();
            let entry = materialize_app(comp, &arg_refs);

            match test_candidate(&entry, inputs, expected, env, &mut seen, target, universe) {
                TestOutcome::Solution => {
                    let (n, r) = wrap_lambda(&entry);
                    return SynthResult::success(n, r, explored);
                }
                TestOutcome::Tested | TestOutcome::Skipped => {
                    new_entries.push(entry);
                }
                TestOutcome::Errored | TestOutcome::Deduped => {
                    // Don't add: errored candidates compose into more
                    // errors; deduped candidates have a behavioural
                    // equivalent already in the pool.
                }
            }
        }

        if new_entries.is_empty() {
            // No progress this depth — search has converged. No point
            // going deeper without new sub-expressions.
            break;
        }

        prev_start = pool.len();
        pool.extend(new_entries);
        prev_end = pool.len();
    }

    SynthResult::not_found(explored)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(s: &str) -> Sym {
        intern(s)
    }

    fn comp_named(name: &str, params: Vec<Sym>, ret: Sym) -> SynthComponent {
        SynthComponent::named(name, sym(name), params, ret, 0.0)
    }

    #[test]
    fn primitive_universe_knows_basic_types() {
        let u = TypeUniverse::primitives();
        assert!(u.is_known(sym("Int")));
        assert!(u.is_known(sym("Num")));
        assert!(u.is_known(sym("String")));
        assert!(u.is_known(sym("Bool")));
        assert!(u.is_known(sym("List")));
        assert!(u.is_known(sym("Any")));
        // Int and Num are distinct (the whole point of Value::Int).
        assert_ne!(sym("Int"), sym("Num"));
    }

    #[test]
    fn slot_accepts_handles_any() {
        let u = TypeUniverse::primitives();
        let any = u.any();
        assert!(u.slot_accepts(sym("Int"), sym("Int")));
        assert!(!u.slot_accepts(sym("Int"), sym("Num")));
        assert!(u.slot_accepts(any, sym("Num")));
        assert!(u.slot_accepts(sym("Num"), any));
    }

    #[test]
    fn forward_reachable_includes_seeds_and_outputs() {
        let u = TypeUniverse::primitives();
        let comps = vec![
            // string-length: String -> Int
            comp_named("string-length", vec![sym("String")], sym("Int")),
            // add: (Int, Int) -> Int
            comp_named("add", vec![sym("Int"), sym("Int")], sym("Int")),
            // grid-width: List -> Int   (mock; in the real universe a
            // grid is a typed List in the future, but for the reachability
            // test all we care about is the Sym wiring)
            comp_named("grid-width", vec![sym("List")], sym("Int")),
        ];
        // Seed with String only — should reach Int via string-length, but
        // not via grid-width (List unreachable).
        let reach = u.forward_reachable(&[sym("String")], &comps);
        assert!(reach.contains(&sym("String")));
        assert!(reach.contains(&sym("Int")));
        // List is NOT reachable from String alone.
        assert!(!reach.contains(&sym("List")));
    }

    #[test]
    fn backward_useful_prunes_unrelated_domains() {
        let u = TypeUniverse::primitives();
        let comps = vec![
            comp_named("string-upper", vec![sym("String")], sym("String")),
            comp_named("add", vec![sym("Int"), sym("Int")], sym("Int")),
            comp_named("string-length", vec![sym("String")], sym("Int")),
        ];
        // Target = String. string-upper produces String (params: String,
        // useful). add produces Int but Int is NOT useful for a String
        // target unless something pulls it back in — and nothing here
        // does, so Int should NOT appear in the useful set.
        let useful = u.backward_useful(Some(sym("String")), &comps);
        assert!(useful.contains(&sym("String")));
        assert!(!useful.contains(&sym("Int")));
    }

    #[test]
    fn backward_useful_propagates_through_consumers() {
        let u = TypeUniverse::primitives();
        let comps = vec![
            // string-length: String -> Int  (consumes String, produces Int)
            comp_named("string-length", vec![sym("String")], sym("Int")),
            // double: Int -> Int            (consumes Int, produces Int)
            comp_named("double", vec![sym("Int")], sym("Int")),
        ];
        // Target = Int. double produces Int (params Int → Int useful).
        // string-length produces Int (params String → String useful via
        // double's iteration pulling in Int, then string-length pulling in
        // String). So both Int AND String must be in the useful set.
        let useful = u.backward_useful(Some(sym("Int")), &comps);
        assert!(useful.contains(&sym("Int")));
        assert!(useful.contains(&sym("String")));
    }

    #[test]
    fn reachable_for_task_intersects_forward_and_backward() {
        let u = TypeUniverse::primitives();
        let comps = vec![
            // String -> Int (forward chain from String input to Int output)
            comp_named("string-length", vec![sym("String")], sym("Int")),
            // Bool -> Bool (totally irrelevant — neither input nor target)
            comp_named("not", vec![sym("Bool")], sym("Bool")),
            // List -> List (also irrelevant for this task)
            comp_named("reverse", vec![sym("List")], sym("List")),
        ];
        let reach = u.reachable_for_task(&[sym("String")], Some(sym("Int")), &comps);
        // Both seed and target should be in the reachable set.
        assert!(reach.contains(&sym("String")));
        assert!(reach.contains(&sym("Int")));
        // Bool was not seeded and is not in the path to Int — should be
        // dropped. (Note: the legacy synth.rs always added BOOL to the
        // reachable set as a convenience for if-conditions. synth_v2
        // treats that as a separate concern and lets the if-expression
        // generator opt in explicitly.)
        assert!(!reach.contains(&sym("Bool")));
        // List is similarly irrelevant.
        assert!(!reach.contains(&sym("List")));
    }

    #[test]
    fn filter_components_drops_unreachable_domains() {
        let u = TypeUniverse::primitives();
        let comps = vec![
            comp_named("string-length", vec![sym("String")], sym("Int")),
            comp_named("string-upper", vec![sym("String")], sym("String")),
            comp_named("add", vec![sym("Int"), sym("Int")], sym("Int")),
            comp_named("grid-width", vec![sym("List")], sym("Int")),
            comp_named("not", vec![sym("Bool")], sym("Bool")),
        ];
        let reach = u.reachable_for_task(&[sym("String")], Some(sym("Int")), &comps);
        let kept = filter_components_by_reach(&comps, &reach, &u);
        let kept_names: Vec<&str> = kept.iter().map(|c| c.name.as_str()).collect();
        // string-length: kept (String is in reach, Int is in reach)
        assert!(kept_names.contains(&"string-length"));
        // string-upper: kept too (String → String, both in reach via
        // backward_useful from Int target → string-length pulls in String)
        assert!(kept_names.contains(&"string-upper"));
        // grid-width: dropped (List is unreachable from String input)
        assert!(!kept_names.contains(&"grid-width"));
        // not: dropped (Bool unreachable)
        assert!(!kept_names.contains(&"not"));
    }

    #[test]
    fn type_sym_or_any_returns_any_for_nil() {
        assert_eq!(type_sym_or_any(&Value::Nil), type_any());
    }

    #[test]
    fn type_sym_or_any_distinguishes_int_from_num() {
        assert_eq!(type_sym_or_any(&Value::Int(0)), sym("Int"));
        assert_eq!(type_sym_or_any(&Value::Num(0.0)), sym("Num"));
        assert_ne!(type_sym_or_any(&Value::Int(0)), type_sym_or_any(&Value::Num(0.0)));
    }

    #[test]
    fn int_subtype_of_num_in_slot_accepts() {
        let u = TypeUniverse::primitives();
        // Int value flows into a Num slot.
        assert!(u.slot_accepts(sym("Num"), sym("Int")));
        // Num value does NOT flow into an Int slot (Int is the narrower).
        assert!(!u.slot_accepts(sym("Int"), sym("Num")));
    }

    #[test]
    fn int_subtype_propagates_through_reachability() {
        let u = TypeUniverse::primitives();
        let comps = vec![
            // string-length: String -> Int
            comp_named("string-length", vec![sym("String")], sym("Int")),
            // multiply: (Num, Num) -> Num   (the typical primitive shape)
            comp_named("multiply", vec![sym("Num"), sym("Num")], sym("Num")),
        ];
        // Forward from String: gets Int via string-length, then multiply
        // should fire because Int satisfies its Num parameter slots
        // (subtype rule).
        let reach = u.forward_reachable(&[sym("String")], &comps);
        assert!(reach.contains(&sym("String")));
        assert!(reach.contains(&sym("Int")));
        assert!(reach.contains(&sym("Num")));
    }

    #[test]
    fn int_returning_component_useful_for_num_target() {
        let u = TypeUniverse::primitives();
        let comps = vec![
            // string-length: String -> Int
            comp_named("string-length", vec![sym("String")], sym("Int")),
            // floor: Num -> Num
            comp_named("floor", vec![sym("Num")], sym("Num")),
        ];
        // Target = Num. string-length produces Int — under the subtype
        // rule an Int return is useful for a Num target, so its String
        // parameter should be pulled into the useful set even though
        // nothing here has Int as a *parameter*. (The useful set tracks
        // "types we want to consume", which is why Int itself isn't in
        // it — only types that appear as parameter slots get added.)
        let useful = u.backward_useful(Some(sym("Num")), &comps);
        assert!(useful.contains(&sym("Num")));
        assert!(useful.contains(&sym("String")));
    }

    // ── Builder tests ──────────────────────────────────────────────────

    use crate::eval_v2;
    use crate::types_v2::{init_special_forms, Node, SpecialForm};
    use std::rc::Rc;

    fn rc<T>(v: Vec<T>) -> Rc<[T]> {
        v.into()
    }

    #[test]
    fn primitive_catalog_contains_expected_builtins() {
        let comps = primitive_components();
        let names: Vec<&str> = comps.iter().map(|c| c.name.as_str()).collect();
        // Spot-check across each domain
        assert!(names.contains(&"add"));
        assert!(names.contains(&"string-upper"));
        assert!(names.contains(&"string-length"));
        assert!(names.contains(&"head"));
        assert!(names.contains(&"<"));
        assert!(names.contains(&"not"));
        assert!(names.contains(&"dispatch"));
        // Integer literals
        assert!(names.contains(&"0"));
        assert!(names.contains(&"-1"));
        // String literals
        assert!(names.contains(&"a"));
        assert!(names.contains(&"("));
        // No grid components
        assert!(!names.iter().any(|n| n.starts_with("grid-")));
        // No `x` (input variable is task-specific)
        assert!(!names.contains(&"x"));
    }

    #[test]
    fn primitive_catalog_uses_sym_types() {
        let comps = primitive_components();
        // Arithmetic is typed Int -> Int -> Int (curriculum bias —
        // see the comment in primitive_components). divide and floor
        // remain Num as the explicit float entry points.
        let add = comps.iter().find(|c| c.name == "add").unwrap();
        assert_eq!(add.arity, 2);
        assert_eq!(add.param_types, vec![sym("Int"), sym("Int")]);
        assert_eq!(add.ret_type, sym("Int"));

        let divide = comps.iter().find(|c| c.name == "divide").unwrap();
        assert_eq!(divide.param_types, vec![sym("Num"), sym("Num")]);
        assert_eq!(divide.ret_type, sym("Num"));

        // string-length returns Int (not Num) — eval_v2 produces Int
        let sl = comps.iter().find(|c| c.name == "string-length").unwrap();
        assert_eq!(sl.ret_type, sym("Int"));
        assert_eq!(sl.param_types, vec![sym("String")]);
    }

    #[test]
    fn input_var_component_carries_task_type() {
        let c = input_var_component(sym("String"));
        assert_eq!(c.name, "x");
        assert_eq!(c.arity, 0);
        assert_eq!(c.ret_type, sym("String"));
        assert!(matches!(c.dispatch, Dispatch::Literal(LiteralKind::InputVar)));
    }

    #[test]
    fn library_probe_discovers_unary_string_function() {
        // Define `(define wrap (lambda (s) (string-upper s)))` in a fresh
        // env, then probe it. Expected: param=String, ret=String.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes = rc(vec![
            // (string-upper s)
            Node::Symbol(intern("string-upper")), // 0
            Node::Symbol(intern("s")),            // 1
            Node::App(vec![0, 1]),                // 2
            // (lambda (s) (string-upper s))
            Node::Lambda(vec![intern("s")], 2),   // 3
            // wrap target
            Node::Symbol(intern("wrap")),         // 4
            // (define wrap ...)
            Node::SpecialApp(SpecialForm::Define, vec![4, 3]), // 5
        ]);
        eval_v2::eval(&nodes, 5, &env).unwrap();

        let skip = default_skip_set();
        let lib = library_components_from_env(&env, &skip);
        let wrap = lib.iter().find(|c| c.name == "wrap").expect("wrap component missing");
        assert_eq!(wrap.arity, 1);
        assert_eq!(wrap.ret_type, sym("String"));
        assert_eq!(wrap.param_types, vec![sym("String")]);
        match wrap.dispatch {
            Dispatch::Named(s) => assert_eq!(s, intern("wrap")),
            _ => panic!("expected Named dispatch"),
        }
    }

    #[test]
    fn library_probe_discovers_binary_numeric_function() {
        // (define plus (lambda (a b) (add a b))) — binary, returns numeric.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("add")),      // 0
            Node::Symbol(intern("a")),        // 1
            Node::Symbol(intern("b")),        // 2
            Node::App(vec![0, 1, 2]),         // 3
            Node::Lambda(vec![intern("a"), intern("b")], 3), // 4
            Node::Symbol(intern("plus")),     // 5
            Node::SpecialApp(SpecialForm::Define, vec![5, 4]), // 6
        ]);
        eval_v2::eval(&nodes, 6, &env).unwrap();

        let skip = default_skip_set();
        let lib = library_components_from_env(&env, &skip);
        let plus = lib.iter().find(|c| c.name == "plus").expect("plus component missing");
        assert_eq!(plus.arity, 2);
        // First successful probe is Int (Int sample tried first), and
        // (add Int Int) returns Int per eval_v2's coercion rule.
        assert_eq!(plus.ret_type, sym("Int"));
        assert_eq!(plus.param_types, vec![sym("Int"), sym("Int")]);
    }

    #[test]
    fn library_probe_skips_functions_that_always_error() {
        // (define broken (lambda (x) (head x))) — head expects a list,
        // so all single-value probes will error out. Expect: not in lib.
        // Actually, the List sample IS one of the probes, so this would
        // succeed. Let's force a real error:
        // (define broken (lambda (x) (string-length 5)))
        // string-length(Int) errors regardless of x.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("string-length")), // 0
            Node::Int(5),                          // 1
            Node::App(vec![0, 1]),                 // 2
            Node::Lambda(vec![intern("x")], 2),    // 3
            Node::Symbol(intern("broken")),        // 4
            Node::SpecialApp(SpecialForm::Define, vec![4, 3]), // 5
        ]);
        eval_v2::eval(&nodes, 5, &env).unwrap();

        let skip = default_skip_set();
        let lib = library_components_from_env(&env, &skip);
        // broken should not appear because every probe failed.
        assert!(lib.iter().all(|c| c.name != "broken"));
    }

    #[test]
    fn fused_components_emit_map_and_reduce_for_library() {
        // Define a unary and a binary function, then check that
        // fused_components emits map_<unary> and reduce_<binary>.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes = rc(vec![
            // (define inc (lambda (n) (add n 1)))
            Node::Symbol(intern("add")),      // 0
            Node::Symbol(intern("n")),        // 1
            Node::Int(1),                     // 2
            Node::App(vec![0, 1, 2]),         // 3
            Node::Lambda(vec![intern("n")], 3), // 4
            Node::Symbol(intern("inc")),      // 5
            Node::SpecialApp(SpecialForm::Define, vec![5, 4]), // 6

            // (define plus (lambda (a b) (add a b)))
            Node::Symbol(intern("add")),      // 7
            Node::Symbol(intern("a")),        // 8
            Node::Symbol(intern("b")),        // 9
            Node::App(vec![7, 8, 9]),         // 10
            Node::Lambda(vec![intern("a"), intern("b")], 10), // 11
            Node::Symbol(intern("plus")),     // 12
            Node::SpecialApp(SpecialForm::Define, vec![12, 11]), // 13

            Node::SpecialApp(SpecialForm::Do, vec![6, 13]), // 14
        ]);
        eval_v2::eval(&nodes, 14, &env).unwrap();

        let skip = default_skip_set();
        let lib = library_components_from_env(&env, &skip);
        let fused = fused_components(&lib);
        let names: Vec<&str> = fused.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"map_inc"));
        assert!(names.contains(&"reduce_plus"));
        // Built-in fused-reduces should also be present.
        assert!(names.contains(&"reduce_add"));
        assert!(names.contains(&"reduce_concat"));

        let map_inc = fused.iter().find(|c| c.name == "map_inc").unwrap();
        assert_eq!(map_inc.arity, 1);
        assert_eq!(map_inc.param_types, vec![sym("List")]);
        assert_eq!(map_inc.ret_type, sym("List"));
        assert!(matches!(map_inc.dispatch, Dispatch::FusedMap(_)));
    }

    #[test]
    fn default_synth_components_combines_all_sources() {
        // End-to-end: load a tiny library, then check that the combined
        // catalog contains primitives, the library function, and fused
        // forms.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("string-upper")), // 0
            Node::Symbol(intern("s")),            // 1
            Node::App(vec![0, 1]),                // 2
            Node::Lambda(vec![intern("s")], 2),   // 3
            Node::Symbol(intern("shout")),        // 4
            Node::SpecialApp(SpecialForm::Define, vec![4, 3]), // 5
        ]);
        eval_v2::eval(&nodes, 5, &env).unwrap();

        let skip = default_skip_set();
        let all = default_synth_components(&env, &skip);
        let names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
        // Primitive
        assert!(names.contains(&"add"));
        // Library
        assert!(names.contains(&"shout"));
        // Fused (unary library function gets a map_ form)
        assert!(names.contains(&"map_shout"));
        // Built-in fused reduce
        assert!(names.contains(&"reduce_add"));
    }

    // ── Synthesis core-loop tests ─────────────────────────────────────

    /// Helper: render a node tree as the same source the value_to_string
    /// function would. Used for inspecting synthesized candidates in
    /// failure messages.
    fn render(nodes: &[Node], root: usize) -> String {
        match &nodes[root] {
            Node::Int(n) => n.to_string(),
            Node::Num(n) => n.to_string(),
            Node::Str(s) => format!("\"{}\"", s),
            Node::Bool(b) => b.to_string(),
            Node::Symbol(s) => crate::intern::resolve(*s),
            Node::App(c) => {
                let parts: Vec<String> = c.iter().map(|&i| render(nodes, i)).collect();
                format!("({})", parts.join(" "))
            }
            Node::SpecialApp(_, c) => {
                let parts: Vec<String> = c.iter().map(|&i| render(nodes, i)).collect();
                format!("({})", parts.join(" "))
            }
            Node::Lambda(p, b) => {
                let params: Vec<String> = p.iter().map(|s| crate::intern::resolve(*s)).collect();
                format!("(lambda ({}) {})", params.join(" "), render(nodes, *b))
            }
            _ => format!("<{:?}>", nodes[root]),
        }
    }

    fn run_synth(
        components: &[SynthComponent],
        inputs: Vec<Value>,
        expected: Vec<Value>,
        max_depth: usize,
        max_candidates: usize,
    ) -> SynthResult {
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        synthesize(
            components,
            &inputs,
            &expected,
            &env,
            &universe,
            max_depth,
            max_candidates,
        )
    }

    #[test]
    fn synth_identity_at_depth_zero() {
        // Identity on integers: x → x. Should be solved at depth 0
        // (the input variable atom matches directly).
        let comps = primitive_components();
        let r = run_synth(
            &comps,
            vec![Value::Int(7), Value::Int(42)],
            vec![Value::Int(7), Value::Int(42)],
            0,
            500,
        );
        assert!(r.found, "expected identity to be found at depth 0");
        let (nodes, root) = (r.nodes.unwrap(), r.root.unwrap());
        // The synthesized program should be (lambda (x) x).
        assert_eq!(render(&nodes, root), "(lambda (x) x)");
    }

    #[test]
    fn synth_unary_primitive_at_depth_one() {
        // string-upper: "abc" → "ABC". Should be solved at depth 1.
        let comps = primitive_components();
        let r = run_synth(
            &comps,
            vec![
                Value::str("hello"),
                Value::str("world"),
            ],
            vec![
                Value::str("HELLO"),
                Value::str("WORLD"),
            ],
            2,
            2000,
        );
        assert!(r.found, "expected string-upper to be found");
        let (nodes, root) = (r.nodes.unwrap(), r.root.unwrap());
        assert_eq!(render(&nodes, root), "(lambda (x) (string-upper x))");
    }

    #[test]
    fn synth_binary_primitive_with_constant() {
        // (add x 1): 5 → 6, 10 → 11. Should be solved at depth 1.
        let comps = primitive_components();
        let r = run_synth(
            &comps,
            vec![Value::Int(5), Value::Int(10), Value::Int(0)],
            vec![Value::Int(6), Value::Int(11), Value::Int(1)],
            2,
            5000,
        );
        assert!(r.found, "expected (add x 1) to be found");
        let (nodes, root) = (r.nodes.unwrap(), r.root.unwrap());
        // Many programs satisfy this; just check it actually works by
        // re-evaluating the result against a held-out input.
        let env = eval_v2::make_default_env();
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        let r2 = eval_v2::apply(&f, &[Value::Int(99)], &env).unwrap();
        assert!(matches!(r2, Value::Int(100)), "synthesized fn failed on held-out input");
    }

    #[test]
    fn synth_two_level_composition() {
        // (string-length (string-upper x)) → length of upper.
        // string-upper is idempotent on length, so this is just
        // string-length composed with anything string→string. Easiest
        // for the synthesizer to find: string-length(x).
        let comps = primitive_components();
        let r = run_synth(
            &comps,
            vec![Value::str("hi"), Value::str("hello")],
            vec![Value::Int(2), Value::Int(5)],
            3,
            5000,
        );
        assert!(r.found, "expected string-length to be found");
        // Verify on a held-out input.
        let env = eval_v2::make_default_env();
        let (nodes, root) = (r.nodes.unwrap(), r.root.unwrap());
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        let r2 = eval_v2::apply(&f, &[Value::str("ab")], &env).unwrap();
        assert!(matches!(r2, Value::Int(2)));
    }

    #[test]
    fn synth_uses_library_function_from_env() {
        // Define a custom library function `inc` and verify the
        // synthesizer can use it. Task: x → x + 1 (an inc).
        // We deliberately exclude `add` from the component list so the
        // ONLY way to solve it is via `inc`.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes = rc(vec![
            Node::Symbol(intern("add")),         // 0
            Node::Symbol(intern("n")),           // 1
            Node::Int(1),                        // 2
            Node::App(vec![0, 1, 2]),            // 3
            Node::Lambda(vec![intern("n")], 3),  // 4
            Node::Symbol(intern("inc")),         // 5
            Node::SpecialApp(SpecialForm::Define, vec![5, 4]), // 6
        ]);
        eval_v2::eval(&nodes, 6, &env).unwrap();

        // Build component catalog WITHOUT add — only inc + the input var
        // and basic constants.
        let skip = default_skip_set();
        let lib = library_components_from_env(&env, &skip);
        // Just inc + integer constants 0..7. No primitive add.
        let mut comps: Vec<SynthComponent> = lib;
        // A few literal integer atoms so the search has something to chew.
        for n in [0i64, 1] {
            comps.push(SynthComponent::literal(
                n.to_string(),
                LiteralKind::Int(n),
                intern("Int"),
                0.0,
            ));
        }

        let universe = TypeUniverse::primitives();
        let r = synthesize(
            &comps,
            &[Value::Int(5), Value::Int(10)],
            &[Value::Int(6), Value::Int(11)],
            &env,
            &universe,
            2,
            2000,
        );
        assert!(r.found, "expected (inc x) via library function");
        let (nodes, root) = (r.nodes.unwrap(), r.root.unwrap());
        assert!(
            render(&nodes, root).contains("inc"),
            "expected synthesized program to use the inc library function, got: {}",
            render(&nodes, root)
        );
    }

    #[test]
    fn synth_returns_not_found_when_impossible() {
        // Impossible task: map an Int to a string that has no relation
        // to it. Synthesis should exhaust the budget without finding
        // a solution.
        let comps = primitive_components();
        let r = run_synth(
            &comps,
            vec![Value::Int(1), Value::Int(2)],
            vec![Value::str("foo"), Value::str("bar")],
            2,
            500,
        );
        assert!(!r.found);
    }

    #[test]
    fn synth_respects_candidate_budget() {
        // Set a tiny budget and verify the explored count never exceeds it.
        let comps = primitive_components();
        let r = run_synth(
            &comps,
            vec![Value::Int(1)],
            vec![Value::str("nothing-could-match")],
            5,
            10,
        );
        assert!(!r.found);
        assert!(
            r.candidates_explored <= 10,
            "explored {} > budget 10",
            r.candidates_explored
        );
    }

    #[test]
    fn synth_dedup_skips_observationally_equivalent_candidates() {
        // We can't directly observe dedup behavior, but we can run a
        // task where many candidates produce the same outputs and check
        // that the search still terminates within a reasonable budget.
        // (negate (negate x)) and (abs x) and just `x` all behave
        // identically on positive inputs.
        let comps = primitive_components();
        let r = run_synth(
            &comps,
            vec![Value::Int(3), Value::Int(7)],
            vec![Value::Int(3), Value::Int(7)],
            3,
            10000,
        );
        assert!(r.found);
    }

    #[test]
    fn synth_handles_empty_examples() {
        let comps = primitive_components();
        let r = run_synth(&comps, vec![], vec![], 2, 100);
        assert!(!r.found);
        assert_eq!(r.candidates_explored, 0);
    }

    // ── Probe-and-filter tests ────────────────────────────────────────

    /// Helper: load `(define <name> <lambda-source>)` into a fresh env
    /// and return that env. The body source is constructed via raw
    /// Node nodes for control over what we're testing.
    fn env_with_definitions(defs: Vec<(Sym, Vec<Sym>, Vec<Node>, usize)>) -> Env {
        // For each (name, params, body_nodes, body_root):
        //   build (define name (lambda (params...) body)) and eval it.
        init_special_forms();
        let env = eval_v2::make_default_env();
        for (name, params, body_nodes, body_root) in defs {
            // Append: lambda node, name node, define node.
            let mut nodes = body_nodes;
            let lambda_idx = nodes.len();
            nodes.push(Node::Lambda(params, body_root));
            let name_idx = nodes.len();
            nodes.push(Node::Symbol(name));
            let def_idx = nodes.len();
            nodes.push(Node::SpecialApp(SpecialForm::Define, vec![name_idx, lambda_idx]));
            let nodes_rc: Rc<[Node]> = nodes.into();
            eval_v2::eval(&nodes_rc, def_idx, &env).unwrap();
        }
        env
    }

    #[test]
    fn probe_drops_library_function_that_errors_on_input() {
        // Define `(define brittle (lambda (s) (string-length 5)))` —
        // it ignores its argument and calls string-length on Int 5,
        // which errors.
        let env = env_with_definitions(vec![(
            intern("brittle"),
            vec![intern("s")],
            vec![
                Node::Symbol(intern("string-length")), // 0
                Node::Int(5),                          // 1
                Node::App(vec![0, 1]),                 // 2
            ],
            2,
        )]);
        // Build a probe-able component pretending brittle is a String→String
        // library function. The skip set excludes builtins so brittle ends
        // up in the library list.
        let comps = vec![SynthComponent::named(
            "brittle",
            intern("brittle"),
            vec![sym("String")],
            sym("String"),
            0.0,
        )];
        let universe = TypeUniverse::primitives();
        let kept = probe_filter_components(
            &comps,
            &env,
            &Value::str("hello"),
            sym("String"),
            &universe,
        );
        assert!(
            kept.iter().all(|c| c.name != "brittle"),
            "expected brittle to be dropped — but kept: {:?}",
            kept.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn probe_keeps_library_function_that_succeeds_on_input() {
        // (define shout (lambda (s) (string-upper s))) — works on any string.
        let env = env_with_definitions(vec![(
            intern("shout"),
            vec![intern("s")],
            vec![
                Node::Symbol(intern("string-upper")), // 0
                Node::Symbol(intern("s")),            // 1
                Node::App(vec![0, 1]),                // 2
            ],
            2,
        )]);
        let comps = vec![SynthComponent::named(
            "shout",
            intern("shout"),
            vec![sym("String")],
            sym("String"),
            0.0,
        )];
        let universe = TypeUniverse::primitives();
        let kept = probe_filter_components(
            &comps,
            &env,
            &Value::str("hello"),
            sym("String"),
            &universe,
        );
        assert!(
            kept.iter().any(|c| c.name == "shout"),
            "expected shout to be kept"
        );
    }

    #[test]
    fn probe_does_not_touch_builtins() {
        // string-length is a Value::Builtin in env. Even if we passed a
        // bad test input, the probe filter should not touch it (the
        // skip-on-non-Function check excludes builtins).
        let env = eval_v2::make_default_env();
        let comps = primitive_components();
        let universe = TypeUniverse::primitives();
        // Pass an Int as the test input — string-length would error if
        // called with it. The filter should still keep string-length
        // because it's a builtin, not a Function.
        let kept = probe_filter_components(
            &comps,
            &env,
            &Value::Int(42),
            sym("Int"),
            &universe,
        );
        assert!(
            kept.iter().any(|c| c.name == "string-length"),
            "expected string-length (builtin) to be kept"
        );
    }

    #[test]
    fn probe_skips_functions_with_mismatched_param_type() {
        // Define a String→String function but probe with an Int input.
        // The function shouldn't be probed (param type doesn't match
        // input type) and should be left alone — it might still be
        // used as an intermediate transform mid-program.
        let env = env_with_definitions(vec![(
            intern("inner_string_helper"),
            vec![intern("s")],
            vec![
                Node::Symbol(intern("string-upper")), // 0
                Node::Symbol(intern("s")),            // 1
                Node::App(vec![0, 1]),                // 2
            ],
            2,
        )]);
        let comps = vec![SynthComponent::named(
            "inner_string_helper",
            intern("inner_string_helper"),
            vec![sym("String")],
            sym("String"),
            0.0,
        )];
        let universe = TypeUniverse::primitives();
        // Input type is Int — doesn't match the helper's String slot.
        let kept = probe_filter_components(
            &comps,
            &env,
            &Value::Int(5),
            sym("Int"),
            &universe,
        );
        assert!(
            kept.iter().any(|c| c.name == "inner_string_helper"),
            "expected helper to be kept (mismatched input type — not probed)"
        );
    }

    #[test]
    fn probe_passes_through_multi_arg_functions() {
        // Even if a binary library function would error on probing,
        // probe_filter_components leaves multi-arg functions alone
        // (we don't know what to put in the other slots).
        let comps = vec![SynthComponent::named(
            "fake_binary",
            intern("fake_binary"),
            vec![sym("String"), sym("String")],
            sym("String"),
            0.0,
        )];
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let kept = probe_filter_components(
            &comps,
            &env,
            &Value::str("hi"),
            sym("String"),
            &universe,
        );
        assert!(kept.iter().any(|c| c.name == "fake_binary"));
    }

    #[test]
    fn probe_integrated_into_synthesize_drops_broken_library() {
        // End-to-end check: define a brittle library function alongside
        // a working one, run synthesize on a task only the working one
        // can solve, and verify the broken one doesn't blow up the
        // search even though it's reachable by type.
        //
        // Task: x → uppercase x. Components include both `brittle`
        // (errors on input) and `shout` (correct).
        let env = env_with_definitions(vec![
            // brittle: errors on any input by calling string-length(5)
            (
                intern("brittle"),
                vec![intern("s")],
                vec![
                    Node::Symbol(intern("string-length")), // 0
                    Node::Int(5),                          // 1
                    Node::App(vec![0, 1]),                 // 2
                ],
                2,
            ),
            // shout: actually upper-cases its argument
            (
                intern("shout"),
                vec![intern("s")],
                vec![
                    Node::Symbol(intern("string-upper")), // 0
                    Node::Symbol(intern("s")),            // 1
                    Node::App(vec![0, 1]),                // 2
                ],
                2,
            ),
        ]);

        // Build a minimal component list: just brittle, shout, and an
        // input variable seed. NO primitives — so the synthesizer must
        // pick a library function.
        let comps = vec![
            SynthComponent::named(
                "brittle",
                intern("brittle"),
                vec![sym("String")],
                sym("String"),
                0.0,
            ),
            SynthComponent::named(
                "shout",
                intern("shout"),
                vec![sym("String")],
                sym("String"),
                0.0,
            ),
        ];

        let universe = TypeUniverse::primitives();
        let r = synthesize(
            &comps,
            &[Value::str("hi"), Value::str("hello")],
            &[Value::str("HI"), Value::str("HELLO")],
            &env,
            &universe,
            2,
            500,
        );
        assert!(r.found, "expected (shout x) to be synthesized");
        let (nodes, root) = (r.nodes.unwrap(), r.root.unwrap());
        assert!(
            render(&nodes, root).contains("shout"),
            "expected synthesized program to use shout, got: {}",
            render(&nodes, root)
        );
    }

    // ── Memorization & dispatcher tests ───────────────────────────────

    #[test]
    fn memorize_emits_lookup_table_for_string_inputs() {
        let inputs = vec![
            Value::str("alice"),
            Value::str("bob"),
            Value::str("carol"),
        ];
        let expected = vec![Value::Int(1), Value::Int(2), Value::Int(3)];

        let (nodes, root) = memorize_from_examples(&inputs, &expected)
            .expect("memorize should succeed on string→int mapping");

        // Evaluate the resulting lambda and check it works on each input.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (inp, exp) in inputs.iter().zip(expected.iter()) {
            let got = eval_v2::apply(&f, &[inp.clone()], &env).unwrap();
            assert!(
                eval_v2::values_equal(&got, exp),
                "memorize lookup mismatch on {:?}: got {:?}, expected {:?}",
                inp,
                got,
                exp
            );
        }
    }

    #[test]
    fn memorize_returns_default_on_unknown_input() {
        let inputs = vec![Value::str("known")];
        let expected = vec![Value::Int(42)];
        let (nodes, root) = memorize_from_examples(&inputs, &expected).unwrap();

        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        // Default for Int outputs is Int(0).
        let got = eval_v2::apply(&f, &[Value::str("missing")], &env).unwrap();
        assert!(matches!(got, Value::Int(0)));
    }

    #[test]
    fn memorize_returns_none_for_non_string_inputs() {
        let r = memorize_from_examples(&[Value::Int(1)], &[Value::Int(2)]);
        assert!(r.is_none());
    }

    #[test]
    fn memorize_returns_none_for_conflicting_outputs() {
        let inputs = vec![Value::str("k"), Value::str("k")];
        let expected = vec![Value::Int(1), Value::Int(2)];
        let r = memorize_from_examples(&inputs, &expected);
        assert!(r.is_none(), "conflicting outputs should yield None");
    }

    #[test]
    fn memorize_works_for_string_to_string_mapping() {
        let inputs = vec![Value::str("a"), Value::str("b")];
        let expected = vec![Value::str("APPLE"), Value::str("BANANA")];
        let (nodes, root) = memorize_from_examples(&inputs, &expected).unwrap();

        init_special_forms();
        let env = eval_v2::make_default_env();
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        let got_a = eval_v2::apply(&f, &[Value::str("a")], &env).unwrap();
        assert_eq!(got_a.as_str().unwrap(), "APPLE");
        let got_b = eval_v2::apply(&f, &[Value::str("b")], &env).unwrap();
        assert_eq!(got_b.as_str().unwrap(), "BANANA");
        // Default for Str is "".
        let got_unknown = eval_v2::apply(&f, &[Value::str("z")], &env).unwrap();
        assert_eq!(got_unknown.as_str().unwrap(), "");
    }

    #[test]
    fn dispatcher_uses_flat_when_solvable_by_enumeration() {
        // Identity task — Flat solves it at depth 0.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let r = synthesize_with_strategies(
            &comps,
            &[Value::Int(7), Value::Int(42)],
            &[Value::Int(7), Value::Int(42)],
            &env,
            &universe,
            1,
            500,
        );
        assert!(r.found);
        assert_eq!(r.strategy, Some(Strategy::Flat));
    }

    #[test]
    fn dispatcher_falls_through_to_memo_when_flat_fails() {
        // String→Int mapping with no algorithmic relationship — Flat
        // can't solve it but Memo can.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let r = synthesize_with_strategies(
            &comps,
            &[
                Value::str("foo"),
                Value::str("xyzzy"),
                Value::str("bar"),
            ],
            &[Value::Int(13), Value::Int(99), Value::Int(7)],
            &env,
            &universe,
            2,
            // Tiny budget — Flat will give up fast.
            200,
        );
        assert!(r.found);
        assert_eq!(r.strategy, Some(Strategy::Memo));

        // Verify the memo solution actually works.
        let (nodes, root) = (r.nodes.unwrap(), r.root.unwrap());
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        let got = eval_v2::apply(&f, &[Value::str("xyzzy")], &env).unwrap();
        assert!(matches!(got, Value::Int(99)));
    }

    #[test]
    fn dispatcher_returns_not_found_when_no_strategy_applies() {
        // Int→string with no consistent mapping the synthesizer can find
        // and inputs aren't strings, so Memo can't help either.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let r = synthesize_with_strategies(
            &comps,
            &[Value::Int(1), Value::Int(2)],
            &[Value::str("foo"), Value::str("bar")],
            &env,
            &universe,
            2,
            200,
        );
        assert!(!r.found);
        assert_eq!(r.strategy, None);
    }

    #[test]
    fn val_hash_distinguishes_different_int_lists() {
        let a = Value::list(vec![Value::Int(1), Value::Int(2)]);
        let b = Value::list(vec![Value::Int(1), Value::Int(3)]);
        assert_ne!(val_hash(&a), val_hash(&b));
    }

    #[test]
    fn val_hash_distinguishes_int_from_num() {
        // Same numeric value, different runtime representation.
        assert_ne!(val_hash(&Value::Int(5)), val_hash(&Value::Num(5.0)));
    }

    #[test]
    fn infer_uniform_type_returns_some_when_consistent() {
        let vs = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        assert_eq!(infer_uniform_type_sym(&vs), Some(intern("Int")));
    }

    #[test]
    fn infer_uniform_type_returns_none_when_mixed() {
        let vs = vec![Value::Int(1), Value::Num(2.0)];
        assert_eq!(infer_uniform_type_sym(&vs), None);
    }

    #[test]
    fn default_skip_set_includes_builtins() {
        let skip = default_skip_set();
        assert!(skip.contains(&intern("add")));
        assert!(skip.contains(&intern("string-upper")));
        assert!(skip.contains(&intern("nil")));
    }

    #[test]
    fn dispatch_variants_carry_correct_data() {
        // Smoke test: each Dispatch variant constructs cleanly and the
        // arity/param_types invariant holds for SynthComponent::named.
        let lit = SynthComponent::literal("0", LiteralKind::Int(0), sym("Int"), 0.0);
        assert_eq!(lit.arity, 0);
        assert!(matches!(lit.dispatch, Dispatch::Literal(LiteralKind::Int(0))));

        let nm = SynthComponent::named("add", sym("add"), vec![sym("Int"), sym("Int")], sym("Int"), 0.0);
        assert_eq!(nm.arity, 2);
        assert_eq!(nm.param_types.len(), 2);
        match nm.dispatch {
            Dispatch::Named(s) => assert_eq!(s, sym("add")),
            _ => panic!("expected Named dispatch"),
        }
    }
}
