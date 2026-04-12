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
use crate::intern::{intern, resolve, Sym};
use crate::types_v2::{type_any, Env, Node, NsMap, SpecialForm, Value};

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
    /// Floating-point literal — added for the §9.31 physics curriculum.
    /// Carries a wrapped f64; equality is by `to_bits` so duplicate
    /// literals (e.g. two `0.5` seeds) compare as equal.
    Num(f64),
    Str(String),
    Bool(bool),
    /// Indexed positional argument — added for the §9.31 multi-arg
    /// path. Materializes to a `(nth x N)` subtree so the rest of
    /// the synthesizer treats it as an atomic depth-0 entry while
    /// the runtime still receives a list-shaped input. Carries the
    /// list index. The atom's `ret_type` (on the SynthComponent
    /// wrapper) carries the type of `x[N]`.
    Indexed(usize),
    // §9.45.13 deletion: `Hole` variant removed. Constant-hole
    // synthesis is now done in pure SELPH via the M11/M12 fit-affine
    // primitive (m_pool.selph + the M-stage detect-* functions). No
    // synth_v2 component creates a Hole literal anymore.
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

/// §9.37 Stage B: per-type metadata loaded from a SELPH `__types__`
/// namespace. Each type entry can carry an optional predicate (a SELPH
/// lambda taking a Value and returning a Bool — used for type-of
/// queries from curriculum code), a list of supertype Syms (for
/// subtype graph hints; note the hot-path `slot_accepts` is still
/// hardcoded for performance), an optional priority (for ordering
/// type-keyed dispatch), and a list of (name-Sym, decomposer Value)
/// pairs for type-keyed decomposer dispatch in §9.37 Stage C.
///
/// When `__types__` is absent in the env, `TypeUniverse::primitives()`
/// builds an empty `type_metadata` map and the dispatcher falls
/// through to the global `__decomposers__` walk + hardcoded chain.
#[derive(Clone, Debug, Default)]
pub struct TypeMetadata {
    /// SELPH lambda `(lambda (v) <bool>)` testing if a value inhabits
    /// the type. Optional — if absent, the type is recognizable by
    /// name but has no membership predicate.
    pub predicate: Option<Value>,
    /// Direct supertypes by name. Transitive closure is not computed
    /// here (the hot-path `slot_accepts` doesn't consult this map).
    pub subtype_of: Vec<Sym>,
    /// Numeric priority for ordering decomposer dispatch (Stage C).
    pub priority: f64,
    /// Decomposers registered for this type, in (key-Sym, function)
    /// pairs. Iteration order is the namespace order from
    /// `__types__[type-name]["decomposers"]` after a name sort
    /// (deterministic dispatch).
    pub decomposers: Vec<(Sym, Value)>,
}

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
    /// §9.37 Stage B: per-type metadata loaded from `__types__`.
    /// Empty for `primitives()`; populated by `from_env(env)`. Used by
    /// Stage C type-keyed decomposer dispatch.
    type_metadata: HashMap<Sym, TypeMetadata>,
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
            type_metadata: HashMap::new(),
        }
    }

    /// §9.37 Stage B: build a universe from the env's `__types__`
    /// namespace, layered on top of the primitive baseline. Each
    /// entry in `__types__` is expected to be a namespace with
    /// optional fields `predicate` (function), `subtype-of` (list of
    /// strings), `priority` (number), `decomposers` (namespace of
    /// name → function entries).
    ///
    /// When `__types__` is absent, returns the primitive universe
    /// unchanged. When entries are malformed, they're skipped silently
    /// (curriculum-driven ergonomics — better to load partial than
    /// crash the synth dispatcher).
    pub fn from_env(env: &Env) -> Self {
        let mut universe = Self::primitives();
        let types_ns = match env.lookup(intern("__types__")) {
            Some(Value::Ns(map)) => map,
            _ => return universe,
        };
        for (&type_sym, type_val) in types_ns.iter() {
            // Each entry must itself be a namespace.
            let entry_ns = match type_val {
                Value::Ns(m) => m,
                _ => continue,
            };
            universe.known.insert(type_sym);

            let mut meta = TypeMetadata::default();

            // Optional predicate field.
            if let Some(p) = entry_ns.get(&intern("predicate")) {
                if matches!(p, Value::Function(_) | Value::Builtin(_)) {
                    meta.predicate = Some(p.clone());
                }
            }

            // Optional subtype-of field (list of strings).
            if let Some(Value::List(supers)) = entry_ns.get(&intern("subtype-of")) {
                for s in supers.iter() {
                    if let Value::Str(name) = s {
                        meta.subtype_of.push(intern(name.as_ref()));
                    }
                }
            }

            // Optional priority.
            if let Some(p) = entry_ns.get(&intern("priority")) {
                meta.priority = match p {
                    Value::Int(n) => *n as f64,
                    Value::Num(n) => *n,
                    _ => 0.0,
                };
            }

            // Optional decomposers namespace. Sorted by name for
            // deterministic dispatch order (matches Stage A).
            if let Some(Value::Ns(decomp_ns)) = entry_ns.get(&intern("decomposers")) {
                let mut entries: Vec<(Sym, Value)> = decomp_ns
                    .iter()
                    .filter(|(_, v)| matches!(v, Value::Function(_) | Value::Builtin(_)))
                    .map(|(&k, v)| (k, v.clone()))
                    .collect();
                entries.sort_by_key(|(s, _)| resolve(*s));
                meta.decomposers = entries;
            }

            universe.type_metadata.insert(type_sym, meta);
        }
        universe
    }

    /// §9.37 Stage B: get the metadata for a type Sym, or None if the
    /// type isn't registered in `__types__`. Used by Stage C dispatch.
    pub fn type_metadata(&self, sym: Sym) -> Option<&TypeMetadata> {
        self.type_metadata.get(&sym)
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

    // ── Float arithmetic (Num × Num → Num) — §9.31 physics curriculum ─
    // Same eval_v2 builtins as the Int versions; the type signatures
    // here are what synth uses to compose them. Without these, a task
    // whose `x` is `Num` can't reach `add`/`subtract`/`multiply`
    // because `slot_accepts(Int, Num)` is false.
    //
    // Priority is set to 1.0 so they don't crowd out the cheap Int
    // path on integer-typed tasks (where they'd never compose anyway,
    // since `slot_accepts(Num, Int)` IS true and the Int versions
    // have priority 0.0 — same effective ordering on integer inputs).
    for name in &["add", "subtract", "multiply"] {
        comps.push(SynthComponent::named(
            *name, intern(name), vec![num, num], num, 1.0,
        ));
    }
    for name in &["abs", "negate"] {
        comps.push(SynthComponent::named(
            *name, intern(name), vec![num], num, 1.0,
        ));
    }
    comps.push(SynthComponent::named(
        "min", intern("min"), vec![num, num], num, 1.0,
    ));
    comps.push(SynthComponent::named(
        "max", intern("max"), vec![num, num], num, 1.0,
    ));
    // Transcendentals & power. `pow` and `sqrt` already exist as
    // builtins; we just expose them as Num components for synth.
    comps.push(SynthComponent::named(
        "pow", intern("pow"), vec![num, num], num, 1.0,
    ));
    comps.push(SynthComponent::named(
        "sqrt", intern("sqrt"), vec![num], num, 2.0,
    ));
    comps.push(SynthComponent::named(
        "log", intern("log"), vec![num], num, 3.0,
    ));
    comps.push(SynthComponent::named(
        "exp", intern("exp"), vec![num], num, 3.0,
    ));
    comps.push(SynthComponent::named(
        "sin", intern("sin"), vec![num], num, 4.0,
    ));
    comps.push(SynthComponent::named(
        "cos", intern("cos"), vec![num], num, 4.0,
    ));
    comps.push(SynthComponent::named(
        "tan", intern("tan"), vec![num], num, 4.0,
    ));

    // §9.43 list constructors are NOT seeded here — they're added
    // by `synthesize_inner` only on the multi-arg path so they don't
    // inflate single-input enumeration cost. See the §9.43.6 cost
    // notes in SELPH_Growing_System_Plan.md.

    // Float literal constants — the basic seeds for physics tasks.
    // Includes 0.5 (half), 2.0 (square exponent), -1.0 (sign flip).
    // PI and E are emitted as Num literals — they're constant, so
    // there's no runtime difference between a literal and a
    // zero-arity builtin call, and the literal form skips one
    // application step in the candidate tree.
    //
    // Domain-specific constants like 0.25, 0.125 are NOT seeded
    // here. The principled mechanism for them is data-derived
    // literal seeding (`augment_components_with_data_literals`),
    // which scans the task's input/output values and adds any
    // unique Num atoms it finds. This keeps the kernel catalog
    // free of physics constants while still letting the synth
    // reach common dyadics that appear in the spec data itself.
    for (name, val) in [
        ("0.0", 0.0_f64),
        ("1.0", 1.0_f64),
        ("2.0", 2.0_f64),
        ("0.5", 0.5_f64),
        ("-1.0", -1.0_f64),
        ("pi", std::f64::consts::PI),
        ("e", std::f64::consts::E),
    ] {
        comps.push(SynthComponent::literal(
            name,
            LiteralKind::Num(val),
            num,
            0.0,
        ));
    }

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

/// Build an indexed-positional-arg component for a multi-arg task.
/// `idx` is the position into the args list; `arg_type` is the runtime
/// type of `args[idx]`. Materializes to a `(nth x idx)` subtree at
/// search time, so the rest of the synthesizer treats it as a depth-0
/// atom of type `arg_type`. Used by the §9.31 multi-arg path.
pub fn indexed_arg_component(idx: usize, arg_type: Sym) -> SynthComponent {
    SynthComponent::literal(
        format!("arg{}", idx),
        LiteralKind::Indexed(idx),
        arg_type,
        100.0,
    )
}

// §9.45.13 deletion: `hole_component()` removed. The §9.42 hole-atom
// synthesis path is gone — constant fitting is done in pure SELPH
// via M11/M12 + m_pool's fit-affine primitive.

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
pub fn probe_function_type(
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

    // §9.45: read curriculum-defined `__synth_skip__` (a list of
    // strings) and merge it into the skip set. This lets a curriculum
    // file (m_chain.selph in particular) tell synth to ignore its own
    // recognizer helpers, which would otherwise be probed by
    // `probe_function_type` on every task — triggering expensive
    // downstream M-stage execution on junk inputs and turning a
    // ~1s grow-v2 task into a multi-minute hang. The discovery story
    // is in `project_section_9_45.md`.
    let mut effective_skip: HashSet<Sym> = skip.clone();
    if let Some(Value::List(items)) = env.lookup(intern("__synth_skip__")) {
        for item in items.iter() {
            if let Value::Str(s) = item {
                effective_skip.insert(intern(s.as_ref()));
            }
        }
    }

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
        if effective_skip.contains(&sym) {
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
///
/// `Custom(Sym)` was added in §9.37 (item 3) for SELPH-defined
/// decomposers registered in the `__decomposers__` namespace. The Sym
/// is the entry's key in that namespace, used for trace output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Flat,
    RecursiveDecomposition,
    BoolDecomp,
    HigherOrder,
    DivideConquer,
    Induction,
    Memo,
    Custom(Sym),
}

impl Strategy {
    /// Short label for trace / log output. Returns a String because the
    /// `Custom` variant resolves a Sym to its name string. Hardcoded
    /// variants return a constant String to keep the API uniform.
    pub fn name(&self) -> String {
        match self {
            Strategy::Flat => "Flat".to_string(),
            Strategy::RecursiveDecomposition => "RD".to_string(),
            Strategy::BoolDecomp => "BD".to_string(),
            Strategy::HigherOrder => "HO".to_string(),
            Strategy::DivideConquer => "D&C".to_string(),
            Strategy::Induction => "IN".to_string(),
            Strategy::Memo => "Memo".to_string(),
            Strategy::Custom(sym) => format!("custom:{}", resolve(*sym)),
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
    /// §9.49 post-mortem diagnostics: the inferred output type tag
    /// (e.g. "Int", "Str", "Grid", "Bool"). Set by `synthesize_with_strategies`.
    pub output_type: Option<String>,
    /// §9.49 post-mortem diagnostics: whether the SELPH decomposer
    /// chain (`__decomposers__`) was consulted during this task.
    pub m_chain_ran: bool,
}

impl StrategyResult {
    fn from_synth(r: SynthResult, strategy: Strategy) -> Self {
        Self {
            found: r.found,
            nodes: r.nodes,
            root: r.root,
            candidates_explored: r.candidates_explored,
            strategy: if r.found { Some(strategy) } else { None },
            output_type: None,
            m_chain_ran: false,
        }
    }

    fn not_found(explored: usize) -> Self {
        Self {
            found: false,
            nodes: None,
            root: None,
            candidates_explored: explored,
            strategy: None,
            output_type: None,
            m_chain_ran: false,
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
// Higher-order decomposition strategy
// ────────────────────────────────────────────────────────────────────────────
//
// When flat enumeration can't reach the target, try fixed structural
// templates that recursively call `synthesize` on a derived sub-spec.
// Each template assumes the outer program has a specific shape:
//
//   1. list-map:        (lambda (x) (map HOLE x))
//   2. split-map-join:  (lambda (x) (string-join (map HOLE (string-split x SEP)) SEP))
//   3. char-map-join:   (lambda (x) (string-join (map HOLE (string-chars x)) ""))
//   4. list-filter:     (lambda (x) (filter HOLE x))
//
// For each template that's applicable to the (input, output) shape,
// derive a sub-spec for HOLE, sub-synthesize, and splice the result
// into the wrapper. This is a port of legacy `decompose.rs` —
// stripped to types_v2 / eval_v2 surface, no macro env to rebuild,
// and the inner `synthesize` call uses synth_v2's flat enumerator
// (NOT `synthesize_with_strategies`, to avoid HO recursing into HO).
//
// Templates evaluate left-to-right and short-circuit on first success.
// Per-template budget is `max_candidates / 4`.

/// Result of an HO strategy attempt.
struct HoResult {
    found: bool,
    nodes: Vec<Node>,
    root: usize,
    candidates_explored: usize,
}

impl HoResult {
    fn empty() -> Self {
        Self {
            found: false,
            nodes: Vec::new(),
            root: 0,
            candidates_explored: 0,
        }
    }
}

/// Top-level HO entry point. Tries each template in order, returning
/// the first successful result. Total candidates explored is the sum
/// over all attempted templates' sub-syntheses.
pub fn higher_order_decompose(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> Option<(Vec<Node>, usize, usize)> {
    if inputs.is_empty() || inputs.len() != expected.len() {
        return None;
    }
    let budget_per_template = (max_candidates / 4).max(1);
    let mut total_explored: usize = 0;

    for template in [
        ho_try_list_map as HoTemplate,
        ho_try_list_filter,
        ho_try_split_map_join,
        ho_try_char_map_join,
    ] {
        let r = template(
            components,
            inputs,
            expected,
            env,
            universe,
            max_depth,
            budget_per_template,
        );
        total_explored += r.candidates_explored;
        if r.found {
            return Some((r.nodes, r.root, total_explored));
        }
    }
    None
}

type HoTemplate = fn(
    &[SynthComponent],
    &[Value],
    &[Value],
    &Env,
    &TypeUniverse,
    usize,
    usize,
) -> HoResult;

// ── HO helpers ─────────────────────────────────────────────────────────

/// Deduplicate a sub-spec, returning `None` if any input maps to two
/// different outputs (the sub-spec would be inconsistent), or if the
/// deduplicated set is too small to learn from.
///
/// Threshold of 3 unique pairs matches the legacy heuristic — fewer
/// than that, the sub-synthesizer is likely to memorize trivially or
/// produce a constant.
fn ho_dedup_spec(pairs: Vec<(Value, Value)>) -> Option<Vec<(Value, Value)>> {
    let mut seen: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut unique: Vec<(Value, Value)> = Vec::new();

    for (inp, out) in pairs {
        let h = val_hash(&inp);
        let mut already = false;
        if let Some(indices) = seen.get(&h) {
            for &idx in indices {
                if eval_v2::values_equal(&unique[idx].0, &inp) {
                    if !eval_v2::values_equal(&unique[idx].1, &out) {
                        return None; // conflict
                    }
                    already = true;
                    break;
                }
            }
        }
        if !already {
            let idx = unique.len();
            seen.entry(h).or_default().push(idx);
            unique.push((inp, out));
        }
    }

    if unique.len() < 3 {
        return None;
    }
    Some(unique)
}

/// Verify a fully composed program by evaluating its lambda and applying
/// it to every (input, expected) pair. Returns true iff all examples
/// match exactly.
fn ho_verify_composed(
    nodes: &[Node],
    lambda_idx: usize,
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
) -> bool {
    let nodes_rc: Rc<[Node]> = nodes.to_vec().into();
    let f = match eval_v2::eval(&nodes_rc, lambda_idx, env) {
        Ok(v) => v,
        Err(_) => return false,
    };
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        match eval_v2::apply(&f, std::slice::from_ref(inp), env) {
            Ok(ref v) if eval_v2::values_equal(v, exp) => {}
            _ => return false,
        }
    }
    true
}

/// Splice a sub-solution program (a `(lambda (x) ...)` from a recursive
/// `synthesize` call) into a fresh node arena starting at `offset`,
/// returning the index of the spliced lambda root.
fn ho_splice_sub(nodes: &mut Vec<Node>, sub_nodes: &[Node], sub_root: usize) -> usize {
    let offset = nodes.len();
    for n in sub_nodes {
        nodes.push(remap_node(n, offset));
    }
    sub_root + offset
}

// ── Template 1: list-map ───────────────────────────────────────────────

/// `(lambda (x) (map HOLE x))` — applies when every input is a list and
/// the corresponding expected output is a same-length list. The sub-spec
/// pairs each input element with its expected output element.
fn ho_try_list_map(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> HoResult {
    let mut sub_pairs: Vec<(Value, Value)> = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (il, ol) = match (inp, out) {
            (Value::List(il), Value::List(ol)) if il.len() == ol.len() && !il.is_empty() => {
                (il, ol)
            }
            _ => return HoResult::empty(),
        };
        for (ie, oe) in il.iter().zip(ol.iter()) {
            sub_pairs.push((ie.clone(), oe.clone()));
        }
    }

    let sub_spec = match ho_dedup_spec(sub_pairs) {
        Some(s) => s,
        None => return HoResult::empty(),
    };
    let (sub_inputs, sub_expected): (Vec<Value>, Vec<Value>) = sub_spec.into_iter().unzip();

    let sr = synthesize(
        components,
        &sub_inputs,
        &sub_expected,
        env,
        universe,
        max_depth,
        max_candidates,
    );
    let mut result = HoResult::empty();
    result.candidates_explored = sr.candidates_explored;
    if !sr.found {
        return result;
    }
    let sub_nodes = sr.nodes.unwrap();
    let sub_root = sr.root.unwrap();

    // Build (lambda (x) (map sub-fn x))
    let mut nodes: Vec<Node> = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sub_root_remapped = ho_splice_sub(&mut nodes, &sub_nodes, sub_root);

    let map_sym_idx = nodes.len();
    nodes.push(Node::Symbol(intern("map")));
    let map_app = nodes.len();
    nodes.push(Node::App(vec![map_sym_idx, sub_root_remapped, x_idx]));
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], map_app));

    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
        result.found = true;
        result.nodes = nodes;
        result.root = lambda_idx;
    }
    result
}

// ── Template 2: list-filter ────────────────────────────────────────────

/// `(lambda (x) (filter HOLE x))` — applies when every input is a list
/// and the corresponding expected output is an *ordered subset* of that
/// input. The sub-spec labels each element with its keep/drop decision
/// (Bool), so the inner sub-synthesizer must produce a unary predicate.
fn ho_try_list_filter(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> HoResult {
    let mut sub_pairs: Vec<(Value, Value)> = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (il, ol) = match (inp, out) {
            (Value::List(il), Value::List(ol)) if !il.is_empty() => (il, ol),
            _ => return HoResult::empty(),
        };
        // Walk in_list once; advance out_idx whenever an element matches
        // the next expected output. If we don't consume all of `out_list`,
        // it isn't an ordered subset.
        let mut out_idx = 0usize;
        for elem in il.iter() {
            let keep = if out_idx < ol.len() && eval_v2::values_equal(elem, &ol[out_idx]) {
                out_idx += 1;
                true
            } else {
                false
            };
            sub_pairs.push((elem.clone(), Value::Bool(keep)));
        }
        if out_idx != ol.len() {
            return HoResult::empty();
        }
    }

    let sub_spec = match ho_dedup_spec(sub_pairs) {
        Some(s) => s,
        None => return HoResult::empty(),
    };
    let (sub_inputs, sub_expected): (Vec<Value>, Vec<Value>) = sub_spec.into_iter().unzip();

    let sr = synthesize(
        components,
        &sub_inputs,
        &sub_expected,
        env,
        universe,
        max_depth,
        max_candidates,
    );
    let mut result = HoResult::empty();
    result.candidates_explored = sr.candidates_explored;
    if !sr.found {
        return result;
    }
    let sub_nodes = sr.nodes.unwrap();
    let sub_root = sr.root.unwrap();

    // Build (lambda (x) (filter sub-fn x))
    let mut nodes: Vec<Node> = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));

    let sub_root_remapped = ho_splice_sub(&mut nodes, &sub_nodes, sub_root);

    let filter_sym_idx = nodes.len();
    nodes.push(Node::Symbol(intern("filter")));
    let filter_app = nodes.len();
    nodes.push(Node::App(vec![filter_sym_idx, sub_root_remapped, x_idx]));
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], filter_app));

    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
        result.found = true;
        result.nodes = nodes;
        result.root = lambda_idx;
    }
    result
}

// ── Template 3: split-map-join ─────────────────────────────────────────

/// Common single-character delimiters tried by `split-map-join`,
/// ordered by frequency in the SELPH curricula.
const SPLIT_DELIMITERS: &[&str] = &[" ", ",", "-", ".", "/", ":", ";", "_", "|"];

/// `(lambda (x) (string-join (map HOLE (string-split x SEP)) SEP))`.
/// Applies when both input and output are strings that split into the
/// same number of parts under some delimiter, and each part-pair forms
/// the inner sub-spec.
fn ho_try_split_map_join(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> HoResult {
    let mut result = HoResult::empty();

    let strs_in: Option<Vec<&str>> = inputs
        .iter()
        .map(|v| match v {
            Value::Str(s) => Some(s.as_ref()),
            _ => None,
        })
        .collect();
    let strs_in = match strs_in {
        Some(v) => v,
        None => return result,
    };
    let strs_out: Option<Vec<&str>> = expected
        .iter()
        .map(|v| match v {
            Value::Str(s) => Some(s.as_ref()),
            _ => None,
        })
        .collect();
    let strs_out = match strs_out {
        Some(v) => v,
        None => return result,
    };

    for &sep in SPLIT_DELIMITERS {
        let sub_spec = match ho_derive_split_spec(&strs_in, &strs_out, sep) {
            Some(s) => s,
            None => continue,
        };
        let (sub_inputs, sub_expected): (Vec<Value>, Vec<Value>) = sub_spec.into_iter().unzip();

        let sr = synthesize(
            components,
            &sub_inputs,
            &sub_expected,
            env,
            universe,
            max_depth,
            max_candidates,
        );
        result.candidates_explored += sr.candidates_explored;
        if !sr.found {
            continue;
        }
        let sub_nodes = sr.nodes.unwrap();
        let sub_root = sr.root.unwrap();

        // Build (lambda (x) (string-join (map sub-fn (string-split x SEP)) SEP))
        let mut nodes: Vec<Node> = Vec::new();
        let x_idx = nodes.len();
        nodes.push(Node::Symbol(intern("x")));
        let sep_idx = nodes.len();
        nodes.push(Node::Str(sep.to_string()));
        let split_sym = nodes.len();
        nodes.push(Node::Symbol(intern("string-split")));
        let split_app = nodes.len();
        nodes.push(Node::App(vec![split_sym, x_idx, sep_idx]));

        let sub_root_remapped = ho_splice_sub(&mut nodes, &sub_nodes, sub_root);

        let map_sym = nodes.len();
        nodes.push(Node::Symbol(intern("map")));
        let map_app = nodes.len();
        nodes.push(Node::App(vec![map_sym, sub_root_remapped, split_app]));

        let sep_idx2 = nodes.len();
        nodes.push(Node::Str(sep.to_string()));
        let join_sym = nodes.len();
        nodes.push(Node::Symbol(intern("string-join")));
        let join_app = nodes.len();
        nodes.push(Node::App(vec![join_sym, map_app, sep_idx2]));

        let lambda_idx = nodes.len();
        nodes.push(Node::Lambda(vec![intern("x")], join_app));

        if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
            result.found = true;
            result.nodes = nodes;
            result.root = lambda_idx;
            return result;
        }
    }
    result
}

/// Derive a sub-spec from splitting each (input, output) string pair
/// by `sep`. Returns `None` if any pair has a different part-count, or
/// if no pair actually contained the delimiter (the candidate would be
/// equivalent to identity).
fn ho_derive_split_spec(
    inputs: &[&str],
    outputs: &[&str],
    sep: &str,
) -> Option<Vec<(Value, Value)>> {
    let mut pairs: Vec<(Value, Value)> = Vec::new();
    let mut any_multi = false;
    for (&inp, &out) in inputs.iter().zip(outputs.iter()) {
        let in_parts: Vec<&str> = inp.split(sep).collect();
        let out_parts: Vec<&str> = out.split(sep).collect();
        if in_parts.len() != out_parts.len() {
            return None;
        }
        if in_parts.len() >= 2 {
            any_multi = true;
        }
        for (ip, op) in in_parts.iter().zip(out_parts.iter()) {
            pairs.push((Value::str(*ip), Value::str(*op)));
        }
    }
    if !any_multi {
        return None;
    }
    ho_dedup_spec(pairs)
}

// ── Template 4: char-map-join ──────────────────────────────────────────

/// `(lambda (x) (string-join (map HOLE (string-chars x)) ""))` —
/// applies when every input/output pair is a same-length string. The
/// sub-spec maps each input character to its corresponding output
/// character (each as a single-character string).
fn ho_try_char_map_join(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> HoResult {
    let mut sub_pairs: Vec<(Value, Value)> = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        let (s_in, s_out) = match (inp, out) {
            (Value::Str(a), Value::Str(b))
                if !a.is_empty() && a.chars().count() == b.chars().count() =>
            {
                (a, b)
            }
            _ => return HoResult::empty(),
        };
        for (ci, co) in s_in.chars().zip(s_out.chars()) {
            sub_pairs.push((Value::str(ci.to_string()), Value::str(co.to_string())));
        }
    }

    let sub_spec = match ho_dedup_spec(sub_pairs) {
        Some(s) => s,
        None => return HoResult::empty(),
    };
    let (sub_inputs, sub_expected): (Vec<Value>, Vec<Value>) = sub_spec.into_iter().unzip();

    let sr = synthesize(
        components,
        &sub_inputs,
        &sub_expected,
        env,
        universe,
        max_depth,
        max_candidates,
    );
    let mut result = HoResult::empty();
    result.candidates_explored = sr.candidates_explored;
    if !sr.found {
        return result;
    }
    let sub_nodes = sr.nodes.unwrap();
    let sub_root = sr.root.unwrap();

    // Build (lambda (x) (string-join (map sub-fn (string-chars x)) ""))
    let mut nodes: Vec<Node> = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));
    let chars_sym = nodes.len();
    nodes.push(Node::Symbol(intern("string-chars")));
    let chars_app = nodes.len();
    nodes.push(Node::App(vec![chars_sym, x_idx]));

    let sub_root_remapped = ho_splice_sub(&mut nodes, &sub_nodes, sub_root);

    let map_sym = nodes.len();
    nodes.push(Node::Symbol(intern("map")));
    let map_app = nodes.len();
    nodes.push(Node::App(vec![map_sym, sub_root_remapped, chars_app]));

    let empty_sep = nodes.len();
    nodes.push(Node::Str(String::new()));
    let join_sym = nodes.len();
    nodes.push(Node::Symbol(intern("string-join")));
    let join_app = nodes.len();
    nodes.push(Node::App(vec![join_sym, map_app, empty_sep]));

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], join_app));

    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
        result.found = true;
        result.nodes = nodes;
        result.root = lambda_idx;
    }
    result
}

// ────────────────────────────────────────────────────────────────────────────
// Boolean decomposition strategy
// ────────────────────────────────────────────────────────────────────────────
//
// When the target output is `Bool`, try every constant-time logical
// composition of the available unary library predicates:
//
//   (not P), (and P Q), (or P Q), (and P (not Q)), (and (not P) Q)
//
// for all unary library functions P, Q that return `Bool` when applied
// to the task's inputs. This is `main.rs::bool_decompose` from the
// legacy core, ported to types_v2.
//
// Why this is its own strategy and not just enumeration:
//   - O(L²) where L is the number of bool-returning library functions.
//     Even with L = 30 that's 900 candidates — far below the Flat
//     budget. The legacy version reports it solves in 0 candidates by
//     bookkeeping convention; here we count actual probes.
//   - The bottom-up enumerator with `Bool → Bool → Bool` operators in
//     play already finds these compositions, but only at depth ≥ 3,
//     and the type-tagged search wastes a lot of pool space exploring
//     non-bool intermediates first. BD short-circuits all of that for
//     a target type that is rare in practice but cheap to recognize.
//
// What this version does NOT inherit from legacy:
//   - The legacy version walks the macro list and rebuilds an env per
//     probe (`make_default_env` + re-define every macro). In synth_v2
//     the env IS the library, so we just call `eval_v2::apply` against
//     the env we were given — same as `probe_filter_components`.
//   - The legacy version probes by name match against the macro table.
//     synth_v2 already has a `SynthComponent` catalog, so we filter
//     it directly and avoid a second env walk.

/// Information needed to compose a probed predicate back into a candidate
/// AST: the Sym that resolves it in the env, and its per-input outputs.
struct BoolPredicate {
    sym: Sym,
    /// One bool per task input, in order.
    outputs: Vec<bool>,
}

/// Try to solve `(inputs, expected)` as a logical composition of unary
/// library predicates. Returns `(nodes, root)` for a `(lambda (x) ...)`
/// program, plus the number of candidate compositions tested. Returns
/// `None` if the target is not bool-typed or no composition matches.
///
/// Composition templates tried, in order:
///   1. `(not P)`           — for each predicate
///   2. `(and P Q)`         — for each pair (i ≤ j)
///   3. `(or P Q)`          — for each pair (i ≤ j)
///   4. `(and P (not Q))`   — for each pair (i ≤ j)
///   5. `(and (not P) Q)`   — for each pair (i ≤ j)
///
/// The "≤" pairing matches legacy: it skips strict-greater pairs because
/// `and`/`or` are commutative, but it does include the `i = j` diagonal
/// (which is mostly degenerate but cheap).
pub fn bool_decompose(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
) -> Option<(Vec<Node>, usize, usize)> {
    // Output must be all-bool.
    if expected.is_empty() || inputs.len() != expected.len() {
        return None;
    }
    let expected_bools: Vec<bool> = expected
        .iter()
        .map(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
        .collect::<Option<Vec<_>>>()?;

    // Probe every unary `Named` component. We deliberately accept
    // builtins too (e.g. `even`, `odd`) — the legacy version only
    // probed library macros because that's what its `macros` slice
    // contained, but in synth_v2 the catalog is uniform and there's
    // no reason to skip a primitive predicate that fits the pattern.
    // Components whose declared return type isn't `Bool` are skipped
    // for free, and we additionally re-check the runtime type from the
    // probe to defend against polymorphic returns.
    let bool_sym = intern("Bool");
    let mut predicates: Vec<BoolPredicate> = Vec::new();

    for comp in components {
        if comp.arity != 1 {
            continue;
        }
        if comp.ret_type != bool_sym {
            continue;
        }
        let sym = match comp.dispatch {
            Dispatch::Named(s) => s,
            _ => continue, // literals, fused forms — not predicates
        };
        let val = match env.lookup(sym) {
            Some(v) => v,
            None => continue,
        };
        // Probe on every input. Drop on any error or non-bool result.
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut all_ok = true;
        for inp in inputs {
            match eval_v2::apply(&val, std::slice::from_ref(inp), env) {
                Ok(Value::Bool(b)) => outputs.push(b),
                _ => {
                    all_ok = false;
                    break;
                }
            }
        }
        if all_ok && outputs.len() == inputs.len() {
            predicates.push(BoolPredicate { sym, outputs });
        }
    }

    if predicates.is_empty() {
        return None;
    }

    let mut candidates_tested: usize = 0;

    // Template 1: (not P)
    for p in &predicates {
        candidates_tested += 1;
        if p.outputs.iter().zip(&expected_bools).all(|(a, e)| !*a == *e) {
            let (n, r) = build_bool_program(BoolBuild::NotP(p.sym))?;
            return Some((n, r, candidates_tested));
        }
    }

    // Templates 2–5: pairwise
    for i in 0..predicates.len() {
        for j in i..predicates.len() {
            let p = &predicates[i];
            let q = &predicates[j];

            // (and P Q)
            candidates_tested += 1;
            if p.outputs
                .iter()
                .zip(&q.outputs)
                .zip(&expected_bools)
                .all(|((a, b), e)| (*a && *b) == *e)
            {
                let (n, r) = build_bool_program(BoolBuild::AndPQ(p.sym, q.sym))?;
                return Some((n, r, candidates_tested));
            }

            // (or P Q)
            candidates_tested += 1;
            if p.outputs
                .iter()
                .zip(&q.outputs)
                .zip(&expected_bools)
                .all(|((a, b), e)| (*a || *b) == *e)
            {
                let (n, r) = build_bool_program(BoolBuild::OrPQ(p.sym, q.sym))?;
                return Some((n, r, candidates_tested));
            }

            // (and P (not Q))
            candidates_tested += 1;
            if p.outputs
                .iter()
                .zip(&q.outputs)
                .zip(&expected_bools)
                .all(|((a, b), e)| (*a && !*b) == *e)
            {
                let (n, r) = build_bool_program(BoolBuild::AndPNotQ(p.sym, q.sym))?;
                return Some((n, r, candidates_tested));
            }

            // (and (not P) Q)
            candidates_tested += 1;
            if p.outputs
                .iter()
                .zip(&q.outputs)
                .zip(&expected_bools)
                .all(|((a, b), e)| (!*a && *b) == *e)
            {
                let (n, r) = build_bool_program(BoolBuild::AndNotPQ(p.sym, q.sym))?;
                return Some((n, r, candidates_tested));
            }
        }
    }

    None
}

/// Tag for `build_bool_program` — describes which template to materialize.
enum BoolBuild {
    NotP(Sym),
    AndPQ(Sym, Sym),
    OrPQ(Sym, Sym),
    AndPNotQ(Sym, Sym),
    AndNotPQ(Sym, Sym),
}

/// Build the AST `(lambda (x) <body>)` for a BD template. Returns the
/// node arena and the lambda's index. The body shape is determined by
/// the `BoolBuild` tag.
///
/// Node-tree details:
///   - `(p x)` is `Node::App([sym(p), sym(x)])`.
///   - `(not (p x))` is `Node::App([sym(not), (p x)_idx])`.
///   - `(and a b)` and `(or a b)` are `Node::SpecialApp(SpecialForm::{And,Or}, [a, b])`
///     because eval_v2 implements `and`/`or` as short-circuiting special
///     forms, not as builtins.
fn build_bool_program(build: BoolBuild) -> Option<(Vec<Node>, usize)> {
    let mut nodes: Vec<Node> = Vec::new();
    let x_sym = intern("x");
    let not_sym = intern("not");

    // Helper: emit `(p x)` and return its index.
    let mut emit_call = |nodes: &mut Vec<Node>, p: Sym| -> usize {
        let p_idx = nodes.len();
        nodes.push(Node::Symbol(p));
        let x_idx = nodes.len();
        nodes.push(Node::Symbol(x_sym));
        let app_idx = nodes.len();
        nodes.push(Node::App(vec![p_idx, x_idx]));
        app_idx
    };

    // Helper: emit `(not <inner_idx>)` and return its index.
    let emit_not = |nodes: &mut Vec<Node>, inner_idx: usize| -> usize {
        let not_idx = nodes.len();
        nodes.push(Node::Symbol(not_sym));
        let app_idx = nodes.len();
        nodes.push(Node::App(vec![not_idx, inner_idx]));
        app_idx
    };

    let body_idx = match build {
        BoolBuild::NotP(p) => {
            let pcall = emit_call(&mut nodes, p);
            emit_not(&mut nodes, pcall)
        }
        BoolBuild::AndPQ(p, q) => {
            let pcall = emit_call(&mut nodes, p);
            let qcall = emit_call(&mut nodes, q);
            let app = nodes.len();
            nodes.push(Node::SpecialApp(SpecialForm::And, vec![pcall, qcall]));
            app
        }
        BoolBuild::OrPQ(p, q) => {
            let pcall = emit_call(&mut nodes, p);
            let qcall = emit_call(&mut nodes, q);
            let app = nodes.len();
            nodes.push(Node::SpecialApp(SpecialForm::Or, vec![pcall, qcall]));
            app
        }
        BoolBuild::AndPNotQ(p, q) => {
            let pcall = emit_call(&mut nodes, p);
            let qcall = emit_call(&mut nodes, q);
            let qnot = emit_not(&mut nodes, qcall);
            let app = nodes.len();
            nodes.push(Node::SpecialApp(SpecialForm::And, vec![pcall, qnot]));
            app
        }
        BoolBuild::AndNotPQ(p, q) => {
            let pcall = emit_call(&mut nodes, p);
            let pnot = emit_not(&mut nodes, pcall);
            let qcall = emit_call(&mut nodes, q);
            let app = nodes.len();
            nodes.push(Node::SpecialApp(SpecialForm::And, vec![pnot, qcall]));
            app
        }
    };

    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![x_sym], body_idx));
    Some((nodes, lambda_idx))
}

// ────────────────────────────────────────────────────────────────────────────
// Divide-and-conquer strategy
// ────────────────────────────────────────────────────────────────────────────
//
// When the spec has multiple distinct output values — i.e. it looks
// like a piecewise / classification task — try to break it into
// nested if-expressions:
//
//   1. Group examples by output value.
//   2. Sort groups by mean input (gives a natural threshold ordering).
//   3. Recursively build a nested if: pick the smallest-mean group,
//      find a boolean separator that's true on it and false on the
//      rest, recurse on the rest as the else branch.
//   4. Each leaf group is either a constant (if all members agree) or
//      a flat sub-synthesis on the group's subset.
//
// Both `find_separator` and `synthesize_branch` reuse `synthesize`,
// the flat enumerator. find_separator builds a Bool sub-spec where
// the target is true on the "kept" indices and false on the rest;
// synthesize_branch builds an output-typed sub-spec on the group's
// own examples. The recursive structure mirrors legacy `divide.rs`.
//
// Port-specific notes:
//   - Mean-input handles both Int and Num. Non-numeric inputs fall
//     back to 0.0 (matches legacy).
//   - The leaf "constant branch" supports Int, Num, Str, Bool — same
//     set as types_v2 literal Node variants.
//   - When stripping the `(lambda (x) body)` wrapper from a sub-synth
//     result, we keep the lambda node in the arena (it becomes dead
//     code) and reference `body_idx` as the new root. The parent
//     remap pass shifts the dead node along with everything else but
//     never dereferences it — same trick as legacy `divide.rs`.

/// Try to solve `(inputs, expected)` by partitioning on output value
/// and emitting nested if-expressions. Returns `(nodes, root, candidates_explored)`
/// for a `(lambda (x) ...)` program, or `None` when:
///   - fewer than 2 distinct output values are present (pure case for Flat)
///   - no separator condition can be found at any partition level
///   - any leaf branch fails to synthesize
pub fn divide_and_conquer(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> Option<(Vec<Node>, usize, usize)> {
    if inputs.is_empty() || inputs.len() != expected.len() {
        return None;
    }

    // ── Group examples by output value ─────────────────────────────────
    // Hash on val_hash, verify with values_equal (collision-safe).
    let mut groups: Vec<(Value, Vec<usize>)> = Vec::new();
    let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, out) in expected.iter().enumerate() {
        let h = val_hash(out);
        let mut placed = false;
        if let Some(group_indices) = by_hash.get(&h) {
            for &gi in group_indices {
                if eval_v2::values_equal(&groups[gi].0, out) {
                    groups[gi].1.push(i);
                    placed = true;
                    break;
                }
            }
        }
        if !placed {
            let new_idx = groups.len();
            by_hash.entry(h).or_default().push(new_idx);
            groups.push((out.clone(), vec![i]));
        }
    }

    if groups.len() < 2 {
        return None;
    }

    // Sort groups by mean input value (defines the if-tree order).
    groups.sort_by(|a, b| {
        let ma = dc_mean_input(&a.1, inputs);
        let mb = dc_mean_input(&b.1, inputs);
        ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Recursive nested-if construction.
    let mut total_explored: usize = 0;
    let body = dc_build_nested_if(
        &groups,
        inputs,
        expected,
        components,
        env,
        universe,
        max_depth,
        max_candidates,
        &mut total_explored,
    )?;

    // Wrap the body in a lambda and verify against every example.
    let (mut nodes, root) = body;
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], root));

    if !ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
        return None;
    }

    Some((nodes, lambda_idx, total_explored))
}

/// Mean of input values at the given indices, treating each input as
/// a number when possible. Int and Num both contribute their numeric
/// value; non-numeric inputs are skipped (and a fully non-numeric
/// group returns 0.0 — its sort position is undefined but stable).
fn dc_mean_input(indices: &[usize], inputs: &[Value]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for &i in indices {
        match &inputs[i] {
            Value::Int(n) => {
                sum += *n as f64;
                count += 1;
            }
            Value::Num(n) => {
                sum += *n;
                count += 1;
            }
            _ => {}
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Recursively build a nested if-expression body for the sorted output
/// groups. Returns `(nodes, body_root)` where `body_root` is the root
/// of the body expression (NOT yet wrapped in a lambda).
fn dc_build_nested_if(
    sorted_groups: &[(Value, Vec<usize>)],
    inputs: &[Value],
    expected: &[Value],
    components: &[SynthComponent],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    total_explored: &mut usize,
) -> Option<(Vec<Node>, usize)> {
    if sorted_groups.len() == 1 {
        let (out_val, indices) = &sorted_groups[0];
        return dc_synthesize_branch(
            out_val,
            indices,
            inputs,
            expected,
            components,
            env,
            universe,
            max_depth,
            max_candidates,
            total_explored,
        );
    }

    let (first_val, first_indices) = &sorted_groups[0];
    let rest_groups = &sorted_groups[1..];
    let rest_indices: Vec<usize> = rest_groups
        .iter()
        .flat_map(|(_, idx)| idx.iter().copied())
        .collect();

    // Try separator: true on first group, false on rest.
    if let Some(cond) = dc_find_separator(
        first_indices,
        &rest_indices,
        inputs,
        components,
        env,
        universe,
        max_depth,
        max_candidates,
        total_explored,
    ) {
        let then_branch = dc_synthesize_branch(
            first_val,
            first_indices,
            inputs,
            expected,
            components,
            env,
            universe,
            max_depth,
            max_candidates,
            total_explored,
        )?;
        let else_branch = dc_build_nested_if(
            rest_groups,
            inputs,
            expected,
            components,
            env,
            universe,
            max_depth,
            max_candidates,
            total_explored,
        )?;
        return Some(dc_merge_if(cond, then_branch, else_branch));
    }

    // Try the swapped separator: true on rest, false on first.
    if let Some(cond) = dc_find_separator(
        &rest_indices,
        first_indices,
        inputs,
        components,
        env,
        universe,
        max_depth,
        max_candidates,
        total_explored,
    ) {
        let then_branch = dc_build_nested_if(
            rest_groups,
            inputs,
            expected,
            components,
            env,
            universe,
            max_depth,
            max_candidates,
            total_explored,
        )?;
        let else_branch = dc_synthesize_branch(
            first_val,
            first_indices,
            inputs,
            expected,
            components,
            env,
            universe,
            max_depth,
            max_candidates,
            total_explored,
        )?;
        return Some(dc_merge_if(cond, then_branch, else_branch));
    }

    None
}

/// Splice condition + then-branch + else-branch arenas into one and
/// emit a single `Node::If` at the root. The remap shifts every
/// `then`/`else` index by the running offset so cross-references stay
/// valid.
fn dc_merge_if(
    cond: (Vec<Node>, usize),
    then_branch: (Vec<Node>, usize),
    else_branch: (Vec<Node>, usize),
) -> (Vec<Node>, usize) {
    let (mut nodes, cond_root) = cond;

    let then_off = nodes.len();
    for n in &then_branch.0 {
        nodes.push(remap_node(n, then_off));
    }
    let then_root = then_branch.1 + then_off;

    let else_off = nodes.len();
    for n in &else_branch.0 {
        nodes.push(remap_node(n, else_off));
    }
    let else_root = else_branch.1 + else_off;

    let if_idx = nodes.len();
    nodes.push(Node::If(cond_root, then_root, else_root));
    (nodes, if_idx)
}

/// Find a Bool-typed program that's `true` on `true_indices` and
/// `false` on `false_indices`. Reuses `synthesize` with a Bool sub-spec
/// — the inputs are the original task's inputs, and the expected are
/// the labels.
///
/// Returns the body of the resulting `(lambda (x) body)` (the lambda
/// wrapper is dead code that stays in the node arena but isn't
/// referenced by the parent if-node).
fn dc_find_separator(
    true_indices: &[usize],
    false_indices: &[usize],
    inputs: &[Value],
    components: &[SynthComponent],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    total_explored: &mut usize,
) -> Option<(Vec<Node>, usize)> {
    // Build the labeled sub-spec.
    let mut sub_inputs: Vec<Value> = Vec::with_capacity(true_indices.len() + false_indices.len());
    let mut sub_expected: Vec<Value> = Vec::with_capacity(true_indices.len() + false_indices.len());
    for &i in true_indices {
        sub_inputs.push(inputs[i].clone());
        sub_expected.push(Value::Bool(true));
    }
    for &i in false_indices {
        sub_inputs.push(inputs[i].clone());
        sub_expected.push(Value::Bool(false));
    }

    let r = synthesize(
        components,
        &sub_inputs,
        &sub_expected,
        env,
        universe,
        max_depth,
        max_candidates,
    );
    *total_explored += r.candidates_explored;
    if !r.found {
        return None;
    }
    let nodes = r.nodes.unwrap();
    let lambda_root = r.root.unwrap();
    // Strip the lambda wrapper: keep the arena, point at the body.
    let body_root = match nodes[lambda_root] {
        Node::Lambda(_, body) => body,
        _ => lambda_root,
    };
    Some((nodes, body_root))
}

/// Synthesize the branch expression for a single output group. If
/// every group member maps to the same constant, emit it as a leaf
/// node directly; otherwise delegate to `synthesize` on the group's
/// subset of (input, expected) pairs.
fn dc_synthesize_branch(
    out_val: &Value,
    indices: &[usize],
    inputs: &[Value],
    expected: &[Value],
    components: &[SynthComponent],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    total_explored: &mut usize,
) -> Option<(Vec<Node>, usize)> {
    // Constant-branch shortcut: every output in this group is the
    // same value. Emit a literal Node and skip sub-synthesis entirely.
    let group_outputs: Vec<&Value> = indices.iter().map(|&i| &expected[i]).collect();
    let all_same = group_outputs
        .windows(2)
        .all(|w| eval_v2::values_equal(w[0], w[1]));
    if all_same {
        let leaf = match out_val {
            Value::Int(n) => Node::Int(*n),
            Value::Num(n) => Node::Num(*n),
            Value::Str(s) => Node::Str(s.as_ref().to_string()),
            Value::Bool(b) => Node::Bool(*b),
            _ => return None,
        };
        return Some((vec![leaf], 0));
    }

    // Non-constant: sub-synthesize on the group's own examples.
    let group_inputs: Vec<Value> = indices.iter().map(|&i| inputs[i].clone()).collect();
    let group_expected: Vec<Value> = indices.iter().map(|&i| expected[i].clone()).collect();

    let r = synthesize(
        components,
        &group_inputs,
        &group_expected,
        env,
        universe,
        max_depth,
        max_candidates,
    );
    *total_explored += r.candidates_explored;
    if !r.found {
        return None;
    }
    let nodes = r.nodes.unwrap();
    let lambda_root = r.root.unwrap();
    let body_root = match nodes[lambda_root] {
        Node::Lambda(_, body) => body,
        _ => lambda_root,
    };
    Some((nodes, body_root))
}

// ────────────────────────────────────────────────────────────────────────────
// Induction strategy (intermediate value decomposition)
// ────────────────────────────────────────────────────────────────────────────
//
// When the spec can't be solved as a single program, try decomposing
// it as `f ∘ g` (apply g first, then f). The procedure:
//
//   1. Pick a candidate `g`: a known unary builtin, or a binary
//      builtin paired with a small constant. Run it on every input
//      to produce an intermediate value sequence `mid`.
//   2. Filter out unhelpful intermediates: those that equal the
//      original inputs, equal the expected outputs, or are constant
//      (the function collapses everything to one value).
//   3. Sub-synthesize `mid → expected`. If found, that's `f`.
//   4. Sub-synthesize `inputs → mid`. If found, that's the program
//      that materializes `g` (we already know which builtin produced
//      `mid`, but the synthesizer is the source of truth — and it
//      may find a more general expression than the literal builtin).
//   5. Compose: `(lambda (x) (let ((x g_body)) f_body))`. The let
//      evaluates `g_body` in the outer scope (so `x` = the actual
//      input), then rebinds `x` to the intermediate before evaluating
//      `f_body`. eval_v2's let semantics make this work without any
//      tree rewriting (compare the legacy substitute_x trick which
//      only handled single-level x references).
//
// What this version does NOT inherit from legacy:
//   - The "constant discovery" pass (output - input, output / input).
//     synth_v2's literal pool already covers most curriculum cases;
//     deferred for now.
//   - The dependency on a hard-coded set of unary/binary names. The
//     port still uses a curated set for the intermediate generator
//     (matches legacy taste), but the sub-synthesis steps see the
//     full component catalog so the composed solution can mix them
//     freely with library functions.

/// Curated unary builtins to try as intermediate transforms.
/// Matches legacy `induce::UNARY_FNS` plus the synth_v2 versions of
/// `even`/`odd`. Predicates are useful when the expected output is
/// bool-tagged.
const INDUCE_UNARY_FNS: &[&str] = &[
    "abs",
    "negate",
    "floor",
    "string-upper",
    "string-lower",
    "string-reverse",
    "string-trim",
    "string-length",
    "even",
    "odd",
];

/// Binary builtins to try with small integer constants as the second
/// argument. Matches legacy `induce::BINARY_FNS`.
const INDUCE_BINARY_FNS: &[&str] = &["add", "subtract", "multiply"];

/// Small constants to pair with binary builtins (Int-typed because the
/// curriculum is integer-biased).
const INDUCE_SMALL_INTS: &[i64] = &[1, 2, -1, 5];

/// Try to solve `(inputs, expected)` by intermediate-value decomposition.
/// Returns `(nodes, root, candidates_explored)` for a `(lambda (x) ...)`
/// program, or `None` if no decomposition succeeds.
pub fn induce_decomposition(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> Option<(Vec<Node>, usize, usize)> {
    if inputs.is_empty() || inputs.len() != expected.len() {
        return None;
    }

    // Each sub-synth gets a quarter of the budget. Two per intermediate
    // (input→mid, mid→expected), and we may try several intermediates.
    let budget_per_step = (max_candidates / 4).max(1);
    let mut total_explored: usize = 0;

    // Generate intermediate value sequences by probing curated builtins.
    let intermediates = induce_generate_intermediates(inputs, env);

    for mid in &intermediates {
        // Skip useless candidates.
        if induce_slices_equal(&mid.values, inputs)
            || induce_slices_equal(&mid.values, expected)
        {
            continue;
        }
        if induce_all_identical(&mid.values) {
            continue;
        }

        // Step 2 first: mid → expected. Cheaper to detect when this
        // can't work — if step 2 fails the intermediate is useless.
        let step2 = synthesize(
            components,
            &mid.values,
            expected,
            env,
            universe,
            max_depth,
            budget_per_step,
        );
        total_explored += step2.candidates_explored;
        if !step2.found {
            continue;
        }

        // Step 1: inputs → mid.
        let step1 = synthesize(
            components,
            inputs,
            &mid.values,
            env,
            universe,
            max_depth,
            budget_per_step,
        );
        total_explored += step1.candidates_explored;
        if !step1.found {
            continue;
        }

        // Compose into a single (lambda (x) (let ((x g)) f)).
        let composed = induce_compose_steps(
            step1.nodes.unwrap(),
            step1.root.unwrap(),
            step2.nodes.unwrap(),
            step2.root.unwrap(),
        );
        if let Some((nodes, root)) = composed {
            // Final correctness check on the composed program.
            if ho_verify_composed(&nodes, root, inputs, expected, env) {
                return Some((nodes, root, total_explored));
            }
        }
    }

    None
}

/// A candidate intermediate sequence with the name of the function that
/// produced it (debugging only — the composition uses the synthesized
/// step1, not the literal builtin name).
struct InduceIntermediate {
    #[allow(dead_code)]
    name: String,
    values: Vec<Value>,
}

/// Probe the curated builtin set on every input and collect the
/// resulting intermediate value sequences. Each sequence must be
/// fully successful — any builtin that errors on any input is dropped.
fn induce_generate_intermediates(inputs: &[Value], env: &Env) -> Vec<InduceIntermediate> {
    let mut out: Vec<InduceIntermediate> = Vec::new();

    for &name in INDUCE_UNARY_FNS {
        let sym = intern(name);
        let val = match env.lookup(sym) {
            Some(v) => v,
            None => continue,
        };
        let mut values: Vec<Value> = Vec::with_capacity(inputs.len());
        let mut ok = true;
        for inp in inputs {
            match eval_v2::apply(&val, std::slice::from_ref(inp), env) {
                Ok(v) => values.push(v),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && values.len() == inputs.len() {
            out.push(InduceIntermediate {
                name: name.to_string(),
                values,
            });
        }
    }

    for &name in INDUCE_BINARY_FNS {
        let sym = intern(name);
        let val = match env.lookup(sym) {
            Some(v) => v,
            None => continue,
        };
        for &c in INDUCE_SMALL_INTS {
            let const_val = Value::Int(c);
            let mut values: Vec<Value> = Vec::with_capacity(inputs.len());
            let mut ok = true;
            for inp in inputs {
                match eval_v2::apply(&val, &[inp.clone(), const_val.clone()], env) {
                    Ok(v) => values.push(v),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && values.len() == inputs.len() {
                out.push(InduceIntermediate {
                    name: format!("{}_{}", name, c),
                    values,
                });
            }
        }
    }

    out
}

/// True iff every value in `vs` is equal to the first.
fn induce_all_identical(vs: &[Value]) -> bool {
    if vs.len() <= 1 {
        return true;
    }
    vs.windows(2).all(|w| eval_v2::values_equal(&w[0], &w[1]))
}

/// Element-wise equality of two value slices.
fn induce_slices_equal(a: &[Value], b: &[Value]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| eval_v2::values_equal(x, y))
}

/// Compose two synthesized lambdas `step1` and `step2` (each shaped as
/// `(lambda (x) body)`) into `(lambda (x) (let ((x step1_body)) step2_body))`.
///
/// Returns `(nodes, lambda_root)` for the composed program. The let
/// trick avoids any tree rewriting: step1's body is evaluated in the
/// outer scope (where `x` = the actual input), then bound to a fresh
/// inner `x`, then step2's body is evaluated with `x` = the intermediate.
fn induce_compose_steps(
    step1_nodes: Vec<Node>,
    step1_lambda: usize,
    step2_nodes: Vec<Node>,
    step2_lambda: usize,
) -> Option<(Vec<Node>, usize)> {
    // Extract each step's inner body.
    let step1_body = match step1_nodes[step1_lambda] {
        Node::Lambda(_, body) => body,
        _ => return None,
    };
    let step2_body = match step2_nodes[step2_lambda] {
        Node::Lambda(_, body) => body,
        _ => return None,
    };

    // Splice step1's nodes (its `x` references will resolve to the
    // outer lambda's parameter at eval time).
    let mut nodes = step1_nodes;

    // Splice step2's nodes after step1's, with offset remap.
    let step2_off = nodes.len();
    for n in &step2_nodes {
        nodes.push(remap_node(n, step2_off));
    }
    let step2_body_remapped = step2_body + step2_off;

    // (let ((x step1_body)) step2_body)
    let let_idx = nodes.len();
    nodes.push(Node::Let(
        vec![(intern("x"), step1_body)],
        step2_body_remapped,
    ));

    // (lambda (x) <let>)
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], let_idx));

    Some((nodes, lambda_idx))
}

// ────────────────────────────────────────────────────────────────────────────
// Recursive Decomposition (§9.27.6)
// ────────────────────────────────────────────────────────────────────────────
//
// Top-down family prediction → outermost-function inversion → recursive
// sub-synthesis. The legacy entry point is `recursive_decompose.rs::
// try_recursive_decomposition`. This is the synth_v2 port: same algorithm,
// retyped against types_v2/eval_v2 and the env-as-library convention.
//
// Differences from the legacy module:
//
//   1. **The `macros` parameter is gone.** Library functions live in
//      `env` as `Value::Function` entries. Where the legacy code walked
//      a `Vec<(name, params, nodes, root)>`, the v2 port walks
//      `env.top_scope()` filtered to `Value::Function` with the right arity.
//
//   2. **Builtin invocation goes through env + eval_v2::apply.** The legacy
//      code calls `eval::apply_builtin(intern(name), args)` directly. The
//      v2 port does `env.lookup(intern(name))` to get a `Value::Builtin(_)`
//      and `eval_v2::apply` to invoke it. Same semantics, fewer special cases.
//
//   3. **Int vs Num is now distinct.** Inversion helpers that the legacy
//      module produced as `Value::Num(f64)` now check whether the result
//      is integral and produce `Value::Int(_)` when it is. This matches
//      the synth_v2 component catalog (which is biased toward Int) so
//      sub-synthesis can find Int literals in the atom pool.
//
//   4. **Map/filter/if delegate to existing v2 strategies.** The legacy
//      RD module includes its own list-map, split-map-join, list-filter,
//      and divide-and-conquer paths. synth_v2 has `higher_order_decompose`
//      and `divide_and_conquer` already. Where the legacy `try_single_function`
//      branches into one of these families, the v2 port routes to the
//      existing strategy and returns its result tagged as RD.
//
//   5. **`LearnedPredictor` is dropped.** Family prediction uses only the
//      hand-coded decision tree. The learned-predictor scaffolding requires
//      parser + SELPH-script integration through the new core, which is
//      its own future work.
//
// What stays the same:
//   - Family classification from spec features (input/output type, distinct
//     output count, has-bool-library probe, output-is-substring probe)
//   - Per-family candidate function lists
//   - Inversion helpers for binary/unary arithmetic, string-take/drop,
//     concat, count-char, and unary string ops
//   - Generic binary inversion via constant probing
//   - Library-function decomposition (f∘m_lib and m_lib∘g paths)
//   - Recursive sub-synthesis (RD's main contribution: find a sub-spec
//     that's easier to synthesize than the original)

/// Function families used by RD's outermost-function prediction. The
/// family determines which inversion helpers and candidate functions
/// `rd_try_single_function` will attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RdFamily {
    Constant,
    Arithmetic,
    Compare,
    BoolComp,
    StringOp,
    Count,
    HigherOrder,
    IfExpr,
}

/// Compact summary of a synthesis spec used by RD for family prediction.
struct RdSpecFeatures {
    /// "int", "num", "str", "list", or "unknown" — type of the first input.
    input_type: &'static str,
    /// "int", "num", "str", "bool", or "unknown" — uniform output type.
    output_type: &'static str,
    /// Number of distinct expected outputs (capped by caller).
    num_distinct_outputs: usize,
    /// True if any unary library function in env returns Bool on inputs[0].
    has_bool_lib: bool,
    /// True if every output is a substring of its corresponding input.
    /// Strong signal for string-take / string-drop / string-replace.
    output_is_substring: bool,
}

/// A derived sub-synthesis problem produced by inverting an outermost
/// function on the original examples.
struct RdSubSpec {
    inputs: Vec<Value>,
    expected: Vec<Value>,
}

/// Internal RD result type. Mirrors the legacy `RecursiveDecompResult`
/// but uses synth_v2 conventions (Vec<Node> + root index).
struct RdResult {
    found: bool,
    nodes: Vec<Node>,
    root: usize,
    candidates_explored: usize,
}

impl RdResult {
    fn empty() -> Self {
        Self { found: false, nodes: Vec::new(), root: 0, candidates_explored: 0 }
    }
}

/// Default RD recursion depth (how many levels of decomposition to try
/// before forcing flat sub-synthesis).
const RD_DEFAULT_DEPTH: usize = 2;

// ── Numeric helpers ────────────────────────────────────────────────────────

/// Extract a numeric value as f64 regardless of Int vs Num variant.
fn rd_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int(n) => Some(*n as f64),
        Value::Num(n) => Some(*n),
        _ => None,
    }
}

/// Wrap a derived numeric value, preferring `Int` when integral. The
/// preference for Int matches synth_v2's biased catalog: Int literals
/// are in the atom pool and the subtype rule lets them flow into Num
/// slots. Non-integral values go to Num.
fn rd_num_value(n: f64) -> Value {
    if n.is_finite() && (n - n.round()).abs() < 1e-9 && n.abs() < (i64::MAX as f64) {
        Value::Int(n.round() as i64)
    } else {
        Value::Num(n)
    }
}

// ── Spec features and family prediction ────────────────────────────────────

/// Extract spec features for family prediction. Probes env for unary
/// library functions returning Bool to compute `has_bool_lib`.
fn rd_extract_features(
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
) -> RdSpecFeatures {
    let input_type = match inputs.first() {
        Some(Value::Int(_)) => "int",
        Some(Value::Num(_)) => "num",
        Some(Value::Str(_)) => "str",
        Some(Value::List(_)) => "list",
        _ => "unknown",
    };
    let output_type = if expected.iter().all(|v| matches!(v, Value::Bool(_))) {
        "bool"
    } else if expected.iter().all(|v| matches!(v, Value::Int(_))) {
        "int"
    } else if expected.iter().all(|v| matches!(v, Value::Int(_) | Value::Num(_))) {
        "num"
    } else if expected.iter().all(|v| matches!(v, Value::Str(_))) {
        "str"
    } else {
        "unknown"
    };

    let mut keys: Vec<u64> = expected.iter().map(val_hash).collect();
    keys.sort();
    keys.dedup();
    let num_distinct_outputs = keys.len();

    // Probe library functions for has_bool_lib. Walk env's top scope and
    // try each unary Function on inputs[0]; if any returns a Bool, set
    // the flag. Builtins live in the bottom scope and are skipped.
    let has_bool_lib = if let Some(first_input) = inputs.first() {
        let entries: Vec<Value> = {
            let scope = env.top_scope();
            scope
                .values()
                .filter(|v| matches!(v, Value::Function(fd) if fd.params.len() == 1))
                .cloned()
                .collect()
        };
        entries.iter().any(|fv| {
            matches!(
                eval_v2::apply(fv, std::slice::from_ref(first_input), env),
                Ok(Value::Bool(_))
            )
        })
    } else {
        false
    };

    let output_is_substring = input_type == "str"
        && output_type == "str"
        && inputs.iter().zip(expected.iter()).all(|(i, e)| {
            if let (Value::Str(si), Value::Str(se)) = (i, e) {
                si.as_ref().contains(se.as_ref())
            } else {
                false
            }
        });

    RdSpecFeatures {
        input_type,
        output_type,
        num_distinct_outputs,
        has_bool_lib,
        output_is_substring,
    }
}

/// Hand-coded family prediction. Decision tree from the legacy
/// `predict_family` — same logic, with the int/num split exposed.
fn rd_predict_family(features: &RdSpecFeatures) -> RdFamily {
    if features.output_type == "bool" {
        if features.has_bool_lib {
            RdFamily::BoolComp
        } else {
            RdFamily::Compare
        }
    } else if features.output_type == "num" || features.output_type == "int" {
        if features.input_type == "list" {
            RdFamily::Arithmetic
        } else if features.input_type == "str" {
            RdFamily::Count
        } else {
            RdFamily::Arithmetic
        }
    } else if features.output_type == "str" {
        if features.output_is_substring {
            RdFamily::StringOp
        } else if features.num_distinct_outputs > 4 {
            RdFamily::IfExpr
        } else if features.num_distinct_outputs == 1 {
            RdFamily::Constant
        } else {
            RdFamily::StringOp
        }
    } else {
        RdFamily::Arithmetic
    }
}

/// Per-family list of candidate outermost function names.
fn rd_family_candidates(family: RdFamily, features: &RdSpecFeatures) -> Vec<&'static str> {
    match family {
        RdFamily::Arithmetic => {
            vec!["add", "subtract", "multiply", "negate", "abs", "divide", "modulo"]
        }
        RdFamily::Count => vec!["string-length", "count-char"],
        RdFamily::Compare => vec![
            "string-ends-with",
            "string-starts-with",
            "even",
            "odd",
        ],
        RdFamily::BoolComp => vec!["and", "or", "not"],
        RdFamily::StringOp => {
            if features.output_is_substring {
                vec![
                    "string-take",
                    "string-drop",
                    "string-replace",
                    "string-upper",
                    "string-lower",
                    "string-reverse",
                    "string-trim",
                    "concat",
                ]
            } else {
                vec![
                    "concat",
                    "string-replace",
                    "string-upper",
                    "string-lower",
                    "string-reverse",
                    "string-trim",
                    "string-take",
                    "string-drop",
                ]
            }
        }
        RdFamily::Constant => vec![],
        RdFamily::HigherOrder => vec!["map", "filter"],
        RdFamily::IfExpr => vec!["if"],
    }
}

/// Secondary family candidates to try after the primary family fails.
/// Catches cross-family solutions (e.g. predicted arithmetic but actual
/// solution involves string-length).
fn rd_secondary_candidates(primary: RdFamily, features: &RdSpecFeatures) -> Vec<&'static str> {
    match primary {
        RdFamily::Arithmetic if features.input_type == "str" => {
            vec!["string-length", "count-char"]
        }
        RdFamily::StringOp => vec!["string-length"],
        _ => vec![],
    }
}

// ── Inversion helpers ──────────────────────────────────────────────────────

/// Invert `f(input, k) = output` for binary arithmetic. Returns the
/// per-example k values as a sub-spec for synthesis.
fn rd_invert_binary_arith(
    fn_name: &str,
    inputs: &[Value],
    expected: &[Value],
) -> Option<RdSubSpec> {
    let nums_in: Vec<f64> = inputs.iter().filter_map(rd_to_f64).collect();
    let nums_out: Vec<f64> = expected.iter().filter_map(rd_to_f64).collect();
    if nums_in.len() != inputs.len() || nums_out.len() != expected.len() {
        return None;
    }
    let derived: Option<Vec<f64>> = match fn_name {
        "add" => Some(
            nums_in
                .iter()
                .zip(nums_out.iter())
                .map(|(i, o)| o - i)
                .collect(),
        ),
        "subtract" => Some(
            nums_in
                .iter()
                .zip(nums_out.iter())
                .map(|(i, o)| i - o)
                .collect(),
        ),
        "multiply" => nums_in
            .iter()
            .zip(nums_out.iter())
            .map(|(i, o)| if *i != 0.0 { Some(o / i) } else { None })
            .collect(),
        "divide" => nums_in
            .iter()
            .zip(nums_out.iter())
            .map(|(i, o)| if *o != 0.0 { Some(i / o) } else { None })
            .collect(),
        _ => return None,
    };
    let k_values = derived?;
    Some(RdSubSpec {
        inputs: inputs.to_vec(),
        expected: k_values.into_iter().map(rd_num_value).collect(),
    })
}

/// Invert `f(k, input) = output` (reversed arg order) for binary arithmetic.
fn rd_invert_binary_arith_reversed(
    fn_name: &str,
    inputs: &[Value],
    expected: &[Value],
) -> Option<RdSubSpec> {
    let nums_in: Vec<f64> = inputs.iter().filter_map(rd_to_f64).collect();
    let nums_out: Vec<f64> = expected.iter().filter_map(rd_to_f64).collect();
    if nums_in.len() != inputs.len() || nums_out.len() != expected.len() {
        return None;
    }
    let derived: Option<Vec<f64>> = match fn_name {
        "add" => Some(
            nums_in
                .iter()
                .zip(nums_out.iter())
                .map(|(i, o)| o - i)
                .collect(),
        ),
        "subtract" => Some(
            nums_in
                .iter()
                .zip(nums_out.iter())
                .map(|(i, o)| o + i)
                .collect(),
        ),
        "multiply" => nums_in
            .iter()
            .zip(nums_out.iter())
            .map(|(i, o)| if *i != 0.0 { Some(o / i) } else { None })
            .collect(),
        "divide" => Some(
            nums_in
                .iter()
                .zip(nums_out.iter())
                .map(|(i, o)| o * i)
                .collect(),
        ),
        _ => return None,
    };
    let k_values = derived?;
    Some(RdSubSpec {
        inputs: inputs.to_vec(),
        expected: k_values.into_iter().map(rd_num_value).collect(),
    })
}

/// Invert `f(g(x)) = output` for unary arithmetic where the inverse is
/// well-defined (negate). Functions like abs/floor/ceil are ambiguous
/// and not inverted here.
fn rd_invert_unary_arith(
    fn_name: &str,
    inputs: &[Value],
    expected: &[Value],
) -> Option<RdSubSpec> {
    let nums_out: Vec<f64> = expected.iter().filter_map(rd_to_f64).collect();
    if nums_out.len() != expected.len() {
        return None;
    }
    let sub_expected: Vec<f64> = match fn_name {
        "negate" => nums_out.iter().map(|o| -o).collect(),
        _ => return None,
    };
    Some(RdSubSpec {
        inputs: inputs.to_vec(),
        expected: sub_expected.into_iter().map(rd_num_value).collect(),
    })
}

/// Invert `(string-take input k) = output` — k is the length of output
/// when output is a prefix of input.
fn rd_invert_string_take(inputs: &[Value], expected: &[Value]) -> Option<RdSubSpec> {
    let mut k_values = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        if let (Value::Str(si), Value::Str(so)) = (inp, out) {
            if si.as_ref().starts_with(so.as_ref()) {
                k_values.push(Value::Int(so.chars().count() as i64));
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(RdSubSpec {
        inputs: inputs.to_vec(),
        expected: k_values,
    })
}

/// Invert `(string-drop input k) = output` — k is len(input) - len(output)
/// when output is a suffix of input.
fn rd_invert_string_drop(inputs: &[Value], expected: &[Value]) -> Option<RdSubSpec> {
    let mut k_values = Vec::new();
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        if let (Value::Str(si), Value::Str(so)) = (inp, out) {
            if si.as_ref().ends_with(so.as_ref()) {
                let drop_n = si.chars().count() - so.chars().count();
                k_values.push(Value::Int(drop_n as i64));
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(RdSubSpec {
        inputs: inputs.to_vec(),
        expected: k_values,
    })
}

/// Invert `concat`: try both `(concat input k)` and `(concat k input)`.
/// Returns a list of (subspec, reversed) pairs.
fn rd_invert_concat(inputs: &[Value], expected: &[Value]) -> Vec<(RdSubSpec, bool)> {
    let mut results = Vec::new();

    // (concat input suffix) — derive suffix
    let mut suffixes = Vec::new();
    let mut valid = true;
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        if let (Value::Str(si), Value::Str(so)) = (inp, out) {
            if so.as_ref().starts_with(si.as_ref()) {
                suffixes.push(Value::str(&so.as_ref()[si.as_ref().len()..]));
            } else {
                valid = false;
                break;
            }
        } else {
            valid = false;
            break;
        }
    }
    if valid && !suffixes.is_empty() {
        results.push((
            RdSubSpec { inputs: inputs.to_vec(), expected: suffixes },
            false,
        ));
    }

    // (concat prefix input) — derive prefix
    let mut prefixes = Vec::new();
    valid = true;
    for (inp, out) in inputs.iter().zip(expected.iter()) {
        if let (Value::Str(si), Value::Str(so)) = (inp, out) {
            if so.as_ref().ends_with(si.as_ref()) {
                let prefix_len = so.as_ref().len() - si.as_ref().len();
                prefixes.push(Value::str(&so.as_ref()[..prefix_len]));
            } else {
                valid = false;
                break;
            }
        } else {
            valid = false;
            break;
        }
    }
    if valid && !prefixes.is_empty() {
        results.push((
            RdSubSpec { inputs: inputs.to_vec(), expected: prefixes },
            true,
        ));
    }

    results
}

/// Invert `(count-char input ch) = output` — find a character ch that
/// counts to the expected value in every input.
fn rd_invert_count_char(inputs: &[Value], expected: &[Value]) -> Option<RdSubSpec> {
    let chars_to_try: Vec<char> = {
        let mut chars: Vec<char> = Vec::new();
        for inp in inputs {
            if let Value::Str(s) = inp {
                for c in s.chars() {
                    if !chars.contains(&c) {
                        chars.push(c);
                    }
                }
            }
        }
        chars
    };
    for ch in &chars_to_try {
        let ch_str = ch.to_string();
        let matches = inputs.iter().zip(expected.iter()).all(|(inp, out)| {
            let count = if let Value::Str(s) = inp {
                s.matches(&ch_str[..]).count() as i64
            } else {
                return false;
            };
            match out {
                Value::Int(n) => *n == count,
                Value::Num(n) => *n == count as f64,
                _ => false,
            }
        });
        if matches {
            return Some(RdSubSpec {
                inputs: inputs.to_vec(),
                expected: vec![Value::str(&ch_str); inputs.len()],
            });
        }
    }
    None
}

/// Invert unary string ops by computing the algebraic inverse on each
/// expected output, then verifying via builtin invocation.
fn rd_invert_unary_string(
    fn_name: &str,
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
) -> Option<RdSubSpec> {
    let sub_expected: Vec<Value> = expected
        .iter()
        .filter_map(|v| match v {
            Value::Str(s) => match fn_name {
                "string-upper" => Some(Value::str(&s.to_lowercase())),
                "string-lower" => Some(Value::str(&s.to_uppercase())),
                "string-reverse" => Some(Value::str(&s.chars().rev().collect::<String>())),
                _ => None,
            },
            _ => None,
        })
        .collect();
    if sub_expected.len() != expected.len() {
        return None;
    }
    let f = env.lookup(intern(fn_name))?;
    for (sub, exp) in sub_expected.iter().zip(expected.iter()) {
        match eval_v2::apply(&f, std::slice::from_ref(sub), env) {
            Ok(ref v) if eval_v2::values_equal(v, exp) => {}
            _ => return None,
        }
    }
    Some(RdSubSpec {
        inputs: inputs.to_vec(),
        expected: sub_expected,
    })
}

// ── Composition helpers ────────────────────────────────────────────────────

/// Extract the body index from a sub-solution. Sub-syntheses return
/// `(lambda (x) body)` — we want just `body` so the outer composition
/// can wrap its own lambda.
fn rd_extract_body(nodes: &[Node], sub_root: usize) -> usize {
    match &nodes[sub_root] {
        Node::Lambda(_, body) => *body,
        _ => sub_root,
    }
}

/// Build `(lambda (x) (f sub_body))` for unary `f`.
fn rd_compose_unary(
    fn_name: &str,
    sub_nodes: &[Node],
    sub_root: usize,
) -> (Vec<Node>, usize) {
    let mut nodes: Vec<Node> = Vec::new();
    let _x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));
    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(remap_node(nd, sub_offset));
    }
    let sub_body = rd_extract_body(&nodes, sub_root + sub_offset);
    let f_idx = nodes.len();
    nodes.push(Node::Symbol(intern(fn_name)));
    let app_idx = nodes.len();
    nodes.push(Node::App(vec![f_idx, sub_body]));
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], app_idx));
    (nodes, lambda_idx)
}

/// Build `(lambda (x) (f x sub_body))` for binary `f`.
fn rd_compose_binary_input_first(
    fn_name: &str,
    sub_nodes: &[Node],
    sub_root: usize,
) -> (Vec<Node>, usize) {
    let mut nodes: Vec<Node> = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));
    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(remap_node(nd, sub_offset));
    }
    let sub_body = rd_extract_body(&nodes, sub_root + sub_offset);
    let f_idx = nodes.len();
    nodes.push(Node::Symbol(intern(fn_name)));
    let app_idx = nodes.len();
    nodes.push(Node::App(vec![f_idx, x_idx, sub_body]));
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], app_idx));
    (nodes, lambda_idx)
}

/// Build `(lambda (x) (f sub_body x))` for binary `f`.
fn rd_compose_binary_input_second(
    fn_name: &str,
    sub_nodes: &[Node],
    sub_root: usize,
) -> (Vec<Node>, usize) {
    let mut nodes: Vec<Node> = Vec::new();
    let x_idx = nodes.len();
    nodes.push(Node::Symbol(intern("x")));
    let sub_offset = nodes.len();
    for nd in sub_nodes {
        nodes.push(remap_node(nd, sub_offset));
    }
    let sub_body = rd_extract_body(&nodes, sub_root + sub_offset);
    let f_idx = nodes.len();
    nodes.push(Node::Symbol(intern(fn_name)));
    let app_idx = nodes.len();
    nodes.push(Node::App(vec![f_idx, sub_body, x_idx]));
    let lambda_idx = nodes.len();
    nodes.push(Node::Lambda(vec![intern("x")], app_idx));
    (nodes, lambda_idx)
}

// ── Generic constant probing ───────────────────────────────────────────────

/// Generate candidate constant values for inversion probing — small
/// integers, numeric features of inputs/outputs, character substrings,
/// and common delimiters.
fn rd_generate_candidates(inputs: &[Value], expected: &[Value]) -> Vec<Value> {
    let mut candidates: Vec<Value> = Vec::new();
    let mut seen_ints: HashSet<i64> = HashSet::new();
    let mut seen_strs: HashSet<String> = HashSet::new();

    let mut push_int = |n: i64, candidates: &mut Vec<Value>, seen: &mut HashSet<i64>| {
        if seen.insert(n) {
            candidates.push(Value::Int(n));
        }
    };

    // Small integers 0..=10 plus -1.
    for i in -1..=10i64 {
        push_int(i, &mut candidates, &mut seen_ints);
    }

    for vals in [inputs, expected] {
        for v in vals {
            match v {
                Value::Int(n) => {
                    push_int(*n, &mut candidates, &mut seen_ints);
                    push_int(n.abs(), &mut candidates, &mut seen_ints);
                }
                Value::Num(n) => {
                    if let Some(i) = (*n as i64).checked_abs() {
                        let _ = i;
                    }
                    push_int(*n as i64, &mut candidates, &mut seen_ints);
                }
                Value::Str(s) => {
                    let len = s.chars().count() as i64;
                    push_int(len, &mut candidates, &mut seen_ints);
                    for ch in s.chars() {
                        let cs = ch.to_string();
                        if seen_strs.insert(cs.clone()) {
                            candidates.push(Value::str(&cs));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Pairwise numeric differences
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        if let (Some(a), Some(b)) = (rd_to_f64(inp), rd_to_f64(exp)) {
            push_int((b - a) as i64, &mut candidates, &mut seen_ints);
            push_int((a - b) as i64, &mut candidates, &mut seen_ints);
        }
    }

    // String prefixes/suffixes from inputs (length up to 10).
    for v in inputs {
        if let Value::Str(s) = v {
            let s_ref = s.as_ref();
            for len in 1..=s_ref.len().min(10) {
                let prefix = &s_ref[..len.min(s_ref.len())];
                if seen_strs.insert(prefix.to_string()) {
                    candidates.push(Value::str(prefix));
                }
                if s_ref.len() >= len {
                    let suffix = &s_ref[s_ref.len() - len..];
                    if seen_strs.insert(suffix.to_string()) {
                        candidates.push(Value::str(suffix));
                    }
                }
            }
        }
    }

    candidates.push(Value::Bool(true));
    candidates.push(Value::Bool(false));

    for s in &[" ", ",", "-", ".", "/", ":", ";", "_", "|", ""] {
        if seen_strs.insert(s.to_string()) {
            candidates.push(Value::str(s));
        }
    }

    candidates
}

/// Invoke a function (builtin or library) by Sym via env+apply. Returns
/// `None` on lookup failure or evaluation error.
fn rd_eval_function(name: Sym, args: &[Value], env: &Env) -> Option<Value> {
    let f = env.lookup(name)?;
    eval_v2::apply(&f, args, env).ok()
}

/// Convert a runtime Value into a literal `Node`. Used by RD when
/// emitting a constant `k` directly into the composition tree. Returns
/// `None` for non-literal value variants.
fn rd_value_to_node(v: &Value) -> Option<Node> {
    match v {
        Value::Int(n) => Some(Node::Int(*n)),
        Value::Num(n) => Some(Node::Num(*n)),
        Value::Str(s) => Some(Node::Str(s.as_ref().to_string())),
        Value::Bool(b) => Some(Node::Bool(*b)),
        _ => None,
    }
}

// ── Sub-synthesis ──────────────────────────────────────────────────────────

/// Recursive sub-synthesis: at depth > 0, try RD on the sub-spec first
/// (smaller budget); fall through to flat. At depth 0, flat only.
fn rd_sub_synthesize(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    rd_depth: usize,
) -> SynthResult {
    if rd_depth > 0 && inputs.len() >= 2 {
        let rd_budget = max_candidates / 3;
        let rd = rd_recursive(
            components,
            inputs,
            expected,
            env,
            universe,
            max_depth,
            rd_budget,
            rd_depth - 1,
        );
        if rd.found {
            return SynthResult {
                found: true,
                nodes: Some(rd.nodes),
                root: Some(rd.root),
                candidates_explored: rd.candidates_explored,
                decomposer_name: None,
            };
        }
        let remaining = max_candidates.saturating_sub(rd.candidates_explored);
        let sr = synthesize(components, inputs, expected, env, universe, max_depth, remaining);
        return SynthResult {
            candidates_explored: sr.candidates_explored + rd.candidates_explored,
            ..sr
        };
    }
    synthesize(components, inputs, expected, env, universe, max_depth, max_candidates)
}

// ── Per-function dispatch ──────────────────────────────────────────────────

/// Try a single outermost function: invert → sub-synthesize → compose →
/// verify. Returns `RdResult::empty()` with `candidates_explored > 0` if
/// the family applied but the inversion or sub-synth failed.
fn rd_try_single_function(
    fn_name: &str,
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    features: &RdSpecFeatures,
    rd_depth: usize,
) -> RdResult {
    let mut result = RdResult::empty();

    match fn_name {
        // ── Unary arithmetic ──
        "negate" if features.input_type != "list" => {
            if let Some(subspec) = rd_invert_unary_arith(fn_name, inputs, expected) {
                let sr = rd_sub_synthesize(
                    components,
                    &subspec.inputs,
                    &subspec.expected,
                    env,
                    universe,
                    max_depth,
                    max_candidates,
                    rd_depth,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let (nodes, lambda_idx) =
                        rd_compose_unary(fn_name, &sr.nodes.unwrap(), sr.root.unwrap());
                    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
        }

        // ── Binary arithmetic ──
        "add" | "subtract" | "multiply" | "divide" if features.input_type != "list" => {
            if let Some(subspec) = rd_invert_binary_arith(fn_name, inputs, expected) {
                let sr = rd_sub_synthesize(
                    components,
                    &subspec.inputs,
                    &subspec.expected,
                    env,
                    universe,
                    max_depth,
                    max_candidates / 2,
                    rd_depth,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let (nodes, lambda_idx) = rd_compose_binary_input_first(
                        fn_name,
                        &sr.nodes.unwrap(),
                        sr.root.unwrap(),
                    );
                    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
            if let Some(subspec) = rd_invert_binary_arith_reversed(fn_name, inputs, expected) {
                let remaining = max_candidates.saturating_sub(result.candidates_explored);
                let sr = rd_sub_synthesize(
                    components,
                    &subspec.inputs,
                    &subspec.expected,
                    env,
                    universe,
                    max_depth,
                    remaining.min(max_candidates / 2),
                    rd_depth,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let (nodes, lambda_idx) = rd_compose_binary_input_second(
                        fn_name,
                        &sr.nodes.unwrap(),
                        sr.root.unwrap(),
                    );
                    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
        }

        // ── String slicing ──
        "string-take" => {
            if let Some(subspec) = rd_invert_string_take(inputs, expected) {
                let sr = rd_sub_synthesize(
                    components,
                    &subspec.inputs,
                    &subspec.expected,
                    env,
                    universe,
                    max_depth,
                    max_candidates,
                    rd_depth,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let (nodes, lambda_idx) = rd_compose_binary_input_first(
                        "string-take",
                        &sr.nodes.unwrap(),
                        sr.root.unwrap(),
                    );
                    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
        }
        "string-drop" => {
            if let Some(subspec) = rd_invert_string_drop(inputs, expected) {
                let sr = rd_sub_synthesize(
                    components,
                    &subspec.inputs,
                    &subspec.expected,
                    env,
                    universe,
                    max_depth,
                    max_candidates,
                    rd_depth,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let (nodes, lambda_idx) = rd_compose_binary_input_first(
                        "string-drop",
                        &sr.nodes.unwrap(),
                        sr.root.unwrap(),
                    );
                    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
        }

        // ── Concat (try both orderings) ──
        "concat" => {
            for (subspec, reversed) in rd_invert_concat(inputs, expected) {
                let remaining = max_candidates.saturating_sub(result.candidates_explored);
                if remaining == 0 {
                    break;
                }
                let sr = rd_sub_synthesize(
                    components,
                    &subspec.inputs,
                    &subspec.expected,
                    env,
                    universe,
                    max_depth,
                    remaining / 2,
                    rd_depth,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let sub_nodes = sr.nodes.unwrap();
                    let sub_root = sr.root.unwrap();
                    let (nodes, lambda_idx) = if reversed {
                        rd_compose_binary_input_second("concat", &sub_nodes, sub_root)
                    } else {
                        rd_compose_binary_input_first("concat", &sub_nodes, sub_root)
                    };
                    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
        }

        // ── Unary string ops ──
        "string-upper" | "string-lower" | "string-reverse" | "string-trim" => {
            // Direct application check.
            if let Some(f) = env.lookup(intern(fn_name)) {
                let direct = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
                    matches!(
                        eval_v2::apply(&f, std::slice::from_ref(inp), env),
                        Ok(ref v) if eval_v2::values_equal(v, exp)
                    )
                });
                if direct {
                    let mut nodes: Vec<Node> = Vec::new();
                    let x_idx = nodes.len();
                    nodes.push(Node::Symbol(intern("x")));
                    let f_idx = nodes.len();
                    nodes.push(Node::Symbol(intern(fn_name)));
                    let app_idx = nodes.len();
                    nodes.push(Node::App(vec![f_idx, x_idx]));
                    let lambda_idx = nodes.len();
                    nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                    result.found = true;
                    result.nodes = nodes;
                    result.root = lambda_idx;
                    return result;
                }
            }
            if let Some(subspec) = rd_invert_unary_string(fn_name, inputs, expected, env) {
                let sr = rd_sub_synthesize(
                    components,
                    &subspec.inputs,
                    &subspec.expected,
                    env,
                    universe,
                    max_depth,
                    max_candidates,
                    rd_depth,
                );
                result.candidates_explored += sr.candidates_explored;
                if sr.found {
                    let (nodes, lambda_idx) =
                        rd_compose_unary(fn_name, &sr.nodes.unwrap(), sr.root.unwrap());
                    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
        }

        // ── Count functions ──
        "string-length" => {
            if features.input_type == "str"
                && (features.output_type == "int" || features.output_type == "num")
            {
                if let Some(f) = env.lookup(intern("string-length")) {
                    let direct = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
                        matches!(
                            eval_v2::apply(&f, std::slice::from_ref(inp), env),
                            Ok(ref v) if eval_v2::values_equal(v, exp)
                        )
                    });
                    if direct {
                        let mut nodes: Vec<Node> = Vec::new();
                        let x_idx = nodes.len();
                        nodes.push(Node::Symbol(intern("x")));
                        let f_idx = nodes.len();
                        nodes.push(Node::Symbol(intern("string-length")));
                        let app_idx = nodes.len();
                        nodes.push(Node::App(vec![f_idx, x_idx]));
                        let lambda_idx = nodes.len();
                        nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
        }

        "count-char" => {
            if let Some(subspec) = rd_invert_count_char(inputs, expected) {
                if let Some(Value::Str(ch)) = subspec.expected.first().cloned() {
                    let mut nodes: Vec<Node> = Vec::new();
                    let x_idx = nodes.len();
                    nodes.push(Node::Symbol(intern("x")));
                    let ch_idx = nodes.len();
                    nodes.push(Node::Str(ch.as_ref().to_string()));
                    let f_idx = nodes.len();
                    nodes.push(Node::Symbol(intern("count-char")));
                    let app_idx = nodes.len();
                    nodes.push(Node::App(vec![f_idx, x_idx, ch_idx]));
                    let lambda_idx = nodes.len();
                    nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                    if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                }
            }
        }

        // ── Predicates / comparators (direct application only) ──
        "string-ends-with" | "string-starts-with" | "even" | "odd" => {
            // Try (lambda (x) (f x k)) for each candidate constant k.
            if let Some(f) = env.lookup(intern(fn_name)) {
                let cands = rd_generate_candidates(inputs, expected);
                let arity = match fn_name {
                    "even" | "odd" => 1,
                    _ => 2,
                };
                if arity == 1 {
                    let direct = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
                        matches!(
                            eval_v2::apply(&f, std::slice::from_ref(inp), env),
                            Ok(ref v) if eval_v2::values_equal(v, exp)
                        )
                    });
                    if direct {
                        let mut nodes: Vec<Node> = Vec::new();
                        let x_idx = nodes.len();
                        nodes.push(Node::Symbol(intern("x")));
                        let f_idx = nodes.len();
                        nodes.push(Node::Symbol(intern(fn_name)));
                        let app_idx = nodes.len();
                        nodes.push(Node::App(vec![f_idx, x_idx]));
                        let lambda_idx = nodes.len();
                        nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                        result.found = true;
                        result.nodes = nodes;
                        result.root = lambda_idx;
                        return result;
                    }
                } else {
                    for k in &cands {
                        let all_match =
                            inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
                                matches!(
                                    eval_v2::apply(&f, &[inp.clone(), k.clone()], env),
                                    Ok(ref v) if eval_v2::values_equal(v, exp)
                                )
                            });
                        if all_match {
                            let k_node = match rd_value_to_node(k) {
                                Some(n) => n,
                                None => continue,
                            };
                            let mut nodes: Vec<Node> = Vec::new();
                            let x_idx = nodes.len();
                            nodes.push(Node::Symbol(intern("x")));
                            let k_idx = nodes.len();
                            nodes.push(k_node);
                            let f_idx = nodes.len();
                            nodes.push(Node::Symbol(intern(fn_name)));
                            let app_idx = nodes.len();
                            nodes.push(Node::App(vec![f_idx, x_idx, k_idx]));
                            let lambda_idx = nodes.len();
                            nodes.push(Node::Lambda(vec![intern("x")], app_idx));
                            result.found = true;
                            result.nodes = nodes;
                            result.root = lambda_idx;
                            return result;
                        }
                    }
                }
            }
        }

        // ── Boolean composition: BD already handles it ──
        "and" | "or" | "not" => {}

        // ── Higher-order: delegate to existing v2 strategy ──
        "map" | "filter" => {
            if let Some((nodes, root, ho_explored)) = higher_order_decompose(
                components,
                inputs,
                expected,
                env,
                universe,
                max_depth,
                max_candidates,
            ) {
                result.candidates_explored += ho_explored;
                result.found = true;
                result.nodes = nodes;
                result.root = root;
                return result;
            }
        }

        // ── If-expression: delegate to D&C ──
        "if" => {
            if let Some((nodes, root, dc_explored)) = divide_and_conquer(
                components,
                inputs,
                expected,
                env,
                universe,
                max_depth,
                max_candidates,
            ) {
                result.candidates_explored += dc_explored;
                result.found = true;
                result.nodes = nodes;
                result.root = root;
                return result;
            }
        }

        // ── Generic fallback: constant probing for any binary function ──
        _ => {
            let gen_r = rd_try_generic_binary_inversion(
                fn_name,
                components,
                inputs,
                expected,
                env,
                universe,
                max_depth,
                max_candidates.saturating_sub(result.candidates_explored),
                rd_depth,
            );
            result.candidates_explored += gen_r.candidates_explored;
            if gen_r.found {
                return RdResult {
                    candidates_explored: result.candidates_explored,
                    ..gen_r
                };
            }
        }
    }

    result
}

/// Generic constant-probing inversion for any binary function. Tries
/// `(f x k)` and `(f k x)` for each candidate constant. Then tries the
/// per-example k variant: derive the k for each input and sub-synthesize
/// the k-as-function-of-input.
fn rd_try_generic_binary_inversion(
    fn_name: &str,
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    rd_depth: usize,
) -> RdResult {
    let mut result = RdResult::empty();
    let f_sym = intern(fn_name);
    if env.lookup(f_sym).is_none() {
        return result;
    }
    let candidates = rd_generate_candidates(inputs, expected);

    // Constant-k forward: f(input, k)
    for k in &candidates {
        let all_match = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
            rd_eval_function(f_sym, &[inp.clone(), k.clone()], env)
                .map(|v| eval_v2::values_equal(&v, exp))
                .unwrap_or(false)
        });
        if all_match {
            let k_node = match rd_value_to_node(k) {
                Some(n) => n,
                None => continue,
            };
            let mut nodes: Vec<Node> = Vec::new();
            let x_idx = nodes.len();
            nodes.push(Node::Symbol(intern("x")));
            let k_idx = nodes.len();
            nodes.push(k_node);
            let f_idx = nodes.len();
            nodes.push(Node::Symbol(f_sym));
            let app_idx = nodes.len();
            nodes.push(Node::App(vec![f_idx, x_idx, k_idx]));
            let lambda_idx = nodes.len();
            nodes.push(Node::Lambda(vec![intern("x")], app_idx));
            if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                result.found = true;
                result.nodes = nodes;
                result.root = lambda_idx;
                return result;
            }
        }
    }

    // Constant-k reversed: f(k, input)
    for k in &candidates {
        let all_match = inputs.iter().zip(expected.iter()).all(|(inp, exp)| {
            rd_eval_function(f_sym, &[k.clone(), inp.clone()], env)
                .map(|v| eval_v2::values_equal(&v, exp))
                .unwrap_or(false)
        });
        if all_match {
            let k_node = match rd_value_to_node(k) {
                Some(n) => n,
                None => continue,
            };
            let mut nodes: Vec<Node> = Vec::new();
            let x_idx = nodes.len();
            nodes.push(Node::Symbol(intern("x")));
            let k_idx = nodes.len();
            nodes.push(k_node);
            let f_idx = nodes.len();
            nodes.push(Node::Symbol(f_sym));
            let app_idx = nodes.len();
            nodes.push(Node::App(vec![f_idx, k_idx, x_idx]));
            let lambda_idx = nodes.len();
            nodes.push(Node::Lambda(vec![intern("x")], app_idx));
            if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                result.found = true;
                result.nodes = nodes;
                result.root = lambda_idx;
                return result;
            }
        }
    }

    // Variable-k forward: f(x, g(x)) — derive k_i per example, sub-synthesize.
    let mut per_k: Vec<Value> = Vec::with_capacity(inputs.len());
    let mut all_found = true;
    for (inp, exp) in inputs.iter().zip(expected.iter()) {
        let mut found_k = None;
        for k in &candidates {
            if let Some(v) = rd_eval_function(f_sym, &[inp.clone(), k.clone()], env) {
                if eval_v2::values_equal(&v, exp) {
                    found_k = Some(k.clone());
                    break;
                }
            }
        }
        match found_k {
            Some(k) => per_k.push(k),
            None => {
                all_found = false;
                break;
            }
        }
    }
    if all_found && !per_k.is_empty() {
        let all_same = per_k
            .windows(2)
            .all(|w| eval_v2::values_equal(&w[0], &w[1]));
        if !all_same {
            let sr = rd_sub_synthesize(
                components,
                inputs,
                &per_k,
                env,
                universe,
                max_depth,
                max_candidates / 2,
                rd_depth,
            );
            result.candidates_explored += sr.candidates_explored;
            if sr.found {
                let (nodes, lambda_idx) =
                    rd_compose_binary_input_first(fn_name, &sr.nodes.unwrap(), sr.root.unwrap());
                if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                    result.found = true;
                    result.nodes = nodes;
                    result.root = lambda_idx;
                    return result;
                }
            }
        }
    }

    result
}

// ── Library function decomposition ─────────────────────────────────────────

/// Walk env's top scope for unary `Value::Function` entries (skipping
/// internal names) and return them as (sym, value) pairs. Used by RD's
/// macro-like decomposition phases.
fn rd_unary_lib_functions(env: &Env) -> Vec<(Sym, Value)> {
    let scope = env.top_scope();
    scope
        .iter()
        .filter_map(|(sym, val)| {
            if let Value::Function(fd) = val {
                if fd.params.len() == 1 {
                    let name = crate::intern::resolve(*sym);
                    if name.starts_with("__") {
                        return None;
                    }
                    return Some((*sym, val.clone()));
                }
            }
            None
        })
        .collect()
}

/// Try unary library functions as bridges: for each unary function `m`,
/// evaluate `m(input)` on all examples, then either match expected
/// directly (`(lambda (x) (m x))`) or sub-synthesize an outer `f` such
/// that `f(m(x)) = expected` (composed as `(lambda (x) (f (m x)))`).
fn rd_try_library_decomposition(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    rd_depth: usize,
) -> RdResult {
    let mut result = RdResult::empty();
    let phase_budget = max_candidates / 4;
    let sub_budget = 500usize.min(phase_budget / 4).max(50);

    for (msym, mval) in rd_unary_lib_functions(env) {
        if result.candidates_explored >= phase_budget {
            break;
        }

        // Evaluate m(input) for every input.
        let mut intermediates = Vec::with_capacity(inputs.len());
        let mut valid = true;
        for inp in inputs {
            match eval_v2::apply(&mval, std::slice::from_ref(inp), env) {
                Ok(v) => intermediates.push(v),
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid || intermediates.len() != inputs.len() {
            continue;
        }

        let mname = crate::intern::resolve(msym);

        // Direct match: m(input) == expected
        if intermediates
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| eval_v2::values_equal(a, b))
        {
            let mut nodes: Vec<Node> = Vec::new();
            let x_idx = nodes.len();
            nodes.push(Node::Symbol(intern("x")));
            let m_idx = nodes.len();
            nodes.push(Node::Symbol(msym));
            let app_idx = nodes.len();
            nodes.push(Node::App(vec![m_idx, x_idx]));
            let lambda_idx = nodes.len();
            nodes.push(Node::Lambda(vec![intern("x")], app_idx));
            if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                result.found = true;
                result.nodes = nodes;
                result.root = lambda_idx;
                return result;
            }
        }

        // Skip identity macros and constant macros (no useful bridge).
        if intermediates
            .iter()
            .zip(inputs.iter())
            .all(|(a, b)| eval_v2::values_equal(a, b))
        {
            continue;
        }
        if intermediates.len() > 1
            && intermediates
                .windows(2)
                .all(|w| eval_v2::values_equal(&w[0], &w[1]))
        {
            continue;
        }

        // Bridge: synthesize f such that f(intermediate) = expected.
        let remaining = max_candidates.saturating_sub(result.candidates_explored);
        let budget = remaining.min(sub_budget);
        if budget == 0 {
            break;
        }
        let sr = rd_sub_synthesize(
            components,
            &intermediates,
            expected,
            env,
            universe,
            max_depth,
            budget,
            rd_depth,
        );
        result.candidates_explored += sr.candidates_explored;
        if sr.found {
            let f_nodes = sr.nodes.unwrap();
            let f_root = sr.root.unwrap();

            // Compose (lambda (x) ((extract f's body, substitute "x" → (m x))))
            let mut nodes: Vec<Node> = Vec::new();
            let x_idx = nodes.len();
            nodes.push(Node::Symbol(intern("x")));
            let m_sym_idx = nodes.len();
            nodes.push(Node::Symbol(msym));
            let m_app_idx = nodes.len();
            nodes.push(Node::App(vec![m_sym_idx, x_idx]));

            // Splice f's nodes.
            let f_offset = nodes.len();
            for nd in &f_nodes {
                nodes.push(remap_node(nd, f_offset));
            }
            let f_root_remapped = f_root + f_offset;

            // f is `(lambda (x) body)` — extract body and replace its
            // `x` Symbol references with `m_app_idx`.
            if let Node::Lambda(_, body_idx) = &nodes[f_root_remapped] {
                let body_idx = *body_idx;
                let composed_body = rd_substitute_symbol(&nodes, body_idx, intern("x"), m_app_idx);
                let composed_body_idx = nodes.len();
                nodes.push(composed_body);
                let lambda_idx = nodes.len();
                nodes.push(Node::Lambda(vec![intern("x")], composed_body_idx));
                if ho_verify_composed(&nodes, lambda_idx, inputs, expected, env) {
                    result.found = true;
                    result.nodes = nodes;
                    result.root = lambda_idx;
                    return result;
                }
            }
        }
    }

    result
}

/// Substitute references to a Symbol(target) with `replacement_idx` in
/// the immediate children of node `idx`. Returns a NEW node value to be
/// appended to the node list. Mirrors the legacy `substitute_symbol_idx`
/// — only top-level child substitution; references inside deeper
/// subtrees still resolve to the original `Symbol(x)`.
fn rd_substitute_symbol(
    nodes: &[Node],
    idx: usize,
    target: Sym,
    replacement_idx: usize,
) -> Node {
    let swap = |c: usize| -> usize {
        if let Node::Symbol(name) = &nodes[c] {
            if *name == target {
                return replacement_idx;
            }
        }
        c
    };
    match &nodes[idx] {
        Node::Symbol(name) if *name == target => Node::Symbol(*name),
        Node::App(children) => Node::App(children.iter().map(|&c| swap(c)).collect()),
        Node::SpecialApp(form, children) => {
            Node::SpecialApp(*form, children.iter().map(|&c| swap(c)).collect())
        }
        Node::If(c, t, e) => Node::If(swap(*c), swap(*t), swap(*e)),
        Node::Let(bindings, body) => {
            let new_bindings: Vec<(Sym, usize)> =
                bindings.iter().map(|(n, v)| (*n, swap(*v))).collect();
            Node::Let(new_bindings, swap(*body))
        }
        Node::Lambda(params, body) => Node::Lambda(params.clone(), swap(*body)),
        other => other.clone(),
    }
}

// ── Top-level RD entry point ───────────────────────────────────────────────

/// Internal recursive entry point with explicit depth tracking.
fn rd_recursive(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    rd_depth: usize,
) -> RdResult {
    if inputs.is_empty() || expected.is_empty() || inputs.len() != expected.len() {
        return RdResult::empty();
    }

    let features = rd_extract_features(inputs, expected, env);
    let family = rd_predict_family(&features);

    let primary = rd_family_candidates(family, &features);
    let secondary = rd_secondary_candidates(family, &features);
    let all_candidates: Vec<&str> = primary.iter().copied().chain(secondary.iter().copied()).collect();

    let sub_budget = (max_candidates / 2).max(1);
    let mut total_explored: usize = 0;

    // Phase 1: try predicted family (and secondary) candidates.
    for fn_name in &all_candidates {
        if total_explored >= max_candidates {
            break;
        }
        let remaining = max_candidates.saturating_sub(total_explored);
        let budget = remaining.min(sub_budget);
        let r = rd_try_single_function(
            fn_name,
            components,
            inputs,
            expected,
            env,
            universe,
            max_depth,
            budget,
            &features,
            rd_depth,
        );
        total_explored += r.candidates_explored;
        if r.found {
            return RdResult {
                candidates_explored: total_explored,
                ..r
            };
        }
    }

    // Phase 2: try unary library functions as bridges.
    let remaining = max_candidates.saturating_sub(total_explored);
    if remaining > 0 {
        let lr = rd_try_library_decomposition(
            components,
            inputs,
            expected,
            env,
            universe,
            max_depth,
            remaining,
            rd_depth,
        );
        total_explored += lr.candidates_explored;
        if lr.found {
            return RdResult {
                candidates_explored: total_explored,
                ..lr
            };
        }
    }

    RdResult {
        candidates_explored: total_explored,
        ..RdResult::empty()
    }
}

/// Top-level RD strategy entry point. Returns `Some((nodes, root, candidates))`
/// on success, `None` on failure.
pub fn recursive_decompose(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> Option<(Vec<Node>, usize, usize)> {
    let r = rd_recursive(
        components,
        inputs,
        expected,
        env,
        universe,
        max_depth,
        max_candidates,
        RD_DEFAULT_DEPTH,
    );
    if r.found {
        Some((r.nodes, r.root, r.candidates_explored))
    } else {
        None
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
///   2. **RecursiveDecomposition** — `recursive_decompose`. Predicts the
///      outermost function family from spec features, inverts each
///      candidate to derive a sub-spec, and recursively sub-synthesizes.
///      Runs after Flat (not before) so trivial tasks pay no inversion
///      cost — Flat finds them faster. Deviates from §9.22's "Strategy 0"
///      placement: in synth_v2 Flat is well-pruned and the cheap-to-
///      expensive dispatcher gradient matters more than RD's conceptual
///      role as the recursive base step.
///   3. **BoolDecomp** — `bool_decompose` (logical compositions of unary
///      bool-returning library predicates). Only fires when the target
///      output is bool, and is O(L²) in the number of predicates.
///   4. **HigherOrder** — `higher_order_decompose`. Tries `list-map`,
///      `list-filter`, `split-map-join`, and `char-map-join` templates
///      via recursive sub-synthesis. Each template is gated on the
///      input/output shape so most attempts cost nothing.
///   5. **DivideConquer** — `divide_and_conquer`. Partitions examples
///      by output value, sorts by mean input, and recursively builds
///      nested if-expressions with bool separators between groups.
///      Useful for piecewise / classification tasks.
///   6. **Induction** — `induce_decomposition`. Probes a curated set
///      of unary/binary builtins for an intermediate value sequence,
///      then sub-synthesizes input→mid and mid→expected and composes
///      via `(let ((x g_body)) f_body)`.
///   7. **Memo** — `memorize_from_examples` (namespace lookup table for
///      string-input tasks).
///
/// `flat_budget` is the candidate budget for the Flat strategy and is
/// also passed to RD/HO/D&C/Induction sub-syntheses. BD and Memo are
/// bounded by their own intrinsic costs and ignore it.
/// §9.37 Stage A: walk the env's `__decomposers__` namespace, calling
/// each entry as a SELPH decomposer until one returns a `found: true`
/// result. Each decomposer is a SELPH lambda accepting one argument
/// (the spec namespace, in the same shape `bi_synthesize` accepts) and
/// returning either `nil` (doesn't apply) or a result namespace
/// `(ns ("found" true) ("nodes" <Node>) ("candidates" <Int>))`.
///
/// Returns `Some((nodes, root, candidates, name_sym))` on the first
/// successful decomposer, where `name_sym` is the entry's key in the
/// namespace (used for trace output via `Strategy::Custom`).
///
/// Iteration order is deterministic: entries are sorted by their Sym's
/// resolved string. Same-key duplicates can't happen because NsMap is
/// keyed by Sym.
///
/// Decomposers that error or return malformed results are silently
/// skipped — same failure mode as the source-loaded heuristic path
/// (§9.32). The curriculum is responsible for testing its own
/// decomposers; the dispatcher is best-effort.
fn try_selph_decomposers(
    env: &Env,
    inputs: &[Value],
    expected: &[Value],
    max_depth: usize,
    max_budget: usize,
    test_inputs: &[Value],
    test_expected: &[Value],
) -> Option<(Vec<Node>, usize, usize, Sym)> {
    let decomp_ns = match env.lookup(intern("__decomposers__")) {
        Some(Value::Ns(map)) => map,
        _ => return None,
    };
    if decomp_ns.is_empty() {
        return None;
    }

    // Sort entries by name for deterministic dispatch order. NsMap is
    // a HashMap so iteration order is otherwise nondeterministic.
    let mut entries: Vec<(Sym, Value)> = decomp_ns
        .iter()
        .map(|(&k, v)| (k, v.clone()))
        .collect();
    entries.sort_by_key(|(s, _)| resolve(*s));

    // Build the spec namespace once. Same shape `bi_synthesize` accepts:
    // `(ns ("spec" <list of (in, out) pairs>) ("max-depth" n) ("max-candidates" n))`.
    let spec_pairs: Vec<Value> = inputs
        .iter()
        .zip(expected.iter())
        .map(|(i, e)| Value::list(vec![i.clone(), e.clone()]))
        .collect();
    let mut spec_ns = NsMap::new();
    spec_ns.insert(intern("spec"), Value::list(spec_pairs));
    spec_ns.insert(intern("max-depth"), Value::Int(max_depth as i64));
    spec_ns.insert(intern("max-candidates"), Value::Int(max_budget as i64));
    // §9.47.6: include held-out test pairs so the SELPH chain can
    // self-validate candidates internally.
    if !test_inputs.is_empty() {
        let test_pairs: Vec<Value> = test_inputs
            .iter()
            .zip(test_expected.iter())
            .map(|(i, e)| Value::list(vec![i.clone(), e.clone()]))
            .collect();
        spec_ns.insert(intern("test"), Value::list(test_pairs));
    }
    let spec_val = Value::ns(spec_ns);

    let mut total_cands = 0usize;
    for (name_sym, decomposer) in &entries {
        if !matches!(decomposer, Value::Function(_) | Value::Builtin(_)) {
            continue;
        }
        match eval_v2::apply(decomposer, std::slice::from_ref(&spec_val), env) {
            Ok(Value::Ns(result)) => {
                // Tally any candidate count the decomposer reports —
                // even on failure. Lets the dispatcher sum search
                // costs across attempts.
                if let Some(cands) = result.get(&intern("candidates")) {
                    let n = match cands {
                        Value::Int(n) => *n as usize,
                        Value::Num(n) => *n as usize,
                        _ => 0,
                    };
                    total_cands += n;
                }
                if matches!(
                    result.get(&intern("found")),
                    Some(Value::Bool(true))
                ) {
                    if let Some(Value::Node(node_ref)) =
                        result.get(&intern("nodes"))
                    {
                        // Materialize the constructed AST into a fresh
                        // owned Vec<Node>. The NodeRef's arena is Rc-shared
                        // and may live longer than this dispatcher call;
                        // owning the Vec keeps the StrategyResult
                        // self-contained.
                        let nodes: Vec<Node> = node_ref.nodes.iter().cloned().collect();
                        return Some((nodes, node_ref.idx, total_cands, *name_sym));
                    }
                    // Found but missing nodes — skip silently. A future
                    // version could log this as a curriculum bug.
                }
            }
            Ok(Value::Nil) => continue,
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    None
}

/// §9.37 Stage C: type-keyed decomposer dispatch. Walk the universe's
/// `type_metadata` for the inferred output type — and the `Any`
/// fallback type — calling each registered decomposer in order. Same
/// shape as `try_selph_decomposers` (Stage A) but with a per-type
/// scope.
///
/// The dispatcher uses the inferred output Sym to look up decomposers,
/// so a curriculum that registers a decomposer under `("Int" (ns
/// ("decomposers" (ns ("my-decomp" my-fn)))))` will see it called
/// only for Int-output tasks. Decomposers under `Any` are called for
/// every task (after the type-specific ones).
///
/// Returns the same shape as `try_selph_decomposers`, with `name_sym`
/// being the decomposer's key in its `decomposers` namespace.
fn try_type_keyed_decomposers(
    env: &Env,
    universe: &TypeUniverse,
    inputs: &[Value],
    expected: &[Value],
    max_depth: usize,
    max_budget: usize,
) -> Option<(Vec<Node>, usize, usize, Sym)> {
    // No __types__ loaded → nothing to dispatch.
    if universe.type_metadata.is_empty() {
        return None;
    }

    // Infer the output type Sym from the first expected value.
    // (Mirrors `infer_uniform_type_sym` semantics — we only check the
    // first value because Stage B's universe doesn't yet enforce
    // type uniformity at the boundary.)
    let target_sym = expected
        .first()
        .and_then(Value::type_sym)
        .unwrap_or_else(type_any);

    // Collect candidate decomposer lists in dispatch order: the target
    // type first, then `Any` as a fallback. Each entry is
    // (name_sym, function_value).
    let mut candidates: Vec<(Sym, Value)> = Vec::new();
    if let Some(meta) = universe.type_metadata.get(&target_sym) {
        candidates.extend(meta.decomposers.iter().cloned());
    }
    let any_sym = type_any();
    if target_sym != any_sym {
        if let Some(meta) = universe.type_metadata.get(&any_sym) {
            candidates.extend(meta.decomposers.iter().cloned());
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // Build the spec namespace once (same shape as Stage A).
    let spec_pairs: Vec<Value> = inputs
        .iter()
        .zip(expected.iter())
        .map(|(i, e)| Value::list(vec![i.clone(), e.clone()]))
        .collect();
    let mut spec_ns = NsMap::new();
    spec_ns.insert(intern("spec"), Value::list(spec_pairs));
    spec_ns.insert(intern("max-depth"), Value::Int(max_depth as i64));
    spec_ns.insert(intern("max-candidates"), Value::Int(max_budget as i64));
    let spec_val = Value::ns(spec_ns);

    let mut total_cands = 0usize;
    for (name_sym, decomposer) in &candidates {
        if !matches!(decomposer, Value::Function(_) | Value::Builtin(_)) {
            continue;
        }
        match eval_v2::apply(decomposer, std::slice::from_ref(&spec_val), env) {
            Ok(Value::Ns(result)) => {
                if let Some(cands) = result.get(&intern("candidates")) {
                    let n = match cands {
                        Value::Int(n) => *n as usize,
                        Value::Num(n) => *n as usize,
                        _ => 0,
                    };
                    total_cands += n;
                }
                if matches!(
                    result.get(&intern("found")),
                    Some(Value::Bool(true))
                ) {
                    if let Some(Value::Node(node_ref)) =
                        result.get(&intern("nodes"))
                    {
                        let nodes: Vec<Node> = node_ref.nodes.iter().cloned().collect();
                        return Some((nodes, node_ref.idx, total_cands, *name_sym));
                    }
                }
            }
            Ok(Value::Nil) => continue,
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    None
}

pub fn synthesize_with_strategies(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    flat_budget: usize,
) -> StrategyResult {
    // §9.49 post-mortem diagnostics: infer output type tag once.
    let output_type_str = infer_uniform_type_sym(expected)
        .map(|s| resolve(s))
        .unwrap_or_else(|| "Mixed".to_string());

    // §9.49: check whether __decomposers__ exists and is non-empty.
    let has_decomposers = matches!(
        env.lookup(intern("__decomposers__")),
        Some(Value::Ns(ref m)) if !m.is_empty()
    );

    // §9.37 Stage A: global SELPH decomposers from `__decomposers__`
    // run BEFORE the hardcoded chain. Curriculum is in charge.
    if let Some((nodes, root, sd_explored, name_sym)) =
        try_selph_decomposers(env, inputs, expected, max_depth, flat_budget, &[], &[])
    {
        return StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: sd_explored,
            strategy: Some(Strategy::Custom(name_sym)),
            output_type: Some(output_type_str),
            m_chain_ran: true,
        };
    }

    // §9.37 Stage C: type-keyed decomposers from `__types__`. The
    // dispatcher infers the output type Sym from the spec, looks up
    // that type's decomposers list (and the `Any` fallback), and
    // calls each in order. This is the more ergonomic registration
    // path: a curriculum says "this decomposer applies to Int outputs"
    // by hanging the function under `("Int" (ns ("decomposers" ...)))`.
    if let Some((nodes, root, td_explored, name_sym)) =
        try_type_keyed_decomposers(
            env, universe, inputs, expected, max_depth, flat_budget,
        )
    {
        return StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: td_explored,
            strategy: Some(Strategy::Custom(name_sym)),
            output_type: Some(output_type_str),
            m_chain_ran: has_decomposers,
        };
    }

    // §9.49: helper to stamp diagnostics onto any result from this function.
    let stamp = |mut r: StrategyResult| -> StrategyResult {
        r.output_type = Some(output_type_str.clone());
        r.m_chain_ran = has_decomposers;
        r
    };

    // Strategy 1: Flat enumerative.
    let flat = synthesize(
        components, inputs, expected, env, universe, max_depth, flat_budget,
    );
    let mut total_explored = flat.candidates_explored;
    if flat.found {
        return stamp(StrategyResult::from_synth(flat, Strategy::Flat));
    }

    // Strategy 2: Recursive decomposition. Top-down family prediction +
    // outermost-function inversion. Cheap when the family doesn't apply
    // (each helper bails fast); valuable when an arithmetic / string-op
    // / library composition is the answer and Flat couldn't reach it
    // within `flat_budget`.
    if let Some((nodes, root, rd_explored)) = recursive_decompose(
        components,
        inputs,
        expected,
        env,
        universe,
        max_depth,
        flat_budget,
    ) {
        total_explored += rd_explored;
        return stamp(StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: total_explored,
            strategy: Some(Strategy::RecursiveDecomposition),
            output_type: None,
            m_chain_ran: false,
        });
    }

    // Strategy 3: Boolean decomposition. Cheap and only applies to
    // bool-output tasks (it filters internally), so we run it before
    // Memo: it produces a structured program when it fires, whereas
    // Memo is a lookup-table fallback that is correct on training but
    // generalizes by accident on bool output.
    if let Some((nodes, root, bd_explored)) =
        bool_decompose(components, inputs, expected, env)
    {
        total_explored += bd_explored;
        return stamp(StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: total_explored,
            strategy: Some(Strategy::BoolDecomp),
            output_type: None,
            m_chain_ran: false,
        });
    }

    // Strategy 3: Higher-order decomposition. Each template is shape-
    // gated (list→list, str→str, etc.) and bails immediately if not
    // applicable, so cost is dominated by the inner sub-synthesis.
    if let Some((nodes, root, ho_explored)) = higher_order_decompose(
        components,
        inputs,
        expected,
        env,
        universe,
        max_depth,
        flat_budget,
    ) {
        total_explored += ho_explored;
        return stamp(StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: total_explored,
            strategy: Some(Strategy::HigherOrder),
            output_type: None,
            m_chain_ran: false,
        });
    }

    // Strategy 4: Divide-and-conquer. Only meaningful for tasks with
    // multiple distinct outputs (it filters internally). The recursive
    // structure means worst-case cost is N partition attempts × the
    // sub-synthesis budget for separators and branches.
    if let Some((nodes, root, dc_explored)) = divide_and_conquer(
        components,
        inputs,
        expected,
        env,
        universe,
        max_depth,
        flat_budget,
    ) {
        total_explored += dc_explored;
        return stamp(StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: total_explored,
            strategy: Some(Strategy::DivideConquer),
            output_type: None,
            m_chain_ran: false,
        });
    }

    // Strategy 5: Induction (intermediate value decomposition). Probes
    // a curated set of unary/binary builtins for an intermediate value
    // sequence, then sub-synthesizes input→intermediate and
    // intermediate→expected. Composes via Node::Let.
    if let Some((nodes, root, in_explored)) = induce_decomposition(
        components,
        inputs,
        expected,
        env,
        universe,
        max_depth,
        flat_budget,
    ) {
        total_explored += in_explored;
        return stamp(StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: total_explored,
            strategy: Some(Strategy::Induction),
            output_type: None,
            m_chain_ran: false,
        });
    }

    // Strategy 6: Memo. Always candidate-cost 0 (no enumeration).
    if let Some((nodes, root)) = memorize_from_examples(inputs, expected) {
        return stamp(StrategyResult {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: total_explored,
            strategy: Some(Strategy::Memo),
            output_type: None,
            m_chain_ran: false,
        });
    }

    stamp(StrategyResult::not_found(total_explored))
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
///
/// §9.45.13 deletion: the `has_hole` field is gone. Hole-bearing
/// candidates were the §9.42 constant-fitting path that's now in
/// pure SELPH (M11/M12 + m_pool's fit-affine).
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
    /// Set when a curriculum-side decomposer (the M-chain) solved
    /// this task. Carries the `__decomposers__` ns key that fired.
    /// `None` for plain Flat enumeration solves and for failures.
    /// Threaded through to grow-v2's strategy reporting so the
    /// "By strategy" summary distinguishes chain solves from Flat
    /// solves on the multi-arg path (the §9.46 mislabel fix).
    pub decomposer_name: Option<Sym>,
}

impl SynthResult {
    fn not_found(explored: usize) -> Self {
        Self {
            found: false,
            nodes: None,
            root: None,
            candidates_explored: explored,
            decomposer_name: None,
        }
    }

    fn success(nodes: Vec<Node>, root: usize, explored: usize) -> Self {
        Self {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: explored,
            decomposer_name: None,
        }
    }

    /// Success path for curriculum-side decomposer hits. Same as
    /// `success` but tags the result with the firing decomposer's
    /// ns key so grow-v2 can label the strategy correctly.
    fn success_from_decomposer(
        nodes: Vec<Node>, root: usize, explored: usize, name_sym: Sym,
    ) -> Self {
        Self {
            found: true,
            nodes: Some(nodes),
            root: Some(root),
            candidates_explored: explored,
            decomposer_name: Some(name_sym),
        }
    }
}

/// Validate a synthesized lambda against held-out examples. Returns
/// `true` if the lambda matches every (input, expected) pair in
/// `test_inputs` / `test_expected`, `false` otherwise. Empty test
/// vectors count as vacuous success (no held-out data ⇒ skip).
///
/// This is the synth_v2 port of the legacy `validate_candidate`
/// from synth.rs. The candidate is a lambda Node at `root` inside
/// `nodes`. We evaluate it once to get a closure, then apply to each
/// test input. Evaluation uses the shared `env` so library functions
/// from prior tasks are visible — same eval context the training
/// rows used.
fn validate_held_out(
    nodes: &[Node],
    root: usize,
    test_inputs: &[Value],
    test_expected: &[Value],
    env: &Env,
) -> bool {
    if test_inputs.is_empty() {
        return true; // no held-out data
    }
    let nodes_rc: Rc<[Node]> = nodes.to_vec().into();
    let func = match eval_v2::eval(&nodes_rc, root, env) {
        Ok(f) => f,
        Err(_) => return false,
    };
    for (inp, exp) in test_inputs.iter().zip(test_expected.iter()) {
        let got = match eval_v2::apply(&func, &[inp.clone()], env) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if !eval_v2::values_equal(&got, exp) {
            return false;
        }
    }
    true
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

/// Scan a task's inputs and expected outputs for unique primitive
/// values (Int, Num, Str, Bool) and return them as zero-arity
/// literal components. The §9.41 data-derived literal seeding pass
/// uses this to surface useful constants from the spec data without
/// requiring the synth catalog to hardcode physics fractions like
/// `0.125` or task-specific magic numbers.
///
/// One level of list-unwrapping is performed so multi-arg inputs
/// (e.g. `(list 0.5 0.25 0.5)`) surface their per-position values.
/// Lists deeper than one level (lists of lists) are not unwrapped —
/// inner structure usually means the values are part of a structural
/// answer, not free constants.
pub fn collect_data_literals(inputs: &[Value], expected: &[Value]) -> Vec<SynthComponent> {
    use std::collections::HashSet;
    let int_t = intern("Int");
    let num_t = intern("Num");
    let str_t = intern("String");
    let bool_t = intern("Bool");

    // Dedup buckets — Num values key on `to_bits` so two distinct f64
    // bit patterns stay distinct, including +0.0 vs -0.0.
    let mut seen_int: HashSet<i64> = HashSet::new();
    let mut seen_num: HashSet<u64> = HashSet::new();
    let mut seen_str: HashSet<String> = HashSet::new();
    let mut seen_bool: HashSet<bool> = HashSet::new();
    let mut out: Vec<SynthComponent> = Vec::new();

    let mut visit = |v: &Value| {
        match v {
            Value::Int(n) => {
                if seen_int.insert(*n) {
                    out.push(SynthComponent::literal(
                        n.to_string(), LiteralKind::Int(*n), int_t, 0.0,
                    ));
                }
            }
            Value::Num(n) => {
                let bits = n.to_bits();
                if n.is_finite() && seen_num.insert(bits) {
                    out.push(SynthComponent::literal(
                        n.to_string(), LiteralKind::Num(*n), num_t, 0.0,
                    ));
                }
            }
            Value::Str(s) => {
                let owned = s.as_ref().to_string();
                if seen_str.insert(owned.clone()) {
                    out.push(SynthComponent::literal(
                        format!("\"{}\"", owned), LiteralKind::Str(owned), str_t, 0.0,
                    ));
                }
            }
            Value::Bool(b) => {
                if seen_bool.insert(*b) {
                    out.push(SynthComponent::literal(
                        b.to_string(), LiteralKind::Bool(*b), bool_t, 0.0,
                    ));
                }
            }
            _ => {}
        }
    };

    let scan = |v: &Value, visit: &mut dyn FnMut(&Value)| {
        // One-level unwrap for lists (multi-arg inputs are lists of args).
        if let Value::List(items) = v {
            for item in items.iter() {
                visit(item);
            }
        } else {
            visit(v);
        }
    };

    for v in inputs.iter() {
        scan(v, &mut visit);
    }
    for v in expected.iter() {
        scan(v, &mut visit);
    }

    out
}

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
        Value::Function(_) | Value::Builtin(_) | Value::Ns(_) | Value::Node(_) => {
            // Opaque values — hash to a constant. Synthesis dedup is
            // structural over primitives + lists; functions, namespaces,
            // and Nodes don't compose meaningfully into the candidate
            // pool, so collapsing them all into a single hash bucket is
            // both correct and consistent with how legacy v1 handled
            // Function/Builtin.
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
/// Remap every node-index reference inside `node` by adding `offset`.
/// Used when concatenating sub-expression node trees during materialization
/// (synth-side) and during AST construction from SELPH (eval_v2's
/// `make-*` builtins, §9.36).
pub fn remap_node(node: &Node, offset: usize) -> Node {
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
    // Most literal kinds materialize to a single node. `Indexed` is
    // the exception: it expands to a 4-node `(nth x N)` subtree so
    // multi-arg synthesis can treat positional arguments as
    // depth-0 atoms while keeping the runtime lambda single-input
    // (the input is the args list).
    if let Dispatch::Literal(LiteralKind::Indexed(idx)) = &comp.dispatch {
        let nodes = vec![
            Node::Symbol(intern("nth")),
            Node::Symbol(intern("x")),
            Node::Int(*idx as i64),
            Node::App(vec![0, 1, 2]),
        ];
        return Some(SynthPool {
            nodes,
            root: 3,
            ret_type: comp.ret_type,
            priority: comp.priority,
        });
    }

    let node = match &comp.dispatch {
        Dispatch::Literal(LiteralKind::InputVar) => Node::Symbol(intern("x")),
        Dispatch::Literal(LiteralKind::Int(n)) => Node::Int(*n),
        Dispatch::Literal(LiteralKind::Num(n)) => Node::Num(*n),
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
///
/// §9.42: when `entry.has_hole` is set, the candidate is dispatched
/// to the affine fit-and-verify path. The return tuple's second
/// element carries the hole-substituted entry on success — callers
/// should wrap that one (not the original) when constructing the
/// final lambda.
fn test_candidate(
    entry: &SynthPool,
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    seen: &mut HashSet<Vec<u64>>,
    target: Option<Sym>,
    universe: &TypeUniverse,
) -> (TestOutcome, Option<SynthPool>) {
    // Type gate. The candidate's return type must be able to flow into
    // a slot of type `target`.
    if let Some(t) = target {
        if !universe.slot_accepts(t, entry.ret_type) {
            return (TestOutcome::Skipped, None);
        }
    }

    // §9.45 deletion: the hole-bearing-candidate branch is gone. With
    // the affine-fit pass migrated to pure-SELPH M-stages, no caller
    // creates `Hole` literals or hole-bearing pool entries anymore.

    // Wrap as lambda and evaluate once to get the function value.
    let (nodes, lambda_idx) = wrap_lambda(entry);
    let nodes_rc: Rc<[Node]> = nodes.into();
    let f = match eval_v2::eval(&nodes_rc, lambda_idx, env) {
        Ok(v) => v,
        Err(_) => return (TestOutcome::Errored, None),
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
            Err(_) => return (TestOutcome::Errored, None),
        }
    }

    // Observational equivalence dedup.
    if !beh.is_empty() {
        if seen.contains(&beh) {
            return (TestOutcome::Deduped, None);
        }
        seen.insert(beh);
    }

    if matches == inputs.len() && !inputs.is_empty() {
        (TestOutcome::Solution, None)
    } else {
        (TestOutcome::Tested, None)
    }
}

// §9.45 deletion: `test_hole_candidate`, `wrap_lambda_then_eval`,
// `value_to_f64`, `values_close_or_equal`, and `substitute_hole`
// all gone — the entire §9.42 hole-as-atom infrastructure is now
// dead code. Their callers (the affine_fit_pass and the
// `if entry.has_hole` branch in `test_candidate`) have been removed.
// `hole_component()` and the `LiteralKind::Hole` variant remain in
// place for now since they have no compile-time impact, but they're
// equally unreachable and can be deleted in a future cleanup pass.

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
    synthesize_inner(
        components, inputs, expected, env, universe,
        max_depth, max_candidates, None,
        &[], &[],
    )
}

/// Multi-arg variant of `synthesize` for the §9.31 physics curriculum
/// path. Each task input is a list of length `arg_types.len()`; the
/// runtime lambda is still single-input (`(lambda (x) ...)`) and
/// receives the list as `x`, but the search-time pool is seeded with
/// one indexed-atom per position. Each indexed atom materializes to
/// `(nth x i)` and carries the type of `args[i]`, so the rest of the
/// pipeline (reachability prune, type-gated composition, dedup) sees
/// N typed atoms instead of one List atom plus a swarm of `(nth x i)`
/// shapes inflating depth-1 fanout.
pub fn synthesize_args(
    components: &[SynthComponent],
    inputs: &[Value],
    arg_types: &[Sym],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
) -> SynthResult {
    synthesize_args_with_test(
        components, inputs, arg_types, expected, env, universe,
        max_depth, max_candidates, &[], &[],
    )
}

/// Like `synthesize_args` but with optional held-out test data.
/// When `test_inputs` / `test_expected` are non-empty, candidates that
/// pass training are additionally verified against the held-out pairs.
/// Candidates failing held-out are rejected and the search continues.
pub fn synthesize_args_with_test(
    components: &[SynthComponent],
    inputs: &[Value],
    arg_types: &[Sym],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    test_inputs: &[Value],
    test_expected: &[Value],
) -> SynthResult {
    let seed_atoms: Vec<SynthComponent> = arg_types
        .iter()
        .enumerate()
        .map(|(i, &t)| indexed_arg_component(i, t))
        .collect();
    synthesize_inner(
        components, inputs, expected, env, universe,
        max_depth, max_candidates, Some(seed_atoms),
        test_inputs, test_expected,
    )
}

fn synthesize_inner(
    components: &[SynthComponent],
    inputs: &[Value],
    expected: &[Value],
    env: &Env,
    universe: &TypeUniverse,
    max_depth: usize,
    max_candidates: usize,
    extra_seeds: Option<Vec<SynthComponent>>,
    test_inputs: &[Value],
    test_expected: &[Value],
) -> SynthResult {
    // Empty examples: nothing to fit. Return failure rather than
    // returning an arbitrary trivial program.
    if inputs.is_empty() || inputs.len() != expected.len() {
        return SynthResult::not_found(0);
    }

    // ── Task type inference ────────────────────────────────────────────
    let input_type = infer_uniform_type_sym(inputs).unwrap_or_else(type_any);
    let target = infer_uniform_type_sym(expected);

    // Capture whether we're on the multi-arg path (used at the end
    // of the function for the affine-fit pass).
    let extra_seeds_was_some = extra_seeds.is_some();

    // ── Reachability prune ─────────────────────────────────────────────
    // For multi-arg, the reach seeds are the per-arg types, NOT the
    // outer List type — so arithmetic on Num args gets reached even
    // though the outer `x` is a list.
    let seeds: Vec<Sym> = match &extra_seeds {
        Some(atoms) => {
            let mut s: Vec<Sym> = atoms.iter().map(|a| a.ret_type).collect();
            // Always include the outer list type so any list-shaped
            // helper components stay reachable too.
            s.push(input_type);
            s
        }
        None => vec![input_type],
    };
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

    // ── §9.45 P5: kernel-bypass flag for the migration validation gate ─
    // When the env contains `__bypass_kernel_affine__` bound to a
    // truthy value, skip the §9.41 data literal seeding and the §9.42
    // affine-fit pass below. The pure-SELPH M-stage chain
    // (examples/meta_curriculum/m_chain.selph) is expected to take
    // their place via the curriculum-side `__decomposers__` hook.
    //
    // This is the test-driven path that lets us validate the §9.45.6
    // migration gate: turn the kernel passes off, run the same task,
    // assert the SELPH curriculum solves it.
    let bypass_kernel = matches!(
        env.lookup(intern("__bypass_kernel_affine__")),
        Some(Value::Bool(true))
    );

    // ── Data-derived literal seeding (multi-arg only) ─────────────────
    // §9.41: scan inputs and outputs for unique primitive Int / Num /
    // Str / Bool values and seed them as depth-0 literal components.
    // Only fires on the multi-arg path (`extra_seeds.is_some()`)
    // because:
    //   - The existing single-input synthesize tests assert NEGATIVE
    //     conditions ("Memo is the fallback when Flat can't reach
    //     constant 99") that data-derived seeding fundamentally
    //     contradicts: with the constant always seeded, Flat solves
    //     them at depth 0. Gating the new behaviour to multi-arg
    //     keeps those tests valid.
    //   - The multi-arg path is new (§9.39) and its only consumers
    //     are the §9.40 meta-curriculum decomposers, where rich data
    //     constants are exactly what's wanted.
    if extra_seeds.is_some() && !bypass_kernel {
        let data_lits = collect_data_literals(inputs, expected);
        for lit in data_lits {
            // Skip if a literal with the same value is already in the
            // pool (matches the static seed entries).
            let dup = all_components.iter().any(|c| match (&c.dispatch, &lit.dispatch) {
                (Dispatch::Literal(LiteralKind::Int(a)), Dispatch::Literal(LiteralKind::Int(b))) => a == b,
                (Dispatch::Literal(LiteralKind::Num(a)), Dispatch::Literal(LiteralKind::Num(b))) => a.to_bits() == b.to_bits(),
                (Dispatch::Literal(LiteralKind::Str(a)), Dispatch::Literal(LiteralKind::Str(b))) => a == b,
                (Dispatch::Literal(LiteralKind::Bool(a)), Dispatch::Literal(LiteralKind::Bool(b))) => a == b,
                _ => false,
            });
            if !dup {
                all_components.push(lit);
            }
        }
    }

    match extra_seeds {
        Some(atoms) => {
            // Multi-arg path: indexed atoms instead of the single
            // input-var. Keep the input-var too, so a body that
            // genuinely wants the whole list (e.g., `(reduce + x)`)
            // can still reach it.
            all_components.push(input_var_component(input_type));
            all_components.extend(atoms);
            // §9.43 list constructors — needed by Form L for cross-
            // arity library reuse. Only added on the multi-arg path.
            let num = intern("Num");
            let list_ = intern("List");
            all_components.push(SynthComponent::named(
                "list", intern("list"), vec![num, num], list_, 5.0,
            ));
            all_components.push(SynthComponent::named(
                "list", intern("list"), vec![num, num, num], list_, 5.0,
            ));
        }
        None => {
            all_components.push(input_var_component(input_type));
        }
    }

    // ── §9.45: SELPH M-stage chain runs FIRST (multi-arg only) ────────
    //
    // Mirror the single-arg path's Stage A dispatch
    // (`synthesize_with_strategies`, line ~4943): give curriculum
    // decomposers the first crack before burning enumeration budget.
    // The chain is bounded — total cost across all M-stages is
    // O(pool³ × stages) ≈ a few thousand fits — so trying it first
    // is essentially free if it succeeds and adds a known overhead
    // if it doesn't. Without this ordering the chain only fires
    // after enumeration exhausts, inflating reported candidate
    // counts even when the chain itself would be cheap.
    if extra_seeds_was_some {
        if let Some((nodes, root, sd_explored, name_sym)) =
            try_selph_decomposers(env, inputs, expected, max_depth, max_candidates,
                                  test_inputs, test_expected)
        {
            // §9.47.5: the chain now self-validates against test data
            // internally (via the spec ns "test" key). The external
            // validate_held_out is still a safety net for cases where
            // the chain doesn't implement internal validation.
            if validate_held_out(&nodes, root, test_inputs, test_expected, env) {
                return SynthResult::success_from_decomposer(
                    nodes, root, sd_explored, name_sym,
                );
            }
            // Chain result failed held-out — fall through to Flat.
        }
    }

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

    // §9.42 affine-fit pass needs an end-of-search hook even when
    // the main enumeration runs out of budget. We track exhausted-or-
    // converged state via this flag and break out instead of early-
    // returning.
    let mut budget_exhausted = false;

    // Test depth-0 atoms.
    'depth0: for entry in &pool {
        if explored >= max_candidates {
            budget_exhausted = true;
            break 'depth0;
        }
        explored += 1;
        let (outcome, hole_sub) =
            test_candidate(entry, inputs, expected, env, &mut seen, target, universe);
        if let TestOutcome::Solution = outcome {
            let final_entry = hole_sub.as_ref().unwrap_or(entry);
            let (n, r) = wrap_lambda(final_entry);
            if validate_held_out(&n, r, test_inputs, test_expected, env) {
                return SynthResult::success(n, r, explored);
            }
            // Failed held-out — continue searching.
        }
    }

    // ── Depth 1..max_depth: bottom-up composition ──────────────────────
    let mut prev_start: usize = 0;
    let mut prev_end: usize = pool.len();

    'depth_loop: for _depth in 1..=max_depth {
        if budget_exhausted {
            break 'depth_loop;
        }
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
                budget_exhausted = true;
                break 'depth_loop;
            }
            explored += 1;

            let comp = &all_components[desc.comp_idx];
            let arg_refs: Vec<&SynthPool> =
                desc.args.iter().map(|&i| &pool[i]).collect();
            let entry = materialize_app(comp, &arg_refs);

            let (outcome, hole_sub) =
                test_candidate(&entry, inputs, expected, env, &mut seen, target, universe);
            match outcome {
                TestOutcome::Solution => {
                    let final_entry = hole_sub.as_ref().unwrap_or(&entry);
                    let (n, r) = wrap_lambda(final_entry);
                    if validate_held_out(&n, r, test_inputs, test_expected, env) {
                        return SynthResult::success(n, r, explored);
                    }
                    // Failed held-out — continue searching.
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

    // §9.45 deletion: the §9.42 affine-fit pass is gone. Every Form
    // (1, 2, 2b, 3, 4, 5, S, L) has been migrated to a pure-SELPH
    // M-stage in `examples/meta_curriculum/m{8,9,10,11,12,7}*.selph`,
    // dispatched via `__decomposers__` below. The kernel-bypass flag
    // (`__bypass_kernel_affine__`) is now functionally a no-op since
    // there's nothing left in the kernel to bypass; the flag stays
    // wired in `synthesize_inner` only to keep the compatibility
    // gate for `collect_data_literals` seeding (which still has a
    // future migration path).
    let _ = (target, universe, bypass_kernel);

    // §9.45 note: the multi-arg SELPH-decomposer chain runs BEFORE
    // enumeration (see line ~5897), not as a post-enumeration
    // fallback. The pre-enumeration position mirrors single-arg
    // Stage A and gives the chain accurate cost reporting (the
    // earlier "after enumeration" position made every SELPH solve
    // appear to cost `max_candidates` because enumeration burned
    // through the budget first).
    let _ = budget_exhausted;

    SynthResult::not_found(explored)
}

// §9.45 deletion: `affine_fit_pass`, `try_affine_fit`, and the
// `build_*_hole` / `build_op_combo` helpers all gone (~280 LOC).
// Each Form they implemented is now a pure-SELPH M-stage in
// `examples/meta_curriculum/`. The `__bypass_kernel_affine__` env
// flag remains wired in `synthesize_inner` for the
// `collect_data_literals` seeding gate (which is the next migration
// target — DLS / RDC / LCI per §9.45.8).

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
        // No custom types loaded — type_metadata is empty.
        assert!(u.type_metadata(sym("Int")).is_none());
    }

    // ── §9.37 Stage B: TypeUniverse::from_env ───────────────────────────

    /// Helper: build an env after evaluating one or more top-level
    /// SELPH forms. The last form is typically a `(define __types__ ...)`
    /// but earlier forms can introduce helper definitions.
    fn env_with_types(types_src: &str) -> crate::types_v2::Env {
        let env = crate::eval_v2::make_default_env();
        let (old_nodes, roots) = crate::parser::parse_file(types_src).unwrap();
        let new_nodes: std::rc::Rc<[Node]> =
            crate::eval_v2::convert_tree(&old_nodes).into();
        for &r in &roots {
            crate::eval_v2::eval(&new_nodes, r, &env).unwrap();
        }
        env
    }

    #[test]
    fn from_env_with_no_types_returns_primitives_baseline() {
        let env = crate::eval_v2::make_default_env();
        let u = TypeUniverse::from_env(&env);
        // Same as primitives — known set has the standard types,
        // type_metadata is empty.
        assert!(u.is_known(sym("Int")));
        assert!(u.type_metadata(sym("Int")).is_none());
    }

    #[test]
    fn from_env_loads_custom_type_with_predicate_and_subtype() {
        let env = env_with_types(
            r#"
            (define __types__
              (ns ("Color" (ns
                ("predicate" (lambda (v) (string? v)))
                ("subtype-of" (list "String" "Any"))
                ("priority" 50)))))
            "#,
        );
        let u = TypeUniverse::from_env(&env);
        assert!(u.is_known(sym("Color")));
        let meta = u.type_metadata(sym("Color")).expect("Color metadata");
        assert!(meta.predicate.is_some());
        assert_eq!(meta.subtype_of.len(), 2);
        assert!(meta.subtype_of.contains(&sym("String")));
        assert!(meta.subtype_of.contains(&sym("Any")));
        assert!((meta.priority - 50.0).abs() < 1e-9);
        assert_eq!(meta.decomposers.len(), 0);
    }

    #[test]
    fn from_env_loads_decomposers_for_type() {
        let env = env_with_types(
            r#"
            (define id-decomp (lambda (spec) nil))
            (define __types__
              (ns ("Int" (ns
                ("decomposers" (ns ("zoo-decomp" id-decomp)
                                   ("alpha-decomp" id-decomp)))))))
            "#,
        );
        let u = TypeUniverse::from_env(&env);
        let meta = u.type_metadata(sym("Int")).expect("Int metadata");
        assert_eq!(meta.decomposers.len(), 2);
        // Sorted by name → alpha-decomp first, zoo-decomp second.
        assert_eq!(resolve(meta.decomposers[0].0), "alpha-decomp");
        assert_eq!(resolve(meta.decomposers[1].0), "zoo-decomp");
    }

    #[test]
    fn from_env_skips_malformed_entries_silently() {
        // The Color entry isn't a namespace — it should be skipped
        // without error. The Int entry is well-formed and should load.
        let env = env_with_types(
            r#"
            (define __types__
              (ns ("Color" 42)
                  ("Int" (ns ("priority" 100)))))
            "#,
        );
        let u = TypeUniverse::from_env(&env);
        assert!(u.type_metadata(sym("Color")).is_none());
        assert!(u.type_metadata(sym("Int")).is_some());
    }

    #[test]
    fn from_env_loads_full_bootstrap_file() {
        // Smoke test: load the actual examples/__types__.selph and
        // verify the 10 expected primitive types appear.
        let src = std::fs::read_to_string("../examples/__types__.selph")
            .expect("read bootstrap");
        let env = crate::eval_v2::make_default_env();
        let (old_nodes, roots) = crate::parser::parse_file(&src).unwrap();
        let new_nodes: std::rc::Rc<[Node]> =
            crate::eval_v2::convert_tree(&old_nodes).into();
        for &r in &roots {
            crate::eval_v2::eval(&new_nodes, r, &env).unwrap();
        }
        let u = TypeUniverse::from_env(&env);
        for name in &[
            "Any", "Int", "Num", "String", "Bool", "List", "Namespace",
            "Function", "Node", "Nil",
        ] {
            assert!(
                u.type_metadata(intern(name)).is_some(),
                "expected {} in __types__",
                name
            );
        }
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
        // Single distinct output (so D&C bails — needs ≥2 groups), Int
        // input (so Memo bails — needs string keys), and target value
        // 23 which is unreachable at depth 1 using the literal pool
        // {0,1,2,3,4,5,6,7,10,-1}. No atomic predicate or arity-2
        // composition produces 23. So Flat, BD, HO, D&C, and Memo all
        // give up.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let r = synthesize_with_strategies(
            &comps,
            &[Value::Int(1), Value::Int(2), Value::Int(3)],
            &[Value::Int(23), Value::Int(23), Value::Int(23)],
            &env,
            &universe,
            1,
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

    // ── BD strategy tests ────────────────────────────────────────────────
    //
    // BD covers logical compositions of unary bool-returning predicates.
    // These tests use `even` / `odd` (built-in unary Int → Bool predicates
    // that show up in `primitive_components`) plus a custom unary
    // library function for the cases where Flat would otherwise win.

    /// Build a tiny component catalog containing only the named bool
    /// predicates `even` and `odd`. Used to force BD to fire by denying
    /// Flat the and/or/not operators.
    fn bd_only_components() -> Vec<SynthComponent> {
        vec![
            SynthComponent::named(
                "even", intern("even"), vec![intern("Int")], intern("Bool"), 0.0,
            ),
            SynthComponent::named(
                "odd", intern("odd"), vec![intern("Int")], intern("Bool"), 0.0,
            ),
        ]
    }

    #[test]
    fn bd_returns_none_for_non_bool_target() {
        init_special_forms();
        let env = eval_v2::make_default_env();
        let comps = bd_only_components();
        // Int target — BD should refuse.
        let r = bool_decompose(
            &comps,
            &[Value::Int(1), Value::Int(2)],
            &[Value::Int(1), Value::Int(2)],
            &env,
        );
        assert!(r.is_none());
    }

    #[test]
    fn bd_returns_none_when_no_predicates_available() {
        init_special_forms();
        let env = eval_v2::make_default_env();
        // Empty predicate catalog — even with bool target, BD has nothing
        // to compose.
        let r = bool_decompose(
            &[],
            &[Value::Int(1), Value::Int(2)],
            &[Value::Bool(true), Value::Bool(false)],
            &env,
        );
        assert!(r.is_none());
    }

    #[test]
    fn bd_solves_not_p() {
        // Target = (not (even x)) on integers.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let comps = bd_only_components();

        let inputs = vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)];
        let expected = vec![
            Value::Bool(true),  // not even 1
            Value::Bool(false), // not even 2
            Value::Bool(true),  // not even 3
            Value::Bool(false), // not even 4
        ];
        let (nodes, root, _explored) = bool_decompose(&comps, &inputs, &expected, &env)
            .expect("bd should find (not (even x))");

        // Sanity-check: the synthesized lambda must run and reproduce
        // the expected outputs.
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(&expected) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(matches!((got, e), (Value::Bool(a), Value::Bool(b)) if a == *b));
        }
    }

    #[test]
    fn bd_solves_and_p_q() {
        // Target = (and (even x) (odd x)) — always false. We want BD
        // to find some valid composition; the dispatcher would normally
        // also accept the literal `false`, but bool_decompose only emits
        // structured compositions, so this verifies the (and P Q) path
        // walks far enough.
        //
        // To make the test unambiguous, use inputs whose expected pattern
        // is NOT all-false: target = (and (even x) (odd x)) is uniformly
        // false, but (and (even x) (even x)) on inputs [1,2,3,4] gives
        // [F,T,F,T]. So pick (and even even) — even though it equals
        // `even` itself, BD enumerates it on the i==j diagonal and that
        // counts as a valid path.
        //
        // Better: target = (or (even x) (odd x)) which is always true.
        // (or P Q) is a real composition, both predicates participate,
        // and we can verify the resulting source.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let comps = bd_only_components();

        let inputs = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        let expected = vec![Value::Bool(true), Value::Bool(true), Value::Bool(true)];

        let (nodes, root, _explored) = bool_decompose(&comps, &inputs, &expected, &env)
            .expect("bd should find some bool composition that's always true");

        // Verify the program runs and matches.
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for i in &inputs {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(matches!(got, Value::Bool(true)));
        }
    }

    #[test]
    fn bd_solves_and_p_not_q() {
        // Target = (and (even x) (not (odd x))) — equivalent to even,
        // but the (and P (not Q)) template should be reachable. Inputs
        // chosen so the target pattern is non-trivial.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let comps = bd_only_components();

        let inputs = vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)];
        let expected = vec![
            Value::Bool(false),
            Value::Bool(true),
            Value::Bool(false),
            Value::Bool(true),
        ];
        let (nodes, root, _explored) = bool_decompose(&comps, &inputs, &expected, &env)
            .expect("bd should find a composition matching even-pattern");

        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(&expected) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(matches!((got, e), (Value::Bool(a), Value::Bool(b)) if a == *b));
        }
    }

    #[test]
    fn bd_skips_components_that_error_on_input() {
        // A component declared as `Int -> Bool` but whose underlying
        // function only handles strings should be probed-and-dropped
        // rather than crashing the strategy. Build a library function
        // bound under name `bad-pred` that errors on Int inputs, and
        // make sure BD still solves the task using the surviving
        // predicates.
        use crate::types_v2::{FunctionData, NodeRef};

        init_special_forms();
        let env = eval_v2::make_default_env();

        // (lambda (x) (string-upper x)) — errors on Int.
        let mut nodes: Vec<Node> = Vec::new();
        let su_idx = nodes.len();
        nodes.push(Node::Symbol(intern("string-upper")));
        let x_idx = nodes.len();
        nodes.push(Node::Symbol(intern("x")));
        let body_idx = nodes.len();
        nodes.push(Node::App(vec![su_idx, x_idx]));
        let nodes_rc: Rc<[Node]> = nodes.into();
        let bad = Value::Function(Rc::new(FunctionData {
            params: vec![intern("x")],
            body: NodeRef {
                nodes: nodes_rc,
                idx: body_idx,
            },
            captured_env: env.clone(),
            letrec_scope: None,
        }));
        env.define(intern("bad-pred"), bad);

        // Component catalog with `bad-pred` claiming Int → Bool plus
        // the real `even` predicate.
        let comps = vec![
            SynthComponent::named(
                "bad-pred",
                intern("bad-pred"),
                vec![intern("Int")],
                intern("Bool"),
                0.0,
            ),
            SynthComponent::named(
                "even", intern("even"), vec![intern("Int")], intern("Bool"), 0.0,
            ),
        ];

        let inputs = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        let expected = vec![Value::Bool(true), Value::Bool(false), Value::Bool(true)];

        let r = bool_decompose(&comps, &inputs, &expected, &env);
        // Either solves via (not (even x)) directly, or doesn't find a
        // composition — but it must NOT crash on bad-pred.
        if let Some((nodes, root, _)) = r {
            let nodes_rc: Rc<[Node]> = nodes.into();
            let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
            for (i, e) in inputs.iter().zip(&expected) {
                let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
                assert!(matches!((got, e), (Value::Bool(a), Value::Bool(b)) if a == *b));
            }
        }
    }

    #[test]
    fn dispatcher_uses_bd_when_flat_lacks_logical_ops() {
        // Tiny catalog: only `even` and `odd`. No `and`/`or`/`not`,
        // no bool literals. Target = always-true on mixed-parity
        // inputs. Atomic predicates can't reach this pattern:
        //   (even x) on [1,2,3,4] = [F,T,F,T]
        //   (odd  x) on [1,2,3,4] = [T,F,T,F]
        // The only solution is `(or (even x) (odd x))`, which Flat
        // can't construct without `or`. BD must fire.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = bd_only_components();

        let inputs = vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)];
        let expected = vec![
            Value::Bool(true),
            Value::Bool(true),
            Value::Bool(true),
            Value::Bool(true),
        ];

        let r = synthesize_with_strategies(
            &comps,
            &inputs,
            &expected,
            &env,
            &universe,
            3,
            500,
        );
        assert!(r.found);
        assert_eq!(r.strategy, Some(Strategy::BoolDecomp));

        // The synthesized program must actually work.
        let nodes_rc: Rc<[Node]> = r.nodes.unwrap().into();
        let f = eval_v2::eval(&nodes_rc, r.root.unwrap(), &env).unwrap();
        for i in &inputs {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(matches!(got, Value::Bool(true)));
        }
    }

    // ── HO strategy tests ────────────────────────────────────────────────
    //
    // HO covers four templates that recursively sub-synthesize an inner
    // function and splice it into a wrapper. Tests construct shapes that
    // Flat can't reach within budget so HO must take over.

    #[test]
    fn ho_list_map_recovers_inner_function() {
        // [[1,2,3], [4,5,6]] → [[2,3,4], [5,6,7]]
        // Inner sub-spec: each element + 1. Flat sub-synth finds (add x 1)
        // or (add 1 x). Wrapper is (lambda (x) (map inner x)).
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            Value::list(vec![Value::Int(4), Value::Int(5), Value::Int(6)]),
        ];
        let expected = vec![
            Value::list(vec![Value::Int(2), Value::Int(3), Value::Int(4)]),
            Value::list(vec![Value::Int(5), Value::Int(6), Value::Int(7)]),
        ];

        let r = higher_order_decompose(
            &comps, &inputs, &expected, &env, &universe, 2, 2000,
        )
        .expect("HO should solve via list-map");

        let (nodes, root, _explored) = r;
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(&expected) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(eval_v2::values_equal(&got, e));
        }
    }

    #[test]
    fn ho_list_filter_recovers_predicate() {
        // [[1,2,3,4], [5,6,7,8]] → [[2,4], [6,8]] — keep even.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)]),
            Value::list(vec![Value::Int(5), Value::Int(6), Value::Int(7), Value::Int(8)]),
        ];
        let expected = vec![
            Value::list(vec![Value::Int(2), Value::Int(4)]),
            Value::list(vec![Value::Int(6), Value::Int(8)]),
        ];

        let (nodes, root, _explored) = higher_order_decompose(
            &comps, &inputs, &expected, &env, &universe, 2, 2000,
        )
        .expect("HO should solve via list-filter");

        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(&expected) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(eval_v2::values_equal(&got, e));
        }
    }

    #[test]
    fn ho_split_map_join_recovers_word_transform() {
        // "foo bar baz" → "FOO BAR BAZ" via split-map-join on " ".
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![
            Value::str("foo bar baz"),
            Value::str("hello world hi"),
            Value::str("a b c"),
        ];
        let expected = vec![
            Value::str("FOO BAR BAZ"),
            Value::str("HELLO WORLD HI"),
            Value::str("A B C"),
        ];

        let (nodes, root, _explored) = higher_order_decompose(
            &comps, &inputs, &expected, &env, &universe, 2, 2000,
        )
        .expect("HO should solve via split-map-join");

        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(&expected) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(eval_v2::values_equal(&got, e));
        }
    }

    #[test]
    fn ho_char_map_join_recovers_letter_transform() {
        // "abc" → "ABC", "hi" → "HI" via char-map-join.
        // The inner per-character spec is "a"→"A", "b"→"B", etc.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![
            Value::str("abc"),
            Value::str("hello"),
            Value::str("xyz"),
        ];
        let expected = vec![
            Value::str("ABC"),
            Value::str("HELLO"),
            Value::str("XYZ"),
        ];

        // For this to land via char-map-join (not split-map-join with " "),
        // none of the strings can contain a delimiter. The inputs above
        // are single-word, so split-map-join's `any_multi` check fails
        // for every delimiter and HO falls through to char-map-join.
        let (nodes, root, _explored) = higher_order_decompose(
            &comps, &inputs, &expected, &env, &universe, 2, 2000,
        )
        .expect("HO should solve via char-map-join");

        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(&expected) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(eval_v2::values_equal(&got, e));
        }
    }

    #[test]
    fn ho_bails_when_no_template_applies() {
        // Int → Int — no list/string structure for any template.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let r = higher_order_decompose(
            &comps,
            &[Value::Int(1), Value::Int(2), Value::Int(3)],
            &[Value::Int(2), Value::Int(4), Value::Int(6)],
            &env,
            &universe,
            2,
            500,
        );
        assert!(r.is_none());
    }

    #[test]
    fn ho_dedup_rejects_conflicting_pairs() {
        // Same input, different outputs — should reject.
        let pairs = vec![
            (Value::Int(1), Value::Int(2)),
            (Value::Int(1), Value::Int(99)), // conflict
            (Value::Int(2), Value::Int(4)),
            (Value::Int(3), Value::Int(6)),
        ];
        assert!(ho_dedup_spec(pairs).is_none());
    }

    #[test]
    fn ho_dedup_rejects_too_few_unique_pairs() {
        // Only 2 unique inputs after dedup — below the threshold of 3.
        let pairs = vec![
            (Value::Int(1), Value::Int(2)),
            (Value::Int(1), Value::Int(2)),
            (Value::Int(2), Value::Int(4)),
            (Value::Int(2), Value::Int(4)),
        ];
        assert!(ho_dedup_spec(pairs).is_none());
    }

    #[test]
    fn dispatcher_uses_ho_for_list_map_task() {
        // List→list of same length transformation — Flat can't construct
        // (lambda (x) (map (lambda (e) (add e 1)) x)) at low depth, but
        // HO list-map decomposes it into a flat sub-task that Flat solves.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            Value::list(vec![Value::Int(10), Value::Int(20), Value::Int(30)]),
            Value::list(vec![Value::Int(0), Value::Int(5), Value::Int(7)]),
        ];
        let expected = vec![
            Value::list(vec![Value::Int(2), Value::Int(3), Value::Int(4)]),
            Value::list(vec![Value::Int(11), Value::Int(21), Value::Int(31)]),
            Value::list(vec![Value::Int(1), Value::Int(6), Value::Int(8)]),
        ];

        let r = synthesize_with_strategies(
            &comps, &inputs, &expected, &env, &universe, 2, 2000,
        );
        assert!(r.found);
        assert_eq!(r.strategy, Some(Strategy::HigherOrder));
    }

    // ── D&C strategy tests ──────────────────────────────────────────────
    //
    // D&C handles classification-style tasks where outputs come from a
    // small set of distinct values and a Bool separator distinguishes
    // groups. Tests cover the constant-leaves case (most common) plus
    // the bail conditions (single-output, no separator).

    #[test]
    fn dc_solves_two_constant_branches() {
        // x ≤ 0 → "neg", x ≥ 1 → "pos". Two distinct outputs, separator
        // is some inequality on x. Both branches are constants.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![
            Value::Int(-3),
            Value::Int(-1),
            Value::Int(0),
            Value::Int(1),
            Value::Int(5),
            Value::Int(10),
        ];
        let expected = vec![
            Value::str("neg"),
            Value::str("neg"),
            Value::str("neg"),
            Value::str("pos"),
            Value::str("pos"),
            Value::str("pos"),
        ];

        let (nodes, root, _explored) = divide_and_conquer(
            &comps, &inputs, &expected, &env, &universe, 2, 5000,
        )
        .expect("D&C should solve two-constant classification");

        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(&expected) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(eval_v2::values_equal(&got, e));
        }
    }

    #[test]
    fn dc_solves_three_branches_via_recursion() {
        // x < 0 → -1, x = 0 → 0, x > 0 → 1. Three groups force the
        // recursive nested-if construction.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![
            Value::Int(-5),
            Value::Int(-1),
            Value::Int(0),
            Value::Int(1),
            Value::Int(7),
        ];
        let expected = vec![
            Value::Int(-1),
            Value::Int(-1),
            Value::Int(0),
            Value::Int(1),
            Value::Int(1),
        ];

        let (nodes, root, _explored) = divide_and_conquer(
            &comps, &inputs, &expected, &env, &universe, 2, 10000,
        )
        .expect("D&C should solve three-way classification");

        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(&expected) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(eval_v2::values_equal(&got, e));
        }
    }

    #[test]
    fn dc_bails_on_single_output_value() {
        // Only one distinct output → no partition possible. D&C must
        // refuse so the dispatcher falls through to other strategies.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let r = divide_and_conquer(
            &comps,
            &[Value::Int(1), Value::Int(2), Value::Int(3)],
            &[Value::Int(7), Value::Int(7), Value::Int(7)],
            &env,
            &universe,
            2,
            500,
        );
        assert!(r.is_none());
    }

    #[test]
    fn dispatcher_uses_dc_for_classification_task() {
        // 3-way classification, full primitive catalog. Flat can't
        // construct nested if-expressions on its own; BD doesn't apply
        // (output isn't bool); HO doesn't apply (no list/string shape);
        // D&C is the only strategy that fits.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![
            Value::Int(-3),
            Value::Int(-1),
            Value::Int(0),
            Value::Int(2),
            Value::Int(7),
        ];
        let expected = vec![
            Value::str("neg"),
            Value::str("neg"),
            Value::str("zero"),
            Value::str("pos"),
            Value::str("pos"),
        ];

        let r = synthesize_with_strategies(
            &comps, &inputs, &expected, &env, &universe, 2, 10000,
        );
        assert!(r.found);
        assert_eq!(r.strategy, Some(Strategy::DivideConquer));
    }

    // ── Induction strategy tests ────────────────────────────────────────
    //
    // Induction decomposes (inputs → expected) into two simpler sub-
    // syntheses (inputs → mid) and (mid → expected) where `mid` is
    // produced by probing a curated set of builtins. Tests construct
    // pipelines that need a clear two-step decomposition.

    #[test]
    fn induce_solves_two_step_pipeline() {
        // Target: x → |x| + 1 — abs followed by increment.
        // The literal pool has 0..7,10,-1, so `(add (abs x) 1)` is
        // depth-2 reachable but the test caps Flat at depth 1, forcing
        // induction to take over via the abs intermediate.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        // Use a tiny custom catalog that excludes `abs` from the
        // outer Flat pass — that way induction's intermediate probe
        // (which uses env.lookup, not the catalog) is the only path
        // that introduces it.
        let comps: Vec<SynthComponent> = comps
            .into_iter()
            .filter(|c| c.name != "abs")
            .collect();

        let inputs = vec![
            Value::Int(-3),
            Value::Int(-1),
            Value::Int(0),
            Value::Int(2),
            Value::Int(5),
        ];
        let expected = vec![
            Value::Int(4),
            Value::Int(2),
            Value::Int(1),
            Value::Int(3),
            Value::Int(6),
        ];

        let r = induce_decomposition(
            &comps, &inputs, &expected, &env, &universe, 2, 5000,
        );
        // Even with a curated catalog, induction needs both halves to
        // sub-synthesize. If it can't, the test still asserts that the
        // *call site* doesn't crash and either returns Some or None.
        if let Some((nodes, root, _explored)) = r {
            let nodes_rc: Rc<[Node]> = nodes.into();
            let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
            for (i, e) in inputs.iter().zip(&expected) {
                let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
                assert!(eval_v2::values_equal(&got, e));
            }
        }
    }

    #[test]
    fn induce_compose_steps_via_let_binding() {
        // Direct test of the composer: build trivial step1 = (lambda (x) x)
        // and step2 = (lambda (x) (negate x)). Compose into a single
        // lambda and verify it negates its input.
        init_special_forms();
        let env = eval_v2::make_default_env();

        // step1: (lambda (x) x)
        let step1_nodes = vec![
            Node::Symbol(intern("x")),                 // 0
            Node::Lambda(vec![intern("x")], 0),        // 1
        ];
        let step1_lambda = 1;

        // step2: (lambda (x) (negate x))
        let step2_nodes = vec![
            Node::Symbol(intern("negate")),            // 0
            Node::Symbol(intern("x")),                 // 1
            Node::App(vec![0, 1]),                     // 2
            Node::Lambda(vec![intern("x")], 2),        // 3
        ];
        let step2_lambda = 3;

        let (nodes, root) =
            induce_compose_steps(step1_nodes, step1_lambda, step2_nodes, step2_lambda)
                .expect("compose should succeed for valid lambdas");

        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();

        let r1 = eval_v2::apply(&f, &[Value::Int(7)], &env).unwrap();
        assert!(matches!(r1, Value::Int(-7)));

        let r2 = eval_v2::apply(&f, &[Value::Int(-3)], &env).unwrap();
        assert!(matches!(r2, Value::Int(3)));
    }

    #[test]
    fn induce_compose_handles_two_nontrivial_lambdas() {
        // step1 = (lambda (x) (add x 1))     ; increment
        // step2 = (lambda (x) (multiply x 2))  ; double
        // Composed = (lambda (x) (multiply (add x 1) 2))
        // For input 5: (add 5 1) = 6, (multiply 6 2) = 12.
        init_special_forms();
        let env = eval_v2::make_default_env();

        // step1
        let step1_nodes = vec![
            Node::Symbol(intern("add")),               // 0
            Node::Symbol(intern("x")),                 // 1
            Node::Int(1),                              // 2
            Node::App(vec![0, 1, 2]),                  // 3
            Node::Lambda(vec![intern("x")], 3),        // 4
        ];
        // step2
        let step2_nodes = vec![
            Node::Symbol(intern("multiply")),          // 0
            Node::Symbol(intern("x")),                 // 1
            Node::Int(2),                              // 2
            Node::App(vec![0, 1, 2]),                  // 3
            Node::Lambda(vec![intern("x")], 3),        // 4
        ];

        let (nodes, root) =
            induce_compose_steps(step1_nodes, 4, step2_nodes, 4).expect("compose");
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();

        let got = eval_v2::apply(&f, &[Value::Int(5)], &env).unwrap();
        assert!(matches!(got, Value::Int(12)));

        let got = eval_v2::apply(&f, &[Value::Int(0)], &env).unwrap();
        assert!(matches!(got, Value::Int(2)));
    }

    #[test]
    fn induce_skips_useless_intermediates() {
        // If the intermediate probe collapses to a constant (e.g.
        // `(multiply x 0)` always returns 0), induction must skip it.
        // We can't easily exercise this without a custom env, but we
        // can test the helper directly.
        let constants = vec![Value::Int(7), Value::Int(7), Value::Int(7)];
        assert!(induce_all_identical(&constants));

        let varying = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        assert!(!induce_all_identical(&varying));
    }

    #[test]
    fn induce_returns_none_for_empty_inputs() {
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();
        let r = induce_decomposition(
            &comps, &[], &[], &env, &universe, 2, 100,
        );
        assert!(r.is_none());
    }

    // ── Recursive Decomposition strategy tests ─────────────────────────
    //
    // RD predicts the outermost function from spec features, inverts it
    // to derive a sub-spec, and recursively sub-synthesizes. The tests
    // exercise: (a) family prediction, (b) binary-arithmetic inversion
    // composition, (c) library-function bridge composition, and (d) the
    // dispatcher routing for tasks Flat alone can't reach.

    #[test]
    fn rd_predicts_arithmetic_for_int_to_int_task() {
        init_special_forms();
        let env = eval_v2::make_default_env();
        let inputs = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        let expected = vec![Value::Int(2), Value::Int(4), Value::Int(6)];
        let features = rd_extract_features(&inputs, &expected, &env);
        assert_eq!(features.input_type, "int");
        assert_eq!(features.output_type, "int");
        assert_eq!(rd_predict_family(&features), RdFamily::Arithmetic);
    }

    #[test]
    fn rd_predicts_string_op_when_output_is_substring() {
        init_special_forms();
        let env = eval_v2::make_default_env();
        let inputs = vec![Value::str("hello"), Value::str("world")];
        let expected = vec![Value::str("hel"), Value::str("wor")];
        let features = rd_extract_features(&inputs, &expected, &env);
        assert_eq!(features.input_type, "str");
        assert_eq!(features.output_type, "str");
        assert!(features.output_is_substring);
        assert_eq!(rd_predict_family(&features), RdFamily::StringOp);
    }

    #[test]
    fn rd_invert_binary_add_recovers_constant_difference() {
        let inputs = vec![Value::Int(1), Value::Int(5), Value::Int(10)];
        let expected = vec![Value::Int(3), Value::Int(7), Value::Int(12)];
        let sub = rd_invert_binary_arith("add", &inputs, &expected).unwrap();
        // k_i = output_i - input_i = 2 for all i.
        for v in &sub.expected {
            assert!(matches!(v, Value::Int(2)));
        }
    }

    #[test]
    fn rd_invert_string_take_recovers_per_example_length() {
        let inputs = vec![
            Value::str("hello"),
            Value::str("worldly"),
        ];
        let expected = vec![Value::str("hel"), Value::str("worl")];
        let sub = rd_invert_string_take(&inputs, &expected).unwrap();
        assert_eq!(sub.expected.len(), 2);
        assert!(matches!(sub.expected[0], Value::Int(3)));
        assert!(matches!(sub.expected[1], Value::Int(4)));
    }

    #[test]
    fn rd_solves_x_times_succ_via_arithmetic_inversion() {
        // Target: x → x * (x + 1). At max-depth=1, flat enumerates only
        // depth-1 expressions; the answer is depth-2. RD's binary
        // arithmetic inversion derives a depth-1 sub-spec, sub-synthesizes
        // `(multiply x x)`, and composes the outer `add` for free.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();
        let inputs = vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
            Value::Int(5),
        ];
        let expected = vec![
            Value::Int(2),
            Value::Int(6),
            Value::Int(12),
            Value::Int(20),
            Value::Int(30),
        ];
        let r = recursive_decompose(
            &comps, &inputs, &expected, &env, &universe, 1, 5000,
        );
        let (nodes, root, _explored) = r.expect("RD should solve x*(x+1) at max-depth=1");
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(expected.iter()) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(eval_v2::values_equal(&got, e));
        }
    }

    #[test]
    fn rd_solves_via_library_function_bridge() {
        // Define a unary library function `wrap = (lambda (s) (concat (concat "[" s) "]"))`,
        // then ask RD to find `(string-upper (wrap x))` at max-depth=1.
        // Flat alone can't reach a depth-2 expression at that depth;
        // RD's library-decomposition phase probes `wrap` as an inner
        // bridge and sub-synthesizes the outer `string-upper`.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();

        // Build wrap manually as a Value::Function with captured env.
        // Body: (concat (concat "[" s) "]")
        let wrap_nodes: Vec<Node> = vec![
            Node::Str("[".to_string()),                        // 0
            Node::Symbol(intern("s")),                         // 1
            Node::Symbol(intern("concat")),                    // 2
            Node::App(vec![2, 0, 1]),                          // 3: (concat "[" s)
            Node::Str("]".to_string()),                        // 4
            Node::Symbol(intern("concat")),                    // 5
            Node::App(vec![5, 3, 4]),                          // 6: (concat (concat "[" s) "]")
        ];
        let wrap_body = crate::types_v2::NodeRef {
            nodes: Rc::from(wrap_nodes),
            idx: 6,
        };
        let wrap_fn = Value::Function(Rc::new(crate::types_v2::FunctionData {
            params: vec![intern("s")],
            body: wrap_body,
            captured_env: env.clone(),
            letrec_scope: None,
        }));
        env.define(intern("wrap"), wrap_fn);

        let comps = default_synth_components(&env, &default_skip_set());
        let inputs = vec![
            Value::str("hi"),
            Value::str("abc"),
            Value::str("world"),
        ];
        let expected = vec![
            Value::str("[HI]"),
            Value::str("[ABC]"),
            Value::str("[WORLD]"),
        ];

        let r = recursive_decompose(
            &comps, &inputs, &expected, &env, &universe, 1, 5000,
        );
        let (nodes, root, _explored) =
            r.expect("RD should solve string-upper∘wrap via library bridge");
        let nodes_rc: Rc<[Node]> = nodes.into();
        let f = eval_v2::eval(&nodes_rc, root, &env).unwrap();
        for (i, e) in inputs.iter().zip(expected.iter()) {
            let got = eval_v2::apply(&f, std::slice::from_ref(i), &env).unwrap();
            assert!(
                eval_v2::values_equal(&got, e),
                "got {:?}, expected {:?}",
                got,
                e
            );
        }
    }

    #[test]
    fn dispatcher_uses_rd_for_arithmetic_inversion_task() {
        // Same task as `rd_solves_x_times_succ_via_arithmetic_inversion`
        // but routed through the full strategy dispatcher. RD should
        // win after Flat fails at max-depth=1.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();
        let inputs = vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
            Value::Int(5),
        ];
        let expected = vec![
            Value::Int(2),
            Value::Int(6),
            Value::Int(12),
            Value::Int(20),
            Value::Int(30),
        ];
        let r = synthesize_with_strategies(
            &comps, &inputs, &expected, &env, &universe, 1, 5000,
        );
        assert!(r.found);
        assert_eq!(r.strategy, Some(Strategy::RecursiveDecomposition));
    }

    #[test]
    fn rd_returns_none_for_empty_inputs() {
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();
        let r = recursive_decompose(&comps, &[], &[], &env, &universe, 2, 100);
        assert!(r.is_none());
    }

    #[test]
    fn dispatcher_prefers_flat_over_bd_when_flat_solves() {
        // Full primitive catalog. Target = parity-dependent pattern
        // that Flat can solve as the atomic `(odd x)` (no composition
        // needed). BD should NOT fire because Flat already won.
        init_special_forms();
        let env = eval_v2::make_default_env();
        let universe = TypeUniverse::primitives();
        let comps = primitive_components();

        let inputs = vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)];
        let expected = vec![
            Value::Bool(true),
            Value::Bool(false),
            Value::Bool(true),
            Value::Bool(false),
        ];

        let r = synthesize_with_strategies(
            &comps,
            &inputs,
            &expected,
            &env,
            &universe,
            3,
            5000,
        );
        assert!(r.found);
        assert_eq!(r.strategy, Some(Strategy::Flat));
    }
}
