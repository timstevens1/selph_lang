//! Hindley-Milner type inference for SELPH synthesis.
//!
//! Provides a richer type system than the u8 type tags used for fast
//! pruning in the synthesizer. The HM system supports type variables,
//! unification with occurs check, and polymorphic builtin signatures.
//!
//! The u8 tags remain as a fast pre-filter; this module provides a
//! more precise second-pass check that catches type errors the tag
//! system cannot (e.g., list element types, function argument/return
//! type consistency).

use std::collections::HashMap;

// ── Type representation ─────────────────────────────────────────────

/// A Hindley-Milner type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    TNum,
    TStr,
    TBool,
    TNil,
    /// Homogeneous list with element type.
    TList(Box<Type>),
    /// 2D grid of integers 0-9 (ARC-AGI).
    TGrid,
    /// Function type: (params...) -> return.
    TFn(Vec<Type>, Box<Type>),
    /// Type variable (for unification / polymorphism).
    TVar(u32),
}

/// Substitution: maps type variable ids to their resolved types.
pub type Subst = HashMap<u32, Type>;

// ── Core operations ─────────────────────────────────────────────────

/// Create a fresh type variable, incrementing the counter.
pub fn fresh_var(counter: &mut u32) -> Type {
    let id = *counter;
    *counter += 1;
    Type::TVar(id)
}

/// Apply a substitution to a type, resolving all type variables.
pub fn apply(ty: &Type, subst: &Subst) -> Type {
    match ty {
        Type::TVar(id) => {
            if let Some(resolved) = subst.get(id) {
                // Follow chains: the resolved type may itself contain TVars.
                apply(resolved, subst)
            } else {
                ty.clone()
            }
        }
        Type::TList(elem) => Type::TList(Box::new(apply(elem, subst))),
        Type::TFn(params, ret) => Type::TFn(
            params.iter().map(|p| apply(p, subst)).collect(),
            Box::new(apply(ret, subst)),
        ),
        // Ground types are unchanged.
        Type::TNum | Type::TStr | Type::TBool | Type::TNil | Type::TGrid => ty.clone(),
    }
}

/// Occurs check: does type variable `var_id` appear anywhere in `ty`
/// (after applying the current substitution)?
fn occurs_in(var_id: u32, ty: &Type, subst: &Subst) -> bool {
    let ty = apply(ty, subst);
    match &ty {
        Type::TVar(id) => *id == var_id,
        Type::TList(elem) => occurs_in(var_id, elem, subst),
        Type::TFn(params, ret) => {
            params.iter().any(|p| occurs_in(var_id, p, subst))
                || occurs_in(var_id, ret, subst)
        }
        Type::TNum | Type::TStr | Type::TBool | Type::TNil | Type::TGrid => false,
    }
}

/// Unify two types, extending the substitution.
///
/// Returns `Ok(())` on success, `Err(message)` on type mismatch.
pub fn unify(t1: &Type, t2: &Type, subst: &mut Subst) -> Result<(), String> {
    let t1 = apply(t1, subst);
    let t2 = apply(t2, subst);

    if t1 == t2 {
        return Ok(());
    }

    match (&t1, &t2) {
        // TVar binds to anything (with occurs check).
        (Type::TVar(id), _) => {
            if occurs_in(*id, &t2, subst) {
                return Err(format!("infinite type: ?t{} in {:?}", id, t2));
            }
            subst.insert(*id, t2);
            Ok(())
        }
        (_, Type::TVar(id)) => {
            if occurs_in(*id, &t1, subst) {
                return Err(format!("infinite type: ?t{} in {:?}", id, t1));
            }
            subst.insert(*id, t1);
            Ok(())
        }

        // Structural unification for compound types.
        (Type::TList(a), Type::TList(b)) => unify(a, b, subst),

        (Type::TFn(params_a, ret_a), Type::TFn(params_b, ret_b)) => {
            if params_a.len() != params_b.len() {
                return Err(format!(
                    "arity mismatch: {} vs {} params",
                    params_a.len(),
                    params_b.len()
                ));
            }
            for (pa, pb) in params_a.iter().zip(params_b.iter()) {
                unify(pa, pb, subst)?;
            }
            unify(ret_a, ret_b, subst)
        }

        // Same ground types already handled by t1 == t2 above.
        // Different ground types or structural mismatch:
        _ => Err(format!("type mismatch: {:?} vs {:?}", t1, t2)),
    }
}

// ── Conversion from u8 type tags ────────────────────────────────────

use crate::synth::{TYPE_NUM, TYPE_STR, TYPE_BOOL, TYPE_LIST, TYPE_GRID, TYPE_ANY};

/// Convert a u8 type tag to an HM Type.
///
/// `TYPE_ANY` (255) maps to a fresh type variable; call with a counter
/// to get unique variables.
pub fn type_from_tag(tag: u8, counter: &mut u32) -> Type {
    match tag {
        TYPE_NUM => Type::TNum,
        TYPE_STR => Type::TStr,
        TYPE_BOOL => Type::TBool,
        TYPE_LIST => Type::TList(Box::new(fresh_var(counter))),
        TYPE_GRID => Type::TGrid,
        TYPE_ANY => fresh_var(counter),
        _ => fresh_var(counter), // unknown tags become variables
    }
}

/// Convert an HM Type back to a u8 type tag (lossy).
///
/// Type variables and compound types map to `TYPE_ANY`.
pub fn type_to_tag(ty: &Type, subst: &Subst) -> u8 {
    let resolved = apply(ty, subst);
    match &resolved {
        Type::TNum => TYPE_NUM,
        Type::TStr => TYPE_STR,
        Type::TBool => TYPE_BOOL,
        Type::TList(_) => TYPE_LIST,
        Type::TGrid => TYPE_GRID,
        _ => TYPE_ANY,
    }
}

// ── Component type signatures ───────────────────────────────────────

use crate::synth::SynthComponent;

/// Compute the HM function type for a SynthComponent by converting its
/// u8 param_types and ret_type into HM types.
///
/// This gives each component call its own set of fresh type variables
/// for any `TYPE_ANY` slots, enabling proper polymorphic instantiation.
pub fn component_fn_type(comp: &SynthComponent, counter: &mut u32) -> (Vec<Type>, Type) {
    let params: Vec<Type> = comp.param_types.iter()
        .map(|&tag| type_from_tag(tag, counter))
        .collect();
    let ret = type_from_tag(comp.ret_type, counter);
    (params, ret)
}

// ── Builtin type signatures (polymorphic) ───────────────────────────

/// Return a fresh HM type signature for a named builtin.
///
/// For polymorphic builtins (list ops, higher-order functions, equality),
/// each call gets fresh type variables so they can be independently
/// constrained.
///
/// Returns `None` for unknown builtins (the caller should fall back to
/// the component's u8-derived types).
pub fn builtin_type(name: &str, counter: &mut u32) -> Option<(Vec<Type>, Type)> {
    let a = fresh_var(counter);
    let b = fresh_var(counter);

    let sig = match name {
        // Arithmetic: (Num, Num) -> Num
        "add" | "+" | "subtract" | "-" | "multiply" | "*" | "divide" | "/"
        | "modulo" | "%" | "min" | "max" => {
            (vec![Type::TNum, Type::TNum], Type::TNum)
        }

        // Unary numeric
        "abs" | "negate" | "floor" | "ceil" => {
            (vec![Type::TNum], Type::TNum)
        }

        // Comparison: (Num, Num) -> Bool
        "<" | ">" | "<=" | ">=" => {
            (vec![Type::TNum, Type::TNum], Type::TBool)
        }

        // Polymorphic equality: (a, a) -> Bool
        "=" | "!=" => {
            (vec![a.clone(), a], Type::TBool)
        }

        // Logic
        "not" => (vec![Type::TBool], Type::TBool),
        "even" | "odd" => (vec![Type::TNum], Type::TBool),

        // String ops: unary Str -> Str
        "string-upper" | "string-lower" | "string-reverse" | "string-trim" => {
            (vec![Type::TStr], Type::TStr)
        }

        // String analysis
        "string-length" => (vec![Type::TStr], Type::TNum),
        "string-contains" | "string-starts-with" | "string-ends-with" => {
            (vec![Type::TStr, Type::TStr], Type::TBool)
        }
        "string-split" => {
            (vec![Type::TStr, Type::TStr], Type::TList(Box::new(Type::TStr)))
        }
        "string-join" => {
            (vec![Type::TList(Box::new(Type::TStr)), Type::TStr], Type::TStr)
        }
        "string-replace" => {
            (vec![Type::TStr, Type::TStr, Type::TStr], Type::TStr)
        }
        "substring" | "char-at" => {
            // Simplified: these need more params but cover the common case
            return None;
        }
        "concat" => (vec![Type::TStr, Type::TStr], Type::TStr),
        "to-string" => (vec![Type::TNum], Type::TStr),
        "to-number" => (vec![Type::TStr], Type::TNum),

        // List operations (polymorphic)
        "list" => (vec![a.clone()], Type::TList(Box::new(a))),
        "cons" => {
            (vec![a.clone(), Type::TList(Box::new(a.clone()))],
             Type::TList(Box::new(a)))
        }
        "head" => {
            (vec![Type::TList(Box::new(a.clone()))], a)
        }
        "tail" => {
            let la = Type::TList(Box::new(a.clone()));
            (vec![la.clone()], la)
        }
        "length" => {
            (vec![Type::TList(Box::new(a))], Type::TNum)
        }
        "nth" => {
            (vec![Type::TList(Box::new(a.clone())), Type::TNum], a)
        }
        "append" => {
            let la = Type::TList(Box::new(a.clone()));
            (vec![la.clone(), la.clone()], la)
        }
        "reverse" | "sort" => {
            let la = Type::TList(Box::new(a.clone()));
            (vec![la.clone()], la)
        }
        "range" => {
            (vec![Type::TNum], Type::TList(Box::new(Type::TNum)))
        }
        "empty?" => {
            (vec![Type::TList(Box::new(a))], Type::TBool)
        }
        "contains" => {
            (vec![Type::TList(Box::new(a.clone())), a], Type::TBool)
        }

        // Higher-order
        "map" => {
            // (a -> b, List a) -> List b
            let fn_ab = Type::TFn(vec![a.clone()], Box::new(b.clone()));
            (vec![fn_ab, Type::TList(Box::new(a))],
             Type::TList(Box::new(b)))
        }
        "filter" => {
            // (a -> Bool, List a) -> List a
            let pred = Type::TFn(vec![a.clone()], Box::new(Type::TBool));
            let la = Type::TList(Box::new(a.clone()));
            (vec![pred, la.clone()], la)
        }
        "reduce" => {
            // (b, a -> b, List a, b) -> b
            let reducer = Type::TFn(vec![b.clone(), a.clone()], Box::new(b.clone()));
            (vec![reducer, Type::TList(Box::new(a)), b.clone()], b)
        }
        "identity" => (vec![a.clone()], a),
        "print" => (vec![a], Type::TNil),

        _ => return None,
    };

    Some(sig)
}

// ── Compatibility check for synthesis ───────────────────────────────

/// Check whether a function with the given param types and return type
/// can accept the given argument types.
///
/// This performs trial unification in a *cloned* substitution so that
/// the caller's substitution is not modified on failure.
///
/// Returns `true` if all argument types unify with the corresponding
/// parameter types and the return type is consistent.
pub fn type_compatible(
    param_types: &[Type],
    ret_type: &Type,
    arg_types: &[Type],
    subst: &mut Subst,
) -> bool {
    if param_types.len() != arg_types.len() {
        return false;
    }

    // Trial unification on a clone — only commit on success.
    let mut trial = subst.clone();

    for (param, arg) in param_types.iter().zip(arg_types.iter()) {
        if unify(param, arg, &mut trial).is_err() {
            return false;
        }
    }

    // Check that the return type is still consistent.
    let _resolved_ret = apply(ret_type, &trial);

    // Success: commit the trial substitution.
    *subst = trial;
    true
}

/// Infer an HM type from a runtime Value.
///
/// This bridges the gap between runtime values (used in synthesis
/// testing) and the static type system.
pub fn type_of_value(v: &crate::types::Value) -> Type {
    match v {
        crate::types::Value::Num(_) => Type::TNum,
        crate::types::Value::Str(_) => Type::TStr,
        crate::types::Value::Bool(_) => Type::TBool,
        crate::types::Value::Nil => Type::TNil,
        crate::types::Value::List(items) => {
            if items.is_empty() {
                // Empty list: polymorphic, use a dummy TVar(u32::MAX)
                // that won't collide with counter-allocated vars.
                Type::TList(Box::new(Type::TVar(u32::MAX)))
            } else {
                Type::TList(Box::new(type_of_value(&items[0])))
            }
        }
        crate::types::Value::Closure(params, _, _, _, _) => {
            // We don't know the exact types, so use TVar placeholders.
            let params: Vec<Type> = (0..params.len())
                .map(|i| Type::TVar(u32::MAX - 1 - i as u32))
                .collect();
            let ret = Type::TVar(u32::MAX - 1 - params.len() as u32);
            Type::TFn(params, Box::new(ret))
        }
        _ => Type::TNil, // Builtins, macros, namespaces -> opaque
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_ground_types() {
        let mut subst = Subst::new();
        assert!(unify(&Type::TNum, &Type::TNum, &mut subst).is_ok());
        assert!(unify(&Type::TStr, &Type::TStr, &mut subst).is_ok());
        assert!(unify(&Type::TNum, &Type::TStr, &mut subst).is_err());
    }

    #[test]
    fn test_unify_tvar() {
        let mut subst = Subst::new();
        let mut counter = 0;
        let a = fresh_var(&mut counter);
        assert!(unify(&a, &Type::TNum, &mut subst).is_ok());
        assert_eq!(apply(&a, &subst), Type::TNum);
    }

    #[test]
    fn test_unify_list() {
        let mut subst = Subst::new();
        let mut counter = 0;
        let a = fresh_var(&mut counter);
        let list_a = Type::TList(Box::new(a.clone()));
        let list_num = Type::TList(Box::new(Type::TNum));
        assert!(unify(&list_a, &list_num, &mut subst).is_ok());
        assert_eq!(apply(&a, &subst), Type::TNum);
    }

    #[test]
    fn test_unify_fn() {
        let mut subst = Subst::new();
        let mut counter = 0;
        let a = fresh_var(&mut counter);
        let fn_a_num = Type::TFn(vec![a.clone()], Box::new(Type::TNum));
        let fn_str_num = Type::TFn(vec![Type::TStr], Box::new(Type::TNum));
        assert!(unify(&fn_a_num, &fn_str_num, &mut subst).is_ok());
        assert_eq!(apply(&a, &subst), Type::TStr);
    }

    #[test]
    fn test_occurs_check() {
        let mut subst = Subst::new();
        let mut counter = 0;
        let a = fresh_var(&mut counter);
        let list_a = Type::TList(Box::new(a.clone()));
        // a = List(a) should fail (infinite type).
        assert!(unify(&a, &list_a, &mut subst).is_err());
    }

    #[test]
    fn test_type_compatible_basic() {
        let mut subst = Subst::new();
        // add: (Num, Num) -> Num
        let params = vec![Type::TNum, Type::TNum];
        let ret = Type::TNum;
        let args = vec![Type::TNum, Type::TNum];
        assert!(type_compatible(&params, &ret, &args, &mut subst));

        let bad_args = vec![Type::TNum, Type::TStr];
        let mut subst2 = Subst::new();
        assert!(!type_compatible(&params, &ret, &bad_args, &mut subst2));
    }

    #[test]
    fn test_type_compatible_polymorphic() {
        let mut subst = Subst::new();
        let mut counter = 0;
        // head: (List a) -> a
        let a = fresh_var(&mut counter);
        let params = vec![Type::TList(Box::new(a.clone()))];
        let ret = a;
        let args = vec![Type::TList(Box::new(Type::TNum))];
        assert!(type_compatible(&params, &ret, &args, &mut subst));
        assert_eq!(apply(&ret, &subst), Type::TNum);
    }

    #[test]
    fn test_builtin_type_add() {
        let mut counter = 0;
        let sig = builtin_type("add", &mut counter);
        assert!(sig.is_some());
        let (params, ret) = sig.unwrap();
        assert_eq!(params, vec![Type::TNum, Type::TNum]);
        assert_eq!(ret, Type::TNum);
    }

    #[test]
    fn test_builtin_type_polymorphic_eq() {
        let mut counter = 0;
        let sig = builtin_type("=", &mut counter).unwrap();
        let (params, ret) = sig;
        // Both params should be the same TVar
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], params[1]);
        assert_eq!(ret, Type::TBool);
    }

    #[test]
    fn test_tag_roundtrip() {
        let mut counter = 0;
        let subst = Subst::new();
        assert_eq!(type_to_tag(&type_from_tag(TYPE_NUM, &mut counter), &subst), TYPE_NUM);
        assert_eq!(type_to_tag(&type_from_tag(TYPE_STR, &mut counter), &subst), TYPE_STR);
        assert_eq!(type_to_tag(&type_from_tag(TYPE_BOOL, &mut counter), &subst), TYPE_BOOL);
    }
}
