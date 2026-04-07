//! String interning for fast symbol comparison and hashing.
//!
//! Symbols are stored once in a global table and referenced by a u32 ID.
//! This makes environment lookups use integer hashing instead of string hashing.

use std::cell::RefCell;
use std::collections::HashMap;

/// An interned symbol — a lightweight, Copy handle to a string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sym(pub u32);

/// The interner: maps strings to unique IDs and back.
struct Interner {
    map: HashMap<String, u32>,
    vec: Vec<String>,
}

impl Interner {
    fn new() -> Self {
        Interner {
            map: HashMap::new(),
            vec: Vec::new(),
        }
    }

    fn intern(&mut self, s: &str) -> Sym {
        if let Some(&id) = self.map.get(s) {
            return Sym(id);
        }
        let id = self.vec.len() as u32;
        self.vec.push(s.to_string());
        self.map.insert(s.to_string(), id);
        Sym(id)
    }

    fn resolve(&self, sym: Sym) -> &str {
        &self.vec[sym.0 as usize]
    }
}

thread_local! {
    static INTERNER: RefCell<Interner> = RefCell::new(Interner::new());
}

/// Intern a string, returning its unique Sym ID.
pub fn intern(s: &str) -> Sym {
    INTERNER.with(|i| i.borrow_mut().intern(s))
}

/// Resolve a Sym back to its string.
pub fn resolve(sym: Sym) -> String {
    INTERNER.with(|i| i.borrow().resolve(sym).to_string())
}

/// Resolve a Sym to a &str — must be used within a callback to avoid borrow issues.
pub fn with_resolved<F, R>(sym: Sym, f: F) -> R
where
    F: FnOnce(&str) -> R,
{
    INTERNER.with(|i| {
        let interner = i.borrow();
        f(interner.resolve(sym))
    })
}

impl std::fmt::Display for Sym {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        INTERNER.with(|i| {
            let interner = i.borrow();
            write!(f, "{}", interner.resolve(*self))
        })
    }
}
