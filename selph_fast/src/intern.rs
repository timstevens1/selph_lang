//! String interning for fast symbol comparison and hashing.
//!
//! Symbols are stored once in a global table and referenced by a u32 ID.
//! This makes environment lookups use integer hashing instead of string hashing.
//!
//! The interner is global (not thread-local) so that Sym IDs are consistent
//! across rayon worker threads — critical for VM bytecode dispatch.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

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

static INTERNER: LazyLock<RwLock<Interner>> = LazyLock::new(|| RwLock::new(Interner::new()));

/// Intern a string, returning its unique Sym ID.
pub fn intern(s: &str) -> Sym {
    // Fast path: check if already interned with a read lock
    {
        let interner = INTERNER.read().unwrap();
        if let Some(&id) = interner.map.get(s) {
            return Sym(id);
        }
    }
    // Slow path: acquire write lock to insert
    INTERNER.write().unwrap().intern(s)
}

/// Resolve a Sym back to its string.
pub fn resolve(sym: Sym) -> String {
    INTERNER.read().unwrap().resolve(sym).to_string()
}

/// Resolve a Sym to a &str — must be used within a callback to avoid borrow issues.
pub fn with_resolved<F, R>(sym: Sym, f: F) -> R
where
    F: FnOnce(&str) -> R,
{
    let interner = INTERNER.read().unwrap();
    f(interner.resolve(sym))
}

impl std::fmt::Display for Sym {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let interner = INTERNER.read().unwrap();
        write!(f, "{}", interner.resolve(*self))
    }
}
