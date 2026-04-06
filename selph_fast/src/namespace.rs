//! First-class Namespace values for SELPH.
//!
//! A namespace is an immutable tree of named entries. Entries can be any
//! SELPH `Value` (functions, data, other namespaces). Supports path-based
//! lookup, merge, flatten, and the usual collection operations.

use std::collections::HashMap;
use crate::types::Value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the inner map from a `Value::Namespace`, or return an error.
fn as_ns(v: &Value) -> Result<&HashMap<String, Value>, String> {
    match v {
        Value::Namespace(map) => Ok(map),
        other => Err(format!("expected namespace, got {:?}", other)),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Get a direct child by name.
pub fn ns_get(ns: &Value, key: &str) -> Result<Value, String> {
    let map = as_ns(ns)?;
    map.get(key)
        .cloned()
        .ok_or_else(|| format!("namespace: no entry '{}'", key))
}

/// Navigate a path of keys through nested namespaces.
///
/// An empty path returns the namespace itself.
pub fn ns_get_path(ns: &Value, path: &[&str]) -> Result<Value, String> {
    if path.is_empty() {
        return Ok(ns.clone());
    }
    let val = ns_get(ns, path[0])?;
    if path.len() == 1 {
        return Ok(val);
    }
    match &val {
        Value::Namespace(_) => ns_get_path(&val, &path[1..]),
        _ => Err(format!(
            "namespace: '{}' is not a namespace, can't traverse further",
            path[0]
        )),
    }
}

/// Return a new namespace with `key` set to `val` (immutable update).
pub fn ns_put(ns: &Value, key: &str, val: Value) -> Result<Value, String> {
    let map = as_ns(ns)?;
    let mut new_map = map.clone();
    new_map.insert(key.to_string(), val);
    Ok(Value::Namespace(new_map))
}

/// Return a sorted list of keys.
pub fn ns_keys(ns: &Value) -> Result<Vec<String>, String> {
    let map = as_ns(ns)?;
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    Ok(keys)
}

/// Return the values (in key-sorted order for determinism).
pub fn ns_values(ns: &Value) -> Result<Vec<Value>, String> {
    let map = as_ns(ns)?;
    let mut entries: Vec<(&String, &Value)> = map.iter().collect();
    entries.sort_by_key(|(k, _)| k.clone());
    Ok(entries.into_iter().map(|(_, v)| v.clone()).collect())
}

/// Merge two namespaces. Entries from `b` win on conflict, except when both
/// sides are namespaces -- those are merged recursively.
pub fn ns_merge(a: &Value, b: &Value) -> Result<Value, String> {
    let map_a = as_ns(a)?;
    let map_b = as_ns(b)?;
    let mut result = map_a.clone();
    for (key, val_b) in map_b {
        let merged = match result.get(key) {
            Some(Value::Namespace(_)) if matches!(val_b, Value::Namespace(_)) => {
                ns_merge(result.get(key).unwrap(), val_b)?
            }
            _ => val_b.clone(),
        };
        if let Value::Namespace(inner) = merged {
            result.insert(key.clone(), Value::Namespace(inner));
        } else {
            result.insert(key.clone(), merged);
        }
    }
    Ok(Value::Namespace(result))
}

/// Number of direct entries in the namespace.
pub fn ns_size(ns: &Value) -> Result<usize, String> {
    let map = as_ns(ns)?;
    Ok(map.len())
}

/// Flatten to a map of dotted-path -> leaf-value.
///
/// Nested namespaces are expanded: `{a: {b: 1}}` becomes `{"a.b": 1}`.
pub fn ns_flatten(ns: &Value) -> Result<HashMap<String, Value>, String> {
    let map = as_ns(ns)?;
    let mut result = HashMap::new();
    flatten_inner(map, "", &mut result);
    Ok(result)
}

fn flatten_inner(map: &HashMap<String, Value>, prefix: &str, out: &mut HashMap<String, Value>) {
    for (key, val) in map {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{}.{}", prefix, key)
        };
        match val {
            Value::Namespace(inner) => flatten_inner(inner, &path, out),
            _ => {
                out.insert(path, val.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_ns() -> Value {
        Value::Namespace(HashMap::new())
    }

    fn sample_ns() -> Value {
        let mut map = HashMap::new();
        map.insert("x".to_string(), Value::Num(1.0));
        map.insert("y".to_string(), Value::Num(2.0));
        Value::Namespace(map)
    }

    #[test]
    fn test_get_and_put() {
        let ns = empty_ns();
        let ns = ns_put(&ns, "a", Value::Num(42.0)).unwrap();
        assert!(matches!(ns_get(&ns, "a").unwrap(), Value::Num(n) if n == 42.0));
        assert!(ns_get(&ns, "missing").is_err());
    }

    #[test]
    fn test_get_path() {
        let inner = sample_ns();
        let outer = ns_put(&empty_ns(), "math", inner).unwrap();
        let val = ns_get_path(&outer, &["math", "x"]).unwrap();
        assert!(matches!(val, Value::Num(n) if n == 1.0));
    }

    #[test]
    fn test_get_path_non_namespace() {
        let ns = ns_put(&empty_ns(), "a", Value::Num(1.0)).unwrap();
        assert!(ns_get_path(&ns, &["a", "b"]).is_err());
    }

    #[test]
    fn test_keys_and_values() {
        let ns = sample_ns();
        let keys = ns_keys(&ns).unwrap();
        assert_eq!(keys, vec!["x".to_string(), "y".to_string()]);
        let vals = ns_values(&ns).unwrap();
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_merge() {
        let a = ns_put(&empty_ns(), "x", Value::Num(1.0)).unwrap();
        let b = ns_put(&empty_ns(), "y", Value::Num(2.0)).unwrap();
        let merged = ns_merge(&a, &b).unwrap();
        assert_eq!(ns_size(&merged).unwrap(), 2);
    }

    #[test]
    fn test_merge_recursive() {
        let inner_a = ns_put(&empty_ns(), "f", Value::Num(1.0)).unwrap();
        let inner_b = ns_put(&empty_ns(), "g", Value::Num(2.0)).unwrap();
        let a = ns_put(&empty_ns(), "math", inner_a).unwrap();
        let b = ns_put(&empty_ns(), "math", inner_b).unwrap();
        let merged = ns_merge(&a, &b).unwrap();
        let math = ns_get(&merged, "math").unwrap();
        assert_eq!(ns_size(&math).unwrap(), 2);
    }

    #[test]
    fn test_flatten() {
        let inner = ns_put(&empty_ns(), "b", Value::Num(1.0)).unwrap();
        let outer = ns_put(&empty_ns(), "a", inner).unwrap();
        let flat = ns_flatten(&outer).unwrap();
        assert!(flat.contains_key("a.b"));
        assert_eq!(flat.len(), 1);
    }

    #[test]
    fn test_size() {
        assert_eq!(ns_size(&empty_ns()).unwrap(), 0);
        assert_eq!(ns_size(&sample_ns()).unwrap(), 2);
    }

    #[test]
    fn test_not_a_namespace() {
        let v = Value::Num(42.0);
        assert!(ns_get(&v, "x").is_err());
        assert!(ns_put(&v, "x", Value::Nil).is_err());
        assert!(ns_keys(&v).is_err());
        assert!(ns_size(&v).is_err());
    }
}
