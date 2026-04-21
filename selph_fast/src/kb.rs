//! Wikidata knowledge base — efficient storage and lookup for millions of entities.
//!
//! Loaded from JSONL (one JSON object per line), builds three indices:
//! - Forward: entity label → {property: value, ...}
//! - Reverse: value → [(entity, property), ...]
//! - Property: property name → [entity, ...]
//!
//! Exposed to SELPH via builtins:
//!   (kb-get entity property)       → value or nil
//!   (kb-properties entity)         → list of property names
//!   (kb-search property pattern)   → list of entities where prop value contains pattern
//!   (kb-is-property value)         → list of property names where value appears
//!   (kb-count)                     → number of entities
//!   (kb-filter property value)     → list of entities where property = value
//!   (kb-path entity1 entity2)      → shortest path as list of (entity property value) triples
//!   (kb-path-count entity1 entity2 max-depth) → number of distinct paths (flow)

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::sync::OnceLock;
use std::time::Instant;

use crate::intern::{intern, resolve, Sym};
use crate::types_v2::{Env, Value};

/// A single entity's properties: property name → value (as string).
/// All values stored as strings for uniformity — numeric values are
/// stringified at load time, SELPH builtins parse as needed.
type PropMap = HashMap<Sym, String>;

/// Reverse index entry: (entity_label_sym, property_name_sym)
type ReverseEntry = (Sym, Sym);

pub struct KnowledgeBase {
    /// entity label → properties
    forward: HashMap<Sym, PropMap>,
    /// value string (lowercased) → list of (entity, property) that have this value
    reverse: HashMap<String, Vec<ReverseEntry>>,
    /// property name → list of entities that have it
    by_property: HashMap<Sym, Vec<Sym>>,
    /// Q-id → resolved label (for resolving references)
    labels: HashMap<String, String>,
}

static KB: OnceLock<KnowledgeBase> = OnceLock::new();

/// Get the global KB, or None if not loaded.
pub fn get_kb() -> Option<&'static KnowledgeBase> {
    KB.get()
}

/// Load KB from a JSONL file + optional labels file for Q-id resolution.
pub fn load_kb(jsonl_path: &str, labels_path: Option<&str>) -> Result<(), String> {
    if KB.get().is_some() {
        return Ok(()); // already loaded
    }

    let t0 = Instant::now();

    // Load labels for Q-id resolution
    let labels: HashMap<String, String> = if let Some(lp) = labels_path {
        let data = std::fs::read_to_string(lp)
            .map_err(|e| format!("Failed to read labels file {}: {}", lp, e))?;
        parse_labels_json(&data)?
    } else {
        HashMap::new()
    };

    eprintln!("KB: loaded {} labels in {:.1}s", labels.len(), t0.elapsed().as_secs_f64());
    let t1 = Instant::now();

    let file = std::fs::File::open(jsonl_path)
        .map_err(|e| format!("Failed to open KB file {}: {}", jsonl_path, e))?;
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let mut forward: HashMap<Sym, PropMap> = HashMap::new();
    let mut reverse: HashMap<String, Vec<ReverseEntry>> = HashMap::new();
    let mut by_property: HashMap<Sym, Vec<Sym>> = HashMap::new();
    let mut count = 0;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Minimal JSON parsing — we know the structure
        let entity = parse_entity_json(line, &labels)?;
        if entity.label.is_empty() {
            continue;
        }

        let label_sym = intern(&entity.label);

        let mut props = PropMap::new();
        for (prop_name, prop_value) in &entity.properties {
            let prop_sym = intern(prop_name);

            // Resolve Q-id references
            let resolved = if prop_value.starts_with('Q')
                && prop_value[1..].chars().all(|c| c.is_ascii_digit())
            {
                labels.get(prop_value).cloned().unwrap_or_else(|| prop_value.clone())
            } else {
                prop_value.clone()
            };

            props.insert(prop_sym, resolved.clone());

            // Reverse index
            let rev_key = resolved.to_lowercase();
            reverse
                .entry(rev_key)
                .or_default()
                .push((label_sym, prop_sym));

            // Property index
            by_property.entry(prop_sym).or_default().push(label_sym);
        }

        forward.insert(label_sym, props);
        count += 1;

        if count % 500_000 == 0 {
            eprintln!("KB: {} entities loaded...", count);
        }
    }

    let elapsed = t1.elapsed().as_secs_f64();
    eprintln!(
        "KB: loaded {} entities, {} reverse entries, {} properties in {:.1}s",
        forward.len(),
        reverse.len(),
        by_property.len(),
        elapsed
    );

    let kb = KnowledgeBase {
        forward,
        reverse,
        by_property,
        labels,
    };

    KB.set(kb).map_err(|_| "KB already initialized".to_string())?;
    Ok(())
}

// ── Minimal JSON parsing for entity JSONL ──────────────────────────────────

struct EntityData {
    label: String,
    properties: Vec<(String, String)>,
}

fn parse_entity_json(line: &str, _labels: &HashMap<String, String>) -> Result<EntityData, String> {
    // We expect: {"id":"Q31","label":"Belgium","properties":{...}, ...}
    // Use a simple state machine — no serde dependency needed.
    let bytes = line.as_bytes();
    let mut label = String::new();
    let mut properties = Vec::new();

    // Find "label":"..."
    if let Some(pos) = find_key(bytes, b"\"label\"") {
        label = extract_string_value(bytes, pos)?;
    }

    // Find "properties":{...}
    if let Some(pos) = find_key(bytes, b"\"properties\"") {
        properties = extract_properties(bytes, pos)?;
    }

    Ok(EntityData { label, properties })
}

fn find_key(bytes: &[u8], key: &[u8]) -> Option<usize> {
    bytes
        .windows(key.len())
        .position(|w| w == key)
        .map(|p| p + key.len())
}

fn extract_string_value(bytes: &[u8], after_key: usize) -> Result<String, String> {
    // Skip :"
    let mut i = after_key;
    while i < bytes.len() && bytes[i] != b'"' {
        i += 1;
    }
    i += 1; // skip opening "
    let start = i;
    while i < bytes.len() && bytes[i] != b'"' {
        if bytes[i] == b'\\' {
            i += 1; // skip escaped char
        }
        i += 1;
    }
    let s = std::str::from_utf8(&bytes[start..i])
        .map_err(|e| format!("UTF-8 error: {}", e))?;
    Ok(unescape_json_string(s))
}

fn extract_properties(bytes: &[u8], after_key: usize) -> Result<Vec<(String, String)>, String> {
    // Find the opening {
    let mut i = after_key;
    while i < bytes.len() && bytes[i] != b'{' {
        i += 1;
    }
    if i >= bytes.len() {
        return Ok(Vec::new());
    }
    i += 1; // skip {

    let mut props = Vec::new();

    loop {
        // Skip whitespace
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        if bytes[i] == b',' {
            i += 1;
            continue;
        }

        // Read key
        if bytes[i] != b'"' {
            break;
        }
        i += 1;
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' { i += 1; }
            i += 1;
        }
        let key = std::str::from_utf8(&bytes[key_start..i])
            .unwrap_or("")
            .to_string();
        i += 1; // skip closing "

        // Skip :
        while i < bytes.len() && bytes[i] != b':' {
            i += 1;
        }
        i += 1;

        // Skip whitespace
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        // Read value
        if i >= bytes.len() {
            break;
        }

        let value;
        if bytes[i] == b'"' {
            // String value
            i += 1;
            let val_start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' { i += 1; }
                i += 1;
            }
            value = unescape_json_string(
                std::str::from_utf8(&bytes[val_start..i]).unwrap_or("")
            );
            i += 1;
        } else if bytes[i] == b'{' || bytes[i] == b'[' {
            // Nested object/array — skip it
            let open = bytes[i];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 1;
            i += 1;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == open { depth += 1; }
                else if bytes[i] == close { depth -= 1; }
                else if bytes[i] == b'"' {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        if bytes[i] == b'\\' { i += 1; }
                        i += 1;
                    }
                }
                i += 1;
            }
            continue; // skip nested values
        } else {
            // Number, bool, null
            let val_start = i;
            while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            value = std::str::from_utf8(&bytes[val_start..i])
                .unwrap_or("")
                .to_string();
        }

        if !key.is_empty() && !value.is_empty() && value != "null" {
            props.push((key, value));
        }
    }

    Ok(props)
}

fn unescape_json_string(s: &str) -> String {
    if !s.contains('\\') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some('u') => {
                    // Unicode escape: \uXXXX
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(cp) {
                            out.push(ch);
                        }
                    }
                }
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_labels_json(data: &str) -> Result<HashMap<String, String>, String> {
    // Simple parser for {"Q123": "Label", ...}
    let mut map = HashMap::new();
    let bytes = data.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Find next key
        while i < bytes.len() && bytes[i] != b'"' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        i += 1;
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' { i += 1; }
            i += 1;
        }
        let key = std::str::from_utf8(&bytes[key_start..i]).unwrap_or("").to_string();
        i += 1;

        // Skip to value
        while i < bytes.len() && bytes[i] != b'"' && bytes[i] != b'}' {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' { i += 1; }
            i += 1;
        }
        let val = unescape_json_string(std::str::from_utf8(&bytes[val_start..i]).unwrap_or(""));
        i += 1;

        if !key.is_empty() && !val.is_empty() {
            map.insert(key, val);
        }
    }

    Ok(map)
}

// ── SELPH builtins ─────────────────────────────────────────────────────────

/// `(kb-get entity property)` → value string or nil
pub fn bi_kb_get(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("kb-get: expected 2 args, got {}", args.len()));
    }
    let kb = get_kb().ok_or("kb-get: no KB loaded")?;
    let entity = intern(args[0].as_str()?);
    let prop = intern(args[1].as_str()?);

    match kb.forward.get(&entity) {
        Some(props) => match props.get(&prop) {
            Some(val) => {
                // Try to return as number if it looks like one
                if let Ok(n) = val.parse::<i64>() {
                    Ok(Value::Int(n))
                } else if let Ok(n) = val.parse::<f64>() {
                    Ok(Value::Num(n))
                } else {
                    Ok(Value::str(val.clone()))
                }
            }
            None => Ok(Value::Nil),
        },
        None => Ok(Value::Nil),
    }
}

/// `(kb-properties entity)` → list of property name strings
pub fn bi_kb_properties(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("kb-properties: expected 1 arg, got {}", args.len()));
    }
    let kb = get_kb().ok_or("kb-properties: no KB loaded")?;
    let entity = intern(args[0].as_str()?);

    match kb.forward.get(&entity) {
        Some(props) => {
            let mut names: Vec<String> = props.keys().map(|s| resolve(*s)).collect();
            names.sort();
            Ok(Value::list(names.into_iter().map(Value::str).collect()))
        }
        None => Ok(Value::Nil),
    }
}

/// `(kb-search property pattern)` → list of entity labels where prop value contains pattern
pub fn bi_kb_search(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("kb-search: expected 2 args, got {}", args.len()));
    }
    let kb = get_kb().ok_or("kb-search: no KB loaded")?;
    let prop = intern(args[0].as_str()?);
    let pattern = args[1].as_str()?.to_lowercase();

    let entities = kb.by_property.get(&prop).cloned().unwrap_or_default();
    let mut results = Vec::new();

    for entity_sym in entities {
        if let Some(props) = kb.forward.get(&entity_sym) {
            if let Some(val) = props.get(&prop) {
                if val.to_lowercase().contains(&pattern) {
                    results.push(Value::str(resolve(entity_sym)));
                }
                if results.len() >= 100 {
                    break; // cap results
                }
            }
        }
    }

    Ok(Value::list(results))
}

/// `(kb-is-property value)` → list of property names where this value appears
pub fn bi_kb_is_property(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 1 {
        return Err(format!("kb-is-property: expected 1 arg, got {}", args.len()));
    }
    let kb = get_kb().ok_or("kb-is-property: no KB loaded")?;
    let value = args[0].as_str()?.to_lowercase();

    match kb.reverse.get(&value) {
        Some(entries) => {
            let mut prop_names: Vec<String> = entries
                .iter()
                .map(|(_, prop_sym)| resolve(*prop_sym))
                .collect();
            prop_names.sort();
            prop_names.dedup();
            Ok(Value::list(prop_names.into_iter().map(Value::str).collect()))
        }
        None => Ok(Value::list(vec![])),
    }
}

/// `(kb-count)` → number of entities in the KB
pub fn bi_kb_count(args: &[Value], _env: &Env) -> Result<Value, String> {
    if !args.is_empty() {
        return Err(format!("kb-count: expected 0 args, got {}", args.len()));
    }
    let kb = get_kb().ok_or("kb-count: no KB loaded")?;
    Ok(Value::Int(kb.forward.len() as i64))
}

/// `(kb-filter property value)` → list of entity labels where property exactly equals value
/// Useful for category queries: (kb-filter "instance_of" "chemical element")
pub fn bi_kb_filter(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("kb-filter: expected 2 args, got {}", args.len()));
    }
    let kb = get_kb().ok_or("kb-filter: no KB loaded")?;
    let prop = intern(args[0].as_str()?);
    let target = args[1].as_str()?.to_lowercase();

    let entities = kb.by_property.get(&prop).cloned().unwrap_or_default();
    let mut results = Vec::new();

    for entity_sym in entities {
        if let Some(props) = kb.forward.get(&entity_sym) {
            if let Some(val) = props.get(&prop) {
                if val.to_lowercase() == target {
                    results.push(Value::str(resolve(entity_sym)));
                }
            }
        }
    }

    results.sort_by(|a, b| {
        let sa = if let Value::Str(s) = a { s.as_ref() } else { "" };
        let sb = if let Value::Str(s) = b { s.as_ref() } else { "" };
        sa.cmp(sb)
    });
    Ok(Value::list(results))
}

/// `(kb-path entity1 entity2)` → shortest path as list of (entity property value) triples.
/// BFS over the graph: forward edges (entity→value via property) and reverse edges
/// (value→entity via property). Returns nil if no path within 6 hops.
///
/// Example: (kb-path "Paris" "Oxygen") might return
///   (("Paris" "capital" "France") ("France" "nationality" "Marie Curie") ...)
pub fn bi_kb_path(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() != 2 {
        return Err(format!("kb-path: expected 2 args, got {}", args.len()));
    }
    let kb = get_kb().ok_or("kb-path: no KB loaded")?;
    let start = args[0].as_str()?;
    let goal = args[1].as_str()?;
    let max_depth = 6;

    let path = bfs_path(kb, start, goal, max_depth);
    match path {
        Some(steps) => {
            let triples: Vec<Value> = steps
                .into_iter()
                .map(|(from, prop, to)| {
                    Value::list(vec![Value::str(from), Value::str(prop), Value::str(to)])
                })
                .collect();
            Ok(Value::list(triples))
        }
        None => Ok(Value::Nil),
    }
}

/// `(kb-path-count entity1 entity2 max-depth)` → number of distinct shortest paths.
/// Returns 0 if unreachable within max-depth hops.
pub fn bi_kb_path_count(args: &[Value], _env: &Env) -> Result<Value, String> {
    if args.len() < 2 || args.len() > 3 {
        return Err(format!("kb-path-count: expected 2-3 args, got {}", args.len()));
    }
    let kb = get_kb().ok_or("kb-path-count: no KB loaded")?;
    let start = args[0].as_str()?;
    let goal = args[1].as_str()?;
    let max_depth = if args.len() == 3 {
        match &args[2] {
            Value::Int(n) => *n as usize,
            _ => 4,
        }
    } else {
        4
    };

    let count = count_paths(kb, start, goal, max_depth);
    Ok(Value::Int(count as i64))
}

// ── Graph algorithms ───────────────────────────────────────────────────────

use std::collections::{HashSet, VecDeque};

/// BFS shortest path. Each node is a string (entity label or property value).
/// Edges: entity →[property]→ value (forward) and value →[property]→ entity (reverse).
fn bfs_path(
    kb: &KnowledgeBase,
    start: &str,
    goal: &str,
    max_depth: usize,
) -> Option<Vec<(String, String, String)>> {
    if start == goal {
        return Some(vec![]);
    }

    let start_lc = start.to_lowercase();
    let goal_lc = goal.to_lowercase();

    // BFS state: (current_node_lowercase, path_so_far)
    let mut queue: VecDeque<(String, Vec<(String, String, String)>)> = VecDeque::new();
    let mut visited: HashSet<String> = HashSet::new();

    queue.push_back((start_lc.clone(), vec![]));
    visited.insert(start_lc.clone());

    while let Some((current, path)) = queue.pop_front() {
        if path.len() >= max_depth {
            continue;
        }

        // Forward edges: if current is an entity, follow its properties
        let current_sym = intern(&current);
        if let Some(props) = kb.forward.get(&current_sym) {
            for (prop_sym, val) in props {
                let val_lc = val.to_lowercase();
                if val_lc == goal_lc {
                    let mut result = path.clone();
                    result.push((resolve(current_sym), resolve(*prop_sym), val.clone()));
                    return Some(result);
                }
                if !visited.contains(&val_lc) {
                    visited.insert(val_lc.clone());
                    let mut new_path = path.clone();
                    new_path.push((resolve(current_sym), resolve(*prop_sym), val.clone()));
                    queue.push_back((val_lc, new_path));
                }
            }
        }

        // Reverse edges: if current appears as a value, find entities that have it
        if let Some(entries) = kb.reverse.get(&current) {
            for (entity_sym, prop_sym) in entries {
                let entity_lc = resolve(*entity_sym).to_lowercase();
                if entity_lc == goal_lc {
                    let mut result = path.clone();
                    result.push((
                        resolve(*entity_sym),
                        resolve(*prop_sym),
                        current.clone(),
                    ));
                    return Some(result);
                }
                if !visited.contains(&entity_lc) {
                    visited.insert(entity_lc.clone());
                    let mut new_path = path.clone();
                    new_path.push((
                        resolve(*entity_sym),
                        resolve(*prop_sym),
                        current.clone(),
                    ));
                    queue.push_back((entity_lc, new_path));
                }
            }
        }
    }

    None
}

/// Count distinct paths up to max_depth (DFS with memoization).
/// Capped at 1000 to avoid explosion.
fn count_paths(
    kb: &KnowledgeBase,
    start: &str,
    goal: &str,
    max_depth: usize,
) -> usize {
    if start.to_lowercase() == goal.to_lowercase() {
        return 1;
    }
    if max_depth == 0 {
        return 0;
    }

    let mut count = 0usize;
    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(start.to_lowercase());

    count_paths_dfs(kb, &start.to_lowercase(), &goal.to_lowercase(), max_depth, &mut visited, &mut count);
    count
}

fn count_paths_dfs(
    kb: &KnowledgeBase,
    current: &str,
    goal: &str,
    remaining: usize,
    visited: &mut HashSet<String>,
    count: &mut usize,
) {
    if remaining == 0 || *count >= 1000 {
        return;
    }

    let current_sym = intern(current);

    // Forward edges
    if let Some(props) = kb.forward.get(&current_sym) {
        for (_, val) in props {
            let val_lc = val.to_lowercase();
            if val_lc == *goal {
                *count += 1;
                if *count >= 1000 { return; }
                continue;
            }
            if !visited.contains(&val_lc) {
                visited.insert(val_lc.clone());
                count_paths_dfs(kb, &val_lc, goal, remaining - 1, visited, count);
                visited.remove(&val_lc);
            }
        }
    }

    // Reverse edges
    if let Some(entries) = kb.reverse.get(current) {
        for (entity_sym, _) in entries {
            let entity_lc = resolve(*entity_sym).to_lowercase();
            if entity_lc == *goal {
                *count += 1;
                if *count >= 1000 { return; }
                continue;
            }
            if !visited.contains(&entity_lc) {
                visited.insert(entity_lc.clone());
                count_paths_dfs(kb, &entity_lc, goal, remaining - 1, visited, count);
                visited.remove(&entity_lc);
            }
        }
    }
}
