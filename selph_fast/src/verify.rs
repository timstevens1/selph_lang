//! Spec verification and reward computation for SELPH.
//!
//! Implements the mechanical verification pipeline from the SELPH spec:
//!   - Type match (hard gate)
//!   - Shape match (hard gate)
//!   - Constraint satisfaction (hard gate)
//!   - Goal satisfaction (Levels 0-3: fully mechanical for 0-2)
//!   - Reward computation
//!
//! This module is the "spec is the critic" system: no separate critic model
//! is needed for Stages 0-2. The spec defines what success looks like, and
//! verification is purely mechanical.

use std::rc::Rc;
use crate::intern::intern;
use crate::eval::{apply, make_default_env, value_to_string};
use crate::types::{Env, Node, Value};

// ── Verification result ──────────────────────────────────────────────

/// Structured result from verifying a program against a spec.
///
/// This is the output that feeds into the reward computation and
/// the critic/retry loop.
#[derive(Clone, Debug)]
pub struct VerificationResult {
    /// Hard gate: does the output type match the spec's expected type?
    pub type_match: bool,
    /// Hard gate: does the output shape match the spec's expected shape?
    pub shape_match: bool,
    /// Hard gate: do all constraints pass?
    pub constraint_match: bool,
    /// Goal satisfaction score (0.0 to 1.0).
    pub goal_score: f64,
    /// Detailed diagnostics.
    pub type_error: Option<String>,
    pub shape_error: Option<String>,
    pub constraint_errors: Vec<String>,
    pub goal_details: Option<String>,
}

impl Default for VerificationResult {
    fn default() -> Self {
        Self {
            type_match: true,
            shape_match: true,
            constraint_match: true,
            goal_score: 1.0,
            type_error: None,
            shape_error: None,
            constraint_errors: Vec::new(),
            goal_details: None,
        }
    }
}

impl VerificationResult {
    pub fn new() -> Self {
        Self::default()
    }

    /// All hard constraints must pass for any reward.
    pub fn hard_gate(&self) -> bool {
        self.type_match && self.shape_match && self.constraint_match
    }

    /// Full pass: hard gate and goal score == 1.0.
    pub fn passed(&self) -> bool {
        self.hard_gate() && (self.goal_score - 1.0).abs() < f64::EPSILON
    }
}

/// Compute a scalar reward from the verification result.
///
/// reward = hard_gate * (w_goal * goal_score + w_parent * parent_success + w_global * global_outcome)
pub fn reward(
    result: &VerificationResult,
    w_goal: f64,
    w_parent: f64,
    w_global: f64,
    parent_success: f64,
    global_outcome: f64,
) -> f64 {
    if !result.hard_gate() {
        return 0.0;
    }
    w_goal * result.goal_score + w_parent * parent_success + w_global * global_outcome
}

/// Convenience: compute reward with default weights (0.7 / 0.2 / 0.1).
pub fn reward_default(result: &VerificationResult) -> f64 {
    reward(result, 0.7, 0.2, 0.1, 0.0, 0.0)
}

// ── Spec representation ──────────────────────────────────────────────

/// A lightweight spec representation for verification.
///
/// In the full SELPH system the spec is parsed from S-expressions. Here
/// we use a Rust-native representation that can be constructed directly
/// or converted from the AST.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Expected output type name (e.g. "number", "string", "list").
    pub expected_type: Option<String>,
    /// Expected shape dimensions (for lists/tensors).
    pub expected_shape: Option<Vec<ShapeDim>>,
    /// Named constraints to check against the output.
    pub constraints: Vec<Constraint>,
    /// The goal to verify.
    pub goal: Option<Goal>,
}

impl Spec {
    pub fn new() -> Self {
        Self {
            expected_type: None,
            expected_shape: None,
            constraints: Vec::new(),
            goal: None,
        }
    }
}

/// A single dimension in a shape specification.
#[derive(Clone, Debug)]
pub enum ShapeDim {
    /// A fixed numeric dimension.
    Fixed(usize),
    /// A symbolic/wildcard dimension (always passes at runtime).
    Symbolic(String),
}

/// A named constraint.
#[derive(Clone, Debug)]
pub enum Constraint {
    MaxLength(usize),
    MinLength(usize),
    NonEmpty,
    OneOf(Vec<Value>),
    /// Regex match (for strings).
    Matches(String),
    /// A predicate closure to apply to the output.
    Predicate(Value),
}

/// Goal types corresponding to verification levels.
#[derive(Clone, Debug)]
pub enum Goal {
    /// Level 0: input/output example pairs. All-or-nothing scoring.
    Examples(Vec<(Value, Value)>),
    /// Level 1: pattern matching with keyword constraints.
    Pattern(Vec<PatternCheck>),
    /// Level 2: predicate satisfaction. The Value must be a Closure or Builtin.
    Satisfy(Value),
    /// Level 3: intent (placeholder/stub).
    Intent(String),
    /// Composite: all sub-goals must pass; score is the average.
    All(Vec<Goal>),
}

/// A single check within a Level 1 pattern goal.
#[derive(Clone, Debug)]
pub enum PatternCheck {
    Length(usize),
    StartsWith(String),
    EndsWith(String),
    Contains(String),
    Min(f64),
    Max(f64),
    Type(String),
}

// ── Value comparison ─────────────────────────────────────────────────

/// Deep equality for SELPH values. Value does not derive PartialEq
/// because closures are not meaningfully comparable, but for verification
/// we need to compare data values.
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => (x - y).abs() < f64::EPSILON,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        (Value::List(xs), Value::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(a, b)| values_equal(a, b))
        }
        _ => false,
    }
}

// ── Type helpers ─────────────────────────────────────────────────────

/// Get the SELPH type name for a runtime value.
pub fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Num(_) => "number",
        Value::Str(_) => "string",
        Value::Bool(_) => "bool",
        Value::List(_) => "list",
        Value::Nil => "nil",
        Value::Closure(..) => "function",
        Value::Builtin(_) => "function",
        Value::RustMacro(..) => "function",
        Value::Namespace(_) => "namespace",
        Value::Alt(_) => "alt",
    }
}

// ── Shape helpers ────────────────────────────────────────────────────

/// Infer the shape of a value. Returns dimensions for nested lists.
pub fn infer_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::List(items) => {
            let mut dims = vec![items.len()];
            // Check if all items are lists of equal length for further dims
            if let Some(Value::List(first)) = items.first() {
                let inner_len = first.len();
                let uniform = items.iter().all(|v| {
                    matches!(v, Value::List(l) if l.len() == inner_len)
                });
                if uniform && !items.is_empty() {
                    // Recurse on the first element to get inner dims
                    let inner_dims = infer_shape(&items[0]);
                    dims.extend_from_slice(&inner_dims);
                }
            }
            dims
        }
        _ => Vec::new(),
    }
}

// ── Main verification entry point ────────────────────────────────────

/// Verify a value against a spec.
///
/// Runs the full verification pipeline: type match, shape match,
/// constraint satisfaction, goal satisfaction.
pub fn verify(spec: &Spec, output: &Value, env: &mut Env) -> VerificationResult {
    let mut result = VerificationResult::new();

    // 1. Type match
    if let Some(ref expected) = spec.expected_type {
        check_type(expected, output, &mut result);
    }

    // 2. Shape match
    if let Some(ref expected_shape) = spec.expected_shape {
        check_shape(expected_shape, output, &mut result);
    }

    // 3. Constraint satisfaction
    if !spec.constraints.is_empty() {
        check_constraints(&spec.constraints, output, env, &mut result);
    }

    // 4. Goal satisfaction
    if let Some(ref goal) = spec.goal {
        check_goal(goal, output, env, &mut result);
    }

    result
}

/// Verify a function against a spec with example pairs.
///
/// Runs the function on each input example and checks the output.
pub fn verify_fn(spec: &Spec, program_fn: &Value, env: &mut Env) -> VerificationResult {
    let mut result = VerificationResult::new();

    if let Some(ref goal) = spec.goal {
        match goal {
            Goal::Examples(pairs) => {
                check_goal_examples_fn(pairs, program_fn, &mut result);
            }
            Goal::All(sub_goals) => {
                check_goal_all_fn(sub_goals, program_fn, env, &mut result);
            }
            _ => {
                result.goal_score = 0.0;
                result.goal_details = Some(
                    "verify_fn requires Examples goal; use verify() for other goal types"
                        .to_string(),
                );
            }
        }
    }

    // Also check type if present and we have example outputs
    if let Some(ref expected) = spec.expected_type {
        if let Some(Goal::Examples(pairs)) = &spec.goal {
            if let Some((input, _)) = pairs.first() {
                let nodes: Rc<[Node]> = Vec::<Node>::new().into();
                if let Ok(first_output) = apply(program_fn, &[input.clone()], &nodes, env) {
                    check_type(expected, &first_output, &mut result);
                }
            }
        }
    }

    result
}

// ── Type checking ────────────────────────────────────────────────────

fn check_type(expected: &str, output: &Value, result: &mut VerificationResult) {
    let actual = value_type_name(output);
    if expected != actual {
        result.type_match = false;
        result.type_error = Some(format!("expected type {}, got {}", expected, actual));
    }
}

// ── Shape checking ───────────────────────────────────────────────────

fn check_shape(
    expected_dims: &[ShapeDim],
    output: &Value,
    result: &mut VerificationResult,
) {
    match output {
        Value::List(items) => {
            check_list_shape(items, expected_dims, 0, result);
        }
        _ => {
            result.shape_match = false;
            result.shape_error = Some(format!(
                "expected shaped value, got {}",
                value_type_name(output)
            ));
        }
    }
}

fn check_list_shape(
    items: &[Value],
    dims: &[ShapeDim],
    depth: usize,
    result: &mut VerificationResult,
) {
    if dims.is_empty() {
        return;
    }

    match &dims[0] {
        ShapeDim::Fixed(expected_len) => {
            if items.len() != *expected_len {
                result.shape_match = false;
                result.shape_error = Some(format!(
                    "dimension {}: expected {}, got {}",
                    depth, expected_len, items.len()
                ));
                return;
            }
        }
        ShapeDim::Symbolic(_) => {
            // Symbolic dims always pass at runtime
        }
    }

    // Check inner dimensions recursively
    if dims.len() > 1 && !items.is_empty() {
        for (i, item) in items.iter().enumerate() {
            match item {
                Value::List(inner) => {
                    check_list_shape(inner, &dims[1..], depth + 1, result);
                    if !result.shape_match {
                        return;
                    }
                }
                _ => {
                    result.shape_match = false;
                    result.shape_error = Some(format!(
                        "expected nested list at dimension {}, got {} at index {}",
                        depth + 1,
                        value_type_name(item),
                        i
                    ));
                    return;
                }
            }
        }
    }
}

// ── Constraint checking ──────────────────────────────────────────────

fn check_constraints(
    constraints: &[Constraint],
    output: &Value,
    env: &mut Env,
    result: &mut VerificationResult,
) {
    for constraint in constraints {
        check_one_constraint(constraint, output, env, result);
    }
}

fn check_one_constraint(
    constraint: &Constraint,
    output: &Value,
    env: &mut Env,
    result: &mut VerificationResult,
) {
    match constraint {
        Constraint::MaxLength(max_len) => {
            let actual_len = value_length(output);
            if actual_len > *max_len {
                result.constraint_match = false;
                result.constraint_errors.push(format!(
                    "max-length {} exceeded: actual length {}",
                    max_len, actual_len
                ));
            }
        }
        Constraint::MinLength(min_len) => {
            let actual_len = value_length(output);
            if actual_len < *min_len {
                result.constraint_match = false;
                result.constraint_errors.push(format!(
                    "min-length {} not met: actual length {}",
                    min_len, actual_len
                ));
            }
        }
        Constraint::NonEmpty => match output {
            Value::Str(s) if s.is_empty() => {
                result.constraint_match = false;
                result.constraint_errors
                    .push("non-empty constraint violated: value is empty".to_string());
            }
            Value::List(l) if l.is_empty() => {
                result.constraint_match = false;
                result.constraint_errors
                    .push("non-empty constraint violated: value is empty".to_string());
            }
            Value::Nil => {
                result.constraint_match = false;
                result.constraint_errors
                    .push("non-empty constraint violated: value is nil".to_string());
            }
            _ => {}
        },
        Constraint::OneOf(allowed) => {
            let found = allowed.iter().any(|a| values_equal(output, a));
            if !found {
                result.constraint_match = false;
                result.constraint_errors.push(format!(
                    "one-of constraint: {} not in allowed values",
                    value_to_string(output)
                ));
            }
        }
        Constraint::Matches(pattern) => {
            // Simple pattern matching without a regex crate.
            // Supports basic character-class patterns like [a-z]+ and literal strings.
            // For full regex support, add `regex` or `regex-lite` to Cargo.toml
            // and replace this with a proper regex engine.
            if let Value::Str(s) = output {
                if !simple_pattern_match(pattern, s) {
                    result.constraint_match = false;
                    result.constraint_errors.push(format!(
                        "regex constraint: {:?} does not match {:?}",
                        s, pattern
                    ));
                }
            } else {
                result.constraint_match = false;
                result.constraint_errors.push(format!(
                    "regex constraint requires string, got {}",
                    value_type_name(output)
                ));
            }
        }
        Constraint::Predicate(pred_val) => {
            let nodes: Rc<[Node]> = Vec::<Node>::new().into();
            match apply(pred_val, &[output.clone()], &nodes, env) {
                Ok(Value::Bool(true)) => {}
                Ok(Value::Bool(false)) => {
                    result.constraint_match = false;
                    result
                        .constraint_errors
                        .push("constraint predicate returned false".to_string());
                }
                Ok(_) => {
                    result.constraint_match = false;
                    result
                        .constraint_errors
                        .push("constraint predicate did not return a bool".to_string());
                }
                Err(e) => {
                    result.constraint_match = false;
                    result
                        .constraint_errors
                        .push(format!("constraint evaluation error: {}", e));
                }
            }
        }
    }
}

/// Simple pattern matching for the Matches constraint.
///
/// This is a lightweight implementation that handles common patterns:
/// - Literal strings: exact match
/// - `[a-z]+`, `[A-Z]+`, `[0-9]+`, `[a-zA-Z]+`: character class repetition
/// - `.*`: match anything
///
/// For full regex support, add `regex` or `regex-lite` to Cargo.toml.
fn simple_pattern_match(pattern: &str, value: &str) -> bool {
    // Exact literal match
    if pattern == value {
        return true;
    }
    // .* matches anything
    if pattern == ".*" {
        return true;
    }
    // .+ matches anything non-empty
    if pattern == ".+" {
        return !value.is_empty();
    }
    // Character class patterns: [a-z]+, [A-Z]+, [0-9]+, [a-zA-Z]+, [a-zA-Z0-9]+
    if pattern.starts_with('[') {
        if let Some(bracket_end) = pattern.find(']') {
            let class = &pattern[1..bracket_end];
            let quantifier = &pattern[bracket_end + 1..];
            let requires_one_or_more = quantifier == "+";
            let allows_zero_or_more = quantifier == "*";

            if requires_one_or_more || allows_zero_or_more {
                if requires_one_or_more && value.is_empty() {
                    return false;
                }
                let char_ok = |c: char| -> bool {
                    let mut i = 0;
                    let bytes = class.as_bytes();
                    while i < bytes.len() {
                        if i + 2 < bytes.len() && bytes[i + 1] == b'-' {
                            let lo = bytes[i] as char;
                            let hi = bytes[i + 2] as char;
                            if c >= lo && c <= hi {
                                return true;
                            }
                            i += 3;
                        } else {
                            if c == bytes[i] as char {
                                return true;
                            }
                            i += 1;
                        }
                    }
                    false
                };
                return value.chars().all(char_ok);
            }
        }
    }
    // Fallback: treat pattern as a literal and check equality (already checked above,
    // so this is unreachable, but included for clarity).
    false
}

/// Get the length of a string or list value, or 0 for other types.
fn value_length(v: &Value) -> usize {
    match v {
        Value::Str(s) => s.len(),
        Value::List(l) => l.len(),
        _ => 0,
    }
}

// ── Goal checking ────────────────────────────────────────────────────

fn check_goal(goal: &Goal, output: &Value, env: &mut Env, result: &mut VerificationResult) {
    match goal {
        Goal::Examples(pairs) => {
            check_goal_examples_value(pairs, output, result);
        }
        Goal::Pattern(checks) => {
            check_goal_pattern(checks, output, result);
        }
        Goal::Satisfy(predicate) => {
            check_goal_satisfy(predicate, output, env, result);
        }
        Goal::Intent(description) => {
            // Level 3: stub -- cannot be verified mechanically
            result.goal_score = 0.0;
            result.goal_details = Some(format!(
                "level 3 intent goal requires model-as-judge: {}",
                description
            ));
        }
        Goal::All(sub_goals) => {
            check_goal_all(sub_goals, output, env, result);
        }
    }
}

// --- Level 0: Examples ---

/// Level 0 (value mode): check if the output matches any expected output.
fn check_goal_examples_value(
    pairs: &[(Value, Value)],
    output: &Value,
    result: &mut VerificationResult,
) {
    let matched = pairs.iter().any(|(_, expected)| values_equal(output, expected));
    if matched {
        result.goal_score = 1.0;
        result.goal_details = Some("level 0: matched".to_string());
    } else {
        result.goal_score = 0.0;
        result.goal_details = Some(format!(
            "level 0: output {} did not match any expected value",
            value_to_string(output)
        ));
    }
}

/// Level 0 (function mode): run the function on each input and check outputs.
fn check_goal_examples_fn(
    pairs: &[(Value, Value)],
    program_fn: &Value,
    result: &mut VerificationResult,
) {
    let total = pairs.len();
    let mut passed = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let nodes: Rc<[Node]> = Vec::<Node>::new().into();
    let mut env = make_default_env();

    for (input, expected) in pairs {
        match apply(program_fn, &[input.clone()], &nodes, &mut env) {
            Ok(actual) => {
                if values_equal(&actual, expected) {
                    passed += 1;
                } else {
                    failures.push(format!(
                        "input={}, expected={}, actual={}",
                        value_to_string(input),
                        value_to_string(expected),
                        value_to_string(&actual)
                    ));
                }
            }
            Err(e) => {
                failures.push(format!(
                    "input={}, expected={}, error={}",
                    value_to_string(input),
                    value_to_string(expected),
                    e
                ));
            }
        }
    }

    result.goal_score = if total > 0 {
        passed as f64 / total as f64
    } else {
        1.0
    };
    result.goal_details = Some(format!(
        "level 0: {}/{} examples passed{}",
        passed,
        total,
        if failures.is_empty() {
            String::new()
        } else {
            format!("; failures: [{}]", failures.join(", "))
        }
    ));
}

// --- Level 1: Patterns ---

fn check_goal_pattern(
    checks: &[PatternCheck],
    output: &Value,
    result: &mut VerificationResult,
) {
    let total = checks.len();
    let mut passed = 0usize;
    let mut errors: Vec<String> = Vec::new();

    for check in checks {
        match check {
            PatternCheck::Length(expected_len) => match output {
                Value::Str(s) => {
                    if s.len() == *expected_len {
                        passed += 1;
                    } else {
                        errors.push(format!(
                            "length: expected {}, got {}",
                            expected_len,
                            s.len()
                        ));
                    }
                }
                Value::List(l) => {
                    if l.len() == *expected_len {
                        passed += 1;
                    } else {
                        errors.push(format!(
                            "length: expected {}, got {}",
                            expected_len,
                            l.len()
                        ));
                    }
                }
                _ => {
                    errors.push(format!(
                        "length check requires string or list, got {}",
                        value_type_name(output)
                    ));
                }
            },
            PatternCheck::StartsWith(prefix) => {
                if let Value::Str(s) = output {
                    if s.starts_with(prefix.as_str()) {
                        passed += 1;
                    } else {
                        errors.push(format!("starts-with: expected prefix {:?}", prefix));
                    }
                } else {
                    errors.push(format!("starts-with: expected string, got {}", value_type_name(output)));
                }
            }
            PatternCheck::EndsWith(suffix) => {
                if let Value::Str(s) = output {
                    if s.ends_with(suffix.as_str()) {
                        passed += 1;
                    } else {
                        errors.push(format!("ends-with: expected suffix {:?}", suffix));
                    }
                } else {
                    errors.push(format!("ends-with: expected string, got {}", value_type_name(output)));
                }
            }
            PatternCheck::Contains(needle) => match output {
                Value::Str(s) => {
                    if s.contains(needle.as_str()) {
                        passed += 1;
                    } else {
                        errors.push(format!("contains: {:?} not found", needle));
                    }
                }
                Value::List(items) => {
                    let needle_val = Value::Str(needle.clone());
                    if items.iter().any(|v| values_equal(v, &needle_val)) {
                        passed += 1;
                    } else {
                        errors.push(format!("contains: {:?} not found in list", needle));
                    }
                }
                _ => {
                    errors.push(format!("contains: expected string or list, got {}", value_type_name(output)));
                }
            },
            PatternCheck::Min(min_val) => {
                if let Value::Num(n) = output {
                    if *n >= *min_val {
                        passed += 1;
                    } else {
                        errors.push(format!("min: expected >= {}, got {}", min_val, n));
                    }
                } else {
                    errors.push(format!("min: expected number, got {}", value_type_name(output)));
                }
            }
            PatternCheck::Max(max_val) => {
                if let Value::Num(n) = output {
                    if *n <= *max_val {
                        passed += 1;
                    } else {
                        errors.push(format!("max: expected <= {}, got {}", max_val, n));
                    }
                } else {
                    errors.push(format!("max: expected number, got {}", value_type_name(output)));
                }
            }
            PatternCheck::Type(expected_type) => {
                let actual = value_type_name(output);
                if expected_type == actual {
                    passed += 1;
                } else {
                    errors.push(format!("type: expected {}, got {}", expected_type, actual));
                }
            }
        }
    }

    result.goal_score = if total > 0 {
        passed as f64 / total as f64
    } else {
        1.0
    };
    result.goal_details = Some(format!(
        "level 1: {}/{} pattern checks passed{}",
        passed,
        total,
        if errors.is_empty() {
            String::new()
        } else {
            format!("; errors: [{}]", errors.join(", "))
        }
    ));
}

// --- Level 2: Satisfy (predicates) ---

fn check_goal_satisfy(
    predicate: &Value,
    output: &Value,
    env: &mut Env,
    result: &mut VerificationResult,
) {
    let nodes: Rc<[Node]> = Vec::<Node>::new().into();
    match apply(predicate, &[output.clone()], &nodes, env) {
        Ok(Value::Bool(true)) => {
            result.goal_score = 1.0;
            result.goal_details = Some("level 2: predicate satisfied".to_string());
        }
        Ok(Value::Bool(false)) => {
            result.goal_score = 0.0;
            result.goal_details = Some("level 2: predicate returned false".to_string());
        }
        Ok(other) => {
            result.goal_score = 0.0;
            result.goal_details = Some(format!(
                "level 2: predicate returned non-bool: {}",
                value_to_string(&other)
            ));
        }
        Err(e) => {
            result.goal_score = 0.0;
            result.goal_details = Some(format!("level 2: predicate error: {}", e));
        }
    }
}

// --- Goal composition: All ---

fn check_goal_all(
    sub_goals: &[Goal],
    output: &Value,
    env: &mut Env,
    result: &mut VerificationResult,
) {
    let n = sub_goals.len();
    if n == 0 {
        result.goal_score = 1.0;
        result.goal_details = Some("all: no sub-goals".to_string());
        return;
    }

    let mut total_score = 0.0;
    let mut sub_details: Vec<String> = Vec::new();

    for sub_goal in sub_goals {
        let mut sub_result = VerificationResult::new();
        check_goal(sub_goal, output, env, &mut sub_result);
        total_score += sub_result.goal_score;
        sub_details.push(format!(
            "score={:.2}{}",
            sub_result.goal_score,
            sub_result
                .goal_details
                .map(|d| format!(" ({})", d))
                .unwrap_or_default()
        ));
    }

    result.goal_score = total_score / n as f64;
    result.goal_details = Some(format!("all: [{}]", sub_details.join(", ")));
}

fn check_goal_all_fn(
    sub_goals: &[Goal],
    program_fn: &Value,
    _env: &mut Env,
    result: &mut VerificationResult,
) {
    let n = sub_goals.len();
    if n == 0 {
        result.goal_score = 1.0;
        result.goal_details = Some("all: no sub-goals".to_string());
        return;
    }

    let mut total_score = 0.0;
    let mut sub_details: Vec<String> = Vec::new();

    for sub_goal in sub_goals {
        let mut sub_result = VerificationResult::new();
        match sub_goal {
            Goal::Examples(pairs) => {
                check_goal_examples_fn(pairs, program_fn, &mut sub_result);
            }
            _ => {
                sub_result.goal_score = 0.0;
                sub_result.goal_details = Some(
                    "non-example goals in :all require verify() with output".to_string(),
                );
            }
        }
        total_score += sub_result.goal_score;
        sub_details.push(format!(
            "score={:.2}{}",
            sub_result.goal_score,
            sub_result
                .goal_details
                .map(|d| format!(" ({})", d))
                .unwrap_or_default()
        ));
    }

    result.goal_score = total_score / n as f64;
    result.goal_details = Some(format!("all: [{}]", sub_details.join(", ")));
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn num(n: f64) -> Value {
        Value::Num(n)
    }
    fn str_val(s: &str) -> Value {
        Value::Str(s.to_string())
    }
    fn bool_val(b: bool) -> Value {
        Value::Bool(b)
    }
    fn list_val(items: Vec<Value>) -> Value {
        Value::List(items)
    }

    // -- VerificationResult basics --

    #[test]
    fn test_default_result_passes() {
        let r = VerificationResult::new();
        assert!(r.hard_gate());
        assert!(r.passed());
    }

    #[test]
    fn test_hard_gate_fails_on_type_mismatch() {
        let mut r = VerificationResult::new();
        r.type_match = false;
        assert!(!r.hard_gate());
        assert!(!r.passed());
    }

    #[test]
    fn test_hard_gate_fails_on_shape_mismatch() {
        let mut r = VerificationResult::new();
        r.shape_match = false;
        assert!(!r.hard_gate());
    }

    #[test]
    fn test_hard_gate_fails_on_constraint() {
        let mut r = VerificationResult::new();
        r.constraint_match = false;
        assert!(!r.hard_gate());
    }

    #[test]
    fn test_passed_requires_goal_score_1() {
        let mut r = VerificationResult::new();
        r.goal_score = 0.5;
        assert!(r.hard_gate());
        assert!(!r.passed());
    }

    // -- Reward --

    #[test]
    fn test_reward_zero_when_hard_gate_fails() {
        let mut r = VerificationResult::new();
        r.type_match = false;
        r.goal_score = 1.0;
        assert_eq!(reward_default(&r), 0.0);
    }

    #[test]
    fn test_reward_with_defaults() {
        let r = VerificationResult::new(); // goal_score = 1.0
        let rw = reward_default(&r);
        // 0.7 * 1.0 + 0.2 * 0.0 + 0.1 * 0.0 = 0.7
        assert!((rw - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reward_with_parent_and_global() {
        let r = VerificationResult::new();
        let rw = reward(&r, 0.7, 0.2, 0.1, 1.0, 1.0);
        // 0.7 * 1.0 + 0.2 * 1.0 + 0.1 * 1.0 = 1.0
        assert!((rw - 1.0).abs() < f64::EPSILON);
    }

    // -- values_equal --

    #[test]
    fn test_values_equal_nums() {
        assert!(values_equal(&num(42.0), &num(42.0)));
        assert!(!values_equal(&num(42.0), &num(43.0)));
    }

    #[test]
    fn test_values_equal_strings() {
        assert!(values_equal(&str_val("hello"), &str_val("hello")));
        assert!(!values_equal(&str_val("hello"), &str_val("world")));
    }

    #[test]
    fn test_values_equal_lists() {
        let a = list_val(vec![num(1.0), num(2.0)]);
        let b = list_val(vec![num(1.0), num(2.0)]);
        let c = list_val(vec![num(1.0), num(3.0)]);
        assert!(values_equal(&a, &b));
        assert!(!values_equal(&a, &c));
    }

    #[test]
    fn test_values_equal_different_types() {
        assert!(!values_equal(&num(1.0), &str_val("1")));
        assert!(!values_equal(&bool_val(true), &num(1.0)));
    }

    // -- value_type_name --

    #[test]
    fn test_value_type_name() {
        assert_eq!(value_type_name(&num(0.0)), "number");
        assert_eq!(value_type_name(&str_val("")), "string");
        assert_eq!(value_type_name(&bool_val(true)), "bool");
        assert_eq!(value_type_name(&list_val(vec![])), "list");
        assert_eq!(value_type_name(&Value::Nil), "nil");
    }

    // -- infer_shape --

    #[test]
    fn test_infer_shape_flat_list() {
        let v = list_val(vec![num(1.0), num(2.0), num(3.0)]);
        assert_eq!(infer_shape(&v), vec![3]);
    }

    #[test]
    fn test_infer_shape_nested_list() {
        let v = list_val(vec![
            list_val(vec![num(1.0), num(2.0)]),
            list_val(vec![num(3.0), num(4.0)]),
        ]);
        assert_eq!(infer_shape(&v), vec![2, 2]);
    }

    #[test]
    fn test_infer_shape_non_list() {
        assert_eq!(infer_shape(&num(42.0)), Vec::<usize>::new());
    }

    // -- Type checking --

    #[test]
    fn test_type_match_passes() {
        let spec = Spec {
            expected_type: Some("number".to_string()),
            expected_shape: None,
            constraints: vec![],
            goal: None,
        };
        let mut env = make_default_env();
        let result = verify(&spec, &num(42.0), &mut env);
        assert!(result.type_match);
    }

    #[test]
    fn test_type_match_fails() {
        let spec = Spec {
            expected_type: Some("string".to_string()),
            expected_shape: None,
            constraints: vec![],
            goal: None,
        };
        let mut env = make_default_env();
        let result = verify(&spec, &num(42.0), &mut env);
        assert!(!result.type_match);
        assert!(result.type_error.is_some());
    }

    // -- Shape checking --

    #[test]
    fn test_shape_match_fixed() {
        let spec = Spec {
            expected_type: None,
            expected_shape: Some(vec![ShapeDim::Fixed(3)]),
            constraints: vec![],
            goal: None,
        };
        let mut env = make_default_env();
        let output = list_val(vec![num(1.0), num(2.0), num(3.0)]);
        let result = verify(&spec, &output, &mut env);
        assert!(result.shape_match);
    }

    #[test]
    fn test_shape_match_fails() {
        let spec = Spec {
            expected_type: None,
            expected_shape: Some(vec![ShapeDim::Fixed(5)]),
            constraints: vec![],
            goal: None,
        };
        let mut env = make_default_env();
        let output = list_val(vec![num(1.0), num(2.0), num(3.0)]);
        let result = verify(&spec, &output, &mut env);
        assert!(!result.shape_match);
    }

    #[test]
    fn test_shape_match_symbolic_passes() {
        let spec = Spec {
            expected_type: None,
            expected_shape: Some(vec![ShapeDim::Symbolic("n".to_string())]),
            constraints: vec![],
            goal: None,
        };
        let mut env = make_default_env();
        let output = list_val(vec![num(1.0)]);
        let result = verify(&spec, &output, &mut env);
        assert!(result.shape_match);
    }

    #[test]
    fn test_shape_match_nested() {
        let spec = Spec {
            expected_type: None,
            expected_shape: Some(vec![ShapeDim::Fixed(2), ShapeDim::Fixed(3)]),
            constraints: vec![],
            goal: None,
        };
        let mut env = make_default_env();
        let output = list_val(vec![
            list_val(vec![num(1.0), num(2.0), num(3.0)]),
            list_val(vec![num(4.0), num(5.0), num(6.0)]),
        ]);
        let result = verify(&spec, &output, &mut env);
        assert!(result.shape_match);
    }

    #[test]
    fn test_shape_non_list_fails() {
        let spec = Spec {
            expected_type: None,
            expected_shape: Some(vec![ShapeDim::Fixed(3)]),
            constraints: vec![],
            goal: None,
        };
        let mut env = make_default_env();
        let result = verify(&spec, &num(42.0), &mut env);
        assert!(!result.shape_match);
    }

    // -- Constraint checking --

    #[test]
    fn test_constraint_max_length_pass() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![Constraint::MaxLength(10)],
            goal: None,
        };
        let mut env = make_default_env();
        let result = verify(&spec, &str_val("hello"), &mut env);
        assert!(result.constraint_match);
    }

    #[test]
    fn test_constraint_max_length_fail() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![Constraint::MaxLength(3)],
            goal: None,
        };
        let mut env = make_default_env();
        let result = verify(&spec, &str_val("hello"), &mut env);
        assert!(!result.constraint_match);
    }

    #[test]
    fn test_constraint_min_length() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![Constraint::MinLength(3)],
            goal: None,
        };
        let mut env = make_default_env();
        assert!(verify(&spec, &str_val("hello"), &mut env).constraint_match);
        assert!(!verify(&spec, &str_val("hi"), &mut env).constraint_match);
    }

    #[test]
    fn test_constraint_non_empty() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![Constraint::NonEmpty],
            goal: None,
        };
        let mut env = make_default_env();
        assert!(verify(&spec, &str_val("hello"), &mut env).constraint_match);
        assert!(!verify(&spec, &str_val(""), &mut env).constraint_match);
        assert!(!verify(&spec, &Value::Nil, &mut env).constraint_match);
        assert!(!verify(&spec, &list_val(vec![]), &mut env).constraint_match);
    }

    #[test]
    fn test_constraint_one_of() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![Constraint::OneOf(vec![
                str_val("red"),
                str_val("green"),
                str_val("blue"),
            ])],
            goal: None,
        };
        let mut env = make_default_env();
        assert!(verify(&spec, &str_val("red"), &mut env).constraint_match);
        assert!(!verify(&spec, &str_val("yellow"), &mut env).constraint_match);
    }

    #[test]
    fn test_constraint_matches_regex() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![Constraint::Matches("[a-z]+".to_string())],
            goal: None,
        };
        let mut env = make_default_env();
        assert!(verify(&spec, &str_val("hello"), &mut env).constraint_match);
        assert!(!verify(&spec, &str_val("HELLO"), &mut env).constraint_match);
        assert!(!verify(&spec, &str_val("hello123"), &mut env).constraint_match);
    }

    #[test]
    fn test_multiple_constraints() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![
                Constraint::NonEmpty,
                Constraint::MaxLength(10),
                Constraint::MinLength(3),
            ],
            goal: None,
        };
        let mut env = make_default_env();
        assert!(verify(&spec, &str_val("hello"), &mut env).constraint_match);
        assert!(!verify(&spec, &str_val(""), &mut env).constraint_match);
        assert!(
            !verify(&spec, &str_val("hi"), &mut env).constraint_match,
            "min-length should fail for 'hi'"
        );
    }

    // -- Level 0: Examples (value mode) --

    #[test]
    fn test_goal_examples_value_match() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Examples(vec![
                (num(1.0), num(2.0)),
                (num(3.0), num(6.0)),
            ])),
        };
        let mut env = make_default_env();
        let result = verify(&spec, &num(2.0), &mut env);
        assert!((result.goal_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_goal_examples_value_no_match() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Examples(vec![
                (num(1.0), num(2.0)),
                (num(3.0), num(6.0)),
            ])),
        };
        let mut env = make_default_env();
        let result = verify(&spec, &num(99.0), &mut env);
        assert!((result.goal_score).abs() < f64::EPSILON);
    }

    // -- Level 0: Examples (function mode) --

    #[test]
    fn test_goal_examples_fn_all_pass() {
        // Build a closure that doubles its input: (lambda (x) (* 2 x))
        // We construct nodes for: [Symbol("*"), Num(2.0), Symbol("x"), App([0,1,2])]
        let nodes = vec![
            Node::Symbol(intern("*")), // 0
            Node::Num(2.0),                // 1
            Node::Symbol(intern("x")), // 2
            Node::App(vec![0, 1, 2]),      // 3: (* 2 x)
        ];
        let env = make_default_env();
        let double_fn = Value::Closure(
            vec![intern("x")],
            3, // body is node 3
            env.clone(),
            nodes.into(),
            None,
        );

        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Examples(vec![
                (num(1.0), num(2.0)),
                (num(3.0), num(6.0)),
                (num(5.0), num(10.0)),
            ])),
        };
        let mut env = make_default_env();
        let result = verify_fn(&spec, &double_fn, &mut env);
        assert!(
            (result.goal_score - 1.0).abs() < f64::EPSILON,
            "all examples should pass, got score {}",
            result.goal_score
        );
    }

    #[test]
    fn test_goal_examples_fn_partial_pass() {
        // Build identity: (lambda (x) x)
        let nodes = vec![
            Node::Symbol(intern("x")), // 0
        ];
        let env = make_default_env();
        let identity_fn = Value::Closure(
            vec![intern("x")],
            0,
            env.clone(),
            nodes.into(),
            None,
        );

        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Examples(vec![
                (num(1.0), num(1.0)),  // pass (identity returns 1)
                (num(3.0), num(6.0)),  // fail (identity returns 3, not 6)
            ])),
        };
        let mut env = make_default_env();
        let result = verify_fn(&spec, &identity_fn, &mut env);
        assert!(
            (result.goal_score - 0.5).abs() < f64::EPSILON,
            "expected 0.5, got {}",
            result.goal_score
        );
    }

    // -- Level 1: Pattern --

    #[test]
    fn test_goal_pattern_all_pass() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Pattern(vec![
                PatternCheck::Length(5),
                PatternCheck::StartsWith("hel".to_string()),
                PatternCheck::EndsWith("lo".to_string()),
                PatternCheck::Contains("ell".to_string()),
                PatternCheck::Type("string".to_string()),
            ])),
        };
        let mut env = make_default_env();
        let result = verify(&spec, &str_val("hello"), &mut env);
        assert!(
            (result.goal_score - 1.0).abs() < f64::EPSILON,
            "all pattern checks should pass, got {}",
            result.goal_score
        );
    }

    #[test]
    fn test_goal_pattern_partial() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Pattern(vec![
                PatternCheck::Length(5),
                PatternCheck::StartsWith("xyz".to_string()), // fails
            ])),
        };
        let mut env = make_default_env();
        let result = verify(&spec, &str_val("hello"), &mut env);
        assert!(
            (result.goal_score - 0.5).abs() < f64::EPSILON,
            "1/2 checks pass, got {}",
            result.goal_score
        );
    }

    #[test]
    fn test_goal_pattern_min_max() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Pattern(vec![
                PatternCheck::Min(0.0),
                PatternCheck::Max(100.0),
            ])),
        };
        let mut env = make_default_env();
        assert!((verify(&spec, &num(50.0), &mut env).goal_score - 1.0).abs() < f64::EPSILON);
        assert!((verify(&spec, &num(150.0), &mut env).goal_score - 0.5).abs() < f64::EPSILON);
    }

    // -- Level 2: Satisfy --

    #[test]
    fn test_goal_satisfy_passes() {
        // Build predicate: (lambda (x) (> x 0))
        let nodes = vec![
            Node::Symbol(intern(">")),  // 0
            Node::Symbol(intern("x")),  // 1
            Node::Num(0.0),                 // 2
            Node::App(vec![0, 1, 2]),       // 3: (> x 0)
        ];
        let env = make_default_env();
        let pred = Value::Closure(
            vec![intern("x")],
            3,
            env.clone(),
            nodes.into(),
            None,
        );

        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Satisfy(pred)),
        };
        let mut env = make_default_env();
        let result = verify(&spec, &num(42.0), &mut env);
        assert!(
            (result.goal_score - 1.0).abs() < f64::EPSILON,
            "predicate should be satisfied"
        );
    }

    #[test]
    fn test_goal_satisfy_fails() {
        // Build predicate: (lambda (x) (> x 0))
        let nodes = vec![
            Node::Symbol(intern(">")),
            Node::Symbol(intern("x")),
            Node::Num(0.0),
            Node::App(vec![0, 1, 2]),
        ];
        let env = make_default_env();
        let pred = Value::Closure(
            vec![intern("x")],
            3,
            env.clone(),
            nodes.into(),
            None,
        );

        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Satisfy(pred)),
        };
        let mut env = make_default_env();
        let result = verify(&spec, &num(-5.0), &mut env);
        assert!(
            result.goal_score.abs() < f64::EPSILON,
            "predicate should fail for negative"
        );
    }

    // -- Level 3: Intent (stub) --

    #[test]
    fn test_goal_intent_stub() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Intent("produce a friendly greeting".to_string())),
        };
        let mut env = make_default_env();
        let result = verify(&spec, &str_val("hello"), &mut env);
        // Intent goals cannot be verified mechanically; score should be 0.
        assert!(result.goal_score.abs() < f64::EPSILON);
        assert!(result.goal_details.as_ref().unwrap().contains("level 3"));
    }

    // -- Goal::All composition --

    #[test]
    fn test_goal_all_average() {
        let spec = Spec {
            expected_type: None,
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::All(vec![
                // This will match (42 is in expected outputs)
                Goal::Examples(vec![(num(1.0), num(42.0))]),
                // This will NOT match (42 is not "hello")
                Goal::Examples(vec![(num(1.0), str_val("hello"))]),
            ])),
        };
        let mut env = make_default_env();
        let result = verify(&spec, &num(42.0), &mut env);
        // One sub-goal scores 1.0, the other 0.0 => average 0.5
        assert!(
            (result.goal_score - 0.5).abs() < f64::EPSILON,
            "expected 0.5, got {}",
            result.goal_score
        );
    }

    // -- Full pipeline --

    #[test]
    fn test_full_pipeline_pass() {
        let spec = Spec {
            expected_type: Some("list".to_string()),
            expected_shape: Some(vec![ShapeDim::Fixed(3)]),
            constraints: vec![Constraint::NonEmpty],
            goal: Some(Goal::Examples(vec![(
                Value::Nil,
                list_val(vec![num(1.0), num(2.0), num(3.0)]),
            )])),
        };
        let mut env = make_default_env();
        let output = list_val(vec![num(1.0), num(2.0), num(3.0)]);
        let result = verify(&spec, &output, &mut env);
        assert!(result.passed(), "full pipeline should pass");
        assert!((reward_default(&result) - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_full_pipeline_type_gate_blocks() {
        let spec = Spec {
            expected_type: Some("string".to_string()),
            expected_shape: None,
            constraints: vec![],
            goal: Some(Goal::Examples(vec![(num(1.0), num(42.0))])),
        };
        let mut env = make_default_env();
        // Output is a number but spec expects string => type gate fails => reward 0
        let result = verify(&spec, &num(42.0), &mut env);
        assert!(!result.type_match);
        assert_eq!(reward_default(&result), 0.0);
    }
}
