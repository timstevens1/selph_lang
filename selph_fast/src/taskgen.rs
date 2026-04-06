//! Automatic task generation for the SELPH 6-stage curriculum.
//!
//! Each task produces numeric sequence examples in the format
//! `"idx v0 v1 v2 v3"` (index plus a context window of 4 previous values)
//! paired with the expected next value (f64).
//!
//! Stages:
//!   0 — Constants          f(i) = c
//!   1 — Linear arithmetic  f(i) = a*i + b
//!   2 — Quadratic          f(i) = a*i^2 + b*i + c
//!   3 — Recurrence         e.g. Fibonacci, doubling
//!   4 — Compositions       using promoted macro library
//!   5 — Modular            f(i) = i mod k, alternating, etc.

// ── Core structs ────────────────────────────────────────────────────

/// A single curriculum task with input/output examples.
#[derive(Clone, Debug)]
pub struct Task {
    pub name: String,
    pub depth: usize,
    pub examples: Vec<(String, f64)>,
}

/// Configuration for one curriculum stage.
#[derive(Clone, Debug)]
pub struct StageConfig {
    pub stage: usize,
    pub task_count: usize,
    pub depth: usize,
    pub description: String,
}

/// An entry in the macro library available for Stage 4 composition tasks.
#[derive(Clone, Debug)]
pub struct LibraryEntry {
    pub name: String,
    /// A closure that computes the sequence value at index i.
    pub func: fn(i64) -> f64,
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Build the context-window input string for a sequence at a given index.
/// Format: "idx v0 v1 v2 v3" where v0..v3 are seq[idx-4]..seq[idx-1].
fn make_input(idx: usize, seq: &[f64]) -> String {
    let window_start = if idx >= 4 { idx - 4 } else { 0 };
    let window: Vec<f64> = seq[window_start..idx].to_vec();
    // Pad with zeros on the left if fewer than 4 previous values.
    let mut padded = vec![0.0_f64; 4 - window.len()];
    padded.extend(window);
    let parts: Vec<String> = std::iter::once(format_num(idx as f64))
        .chain(padded.iter().map(|v| format_num(*v)))
        .collect();
    parts.join(" ")
}

/// Format a number: integer form if whole, otherwise decimal.
fn format_num(n: f64) -> String {
    if n == (n as i64) as f64 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

/// Generate examples for a sequence defined by a function f(i) -> f64.
/// Produces examples for indices `start..start+count`, each with a
/// 4-value context window.
fn examples_from_fn<F: Fn(i64) -> f64>(
    f: &F,
    start: usize,
    count: usize,
) -> Vec<(String, f64)> {
    // Pre-compute enough of the sequence to fill context windows.
    let total = start + count;
    let seq: Vec<f64> = (0..total).map(|i| f(i as i64)).collect();
    let mut examples = Vec::new();
    for idx in start..total {
        let input = make_input(idx, &seq);
        let expected = seq[idx];
        examples.push((input, expected));
    }
    examples
}

/// Generate examples for a recurrence relation where each value depends
/// on previous values.  `seeds` are the initial values and `recur` computes
/// the next value given the full sequence so far.
fn examples_from_recurrence<F: Fn(&[f64]) -> f64>(
    seeds: &[f64],
    recur: &F,
    count: usize,
) -> Vec<(String, f64)> {
    let total = seeds.len() + count;
    let mut seq: Vec<f64> = seeds.to_vec();
    while seq.len() < total {
        let next = recur(&seq);
        seq.push(next);
    }
    let start = if seeds.len() >= 4 { seeds.len() } else { 4.min(total) };
    let mut examples = Vec::new();
    for idx in start..total {
        let input = make_input(idx, &seq);
        let expected = seq[idx];
        examples.push((input, expected));
    }
    examples
}

// ── Stage 0: Constants ──────────────────────────────────────────────

/// Generate tasks where f(i) = c for various constants.
pub fn gen_constant_tasks(count: usize) -> Vec<Task> {
    let constants: Vec<f64> = vec![
        0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, -1.0, -3.0, 42.0,
        100.0, 0.5, -5.0, 8.0, 13.0, 21.0, 99.0, 4.0, 6.0, 9.0,
    ];
    constants
        .iter()
        .take(count)
        .map(|&c| {
            let name = format!("const_{}", format_num(c));
            let examples = examples_from_fn(&|_i| c, 4, 5);
            Task { name, depth: 1, examples }
        })
        .collect()
}

// ── Stage 1: Linear arithmetic ──────────────────────────────────────

/// Generate tasks where f(i) = a*i + b.
pub fn gen_linear_tasks(count: usize) -> Vec<Task> {
    let params: Vec<(i64, i64, &str)> = vec![
        (1, 0, "identity"),        // f(i) = i
        (2, 0, "double_idx"),      // f(i) = 2i
        (3, 0, "triple_idx"),      // f(i) = 3i
        (1, 1, "idx_plus_1"),      // f(i) = i+1
        (1, -1, "idx_minus_1"),    // f(i) = i-1
        (2, 1, "2i_plus_1"),       // f(i) = 2i+1
        (2, 3, "2i_plus_3"),       // f(i) = 2i+3
        (3, 1, "3i_plus_1"),       // f(i) = 3i+1
        (-1, 10, "neg_i_plus_10"), // f(i) = -i+10
        (5, 0, "5i"),              // f(i) = 5i
        (1, 5, "idx_plus_5"),      // f(i) = i+5
        (4, 2, "4i_plus_2"),       // f(i) = 4i+2
        (-2, 20, "neg2i_plus_20"), // f(i) = -2i+20
        (1, 10, "idx_plus_10"),    // f(i) = i+10
        (10, 0, "10i"),            // f(i) = 10i
        (7, 3, "7i_plus_3"),       // f(i) = 7i+3
        (2, -1, "2i_minus_1"),     // f(i) = 2i-1
        (3, -2, "3i_minus_2"),     // f(i) = 3i-2
        (6, 1, "6i_plus_1"),       // f(i) = 6i+1
        (1, 100, "idx_plus_100"),  // f(i) = i+100
    ];
    params
        .iter()
        .take(count)
        .map(|&(a, b, name)| {
            let examples = examples_from_fn(&|i| (a * i + b) as f64, 4, 5);
            Task {
                name: name.to_string(),
                depth: 2,
                examples,
            }
        })
        .collect()
}

// ── Stage 2: Quadratic ──────────────────────────────────────────────

/// Generate tasks where f(i) = a*i^2 + b*i + c.
pub fn gen_quadratic_tasks(count: usize) -> Vec<Task> {
    let params: Vec<(i64, i64, i64, &str)> = vec![
        (1, 0, 0, "squares"),          // i^2
        (1, 1, 0, "i2_plus_i"),        // i^2 + i
        (1, 0, 1, "i2_plus_1"),        // i^2 + 1
        (2, 0, 0, "2i2"),              // 2i^2
        (1, -1, 0, "i2_minus_i"),      // i^2 - i
        (1, 0, -1, "i2_minus_1"),      // i^2 - 1
        (0, 1, 0, "triangular_raw"),   // i*(i+1)/2 handled separately
        (3, 0, 0, "3i2"),              // 3i^2
        (1, 2, 1, "i_plus_1_sq"),      // (i+1)^2
        (2, 1, 0, "2i2_plus_i"),       // 2i^2 + i
        (1, 0, 5, "i2_plus_5"),        // i^2 + 5
        (1, 3, 0, "i2_plus_3i"),       // i^2 + 3i
        (2, -1, 0, "2i2_minus_i"),     // 2i^2 - i
        (1, 1, 1, "i2_plus_i_plus_1"), // i^2 + i + 1
        (4, 0, 0, "4i2"),              // 4i^2
    ];

    let mut tasks: Vec<Task> = Vec::new();

    for &(a, b, c, name) in params.iter().take(count) {
        // Special case: triangular numbers use i*(i+1)/2
        if name == "triangular_raw" {
            let examples = examples_from_fn(&|i| (i * (i + 1)) as f64 / 2.0, 4, 5);
            tasks.push(Task {
                name: "triangular".to_string(),
                depth: 3,
                examples,
            });
        } else {
            let examples =
                examples_from_fn(&|i| (a * i * i + b * i + c) as f64, 4, 5);
            tasks.push(Task {
                name: name.to_string(),
                depth: 3,
                examples,
            });
        }
    }
    tasks
}

// ── Stage 3: Recurrence ─────────────────────────────────────────────

/// Generate recurrence-based tasks (Fibonacci, doubling, etc.).
pub fn gen_recurrence_tasks(count: usize) -> Vec<Task> {
    struct RecurrenceDef {
        name: &'static str,
        seeds: Vec<f64>,
        recur: fn(&[f64]) -> f64,
    }

    let defs: Vec<RecurrenceDef> = vec![
        RecurrenceDef {
            name: "fibonacci",
            seeds: vec![1.0, 1.0],
            recur: |s| s[s.len() - 1] + s[s.len() - 2],
        },
        RecurrenceDef {
            name: "double_prev",
            seeds: vec![1.0],
            recur: |s| s[s.len() - 1] * 2.0,
        },
        RecurrenceDef {
            name: "triple_prev",
            seeds: vec![1.0],
            recur: |s| s[s.len() - 1] * 3.0,
        },
        RecurrenceDef {
            name: "lucas",
            seeds: vec![2.0, 1.0],
            recur: |s| s[s.len() - 1] + s[s.len() - 2],
        },
        RecurrenceDef {
            name: "pell",
            seeds: vec![0.0, 1.0],
            recur: |s| 2.0 * s[s.len() - 1] + s[s.len() - 2],
        },
        RecurrenceDef {
            name: "tribonacci",
            seeds: vec![0.0, 0.0, 1.0],
            recur: |s| {
                let n = s.len();
                s[n - 1] + s[n - 2] + s[n - 3]
            },
        },
        RecurrenceDef {
            name: "add_last_two_plus_1",
            seeds: vec![1.0, 1.0],
            recur: |s| s[s.len() - 1] + s[s.len() - 2] + 1.0,
        },
        RecurrenceDef {
            name: "prev_plus_idx",
            seeds: vec![0.0],
            recur: |s| s[s.len() - 1] + s.len() as f64,
        },
        RecurrenceDef {
            name: "factorial_seq",
            seeds: vec![1.0],
            recur: |s| s[s.len() - 1] * s.len() as f64,
        },
        RecurrenceDef {
            name: "powers_of_2",
            seeds: vec![1.0],
            recur: |s| s[s.len() - 1] * 2.0,
        },
    ];

    defs.into_iter()
        .take(count)
        .map(|def| {
            let examples = examples_from_recurrence(&def.seeds, &def.recur, 8);
            Task {
                name: def.name.to_string(),
                depth: 3,
                examples,
            }
        })
        .collect()
}

// ── Stage 4: Compositions ───────────────────────────────────────────

/// Generate composition tasks that combine library functions.
///
/// `library` maps names to functions that compute sequence values.
/// Compositions chain two library functions, e.g. "double then add-1".
pub fn gen_composition_tasks(count: usize, library: &[LibraryEntry]) -> Vec<Task> {
    let mut tasks = Vec::new();

    // Also include some hard-coded curated compositions.
    let curated: Vec<(&str, Box<dyn Fn(i64) -> f64>)> = vec![
        ("cubes", Box::new(|i: i64| (i * i * i) as f64)),
        ("i4", Box::new(|i: i64| (i * i * i * i) as f64)),
        ("2i2_plus_3i_plus_1", Box::new(|i: i64| (2 * i * i + 3 * i + 1) as f64)),
        ("sum_first_i", Box::new(|i: i64| (i * (i + 1) / 2) as f64)),
        ("i_times_i_minus_1", Box::new(|i: i64| (i * (i - 1)) as f64)),
    ];

    for (name, f) in curated.iter() {
        if tasks.len() >= count {
            break;
        }
        let examples = examples_from_fn(f, 4, 5);
        tasks.push(Task {
            name: name.to_string(),
            depth: 3,
            examples,
        });
    }

    // Compose pairs from the library: f_b(f_a(i))
    'outer: for a in library {
        for b in library {
            if tasks.len() >= count {
                break 'outer;
            }
            if a.name == b.name {
                continue;
            }
            let fa = a.func;
            let fb = b.func;
            let name = format!("{}__then__{}", a.name, b.name);
            let examples = examples_from_fn(&|i| fb(fa(i) as i64), 4, 5);
            tasks.push(Task { name, depth: 3, examples });
        }
    }

    tasks.into_iter().take(count).collect()
}

// ── Stage 5: Modular ────────────────────────────────────────────────

/// Generate modular/periodic tasks.
pub fn gen_modular_tasks(count: usize) -> Vec<Task> {
    let mut tasks = Vec::new();

    // i mod k for various k
    for k in [2, 3, 4, 5, 7] {
        if tasks.len() >= count {
            break;
        }
        let name = format!("i_mod_{}", k);
        let examples = examples_from_fn(&|i| (i % k) as f64, 4, 6);
        tasks.push(Task { name, depth: 2, examples });
    }

    // Alternating patterns
    let alternating: Vec<(&str, Box<dyn Fn(i64) -> f64>)> = vec![
        ("alt_0_1", Box::new(|i: i64| (i % 2) as f64)),
        ("alt_1_neg1", Box::new(|i: i64| if i % 2 == 0 { 1.0 } else { -1.0 })),
        ("alt_0_1_2", Box::new(|i: i64| (i % 3) as f64)),
        ("sawtooth_5", Box::new(|i: i64| (i % 5) as f64)),
        ("triangle_wave_4", Box::new(|i: i64| {
            let m = i % 8;
            if m <= 4 { m as f64 } else { (8 - m) as f64 }
        })),
        ("step_3", Box::new(|i: i64| (i / 3) as f64)),
        ("step_5", Box::new(|i: i64| (i / 5) as f64)),
        ("even_flag", Box::new(|i: i64| if i % 2 == 0 { 1.0 } else { 0.0 })),
        ("odd_flag", Box::new(|i: i64| if i % 2 != 0 { 1.0 } else { 0.0 })),
        ("div3_flag", Box::new(|i: i64| if i % 3 == 0 { 1.0 } else { 0.0 })),
        ("abs_sin_period", Box::new(|i: i64| (i % 4 - 2).unsigned_abs() as f64)),
        ("zigzag_3", Box::new(|i: i64| {
            let m = i % 6;
            if m <= 3 { m as f64 } else { (6 - m) as f64 }
        })),
        ("square_mod3", Box::new(|i: i64| ((i * i) % 3) as f64)),
        ("cube_mod5", Box::new(|i: i64| ((i * i * i) % 5) as f64)),
        ("i_mod_i_plus_3", Box::new(|i: i64| if i + 3 == 0 { 0.0 } else { (i % (i + 3)) as f64 })),
    ];

    for (name, f) in alternating.iter() {
        if tasks.len() >= count {
            break;
        }
        let examples = examples_from_fn(f, 4, 6);
        tasks.push(Task {
            name: name.to_string(),
            depth: 2,
            examples,
        });
    }

    tasks.into_iter().take(count).collect()
}

// ── Curriculum assembly ─────────────────────────────────────────────

/// Default library entries for Stage 4 composition.
fn default_library() -> Vec<LibraryEntry> {
    vec![
        LibraryEntry { name: "double".into(), func: |i| (2 * i) as f64 },
        LibraryEntry { name: "triple".into(), func: |i| (3 * i) as f64 },
        LibraryEntry { name: "square".into(), func: |i| (i * i) as f64 },
        LibraryEntry { name: "add1".into(), func: |i| (i + 1) as f64 },
        LibraryEntry { name: "sub1".into(), func: |i| (i - 1) as f64 },
    ]
}

/// Generate the full curriculum from a list of stage configurations.
pub fn generate_curriculum(stages: &[StageConfig]) -> Vec<Task> {
    let library = default_library();
    let mut all_tasks = Vec::new();

    for cfg in stages {
        let mut stage_tasks = match cfg.stage {
            0 => gen_constant_tasks(cfg.task_count),
            1 => gen_linear_tasks(cfg.task_count),
            2 => gen_quadratic_tasks(cfg.task_count),
            3 => gen_recurrence_tasks(cfg.task_count),
            4 => gen_composition_tasks(cfg.task_count, &library),
            5 => gen_modular_tasks(cfg.task_count),
            _ => Vec::new(),
        };
        // Override depth from config.
        for t in &mut stage_tasks {
            t.depth = cfg.depth;
        }
        all_tasks.extend(stage_tasks);
    }
    all_tasks
}

/// Serialize a task to SELPH format for the CLI.
///
/// Produces:
/// ```text
/// (task "name" depth
///   ("idx v0 v1 v2 v3" expected) ...)
/// ```
pub fn task_to_selph(task: &Task) -> String {
    let mut lines = Vec::new();
    lines.push(format!("(task \"{}\" {}", task.name, task.depth));
    for (i, (input, expected)) in task.examples.iter().enumerate() {
        let trailing = if i + 1 < task.examples.len() { "" } else { ")" };
        lines.push(format!("  (\"{}\" {}){}",
            input,
            format_num(*expected),
            trailing,
        ));
    }
    lines.join("\n")
}

/// Default 6-stage curriculum matching the SELPH sequence-prediction design.
pub fn default_curriculum() -> Vec<Task> {
    let stages = vec![
        StageConfig {
            stage: 0,
            task_count: 5,
            depth: 2,
            description: "Constants: f(i) = c".into(),
        },
        StageConfig {
            stage: 1,
            task_count: 8,
            depth: 2,
            description: "Linear arithmetic: f(i) = a*i + b".into(),
        },
        StageConfig {
            stage: 2,
            task_count: 5,
            depth: 3,
            description: "Quadratic: f(i) = a*i^2 + b*i + c".into(),
        },
        StageConfig {
            stage: 3,
            task_count: 5,
            depth: 3,
            description: "Recurrence: Fibonacci, doubling, etc.".into(),
        },
        StageConfig {
            stage: 4,
            task_count: 5,
            depth: 3,
            description: "Compositions: chaining library functions".into(),
        },
        StageConfig {
            stage: 5,
            task_count: 5,
            depth: 2,
            description: "Modular: i mod k, alternating patterns".into(),
        },
    ];
    generate_curriculum(&stages)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_num() {
        assert_eq!(format_num(5.0), "5");
        assert_eq!(format_num(-3.0), "-3");
        assert_eq!(format_num(0.5), "0.5");
        assert_eq!(format_num(0.0), "0");
    }

    #[test]
    fn test_make_input_basic() {
        // seq = [10, 20, 30, 40, 50], idx=4 -> "4 10 20 30 40"
        let seq = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let input = make_input(4, &seq);
        assert_eq!(input, "4 10 20 30 40");
    }

    #[test]
    fn test_make_input_short_context() {
        // seq = [5, 10], idx=1 -> "1 0 0 0 5"
        let seq = vec![5.0, 10.0];
        let input = make_input(1, &seq);
        assert_eq!(input, "1 0 0 0 5");
    }

    #[test]
    fn test_stage0_constants() {
        let tasks = gen_constant_tasks(3);
        assert_eq!(tasks.len(), 3);
        // All examples for const_0 should have expected value 0.
        let t = &tasks[0]; // const_0
        assert_eq!(t.name, "const_0");
        for (_input, expected) in &t.examples {
            assert_eq!(*expected, 0.0);
        }
        // const_1
        let t1 = &tasks[1];
        assert_eq!(t1.name, "const_1");
        for (_input, expected) in &t1.examples {
            assert_eq!(*expected, 1.0);
        }
    }

    #[test]
    fn test_stage1_identity() {
        let tasks = gen_linear_tasks(1);
        assert_eq!(tasks.len(), 1);
        let t = &tasks[0];
        assert_eq!(t.name, "identity");
        // f(i) = i, starting at idx 4
        // idx=4 -> expected 4.0
        assert_eq!(t.examples[0].1, 4.0);
        assert_eq!(t.examples[1].1, 5.0);
    }

    #[test]
    fn test_stage1_double_idx() {
        let tasks = gen_linear_tasks(2);
        let t = &tasks[1]; // double_idx: f(i) = 2i
        assert_eq!(t.name, "double_idx");
        // idx=4 -> 8, idx=5 -> 10
        assert_eq!(t.examples[0].1, 8.0);
        assert_eq!(t.examples[1].1, 10.0);
    }

    #[test]
    fn test_stage1_input_format() {
        let tasks = gen_linear_tasks(1); // identity: f(i) = i
        let t = &tasks[0];
        // idx=4, seq = [0,1,2,3,4,...], context = [0,1,2,3]
        assert_eq!(t.examples[0].0, "4 0 1 2 3");
        // idx=5, context = [1,2,3,4]
        assert_eq!(t.examples[1].0, "5 1 2 3 4");
    }

    #[test]
    fn test_stage2_squares() {
        let tasks = gen_quadratic_tasks(1);
        let t = &tasks[0];
        assert_eq!(t.name, "squares");
        // f(4) = 16, f(5) = 25, f(6) = 36
        assert_eq!(t.examples[0].1, 16.0);
        assert_eq!(t.examples[1].1, 25.0);
        assert_eq!(t.examples[2].1, 36.0);
    }

    #[test]
    fn test_stage2_triangular() {
        let tasks = gen_quadratic_tasks(7); // triangular is the 7th entry
        let t = tasks.iter().find(|t| t.name == "triangular").expect("triangular task");
        // T(4) = 10, T(5) = 15, T(6) = 21
        assert_eq!(t.examples[0].1, 10.0);
        assert_eq!(t.examples[1].1, 15.0);
        assert_eq!(t.examples[2].1, 21.0);
    }

    #[test]
    fn test_stage3_fibonacci() {
        let tasks = gen_recurrence_tasks(1);
        let t = &tasks[0];
        assert_eq!(t.name, "fibonacci");
        // fib: 1,1,2,3,5,8,13,21,...
        // Starting examples from idx 4 onward.
        // seq[0..] = 1,1,2,3,5,8,13,21,...
        // idx=4 -> 5, idx=5 -> 8, idx=6 -> 13
        assert_eq!(t.examples[0].1, 5.0);
        assert_eq!(t.examples[1].1, 8.0);
        assert_eq!(t.examples[2].1, 13.0);
    }

    #[test]
    fn test_stage3_double_prev() {
        let tasks = gen_recurrence_tasks(2);
        let t = &tasks[1];
        assert_eq!(t.name, "double_prev");
        // seq: 1, 2, 4, 8, 16, 32, 64, 128, ...
        // idx=4 -> 16, idx=5 -> 32
        assert_eq!(t.examples[0].1, 16.0);
        assert_eq!(t.examples[1].1, 32.0);
    }

    #[test]
    fn test_stage4_cubes() {
        let library = default_library();
        let tasks = gen_composition_tasks(1, &library);
        let t = &tasks[0];
        assert_eq!(t.name, "cubes");
        // f(4) = 64, f(5) = 125
        assert_eq!(t.examples[0].1, 64.0);
        assert_eq!(t.examples[1].1, 125.0);
    }

    #[test]
    fn test_stage4_with_library() {
        let library = default_library();
        let tasks = gen_composition_tasks(10, &library);
        assert!(tasks.len() >= 5); // at least the curated ones
        // Verify composed tasks have examples
        for t in &tasks {
            assert!(!t.examples.is_empty(), "task {} has no examples", t.name);
        }
    }

    #[test]
    fn test_stage5_modular() {
        let tasks = gen_modular_tasks(3);
        assert_eq!(tasks.len(), 3);
        let t = &tasks[0]; // i_mod_2
        assert_eq!(t.name, "i_mod_2");
        // f(4) = 0, f(5) = 1, f(6) = 0
        assert_eq!(t.examples[0].1, 0.0);
        assert_eq!(t.examples[1].1, 1.0);
        assert_eq!(t.examples[2].1, 0.0);
    }

    #[test]
    fn test_stage5_alternating() {
        let tasks = gen_modular_tasks(10);
        let t = tasks.iter().find(|t| t.name == "alt_1_neg1").expect("alt_1_neg1 task");
        // Even indices -> 1, odd -> -1
        // idx=4 -> 1, idx=5 -> -1, idx=6 -> 1
        assert_eq!(t.examples[0].1, 1.0);
        assert_eq!(t.examples[1].1, -1.0);
        assert_eq!(t.examples[2].1, 1.0);
    }

    #[test]
    fn test_task_to_selph() {
        let task = Task {
            name: "test_task".to_string(),
            depth: 2,
            examples: vec![
                ("4 0 1 2 3".to_string(), 4.0),
                ("5 1 2 3 4".to_string(), 5.0),
            ],
        };
        let selph = task_to_selph(&task);
        assert!(selph.starts_with("(task \"test_task\" 2"));
        assert!(selph.contains("(\"4 0 1 2 3\" 4)"));
        assert!(selph.contains("(\"5 1 2 3 4\" 5)"));
        // Last example line should close the outer paren
        assert!(selph.ends_with(')'));
    }

    #[test]
    fn test_default_curriculum_structure() {
        let tasks = default_curriculum();
        // 5+8+5+5+5+5 = 33 tasks
        assert_eq!(tasks.len(), 33);
        // Every task must have at least one example
        for t in &tasks {
            assert!(!t.examples.is_empty(), "task {} has no examples", t.name);
        }
    }

    #[test]
    fn test_default_curriculum_matches_selph_format() {
        let tasks = default_curriculum();
        // Verify all examples match "idx v0 v1 v2 v3" format (5 space-separated numbers)
        for t in &tasks {
            for (input, _expected) in &t.examples {
                let parts: Vec<&str> = input.split(' ').collect();
                assert_eq!(parts.len(), 5,
                    "task {} input '{}' should have 5 parts", t.name, input);
                // Each part should parse as a number
                for p in &parts {
                    p.parse::<f64>().unwrap_or_else(|_|
                        panic!("task {} input part '{}' is not a number", t.name, p));
                }
            }
        }
    }

    #[test]
    fn test_generate_curriculum_custom() {
        let stages = vec![
            StageConfig {
                stage: 0,
                task_count: 2,
                depth: 1,
                description: "test constants".into(),
            },
            StageConfig {
                stage: 1,
                task_count: 3,
                depth: 2,
                description: "test linear".into(),
            },
        ];
        let tasks = generate_curriculum(&stages);
        assert_eq!(tasks.len(), 5);
        // First 2 should have depth 1 (overridden from config)
        assert_eq!(tasks[0].depth, 1);
        assert_eq!(tasks[1].depth, 1);
        // Next 3 should have depth 2
        assert_eq!(tasks[2].depth, 2);
    }

    #[test]
    fn test_selph_output_roundtrip_spot_check() {
        // Spot-check that task_to_selph produces output matching
        // the style of examples/sequence_tasks.selph.
        let tasks = gen_linear_tasks(1); // identity
        let selph = task_to_selph(&tasks[0]);
        // Should match the pattern from the example file.
        assert!(selph.contains("(task \"identity\" 2"),
            "unexpected selph output: {}", selph);
        assert!(selph.contains("(\"4 0 1 2 3\" 4)"),
            "identity idx=4 mismatch: {}", selph);
    }
}
