//! SolveTrace instrumentation for the curriculum runner.
//!
//! Records the full decision tree for each task solve: which strategies
//! were tried, how many candidates each explored, accuracy (best partial
//! match fraction), and wall-clock time. Output as JSON via `--trace`.

/// Which strategy was attempted at a decision node.
#[derive(Debug, Clone)]
pub enum SolveStrategy {
    Flat {
        depth: usize,
        budget: usize,
    },
    BoolDecomposition {
        bool_macros_available: usize,
        budget: usize,
    },
    Induction {
        budget: usize,
        /// Human-readable description of the bridge function, if found.
        bridge: Option<String>,
    },
    DivideAndConquer {
        budget: usize,
        /// Number of distinct output value groups.
        output_groups: usize,
    },
    Memorization {
        /// Number of unique input→output entries in the lookup table.
        num_entries: usize,
    },
}

/// One node in the solve decision tree.
#[derive(Debug, Clone)]
pub struct SolveStep {
    pub strategy: SolveStrategy,
    pub candidates_explored: usize,
    pub succeeded: bool,
    /// Best partial match fraction seen (0.0–1.0). Key signal for
    /// policy learning: 0.8 means "close, try harder"; 0.0 means
    /// "wrong approach, try decomposing".
    pub best_match_fraction: f64,
    pub wall_time_ms: f64,
    /// Source code of the solution, if found.
    pub solution_source: Option<String>,
}

/// Features of the task at solve time — needed for counterfactual analysis.
#[derive(Debug, Clone)]
pub struct TaskFeatures {
    pub num_examples: usize,
    pub input_type: String,
    pub output_is_boolean: bool,
    pub num_distinct_outputs: usize,
    pub library_size: usize,
    pub bool_macros_available: usize,
}

/// Per-task trace: features, the sequence of strategies tried, and outcome.
#[derive(Debug, Clone)]
pub struct TaskTrace {
    pub task_name: String,
    pub task_depth: usize,
    pub features: TaskFeatures,
    pub steps: Vec<SolveStep>,
    pub solved: bool,
    /// Which strategy solved it: "Flat", "BD", "Induction", "D&C", "Memo", or null.
    pub solving_strategy: Option<String>,
    pub total_candidates: usize,
    pub total_wall_time_ms: f64,
    /// Components used in the solution (for meta-optimization training).
    pub components_used: Vec<String>,
    /// Number of components available at solve time.
    pub components_available: usize,
    /// Names of all synthesis components available at solve time.
    pub all_components_available: Vec<String>,
}

/// Top-level trace for the entire curriculum run.
#[derive(Debug, Clone)]
pub struct CurriculumTrace {
    pub tasks: Vec<TaskTrace>,
    pub total_solved: usize,
    pub total_tasks: usize,
    pub total_candidates: usize,
    pub budget: usize,
    pub default_depth: usize,
}

impl CurriculumTrace {
    pub fn new(budget: usize, default_depth: usize) -> Self {
        CurriculumTrace {
            tasks: Vec::new(),
            total_solved: 0,
            total_tasks: 0,
            total_candidates: 0,
            budget,
            default_depth,
        }
    }

    pub fn finalize(&mut self) {
        self.total_tasks = self.tasks.len();
        self.total_solved = self.tasks.iter().filter(|t| t.solved).count();
        self.total_candidates = self.tasks.iter().map(|t| t.total_candidates).sum();
    }

    /// Serialize the entire trace to JSON.
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"total_solved\": {},\n", self.total_solved));
        s.push_str(&format!("  \"total_tasks\": {},\n", self.total_tasks));
        s.push_str(&format!("  \"total_candidates\": {},\n", self.total_candidates));
        s.push_str(&format!("  \"budget\": {},\n", self.budget));
        s.push_str(&format!("  \"default_depth\": {},\n", self.default_depth));
        s.push_str("  \"tasks\": [\n");
        for (i, task) in self.tasks.iter().enumerate() {
            s.push_str(&task.to_json(4));
            if i + 1 < self.tasks.len() { s.push(','); }
            s.push('\n');
        }
        s.push_str("  ]\n");
        s.push('}');
        s
    }
}

fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn indent(n: usize) -> String {
    " ".repeat(n)
}

impl TaskTrace {
    pub fn to_json(&self, ind: usize) -> String {
        let i = indent(ind);
        let i2 = indent(ind + 2);
        let mut s = format!("{}{{\n", i);
        s.push_str(&format!("{}\"task_name\": \"{}\",\n", i2, json_escape(&self.task_name)));
        s.push_str(&format!("{}\"task_depth\": {},\n", i2, self.task_depth));
        s.push_str(&format!("{}\"solved\": {},\n", i2, self.solved));
        s.push_str(&format!("{}\"solving_strategy\": {},\n", i2,
            match &self.solving_strategy {
                Some(st) => format!("\"{}\"", json_escape(st)),
                None => "null".to_string(),
            }));
        s.push_str(&format!("{}\"total_candidates\": {},\n", i2, self.total_candidates));
        s.push_str(&format!("{}\"total_wall_time_ms\": {:.1},\n", i2, self.total_wall_time_ms));
        s.push_str(&format!("{}\"components_available\": {},\n", i2, self.components_available));
        // components_used
        s.push_str(&format!("{}\"components_used\": [", i2));
        for (j, c) in self.components_used.iter().enumerate() {
            s.push_str(&format!("\"{}\"", json_escape(c)));
            if j + 1 < self.components_used.len() { s.push_str(", "); }
        }
        s.push_str("],\n");
        // all_components_available
        s.push_str(&format!("{}\"all_components_available\": [", i2));
        for (j, c) in self.all_components_available.iter().enumerate() {
            s.push_str(&format!("\"{}\"", json_escape(c)));
            if j + 1 < self.all_components_available.len() { s.push_str(", "); }
        }
        s.push_str("],\n");
        // features
        s.push_str(&format!("{}\"features\": {{\n", i2));
        let i3 = indent(ind + 4);
        s.push_str(&format!("{}\"num_examples\": {},\n", i3, self.features.num_examples));
        s.push_str(&format!("{}\"input_type\": \"{}\",\n", i3, json_escape(&self.features.input_type)));
        s.push_str(&format!("{}\"output_is_boolean\": {},\n", i3, self.features.output_is_boolean));
        s.push_str(&format!("{}\"num_distinct_outputs\": {},\n", i3, self.features.num_distinct_outputs));
        s.push_str(&format!("{}\"library_size\": {},\n", i3, self.features.library_size));
        s.push_str(&format!("{}\"bool_macros_available\": {}\n", i3, self.features.bool_macros_available));
        s.push_str(&format!("{}}},\n", i2));
        // steps
        s.push_str(&format!("{}\"steps\": [\n", i2));
        for (j, step) in self.steps.iter().enumerate() {
            s.push_str(&step.to_json(ind + 4));
            if j + 1 < self.steps.len() { s.push(','); }
            s.push('\n');
        }
        s.push_str(&format!("{}]\n", i2));
        s.push_str(&format!("{}}}", i));
        s
    }
}

impl SolveStep {
    pub fn to_json(&self, ind: usize) -> String {
        let i = indent(ind);
        let i2 = indent(ind + 2);
        let mut s = format!("{}{{\n", i);
        // strategy
        match &self.strategy {
            SolveStrategy::Flat { depth, budget } => {
                s.push_str(&format!("{}\"strategy\": \"Flat\",\n", i2));
                s.push_str(&format!("{}\"depth\": {},\n", i2, depth));
                s.push_str(&format!("{}\"budget\": {},\n", i2, budget));
            }
            SolveStrategy::BoolDecomposition { bool_macros_available, budget } => {
                s.push_str(&format!("{}\"strategy\": \"BoolDecomposition\",\n", i2));
                s.push_str(&format!("{}\"bool_macros_available\": {},\n", i2, bool_macros_available));
                s.push_str(&format!("{}\"budget\": {},\n", i2, budget));
            }
            SolveStrategy::Induction { budget, bridge } => {
                s.push_str(&format!("{}\"strategy\": \"Induction\",\n", i2));
                s.push_str(&format!("{}\"budget\": {},\n", i2, budget));
                s.push_str(&format!("{}\"bridge\": {},\n", i2,
                    match bridge {
                        Some(b) => format!("\"{}\"", json_escape(b)),
                        None => "null".to_string(),
                    }));
            }
            SolveStrategy::DivideAndConquer { budget, output_groups } => {
                s.push_str(&format!("{}\"strategy\": \"DivideAndConquer\",\n", i2));
                s.push_str(&format!("{}\"budget\": {},\n", i2, budget));
                s.push_str(&format!("{}\"output_groups\": {},\n", i2, output_groups));
            }
            SolveStrategy::Memorization { num_entries } => {
                s.push_str(&format!("{}\"strategy\": \"Memorization\",\n", i2));
                s.push_str(&format!("{}\"num_entries\": {},\n", i2, num_entries));
            }
        }
        s.push_str(&format!("{}\"candidates_explored\": {},\n", i2, self.candidates_explored));
        s.push_str(&format!("{}\"succeeded\": {},\n", i2, self.succeeded));
        s.push_str(&format!("{}\"best_match_fraction\": {:.4},\n", i2, self.best_match_fraction));
        s.push_str(&format!("{}\"wall_time_ms\": {:.1},\n", i2, self.wall_time_ms));
        s.push_str(&format!("{}\"solution_source\": {}\n", i2,
            match &self.solution_source {
                Some(src) => format!("\"{}\"", json_escape(src)),
                None => "null".to_string(),
            }));
        s.push_str(&format!("{}}}", i));
        s
    }
}
