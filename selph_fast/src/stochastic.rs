//! Stochastic process benchmarks for the SELPH curriculum.
//!
//! Defines processes with known information-theoretic limits, frames them
//! as SELPH benchmark tasks, and provides Bayes-optimal baselines.
//!
//! Process hierarchy (maps to curriculum stages):
//!
//!   Stage 0 -- Deterministic sequences:
//!     Fixed rules like modular arithmetic, periodic patterns.
//!     Optimal predictor is the rule itself. Entropy rate = 0.
//!
//!   Stage 1 -- First-order Markov chains:
//!     Next state depends only on current state. Transition matrix is known.
//!     Optimal predictor: empirical transition frequencies -> entropy rate.
//!
//!   Stage 2 -- Higher-order / piecewise processes:
//!     Second-order deterministic, piecewise linear, threshold functions.
//!     Optimal predictor: needs more context or branching logic.

use crate::types::Value;

// ── Benchmark result ─────────────────────────────────────────────────

/// A single benchmark task with its optimality bound.
#[derive(Clone, Debug)]
pub struct BenchmarkTask {
    pub name: String,
    pub stage: u32,
    pub process_type: String,
    pub optimal_accuracy: f64,
    pub entropy_rate: f64,
    pub reference_program: Option<String>,
    pub inputs: Vec<Value>,
    pub expected: Vec<Value>,
}

// ── Deterministic sequence generators (Stage 0) ─────────────────────

/// Generate a modular arithmetic sequence: x[n+1] = (x[n] + step) % modulus.
pub fn gen_modular_arithmetic(start: i64, step: i64, modulus: i64, length: usize) -> Vec<i64> {
    let mut seq = Vec::with_capacity(length);
    if length == 0 {
        return seq;
    }
    seq.push(start);
    for _ in 1..length {
        let next = (seq.last().unwrap() + step).rem_euclid(modulus);
        seq.push(next);
    }
    seq
}

/// Generate the Fibonacci sequence mod `modulus`.
pub fn gen_fibonacci_mod(modulus: i64, length: usize) -> Vec<i64> {
    let mut seq = Vec::with_capacity(length);
    if length == 0 {
        return seq;
    }
    seq.push(0);
    if length == 1 {
        return seq;
    }
    seq.push(1);
    for _ in 2..length {
        let n = seq.len();
        let next = (seq[n - 1] + seq[n - 2]).rem_euclid(modulus);
        seq.push(next);
    }
    seq
}

/// Generate the parity (0/1) of the Collatz sequence starting from `start`.
pub fn gen_collatz_parity(start: i64, length: usize) -> Vec<i64> {
    let mut vals = Vec::with_capacity(length);
    if length == 0 {
        return vals;
    }
    vals.push(start);
    for _ in 1..length {
        let n = *vals.last().unwrap();
        if n <= 1 {
            vals.push(1);
        } else if n % 2 == 0 {
            vals.push(n / 2);
        } else {
            vals.push(3 * n + 1);
        }
    }
    vals.iter().map(|x| x.rem_euclid(2)).collect()
}

/// Repeat a fixed pattern to produce a sequence of given length.
pub fn gen_periodic_sequence(pattern: &[i64], length: usize) -> Vec<i64> {
    (0..length).map(|i| pattern[i % pattern.len()]).collect()
}

// ── Markov chain generators (Stage 1) ───────────────────────────────

/// A simple seeded linear-congruential RNG for reproducibility.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return a float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // LCG parameters (Numerical Recipes)
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use upper bits for quality
        let upper = (self.state >> 11) as f64;
        upper / ((1u64 << 53) as f64)
    }

    /// Return a sample from N(0, std).
    fn next_gaussian(&mut self, std: f64) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z * std
    }
}

/// Generate a sequence from a first-order Markov chain.
///
/// `transition_matrix[i][j]` = P(next=j | current=i).
pub fn gen_markov_chain(
    transition_matrix: &[Vec<f64>],
    initial_state: usize,
    length: usize,
    seed: u64,
) -> Vec<i64> {
    let mut rng = SimpleRng::new(seed);
    let mut seq = Vec::with_capacity(length);
    if length == 0 {
        return seq;
    }
    seq.push(initial_state as i64);
    for _ in 1..length {
        let current = *seq.last().unwrap() as usize;
        let probs = &transition_matrix[current];
        let r = rng.next_f64();
        let mut cumulative = 0.0;
        let mut chosen = probs.len() - 1;
        for (j, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                chosen = j;
                break;
            }
        }
        seq.push(chosen as i64);
    }
    seq
}

/// Generate an i.i.d. Bernoulli (biased coin) sequence.
///
/// Each element is 1 with probability `p`, 0 otherwise.
pub fn gen_iid_coin(p: f64, length: usize, seed: u64) -> Vec<i64> {
    let tm = vec![vec![1.0 - p, p], vec![1.0 - p, p]];
    gen_markov_chain(&tm, 0, length, seed)
}

// ── Information-theoretic functions ──────────────────────────────────

/// Compute the stationary distribution of a Markov chain via power iteration.
fn stationary_distribution(transition_matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = transition_matrix.len();
    let mut pi = vec![1.0 / n as f64; n];

    for _ in 0..1000 {
        let mut new_pi = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                new_pi[j] += pi[i] * transition_matrix[i][j];
            }
        }
        pi = new_pi;
    }
    pi
}

/// Compute the entropy rate of a Markov chain (bits per symbol).
///
/// H = sum_i pi_i * sum_j -p_ij * log2(p_ij)
/// where pi is the stationary distribution.
pub fn markov_entropy_rate(transition_matrix: &[Vec<f64>]) -> f64 {
    let n = transition_matrix.len();
    let pi = stationary_distribution(transition_matrix);

    let mut h = 0.0;
    for i in 0..n {
        for j in 0..n {
            let p = transition_matrix[i][j];
            if p > 0.0 && pi[i] > 0.0 {
                h -= pi[i] * p * p.log2();
            }
        }
    }
    h
}

/// Compute the best possible prediction accuracy for a Markov chain.
///
/// The optimal predictor always predicts the most likely next state.
/// accuracy = sum_i pi_i * max_j p_ij
pub fn markov_optimal_accuracy(transition_matrix: &[Vec<f64>]) -> f64 {
    let n = transition_matrix.len();
    let pi = stationary_distribution(transition_matrix);

    let mut accuracy = 0.0;
    for i in 0..n {
        let max_p = transition_matrix[i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        accuracy += pi[i] * max_p;
    }
    accuracy
}

// ── Helper: sequence to SELPH input format ──────────────────────────

/// Convert a sequence into SELPH-format input/expected pairs.
///
/// Each input string is "idx v0 v1 v2 v3" where idx is the position and
/// v0..v3 are a sliding window of context. The expected output is the next value.
/// `context_len` controls the window size (default 4).
pub fn seq_to_selph_io(seq: &[i64], context_len: usize) -> (Vec<Value>, Vec<Value>) {
    let mut inputs = Vec::new();
    let mut expected = Vec::new();

    if seq.len() <= context_len {
        return (inputs, expected);
    }

    for i in context_len..seq.len() {
        let mut parts = vec![i.to_string()];
        for j in 0..context_len {
            parts.push(seq[i - context_len + j].to_string());
        }
        let input_str = parts.join(" ");
        inputs.push(Value::Str(input_str));
        expected.push(Value::Num(seq[i] as f64));
    }

    (inputs, expected)
}

/// Convert a sequence to mapping-style pairs (state -> most-likely-next-state).
///
/// For deterministic sequences, each unique (current, next) pair becomes one example.
fn seq_to_mapping_io(seq: &[i64]) -> (Vec<Value>, Vec<Value>) {
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for i in 0..seq.len().saturating_sub(1) {
        let key = (seq[i], seq[i + 1]);
        if seen.insert(key) {
            inputs.push(Value::Num(seq[i] as f64));
            expected.push(Value::Num(seq[i + 1] as f64));
        }
    }

    (inputs, expected)
}

/// For a Markov chain, create mapping from state -> argmax next state.
fn markov_deterministic_io(transition_matrix: &[Vec<f64>]) -> (Vec<Value>, Vec<Value>) {
    let n = transition_matrix.len();
    let mut inputs = Vec::new();
    let mut expected = Vec::new();

    for i in 0..n {
        let best_j = transition_matrix[i]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j)
            .unwrap_or(0);
        inputs.push(Value::Num(i as f64));
        expected.push(Value::Num(best_j as f64));
    }

    (inputs, expected)
}

// ── Stage builders ──────────────────────────────────────────────────

fn build_stage0_benchmarks() -> Vec<BenchmarkTask> {
    let mut tasks = Vec::new();

    // Modular arithmetic sequences
    for &(step, modulus) in &[(1i64, 5i64), (2, 7), (3, 10), (1, 3)] {
        let seq = gen_modular_arithmetic(0, step, modulus, 20);
        let (inputs, expected) = seq_to_mapping_io(&seq);
        tasks.push(BenchmarkTask {
            name: format!("mod_step{}_mod{}", step, modulus),
            stage: 0,
            process_type: "deterministic".to_string(),
            optimal_accuracy: 1.0,
            entropy_rate: 0.0,
            reference_program: Some(format!(
                "(lambda (x) (modulo (add x {}) {}))",
                step, modulus
            )),
            inputs,
            expected,
        });
    }

    // Periodic sequences
    for pattern in &[vec![0i64, 1], vec![0, 1, 2], vec![1, 0, 1, 0, 2]] {
        let seq = gen_periodic_sequence(pattern, 30);
        let (inputs, expected) = seq_to_mapping_io(&seq);
        tasks.push(BenchmarkTask {
            name: format!("periodic_{}", pattern.len()),
            stage: 0,
            process_type: "deterministic".to_string(),
            optimal_accuracy: 1.0,
            entropy_rate: 0.0,
            reference_program: None,
            inputs,
            expected,
        });
    }

    // Fibonacci mod
    for &modulus in &[3i64, 5] {
        let seq = gen_fibonacci_mod(modulus, 30);
        let (inputs, expected) = seq_to_selph_io(&seq, 1);
        tasks.push(BenchmarkTask {
            name: format!("fib_mod{}", modulus),
            stage: 0,
            process_type: "deterministic".to_string(),
            optimal_accuracy: 1.0,
            entropy_rate: 0.0,
            reference_program: None,
            inputs,
            expected,
        });
    }

    // Collatz parity
    let seq = gen_collatz_parity(27, 40);
    let (inputs, expected) = seq_to_selph_io(&seq, 4);
    tasks.push(BenchmarkTask {
        name: "collatz_parity_27".to_string(),
        stage: 0,
        process_type: "deterministic".to_string(),
        optimal_accuracy: 1.0,
        entropy_rate: 0.0,
        reference_program: None,
        inputs,
        expected,
    });

    tasks
}

fn build_stage1_benchmarks() -> Vec<BenchmarkTask> {
    let mut tasks = Vec::new();
    let mut seed_counter: u64 = 42;
    let mut next_seed = || {
        seed_counter += 1;
        seed_counter
    };

    // Nearly deterministic 2-state chain
    let tm1 = vec![vec![0.1, 0.9], vec![0.8, 0.2]];
    let _seq1 = gen_markov_chain(&tm1, 0, 50, next_seed());
    let (inputs, expected) = markov_deterministic_io(&tm1);
    tasks.push(BenchmarkTask {
        name: "markov_2state_biased".to_string(),
        stage: 1,
        process_type: "markov".to_string(),
        optimal_accuracy: markov_optimal_accuracy(&tm1),
        entropy_rate: markov_entropy_rate(&tm1),
        reference_program: Some("(lambda (x) (if (= x 0) 1 0))".to_string()),
        inputs,
        expected,
    });

    // 3-state cycle chain
    let tm2 = vec![
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.9, 0.05, 0.05],
    ];
    let _seq2 = gen_markov_chain(&tm2, 0, 50, next_seed());
    let (inputs, expected) = markov_deterministic_io(&tm2);
    tasks.push(BenchmarkTask {
        name: "markov_3state_cycle".to_string(),
        stage: 1,
        process_type: "markov".to_string(),
        optimal_accuracy: markov_optimal_accuracy(&tm2),
        entropy_rate: markov_entropy_rate(&tm2),
        reference_program: Some("(lambda (x) (modulo (add x 1) 3))".to_string()),
        inputs,
        expected,
    });

    // Biased coin (i.i.d.)
    let tm3 = vec![vec![0.3, 0.7], vec![0.3, 0.7]];
    let _seq3 = gen_markov_chain(&tm3, 0, 50, next_seed());
    let (inputs, expected) = markov_deterministic_io(&tm3);
    tasks.push(BenchmarkTask {
        name: "iid_biased_coin".to_string(),
        stage: 1,
        process_type: "markov".to_string(),
        optimal_accuracy: markov_optimal_accuracy(&tm3),
        entropy_rate: markov_entropy_rate(&tm3),
        reference_program: Some("(lambda (x) 1)".to_string()),
        inputs,
        expected,
    });

    // 4-state ring chain
    let tm4 = vec![
        vec![0.0, 0.9, 0.1, 0.0],
        vec![0.0, 0.0, 0.9, 0.1],
        vec![0.1, 0.0, 0.0, 0.9],
        vec![0.9, 0.1, 0.0, 0.0],
    ];
    let _seq4 = gen_markov_chain(&tm4, 0, 80, next_seed());
    let (inputs, expected) = markov_deterministic_io(&tm4);
    tasks.push(BenchmarkTask {
        name: "markov_4state_ring".to_string(),
        stage: 1,
        process_type: "markov".to_string(),
        optimal_accuracy: markov_optimal_accuracy(&tm4),
        entropy_rate: markov_entropy_rate(&tm4),
        reference_program: Some("(lambda (x) (modulo (add x 1) 4))".to_string()),
        inputs,
        expected,
    });

    tasks
}

fn build_stage2_benchmarks() -> Vec<BenchmarkTask> {
    let mut tasks = Vec::new();

    // Second-order deterministic: fib mod with encoded pairs
    for &modulus in &[4i64, 7] {
        let seq = gen_fibonacci_mod(modulus, 40);
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for i in 0..seq.len().saturating_sub(2) {
            let encoded_in = seq[i] * modulus + seq[i + 1];
            let out = seq[i + 2];
            let key = (encoded_in, out);
            if seen.insert(key) {
                inputs.push(Value::Num(encoded_in as f64));
                expected.push(Value::Num(out as f64));
            }
        }

        // Limit to 20 pairs as in Python
        inputs.truncate(20);
        expected.truncate(20);

        tasks.push(BenchmarkTask {
            name: format!("fib_mod{}_order2", modulus),
            stage: 2,
            process_type: "deterministic_order2".to_string(),
            optimal_accuracy: 1.0,
            entropy_rate: 0.0,
            reference_program: None,
            inputs,
            expected,
        });
    }

    // Piecewise linear: abs for negative, double for non-negative
    {
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for x in -5..=5 {
            let y = if x < 0 { -x } else { x * 2 };
            inputs.push(Value::Num(x as f64));
            expected.push(Value::Num(y as f64));
        }
        tasks.push(BenchmarkTask {
            name: "piecewise_abs_or_double".to_string(),
            stage: 2,
            process_type: "piecewise".to_string(),
            optimal_accuracy: 1.0,
            entropy_rate: 0.0,
            reference_program: Some("(lambda (x) (if (< x 0) (negate x) (add x x)))".to_string()),
            inputs,
            expected,
        });
    }

    // Threshold functions
    for &thresh in &[3, 0] {
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for x in -5..8 {
            let y = if x > thresh { 1.0 } else { 0.0 };
            inputs.push(Value::Num(x as f64));
            expected.push(Value::Num(y));
        }
        tasks.push(BenchmarkTask {
            name: format!("threshold_{}", thresh),
            stage: 2,
            process_type: "threshold".to_string(),
            optimal_accuracy: 1.0,
            entropy_rate: 0.0,
            reference_program: Some(format!("(lambda (x) (if (> x {}) 1 0))", thresh)),
            inputs,
            expected,
        });
    }

    tasks
}

// ── Full benchmark suite ─────────────────────────────────────────────

/// Generate the complete benchmark suite across all stages.
pub fn generate_benchmark_suite() -> Vec<BenchmarkTask> {
    let mut tasks = Vec::new();
    tasks.extend(build_stage0_benchmarks());
    tasks.extend(build_stage1_benchmarks());
    tasks.extend(build_stage2_benchmarks());
    tasks
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Deterministic generators ---

    #[test]
    fn test_modular_arithmetic() {
        let seq = gen_modular_arithmetic(0, 1, 5, 10);
        assert_eq!(seq, vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_modular_arithmetic_step2() {
        let seq = gen_modular_arithmetic(0, 2, 7, 8);
        assert_eq!(seq, vec![0, 2, 4, 6, 1, 3, 5, 0]);
    }

    #[test]
    fn test_fibonacci_mod() {
        let seq = gen_fibonacci_mod(3, 10);
        // Fib: 0,1,1,2,3,5,8,13,21,34  mod 3: 0,1,1,2,0,2,2,1,0,1
        assert_eq!(seq, vec![0, 1, 1, 2, 0, 2, 2, 1, 0, 1]);
    }

    #[test]
    fn test_fibonacci_mod_5() {
        let seq = gen_fibonacci_mod(5, 8);
        // Fib: 0,1,1,2,3,5,8,13 mod 5: 0,1,1,2,3,0,3,3
        assert_eq!(seq, vec![0, 1, 1, 2, 3, 0, 3, 3]);
    }

    #[test]
    fn test_collatz_parity() {
        // Collatz starting at 6: 6,3,10,5,16,8,4,2,1,1,...
        // Parity:                 0,1,0, 1,0, 0,0,0,1,1,...
        let seq = gen_collatz_parity(6, 10);
        assert_eq!(seq, vec![0, 1, 0, 1, 0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_periodic_sequence() {
        let seq = gen_periodic_sequence(&[0, 1, 2], 7);
        assert_eq!(seq, vec![0, 1, 2, 0, 1, 2, 0]);
    }

    // --- Markov chain generators ---

    #[test]
    fn test_markov_chain_deterministic() {
        // A fully deterministic transition: 0->1->0->1->...
        let tm = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let seq = gen_markov_chain(&tm, 0, 6, 42);
        assert_eq!(seq, vec![0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_markov_chain_length() {
        let tm = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let seq = gen_markov_chain(&tm, 0, 100, 42);
        assert_eq!(seq.len(), 100);
        // All values should be 0 or 1
        assert!(seq.iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn test_iid_coin_length() {
        let seq = gen_iid_coin(0.7, 50, 42);
        assert_eq!(seq.len(), 50);
        assert!(seq.iter().all(|&v| v == 0 || v == 1));
    }

    #[test]
    fn test_iid_coin_bias() {
        // With p=1.0, all generated transitions should be 1
        // (skip index 0 which is the initial state)
        let seq = gen_iid_coin(1.0, 20, 42);
        assert!(seq[1..].iter().all(|&v| v == 1));

        // With p=0.0, all generated transitions should be 0
        let seq = gen_iid_coin(0.0, 20, 42);
        assert!(seq[1..].iter().all(|&v| v == 0));
    }

    // --- Entropy rate ---

    #[test]
    fn test_entropy_rate_deterministic() {
        // Fully deterministic chain: entropy = 0
        let tm = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let h = markov_entropy_rate(&tm);
        assert!(h.abs() < 1e-10, "Expected ~0 entropy, got {}", h);
    }

    #[test]
    fn test_entropy_rate_fair_coin() {
        // Fair coin: H = 1.0 bit
        let tm = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let h = markov_entropy_rate(&tm);
        assert!((h - 1.0).abs() < 1e-6, "Expected ~1.0 bit, got {}", h);
    }

    #[test]
    fn test_entropy_rate_biased() {
        // Biased coin p=0.7: H = -0.7*log2(0.7) - 0.3*log2(0.3) ~ 0.8813
        let tm = vec![vec![0.3, 0.7], vec![0.3, 0.7]];
        let h = markov_entropy_rate(&tm);
        let expected = -0.7 * 0.7_f64.log2() - 0.3 * 0.3_f64.log2();
        assert!(
            (h - expected).abs() < 1e-6,
            "Expected ~{}, got {}",
            expected,
            h
        );
    }

    #[test]
    fn test_entropy_rate_3state_cycle() {
        // 2 states deterministic, 1 state with noise
        let tm = vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.9, 0.05, 0.05],
        ];
        let h = markov_entropy_rate(&tm);
        // Only state 2 contributes entropy; stationary dist is ~(1/3, 1/3, 1/3)
        assert!(h > 0.0, "Expected positive entropy, got {}", h);
        assert!(h < 1.0, "Expected < 1.0 bit, got {}", h);
    }

    // --- Optimal accuracy ---

    #[test]
    fn test_optimal_accuracy_deterministic() {
        let tm = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let acc = markov_optimal_accuracy(&tm);
        assert!((acc - 1.0).abs() < 1e-10, "Expected 1.0, got {}", acc);
    }

    #[test]
    fn test_optimal_accuracy_fair_coin() {
        let tm = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let acc = markov_optimal_accuracy(&tm);
        assert!((acc - 0.5).abs() < 1e-10, "Expected 0.5, got {}", acc);
    }

    #[test]
    fn test_optimal_accuracy_biased_coin() {
        let tm = vec![vec![0.3, 0.7], vec![0.3, 0.7]];
        let acc = markov_optimal_accuracy(&tm);
        assert!((acc - 0.7).abs() < 1e-6, "Expected 0.7, got {}", acc);
    }

    #[test]
    fn test_optimal_accuracy_2state_biased() {
        let tm = vec![vec![0.1, 0.9], vec![0.8, 0.2]];
        let acc = markov_optimal_accuracy(&tm);
        // pi ~ (8/17, 9/17), acc = (8/17)*0.9 + (9/17)*0.8 ~ 0.847
        assert!(acc > 0.8 && acc < 0.9, "Expected ~0.847, got {}", acc);
    }

    // --- SELPH I/O conversion ---

    #[test]
    fn test_seq_to_selph_io() {
        let seq = vec![10, 20, 30, 40, 50];
        let (inputs, expected) = seq_to_selph_io(&seq, 2);
        // With context_len=2, first input at index 2: "2 10 20", expected=30
        assert_eq!(inputs.len(), 3);
        assert_eq!(expected.len(), 3);

        if let Value::Str(ref s) = inputs[0] {
            assert_eq!(s, "2 10 20");
        } else {
            panic!("Expected Str value");
        }
        if let Value::Num(n) = expected[0] {
            assert!((n - 30.0).abs() < 1e-10);
        } else {
            panic!("Expected Num value");
        }
    }

    #[test]
    fn test_seq_to_mapping_io_dedup() {
        let seq = vec![0, 1, 0, 1, 0, 1];
        let (inputs, expected) = seq_to_mapping_io(&seq);
        // Only unique pairs: (0,1) and (1,0)
        assert_eq!(inputs.len(), 2);
        assert_eq!(expected.len(), 2);
    }

    // --- Full suite ---

    #[test]
    fn test_generate_benchmark_suite() {
        let suite = generate_benchmark_suite();
        assert!(!suite.is_empty(), "Suite should not be empty");

        // Check we have all three stages
        let stages: std::collections::HashSet<u32> = suite.iter().map(|t| t.stage).collect();
        assert!(stages.contains(&0), "Missing stage 0 tasks");
        assert!(stages.contains(&1), "Missing stage 1 tasks");
        assert!(stages.contains(&2), "Missing stage 2 tasks");

        // All stage 0 should be deterministic
        for task in suite.iter().filter(|t| t.stage == 0) {
            assert_eq!(task.optimal_accuracy, 1.0);
            assert_eq!(task.entropy_rate, 0.0);
        }

        // All stage 1 should be markov with bounded accuracy
        for task in suite.iter().filter(|t| t.stage == 1) {
            assert_eq!(task.process_type, "markov");
            assert!(task.optimal_accuracy <= 1.0);
            assert!(task.entropy_rate >= 0.0);
        }

        // Every task should have at least some inputs
        for task in &suite {
            assert!(
                !task.inputs.is_empty(),
                "Task '{}' has no inputs",
                task.name
            );
            assert_eq!(
                task.inputs.len(),
                task.expected.len(),
                "Task '{}' has mismatched input/expected lengths",
                task.name
            );
        }
    }

    #[test]
    fn test_suite_task_count() {
        let suite = generate_benchmark_suite();
        // Stage 0: 4 modular + 3 periodic + 2 fib + 1 collatz = 10
        // Stage 1: 4 markov chains
        // Stage 2: 2 fib_order2 + 1 piecewise + 2 threshold = 5
        let s0 = suite.iter().filter(|t| t.stage == 0).count();
        let s1 = suite.iter().filter(|t| t.stage == 1).count();
        let s2 = suite.iter().filter(|t| t.stage == 2).count();
        assert_eq!(s0, 10, "Expected 10 stage-0 tasks, got {}", s0);
        assert_eq!(s1, 4, "Expected 4 stage-1 tasks, got {}", s1);
        assert_eq!(s2, 5, "Expected 5 stage-2 tasks, got {}", s2);
    }
}
