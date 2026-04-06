"""Stochastic process benchmarks for SELPH curriculum.

Defines processes with known information-theoretic limits, frames them
as SELPH specs, and provides Bayes-optimal baselines to measure against.

Process hierarchy (maps to curriculum stages):

  Stage 0 — Deterministic sequences:
    Fixed rules like modular arithmetic, periodic patterns.
    Optimal predictor is the rule itself. BPB = 0 for exact match.

  Stage 1 — First-order Markov chains:
    Next state depends only on current state. Transition matrix is known.
    Optimal predictor: empirical transition frequencies → entropy rate.

  Stage 2 — Higher-order / hidden processes:
    AR(1) with noise, HMMs, switching processes.
    Optimal predictor: forward algorithm / Kalman filter.

Each benchmark produces:
  - A set of SELPH specs (input/output examples from the process)
  - The Bayes-optimal prediction accuracy / bits-per-symbol
  - A reference SELPH program that achieves optimal (when expressible)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any
from .ast import Node, Symbol, Number, String, List, Spec, GoalExamples


# ── Benchmark result ─────────────────────────────────────────────────

@dataclass
class BenchmarkTask:
    """A single benchmark task with its optimality bound."""
    name: str
    spec: Spec
    process_type: str          # "deterministic", "markov", "hmm", etc.
    optimal_accuracy: float    # best possible accuracy (1.0 for deterministic)
    entropy_rate: float        # bits per symbol of the process
    reference_program: str | None = None  # SELPH source achieving optimal
    stage: int = 0


@dataclass
class BenchmarkSuite:
    """A collection of benchmark tasks with metadata."""
    name: str
    tasks: list[BenchmarkTask] = field(default_factory=list)
    description: str = ""

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    def tasks_for_stage(self, stage: int) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.stage == stage]


# ── Process generators ───────────────────────────────────────────────

# --- Deterministic sequences (Stage 0) ---

def generate_modular_sequence(start: int, step: int, modulus: int,
                              length: int) -> list[int]:
    """Generate a sequence: x[n+1] = (x[n] + step) % modulus."""
    seq = [start]
    for _ in range(length - 1):
        seq.append((seq[-1] + step) % modulus)
    return seq


def generate_periodic_sequence(pattern: list[int], length: int) -> list[int]:
    """Repeat a fixed pattern."""
    return [pattern[i % len(pattern)] for i in range(length)]


def generate_fibonacci_mod(modulus: int, length: int) -> list[int]:
    """Fibonacci sequence mod m."""
    seq = [0, 1]
    for i in range(2, length):
        seq.append((seq[-1] + seq[-2]) % modulus)
    return seq


def generate_collatz_parity(start: int, length: int) -> list[int]:
    """Parity (0/1) of the Collatz sequence."""
    seq = [start]
    for _ in range(length - 1):
        n = seq[-1]
        if n <= 1:
            seq.append(1)
        elif n % 2 == 0:
            seq.append(n // 2)
        else:
            seq.append(3 * n + 1)
    return [x % 2 for x in seq]


# --- Markov chains (Stage 1) ---

def generate_markov_chain(transition_matrix: list[list[float]],
                          start_state: int, length: int,
                          rng: random.Random | None = None) -> list[int]:
    """Generate a sequence from a first-order Markov chain.

    transition_matrix[i][j] = P(next=j | current=i)
    """
    if rng is None:
        rng = random.Random(42)

    seq = [start_state]
    for _ in range(length - 1):
        current = seq[-1]
        probs = transition_matrix[current]
        r = rng.random()
        cumulative = 0.0
        for j, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                seq.append(j)
                break
        else:
            seq.append(len(probs) - 1)
    return seq


def markov_entropy_rate(transition_matrix: list[list[float]]) -> float:
    """Compute the entropy rate of a Markov chain.

    H = sum_i pi_i * sum_j -p_ij * log2(p_ij)
    where pi is the stationary distribution.
    """
    n = len(transition_matrix)

    # Find stationary distribution via power iteration
    pi = [1.0 / n] * n
    for _ in range(1000):
        new_pi = [0.0] * n
        for j in range(n):
            for i in range(n):
                new_pi[j] += pi[i] * transition_matrix[i][j]
        pi = new_pi

    # Compute entropy rate
    H = 0.0
    for i in range(n):
        for j in range(n):
            p = transition_matrix[i][j]
            if p > 0 and pi[i] > 0:
                H -= pi[i] * p * math.log2(p)
    return H


def markov_optimal_accuracy(transition_matrix: list[list[float]]) -> float:
    """Compute the best possible prediction accuracy for a Markov chain.

    The optimal predictor always predicts the most likely next state.
    accuracy = sum_i pi_i * max_j p_ij
    """
    n = len(transition_matrix)
    pi = [1.0 / n] * n
    for _ in range(1000):
        new_pi = [0.0] * n
        for j in range(n):
            for i in range(n):
                new_pi[j] += pi[i] * transition_matrix[i][j]
        pi = new_pi

    accuracy = 0.0
    for i in range(n):
        accuracy += pi[i] * max(transition_matrix[i])
    return accuracy


# --- Higher-order processes (Stage 2) ---

def generate_ar1(coeff: float, noise_std: float, start: float,
                 length: int, rng: random.Random | None = None) -> list[float]:
    """Generate AR(1): x[n+1] = coeff * x[n] + noise."""
    if rng is None:
        rng = random.Random(42)
    seq = [start]
    for _ in range(length - 1):
        seq.append(coeff * seq[-1] + rng.gauss(0, noise_std))
    return seq


def generate_switching_process(patterns: list[list[int]],
                               switch_prob: float, length: int,
                               rng: random.Random | None = None) -> list[int]:
    """Generate a process that switches between periodic patterns."""
    if rng is None:
        rng = random.Random(42)
    current_pattern = 0
    seq = []
    pos = 0
    for _ in range(length):
        seq.append(patterns[current_pattern][pos % len(patterns[current_pattern])])
        pos += 1
        if rng.random() < switch_prob:
            current_pattern = (current_pattern + 1) % len(patterns)
            pos = 0
    return seq


# ── Spec builders ────────────────────────────────────────────────────

def _seq_to_prediction_spec(seq: list, context_len: int = 1,
                            type_name: str = "number") -> Spec:
    """Convert a sequence to a next-element prediction spec.

    Each example: (context of last `context_len` values) -> next value.
    For context_len=1: x[n] -> x[n+1].
    """
    pairs = []
    for i in range(len(seq) - context_len):
        if context_len == 1:
            in_node = Number(float(seq[i]))
            out_node = Number(float(seq[i + 1]))
        else:
            # Pack context as a string of comma-separated values
            context = ",".join(str(seq[i + j]) for j in range(context_len))
            in_node = String(context)
            out_node = Number(float(seq[i + context_len]))
        pairs.append((in_node, out_node))

    return Spec(
        type_expr=Symbol(type_name),
        goal=GoalExamples(pairs=tuple(pairs)),
    )


def _seq_to_mapping_spec(seq: list, type_name: str = "number") -> Spec:
    """Convert a sequence to a state-transition mapping spec.

    Each unique (current, next) pair becomes an example.
    """
    seen = set()
    pairs = []
    for i in range(len(seq) - 1):
        key = (seq[i], seq[i + 1])
        if key not in seen:
            seen.add(key)
            pairs.append((Number(float(seq[i])), Number(float(seq[i + 1]))))

    return Spec(
        type_expr=Symbol(type_name),
        goal=GoalExamples(pairs=tuple(pairs)),
    )


# ── Benchmark suite builders ─────────────────────────────────────────

def build_stage0_benchmarks(seed: int = 42) -> list[BenchmarkTask]:
    """Stage 0: Deterministic sequence prediction.

    These have entropy rate = 0 (fully predictable).
    The synthesizer should find the exact generating rule.
    """
    tasks = []

    # Modular arithmetic sequences
    for step, mod in [(1, 5), (2, 7), (3, 10), (1, 3)]:
        seq = generate_modular_sequence(0, step, mod, 20)
        spec = _seq_to_mapping_spec(seq)
        tasks.append(BenchmarkTask(
            name=f"mod_step{step}_mod{mod}",
            spec=spec,
            process_type="deterministic",
            optimal_accuracy=1.0,
            entropy_rate=0.0,
            reference_program=f"(lambda (x) (modulo (add x {step}) {mod}))",
            stage=0,
        ))

    # Periodic sequences
    for pattern in [[0, 1], [0, 1, 2], [1, 0, 1, 0, 2]]:
        seq = generate_periodic_sequence(pattern, 30)
        spec = _seq_to_mapping_spec(seq)
        period = len(pattern)
        tasks.append(BenchmarkTask(
            name=f"periodic_{period}",
            spec=spec,
            process_type="deterministic",
            optimal_accuracy=1.0,
            entropy_rate=0.0,
            stage=0,
        ))

    # Fibonacci mod
    for mod in [3, 5]:
        seq = generate_fibonacci_mod(mod, 30)
        # Fib mod has a period (Pisano period), so it's deterministic
        # but needs 2-element context
        spec = _seq_to_prediction_spec(seq, context_len=1)
        tasks.append(BenchmarkTask(
            name=f"fib_mod{mod}",
            spec=spec,
            process_type="deterministic",
            optimal_accuracy=1.0,  # with sufficient context
            entropy_rate=0.0,
            stage=0,
        ))

    return tasks


def build_stage1_benchmarks(seed: int = 42) -> list[BenchmarkTask]:
    """Stage 1: Markov chain prediction.

    These have entropy rate > 0 (inherently unpredictable).
    The optimal predictor uses the transition probabilities.
    We test whether the system can learn the *deterministic* part
    (most likely next state given current state).
    """
    rng = random.Random(seed)
    tasks = []

    # Nearly deterministic chain (one dominant transition per state)
    tm1 = [
        [0.1, 0.9],  # state 0 -> state 1 with 90%
        [0.8, 0.2],  # state 1 -> state 0 with 80%
    ]
    seq1 = generate_markov_chain(tm1, 0, 50, rng)
    # For spec: use the most common transition as the "correct" answer
    spec1 = _deterministic_markov_spec(seq1, tm1)
    tasks.append(BenchmarkTask(
        name="markov_2state_biased",
        spec=spec1,
        process_type="markov",
        optimal_accuracy=markov_optimal_accuracy(tm1),
        entropy_rate=markov_entropy_rate(tm1),
        reference_program="(lambda (x) (if (= x 0) 1 0))",
        stage=1,
    ))

    # 3-state chain with clear structure
    tm2 = [
        [0.0, 1.0, 0.0],  # 0 -> 1 always
        [0.0, 0.0, 1.0],  # 1 -> 2 always
        [0.9, 0.05, 0.05],  # 2 -> 0 usually
    ]
    seq2 = generate_markov_chain(tm2, 0, 50, rng)
    spec2 = _deterministic_markov_spec(seq2, tm2)
    tasks.append(BenchmarkTask(
        name="markov_3state_cycle",
        spec=spec2,
        process_type="markov",
        optimal_accuracy=markov_optimal_accuracy(tm2),
        entropy_rate=markov_entropy_rate(tm2),
        reference_program="(lambda (x) (modulo (add x 1) 3))",
        stage=1,
    ))

    # Biased coin flip (i.i.d., no state dependence)
    tm3 = [
        [0.3, 0.7],
        [0.3, 0.7],  # same row = i.i.d.
    ]
    seq3 = generate_markov_chain(tm3, 0, 50, rng)
    spec3 = _deterministic_markov_spec(seq3, tm3)
    tasks.append(BenchmarkTask(
        name="iid_biased_coin",
        spec=spec3,
        process_type="markov",
        optimal_accuracy=markov_optimal_accuracy(tm3),
        entropy_rate=markov_entropy_rate(tm3),
        reference_program="(lambda (x) 1)",  # always predict 1 (70% likely)
        stage=1,
    ))

    # 4-state chain with block structure
    tm4 = [
        [0.0, 0.9, 0.1, 0.0],
        [0.0, 0.0, 0.9, 0.1],
        [0.1, 0.0, 0.0, 0.9],
        [0.9, 0.1, 0.0, 0.0],
    ]
    seq4 = generate_markov_chain(tm4, 0, 80, rng)
    spec4 = _deterministic_markov_spec(seq4, tm4)
    tasks.append(BenchmarkTask(
        name="markov_4state_ring",
        spec=spec4,
        process_type="markov",
        optimal_accuracy=markov_optimal_accuracy(tm4),
        entropy_rate=markov_entropy_rate(tm4),
        reference_program="(lambda (x) (modulo (add x 1) 4))",
        stage=1,
    ))

    return tasks


def _deterministic_markov_spec(seq: list[int],
                               transition_matrix: list[list[float]]) -> Spec:
    """Create a spec using the most-likely-next-state mapping.

    For each state, the "correct" output is argmax of the transition row.
    This is the deterministic approximation of the stochastic process.
    """
    n = len(transition_matrix)
    best_next = {}
    for i in range(n):
        best_j = max(range(n), key=lambda j: transition_matrix[i][j])
        best_next[i] = best_j

    pairs = []
    seen = set()
    for state in range(n):
        if state not in seen and state in best_next:
            seen.add(state)
            pairs.append((Number(float(state)), Number(float(best_next[state]))))

    return Spec(
        type_expr=Symbol("number"),
        goal=GoalExamples(pairs=tuple(pairs)),
    )


def build_stage2_benchmarks(seed: int = 42) -> list[BenchmarkTask]:
    """Stage 2: Higher-order and switching processes.

    These require remembering more context or detecting regime changes.
    Framed as mapping tasks where the program needs to use more
    complex logic (if/let/comparison).
    """
    rng = random.Random(seed)
    tasks = []

    # Second-order deterministic: x[n] = (x[n-1] + x[n-2]) % mod
    for mod in [4, 7]:
        seq = generate_fibonacci_mod(mod, 40)
        # Pack pairs of consecutive values as input
        pairs = []
        for i in range(len(seq) - 2):
            # Encode two values as a single number: prev*mod + current
            encoded_in = seq[i] * mod + seq[i + 1]
            out = seq[i + 2]
            pairs.append((Number(float(encoded_in)), Number(float(out))))

        # Deduplicate
        seen = set()
        unique_pairs = []
        for p in pairs:
            key = (p[0].value, p[1].value)
            if key not in seen:
                seen.add(key)
                unique_pairs.append(p)

        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=tuple(unique_pairs[:20])),
        )
        tasks.append(BenchmarkTask(
            name=f"fib_mod{mod}_order2",
            spec=spec,
            process_type="deterministic_order2",
            optimal_accuracy=1.0,
            entropy_rate=0.0,
            stage=2,
        ))

    # Piecewise linear: different rule for different input ranges
    pairs = []
    for x in range(-5, 6):
        if x < 0:
            y = -x  # abs
        else:
            y = x * 2  # double
        pairs.append((Number(float(x)), Number(float(y))))

    spec = Spec(
        type_expr=Symbol("number"),
        goal=GoalExamples(pairs=tuple(pairs)),
    )
    tasks.append(BenchmarkTask(
        name="piecewise_abs_or_double",
        spec=spec,
        process_type="piecewise",
        optimal_accuracy=1.0,
        entropy_rate=0.0,
        reference_program="(lambda (x) (if (< x 0) (negate x) (add x x)))",
        stage=2,
    ))

    # Threshold function: 1 if x > threshold, 0 otherwise
    for thresh in [3, 0]:
        pairs = []
        for x in range(-5, 8):
            y = 1.0 if x > thresh else 0.0
            pairs.append((Number(float(x)), Number(y)))

        spec = Spec(
            type_expr=Symbol("number"),
            goal=GoalExamples(pairs=tuple(pairs)),
        )
        tasks.append(BenchmarkTask(
            name=f"threshold_{thresh}",
            spec=spec,
            process_type="threshold",
            optimal_accuracy=1.0,
            entropy_rate=0.0,
            reference_program=f"(lambda (x) (if (> x {thresh}) 1 0))",
            stage=2,
        ))

    return tasks


# ── Full suite ───────────────────────────────────────────────────────

def build_full_suite(seed: int = 42) -> BenchmarkSuite:
    """Build the complete stochastic process benchmark suite."""
    suite = BenchmarkSuite(
        name="SELPH Stochastic Process Benchmarks v0.1",
        description=(
            "Processes with known information-theoretic limits. "
            "Stage 0: deterministic sequences. "
            "Stage 1: Markov chains. "
            "Stage 2: higher-order / piecewise processes."
        ),
    )
    suite.tasks.extend(build_stage0_benchmarks(seed))
    suite.tasks.extend(build_stage1_benchmarks(seed))
    suite.tasks.extend(build_stage2_benchmarks(seed))
    return suite
