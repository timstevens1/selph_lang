"""Growing synthesis: sequence prediction with position awareness.

The curriculum teaches concepts incrementally:

Level 0: Constants (1,1,1 -> 1)
Level 1: "Next = last + constant" (arithmetic sequences)
Level 1.5: "Position matters" — learn to extract index from context
Level 2: "Next depends on position" (squares, triangular, cubes)
Level 3: Modular/cyclic patterns
Level 4: Second-order (Fibonacci, sums of previous)
Level 5: Compositions of earlier patterns

Input encoding: "idx v0 v1 v2 v3" where idx = position of the value to predict.
This lets the system learn position-dependent functions.
"""

import sys
import os
import time
import random
sys.path.insert(0, ".")

import selph_fast
from selph.parser import parse
from selph.logger import SynthesisLogger, SynthesisLog, extract_components_from_source, estimate_depth
from selph.state import CurriculumState


# ── Sequence generators ──────────────────────────────────────────────

def make_tasks(name, fn, n_examples=10, context_len=4, seed=42):
    """Build tasks from a sequence function f(i) -> value.

    Input: "idx v_{i-4} v_{i-3} v_{i-2} v_{i-1}"
    Output: f(idx) = v_i
    """
    full_seq = [fn(i) for i in range(context_len + n_examples + 2)]

    inputs = []
    expected = []
    for i in range(context_len, min(context_len + n_examples, len(full_seq))):
        context = full_seq[i - context_len:i]
        idx = i
        next_val = full_seq[i]
        input_str = f"{idx} " + " ".join(str(int(v)) for v in context)
        inputs.append(input_str)
        expected.append(float(next_val))

    return name, inputs, expected


# ── Macro builders ───────────────────────────────────────────────────

def build_macros(context_len=4):
    """Build accessor macros for the input encoding.

    Input string: "idx v0 v1 v2 v3"
    idx = position (first element)
    v0..v3 = last context_len values (v3 is the most recent)
    """
    macros = []

    # idx: first element of the input string
    macros.append({
        'name': 'idx',
        'params': ['s'],
        'body': parse('(to-number (head (string-split s " ")))'),
    })

    # v0..v3: elements 1..4 of the input string
    for i in range(context_len):
        # v0 = element at position 1, v1 at position 2, etc.
        pos = i + 1  # skip idx
        expr = '(string-split s " ")'
        for _ in range(pos):
            expr = f'(tail {expr})'
        body = parse(f'(to-number (head {expr}))')
        macros.append({
            'name': f'v{i}',
            'params': ['s'],
            'body': body,
        })

    # last: the most recent value (= v_{context_len-1})
    macros.append({
        'name': 'last',
        'params': ['s'],
        'body': macros[context_len]['body'],  # same as v3
    })

    return macros


def build_components(context_len=4):
    """Components for sequence prediction."""
    comps = [
        # Input (string)
        {'name': 'x', 'builtin': None, 'arity': 0, 'ret_type': 1, 'param_types': [], 'priority': 0.0},
        # Constants
        {'name': '0', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '1', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '2', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '3', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '5', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '10', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
        {'name': '-1', 'builtin': None, 'arity': 0, 'ret_type': 0, 'param_types': [], 'priority': 0.0},
    ]

    # Sequence accessors (str -> num)
    comps.append({'name': 'idx', 'builtin': 'idx', 'arity': 1, 'ret_type': 0,
                  'param_types': [1], 'priority': 50.0})
    for i in range(context_len):
        comps.append({'name': f'v{i}', 'builtin': f'v{i}', 'arity': 1, 'ret_type': 0,
                      'param_types': [1], 'priority': 40.0})
    comps.append({'name': 'last', 'builtin': 'last', 'arity': 1, 'ret_type': 0,
                  'param_types': [1], 'priority': 45.0})

    # Arithmetic (num -> num -> num)
    for name in ['add', 'subtract', 'multiply', 'min', 'max', 'modulo']:
        comps.append({'name': name, 'builtin': name, 'arity': 2, 'ret_type': 0,
                      'param_types': [0, 0], 'priority': 0.0})

    # Unary (num -> num)
    for name in ['abs', 'negate']:
        comps.append({'name': name, 'builtin': name, 'arity': 1, 'ret_type': 0,
                      'param_types': [0], 'priority': 0.0})

    return comps


# ── Curriculum ───────────────────────────────────────────────────────

STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "curriculum_state")

def run():
    print("=" * 60)
    print("SEQUENCE PREDICTION CURRICULUM (position-aware)")
    print("=" * 60)

    context_len = 4
    base_macros = build_macros(context_len)
    base_components = build_components(context_len)

    # Load persistent state (or start fresh)
    state = CurriculumState.load(STATE_DIR)
    if state.macros:
        print(f"\n  Loaded existing state: {state.summary()}")
    else:
        print(f"\n  Starting fresh")

    # Merge base + persisted
    promoted_macros = list(base_macros) + list(state.macros)
    promoted_comps = list(base_components) + list(state.components)
    logger = state.logger

    levels = [
        ("Level 0: Constants", [
            make_tasks("const_0", lambda i: 0),
            make_tasks("const_1", lambda i: 1),
            make_tasks("const_5", lambda i: 5),
            make_tasks("const_neg1", lambda i: -1),
        ]),
        ("Level 1: Last + constant", [
            make_tasks("inc_1", lambda i: i),           # 0,1,2,3,...  next = last+1
            make_tasks("inc_2", lambda i: 2*i),         # 0,2,4,6,...  next = last+2
            make_tasks("inc_3", lambda i: 3*i),         # 0,3,6,9,...  next = last+3
            make_tasks("dec_1", lambda i: 10-i),        # 10,9,8,...   next = last-1
            make_tasks("inc_5", lambda i: 5*i),         # 0,5,10,...   next = last+5
        ]),
        ("Level 1.5: Position-dependent", [
            # These teach the system that idx matters
            make_tasks("identity", lambda i: i),        # f(i) = i (same as inc_1 but framed as idx)
            make_tasks("idx_plus1", lambda i: i+1),     # f(i) = i+1
            make_tasks("idx_times2", lambda i: 2*i),    # f(i) = 2*i
            make_tasks("idx_times3", lambda i: 3*i),    # f(i) = 3*i
        ]),
        ("Level 2: Quadratic/position", [
            make_tasks("squares", lambda i: i*i),               # 0,1,4,9,16,...
            make_tasks("triangular", lambda i: i*(i+1)//2),     # 0,1,3,6,10,...
            make_tasks("idx_sq_plus1", lambda i: i*i + 1),      # 1,2,5,10,17,...
            make_tasks("double_idx", lambda i: 2*i + 1),        # 1,3,5,7,9,...
        ]),
        ("Level 3: Modular/cyclic", [
            make_tasks("mod2", lambda i: i % 2),                # 0,1,0,1,...
            make_tasks("mod3", lambda i: i % 3),                # 0,1,2,0,1,2,...
            make_tasks("mod5", lambda i: i % 5),                # 0,1,2,3,4,0,...
        ]),
        ("Level 4: Second-order", [
            make_tasks("fib", lambda i: _fib(i)),               # 0,1,1,2,3,5,8,...
            make_tasks("sum_prev2", lambda i: _sum_prev(i)),    # uses last two values
            make_tasks("double_prev", lambda i: 2**i),          # 1,2,4,8,16,...
        ]),
        ("Level 5: Compositions", [
            make_tasks("sq_mod3", lambda i: (i*i) % 3),         # squares mod 3
            make_tasks("tri_mod5", lambda i: (i*(i+1)//2) % 5), # triangular mod 5
            make_tasks("cubes", lambda i: i*i*i),               # 0,1,8,27,...
        ]),
    ]

    total_solved = 0
    total_tasks = 0

    for level_name, tasks in levels:
        print(f"\n--- {level_name} ---")
        print(f"  Library: {len(promoted_comps)} components, {len(promoted_macros)} macros\n")

        level_solved = 0
        for name, inputs, expected in tasks:
            total_tasks += 1
            t0 = time.perf_counter()

            found, source, explored = selph_fast.fast_synthesize(
                promoted_comps, inputs, expected, 3, 100000, promoted_macros)
            elapsed = time.perf_counter() - t0

            status = "OK" if found else "--"
            print(f"  {status}  {name:20s}  cand={explored:6d}  t={elapsed:.3f}s"
                  + (f"  {source}" if found else ""))

            log = SynthesisLog(
                task_name=name,
                input_type="string", output_type="number",
                num_examples=len(inputs),
                example_inputs=inputs[:3],
                example_outputs=[str(e) for e in expected[:3]],
                found=found, source=source,
                candidates_explored=explored,
                time_seconds=elapsed,
            )
            if found:
                log.components_used = extract_components_from_source(source)
                log.solution_depth = estimate_depth(source)
            logger.log(log)

            if found:
                level_solved += 1
                total_solved += 1

                # Promote: add solution to state + local lists
                macro_name = f"seq_{name}"
                try:
                    body_source = source.replace("(lambda (x) ", "", 1)
                    if body_source.endswith(")"):
                        body_source = body_source[:-1]
                    body_node = parse(body_source)

                    macro_def = {
                        'name': macro_name, 'params': ['s'], 'body': body_node,
                    }
                    comp_def = {
                        'name': macro_name, 'builtin': macro_name, 'arity': 1,
                        'ret_type': 0, 'param_types': [1], 'priority': 30.0,
                    }
                    promoted_macros.append(macro_def)
                    promoted_comps.append(comp_def)

                    # Also persist in state
                    state.add_macro(macro_name, ['s'], body_source)
                    state.add_component(comp_def)
                except Exception:
                    pass

        print(f"\n  {level_name}: {level_solved}/{len(tasks)} solved")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Total: {total_solved}/{total_tasks} solved ({total_solved/total_tasks:.0%})")
    print(f"  Library: {len(promoted_comps)} components, {len(promoted_macros)} macros")
    print(f"\n{logger.summary()}")

    # Show library growth
    print("\n  Promoted solutions (library growth):")
    for m in promoted_macros:
        if m['name'].startswith('seq_') or m['name'].startswith('lib_'):
            print(f"    {m['name']}")

    # Save persistent state
    state.logger = logger
    state.metadata["runs_completed"] = state.metadata.get("runs_completed", 0) + 1
    state.metadata["total_tasks"] = total_tasks
    state.metadata["total_solved"] = total_solved
    state.save()
    print(f"\n  State saved to {STATE_DIR}/")
    print(f"  {state.summary()}")


# Helpers for sequence generators
def _fib(i):
    if i <= 0: return 0
    if i == 1: return 1
    a, b = 0, 1
    for _ in range(2, i + 1):
        a, b = b, a + b
    return b

def _sum_prev(i):
    # Same as fibonacci starting from 0, 1
    return _fib(i)


if __name__ == "__main__":
    run()
