"""Formal language curriculum: teach SELPH to recognize and predict
patterns from the Chomsky hierarchy.

Stage 0: Finite languages (fixed vocabulary lookup)
Stage 1: Regular languages (repetition, alternation)
Stage 2: Context-free patterns (matching, nesting counts)
Stage 3: Compositions

Each task: given a sequence of tokens, predict the next token.
Tokens are single characters encoded as their ASCII values.

The key question: can the library build up from simple pattern
recognizers to handle increasingly structured languages?
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


# ── Language generators ──────────────────────────────────────────────

def repeat_lang(char, n):
    """aaa... — just one character repeated."""
    return [float(ord(char))] * n

def alternate_lang(a, b, n):
    """ababab... — two characters alternating."""
    return [float(ord(a)) if i % 2 == 0 else float(ord(b)) for i in range(n)]

def cycle_lang(chars, n):
    """abcabc... — cycle through a list of characters."""
    return [float(ord(chars[i % len(chars)])) for i in range(n)]

def ascending_lang(start, n):
    """abcde... — alphabetical sequence."""
    return [float(ord(start) + i) for i in range(n)]

def descending_lang(start, n):
    """edcba... — reverse alphabetical."""
    return [float(ord(start) - i) for i in range(n)]

def repeat_each_lang(chars, repeat, n):
    """aabbcc... — each character repeated `repeat` times."""
    seq = []
    for c in chars * (n // (len(chars) * repeat) + 1):
        seq.extend([float(ord(c))] * repeat)
    return seq[:n]

def mirror_lang(chars, n):
    """abcba abcba... — palindrome repeated."""
    pattern = list(chars) + list(reversed(chars[:-1]))
    seq = []
    for i in range(n):
        seq.append(float(ord(pattern[i % len(pattern)])))
    return seq

def anbn_lang(n_pairs, repeats=1):
    """a^n b^n — n a's followed by n b's. Repeats the pattern."""
    pattern = [float(ord('a'))] * n_pairs + [float(ord('b'))] * n_pairs
    return pattern * repeats

def matched_parens(depth, n):
    """((()))-style nesting as numeric depth markers.
    Encode as: depth at each position. 0,1,2,2,1,0,..."""
    seq = []
    for _ in range(n // (2 * depth)):
        for d in range(depth):
            seq.append(float(d + 1))
        for d in range(depth - 1, -1, -1):
            seq.append(float(d))
    return seq[:n]


# ── Task builder ─────────────────────────────────────────────────────

def make_lang_tasks(name, sequence, context_len=4, n_examples=10):
    """Build prediction tasks from a sequence.

    Input: "idx v0 v1 v2 v3" (position + context window)
    Output: next value
    """
    inputs = []
    expected = []
    for i in range(context_len, min(context_len + n_examples, len(sequence))):
        context = sequence[i - context_len:i]
        idx = i
        next_val = sequence[i]
        input_str = f"{idx} " + " ".join(str(int(v)) for v in context)
        inputs.append(input_str)
        expected.append(float(next_val))

    return name, inputs, expected


# ── Macros and components ────────────────────────────────────────────

def build_macros(context_len=4):
    """Accessor macros for "idx v0 v1 v2 v3" input format."""
    macros = []

    # idx
    macros.append({
        'name': 'idx',
        'params': ['s'],
        'body': parse('(to-number (head (string-split s " ")))'),
    })

    # v0..v3
    for i in range(context_len):
        pos = i + 1
        expr = '(string-split s " ")'
        for _ in range(pos):
            expr = f'(tail {expr})'
        body = parse(f'(to-number (head {expr}))')
        macros.append({'name': f'v{i}', 'params': ['s'], 'body': body})

    # last = most recent value
    macros.append({
        'name': 'last',
        'params': ['s'],
        'body': macros[context_len]['body'],
    })

    return macros


def build_components(context_len=4):
    """Components for formal language prediction."""
    comps = [
        {'name': 'x', 'builtin': None, 'arity': 0, 'ret_type': 1, 'param_types': [], 'priority': 0.0},
    ]

    # Character constants (ASCII values for a-z and common chars)
    for c in 'abcdefghij':
        comps.append({'name': str(ord(c)), 'builtin': None, 'arity': 0,
                      'ret_type': 0, 'param_types': [], 'priority': 0.0})

    # Small numeric constants
    for v in [0, 1, 2, 3, 4, 5, -1]:
        comps.append({'name': str(v), 'builtin': None, 'arity': 0,
                      'ret_type': 0, 'param_types': [], 'priority': 0.0})

    # Accessors
    comps.append({'name': 'idx', 'builtin': 'idx', 'arity': 1, 'ret_type': 0,
                  'param_types': [1], 'priority': 50.0})
    for i in range(context_len):
        comps.append({'name': f'v{i}', 'builtin': f'v{i}', 'arity': 1, 'ret_type': 0,
                      'param_types': [1], 'priority': 40.0})
    comps.append({'name': 'last', 'builtin': 'last', 'arity': 1, 'ret_type': 0,
                  'param_types': [1], 'priority': 45.0})

    # Arithmetic
    for name in ['add', 'subtract', 'multiply', 'modulo', 'min', 'max']:
        comps.append({'name': name, 'builtin': name, 'arity': 2, 'ret_type': 0,
                      'param_types': [0, 0], 'priority': 0.0})

    # Unary
    for name in ['abs', 'negate']:
        comps.append({'name': name, 'builtin': name, 'arity': 1, 'ret_type': 0,
                      'param_types': [0], 'priority': 0.0})

    return comps


# ── Curriculum ───────────────────────────────────────────────────────

STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "formal_lang_state")

def run():
    print("=" * 60)
    print("FORMAL LANGUAGE CURRICULUM")
    print("Learning pattern recognition from Chomsky hierarchy")
    print("=" * 60)

    context_len = 4
    base_macros = build_macros(context_len)
    base_components = build_components(context_len)

    state = CurriculumState.load(STATE_DIR)
    if state.macros:
        print(f"\n  Loaded: {state.summary()}")
    else:
        print(f"\n  Starting fresh")

    promoted_macros = list(base_macros) + list(state.macros)
    promoted_comps = list(base_components) + list(state.components)
    logger = state.logger

    a, b, c, d, e = ord('a'), ord('b'), ord('c'), ord('d'), ord('e')

    levels = [
        ("Stage 0: Constant/repeat", [
            make_lang_tasks("repeat_a", repeat_lang('a', 20)),
            make_lang_tasks("repeat_b", repeat_lang('b', 20)),
            make_lang_tasks("repeat_x", repeat_lang('x', 20)),
        ]),
        ("Stage 1: Simple alternation", [
            make_lang_tasks("alt_ab", alternate_lang('a', 'b', 20)),
            make_lang_tasks("alt_cd", alternate_lang('c', 'd', 20)),
            make_lang_tasks("cycle_abc", cycle_lang("abc", 24)),
            make_lang_tasks("cycle_abcd", cycle_lang("abcd", 24)),
        ]),
        ("Stage 2: Arithmetic on characters", [
            make_lang_tasks("ascending_a", ascending_lang('a', 20)),
            make_lang_tasks("ascending_d", ascending_lang('d', 20)),
            make_lang_tasks("descending_j", descending_lang('j', 20)),
            make_lang_tasks("step2_a", [float(ord('a') + 2*i) for i in range(20)]),
        ]),
        ("Stage 3: Repeat-each patterns", [
            make_lang_tasks("each2_ab", repeat_each_lang("ab", 2, 24)),
            make_lang_tasks("each3_ab", repeat_each_lang("ab", 3, 24)),
            make_lang_tasks("each2_abc", repeat_each_lang("abc", 2, 24)),
        ]),
        ("Stage 4: Mirror/palindrome", [
            make_lang_tasks("mirror_abc", mirror_lang("abc", 24)),
            make_lang_tasks("mirror_ab", mirror_lang("ab", 24)),
        ]),
        ("Stage 5: Depth patterns (proto context-free)", [
            make_lang_tasks("depth2", matched_parens(2, 24)),
            make_lang_tasks("depth3", matched_parens(3, 24)),
            make_lang_tasks("anbn_3", anbn_lang(3, 3)),
        ]),
    ]

    total_solved = 0
    total_tasks = 0

    for level_name, tasks in levels:
        print(f"\n--- {level_name} ---")
        print(f"  Library: {len(promoted_comps)} components, {len(promoted_macros)} macros\n")

        level_solved = 0
        for name, inputs, expected in tasks:
            if len(inputs) < 3:
                print(f"  --  {name:25s}  (insufficient examples)")
                total_tasks += 1
                continue

            total_tasks += 1
            t0 = time.perf_counter()

            found, source, explored = selph_fast.fast_synthesize(
                promoted_comps, inputs, expected, 3, 100000, promoted_macros)
            elapsed = time.perf_counter() - t0

            status = "OK" if found else "--"
            # Decode: show what characters the solution produces
            decoded = ""
            if found and expected:
                decoded = f"  [{' '.join(chr(int(e)) for e in expected[:5])}...]"

            print(f"  {status}  {name:25s}  cand={explored:6d}  t={elapsed:.3f}s"
                  + (f"  {source}{decoded}" if found else ""))

            log = SynthesisLog(
                task_name=name, input_type="string", output_type="number",
                num_examples=len(inputs),
                example_inputs=inputs[:3],
                example_outputs=[str(e) for e in expected[:3]],
                found=found, source=source,
                candidates_explored=explored, time_seconds=elapsed,
            )
            if found:
                log.components_used = extract_components_from_source(source)
                log.solution_depth = estimate_depth(source)
            logger.log(log)

            if found:
                level_solved += 1
                total_solved += 1

                macro_name = f"lang_{name}"
                try:
                    body_source = source.replace("(lambda (x) ", "", 1)
                    if body_source.endswith(")"):
                        body_source = body_source[:-1]
                    body_node = parse(body_source)
                    macro_def = {'name': macro_name, 'params': ['s'], 'body': body_node}
                    comp_def = {'name': macro_name, 'builtin': macro_name, 'arity': 1,
                                'ret_type': 0, 'param_types': [1], 'priority': 30.0}
                    promoted_macros.append(macro_def)
                    promoted_comps.append(comp_def)
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
    print(f"  Library: {len(promoted_comps)} components")

    print("\n  Per-stage:")
    for level_name, tasks in levels:
        level_logs = [l for l in logger.logs if l.task_name in [t[0] for t in tasks]]
        solved = sum(1 for l in level_logs if l.found)
        print(f"    {level_name:40s}  {solved}/{len(tasks)}")

    print("\n  Promoted solutions:")
    for m in promoted_macros:
        if m['name'].startswith('lang_'):
            print(f"    {m['name']}")

    # Save state
    state.logger = logger
    state.metadata["runs_completed"] = state.metadata.get("runs_completed", 0) + 1
    state.metadata["total_tasks"] = total_tasks
    state.metadata["total_solved"] = total_solved
    state.save()
    print(f"\n  State saved to {STATE_DIR}/")


if __name__ == "__main__":
    run()
