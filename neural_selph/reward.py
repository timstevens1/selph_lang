"""Reward function for Neural SELPH RL training.

Scores a generated s-expression on:
1. Validity: does it parse?
2. Grounding: are all string literals from the prompt? (no baked-in knowledge)
3. Correctness: does it evaluate to the expected answer in SELPH?
4. Simplicity: fewer tokens = better (tiebreaker)

The key principle: the model contributes STRUCTURE, SELPH contributes KNOWLEDGE.
String literals not present in the prompt are forbidden — the model can't smuggle
in memorized facts as literal strings.
"""
import re
import math
from typing import Optional, Tuple, Set


def extract_prompt_strings(prompt: str) -> Set[str]:
    """Extract plausible string constants from the prompt.

    Pulls out quoted strings and capitalized proper nouns / entity names.
    These are the only string literals the model is allowed to use.
    """
    strings = set()

    # Quoted strings in the prompt
    for m in re.finditer(r'"([^"]*)"', prompt):
        strings.add(m.group(1))

    # Entity names from the function signature block (if present)
    # These appear as arguments in the Q: line
    # e.g., "Q: What is the capital of Spain?" → "Spain"
    q_match = re.search(r'Q:\s*(.*?)(?:\n|$)', prompt)
    if q_match:
        question = q_match.group(1)
        # Extract capitalized words/phrases that look like entity names
        # This is a heuristic — proper nouns, country names, city names
        for m in re.finditer(r'(?:^|(?<=\s))([A-Z][a-zA-Z]+(?:\s+(?:of|and|the|la|de|del|el|al|bin|von)\s+[A-Z]?[a-zA-Z]+)*(?:\s+[A-Z][a-zA-Z]+)*)', question):
            candidate = m.group(1).strip()
            # Filter out common sentence starters
            if candidate not in {"What", "Who", "Where", "Which", "How", "The", "Is",
                                 "Does", "Do", "Can", "Solve", "Find", "Calculate",
                                 "Functions", "Expression"}:
                strings.add(candidate)

    return strings


def extract_sexpr_strings(sexpr: str) -> Set[str]:
    """Extract all string literals from an s-expression."""
    strings = set()
    for m in re.finditer(r'"([^"]*)"', sexpr):
        strings.add(m.group(1))
    return strings


def tokenize_sexpr(s):
    tokens, i = [], 0
    while i < len(s):
        c = s[i]
        if c in '()': tokens.append(c); i += 1
        elif c in ' \t\n': i += 1
        elif c == '"':
            j = i + 1
            while j < len(s) and s[j] != '"': j += 1
            tokens.append(f'"{s[i+1:j]}"'); i = j + 1
        elif c == '-' and i + 1 < len(s) and (s[i+1].isdigit() or s[i+1] == '.'):
            j = i + 1
            while j < len(s) and (s[j].isdigit() or s[j] == '.'): j += 1
            tokens.append(s[i:j]); i = j
        else:
            j = i
            while j < len(s) and s[j] not in '() \t\n"': j += 1
            tokens.append(s[i:j]); i = j
    return tokens


def sexpr_depth(sexpr: str) -> int:
    """Count maximum nesting depth of an s-expression."""
    d, mx = 0, 0
    for c in sexpr:
        if c == '(': d += 1; mx = max(mx, d)
        elif c == ')': d -= 1
    return mx


def sexpr_token_count(sexpr: str) -> int:
    """Count tokens in an s-expression."""
    return len(tokenize_sexpr(sexpr))


def check_grounding(sexpr: str, prompt: str) -> Tuple[bool, Set[str]]:
    """Check if all string literals in the s-expression come from the prompt.

    Returns (is_grounded, violating_strings).
    """
    prompt_strings = extract_prompt_strings(prompt)
    expr_strings = extract_sexpr_strings(sexpr)

    violations = expr_strings - prompt_strings
    return len(violations) == 0, violations


def compute_reward(
    sexpr: Optional[str],
    prompt: str,
    expected_answer,
    eval_fn,
    correctness_weight: float = 1.0,
    simplicity_weight: float = 0.1,
    max_tokens: int = 50,
) -> Tuple[float, dict]:
    """Compute reward for a generated s-expression.

    Args:
        sexpr: The generated s-expression (None if unparseable)
        prompt: The original prompt (for grounding check)
        expected_answer: The expected answer (string or number)
        eval_fn: Function that evaluates an s-expression string → value
        correctness_weight: Weight for correctness component
        simplicity_weight: Weight for simplicity bonus
        max_tokens: Maximum tokens for simplicity normalization

    Returns:
        (reward, info_dict) where reward is in [0, 1+simplicity_bonus]
    """
    info = {
        "parseable": False,
        "grounded": False,
        "correct": False,
        "violations": set(),
        "generated_value": None,
        "token_count": 0,
        "depth": 0,
    }

    # 1. Parseability
    if sexpr is None:
        return 0.0, info
    info["parseable"] = True
    info["token_count"] = sexpr_token_count(sexpr)
    info["depth"] = sexpr_depth(sexpr)

    # 2. Grounding: no smuggled string constants
    grounded, violations = check_grounding(sexpr, prompt)
    info["grounded"] = grounded
    info["violations"] = violations
    if not grounded:
        return 0.0, info

    # 3. Correctness: evaluate and compare
    try:
        gen_value = eval_fn(sexpr)
    except Exception:
        gen_value = None
    info["generated_value"] = gen_value

    if gen_value is None:
        return 0.0, info

    # Compare with expected answer
    correct = False
    if isinstance(expected_answer, str) and isinstance(gen_value, str):
        correct = gen_value.lower() == expected_answer.lower()
    elif isinstance(expected_answer, (int, float)) and isinstance(gen_value, (int, float)):
        if expected_answer == 0:
            correct = abs(gen_value) < 1e-6
        else:
            correct = abs(gen_value - expected_answer) / abs(expected_answer) < 0.01
    else:
        correct = str(gen_value) == str(expected_answer)

    info["correct"] = correct
    if not correct:
        return 0.0, info

    # 4. Simplicity bonus: reward shorter expressions
    token_count = info["token_count"]
    simplicity = max(0, 1 - token_count / max_tokens)  # 0 to 1
    reward = correctness_weight + simplicity_weight * simplicity

    return reward, info


if __name__ == "__main__":
    # Test the reward function
    print("=== Reward Function Tests ===\n")

    # Mock eval function for geography
    import json
    from pathlib import Path

    kb = json.loads((Path("data/geo_kb.json")).read_text())
    capital_to_country = {}
    for name, entry in kb.items():
        if "capital" in entry:
            capital_to_country[entry["capital"]] = name

    def eval_geo(sexpr):
        tokens = tokenize_sexpr(sexpr)
        pos = [0]
        def parse():
            if pos[0] >= len(tokens): return None
            tok = tokens[pos[0]]
            if tok == '(':
                pos[0] += 1; op = tokens[pos[0]]; pos[0] += 1
                args = []
                while pos[0] < len(tokens) and tokens[pos[0]] != ')': args.append(parse())
                pos[0] += 1; return (op, args)
            else:
                pos[0] += 1
                if tok.startswith('"') and tok.endswith('"'): return tok[1:-1]
                try: return float(tok)
                except: return tok
        def ev(tree):
            if isinstance(tree, (int, float, str)): return tree
            if tree is None: return None
            op, args = tree
            vals = [ev(a) for a in args]
            if any(v is None for v in vals): return None
            if op == "capital": return kb.get(str(vals[0]), {}).get("capital")
            elif op == "population":
                e = kb.get(str(vals[0]))
                if e and "population" in e: return e["population"]
                for n, e2 in kb.items():
                    if e2.get("capital") == str(vals[0]) and "capital_population" in e2:
                        return e2["capital_population"]
                return None
            elif op == "continent": return kb.get(str(vals[0]), {}).get("continent")
            elif op == "currency": return kb.get(str(vals[0]), {}).get("currency")
            elif op == "language": return kb.get(str(vals[0]), {}).get("language")
            elif op == "country-of": return capital_to_country.get(str(vals[0]))
            elif op == "divide": return vals[0] / vals[1] if vals[1] else None
            return None
        return ev(parse())

    tests = [
        # (prompt, sexpr, expected_answer, description)
        (
            'Functions: capital(country) → city\n\nQ: What is the capital of Spain?\nA:',
            '(capital "Spain")',
            "Madrid",
            "Correct, grounded"
        ),
        (
            'Functions: continent(country) → continent\n\nQ: What continent is Germany in?\nA:',
            '"Europe"',
            "Europe",
            "Correct but UNGROUNDED — smuggled answer"
        ),
        (
            'Functions: continent(country) → continent\n\nQ: What continent is Germany in?\nA:',
            '(continent "Germany")',
            "Europe",
            "Correct and grounded — uses prompt constant"
        ),
        (
            'Functions: continent(country) → continent\n\nQ: What continent is Germany in?\nA:',
            '(continent "France")',
            "Europe",
            "Wrong entity — France not in prompt"
        ),
        (
            'Functions: population(entity) → number\n\nQ: What is the population of Japan?\nA:',
            '(population "Japan")',
            kb.get("Japan", {}).get("population", 0),
            "Numeric answer, grounded"
        ),
        (
            'Functions: capital(country) → city\n\nQ: What is the capital of Spain?\nA:',
            '(capital "Madrid")',
            "Madrid",
            "Ungrounded — Madrid is not in the prompt"
        ),
        (
            'Functions: currency(country) → currency\n\nQ: What currency is used in Tokyo?\nA:',
            '(currency (country-of "Tokyo"))',
            "yen",
            "Composition, grounded — Tokyo is in prompt"
        ),
    ]

    for prompt, sexpr, expected, desc in tests:
        reward, info = compute_reward(sexpr, prompt, expected, eval_geo)
        status = "✓" if reward > 0 else "✗"
        print(f"{status} {desc}")
        print(f"  Expression: {sexpr}")
        print(f"  Reward: {reward:.3f}")
        if info["violations"]:
            print(f"  Violations: {info['violations']}")
        if info["generated_value"] is not None:
            print(f"  Evaluated: {info['generated_value']}")
        print()
