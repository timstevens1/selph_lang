"""Annotate natural language passages with inline SELPH s-expressions.

Takes plain text containing factual claims about KB entities and replaces
factual spans with s-expressions that evaluate to the same text.

Input:  "Madrid, the capital of Spain, has over 3 million residents."
Output: "<tool_call>(capital \"Spain\")</tool_call>, the capital of Spain, has over 3 million residents."

The annotation pipeline:
1. Load KB facts as (entity, property, value) triples
2. Build a trie/index of entity names and fact values for fast matching
3. For each passage, find spans that match KB values
4. Replace matchable spans with s-expressions that produce those values
5. Prefer composed expressions over direct value insertion
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict


def build_fact_index(kb: dict) -> Tuple[dict, dict]:
    """Build lookup indices from the KB.

    Returns:
        value_to_exprs: maps a fact value (string) → list of (sexpr, entity, property)
        entity_names: set of all entity names in KB
    """
    # value → [(sexpr, entity_name, property_name)]
    value_to_exprs = defaultdict(list)
    entity_names = set()

    for entity, props in kb.items():
        entity_names.add(entity)

        prop_map = {
            "capital": "capital",
            "continent": "continent",
            "currency": "currency",
            "language": "language",
            "head_of_state": "head-of-state",
        }

        for prop_key, selph_fn in prop_map.items():
            if prop_key in props:
                val = str(props[prop_key])
                sexpr = f'({selph_fn} "{entity}")'
                value_to_exprs[val].append((sexpr, entity, prop_key))

        # Population — store as string for matching
        if "population" in props:
            pop = props["population"]
            value_to_exprs[str(pop)].append(
                (f'(population "{entity}")', entity, "population")
            )
            # Also store common formatted versions
            if isinstance(pop, int):
                # "3,200,000" format
                formatted = f"{pop:,}"
                value_to_exprs[formatted].append(
                    (f'(population "{entity}")', entity, "population")
                )
                # "3.2 million" style — approximate
                if pop >= 1_000_000:
                    millions = round(pop / 1_000_000, 1)
                    value_to_exprs[f"{millions} million"].append(
                        (f'(population "{entity}")', entity, "population")
                    )
                    # Integer millions
                    if millions == int(millions):
                        value_to_exprs[f"{int(millions)} million"].append(
                            (f'(population "{entity}")', entity, "population")
                        )

        # Capital population — also via composition
        if "capital" in props and "capital_population" in props:
            cap = props["capital"]
            cap_pop = props["capital_population"]
            sexpr = f'(population (capital "{entity}"))'
            value_to_exprs[str(cap_pop)].append((sexpr, entity, "capital_population"))
            if isinstance(cap_pop, int) and cap_pop >= 1_000_000:
                millions = round(cap_pop / 1_000_000, 1)
                value_to_exprs[f"{millions} million"].append(
                    (sexpr, entity, "capital_population"))

    return dict(value_to_exprs), entity_names


def find_entity_mentions(text: str, entity_names: set) -> List[Tuple[int, int, str]]:
    """Find all entity name mentions in text, sorted by position.

    Returns list of (start, end, entity_name).
    """
    mentions = []
    # Sort by length descending to match longest first
    for entity in sorted(entity_names, key=len, reverse=True):
        # Word-boundary match
        pattern = r'\b' + re.escape(entity) + r'\b'
        for m in re.finditer(pattern, text):
            mentions.append((m.start(), m.end(), entity))
    # Remove overlapping matches (keep longest)
    mentions.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = -1
    for start, end, entity in mentions:
        if start >= last_end:
            filtered.append((start, end, entity))
            last_end = end
    return filtered


def find_fact_spans(
    text: str,
    entity_mentions: List[Tuple[int, int, str]],
    value_to_exprs: dict,
    kb: dict,
) -> List[Tuple[int, int, str, str]]:
    """Find spans in text that match KB fact values near mentioned entities.

    Returns list of (start, end, matched_value, sexpr) sorted by position.
    """
    spans = []

    # For each value in the KB, check if it appears in the text
    # Only match if a related entity is also mentioned nearby
    entity_positions = {e: (s, end) for s, end, e in entity_mentions}
    mentioned_entities = {e for _, _, e in entity_mentions}

    for value, expr_list in value_to_exprs.items():
        if len(value) < 3:  # Skip very short values
            continue

        # Check if this value appears in the text
        pattern = r'\b' + re.escape(value) + r'\b'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val_start, val_end = m.start(), m.end()

            # Find the best matching expression
            # Prefer expressions whose entity is mentioned in the text
            best_expr = None
            for sexpr, entity, prop in expr_list:
                if entity in mentioned_entities:
                    best_expr = sexpr
                    break

            if best_expr:
                # Don't replace if the span IS an entity name
                matched_text = text[val_start:val_end]
                if matched_text not in mentioned_entities:
                    spans.append((val_start, val_end, matched_text, best_expr))

    # Sort by position, remove overlaps
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = -1
    for start, end, val, expr in spans:
        if start >= last_end:
            filtered.append((start, end, val, expr))
            last_end = end
    return filtered


def annotate_passage(
    text: str,
    value_to_exprs: dict,
    entity_names: set,
    kb: dict,
    min_annotations: int = 1,
) -> Optional[Tuple[str, int]]:
    """Annotate a passage with inline s-expressions.

    Returns (annotated_text, num_annotations) or None if too few annotations.
    """
    entity_mentions = find_entity_mentions(text, entity_names)
    if not entity_mentions:
        return None

    fact_spans = find_fact_spans(text, entity_mentions, value_to_exprs, kb)
    if len(fact_spans) < min_annotations:
        return None

    # Build annotated text by replacing fact spans with <tool_call>expr</tool_call>
    result = []
    last_pos = 0
    for start, end, val, sexpr in fact_spans:
        result.append(text[last_pos:start])
        result.append(f'<tool_call>{sexpr}</tool_call>')
        last_pos = end

    result.append(text[last_pos:])
    annotated = ''.join(result)

    return annotated, len(fact_spans)


def generate_synthetic_passages(kb: dict) -> List[str]:
    """Generate synthetic passages that contain multiple related facts.

    These are templates that exercise composition naturally.
    """
    passages = []

    for entity, props in kb.items():
        cap = props.get("capital")
        pop = props.get("population")
        cont = props.get("continent")
        curr = props.get("currency")
        lang = props.get("language")
        cap_pop = props.get("capital_population")
        head = props.get("head_of_state")

        if cap and cont:
            passages.append(
                f"{cap} is the capital of {entity}, a country in {cont}."
            )
        if cap and pop and cap_pop:
            passages.append(
                f"{entity} has a population of {pop}. "
                f"Its capital, {cap}, has {cap_pop} residents."
            )
        if cap and curr and lang:
            passages.append(
                f"In {entity}, the official language is {lang} and "
                f"the currency is {curr}. The capital is {cap}."
            )
        if cap and head and cont:
            passages.append(
                f"{entity} is located in {cont}. "
                f"The head of state is {head} and the capital is {cap}."
            )
        # Composition-requiring: facts about the capital
        if cap and cap_pop and cont:
            passages.append(
                f"{cap}, the capital of {entity}, is a city in {cont} "
                f"with a population of {cap_pop}."
            )

    return passages


def format_for_training(
    original: str,
    annotated: str,
    num_annotations: int,
) -> dict:
    """Format an annotated passage as an MLX training example.

    The prompt is the original text up to the first annotation point.
    The completion is the full annotated text.

    For inline generation, we use a different format: the model sees
    the beginning of the passage and must produce the rest with
    s-expressions inline.
    """
    return {
        "prompt": f"Complete with SELPH expressions for facts:\n",
        "completion": annotated,
        "original": original,
        "num_annotations": num_annotations,
    }


if __name__ == "__main__":
    kb = json.loads((Path("data/geo_kb.json")).read_text())
    value_to_exprs, entity_names = build_fact_index(kb)

    print(f"KB: {len(kb)} entities")
    print(f"Fact values indexed: {len(value_to_exprs)}")
    print(f"Entity names: {len(entity_names)}")

    # Generate synthetic passages
    print("\n=== Generating synthetic passages ===")
    passages = generate_synthetic_passages(kb)
    print(f"Generated {len(passages)} passages")

    # Annotate them
    annotated_examples = []
    total_annotations = 0
    for passage in passages:
        result = annotate_passage(passage, value_to_exprs, entity_names, kb)
        if result:
            annotated, n = result
            annotated_examples.append({
                "original": passage,
                "annotated": annotated,
                "num_annotations": n,
            })
            total_annotations += n

    print(f"Annotated {len(annotated_examples)} passages ({total_annotations} total annotations)")

    # Show examples
    print("\n=== Sample annotated passages ===")
    import random
    random.seed(42)
    samples = random.sample(annotated_examples, min(15, len(annotated_examples)))
    for ex in samples:
        print(f"\nOriginal:  {ex['original']}")
        print(f"Annotated: {ex['annotated']}")
        print(f"  ({ex['num_annotations']} annotations)")

    # Save
    out_path = Path("data/annotated_passages.json")
    with open(out_path, "w") as f:
        json.dump(annotated_examples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(annotated_examples)} examples to {out_path}")

    # Stats on annotation patterns
    from collections import Counter
    ann_counts = Counter(ex["num_annotations"] for ex in annotated_examples)
    print(f"\nAnnotations per passage:")
    for n in sorted(ann_counts):
        print(f"  {n}: {ann_counts[n]} passages")
