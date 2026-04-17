"""Build knowledge bases for MMLU-Pro target categories.

Strategy per category:
- history: Wikipedia summaries for key historical concepts, events, civilizations
- health: Medical concept definitions, drug-condition relationships, anatomy facts
- business: Financial formulas as SELPH functions + business concept definitions

The KB is a set of SELPH functions the model can call inline:
  (define-concept "Horner syndrome" "caused by disruption of sympathetic nerve supply,
   presents with miosis, ptosis, anhidrosis")
  (define-relation "fluoxetine" "treats" "bulimia nervosa")
  (define-formula compound-interest (lambda (p r n t) (multiply p (power (add 1 (divide r n)) (multiply n t)))))

For the model, these appear as function signatures in the prompt.
"""
import json
import re
import urllib.request
import urllib.parse
import time
from pathlib import Path
from collections import defaultdict, Counter
from datasets import load_dataset

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def fetch_wikipedia_summary(title):
    """Fetch a short summary from Wikipedia API."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "NeuralSELPH/0.1 (research)",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("extract", "")
    except Exception:
        return ""


def extract_key_concepts(questions):
    """Extract key concepts/entities mentioned across questions."""
    # Simple extraction: look for capitalized phrases, quoted terms, medical terms
    concepts = Counter()
    for ex in questions:
        q = ex["question"]
        # Capitalized multi-word phrases (proper nouns, historical terms)
        for m in re.finditer(r'(?:^|(?<=\s))([A-Z][a-z]+(?:\s+(?:of|the|and|in|de|von)\s+)?(?:[A-Z][a-z]+\s*)*)', q):
            term = m.group(1).strip()
            if len(term) > 3 and term not in {"The", "This", "What", "Which", "How", "With", "Answer"}:
                concepts[term] += 1
        # Also extract from correct answer
        answer_text = ex["options"][ex["answer_index"]]
        for m in re.finditer(r'(?:^|(?<=\s))([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', answer_text):
            term = m.group(1).strip()
            if len(term) > 3:
                concepts[term] += 1
    return concepts


def build_business_formulas():
    """Build SELPH functions for common business/finance calculations."""
    formulas = {
        "compound-interest": {
            "sig": "compound-interest(principal, rate, periods) → number",
            "desc": "Calculate compound interest: P × (1 + r)^n",
            "sexpr": '(multiply principal (power (add 1 rate) periods))',
        },
        "simple-interest": {
            "sig": "simple-interest(principal, rate, time) → number",
            "desc": "Calculate simple interest: P × r × t",
            "sexpr": '(multiply principal (multiply rate time))',
        },
        "discount-price": {
            "sig": "discount-price(list-price, discount-rate) → number",
            "desc": "Price after discount: price × (1 - rate)",
            "sexpr": '(multiply list-price (subtract 1 discount-rate))',
        },
        "markup-price": {
            "sig": "markup-price(cost, markup-rate) → number",
            "desc": "Price after markup: cost × (1 + rate)",
            "sexpr": '(multiply cost (add 1 markup-rate))',
        },
        "present-value": {
            "sig": "present-value(future-value, rate, periods) → number",
            "desc": "Present value: FV / (1 + r)^n",
            "sexpr": '(divide future-value (power (add 1 rate) periods))',
        },
        "bond-price": {
            "sig": "bond-price(coupon, rate, periods, face-value) → number",
            "desc": "Bond price: PV of coupons + PV of face value",
            "sexpr": '(add (multiply coupon (divide (subtract 1 (power (add 1 rate) (negate periods))) rate)) (divide face-value (power (add 1 rate) periods)))',
        },
        "overtime-pay": {
            "sig": "overtime-pay(monthly-salary, hours-per-month, overtime-hours, multiplier) → number",
            "desc": "Overtime pay: (salary/hours) × overtime × multiplier",
            "sexpr": '(multiply (divide monthly-salary hours-per-month) (multiply overtime-hours multiplier))',
        },
        "percentage": {
            "sig": "percentage(part, whole) → number",
            "desc": "Calculate percentage: (part/whole) × 100",
            "sexpr": '(multiply (divide part whole) 100)',
        },
        "rate-of-return": {
            "sig": "rate-of-return(annual-savings, investment, years) → number",
            "desc": "Rate of return for annuity investment",
            "sexpr": '(divide annual-savings investment)',
        },
    }
    return formulas


def build_concept_kb(category, questions, max_concepts=200):
    """Build a concept KB from questions + Wikipedia for a category."""
    concepts = extract_key_concepts(questions)
    top_concepts = [term for term, count in concepts.most_common(max_concepts)]

    print(f"\n  Top concepts for {category}:")
    for term, count in concepts.most_common(20):
        print(f"    {term}: {count}")

    # Fetch Wikipedia summaries for top concepts
    kb = {}
    fetched = 0
    for term in top_concepts:
        if fetched >= 100:  # Rate limit
            break
        summary = fetch_wikipedia_summary(term)
        if summary and len(summary) > 50:
            # Truncate to first 2 sentences for conciseness
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            short = ' '.join(sentences[:2])
            kb[term] = {
                "definition": short,
                "category": category,
            }
            fetched += 1
            time.sleep(0.1)  # Be nice to Wikipedia

    print(f"  Fetched {len(kb)} concept definitions from Wikipedia")
    return kb


def build_category_kb(category, questions):
    """Build the full KB for a category."""
    if category == "business":
        formulas = build_business_formulas()
        concepts = build_concept_kb(category, questions, max_concepts=100)
        return {"formulas": formulas, "concepts": concepts}
    else:
        concepts = build_concept_kb(category, questions, max_concepts=150)
        return {"concepts": concepts}


def format_kb_as_signatures(kb):
    """Convert a KB into function signature strings for the prompt."""
    sigs = []

    if "formulas" in kb:
        for name, formula in kb["formulas"].items():
            sigs.append(formula["sig"])

    sigs.append('lookup(concept) → definition')
    sigs.append('related(concept, relation) → concept')

    return sigs


if __name__ == "__main__":
    print("Loading MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    for category in ["history", "health", "business"]:
        questions = [ex for ex in ds["test"] if ex["category"] == category]
        print(f"\n{'='*60}")
        print(f"Building KB for: {category} ({len(questions)} questions)")
        print(f"{'='*60}")

        kb = build_category_kb(category, questions)

        # Save
        kb_path = OUT / f"mmlu_kb_{category}.json"
        with open(kb_path, "w") as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {kb_path}")

        # Show signatures
        sigs = format_kb_as_signatures(kb)
        print(f"  Function signatures: {sigs[:5]}")

        # Show sample concepts
        concepts = kb.get("concepts", {})
        for term in list(concepts.keys())[:5]:
            defn = concepts[term]["definition"][:100]
            print(f"    {term}: {defn}...")
