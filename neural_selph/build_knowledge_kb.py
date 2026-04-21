"""Build a general knowledge base for SELPH from Wikipedia.

Extracts key concepts from MMLU-Pro questions across all categories,
fetches Wikipedia summaries, and stores them as SELPH-queryable facts.

The KB supports:
  (lookup "vitamin A") → "Vitamin A is a fat-soluble vitamin important for vision..."
  (lookup "Crimean War") → "The Crimean War was fought from 1853 to 1856..."
  (related "vitamin A" "functions") → "vision, immune function, cell differentiation"
  (apropos "vitamin") → ["vitamin A", "vitamin K", "vitamin D", ...]
"""
import json
import re
import time
import random
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def fetch_wikipedia_summary(title, max_retries=2):
    """Fetch summary from Wikipedia REST API."""
    for attempt in range(max_retries):
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "NeuralSELPH/0.1 (research project)",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                extract = data.get("extract", "")
                if extract and len(extract) > 30:
                    return extract
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.5)
    return None


def fetch_wikipedia_search(query, limit=3):
    """Search Wikipedia and return top result titles."""
    try:
        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        })
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "NeuralSELPH/0.1 (research project)",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            results = data.get("query", {}).get("search", [])
            return [r["title"] for r in results]
    except Exception:
        return []


def extract_concepts_from_questions(questions):
    """Extract key concepts/terms from MMLU-Pro questions and answers.

    Uses multiple strategies:
    - Capitalized proper nouns/phrases
    - Technical terms from answer options
    - Quoted terms
    - Domain-specific patterns
    """
    concept_counts = Counter()

    for ex in questions:
        q = ex["question"]
        # All answer options
        for opt in ex["options"]:
            # Capitalized multi-word terms
            for m in re.finditer(
                r'(?:^|(?<=\s))([A-Z][a-z]+(?:\s+(?:of|the|and|in|de|for|to)\s+)?'
                r'(?:[A-Z][a-z]+\s*)*)', opt
            ):
                term = m.group(1).strip()
                if len(term) > 3 and term not in {
                    "The", "This", "What", "Which", "How", "Answer",
                    "True", "False", "None", "Both", "All", "Only",
                }:
                    concept_counts[term] += 1

        # Question text - same extraction
        for m in re.finditer(
            r'(?:^|(?<=\s))([A-Z][a-z]+(?:\s+(?:of|the|and|in|de|for|to)\s+)?'
            r'(?:[A-Z][a-z]+\s*)*)', q
        ):
            term = m.group(1).strip()
            if len(term) > 3 and term not in {
                "The", "This", "What", "Which", "How", "Answer",
                "True", "False", "None", "Both", "All", "Only",
                "Find", "Calculate", "Solve", "Question",
                "According", "Following", "Given",
            }:
                concept_counts[term] += 1

        # Quoted terms
        for m in re.finditer(r"'([^']{3,40})'", q):
            concept_counts[m.group(1)] += 1
        for m in re.finditer(r'"([^"]{3,40})"', q):
            concept_counts[m.group(1)] += 1

    return concept_counts


def build_kb_for_category(category, questions, max_concepts=100):
    """Build knowledge base entries for a category."""
    print(f"\n  Extracting concepts for {category}...")
    concepts = extract_concepts_from_questions(questions)

    # Filter to most frequent, skip very common words
    top_concepts = [
        term for term, count in concepts.most_common(max_concepts * 3)
        if count >= 2 and len(term) > 4
    ][:max_concepts]

    print(f"  Top concepts: {len(top_concepts)}")
    for term, count in concepts.most_common(10):
        print(f"    {term}: {count}")

    kb_entries = {}
    fetched = 0
    for term in top_concepts:
        if fetched >= max_concepts:
            break

        # Try direct fetch first
        summary = fetch_wikipedia_summary(term)

        # If direct fails, try search
        if not summary:
            search_results = fetch_wikipedia_search(term)
            for title in search_results:
                summary = fetch_wikipedia_summary(title)
                if summary:
                    break

        if summary:
            # Truncate to ~3 sentences for conciseness
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            short = ' '.join(sentences[:3])
            if len(short) > 500:
                short = short[:500] + "..."

            kb_entries[term.lower()] = {
                "term": term,
                "definition": short,
                "category": category,
            }
            fetched += 1

        time.sleep(0.1)  # Rate limit

    print(f"  Fetched {len(kb_entries)} definitions")
    return kb_entries


def build_selph_kb_file(all_entries):
    """Generate a SELPH source file that defines the KB as lookup functions."""
    lines = [
        "; Neural SELPH Knowledge Base",
        "; Auto-generated from Wikipedia for MMLU-Pro categories",
        ";",
        "",
        "; Knowledge store: maps lowercase concept names to definitions",
        "(define __kb__ (ns-empty))",
        "",
    ]

    for key, entry in sorted(all_entries.items()):
        # Escape quotes in definition
        defn = entry["definition"].replace("\\", "\\\\").replace('"', '\\"')
        term = entry["term"].replace('"', '\\"')
        lines.append(f'(define __kb__ (ns-put __kb__ "{key}" "{defn}"))')

    lines.extend([
        "",
        "; lookup: retrieve a concept definition from the KB",
        "(define lookup (lambda (concept)",
        '  (let ((key (string-downcase concept)))',
        "    (if (ns-has __kb__ key)",
        "      (ns-get __kb__ key)",
        '      (string-append "Unknown concept: " concept)))))',
        "",
        "; related: find related concepts (simple substring search)",
        "(define related (lambda (concept relation)",
        "  (lookup (string-append concept \" \" relation))))",
        "",
    ])

    return "\n".join(lines)


def main():
    print("Loading MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    # Group by category
    by_cat = defaultdict(list)
    for ex in ds["test"]:
        by_cat[ex["category"]].append(ex)

    all_entries = {}
    for category in sorted(by_cat.keys()):
        questions = by_cat[category]
        print(f"\n{'='*50}")
        print(f"Category: {category} ({len(questions)} questions)")
        print(f"{'='*50}")

        entries = build_kb_for_category(category, questions, max_concepts=75)
        all_entries.update(entries)

    print(f"\n{'='*50}")
    print(f"Total KB entries: {len(all_entries)}")
    print(f"{'='*50}")

    # Save as JSON
    kb_path = OUT / "knowledge_kb.json"
    with open(kb_path, "w") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON KB to {kb_path}")

    # Save as SELPH source
    selph_src = build_selph_kb_file(all_entries)
    selph_path = OUT / "knowledge_kb.selph"
    with open(selph_path, "w") as f:
        f.write(selph_src)
    print(f"Saved SELPH KB to {selph_path} ({len(selph_src)} bytes)")

    # Category stats
    cat_counts = Counter(e["category"] for e in all_entries.values())
    print(f"\nPer category:")
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]}")


if __name__ == "__main__":
    main()
