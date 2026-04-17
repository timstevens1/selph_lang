"""Build a geography knowledge base from Wikidata and generate
compositional training data for Neural SELPH.

Generates:
1. A SELPH KB file (geo_kb.selph) with lookup functions
2. Training data with NL prompts → s-expression pairs at multiple depths:
   - Depth 1: (capital "Spain") → "Madrid"
   - Depth 2: (population (capital "Spain")) → 3200000
   - Depth 3: (language (largest-by-population (continent "Europe"))) → ...
"""
import json
import random
import urllib.request
import urllib.parse
import time
from pathlib import Path
from collections import defaultdict

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def sparql_query(query):
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    url = f"{WIKIDATA_SPARQL}?{params}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "NeuralSELPH/0.1 (research project)",
        "Accept": "application/sparql-results+json",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def fetch_all():
    """Fetch all geography data from Wikidata."""
    print("Fetching country data...")
    country_q = """
    SELECT ?country ?countryLabel ?capitalLabel ?population ?continentLabel
           ?currencyLabel ?headOfStateLabel ?areaKm2
    WHERE {
      ?country wdt:P31 wd:Q6256.
      OPTIONAL { ?country wdt:P36 ?capital. }
      OPTIONAL { ?country wdt:P1082 ?population. }
      OPTIONAL { ?country wdt:P30 ?continent. }
      OPTIONAL { ?country wdt:P38 ?currency. }
      OPTIONAL { ?country wdt:P35 ?headOfState. }
      OPTIONAL { ?country wdt:P2046 ?areaKm2. }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    raw = sparql_query(country_q)["results"]["bindings"]

    # Deduplicate, taking first value per country
    countries = {}
    for row in raw:
        name = row.get("countryLabel", {}).get("value")
        if not name or name in countries:
            continue
        countries[name] = {
            "capital": row.get("capitalLabel", {}).get("value"),
            "population": row.get("population", {}).get("value"),
            "continent": row.get("continentLabel", {}).get("value"),
            "currency": row.get("currencyLabel", {}).get("value"),
            "head_of_state": row.get("headOfStateLabel", {}).get("value"),
            "area_km2": row.get("areaKm2", {}).get("value"),
        }

    time.sleep(1)

    print("Fetching capital populations...")
    cap_q = """
    SELECT ?cityLabel ?population ?countryLabel
    WHERE {
      ?country wdt:P31 wd:Q6256.
      ?country wdt:P36 ?city.
      ?city wdt:P1082 ?population.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    raw = sparql_query(cap_q)["results"]["bindings"]
    capital_pops = {}
    for row in raw:
        city = row.get("cityLabel", {}).get("value")
        if city and city not in capital_pops:
            capital_pops[city] = {
                "population": row.get("population", {}).get("value"),
                "country": row.get("countryLabel", {}).get("value"),
            }

    time.sleep(1)

    print("Fetching official languages...")
    lang_q = """
    SELECT ?countryLabel ?languageLabel
    WHERE {
      ?country wdt:P31 wd:Q6256.
      ?country wdt:P37 ?language.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    raw = sparql_query(lang_q)["results"]["bindings"]
    # Take first language per country
    languages = {}
    for row in raw:
        country = row.get("countryLabel", {}).get("value")
        lang = row.get("languageLabel", {}).get("value")
        if country and lang and country not in languages:
            languages[country] = lang

    return countries, capital_pops, languages


def build_kb(countries, capital_pops, languages):
    """Build structured KB dictionary."""
    kb = {}
    for name, data in countries.items():
        entry = {"name": name}
        if data["capital"]:
            entry["capital"] = data["capital"]
        if data["population"]:
            try:
                entry["population"] = int(float(data["population"]))
            except (ValueError, TypeError):
                pass
        if data["continent"]:
            entry["continent"] = data["continent"]
        if data["currency"]:
            entry["currency"] = data["currency"]
        if data["head_of_state"]:
            entry["head_of_state"] = data["head_of_state"]
        if data["area_km2"]:
            try:
                entry["area_km2"] = round(float(data["area_km2"]))
            except (ValueError, TypeError):
                pass
        if name in languages:
            entry["language"] = languages[name]
        kb[name] = entry

    # Add capital populations
    for city, data in capital_pops.items():
        if data["population"]:
            try:
                pop = int(float(data["population"]))
                # Find which country this is the capital of
                for name, entry in kb.items():
                    if entry.get("capital") == city:
                        entry["capital_population"] = pop
                        break
            except (ValueError, TypeError):
                pass

    # Build reverse lookups
    capital_to_country = {}
    for name, entry in kb.items():
        if "capital" in entry:
            capital_to_country[entry["capital"]] = name

    # Continent aggregates
    continent_countries = defaultdict(list)
    for name, entry in kb.items():
        if "continent" in entry and "population" in entry:
            continent_countries[entry["continent"]].append((name, entry["population"]))

    largest_by_continent = {}
    for cont, pairs in continent_countries.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        largest_by_continent[cont] = pairs[0][0]

    return kb, capital_to_country, largest_by_continent


def generate_training_data(kb, capital_to_country, largest_by_continent):
    """Generate (NL prompt, s-expression, answer) triples at multiple depths."""
    examples = []

    # Helper to quote strings in s-expressions
    def q(s):
        return f'"{s}"'

    # === DEPTH 1: Direct lookups ===
    for name, entry in kb.items():
        if "capital" in entry:
            examples.append({
                "depth": 1,
                "nl": f"What is the capital of {name}?",
                "sexpr": f'(capital {q(name)})',
                "answer": entry["capital"],
                "type": "string",
            })
        if "population" in entry:
            examples.append({
                "depth": 1,
                "nl": f"What is the population of {name}?",
                "sexpr": f'(population {q(name)})',
                "answer": entry["population"],
                "type": "number",
            })
        if "continent" in entry:
            examples.append({
                "depth": 1,
                "nl": f"What continent is {name} in?",
                "sexpr": f'(continent {q(name)})',
                "answer": entry["continent"],
                "type": "string",
            })
        if "currency" in entry:
            examples.append({
                "depth": 1,
                "nl": f"What currency does {name} use?",
                "sexpr": f'(currency {q(name)})',
                "answer": entry["currency"],
                "type": "string",
            })
        if "language" in entry:
            examples.append({
                "depth": 1,
                "nl": f"What is the official language of {name}?",
                "sexpr": f'(language {q(name)})',
                "answer": entry["language"],
                "type": "string",
            })
        if "area_km2" in entry:
            examples.append({
                "depth": 1,
                "nl": f"What is the area of {name} in square kilometers?",
                "sexpr": f'(area {q(name)})',
                "answer": entry["area_km2"],
                "type": "number",
            })
        if "head_of_state" in entry:
            examples.append({
                "depth": 1,
                "nl": f"Who is the head of state of {name}?",
                "sexpr": f'(head-of-state {q(name)})',
                "answer": entry["head_of_state"],
                "type": "string",
            })

    # === DEPTH 2: One composition ===
    for name, entry in kb.items():
        cap = entry.get("capital")
        if not cap:
            continue

        # (population (capital X)) — population of the capital city
        if "capital_population" in entry:
            examples.append({
                "depth": 2,
                "nl": f"What is the population of the capital of {name}?",
                "sexpr": f'(population (capital {q(name)}))',
                "answer": entry["capital_population"],
                "type": "number",
            })

        # (continent (country-of capital)) — what continent is the capital's country in
        if "continent" in entry:
            examples.append({
                "depth": 2,
                "nl": f"What continent is the capital of {name} in?",
                "sexpr": f'(continent (country-of (capital {q(name)})))',
                "answer": entry["continent"],
                "type": "string",
                "depth": 3,  # actually depth 3
            })

        # (currency (country-of capital)) — what currency in the capital
        if "currency" in entry:
            examples.append({
                "depth": 2,
                "nl": f"What currency is used in the capital of {name}?",
                "sexpr": f'(currency {q(name)})',
                "answer": entry["currency"],
                "type": "string",
            })
            # Alternate phrasing
            examples.append({
                "depth": 2,
                "nl": f"What currency is used in {cap}?",
                "sexpr": f'(currency (country-of {q(cap)}))',
                "answer": entry["currency"],
                "type": "string",
            })

        # (language (country-of capital))
        if "language" in entry:
            examples.append({
                "depth": 2,
                "nl": f"What language is spoken in {cap}?",
                "sexpr": f'(language (country-of {q(cap)}))',
                "answer": entry["language"],
                "type": "string",
            })

        # (head-of-state (country-of capital))
        if "head_of_state" in entry:
            examples.append({
                "depth": 2,
                "nl": f"Who is the head of state of the country whose capital is {cap}?",
                "sexpr": f'(head-of-state (country-of {q(cap)}))',
                "answer": entry["head_of_state"],
                "type": "string",
            })

    # Computed: population density
    for name, entry in kb.items():
        if "population" in entry and "area_km2" in entry and entry["area_km2"] > 0:
            density = round(entry["population"] / entry["area_km2"])
            examples.append({
                "depth": 2,
                "nl": f"What is the population density of {name}?",
                "sexpr": f'(divide (population {q(name)}) (area {q(name)}))',
                "answer": density,
                "type": "number",
            })

    # === DEPTH 3: Two compositions ===
    for cont, country in largest_by_continent.items():
        entry = kb[country]

        # (capital (largest-by-population (continent X)))
        if "capital" in entry:
            examples.append({
                "depth": 3,
                "nl": f"What is the capital of the most populous country in {cont}?",
                "sexpr": f'(capital (largest-by-population {q(cont)}))',
                "answer": entry["capital"],
                "type": "string",
            })

        # (population (capital (largest-by-population (continent X))))
        if "capital_population" in entry:
            examples.append({
                "depth": 4,
                "nl": f"What is the population of the capital of the most populous country in {cont}?",
                "sexpr": f'(population (capital (largest-by-population {q(cont)})))',
                "answer": entry["capital_population"],
                "type": "number",
            })

        # (language (largest-by-population (continent X)))
        if "language" in entry:
            examples.append({
                "depth": 3,
                "nl": f"What language is spoken in the most populous country in {cont}?",
                "sexpr": f'(language (largest-by-population {q(cont)}))',
                "answer": entry["language"],
                "type": "string",
            })

        # (currency (largest-by-population (continent X)))
        if "currency" in entry:
            examples.append({
                "depth": 3,
                "nl": f"What currency is used in the most populous country in {cont}?",
                "sexpr": f'(currency (largest-by-population {q(cont)}))',
                "answer": entry["currency"],
                "type": "string",
            })

    return examples


def build_function_signatures(kb):
    """Build function signature block from what's actually in the KB.

    This is the key to generalization: the model reads available functions
    from the prompt, not from memorized training. When SELPH learns new
    functions, they appear here automatically.
    """
    # Detect which properties exist in the KB
    has = lambda prop: any(prop in e for e in kb.values())

    sigs = []
    if has("capital"):
        sigs.append("capital(country) → city")
    if has("population"):
        sigs.append("population(entity) → number")
    if has("continent"):
        sigs.append("continent(country) → continent")
    if has("currency"):
        sigs.append("currency(country) → currency")
    if has("language"):
        sigs.append("language(country) → language")
    if has("area_km2"):
        sigs.append("area(country) → number")
    if has("head_of_state"):
        sigs.append("head-of-state(country) → person")
    if has("capital_population"):
        # This is derived — population works on cities too
        pass
    # Inverse lookups
    sigs.append("country-of(city) → country")
    # Aggregation
    sigs.append("largest-by-population(continent) → country")
    # Arithmetic (always available)
    sigs.append("divide(a, b) → number")

    return sigs


def format_for_training(examples, kb):
    """Format examples as MLX LoRA completions with <tool_call> delimiters.

    Each prompt includes the function signature block so the model learns
    to read what's available rather than memorize a fixed function set.
    """
    sigs = build_function_signatures(kb)
    sig_block = "Functions: " + ", ".join(sigs)

    formatted = []
    for ex in examples:
        prompt = f"{sig_block}\n\nQ: {ex['nl']}\nA:"
        completion = f" <tool_call>{ex['sexpr']}</tool_call>"
        formatted.append({
            "prompt": prompt,
            "completion": completion,
            "answer": str(ex["answer"]),
            "depth": ex["depth"],
            "type": ex["type"],
        })
    return formatted


if __name__ == "__main__":
    countries, capital_pops, languages = fetch_all()
    kb, capital_to_country, largest_by_continent = build_kb(countries, capital_pops, languages)

    print(f"\nKB: {len(kb)} countries")
    print(f"  with capital: {sum(1 for e in kb.values() if 'capital' in e)}")
    print(f"  with population: {sum(1 for e in kb.values() if 'population' in e)}")
    print(f"  with capital_population: {sum(1 for e in kb.values() if 'capital_population' in e)}")
    print(f"  with continent: {sum(1 for e in kb.values() if 'continent' in e)}")
    print(f"  with currency: {sum(1 for e in kb.values() if 'currency' in e)}")
    print(f"  with language: {sum(1 for e in kb.values() if 'language' in e)}")
    print(f"  with area: {sum(1 for e in kb.values() if 'area_km2' in e)}")
    print(f"  with head_of_state: {sum(1 for e in kb.values() if 'head_of_state' in e)}")

    # Save KB
    kb_path = OUT / "geo_kb.json"
    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"\nSaved KB to {kb_path}")

    # Generate training data
    examples = generate_training_data(kb, capital_to_country, largest_by_continent)
    formatted = format_for_training(examples, kb)

    # Count by depth
    from collections import Counter
    depth_counts = Counter(ex["depth"] for ex in formatted)
    print(f"\nTraining examples: {len(formatted)}")
    for d in sorted(depth_counts):
        print(f"  Depth {d}: {depth_counts[d]}")

    # Show samples at each depth
    for d in sorted(depth_counts):
        print(f"\n--- Depth {d} samples ---")
        depth_examples = [ex for ex in formatted if ex["depth"] == d]
        for ex in random.Random(42).sample(depth_examples, min(3, len(depth_examples))):
            print(f"  {ex['prompt']}")
            print(f"  {ex['completion']}")
            print(f"  Answer: {ex['answer']}")
            print()

    # Save training data
    train_path = OUT / "geo_train.json"
    with open(train_path, "w") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)
    print(f"Saved training data to {train_path}")
