"""Explore Wikidata SPARQL for geography KB extraction.

Uses the Wikidata Query Service to pull structured facts about countries,
capitals, populations, etc. — the building blocks for compositional
s-expression training data.
"""
import json
import urllib.request
import urllib.parse
import time

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def sparql_query(query):
    """Execute a SPARQL query against Wikidata."""
    params = urllib.parse.urlencode({
        "query": query,
        "format": "json",
    })
    url = f"{WIKIDATA_SPARQL}?{params}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "NeuralSELPH/0.1 (research project)",
        "Accept": "application/sparql-results+json",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# Query 1: Countries with capitals, populations, continents, currencies, languages
print("=== Fetching country data ===")
country_query = """
SELECT ?country ?countryLabel ?capitalLabel ?population ?continentLabel
       ?currencyLabel ?headOfStateLabel ?areaKm2
WHERE {
  ?country wdt:P31 wd:Q6256.           # instance of: country
  OPTIONAL { ?country wdt:P36 ?capital. }       # capital
  OPTIONAL { ?country wdt:P1082 ?population. }  # population
  OPTIONAL { ?country wdt:P30 ?continent. }     # continent
  OPTIONAL { ?country wdt:P38 ?currency. }      # currency
  OPTIONAL { ?country wdt:P35 ?headOfState. }   # head of state
  OPTIONAL { ?country wdt:P2046 ?areaKm2. }     # area
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?countryLabel
"""

results = sparql_query(country_query)
countries = results["results"]["bindings"]
print(f"Raw rows: {len(countries)}")

# Deduplicate by country (take first row per country)
seen = set()
unique = []
for row in countries:
    name = row.get("countryLabel", {}).get("value", "")
    if name and name not in seen:
        seen.add(name)
        unique.append(row)

print(f"Unique countries: {len(unique)}")

# Show first 10
for row in unique[:10]:
    name = row.get("countryLabel", {}).get("value", "?")
    cap = row.get("capitalLabel", {}).get("value", "?")
    pop = row.get("population", {}).get("value", "?")
    cont = row.get("continentLabel", {}).get("value", "?")
    curr = row.get("currencyLabel", {}).get("value", "?")
    head = row.get("headOfStateLabel", {}).get("value", "?")
    area = row.get("areaKm2", {}).get("value", "?")
    print(f"  {name}: capital={cap}, pop={pop}, continent={cont}, currency={curr}, head={head}, area={area}")

time.sleep(1)  # be nice to the API

# Query 2: Capitals with their own populations (for composition)
print("\n=== Fetching capital city populations ===")
capital_query = """
SELECT ?city ?cityLabel ?population ?countryLabel
WHERE {
  ?country wdt:P31 wd:Q6256.
  ?country wdt:P36 ?city.
  ?city wdt:P1082 ?population.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?cityLabel
"""

results = sparql_query(capital_query)
capitals = results["results"]["bindings"]
print(f"Capitals with population: {len(capitals)}")

# Deduplicate
seen_cap = set()
unique_cap = []
for row in capitals:
    name = row.get("cityLabel", {}).get("value", "")
    if name and name not in seen_cap:
        seen_cap.add(name)
        unique_cap.append(row)

print(f"Unique capitals with population: {len(unique_cap)}")

for row in unique_cap[:10]:
    city = row.get("cityLabel", {}).get("value", "?")
    pop = row.get("population", {}).get("value", "?")
    country = row.get("countryLabel", {}).get("value", "?")
    print(f"  {city} ({country}): pop={pop}")

time.sleep(1)

# Query 3: Official languages per country
print("\n=== Fetching official languages ===")
lang_query = """
SELECT ?countryLabel ?languageLabel
WHERE {
  ?country wdt:P31 wd:Q6256.
  ?country wdt:P37 ?language.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?countryLabel
"""

results = sparql_query(lang_query)
languages = results["results"]["bindings"]
print(f"Country-language pairs: {len(languages)}")

# Show first 15
for row in languages[:15]:
    country = row.get("countryLabel", {}).get("value", "?")
    lang = row.get("languageLabel", {}).get("value", "?")
    print(f"  {country}: {lang}")

# Summary
print(f"\n=== Summary ===")
print(f"Countries: {len(unique)}")
print(f"Capitals with population: {len(unique_cap)}")
print(f"Language pairs: {len(languages)}")
print(f"\nComposition potential:")
print(f"  (population (capital X)) — {len(unique_cap)} examples")
print(f"  (continent (capital X)) — needs capital→country→continent chain")
print(f"  (currency (country-of city)) — needs city→country→currency chain")
