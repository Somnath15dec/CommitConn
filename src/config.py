"""Shared vocabulary config used by prep (desc) and query generation (Step 5+)."""

# Canonical context word -> synonyms. Desc carries canonical words only; Step 5 may phrase
# queries with any synonym, and evaluation treats a synonym as its canonical word.
CONTEXT_SYNONYMS = {
    "sleep": ["wind down"],
    "study": ["focus", "late night"],
    "rainy day": ["reflective"],
}
SYNONYM_TO_CANONICAL = {s: c for c, syns in CONTEXT_SYNONYMS.items() for s in syns}


def canonical(word: str) -> str:
    return SYNONYM_TO_CANONICAL.get(word, word)


# Spotify category -> audio tags it contradicts. A contradicted category is left out of desc
# (the `category` column keeps it).
CATEGORY_CONFLICTS = {
    "happy": {"melancholic"},
    "sad": {"happy"},
    "sleep": {"energetic"},
    "study": {"energetic"},
    "party": {"calm"},
}


def desc_categories(category, tags) -> list[str]:
    t = set(tags)
    return [c for c in category if not (CATEGORY_CONFLICTS.get(c, set()) & t)]
