"""Step 5: templated synthetic queries (no LLM).

Train and eval use DISJOINT template sets. Slots fill only from a track's own tags, context and
genres. Spotify `category` may fill TRAIN templates only. Playlist names and MuSe scores are
never used. Context slots may use any synonym from CONTEXT_SYNONYMS; `wanted_tags` always stores
the canonical word.
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from src.config import CONTEXT_SYNONYMS, desc_categories
from src.paths import CATALOG, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, TRAIN_PAIRS
from src.prep import CONTEXT_TO_PLAYLIST_WORDS

TRAIN_QUERIES_PER_TRACK = 5
EVAL_N_TRACKS = 500
TRAIN_RANDOM_STATE = 6
EVAL_RANDOM_STATE = 7
MIN_PAIRS_PER_CONTEXT = 300   # report-only flag
EVAL_B_PER_WORD = 12
EVAL_B_RANDOM_STATE = 8

# Genre ids keep hyphens in these; every other "-" becomes a space in query text.
GENRE_KEEP_HYPHEN = {"hip-hop", "k-pop", "j-pop", "j-rock", "j-dance", "j-idol"}
GENRE_SURFACE_OVERRIDES = {"r-n-b": "r&b"}


def humanize_genre(genre: str) -> str:
    """Surface form of a genre id in query text (wanted_tags keep the id)."""
    if genre in GENRE_SURFACE_OVERRIDES:
        return GENRE_SURFACE_OVERRIDES[genre]
    return genre if genre in GENRE_KEEP_HYPHEN else genre.replace("-", " ")

# Slot vocab (tag -> surface form). "neutral mood" and "moderate energy" are not requestable.
MOOD_TAGS = {"happy": "happy", "melancholic": "melancholic", "calm": "calm", "energetic": "energetic"}
TEMPO_TAGS = {"slow": "slow", "fast": "fast"}
SOUND_TAGS = {"acoustic": "acoustic", "danceable": "danceable", "instrumental": "instrumental",
              "spoken word or rap": "rap"}
# Spotify category -> slot (TRAIN only). Moods fill {mood}, activities fill {context}.
CATEGORY_MOODS = {"happy": "happy", "sad": "sad", "chill": "chill", "romance": "romantic"}
CATEGORY_CONTEXTS = {"party", "study", "sleep"}
CATEGORY_VALUES = set(CATEGORY_MOODS) | CATEGORY_CONTEXTS

# Context word (canonical or synonym) -> noun phrase used in "{context}" slots.
CONTEXT_PHRASES = {
    "workout": "a workout", "running": "running", "gym": "the gym", "morning": "the morning",
    "coffee": "a coffee break", "relaxing": "relaxing", "feel-good": "feeling good",
    "dance": "dancing", "intense": "an intense session", "cathartic": "a cathartic release",
    "party": "a party", "study": "studying", "sleep": "sleep", "rainy day": "a rainy day",
    # synonyms
    "wind down": "winding down", "focus": "focusing", "late night": "late nights",
    "reflective": "reflective moments",
}

# template_id -> (type, text)
TRAIN_TEMPLATES = {
    "tr_mood_1": ("mood", "something {mood}"),
    "tr_mood_2": ("mood", "{mood} music"),
    "tr_mood_3": ("mood", "songs that sound {mood}"),
    "tr_context_1": ("context", "music for {context}"),
    "tr_context_2": ("context", "a playlist for {context}"),
    "tr_context_3": ("context", "songs to put on for {context}"),
    "tr_genre_1": ("genre", "{genre} songs"),
    "tr_genre_2": ("genre", "some {genre} music"),
    "tr_genre_3": ("genre", "{genre} tracks please"),
    "tr_combo_1": ("combo", "{mood} {genre} for {context}"),
    "tr_combo_2": ("combo", "{mood} songs for {context}"),
    "tr_combo_3": ("combo", "{genre} for {context}"),
    "tr_sound_1": ("sound", "{tempo} {sound} tracks"),
    "tr_sound_2": ("sound", "{tempo} songs"),
    "tr_sound_3": ("sound", "{sound} music"),
}
EVAL_TEMPLATES = {
    "ev_combo_1": ("combo", "I need {mood} vibes for {context}"),
    "ev_combo_2": ("combo", "play me some {genre} that feels {mood}"),
    "ev_combo_3": ("combo", "looking for {genre} that works for {context}"),
    "ev_combo_4": ("combo", "need a {mood} soundtrack for {context}"),
    "ev_context_1": ("context", "any recommendations for {context}?"),
    "ev_mood_1": ("mood", "give me tracks with a {mood} feel"),
    "ev_genre_1": ("genre", "I'm into {genre} lately, anything good?"),
    "ev_sound_1": ("sound", "can you find {tempo} {sound} stuff"),
}
# Eval B (headline set): context slot only, EVAL-style phrasings, disjoint from TRAIN_TEMPLATES.
EVAL_B_TEMPLATES = {
    "evb_1": "any recommendations for {context}?",
    "evb_2": "need a soundtrack for {context}",
    "evb_3": "put something on for {context}",
    "evb_4": "I want tunes for {context}",
    "evb_5": "got anything good for {context}?",
    "evb_6": "help me pick songs for {context}",
    "evb_7": "queue up some tracks for {context}",
    "evb_8": "what would you play for {context}?",
    "evb_9": "pick the perfect tracks for {context}",
    "evb_10": "background audio for {context}, please",
    "evb_11": "recommend me something for {context}",
    "evb_12": "what's good to listen to for {context}?",
}
TYPE_ORDER = ["context", "combo", "mood", "genre", "sound"]
_SLOT_RE = re.compile(r"\{(\w+)\}")


def slots_of(text: str) -> list[str]:
    return _SLOT_RE.findall(text)


def eval_banned_words(row) -> set[str]:
    """Words an eval query for this track must not contain: its category values and its
    primary_genre when that is not one of its genres (category-only primary_genre)."""
    banned = set(row.category)
    if row.primary_genre not in set(row.genres):
        banned.add(row.primary_genre)
    return banned


def contains_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text, flags=re.I) is not None


def slot_options(row, split: str) -> dict[str, list[tuple[str, str]]]:
    """slot -> [(surface, canonical)] from the track's own tags/context/genres (+category in train)."""
    tags = set(row.tags)
    opts = {
        "mood": [(s, t) for t, s in MOOD_TAGS.items() if t in tags],
        "tempo": [(s, t) for t, s in TEMPO_TAGS.items() if t in tags],
        "sound": [(s, t) for t, s in SOUND_TAGS.items() if t in tags],
        "genre": [(humanize_genre(g), g) for g in row.genres],
    }
    contexts = list(dict.fromkeys(row.context))
    if split == "train":
        cats = desc_categories(row.category, row.tags)  # drop categories contradicting audio tags
        opts["mood"] += [(CATEGORY_MOODS[c], c) for c in cats
                         if c in CATEGORY_MOODS and c not in {m for _, m in opts["mood"]}]
        contexts += [c for c in cats if c in CATEGORY_CONTEXTS and c not in contexts]
    opts["context"] = [(CONTEXT_PHRASES[w], c) for c in contexts for w in [c, *CONTEXT_SYNONYMS.get(c, [])]]
    if split == "eval":
        banned = eval_banned_words(row)
        opts = {k: [o for o in v if not any(contains_word(o[0], b) or o[1] == b for b in banned)]
                for k, v in opts.items()}
    return opts


def fill(text: str, opts, rng, ctx_order: list[str] | None = None, ctx_i: int = 0):
    """Return (query, wanted_tags). Context picks the canonical word ctx_order[ctx_i % n], then a
    random surface (canonical or synonym) for it."""
    values, wanted = {}, []
    for slot in slots_of(text):
        choices = opts[slot]
        if slot == "context" and ctx_order:
            want = ctx_order[ctx_i % len(ctx_order)]
            choices = [o for o in choices if o[1] == want]
        surface, canon = choices[rng.integers(len(choices))]
        values[slot] = surface
        wanted.append(canon)
    return text.format(**values), list(dict.fromkeys(wanted))


def fillable(text: str, opts) -> bool:
    return all(opts.get(s) for s in slots_of(text))


def train_queries_for(row, rng, ctx_freq: Counter, n: int = TRAIN_QUERIES_PER_TRACK) -> list[dict]:
    """Context slots take the track's context words rarest-first (catalog-wide), so thin words
    such as sleep get a query from every track that carries them."""
    opts = slot_options(row, "train")
    ctx_order = sorted(dict.fromkeys(c for _, c in opts["context"]), key=lambda c: (ctx_freq[c], c))
    out, seen, used, ctx_i = [], set(), set(), 0
    progress = True
    while len(out) < n and progress:
        progress = False
        for typ in TYPE_ORDER:
            if len(out) >= n:
                break
            cands = [tid for tid, (t, txt) in TRAIN_TEMPLATES.items()
                     if t == typ and tid not in used and fillable(txt, opts)]
            if not cands:
                continue
            tid = cands[rng.integers(len(cands))]
            used.add(tid)
            progress = True
            text = TRAIN_TEMPLATES[tid][1]
            q, wanted = fill(text, opts, rng, ctx_order, ctx_i)
            if "context" in slots_of(text):
                ctx_i += 1
            if q not in seen:
                seen.add(q)
                out.append({"query": q, "track_id": row.track_id, "template_id": tid,
                            "type": typ, "wanted_tags": wanted})
    return out


def eval_query_for(row, rng, train_keys: set) -> dict | None:
    opts = slot_options(row, "eval")
    cands = [tid for tid, (_, txt) in EVAL_TEMPLATES.items() if fillable(txt, opts)]
    rng.shuffle(cands)
    banned = eval_banned_words(row)
    for tid in cands:
        typ, text = EVAL_TEMPLATES[tid]
        q, wanted = fill(text, opts, rng)
        if (q, row.track_id) in train_keys or any(contains_word(q, b) for b in banned):
            continue
        return {"query": q, "track_id": row.track_id, "template_id": tid, "type": typ,
                "wanted_tags": wanted}
    return None


def generate(cat: pd.DataFrame):
    rng = np.random.default_rng(TRAIN_RANDOM_STATE)
    rows = list(cat.itertuples(index=False))
    ctx_freq = Counter(c for row in rows for c in dict.fromkeys(o[1] for o in slot_options(row, "train")["context"]))
    train = [p for row in rows for p in train_queries_for(row, rng, ctx_freq)]
    train_keys = {(p["query"], p["track_id"]) for p in train}

    erng = np.random.default_rng(EVAL_RANDOM_STATE)
    eligible = [row for row in cat.itertuples(index=False)
                if any(fillable(t, slot_options(row, "eval")) for _, t in EVAL_TEMPLATES.values())]
    idx = erng.permutation(len(eligible))
    evals = []
    for i in idx:
        if len(evals) >= EVAL_N_TRACKS:
            break
        rec = eval_query_for(eligible[i], erng, train_keys)
        if rec:
            evals.append(rec)
    return train, evals, len(eligible)


def eval_b_words() -> list[str]:
    """Canonical context words with a playlist equivalent (synonym keys are not canonical)."""
    synonyms = {s for syns in CONTEXT_SYNONYMS.values() for s in syns}
    return [w for w, pw in CONTEXT_TO_PLAYLIST_WORDS.items() if pw and w not in synonyms]


def generate_eval_b(train_queries: set[str], per_word: int = EVAL_B_PER_WORD) -> list[dict]:
    """Up to `per_word` distinct context-only queries per canonical word. Surfaces (canonical +
    synonyms) are interleaved so each gets a fair share. playlist_words are the canonical word's
    (synonyms map to canonical)."""
    rng = np.random.default_rng(EVAL_B_RANDOM_STATE)
    out = []
    for word in eval_b_words():
        surfaces = [word, *CONTEXT_SYNONYMS.get(word, [])]
        per_surface = []
        for surf in surfaces:
            tids = list(EVAL_B_TEMPLATES)
            rng.shuffle(tids)
            per_surface.append([(tid, surf) for tid in tids])
        seen, picked = set(), []
        for combos in zip(*per_surface):          # round-robin across surfaces
            for tid, surf in combos:
                q = EVAL_B_TEMPLATES[tid].format(context=CONTEXT_PHRASES[surf])
                if len(picked) < per_word and q not in seen and q not in train_queries:
                    seen.add(q)
                    picked.append({"query": q, "context_word": word,
                                   "playlist_words": CONTEXT_TO_PLAYLIST_WORDS[word],
                                   "surface": surf, "template_id": tid})
        out += picked
    return out


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def context_vocab(cat: pd.DataFrame) -> list[str]:
    words = Counter(w for ctx in cat.context for w in ctx)
    return sorted(set(words) | CATEGORY_CONTEXTS)


def main():
    cat = pd.read_parquet(CATALOG)
    train, evals, n_eligible = generate(cat)
    write_jsonl(TRAIN_PAIRS, train)
    write_jsonl(EVAL_PAIRS, evals)
    eval_b = generate_eval_b({p["query"] for p in train})
    write_jsonl(EVAL_CONTEXT_QUERIES, eval_b)

    tr, ev = pd.DataFrame(train), pd.DataFrame(evals)
    print(f"train pairs: {len(tr)}  per type: {tr.type.value_counts().to_dict()}")
    print(f"eval pairs:  {len(ev)} (eligible tracks {n_eligible})  per type: {ev.type.value_counts().to_dict()}")
    print(f"train templates used: {tr.template_id.nunique()}/{len(TRAIN_TEMPLATES)}  "
          f"eval templates used: {ev.template_id.nunique()}/{len(EVAL_TEMPLATES)}")

    print("\ntraining pairs per context word (canonical):")
    tr_ctx = Counter(w for ws in tr.wanted_tags for w in ws)
    ev_ctx = Counter(w for ws in ev.wanted_tags for w in ws)
    for w in sorted(context_vocab(cat), key=lambda w: tr_ctx[w]):
        flag = "  <-- UNDER 300" if tr_ctx[w] < MIN_PAIRS_PER_CONTEXT else ""
        print(f"  {w:<12} train {tr_ctx[w]:>5}  eval {ev_ctx[w]:>4}{flag}")

    print("\n10 random train queries:")
    for r in tr.sample(10, random_state=0).itertuples():
        print(f"  [{r.template_id}] {r.query!r} -> {r.wanted_tags}")
    print("\n10 random eval queries:")
    for r in ev.sample(10, random_state=0).itertuples():
        print(f"  [{r.template_id}] {r.query!r} -> {r.wanted_tags}")

    eb = pd.DataFrame(eval_b)
    print(f"\nEval B: {len(eb)} queries over {eb.context_word.nunique()} context words")
    for w, g in eb.groupby("context_word"):
        print(f"  {w:<10} {len(g):>3}  surfaces {g.surface.value_counts().to_dict()}  "
              f"playlist_words {g.playlist_words.iloc[0]}")
    print("\n10 random Eval B queries:")
    for r in eb.sample(10, random_state=0).itertuples():
        print(f"  [{r.template_id}] {r.query!r} -> {r.context_word}")

    zero = set(cat.track_id) - set(tr.track_id)
    print(f"\ntracks with 0 train queries: {len(zero)}")
    per_track = tr.groupby("track_id").size()
    print(f"queries per track: {per_track.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
