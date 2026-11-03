"""Step 5b: human-signal training pairs from TUNE-half context labels (build side only).

For each tune-half label (track, playlist word), the playlist word is mapped to canonical context words by
inverting the canonical (non-synonym) entries of CONTEXT_TO_PLAYLIST_WORDS (Eval B's relevance mapping), with two
approved overrides: party -> party only (not dance), focus -> study. late night, sad, drive and night drive stay
unmapped.
Each (track, canonical word) unit gets 2 pairs: one TRAIN context template, and one TRAIN combo template with the
track's own genre (from `genres` only, humanized) or mood tag (calm / energetic / happy / melancholic), chosen
with seed 18. Tracks with neither get a second context template instead. The context surface is the canonical
word or one of its CONTEXT_SYNONYMS. Playlist names are never used.
The eval half and the Eval B candidate pool are untouched.

Writes data/proc/human_pairs.jsonl: {query, track_id, template_id, type, wanted_tags, context_word,
playlist_words, source="tune_context"}.
"""
from __future__ import annotations

import json
import sys
import zlib
from collections import Counter

import numpy as np
import pandas as pd

from src.config import CONTEXT_SYNONYMS
from src.gen_queries import (CONTEXT_PHRASES, MOOD_TAGS, TRAIN_TEMPLATES, eval_b_words, humanize_genre,
                              read_jsonl, write_jsonl)
from src.paraphrase import GOLD, normalize
from src.paths import CATALOG, CONTEXT_LABELS, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, PROC, TRAIN_PAIRS
from src.prep import CONTEXT_TO_PLAYLIST_WORDS

HUMAN_PAIRS = PROC / "human_pairs.jsonl"
PAIRS_PER_TRACK_CONTEXT = 2
HUMAN_PAIRS_SEED = 18
SOURCE = "tune_context"
CONTEXT_TEMPLATE_IDS = sorted(t for t, (typ, _) in TRAIN_TEMPLATES.items() if typ == "context")
COMBO_GENRE_TEMPLATE, COMBO_MOOD_TEMPLATE = "tr_combo_3", "tr_combo_2"   # "{genre} for {context}", "{mood} songs for {context}"
PLAYLIST_WORD_OVERRIDES = {"party": ["party"], "focus": ["study"]}     # approved Step 5b changes
EPOCH_CAP_PER_STRING = 150                                               # Step 15 per-epoch sampling (report only)


def playlist_word_map() -> dict[str, list[str]]:
    """playlist word -> canonical context words (Eval B's canonical words only)."""
    out: dict[str, list[str]] = {}
    for w in eval_b_words():
        for pw in CONTEXT_TO_PLAYLIST_WORDS[w]:
            out.setdefault(pw, []).append(w)
    out.update(PLAYLIST_WORD_OVERRIDES)
    return {k: sorted(v) for k, v in out.items()}


def track_contexts(labels: pd.DataFrame) -> pd.DataFrame:
    """Tune-half rows -> one row per (track_id, canonical context_word) with the playlist words behind it."""
    pmap = playlist_word_map()
    tune = labels[labels.label_split == "tune"]
    rows = [(t, c, pw) for t, pw in zip(tune.track_id, tune.context_word) for c in pmap.get(pw, [])]
    df = pd.DataFrame(rows, columns=["track_id", "context_word", "playlist_word"])
    return (df.groupby(["track_id", "context_word"]).playlist_word.agg(lambda s: sorted(set(s)))
              .rename("playlist_words").reset_index())


def combo_options(genres, tags) -> list[tuple[str, str, str]]:
    """(slot, surface, wanted tag) options from the track's own genres (humanized) and mood tags."""
    return ([("genre", humanize_genre(g), g) for g in genres] +
            [("mood", MOOD_TAGS[t], t) for t in tags if t in MOOD_TAGS])


def pairs_for(track_id: str, word: str, genres=(), tags=()) -> list[dict]:
    """2 pairs per (track, context word): one context template + one combo (genre or mood), else a second
    context template. rng seeded by (HUMAN_PAIRS_SEED, crc32(track|word))."""
    rng = np.random.default_rng([HUMAN_PAIRS_SEED, zlib.crc32(f"{track_id}|{word}".encode())])
    surfaces = [word, *CONTEXT_SYNONYMS.get(word, [])]
    base = {"track_id": track_id, "context_word": word, "source": SOURCE}
    ctx = [(tid, s) for tid in CONTEXT_TEMPLATE_IDS for s in surfaces]
    order = rng.permutation(len(ctx))
    tid, surface = ctx[order[0]]
    out = [{"query": TRAIN_TEMPLATES[tid][1].format(context=CONTEXT_PHRASES[surface]), "template_id": tid,
            "type": "context", "wanted_tags": [word], "surface": surface, **base}]
    opts = combo_options(genres, tags)
    if opts:
        slot, val, tag = opts[rng.integers(len(opts))]
        surface = surfaces[rng.integers(len(surfaces))]
        tid = COMBO_GENRE_TEMPLATE if slot == "genre" else COMBO_MOOD_TEMPLATE
        out.append({"query": TRAIN_TEMPLATES[tid][1].format(**{slot: val, "context": CONTEXT_PHRASES[surface]}),
                    "template_id": tid, "type": "combo", "wanted_tags": [tag, word], "surface": surface, **base})
    elif len(ctx) > 1:
        tid, surface = ctx[order[1]]
        out.append({"query": TRAIN_TEMPLATES[tid][1].format(context=CONTEXT_PHRASES[surface]), "template_id": tid,
                    "type": "context", "wanted_tags": [word], "surface": surface, **base})
    return out


def eval_queries_normalized() -> set[str]:
    return {normalize(r["query"]) for p in (EVAL_PAIRS, EVAL_CONTEXT_QUERIES, GOLD) for r in read_jsonl(p)}


def build() -> tuple[list[dict], dict]:
    cat = pd.read_parquet(CATALOG)
    labels = pd.read_parquet(CONTEXT_LABELS)
    tc = track_contexts(labels)
    in_cat = tc.track_id.isin(set(cat.track_id))
    tc = tc[in_cat]
    meta = cat.set_index("track_id")[["genres", "tags"]]
    evq = eval_queries_normalized()
    pairs, n_eval_match = [], 0
    for r in tc.itertuples(index=False):
        for p in pairs_for(r.track_id, r.context_word, meta.at[r.track_id, "genres"], meta.at[r.track_id, "tags"]):
            if normalize(p["query"]) in evq:
                n_eval_match += 1
                continue
            pairs.append({**p, "playlist_words": r.playlist_words})
    tune_words = labels[labels.label_split == "tune"].context_word
    unmapped = tune_words[~tune_words.isin(playlist_word_map())].value_counts().to_dict()
    write_jsonl(HUMAN_PAIRS, pairs)
    return pairs, {"track_contexts": len(tc), "dropped_not_in_catalog": int((~in_cat).sum()),
                   "dropped_eval_match": n_eval_match, "unmapped_playlist_label_rows": unmapped,
                   "tune_label_tracks": int(labels[labels.label_split == "tune"].track_id.nunique())}


def report(pairs: list[dict], info: dict) -> None:
    cat = pd.read_parquet(CATALOG).set_index("track_id")
    df = pd.DataFrame(pairs)
    print(f"pairs: {len(df):,} | source counts: {df.source.value_counts().to_dict()}")
    print(f"tracks covered: {df.track_id.nunique():,} of {info['tune_label_tracks']:,} tune-labeled tracks "
          f"| (track, context) units: {info['track_contexts']:,}")
    print(f"dropped: not in catalog {info['dropped_not_in_catalog']} | eval-query match {info['dropped_eval_match']}")
    print(f"tune label rows with no canonical mapping (playlist word: rows): {info['unmapped_playlist_label_rows']}")
    print("playlist word -> canonical:", playlist_word_map())
    per = df.groupby("context_word").agg(pairs=("query", "size"), tracks=("track_id", "nunique"))
    per["catalog_tracks_with_rule_context"] = [int(cat.context.map(lambda c, w=w: w in c).sum()) for w in per.index]
    print("\nper context word:\n" + per.sort_values("pairs", ascending=False).to_string())
    print("\ntemplate use:", Counter(df.template_id).most_common(), "| by type:", df.type.value_counts().to_dict())
    per_q = df["query"].value_counts()
    print(f"distinct query strings: {len(per_q):,} | pairs surviving one epoch at <= {EPOCH_CAP_PER_STRING}/string "
          f"(human pairs alone): {int(per_q.clip(upper=EPOCH_CAP_PER_STRING).sum()):,} of {len(df):,}")
    tr = pd.DataFrame(read_jsonl(TRAIN_PAIRS))
    allq = pd.concat([tr["query"], df["query"]]).value_counts()
    share = df["query"].map(allq.clip(upper=EPOCH_CAP_PER_STRING) / allq)
    print(f"  with train_pairs.jsonl in the same pool (proportional share of each capped string): "
          f"{share.sum():,.0f} human pairs / epoch")
    print("\n10 random combo pairs (random_state=0):")
    for r in df[df.type == "combo"].sample(10, random_state=0).itertuples():
        print(f"  {r.query!r} -> {cat.loc[r.track_id, 'track_name']} / {cat.loc[r.track_id, 'artists']} "
              f"[{r.wanted_tags}; playlist words {r.playlist_words}]")


if __name__ == "__main__":
    sys.exit(report(*build()))
