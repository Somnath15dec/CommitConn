"""Step 6 / C: LLM paraphrase queries via local Ollama (train-only data).

Input to the LLM: content_desc ONLY (never title, artist, playlist names or MuSe scores), shown through
prompt_desc(): humanized genre names (new-age -> new age, r-n-b -> r&b), no "neutral mood" / "moderate energy",
and "Spotify category: X" turned into plain words (mood categories join Sound, romance -> romantic; party/study/
sleep join Good for). The stored desc is unchanged.
Subset: 2,000 tracks (seed 2); tracks with a sleep / study / rainy day context word come first.
Batches of 10 descs -> 3 queries per track, one per style (short / situation / casual), JSON-schema output.
A failed batch is retried once, then skipped (logged; a later run retries it). Kept pairs are appended to
data/proc/paraphrase_pairs.jsonl and every batch to data/proc/paraphrase_batches.jsonl, so runs are resumable.

    python -m src.paraphrase --max-batches 5     # test run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.analyze_readiness import content_desc
from src.config import canonical
from src.gen_queries import CATEGORY_MOODS, contains_word, humanize_genre
from src.paths import CATALOG, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, PROC, ROOT

PARAPHRASE_PAIRS = PROC / "paraphrase_pairs.jsonl"
PARAPHRASE_BATCHES = PROC / "paraphrase_batches.jsonl"
GOLD = ROOT / "eval" / "gold_queries.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5-6k"
OPTIONS = {"temperature": 0.8, "num_ctx": 6144}
TIMEOUT_S = 300
SUBSET_N, SUBSET_SEED = 2000, 2
PRIORITY_CONTEXTS = ("sleep", "study", "rainy day")
BATCH_SIZE = 10
STYLES = ("short", "situation", "casual")
# word-count bounds per style: the global 3-20 filter, except SHORT which is 2-5 by design
LENGTH_BOUNDS = {"short": (2, 5), "situation": (3, 20), "casual": (3, 20)}
NAME_WORD_MIN_LEN = 4                       # title/artist words longer than 3 chars are banned
BANNED_PHRASES = ("neutral mood", "moderate energy", "spoken word", "category")
BANNED_FIRST_WORDS = ("play", "find")
INSTRUMENTS = ("piano", "guitar", "strings", "violin", "synth", "drums", "sax", "saxophone", "cello")
HIDDEN_SOUND_WORDS = ("neutral mood", "moderate energy")
CONTEXT_WORDS = ("study", "focus", "late night", "relaxing", "morning", "coffee", "sleep", "wind down", "party",
                 "dance", "running", "gym", "workout", "rainy day", "reflective", "feel-good", "intense",
                 "cathartic")   # canonical context words + synonyms (synonyms count as canonical)
RULE_REASONS = ("rule_banned_term", "rule_first_word", "rule_multi_context", "rule_instrument")

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, **{s: {"type": "string"} for s in STYLES}},
        "required": ["id", *STYLES]}}},
    "required": ["items"],
}
SYSTEM = ("You write what real people type or say to a music app when they want songs. "
          "Answer with JSON only.")
# Few-shot examples (written for this step; tests check they don't resemble gold / Eval A / Eval B queries)
FEW_SHOT = [
    {"desc": "Genre: ambient. Sound: slow, calm, melancholic, instrumental. Good for: study, sleep, relaxing.",
     "short": "melancholic ambient",
     "situation": "reading in bed until my eyes get heavy",
     "casual": "anything mellow i can do homework to?"},
    {"desc": "Genre: punk, rock. Sound: fast, energetic, happy. Good for: running, gym, feel-good.",
     "short": "upbeat punk rock",
     "situation": "pushing through the last mile of an evening jog",
     "casual": "need loud fun rock that makes me grin"},
]
PROMPT = """Each numbered line below describes one track (genre, sound, what it is good for).
For EACH track write exactly 3 requests a person might send to a music app, one in each style:
- "short": 2 to 5 words, like typing into a search box.
- "situation": describe what the person is doing or going through, 3 to 20 words. Do NOT name a genre.
- "casual": chatty, 3 to 20 words, lowercase is fine, may mention the genre or a mood.
Rules:
- Each of the 3 requests uses a DIFFERENT one of the track's moods or "good for" uses. Never put more than one
  "good for" use in a single request.
- Never write "neutral mood", "moderate energy" or "spoken word". Never start a request with "play" or "find".
- Write genre names exactly as shown in the description (no hyphens). Never write the word "category".
- Never mention a genre, instrument or era that is not in the description. If the description has no genre,
  do not mention one.
- Never invent song titles or artist names.

Example input:
{ex_lines}
Example output:
{ex_output}

Return {{"items": [{{"id": <line number>, "short": ..., "situation": ..., "casual": ...}}, ...]}} with one item
per line.

{lines}"""


# ---------------------------------------------------------------- text helpers
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9&]+", " ", t).split())


def name_words(title: str, artists: str) -> set[str]:
    return {w for w in normalize(f"{title} {artists.replace(';', ' ')}").split() if len(w) >= NAME_WORD_MIN_LEN}


def humanize_desc(desc: str) -> str:
    """content_desc with genre ids in the 'Genre:' sentence humanized (new-age -> new age)."""
    def sub(m):
        return "Genre: " + ", ".join(humanize_genre(g) for g in m.group(1).split(", ")) + "."
    return re.sub(r"^Genre: ([^.]*)\.", sub, desc)


def _sentences(desc: str) -> dict[str, list[str]]:
    out = {}
    for m in re.finditer(r"(Genre|Spotify category|Sound|Good for): ([^.]*)\.", desc):
        out[m.group(1)] = [w for w in m.group(2).split(", ") if w]
    return out


def prompt_desc(desc: str) -> str:
    """The LLM's view of a content_desc (stored desc unchanged): humanized genres, no neutral mood / moderate
    energy, Spotify category as plain words (moods -> Sound, party/study/sleep -> Good for)."""
    s = _sentences(desc)
    sound = [w for w in s.get("Sound", []) if w not in HIDDEN_SOUND_WORDS]
    good = [w for w in s.get("Good for", []) if w != "general listening"]
    for c in s.get("Spotify category", []):
        if c in CATEGORY_MOODS:
            sound.append(CATEGORY_MOODS[c])
        elif c not in good:
            good.append(c)
    parts = []
    if s.get("Genre"):
        parts.append("Genre: " + ", ".join(humanize_genre(g) for g in s["Genre"]) + ".")
    if sound:
        parts.append("Sound: " + ", ".join(dict.fromkeys(sound)) + ".")
    parts.append("Good for: " + (", ".join(good) or "general listening") + ".")
    return " ".join(parts)


def eval_queries_raw() -> list[str]:
    qs = []
    for path in (EVAL_PAIRS, EVAL_CONTEXT_QUERIES, GOLD):
        with path.open(encoding="utf-8") as f:
            qs += [json.loads(line)["query"] for line in f]
    return qs


def eval_query_set() -> set[str]:
    return {normalize(q) for q in eval_queries_raw()}


def context_words(q: str) -> set[str]:
    """Distinct canonical context words in a query (synonyms map to canonical)."""
    return {canonical(w) for w in CONTEXT_WORDS if contains_word(q, w)}


def rule_violation(q: str, desc: str, hyphen_genres: set[str]) -> str | None:
    """Fifth filter: the prompt rule a query breaks, else None."""
    low = q.lower()
    if any(contains_word(low, t) for t in (*BANNED_PHRASES, *hyphen_genres)):
        return "rule_banned_term"
    words = normalize(q).split()
    if words and words[0] in BANNED_FIRST_WORDS:
        return "rule_first_word"
    if len(context_words(q)) > 1:
        return "rule_multi_context"
    if any(contains_word(low, i) and not contains_word(desc, i) for i in INSTRUMENTS):
        return "rule_instrument"
    return None


def filter_queries(queries: list[tuple[str, str]], banned_names: set[str], eval_queries: set[str],
                   desc: str = "", hyphen_genres: set[str] = frozenset()) -> tuple[list[tuple[str, str]], dict]:
    """queries = [(style, text)]. Filters in order: name word, length (per style), eval match, within-track
    duplicate, prompt rules (rule_violation; desc = the track's stored desc). Returns (kept, rejected counts)."""
    counts = {"name_word": 0, "length": 0, "eval_match": 0, "duplicate": 0, **dict.fromkeys(RULE_REASONS, 0)}
    kept, seen = [], set()
    for style, q in queries:
        q = " ".join(str(q).split())
        n = normalize(q)
        lo, hi = LENGTH_BOUNDS.get(style, (3, 20))
        if set(n.split()) & banned_names:
            counts["name_word"] += 1
        elif not lo <= len(q.split()) <= hi:
            counts["length"] += 1
        elif n in eval_queries:
            counts["eval_match"] += 1
        elif n in seen:
            counts["duplicate"] += 1
        elif why := rule_violation(q, desc, hyphen_genres):
            counts[why] += 1
        else:
            seen.add(n)
            kept.append((style, q))
    return kept, counts


def banned_terms(q: str, hyphen_genres: set[str]) -> list[str]:
    """Banned phrases, banned first words and raw hyphenated genre ids found in a query (report metric)."""
    low = q.lower()
    hits = [p for p in BANNED_PHRASES if p in low]
    first = normalize(q).split()[:1]
    hits += [f"first word '{w}'" for w in first if w in BANNED_FIRST_WORDS]
    hits += [g for g in hyphen_genres if re.search(rf"(?<![\w-]){re.escape(g)}(?![\w-])", low)]
    return hits


def hyphenated_genre_ids(cat: pd.DataFrame) -> set[str]:
    return {g for gs in cat.genres for g in gs if "-" in g and humanize_genre(g) != g}


# ---------------------------------------------------------------- subset / state
def select_subset(cat: pd.DataFrame, n: int = SUBSET_N, seed: int = SUBSET_SEED) -> pd.DataFrame:
    """Priority tracks (any PRIORITY_CONTEXTS word) first, then random others; each part shuffled with `seed`."""
    pri = cat.context.map(lambda c: bool(set(c) & set(PRIORITY_CONTEXTS)))
    first = cat[pri].sample(frac=1, random_state=seed)
    rest = cat[~pri].sample(frac=1, random_state=seed)
    return pd.concat([first, rest]).head(n).reset_index(drop=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def done_track_ids(batches_path: Path) -> set[str]:
    return {t for b in read_jsonl(batches_path) if b["status"] == "ok" for t in b["track_ids"]}


# ---------------------------------------------------------------- LLM
def build_prompt(descs: list[str]) -> str:
    ex_lines = "\n".join(f"{i}: {e['desc']}" for i, e in enumerate(FEW_SHOT, 1))
    ex_output = json.dumps({"items": [{"id": i, **{s: e[s] for s in STYLES}}
                                      for i, e in enumerate(FEW_SHOT, 1)]})
    lines = "\n".join(f"{i}: {prompt_desc(d)}" for i, d in enumerate(descs, 1))
    return PROMPT.format(ex_lines=ex_lines, ex_output=ex_output, lines=lines)


def parse_response(content: str, n_items: int) -> dict[int, list[tuple[str, str]]]:
    """{id: [(style, query)]} for ids 1..n_items; raises ValueError unless every id has all 3 styles."""
    data = json.loads(content)
    out = {}
    for it in data["items"]:
        if not all(isinstance(it.get(s), str) and it[s].strip() for s in STYLES):
            raise ValueError(f"missing style for id {it.get('id')}")
        out[int(it["id"])] = [(s, it[s]) for s in STYLES]
    if set(out) != set(range(1, n_items + 1)):
        raise ValueError(f"ids {sorted(out)} != 1..{n_items}")
    return out


def call_llm(prompt: str) -> str:
    r = requests.post(OLLAMA_URL, timeout=TIMEOUT_S, json={
        "model": MODEL, "think": False, "stream": False, "format": SCHEMA, "options": OPTIONS,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


def run_batch(rows: pd.DataFrame, eval_queries: set[str], llm=call_llm,
              hyphen_genres: set[str] = frozenset()) -> tuple[list[dict], dict]:
    descs = rows.content_desc.tolist()
    prompt, err, parsed = build_prompt(descs), None, None
    t0 = time.perf_counter()
    for _ in range(2):                                   # one retry
        try:
            parsed = parse_response(llm(prompt), len(descs))
            break
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, requests.RequestException) as e:
            err = f"{type(e).__name__}: {e}"
    secs = time.perf_counter() - t0
    counts = {"generated": 0, "name_word": 0, "length": 0, "eval_match": 0, "duplicate": 0,
              **dict.fromkeys(RULE_REASONS, 0), "kept": 0}
    if parsed is None:
        return [], {"status": "failed", "error": err, "seconds": secs, "counts": counts}
    pairs = []
    for i, row in enumerate(rows.itertuples(index=False), 1):
        kept, rej = filter_queries(parsed[i], name_words(row.track_name, row.artists), eval_queries,
                                   desc=row.desc, hyphen_genres=hyphen_genres)
        counts["generated"] += len(parsed[i])
        for k, v in rej.items():
            counts[k] += v
        counts["kept"] += len(kept)
        pairs += [{"query": q, "track_id": row.track_id, "type": "paraphrase", "style": s, "model": MODEL}
                  for s, q in kept]
    return pairs, {"status": "ok", "seconds": secs, "counts": counts,
                   "raw": {row.track_id: dict(parsed[i]) for i, row in enumerate(rows.itertuples(index=False), 1)}}


def run(max_batches: int | None = None, pairs_path: Path = PARAPHRASE_PAIRS,
        batches_path: Path = PARAPHRASE_BATCHES, llm=call_llm) -> list[dict]:
    cat = pd.read_parquet(CATALOG)
    cat["content_desc"] = cat.apply(content_desc, axis=1)
    sub = select_subset(cat)
    done = done_track_ids(batches_path)
    todo = sub[~sub.track_id.isin(done)].reset_index(drop=True)
    eval_queries = eval_query_set()
    hyphen_genres = hyphenated_genre_ids(cat)
    n_prev = len(read_jsonl(batches_path))
    print(f"subset {len(sub)} tracks | done {len(sub) - len(todo)} | to do {len(todo)} "
          f"({int(np.ceil(len(todo) / BATCH_SIZE))} batches) | model {MODEL} {OPTIONS}", flush=True)
    logs = []
    for k, s in enumerate(range(0, len(todo), BATCH_SIZE)):
        if max_batches is not None and k >= max_batches:
            break
        rows = todo.iloc[s:s + BATCH_SIZE]
        pairs, log = run_batch(rows, eval_queries, llm, hyphen_genres)
        log = {"batch": n_prev + k, "track_ids": rows.track_id.tolist(), **log}
        append_jsonl(pairs_path, pairs)
        append_jsonl(batches_path, [log])
        logs.append(log)
        print(f"batch {log['batch']}: {log['status']} {log['seconds']:.1f}s {log['counts']}"
              + (f" error={log.get('error')}" if log["status"] != "ok" else ""), flush=True)
    return logs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batches", type=int, default=None)
    run(ap.parse_args().max_batches)
    sys.exit(0)
