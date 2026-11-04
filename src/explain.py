"""Step J: one-sentence explanation for a recommended track, grounded in that track's own tags / context.

LLM: a small local instruct model via Ollama (EXPLAIN_MODEL), JSON-schema output, think off.
Input: the user's query + each track's cleaned content_desc (paraphrase.prompt_desc: no "neutral mood" /
"moderate energy"), tags, context, cluster name, and the facts that match the request. Batch mode: one call explains
all results (JSON list); grounding and the template fallback apply per item. Timeout: 3 s for one track, plus
BATCH_EXTRA_S per extra track.
Grounding: every mood / sound / context word the explanation uses (LEXICON, synonyms included) must be supported
by the track's tags or context. If the explanation is ungrounded, longer than 25 words, malformed, slow (timeout)
or Ollama is unreachable, the template fallback is used:
    "{tags} — matches '{query words that hit tags/context}'"
"""
from __future__ import annotations

import json
import re

import requests

from src.config import SYNONYM_TO_CANONICAL

OLLAMA_URL = "http://localhost:11434/api/chat"
EXPLAIN_MODEL = "qwen3:4b-instruct"
TIMEOUT_S = 3.0
BATCH_EXTRA_S = 0.35
MAX_WORDS = 25
TOKENS_PER_ITEM = 48
OPTIONS = {"temperature": 0.3, "num_ctx": 4096}
KEEP_ALIVE = "30m"
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "explanation": {"type": "string"}},
    "required": ["id", "explanation"]}}}, "required": ["items"]}
HIDDEN = {"neutral mood", "moderate energy"}           # not shown in the fallback (they say nothing to a user)

# word in an explanation -> tags / context words that justify it (any one suffices)
LEXICON: dict[str, set[str]] = {
    # sound / tempo / energy
    "calm": {"calm"}, "mellow": {"calm"}, "soft": {"calm", "acoustic"}, "gentle": {"calm"}, "quiet": {"calm"},
    "peaceful": {"calm", "relaxing"}, "soothing": {"calm", "relaxing", "sleep"}, "chill": {"calm", "relaxing"},
    "energetic": {"energetic"}, "high-energy": {"energetic"}, "intense": {"energetic", "intense"},
    "upbeat": {"energetic", "happy"}, "driving": {"energetic", "fast"}, "pumping": {"energetic"},
    "slow": {"slow"}, "fast": {"fast"}, "uptempo": {"fast"}, "up-tempo": {"fast"},
    "instrumental": {"instrumental"}, "acoustic": {"acoustic"}, "danceable": {"danceable", "dance"},
    "rap": {"spoken word or rap"}, "spoken": {"spoken word or rap"},
    # mood
    "happy": {"happy"}, "cheerful": {"happy"}, "joyful": {"happy"}, "bright": {"happy"}, "feel-good": {"happy", "feel-good"},
    "melancholic": {"melancholic"}, "melancholy": {"melancholic"}, "sad": {"melancholic"}, "wistful": {"melancholic"},
    "moody": {"melancholic"}, "somber": {"melancholic"}, "dark": {"melancholic"}, "cathartic": {"cathartic"},
    "romantic": set(),
    # context (canonical + synonyms + common forms)
    "study": {"study"}, "studying": {"study"}, "focus": {"study"}, "focused": {"study"}, "concentration": {"study"},
    "late-night": {"study"}, "sleep": {"sleep"}, "sleeping": {"sleep"}, "asleep": {"sleep"}, "bedtime": {"sleep"},
    "workout": {"workout", "gym"}, "gym": {"gym", "workout"}, "training": {"workout", "gym"},
    "running": {"running"}, "run": {"running"}, "jogging": {"running"},
    "party": {"party"}, "dance": {"dance", "danceable"}, "dancing": {"dance", "danceable"},
    "morning": {"morning"}, "coffee": {"coffee"}, "relaxing": {"relaxing"}, "relax": {"relaxing"},
    "unwind": {"relaxing", "sleep"}, "rainy": {"rainy day"}, "rain": {"rainy day"}, "reflective": {"rainy day"},
}
PHRASES = {"wind down": {"sleep"}, "winding down": {"sleep"}, "late night": {"study"}, "rainy day": {"rainy day"},
           "moderate energy": {"moderate energy"}, "neutral mood": {"neutral mood"}, "spoken word": {"spoken word or rap"}}

SYSTEM = "You explain music recommendations in short, friendly sentences. Answer with JSON only."
PROMPT = """A listener asked: "{query}"
For EACH numbered track below, write ONE sentence (at most 20 words) explaining why it fits the request.
Rules:
- Mention at most 2 facts, preferring the facts listed under "matches the request".
- Never mention a "good for" use the listener did not ask for, unless nothing matches the request.
- Use ONLY sound, mood and use words from that track's facts; never invent others. Do not name tracks or artists.
Return {{"items": [{{"id": <track number>, "explanation": "..."}}, ...]}} with one item per track.

{tracks}"""


# ---------------------------------------------------------------- grounding
def track_terms(track: dict) -> set[str]:
    """Canonical tags + context words of the track (context synonyms map to canonical)."""
    return set(track["tags"]) | {SYNONYM_TO_CANONICAL.get(c, c) for c in track["context"]}


def mentioned(text: str) -> dict[str, set[str]]:
    """Lexicon words / phrases found in text -> the terms that would justify each."""
    low = text.lower()
    out = {p: need for p, need in PHRASES.items() if re.search(rf"\b{re.escape(p)}\b", low)}
    for p in PHRASES:
        low = re.sub(rf"\b{re.escape(p)}\b", " ", low)
    for w in re.findall(r"[a-z]+(?:-[a-z]+)*", low):
        if w in LEXICON:
            out[w] = LEXICON[w]
    return out


def ungrounded(text: str, track: dict) -> list[str]:
    terms = track_terms(track)
    return sorted(w for w, need in mentioned(text).items() if not (need & terms))


def query_hits(query: str, track: dict) -> list[str]:
    terms = track_terms(track)
    return [w for w, need in mentioned(query).items() if need & terms]


def fallback(query: str, track: dict) -> str:
    tags = ", ".join(t for t in track["tags"] if t not in HIDDEN) or ", ".join(track["tags"])
    hits = query_hits(query, track)
    return f"{tags} — matches '{' '.join(hits)}'" if hits else f"{tags} — closest match for '{query}'"


# ---------------------------------------------------------------- LLM
def matching_facts(query: str, track: dict) -> list[str]:
    """The track's tags / context words that the query's words point to."""
    terms = track_terms(track)
    return sorted({t for w, need in mentioned(query).items() for t in need & terms} - HIDDEN)


def facts(query: str, track: dict) -> str:
    from src.paraphrase import prompt_desc
    m = matching_facts(query, track)
    return (f"{prompt_desc(track['content_desc'])} Cluster: {track.get('cluster_name', '')}. "
            f"Matches the request: {', '.join(m) if m else 'nothing directly'}.")


def timeout_for(n: int, timeout: float = TIMEOUT_S) -> float:
    return timeout + BATCH_EXTRA_S * max(0, n - 1)


def ask_batch(query: str, tracks: list[dict], timeout: float, url: str = OLLAMA_URL, post=requests.post) -> dict[int, str]:
    body = "\n".join(f"{i}: {facts(query, t)}" for i, t in enumerate(tracks, 1))
    r = post(url, timeout=timeout, json={
        "model": EXPLAIN_MODEL, "think": False, "stream": False, "format": SCHEMA, "keep_alive": KEEP_ALIVE,
        "options": {**OPTIONS, "num_predict": TOKENS_PER_ITEM * len(tracks) + 16},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT.format(query=query, tracks=body)}]})
    r.raise_for_status()
    items = json.loads(r.json()["message"]["content"])["items"]
    return {int(it["id"]): " ".join(str(it["explanation"]).split()) for it in items}


def check(text: str | None, query: str, track: dict) -> dict:
    """Per-item grounding / length check -> result dict (template fallback on failure)."""
    if not text:
        return {"text": fallback(query, track), "source": "fallback", "reason": "missing item"}
    if len(text.split()) > MAX_WORDS:
        return {"text": fallback(query, track), "source": "fallback", "reason": "too long", "llm_text": text}
    bad = ungrounded(text, track)
    if bad:
        return {"text": fallback(query, track), "source": "fallback", "reason": f"ungrounded: {', '.join(bad)}",
                "llm_text": text}
    return {"text": text, "source": "llm", "reason": "ok"}


def explain_batch(query: str, tracks: list[dict], timeout: float | None = None, url: str = OLLAMA_URL,
                  post=requests.post) -> list[dict]:
    """One LLM call for all tracks; per-item grounding and fallback. Never raises."""
    if not tracks:
        return []
    t = timeout if timeout is not None else timeout_for(len(tracks))
    try:
        got = ask_batch(query, tracks, t, url, post)
    except requests.Timeout:
        return [{"text": fallback(query, tr), "source": "fallback", "reason": "timeout"} for tr in tracks]
    except requests.RequestException as e:
        why = f"ollama unavailable ({type(e).__name__})"
        return [{"text": fallback(query, tr), "source": "fallback", "reason": why} for tr in tracks]
    except (ValueError, KeyError, TypeError) as e:
        why = f"bad response ({type(e).__name__})"
        return [{"text": fallback(query, tr), "source": "fallback", "reason": why} for tr in tracks]
    return [check(got.get(i), query, tr) for i, tr in enumerate(tracks, 1)]


def explain(query: str, track: dict, timeout: float = TIMEOUT_S, url: str = OLLAMA_URL, post=requests.post) -> dict:
    """Single-track explanation (a batch of one). {"text", "source": "llm" | "fallback", "reason"}; never raises."""
    return explain_batch(query, [track], timeout, url, post)[0]


# ---------------------------------------------------------------- demo (python -m src.explain)
def _gpu_used_mib() -> int:
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def demo(gold_nums=(1, 4, 6, 16, 26), top=2) -> dict:
    """SID model (final, beam 50) top-`top` for the given gold queries, explained; latency per explanation and
    peak GPU memory (nvidia-smi, all processes) with the SID model and the explainer loaded together."""
    import threading
    import time

    import pandas as pd

    from src.analyze_readiness import content_desc
    from src.cluster_names import CLUSTER_NAMES
    from src.decode import SidDecoder
    from src.evaluate_final import BEAMS, load_sid_model
    from src.paths import CATALOG, ROOT
    from src.train_data import SIDS

    base_mib = _gpu_used_mib()
    peak, stop = [base_mib], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], _gpu_used_mib())
            time.sleep(0.2)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    cat = pd.read_parquet(CATALOG)
    cat["content_desc"] = cat.apply(content_desc, axis=1)
    meta = cat.set_index("track_id")
    sids = pd.read_parquet(SIDS)
    level1 = dict(zip(sids.track_id, sids.a.astype(int)))
    names = json.loads(CLUSTER_NAMES.read_text(encoding="utf-8"))
    gold = [json.loads(l) for l in (ROOT / "eval" / "gold_queries.jsonl").open(encoding="utf-8")]
    model, tok = load_sid_model()
    dec = SidDecoder(model, tok, sids)

    def track(t):
        r = meta.loc[t]
        return {"content_desc": r.content_desc, "tags": list(r.tags), "context": list(r.context),
                "cluster_name": names[str(level1[t])]["name"], "track_name": r.track_name, "artists": r.artists}

    t0 = time.perf_counter()
    cold = explain("warm-up request", track(sids.track_id.iloc[0]), timeout=120)     # loads the explainer model
    cold_s = time.perf_counter() - t0
    rows = []
    for n in gold_nums:
        q = gold[n - 1]["query"]
        trs = [track(t) for t, _ in dec.search([q], n=top, num_beams=BEAMS, batch_size=1)[0]]
        t1 = time.perf_counter()
        es = explain_batch(q, trs)                                    # one call per query
        ms = (time.perf_counter() - t1) * 1000
        for tr, e in zip(trs, es):
            rows.append({"gold": n, "query": q, "track": f"{tr['track_name']} — {tr['artists'].replace(';', ', ')}",
                         "tags": tr["tags"], "context": tr["context"], **e, "ms": ms, "batch": len(trs)})
    stop.set(); th.join()
    return {"rows": rows, "cold_load_s": cold_s, "cold_result": cold["source"], "gpu_baseline_mib": base_mib,
            "gpu_peak_mib": peak[0], "model": EXPLAIN_MODEL}


if __name__ == "__main__":
    import sys
    r = demo()
    for x in r["rows"]:
        print(f"#{x['gold']} {x['query']!r} -> {x['track']} [{x['source']}, {x['ms']:.0f} ms]\n    {x['text']}"
              + (f"\n    (fallback: {x['reason']}; LLM text: {x.get('llm_text')!r})" if x["source"] == "fallback" else ""))
    ms = sorted({(x["gold"], x["ms"]) for x in r["rows"]}, key=lambda v: v[1])
    ms = [m for _, m in ms]
    print(f"\nlatency per batch call ({r['rows'][0]['batch']} tracks): p50 {ms[len(ms) // 2]:.0f} ms, max {ms[-1]:.0f} ms | cold load {r['cold_load_s']:.1f}s"
          f" | llm {sum(x['source'] == 'llm' for x in r['rows'])}/{len(r['rows'])}")
    print(f"GPU memory (nvidia-smi, all processes): baseline {r['gpu_baseline_mib']} MiB, peak {r['gpu_peak_mib']} MiB")
    sys.exit(0)
