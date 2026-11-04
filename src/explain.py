"""Step J: one-sentence explanation for a recommended track, grounded in that track's own tags / context.

LLM: a small local instruct model via Ollama (EXPLAIN_MODEL), JSON-schema output, think off, 3 s timeout.
Input: the user's query + the track's content_desc, tags, context and level-1 cluster name (no title/artist needed).
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
MAX_WORDS = 25
OPTIONS = {"temperature": 0.3, "num_predict": 60, "num_ctx": 1024}
KEEP_ALIVE = "30m"
SCHEMA = {"type": "object", "properties": {"explanation": {"type": "string"}}, "required": ["explanation"]}
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

SYSTEM = "You explain music recommendations in one short, friendly sentence. Answer with JSON only."
PROMPT = """A listener asked: "{query}"
Recommended track facts: {facts}
Write ONE sentence (at most 20 words) explaining why this track fits the request.
Use ONLY the sound, mood and use words listed in the facts; never invent others. Do not name the track or artist.
Return {{"explanation": "..."}}."""


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
def facts(track: dict) -> str:
    return (f"{track['content_desc']} Cluster: {track.get('cluster_name', '')}. "
            f"Sound words: {', '.join(t for t in track['tags'] if t not in HIDDEN)}. "
            f"Use words: {', '.join(track['context']) or 'general listening'}.")


def ask(query: str, track: dict, timeout: float = TIMEOUT_S, url: str = OLLAMA_URL, post=requests.post) -> str:
    r = post(url, timeout=timeout, json={
        "model": EXPLAIN_MODEL, "think": False, "stream": False, "format": SCHEMA, "options": OPTIONS,
        "keep_alive": KEEP_ALIVE,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT.format(query=query, facts=facts(track))}]})
    r.raise_for_status()
    return " ".join(json.loads(r.json()["message"]["content"])["explanation"].split())


def explain(query: str, track: dict, timeout: float = TIMEOUT_S, url: str = OLLAMA_URL, post=requests.post) -> dict:
    """{"text", "source": "llm" | "fallback", "reason"}; never raises."""
    try:
        text = ask(query, track, timeout, url, post)
    except requests.Timeout:
        return {"text": fallback(query, track), "source": "fallback", "reason": "timeout"}
    except requests.RequestException as e:
        return {"text": fallback(query, track), "source": "fallback", "reason": f"ollama unavailable ({type(e).__name__})"}
    except (ValueError, KeyError, TypeError) as e:
        return {"text": fallback(query, track), "source": "fallback", "reason": f"bad response ({type(e).__name__})"}
    if not text or len(text.split()) > MAX_WORDS:
        return {"text": fallback(query, track), "source": "fallback", "reason": "empty or too long", "llm_text": text}
    bad = ungrounded(text, track)
    if bad:
        return {"text": fallback(query, track), "source": "fallback", "reason": f"ungrounded: {', '.join(bad)}",
                "llm_text": text}
    return {"text": text, "source": "llm", "reason": "ok"}


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
        for t, _ in dec.search([q], n=top, num_beams=BEAMS, batch_size=1)[0]:
            tr = track(t)
            t1 = time.perf_counter()
            e = explain(q, tr)
            rows.append({"gold": n, "query": q, "track": f"{tr['track_name']} — {tr['artists'].replace(';', ', ')}",
                         "tags": tr["tags"], "context": tr["context"], **e, "ms": (time.perf_counter() - t1) * 1000})
    stop.set(); th.join()
    return {"rows": rows, "cold_load_s": cold_s, "cold_result": cold["source"], "gpu_baseline_mib": base_mib,
            "gpu_peak_mib": peak[0], "model": EXPLAIN_MODEL}


if __name__ == "__main__":
    import sys
    r = demo()
    for x in r["rows"]:
        print(f"#{x['gold']} {x['query']!r} -> {x['track']} [{x['source']}, {x['ms']:.0f} ms]\n    {x['text']}"
              + (f"\n    (fallback: {x['reason']}; LLM text: {x.get('llm_text')!r})" if x["source"] == "fallback" else ""))
    ms = sorted(x["ms"] for x in r["rows"])
    print(f"\nlatency per explanation: p50 {ms[len(ms) // 2]:.0f} ms, max {ms[-1]:.0f} ms | cold load {r['cold_load_s']:.1f}s"
          f" | llm {sum(x['source'] == 'llm' for x in r['rows'])}/{len(r['rows'])}")
    print(f"GPU memory (nvidia-smi, all processes): baseline {r['gpu_baseline_mib']} MiB, peak {r['gpu_peak_mib']} MiB")
    sys.exit(0)
