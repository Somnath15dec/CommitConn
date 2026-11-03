"""Step E: names for the 256 level-1 SID clusters via local Ollama (qwen3.5-6k, think off, JSON schema).

Per cluster: 10 sample content_descs (distinct descs first, seed 22), shown through paraphrase.prompt_desc (humanized
genres, no "neutral mood" / "moderate energy", category as plain words). The model returns a 2-4 word name; a name
outside 2-4 words is retried once, then kept with valid=False. Resumable: clusters already named are skipped.

Writes data/proc/cluster_names.json: {"<a>": {"name": ..., "valid": bool, "n_tracks": int, "samples": [...]}}.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import requests

from src.analyze_readiness import content_desc
from src.paraphrase import MODEL, OLLAMA_URL, prompt_desc
from src.paths import CATALOG, PROC

CLUSTER_NAMES = PROC / "cluster_names.json"
SIDS = PROC / "sids.parquet"
N_SAMPLES, SAMPLE_SEED = 10, 22
MIN_WORDS, MAX_WORDS = 2, 4
OPTIONS = {"temperature": 0.3, "num_ctx": 6144}
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
PROMPT = """These {n} descriptions are tracks from ONE cluster of a music catalog.
Give the cluster a short, human-friendly name of 2 to 4 words that captures what the tracks share
(genre, mood, energy or use). Use only what the descriptions say. No quotes, no emojis, no numbering.
Return {{"name": "..."}}.

{lines}"""


def samples(descs: pd.Series, n: int = N_SAMPLES, seed: int = SAMPLE_SEED) -> list[str]:
    """Up to n descs: distinct ones first (random order), then repeats if the cluster has fewer distinct descs."""
    rng = np.random.default_rng(seed)
    uniq = descs.drop_duplicates().tolist()
    rng.shuffle(uniq)
    out = uniq[:n]
    if len(out) < n:
        rest = descs.tolist()
        out += [rest[i] for i in rng.permutation(len(rest))[:n - len(out)]]
    return out


def valid_name(name: str) -> bool:
    return MIN_WORDS <= len(name.split()) <= MAX_WORDS


def clean(name: str) -> str:
    return " ".join(str(name).strip().strip("\"'.").split())


def ask(lines: list[str]) -> str:
    prompt = PROMPT.format(n=len(lines), lines="\n".join(f"- {prompt_desc(d)}" for d in lines))
    r = requests.post(OLLAMA_URL, timeout=300, json={
        "model": MODEL, "think": False, "stream": False, "format": SCHEMA, "options": OPTIONS,
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return clean(json.loads(r.json()["message"]["content"])["name"])


def run(ask_fn=ask, out_path=CLUSTER_NAMES) -> dict:
    cat = pd.read_parquet(CATALOG)
    cat["content_desc"] = cat.apply(content_desc, axis=1)
    m = pd.read_parquet(SIDS).merge(cat[["track_id", "content_desc"]], on="track_id")
    names = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for a, g in m.groupby("a", sort=True):
        key = str(int(a))
        if key in names:
            continue
        lines = samples(g.content_desc.reset_index(drop=True), seed=SAMPLE_SEED + int(a))
        t0, name, ok = time.perf_counter(), "", False
        for _ in range(2):
            try:
                name = ask_fn(lines)
                ok = valid_name(name)
            except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
                name, ok = f"ERROR {type(e).__name__}", False
            if ok:
                break
        names[key] = {"name": name, "valid": ok, "n_tracks": len(g), "samples": lines}
        out_path.write_text(json.dumps(names, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"cluster {key}: {name!r} valid={ok} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return names


if __name__ == "__main__":
    n = run()
    print(f"done: {len(n)} clusters, invalid {sum(not v['valid'] for v in n.values())}")
    sys.exit(0)
