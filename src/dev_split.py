"""Tune-half DEV set for model selection (build side only; never touches Eval B strings or eval-half labels).

- DEV tracks: 15% of tune-labeled catalog tracks (seed 21, over sorted track_ids), saved to data/proc/dev_tracks.csv.
  Their human (tune_context) pairs are removed from training; their template / paraphrase pairs stay, exactly
  like eval-half tracks for Eval B.
- Dev queries: per canonical context word (Eval B's word list), up to 6 distinct TRAIN context-template queries
  (3 templates x the word and its CONTEXT_SYNONYMS); words without synonyms get 3.
- Pool: catalog minus TRAINING-side tune tracks (tune-labeled, not dev).
- Hit: a dev track whose TUNE labels contain one of the word's playlist words (CONTEXT_TO_PLAYLIST_WORDS).
  dev playlist precision@10 = |top10 & relevant| / 10, mean over all dev queries; chance = |relevant| / |pool|.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.config import CONTEXT_SYNONYMS
from src.gen_queries import CONTEXT_PHRASES, TRAIN_TEMPLATES, eval_b_words
from src.paths import CATALOG, CONTEXT_LABELS, PROC, ROOT
from src.prep import CONTEXT_TO_PLAYLIST_WORDS

DEV_TRACKS = PROC / "dev_tracks.csv"
DEV_FAISS = ROOT / "eval" / "dev_faiss.json"
DEV_FRAC, DEV_SEED = 0.15, 21
DEV_QUERIES_PER_WORD = 6
K = 10
CONTEXT_TEMPLATE_IDS = sorted(t for t, (typ, _) in TRAIN_TEMPLATES.items() if typ == "context")


def tune_labels() -> pd.DataFrame:
    lab = pd.read_parquet(CONTEXT_LABELS)
    cat_ids = set(pd.read_parquet(CATALOG, columns=["track_id"]).track_id)
    return lab[(lab.label_split == "tune") & lab.track_id.isin(cat_ids)]


def select_dev_tracks(tune: pd.DataFrame, frac: float = DEV_FRAC, seed: int = DEV_SEED) -> list[str]:
    ids = np.array(sorted(tune.track_id.unique()))
    n = int(round(frac * len(ids)))
    return sorted(ids[np.random.default_rng(seed).permutation(len(ids))[:n]].tolist())


def dev_tracks() -> set[str]:
    """Frozen DEV tracks (file), created on first use."""
    if not DEV_TRACKS.exists():
        pd.DataFrame({"track_id": select_dev_tracks(tune_labels())}).to_csv(DEV_TRACKS, index=False)
    return set(pd.read_csv(DEV_TRACKS).track_id)


def training_tune_tracks() -> set[str]:
    return set(tune_labels().track_id) - dev_tracks()


def dev_pool() -> set[str]:
    return set(pd.read_parquet(CATALOG, columns=["track_id"]).track_id) - training_tune_tracks()


def dev_queries() -> list[dict]:
    out = []
    for w in eval_b_words():
        combos = [(tid, s) for s in [w, *CONTEXT_SYNONYMS.get(w, [])] for tid in CONTEXT_TEMPLATE_IDS]
        for tid, s in combos[:DEV_QUERIES_PER_WORD]:
            out.append({"query": TRAIN_TEMPLATES[tid][1].format(context=CONTEXT_PHRASES[s]), "context_word": w,
                        "surface": s, "template_id": tid, "playlist_words": CONTEXT_TO_PLAYLIST_WORDS[w]})
    return out


def relevant_by_word() -> dict[str, set[str]]:
    tune, dev = tune_labels(), dev_tracks()
    t = tune[tune.track_id.isin(dev)]
    return {w: set(t.track_id[t.context_word.isin(CONTEXT_TO_PLAYLIST_WORDS[w])]) for w in eval_b_words()}


def dev_precision(search, queries: list[dict] | None = None) -> dict:
    """search(list[str]) -> list[list[track_id]] (top K from the dev pool). Returns overall, per-word and chance."""
    queries = queries or dev_queries()
    rel, pool = relevant_by_word(), dev_pool()
    tops = search([q["query"] for q in queries])
    rows = [{"context_word": q["context_word"], "p": len(set(t[:K]) & rel[q["context_word"]]) / K}
            for q, t in zip(queries, tops)]
    df = pd.DataFrame(rows)
    per = df.groupby("context_word").p.mean()
    return {"dev_precision@10": float(df.p.mean()), "n_queries": len(df),
            "chance": float(np.mean([len(rel[q["context_word"]] & pool) / len(pool) for q in queries])),
            "per_word": {w: {"precision@10": float(per[w]), "chance": len(rel[w] & pool) / len(pool),
                             "relevant_dev_tracks": len(rel[w])} for w in per.index}}


def faiss_dev() -> dict:
    from src.baseline import FaissBaseline, load_catalog
    cat = load_catalog()
    fb = FaissBaseline(cat)
    ix = fb.restrict(dev_pool())
    res = dev_precision(lambda qs: fb.search(qs, K, ix))
    res.update({"system": "faiss_baseline (frozen, 9dcece8)", "n_dev_tracks": len(dev_tracks()),
                "pool_size": len(dev_pool())})
    DEV_FAISS.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = faiss_dev()
    print(json.dumps({k: v for k, v in r.items() if k != "per_word"}, indent=1))
    for w, v in r["per_word"].items():
        print(f"  {w:<10} p@10 {v['precision@10']:.3f}  chance {v['chance']:.4f}  relevant dev tracks {v['relevant_dev_tracks']}")
    sys.exit(0)
