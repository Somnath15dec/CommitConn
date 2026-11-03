"""Step F-prep: training pool for the SID generator.

Pool = train_pairs.jsonl (templated) + human_pairs.jsonl (tune-half labels) + paraphrase_pairs.jsonl (LLM; whatever
exists now, so this can be rebuilt when the paraphrase run finishes). Each record: query -> SID of its track.
Exact (query, track_id) duplicates across sources are dropped (first source wins, in that order). Any query that
normalized-matches an Eval A / Eval B / gold query is dropped (expected 0).
Human (tune_context) pairs of DEV tracks (src/dev_split.py) are removed, so dev precision is measured on tracks
whose playlist labels were never trained on.
Validation: 1,000 random pairs (seed 19) held out from this pool only (never Eval A/B/gold).
Per-epoch sampler: at most MAX_PER_STRING pairs per query string, resampled each epoch (seed 100 + epoch).

Frozen: `python -m src.train_data --freeze` writes data/proc/train_pool.parquet and val_pairs.parquet; training
reads those files (frozen_split) and never re-derives them.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.dev_split import dev_tracks
from src.gen_queries import read_jsonl
from src.paraphrase import GOLD, PARAPHRASE_PAIRS, normalize
from src.paths import CONTEXT_LABELS, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, PROC, TRAIN_PAIRS

HUMAN_PAIRS = PROC / "human_pairs.jsonl"
SIDS = PROC / "sids.parquet"
TRAIN_POOL = PROC / "train_pool.parquet"
VAL_PAIRS = PROC / "val_pairs.parquet"
SOURCES = [("template", TRAIN_PAIRS), ("tune_context", HUMAN_PAIRS), ("paraphrase", PARAPHRASE_PAIRS)]
VAL_N, VAL_SEED = 1000, 19
MAX_PER_STRING = 150
EPOCH_SEED_BASE = 100


def eval_strings() -> set[str]:
    return {normalize(r["query"]) for p in (EVAL_PAIRS, EVAL_CONTEXT_QUERIES, GOLD) for r in read_jsonl(p)}


def load_pool(sources=SOURCES) -> tuple[pd.DataFrame, dict]:
    """DataFrame [query, track_id, source, a, b, c, d] and build stats."""
    frames, stats = [], {"raw": {}, "missing_files": []}
    for name, path in sources:
        if not path.exists():
            stats["missing_files"].append(str(path.name))
            continue
        rows = read_jsonl(path)
        stats["raw"][name] = len(rows)
        frames.append(pd.DataFrame({"query": [r["query"] for r in rows], "track_id": [r["track_id"] for r in rows],
                                    "source": name}))
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.drop_duplicates(["query", "track_id"]).reset_index(drop=True)
    stats["dropped_duplicate_pairs"] = n0 - len(df)
    ev = eval_strings()
    leak = df["query"].map(normalize).isin(ev)
    stats["dropped_eval_match"] = int(leak.sum())
    df = df[~leak].reset_index(drop=True)
    dev_human = (df.source == "tune_context") & df.track_id.isin(dev_tracks())
    stats["dropped_dev_track_human_pairs"] = int(dev_human.sum())
    df = df[~dev_human].reset_index(drop=True)
    sids = pd.read_parquet(SIDS)
    df = df.merge(sids, on="track_id", how="inner")
    stats["pool"] = df.source.value_counts().to_dict()
    return df, stats


def split_val(pool: pd.DataFrame, n: int = VAL_N, seed: int = VAL_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, val): n random pairs held out, chosen by a seed-19 permutation over a stable (sorted) pool order."""
    p = pool.sort_values(["source", "query", "track_id"]).reset_index(drop=True)
    idx = np.random.default_rng(seed).permutation(len(p))
    return p.drop(index=idx[:n]).reset_index(drop=True), p.iloc[idx[:n]].reset_index(drop=True)


def epoch_indices(train: pd.DataFrame, epoch: int, cap: int = MAX_PER_STRING) -> np.ndarray:
    """Row indices for one epoch: at most `cap` per query string (random subset per epoch), shuffled."""
    rng = np.random.default_rng(EPOCH_SEED_BASE + epoch)
    out = []
    for _, ix in train.groupby("query", sort=True).indices.items():
        out.append(ix if len(ix) <= cap else rng.choice(ix, cap, replace=False))
    out = np.concatenate(out)
    rng.shuffle(out)
    return out


def freeze() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pool, stats = load_pool()
    train, val = split_val(pool)
    train.to_parquet(TRAIN_POOL, index=False)
    val.to_parquet(VAL_PAIRS, index=False)
    return train, val, stats


def frozen_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (TRAIN_POOL.exists() and VAL_PAIRS.exists()):
        raise FileNotFoundError("run `python -m src.train_data --freeze` first")
    return pd.read_parquet(TRAIN_POOL), pd.read_parquet(VAL_PAIRS)


def eval_half_only_tracks() -> set[str]:
    lab = pd.read_parquet(CONTEXT_LABELS)
    return set(lab.track_id[lab.label_split == "eval"]) - set(lab.track_id[lab.label_split == "tune"])


def main() -> None:
    if "--freeze" in sys.argv:
        train, val, stats = freeze()
        print(f"FROZEN -> {TRAIN_POOL.name}, {VAL_PAIRS.name}")
    else:
        pool, stats = load_pool()
        train, val = split_val(pool)
    pool = pd.concat([train, val])
    ep = epoch_indices(train, 0)
    print(f"raw per source: {stats['raw']} | missing: {stats['missing_files']}")
    print(f"dropped duplicate (query, track) pairs: {stats['dropped_duplicate_pairs']:,} | "
          f"eval-string matches: {stats['dropped_eval_match']} | "
          f"dev-track human pairs: {stats['dropped_dev_track_human_pairs']:,}")
    print(f"pool: {len(pool):,} {stats['pool']} | train {len(train):,} | val {len(val):,} "
          f"{val.source.value_counts().to_dict()}")
    print(f"distinct query strings: {pool['query'].nunique():,} | tracks: {pool.track_id.nunique():,}")
    print(f"epoch 0 after {MAX_PER_STRING}/string cap: {len(ep):,} pairs "
          f"{train.iloc[ep].source.value_counts().to_dict()}")


if __name__ == "__main__":
    sys.exit(main())
