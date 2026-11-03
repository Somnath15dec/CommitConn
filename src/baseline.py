"""Step 13 / D: FAISS dense-retrieval baseline, scored exactly per eval/PLAN.md (incl. Amendments 1-5).

Items: all-MiniLM-L6-v2 (normalized) embedding of content_desc only (no audio). Queries: same encoder.
Exact inner-product search (faiss.IndexFlatIP). Encoder on CPU (the GPU is busy with the Ollama paraphrase run).
Frozen: nothing here is tuned on results.

Writes eval/results_baseline.json and eval/results_baseline.md.
"""
from __future__ import annotations

import json
import sys
import time

import faiss
import numpy as np
import pandas as pd

from src.analyze_readiness import EMBED_MODEL, content_desc
from src.paths import CATALOG, CONTEXT_LABELS, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, ROOT

GOLD = ROOT / "eval" / "gold_queries.jsonl"
RESULTS_JSON = ROOT / "eval" / "results_baseline.json"
RESULTS_MD = ROOT / "eval" / "results_baseline.md"
K, K_CONDENSED = 10, 50
BOOT_N, BOOT_SEED = 1000, 13
N_WARMUP = 10
DEVICE = "cpu"
SANITY_GOLD = [1, 4, 6, 16, 26]          # 1-based line numbers in gold_queries.jsonl


# ---------------------------------------------------------------- io
def read_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_catalog() -> pd.DataFrame:
    cat = pd.read_parquet(CATALOG)
    cat["content_desc"] = cat.apply(content_desc, axis=1)
    cat["T"] = [set(t) | set(g) | set(c) for t, g, c in zip(cat.tags, cat.genres, cat.context)]
    return cat


# ---------------------------------------------------------------- system
class FaissBaseline:
    def __init__(self, cat: pd.DataFrame, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        self.model = model
        self.ids = cat.track_id.to_numpy()
        self.vecs = self.encode(cat.content_desc.tolist())
        self.full = self._index(np.arange(len(self.ids)))

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=256, normalize_embeddings=True,
                                            show_progress_bar=False), dtype="float32")

    def _index(self, rows: np.ndarray):
        ix = faiss.IndexFlatIP(self.vecs.shape[1])
        ix.add(self.vecs[rows])
        return ix, self.ids[rows]

    def restrict(self, keep_ids: set[str]):
        return self._index(np.flatnonzero(np.isin(self.ids, list(keep_ids))))

    def search(self, queries: list[str], k: int = K, index=None) -> list[list[str]]:
        ix, ids = index or self.full
        _, nn = ix.search(self.encode(queries), k)
        return [[ids[j] for j in row if j >= 0] for row in nn]       # catalog ids are unique -> distinct


# ---------------------------------------------------------------- metrics (PLAN.md)
def bootstrap_ci(values, n: int = BOOT_N, seed: int = BOOT_SEED) -> list[float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summary(values) -> dict:
    return {"mean": float(np.mean(values)), "ci95": bootstrap_ci(values), "n": len(values)}


def precision_at_k(top: list[str], relevant: set[str], k: int = K) -> float:
    return len(set(top[:k]) & relevant) / k


def condensed_precision(top50: list[str], relevant: set[str], judged: set[str], k: int = K) -> tuple[float, int]:
    """Amendment 3: drop unjudged, keep order, m = min(k, judged in top 50); score hits/m (0 if m == 0)."""
    j = [t for t in top50 if t in judged]
    m = min(k, len(j))
    return (sum(t in relevant for t in j[:m]) / m if m else 0.0), len(j)


def hit_ndcg(top: list[str], target: str, k: int = K) -> tuple[float, float]:
    if target in top[:k]:
        r = top.index(target) + 1
        return 1.0, 1.0 / np.log2(r + 1)
    return 0.0, 0.0


def tag_match(top: list[str], wanted: set[str], T: dict[str, set], k: int = K) -> float:
    return sum(len(wanted & T[t]) / len(wanted) for t in top[:k]) / k


def avoid_violation(top: list[str], avoid: set[str], TG: dict[str, set], k: int = K) -> float:
    got = top[:k]                                      # empty slots are left out
    return sum(bool(avoid & TG[t]) for t in got) / len(got) if got else 0.0


def query_aware_ceiling(evals: list[dict], cat: pd.DataFrame) -> float:
    """PLAN.md's "query-aware ceiling", reported as the random-within-match expectation (PLAN.md note 1): expected
    Hit@10 of a random choice among tracks matching the query's wanted_tags. Not an upper bound."""
    T = cat["T"].tolist()
    vals = []
    for p in evals:
        w = set(p["wanted_tags"])
        n = sum(w <= t for t in T)
        vals.append(min(1.0, K / n))
    return float(np.mean(vals))


def identical_desc_ceiling(evals: list[dict], cat: pd.DataFrame) -> float:
    size = cat.content_desc.map(cat.content_desc.value_counts())
    g = dict(zip(cat.track_id, size))
    return float(np.mean([min(1.0, K / g[p["track_id"]]) for p in evals]))


# ---------------------------------------------------------------- evaluation
def evaluate(cat: pd.DataFrame, sys_: FaissBaseline) -> tuple[dict, dict]:
    labels = pd.read_parquet(CONTEXT_LABELS)
    ev_lab = labels[labels.label_split == "eval"]
    tune_ids = set(labels.track_id[labels.label_split == "tune"])
    pool = set(cat.track_id) - tune_ids
    judged = set(ev_lab.track_id)
    T = dict(zip(cat.track_id, cat["T"]))
    TG = {t: set(a) | set(b) for t, a, b in zip(cat.track_id, cat.tags, cat.genres)}
    valid_ids = set(cat.track_id)

    evb, eva = read_jsonl(EVAL_CONTEXT_QUERIES), read_jsonl(EVAL_PAIRS)
    gold_all = read_jsonl(GOLD)
    gold = [(i + 1, g) for i, g in enumerate(gold_all) if g["type"] == "accuracy"]
    res, extra = {"system": "faiss_baseline", "encoder": EMBED_MODEL, "device": DEVICE,
                  "index": "IndexFlatIP (exact cosine)", "k": K}, {}

    # ---- Eval B (pool only)
    pool_ix = sys_.restrict(pool)
    top50 = sys_.search([q["query"] for q in evb], K_CONDENSED, pool_ix)
    rel_by_words = lambda pw: set(ev_lab.track_id[ev_lab.context_word.isin(pw)])
    rows = []
    for q, t50 in zip(evb, top50):
        rel = rel_by_words(q["playlist_words"])
        cp, nj = condensed_precision(t50, rel, judged)
        rows.append({"query": q["query"], "context_word": q["context_word"], "surface": q["surface"],
                     "precision@10": precision_at_k(t50, rel), "condensed_precision@10": cp,
                     "judged_in_top50": nj, "chance": len(rel & pool) / len(pool), "top10": t50[:K]})
    b = pd.DataFrame(rows)
    per_word = {}
    for w, g in b.groupby("context_word", sort=True):
        per_word[w] = {"precision@10": summary(g["precision@10"]), "chance": float(g.chance.iloc[0]),
                       "condensed_precision@10": float(g["condensed_precision@10"].mean()),
                       "judged_in_top50": float(g.judged_in_top50.mean()), "n": len(g)}
    res["eval_b"] = {"pool_size": len(pool), "n_queries": len(b),
                     "precision@10": summary(b["precision@10"]), "chance": float(b.chance.mean()),
                     "condensed_precision@10": summary(b["condensed_precision@10"]),
                     "judged_in_top50_mean": float(b.judged_in_top50.mean()), "per_context_word": per_word,
                     "per_query": b.drop(columns=["top10"]).to_dict("records")}

    # ---- Eval A (full catalog)
    topA = sys_.search([p["query"] for p in eva])
    hn = [hit_ndcg(t, p["track_id"]) for t, p in zip(topA, eva)]
    res["eval_a"] = {"n_queries": len(eva), "hit@10": summary([h for h, _ in hn]),
                     "ndcg@10": summary([n for _, n in hn]),
                     "random_within_match_expectation": query_aware_ceiling(eva, cat),
                     "ceiling_identical_desc": identical_desc_ceiling(eva, cat)}

    # ---- gold accuracy (full catalog)
    topG = sys_.search([g["query"] for _, g in gold])
    gq = []
    for (num, g), top in zip(gold, topG):
        w = set(g["wanted_tags"]) | set(g["wanted_context"])
        r = {"num": num, "query": g["query"], "tag_match@10": tag_match(top, w, T), "top10": top}
        if g.get("avoid_tags"):
            r["avoid_violation@10"] = avoid_violation(top, set(g["avoid_tags"]), TG)
        gq.append(r)
    av = [r for r in gq if "avoid_violation@10" in r]
    res["gold"] = {"n_accuracy": len(gq), "tag_match@10": summary([r["tag_match@10"] for r in gq]),
                   "avoid_violation@10": {"per_query": {str(r["num"]): r["avoid_violation@10"] for r in av},
                                          "mean": float(np.mean([r["avoid_violation@10"] for r in av]))},
                   "per_query": [{k: v for k, v in r.items() if k != "top10"} for r in gq],
                   "robustness": "n/a (guards come later)"}

    # ---- valid-item rate (Eval A + Eval B + gold accuracy)
    all_top = topA + [t[:K] for t in top50] + topG
    n_valid = sum(len({t for t in top if t in valid_ids}) for top in all_top)
    res["valid_item_rate"] = n_valid / (K * len(all_top))

    # ---- latency: batch size 1, query string -> top-10 ids, 10 warm-ups discarded
    queries = [(p["query"], None) for p in eva] + [(q["query"], pool_ix) for q in evb] + \
              [(g["query"], None) for _, g in gold]
    for q, ix in queries[:N_WARMUP]:
        sys_.search([q], K, ix)
    ms = []
    for q, ix in queries:
        t0 = time.perf_counter()
        sys_.search([q], K, ix)
        ms.append((time.perf_counter() - t0) * 1000)
    res["latency_ms"] = {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
                         "n": len(ms), "batch_size": 1, "warmup": N_WARMUP,
                         "note": "encoder on CPU; Ollama paraphrase run was using the GPU concurrently",
                         "end_to_end_with_explanation": "n/a (no explanation step)"}

    rows_by_num = {r["num"]: r for r in gq}
    by_id = cat.set_index("track_id")
    extra["sanity"] = {n: [(by_id.loc[t, "track_name"], by_id.loc[t, "artists"], by_id.loc[t, "content_desc"])
                           for t in rows_by_num[n]["top10"][:5]] for n in SANITY_GOLD}
    extra["sanity_queries"] = {n: rows_by_num[n]["query"] for n in SANITY_GOLD}
    return res, extra


# ---------------------------------------------------------------- report
def fmt(s: dict) -> str:
    return f"{s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"


def markdown(res: dict, extra: dict) -> str:
    b, a, g = res["eval_b"], res["eval_a"], res["gold"]
    L = ["# FAISS baseline results (frozen)", "",
         f"System: `{res['encoder']}` (normalized) on content_desc only, {res['index']}, encoder on {res['device']}. "
         "Scored per `eval/PLAN.md` (Amendments 1-5). Means with bootstrap 95% CI over queries "
         f"({BOOT_N} resamples, seed {BOOT_SEED}).", "",
         "## Summary", "", "| Metric | Value |", "|---|---|",
         f"| **Eval B playlist precision@10 (headline)** | **{fmt(b['precision@10'])}** |",
         f"| Eval B chance | {b['chance']:.3f} |",
         f"| Eval B condensed precision@10 | {fmt(b['condensed_precision@10'])} |",
         f"| Eval B judged tracks in top 50 (mean) | {b['judged_in_top50_mean']:.1f} |",
         f"| Eval A Hit@10 | {fmt(a['hit@10'])} |",
         f"| Eval A NDCG@10 | {fmt(a['ndcg@10'])} |",
         f"| Eval A random-within-match expectation (primary reference) | {a['random_within_match_expectation']:.3f} |",
         f"| Eval A ceiling, identical description | {a['ceiling_identical_desc']:.3f} |",
         f"| Gold tag-match@10 | {fmt(g['tag_match@10'])} |",
         f"| Gold avoid-violation@10 (mean of 3) | {g['avoid_violation@10']['mean']:.3f} "
         f"({', '.join(f'#{k}: {v:.1f}' for k, v in g['avoid_violation@10']['per_query'].items())}) |",
         f"| Robustness | {g['robustness']} |",
         f"| Valid-item rate | {res['valid_item_rate']:.3f} |",
         f"| Latency p50 / p90 (ms, batch 1) | {res['latency_ms']['p50']:.1f} / {res['latency_ms']['p90']:.1f} |",
         "| End-to-end latency incl. explanation | n/a |", "",
         f"Eval B pool: {b['pool_size']:,} tracks (no tune-half label). Latency note: {res['latency_ms']['note']}.", "",
         "## Eval B per context word", "",
         "| Context word | precision@10 [95% CI] | chance | lift vs chance | condensed p@10 | judged in top 50 |",
         "|---|---|---|---|---|---|"]
    for w, v in b["per_context_word"].items():
        p = v["precision@10"]
        lift = p["mean"] / v["chance"] if v["chance"] else float("nan")
        L.append(f"| {w} | {fmt(p)} | {v['chance']:.3f} | {lift:.1f}x | {v['condensed_precision@10']:.3f} | "
                 f"{v['judged_in_top50']:.1f} |")
    L += ["", "## Sanity view (gold queries, top 5)", ""]
    for n, items in extra["sanity"].items():
        L.append(f"**#{n} \"{extra['sanity_queries'][n]}\"**")
        L.append("")
        for name, art, cd in items:
            L.append(f"1. {name} — {art.replace(';', ', ')} — _{cd}_")
        L.append("")
    return "\n".join(L)


def main() -> None:
    cat = load_catalog()
    res, extra = evaluate(cat, FaissBaseline(cat))
    RESULTS_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    RESULTS_MD.write_text(markdown(res, extra) + "\n", encoding="utf-8")
    print(markdown(res, extra))


if __name__ == "__main__":
    sys.exit(main())
