"""Step I: final evaluation (run once), per eval/PLAN.md and its note 4 (pre-registered 56be084).

Systems: FAISS baseline (frozen; recomputed and checked against eval/results_baseline.json), SID model
(checkpoints/run1/epoch3, trie-constrained beam 50, top 10 distinct), hybrid (RRF k=60 over both top-50 lists,
same pool; ties by best rank, then track_id). Metric code is imported from src/baseline.py, so scoring is identical.

Writes eval/results_final.json and eval/results_final.md.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from src import baseline as B
from src.decode import SidDecoder
from src.paths import CONTEXT_LABELS, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, ROOT
from src.train import BASE_MODEL
from src.train_data import SIDS

CKPT = ROOT / "checkpoints" / "run1" / "epoch3"
RESULTS_JSON = ROOT / "eval" / "results_final.json"
RESULTS_MD = ROOT / "eval" / "results_final.md"
BEAMS, TOP_N = 50, 50
DECODE_BATCH = 2
RRF_K = 60
SYSTEMS = ("faiss", "sid", "hybrid")


# ---------------------------------------------------------------- systems
def load_sid_model(ckpt=CKPT, device="cuda"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    base.resize_token_embeddings(len(tok), mean_resizing=False)       # new rows are replaced by the adapter
    model = PeftModel.from_pretrained(base, ckpt).to(device).eval()
    return model, tok


def rrf(lists: list[list[str]], k: int = RRF_K, n: int = TOP_N) -> list[str]:
    """Reciprocal rank fusion: score = sum 1/(k + rank), rank 1-based; ties by best rank, then track_id."""
    score, best = {}, {}
    for lst in lists:
        for r, t in enumerate(lst, 1):
            score[t] = score.get(t, 0.0) + 1.0 / (k + r)
            best[t] = min(best.get(t, r), r)
    return sorted(score, key=lambda t: (-score[t], best[t], t))[:n]


class Systems:
    def __init__(self, cat: pd.DataFrame, sids: pd.DataFrame, model, tok, tune_ids: set[str]):
        self.fb = B.FaissBaseline(cat)
        self.pool_ix = self.fb.restrict(set(cat.track_id) - tune_ids)
        self.dec_full = SidDecoder(model, tok, sids)
        self.dec_pool = SidDecoder(model, tok, sids, exclude=tune_ids)

    def faiss(self, qs, pool: bool, n=TOP_N):
        return self.fb.search(qs, n, self.pool_ix if pool else None)

    def sid(self, qs, pool: bool, n=TOP_N):
        dec = self.dec_pool if pool else self.dec_full
        return [[t for t, _ in r] for r in dec.search(qs, n=n, num_beams=BEAMS, batch_size=DECODE_BATCH)]

    def all(self, qs, pool: bool) -> dict[str, list[list[str]]]:
        f, s = self.faiss(qs, pool), self.sid(qs, pool)
        return {"faiss": f, "sid": s, "hybrid": [rrf([a, b]) for a, b in zip(s, f)]}


# ---------------------------------------------------------------- scoring (PLAN.md, via src/baseline.py)
def paired_diff(a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"mean_diff": float(d.mean()), "ci95": B.bootstrap_ci(d), "n": len(d),
            "queries_better": int((d > 0).sum()), "queries_worse": int((d < 0).sum()), "ties": int((d == 0).sum())}


def evaluate(cat, sys_: Systems) -> tuple[dict, dict]:
    labels = pd.read_parquet(CONTEXT_LABELS)
    ev_lab = labels[labels.label_split == "eval"]
    tune_ids = set(labels.track_id[labels.label_split == "tune"])
    pool = set(cat.track_id) - tune_ids
    judged = set(ev_lab.track_id)
    T = dict(zip(cat.track_id, cat["T"]))
    TG = {t: set(a) | set(b) for t, a, b in zip(cat.track_id, cat.tags, cat.genres)}
    valid = set(cat.track_id)
    evb, eva = B.read_jsonl(EVAL_CONTEXT_QUERIES), B.read_jsonl(EVAL_PAIRS)
    gold = [(i + 1, g) for i, g in enumerate(B.read_jsonl(B.GOLD)) if g["type"] == "accuracy"]

    t0 = time.time()
    tb = sys_.all([q["query"] for q in evb], pool=True)
    print(f"Eval B ranked ({time.time() - t0:.0f}s)", flush=True)
    ta = sys_.all([p["query"] for p in eva], pool=False)
    print(f"Eval A ranked ({time.time() - t0:.0f}s)", flush=True)
    tg = sys_.all([g["query"] for _, g in gold], pool=False)
    print(f"gold ranked ({time.time() - t0:.0f}s)", flush=True)

    res = {"note": "eval/PLAN.md note 4 (pre-registered 56be084); run once", "checkpoint": str(CKPT.relative_to(ROOT)),
           "decoding": f"trie-constrained beam {BEAMS}, top 10 distinct", "hybrid": f"RRF k={RRF_K}, top 50 + top 50",
           "systems": {}}
    rel_of = lambda pw: set(ev_lab.track_id[ev_lab.context_word.isin(pw)])
    per_query_b = {}
    for s in SYSTEMS:
        rows = []
        for q, t50 in zip(evb, tb[s]):
            rel = rel_of(q["playlist_words"])
            cp, nj = B.condensed_precision(t50, rel, judged)
            rows.append({"query": q["query"], "context_word": q["context_word"], "surface": q["surface"],
                         "precision@10": B.precision_at_k(t50, rel), "condensed_precision@10": cp,
                         "judged_in_top50": nj, "chance": len(rel & pool) / len(pool), "top10": t50[:10]})
        b = pd.DataFrame(rows)
        per_query_b[s] = b["precision@10"].tolist()
        per_word = {w: {"precision@10": B.summary(g["precision@10"]), "chance": float(g.chance.iloc[0]),
                        "condensed_precision@10": float(g["condensed_precision@10"].mean()),
                        "judged_in_top50": float(g.judged_in_top50.mean()), "n": len(g)}
                    for w, g in b.groupby("context_word", sort=True)}
        hn = [B.hit_ndcg(t[:10], p["track_id"]) for t, p in zip(ta[s], eva)]
        gq = []
        for (num, g), top in zip(gold, tg[s]):
            w = set(g["wanted_tags"]) | set(g["wanted_context"])
            r = {"num": num, "query": g["query"], "tag_match@10": B.tag_match(top[:10], w, T)}
            if g.get("avoid_tags"):
                r["avoid_violation@10"] = B.avoid_violation(top[:10], set(g["avoid_tags"]), TG)
            gq.append(r)
        av = [r for r in gq if "avoid_violation@10" in r]
        all_top = [t[:10] for t in ta[s] + tb[s] + tg[s]]
        res["systems"][s] = {
            "eval_b": {"pool_size": len(pool), "n_queries": len(b), "precision@10": B.summary(b["precision@10"]),
                       "chance": float(b.chance.mean()), "condensed_precision@10": B.summary(b["condensed_precision@10"]),
                       "judged_in_top50_mean": float(b.judged_in_top50.mean()), "per_context_word": per_word,
                       "per_query": b.drop(columns=["top10"]).to_dict("records")},
            "eval_a": {"n_queries": len(eva), "hit@10": B.summary([h for h, _ in hn]),
                       "ndcg@10": B.summary([x for _, x in hn]),
                       "random_within_match_expectation": B.query_aware_ceiling(eva, cat),
                       "ceiling_identical_desc": B.identical_desc_ceiling(eva, cat)},
            "gold": {"n_accuracy": len(gq), "tag_match@10": B.summary([r["tag_match@10"] for r in gq]),
                     "avoid_violation@10": {"per_query": {str(r["num"]): r["avoid_violation@10"] for r in av},
                                            "mean": float(np.mean([r["avoid_violation@10"] for r in av]))},
                     "per_query": gq, "robustness": "n/a (guards come in phase 4)"},
            "valid_item_rate": sum(len({x for x in t if x in valid}) for t in all_top) / (10 * len(all_top)),
            "returned_fewer_than_10": int(sum(len(t) < 10 for t in all_top))}
    res["additional_analysis_paired_eval_b"] = {
        "label": "additional analysis (PLAN.md note 4), not a pre-registered primary metric",
        "sid_minus_faiss": paired_diff(per_query_b["sid"], per_query_b["faiss"]),
        "hybrid_minus_faiss": paired_diff(per_query_b["hybrid"], per_query_b["faiss"])}
    extra = {"tg": tg, "gold": gold}
    return res, extra


def check_faiss_reproduces(res: dict) -> None:
    frozen = json.loads(B.RESULTS_JSON.read_text(encoding="utf-8"))
    a = [r["precision@10"] for r in frozen["eval_b"]["per_query"]]
    b = [r["precision@10"] for r in res["systems"]["faiss"]["eval_b"]["per_query"]]
    assert a == b, "FAISS recomputation does not reproduce the frozen baseline"
    assert abs(frozen["eval_a"]["hit@10"]["mean"] - res["systems"]["faiss"]["eval_a"]["hit@10"]["mean"]) < 1e-12
    assert abs(frozen["gold"]["tag_match@10"]["mean"] - res["systems"]["faiss"]["gold"]["tag_match@10"]["mean"]) < 1e-12


def latency(sys_: Systems, cat) -> dict:
    eva, evb = B.read_jsonl(EVAL_PAIRS), B.read_jsonl(EVAL_CONTEXT_QUERIES)
    gold = [g for g in B.read_jsonl(B.GOLD) if g["type"] == "accuracy"]
    qs = [(p["query"], False) for p in eva] + [(q["query"], True) for q in evb] + [(g["query"], False) for g in gold]
    out = {}
    fns = {"sid": lambda q, pool: sys_.sid([q], pool, n=10),
           "hybrid": lambda q, pool: [rrf([sys_.sid([q], pool)[0], sys_.faiss([q], pool)[0]])[:10]]}
    for name, fn in fns.items():
        for q, pool in qs[:B.N_WARMUP]:
            fn(q, pool)
        ms = []
        for q, pool in qs:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(q, pool)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
        out[name] = {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)), "n": len(ms),
                     "batch_size": 1, "warmup": B.N_WARMUP, "device": torch.cuda.get_device_name(0),
                     "end_to_end_with_explanation": "n/a (no explanation step yet)"}
        print(f"latency {name}: p50 {out[name]['p50']:.0f} ms", flush=True)
    return out


# ---------------------------------------------------------------- report
def fmt(s):
    return f"{s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"


def markdown(res: dict, frozen_latency: dict, sanity: dict) -> str:
    S = res["systems"]
    lat = lambda s: (f"{frozen_latency['p50']:.1f} / {frozen_latency['p90']:.1f} (CPU, frozen run)" if s == "faiss"
                     else f"{res['latency_ms'][s]['p50']:.0f} / {res['latency_ms'][s]['p90']:.0f}")
    rows = [("**Eval B playlist precision@10 (headline)**", lambda s: f"**{fmt(S[s]['eval_b']['precision@10'])}**"),
            ("Eval B chance", lambda s: f"{S[s]['eval_b']['chance']:.3f}"),
            ("Eval B condensed precision@10", lambda s: fmt(S[s]["eval_b"]["condensed_precision@10"])),
            ("Eval B judged tracks in top 50 (mean)", lambda s: f"{S[s]['eval_b']['judged_in_top50_mean']:.1f}"),
            ("Eval A Hit@10", lambda s: fmt(S[s]["eval_a"]["hit@10"])),
            ("Eval A NDCG@10", lambda s: fmt(S[s]["eval_a"]["ndcg@10"])),
            ("Eval A random-within-match expectation", lambda s: f"{S[s]['eval_a']['random_within_match_expectation']:.3f}"),
            ("Eval A identical-description ceiling", lambda s: f"{S[s]['eval_a']['ceiling_identical_desc']:.3f}"),
            ("Gold tag-match@10", lambda s: fmt(S[s]["gold"]["tag_match@10"])),
            ("Gold avoid-violation@10 (mean of #26-28)", lambda s: f"{S[s]['gold']['avoid_violation@10']['mean']:.3f} (" +
             ", ".join(f"#{k}: {v:.1f}" for k, v in S[s]['gold']['avoid_violation@10']['per_query'].items()) + ")"),
            ("Robustness", lambda s: "n/a"),
            ("Valid-item rate", lambda s: f"{S[s]['valid_item_rate']:.3f}"),
            ("Latency p50 / p90 (ms, batch 1)", lat),
            ("End-to-end latency incl. explanation", lambda s: "n/a")]
    L = ["# Final evaluation (run once)", "",
         f"Pre-registered in `eval/PLAN.md` note 4 (commit 56be084) before any of these numbers existed. Model: "
         f"`{res['checkpoint']}`; SID decoding: {res['decoding']}; hybrid: {res['hybrid']}. FAISS is the frozen "
         "baseline (recomputed; identical to `eval/results_baseline.json`). Means with bootstrap 95% CI over queries "
         "(1,000 resamples, seed 13).", "",
         "| Metric | FAISS baseline | SID model | Hybrid (RRF) |", "|---|---|---|---|"]
    L += [f"| {name} | " + " | ".join(f(s) for s in SYSTEMS) + " |" for name, f in rows]
    pd_ = res["additional_analysis_paired_eval_b"]
    L += ["", "## Additional analysis: paired Eval B differences (not a pre-registered primary metric)", "",
          "| Comparison | mean difference in precision@10 [95% CI] | queries better / worse / tied |", "|---|---|---|"]
    for k, lab in [("sid_minus_faiss", "SID model − FAISS"), ("hybrid_minus_faiss", "Hybrid − FAISS")]:
        v = pd_[k]
        L.append(f"| {lab} | {v['mean_diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                 f"{v['queries_better']} / {v['queries_worse']} / {v['ties']} |")
    L += ["", "## Eval B precision@10 per context word", "",
          "| Context word | chance | FAISS | SID model | Hybrid |", "|---|---|---|---|---|"]
    for w in S["faiss"]["eval_b"]["per_context_word"]:
        c = S["faiss"]["eval_b"]["per_context_word"][w]["chance"]
        L.append(f"| {w} | {c:.3f} | " + " | ".join(
            fmt(S[s]["eval_b"]["per_context_word"][w]["precision@10"]) for s in SYSTEMS) + " |")
    L += ["", "## Sanity view (gold queries, top 5)", ""]
    for s in ("sid", "hybrid"):
        L += [f"### {'SID model' if s == 'sid' else 'Hybrid'}", ""]
        for num, (q, items) in sanity[s].items():
            L += [f"**#{num} \"{q}\"**", ""] + [f"1. {n} — {a} — _{d}_" for n, a, d in items] + [""]
    return "\n".join(L)


def main() -> None:
    cat = B.load_catalog()
    sids = pd.read_parquet(SIDS)
    labels = pd.read_parquet(CONTEXT_LABELS)
    tune_ids = set(labels.track_id[labels.label_split == "tune"])
    model, tok = load_sid_model()
    sys_ = Systems(cat, sids, model, tok, tune_ids)
    res, extra = evaluate(cat, sys_)
    check_faiss_reproduces(res)
    res["latency_ms"] = latency(sys_, cat)
    by_id = cat.set_index("track_id")
    nums = {num: g["query"] for num, g in extra["gold"]}
    idx = {num: i for i, (num, _) in enumerate(extra["gold"])}
    sanity = {s: {n: (nums[n], [(by_id.loc[t, "track_name"], by_id.loc[t, "artists"].replace(";", ", "),
                                 by_id.loc[t, "content_desc"]) for t in extra["tg"][s][idx[n]][:5]])
                  for n in B.SANITY_GOLD} for s in ("sid", "hybrid")}
    res["sanity_top5"] = {s: {str(n): [x[0] + " — " + x[1] for x in v[1]] for n, v in d.items()} for s, d in sanity.items()}
    frozen = json.loads(B.RESULTS_JSON.read_text(encoding="utf-8"))
    RESULTS_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    RESULTS_MD.write_text(markdown(res, frozen["latency_ms"], sanity) + "\n", encoding="utf-8")
    print(markdown(res, frozen["latency_ms"], sanity))


if __name__ == "__main__":
    sys.exit(main())
