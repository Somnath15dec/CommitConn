"""Step H: discovery dial (eval/PLAN.md note 5, pre-registered 919201a).

FirstTokenBias adds lambda to the scores of the level-1 SID tokens <a_k>, k in H (the user's history level-1 codes),
at the FIRST decode step only. It composes with the trie constraint: trie-disallowed tokens are -inf, and
-inf + lambda stays -inf whichever processor runs first. lambda > 0 familiar, < 0 non-habitual, 0 neutral.

    python -m src.dial      # run the pre-registered evaluation once -> eval/results_dial.json / .md
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import LogitsProcessor

from src.baseline import bootstrap_ci
from src.cluster_names import CLUSTER_NAMES
from src.decode import SidDecoder, sid_token
from src.gen_queries import CONTEXT_PHRASES, TRAIN_TEMPLATES
from src.paths import CATALOG, REAL_USERS, ROOT
from src.train_data import SIDS

LAMBDAS = (-4, -2, 0, 2, 4)
N_USERS, USER_SEED, MIN_HISTORY = 200, 22, 30
QUERIES_PER_USER, QUERY_SEED = 2, 23
BEAMS, K = 50, 10
DECODE_BATCH = 2
RESULTS_JSON = ROOT / "eval" / "results_dial.json"
RESULTS_MD = ROOT / "eval" / "results_dial.md"
CONTEXT_TEMPLATE_IDS = sorted(t for t, (typ, _) in TRAIN_TEMPLATES.items() if typ == "context")
N_EXAMPLE_USERS = 2


class FirstTokenBias(LogitsProcessor):
    """Add `lam` to the given token ids at the first generated position (input length == prompt_len)."""

    def __init__(self, prompt_len: int, token_ids: list[int], lam: float):
        self.prompt_len, self.ids, self.lam = prompt_len, torch.tensor(sorted(token_ids), dtype=torch.long), lam

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.lam == 0 or input_ids.shape[1] != self.prompt_len or len(self.ids) == 0:
            return scores
        scores = scores.clone()
        scores[:, self.ids.to(scores.device)] += self.lam
        return scores


def dial_fn(tok, level1_codes: set[int], lam: float):
    """logits_processor_fn for SidDecoder.search."""
    ids = [tok.convert_tokens_to_ids(sid_token("a", int(k))) for k in level1_codes]
    return lambda plen: FirstTokenBias(plen, ids, lam)


# ---------------------------------------------------------------- users / queries
def select_users(users: pd.DataFrame, cat: pd.DataFrame, n: int = N_USERS, seed: int = USER_SEED) -> list[dict]:
    ctx = cat.set_index("track_id").context
    hist = users[users.split == "history"].groupby("user_id").track_id.agg(list)
    held = users[users.split == "heldout"].groupby("user_id").track_id.agg(list)
    elig = sorted(u for u, h in hist.items() if len(h) >= MIN_HISTORY)
    picked = np.random.default_rng(seed).choice(elig, n, replace=False)
    return [{"user_id": u, "history": hist[u], "heldout": held.get(u, []),
             "context_words": sorted({w for t in hist[u] for w in ctx[t]})} for u in sorted(picked)]


def user_queries(users: list[dict], seed: int = QUERY_SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for u in users:
        words = rng.choice(u["context_words"], QUERIES_PER_USER, replace=False)
        for w in words:
            tid = CONTEXT_TEMPLATE_IDS[rng.integers(len(CONTEXT_TEMPLATE_IDS))]
            out.append({"user_id": u["user_id"], "context_word": str(w), "template_id": tid,
                        "query": TRAIN_TEMPLATES[tid][1].format(context=CONTEXT_PHRASES[str(w)])})
    return out


# ---------------------------------------------------------------- metrics
def novelty(top: list[str], level1: dict[str, int], H: set[int]) -> float:
    return sum(level1[t] not in H for t in top[:K]) / K


def familiar_hit(top: list[str], heldout: set[str]) -> float:
    return sum(t in heldout for t in top[:K]) / K


def relevance(top: list[str], ctx: dict[str, list[str]], word: str) -> float:
    return sum(word in ctx[t] for t in top[:K]) / K


def summary(v) -> dict:
    return {"mean": float(np.mean(v)), "ci95": bootstrap_ci(v), "n": len(v)}


def criteria(table: dict) -> dict:
    nov = [table[l]["novelty@10"]["mean"] for l in (4, 2, 0, -2, -4)]
    rel0, relm4 = table[0]["relevance@10"]["mean"], table[-4]["relevance@10"]["mean"]
    fam = {l: table[l]["familiar_hit@10"]["mean"] for l in LAMBDAS}
    return {"novelty_strictly_increases_+4_to_-4": bool(all(a < b for a, b in zip(nov, nov[1:]))),
            "relevance_at_-4_ge_70pct_of_0": bool(relm4 >= 0.7 * rel0), "relevance_ratio_-4_vs_0": relm4 / rel0 if rel0 else None,
            "familiar_hit_highest_at_+4": bool(fam[4] == max(fam.values()) and list(fam.values()).count(fam[4]) == 1)}


# ---------------------------------------------------------------- run
def run() -> dict:
    from src.evaluate_final import load_sid_model
    cat = pd.read_parquet(CATALOG)
    sids = pd.read_parquet(SIDS)
    level1 = dict(zip(sids.track_id, sids.a.astype(int)))
    ctx = dict(zip(cat.track_id, cat.context))
    users = select_users(pd.read_parquet(REAL_USERS), cat)
    by_user = {u["user_id"]: u for u in users}
    qs = user_queries(users)
    model, tok = load_sid_model()
    rows, tops = [], {}
    t0 = time.time()
    for u in users:
        H = {level1[t] for t in u["history"]}
        dec = SidDecoder(model, tok, sids, exclude=set(u["history"]))
        uq = [q for q in qs if q["user_id"] == u["user_id"]]
        for lam in LAMBDAS:
            res = dec.search([q["query"] for q in uq], n=K, num_beams=BEAMS, batch_size=DECODE_BATCH,
                             logits_processor_fn=dial_fn(tok, H, lam))
            for q, r in zip(uq, res):
                top = [t for t, _ in r]
                tops[(u["user_id"], q["query"], lam)] = top
                rows.append({"user_id": u["user_id"], "query": q["query"], "context_word": q["context_word"],
                             "lambda": lam, "novelty@10": novelty(top, level1, H),
                             "familiar_hit@10": familiar_hit(top, set(u["heldout"])),
                             "relevance@10": relevance(top, ctx, q["context_word"]), "n_returned": len(top)})
    print(f"decoded {len(users)} users x {len(LAMBDAS)} settings in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    table = {int(l): {m: summary(g[m]) for m in ("novelty@10", "familiar_hit@10", "relevance@10")}
             for l, g in df.groupby("lambda")}
    for l in table:
        table[l]["returned_fewer_than_10"] = int((df[df["lambda"] == l].n_returned < K).sum())
    names = json.loads(CLUSTER_NAMES.read_text(encoding="utf-8"))
    meta = cat.set_index("track_id")
    rng = np.random.default_rng(USER_SEED + 1)
    examples = []
    for uid in rng.choice(sorted(by_user), N_EXAMPLE_USERS, replace=False):
        u = by_user[uid]
        H = {level1[t] for t in u["history"]}
        q = next(x for x in qs if x["user_id"] == uid)
        hist_clusters = pd.Series([level1[t] for t in u["history"]]).value_counts().head(3)
        ex = {"user_id": str(uid), "query": q["query"], "n_history": len(u["history"]),
              "top_history_clusters": [f"{names[str(k)]['name']} ({v})" for k, v in hist_clusters.items()],
              "results": {}}
        for lam in (4, 0, -4):
            ex["results"][str(lam)] = [{"track": f"{meta.at[t, 'track_name']} — {meta.at[t, 'artists'].replace(';', ', ')}",
                                        "cluster": names[str(level1[t])]["name"], "in_history_cluster": level1[t] in H}
                                       for t in tops[(uid, q["query"], lam)][:5]]
        examples.append(ex)
    res = {"note": "eval/PLAN.md note 5 (pre-registered 919201a); run once", "checkpoint": "checkpoints/run1/epoch3",
           "n_users": len(users), "n_queries": len(qs), "lambdas": list(LAMBDAS), "table": table,
           "criteria": criteria(table), "examples": examples, "per_query": rows}
    return res


def fmt(s):
    return f"{s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"


def markdown(res: dict) -> str:
    t, c = res["table"], res["criteria"]
    L = ["# Discovery dial (run once)", "",
         f"Pre-registered in `eval/PLAN.md` note 5 (commit 919201a). Model `{res['checkpoint']}`, trie-constrained beam "
         f"{BEAMS}, top {K}; the user's history tracks are excluded. {res['n_users']} real users, {res['n_queries']} "
         "queries. Means with bootstrap 95% CI over queries (1,000 resamples, seed 13).", "",
         "| λ | novelty@10 | familiar hit@10 | relevance@10 |", "|---|---|---|---|"]
    for l in sorted(t, reverse=True):
        L.append(f"| {l:+d} | {fmt(t[l]['novelty@10'])} | {fmt(t[l]['familiar_hit@10'])} | {fmt(t[l]['relevance@10'])} |")
    L += ["", "## Pre-registered success criteria", "",
          f"- Novelty strictly increases from λ = +4 to −4: **{'pass' if c['novelty_strictly_increases_+4_to_-4'] else 'FAIL'}**",
          f"- Relevance at λ = −4 ≥ 70% of relevance at λ = 0: **{'pass' if c['relevance_at_-4_ge_70pct_of_0'] else 'FAIL'}** "
          f"(ratio {c['relevance_ratio_-4_vs_0']:.2f})",
          f"- Familiar hit@10 is highest at λ = +4: **{'pass' if c['familiar_hit_highest_at_+4'] else 'FAIL'}**", "",
          "## Examples (same query, λ = +4 / 0 / −4, top 5)", ""]
    for ex in res["examples"]:
        L += [f"**User {ex['user_id']}** ({ex['n_history']} history tracks; top history clusters: "
              f"{', '.join(ex['top_history_clusters'])}). Query: \"{ex['query']}\"", ""]
        for lam, items in ex["results"].items():
            L.append(f"- λ = {int(lam):+d}:")
            L += [f"  {i}. {x['track']} — _{x['cluster']}_{' (history cluster)' if x['in_history_cluster'] else ''}"
                  for i, x in enumerate(items, 1)]
        L.append("")
    return "\n".join(L)


def main() -> None:
    res = run()
    res["table"] = {str(k): v for k, v in res["table"].items()}
    RESULTS_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    res["table"] = {int(k): v for k, v in res["table"].items()}
    RESULTS_MD.write_text(markdown(res) + "\n", encoding="utf-8")
    print(markdown(res))


if __name__ == "__main__":
    sys.exit(main())
