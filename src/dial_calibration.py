"""Dial calibration (eval/PLAN.md note 6, pre-registered 52d788b): descriptive, no pass/fail.

Same users, queries, decoding and metrics as the discovery-dial run (ad4fb22) via src/dial.run, with
lambda in {-1, -0.5, +0.5, +1}. The lambda = 0 and +-2 rows of ad4fb22 are shown for reference, labelled as such.

    python -m src.dial_calibration   -> eval/results_dial_calibration.json / .md
"""
from __future__ import annotations

import json
import sys

from src import dial as Dl
from src.paths import ROOT

LAMBDAS = (-1.0, -0.5, 0.5, 1.0)
REFERENCE = (2, 0, -2)
RESULTS_JSON = ROOT / "eval" / "results_dial_calibration.json"
RESULTS_MD = ROOT / "eval" / "results_dial_calibration.md"


def markdown(res: dict, ref: dict) -> str:
    rows = [(float(l), v, "calibration") for l, v in res["table"].items()]
    rows += [(float(l), ref["table"][str(l)], "reference: ad4fb22") for l in REFERENCE]
    L = ["# Dial calibration (descriptive)", "",
         "Pre-registered in `eval/PLAN.md` note 6 (commit 52d788b); no pass/fail criteria. Same model, 200 users, "
         "400 queries, beam 50, top 10, history tracks excluded, as the discovery-dial run (ad4fb22). Means with "
         "bootstrap 95% CI over queries.", "",
         "| λ | novelty@10 | familiar hit@10 | relevance@10 | source |", "|---|---|---|---|---|"]
    for l, v, src in sorted(rows, key=lambda r: -r[0]):
        L.append(f"| {l:+.1f} | {Dl.fmt(v['novelty@10'])} | {Dl.fmt(v['familiar_hit@10'])} | "
                 f"{Dl.fmt(v['relevance@10'])} | {src} |")
    L += ["", "## Examples (same query, λ = +1 / +0.5 / −0.5 / −1, top 5)", ""]
    for ex in res["examples"]:
        L += [f"**User {ex['user_id']}** ({ex['n_history']} history tracks; top history clusters: "
              f"{', '.join(ex['top_history_clusters'])}). Query: \"{ex['query']}\"", ""]
        for lam, items in ex["results"].items():
            L.append(f"- λ = {float(lam):+.1f}:")
            L += [f"  {i}. {x['track']} — _{x['cluster']}_{' (history cluster)' if x['in_history_cluster'] else ''}"
                  for i, x in enumerate(items, 1)]
        L.append("")
    return "\n".join(L)


def main() -> None:
    res = Dl.run(LAMBDAS, example_lambdas=(1.0, 0.5, -0.5, -1.0))
    res["note"] = "eval/PLAN.md note 6 (pre-registered 52d788b); descriptive, no pass/fail"
    res["table"] = {str(float(k)): v for k, v in res["table"].items()}
    RESULTS_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    ref = json.loads(Dl.RESULTS_JSON.read_text(encoding="utf-8"))
    RESULTS_MD.write_text(markdown(res, ref) + "\n", encoding="utf-8")
    print(markdown(res, ref))


if __name__ == "__main__":
    sys.exit(main())
