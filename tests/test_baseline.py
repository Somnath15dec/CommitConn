import json

import numpy as np
import pandas as pd
import pytest

from src import baseline as B


def test_precision_and_condensed():
    top = ["a", "b", "c", "d"]                       # fewer than 10 -> empty slots are misses
    assert B.precision_at_k(top, {"a", "c", "z"}) == pytest.approx(0.2)
    # judged: a, c, x ; relevant: c  -> judged order a, c, x ; m = 3 ; hits 1
    cp, nj = B.condensed_precision(["a", "b", "c", "d", "x"], {"c"}, {"a", "c", "x"})
    assert (cp, nj) == (pytest.approx(1 / 3), 3)
    assert B.condensed_precision(["b"], {"c"}, {"a"}) == (0.0, 0)
    top50 = [f"t{i}" for i in range(50)]
    cp, nj = B.condensed_precision(top50, {"t0"}, set(top50))
    assert (cp, nj) == (pytest.approx(0.1), 50)          # m capped at 10


def test_hit_ndcg():
    assert B.hit_ndcg(["x", "t"], "t") == (1.0, pytest.approx(1 / np.log2(3)))
    assert B.hit_ndcg(["t"], "t") == (1.0, 1.0)
    assert B.hit_ndcg(["x"] * 10 + ["t"], "t") == (0.0, 0.0)


def test_tag_match_and_avoid():
    T = {"a": {"calm", "study"}, "b": {"calm"}, "c": set()}
    assert B.tag_match(["a", "b", "c"], {"calm", "study"}, T) == pytest.approx((1 + 0.5 + 0) / 10)
    TG = {"a": {"rock"}, "b": set(), "c": {"metal"}}
    assert B.avoid_violation(["a", "b", "c"], {"metal", "rock"}, TG) == pytest.approx(2 / 3)
    assert B.avoid_violation([], {"metal"}, TG) == 0.0


def test_ceilings():
    cat = pd.DataFrame({"track_id": ["a", "b", "c"], "content_desc": ["x", "x", "y"],
                        "T": [{"calm"}, {"calm", "jazz"}, {"jazz"}]})
    evals = [{"track_id": "a", "wanted_tags": ["calm"]}, {"track_id": "c", "wanted_tags": ["calm", "jazz"]}]
    assert B.query_aware_ceiling(evals, cat) == 1.0          # N = 2 and 1 -> min(1, 10/N) = 1
    assert B.identical_desc_ceiling(evals, cat) == 1.0


def test_bootstrap_ci_deterministic_and_brackets_mean():
    v = np.random.default_rng(0).random(132)
    lo, hi = B.bootstrap_ci(v)
    assert lo < v.mean() < hi and B.bootstrap_ci(v) == [lo, hi]


class _FakeModel:
    def encode(self, texts, **_):
        rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
        v = np.array([[len(t), t.count("a") + 1, 1.0] for t in texts], dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_search_distinct_and_restricted():
    cat = pd.DataFrame({"track_id": [f"t{i}" for i in range(30)],
                        "content_desc": [("a" * i) + "b" * (30 - i) for i in range(30)]})
    s = B.FaissBaseline(cat, model=_FakeModel())
    top = s.search(["aaab"], 10)[0]
    assert len(top) == 10 and len(set(top)) == 10
    keep = {f"t{i}" for i in range(0, 30, 2)}
    assert set(s.search(["aaab"], 10, s.restrict(keep))[0]) <= keep


def test_results_file_if_present():
    if not B.RESULTS_JSON.exists():
        pytest.skip("baseline not run yet")
    r = json.loads(B.RESULTS_JSON.read_text(encoding="utf-8"))
    assert r["eval_b"]["n_queries"] == 132 and r["eval_a"]["n_queries"] == 500 and r["gold"]["n_accuracy"] == 30
    assert len(r["eval_b"]["per_query"]) == 132 and len(r["gold"]["per_query"]) == 30
    assert r["valid_item_rate"] == 1.0
    assert set(r["gold"]["avoid_violation@10"]["per_query"]) == {"26", "27", "28"}
