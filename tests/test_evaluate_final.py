import json

import pytest

from src import evaluate_final as F


def test_rrf_scores_and_ties():
    # a: rank1 in both lists -> highest. b (1st in list 2 only) vs c (1st in list 1 only): tie -> best rank 1 both -> id
    assert F.rrf([["a", "c"], ["a", "b"]]) == ["a", "b", "c"]
    out = F.rrf([["x", "y", "z"], ["z", "y", "x"]])
    # x and z: 1/61 + 1/63 each; y: 2/62 (smaller); x,z tie -> best rank 1 -> track_id order
    assert out == ["x", "z", "y"]
    assert F.rrf([["a"], []]) == ["a"]
    assert len(F.rrf([[f"s{i}" for i in range(50)], [f"f{i}" for i in range(50)]])) == F.TOP_N


def test_rrf_formula():
    k = F.RRF_K
    s_a = 1 / (k + 1) + 1 / (k + 3)
    s_b = 1 / (k + 2) + 1 / (k + 1)
    assert s_b > s_a and F.rrf([["a", "b"], ["b", "x", "a"]])[0] == "b"


def test_paired_diff():
    d = F.paired_diff([0.5, 0.2, 0.1], [0.1, 0.2, 0.3])
    assert d["mean_diff"] == pytest.approx((0.4 + 0 - 0.2) / 3)
    assert (d["queries_better"], d["queries_worse"], d["ties"]) == (1, 1, 1)


def test_results_file_if_present():
    if not F.RESULTS_JSON.exists():
        pytest.skip("final evaluation not run yet")
    r = json.loads(F.RESULTS_JSON.read_text(encoding="utf-8"))
    for s in F.SYSTEMS:
        assert r["systems"][s]["eval_b"]["n_queries"] == 132 and r["systems"][s]["valid_item_rate"] == 1.0
    assert r["checkpoint"].replace("\\", "/") == "checkpoints/run1/epoch3"
