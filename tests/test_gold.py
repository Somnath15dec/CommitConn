import json

import pandas as pd
import pytest

from src.config import SYNONYM_TO_CANONICAL
from src.paths import CATALOG, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, ROOT, TRAIN_PAIRS

GOLD = ROOT / "eval" / "gold_queries.jsonl"


def _load(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def gold():
    return _load(GOLD)


@pytest.fixture(scope="module")
def vocab():
    cat = pd.read_parquet(CATALOG)
    tags = {t for ts in cat.tags for t in ts}
    genres = {g for gs in cat.genres for g in gs}
    contexts = {c for cs in cat.context for c in cs}
    return tags, genres, contexts


def test_counts(gold):
    acc = [g for g in gold if g["type"] == "accuracy"]
    rob = [g for g in gold if g["type"] == "robustness"]
    assert len(acc) == 30 and len(rob) == 8 and len(gold) == 38
    assert "something calm for a 3 a.m. study session" in {g["query"] for g in acc}
    subtypes = pd.Series([g["subtype"] for g in rob]).value_counts().to_dict()
    assert subtypes == {"empty": 1, "gibberish": 1, "non_music": 2, "prompt_injection": 2,
                        "extreme_length": 1, "odd_characters": 1}


def test_unique_queries(gold):
    assert len({g["query"] for g in gold}) == len(gold)


def test_not_in_train_or_eval(gold):
    other = {p["query"].lower() for path in (TRAIN_PAIRS, EVAL_PAIRS, EVAL_CONTEXT_QUERIES)
             for p in _load(path)}
    assert not any(g["query"].lower() in other for g in gold)


def test_canonical_vocab(gold, vocab):
    tags, genres, contexts = vocab
    for g in gold:
        if g["type"] != "accuracy":
            continue
        assert set(g["wanted_tags"]) | set(g["avoid_tags"]) <= tags | genres, g
        assert set(g["wanted_context"]) <= contexts, g
        assert not set(g["wanted_context"]) & set(SYNONYM_TO_CANONICAL), g
        assert g["wanted_tags"] or g["wanted_context"], g


def test_coverage(gold):
    acc = [g for g in gold if g["type"] == "accuracy"]
    ctx = {c for g in acc for c in g["wanted_context"]}
    for need in [{"study"}, {"sleep"}, {"workout", "gym"}, {"running"}, {"party", "dance"},
                 {"relaxing", "coffee"}, {"morning"}, {"rainy day"}]:
        assert ctx & need, need
    genres = {g for gs in pd.read_parquet(CATALOG).genres for g in gs}
    assert sum(bool(set(g["wanted_tags"]) & genres) and bool(g["wanted_context"]) for g in acc) >= 5
    assert sum(bool(g["avoid_tags"]) for g in acc) >= 3



ROBUSTNESS_EXPECTED = {"empty": "rephrase", "gibberish": "rephrase", "non_music": "redirect",
                       "prompt_injection": "refuse_or_ignore", "extreme_length": "truncate_and_answer",
                       "odd_characters": "answer_or_rephrase_no_crash"}


def test_robustness_expected(gold):
    for g in gold:
        if g["type"] == "robustness":
            assert g["expected"] == ROBUSTNESS_EXPECTED[g["subtype"]], g


def test_no_display_labels_in_queries(gold):
    assert not any("(negation)" in g["query"] or "(mixed)" in g["query"] for g in gold)
