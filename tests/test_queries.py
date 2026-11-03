import json
import re

import pandas as pd
import pytest

from src import gen_queries as gq
from src.paths import CATALOG, EVAL_PAIRS, TRAIN_PAIRS


def _load(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def cat():
    return pd.read_parquet(CATALOG).set_index("track_id", drop=False)


@pytest.fixture(scope="module")
def train():
    return _load(TRAIN_PAIRS)


@pytest.fixture(scope="module")
def evals():
    return _load(EVAL_PAIRS)


def _skeleton(text):
    return re.sub(r"\{\w+\}", "{}", text).lower()


def test_no_template_overlap():
    tr = {t for _, t in gq.TRAIN_TEMPLATES.values()}
    ev = {t for _, t in gq.EVAL_TEMPLATES.values()}
    assert not tr & ev
    assert not {_skeleton(t) for t in tr} & {_skeleton(t) for t in ev}
    assert not set(gq.TRAIN_TEMPLATES) & set(gq.EVAL_TEMPLATES)


def test_track_ids_exist(cat, train, evals):
    ids = set(cat.track_id)
    assert all(p["track_id"] in ids for p in train + evals)


def test_eval_pairs_not_in_train(train, evals):
    keys = {(p["query"], p["track_id"]) for p in train}
    assert not any((p["query"], p["track_id"]) in keys for p in evals)


def test_eval_uses_eval_templates_only(evals, train):
    assert all(p["template_id"] in gq.EVAL_TEMPLATES for p in evals)
    assert all(p["template_id"] in gq.TRAIN_TEMPLATES for p in train)


def test_eval_no_category_or_primary_genre_only(cat, evals):
    for p in evals:
        row = cat.loc[p["track_id"]]
        for word in gq.eval_banned_words(row):
            assert not gq.contains_word(p["query"], word), (p, word)
            assert word not in p["wanted_tags"], (p, word)


def test_eval_slots_from_own_tags_context_genres(cat, evals):
    for p in evals:
        row = cat.loc[p["track_id"]]
        own = set(row.tags) | set(row.context) | set(row.genres)
        assert set(p["wanted_tags"]) <= own, p


def test_train_slots_from_own_fields(cat, train):
    for p in train:
        row = cat.loc[p["track_id"]]
        own = set(row.tags) | set(row.context) | set(row.genres) | set(row.category)
        assert set(p["wanted_tags"]) <= own, p


def test_wanted_tags_canonical(train, evals):
    from src.config import SYNONYM_TO_CANONICAL
    assert not any(set(p["wanted_tags"]) & set(SYNONYM_TO_CANONICAL) for p in train + evals)


def test_train_cap_and_sizes(train, evals):
    per_track = pd.Series([p["track_id"] for p in train]).value_counts()
    assert per_track.max() <= gq.TRAIN_QUERIES_PER_TRACK
    assert len(evals) == gq.EVAL_N_TRACKS
    assert len({p["track_id"] for p in evals}) == gq.EVAL_N_TRACKS


def test_genre_slot_never_from_primary_genre_or_category():
    row = pd.Series({"track_id": "x", "tags": ["happy"], "context": [], "genres": [],
                     "category": ["study"], "primary_genre": "study"})
    row = type("R", (), row.to_dict())
    for split in ("train", "eval"):
        assert gq.slot_options(row, split)["genre"] == []
    assert gq.slot_options(row, "eval")["context"] == []          # category never in eval
    assert ("studying", "study") in gq.slot_options(row, "train")["context"]


def test_synonym_surface_maps_to_canonical():
    row = type("R", (), {"track_id": "x", "tags": [], "context": ["sleep"], "genres": [],
                         "category": [], "primary_genre": "pop"})
    opts = gq.slot_options(row, "eval")["context"]
    assert ("winding down", "sleep") in opts and all(c == "sleep" for _, c in opts)


# ---------- genre humanizing ----------
def test_humanize_genre():
    assert gq.humanize_genre("hard-rock") == "hard rock"
    assert gq.humanize_genre("r-n-b") == "r&b"
    assert gq.humanize_genre("rock-n-roll") == "rock n roll"
    for g in ("hip-hop", "k-pop", "j-pop", "j-rock", "j-dance", "j-idol"):
        assert gq.humanize_genre(g) == g


def test_queries_use_humanized_genres(cat, train, evals):
    raw = {g for gs in cat.genres for g in gs if gq.humanize_genre(g) != g}
    assert "hard-rock" in raw and "r-n-b" in raw
    for p in train + evals:
        assert not any(gq.contains_word(p["query"], g) for g in raw), p
    assert any("hard rock" in p["query"] for p in train)
    assert any("r&b" in p["query"] for p in train)


# ---------- Eval B ----------
@pytest.fixture(scope="module")
def eval_b():
    from src.paths import EVAL_CONTEXT_QUERIES
    return _load(EVAL_CONTEXT_QUERIES)


def test_eval_b_not_in_train(train, eval_b):
    train_q = {p["query"] for p in train}
    assert not any(p["query"] in train_q for p in eval_b)


def test_eval_b_templates_disjoint_from_train():
    tr = {t for _, t in gq.TRAIN_TEMPLATES.values()}
    evb = set(gq.EVAL_B_TEMPLATES.values())
    assert not tr & evb
    assert not {_skeleton(t) for t in tr} & {_skeleton(t) for t in evb}
    assert all(gq.slots_of(t) == ["context"] for t in evb)


def test_eval_b_counts_and_fields(eval_b):
    from src.config import CONTEXT_SYNONYMS
    from src.prep import CONTEXT_TO_PLAYLIST_WORDS
    counts = pd.Series([p["context_word"] for p in eval_b]).value_counts()
    assert set(counts.index) == set(gq.eval_b_words())
    assert counts.between(10, 12).all()
    assert len({p["query"] for p in eval_b}) == len(eval_b)
    synonyms = {s for v in CONTEXT_SYNONYMS.values() for s in v}
    for p in eval_b:
        assert p["context_word"] not in synonyms
        assert p["playlist_words"] == CONTEXT_TO_PLAYLIST_WORDS[p["context_word"]]
        assert p["surface"] in [p["context_word"], *CONTEXT_SYNONYMS.get(p["context_word"], [])]



def test_incremental_train_keeps_previous_pairs(cat):
    sub = cat.head(30).reset_index(drop=True)
    fresh, _ = gq.generate_train(sub)
    prev = [p for p in fresh if p["track_id"] != sub.track_id[0]]      # pretend track 0 is new
    again, n_new = gq.generate_train(sub, prev)
    assert n_new == 1
    assert [p for p in again if p["track_id"] != sub.track_id[0]] == prev
    assert any(p["track_id"] == sub.track_id[0] for p in again)
    assert gq.generate_train(sub, prev) == (again, 1)                      # deterministic


def test_retained_tracks_have_valid_pairs(cat, train):
    by_track = {}
    for p in train:
        by_track.setdefault(p["track_id"], []).append(p)
    assert set(by_track) == set(cat.track_id)
    assert all(gq.train_pairs_valid(cat.loc[t], ps) for t, ps in by_track.items())


def test_repair_eval_a_replaces_only_invalid_pairs(cat, train, evals):
    kept, changes = gq.repair_eval_a(cat, evals, train)
    assert kept == evals and changes == []                      # current Eval A is fully valid
    broken = [dict(evals[0], track_id="not-a-track"), *evals[1:]]
    fixed, changes = gq.repair_eval_a(cat, broken, train)
    assert len(changes) == 1 and fixed[1:] == evals[1:]
    assert fixed[0]["track_id"] in cat.index and fixed[0]["track_id"] not in {p["track_id"] for p in evals[1:]}
