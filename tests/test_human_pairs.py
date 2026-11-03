from collections import Counter

import pandas as pd
import pytest

from src import human_pairs as H
from src.config import SYNONYM_TO_CANONICAL
from src.gen_queries import TRAIN_TEMPLATES, read_jsonl
from src.paraphrase import normalize
from src.paths import CATALOG, CONTEXT_LABELS


@pytest.fixture(scope="module")
def pairs():
    return read_jsonl(H.HUMAN_PAIRS)


@pytest.fixture(scope="module")
def labels():
    return pd.read_parquet(CONTEXT_LABELS)


def test_no_pair_targets_eval_half_track(pairs, labels):
    eval_ids = set(labels.track_id[labels.label_split == "eval"])
    tune_ids = set(labels.track_id[labels.label_split == "tune"])
    targets = {p["track_id"] for p in pairs}
    assert not targets & eval_ids
    assert targets <= tune_ids


def test_targets_in_catalog(pairs):
    assert {p["track_id"] for p in pairs} <= set(pd.read_parquet(CATALOG).track_id)


def test_no_query_matches_eval_or_gold(pairs):
    evq = H.eval_queries_normalized()
    assert len(evq) > 300                     # distinct normalized Eval A + Eval B + gold strings were loaded
    assert not {normalize(p["query"]) for p in pairs} & evq


def test_fields_templates_and_cap(pairs):
    assert {p["source"] for p in pairs} == {"tune_context"}
    allowed = set(H.CONTEXT_TEMPLATE_IDS) | {H.COMBO_GENRE_TEMPLATE, H.COMBO_MOOD_TEMPLATE}
    assert {p["template_id"] for p in pairs} <= allowed
    assert all(TRAIN_TEMPLATES[p["template_id"]][0] == p["type"] for p in pairs)
    per = Counter((p["track_id"], p["context_word"]) for p in pairs)
    assert max(per.values()) <= H.PAIRS_PER_TRACK_CONTEXT
    per_type = Counter((p["track_id"], p["context_word"], p["type"]) for p in pairs)
    assert all(v == 1 for k, v in per_type.items() if k[2] == "combo")               # one combo per unit
    assert not any(p["context_word"] in SYNONYM_TO_CANONICAL for p in pairs)       # canonical only
    assert all(p["wanted_tags"][-1] == p["context_word"] for p in pairs)
    assert all(p["wanted_tags"] == [p["context_word"]] for p in pairs if p["type"] == "context")
    assert len({(p["query"], p["track_id"]) for p in pairs}) == len(pairs)          # distinct per track


def test_context_backed_by_tune_label(pairs, labels):
    pmap = H.playlist_word_map()
    tune = labels[labels.label_split == "tune"].groupby("track_id").context_word.agg(set)
    for p in pairs[:: max(1, len(pairs) // 2000)]:
        assert any(p["context_word"] in pmap.get(w, []) for w in tune[p["track_id"]])


def test_combo_slots_from_own_genres_and_mood_tags(pairs):
    cat = pd.read_parquet(CATALOG).set_index("track_id")
    for p in [q for q in pairs if q["type"] == "combo"][::50]:
        tag = p["wanted_tags"][0]
        row = cat.loc[p["track_id"]]
        if p["template_id"] == H.COMBO_GENRE_TEMPLATE:
            assert tag in set(row.genres) and "-" not in p["query"].replace("hip-hop", "")
        else:
            assert tag in set(row.tags) and tag in {"calm", "energetic", "happy", "melancholic"}


def test_playlist_word_map_eval_b_inverse_plus_overrides():
    from src.prep import CONTEXT_TO_PLAYLIST_WORDS
    m = H.playlist_word_map()
    assert m["party"] == ["party"] and m["focus"] == ["study"]
    for pw, words in m.items():
        if pw not in H.PLAYLIST_WORD_OVERRIDES:
            assert all(pw in CONTEXT_TO_PLAYLIST_WORDS[w] for w in words)
    assert not {"sad", "late night", "drive", "night drive"} & set(m)


def test_pairs_for_deterministic_and_shapes():
    a = H.pairs_for("t1", "sleep", ["new-age"], ["calm", "neutral mood"])
    assert a == H.pairs_for("t1", "sleep", ["new-age"], ["calm", "neutral mood"])
    assert [p["type"] for p in a] == ["context", "combo"]
    assert a[1]["query"] in {"new age for sleep", "new age for winding down", "calm songs for sleep",
                             "calm songs for winding down"}
    b = H.pairs_for("t2", "sleep", [], ["neutral mood", "moderate energy"])           # no genre, no mood tag
    assert [p["type"] for p in b] == ["context", "context"] and len({p["query"] for p in b}) == 2
