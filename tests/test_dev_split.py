import pandas as pd

from src import dev_split as S
from src import train_data as D
from src.paraphrase import normalize


def test_dev_tracks_frozen_and_sized():
    tune = S.tune_labels()
    dev = S.dev_tracks()
    assert dev == set(S.select_dev_tracks(tune))                          # file matches seed-21 selection
    assert len(dev) == round(S.DEV_FRAC * tune.track_id.nunique())
    assert dev <= set(tune.track_id)


def test_dev_tracks_have_no_human_training_pairs():
    train, val = D.frozen_split()
    dev = S.dev_tracks()
    for df in (train, val):
        assert not set(df.track_id[df.source == "tune_context"]) & dev
    assert not set(D.load_pool()[0].query(" source == 'tune_context' ").track_id) & dev


def test_dev_pool_excludes_training_tune_tracks():
    pool, train_tune = S.dev_pool(), S.training_tune_tracks()
    assert not pool & train_tune and S.dev_tracks() <= pool
    cat = set(pd.read_parquet(S.CATALOG, columns=["track_id"]).track_id)
    assert pool | train_tune == cat


def test_dev_queries_are_train_templates_not_eval_strings():
    qs = S.dev_queries()
    assert {q["template_id"] for q in qs} <= set(S.CONTEXT_TEMPLATE_IDS)
    assert not {normalize(q["query"]) for q in qs} & D.eval_strings()
    per = pd.Series([q["context_word"] for q in qs]).value_counts()
    assert per.max() <= S.DEV_QUERIES_PER_WORD and set(per.index) == set(S.eval_b_words())


def test_dev_relevance_uses_dev_tune_labels_only():
    rel = S.relevant_by_word()
    lab = pd.read_parquet(S.CONTEXT_LABELS)
    eval_half = set(lab.track_id[lab.label_split == "eval"])
    allrel = set().union(*rel.values())
    assert allrel <= S.dev_tracks() and not allrel & eval_half


def test_dev_precision_scoring():
    rel = S.relevant_by_word()
    w = "party"
    q = [d for d in S.dev_queries() if d["context_word"] == w][:1]
    perfect = sorted(rel[w])[:10]
    r = S.dev_precision(lambda qs: [perfect for _ in qs], q)
    assert r["dev_precision@10"] == 1.0 and r["n_queries"] == 1


def test_frozen_pool_matches_rebuild():
    train, val = D.frozen_split()
    t2, v2 = D.split_val(D.load_pool()[0])
    key = lambda df: set(zip(df["query"], df.track_id))
    assert key(train) == key(t2) and key(val) == key(v2) and len(val) == D.VAL_N
