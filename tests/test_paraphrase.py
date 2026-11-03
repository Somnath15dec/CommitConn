import json

import pandas as pd
import pytest

from src import paraphrase as P
from src.paths import CATALOG

FEW_SHOT_MAX_JACCARD = 0.5


def _tokens(q):
    return set(P.normalize(q).split())


def test_normalize_and_name_words():
    assert P.normalize("  Café-Music, PLEASE! ") == "cafe music please"
    assert P.name_words("Rain on Me", "Lady Gaga;Ariana Grande") == {"rain", "lady", "gaga", "ariana", "grande"}


def test_filter_queries_order_counts_and_style_lengths():
    evals = {P.normalize("music for sleep please")}
    kept, c = P.filter_queries(
        [("casual", "play some lady stuff for me"), ("casual", "too short"), ("short", "gym rap"),
         ("short", "way too long for a search box"), ("situation", "Music for sleep, please!"),
         ("casual", "calm piano for late nights"), ("situation", "Calm piano for late nights"),
         ("casual", " ".join(["word"] * 21))],
        banned_names={"lady"}, eval_queries=evals, desc="Genre: piano. Sound: calm.")
    assert kept == [("short", "gym rap"), ("casual", "calm piano for late nights")]
    assert c == {"name_word": 1, "length": 3, "eval_match": 1, "duplicate": 1,
                 **dict.fromkeys(P.RULE_REASONS, 0)}


def test_rule_filter_reasons():
    hy = {"new-age"}
    kept, c = P.filter_queries(
        [("casual", "any neutral mood stuff for tonight"), ("casual", "something from the chill category"),
         ("short", "new-age for sleeping"), ("casual", "find me calm tunes now"),
         ("situation", "sipping coffee on a slow morning"), ("casual", "focus music for study time"),
         ("casual", "soft piano for a quiet night"), ("short", "soft guitar evenings")],
        banned_names=set(), eval_queries=set(), desc="Genre: guitar. Sound: calm.", hyphen_genres=hy)
    assert kept == [("casual", "focus music for study time"), ("short", "soft guitar evenings")]
    assert {k: c[k] for k in P.RULE_REASONS} == {"rule_banned_term": 3, "rule_first_word": 1,
                                                 "rule_multi_context": 1, "rule_instrument": 1}


def test_prompt_desc():
    assert P.prompt_desc("Spotify category: romance. Sound: slow, moderate energy, neutral mood. "
                         "Good for: sleep.") == "Sound: slow, romantic. Good for: sleep."
    assert P.prompt_desc("Genre: r-n-b. Spotify category: party. Sound: neutral mood. Good for: general listening."
                         ) == "Genre: r&b. Good for: party."
    cat = pd.read_parquet(CATALOG)
    from src.analyze_readiness import content_desc
    for d in cat.head(300).apply(content_desc, axis=1).map(P.prompt_desc):
        assert not any(w in d for w in ("neutral mood", "moderate energy", "category"))
        assert not any(g in d for g in P.hyphenated_genre_ids(cat))


def test_humanize_desc_only_touches_genre_sentence():
    d = "Genre: new-age, r-n-b, hip-hop. Sound: calm, neutral mood. Good for: feel-good."
    assert P.humanize_desc(d) == "Genre: new age, r&b, hip-hop. Sound: calm, neutral mood. Good for: feel-good."
    assert P.humanize_desc("Sound: calm. Good for: sleep.") == "Sound: calm. Good for: sleep."


def test_banned_terms():
    hy = {"new-age", "honky-tonk"}
    assert P.banned_terms("Play some new-age", hy) == ["first word 'play'", "new-age"]
    assert P.banned_terms("neutral mood honky tonk", hy) == ["neutral mood"]
    assert P.banned_terms("any chill hip-hop tonight?", hy) == []


def test_few_shot_examples_follow_rules_and_differ_from_eval_queries():
    evals = P.eval_queries_raw()
    hy = P.hyphenated_genre_ids(pd.read_parquet(CATALOG))
    for ex in P.FEW_SHOT:
        for style in P.STYLES:
            q = ex[style]
            lo, hi = P.LENGTH_BOUNDS[style]
            assert lo <= len(q.split()) <= hi and not P.banned_terms(q, hy), q
            assert P.rule_violation(q, ex["desc"], hy) is None, q
            assert P.normalize(q) not in {P.normalize(e) for e in evals}, q
            worst = max(len(_tokens(q) & _tokens(e)) / len(_tokens(q) | _tokens(e)) for e in evals)
            assert worst < FEW_SHOT_MAX_JACCARD, (q, worst)


def test_subset_priority_first_and_size():
    cat = pd.read_parquet(CATALOG)
    sub = P.select_subset(cat)
    assert len(sub) == P.SUBSET_N and sub.track_id.is_unique
    pri = sub.context.map(lambda c: bool(set(c) & set(P.PRIORITY_CONTEXTS))).to_numpy()
    n_pri = int(pri.sum())
    assert pri[:n_pri].all() and not pri[n_pri:].any()
    assert sub.track_id.tolist() == P.select_subset(cat).track_id.tolist()      # deterministic


def _items(n):
    return {"items": [{"id": i, "short": f"calm tune {i}", "situation": f"walking home late number {i}",
                       "casual": f"any mellow stuff number {i}?"} for i in range(1, n + 1)]}


def test_parse_response_validates():
    assert [s for s, _ in P.parse_response(json.dumps(_items(2)), 2)[1]] == list(P.STYLES)
    with pytest.raises(ValueError):
        P.parse_response(json.dumps(_items(1)), 2)
    bad = _items(1); del bad["items"][0]["casual"]
    with pytest.raises(ValueError):
        P.parse_response(json.dumps(bad), 1)


def test_prompt_has_content_desc_only():
    row = pd.read_parquet(CATALOG).iloc[0]
    from src.analyze_readiness import content_desc
    prompt = P.build_prompt([content_desc(row)])
    assert row.track_name not in prompt and row.track_id not in prompt


def test_run_is_resumable_and_retries(tmp_path):
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"                                   # first attempt fails -> retried
        data = prompt.split("per line.")[-1]
        n = sum(1 for line in data.splitlines() if line.split(":")[0].isdigit())
        return json.dumps(_items(n))

    pairs, batches = tmp_path / "pairs.jsonl", tmp_path / "batches.jsonl"
    logs = P.run(max_batches=2, pairs_path=pairs, batches_path=batches, llm=fake_llm)
    assert [b["status"] for b in logs] == ["ok", "ok"]
    first = {t for b in logs for t in b["track_ids"]}
    logs2 = P.run(max_batches=1, pairs_path=pairs, batches_path=batches, llm=fake_llm)
    assert not set(logs2[0]["track_ids"]) & first                # done tracks skipped
    rows = P.read_jsonl(pairs)
    assert len(P.read_jsonl(batches)) == 3 and rows and {r["style"] for r in rows} <= set(P.STYLES)
