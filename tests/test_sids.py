import json

import numpy as np
import pandas as pd
import pytest

from src import sids as S
from src.paths import CATALOG


@pytest.fixture(scope="module")
def sids():
    return pd.read_parquet(S.SIDS)


@pytest.fixture(scope="module")
def trie():
    return json.loads(S.SID_TRIE.read_text(encoding="utf-8"))


def test_every_catalog_track_has_one_sid(sids):
    cat = pd.read_parquet(CATALOG)
    assert list(sids.columns) == ["track_id", "a", "b", "c", "d"]
    assert sids.track_id.is_unique and set(sids.track_id) == set(cat.track_id)


def test_full_sid_unique_and_in_range(sids):
    assert not sids.duplicated(["a", "b", "c", "d"]).any()
    for col in "abc":
        assert sids[col].between(0, S.SID_K - 1).all()
    assert sids.d.min() == 0


def test_dedupe_digit_is_dense_per_code(sids):
    g = sids.groupby(["a", "b", "c"]).d
    assert (g.max() + 1 == g.size()).all()          # d = 0..n-1 within each code


def test_trie_yields_exactly_catalog_sids(sids, trie):
    from_trie = sorted(S.trie_sids(trie))
    catalog = sorted(map(tuple, sids[["a", "b", "c", "d"]].to_numpy().tolist()))
    assert from_trie == catalog


def test_allowed_next_follows_trie(sids, trie):
    assert S.allowed_next(trie, ()) == sorted(sids.a.unique().tolist())
    r = sids.iloc[0]
    assert r.b in S.allowed_next(trie, (r.a,))
    assert r.c in S.allowed_next(trie, (r.a, r.b))
    assert r.d in S.allowed_next(trie, (r.a, r.b, r.c))
    with pytest.raises(KeyError):
        S.allowed_next(trie, (S.SID_K + 1,))


def test_checks_within_limits(sids):
    s = S.stats(sids)
    assert s["collision_rate"] <= S.MAX_COLLISION_RATE
    assert s["max_tracks_per_code"] <= S.MAX_TRACKS_PER_CODE


def test_dedupe_digit_orders_by_residual():
    codes = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [0, 0, 0]])
    d = S.dedupe_digit(["t0", "t1", "t2", "t3"], codes, np.array([0.5, 0.1, 0.5, 9.0]))
    assert d.tolist() == [1, 0, 2, 0]                 # closest first, ties by track_id


def test_residual_codes_deterministic():
    import torch
    x = torch.randn(300, 8, generator=torch.Generator().manual_seed(0))
    a, ra = S.residual_codes(x, levels=2, k=8)
    b, rb = S.residual_codes(x, levels=2, k=8)
    assert (a == b).all() and np.allclose(ra, rb) and a.shape == (300, 2)


def test_purity_and_baseline():
    groups = np.array([0, 0, 0, 1, 1])
    p = S.purity(np.array(["rock", "rock", "pop", "jazz", "jazz"]),
                 [{"gym"}, {"gym"}, set(), {"sleep"}, {"study"}], groups)
    assert p["genre"] == pytest.approx((2 / 3 + 1) / 2)
    assert p["context"] == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_sample_pairs_share_key():
    keys = np.array([0, 0, 1, 1, 1, 2])
    pairs = S.sample_pairs(keys, 50, np.random.default_rng(0))
    assert len(pairs) == 50 and (pairs[:, 0] != pairs[:, 1]).all()
    assert (keys[pairs[:, 0]] == keys[pairs[:, 1]]).all() and 5 not in pairs
