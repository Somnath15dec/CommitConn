import json

import pandas as pd
import pytest

from src import cluster_names as C


def test_samples_distinct_first_and_padded():
    s = pd.Series(["x"] * 20 + ["y", "z"])
    out = C.samples(s, n=10, seed=1)
    assert len(out) == 10 and set(out[:3]) == {"x", "y", "z"}
    assert C.samples(pd.Series([f"d{i}" for i in range(30)]), n=10, seed=1) == \
        C.samples(pd.Series([f"d{i}" for i in range(30)]), n=10, seed=1)
    assert len(set(C.samples(pd.Series([f"d{i}" for i in range(30)]), n=10, seed=1))) == 10


def test_valid_name_and_clean():
    assert C.valid_name("Mellow Acoustic Mornings") and not C.valid_name("Chill") and not C.valid_name("a b c d e")
    assert C.clean(' "Late Night  Jazz." ') == "Late Night Jazz"


def test_run_retries_once_and_is_resumable(tmp_path):
    calls = []

    def fake(lines):
        calls.append(1)
        return "Chill" if len(calls) == 1 else "Calm Piano Evenings"

    out = tmp_path / "names.json"
    names = C.run(fake, out)
    assert len(names) == 256 and names["0"]["name"] == "Calm Piano Evenings" and names["0"]["valid"]
    n_calls = len(calls)
    C.run(fake, out)
    assert len(calls) == n_calls                                   # nothing redone
    assert all(len(v["samples"]) == C.N_SAMPLES or v["n_tracks"] < C.N_SAMPLES for v in names.values())


def test_cluster_names_file_if_present():
    if not C.CLUSTER_NAMES.exists():
        pytest.skip("names not generated yet")
    names = json.loads(C.CLUSTER_NAMES.read_text(encoding="utf-8"))
    sids = pd.read_parquet(C.SIDS)
    assert set(names) == {str(a) for a in sids.a.unique()}
    assert sum(v["n_tracks"] for v in names.values()) == len(sids)
    assert all(C.valid_name(v["name"]) for v in names.values() if v["valid"])
