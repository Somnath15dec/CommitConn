"""data/DATA.md row counts must match the files they describe."""
import json
import re

import pandas as pd

from src.paths import PROC, ROOT

CARD = ROOT / "data" / "DATA.md"


def rows(path):
    if path.suffix == ".parquet":
        return len(pd.read_parquet(path))
    if path.suffix == ".jsonl":
        return sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if path.suffix == ".csv":
        return len(pd.read_csv(path))
    return len(json.loads(path.read_text(encoding="utf-8")))


def test_row_counts_match_files():
    checked = 0
    for name, n in re.findall(r"^\| `([^`]+)` \| ([\d,]+)(?: |\|)", CARD.read_text(encoding="utf-8"), re.M):
        path = ROOT / name if "/" in name else PROC / name
        assert path.exists(), name
        assert rows(path) == int(n.replace(",", "")), name
        checked += 1
    assert checked >= 17


def test_licenses_named():
    text = CARD.read_text(encoding="utf-8")
    assert "ODbL-1.0" in text and text.count("CC BY 4.0") >= 2
