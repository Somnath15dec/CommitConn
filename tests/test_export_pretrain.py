import pandas as pd
import pytest

from src import export_pretrain as E


@pytest.fixture(scope="module")
def built():
    return E.build()


def test_sheets_and_row_counts(built):
    sheets, blocks = built
    assert set(sheets) == set(E.README_ROWS) - {"summary"}
    assert len(sheets["catalog"]) == 10_000 and len(sheets["val_pairs"]) == 1000
    assert len(sheets["cluster_names"]) == 256 and len(sheets["dev"]) == 441 + 42
    assert sheets["catalog"].is_dev_track.sum() == 441
    assert [t for t, _ in blocks][0] == "Catalog composition by source"


def test_lists_joined_and_sid_format(built):
    sheets, _ = built
    c = sheets["catalog"]
    assert c.tags.map(lambda v: isinstance(v, str)).all()
    assert c.sid.str.fullmatch(r"\d+-\d+-\d+-\d+").all()
    assert set(sheets["human_pairs"].kind) == {"context", "combo"}


def test_write(tmp_path, built):
    sheets, blocks = built
    out = tmp_path / "x.xlsx"
    counts = E.write(out, {k: v.head(50) for k, v in sheets.items()}, blocks)
    assert out.exists() and counts["catalog"] == 50
    assert pd.read_excel(out, sheet_name="README").sheet.iloc[0] == "README"
