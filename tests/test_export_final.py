import pytest
from openpyxl import load_workbook

from src import export_final as ef

SHEET_ORDER = ["README", "summary", "catalog", "train_pairs", "eval_A", "eval_B", "gold", "context_labels",
               "rule_lift", "readiness", "real_users"]


@pytest.fixture(scope="module")
def built():
    return ef.build_sheets(n_tests=None)   # no pytest-in-pytest


def test_sheet_columns(built):
    sheets, _ = built
    assert list(sheets["catalog"].columns)[:4] == ["track_id", "track_name", "artists", "primary_genre"]
    assert len(sheets["catalog"].columns) == 25
    assert list(sheets["train_pairs"].columns) == list(sheets["eval_A"].columns) == [
        "query", "track_id", "track_name", "artists", "type", "template_id", "wanted_tags"]
    assert list(sheets["rule_lift"].columns) == ["rule", "context", "tagged", "precision", "base_rate", "lift"]
    assert list(sheets["real_users"].columns) == ["user_id", "n_history", "n_heldout", "top3_primary_genres"]


def test_sizes_and_joins(built):
    sheets, _ = built
    assert len(sheets["catalog"]) == 10_000
    assert len(sheets["eval_A"]) == 500 and len(sheets["eval_B"]) == 132 and len(sheets["gold"]) == 38
    assert sheets["train_pairs"].track_name.notna().all()
    assert sheets["real_users"].user_id.is_unique
    # lists are joined with "; "
    assert sheets["catalog"].genres.map(lambda v: isinstance(v, str)).all()


def test_summary_numbers(built):
    _, blocks = built
    key = dict(blocks[0][1].values)
    assert key["duplicate songs (normalized title + first artist)"] == 0
    assert key["Hit@10 oracle ceiling, Eval A tracks"] == pytest.approx(0.889, abs=1e-3)
    assert key["AUDIO_WEIGHT decision (option B)"] == 1.0
    sweep = dict(blocks)[next(t for t, _ in blocks if t.startswith("AUDIO_WEIGHT sweep"))]
    assert "B: w=1.0" in set(sweep.option)


def test_workbook_written(built, tmp_path):
    sheets, blocks = built
    small = {k: v.head(50) for k, v in sheets.items()}
    out = tmp_path / "wb.xlsx"
    ef.write_workbook(out, small, blocks, "deadbeef")
    wb = load_workbook(out)
    assert wb.sheetnames == SHEET_ORDER
    assert wb["catalog"].freeze_panes == "A2" and wb["catalog"].auto_filter.ref
    assert any("deadbeef" in str(c.value) for row in wb["README"].iter_rows() for c in row)
