"""Export exports/phase1_final.xlsx: one review workbook with all final Phase 1 data (not committed)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from src.analyze_readiness import content_desc
from src.muse_match import song_key
from src.paths import (CATALOG, CONTEXT_LABELS, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, EXPORTS, MUSE_VALIDATION,
                       REAL_USERS, ROOT, RULE_PRECISION, TRAIN_PAIRS)

OUT = EXPORTS / "phase1_final.xlsx"
GOLD = ROOT / "eval" / "gold_queries.jsonl"
READINESS_REPORT = ROOT / "eval" / "readiness_report.md"

AUDIO_WEIGHT_DECISION = 1.0
ORACLE_READINESS = 0.893      # readiness v2 sample (500 random tracks, random_state=7)
WRAP_COLUMNS = {"desc", "content_desc", "query", "contents", "detail", "wanted_tags", "genres", "tags", "context",
                "playlist_context_words", "source_file"}
MAX_WIDTH, MIN_WIDTH = 60, 8
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ---------------------------------------------------------------- helpers
def join_list(v):
    if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
        return "; ".join(map(str, list(v)))
    return v


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(join_list).map(lambda s: _ILLEGAL.sub("", s) if isinstance(s, str) else s)
    return df


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def git_commit() -> str:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    h = run("rev-parse", "HEAD")
    return h + (" (working tree dirty)" if run("status", "--porcelain") else "")


def md_table(text: str, header_startswith: str) -> pd.DataFrame:
    """Parse the first markdown table whose header row starts with `header_startswith`."""
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(header_startswith))
    rows = []
    for l in lines[i:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip().strip("|").split("|")])
    df = pd.DataFrame(rows[2:], columns=rows[0])
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


def oracle_ceiling(cat: pd.DataFrame, track_ids) -> float:
    grp = cat.set_index("track_id").content_desc.map(cat.content_desc.value_counts())
    return round(float((10 / grp.loc[list(track_ids)]).clip(upper=1).mean()), 3)


def duplicate_count(cat: pd.DataFrame) -> int:
    key = pd.Series([song_key(n, a) for n, a in zip(cat.track_name, cat.artists)])
    return int(key.duplicated().sum())


def test_count() -> int | None:
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    return int(m.group(1)) if m else None


def with_names(df: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    return df.merge(cat[["track_id", "track_name", "artists"]], on="track_id", how="left")


# ---------------------------------------------------------------- sheets
def build_sheets(n_tests: int | None = None) -> tuple[dict[str, pd.DataFrame], list[tuple[str, pd.DataFrame]]]:
    cat = pd.read_parquet(CATALOG)
    cat["content_desc"] = cat.apply(content_desc, axis=1)
    cat["artists"] = cat.artists.str.replace(";", ", ")
    lab = pd.read_parquet(CONTEXT_LABELS)
    users = pd.read_parquet(REAL_USERS)
    train, eval_a = read_jsonl(TRAIN_PAIRS), read_jsonl(EVAL_PAIRS)
    eval_b, gold = read_jsonl(EVAL_CONTEXT_QUERIES), read_jsonl(GOLD)
    prec, muse = pd.read_csv(RULE_PRECISION), pd.read_csv(MUSE_VALIDATION)
    report = READINESS_REPORT.read_text(encoding="utf-8")

    # catalog
    words = lab.sort_values(["track_id", "playlist_count"], ascending=[True, False]) \
               .groupby("track_id").context_word.agg(list)
    split = lab.drop_duplicates("track_id").set_index("track_id").label_split
    n_real = users.groupby("track_id").user_id.nunique()
    c = cat.assign(playlist_context_words=cat.track_id.map(words).map(lambda v: v if isinstance(v, list) else []),
                   label_split=cat.track_id.map(split).fillna(""),
                   n_real_users=cat.track_id.map(n_real).fillna(0).astype(int))
    c = c[["track_id", "track_name", "artists", "primary_genre", "genres", "category", "tags", "context", "desc",
           "content_desc", "source", "muse_match", "muse_valence", "muse_arousal", "playlist_context_words",
           "label_split", "n_real_users", "popularity", "energy", "valence", "tempo", "danceability",
           "acousticness", "instrumentalness", "speechiness"]]

    pair_cols = ["query", "track_id", "track_name", "artists", "type", "template_id", "wanted_tags"]
    tr = with_names(train, cat)[pair_cols]
    ea = with_names(eval_a, cat)[pair_cols]
    eb = eval_b[["query", "context_word", "playlist_words", "surface", "template_id"]]
    g = gold.reindex(columns=["query", "type", "subtype", "wanted_tags", "wanted_context", "avoid_tags",
                              "expected"]).map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v))
                                               else v)
    cl = with_names(lab, cat)[["track_id", "track_name", "artists", "context_word", "playlist_count",
                               "label_split"]]
    rl = prec.rename(columns={"tracks_tagged": "tagged"})[["rule", "context", "tagged", "precision", "base_rate",
                                                            "lift"]]

    # readiness
    vc = cat.content_desc.value_counts()
    ctx_counts = pd.Series([w for ws in cat.context for w in ws]).value_counts()
    rd = pd.concat([
        pd.DataFrame({"table": "largest identical content_desc groups", "item": vc.head(10).index,
                      "value": vc.head(10).values}),
        pd.DataFrame({"table": "tracks per context word", "item": ctx_counts.index, "value": ctx_counts.values}),
    ], ignore_index=True)

    # real users: one row per user
    pg = cat.set_index("track_id").primary_genre
    sz = users.groupby(["user_id", "split"]).size().unstack(fill_value=0)
    hist = users[users.split == "history"].assign(g=lambda d: d.track_id.map(pg))
    top3 = hist.groupby("user_id").g.agg(lambda s: list(s.value_counts().head(3).index))
    ru = pd.DataFrame({"user_id": sz.index, "n_history": sz.get("history", 0).values,
                       "n_heldout": sz.get("heldout", 0).values,
                       "top3_primary_genres": top3.reindex(sz.index).values})

    # summary blocks
    n = len(cat)
    sweep = md_table(report, "| option |")
    key = pd.DataFrame([
        ("catalog tracks", n),
        ("% general listening (no context tag)", round(100 * (cat.context.str.len() == 0).mean(), 1)),
        ("workout share (% tracks with workout context)", round(100 * cat.context.map(lambda v: "workout" in v)
                                                                 .mean(), 1)),
        ("duplicate songs (song_key: version-normalized title + first artist)", duplicate_count(cat)),
        ("AUDIO_WEIGHT decision (option B)", AUDIO_WEIGHT_DECISION),
        ("Hit@10 oracle ceiling, Eval A tracks", oracle_ceiling(cat, eval_a.track_id)),
        ("Hit@10 oracle ceiling, readiness v2 sample", ORACLE_READINESS),
        ("train pairs / Eval A / Eval B / gold", f"{len(train)} / {len(eval_a)} / {len(eval_b)} / {len(gold)}"),
        ("pytest tests collected", n_tests if n_tests is not None else "n/a"),
    ], columns=["metric", "value"])
    blocks = [
        ("Key numbers", key),
        ("Catalog composition by source", cat.source.value_counts().rename_axis("source")
            .reset_index(name="tracks").assign(pct=lambda d: (100 * d.tracks / n).round(1))),
        ("Primary genre, top 15", cat.primary_genre.value_counts().head(15).rename_axis("primary_genre")
            .reset_index(name="tracks")),
        ("Context tag counts (rule-based)", ctx_counts.rename_axis("context").reset_index(name="tracks")
            .assign(pct=lambda d: (100 * d.tracks / n).round(1))),
        ("MuSe correlations (Pearson r)", muse),
        (f"AUDIO_WEIGHT sweep (decision: {AUDIO_WEIGHT_DECISION}, fallback 1.5)", sweep),
    ]

    sheets = {"catalog": c, "train_pairs": tr, "eval_A": ea, "eval_B": eb, "gold": g, "context_labels": cl,
              "rule_lift": rl, "readiness": rd, "real_users": ru}
    sheets = {k: clean(v) for k, v in sheets.items()}
    return sheets, blocks


README_ROWS = {
    "summary": ("Key numbers, composition by source, genre top 15, context counts, MuSe correlations, "
                "AUDIO_WEIGHT sweep", "catalog.parquet, muse_validation.csv, eval/readiness_report.md"),
    "catalog": ("All catalog tracks with tags, context, desc, content_desc, playlist labels, audio features",
                "data/proc/catalog.parquet + context_labels.parquet + real_users.parquet"),
    "train_pairs": ("Templated training queries (TRAIN_TEMPLATES)", "data/proc/train_pairs.jsonl"),
    "eval_A": ("Eval A pair holdout (EVAL_TEMPLATES), 1 query per track", "data/proc/eval_pairs.jsonl"),
    "eval_B": ("Eval B context-only queries, headline playlist precision set",
               "data/proc/eval_context_queries.jsonl"),
    "gold": ("Hand-written gold queries: 30 accuracy + 8 robustness", "eval/gold_queries.jsonl"),
    "context_labels": ("Playlist-derived context labels per track (tune/eval split by track)",
                       "data/proc/context_labels.parquet"),
    "rule_lift": ("Context rule precision and lift, TUNE half only", "data/proc/rule_precision_tune.csv"),
    "readiness": ("10 largest identical content_desc groups; tracks per context word",
                  "catalog.parquet (recomputed)"),
    "real_users": ("One row per real user: history/heldout sizes, top 3 primary genres of history",
                   "data/proc/real_users.parquet"),
}


def readme(sheets, blocks, commit: str) -> pd.DataFrame:
    rows = [("README", "This index", len(README_ROWS) + 1, "–")]
    for name in README_ROWS:
        n = sum(len(d) for _, d in blocks) if name == "summary" else len(sheets[name])
        rows.append((name, README_ROWS[name][0], n, README_ROWS[name][1]))
    rows += [("", "", "", ""), ("git commit", commit, "", ""),
             ("exported at", datetime.now().strftime("%Y-%m-%d %H:%M"), "", "")]
    return pd.DataFrame(rows, columns=["sheet", "contents", "rows", "source_file"])


# ---------------------------------------------------------------- writing
def format_sheet(ws, filters: bool = True, wrap: set[str] = WRAP_COLUMNS) -> None:
    ws.freeze_panes = "A2"
    if filters and ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions
    header = {cell.column: str(cell.value or "") for cell in ws[1]}
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        sample = [ws.cell(row=r, column=col_idx).value for r in range(1, min(ws.max_row, 300) + 1)]
        width = max((len(str(v)) for v in sample if v is not None), default=MIN_WIDTH)
        ws.column_dimensions[letter].width = min(max(MIN_WIDTH, width + 2), MAX_WIDTH)
        if header.get(col_idx) in wrap or width > MAX_WIDTH:
            for (cell,) in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                cell.alignment = Alignment(wrap_text=True, vertical="top")


def write_workbook(out: Path, sheets, blocks, commit: str) -> dict[str, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    rd = readme(sheets, blocks, commit)
    counts = {"README": len(rd), "summary": sum(len(d) for _, d in blocks)}
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        rd.to_excel(xw, sheet_name="README", index=False)
        # summary: stacked tables, one bold title row each
        row = 0
        pd.DataFrame(columns=["Phase 1 summary"]).to_excel(xw, sheet_name="summary", index=False)
        row = 2
        titles = []
        for title, df in blocks:
            pd.DataFrame([[title]]).to_excel(xw, sheet_name="summary", startrow=row, index=False, header=False)
            titles.append(row + 1)
            df.to_excel(xw, sheet_name="summary", startrow=row + 1, index=False)
            titles.append(row + 2)
            row += len(df) + 4
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            counts[name] = len(df)
        format_sheet(xw.sheets["README"])
        ws = xw.sheets["summary"]
        format_sheet(ws, filters=False)
        for r in titles:
            for cell in ws[r]:
                cell.font = Font(bold=True)
        for name in sheets:
            format_sheet(xw.sheets[name])
    return counts


def main(out: Path = OUT) -> None:
    sheets, blocks = build_sheets(n_tests=test_count())
    counts = write_workbook(out, sheets, blocks, git_commit())
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    for name, n in counts.items():
        print(f"  {name:15s} {n:>7,} rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
