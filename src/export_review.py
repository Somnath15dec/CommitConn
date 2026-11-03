"""Export exports/catalog_review.xlsx for manual review (not committed)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.paths import (CATALOG, CONTEXT_LABELS, EXPORTS, MUSE_VALIDATION, PLAYLIST_CONTEXT,
                       PLAYLIST_MATCHED, PLAYLIST_STATS, REAL_USERS, RULE_PRECISION)
from src.playlists import clean_context_playlists
from src.spotify import build_tracks, load_raw

OUT = EXPORTS / "catalog_review.xlsx"
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _join(v):
    return "; ".join(map(str, v)) if isinstance(v, (list, tuple)) or hasattr(v, "tolist") else v


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_join).map(lambda s: _ILLEGAL.sub("", s) if isinstance(s, str) else s)
    return df


def main(out: Path = OUT) -> None:
    cat = pd.read_parquet(CATALOG)
    lab = pd.read_parquet(CONTEXT_LABELS)
    users = pd.read_parquet(REAL_USERS)
    prec = pd.read_csv(RULE_PRECISION)
    val = pd.read_csv(MUSE_VALIDATION)
    stats = json.loads(PLAYLIST_STATS.read_text())

    # distinct cleaned context playlists per catalog track
    matched = pd.read_parquet(PLAYLIST_MATCHED)
    clean_ctx = clean_context_playlists(matched, pd.read_parquet(PLAYLIST_CONTEXT), build_tracks(load_raw()))
    in_ctx = matched.merge(clean_ctx[["user_id", "playlistname"]], on=["user_id", "playlistname"])
    ctx_count = in_ctx.groupby("track_id").size()
    ctx_words = lab.sort_values(["track_id", "playlist_count"], ascending=[True, False]) \
                   .groupby("track_id").context_word.agg("; ".join)
    n_real = users.groupby("track_id").user_id.nunique()

    c = cat.copy()
    c["artists"] = c.artists.str.replace(";", ", ")
    c["playlist_context_words"] = c.track_id.map(ctx_words).fillna("")
    c["playlist_context_count"] = c.track_id.map(ctx_count).fillna(0).astype(int)
    c["n_real_users"] = c.track_id.map(n_real).fillna(0).astype(int)
    c = c[["track_id", "track_name", "artists", "primary_genre", "genres", "category", "tags", "context", "desc",
           "source", "muse_match", "muse_valence", "muse_arousal", "muse_n_tags",
           "playlist_context_words", "playlist_context_count", "n_real_users", "popularity",
           "tempo", "energy", "valence", "danceability", "acousticness", "instrumentalness", "speechiness"]]

    l = lab.merge(c[["track_id", "track_name", "artists"]], on="track_id")[
        ["track_id", "track_name", "artists", "context_word", "playlist_count", "label_split"]]

    g = cat.set_index("track_id").primary_genre
    sz = users.groupby(["user_id", "split"]).size().unstack(fill_value=0)
    hist = users[users.split == "history"].assign(g=lambda d: d.track_id.map(g))
    top3 = hist.groupby("user_id").g.agg(lambda s: "; ".join(s.value_counts().head(3).index))
    ru = pd.DataFrame({"user_id": sz.index, "n_history": sz["history"].values,
                       "n_heldout": sz["heldout"].values, "top3_primary_genres": top3.reindex(sz.index).values})

    blocks = [
        ("Composition by source", cat.source.value_counts().rename_axis("source").reset_index(name="tracks")),
        ("Genre distribution", cat.primary_genre.value_counts().rename_axis("primary_genre").reset_index(name="tracks")),
        ("Context labels per word (cleaned playlists, catalog tracks)",
         lab.groupby("context_word").agg(tracks=("track_id", "nunique"), label_rows=("track_id", "size"),
                                         playlist_count_sum=("playlist_count", "sum"))
            .sort_values("tracks", ascending=False).reset_index()),
        ("Rule precision (TUNE half only)", prec),
        ("Key numbers", pd.DataFrame([
            ("catalog tracks", len(cat)),
            ("real users", users.user_id.nunique()),
            ("playlist CSV malformed lines skipped", stats["malformed_lines_skipped"]),
            ("playlist CSV null-user rows dropped", stats["rows_null_user_dropped"]),
            ("% tracks with no rule-based context tag", round(100 * (cat.context.str.len() == 0).mean(), 1)),
        ], columns=["metric", "value"])),
    ]

    EXPORTS.mkdir(parents=True, exist_ok=True)
    sheets = {"catalog": _clean(c), "context_labels": _clean(l), "real_users": ru, "muse_validation": val,
              "rule_precision_tune": prec}
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
        row = 0
        for title, df in blocks:
            pd.DataFrame([[title]]).to_excel(xw, sheet_name="summary", startrow=row, index=False, header=False)
            df.to_excel(xw, sheet_name="summary", startrow=row + 1, index=False)
            row += len(df) + 4
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(cell.value or "")) for cell in list(col)[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)

    print(f"Wrote {out}")
    for name, df in sheets.items():
        print(f"  {name:16s} {len(df):,} rows")
    print(f"  {'summary':16s} {sum(len(d) for _, d in blocks):,} rows in {len(blocks)} tables")


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT))
