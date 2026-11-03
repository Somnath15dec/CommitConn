"""Export exports/pretrain_dataset.xlsx: snapshot of everything the SID model is trained and selected on
(exports/ is gitignored). Reuses the workbook formatting of src/export_final.py.

    python -m src.export_pretrain [out.xlsx]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font

from src.analyze_readiness import content_desc
from src.cluster_names import CLUSTER_NAMES
from src.dev_split import DEV_FAISS, DEV_TRACKS, dev_queries, dev_tracks
from src.export_final import GOLD, clean, format_sheet, git_commit, read_jsonl
from src.paths import CATALOG, CONTEXT_LABELS, EVAL_CONTEXT_QUERIES, EVAL_PAIRS, EXPORTS, ROOT
from src.sids import purity_report, stats as sid_stats
from src.train_data import (HUMAN_PAIRS, MAX_PER_STRING, SIDS, TRAIN_POOL, VAL_PAIRS, epoch_indices,
                            frozen_split)
from src.paraphrase import PARAPHRASE_PAIRS

OUT = EXPORTS / "pretrain_dataset.xlsx"
RESULTS_BASELINE = ROOT / "eval" / "results_baseline.json"
WRAP = {"content_desc", "query", "contents", "genres", "tags", "context", "tune_labels", "playlist_words", "value",
        "wanted_tags", "source_file", "detail"}
README_ROWS = {
    "summary": ("Headline counts: catalog, training pool, dev set, FAISS baseline, SID stats", "several"),
    "catalog": ("10,000 tracks with SID, cluster name, label split and dev flag", "catalog.parquet, sids.parquet"),
    "train_pool": ("All training pairs (frozen)", "train_pool.parquet"),
    "val_pairs": ("1,000 validation pairs held out from the pool (frozen)", "val_pairs.parquet"),
    "paraphrases": ("LLM paraphrase pairs (kept after filters)", "paraphrase_pairs.jsonl"),
    "human_pairs": ("Tune-half context-label pairs", "human_pairs.jsonl"),
    "dev": ("441 dev tracks with tune labels, then the 42 dev queries", "dev_tracks.csv, context_labels.parquet"),
    "cluster_names": ("256 level-1 clusters: name, size, top genre / context", "cluster_names.json"),
    "eval_A": ("Eval A pair holdout (500)", "eval_pairs.jsonl"),
    "eval_B": ("Eval B context queries (132)", "eval_context_queries.jsonl"),
    "gold": ("Gold queries (30 accuracy + 8 robustness)", "eval/gold_queries.jsonl"),
}


def sid_str(df: pd.DataFrame) -> pd.Series:
    return df[["a", "b", "c", "d"]].astype(int).astype(str).agg("-".join, axis=1)


def kv(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["metric", "value"])


def build() -> tuple[dict[str, pd.DataFrame], list[tuple[str, pd.DataFrame]]]:
    cat = pd.read_parquet(CATALOG)
    cat["content_desc"] = cat.apply(content_desc, axis=1)
    cat["artists"] = cat.artists.str.replace(";", ", ")
    sids = pd.read_parquet(SIDS)
    names = json.loads(CLUSTER_NAMES.read_text(encoding="utf-8"))
    lab = pd.read_parquet(CONTEXT_LABELS)
    dev = dev_tracks()
    split = lab.drop_duplicates("track_id").set_index("track_id").label_split
    c = cat.merge(sids, on="track_id")
    c["sid"] = sid_str(c)
    c["cluster_name"] = c.a.astype(int).astype(str).map(lambda a: names[a]["name"])
    c["label_split"] = c.track_id.map(split).fillna("")
    c["is_dev_track"] = c.track_id.isin(dev)
    catalog = c[["track_id", "track_name", "artists", "primary_genre", "genres", "tags", "context", "content_desc",
                 "source", "label_split", "is_dev_track", "sid", "cluster_name", "energy", "valence", "tempo",
                 "acousticness", "instrumentalness", "danceability", "speechiness"]]
    names_by_id = cat.set_index("track_id")[["track_name", "artists", "content_desc"]]

    def pairs(df):
        d = df.copy()
        d["sid"] = sid_str(d)
        d = d.join(names_by_id[["track_name", "artists"]], on="track_id")
        return d[["query", "track_id", "track_name", "artists", "source", "sid"]]

    tr, val = frozen_split()
    para = read_jsonl(PARAPHRASE_PAIRS).join(names_by_id[["track_name", "content_desc"]], on="track_id")
    human = read_jsonl(HUMAN_PAIRS).join(names_by_id[["track_name"]], on="track_id")
    human = human.rename(columns={"type": "kind"})[["query", "track_id", "track_name", "context_word", "kind"]]

    tune = lab[lab.label_split == "tune"].sort_values(["track_id", "playlist_count"], ascending=[True, False])
    tl = tune.groupby("track_id").context_word.agg(list)
    dv = pd.DataFrame({"track_id": sorted(dev)}).join(names_by_id[["track_name", "artists"]], on="track_id")
    dv["tune_labels"] = dv.track_id.map(tl)
    dq = pd.DataFrame(dev_queries())
    dev_sheet = pd.concat([dv.assign(row_type="dev_track"),
                           dq.rename(columns={"query": "query"}).assign(row_type="dev_query")], ignore_index=True)
    dev_sheet = dev_sheet[["row_type", "track_id", "track_name", "artists", "tune_labels", "query", "context_word",
                           "surface", "template_id", "playlist_words"]]

    m = sids.merge(cat[["track_id", "primary_genre", "context"]], on="track_id")
    cl = []
    for a, g in m.groupby("a"):
        gen = g.primary_genre.value_counts()
        ctx = Counter(w for ws in g.context for w in ws).most_common(1)
        cl.append({"cluster_id": int(a), "name": names[str(int(a))]["name"], "size": len(g),
                   "top_genre": gen.index[0], "top_genre_share": round(gen.iloc[0] / len(g), 3),
                   "top_context": ctx[0][0] if ctx else "", "top_context_share": round(ctx[0][1] / len(g), 3) if ctx else 0.0})

    eval_a = read_jsonl(EVAL_PAIRS).join(names_by_id[["track_name", "artists"]], on="track_id")
    eval_a = eval_a[["query", "track_id", "track_name", "artists", "type", "template_id", "wanted_tags"]]
    eval_b = read_jsonl(EVAL_CONTEXT_QUERIES)[["query", "context_word", "playlist_words", "surface", "template_id"]]
    gold = read_jsonl(GOLD).reindex(columns=["query", "type", "subtype", "wanted_tags", "wanted_context",
                                             "avoid_tags", "expected"]).map(
        lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else v)

    sheets = {"catalog": catalog, "train_pool": pairs(tr), "val_pairs": pairs(val),
              "paraphrases": para[["query", "style", "track_id", "track_name", "content_desc"]],
              "human_pairs": human, "dev": dev_sheet, "cluster_names": pd.DataFrame(cl),
              "eval_A": eval_a, "eval_B": eval_b, "gold": gold}
    sheets = {k: clean(v) for k, v in sheets.items()}

    # summary blocks
    ep = epoch_indices(tr, 0)
    rb = json.loads(RESULTS_BASELINE.read_text(encoding="utf-8"))
    fd = json.loads(DEV_FAISS.read_text(encoding="utf-8"))
    ss = sid_stats(sids)
    pu = purity_report(sids, cat.assign(artists=cat.artists))
    fmt = lambda s: f"{s['mean']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
    pool_rows = [(f"train pool: {k}", int(v)) for k, v in tr.source.value_counts().items()]
    pool_rows += [("train pool: total", len(tr)), ("validation pairs", len(val)),
                  (f"epoch 0 after {MAX_PER_STRING}/string cap", len(ep))]
    pool_rows += [(f"epoch 0: {k}", int(v)) for k, v in tr.iloc[ep].source.value_counts().items()]
    blocks = [
        ("Catalog composition by source", cat.source.value_counts().rename_axis("source").reset_index(name="tracks")),
        ("Training pool (frozen)", kv(pool_rows)),
        ("Dev set", kv([("dev tracks", len(dev)), ("dev queries", len(dq)),
                        ("dev pool size (for condensed metric)", fd["dev_pool_size"])])),
        ("FAISS baseline (frozen, eval/results_baseline.json)", kv([
            ("Eval B playlist precision@10 (headline)", fmt(rb["eval_b"]["precision@10"])),
            ("Eval B chance", round(rb["eval_b"]["chance"], 3)),
            ("Eval B condensed precision@10", fmt(rb["eval_b"]["condensed_precision@10"])),
            ("Eval A Hit@10", fmt(rb["eval_a"]["hit@10"])), ("Eval A NDCG@10", fmt(rb["eval_a"]["ndcg@10"])),
            ("Eval A random-within-match expectation", round(rb["eval_a"]["random_within_match_expectation"], 3)),
            ("Eval A identical-description ceiling", round(rb["eval_a"]["ceiling_identical_desc"], 3)),
            ("Gold tag-match@10", fmt(rb["gold"]["tag_match@10"])),
            ("Gold avoid-violation@10 (mean of 3)", round(rb["gold"]["avoid_violation@10"]["mean"], 3)),
            ("Valid-item rate", rb["valid_item_rate"]), ("Latency p50 (ms)", round(rb["latency_ms"]["p50"], 1))])),
        ("FAISS on the dev set (eval/dev_faiss.json)", kv([
            ("dev (b) precision@10 within dev tracks (primary)", round(fd["dev_b_precision@10"], 3)),
            ("dev (b) chance", round(fd["dev_b_chance"], 3)),
            ("dev (a) condensed precision@10", round(fd["dev_a_condensed_precision@10"], 3)),
            ("dev (a) judged in top 50 (mean)", round(fd["dev_a_judged_in_top50"], 2))])),
        ("Semantic IDs", kv([
            ("collision rate", round(ss["collision_rate"], 4)), ("max tracks per code", ss["max_tracks_per_code"]),
            ("level-1 clusters used", ss["l1_used"]),
            ("level-1 purity: top genre share (random)", f"{pu['level1']['genre']:.3f} ({pu['random']['genre']:.3f})"),
            ("level-1 purity: top context share (random)",
             f"{pu['level1']['context']:.3f} ({pu['random']['context']:.3f})")])),
    ]
    blocks = [(t, clean(d)) for t, d in blocks]
    return sheets, blocks


def write(out: Path, sheets, blocks) -> dict[str, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [("README", "This index", len(README_ROWS) + 4, "–")]
    for name, (what, src) in README_ROWS.items():
        n = sum(len(d) for _, d in blocks) if name == "summary" else len(sheets[name])
        rows.append((name, what, n, src))
    rows += [("", "", "", ""), ("git commit", git_commit(), "", ""),
             ("exported at", datetime.now().strftime("%Y-%m-%d %H:%M"), "", "")]
    rd = pd.DataFrame(rows, columns=["sheet", "contents", "rows", "source_file"])
    counts = {"README": len(rd), "summary": sum(len(d) for _, d in blocks)}
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        rd.to_excel(xw, sheet_name="README", index=False)
        pd.DataFrame(columns=["Pre-training dataset summary"]).to_excel(xw, sheet_name="summary", index=False)
        row, bold = 2, []
        for title, df in blocks:
            pd.DataFrame([[title]]).to_excel(xw, sheet_name="summary", startrow=row, index=False, header=False)
            df.to_excel(xw, sheet_name="summary", startrow=row + 1, index=False)
            bold += [row + 1, row + 2]
            row += len(df) + 4
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            counts[name] = len(df)
        format_sheet(xw.sheets["README"], wrap=WRAP)
        ws = xw.sheets["summary"]
        format_sheet(ws, filters=False, wrap=WRAP)
        for r in bold:
            for cell in ws[r]:
                cell.font = Font(bold=True)
        for name in sheets:
            format_sheet(xw.sheets[name], wrap=WRAP)
    return counts


def main(out: Path = OUT) -> None:
    sheets, blocks = build()
    counts = write(out, sheets, blocks)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    for name, n in counts.items():
        print(f"  {name:15s} {n:>7,} rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
