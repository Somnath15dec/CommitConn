"""Step 4.5: data readiness analysis (read-only; never modifies the catalog).

Writes eval/readiness_report.md and prints a summary.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

from src.muse_match import normalize_artist, normalize_title
from src.paths import CATALOG, ROOT

REPORT = ROOT / "eval" / "readiness_report.md"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AUDIO_COLS = ["energy", "valence", "tempo", "acousticness", "instrumentalness", "danceability", "speechiness"]
AUDIO_WEIGHT = 0.5                      # default for option B neighbour listing
AUDIO_WEIGHT_SWEEP = [0.5, 1.0, 1.5, 2.0]
MIN_SAME_CONTEXT = 0.90                 # recommend highest weight keeping 5-NN same-context >= this
ORACLE_N, ORACLE_RANDOM_STATE, HIT_K = 500, 7, 10
MOOD_TAGS = ["happy", "melancholic", "neutral mood"]
THIN_CONTEXT = 150
NN_RANDOM_STATE = 6
N_NN_SAMPLES, K_NN = 10, 5
SID_LEVELS, SID_K, KMEANS_ITERS, KMEANS_SEED = 3, 256, 30, 0


# ---------------------------------------------------------------- helpers
def content_desc(row) -> str:
    """desc without the leading '{title} by {artists}. ' sentence."""
    prefix = f"{row.track_name} by {row.artists.replace(';', ', ')}. "
    assert row.desc.startswith(prefix), row.track_id
    return row.desc[len(prefix):]


def kmeans(x: torch.Tensor, k: int, iters: int = KMEANS_ITERS, seed: int = KMEANS_SEED):
    """Plain Lloyd k-means (Euclidean). Empty clusters are re-seeded from random points."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k).to(x.dtype)
        empty = cnt == 0
        c = sums / cnt.clamp(min=1).unsqueeze(1)
        if empty.any():
            c[empty] = x[torch.randint(len(x), (int(empty.sum()),), generator=g).to(x.device)]
    a = torch.cdist(x, c).argmin(1)
    return a, c


def residual_kmeans(x: torch.Tensor, levels: int = SID_LEVELS, k: int = SID_K) -> np.ndarray:
    codes, r = [], x.clone()
    for lv in range(levels):
        a, c = kmeans(r, k, seed=KMEANS_SEED + lv)
        codes.append(a.cpu().numpy())
        r = r - c[a]
    return np.stack(codes, 1)


def sid_stats(codes: np.ndarray) -> dict:
    keys = [tuple(r) for r in codes]
    cnt = Counter(keys)
    l1 = Counter(codes[:, 0].tolist())
    return {"collision_rate": sum(1 for k in keys if cnt[k] > 1) / len(keys),
            "colliding_groups": sum(1 for v in cnt.values() if v > 1),
            "max_tracks_per_sid": max(cnt.values()),
            "l1_used": len(l1), "l1_largest": max(l1.values()),
            "l1_median": int(np.median(list(l1.values())))}


def nearest(x: torch.Tensor, idx: list[int], k: int = K_NN) -> np.ndarray:
    d = torch.cdist(x[idx], x)
    d[torch.arange(len(idx)), torch.tensor(idx)] = float("inf")
    return d.topk(k, largest=False).indices.cpu().numpy()


def neighbour_agreement(x: torch.Tensor, cat: pd.DataFrame, k: int = K_NN, batch: int = 2000) -> dict:
    """Over ALL tracks: share of k-NN with same primary_genre / sharing a context word / same content_desc."""
    g = cat.primary_genre.to_numpy(); cd = cat.content_desc.to_numpy()
    ctx = [set(c) for c in cat.context]
    same_g = share_ctx = same_cd = n = 0
    for s in range(0, len(x), batch):
        idx = list(range(s, min(s + batch, len(x))))
        nn = nearest(x, idx, k)
        for i, row in zip(idx, nn):
            for j in row:
                same_g += g[i] == g[j]; same_cd += cd[i] == cd[j]
                share_ctx += bool(ctx[i] & ctx[j]) if ctx[i] else 0
                n += 1
    has_ctx = sum(1 for c in ctx if c) * k
    return {"same_primary_genre": same_g / n, "share_context_word": share_ctx / has_ctx,
            "identical_content_desc": same_cd / n}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- analysis
def main() -> None:
    cat = pd.read_parquet(CATALOG)
    cat["content_desc"] = cat.apply(content_desc, axis=1)
    md, summary = ["# Data readiness report (Step 4.5)", "",
                   f"Catalog: {len(cat):,} tracks. Read-only analysis; catalog unchanged.", ""], []

    # 1. content_desc uniqueness
    vc = cat.content_desc.value_counts()
    n_in_dup = int(vc[vc > 1].sum())
    md += ["## 1. content_desc uniqueness", "",
           f"- Unique content_desc: **{len(vc):,} / {len(cat):,}**",
           f"- Tracks sharing their content_desc with at least one other track: {n_in_dup:,} "
           f"({n_in_dup / len(cat):.1%})", "", "10 largest identical groups:", "",
           md_table(vc.head(10).rename_axis("content_desc").reset_index(name="tracks")), ""]
    summary.append(f"1. content_desc unique: {len(vc):,}/{len(cat):,}; {n_in_dup / len(cat):.1%} of tracks share one; "
                   f"largest groups: {vc.head(10).tolist()}")

    # 2. duplicate songs
    key = list(zip(cat.track_name.map(normalize_title), cat.artists.map(normalize_artist)))
    cat["song_key"] = key
    kc = cat.song_key.value_counts()
    dup_keys = kc[kc > 1]
    ex = cat[cat.song_key.isin(dup_keys.index[:5])].sort_values("song_key")
    md += ["## 2. Duplicate songs (same normalized title + first artist)", "",
           f"- Groups: **{len(dup_keys):,}**, tracks involved: {int(dup_keys.sum()):,}, largest group: "
           f"{int(dup_keys.max()) if len(dup_keys) else 0}", "", "Examples (5 groups):", "",
           md_table(ex[["track_id", "track_name", "artists", "primary_genre", "source"]]), ""]
    summary.append(f"2. duplicate songs: {len(dup_keys):,} groups / {int(dup_keys.sum()):,} tracks")
    md[0] = "# Data readiness report (Step 4.5, v2 after Step 4.6 catalog changes)"

    # 3. tracks per context word
    cw = cat.context.explode().dropna().value_counts()
    thin = cw[cw < THIN_CONTEXT]
    md += ["## 3. Tracks per context word", "", f"Flag: under {THIN_CONTEXT} tracks.", "",
           md_table(cw.rename_axis("context").reset_index(name="tracks")
                    .assign(flag=lambda d: np.where(d.tracks < THIN_CONTEXT, "THIN", ""))), ""]
    summary.append(f"3. context words under {THIN_CONTEXT}: {thin.to_dict() or 'none'}")

    # 4. ambiguity: tracks matching a context word, and mood + context combos
    combos = []
    for w in cw.index:
        has_w = cat.context.map(lambda c: w in c)
        for m in MOOD_TAGS:
            combos.append({"mood": m, "context": w, "tracks": int((has_w & cat.tags.map(lambda t: m in t)).sum())})
    cb = pd.DataFrame(combos)
    cb_nz = cb[cb.tracks > 0]

    def mmm(s):
        return f"min {s.min():,} / median {int(s.median()):,} / max {s.max():,}"
    md += ["## 4. Query ambiguity (matching tracks per query intent)", "",
           f"- Single context word: {mmm(cw)} (over {len(cw)} words)",
           f"- Mood + context combos (non-empty, {len(cb_nz)} of {len(cb)}): {mmm(cb_nz.tracks)}",
           f"- Empty combos: {len(cb) - len(cb_nz)}", "",
           md_table(cb.pivot(index="context", columns="mood", values="tracks").reindex(cw.index)
                    .reset_index()), ""]
    summary.append(f"4. ambiguity: context word {mmm(cw)}; mood+context {mmm(cb_nz.tracks)}; "
                   f"{len(cb) - len(cb_nz)} empty combos")

    # 5. embeddings + SID preview
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else None
    assert dev == "cuda", "CUDA not available; refusing to silently fall back to CPU"
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    enc = lambda texts: model.encode(list(texts), batch_size=256, convert_to_tensor=True,
                                     normalize_embeddings=True, show_progress_bar=False).float()
    xa = enc(cat.desc)
    xt = enc(cat.content_desc)
    feats = cat[AUDIO_COLS].astype(float)
    feats = (feats - feats.min()) / (feats.max() - feats.min())
    ft = torch.tensor(feats.to_numpy(), dtype=torch.float32, device=dev)
    option_b = lambda w: torch.cat([xt, w * ft], 1)

    md += ["## 5. Embedding options", "",
           f"Model `{EMBED_MODEL}` on {torch.cuda.get_device_name(0)}. Neighbours by Euclidean distance "
           f"(A: normalized full-desc vector; B: normalized content_desc vector + w x min-max audio "
           f"features {AUDIO_COLS}). SID preview: {SID_LEVELS} x {SID_K} residual k-means "
           f"({KMEANS_ITERS} Lloyd iterations, torch).", ""]
    rows = []
    for name, x in [("A: full desc", xa)] + [(f"B: w={w}", option_b(w)) for w in AUDIO_WEIGHT_SWEEP]:
        s = sid_stats(residual_kmeans(x))
        a = neighbour_agreement(x, cat)
        rows.append({"option": name, "collision_rate": round(s["collision_rate"], 4),
                     "max_tracks_per_code": s["max_tracks_per_sid"], "l1_used": s["l1_used"],
                     "l1_largest": s["l1_largest"], "nn5_same_context": round(a["share_context_word"], 3),
                     "nn5_same_genre": round(a["same_primary_genre"], 3),
                     "nn5_identical_content_desc": round(a["identical_content_desc"], 3)})
    sw = pd.DataFrame(rows)
    ok = sw[sw.option.str.startswith("B") & (sw.nn5_same_context >= MIN_SAME_CONTEXT)]
    best_w = AUDIO_WEIGHT_SWEEP[sw.option.tolist().index(ok.option.iloc[-1]) - 1] if len(ok) else None
    rec = (f"Recommended AUDIO_WEIGHT = {best_w} (highest weight with 5-NN same context >= "
           f"{MIN_SAME_CONTEXT:.0%})" if best_w is not None else
           f"No weight keeps 5-NN same context >= {MIN_SAME_CONTEXT:.0%}")
    md += ["### AUDIO_WEIGHT sweep (option B) vs option A", "", md_table(sw), "", f"**{rec}**", ""]
    summary.append("5. AUDIO_WEIGHT sweep:\n" + sw.to_string(index=False) + "\n   " + rec)

    sample = cat.sample(N_NN_SAMPLES, random_state=NN_RANDOM_STATE).index.tolist()
    for name, x in [("A: full desc", xa), (f"B: w={best_w or AUDIO_WEIGHT}", option_b(best_w or AUDIO_WEIGHT))]:
        md += [f"### {name}: 10 random tracks and their 5 nearest neighbours", ""]
        for i, row in zip(sample, nearest(x, sample)):
            md.append(f"- **{cat.track_name[i]}** ({cat.primary_genre[i]}): {cat.content_desc[i]}")
            md += [f"    - {cat.track_name[j]} ({cat.primary_genre[j]}): {cat.content_desc[j]}" for j in row]
        md.append("")

    # 6. Hit@10 oracle ceiling: a perfect system cannot separate tracks with identical content_desc
    grp = cat.content_desc.map(cat.content_desc.value_counts())
    samp = grp.sample(ORACLE_N, random_state=ORACLE_RANDOM_STATE)
    ceil = (HIT_K / samp).clip(upper=1.0)
    md += ["## 6. Hit@10 oracle ceiling", "",
           f"{ORACLE_N} tracks (random_state={ORACLE_RANDOM_STATE}); per track min(1, {HIT_K} / size of its "
           f"identical-content_desc group).", "",
           f"- **Mean ceiling: {ceil.mean():.3f}**",
           f"- Tracks with ceiling 1.0: {(ceil == 1).mean():.1%}; median group size {int(samp.median())}, "
           f"max {int(samp.max())}", ""]
    summary.append(f"6. Hit@10 oracle ceiling (n={ORACLE_N}): mean {ceil.mean():.3f}; "
                   f"{(ceil == 1).mean():.1%} of tracks at 1.0; max group {int(samp.max())}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    sys.exit(main())
