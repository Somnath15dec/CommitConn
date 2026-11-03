"""Step 11: Semantic IDs.

Item vector (option B): normalized all-MiniLM-L6-v2 embedding of content_desc + AUDIO_WEIGHT x min-max audio
features. Residual k-means, 3 levels x 256 (seed SID_SEED + level). Dedupe digit d makes every full SID unique:
within an (a, b, c) code, tracks are ordered by distance to the code's reconstruction (closest = 0), then track_id.

Writes data/proc/sids.parquet (track_id, a, b, c, d) and data/proc/sid_trie.json ({a: {b: {c: [d, ...]}}}).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from src.analyze_readiness import AUDIO_COLS, EMBED_MODEL, content_desc, kmeans
from src.paths import CATALOG, PROC

SIDS = PROC / "sids.parquet"
SID_TRIE = PROC / "sid_trie.json"
AUDIO_WEIGHT = 1.0
SID_LEVELS, SID_K, SID_SEED = 3, 256, 11
MAX_COLLISION_RATE = 0.20      # check only (report if exceeded; weight is a user decision)
MAX_TRACKS_PER_CODE = 16
SAMPLE_RANDOM_STATE = 12
PURITY_BASELINE_SEED = 15      # random clusters of the same sizes
PREFIX_PAIRS, PREFIX_PAIR_SEED = 2000, 16


def audio_matrix(cat: pd.DataFrame) -> np.ndarray:
    f = cat[AUDIO_COLS].astype(float)
    return ((f - f.min()) / (f.max() - f.min())).to_numpy()


def item_vectors(cat: pd.DataFrame, device: str) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=device)
    text = model.encode(cat.content_desc.tolist(), batch_size=256, convert_to_tensor=True,
                        normalize_embeddings=True, show_progress_bar=False).float()
    audio = torch.tensor(audio_matrix(cat), dtype=torch.float32, device=text.device)
    return torch.cat([text, AUDIO_WEIGHT * audio], 1)


def residual_codes(x: torch.Tensor, levels: int = SID_LEVELS, k: int = SID_K, seed: int = SID_SEED):
    """Returns (codes [n, levels], squared distance of each item to its full reconstruction)."""
    codes, r = [], x.clone()
    for lv in range(levels):
        a, c = kmeans(r, k, seed=seed + lv)
        codes.append(a.cpu().numpy())
        r = r - c[a]
    return np.stack(codes, 1), (r ** 2).sum(1).cpu().numpy()


def dedupe_digit(track_ids, codes: np.ndarray, resid: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"track_id": list(track_ids), "a": codes[:, 0], "b": codes[:, 1], "c": codes[:, 2],
                       "r": resid, "i": range(len(codes))})
    df = df.sort_values(["a", "b", "c", "r", "track_id"])
    df["d"] = df.groupby(["a", "b", "c"]).cumcount()
    return df.sort_values("i").d.to_numpy()


def build_trie(sids: pd.DataFrame) -> dict:
    trie: dict = {}
    for a, b, c, d in sids[["a", "b", "c", "d"]].itertuples(index=False):
        trie.setdefault(str(a), {}).setdefault(str(b), {}).setdefault(str(c), []).append(int(d))
    for bs in trie.values():
        for cs in bs.values():
            for k in cs:
                cs[k] = sorted(cs[k])
    return trie


def allowed_next(trie: dict, prefix: tuple[int, ...]) -> list[int]:
    """Valid next tokens after a SID prefix of length 0-3 (constrained decoding)."""
    node = trie
    for t in prefix:
        node = node[str(t)]
    return sorted(node) if isinstance(node, list) else sorted(int(k) for k in node)


def trie_sids(trie: dict):
    for a, bs in trie.items():
        for b, cs in bs.items():
            for c, ds in cs.items():
                for d in ds:
                    yield int(a), int(b), int(c), int(d)


def stats(sids: pd.DataFrame) -> dict:
    per_code = sids.groupby(["a", "b", "c"]).size()
    l1 = sids.a.value_counts()
    return {"tracks": len(sids), "codes_used": len(per_code),
            "collision_rate": float((per_code[per_code > 1].sum()) / len(sids)),
            "max_tracks_per_code": int(per_code.max()), "max_d": int(sids.d.max()),
            "l1_used": int(len(l1)), "l1_min": int(l1.min()), "l1_median": float(l1.median()),
            "l1_mean": float(l1.mean()), "l1_max": int(l1.max()), "l1_sizes": l1}


def purity(labels: np.ndarray, contexts: list[set], groups: np.ndarray) -> dict:
    """Unweighted mean over clusters of (a) the share of the most common primary_genre and (b) the share of
    tracks carrying the cluster's most common context word (tracks without context count as not carrying it)."""
    df = pd.DataFrame({"g": groups, "genre": labels, "ctx": contexts})
    gp, cp = [], []
    for _, d in df.groupby("g"):
        gp.append(d.genre.value_counts().iloc[0] / len(d))
        words = pd.Series([w for c in d.ctx for w in c])
        cp.append(words.value_counts().iloc[0] / len(d) if len(words) else 0.0)
    return {"genre": float(np.mean(gp)), "context": float(np.mean(cp))}


def purity_report(sids: pd.DataFrame, cat: pd.DataFrame) -> dict:
    m = sids.merge(cat[["track_id", "primary_genre", "context"]], on="track_id")
    ctx = [set(c) for c in m.context]
    real = purity(m.primary_genre.to_numpy(), ctx, m.a.to_numpy())
    shuffled = np.random.default_rng(PURITY_BASELINE_SEED).permutation(m.a.to_numpy())   # same sizes
    return {"level1": real, "random": purity(m.primary_genre.to_numpy(), ctx, shuffled)}


def sample_pairs(keys: np.ndarray, n: int, rng) -> np.ndarray:
    """n random (i, j), i != j, with keys[i] == keys[j]; only groups of size >= 2 are eligible."""
    df = pd.DataFrame({"k": keys, "i": range(len(keys))})
    groups = [g.to_numpy() for _, g in df.groupby("k").i if len(g) > 1]
    elig = np.concatenate(groups)
    member = {i: g for g in groups for i in g}
    out = []
    for i in rng.choice(elig, n):
        g = member[i]
        j = rng.choice(g[g != i])
        out.append((i, j))
    return np.array(out)


def cosine(x: torch.Tensor, pairs: np.ndarray) -> float:
    xn = torch.nn.functional.normalize(x, dim=1)
    i, j = torch.as_tensor(pairs[:, 0], device=x.device), torch.as_tensor(pairs[:, 1], device=x.device)
    return float((xn[i] * xn[j]).sum(1).mean())


def prefix_similarity(x: torch.Tensor, sids: pd.DataFrame) -> dict:
    rng = np.random.default_rng(PREFIX_PAIR_SEED)
    a = sids.a.to_numpy()
    ab = (sids.a.astype(int) * SID_K + sids.b.astype(int)).to_numpy()
    rand = np.array([rng.choice(len(sids), 2, replace=False) for _ in range(PREFIX_PAIRS)])
    return {"share a": cosine(x, sample_pairs(a, PREFIX_PAIRS, rng)),
            "share a,b": cosine(x, sample_pairs(ab, PREFIX_PAIRS, rng)),
            "random": cosine(x, rand), "n_random": len(rand)}


def build() -> tuple[pd.DataFrame, dict, pd.DataFrame, torch.Tensor]:
    cat = pd.read_parquet(CATALOG)
    cat["content_desc"] = cat.apply(content_desc, axis=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = item_vectors(cat, device)
    codes, resid = residual_codes(x)
    sids = pd.DataFrame({"track_id": cat.track_id, "a": codes[:, 0], "b": codes[:, 1], "c": codes[:, 2]})
    sids["d"] = dedupe_digit(cat.track_id, codes, resid)
    sids[["a", "b", "c", "d"]] = sids[["a", "b", "c", "d"]].astype("int16")
    trie = build_trie(sids)
    sids.to_parquet(SIDS, index=False)
    SID_TRIE.write_text(json.dumps(trie), encoding="utf-8")
    return sids, trie, cat, x


def report(sids: pd.DataFrame, cat: pd.DataFrame, x: torch.Tensor) -> None:
    s = stats(sids)
    ok_c = s["collision_rate"] <= MAX_COLLISION_RATE
    ok_m = s["max_tracks_per_code"] <= MAX_TRACKS_PER_CODE
    print(f"tracks {s['tracks']:,} | distinct (a,b,c) codes {s['codes_used']:,} | AUDIO_WEIGHT {AUDIO_WEIGHT}")
    print(f"collision rate (tracks sharing an (a,b,c) code): {s['collision_rate']:.1%}  "
          f"[{'OK' if ok_c else 'FAIL'} <= {MAX_COLLISION_RATE:.0%}]")
    print(f"max tracks per code: {s['max_tracks_per_code']}  [{'OK' if ok_m else 'FAIL'} <= {MAX_TRACKS_PER_CODE}]"
          f" | max dedupe digit d: {s['max_d']}")
    per_code = sids.groupby(["a", "b", "c"]).size()
    print("tracks-per-code histogram:", per_code.value_counts().sort_index().to_dict())
    l1 = s["l1_sizes"]
    print(f"\nlevel-1 clusters used: {s['l1_used']}/{SID_K} | size min {s['l1_min']} / median {s['l1_median']:.0f}"
          f" / mean {s['l1_mean']:.1f} / max {s['l1_max']}")
    q = l1.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round().astype(int).to_dict()
    print("level-1 size quantiles:", {f"p{int(k * 100)}": v for k, v in q.items()})
    bins = pd.cut(l1, [0, 10, 20, 30, 40, 60, 80, 120, 10_000]).value_counts().sort_index()
    print("level-1 size buckets:", {str(k): int(v) for k, v in bins.items()})
    print("10 largest level-1 clusters:", l1.head(10).to_dict())

    pu = purity_report(sids, cat)
    print(f"\nlevel-1 purity (unweighted mean over {s['l1_used']} clusters):")
    print(f"  top primary_genre share: {pu['level1']['genre']:.3f}  | random same-size clusters: {pu['random']['genre']:.3f}")
    print(f"  top context word share:  {pu['level1']['context']:.3f}  | random same-size clusters: {pu['random']['context']:.3f}")
    ps = prefix_similarity(x, sids)
    print(f"\nprefix similarity (mean cosine, option B vectors, {PREFIX_PAIRS} pairs each):")
    print(f"  share (a): {ps['share a']:.3f} | share (a,b): {ps['share a,b']:.3f} | random: {ps['random']:.3f}"
          f" (n={ps['n_random']})")

    m = sids.merge(cat[["track_id", "content_desc", "primary_genre"]], on="track_id")
    rng = np.random.default_rng(SAMPLE_RANDOM_STATE)
    for a in sorted(rng.choice(l1.index.to_numpy(), 5, replace=False)):
        g = m[m.a == a]
        top = g.primary_genre.value_counts().head(3).to_dict()
        print(f"\n-- level-1 cluster {a} ({len(g)} tracks; top genres {top})")
        for d in g.sample(min(5, len(g)), random_state=SAMPLE_RANDOM_STATE).content_desc:
            print("  *", d)


if __name__ == "__main__":
    sids, _, cat, x = build()
    sys.exit(report(sids, cat, x))
