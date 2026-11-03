"""Step 4: build data/proc/catalog.parquet (+ context_labels.parquet, real_users.json, muse_validation.csv).

Requires the cached playlist pass (python -m src.playlists).
Playlist names are evaluation ground truth only: they never reach desc, tags or context.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from src.config import canonical, desc_categories
from src.muse_match import load_muse, match_muse, normalize_artist, normalize_title, song_key, validate
from src.paths import (CATALOG, CONTEXT_LABELS, MUSE_VALIDATION, PLAYLIST_CONTEXT, PLAYLIST_MATCHED,
                       PLAYLIST_STATS, PROC, REAL_USERS, RULE_PRECISION, LABEL_SPLIT_FROZEN)
from src.playlists import clean_context_playlists, context_track_labels
from src.spotify import build_tracks, load_raw

# ---- catalog composition
CATALOG_SIZE = 10_000
MAX_GENRE_FRAC = 0.10          # cap per primary_genre
PLAYLIST_STAGE_LIMIT = 8_500   # stages (a)+(b)+(c) fill at most this many slots; rest = genre fill
FILL_RANDOM_STATE = 0
DESC_SAMPLE_RANDOM_STATE = 1

# ---- real users
REAL_USER_MIN_TRACKS = 20
REAL_USER_HISTORY_FRAC = 0.8
REAL_USER_RANDOM_STATE = 4
REAL_USER_MAX_HISTORY = 200

# ---- context label tune/eval split (eval half never used for rule tuning)
LABEL_SPLIT_FRAC = 0.5
LABEL_SPLIT_RANDOM_STATE = 5

# ---- rule-based tag thresholds
TEMPO_SLOW, TEMPO_FAST = 90, 130
SLOW_MAX_ENERGY, FAST_MIN_ENERGY = 0.5, 0.6   # tempo word only if energy agrees (half/double-time guard)
ENERGY_CALM, ENERGY_HIGH = 0.35, 0.7
VALENCE_HAPPY, VALENCE_SAD = 0.6, 0.3
INSTRUMENTAL_MIN = 0.5
ACOUSTIC_MIN = 0.6
DANCEABLE_MIN = 0.7
SPEECH_MIN = 0.33
WORKOUT_MIN_ENERGY = 0.8
MAX_CONTEXT_WORDS = 4   # keep words from highest-lift rules first (lift from TUNE half)

# (rule name, required tags, excluded tags, min energy or None) -> context tags
CONTEXT_RULES = [
    ("calm+instrumental", {"calm", "instrumental"}, set(), None, ["study", "focus", "late night"]),
    ("calm+acoustic", {"calm", "acoustic"}, set(), None, ["relaxing", "morning", "coffee"]),
    ("calm+slow", {"calm", "slow"}, set(), None, ["sleep", "wind down"]),
    ("energetic+danceable", {"energetic", "danceable"}, set(), None, ["party", "dance"]),
    ("energetic+fast", {"energetic", "fast"}, set(), None, ["running", "gym"]),
    ("melancholic+slow", {"melancholic", "slow"}, set(), None, ["rainy day", "reflective"]),
    ("happy+danceable", {"happy", "danceable"}, set(), None, ["dance", "feel-good"]),
    ("happy+moderate", {"happy", "moderate energy"}, set(), None, ["morning", "feel-good"]),
    ("acoustic", {"acoustic"}, set(), None, ["coffee", "relaxing"]),
    ("workout(energetic,!danceable,!melancholic,energy>0.8)", {"energetic"}, {"danceable", "melancholic"},
     WORKOUT_MIN_ENERGY, ["workout"]),
    ("energetic+melancholic", {"energetic", "melancholic"}, set(), None, ["intense", "cathartic"]),
]
DEFAULT_RULE_PRIORITY = [r[0] for r in CONTEXT_RULES]
# context tag -> playlist words that count as a hit (evaluation only; None = no playlist equivalent)
CONTEXT_TO_PLAYLIST_WORDS = {
    "study": ["study"], "focus": ["focus", "study"], "late night": ["late night", "night drive"],
    "relaxing": ["relax", "chill"], "sleep": ["sleep"],
    "wind down": ["sleep", "relax", "chill"], "party": ["party"], "workout": ["workout", "gym", "running"],
    "running": ["running", "run playlist"], "gym": ["gym", "workout"], "rainy day": ["rain"],
    "reflective": ["sad", "rain"], "dance": ["party"], "feel-good": None, "morning": ["morning"],
    "coffee": ["coffee"], "intense": None, "cathartic": None,
}
# ---- non-music noise filter (Step A, approved): applied to the selected catalog AND every fill candidate
NOISE_PATTERNS = ["white noise", "pink noise", "brown noise", "vacuum", "air conditioner", "fan noise",
                  "rain sounds", "nature sounds", "ocean waves", "thunderstorm", "loopable", "sleep rain",
                  "healing rain"]                          # case-insensitive substring of title or artists
NOISE_ARTISTS = ["Nature Sounds", "Rain Sounds", "White Noise", "Loopable", "Granular", "Baby Sleep",
                 "Sleep Sounds", "Rain Sounds Lab", "Rain Sounds XLE Library", "Meditation Spa"]  # exact artist
NOISE_MAX_ENERGY = 0.01                                   # with speechiness == 0
# genre fill stops adding to these genres after N fill tracks (Step A, approved): the category-only "sleep"
# pool is mostly noise the filter can't catch, so sleep keeps only the 11 non-noise fill tracks it already had
FILL_MAX_ADDED = {"sleep": 11}
NOISE_SPARE_IDS = frozenset({"5xw2cHVLw1rlDPp3cL9Zuv"})   # "She's Thunderstorms" - Arctic Monkeys
_NOISE_RE = re.compile("|".join(re.escape(p) for p in NOISE_PATTERNS), re.IGNORECASE)
_NOISE_ARTISTS = {a.casefold() for a in NOISE_ARTISTS}

AUDIO_FEATURES = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence", "tempo", "time_signature", "duration_ms", "explicit"]


def is_noise(df: pd.DataFrame) -> pd.Series:
    """Non-music noise: title/artists match NOISE_PATTERNS, any artist is on NOISE_ARTISTS, or
    energy < NOISE_MAX_ENERGY with speechiness == 0. NOISE_SPARE_IDS are never noise."""
    text = df.track_name.fillna("") + " " + df.artists.fillna("")
    kw = text.str.contains(_NOISE_RE)
    blocked = df.artists.fillna("").map(lambda a: any(x.strip().casefold() in _NOISE_ARTISTS for x in a.split(";")))
    audio = (df.energy < NOISE_MAX_ENERGY) & (df.speechiness == 0)
    ids = df.track_id if "track_id" in df.columns else pd.Series(df.index, index=df.index)
    return (kw | blocked | audio) & ~ids.isin(NOISE_SPARE_IDS)


def tempo_missing(tempo) -> bool:
    """Spotify reports tempo 0 when it could not be estimated."""
    return not tempo > 0


def make_tags(r) -> list[str]:
    t = []
    if tempo_missing(r.tempo):
        pass                                     # no tempo word
    elif r.tempo < TEMPO_SLOW and r.energy < SLOW_MAX_ENERGY:
        t.append("slow")
    elif r.tempo > TEMPO_FAST and r.energy > FAST_MIN_ENERGY:
        t.append("fast")
    t += ["calm" if r.energy < ENERGY_CALM else "energetic" if r.energy > ENERGY_HIGH else "moderate energy",
          "happy" if r.valence > VALENCE_HAPPY else "melancholic" if r.valence < VALENCE_SAD else "neutral mood"]
    if r.instrumentalness > INSTRUMENTAL_MIN: t.append("instrumental")
    if r.acousticness > ACOUSTIC_MIN: t.append("acoustic")
    if r.danceability > DANCEABLE_MIN: t.append("danceable")
    if r.speechiness > SPEECH_MIN and "instrumental" not in t: t.append("spoken word or rap")
    return t


def rule_fires(rule, tags, energy) -> bool:
    _, need, excl, min_energy, _ = rule
    s = set(tags)
    return need <= s and not (excl & s) and (min_energy is None or energy > min_energy)


def make_context(tags: list[str], energy: float = 1.0, priority: list[str] | None = None) -> list[str]:
    """Round-robin cap: first the lead word of EACH fired rule (rules in priority order), then the
    remaining words by priority order. Synonyms mapped to canonical words (src.config), deduped,
    at most MAX_CONTEXT_WORDS."""
    rules = {r[0]: r for r in CONTEXT_RULES}
    fired = [[canonical(w) for w in rules[n][4]]
             for n in (priority or DEFAULT_RULE_PRIORITY) if rule_fires(rules[n], tags, energy)]
    out = []
    for words in fired:                      # pass 1: lead word per rule
        if words[0] not in out:
            out.append(words[0])
    for words in fired:                      # pass 2: fill by priority
        out += [w for w in words[1:] if w not in out]
    return out[:MAX_CONTEXT_WORDS]


def lift_priority(prec: pd.DataFrame) -> list[str]:
    """Rule names by TUNE-half lift, descending; rules without lift go last (stable)."""
    p = prec.assign(_o=range(len(prec)), _nan=prec.lift.isna())
    return p.sort_values(["_nan", "lift", "_o"], ascending=[True, False, True]).rule.tolist()


def split_labels(labels: pd.DataFrame, frozen: pd.Series | None = None) -> pd.DataFrame:
    """By track_id into label_split = tune | eval. Tracks in `frozen` (track_id -> split) keep their split,
    so tracks already used for rule tuning never move to eval. New tracks are split 50/50
    (random_state LABEL_SPLIT_RANDOM_STATE)."""
    frozen = frozen if frozen is not None else pd.Series(dtype=str)
    ids = pd.Series(sorted(labels.track_id.unique()))
    new = ids[~ids.isin(frozen.index)].reset_index(drop=True)
    tune = set(frozen.index[frozen == "tune"]) | set(new.sample(frac=LABEL_SPLIT_FRAC,
                                                                   random_state=LABEL_SPLIT_RANDOM_STATE))
    return labels.assign(label_split=labels.track_id.map(lambda t: "tune" if t in tune else "eval"))


def load_frozen_split() -> pd.Series | None:
    """Split frozen at Step 5 (data/proc/label_split_frozen.csv); None if absent."""
    if not LABEL_SPLIT_FROZEN.exists():
        return None
    f = pd.read_csv(LABEL_SPLIT_FROZEN)
    return f.set_index("track_id").label_split


def rule_precision(cat: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """TUNE half only. Per rule: tracks tagged, precision (tagged tracks whose playlist labels hit a
    mapped word), base rate of those words among all tune tracks, lift."""
    lab = labels[labels.label_split == "tune"]
    words = lab.groupby("track_id").context_word.agg(set)
    tc = cat.set_index("track_id").loc[words.index, ["tags", "energy"]]
    rows = []
    for rule in CONTEXT_RULES:
        name, ctx = rule[0], rule[4]
        target = sorted({w for c in ctx for w in (CONTEXT_TO_PLAYLIST_WORDS.get(c) or [])})
        fired = pd.Series([rule_fires(rule, t, e) for t, e in zip(tc.tags, tc.energy)], index=tc.index)
        hit = words.map(lambda ws: bool(ws & set(target)))
        n = int(fired.sum())
        prec = hit[fired].mean() if n and target else float("nan")
        base = hit.mean() if target else float("nan")
        rows.append({"rule": name, "context": ", ".join(ctx), "playlist_words": ", ".join(target) or "-",
                     "tracks_tagged": n, "precision": round(prec, 3), "base_rate": round(base, 3),
                     "lift": round(prec / base, 2) if base == base and base else float("nan")})
    out = pd.DataFrame(rows)
    out.attrs["n_tune_tracks"] = len(words)
    return out


def make_desc(r) -> str:
    parts = [f"{r.track_name} by {r.artists.replace(';', ', ')}."]
    if len(r.genres):
        parts.append(f"Genre: {', '.join(r.genres)}.")
    shown = desc_categories(r.category, r.tags)      # contradicted categories left out of desc
    if shown:
        parts.append(f"Spotify category: {', '.join(shown)}.")
    parts.append(f"Sound: {', '.join(r.tags)}.")
    parts.append(f"Good for: {', '.join(r.context) or 'general listening'}.")
    return " ".join(parts)


def dedupe_muse(tracks: pd.DataFrame, muse: pd.DataFrame) -> pd.DataFrame:
    """One track_id per MuSe entry: prefer id match, then higher popularity."""
    m = muse.merge(tracks[["track_id", "popularity"]], on="track_id")
    m["_id_first"] = (m.muse_match != "id").astype(int)
    m = m.sort_values(["_id_first", "popularity", "track_id"], ascending=[True, False, True])
    return m.drop_duplicates(["muse_track", "muse_artist"]).drop(columns=["_id_first"])


def select_catalog(tracks, muse, ctx_track_ids, n_users, size=CATALOG_SIZE, max_frac=MAX_GENRE_FRAC,
                   stage_limit=PLAYLIST_STAGE_LIMIT, random_state=FILL_RANDOM_STATE, fill_max_added=None):
    """Priority: (a) MuSe, (b) cleaned context playlists, (c) other playlist-matched (most users first),
    together limited to stage_limit slots; then (d) genre fill that always tops up the currently
    smallest primary_genre (underrepresented first). Every stage respects the per-genre cap.
    Songs are deduped by song_key (accent-stripped title without version suffixes, first artist): after (a)-(c), duplicates are dropped keeping
    stage priority then popularity; freed slots go to the genre fill, which also skips known songs.
    Non-music noise (is_noise) is dropped from (a)-(c) and skipped by the fill; the fill adds at most
    fill_max_added[g] tracks to genre g (default FILL_MAX_ADDED). Returns (catalog, cap_log, n_dupes_dropped)."""
    cap = int(size * max_frac)
    t = tracks.set_index("track_id").copy()
    t["song_key"] = [song_key(n, a) for n, a in zip(t.track_name, t.artists)]
    # legacy key (Step 4.6) only fixes the genre-fill shuffle order, so a stricter song_key does not reshuffle it
    t["order_key"] = list(zip(t.track_name.map(normalize_title), t.artists.map(normalize_artist)))
    nu = n_users.reindex(t.index).fillna(0)
    a = muse.track_id.tolist()                      # already id-first, popularity-desc
    taken = set(a)
    key = pd.DataFrame({"nu": nu, "pop": t.popularity})
    b_ids = [i for i in key.loc[list(ctx_track_ids - taken)].sort_values(
        ["nu", "pop"], ascending=False).index]
    taken |= set(b_ids)
    c_ids = [i for i in key.loc[key.index[(nu > 0) & ~key.index.isin(list(taken))]].sort_values(
        ["nu", "pop"], ascending=False).index]
    chosen, source, per_genre, cap_log = [], [], {}, {}
    genre = t.primary_genre
    limit = min(stage_limit, size)
    for stage, ids in [("muse", a), ("context_playlist", b_ids), ("playlist", c_ids)]:
        for i in ids:
            if len(chosen) >= limit:
                break
            g = genre[i]
            if per_genre.get(g, 0) >= cap:
                cap_log[(stage, g)] = cap_log.get((stage, g), 0) + 1
                continue
            per_genre[g] = per_genre.get(g, 0) + 1
            chosen.append(i); source.append(stage)

    # song dedupe among (a)-(c): stage priority, then popularity
    stage_rank = {"muse": 0, "context_playlist": 1, "playlist": 2}
    pick = pd.DataFrame({"track_id": chosen, "source": source})
    pick["rank"] = pick.source.map(stage_rank)
    pick["pop"] = t.popularity.loc[pick.track_id].to_numpy()
    pick["song_key"] = t.song_key.loc[pick.track_id].to_numpy()
    pick["order_key"] = t.order_key.loc[pick.track_id].to_numpy()
    ordered = pick.sort_values(["rank", "pop", "track_id"], ascending=[True, False, True])
    keep = ordered.drop_duplicates("song_key")
    legacy_keep = ordered.drop_duplicates("order_key")
    n_dupes = len(pick) - len(keep)
    for g in genre.loc[pick.track_id[~pick.track_id.isin(keep.track_id)]]:
        per_genre[g] -= 1
    pick = pick[pick.track_id.isin(keep.track_id)]
    # non-music noise out of (a)-(c); freed slots go to the genre fill (which also skips noise)
    noise = is_noise(t)
    noisy = noise.loc[pick.track_id].to_numpy()
    noise_dropped = pick.source[noisy].value_counts().to_dict()
    for g in genre.loc[pick.track_id[noisy]]:
        per_genre[g] -= 1
    pick = pick[~noisy]
    chosen, source = pick.track_id.tolist(), pick.source.tolist()
    songs = set(t.song_key.loc[chosen])

    # (d) genre fill: always add to the currently smallest genre; random order within genre; new songs only.
    # The shuffle is drawn over the legacy-key pool (unchanged since Step 4.6); song_key duplicates are skipped below.
    legacy_songs = set(legacy_keep.order_key)
    pool = (t.loc[~t.index.isin(legacy_keep.track_id) & ~t.order_key.isin(legacy_songs)]
            .sample(frac=1, random_state=random_state))
    queues = {g: list(ix) for g, ix in pool.groupby("primary_genre", sort=True).groups.items()}
    chosen_set = set(chosen)
    for g in genre.unique():
        per_genre.setdefault(g, 0)
    max_added = FILL_MAX_ADDED if fill_max_added is None else fill_max_added
    added = dict.fromkeys(per_genre, 0)
    while len(chosen) < size:
        open_g = [g for g in sorted(per_genre) if queues.get(g) and per_genre[g] < cap
                  and added[g] < max_added.get(g, size)]
        if not open_g:
            break
        g = min(open_g, key=lambda x: per_genre[x])
        i = queues[g].pop(0)
        if t.song_key[i] in songs or i in chosen_set:
            continue
        if noise[i]:
            noise_dropped["genre_fill_skipped"] = noise_dropped.get("genre_fill_skipped", 0) + 1
            continue
        songs.add(t.song_key[i])
        chosen.append(i); source.append("genre_fill"); per_genre[g] += 1; added[g] += 1
    cat = t.loc[chosen].drop(columns=["song_key", "order_key"]).reset_index()
    cat["source"] = source
    cat = cat.merge(muse[["track_id", "muse_valence", "muse_arousal", "muse_n_tags", "muse_match"]],
                    on="track_id", how="left")
    cat.attrs["noise_dropped"] = noise_dropped
    return cat, cap_log, n_dupes


def split_real_users(matched: pd.DataFrame, catalog_ids: set) -> pd.DataFrame:
    """Long format: user_id, track_id, split (history|heldout). 80/20 split, then history capped."""
    m = matched[matched.track_id.isin(catalog_ids)]
    per_user = m.groupby("user_id").track_id.agg(lambda s: sorted(set(s)))
    per_user = per_user[per_user.str.len() >= REAL_USER_MIN_TRACKS].sort_index()
    rng = np.random.default_rng(REAL_USER_RANDOM_STATE)
    out = []
    for uid, ids in per_user.items():
        perm = [ids[j] for j in rng.permutation(len(ids))]
        k = int(round(REAL_USER_HISTORY_FRAC * len(ids)))
        out += [(uid, tid, "history") for tid in perm[:min(k, REAL_USER_MAX_HISTORY)]]
        out += [(uid, tid, "heldout") for tid in perm[k:]]
    return pd.DataFrame(out, columns=["user_id", "track_id", "split"])


def build() -> dict:
    tracks = build_tracks(load_raw())
    muse_all = match_muse(tracks, load_muse())
    muse = dedupe_muse(tracks, muse_all)

    matched = pd.read_parquet(PLAYLIST_MATCHED)
    clean_ctx = clean_context_playlists(matched, pd.read_parquet(PLAYLIST_CONTEXT), tracks)
    labels_all = context_track_labels(matched, clean_ctx)
    n_users = matched.groupby("track_id").user_id.nunique()

    cat, cap_log, n_dupes = select_catalog(tracks, muse, set(labels_all.track_id), n_users)
    noise_dropped = cat.attrs.get("noise_dropped", {})
    cat["tags"] = cat.apply(make_tags, axis=1)     # tempo 0 = missing -> no tempo word
    cat["tempo_missing"] = cat.tempo.map(tempo_missing)
    cat.loc[cat.tempo_missing, "tempo"] = cat.tempo[~cat.tempo_missing].median()   # audio-vector input
    cat_ids = set(cat.track_id)
    labels = split_labels(labels_all[labels_all.track_id.isin(cat_ids)].reset_index(drop=True),
                          load_frozen_split())
    prec = rule_precision(cat, labels)          # TUNE half only
    priority = lift_priority(prec)
    cat["context"] = [make_context(t, e, priority) for t, e in zip(cat.tags, cat.energy)]
    cat["desc"] = cat.apply(make_desc, axis=1)
    cols = ["track_id", "track_name", "artists", "primary_genre", "genres", "category", "tags", "context", "desc",
            "source", "muse_match", "muse_valence", "muse_arousal", "muse_n_tags", *AUDIO_FEATURES, "tempo_missing", "popularity"]
    cat = cat[cols]
    users = split_real_users(matched, cat_ids)

    feats = tracks[["track_id", "valence", "energy"]]
    val = pd.concat([validate(muse_all.merge(feats, on="track_id"), "all_matches"),
                     validate(cat.dropna(subset=["muse_match"]), "catalog")], ignore_index=True)

    PROC.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(CATALOG, index=False)
    labels.to_parquet(CONTEXT_LABELS, index=False)
    users.to_parquet(REAL_USERS, index=False)
    legacy = PROC / "real_users.json"
    if legacy.exists():
        legacy.unlink()
    prec.to_csv(RULE_PRECISION, index=False)
    val.to_csv(MUSE_VALIDATION, index=False)
    return {"tracks": tracks, "cat": cat, "labels": labels, "users": users, "val": val,
            "cap_log": cap_log, "prec": prec, "priority": priority, "n_dupes": n_dupes, "noise_dropped": noise_dropped, "clean_ctx": clean_ctx, "muse_all": muse_all, "muse": muse}


def report(r: dict) -> None:
    cat, tracks = r["cat"], r["tracks"]
    stats = json.loads(PLAYLIST_STATS.read_text())
    print(f"Unique Spotify tracks: {len(tracks):,} | catalog: {len(cat):,}")
    print(f"MuSe matches (all): {r['muse_all'].muse_match.value_counts().to_dict()} | "
          f"after one-track-per-entry dedupe: {len(r['muse']):,}")
    print("\n-- composition by source"); print(cat.source.value_counts().to_string())
    keys = [song_key(n, a) for n, a in zip(cat.track_name, cat.artists)]
    print(f"song duplicates dropped from stages a-c: {r['n_dupes']:,} | duplicate songs left: "
          f"{len(keys) - len(set(keys)):,}")
    print(f"non-music noise dropped/skipped by stage: {r['noise_dropped']} | "
          f"tempo missing (0 -> catalog median): {int(cat.tempo_missing.sum())}")
    n_cat = (cat.category.str.len() > 0).sum()
    n_hidden = sum(len(desc_categories(c, t)) < len(c) for c, t in zip(cat.category, cat.tags))
    print(f"tracks with category: {n_cat:,} | category value(s) left out of desc as contradictory: {n_hidden:,}")
    print("\n-- MuSe in catalog by match type"); print(cat.muse_match.value_counts(dropna=False).to_string())
    if r["cap_log"]:
        print("\n-- GENRE CAP TRIGGERED (tracks skipped per stage/genre):")
        for (s, g), n in sorted(r["cap_log"].items()): print(f"   {s:16s} {g:15s} {n:,}")
    else:
        print("\n-- genre cap: not triggered")
    print("\n-- primary_genre top 15 (of", cat.primary_genre.nunique(), "genres)")
    print(cat.primary_genre.value_counts().head(15).to_string())
    print("\n-- tag frequency"); print(cat.tags.explode().value_counts().to_string())
    print("\n-- context frequency"); print(cat.context.explode().value_counts().to_string())
    print(f"\n% tracks with no context tag ('general listening'): {(cat.context.str.len() == 0).mean():.1%}")
    print(f"% tracks tagged workout: {cat.context.map(lambda c: 'workout' in c).mean():.1%}")
    n_ctx = cat.context.str.len()
    print(f"context words per track: max {n_ctx.max()} / mean {n_ctx.mean():.2f}")
    print(f"tracks with a Spotify category: {(cat.category.str.len() > 0).sum():,} | "
          f"category-only (primary_genre = category): {(cat.genres.str.len() == 0).sum():,}")
    print("rule priority (tune-half lift):", " > ".join(r["priority"]))
    pr = r["prec"]
    print(f"\n-- rule precision on TUNE half ({pr.attrs['n_tune_tracks']:,} labeled tracks)")
    print(pr.to_string(index=False))
    cc, lab = r["clean_ctx"], r["labels"]
    lab_split = lab.drop_duplicates("track_id").label_split.value_counts().to_dict()
    print(f"\n-- context label split (tracks): {lab_split}")
    print(f"-- context playlists: {stats['context_named_playlists']:,} named, "
          f"{stats['context_album_named']:,} album-named dropped, {len(cc):,} kept after >=5 tracks/>=3 artists")
    per = cc.explode("words").groupby("words").size().rename("playlists").to_frame()
    per["catalog_tracks_labeled"] = lab.groupby("context_word").track_id.nunique()
    print(per.fillna(0).astype(int).sort_values("playlists", ascending=False).to_string())
    u = r["users"]
    sz = u.groupby(["user_id", "split"]).size().unstack(fill_value=0)
    h, ho = sz["history"], sz["heldout"]
    print(f"\n-- real users (>= {REAL_USER_MIN_TRACKS} catalog tracks): {len(sz):,} "
          f"(history capped at {REAL_USER_MAX_HISTORY}; users at cap: {int((h == REAL_USER_MAX_HISTORY).sum()):,})")
    print(f"   history  size: min {h.min()} / median {h.median():.0f} / mean {h.mean():.0f} / max {h.max()}")
    print(f"   heldout  size: min {ho.min()} / median {ho.median():.0f} / mean {ho.mean():.0f} / max {ho.max()}")
    print(f"\n-- playlist CSV: malformed lines skipped: {stats['malformed_lines_skipped']:,}, "
          f"null-user rows dropped: {stats['rows_null_user_dropped']:,}")
    print("\n-- MuSe validation (Pearson r)"); print(r["val"].to_string(index=False))
    print("\n-- 8 random descs (random_state=1)")
    for d in cat.desc.sample(8, random_state=DESC_SAMPLE_RANDOM_STATE): print("  *", d)
    sizes = {p.name: p.stat().st_size / 1e6 for p in PROC.glob("*") if p.is_file()}
    print(f"\n-- data/proc sizes (MB): {', '.join(f'{k}={v:.1f}' for k, v in sizes.items())} "
          f"| total {sum(sizes.values()):.1f}")


if __name__ == "__main__":
    sys.exit(report(build()))
