"""Innsbruck Spotify Playlists dataset (Zenodo 2594557): match to our Spotify tracks.

One streaming pass over the 1.2 GB CSV inside the zip (chunks). Exact match on
normalized (track_name, first artist). A normalized pair that maps to several of
our Spotify track_ids goes to the most popular one (ties: track_id).

Outputs (data/raw/playlists/, NOT committed):
  matched.parquet            user_id, playlistname, track_id   (matched rows only)
  context_playlists.parquet  user_id, playlistname, words, is_album   (names with a context phrase)
  pass_stats.json            row / user / playlist / malformed counts

Playlist names are evaluation ground truth ONLY: never use them in desc, tags or training queries.
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path

import pandas as pd

from src.muse_match import normalize_artist, normalize_name, normalize_title
from src.paths import PLAYLIST_CONTEXT, PLAYLIST_MATCHED, PLAYLIST_STATS, PLAYLIST_ZIP
from src.spotify import build_tracks, load_raw

CHUNK = 500_000

# Context phrase -> regex (word boundaries, common inflections).
CONTEXT_PHRASES = {
    "study": r"stud(?:y|ying|ies)", "focus": r"focus(?:ed|ing)?", "sleep": r"sleep(?:ing|y)?",
    "workout": r"work ?outs?", "gym": r"gym", "party": r"part(?:y|ies|ying)",
    "chill": r"chill(?:in|ing|out|ed)?", "relax": r"relax(?:ing|ation|ed)?", "sad": r"sad(?:ness)?",
    "rain": r"rain(?:y|ing)?", "morning": r"mornings?", "drive": r"driv(?:e|ing)",
    "coffee": r"coffee", "late night": r"late[ -]?nights?", "night drive": r"night[ -]?driv(?:e|es|ing)",
    "running": r"running", "run playlist": r"run playlist",
}
PHRASE_RE = {w: re.compile(rf"\b{p}\b", re.I) for w, p in CONTEXT_PHRASES.items()}

# Album-name rule: "<artist> – <title>" (en dash, em dash or spaced hyphen) where <artist> is a known artist.
ALBUM_SEPARATORS = re.compile(r" – | — | - ")
# Cleaned context playlist: at least this many matched tracks from at least this many distinct artists.
MIN_PLAYLIST_TRACKS = 5
MIN_PLAYLIST_ARTISTS = 3


def phrases_in(name: str) -> list[str]:
    return [w for w, r in PHRASE_RE.items() if r.search(name)]


def is_album_name(name: str, artist_set: set[str]) -> bool:
    parts = ALBUM_SEPARATORS.split(name, maxsplit=1)
    return len(parts) == 2 and normalize_name(parts[0]) in artist_set


def spotify_key_index(tracks: pd.DataFrame) -> tuple[dict, set]:
    """Normalized key -> most popular track_id. Returns (mapping, keys that had >1 track_id)."""
    k = pd.DataFrame({"track_id": tracks.track_id, "popularity": tracks.popularity,
                      "k": list(zip(tracks.track_name.map(normalize_title),
                                    tracks.artists.map(normalize_artist)))})
    multi = k.groupby("k").track_id.nunique()
    k = k.sort_values(["popularity", "track_id"], ascending=[False, True]).drop_duplicates("k")
    return dict(zip(k.k, k.track_id)), set(multi[multi > 1].index)


def run_pass() -> dict:
    t0 = time.time()
    tracks = build_tracks(load_raw())
    key2id, multi_keys = spotify_key_index(tracks)
    print(f"[{time.time()-t0:.0f}s] keys={len(key2id):,} (multi-track keys -> most popular: {len(multi_keys):,})", flush=True)

    norm_cache: dict = {}
    users, playlists, raw_artists = set(), set(), set()
    n_rows = n_null_user = n_matched = n_multi_rows = n_malformed = 0
    parts = []
    with zipfile.ZipFile(PLAYLIST_ZIP) as z, z.open("spotify_dataset.csv") as f, \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader = pd.read_csv(f, escapechar="\\", skipinitialspace=True, chunksize=CHUNK,
                             on_bad_lines="warn", dtype=str, encoding="utf-8", encoding_errors="replace")
        for i, df in enumerate(reader):
            df.columns = [c.strip().strip('"') for c in df.columns]
            n_null_user += int(df.user_id.isna().sum())
            df = df.dropna(subset=["user_id"])
            n_rows += len(df)
            df["playlistname"] = df.playlistname.fillna("")
            users.update(df.user_id.unique())
            playlists.update(zip(df.user_id, df.playlistname))
            raw_artists.update(df.artistname.dropna().unique())
            keys = []
            for p in zip(df.trackname.fillna(""), df.artistname.fillna("")):
                k = norm_cache.get(p)
                if k is None:
                    k = norm_cache[p] = (normalize_title(p[0]), normalize_artist(p[1]))
                keys.append(k)
            df["track_id"] = [key2id.get(k) for k in keys]
            n_multi_rows += sum(k in multi_keys for k in keys)
            m = df.loc[df.track_id.notna(), ["user_id", "playlistname", "track_id"]]
            n_matched += len(m)
            parts.append(m)
            print(f"[{time.time()-t0:.0f}s] chunk {i}: rows={n_rows:,} matched={n_matched:,}", flush=True)
        for w in caught:
            n_malformed += str(w.message).count("Skipping line")

    matched = pd.concat(parts, ignore_index=True).drop_duplicates()
    matched.to_parquet(PLAYLIST_MATCHED, index=False)

    artist_set = {normalize_name(a) for a in raw_artists} | {normalize_artist(a) for a in raw_artists}
    artist_set |= {normalize_name(a) for a in tracks.artists.str.replace(";", ", ")} | {"various artists"}
    artist_set.discard("")
    ctx = pd.DataFrame(sorted(playlists), columns=["user_id", "playlistname"])
    names = pd.Series(ctx.playlistname.unique())
    words = dict(zip(names, names.map(phrases_in)))
    ctx["words"] = ctx.playlistname.map(words)
    ctx = ctx[ctx.words.str.len() > 0].copy()
    album = {n: is_album_name(n, artist_set) for n in ctx.playlistname.unique()}
    ctx["is_album"] = ctx.playlistname.map(album)
    ctx.to_parquet(PLAYLIST_CONTEXT, index=False)

    stats = {"rows": n_rows, "users": len(users), "playlists": len(playlists),
             "malformed_lines_skipped": n_malformed, "rows_null_user_dropped": n_null_user,
             "matched_rows": n_matched, "matched_rows_dedup": len(matched),
             "rows_on_multi_track_keys": n_multi_rows, "multi_track_keys": len(multi_keys),
             "context_named_playlists": len(ctx), "context_album_named": int(ctx.is_album.sum()),
             "seconds": round(time.time() - t0)}
    PLAYLIST_STATS.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def clean_context_playlists(matched: pd.DataFrame, ctx: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Drop album-named playlists; keep those with >= MIN_PLAYLIST_TRACKS matched tracks
    from >= MIN_PLAYLIST_ARTISTS distinct (first) artists. Returns ctx rows + n_tracks, n_artists."""
    ctx = ctx[~ctx.is_album]
    first_artist = tracks.set_index("track_id").artists.map(normalize_artist)
    m = matched.merge(ctx[["user_id", "playlistname"]], on=["user_id", "playlistname"])
    m = m.assign(artist=m.track_id.map(first_artist))
    agg = m.groupby(["user_id", "playlistname"]).agg(n_tracks=("track_id", "nunique"),
                                                     n_artists=("artist", "nunique")).reset_index()
    agg = agg[(agg.n_tracks >= MIN_PLAYLIST_TRACKS) & (agg.n_artists >= MIN_PLAYLIST_ARTISTS)]
    return ctx.merge(agg, on=["user_id", "playlistname"])


def context_track_labels(matched: pd.DataFrame, clean_ctx: pd.DataFrame) -> pd.DataFrame:
    """track_id, context_word, playlist_count (number of cleaned context playlists)."""
    m = matched.merge(clean_ctx[["user_id", "playlistname", "words"]], on=["user_id", "playlistname"])
    m = m.explode("words").rename(columns={"words": "context_word"})
    return (m.drop_duplicates(["user_id", "playlistname", "track_id", "context_word"])
             .groupby(["track_id", "context_word"]).size().rename("playlist_count").reset_index())


if __name__ == "__main__":
    sys.exit(run_pass() and None)
