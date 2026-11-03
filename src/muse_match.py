"""Join MuSe valence/arousal scores onto Spotify tracks, and validate the join.

Pass 1: exact spotify_id == track_id (duplicate MuSe rows per id: scores averaged, tag counts summed).
Pass 2 (only for tracks without an ID match): exact match on normalized
(track_name, first artist). No fuzzy matching. Normalized pairs that map to
MuSe rows with different scores are skipped as ambiguous.
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from src.paths import MUSE_CSV

# Version suffixes stripped from titles, either " - <kw>..." or "(<kw>...)" / "[<kw>...]".
SUFFIX_KEYWORDS = r"remaster|live|acoustic|feat|ft\.|with |radio edit|single version|album version|mono|stereo"
_DASH_SUFFIX = re.compile(rf"\s+-\s+(?:\d{{4}}\s+)?(?:{SUFFIX_KEYWORDS}).*$")
_BRACKET_SUFFIX = re.compile(rf"\s*[\(\[]\s*(?:\d{{4}}\s+)?(?:{SUFFIX_KEYWORDS})[^\)\]]*[\)\]]")
_FEAT_TAIL = re.compile(r"\s+(?:feat\.?|ft\.|featuring)\s+.*$")
# First-artist split: ';', ',', '&', ' and ', ' feat.', ' ft.'
_ARTIST_SPLIT = re.compile(r";|,|&|\s+and\s+|\s+feat\.?\s+|\s+ft\.\s+|\s+featuring\s+")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def _clean(s: str) -> str:
    s = _PUNCT.sub(" ", s).replace("_", " ")
    return _SPACES.sub(" ", s).strip()


def normalize_title(s: str) -> str:
    s = str(s).lower().strip()
    s = _BRACKET_SUFFIX.sub("", s)
    s = _DASH_SUFFIX.sub("", s)
    s = _FEAT_TAIL.sub("", s)
    return _clean(s)


def normalize_artist(s: str) -> str:
    """First artist only, normalized."""
    return _clean(_ARTIST_SPLIT.split(str(s).lower().strip(), maxsplit=1)[0])


def normalize_name(s: str) -> str:
    """Whole string normalized (no artist splitting); used for album-name detection."""
    return _clean(str(s).lower().strip())


def load_muse() -> pd.DataFrame:
    return pd.read_csv(MUSE_CSV)


def match_muse(tracks: pd.DataFrame, mu: pd.DataFrame) -> pd.DataFrame:
    """One row per matched Spotify track_id: track_id, muse_valence, muse_arousal,
    muse_n_tags, muse_match, muse_track, muse_artist. Several track_ids may share a MuSe entry."""
    by_id = (mu.dropna(subset=["spotify_id"])
               .groupby("spotify_id")
               .agg(muse_valence=("valence_tags", "mean"),
                    muse_arousal=("arousal_tags", "mean"),
                    muse_n_tags=("number_of_emotion_tags", "sum"),
                    muse_track=("track", "first"),
                    muse_artist=("artist", "first")))
    id_hits = tracks[["track_id"]].merge(by_id, left_on="track_id", right_index=True)
    id_hits["muse_match"] = "id"

    rest = tracks[~tracks.track_id.isin(id_hits.track_id)].copy()
    rest["k_title"] = rest.track_name.map(normalize_title)
    rest["k_artist"] = rest.artists.map(normalize_artist)

    m = mu.dropna(subset=["track", "artist"]).copy()
    m["k_title"] = m.track.map(normalize_title)
    m["k_artist"] = m.artist.map(normalize_artist)
    m = m[(m.k_title != "") & (m.k_artist != "")]
    g = m.groupby(["k_title", "k_artist"])
    stats = g.agg(muse_valence=("valence_tags", "first"), muse_arousal=("arousal_tags", "first"),
                  muse_n_tags=("number_of_emotion_tags", "first"),
                  muse_track=("track", "first"), muse_artist=("artist", "first"))
    n_scores = g.apply(lambda d: len(set(zip(d.valence_tags, d.arousal_tags))), include_groups=False)
    ambiguous = n_scores[n_scores > 1].index
    stats = stats[n_scores == 1]

    name_hits = rest.merge(stats, left_on=["k_title", "k_artist"], right_index=True)
    name_hits["muse_match"] = "name"

    cols = ["track_id", "muse_valence", "muse_arousal", "muse_n_tags", "muse_match", "muse_track", "muse_artist"]
    out = pd.concat([id_hits[cols], name_hits[cols]], ignore_index=True)
    out.attrs["n_ambiguous_keys"] = len(ambiguous)
    return out


def validate(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Pearson r: muse_valence vs Spotify valence, muse_arousal vs Spotify energy, per match type."""
    rows = []
    for name, d in [("id", df[df.muse_match == "id"]), ("name", df[df.muse_match == "name"]),
                    ("combined", df[df.muse_match.notna()])]:
        rows.append({"scope": scope, "match": name, "n": len(d),
                     "r_valence": round(d.muse_valence.corr(d.valence), 3),
                     "r_arousal_energy": round(d.muse_arousal.corr(d.energy), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    from src.spotify import build_tracks, load_raw
    tracks = build_tracks(load_raw())
    res = match_muse(tracks, load_muse())
    print(res.muse_match.value_counts().to_string())
    print(validate(res.merge(tracks[["track_id", "valence", "energy"]], on="track_id"), "all").to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
