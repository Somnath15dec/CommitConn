"""Load and dedupe the Spotify tracks dataset."""
from __future__ import annotations

import pandas as pd

from src.paths import SPOTIFY_CSV

# Spotify "genres" that are moods/activities, not musical genres. Moved to `category`.
# Category must never fill eval templates.
CATEGORY_GENRES = frozenset({"chill", "happy", "party", "romance", "sad", "sleep", "study"})


def load_raw() -> pd.DataFrame:
    return pd.read_csv(SPOTIFY_CSV, index_col=0)


def build_tracks(raw: pd.DataFrame) -> pd.DataFrame:
    """Dedupe by track_id. genres = sorted real genres; category = sorted mood/activity labels;
    primary_genre = most common real genre (ties alphabetical); for category-only tracks the most
    common category (so stratification and the genre cap never see an empty genre)."""
    labels = raw.groupby("track_id").track_genre.agg(lambda g: sorted(set(g)))

    def most_common(df):
        counts = df.groupby(["track_id", "track_genre"]).size().reset_index(name="n")
        return (counts.sort_values(["track_id", "n", "track_genre"], ascending=[True, False, True])
                      .drop_duplicates("track_id").set_index("track_id").track_genre)

    real = most_common(raw[~raw.track_genre.isin(CATEGORY_GENRES)])
    cat_only = most_common(raw[raw.track_genre.isin(CATEGORY_GENRES)])
    tracks = raw.drop(columns=["track_genre"]).drop_duplicates("track_id").set_index("track_id")
    tracks["genres"] = labels.map(lambda g: [x for x in g if x not in CATEGORY_GENRES])
    tracks["category"] = labels.map(lambda g: [x for x in g if x in CATEGORY_GENRES])
    tracks["primary_genre"] = real.reindex(tracks.index).fillna(cat_only.reindex(tracks.index))
    missing = tracks.track_name.isna() | tracks.artists.isna()   # unusable: no title/artist for desc
    tracks.attrs["n_missing_title_or_artist"] = int(missing.sum())
    return tracks[~missing].reset_index()
