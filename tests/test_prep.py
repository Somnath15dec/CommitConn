from types import SimpleNamespace

import pandas as pd
import pytest

from src import prep
from src.muse_match import normalize_artist, normalize_title
from src.paths import CATALOG, CONTEXT_LABELS, REAL_USERS


@pytest.fixture(scope="module")
def cat():
    return pd.read_parquet(CATALOG)


def test_size(cat):
    assert len(cat) == prep.CATALOG_SIZE


def test_unique_track_ids(cat):
    assert cat.track_id.is_unique


def test_desc_non_empty(cat):
    assert cat.desc.str.strip().str.len().gt(0).all()


def test_genre_cap(cat):
    assert cat.primary_genre.value_counts().max() <= prep.CATALOG_SIZE * prep.MAX_GENRE_FRAC


def test_no_playlist_fields_in_catalog(cat):
    assert not any("playlist" in c.lower() for c in cat.columns if c != "source")


def test_tempo_guard():
    base = dict(valence=0.5, instrumentalness=0, acousticness=0, danceability=0, speechiness=0)
    assert "slow" not in prep.make_tags(SimpleNamespace(tempo=70, energy=0.8, **base))   # half-time
    assert "fast" not in prep.make_tags(SimpleNamespace(tempo=160, energy=0.3, **base))  # double-time
    assert "fast" in prep.make_tags(SimpleNamespace(tempo=160, energy=0.8, **base))
    assert not {"slow", "fast", "mid-tempo"} & set(prep.make_tags(SimpleNamespace(tempo=110, energy=0.5, **base)))


def test_excluded_tag_rule():
    assert "workout" in prep.make_context(["energetic", "neutral mood"], energy=0.9)
    assert "workout" not in prep.make_context(["energetic", "neutral mood"], energy=0.75)
    assert "workout" not in prep.make_context(["energetic", "danceable"], energy=0.9)
    ctx = prep.make_context(["energetic", "melancholic"], energy=0.9)
    assert "workout" not in ctx and {"intense", "cathartic"} <= set(ctx)


def test_context_cap_and_dedupe(cat):
    assert cat.context.str.len().max() <= prep.MAX_CONTEXT_WORDS
    assert all(len(c) == len(set(c)) for c in cat.context)
    assert not any("morning coffee" in c for c in cat.context)


def test_categories_not_in_genres(cat):
    from src.spotify import CATEGORY_GENRES
    assert not any(set(g) & CATEGORY_GENRES for g in cat.genres)
    assert all(set(c) <= CATEGORY_GENRES for c in cat.category)
    # primary_genre may be a category only for category-only tracks
    bad = cat[cat.primary_genre.isin(CATEGORY_GENRES) & (cat.genres.str.len() > 0)]
    assert bad.empty
    assert cat.primary_genre.notna().all() and (cat.primary_genre != "").all()
    only = cat[cat.genres.str.len() == 0]
    assert not only.desc.str.contains("Genre:").any()
    from src.config import desc_categories
    shown = [bool(desc_categories(c, t)) for c, t in zip(only.category, only.tags)]
    assert (only.desc.str.contains("Spotify category:") == shown).all()   # hidden only when contradicted


def test_workout_share(cat):
    assert cat.context.map(lambda c: "workout" in c).mean() < 0.25


def test_general_listening_share(cat):
    assert (cat.context.str.len() == 0).mean() < 0.35


def test_tags_and_context_rules():
    r = SimpleNamespace(tempo=70, energy=0.2, valence=0.2, instrumentalness=0.9,
                        acousticness=0.9, danceability=0.1, speechiness=0.05)
    tags = prep.make_tags(r)
    assert tags[:3] == ["slow", "calm", "melancholic"] and "instrumental" in tags and "acoustic" in tags
    # fired (default order): calm+instrumental, calm+acoustic, calm+slow, melancholic+slow, acoustic
    ctx = prep.make_context(tags)          # round-robin: lead word of each rule first, then cap
    assert ctx == ["study", "relaxing", "sleep", "rainy day"]   # synonyms collapsed to canonical
    assert len(ctx) == len(set(ctx))


def test_no_duplicate_songs(cat):
    keys = list(zip(cat.track_name.map(normalize_title), cat.artists.map(normalize_artist)))
    assert len(keys) == len(set(keys))


def test_category_conflicts_hidden_from_desc():
    from src.config import desc_categories
    assert desc_categories(["happy"], ["energetic", "melancholic"]) == []
    assert desc_categories(["sleep", "chill"], ["energetic", "neutral mood"]) == ["chill"]
    assert desc_categories(["party"], ["calm"]) == []
    assert desc_categories(["party"], ["energetic"]) == ["party"]


def test_desc_has_canonical_context_only(cat):
    from src.config import SYNONYM_TO_CANONICAL
    assert not any(set(c) & set(SYNONYM_TO_CANONICAL) for c in cat.context)


def test_instrumental_drops_speech():
    r = SimpleNamespace(tempo=110, energy=0.5, valence=0.5, instrumentalness=0.9,
                        acousticness=0, danceability=0, speechiness=0.5)
    assert "spoken word or rap" not in prep.make_tags(r)
    r.instrumentalness = 0.1
    assert "spoken word or rap" in prep.make_tags(r)


def test_normalizers():
    assert normalize_title("I've Just Seen A Face - Remastered 2009") == "i ve just seen a face"
    assert normalize_title("Song (feat. X)") == "song"
    assert normalize_artist("Elvis Costello & The Attractions") == "elvis costello"
    assert normalize_artist("A;B") == normalize_artist("A, B") == normalize_artist("A feat. B") == "a"


def test_context_labels_in_catalog(cat):
    lab = pd.read_parquet(CONTEXT_LABELS)
    assert list(lab.columns) == ["track_id", "context_word", "playlist_count", "label_split"]
    assert lab.track_id.isin(cat.track_id).all()
    assert set(lab.label_split) == {"tune", "eval"}
    assert lab.groupby("track_id").label_split.nunique().max() == 1   # split is by track
    frac = (lab.drop_duplicates("track_id").label_split == "tune").mean()
    assert abs(frac - 0.5) < 0.01


def test_real_users(cat):
    u = pd.read_parquet(REAL_USERS)
    assert list(u.columns) == ["user_id", "track_id", "split"]
    assert set(u.split) == {"history", "heldout"}
    assert u.track_id.isin(cat.track_id).all()
    assert not u.duplicated(["user_id", "track_id"]).any()          # history/heldout disjoint
    assert u[u.split == "history"].groupby("user_id").size().max() <= prep.REAL_USER_MAX_HISTORY
