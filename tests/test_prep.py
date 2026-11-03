from types import SimpleNamespace

import pandas as pd
import pytest

from src import prep
from src.muse_match import normalize_artist, normalize_title, song_key, song_title
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
    keys = [song_key(n, a) for n, a in zip(cat.track_name, cat.artists)]
    assert len(keys) == len(set(keys))


def test_song_title_versions():
    same = ["Come On Eileen", "Come On Eileen - Single Edit", "Come On Eileen - Radio Edit",
            "Come On Eileen - Remastered 2011", "Come On Eileen - 2011 Remaster", "Come On Eileen - Edit",
            "Come On Eileen - 12\" Version", "Come On Eileen (feat. X)", "Come On Eileen (7\" Edit)",
            "Come On Eileen [Digitally Remastered]", "Come On Eileen - Live"]
    assert {song_title(t) for t in same} == {"come on eileen"}
    assert song_title("El Día Que Me Quieras") == song_title("El dia que me quieras")
    # remixes / mixes stay distinct
    for t in ["Song - Club Mix", "Song (Extended Mix)", "Song - Remix", "Song - Radio Edit Remix"]:
        assert song_title(t) != "song", t
    assert song_title("Credit Card") == "credit card" and song_title("Mix Tape") == "mix tape"
    assert song_key("A", "Beyoncé;Jay-Z") == song_key("A", "Beyonce")


def test_frozen_split_preserved():
    from src.paths import CONTEXT_LABELS, LABEL_SPLIT_FROZEN
    frozen = pd.read_csv(LABEL_SPLIT_FROZEN).set_index("track_id").label_split
    now = pd.read_parquet(CONTEXT_LABELS).drop_duplicates("track_id").set_index("track_id").label_split
    common = frozen.index.intersection(now.index)
    assert (frozen[common] == now[common]).all()


def test_split_labels_keeps_frozen_and_splits_new():
    labels = pd.DataFrame({"track_id": [f"t{i}" for i in range(20)], "context_word": "party",
                           "playlist_count": 1})
    frozen = pd.Series({"t0": "eval", "t1": "tune"})
    out = prep.split_labels(labels, frozen).set_index("track_id").label_split
    assert out["t0"] == "eval" and out["t1"] == "tune"
    assert (out.drop(["t0", "t1"]) == "tune").sum() == 9


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


# ---- Step A: non-music noise filter, tempo missing values

def _rows(**cols):
    n = len(next(iter(cols.values())))
    base = {"track_id": [f"t{i}" for i in range(n)], "energy": [0.5] * n, "speechiness": [0.05] * n}
    return pd.DataFrame({**base, **cols})


def test_is_noise_patterns_blocklist_audio_and_spare():
    df = _rows(track_name=["Pure WHITE noise", "Song", "Song", "Song", "November Rain", "She's Thunderstorms",
                           "Silent", "Piano Baby Lullabies"],
               artists=["X", "Rain Sounds Lab", "A;Meditation Spa", "Loopable Labs", "Guns N' Roses",
                        "Arctic Monkeys", "Y", "Baby Sleep Music"],
               energy=[0.5, 0.5, 0.5, 0.5, 0.6, 0.9, 0.005, 0.01],
               speechiness=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0])
    df.loc[5, "track_id"] = next(iter(prep.NOISE_SPARE_IDS))
    # "Loopable Labs" hits the "loopable" pattern; "Baby Sleep Music" is not an exact blocklisted artist
    assert prep.is_noise(df).tolist() == [True, True, True, True, False, False, True, False]


def test_no_noise_in_catalog(cat):
    assert not prep.is_noise(cat).any()
    text = (cat.track_name + " " + cat.artists).str.lower()
    spared = cat.track_id.isin(prep.NOISE_SPARE_IDS)
    for p in prep.NOISE_PATTERNS:
        assert not text[~spared].str.contains(p, regex=False).any(), p
    artists = {a.strip().casefold() for s in cat.artists for a in s.split(";")}
    assert not artists & {a.casefold() for a in prep.NOISE_ARTISTS}


def test_refill_pool_skips_noise_and_respects_fill_limit():
    names = ["Rain Sounds 1", "Song A", "Song B", "Fan Noise", "Song C", "Song D", "Song E", "Song F"]
    artists = ["P", "Q", "R", "S", "Nature Sounds", "T", "U", "V"]
    tracks = _rows(track_name=names, artists=artists,
                   primary_genre=["sleep", "sleep", "sleep", "rock", "rock", "rock", "rock", "rock"],
                   popularity=[50] * 8)
    tracks.loc[1, "track_id"], tracks.loc[0, "track_id"] = "t0", "t1"   # MuSe track t1 is "Rain Sounds 1"
    tracks = tracks.sort_values("track_id").reset_index(drop=True)
    muse = pd.DataFrame({"track_id": ["t1"], "muse_valence": [0.5], "muse_arousal": [0.5],
                         "muse_n_tags": [1], "muse_match": ["id"]})
    cat, _, _ = prep.select_catalog(tracks, muse, set(), pd.Series(dtype=float), size=5, max_frac=1.0,
                                    stage_limit=5, fill_max_added={"sleep": 1})
    assert not prep.is_noise(cat).any()
    assert "t1" not in set(cat.track_id)                       # noisy MuSe track dropped from stage (a)
    assert (cat[cat.source == "genre_fill"].primary_genre == "sleep").sum() == 1
    assert cat.attrs["noise_dropped"]["muse"] == 1


def test_genre_fill_tracks_not_noise_and_sleep_frozen(cat):
    fill = cat[cat.source == "genre_fill"]
    assert not prep.is_noise(fill).any()
    assert (fill.primary_genre == "sleep").sum() <= prep.FILL_MAX_ADDED["sleep"]


def test_tempo_missing_handling(cat):
    base = dict(energy=0.2, valence=0.5, instrumentalness=0, acousticness=0, danceability=0, speechiness=0)
    assert not {"slow", "fast"} & set(prep.make_tags(SimpleNamespace(tempo=0.0, **base)))
    assert "slow" in prep.make_tags(SimpleNamespace(tempo=70, **base))
    assert (cat.tempo > 0).all()                                  # audio-vector input: no tempo == 0 left
    miss = cat[cat.tempo_missing]
    assert (miss.tempo == cat.tempo[~cat.tempo_missing].median()).all()
    assert not any({"slow", "fast"} & set(t) for t in miss.tags)
    assert not miss.desc.str.contains(r"\b(?:slow|fast)\b").any()


def test_acoustic_rule_excludes_energetic(cat):
    assert not {"coffee", "relaxing"} & set(prep.make_context(["energetic", "neutral mood", "acoustic"], 0.75))
    assert {"coffee", "relaxing"} <= set(prep.make_context(["moderate energy", "neutral mood", "acoustic"], 0.5))
    energetic = cat[cat.tags.map(lambda t: "energetic" in t)]
    assert not any({"coffee", "relaxing"} & set(c) for c in energetic.context)


def test_frozen_rule_priority_covers_all_rules():
    assert sorted(prep.FROZEN_RULE_PRIORITY) == sorted(r[0] for r in prep.CONTEXT_RULES)
