# Data card — Shelf (language-steerable music recommender)

Everything the model is trained, selected and evaluated on, where it comes from, and what is real vs synthetic.
Process details: `docs/PHASE1.md`; evaluation rules: `eval/PLAN.md`.

## Sources and licenses

| Source | What we use | License (as stated on the dataset page) |
|---|---|---|
| [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle, maharshipandya) | track metadata, audio features, `track_genre` | **ODbL-1.0** |
| [MuSe: The Musical Sentiment Dataset](https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset) (Kaggle, cakiki) | catalog priority; valence/arousal for validation only | **CC BY 4.0** |
| [Spotify Playlists Dataset](https://zenodo.org/records/2594557) (Zenodo 2594557, Univ. of Innsbruck, 2019) | playlist membership → context labels and real-user histories | **CC BY 4.0** |

Licenses were read from the dataset pages / APIs on 2026-11-04 (Kaggle metadata, Zenodo record API).
ODbL-1.0 is share-alike for derived databases: the files in `data/proc/` derived from the Spotify Tracks data
(catalog, pairs, SIDs) carry its attribution and share-alike obligations if the database is redistributed.

## What is real and what is synthetic

| Data | Real or synthetic |
|---|---|
| Tracks, artists, genres, audio features | **Real** (Spotify Tracks Dataset) |
| Tags (`slow`, `calm`, `melancholic`, …) and context words (`study`, `sleep`, …) | **Rule-derived** from audio features (thresholds in `src/prep.py`) |
| Context labels (`context_labels.parquet`) | **Real**: words in real playlist names (e.g. "study", "rain"), mapped to tracks |
| Real-user histories (`real_users.parquet`) | **Real**, public research data: hashed user ids from the playlist dataset |
| Templated queries (`train_pairs`, Eval A, Eval B) | **Synthetic** (fixed templates filled from each track's own fields) |
| Human-signal pairs (`human_pairs.jsonl`) | **Synthetic queries, real labels**: TRAIN templates filled with tune-half playlist context words |
| Paraphrases (`paraphrase_pairs.jsonl`) | **LLM-generated** (qwen3.5-6k via Ollama) from content descriptions only |
| Gold queries (`eval/gold_queries.jsonl`) | **Hand-written** (30 accuracy + 8 robustness) |
| Cluster names (`cluster_names.json`) | **LLM-generated** (qwen3.5-6k) from 10 sample descriptions per cluster |
| Simulated users | **None.** Step 7 (synthetic users) was not built; the demo and dial evaluation use real users. |

## Files (`data/proc/`) and row counts

| File | Rows | Contents |
|---|---|---|
| `catalog.parquet` | 10,000 | tracks: metadata, genres, tags, context, desc, audio features, source |
| `context_labels.parquet` | 16,916 | (track, playlist context word, playlist count, tune/eval half) |
| `label_split_frozen.csv` | 5,910 | frozen tune/eval half per labeled track |
| `real_users.parquet` | 916,177 | (user, track, history/heldout) for 9,804 users |
| `train_pairs.jsonl` | 48,460 | templated train queries (≤ 5 per track) |
| `human_pairs.jsonl` | 15,390 | tune-half context-label pairs (context + combo templates) |
| `paraphrase_pairs.jsonl` | 5,463 | LLM paraphrases (3 styles) for 1,995 tracks |
| `paraphrase_batches.jsonl` | 200 | paraphrase run log (raw output, filter counts) |
| `train_pool.parquet` | 65,640 | frozen training pool: template 47,711 · tune_context 12,550 · paraphrase 5,379 |
| `val_pairs.parquet` | 1,000 | frozen validation pairs (from the pool only) |
| `dev_tracks.csv` | 441 | tune-half dev tracks for model selection |
| `eval_pairs.jsonl` | 500 | Eval A (pair holdout) |
| `eval_context_queries.jsonl` | 132 | Eval B (11 context words × 12) |
| `sids.parquet` | 10,000 | Semantic ID (a, b, c, d) per track |
| `sid_trie.json` | 256 top-level codes | prefix trie for constrained decoding |
| `cluster_names.json` | 256 | level-1 cluster names |
| `rule_precision_tune.csv` | 11 | rule precision / lift on the tune half |
| `muse_validation.csv` | 6 | MuSe valence/arousal vs audio features (validation only) |
| `eval/gold_queries.jsonl` | 38 | gold accuracy + robustness queries |

Catalog composition: context-playlist 4,713 · MuSe 2,141 · playlist 1,588 · genre fill 1,558 (see `docs/PHASE1.md`
Steps 4–4.7 and Step A). Non-music noise tracks (white noise, rain sounds, …) were removed (Step A).

## How train and eval are kept apart
- **Template split:** TRAIN and EVAL templates are disjoint, including their skeletons (`tests/test_queries.py`).
  Eval B uses its own 12 templates, also disjoint from TRAIN.
- **Pair holdout (Eval A):** 500 tracks get one EVAL-template query each; the tracks stay in training through their
  TRAIN queries, so only the phrasing is unseen. No Eval A (query, track) pair is in any training file.
- **Context labels, tune/eval split by track (seed 5, frozen):** only the **tune** half is ever used to tune rules,
  build human pairs or select models. The **eval** half is Eval B's ground truth only. Eval B ranks only tracks with
  **no** tune-half label (Amendment 1), so tracks whose labels were trained on can never score.
- **Dev set:** 15% of tune-labeled tracks (seed 21) with their human pairs removed from training; all checkpoint and
  configuration choices use it (PLAN.md note 3). Eval B was run once, on the final model.
- **Validation:** 1,000 pairs held out from the training pool only.
- **Eval strings never trained on:** every training source is checked against Eval A, Eval B and gold query strings
  (normalized); matches are dropped (0 found) and tests enforce it.
- **Real users:** each user's tracks are split 80/20 into history and heldout (seed 4); the dial evaluation uses
  history to define familiarity and heldout for familiar hit@10, and excludes history tracks from candidates.
- **Playlist names never appear** in descriptions, tags, context, training queries or prompts. Spotify `category`
  may fill TRAIN templates only.

## Notes on derived labels
- **Rule precision is relative:** it is measured over context-playlist tracks only (tune half), and rules are compared
  by **lift** over that half's base rate.
- **Context cap:** at most 4 context words per track, round-robin by rule (first word of each fired rule, then the
  rest), in a frozen lift order (`FROZEN_RULE_PRIORITY`).
- **Playlist word → context mapping** (human pairs, Eval B): rain → rainy day, chill/relax → relaxing, gym/workout
  → gym + workout, focus/study → study, party → party (Eval B also counts party playlists for "dance"). "drive",
  "night drive", "late night" and "sad" playlists are not mapped.

## Known limitations
- **Synthetic queries:** most training queries are templated; paraphrases are LLM-written from descriptions, not real
  user requests. Real playlist labels are used only as context words, never as query text.
- **Rule-based tags:** tags/context come from audio-feature thresholds, so e.g. every slow + calm track is "sleep"
  regardless of how it actually feels; the model learns these rules, not human taste.
- **Catalog skew:** the catalog is genre-balanced (≤ 10% per genre) but filled first from playlists, so it leans
  rock/indie; data was collected up to 2022.
- **Playlist data:** 10,284 malformed playlist lines were skipped; playlist names are noisy context signals.
- **Eval A:** only 6 sound-type pairs and thin per-context-word coverage; context quality is measured on Eval B.
- **Eval B:** rainy day stays at or below chance for every system (few "rain" playlists, and rule tags disagree with
  them); absolute precision is a lower bound because unlabeled tracks count as misses.
- **Gold query 19** ("lo-fi hip hop for studying") maps "lo-fi" to hip-hop (+ calm): there is no lo-fi genre.
- **Version dedupe (Step 4.7):** 4 merged groups are arguably distinct recordings (Bailando EN/ES, La Valse d'Amélie
  piano/orchestra, the Habits and Prayer in C radio edits).
- **Category-only tracks** whose category was hidden from the description (it contradicted their audio tags) have no
  genre in their description and cluster on audio alone: 12 such tracks form one level-1 cluster (cluster 169).
- **Sleep pool:** the category-only "sleep" pool is mostly noise recordings the filters can't fully catch, so the
  genre fill adds no new sleep tracks (`FILL_MAX_ADDED`).
- **Explanations** are grounded only against tags/context words; they can still be bland or template-like.

## Data & permission boundary
- **Listening history stays local.** The demo reads the public research histories in `data/proc/` and writes feedback
  (save / skip / replay) to `logs/feedback.jsonl` on the local machine; it is not committed and not sent anywhere.
  All models (SID generator, MiniLM, Ollama explainer) run locally.
- **Dataset licenses:** ODbL-1.0 (Spotify Tracks) and CC BY 4.0 (MuSe, Spotify Playlists); attribution above.
- **Non-commercial use only:** this is a research / hackathon demo and is not intended for commercial use.
- **Simulated users:** none; see above.
- **User data (proposed wording, open item 1 — confirm):** "No private or personally collected user data is used.
  Real-user listening histories come from a public, anonymized research dataset (hashed user ids)."
