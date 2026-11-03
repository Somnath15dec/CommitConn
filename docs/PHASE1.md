# Phase 1 plan: data only

Phase 1 produces the data everything else depends on: catalog, descriptions, synthetic queries,
users, and eval sets. No Semantic IDs, model training or UI in this phase.
Work step by step and stop at every **[STOP]** for review. Commit after each milestone (conventional commits).

## Global rules (also in CLAUDE.md)
- Windows 11, Python 3.11 venv (`.venv`), torch cu130. Use pathlib for every path; no hardcoded separators.
- Long-running jobs run in the background, writing to a log file. Don't poll the log.
- Never print whole data files: shape, columns and head() only. Trim logs to the last 50 lines.
- Never silently change data thresholds or templates; ask first.
- Every module gets a pytest test.
- Train and eval query templates must never overlap.
- **Playlist names** are evaluation ground truth only. They never appear in desc, tags, context, training
  queries or paraphrase prompts.
- **Spotify `category`** (chill, happy, party, romance, sad, sleep, study) may fill TRAIN templates only,
  never EVAL templates.
- `context_labels.parquet` rows with `label_split == "eval"` are never used for rule tuning.
- `data/proc/` is committed while its total size is under 50 MB. Check the size before each commit.

## Done
| Step | Commit | Output |
|---|---|---|
| 1 Environment | – | `.venv`, torch 2.14.1+cu130, CUDA OK on RTX 4060 Laptop |
| 2 Skeleton | `chore: project skeleton`, `chore: windows setup notes` | dirs, `.gitignore`, `CLAUDE.md`, `requirements.txt` |
| 3 Download | – | `data/raw/{spotify,muse,playlists}` |
| 4 Catalog | `feat: catalog prep with rule-based tags`, `fix: round-robin context cap, instrumental/speech conflict` | `catalog.parquet`, `context_labels.parquet`, `real_users.parquet`, `muse_validation.csv`, `rule_precision_tune.csv` |

### Step 4 decisions (for reference)
- **MuSe join:** first by exact `spotify_id`, then by exact normalized (title, first artist). Ambiguous
  pairs are skipped. Each MuSe entry keeps one track_id (ID match first, then higher popularity).
  `muse_match` = id | name | null. MuSe scores are evaluation-only and never appear in desc.
- **Playlists (Zenodo 2594557):** matched with the same normalizers. A pair that maps to several
  Spotify ids goes to the most popular one. Malformed CSV lines are skipped and counted.
- **Context playlists:** album-named playlists are dropped ("<artist> – <title>"). A playlist is kept only
  with ≥5 matched tracks from ≥3 artists. Phrases: study, focus, sleep, workout, gym, party, chill, relax,
  sad, rain, morning, drive, coffee, late night, night drive, running, run playlist.
- **Catalog priority (10,000 tracks):**
  1. MuSe tracks.
  2. Tracks from cleaned context playlists.
  3. Other playlist-matched tracks, most users first.
  4. Together, 1–3 are capped at 8,500. The last 1,500 come from a genre fill that always tops up the smallest primary_genre.

  No primary_genre may exceed 10% of the catalog.
- **Genres vs category:** mood/activity labels moved to `category`. Category-only tracks have
  `primary_genre` set to their category, and their desc renders "Spotify category: X" with no "Genre:".
- **Tags:** "slow" requires tempo < 90 and energy < 0.5; "fast" requires tempo > 130 and energy > 0.6.
  "Spoken word or rap" is dropped when "instrumental" is present.
- **Context rules:** named constants in `src/prep.py`. Workout requires energy > 0.8. Words are deduped
  and capped at 4 per track by round-robin: first the lead word of every fired rule, then the remaining
  words, both in tune-half lift order.
- **Real users:** users with ≥20 catalog tracks, split 80/20 into history and heldout (random_state=4).
  History is capped at 200. Stored in long format.

## Step 4.5: data readiness analysis (done, `chore: data readiness analysis`)
`src/analyze_readiness.py` → `eval/readiness_report.md`. Analysis only. It covers:
- content_desc uniqueness
- duplicate songs
- tracks per context word
- query ambiguity
- embedding options A/B with a 3x256 residual k-means preview

## Step 4.6: catalog dedupe, category conflict filter, readiness v2
- **Song dedupe:** the whole catalog is deduped by normalized (title, first artist). Priority is
  muse > context_playlist > playlist > genre_fill, then popularity. Freed slots go to the genre fill,
  which also dedupes by song.
- **Category conflicts:** a category is left out of desc when it contradicts the audio tags:
  happy vs melancholic, sad vs happy, sleep/study vs energetic, party vs calm. The `category` column keeps it.
- **Synonyms:** `CONTEXT_SYNONYMS` in `src/config.py`:
  - sleep ← wind down
  - study ← focus, late night
  - rainy day ← reflective

  Desc carries canonical words only.
- **Readiness v2:**
  - `AUDIO_WEIGHT` sweep {0.5, 1.0, 1.5, 2.0} for option B
  - Hit@10 oracle ceiling
- Tracks with no title or artist are excluded from the candidate pool (1 in Spotify).

### Decision: AUDIO_WEIGHT for Semantic IDs
- **Default: `AUDIO_WEIGHT = 1.0`** (option B: content_desc embedding + 1.0 x min-max audio features).
  Chosen for genre coherence: 5-NN same genre 83.8%, same context 95.7%, collision rate 11.5%,
  max 9 tracks per code.
- **Fallback: 1.5** (same genre 73.1%, same context 94.8%, collisions 6.7%, max 4 per code), if
  collisions at 1.0 prove a problem.
- 2.0 was rejected: it had the fewest collisions, but same genre fell to 61.9%.

## Execution order for the remaining steps
**5 → 8 → 10 → 6 (background) → 11 → 12 → 7 → 13 → 9.**
- Step 10 (`eval/PLAN.md`) must be committed before Step 13 produces any numbers.
- Step 6's full paraphrase run goes in the background (log file, not polled) while later steps proceed.

## Step 4b: skipped for now

## Step 5: `src/gen_queries.py`, templated queries (no LLM)
- Two DISJOINT template sets:
  - `TRAIN_TEMPLATES`: ~15 templates across 5 types:
    - mood ("something {mood}")
    - context ("music for {context}")
    - genre ("{genre} songs")
    - combo ("{mood} {genre} for {context}")
    - sound ("{tempo} {sound} tracks")
  - `EVAL_TEMPLATES`: ~8 different phrasings, e.g. "I need {mood} vibes while I {context}" or
    "play me some {genre} that feels {mood}". No template string may appear in both sets.
- Fill slots ONLY from the track's own tags, context and genres. Skip a template if a needed slot is empty.
  - **Genre slots fill from `genres` only**, never from `primary_genre` or `category`.
  - **Category may fill TRAIN templates only**, never EVAL templates.
  - Playlist names and MuSe scores are never used.
  - Context slots may use any synonym from `CONTEXT_SYNONYMS` (e.g. "music to wind down" for a track
    tagged sleep). `wanted_tags` always stores the canonical word.
- **Train:** up to 5 queries per track → `data/proc/train_pairs.jsonl`, with fields
  `{query, track_id, template_id, type, wanted_tags}`.
- **Eval A (pair holdout):** sample 500 tracks, 1 EVAL_TEMPLATES query each → `data/proc/eval_pairs.jsonl`.
  These tracks stay in training through their train queries; only the phrasing is unseen.
- `wanted_tags` holds the tags and context the query asked for (used later for tag-match).
- Print:
  - total pairs and pairs per type
  - **training pairs per context word**, with any context word under 300 pairs flagged
  - 10 random train queries and 10 random eval queries
  - tracks with 0 queries
- `tests/test_queries.py`:
  - no template overlap
  - every track_id exists in the catalog
  - no eval (query, track_id) pair exists in train
  - eval queries never contain a category value or a primary_genre-only value
- **Eval B (headline set)** → `data/proc/eval_context_queries.jsonl`, fields
  `{query, context_word, playlist_words, surface, template_id}`:
  - One entry per canonical context word with a non-null `CONTEXT_TO_PLAYLIST_WORDS` entry (11 words; synonym keys
    are not canonical). 12 distinct queries each, context slot only, from `EVAL_B_TEMPLATES` (12 EVAL-style
    phrasings, disjoint from TRAIN_TEMPLATES). Surfaces (canonical + synonyms) are interleaved.
  - `playlist_words` are the canonical word's (synonyms map to canonical). random_state=8.
  - Tests: no Eval B query string appears in train_pairs; no template string or skeleton is shared with TRAIN_TEMPLATES.
- **[STOP]** Show samples. Commit: `feat: templated synthetic queries + eval B`.

### Step 5 decisions (approved)
- Context slots take the track's context words **rarest-first** (catalog-wide), so thin words get a query from
  every track carrying them. Result: every context word ≥300 train pairs (sleep 338, lowest).
- Eval category check is **per track**: an eval query may not contain that track's own category values or its
  category-only primary_genre. Category words that are also tags/context (happy, party, sleep, study) remain
  allowed for other tracks.
- Slot vocab: "neutral mood" and "moderate energy" never fill slots; "spoken word or rap" renders as "rap";
  category "romance" renders as "romantic" (train only); context words render as noun phrases
  (`CONTEXT_PHRASES`). `wanted_tags` holds canonical words and raw genre ids.
- **Genre humanizing** (train and eval): "-" → space (hard-rock → hard rock), r-n-b → r&b; hip-hop, k-pop, j-pop,
  j-rock, j-dance, j-idol keep the hyphen. `wanted_tags` keep the genre id.
- Seeds: train 6, Eval A sampling 7, Eval B 8.
- Known gap (goes in DATA.md): Eval A has only 6 sound-type pairs (needs both a tempo and a sound tag) and thin
  per-context-word coverage (e.g. rainy day 1). Context performance is measured on Eval B instead.

## Step 6: `src/paraphrase.py`, LLM paraphrases via local Ollama
- Run `ollama list`, show the models and ask which one to use. Don't pick yourself.
- Subset of 2,000 tracks (random_state=2), in batches of 10 descs per request. Ask for 3 natural,
  human-sounding app requests per track as strict JSON: `[{"track_id": ..., "queries": [...]}]`.
- Validate the JSON. Retry a failed batch once, then skip it and log the failure.
- Reject queries that contain the track name or artist name.
- Append to `data/proc/paraphrase_pairs.jsonl` as each batch completes. Resumable: skip track_ids
  already done.
- First run ONLY 5 batches, time them, and print the results plus a projected total runtime.
- **[STOP]** Show samples and the runtime estimate. The full run goes in the background with a log
  file, and only on the user's go-ahead.

## Step 7: `src/gen_users.py`, synthetic users (fallback to real users)
- 300 users, random_state=3. Each has 2–3 favorite primary_genres and 1 favorite mood tag.
- History of 50 tracks: 80% from favorite genres (weighted toward the favorite mood), 20% random.
- Output `data/proc/users.json`: `[{user_id, fav_genres, fav_mood, history: [track_id...]}]`.
- Print 3 sample users with their history's genre distribution.
- `tests/test_users.py`: all track_ids exist; 50 unique tracks per user.
- Real users (`real_users.parquet`) are primary. Simulated users are the fallback.
- Commit: `feat: synthetic users`.

## Step 8: gold eval set
- `eval/gold_queries.jsonl`: 30 hand-written queries, each with `wanted_tags`. Cover study/focus, sleep,
  workout, party, sad/reflective, morning and driving, plus 5 tricky ones (negation, mixed moods).
- Plus **8 robustness queries** with `"type": "robustness"`, not scored for accuracy:
  - 2 empty or gibberish
  - 2 non-music requests
  - 2 prompt-injection attempts
  - 2 negation queries ("no vocals", "not sad")
- **[STOP]** Show all 38 for editing. Commit: `feat: gold eval queries`.

## Step 9: `data/DATA.md`, data card
- Sources and licenses: check each dataset page and quote only the license name. Covers Spotify Tracks
  (Kaggle), MuSe (Kaggle) and the Innsbruck Spotify Playlists dataset (Zenodo, CC-BY-4.0 per the API).
- What is real and what is synthetic, row counts per file, and how train and eval are separated:
  - template split
  - pair holdout
  - context_labels tune/eval split
  - real-user history/heldout
- Known limitations:
  - synthetic queries and simulated users
  - genre-balanced catalog with playlist and rock/indie skew
  - data collected up to 2022
  - 10,284 malformed playlist lines skipped
  - Eval A: only 6 sound-type pairs and thin per-context-word coverage (see Step 5 decisions)
  - Gold query 19 ("lo-fi hip hop for studying") maps "lo-fi" to hip-hop (+ calm): the catalog has no lo-fi genre
  - Version dedupe (Step 4.7): 4 merged groups are arguably distinct recordings (Bailando EN/ES, La Valse
    d'Amélie piano/orchestra, the Habits and Prayer in C radio edits)
- Rule precision note: precisions are **relative**, computed over context-playlist tracks only (tune
  half), and rules are compared by **lift**.
- Context cap note: at most 4 context words per track, round-robin in lift order.
- **"Data & permission boundary"** section:
  - listening history stays local
  - dataset licenses
  - non-commercial use only
  - simulated users
  - and "no real user data", **wording TBD**: see open item 1.
- Commit: `docs: data card`.

## Step 10: `eval/PLAN.md`, pre-registered evaluation
- **Headline metric:** playlist precision@10 on Eval B context queries (`eval_context_queries.jsonl`), using ONLY `label_split == "eval"` rows of
  `context_labels.parquet`, compared with a FAISS dense-retrieval baseline. A retrieved track counts as
  correct if its eval-half playlist labels contain the queried context word (synonyms map to canonical).
- **Secondary metrics:**
  - Hit@10 and NDCG@10 on `eval_pairs.jsonl` (Eval A), Hit@10 reported next to its oracle ceiling (readiness v2:
    mean min(1, 10 / identical-content_desc group size) over 500 sampled tracks)
  - tag-match@10 on the gold queries
  - valid-item rate
  - p50 latency
- Commit `docs: pre-registered eval plan` **before any model code exists**.

## Step 4.7: version dedupe (done, `fix: version dedupe; defer query balancing to training sampling`)
- `song_key` (`src/muse_match.py`) is used for catalog dedupe only. The MuSe and playlist joins keep `normalize_title`.
  - Accents are stripped (NFKD).
  - Version suffixes are removed: " - remastered...", " - remaster...", " - single edit", " - radio edit",
    " - ... version", " - edit", "(feat. ...)", their bracketed forms, and the legacy live/acoustic/mono/stereo/with
    suffixes.
  - A suffix containing "mix" is never stripped, so remixes stay distinct.
- 26 groups removed (context_playlist 18, playlist 7, genre_fill 1). Refilled by the genre fill (25) plus 1 MuSe track
  that is no longer merged with a "Stereo Mix" version.
- **Frozen label split:** `data/proc/label_split_frozen.csv`. Existing tracks keep their tune/eval half. New labeled
  tracks are split 50/50 (random_state 5). Result: 0 retained tracks changed half or context.
- **Stable genre fill:** the shuffle is drawn over the legacy-key pool, so only the slots that had to change did.
- **Train pairs are incremental:** tracks with valid existing pairs keep them. New tracks get pairs from a
  per-track rng (TRAIN_RANDOM_STATE, crc32(track_id)). `--fresh` regenerates everything.
- Eval A was regenerated once because 1 of its tracks was removed.
- No query cap on the data. Balancing is deferred to Step 15.

## Step 11: `src/sids.py`, Semantic IDs
- Item vector (option B): content_desc embedded with `all-MiniLM-L6-v2` (normalized) + min-max scaled audio
  features × `AUDIO_WEIGHT = 1.0`.
- Residual k-means, 3 levels × 256, seed 11. Dedupe digit `d` makes every full SID unique.
- Save `data/proc/sids.parquet` (track_id, a, b, c, d) and the prefix trie.
- Report: collision rate, max tracks per code, level-1 cluster size distribution.
- Tests: every track has a unique full SID; the trie yields only catalog SIDs.

## Step 12: `src/cluster_names.py`, level-1 cluster names
- For each level-1 cluster, send 10 sample content_descs to Ollama and get a 2–4 word name.
- Save `data/proc/cluster_names.json`.
- **[STOP]** Show 20 names for review.

## Step 13: `src/baseline.py`, FAISS dense-retrieval baseline
- **Requires Step 10 (`eval/PLAN.md`) committed first.**
- Item vectors: text embedding of content_desc only (queries carry no audio).
- Report all pre-registered metrics:
  - playlist precision@10 on Eval B (eval-half labels only)
  - Hit@10 + oracle ceiling and NDCG@10 on Eval A
  - tag-match@10 on gold queries
  - p50 latency
- Save `eval/results_baseline.json`.


## Step 15: training (spec TBD)
- **Per-epoch sampling:** at most `MAX_PER_STRING = 150` pairs per query string per epoch, resampled each epoch
  (the seed varies by epoch). No data is dropped from `train_pairs.jsonl`.

---

## Open items
1. **"No real user data" vs `real_users.parquet`.** The playlist dataset holds real (hashed) user ids and
   playlists from a public research dataset. The boundary section needs wording that is true. Suggested:
   "no private or personally collected user data; real-user histories come from a public, anonymized
   research dataset."
2. **Category-only primary_genres in Step 7.** `primary_genre` can be a category (e.g. "study") for 295
   catalog tracks. Should synthetic users' `fav_genres` exclude categories?
3. **Thin context words.** Resolved by merging synonyms into canonical words in Step 4.6. Recheck the
   under-300-pairs flag in Step 5.
4. ~~AUDIO_WEIGHT for option B~~: decided. 1.0 is the default and 1.5 the fallback (see Step 4.6).
