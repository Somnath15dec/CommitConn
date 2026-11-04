# Shelf — a language-steerable music recommender

Natural-language query → Semantic IDs (3×256 residual k-means + dedupe digit) → trie-constrained beam search with a
Qwen3-0.6B + LoRA generator → grounded one-line explanation. A "Familiar ⟷ Discover" dial biases the first SID
token toward or away from the listener's history clusters. See `docs/PHASE1.md` and `eval/PLAN.md` for data and
evaluation details, and `eval/results_*.md` for results.

## Team

- Priyanshu Kumar
- Vanisha (Abba)
- Mukesh Kumar Pandey

## Run the demo locally

Requirements: the Python venv (`requirements.txt`), the trained adapter in `checkpoints/run1/epoch3`, Ollama with
`qwen3:4b-instruct` pulled (`ollama pull qwen3:4b-instruct`), and Node 18+.

**1. API** (from the repo root; loads the SID model, FAISS and the explainer, then warms up — about 30–60 s):

```powershell
.venv\Scripts\python -m uvicorn src.api:app --port 8000
```

Check it: `http://localhost:8000/health` · interactive docs: `http://localhost:8000/docs`.

**2. UI** (in a second terminal):

```powershell
cd ui
npm install        # first time only
npm run dev        # http://localhost:5173
```

The UI talks to `http://localhost:8000` by default; set `VITE_API_URL` to point elsewhere. CORS is enabled for
`localhost:5173`.

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/recommend` | `{query, user_id?, dial ∈ [-1, 1], k, explain: "template" \| "llm"}` → results + `meta` (system, guard, latency) |
| POST | `/explain` | `{query (cleaned), track_ids}` → grounded LLM explanations, template fallback per item |
| GET | `/users` | 20 demo listeners with their top 3 clusters |
| POST | `/feedback` | `{user_id, query, track_id, action: skip \| save \| replay, dial, timestamp}` → `logs/feedback.jsonl` |
| GET | `/health` | SID model, FAISS and explainer status |

The UI calls `/recommend` with template explanations (fast) and then `/explain` to replace them with AI text.

## Project structure

### Root
| File | Purpose |
|---|---|
| `CLAUDE.md` | Project rules: stack, hardware, data-leak rules, commit conventions. |
| `README.md` | This file. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Keeps `data/raw/`, `checkpoints/`, `exports/`, `logs/`, `node_modules/` and `.venv/` out of git. |

### `src/` — Python code
Every script runs as `python -m src.<name>`. Listed in pipeline order.

**Shared helpers**
| File | Purpose |
|---|---|
| `paths.py` | Every file path in the project, built with pathlib (raw inputs, `data/proc` outputs, exports). |
| `config.py` | Context-word synonyms (e.g. `study` ↔ `focus`, `late night`) and `canonical()`. |

**Phase 1: building the data**
| File | Purpose | Reads → Writes |
|---|---|---|
| `spotify.py` | Loads and dedupes the Spotify tracks CSV. Mood/activity "genres" (chill, sad, …) move to `category`. | `data/raw/spotify/dataset.csv` → DataFrame |
| `muse_match.py` | Attaches MuSe valence/arousal scores to tracks: first by exact Spotify ID, then by exact title + artist. | `data/raw/muse/` → `muse_validation.csv` |
| `playlists.py` | One streaming pass over the 1.2 GB Innsbruck playlists zip. Matches playlist tracks to our tracks and finds playlist names with context words. These names are used only as evaluation ground truth. | zip → `data/raw/playlists/*.parquet` |
| `prep.py` | Builds the catalog: noise filtering, tags, context rules, the `content_desc` text, genre balancing, the tune/eval label split, and real users. | → `catalog.parquet`, `context_labels.parquet`, `label_split_frozen.csv`, `rule_precision_tune.csv`, `real_users.parquet` |
| `analyze_readiness.py` | Read-only checks on the catalog (trial SID stats, neighbour agreement). | → `eval/readiness_report.md` |
| `gen_queries.py` | Template-filled synthetic queries, no LLM. Train and eval use separate template sets. | → `train_pairs.jsonl`, `eval_pairs.jsonl` (Eval A), `eval_context_queries.jsonl` (Eval B) |
| `human_pairs.py` | Training pairs built from the tune-half playlist labels. | → `human_pairs.jsonl` |
| `paraphrase.py` | LLM paraphrases via local Ollama, generated from `content_desc` only. Filters for leaks and can resume where it stopped. | → `paraphrase_pairs.jsonl`, `paraphrase_batches.jsonl` |

**Phase 2: Semantic IDs, baseline, training**
| File | Purpose | Reads → Writes |
|---|---|---|
| `sids.py` | Item vector = MiniLM embedding of the description + weighted audio features. Residual k-means (3 levels × 256 clusters) plus a dedupe digit gives each track a unique SID. Also builds the trie. | catalog → `sids.parquet`, `sid_trie.json` |
| `cluster_names.py` | Ollama names each of the 256 level-1 clusters (2–4 words each). | → `cluster_names.json` |
| `baseline.py` | FAISS dense-retrieval baseline. Also holds the shared metric code (precision@10, Hit/NDCG, tag-match, avoid-violation, bootstrap CIs). | → `eval/results_baseline.{json,md}` |
| `dev_split.py` | A dev set taken from the tune half, used to pick the best checkpoint. | → `dev_tracks.csv`, `eval/dev_faiss.json` |
| `train_data.py` | Merges templated, human and paraphrase pairs, drops duplicates, drops anything matching an eval query, and holds out a validation set. `--freeze` writes the split. | → `train_pool.parquet`, `val_pairs.parquet` |
| `train.py` | LoRA fine-tune of Qwen3-0.6B. Adds new `<a_i><b_j><c_k><d_l>` tokens and computes loss on the SID tokens only. Saves one checkpoint per epoch. | → `checkpoints/run1/epoch{1..3}` |
| `decode.py` | `SidDecoder`: beam search restricted by the trie, so every output is a real catalog track. Can exclude given tracks. | used by train / eval / API |

**Evaluation**
| File | Purpose | Writes |
|---|---|---|
| `dial.py` | The discovery dial (familiar vs non-habitual switch). Adds λ to the scores of the user's history level-1 tokens at the first decode step only. Includes its pre-registered evaluation. | `eval/results_dial.{json,md}` |
| `dial_calibration.py` | Re-runs the dial evaluation at small λ values (±0.5, ±1). Descriptive only, no pass/fail. | `eval/results_dial_calibration.*` |
| `evaluate_final.py` | Final one-time comparison: FAISS vs the SID model vs a hybrid (RRF over both), plus latency. | `eval/results_final.{json,md}` |
| `api_eval.py` | Robustness checks, latency and example responses, all run through the real API. | `eval/results_robustness.json`, `eval/results_api.json` |

**Serving**
| File | Purpose |
|---|---|
| `guards.py` | Input safety, run before any model sees the query: cleaning → self-harm (helpline message) → gibberish → stripping prompt injection → non-music classifier. |
| `explain.py` | One-sentence explanations from Ollama, checked against the track's own tags/context. Falls back to a template if the text is ungrounded, too slow or malformed. |
| `api.py` | FastAPI app: `/recommend` (guards → SID model with a 2 s budget, else FAISS fallback), `/explain` (fills in LLM text in a second phase), `/users`, `/feedback`, `/health`. |

**Excel exports** (to the gitignored `exports/`): `export_review.py` (catalog review), `export_pretrain.py` (training-data snapshot), `export_final.py` (full Phase 1 workbook).

### `data/`
- `DATA.md`: the data card (sources, licenses, what's real vs synthetic, how train and eval are kept apart).
- `raw/` (gitignored): the Spotify CSV, MuSe and the playlists zip.
- `proc/`: every generated file listed in the tables above.

### `eval/`
- `PLAN.md`: the evaluation plan, written before any results, plus later amendments.
- `gold_queries.jsonl`: hand-written gold queries (accuracy, negation and robustness cases).
- `results_*.json/md`, `readiness_report.md`, `dev_faiss.json`: outputs of the scripts above.

### `tests/`
One pytest file per module (`test_<module>.py`). Also `test_gold.py`, `test_queries.py` and `test_readiness.py` (for `gen_queries` and `analyze_readiness`), and `test_data_card.py`, which checks `DATA.md` against the actual data.

### `ui/` (React + Vite + TypeScript)
- `index.html` and `src/main.tsx`: entry point.
- `src/App.tsx`: main screen (query box, user picker, discovery dial, compare mode, result cards, feedback).
- `src/api.ts`: client for `/recommend`, `/explain`, `/users` and `/feedback`.
- `src/icons.tsx` and `src/styles.css`: icons and the light Spotify-style theme.
- `package.json`, `tsconfig.json`, `vite.config.ts`: build config.

### `docs/`
- `PHASE1.md`: step-by-step plan, decisions and open items.
- `screenshots/`: UI screenshots (normal result, compare mode, guard message).

### Gitignored folders
- `checkpoints/`: trained LoRA runs.
- `logs/`: API logs and `feedback.jsonl`.
- `exports/`: xlsx workbooks.

## Workflow

```
raw data ─► spotify.py + muse_match.py + playlists.py
                 │
                 ▼
             prep.py ──► catalog.parquet, context_labels, real_users
                 │
     ┌───────────┼──────────────┬───────────────┐
     ▼           ▼              ▼               ▼
 gen_queries  human_pairs   paraphrase       sids.py ──► sids.parquet + sid_trie.json
 (train/EvalA/EvalB)        (Ollama)            │
     └───────────┴──────┬───────┘          cluster_names.py
                        ▼
              train_data.py --freeze ──► train_pool / val_pairs
                        │       (dev_split.py removes dev-track human pairs)
                        ▼
                    train.py (uses decode.py) ──► checkpoints/run1/epochN
                        │
     baseline.py (FAISS) ─┤
                        ▼
   evaluate_final.py · dial.py · dial_calibration.py ──► eval/results_*
                        │
                        ▼
  api.py = guards.py → SidDecoder + dial (FAISS fallback) → explain.py
                        │                      └─ api_eval.py checks it end to end
                        ▼
                 ui/ (React) calls /recommend, then /explain
```

### What happens on one request
1. The UI sends the query, user and dial value to `/recommend`.
2. `guards.py` cleans and checks the text.
3. The SID model runs beam search restricted by the trie, with the dial biasing the user's level-1 history tokens and the user's own history tracks excluded. If that takes more than 2 s, the FAISS fallback answers instead.
4. The cards come back with template explanations in about 0.4 s.
5. The UI then calls `/explain` to swap in the LLM-written sentences.
