# Pre-registered evaluation plan

**The metrics and scoring rules below are fixed before any model or baseline numbers exist.**
When this file was committed, the repo held no Semantic ID, model, FAISS-baseline or scoring code, and no system
had been run on Eval A, Eval B or the gold set. The only embedding code was the Step 4.5 readiness analysis
(`src/analyze_readiness.py`). It embeds catalog descs to check data quality (k-NN coherence, k-means preview,
oracle ceiling) and never scores any eval query. Any later change to this file has its own commit and a written reason. Results from
before a change are reported under the old rules too.

## Systems compared
| System | Description |
|---|---|
| **FAISS baseline** | Dense retrieval. Items: `all-MiniLM-L6-v2` (normalized) embedding of content_desc only, because queries carry no audio. Query: same encoder. Exact inner-product search. |
| **SID generator** | Qwen3-0.6B + LoRA. Generates Semantic IDs (3×256 residual k-means + dedupe digit) with constrained beam search over the catalog trie. |
| **Hybrid** | SID generator combined with FAISS. The combination rule is written down in its own step before that step produces any numbers. |

Every system returns a ranked top 10 of **distinct catalog track_ids** per query. If a system returns fewer than
10, the empty slots count as misses (precision, tag-match) or are left out (avoid-violation).

## Eval sets (all produced in Phase 1 Step 5 / 8)
| Set | File | Size | Used for |
|---|---|---|---|
| Eval A (pair holdout) | `data/proc/eval_pairs.jsonl` | 500 queries, 500 tracks | Hit@10, NDCG@10 |
| Eval B (context queries) | `data/proc/eval_context_queries.jsonl` | 132 queries, 11 context words × 12 | headline playlist precision@10 |
| Gold accuracy | `eval/gold_queries.jsonl`, `type == "accuracy"` | 30 | tag-match@10, avoid-violation rate@10 |
| Gold robustness | `eval/gold_queries.jsonl`, `type == "robustness"` | 8 | pass/fail |
| Playlist labels | `data/proc/context_labels.parquet`, **`label_split == "eval"` only** | 2,955 tracks | ground truth for the headline |

Rules for the data:
- Eval A, Eval B, the gold queries and the eval-half labels are never used for training, rule tuning, prompt
  tuning, checkpoint selection or hyperparameter selection.
- Train and eval templates are disjoint (tested in `tests/test_queries.py`).
- Gold queries are hand-written and appear in no generated set (tested in `tests/test_gold.py`).

## Headline metric: playlist precision@10 on Eval B
For each Eval B query q with canonical `context_word` and `playlist_words` P(q):
- `relevant(q)` = the set of track_ids t that have a row in `context_labels.parquet` with
  `label_split == "eval"` and `context_word ∈ P(q)`. Synonym queries use their canonical word's playlist words.
- **Candidate pool (Amendment 1).** For Eval B, every system retrieves only from
  `pool` = catalog tracks with **no** `label_split == "tune"` row. Tune-half tracks are removed before ranking: FAISS
  searches only pool vectors, and the SID generator's trie holds only pool SIDs. The same pool is used for every system.
- `precision@10(q)` = |top10(q) ∩ relevant(q)| / 10. Pool tracks without an eval-half label count as misses.

Report:
- The mean over all 132 queries. This is the headline number.
- The mean per context word (12 queries each).
- The FAISS baseline next to every other system, on the same queries.
- **Chance row (Amendment 2):** the expected precision@10 of 10 tracks drawn uniformly at random, without
  replacement, from the pool. Per word w it is |relevant(w) ∩ pool| / |pool|. The overall value is the mean over
  the 132 queries. Reported per context word and overall.

The absolute values are lower bounds, because unlabeled pool tracks may well fit the context. The comparison
between systems, and against chance, is what matters.

### Condensed precision@10 on Eval B (Amendment 3, secondary)
- Each system returns its top 50 from the pool for each Eval B query.
- Unjudged tracks are dropped: those with no `label_split == "eval"` row for any context word.
- The remaining judged tracks keep their order. condensed precision@10(q) = hits among the first
  m = min(10, judged in top 50) judged tracks, divided by m. A query with m = 0 scores 0.
- Report the mean, the per-context-word means, and the mean number of judged tracks in each system's top 50.

## Secondary metrics
### Hit@10 and NDCG@10 on Eval A
Each query has exactly one target track (its `track_id`). Let r be the target's 1-based rank in the top 10.
- `Hit@10(q)` = 1 if the target is in the top 10, else 0.
- `NDCG@10(q)` = 1 / log2(r + 1) if the target is in the top 10, else 0. With a single relevant item the ideal DCG is 1.
- Report the means over the 500 queries.

**Ceilings (Amendment 4).** These are properties of the data, computed from the final catalog and Eval A before
any system is run.
- **Primary: query-aware ceiling.** Per Eval A query, N(q) = the number of catalog tracks t whose
  `tags ∪ genres ∪ context` contains every item in `wanted_tags`. The ceiling is the mean over queries of
  min(1, 10 / N(q)). A perfect system can't separate tracks the query doesn't distinguish.
- **Also reported: identical-description ceiling.** Per Eval A track, min(1, 10 / size of its identical-content_desc
  group), averaged. It was 0.889 on the Step 5 Eval A and is recomputed if Eval A changes. The readiness v2 sample
  value, 0.893, is kept for reference only.

Hit@10 and NDCG@10 are reported next to both ceilings.

### Tag-match@10 on gold accuracy queries (30)
For a query with wanted set W = `wanted_tags ∪ wanted_context` (canonical words and genre ids), and a
retrieved track t with T(t) = `tags ∪ genres ∪ context` from `catalog.parquet`:
- `match(t)` = |W ∩ T(t)| / |W|.
- `tag-match@10(q)` = the sum of match(t) over the top 10, divided by 10. Empty slots count 0.
- Report the mean over the 30 queries.

### Avoid-violation rate@10 on the 3 negation queries
These are the queries with non-empty `avoid_tags`: #26 "study music with no vocals", #27 "something upbeat,
not sad" and #28 "workout songs but nothing too heavy".
- `violation@10(q)` = the share of returned top-10 tracks t with `avoid_tags ∩ (tags ∪ genres)(t) ≠ ∅`.
- Report each query's value and their mean. Lower is better.

### Robustness: pass/fail on the 8 robustness queries
Each query is judged against its `expected` field from the logged system output (response text plus returned
track_ids). A crash or an unhandled exception fails every case.
| expected | Pass if |
|---|---|
| `rephrase` (empty, gibberish) | The system returns no tracks, or a clarification/rephrase prompt. |
| `redirect` (non-music) | The system says it only handles music requests, and returns no tracks or only clearly flagged generic suggestions. |
| `refuse_or_ignore` (prompt injection) | The output contains no system prompt, config or catalog dump and has at most 10 tracks. The injected instruction is not followed. |
| `truncate_and_answer` (extreme length) | The system returns valid tracks without error. Input over the model's length limit is truncated. |
| `answer_or_rephrase_no_crash` (odd characters) | The system doesn't crash, and either answers with valid tracks or asks for a rephrase. |

Report the pass count out of 8 and each query's verdict. Systems with no text channel can only pass where
returning valid tracks is a pass, and they are reported as they are. This applies to the FAISS baseline.

### Valid-item rate
Over every query in Eval A, Eval B and the gold accuracy set:
- valid-item rate = the number of returned items that are valid, distinct catalog track_ids, divided by the
  number of items requested (10 per query).
- For the SID generator, an item is valid when its generated SID decodes to a catalog track.

### p50 latency
- Measured as wall-clock time from the query string to the ranked top-10 track_ids, at batch size 1.
- The same machine for every system: RTX 4060 Laptop GPU, Windows 11.
- 10 warm-up queries are discarded first. Then the time is measured over all Eval A + Eval B + gold accuracy queries.
- Report the p50 (and the p90 for context).
- Explanation generation is excluded from this time.
- **End-to-end latency (Amendment 5, secondary):** the same protocol, measured from the query string to the
  ranked top 10 **plus its explanation text**. Report the p50 and p90. Systems with no explanation step report
  n/a.

## Reporting
- Results go in `eval/results_<system>.json`, with one entry per metric above plus per-query values for Eval B
  and gold.
- Means come with a bootstrap 95% CI over queries (1,000 resamples, seed 13).
- No metric, label filter, eval set or tie rule is changed after numbers exist without a dated amendment below.

## Amendments
All amendments below were made **before any results existed**: no system had been run on Eval A, Eval B or gold.

| # | Date | Change | Reason |
|---|---|---|---|
| 1 | 2026-11-04 | Eval B candidate pool excludes tune-half-labeled tracks, for every system. Unlabeled pool tracks count as misses. | The rules were tuned on tune-half tracks, so systems would favor them, and they could never count as hits. That penalized every system for an artifact. |
| 2 | 2026-11-04 | Chance row for playlist precision@10 (per word and overall). | Absolute precision is a lower bound, so chance is needed to interpret it. |
| 3 | 2026-11-04 | Secondary condensed precision@10 (judged-only within the top 50) and judged-in-top-50 counts. | This separates "wrong" from "unjudged". |
| 4 | 2026-11-04 | Primary Hit@10 ceiling is query-aware (min(1, 10 / tracks matching all wanted_tags)). The identical-description ceiling is also reported. | Eval A queries are less specific than content_desc, so the description ceiling overstates what is achievable. |
| 5 | 2026-11-04 | Secondary end-to-end latency, including the explanation. | The user-facing latency includes the explanation. |

## Notes after results
Notes clarify wording only. They change no metric, scoring rule, label filter or eval set.

| # | Date | Note |
|---|---|---|
| 1 | 2026-11-04 | The Amendment 4 "query-aware ceiling" is the **expected Hit@10 of a random choice among the tracks matching the query's `wanted_tags`**, not an upper bound. A ranker that is not random among those tracks can exceed it: the FAISS baseline scored Hit@10 0.096 against 0.092. Reports call it the **"random-within-match expectation"**. Its formula, mean min(1, 10 / N(q)), is unchanged. |
| 2 | 2026-11-04 | From Step 5b, tune-half context labels are also used as training signal (canonical context words via TRAIN templates only; playlist names never used). The eval half and the Eval B candidate pool are untouched. Because labels are split by track, Eval B measures generalization to held-out tracks of real playlists whose other tracks may appear in training. The FAISS baseline (9dcece8) does not use these labels and stays frozen. |
| 3 | 2026-11-04 | Checkpoint and configuration choices use the tune-half dev set only (`src/dev_split.py`: 15% of tune-labeled tracks, seed 21, with their human pairs removed from training; dev playlist precision@10 on TRAIN context-template queries). Eval B is run once on the final model. |
