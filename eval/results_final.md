# Final evaluation (run once)

Pre-registered in `eval/PLAN.md` note 4 (commit 56be084) before any of these numbers existed. Model: `checkpoints\run1\epoch3`; SID decoding: trie-constrained beam 50, top 10 distinct; hybrid: RRF k=60, top 50 + top 50. FAISS is the frozen baseline (recomputed; identical to `eval/results_baseline.json`). Means with bootstrap 95% CI over queries (1,000 resamples, seed 13).

| Metric | FAISS baseline | SID model | Hybrid (RRF) |
|---|---|---|---|
| **Eval B playlist precision@10 (headline)** | **0.157 [0.124, 0.195]** | **0.182 [0.158, 0.209]** | **0.145 [0.121, 0.169]** |
| Eval B chance | 0.118 | 0.118 | 0.118 |
| Eval B condensed precision@10 | 0.381 [0.329, 0.436] | 0.449 [0.400, 0.495] | 0.395 [0.347, 0.444] |
| Eval B judged tracks in top 50 (mean) | 20.3 | 21.7 | 20.6 |
| Eval A Hit@10 | 0.096 [0.070, 0.122] | 0.090 [0.066, 0.116] | 0.104 [0.080, 0.132] |
| Eval A NDCG@10 | 0.047 [0.034, 0.062] | 0.041 [0.030, 0.054] | 0.045 [0.033, 0.059] |
| Eval A random-within-match expectation | 0.092 | 0.092 | 0.092 |
| Eval A identical-description ceiling | 0.902 | 0.902 | 0.902 |
| Gold tag-match@10 | 0.634 [0.501, 0.750] | 0.686 [0.588, 0.770] | 0.665 [0.552, 0.763] |
| Gold avoid-violation@10 (mean of #26-28) | 0.100 (#26: 0.0, #27: 0.3, #28: 0.0) | 0.167 (#26: 0.0, #27: 0.0, #28: 0.5) | 0.033 (#26: 0.0, #27: 0.0, #28: 0.1) |
| Robustness | n/a | n/a | n/a |
| Valid-item rate | 1.000 | 1.000 | 1.000 |
| Latency p50 / p90 (ms, batch 1) | 9.0 / 10.2 (CPU, frozen run) | 407 / 410 | 455 / 636 |
| End-to-end latency incl. explanation | n/a | n/a | n/a |

## Additional analysis: paired Eval B differences (not a pre-registered primary metric)

| Comparison | mean difference in precision@10 [95% CI] | queries better / worse / tied |
|---|---|---|
| SID model − FAISS | +0.025 [-0.014, +0.061] | 61 / 31 / 40 |
| Hybrid − FAISS | -0.012 [-0.041, +0.013] | 39 / 35 / 58 |

## Eval B precision@10 per context word

| Context word | chance | FAISS | SID model | Hybrid |
|---|---|---|---|---|
| coffee | 0.036 | 0.017 [0.000, 0.042] | 0.050 [0.025, 0.075] | 0.033 [0.008, 0.067] |
| dance | 0.222 | 0.483 [0.333, 0.667] | 0.300 [0.267, 0.333] | 0.283 [0.192, 0.367] |
| gym | 0.134 | 0.158 [0.100, 0.225] | 0.250 [0.183, 0.308] | 0.217 [0.158, 0.283] |
| morning | 0.069 | 0.075 [0.033, 0.125] | 0.083 [0.033, 0.142] | 0.050 [0.017, 0.092] |
| party | 0.222 | 0.175 [0.125, 0.217] | 0.333 [0.258, 0.408] | 0.258 [0.217, 0.300] |
| rainy day | 0.033 | 0.008 [0.000, 0.025] | 0.000 [0.000, 0.000] | 0.008 [0.000, 0.025] |
| relaxing | 0.236 | 0.317 [0.167, 0.467] | 0.325 [0.258, 0.392] | 0.267 [0.192, 0.358] |
| running | 0.101 | 0.050 [0.008, 0.092] | 0.242 [0.200, 0.275] | 0.150 [0.083, 0.208] |
| sleep | 0.050 | 0.058 [0.017, 0.100] | 0.125 [0.075, 0.175] | 0.042 [0.008, 0.083] |
| study | 0.035 | 0.050 [0.025, 0.083] | 0.058 [0.025, 0.092] | 0.058 [0.025, 0.092] |
| workout | 0.164 | 0.333 [0.208, 0.467] | 0.233 [0.167, 0.300] | 0.225 [0.142, 0.309] |

## Sanity view (gold queries, top 5)

### SID model

**#1 "something calm for a 3 a.m. study session"**

1. They Move on Tracks of Never-Ending Light — This Will Destroy You — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. Anything You Synthesize - Ambient — The American Dollar — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. How to Disappear Completely — Radiohead — _Genre: alt-rock. Sound: calm, melancholic, instrumental. Good for: study._
1. The Imitation Game — Alexandre Desplat, London Symphony Orchestra — _Genre: french. Sound: calm, melancholic, instrumental. Good for: study._
1. Avril 14th — Aphex Twin — _Genre: ambient, electronic. Sound: calm, neutral mood, instrumental, acoustic. Good for: study, relaxing, coffee, morning._

**#4 "help me fall asleep"**

1. I Swear — The Minions — _Genre: children. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. Twinkle, Twinkle, Little Star — LooLoo Kids — _Genre: children. Sound: slow, calm, melancholic. Good for: rainy day, sleep._
1. The Longest Time — Billy Joel — _Genre: piano. Sound: slow, calm, neutral mood. Good for: sleep._
1. Time — Hans Zimmer — _Genre: german. Sound: slow, calm, melancholic, instrumental. Good for: study, rainy day, sleep._
1. Ungodly Hour — The Fray — _Genre: piano. Sound: slow, calm, melancholic. Good for: rainy day, sleep._

**#6 "pump me up for leg day"**

1. D.A.N.C.E — Justice — _Genre: electronic. Sound: energetic, neutral mood. Good for: workout._
1. Who Taught You to Live Like That? — Sloan — _Genre: power-pop. Sound: energetic, neutral mood. Good for: workout._
1. Work — Jimmy Eat World — _Genre: power-pop. Sound: energetic, neutral mood. Good for: workout._
1. Polish Girl — Neon Indian — _Genre: synth-pop. Sound: energetic, neutral mood. Good for: workout._
1. Naive — The Kooks — _Genre: indie, rock. Sound: energetic, happy. Good for: workout._

**#16 "rainy afternoon staring out the window"**

1. Until The Morning — Thievery Corporation, Emilíana Torrini — _Genre: trip-hop. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. No Surprises — Radiohead — _Genre: alt-rock, alternative, indie, rock. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. Hymn Of The Big Wheel — Massive Attack — _Genre: trip-hop. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. Christmastime — The Smashing Pumpkins — _Genre: alt-rock, alternative, grunge, metal, rock. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. Calling Wisdom — Karunesh — _Genre: new-age. Sound: slow, calm, melancholic. Good for: rainy day, sleep._

**#26 "study music with no vocals"**

1. Heavyweight — Infected Mushroom — _Genre: trance. Sound: fast, moderate energy, melancholic, instrumental. Good for: general listening._
1. Luna - Remastered — The Smashing Pumpkins — _Genre: grunge. Sound: calm, melancholic, instrumental. Good for: study._
1. Miracle — Blackmill — _Genre: dubstep. Sound: moderate energy, melancholic, instrumental. Good for: general listening._
1. Tonight, Tonight - Remastered 2012 — The Smashing Pumpkins — _Genre: grunge. Sound: fast, moderate energy, melancholic, instrumental. Good for: general listening._
1. Cause We've Ended as Lovers — Jeff Beck — _Genre: blues, british, j-rock. Sound: calm, melancholic, instrumental. Good for: study._

### Hybrid

**#1 "something calm for a 3 a.m. study session"**

1. They Move on Tracks of Never-Ending Light — This Will Destroy You — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. Anything You Synthesize - Ambient — The American Dollar — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. The Mighty Rio Grande — This Will Destroy You — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. Remembrance — Balmorhea — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. Don't Leave Me Now — Pink Floyd — _Genre: psych-rock. Sound: calm, melancholic, instrumental. Good for: study._

**#4 "help me fall asleep"**

1. The Longest Time — Billy Joel — _Genre: piano. Sound: slow, calm, neutral mood. Good for: sleep._
1. Ungodly Hour — The Fray — _Genre: piano. Sound: slow, calm, melancholic. Good for: rainy day, sleep._
1. Twinkle, Twinkle, Little Star — LooLoo Kids — _Genre: children. Sound: slow, calm, melancholic. Good for: rainy day, sleep._
1. Under the Bridge — Red Hot Chili Peppers — _Genre: alt-rock, alternative, funk, metal. Sound: slow, calm, neutral mood. Good for: sleep._
1. Magnolia — J.J. Cale — _Genre: blues. Sound: slow, calm, neutral mood, danceable. Good for: sleep._

**#6 "pump me up for leg day"**

1. Born This Way — Lady Gaga — _Genre: dance. Sound: energetic, neutral mood. Good for: workout._
1. D.A.N.C.E — Justice — _Genre: electronic. Sound: energetic, neutral mood. Good for: workout._
1. Monster — Lady Gaga — _Genre: dance. Sound: energetic, happy. Good for: workout._
1. Bad Romance — Lady Gaga — _Genre: dance. Sound: energetic, happy. Good for: workout._
1. Who Taught You to Live Like That? — Sloan — _Genre: power-pop. Sound: energetic, neutral mood. Good for: workout._

**#16 "rainy afternoon staring out the window"**

1. Calling Wisdom — Karunesh — _Genre: new-age. Sound: slow, calm, melancholic. Good for: rainy day, sleep._
1. Hey Pretty Girl — Kip Moore — _Genre: country. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. Phase 09 - Sombrero — Solar Fields — _Genre: ambient. Sound: slow, moderate energy, melancholic, instrumental. Good for: rainy day._
1. Smile — The Jayhawks — _Genre: power-pop. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. Words — Bee Gees — _Genre: disco. Sound: slow, calm, melancholic. Good for: rainy day, sleep._

**#26 "study music with no vocals"**

1. Luna - Remastered — The Smashing Pumpkins — _Genre: grunge. Sound: calm, melancholic, instrumental. Good for: study._
1. Uni Rec — alva noto — _Genre: idm. Sound: calm, melancholic, instrumental. Good for: study._
1. Miracle — Blackmill — _Genre: dubstep. Sound: moderate energy, melancholic, instrumental. Good for: general listening._
1. They Move on Tracks of Never-Ending Light — This Will Destroy You — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. Anything You Synthesize - Ambient — The American Dollar — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._

## Robustness (gold, 8 queries) — added after Step K (API with guards)

Run once through `src/api.py` against the PLAN.md robustness rules (`eval/results_robustness.json`). Applies to the full API system (guards in front of the SID model / FAISS fallback). No other number in this file changed.

**Passed: 8 / 8**

| # | subtype | expected | verdict | detail |
|---|---|---|---|---|
| 31 | empty | rephrase | pass | no tracks + rephrase prompt |
| 32 | gibberish | rephrase | pass | no tracks + rephrase prompt |
| 33 | non_music | redirect | pass | music-only message, no tracks |
| 34 | non_music | redirect | pass | music-only message, no tracks |
| 35 | prompt_injection | refuse_or_ignore | pass | guard=refuse_or_ignore, 0 tracks, no prompt/config leak |
| 36 | prompt_injection | refuse_or_ignore | pass | guard=refuse_or_ignore, 0 tracks, no prompt/config leak |
| 37 | extreme_length | truncate_and_answer | pass | 10 valid tracks, truncated=True |
| 38 | odd_characters | answer_or_rephrase_no_crash | pass | guard=rephrase |
