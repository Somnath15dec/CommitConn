# FAISS baseline results (frozen)

System: `sentence-transformers/all-MiniLM-L6-v2` (normalized) on content_desc only, IndexFlatIP (exact cosine), encoder on cpu. Scored per `eval/PLAN.md` (Amendments 1-5). Means with bootstrap 95% CI over queries (1000 resamples, seed 13).

## Summary

| Metric | Value |
|---|---|
| **Eval B playlist precision@10 (headline)** | **0.157 [0.124, 0.195]** |
| Eval B chance | 0.118 |
| Eval B condensed precision@10 | 0.381 [0.329, 0.436] |
| Eval B judged tracks in top 50 (mean) | 20.3 |
| Eval A Hit@10 | 0.096 [0.072, 0.122] |
| Eval A NDCG@10 | 0.047 [0.034, 0.062] |
| Eval A ceiling, query-aware (primary) | 0.092 |
| Eval A ceiling, identical description | 0.902 |
| Gold tag-match@10 | 0.634 [0.501, 0.750] |
| Gold avoid-violation@10 (mean of 3) | 0.100 (#26: 0.0, #27: 0.3, #28: 0.0) |
| Robustness | n/a (guards come later) |
| Valid-item rate | 1.000 |
| Latency p50 / p90 (ms, batch 1) | 9.0 / 10.2 |
| End-to-end latency incl. explanation | n/a |

Eval B pool: 7,058 tracks (no tune-half label). Latency note: encoder on CPU; Ollama paraphrase run was using the GPU concurrently.

## Eval B per context word

| Context word | precision@10 [95% CI] | chance | lift vs chance | condensed p@10 | judged in top 50 |
|---|---|---|---|---|---|
| coffee | 0.017 [0.000, 0.042] | 0.036 | 0.5x | 0.098 | 13.4 |
| dance | 0.483 [0.333, 0.667] | 0.222 | 2.2x | 0.917 | 23.4 |
| gym | 0.158 [0.100, 0.225] | 0.134 | 1.2x | 0.467 | 26.4 |
| morning | 0.075 [0.033, 0.125] | 0.069 | 1.1x | 0.217 | 20.2 |
| party | 0.175 [0.125, 0.217] | 0.222 | 0.8x | 0.608 | 16.9 |
| rainy day | 0.008 [0.000, 0.025] | 0.033 | 0.3x | 0.025 | 17.8 |
| relaxing | 0.317 [0.167, 0.467] | 0.236 | 1.3x | 0.714 | 19.2 |
| running | 0.050 [0.008, 0.092] | 0.101 | 0.5x | 0.317 | 21.1 |
| sleep | 0.058 [0.017, 0.100] | 0.050 | 1.2x | 0.200 | 17.4 |
| study | 0.050 [0.025, 0.083] | 0.035 | 1.4x | 0.067 | 20.2 |
| workout | 0.333 [0.208, 0.467] | 0.164 | 2.0x | 0.558 | 27.8 |

## Sanity view (gold queries, top 5)

**#1 "something calm for a 3 a.m. study session"**

1. Remembrance — Balmorhea — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. They Move on Tracks of Never-Ending Light — This Will Destroy You — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. The Mighty Rio Grande — This Will Destroy You — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. The Big Ship - 2004 Digital Remaster — Brian Eno — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._
1. Anything You Synthesize - Ambient — The American Dollar — _Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study._

**#4 "help me fall asleep"**

1. Straight From The Heart — Bryan Adams — _Genre: singer-songwriter, songwriter. Sound: slow, calm, neutral mood. Good for: sleep._
1. Rainy Night in Georgia — Randy Crawford — _Genre: disco. Sound: slow, calm, neutral mood. Good for: sleep._
1. Angel Of The Morning — Juice Newton — _Genre: country. Sound: slow, calm, neutral mood. Good for: sleep._
1. The Longest Time — Billy Joel — _Genre: piano. Sound: slow, calm, neutral mood. Good for: sleep._
1. You've Lost That Loving Feeling - Live at The International Hotel, Las Vegas — Elvis Presley — _Genre: rock-n-roll, rockabilly. Sound: slow, calm, neutral mood. Good for: sleep._

**#6 "pump me up for leg day"**

1. Monster — Lady Gaga — _Genre: dance. Sound: energetic, happy. Good for: workout._
1. Bad Romance — Lady Gaga — _Genre: dance. Sound: energetic, happy. Good for: workout._
1. Firework — Katy Perry — _Genre: dance. Sound: energetic, happy. Good for: workout._
1. Last Friday Night (T.G.I.F.) — Katy Perry — _Genre: dance. Sound: energetic, happy. Good for: workout._
1. Heart Attack — Demi Lovato — _Genre: dance. Sound: fast, energetic, neutral mood. Good for: running, gym._

**#16 "rainy afternoon staring out the window"**

1. Nadhiyae Nadhiyae — Unni Menon — _Genre: pop-film. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. Dreaming of You — Cigarettes After Sex — _Genre: ambient, indie-pop. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. 漁火閃閃 — Albert Au — _Genre: cantopop. Sound: slow, calm, melancholic. Good for: rainy day, sleep._
1. The Swordfight — Andrew Lloyd Webber, Gerard Butler, Patrick Wilson, Emmy Rossum — _Genre: show-tunes. Sound: slow, moderate energy, melancholic. Good for: rainy day._
1. Melt My Heart To Stone — Adele — _Genre: british. Sound: slow, moderate energy, melancholic. Good for: rainy day._

**#26 "study music with no vocals"**

1. In A Beautiful Place Out In The Country — Boards of Canada — _Genre: trip-hop. Sound: calm, melancholic, instrumental. Good for: study._
1. Les Nuits — Nightmares On Wax — _Genre: trip-hop. Sound: calm, melancholic, instrumental. Good for: study._
1. Hommage — DAVID AUGUST, Wanja — _Genre: minimal-techno. Sound: calm, melancholic, instrumental. Good for: study._
1. Velvet — DAVID AUGUST — _Genre: minimal-techno. Sound: calm, melancholic, instrumental. Good for: study._
1. Thrown — Kiasmos — _Genre: minimal-techno. Sound: calm, melancholic, instrumental, danceable. Good for: study._

