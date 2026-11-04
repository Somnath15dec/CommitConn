# Discovery dial (run once)

Pre-registered in `eval/PLAN.md` note 5 (commit 919201a). Model `checkpoints/run1/epoch3`, trie-constrained beam 50, top 10; the user's history tracks are excluded. 200 real users, 400 queries. Means with bootstrap 95% CI over queries (1,000 resamples, seed 13).

| λ | novelty@10 | familiar hit@10 | relevance@10 |
|---|---|---|---|
| +4 | 0.002 [0.000, 0.004] | 0.015 [0.011, 0.020] | 0.725 [0.693, 0.756] |
| +2 | 0.020 [0.012, 0.029] | 0.015 [0.011, 0.020] | 0.732 [0.700, 0.762] |
| +0 | 0.685 [0.660, 0.708] | 0.013 [0.009, 0.017] | 0.811 [0.786, 0.835] |
| -2 | 1.000 [1.000, 1.000] | 0.007 [0.004, 0.009] | 0.826 [0.803, 0.847] |
| -4 | 1.000 [1.000, 1.000] | 0.007 [0.004, 0.009] | 0.826 [0.803, 0.847] |

## Pre-registered success criteria

- Novelty strictly increases from λ = +4 to −4: **FAIL**
- Relevance at λ = −4 ≥ 70% of relevance at λ = 0: **pass** (ratio 1.02)
- Familiar hit@10 is highest at λ = +4: **FAIL**

## Examples (same query, λ = +4 / 0 / −4, top 5)

**User ae001ef8b668a64b7d9c8b0281fabfe1** (78 history tracks; top history clusters: Energetic Punk Workout (3), Fast Energetic Punk Rock (3), Acoustic Folk Coffee (3)). Query: "music for the morning"

- λ = +4:
  1. Somewhere Only We Know — Keane — _Fast General Listening_ (history cluster)
  2. I'm Yours — Jason Mraz — _Happy Dance Morning_ (history cluster)
  3. Home Again — Michael Kiwanuka — _Calm Melancholic Acoustic_ (history cluster)
  4. Someday — The Strokes — _Happy Garage Energy_ (history cluster)
  5. Midnight City — M83 — _Energetic Melancholic Dance_ (history cluster)
- λ = +0:
  1. Boston — Augustana — _Melancholic Piano_
  2. Somewhere Only We Know — Keane — _Fast General Listening_ (history cluster)
  3. I'm Yours — Jason Mraz — _Happy Dance Morning_ (history cluster)
  4. Там, где избушка над лесом стояла — Галина Романова (Лёля Белая) и группа Первомайка — _Acoustic Chill_
  5. Ain't No Mountain High Enough — Marvin Gaye, Tammi Terrell — _Happy Morning Vibes_
- λ = -4:
  1. Boston — Augustana — _Melancholic Piano_
  2. Там, где избушка над лесом стояла — Галина Романова (Лёля Белая) и группа Первомайка — _Acoustic Chill_
  3. Ain't No Mountain High Enough — Marvin Gaye, Tammi Terrell — _Happy Morning Vibes_
  4. Budapest — George Ezra — _Melancholic Singer Songwriter_
  5. Landslide - 2018 Remaster — Fleetwood Mac — _Calm Acoustic Morning_

**User 0a65736e0eb11a4a05f0ba47f4fd2b5f** (42 history tracks; top history clusters: Energetic EDM Workout (6), Energetic Progressive House (4), Energetic Dubstep Workout (4)). Query: "music for the morning"

- λ = +4:
  1. Midnight City — M83 — _Energetic Melancholic Dance_ (history cluster)
  2. Your Body Is a Wonderland — John Mayer — _Energetic Folk Singer Songwriter_ (history cluster)
  3. Rolling in the Deep — Adele — _Energetic Party Dance_ (history cluster)
  4. Love Me Harder — Ariana Grande, The Weeknd — _Energetic Melancholic Dance_ (history cluster)
  5. Mess Is Mine — Vance Joy — _Energetic Melancholic Dance_ (history cluster)
- λ = +0:
  1. Karma Police — Radiohead — _Diverse Rock and Funk_
  2. Boston — Augustana — _Melancholic Piano_
  3. Somewhere Only We Know — Keane — _Fast General Listening_
  4. I'm Yours — Jason Mraz — _Happy Dance Morning_
  5. Там, где избушка над лесом стояла — Галина Романова (Лёля Белая) и группа Первомайка — _Acoustic Chill_
- λ = -4:
  1. Karma Police — Radiohead — _Diverse Rock and Funk_
  2. Boston — Augustana — _Melancholic Piano_
  3. Somewhere Only We Know — Keane — _Fast General Listening_
  4. I'm Yours — Jason Mraz — _Happy Dance Morning_
  5. Там, где избушка над лесом стояла — Галина Романова (Лёля Белая) и группа Первомайка — _Acoustic Chill_

