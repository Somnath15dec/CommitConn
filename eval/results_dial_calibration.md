# Dial calibration (descriptive)

Pre-registered in `eval/PLAN.md` note 6 (commit 52d788b); no pass/fail criteria. Same model, 200 users, 400 queries, beam 50, top 10, history tracks excluded, as the discovery-dial run (ad4fb22). Means with bootstrap 95% CI over queries.

| λ | novelty@10 | familiar hit@10 | relevance@10 | source |
|---|---|---|---|---|
| +2.0 | 0.020 [0.012, 0.029] | 0.015 [0.011, 0.020] | 0.732 [0.700, 0.762] | reference: ad4fb22 |
| +1.0 | 0.175 [0.154, 0.196] | 0.015 [0.012, 0.020] | 0.768 [0.739, 0.797] | calibration |
| +0.5 | 0.387 [0.359, 0.414] | 0.015 [0.011, 0.019] | 0.792 [0.765, 0.817] | calibration |
| +0.0 | 0.685 [0.660, 0.708] | 0.013 [0.009, 0.017] | 0.811 [0.786, 0.835] | reference: ad4fb22 |
| -0.5 | 0.899 [0.886, 0.911] | 0.009 [0.006, 0.012] | 0.818 [0.794, 0.841] | calibration |
| -1.0 | 0.980 [0.975, 0.984] | 0.007 [0.005, 0.010] | 0.826 [0.803, 0.847] | calibration |
| -2.0 | 1.000 [1.000, 1.000] | 0.007 [0.004, 0.009] | 0.826 [0.803, 0.847] | reference: ad4fb22 |

## Examples (same query, λ = +1 / +0.5 / −0.5 / −1, top 5)

**User ae001ef8b668a64b7d9c8b0281fabfe1** (78 history tracks; top history clusters: Energetic Punk Workout (3), Fast Energetic Punk Rock (3), Acoustic Folk Coffee (3)). Query: "music for the morning"

- λ = +1.0:
  1. Somewhere Only We Know — Keane — _Fast General Listening_ (history cluster)
  2. I'm Yours — Jason Mraz — _Happy Dance Morning_ (history cluster)
  3. Home Again — Michael Kiwanuka — _Calm Melancholic Acoustic_ (history cluster)
  4. Someday — The Strokes — _Happy Garage Energy_ (history cluster)
  5. Midnight City — M83 — _Energetic Melancholic Dance_ (history cluster)
- λ = +0.5:
  1. Somewhere Only We Know — Keane — _Fast General Listening_ (history cluster)
  2. I'm Yours — Jason Mraz — _Happy Dance Morning_ (history cluster)
  3. Home Again — Michael Kiwanuka — _Calm Melancholic Acoustic_ (history cluster)
  4. Someday — The Strokes — _Happy Garage Energy_ (history cluster)
  5. Midnight City — M83 — _Energetic Melancholic Dance_ (history cluster)
- λ = -0.5:
  1. Boston — Augustana — _Melancholic Piano_
  2. Там, где избушка над лесом стояла — Галина Романова (Лёля Белая) и группа Первомайка — _Acoustic Chill_
  3. Ain't No Mountain High Enough — Marvin Gaye, Tammi Terrell — _Happy Morning Vibes_
  4. Budapest — George Ezra — _Melancholic Singer Songwriter_
  5. Stand by Me — Ben E. King — _Calm Afrobeat and Soul_
- λ = -1.0:
  1. Boston — Augustana — _Melancholic Piano_
  2. Там, где избушка над лесом стояла — Галина Романова (Лёля Белая) и группа Первомайка — _Acoustic Chill_
  3. Ain't No Mountain High Enough — Marvin Gaye, Tammi Terrell — _Happy Morning Vibes_
  4. Budapest — George Ezra — _Melancholic Singer Songwriter_
  5. Landslide - 2018 Remaster — Fleetwood Mac — _Calm Acoustic Morning_

**User 0a65736e0eb11a4a05f0ba47f4fd2b5f** (42 history tracks; top history clusters: Energetic EDM Workout (6), Energetic Progressive House (4), Energetic Dubstep Workout (4)). Query: "music for the morning"

- λ = +1.0:
  1. Midnight City — M83 — _Energetic Melancholic Dance_ (history cluster)
  2. Your Body Is a Wonderland — John Mayer — _Energetic Folk Singer Songwriter_ (history cluster)
  3. Rolling in the Deep — Adele — _Energetic Party Dance_ (history cluster)
  4. Love Me Harder — Ariana Grande, The Weeknd — _Energetic Melancholic Dance_ (history cluster)
  5. Mess Is Mine — Vance Joy — _Energetic Melancholic Dance_ (history cluster)
- λ = +0.5:
  1. Midnight City — M83 — _Energetic Melancholic Dance_ (history cluster)
  2. Karma Police — Radiohead — _Diverse Rock and Funk_
  3. Your Body Is a Wonderland — John Mayer — _Energetic Folk Singer Songwriter_ (history cluster)
  4. Boston — Augustana — _Melancholic Piano_
  5. Somewhere Only We Know — Keane — _Fast General Listening_
- λ = -0.5:
  1. Karma Police — Radiohead — _Diverse Rock and Funk_
  2. Boston — Augustana — _Melancholic Piano_
  3. Somewhere Only We Know — Keane — _Fast General Listening_
  4. I'm Yours — Jason Mraz — _Happy Dance Morning_
  5. Там, где избушка над лесом стояла — Галина Романова (Лёля Белая) и группа Первомайка — _Acoustic Chill_
- λ = -1.0:
  1. Karma Police — Radiohead — _Diverse Rock and Funk_
  2. Boston — Augustana — _Melancholic Piano_
  3. Somewhere Only We Know — Keane — _Fast General Listening_
  4. I'm Yours — Jason Mraz — _Happy Dance Morning_
  5. Там, где избушка над лесом стояла — Галина Романова (Лёля Белая) и группа Первомайка — _Acoustic Chill_

