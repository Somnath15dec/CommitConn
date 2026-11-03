# Data readiness report (Step 4.5)

Catalog: 10,000 tracks. Read-only analysis; catalog unchanged.

## 1. content_desc uniqueness

- Unique content_desc: **4,226 / 10,000**
- Tracks sharing their content_desc with at least one other track: 7,382 (73.8%)

10 largest identical groups:

| content_desc | tracks |
|---|---|
| Genre: grunge. Sound: fast, energetic, neutral mood. Good for: workout, running, gym. | 61 |
| Genre: grunge. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic. | 58 |
| Genre: grunge. Sound: energetic, neutral mood. Good for: workout. | 54 |
| Genre: new-age. Sound: calm, melancholic, instrumental, acoustic. Good for: study, relaxing, coffee, focus. | 41 |
| Genre: disco. Sound: energetic, happy, danceable. Good for: party, dance, feel-good. | 40 |
| Genre: grunge. Sound: energetic, melancholic. Good for: intense, cathartic. | 38 |
| Genre: psych-rock. Sound: moderate energy, happy. Good for: morning, feel-good. | 37 |
| Genre: ambient. Sound: calm, melancholic, instrumental, acoustic. Good for: study, relaxing, coffee, focus. | 37 |
| Genre: industrial. Sound: energetic, neutral mood. Good for: workout. | 36 |
| Genre: synth-pop. Sound: energetic, happy. Good for: workout. | 36 |

## 2. Duplicate songs (same normalized title + first artist)

- Groups: **35**, tracks involved: 77, largest group: 6

Examples (5 groups):

| track_id | track_name | artists | primary_genre | source |
|---|---|---|---|---|
| 7aOy14i2bhauyZl4r3TcP2 | 21 Reasons | Nathan Dawe;Ella Henderson | house | genre_fill |
| 4VnsZmAkvBEX5BWxMAWBBG | 21 Reasons | Nathan Dawe;Ella Henderson | house | genre_fill |
| 4BsasVXlffiJ69bRrJXWZO | 21 Reasons | Nathan Dawe;Ella Henderson | house | genre_fill |
| 2scEOSUnDnquAY1qWCpTL6 | 21 Reasons | Nathan Dawe;Ella Henderson | house | genre_fill |
| 5FImqqYMK6ZwLwClPDds8c | 21 Reasons | Nathan Dawe;Ella Henderson | house | genre_fill |
| 70TpTZFReW52csA4Az7alJ | 21 Reasons | Nathan Dawe;Ella Henderson | house | genre_fill |
| 6ORqU0bHbVCRjXm9AjyHyZ | Good Riddance (Time of Your Life) | Green Day | punk | muse |
| 7CmA9WWvO4CyeQb8sHQ0wB | Good Riddance (Time of Your Life) | Green Day | punk | muse |
| 5fVZC9GiM4e8vu99W0Xf6J | How to Save a Life | The Fray | piano | muse |
| 3OPVsn0Dk3s8vWhXxnK35p | How to Save a Life - Live in NYC - 2009 | The Fray | piano | muse |
| 2QbGkMyeZuALMT6UuM8aED | I Just Called | NEIKED;Anne-Marie;Latto | house | genre_fill |
| 1kmqlqDBu1fHGsrpzbBxif | I Just Called | NEIKED;Anne-Marie;Latto | house | genre_fill |
| 0H8Tclo4x4kbZruQtZFNSX | I Just Called | NEIKED;Anne-Marie;Latto | house | genre_fill |
| 3C1hNCIzLbYri8YoCPt6HV | I Just Called | NEIKED;Anne-Marie;Latto | house | genre_fill |
| 704LUEih663hiE3CnyUrFW | SEJODIOTO | KAROL G | reggaeton | genre_fill |
| 7guW1bVC4aTIjxCeZpjbsR | SEJODIOTO | KAROL G | reggae | genre_fill |
| 34bTBINpWPa360R72PDJj1 | SEJODIOTO | KAROL G | reggae | genre_fill |

## 3. Tracks per context word

Flag: under 150 tracks.

| context | tracks | flag |
|---|---|---|
| workout | 2313 |  |
| running | 2139 |  |
| gym | 2006 |  |
| coffee | 1975 |  |
| relaxing | 1885 |  |
| morning | 1845 |  |
| feel-good | 1812 |  |
| dance | 1329 |  |
| intense | 1072 |  |
| cathartic | 1066 |  |
| party | 808 |  |
| study | 418 |  |
| rainy day | 377 |  |
| focus | 291 |  |
| sleep | 288 |  |
| reflective | 127 | THIN |
| late night | 67 | THIN |
| wind down | 45 | THIN |

## 4. Query ambiguity (matching tracks per query intent)

- Single context word: min 45 / median 1,069 / max 2,313 (over 18 words)
- Mood + context combos (non-empty, 43 of 54): min 3 / median 337 / max 1,812
- Empty combos: 11

| context | happy | melancholic | neutral mood |
|---|---|---|---|
| workout | 1077 | 0 | 1236 |
| running | 854 | 501 | 784 |
| gym | 727 | 495 | 784 |
| coffee | 433 | 878 | 664 |
| relaxing | 343 | 878 | 664 |
| morning | 1277 | 282 | 286 |
| feel-good | 1812 | 0 | 0 |
| dance | 1109 | 33 | 187 |
| intense | 0 | 1072 | 0 |
| cathartic | 0 | 1066 | 0 |
| party | 588 | 33 | 187 |
| study | 12 | 337 | 69 |
| rainy day | 0 | 377 | 0 |
| focus | 8 | 230 | 53 |
| sleep | 21 | 170 | 97 |
| reflective | 0 | 127 | 0 |
| late night | 3 | 49 | 15 |
| wind down | 3 | 27 | 15 |

## 5. Embedding options

Model `sentence-transformers/all-MiniLM-L6-v2` on NVIDIA GeForce RTX 4060 Laptop GPU. Neighbours by Euclidean distance (A: normalized text vector; B: normalized content_desc vector + 0.5 x min-max audio features ['energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'danceability', 'speechiness']). SID preview: 3 x 256 residual k-means (30 Lloyd iterations, torch).

### A: full desc: 10 random tracks and their 5 nearest neighbours

- **Anything You Synthesize - Ambient** (ambient): Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study, focus, late night.
    - Says (ambient): Genre: ambient. Sound: calm, melancholic, instrumental, acoustic. Good for: study, relaxing, coffee, focus.
    - The Choice (ambient): Genre: ambient. Sound: slow, calm, neutral mood, instrumental, acoustic. Good for: study, relaxing, coffee, sleep.
    - The Living Daylights (synth-pop): Genre: synth-pop. Sound: energetic, happy. Good for: general listening.
    - Quiet (ambient): Genre: ambient. Sound: moderate energy, melancholic, instrumental. Good for: general listening.
    - Light (ambient): Genre: ambient. Sound: moderate energy, melancholic. Good for: general listening.
- **Angerfist - Dance With the Wolves - The Sickest Squad-Remix** (happy): Spotify category: happy. Sound: fast, energetic, melancholic, spoken word or rap. Good for: running, intense, gym, cathartic.
    - Emotional Crank (chill): Spotify category: chill. Sound: calm, neutral mood, acoustic, danceable. Good for: relaxing, coffee, morning.
    - Guerrilla Radio (metal): Genre: metal. Sound: energetic, neutral mood. Good for: workout.
    - Pretty Rave Girl (happy): Spotify category: happy. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Gunfight (grunge): Genre: grunge. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Alone (sad): Spotify category: sad. Sound: moderate energy, neutral mood, danceable, spoken word or rap. Good for: general listening.
- **10th Floor Ghost Girl** (trip-hop): Genre: trip-hop. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Ghost (dance): Genre: dance. Sound: moderate energy, neutral mood. Good for: general listening.
    - Ghostbusters (disco): Genre: disco. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Giving Up The Ghost (trip-hop): Genre: trip-hop. Sound: energetic, neutral mood, instrumental. Good for: general listening.
    - Ghost On The Dance Floor (punk): Genre: punk, punk-rock, rock. Sound: energetic, neutral mood. Good for: workout.
    - Ghost of Days Gone By (grunge): Genre: grunge. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
- **Untrue** (groove): Genre: groove. Sound: energetic, neutral mood, instrumental. Good for: workout.
    - Unreal (trip-hop): Genre: trip-hop. Sound: energetic, neutral mood, instrumental. Good for: workout.
    - History (trip-hop): Genre: trip-hop. Sound: energetic, neutral mood. Good for: workout.
    - One (grunge): Genre: grunge. Sound: energetic, neutral mood. Good for: workout.
    - R. Mullen (trip-hop): Genre: trip-hop. Sound: energetic, neutral mood, instrumental. Good for: workout.
    - Breaking the Habit (alternative): Genre: alternative, grunge, metal. Sound: energetic, neutral mood. Good for: workout.
- **Vele** (pop-film): Genre: pop-film. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Otha Sollaala (pop-film): Genre: pop-film. Sound: energetic, happy, acoustic, danceable. Good for: party, dance, coffee, feel-good.
    - Tere Bina (pop): Genre: pop, pop-film. Sound: fast, moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Raanjhanaa (pop): Genre: pop, pop-film. Sound: energetic, happy. Good for: general listening.
    - Hua Hain Aaj Pehli Baar (From "Sanam Re") (pop-film): Genre: pop-film. Sound: fast, energetic, happy. Good for: running, gym.
    - Arere Vaanaa (k-pop): Genre: k-pop. Sound: fast, moderate energy, happy, danceable. Good for: dance, morning, feel-good.
- **Little Criminals** (disney): Genre: disney. Sound: fast, energetic, neutral mood. Good for: running, gym.
    - Smooth Criminal (grunge): Genre: grunge, metal. Sound: energetic, happy. Good for: workout.
    - Little Monster (garage): Genre: garage. Sound: energetic, neutral mood. Good for: workout.
    - The Scare Floor (disney): Genre: disney. Sound: moderate energy, neutral mood. Good for: general listening.
    - Little Room (garage): Genre: garage. Sound: fast, energetic, happy. Good for: running, gym.
    - Animals (pop): Genre: pop. Sound: fast, energetic, neutral mood. Good for: running, gym.
- **I Remember / Stranger Than You Dreamt It** (show-tunes): Genre: show-tunes. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Wishing You Were Somehow Here Again (show-tunes): Genre: show-tunes. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - The Mirror (Angel Of Music) (show-tunes): Genre: show-tunes. Sound: slow, calm, melancholic, acoustic. Good for: relaxing, rainy day, coffee, sleep.
    - I Remember You (hard-rock): Genre: hard-rock, metal. Sound: moderate energy, neutral mood. Good for: general listening.
    - I Dreamed a Dream (opera): Genre: opera. Sound: slow, moderate energy, melancholic, acoustic. Good for: rainy day, coffee, reflective, relaxing.
    - Angel Of Music (show-tunes): Genre: show-tunes. Sound: slow, calm, melancholic, acoustic. Good for: relaxing, rainy day, coffee, sleep.
- **Just Jammin'** (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Hit That Jive (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Muy Tranquilo (trip-hop): Genre: trip-hop. Sound: moderate energy, happy. Good for: morning, feel-good.
    - Superstylin' (trip-hop): Genre: trip-hop. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - History (trip-hop): Genre: trip-hop. Sound: energetic, neutral mood. Good for: workout.
    - Let Me Show Ya (trip-hop): Genre: trip-hop. Sound: moderate energy, happy. Good for: morning, feel-good.
- **Finally** (deep-house): Genre: deep-house, disco, garage. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Mission (j-dance): Genre: j-dance. Sound: energetic, happy. Good for: general listening.
    - Lady - Hear Me Tonight (disco): Genre: disco, house. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Just Fine (soul): Genre: soul. Sound: energetic, neutral mood, danceable. Good for: party, dance.
    - Paradise (british): Genre: british. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - King In A Catholic Style (Wake Up) (synth-pop): Genre: synth-pop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
- **Hounds Of Love** (piano): Genre: piano. Sound: fast, energetic, happy. Good for: running, gym.
    - Heartbeat (piano): Genre: piano. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - You Found Me (piano): Genre: piano. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Gravity Of Love (new-age): Genre: new-age. Sound: fast, energetic, neutral mood. Good for: running, gym.
    - Wuthering Heights (piano): Genre: piano. Sound: moderate energy, neutral mood. Good for: general listening.
    - Cecilia And The Satellite (piano): Genre: piano. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.

### B: content_desc + audio: 10 random tracks and their 5 nearest neighbours

- **Anything You Synthesize - Ambient** (ambient): Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study, focus, late night.
    - Remembrance (ambient): Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study, focus, late night.
    - They Move on Tracks of Never-Ending Light (ambient): Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study, focus, late night.
    - The Mighty Rio Grande (ambient): Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study, focus, late night.
    - The Big Ship - 2004 Digital Remaster (ambient): Genre: ambient. Sound: calm, melancholic, instrumental. Good for: study, focus, late night.
    - The Wings (ambient): Genre: ambient. Sound: slow, calm, neutral mood, instrumental. Good for: study, sleep, focus, late night.
- **Angerfist - Dance With the Wolves - The Sickest Squad-Remix** (happy): Spotify category: happy. Sound: fast, energetic, melancholic, spoken word or rap. Good for: running, intense, gym, cathartic.
    - Eardrumz - Partyraiser & Repix Remix (happy): Spotify category: happy. Sound: fast, energetic, melancholic, spoken word or rap. Good for: running, intense, gym, cathartic.
    - Sanctity of Space : Cassini’s Journey (happy): Spotify category: happy. Sound: fast, energetic, melancholic, spoken word or rap. Good for: running, intense, gym, cathartic.
    - Sanctity Of Space - Edit Version (happy): Spotify category: happy. Sound: fast, energetic, melancholic, spoken word or rap. Good for: running, intense, gym, cathartic.
    - One (Always Hardcore) - Live in Hamburg (happy): Spotify category: happy. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
    - Endless Summer (happy): Spotify category: happy. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
- **10th Floor Ghost Girl** (trip-hop): Genre: trip-hop. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Don't Stop (trip-hop): Genre: trip-hop. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Hide U (trip-hop): Genre: trip-hop. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Listen (trip-hop): Genre: trip-hop. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Sing It Back (trip-hop): Genre: trip-hop. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Superstylin' (trip-hop): Genre: trip-hop. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
- **Untrue** (groove): Genre: groove. Sound: energetic, neutral mood, instrumental. Good for: workout.
    - 5 Minutes Alone (groove): Genre: groove. Sound: energetic, neutral mood. Good for: workout.
    - Blood and Thunder (groove): Genre: groove, hard-rock, metal. Sound: energetic, neutral mood, instrumental. Good for: workout.
    - Born in Winter (groove): Genre: groove. Sound: moderate energy, melancholic, instrumental. Good for: general listening.
    - Black Tongue (groove): Genre: groove. Sound: fast, energetic, melancholic, instrumental. Good for: running, intense, gym, cathartic.
    - Black Label (groove): Genre: groove, metalcore. Sound: energetic, neutral mood, instrumental. Good for: workout.
- **Vele** (pop-film): Genre: pop-film. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - O O Jaane Jaana (pop-film): Genre: pop-film. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Sheila Ki Jawani (pop-film): Genre: pop-film. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Dil Kya Kare (pop-film): Genre: pop-film. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Tere Bina (pop): Genre: pop, pop-film. Sound: fast, moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Aaj Se Teri (pop-film): Genre: pop-film. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
- **Little Criminals** (disney): Genre: disney. Sound: fast, energetic, neutral mood. Good for: running, gym.
    - Calling All the Monsters (disney): Genre: disney. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Zero To Hero (disney): Genre: disney. Sound: fast, energetic, happy. Good for: workout, running, gym.
    - Człowiekiem gdybym był (disney): Genre: disney. Sound: energetic, neutral mood. Good for: workout.
    - Dig a Little Deeper (disney): Genre: disney. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Voy A Ser el Rey León (disney): Genre: disney. Sound: energetic, happy. Good for: general listening.
- **I Remember / Stranger Than You Dreamt It** (show-tunes): Genre: show-tunes. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - All I Ask Of You (show-tunes): Genre: show-tunes. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Wishing You Were Somehow Here Again (show-tunes): Genre: show-tunes. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - All to Me (show-tunes): Genre: show-tunes. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - A Whole New World (show-tunes): Genre: show-tunes. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Falling Slowly (show-tunes): Genre: show-tunes. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
- **Just Jammin'** (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Black Is the Night (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Hit That Jive (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Aqualung (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - You're Always the Sun (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Don't Call It Love - 12" Version (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
- **Finally** (deep-house): Genre: deep-house, disco, garage. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Lola's Theme - Radio Edit (deep-house): Genre: deep-house. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - Palmtree (deep-house): Genre: deep-house. Sound: energetic, happy, danceable. Good for: party, dance, feel-good.
    - WTF (feat. Amber Van Day) - Tujamo Remix (deep-house): Genre: deep-house. Sound: energetic, neutral mood, danceable. Good for: party, dance.
    - Fade Out Lines (deep-house): Genre: deep-house. Sound: moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Promesses - Radio Edit (deep-house): Genre: deep-house, groove. Sound: energetic, neutral mood, danceable. Good for: party, dance.
- **Hounds Of Love** (piano): Genre: piano. Sound: fast, energetic, happy. Good for: running, gym.
    - Movin' Out (Anthony's Song) (piano): Genre: piano. Sound: fast, energetic, happy. Good for: workout, running, gym.
    - I'm Still Standing (piano): Genre: piano. Sound: fast, energetic, happy. Good for: workout, running, gym.
    - You May Be Right (piano): Genre: piano. Sound: fast, energetic, happy. Good for: workout, running, gym.
    - All The Right Moves (piano): Genre: piano. Sound: fast, energetic, happy. Good for: workout, running, gym.
    - Crocodile Rock (piano): Genre: piano. Sound: fast, energetic, happy. Good for: running, gym.

### SID preview

| option | collision_rate | colliding_groups | max_tracks_per_sid | l1_used | l1_largest | l1_median |
|---|---|---|---|---|---|---|
| A: full desc | 0.065 | 298 | 6 | 256 | 172 | 32 |
| B: content_desc + audio | 0.287 | 1119 | 14 | 256 | 140 | 34 |

### Neighbour agreement over all tracks (5-NN)

| option | same_primary_genre | share_context_word | identical_content_desc |
|---|---|---|---|
| A: full desc | 0.603 | 0.679 | 0.104 |
| B: content_desc + audio | 0.916 | 0.968 | 0.512 |
