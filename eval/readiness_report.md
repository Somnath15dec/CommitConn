# Data readiness report (Step 4.5, v2 after Step 4.6 catalog changes)

Catalog: 10,000 tracks. Read-only analysis; catalog unchanged.

## 1. content_desc uniqueness

- Unique content_desc: **4,235 / 10,000**
- Tracks sharing their content_desc with at least one other track: 7,364 (73.6%)

10 largest identical groups:

| content_desc | tracks |
|---|---|
| Genre: grunge. Sound: fast, energetic, neutral mood. Good for: workout, running, gym. | 61 |
| Genre: grunge. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic. | 58 |
| Genre: grunge. Sound: energetic, neutral mood. Good for: workout. | 54 |
| Genre: new-age. Sound: calm, melancholic, instrumental, acoustic. Good for: study, relaxing, coffee, morning. | 41 |
| Genre: disco. Sound: energetic, happy, danceable. Good for: party, dance, feel-good. | 40 |
| Genre: grunge. Sound: energetic, melancholic. Good for: intense, cathartic. | 38 |
| Genre: psych-rock. Sound: moderate energy, happy. Good for: morning, feel-good. | 37 |
| Genre: ambient. Sound: calm, melancholic, instrumental, acoustic. Good for: study, relaxing, coffee, morning. | 37 |
| Genre: industrial. Sound: energetic, neutral mood. Good for: workout. | 36 |
| Genre: synth-pop. Sound: energetic, happy. Good for: workout. | 36 |

## 2. Duplicate songs (same normalized title + first artist)

- Groups: **0**, tracks involved: 0, largest group: 0

Examples (5 groups):

| track_id | track_name | artists | primary_genre | source |
|---|---|---|---|---|

## 3. Tracks per context word

Flag: under 150 tracks.

| context | tracks | flag |
|---|---|---|
| workout | 2311 |  |
| running | 2141 |  |
| gym | 2128 |  |
| morning | 2044 |  |
| coffee | 1973 |  |
| relaxing | 1874 |  |
| feel-good | 1711 |  |
| dance | 1343 |  |
| intense | 1067 |  |
| cathartic | 1063 |  |
| party | 830 |  |
| study | 431 |  |
| rainy day | 373 |  |
| sleep | 297 |  |

## 4. Query ambiguity (matching tracks per query intent)

- Single context word: min 297 / median 1,527 / max 2,311 (over 14 words)
- Mood + context combos (non-empty, 33 of 42): min 16 / median 502 / max 1,711
- Empty combos: 9

| context | happy | melancholic | neutral mood |
|---|---|---|---|
| workout | 1076 | 0 | 1235 |
| running | 850 | 502 | 789 |
| gym | 841 | 499 | 788 |
| morning | 1281 | 441 | 322 |
| coffee | 452 | 858 | 663 |
| relaxing | 356 | 857 | 661 |
| feel-good | 1711 | 0 | 0 |
| dance | 1121 | 36 | 186 |
| intense | 0 | 1067 | 0 |
| cathartic | 0 | 1063 | 0 |
| party | 608 | 36 | 186 |
| study | 16 | 348 | 67 |
| rainy day | 0 | 373 | 0 |
| sleep | 22 | 169 | 106 |

## 5. Embedding options

Model `sentence-transformers/all-MiniLM-L6-v2` on NVIDIA GeForce RTX 4060 Laptop GPU. Neighbours by Euclidean distance (A: normalized full-desc vector; B: normalized content_desc vector + w x min-max audio features ['energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'danceability', 'speechiness']). SID preview: 3 x 256 residual k-means (30 Lloyd iterations, torch).

### AUDIO_WEIGHT sweep (option B) vs option A

| option | collision_rate | max_tracks_per_code | l1_used | l1_largest | nn5_same_context | nn5_same_genre | nn5_identical_content_desc |
|---|---|---|---|---|---|---|---|
| A: full desc | 0.103 | 6 | 256 | 152 | 0.678 | 0.600 | 0.105 |
| B: w=0.5 | 0.278 | 15 | 256 | 116 | 0.969 | 0.914 | 0.513 |
| B: w=1.0 | 0.115 | 9 | 256 | 139 | 0.957 | 0.838 | 0.441 |
| B: w=1.5 | 0.067 | 4 | 256 | 139 | 0.948 | 0.731 | 0.370 |
| B: w=2.0 | 0.043 | 4 | 256 | 112 | 0.946 | 0.619 | 0.311 |

**Recommended AUDIO_WEIGHT = 2.0 (highest weight with 5-NN same context >= 90%)**

### A: full desc: 10 random tracks and their 5 nearest neighbours

- **Dissolved Girl** (trip-hop): Genre: trip-hop. Sound: fast, moderate energy, happy, instrumental. Good for: morning, feel-good.
    - Girl I Love You (trip-hop): Genre: trip-hop. Sound: energetic, neutral mood, instrumental. Good for: workout.
    - These Early Days (trip-hop): Genre: trip-hop. Sound: fast, moderate energy, happy. Good for: morning, feel-good.
    - Cherry Blossom Girl (ambient): Genre: ambient, trip-hop. Sound: energetic, neutral mood, instrumental. Good for: general listening.
    - Goodbye Girl (power-pop): Genre: power-pop, synth-pop. Sound: fast, moderate energy, happy, danceable. Good for: dance, morning, feel-good.
    - Wrong (trip-hop): Genre: trip-hop. Sound: moderate energy, neutral mood, danceable. Good for: general listening.
- **Kaattu** (malay): Genre: malay. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
    - Kabira (k-pop): Genre: k-pop. Sound: moderate energy, neutral mood, acoustic. Good for: coffee, relaxing.
    - Ummaal Koodaatha (malay): Genre: malay. Sound: fast, energetic, neutral mood. Good for: running, gym.
    - Angekkal Vere Onnineyum (malay): Genre: malay. Sound: moderate energy, neutral mood. Good for: general listening.
    - Thedinen (malay): Genre: malay. Sound: moderate energy, neutral mood, danceable. Good for: general listening.
    - Innalekalil (malay): Genre: malay. Sound: moderate energy, neutral mood. Good for: general listening.
- **Happy Holiday** (jazz): Genre: jazz, soul. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - The Christmas Waltz (jazz): Genre: jazz, soul. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - It's Christmas Time Again (jazz): Genre: jazz, soul. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - My Funny Valentine (jazz): Genre: jazz. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - What A Wonderful World (jazz): Genre: jazz. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Blue Moon (blues): Genre: blues, jazz. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
- **Frosty The Snowman** (blues): Genre: blues, jazz. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Winter Wonderland (blues): Genre: blues, jazz. Sound: calm, happy, acoustic. Good for: relaxing, coffee, morning.
    - Let It Snow! Let It Snow! Let It Snow! (blues): Genre: blues, jazz. Sound: slow, moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Winter Wonderland (jazz): Genre: jazz. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
    - Let It Snow! Let It Snow! Let It Snow! (jazz): Genre: jazz. Sound: calm, happy, acoustic. Good for: relaxing, coffee, morning.
    - Winter Wonderland (jazz): Genre: jazz. Sound: moderate energy, neutral mood. Good for: general listening.
- **Ainda Estás Aqui** (gospel): Genre: gospel. Sound: fast, moderate energy, neutral mood. Good for: general listening.
    - Tua Unção - Ao Vivo (gospel): Genre: gospel. Sound: moderate energy, melancholic. Good for: general listening.
    - Ao Teu Encontro / Manso e Suave - Ao Vivo (gospel): Genre: gospel. Sound: moderate energy, melancholic. Good for: general listening.
    - As Portas Vão se Abrir (gospel): Genre: gospel. Sound: moderate energy, melancholic. Good for: general listening.
    - Se Até os Anjos se Prostram (brazil): Genre: brazil, gospel. Sound: fast, moderate energy, neutral mood, instrumental. Good for: general listening.
    - No Céu Não Entra Pecado (gospel): Genre: gospel. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
- **Why Should The Fire Die?** (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Best Of Luck (bluegrass): Genre: bluegrass. Sound: energetic, happy. Good for: general listening.
    - Hanging By A Thread (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Beauty And The Mess (bluegrass): Genre: bluegrass. Sound: moderate energy, neutral mood, acoustic. Good for: coffee, relaxing.
    - Sweet Afton (bluegrass): Genre: bluegrass. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
    - The Lighthouse’s Tale (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
- **Du e för fin för mig** (psych-rock): Genre: psych-rock. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
    - Ge mig en öl (swedish): Genre: swedish. Sound: energetic, happy. Good for: workout.
    - En famn för mig (swedish): Genre: swedish. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Kriget med mig själv (punk): Genre: punk, swedish. Sound: fast, energetic, neutral mood. Good for: running, gym.
    - Baile De Los Pobres (afrobeat): Genre: afrobeat. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Ping Pong (trance): Genre: trance. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
- **Honky Tonk Badonkadonk** (country): Genre: country. Sound: energetic, happy. Good for: workout.
    - Honky Tonkin' (honky-tonk): Genre: honky-tonk. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Back up Buddy (honky-tonk): Genre: honky-tonk. Sound: calm, happy, acoustic, danceable. Good for: dance, relaxing, coffee, feel-good.
    - Lost Highway (honky-tonk): Genre: honky-tonk. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Wretches and Kings (grunge): Genre: grunge. Sound: energetic, neutral mood. Good for: workout.
    - Country Girl (Shake It For Me) (country): Genre: country. Sound: energetic, happy. Good for: workout.
- **(You Don't Know) How Glad I Am** (jazz): Genre: jazz, soul. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - What A Wonderful World (jazz): Genre: jazz. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Happy Holiday (jazz): Genre: jazz, soul. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - I'm In The Mood For Love (jazz): Genre: jazz. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Perhaps, Perhaps, Perhaps (j-pop): Genre: j-pop, jazz. Sound: calm, happy, acoustic. Good for: relaxing, coffee, morning.
    - Feeling Good (j-pop): Genre: j-pop, jazz. Sound: moderate energy, neutral mood. Good for: general listening.
- **Hero (feat. Josey Scott)** (metal): Genre: metal. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Hero (alt-rock): Genre: alt-rock, alternative, metal. Sound: fast, energetic, happy. Good for: workout, running, gym.
    - Hero (hardstyle): Genre: hardstyle. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
    - Heroes (breakbeat): Genre: breakbeat. Sound: fast, energetic, happy. Good for: workout, running, gym.
    - My Hero (grunge): Genre: grunge, metal. Sound: energetic, neutral mood. Good for: workout.
    - Heroes (grunge): Genre: grunge. Sound: fast, energetic, happy. Good for: workout, running, gym.

### B: w=2.0: 10 random tracks and their 5 nearest neighbours

- **Dissolved Girl** (trip-hop): Genre: trip-hop. Sound: fast, moderate energy, happy, instrumental. Good for: morning, feel-good.
    - Tricky Two (trip-hop): Genre: trip-hop. Sound: fast, moderate energy, happy, instrumental. Good for: morning, feel-good.
    - Organ Donor - Extended Overhaul (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, instrumental. Good for: morning, feel-good.
    - So Long (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, instrumental. Good for: morning, feel-good.
    - All In Forms (trip-hop): Genre: trip-hop. Sound: moderate energy, neutral mood, instrumental, danceable. Good for: general listening.
    - Rush Minute (trip-hop): Genre: trip-hop. Sound: moderate energy, happy, instrumental. Good for: morning, feel-good.
- **Kaattu** (malay): Genre: malay. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
    - Tham Kirubai and He Is Lord (malay): Genre: malay. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
    - Ente Maathram (malay): Genre: malay. Sound: moderate energy, neutral mood, acoustic. Good for: coffee, relaxing.
    - Big Black Car (folk): Genre: folk. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
    - You And I (acoustic): Genre: acoustic. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
    - Någon gång måste du bli själv (swedish): Genre: swedish. Sound: calm, neutral mood, acoustic. Good for: relaxing, coffee, morning.
- **Happy Holiday** (jazz): Genre: jazz, soul. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Almost Like Being In Love - Remastered (jazz): Genre: jazz. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Blue Skies (blues): Genre: blues, jazz. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Don't Let Me Be Misunderstood (blues): Genre: blues, jazz. Sound: moderate energy, neutral mood, acoustic. Good for: coffee, relaxing.
    - Sinnerman (blues): Genre: blues, jazz. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - The Times They Are A-Changin' (folk): Genre: folk. Sound: moderate energy, neutral mood, acoustic. Good for: coffee, relaxing.
- **Frosty The Snowman** (blues): Genre: blues, jazz. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Buona Sera - Remastered 2002 (blues): Genre: blues, jazz. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Let It Snow! Let It Snow! Let It Snow! (blues): Genre: blues, jazz. Sound: slow, moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - I've Got My Love To Keep Me Warm (blues): Genre: blues, jazz. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - I'd Rather Go Blind (blues): Genre: blues, jazz. Sound: slow, moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Deck The Halls (jazz): Genre: jazz. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
- **Ainda Estás Aqui** (gospel): Genre: gospel. Sound: fast, moderate energy, neutral mood. Good for: general listening.
    - O Poder do Teu Amor (The Power Of Your Love) - Ao Vivo (gospel): Genre: gospel. Sound: moderate energy, neutral mood. Good for: general listening.
    - Tu És a Fonte (gospel): Genre: gospel. Sound: moderate energy, melancholic. Good for: general listening.
    - Como Ninguém Me Vê - Ao Vivo (gospel): Genre: gospel. Sound: fast, energetic, neutral mood. Good for: running, gym.
    - Deus de Promessas - Acústico (gospel): Genre: gospel. Sound: moderate energy, neutral mood. Good for: general listening.
    - Minha Luta (gospel): Genre: gospel. Sound: moderate energy, melancholic. Good for: general listening.
- **Why Should The Fire Die?** (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Hanging By A Thread (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - April the 14th Part 1 (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Brand New Sidewalk (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Missy (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
    - Find My Love (bluegrass): Genre: bluegrass. Sound: calm, melancholic, acoustic. Good for: relaxing, coffee, morning.
- **Du e för fin för mig** (psych-rock): Genre: psych-rock. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
    - Man It Feels Like Space Again (psych-rock): Genre: psych-rock. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
    - Live And Let Die (hard-rock): Genre: hard-rock, metal. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
    - Ich tu dir weh (industrial): Genre: industrial. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
    - Can You Feel My Heart (metal): Genre: metal, metalcore, rock. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Who Was In My Room Last Night? (grunge): Genre: grunge, hardcore, psych-rock, punk-rock. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.
- **Honky Tonk Badonkadonk** (country): Genre: country. Sound: energetic, happy. Good for: workout.
    - Boot Scootin' Boogie (country): Genre: country. Sound: fast, energetic, happy. Good for: workout, running, gym.
    - Mere Dholna (Arijit Version) (indian): Genre: indian. Sound: energetic, happy. Good for: workout.
    - Hardum Humdum (indian): Genre: indian. Sound: energetic, happy. Good for: workout.
    - Crazy (british): Genre: british. Sound: energetic, happy. Good for: workout.
    - 9 to 5 (country): Genre: country. Sound: energetic, happy. Good for: general listening.
- **(You Don't Know) How Glad I Am** (jazz): Genre: jazz, soul. Sound: moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Joy To The World - Remastered 1999 (jazz): Genre: jazz. Sound: calm, happy, acoustic. Good for: relaxing, coffee, morning.
    - I'd Rather Go Blind (blues): Genre: blues, jazz. Sound: slow, moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - L-O-V-E (jazz): Genre: jazz. Sound: slow, moderate energy, happy, acoustic. Good for: morning, coffee, feel-good, relaxing.
    - Something's Gotta Give - Single Version (jazz): Genre: jazz. Sound: calm, happy, acoustic. Good for: relaxing, coffee, morning.
    - Quando, Quando, Quando (j-pop): Genre: j-pop, jazz. Sound: moderate energy, neutral mood, acoustic. Good for: coffee, relaxing.
- **Hero (feat. Josey Scott)** (metal): Genre: metal. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Seize the Day (alternative): Genre: alternative, metal. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Mother (hard-rock): Genre: hard-rock, metal. Sound: fast, energetic, neutral mood. Good for: running, gym.
    - Almost Easy (metal): Genre: metal. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Mi Digna Lealtad (heavy-metal): Genre: heavy-metal. Sound: fast, energetic, neutral mood. Good for: workout, running, gym.
    - Never Say Goodbye (metal): Genre: metal. Sound: fast, energetic, melancholic. Good for: running, intense, gym, cathartic.

## 6. Hit@10 oracle ceiling

500 tracks (random_state=7); per track min(1, 10 / size of its identical-content_desc group).

- **Mean ceiling: 0.893**
- Tracks with ceiling 1.0: 77.8%; median group size 4, max 61
