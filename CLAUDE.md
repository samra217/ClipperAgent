# ClipperAgent

Kursprojekt (LiU, AI för sociala medier). Verktyget tar en lång YouTube-video
(podd, intervju, talkshow, sketch, standup) och föreslår korta klipp (15–60 s)
som passar för TikTok.

## Arbetssätt

- Svara på svenska.
- Koden ska vara **så enkel som möjligt och göra jobbet**. Inga failsafes,
  argparse eller specialfall i förväg – vi lägger till sådant när ett problem
  faktiskt dyker upp.
- Använd färdiga biblioteksfunktioner (t.ex. `librosa.feature.rms`) i stället
  för att skriva egna, och säg till när en sådan finns.
- Ett skript gör en sak. Feature-skript läser ljud/transkript och skriver
  features – de rör inte heatmaps (heatmaps läses in i träningskoden).
- Formell notation och stegvisa härledningar uppskattas vid tekniska förklaringar.

## Pipeline (plan)

1. **Datainsamling** – yt-dlp: ladda ner ljud + heatmap ("most replayed").
2. **Transkribering** – Whisper med tidsstämplar per ord.
3. **Features per segment** – videon delas i 100 lika långa segment
   (t_k = k·D/100, samma uppdelning som yt-dlp:s heatmap). Ljud: RMS-kurva,
   MFCC/log-mel. Text: från transkriptet.
4. **Egen modell (CNN)** – förutsäger heatmapens 100 värden ur features.
   Heatmapen är facit (y) vid träning; vid inferens finns ingen heatmap.
5. **Klippgränser** – ta toppregionen från modellen, välj start/slut.
   Först regelbaserat (meningsslut + paus > 0,3 s + talarbyte) som baslinje,
   sedan ev. LLM som väljer bland meningsindex (aldrig fria tidsstämplar).
   Klipplängd 15–60 s.
6. **Klippning** – FFmpeg, beskärning till 9:16, inbrända undertexter.

Publicering via TikToks API är inte aktuell: ogranskade API-klienter kan bara
posta privat. Export som MP4 räcker.

## Data

- `data_insamling/` (ignoreras av git):
  - `audio/<video_id>.wav`
  - `mfcc/<video_id>.npy`
  - `transcripts/<video_id>.json` – `{language, duration, segments:[{start,end,text}]}`
- Heatmap från yt-dlp: `info["heatmap"]` = lista med 100
  `{start_time, end_time, value}`; värdena är relativa inom videon.
- Filnamn = YouTube-video-ID överallt, så att ljud, heatmap och features
  kan matchas.

## Features

- **RMS** (`RMS.py`): mono 16 kHz, RMS i dB per 50 ms-ram (`HOP = 800`),
  delas med `np.array_split` i 100 segment, varje segment omsamplas till 64
  värden → `X ∈ ℝ^{100×64}`, z-score-normaliserat per video.
  Kör: `python RMS.py <ljudmapp> <utmapp>` → `<id>_rms.npy`.
  Librosa-variant: `librosa.feature.rms(y=audio, frame_length=800,
  hop_length=800, center=False)` ger identiskt resultat. Använd inte
  `librosa.amplitude_to_db` utan `top_db=None` (klipper annars vid 80 dB).
- **MFCC** (`MFCC.py`): `librosa.feature.mfcc`, 13 koefficienter.

## Modell och utvärdering – beslut och tankar

- Träna på 100 segment per video. ~100 videor → ~10 000 segment, men
  segment inom samma video är starkt korrelerade, så den effektiva
  datamängden är mycket mindre. Antalet videor/kanaler är flaskhalsen.
- Dela upp train/test **per kanal**, inte per segment eller video.
- Heatmapen är relativ inom videon → jämför inom video (parvis rankning
  eller normalisering per video) snarare än absoluta värden.
  Början av videon har ofta höga värden bara för att alla börjar där.
- Gör en inlärningskurva (25/50/75/100 % av videorna) för att se om mer
  data behövs.
- Baslinjer att jämföra mot: slumpmässigt segment, zero-shot-LLM på hela
  transkriptet, vår modell, vår modell + klippgränser.
  Mått: top-k-träff mot heatmapens toppar eller NDCG@k.
- Begränsning att nämna i rapporten: bara populära videor har heatmap.

## Miljö

- MacBook Air M5, Python i `.venv`.
- API-nyckel (Gemini) i `.env` som `GEMINI_API_KEY`, läses med
  `load_dotenv(override=True)` och skickas explicit:
  `genai.Client(api_key=os.environ["GEMINI_API_KEY"])`.
  Nyckeln har nya formatet `AQ.…`. Skriv aldrig ut nyckeln.
- Whisper på Mac: **faster-whisper kan inte använda GPU på Mac**
  (CTranslate2 saknar Metal/MPS, `device="mps"` ger
  `ValueError: unsupported device mps`). Använd `device="cpu",
  compute_type="int8"`, eller mlx-whisper / whisper.cpp för GPU.
  Alternativ: transkribera allt en gång i Google Colab med GPU.

## Kända problem i koden just nu

- `test.py`: `WhisperModel(..., device="mps", compute_type="float16")`
  fungerar inte (se ovan).
- `MFCC.py`: `librosa.load` utan `sr` laddar i 22 050 Hz medan RMS använder
  16 kHz, och MFCC delas ännu inte i 100 segment – behöver samma
  segmentformat som RMS för att kunna matas in i CNN:en.
- `pre_process.py`: `print(mfcc + "\n" + audio)` försöker addera
  numpy-arrayer med en sträng → TypeError.
- `RMS.py`: docstringen säger `audio_rms.py`.
- `.gitignore` ignorerar `/venv/` men mappen heter `.venv`.
- `CNN.py` och `requirements.txt` är tomma.

## Juridik/etik

Använd egna, CC-licensierade eller tillåtna videor där det går. Spara bara
det som behövs (ljud, transkript, heatmap). Stäm av med handledaren.
