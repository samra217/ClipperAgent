from faster_whisper import WhisperModel
import librosa
audio, _ = librosa.load("data_insamling/audio/7oxlCKMlpZw.wav", sr=16000, mono=True)


model_size = "large-v3"

model = WhisperModel(model_size, device="mps", compute_type="float16")

segments, info = model.transcribe(audio)
segments = list(segments) 



import json
from pathlib import Path

vid = "7oxlCKMlpZw"
out = Path("data_insamling/transcripts")
out.mkdir(parents=True, exist_ok=True)

data = {
    "language": info.language,
    "duration": info.duration,
    "segments": [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ],
}
(out / f"{vid}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))