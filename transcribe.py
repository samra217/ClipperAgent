
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d


# Features
COLUMNS = [
    "words_per_s",
    "speech-frac",
    "excl_q_count",
    "hype_words"
]

HYPE_WORDS = {
    "wow",
    "omg",
    "insane",
    "crazy",
    "holy",
    "unbelievable",
    "lol",
    "haha",
    "goat",
    "fire",
    "lit",
    "bet",
    "huge",
    "game-changing",
    "revolutionary",
    "groundbreaking",
    "cutting-edge",
    "next-gen",
    "state-of-the-art",
    "massive",
    "wild",
    "unprecedented",
    "bussin",
    "gas",
    "hype",
    "glaze"
}

HYPE = {w.lower() for w in HYPE_WORDS}


# Paths
BASE_DIR = Path(__file__).resolve().parent

AUDIO = BASE_DIR / "data_insamling" / "audio"
OUT_JSON = BASE_DIR / "data_insamling" / "transcripts"
OUT_FEATURES = BASE_DIR / "data_insamling" / "features"


# Settings
N_BINS = 2000
SMOOTH_SECONDS = 2.0


def transcription_matrix(words, duration, n_bins=N_BINS):

    bin_len = duration / n_bins

    # Create matrix: (time, features)
    X = np.zeros((n_bins, len(COLUMNS)))

    # Extract features for each word
    for w in words:

        mid = (w["start"] + w["end"]) / 2
        b = min(max(int(mid / bin_len), 0), n_bins - 1)

        token = w["word"].lower().strip(".,!?\"'")

        X[b, 0] += 1
        X[b, 1] += max(w["end"] - w["start"], 0)
        X[b, 2] += w["word"].count("?") + w["word"].count("!")
        X[b, 3] += token in HYPE

    # Convert to rates/fractions
    X[:, 0] /= bin_len
    X[:, 1] = np.clip(X[:, 1] / bin_len, 0, 1)

    # Smooth features over approximately 2 seconds
    window_bins = max(1, round(SMOOTH_SECONDS / bin_len))

    X = uniform_filter1d(
        X,
        size=window_bins,
        axis=0,
        mode="nearest"
    )

    return X


def main():

    from faster_whisper import WhisperModel

    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    else:
        model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

    OUT_JSON.mkdir(parents=True, exist_ok=True)
    OUT_FEATURES.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(AUDIO.glob("*.wav"))

    #test only on one file
    wav_files = wav_files[:1]

    for wav in wav_files:

        vid = wav.stem

        out_json = OUT_JSON / f"{vid}.json"
        out_npz = OUT_FEATURES / f"{vid}_text.npz"

        if out_json.exists() and out_npz.exists():
            print(f"Skipping {vid}: files already exist")
            continue

        print(f"\nTranscribing: {vid}")

        segments, info = model.transcribe(
            str(wav),
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=True,
        )

        segs = []

        for s in segments:

            segs.append({
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "words": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word.strip()
                    }
                    for w in (s.words or [])
                ],
            })

        data = {
            "language": info.language,
            "duration": info.duration,
            "segments": segs
        }

        out_json.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8"
        )

        # Collect all words
        words = [
            w
            for s in segs
            for w in s["words"]
        ]

        # Generate features
        X = transcription_matrix(words, info.duration)

        np.savez(
            out_npz,
            X=X.T,
            cols=np.array(COLUMNS),
            duration=info.duration
        )

        # Print results
        print(f"Duration: {info.duration:.2f} seconds")
        print(f"Words: {len(words)}")
        print(f"Bin length: {info.duration / N_BINS:.3f} seconds")
        print(f"Smoothing window: {SMOOTH_SECONDS} seconds")
        print(f"Feature shape: {X.shape}")
        print(f"Saved: {out_npz}")

    print("\nDone!")


if __name__ == "__main__":
    main()
