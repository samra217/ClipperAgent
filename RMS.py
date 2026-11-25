"""
RMS-features per segment. Videon delas i 100 lika långa segment.

Kör: python audio_rms.py ljud features
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 16_000      # samplingsfrekvens
HOP = 800        # 50 ms per RMS-ram vid 16 kHz
N_SEG = 100      # antal segment (som heatmapen)
N_FRAMES = 64    # RMS-värden per segment


def load_audio(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


def rms_features(audio):
    # RMS i dB per 50 ms-ram
    n = len(audio) // HOP
    frames = audio[: n * HOP].reshape(n, HOP)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-10)

    # Dela i 100 segment och sampla om varje segment till N_FRAMES värden
    X = np.array([np.interp(np.linspace(0, len(s) - 1, N_FRAMES), np.arange(len(s)), s)
                  for s in np.array_split(rms_db, N_SEG)])

    # Normalisera per video
    return ((X - X.mean()) / X.std()).astype(np.float32)


audio_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(exist_ok=True)

for path in sorted(audio_dir.iterdir()):
    X = rms_features(load_audio(path))
    np.save(out_dir / f"{path.stem}_rms.npy", X)
    print(path.stem, X.shape)