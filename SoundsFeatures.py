
import urllib.request
from pathlib import Path

import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent
AUDIO = BASE_DIR / "data_insamling" / "audio"
OUT = BASE_DIR / "data_insamling" / "features"

# Settings
N_BINS = 2000        
SR = 32000           
CHUNK_S = 60          

PANNS_DIR = Path.home() / "panns_data"
LABELS_CSV = PANNS_DIR / "class_labels_indices.csv"
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
CHECKPOINT = PANNS_DIR / "Cnn14_DecisionLevelMax.pth"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"

# Audioset class groups
GROUPS = {
    "laughter": ["Laughter", "Giggle", "Belly laugh", "Chuckle, chortle", "Snicker"],
    "applause": ["Applause", "Cheering", "Clapping", "Crowd"],
    "scream":   ["Screaming", "Shout", "Yell", "Whoop"],
    "gasp_cry": ["Gasp", "Crying, sobbing", "Sigh", "Groan"],
    "impact":   ["Explosion", "Gunshot, gunfire", "Smash, crash", "Bang", "Air horn, truck horn"],
    "speech":   ["Speech"],
    "music":    ["Music"],
    "silence":  ["Silence"],
}


def ensure_panns_files():
    
    PANNS_DIR.mkdir(parents=True, exist_ok=True)
    if not LABELS_CSV.exists():
        print("Laddar ner etikettfilen ...")
        urllib.request.urlretrieve(LABELS_URL, LABELS_CSV)
    if not CHECKPOINT.exists() or CHECKPOINT.stat().st_size < 3e8:
        print("Laddar ner PANNs-vikter (~320 MB) ...")
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)


def bin_frames(frames, duration, n_bins=N_BINS):
    
    T = frames.shape[0]
    t = (np.arange(T) + 0.5) * duration / T          # ramarnas mittpunkter i sekunder
    b = np.minimum((t / duration * n_bins).astype(int), n_bins - 1)   # växande, så bins är sammanhängande
    lo = np.searchsorted(b, np.arange(n_bins), side="left")
    hi = np.searchsorted(b, np.arange(n_bins), side="right")
    mx = np.zeros((n_bins, frames.shape[1]))
    mean = np.zeros_like(mx)
    for i in range(n_bins):
        seg = frames[lo[i]:hi[i]]
        if len(seg):
            mx[i] = seg.max(axis=0)
            mean[i] = seg.mean(axis=0)
    return mx, mean


def group_frames(frames, labels):
    
    idx = {name: i for i, name in enumerate(labels)}
    cols, out = [], []
    for g, names in GROUPS.items():
        ids = [idx[n] for n in names if n in idx]
        missing = [n for n in names if n not in idx]
        if missing:
            print(f"[varning] okända etiketter i gruppen {g}: {missing}")
        if not ids:
            continue
        out.append(frames[:, ids].max(axis=1))
        cols.append(g)
    return np.stack(out, axis=1), cols


def run_panns(sed, y):
    
    chunk = CHUNK_S * SR
    bounds = [(s, min(s + chunk, len(y))) for s in range(0, len(y), chunk)]
    
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < SR:
        last = bounds.pop()
        bounds[-1] = (bounds[-1][0], last[1])
    out = []
    for s, e in bounds:
        out.append(sed.inference(y[s:e][None, :])[0])   # (frames, 527)
    return np.concatenate(out, axis=0)


def main():
    ensure_panns_files()

    import librosa
    import torch
    from panns_inference import SoundEventDetection, labels

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kör på: {device}")
    sed = SoundEventDetection(checkpoint_path=None, device=device)

    OUT.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(AUDIO.glob("*.wav"))
    print(f"Hittade {len(wav_files)} wav-filer i {AUDIO}")

    for wav in wav_files:
        vid = wav.stem
        out = OUT / f"{vid}_sounds.npz"
        if out.exists():
            print(f"Skipping {vid}: file already exists")
            continue

        print(f"PANNs: {vid}")
        y, _ = librosa.load(wav, sr=SR, mono=True)
        duration = len(y) / SR
        frames = run_panns(sed, y)                       # (T, 527)

        G, gcols = group_frames(frames, labels)
        g_max, g_mean = bin_frames(G, duration)
        X = np.concatenate([g_max, g_mean], axis=1).T    # (2*G, N_BINS)
        cols = [f"{c}_max" for c in gcols] + [f"{c}_mean" for c in gcols]

        full, _ = bin_frames(frames, duration)
        np.savez(out, X=X, cols=np.array(cols), full=full.T, duration=duration)
        print(f"  Sparade {out.name}: X {X.shape}, full {full.T.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()