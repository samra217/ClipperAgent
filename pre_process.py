import numpy as np
import librosa
import os


import numpy as np
import librosa

SR, HOP, N_BINS, STEPS_PER_BIN = 22050, 512, 100, 20
L = N_BINS * STEPS_PER_BIN  # 2000

                          # (14, 2000)


def fetch_MFCC(file_name):
    return np.load("data_insamling/mfcc/"+file_name+".npy")

def fetch_audio(file_name):
    y, sr = librosa.load("data_insamling/audio/"+file_name+".wav")
    return y,sr

def fetch_RMS(file_name):
    return np.load("data_insamling/RMS/"+file_name+"_rms.npy")



def pre_process_data(file_name):
    mfcc = fetch_MFCC(file_name)
    #[audio,sr] = fetch_audio(file_name)
    rms = fetch_RMS(file_name)

    return mfcc,rms



def build_input() -> np.ndarray:
    xs= []
    for file_name in os.listdir("data_insamling/audio/"):
        file_name = file_name[:-4]
        [mfcc,rms] = pre_process_data(file_name)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max, top_db=None)          # (1, T)
        x = np.concatenate([mfcc, rms_db], axis=0)                 #  (14, T)

    # z-normalisering per feature inom videon
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)

    # medelpoolning till fast längd L
        edges = np.linspace(0, x.shape[1], L + 1).astype(int)
        x = np.stack([x[:, a:max(b, a + 1)].mean(axis=1)
                  for a, b in zip(edges[:-1], edges[1:])], axis=1)
        xs.append(x)
        
    return np.array(xs)

if __name__ == "__main__":
    x = build_input()
    print(x.shape)
    np.save("training_data.npy", "./data_insamling/")