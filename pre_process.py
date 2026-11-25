import numpy as np
import librosa


def fetch_MFCC(file_name):
    return np.load("data_insamling/mfcc/"+file_name+".npy")

def fetch_audio(file_name):
    y, sr, = librosa.load("data_insamling/audio/"+file_name+".wav")
    return y



def pre_process_data():
    mfcc = fetch_MFCC("7oxlCKMlpZw")
    audio = fetch_audio("7oxlCKMlpZw")
    print(mfcc +"\n"+ audio)


if __name__ == "__main__":
    pre_process_data()