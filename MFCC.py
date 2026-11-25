import librosa
import numpy as np
import matplotlib.pyplot as plt
import os


def save_MFCC(file_name):
    y, sr = librosa.load("data_insamling/audio/"+file_name+".wav")
    mfccs = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)

    np.save("data_insamling/mfcc/"+file_name+".npy",mfccs)



def display_MFCC(file_name):
    mfcc = np.load("data_insamling/mfcc/"+file_name+".npy")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient')
    plt.show()


if __name__ == "__main__":
    for file_name in os.listdir("data_insamling/audio"):
        file_name = file_name[:-4]
        save_MFCC(file_name=file_name)