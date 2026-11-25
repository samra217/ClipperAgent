import librosa
import numpy as np
import matplotlib.pyplot as plt
import os


def save_RMS(file_name):
    y, sr = librosa.load("data_insamling/audio/"+file_name+".wav")
    rms = librosa.feature.rms(y=y,hop_length=512)

    np.save("data_insamling/RMS/"+file_name+"_rms.npy",rms)



if __name__ == "__main__":
    for file_name in os.listdir("data_insamling/audio"):
        file_name = file_name[:-4]
        save_RMS(file_name=file_name)