import os
import sys
import pathlib
import numpy as np
import pandas as pd
import librosa
from utils import *
from math import ceil
import pathlib
from utils import shade_silence

def extract_mfccs_from_file(fn, active, sr):
    df = pd.DataFrame(columns='filename accent sex mfccs'.split())
    y, sr = librosa.load(fn, sr=sr)
    for region in active:
        #print((region[1] - region[0]) / sr)
        region_dur = ceil((region[1] - region[0]) / sr)
        y_sec = np.zeros(sr * region_dur) # creating a padding array to full seconds
        y_sec[0:len(y[int(region[0]):int(region[1])])] = y[int(region[0]):int(region[1])]
        #print(y_sec)
        for i in range(0, len(y_sec), sr):
            mfccs_region = librosa.feature.mfcc(y_sec[i:i+sr], sr, n_mfcc=50)
            #print(len(mfccs_region))
            df = df.append({'filename': fn, 'mfccs': mfccs_region}, ignore_index=True)
        #print(k)
    return df

def show_spectrum(wregion, sr, disp=False, output=True):
    Xdb = librosa.amplitude_to_db(abs(X))
    if disp:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
    if output:
        return Xdb