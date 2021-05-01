import numpy as np
import pandas as pd
import pathlib
import librosa
import os
from utils import mfccs, post2df
import matplotlib.pyplot as plt
from feature_extraction import *

NAME = 'p225_003'
PATH = f'/home/andriy/diploma/data/p225/{NAME}.wav'
df = pd.DataFrame(columns=['name', 'active_range', 'frquency', 'duration'])
df['name'], df['active_range'], df['frquency'], df['duration'], rmax = post2df([PATH])
print(df.head())
dataset = pd.DataFrame(columns='filename accent sex mfccs'.split())
for i, row in df.iterrows():
    dataset = dataset.append(extract_mfccs_from_file(row['name'], row['active_range'], row['frquency']))
print(np.array(dataset['mfccs'][0]).shape)
plt.savefig('demo.png', bbox_inches='tight')
# dataset = dataset.drop('filename accent sex'.split(), axis=1)
# dataset.to_pickle(f'/home/andriy/diploma/CNN/cnn/{NAME}.pkl')
