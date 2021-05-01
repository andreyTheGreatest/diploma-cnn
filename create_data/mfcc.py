import numpy as np
import pandas as pd
import pathlib
import librosa
import os
from utils import mfccs, post2df
import matplotlib.pyplot as plt
from feature_extraction import *

os.system('clear')
# get active range
files = []
abs_path = str(pathlib.Path().absolute()) + '/data'

dataset = pd.DataFrame(columns='filename accent sex mfccs'.split())
filenames = [f for f in os.listdir(abs_path)]
k = 0
for file in filenames:
    df = pd.read_pickle(os.path.join(abs_path, file))
    for i, row in df.iterrows():
        dataset = dataset.append(extract_mfccs_from_file(row['name'], row['active_range'], row['frquency']))
        count_progress(i, len(df) - 1, file)
    k += 1
    
print(dataset.head)
pkl_path = abs_path + '/dataset.pkl'
dataset.to_pickle(pkl_path)
# csv_path = abs_path + '/dataset.csv'
# dataset.to_csv(csv_path)
plt.show()