import os
import sys
import pathlib
import numpy as np
import pandas as pd
from utils import *

PATH = '/home/andriy/diploma/data/'
files = []
data = []
max_total = 0
abs_path = str(pathlib.Path().absolute())
os.system('clear')
for dir_name in os.listdir(PATH):
    for path, _, filenames in os.walk(PATH + dir_name):
        for file in filenames:
            files.append(os.path.join(path, file))
    df = pd.DataFrame(columns=['name', 'active_range', 'frquency', 'duration'])
    df['name'], df['active_range'], df['frquency'], df['duration'], rmax = post2df(files)
    max_total = max(rmax, max_total)
    pkl_path = abs_path + '/' + dir_name + '.pkl'
    df.to_pickle(pkl_path)
    #plt.savefig('demo.png', bbox_inches='tight')
    print(len(df), end='\n')
    files = []
length = len(files)
print('MAX_TOTAL: ', max_total)
