import numpy as np
import pandas as pd
import os

os.system('clear')
df = pd.read_pickle('../dataset.pkl')
df = df.reset_index()
df = df.drop('index', axis=1)
PATH = '/home/andriy/diploma/CNN/data/p225'
dictionary = {'': ['M', 'English'], 'p226': ['F', 'English'], 'p234': ['M', 'Scottish'], 'p237': ['F', 'Scottish'], 'p245': ['M', 'Irish'], 'p266': ['F', 'Irish'], 'p253': ['F', 'Welsh'], 'p306': ['F', 'NY'], 'p360': ['M', 'NY'],'p326': ['M', 'Australian'], 'p374': ['M', 'Australian'], 'p302': ['M', 'Canadian'], 'p303': ['F', 'Canadian'], 'p329': ['M', 'American'], 'p345': ['M', 'American']}

print(df.columns)

for k, v in dictionary.items():
    df.loc[df.filename.str.contains(k), 'sex'] = v[0]
    df.loc[df.filename.str.contains(k), 'accent'] = v[1]
df.drop(['mfccs'], axis=1)
var = df[df.filename.str.contains('p225')]
print(var)
df.to_pickle('dataset.pkl')