import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display as display
import sklearn
import pandas as pd
import numpy as np
from itertools import compress
import os
from collections import Counter

def mfccs(audio, start, end, show=False):
    x, sr = librosa.load(audio)
    if (end - start) / sr < 5:
        y = np.zeros(sr * 5) # 5 seconds length for all
    else: y = np.zeros(sr * 7) # 10 sec - final measure
    y[0:len(x[start:end])] = x[start:end]
    mfccs = librosa.feature.mfcc(y, sr)
    if show: librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    return mfccs

def findsilence(y,sr,ind_i):
    hop = int(round(sr*0.1)) # hop and width define search window (sr - Hz in a second, num to multiply - time window)
    width = sr*0.1
    n_slice = int(len(y)/hop)
    starts = np.arange(n_slice)*hop
    ends = starts+width
    if hop != width:
        cutoff = np.argmax(ends>len(y))
        starts = starts[:cutoff]
        ends = ends[:cutoff]
        n_slice = len(starts)
    # not_talking = np.dot(y,y)/len(y) < 0.0001
    # if not_talking: 
    #     mask = np.full(n_slice, True)
    # else: 
    mask = list(map(lambda i: np.dot(y[int(starts[i]):int(ends[i])],y[int(starts[i]):int(ends[i])])/width, range(n_slice))) < np.dot(y,y)/len(y) * 0.048
    
    # print(np.dot(y,y)/len(y), not_talking, sep='\n')
    starts =  list(compress(starts+ind_i,mask))
    ends = list(compress(ends+ind_i,mask))
    return zip(starts,ends)

def merger(tulist):
    tu=()
    for tt in tulist:
        tu += tt
    cnt = Counter(tu)
    res = list(filter(lambda x: cnt[x] < 2, tu))
    return list(map(lambda x: tuple(x),np.array(res).reshape((int(len(res)/2),2))))

def shade_silence(filename,start=0,end=None,disp=True,output=False, itr=''):
    '''
    Find signal (as output) or silence (as shaded reagion  in plot) in a audio file
    filename: (filepath) works best with .wav format
    start/end: (float or int) start/end time for duration of interest in second (default= entire length)
    disp: (bool) whether to display a plot(default= True)
    output: (bool) whether to return an output (default = False)
    itr: (int) iteration use for debugging purpose
    '''
    try:
        y, sr = librosa.load(filename, sr=44100)
    except:
        print(itr, ' : librosa.load failed for '+filename)

    t = np.arange(len(y))/sr

    i = int(round(start * sr))
    if end != None:
        j = int(round(end * sr))
    else:
        j = len(y)
    fills = findsilence(y[i:j],sr,i)
    if disp:
        fig, ax = plt.subplots()
        ax.set_title(filename)
        ax.plot(t[i:j],y[i:j])
    if fills != None:
        shades = list(map(lambda x: (max(x[0],i),min(x[1],j)), fills))
        if len(shades)>0:
            shades = merger(shades)
            if disp:
                for s in shades:
                    ax.axvspan(s[0]/sr, s[1]/sr, alpha=0.5, color='red')
    if len(shades)>1:
        live = list(map(lambda i: (shades[i][1],shades[i+1][0]), range(len(shades)-1)))
    elif len(shades)==1:
        a = [i,shades[0][0],shades[0][1],j]
        live = list(filter(lambda x: x != None, map(lambda x: tuple(x) if x[0]!=x[1] else None,np.sort(a).reshape((int(len(a)/2),2)))))
    else:
        live = [(i,j)]
    if output:
        return live, sr, len(y)

def region_max(regions):
    dur_list = []
    for start, end in regions:
        dur_list.append(end - start)
    #print(dur_list, end=' ')
    try:
        return max(dur_list)
    except ValueError:
        return 0

def post2df(filepathlist):
    active = []
    srs = []
    dur = []
    fns = []
    rmax = 0
    i = 0
    for fn in filepathlist:
        tlist, sr , leng = shade_silence(fn, disp=False, output=True, itr=i)
        #tlist = list(filter(lambda x: x[1] - x[0] > 0.5 * sr, tlist))
        rmax = max(region_max(tlist), rmax)
        #print(fn[-17:])
        active.append(tlist)
        srs.append(sr)
        dur.append(leng/sr)
        fns.append(fn)
        i += 1
        count_progress(i, len(filepathlist), fn[-17:-13])
        # if i == 3: break # temp
    print('MAX DURATION (sec): ', rmax)
    return fns, active, srs, dur, rmax

def count_progress(i, length, fn):
    if i == 1: print(f'Processing {fn}...')
    progress = int(i / length * 100)
    print(int(progress / 5) * '▓', (20 - int(progress / 5)) * '░', progress, '%', sep='', end="\r")
    if i >= length:
        print(f'\nProcessed {i} of {length}')
    