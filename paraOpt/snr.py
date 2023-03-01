import sys
sys.path.append('.')

from scipy.stats import zscore
import pickle
import numpy as np
import pandas as pds
import os
from tqdm import tqdm
from spatialFilters import *
import random
import utils
# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
chnNames = ['PZ', 'PO5', 'POZ', 'PO3', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
seedNUM = int(1)
n_band = 1
targetNUM = 40
saveFILE = 'snr.csv'
winLEN = 0.3
lag = 0.14
tmin=0
tmax=0.3
N = int(srate*winLEN)

# %%
dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)


# %%
for sub in tqdm(wholeset):

    frames = []

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    add = 'results/%s/%s' % (expName, subName)
    for tag in ['WN', 'SSVEP']:

        X = sub[tag]['X'][:, chnINX]
        y = sub[tag]['y'][:]
        _classes = np.unique(y)

        X = X-np.mean(X,axis=-1,keepdims=True)

        model = TDCA(winLEN=winLEN,lag=lag,srate=srate,montage=targetNUM,n_band=n_band)
        # filter signal
        enhanced = model.fit_transform(X,y)
        # filtered snr
        X_ = model.filter(X)
        
        for i, (evoked, _class) in enumerate(zip(enhanced, _classes)):

            epochs = X_[y == _class]

            for j, epoch in enumerate(epochs):

                noise = epoch-evoked

                s = np.abs(np.fft.fftshift(np.fft.fft(evoked, axis=-1)/N))**2
                n = np.abs(np.fft.fftshift(np.fft.fft(noise, axis=-1)/N))**2

                # break
                s_power = np.sum(s[:, N//2:], axis=-1)
                n_power = np.sum(n[:, N//2:], axis=-1)

                snr = 10*np.log10(s_power/n_power)

                f = pd.DataFrame({
                    'snr':snr,
                    'epoch':[j],
                    'class':[i]
                })
                
                f['method'] = tag
                f['subject'] = subName
                frames.append(f)
        
    df = pd.concat(frames, axis=0, ignore_index=True)
    add = 'results/%s/%s' % (expName, subName)
    if not os.path.exists(add):
        os.makedirs(add)
    df.to_csv(add+os.sep+saveFILE)
