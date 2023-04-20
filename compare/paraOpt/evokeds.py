import sys
sys.path.append('.')

from scipy.stats import zscore
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from compare.spatialFilters import *
import random
import compare.utils as utils
# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
chnNames = ['PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3',
            'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2']
seedNUM = int(1)
n_band = 1
targetNUM = 40
codespace = 40
saveFILE = 'evoked.csv'
winLEN = 0.5
lag = 0
T = np.arange(0,winLEN,step=1/srate)

# %%
dir = './data/datasets/%s.pickle' % expName
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

        model = TDCA(winLEN=winLEN,lag=lag,srate=srate,montage=codespace,n_band=n_band)
        enhanced = model.fit_transform(X,y)

        # lowpass
        enhanced = np.squeeze(enhanced)
        # enhanced = zscore(enhanced,axis=-1)

        f = pd.DataFrame(data=enhanced,index=np.unique(y),columns=T)
        f.reset_index(level=0, inplace=True)
        f = f.melt(id_vars='index', value_name='evoked',
                            var_name='time')
        f = f.rename(columns={'index': 'class'})
        f['tag'] = tag
        f['subject'] = subName
        f['filtered'] = 'no'
        frames.append(f)

        enhancedF = utils.lowFilter(enhanced, srate=srate, band=[4, 15])
        f = pd.DataFrame(data=enhancedF, index=np.unique(y), columns=T)
        f.reset_index(level=0, inplace=True)
        f = f.melt(id_vars='index', value_name='evoked',
                   var_name='time')
        f = f.rename(columns={'index': 'class'})
        f['filtered'] = 'yes'
        f['tag'] = tag
        f['subject'] = subName

        frames.append(f)

    df = pd.concat(frames, axis=0, ignore_index=True)
    add = 'results/%s/%s' % (expName, subName)
    if not os.path.exists(add):
        os.makedirs(add)
    df.to_csv(add+os.sep+saveFILE)

