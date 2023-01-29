import sys
sys.path.append('.')
import random
from spatialFilters import *
from tqdm import tqdm
import os
import pandas as pds
import numpy as np
import pickle
from modeling import Code2EEG


# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
chnNames = ['PZ', 'PO5', 'POZ', 'PO3', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
seedNUM = int(1)
n_band = 1
targetNUM = 40
codespace = 40
saveFILE = 'trf.csv'
winLEN = 0.3
lag = 0
tmin = 0
tmax = 0.3
tau = np.arange(tmin,tmax+1/srate,1/srate)

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
    for tag in ['WN']:

        X = sub[tag]['X'][:, chnINX]
        y = sub[tag]['y'][:]
        S = sub[tag]['STI']

        _class = np.unique(y).tolist()


        X = X-np.mean(X, axis=-1, keepdims=True)

        code2EEG = Code2EEG(srate=srate,winLEN=winLEN,tmin=tmin,tmax=tmax,S=(S,_class),estimator=0.9
        ,padding=True,n_band=1,component=1)
        code2EEG.fit(X,y)

        trf = np.squeeze(code2EEG.trf)
        f = pd.DataFrame(data=trf, index=tau)
        f.reset_index(level=0, inplace=True)
        f = f.melt(id_vars='index', value_name='trf',
                   var_name='band')
        f = f.rename(columns={'index': 'tau'})
        f['tag'] = tag
        f['subject'] = subName
        frames.append(f)

    df = pd.concat(frames, axis=0, ignore_index=True)
    add = 'results/%s/%s' % (expName, subName)
    if not os.path.exists(add):
        os.makedirs(add)
    df.to_csv(add+os.sep+saveFILE)
