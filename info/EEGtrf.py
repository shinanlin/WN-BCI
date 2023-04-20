import sys
sys.path.append('.')
sys.path.append('./compare')

import pandas as pd
from utils import returnFFT
from compare.modeling import Code2EEG
import pickle
import numpy as np
import pandas as pds
import os
from tqdm import tqdm
from compare.spatialFilters import *
import random
import sys
sys.path.append('.')
sys.path.append('./compare')


# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'simuMEG'
n_band = 1
n_component = 1
saveFILE = 'ttrf.csv'
winLEN = 5
lag = 0
tmin = 0
tmax = 0.5
tau = np.arange(tmin, tmax+1/srate, 1/srate)

# %%
add = 'results'+os.sep+expName
if os.path.exists(add):
    for fnames in os.listdir(add):
        f = add+os.sep+fnames+os.sep+saveFILE
        if os.path.exists(f):
            os.remove(f)

# %%
dir = './data/datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)
# %%
for sub in tqdm(wholeset):

    temporal = []
    spectral = []

    subName = sub['name']
    chnINX = sub['channel']

    add = 'results/%s/%s' % (expName, subName)
    for tag in ['SSVEP']:

        chnNames = sub['channel']
        chnINX = [sub['channel'].index(i) for i in chnNames]
        X = sub[tag]['X'][:, chnINX]
        y = sub[tag]['y'][:]
        S = sub[tag]['STI']

        _class = np.unique(y).tolist()

        code2EEG = Code2EEG(srate=srate, winLEN=winLEN, tmin=tmin, tmax=tmax, S=(
            S, _class), estimator=0.9, padding=True, n_band=n_band, component=n_component)
        code2EEG.fit(X, y)

        trf = code2EEG.regressor.Csr.mean(axis=(0,1))

        # %%
        f = pd.DataFrame(data=trf, index=tau)
        f.reset_index(level=0, inplace=True)
        f = f.melt(id_vars='index', value_name='trf',
                   var_name='band')
        f = f.rename(columns={'index': 'tau'})
        f['tag'] = tag
        f['subject'] = subName
        f['exp'] = expName
        temporal.append(f)

        # %%
        freqz, ftrf = returnFFT(trf, srate=srate)

        f = pd.DataFrame({

            'f': freqz,
            'amp': np.abs(ftrf),

        })
        f['subject'] = subName
        f['tag'] = tag
        f['exp'] = expName

        spectral.append(f)

    add = 'results/%s/%s' % (expName, subName)
    if not os.path.exists(add):
        os.makedirs(add)

    df = pd.concat(temporal, axis=0, ignore_index=True)
    df.to_csv(add+os.sep+saveFILE)

    df = pd.concat(spectral, axis=0, ignore_index=True)
    df.to_csv(add+os.sep+'ftrf.csv')
