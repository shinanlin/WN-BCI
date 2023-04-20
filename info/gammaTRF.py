import sys
sys.path.append('.')
sys.path.append('./compare')

import random
from compare.spatialFilters import *
from tqdm import tqdm
import os
import pandas as pds
import numpy as np
import pickle
from compare.modeling import Code2EEG
from utils import returnFFT
import pandas as pd


# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'gamma'
chnNames = ['PZ', 'PO5', 'POZ', 'PO3', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
n_band = 1
n_component = 1
saveFILE = 'ttrf.csv'
winLEN = 3
lag = 0
tmin = 0
tmax = 0.5
tau = np.arange(tmin,tmax+1/srate,1/srate)
 
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
    chnINX = [78, 75, 71, 72, 77, 76, 87, 90]

    add = 'results/%s/%s' % (expName, subName)
    for tag in ['WN']:

        X = sub[tag]['X'][:, chnINX]
        y = sub[tag]['y'][:]
        S = sub[tag]['STI'][0]

        _class = np.unique(y).tolist()

        code2EEG = Code2EEG(srate=srate,winLEN=winLEN,tmin=tmin,tmax=tmax,S=(S,_class),estimator=0.98
                            , padding=True, n_band=n_band, component=n_component)
        code2EEG.fit(X,y)

        trf = code2EEG.trf.mean(axis=0)

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
        freqz, ftrf= returnFFT(trf,srate=srate)

        f = pd.DataFrame({

            'f': freqz,
            'amp': np.abs(ftrf),

        })
        f['subject']=subName
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
