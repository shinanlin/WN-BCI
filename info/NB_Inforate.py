
import sys
sys.path.append('.')

import numpy as np
import pickle
import pandas as pd
from compare.spatialFilters import TDCA
from tqdm import tqdm
from compare.modeling  import Code2EEG,EEG2Code
from compare.utils import returnFFT,returnPSD
import os

#%%
# this script is for generating codespace for optimization

#%%
# parameters
srate = 250
winLEN = 0.5
n_band = 1
n_component = 1
_t = np.arange(0,winLEN,1/srate)

expName = 'compare'
tags = ['WN', 'SSVEP']
saveFILE = 'NaB_info.csv'


#%% load dataset

dir = 'data/datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

chnNames = ['PZ', 'PO3','PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ','O2']
chnINX = [wholeset[0]['channel'].index(i) for i in chnNames]

#%% refresh

add = 'results'+os.sep+expName
for fnames in os.listdir(add):
    f = add+os.sep+fnames+os.sep+saveFILE
    if os.path.exists(f):
        os.remove(f)
#%%

for i,sub in tqdm(enumerate(wholeset)):

    frames = []

    for tag in tags:

        X = sub[tag]['X'][:, chnINX]
        y = sub[tag]['y']
        subName = sub['name']

        _class = np.unique(y)

        #%% build backward model

        model = TDCA(winLEN=winLEN,lag=0)
        aveEvoked = model.fit_transform(X,y)
        xX = model.transform(X)
        aveEvoked, xX = np.squeeze(aveEvoked), np.squeeze(xX)

        #%% compute the upper bound
        # keep the first component
        
        aveEvoked = np.concatenate([aveEvoked[_class == i] for i in y])
        xNoise = xX - aveEvoked

        # freqz, rF = returnFFT(np.squeeze(model.enhanced))
        # freqz, S_F = returnFFT(model.STI)
        # rConjugate = np.conjugate(rF)
        # Hf: frequency TRF
        # Hf = (rConjugate*S_F).mean(axis=0)/(rConjugate*rF).mean(axis=0)

        freqz, ss = returnFFT(aveEvoked)
        freqz, nn = returnFFT(xNoise)
        freqz, xx = returnFFT(xX)


        xPower = 1/(srate*winLEN) * (np.abs(xx)**2)
        sPower = 1/(srate*winLEN) * (np.abs(ss)**2)
        nPower = 1/(srate*winLEN) * (np.abs(nn)**2)
        
        ubSNR = sPower.mean(axis=0)/nPower.mean(axis=0)
        ubINFOrate = np.cumsum(np.log2(1+ubSNR))
        ubINFO = np.sum(np.log2(1+ubSNR))

        #%% record

        f = pd.DataFrame({

            'f': freqz,
            'ubSNR': np.abs(ubSNR),
            'ubrate': np.abs(ubINFOrate),
            'sPower':np.abs(sPower).mean(axis=0),
            'nPower':np.abs(nPower).mean(axis=0),
            'X':np.abs(xPower).mean(axis=0),
            
        })

        f['subject'] = subName
        f['ubINFO'] = ubINFO
        f['tag'] = tag

        frames.append(f)

    df = pd.concat(frames, axis=0, ignore_index=True)
    add = 'results/%s/%s' % (expName, subName)
    if not os.path.exists(add):
        os.makedirs(add)
    df.to_csv(add+os.sep+saveFILE)
