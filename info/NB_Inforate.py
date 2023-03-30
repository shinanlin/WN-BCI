
import sys
sys.path.append('.')
from sklearn import preprocessing
import numpy as np
import pickle
import pandas as pd
from compare.spatialFilters import TDCA,vanilla
from tqdm import tqdm
from compare.utils import returnFFT
import os
from scipy.integrate import simpson
#%%
# this script is for generating codespace for optimization

#%%
# parameters
srate = 250
n_band = 1
n_component = 1

expName = 'compare'
tags = ['WN','SSVEP']
saveFILE = 'NaB_info.csv'

#%% load dataset

dir = 'data/datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

chnNames = ['PZ', 'PO3','PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ','O2']

chnNames = wholeset[0]['channel']
chnINX = [wholeset[0]['channel'].index(i) for i in chnNames]

#%% refresh

add = 'results'+os.sep+expName
if os.path.exists(add):
    for fnames in os.listdir(add):
        f = add+os.sep+fnames+os.sep+saveFILE
        if os.path.exists(f):
            os.remove(f)
else:
    os.mkdir(add)

#%%

for i,sub in tqdm(enumerate(wholeset)):

    frames = []

    for tag in tags:
        
        X = sub[tag]['X'][:, chnINX]
        y = sub[tag]['y']
        subName = sub['name']

        # preprocessing
        _class = np.unique(y)
        N = X.shape[-1]
        winLEN = N/srate
        #%% build backward model

        model = vanilla(winLEN=winLEN,lag=0)
        aveEvoked = model.fit_transform(X,y)
        xX = model.transform(X)
        aveEvoked, xX = np.squeeze(aveEvoked), np.squeeze(xX)

        # %% padding 
        if winLEN<1:
            pad = np.zeros_like(aveEvoked)
            aveEvoked = np.concatenate([aveEvoked,pad],axis=-1)

            pad = np.zeros_like(xX)
            xX = np.concatenate([xX,pad],axis=-1)

        if winLEN>=1:
            aveEvoked = aveEvoked[:,:int(srate)]
            xX = xX[:,:int(srate)]

        #%% compute the upper bound
        # keep the first component
        

        aveEvoked = np.concatenate([aveEvoked[_class == i] for i in y])
        xNoise = xX - aveEvoked

        freqz, ss = returnFFT(aveEvoked)
        freqz, nn = returnFFT(xNoise)
        freqz, xx = returnFFT(xX)

        xPower = (1/(srate*N)) * (np.abs(xx)**2)
        sPower = (1/(srate*N)) * (np.abs(ss)**2)
        nPower = (1/(srate*N)) * (np.abs(nn)**2)
        
        ubSNR = sPower.mean(axis=0)/nPower.mean(axis=0)

        logSNR = np.log2(1+ubSNR)
        ubINFOrate = [simpson(logSNR[:n],freqz[:n]) for n in np.arange(1,len(freqz),1)]
        ubINFOrate.insert(0,0)
        ubINFO = ubINFOrate[-1]

        #%% record

        f = pd.DataFrame({

            'f': freqz,
            'ubSNR': np.abs(ubSNR),
            'ubrate': ubINFOrate,
            'sPower':np.abs(sPower).mean(axis=0),
            'nPower':np.abs(nPower).mean(axis=0),
            'X':np.abs(xPower).mean(axis=0),
            'exp': expName
            
        })

        f['subject'] = subName
        f['ubINFO'] = ubINFO
        f['tag'] = tag.upper()

        frames.append(f)

    df = pd.concat(frames, axis=0, ignore_index=True)
    add = 'results/%s/%s' % (expName, subName)
    if not os.path.exists(add):
        os.makedirs(add)
    df.to_csv(add+os.sep+saveFILE)
