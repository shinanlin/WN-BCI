
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
from collections import Counter
#%%
# this script is for generating codespace for optimization

#%%
# parameters
srate = 250
n_band = 1
n_component = 1

expNames = ['range', 'compare',
            'benchmark']

expNames = ['fullcontrast', 'halfcontrast']

expNames = ['compare']
 
tags = ['WN', 'SSVEP']
saveFILE = 'NaB_info.csv'
# chnNames = ['PZ', 'PO5', 'POZ', 'PO3', 'PO4', 'PO6', 'O1', 'OZ', 'O2']

chnNames = ['PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3',
            'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2']

#%% load datasets
DATASETs = []

for exp in expNames:
        
    dir = 'data/datasets/%s.pickle' % exp
    with open(dir, "rb") as fp:
        set = pickle.load(fp)
    DATASETs.append(set)

#%%

for (expName,dataset) in zip(expNames,DATASETs):
        
    #%% refresh
    add = 'results'+os.sep+expName
    if os.path.exists(add):
        for fnames in os.listdir(add):
            f = add+os.sep+fnames+os.sep+saveFILE
            if os.path.exists(f):
                os.remove(f)
    else:
        os.mkdir(add)
    # %% per dataset
    for i, sub in tqdm(enumerate(dataset)):

        frames = []

        for tag in tags:

            if tag in sub.keys():
                # chnNames = sub['channel']
                chnINX = [sub['channel'].index(i) for i in chnNames]
                X = sub[tag]['X'][:, chnINX]
                y = sub[tag]['y']
                subName = sub['name']

                # preprocessing
                _class = np.unique(y)
                N = X.shape[-1]
                winLEN = N/srate
                #%% build backward model

                model = vanilla(winLEN=winLEN,lag=0,n_band=1)
                aveEvoked = model.fit_transform(X,y)
                xX = model.transform(X)
                aveEvoked, xX = np.squeeze(aveEvoked), np.squeeze(xX)

                if len(_class) == 1:
                    aveEvoked = aveEvoked[np.newaxis,:]


                # %% padding 
                if winLEN<1:
                    pad = np.zeros_like(aveEvoked)
                    aveEvoked = np.concatenate([aveEvoked,pad],axis=-1)

                    pad = np.zeros_like(xX)
                    xX = np.concatenate([xX,pad],axis=-1)

                if winLEN>=1:
                    aveEvoked = aveEvoked[:,:3*int(srate)]
                    xX = xX[:,:3*int(srate)]

                #%% compute the upper bound
                # keep the first component

                aveEvoked = np.concatenate([aveEvoked[_class == i] for i in y])
                xNoise = xX - aveEvoked

                freqz, ss = returnFFT(aveEvoked,srate=srate)
                freqz, nn = returnFFT(xNoise,srate=srate)
                freqz, xx = returnFFT(xX,srate=srate)

                xPower = (1/(srate*N)) * (np.abs(xx)**2)
                sPower = (1/(srate*N)) * (np.abs(ss)**2)
                nPower = (1/(srate*N)) * (np.abs(nn)**2)
                
                ubSNR = [sPower[y==i].mean(axis=0)/nPower[y==i].mean(axis=0) for i in _class]
                logSNR = [np.log2(1+((K-1)/K*snr-1/K)) for (snr,K) in zip(ubSNR,list(Counter(y).values()))]
                logSNR = np.mean(logSNR, axis=0)
                ubINFOrate = [simpson(logSNR[:n],freqz[:n]) for n in np.arange(1,len(freqz),1)]
                ubINFOrate.insert(0,0)
                ubINFO = ubINFOrate[-1]
                ubSNR = np.mean(ubSNR,axis=0)

                #%% record

                f = pd.DataFrame({

                    'f': freqz,
                    'ubSNR': ubSNR,
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
