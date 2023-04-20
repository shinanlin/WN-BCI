import sys

sys.path.append('.')
sys.path.append('./compare')

import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import os
from compare.utils import returnFFT
from collections import Counter


#%%
# parameters
srate = 250
winLEN = 0.5
n_band = 1
n_component = 1
_t = np.arange(0, winLEN, 1/srate)

expNames = ['halfcontrast', 'fullcontrast',
            'range', 'compare', 'benchmark', 'NBB']

expNames = ['benchmark']

tags = ['SSVEP']
saveFILE = 'scSxN.csv'

#%% load dataset

DATASETs = []

for exp in expNames:

    dir = 'data/datasets/%s.pickle' % exp
    with open(dir, "rb") as fp:
        set = pickle.load(fp)
    DATASETs.append(set)


#%% refresh
for (expName, dataset) in zip(expNames, DATASETs):

    add = 'results'+os.sep+expName
    if os.path.exists(add):
        for fnames in os.listdir(add):
            f = add+os.sep+fnames+os.sep+saveFILE
            if os.path.exists(f):
                os.remove(f)
    else:
        os.mkdir(add)
#%%
    for i, sub in tqdm(enumerate(dataset[:])):

        frames = []
        for tag in tags:

            if tag in sub.keys():

                chnNames = sub['channel']
                X = sub[tag]['X'][:, :]
                y = sub[tag]['y']

                subName = sub['name']
                _class = np.unique(y)
                N = X.shape[-1]
                winLEN = N/srate
                #%%
                X_inBlock = np.stack([X[y == i] for i in _class])
                aveEvoked = X_inBlock.mean(axis=1, keepdims=True)
                xX = X_inBlock

                # %% padding
                if winLEN < 1:
                    pad = np.zeros_like(aveEvoked)
                    aveEvoked = np.concatenate([aveEvoked, pad], axis=-1)

                    pad = np.zeros_like(xX)
                    xX = np.concatenate([xX, pad], axis=-1)

                if winLEN >= 1:
                    aveEvoked = aveEvoked[..., :3*int(srate)]
                    xX = xX[..., :3*int(srate)]

                #%%

                aveEvoked = np.concatenate(
                    [aveEvoked for _ in range(xX.shape[1])], axis=1)
                xNoise = xX - aveEvoked
                # get FFT
                freqz, ss = returnFFT(aveEvoked, srate=srate)
                freqz, nn = returnFFT(xNoise, srate=srate)
                freqz, xx = returnFFT(xX, srate=srate)

                # spectral power
                xPower = (1/(srate*N)) * (np.abs(xx)**2)
                sPower = (1/(srate*N)) * (np.abs(ss)**2)
                nPower = (1/(srate*N)) * (np.abs(nn)**2)
                
                ubSNR = [sPower[_class == i].mean(
                    axis=1)/nPower[_class == i].mean(axis=1) for i in _class]
                
                logSNR = [np.log2(1+((K-1)/K*snr-1/K)) for (snr,K) in zip(ubSNR,list(Counter(y).values()))]
                # average across trial
                logSNR = np.concatenate(logSNR)

                logSNR = np.transpose(logSNR,axes=(1,0,-1))
             
                #%% record
                
                for chnName, chn in zip(chnNames, logSNR):

                    f = pd.DataFrame(columns=freqz, index=_class, data=chn)
                    f.reset_index(level=0, inplace=True)
                    f = f.melt(id_vars='index', value_name='psd',
                                var_name='f')
                    f = f.rename(columns={'index': 'label'})

                    f['tag'] = tag
                    f['channel'] = chnName
                    frames.append(f)

        df = pd.concat(frames, axis=0, ignore_index=True)
        df['subject'] = subName
        df['exp'] = expName

        #%% save
        add = 'results/%s/%s' % (expName, subName)
        if not os.path.exists(add):
            os.makedirs(add)
        df.to_csv(add+os.sep+saveFILE)
