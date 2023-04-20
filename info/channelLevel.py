import sys
sys.path.append('.')
sys.path.append('./compare')
from compare.utils import returnFFT
import os
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np


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

tags = ['WN', 'SSVEP']
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
                X_inBlock = np.stack([X[y==i] for i in _class])
                aveEvoked = X_inBlock.mean(axis=1,keepdims=True)
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

                aveEvoked = np.concatenate([aveEvoked for _ in range(xX.shape[1])],axis=1)
                xNoise = xX - aveEvoked
                # get FFT
                freqz, ss = returnFFT(aveEvoked, srate=srate)
                freqz, nn = returnFFT(xNoise, srate=srate)
                freqz, xx = returnFFT(xX, srate=srate)

                # spectral power
                xPower = (1/(srate*N)) * (np.abs(xx)**2)
                sPower = (1/(srate*N)) * (np.abs(ss)**2)
                nPower = (1/(srate*N)) * (np.abs(nn)**2)

                # average across trial
                xPower = np.mean(xPower, axis=1)
                sPower = np.mean(sPower, axis=1)
                nPower = np.mean(nPower, axis=1)

                xPower = np.transpose(xPower, axes=(1, 0, -1))
                sPower = np.transpose(sPower, axes=(1, 0, -1))
                nPower = np.transpose(nPower, axes=(1, 0, -1))

                xPower =  xPower - np.mean(xPower,axis=0,keepdims=True)
                sPower =  sPower - np.mean(sPower,axis=0,keepdims=True)
                nPower =  nPower - np.mean(nPower,axis=0,keepdims=True)

                # inINX = np.argwhere((freqz<stiRange[1])&(freqz>stiRange[0])).T.tolist()
                # freqz = freqz[inINX]
                # xPower = np.squeeze(xPower[...,inINX])
                # sPower = np.squeeze(sPower[...,inINX])
                # nPower = np.squeeze(nPower[...,inINX])
                # %% record

                for name, sig in zip(['S', 'N'], [sPower, nPower]):

                    for chnName,chn in zip(chnNames,sig):

                        f = pd.DataFrame(columns=freqz, index=_class, data=chn)
                        f.reset_index(level=0, inplace=True)
                        f = f.melt(id_vars='index', value_name='psd',
                                var_name='f')
                        f = f.rename(columns={'index': 'label'})

                        f['component'] = name
                        f['tag'] = tag
                        f['channel'] = chnName
                        frames.append(f)

        df = pd.concat(frames, axis=0, ignore_index=True)
        df['subject'] = subName
        df['exp'] = expName

        # %% save
        add = 'results/%s/%s' % (expName, subName)
        if not os.path.exists(add):
            os.makedirs(add)
        df.to_csv(add+os.sep+saveFILE)

