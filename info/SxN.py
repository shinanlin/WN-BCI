
import sys
sys.path.append('.')
sys.path.append('./compare')

import numpy as np
import pickle
import pandas as pd
from compare.spatialFilters import TDCA,vanilla
from tqdm import tqdm
from compare.modeling  import Code2EEG,EEG2Code
from compare.utils import returnFFT,returnPSD
import os


#%%
# parameters
srate = 250
winLEN = 0.5
n_band = 1
n_component = 1
_t = np.arange(0,winLEN,1/srate)

expNames = ['halfcontrast', 'fullcontrast',
            'range', 'compare', 'benchmark','NBB']

expNames = ['simuMEG']

tags = ['WN', 'SSVEP']
saveFILE = 'SxN.csv'

chnNames = ['PZ', 'PO3','PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ','O2']

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
    for i,sub in tqdm(enumerate(dataset)):
        
        frames = []
        for tag in tags:
            
            if tag in sub.keys():
                
                chnNames = sub['channel']
                chnINX = [sub['channel'].index(i) for i in chnNames] 

                X = sub[tag]['X'][:, chnINX]
                y = sub[tag]['y']
                
                subName = sub['name']

                _class = np.unique(y)
                N = X.shape[-1]
                winLEN = N/srate
                #%% 

                model = vanilla(winLEN=winLEN, lag=0)
                aveEvoked = model.fit_transform(X,y)
                xX = model.transform(X)
                aveEvoked, xX = np.squeeze(aveEvoked), np.squeeze(xX)

                # %% padding 
                if winLEN<1:
                    pad = np.zeros_like(aveEvoked)
                    aveEvoked = np.concatenate([aveEvoked,pad],axis=-1)

                    pad = np.zeros_like(xX)
                    xX = np.concatenate([xX, pad], axis=-1)

                if winLEN>=1:
                    aveEvoked = aveEvoked[:,:5*int(srate)]
                    xX = xX[:,:5*int(srate)]

                #%% 
                aveEvoked = np.concatenate([aveEvoked[_class == i] for i in y])
                xNoise = xX - aveEvoked

                # get FFT
                freqz, ss = returnFFT(aveEvoked,srate=srate)
                freqz, nn = returnFFT(xNoise,srate=srate)
                freqz, xx = returnFFT(xX,srate=srate)
                # spectral power
                xPower = (1/(srate*N)) * (np.abs(xx)**2)
                sPower = (1/(srate*N)) * (np.abs(ss)**2)
                nPower = (1/(srate*N)) * (np.abs(nn)**2)
                # average across trial
                xPower = np.stack([xPower[y == k] for k in np.unique(y)])
                sPower = np.stack([sPower[y == k] for k in np.unique(y)])
                nPower = np.stack([nPower[y == k] for k in np.unique(y)])

                xPower = np.mean(xPower, axis=1)
                sPower = np.mean(sPower, axis=1)
                nPower = np.mean(nPower, axis=1)

                #%% record
                
                for name,sig in zip(['X','sig','noise'],[xPower,sPower,nPower]):
                    f = pd.DataFrame(columns=freqz,index=_class,data=sig)
                    f.reset_index(level=0, inplace=True)
                    f = f.melt(id_vars='index', value_name='psd',
                            var_name='f')
                    f = f.rename(columns={'index': 'label'})

                    f['component'] = name
                    f['tag'] = tag
                    frames.append(f)

        df = pd.concat(frames, axis=0, ignore_index=True)
        df['subject'] = subName
        df['exp'] = expName

        #%% save
        add = 'results/%s/%s' % (expName, subName)
        if not os.path.exists(add):
            os.makedirs(add)
        df.to_csv(add+os.sep+saveFILE)
