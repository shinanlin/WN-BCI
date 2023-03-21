
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
# parameters
srate = 250
winLEN = 0.5
n_band = 1
n_component = 1
_t = np.arange(0,winLEN,1/srate)

expName = 'compare'
tags = ['WN', 'SSVEP']
saveFILE = 'SxN.csv'


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

        #%% 

        model = TDCA(winLEN=winLEN,lag=0)
        aveEvoked = model.fit_transform(X,y)
        xX = model.transform(X)
        
        aveEvoked, xX = np.squeeze(aveEvoked), np.squeeze(xX)
        aveEvoked = np.concatenate([aveEvoked[_class == i] for i in y])
        xNoise = xX - aveEvoked

        # resp
        f = pd.DataFrame(columns=_t,index=y,data=xX)
        f.reset_index(level=0, inplace=True)
        f = f.melt(id_vars='index', value_name='EEG',
                           var_name='time')
        f = f.rename(columns={'index': 'label'})
        f['component'] = 'resp'
        f['tag'] = tag
        frames.append(f)


        # noise
        f = pd.DataFrame(columns=_t, index=y, data=xNoise)
        f.reset_index(level=0, inplace=True)
        f = f.melt(id_vars='index', value_name='EEG',
                           var_name='time')
        f = f.rename(columns={'index': 'label'})
        f['component'] = 'noise'
        f['tag'] = tag
        frames.append(f)
        

    df = pd.concat(frames, axis=0, ignore_index=True)
    df['subject'] = subName

    #%% record

    add = 'results/%s/%s' % (expName, subName)
    if not os.path.exists(add):
        os.makedirs(add)
    df.to_csv(add+os.sep+saveFILE)
