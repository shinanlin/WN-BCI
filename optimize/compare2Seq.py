import sys
sys.path.append('.')
import os
from compare.modeling import Code2EEG
import random
from compare.utils import codeDistance
from tqdm import tqdm
from scipy.signal import resample
from scipy.io import loadmat
import pickle
import numpy as np
import pandas as pd

#%%
# this script is for calculate the codeset distance under 2 codesets

#%%
# parameters

srate = 250
winLEN = 1 
classNUM = 160
tmin, tmax = 0, .8
n_band = 1
n_component = 1
p = 0.9
T, lag = int(0.3*srate), int(0.14*srate)
refreshrate = 60
chnNames = ['PZ', 'PO3', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']

expName = 'sweep'

#%% load dataset and stimulus

dir = 'datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

chnINX = [wholeset[0]['channel'].index(i) for i in chnNames]
_class = np.unique(wholeset[0]['wn']['y'])

# codeset C1
C1 = loadmat('./stimulation/sweep/STI.mat')['wn']

# codeset C2
C2 = loadmat('./stimulation/optimize/STI.mat')['WN']

C1 = resample(C1, srate, axis=-1)
C2 = resample(C2, srate, axis=-1)

#%%
for i, sub in tqdm(enumerate(wholeset)):

    frames = []

    X_sub = sub['wn']['X'][:, chnINX]
    y_sub = sub['wn']['y']
    subName = sub['name']

    code2EEG = Code2EEG(srate=srate, winLEN=winLEN, tmin=tmin, tmax=tmax, S=(C1, _class), estimator=p, padding=True, n_band=n_band, component=n_component)
    code2EEG.fit(X_sub, y_sub)


    for C,csName in zip([C1,C2],['C1','C2']):

        S = code2EEG.predict(C).squeeze()
        distance = codeDistance(S[:,lag:lag+T])

        f = pd.DataFrame(index=_class, columns=_class,data=distance)
        f.reset_index(level=0, inplace=True)
        f = f.melt(id_vars='index', value_name='distance',
                           var_name='i')
        f = f.rename(columns={'index': 'j'})

        f['codeset'] = csName
        frames.append(f)

    df = pd.concat(frames,ignore_index=True,axis=0)
    df['subject'] = subName
    df.to_csv('./results/%s/%s/distance.csv' % (expName,subName))


        

        

        
