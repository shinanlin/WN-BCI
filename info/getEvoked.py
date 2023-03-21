
import sys
sys.path.append('.')

import numpy as np
import pickle
from scipy.signal import resample
from tqdm import tqdm
import random
from compare.modeling  import Code2EEG
from compare.spatialFilters import TDCA
import os
from compare.utils import recordModule
#%%
# this script is for generating codespace for optimization

#%%
# parameters
srate = 250
winLEN = 1 
n_band = 1
n_component = 1
p = 0.9
refreshrate = 60

expName = 'optimize'
#%% load dataset

dir = 'datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

chnNames = ['PZ', 'PO3','PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ','O2']
chnINX = [wholeset[0]['channel'].index(i) for i in chnNames]

#%%

Es = dict()

for i,sub in tqdm(enumerate(wholeset)):

    subMAT = dict()

    for tag in ['WN']:

        X_sub = sub[tag]['X'][:, chnINX]
        y_sub = sub[tag]['y']
        S = sub[tag]['STI']
        name = sub['name']

        model = TDCA(winLEN=winLEN,n_band=1,montage=40,lag=0,srate=srate)
        model.fit(X_sub, y_sub)
        evoked = model.transform(X_sub,y_sub)

        subMAT[tag] = np.transpose(evoked, (1, -1, 0))
        subMAT['name'] = name
        subMAT['stimulus'] = sub['WN']['STI']

    Es['S%s' % i] = subMAT

from scipy.io import savemat
savemat("sysIdentify/mat/%s.mat"%expName, Es)

