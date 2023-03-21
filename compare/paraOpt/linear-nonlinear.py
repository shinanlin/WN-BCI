import sys
sys.path.append('.')

from scipy import rand
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut,LeavePOut,ShuffleSplit
import compare.utils as utils
import compare.modeling as modeling
from compare.spatialFilters import *

# %%
srate = 250
expName = 'compare'
chnNames = ['PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3','PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2']
n_band = 5
targetNUM = 40
saveFILE = 'nonlinear.csv'
winLENs = [0.1, 0.2, 0.3, 0.4, 0.5]
lag = 0.14

tag = 'WN'
# %%
dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

# %%

for sub in tqdm(wholeset):
    
    frames = []
    subName = sub['name']

    add = 'results/%s/%s' % (expName, subName)
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:
        X = sub[tag]['X']
        y = sub[tag]['y']
        S = sub[tag]['STI']

        X = np.stack([X[y == i] for i in np.unique(y)])
        y = np.stack([y[y == i] for i in np.unique(y)])

        # classification
        X = np.transpose(X, axes=(1, 0, -2, -1))
        y = np.transpose(y, axes=(-1, 0))
        

        encode = modeling.Code2EEG(S=(S,y))
