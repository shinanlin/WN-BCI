import sys
sys.path.append('.')

import matplotlib.pyplot as plt
from scipy import rand
import seaborn as sns
from mne.decoding import ReceptiveField
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from scipy.stats import stats
import utils
from spatialFilters import *
import random

# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
chnNames = ['PZ','PO5', 'POZ', 'PO3','PO4', 'PO6', 'O1', 'OZ','O2']
seedNUM = int(1)
n_band=5
targetNUM = 40
saveFILE = 'error.csv'
winLENs = [0.1,0.2]
lag = 0.14

# %%

dir = './datasets/%s.pickle' % expName
with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

# %%  
for sub in tqdm(wholeset):
    
    frames = []

    subName = sub['name']
    chnINX = [sub['channel'].index(i) for i in chnNames]

    add = 'results/%s/%s' % (expName,subName)        
    if os.path.exists(add+os.sep+saveFILE):
        pass
    else:
        for tag in ['WN','SSVEP']:

            X = sub[tag]['X'][:, chnINX]
            y = sub[tag]['y'][:]

            _class = np.unique(y)
            X = np.stack([X[y == i] for i in _class])
            y = np.stack([y[y == i] for i in _class])

            # classification
            X = np.transpose(X, axes=(1, 0, -2, -1))
            y = np.transpose(y, axes=(-1, 0))

            loo = LeaveOneOut()
            loo.get_n_splits(X)

            for cv, (train_index, test_index) in enumerate(loo.split(X, y)):

                X_train, X_test = np.concatenate(
                    X[train_index]), np.concatenate(X[test_index])
                y_train, y_test = np.concatenate(
                    y[train_index]), np.concatenate(y[test_index])

                # predict
                for winLEN in winLENs:

                    model = TDCA(winLEN=winLEN,lag=lag,srate=srate,montage=targetNUM,n_band=n_band)
                    model.fit(X_train,y_train)

                    y_predicted = model.predict(X_test)
                    error = y_predicted != y_test
                    error = error.astype('int')
                   
                    f = pd.DataFrame(index=y_test, columns=[cv], data=error)
                    f.reset_index(level=0, inplace=True)
                    f = f.melt(id_vars='index', value_name='error',var_name='cv')
                    f = f.rename(columns={'index': 'class'})

                    f['winLEN'] = winLEN
                    f['type'] = tag
                    f['subject'] = subName
                    frames.append(f)

                df = pd.concat(frames,axis=0,ignore_index=True)
                add = 'results/%s/%s' % (expName,subName)
                if not os.path.exists(add):
                    os.makedirs(add)
                df.to_csv(add+os.sep+saveFILE)
