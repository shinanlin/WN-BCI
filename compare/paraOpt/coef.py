import sys
sys.path.append('.')

import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
import compare.utils as utils
from compare.spatialFilters import *

# %%
# this script is for computing the performance in general
# parameters
srate = 250
expName = 'compare'
n_band = 5
targetNUM = 40
saveFILE = 'coef.csv'
winLEN = .2
lag = 0.14

# montages
monatge = ['PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POZ', 'PO3','PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2']


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
        for tag in ['WN','SSVEP']:

            X = sub[tag]['X']
            y = sub[tag]['y']

            X = np.stack([X[y == i] for i in np.unique(y)])
            y = np.stack([y[y == i] for i in np.unique(y)])

            # classification
            X = np.transpose(X, axes=(1, 0, -2, -1))
            y = np.transpose(y, axes=(-1, 0))
            

            loo = LeaveOneOut()
            loo.get_n_splits(X)

            chnINX = [sub['channel'].index(i) for i in monatge]
            X_ = X[:,:,chnINX]

            for cv, (train_index, test_index) in enumerate(loo.split(X_, y)):

                X_train, X_test = np.concatenate(
                    X_[train_index]), np.concatenate(X_[test_index])
                y_train, y_test = np.concatenate(
                    y[train_index]), np.concatenate(y[test_index])

                # predict

                model = TDCA(winLEN=winLEN, lag=lag, srate=srate,
                            montage=targetNUM, n_band=n_band)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)

                f = pd.DataFrame(index=model._classes,
                                 columns=model._classes, data=model.rho)
                f.reset_index(level=0, inplace=True)
                f = f.melt(id_vars='index', value_name='rho',
                                var_name='i')
                f = f.rename(columns={'index': 'j'})

                f['subject'] = subName
                f['method'] = tag
                f['winLEN'] = winLEN
                frames.append(f)

            df = pd.concat(frames, axis=0, ignore_index=True)
            add = 'results/%s/%s' % (expName, subName)
            if not os.path.exists(add):
                os.makedirs(add)
            df.to_csv(add+os.sep+saveFILE)
